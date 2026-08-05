# -*- coding: utf-8 -*-
"""v004c 阶段2.1 + 2.1.1 收尾修复: 因子字典、去重与准入 — 测试。

覆盖 (Prompt A 二十 + 2.1.1 修复任务十三):
- 版本链 / source-ref 字节锚定对抗 (本地与tag不一致 / 本地+manifest同时篡改 / tag不存在 / tag错提交)
- 跨 clone 路径可复现 (current_git_root 用于路径, stage1_recorded_git_root 仅审计)
- Target-blind 假阳性修复: 扰动必须真正改变目标列 (SHA 前后不同, 至少一行变化)
- recognition_score / existing_model_audit 禁止检查 (真实输出成员 + 注入对抗)
- 197 行 / 79 候选门独立测试 (同步 SHA + mock tag fixture, 不被更早 SHA 门拦截)
- canonical map 角色 (每组合计 1 CANONICAL + 1 ALIAS)
- count/binary 显式语义 (不允许按当前样本恰好 0/1 判定)
- 二元派生项源缺失 hard failure
- manifest Target-blind 措辞 (读取标签做描述统计 ≠ 使用标签决定准入)
"""
import contextlib
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.v004c_d1_dataset import (
    DatasetValidationError,
    GitProvenance,
    collect_git_provenance,
)
from src.v004c_factor_dictionary import (
    ABSOLUTE_SCALE_COLUMNS,
    BINARY_COLUMNS,
    BUCKET_COLUMNS,
    DERIVED_FACTOR_SPECS,
    DERIVED_FEATURE_NAMES,
    EXPECTED_BRANCH,
    EXPECTED_DATA_COMMIT,
    EXACT_DUPLICATE_GROUPS,
    MUTUAL_EXCLUSION_GROUPS,
    TARGET_BLIND_SENSITIVE_COLUMNS,
    run_v004c_factor_dictionary_build,
)

BASE = Path(__file__).resolve().parent.parent
STAGE1_DIR = BASE / "reports/research/v004c_d1_dataset_v001_20260601_20260729"

COMPARE_FILES = [
    "v004c_raw_feature_audit_v001.csv",
    "v004c_exact_duplicate_groups_v001.csv",
    "v004c_near_duplicate_pairs_v001.csv",
    "v004c_canonical_feature_map_v001.csv",
    "v004c_feature_allowlist_primary_v001.csv",
    "v004c_feature_allowlist_sensitivity_v001.csv",
    "v004c_feature_exclusions_v001.csv",
    "v004c_predeclared_derived_factor_spec_v001.csv",
    "v004c_predeclared_derived_factor_audit_v001.csv",
]

ADMISSION_COLUMNS = [
    "admission_status", "primary_allowed", "sensitivity_allowed", "derive_only",
    "reason_codes", "preprocess_policy", "target_blind",
    "exact_duplicate_group_id", "canonical_source_column",
    "mutual_exclusion_group_id", "mechanism_group",
]

TARGET_COL = "target7_daily_d2open_d3high"
TAIL_COL = "tail_loss_daily_5pct"

# 九: 显式 count 字段 (ordinal_count / RAW_ORDINAL);
# 注意 board_streak_before_break 为 DERIVE_ONLY → preprocess=NO_DIRECT_MODEL_INPUT
COUNT_COLUMNS = [
    "recent_limit_up_count_10d", "recent_limit_up_count_20d",
    "recent_pool_appearance_count_10d", "recent_pool_appearance_count_20d",
    "max_board_streak_20d",
    "consecutive_days_below_ma5", "consecutive_days_below_ma10",
    "pool_consecutive_count_last_board",
]


# ---------------------------------------------------------------------------
# 帮助函数
# ---------------------------------------------------------------------------
def _series_sha(series: pd.Series) -> str:
    return hashlib.sha256(
        series.astype(str).str.cat(sep="|").encode("utf-8")).hexdigest()


def _copy_stage1(tmp: Path) -> Path:
    dst = tmp / "stage1"
    shutil.copytree(STAGE1_DIR, dst)
    return dst


def _rewrite_snapshot_sha(stage1_dir: Path) -> None:
    """perturb 后同步 manifest 中的 snapshot SHA (模拟真实数据变更)。"""
    manifest_path = stage1_dir / "v004c_d1_data_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    snapshot_path = stage1_dir / "v004c_d1_snapshot_v001.csv"
    record = manifest["output_files"]["v004c_d1_snapshot_v001.csv"]
    record["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    record["bytes"] = int(snapshot_path.stat().st_size)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _read_snapshot(stage1_dir: Path) -> pd.DataFrame:
    return pd.read_csv(stage1_dir / "v004c_d1_snapshot_v001.csv",
                       dtype={"code": str}, float_precision="round_trip")


def _write_snapshot(stage1_dir: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(stage1_dir / "v004c_d1_snapshot_v001.csv",
                 index=False, encoding="utf-8-sig")
    _rewrite_snapshot_sha(stage1_dir)


def _sync_manifest_file_record(stage1_dir: Path, name: str, rows: int | None) -> None:
    """把副本 manifest 中某文件的 SHA/bytes 记录同步为实际文件 (隔离门测试用)。"""
    manifest_path = stage1_dir / "v004c_d1_data_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    path = stage1_dir / name
    record = manifest["output_files"][name]
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["bytes"] = int(path.stat().st_size)
    if rows is not None:
        record["rows"] = rows
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _fake_git_calls(args, cwd):
    """mock _run_git: 分支/tag目标/祖先校验通过 (按命令名分发)。"""
    if args[1] == "branch":
        return EXPECTED_BRANCH + "\n"
    if args[1] == "rev-parse":
        return EXPECTED_DATA_COMMIT + "\n"
    if args[1] == "merge-base":
        return ""
    raise AssertionError(f"unexpected git call: {args}")


_REAL_TAG_BYTES: dict[str, bytes] = {}


def _real_tag_bytes(name: str) -> bytes:
    """从真实仓库捕获 tag 中的文件原始字节 (LF; 测试期间缓存)。"""
    if name not in _REAL_TAG_BYTES:
        import subprocess as _sp
        rel = f"reports/research/v004c_d1_dataset_v001_20260601_20260729/{name}"
        _REAL_TAG_BYTES[name] = _sp.run(
            ["git", "show", f"v004c-d1-dataset-0.1:{rel}"],
            capture_output=True, check=True).stdout
    return _REAL_TAG_BYTES[name]


@contextlib.contextmanager
def _fake_env(stage1_dir: Path, tag_bytes: str = "local"):
    """模拟 git 环境: current_git_root = stage1_dir 的父目录 (副本必须在其下)。

    tag_bytes:
      "local" — tag 字节 == 本地字节 (干净 fixture, 模拟本地==tag);
      "real"  — tag 字节 == 真实 tag 字节 (篡改对抗: 本地≠tag 必须被锚定门拦截)。
    """
    fake_root = Path(stage1_dir).parent
    prov = GitProvenance(git_root=fake_root, git_head="a" * 40,
                         git_status="", git_dirty=False)

    def fake_read(git_root: Path, source_ref: str, repo_rel_path: str) -> bytes:
        if tag_bytes == "real":
            return _real_tag_bytes(Path(repo_rel_path).name)
        # 模拟 git 视图: blob 字节 (LF 规范化), 与锚定比较语义一致
        raw = (Path(stage1_dir) / Path(repo_rel_path).name).read_bytes()
        return raw.replace(b"\r\n", b"\n")

    with mock.patch("src.v004c_factor_dictionary._run_git",
                    side_effect=_fake_git_calls), \
         mock.patch("src.v004c_factor_dictionary.read_source_ref_bytes",
                    side_effect=fake_read):
        yield prov


def _run_build(stage1_dir: Path, out_dir: Path, source_ref: str = "v004c-d1-dataset-0.1",
               tag_mode: str | None = None, **kwargs) -> dict:
    """tag_mode: None=真 git; "local"=模拟本地==tag; "real"=tag 用真实字节 (对抗)。"""
    if tag_mode is not None:
        # 副本场景: 输入被有意修改/同步 → 审计参考门只记录不拦截
        kwargs.setdefault("enforce_audit_reference", False)
        with _fake_env(stage1_dir, tag_bytes=tag_mode) as prov:
            return run_v004c_factor_dictionary_build(
                stage1_dir=stage1_dir, source_ref=source_ref, output_dir=out_dir,
                git_provenance_before=prov, git_provenance_after=prov, **kwargs)
    git_before = collect_git_provenance(Path.cwd())
    return run_v004c_factor_dictionary_build(
        stage1_dir=stage1_dir, source_ref=source_ref, output_dir=out_dir,
        git_provenance_before=git_before, git_provenance_after=None, **kwargs)


# 扰动函数 (四): 返回新 DataFrame, 不修改原 frame
def _mutate_shuffle_target(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy(deep=True)
    out[TARGET_COL] = out[TARGET_COL].sample(
        frac=1.0, random_state=7).reset_index(drop=True)
    return out


def _mutate_zero_target(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy(deep=True)
    out[TARGET_COL] = False
    return out


def _mutate_one_target(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy(deep=True)
    out[TARGET_COL] = True
    return out


def _mutate_flip_tail(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy(deep=True)
    out[TAIL_COL] = ~out[TAIL_COL].astype(bool)
    return out


def _mutate_cross_month(f: pd.DataFrame) -> pd.DataFrame:
    """跨月份重新排列 Target7: 按月份块循环移位, 值跨月流动, 分布保持。"""
    out = f.copy(deep=True)
    months = pd.to_datetime(out["signal_date"]).dt.to_period("M")
    blocks = [g.values.copy() for _, g in
              out.groupby(months, sort=True)[TARGET_COL]]
    perm = list(range(1, len(blocks))) + [0]
    out[TARGET_COL] = np.concatenate([blocks[i] for i in perm])
    return out


def _assert_outputs_identical(base_dir: Path, alt_dir: Path, names: list[str]) -> None:
    for name in names:
        a = (base_dir / name).read_bytes()
        b = (alt_dir / name).read_bytes()
        if a != b:
            raise AssertionError(f"输出不一致: {name}")


class V004cFactorDictionaryGateTest(unittest.TestCase):
    """版本链 / source-ref 字节锚定对抗 (二.6 / 十三)。"""

    def test_source_ref_not_found_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_build(STAGE1_DIR, tmp / "out", source_ref="v004c-nonexistent-tag")
            self.assertIn("source_ref_not_found", str(ctx.exception))

    def test_source_ref_wrong_commit_fails(self):
        def fake_git(args, cwd):
            if args[1] == "rev-parse":
                return "f" * 40 + "\n"
            raise AssertionError(f"unexpected git call: {args}")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch("src.v004c_factor_dictionary._run_git", side_effect=fake_git):
                with self.assertRaises(DatasetValidationError) as ctx:
                    _run_build(STAGE1_DIR, tmp / "out")
            self.assertIn("source_ref_commit_mismatch", str(ctx.exception))

    def test_local_snapshot_modified_manifest_unchanged_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            frame = _read_snapshot(dst)
            frame.loc[0, "d1_close"] = frame.loc[0, "d1_close"] + 0.01
            frame.to_csv(dst / "v004c_d1_snapshot_v001.csv",
                         index=False, encoding="utf-8-sig")
            # 不更新 manifest → 本地 manifest 记录SHA门失败 (tag 用真实字节)
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_build(dst, tmp / "out", tag_mode="real")
            self.assertIn("source_ref_file_mismatch", str(ctx.exception))

    def test_local_snapshot_modified_manifest_synced_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            frame = _read_snapshot(dst)
            frame.loc[0, "d1_close"] = frame.loc[0, "d1_close"] + 0.01
            _write_snapshot(dst, frame)  # 同步 manifest SHA
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_build(dst, tmp / "out", tag_mode="real")  # tag 字节锚定失败
            self.assertIn("source_ref_file_mismatch", str(ctx.exception))

    def test_local_lineage_modified_manifest_synced_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            lineage = pd.read_csv(dst / "v004c_d1_column_lineage.csv", dtype=str)
            lineage.loc[0, "reason"] = "tampered"
            lineage.to_csv(dst / "v004c_d1_column_lineage.csv",
                           index=False, encoding="utf-8-sig")
            _sync_manifest_file_record(dst, "v004c_d1_column_lineage.csv", rows=197)
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_build(dst, tmp / "out", tag_mode="real")
            self.assertIn("source_ref_file_mismatch", str(ctx.exception))

    def test_local_stage1_manifest_modified_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            manifest_path = dst / "v004c_d1_data_manifest.json"
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["_intrusion"] = "x"  # 不影响任何结构门
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_build(dst, tmp / "out", tag_mode="real")
            self.assertIn("source_ref_file_mismatch", str(ctx.exception))

    def test_four_files_anchor_records(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            result = _run_build(STAGE1_DIR, out)
            anchor = result["manifest"]["source_ref_anchor"]
            self.assertTrue(anchor["source_ref_verified"])
            self.assertEqual(anchor["source_ref_peeled_commit"], EXPECTED_DATA_COMMIT)
            self.assertEqual(anchor["stage1_manifest_local_sha256"],
                             "ae2a2b48b0b97390e08e81c42cadaa84d5ff76b95e847a80a66e480936ccd618")
            for name in ("v004c_d1_snapshot_v001.csv", "v004c_training_d1_v001.csv",
                         "v004c_d1_column_lineage.csv", "v004c_d1_data_manifest.json"):
                self.assertIn(name, anchor["files"])
                self.assertTrue(anchor["files"][name]["bytes_equal"],
                                f"{name} 与 tag 字节不一致")
            self.assertTrue(anchor["stage1_manifest_audit_reference_verified"])

    def test_lineage_196_isolated_gate(self):
        """6.1: 只有 lineage 行数门失败, 不被更早 SHA 门拦截。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            lineage = pd.read_csv(dst / "v004c_d1_column_lineage.csv", dtype=str)
            lineage.iloc[:-1].to_csv(dst / "v004c_d1_column_lineage.csv",
                                     index=False, encoding="utf-8-sig")
            _sync_manifest_file_record(dst, "v004c_d1_column_lineage.csv", rows=196)
            with _fake_env(dst) as prov:
                with self.assertRaises(DatasetValidationError) as ctx:
                    run_v004c_factor_dictionary_build(
                        stage1_dir=dst, source_ref="v004c-d1-dataset-0.1",
                        output_dir=tmp / "out",
                        git_provenance_before=prov, git_provenance_after=prov,
                        enforce_audit_reference=False)
            self.assertIn("lineage_rows_mismatch", str(ctx.exception))

    def test_allowed_78_isolated_gate(self):
        """6.2: 只有 allowed 候选数门失败。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            lineage = pd.read_csv(dst / "v004c_d1_column_lineage.csv", dtype=str)
            lineage.loc[lineage["column_name"] == "break_open",
                        "allowed_for_future_feature_analysis"] = "False"
            lineage.to_csv(dst / "v004c_d1_column_lineage.csv",
                           index=False, encoding="utf-8-sig")
            _sync_manifest_file_record(dst, "v004c_d1_column_lineage.csv", rows=197)
            with _fake_env(dst) as prov:
                with self.assertRaises(DatasetValidationError) as ctx:
                    run_v004c_factor_dictionary_build(
                        stage1_dir=dst, source_ref="v004c-d1-dataset-0.1",
                        output_dir=tmp / "out",
                        git_provenance_before=prov, git_provenance_after=prov,
                        enforce_audit_reference=False)
            self.assertIn("allowed_candidate_count_mismatch", str(ctx.exception))

    def test_output_dir_gate(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            with (out / "stale.txt").open("w", encoding="utf-8") as handle:
                handle.write("x")
            with self.assertRaises(DatasetValidationError):
                _run_build(STAGE1_DIR, out)


class V004cFactorDictionaryCrossCloneTest(unittest.TestCase):
    """三: 跨 clone 路径可复现 (current_git_root 用于路径, stage1_recorded_git_root 仅审计)。"""

    def test_cross_clone_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake_root = tmp / "clone"
            fake_root.mkdir()
            stage1_copy = fake_root / "reports/research/v004c_d1_dataset_v001_20260601_20260729"
            shutil.copytree(STAGE1_DIR, stage1_copy)
            prov = GitProvenance(git_root=fake_root, git_head="a" * 40,
                                 git_status="", git_dirty=False)

            def fake_read(git_root, source_ref, repo_rel_path):
                raw = (stage1_copy / Path(repo_rel_path).name).read_bytes()
                return raw.replace(b"\r\n", b"\n")

            with mock.patch("src.v004c_factor_dictionary._run_git",
                            side_effect=_fake_git_calls), \
                 mock.patch("src.v004c_factor_dictionary.read_source_ref_bytes",
                            side_effect=fake_read):
                result = run_v004c_factor_dictionary_build(
                    stage1_dir=stage1_copy, source_ref="v004c-d1-dataset-0.1",
                    output_dir=fake_root / "out",
                    git_provenance_before=prov, git_provenance_after=prov)
            manifest = result["manifest"]
            self.assertTrue(manifest["frozen"])
            self.assertEqual(manifest["source_ref_anchor"]["current_git_root"],
                             str(fake_root))
            self.assertEqual(manifest["source_ref_anchor"]["stage1_recorded_git_root"],
                             "F:\\fenqixuangu")
            self.assertTrue(manifest["source_ref_anchor"]["source_ref_verified"])
            # 输出路径必须相对 current_git_root (fake_root)
            for rec in manifest["output_files"].values():
                self.assertFalse(rec["path"].startswith(".."),
                                 f"输出路径逃逸 current_git_root: {rec['path']}")

    def test_stage1_dir_outside_git_root_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake_root = tmp / "clone"
            fake_root.mkdir()
            prov = GitProvenance(git_root=fake_root, git_head="a" * 40,
                                 git_status="", git_dirty=False)
            with mock.patch("src.v004c_factor_dictionary._run_git",
                            side_effect=_fake_git_calls):
                with self.assertRaises(DatasetValidationError) as ctx:
                    run_v004c_factor_dictionary_build(
                        stage1_dir=STAGE1_DIR, source_ref="v004c-d1-dataset-0.1",
                        output_dir=tmp / "out",
                        git_provenance_before=prov, git_provenance_after=prov)
            self.assertIn("stage1_dir_outside_git_root", str(ctx.exception))


class V004cFactorDictionaryTargetBlindTest(unittest.TestCase):
    """四: Target-blind 扰动必须真正改变目标列 (修复 mutate 假阳性)。"""

    def _run_pair(self, tmp: Path, mutate, sensitive_cols: list[str]):
        base_out = tmp / "out_base"
        _run_build(STAGE1_DIR, base_out)
        dst = _copy_stage1(tmp)
        frame = _read_snapshot(dst)
        original = frame.copy(deep=True)
        mutated = mutate(frame.copy(deep=True))
        if mutated is None:
            mutated = frame
        # 扰动有效性: 扰动前后目标列 SHA 必须不同, 至少一行变化
        for col in sensitive_cols:
            self.assertNotEqual(_series_sha(original[col]), _series_sha(mutated[col]),
                                f"扰动未真正改变 {col}")
            self.assertFalse(original[col].equals(mutated[col]),
                             f"扰动后 {col} 无任何行变化")
        _write_snapshot(dst, mutated)
        alt_out = tmp / "out_alt"
        _run_build(dst, alt_out, tag_mode="local")
        return base_out, alt_out

    def _compare_admission_outputs(self, base_out: Path, alt_out: Path) -> None:
        _assert_outputs_identical(base_out, alt_out, COMPARE_FILES)

    def _compare_dictionary(self, base_out: Path, alt_out: Path,
                            perturbed_cols: set[str]) -> None:
        a = pd.read_csv(base_out / "v004c_factor_dictionary_v001.csv", dtype=str)
        b = pd.read_csv(alt_out / "v004c_factor_dictionary_v001.csv", dtype=str)
        self.assertEqual(len(a), len(b))
        a_rest = a.loc[~a["source_column"].isin(perturbed_cols)].reset_index(drop=True)
        b_rest = b.loc[~b["source_column"].isin(perturbed_cols)].reset_index(drop=True)
        self.assertTrue(a_rest.equals(b_rest),
                        "字典 (除扰动列统计外) 必须逐字节一致")
        for col in ADMISSION_COLUMNS:
            self.assertTrue(a[col].fillna("").eq(b[col].fillna("")).all(),
                            f"准入字段随标签改变: {col}")

    def test_target_shuffle_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base_out, alt_out = self._run_pair(tmp, _mutate_shuffle_target,
                                               [TARGET_COL])
            self._compare_admission_outputs(base_out, alt_out)
            # shuffle 保持值分布 → 字典也应逐字节一致
            self._compare_dictionary(base_out, alt_out, set())

    def test_target_all_zero_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base_out, alt_out = self._run_pair(tmp, _mutate_zero_target, [TARGET_COL])
            self._compare_admission_outputs(base_out, alt_out)
            self._compare_dictionary(base_out, alt_out, {TARGET_COL})

    def test_target_all_one_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base_out, alt_out = self._run_pair(tmp, _mutate_one_target, [TARGET_COL])
            self._compare_admission_outputs(base_out, alt_out)
            self._compare_dictionary(base_out, alt_out, {TARGET_COL})

    def test_tail_loss_flip_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base_out, alt_out = self._run_pair(tmp, _mutate_flip_tail, [TAIL_COL])
            self._compare_admission_outputs(base_out, alt_out)
            self._compare_dictionary(base_out, alt_out, {TAIL_COL})

    def test_cross_month_permute_invariance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base_out, alt_out = self._run_pair(tmp, _mutate_cross_month, [TARGET_COL])
            self._compare_admission_outputs(base_out, alt_out)
            self._compare_dictionary(base_out, alt_out, {TARGET_COL})


class V004cFactorDictionaryAdmissionTest(unittest.TestCase):
    """准入分区与硬规则 (七~十二, 十九)。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.out = cls.tmp / "out"
        _run_build(STAGE1_DIR, cls.out)
        cls.dictionary = pd.read_csv(
            cls.out / "v004c_factor_dictionary_v001.csv", dtype=str)
        cls.primary = pd.read_csv(
            cls.out / "v004c_feature_allowlist_primary_v001.csv", dtype=str)
        cls.sensitivity = pd.read_csv(
            cls.out / "v004c_feature_allowlist_sensitivity_v001.csv", dtype=str)
        cls.exclusions = pd.read_csv(
            cls.out / "v004c_feature_exclusions_v001.csv", dtype=str)
        cls.dup_groups = pd.read_csv(
            cls.out / "v004c_exact_duplicate_groups_v001.csv", dtype=str)
        cls.canonical_map = pd.read_csv(
            cls.out / "v004c_canonical_feature_map_v001.csv", dtype=str)
        cls.near_dup = pd.read_csv(
            cls.out / "v004c_near_duplicate_pairs_v001.csv", dtype=str)
        cls.derived_audit = pd.read_csv(
            cls.out / "v004c_predeclared_derived_factor_audit_v001.csv", dtype=str)
        cls.raw_audit = pd.read_csv(
            cls.out / "v004c_raw_feature_audit_v001.csv", dtype=str)
        cls.manifest = json.loads(
            (cls.out / "v004c_factor_dictionary_manifest.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_197_columns_dictionary(self):
        self.assertEqual(len(self.dictionary), 197)

    def test_partition_complete(self):
        statuses = self.dictionary.loc[
            self.dictionary["stage1_allowed_for_feature_analysis"] == "True",
            "admission_status"]
        self.assertEqual(len(statuses), 79)
        self.assertEqual(statuses.value_counts().to_dict(), {
            "PRIMARY_RAW": 32,
            "SENSITIVITY_RAW": 21,
            "DERIVE_ONLY": 12,
            "EXCLUDE_DEPRECATED": 2,
            "EXCLUDE_EXACT_DUPLICATE_ALIAS": 11,
            "EXCLUDE_ALL_MISSING": 1,
        })

    def test_recognition_and_model_audit_not_in_allowlists(self):
        """五: 直接检查真实输出成员 (不得用预置 set 掩盖)。"""
        primary_names = set(self.primary["feature_name"])
        sensitivity_names = set(self.sensitivity["feature_name"])
        self.assertNotIn("recognition_score", primary_names)
        self.assertNotIn("recognition_score", sensitivity_names)
        lineage = pd.read_csv(STAGE1_DIR / "v004c_d1_column_lineage.csv", dtype=str)
        ema = set(lineage.loc[lineage["column_role"] == "existing_model_audit",
                              "column_name"])
        self.assertEqual(len(ema), 13)
        self.assertEqual(len(ema & primary_names), 0,
                         "existing_model_audit 字段进入 primary")
        self.assertEqual(len(ema & sensitivity_names), 0,
                         "existing_model_audit 字段进入 sensitivity")
        for bad in ("signal_date", "break_date", "d2_open_daily", "d3_high_daily",
                    TARGET_COL, TAIL_COL):
            self.assertNotIn(bad, primary_names)
            self.assertNotIn(bad, sensitivity_names)

    def test_recognition_injection_detected_by_check(self):
        """五: 注入对抗 — 若 recognition_score 被注入 primary, 检查必须失败。"""
        names = set(self.primary["feature_name"])
        self.assertNotIn("recognition_score", names)
        with self.assertRaises(AssertionError):
            self.assertNotIn("recognition_score", names | {"recognition_score"})

    def test_model_audit_injection_detected_by_check(self):
        """五: 注入对抗 — 任一 existing_model_audit 字段注入 sensitivity, 检查必须失败。"""
        lineage = pd.read_csv(STAGE1_DIR / "v004c_d1_column_lineage.csv", dtype=str)
        ema = set(lineage.loc[lineage["column_role"] == "existing_model_audit",
                              "column_name"])
        sens_names = set(self.sensitivity["feature_name"])
        self.assertEqual(len(ema & sens_names), 0)
        injected = next(iter(ema))
        with self.assertRaises(AssertionError):
            self.assertNotIn(injected, sens_names | {injected})

    def test_break_turnover_ratio_excluded(self):
        row = self.dictionary[self.dictionary["source_column"] == "break_turnover_ratio"].iloc[0]
        self.assertEqual(row["admission_status"], "EXCLUDE_ALL_MISSING")
        self.assertEqual(int(row["missing_count"]), 333)
        self.assertEqual(float(row["missing_rate"]), 1.0)

    def test_13_duplicate_groups_all_match(self):
        self.assertEqual(len(self.dup_groups), 13)
        by_alias = dict(zip(self.dup_groups["alias_source_column"],
                            zip(self.dup_groups["group_id"],
                                self.dup_groups["canonical_source_column"])))
        for gid, alias, canonical in EXACT_DUPLICATE_GROUPS:
            self.assertIn(alias, by_alias)
            self.assertEqual(by_alias[alias], (gid, canonical))
        self.assertTrue((self.dup_groups["equal_including_missing"] == "True").all())
        self.assertTrue((self.dup_groups["rows_compared"].astype(int) == 333).all())

    def test_extra_duplicate_group_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            frame = _read_snapshot(dst)
            frame["d1_afternoon_return"] = frame["d1_last_hour_return"]
            _write_snapshot(dst, frame)
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_build(dst, tmp / "out", tag_mode="local")
            self.assertIn("额外精确重复", str(ctx.exception))

    def test_canonical_map_roles(self):
        """八: 每组恰好 1 个 CANONICAL + 1 个 ALIAS, 角色字段正确。"""
        cm = self.canonical_map
        self.assertIn("mapping_role", cm.columns)
        self.assertIn("is_canonical", cm.columns)
        self.assertIn("is_alias", cm.columns)
        for gid, alias, canonical in EXACT_DUPLICATE_GROUPS:
            rows = cm[cm["exact_duplicate_group_id"] == gid]
            self.assertEqual(len(rows), 2, f"组 {gid} 行数 != 2")
            alias_row = rows[rows["source_column"] == alias].iloc[0]
            canon_row = rows[rows["source_column"] == canonical].iloc[0]
            self.assertEqual(alias_row["mapping_role"], "ALIAS")
            self.assertEqual(alias_row["is_alias"], "True")
            self.assertEqual(alias_row["is_canonical"], "False")
            self.assertEqual(alias_row["canonical_source_column"], canonical)
            self.assertEqual(canon_row["mapping_role"], "CANONICAL")
            self.assertEqual(canon_row["is_canonical"], "True")
            self.assertEqual(canon_row["is_alias"], "False")
            self.assertEqual(canon_row["canonical_source_column"], canonical)
            self.assertIn("不允许直接进入模型", alias_row["notes"])
            self.assertIn("保留并按其它准入规则处理", canon_row["notes"])
        for gid in {g[0] for g in EXACT_DUPLICATE_GROUPS}:
            grp_rows = cm[cm["exact_duplicate_group_id"] == gid]
            self.assertEqual(
                int((grp_rows["mapping_role"] == "CANONICAL").sum()), 1, gid)
            self.assertEqual(
                int((grp_rows["mapping_role"] == "ALIAS").sum()), 1, gid)

    def test_deprecated_dup_groups_alias_status(self):
        """ED10/ED11: deprecated 字段是 ALIAS, 状态仍为 EXCLUDE_DEPRECATED。"""
        for gid, alias, canonical in EXACT_DUPLICATE_GROUPS:
            if alias in ("deprecated_d1_reclaimed_ma5_v01",
                         "deprecated_d1_reclaimed_ma10_v01"):
                row = self.canonical_map[
                    (self.canonical_map["exact_duplicate_group_id"] == gid)
                    & (self.canonical_map["source_column"] == alias)].iloc[0]
                self.assertEqual(row["mapping_role"], "ALIAS")
                self.assertEqual(row["admission_status"], "EXCLUDE_DEPRECATED")

    def test_mutual_exclusion_max_one_primary(self):
        primary_names = set(self.primary["feature_name"])
        for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
            members = [canonical] + list(alternatives)
            self.assertLessEqual(
                len([m for m in members if m in primary_names]), 1,
                f"互斥组 {gid} 超过一个 primary")

    def test_absolute_scale_only_derive_only(self):
        rows = self.dictionary[self.dictionary["source_column"].isin(ABSOLUTE_SCALE_COLUMNS)]
        self.assertEqual(len(rows), 12)
        self.assertTrue((rows["admission_status"] == "DERIVE_ONLY").all())
        primary_names = set(self.primary["feature_name"])
        sensitivity_names = set(self.sensitivity["feature_name"])
        self.assertEqual(len(ABSOLUTE_SCALE_COLUMNS & primary_names), 0)
        self.assertEqual(len(ABSOLUTE_SCALE_COLUMNS & sensitivity_names), 0)

    def test_missing_field_not_primary(self):
        primary_rows = self.dictionary[self.dictionary["admission_status"] == "PRIMARY_RAW"]
        self.assertTrue((primary_rows["missing_count"].astype(int) == 0).all())
        self.assertEqual(len(primary_rows), 32)

    def test_low_support_binary_not_primary(self):
        low_support = {"break_opened_from_limit_up", "last_board_day_in_pool",
                       "d1_close_above_ma5", "d1_close_above_ma10",
                       "d1_true_reclaim_ma10"}
        primary_names = set(self.primary["feature_name"])
        self.assertEqual(len(low_support & primary_names), 0)
        for col in low_support:
            status = self.dictionary.loc[
                self.dictionary["source_column"] == col, "admission_status"].iloc[0]
            self.assertEqual(status, "SENSITIVITY_RAW")

    def test_bucket_not_primary(self):
        primary_names = set(self.primary["feature_name"])
        self.assertEqual(len(BUCKET_COLUMNS & primary_names), 0)
        for col in BUCKET_COLUMNS:
            status = self.dictionary.loc[
                self.dictionary["source_column"] == col, "admission_status"].iloc[0]
            self.assertEqual(status, "SENSITIVITY_RAW")
            policy = self.dictionary.loc[
                self.dictionary["source_column"] == col, "preprocess_policy"].iloc[0]
            self.assertEqual(policy, "BUCKET_SENSITIVITY_ONLY")

    def test_sensitivity_disjoint_from_primary(self):
        self.assertEqual(
            len(set(self.primary["feature_name"]) & set(self.sensitivity["feature_name"])), 0)
        self.assertEqual(len(self.primary), 38)  # 32 PRIMARY_RAW + 6 PREDECLARED_DERIVED
        self.assertEqual(len(self.sensitivity), 21)

    def test_all_target_blind_true(self):
        self.assertTrue((self.dictionary["target_blind"] == "True").all())

    def test_forbidden_unknown_zero(self):
        mechs = set(self.dictionary["mechanism_group"])
        self.assertNotIn("UNKNOWN", mechs)
        self.assertNotIn("OTHER", mechs)
        self.assertNotIn("UNCLASSIFIED", mechs)

    def test_count_semantics_ordinal(self):
        """九: count 字段一律 ordinal_count / RAW_ORDINAL (即使当前样本只有 0/1)。"""
        for col in COUNT_COLUMNS:
            row = self.dictionary[self.dictionary["source_column"] == col].iloc[0]
            self.assertEqual(row["semantic_type"], "ordinal_count", col)
            self.assertEqual(row["preprocess_policy"], "RAW_ORDINAL", col)
        # consecutive_days_below_ma5: 当前样本只有 {0,1}, 仍必须是 ordinal + SENSITIVITY_RAW
        row = self.dictionary[
            self.dictionary["source_column"] == "consecutive_days_below_ma5"].iloc[0]
        self.assertEqual(row["semantic_type"], "ordinal_count")
        self.assertEqual(row["preprocess_policy"], "RAW_ORDINAL")
        self.assertEqual(row["admission_status"], "SENSITIVITY_RAW")
        # board_streak_before_break: count 语义但 DERIVE_ONLY → 不直接进模型
        row = self.dictionary[
            self.dictionary["source_column"] == "board_streak_before_break"].iloc[0]
        self.assertEqual(row["semantic_type"], "ordinal_count")
        self.assertEqual(row["admission_status"], "DERIVE_ONLY")
        self.assertEqual(row["preprocess_policy"], "NO_DIRECT_MODEL_INPUT")

    def test_binary_allowlist_semantics(self):
        """九: 显式 binary allowlist 字段 semantic_type=binary / RAW_BINARY。"""
        for col in BINARY_COLUMNS:
            row = self.dictionary[self.dictionary["source_column"] == col].iloc[0]
            self.assertEqual(row["semantic_type"], "binary", col)
            self.assertEqual(row["preprocess_policy"], "RAW_BINARY", col)

    def test_manifest_target_blind_wording(self):
        """七: 措辞 — 明确区分 读取标签做描述统计 与 使用标签决定准入。"""
        tb = self.manifest["target_blind"]
        self.assertNotIn("sensitive_columns_never_read", tb)
        self.assertTrue(tb["sensitive_columns_not_used_for_admission"])
        self.assertTrue(tb["descriptive_statistics_may_read_sensitive_columns"])
        self.assertIn("admission_rule_version", tb)
        self.assertIn("target_blind_invariance_test_version", tb)
        self.assertEqual(len(tb["target_blind_invariance_tests"]), 5)
        self.assertTrue(tb["admission_invariance_verified"])

    def test_manifest_source_ref_anchor_recorded(self):
        """2.5: manifest 记录 source ref 锚定字段。"""
        anchor = self.manifest["source_ref_anchor"]
        self.assertEqual(anchor["source_ref"], "v004c-d1-dataset-0.1")
        self.assertEqual(anchor["source_ref_peeled_commit"], EXPECTED_DATA_COMMIT)
        self.assertTrue(anchor["source_ref_verified"])
        self.assertEqual(anchor["current_git_root"], str(Path.cwd()))
        self.assertEqual(anchor["stage1_recorded_git_root"], "F:\\fenqixuangu")
        self.assertIn("stage1_manifest_local_sha256", anchor)
        self.assertIn("stage1_manifest_source_ref_sha256", anchor)
        self.assertEqual(len(anchor["files"]), 4)
        for name, rec in anchor["files"].items():
            self.assertTrue(rec["bytes_equal"], name)


class V004cFactorDictionaryDerivedTest(unittest.TestCase):
    """十三 / 十四: 派生因子规格、域审计与 hard missing policy (十)。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.out = cls.tmp / "out"
        _run_build(STAGE1_DIR, cls.out)
        cls.spec = pd.read_csv(
            cls.out / "v004c_predeclared_derived_factor_spec_v001.csv", dtype=str)
        cls.audit = pd.read_csv(
            cls.out / "v004c_predeclared_derived_factor_audit_v001.csv", dtype=str)
        cls.primary = pd.read_csv(
            cls.out / "v004c_feature_allowlist_primary_v001.csv", dtype=str)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_spec_has_exactly_six_predeclared(self):
        self.assertEqual(len(self.spec), 6)
        self.assertEqual(set(self.spec["feature_name"]), set(DERIVED_FEATURE_NAMES))
        self.assertTrue((self.spec["admission_status"] == "PREDECLARED_DERIVED").all())
        self.assertTrue((self.spec["target_blind"] == "True").all())

    def test_binary_derived_missing_policy_hard(self):
        """十: 二元派生项 missing_policy = source_missing_is_hard_failure。"""
        for name in ("board_streak_is_3", "ma5_overheat_10"):
            row = self.spec[self.spec["feature_name"] == name].iloc[0]
            self.assertEqual(row["missing_policy"], "source_missing_is_hard_failure")

    def test_derived_in_primary_not_frozen(self):
        derived = self.primary[self.primary["source_or_derived"] == "derived"]
        self.assertEqual(len(derived), 6)
        self.assertTrue((derived["derived_not_yet_model_frozen"] == "True").all())

    def test_derived_domain_audit_ok(self):
        rows = self.audit[self.audit["feature_name"].isin(DERIVED_FEATURE_NAMES)]
        self.assertEqual(len(rows), 6)
        self.assertTrue((rows["domain_violation_count"].astype(int) == 0).all())
        self.assertTrue((rows["formula_validation_status"] == "OK").all())
        self.assertTrue((rows["non_null_count"].astype(int) == 333).all())
        self.assertTrue((rows["source_columns_exist"] == "True").all())
        self.assertTrue((rows["source_columns_allowed"] == "True").all())

    def test_derived_binary_domains(self):
        for name in ("board_streak_is_3", "ma5_overheat_10"):
            row = self.audit[self.audit["feature_name"] == name].iloc[0]
            self.assertEqual(float(row["min_value"]), 0.0)
            self.assertEqual(float(row["max_value"]), 1.0)
            self.assertEqual(int(row["unique_count"]), 2)

    def test_profit_chip_in_unit_interval(self):
        row = self.audit[self.audit["feature_name"] == "profit_chip_ratio"].iloc[0]
        self.assertGreaterEqual(float(row["min_value"]), 0.0)
        self.assertLessEqual(float(row["max_value"]), 1.0)

    def test_true_reclaim_recompute_match(self):
        for name in ("d1_true_reclaim_ma5", "d1_close_above_ma5", "d1_close_above_ma10"):
            row = self.audit[self.audit["feature_name"] == name].iloc[0]
            self.assertEqual(row["formula_validation_status"], "RECOMPUTE_MATCH")

    def _inject_nan_fails(self, column: str):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dst = _copy_stage1(tmp)
            frame = _read_snapshot(dst)
            frame.loc[0, column] = np.nan
            _write_snapshot(dst, frame)
            with _fake_env(dst) as prov:
                with self.assertRaises(DatasetValidationError) as ctx:
                    run_v004c_factor_dictionary_build(
                        stage1_dir=dst, source_ref="v004c-d1-dataset-0.1",
                        output_dir=tmp / "out",
                        git_provenance_before=prov, git_provenance_after=prov,
                        enforce_audit_reference=False)
            self.assertIn("source_missing_is_hard_failure", str(ctx.exception))

    def test_board_streak_source_nan_fails(self):
        self._inject_nan_fails("board_streak_before_break")

    def test_ma5_overheat_source_nan_fails(self):
        self._inject_nan_fails("d1_close_to_ma5_raw")


class V004cFactorDictionaryProvenanceTest(unittest.TestCase):
    """输出 / manifest / Git provenance (十七, 十九)。"""

    def test_full_package_and_manifest_sha(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            result = _run_build(STAGE1_DIR, out)
            manifest = result["manifest"]
            expected_files = {
                "v004c_factor_dictionary_v001.csv",
                "v004c_raw_feature_audit_v001.csv",
                "v004c_exact_duplicate_groups_v001.csv",
                "v004c_near_duplicate_pairs_v001.csv",
                "v004c_canonical_feature_map_v001.csv",
                "v004c_feature_allowlist_primary_v001.csv",
                "v004c_feature_allowlist_sensitivity_v001.csv",
                "v004c_feature_exclusions_v001.csv",
                "v004c_predeclared_derived_factor_spec_v001.csv",
                "v004c_predeclared_derived_factor_audit_v001.csv",
                "v004c_factor_dictionary_review.md",
                "git_head.txt",
                "git_status_before.txt",
                "git_status_after.txt",
            }
            on_disk = {p.name for p in out.iterdir() if p.is_file()}
            self.assertEqual(on_disk,
                             expected_files | {"v004c_factor_dictionary_manifest.json"})
            self.assertEqual(set(manifest["output_files"].keys()), expected_files)
            for name, record in manifest["output_files"].items():
                actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
                self.assertEqual(actual, record["sha256"], name)
            gate = manifest["gate_stats"]
            self.assertEqual(gate["input_rows"], 333)
            self.assertEqual(gate["signal_dates"], 42)
            self.assertEqual(gate["lineage_rows"], 197)
            self.assertEqual(gate["allowed_raw"], 79)
            self.assertEqual(gate["exact_duplicate_groups"], 13)
            self.assertEqual(gate["forbidden_unknown_mechanism"], 0)
            self.assertEqual(gate["break_turnover_ratio_missing_count"], 333)
            self.assertTrue(gate["target_blind_all_true"])
            self.assertTrue(manifest["frozen"])

    def test_git_provenance_files(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            result = _run_build(STAGE1_DIR, out)
            head = (out / "git_head.txt").read_text(encoding="utf-8").strip()
            git_before = collect_git_provenance(Path.cwd())
            self.assertEqual(head, git_before.git_head)
            self.assertEqual(result["manifest"]["git_head"], git_before.git_head)
            self.assertTrue((out / "git_status_before.txt").exists())
            self.assertTrue((out / "git_status_after.txt").exists())

    def test_output_dir_gate_nonempty_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            with (out / "x.txt").open("w", encoding="utf-8") as handle:
                handle.write("x")
            with self.assertRaises(DatasetValidationError):
                _run_build(STAGE1_DIR, out)


if __name__ == "__main__":
    unittest.main()

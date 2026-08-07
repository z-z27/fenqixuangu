# -*- coding: utf-8 -*-
"""v004c 阶段2.3: 人工候选模型规格冻结 — 测试。

覆盖 (任务二十四/二十五):
- 版本锚定 / 输入 manifest SHA / 阶段2.2 tag 目标 / 目录 git_head.txt;
- 59 因子完整性 / 12 个唯一入选因子 / 4 个模型 / 3 个实质模型 /
  M1/M2/M3 各 5 特征与固定顺序;
- M1 全 primary / M2 三个固定 sensitivity 替代表达 / M3 派生与底层机制 /
  禁止字段 / 无 bucket / 互斥组每模型最多一次 / 无精确重复 /
  无 canonical+alias 共存 / M3 无派生与直接源共存;
- 阶段2.2 证据扰动 (effect_value/AUC/CI/flags/LODO/board) →
  模型目录/特征规格/预处理规格/候选 JSON/协议 JSON 逐字节不变;
- 标签扰动 (六月/七月 Target7、tail_loss、outcome) → 所有输出不变;
- 特征扰动 → 冗余审计改变, 成员不变;
- 30 行冗余审计 / Spearman fixture / 重复值 fixture / 缺失 pair count /
  六月七月切片分离 / 冗余 flag 阈值;
- 预处理政策解析 / FOLD_CLIP_Z / RAW_BINARY / FAIL_CLOSED;
- evaluation protocol 完整 / 资格门 / 比较层级;
- 输出目录门 / 14 个输出文件 / manifest SHA 与行数 / 确定性重建 /
  无模型拟合接口 / 锁定审计全 PASS;
- ResourceWarning 由运行参数 -W error::ResourceWarning 兜底。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from typing import Any

import numpy as np
import pandas as pd

from src.v004c_candidate_model_specs import (
    ALLOWED_INPUT_FILES,
    EXPECTED_BRANCH,
    EXPECTED_STAGE1_DATA_COMMIT,
    EXPECTED_STAGE2_1_DATA_COMMIT,
    EXPECTED_STAGE2_2_DATA_COMMIT,
    EXPECTED_STAGE2_2_TAG,
    MODEL_BY_ID,
    SNAPSHOT_READ_COLUMNS,
    UNIQUE_SELECTED_FEATURES,
    _allowlist_lookup,
    _derive_needed_series,
    _dump_json,
    build_candidate_boundary,
    build_candidate_redundancy_audit,
    build_candidate_specs_json,
    build_design_disclosure,
    build_evaluation_protocol_json,
    build_holdout_lock_audit,
    build_model_catalog,
    build_model_feature_spec,
    build_preprocessing_spec,
    build_selected_feature_evidence,
    compute_vwap_reparameterization,
    load_inputs,
    run_v004c_stage2_3_candidate_specs,
    scan_source_for_forbidden_calls,
    select_final_candidate,
    validate_git_chain,
)
from src.v004c_d1_dataset import DatasetValidationError, GitProvenance

REPO_ROOT = Path.cwd()
STAGE1_DIR = REPO_ROOT / "reports/research/v004c_d1_dataset_v001_20260601_20260729"
STAGE2_1_DIR = REPO_ROOT / "reports/research/v004c_factor_dictionary_v001_20260601_20260729"
STAGE2_2_DIR = REPO_ROOT / "reports/research/v004c_stage2_2_univariate_v001_20260601_20260729"

FIXED_GENERATED_AT = "2026-08-06T00:00:00+0800"

OUTPUT_NAMES = (
    "v004c_stage2_3_model_catalog_v001.csv",
    "v004c_stage2_3_model_feature_spec_v001.csv",
    "v004c_stage2_3_preprocessing_spec_v001.csv",
    "v004c_stage2_3_selected_feature_evidence_v001.csv",
    "v004c_stage2_3_candidate_redundancy_audit_v001.csv",
    "v004c_stage2_3_candidate_boundary_v001.csv",
    "v004c_stage2_3_holdout_lock_audit_v001.csv",
    "v004c_stage2_3_candidate_specs_v001.json",
    "v004c_stage2_3_evaluation_protocol_v001.json",
    "v004c_stage2_3_design_disclosure_v001.json",
    "v004c_stage2_3_manifest.json",
    "v004c_stage2_3_review.md",
    "git_head.txt",
    "git_status_before.txt",
    "git_status_after.txt",
)

EXPECTED_M1 = ("break_open_return", "down_bar_volume_ratio",
               "high_zone_volume_ratio", "late_day_sell_volume_ratio",
               "d1_close_to_vwap_raw")
EXPECTED_M2 = ("break_open_return", "down_bar_volume_ratio",
               "high_zone_amount_ratio", "late_day_sell_amount_ratio",
               "d1_vwap_to_close_gap")
EXPECTED_M3 = ("overrepair", "ma5_overheat_10", "break_volume_abnormality",
               "profit_chip_ratio", "late_day_sell_volume_ratio")
EXPECTED_UNIQUE = tuple(dict.fromkeys(EXPECTED_M1 + EXPECTED_M2 + EXPECTED_M3))


def _fake_git_calls(args, cwd):
    """mock _run_git: 分支 / 三个 tag 目标 / 祖先 / autocrlf / 版本。"""
    if args[1] == "rev-parse":
        ref = args[-1]
        if "v004c-d1-dataset-0.1" in ref:
            return EXPECTED_STAGE1_DATA_COMMIT + "\n"
        if "v004c-factor-dictionary-0.1" in ref:
            return EXPECTED_STAGE2_1_DATA_COMMIT + "\n"
        if "v004c-stage2-2-univariate-0.1" in ref:
            return EXPECTED_STAGE2_2_DATA_COMMIT + "\n"
        raise DatasetValidationError(
            f"Git 命令失败 (rev-parse --verify): rc=128 stderr=unknown revision "
            f"{ref}")
    if args[1] == "branch":
        return EXPECTED_BRANCH + "\n"
    if args[1] == "merge-base":
        return ""
    if args[1] == "config":
        return "true\n"
    if args[1] == "--version":
        return "git version 2.49.0.windows.1\n"
    raise AssertionError(f"unexpected git call: {args}")


@contextlib.contextmanager
def _fake_env(stage1_dir: Path, stage2_1_dir: Path, stage2_2_dir: Path):
    """模拟 git: 三个输入目录须同父 (真实目录共用 reports/research;
    临时拷贝共用临时父目录), git_root = 共同父目录。"""
    common = Path(stage1_dir).parent
    assert Path(stage2_1_dir).parent == common, "输入目录必须同父"
    assert Path(stage2_2_dir).parent == common, "输入目录必须同父"
    prov = GitProvenance(git_root=common, git_head="a" * 40,
                         git_status="", git_dirty=False)
    with mock.patch("src.v004c_candidate_model_specs._run_git",
                    side_effect=_fake_git_calls):
        yield prov


def _run_specs(stage1_dir: Path, stage2_1_dir: Path, stage2_2_dir: Path,
               out_dir: Path, *, generated_at: str = FIXED_GENERATED_AT,
               snapshot_override: pd.DataFrame | None = None,
               enforce_input_sha: bool = True, **kwargs) -> dict:
    with _fake_env(stage1_dir, stage2_1_dir, stage2_2_dir) as prov:
        return run_v004c_stage2_3_candidate_specs(
            stage1_dir=stage1_dir, stage2_1_dir=stage2_1_dir,
            stage2_2_dir=stage2_2_dir,
            stage2_2_ref=kwargs.pop("stage2_2_ref", EXPECTED_STAGE2_2_TAG),
            output_dir=out_dir,
            git_provenance_before=prov, git_provenance_after=prov,
            snapshot_override=snapshot_override,
            generated_at=generated_at, enforce_input_sha=enforce_input_sha,
            **kwargs)


def _normalize_manifest_paths(raw: bytes) -> bytes:
    m = json.loads(raw.decode("utf-8"))
    for rec in m.get("output_files", {}).values():
        rec["path"] = "<OUT>/" + str(rec.get("path", "")).split("/")[-1]
    return json.dumps(m, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _load_payload() -> dict:
    return load_inputs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR)


def _load_full_snapshot() -> pd.DataFrame:
    return pd.read_csv(STAGE1_DIR / "v004c_d1_snapshot_v001.csv",
                       dtype={"code": str}, float_precision="round_trip")


def _assert_bytes_equal(a: bytes, b: bytes, what: str) -> None:
    if a != b:
        raise AssertionError(f"输出不一致: {what}")


def _df_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _small_snapshot_frame() -> pd.DataFrame:
    """合成 snapshot: 6 行六月 + 6 行七月, 13 个读取列 + 标签/结果列。"""
    rng = np.random.default_rng(20260806)
    dates = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
             "2026-06-05", "2026-06-08",
             "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06",
             "2026-07-07", "2026-07-08"]
    n = len(dates)
    data: dict[str, Any] = {"signal_date": dates}
    for col in SNAPSHOT_READ_COLUMNS:
        if col == "signal_date":
            continue
        data[col] = np.round(rng.uniform(0.0, 1.0, n), 6)
    # 派生源列按合理量纲 (供派生公式使用)
    data["d1_open_to_close_return_raw"] = np.round(
        rng.uniform(-0.1, 0.1, n), 6)
    data["d1_close_to_ma5_raw"] = np.round(rng.uniform(-0.05, 0.2, n), 6)
    data["break_volume_ratio_vs_board_days"] = np.round(
        rng.uniform(0.1, 5.0, n), 6)
    data["volume_above_d1_close_ratio"] = np.round(rng.uniform(0.0, 1.0, n), 6)
    # 标签/结果列 (阶段2.3 不得读取; 供标签扰动测试; bool dtype 保持 setitem 严格)
    data["target7_daily_d2open_d3high"] = np.array(
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    data["tail_loss_daily_5pct"] = np.array(
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    data["d2open_to_d3high_return"] = np.round(
        rng.uniform(-0.2, 0.3, n), 6)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 版本锚定 (任务二十五)
# ---------------------------------------------------------------------------
class V004cStage23VersionAnchorTest(unittest.TestCase):
    def test_version_constants(self):
        self.assertEqual(EXPECTED_BRANCH, "research-sample-analysis")
        self.assertEqual(EXPECTED_STAGE1_DATA_COMMIT,
                         "a65661f4738b849a06efb4859d4100842871e971")
        self.assertEqual(EXPECTED_STAGE2_1_DATA_COMMIT,
                         "11f455dc9a6643a67250321cce68868be904cf06")
        self.assertEqual(EXPECTED_STAGE2_2_DATA_COMMIT,
                         "fba302ffc58deb7b15812e4dd2ada113ae1f3a34")
        self.assertEqual(EXPECTED_STAGE2_2_TAG,
                         "v004c-stage2-2-univariate-0.1")

    def test_stage2_2_tag_target_resolves(self):
        with _fake_env(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR) as prov:
            chain = validate_git_chain(prov.git_root, EXPECTED_STAGE2_2_TAG)
        self.assertEqual(chain["stage2_2_tag_target"],
                         EXPECTED_STAGE2_2_DATA_COMMIT)
        self.assertTrue(chain["data_commit_ancestor"])
        self.assertEqual(chain["branch"], EXPECTED_BRANCH)

    def test_wrong_ref_rejected(self):
        with _fake_env(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR) as prov:
            with self.assertRaises(DatasetValidationError):
                validate_git_chain(prov.git_root, "v004c-stage2-2-other-0.1")

    def test_tag_target_mismatch_rejected(self):
        def bad_git(args, cwd):
            if args[1] == "rev-parse" and "stage2-2" in args[-1]:
                return "0" * 40 + "\n"
            return _fake_git_calls(args, cwd)

        with mock.patch("src.v004c_candidate_model_specs._run_git",
                        side_effect=bad_git):
            prov = GitProvenance(git_root=STAGE1_DIR.parent, git_head="a" * 40,
                                 git_status="", git_dirty=False)
            with self.assertRaises(DatasetValidationError):
                validate_git_chain(prov.git_root, EXPECTED_STAGE2_2_TAG)

    def test_wrong_branch_rejected(self):
        def bad_git(args, cwd):
            if args[1] == "branch":
                return "other-branch\n"
            return _fake_git_calls(args, cwd)

        with mock.patch("src.v004c_candidate_model_specs._run_git",
                        side_effect=bad_git):
            prov = GitProvenance(git_root=STAGE1_DIR.parent, git_head="a" * 40,
                                 git_status="", git_dirty=False)
            with self.assertRaises(DatasetValidationError):
                validate_git_chain(prov.git_root, EXPECTED_STAGE2_2_TAG)


# ---------------------------------------------------------------------------
# 输入与 SHA 验证
# ---------------------------------------------------------------------------
class V004cStage23InputVerificationTest(unittest.TestCase):
    def test_snapshot_reads_only_allowed_columns(self):
        payload = _load_payload()
        self.assertEqual(
            sorted(payload["snapshot"].columns), sorted(SNAPSHOT_READ_COLUMNS))
        self.assertNotIn("target7_daily_d2open_d3high",
                         payload["snapshot"].columns)
        self.assertNotIn("recognition_score", payload["snapshot"].columns)

    def test_forbidden_input_files_not_loaded(self):
        payload = _load_payload()
        loaded = set(payload["stage21_frames"]) | set(payload["stage22_frames"])
        self.assertNotIn("v004c_training_d1_v001.csv", loaded)
        self.assertNotIn("v004c_d1_existing_model_audit_v001.csv", loaded)

    def test_git_head_files_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = base / "s1"
            s21 = base / "s21"
            s22 = base / "s22"
            shutil.copytree(STAGE1_DIR, s1)
            shutil.copytree(STAGE2_1_DIR, s21)
            shutil.copytree(STAGE2_2_DIR, s22)
            (s22 / "git_head.txt").write_text("0" * 40 + "\n", encoding="utf-8")
            with _fake_env(s1, s21, s22) as prov:
                with self.assertRaises(DatasetValidationError):
                    run_v004c_stage2_3_candidate_specs(
                        stage1_dir=s1, stage2_1_dir=s21, stage2_2_dir=s22,
                        stage2_2_ref=EXPECTED_STAGE2_2_TAG,
                        output_dir=base / "out",
                        git_provenance_before=prov,
                        git_provenance_after=prov,
                        generated_at=FIXED_GENERATED_AT)

    def test_input_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = base / "s1"
            s21 = base / "s21"
            s22 = base / "s22"
            shutil.copytree(STAGE1_DIR, s1)
            shutil.copytree(STAGE2_1_DIR, s21)
            shutil.copytree(STAGE2_2_DIR, s22)
            # 修改阶段2.2 summary 文件 → 与 manifest SHA 不符
            bad = s22 / "v004c_stage2_2_univariate_summary_dev_v001.csv"
            bad.write_text(bad.read_text(encoding="utf-8-sig") + "x",
                           encoding="utf-8-sig")
            with _fake_env(s1, s21, s22) as prov:
                with self.assertRaises(DatasetValidationError):
                    run_v004c_stage2_3_candidate_specs(
                        stage1_dir=s1, stage2_1_dir=s21, stage2_2_dir=s22,
                        stage2_2_ref=EXPECTED_STAGE2_2_TAG,
                        output_dir=base / "out",
                        git_provenance_before=prov,
                        git_provenance_after=prov,
                        generated_at=FIXED_GENERATED_AT)

    def test_input_sha_all_verified_on_real_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            iv = result["manifest"]["input_verification"]
            self.assertTrue(iv["input_sha_all_verified"])
            self.assertTrue(iv["stage1_files_unchanged"])
            self.assertTrue(iv["stage2_1_files_unchanged"])
            self.assertTrue(iv["stage2_2_files_unchanged"])
            self.assertEqual(iv["input_rows"], 333)
            self.assertEqual(iv["signal_dates"], 42)
            self.assertEqual(iv["dev_rows"], 173)
            self.assertEqual(iv["locked_rows"], 160)


# ---------------------------------------------------------------------------
# 成员与硬门 (任务七/八)
# ---------------------------------------------------------------------------
class V004cStage23MembershipGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _load_payload()

    def test_fifty_nine_features_complete(self):
        boundary = build_candidate_boundary(self.payload)
        self.assertEqual(len(boundary), 59)
        self.assertEqual(boundary["feature_name"].nunique(), 59)

    def test_twelve_unique_selected_features(self):
        self.assertEqual(UNIQUE_SELECTED_FEATURES, EXPECTED_UNIQUE)
        self.assertEqual(len(UNIQUE_SELECTED_FEATURES), 12)

    def test_four_models_three_substantive(self):
        catalog = build_model_catalog()
        self.assertEqual(len(catalog), 4)
        self.assertEqual(int((catalog["role"] != "BASELINE").sum()), 3)
        self.assertEqual(set(catalog["model_id"]), {"M0", "M1", "M2", "M3"})

    def test_m1_m2_m3_five_features_and_order(self):
        for mid, expected in (("M1", EXPECTED_M1), ("M2", EXPECTED_M2),
                              ("M3", EXPECTED_M3)):
            feats = [f["feature_name"] for f in MODEL_BY_ID[mid]["features"]]
            self.assertEqual(tuple(feats), expected)

    def test_m1_all_primary(self):
        spec, checks = build_model_feature_spec(self.payload)
        self.assertTrue(checks["m1_all_primary"])
        m1 = spec[spec["model_id"] == "M1"]
        self.assertEqual(set(m1["allowlist_tier"]), {"PRIMARY"})

    def test_m2_three_sensitivity_alternatives(self):
        spec, checks = build_model_feature_spec(self.payload)
        self.assertEqual(checks["m2_sensitivity_alternatives"],
                         sorted(["high_zone_amount_ratio",
                                 "late_day_sell_amount_ratio",
                                 "d1_vwap_to_close_gap"]))
        m2 = spec[spec["model_id"] == "M2"]
        sens = m2[m2["allowlist_tier"] == "SENSITIVITY"]["feature_name"]
        self.assertEqual(sorted(sens), sorted(checks["m2_sensitivity_alternatives"]))

    def test_m3_derived_and_mechanism(self):
        spec, checks = build_model_feature_spec(self.payload)
        self.assertTrue(checks["m3_derived_primary"])
        m3 = spec[spec["model_id"] == "M3"]
        derived = m3[m3["feature_name"] != "late_day_sell_volume_ratio"]
        self.assertEqual(set(derived["source_or_derived"]), {"derived"})
        self.assertEqual(set(derived["allowlist_tier"]), {"PRIMARY"})
        # allowlist 机制组为 PREDECLARED_DERIVED (阶段2.1 冻结准入属性)
        self.assertEqual(set(derived["mechanism_group"]),
                         {"PREDECLARED_DERIVED"})
        # 底层机制 (任务 7.4 人工声明)
        self.assertEqual(
            set(derived["underlying_mechanism"]),
            {"D1_PRICE_ACTION", "D1_MA_POSITION", "D1_VOLUME_ACTIVITY",
             "D1_CHIP_DISTRIBUTION"})

    def test_forbidden_fields_absent(self):
        spec, _ = build_model_feature_spec(self.payload)
        names = set(spec["feature_name"])
        for token in ("recognition_score", "v004a", "v002", "v005",
                      "signal_date", "break_date", "d2_", "d3_", "target",
                      "outcome", "bucket", "rank", "tail_loss", "label"):
            self.assertFalse(any(token in n for n in names),
                             f"成员包含禁止字段 {token}")

    def test_no_bucket_members(self):
        spec, _ = build_model_feature_spec(self.payload)
        self.assertNotIn("BUCKET", set(spec["analysis_type"]))

    def test_mutual_exclusion_group_at_most_once_per_model(self):
        spec, checks = build_model_feature_spec(self.payload)
        for mid in ("M1", "M2", "M3"):
            groups = [g for g in checks[f"{mid}_mutual_exclusion_groups"] if g]
            self.assertEqual(len(groups), len(set(groups)),
                             f"{mid} 互斥组重复")
            sub = spec[spec["model_id"] == mid]
            self.assertEqual(len(sub), 5)

    def test_no_exact_duplicate_or_canonical_alias_in_model(self):
        payload = _load_payload()
        _, checks = build_model_feature_spec(payload)
        self.assertGreater(checks["exact_duplicate_pairs_checked"], 0)
        self.assertGreater(checks["exact_inverse_pairs_checked"], 0)
        # 构建不抛错即通过 (硬门 fail closed)

    def test_no_derived_source_coexistence_in_m3(self):
        m3_names = set(f["feature_name"] for f in MODEL_BY_ID["M3"]["features"])
        direct_sources = {
            "d1_open_to_close_return_raw", "d1_close_to_ma5_raw",
            "break_volume_ratio_vs_board_days", "volume_above_d1_close_ratio"}
        self.assertFalse(m3_names & direct_sources)

    def test_membership_gate_fails_closed_on_missing_allowlist_entry(self):
        payload = _load_payload()
        bad_primary = payload["stage21_frames"][
            "v004c_feature_allowlist_primary_v001.csv"].copy()
        bad_primary = bad_primary[bad_primary["feature_name"] != "break_open_return"]
        payload["stage21_frames"][
            "v004c_feature_allowlist_primary_v001.csv"] = bad_primary
        with self.assertRaises(DatasetValidationError):
            build_model_feature_spec(payload)

    def test_feature_spec_15_rows_with_directions(self):
        spec, _ = build_model_feature_spec(self.payload)
        self.assertEqual(len(spec), 15)
        expected_dir = {
            ("M1", "break_open_return"): "+",
            ("M1", "down_bar_volume_ratio"): "+",
            ("M1", "high_zone_volume_ratio"): "-",
            ("M1", "late_day_sell_volume_ratio"): "-",
            ("M1", "d1_close_to_vwap_raw"): "-",
            ("M2", "break_open_return"): "+",
            ("M2", "down_bar_volume_ratio"): "+",
            ("M2", "high_zone_amount_ratio"): "-",
            ("M2", "late_day_sell_amount_ratio"): "-",
            ("M2", "d1_vwap_to_close_gap"): "+",
            ("M3", "overrepair"): "-",
            ("M3", "ma5_overheat_10"): "-",
            ("M3", "break_volume_abnormality"): "-",
            ("M3", "profit_chip_ratio"): "+",
            ("M3", "late_day_sell_volume_ratio"): "-",
        }
        for _, r in spec.iterrows():
            self.assertEqual(r["expected_direction"],
                             expected_dir[(r["model_id"], r["feature_name"])])
            self.assertFalse(bool(r["direction_is_constraint"]))


# ---------------------------------------------------------------------------
# 证据扰动不变 (任务八 8.7 / 二十四 24.1)
# ---------------------------------------------------------------------------
def _perturbed_evidence_payload() -> dict:
    import copy

    payload = _load_payload()
    p = copy.deepcopy(payload)
    rng = np.random.default_rng(7)
    summary = p["stage22_frames"][
        "v004c_stage2_2_univariate_summary_dev_v001.csv"].copy()
    n = len(summary)
    summary["effect_value"] = 0.0
    summary["effect_direction"] = summary["effect_direction"].map(
        lambda d: {"+": "-", "-": "+", "": "", "None": ""}.get(str(d).strip(), str(d).strip()))
    summary["signed_auc_raw"] = rng.permutation(
        summary["signed_auc_raw"].to_numpy())
    summary["flags"] = rng.permutation(summary["flags"].to_numpy())
    summary["lodo_sign_consistency"] = rng.permutation(
        summary["lodo_sign_consistency"].to_numpy())
    summary["lodo_max_absolute_delta"] = rng.permutation(
        summary["lodo_max_absolute_delta"].to_numpy())
    summary["board_direction_agreement_status"] = rng.permutation(
        summary["board_direction_agreement_status"].to_numpy())
    summary["board_2_effect"] = rng.permutation(
        summary["board_2_effect"].to_numpy())
    summary["board_3_effect"] = rng.permutation(
        summary["board_3_effect"].to_numpy())
    p["stage22_frames"][
        "v004c_stage2_2_univariate_summary_dev_v001.csv"] = summary
    bootstrap = p["stage22_frames"][
        "v004c_stage2_2_cluster_bootstrap_dev_v001.csv"].copy()
    bootstrap["bootstrap_ci_2_5"] = rng.permutation(
        bootstrap["bootstrap_ci_2_5"].to_numpy())
    bootstrap["bootstrap_ci_97_5"] = rng.permutation(
        bootstrap["bootstrap_ci_97_5"].to_numpy())
    p["stage22_frames"][
        "v004c_stage2_2_cluster_bootstrap_dev_v001.csv"] = bootstrap
    return p


class V004cStage23EvidencePerturbationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _load_payload()
        cls.perturbed = _perturbed_evidence_payload()

    def _spec_outputs(self, payload) -> dict[str, bytes]:
        catalog = build_model_catalog()
        fspec, _ = build_model_feature_spec(payload)
        prep = build_preprocessing_spec(catalog, fspec)
        cspec = build_candidate_specs_json(fspec)
        proto = build_evaluation_protocol_json()
        return {
            "catalog": _df_bytes(catalog),
            "feature_spec": _df_bytes(fspec),
            "preprocessing": _df_bytes(prep),
            "candidate_specs": _dump_json(cspec).encode("utf-8"),
            "protocol": _dump_json(proto).encode("utf-8"),
        }

    def test_runtime_membership_invariant_to_evidence_field_mutation(self):
        base_out = self._spec_outputs(self.base)
        pert_out = self._spec_outputs(self.perturbed)
        for name in base_out:
            _assert_bytes_equal(base_out[name], pert_out[name], name)

    def test_evidence_changes_under_perturbation(self):
        base_ev = _df_bytes(build_selected_feature_evidence(self.base))
        pert_ev = _df_bytes(build_selected_feature_evidence(self.perturbed))
        self.assertNotEqual(base_ev, pert_ev,
                            "证据扰动必须实际改变证据表 (否则扰动无效)")

    def test_boundary_selection_unchanged_under_evidence_perturbation(self):
        base = build_candidate_boundary(self.base)
        pert = build_candidate_boundary(self.perturbed)
        self.assertEqual(
            base[["feature_name", "boundary_status"]].to_csv(index=False),
            pert[["feature_name", "boundary_status"]].to_csv(index=False))


# ---------------------------------------------------------------------------
# 标签扰动不变 (任务二十四 24.2)
# ---------------------------------------------------------------------------
class V004cStage23LabelPerturbationTest(unittest.TestCase):
    def _mutate(self, frame: pd.DataFrame, kind: str) -> pd.DataFrame:
        out = frame.copy()
        target = "target7_daily_d2open_d3high"
        june_mask = frame["signal_date"] <= "2026-06-30"
        july_mask = ~june_mask
        if kind == "june_target_shuffle":
            out.loc[june_mask, target] = np.random.default_rng(1).permutation(
                frame.loc[june_mask, target].to_numpy())
        elif kind == "june_target_all_zero":
            out.loc[june_mask, target] = False
        elif kind == "july_target_shuffle":
            out.loc[july_mask, target] = np.random.default_rng(2).permutation(
                frame.loc[july_mask, target].to_numpy())
        elif kind == "july_target_all_one":
            out.loc[july_mask, target] = True
        elif kind == "tail_loss_flip":
            out["tail_loss_daily_5pct"] = ~frame["tail_loss_daily_5pct"]
        elif kind == "outcome_column_modified":
            out["d2open_to_d3high_return"] = -frame["d2open_to_d3high_return"]
        else:
            raise AssertionError(kind)
        return out

    def test_runtime_membership_invariant_to_target_column_mutation_redundancy(self):
        base = _small_snapshot_frame()
        base_bytes = _df_bytes(build_candidate_redundancy_audit(base))
        for kind in ("june_target_shuffle", "june_target_all_zero",
                     "july_target_shuffle", "july_target_all_one",
                     "tail_loss_flip", "outcome_column_modified"):
            mutated = self._mutate(base, kind)
            _assert_bytes_equal(
                base_bytes, _df_bytes(build_candidate_redundancy_audit(mutated)),
                f"Target 列扰动 {kind} 改变冗余审计输出")

    def test_label_perturbations_do_not_change_other_outputs(self):
        base = _small_snapshot_frame()
        catalog = build_model_catalog()
        fspec, _ = build_model_feature_spec(_load_payload())
        prep = build_preprocessing_spec(catalog, fspec)
        cspec = build_candidate_specs_json(fspec)
        proto = build_evaluation_protocol_json()
        refs = {
            "catalog": _df_bytes(catalog),
            "preprocessing": _df_bytes(prep),
            "candidate_specs": _dump_json(cspec).encode("utf-8"),
            "protocol": _dump_json(proto).encode("utf-8"),
        }
        for kind in ("july_target_all_one", "june_target_shuffle",
                     "outcome_column_modified"):
            mutated = self._mutate(base, kind)
            # 这些构建不接收 snapshot → 逐字节不变
            _assert_bytes_equal(refs["catalog"], _df_bytes(build_model_catalog()),
                                f"标签扰动 {kind} 改变 catalog")

    def test_runtime_membership_invariant_to_target_column_mutation_full_run_july_all_one(self):
        base = _load_full_snapshot()
        mutated = base.copy()
        mutated.loc[mutated["signal_date"] >= "2026-07-01",
                    "target7_daily_d2open_d3high"] = True
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_a = tmp / "out_a"
            out_b = tmp / "out_b"
            result_a = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out_a,
                                  snapshot_override=base)
            result_b = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out_b,
                                  snapshot_override=mutated)
            for name in OUTPUT_NAMES:
                a = (out_a / name).read_bytes()
                b = (out_b / name).read_bytes()
                if name == "v004c_stage2_3_manifest.json":
                    a = _normalize_manifest_paths(a)
                    b = _normalize_manifest_paths(b)
                _assert_bytes_equal(a, b, f"七月标签全1后 {name} 变化")
            self.assertTrue(result_a["manifest"]["holdout_lock_audit"]["all_pass"])

    def test_runtime_membership_invariant_to_target_column_mutation_full_run_june_shuffle(self):
        base = _load_full_snapshot()
        mutated = base.copy()
        june = mutated["signal_date"] <= "2026-06-30"
        mutated.loc[june, "target7_daily_d2open_d3high"] = np.random.default_rng(
            3).permutation(base.loc[june, "target7_daily_d2open_d3high"].to_numpy())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_a = tmp / "out_a"
            out_b = tmp / "out_b"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out_a,
                       snapshot_override=base)
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out_b,
                       snapshot_override=mutated)
            for name in OUTPUT_NAMES:
                a = (out_a / name).read_bytes()
                b = (out_b / name).read_bytes()
                if name == "v004c_stage2_3_manifest.json":
                    a = _normalize_manifest_paths(a)
                    b = _normalize_manifest_paths(b)
                _assert_bytes_equal(a, b, f"六月标签打乱后 {name} 变化")


# ---------------------------------------------------------------------------
# 特征扰动 (任务二十四 24.3)
# ---------------------------------------------------------------------------
class V004cStage23FeaturePerturbationTest(unittest.TestCase):
    def test_feature_perturbation_changes_redundancy_but_not_membership(self):
        base = _small_snapshot_frame()
        mutated = base.copy()
        # Spearman 对常数倍缩放不变 → 用改变秩的扰动 (单行值改为离群)
        mutated.loc[0, "break_open_return"] = 99.0
        base_red = _df_bytes(build_candidate_redundancy_audit(base))
        mut_red = _df_bytes(build_candidate_redundancy_audit(mutated))
        self.assertNotEqual(base_red, mut_red,
                            "特征扰动必须改变冗余审计输出")
        # 成员不得改变 (构建函数不接收特征值)
        payload = _load_payload()
        fspec, _ = build_model_feature_spec(payload)
        self.assertEqual(len(fspec), 15)


# ---------------------------------------------------------------------------
# 冗余审计 (任务十二)
# ---------------------------------------------------------------------------
class V004cStage23RedundancyAuditTest(unittest.TestCase):
    def test_thirty_rows_ten_per_model(self):
        frame = _small_snapshot_frame()
        red = build_candidate_redundancy_audit(frame)
        self.assertEqual(len(red), 30)
        for mid in ("M1", "M2", "M3"):
            self.assertEqual(int((red["model_id"] == mid).sum()), 10)
        self.assertEqual(
            list(red.columns),
            ["model_id", "position_a", "position_b", "feature_a",
             "feature_b", "dev_non_null_pair_count", "dev_spearman_rho",
             "locked_non_null_pair_count", "locked_spearman_rho",
             "absolute_dev_rho", "absolute_locked_rho",
             "maximum_absolute_rho", "absolute_rho_shift", "redundancy_flag"])

    def test_spearman_fixture_known_values(self):
        frame = pd.DataFrame({
            "signal_date": ["2026-06-01"] * 6,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "z": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        })
        # 手工调用内部 _spearman_pair 验证已知值
        from src.v004c_candidate_model_specs import _spearman_pair
        count, rho = _spearman_pair(frame["x"], frame["y"])
        self.assertEqual(count, 6)
        self.assertAlmostEqual(rho, 1.0)
        _, rho_inv = _spearman_pair(frame["x"], frame["z"])
        self.assertAlmostEqual(rho_inv, -1.0)

    def test_constant_feature_rho_none(self):
        from src.v004c_candidate_model_specs import _spearman_pair
        count, rho = _spearman_pair(pd.Series([1.0] * 5),
                                    pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(count, 5)
        self.assertIsNone(rho)

    def test_missing_pair_count_excluded(self):
        from src.v004c_candidate_model_specs import _spearman_pair
        a = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        b = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        count, rho = _spearman_pair(a, b)
        self.assertEqual(count, 2)  # 只有 (1,1) 与 (5,5) 双非缺失
        self.assertAlmostEqual(rho, 1.0)

    def test_dev_locked_slices_separate(self):
        frame = _small_snapshot_frame()
        red = build_candidate_redundancy_audit(frame)
        self.assertEqual(len(red), 30)
        # 六月 6 行 / 七月 6 行: 所有配对计数必须分别等于 6 / 6
        self.assertEqual(set(red["dev_non_null_pair_count"]), {6})
        self.assertEqual(set(red["locked_non_null_pair_count"]), {6})

    def test_redundancy_flag_thresholds(self):
        frame = pd.DataFrame({
            "signal_date": ["2026-06-01"] * 8,
            "x": list(range(1, 9)),
            "y": list(range(1, 9)),
            "z": list(range(8, 0, -1)),
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
        })
        from src.v004c_candidate_model_specs import _spearman_pair
        _, rho_xy = _spearman_pair(frame["x"], frame["y"])
        self.assertEqual(rho_xy, 1.0)
        _, rho_xz = _spearman_pair(frame["x"], frame["z"])
        self.assertEqual(rho_xz, -1.0)
        _, rho_xw = _spearman_pair(frame["x"], frame["w"])
        # 单调 + 末位离群 → 仍应高相关
        self.assertGreaterEqual(rho_xw, 0.85)


# ---------------------------------------------------------------------------
# 预处理规格 (任务十/十一)
# ---------------------------------------------------------------------------
class V004cStage23PreprocessingSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _load_payload()
        cls.catalog = build_model_catalog()
        cls.fspec, _ = build_model_feature_spec(cls.payload)
        cls.prep = build_preprocessing_spec(cls.catalog, cls.fspec)

    def test_model_hyperparameters_frozen(self):
        m1 = self.catalog[self.catalog["model_id"] == "M1"].iloc[0]
        self.assertEqual(m1["model_family"], "binary_logistic_regression")
        self.assertEqual(m1["penalty"], "L2")
        self.assertEqual(m1["C"], 1.0)
        self.assertEqual(m1["solver"], "lbfgs")
        self.assertTrue(bool(m1["fit_intercept"]))
        self.assertIsNone(m1["class_weight"])
        self.assertEqual(m1["max_iter"], 5000)
        self.assertEqual(m1["tol"], 1e-8)
        self.assertEqual(m1["random_seed"], 20260806)
        self.assertFalse(bool(m1["direction_is_constraint"]))
        m0 = self.catalog[self.catalog["model_id"] == "M0"].iloc[0]
        self.assertEqual(m0["model_family"], "intercept_prevalence_baseline")
        self.assertEqual(m0["penalty"], "none")
        self.assertEqual(m0["feature_count"], 0)

    def test_tol_unified_1e8_no_1e10_leftover(self):
        # 任务 8: 候选模型规格统一 tol=1e-8, max_iter=5000 (M0 无 tol)
        self.assertEqual(
            set(self.catalog.loc[self.catalog["model_id"] != "M0", "tol"]),
            {1e-8})
        proto = build_evaluation_protocol_json()
        self.assertEqual(proto["tol"], 1e-8)
        self.assertEqual(proto["max_iter"], 5000)
        cspec = build_candidate_specs_json(self.fspec)
        text = (
            self.prep.to_csv() + self.catalog.to_csv() + _dump_json(cspec))
        self.assertNotIn("1e-10", text)
        self.assertNotIn("1e-06", text)
        # 校准指标估计参数 (任务 9.5) 固定 tol=1e-10 — 合法存在, 非候选超参数
        self.assertEqual(proto["metric_definitions"]["calibration"]["tol"],
                         1e-10)

    def test_preprocessing_rows_19(self):
        self.assertEqual(len(self.prep), 19)  # 4 模型级 + 15 特征级
        model_rows = self.prep[self.prep["spec_level"] == "MODEL"]
        feature_rows = self.prep[self.prep["spec_level"] == "FEATURE"]
        self.assertEqual(len(model_rows), 4)
        self.assertEqual(len(feature_rows), 15)

    def test_fold_clip_z_spec(self):
        fc = self.prep[self.prep["preprocess_policy"] == "FOLD_CLIP_Z"]
        self.assertGreater(len(fc), 0)
        for _, r in fc.iterrows():
            self.assertEqual(r["fold_clip_lower_quantile"], 0.01)
            self.assertEqual(r["fold_clip_upper_quantile"], 0.99)
            self.assertEqual(r["fold_clip_ddof"], 0)
            self.assertEqual(r["fold_std_floor_fail_closed"], 1e-12)
            self.assertIn("训练窗口", r["train_only_parameter_scope"])

    def test_raw_binary_spec(self):
        rb = self.prep[(self.prep["feature_name"] == "ma5_overheat_10")]
        self.assertEqual(len(rb), 1)
        row = rb.iloc[0]
        self.assertEqual(row["preprocess_policy"], "RAW_BINARY")
        self.assertEqual(row["binary_allowed_values"], "0|1")

    def test_missing_policy_fail_closed(self):
        feature_rows = self.prep[self.prep["spec_level"] == "FEATURE"]
        self.assertEqual(set(feature_rows["missing_policy"]),
                         {"FAIL_CLOSED"})

    def test_derived_row_wise_compute(self):
        derived_rows = self.prep[
            self.prep["feature_name"].isin(
                ("overrepair", "ma5_overheat_10",
                 "break_volume_abnormality", "profit_chip_ratio"))]
        self.assertEqual(len(derived_rows), 4)
        for _, r in derived_rows.iterrows():
            self.assertIn("逐行", r["derived_row_wise_compute"])

    def test_forbidden_preprocessing_absent(self):
        prep = build_preprocessing_spec(self.catalog, self.fspec)
        joined = " ".join(str(v) for v in prep["preprocess_policy"]) + \
            self.prep.to_csv()
        for token in ("PCA", "bucket", "BUCKET"):
            self.assertNotIn(token, joined)


# ---------------------------------------------------------------------------
# 候选规格 JSON / 评价协议 JSON
# ---------------------------------------------------------------------------
class V004cStage23SpecJsonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _load_payload()
        cls.catalog = build_model_catalog()
        cls.fspec, _ = build_model_feature_spec(cls.payload)
        cls.cspec = build_candidate_specs_json(cls.fspec)
        cls.proto = build_evaluation_protocol_json()

    def test_candidate_specs_json_content(self):
        c = self.cspec
        self.assertEqual(c["candidate_selection_mode"],
                         "human_fixed_after_june_development_review")
        self.assertEqual(c["model_count"], 4)
        self.assertEqual(c["substantive_model_count"], 3)
        self.assertEqual(c["selected_unique_feature_count"], 12)
        self.assertEqual(c["unique_selected_features"],
                         list(UNIQUE_SELECTED_FEATURES))
        self.assertFalse(c["direction_is_constraint"])
        by_id = {m["model_id"]: m for m in c["models"]}
        self.assertEqual(by_id["M0"]["feature_count"], 0)
        self.assertEqual(by_id["M0"]["features"], [])
        for mid, expected in (("M1", EXPECTED_M1), ("M2", EXPECTED_M2),
                              ("M3", EXPECTED_M3)):
            feats = [f["feature_name"] for f in by_id[mid]["features"]]
            self.assertEqual(tuple(feats), expected)
            self.assertTrue(all(f["direction_is_constraint"] is False
                                for f in by_id[mid]["features"]))
            self.assertTrue(all(f["mechanism_group"]
                                for f in by_id[mid]["features"]))
            self.assertTrue(all(f["allowlist_tier"]
                                for f in by_id[mid]["features"]))
            self.assertTrue(all(f["preprocess_policy"]
                                for f in by_id[mid]["features"]))
            self.assertTrue(all(f["selection_rationale"]
                                for f in by_id[mid]["features"]))

    def test_candidate_specs_json_no_forbidden_content(self):
        text = _dump_json(self.cspec)
        # intercept_prevalence_baseline 是 M0 的合法模型族名, 不检查 intercept_;
        # 检查系数/预测概率/表现/七月标签指标键 (july 仅出现在规范字段名中)
        for token in ("coefficient", "predicted_probability",
                      "performance", "july_auc", "july_log_loss",
                      "july_brier", "july_target", "july_accuracy", "auc_"):
            self.assertNotIn(token, text.lower())
        parsed = json.loads(text)
        self.assertEqual(parsed["models"][0]["model_family"],
                         "intercept_prevalence_baseline")
        # 候选规格 JSON 记录规格在七月揭示前冻结 (研究设计披露)
        self.assertTrue(any("candidate_specs_frozen_before_july_label_access"
                            in note for note in parsed["notes"]))

    def test_evaluation_protocol_complete(self):
        p = self.proto
        self.assertEqual(p["model_family"], "binary_logistic_regression")
        self.assertEqual(p["penalty"], "L2")
        self.assertEqual(p["C"], 1.0)
        self.assertEqual(p["solver"], "lbfgs")
        self.assertEqual(p["max_iter"], 5000)
        self.assertEqual(p["tol"], 1e-8)
        self.assertIsNone(p["class_weight"])
        self.assertEqual(p["random_seed"], 20260806)
        self.assertTrue(p["fit_intercept"])
        self.assertEqual(p["preprocessing"]["missing_policy"], "FAIL_CLOSED")
        self.assertEqual(p["preprocessing"]["fold_clip_z"]["lower_quantile"], 0.01)
        self.assertEqual(p["preprocessing"]["fold_clip_z"]["upper_quantile"], 0.99)
        self.assertEqual(p["july_evaluation"]["train_range"],
                         ["2026-06-01", "2026-06-30"])
        self.assertEqual(p["july_evaluation"]["test_range"],
                         ["2026-07-01", "2026-07-29"])

    def test_eligibility_gates_six(self):
        gates = self.proto["final_eligibility_gates"]
        self.assertEqual(len(gates), 6)
        joined = " ".join(gates)
        self.assertIn("收敛", joined)
        self.assertIn("[0,1]", joined)
        self.assertIn("log loss", joined)
        self.assertIn("0.50", joined)
        self.assertEqual(self.proto["rejection_state"],
                         "REJECT_NO_STABLE_MODEL")

    def test_comparison_algorithm_spec(self):
        cc = self.proto["candidate_comparison"]
        self.assertEqual(cc["mode"], "global_sequential_tolerance_filter")
        self.assertTrue(cc["eligibility_first"])
        self.assertEqual(
            cc["oos_log_loss"]["operation"],
            "retain_within_absolute_difference_of_global_minimum")
        self.assertEqual(cc["oos_log_loss"]["tolerance"], 0.005)
        self.assertEqual(
            cc["oos_brier"]["operation"],
            "retain_within_absolute_difference_of_stage_minimum")
        self.assertEqual(cc["oos_brier"]["tolerance"], 0.002)
        self.assertEqual(cc["july_log_loss"]["operation"], "minimum")
        self.assertEqual(cc["july_log_loss"]["tolerance"], 0.0)
        self.assertEqual(cc["feature_count"]["operation"], "minimum")
        self.assertEqual(cc["fixed_model_priority"], ["M1", "M2", "M3"])
        self.assertIn("不进行任何两模型逐对比较",
                      cc["algorithm_note"])
        self.assertNotIn("candidate_comparison_hierarchy", self.proto)

    def test_convergence_status_defined(self):
        cs = self.proto["convergence_status"]
        self.assertEqual(cs["definitions"],
                         ["CONVERGED", "MAX_ITER_REACHED",
                          "NUMERICAL_FAILURE", "NON_FINITE_COEFFICIENT",
                          "NON_FINITE_PREDICTION", "INPUT_VALIDATION_FAILURE"])
        self.assertIn("convergence_status = FAILED", cs["failed_rule"])
        self.assertIn("不得将未收敛模型标为收敛", cs["failed_rule"])

    def test_protocol_flags_all_false(self):
        flags = self.proto["flags"]
        for key in ("model_training_performed", "prediction_generated",
                    "holdout_label_access", "automatic_feature_selection",
                    "hyperparameter_search", "threshold_search",
                    "interaction_search"):
            self.assertFalse(flags[key], key)

    def test_walk_forward_gates(self):
        wf = self.proto["walk_forward"]
        self.assertEqual(wf["minimum_training_gates"],
                         {"train_signal_dates": 10, "train_rows": 80,
                          "train_positive": 20, "train_negative": 40})
        self.assertEqual(wf["split_rule"].count("signal_date"), 2)


# ---------------------------------------------------------------------------
# 证据表 (任务十九)
# ---------------------------------------------------------------------------
class V004cStage23EvidenceTableTest(unittest.TestCase):
    def test_evidence_rows_and_order(self):
        payload = _load_payload()
        ev = build_selected_feature_evidence(payload)
        self.assertEqual(len(ev), 12)
        self.assertEqual(list(ev["feature_name"]), list(UNIQUE_SELECTED_FEATURES))
        self.assertEqual(ev["feature_name"].nunique(), 12)
        # 证据值来自阶段2.2 冻结资产 (不允许为空的关键字段)
        self.assertTrue(ev["effect_metric_name"].notna().all())
        self.assertTrue(ev["analysis_type"].notna().all())
        self.assertTrue(ev["bootstrap_valid"].notna().all())


# ---------------------------------------------------------------------------
# 候选边界 (任务十八)
# ---------------------------------------------------------------------------
class V004cStage23BoundaryTest(unittest.TestCase):
    def test_boundary_statuses_allowed_only(self):
        payload = _load_payload()
        boundary = build_candidate_boundary(payload)
        statuses = set(boundary["boundary_status"])
        self.assertTrue(statuses <= {"SELECTED_FIXED",
                                     "NOT_SELECTED_NOT_REJECTED"})
        self.assertEqual(int((boundary["boundary_status"] ==
                              "SELECTED_FIXED").sum()), 12)
        self.assertEqual(int((boundary["boundary_status"] ==
                              "NOT_SELECTED_NOT_REJECTED").sum()), 47)
        selected_models = set(boundary.loc[
            boundary["selected_any_model"], "selected_models"])
        self.assertTrue(selected_models <= {"M1", "M2", "M3",
                                            "M1|M2", "M1|M3", "M2|M3"})


# ---------------------------------------------------------------------------
# 锁定审计 / 输出包 (任务二十三/二十七)
# ---------------------------------------------------------------------------
class V004cStage23OutputPackageTest(unittest.TestCase):
    def test_output_dir_gate_nonempty_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "stale.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)

    def test_deterministic_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_a = tmp / "out_a"
            out_b = tmp / "out_b"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out_a)
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out_b)
            for name in OUTPUT_NAMES:
                a = (out_a / name).read_bytes()
                b = (out_b / name).read_bytes()
                if name == "v004c_stage2_3_manifest.json":
                    a = _normalize_manifest_paths(a)
                    b = _normalize_manifest_paths(b)
                _assert_bytes_equal(a, b, f"确定性重建失败: {name}")

    def test_output_files_count_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            files = sorted(p.name for p in out.iterdir() if p.is_file())
            self.assertEqual(tuple(files), tuple(sorted(OUTPUT_NAMES)))
            self.assertEqual(len(files), 15)
            # manifest 记录其余 14 个非 manifest 文件 (从实际输出枚举)
            recorded = sorted(result["manifest"]["output_files"])
            expected = sorted(n for n in files if n != "v004c_stage2_3_manifest.json")
            self.assertEqual(recorded, expected)
            self.assertEqual(len(recorded), 14)
            self.assertNotIn("v004c_stage2_3_manifest.json", recorded)

    def test_holdout_lock_audit_all_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            audit = result["manifest"]["holdout_lock_audit"]
            self.assertTrue(audit["all_pass"])
            self.assertGreaterEqual(audit["check_count"], 20)
            self.assertEqual(audit["pass_count"], audit["check_count"])
            audit_csv = pd.read_csv(
                out / "v004c_stage2_3_holdout_lock_audit_v001.csv", dtype=str)
            self.assertEqual(len(audit_csv), audit["check_count"])
            required = {
                "model_count_is_4", "substantive_model_count_is_3",
                "m1_m2_m3_feature_count_is_5",
                "candidate_boundary_rows_is_59",
                "selected_unique_feature_count_is_12",
                "no_model_fit_call", "no_prediction_generated",
                "no_walk_forward_run", "no_automatic_feature_selection",
                "no_hyperparameter_search", "no_threshold_search",
                "no_ml_library_import", "no_logistic_instantiation",
                "july_label_columns_not_accessed",
                "snapshot_label_columns_not_accessed",
                "RUNTIME_TARGET_COLUMN_ACCESS", "JULY_LABEL_ACCESS",
                "AUTOMATIC_TARGET_DRIVEN_SELECTION",
                "JUNE_EVIDENCE_USED_FOR_MANUAL_DESIGN_DECLARED",
                "SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS",
                "JUNE_DEVELOPMENT_EVIDENCE_DISCLOSED",
                "actual_loaded_input_files_match_allowed",
                "actual_loaded_input_files_disjoint_forbidden",
                "candidate_membership_fixed_after_june_development_review",
            }
            self.assertTrue(required <= set(audit_csv["check_name"]))

    def test_manifest_complete_with_output_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            manifest = result["manifest"]
            self.assertEqual(manifest["stage2_3_version"],
                             "v004c-stage2-3-candidate-specs-0.1")
            self.assertEqual(manifest["candidate_selection_mode"],
                             "human_fixed_after_june_development_review")
            self.assertEqual(manifest["model_count"], 4)
            self.assertEqual(manifest["substantive_model_count"], 3)
            self.assertEqual(manifest["selected_unique_feature_count"], 12)
            self.assertFalse(manifest["model_training_performed"])
            self.assertFalse(manifest["prediction_generated"])
            self.assertFalse(manifest["holdout_label_access"])
            self.assertFalse(manifest["automatic_feature_selection"])
            self.assertFalse(manifest["automatic_target_driven_selection"])
            self.assertFalse(manifest["hyperparameter_search"])
            self.assertFalse(manifest["threshold_search"])
            # 阶段2.3.1 研究设计披露字段
            self.assertTrue(
                manifest["june_development_evidence_used_for_manual_design"])
            self.assertTrue(
                manifest["candidate_specs_frozen_before_july_label_access"])
            self.assertFalse(
                manifest["runtime_candidate_membership_uses_target_columns"])
            self.assertTrue(
                manifest["runtime_membership_invariant_to_target_column_mutation"])
            self.assertTrue(
                manifest["runtime_membership_invariant_to_evidence_field_mutation"])
            # 不再把旧字段作为无条件研究结论
            self.assertNotIn("candidate_membership_invariant_to_target_values",
                             manifest)
            self.assertNotIn("human_predeclared_fixed",
                             json.dumps(manifest, ensure_ascii=False))
            # 比较算法与 VWAP 核查记录
            self.assertEqual(manifest["candidate_comparison"]["mode"],
                             "global_sequential_tolerance_filter")
            self.assertIn("relationship_type",
                          manifest["vwap_reparameterization"])
            # 输出 SHA 与文件一致
            for name, rec in manifest["output_files"].items():
                path = out / name
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, rec["sha256"], name)
            # 行数
            self.assertEqual(manifest["output_files"][
                "v004c_stage2_3_model_catalog_v001.csv"]["rows"], 4)
            self.assertEqual(manifest["output_files"][
                "v004c_stage2_3_model_feature_spec_v001.csv"]["rows"], 15)
            self.assertEqual(manifest["output_files"][
                "v004c_stage2_3_preprocessing_spec_v001.csv"]["rows"], 19)
            self.assertEqual(manifest["output_files"][
                "v004c_stage2_3_selected_feature_evidence_v001.csv"]["rows"], 12)
            self.assertEqual(manifest["output_files"][
                "v004c_stage2_3_candidate_redundancy_audit_v001.csv"]["rows"], 30)
            self.assertEqual(manifest["output_files"][
                "v004c_stage2_3_candidate_boundary_v001.csv"]["rows"], 59)

    def test_manifest_no_july_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            text = (out / "v004c_stage2_3_manifest.json").read_text(
                encoding="utf-8").lower()
            # 规范字段名 (candidate_specs_frozen_before_july_label_access) 允许;
            # 七月标签指标 token 禁止
            for token in ("july_auc", "july_log_loss", "july_brier",
                          "july_target", "july_accuracy", "july_hit",
                          "july_prediction", "locked_target"):
                self.assertNotIn(token, text)

    def test_review_contains_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            review = (out / "v004c_stage2_3_review.md").read_text(
                encoding="utf-8")
            for section in ("人工设计 + 规格先冻结", "INTERCEPT_BASELINE",
                            "PRIMARY_MECHANISM_CORE",
                            "SENSITIVITY_EXPRESSION_ALTERNATIVES",
                            "PREDECLARED_DERIVED_MECHANISMS",
                            "表达替换关系", "派生与底层机制关系",
                            "12 个唯一入选特征", "无标签冗余审计",
                            "VWAP 符号重参数化披露",
                            "固定预处理", "固定模型族与超参数",
                            "七月一次性评价协议", "Walk-forward 协议",
                            "最终候选资格门",
                            "最终候选比较算法 (global_sequential_tolerance_filter)",
                            "阶段2.4 指标精确定义",
                            "Target-blind 锁定测试"):
                self.assertIn(section, review, f"review 缺少 {section}")
            # 阶段2.3.1 研究设计披露文本 (任务 4.3)
            for phrase in ("由研究人员在阅读六月阶段2.2",
                           "候选规格在任何七月标签揭示前冻结",
                           "未读取 Target 列",
                           "未根据 AUC、bootstrap、flags 或其它统计"):
                self.assertIn(phrase, review, f"review 缺少披露: {phrase}")
            for name in UNIQUE_SELECTED_FEATURES:
                self.assertIn(name, review)

    def test_no_model_fitting_interface_in_module(self):
        from src.v004c_candidate_model_specs import (
            _module_ast_scan_checks)
        for check in _module_ast_scan_checks():
            self.assertEqual(check["status"], "PASS", check["check_name"])

    def test_catalog_mechanism_groups(self):
        catalog = build_model_catalog()
        m1 = catalog[catalog["model_id"] == "M1"].iloc[0]
        self.assertEqual(m1["mechanism_groups"],
                         "D1_PRICE_ACTION|D1_VOLUME_ACTIVITY|"
                         "D1_CHIP_DISTRIBUTION|D1_LATE_DAY_PRESSURE|"
                         "D1_VWAP_POSITION")
        m2 = catalog[catalog["model_id"] == "M2"].iloc[0]
        self.assertEqual(m2["mechanism_groups"],
                         m1["mechanism_groups"])
        m3 = catalog[catalog["model_id"] == "M3"].iloc[0]
        self.assertEqual(m3["mechanism_groups"],
                         "D1_PRICE_ACTION|D1_MA_POSITION|"
                         "D1_VOLUME_ACTIVITY|D1_CHIP_DISTRIBUTION|"
                         "D1_LATE_DAY_PRESSURE")


# ---------------------------------------------------------------------------
# 固定确定性比较算法 (阶段2.3.1 修正 6.x / 任务 10.4)
# ---------------------------------------------------------------------------
def _cand(mid, eligible=True, ll=None, brier=None, july=None, fc=5):
    return {"model_id": mid, "eligible": eligible, "oos_log_loss": ll,
            "oos_brier": brier, "july_log_loss": july, "feature_count": fc}


class V004cStage231ComparisonAlgorithmTest(unittest.TestCase):
    def test_scenario_a_log_loss_wins_alone(self):
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610),
                _cand("M2", ll=0.607, brier=0.220, july=0.610),
                _cand("M3", ll=0.620, brier=0.220, july=0.610)]
        out = select_final_candidate(rows)
        self.assertEqual(out["selected_model_id"], "M1")
        self.assertEqual(out["stage"]["stage_a"], ["M1"])

    def test_scenario_b_brier_tolerance_retains_m1_m2(self):
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610),
                _cand("M2", ll=0.603, brier=0.2185, july=0.610),
                _cand("M3", ll=0.604, brier=0.225, july=0.610)]
        out = select_final_candidate(rows)
        self.assertEqual(out["stage"]["stage_a"], ["M1", "M2", "M3"])
        self.assertEqual(out["stage"]["stage_b"], ["M1", "M2"])
        self.assertEqual(out["best_oos_brier"], 0.2185)
        self.assertNotIn("M3", out["stage"]["stage_b"])

    def test_scenario_c_july_log_loss_breaks_tie(self):
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610),
                _cand("M2", ll=0.600, brier=0.2185, july=0.605)]
        out = select_final_candidate(rows)
        self.assertEqual(out["selected_model_id"], "M2")

    def test_scenario_d_feature_count_breaks_full_tie(self):
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610, fc=5),
                _cand("M2", ll=0.600, brier=0.220, july=0.610, fc=4)]
        out = select_final_candidate(rows)
        self.assertEqual(out["selected_model_id"], "M2")

    def test_scenario_e_fixed_priority_on_complete_tie(self):
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610),
                _cand("M2", ll=0.600, brier=0.220, july=0.610),
                _cand("M3", ll=0.600, brier=0.220, july=0.610)]
        out = select_final_candidate(rows)
        self.assertEqual(out["selected_model_id"], "M1")

    def test_scenario_f_empty_eligible_set_rejects(self):
        rows = [_cand("M1", eligible=False, ll=0.6, brier=0.22, july=0.6),
                _cand("M2", eligible=False, ll=0.6, brier=0.22, july=0.6)]
        out = select_final_candidate(rows)
        self.assertEqual(out["final_status"], "REJECT_NO_STABLE_MODEL")
        self.assertIsNone(out["selected_model_id"])

    def test_scenario_g_input_order_invariance(self):
        # 构造潜在非传递陷阱: 逐对比较可能不一致, 全局过滤必须唯一
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610),
                _cand("M2", ll=0.603, brier=0.2185, july=0.605),
                _cand("M3", ll=0.601, brier=0.2190, july=0.608)]
        import itertools
        results = []
        for perm in itertools.permutations(rows):
            results.append(select_final_candidate(list(perm)))
        self.assertEqual(len({r["selected_model_id"] for r in results}), 1)
        self.assertEqual(results[0]["selected_model_id"], "M2")

    def test_scenario_g_relative_difference_not_used(self):
        # 绝对差 0.004 (0.600 vs 0.604) 在容差内保留;
        # 若是相对百分比差 (0.67%) 则可能被误排除
        rows = [_cand("M1", ll=0.600, brier=0.220, july=0.610),
                _cand("M2", ll=0.604, brier=0.220, july=0.610)]
        out = select_final_candidate(rows)
        self.assertEqual(out["stage"]["stage_a"], ["M1", "M2"])


# ---------------------------------------------------------------------------
# VWAP 符号重参数化 (阶段2.3.1 修正 7.x / 任务 10.5)
# ---------------------------------------------------------------------------
class V004cStage231VwapCheckTest(unittest.TestCase):
    # 六月 5 行 + 七月 5 行 (两切片都需参与核查)
    _DATES = (["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
               "2026-06-05", "2026-07-01", "2026-07-02", "2026-07-03",
               "2026-07-04", "2026-07-05"])

    def _frame(self, a: list, b: list):
        # 同一组值重复到六月与七月两个切片
        return pd.DataFrame({
            "signal_date": self._DATES,
            "d1_close_to_vwap_raw": a + a,
            "d1_vwap_to_close_gap": b + b,
        })

    def test_exact_negative_fixture(self):
        x = [round(0.01 * i, 6) for i in range(1, 6)]
        frame = self._frame(x, [-v for v in x])
        out = compute_vwap_reparameterization(frame)
        self.assertEqual(out["relationship_type"],
                         "EXACT_NEGATIVE_REPARAMETERIZATION")
        self.assertEqual(out["dev"]["pair_non_null"], 5)
        self.assertEqual(out["locked"]["pair_non_null"], 5)
        self.assertLessEqual(out["dev"]["max_abs_sum"], 1e-12)
        self.assertLessEqual(out["locked"]["max_abs_sum"], 1e-12)
        self.assertAlmostEqual(out["dev"]["pearson_rho"], -1.0, places=12)
        self.assertAlmostEqual(out["dev"]["spearman_rho"], -1.0, places=12)
        self.assertTrue(out["dev"]["exact_negative_equivalence"])
        self.assertTrue(out["locked"]["exact_negative_equivalence"])

    def test_nonexact_fixture_not_misclassified(self):
        # 非精确反向 (含噪声) → 不得误判
        rng = np.random.default_rng(5)
        x = [round(0.01 * i, 6) for i in range(1, 6)]
        b = [-v + 0.002 * float(rng.normal()) for v in x]
        frame = self._frame(x, b)
        out = compute_vwap_reparameterization(frame)
        self.assertEqual(out["relationship_type"], "NOT_EXACT_NEGATIVE")
        self.assertFalse(out["dev"]["exact_negative_equivalence"])
        self.assertFalse(out["locked"]["exact_negative_equivalence"])

    def test_real_data_reported_not_hardcoded(self):
        payload = _load_payload()
        out = compute_vwap_reparameterization(payload["snapshot"])
        self.assertEqual(out["dev"]["pair_non_null"], 173)
        self.assertEqual(out["locked"]["pair_non_null"], 160)
        # 由数据决定: 与直接重算一致
        snapshot = payload["snapshot"]
        a = pd.to_numeric(snapshot["d1_close_to_vwap_raw"], errors="coerce")
        b = pd.to_numeric(snapshot["d1_vwap_to_close_gap"], errors="coerce")
        dev = snapshot["signal_date"] <= "2026-06-30"
        pair = dev & a.notna() & b.notna()
        self.assertAlmostEqual(
            out["dev"]["max_abs_sum"],
            float(np.max(np.abs(a[pair] + b[pair]))), places=12)
        # 真实数据不满足严格 1e-12 条件 → 如实记录 (Spearman=-1 但 max_abs_sum>0)
        self.assertEqual(out["relationship_type"], "NOT_EXACT_NEGATIVE")
        self.assertAlmostEqual(out["dev"]["spearman_rho"], -1.0, places=12)
        self.assertGreater(out["dev"]["max_abs_sum"], 1e-12)

    def test_design_disclosure_contains_vwap_object(self):
        payload = _load_payload()
        vw = compute_vwap_reparameterization(payload["snapshot"])
        dd = build_design_disclosure(vw)
        self.assertEqual(dd["vwap_reparameterization"]["relationship_type"],
                         vw["relationship_type"])


# ---------------------------------------------------------------------------
# 研究设计披露 (阶段2.3.1 修正 4.x / 任务 10.1-10.2)
# ---------------------------------------------------------------------------
class V004cStage231DesignDisclosureTest(unittest.TestCase):
    def test_design_disclosure_declares_june_evidence_use(self):
        payload = _load_payload()
        vw = compute_vwap_reparameterization(payload["snapshot"])
        dd = build_design_disclosure(vw)
        self.assertEqual(
            dd["june_role"], "development_and_manual_candidate_design")
        self.assertEqual(dd["july_role"], "locked_holdout_evaluation")
        self.assertTrue(dd["june_development_evidence_used_for_manual_design"])
        self.assertFalse(dd["runtime_candidate_membership_uses_target_columns"])
        self.assertFalse(dd["automatic_target_driven_selection"])
        self.assertFalse(dd["automatic_feature_selection"])
        self.assertFalse(dd["model_training_performed"])
        self.assertFalse(dd["prediction_generated"])
        self.assertEqual(
            dd["candidate_selection_mode"],
            "human_fixed_after_june_development_review")

    def test_specs_frozen_before_july_label_access(self):
        payload = _load_payload()
        vw = compute_vwap_reparameterization(payload["snapshot"])
        dd = build_design_disclosure(vw)
        self.assertTrue(dd["candidate_specs_frozen_before_july_label_access"])
        self.assertTrue(
            dd["runtime_membership_invariant_to_target_column_mutation"])

    def test_full_run_manifest_and_disclosure_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            manifest = result["manifest"]
            self.assertEqual(
                manifest["candidate_selection_mode"],
                "human_fixed_after_june_development_review")
            self.assertTrue(
                manifest["june_development_evidence_used_for_manual_design"])
            self.assertTrue(
                manifest["candidate_specs_frozen_before_july_label_access"])
            self.assertFalse(
                manifest["runtime_candidate_membership_uses_target_columns"])
            self.assertFalse(manifest["automatic_target_driven_selection"])
            self.assertNotIn("candidate_membership_invariant_to_target_values",
                             manifest)
            self.assertNotIn("human_predeclared_fixed",
                             json.dumps(manifest, ensure_ascii=False))
            disclosure = json.loads(
                (out / "v004c_stage2_3_design_disclosure_v001.json").read_text(
                    encoding="utf-8"))
            for key in ("june_role", "july_role",
                        "june_development_evidence_used_for_manual_design",
                        "candidate_specs_frozen_before_july_label_access",
                        "runtime_candidate_membership_uses_target_columns",
                        "automatic_target_driven_selection",
                        "vwap_reparameterization"):
                self.assertIn(key, disclosure)
            self.assertFalse(disclosure["model_training_performed"])
            self.assertFalse(disclosure["prediction_generated"])

    def test_design_disclosure_has_no_july_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            text = (out / "v004c_stage2_3_design_disclosure_v001.json").read_text(
                encoding="utf-8").lower()
            for token in ("july_auc", "july_log_loss", "july_brier",
                          "july_target", "july_accuracy", "locked_target"):
                self.assertNotIn(token, text)

    def test_review_discloses_june_design_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            review = (out / "v004c_stage2_3_review.md").read_text(
                encoding="utf-8")
            for phrase in ("M1、M2和M3由研究人员在阅读六月阶段2.2",
                           "六月是开发与候选设计数据",
                           "候选规格在任何七月标签揭示前冻结",
                           "阶段2.3运行时未读取 Target 列",
                           "未根据 AUC、bootstrap、flags 或其它统计 "
                           "自动搜索、增加、删除或替换候选成员"):
                self.assertIn(phrase, review, f"review 缺少: {phrase}")
            for banned in ("候选完全与Target无关",
                           "候选在六月分析前预注册",
                           "候选从未使用任何标签证据"):
                self.assertNotIn(banned, review)


# ---------------------------------------------------------------------------
# 阶段2.3.2 定向修复测试 (任务 13.x)
# ---------------------------------------------------------------------------
class V004cStage232MissingPropagationTest(unittest.TestCase):
    """13.1/13.2: ma5_overheat_10 源缺失传播 + 冻结数据回归。"""

    def _source_frame(self):
        return pd.DataFrame({
            "signal_date": ["2026-06-01"] * 4,
            "d1_open_to_close_return_raw": [0.10, 0.10, 0.10, 0.10],
            "d1_close_to_ma5_raw": [0.20, np.nan, 0.05, 0.10],
            "break_volume_ratio_vs_board_days": [1.0, 1.0, 1.0, 1.0],
            "volume_above_d1_close_ratio": [0.5, 0.5, 0.5, 0.5],
        })

    def test_ma5_overheat_missing_propagation(self):
        snap = self._source_frame()
        derived = _derive_needed_series(snap)
        m = derived["ma5_overheat_10"]
        # [0.20 → 1, NaN → NaN, 0.05 → 0, 0.10 → 1]
        self.assertEqual(m.isna().tolist(), [False, True, False, False])
        np.testing.assert_allclose(m.dropna().to_numpy(),
                                   np.array([1.0, 0.0, 1.0]))
        # source is NaN → derived is NaN (逐行一致)
        self.assertTrue(
            snap["d1_close_to_ma5_raw"].isna().equals(m.isna()))
        # 非缺失值严格属于 {0.0, 1.0}
        self.assertEqual(sorted(m.dropna().unique().tolist()), [0.0, 1.0])
        # 源缺失行不计入相关性 pair count
        from src.v004c_candidate_model_specs import _spearman_pair
        count, _ = _spearman_pair(m, pd.Series([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(count, 3)

    def test_frozen_data_missing_zero_and_values(self):
        payload = _load_payload()
        snap = payload["snapshot"]
        for col in ("d1_open_to_close_return_raw", "d1_close_to_ma5_raw",
                    "break_volume_ratio_vs_board_days",
                    "volume_above_d1_close_ratio"):
            self.assertEqual(int(snap[col].isna().sum()), 0, col)
        derived = _derive_needed_series(snap)
        m = derived["ma5_overheat_10"]
        self.assertEqual(int(m.isna().sum()), 0)
        self.assertEqual(sorted(m.dropna().unique().tolist()), [0.0, 1.0])

    def test_frozen_data_numeric_regression(self):
        payload = _load_payload()
        red = build_candidate_redundancy_audit(payload["snapshot"])
        self.assertEqual(len(red), 30)
        top = red.loc[red["maximum_absolute_rho"].idxmax()]
        self.assertEqual((top["model_id"], top["feature_a"],
                          top["feature_b"]),
                         ("M3", "overrepair", "ma5_overheat_10"))
        self.assertAlmostEqual(float(top["maximum_absolute_rho"]),
                               0.581433670594, places=9)
        shift = red.loc[red["absolute_rho_shift"].idxmax()]
        self.assertEqual((shift["model_id"], shift["feature_a"],
                          shift["feature_b"]),
                         ("M3", "overrepair", "break_volume_abnormality"))
        self.assertAlmostEqual(float(shift["absolute_rho_shift"]),
                               0.321675541883, places=9)
        self.assertEqual(int((red["redundancy_flag"] ==
                              "ABS_RHO_LT_0_70").sum()), 30)
        vw = compute_vwap_reparameterization(payload["snapshot"])
        self.assertEqual(vw["dev"]["pair_non_null"], 173)
        self.assertAlmostEqual(vw["dev"]["max_abs_sum"],
                               0.008057137084739804, places=12)
        self.assertAlmostEqual(vw["dev"]["pearson_rho"],
                               -0.9991696599734359, places=10)
        self.assertEqual(vw["dev"]["spearman_rho"], -1.0)
        self.assertEqual(vw["locked"]["pair_non_null"], 160)
        self.assertAlmostEqual(vw["locked"]["max_abs_sum"],
                               0.0066904950411876, places=12)
        self.assertAlmostEqual(vw["locked"]["pearson_rho"],
                               -0.9993953570493527, places=10)
        self.assertEqual(vw["locked"]["spearman_rho"], -1.0)
        self.assertEqual(vw["relationship_type"], "NOT_EXACT_NEGATIVE")


class V004cStage232InputAuditTest(unittest.TestCase):
    """13.3: 实际输入文件集合审计。"""

    def _base_checks(self):
        names = sorted(
            f"reports/x/{name}" for group in ALLOWED_INPUT_FILES.values()
            for name in group)
        return {
            "loaded_input_files": set(names),
            "expected_input_files": set(names),
            "forbidden_input_files": {
                "reports/x/v004c_training_d1_v001.csv",
                "reports/x/v004c_d1_existing_model_audit_v001.csv",
            },
            "snapshot_read_columns": list(SNAPSHOT_READ_COLUMNS),
            "snapshot_label_columns_not_read": True,
        }

    def _audit(self, checks):
        payload = _load_payload()
        catalog = build_model_catalog()
        fspec, _ = build_model_feature_spec(payload)
        boundary = build_candidate_boundary(payload)
        evidence = build_selected_feature_evidence(payload)
        return build_holdout_lock_audit(
            catalog=catalog, feature_spec=fspec, boundary=boundary,
            evidence=evidence, input_checks=checks,
            allowlist=_allowlist_lookup(payload))

    def test_actual_loaded_matches_allowed(self):
        audit = self._audit(self._base_checks())
        for name in ("actual_loaded_input_files_match_allowed",
                     "actual_loaded_input_files_disjoint_forbidden",
                     "stage2_3_not_read_training_d1_table",
                     "stage2_3_not_read_existing_model_audit"):
            row = audit[audit["check_name"] == name].iloc[0]
            self.assertEqual(row["status"], "PASS", name)

    def test_extra_forbidden_file_fails_audit(self):
        checks = self._base_checks()
        checks["loaded_input_files"] |= {
            "reports/x/v004c_training_d1_v001.csv"}
        audit = self._audit(checks)
        row = audit[audit["check_name"] ==
                    "actual_loaded_input_files_disjoint_forbidden"].iloc[0]
        self.assertEqual(row["status"], "FAIL")

    def test_missing_required_file_fails_audit(self):
        checks = self._base_checks()
        one = sorted(checks["loaded_input_files"])[0]
        checks["loaded_input_files"] = checks["loaded_input_files"] - {one}
        audit = self._audit(checks)
        row = audit[audit["check_name"] ==
                    "actual_loaded_input_files_match_allowed"].iloc[0]
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("少读", row["details"])

    def test_real_run_records_19_loaded_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            manifest = result["manifest"]
            self.assertIn("actual_loaded_input_file_count",
                          manifest["input_verification"])
            self.assertEqual(
                manifest["input_verification"]["actual_loaded_input_file_count"],
                19)


class V004cStage232MemberStructureAuditTest(unittest.TestCase):
    """13.4: 固定成员结构级比较 (5.5)。"""

    @classmethod
    def setUpClass(cls):
        cls.payload = _load_payload()
        cls.catalog = build_model_catalog()
        cls.fspec, _ = build_model_feature_spec(cls.payload)
        cls.boundary = build_candidate_boundary(cls.payload)
        cls.evidence = build_selected_feature_evidence(cls.payload)
        cls.allowlist = _allowlist_lookup(cls.payload)
        cls.checks = {
            "loaded_input_files": set(),
            "expected_input_files": set(),
            "forbidden_input_files": set(),
            "snapshot_read_columns": list(SNAPSHOT_READ_COLUMNS),
            "snapshot_label_columns_not_read": True,
        }

    def _audit(self, catalog=None, fspec=None):
        return build_holdout_lock_audit(
            catalog=catalog if catalog is not None else self.catalog,
            feature_spec=fspec if fspec is not None else self.fspec,
            boundary=self.boundary, evidence=self.evidence,
            input_checks=self.checks, allowlist=self.allowlist)

    def test_base_structure_matches(self):
        audit = self._audit()
        row = audit[audit["check_name"] ==
                    "candidate_membership_fixed_after_june_development_review"
                    ].iloc[0]
        self.assertEqual(row["status"], "PASS")

    def test_swapped_feature_order_fails(self):
        fspec = self.fspec.copy()
        idx1 = fspec[(fspec["model_id"] == "M1") &
                     (fspec["position"] == 1)].index[0]
        idx2 = fspec[(fspec["model_id"] == "M1") &
                     (fspec["position"] == 2)].index[0]
        fspec.loc[idx1, "position"] = 2
        fspec.loc[idx2, "position"] = 1
        audit = self._audit(fspec=fspec)
        row = audit[audit["check_name"] ==
                    "candidate_membership_fixed_after_june_development_review"
                    ].iloc[0]
        self.assertEqual(row["status"], "FAIL")

    def test_modified_direction_fails(self):
        fspec = self.fspec.copy()
        fspec.loc[fspec["feature_name"] == "break_open_return",
                  "expected_direction"] = "-"
        audit = self._audit(fspec=fspec)
        row = audit[audit["check_name"] ==
                    "candidate_membership_fixed_after_june_development_review"
                    ].iloc[0]
        self.assertEqual(row["status"], "FAIL")

    def test_deleted_model_fails(self):
        catalog = self.catalog[self.catalog["model_id"] != "M3"].copy()
        audit = self._audit(catalog=catalog)
        row = audit[audit["check_name"] ==
                    "candidate_membership_fixed_after_june_development_review"
                    ].iloc[0]
        self.assertEqual(row["status"], "FAIL")

    def test_added_model_fails(self):
        catalog = self.catalog.copy()
        extra = self.catalog.iloc[0].copy()
        extra["model_id"] = "M4"
        catalog = pd.concat([catalog, extra.to_frame().T], ignore_index=True)
        audit = self._audit(catalog=catalog)
        row = audit[audit["check_name"] ==
                    "candidate_membership_fixed_after_june_development_review"
                    ].iloc[0]
        self.assertEqual(row["status"], "FAIL")


class V004cStage232WordingTest(unittest.TestCase):
    """13.5: 候选选择措辞清理。"""

    def test_outputs_clean_of_predeclared_wording(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _run_specs(STAGE1_DIR, STAGE2_1_DIR, STAGE2_2_DIR, out)
            texts = []
            for name in ("v004c_stage2_3_manifest.json",
                         "v004c_stage2_3_review.md",
                         "v004c_stage2_3_candidate_boundary_v001.csv",
                         "v004c_stage2_3_holdout_lock_audit_v001.csv",
                         "v004c_stage2_3_candidate_specs_v001.json",
                         "v004c_stage2_3_design_disclosure_v001.json"):
                texts.append((out / name).read_text(encoding="utf-8"))
            joined = "\n".join(texts)
            for banned in ("human_predeclared_fixed",
                           "candidate_membership_fixed_human_predeclared",
                           "人工预声明固定成员", "模型成员固定人工预声明"):
                self.assertNotIn(banned, joined, banned)
            # 合法阶段2.1 派生因子定义必须保留
            self.assertIn("PREDECLARED_DERIVED", joined)
            self.assertIn("v004c_predeclared_derived_factor_spec_v001.csv",
                          joined)


class V004cStage232FailureIsolationTest(unittest.TestCase):
    """13.7: 失败路径隔离 (8.x)。"""

    def test_failure_does_not_pollute_formal_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = base / "s1"
            s21 = base / "s21"
            s22 = base / "s22"
            shutil.copytree(STAGE1_DIR, s1)
            shutil.copytree(STAGE2_1_DIR, s21)
            shutil.copytree(STAGE2_2_DIR, s22)
            bad = s22 / "v004c_stage2_2_univariate_summary_dev_v001.csv"
            bad.write_text(bad.read_text(encoding="utf-8-sig") + "x",
                           encoding="utf-8-sig")
            out = base / "formal_out"
            failure_dir = base / "failure_audits"
            with _fake_env(s1, s21, s22) as prov:
                with self.assertRaises(DatasetValidationError):
                    run_v004c_stage2_3_candidate_specs(
                        stage1_dir=s1, stage2_1_dir=s21, stage2_2_dir=s22,
                        stage2_2_ref=EXPECTED_STAGE2_2_TAG, output_dir=out,
                        git_provenance_before=prov,
                        git_provenance_after=prov,
                        generated_at=FIXED_GENERATED_AT,
                        failure_audit_dir=failure_dir)
            # 正式输出目录不存在 (未被污染)
            self.assertFalse(out.exists())
            # 临时目录被清理
            self.assertFalse(list(base.glob("formal_out.tmp_*")))
            # 外部 failure audit 存在且内容正确
            audits = sorted(failure_dir.glob("v004c_stage2_3_failure_*.json"))
            self.assertEqual(len(audits), 1)
            data = json.loads(audits[0].read_text(encoding="utf-8"))
            self.assertEqual(data["stage"], "v004c_stage2_3")
            self.assertFalse(data["formal_output_directory_created"])
            self.assertEqual(data["exception_type"],
                             "DatasetValidationError")
            self.assertEqual(data["requested_output_dir"], str(out))
            self.assertEqual(data["git_head"], "a" * 40)
            self.assertEqual(data["branch"], EXPECTED_BRANCH)
            self.assertIn("input_version_anchors", data)
            self.assertIn("stage2_2_commit", data["input_version_anchors"])


class V004cStage232MetricProtocolTest(unittest.TestCase):
    """13.8: 指标协议完整性 (9.x)。"""

    @classmethod
    def setUpClass(cls):
        cls.proto = build_evaluation_protocol_json()
        cls.md = cls.proto["metric_definitions"]

    def test_metric_definitions_complete(self):
        md = self.md
        self.assertEqual(md["roc_auc"]["implementation"],
                         "sklearn.metrics.roc_auc_score")
        self.assertEqual(md["roc_auc"]["positive_label"], 1)
        self.assertIn("METRIC_UNDEFINED", md["roc_auc"]["label_requirement"])
        self.assertEqual(md["pr_auc"]["implementation"],
                         "sklearn.metrics.average_precision_score")
        self.assertEqual(md["pr_auc"]["metric_name"], "average_precision")
        self.assertIn("梯形", md["pr_auc"]["note"])
        self.assertIn("METRIC_UNDEFINED", md["pr_auc"]["no_positive_rule"])
        self.assertEqual(md["brier"]["formula"], "mean((y - p)^2)")
        self.assertIn("[0,1]", md["brier"]["prediction_requirement"])
        self.assertEqual(md["log_loss"]["epsilon"], 1e-15)
        self.assertIn("不得修改保存的原始预测概率",
                      md["log_loss"]["clip_scope"])

    def test_calibration_definitions(self):
        cal = self.md["calibration"]
        self.assertEqual(cal["implementation"], "statsmodels GLM Binomial")
        self.assertEqual(cal["maxiter"], 100)
        self.assertEqual(cal["tol"], 1e-10)
        self.assertEqual(cal["penalty"], "none")
        self.assertEqual(cal["calibration_intercept"], "alpha")
        self.assertEqual(cal["calibration_slope"], "beta")
        self.assertIn("1e-12", " ".join(cal["undefined_rules"]))
        self.assertEqual(cal["m0"]["calibration_slope"], "NOT_APPLICABLE")
        self.assertIn("calibration-in-the-large",
                      cal["m0"]["calibration_intercept_rule"])
        self.assertIn("METRIC_UNDEFINED",
                      cal["m0"]["single_class_rule"])
        self.assertIn("descriptive only", cal["descriptive_only"])
        self.assertIn("不参与候选比较算法", cal["descriptive_only"])

    def test_metric_status_values(self):
        self.assertEqual(self.md["metric_status_values"],
                         ["OK", "METRIC_UNDEFINED", "NOT_APPLICABLE",
                          "INPUT_VALIDATION_FAILURE", "NUMERICAL_FAILURE"])
        self.assertIn("0.5", self.md["no_placeholder_rule"])

    def test_design_disclosure_metric_summary(self):
        payload = _load_payload()
        vw = compute_vwap_reparameterization(payload["snapshot"])
        dd = build_design_disclosure(vw)
        m = dd["evaluation_metric_definitions"]
        self.assertEqual(m["log_loss_probability_clip_epsilon"], 1e-15)
        self.assertEqual(m["m0_calibration_slope"], "NOT_APPLICABLE")
        self.assertEqual(m["roc_auc_implementation"],
                         "sklearn.metrics.roc_auc_score")
        self.assertTrue(m["calibration_descriptive_only"])


class V004cStage232AstGateTest(unittest.TestCase):
    """13.9: AST 静态门 (11.x)。"""

    def test_string_constants_not_false_positive(self):
        src = ('x = "statsmodels GLM Binomial"\n'
               'y = "sklearn.metrics.roc_auc_score"\n'
               'z = "LogisticRegression("\n'
               'w = "cross_val_score("\n'
               "print(x, y, z, w)")
        found = scan_source_for_forbidden_calls(src)
        self.assertEqual(found, {"imports": [], "calls": []})

    def test_real_fit_call_detected(self):
        src = ("import statsmodels.api as sm\n"
               "m = LogisticRegression()\n"
               "m.fit(X, y)\n"
               "m.predict_proba(X)\n"
               "cross_val_score(est, X, y)")
        found = scan_source_for_forbidden_calls(src)
        self.assertIn("statsmodels.api", found["imports"])
        for tok in (".fit(", ".predict_proba(", "LogisticRegression(",
                    "cross_val_score("):
            self.assertIn(tok, found["calls"], tok)

    def test_module_ast_clean(self):
        from src.v004c_candidate_model_specs import _module_ast_scan_checks
        for check in _module_ast_scan_checks():
            self.assertEqual(check["status"], "PASS", check["check_name"])


if __name__ == "__main__":
    unittest.main()

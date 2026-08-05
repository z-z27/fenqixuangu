# -*- coding: utf-8 -*-
"""v004c 阶段2.2: 单因子结构与六月开发集稳定性审计 — 测试。

覆盖 (任务二十):
- 版本与输入 SHA / 333/197/42 / 173 六月 / 59 正样本 / 160 七月锁定 /
  38 primary / 21 sensitivity / 59 因子唯一 / 禁止字段 / 输出目录门禁;
- 6 个派生公式重算 / 无新派生因子;
- 连续 AUC / 二元风险差 / Haldane 优势比 / bucket Cramér's V / 四分位塌缩 /
  单一类别 AUC 不可计算;
- bootstrap 固定 seed 可复现 / 按日期 cluster / 低有效 replicate flag;
- LODO 删除整日 / 二板三板支持门 / 缺失结构 / 分布漂移;
- 七月标签 5 种扰动输出不变 / 六月标签扰动改变关联输出 / 七月标签 NaN 可运行;
- 输出无 Top/Rank/Select / review 无 Top / manifest 未训练模型 / manifest SHA 完整;
- ResourceWarning 由运行参数 -W error::ResourceWarning 兜底。
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.v004c_d1_dataset import DatasetValidationError, GitProvenance
from src.v004c_univariate_stability import (
    EXPECTED_BRANCH,
    EXPECTED_STAGE1_DATA_COMMIT,
    EXPECTED_STAGE2_1_DATA_COMMIT,
    EXPECTED_STAGE2_1_TAG,
    DERIVED_FEATURE_NAMES,
    build_feature_table,
    compute_board_subgroups,
    compute_cluster_bootstrap,
    compute_lodo,
    compute_missingness,
    compute_univariate_summary,
    load_inputs,
    materialize_derived,
    run_v004c_stage2_2_analysis,
    _auc,
    _bootstrap_draw_matrix,
    _binary_metrics,
    _bucket_level_rows,
    _cramers_v,
    _continuous_metrics,
    _quartile_stats,
)

REPO_ROOT = Path.cwd()
STAGE1_DIR = REPO_ROOT / "reports/research/v004c_d1_dataset_v001_20260601_20260729"
STAGE2_1_DIR = REPO_ROOT / "reports/research/v004c_factor_dictionary_v001_20260601_20260729"

FIXED_GENERATED_AT = "2026-08-05T00:00:00+0800"

# 与研究输出比较时排除 audit CSV (各次运行自身的审计工件)
_COMPARE_FILES = (
    "v004c_stage2_2_feature_structure_v001.csv",
    "v004c_stage2_2_univariate_summary_dev_v001.csv",
    "v004c_stage2_2_level_bin_tables_dev_v001.csv",
    "v004c_stage2_2_cluster_bootstrap_dev_v001.csv",
    "v004c_stage2_2_lodo_date_stability_v001.csv",
    "v004c_stage2_2_board_subgroup_dev_v001.csv",
    "v004c_stage2_2_missingness_dev_v001.csv",
    "v004c_stage2_2_distribution_shift_unlabeled_v001.csv",
    "v004c_stage2_2_mechanism_summary_v001.csv",
    "v004c_stage2_2_manifest.json",
    "v004c_stage2_2_review.md",
    "git_head.txt",
    "git_status_before.txt",
    "git_status_after.txt",
)
_ASSOCIATION_OUTPUTS = (
    "v004c_stage2_2_univariate_summary_dev_v001.csv",
    "v004c_stage2_2_level_bin_tables_dev_v001.csv",
    "v004c_stage2_2_cluster_bootstrap_dev_v001.csv",
    "v004c_stage2_2_lodo_date_stability_v001.csv",
    "v004c_stage2_2_board_subgroup_dev_v001.csv",
    "v004c_stage2_2_missingness_dev_v001.csv",
)


def _fake_git_calls(args, cwd):
    """mock _run_git: 分支 / 两个 tag 目标 / 祖先 / autocrlf / 版本。"""
    if args[1] == "rev-parse":
        ref = args[-1]
        if "v004c-d1-dataset-0.1" in ref:
            return EXPECTED_STAGE1_DATA_COMMIT + "\n"
        if "v004c-factor-dictionary-0.1" in ref:
            return EXPECTED_STAGE2_1_DATA_COMMIT + "\n"
        # 未知 ref → 模拟 git rev-parse --verify 失败
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
def _fake_env(stage1_dir: Path, stage2_1_dir: Path):
    """模拟 git: 两个输入目录须同父 (真实目录共用 reports/research;
    临时拷贝共用临时父目录), git_root = 共同父目录。"""
    common = Path(stage1_dir).parent
    assert Path(stage2_1_dir).parent == common, "两个输入目录必须同父"
    prov = GitProvenance(git_root=common, git_head="a" * 40,
                         git_status="", git_dirty=False)
    with mock.patch("src.v004c_univariate_stability._run_git",
                    side_effect=_fake_git_calls):
        yield prov


def _run_analysis(stage1_dir: Path, stage2_1_dir: Path, out_dir: Path,
                  *, audit: bool = False, snapshot_override: pd.DataFrame | None = None,
                  generated_at: str = FIXED_GENERATED_AT,
                  enforce_input_sha: bool = True, **kwargs) -> dict:
    """带 fake git 的完整构建 (默认关审计以控制测试耗时)。"""
    with _fake_env(stage1_dir, stage2_1_dir) as prov:
        return run_v004c_stage2_2_analysis(
            stage1_dir=stage1_dir, stage2_1_dir=stage2_1_dir,
            stage2_1_ref=kwargs.pop("stage2_1_ref", EXPECTED_STAGE2_1_TAG),
            output_dir=out_dir,
            git_provenance_before=prov, git_provenance_after=prov,
            audit_holdout_lock=audit, snapshot_override=snapshot_override,
            generated_at=generated_at, enforce_input_sha=enforce_input_sha,
            **kwargs)


def _assert_files_identical(base_dir: Path, alt_dir: Path,
                            names: tuple[str, ...] = _COMPARE_FILES) -> None:
    for name in names:
        a = (base_dir / name).read_bytes()
        b = (alt_dir / name).read_bytes()
        if name == "v004c_stage2_2_manifest.json":
            # 任务允许 "测试专用临时路径" 不同: 归一化 output_files path 字段
            a = _normalize_manifest_paths(a)
            b = _normalize_manifest_paths(b)
        if a != b:
            raise AssertionError(f"输出文件不一致: {name}")


def _normalize_manifest_paths(raw: bytes) -> bytes:
    import json
    m = json.loads(raw.decode("utf-8"))
    for rec in m.get("output_files", {}).values():
        rec["path"] = "<OUT>/" + str(rec.get("path", "")).split("/")[-1]
    return json.dumps(m, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _load_snapshot() -> pd.DataFrame:
    return pd.read_csv(STAGE1_DIR / "v004c_d1_snapshot_v001.csv",
                       dtype={"code": str}, float_precision="round_trip")


def _load_payload() -> dict:
    _, _, _, _, payload = load_inputs(STAGE1_DIR, STAGE2_1_DIR)
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 指标 fixtures (不构建完整流水线)
# ---------------------------------------------------------------------------
class V004cStage22MetricTest(unittest.TestCase):
    def test_auc_continuous_fixture(self):
        # 全部正样本值 > 全部负样本值 → AUC = 1.0
        auc = _auc(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                   np.array([0, 0, 1, 1, 1]))
        self.assertEqual(auc, 1.0)
        # 交替 → AUC = 0.5
        auc = _auc(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 1]))
        self.assertAlmostEqual(auc, 0.5)
        # 完全反序 → AUC = 0.0
        auc = _auc(np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                   np.array([0, 0, 1, 1, 1]))
        self.assertEqual(auc, 0.0)

    def test_auc_single_class_uncomputable(self):
        self.assertIsNone(_auc(np.array([1.0, 2.0, 3.0]),
                               np.array([0, 0, 0])))
        self.assertIsNone(_auc(np.array([1.0, 2.0, 3.0]),
                               np.array([1, 1, 1])))
        m = _continuous_metrics(np.array([1.0, 2.0, 3.0, 4.0]),
                                np.array([0, 0, 0, 0]))
        self.assertFalse(m["dev_effect_valid"])
        self.assertIsNone(m["signed_auc_raw"])
        self.assertIsNone(m["auc_minus_half"])

    def test_auc_missing_values_excluded(self):
        # rankdata 会传播 NaN → 缺失行必须在 AUC 前剔除
        v = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        t = np.array([1, 0, 1, 0, 1])
        auc = _auc(v, t)
        self.assertIsNotNone(auc)
        clean = _auc(np.array([1.0, 2.0, 3.0, 4.0]),
                     np.array([1, 0, 0, 1]))
        self.assertEqual(auc, clean)

    def test_binary_risk_difference_fixture(self):
        # value=1: 10 正 / 5 负 → rate 2/3; value=0: 4 正 / 8 负 → rate 1/3
        v = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        t = np.array([1] * 10 + [0] * 5 + [1] * 4 + [0] * 8)
        m = _binary_metrics(v.astype(float), t.astype(int))
        self.assertEqual(m["value_1_count"], 15)
        self.assertEqual(m["value_0_count"], 12)
        self.assertEqual(m["value_1_positive_count"], 10)
        self.assertEqual(m["value_0_positive_count"], 4)
        self.assertAlmostEqual(m["target_rate_1"], 10 / 15)
        self.assertAlmostEqual(m["target_rate_0"], 4 / 12)
        self.assertAlmostEqual(m["risk_difference_1_minus_0"], 10 / 15 - 4 / 12)
        self.assertTrue(m["dev_effect_valid"])

    def test_haldane_log_odds_ratio_fixture(self):
        # 零格: value1: 5 正/0 负; value0: 3 正/7 负 → Haldane 每格 +0.5
        v = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        t = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        m = _binary_metrics(v.astype(float), t.astype(int))
        expected = np.log((5.5 * 7.5) / (0.5 * 3.5))
        self.assertAlmostEqual(m["log_odds_ratio_haldane"], float(expected))

    def test_bucket_cramers_v_fixture(self):
        # 2×2: A 10 正/10 负; B 2 正/18 负 → 手工 chi2
        v = np.array(["A"] * 20 + ["B"] * 20)
        t = np.array([1] * 10 + [0] * 10 + [1] * 2 + [0] * 18)
        rows, summary = _bucket_level_rows(v, t)
        self.assertEqual(summary["level_count"], 2)
        self.assertAlmostEqual(summary["min_level_target_rate"], 2 / 20)
        self.assertAlmostEqual(summary["max_level_target_rate"], 10 / 20)
        self.assertAlmostEqual(summary["max_minus_min_target_rate"], 0.4)
        n = 40
        chi2 = (n * (10 * 18 - 2 * 10) ** 2
                / ((10 + 10) * (2 + 18) * (10 + 2) * (10 + 18)))
        expected_v = np.sqrt(chi2 / (n * 1))
        self.assertAlmostEqual(summary["cramers_v"], float(expected_v))
        self.assertEqual(_cramers_v(v, t), summary["cramers_v"])

    def test_quartile_collapse(self):
        # 只有 2 个唯一值 → 无法形成 4 组 → 记录实际组数 2
        v = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                      1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        t = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
        q = _quartile_stats(v, t)
        self.assertLess(q["quartile_count"], 4)
        self.assertGreaterEqual(q["quartile_count"], 2)
        self.assertIsNotNone(q["q1_n"])
        self.assertIsNotNone(q["q4_n"])

    def test_quartile_boundaries_do_not_use_target(self):
        # 相同的值分布 → 相同的分位组数 (标签只影响组内 target rate)
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0])
        t1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        t2 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        q1 = _quartile_stats(v, t1)
        q2 = _quartile_stats(v, t2)
        self.assertEqual(q1["quartile_count"], q2["quartile_count"])
        self.assertEqual(q1["q1_n"], q2["q1_n"])
        self.assertEqual(q1["q4_n"], q2["q4_n"])
        self.assertNotEqual(q1["q1_target_rate"], q2["q1_target_rate"])


# ---------------------------------------------------------------------------
# bootstrap / LODO / 子组 / 缺失 / 漂移 fixtures
# ---------------------------------------------------------------------------
def _small_dev_frame() -> pd.DataFrame:
    """3 个日期 × 每日期 3 行; 特征值按日期唯一可追踪。"""
    dates = ["2026-06-01", "2026-06-02", "2026-06-03"]
    rows = []
    for i, d in enumerate(dates):
        for j in range(3):
            rows.append({
                "signal_date": d,
                "target7_daily_d2open_d3high": 1 if (i + j) % 2 else 0,
                "feat": 1.0 * (i * 3 + j + 1),
            })
    return pd.DataFrame(rows)


def _one_feature_table(analysis_type: str = "CONTINUOUS") -> pd.DataFrame:
    return pd.DataFrame([{
        "feature_name": "feat", "source_or_derived": "source",
        "source_column": "feat", "allowlist_tier": "PRIMARY",
        "mechanism_group": "M", "preprocess_policy": "FOLD_CLIP_Z",
        "mutual_exclusion_group_id": "", "analysis_type": analysis_type,
    }])


class V004cStage22BootstrapTest(unittest.TestCase):
    def test_bootstrap_fixed_seed_reproducible(self):
        frame = _small_dev_frame()
        ft = _one_feature_table()
        b1 = compute_cluster_bootstrap(frame, ft, replicates=20)
        b2 = compute_cluster_bootstrap(frame, ft, replicates=20)
        self.assertTrue(b1.equals(b2))
        self.assertEqual(int(b1["bootstrap_requested"].iloc[0]), 20)

    def test_bootstrap_different_seed_changes_matrix(self):
        from src.v004c_univariate_stability import _bootstrap_draw_matrix
        m1 = _bootstrap_draw_matrix(3, 50, seed=20260805)
        m2 = _bootstrap_draw_matrix(3, 50, seed=20260806)
        self.assertEqual(m1.shape, (50, 3))
        self.assertFalse(np.array_equal(m1, m2))

    def test_bootstrap_each_replicate_draws_full_cluster_set(self):
        # 任务 12.1: 每个 replicate 抽取 len(dates) 个 cluster (不是 1 个)
        from src.v004c_univariate_stability import (
            _bootstrap_draw_matrix, _replicate_row_indices,
        )
        frame = _small_dev_frame()
        dates = sorted(frame["signal_date"].unique())
        date_rows = frame.groupby("signal_date").indices
        rows_by_date = [date_rows[d] for d in dates]
        matrix = _bootstrap_draw_matrix(len(dates), replicates=10, seed=20260805)
        self.assertEqual(matrix.shape, (10, len(dates)))
        # 固定抽样结果 [date_1, date_1, date_3]: date_1 全部行出现 2 次,
        # date_3 全部行出现 1 次, date_2 不出现
        draw = np.array([0, 0, 2])
        idx = _replicate_row_indices(draw, rows_by_date)
        self.assertEqual(len(idx), 3 * 3)  # 3 个 cluster × 每日期 3 行
        counts = {d: int(np.isin(idx, rows_by_date[d]).sum())
                  for d in range(3)}
        self.assertEqual(counts[0], 6)   # date_1 全部行出现 2 次
        self.assertEqual(counts[2], 3)   # date_3 全部行出现 1 次
        self.assertEqual(counts[1], 0)   # date_2 不出现
        # 12.2: 重复日期不去重 → 行数 = 每 cluster 行数之和 (含重复)
        self.assertEqual(len(idx), 3 * len(rows_by_date[0]))

    def test_bootstrap_cluster_not_split(self):
        # 任务 12.3: 一个日期中的所有行必须同时出现相同次数 (不拆分 cluster)
        from src.v004c_univariate_stability import (
            _bootstrap_draw_matrix, _replicate_row_indices,
        )
        frame = _small_dev_frame()
        date_rows = frame.groupby("signal_date").indices
        rows_by_date = [date_rows[d] for d in sorted(frame["signal_date"].unique())]
        matrix = _bootstrap_draw_matrix(3, replicates=50, seed=20260805)
        for draw in matrix:
            idx = _replicate_row_indices(draw, rows_by_date)
            for d in range(3):
                present = np.isin(rows_by_date[d], idx).sum()
                self.assertIn(present, (0, len(rows_by_date[d])),
                              f"cluster 被拆分: date {d} 只出现 {present} 行")

    def test_bootstrap_shared_draw_matrix_across_features(self):
        # 任务 12.4: 全部因子收到同一个 replicate × cluster 矩阵, 只生成一次
        from src.v004c_univariate_stability import _bootstrap_draw_matrix
        frame = _small_dev_frame()
        ft = pd.concat([_one_feature_table("CONTINUOUS"),
                        _one_feature_table("BINARY")], ignore_index=True)
        ft.loc[1, "feature_name"] = "feat_bin"
        frame["feat_bin"] = (np.arange(len(frame)) % 2).astype(float)
        calls: list[int] = []
        orig = _bootstrap_draw_matrix

        def spy(n_dates, replicates, seed):
            calls.append((n_dates, replicates, seed))
            return orig(n_dates, replicates, seed)

        with mock.patch("src.v004c_univariate_stability._bootstrap_draw_matrix",
                        side_effect=spy):
            out = compute_cluster_bootstrap(frame, ft, replicates=20)
        self.assertEqual(len(calls), 1, "抽样矩阵必须只生成一次并复用于全部因子")
        self.assertEqual(len(out), 2)

    def test_bootstrap_clusters_by_signal_date_replay(self):
        # 重放同一 1000×21 抽样矩阵的整日抽样, 手工重算效应 → 必须与函数一致
        frame = _small_dev_frame()
        ft = _one_feature_table()
        dates = sorted(frame["signal_date"].unique())
        date_rows = frame.groupby("signal_date").indices
        rows_by_date = [date_rows[d] for d in dates]
        values = frame["feat"].to_numpy(dtype=float)
        target = frame["target7_daily_d2open_d3high"].astype(int).to_numpy()
        out = compute_cluster_bootstrap(frame, ft, replicates=50)
        matrix = _bootstrap_draw_matrix(len(dates), 50, seed=20260805)
        effects = []
        for draw in matrix:
            idx = np.concatenate([rows_by_date[d] for d in draw])
            val = _auc(values[idx], target[idx])
            if val is not None:
                effects.append(val - 0.5)
        eff = np.asarray(effects)
        self.assertEqual(out["bootstrap_valid"].iloc[0], len(eff))
        self.assertAlmostEqual(out["bootstrap_mean"].iloc[0],
                               float(eff.mean()), places=10)
        self.assertAlmostEqual(out["bootstrap_median"].iloc[0],
                               float(np.median(eff)), places=10)
        self.assertAlmostEqual(out["bootstrap_ci_2_5"].iloc[0],
                               float(np.percentile(eff, 2.5)), places=10)

    def test_bootstrap_low_valid_flag(self):
        # 单一类别标签 → 全部 replicate 无效 → bootstrap_valid=0 < 800
        frame = _small_dev_frame()
        frame["target7_daily_d2open_d3high"] = 0
        ft = _one_feature_table()
        out = compute_cluster_bootstrap(frame, ft, replicates=10)
        self.assertEqual(int(out["bootstrap_valid"].iloc[0]), 0)
        self.assertTrue(bool(out["bootstrap_low_valid_replicates"].iloc[0]))

    def test_bootstrap_invalid_reason_counts(self):
        # 任务 12.6: 无效原因分类 + valid + 各类 invalid 之和 = requested
        from src.v004c_univariate_stability import _classify_bootstrap_invalid
        # 单一 Target 类别 (complete-case)
        self.assertEqual(
            _classify_bootstrap_invalid("CONTINUOUS",
                                        np.array([1.0, 2.0, 3.0]),
                                        np.array([0, 0, 0]), None),
            "INVALID_SINGLE_TARGET_CLASS")
        # 单一二元取值
        self.assertEqual(
            _classify_bootstrap_invalid("BINARY",
                                        np.array([1.0, 1.0, 1.0]),
                                        np.array([1, 0, 1]), None),
            "INVALID_SINGLE_FEATURE_LEVEL")
        # 全部缺失 (非空 < 2)
        self.assertEqual(
            _classify_bootstrap_invalid("CONTINUOUS",
                                        np.array([], dtype=float),
                                        np.array([], dtype=int), None),
            "INVALID_INSUFFICIENT_NON_NULL")
        # bucket 区间效应不依赖标签类别 → target 检查不适用, 落到 METRIC_UNDEFINED
        self.assertEqual(
            _classify_bootstrap_invalid("BUCKET",
                                        np.array(["A", "A", "B"]),
                                        np.array([1, 1, 1]), None),
            "INVALID_METRIC_UNDEFINED")
        # 优先级: 单类 target 且单取值 → SINGLE_TARGET_CLASS 先于 SINGLE_FEATURE_LEVEL
        self.assertEqual(
            _classify_bootstrap_invalid("BINARY",
                                        np.array([1.0, 1.0]),
                                        np.array([1, 1]), None),
            "INVALID_SINGLE_TARGET_CLASS")
        # 全流程: 全部缺失的因子 → 全部 replicate 记为 INSUFFICIENT_NON_NULL
        frame = _small_dev_frame()
        frame["feat"] = np.nan
        ft = _one_feature_table()
        out = compute_cluster_bootstrap(frame, ft, replicates=10)
        row = out.iloc[0]
        self.assertEqual(int(row["bootstrap_valid"]), 0)
        self.assertEqual(int(row["invalid_insufficient_non_null"]), 10)
        self.assertEqual(
            int(row["bootstrap_valid"]) + int(row["invalid_single_target_class"])
            + int(row["invalid_single_feature_level"])
            + int(row["invalid_insufficient_non_null"])
            + int(row["invalid_metric_undefined"]),
            int(row["bootstrap_requested"]))

    def test_bootstrap_complete_case_target_single_class(self):
        # 任务 4.1 (MINOR-2): 完整 replicate Target 同时含 0/1, 但连续因子
        # 缺失结构使 complete-case 行 Target 只有一类 → INVALID_SINGLE_TARGET_CLASS
        # (不得分类为 INVALID_METRIC_UNDEFINED)
        frame = _small_dev_frame()
        target = frame["target7_daily_d2open_d3high"].astype(int).to_numpy()
        self.assertEqual(np.unique(target).size, 2)  # 完整 Target 两类
        # 缺失结构: 所有 target=1 行特征为 NaN → complete-case 只剩 target=0
        frame["feat"] = frame["feat"].where(target == 0)
        self.assertGreater(int(frame["feat"].notna().sum()), 2)
        ft = _one_feature_table()  # CONTINUOUS
        out = compute_cluster_bootstrap(frame, ft, replicates=20)
        row = out.iloc[0]
        self.assertEqual(int(row["bootstrap_valid"]), 0)
        self.assertEqual(int(row["invalid_single_target_class"]), 20)
        self.assertEqual(int(row["invalid_metric_undefined"]), 0)
        self.assertEqual(int(row["invalid_single_feature_level"]), 0)
        self.assertEqual(int(row["invalid_insufficient_non_null"]), 0)

    def test_bootstrap_complete_case_valid_control(self):
        # 任务 4.2: complete-case Target 有 0/1 且特征有效 → 效应正常计算
        # (不进入 invalid 分类)
        frame = _small_dev_frame()
        ft = _one_feature_table()
        out = compute_cluster_bootstrap(frame, ft, replicates=20)
        row = out.iloc[0]
        self.assertEqual(int(row["bootstrap_valid"]), 20)
        self.assertIsNotNone(row["bootstrap_mean"])
        for col in ("invalid_single_target_class", "invalid_single_feature_level",
                    "invalid_insufficient_non_null", "invalid_metric_undefined"):
            self.assertEqual(int(row[col]), 0)

    def test_bucket_bootstrap_zero_reference_semantics(self):
        # 任务 12.7: bucket effect_directional=false / zero_reference_applicable=false
        # / bootstrap_ci_crosses_zero 为空 / 无 BOOTSTRAP_CI_CROSSES_ZERO flag
        frame = _small_dev_frame()
        frame["feat"] = ["A", "B", "A", "B", "A", "B", "A", "B", "A"]
        ft = _one_feature_table("BUCKET")
        out = compute_cluster_bootstrap(frame, ft, replicates=50)
        row = out.iloc[0]
        self.assertFalse(bool(row["effect_directional"]))
        self.assertFalse(bool(row["zero_reference_applicable"]))
        self.assertTrue(pd.isna(row["bootstrap_ci_crosses_zero"]))
        # 汇总 flags 不含 BOOTSTRAP_CI_CROSSES_ZERO
        summary = pd.DataFrame([{
            "feature_name": "feat", "allowlist_tier": "SENSITIVITY",
            "source_or_derived": "source", "source_column": "feat",
            "mechanism_group": "M", "preprocess_policy": "BUCKET_SENSITIVITY_ONLY",
            "mutual_exclusion_group_id": "",
            "dev_rows": 9, "dev_non_null": 9, "dev_missing": 0,
            "dev_missing_rate": 0.0, "dev_target_positive": 4,
            "dev_target_negative": 5, "analysis_type": "BUCKET",
            "effect_metric_name": "max_minus_min_target_rate",
            "effect_value": 0.5, "effect_direction": "",
            "dev_effect_valid": True,
            "quartile_count": None, "clip_direction_flip": False,
            "missing_rate_delta": 0.0,
        }])
        from src.v004c_univariate_stability import _collect_flags
        flags = _collect_flags(
            meta=summary.iloc[0], row=summary.iloc[0].to_dict(),
            metrics={}, struct=pd.Series({"total_dominant_rate": 0.5,
                                          "missing_rate_delta": 0.0}),
            boot=row, lodo={}, board={}, shift=pd.Series({"distribution_shift": False}),
            me_alternatives=set())
        self.assertNotIn("BOOTSTRAP_CI_CROSSES_ZERO", flags)
        self.assertNotIn("BOARD_DIRECTION_DISAGREEMENT", flags)


class V004cStage22LodoTest(unittest.TestCase):
    def test_lodo_removes_whole_day(self):
        frame = _small_dev_frame()
        ft = pd.DataFrame([{
            "feature_name": "feat", "source_or_derived": "source",
            "source_column": "feat", "allowlist_tier": "PRIMARY",
            "mechanism_group": "M", "preprocess_policy": "FOLD_CLIP_Z",
            "mutual_exclusion_group_id": "", "analysis_type": "CONTINUOUS",
        }])
        lodo, summary = compute_lodo(frame, ft)
        self.assertEqual(len(lodo), 3)
        self.assertEqual(summary["feat"]["date_count"], 3)
        values = frame["feat"].to_numpy(dtype=float)
        target = frame["target7_daily_d2open_d3high"].astype(int).to_numpy()
        for _, row in lodo.iterrows():
            excluded = row["excluded_signal_date"]
            mask = frame["signal_date"] != excluded
            expected = _auc(values[mask.to_numpy()], target[mask.to_numpy()])
            if expected is None:
                self.assertFalse(row["valid"])
            else:
                self.assertAlmostEqual(row["effect_without_date"],
                                       expected - 0.5, places=10)
        # 删除整日 → 该日行数必然不再参与
        for _, row in lodo.iterrows():
            self.assertNotEqual(row["excluded_signal_date"], "")


class V004cStage22BoardSubgroupTest(unittest.TestCase):
    def test_board_support_gate_uses_complete_case(self):
        # 任务 12.8: board 总行数满足 20, 但因子 complete-case 不足 20
        # → NOT_ENOUGH_SUPPORT; complete-case 满足 20/5/5 → VERIFIED
        rows = []
        # board=2: 30 行但其中 15 行 feat 缺失 → complete-case 15 (<20)
        for i in range(30):
            rows.append({
                "signal_date": "2026-06-01",
                "board_streak_before_break": 2,
                "target7_daily_d2open_d3high": 1 if i % 3 == 0 else 0,
                "feat": np.nan if i < 15 else 1.0 * i,
            })
        # board=3: 28 行全部非空, 9 正 / 19 负 → 满足 20/5/5
        for i in range(28):
            rows.append({
                "signal_date": "2026-06-01",
                "board_streak_before_break": 3,
                "target7_daily_d2open_d3high": 1 if i < 9 else 0,
                "feat": 1.0 * i,
            })
        frame = pd.DataFrame(rows)
        ft = _one_feature_table()
        out, summary = compute_board_subgroups(frame, ft)
        b2 = out[out["board_group"] == 2].iloc[0]
        b3 = out[out["board_group"] == 3].iloc[0]
        # board=2: 总行数 30 但 complete-case 15 → NOT_ENOUGH_SUPPORT
        self.assertEqual(int(b2["subgroup_total_rows"]), 30)
        self.assertEqual(int(b2["subgroup_non_null"]), 15)
        self.assertEqual(int(b2["subgroup_missing"]), 15)
        self.assertEqual(int(b2["subgroup_positive"]),
                         int((np.arange(15, 30) % 3 == 0).sum()))
        self.assertEqual(int(b2["subgroup_negative"]),
                         15 - int((np.arange(15, 30) % 3 == 0).sum()))
        self.assertEqual(b2["status"], "NOT_ENOUGH_SUPPORT")
        # board=3: complete-case 28 → VERIFIED
        self.assertEqual(int(b3["subgroup_total_rows"]), 28)
        self.assertEqual(int(b3["subgroup_non_null"]), 28)
        self.assertEqual(int(b3["subgroup_positive"]), 9)
        self.assertEqual(b3["status"], "VERIFIED")
        self.assertTrue(summary["feat"]["any_insufficient"])
        self.assertEqual(summary["feat"]["agreement_status"], "NOT_BOTH_VERIFIED")

    def test_board_direction_counts(self):
        # 任务 12.9: 2 一致 + 1 相反 + 1 不可判定 → evaluable=3, agree=2, disagree=1
        from src.v004c_univariate_stability import _board_direction_counts
        summary = pd.DataFrame({
            "board_direction_agreement": [True, True, False, None],
        })
        counts = _board_direction_counts(summary)
        self.assertEqual(counts["board_direction_evaluable_count"], 3)
        self.assertEqual(counts["board_direction_agreement_count"], 2)
        self.assertEqual(counts["board_direction_disagreement_count"], 1)
        self.assertEqual(counts["board_direction_not_evaluable_count"], 1)
        # 硬检查失败时 fail closed (非 True/False/NaN 的取值 → 加总不一致)
        bad = pd.DataFrame({"board_direction_agreement": [True, True, False, "x"]})
        with self.assertRaises(DatasetValidationError):
            _board_direction_counts(bad)

    def test_board_near_zero_description(self):
        # 任务 12.10: board2=-0.005, board3=+0.020 → disagreement 仍为 True,
        # one_board_effect_near_zero=True, 不改变 disagreement flag
        from src.v004c_univariate_stability import _near_zero_fields
        nz = _near_zero_fields(-0.005, 0.020, "CONTINUOUS")
        self.assertTrue(bool(nz["one_board_effect_near_zero"]))
        self.assertAlmostEqual(nz["minimum_absolute_board_effect"], 0.005)
        self.assertAlmostEqual(nz["maximum_absolute_board_effect"], 0.020)
        # 非近零异号: near_zero=False
        nz2 = _near_zero_fields(-0.05, 0.020, "CONTINUOUS")
        self.assertFalse(bool(nz2["one_board_effect_near_zero"]))
        # bucket 无方向 → 留空
        nz3 = _near_zero_fields(0.3, 0.1, "BUCKET")
        self.assertTrue(pd.isna(nz3["one_board_effect_near_zero"]))
        # 方向冲突定义不变: -0.005 与 +0.020 异号 → agreement False
        self.assertEqual(((-0.005 > 0) == (0.020 > 0)), False)


class V004cStage22MissingnessTest(unittest.TestCase):
    def test_missingness_counts(self):
        frame = pd.DataFrame({
            "signal_date": ["2026-06-01"] * 10,
            "target7_daily_d2open_d3high": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "feat": [1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        ft = pd.DataFrame([{
            "feature_name": "feat", "source_or_derived": "source",
            "source_column": "feat", "allowlist_tier": "PRIMARY",
            "mechanism_group": "M", "preprocess_policy": "FOLD_CLIP_Z",
            "mutual_exclusion_group_id": "", "analysis_type": "CONTINUOUS",
        }])
        out = compute_missingness(frame, ft)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["missing_count"], 2)
        self.assertEqual(row["present_count"], 8)
        self.assertEqual(row["missing_positive_count"], 0)  # 缺失行标签: 0,0
        self.assertEqual(row["present_positive_count"], 5)
        self.assertAlmostEqual(row["missing_target_rate"], 0.0)
        self.assertAlmostEqual(row["present_target_rate"], 5 / 8)
        self.assertAlmostEqual(row["missing_minus_present_rate"], -5 / 8)


# ---------------------------------------------------------------------------
# 完整构建门禁 (共享一次构建)
# ---------------------------------------------------------------------------
class V004cStage22GateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="v004c_stage22_gate_"))
        cls.out_dir = cls._tmp / "out"
        cls.result = _run_analysis(STAGE1_DIR, STAGE2_1_DIR, cls.out_dir)
        cls.manifest = cls.result["manifest"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_versions_and_git_chain(self):
        m = self.manifest
        self.assertEqual(m["stage2_2_version"], "v004c-stage2-2-univariate-0.1")
        self.assertEqual(m["source_stage2_1_ref"], EXPECTED_STAGE2_1_TAG)
        self.assertEqual(m["source_stage2_1_commit"],
                         EXPECTED_STAGE2_1_DATA_COMMIT)
        self.assertEqual(m["source_stage1_commit"], EXPECTED_STAGE1_DATA_COMMIT)
        self.assertEqual(m["git_branch"], EXPECTED_BRANCH)
        self.assertEqual(m["git_head"], "a" * 40)
        self.assertEqual(m["locked_target_access"], False)

    def test_333_197_42_structure(self):
        iv = self.manifest["input_verification"]
        self.assertEqual(iv["input_rows"], 333)
        self.assertEqual(iv["columns"], 197)
        self.assertEqual(iv["signal_dates"], 42)
        self.assertEqual(iv["lineage_rows"], 197)
        self.assertTrue(iv["input_sha_all_verified"])
        self.assertTrue(iv["stage1_manifest_audit_reference_verified"])

    def test_dev_173_and_59_positives(self):
        iv = self.manifest["input_verification"]
        self.assertEqual(iv["dev_rows"], 173)
        self.assertEqual(iv["dev_target_positive"], 59)
        self.assertEqual(self.manifest["development_target_positive"], 59)

    def test_locked_160(self):
        self.assertEqual(self.manifest["locked_rows"], 160)
        self.assertEqual(self.manifest["input_verification"]["locked_rows"], 160)

    def test_primary_sensitivity_59(self):
        self.assertEqual(self.manifest["primary_raw_count"], 32)
        self.assertEqual(self.manifest["predeclared_derived_count"], 6)
        self.assertEqual(self.manifest["primary_analysis_count"], 38)
        self.assertEqual(self.manifest["sensitivity_count"], 21)
        self.assertEqual(self.manifest["total_analyzed_features"], 59)

    def test_59_features_unique_in_outputs(self):
        summary = pd.read_csv(self.out_dir /
                              "v004c_stage2_2_univariate_summary_dev_v001.csv")
        self.assertEqual(len(summary), 59)
        self.assertEqual(summary["feature_name"].nunique(), 59)
        structure = pd.read_csv(self.out_dir /
                                "v004c_stage2_2_feature_structure_v001.csv")
        self.assertEqual(len(structure), 59)

    def test_analysis_type_counts(self):
        m = self.manifest["analysis_type_counts"]
        self.assertEqual(m["CONTINUOUS"] + m["ORDINAL"] + m["RANK"]
                         + m["BINARY"] + m["BUCKET"], 59)
        self.assertEqual(m["BINARY"], 10)
        self.assertEqual(m["BUCKET"], 2)
        self.assertEqual(m["RANK"], 3)

    def test_forbidden_fields_not_in_features(self):
        summary = pd.read_csv(self.out_dir /
                              "v004c_stage2_2_univariate_summary_dev_v001.csv")
        tokens = ("derive_only", "exclude_", "audit_only", "identifier",
                  "date", "d2_", "d3_", "label", "existing_model_audit",
                  "target7", "tail_loss", "recognition", "v004a", "v002",
                  "v005")
        for name in summary["feature_name"]:
            low = str(name).lower()
            for token in tokens:
                self.assertNotIn(token, low, f"禁止字段进入因子: {name}")

    def test_derived_recompute_matches(self):
        self.assertTrue(self.manifest["derived_factors"]
                        ["recompute_matches_stage2_1_audit"])
        self.assertEqual(self.manifest["derived_factors"]["spec_count"], 6)

    def test_no_new_derived_factors(self):
        summary = pd.read_csv(self.out_dir /
                              "v004c_stage2_2_univariate_summary_dev_v001.csv")
        derived = set(summary.loc[summary["source_or_derived"] == "derived",
                                  "feature_name"])
        self.assertEqual(derived, set(DERIVED_FEATURE_NAMES))

    def test_output_columns_no_top_rank_select(self):
        for name in sorted(p.name for p in self.out_dir.iterdir()
                           if p.name.endswith(".csv")):
            cols = pd.read_csv(self.out_dir / name, nrows=0).columns
            for col in cols:
                low = str(col).lower()
                self.assertNotIn("top", low, f"{name} 含 Top 字段: {col}")
                self.assertNotIn("rank", low, f"{name} 含 Rank 字段: {col}")
                self.assertNotIn("select", low, f"{name} 含 Select 字段: {col}")

    def test_review_no_top_factors(self):
        review = (self.out_dir / "v004c_stage2_2_review.md").read_text(
            encoding="utf-8")
        for word in ("Top", "top", "Rank", "rank", "Select", "select",
                     "前10", "TOP"):
            self.assertNotIn(word, review, f"review 含 {word!r}")
        self.assertIn("七月标签未用于任何因子分析", review)
        self.assertIn("没有训练模型", review)
        self.assertIn("没有运行完整 Walk-forward", review)

    def test_review_shared_matrix_text_uses_feature_count(self):
        # 任务 4.3 (MINOR-1): review 文案按分析因子数生成 (len(feature_table)),
        # 不得写死 replicate 数 1000
        payload = _load_payload()
        ft, _ = build_feature_table(payload)
        self.assertGreaterEqual(len(ft), 2)
        expected = f"{len(ft)} 个因子共用同一抽样矩阵"
        review = (self.out_dir / "v004c_stage2_2_review.md").read_text(
            encoding="utf-8")
        self.assertIn(expected, review)
        self.assertNotIn("1000 个因子共用同一抽样矩阵", review)

    def test_manifest_declares_no_model(self):
        m = self.manifest
        self.assertFalse(m["model_training_performed"])
        self.assertFalse(m["feature_selection_performed"])
        self.assertFalse(m["holdout_label_analysis_performed"])
        self.assertEqual(m["bootstrap_replicates"], 1000)
        self.assertEqual(m["bootstrap_seed"], 20260805)
        self.assertEqual(m["lodo_stats"]["date_count"], 21)

    def test_manifest_sha_complete(self):
        out = self.manifest["output_files"]
        for name in sorted(p.name for p in self.out_dir.iterdir() if p.is_file()):
            if name == "v004c_stage2_2_manifest.json":
                continue
            self.assertIn(name, out, f"manifest 缺少输出记录: {name}")
            rec = out[name]
            self.assertEqual(rec["sha256"], _sha(self.out_dir / name),
                             f"SHA 不一致: {name}")
            self.assertIn("path", rec)
        # 15 个文件 = 14 个记录 (manifest 自身除外)
        self.assertEqual(len(out), 14)

    def test_manifest_input_files_sha(self):
        inputs = self.manifest["input_files"]
        self.assertEqual(len(inputs), 11)
        for name, rec in inputs.items():
            path = (STAGE1_DIR if rec["source_manifest"] == "stage1"
                    else STAGE2_1_DIR) / name
            self.assertEqual(rec["sha256"], _sha(path), f"输入 SHA 不一致: {name}")

    def test_output_dir_gate_nonempty_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            (out / "placeholder.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_analysis(STAGE1_DIR, STAGE2_1_DIR, out)
            self.assertIn("输出目录非空", str(ctx.exception))

    def test_input_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            d1 = tmp / "stage1"
            d21 = tmp / "stage21"
            shutil.copytree(STAGE1_DIR, d1)
            shutil.copytree(STAGE2_1_DIR, d21)
            # 篡改 snapshot 一个值 → SHA 与 manifest 记录不一致 → fail closed
            snap_path = d1 / "v004c_d1_snapshot_v001.csv"
            frame = pd.read_csv(snap_path, dtype={"code": str},
                                float_precision="round_trip")
            frame.loc[0, "d1_close"] = float(frame.loc[0, "d1_close"]) + 0.01
            frame.to_csv(snap_path, index=False, encoding="utf-8-sig")
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_analysis(d1, d21, tmp / "out")
            self.assertIn("input_sha_mismatch", str(ctx.exception))

    def test_source_ref_wrong_tag_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            with self.assertRaises(DatasetValidationError) as ctx:
                _run_analysis(STAGE1_DIR, STAGE2_1_DIR, out,
                              stage2_1_ref="v004c-nonexistent-tag")
            self.assertIn("source_ref_not_found", str(ctx.exception))

    def test_holdout_lock_audit_csv_present(self):
        audit = pd.read_csv(self.out_dir /
                            "v004c_stage2_2_holdout_lock_audit_v001.csv")
        self.assertIn("check_name", audit.columns)
        self.assertIn("status", audit.columns)
        self.assertGreaterEqual(len(audit), 12)
        # 审计 CSV 自身列名干净
        for col in audit.columns:
            self.assertNotIn("july", str(col).lower())


# ---------------------------------------------------------------------------
# 七月锁定不变性 (任务十六)
# ---------------------------------------------------------------------------
class V004cStage22HoldoutLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="v004c_stage22_lock_"))
        cls.base_dir = cls._tmp / "base"
        _run_analysis(STAGE1_DIR, STAGE2_1_DIR, cls.base_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _mutated_build(self, mutation_fn) -> Path:
        out = self._tmp / f"mut_{len(list(self._tmp.glob('mut_*')))}"
        _run_analysis(STAGE1_DIR, STAGE2_1_DIR, out,
                      snapshot_override=mutation_fn(_load_snapshot()))
        return out

    def test_july_target_shuffle_no_output_change(self):
        from src.v004c_univariate_stability import _mutate_july_target_shuffle
        out = self._mutated_build(_mutate_july_target_shuffle)
        _assert_files_identical(self.base_dir, out)

    def test_july_target_all_zero_no_output_change(self):
        from src.v004c_univariate_stability import _mutate_july_target_all_zero
        out = self._mutated_build(_mutate_july_target_all_zero)
        _assert_files_identical(self.base_dir, out)

    def test_july_target_all_one_no_output_change(self):
        from src.v004c_univariate_stability import _mutate_july_target_all_one
        out = self._mutated_build(_mutate_july_target_all_one)
        _assert_files_identical(self.base_dir, out)

    def test_july_tail_loss_mutation_no_output_change(self):
        from src.v004c_univariate_stability import _mutate_july_tail_loss_flip
        out = self._mutated_build(_mutate_july_tail_loss_flip)
        _assert_files_identical(self.base_dir, out)

    def test_july_labels_nan_still_runs(self):
        def nan_mutation(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy(deep=True)
            out["target7_daily_d2open_d3high"] = \
                out["target7_daily_d2open_d3high"].astype(float)
            mask = (out["signal_date"] >= "2026-07-01") \
                & (out["signal_date"] <= "2026-07-29")
            out.loc[mask, "target7_daily_d2open_d3high"] = np.nan
            return out
        out = self._mutated_build(nan_mutation)
        _assert_files_identical(self.base_dir, out)

    def test_june_target_mutation_changes_association_output(self):
        from src.v004c_univariate_stability import _mutate_june_target_shuffle
        out = self._mutated_build(_mutate_june_target_shuffle)
        diffs = []
        for name in _ASSOCIATION_OUTPUTS:
            a = (self.base_dir / name).read_bytes()
            b = (out / name).read_bytes()
            if a != b:
                diffs.append(name)
        self.assertTrue(diffs, "六月标签打乱后关联输出未改变 (fail)")

    def test_holdout_lock_audit_all_pass(self):
        out = self._tmp / "audit_out"
        result = _run_analysis(STAGE1_DIR, STAGE2_1_DIR, out, audit=True)
        audit = result["manifest"]["holdout_lock_audit"]
        self.assertTrue(audit["all_pass"])
        self.assertEqual(audit["pass_count"], audit["check_count"])
        self.assertGreaterEqual(audit["check_count"], 14)


# ---------------------------------------------------------------------------
# 输入表结构函数直接测试 (禁止字段 / 派生门)
# ---------------------------------------------------------------------------
class V004cStage22FeatureTableTest(unittest.TestCase):
    def test_forbidden_feature_name_fails_closed(self):
        payload = _load_payload()
        primary = payload["frames"]["v004c_feature_allowlist_primary_v001.csv"].copy()
        # 替换一个既有 source 行 (保持 38 项数量), 注入禁止字段名
        idx = primary.index[
            (primary["source_or_derived"] == "source")].tolist()[0]
        primary.loc[idx, "feature_name"] = "d2_evil_double"
        payload["frames"]["v004c_feature_allowlist_primary_v001.csv"] = primary
        with self.assertRaises(DatasetValidationError) as ctx:
            build_feature_table(payload)
        self.assertIn("forbidden_feature_name", str(ctx.exception))

    def test_derived_spec_mismatch_fails_closed(self):
        payload = _load_payload()
        primary = payload["frames"]["v004c_feature_allowlist_primary_v001.csv"].copy()
        # 替换一个派生行 (保持 59 唯一), 引入新派生名
        idx = primary.index[
            primary["source_or_derived"] == "derived"].tolist()[0]
        primary.loc[idx, "feature_name"] = "brand_new_derived"
        payload["frames"]["v004c_feature_allowlist_primary_v001.csv"] = primary
        with self.assertRaises(DatasetValidationError) as ctx:
            build_feature_table(payload)
        self.assertIn("derived_spec_mismatch", str(ctx.exception))

    def test_derived_audit_mismatch_fails_closed(self):
        payload = _load_payload()
        audit = payload["frames"][
            "v004c_predeclared_derived_factor_audit_v001.csv"].copy()
        audit.loc[audit["feature_name"] == "overrepair", "unique_count"] = "999"
        payload["frames"]["v004c_predeclared_derived_factor_audit_v001.csv"] = audit
        snap = _load_snapshot()
        lineage = pd.read_csv(STAGE1_DIR / "v004c_d1_column_lineage.csv", dtype=str)
        with self.assertRaises(DatasetValidationError) as ctx:
            materialize_derived(snap, lineage, payload)
        self.assertIn("不一致", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

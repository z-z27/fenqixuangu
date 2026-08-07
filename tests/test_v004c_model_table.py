"""v004c 模型表模块测试 (时序/泄漏/缺失/公式/完整性/原子输出)。"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_model_table import (
    FEATURE_COLUMNS,
    PRIMARY_FEATURE_COLUMNS,
    RECENT_7D_FIELDS,
    SENSITIVITY_FEATURE_COLUMNS,
    ModelTableError,
    V004CModelTableConfig,
    build_v004c_model_table,
    compute_recent_7d_features,
    extract_recent_7d_window,
    materialize_predeclared_derived,
    recent_7d_close_position,
    recent_7d_cumulative_return,
    recent_7d_max_drawdown,
    scan_for_leakage,
    verify_feature_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO_ROOT / "reports/research/v004c_d1_dataset_v001_20260601_20260729"
DICTIONARY_DIR = REPO_ROOT / "reports/research/v004c_factor_dictionary_v001_20260601_20260729"
CACHE_DIR = REPO_ROOT / "data/cache/daily"


class Recent7dFormulaTests(unittest.TestCase):
    def test_cumulative_return(self):
        closes = np.array([100.0, 103.0, 99.0, 105.0, 110.0, 115.0, 120.0])
        self.assertAlmostEqual(recent_7d_cumulative_return(closes), 0.20, places=12)

    def test_cumulative_return_invalid_base(self):
        self.assertIsNone(recent_7d_cumulative_return(
            np.array([0.0, 103.0, 99.0, 105.0, 110.0, 115.0, 120.0])))
        self.assertIsNone(recent_7d_cumulative_return(
            np.array([np.nan, 103.0, 99.0, 105.0, 110.0, 115.0, 120.0])))

    def test_drawdown_case_a_no_same_day_high(self):
        # day1 high=10 low=9; day2 high=15 low=8:
        # day2 只能用 day1 prior peak=10 -> (10-8)/10 = 0.20, 不得 (15-8)/15
        self.assertAlmostEqual(
            recent_7d_max_drawdown(np.array([10.0, 15.0]), np.array([9.0, 8.0])),
            0.20, places=12)

    def test_drawdown_case_b_future_low(self):
        # day1 high=10; day2 high=15; day3 low=9 -> (15-9)/15 = 0.40
        self.assertAlmostEqual(
            recent_7d_max_drawdown(
                np.array([10.0, 15.0, 15.0]), np.array([np.nan, np.nan, 9.0])),
            0.40, places=12)

    def test_drawdown_case_c_rising(self):
        # 一路上涨且后续 low 始终高于 prior peak -> 0.0
        self.assertEqual(
            recent_7d_max_drawdown(
                np.array([10.0, 11.0, 12.0]), np.array([9.5, 10.5, 11.5])),
            0.0)

    def test_close_position(self):
        highs = np.array([11.0, 12.0, 10.0])
        lows = np.array([9.0, 10.0, 8.0])
        # (10.5 - 8) / (12 - 8) = 0.625
        self.assertAlmostEqual(recent_7d_close_position(highs, lows, 10.5), 0.625, places=12)

    def test_close_position_degenerate_range(self):
        # high_7d == low_7d -> NaN (显式缺失), 不得 inf/0/0.5
        self.assertIsNone(recent_7d_close_position(
            np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0]), 10.0))

    def test_window_extraction_insufficient_history(self):
        # 少于 7 行 -> 显式缺失 (None), 不得自动换成 5/6 日
        daily = pd.DataFrame({
            "date": [f"2026-06-{d:02d}" for d in range(1, 7)],
            "open": np.arange(6.0) + 10, "high": np.arange(6.0) + 11,
            "low": np.arange(6.0) + 9, "close": np.arange(6.0) + 10,
        })
        self.assertIsNone(extract_recent_7d_window(daily, "2026-06-06"))

    def test_window_extraction_seven_rows(self):
        daily = pd.DataFrame({
            "date": [f"2026-06-{d:02d}" for d in range(1, 9)],
            "open": np.arange(8.0) + 10, "high": np.arange(8.0) + 11,
            "low": np.arange(8.0) + 9, "close": np.arange(8.0) + 10,
        })
        window = extract_recent_7d_window(daily, "2026-06-08")
        self.assertIsNotNone(window)
        self.assertEqual(len(window), 7)
        self.assertEqual(str(window["date"].iloc[0]), "2026-06-02")

    def test_compute_recent_7d_insufficient_raises(self):
        daily = pd.DataFrame({
            "date": [f"2026-06-{d:02d}" for d in range(1, 5)],
            "open": np.arange(4.0) + 10, "high": np.arange(4.0) + 11,
            "low": np.arange(4.0) + 9, "close": np.arange(4.0) + 10,
        })
        with self.assertRaises(ModelTableError):
            compute_recent_7d_features(daily, "2026-06-04")

    def test_compute_recent_7d_values(self):
        closes = [100.0, 103.0, 99.0, 105.0, 110.0, 115.0, 120.0]
        daily = pd.DataFrame({
            "date": [f"2026-06-{d:02d}" for d in range(1, 8)],
            "open": closes, "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes], "close": closes,
        })
        out = compute_recent_7d_features(daily, "2026-06-07")
        self.assertAlmostEqual(out["recent_7d_cumulative_return"], 0.20, places=12)
        self.assertAlmostEqual(out["recent_7d_close_position"],
                               (120.0 - 98.0) / (121.0 - 98.0), places=12)


class DerivedFeatureTests(unittest.TestCase):
    def test_predeclared_derived_values(self):
        table = pd.DataFrame({
            "board_streak_before_break": [2, 3],
            "d1_close_to_ma5_raw": [0.05, 0.12],
            "d1_open_to_close_return_raw": [0.02, 0.05],
            "break_volume_ratio_vs_board_days": [0.5, 2.0],
            "volume_above_d1_close_ratio": [0.7, 0.3],
            "d1_low_to_close_recovery": [0.01, 0.02],
        })
        out = materialize_predeclared_derived(table)
        self.assertEqual(out["board_streak_is_3"].tolist(), [0, 1])
        self.assertEqual(out["ma5_overheat_10"].tolist(), [0, 1])
        self.assertAlmostEqual(out["overrepair"].iloc[0], 0.0)
        self.assertAlmostEqual(out["overrepair"].iloc[1], 0.02)
        self.assertAlmostEqual(out["break_volume_abnormality"].iloc[1],
                               abs(np.log(2.0)), places=12)
        # profit_pressure = chip * recovery * log1p(ratio)
        chip = 1.0 - 0.3
        expected = chip * 0.02 * np.log1p(2.0)
        self.assertAlmostEqual(out["profit_pressure"].iloc[1], expected, places=12)

    def test_derived_matches_frozen_audit_stats(self):
        # 与冻结的预声明派生审计 (min/max) 交叉验证
        training = pd.read_csv(STAGE1_DIR / "v004c_training_d1_v001.csv", dtype={"code": str})
        out = materialize_predeclared_derived(training)
        audit = pd.read_csv(DICTIONARY_DIR / "v004c_predeclared_derived_factor_audit_v001.csv",
                            encoding="utf-8-sig").set_index("feature_name")
        for name in ("board_streak_is_3", "ma5_overheat_10", "overrepair",
                     "break_volume_abnormality", "profit_pressure"):
            self.assertAlmostEqual(float(out[name].min()), float(audit.loc[name, "min_value"]),
                                   places=6, msg=name)
            self.assertAlmostEqual(float(out[name].max()), float(audit.loc[name, "max_value"]),
                                   places=6, msg=name)
            self.assertEqual(int(out[name].notna().sum()), int(audit.loc[name, "non_null_count"]))


class LeakageTests(unittest.TestCase):
    def test_leakage_scan_rejects_forbidden_columns(self):
        columns = ("d1_close_to_ma5_raw", "target7_daily_d2open_d3high",
                   "d2_open_daily", "d3_high_daily", "v004a_probability", "v002_rank")
        with self.assertRaises(ModelTableError):
            scan_for_leakage(columns)

    def test_leakage_scan_clean_universe(self):
        scan_for_leakage(FEATURE_COLUMNS)  # 不应抛错

    def test_target_only_in_label(self):
        self.assertNotIn("target7_daily_d2open_d3high", FEATURE_COLUMNS)


class ContractTests(unittest.TestCase):
    def test_contract_consistent_with_coverage(self):
        coverage = pd.read_csv(REPO_ROOT / "reports/research/v004c_feature_coverage_v001.csv",
                               encoding="utf-8-sig")
        allowlist = pd.read_csv(DICTIONARY_DIR / "v004c_feature_allowlist_primary_v001.csv",
                                encoding="utf-8-sig")
        training = pd.read_csv(STAGE1_DIR / "v004c_training_d1_v001.csv",
                               dtype={"code": str}, nrows=1)
        verify_feature_contract(coverage, allowlist, set(training.columns))

    def test_primary_sensitivity_disjoint(self):
        self.assertTrue(set(PRIMARY_FEATURE_COLUMNS).isdisjoint(set(SENSITIVITY_FEATURE_COLUMNS)))
        self.assertEqual(len(PRIMARY_FEATURE_COLUMNS), 29)
        self.assertEqual(len(SENSITIVITY_FEATURE_COLUMNS), 26)


class ModelTableIntegrityTests(unittest.TestCase):
    """针对正式 333 行输出的完整性 (用冻结输入构建到 tmp 目录, 与正式输出同源)。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="v004c_model_table_test_")
        cls.output_dir = Path(cls.tmp) / "v004c_model_table_v001_test"
        config = V004CModelTableConfig(
            stage1_dir=STAGE1_DIR,
            dictionary_dir=DICTIONARY_DIR,
            cache_dir=CACHE_DIR,
            output_dir=cls.output_dir,
        )
        cls.checks = build_v004c_model_table(config)
        cls.table = pd.read_csv(cls.output_dir / "v004c_model_table_v001.csv",
                                dtype={"code": str})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_rows_and_dates(self):
        self.assertEqual(len(self.table), 333)
        self.assertEqual(self.table["signal_date"].nunique(), 42)
        self.assertTrue(self.table["event_id"].is_unique)

    def test_new_required_fields_complete_and_finite(self):
        for field in RECENT_7D_FIELDS:
            series = pd.to_numeric(self.table[field], errors="coerce")
            self.assertEqual(int(series.isna().sum()), 0, field)
            self.assertFalse(bool(np.isinf(series).any()), field)

    def test_target_binary(self):
        self.assertTrue(self.table["target7_daily_d2open_d3high"].isin([True, False]).all())

    def test_primary_no_missing(self):
        self.assertEqual(int(self.table[list(PRIMARY_FEATURE_COLUMNS)].isna().sum().sum()), 0)

    def test_no_d2_d3_in_columns(self):
        # 泄漏检查作用于 FEATURE 列 (LABEL target7 属合法 LABEL 角色)
        for column in FEATURE_COLUMNS:
            lowered = column.lower()
            for token in ("d2_", "d3_", "target", "v002", "v004a", "v004b", "v005"):
                self.assertNotIn(token, lowered, column)

    def test_atomic_refuses_existing_dir(self):
        config = V004CModelTableConfig(
            stage1_dir=STAGE1_DIR,
            dictionary_dir=DICTIONARY_DIR,
            cache_dir=CACHE_DIR,
            output_dir=self.output_dir,  # 已存在
        )
        with self.assertRaises(ModelTableError):
            build_v004c_model_table(config)


if __name__ == "__main__":
    unittest.main()

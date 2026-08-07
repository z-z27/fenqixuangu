"""v004c 因子规格模块测试 (target-blind / factor 公式 / reference transform / VIF / condition number / 审计流水线)。

覆盖任务要求:
- A. X loader 返回列中不存在 Target (即使 model table 中存在)
- B. Target 随机化后 factor/correlation/VIF/condition number 输出完全不变 (字节级)
- C. primitive list 不含 target/d2_/d3_/outcome/future/v002/v004a/v004b/v005/recognition/policy
- factor 公式 (OPEN/RESET/SUPPLY/REGIME)
- VIF/condition number 独立与完全复制 (不得静默输出正常值)
- reference transform (June fit / July 只 apply June 参数)
"""

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_factor_spec import (
    CONTINUOUS_FACTORS,
    CONTINUOUS_PRIMITIVES,
    FACTOR_NAMES,
    FACTOR_SPEC_FIELDS,
    FORBIDDEN_TOKENS,
    M1_FACTORS,
    M2_FACTORS,
    PRIMITIVE_COLUMNS,
    FactorSpecError,
    apply_primitive_transform,
    composite_reset,
    composite_supply,
    condition_number_diagnostic,
    construct_factors,
    correlation_severity,
    factor_spec_rows,
    fit_reference_transform,
    ks_2samp_statistic,
    pearson,
    spearman,
    standardized_mean_difference,
    vif_matrix,
)
from tools.v004c_factor_dependence_audit import (
    OUTPUT_FILES,
    load_x_table,
    run_factor_dependence_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE_CSV = (REPO_ROOT
                   / "reports/research/v004c_model_table_v001_20260601_20260729"
                   / "v004c_model_table_v001.csv")


def _df_with(**cols) -> pd.DataFrame:
    """构造包含全部 11 个 primitive 列的 DataFrame (其余列填 0)."""
    n = max(len(v) if isinstance(v, (list, np.ndarray, pd.Series)) else 1
            for v in cols.values())
    data = {p: np.linspace(1.0, 2.0, n) for p in CONTINUOUS_PRIMITIVES}
    data["board_streak_is_3"] = np.zeros(n)
    for k, v in cols.items():
        data[k] = v if isinstance(v, (list, np.ndarray, pd.Series)) else [v] * n
    return pd.DataFrame(data)


def simple_params() -> dict:
    """单位 transform: clip 无效, mu=0 sigma=1 (z = 原值)."""
    params = {p: {"q01": -1e9, "q99": 1e9, "mu": 0.0, "sigma": 1.0}
              for p in CONTINUOUS_PRIMITIVES}
    params["composite_RESET"] = {"mu": 0.0, "sigma": 1.0}
    params["composite_SUPPLY"] = {"mu": 0.0, "sigma": 1.0}
    return params


class XLoaderTests(unittest.TestCase):
    """A: loader target-blind; C: primitive list 无禁止 token."""

    def test_loader_returns_no_target(self):
        df = load_x_table(MODEL_TABLE_CSV)
        expected = ["event_id", "code", "signal_date"] + list(PRIMITIVE_COLUMNS)
        self.assertEqual(df.columns.tolist(), expected)
        self.assertNotIn("target7_daily_d2open_d3high", df.columns)
        self.assertNotIn("break_date", df.columns)
        self.assertNotIn("tail_loss", df.columns)

    def test_loader_requesting_forbidden_column_fails(self):
        for forbidden in ("d2_open_daily", "d3_high_daily", "v004a_probability",
                          "v002_rank", "recognition_score", "target7_daily_d2open_d3high"):
            with self.assertRaises(FactorSpecError, msg=forbidden):
                load_x_table(MODEL_TABLE_CSV, primitive_columns=(forbidden,))

    def test_loader_missing_column_fails(self):
        with self.assertRaises(FactorSpecError):
            load_x_table(MODEL_TABLE_CSV, primitive_columns=("not_a_real_column",))

    def test_primitive_list_has_no_forbidden_tokens(self):
        for column in PRIMITIVE_COLUMNS:
            lowered = column.lower()
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(token, lowered, column)


class ReferenceTransformTests(unittest.TestCase):
    """June fit / July 只 apply June 参数 (41)."""

    def test_fit_uses_june_only_and_is_deterministic(self):
        june = _df_with(d1_ma10_slope=np.linspace(-3.0, 3.0, 200))
        params1 = fit_reference_transform(june)
        params2 = fit_reference_transform(june)
        self.assertEqual(params1, params2)
        # 修改 July 数据不得改变 June reference 参数 (fit 只读 June)
        july_modified = _df_with(d1_ma10_slope=np.linspace(300.0, 400.0, 160))
        _ = construct_factors(july_modified, params1)  # apply June 参数可行
        self.assertEqual(fit_reference_transform(june), params1)

    def test_july_apply_uses_june_params_not_july_own(self):
        # June: [-3, 3] 线性; July: [1, 2] 完全落在 June q01/q99 内 -> 无 clip
        june = _df_with(d1_ma10_slope=np.linspace(-3.0, 3.0, 200))
        params = fit_reference_transform(june)
        sigma_jun = params["d1_ma10_slope"]["sigma"]
        july_vals = np.linspace(1.0, 2.0, 160)
        july = _df_with(d1_ma10_slope=july_vals)
        z = apply_primitive_transform(july, params)
        expected = (july_vals - params["d1_ma10_slope"]["mu"]) / sigma_jun
        np.testing.assert_allclose(z["d1_ma10_slope"], expected, rtol=1e-10)
        # 若错误地用 July 自身 mean/std 标准化, 会得到 mean=0 std=1;
        # 而 June 参数映射后的 mean 明显非 0 -> 证明 July 只用 June 参数
        self.assertGreater(abs(z["d1_ma10_slope"].mean()), 0.5)

    def test_in_sample_reference_standardized(self):
        rng = np.random.default_rng(7)
        june = _df_with(d1_ma10_slope=rng.normal(0.0, 1.0, 500))
        params = fit_reference_transform(june)
        z = apply_primitive_transform(june, params)
        self.assertLess(abs(z["d1_ma10_slope"].mean()), 0.1)
        self.assertGreater(z["d1_ma10_slope"].std(ddof=0), 0.85)
        self.assertLess(z["d1_ma10_slope"].std(ddof=0), 1.15)

    def test_fit_rejects_degenerate_primitive(self):
        june = _df_with(d1_ma10_slope=[1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(FactorSpecError):
            fit_reference_transform(june)

    def test_fit_rejects_empty_june(self):
        june = _df_with(d1_ma10_slope=[1.0, 2.0]).iloc[0:0]
        with self.assertRaises(FactorSpecError):
            fit_reference_transform(june)


class FactorFormulaTests(unittest.TestCase):
    """factor 公式 (39): OPEN / RESET / SUPPLY / REGIME."""

    def test_open_is_z(self):
        params = simple_params()
        params["break_open_return"] = {"q01": -1e9, "q99": 1e9, "mu": 0.5, "sigma": 2.0}
        df = _df_with(break_open_return=[2.5])
        f = construct_factors(df, params)
        self.assertAlmostEqual(f["OPEN"].iloc[0], (2.5 - 0.5) / 2.0, places=12)

    def test_reset_composite_formula(self):
        # z_OC=-1, z_HC=+2, z_VWAP=-0.5 -> G_RESET = (1+2+0.5)/3
        z = pd.DataFrame({
            "d1_open_to_close_return_raw": [-1.0],
            "d1_high_to_close_drawdown_raw": [2.0],
            "d1_close_to_vwap_raw": [-0.5],
        })
        g = composite_reset(z)
        self.assertAlmostEqual(float(g[0]), (1.0 + 2.0 + 0.5) / 3.0, places=12)

    def test_reset_composite_standardization(self):
        params = simple_params()
        params["composite_RESET"] = {"mu": 0.5, "sigma": 2.0}
        df = _df_with(d1_open_to_close_return_raw=[-1.0],
                      d1_high_to_close_drawdown_raw=[2.0],
                      d1_close_to_vwap_raw=[-0.5])
        f = construct_factors(df, params)
        g = (1.0 + 2.0 + 0.5) / 3.0
        self.assertAlmostEqual(f["RESET"].iloc[0], (g - 0.5) / 2.0, places=12)

    def test_supply_composite(self):
        # z_high_zone=1, z_late_sell=3 -> G_SUPPLY = 2; F = z(G) = 2 (mu=0 sigma=1)
        params = simple_params()
        df = _df_with(high_zone_volume_ratio=[1.0], late_day_sell_volume_ratio=[3.0])
        f = construct_factors(df, params)
        self.assertAlmostEqual(float(composite_supply(
            apply_primitive_transform(df, params)).iloc[0]), 2.0, places=12)
        self.assertAlmostEqual(f["SUPPLY"].iloc[0], 2.0, places=12)

    def test_single_primitive_factors_are_z(self):
        params = simple_params()
        params["recent_7d_cumulative_return"] = {"q01": -1e9, "q99": 1e9,
                                                 "mu": 1.0, "sigma": 2.0}
        df = _df_with(recent_7d_cumulative_return=[3.0])
        f = construct_factors(df, params)
        self.assertAlmostEqual(f["MOM7"].iloc[0], (3.0 - 1.0) / 2.0, places=12)

    def test_regime_stays_binary_not_zscored(self):
        params = simple_params()
        df = _df_with(board_streak_is_3=[0.0, 1.0, 1.0, 0.0])
        f = construct_factors(df, params)
        self.assertEqual(f["REGIME"].tolist(), [0.0, 1.0, 1.0, 0.0])
        self.assertEqual(f["REGIME"].min(), 0.0)
        self.assertEqual(f["REGIME"].max(), 1.0)

    def test_regime_invalid_value_fails(self):
        with self.assertRaises(FactorSpecError):
            construct_factors(_df_with(board_streak_is_3=[2.0]), simple_params())

    def test_factor_spec_rows(self):
        rows = factor_spec_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["factor_name"] for r in rows}, set(FACTOR_NAMES))
        for row in rows:
            self.assertEqual(list(row.keys()), list(FACTOR_SPEC_FIELDS))
            self.assertIs(row["target_used_to_construct"], False)
            self.assertEqual(row["m1_member"], row["factor_name"] in M1_FACTORS)
            self.assertEqual(row["m2_member"], row["factor_name"] in M2_FACTORS)
            self.assertIn(row["final_model_preprocessing"],
                          ("0/1 原值, 不做 z-score; 只有 condition number 诊断副本中临时 "
                           "center/scale (diagnostic scaling != future model preprocessing)",
                           "walk-forward 中每个 training fold 内拟合: clip [q01, q99] "
                           "+ (x - mu)/sigma (fold 参数; 禁止全样本参数)"))


class VifConditionNumberTests(unittest.TestCase):
    """VIF / condition number (40): 独立 -> 正常; 完全复制 -> 显式 severe/near-singular."""

    def _matrix(self, duplicate=False):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(500, 3))
        if duplicate:
            x[:, 1] = x[:, 0]
        return pd.DataFrame(x, columns=["F1", "F2", "F3"])

    def test_vif_independent_near_one(self):
        vifs = vif_matrix(self._matrix())
        self.assertLess(max(vifs.values()), 2.0)
        for v in vifs.values():
            self.assertGreater(v, 0.9)

    def test_vif_duplicate_is_severe_not_silent(self):
        vifs = vif_matrix(self._matrix(duplicate=True))
        # 完全复制 -> VIF 必须显式 severe/inf, 不得静默输出正常值
        self.assertGreaterEqual(max(vifs.values()), 1e6)

    def test_vif_insufficient_samples_fails(self):
        small = pd.DataFrame(np.zeros((2, 3)), columns=["A", "B", "C"])
        with self.assertRaises(FactorSpecError):
            vif_matrix(small)

    def test_condition_independent_reasonable(self):
        s_max, s_min, kappa, near = condition_number_diagnostic(
            self._matrix().to_numpy(dtype=float))
        self.assertFalse(near)
        self.assertLess(kappa, 30.0)
        self.assertGreater(s_max, 0.0)

    def test_condition_duplicate_near_singular(self):
        s_max, s_min, kappa, near = condition_number_diagnostic(
            self._matrix(duplicate=True).to_numpy(dtype=float))
        # 完全复制 -> 必须 near singular / kappa severe, 不得静默输出正常值
        self.assertTrue(near)
        self.assertGreaterEqual(kappa, 1e6)

    def test_condition_zero_variance_column(self):
        mat = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
        _, _, kappa, near = condition_number_diagnostic(mat)
        self.assertTrue(near)
        self.assertTrue(np.isinf(kappa) or kappa >= 1e6)


class CorrelationDiagnosticTests(unittest.TestCase):
    def test_severity_bands(self):
        self.assertEqual(correlation_severity(0.49), "LOW")
        self.assertEqual(correlation_severity(0.60), "MODERATE")
        self.assertEqual(correlation_severity(0.70), "HIGH")
        self.assertEqual(correlation_severity(0.80), "HIGH")
        self.assertEqual(correlation_severity(0.85), "SEVERE")
        self.assertEqual(correlation_severity(0.95), "SEVERE")

    def test_pearson_spearman_perfect(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson(a, a * 2.0), 1.0, places=12)
        self.assertAlmostEqual(spearman(a, a * 2.0), 1.0, places=12)
        self.assertAlmostEqual(pearson(a, -a * 2.0), -1.0, places=12)

    def test_correlations_drop_nan_pairs(self):
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(pearson(a, b), 1.0, places=12)
        self.assertAlmostEqual(spearman(a, b), 1.0, places=12)

    def test_ks_statistic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(ks_2samp_statistic(a, a), 0.0, places=12)
        b = np.array([100.0, 200.0, 300.0, 400.0])
        self.assertAlmostEqual(ks_2samp_statistic(a, b), 1.0, places=12)

    def test_smd_formula(self):
        # SMD = (mu_b - mu_a) / sqrt((sigma_b^2 + sigma_a^2)/2)
        self.assertAlmostEqual(standardized_mean_difference(0.0, 1.0, 1.0, 1.0),
                               1.0, places=12)


class AuditPipelineTests(unittest.TestCase):
    """B: Target 随机化后输出完全不变 (字节级); 输出文件结构完整."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="v004c_factor_audit_test_"))
        cls.out1 = cls.tmp / "run1"
        cls.res1 = run_factor_dependence_audit(MODEL_TABLE_CSV, cls.out1)

        cls.out2 = cls.tmp / "run2"
        run_factor_dependence_audit(MODEL_TABLE_CSV, cls.out2)

        # Target 随机化的拷贝
        randomized_csv = cls.tmp / "model_table_randomized_target.csv"
        table = pd.read_csv(MODEL_TABLE_CSV, dtype={"code": str})
        table["target7_daily_d2open_d3high"] = np.random.default_rng(42).choice(
            [True, False], size=len(table))
        table.to_csv(randomized_csv, index=False, encoding="utf-8-sig")
        cls.out3 = cls.tmp / "run3_randomized_target"
        cls.res3 = run_factor_dependence_audit(randomized_csv, cls.out3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_june_july_rows_and_dates(self):
        self.assertEqual(self.res1["june_rows"], 173)
        self.assertEqual(self.res1["june_date_count"], 21)
        self.assertEqual(self.res1["july_rows"], 160)
        self.assertEqual(self.res1["july_date_count"], 21)

    def test_output_has_exactly_six_files(self):
        self.assertEqual(sorted(p.name for p in self.out1.iterdir()),
                         sorted(OUTPUT_FILES))

    def test_no_degenerate_factors(self):
        self.assertEqual(self.res1["degenerate_factors"], [])
        self.assertIn(self.res1["status"], ("PASS_X_STRUCTURE", "REVIEW_REQUIRED"))

    def test_pair_counts(self):
        self.assertEqual(len(self.res1["primitive_pairs"]), 55)
        self.assertEqual(len(self.res1["factor_pairs"]), 28)

    def test_deterministic_same_input(self):
        for name in OUTPUT_FILES:
            self.assertEqual((self.out1 / name).read_bytes(),
                             (self.out2 / name).read_bytes(), name)

    def test_target_randomization_changes_nothing(self):
        # target 随机化后: factor/correlation/VIF/condition number 输出必须完全不变
        # (review md 中只有输入文件 SHA256 一行是文件级 provenance, 允许不同;
        #   该行在 analysis 语义之外)
        analysis = [n for n in OUTPUT_FILES if n != "v004c_factor_dependence_review.md"]
        for name in analysis:
            self.assertEqual((self.out1 / name).read_bytes(),
                             (self.out3 / name).read_bytes(), name)
        md1 = (self.out1 / "v004c_factor_dependence_review.md").read_text(encoding="utf-8")
        md3 = (self.out3 / "v004c_factor_dependence_review.md").read_text(encoding="utf-8")
        norm = lambda s: re.sub(r"SHA256: `[0-9a-f]{64}`", "SHA256: <sha>", s)  # noqa: E731
        self.assertEqual(norm(md1), norm(md3))

    def test_factor_spec_csv_has_eight_rows(self):
        spec = pd.read_csv(self.out1 / "v004c_factor_spec_v001.csv", encoding="utf-8-sig")
        self.assertEqual(len(spec), 8)
        self.assertTrue((spec["target_used_to_construct"] == False).all())  # noqa: E712


if __name__ == "__main__":
    unittest.main()

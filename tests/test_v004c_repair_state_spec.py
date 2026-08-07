"""v004c Repair-State Model v001 — spec + 纯 X 审计测试。

覆盖任务 §37 的 24 项要求:

1  R1 正好 4 factors
2  R2 正好 7 factors
3  DIVERGENCE 公式与旧 RESET 完全一致
4  RECLAIM 严格等于两个标准化 primitive 的等权平均再标准化
5  SUPPLY 只来自 late_day_sell_volume_ratio
6  HIGHZONE 不在 R1/R2
7  MOM7/DAMAGE7/REGIME 不在 R1/R2
8  POS7/TREND 不在 R1/R2
9  derived hierarchy 成立
10 DIVERGENCE_SQ = standardized(D^2)
11 DIVERGENCE_X_RECLAIM = standardized(D*Q)
12 DIVERGENCE_X_SUPPLY = standardized(D*S)
13 derived terms 不做第二次 clip
14 primitive 执行 clip -> clipped mean/std
15 July 不能改变 June primitive params
16 July 不能改变 June composite params
17 July 不能改变 June derived params
18 target column 不在 usecols
19 target 随机化不能改变任何正式输出
20 future/D2/D3 字段无法进入 audit
21 VIF severe 规则
22 condition-number severe 规则
23 RECLAIM_COMPOSITE_REVIEW 规则
24 deterministic rebuild

全部测试确定性 (固定 seed / 显式构造), 不读取真实 Target。
"""

from __future__ import annotations

import filecmp
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.v004c_factor_spec import (
    PRIMITIVE_COLUMNS,
    FactorSpecError,
    condition_number_diagnostic,
    construct_factors,
    fit_reference_transform,
    vif_matrix,
)
from src.v004c_repair_state_spec import (
    BASE_FACTORS,
    DERIVED_FACTORS,
    DERIVED_PARENTS,
    R1_FACTORS,
    R2_FACTORS,
    LEGACY_SENSITIVITY_FACTORS,
    RECLAIM_PRIMITIVES,
    REPAIR_PRIMITIVES,
    RepairStateError,
    audit_status,
    composite_reclaim,
    composite_reset,
    construct_repair_factors,
    fit_repair_transform,
)
from tools import v004c_repair_state_dependence_audit as audit_mod
from tools.v004c_repair_state_dependence_audit import (
    OUTPUT_FILES,
    load_x_table,
    run_repair_state_audit,
)

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"


# ---------------------------------------------------------------------------
# 合成数据生成器 (确定性)
# ---------------------------------------------------------------------------

def synth_df(n: int = 200, seed: int = 11) -> pd.DataFrame:
    """7 个 primitive 的合成 X 数据 (低相关 base + 高相关 RECLAIM 内部对)."""
    rng = np.random.default_rng(seed)
    z_div = rng.normal(size=n)
    z_rec = rng.normal(size=n)
    z_sup = rng.normal(size=n)
    return pd.DataFrame({
        "break_open_return": z_div + rng.normal(scale=0.5, size=n),
        "d1_open_to_close_return_raw": -z_div * 0.8 + rng.normal(scale=0.5, size=n),
        "d1_high_to_close_drawdown_raw": z_div * 0.9 + rng.normal(scale=0.4, size=n),
        "d1_close_to_vwap_raw": -z_div * 0.7 + rng.normal(scale=0.5, size=n),
        "late_day_sell_volume_ratio": z_sup + rng.normal(scale=0.5, size=n),
        "d1_low_to_close_recovery": z_rec + rng.normal(scale=0.6, size=n),
        "d1_afternoon_return": z_rec + rng.normal(scale=0.6, size=n),
    })


def synth_df_weak_reclaim(n: int = 300, seed: int = 21) -> pd.DataFrame:
    """RECLAIM 两 primitive 接近独立的合成 X 数据 (触发 RECLAIM_COMPOSITE_REVIEW)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    b = b - a * (np.dot(a, b) / np.dot(a, a))  # 对 a 正交化 => 相关 ~ 0
    df = synth_df(n=n, seed=seed)
    df["d1_low_to_close_recovery"] = a
    df["d1_afternoon_return"] = b
    return df


def synth_full_table(n: int = 200, seed: int = 11) -> pd.DataFrame:
    """13 列 df: PRIMITIVE_COLUMNS 全部 11 列 + RECLAIM 2 列 (供 RESET 对照)."""
    rng = np.random.default_rng(seed + 1)
    df = synth_df(n=n, seed=seed)
    for col in PRIMITIVE_COLUMNS:
        if col not in df:
            df[col] = rng.normal(size=n)
    df["board_streak_is_3"] = rng.integers(0, 2, size=n)
    return df


def synth_audit_csv(rows: int = 59, seed: int = 31,
                    target_kind: str = "zeros") -> tuple[pathlib.Path, pathlib.Path]:
    """写一个含 target 列的合成 model table CSV (June/July 窗口内), 返回 (csv, dir).

    日期池 = June 1..30 (30 个) + July 1..29 (29 个) 循环取前 rows 行;
    rows <= 59 时全部落在窗口内。
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_audit_"))
    df = synth_df(n=rows, seed=seed)
    pool = ([f"2026-06-{d:02d}" for d in range(1, 31)]
            + [f"2026-07-{d:02d}" for d in range(1, 30)])
    ids = pd.DataFrame({
        "event_id": [f"E{i:05d}" for i in range(rows)],
        "code": [f"{600000 + i % 3000}" for i in range(rows)],
        "signal_date": [pool[i % len(pool)] for i in range(rows)],
    })
    table = pd.concat([ids, df], axis=1)
    if target_kind == "zeros":
        table["target7_daily_d2open_d3high"] = 0
    else:
        rng = np.random.default_rng(seed + 100)
        table["target7_daily_d2open_d3high"] = rng.integers(0, 2, size=rows)
    csv = tmp / "synth_model_table.csv"
    table.to_csv(csv, index=False)
    return csv, tmp


def _manual_primitive_params(df: pd.DataFrame) -> dict:
    """手工复算 primitive params (与 fit_repair_transform 相同 FOLD_CLIP_Z 顺序)."""
    params = {}
    for col in REPAIR_PRIMITIVES:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = x[np.isfinite(x)]
        q01, q99 = np.quantile(finite, [0.01, 0.99])
        clipped = np.clip(finite, q01, q99)
        params[col] = {"q01": float(q01), "q99": float(q99),
                       "mu": float(clipped.mean()), "sigma": float(clipped.std(ddof=0))}
    return params


def _z(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = {}
    for col in REPAIR_PRIMITIVES:
        p = params[col]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        out[col] = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
    return pd.DataFrame(out, index=df.index)


class R1R2MembershipTests(unittest.TestCase):
    """§37 1/2/6/7/8: 成员冻结."""

    def test_r1_exactly_4_factors(self):
        self.assertEqual(R1_FACTORS, ("OPEN", "DIVERGENCE", "SUPPLY", "RECLAIM"))
        self.assertEqual(len(R1_FACTORS), 4)

    def test_r2_exactly_7_factors(self):
        self.assertEqual(
            R2_FACTORS,
            ("OPEN", "DIVERGENCE", "SUPPLY", "RECLAIM",
             "DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_SUPPLY"))
        self.assertEqual(len(R2_FACTORS), 7)
        self.assertEqual(BASE_FACTORS, R1_FACTORS)
        self.assertEqual(len(DERIVED_FACTORS), 3)

    def test_legacy_highzone_not_in_models(self):
        self.assertNotIn("HIGHZONE", R1_FACTORS)
        self.assertNotIn("HIGHZONE", R2_FACTORS)

    def test_legacy_path_factors_not_in_models(self):
        for f in ("MOM7", "DAMAGE7", "REGIME"):
            self.assertNotIn(f, R1_FACTORS)
            self.assertNotIn(f, R2_FACTORS)

    def test_legacy_sensitivity_factors_not_in_models(self):
        for f in ("POS7", "TREND"):
            self.assertNotIn(f, R1_FACTORS)
            self.assertNotIn(f, R2_FACTORS)

    def test_all_legacy_registered(self):
        self.assertEqual(
            LEGACY_SENSITIVITY_FACTORS,
            ("HIGHZONE", "MOM7", "DAMAGE7", "REGIME", "POS7", "TREND"))


class DerivedHierarchyTests(unittest.TestCase):
    """§37 9: hierarchy 原则."""

    def test_every_derived_parent_present_in_r2(self):
        for d, parents in DERIVED_PARENTS.items():
            self.assertIn(d, R2_FACTORS)
            for p in parents:
                self.assertIn(p, R2_FACTORS)
                self.assertIn(p, R1_FACTORS)

    def test_derived_not_in_r1(self):
        for d in DERIVED_FACTORS:
            self.assertNotIn(d, R1_FACTORS)

    def test_exactly_three_derived_terms(self):
        self.assertEqual(
            set(DERIVED_PARENTS),
            {"DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_SUPPLY"})

    def test_no_unlisted_interactions(self):
        # 禁止自动增加 interaction 不属于 v001: 无其他 derived 常量
        self.assertEqual(set(DERIVED_FACTORS), set(DERIVED_PARENTS))


class FormulaTests(unittest.TestCase):
    """§37 3/4/5/10/11/12/13: 公式冻结."""

    def _fit(self, df: pd.DataFrame):
        return fit_repair_transform(df)

    def test_divergence_matches_legacy_reset(self):
        """§37 3: DIVERGENCE 与旧 RESET 数学定义完全一致 (数值全等)."""
        n = 200
        df = synth_full_table(n=n, seed=11)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)

        legacy_params = fit_reference_transform(june)
        legacy_factors = construct_factors(apply, legacy_params)
        repair_params = fit_repair_transform(june)
        repair_factors = construct_repair_factors(apply, repair_params)

        self.assertEqual(
            list(legacy_factors["RESET"].to_numpy()),
            list(repair_factors["DIVERGENCE"].to_numpy()))

    def test_reclaim_is_equal_weight_z_then_standardized(self):
        """§37 4: RECLAIM = z((z(a) + z(b)) / 2)."""
        df = synth_df(n=200, seed=12)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = self._fit(june)

        z = _z(apply, params)
        g = composite_reclaim(z).to_numpy(dtype=float)
        c = params["composite_RECLAIM"]
        expected = (g - c["mu"]) / c["sigma"]

        factors = construct_repair_factors(apply, params)
        np.testing.assert_allclose(factors["RECLAIM"].to_numpy(), expected,
                                   rtol=1e-12, atol=1e-12)

    def test_reclaim_composite_mu_sigma_from_reference(self):
        df = synth_df(n=200, seed=13)
        june = df.iloc[:150].reset_index(drop=True)
        params = self._fit(june)
        z = _z(june, params)
        g = composite_reclaim(z).to_numpy(dtype=float)
        self.assertAlmostEqual(params["composite_RECLAIM"]["mu"],
                               float(g.mean()), places=12)
        self.assertAlmostEqual(params["composite_RECLAIM"]["sigma"],
                               float(g.std(ddof=0)), places=12)

    def test_supply_only_from_primitive(self):
        """§37 5: SUPPLY 只来自 late_day_sell_volume_ratio."""
        df = synth_df(n=200, seed=14)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = self._fit(june)
        factors = construct_repair_factors(apply, params)
        z = _z(apply, params)
        np.testing.assert_allclose(
            factors["SUPPLY"].to_numpy(),
            z["late_day_sell_volume_ratio"].to_numpy(),
            rtol=1e-12, atol=1e-12)

    def test_open_only_from_primitive(self):
        df = synth_df(n=200, seed=15)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = self._fit(june)
        factors = construct_repair_factors(apply, params)
        z = _z(apply, params)
        np.testing.assert_allclose(
            factors["OPEN"].to_numpy(),
            z["break_open_return"].to_numpy(),
            rtol=1e-12, atol=1e-12)

    def test_divergence_sq_formula(self):
        """§37 10: DIVERGENCE_SQ = z(D^2)."""
        df = synth_df(n=200, seed=16)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = self._fit(june)
        base = construct_repair_factors(apply, params)
        raw = base["DIVERGENCE"].to_numpy() ** 2
        p = params["DIVERGENCE_SQ"]
        expected = (raw - p["mu"]) / p["sigma"]
        np.testing.assert_allclose(base["DIVERGENCE_SQ"].to_numpy(), expected,
                                   rtol=1e-12, atol=1e-12)

    def test_divergence_x_reclaim_formula(self):
        """§37 11: DIVERGENCE_X_RECLAIM = z(D*Q)."""
        df = synth_df(n=200, seed=17)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = self._fit(june)
        base = construct_repair_factors(apply, params)
        raw = (base["DIVERGENCE"].to_numpy()
               * base["RECLAIM"].to_numpy())
        p = params["DIVERGENCE_X_RECLAIM"]
        expected = (raw - p["mu"]) / p["sigma"]
        np.testing.assert_allclose(base["DIVERGENCE_X_RECLAIM"].to_numpy(),
                                   expected, rtol=1e-12, atol=1e-12)

    def test_divergence_x_supply_formula(self):
        """§37 12: DIVERGENCE_X_SUPPLY = z(D*S)."""
        df = synth_df(n=200, seed=18)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = self._fit(june)
        base = construct_repair_factors(apply, params)
        raw = (base["DIVERGENCE"].to_numpy()
               * base["SUPPLY"].to_numpy())
        p = params["DIVERGENCE_X_SUPPLY"]
        expected = (raw - p["mu"]) / p["sigma"]
        np.testing.assert_allclose(base["DIVERGENCE_X_SUPPLY"].to_numpy(),
                                   expected, rtol=1e-12, atol=1e-12)

    def test_derived_params_from_reference(self):
        df = synth_df(n=200, seed=19)
        june = df.iloc[:150].reset_index(drop=True)
        params = self._fit(june)
        base = construct_repair_factors(june, params)
        raw = {"DIVERGENCE_SQ": base["DIVERGENCE"].to_numpy() ** 2,
               "DIVERGENCE_X_RECLAIM": (base["DIVERGENCE"].to_numpy()
                                        * base["RECLAIM"].to_numpy()),
               "DIVERGENCE_X_SUPPLY": (base["DIVERGENCE"].to_numpy()
                                       * base["SUPPLY"].to_numpy())}
        for name in DERIVED_FACTORS:
            self.assertAlmostEqual(params[name]["mu"],
                                   float(np.mean(raw[name])), places=12)
            self.assertAlmostEqual(params[name]["sigma"],
                                   float(np.std(raw[name], ddof=0)), places=12)

    def test_derived_no_second_clip(self):
        """§37 13: derived terms 不做第二次 clip (params 无 q01/q99)."""
        df = synth_df(n=200, seed=20)
        params = self._fit(df)
        for name in DERIVED_FACTORS:
            self.assertNotIn("q01", params[name])
            self.assertNotIn("q99", params[name])
            self.assertIn("mu", params[name])
            self.assertIn("sigma", params[name])


class TransformOrderTests(unittest.TestCase):
    """§37 14/15/16/17: FOLD_CLIP_Z 顺序 + July 不能改变 June 参数."""

    def test_primitive_clip_then_clipped_mean_std(self):
        """§37 14: primitive 参数 = q01/q99(raw) -> clip -> mean/std(clipped)."""
        df = synth_df(n=200, seed=22)
        june = df.iloc[:150].reset_index(drop=True)
        params = fit_repair_transform(june)
        manual = _manual_primitive_params(june)
        for col in REPAIR_PRIMITIVES:
            self.assertAlmostEqual(params[col]["q01"], manual[col]["q01"],
                                   places=12)
            self.assertAlmostEqual(params[col]["q99"], manual[col]["q99"],
                                   places=12)
            self.assertAlmostEqual(params[col]["mu"], manual[col]["mu"],
                                   places=12)
            self.assertAlmostEqual(params[col]["sigma"], manual[col]["sigma"],
                                   places=12)

    def test_july_uses_june_primitive_params(self):
        """§37 15: July 构造只 apply June primitive 参数."""
        df = synth_df(n=240, seed=23)
        june = df.iloc[:180].reset_index(drop=True)
        july = df.iloc[180:].reset_index(drop=True)
        params = fit_repair_transform(june)
        zj = _z(july, params)
        manual = _manual_primitive_params(june)
        for col in REPAIR_PRIMITIVES:
            p = manual[col]
            x = pd.to_numeric(july[col], errors="coerce").to_numpy(dtype=float)
            expected = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
            np.testing.assert_allclose(zj[col].to_numpy(), expected,
                                       rtol=1e-12, atol=1e-12)

    def test_july_cannot_change_june_primitive_params(self):
        """§37 15: July 不能改变 June primitive params (fit 只读 June)."""
        # spy: 审计流程中 fit_repair_transform 只收到 June 数据 (primitive/composite/
        # derived 参数全部来自 June, July 从不进入 fit)
        csv, tmp = synth_audit_csv(rows=59, seed=24)
        out = tmp / "out"
        seen: list[pd.DataFrame] = []
        real_fit = audit_mod.fit_repair_transform

        def spy(df):
            seen.append(df.copy())
            return real_fit(df)

        audit_mod.fit_repair_transform = spy
        try:
            run_repair_state_audit(csv, out)
        finally:
            audit_mod.fit_repair_transform = real_fit
            _rmtree(tmp)
        self.assertEqual(len(seen), 1, "fit_repair_transform 应恰好被调用一次 (June)")
        self.assertGreater(len(seen[0]), 0)
        dates = set(seen[0]["signal_date"].astype(str))
        self.assertTrue(dates, "fit 入参无数据")
        self.assertTrue(all(JUNE_START <= d <= JUNE_END for d in dates),
                        "fit 入参包含 June 窗口外的日期 (July 泄漏进 fit)")

    def test_july_cannot_change_june_composite_params(self):
        """§37 16: composite 参数只来自 June (June-only 两次拟合必须完全一致)."""
        df = synth_df(n=240, seed=24)
        june = df.iloc[:180].reset_index(drop=True)
        p_june = fit_repair_transform(june)
        p_again = fit_repair_transform(june)
        self.assertEqual(p_june["composite_DIVERGENCE"],
                         p_again["composite_DIVERGENCE"])
        self.assertEqual(p_june["composite_RECLAIM"], p_again["composite_RECLAIM"])
        # composite 参数 == 手工 June 计算
        z = _z(june, p_june)
        g_d = composite_reset(z).to_numpy(dtype=float)
        self.assertAlmostEqual(p_june["composite_DIVERGENCE"]["mu"],
                               float(g_d.mean()), places=12)
        self.assertAlmostEqual(p_june["composite_DIVERGENCE"]["sigma"],
                               float(g_d.std(ddof=0)), places=12)

    def test_july_cannot_change_june_derived_params(self):
        """§37 17: derived 参数只来自 June (June-only 两次拟合必须完全一致)."""
        df = synth_df(n=240, seed=25)
        june = df.iloc[:180].reset_index(drop=True)
        p_june = fit_repair_transform(june)
        p_again = fit_repair_transform(june)
        for name in DERIVED_FACTORS:
            self.assertEqual(p_june[name], p_again[name])

    def test_july_factors_finite_with_june_params(self):
        df = synth_df(n=240, seed=26)
        june = df.iloc[:180].reset_index(drop=True)
        july = df.iloc[180:].reset_index(drop=True)
        params = fit_repair_transform(june)
        factors = construct_repair_factors(july, params)
        for col in R2_FACTORS:
            self.assertTrue(np.all(np.isfinite(factors[col])))

    def test_empty_reference_fails(self):
        with self.assertRaises(RepairStateError):
            fit_repair_transform(synth_df(n=0))

    def test_degenerate_primitive_fails(self):
        df = synth_df(n=200, seed=27)
        df["d1_afternoon_return"] = 1.0  # 常数列 => sigma = 0
        with self.assertRaises(RepairStateError):
            fit_repair_transform(df)


class TargetBlindTests(unittest.TestCase):
    """§37 18/19/20: target-blind 读取入口."""

    def test_target_column_not_in_usecols(self):
        """§37 18: 请求列含 target => 显式失败; 正常 usecols 不含 target."""
        csv, tmp = synth_audit_csv(rows=40, seed=32)
        try:
            with self.assertRaises(FactorSpecError):
                load_x_table(csv, primitive_columns=REPAIR_PRIMITIVES
                             + ("target7_daily_d2open_d3high",))
            df = load_x_table(csv)
            self.assertNotIn("target7_daily_d2open_d3high", df.columns)
            self.assertEqual(list(df.columns),
                             ["event_id", "code", "signal_date"]
                             + list(REPAIR_PRIMITIVES))
        finally:
            _rmtree(tmp)

    def test_future_d2_d3_fields_rejected(self):
        """§37 20: future/D2/D3 字段无法进入 audit."""
        csv, tmp = synth_audit_csv(rows=40, seed=33)
        try:
            for bad in ("d2_open", "d3_high", "future_return",
                        "m1_probability", "m1_rank_daily"):
                with self.assertRaises(FactorSpecError, msg=bad):
                    load_x_table(csv, primitive_columns=REPAIR_PRIMITIVES + (bad,))
        finally:
            _rmtree(tmp)

    def test_target_randomization_does_not_change_outputs(self):
        """§37 19: target 随机化不能改变任何正式输出 (loader 从不读取 target).

        注意: 输入文件不同 => input SHA256 指纹必然不同, review.md 仅该行例外;
        其余全部内容必须字节级一致。
        """
        csv_a, tmp_a = synth_audit_csv(rows=59, seed=34, target_kind="zeros")
        csv_b, tmp_b = synth_audit_csv(rows=59, seed=34, target_kind="random")
        out_a = tmp_a / "out_a"
        out_b = tmp_b / "out_b"
        try:
            run_repair_state_audit(csv_a, out_a)
            run_repair_state_audit(csv_b, out_b)
            for name in OUTPUT_FILES:
                if name == "v004c_repair_state_review_v001.md":
                    lines_a = [ln for ln in (out_a / name).read_text(encoding="utf-8")
                               .splitlines() if "SHA256" not in ln]
                    lines_b = [ln for ln in (out_b / name).read_text(encoding="utf-8")
                               .splitlines() if "SHA256" not in ln]
                    self.assertEqual(lines_a, lines_b,
                                     f"{name} 除 SHA256 指纹外因 target 随机化而改变")
                else:
                    self.assertTrue(
                        filecmp.cmp(out_a / name, out_b / name, shallow=False),
                        f"{name} 因 target 随机化而改变")
        finally:
            _rmtree(tmp_a)
            _rmtree(tmp_b)


class AuditStatusTests(unittest.TestCase):
    """§37 21/22/23: 判定门规则."""

    def test_vif_severe_rule(self):
        """§37 21: VIF >= 10 => REVIEW_REQUIRED."""
        rng = np.random.default_rng(41)
        n = 400
        a = rng.normal(size=n)
        b = 2.0 * a + 1e-4 * rng.normal(size=n)
        df = pd.DataFrame({"OPEN": a, "DIVERGENCE": b,
                           "SUPPLY": rng.normal(size=n),
                           "RECLAIM": rng.normal(size=n)})
        vifs = {"OPEN": 1.1, "DIVERGENCE": 12.0, "SUPPLY": 1.0, "RECLAIM": 1.0}
        kappa = {name: {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                        "near_singular": False}
                 for name in ("R1", "R2", "REPAIR_BLOCK", "CONDITIONAL_BLOCK")}
        status, reasons = audit_status([], vifs, vifs, kappa, False)
        self.assertEqual(status, "REVIEW_REQUIRED")
        self.assertTrue(any("VIF" in r for r in reasons))
        # 直接验证 vif_matrix 在该矩阵上确实 >= 10 (数字真实)
        self.assertGreaterEqual(vif_matrix(df)["DIVERGENCE"], 10.0)

    def test_condition_number_severe_rule(self):
        """§37 22: kappa >= 100 => REVIEW_REQUIRED."""
        rng = np.random.default_rng(42)
        n = 400
        a = rng.normal(size=n)
        b = 2.0 * a + 1e-6 * rng.normal(size=n)
        c = -a + 1e-6 * rng.normal(size=n)
        df = pd.DataFrame({"A": a, "B": b, "C": c})
        s_max, s_min, kappa, near = condition_number_diagnostic(
            df.to_numpy(dtype=float))
        self.assertGreaterEqual(kappa, 100.0)
        blocks = {"R1": {"s_max": s_max, "s_min": s_min, "kappa": kappa,
                         "near_singular": near},
                  "R2": {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                         "near_singular": False},
                  "REPAIR_BLOCK": {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                                   "near_singular": False},
                  "CONDITIONAL_BLOCK": {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                                        "near_singular": False}}
        status, reasons = audit_status([], {"a": 1.0, "b": 1.0, "c": 1.0},
                                       {"a": 1.0, "b": 1.0, "c": 1.0},
                                       blocks, False)
        self.assertEqual(status, "REVIEW_REQUIRED")
        self.assertTrue(any("kappa" in r or "condition" in r for r in reasons))

    def test_reclaim_composite_review_rule(self):
        """§37 23: RECLAIM 两 primitive 双低相关 => RECLAIM_COMPOSITE_REVIEW."""
        df = synth_df_weak_reclaim(n=300, seed=43)
        june = df.iloc[:240].reset_index(drop=True)
        params = fit_repair_transform(june)
        factors = construct_repair_factors(june, params)
        # 直接用 raw primitive 相关 (与 audit 的 reclaim 判定一致)
        rp = _corr(june["d1_low_to_close_recovery"], june["d1_afternoon_return"])
        rs = _spearman(june["d1_low_to_close_recovery"], june["d1_afternoon_return"])
        self.assertLess(abs(rp), 0.10)
        self.assertLess(abs(rs), 0.10)
        status, reasons = audit_status([], {}, {}, {}, True)
        self.assertEqual(status, "RECLAIM_COMPOSITE_REVIEW")
        self.assertTrue(any("RECLAIM" in r for r in reasons))

    def test_pass_status(self):
        status, reasons = audit_status(
            [], {"OPEN": 1.1, "DIVERGENCE": 1.2, "SUPPLY": 1.0, "RECLAIM": 1.3},
            {"OPEN": 1.1, "DIVERGENCE": 1.2, "SUPPLY": 1.0, "RECLAIM": 1.3,
             "DIVERGENCE_SQ": 1.5, "DIVERGENCE_X_RECLAIM": 1.4,
             "DIVERGENCE_X_SUPPLY": 1.2},
            {name: {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                    "near_singular": False}
             for name in ("R1", "R2", "REPAIR_BLOCK", "CONDITIONAL_BLOCK")},
            False)
        self.assertEqual(status, "PASS_REPAIR_STATE_X_V001")
        self.assertEqual(reasons, [])

    def test_review_outranks_composite_review(self):
        # REVIEW_REQUIRED 优先级高于 RECLAIM_COMPOSITE_REVIEW
        status, _ = audit_status(
            [], {"a": 1.0}, {"a": 20.0},
            {"R1": {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                    "near_singular": False}}, True)
        self.assertEqual(status, "REVIEW_REQUIRED")


def _corr(a, b) -> float:
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b) -> float:
    s = pd.DataFrame({"a": pd.Series(pd.to_numeric(a, errors="coerce")),
                      "b": pd.Series(pd.to_numeric(b, errors="coerce"))}).dropna()
    return float(s["a"].corr(s["b"], method="spearman"))


def _rmtree(path: pathlib.Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


class DeterministicRebuildTests(unittest.TestCase):
    """§37 24: deterministic rebuild — 同一输入跑两遍 6 资产字节级一致."""

    def test_synthetic_rebuild_byte_identical(self):
        csv, tmp = synth_audit_csv(rows=59, seed=51)
        out1 = tmp / "out1"
        out2 = tmp / "out2"
        try:
            run_repair_state_audit(csv, out1)
            run_repair_state_audit(csv, out2)
            for name in OUTPUT_FILES:
                self.assertTrue(
                    filecmp.cmp(out1 / name, out2 / name, shallow=False),
                    f"{name} 两次运行不一致")
        finally:
            _rmtree(tmp)

    def test_synthetic_audit_runs_and_status_valid(self):
        csv, tmp = synth_audit_csv(rows=59, seed=52)
        out = tmp / "out"
        try:
            r = run_repair_state_audit(csv, out)
            self.assertIn(r["status"],
                          ("PASS_REPAIR_STATE_X_V001", "RECLAIM_COMPOSITE_REVIEW",
                           "REVIEW_REQUIRED"))
            self.assertEqual(r["june_rows"], 30)
            self.assertEqual(r["july_rows"], 29)
            self.assertEqual(len(list(out.glob("*.csv"))) +
                             len(list(out.glob("*.md"))), 6)
        finally:
            _rmtree(tmp)


class RealDataTests(unittest.TestCase):
    """真实 model table 的 June/July 形状与正式输出 (只写临时目录)."""

    MODEL_TABLE = (pathlib.Path(__file__).resolve().parents[1]
                   / "reports/research/v004c_model_table_v001_20260601_20260729"
                   / "v004c_model_table_v001.csv")

    def test_real_june_july_shape(self):
        if not self.MODEL_TABLE.exists():
            self.skipTest("model table 不存在")
        with tempfile.TemporaryDirectory(prefix="repair_real_") as tmp:
            r = run_repair_state_audit(self.MODEL_TABLE, pathlib.Path(tmp))
            self.assertEqual(r["june_rows"], 173)
            self.assertEqual(r["june_date_count"], 21)
            self.assertEqual(r["july_rows"], 160)
            self.assertEqual(r["july_date_count"], 21)
            self.assertIn(r["status"],
                          ("PASS_REPAIR_STATE_X_V001", "RECLAIM_COMPOSITE_REVIEW",
                           "REVIEW_REQUIRED"))

    def test_real_target_never_loaded(self):
        if not self.MODEL_TABLE.exists():
            self.skipTest("model table 不存在")
        x = load_x_table(self.MODEL_TABLE)
        self.assertNotIn("target7_daily_d2open_d3high", x.columns)
        self.assertEqual(len(x), 333)


if __name__ == "__main__":
    unittest.main()

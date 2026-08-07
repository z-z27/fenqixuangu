"""v004c Repair-State Model v002 — 语义修订 spec + 纯 X 审计测试。

覆盖任务 §39 的 28 项要求:

1  v001 历史文件未修改 (SHA256 快照)
2  d1_intraday_range definition audit 存在于 review
3  DIVERGENCE 只来自合法 D1 range primitive
4  DIVERGENCE 不使用 close-to-VWAP/open-to-close/high-to-close
5  CLOSE_DAMAGE 与旧 RESET 逐行相等
6  SUPPLY 保持单 primitive
7  RECLAIM 与 v001 逐行相等
8  R1 严格 5 factors
9  R2 严格 8 factors
10 DIV_SQ 公式
11 DIV_X_RECLAIM 公式
12 DIV_X_DAMAGE 公式
13 DIV_X_SUPPLY 不在 R2
14 TURNOVER_COST 不在 R1/R2
15 derived 不做 second clipping
16 primitive clip -> clipped mean/std
17 June-only fit
18 July 只能 apply
19 target 不在 usecols
20 D2/D3/future 字段无法进入
21 Target 随机化不影响所有正式资产
22 semantic decoupling gate
23 VIF gate
24 condition-number gate
25 hierarchy
26 deterministic rebuild
27 old v001 40 tests 仍全部通过 (跑测试命令验证)
28 old factor-spec 45 tests 仍全部通过 (跑测试命令验证)

全部测试确定性 (固定 seed / 显式构造), 不读取真实 Target。
"""

from __future__ import annotations

import filecmp
import hashlib
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
from src.v004c_repair_state_spec import (  # v001 (只读, 供逐行相等对照)
    BASE_FACTORS as BASE_FACTORS_V1,
    R2_FACTORS as R2_FACTORS_V1,
    construct_repair_factors as construct_repair_factors_v1,
    fit_repair_transform as fit_repair_transform_v1,
)
import src.v004c_repair_state_spec_v002 as spec_mod  # monkeypatch 目标 (模块全局)
from src.v004c_repair_state_spec_v002 import (
    BASE_FACTORS,
    BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH,
    CLOSE_DAMAGE_PRIMITIVES,
    DERIVED_FACTORS,
    DERIVED_PARENTS,
    DIVERGENCE_PRIMITIVE,
    FACTOR_PRIMITIVES,
    LEGACY_SENSITIVITY_FACTORS,
    R1_FACTORS,
    R2_FACTORS,
    RECLAIM_PRIMITIVES,
    REPAIR_PRIMITIVES,
    SENSITIVITY_FACTORS,
    SENSITIVITY_PRIMITIVES,
    RepairStateError,
    audit_status_v2,
    construct_repair_factors_v2,
    fit_repair_state_transform_v2,
    semantic_decoupling_status,
)
from tools import v004c_repair_state_dependence_audit_v002 as audit_mod
from tools.v004c_repair_state_dependence_audit_v002 import (
    OUTPUT_FILES,
    load_x_table,
    run_repair_state_audit,
)

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# v001 历史文件 SHA256 快照 (commit 0223c274bd30889d5856e9dcd8efc540aadd5b99 之后
# 不得修改; 任何修改 => 测试失败)
V001_SNAPSHOTS: dict[str, str] = {
    "src/v004c_repair_state_spec.py":
        "693e994a65ea9400c9e460ce77bdc879636c1fdff63eeef40c4a90f824b89203",
    "tools/v004c_repair_state_dependence_audit.py":
        "8f9d0fa1b5a4a3885433f4665d80e938e25147c39e2525a82d84c128a7f41ee5",
    "tests/test_v004c_repair_state_spec.py":
        "9da42feeb3b26399cf7dbe22d4caadb910e08bd84e5a41c74684fb930dc7a4bd",
    "reports/research/v004c_repair_state_dependence_v001_20260601_20260729/"
    "v004c_repair_state_primitive_dependence_v001.csv":
        "08f772535efd4d647246e1bc6703af5a95e8e1092990eb0cb003a70d3324fab9",
    "reports/research/v004c_repair_state_dependence_v001_20260601_20260729/"
    "v004c_repair_state_factor_dependence_v001.csv":
        "995ee7fbcabdef56b3099029e02386a89bea7422d8315ee568e64f5434a632d3",
    "reports/research/v004c_repair_state_dependence_v001_20260601_20260729/"
    "v004c_repair_state_diagnostics_v001.csv":
        "1a56ab10da87b7043543c6124b39c5fb78f01e23e2ca2dce792a45b3d7894634",
    "reports/research/v004c_repair_state_dependence_v001_20260601_20260729/"
    "v004c_repair_state_spec_v001.csv":
        "5165904cec2987828f601fc4ff26b8a727e1de63db046a84010d080df6242408",
    "reports/research/v004c_repair_state_dependence_v001_20260601_20260729/"
    "v004c_repair_state_model_spec_v001.md":
        "0d10cf2b2c905105950429138148bd9b0c154fc15b999e679a89a08837644080",
    "reports/research/v004c_repair_state_dependence_v001_20260601_20260729/"
    "v004c_repair_state_review_v001.md":
        "805581239fe0bc3f215fa15c2ea53893b03b56a8f7bda1f4489fbfaf690dc2ea",
}

# v001 被禁止修改的契约: 成员冻结 (v002 的拆分基础)
V1_R1_FACTORS = ("OPEN", "DIVERGENCE", "SUPPLY", "RECLAIM")
V1_R2_FACTORS = ("OPEN", "DIVERGENCE", "SUPPLY", "RECLAIM",
                 "DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_SUPPLY")


# ---------------------------------------------------------------------------
# 合成数据生成器 (确定性)
# ---------------------------------------------------------------------------

def synth_df(n: int = 200, seed: int = 11) -> pd.DataFrame:
    """8 个 core primitive 的合成 X 数据 (低相关 base + 高相关 RECLAIM 内部对).

    d1_intraday_range 与 RECLAIM 潜在变量独立 => DIVERGENCE vs RECLAIM 解耦。
    """
    rng = np.random.default_rng(seed)
    z_div = rng.normal(size=n)
    z_rec = rng.normal(size=n)
    z_sup = rng.normal(size=n)
    return pd.DataFrame({
        "break_open_return": z_div + rng.normal(scale=0.5, size=n),
        "d1_intraday_range": z_div + rng.normal(scale=0.3, size=n),
        "d1_open_to_close_return_raw": -z_div * 0.8 + rng.normal(scale=0.5, size=n),
        "d1_high_to_close_drawdown_raw": z_div * 0.9 + rng.normal(scale=0.4, size=n),
        "d1_close_to_vwap_raw": -z_div * 0.7 + rng.normal(scale=0.5, size=n),
        "late_day_sell_volume_ratio": z_sup + rng.normal(scale=0.5, size=n),
        "d1_low_to_close_recovery": z_rec + rng.normal(scale=0.6, size=n),
        "d1_afternoon_return": z_rec + rng.normal(scale=0.6, size=n),
    })


def synth_df_coupled(n: int = 300, seed: int = 61) -> pd.DataFrame:
    """DIVERGENCE 与 RECLAIM 强负相关的合成 X (触发 SEMANTIC_DECOUPLING_REVIEW).

    RECLAIM 两个 primitive 都由 -d1_intraday_range 构造 =>
    Pearson(DIVERGENCE, RECLAIM) 显著 <= -0.70。
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    df = synth_df(n=n, seed=seed)
    df["d1_intraday_range"] = v
    df["d1_low_to_close_recovery"] = -v + rng.normal(scale=0.2, size=n)
    df["d1_afternoon_return"] = -v + rng.normal(scale=0.2, size=n)
    return df


def synth_full_table(n: int = 200, seed: int = 11) -> pd.DataFrame:
    """14 列 df: PRIMITIVE_COLUMNS 全部 11 列 + RECLAIM 2 列 + d1_intraday_range."""
    rng = np.random.default_rng(seed + 1)
    df = synth_df(n=n, seed=seed)
    for col in PRIMITIVE_COLUMNS:
        if col not in df:
            df[col] = rng.normal(size=n)
    df["board_streak_is_3"] = rng.integers(0, 2, size=n)
    return df


def synth_audit_csv(rows: int = 59, seed: int = 31,
                    target_kind: str = "zeros",
                    coupled: bool = False) -> tuple[pathlib.Path, pathlib.Path]:
    """写一个含 target 列的合成 model table CSV (June/July 窗口内), 返回 (csv, dir).

    日期池 = June 1..30 (30 个) + July 1..29 (29 个) 循环取前 rows 行;
    rows <= 59 时全部落在窗口内。
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_audit_v2_"))
    df = synth_df_coupled(n=rows, seed=seed) if coupled else synth_df(n=rows, seed=seed)
    pool = ([f"2026-06-{d:02d}" for d in range(1, 31)]
            + [f"2026-07-{d:02d}" for d in range(1, 30)])
    ids = pd.DataFrame({
        "event_id": [f"E{i:05d}" for i in range(rows)],
        "code": [f"{600000 + i % 3000}" for i in range(rows)],
        "signal_date": [pool[i % len(pool)] for i in range(rows)],
    })
    sens = pd.DataFrame({col: np.random.default_rng(seed + 200).normal(size=rows)
                         for col in SENSITIVITY_PRIMITIVES})
    table = pd.concat([ids, df, sens], axis=1)
    if target_kind == "zeros":
        table["target7_daily_d2open_d3high"] = 0
    else:
        rng = np.random.default_rng(seed + 100)
        table["target7_daily_d2open_d3high"] = rng.integers(0, 2, size=rows)
    csv = tmp / "synth_model_table.csv"
    table.to_csv(csv, index=False)
    return csv, tmp


def _manual_primitive_params(df: pd.DataFrame) -> dict:
    """手工复算 primitive params (与 fit_repair_state_transform_v2 相同 FOLD_CLIP_Z)."""
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


def _corr(a, b) -> float:
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b) -> float:
    s = pd.DataFrame({"a": pd.Series(pd.to_numeric(a, errors="coerce")),
                      "b": pd.Series(pd.to_numeric(b, errors="coerce"))}).dropna()
    return float(s["a"].corr(s["b"], method="spearman"))


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rmtree(path: pathlib.Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# §39 1: v001 历史文件未修改
# ---------------------------------------------------------------------------

class V001ImmutabilityTests(unittest.TestCase):
    """§39 1: v001 全部历史资产字节级不变 (SHA256 快照)."""

    def test_v001_source_files_unchanged(self):
        for rel, snapshot in V001_SNAPSHOTS.items():
            path = REPO_ROOT / rel
            self.assertTrue(path.exists(), f"v001 资产缺失: {rel}")
            self.assertEqual(_sha256_file(path), snapshot,
                             f"v001 历史文件被修改: {rel}")

    def test_v001_membership_contract(self):
        # v002 的拆分前提: v001 的成员冻结
        self.assertEqual(BASE_FACTORS_V1, V1_R1_FACTORS)
        self.assertEqual(R2_FACTORS_V1, V1_R2_FACTORS)


# ---------------------------------------------------------------------------
# §39 2/3/4: DIVERGENCE primitive
# ---------------------------------------------------------------------------

class DivergencePrimitiveTests(unittest.TestCase):
    """§39 2/3/4: DIVERGENCE 只来自合法 D1 range primitive."""

    def test_divergence_primitive_is_intraday_range(self):
        """§39 3: DIVERGENCE 构造源 = d1_intraday_range 单 primitive."""
        self.assertEqual(DIVERGENCE_PRIMITIVE, "d1_intraday_range")
        self.assertEqual(FACTOR_PRIMITIVES["DIVERGENCE"],
                         ("d1_intraday_range",))
        self.assertIn("d1_intraday_range", REPAIR_PRIMITIVES)
        self.assertEqual(FACTOR_PRIMITIVES["DIVERGENCE"], (DIVERGENCE_PRIMITIVE,))

    def test_divergence_forbids_old_reset_primitives(self):
        """§39 4: DIVERGENCE 不使用 close-to-VWAP / open-to-close / high-to-close."""
        self.assertNotIn("d1_open_to_close_return_raw",
                         FACTOR_PRIMITIVES["DIVERGENCE"])
        self.assertNotIn("d1_high_to_close_drawdown_raw",
                         FACTOR_PRIMITIVES["DIVERGENCE"])
        self.assertNotIn("d1_close_to_vwap_raw", FACTOR_PRIMITIVES["DIVERGENCE"])
        self.assertEqual(
            set(CLOSE_DAMAGE_PRIMITIVES),
            {"d1_open_to_close_return_raw", "d1_high_to_close_drawdown_raw",
             "d1_close_to_vwap_raw"})

    def test_divergence_equals_z_intraday_range(self):
        """DIVERGENCE 数值 == z(d1_intraday_range) (单 primitive 直接标准化)."""
        df = synth_df(n=200, seed=12)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        factors = construct_repair_factors_v2(apply, params)
        z = _z(apply, params)
        np.testing.assert_allclose(
            factors["DIVERGENCE"].to_numpy(),
            z["d1_intraday_range"].to_numpy(),
            rtol=1e-12, atol=1e-12)

    def test_definition_audit_present_in_review(self):
        """§39 2: d1_intraday_range definition audit 存在于 review md."""
        csv, tmp = synth_audit_csv(rows=59, seed=13)
        out = tmp / "out"
        try:
            run_repair_state_audit(csv, out)
            text = (out / "v004c_repair_state_review_v002.md").read_text(
                encoding="utf-8")
        finally:
            _rmtree(tmp)
        self.assertIn("DIVERGENCE primitive 核验", text)
        self.assertIn("(d1_high - d1_low) / prev_close", text)
        self.assertIn("uses only D1 data", text)
        self.assertIn("semantic match", text)
        self.assertIn("exact formula", text)

    def test_blocked_when_divergence_source_mismatch(self):
        """构造源 != d1_intraday_range => 显式失败 (BLOCKED 防御).

        fit_repair_state_transform_v2 内部的构造源检查引用 spec 模块的
        模块级 FACTOR_PRIMITIVES, 所以必须 patch spec_mod 而不是 audit 模块
        的 import 副本。
        """
        df = synth_df(n=100, seed=14)
        old = spec_mod.FACTOR_PRIMITIVES["DIVERGENCE"]
        spec_mod.FACTOR_PRIMITIVES["DIVERGENCE"] = ("break_open_return",)
        try:
            with self.assertRaises(RepairStateError):
                fit_repair_state_transform_v2(df)
        finally:
            spec_mod.FACTOR_PRIMITIVES["DIVERGENCE"] = old


# ---------------------------------------------------------------------------
# §39 5/6/7: CLOSE_DAMAGE / SUPPLY / RECLAIM 公式冻结
# ---------------------------------------------------------------------------

class FormulaFrozenTests(unittest.TestCase):
    """§39 5/6/7: 逐行相等对照."""

    def test_close_damage_equals_legacy_reset(self):
        """§39 5: CLOSE_DAMAGE 与旧 RESET 逐行相等 (同一 reference transform)."""
        n = 200
        df = synth_full_table(n=n, seed=11)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)

        legacy_params = fit_reference_transform(june)
        legacy_factors = construct_factors(apply, legacy_params)
        v2_params = fit_repair_state_transform_v2(june)
        v2_factors = construct_repair_factors_v2(apply, v2_params)

        self.assertEqual(
            list(legacy_factors["RESET"].to_numpy()),
            list(v2_factors["CLOSE_DAMAGE"].to_numpy()))

    def test_supply_only_from_primitive(self):
        """§39 6: SUPPLY 保持单 primitive."""
        df = synth_df(n=200, seed=15)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        factors = construct_repair_factors_v2(apply, params)
        z = _z(apply, params)
        np.testing.assert_allclose(
            factors["SUPPLY"].to_numpy(),
            z["late_day_sell_volume_ratio"].to_numpy(),
            rtol=1e-12, atol=1e-12)
        self.assertEqual(FACTOR_PRIMITIVES["SUPPLY"],
                         ("late_day_sell_volume_ratio",))

    def test_reclaim_equals_v001_reclaim(self):
        """§39 7: RECLAIM 与 v001 逐行相等 (公式 + 参数都冻结)."""
        n = 200
        df = synth_full_table(n=n, seed=16)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)

        v1_params = fit_repair_transform_v1(june)
        v1_factors = construct_repair_factors_v1(apply, v1_params)
        v2_params = fit_repair_state_transform_v2(june)
        v2_factors = construct_repair_factors_v2(apply, v2_params)

        self.assertEqual(
            list(v1_factors["RECLAIM"].to_numpy()),
            list(v2_factors["RECLAIM"].to_numpy()))

    def test_reclaim_composite_from_reference(self):
        df = synth_df(n=200, seed=17)
        june = df.iloc[:150].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        z = _z(june, params)
        g = (z["d1_low_to_close_recovery"] + z["d1_afternoon_return"]) / 2.0
        c = params["composite_RECLAIM"]
        self.assertAlmostEqual(c["mu"], float(g.mean()), places=12)
        self.assertAlmostEqual(c["sigma"], float(g.std(ddof=0)), places=12)

    def test_close_damage_composite_from_reference(self):
        df = synth_df(n=200, seed=18)
        june = df.iloc[:150].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        z = _z(june, params)
        g = (-z["d1_open_to_close_return_raw"]
             + z["d1_high_to_close_drawdown_raw"]
             - z["d1_close_to_vwap_raw"]) / 3.0
        c = params["composite_CLOSE_DAMAGE"]
        self.assertAlmostEqual(c["mu"], float(g.mean()), places=12)
        self.assertAlmostEqual(c["sigma"], float(g.std(ddof=0)), places=12)


# ---------------------------------------------------------------------------
# §39 8/9: R1/R2 成员
# ---------------------------------------------------------------------------

class R1R2MembershipTests(unittest.TestCase):
    """§39 8/9/13/14: 成员冻结."""

    def test_r1_exactly_5_factors(self):
        self.assertEqual(R1_FACTORS,
                         ("OPEN", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY", "RECLAIM"))
        self.assertEqual(len(R1_FACTORS), 5)
        self.assertEqual(BASE_FACTORS, R1_FACTORS)

    def test_r2_exactly_8_factors(self):
        self.assertEqual(
            R2_FACTORS,
            ("OPEN", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY", "RECLAIM",
             "DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_DAMAGE"))
        self.assertEqual(len(R2_FACTORS), 8)
        self.assertEqual(len(DERIVED_FACTORS), 3)

    def test_div_x_supply_not_in_r2(self):
        """§39 13: DIV_X_SUPPLY 不在 R2 (降为 sensitivity, 定义保留)."""
        self.assertNotIn("DIVERGENCE_X_SUPPLY", R1_FACTORS)
        self.assertNotIn("DIVERGENCE_X_SUPPLY", R2_FACTORS)
        self.assertIn("DIVERGENCE_X_SUPPLY", SENSITIVITY_FACTORS)

    def test_turnover_cost_not_in_models(self):
        """§39 14: TURNOVER_COST 不在 R1/R2."""
        self.assertNotIn("TURNOVER_COST", R1_FACTORS)
        self.assertNotIn("TURNOVER_COST", R2_FACTORS)
        self.assertIn("TURNOVER_COST", SENSITIVITY_FACTORS)
        self.assertEqual(SENSITIVITY_PRIMITIVES,
                         ("break_volume_ratio_vs_board_days",))

    def test_legacy_factors_not_in_models(self):
        for f in LEGACY_SENSITIVITY_FACTORS:
            self.assertNotIn(f, R1_FACTORS)
            self.assertNotIn(f, R2_FACTORS)
        self.assertEqual(
            LEGACY_SENSITIVITY_FACTORS,
            ("HIGHZONE", "MOM7", "DAMAGE7", "REGIME", "POS7", "TREND"))

    def test_sensitivity_all_excluded(self):
        self.assertFalse(set(SENSITIVITY_FACTORS) & set(R2_FACTORS))


# ---------------------------------------------------------------------------
# §39 25/10/11/12/15: hierarchy + derived 公式
# ---------------------------------------------------------------------------

class DerivedFormulaTests(unittest.TestCase):
    """§39 25/10/11/12/15: hierarchy + derived 公式冻结."""

    def test_hierarchy_parents_present(self):
        """§39 25: R2 包含每个 derived term 的全部父 factor."""
        self.assertEqual(
            set(DERIVED_PARENTS),
            {"DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_DAMAGE"})
        for d, parents in DERIVED_PARENTS.items():
            self.assertIn(d, R2_FACTORS)
            for p in parents:
                self.assertIn(p, R2_FACTORS)
                self.assertIn(p, R1_FACTORS)

    def test_derived_not_in_r1(self):
        for d in DERIVED_FACTORS:
            self.assertNotIn(d, R1_FACTORS)

    def test_no_unlisted_interactions(self):
        self.assertEqual(set(DERIVED_FACTORS), set(DERIVED_PARENTS))
        for forbidden in ("OPEN_X_DIVERGENCE", "DIV_X_TURNOVER",
                          "RECLAIM_SQ", "SUPPLY_SQ", "CLOSE_DAMAGE_SQ"):
            self.assertNotIn(forbidden, DERIVED_FACTORS)

    def test_divergence_sq_formula(self):
        """§39 10: DIVERGENCE_SQ = z(F_V^2)."""
        df = synth_df(n=200, seed=19)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        base = construct_repair_factors_v2(apply, params)
        raw = base["DIVERGENCE"].to_numpy() ** 2
        p = params["DIVERGENCE_SQ"]
        expected = (raw - p["mu"]) / p["sigma"]
        np.testing.assert_allclose(base["DIVERGENCE_SQ"].to_numpy(), expected,
                                   rtol=1e-12, atol=1e-12)

    def test_divergence_x_reclaim_formula(self):
        """§39 11: DIVERGENCE_X_RECLAIM = z(F_V * F_Q)."""
        df = synth_df(n=200, seed=20)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        base = construct_repair_factors_v2(apply, params)
        raw = (base["DIVERGENCE"].to_numpy() * base["RECLAIM"].to_numpy())
        p = params["DIVERGENCE_X_RECLAIM"]
        expected = (raw - p["mu"]) / p["sigma"]
        np.testing.assert_allclose(base["DIVERGENCE_X_RECLAIM"].to_numpy(),
                                   expected, rtol=1e-12, atol=1e-12)

    def test_divergence_x_damage_formula(self):
        """§39 12: DIVERGENCE_X_DAMAGE = z(F_V * F_C)."""
        df = synth_df(n=200, seed=21)
        june = df.iloc[:150].reset_index(drop=True)
        apply = df.iloc[150:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        base = construct_repair_factors_v2(apply, params)
        raw = (base["DIVERGENCE"].to_numpy() * base["CLOSE_DAMAGE"].to_numpy())
        p = params["DIVERGENCE_X_DAMAGE"]
        expected = (raw - p["mu"]) / p["sigma"]
        np.testing.assert_allclose(base["DIVERGENCE_X_DAMAGE"].to_numpy(),
                                   expected, rtol=1e-12, atol=1e-12)

    def test_derived_params_from_reference(self):
        df = synth_df(n=200, seed=22)
        june = df.iloc[:150].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        base = construct_repair_factors_v2(june, params)
        raw = {"DIVERGENCE_SQ": base["DIVERGENCE"].to_numpy() ** 2,
               "DIVERGENCE_X_RECLAIM": (base["DIVERGENCE"].to_numpy()
                                        * base["RECLAIM"].to_numpy()),
               "DIVERGENCE_X_DAMAGE": (base["DIVERGENCE"].to_numpy()
                                       * base["CLOSE_DAMAGE"].to_numpy())}
        for name in DERIVED_FACTORS:
            self.assertAlmostEqual(params[name]["mu"],
                                   float(np.mean(raw[name])), places=12)
            self.assertAlmostEqual(params[name]["sigma"],
                                   float(np.std(raw[name], ddof=0)), places=12)

    def test_derived_no_second_clip(self):
        """§39 15: derived 不做 second clipping (params 无 q01/q99)."""
        df = synth_df(n=200, seed=23)
        params = fit_repair_state_transform_v2(df)
        for name in DERIVED_FACTORS:
            self.assertNotIn("q01", params[name])
            self.assertNotIn("q99", params[name])
            self.assertIn("mu", params[name])
            self.assertIn("sigma", params[name])


# ---------------------------------------------------------------------------
# §39 16/17/18: transform 顺序 + June-only / July apply
# ---------------------------------------------------------------------------

class TransformOrderTests(unittest.TestCase):
    """§39 16/17/18: FOLD_CLIP_Z 顺序 + July 不能改变 June 参数."""

    def test_primitive_clip_then_clipped_mean_std(self):
        """§39 16: primitive 参数 = q01/q99(raw) -> clip -> mean/std(clipped)."""
        df = synth_df(n=200, seed=24)
        june = df.iloc[:150].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
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
        df = synth_df(n=240, seed=25)
        june = df.iloc[:180].reset_index(drop=True)
        july = df.iloc[180:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        zj = _z(july, params)
        manual = _manual_primitive_params(june)
        for col in REPAIR_PRIMITIVES:
            p = manual[col]
            x = pd.to_numeric(july[col], errors="coerce").to_numpy(dtype=float)
            expected = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
            np.testing.assert_allclose(zj[col].to_numpy(), expected,
                                       rtol=1e-12, atol=1e-12)

    def test_fit_receives_june_only(self):
        """§39 17: June-only fit (spy: fit 恰好一次且只收到 June 数据)."""
        csv, tmp = synth_audit_csv(rows=59, seed=26)
        out = tmp / "out"
        seen: list[pd.DataFrame] = []
        real_fit = audit_mod.fit_repair_state_transform_v2

        def spy(df):
            seen.append(df.copy())
            return real_fit(df)

        audit_mod.fit_repair_state_transform_v2 = spy
        try:
            run_repair_state_audit(csv, out)
        finally:
            audit_mod.fit_repair_state_transform_v2 = real_fit
            _rmtree(tmp)
        self.assertEqual(len(seen), 1, "fit 应恰好被调用一次 (June)")
        self.assertGreater(len(seen[0]), 0)
        dates = set(seen[0]["signal_date"].astype(str))
        self.assertTrue(all(JUNE_START <= d <= JUNE_END for d in dates),
                        "fit 入参包含 June 窗口外的日期 (July 泄漏进 fit)")

    def test_composite_params_june_only_reproducible(self):
        """§39 17/18: composite 参数只来自 June (June-only 两次拟合完全一致)."""
        df = synth_df(n=240, seed=27)
        june = df.iloc[:180].reset_index(drop=True)
        p1 = fit_repair_state_transform_v2(june)
        p2 = fit_repair_state_transform_v2(june)
        self.assertEqual(p1["composite_CLOSE_DAMAGE"],
                         p2["composite_CLOSE_DAMAGE"])
        self.assertEqual(p1["composite_RECLAIM"], p2["composite_RECLAIM"])

    def test_derived_params_june_only_reproducible(self):
        df = synth_df(n=240, seed=28)
        june = df.iloc[:180].reset_index(drop=True)
        p1 = fit_repair_state_transform_v2(june)
        p2 = fit_repair_state_transform_v2(june)
        for name in DERIVED_FACTORS:
            self.assertEqual(p1[name], p2[name])

    def test_july_factors_finite_with_june_params(self):
        df = synth_df(n=240, seed=29)
        june = df.iloc[:180].reset_index(drop=True)
        july = df.iloc[180:].reset_index(drop=True)
        params = fit_repair_state_transform_v2(june)
        factors = construct_repair_factors_v2(july, params)
        for col in R2_FACTORS:
            self.assertTrue(np.all(np.isfinite(factors[col])))

    def test_empty_reference_fails(self):
        with self.assertRaises(RepairStateError):
            fit_repair_state_transform_v2(synth_df(n=0))

    def test_degenerate_primitive_fails(self):
        df = synth_df(n=200, seed=30)
        df["d1_afternoon_return"] = 1.0  # 常数列 => sigma = 0
        with self.assertRaises(RepairStateError):
            fit_repair_state_transform_v2(df)


# ---------------------------------------------------------------------------
# §39 19/20/21: target-blind 读取入口
# ---------------------------------------------------------------------------

class TargetBlindTests(unittest.TestCase):
    """§39 19/20/21: target-blind 读取入口."""

    def test_target_column_not_in_usecols(self):
        """§39 19: 请求列含 target => 显式失败; 正常 usecols 不含 target."""
        csv, tmp = synth_audit_csv(rows=40, seed=32)
        try:
            with self.assertRaises(FactorSpecError):
                load_x_table(csv, primitive_columns=REPAIR_PRIMITIVES
                             + ("target7_daily_d2open_d3high",))
            df = load_x_table(csv)
            self.assertNotIn("target7_daily_d2open_d3high", df.columns)
            self.assertEqual(list(df.columns),
                             ["event_id", "code", "signal_date"]
                             + list(REPAIR_PRIMITIVES)
                             + list(SENSITIVITY_PRIMITIVES))
        finally:
            _rmtree(tmp)

    def test_future_d2_d3_fields_rejected(self):
        """§39 20: future/D2/D3 字段无法进入 audit."""
        csv, tmp = synth_audit_csv(rows=40, seed=33)
        try:
            for bad in ("d2_open", "d3_high", "future_return",
                        "m1_probability", "m1_rank_daily"):
                with self.assertRaises(FactorSpecError, msg=bad):
                    load_x_table(csv, primitive_columns=REPAIR_PRIMITIVES + (bad,))
        finally:
            _rmtree(tmp)

    def test_target_randomization_does_not_change_outputs(self):
        """§39 21: target 随机化不能改变任何正式输出.

        输入文件不同 => input SHA256 指纹必然不同, review.md 仅该行例外;
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
                if name == "v004c_repair_state_review_v002.md":
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


# ---------------------------------------------------------------------------
# §39 22/23/24: 判定门
# ---------------------------------------------------------------------------

def _ok_kappa_blocks():
    return {name: {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                   "near_singular": False}
            for name in ("R1", "R2", "STATE_BLOCK", "CONDITIONAL_BLOCK")}


class AuditStatusTests(unittest.TestCase):
    """§39 22/23/24: 判定门规则."""

    def test_semantic_decoupling_review_rule(self):
        """§39 22: DIVERGENCE vs RECLAIM June |corr| >= 0.70 => 语义阻塞."""
        div_rec = {"pearson": -0.80, "spearman": -0.85}
        status, reasons = audit_status_v2([], {"OPEN": 1.0}, {"OPEN": 1.0},
                                          _ok_kappa_blocks(), div_rec)
        self.assertEqual(status, "SEMANTIC_DECOUPLING_REVIEW")
        self.assertTrue(any("0.70" in r for r in reasons))

    def test_semantic_decoupling_review_integrated(self):
        """集成: 耦合合成数据跑完整 audit => SEMANTIC_DECOUPLING_REVIEW."""
        csv, tmp = synth_audit_csv(rows=59, seed=35, coupled=True)
        out = tmp / "out"
        try:
            r = run_repair_state_audit(csv, out)
            self.assertGreaterEqual(r["decoupling_max_corr"], 0.70)
            self.assertEqual(r["status"], "SEMANTIC_DECOUPLING_REVIEW")
        finally:
            _rmtree(tmp)

    def test_semantic_decoupling_watch_only(self):
        """0.50 <= |corr| < 0.70 => WATCH 记录, 不阻塞 (无数值阻塞 => PASS)."""
        div_rec = {"pearson": -0.55, "spearman": -0.52}
        status, _ = semantic_decoupling_status(div_rec)
        self.assertEqual(status, "SEMANTIC_DECOUPLING_WATCH")
        status, _ = audit_status_v2([], {"OPEN": 1.0}, {"OPEN": 1.0},
                                    _ok_kappa_blocks(), div_rec)
        self.assertEqual(status, "PASS_REPAIR_STATE_X_V002")

    def test_semantic_decoupling_ok(self):
        status, max_corr = semantic_decoupling_status(
            {"pearson": -0.30, "spearman": -0.25})
        self.assertEqual(status, "SEMANTIC_DECOUPLING_OK")
        self.assertAlmostEqual(max_corr, 0.30)

    def test_semantic_review_outranks_numeric(self):
        """语义阻塞优先于数值阻塞; 数值原因一并列出."""
        div_rec = {"pearson": -0.80, "spearman": -0.85}
        status, reasons = audit_status_v2(
            [], {"a": 1.0}, {"a": 20.0}, _ok_kappa_blocks(), div_rec)
        self.assertEqual(status, "SEMANTIC_DECOUPLING_REVIEW")
        self.assertTrue(any("VIF" in r for r in reasons))

    def test_vif_severe_rule(self):
        """§39 23: VIF >= 10 => REVIEW_REQUIRED."""
        rng = np.random.default_rng(41)
        n = 400
        a = rng.normal(size=n)
        b = 2.0 * a + 1e-4 * rng.normal(size=n)
        df = pd.DataFrame({"OPEN": a, "DIVERGENCE": b,
                           "CLOSE_DAMAGE": rng.normal(size=n),
                           "SUPPLY": rng.normal(size=n),
                           "RECLAIM": rng.normal(size=n)})
        vifs = {"OPEN": 1.1, "DIVERGENCE": 12.0, "CLOSE_DAMAGE": 1.0,
                "SUPPLY": 1.0, "RECLAIM": 1.0}
        status, reasons = audit_status_v2([], vifs, vifs, _ok_kappa_blocks(),
                                          {"pearson": 0.2, "spearman": 0.1})
        self.assertEqual(status, "REVIEW_REQUIRED")
        self.assertTrue(any("VIF" in r for r in reasons))
        # 直接验证 vif_matrix 在该矩阵上确实 >= 10 (数字真实)
        self.assertGreaterEqual(vif_matrix(df)["DIVERGENCE"], 10.0)

    def test_condition_number_severe_rule(self):
        """§39 24: kappa >= 100 => REVIEW_REQUIRED."""
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
                  "STATE_BLOCK": {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                                  "near_singular": False},
                  "CONDITIONAL_BLOCK": {"s_max": 3.0, "s_min": 1.0, "kappa": 3.0,
                                        "near_singular": False}}
        status, reasons = audit_status_v2([], {"a": 1.0, "b": 1.0, "c": 1.0},
                                          {"a": 1.0, "b": 1.0, "c": 1.0},
                                          blocks, {"pearson": 0.2, "spearman": 0.1})
        self.assertEqual(status, "REVIEW_REQUIRED")
        self.assertTrue(any("kappa" in r or "condition" in r for r in reasons))

    def test_pass_status(self):
        status, reasons = audit_status_v2(
            [], {"OPEN": 1.1, "DIVERGENCE": 1.2, "CLOSE_DAMAGE": 1.3,
                 "SUPPLY": 1.0, "RECLAIM": 1.4},
            {"OPEN": 1.1, "DIVERGENCE": 1.2, "CLOSE_DAMAGE": 1.3,
             "SUPPLY": 1.0, "RECLAIM": 1.4,
             "DIVERGENCE_SQ": 1.5, "DIVERGENCE_X_RECLAIM": 1.6,
             "DIVERGENCE_X_DAMAGE": 1.7},
            _ok_kappa_blocks(), {"pearson": -0.30, "spearman": -0.25})
        self.assertEqual(status, "PASS_REPAIR_STATE_X_V002")
        self.assertEqual(reasons, [])

    def test_review_outranks_pass_only_after_semantic_ok(self):
        # 无数值/语义阻塞但 RECLAIM composite 弱 => 仍 PASS (v002 判定门不含
        # RECLAIM_COMPOSITE_REVIEW; 该审查继续由 RECLAIM 内部审计小节记录)
        status, _ = audit_status_v2([], {"OPEN": 1.0}, {"OPEN": 1.0},
                                    _ok_kappa_blocks(),
                                    {"pearson": 0.2, "spearman": 0.1})
        self.assertEqual(status, "PASS_REPAIR_STATE_X_V002")


# ---------------------------------------------------------------------------
# §39 26: deterministic rebuild
# ---------------------------------------------------------------------------

class DeterministicRebuildTests(unittest.TestCase):
    """§39 26: deterministic rebuild — 同一输入跑两遍 6 资产字节级一致."""

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

    def test_synthetic_audit_runs_and_shape(self):
        csv, tmp = synth_audit_csv(rows=59, seed=52)
        out = tmp / "out"
        try:
            r = run_repair_state_audit(csv, out)
            self.assertIn(r["status"],
                          ("PASS_REPAIR_STATE_X_V002", "SEMANTIC_DECOUPLING_REVIEW",
                           "REVIEW_REQUIRED"))
            self.assertEqual(r["june_rows"], 30)
            self.assertEqual(r["july_rows"], 29)
            self.assertEqual(len(list(out.glob("*.csv"))) +
                             len(list(out.glob("*.md"))), 6)
        finally:
            _rmtree(tmp)

    def test_spec_csv_16_rows(self):
        csv, tmp = synth_audit_csv(rows=59, seed=53)
        out = tmp / "out"
        try:
            run_repair_state_audit(csv, out)
            spec = pd.read_csv(out / "v004c_repair_state_spec_v002.csv")
            self.assertEqual(len(spec), 16)  # 8 active + 8 sensitivity
            active_r2 = spec[spec["active_in_r2"] == True]["factor_name"].tolist()  # noqa: E712
            self.assertEqual(sorted(active_r2), sorted(R2_FACTORS))
            turnover_row = spec[spec["factor_name"] == "TURNOVER_COST"].iloc[0]
            self.assertEqual(turnover_row["factor_role"],
                             "SENSITIVITY_TURNOVER_COST")
            divsup_row = spec[spec["factor_name"] == "DIVERGENCE_X_SUPPLY"].iloc[0]
            self.assertEqual(divsup_row["factor_role"], "SENSITIVITY")
        finally:
            _rmtree(tmp)


class RealDataTests(unittest.TestCase):
    """真实 model table 的 June/July 形状与正式输出 (只写临时目录)."""

    MODEL_TABLE = (REPO_ROOT
                   / "reports/research/v004c_model_table_v001_20260601_20260729"
                   / "v004c_model_table_v001.csv")

    def test_real_june_july_shape(self):
        if not self.MODEL_TABLE.exists():
            self.skipTest("model table 不存在")
        with tempfile.TemporaryDirectory(prefix="repair_real_v2_") as tmp:
            r = run_repair_state_audit(self.MODEL_TABLE, pathlib.Path(tmp))
            self.assertEqual(r["june_rows"], 173)
            self.assertEqual(r["june_date_count"], 21)
            self.assertEqual(r["july_rows"], 160)
            self.assertEqual(r["july_date_count"], 21)
            self.assertIn(r["status"],
                          ("PASS_REPAIR_STATE_X_V002", "SEMANTIC_DECOUPLING_REVIEW",
                           "REVIEW_REQUIRED"))

    def test_real_target_never_loaded(self):
        if not self.MODEL_TABLE.exists():
            self.skipTest("model table 不存在")
        x = load_x_table(self.MODEL_TABLE)
        self.assertNotIn("target7_daily_d2open_d3high", x.columns)
        self.assertEqual(len(x), 333)


if __name__ == "__main__":
    unittest.main()

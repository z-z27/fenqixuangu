"""v004c Repair-State 模型 v001 — 纯 X 侧规格 (specification, 不拟合)。

本模块在 v004c_factor_spec.py (v002 历史规格) 之外, 独立冻结
Repair-State Model v001 的数学定义:

- 交易假设: 2板/3板连续强势 -> 首次断板 -> D1 首次明显分歧 -> 判断分歧是否充分
  -> 判断卖压是否仍持续 -> 判断低位是否重新获得承接 -> 预测 D2-D3 强修复
  (Target 仍为 target7_daily_d2open_d3high, 但本模块/本阶段禁止读取 Target)
- 4 个基础 factor: OPEN / DIVERGENCE / SUPPLY / RECLAIM
  - DIVERGENCE 的数学定义与旧 RESET 完全一致 (禁止改变), 只是重新命名与重新解释
  - RECLAIM 是新模型最重要的新增 factor (旧模型没有对应的承接刻画)
- 3 个预声明 derived terms: DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM /
  DIVERGENCE_X_SUPPLY (只做 mean/std 标准化, 不做第二次 clip)
- R1 = 4 base factors + intercept; R2 = 7 factors + intercept
- 6 个旧 factor (HIGHZONE/MOM7/DAMAGE7/REGIME/POS7/TREND) 统一记录为
  LEGACY_SENSITIVITY, 不得进入 R1/R2

本模块是 post-June hypothesis 的数学规格:
Repair-State v001 是在 June walk-forward 失败 (REJECT_NO_STABLE_MODEL) 之后形成的
新模型假设, June 不能再作为该模型的独立验证集。

明确禁止 (与 v004c_factor_spec 一致):
- 任何模型训练 / 预测 / 指标 (无 LogisticRegression / fit / predict / AUC)
- 任何 Target 读取 (本模块只处理 X)
- 全样本预处理 (clip/标准化参数只允许来自 reference 样本)
- 根据诊断自动修改 factor 定义 / composite 权重 / interaction 成员
- 增加任何规格之外的新 interaction (OPEN_X_DIVERGENCE 等均不属于 v001)

transform 约定 (与 v004c_factor_spec 相同 FOLD_CLIP_Z 顺序, 不做第二套实现):
- primitive: 在 reference 样本原始有限值上拟合 q01/q99 -> clip ->
  在 clipped reference 值上拟合 mu/sigma -> (clip(x)-mu)/sigma
- composite: 由 standardized primitive 构造 G -> 在 reference 上拟合 G 的 mu/sigma
  -> z(G) (DIVERGENCE 复用 factor_spec.composite_reset, 公式零复制)
- derived: 由标准化 base factor 构造 raw (F_D^2 / F_D*F_Q / F_D*F_S) ->
  在 reference 上拟合 raw 的 mu/sigma -> 标准化; 不做第二次 clip
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.v004c_factor_spec import (
    FORBIDDEN_TOKENS,   # re-export: target-blind 读取约定
    X_ID_COLUMNS,       # re-export: 3 个标识列约定
    DEGENERATE_STD,
    FactorSpecError,
    composite_reset,    # DIVERGENCE 与旧 RESET 共用同一公式函数 (公式零复制)
)

__all__ = [
    "REPAIR_PRIMITIVES", "X_ID_COLUMNS", "FORBIDDEN_TOKENS",
    "BASE_FACTORS", "DERIVED_FACTORS", "R1_FACTORS", "R2_FACTORS",
    "LEGACY_SENSITIVITY_FACTORS", "RECLAIM_PRIMITIVES",
    "DIVERGENCE_SIGNS", "DIVERGENCE_WEIGHTS", "RECLAIM_WEIGHTS",
    "DERIVED_PARENTS", "HIERARCHY_EXPECTED_PAIRS",
    "DERIVED_TAIL_MAX_ABS", "RECLAIM_WEAK_CORR", "VIF_WATCH", "VIF_SEVERE",
    "KAPPA_WATCH", "KAPPA_SEVERE", "SMD_WATCH", "KS_WATCH",
    "RepairStateError",
    "fit_repair_transform", "apply_repair_primitive_transform",
    "composite_reclaim", "construct_repair_factors",
    "repair_spec_rows", "audit_status",
]


class RepairStateError(FactorSpecError):
    """Repair-State 规格/审计失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# primitive 常量 (本模型只使用这 7 个 primitive)
# ---------------------------------------------------------------------------

REPAIR_PRIMITIVES: tuple[str, ...] = (
    "break_open_return",
    "d1_open_to_close_return_raw",
    "d1_high_to_close_drawdown_raw",
    "d1_close_to_vwap_raw",
    "late_day_sell_volume_ratio",
    "d1_low_to_close_recovery",
    "d1_afternoon_return",
)

# X 读取入口只允许 3 个标识列 + 7 个 primitive (target-blind 约定, 同 factor audit)
# FORBIDDEN_TOKENS 由 v004c_factor_spec 提供 (target/d2_/d3_/future/...)

RECLAIM_PRIMITIVES: tuple[str, ...] = (
    "d1_low_to_close_recovery",
    "d1_afternoon_return",
)

# ---------------------------------------------------------------------------
# factor 常量 (R1 = 4 base; R2 = 7)
# ---------------------------------------------------------------------------

BASE_FACTORS: tuple[str, ...] = ("OPEN", "DIVERGENCE", "SUPPLY", "RECLAIM")
DERIVED_FACTORS: tuple[str, ...] = (
    "DIVERGENCE_SQ",
    "DIVERGENCE_X_RECLAIM",
    "DIVERGENCE_X_SUPPLY",
)
R1_FACTORS: tuple[str, ...] = BASE_FACTORS
R2_FACTORS: tuple[str, ...] = BASE_FACTORS + DERIVED_FACTORS
FACTOR_NAMES: tuple[str, ...] = R2_FACTORS

# 旧 factor 统一记录为 LEGACY_SENSITIVITY (禁止进入 R1/R2 core)
LEGACY_SENSITIVITY_FACTORS: tuple[str, ...] = (
    "HIGHZONE", "MOM7", "DAMAGE7", "REGIME", "POS7", "TREND",
)

# factor -> 组成 primitive (顺序即权重/符号顺序)
FACTOR_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "OPEN": ("break_open_return",),
    "DIVERGENCE": ("d1_open_to_close_return_raw",
                   "d1_high_to_close_drawdown_raw",
                   "d1_close_to_vwap_raw"),
    "SUPPLY": ("late_day_sell_volume_ratio",),
    "RECLAIM": ("d1_low_to_close_recovery", "d1_afternoon_return"),
    "DIVERGENCE_SQ": ("DIVERGENCE",),
    "DIVERGENCE_X_RECLAIM": ("DIVERGENCE", "RECLAIM"),
    "DIVERGENCE_X_SUPPLY": ("DIVERGENCE", "SUPPLY"),
}

# DIVERGENCE: G = (-z_OC + z_HC - z_VWAP) / 3 —— 与旧 RESET 数学定义完全一致
# (权重固定, 禁止修改; 公式函数直接复用 factor_spec.composite_reset)
DIVERGENCE_SIGNS: tuple[float, ...] = (-1.0, +1.0, -1.0)
DIVERGENCE_WEIGHTS: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
# RECLAIM: G = (z_low_to_close_recovery + z_afternoon_return) / 2 (等权, 固定)
RECLAIM_WEIGHTS: tuple[float, ...] = (0.5, 0.5)

# derived hierarchy: derived factor -> 其父 factor (R2 必须同时包含全部 parents)
DERIVED_PARENTS: dict[str, tuple[str, ...]] = {
    "DIVERGENCE_SQ": ("DIVERGENCE",),
    "DIVERGENCE_X_RECLAIM": ("DIVERGENCE", "RECLAIM"),
    "DIVERGENCE_X_SUPPLY": ("DIVERGENCE", "SUPPLY"),
}
# 天然层级依赖 pair (derived 与其 parent); 高相关属 EXPECTED_HIERARCHICAL_DEPENDENCE
HIERARCHY_EXPECTED_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (p, d) for d, parents in DERIVED_PARENTS.items() for p in parents)

# 诊断解释阈值 (只解释, 禁止程序自动删除/换因子)
VIF_WATCH, VIF_SEVERE = 5.0, 10.0
KAPPA_WATCH, KAPPA_SEVERE = 30.0, 100.0
SMD_WATCH, KS_WATCH = 0.50, 0.25
RECLAIM_WEAK_CORR = 0.10          # |rho| 双低 => RECLAIM_COMPOSITE_REVIEW
DERIVED_TAIL_MAX_ABS = 5.0        # June 标准化后 max|raw| 上限 => DERIVED_TAIL_WATCH

DIVERGENCE_COMPOSITE_KEY = "composite_DIVERGENCE"
RECLAIM_COMPOSITE_KEY = "composite_RECLAIM"


# ---------------------------------------------------------------------------
# reference transform 纯函数 (与 v004c_factor_spec 相同 FOLD_CLIP_Z 顺序)
# ---------------------------------------------------------------------------

def fit_repair_transform(reference_df: pd.DataFrame) -> dict:
    """仅用 reference 样本 (本任务为 June) 拟合全部 transform 参数。

    与 v004c_factor_spec.fit_reference_transform 完全相同的 FOLD_CLIP_Z 顺序
    (① q01/q99 在 reference 原始有限值上拟合 -> ② clip -> ③ mu/sigma 在
    clipped reference 值上拟合), 只作用于本模型的 7 个 primitive。

    然后:
    - composite DIVERGENCE / RECLAIM: 由 standardized primitive 构造 G,
      在 reference 上拟合 G 的 mu/sigma
    - 3 个 derived terms: 由标准化 base factor 构造 raw (F_D^2 / F_D*F_Q /
      F_D*F_S), 在 reference 上拟合 raw 的 mu/sigma (无 q01/q99 => 不做
      第二次 clip)

    任何退化 (sigma <= 1e-8) 直接失败, 不静默。
    """
    if len(reference_df) == 0:
        raise RepairStateError("reference transform 需要非空 reference 样本")
    params: dict = {}
    for col in REPAIR_PRIMITIVES:
        x = pd.to_numeric(reference_df[col], errors="coerce").to_numpy(dtype=float)
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            raise RepairStateError(f"reference 中 primitive {col} 无有限值")
        q01, q99 = np.quantile(finite, [0.01, 0.99])
        clipped = np.clip(finite, q01, q99)
        mu = float(clipped.mean())
        sigma = float(clipped.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= DEGENERATE_STD:
            raise RepairStateError(
                f"reference 中 primitive {col} 退化 (sigma={sigma}, 禁止继续)")
        params[col] = {"q01": float(q01), "q99": float(q99), "mu": mu, "sigma": sigma}

    z = apply_repair_primitive_transform(reference_df, params)

    # composite DIVERGENCE (公式与旧 RESET 完全一致, 复用 factor_spec.composite_reset)
    for key, g in ((DIVERGENCE_COMPOSITE_KEY, composite_reset(z)),
                   (RECLAIM_COMPOSITE_KEY, composite_reclaim(z))):
        finite = g[np.isfinite(g)]
        if len(finite) == 0:
            raise RepairStateError(f"reference 中 composite {key} 无有限值")
        mu = float(finite.mean())
        sigma = float(finite.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= DEGENERATE_STD:
            raise RepairStateError(
                f"reference 中 composite {key} 退化 (sigma={sigma}, 禁止继续)")
        params[key] = {"mu": mu, "sigma": sigma}

    # 3 个 derived terms: 由标准化 base factor 构造 raw, 只 fit mean/std
    base = _base_factors(reference_df, params)
    for name, raw in _derived_raw(base):
        finite = raw[np.isfinite(raw)]
        if len(finite) == 0:
            raise RepairStateError(f"reference 中 derived {name} 无有限值")
        mu = float(finite.mean())
        sigma = float(finite.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= DEGENERATE_STD:
            raise RepairStateError(
                f"reference 中 derived {name} 退化 (sigma={sigma}, 禁止继续)")
        params[name] = {"mu": mu, "sigma": sigma}  # 无 q01/q99: derived 不做第二次 clip
    return params


def apply_repair_primitive_transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """用给定参数 (只允许 reference 参数) 构造 reference standardized primitive。

    与 v004c_factor_spec.apply_primitive_transform 相同的 clip/z 应用逻辑
    ((clip(x, q01, q99) - mu) / sigma), 只作用于本模型的 7 个 primitive。
    July 不得用自身参数重算。
    """
    out: dict[str, np.ndarray] = {}
    for col in REPAIR_PRIMITIVES:
        p = params[col]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        out[col] = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
    return pd.DataFrame(out, index=df.index)


def composite_reclaim(z: pd.DataFrame) -> np.ndarray:
    """G_RECLAIM = (z_low_to_close_recovery + z_afternoon_return) / 2 (等权, 固定)."""
    return (z["d1_low_to_close_recovery"] + z["d1_afternoon_return"]) / 2.0


def _base_factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """4 个基础 factor (OPEN/DIVERGENCE/SUPPLY/RECLAIM), 全部 reference 参数."""
    z = apply_repair_primitive_transform(df, params)
    c_d = params[DIVERGENCE_COMPOSITE_KEY]
    c_q = params[RECLAIM_COMPOSITE_KEY]
    return pd.DataFrame({
        "OPEN": z["break_open_return"],
        "DIVERGENCE": (composite_reset(z) - c_d["mu"]) / c_d["sigma"],
        "SUPPLY": z["late_day_sell_volume_ratio"],
        "RECLAIM": (composite_reclaim(z) - c_q["mu"]) / c_q["sigma"],
    }, index=df.index)


def _derived_raw(base: pd.DataFrame):
    """3 个 derived 的 raw 值 (由标准化 base factor 构造, 未标准化)."""
    return (
        ("DIVERGENCE_SQ", base["DIVERGENCE"] ** 2),
        ("DIVERGENCE_X_RECLAIM", base["DIVERGENCE"] * base["RECLAIM"]),
        ("DIVERGENCE_X_SUPPLY", base["DIVERGENCE"] * base["SUPPLY"]),
    )


def construct_repair_factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """构造 7 个 factor (June/July 均用同一组 reference 参数).

    - 基础 factor: 同 _base_factors
    - derived: (raw - mu) / sigma, raw 由标准化 base factor 构造;
      不经过任何 clip (derived 不做第二次 q01/q99 裁剪)
    - 任何非有限连续 factor => 显式失败
    """
    base = _base_factors(df, params)
    factors = base.copy()
    for name, raw in _derived_raw(base):
        p = params[name]
        factors[name] = (raw - p["mu"]) / p["sigma"]
    for col in FACTOR_NAMES:
        if not np.all(np.isfinite(factors[col])):
            raise RepairStateError(f"factor {col} 包含非有限值")
    return factors


# ---------------------------------------------------------------------------
# 静态规格表 (spec CSV / model spec md 的数据源)
# ---------------------------------------------------------------------------

REPAIR_SPEC_FIELDS: tuple[str, ...] = (
    "factor_name", "factor_role", "factor_type", "formula",
    "primitive_dependencies", "active_in_r1", "active_in_r2",
    "economic_meaning", "preprocessing",
)

_PRIM_PREPROCESS = ("reference 样本内拟合 (FOLD_CLIP_Z 顺序: 在 reference 原始值上拟合 "
                    "q01/q99 -> clip -> 在 clipped reference 值上拟合 mu/sigma) + "
                    "(clip(x)-mu)/sigma (reference 参数; 未来正式 fold 中参数只来自 "
                    "当前 training fold, 禁止全样本参数)")
_COMPOSITE_PREPROCESS = ("先对各 primitive 执行 FOLD_CLIP_Z (reference 参数), 再构造 G, "
                         "然后在 reference 上拟合 G 的 mu/sigma -> z(G)")
_DERIVED_PREPROCESS = ("由标准化 base factor 构造 raw, 在 reference 上拟合 raw 的 "
                       "mu/sigma -> 标准化; 不做第二次 clip (规格明确禁止对 derived "
                       "再施加第二套 q01/q99 裁剪)")


def repair_spec_rows() -> list[dict]:
    """13 行静态规格: 7 个 active factor (R1/R2) + 6 个 LEGACY_SENSITIVITY.

    所有公式不读取 Target (target_used_to_construct 恒为 False 语义)。
    """
    rows: list[dict] = []

    def base_row(name, role, ftype, formula, prims, econ, pre, r1, r2):
        rows.append({
            "factor_name": name,
            "factor_role": role,
            "factor_type": ftype,
            "formula": formula,
            "primitive_dependencies": "|".join(prims),
            "active_in_r1": r1,
            "active_in_r2": r2,
            "economic_meaning": econ,
            "preprocessing": pre,
        })

    base_row(
        "OPEN", "CORE 基础因子 (单 primitive)", "continuous_standardized",
        "F_O = z(break_open_return)",
        FACTOR_PRIMITIVES["OPEN"],
        ("D1 开盘时, 前面 2/3 板形成的资金共识和隔夜认可是否仍存在 (overnight demand "
         "/ opening strength)。不假设 OPEN 越高一定越好, 方向留给未来模型估计。"),
        _PRIM_PREPROCESS, True, True)
    base_row(
        "DIVERGENCE", "CORE 基础因子 (3 primitive composite)", "continuous_standardized",
        ("G_D = (-z(d1_open_to_close_return_raw) + z(d1_high_to_close_drawdown_raw) "
         "- z(d1_close_to_vwap_raw)) / 3; F_D = z(G_D)"),
        FACTOR_PRIMITIVES["DIVERGENCE"],
        ("D1 首次断板后价格分歧的充分程度 (price reset intensity)。只描述'跌下来多少 / "
         "价格重置多充分', 不描述'跌下来以后有没有被资金重新买回去'。与旧 RESET 数学定义"
         "完全一致 (权重 -1/3, +1/3, -1/3 固定, 禁止修改), 只是重新命名与重新解释。"),
        _COMPOSITE_PREPROCESS, True, True)
    base_row(
        "SUPPLY", "CORE 基础因子 (单 primitive)", "continuous_standardized",
        "F_S = z(late_day_sell_volume_ratio)",
        FACTOR_PRIMITIVES["SUPPLY"],
        ("D1 进入后半段以后, 主动供应/卖压是否仍然持续存在。只使用 "
         "late_day_sell_volume_ratio 单独作为 SUPPLY; 不重新使用旧的 "
         "HIGHZONE + LATESELL composite。"),
        _PRIM_PREPROCESS, True, True)
    base_row(
        "RECLAIM", "CORE 基础因子 (2 primitive composite)", "continuous_standardized",
        ("G_Q = (z(d1_low_to_close_recovery) + z(d1_afternoon_return)) / 2; "
         "F_Q = z(G_Q)"),
        FACTOR_PRIMITIVES["RECLAIM"],
        ("D1 盘中出现低点和分歧以后, 资金是否重新形成承接, 并把价格从低位收回来。"
         "RECLAIM 必须与 DIVERGENCE 明确分开: DIVERGENCE 只描述分歧程度, RECLAIM "
         "描述分歧后是否被资金买回。第一版只使用这两个 primitive, 不加入 "
         "d1_last_hour_return / d1_close_location / d1_true_reclaim_ma5 / "
         "d1_true_reclaim_ma10 / d1_up_bar_volume_ratio (可进入未来 sensitivity)。"),
        _COMPOSITE_PREPROCESS, True, True)
    base_row(
        "DIVERGENCE_SQ", "CORE 派生项 (预声明非线性项)", "continuous_derived_standardized",
        "H = F_D^2; F_D2 = z(H)",
        FACTOR_PRIMITIVES["DIVERGENCE_SQ"],
        ("允许未来 Logistic 表达'分歧存在最佳区间', 而不是强制 DIVERGENCE 无限单调 "
         "(分歧不足 -> 不充分; 适中分歧 -> 健康换手; 极端分歧 -> 可能结构破坏)。"
         "禁止根据 Target 决定平方项方向。"),
        _DERIVED_PREPROCESS, False, True)
    base_row(
        "DIVERGENCE_X_RECLAIM", "CORE 派生项 (预声明条件项)",
        "continuous_derived_standardized",
        "H = F_D * F_Q; F_DQ = z(H)",
        FACTOR_PRIMITIVES["DIVERGENCE_X_RECLAIM"],
        ("Repair-State 模型的核心条件项: 区分'D高+Q高 = 充分分歧 + 明显重新承接'与"
         "'D高+Q低 = 价格破坏 + 缺乏承接'。"),
        _DERIVED_PREPROCESS, False, True)
    base_row(
        "DIVERGENCE_X_SUPPLY", "CORE 派生项 (预声明条件项)",
        "continuous_derived_standardized",
        "H = F_D * F_S; F_DS = z(H)",
        FACTOR_PRIMITIVES["DIVERGENCE_X_SUPPLY"],
        ("同样发生充分分歧时, 尾盘供应是否仍然持续: 区分'D高+S低 = 充分分歧 + 卖压衰竭'"
         "与'D高+S高 = 充分分歧 + 尾盘仍持续释放'。"),
        _DERIVED_PREPROCESS, False, True)

    # LEGACY_SENSITIVITY (6 个旧 factor, 禁止进入 R1/R2; 历史定义不删除)
    legacy = (
        ("HIGHZONE", "F_HIGHZONE = z(high_zone_volume_ratio)",
         ("高位筹码堆积。其意义明显依赖价格结果/承接状态, 第一版不再作为独立线性 "
          "core factor。")),
        ("MOM7", "F_MOM7 = z(recent_7d_cumulative_return)",
         ("最近 7 日整体价格扩张。旧 M2 加入 path layer 以后 June OOF 概率质量系统性"
          "恶化, 本版本不再作为核心 Alpha 输入。")),
        ("DAMAGE7", "F_DAMAGE7 = z(recent_7d_max_drawdown)",
         ("最近 7 日路径峰值破坏。同 MOM7, 不再作为核心 Alpha 输入。")),
        ("REGIME", "F_REGIME = board_streak_is_3 (0/1 原值, 不标准化)",
         ("二/三板 regime 控制。同 MOM7/DAMAGE7, 不再作为核心 Alpha 输入。")),
        ("POS7", "F_POS7 = z(recent_7d_close_position)",
         ("沿用此前 sensitivity 身份 (与 RESET 结构重复, v002 决策)。")),
        ("TREND", "F_TREND = z(d1_ma10_slope)",
         ("沿用此前 sensitivity 身份 (与 MOM7 高度相关 + June→July shift, v002 决策)。")),
    )
    for name, formula, econ in legacy:
        base_row(
            name, "LEGACY_SENSITIVITY", "continuous_standardized",
            formula, (name,), econ,
            "历史定义保留在 v004c_factor_spec.py (v002 规格), 本模块不复制公式实现",
            False, False)
    return rows


# ---------------------------------------------------------------------------
# 正式判定门 (§32; 只用 X 结构 / 数值稳定性 / June→July X 稳定性, 与 Target 无关)
# ---------------------------------------------------------------------------

def audit_status(degenerate_factors: list[str],
                 vif_r1: dict[str, float],
                 vif_r2: dict[str, float],
                 kappa_blocks: dict[str, dict],
                 reclaim_weak: bool) -> tuple[str, list[str]]:
    """返回 (status, reasons)。

    - degenerate factor / VIF >= 10 / condition number >= 100 => REVIEW_REQUIRED
    - 否则若 RECLAIM 两 primitive 在 June 中 |Pearson| < 0.10 且 |Spearman| < 0.10
      => RECLAIM_COMPOSITE_REVIEW (进入人工审查, 不自动拆 RECLAIM)
    - 否则 PASS_REPAIR_STATE_X_V001
    - SHIFT_WATCH / DERIVED_TAIL_WATCH / hierarchy 相关只记录, 不进入本判定门
    """
    reasons: list[str] = []
    if degenerate_factors:
        reasons.append("degenerate factor: {} (June std <= 1e-8)".format(
            ", ".join(degenerate_factors)))
    for block, vifs in (("R1", vif_r1), ("R2", vif_r2)):
        for f, v in vifs.items():
            if v >= VIF_SEVERE:
                reasons.append(f"{block} factor {f} VIF={v:.4g} >= 10 SEVERE")
    for name, k in kappa_blocks.items():
        if k["near_singular"] or k["kappa"] >= KAPPA_SEVERE:
            reasons.append(
                f"{name} condition matrix near singular or kappa="
                f"{k['kappa']:.3g} >= 100")
    if reasons:
        return "REVIEW_REQUIRED", reasons
    if reclaim_weak:
        return "RECLAIM_COMPOSITE_REVIEW", [
            "RECLAIM 两个 primitive (d1_low_to_close_recovery / d1_afternoon_return) "
            "在 June 中 |Pearson| < 0.10 且 |Spearman| < 0.10 => 人工审查 composite "
            "是否合理 (禁止自动拆分 RECLAIM)"]
    return "PASS_REPAIR_STATE_X_V001", []

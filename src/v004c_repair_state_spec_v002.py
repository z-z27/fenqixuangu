"""v004c Repair-State 模型 v002 — 语义修订后的纯 X 侧规格 (specification, 不拟合)。

本模块是 Repair-State v001 (src/v004c_repair_state_spec.py) 的人工语义修订,
v001 文件本身禁止修改。修订动机 (人工复核):

- v001 DIVERGENCE 沿用旧 RESET ((-z_OC + z_HC - z_VWAP)/3), 主要描述的是
  "D1 最终收盘留下多少价格损伤", 而不是 "D1 盘中发生过多大的首次分歧";
  且 Corr(DIVERGENCE, RECLAIM) June Pearson = -0.7340 / Spearman = -0.7725,
  primitive 层面存在明显机械反向关系
- v002 把 "盘中分歧强度" 与 "最终价格损伤" 拆成两个独立 factor:
  - DIVERGENCE  = z(d1_intraday_range): D1 盘中实际发生过多大的首次价格分歧
    (过程强度, 只描述价格探索范围, 不因收盘收回而自动消失)
  - CLOSE_DAMAGE = z(G_C): 分歧结束以后, D1 收盘最终留下多少结构损伤
    (G_C = (-z_OC + z_HC - z_VWAP)/3, 与旧 RESET 数学定义完全相同,
    仅重新命名与重新解释)

模型结构 (v002):

- 5 个基础 factor: OPEN / DIVERGENCE / CLOSE_DAMAGE / SUPPLY / RECLAIM
  - OPEN = z(break_open_return)                      (与 v001 相同)
  - DIVERGENCE = z(d1_intraday_range)                (新: 单 primitive, 过程强度)
  - CLOSE_DAMAGE = z(G_C)                            (旧 RESET 数学, 损伤)
  - SUPPLY = z(late_day_sell_volume_ratio)           (与 v001 相同)
  - RECLAIM = z(G_Q)                                 (与 v001 相同, 公式冻结)
- 3 个预声明 derived terms (R2 唯一允许的非线性项):
  DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM / DIVERGENCE_X_DAMAGE
  (只做 mean/std 标准化, 不做第二次 clip)
- R1 = 5 base factors + intercept; R2 = 8 factors + intercept
- 8 个 sensitivity (禁止进入 R1/R2):
  - 6 个 LEGACY_SENSITIVITY (HIGHZONE/MOM7/DAMAGE7/REGIME/POS7/TREND, 历史定义保留)
  - DIVERGENCE_X_SUPPLY (v001 derived, 本版降为 SENSITIVITY, 定义保留)
  - TURNOVER_COST (SENSITIVITY_TURNOVER_COST = z(break_volume_ratio_vs_board_days),
    仅预声明; 禁止构造 DIV_X_TURNOVER)

核心交易逻辑:
2/3板强势确认 -> D1 首次断板 -> 发生分歧 -> 分歧不能不足也不能过度透支 ->
最终损伤必须可控 -> 尾盘供应最好衰竭 -> 分歧后需要出现真实重新承接 -> D2-D3 强修复。

DIVERGENCE != CLOSE_DAMAGE != RECLAIM: 三者是不同阶段 (过程分歧 / 最终损伤 /
弱点后回收)。目标不是让三者完全统计独立, 但必须避免 v001 那种
"DIVERGENCE 本身就是 RECLAIM 的机械反面" (语义解耦门 §27)。

本模块是 post-June hypothesis 的数学规格 (同 v001):
Repair-State 是 June walk-forward 失败 (REJECT_NO_STABLE_MODEL) 之后形成的
新模型假设, June 不能再作为该模型的独立验证集。

明确禁止 (与 v001 一致):
- 任何模型训练 / 预测 / 指标 (无 LogisticRegression / fit / predict / AUC)
- 任何 Target 读取 (本模块只处理 X)
- 全样本预处理 (clip/标准化参数只允许来自 reference 样本)
- 根据诊断自动修改 factor 定义 / composite 权重 / interaction 成员
- 增加任何规格之外的新 interaction (OPEN_X_DIVERGENCE / DIV_X_TURNOVER 等)

transform 约定 (与 v001 相同 FOLD_CLIP_Z 顺序):
- primitive: 在 reference 样本原始有限值上拟合 q01/q99 -> clip ->
  在 clipped reference 值上拟合 mu/sigma -> (clip(x)-mu)/sigma
- composite (CLOSE_DAMAGE / RECLAIM): 由 standardized primitive 构造 G ->
  在 reference 上拟合 G 的 mu/sigma -> z(G)
  (CLOSE_DAMAGE 复用 factor_spec.composite_reset, 公式零复制;
   RECLAIM 复用 v001 的 composite_reclaim, 保证与 v001 逐行相等)
- derived: 由标准化 base factor 构造 raw (F_V^2 / F_V*F_Q / F_V*F_C) ->
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
    composite_reset,    # CLOSE_DAMAGE 与旧 RESET 共用同一公式函数 (公式零复制)
)
from src.v004c_repair_state_spec import (
    composite_reclaim,  # RECLAIM 公式与 v001 共用 (公式零复制, 保证逐行相等)
)

__all__ = [
    "REPAIR_PRIMITIVES", "SENSITIVITY_PRIMITIVES", "X_ID_COLUMNS",
    "FORBIDDEN_TOKENS",
    "DIVERGENCE_PRIMITIVE", "CLOSE_DAMAGE_PRIMITIVES",
    "BASE_FACTORS", "DERIVED_FACTORS", "R1_FACTORS", "R2_FACTORS",
    "FACTOR_PRIMITIVES",
    "LEGACY_SENSITIVITY_FACTORS", "SENSITIVITY_FACTORS",
    "DERIVED_PARENTS", "HIERARCHY_EXPECTED_PAIRS",
    "SEMANTIC_STATE_PAIRS",
    "CLOSE_DAMAGE_SIGNS", "CLOSE_DAMAGE_WEIGHTS", "RECLAIM_WEIGHTS",
    "SEMANTIC_DECOUPLING_WATCH_CORR", "SEMANTIC_DECOUPLING_REVIEW_CORR",
    "DERIVED_TAIL_MAX_ABS", "NONLINEARITY_NARROW_P95P05",
    "VIF_WATCH", "VIF_SEVERE", "KAPPA_WATCH", "KAPPA_SEVERE",
    "SMD_WATCH", "KS_WATCH",
    "RepairStateError",
    "fit_repair_state_transform_v2", "apply_repair_primitive_transform_v2",
    "construct_repair_factors_v2",
    "repair_spec_rows_v2", "audit_status_v2",
    "semantic_decoupling_status",
]


class RepairStateError(FactorSpecError):
    """Repair-State v002 规格/审计失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# primitive 常量 (本模型 core 只使用这 8 个 primitive)
# ---------------------------------------------------------------------------

REPAIR_PRIMITIVES: tuple[str, ...] = (
    "break_open_return",
    "d1_intraday_range",
    "d1_open_to_close_return_raw",
    "d1_high_to_close_drawdown_raw",
    "d1_close_to_vwap_raw",
    "late_day_sell_volume_ratio",
    "d1_low_to_close_recovery",
    "d1_afternoon_return",
)

# TURNOVER_COST sensitivity 的 primitive (只记录, 禁止构造 DIV_X_TURNOVER)
SENSITIVITY_PRIMITIVES: tuple[str, ...] = ("break_volume_ratio_vs_board_days",)

# DIVERGENCE 的唯一合法构造源 (§4/§7: 只允许 D1 盘中 high-low range 类 primitive;
# 禁止使用 open_to_close / high_to_close / close_to_vwap 直接定义 DIVERGENCE)
DIVERGENCE_PRIMITIVE: str = "d1_intraday_range"

# CLOSE_DAMAGE 的 3 个 primitive (与旧 RESET 相同)
CLOSE_DAMAGE_PRIMITIVES: tuple[str, ...] = (
    "d1_open_to_close_return_raw",
    "d1_high_to_close_drawdown_raw",
    "d1_close_to_vwap_raw",
)

RECLAIM_PRIMITIVES: tuple[str, ...] = (
    "d1_low_to_close_recovery",
    "d1_afternoon_return",
)

# X 读取入口只允许 3 个标识列 + 8 个 core primitive (+ 可选 sensitivity 列)
# FORBIDDEN_TOKENS 由 v004c_factor_spec 提供 (target/d2_/d3_/future/...)

# ---------------------------------------------------------------------------
# factor 常量 (R1 = 5 base; R2 = 8)
# ---------------------------------------------------------------------------

BASE_FACTORS: tuple[str, ...] = (
    "OPEN", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY", "RECLAIM",
)
DERIVED_FACTORS: tuple[str, ...] = (
    "DIVERGENCE_SQ",
    "DIVERGENCE_X_RECLAIM",
    "DIVERGENCE_X_DAMAGE",
)
R1_FACTORS: tuple[str, ...] = BASE_FACTORS
R2_FACTORS: tuple[str, ...] = BASE_FACTORS + DERIVED_FACTORS
FACTOR_NAMES: tuple[str, ...] = R2_FACTORS

# 旧 factor 统一记录为 LEGACY_SENSITIVITY (禁止进入 R1/R2 core)
LEGACY_SENSITIVITY_FACTORS: tuple[str, ...] = (
    "HIGHZONE", "MOM7", "DAMAGE7", "REGIME", "POS7", "TREND",
)

# 全部 sensitivity (禁止进入 R1/R2):
# 6 个 legacy + v001 derived DIVERGENCE_X_SUPPLY (本版降级) + TURNOVER_COST
SENSITIVITY_FACTORS: tuple[str, ...] = LEGACY_SENSITIVITY_FACTORS + (
    "DIVERGENCE_X_SUPPLY",
    "TURNOVER_COST",
)

# factor -> 组成 primitive (顺序即权重/符号顺序)
FACTOR_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "OPEN": ("break_open_return",),
    "DIVERGENCE": (DIVERGENCE_PRIMITIVE,),           # 只允许单 primitive range
    "CLOSE_DAMAGE": CLOSE_DAMAGE_PRIMITIVES,
    "SUPPLY": ("late_day_sell_volume_ratio",),
    "RECLAIM": RECLAIM_PRIMITIVES,
    "DIVERGENCE_SQ": ("DIVERGENCE",),
    "DIVERGENCE_X_RECLAIM": ("DIVERGENCE", "RECLAIM"),
    "DIVERGENCE_X_DAMAGE": ("DIVERGENCE", "CLOSE_DAMAGE"),
}

# CLOSE_DAMAGE: G_C = (-z_OC + z_HC - z_VWAP) / 3 —— 与旧 RESET 数学定义完全一致
# (权重固定, 禁止修改; 公式函数直接复用 factor_spec.composite_reset)
CLOSE_DAMAGE_SIGNS: tuple[float, ...] = (-1.0, +1.0, -1.0)
CLOSE_DAMAGE_WEIGHTS: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
# RECLAIM: G = (z_low_to_close_recovery + z_afternoon_return) / 2 (等权, 固定, 冻结)
RECLAIM_WEIGHTS: tuple[float, ...] = (0.5, 0.5)

# derived hierarchy: derived factor -> 其父 factor (R2 必须同时包含全部 parents)
DERIVED_PARENTS: dict[str, tuple[str, ...]] = {
    "DIVERGENCE_SQ": ("DIVERGENCE",),
    "DIVERGENCE_X_RECLAIM": ("DIVERGENCE", "RECLAIM"),
    "DIVERGENCE_X_DAMAGE": ("DIVERGENCE", "CLOSE_DAMAGE"),
}
# 天然层级依赖 pair (derived 与其 parent); 高相关属 EXPECTED_HIERARCHICAL_DEPENDENCE
HIERARCHY_EXPECTED_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (p, d) for d, parents in DERIVED_PARENTS.items() for p in parents)

# 三个状态 factor 之间的经济学相关对 (§28/§29: 允许相关, 只记录
# SEMANTIC_STATE_DEPENDENCE, 不设硬失败; 由 VIF/kappa 整体判断)
SEMANTIC_STATE_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("DIVERGENCE", "RECLAIM"),
    ("DIVERGENCE", "CLOSE_DAMAGE"),
    ("CLOSE_DAMAGE", "RECLAIM"),
})

# 语义解耦门 (§27, 只用于 DIVERGENCE vs RECLAIM):
# |corr| >= 0.70 => SEMANTIC_DECOUPLING_REVIEW (人工语义阻塞)
# 0.50 <= |corr| < 0.70 => SEMANTIC_DECOUPLING_WATCH (只记录)
# |corr| < 0.50 => SEMANTIC_DECOUPLING_OK (记录)
SEMANTIC_DECOUPLING_WATCH_CORR = 0.50
SEMANTIC_DECOUPLING_REVIEW_CORR = 0.70

# 诊断解释阈值 (只解释, 禁止程序自动删除/换因子)
VIF_WATCH, VIF_SEVERE = 5.0, 10.0
KAPPA_WATCH, KAPPA_SEVERE = 30.0, 100.0
SMD_WATCH, KS_WATCH = 0.50, 0.25
DERIVED_TAIL_MAX_ABS = 5.0          # June 标准化后 max|raw| 上限 => DERIVED_TAIL_WATCH
NONLINEARITY_NARROW_P95P05 = 0.50   # June DIVERGENCE p95-p05 下限 =>
                                    # DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW (只记录)

CLOSE_DAMAGE_COMPOSITE_KEY = "composite_CLOSE_DAMAGE"
RECLAIM_COMPOSITE_KEY = "composite_RECLAIM"

# 正式状态常量
PASS_REPAIR_STATE_X_V002 = "PASS_REPAIR_STATE_X_V002"
SEMANTIC_DECOUPLING_REVIEW = "SEMANTIC_DECOUPLING_REVIEW"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH = "BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH"


# ---------------------------------------------------------------------------
# reference transform 纯函数 (与 v001 相同 FOLD_CLIP_Z 顺序)
# ---------------------------------------------------------------------------

def fit_repair_state_transform_v2(reference_df: pd.DataFrame) -> dict:
    """仅用 reference 样本 (本任务为 June) 拟合全部 transform 参数。

    与 v001 fit_repair_transform 完全相同的 FOLD_CLIP_Z 顺序
    (① q01/q99 在 reference 原始有限值上拟合 -> ② clip -> ③ mu/sigma 在
    clipped reference 值上拟合), 只作用于本模型的 8 个 core primitive。

    然后:
    - composite CLOSE_DAMAGE / RECLAIM: 由 standardized primitive 构造 G,
      在 reference 上拟合 G 的 mu/sigma (CLOSE_DAMAGE 复用
      factor_spec.composite_reset; RECLAIM 复用 v001 composite_reclaim)
    - 3 个 derived terms: 由标准化 base factor 构造 raw (F_V^2 / F_V*F_Q /
      F_V*F_C), 在 reference 上拟合 raw 的 mu/sigma (无 q01/q99 => 不做
      第二次 clip)

    DIVERGENCE 构造源必须且只能是 DIVERGENCE_PRIMITIVE (d1_intraday_range);
    否则直接失败 (BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH 语义的防御)。

    任何退化 (sigma <= 1e-8) 直接失败, 不静默。
    """
    if len(reference_df) == 0:
        raise RepairStateError("reference transform 需要非空 reference 样本")
    if FACTOR_PRIMITIVES["DIVERGENCE"] != (DIVERGENCE_PRIMITIVE,):
        raise RepairStateError(
            "DIVERGENCE 构造源必须是单一 primitive {} "
            "(BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH)".format(DIVERGENCE_PRIMITIVE))

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

    z = apply_repair_primitive_transform_v2(reference_df, params)

    # composite CLOSE_DAMAGE (公式与旧 RESET 完全一致, 复用 factor_spec.composite_reset)
    # composite RECLAIM (公式与 v001 完全一致, 复用 v001 composite_reclaim)
    for key, g in ((CLOSE_DAMAGE_COMPOSITE_KEY, composite_reset(z)),
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
    base = _base_factors_v2(reference_df, params)
    for name, raw in _derived_raw_v2(base):
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


def apply_repair_primitive_transform_v2(df: pd.DataFrame,
                                        params: dict) -> pd.DataFrame:
    """用给定参数 (只允许 reference 参数) 构造 reference standardized primitive。

    与 v001 apply_repair_primitive_transform 相同的 clip/z 应用逻辑
    ((clip(x, q01, q99) - mu) / sigma), 只作用于本模型的 8 个 core primitive。
    July 不得用自身参数重算。
    """
    out: dict[str, np.ndarray] = {}
    for col in REPAIR_PRIMITIVES:
        p = params[col]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        out[col] = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
    return pd.DataFrame(out, index=df.index)


def _base_factors_v2(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """5 个基础 factor (OPEN/DIVERGENCE/CLOSE_DAMAGE/SUPPLY/RECLAIM).

    - DIVERGENCE = z(d1_intraday_range): 单 primitive 直接标准化 (过程强度)
    - CLOSE_DAMAGE = z(G_C), G_C 复用 composite_reset (与旧 RESET 数学一致)
    - RECLAIM = z(G_Q), G_Q 复用 v001 composite_reclaim (与 v001 逐行相等)
    """
    z = apply_repair_primitive_transform_v2(df, params)
    c_c = params[CLOSE_DAMAGE_COMPOSITE_KEY]
    c_q = params[RECLAIM_COMPOSITE_KEY]
    return pd.DataFrame({
        "OPEN": z["break_open_return"],
        "DIVERGENCE": z[DIVERGENCE_PRIMITIVE],
        "CLOSE_DAMAGE": (composite_reset(z) - c_c["mu"]) / c_c["sigma"],
        "SUPPLY": z["late_day_sell_volume_ratio"],
        "RECLAIM": (composite_reclaim(z) - c_q["mu"]) / c_q["sigma"],
    }, index=df.index)


def _derived_raw_v2(base: pd.DataFrame):
    """3 个 derived 的 raw 值 (由标准化 base factor 构造, 未标准化)."""
    return (
        ("DIVERGENCE_SQ", base["DIVERGENCE"] ** 2),
        ("DIVERGENCE_X_RECLAIM", base["DIVERGENCE"] * base["RECLAIM"]),
        ("DIVERGENCE_X_DAMAGE", base["DIVERGENCE"] * base["CLOSE_DAMAGE"]),
    )


def construct_repair_factors_v2(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """构造 8 个 factor (June/July 均用同一组 reference 参数).

    - 基础 factor: 同 _base_factors_v2
    - derived: (raw - mu) / sigma, raw 由标准化 base factor 构造;
      不经过任何 clip (derived 不做第二次 q01/q99 裁剪)
    - 任何非有限连续 factor => 显式失败
    """
    base = _base_factors_v2(df, params)
    factors = base.copy()
    for name, raw in _derived_raw_v2(base):
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
_SENSITIVITY_PREPROCESS = ("只记录定义, 本版本不构造数值, 不进入 R1/R2; "
                           "历史定义保留在 v001/v004c_factor_spec, 不删除")


def repair_spec_rows_v2() -> list[dict]:
    """16 行静态规格: 8 个 active factor (R1/R2) + 8 个 sensitivity.

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
        "DIVERGENCE", "CORE 基础因子 (单 primitive, 过程强度)", "continuous_standardized",
        "F_V = z(d1_intraday_range)",
        FACTOR_PRIMITIVES["DIVERGENCE"],
        ("D1 盘中实际发生过多大的首次价格分歧 (过程强度)。只描述价格探索范围 "
         "(D1 5min high-low range / prev_close), 不应该因为最后价格重新收回来就"
         "自动认为'分歧没有发生'。v002 语义修订: 不再使用 open_to_close / "
         "high_to_close / close_to_vwap 直接定义 DIVERGENCE (那描述的是收盘损伤, "
         "已拆分给 CLOSE_DAMAGE)。"),
        _PRIM_PREPROCESS, True, True)
    base_row(
        "CLOSE_DAMAGE", "CORE 基础因子 (3 primitive composite, 最终损伤)",
        "continuous_standardized",
        ("G_C = (-z(d1_open_to_close_return_raw) + z(d1_high_to_close_drawdown_raw) "
         "- z(d1_close_to_vwap_raw)) / 3; F_C = z(G_C)"),
        FACTOR_PRIMITIVES["CLOSE_DAMAGE"],
        ("D1 发生分歧以后, 到收盘时仍然留下多少价格结构损伤。与旧 RESET 数学定义"
         "完全一致 (权重 -1/3, +1/3, -1/3 固定, 禁止修改), 只是重新命名与重新解释; "
         "与 DAMAGE7 (最近 7 日路径峰值破坏) 完全不同。"),
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
         "v001 已冻结 (June Pearson 0.5787 / Spearman 0.5414; July 0.4409 / 0.4306, "
         "COMPOSITE_SUPPORTED), 不得改权重, 不得加第三个 primitive。"),
        _COMPOSITE_PREPROCESS, True, True)
    base_row(
        "DIVERGENCE_SQ", "CORE 派生项 (预声明非线性项)", "continuous_derived_standardized",
        "H = F_V^2; F_V2 = z(H)",
        FACTOR_PRIMITIVES["DIVERGENCE_SQ"],
        ("允许未来 Logistic 表达'分歧可能存在最佳区间': 分歧太小 -> 浮筹释放不足; "
         "适度分歧 -> 健康换手; 极端分歧 -> 大量兑现、承接消耗、后续修复预算透支。"
         "若未来 linear DIV > 0 且 DIV_SQ < 0 可形成倒 U 结构, 但本阶段禁止使用 "
         "Target 验证方向。"),
        _DERIVED_PREPROCESS, False, True)
    base_row(
        "DIVERGENCE_X_RECLAIM", "CORE 派生项 (预声明条件项)",
        "continuous_derived_standardized",
        "H = F_V * F_Q; F_VQ = z(H)",
        FACTOR_PRIMITIVES["DIVERGENCE_X_RECLAIM"],
        ("Repair-State 模型的核心条件项: 区分'DIV高 + RECLAIM高 = 大分歧后出现强回收'"
         "与'DIV高 + RECLAIM低 = 大分歧后缺乏承接'。"),
        _DERIVED_PREPROCESS, False, True)
    base_row(
        "DIVERGENCE_X_DAMAGE", "CORE 派生项 (预声明条件项)",
        "continuous_derived_standardized",
        "H = F_V * F_C; F_VC = z(H)",
        FACTOR_PRIMITIVES["DIVERGENCE_X_DAMAGE"],
        ("区分'DIV高 + DAMAGE低 = 过程非常剧烈但最终损伤可控 (剧烈换手)'与"
         "'DIV高 + DAMAGE高 = 过程剧烈且最终价格结构严重受损 (获利盘大量兑现 + "
         "承接资金消耗 + 后续修复预算被透支)'。"),
        _DERIVED_PREPROCESS, False, True)

    # SENSITIVITY / LEGACY_SENSITIVITY (8 个, 禁止进入 R1/R2; 历史定义不删除)
    sensitivity = (
        ("DIVERGENCE_X_SUPPLY", "SENSITIVITY",
         "H = F_D * F_S; F_DS = z(H) (v001 derived 定义保留)",
         ("v001 预声明条件项, v002 降为 SENSITIVITY: 当前优先验证'分歧过度'"
          "'分歧后的回收''分歧是否留下真实损伤'三个核心机制, 避免 interaction "
          "继续膨胀。定义不删除。")),
        ("TURNOVER_COST", "SENSITIVITY_TURNOVER_COST",
         "TURNOVER_COST = z(break_volume_ratio_vs_board_days)",
         ("未来机制预声明: 同样的价格分歧, 如果伴随极端放量, 可能意味着更大的筹码"
          "兑现和修复预算消耗。字段定义 = break_volume / mean(断板前 streak 个板日"
          "volume) (冻结字段, 见 model table schema)。本版本禁止构造 DIV_X_TURNOVER, "
          "禁止进入 R1/R2。")),
    )
    for name, role, formula, econ in sensitivity:
        base_row(name, role, "continuous_standardized", formula, (name,), econ,
                 _SENSITIVITY_PREPROCESS, False, False)

    legacy = (
        ("HIGHZONE", "F_HIGHZONE = z(high_zone_volume_ratio)",
         ("高位筹码堆积。其意义明显依赖价格结果/承接状态, 不再作为独立线性 "
          "core factor。")),
        ("MOM7", "F_MOM7 = z(recent_7d_cumulative_return)",
         ("最近 7 日整体价格扩张。旧 M2 加入 path layer 以后 June OOF 概率质量系统性"
          "恶化, 不再作为核心 Alpha 输入。")),
        ("DAMAGE7", "F_DAMAGE7 = z(recent_7d_max_drawdown)",
         ("最近 7 日路径峰值破坏。同 MOM7, 不再作为核心 Alpha 输入。"
          "注意: 与 CLOSE_DAMAGE 完全不同。")),
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
# 语义解耦状态 (§27; 只用于 DIVERGENCE vs RECLAIM)
# ---------------------------------------------------------------------------

def semantic_decoupling_status(div_reclaim_june: dict[str, float]) -> tuple[str, float]:
    """返回 (status, max_abs_corr).

    - |Pearson| >= 0.70 或 |Spearman| >= 0.70 => SEMANTIC_DECOUPLING_REVIEW
      (人工语义阻塞, 进入正式判定门)
    - 0.50 <= max abs corr < 0.70 => SEMANTIC_DECOUPLING_WATCH (只记录)
    - < 0.50 => SEMANTIC_DECOUPLING_OK (记录)
    """
    max_corr = max(abs(div_reclaim_june["pearson"]),
                   abs(div_reclaim_june["spearman"]))
    if max_corr >= SEMANTIC_DECOUPLING_REVIEW_CORR:
        return "SEMANTIC_DECOUPLING_REVIEW", max_corr
    if max_corr >= SEMANTIC_DECOUPLING_WATCH_CORR:
        return "SEMANTIC_DECOUPLING_WATCH", max_corr
    return "SEMANTIC_DECOUPLING_OK", max_corr


# ---------------------------------------------------------------------------
# 正式判定门 (§37; 只用 X 结构 / 数值稳定性 / June→July X 稳定性, 与 Target 无关)
# ---------------------------------------------------------------------------

def audit_status_v2(degenerate_factors: list[str],
                    vif_r1: dict[str, float],
                    vif_r2: dict[str, float],
                    kappa_blocks: dict[str, dict],
                    div_reclaim_june: dict[str, float]) -> tuple[str, list[str]]:
    """返回 (status, reasons)。

    判定顺序 (§37):
    - DIVERGENCE vs RECLAIM June |Pearson| >= 0.70 或 |Spearman| >= 0.70
      => SEMANTIC_DECOUPLING_REVIEW (人工语义阻塞; 数值原因一并列出)
    - degenerate active factor / VIF >= 10 / condition number >= 100
      => REVIEW_REQUIRED
    - 否则 PASS_REPAIR_STATE_X_V002
    - SEMANTIC_DECOUPLING_WATCH / SHIFT_WATCH / DERIVED_TAIL_WATCH /
      DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW 只记录, 不进入本判定门
    - BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH 由构造源检查前置触发
      (FACTOR_PRIMITIVES["DIVERGENCE"] != (DIVERGENCE_PRIMITIVE,))
    """
    decoupling_status, max_corr = semantic_decoupling_status(div_reclaim_june)

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

    if decoupling_status == SEMANTIC_DECOUPLING_REVIEW:
        return SEMANTIC_DECOUPLING_REVIEW, [
            "DIVERGENCE vs RECLAIM June max|corr| = {:.4f} >= 0.70 => 新 DIVERGENCE "
            "仍是 RECLAIM 的机械反面 (人工语义阻塞; 禁止自动修改 factor 定义, "
            "交人工审查)".format(max_corr)] + reasons
    if reasons:
        return REVIEW_REQUIRED, reasons
    return PASS_REPAIR_STATE_X_V002, []

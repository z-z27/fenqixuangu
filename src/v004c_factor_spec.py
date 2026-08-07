"""v004c 因子规格 v001 — 纯 X 侧因子数学定义与 reference transform。

本模块是 v004c 多变量建模前的因子结构规格, 只包含:

- primitive 列常量与 8 个因子 (OPEN/RESET/SUPPLY/MOM7/DAMAGE7/POS7/TREND/REGIME) 的定义
- June reference transform 纯函数 (fit 只读 June; apply 用同一组参数)
- factor construction 纯函数
- M0/M1/M2 因子成员常量 (公式记录, 不拟合)
- 相关性 / VIF / condition number / KS 诊断纯函数

明确禁止:

- 任何模型训练 / 预测 / 指标 (无 LogisticRegression / fit / predict / AUC)
- 任何 Target 读取 (本模块只处理 X)
- 全样本预处理 (clip/标准化参数只允许来自 reference/fold 样本)

说明: 本模块记录的是 "factor definition freeze 候选" 的数学定义,
不是 "model coefficient freeze" (系数必须等下一阶段 walk-forward 训练估计)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# primitive 常量
# ---------------------------------------------------------------------------

PRIMITIVE_COLUMNS: tuple[str, ...] = (
    "break_open_return",
    "d1_open_to_close_return_raw",
    "d1_high_to_close_drawdown_raw",
    "d1_close_to_vwap_raw",
    "high_zone_volume_ratio",
    "late_day_sell_volume_ratio",
    "recent_7d_cumulative_return",
    "recent_7d_max_drawdown",
    "recent_7d_close_position",
    "d1_ma10_slope",
    "board_streak_is_3",
)

CONTINUOUS_PRIMITIVES: tuple[str, ...] = tuple(
    p for p in PRIMITIVE_COLUMNS if p != "board_streak_is_3")
BINARY_PRIMITIVES: tuple[str, ...] = ("board_streak_is_3",)

# X 读取入口只允许这 3 个标识列 + 11 个 primitive (target-blind 约定)
X_ID_COLUMNS: tuple[str, ...] = ("event_id", "code", "signal_date")

FORBIDDEN_TOKENS: tuple[str, ...] = (
    "target", "d2_", "d3_", "outcome", "future",
    "v002", "v004a", "v004b", "v005", "recognition", "policy",
)

# ---------------------------------------------------------------------------
# factor 常量
# ---------------------------------------------------------------------------

FACTOR_NAMES: tuple[str, ...] = (
    "OPEN", "RESET", "SUPPLY", "MOM7", "DAMAGE7", "POS7", "TREND", "REGIME",
)

# 连续 factor (REGIME 是 0/1 regime 控制变量, 不标准化)
CONTINUOUS_FACTORS: tuple[str, ...] = tuple(
    f for f in FACTOR_NAMES if f != "REGIME")

# factor -> 组成 primitive (顺序即权重/符号顺序)
FACTOR_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "OPEN": ("break_open_return",),
    "RESET": ("d1_open_to_close_return_raw",
              "d1_high_to_close_drawdown_raw",
              "d1_close_to_vwap_raw"),
    "SUPPLY": ("high_zone_volume_ratio", "late_day_sell_volume_ratio"),
    "MOM7": ("recent_7d_cumulative_return",),
    "DAMAGE7": ("recent_7d_max_drawdown",),
    "POS7": ("recent_7d_close_position",),
    "TREND": ("d1_ma10_slope",),
    "REGIME": ("board_streak_is_3",),
}

# RESET: G = (-z_OC + z_HC - z_VWAP) / 3
RESET_SIGNS: tuple[float, ...] = (-1.0, +1.0, -1.0)
RESET_WEIGHTS: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

# SUPPLY: G = (z_high_zone + z_late_sell) / 2
SUPPLY_SIGNS: tuple[float, ...] = (+1.0, +1.0)
SUPPLY_WEIGHTS: tuple[float, ...] = (0.5, 0.5)

M1_FACTORS: tuple[str, ...] = ("OPEN", "RESET", "SUPPLY")
M2_FACTORS: tuple[str, ...] = FACTOR_NAMES
PATH_BLOCK_FACTORS: tuple[str, ...] = ("MOM7", "DAMAGE7", "POS7")

# 诊断解释阈值 (只解释, 禁止程序自动删除/换因子)
CORR_LOW, CORR_MODERATE, CORR_HIGH, CORR_SEVERE = 0.50, 0.70, 0.85, 0.85
VIF_OK, VIF_WATCH, VIF_SEVERE = 5.0, 10.0, 10.0
KAPPA_OK, KAPPA_WATCH, KAPPA_SEVERE = 30.0, 100.0, 100.0
SMD_WATCH = 0.50
KS_WATCH = 0.25
DEGENERATE_STD = 1e-8
NEAR_SINGULAR_SMIN = 1e-10

RESET_COMPOSITE_KEY = "composite_RESET"
SUPPLY_COMPOSITE_KEY = "composite_SUPPLY"


class FactorSpecError(Exception):
    """因子规格/审计失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# reference transform 纯函数
# ---------------------------------------------------------------------------

def fit_reference_transform(june_df: pd.DataFrame) -> dict:
    """仅用 June (reference X sample) 拟合 transform 参数。

    连续 primitive: q01/q99 (clip) + mu/sigma (standardize), 全部来自 June。
    RESET/SUPPLY composite G 的 mu/sigma 同样只来自 June 上构造的 G。
    任何退化 (sigma <= 1e-8) 直接失败, 不静默。
    """
    if len(june_df) == 0:
        raise FactorSpecError("reference transform 需要非空 June 样本")
    params: dict = {}
    for col in CONTINUOUS_PRIMITIVES:
        x = pd.to_numeric(june_df[col], errors="coerce").to_numpy(dtype=float)
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            raise FactorSpecError(f"June 中 primitive {col} 无有限值")
        q01, q99 = np.quantile(finite, [0.01, 0.99])
        mu = float(finite.mean())
        sigma = float(finite.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= DEGENERATE_STD:
            raise FactorSpecError(
                f"June 中 primitive {col} 退化 (sigma={sigma}, 禁止继续)")
        params[col] = {"q01": float(q01), "q99": float(q99), "mu": mu, "sigma": sigma}
    z = apply_primitive_transform(june_df, params)
    for key, g in ((RESET_COMPOSITE_KEY, composite_reset(z)),
                   (SUPPLY_COMPOSITE_KEY, composite_supply(z))):
        finite = g[np.isfinite(g)]
        if len(finite) == 0:
            raise FactorSpecError(f"June 中 composite {key} 无有限值")
        mu = float(finite.mean())
        sigma = float(finite.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= DEGENERATE_STD:
            raise FactorSpecError(
                f"June 中 composite {key} 退化 (sigma={sigma}, 禁止继续)")
        params[key] = {"mu": mu, "sigma": sigma}
    return params


def apply_primitive_transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """用给定参数 (只允许 June reference 参数) 构造 reference standardized primitive。

    对每个连续 primitive: clip [q01, q99] 后 (x - mu) / sigma。
    June/July 共用同一组 June 参数; July 不得用自身参数重算。
    """
    out: dict[str, np.ndarray] = {}
    for col in CONTINUOUS_PRIMITIVES:
        p = params[col]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        out[col] = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
    return pd.DataFrame(out, index=df.index)


def composite_reset(z: pd.DataFrame) -> np.ndarray:
    """G_RESET = (-z_OC + z_HC - z_VWAP) / 3 (权重固定, 禁止优化)."""
    return (-z["d1_open_to_close_return_raw"]
            + z["d1_high_to_close_drawdown_raw"]
            - z["d1_close_to_vwap_raw"]) / 3.0


def composite_supply(z: pd.DataFrame) -> np.ndarray:
    """G_SUPPLY = (z_high_zone + z_late_sell) / 2 (权重固定, 禁止优化)."""
    return (z["high_zone_volume_ratio"] + z["late_day_sell_volume_ratio"]) / 2.0


def construct_factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """构造 8 个 factor (June/July 均用 June reference 参数)。

    - 连续 factor = z(primitive) 或 composite 再标准化 (June composite 参数)
    - REGIME = board_streak_is_3 原值 0/1, 不标准化 (最终模型保持 0/1)
    - 任何非有限连续 factor / 非法 REGIME 值 => 显式失败
    """
    z = apply_primitive_transform(df, params)
    factors = pd.DataFrame(index=df.index)
    factors["OPEN"] = z["break_open_return"]
    for key, g in ((RESET_COMPOSITE_KEY, composite_reset(z)),
                   (SUPPLY_COMPOSITE_KEY, composite_supply(z))):
        c = params[key]
        factors[key.replace("composite_", "")] = (g - c["mu"]) / c["sigma"]
    factors["MOM7"] = z["recent_7d_cumulative_return"]
    factors["DAMAGE7"] = z["recent_7d_max_drawdown"]
    factors["POS7"] = z["recent_7d_close_position"]
    factors["TREND"] = z["d1_ma10_slope"]
    regime = pd.to_numeric(df["board_streak_is_3"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isin(regime, [0.0, 1.0])):
        raise FactorSpecError("REGIME (board_streak_is_3) 必须严格取值 0/1, 发现其他值")
    factors["REGIME"] = regime
    for col in CONTINUOUS_FACTORS:
        if not np.all(np.isfinite(factors[col])):
            raise FactorSpecError(f"factor {col} 包含非有限值")
    return factors


# ---------------------------------------------------------------------------
# 诊断纯函数 (相关性 / VIF / condition number / KS / SMD)
# ---------------------------------------------------------------------------

def pearson(a, b) -> float:
    """Pearson 相关系数 (逐对删缺失; 有效样本 <2 返回 NaN)."""
    a = np.asarray(pd.to_numeric(a, errors="coerce"), dtype=float)
    b = np.asarray(pd.to_numeric(b, errors="coerce"), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def spearman(a, b) -> float:
    """Spearman 相关系数 (平均秩处理平局, 逐对删缺失)."""
    s = pd.DataFrame({"a": pd.Series(pd.to_numeric(a, errors="coerce")),
                      "b": pd.Series(pd.to_numeric(b, errors="coerce"))}).dropna()
    if len(s) < 2:
        return float("nan")
    return float(s["a"].corr(s["b"], method="spearman"))


def correlation_severity(rho: float) -> str:
    """|rho| 解释: <0.50 LOW / 0.50-0.70 MODERATE / 0.70-0.85 HIGH / >=0.85 SEVERE."""
    a = abs(float(rho))
    if a < CORR_LOW:
        return "LOW"
    if a < CORR_MODERATE:
        return "MODERATE"
    if a < CORR_HIGH:
        return "HIGH"
    return "SEVERE"


def vif_matrix(df: pd.DataFrame) -> dict[str, float]:
    """VIF: 每个 factor 用其余 factor 线性解释 (含 intercept), VIF = 1 / (1 - R^2)。

    退化/奇异必须显式报告: R^2 ~ 1 时 VIF 返回 inf, 不得静默输出正常值。
    样本不足 (n < k + 2) 直接失败。
    """
    X = df.to_numpy(dtype=float)
    n, k = X.shape
    if n < k + 2:
        raise FactorSpecError(f"VIF 样本不足: n={n}, k={k}")
    if not np.all(np.isfinite(X)):
        raise FactorSpecError("VIF 矩阵包含非有限值")
    out: dict[str, float] = {}
    for j, name in enumerate(df.columns):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        a = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        resid = y - a @ coef
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        denom = 1.0 - r2
        out[name] = float("inf") if denom <= 1e-10 else 1.0 / denom
    return out


def condition_number_diagnostic(matrix: np.ndarray):
    """centered + unit-variance 诊断矩阵的 SVD condition number。

    REGIME 只在这个诊断副本中 center/scale; 正式模型仍保持 0/1。
    返回 (s_max, s_min, kappa, near_singular); s_min ~ 0 => near singular (SEVERE)。
    """
    x = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(x)):
        raise FactorSpecError("condition number 矩阵包含非有限值")
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    if np.any(stds <= 0):
        return float("nan"), float("nan"), float("inf"), True
    xc = (x - means) / stds
    s = np.linalg.svd(xc, compute_uv=False)
    s_max = float(s[0])
    s_min = float(s[-1])
    kappa = float("inf") if s_min <= 0.0 else s_max / s_min
    near = (s_min <= NEAR_SINGULAR_SMIN) or kappa >= 1e6 or not np.isfinite(kappa)
    return s_max, s_min, kappa, near


def standardized_mean_difference(mu_a, sigma_a, mu_b, sigma_b) -> float:
    """SMD = (mu_b - mu_a) / sqrt((sigma_b^2 + sigma_a^2) / 2)."""
    denom = np.sqrt((float(sigma_a) ** 2 + float(sigma_b) ** 2) / 2.0)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float((float(mu_b) - float(mu_a)) / denom)


def ks_2samp_statistic(a, b) -> float:
    """two-sample KS D 统计量 = 经验 CDF 最大距离 (自带实现, 不依赖 scipy)."""
    a = np.asarray(pd.to_numeric(a, errors="coerce"), dtype=float)
    b = np.asarray(pd.to_numeric(b, errors="coerce"), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    sa, sb = np.sort(a), np.sort(b)
    vals = np.unique(np.concatenate([sa, sb]))
    ca = np.searchsorted(sa, vals, side="right") / len(a)
    cb = np.searchsorted(sb, vals, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


# ---------------------------------------------------------------------------
# factor spec 静态表 (v004c factor specification v001)
# ---------------------------------------------------------------------------

FACTOR_SPEC_FIELDS: tuple[str, ...] = (
    "factor_name", "factor_role", "primitive_columns", "primitive_signs",
    "primitive_weights", "economic_definition", "construction",
    "final_model_preprocessing", "m1_member", "m2_member",
    "target_used_to_construct",
)

_FOLD_PREPROCESS = ("walk-forward 中每个 training fold 内拟合: clip [q01, q99] "
                    "+ (x - mu)/sigma (fold 参数; 禁止全样本参数)")


def factor_spec_rows() -> list[dict]:
    """8 个 factor 的静态规格行 (v004c factor specification v001)。

    所有 target_used_to_construct 必须为 False:
    正式计算公式本身不读取 Target (不宣称历史研究从未看过 June Target)。
    """
    rows: list[dict] = []
    for f in FACTOR_NAMES:
        prims = FACTOR_PRIMITIVES[f]
        if f == "OPEN":
            role = "连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("D0 连板结束后, D1 开盘时隔夜资金仍愿意给予多强的价格承接 "
                    "(overnight demand / opening strength)。高值 = D1 开盘相对 D0 收盘更强。")
            construction = "F_OPEN = z(break_open_return); break_open_return = Open_D1/Close_D0 - 1"
            pre = _FOLD_PREPROCESS
        elif f == "RESET":
            role = "连续标准化因子 (3 primitive composite)"
            signs = "|".join(str(int(s)) for s in RESET_SIGNS)
            weights = "|".join("1/3" for _ in RESET_SIGNS)
            econ = ("D1 从盘中强势定价锚点向收盘价格发生多大程度的价格重置 (price reset "
                    "intensity)。高 = 收盘相对开盘更低 + 最高价回撤更大 + 收盘相对 VWAP 更低。"
                    "只描述重置强度, 不是'股票越弱越好'; 未来系数方向由模型估计, 禁止约束 beta 符号。")
            construction = ("G_RESET = (-z_OC + z_HC - z_VWAP)/3; "
                            "F_RESET = z(G_RESET) (June composite 参数)")
            pre = _FOLD_PREPROCESS
        elif f == "SUPPLY":
            role = "连续标准化因子 (2 primitive composite)"
            signs = "|".join(str(int(s)) for s in SUPPLY_SIGNS)
            weights = "1/2|1/2"
            econ = ("D1 高位筹码堆积与尾盘主动卖压程度 (supply pressure)。高 = 更多成交集中在"
                    "高位 + 更多尾盘下跌 bar 成交。不假定未来系数必须为负。")
            construction = ("G_SUPPLY = (z(high_zone_volume_ratio) + "
                            "z(late_day_sell_volume_ratio))/2; "
                            "F_SUPPLY = z(G_SUPPLY) (June composite 参数)")
            pre = _FOLD_PREPROCESS
        elif f == "MOM7":
            role = "连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = "最近 7 个交易日整体价格扩张程度。"
            construction = ("F_MOM7 = z(recent_7d_cumulative_return); "
                            "cumulative = Close_D1/Close_T-6 - 1 (7 行相邻有效日线窗口)")
            pre = _FOLD_PREPROCESS
        elif f == "DAMAGE7":
            role = "连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("最近 7 日路径中峰值破坏的严重程度 (path damage)。公式由 model-table 阶段"
                    "冻结: prior_peak_t = max(high_s), s<t (严格较早交易日); "
                    "DD_t = max(0, (prior_peak_t - low_t)/prior_peak_t); MaxDD_7 = max_t DD_t。"
                    "禁止修改该公式。")
            construction = "F_DAMAGE7 = z(recent_7d_max_drawdown)"
            pre = _FOLD_PREPROCESS
        elif f == "POS7":
            role = "连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("D1 收盘位于最近 7 日完整价格区间中的相对位置。高 = 仍靠近 7 日高位; "
                    "低 = 已明显回落到 7 日区间下方。")
            construction = ("F_POS7 = z(recent_7d_close_position); "
                            "position = (Close_D1 - Low_7)/(High_7 - Low_7)")
            pre = _FOLD_PREPROCESS
        elif f == "TREND":
            role = "连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("中期价格中心本身的方向 (MA10 上移还是下移)。不同于'当前价格距离 MA10 多远'。")
            construction = ("F_TREND = z(d1_ma10_slope); "
                            "slope = MA10_D1/MA10_{D1-1} - 1")
            pre = _FOLD_PREPROCESS
        elif f == "REGIME":
            role = "二元 regime 控制变量 (0/1, 不标准化)"
            signs, weights = "+1", "1"
            econ = "连板 regime 控制: 0 = 2 板, 1 = 3 板 (board_streak_before_break)。"
            construction = "F_REGIME = board_streak_is_3 (0/1 原值)"
            pre = ("0/1 原值, 不做 z-score; 只有 condition number 诊断副本中临时 "
                   "center/scale (diagnostic scaling != future model preprocessing)")
        else:  # pragma: no cover - 常量表完整
            raise FactorSpecError(f"未知 factor: {f}")
        rows.append({
            "factor_name": f,
            "factor_role": role,
            "primitive_columns": "|".join(prims),
            "primitive_signs": signs,
            "primitive_weights": weights,
            "economic_definition": econ,
            "construction": construction,
            "final_model_preprocessing": pre,
            "m1_member": f in M1_FACTORS,
            "m2_member": f in M2_FACTORS,
            "target_used_to_construct": False,
        })
    return rows

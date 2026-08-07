"""v004c 因子规格 v002 — 纯 X 侧因子数学定义与 reference transform。

本模块是 v004c 多变量建模前的因子结构规格, 只包含:

- primitive 列常量与 9 个 factor (7 CORE: OPEN/RESET/HIGHZONE/LATESELL/MOM7/DAMAGE7/REGIME;
  2 SENSITIVITY: POS7/TREND) 的定义
- June reference transform 纯函数 (fit 只读 June; apply 用同一组参数)
- factor construction 纯函数
- M0/M1/M2 因子成员常量 (公式记录, 不拟合)
- 相关性 / VIF / condition number / KS 诊断纯函数

v002 修订 (依据 v001 纯 X 结构审查结论, 与 Target 无关):

- SUPPLY composite 拆分: high_zone_volume_ratio 与 late_day_sell_volume_ratio
  在 June 中接近独立, 不再强制压缩成 50/50 composite; HIGHZONE / LATESELL
  作为独立 factor, 未来必须由 Logistic 独立估计系数, 禁止重新合成 SUPPLY
  (v001 历史 SUPPLY 定义只保留在 v001 报告资产)
- POS7 移入 SENSITIVITY (SENSITIVITY_STRUCTURAL_REDUNDANCY: 与 RESET 结构重复)
- TREND 移入 SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY: 与 MOM7 高度相关
  + June→July X shift)
- v001 历史资产 (src/tools/reports) 保持不变

明确禁止:

- 任何模型训练 / 预测 / 指标 (无 LogisticRegression / fit / predict / AUC)
- 任何 Target 读取 (本模块只处理 X)
- 全样本预处理 (clip/标准化参数只允许来自 reference/fold 样本)
- 根据诊断自动修改 factor 定义 / composite 权重 (SEVERE/VIF/kappa 只记录给人工)

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
# factor 常量 (v002: 7 CORE + 2 SENSITIVITY, 无 active SUPPLY composite)
# ---------------------------------------------------------------------------

CORE_FACTORS: tuple[str, ...] = (
    "OPEN", "RESET", "HIGHZONE", "LATESELL", "MOM7", "DAMAGE7", "REGIME",
)
SENSITIVITY_FACTORS: tuple[str, ...] = ("POS7", "TREND")
FACTOR_NAMES: tuple[str, ...] = CORE_FACTORS + SENSITIVITY_FACTORS

# 连续 factor (REGIME 是 0/1 regime 控制变量, 不标准化)
CONTINUOUS_FACTORS: tuple[str, ...] = tuple(
    f for f in FACTOR_NAMES if f != "REGIME")

# factor -> 组成 primitive (顺序即权重/符号顺序)
FACTOR_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "OPEN": ("break_open_return",),
    "RESET": ("d1_open_to_close_return_raw",
              "d1_high_to_close_drawdown_raw",
              "d1_close_to_vwap_raw"),
    "HIGHZONE": ("high_zone_volume_ratio",),
    "LATESELL": ("late_day_sell_volume_ratio",),
    "MOM7": ("recent_7d_cumulative_return",),
    "DAMAGE7": ("recent_7d_max_drawdown",),
    "POS7": ("recent_7d_close_position",),
    "TREND": ("d1_ma10_slope",),
    "REGIME": ("board_streak_is_3",),
}

# RESET: G = (-z_OC + z_HC - z_VWAP) / 3 (权重固定, 禁止修改)
RESET_SIGNS: tuple[float, ...] = (-1.0, +1.0, -1.0)
RESET_WEIGHTS: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
# v002 起不存在 active SUPPLY composite; HIGHZONE / LATESELL 独立建模

# M0 = alpha; M1 = D1 STRUCTURE (4 factors); M2 = INTEGRATED PATH (7 factors)
M1_FACTORS: tuple[str, ...] = ("OPEN", "RESET", "HIGHZONE", "LATESELL")
M2_FACTORS: tuple[str, ...] = CORE_FACTORS
PATH_BLOCK_FACTORS: tuple[str, ...] = ("MOM7", "DAMAGE7")

# sensitivity 状态 (v002 决策, 只记录, 不实施修改)
SENSITIVITY_STRUCTURAL_REDUNDANCY = "SENSITIVITY_STRUCTURAL_REDUNDANCY"
SENSITIVITY_HORIZON_STATIONARITY = "SENSITIVITY_HORIZON_STATIONARITY"
SENSITIVITY_STATUS: dict[str, str] = {
    "POS7": SENSITIVITY_STRUCTURAL_REDUNDANCY,
    "TREND": SENSITIVITY_HORIZON_STATIONARITY,
}

# 诊断解释阈值 (只解释, 禁止程序自动删除/换因子)
CORR_LOW, CORR_MODERATE, CORR_HIGH, CORR_SEVERE = 0.50, 0.70, 0.85, 0.85
VIF_OK, VIF_WATCH, VIF_SEVERE = 5.0, 10.0, 10.0
KAPPA_OK, KAPPA_WATCH, KAPPA_SEVERE = 30.0, 100.0, 100.0
SMD_WATCH = 0.50
KS_WATCH = 0.25
DEGENERATE_STD = 1e-8
NEAR_SINGULAR_SMIN = 1e-10

RESET_COMPOSITE_KEY = "composite_RESET"


class FactorSpecError(Exception):
    """因子规格/审计失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# reference transform 纯函数
# ---------------------------------------------------------------------------

def fit_reference_transform(june_df: pd.DataFrame) -> dict:
    """仅用 June (reference X sample) 拟合 transform 参数。

    连续 primitive: q01/q99 (clip) + mu/sigma (standardize), 全部来自 June。
    RESET composite G 的 mu/sigma 同样只来自 June 上构造的 G。
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
    for key, g in ((RESET_COMPOSITE_KEY, composite_reset(z)),):
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


def construct_factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """构造 9 个 factor (June/July 均用 June reference 参数)。

    - 连续 factor = z(primitive) 或 composite 再标准化 (June composite 参数)
    - HIGHZONE / LATESELL 各自独立 = z(primitive), 不再合成 SUPPLY composite
    - REGIME = board_streak_is_3 原值 0/1, 不标准化 (最终模型保持 0/1)
    - 任何非有限连续 factor / 非法 REGIME 值 => 显式失败
    """
    z = apply_primitive_transform(df, params)
    factors = pd.DataFrame(index=df.index)
    factors["OPEN"] = z["break_open_return"]
    c = params[RESET_COMPOSITE_KEY]
    factors["RESET"] = (composite_reset(z) - c["mu"]) / c["sigma"]
    factors["HIGHZONE"] = z["high_zone_volume_ratio"]
    factors["LATESELL"] = z["late_day_sell_volume_ratio"]
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
# factor spec 静态表 (v004c factor specification v002)
# ---------------------------------------------------------------------------

FACTOR_SPEC_FIELDS: tuple[str, ...] = (
    "factor_name", "factor_role", "primitive_columns", "primitive_signs",
    "primitive_weights", "economic_definition", "construction",
    "final_model_preprocessing", "m1_member", "m2_member",
    "target_used_to_construct", "sensitivity_status",
)

_FOLD_PREPROCESS = ("walk-forward 中每个 training fold 内拟合: clip [q01, q99] "
                    "+ (x - mu)/sigma (fold 参数; 禁止全样本参数)")


def factor_spec_rows() -> list[dict]:
    """9 个 factor 的静态规格行 (v004c factor specification v002)。

    7 CORE (OPEN/RESET/HIGHZONE/LATESELL/MOM7/DAMAGE7/REGIME) + 2 SENSITIVITY
    (POS7/TREND, 各自带 v002 sensitivity 状态)。不含 active SUPPLY composite
    (历史 SUPPLY 定义只存在于 v001 报告资产)。

    所有 target_used_to_construct 必须为 False:
    正式计算公式本身不读取 Target (不宣称历史研究从未看过 June Target)。
    """
    rows: list[dict] = []
    for f in FACTOR_NAMES:
        prims = FACTOR_PRIMITIVES[f]
        if f == "OPEN":
            role = "CORE 连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("D0 连板结束后, D1 开盘时隔夜资金仍愿意给予多强的价格承接 "
                    "(overnight demand / opening strength)。高值 = D1 开盘相对 D0 收盘更强。")
            construction = "F_OPEN = z(break_open_return); break_open_return = Open_D1/Close_D0 - 1"
            pre = _FOLD_PREPROCESS
        elif f == "RESET":
            role = "CORE 连续标准化因子 (3 primitive composite)"
            signs = "|".join(str(int(s)) for s in RESET_SIGNS)
            weights = "|".join("1/3" for _ in RESET_SIGNS)
            econ = ("D1 从盘中强势定价锚点向收盘价格发生多大程度的价格重置 (price reset "
                    "intensity)。高 = 收盘相对开盘更低 + 最高价回撤更大 + 收盘相对 VWAP 更低。"
                    "只描述重置强度, 不是'股票越弱越好'; 未来系数方向由模型估计, 禁止约束 beta 符号。"
                    "权重 -1/3, +1/3, -1/3 固定, 禁止修改。")
            construction = ("G_RESET = (-z_OC + z_HC - z_VWAP)/3; "
                            "F_RESET = z(G_RESET) (June composite 参数)")
            pre = _FOLD_PREPROCESS
        elif f == "HIGHZONE":
            role = "CORE 连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("D1 成交筹码在高位价格区域的堆积程度 (高位区成交占比)。v002 起作为独立"
                    " factor 建模, 不再与 late_day_sell_volume_ratio 压缩成 SUPPLY composite。")
            construction = "F_HIGHZONE = z(high_zone_volume_ratio)"
            pre = _FOLD_PREPROCESS
        elif f == "LATESELL":
            role = "CORE 连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("D1 尾盘下跌 bar 对应的主动供应/卖压程度。v002 起作为独立 factor 建模;"
                    " HIGHZONE 与 LATESELL 必须由 Logistic 独立估计系数, 禁止重新合成 SUPPLY。")
            construction = "F_LATESELL = z(late_day_sell_volume_ratio)"
            pre = _FOLD_PREPROCESS
        elif f == "MOM7":
            role = "CORE 连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = "最近 7 个交易日整体价格扩张程度。"
            construction = ("F_MOM7 = z(recent_7d_cumulative_return); "
                            "cumulative = Close_D1/Close_T-6 - 1 (7 行相邻有效日线窗口)")
            pre = _FOLD_PREPROCESS
        elif f == "DAMAGE7":
            role = "CORE 连续标准化因子 (单 primitive)"
            signs, weights = "+1", "1"
            econ = ("最近 7 日路径中峰值破坏的严重程度 (path damage)。公式由 model-table 阶段"
                    "冻结: prior_peak_t = max(high_s), s<t (严格较早交易日); "
                    "DD_t = max(0, (prior_peak_t - low_t)/prior_peak_t); MaxDD_7 = max_t DD_t。"
                    "禁止修改该公式。即使存在 SHIFT_WATCH 也继续属于 core。")
            construction = "F_DAMAGE7 = z(recent_7d_max_drawdown)"
            pre = _FOLD_PREPROCESS
        elif f == "POS7":
            role = "SENSITIVITY 连续标准化因子 (单 primitive, 不进 M1/M2 core)"
            signs, weights = "+1", "1"
            econ = ("D1 收盘位于最近 7 日完整价格区间中的相对位置。高 = 仍靠近 7 日高位; "
                    "低 = 已明显回落到 7 日区间下方。v002 状态: "
                    "SENSITIVITY_STRUCTURAL_REDUNDANCY (与 RESET 严重重复, 纯 X 结构发现)。")
            construction = ("F_POS7 = z(recent_7d_close_position); "
                            "position = (Close_D1 - Low_7)/(High_7 - Low_7)")
            pre = _FOLD_PREPROCESS
        elif f == "TREND":
            role = "SENSITIVITY 连续标准化因子 (单 primitive, 不进 M1/M2 core)"
            signs, weights = "+1", "1"
            econ = ("中期价格中心本身的方向 (MA10 上移还是下移)。不同于'当前价格距离 MA10 多远'。"
                    "v002 状态: SENSITIVITY_HORIZON_STATIONARITY (与 MOM7 高度相关"
                    " + June→July X shift)。")
            construction = ("F_TREND = z(d1_ma10_slope); "
                            "slope = MA10_D1/MA10_{D1-1} - 1")
            pre = _FOLD_PREPROCESS
        elif f == "REGIME":
            role = "CORE 二元 regime 控制变量 (0/1, 不标准化)"
            signs, weights = "+1", "1"
            econ = "连板 regime 控制: 0 = 2 板, 1 = 3 板 (board_streak_before_break)。"
            construction = "F_REGIME = board_streak_is_3 (0/1 原值)"
            pre = ("0/1 原值, 不做 z-score; 只有 condition number 诊断副本中临时 "
                   "center/scale (diagnostic scaling != future model preprocessing); "
                   "第一版禁止 interaction")
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
            "sensitivity_status": SENSITIVITY_STATUS.get(f, ""),
        })
    return rows

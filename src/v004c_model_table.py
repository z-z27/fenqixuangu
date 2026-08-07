"""v004c 统一建模表 (v004c_model_table_v001) 构建模块。

本模块是 v004c 正式多变量模型开发前的最后一个数据工程阶段:
读取冻结 Stage 1 资产 + Stage 2.1 结构定义, 从现有 data/cache/daily
计算 3 个已批准的 recent-7d 字段, 生成紧凑统一建模表。

严格约束 (本模块):
- 不训练任何模型, 不拟合 M0/M1/M2, 不生成概率/rank/Top3;
- 不使用 Target 选择/删除特征 (feature contract 只来自 Stage 2.1 规则
  与 v004c_feature_coverage_v001.csv 覆盖结论);
- 不做任何全样本预处理 (无 z-score/winsorize/clip/标准化);
- 不联网, 不新增行情源, 不修改 180/120 数据窗口;
- 所有模型输入 available_as_of <= D1_CLOSE。

列角色 (数据角色定义, 不是模型冻结):
- IDENTIFIER: 事件标识与切分主键 (下一阶段 walk-forward 统一按 signal_date 切分)
- FEATURE: 正式模型候选 (PRIMARY 29 + SENSITIVITY 26)
- LABEL: 仅 target7_daily_d2open_d3high (只属于 LABEL, 绝不允许进入 FEATURE)
- AUDIT_ONLY: 数据质量/来源信息, 不可被模型消费

recent-7d 三字段的严格日级公式 (已修正, 与 v004c_feature_coverage_v001.csv
一致):
- recent_7d_cumulative_return = close(D1) / close(T-6) - 1, 窗口 = D1 前 7 行
- recent_7d_max_drawdown = max_{t∈[T-5..D1]} max(0, (prior_peak_t - low_t)/prior_peak_t),
  prior_peak_t = max(high_s), s < t (只使用严格较早交易日 high, 不使用同日
  high/low 先后顺序; 当前日 high 只成为后续 prior peak)
- recent_7d_close_position = (close(D1) - 7d_low) / (7d_high - 7d_low);
  high_7d == low_7d 时置空 (NaN), 禁止 inf/0/0.5 填充

输出 (原子, 3 个文件):
- v004c_model_table_v001.csv
- v004c_model_table_schema_v001.csv
- v004c_model_table_review.md
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .v004c_d1_dataset import DatasetValidationError, normalize_binary_series

# ---------------------------------------------------------------------------
# 数据角色定义 (feature contract; 不是最终模型冻结)
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS = (
    "event_id",
    "code",
    "signal_date",
    "board_streak_before_break",
)

LABEL_COLUMNS = ("target7_daily_d2open_d3high",)

AUDIT_COLUMNS = (
    "break_date",  # legacy identifier, 与 signal_date 逐行一致 (已验证)
    "daily_label_quality_ok",
    "target_training_eligible",
    "sample_role_v022",
    "d1_factor_quality_ok",
)

# Stage 2.1 准入 (coverage 结论 role = M1_BASELINE_CANDIDATE / M2_PRIMARY_CANDIDATE)
PRIMARY_FEATURE_COLUMNS = (
    # D0_TO_D1_TRANSITION / D1_PRICE_ACTION / BOARD_STATE / CHIP / LATE / VWAP / MA
    "break_open_return",
    "break_high_return",
    "break_close_return",
    "break_touched_limit_up",
    "volume_above_d1_close_ratio",
    "late_day_sell_volume_ratio",
    "d1_close_to_ma5_raw",
    "d1_high_to_close_drawdown_raw",
    "d1_low_to_close_recovery",
    "d1_open_to_close_return_raw",
    "break_volume_ratio_vs_board_days",
    "d1_close_to_vwap_raw",
    "break_day_in_pool",
    "high_zone_volume_ratio",
    "d1_close_to_ma10_raw",
    "d1_close_to_ma20",
    "d1_ma10_slope",
    "d1_low_to_ma5_raw",
    "d1_ma5_slope",
    "d1_true_reclaim_ma5",
    "d1_afternoon_return",
    "d1_intraday_range",
    "d1_last_hour_return",
    "d1_up_bar_volume_ratio",
    "down_bar_volume_ratio",
    "board_streak_is_3",  # PREDECLARED_DERIVED (由 board_streak_before_break 派生)
    # MODEL_TABLE_REQUIRED_COVERAGE_FIELDS (RECENT_7D_PATH)
    "recent_7d_cumulative_return",
    "recent_7d_max_drawdown",
    "recent_7d_close_position",
)

# Stage 2.1 准入 (coverage 结论 role = SENSITIVITY_ONLY)
SENSITIVITY_FEATURE_COLUMNS = (
    "board_day_amount_rank",
    "board_day_turnover_rank",
    "board_day_volume_rank",
    "break_upper_shadow_ratio",
    "break_lower_shadow_ratio",
    "break_amount_ratio_vs_board_days",
    "break_opened_from_limit_up",
    "consecutive_days_below_ma5",
    "consecutive_days_below_ma10",
    "d1_close_location",
    "amount_above_d1_close_ratio",
    "high_zone_amount_ratio",
    "late_day_sell_amount_ratio",
    "last_board_day_in_pool",
    "pool_consecutive_count_last_board",
    "d1_close_above_ma5",
    "d1_close_above_ma10",
    "d1_true_reclaim_ma10",
    "d1_close_to_ma5_bucket",
    "d1_open_to_close_bucket",
    "d1_high_to_ma5_raw",
    "d1_low_to_ma10_raw",
    "ma5_overheat_10",  # PREDECLARED_DERIVED
    "overrepair",  # PREDECLARED_DERIVED
    "break_volume_abnormality",  # PREDECLARED_DERIVED
    "profit_pressure",  # PREDECLARED_DERIVED
)

FEATURE_COLUMNS = PRIMARY_FEATURE_COLUMNS + SENSITIVITY_FEATURE_COLUMNS

# 明确排除 (不得进入任何 feature universe; 由 coverage 结论决定)
CONTEXT_ONLY_COLUMNS = (
    "recent_limit_up_count_10d",
    "recent_limit_up_count_20d",
    "recent_pool_appearance_count_10d",
    "recent_pool_appearance_count_20d",
    "max_board_streak_20d",
)
DEFER_EXCLUDED_COLUMNS = (
    "d1_ma20_slope",
    "recent_7d_limit_up_count",
    "board_stage_volume_price_decomposition",
)
REDUNDANT_EXCLUDED_COLUMNS = (
    "profit_chip_ratio",
    "d1_vwap_to_close_gap",
)

# 泄漏扫描 token (任何 feature 名称含下列 token 之一即硬失败)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "tail_loss",
    "d2_",
    "d3_",
    "outcome",
    "post_",
    "recognition",
    "v002",
    "v004a",
    "v004b",
    "v005",
    "policy",
    "realized",
    "future",
)

RECENT_7D_FIELDS = (
    "recent_7d_cumulative_return",
    "recent_7d_max_drawdown",
    "recent_7d_close_position",
)
RECENT_7D_WINDOW = 7  # T-6 .. D1, 共 7 个个股有效交易日

PREDECLARED_DERIVED_FIELDS = (
    "board_streak_is_3",
    "ma5_overheat_10",
    "overrepair",
    "break_volume_abnormality",
    "profit_pressure",
)

# 冻结资产文件名
TRAINING_CSV_NAME = "v004c_training_d1_v001.csv"
LINEAGE_CSV_NAME = "v004c_d1_column_lineage.csv"
COVERAGE_CSV_NAME = "v004c_feature_coverage_v001.csv"
ALLOWLIST_PRIMARY_NAME = "v004c_feature_allowlist_primary_v001.csv"
ALLOWLIST_SENSITIVITY_NAME = "v004c_feature_allowlist_sensitivity_v001.csv"
EXCLUSIONS_NAME = "v004c_feature_exclusions_v001.csv"
DERIVED_SPEC_NAME = "v004c_predeclared_derived_factor_spec_v001.csv"
TABLE_CSV_NAME = "v004c_model_table_v001.csv"
SCHEMA_CSV_NAME = "v004c_model_table_schema_v001.csv"
REVIEW_MD_NAME = "v004c_model_table_review.md"

EXPECTED_SIGNAL_DATES = 42  # 冻结资产当前真实计数 (333 行 / 42 signal dates)


class ModelTableError(RuntimeError):
    """模型表构建校验失败; 任何硬性失败都阻止模型表输出。"""


@dataclass(frozen=True)
class V004CModelTableConfig:
    """模型表构建配置 (只放真正需要的路径与版本)。"""

    stage1_dir: Path
    dictionary_dir: Path
    cache_dir: Path
    output_dir: Path
    coverage_csv: Path | None = None
    source_ref: str = "v004c-d1-dataset-0.1"
    stage2_1_ref: str = "v004c-factor-dictionary-0.1"


# ---------------------------------------------------------------------------
# recent-7d 纯函数 (严格日级公式, 已修正, 确定性)
# ---------------------------------------------------------------------------

def recent_7d_cumulative_return(closes: np.ndarray) -> float | None:
    """close(D1) / close(T-6) - 1 (窗口 7 行相邻日线)。

    base/D1 close 必须 finite 且 > 0, 否则显式缺失 (None)。
    """
    if closes.size < RECENT_7D_WINDOW:
        return None
    base = float(closes[0])
    final = float(closes[-1])
    if not (np.isfinite(base) and base > 0 and np.isfinite(final) and final > 0):
        return None
    return final / base - 1.0


def recent_7d_max_drawdown(highs: np.ndarray, lows: np.ndarray) -> float | None:
    """严格日级最大回撤: prior_peak_t = max(high_s), s < t。

    只允许使用严格早于 t 的历史交易日 high 作为 prior peak;
    当前日 high 不用于当前日 low 的回撤 (日线 OHLC 无法确定同日先后顺序);
    当前日 high 只成为后续交易日的 prior peak。
    所有后续 low 高于 prior peak 时回撤为 0.0; prior peak 必须有限且 > 0。
    """
    if highs.size < 2 or lows.size < 2:
        return None
    prior_peak = float(highs[0])
    drawdowns: list[float] = []
    for t in range(1, min(highs.size, lows.size)):
        low_t = float(lows[t])
        if np.isfinite(prior_peak) and prior_peak > 0 and np.isfinite(low_t):
            drawdowns.append(max(0.0, (prior_peak - low_t) / prior_peak))
        high_t = float(highs[t])
        if np.isfinite(high_t):
            prior_peak = max(prior_peak, high_t)
    return max(drawdowns) if drawdowns else None


def recent_7d_close_position(highs: np.ndarray, lows: np.ndarray, close_d1: float) -> float | None:
    """(close(D1) - 7d_low) / (7d_high - 7d_low)。

    high_7d == low_7d (退化区间) 或任一输入非法时置空 (None -> NaN),
    禁止 inf / divide-by-zero / 填 0 / 填 0.5。
    """
    high_7d = float(np.nanmax(highs))
    low_7d = float(np.nanmin(lows))
    if not (np.isfinite(high_7d) and np.isfinite(low_7d) and high_7d > low_7d):
        return None
    if not (np.isfinite(close_d1)):
        return None
    return (close_d1 - low_7d) / (high_7d - low_7d)


def extract_recent_7d_window(daily: pd.DataFrame, d1_date: str) -> pd.DataFrame | None:
    """取 T-6..D1 共 7 行个股实际有效日线相邻行。

    少于 7 行或 D1 不在缓存 -> None (显式缺失, 失败当前事件, 不换窗口)。
    """
    frame = daily.copy()
    frame["date"] = frame["date"].astype(str)
    if not frame["date"].is_unique or not frame["date"].is_monotonic_increasing:
        raise ModelTableError(
            f"日线缓存 date 必须唯一且严格排序 (unique={frame['date'].is_unique})")
    hits = frame.index[frame["date"] == str(d1_date)]
    if len(hits) == 0:
        return None
    i = int(hits[0])
    if i < RECENT_7D_WINDOW - 1:
        return None
    return frame.loc[i - (RECENT_7D_WINDOW - 1):i,
                     ["date", "open", "high", "low", "close"]].reset_index(drop=True)


def compute_recent_7d_features(daily: pd.DataFrame, d1_date: str) -> dict[str, float]:
    """计算 3 个 recent-7d 字段; 窗口不合法时抛 ModelTableError (fail 当前事件)。"""
    window = extract_recent_7d_window(daily, d1_date)
    if window is None:
        raise ModelTableError(
            f"recent-7d 窗口不足 (需要 D1 前至少 6 个有效交易日, D1={d1_date})")
    numeric = window[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (
            numeric.to_numpy(dtype=float) <= 0).any():
        raise ModelTableError(f"recent-7d 窗口内 OHLC 必须有限且 > 0 (D1={d1_date})")
    closes = numeric["close"].to_numpy(dtype=float)
    highs = numeric["high"].to_numpy(dtype=float)
    lows = numeric["low"].to_numpy(dtype=float)
    cum = recent_7d_cumulative_return(closes)
    dd = recent_7d_max_drawdown(highs, lows)
    pos = recent_7d_close_position(highs, lows, float(closes[-1]))
    out: dict[str, float] = {
        "recent_7d_cumulative_return": cum,
        "recent_7d_max_drawdown": dd,
        "recent_7d_close_position": pos,
    }
    for name, value in out.items():
        if value is None or not np.isfinite(value):
            raise ModelTableError(f"{name} 计算失败 (显式缺失, D1={d1_date})")
    return out


# ---------------------------------------------------------------------------
# 预声明派生字段 (按阶段2.1 冻结公式; 与 v004c_univariate_stability._derive_series 一致)
# ---------------------------------------------------------------------------

def materialize_predeclared_derived(table: pd.DataFrame) -> pd.DataFrame:
    """按 Stage 2.1 预声明派生 spec 计算 5 个派生字段 (profit_chip_ratio 为内部中间量)。"""
    out = table.copy()
    out["board_streak_is_3"] = (
        pd.to_numeric(out["board_streak_before_break"], errors="coerce") == 3
    ).astype(int)
    out["ma5_overheat_10"] = (
        pd.to_numeric(out["d1_close_to_ma5_raw"], errors="coerce") >= 0.10
    ).astype(int)
    out["overrepair"] = np.maximum(
        pd.to_numeric(out["d1_open_to_close_return_raw"], errors="coerce") - 0.03, 0.0)
    out["break_volume_abnormality"] = np.abs(np.log(np.maximum(
        pd.to_numeric(out["break_volume_ratio_vs_board_days"], errors="coerce"), 1e-6)))
    chip = 1.0 - pd.to_numeric(out["volume_above_d1_close_ratio"], errors="coerce")
    out["profit_pressure"] = (
        chip
        * pd.to_numeric(out["d1_low_to_close_recovery"], errors="coerce")
        * np.log1p(pd.to_numeric(out["break_volume_ratio_vs_board_days"], errors="coerce"))
    )
    return out


# ---------------------------------------------------------------------------
# feature contract 校验 (来源: Stage 2.1 规则 + 91bd2009 覆盖结论, 不用 Target)
# ---------------------------------------------------------------------------

def _coverage_row(coverage: pd.DataFrame, name: str) -> pd.Series:
    matches = coverage[coverage["feature_name"] == name]
    if len(matches) != 1:
        raise ModelTableError(f"feature contract 无法在覆盖表定位: {name}")
    return matches.iloc[0]


def _check_available_as_of(name: str, row: pd.Series) -> None:
    """feature 可用时点必须 <= D1_CLOSE (预测时点 D1 收盘后)。

    D0_CLOSE 字段 (如末板日横截面名次) 在 D1 收盘前即可得, 合法。
    """
    available = str(row["available_as_of"])
    if available not in ("D1_CLOSE", "D0_CLOSE"):
        raise ModelTableError(f"{name} 可用时点晚于 D1_CLOSE: {available}")


def _check_new_required_exception(name: str, row: pd.Series) -> None:
    """3 个 recent-7d 字段的授权例外: 不要求存在于旧 Stage 2.1 allowlist。

    授权来源是 feature coverage: NEW_REQUIRED / M2_PRIMARY_CANDIDATE /
    D1_CLOSE / RECENT_7D_PATH。
    """
    if row["coverage_status"] != "NEW_REQUIRED":
        raise ModelTableError(f"{name} 应为 NEW_REQUIRED: {row['coverage_status']}")
    if row["recommended_role"] != "M2_PRIMARY_CANDIDATE":
        raise ModelTableError(f"{name} recommended_role 应为 M2_PRIMARY_CANDIDATE")
    if str(row["available_as_of"]) != "D1_CLOSE":
        raise ModelTableError(f"{name} 可用时点应为 D1_CLOSE: {row['available_as_of']}")
    if str(row["coverage_mechanism"]) != "RECENT_7D_PATH":
        raise ModelTableError(f"{name} 机制应为 RECENT_7D_PATH: {row['coverage_mechanism']}")


def _check_stage21_membership(
    name: str,
    tier: str,
    allow_names: set[str],
    exclusion_status: dict[str, str],
) -> None:
    """字段必须来自 Stage 2.1 允许集合, 且不得来自 Stage 2.1 exclusion。"""
    if name not in allow_names:
        raise ModelTableError(f"{name} 不在 Stage 2.1 {tier} allowlist 中")
    if name in exclusion_status:
        raise ModelTableError(f"{name} 来自 Stage 2.1 exclusion: {exclusion_status[name]}")


def _check_preprocess_and_me(name: str, allow_map: pd.DataFrame, coverage_row: pd.Series) -> None:
    """preprocess_policy 与 mutual_exclusion_group 必须与 Stage 2.1 定义一致。"""
    allow_row = allow_map.loc[name]
    pp_allow = str(allow_row["preprocess_policy"])
    pp_coverage = str(coverage_row["preprocess_policy"])
    if pp_allow != pp_coverage:
        raise ModelTableError(
            f"{name} preprocess_policy 不一致 (Stage2.1={pp_allow}, coverage={pp_coverage})")
    me_allow = str(allow_row["mutual_exclusion_group_id"])
    me_coverage = str(coverage_row["mutual_exclusion_group"])
    na_allow = me_allow in ("nan", "")
    na_coverage = me_coverage in ("nan", "")
    if na_allow != na_coverage or (not na_allow and me_allow != me_coverage):
        raise ModelTableError(
            f"{name} mutual_exclusion_group 不一致 (Stage2.1={me_allow}, coverage={me_coverage})")


def verify_feature_contract(
    coverage: pd.DataFrame,
    allowlist_primary: pd.DataFrame,
    allowlist_sensitivity: pd.DataFrame,
    exclusions: pd.DataFrame,
    training_columns: set[str],
) -> None:
    """校验模块内 feature contract 与冻结资产交叉一致 (fail closed)。

    规则: Stage 2.1 allowlist 决定字段历史合法性与原始结构 (成员/排除/
    preprocess/mutual exclusion); feature coverage 决定当前 V001 模型表
    角色。两者交叉验证, 旧 allowlist 不得覆盖最新 coverage 角色。
    """
    coverage_names = set(coverage["feature_name"])
    primary_allow = {str(n) for n in allowlist_primary["feature_name"]}
    sensitivity_allow = {str(n) for n in allowlist_sensitivity["feature_name"]}
    primary_map = allowlist_primary.set_index("feature_name")
    sensitivity_map = allowlist_sensitivity.set_index("feature_name")
    exclusion_status = {
        str(source): str(status)
        for source, status in zip(exclusions["source_column"], exclusions["admission_status"])
    }

    for name in PRIMARY_FEATURE_COLUMNS:
        if name not in coverage_names:
            raise ModelTableError(f"PRIMARY feature 不在覆盖表中: {name}")
        row = _coverage_row(coverage, name)
        if name in RECENT_7D_FIELDS:
            _check_new_required_exception(name, row)
        else:
            _check_stage21_membership(name, "PRIMARY", primary_allow, exclusion_status)
            _check_preprocess_and_me(name, primary_map, row)
            if row["coverage_status"] != "EXISTING_REUSE":
                raise ModelTableError(f"{name} 应为 EXISTING_REUSE: {row['coverage_status']}")
            if row["recommended_role"] not in ("M1_BASELINE_CANDIDATE", "M2_PRIMARY_CANDIDATE"):
                raise ModelTableError(
                    f"{name} 不再是合法模型候选角色: {row['recommended_role']}")
        _check_available_as_of(name, row)

    for name in SENSITIVITY_FEATURE_COLUMNS:
        if name not in coverage_names:
            raise ModelTableError(f"SENSITIVITY feature 不在覆盖表中: {name}")
        row = _coverage_row(coverage, name)
        if row["coverage_status"] != "EXISTING_REUSE":
            raise ModelTableError(f"{name} 应为 EXISTING_REUSE: {row['coverage_status']}")
        if row["recommended_role"] != "SENSITIVITY_ONLY":
            raise ModelTableError(f"{name} 必须 SENSITIVITY_ONLY: {row['recommended_role']}")
        _check_available_as_of(name, row)
        if name in sensitivity_allow:
            _check_preprocess_and_me(name, sensitivity_map, row)
        elif name in primary_allow:
            # 合法敏感性表达: Stage 2.1 primary 准入 + coverage 明确 SENSITIVITY_ONLY
            _check_preprocess_and_me(name, primary_map, row)
        else:
            raise ModelTableError(f"{name} 不在 Stage 2.1 allowlist (primary/sensitivity) 中")

    # 排除项不得混入 feature universe
    excluded = (
        set(CONTEXT_ONLY_COLUMNS) | set(DEFER_EXCLUDED_COLUMNS)
        | set(REDUNDANT_EXCLUDED_COLUMNS)
        | {"d2_open_daily", "d3_high_daily", "recognition_score",
           "v002_rank", "v004a_probability"}
    )
    overlap = excluded & set(FEATURE_COLUMNS)
    if overlap:
        raise ModelTableError(f"已排除字段混入 feature universe: {sorted(overlap)}")

    # 已有字段必须在训练表中 (派生字段的源字段必须存在)
    computed = set(PREDECLARED_DERIVED_FIELDS) | set(RECENT_7D_FIELDS)
    missing = [c for c in FEATURE_COLUMNS if c not in computed and c not in training_columns]
    if missing:
        raise ModelTableError(f"feature 不存在于冻结训练表: {missing}")

    # Stage 2.1 互斥组: PRIMARY 内不得共享同一互斥组 (primary canonical / sensitivity alternative)
    primary_me: dict[str, str] = {}
    for name in PRIMARY_FEATURE_COLUMNS:
        me = str(_coverage_row(coverage, name)["mutual_exclusion_group"])
        if me and me != "nan":
            if me in primary_me:
                raise ModelTableError(
                    f"PRIMARY 内互斥组冲突: {me} 同时含 {primary_me[me]} 与 {name}")
            primary_me[me] = name


def scan_for_leakage(feature_columns: tuple[str, ...]) -> None:
    """硬泄漏扫描: 任何 feature 名称含禁止 token 即失败。"""
    hits = sorted({c for c in feature_columns
                   if any(token in c.lower() for token in FORBIDDEN_FEATURE_TOKENS)})
    if hits:
        raise ModelTableError(f"泄漏扫描失败: feature 名称含禁止 token: {hits}")


# ---------------------------------------------------------------------------
# 表完整性检查 (硬失败)
# ---------------------------------------------------------------------------

def _check_table(table: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    checks: dict = {}
    checks["rows"] = int(len(table))
    checks["unique_event_id"] = bool(table["event_id"].is_unique)
    checks["signal_dates"] = int(table["signal_date"].nunique())
    checks["code_valid"] = bool(table["code"].astype(str).str.fullmatch(r"\d{6}").all())
    checks["signal_date_valid"] = bool(
        pd.to_datetime(table["signal_date"], format="%Y-%m-%d", errors="coerce").notna().all())
    checks["signal_break_consistent"] = bool(
        (table["signal_date"].astype(str) == table["break_date"].astype(str)).all())

    # primary 严格缺失策略: 任何缺失/非有限 -> FAIL MODEL TABLE BUILD
    primary_missing = int(table[list(PRIMARY_FEATURE_COLUMNS)].isna().sum().sum())
    primary_inf = int((
        table[list(PRIMARY_FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .apply(lambda s: np.isinf(s).sum()).sum()))
    checks["primary_missing"] = primary_missing
    checks["primary_inf"] = primary_inf

    # sensitivity 结构性缺失只记录 (schema/review 明确)
    sensitivity_missing = table[list(SENSITIVITY_FEATURE_COLUMNS)].isna().sum()
    checks["sensitivity_missing_detail"] = {
        str(k): int(v) for k, v in sensitivity_missing[sensitivity_missing > 0].items()}

    # 严格二元验证 (复用 Stage 1 helper; 禁止隐式 truthiness)
    binary_columns = [c for c in FEATURE_COLUMNS
                      if str(_coverage_row(coverage, c)["preprocess_policy"]) == "RAW_BINARY"]
    for column in binary_columns:
        try:
            normalize_binary_series(table[column], column_name=column, allow_missing=False)
        except DatasetValidationError as exc:
            raise ModelTableError(f"binary 非法值: {exc}") from exc
    try:
        normalize_binary_series(table["target7_daily_d2open_d3high"],
                                column_name="target7_daily_d2open_d3high")
    except DatasetValidationError as exc:
        raise ModelTableError(f"Target 非严格二元: {exc}") from exc
    checks["binary_columns_validated"] = sorted(binary_columns)

    # 全部数值 feature 必须 finite (含 sensitivity)
    for column in FEATURE_COLUMNS:
        series = pd.to_numeric(table[column], errors="coerce")
        if np.isinf(series).any():
            raise ModelTableError(f"feature {column} 含 ±inf")

    # 关键硬门
    if checks["rows"] != len(table.index.unique()):
        raise ModelTableError(f"行数不一致: {checks['rows']}")
    if not checks["unique_event_id"]:
        raise ModelTableError("event_id 必须唯一")
    if checks["signal_dates"] != EXPECTED_SIGNAL_DATES:
        raise ModelTableError(
            f"signal dates 应为 {EXPECTED_SIGNAL_DATES}, 实际 {checks['signal_dates']}"
            " (先调查冻结资产原因, 不得修改期望值迎合输出)")
    if not checks["code_valid"]:
        raise ModelTableError("存在非法 code (必须 6 位数字)")
    if not checks["signal_date_valid"]:
        raise ModelTableError("存在非法 signal_date (必须 YYYY-MM-DD)")
    if not checks["signal_break_consistent"]:
        raise ModelTableError("signal_date 与 break_date 必须逐行一致 (不产生两套切分语义)")
    if primary_missing > 0:
        raise ModelTableError(f"PRIMARY feature 缺失 {primary_missing} 个值: FAIL MODEL TABLE BUILD")
    if primary_inf > 0:
        raise ModelTableError(f"PRIMARY feature 含 {primary_inf} 个 ±inf")
    for field in RECENT_7D_FIELDS:
        series = pd.to_numeric(table[field], errors="coerce")
        if series.isna().any() or np.isinf(series).any():
            raise ModelTableError(f"{field} 必须 333/333 非空且 finite")
    return checks


# ---------------------------------------------------------------------------
# 原子输出
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_written_outputs(tmp_dir: Path, expected_checks: dict) -> None:
    """rename 前对磁盘文件做最终验证 (round-trip 读回)。"""
    table = pd.read_csv(tmp_dir / TABLE_CSV_NAME, dtype={"code": str})
    expected_columns = (list(IDENTIFIER_COLUMNS) + list(PRIMARY_FEATURE_COLUMNS)
                        + list(SENSITIVITY_FEATURE_COLUMNS) + list(LABEL_COLUMNS)
                        + list(AUDIT_COLUMNS))
    if list(table.columns) != expected_columns:
        raise ModelTableError(
            f"原子输出读回列集不一致: 多出 {sorted(set(table.columns) - set(expected_columns))}, "
            f"缺少 {sorted(set(expected_columns) - set(table.columns))}")
    if len(table) != expected_checks["rows"]:
        raise ModelTableError("原子输出读回行数不一致")
    if not table["event_id"].is_unique:
        raise ModelTableError("原子输出读回 event_id 不唯一")
    if table["signal_date"].nunique() != expected_checks["signal_dates"]:
        raise ModelTableError("原子输出读回 signal dates 不一致")
    for field in RECENT_7D_FIELDS:
        series = pd.to_numeric(table[field], errors="coerce")
        if series.isna().any() or np.isinf(series).any():
            raise ModelTableError(f"原子输出读回 {field} 非空/finite 检查失败")
    for column in LABEL_COLUMNS:
        try:
            normalize_binary_series(table[column], column_name=column)
        except DatasetValidationError as exc:
            raise ModelTableError(f"原子输出读回 Target 非严格二元: {exc}") from exc

    # schema 与 table 不得漂移: 行数一致且 column_name 一一对应
    schema = pd.read_csv(tmp_dir / SCHEMA_CSV_NAME, encoding="utf-8-sig")
    if len(schema) != len(table.columns):
        raise ModelTableError(
            f"原子输出 schema 行数 {len(schema)} != table 列数 {len(table.columns)}")
    if list(schema["column_name"]) != list(table.columns):
        raise ModelTableError("原子输出 schema 列名与 table 列不一致 (顺序或成员)")


# ---------------------------------------------------------------------------
# 正式构建
# ---------------------------------------------------------------------------

def build_v004c_model_table(config: V004CModelTableConfig) -> dict:
    """构建统一建模表 (offline deterministic; 原子输出)。"""
    stage1_dir = Path(config.stage1_dir)
    dictionary_dir = Path(config.dictionary_dir)
    cache_dir = Path(config.cache_dir)
    output_dir = Path(config.output_dir)
    coverage_csv = Path(config.coverage_csv) if config.coverage_csv else (
        stage1_dir.parent / COVERAGE_CSV_NAME)

    if output_dir.exists():
        raise ModelTableError(f"输出目录已存在, 禁止覆盖正式结果: {output_dir}")

    for path in (stage1_dir, dictionary_dir, cache_dir, coverage_csv):
        if not path.exists():
            raise ModelTableError(f"输入不存在: {path}")

    training_path = stage1_dir / TRAINING_CSV_NAME
    coverage = pd.read_csv(coverage_csv, encoding="utf-8-sig")
    allowlist_primary = pd.read_csv(dictionary_dir / ALLOWLIST_PRIMARY_NAME, encoding="utf-8-sig")
    allowlist_sensitivity = pd.read_csv(dictionary_dir / ALLOWLIST_SENSITIVITY_NAME,
                                        encoding="utf-8-sig")
    exclusions = pd.read_csv(dictionary_dir / EXCLUSIONS_NAME, encoding="utf-8-sig")

    # 1. 最小读取训练表 (identifiers + 已有 feature + label + audit; 不读其它列)
    existing_features = [c for c in FEATURE_COLUMNS
                         if c not in PREDECLARED_DERIVED_FIELDS and c not in RECENT_7D_FIELDS]
    usecols = (list(IDENTIFIER_COLUMNS) + existing_features
               + list(LABEL_COLUMNS) + list(AUDIT_COLUMNS))
    training = pd.read_csv(training_path, dtype={"code": str}, usecols=usecols)
    missing_columns = sorted(set(usecols) - set(training.columns))
    unexpected_columns = sorted(set(training.columns) - set(usecols))
    if missing_columns or unexpected_columns:
        raise ModelTableError(
            f"训练表列集与声明不一致 (missing={missing_columns}, "
            f"unexpected={unexpected_columns})")

    # 2. feature contract 校验 (Stage 2.1 allowlist + coverage 交叉验证; 不用 Target)
    verify_feature_contract(coverage, allowlist_primary, allowlist_sensitivity,
                            exclusions, set(training.columns))
    scan_for_leakage(FEATURE_COLUMNS)

    # 3. 派生字段 (按冻结公式)
    table = materialize_predeclared_derived(training)

    # 4. recent-7d 三字段 (离线确定性; 从现有 data/cache/daily)
    cache: dict[str, pd.DataFrame] = {}
    recent_rows: list[dict[str, object]] = []
    for _, row in training.iterrows():
        code = str(row["code"])
        if code not in cache:
            cache[code] = pd.read_pickle(cache_dir / f"{code}_daily.pkl")
        values = compute_recent_7d_features(cache[code], str(row["break_date"]))
        values["event_id"] = row["event_id"]
        recent_rows.append(values)
    recent = pd.DataFrame(recent_rows).set_index("event_id")
    for field in RECENT_7D_FIELDS:
        table[field] = recent.loc[table["event_id"], field].to_numpy()

    # 5. 列顺序: identifier -> primary -> sensitivity -> label -> audit
    column_order = (list(IDENTIFIER_COLUMNS) + list(PRIMARY_FEATURE_COLUMNS)
                    + list(SENSITIVITY_FEATURE_COLUMNS) + list(LABEL_COLUMNS)
                    + list(AUDIT_COLUMNS))
    table = table[column_order]

    # 6. 完整性检查 (含 strict missing / binary / leak / inf)
    checks = _check_table(table, coverage)
    checks["feature_count"] = len(FEATURE_COLUMNS)
    checks["primary_count"] = len(PRIMARY_FEATURE_COLUMNS)
    checks["sensitivity_count"] = len(SENSITIVITY_FEATURE_COLUMNS)
    checks["identifier_count"] = len(IDENTIFIER_COLUMNS)
    checks["label_count"] = len(LABEL_COLUMNS)
    checks["audit_count"] = len(AUDIT_COLUMNS)
    checks["columns"] = int(table.shape[1])
    for field in RECENT_7D_FIELDS:
        series = pd.to_numeric(table[field], errors="coerce")
        checks[f"{field}_min"] = float(series.min())
        checks[f"{field}_median"] = float(series.median())
        checks[f"{field}_max"] = float(series.max())
    target = table["target7_daily_d2open_d3high"]
    checks["target_positive"] = int(target.astype(bool).sum())
    checks["target_negative"] = int((~target.astype(bool)).sum())
    checks["target_binary"] = bool(target.isin([True, False]).all())
    checks["output_dir"] = str(output_dir)

    # 7. schema 与 review 内容 (schema 只描述 model table 实际列)
    schema_rows = _build_schema_rows(table, coverage, training_path)
    if list(schema_rows["column_name"]) != list(table.columns):
        raise ModelTableError(
            "schema column_name 必须与 model table 列一一对应且顺序一致")
    review_md = _build_review_md(config, table, coverage, checks, training_path)
    files: dict[str, str] = {
        TABLE_CSV_NAME: table.to_csv(index=False, encoding="utf-8-sig"),
        SCHEMA_CSV_NAME: schema_rows.to_csv(index=False, encoding="utf-8-sig"),
        REVIEW_MD_NAME: review_md,
    }

    # 8. 原子输出: sibling tmp -> 写 3 文件 -> 磁盘读回全验证 -> rename
    tmp_dir = output_dir.parent / (output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        for name, content in files.items():
            (tmp_dir / name).write_text(content, encoding="utf-8-sig")
        _validate_written_outputs(tmp_dir, checks)
        tmp_dir.rename(output_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    return checks


# ---------------------------------------------------------------------------
# schema / review 生成
# ---------------------------------------------------------------------------

def _build_schema_rows(table: pd.DataFrame, coverage: pd.DataFrame,
                       training_path: Path) -> pd.DataFrame:
    coverage_map = coverage.set_index("feature_name")
    rows: list[dict] = []
    for column in table.columns:
        if column in FEATURE_COLUMNS:
            row = coverage_map.loc[column]
            rows.append({
                "column_name": column,
                "role": "FEATURE",
                "source": _feature_source(column),
                "available_as_of": str(row["available_as_of"]),
                "mechanism": str(row["coverage_mechanism"]),
                "preprocess_policy": _feature_preprocess(column, row),
                "model_universe": ("PRIMARY" if column in PRIMARY_FEATURE_COLUMNS
                                   else "SENSITIVITY"),
                "formula_or_definition": str(row["formula_or_definition"]),
                "forbidden_reason": "-",
            })
        elif column in IDENTIFIER_COLUMNS:
            rows.append({
                "column_name": column,
                "role": "IDENTIFIER",
                "source": f"{TRAINING_CSV_NAME} (Stage 1 冻结)",
                "available_as_of": "D1_CLOSE",
                "mechanism": "-",
                "preprocess_policy": "NO_MODEL_INPUT",
                "model_universe": "NONE",
                "formula_or_definition": _identifier_definition(column),
                "forbidden_reason": "-",
            })
        elif column in LABEL_COLUMNS:
            rows.append({
                "column_name": column,
                "role": "LABEL",
                "source": f"{TRAINING_CSV_NAME} (Stage 1 冻结)",
                "available_as_of": "D3_CLOSE (LABEL, 预测时点 D1 收盘后不可得)",
                "mechanism": "LABEL",
                "preprocess_policy": "NO_MODEL_INPUT",
                "model_universe": "NONE",
                "formula_or_definition": "1[D3 high / D2 open - 1 >= 0.07] (冻结定义)",
                "forbidden_reason": "LABEL 只属于 LABEL 角色, 绝不允许出现在 FEATURE_COLUMNS",
            })
        else:
            rows.append({
                "column_name": column,
                "role": "AUDIT_ONLY",
                "source": f"{TRAINING_CSV_NAME} (Stage 1 冻结)",
                "available_as_of": ("D1_CLOSE" if column == "break_date"
                                    else "POST_D3 (AUDIT_ONLY, 预测时点不可得)"),
                "mechanism": "-",
                "preprocess_policy": "NO_MODEL_INPUT",
                "model_universe": "NONE",
                "formula_or_definition": _audit_definition(column),
                "forbidden_reason": "-",
            })
    # 注意: schema 只描述 model table 实际输出列; 被排除的候选字段与排除
    # 原因由 feature coverage 资产 (v004c_feature_coverage_v001.csv) 负责,
    # 不在此重复维护第二套排除清单。
    return pd.DataFrame(rows, columns=[
        "column_name", "role", "source", "available_as_of", "mechanism",
        "preprocess_policy", "model_universe", "formula_or_definition", "forbidden_reason"])


def _feature_source(column: str) -> str:
    if column in RECENT_7D_FIELDS:
        return "data/cache/daily 日线缓存离线计算 (7 行相邻日线窗口)"
    if column in PREDECLARED_DERIVED_FIELDS:
        return f"{DERIVED_SPEC_NAME} 预声明派生, 由冻结源字段计算"
    return f"{TRAINING_CSV_NAME} (Stage 1 冻结)"


def _feature_preprocess(column: str, row: pd.Series) -> str:
    if column in RECENT_7D_FIELDS:
        return "FOLD_CLIP_Z (临时约定, 训练阶段 fold 内决定)"
    return str(row["preprocess_policy"])


def _identifier_definition(column: str) -> str:
    definitions = {
        "event_id": "code + '_' + break_date (冻结 v0.2.2 来源)",
        "code": "6 位股票代码",
        "signal_date": "D1 信号日 (正式切分主键; walk-forward 统一按此切分)",
        "board_streak_before_break": "断板前连板数 {2,3}; board_streak_is_3 由它派生",
    }
    return definitions.get(column, "-")


def _audit_definition(column: str) -> str:
    definitions = {
        "break_date": "D1 日期 (legacy identifier; 与 signal_date 逐行一致)",
        "daily_label_quality_ok": "日线标签质量标志 (AUDIT_ONLY)",
        "target_training_eligible": "训练资格标志 (AUDIT_ONLY)",
        "sample_role_v022": "v0.2.2 样本角色 (AUDIT_ONLY)",
        "d1_factor_quality_ok": "D1 因子质量标志 (AUDIT_ONLY)",
    }
    return definitions.get(column, "-")


def _build_review_md(config: V004CModelTableConfig, table: pd.DataFrame,
                     coverage: pd.DataFrame, checks: dict, training_path: Path) -> str:
    stage1_dir = Path(config.stage1_dir)
    dictionary_dir = Path(config.dictionary_dir)
    coverage_csv = Path(config.coverage_csv) if config.coverage_csv else (
        stage1_dir.parent / COVERAGE_CSV_NAME)
    lines: list[str] = []
    lines.append("# v004c 统一建模表 v001 (v004c_model_table_v001)")
    lines.append("")
    lines.append(f"- 阶段: 正式多变量模型开发前的最后一个数据工程阶段 (构建模型输入表, 不训练模型)")
    lines.append(f"- 构建日期: 2026-08-07")
    lines.append(f"- 分支: research-sample-analysis (模型表构建 commit ab2b067; "
                 f"特征覆盖 commit 91bd2009)")
    lines.append("")
    lines.append("## 0. 输入版本与来源")
    lines.append("")
    lines.append(f"1. **Stage 1 输入版本**: `{stage1_dir.name}` (ref `{config.source_ref}`)")
    for name in (TRAINING_CSV_NAME, LINEAGE_CSV_NAME):
        path = stage1_dir / name
        lines.append(f"   - `{name}` SHA256: `{_sha256(path)}`")
    lines.append(f"2. **Stage 2.1 输入版本**: `{dictionary_dir.name}` (ref `{config.stage2_1_ref}`)")
    for name in (ALLOWLIST_PRIMARY_NAME, ALLOWLIST_SENSITIVITY_NAME,
                 EXCLUSIONS_NAME, DERIVED_SPEC_NAME):
        path = dictionary_dir / name
        lines.append(f"   - `{name}` SHA256: `{_sha256(path)}`")
    lines.append(f"3. **特征覆盖结论**: `{coverage_csv.name}` SHA256: `{_sha256(coverage_csv)}` (commit 91bd2009)")
    lines.append(f"4. **日线缓存**: `{Path(config.cache_dir)}` (offline, 未联网, 未修改 180/120 窗口)")
    lines.append("")
    lines.append("## 1. 表结构总览")
    lines.append("")
    lines.append(f"- 总行数: {checks['rows']} (一行一个 D1 事件)")
    lines.append(f"- signal dates: {checks['signal_dates']}")
    lines.append(f"- model table 列数: {checks['columns']}")
    lines.append(f"- schema rows: {checks['columns']} (schema 只描述实际输出表列, 与 table 列一一对应)")
    lines.append(f"- feature 数量: {checks['feature_count']} "
                 f"(primary {checks['primary_count']} + sensitivity {checks['sensitivity_count']})")
    lines.append(f"- identifier 数量: {checks['identifier_count']}")
    lines.append(f"- label 数量: {checks['label_count']}")
    lines.append(f"- audit 数量: {checks['audit_count']}")
    lines.append(f"- event_id 唯一: {checks['unique_event_id']}")
    lines.append("")
    lines.append("## 2. 列角色")
    lines.append("")
    lines.append("- **IDENTIFIER**: " + ", ".join(IDENTIFIER_COLUMNS))
    lines.append("- **FEATURE (PRIMARY, 29)**: " + ", ".join(PRIMARY_FEATURE_COLUMNS))
    lines.append("- **FEATURE (SENSITIVITY, 26)**: " + ", ".join(SENSITIVITY_FEATURE_COLUMNS))
    lines.append("- **LABEL**: " + ", ".join(LABEL_COLUMNS))
    lines.append("- **AUDIT_ONLY**: " + ", ".join(AUDIT_COLUMNS))
    lines.append("")
    lines.append("## 3. 3 个 recent-7d 字段 (RECENT_7D_PATH 机制)")
    lines.append("")
    lines.append("窗口: T-6..D1 共 7 个个股实际有效日线相邻行 (不使用自然日; 不足 7 行失败当前事件, 不换窗口)。")
    lines.append("")
    lines.append("| field | 定义 | 覆盖率 | min | median | max |")
    lines.append("|---|---|---|---|---|---|")
    for field in RECENT_7D_FIELDS:
        lines.append(f"| {field} | {_coverage_row(coverage, field)['formula_or_definition']} | "
                     f"333/333 (0 缺失) | {checks[f'{field}_min']:.6f} | "
                     f"{checks[f'{field}_median']:.6f} | {checks[f'{field}_max']:.6f} |")
    lines.append("")
    lines.append("严格日级最大回撤: `prior_peak_t = max(high_s), s < t` (只用严格较早交易日 high, "
                 "不使用同日 high/low 先后顺序; 当前日 high 只成为后续 prior peak; "
                 "所有后续 low 高于 prior peak 时回撤 = 0)。")
    lines.append("")
    lines.append("recent-7d 三字段的授权来源是 feature coverage "
                 "(coverage_status=NEW_REQUIRED / recommended_role=M2_PRIMARY_CANDIDATE / "
                 "available_as_of=D1_CLOSE / mechanism=RECENT_7D_PATH), "
                 "不要求存在于旧 Stage 2.1 allowlist。")
    lines.append("")
    lines.append("## 4. 六月/七月数据完整性描述 (只允许行数与标签计数, 禁止模型表现)")
    lines.append("")
    lines.append("| 月份 | 行数 | signal date 数 | Target7 阳性计数 |")
    lines.append("|---|---|---|---|")
    months = pd.to_datetime(table["signal_date"]).dt.strftime("%Y-%m")
    for month, group in table.groupby(months, sort=True):
        lines.append(f"| {month} | {len(group)} | {group['signal_date'].nunique()} | "
                     f"{int(group['target7_daily_d2open_d3high'].astype(bool).sum())} |")
    lines.append("")
    lines.append("## 5. Target 处理")
    lines.append("")
    lines.append(f"- `target7_daily_d2open_d3high`: binary (True {checks['target_positive']} / "
                 f"False {checks['target_negative']}), 严格二元验证通过, "
                 f"未标准化/未裁剪/未做任何预处理。")
    lines.append(f"- 只属于 LABEL 角色 ({checks['label_count']} 列), 绝不在 FEATURE_COLUMNS 中; "
                 f"feature contract 来自 Stage 2.1 规则 + 91bd2009 覆盖结论, 与标签表现无关。")
    lines.append("")
    lines.append("## 6. 数据质量与泄漏检查")
    lines.append("")
    lines.append(f"- 未来字段 (D2/D3/POST_D3/future/outcome/tail_loss 等 token 扫描): 无")
    lines.append(f"- 旧模型输出 (v002/v004a/v004b/v005/recognition/policy): 无")
    lines.append(f"- duplicate alias (ED01-13 别名列): 无 (仅 canonical 列进入 universe)")
    lines.append(f"- primary 缺失: {checks['primary_missing']} (任何缺失 => FAIL MODEL TABLE BUILD)")
    lines.append(f"- primary ±inf: {checks['primary_inf']}")
    lines.append(f"- binary 非法值: 0 (严格二元验证: {', '.join(checks['binary_columns_validated'])})")
    lines.append(f"- 全部数值 feature ±inf: 0")
    lines.append(f"- event_id 唯一: {checks['unique_event_id']}")
    lines.append(f"- code 合法 (6 位数字): {checks['code_valid']}")
    lines.append(f"- signal_date 有效 (YYYY-MM-DD 且存在于个股日线缓存): {checks['signal_date_valid']}")
    lines.append(f"- signal_date == break_date 逐行一致: {checks['signal_break_consistent']}")
    lines.append("")
    lines.append("## 7. Sensitivity 字段结构性缺失 (允许, 明确记录)")
    lines.append("")
    lines.append("以下 sensitivity 字段在冻结 Stage 1 中既有结构性缺失, 模型表保留原值 (不填充):")
    lines.append("")
    lines.append("| feature | missing |")
    lines.append("|---|---|")
    for name, count in sorted(checks["sensitivity_missing_detail"].items()):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("未来第一版 M1/M2 预计只使用完整 primary 字段。")
    lines.append("")
    lines.append("## 8. 明确排除的字段 (排除清单与原因由 feature coverage 资产负责)")
    lines.append("")
    lines.append("以下字段不得进入 PRIMARY/SENSITIVITY universe; 逐字段排除原因见 "
                 f"`{COVERAGE_CSV_NAME}` (coverage_status / recommended_role / reason 列), "
                 "model-table schema 不重复维护排除清单:")
    lines.append("")
    for group_label, names in (
        ("CONTEXT_ONLY (REUSE_AS_CONTEXT, 不升级为 primary)",
         CONTEXT_ONLY_COLUMNS),
        ("DEFER_NOT_REQUIRED_FOR_V001", DEFER_EXCLUDED_COLUMNS),
        ("REDUNDANT (可由已有字段精确确定)", REDUNDANT_EXCLUDED_COLUMNS),
        ("FORBIDDEN (未来/标签/旧模型输出)", ("d2_open_daily", "d3_high_daily",
                                            "recognition_score", "v002_rank",
                                            "v004a_probability")),
    ):
        lines.append(f"- **{group_label}**: {', '.join(names)}")
    lines.append(f"- 其余 REDUNDANT / NOT_NEEDED / FORBIDDEN / AUDIT_ONLY / DERIVE_ONLY "
                 f"候选字段同样不进入模型表; 全部由 `{COVERAGE_CSV_NAME}` 统一记录。")
    lines.append(f"- 本 schema (`{SCHEMA_CSV_NAME}`) 只描述实际输出表列 "
                 f"({checks['columns']} 行 = model table 列数), 不含任何不在表中的字段。")
    lines.append("")
    lines.append("## 9. M1/M2 槽位 (本阶段不选模型)")
    lines.append("")
    lines.append("- **M1 D1 STRUCTURE BASELINE**: exact features = **NOT_FROZEN**")
    lines.append("- **M2 INTEGRATED PRIMARY**: exact features = **NOT_FROZEN**")
    lines.append("- 不输出 m1_features.json / m2_features.json / candidate_model_specs")
    lines.append("")
    lines.append("## 10. 预处理声明")
    lines.append("")
    lines.append("- 本阶段未做任何全样本预处理 (无 z-score / winsorize / P01-P99 clip / 标准化)。")
    lines.append("- 模型表保存 raw legal feature values; 预处理参数必须由下一阶段每个 "
                 "walk-forward fold 只用训练 fold 计算。")
    lines.append("- recent-7d 三字段在 schema 中的 preprocess_policy 标记为 "
                 "FOLD_CLIP_Z (临时约定, 与 Stage 2.1 连续变量约定一致, 不构成模型冻结); "
                 "实际 clip 阈值、均值、标准差等参数只能在下一阶段每个 training fold "
                 "内部拟合, 本阶段未生成任何全样本预处理参数。")
    lines.append("")
    lines.append("## 11. 原子输出")
    lines.append("")
    lines.append(f"- 构建到 sibling tmp 目录 → 写 3 文件 → 磁盘读回全验证 → rename → 正式目录")
    lines.append(f"- 失败时正式目录不存在; 若正式目录已存在则 FAIL (不覆盖)")
    lines.append(f"- 正式输出目录: `{checks['output_dir']}`")
    lines.append("")
    lines.append("## 12. 最终状态")
    lines.append("")
    lines.append(f"- 完成: `{TABLE_CSV_NAME}` 确定性生成 "
                 f"({checks['rows']} 行 / {checks['signal_dates']} signal dates / "
                 f"{checks['feature_count']} features) ✓")
    lines.append(f"- 训练: 无 (本阶段禁止 Logistic Regression / M0 / M1 / M2)")
    lines.append(f"- 预测: 无 (无概率 / rank / Top3 输出)")
    lines.append(f"- M1/M2 冻结: 无")
    return "\n".join(lines) + "\n"

# -*- coding: utf-8 -*-
"""v004c Pairwise v1 Feature Contract — 冻结特征契约 (只读数据 + 审计函数)。

本模块定义 PAIRWISE_V1_FEATURE_CONTRACT (53 个 FEATURE) 与完整候选
inventory (含全部被排除候选及其决策), 供
tools/build_v004c_pairwise_v1_feature_contract.py 确定性消费。

严格约束 (本模块):
- 不训练 / 不筛选 / 不调参 / 不计算模型概率 / 不生成排名;
- 特征选择不得基于 Target / tail-loss / 单因子 AUC / IC / 旧模型表现;
- 所有 FEATURE available_as_of <= D1_CLOSE (D0_CLOSE 横截面字段合法);
- FEATURE 不包含: 未来信息, 模型输出, Target 派生, 手动综合评分,
  修复状态综合 (DIVERGENCE/RECLAIM/DAMAGE/SUPPLY), v004a/v002/repair-state 输出;
- 不做任何全样本预处理; 缺失保持 NaN (FOLD_ONLY_IMPUTATION_LATER);
- 不做相关性删除; 仅允许 exact duplicate / exact deterministic transform 剔除;
- 不创建交互列 (feature_x_board3), 只冻结 base X + board group indicator;
- signal_date = GROUP_KEY (非 FEATURE); source_window = AUDIT_ONLY;
- 特征顺序从 1 开始, 唯一/连续/确定; 未来 Pairwise Ridge 只按此顺序读 X。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 决策枚举 (feature contract 每个候选字段一个)
# ---------------------------------------------------------------------------
FEATURE = "FEATURE"
IDENTIFIER = "IDENTIFIER"
LABEL_ONLY = "LABEL_ONLY"
AUDIT_ONLY = "AUDIT_ONLY"
EXCLUDE_MANUAL_COMPOSITE = "EXCLUDE_MANUAL_COMPOSITE"
EXCLUDE_MODEL_OUTPUT = "EXCLUDE_MODEL_OUTPUT"
EXCLUDE_TARGET_LINEAGE = "EXCLUDE_TARGET_LINEAGE"
EXCLUDE_SOURCE_UNAVAILABLE = "EXCLUDE_SOURCE_UNAVAILABLE"
EXCLUDE_EXACT_DUPLICATE_ALIAS = "EXCLUDE_EXACT_DUPLICATE_ALIAS"
EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM = "EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM"
EXCLUDE_DEPRECATED = "EXCLUDE_DEPRECATED"
EXCLUDE_ALL_MISSING = "EXCLUDE_ALL_MISSING"
PENDING_EVIDENCE = "PENDING_EVIDENCE"

# ---------------------------------------------------------------------------
# 泄漏扫描 token (FEATURE 名称硬禁止; 市场横截面 rank 由 lineage 单独判定)
# ---------------------------------------------------------------------------
FEATURE_FORBIDDEN_TOKENS = (
    "target", "label", "tail", "d2_", "d3_", "future", "outcome",
    "realized", "post_", "prediction", "prob", "model", "v004a", "v004b",
    "v002", "repair_state", "recognition", "policy", "score",
)

# 名称含 "rank" 的 FEATURE 必须属于市场横截面 (D0_CROSS_SECTION 且
# raw_dependencies 含 market 数据) 才合法; 否则视为模型输出泄漏。
RANK_REQUIRED_SEMANTIC_GROUP = "D0_CROSS_SECTION"
RANK_MARKET_DEPENDENCY_KEYWORDS = ("amount", "turnover", "volume", "pool")

# ---------------------------------------------------------------------------
# semantic_group (mechanism_group) 合法值
# ---------------------------------------------------------------------------
SEMANTIC_GROUPS = (
    "BOARD_HISTORY",
    "D0_CROSS_SECTION",
    "D1_PRICE_ACTION",
    "D1_VOLUME_ACTIVITY",
    "D1_MA_POSITION",
    "D1_CHIP_DISTRIBUTION",
    "D1_LATE_DAY_PRESSURE",
    "D1_VWAP_POSITION",
    "POOL_MEMBERSHIP",
    "RECENT_7D_PATH",
)

# information_class 合法值
INFORMATION_CLASSES = ("RECENT_PATH", "KNOWN_STATE", "D1_STRUCTURE", "EVENT_CONTEXT")

SEMANTIC_GROUP_INFORMATION_CLASS = {
    "BOARD_HISTORY": "KNOWN_STATE",
    "D0_CROSS_SECTION": "KNOWN_STATE",
    "D1_PRICE_ACTION": "D1_STRUCTURE",
    "D1_VOLUME_ACTIVITY": "D1_STRUCTURE",
    "D1_MA_POSITION": "KNOWN_STATE",
    "D1_CHIP_DISTRIBUTION": "D1_STRUCTURE",
    "D1_LATE_DAY_PRESSURE": "D1_STRUCTURE",
    "D1_VWAP_POSITION": "D1_STRUCTURE",
    "POOL_MEMBERSHIP": "KNOWN_STATE",
    "RECENT_7D_PATH": "RECENT_PATH",
}

# ---------------------------------------------------------------------------
# 53-FEATURE 冻结契约 (feature_order 从 1 开始, 唯一/连续/确定)
# 字段: feature_order, feature_name, semantic_group, information_class,
#       dtype, formula, raw_dependencies, source, lookback, available_as_of,
#       missing_semantics, expected_range_or_domain, notes
# ---------------------------------------------------------------------------
FEATURE_CONTRACT: List[Dict[str, Any]] = [
    # ---- BOARD_HISTORY (6) ----
    {
        "feature_name": "board_streak_is_3",
        "semantic_group": "BOARD_HISTORY",
        "dtype": "binary",
        "formula": "1[board_streak_before_break == 3]",
        "raw_dependencies": ("board_streak_before_break",),
        "source": "v002_derived",
        "lookback": "board history",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if board_streak_before_break missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "二板/三板事件状态二元项 (PREDECLARED_DERIVED, D1_CLOSE 可得)",
    },
    {
        "feature_name": "max_board_streak_20d",
        "semantic_group": "BOARD_HISTORY",
        "dtype": "ordinal",
        "formula": "20 窗口 (obs-19..obs) 内日线连续涨停天数最大值",
        "raw_dependencies": ("daily OHLCVA",),
        "source": "daily_cache_rebuild",
        "lookback": "20d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if insufficient daily history",
        "expected_range_or_domain": ">=1 (正整数)",
        "notes": "20 日最大连板 (含本轮), 日线 is_limit_up_day 判定",
    },
    {
        "feature_name": "recent_limit_up_count_10d",
        "semantic_group": "BOARD_HISTORY",
        "dtype": "ordinal",
        "formula": "obs-9..obs 窗口内日线涨停天数之和",
        "raw_dependencies": ("daily OHLCVA",),
        "source": "daily_cache_rebuild",
        "lookback": "10d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if insufficient daily history",
        "expected_range_or_domain": "0..10",
        "notes": "近 10 交易日涨停次数 (v0.2.2 口径, 日线 flags)",
    },
    {
        "feature_name": "recent_limit_up_count_20d",
        "semantic_group": "BOARD_HISTORY",
        "dtype": "ordinal",
        "formula": "obs-19..obs 窗口内日线涨停天数之和",
        "raw_dependencies": ("daily OHLCVA",),
        "source": "daily_cache_rebuild",
        "lookback": "20d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if insufficient daily history",
        "expected_range_or_domain": "0..20",
        "notes": "近 20 交易日涨停次数 (v0.2.2 口径)",
    },
    {
        "feature_name": "recent_pool_appearance_count_10d",
        "semantic_group": "BOARD_HISTORY",
        "dtype": "ordinal",
        "formula": "10 窗口内该股出现在涨停池的交易日数 (池覆盖子集)",
        "raw_dependencies": ("pool membership",),
        "source": "pool_cache_rebuild",
        "lookback": "10d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "COVERAGE_CAVEAT: 2026-05-06 前无池数据, 按池覆盖子集计",
        "expected_range_or_domain": "0..10",
        "notes": "池出现频率 10 日; 5/6 前无池数据 -> 计数为覆盖下界",
    },
    {
        "feature_name": "recent_pool_appearance_count_20d",
        "semantic_group": "BOARD_HISTORY",
        "dtype": "ordinal",
        "formula": "20 窗口内该股出现在涨停池的交易日数 (池覆盖子集)",
        "raw_dependencies": ("pool membership",),
        "source": "pool_cache_rebuild",
        "lookback": "20d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "COVERAGE_CAVEAT: 2026-05-06 前无池数据, 按池覆盖子集计",
        "expected_range_or_domain": "0..20",
        "notes": "池出现频率 20 日; 同 10d 覆盖下界",
    },
    # ---- D0_CROSS_SECTION (1) ----
    {
        "feature_name": "board_day_volume_rank",
        "semantic_group": "D0_CROSS_SECTION",
        "dtype": "continuous",
        "formula": "D0 当日涨停池成员按 daily volume 降序 rank(method='min') (1=最高)",
        "raw_dependencies": ("pool membership", "daily volume"),
        "source": "pool_cache_rebuild",
        "lookback": "D0",
        "available_as_of": "D0_CLOSE",
        "missing_semantics": "NA if D0 无池数据 (2026-04-30 事件, 池窗口外)",
        "expected_range_or_domain": "1..n_pool_members",
        "notes": "市场横截面 volume 排名 (market rank, 非模型 rank); 池口径合法",
    },
    # ---- D1_PRICE_ACTION (14) ----
    {
        "feature_name": "break_open_return",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "d1_open / d0_close - 1",
        "raw_dependencies": ("d1_open", "d0_close"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d0_close missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "D1 开盘相对 D0 收盘缺口 (模型表 PRIMARY)",
    },
    {
        "feature_name": "break_high_return",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "d1_high / d0_close - 1",
        "raw_dependencies": ("d1_high", "d0_close"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d0_close missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "D1 最高相对 D0 收盘 (模型表 PRIMARY)",
    },
    {
        "feature_name": "break_close_return",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "d1_close / d0_close - 1",
        "raw_dependencies": ("d1_close", "d0_close"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d0_close missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "D1 收盘相对 D0 收盘 (模型表 PRIMARY)",
    },
    {
        "feature_name": "break_touched_limit_up",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "binary",
        "formula": "1[d1_high >= round(d0_close*1.10, 2) - 0.011]",
        "raw_dependencies": ("d1_high", "d0_close"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d0_close missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "D1 盘中触板事件 (模型表 PRIMARY)",
    },
    {
        "feature_name": "break_opened_from_limit_up",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "binary",
        "formula": "1[d1_open >= round(d0_close*1.10, 2) - 0.011]",
        "raw_dependencies": ("d1_open", "d0_close"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d0_close missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "D1 开在涨停 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "break_upper_shadow_ratio",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "(d1_high - max(d1_open, d1_close)) / (d1_high - d1_low)",
        "raw_dependencies": ("d1_open", "d1_high", "d1_low", "d1_close"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d1_high == d1_low (公式退化)",
        "expected_range_or_domain": "[0,1]",
        "notes": "D1 上影线占比 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "break_lower_shadow_ratio",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "(min(d1_open, d1_close) - d1_low) / (d1_high - d1_low)",
        "raw_dependencies": ("d1_open", "d1_high", "d1_low", "d1_close"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d1_high == d1_low (公式退化)",
        "expected_range_or_domain": "[0,1]",
        "notes": "D1 下影线占比 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "d1_high_to_close_drawdown_raw",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "(d1_high - d1_close) / d1_high",
        "raw_dependencies": ("d1_high", "d1_close"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d1_high missing",
        "expected_range_or_domain": "[0,1]",
        "notes": "D1 日内冲高回落 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_low_to_close_recovery",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "(d1_close - d1_low) / d1_low",
        "raw_dependencies": ("d1_low", "d1_close"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d1_low missing",
        "expected_range_or_domain": "[0, +inf)",
        "notes": "D1 低点回收强度 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_open_to_close_return_raw",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "d1_close / d1_open - 1",
        "raw_dependencies": ("d1_open", "d1_close"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d1_open missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "D1 当日 open->close (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_close_location",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "(d1_close - d1_low) / (d1_high - d1_low)",
        "raw_dependencies": ("d1_low", "d1_high", "d1_close"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "STRUCTURAL_MISSING if d1_high==d1_low (603065 型一字板)",
        "expected_range_or_domain": "[0,1]",
        "notes": "D1 收盘日内位置; 官方公式分母 0 -> None (允许)",
    },
    {
        "feature_name": "d1_intraday_range",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "(d1_high - d1_low) / d0_close",
        "raw_dependencies": ("d1_high", "d1_low", "d0_close"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d0_close missing",
        "expected_range_or_domain": "[0, +inf)",
        "notes": "D1 振幅 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_afternoon_return",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "d1_close / 13:00 前最后 bar close - 1",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "午后段收益 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_last_hour_return",
        "semantic_group": "D1_PRICE_ACTION",
        "dtype": "continuous",
        "formula": "d1_close / 14:00 bar close - 1",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "尾盘 1 小时收益 (模型表 PRIMARY)",
    },
    # ---- D1_VOLUME_ACTIVITY (3) ----
    {
        "feature_name": "break_volume_ratio_vs_board_days",
        "semantic_group": "D1_VOLUME_ACTIVITY",
        "dtype": "continuous",
        "formula": "d1_volume / mean(board 日 volume)",
        "raw_dependencies": ("d1_volume", "board 日 daily volume"),
        "source": "daily_cache_rebuild",
        "lookback": "D1 + board days",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if board 日 volume 缺失",
        "expected_range_or_domain": "[0, +inf)",
        "notes": "D1 相对本轮连板量能放大 (ME01 canonical)",
    },
    {
        "feature_name": "d1_up_bar_volume_ratio",
        "semantic_group": "D1_VOLUME_ACTIVITY",
        "dtype": "continuous",
        "formula": "Σvolume(bar close>open) / Σvolume",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "阳线量占比 (模型表 PRIMARY)",
    },
    {
        "feature_name": "down_bar_volume_ratio",
        "semantic_group": "D1_VOLUME_ACTIVITY",
        "dtype": "continuous",
        "formula": "Σvolume(bar close<open) / Σvolume",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "阴线量占比 (canonical 名, 模型表 PRIMARY)",
    },
    # ---- D1_CHIP_DISTRIBUTION (4) ----
    {
        "feature_name": "volume_above_d1_close_ratio",
        "semantic_group": "D1_CHIP_DISTRIBUTION",
        "dtype": "continuous",
        "formula": "Σvolume(bar close >= d1_close) / Σvolume",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "收盘上方量占比 (ME02 canonical, 模型表 PRIMARY)",
    },
    {
        "feature_name": "amount_above_d1_close_ratio",
        "semantic_group": "D1_CHIP_DISTRIBUTION",
        "dtype": "continuous",
        "formula": "Σamount(bar close >= d1_close) / Σamount",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "ME02 金额替代表达 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "high_zone_volume_ratio",
        "semantic_group": "D1_CHIP_DISTRIBUTION",
        "dtype": "continuous",
        "formula": "Σvolume(bar close >= d1_low + 0.7*(d1_high-d1_low)) / Σvolume",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "高位区量占比 (ME03 canonical, 模型表 PRIMARY)",
    },
    {
        "feature_name": "high_zone_amount_ratio",
        "semantic_group": "D1_CHIP_DISTRIBUTION",
        "dtype": "continuous",
        "formula": "Σamount(bar close >= d1_low + 0.7*(d1_high-d1_low)) / Σamount",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "ME03 金额替代表达 (模型表 SENSITIVITY)",
    },
    # ---- D1_LATE_DAY_PRESSURE (2) ----
    {
        "feature_name": "late_day_sell_volume_ratio",
        "semantic_group": "D1_LATE_DAY_PRESSURE",
        "dtype": "continuous",
        "formula": "Σvolume(time>=14:00 and bar close<bar open) / Σvolume",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "尾盘下跌量占比 (ME04 canonical, 模型表 PRIMARY)",
    },
    {
        "feature_name": "late_day_sell_amount_ratio",
        "semantic_group": "D1_LATE_DAY_PRESSURE",
        "dtype": "continuous",
        "formula": "Σamount(time>=14:00 and bar close<bar open) / Σamount",
        "raw_dependencies": ("5min OHLCVA",),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if minute data incomplete",
        "expected_range_or_domain": "[0,1]",
        "notes": "ME04 金额替代表达 (模型表 SENSITIVITY)",
    },
    # ---- D1_VWAP_POSITION (1) ----
    {
        "feature_name": "d1_close_to_vwap_raw",
        "semantic_group": "D1_VWAP_POSITION",
        "dtype": "continuous",
        "formula": "d1_close / d1_vwap - 1",
        "raw_dependencies": ("d1_close", "d1_vwap"),
        "source": "v002",
        "lookback": "D1 intraday",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if d1_vwap missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "D1 收盘相对日内成本线 (ME05 canonical, 模型表 PRIMARY)",
    },
    # ---- D1_MA_POSITION (15) ----
    {
        "feature_name": "d1_close_to_ma5_raw",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "d1_close / ma5(D1) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma5 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "close/ma5 - 1 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_low_to_ma5_raw",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "d1_low / ma5(D1) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma5 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "low/ma5 - 1 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_high_to_ma5_raw",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "d1_high / ma5(D1) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma5 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "high/ma5 - 1 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "d1_close_to_ma10_raw",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "d1_close / ma10(D1) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA10 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma10 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "close/ma10 - 1 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_low_to_ma10_raw",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "d1_low / ma10(D1) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA10 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma10 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "low/ma10 - 1 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "d1_close_to_ma20",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "d1_close / ma20(D1) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA20 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma20 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "close/ma20 - 1 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_ma5_slope",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "ma5(D1) / ma5(D1 前一日) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma5 history missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "MA5 1 日变化率 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_ma10_slope",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "ma10(D1) / ma10(D1 前一日) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA10 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma10 history missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "MA10 1 日变化率 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_ma20_slope",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "continuous",
        "formula": "ma20(D1) / ma20(D1 前一日) - 1",
        "raw_dependencies": ("daily close",),
        "source": "daily_cache_rebuild",
        "lookback": "MA20 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma20 history missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "MA20 slope (DEFER 提升; D1_CLOSE 可得, 定义稳定)",
    },
    {
        "feature_name": "d1_close_above_ma5",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "binary",
        "formula": "1[d1_close >= ma5(D1)]",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma5 missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "ME06 canonical 二元项 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "d1_close_above_ma10",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "binary",
        "formula": "1[d1_close >= ma10(D1)]",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA10 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma10 missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "ME07 canonical 二元项 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "d1_true_reclaim_ma5",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "binary",
        "formula": "1[d1_low <= ma5(D1) and d1_close >= ma5(D1)]",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma5 missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "下探收回 MA5 (模型表 PRIMARY)",
    },
    {
        "feature_name": "d1_true_reclaim_ma10",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "binary",
        "formula": "1[d1_low <= ma10(D1) and d1_close >= ma10(D1)]",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA10 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if ma10 missing",
        "expected_range_or_domain": "{0,1}",
        "notes": "下探收回 MA10 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "consecutive_days_below_ma5",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "ordinal",
        "formula": "signal_date(含)向前 close < ma5 的连续天数",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA5 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "0 if no day below",
        "expected_range_or_domain": ">=0 (整数)",
        "notes": "ME06 替代 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "consecutive_days_below_ma10",
        "semantic_group": "D1_MA_POSITION",
        "dtype": "ordinal",
        "formula": "signal_date(含)向前 close < ma10 的连续天数",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "MA10 window",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "0 if no day below",
        "expected_range_or_domain": ">=0 (整数)",
        "notes": "ME07 替代 (模型表 SENSITIVITY)",
    },
    # ---- POOL_MEMBERSHIP (3) ----
    {
        "feature_name": "break_day_in_pool",
        "semantic_group": "POOL_MEMBERSHIP",
        "dtype": "binary",
        "formula": "1[code in pool(D1=break_date)]",
        "raw_dependencies": ("pool membership",),
        "source": "pool_cache_rebuild",
        "lookback": "D1",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if D1 无池数据",
        "expected_range_or_domain": "{0,1}",
        "notes": "D1 断板日是否出现在涨停池 (正常 False, 模型表 PRIMARY)",
    },
    {
        "feature_name": "last_board_day_in_pool",
        "semantic_group": "POOL_MEMBERSHIP",
        "dtype": "binary",
        "formula": "1[code in pool(D0=末板日)]",
        "raw_dependencies": ("pool membership",),
        "source": "pool_cache_rebuild",
        "lookback": "D0",
        "available_as_of": "D0_CLOSE",
        "missing_semantics": "NA if D0 无池数据 (2026-04-30 事件, 池窗口外)",
        "expected_range_or_domain": "{0,1}",
        "notes": "末板日是否在涨停池 (模型表 SENSITIVITY)",
    },
    {
        "feature_name": "pool_consecutive_count_last_board",
        "semantic_group": "POOL_MEMBERSHIP",
        "dtype": "ordinal",
        "formula": "pool(D0) consecutive_limit_up_count",
        "raw_dependencies": ("pool membership",),
        "source": "pool_cache_rebuild",
        "lookback": "D0",
        "available_as_of": "D0_CLOSE",
        "missing_semantics": "NA if D0 无池数据 (2026-04-30 事件, 池窗口外)",
        "expected_range_or_domain": ">=1 (正整数)",
        "notes": "池口径连板数 (模型表 SENSITIVITY, 与 board_streak 交叉核对)",
    },
    # ---- RECENT_7D_PATH (4) ----
    {
        "feature_name": "recent_7d_cumulative_return",
        "semantic_group": "RECENT_7D_PATH",
        "dtype": "continuous",
        "formula": "close(D1) / close(T-6) - 1",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "7d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if T-6 missing",
        "expected_range_or_domain": "(-1, +inf)",
        "notes": "最近 7 交易日累计涨幅 (模型表 PRIMARY)",
    },
    {
        "feature_name": "recent_7d_max_drawdown",
        "semantic_group": "RECENT_7D_PATH",
        "dtype": "continuous",
        "formula": "max_{t∈[T-5..D1]} max(0, (prior_peak_t - low_t)/prior_peak_t); prior_peak_t=max(high_s), s<t",
        "raw_dependencies": ("daily OHLC",),
        "source": "v002",
        "lookback": "7d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if insufficient history",
        "expected_range_or_domain": "[0,1]",
        "notes": "7 日窗口自峰值到后续低点最大回撤 (严格时序, 仅 <=D1 数据)",
    },
    {
        "feature_name": "recent_7d_close_position",
        "semantic_group": "RECENT_7D_PATH",
        "dtype": "continuous",
        "formula": "(close(D1) - 7d_low) / (7d_high - 7d_low)",
        "raw_dependencies": ("daily close",),
        "source": "v002",
        "lookback": "7d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if 7d_high == 7d_low (公式退化)",
        "expected_range_or_domain": "[0,1]",
        "notes": "D1 收盘在 7 日区间的位置 (模型表 PRIMARY)",
    },
    {
        "feature_name": "recent_7d_limit_up_count",
        "semantic_group": "RECENT_7D_PATH",
        "dtype": "ordinal",
        "formula": "7 窗口 (T-6..D1) 内日线涨停天数之和",
        "raw_dependencies": ("daily OHLCVA",),
        "source": "daily_cache_rebuild",
        "lookback": "7d",
        "available_as_of": "D1_CLOSE",
        "missing_semantics": "NA if insufficient daily history",
        "expected_range_or_domain": "0..7",
        "notes": "近 7 日涨停次数 (DEFER 提升; D1_CLOSE 可得, 定义稳定)",
    },
]

CONTRACT_FEATURE_NAMES: List[str] = [c["feature_name"] for c in FEATURE_CONTRACT]
CONTRACT_FEATURE_ORDER: Dict[str, int] = {
    c["feature_name"]: i + 1 for i, c in enumerate(FEATURE_CONTRACT)
}

# information_class 由 semantic_group 唯一映射注入 (单一来源, 避免双写漂移)
for _c in FEATURE_CONTRACT:
    _c["information_class"] = SEMANTIC_GROUP_INFORMATION_CLASS[_c["semantic_group"]]

# ---------------------------------------------------------------------------
# 身份 / 分组 / 审计 / 标签列 (非 FEATURE)
# ---------------------------------------------------------------------------
IDENTIFIER_COLUMNS = ("event_id", "code", "signal_date", "board_streak_before_break")
GROUP_KEY_COLUMN = "signal_date"
AUDIT_COLUMNS = (
    "source_window", "break_date", "minute_source", "daily_source",
    "d1_minute_complete", "minute_feature_count", "dev_feature_source_complete",
    "structural_missing_count", "unexpected_missing_count", "provenance",
)
LABEL_COLUMNS = ("target7_daily_d2open_d3high", "tail_loss_daily_5pct")
DEV_ELIGIBILITY_COLUMNS = (
    "dev_label_complete", "dev_feature_source_complete", "dev_training_eligible",
)

# ---------------------------------------------------------------------------
# 完整候选 inventory (132 个 repo-wide candidates + v002 附加审计字段)
# decision = FEATURE 或 EXCLUDE_*; 未来 Pairwise 只消费 FEATURE。
# ---------------------------------------------------------------------------
# 被排除候选的决策表 (name -> (decision, reason))。FEATURE 候选全部在
# FEATURE_CONTRACT 中, 这里列出所有非 FEATURE 候选及其明确理由。
EXCLUDED_CANDIDATE_DECISIONS: Dict[str, tuple[str, str]] = {
    # ---- 未来信息 (标签) ----
    "d2_open_daily": (LABEL_ONLY, "D2 开盘, 未来信息, 标签序列"),
    "d3_high_daily": (LABEL_ONLY, "D3 最高, 未来信息, 标签序列"),
    "d3_close_daily": (LABEL_ONLY, "D3 收盘, 未来信息, 标签序列"),
    "daily_d2open_to_d3high_return": (LABEL_ONLY, "标签收益, 未来信息"),
    "daily_d2open_to_d3close_return": (LABEL_ONLY, "标签收益, 未来信息"),
    "target7_daily_d2open_d3high": (LABEL_ONLY, "Target7 标签, 未来信息"),
    "tail_loss_daily_5pct": (LABEL_ONLY, "tail-loss 标签, 未来信息"),
    "label_d2_date": (AUDIT_ONLY, "标签审计元数据"),
    "label_d3_date": (AUDIT_ONLY, "标签审计元数据"),
    "daily_label_complete": (AUDIT_ONLY, "标签质量审计"),
    "daily_label_quality_ok": (AUDIT_ONLY, "标签质量审计"),
    # ---- 旧模型输出 (禁止) ----
    "recognition_score": (EXCLUDE_MODEL_OUTPUT, "旧模型/辨识度综合评分, 禁止作为 v004c 输入"),
    "is_v004a_scorable": (EXCLUDE_MODEL_OUTPUT, "v004a 模型可打分标记"),
    "v004a_probability": (EXCLUDE_MODEL_OUTPUT, "v004a 模型概率输出"),
    "v004a_rank": (EXCLUDE_MODEL_OUTPUT, "v004a 模型排名输出"),
    "is_v004a_top3": (EXCLUDE_MODEL_OUTPUT, "v004a Top3 标记"),
    "is_v004a_top10": (EXCLUDE_MODEL_OUTPUT, "v004a Top10 标记"),
    "is_v004a_top15": (EXCLUDE_MODEL_OUTPUT, "v004a Top15 标记"),
    "v004a_score_source": (AUDIT_ONLY, "v004a 分数来源审计"),
    "is_v002_scorable": (EXCLUDE_MODEL_OUTPUT, "v002 模型可打分标记"),
    "v002_rank": (EXCLUDE_MODEL_OUTPUT, "v002 模型排名输出"),
    "is_v002_top3": (EXCLUDE_MODEL_OUTPUT, "v002 Top3 标记"),
    "is_v002_top10": (EXCLUDE_MODEL_OUTPUT, "v002 Top10 标记"),
    "is_v002_top15": (EXCLUDE_MODEL_OUTPUT, "v002 Top15 标记"),
    "m0_probability": (EXCLUDE_MODEL_OUTPUT, "logistic walkforward 模型概率"),
    "m1_probability": (EXCLUDE_MODEL_OUTPUT, "logistic walkforward 模型概率"),
    "m2_probability": (EXCLUDE_MODEL_OUTPUT, "logistic walkforward 模型概率"),
    "m1_rank_daily": (EXCLUDE_MODEL_OUTPUT, "logistic walkforward 模型排名"),
    "m2_rank_daily": (EXCLUDE_MODEL_OUTPUT, "logistic walkforward 模型排名"),
    "r0_probability": (EXCLUDE_MODEL_OUTPUT, "repair-state 模型概率"),
    "r1_probability": (EXCLUDE_MODEL_OUTPUT, "repair-state 模型概率"),
    "r2_probability": (EXCLUDE_MODEL_OUTPUT, "repair-state 模型概率"),
    "r1_rank": (EXCLUDE_MODEL_OUTPUT, "repair-state 模型排名"),
    "r2_rank": (EXCLUDE_MODEL_OUTPUT, "repair-state 模型排名"),
    # ---- Repair-State 综合因子 (手动综合, 禁止) ----
    "OPEN": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 核心因子 (标准化合成分)"),
    "RESET": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "HIGHZONE": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "LATESELL": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "MOM7": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "DAMAGE7": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "REGIME": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "POS7": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "TREND": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 因子"),
    "DIVERGENCE": (EXCLUDE_MANUAL_COMPOSITE, "repair-state DIVERGENCE 综合"),
    "CLOSE_DAMAGE": (EXCLUDE_MANUAL_COMPOSITE, "repair-state CLOSE_DAMAGE 综合"),
    "SUPPLY": (EXCLUDE_MANUAL_COMPOSITE, "repair-state SUPPLY 综合"),
    "RECLAIM": (EXCLUDE_MANUAL_COMPOSITE, "repair-state RECLAIM 综合"),
    "DIVERGENCE_SQ": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 非线性变换"),
    "DIVERGENCE_X_RECLAIM": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 交互项"),
    "DIVERGENCE_X_DAMAGE": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 交互项"),
    "DIVERGENCE_X_SUPPLY": (EXCLUDE_MANUAL_COMPOSITE, "repair-state 交互项"),
    "TURNOVER_COST": (EXCLUDE_MANUAL_COMPOSITE, "repair-state TURNOVER_COST 综合"),
    # ---- 手动综合 / 预声明派生非线性 (禁止; 源码已在契约中) ----
    "overrepair": (EXCLUDE_MANUAL_COMPOSITE, "max(open_to_close-0.03,0): 同机制再表达, 非新机制"),
    "break_volume_abnormality": (EXCLUDE_MANUAL_COMPOSITE, "|log(volume_ratio)|: 同机制再表达"),
    "profit_pressure": (EXCLUDE_MANUAL_COMPOSITE, "三字段交互项: 非新机制, 留待人工评估"),
    "ma5_overheat_10": (EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM, "close_to_ma5>=0.10 二元化: 确定性阈值变换"),
    # ---- bucket / 预分桶 (确定性变换, 源码保留) ----
    "d1_close_to_ma5_bucket": (EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM, "预分桶表达, 源码 d1_close_to_ma5_raw 已入契约"),
    "d1_open_to_close_bucket": (EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM, "预分桶表达, 源码 d1_open_to_close_return_raw 已入契约"),
    # ---- vwap gap (确定性反向表达, 源码保留) ----
    "d1_vwap_to_close_gap": (EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM, "(vwap-close)/close 与 close_to_vwap 反向 (spearman -1), 只留一种"),
    # ---- 精确重复别名 (ED group) ----
    "break_open": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED01: 与 d1_open 精确重复"),
    "break_high": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED02: 与 d1_high 精确重复"),
    "break_low": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED03: 与 d1_low 精确重复"),
    "break_close": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED04: 与 d1_close 精确重复"),
    "break_intraday_range": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED05: 与 d1_intraday_range 精确重复"),
    "break_high_to_close_drawdown": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED06: 与 d1_high_to_close_drawdown_raw 精确重复"),
    "break_close_location": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED07: 与 d1_close_location 精确重复"),
    "break_volume": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED08: 与 d1_volume 精确重复"),
    "break_amount": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED09: 与 d1_amount 精确重复"),
    "d1_down_bar_volume_ratio": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED12: 与 down_bar_volume_ratio 精确重复"),
    "volume_above_break_close_ratio": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED13: 与 volume_above_d1_close_ratio 精确重复 (signal==break)"),
    "deprecated_d1_reclaimed_ma5_v01": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED10: 与 d1_close_above_ma5 精确重复, 命名废弃"),
    "deprecated_d1_reclaimed_ma10_v01": (EXCLUDE_EXACT_DUPLICATE_ALIAS, "ED11: 与 d1_close_above_ma10 精确重复, 命名废弃"),
    # ---- 数据源不可用 (May 窗口无法确定性重建) ----
    "board_day_amount_rank": (EXCLUDE_SOURCE_UNAVAILABLE,
                              "pool amount 在 2026-05-06..06-04 全 NaN (daily_limitup_derived), "
                              "daily amount 全 NaN (0/463902), 无法确定性重建 May 窗口 -> 排除"),
    "board_day_turnover_rank": (EXCLUDE_SOURCE_UNAVAILABLE,
                                "pool turnover_rate 在 2026-05-06..06-04 全 NaN, "
                                "daily turnover_rate 全 NaN, 无法确定性重建 May 窗口 -> 排除"),
    "break_amount_ratio_vs_board_days": (EXCLUDE_SOURCE_UNAVAILABLE,
                                         "分母 mean(board 日 amount) 在 May 窗口不可得 "
                                         "(daily amount 全 NaN, pool amount 5 月 NaN) -> 排除"),
    "break_turnover_ratio": (EXCLUDE_ALL_MISSING, "100% 缺失 (日线缓存无 turnover_rate), 字段不可用"),
    # ---- 提案 / 未实现 / 拒绝 ----
    "d1_to_d0_amount_ratio": (PENDING_EVIDENCE, "提案未实现; 且 volume 版已覆盖, 金额版数据源不可得"),
    "d1_to_d0_volume_ratio": (PENDING_EVIDENCE, "提案未实现; break_volume_ratio_vs_board_days 已覆盖"),
    "ma5_ma10_spread": (PENDING_EVIDENCE, "提案未实现; 可由 a/b 精确确定, 非新机制"),
    "board_stage_volume_price_decomposition": (PENDING_EVIDENCE, "提案未实现; 逐阶段量价重建复杂, 数据质量不稳定"),
    "profit_chip_ratio": (EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM, "1 - volume_above_d1_close_ratio: 精确互补"),
    "break_prev_close": (AUDIT_ONLY, "绝对价格锚点, DERIVE_ONLY; 派生后使用"),
    "d1_open": (AUDIT_ONLY, "绝对价格 DERIVE_ONLY"),
    "d1_high": (AUDIT_ONLY, "绝对价格 DERIVE_ONLY"),
    "d1_low": (AUDIT_ONLY, "绝对价格 DERIVE_ONLY"),
    "d1_close": (AUDIT_ONLY, "绝对价格 DERIVE_ONLY"),
    "d1_volume": (AUDIT_ONLY, "绝对量 DERIVE_ONLY"),
    "d1_amount": (AUDIT_ONLY, "绝对额 DERIVE_ONLY"),
    "d1_ma5": (AUDIT_ONLY, "绝对 MA5 DERIVE_ONLY"),
    "d1_ma10": (AUDIT_ONLY, "绝对 MA10 DERIVE_ONLY"),
    "d1_ma20": (AUDIT_ONLY, "绝对 MA20 DERIVE_ONLY"),
    "d1_vwap": (AUDIT_ONLY, "绝对 VWAP DERIVE_ONLY"),
    "board_streak_before_break": (IDENTIFIER, "身份列, 只经派生 board_streak_is_3 进入模型"),
    "signal_date": (IDENTIFIER, "GROUP_KEY, 非 FEATURE"),
    "event_id": (IDENTIFIER, "事件身份"),
    "code": (IDENTIFIER, "股票代码"),
    "source_window": (AUDIT_ONLY, "may/june 来源窗口, 非特征"),
    "break_date": (AUDIT_ONLY, "与 signal_date 逐行一致 (D1 identity)"),
}

# 数据质量 / 来源审计列 (AUDIT_ONLY, 不进 X)
AUDIT_ONLY_V002_COLUMNS = (
    "name", "minute_source", "minute_adjustment", "minute_interval",
    "daily_source", "d1_minute_complete", "minute_feature_count",
    "minute_feature_complete", "daily_feature_complete", "final_x_data_complete",
    "target_training_eligible", "structural_missing_count", "unexpected_missing_count",
    "provenance", "dev_label_complete", "dev_feature_source_complete", "dev_training_eligible",
)

# ---------------------------------------------------------------------------
# 审计函数
# ---------------------------------------------------------------------------
def assert_contract_integrity() -> None:
    """契约本身的不变量: 顺序唯一/连续, 字段合法, 无泄漏 token。"""
    names = [c["feature_name"] for c in FEATURE_CONTRACT]
    expected_order = {n: i + 1 for i, n in enumerate(names)}
    if CONTRACT_FEATURE_ORDER != expected_order:
        raise AssertionError("feature_order 必须从 1 连续唯一")
    if len(set(names)) != len(names):
        raise AssertionError("feature_name 必须唯一")
    if len(set(names)) != len(names):
        raise AssertionError("feature_name 必须唯一")
    for c in FEATURE_CONTRACT:
        sg = c["semantic_group"]
        if sg not in SEMANTIC_GROUPS:
            raise AssertionError(f"{c['feature_name']}: 非法 semantic_group {sg}")
        if c["information_class"] not in INFORMATION_CLASSES:
            raise AssertionError(f"{c['feature_name']}: 非法 information_class")
        if c["information_class"] != SEMANTIC_GROUP_INFORMATION_CLASS[sg]:
            raise AssertionError(
                f"{c['feature_name']}: information_class {c['information_class']} "
                f"与 group {sg} 不一致")
        if c["available_as_of"] not in ("D1_CLOSE", "D0_CLOSE"):
            raise AssertionError(f"{c['feature_name']}: available_as_of 非法")
        for tok in FEATURE_FORBIDDEN_TOKENS:
            if tok in c["feature_name"]:
                raise AssertionError(
                    f"{c['feature_name']}: 命中 FEATURE 禁止 token {tok}")
        if "rank" in c["feature_name"]:
            if c["semantic_group"] != RANK_REQUIRED_SEMANTIC_GROUP:
                raise AssertionError(
                    f"{c['feature_name']}: rank 名必须属 {RANK_REQUIRED_SEMANTIC_GROUP}")
            deps = " ".join(c["raw_dependencies"]).lower()
            if not any(k in deps for k in RANK_MARKET_DEPENDENCY_KEYWORDS):
                raise AssertionError(
                    f"{c['feature_name']}: rank 必须依赖市场横截面数据")


def leakage_scan(feature_names: List[str]) -> Dict[str, Any]:
    """泄漏审计: 对候选 FEATURE 名做 token + lineage 扫描。

    token 扫描 (FEATURE 禁止 token) 是硬失败;
    rank 名必须属市场横截面 (模型输出排名泄漏判定)。
    """
    future_hits: List[str] = []
    model_hits: List[str] = []
    label_hits: List[str] = []
    for name in feature_names:
        lower = name.lower()
        for tok in ("d2_", "d3_", "future", "outcome", "realized", "post_"):
            if tok in lower:
                future_hits.append(name)
                break
        for tok in ("prediction", "prob", "model", "v004a", "v004b", "v002",
                    "repair_state", "recognition", "policy", "rank_output"):
            if tok in lower:
                model_hits.append(name)
                break
        if "rank" in lower and name not in CONTRACT_FEATURE_NAMES:
            label_hits.append(name)
    return {
        "future_leakage": len(future_hits),
        "model_output_leakage": len(model_hits),
        "label_lineage_leakage": len(label_hits),
        "future_hits": future_hits,
        "model_hits": model_hits,
        "label_hits": label_hits,
    }

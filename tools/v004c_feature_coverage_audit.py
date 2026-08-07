"""v004c 特征覆盖核对 — 一次性研究审计脚本 (research utility only).

本脚本不属于 src/, 不是正式运行入口, 不训练模型, 不计算任何模型指标,
不使用任何 Target/标签 (仅读取 D1 时点特征与日线缓存, 全部数据 <= D1)。

用途:
1. 从 v004c 正式资产 (阶段1/2.1/2.2) 读取 59 个准入因子 + 197 列字典,
   逐字段归类到 coverage mechanism 并给出覆盖状态与机制候选角色;
2. 从 data/cache/daily 计算 4 个候选字段在 D1 时点的可用性与缺失率
   (3 个 NEW_REQUIRED + d1_ma20_slope 仅作 DEFER 覆盖验证;
   验证 "可以用现有日线缓存稳定计算" 这一准入条件);
3. 输出 reports/research/v004c_feature_coverage_v001.csv。

Target-blind 输入: 训练表通过 usecols 仅读取 event_id/code/break_date,
没有将任何标签、D2/D3 或旧模型字段读入内存。

候选字段严格公式 (全部只依赖 <= D1 的日线, 与冻结数据集同源同口径):
- recent_7d_cumulative_return = close(D1) / close(D1 前第 6 个交易日) - 1
- recent_7d_max_drawdown      = max_{t in [T-5..D1]} max(0, (prior_peak_t - low_t) / prior_peak_t),
                                prior_peak_t = max(high_s), s < t
                                (只使用严格较早交易日的 high, 不使用同日 high 计算当日 low 的回撤;
                                不使用分钟线; T-6 无 prior peak 不计算)
- recent_7d_close_position    = (close(D1) - 7d_low) / (7d_high - 7d_low);
                                7d_high == 7d_low 时置空 (不产生 inf, 同 d1_close_location 政策)
- d1_ma20_slope               = ma20(D1) / ma20(D1 前一日) - 1, ma20 = close 20 日滚动均值
                                (本阶段判定 DEFER_NOT_REQUIRED_FOR_V001; 脚本仅保留确定性计算能力)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DICT_DIR = REPO / "reports/research/v004c_factor_dictionary_v001_20260601_20260729"
STAGE1_DIR = REPO / "reports/research/v004c_d1_dataset_v001_20260601_20260729"
OUT_CSV = REPO / "reports/research/v004c_feature_coverage_v001.csv"
CACHE_DIR = REPO / "data/cache/daily"

# 训练表最小读取列: 本工具只读身份/日期列, 绝不读取 Target/tail/D2/D3/旧模型字段
COVERAGE_INPUT_COLUMNS = ["event_id", "code", "break_date"]

# coverage mechanism (建模视角, 本次新增) 与 coverage_status / recommended_role
COVERAGE_MECHANISM = {
    # BOARD_OR_EVENT_STATE
    "board_streak_before_break": "BOARD_OR_EVENT_STATE",
    "board_streak_is_3": "BOARD_OR_EVENT_STATE",
    "recent_limit_up_count_10d": "BOARD_OR_EVENT_STATE",
    "recent_limit_up_count_20d": "BOARD_OR_EVENT_STATE",
    "recent_pool_appearance_count_10d": "BOARD_OR_EVENT_STATE",
    "recent_pool_appearance_count_20d": "BOARD_OR_EVENT_STATE",
    "max_board_streak_20d": "BOARD_OR_EVENT_STATE",
    "break_day_in_pool": "BOARD_OR_EVENT_STATE",
    "last_board_day_in_pool": "BOARD_OR_EVENT_STATE",
    "pool_consecutive_count_last_board": "BOARD_OR_EVENT_STATE",
    # D0_CROSS_SECTION
    "board_day_amount_rank": "D0_CROSS_SECTION",
    "board_day_turnover_rank": "D0_CROSS_SECTION",
    "board_day_volume_rank": "D0_CROSS_SECTION",
    # D0_TO_D1_TRANSITION
    "break_prev_close": "D0_TO_D1_TRANSITION",
    "break_open_return": "D0_TO_D1_TRANSITION",
    "break_high_return": "D0_TO_D1_TRANSITION",
    "break_close_return": "D0_TO_D1_TRANSITION",
    "break_touched_limit_up": "D0_TO_D1_TRANSITION",
    "break_opened_from_limit_up": "D0_TO_D1_TRANSITION",
    # D1_PRICE_ACTION (绝对尺度 DERIVE_ONLY 锚点)
    "d1_open": "D1_PRICE_ACTION",
    "d1_high": "D1_PRICE_ACTION",
    "d1_low": "D1_PRICE_ACTION",
    "d1_close": "D1_PRICE_ACTION",
    "d1_open_to_close_return_raw": "D1_PRICE_ACTION",
    "d1_low_to_close_recovery": "D1_PRICE_ACTION",
    "d1_high_to_close_drawdown_raw": "D1_PRICE_ACTION",
    "d1_close_location": "D1_PRICE_ACTION",
    "d1_intraday_range": "D1_PRICE_ACTION",
    "d1_afternoon_return": "D1_PRICE_ACTION",
    "d1_last_hour_return": "D1_PRICE_ACTION",
    "break_upper_shadow_ratio": "D1_PRICE_ACTION",
    "break_lower_shadow_ratio": "D1_PRICE_ACTION",
    "d1_open_to_close_bucket": "D1_PRICE_ACTION",
    # D1_VOLUME_ACTIVITY
    "d1_volume": "D1_VOLUME_ACTIVITY",
    "d1_amount": "D1_VOLUME_ACTIVITY",
    "break_volume_ratio_vs_board_days": "D1_VOLUME_ACTIVITY",
    "break_amount_ratio_vs_board_days": "D1_VOLUME_ACTIVITY",
    "d1_up_bar_volume_ratio": "D1_VOLUME_ACTIVITY",
    "down_bar_volume_ratio": "D1_VOLUME_ACTIVITY",
    "d1_down_bar_volume_ratio": "D1_VOLUME_ACTIVITY",
    # D1_CHIP_DISTRIBUTION
    "volume_above_d1_close_ratio": "D1_CHIP_DISTRIBUTION",
    "amount_above_d1_close_ratio": "D1_CHIP_DISTRIBUTION",
    "volume_above_break_close_ratio": "D1_CHIP_DISTRIBUTION",
    "high_zone_volume_ratio": "D1_CHIP_DISTRIBUTION",
    "high_zone_amount_ratio": "D1_CHIP_DISTRIBUTION",
    # D1_LATE_DAY_PRESSURE
    "late_day_sell_volume_ratio": "D1_LATE_DAY_PRESSURE",
    "late_day_sell_amount_ratio": "D1_LATE_DAY_PRESSURE",
    # D1_VWAP_POSITION
    "d1_vwap": "D1_VWAP_POSITION",
    "d1_close_to_vwap_raw": "D1_VWAP_POSITION",
    "d1_vwap_to_close_gap": "D1_VWAP_POSITION",
    # D1_MA5_STATE
    "d1_ma5": "D1_MA5_STATE",
    "d1_close_to_ma5_raw": "D1_MA5_STATE",
    "d1_low_to_ma5_raw": "D1_MA5_STATE",
    "d1_high_to_ma5_raw": "D1_MA5_STATE",
    "d1_ma5_slope": "D1_MA5_STATE",
    "d1_close_above_ma5": "D1_MA5_STATE",
    "d1_true_reclaim_ma5": "D1_MA5_STATE",
    "consecutive_days_below_ma5": "D1_MA5_STATE",
    "d1_close_to_ma5_bucket": "D1_MA5_STATE",
    # D1_MA10_MA20_STATE
    "d1_ma10": "D1_MA10_MA20_STATE",
    "d1_ma20": "D1_MA10_MA20_STATE",
    "d1_close_to_ma10_raw": "D1_MA10_MA20_STATE",
    "d1_low_to_ma10_raw": "D1_MA10_MA20_STATE",
    "d1_close_to_ma20": "D1_MA10_MA20_STATE",
    "d1_ma10_slope": "D1_MA10_MA20_STATE",
    "d1_close_above_ma10": "D1_MA10_MA20_STATE",
    "d1_true_reclaim_ma10": "D1_MA10_MA20_STATE",
    "consecutive_days_below_ma10": "D1_MA10_MA20_STATE",
    # PREDECLARED_DERIVED (派生, 保持原机制归属)
    "ma5_overheat_10": "D1_MA5_STATE",
    "overrepair": "D1_PRICE_ACTION",
    "break_volume_abnormality": "D1_VOLUME_ACTIVITY",
    "profit_chip_ratio": "D1_CHIP_DISTRIBUTION",
    "profit_pressure": "D1_PRICE_ACTION",
    # PROPOSED (NEW_REQUIRED)
    "recent_7d_cumulative_return": "RECENT_7D_PATH",
    "recent_7d_max_drawdown": "RECENT_7D_PATH",
    "recent_7d_close_position": "RECENT_7D_PATH",
    "d1_ma20_slope": "D1_MA10_MA20_STATE",
    # NOT_NEEDED / DEFER 提案 (未实现字段)
    "recent_7d_limit_up_count": "RECENT_7D_PATH",
    "d1_to_d0_volume_ratio": "D0_TO_D1_TRANSITION",
    "d1_to_d0_amount_ratio": "D0_TO_D1_TRANSITION",
    "ma5_ma10_spread": "D1_MA10_MA20_STATE",
    "board_stage_volume_price_decomposition": "RECENT_7D_PATH",
    # FORBIDDEN 代表
    "target7_daily_d2open_d3high": "FORBIDDEN",
    "d2_open_daily": "FORBIDDEN",
    "d3_high_daily": "FORBIDDEN",
    "recognition_score": "FORBIDDEN",
    "v004a_probability": "FORBIDDEN",
    "v002_rank": "FORBIDDEN",
    # REDUNDANT (精确重复别名 / 废弃字段 / 全缺失字段)
    "break_open": "D1_PRICE_ACTION",
    "break_high": "D1_PRICE_ACTION",
    "break_low": "D1_PRICE_ACTION",
    "break_close": "D1_PRICE_ACTION",
    "break_intraday_range": "D1_PRICE_ACTION",
    "break_high_to_close_drawdown": "D1_PRICE_ACTION",
    "break_close_location": "D1_PRICE_ACTION",
    "break_volume": "D1_VOLUME_ACTIVITY",
    "break_amount": "D1_VOLUME_ACTIVITY",
    "break_turnover_ratio": "D1_VOLUME_ACTIVITY",
    "deprecated_d1_reclaimed_ma5_v01": "D1_MA5_STATE",
    "deprecated_d1_reclaimed_ma10_v01": "D1_MA10_MA20_STATE",
}

# coverage_status / recommended_role / reason (分析结论, 人工判定)
AUDIT = {
    # --- BOARD_OR_EVENT_STATE ---
    "board_streak_before_break": ("EXISTING_REUSE", "NO_MODEL_INPUT",
                                  "DERIVE_ONLY 绝对尺度; 只能经派生 (board_streak_is_3) 进入模型"),
    "break_prev_close": ("EXISTING_REUSE", "NO_MODEL_INPUT",
                         "DERIVE_ONLY 绝对尺度锚点 (D0 收盘), 只允许派生为收益/距离后使用"),
    "board_streak_is_3": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                          "二板/三板事件状态二元项, 预声明派生, D1_CLOSE 可得"),
    "recent_limit_up_count_10d": ("EXISTING_REUSE", "CONTEXT_ONLY",
                                  "属于已有事件背景信息, 可保留用于上下文和后续敏感性比较, 但不是 V001 M2 的默认主模型输入; REUSE_AS_CONTEXT"),
    "recent_limit_up_count_20d": ("EXISTING_REUSE", "CONTEXT_ONLY",
                                  "20 日事件统计超出核心 7 日窗口; 不得自动进入主模型, REUSE_AS_CONTEXT"),
    "recent_pool_appearance_count_10d": ("EXISTING_REUSE", "CONTEXT_ONLY",
                                         "池出现频率 10 日; 覆盖偏倚 (2026-05-06 前无池数据); CONTEXT_ONLY"),
    "recent_pool_appearance_count_20d": ("EXISTING_REUSE", "CONTEXT_ONLY",
                                         "池出现频率 20 日; 同上, CONTEXT_ONLY"),
    "max_board_streak_20d": ("EXISTING_REUSE", "CONTEXT_ONLY",
                             "20 日最大连板 (含本轮), 与 board_streak_before_break 强相关; CONTEXT_ONLY"),
    "break_day_in_pool": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                          "D1 断板日池状态二元项 (正常 95% 为 False, 罕见反例含事件信息)"),
    "last_board_day_in_pool": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                               "D0 池状态, 97% 恒 True, 信息量极低"),
    "pool_consecutive_count_last_board": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                          "池口径连板数与日线相邻口径交叉核对字段, 与 board_streak 近重复"),
    # --- D0_CROSS_SECTION ---
    "board_day_amount_rank": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                              "D0 横截面排名 (SENSITIVITY 层); 缺失 10.2% (池覆盖); 仅敏感性"),
    "board_day_turnover_rank": ("EXISTING_REUSE", "SENSITIVITY_ONLY", "同上, turnover 口径"),
    "board_day_volume_rank": ("EXISTING_REUSE", "SENSITIVITY_ONLY", "同上, volume 口径 (缺失 2.7%)"),
    # --- D0_TO_D1_TRANSITION ---
    "break_open_return": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                          "D1 open / D0 close - 1 (prev_close=末板日收盘=D0 close): 真 D0→D1 开盘缺口"),
    "break_high_return": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                          "D1 high / D0 close - 1: D0→D1 最高延伸"),
    "break_close_return": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                           "D1 close / D0 close - 1: 核心 D0→D1 收盘过渡 (名称含 break 但确为 D0→D1, 已核对公式)"),
    "break_touched_limit_up": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                               "D1 盘中触板事件 (15% 正例), 与价格过渡互补"),
    "break_opened_from_limit_up": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                   "D1 开在涨停 (2.4% 正例), 少数值二元项"),
    # --- D1_PRICE_ACTION ---
    "d1_open": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对价格, 只允许派生后使用"),
    "d1_high": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对价格, 只允许派生后使用"),
    "d1_low": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对价格, 只允许派生后使用"),
    "d1_close": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对价格, 只允许派生后使用"),
    "d1_open_to_close_return_raw": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                                     "D1 当日 open→close (非 D0→D1); D1 日内价格结构核心"),
    "d1_low_to_close_recovery": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                                 "(close - low) / low: 日内低点回收强度"),
    "d1_high_to_close_drawdown_raw": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                                      "(high - close) / high: 日内冲高回落"),
    "d1_close_location": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                          "(close-low)/(high-low); 与 low_to_close_recovery/drawdown 同机制不同表达"),
    "d1_intraday_range": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                          "(high-low)/prev_close: D1 振幅, 波动性表达"),
    "d1_afternoon_return": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                            "午后段收益 (13:00 前最后 bar→收盘)"),
    "d1_last_hour_return": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                            "尾盘 1 小时收益 (14:00 bar→收盘); 与 afternoon_return 近重复, 紧凑模型只留一种"),
    "break_upper_shadow_ratio": ("EXISTING_REUSE", "SENSITIVITY_ONLY", "上影线占比 (SENSITIVITY)"),
    "break_lower_shadow_ratio": ("EXISTING_REUSE", "SENSITIVITY_ONLY", "下影线占比 (SENSITIVITY)"),
    "d1_open_to_close_bucket": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                "预分桶研究表达, 只做 bucket 敏感性"),
    # --- D1_VOLUME_ACTIVITY ---
    "d1_volume": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对量, 只允许派生后使用"),
    "d1_amount": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对额, 只允许派生后使用"),
    "break_volume_ratio_vs_board_days": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                                         "D1 vol / mean(板日 vol) (ME01 canonical): D1 相对本轮连板的量能放大"),
    "break_amount_ratio_vs_board_days": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                         "ME01 金额替代表达 (MUTUALLY_EXCLUSIVE_EXPRESSION), 不得与 volume 版同入模型"),
    "d1_up_bar_volume_ratio": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                               "阳线量占比; 与 down_bar 近互补 (up+down+doji=1), 紧凑模型只留一种"),
    "down_bar_volume_ratio": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                              "阴线量占比 (canonical 名), 与 up_bar 近互补"),
    "d1_down_bar_volume_ratio": ("REDUNDANT", "NO_MODEL_INPUT",
                                 "ED12: 与 down_bar_volume_ratio 精确重复 (333/333 相等)"),
    # --- D1_CHIP_DISTRIBUTION ---
    "volume_above_d1_close_ratio": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                                    "收盘上方量占比 (ME02 canonical): D1 筹码结构核心"),
    "amount_above_d1_close_ratio": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                    "ME02 金额替代表达 (MUTUALLY_EXCLUSIVE_EXPRESSION), spearman 0.9999"),
    "volume_above_break_close_ratio": ("REDUNDANT", "NO_MODEL_INPUT",
                                       "ED13: 与 volume_above_d1_close_ratio 精确重复 (break_close==d1_close)"),
    "high_zone_volume_ratio": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                               "高位区量占比 (ME03 canonical), 28% 为 0 (未触高位区)"),
    "high_zone_amount_ratio": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                               "ME03 金额替代表达 (MUTUALLY_EXCLUSIVE_EXPRESSION), spearman 0.9998"),
    # --- D1_LATE_DAY_PRESSURE ---
    "late_day_sell_volume_ratio": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                                   "尾盘 (>=14:00) 下跌量占比 (ME04 canonical): 尾盘抛压"),
    "late_day_sell_amount_ratio": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                   "ME04 金额替代表达 (MUTUALLY_EXCLUSIVE_EXPRESSION), spearman 0.9989"),
    # --- D1_VWAP_POSITION ---
    "d1_vwap": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对 VWAP, 只允许派生后使用"),
    "d1_close_to_vwap_raw": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                             "close / vwap - 1 (ME05 canonical): D1 收盘相对日内成本线"),
    "d1_vwap_to_close_gap": ("REDUNDANT", "NO_MODEL_INPUT",
                             "NP05: (vwap-close)/close 与 close_to_vwap 反向表达 (spearman -1.0), 模型只留一种"),
    # --- D1_MA5_STATE ---
    "d1_ma5": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对 MA5, 只允许派生后使用"),
    "d1_close_to_ma5_raw": ("EXISTING_REUSE", "M1_BASELINE_CANDIDATE",
                            "close/ma5 - 1: MA5 技术状态核心锚点 (长历史计算, D1_CLOSE 可得)"),
    "d1_low_to_ma5_raw": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                          "low/ma5 - 1: 日内低点对 MA5 的支撑检验, 与 close 版同一 MA 机制"),
    "d1_high_to_ma5_raw": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                           "high/ma5 - 1: 与 close_to_ma5 同机制, 紧凑模型优先 close/low 版"),
    "d1_ma5_slope": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                     "ma5 1 日变化率: MA5 趋势方向"),
    "d1_close_above_ma5": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                           "ME06 canonical 二元项, 99% 恒 True (少数值); 与 consecutive_below_ma5 互斥"),
    "d1_true_reclaim_ma5": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                            "low<=ma5 且 close>=ma5: 下探收回 MA5 事件 (6% 正例)"),
    "consecutive_days_below_ma5": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                   "ME06 替代 (近单调, NP04); 紧凑模型与 close_above_ma5 二选一"),
    "d1_close_to_ma5_bucket": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                               "预分桶研究表达, 只做 bucket 敏感性"),
    # --- D1_MA10_MA20_STATE ---
    "d1_ma10": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对 MA10, 只允许派生后使用"),
    "d1_ma20": ("EXISTING_REUSE", "NO_MODEL_INPUT", "DERIVE_ONLY 绝对 MA20, 只允许派生后使用"),
    "d1_close_to_ma10_raw": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                             "close/ma10 - 1: MA10 水平; 注意 Stage2.2 无标签分布漂移标记 (六月→七月)"),
    "d1_low_to_ma10_raw": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                           "low/ma10 - 1: 与 close_to_ma10 同机制; 亦被漂移标记"),
    "d1_close_to_ma20": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                         "close/ma20 - 1: MA20 水平 (MA20 允许作为技术状态进入模型); 有漂移标记"),
    "d1_ma10_slope": ("EXISTING_REUSE", "M2_PRIMARY_CANDIDATE",
                      "ma10 1 日变化率; 有漂移标记"),
    "d1_close_above_ma10": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                            "ME07 canonical 二元项, 98.5% 恒 True; 与 consecutive_below_ma10 互斥"),
    "d1_true_reclaim_ma10": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                             "下探收回 MA10 (2.2% 正例), 少数值二元项"),
    "consecutive_days_below_ma10": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                    "ME07 替代 (NP03); 紧凑模型与 close_above_ma10 二选一"),
    # --- PREDECLARED_DERIVED ---
    "ma5_overheat_10": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                        "close_to_ma5 >= 0.10 二元化; 同一机制的再表达, 非新机制"),
    "overrepair": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                   "max(open_to_close - 0.03, 0): 同一机制的再表达"),
    "break_volume_abnormality": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                                 "|log(volume_ratio)|: 同一机制的再表达"),
    "profit_chip_ratio": ("REDUNDANT", "NO_MODEL_INPUT",
                          "1 - volume_above_d1_close_ratio: ME02 canonical 的精确互补"),
    "profit_pressure": ("EXISTING_REUSE", "SENSITIVITY_ONLY",
                        "三字段交互项: 非新机制, 交互项留待模型表阶段人工评估"),
    # --- PROPOSED NEW_REQUIRED ---
    "recent_7d_cumulative_return": ("NEW_REQUIRED", "M2_PRIMARY_CANDIDATE",
                                    "最近 7 交易日累计涨幅 (D1 close / T-6 close - 1); 现有字段无等价表达"),
    "recent_7d_max_drawdown": ("NEW_REQUIRED", "M2_PRIMARY_CANDIDATE",
                               "7 日窗口内自峰值到后续低点的最大回撤; 严格时序公式, 仅用 <=D1 数据"),
    "recent_7d_close_position": ("NEW_REQUIRED", "M2_PRIMARY_CANDIDATE",
                                 "(close - 7d_low)/(7d_high - 7d_low); 7d_high==7d_low 置空"),
    "d1_ma20_slope": ("DEFER_NOT_REQUIRED_FOR_V001", "SENSITIVITY_ONLY",
                      "MA20 slope 概念上合理, 但 MA10/MA20 技术背景已有充分表达 (close_to_ma10/low_to_ma10/close_to_ma20/ma10_slope 等); 为控制 V001 特征数量和避免增加相近趋势表达, 本阶段不把它作为必补字段, 后续可作为敏感性候选重新评估; 脚本保留确定性计算能力"),
    # --- NOT_NEEDED / DEFER ---
    "recent_7d_limit_up_count": ("DEFER_NOT_REQUIRED_FOR_V001", "CONTEXT_ONLY",
                                 "最近 7 日涨停次数可能包含当前连板之前的独立涨停事件, 并不严格等价于 board_streak_before_break; V001 已使用 board/event state + 7 日价格路径, 为控制模型复杂度暂不新增该事件计数字段"),
    "d1_to_d0_volume_ratio": ("NOT_NEEDED", "NO_MODEL_INPUT",
                              "break_volume_ratio_vs_board_days 已覆盖 D1 相对本轮连板量能; 纯 D1/D0 版更噪声"),
    "d1_to_d0_amount_ratio": ("NOT_NEEDED", "NO_MODEL_INPUT",
                              "volume 版的金额替代表达, 按原则不与其同入模型"),
    "ma5_ma10_spread": ("NOT_NEEDED", "NO_MODEL_INPUT",
                        "可由已有 MA5/MA10 距离字段精确确定: a=d1_close_to_ma5_raw, b=d1_close_to_ma10_raw, MA5/MA10 - 1 = (1+b)/(1+a) - 1; 未形成新的独立机制, 因此 V001 不新增字段"),
    "board_stage_volume_price_decomposition": ("DEFER_NOT_REQUIRED_FOR_V001", "NO_MODEL_INPUT",
                                               "第一/二/三板逐阶段量价重建复杂, 数据质量不稳定; V001 不需要"),
    # --- FORBIDDEN ---
    "target7_daily_d2open_d3high": ("FORBIDDEN", "NO_MODEL_INPUT", "D3 标签, 未来信息"),
    "d2_open_daily": ("FORBIDDEN", "NO_MODEL_INPUT", "D2 开盘, 未来信息"),
    "d3_high_daily": ("FORBIDDEN", "NO_MODEL_INPUT", "D3 最高, 未来信息"),
    "recognition_score": ("FORBIDDEN", "NO_MODEL_INPUT", "旧模型/辨识度输出, 禁止作为 v004c 输入"),
    "v004a_probability": ("FORBIDDEN", "NO_MODEL_INPUT", "旧模型输出, 禁止作为 v004c 输入"),
    "v002_rank": ("FORBIDDEN", "NO_MODEL_INPUT", "旧模型输出, 禁止作为 v004c 输入"),
    # --- REDUNDANT (精确重复别名 / 废弃) ---
    "break_open": ("REDUNDANT", "NO_MODEL_INPUT", "ED01: 与 d1_open 精确重复 (333/333 相等)"),
    "break_high": ("REDUNDANT", "NO_MODEL_INPUT", "ED02: 与 d1_high 精确重复"),
    "break_low": ("REDUNDANT", "NO_MODEL_INPUT", "ED03: 与 d1_low 精确重复"),
    "break_close": ("REDUNDANT", "NO_MODEL_INPUT", "ED04: 与 d1_close 精确重复"),
    "break_intraday_range": ("REDUNDANT", "NO_MODEL_INPUT", "ED05: 与 d1_intraday_range 精确重复"),
    "break_high_to_close_drawdown": ("REDUNDANT", "NO_MODEL_INPUT", "ED06: 与 d1_high_to_close_drawdown_raw 精确重复"),
    "break_close_location": ("REDUNDANT", "NO_MODEL_INPUT", "ED07: 与 d1_close_location 精确重复"),
    "break_volume": ("REDUNDANT", "NO_MODEL_INPUT", "ED08: 与 d1_volume 精确重复"),
    "break_amount": ("REDUNDANT", "NO_MODEL_INPUT", "ED09: 与 d1_amount 精确重复"),
    "break_turnover_ratio": ("NOT_NEEDED", "NO_MODEL_INPUT", "100% 缺失 (日线缓存无 turnover_rate), 字段不可用"),
    "deprecated_d1_reclaimed_ma5_v01": ("REDUNDANT", "NO_MODEL_INPUT",
                                        "ED10: 与 d1_close_above_ma5 精确重复, 命名废弃"),
    "deprecated_d1_reclaimed_ma10_v01": ("REDUNDANT", "NO_MODEL_INPUT",
                                         "ED11: 与 d1_close_above_ma10 精确重复, 命名废弃"),
}

FORMULA_OVERRIDE = {
    "board_streak_before_break": "断板前连续涨停天数 ∈ {2,3} (日线相邻交易日判定)",
    "recent_limit_up_count_10d": "信号日(含)往前 10 个交易日中日线涨停天数",
    "recent_limit_up_count_20d": "信号日(含)往前 20 个交易日中日线涨停天数",
    "recent_pool_appearance_count_10d": "同一 10 日窗口内该股出现在涨停池的天数",
    "recent_pool_appearance_count_20d": "同一 20 日窗口内该股出现在涨停池的天数",
    "max_board_streak_20d": "20 日窗口内最长连续涨停天数",
    "break_prev_close": "末板日收盘 = D0 close (断板前最后涨停日收盘; 绝对价格 DERIVE_ONLY)",
    "break_open_return": "break_open / break_prev_close - 1 (prev_close=末板日收盘=D0 close)",
    "break_high_return": "break_high / break_prev_close - 1",
    "break_close_return": "break_close / break_prev_close - 1",
    "break_intraday_range": "(high - low) / prev_close",
    "break_high_to_close_drawdown": "(high - close) / high",
    "break_close_location": "(close - low) / (high - low)",
    "break_upper_shadow_ratio": "(high - max(open, close)) / (high - low)",
    "break_lower_shadow_ratio": "(min(open, close) - low) / (high - low)",
    "break_volume_ratio_vs_board_days": "break_volume / mean(断板前 streak 个板日 volume)",
    "break_amount_ratio_vs_board_days": "break_amount / mean(板日 amount)",
    "break_touched_limit_up": "break_high >= 涨停价 - 0.011 (涨停价=前收×1.10 四舍五入到分)",
    "break_opened_from_limit_up": "break_open >= 涨停价 - 0.011",
    "d1_close_to_ma5_raw": "d1_close / d1_ma5 - 1",
    "d1_low_to_ma5_raw": "d1_low / d1_ma5 - 1",
    "d1_high_to_ma5_raw": "d1_high / d1_ma5 - 1",
    "d1_close_to_ma10_raw": "d1_close / d1_ma10 - 1",
    "d1_low_to_ma10_raw": "d1_low / d1_ma10 - 1",
    "d1_close_to_ma20": "d1_close / d1_ma20 - 1",
    "d1_ma5_slope": "ma5(D1) / ma5(D1 前一日) - 1",
    "d1_ma10_slope": "ma10(D1) / ma10(D1 前一日) - 1",
    "d1_close_above_ma5": "d1_close >= d1_ma5",
    "d1_close_above_ma10": "d1_close >= d1_ma10",
    "d1_true_reclaim_ma5": "d1_low <= d1_ma5 AND d1_close >= d1_ma5",
    "d1_true_reclaim_ma10": "d1_low <= d1_ma10 AND d1_close >= d1_ma10",
    "consecutive_days_below_ma5": "从 signal_date(含)向前数 close < ma5 的连续天数",
    "consecutive_days_below_ma10": "从 signal_date(含)向前数 close < ma10 的连续天数",
    "d1_open_to_close_return_raw": "d1_close / d1_open - 1 (5min 首/末 bar)",
    "d1_low_to_close_recovery": "(d1_close - d1_low) / d1_low (5min 日内低点)",
    "d1_high_to_close_drawdown_raw": "(d1_high - d1_close) / d1_high",
    "d1_close_location": "(d1_close - d1_low) / (d1_high - d1_low)",
    "d1_vwap": "5min 累计额/累计量 (手数自适应) 末值",
    "d1_close_to_vwap_raw": "d1_close / d1_vwap - 1",
    "d1_vwap_to_close_gap": "(d1_vwap - d1_close) / d1_close (正=卖压)",
    "d1_intraday_range": "(d1_high - d1_low) / prev_close",
    "d1_afternoon_return": "d1_close / 13:00 前最后一根 bar 的 close - 1",
    "d1_last_hour_return": "d1_close / 14:00 bar 的 close - 1",
    "d1_up_bar_volume_ratio": "Σvolume(close>open) / Σvolume",
    "down_bar_volume_ratio": "Σvolume(close<open) / Σvolume",
    "volume_above_d1_close_ratio": "Σvolume(bar close >= d1_close) / Σvolume",
    "high_zone_volume_ratio": "Σvolume(typical >= d1_low+0.7×(d1_high−d1_low)) / Σvolume",
    "late_day_sell_volume_ratio": "Σvolume(time>=14:00 且 close<open) / Σvolume",
    "break_day_in_pool": "D1 断板日是否出现在涨停池 (正常为 False)",
    "board_day_amount_rank": "末板日该股 amount 在当日涨停池成员中的降序名次 (1=最高)",
    "board_day_turnover_rank": "末板日该股 turnover_rate 在当日涨停池成员中的降序名次",
    "board_day_volume_rank": "末板日该股 volume 在当日涨停池成员中的降序名次",
    "last_board_day_in_pool": "末板日是否在涨停池",
    "pool_consecutive_count_last_board": "池口径 consecutive_limit_up_count (交叉核对用)",
    "recent_7d_cumulative_return": "close(D1) / close(T-6) - 1, T-6 = D1 前第 6 个交易日 (窗口 7 行)",
    "recent_7d_max_drawdown": "max_{t∈[T-5..D1]} max(0, (prior_peak_t - low_t) / prior_peak_t); prior_peak_t = max(high_s), s < t (严格较早交易日, 不使用同日 high/low 顺序)",
    "recent_7d_close_position": "(close(D1) - 7d_low) / (7d_high - 7d_low); 7d_high==7d_low 置空",
    "d1_ma20_slope": "ma20(D1) / ma20(D1 前一日) - 1; ma20 = close 20 日滚动均值 (min_periods=20)",
    "recent_7d_limit_up_count": "(提案, 未实现) 7 日窗口内涨停天数",
    "d1_to_d0_volume_ratio": "(提案, 未实现) d1_volume / d0_volume",
    "d1_to_d0_amount_ratio": "(提案, 未实现) d1_amount / d0_amount",
    "ma5_ma10_spread": "(提案, 未实现) MA5/MA10 - 1 = (1+b)/(1+a) - 1, a = d1_close_to_ma5_raw, b = d1_close_to_ma10_raw",
    "board_stage_volume_price_decomposition": "(提案, 未实现) 第一/二/三板与 D1 逐阶段量价结构",
}

PROPOSED_FIELDS = ["recent_7d_cumulative_return", "recent_7d_max_drawdown",
                   "recent_7d_close_position", "d1_ma20_slope"]


def strict_7d_max_drawdown(highs: np.ndarray, lows: np.ndarray) -> float | None:
    """严格日级最大回撤: prior_peak_t = max(high_s), s < t。

    只允许使用严格早于 t 的历史交易日 high 作为 prior peak;
    当前日 high 不用于当前日 low 的回撤 (日线 OHLC 无法确定同日先后顺序);
    当前日 high 只成为后续交易日的 prior peak。
    所有后续 low 高于 prior peak 时回撤为 0.0; prior peak 必须有限且 > 0。
    """
    if len(highs) < 2:
        return None
    prior_peak = highs[0]
    drawdowns: list[float] = []
    for t in range(1, len(highs)):
        low_t = lows[t]
        if np.isfinite(prior_peak) and prior_peak > 0 and np.isfinite(low_t):
            drawdowns.append(max(0.0, (prior_peak - low_t) / prior_peak))
        high_t = highs[t]
        if np.isfinite(high_t):
            prior_peak = max(prior_peak, high_t)
    return max(drawdowns) if drawdowns else None


def _self_check_max_drawdown() -> None:
    """严格日级最大回撤的确定性 self-check (Case A/B/C)。"""
    # Case A: 同日 high/low 不允许虚构顺序.
    # day1 high=10 low=9; day2 high=15 low=8 -> day2 low=8 只相对 day1 prior peak=10 -> 20%,
    # 不得使用 day2 high=15 得到 46.67%.
    a = strict_7d_max_drawdown(np.array([10.0, 15.0]), np.array([9.0, 8.0]))
    assert a is not None and abs(a - 0.20) < 1e-12, f"Case A failed: {a}"
    # Case B: 先高后未来低. day1 high=10; day2 high=15; day3 low=9 ->
    # day3 相对 day2 已形成的 prior peak=15 -> 40%.
    b = strict_7d_max_drawdown(np.array([10.0, 15.0, 15.0]), np.array([np.nan, np.nan, 9.0]))
    assert b is not None and abs(b - 0.40) < 1e-12, f"Case B failed: {b}"
    # Case C: 一路上涨, 所有后续 low 高于 prior peak -> max_drawdown = 0.
    c = strict_7d_max_drawdown(np.array([10.0, 11.0, 12.0]), np.array([9.5, 10.5, 11.5]))
    assert c is not None and abs(c - 0.0) < 1e-12, f"Case C failed: {c}"
    print("strict max drawdown self-check: Case A/B/C OK")


def compute_proposed_features(training: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """从 data/cache/daily 计算 4 个候选字段 (只用 <= D1 数据, 不用任何标签)。"""
    cache = {p.stem.split("_")[0]: pd.read_pickle(p) for p in cache_dir.glob("*_daily.pkl")}
    rows = []
    for _, row in training.iterrows():
        code, d1_date = str(row["code"]), str(row["break_date"])
        frame = cache.get(code)
        if frame is None:
            rows.append({"event_id": row["event_id"], "recent_7d_cumulative_return": np.nan,
                         "recent_7d_max_drawdown": np.nan, "recent_7d_close_position": np.nan,
                         "d1_ma20_slope": np.nan, "reason": "cache_missing"})
            continue
        frame = frame.copy().sort_values("date").reset_index(drop=True)
        for col in ("open", "high", "low", "close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["ma20"] = frame["close"].rolling(20, min_periods=20).mean()
        hits = frame.index[frame["date"].astype(str) == d1_date]
        reason = ""
        if len(hits) == 0:
            cum = dd = pos = slope = np.nan
            reason = "d1_date_not_in_cache"
        else:
            i = int(hits[0])
            cum = np.nan
            dd = np.nan
            pos = np.nan
            slope = np.nan
            if i >= 6 and not frame.loc[i, "close"] is None:
                base = frame.loc[i - 6, "close"]
                if pd.notna(base) and base > 0 and pd.notna(frame.loc[i, "close"]):
                    cum = frame.loc[i, "close"] / base - 1
            if i >= 6:
                win = frame.loc[i - 6:i]
                dd = strict_7d_max_drawdown(
                    pd.to_numeric(win["high"], errors="coerce").to_numpy(),
                    pd.to_numeric(win["low"], errors="coerce").to_numpy(),
                )
                if dd is None:
                    dd = np.nan
                hi7 = win["high"].max()
                lo7 = win["low"].min()
                if pd.notna(hi7) and pd.notna(lo7) and hi7 > lo7 and pd.notna(frame.loc[i, "close"]):
                    pos = (frame.loc[i, "close"] - lo7) / (hi7 - lo7)
            if i >= 1 and pd.notna(frame.loc[i - 1, "ma20"]) and frame.loc[i - 1, "ma20"] > 0:
                slope = frame.loc[i, "ma20"] / frame.loc[i - 1, "ma20"] - 1
            else:
                reason = reason or "ma20_history_insufficient"
        rows.append({"event_id": row["event_id"], "recent_7d_cumulative_return": cum,
                     "recent_7d_max_drawdown": dd, "recent_7d_close_position": pos,
                     "d1_ma20_slope": slope, "reason": reason})
    return pd.DataFrame(rows)


def main() -> None:
    _self_check_max_drawdown()
    dict_df = pd.read_csv(DICT_DIR / "v004c_factor_dictionary_v001.csv")
    primary = pd.read_csv(DICT_DIR / "v004c_feature_allowlist_primary_v001.csv")
    sensitivity = pd.read_csv(DICT_DIR / "v004c_feature_allowlist_sensitivity_v001.csv")
    # Target-blind 输入: 训练表只读最小身份/日期列, 不读入 Target/tail/D2/D3/旧模型字段
    training = pd.read_csv(
        STAGE1_DIR / "v004c_training_d1_v001.csv",
        dtype={"code": str},
        usecols=COVERAGE_INPUT_COLUMNS,
    )
    assert set(training.columns) == set(COVERAGE_INPUT_COLUMNS), (
        f"训练表读取列异常: {sorted(training.columns)}")

    # 1) 现有准入因子: 字典中 stage1_allowed=True 的 51 个源字段
    allowed = dict_df[dict_df["stage1_allowed_for_feature_analysis"] == True].copy()  # noqa: E712
    # 2) 预声明派生 6 个 (allowlist primary 中 source_or_derived=derived)
    derived_spec = pd.read_csv(DICT_DIR / "v004c_predeclared_derived_factor_spec_v001.csv")
    # 3) 额外审计行 (REDUNDANT/FORBIDDEN/NOT_NEEDED/DEFER + 4 提案)
    extra_names = [n for n in AUDIT if n not in set(allowed["source_column"]) | set(derived_spec["feature_name"])]

    # 计算 4 个提案字段
    proposed = compute_proposed_features(training, CACHE_DIR)

    rows = []
    for _, r in allowed.iterrows():
        name = r["source_column"]
        status, role, reason = AUDIT[name]
        rows.append({
            "feature_name": name,
            "source": f"v004c-d1-dataset-0.1 (v004c-features-0.3)",
            "available_as_of": r["available_as_of"],
            "existing_or_proposed": "EXISTING",
            "coverage_mechanism": COVERAGE_MECHANISM[name],
            "existing_stage2_1_mechanism": r["mechanism_group"],
            "formula_or_definition": FORMULA_OVERRIDE.get(name, r["source_or_formula"]),
            "preprocess_policy": r["preprocess_policy"],
            "missing_rate": round(r["missing_count"] / 333, 4),
            "mutual_exclusion_group": "" if pd.isna(r["mutual_exclusion_group_id"]) else r["mutual_exclusion_group_id"],
            "exact_duplicate_group": "" if pd.isna(r["exact_duplicate_group_id"]) else r["exact_duplicate_group_id"],
            "coverage_status": status,
            "recommended_role": role,
            "reason": reason,
        })
    for _, r in derived_spec.iterrows():
        name = r["feature_name"]
        status, role, reason = AUDIT[name]
        rows.append({
            "feature_name": name,
            "source": "v004c-factor-dictionary-0.1 (预声明派生)",
            "available_as_of": r["available_as_of"],
            "existing_or_proposed": "EXISTING",
            "coverage_mechanism": COVERAGE_MECHANISM[name],
            "existing_stage2_1_mechanism": "PREDECLARED_DERIVED",
            "formula_or_definition": r["formula"],
            "preprocess_policy": r["preprocess_policy"],
            "missing_rate": 0.0,
            "mutual_exclusion_group": "",
            "exact_duplicate_group": "",
            "coverage_status": status,
            "recommended_role": role,
            "reason": reason,
        })
    for name in extra_names:
        status, role, reason = AUDIT[name]
        src = "v004c-d1-dataset-0.1" if name in set(dict_df["source_column"]) else "PROPOSED (data/cache/daily)"
        avail = "D1_CLOSE"
        missing_rate = None
        if name in set(dict_df["source_column"]):
            drow = dict_df[dict_df["source_column"] == name].iloc[0]
            avail = drow["available_as_of"]
            missing_rate = round(drow["missing_count"] / 333, 4)
        elif name in PROPOSED_FIELDS:
            series = proposed[name]
            missing_rate = round(float(series.isna().mean()), 4)
        rows.append({
            "feature_name": name,
            "source": src,
            "available_as_of": avail,
            "existing_or_proposed": "EXISTING" if name in set(dict_df["source_column"]) else "PROPOSED",
            "coverage_mechanism": COVERAGE_MECHANISM[name],
            "existing_stage2_1_mechanism": "",
            "formula_or_definition": FORMULA_OVERRIDE.get(name, ""),
            "preprocess_policy": "",
            "missing_rate": missing_rate,
            "mutual_exclusion_group": "",
            "exact_duplicate_group": "",
            "coverage_status": status,
            "recommended_role": role,
            "reason": reason,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["coverage_mechanism", "coverage_status", "feature_name"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} -> {OUT_CSV.name}")

    # 提案字段统计 (供 review 引用; d1_ma20_slope 为 DEFER, 仅保留覆盖验证能力)
    for name in PROPOSED_FIELDS:
        s = proposed[name]
        ok = s.notna()
        label = "DEFER_NOT_REQUIRED_FOR_V001" if name == "d1_ma20_slope" else "NEW_REQUIRED"
        print(f"{name} [{label}]: n={int(ok.sum())}/{len(s)} missing={int((~ok).sum())} "
              f"rate={s.isna().mean():.4f} min={s.min():.6f} max={s.max():.6f} "
              f"median={s.median():.6f}")
    print("cache coverage reasons:", proposed["reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()

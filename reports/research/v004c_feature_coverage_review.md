# v004c 特征覆盖核对 — 正式模型开发前的机制覆盖与缺口审查

- 阶段: v004c 特征覆盖核对 (正式模型开发前)
- 范围: 只回答"现有数据已经有什么、真正缺什么、什么不应该进入模型"
- 生成日期: 2026-08-07
- 输入资产: Stage 1 `v004c-d1-dataset-0.1` / Stage 2.1 `v004c-factor-dictionary-0.1` / Stage 2.2 `v004c-stage2-2-univariate-0.1`
- 配套 CSV: `v004c_feature_coverage_v001.csv` (100 行)
- 复现脚本: `tools/v004c_feature_coverage_audit.py` (research utility only, 不进入 src/, 非正式运行入口)
- 字段公式核对依据: `v004c_feature_definitions.md` (v004c-features-0.1) / `v004c_feature_definitions_v02.md` (v004c-features-0.2) / `v004c_factor_dictionary_v001.csv`
- 修正版本: 2026-08-07 — 严格日级最大回撤时序定义 (prior peak 只用严格较早交易日)、
  Target-blind usecols 最小读取、角色收敛为 3 个 NEW_REQUIRED
  (提交 `research(v004c): correct feature coverage definitions`)

## 声明

本审查未训练任何模型, 未运行 Logistic Regression, 未生成预测,
未使用七月 Target7 / AUC / 命中率 / Top3 做任何决定;
只读取了七月特征值、缺失率与无标签分布 (用于"字段是否稳定生成/是否漂移"判断)。
覆盖工具通过 usecols 仅读取训练表 event_id/code/break_date,
没有将任何标签、D2/D3 或旧模型字段读入内存。
本阶段未冻结任何 M1/M2 成员, 未开始新的 Stage 2.3 冻结体系。

---

## 一、核心结论 (十个问题的直接回答)

1. **已有可直接用于正式模型的字段**: 59 个准入因子全部 D1_CLOSE 可得且无泄漏;
   可直接复用的核心机制字段见第三节覆盖矩阵。
2. **同一机制的不同表达**: 7 个官方互斥组 (ME01..ME07, 来自 Stage 2.1 近重复审计,
   其中金额/量能替代表达对 spearman ≥ 0.999); 13 个精确重复组 (ED01..ED13);
   另有非官方近互补对 (d1_up_bar_volume_ratio ↔ down_bar_volume_ratio, up+down+doji=1)。
3. **不应该进入新模型的字段**: 全部 label / outcome_audit / existing_model_audit 字段
   (含 v004a / v002 / recognition_score), 全部绝对尺度 DERIVE_ONLY 原始价量字段,
   以及本表标记 REDUNDANT / FORBIDDEN / NO_MODEL_INPUT 的 39 行。
4. **MA5 / MA10 / MA20 技术状态**: 充分存在 (水平 + 斜率 + 二元 + 收回事件);
   MA10/MA20 已有足够 V001 技术状态表达, d1_ma20_slope 延后为 sensitivity;
   MA5/MA10 spread 可由冻结字段精确派生 (见第四节 B)。
5. **D0→D1 变化**: 已存在且公式已核实 — `break_open_return / break_high_return /
   break_close_return` 的基准均为 `break_prev_close = 末板日收盘 = D0 close`,
   确为 D0→D1 transition (不是 D1 当日收益)。量能变化经
   `break_volume_ratio_vs_board_days` (D1 vol / mean 板日 vol, 含 D0) 覆盖。
6. **最近 7 个交易日短期事件路径目前缺少**: 3 个机制 — 7 日累计涨幅、
   7 日最大回撤、7 日收盘在 7 日区间内的位置。现有字段全部锚定
   D1 当日 / D0 close / MA / 10·20 日计数, 无任何字段锚定整条路径。
7. **是否真的需要新增特征**: 需要, 但只需要 3 个 (全部满足第六节 E 的 13 条准入)。
8. **最少应新增字段**: 3 个 (7 日路径三机制: 累计延伸度 / 严格时序回撤 / 区间位置),
   不超过 6 个。
9. **M1 D1-structure baseline 支撑字段**: 已充分存在, 5 个槽位全部有 preferred 候选
   (见第六节 F)。
10. **M2 integrated model 槽位**: 8 个槽位定义见第六节 F, 全部 NOT_FROZEN。

---

## 二、数据窗口正确理解

底层配置未变 (`src/config.py`): `daily_history_days = 180`、`indicator_warmup_trading_days = 120`、
`adjust = none`; `src/indicators.py` 在完整历史上计算 MA5/MA10/MA20/MA30 后再筛选目标日期。

- **数据获取 / 指标预热窗口**: 约 120~180 个交易日 (现有系统原样), 用于正确计算 MA。
- **正式模型事件观察窗口**: 截至 D1 的最近 7 个有效交易日 (个股日线相邻交易日序列,
  与冻结数据集同口径), 描述当前短线事件路径。

本审查**没有修改** `daily_history_days`、`indicators.py`、行情加载器,
也没有为"7 日路径"重写任何 MA 或加载器。7 日路径字段严格从 `data/cache/daily`
(tencent_daily, 未复权, 1612 股, 逐股约 289 日历史) 计算, 与冻结数据集同源同口径。

---

## 三、A. 当前已有特征覆盖矩阵

覆盖判定: **充分覆盖** = 该机制有 ≥1 个可直接复用且非近重复的准入字段;
**部分覆盖** = 机制存在但层内残缺或全部为 sensitivity 层;
**缺失** = 无任何字段表达该机制。

### 1. 事件 / 连板状态 (BOARD_OR_EVENT_STATE) — 充分覆盖

| 字段 | 判定 | 角色 |
|---|---|---|
| board_streak_before_break (经派生 board_streak_is_3) | 二板/三板事件状态 | M1_BASELINE_CANDIDATE |
| break_day_in_pool | D1 断板日池状态 | M2_PRIMARY_CANDIDATE |
| recent_limit_up_count_10d | 10 日涨停次数 | CONTEXT_ONLY |
| recent_limit_up_count_20d / recent_pool_appearance_count_10d·20d / max_board_streak_20d | 20 日事件统计 | CONTEXT_ONLY (REUSE_AS_CONTEXT, 不自动进入主模型) |
| last_board_day_in_pool / pool_consecutive_count_last_board | D0 池交叉核对 | SENSITIVITY_ONLY |

核心事件窗口是最近 7 个交易日; 10/20 日事件统计 (recent_limit_up_count_10d/20d,
recent_pool_appearance_count_10d/20d, max_board_streak_20d) 全部判定为 **CONTEXT_ONLY /
REUSE_AS_CONTEXT**: 属于已有事件背景信息, 可保留用于上下文和后续敏感性比较,
但不是 V001 M2 的默认主模型输入, 不得因"已存在"自动成为主模型输入。

### 2. D0 横截面 (D0_CROSS_SECTION) — 部分覆盖

- board_day_amount_rank / board_day_turnover_rank / board_day_volume_rank
- 全部为 SENSITIVITY 层; amount/turnover 版缺失 10.2% (2026-05-06 前无池数据),
  volume 版缺失 2.7%。机制存在但层内为敏感性角色, 不作主模型输入。

### 3. D0→D1 transition (D0_TO_D1_TRANSITION) — 充分覆盖

- break_open_return = D1 open / D0 close − 1 (开盘缺口)
- break_high_return = D1 high / D0 close − 1 (最高延伸)
- **break_close_return = D1 close / D0 close − 1 (核心收盘过渡)**
- break_touched_limit_up (D1 触板, 15% 正例) / break_opened_from_limit_up (2.4% 正例, 敏感性)
- 量能变化: break_volume_ratio_vs_board_days = D1 vol / mean(板日 vol) (ME01 canonical)

**公式核实结论**: 三个 break_*_return 的基准是 `break_prev_close` (末板日收盘 = D0 收盘),
不是 D1 当日 open。因此它们确为 D0→D1 transition; `d1_open_to_close_return_raw`
才是 D1 当日 open→close, 属于 D1 价格结构, 不表达 D0→D1。

### 4. D1 价格结构 (D1_PRICE_ACTION) — 充分覆盖

- d1_open_to_close_return_raw (D1 日内收益)
- d1_low_to_close_recovery = (close−low)/low (低点回收)
- d1_high_to_close_drawdown_raw = (high−close)/high (冲高回落)
- d1_intraday_range = (high−low)/prev_close (振幅)
- d1_afternoon_return / d1_last_hour_return (尾盘时段, 近重复, 紧凑模型二选一)
- d1_close_location (敏感性), break_upper/lower_shadow_ratio (敏感性), d1_open_to_close_bucket (敏感性)

### 5. D1 成交结构 (D1_VOLUME_ACTIVITY) — 充分覆盖

- break_volume_ratio_vs_board_days (ME01 canonical, 相对本轮连板量能)
- d1_up_bar_volume_ratio (阳线量占比)
- down_bar_volume_ratio (阴线量占比, canonical 名)
- break_amount_ratio_vs_board_days (ME01 金额替代表达, 敏感性)

**原则执行**: volume 版与 amount 版为同一机制的替代表达, 已标记
MUTUALLY_EXCLUSIVE_EXPRESSION (ME01/ME02/ME03/ME04), 正式模型不得同时放入。

### 6. D1 筹码结构 (D1_CHIP_DISTRIBUTION) — 充分覆盖

- volume_above_d1_close_ratio (ME02 canonical, 收盘上方量占比)
- high_zone_volume_ratio (ME03 canonical, 高位区量占比, 28% 为 0)
- amount_above_d1_close_ratio / high_zone_amount_ratio (金额替代, 敏感性)

### 7. 尾盘压力 (D1_LATE_DAY_PRESSURE) — 充分覆盖

- late_day_sell_volume_ratio (ME04 canonical, time≥14:00 且 close<open 的量占比)
- late_day_sell_amount_ratio (金额替代, 敏感性)

### 8. VWAP (D1_VWAP_POSITION) — 充分覆盖

- d1_close_to_vwap_raw = close/vwap − 1 (ME05 canonical)
- d1_vwap_to_close_gap = (vwap−close)/close — NP05 判定 spearman −1.0, 与 canonical
  近乎完全反向 → 标记 REDUNDANT (模型只留一种方向, 保留 close_to_vwap_raw)

### 9. MA5 (D1_MA5_STATE) — 充分覆盖

- d1_close_to_ma5_raw / d1_low_to_ma5_raw / d1_high_to_ma5_raw (三锚点同机制)
- d1_ma5_slope (趋势方向)
- d1_true_reclaim_ma5 (下探收回, 6% 正例)
- d1_close_above_ma5 (ME06 canonical 二元项, 99% 恒 True) / consecutive_days_below_ma5 (ME06 替代, NP04 近单调)

### 10. MA10 / MA20 (D1_MA10_MA20_STATE) — 充分覆盖

- 已有: d1_close_to_ma10_raw / d1_low_to_ma10_raw / d1_close_to_ma20 (水平),
  d1_ma10_slope (趋势), d1_close_above_ma10 (ME07 canonical), d1_true_reclaim_ma10 (2.2%),
  consecutive_days_below_ma10 (ME07 替代) — MA10/MA20 技术状态已有足够 V001 表达
- d1_ma20_slope: 概念上合理, 但 MA10/MA20 技术背景已有充分表达;
  为控制 V001 特征数量和避免增加相近趋势表达, 本阶段不把它作为必补字段,
  判定 **DEFER_NOT_REQUIRED_FOR_V001 / SENSITIVITY_ONLY** (后续可作为敏感性候选重新评估;
  脚本保留其确定性计算能力)
- MA5/MA10 spread: 不新增字段 — 可由已有 MA5/MA10 距离字段精确确定:
  设 a = d1_close_to_ma5_raw, b = d1_close_to_ma10_raw, 则 MA5/MA10 − 1 = (1+b)/(1+a) − 1
  (不是简单做差; d1_ma5/d1_ma10 原始锚点也在冻结表中可作为模型表阶段的派生来源)
- 稳定性提示: Stage 2.2 无标签分布漂移 (六月→七月) 标记了
  d1_close_to_ma10_raw / d1_low_to_ma10_raw / d1_close_to_ma20 / d1_ma10_slope —
  记为建模前注意事项, 不构成字段删除理由, 也不用于字段选择。

### 11. 最近 7 日路径 (RECENT_7D_PATH) — 缺失

无任何现有字段锚定"本轮走势起点 → D1"的多日路径。缺口与新增见第五节 C / E。

---

## 四、B. MA 结论

1. **现有 MA 已从长历史正确计算**: 冻结数据集的 d1_ma5/10/20 由日线缓存
   close 滚动均值 (min_periods=周期) 派生; `src/indicators.py` 同样在完整历史上
   计算 MA5/10/20/30; `daily_history_days=180 / indicator_warmup_trading_days=120`
   满足 MA20 预热。**无需修改 indicators.py, 无需重算任何 MA。**
2. **最适合建模的现有 MA 字段**: d1_close_to_ma5_raw (M1 首选), d1_close_to_ma10_raw,
   d1_close_to_ma20 (水平三锚点), d1_ma5_slope / d1_ma10_slope (趋势),
   d1_true_reclaim_ma5 (事件)。MA5/10/20 是 D1 技术状态, 属于"长历史计算、
   D1 时点可得"的技术状态字段, 可以进入模型。
3. **高度重复的 MA 字段**: 同一 MA 的 close/low/high 三锚点 (机制相同);
   above 二元项 (99% 恒 True); reclaim 二元项与 low 锚点近重复; consecutive_below 与
   above 近单调 (NP03/NP04)。紧凑模型每组只留一个代表性表达。
4. **需要新增的 MA 字段**: 无。`d1_ma20_slope` 概念上合理 (MA20 趋势方向;
   冻结数据无法直接导出, 因为 ma20(D0) 不在冻结表中), 但 MA10/MA20 技术背景
   已有充分表达, 判定 DEFER_NOT_REQUIRED_FOR_V001 / SENSITIVITY_ONLY,
   本阶段不作为 V001 必补字段。MA5/MA10 spread 不新增
   (可由 a=d1_close_to_ma5_raw, b=d1_close_to_ma10_raw 精确确定:
   MA5/MA10 − 1 = (1+b)/(1+a) − 1)。

---

## 五、C. 最近 7 日缺口 + D. D0→D1 缺口

### C1. 真正缺失的短周期机制 (3 个)

| 机制 | 为什么现有字段不能表达 | 需要哪些原始数据 | 严格公式 (全部 ≤ D1) | D1 可用性 | 建议 |
|---|---|---|---|---|---|
| 7 日累计涨幅 (路径延伸度) | board_streak 只有 {2,3} 两个离散值; break_close_return 只锚 D0 close; 10/20 日字段是计数不是收益 | data/cache/daily OHLCV (逐股相邻交易日) | close(D1) / close(T−6) − 1, T−6 = D1 前第 6 个交易日 | 是 | **NEW_REQUIRED** |
| 7 日最大回撤 (路径质量) | d1_high_to_close_drawdown_raw / break_high_to_close_drawdown 均为 D1 当日日内回撤, 不含多日路径 | 同上 | max_{t∈[T−5..D1]} max(0, (prior_peak_t − low_t) / prior_peak_t); prior_peak_t = max(high_s), s < t (严格较早交易日的 high, 避免利用日线无法确定的同日 high/low 先后顺序) | 是 | **NEW_REQUIRED** |
| 7 日区间收盘位置 | d1_close_location 是 D1 日内位置; break_close_return 锚 D0 close; 两者都不锚 7 日区间 | 同上 | (close(D1) − 7d_low) / (7d_high − 7d_low); 7d_high == 7d_low 置空 (不产生 inf, 同 d1_close_location 政策) | 是 | **NEW_REQUIRED** |

### C2. 明确 NOT_NEEDED / DEFER 的候选机制

- **7 日涨停次数** (候选: recent_7d_limit_up_count): DEFER_NOT_REQUIRED_FOR_V001 —
  最近 7 日涨停次数可能包含当前连板之前的独立涨停事件 (先涨停→中断→再形成当前二板→D1 断板),
  因此并不严格等价于 board_streak_before_break; V001 已使用 board/event state + 7 日价格路径,
  为控制模型复杂度暂不新增该事件计数字段 (本阶段不实现该字段)。
- **D1/D0 成交量变化** (候选: d1_to_d0_volume_ratio): NOT_NEEDED —
  break_volume_ratio_vs_board_days 已表达"D1 相对本轮连板 (含 D0) 的量能",
  纯 D1/D0 比值更噪声 (D0 可能为一字板小量)。
- **D1/D0 成交额变化** (候选: d1_to_d0_amount_ratio): NOT_NEEDED —
  volume 版的金额替代表达, 按互斥原则不与 volume 版同入模型。
- **本轮连板逐阶段量价分解** (候选: board_stage_volume_price_decomposition):
  DEFER_NOT_REQUIRED_FOR_V001 — 需大量重建第一/二/三板逐日状态, 数据质量不稳定,
  V001 不需要。

### D1. D0→D1 缺口结论

- 价格过渡: **已充分覆盖** (break_open/high/close_return 全部以 D0 close 为基准,
  公式已逐字段核实; d1_open_to_close_return_raw 是 D1 当日收益, 属 D1 价格结构)。
- 成交量/额变化: **已存在** (break_volume_ratio_vs_board_days, ME01 canonical)。
- 需要补的: **无**。不需要新增 D0→D1 相关字段。

---

## 六、E. 最终新增字段建议 (3 个, ≤ 6)

全部满足 13 条准入: 现有 59 准入字段无等价表达 / 对应明确缺失机制 / D1 收盘可知 /
只依赖 D1 及更早 / 现有日线缓存稳定计算 (实测 333/333, 0 缺失) / 无新数据源 /
无指数 / 无集合竞价 / 无 D2 / 无旧模型输出 / 确定性可重算 / 缺失处理明确 /
不因六月单因子表现创建。

| priority | feature_name | mechanism | why_needed | source_data | formula |
|---|---|---|---|---|---|
| 1 | recent_7d_cumulative_return | RECENT_7D_PATH | 现有字段无多日累计延伸度表达; 二/三板类内区分度缺失 | data/cache/daily | close(D1)/close(T−6) − 1 |
| 2 | recent_7d_max_drawdown | RECENT_7D_PATH | 现有字段只有 D1 日内回撤; 路径内洗盘/分歧强度缺失 | data/cache/daily | max_{t∈[T−5..D1]} max(0, (prior_peak_t − low_t)/prior_peak_t), prior_peak_t = max(high_s), s < t |
| 3 | recent_7d_close_position | RECENT_7D_PATH | D1 收盘相对整条路径区间的位置缺失 | data/cache/daily | (close − 7d_low)/(7d_high − 7d_low), 退化置空 |

实测统计 (333 事件, D1 时点重算; 脚本对最大回撤内置 Case A/B/C 确定性 self-check):

| 字段 | 覆盖 | 缺失率 | min | median | max |
|---|---|---|---|---|---|
| recent_7d_cumulative_return | 333/333 | 0.0% | −0.207 | 0.190 | 0.639 |
| recent_7d_max_drawdown (严格日级定义) | 333/333 | 0.0% | 0.025 | 0.101 | 0.404 |
| recent_7d_close_position | 333/333 | 0.0% | 0.357 | 0.779 | 1.000 |

补充 (DEFER, 仅覆盖验证): d1_ma20_slope 仍由脚本确定性计算 (333/333, 0.0% 缺失,
min −0.031 / median 0.008 / max 0.054), 但正式结论为 DEFER_NOT_REQUIRED_FOR_V001 /
SENSITIVITY_ONLY, 不作为 V001 必补字段。

(本样本无 7d_high==7d_low 退化行, 但 NaN 政策已按规范编码, 防止静默 inf。)

---

## 七、F. 未来模型机制槽位 (只定义槽位, 不冻结成员)

### M0 — INTERCEPT BASELINE

- 槽位: 1 个 (截距)。概念保留 (见 v004c_stage2_3_superseded_20260807.md Still retained)。

### M1 — D1 STRUCTURE BASELINE (4~5 个槽位)

| 槽位 | preferred existing candidate | possible alternative | 状态 |
|---|---|---|---|
| D0→D1 price transition | break_close_return | break_open_return / break_high_return | NOT_FROZEN |
| D1 volume/activity | break_volume_ratio_vs_board_days | down_bar_volume_ratio | NOT_FROZEN |
| D1 chip structure | volume_above_d1_close_ratio | high_zone_volume_ratio | NOT_FROZEN |
| late-day pressure | late_day_sell_volume_ratio | — | NOT_FROZEN |
| VWAP | d1_close_to_vwap_raw | — | NOT_FROZEN |

### M2 — INTEGRATED V004C PRIMARY (6~8 个槽位)

| 槽位 | preferred existing / proposed candidate | possible alternative | 状态 |
|---|---|---|---|
| board/event state | board_streak_is_3, break_day_in_pool | recent_limit_up_count_10d (context) | NOT_FROZEN |
| recent 7d path | recent_7d_cumulative_return, recent_7d_max_drawdown | recent_7d_close_position | NOT_FROZEN |
| D0→D1 transition | break_close_return | break_high_return | NOT_FROZEN |
| D1 price structure | d1_open_to_close_return_raw | d1_high_to_close_drawdown_raw / d1_intraday_range | NOT_FROZEN |
| volume/chip structure | volume_above_d1_close_ratio | high_zone_volume_ratio / late_day_sell_volume_ratio | NOT_FROZEN |
| VWAP | d1_close_to_vwap_raw | — | NOT_FROZEN |
| MA5 | d1_close_to_ma5_raw | d1_ma5_slope / d1_true_reclaim_ma5 | NOT_FROZEN |
| MA10/MA20 context | d1_close_to_ma10_raw / d1_close_to_ma20 | d1_low_to_ma10_raw / d1_ma10_slope (d1_ma20_slope 延后为 sensitivity) | NOT_FROZEN |

M1 与 M2 的槽位与候选全部标记 **NOT_FROZEN**, 由下一阶段 (build compact v004c model table)
在模型表构建时决策。

---

## 八、不应进入新模型的字段清单

- **标签 / 未来信息 (FORBIDDEN)**: target7_daily_d2open_d3high, d2_open_daily,
  d3_high_daily, d3_close_daily, daily_d2open_to_d3high_return, 全部 outcome_audit 列,
  全部 POST_D3 时点 data_quality 列 (D2/D3 内容)。
- **旧模型输出 (FORBIDDEN)**: recognition_score, v004a_probability, v004a_rank,
  is_v004a_top3/10/15, v002_rank, is_v002_top3/10/15, v004a_score_source。
- **绝对尺度原始价量 (NO_MODEL_INPUT, DERIVE_ONLY)**: break_prev_close, d1_open/high/low/close,
  d1_volume, d1_amount, d1_ma5/10/20, d1_vwap — 只允许派生后使用。
- **精确重复别名 (REDUNDANT, ED01..ED13)**: break_open/high/low/close/intraday_range/
  high_to_close_drawdown/close_location/volume/amount, d1_down_bar_volume_ratio,
  volume_above_break_close_ratio, deprecated_d1_reclaimed_ma5/ma10_v01。
- **全缺失字段**: break_turnover_ratio (100% 缺失, 日线缓存无 turnover_rate)。
- **本阶段禁止的新增**: 市场指数 / 上证状态 / 创业板状态 / 全市场流动性 / 情绪周期 /
  人工 regime 标签 / 集合竞价 / D2 早盘·量·高低点 / D3 任何字段 / stage_post / no_trade /
  v002·v004a·v004b·v005 分数 / recognition_score / 旧 policy 输出;
  自动特征搜索 / 自动交互项 / PCA / 树模型 / RF / XGB / LGBM / NN / 自动 bucket·阈值搜索。

---

## 九、发现的字段定义问题 (BLOCKER 检查)

- 未发现现有字段定义 bug。全部关键公式与 v0.1/v0.2 特征定义文档逐条核对一致
  (break_*_return 基准、MA 滚动口径、VWAP 手数自适应、尾盘/高位区/下跌 bar 定义)。
- 无 BLOCKER; 无需停止后续模型开发。

---

## 十、遗留注意事项 (非阻塞)

1. Stage 2.2 无标签分布漂移标记: d1_close_to_ma10_raw, d1_low_to_ma10_raw,
   d1_close_to_ma20, d1_ma10_slope (六月→七月)。建模时按 train-fold-only 预处理
   与日期稳定性惯例处理, 不构成字段删除理由。
2. 10/20 日事件统计字段 (recent_limit_up_count_10d/20d, recent_pool_appearance_count_10d/20d,
   max_board_streak_20d) 标记 CONTEXT_ONLY / REUSE_AS_CONTEXT, 是否进入 M2 由模型表阶段决定,
   本阶段不冻结。
3. 新字段最近 7 日窗口使用个股日线相邻交易日序列 (与冻结数据集同口径);
   停牌日自然不在序列中, 无需额外处理。

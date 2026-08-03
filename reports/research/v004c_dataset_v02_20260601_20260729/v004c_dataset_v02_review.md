# v004c 断板修复研究数据集 0.2 — 研究说明报告

- 数据集版本: `v004c-dataset-0.2`;候选定义: `break-repair-0.1`;特征定义: `v004c-features-0.2`
- 目标: `target7_d2open_d3high`;部署状态: `research_only`
- 只描述数据,不训练模型。生成日期 2026-08-03,时区 Asia/Shanghai。
- 配套: `v004c_dataset_v02_all.csv`(全量 1020 行)、`v004c_training_primary_20260601_20260729.csv`(877 行)、`v004c_may_sensitivity_complete.csv`(141 行)、`v004c_candidate_audit_v02.csv`(9803 行)、标签质量/因子质量/冗余/差异/汇总报告、`v004c_feature_definitions_v02.md`、`v004c_dataset_manifest_v02.json`。

## 1. 为什么五月不进入首版主训练?

**原因是数据覆盖选择偏差(minute_data_coverage_selection_bias),不是行情质量问题。**

本地 5min 缓存为历史运行逐步累积,五月期间覆盖最碎片化: 在全部 1020 个候选观察中,五月候选 141 个(13.8%);而在 1576 只股票的 5min 缓存中,有 34 只止于 05-22 前后、26 只止于 06-12、33 只止于 06-23——大量股票在五月的分钟数据要么尚未缓存、要么后来被更晚的缓存状态覆盖。五月另有 268 个观察(108 个事件)因 `missing_minute_data`(259)/`missing_future_label`(6)/`insufficient_daily_history`(3)被判为不完整,只进审计。**完整样本与不完整样本的日线因子差异很小**(见第 14 节对比表),说明缺失主要由缓存覆盖决定,而非行情状态。

因此五月样本存在"只有分钟数据完整的那部分股票才能进入样本"的选择偏差。为了避免首版模型在未知偏差上拟合,五月暂不进入主训练,但**完整保留为敏感性数据**,且将来补齐分钟数据后可按同一质量规则重新评估、重新纳入。

## 2. 是否因为行情不同而删除五月?

**没有。** 五月数据全部保留: 141 行完整样本进入 `SENSITIVITY_MAY_COMPLETE`,268 行不完整观察进入 `AUDIT_ONLY_MAY_INCOMPLETE`(审计表),没有任何五月行被丢弃。排除原因字段写的是 `minute_data_coverage_selection_bias`,不是 `market_regime_bad`/`market_not_similar`/`old_market_environment`。**不得声称五月行情无效,也不得把五月解释为"坏行情样本"。**

## 3. 六月至七月主训练样本有多少行?

**877 行**(`v004c_training_primary_20260601_20260729.csv`,sample_role=TRAIN_PRIMARY 且 label_quality_ok=true)。

## 4. 独立 event_id 有多少?

- 全量(1020 行): **395 个事件**。
- 主训练(877 行): **343 个事件**(其中 332 个事件含 d0 观察,11 个事件只有 post 观察)。
- 五月敏感性(141 行): 56 个事件。

## 5. 信号日有多少?

主训练覆盖 **42 个 signal_date**(2026-06-01..07-29);每日候选 6~43 个(均值 20.9),每日 Target7 数 0~20。全量研究区间(05-06..07-29)共 60 个有候选的信号日。

## 6. Target7 正样本有多少?

- 全量: 304(29.8%)。
- 主训练: **260(29.6%)**。
- 五月敏感性: 42(29.8%)。

## 7. 尾亏样本有多少?

- 全量: 325(31.9%)。
- 主训练: **288(32.8%)**。
- 五月敏感性: 37(26.2%)。

## 8. D2/D3 标签质量审计排除了多少样本?

**2 行被排除**(label_quality_ok=false → sample_role=EXCLUDED_QUALITY),原因均为 D2/D3 分钟网格不完整:

| event_id | 原因 |
|---|---|
| 603580_2026-07-06 | D2(07-14)43 bar、D3(07-15)42 bar(复牌后数据不完整;停牌 07-07..07-13 有严格证明,日期偏移本身合格,但 bar 不完整) |
| 605028_2026-07-16 | D3(07-22)47 bar(缺 1 根) |

其余 1018 行全部通过: D1 全部 48 bar/首 09:35/末 15:00;D2/D3 日期与统一日历一致(唯一有日期偏移的 603580 有严格停牌证明)。注意: 被排除的 2 行恰巧都是 Target7 命中(2/2)——这是巧合(排除完全基于标签质量,与结果无关),但提醒后续不要在训练池外引用它们的标签。

## 9. v0.1 和 v0.2 有多少标签发生变化?

**0 条。** 标签值(v0.2 全量表)与 v0.1 完全一致: Target7 变化 0、tail_loss 变化 0(差异文件 `v004c_dataset_v01_v02_diff.csv` 中无 label_quality_changed 行)。v0.2 的变化在于**标签质量审计**(哪些样本的标签可信)与**训练视图划分**,而非标签数值本身。v0.1 的 1020 行在 v0.2 全量表中全部保留(1020/1020)。

## 10. MA5 reclaimed 字段如何修正?

v0.1 的 `d1_reclaimed_ma5` 实际只是 `d1_close >= d1_ma5`(收盘站上 MA5),并非"下探后收回"。v0.2:

- 旧字段重命名并废弃: `deprecated_d1_reclaimed_ma5_v01` / `deprecated_d1_reclaimed_ma10_v01`(仅全量表审计保留;与新版 `d1_close_above_*` pearson=1.0,见冗余报告)。
- 新增正确命名: `d1_close_above_ma5 = d1_close >= d1_ma5`;`d1_close_above_ma10` 同理。
- 新增真正回收语义: `d1_true_reclaim_ma5 = (d1_low <= d1_ma5) AND (d1_close >= d1_ma5)`(盘中下探到 MA5 下方、收盘收回);`d1_true_reclaim_ma10` 同理。
- 新增研究分桶 `d1_close_to_ma5_bucket`(6 桶: <−5% / −5~−2% / −2~+2% / +2~+5% / +5~+10% / ≥+10%),只作研究表达,未据此删除样本。

训练视图中三者并存(close_above、true_reclaim、bucket),由未来模型选择。

## 11. 哪些重复因子被从训练视图删除?

共 **9 个字段**从训练视图移除(全量表全部保留),冗余报告记录 Pearson/Spearman:

| 删除字段 | 保留字段 | 相关性(Spearman) | 关系 |
|---|---|---|---|
| amount_above_d1_close_ratio | volume_above_d1_close_ratio | 0.99996 | volume/amount 双版本 |
| high_zone_amount_ratio | high_zone_volume_ratio | 0.99989 | volume/amount 双版本 |
| late_day_sell_amount_ratio | late_day_sell_volume_ratio | 0.99890 | volume/amount 双版本 |
| d1_down_bar_volume_ratio | down_bar_volume_ratio | 1.00000 | 公式完全相同 |
| break_amount_ratio_vs_board_days | break_volume_ratio_vs_board_days | 0.99123 | volume/amount 双版本 |
| d1_vwap_to_close_gap | d1_close_to_vwap_raw | −1.00000 | 完全反向表达 |
| days_since_break | stage_group + post_day | (阶段重复) | 训练视图只保留一套阶段表达 |
| days_since_last_limit_up | post_day | 1.00000(常数平移) | 阶段重复 |
| repair_attempt_count | post_day | 0.92555(常数平移) | 阶段重复 |

另: 训练视图扫描发现 4 对 |spearman|≥0.85 的高相关对(d1_high_to_close_drawdown_raw↔d1_close_to_vwap_raw −0.85;d1_close_location↔d1_close_to_vwap_raw 0.90;d1_close_location↔volume_above_d1_close_ratio −0.88;d1_close_to_vwap_raw↔volume_above_d1_close_ratio −0.88),**仅记录不删除**(语义不同,后续建模时处理)。

## 12. 哪些字段暂时标记为 audit_only?

| 字段 | 原因 |
|---|---|
| recognition_score、board_day_amount_rank、board_day_turnover_rank、board_day_volume_rank | recognition 缺失率 24%(板日早于涨停池起始);单因子排序能力不足;只作审计 |
| v004a_probability、v004a_rank、v002_rank(及其 is_*_top3/10/15、is_*_scorable、v004a_score_source) | 现有模型覆盖不完整(v004a 覆盖 57.3%、v002 覆盖 19.6%);**避免 v004c 变成 v004a 下游重排模型** |

另: 涨停频率类(recent_limit_up_count_*、recent_pool_appearance_count_*、max_board_streak_20d)标记 research_hold(保留研究、未进 v0.2 训练视图);d1_ma5/10/20、斜率、连续低于天数等原始研究字段同样保留在全量表。

## 13. 当前数据是否适合训练四个头?

| 头 | 定义 | 主训练样本 | Target7/尾亏数 |
|---|---|---|---|
| d0 Target7 | stage_group=d0, target7 | 332 行 | 99 正样本(29.8%) |
| post Target7 | stage_group=post, target7 | 545 行(post_day1 301 + post_day2 244) | 161 正样本(29.5%) |
| d0 尾亏头 | stage_group=d0, tail_loss | 332 行 | 96 尾亏(28.9%) |
| post 尾亏头 | stage_group=post, tail_loss | 545 行 | 192 尾亏(35.2%) |

**判断**: 每个头都有 90+ 正样本/尾亏样本,但都处于"低复杂度模型"边界(正样本 40~100 区间建议 4~8 自由度;100+ 可尝试低复杂度概率模型)。d0 头(332 行/99 正样本)偏紧;post 头(545 行/161 正样本)更宽裕。**四个头均暂不宜各自单独训练正式概率模型**,更合理的做法是: (a) 合并 d0+post 用 stage_group 作特征(877 行/260 正样本,满足 100+ 门槛);或 (b) 对 d0/post 分别只训练 ≤4 自由度的极简模型;尾亏头同理。同一事件在 d0/post 之间的观察强相关,训练/验证必须按 event_id 分组切分。

## 14. 是否仍满足低复杂度模型样本门槛?

按任务给定标准(候选≥500 且 Target7 正样本≥100 → 可尝试低复杂度概率模型):

| 指标 | v0.2 主训练 | 门槛 | 判定 |
|---|---|---|---|
| 候选行数 | 877 | ≥500 | ✓ |
| Target7 正样本 | 260 | ≥100 | ✓ |
| 独立事件数 | 343 | — | ✓ |

**满足"低复杂度概率模型"门槛**(建议有效自由度 ≤8,按 event_id 分组切分)。五月敏感性(141 行/42 正样本)作为稳健性检查数据,不进入首版训练。

## 15. 下一步是否可以进入 v004c.0-research 训练?

**数据侧可以,训练侧仍需流程准备。** 具备条件:

1. 主训练视图(877 行)已就绪: 标签质量门全通过、阶段表达唯一、冗余因子已从训练视图移除、建议权重(1/(日数×事件数),均值归一化=1)已输出。
2. 标签质量审计完整: D1/D2/D3 分钟网格(48 bar/09:35/15:00)+ 统一日历 + 严格停牌证明。
3. 五个审计发现已记录: 2 行标签质量排除、4 对训练视图高相关对、break_turnover_ratio 全空、v004a/v002 覆盖缺口、五月覆盖偏差。

进入 v004c.0-research 前的建议(本任务未执行):
- 以 `stage_group` 为阶段特征、按 event_id 分组做训练/验证切分;
- 自由度 ≤ 8,优先考虑 d1_close_to_ma5_raw/bucket、d1_close_to_vwap_raw、volume_above_d1_close_ratio、break_close_location、break_volume_ratio_vs_board_days、d1_high_to_close_drawdown_raw、board_streak_before_break、stage_group 等;
- 尾亏目标(tail_loss_5pct)与 Target7 同时建模或作为约束;
- 用五月敏感性数据做系数稳健性检查(不参与训练);
- 对 d1_close_location 等 10 行一字板缺失(d1_high==d1_low)和 break_close_location 5 行缺失明确处理策略(如缺失即排除或单独分桶);
- 保持冻结: 训练窗口固定为 2026-06-01..07-29,标签只读到 07-31。

## 附: 五月完整 vs 不完整样本的日线因子对比(第 1 节引用的证据)

基于审计表与日线缓存重算(可用日线因子):

| 因子 | 完整(141 观察) | 不完整(268 观察) |
|---|---|---|
| break_close_return 均值 | −1.12% | −0.54% |
| break_open_return 均值 | −0.12% | −0.02% |
| break_high_to_close_drawdown 均值 | 5.14% | 4.80% |
| break_close_location 均值 | 0.395 | 0.421 |
| 触板率(break_touched_limit_up) | 9.9% | 9.7% |

差异幅度小(收盘收益差约 0.6 个百分点),未显示"不完整样本系统性更弱/更强"的明确模式;但样本量小,不能排除细微偏差,这正是五月暂不进入主训练的原因(记录在 `v004c_candidate_audit_v02.csv` 与 `v004c_dataset_summary_v02.csv`)。

## 附: 未执行事项声明

本任务未训练任何模型(逻辑回归/Elastic Net/尾风险模型/超参数选择/自动特征选择均未执行);未输出交易名单;未修改 CLI/holdout/v006;未联网;未刷新缓存;未修改任何现有文件;五月数据未删除。

# v004c May D1 Candidate Coverage / Recovery — review (自动生成)

- 任务窗口: May 2026-05-06 ~ 2026-05-31; June+July 2026-06-01 ~ 2026-07-29
- git HEAD: 17a19f7e5cc4e6589465595794184c3e80b8f25a
- git status: ?? reports/research/v004c_may_d1_coverage_v001_202605/
?? src/v004c_may_d1_coverage.py
?? tests/test_v004c_may_d1_coverage.py
?? tools/v004c_may_d1_coverage.py
- 本任务不训练模型 / 不做因子筛选 / 不修改冻结资产 (v004c_d1_dataset / v022 / Repair-State 未动)

## 1. June+July 复现 (冻结 v004c_training_d1_v001.csv 精确对齐)

业务语义选择器 (days_since_break==0 + board_streak in {2,3} + signal_date==break_date)
直接在 v0.2 候选审计表上重建, 不读取任何 v0.2 判定结果: 

- 业务语义全集: 349 行
- v02 CANDIDATE: 333 行 == 冻结 333 行: PASS
- signal_dates: candidate=42 / frozen=42 PASS
- 重复 event_id: candidate=0 / frozen=0
- event_id 差异 (frozen-candidate): []
- event_id 差异 (candidate-frozen): []

业务全集 349 中 v02 判定非 CANDIDATE 的 16 行 (QUALITY_FAILED / LABEL_UNAVAILABLE) 逐条记录于 v004c_may_d1_recovery_v001.csv; 它们是 v02 时代的覆盖排除, 与 May 的 97 行同性质 (minute_data_coverage_selection_bias)。不为凑 333 调整规则。

## 2. May D1 候选全集 (§4)

- rows: 146 | signal_dates: 18 (2026-05-06 ~ 2026-05-29) | unique stocks: 137
- board_streak: 2板=116 / 3板=30
- candidate_status 分布: {'QUALITY_FAILED': 92, 'CANDIDATE': 49, 'LABEL_UNAVAILABLE': 5}

## 3. 规则交叉验证 vs v0.2.2 全量表 (49 个 May d0 行)

| 检查项 | mismatch 行数 |
|---|---|
| label_d2_date | 0 |
| label_d3_date | 0 |
| daily_label_quality_ok | 0 |
| target7_daily_d2open_d3high | 0 |
| d1_minute_complete | 14 |
| d1_factor_quality_ok | 14 |

mismatch d1_minute_complete: ['000720_2026-05-11', '002421_2026-05-26', '002579_2026-05-29', '002938_2026-05-26', '002971_2026-05-18', '002975_2026-05-26', '002980_2026-05-26', '600172_2026-05-27', '600186_2026-05-26', '600758_2026-05-20', '600936_2026-05-11', '603206_2026-05-20', '603661_2026-05-26', '603938_2026-05-21']
mismatch d1_factor_quality_ok: ['000720_2026-05-11', '002421_2026-05-26', '002579_2026-05-29', '002938_2026-05-26', '002971_2026-05-18', '002975_2026-05-26', '002980_2026-05-26', '600172_2026-05-27', '600186_2026-05-26', '600758_2026-05-20', '600936_2026-05-11', '603206_2026-05-20', '603661_2026-05-26', '603938_2026-05-21']

label_d2/d3 与 daily_label_quality_ok 与 target7 必须零 mismatch (同一规则);
d1_minute_complete / d1_factor_quality_ok 的 mismatch 反映 v02 构建后 cache 被后续刷新覆盖
(见缺失模式)。

## 4. 四层覆盖 before -> after (§5 / §9)

| layer | before | after |
|---|---|---|
| DAILY_COMPLETE | 146 | 146 |
| MINUTE_COMPLETE | 44 | 44 |
| LABEL_COMPLETE | 142 | 142 |
| D1_FEATURE_COMPLETE | 44 | 44 |
| FULLY_COMPLETE | 40 | 40 |

说明: 工具幂等重跑, before 列为本次运行审计时的 cache 状态 (恢复已在先前运行完成); 恢复执行证据 (每行 rows/bars 变化) 见第 7 节, 恢复前这 4 行均为 d1_minute_complete=False, 恢复后=True (backup 保留恢复前副本)。

四层交叉统计 (恢复后, 全部 16 种交集):

| combination | rows |
|---|---|
| DAILY_COMPLETE+LABEL_COMPLETE | 102 |
| D1_FEATURE_COMPLETE+DAILY_COMPLETE+LABEL_COMPLETE+MINUTE_COMPLETE | 40 |
| D1_FEATURE_COMPLETE+DAILY_COMPLETE+MINUTE_COMPLETE | 4 |
| D1_FEATURE_COMPLETE | 0 |
| D1_FEATURE_COMPLETE+DAILY_COMPLETE | 0 |
| D1_FEATURE_COMPLETE+DAILY_COMPLETE+LABEL_COMPLETE | 0 |
| D1_FEATURE_COMPLETE+LABEL_COMPLETE | 0 |
| D1_FEATURE_COMPLETE+LABEL_COMPLETE+MINUTE_COMPLETE | 0 |
| D1_FEATURE_COMPLETE+MINUTE_COMPLETE | 0 |
| DAILY_COMPLETE | 0 |
| DAILY_COMPLETE+LABEL_COMPLETE+MINUTE_COMPLETE | 0 |
| DAILY_COMPLETE+MINUTE_COMPLETE | 0 |
| LABEL_COMPLETE | 0 |
| LABEL_COMPLETE+MINUTE_COMPLETE | 0 |
| MINUTE_COMPLETE | 0 |
| NONE_COMPLETE | 0 |

## 5. 日线完整但分钟缺失 (§6)

- rows: 102 | stocks: 96 | dates: 16
- 全部 minute-incomplete rows: 102 (stocks 96, dates 16)
- missing_reason 分布: {'d1_date_missing': 100, 'bar_grid_incomplete': 2}
- 按日期: {'2026-05-06': 2, '2026-05-07': 6, '2026-05-08': 11, '2026-05-11': 10, '2026-05-12': 8, '2026-05-13': 8, '2026-05-14': 7, '2026-05-15': 8, '2026-05-18': 1, '2026-05-19': 3, '2026-05-20': 8, '2026-05-21': 3, '2026-05-25': 5, '2026-05-26': 12, '2026-05-27': 8, '2026-05-29': 2}
- top codes: {'002552': 2, '002208': 2, '600396': 2, '600172': 2, '603989': 2, '603256': 2, '000066': 1, '000417': 1, '000783': 1, '000818': 1}

缺失集中在 2026-05-06..05-29 整体: 5min 缓存是滚动窗口 (~41 交易日), May 数据在 6/7 月刷新中被逐批覆盖, 属缓存 cutoff 问题而非单点损坏。

## 6. 恢复调查 (§7)

### Priority 1 — 本地合法副本 (raw/minute_5m, get_stock_bars 镜像)

- 全量扫描: 仅 4 行 raw 具备完整 48-bar D1 网格且 cache 缺失 (同源 sina_5m / adjust=none)
- 其余 106 行 minute-incomplete 中, raw 快照窗口为 6/7 月 (不同时间点写入), 无 May 数据

### Priority 2 — canonical 5min 端点 (现有实现, 只读)

- probe 000980: ok window=2026-06-10..2026-08-07 n_dates=42
- probe 603937: ok window=2026-06-10..2026-08-07 n_dates=42
- probe 002272: ok window=2026-06-10..2026-08-07 n_dates=42
- 结论: sina_5m datalen=1970 (~41 交易日) 滚动窗口不包含 May; Priority 2 不可用

### Priority 3 — UNRECOVERABLE_WITH_CURRENT_CANONICAL_SOURCE

- 不可恢复行数: 102; 不造数据 / 不用其他日期代替 / 不插值。

## 7. 恢复执行 (§8, before/after provenance)

- 002272_2026-05-21: rows 98 -> 146, d1_minute_complete False -> True, rows_added=48, backup=F:\fenqixuangu\data\backups\minute_bar_repairs\002272_5min.before_may_recovery.pkl
- 603158_2026-05-08: rows 50 -> 98, d1_minute_complete False -> True, rows_added=48, backup=F:\fenqixuangu\data\backups\minute_bar_repairs\603158_5min.before_may_recovery.pkl
- 603459_2026-05-12: rows 1970 -> 2018, d1_minute_complete False -> True, rows_added=48, backup=F:\fenqixuangu\data\backups\minute_bar_repairs\603459_5min.before_may_recovery.pkl
- 603937_2026-05-07: rows 50 -> 98, d1_minute_complete False -> True, rows_added=48, backup=F:\fenqixuangu\data\backups\minute_bar_repairs\603937_5min.before_may_recovery.pkl

恢复前每行记录 cache path/exists/size/mtime/sha256; 现有 cache 先备份到 data/backups/minute_bar_repairs/ 再合并 (datetime 去重, 保留 cache 既有行优先);
恢复只来自同源 raw 副本 (source=sina_5m / adjust=none), 未静默覆盖来源不明文件。

## 8. Q1-Q18

### Q1: June+July 业务语义候选全集与冻结 333 的复现是否精确?

全集 349 行; CANDIDATE 333 行 == 冻结 333 行; event_id 双向零差异 (PASS); signal_dates 42 == 冻结 42 (PASS); 无重复

### Q2: May D1 候选全集是什么?

146 行 / 18 dates / 137 股票; 2板 116 / 3板 30; 状态 {'QUALITY_FAILED': 92, 'CANDIDATE': 49, 'LABEL_UNAVAILABLE': 5}

### Q3: 同一 selector 是否应用于 May 与 June+July?

是。业务语义选择器 (d0+streak 2/3+signal==break) 唯一, 未按月份调整规则

### Q4: D1 日线覆盖?

146/146 行 (break 日 OHLC 有效 + MA 历史 >= 20 交易日); 明细见 candidates CSV 的 d1_daily_complete 列

### Q5: D1 5min 覆盖?

before 44 -> after 44 / 146; 正式规则 48 bar / 09:35 首 / 15:00 末; missing_reason 分布 {'d1_date_missing': 100, 'bar_grid_incomplete': 2}

### Q6: target7 标签覆盖?

142/146 行可判定; May 非未揭盲 holdout, 允许查看 Target 但禁止用 Target 决定行保留 (本任务未做)

### Q7: D1 特征构造完整?

44/146 (日线层 AND 5min 层)

### Q8: 四层交叉?

见第 4 节交叉表 (16 种交集全部列出); 

### Q9: 日线完整但分钟缺失清单?

102 行 (stocks 96, dates 16); 明细 v004c_may_d1_missing_minute_v001.csv

### Q10: 缺失是否集中?

是。缺失整体覆盖 05-06..05-29 (cache 滚动窗口被 6/7 月刷新覆盖); 无单点集中; top codes {'002552': 2, '002208': 2, '600396': 2, '600172': 2, '603989': 2, '603256': 2, '000066': 1, '000417': 1, '000783': 1, '000818': 1}

### Q11: Priority 1 恢复?

4 行可恢复 (raw 完整 48-bar + 同源); 已全部恢复

### Q12: Priority 2 恢复?

不可用: canonical sina_5m 滚动窗口不包含 May (只读探测结果见第 6 节)

### Q13: Priority 3 不可恢复?

102 行; 不造数据 / 不插值 / 不用其他日期代替

### Q14: 恢复 before/after?

见第 7 节; 每行含 path/size/mtime/sha256/rows/date range 记录, 恢复前已备份

### Q15: 恢复后整体覆盖?

MINUTE_COMPLETE 44 -> 44; FULLY_COMPLETE 40 -> 40; 剩余 incomplete 102 行

### Q16: 选择偏差评估?

May 146 个 d0 候选中仅 44 行 5min 完整 (~30.1%); v02 时代曾完整 49 行, 其中 14 行被后续缓存刷新覆盖; 恢复后 44 行。minute_data_coverage_selection_bias 依然存在且已量化: 若仅准入完整行训练, 覆盖分布 (日期/股票/2-3板) 与全候选池的可比性必须由后续准入评审检查; June+July 中同性质排除 16 行

### Q17: 本任务未做什么?

未训练模型 / 未做 Pairwise / 未筛选因子 / 未计算 IC-AUC-Top1-Top3 / 未修改 Repair-State / 未把 May 并入训练 / 未修改任何冻结历史资产

### Q18: 最终判定

READY_FOR_MAY_ELIGIBILITY_REVIEW — 复现精确、审计完整、恢复已执行、偏差已量化; May 是否准入训练由后续 May eligibility review 决定, 本任务不并入训练

## 声明

所有输出确定性 (固定排序 / 无时间戳); 恢复只写 data/cache/minute_5m 与 data/backups/ (均不入 Git); 本任务未 commit 任何冻结资产。
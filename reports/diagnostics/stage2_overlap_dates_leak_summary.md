# 两个重叠日期候选域漂移汇总

## 总表

| signal_date | old_signal_count | current_signal_count | shared_signal_count | added_signal_count | removed_signal_count | old_eligible_count | current_eligible_count | added_eligible_count | removed_eligible_count | old_scorable_count | current_scorable_count | added_scorable_count | removed_scorable_count | shared_business_field_mismatch_count | shared_all_signal_feature_mismatch_count | shared_model_domain_feature_mismatch_count | shared_model_feature_hash_match | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-26 | 333 | 333 | 333 | 0 | 0 | 205 | 205 | 0 | 0 | 205 | 205 | 0 | 0 | 0 | 0 | 0 | True | STABLE |
| 2026-06-29 | 131 | 194 | 131 | 63 | 0 | 21 | 41 | 20 | 0 | 21 | 41 | 20 | 0 | 0 | 12 | 0 | True | UPSTREAM_EXPANSION_WITH_NONSCORABLE_FEATURE_DRIFT |

## 正式结论

- 2026-06-26：`STABLE`。
- 2026-06-29：`UPSTREAM_EXPANSION_WITH_NONSCORABLE_FEATURE_DRIFT`。
- overall：`DATE_SPECIFIC_MIXED_DRIFT`。

## 问题回答

1. 6月26日 signal/eligible/scorable 均为 333/205/205，旧新完全一致。
2. 6月29日 signal 131→194（新增63、删除0），eligible/scorable 21→41（各新增20、删除0）。
3. 6月29日新增 signal：D0=2026-06-25 有43只（days_since_d0=4，WATCH_ONLY）；D0=2026-06-26 有20只（days_since_d0=3，全部 eligible/scorable）。
4. 两个日期所有旧 signal、eligible、scorable 均保留。
5. 共同候选 eligibility/scorable 状态无变化；旧21只模型域特征完全一致。
6. 6月29日另有12只共同 WATCH_ONLY 出现实质特征漂移；10只由盘中日线缓存直接解释，2只因旧 raw 缺失未完全解决。
7. 两日期不呈现同一种模式：6月26日稳定，6月29日同时有上游扩张和非模型域特征漂移。
8. 候选集合差异最早可观察于 signal pool；12只的数值漂移最早可观察于日线侧输入/派生层。
9. 不能仅凭旧候选文件证明63只扩张由具体 provider 或缓存导致；旧运行未保存 raw-source snapshot。
10. 阶段二两个正式报告任务完成；12只血缘诊断已单独记录。

旧21只与当前同21只模型输入既定语义 hash 均为 `2d0d3521c2af11a4e9a9a33630564d944466f84e0fd4ba8ac8ebdc8d099f65d8`。当前冻结收益标签若被查看，仅描述后续表现，不用于判断候选是否应纳入。

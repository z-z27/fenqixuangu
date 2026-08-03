# 2026-06-26 旧版与冻结版候选域对比

## 结果

- signal：333 / 333；新增0，删除0，重合333。
- eligible：205 / 205；新增0，删除0。
- scorable：205 / 205；新增0，删除0。
- 共同候选业务字段 mismatch：0；指定模型字段超容差 mismatch：0。
- 模型域既定语义 hash（旧/当前一致）：`4dfe944067b68a69b06f87de25be4721c6de1bb2439a1bb227b80f56384a89bf`。
- 分类：`STABLE`。

## 当前冻结规则的结构性排除

| code | name | exclusion_reason | listing_date | suspension_start_date | suspension_end_date | proof_method |
| --- | --- | --- | --- | --- | --- | --- |
| 603001 | 奥康国际 | suspended_on_signal_date |  | 2026-06-25 | 2026-07-01 | eastmoney_suspension_status_v1 |
| 603407 | 长裕集团 | insufficient_listing_history | 2026-05-11 |  |  | weekday_upper_bound_v1 |
| 603459 | 红板科技 | insufficient_listing_history | 2026-04-08 |  |  | weekday_upper_bound_v1 |

该表只描述当前冻结 membership，不推断旧运行拥有相同排除证据。

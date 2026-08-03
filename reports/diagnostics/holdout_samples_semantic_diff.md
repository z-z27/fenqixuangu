# Frozen holdout samples semantic diff

## 结论

`REPRESENTATION_ONLY_DIFFERENCE`

create attempt、verify attempt 与 published candidate CSV 的字节完全相同。manifest expected hash 可以由同一 CSV 使用 `float_precision="round_trip"` 和项目现有 `canonical_rows_sha256` 精确复现；holdout 默认 pandas 浮点解析产生了少量 binary-float 末位漂移。主键、离散业务字段及 `np.isclose(rtol=1e-10, atol=1e-10, equal_nan=True)` 下的全部数值字段均一致。

## 1. 文件身份

- Manifest：`F:\fenqixuangu\reports\history_samples\2026-06-26_2026-07-08\history_universe_manifest_2026-06-26_2026-07-08.json`
- Manifest 中记录的 samples 路径：**无此字段**。
- 按 holdout 现有同目录/同后缀规则推断的 samples：`F:\fenqixuangu\reports\history_samples\2026-06-26_2026-07-08\history_candidates_2026-06-26_2026-07-08.csv`
- Holdout 实际读取的 samples：`F:\fenqixuangu\reports\history_samples\2026-06-26_2026-07-08\history_candidates_2026-06-26_2026-07-08.csv`
- 推断路径与实际路径相同：`True`
- Create attempt 样本：`F:\fenqixuangu\reports\history_samples\2026-06-26_2026-07-08\attempts\20260714T135016189428Z-ba94ee7a467e46f4a0c1209b4c8ab2ff\history_candidates_2026-06-26_2026-07-08.csv`
- Create attempt 与 published 路径字符串相同：`False`
- Create attempt 与 published 文件内容相同：`True`

| 工件 | 大小 | mtime UTC | 字节 SHA256 |
|---|---:|---|---|
| create attempt samples | 3107138 | 2026-07-14T14:03:57.227825165+00:00 | `f0a98512f1733f08b90d1555795b52fc9610b3ece754cfcf3141b6780ee001fc` |
| holdout/published samples | 3107138 | 2026-07-14T14:17:17.705963850+00:00 | `f0a98512f1733f08b90d1555795b52fc9610b3ece754cfcf3141b6780ee001fc` |
| manifest | 36431 | 2026-07-14T14:17:18.455002308+00:00 | `255db6f78f6fdfc55f1cf743cc4a06ab4bd9ddd06fc9f88e03d7b7d9043b76bc` |

两个 attempt 与 published CSV 均为 `3107138` bytes、SHA256 `f0a98512f1733f08b90d1555795b52fc9610b3ece754cfcf3141b6780ee001fc`。因此不是错误输入文件。

## 2. Schema

- Manifest 生成常量列数：57
- Holdout 读取列数：57
- 列名一致：`True`
- 列顺序一致：`True`
- 缺失列：`[]`
- 额外列：`[]`

列顺序：

```text
requested_signal_date, signal_date, code, name, d0_date, days_since_d0, consecutive_boards, signal_type, allowed_bool, eligible_for_trade, v004a_scorable_bool, v004a_exclusion_reason, total_score, graph_quality_score, active_money_score, active_cooling_score, support_score, theme_score, trend_hold_score, entry_width_score, d1_low_ma10_pct, d1_close_ma10_pct, d1_close_vwap_pct, low_absorb_width_pct, invalid_distance_pct, support_type, low_absorb_min, low_absorb_max, invalid_price, candidate_base_price, candidate_evaluable, future_trade_days_available, d2_trade_date, d3_trade_date, d2_open_price, d3_high_price, d3_close_price, d2open_d3high_return_pct, d2open_d3close_return_pct, candidate_d2_max_return_pct, candidate_d2_close_return_pct, candidate_d2_max_drawdown_pct, candidate_d3_max_return_pct, candidate_d3_close_return_pct, candidate_d3_max_drawdown_pct, candidate_d5_max_return_pct, candidate_d5_close_return_pct, candidate_d5_max_drawdown_pct, candidate_d10_max_return_pct, candidate_d10_close_return_pct, candidate_d10_max_drawdown_pct, target7, target10, target7_d2open_d3high, target7_d2open_d3close, reasons, key_zones_json
```

## 3. 行数与主键

主键：`signal_date + code`；两份 CSV 均按 `dtype=str, keep_default_na=False, na_filter=False` 读取，code 规范为六位字符串。

| 指标 | Expected | Actual | 一致 |
|---|---:|---:|---|
| row count | 2007 | 2007 | True |
| unique key count | 2007 | 2007 | True |
| duplicate key rows | 0 | 0 | True |
| missing keys | 0 | 0 | True |
| extra keys | 0 | 0 | True |

## 4. 逐列语义比较

- 精确字段列数：20
- 数值字段列数：37
- 精确业务字段 mismatch cells：0
- 数值字段超出 `rtol=1e-10, atol=1e-10` 的 mismatch cells：0
- 默认 parser 与 round-trip parser 的 raw float 差异 cells：5537
- 这些差异中改变项目 12-significant-digit canonical 文本的 cells：1
- 存在表示差异的列：`low_absorb_width_pct`, `invalid_distance_pct`, `low_absorb_min`, `low_absorb_max`, `invalid_price`, `d2open_d3high_return_pct`, `d2open_d3close_return_pct`, `candidate_d2_max_return_pct`, `candidate_d2_close_return_pct`, `candidate_d2_max_drawdown_pct`, `candidate_d3_max_return_pct`, `candidate_d3_close_return_pct`, `candidate_d3_max_drawdown_pct`, `candidate_d5_max_return_pct`, `candidate_d5_close_return_pct`, `candidate_d5_max_drawdown_pct`, `candidate_d10_max_return_pct`, `candidate_d10_close_return_pct`, `candidate_d10_max_drawdown_pct`

唯一改变 12-significant-digit canonical 文本、从而触发整表 hash 不同的单元格为：

```text
signal_date=2026-07-02
code=002317
column=candidate_d10_max_drawdown_pct
round_trip float=-4.0015100037749995
holdout default float=-4.0015100037750004
round_trip canonical 12g=-4.00151000377
holdout canonical 12g=-4.00151000378
absolute difference=8.881784197001252e-16
relative difference=2.2196081450807897e-16
np.isclose=True
```

| 列 | raw float 差异 | canonical 12g 差异 | 语义 mismatch | 最大绝对差 | 最大相对差 |
|---|---:|---:|---:|---:|---:|
| `low_absorb_width_pct` | 311 | 0 | 0 | 8.8817841970012523e-16 | 7.1994446212073338e-14 |
| `invalid_distance_pct` | 311 | 0 | 0 | 8.8817841970012523e-16 | 7.1994446212073338e-14 |
| `low_absorb_min` | 53 | 0 | 0 | 1.4210854715202004e-14 | 1.880936932867694e-16 |
| `low_absorb_max` | 149 | 0 | 0 | 1.1368683772161603e-13 | 2.1843362407700823e-16 |
| `invalid_price` | 53 | 0 | 0 | 1.4210854715202004e-14 | 1.880936932867694e-16 |
| `d2open_d3high_return_pct` | 327 | 0 | 0 | 3.5527136788005009e-15 | 5.2325591776215245e-15 |
| `d2open_d3close_return_pct` | 317 | 0 | 0 | 3.5527136788005009e-15 | 2.1471305636269301e-14 |
| `candidate_d2_max_return_pct` | 327 | 0 | 0 | 3.5527136788005009e-15 | 3.4643468649321258e-15 |
| `candidate_d2_close_return_pct` | 358 | 0 | 0 | 3.5527136788005009e-15 | 4.1494585545384537e-15 |
| `candidate_d2_max_drawdown_pct` | 305 | 0 | 0 | 3.5527136788005009e-15 | 1.0616629103611897e-14 |
| `candidate_d3_max_return_pct` | 303 | 0 | 0 | 3.5527136788005009e-15 | 3.4643468649321258e-15 |
| `candidate_d3_close_return_pct` | 348 | 0 | 0 | 7.1054273576010019e-15 | 5.5629112427597432e-15 |
| `candidate_d3_max_drawdown_pct` | 310 | 0 | 0 | 7.1054273576010019e-15 | 1.0616629103611897e-14 |
| `candidate_d5_max_return_pct` | 310 | 0 | 0 | 3.5527136788005009e-15 | 3.4643468649321258e-15 |
| `candidate_d5_close_return_pct` | 342 | 0 | 0 | 7.1054273576010019e-15 | 8.7929663550309902e-16 |
| `candidate_d5_max_drawdown_pct` | 338 | 0 | 0 | 3.5527136788005009e-15 | 1.1624035067824512e-15 |
| `candidate_d10_max_return_pct` | 314 | 0 | 0 | 7.1054273576010019e-15 | 3.2333163924035106e-15 |
| `candidate_d10_close_return_pct` | 367 | 0 | 0 | 7.1054273576010019e-15 | 3.8243366806714949e-15 |
| `candidate_d10_max_drawdown_pct` | 394 | 1 | 0 | 7.1054273576010019e-15 | 7.1637140663946308e-16 |

每列前 10 条差异记录、缺失值数量及 dtype 均在伴随 CSV 的 `sample_differences_json` 字段中。所有表示差异均通过指定的 `np.isclose` 检查，没有业务意义差异。

## 5. 四种摘要

| 摘要 | Expected | Actual | 一致 |
|---|---|---|---|
| 文件字节 hash | `f0a98512f1733f08b90d1555795b52fc9610b3ece754cfcf3141b6780ee001fc` | `f0a98512f1733f08b90d1555795b52fc9610b3ece754cfcf3141b6780ee001fc` | True |
| 主键集合 hash | `9a18f08d07d6f5b7ac688b8eb529d9829191adc1e11eb2cf72b8cd9ee086f24d` | `9a18f08d07d6f5b7ac688b8eb529d9829191adc1e11eb2cf72b8cd9ee086f24d` | True |
| 关键离散业务字段 hash | `43f3b2f9062c77918e68a6d8d6e57ea2964946d477f27ffccf9a3e2c1f2a4547` | `43f3b2f9062c77918e68a6d8d6e57ea2964946d477f27ffccf9a3e2c1f2a4547` | True |
| 统一规范化完整语义 hash | `b16c5f38476e9fa879634c97057da9ec3d58a01d6038650325f46c5526c49cca` | `b16c5f38476e9fa879634c97057da9ec3d58a01d6038650325f46c5526c49cca` | True |

现有项目 hash 复现：

| 读取方式 | `canonical_rows_sha256` | 与 manifest 一致 |
|---|---|---|
| manifest / round-trip parser | `b16c5f38476e9fa879634c97057da9ec3d58a01d6038650325f46c5526c49cca` | True |
| holdout 当前默认 parser | `8e9f29a04add3226fd84b97e5e3635f35c0efd5c07a05d99f2cf7d1e8a89663b` | False |

同一个 hash 实现对 round-trip frame 精确产生 manifest expected hash，因此不是 hash 算法实现不一致。

## 6. 回答

- 是否是同一个文件：manifest 未保存 samples 路径；holdout 推断路径与实际读取路径相同。create、verify、published 三份物理副本路径不同，但字节完全相同。
- 行数是否相同：是，均为 2007。
- Key set 是否相同：是，2007 个唯一键，无重复、missing 或 extra。
- 哪些列不同：仅上表所列数值列存在默认 parser 与 round-trip parser 的 binary-float 表示差异；精确业务列无差异。
- 是否存在有业务意义的差异：否；全部数值差异均在指定容差内。
- 是否可以安全进入 holdout：业务内容语义上可以；但当前严格 invariant 不应手工绕过，必须先在未来获授权的代码修改中统一 CSV 读取规范。
- 现有 hash 是否应该重算或修复：不应重算或改写 manifest。manifest hash 可被 round-trip 读取精确复现；应修复 holdout 读取边界（例如 round-trip 浮点解析，或全字符串读取后按 schema 统一解析），而不是改 accepted hash。

## 只读保证

诊断前后源 samples 与 manifest 的字节 SHA256 均已断言不变。本次只新增本 Markdown 与伴随 CSV。

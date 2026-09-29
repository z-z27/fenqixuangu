# 问题记录：交易日口径问题

记录时间：2026-09-29
触发场景：`python -m src.run_daily_v005 --date 2026-09-28 --lookback-days 5 --days 10 --workers 6 --coefficient-predict-date 2026-06-26 --grid-id 4` 卡死

本文件记录两个问题，**同一个病根：该用交易日的地方用了自然日。**

| # | 问题 | 状态 |
|---|---|---|
| 1 | 节假日被误判为交易日，触发全市场 3099 只日线回扫，卡死数小时 | **已修复** |
| 2 | 跨周末时"涨停次日"被算成"涨停后第 3 日" | **未修复，需决策** |

---

## 问题 1：节假日被误判为交易日，导致全市场回扫卡死【已修复】

### 现象

```
PS F:\fenqixuangu> python -m src.run_daily_v005 --date 2026-09-28 --lookback-days 5 `
>>   --days 10 --workers 6 --coefficient-predict-date 2026-06-26 --grid-id 4
[derive-limitups] 2026-09-25 start universe=3099 workers=6
<卡死，数小时无输出>
```

### 根因链条

1. **`--lookback-days 5` 是 5 个自然日**（首版设计如此）。锚点 2026-09-28 往前数 5 个自然日 = 09-24 ~ 09-28。
2. **09-25 是中秋节假期**（周五），但 `collect_limit_ups` 循环里只有一句 `if current.weekday() >= 5: continue` —— 只能识别周末，**识别不出节假日**。
3. 09-25 被当成交易日 → 抓涨停池 → akshare 返回空 → 异常。
4. 回落到 `_derive_limit_up_pool_from_daily`（`src/loaders.py:436`）→ **扫描全市场 3099 只个股的日线**。
5. 日线缓存的判定 `_daily_has_date_and_previous`（`src/loaders.py:924`）要求缓存帧里**必须含有 09-25 这一行**；但所有缓存日线最晚只到 09-24 → 条件永远不成立 → 3099 只全部重新走网络抓取。
6. **结果必然是空的**：`_derive_limit_up_row_from_daily`（`src/loaders.py:806`）在找不到当日 bar 时返回 `None`，节假日本来就没有 bar。
7. 最终错误被吞进 `errors` 列表后 `continue`，静默跳过。

### 代价

| | 修复前 | 修复后 |
|---|---|---|
| 09-25 处理方式 | 全市场 3099 只日线回扫 | 查日历，直接跳过 |
| 单只耗时（实测） | 5 ~ 30 秒 | — |
| 总耗时估算 | 3099 ÷ 6 线程 × 5~30s ≈ **43 分钟 ~ 4.3 小时** | **1.31 秒** |
| 产出 | 空（白跑） | 正常 |

### 触发条件

**任何 5 个自然日窗口内包含节假日的运行。**

日常使用中，每周一/周二窗口跨周末不受影响（周末本来就被 `weekday()` 挡住了），所以只在节假日前后爆发 —— 这正是它长期没被发现的原因。

### 修复内容

**新增 `src/trading_calendar.py`** —— 多源级联 + 本地缓存 + fail-closed：

| 顺序 | 源 | 覆盖范围 | 特点 |
|---|---|---|---|
| 1 | `ak.tool_trade_date_hist_sina()` | 1990-12-19 → 2026-12-31（**含未来日期**） | 主源，实测 8797 行 |
| 2 | `baostock.query_trade_dates()` | 含未来日期 | 备用 |
| 3 | 本地日线缓存横截面反推 | 仅历史 | **离线可用**；样本 < 50 只则拒绝，不硬推 |

- 缓存落 `data/cache/trade_calendar/`（`calendar_<source>.pkl` + `.meta.json`，带 coverage 区间），命中覆盖区间时**零网络开销**。
- **全部源失败 → 抛 `TradingCalendarError`，绝不退回 `weekday()` 猜一个**（fail-closed）。
- 导出 `NonTradingDayError`：窗口内一个交易日都没有时抛出。

**`src/loaders.py:94-102`** —— `collect_limit_ups` 的 `weekday()` 判断换成查日历：

```python
window_start = (anchor - pd.Timedelta(days=max(1, lookback_days) - 1)).strftime("%Y-%m-%d")
window_end = anchor.strftime("%Y-%m-%d")
trading_dates = self.trading_calendar.trading_days(window_start, window_end)
if not trading_dates:
    raise NonTradingDayError(...)
```

**窗口语义刻意保持不变**：`lookback_days` 仍是自然日，日历只负责把非交易日剔出去，不改变窗口长度。

**`src/history_samples.py:1779`** —— 按异常**类型**（而非 message 正则）归类为 `exchange_calendar` 证明，绕过昂贵的日线扫描。

### 影响面评估（关键结论）

**修复前后 `collect_limit_ups` 的返回值逐字节一致。**

节假日在修复前也**不产出任何行**（第 6 步：结果必然为空，只是要花几小时才发现）。因此：

- `raw_source_pool` 不变
- canonical snapshot 不受影响
- 现有报告、冻结产物全部有效

这是本次修复可以安全落地的核心前提。

### 验证

```
修复前: [derive-limitups] 2026-09-25 start universe=3099 → 数小时 → 结果必然为空
修复后: elapsed 1.31s   dates: ['2026-09-24', '2026-09-28']   共 77 行
```

窗口逐日核对：

| 日期 | 星期 | 涨停池 |
|---|---|---|
| 2026-09-24 | 周四 | 46 只 |
| 2026-09-25 | 周五 | — 中秋节休市（正确跳过） |
| 2026-09-26 | 周六 | — 休市 |
| 2026-09-27 | 周日 | — 休市 |
| 2026-09-28 | 周一 | 31 只 |

**测试**：新增 `tests/test_trading_calendar.py`（10 个用例，全离线，通过注入 `fetchers` 避免联网）+ `tests/test_history_universe_integrity.py::test_exchange_calendar_proof_marks_holiday_non_trading`。

`tools/verify.ps1` 全量 **825 个测试通过**（含新增 11 个）。

---

## 问题 2：涨停"次日"被算成"涨停后第 3 日"【未修复，需决策】

### 现象

跨周末时，**涨停后的第 1 个交易日（涨停次日）会被算成"涨停后第 3 天"**。周四、周三涨停的股票更被算成第 4、5 天，直接遭丢弃。

### 位置

`src/signal_engine.py:66`

```python
days_since_d0 = (pd.Timestamp(trade_date) - pd.Timestamp(d0_date)).days
```

**`git log -S` 确认：该行由 2026-06-26 `4e0389e part finish` 引入，之后再未修改。**

### 规则

`src/signal_engine.py:99-103`：

- `days_since_d0 <= 0` → "今日仍涨停，等待首次分歧日" → 不可买
- `days_since_d0 > 3` → "涨停后 N 天，分歧时效已过" → 不可买
- **只有 1 / 2 / 3 可买**

### 偏移量：跨一个周末固定 +2

| 涨停日 D0 | 评估日 | 真实（交易日） | 被算成（自然日） | 结果 |
|---|---|---|---|---|
| 周五 | 周一 | 第 1 天（**次日**） | **3** | 保留，但口径错 |
| 周四 | 周一 | 第 2 天 | **4** | **丢弃** |
| 周三 | 周一 | 第 3 天 | **5** | **丢弃** |
| 周五 | 周二 | 第 2 天 | **4** | **丢弃** |

净效果：**每跨一次周末，"第 2 天"和"第 3 天的分歧"全部消失，只剩下被误标成"第 3 天"的"次日"。**

### 实证：只有周一、周二失真

用交易日历重算现有全部 `reports/daily_signals/signals_*.csv`：

| 日期 | 星期 | 信号数 | 按自然日可买 | 按交易日可买 | 差 |
|---|---|---|---|---|---|
| 2026-09-09 | 三 | 167 | 119 | 119 | 0 |
| 2026-09-10 | 四 | 191 | 157 | 157 | 0 |
| 2026-09-11 | 五 | 217 | 119 | 119 | 0 |
| 2026-09-14 | **一** | 99 | 27 | 52 | **+25** |
| 2026-09-15 | **二** | 97 | 40 | 67 | **+27** |
| 2026-09-16 | 三 | 138 | 59 | 59 | 0 |
| 2026-09-17 | 四 | 167 | 124 | 124 | 0 |
| 2026-09-18 | 五 | 218 | 116 | 116 | 0 |
| 2026-09-21 | **一** | 170 | 49 | 78 | **+29** |
| 2026-09-22 | **二** | 175 | 68 | 117 | **+49** |
| 2026-09-23 | 三 | 158 | 114 | 114 | 0 |
| 2026-09-24 | 四 | 186 | 140 | 140 | 0 |
| 2026-09-28 | **一** | 70 | **0** | **39** | **+39** |

**规律：周一、周二失真，周三到周五一条不差。** 因为 5 个自然日的窗口只有在评估日是周一或周二时才会往回够到上周末。

**汇总：11511 条信号中，本该可买 7116 条，实际只认了 6151 条 —— 965 条（13.6%）被静默丢弃，且固定每周发生在周一和周二。**

### 方向性：只会漏，不会多

自然日 ≥ 交易日恒成立 ⇒ 自然日判定可买 ⊆ 交易日判定可买。**不会把不该买的算成可买，只会把该买的扔掉。**

### 极端案例：2026-09-28

```
RuntimeError: no eligible v005 daily rows after allowed/signal_type/base-price filters
  src/v005_daily_selector.py:483
```

09-28 手上只有两批：

| 来源 | 数量 | `days_since_d0` | 判定 |
|---|---|---|---|
| 09-28 当日涨停 | 31 只 | 0 | 今天还在涨停，不可买 |
| 09-24 涨停 | 39 只 | **4** | 分歧时效已过，丢弃 |

而要落在 1~3，必须有股票在 09-25 / 09-26 / 09-27 涨停 —— 那三天全休市，不可能有。

**09-24 涨停的股票，09-28 是它涨停后的第 1 个交易日（次日），正是该低吸的时候，却因为被算成"第 4 天"而全部作废。**

### 为什么长期没暴露

平时周一还能出结果，是因为周五的池子被标成 3、卡在边界上过关。**每次周一买到的那批，其实都是"次日"的股票，却被当成"第 3 天"来打分。**

### 可预测的下一步故障

**2026 国庆长假后（10-09 首个交易日）会一只都出不来**：5 个自然日窗口（10-05 ~ 10-09）内只有 10-09 一个交易日，窗口内所有候选的 `days_since_d0` 全是 0 → 全部"今日仍涨停" → eligible 为空 → 同样报错。

### 为什么没有直接修

`days_since_d0` 不只是判据，还是**模型特征**：

- `src/v004a.py:323-326` 派生 `days_since_d0_le1` / `eq2` / `ge3`
- `src/history_samples.py:1923` 用 `days_since_d0` 计算 `eligible_for_trade`
- 冻结系数（`--coefficient-predict-date 2026-06-26`、`--grid-id 4`）是按自然日语义训练出来的
- 现有 canonical history snapshot 全部基于该语义

**改成交易日 = 改特征定义 = 需要重训 + 重新冻结 + 现有 canonical 样本作废。** 这是研究决策，不是随手能打的补丁。

### 待决策的选项

| 方案 | 说明 | 代价 |
|---|---|---|
| A. 改定义 + 重训重冻 | 语义正确，长期解法 | 现有 canonical 样本 + 冻结系数作废，需完整重跑 |
| B. 保持现状，仅在文档标注 | 零风险 | 策略在长假后系统性失效，且周一/周二的"第 3 天"含义不实 |
| C. 新增交易日口径特征，与旧特征并存 | 可做对照，验证口径影响 | 需重训 |

---

## 复现与验证命令

```powershell
# 复现问题 1（修复前会卡死数小时）
python -c "
from src.loaders import MarketDataService
svc = MarketDataService()
pool = svc.collect_limit_ups(trade_date='2026-09-28', lookback_days=5, write_processed=False)
print(len(pool), sorted(pool['trade_date'].astype(str).unique()))
"
# 期望：1~2 秒完成，77 行，日期为 ['2026-09-24', '2026-09-28']

# 交易日历单测（离线）
python -m unittest tests.test_trading_calendar -v

# 全量离线验证
.\tools\verify.ps1
```

## 本次改动的文件清单

| 文件 | 改动 |
|---|---|
| `src/trading_calendar.py` | **新增** —— 交易日历模块 |
| `src/loaders.py` | `collect_limit_ups` 改用日历；新增 `NonTradingDayError` 分支；错误信息区分"无交易日"与"取数失败" |
| `src/history_samples.py` | 新增 `exchange_calendar` 证明分支 |
| `src/config.py` | `ensure_directories` 增加 `cache_dir / "trade_calendar"` |
| `tests/test_trading_calendar.py` | **新增** —— 10 个离线用例 |
| `tests/test_history_universe_integrity.py` | 新增 `test_exchange_calendar_proof_marks_holiday_non_trading` |

**注：问题 2 未做任何代码改动**，`src/signal_engine.py` 保持原样。

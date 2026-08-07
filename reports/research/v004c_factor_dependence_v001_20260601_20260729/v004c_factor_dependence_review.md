# v004c 因子结构纯 X 侧审查 (v004c factor dependence v001)

- 阶段: 多变量建模前的纯 X 侧因子结构审查 (X ONLY, 不训练模型, 不读取 Target)
- 分支: research-sample-analysis
- 输入: `v004c_model_table_v001.csv` SHA256: `c0de3bc639142c3e0b14b8018ff7ca13351ba46ed8f43cba3f54b18a261bd274`

## 0. 输入与 target-blind 约定

- X loader 使用显式 `usecols` 只加载 3 个标识列 (event_id, code, signal_date) + 11 个 primitive 列; `target7_daily_d2open_d3high` / D2 / D3 / 旧模型输出 从未被加载 (读取入口即 target-blind)
- June = **development / reference X sample** (2026-06-01 ~ 2026-06-30, 用于拟合 q01/q99/mu/sigma)
- July = **retrospective unlabeled X-stability slice** (2026-07-01 ~ 2026-07-29, 只 apply June reference 参数; 不得用 July 重算 mean/std, 不得读取 July Target)

- Stage 2.1 / coverage 结构交叉检查: 11 个 primitive 全部有授权来源 (allowlist: `break_open_return, d1_open_to_close_return_raw, d1_high_to_close_drawdown_raw, d1_close_to_vwap_raw, high_zone_volume_ratio, late_day_sell_volume_ratio, d1_ma10_slope, board_streak_is_3`; NEW_REQUIRED recent-7d: `recent_7d_cumulative_return, recent_7d_max_drawdown, recent_7d_close_position`)
- preprocess_policy 记录: break_open_return=FOLD_CLIP_Z; d1_open_to_close_return_raw=FOLD_CLIP_Z; d1_high_to_close_drawdown_raw=FOLD_CLIP_Z; d1_close_to_vwap_raw=FOLD_CLIP_Z; high_zone_volume_ratio=FOLD_CLIP_Z; late_day_sell_volume_ratio=FOLD_CLIP_Z; recent_7d_cumulative_return=FOLD_CLIP_Z (model-table schema 临时约定); recent_7d_max_drawdown=FOLD_CLIP_Z (model-table schema 临时约定); recent_7d_close_position=FOLD_CLIP_Z (model-table schema 临时约定); d1_ma10_slope=FOLD_CLIP_Z; board_streak_is_3=RAW_BINARY
- near-duplicate pairs (v004c_near_duplicate_pairs_v001) 命中本 universe: 无 (11 个 primitive 内部无 near-duplicate pair)

## 1. 样本规模

| 切片 | rows | signal dates | 日期范围 |
|---|---|---|---|
| June (development / reference X sample) | 173 | 21 | 2026-06-01 ~ 2026-06-30 |
| July (retrospective unlabeled X-stability slice) | 160 | 21 | 2026-07-01 ~ 2026-07-29 |

- 窗口外行: 0 (全部 333 行都在 June/July 窗口内)

## 2. primitive 分布 (raw 值, June / July)

### June

| stat | break_open_return | d1_open_to_close_return_raw | d1_high_to_close_drawdown_raw | d1_close_to_vwap_raw | high_zone_volume_ratio | late_day_sell_volume_ratio | recent_7d_cumulative_return | recent_7d_max_drawdown | recent_7d_close_position | d1_ma10_slope |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 173.0000 | 173.0000 | 173.0000 | 173.0000 | 173.0000 | 173.0000 | 173.0000 | 173.0000 | 173.0000 | 173.0000 |
| missing | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unique | 171.0000 | 172.0000 | 172.0000 | 173.0000 | 122.0000 | 167.0000 | 172.0000 | 173.0000 | 173.0000 | 173.0000 |
| mean | 0.0087 | -0.0119 | 0.0546 | -0.0066 | 0.1894 | 0.0732 | 0.2321 | 0.1039 | 0.7891 | 0.0236 |
| std | 0.0355 | 0.0527 | 0.0342 | 0.0248 | 0.1973 | 0.0365 | 0.1313 | 0.0488 | 0.1329 | 0.0148 |
| min | -0.0735 | -0.1566 | 0.0000 | -0.0858 | 0.0000 | 0.0000 | -0.2066 | 0.0253 | 0.3568 | -0.0283 |
| p01 | -0.0684 | -0.1376 | 0.0035 | -0.0602 | 0.0000 | 0.0000 | -0.0206 | 0.0388 | 0.4625 | -0.0110 |
| p05 | -0.0410 | -0.0968 | 0.0081 | -0.0476 | 0.0000 | 0.0129 | 0.0401 | 0.0471 | 0.5523 | 0.0046 |
| p25 | -0.0141 | -0.0485 | 0.0276 | -0.0226 | 0.0000 | 0.0505 | 0.1291 | 0.0695 | 0.7039 | 0.0134 |
| median | 0.0011 | -0.0102 | 0.0480 | -0.0041 | 0.1505 | 0.0693 | 0.2186 | 0.0939 | 0.8000 | 0.0224 |
| p75 | 0.0297 | 0.0294 | 0.0747 | 0.0075 | 0.2854 | 0.0935 | 0.3186 | 0.1266 | 0.9020 | 0.0341 |
| p95 | 0.0801 | 0.0733 | 0.1138 | 0.0309 | 0.5902 | 0.1438 | 0.4511 | 0.1913 | 0.9743 | 0.0496 |
| p99 | 0.1001 | 0.0953 | 0.1476 | 0.0596 | 0.7228 | 0.1576 | 0.4840 | 0.2460 | 0.9913 | 0.0590 |
| max | 0.1005 | 0.0979 | 0.1730 | 0.0751 | 1.0000 | 0.1721 | 0.6393 | 0.3836 | 1.0000 | 0.0691 |
| skew | 0.5197 | -0.1545 | 0.7530 | 0.0841 | 1.2295 | 0.3469 | 0.0944 | 1.7226 | -0.6250 | 0.1637 |

### July

| stat | break_open_return | d1_open_to_close_return_raw | d1_high_to_close_drawdown_raw | d1_close_to_vwap_raw | high_zone_volume_ratio | late_day_sell_volume_ratio | recent_7d_cumulative_return | recent_7d_max_drawdown | recent_7d_close_position | d1_ma10_slope |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 160.0000 | 160.0000 | 160.0000 | 160.0000 | 160.0000 | 160.0000 | 160.0000 | 160.0000 | 160.0000 | 160.0000 |
| missing | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unique | 148.0000 | 159.0000 | 159.0000 | 160.0000 | 120.0000 | 153.0000 | 160.0000 | 159.0000 | 159.0000 | 160.0000 |
| mean | 0.0041 | -0.0222 | 0.0621 | -0.0121 | 0.1957 | 0.0777 | 0.1737 | 0.1246 | 0.7414 | 0.0144 |
| std | 0.0441 | 0.0517 | 0.0346 | 0.0264 | 0.1840 | 0.0424 | 0.1089 | 0.0535 | 0.1328 | 0.0150 |
| min | -0.1005 | -0.1467 | 0.0000 | -0.0785 | 0.0000 | 0.0000 | -0.0851 | 0.0280 | 0.3797 | -0.0240 |
| p01 | -0.1000 | -0.1307 | 0.0025 | -0.0731 | 0.0000 | 0.0000 | -0.0484 | 0.0494 | 0.4339 | -0.0199 |
| p05 | -0.0699 | -0.1054 | 0.0147 | -0.0574 | 0.0000 | 0.0114 | 0.0152 | 0.0638 | 0.5143 | -0.0094 |
| p25 | -0.0227 | -0.0543 | 0.0380 | -0.0288 | 0.0000 | 0.0501 | 0.0967 | 0.0935 | 0.6493 | 0.0062 |
| median | 0.0000 | -0.0257 | 0.0537 | -0.0087 | 0.1677 | 0.0731 | 0.1638 | 0.1111 | 0.7570 | 0.0133 |
| p75 | 0.0286 | 0.0078 | 0.0792 | 0.0058 | 0.3025 | 0.1015 | 0.2384 | 0.1405 | 0.8468 | 0.0235 |
| p95 | 0.0888 | 0.0721 | 0.1280 | 0.0271 | 0.5448 | 0.1507 | 0.3799 | 0.2399 | 0.9252 | 0.0370 |
| p99 | 0.1001 | 0.0968 | 0.1457 | 0.0453 | 0.6947 | 0.1788 | 0.4801 | 0.2991 | 0.9843 | 0.0537 |
| max | 0.1003 | 0.1080 | 0.1500 | 0.0585 | 1.0000 | 0.2279 | 0.5175 | 0.4037 | 1.0000 | 0.0670 |
| skew | 0.1833 | 0.1693 | 0.6235 | -0.1942 | 1.1050 | 0.4959 | 0.5156 | 1.9345 | -0.4211 | 0.2449 |

### REGIME (binary)

| 切片 | count_0 | count_1 | rate_1 | minority_count |
|---|---|---|---|---|
| June | 145 | 28 | 0.1618 | 28 |
| July | 139 | 21 | 0.1313 | 21 |

## 3. primitive 相关性 (June 主参考; July 仅稳定性描述)

- 最大 June Pearson pair: `d1_high_to_close_drawdown_raw-recent_7d_close_position` rho=-0.8581 (severity=SEVERE)
- 最大 June Spearman pair: `d1_high_to_close_drawdown_raw-recent_7d_close_position` rho=-0.9102 (severity=SEVERE)
- 跨因子 SEVERE primitive pairs (|rho| >= 0.85): d1_high_to_close_drawdown_raw-recent_7d_close_position (0.9102)
- 因子内 SEVERE primitive pairs (同因子多观测, 设计允许): d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw (0.8554)

完整 55 对明细见 `v004c_primitive_dependence_v001.csv` (same_factor 标记 因子内/因子间)。

## 4. factor 构造验证与 June 分布

- 构造: 8 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8 (degenerate 检查通过: 无退化)
- REGIME 严格 0/1 (未标准化); 只有 condition number 诊断矩阵中临时 center/scale

| factor | membership | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPEN | M1,M2 | 0.0008 | 0.9979 | -2.1688 | -2.1679 | -1.3976 | -0.6401 | -0.2140 | 0.5923 | 2.0123 | 2.5740 | 2.5742 | 0.5323 |
| RESET | M1,M2 | -0.0000 | 1.0000 | -2.1400 | -1.8418 | -1.4115 | -0.7263 | -0.0897 | 0.6964 | 1.6843 | 2.2606 | 2.6678 | 0.3493 |
| SUPPLY | M1,M2 | 0.0000 | 1.0000 | -2.1815 | -2.1815 | -1.4240 | -0.7192 | -0.0507 | 0.6372 | 1.7539 | 2.1880 | 2.7343 | 0.2033 |
| MOM7 | M2 | 0.0028 | 0.9575 | -1.9245 | -1.8559 | -1.4619 | -0.7840 | -0.1029 | 0.6586 | 1.6682 | 1.9178 | 1.9183 | 0.1683 |
| DAMAGE7 | M2 | -0.0152 | 0.9225 | -1.3344 | -1.3333 | -1.1651 | -0.7065 | -0.2054 | 0.4637 | 1.7906 | 2.8835 | 2.9101 | 1.0261 |
| POS7 | M2 | 0.0069 | 0.9781 | -2.4572 | -2.3194 | -1.7812 | -0.6410 | 0.0819 | 0.8489 | 1.3932 | 1.5136 | 1.5208 | -0.5041 |
| TREND | M2 | 0.0026 | 0.9592 | -2.3403 | -2.2570 | -1.2854 | -0.6844 | -0.0811 | 0.7144 | 1.7608 | 2.3067 | 2.4017 | 0.2679 |
| REGIME | M2 | 0.1618 (rate1) | 0.3683 | 0 | - | - | - | - | - | - | - | 1 | - |

## 5. factor 相关性 (June 主参考; July 稳定性)

- 最大 June Pearson pair: `RESET-POS7` rho=-0.8681 (severity=SEVERE)
- 最大 June Spearman pair: `RESET-POS7` rho=-0.8980 (severity=SEVERE)
- severe pairs (|rho| >= 0.85): RESET-POS7 (0.8980)
- high pairs (0.70 <= |rho| < 0.85): MOM7-TREND (0.7770)

完整 28 对明细见 `v004c_factor_dependence_v001.csv` (REGIME 为二元变量, 其 Pearson/Spearman 只作描述性参考, 不作连续线性解释)。

## 6. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)

| factor | M1 VIF | M2 VIF |
|---|---|---|
| OPEN | 1.1394 | 1.9270 |
| RESET | 1.1456 | 8.0900 |
| SUPPLY | 1.0249 | 1.0601 |
| MOM7 | — | 5.5046 |
| DAMAGE7 | — | 1.7844 |
| POS7 | — | 8.3739 |
| TREND | — | 2.9205 |
| REGIME | — | 1.2018 |

- M1 max VIF: 1.146 (factor `RESET`); VIF >= 5: 无; VIF >= 10: 无
- M2 max VIF: 8.374 (factor `POS7`); VIF >= 5: RESET=8.090, MOM7=5.505, POS7=8.374; VIF >= 10: 无
- 解释: VIF < 5 OK; 5 <= VIF < 10 WATCH; VIF >= 10 SEVERE (只解释, 禁止自动删因子)

## 7. condition number (centered + unit-variance 诊断矩阵, SVD)

| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |
|---|---|---|---|---|---|
| M1 | OPEN RESET SUPPLY | 15.189 | 10.398920 | 1.461 | OK |
| M2 | 8 factors | 22.702 | 3.249196 | 6.987 | OK |
| PATH | MOM7 DAMAGE7 POS7 | 18.304 | 6.801733 | 2.691 | OK |
- 参考: kappa < 30 OK; 30-100 WATCH; >= 100 SEVERE; smallest singular ~ 0 => SEVERE / near singular
- REGIME 只在此诊断副本中 center/scale (diagnostic scaling != future model preprocessing)

## 8. July 无标签 X 稳定性 (June reference transform 映射)

| factor | June mean | June std | July mean | July std | SMD | KS | flag |
|---|---|---|---|---|---|---|---|
| OPEN | 0.0008 | 0.9979 | -0.0984 | 1.1799 | -0.0908 | 0.1246 | — |
| RESET | -0.0000 | 1.0000 | 0.2327 | 1.0146 | 0.2310 | 0.1230 | — |
| SUPPLY | 0.0000 | 1.0000 | 0.1000 | 1.0723 | 0.0965 | 0.0921 | — |
| MOM7 | 0.0028 | 0.9575 | -0.4396 | 0.8115 | -0.4984 | 0.2402 | — |
| DAMAGE7 | -0.0152 | 0.9225 | 0.3864 | 0.9541 | 0.4279 | 0.2804 | SHIFT_WATCH |
| POS7 | 0.0069 | 0.9781 | -0.3513 | 0.9800 | -0.3659 | 0.1757 | — |
| TREND | 0.0026 | 0.9592 | -0.6006 | 0.9507 | -0.6316 | 0.2688 | SHIFT_WATCH |
| REGIME | 0.1618 (rate1) | 0.3683 | 0.1313 (rate1) | 0.3377 | — | — | — |

- 判定: |SMD| >= 0.50 或 KS >= 0.25 => SHIFT_WATCH (只诊断, 不因 July X shift 自动换因子)
- REGIME: June rate1=0.1618 ((145, 28)), July rate1=0.1313 ((139, 21)), 差值=-0.0306
- STATIONARITY_WATCH 因子: `DAMAGE7, TREND` — 进入未来模型阶段后持续监控

## 9. 研究问题回答

**Q1. 11 个 primitive 中有没有严重重复?**
- 跨因子 SEVERE (|rho| >= 0.85): `d1_high_to_close_drawdown_raw-recent_7d_close_position` rho=0.9102
- 因子内 SEVERE (同一潜在因子的多个观测, 设计允许): `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw` rho=0.8554

**Q2. RESET 内部 3 个 primitive 是否合理描述同一潜变量?**
- `d1_open_to_close_return_raw-d1_high_to_close_drawdown_raw`: pearson=-0.8018 / spearman=-0.7840 (severity=HIGH)
- `d1_open_to_close_return_raw-d1_close_to_vwap_raw`: pearson=0.7448 / spearman=0.7620 (severity=HIGH)
- `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw`: pearson=-0.8261 / spearman=-0.8554 (severity=SEVERE)
- 结构说明: 三者的经济含义都锚定'D1 收盘相对盘中强度的重置程度' (OC 正=收盘强; HC 回撤大=收盘弱; VWAP gap 正=收盘强)。composite 对 OC 与 VWAP 取负、对 HC 取正, 使三者对齐同一方向; 内部相关性因此是设计的一部分 (同一潜在经济因子的多个观测)。

**Q3. SUPPLY 内部 2 个 primitive 是否合理描述同一潜变量?**
- `high_zone_volume_ratio-late_day_sell_volume_ratio`: pearson=-0.0649 / spearman=-0.0264 (severity=LOW)
- 结构说明: 高位成交占比与尾盘下跌 bar 成交占比都是 D1 供应压力来源的观测; composite 等权 (+1/2, +1/2) 固定, 不因相关性强弱调整。

**Q4. 8 个 factor 之间最大的 Pearson/Spearman pair 是什么?**
- 最大 Pearson: `RESET-POS7` rho=-0.8681
- 最大 Spearman: `RESET-POS7` rho=-0.8980

**Q5. MOM7/DAMAGE7/POS7 是否存在明显路径信息重复?**
- `MOM7-DAMAGE7`: pearson=-0.5838 / spearman=-0.5437 (severity=MODERATE)
- `MOM7-POS7`: pearson=0.5877 / spearman=0.5707 (severity=MODERATE)
- `DAMAGE7-POS7`: pearson=-0.2041 / spearman=-0.2277 (severity=LOW)

**Q6. MOM7 和 TREND 是否高度重复?**
- `MOM7-TREND`: pearson=0.7770 / spearman=0.7764 (severity=HIGH)

**Q7. RESET 和 DAMAGE7 是否实际上描述同一回撤?**
- `RESET-DAMAGE7`: pearson=0.0897 / spearman=0.1419 (severity=LOW)
- 注意: RESET 是 D1 单日价格重置强度, DAMAGE7 是最近 7 日路径峰值破坏, 时间尺度不同; 相关性高低只记录, 不做自动处理。

**Q8. M1 最大 VIF 是多少?**
- M1 max VIF = 1.146 (factor `RESET`)

**Q9. M2 最大 VIF 是多少?**
- M2 max VIF = 8.374 (factor `POS7`)

**Q10. M1 condition number 是多少?**
- M1 kappa = 1.461 (s_max=15.189, s_min=10.398920)

**Q11. M2 condition number 是多少?**
- M2 kappa = 6.987 (s_max=22.702, s_min=3.249196)

**Q12. PATH BLOCK condition number 是多少?**
- PATH BLOCK kappa = 2.691 (s_max=18.304, s_min=6.801733)

**Q13. 哪些 factor 出现 June→July X shift?**
- `DAMAGE7`: SMD=0.4279, KS=0.2804 (SHIFT_WATCH)
- `TREND`: SMD=-0.6316, KS=0.2688 (SHIFT_WATCH)
- REGIME: June rate1=0.1618, July rate1=0.1313, 差值=-0.0306

**Q14. 有没有 factor 退化或过度稀疏?**
- 退化 factor: 无 (8 个 factor 全部有限, 连续 factor June std > 1e-8)
- REGIME 稀疏性: June count0=145 count1=28 (rate1=0.1618); July count0=139 count1=21 (rate1=0.1313); minority 计数 28 / 21; 是否过疏由人工判断, 本阶段不处理

**Q15. 8 因子结构是否可以进入正式 walk-forward?**
- 最终状态: **REVIEW_REQUIRED**
- 触发原因: factor pair RESET-POS7 SEVERE (|rho|=0.898)
- 含义: 该状态只回答 'factor 结构是否适合进入 M0/M1/M2 expanding-date walk-forward', 不是 model coefficient freeze; 是否继续由人工审查 primitive/factor dependence、VIF、condition number、distribution stability 后决定。

## 10. 结构冲突说明 (仅记录, 不实施)

- 触发 REVIEW_REQUIRED 的 factor pair 说明如下; 本阶段不实施任何修正 (禁止 drop / swap / PCA / Lasso / 改权重)。

### RESET-POS7 (June Pearson=-0.8681 / Spearman=-0.8980, severity=SEVERE; July Pearson=-0.8475 / Spearman=-0.8355)

- 冲突: RESET (D1 单日价格重置强度) 与 POS7 (最近 7 日区间收盘相对位置) 高度负相关。
- 为什么: 两者共享同一 D1 收盘锚定信息 — RESET 高表示 D1 收盘相对开盘/最高价/VWAP 大幅向下重置, POS7 低表示收盘位于 7 日区间下部; 大幅向下重置的 D1 收盘 在机制上几乎必然落在 7 日区间底部, 两者近似互为逆观测。primitive 层面 `d1_high_to_close_drawdown_raw`-`recent_7d_close_position` 原始相关性已 SEVERE (Pearson=-0.8581 / Spearman=-0.9102)。该冲突为纯 X 结构发现, 不涉及 Target。
- 最小可考虑修正方向 (本阶段不实施):
  1) M2 membership 层面考虑不同时保留 RESET 与 POS7 (例如 POS7 移入 sensitivity 集), 不修改 factor 定义与 composite 权重;
  2) 或保留两者, 接受 M2 VIF WATCH (POS7 VIF=8.37, RESET VIF=8.09), 由下一阶段 walk-forward 的训练证据决定去留;
  3) 上述方向仅供人工参考, 本阶段禁止实施。

## 11. 最终状态

- 最终状态: **REVIEW_REQUIRED**
- 触发原因:
  - factor pair RESET-POS7 SEVERE (|rho|=0.898)
- 本审查只做 X 结构诊断: 未读取 Target, 未做任何 Target 方向解释 (不得据此声称某 factor 应取正/负系数), 未执行任何训练/预测
- 禁止自动修正: 即使出现 SEVERE / VIF >= 10 / kappa >= 100 / 退化, 本阶段也不 drop / swap / PCA / Lasso / 改变权重; 修正方向只记录给人工


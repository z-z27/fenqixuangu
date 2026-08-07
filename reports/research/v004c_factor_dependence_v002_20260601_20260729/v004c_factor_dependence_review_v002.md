# v004c 因子结构纯 X 侧审查 (v004c factor dependence v002)

- 阶段: 多变量建模前的纯 X 侧因子结构审查 v002 (X ONLY, 不训练模型, 不读取 Target)
- 分支: research-sample-analysis
- 输入: `v004c_model_table_v001.csv` SHA256: `c0de3bc639142c3e0b14b8018ff7ca13351ba46ed8f43cba3f54b18a261bd274`
- v002 修订依据 (v001 纯 X 审查结论, 与 Target 无关): SUPPLY composite 拆分 (HIGHZONE/LATESELL 独立); POS7 移入 SENSITIVITY (SENSITIVITY_STRUCTURAL_REDUNDANCY); TREND 移入 SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY)。v001 历史资产保持不变。

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
- 注: primitive 级跨因子 SEVERE 只作诊断; factor 级判定只针对 CORE 矩阵 (§5), 含 SENSITIVITY 成员的 pair 不触发状态 (§12)。

完整 55 对明细见 `v004c_primitive_dependence_v002.csv` (same_factor 标记 因子内/因子间)。

## 4. factor 构造验证与 June 分布 (9 factors: 7 CORE + 2 SENSITIVITY)

- 构造: 9 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8 (degenerate 检查通过: 无退化)
- REGIME 严格 0/1 (未标准化); 只有 condition number 诊断矩阵中临时 center/scale
- HIGHZONE / LATESELL 独立构造 (各自 z-score), 无 SUPPLY composite

| factor | membership | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPEN | M1,M2 | -0.0000 | 1.0000 | -2.1741 | -2.1732 | -1.4013 | -0.6423 | -0.2152 | 0.5928 | 2.0157 | 2.5785 | 2.5787 | 0.5323 |
| RESET | M1,M2 | 0.0000 | 1.0000 | -2.1411 | -1.8466 | -1.4110 | -0.7259 | -0.0921 | 0.6937 | 1.6833 | 2.2581 | 2.6676 | 0.3490 |
| HIGHZONE | M1,M2 | -0.0000 | 1.0000 | -0.9794 | -0.9794 | -0.9794 | -0.9794 | -0.1945 | 0.5095 | 2.0994 | 2.7872 | 2.7913 | 1.0485 |
| LATESELL | M1,M2 | -0.0000 | 1.0000 | -2.0145 | -2.0145 | -1.6577 | -0.6208 | -0.1037 | 0.5621 | 1.9502 | 2.3039 | 2.3306 | 0.3088 |
| MOM7 | M2 | -0.0000 | 1.0000 | -2.0129 | -1.9412 | -1.5297 | -0.8217 | -0.1104 | 0.6850 | 1.7395 | 2.0001 | 2.0007 | 0.1683 |
| DAMAGE7 | M2 | 0.0000 | 1.0000 | -1.4300 | -1.4288 | -1.2465 | -0.7494 | -0.2062 | 0.5191 | 1.9575 | 3.1422 | 3.1711 | 1.0261 |
| REGIME | M2 | 0.1618 (rate1) | 0.3683 | 0 | - | - | - | - | - | - | - | 1 | - |
| POS7 | SENSITIVITY | 0.0000 | 1.0000 | -2.5193 | -2.3784 | -1.8281 | -0.6624 | 0.0767 | 0.8609 | 1.4173 | 1.5404 | 1.5478 | -0.5041 |
| TREND | SENSITIVITY | 0.0000 | 1.0000 | -2.4425 | -2.3556 | -1.3428 | -0.7162 | -0.0872 | 0.7421 | 1.8330 | 2.4020 | 2.5011 | 0.2679 |

## 5. factor 相关性 (June 主参考; July 稳定性; 判定只看 CORE 矩阵)

- 最大 June Pearson pair (9 factor 全集): `RESET-POS7` rho=-0.8682 (severity=SEVERE) — 含 SENSITIVITY 成员
- 最大 June Spearman pair (9 factor 全集): `RESET-POS7` rho=-0.8988 (severity=SEVERE) — 含 SENSITIVITY 成员
- 最大 June Pearson pair (**CORE**): `MOM7-DAMAGE7` rho=-0.5838 (severity=MODERATE)
- 最大 June Spearman pair (**CORE**): `MOM7-DAMAGE7` rho=-0.5437 (severity=MODERATE)
- CORE severe pairs (|rho| >= 0.85): 无
- CORE high pairs (0.70 <= |rho| < 0.85): 无
- sensitivity 级 SEVERE pairs (非阻塞, 见 §12): RESET-POS7 (0.8988)

完整 36 对明细见 `v004c_factor_dependence_v002.csv` (core_pair 标记 CORE×CORE; REGIME 为二元变量, 其 Pearson/Spearman 只作描述性参考, 不作连续线性解释)。

## 6. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)

| factor | M1 VIF | M2 VIF |
|---|---|---|
| OPEN | 1.2075 | 1.4736 |
| RESET | 1.3370 | 1.7295 |
| HIGHZONE | 1.1762 | 1.2270 |
| LATESELL | 1.0316 | 1.0341 |
| MOM7 | — | 2.5879 |
| DAMAGE7 | — | 1.6214 |
| REGIME | — | 1.2084 |
| POS7 | — | — |
| TREND | — | — |

- M1 max VIF: 1.337 (factor `RESET`); VIF >= 5: 无; VIF >= 10: 无
- M2 max VIF: 2.588 (factor `MOM7`); VIF >= 5: 无; VIF >= 10: 无
- 解释: VIF < 5 OK; 5 <= VIF < 10 WATCH; VIF >= 10 SEVERE (只解释, 禁止自动删因子)
- v001 对照 (冻结 v001 报告, 只读): v001 M2 max VIF = 8.374 (factor `POS7`); POS7=8.374, RESET=8.090, TREND=2.921, MOM7=5.505。v002 M2 移除 POS7/TREND 后 max VIF = 2.588 (factor `MOM7`)。

## 7. condition number (centered + unit-variance 诊断矩阵, SVD)

| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |
|---|---|---|---|---|---|
| M1 | OPEN RESET HIGHZONE LATESELL | 15.731 | 9.002240 | 1.747 | OK |
| M2 | 7 CORE factors | 18.759 | 6.241049 | 3.006 | OK |
| PATH | MOM7 DAMAGE7 | 16.553 | 8.485274 | 1.951 | OK |
- 参考: kappa < 30 OK; 30-100 WATCH; >= 100 SEVERE; smallest singular ~ 0 => SEVERE / near singular
- REGIME 只在此诊断副本中 center/scale (diagnostic scaling != future model preprocessing)

## 8. July 无标签 X 稳定性 (June reference transform 映射)

| factor | June mean | June std | July mean | July std | SMD | KS | flag |
|---|---|---|---|---|---|---|---|
| OPEN | -0.0000 | 1.0000 | -0.0994 | 1.1824 | -0.0908 | 0.1246 | — |
| RESET | 0.0000 | 1.0000 | 0.2327 | 1.0151 | 0.2310 | 0.1226 | — |
| HIGHZONE | -0.0000 | 1.0000 | 0.0323 | 0.9258 | 0.0335 | 0.0690 | — |
| LATESELL | -0.0000 | 1.0000 | 0.1047 | 1.1109 | 0.0991 | 0.0774 | — |
| MOM7 | -0.0000 | 1.0000 | -0.4620 | 0.8476 | -0.4984 | 0.2402 | — |
| DAMAGE7 | 0.0000 | 1.0000 | 0.4353 | 1.0342 | 0.4279 | 0.2804 | SHIFT_WATCH |
| REGIME | 0.1618 (rate1) | 0.3683 | 0.1313 (rate1) | 0.3377 | — | — | — |
| POS7 | 0.0000 | 1.0000 | -0.3663 | 1.0019 | -0.3659 | 0.1757 | — |
| TREND | 0.0000 | 1.0000 | -0.6288 | 0.9911 | -0.6316 | 0.2688 | SHIFT_WATCH |

- 判定: |SMD| >= 0.50 或 KS >= 0.25 => SHIFT_WATCH (只诊断, 不因 July X shift 自动换因子)
- REGIME: June rate1=0.1618 ((145, 28)), July rate1=0.1313 ((139, 21)), 差值=-0.0306
- STATIONARITY_WATCH 因子: `DAMAGE7, TREND` — 进入未来模型阶段后持续监控

## 9. 研究问题回答

**Q1. 11 个 primitive 中有没有严重重复?**
- 跨因子 SEVERE (|rho| >= 0.85): `d1_high_to_close_drawdown_raw-recent_7d_close_position` rho=0.9102
- 因子内 SEVERE (同一潜在因子的多个观测, 设计允许): `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw` rho=0.8554
- 注: 跨因子 SEVERE 对应 factor 级 pair 为 RESET×POS7, POS7 已移入 SENSITIVITY (见 §5/§12), 不触发 CORE 判定。

**Q2. RESET 内部 3 个 primitive 是否合理描述同一潜变量?**
- `d1_open_to_close_return_raw-d1_high_to_close_drawdown_raw`: pearson=-0.8018 / spearman=-0.7840 (severity=HIGH)
- `d1_open_to_close_return_raw-d1_close_to_vwap_raw`: pearson=0.7448 / spearman=0.7620 (severity=HIGH)
- `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw`: pearson=-0.8261 / spearman=-0.8554 (severity=SEVERE)
- 结构说明: 三者的经济含义都锚定'D1 收盘相对盘中强度的重置程度' (OC 正=收盘强; HC 回撤大=收盘弱; VWAP gap 正=收盘强)。composite 对 OC 与 VWAP 取负、对 HC 取正, 使三者对齐同一方向; 内部相关性因此是设计的一部分 (同一潜在经济因子的多个观测)。

**Q3. HIGHZONE 与 LATESELL 是否接近独立 (SUPPLY 拆分依据)?**
- `high_zone_volume_ratio-late_day_sell_volume_ratio`: pearson=-0.0649 / spearman=-0.0264 (severity=LOW)
- 结构说明: 两者接近独立, 缺乏同一潜变量依据 => v002 不再压缩成 50/50 SUPPLY composite; HIGHZONE/LATESELL 各自独立 z-score, 未来由 Logistic 独立估计系数 (禁止重新合成 SUPPLY)。

**Q4. 9 个 factor 中最大 Pearson/Spearman pair 是什么 (全集 与 CORE)?**
- 全集最大 Pearson: `RESET-POS7` rho=-0.8682 (severity=SEVERE) — 含 SENSITIVITY 成员
- 全集最大 Spearman: `RESET-POS7` rho=-0.8988 (severity=SEVERE) — 含 SENSITIVITY 成员
- CORE 最大 Pearson: `MOM7-DAMAGE7` rho=-0.5838 (severity=MODERATE)
- CORE 最大 Spearman: `MOM7-DAMAGE7` rho=-0.5437 (severity=MODERATE)

**Q5. MOM7/DAMAGE7/POS7 是否存在明显路径信息重复?**
- `MOM7-DAMAGE7`: pearson=-0.5838 / spearman=-0.5437 (severity=MODERATE)
- `MOM7-POS7`: pearson=0.5877 / spearman=0.5707 (severity=MODERATE) (POS7 为 SENSITIVITY)
- `DAMAGE7-POS7`: pearson=-0.2041 / spearman=-0.2277 (severity=LOW) (POS7 为 SENSITIVITY)

**Q6. MOM7 和 TREND 是否高度重复?**
- `MOM7-TREND`: pearson=0.7770 / spearman=0.7764 (severity=HIGH)
- v001 对照: June Pearson=0.7770 / Spearman=0.7764 — 该关联是 TREND 移入 SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY) 的依据之一。

**Q7. RESET 和 DAMAGE7 是否实际上描述同一回撤?**
- `RESET-DAMAGE7`: pearson=0.0897 / spearman=0.1431 (severity=LOW)
- 注意: RESET 是 D1 单日价格重置强度, DAMAGE7 是最近 7 日路径峰值破坏, 时间尺度不同; 相关性高低只记录, 不做自动处理。

**Q8. M1 最大 VIF 是多少?**
- M1 max VIF = 1.337 (factor `RESET`)

**Q9. M2 最大 VIF 是多少?**
- M2 max VIF = 2.588 (factor `MOM7`)
- v001 对照: v001 M2 max VIF = 8.374 (factor `POS7`); POS7=8.374, TREND=2.921, RESET=8.090, MOM7=5.505。v002 移除 POS7/TREND 后 max VIF = 2.588 (factor `MOM7`)。

**Q10. M1 condition number 是多少?**
- M1 kappa = 1.747 (s_max=15.731, s_min=9.002240)

**Q11. M2 condition number 是多少?**
- M2 kappa = 3.006 (s_max=18.759, s_min=6.241049)

**Q12. PATH BLOCK condition number 是多少?**
- PATH BLOCK kappa = 1.951 (s_max=16.553, s_min=8.485274)

**Q13. 哪些 factor 出现 June→July X shift?**
- `DAMAGE7`: SMD=0.4279, KS=0.2804 (SHIFT_WATCH)
- `TREND`: SMD=-0.6316, KS=0.2688 (SHIFT_WATCH) — 该 shift 是 TREND 移入 SENSITIVITY 的依据之一
- REGIME: June rate1=0.1618, July rate1=0.1313, 差值=-0.0306

**Q14. 有没有 factor 退化或过度稀疏?**
- 退化 factor: 无 (9 个 factor 全部有限, 连续 factor June std > 1e-8)
- REGIME 稀疏性: June count0=145 count1=28 (rate1=0.1618); July count0=139 count1=21 (rate1=0.1313); minority 计数 28 / 21; 是否过疏由人工判断, 本阶段不处理

**Q15. v002 CORE 结构是否可以进入正式 walk-forward?**
- 最终状态: **PASS_X_STRUCTURE_V002**
- 含义: 该状态只回答 'CORE factor 结构是否适合进入 M0/M1/M2 expanding-date walk-forward', 不是 model coefficient freeze; 是否继续由人工审查 primitive/factor dependence、VIF、condition number、distribution stability 后决定。

## 10. v002 修订要点确认 (v001 → v002 结构变化的效果, 纯 X 数据)

**1. RESET 从 POS7 移出 core 后, VIF 是否明显改善?**
- v002 M2 中 RESET VIF = 1.730; POS7 已不在任何 CORE 矩阵。
- v001 对照: v001 M2 中 RESET VIF = 8.090, POS7 VIF = 8.374。

**2. MOM7 移除 TREND 后, 是否不再出现明显共线性?**
- v002 CORE 中 MOM7 最大相关 pair = `MOM7-DAMAGE7`: pearson=-0.5838 / spearman=-0.5437 (severity=MODERATE)。
- v002 M2 中 MOM7 VIF = 2.588。
- v001 对照: v001 `MOM7-TREND` pearson=0.7770 / spearman=0.7764 (severity=HIGH); TREND 现已移出 CORE。

**3. HIGHZONE 与 LATESELL 独立进入 M1/M2 后, 矩阵是否仍健康?**
- `HIGHZONE-LATESELL`: pearson=-0.0513 / spearman=-0.0263 (severity=LOW) — 接近独立, 支持拆分。
- v002 M1/M2 中 HIGHZONE VIF = 1.176 / 1.227, LATESELL VIF = 1.032 / 1.034。

**4. M1 是否存在严重共线性?**
- M1 max VIF = 1.337 (factor `RESET`); M1 kappa = 1.747 (s_max=15.731, s_min=9.002240)。

**5. M2 是否存在严重共线性?**
- M2 max VIF = 2.588 (factor `MOM7`); M2 kappa = 3.006 (s_max=18.759, s_min=6.241049)。

**6. 是否仍有 cross-factor SEVERE pair?**
- CORE: 无 (7 个 CORE factor 内部无 |rho| >= 0.85 pair)。
- SENSITIVITY (非阻塞, 只记录): `RESET-POS7` (|rho|=0.8988)

## 11. 结构冲突说明 (仅记录, 不实施)

- CORE M1/M2 无 cross-factor SEVERE pair (|rho| >= 0.85), 无需冲突说明。

- 含 SENSITIVITY 成员的 SEVERE pair (如 RESET-POS7) 按 v002 决策记录为 非阻塞, 见 §12。

## 12. 非阻塞 SENSITIVITY 记录 (v002 决策, 只记录不实施)

- `RESET-POS7` (June Pearson=-0.8682 / Spearman=-0.8988; July Pearson=-0.8474 / Spearman=-0.8352): `POS7` 已按 v002 决策移入 SENSITIVITY (状态 `SENSITIVITY_STRUCTURAL_REDUNDANCY`), 该结构冗余只记录, 不阻塞 CORE 矩阵健康判定。
- `MOM7-TREND` (June Pearson=0.7770 / Spearman=0.7764, severity=HIGH): TREND 已移入 SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY), 该关联不再属于 CORE 矩阵。

## 13. 最终状态

- 最终状态: **PASS_X_STRUCTURE_V002**
- 判定阈值 (任务 §12): CORE M1/M2 cross-factor |Pearson or Spearman| >= 0.85, 或 VIF >= 10, 或 condition number >= 100, 或 degenerate factor => REVIEW_REQUIRED; 否则 PASS_X_STRUCTURE_V002。VIF 5-10 或相关 0.70-0.85 只记 WATCH, 不自动失败。
- 本审查只做 X 结构诊断: 未读取 Target, 未做任何 Target 方向解释 (不得据此声称某 factor 应取正/负系数), 未执行任何训练/预测
- 禁止自动修正: 即使出现 SEVERE / VIF >= 10 / kappa >= 100 / 退化, 本阶段也不 drop / swap / PCA / Lasso / 改变权重; 修正方向只记录给人工


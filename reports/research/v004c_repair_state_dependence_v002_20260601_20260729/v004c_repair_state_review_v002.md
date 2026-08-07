# v004c Repair-State Model v002 — 语义修订 + 纯 X 结构审计

- 阶段: Repair-State v002 语义修订 + 纯 X 结构审计 (X ONLY, 不训练模型, 不读取 Target)
- 分支: research-sample-analysis
- 输入: `v004c_model_table_v001.csv` SHA256: `c0de3bc639142c3e0b14b8018ff7ca13351ba46ed8f43cba3f54b18a261bd274`
- 背景: Repair-State v001 纯 X 审计通过 (PASS_REPAIR_STATE_X_V001), 但人工 复核发现 v001 DIVERGENCE 沿用旧 RESET ((-z_OC+z_HC-z_VWAP)/3), 主要描述 'D1 最终收盘留下多少价格损伤' 而非 '盘中发生过多大的第一次分歧', 且  Corr(DIVERGENCE, RECLAIM) June Pearson = -0.7340 / Spearman = -0.7725  存在明显机械反向关系; v002 把 '盘中分歧强度' 与 '最终价格损伤' 拆成 两个独立 factor (DIVERGENCE / CLOSE_DAMAGE)。

- 修订内容 (v001 -> v002): DIVERGENCE 改为 z(d1_intraday_range) (单 primitive, 过程强度); 旧 RESET 数学整体移入 CLOSE_DAMAGE (公式零复制, 语义改为最终损伤); 新增 derived DIVERGENCE_X_DAMAGE (替代  DIVERGENCE_X_SUPPLY); DIVERGENCE_X_SUPPLY 降为 SENSITIVITY; 预声明 TURNOVER_COST sensitivity。

## 0. 方法 (固定, 不搜索)

- 基础 factor 只有 5 个: OPEN / DIVERGENCE / CLOSE_DAMAGE / SUPPLY / RECLAIM (把 v001 的'分歧+损伤'混合因子拆成两个独立状态)
- DIVERGENCE 只来自 `d1_intraday_range` 单 primitive (D1 盘中 high-low range / prev_close, 过程强度; 禁止使用 open_to_close / high_to_close / close_to_vwap 直接定义 DIVERGENCE)
- CLOSE_DAMAGE 与旧 RESET 数学定义完全一致 (权重 -1/3, +1/3, -1/3 固定, 禁止修改); 公式函数直接复用 `v004c_factor_spec.composite_reset`, 逐行误差只允许浮点容差
- RECLAIM 公式与 v001 完全一致 (权重冻结, 不得改权重, 不得加第三个 primitive); 复用 v001 `composite_reclaim`
- SUPPLY 只使用 `late_day_sell_volume_ratio` 单独构造 (不重新使用旧的 HIGHZONE + LATESELL composite)
- 预声明 derived terms 只有 3 个: DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM / DIVERGENCE_X_DAMAGE (只做 mean/std 标准化, 不做第二次 clip); 禁止自动搜索 interaction
- X loader 使用显式 `usecols` 只加载 3 个标识列 (event_id, code, signal_date) + 8 个 core primitive (+ 1 个 sensitivity primitive break_volume_ratio_vs_board_days); `target7_daily_d2open_d3high` / D2 / D3 / future return / 旧模型输出从未被加载 (读取入口即 target-blind)
- June = **development / reference X sample** (2026-06-01 ~ 2026-06-30, 用于拟合 primitive q01/q99/mu/sigma + composite + derived mean/std)
- July = **retrospective unlabeled X-stability slice** (2026-07-01 ~ 2026-07-29, 只 apply June reference 参数; 不得用 July 重算 mean/std, 不得读取 July Target)
- transform 顺序: primitive 先 FOLD_CLIP_Z (reference 原始有限值 -> q01/q99 -> clip -> clipped mean/std -> z); composite 再 z(G); derived 最后 z(raw)。July 全程只 apply。

## 1. 样本规模

| 切片 | rows | signal dates | 日期范围 |
|---|---|---|---|
| June (development / reference X sample) | 173 | 21 | 2026-06-01 ~ 2026-06-30 |
| July (retrospective unlabeled X-stability slice) | 160 | 21 | 2026-07-01 ~ 2026-07-29 |

- 窗口外行: 0 (全部行都在 June/July 窗口内)

## 2. DIVERGENCE primitive 核验 (d1_intraday_range)

| 项目 | 内容 |
|---|---|
| source file (定义文档) | `reports/research/v004c_dataset_20260506_20260729/v004c_feature_definitions.md` (D1 振幅, 行 113: `(high - low) / prev_close`) |
| source file (生成代码) | `reports/research/v004c_dataset_20260506_20260729/_scratch/build_v004c_dataset.py:684` |
| source function / definition | `r[\"d1_intraday_range\"] = _safe_div(h_c - l_c, prev_close_d1)`, 其中 `h_c`/`l_c` = D1 日内 5min high 最大值 / low 最小值, `prev_close_d1` = 断板日前收盘 (最后板日收盘) |
| exact formula | `(d1_high - d1_low) / prev_close` (D1 盘中 high-low price range / amplitude, 以前收盘归一化) |
| model table schema | `v004c_model_table_schema_v001.csv`: D1_CLOSE / D1_PRICE_ACTION / FOLD_CLIP_Z / PRIMARY |
| uses only D1 data | **YES** (只使用 D1 日内 high/low 与 D1 开盘前已知的 prev_close; 不依赖 D2/D3, 不依赖 Target) |
| semantic match | **YES** (仅描述 D1 盘中价格探索范围, 与收盘位置无关; 价格重新收回来也不会自动消除分歧记录) |
- 结论: 满足 §4 语义要求, 直接复用 `d1_intraday_range` 作为 DIVERGENCE 的 唯一 primitive。不构造新公式。

## 3. primitive 相关性 (June 主参考; July 仅稳定性描述)

- 最大 June Pearson pair: `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw` rho=-0.8261 (severity=SEVERE)
- 最大 June Spearman pair: `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw` rho=-0.8554 (severity=SEVERE)
- 完整 28 对明细见 `v004c_repair_state_primitive_dependence_v002.csv` (reclaim_internal 标记 RECLAIM 内部对)。

## 4. RECLAIM 内部结构审计 (d1_low_to_close_recovery vs d1_afternoon_return)

| 切片 | Pearson | Spearman |
|---|---|---|
| June | 0.5787 | 0.5414 |
| July | 0.4409 | 0.4306 |

- v001 冻结依据 (2026-06 审计): June 0.5787 / 0.5414; July 0.4409 / 0.4306 (COMPOSITE_SUPPORTED)。
- composite status: **COMPOSITE_SUPPORTED**
- June |Pearson| / |Spearman| 至少一项 >= 0.10: 存在同一潜变量的可辨 证据, composite 构造合理 (COMPOSITE_SUPPORTED)。

## 5. factor 构造与 June 分布 (8 factors)

- 构造: 8 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8 (degenerate 检查: 无退化)
- DIVERGENCE = z(d1_intraday_range) (单 primitive); CLOSE_DAMAGE = z(G_C), G_C = (-z_OC + z_HC - z_VWAP)/3 (与旧 RESET 数学一致); RECLAIM = z(G_Q) (与 v001 一致)
- derived terms: raw = F_V^2 / F_V*F_Q / F_V*F_C, 只做 mean/std 标准化, 未施加第二次 clip

| factor | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPEN | -0.0000 | 1.0000 | -2.1741 | -2.1732 | -1.4013 | -0.6423 | -0.2152 | 0.5928 | 2.0157 | 2.5785 | 2.5787 | 0.5323 |
| DIVERGENCE | 0.0000 | 1.0000 | -1.8575 | -1.8478 | -1.4190 | -0.6701 | -0.1544 | 0.6542 | 1.8438 | 2.8659 | 2.8713 | 0.6405 |
| CLOSE_DAMAGE | 0.0000 | 1.0000 | -2.1411 | -1.8466 | -1.4110 | -0.7259 | -0.0921 | 0.6937 | 1.6833 | 2.2581 | 2.6676 | 0.3490 |
| SUPPLY | -0.0000 | 1.0000 | -2.0145 | -2.0145 | -1.6577 | -0.6208 | -0.1037 | 0.5621 | 1.9502 | 2.3039 | 2.3306 | 0.3088 |
| RECLAIM | 0.0000 | 1.0000 | -2.1773 | -1.9649 | -1.5261 | -0.6864 | -0.1069 | 0.5853 | 1.6802 | 3.2483 | 3.5383 | 0.7092 |
| DIVERGENCE_SQ | 0.0000 | 1.0000 | -0.6680 | -0.6675 | -0.6645 | -0.5965 | -0.3681 | 0.2584 | 1.7035 | 4.8187 | 4.8396 | 2.8305 |
| DIVERGENCE_X_RECLAIM | 0.0000 | 1.0000 | -4.5572 | -2.1011 | -0.8960 | -0.3190 | -0.1133 | 0.1658 | 0.9553 | 4.7257 | 6.4472 | 2.6047 |
| DIVERGENCE_X_DAMAGE | -0.0000 | 1.0000 | -3.7975 | -2.1785 | -0.9512 | -0.3806 | -0.1552 | 0.2463 | 1.3983 | 4.4429 | 5.6205 | 1.9468 |

## 6. 语义解耦检查 (§27-29: DIVERGENCE / CLOSE_DAMAGE / RECLAIM)

| pair | 切片 | Pearson | Spearman |
|---|---|---|---|
| DIVERGENCE vs RECLAIM | June | 0.2141 | 0.1371 |
|  | July | 0.1892 | 0.1094 |
| DIVERGENCE vs CLOSE_DAMAGE | June | 0.3299 | 0.2745 |
|  | July | 0.3442 | 0.2816 |
| CLOSE_DAMAGE vs RECLAIM | June | -0.7340 | -0.7725 |
|  | July | -0.7195 | -0.7628 |

- v001 参考: old DIVERGENCE vs RECLAIM June Pearson = -0.7340 / Spearman = -0.7725 (v002 的核心修订目标: 新 DIVERGENCE 必须不再是 RECLAIM 的机械反面)
- DIVERGENCE vs RECLAIM 判定门 (§27): June max|corr| = 0.2141 (Pearson 0.2141 / Spearman 0.1371) => **SEMANTIC_DECOUPLING_OK**
- 判据: |corr| >= 0.70 => SEMANTIC_DECOUPLING_REVIEW (人工语义阻塞); 0.50 <= |corr| < 0.70 => SEMANTIC_DECOUPLING_WATCH (只记录); < 0.50 => SEMANTIC_DECOUPLING_OK
- DIVERGENCE vs CLOSE_DAMAGE / CLOSE_DAMAGE vs RECLAIM 允许相关 (§28/§29, 不设 0.70 硬失败): 分歧越剧烈通常最终损伤可能越大; 回收越强收盘损伤通常 越小。是否构成问题由 VIF / condition number 整体判断。

## 7. factor 相关性 (June 主参考; July 稳定性; 三层分类)

- 最大 June Pearson pair: `CLOSE_DAMAGE-RECLAIM` rho=-0.7340 (severity=HIGH)
- 最大 June Spearman pair: `CLOSE_DAMAGE-RECLAIM` rho=-0.7725 (severity=HIGH)
- 完整 28 对明细见 `v004c_repair_state_factor_dependence_v002.csv` (classification 列 = EXPECTED_HIERARCHICAL_DEPENDENCE / SEMANTIC_STATE_DEPENDENCE / UNEXPECTED_REDUNDANCY)。

### 层级项分类 (EXPECTED_HIERARCHICAL_DEPENDENCE)

- 天然层级依赖 pair (derived 与其父 factor, 高相关属预期):
  - `DIVERGENCE-DIVERGENCE_SQ`: June Pearson=0.4241 / Spearman=-0.0073 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `DIVERGENCE-DIVERGENCE_X_RECLAIM`: June Pearson=0.1358 / Spearman=-0.1160 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `DIVERGENCE-DIVERGENCE_X_DAMAGE`: June Pearson=0.1867 / Spearman=-0.0621 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `CLOSE_DAMAGE-DIVERGENCE_X_DAMAGE`: June Pearson=0.5411 / Spearman=0.2312 (severity=MODERATE) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `RECLAIM-DIVERGENCE_X_RECLAIM`: June Pearson=0.5707 / Spearman=0.2097 (severity=MODERATE) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)

### 状态因子相关对 (SEMANTIC_STATE_DEPENDENCE, §28/§29 允许)

  - `CLOSE_DAMAGE-RECLAIM`: June Pearson=-0.7340 / Spearman=-0.7725 (July Pearson=-0.7195 / Spearman=-0.7628) — SEMANTIC_STATE_DEPENDENCE (只记录, 不机械删变量; 由 VIF/kappa 整体判断)

- 非层级 SEVERE pairs: 无 (所有高相关 pair 都属层级依赖或状态相关)。

## 8. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)

| factor | R1 VIF | R2 VIF |
|---|---|---|
| OPEN | 1.181 | 1.365 |
| DIVERGENCE | 2.326 | 2.739 |
| CLOSE_DAMAGE | 5.043 | 6.343 |
| SUPPLY | 1.111 | 1.197 |
| RECLAIM | 4.765 | 6.620 |
| DIVERGENCE_SQ | — | 4.424 |
| DIVERGENCE_X_RECLAIM | — | 8.096 |
| DIVERGENCE_X_DAMAGE | — | 9.088 |

- R1 max VIF: 5.043 (factor `CLOSE_DAMAGE`); VIF >= 5: CLOSE_DAMAGE=5.043; VIF >= 10: 无
- R2 max VIF: 9.088 (factor `DIVERGENCE_X_DAMAGE`); VIF >= 5: CLOSE_DAMAGE=6.343, RECLAIM=6.620, DIVERGENCE_X_RECLAIM=8.096, DIVERGENCE_X_DAMAGE=9.088; VIF >= 10: 无
- 解释: VIF < 5 OK; 5 <= VIF < 10 WATCH; VIF >= 10 SEVERE (只解释, 禁止自动删因子; VIF/condition number 必须真实报告, 不因层级项人为放宽)

## 9. condition number (centered + unit-variance 诊断矩阵, SVD)

| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |
|---|---|---|---|---|---|
| R1 | OPEN DIVERGENCE CLOSE_DAMAGE SUPPLY RECLAIM | 18.435 | 4.037508 | 4.566 | OK |
| R2 | 8 factors (含 3 个层级项) | 22.163 | 2.652936 | 8.354 | OK |
| STATE_BLOCK | DIVERGENCE CLOSE_DAMAGE RECLAIM SUPPLY | 17.893 | 4.073469 | 4.392 | OK |
| CONDITIONAL_BLOCK | DIVERGENCE_SQ X_RECLAIM X_DAMAGE | 17.100 | 3.559970 | 4.803 | OK |
- 参考: kappa < 30 OK; 30-100 WATCH; >= 100 SEVERE; smallest singular ~ 0 => SEVERE / near singular

## 10. DIVERGENCE 分布与非线性支持检查 (§35, 只用 X 数学, 不使用 Target)

| 统计 | 值 |
|---|---|
| min | -1.8575 |
| p05 | -1.4190 |
| p25 | -0.6701 |
| median | -0.1544 |
| p75 | 0.6542 |
| p95 | 1.8438 |
| max | 2.8713 |
| p95 - p05 | 3.2628 |

- 非线性支持: **NONLINEARITY_SUPPORT_OK**
- DIVERGENCE June p95-p05 = 3.2628 >= 0.5, 覆盖 low/middle/high/extreme 区间, 平方项 (DIVERGENCE_SQ) 有表达空间 (只做 X 数学检查)。

## 11. July 无标签 X 稳定性 (June reference transform 映射)

| factor | June mean | June std | July mean | July std | SMD | KS | flag |
|---|---|---|---|---|---|---|---|
| OPEN | -0.0000 | 1.0000 | -0.0994 | 1.1824 | -0.0908 | 0.1246 | — |
| DIVERGENCE | 0.0000 | 1.0000 | 0.1488 | 1.0246 | 0.1470 | 0.1190 | — |
| CLOSE_DAMAGE | 0.0000 | 1.0000 | 0.2327 | 1.0151 | 0.2310 | 0.1226 | — |
| SUPPLY | -0.0000 | 1.0000 | 0.1047 | 1.1109 | 0.0991 | 0.0774 | — |
| RECLAIM | 0.0000 | 1.0000 | -0.2235 | 0.9769 | -0.2261 | 0.1367 | — |
| DIVERGENCE_SQ | 0.0000 | 1.0000 | 0.0480 | 0.8601 | 0.0515 | 0.1238 | — |
| DIVERGENCE_X_RECLAIM | 0.0000 | 1.0000 | -0.0436 | 0.9065 | -0.0457 | 0.0855 | — |
| DIVERGENCE_X_DAMAGE | -0.0000 | 1.0000 | 0.0484 | 1.0360 | 0.0475 | 0.0916 | — |

- 判定: |SMD| >= 0.50 或 KS >= 0.25 => SHIFT_WATCH (只记录, 禁止自动换因子)
- SHIFT_WATCH 因子: 无

## 12. derived tail 诊断 (interaction 是否被少量极端值主导)

| factor | 切片 | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | mean | std | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DIVERGENCE_SQ | june | -0.6680 | -0.6675 | -0.6645 | -0.5965 | -0.3681 | 0.2584 | 1.7035 | 4.8187 | 4.8396 | 0.0000 | 1.0000 | 2.8305 |
| DIVERGENCE_SQ | july | -0.6680 | -0.6676 | -0.6610 | -0.5824 | -0.2770 | 0.3139 | 1.8479 | 3.0226 | 3.7783 | 0.0480 | 0.8601 | 1.8641 |
| DIVERGENCE_X_RECLAIM | june | -4.5572 | -2.1011 | -0.8960 | -0.3190 | -0.1133 | 0.1658 | 0.9553 | 4.7257 | 6.4472 | 0.0000 | 1.0000 | 2.6047 |
| DIVERGENCE_X_RECLAIM | july | -2.7303 | -2.0472 | -1.3600 | -0.3421 | -0.1130 | 0.3102 | 1.1409 | 3.3617 | 4.7130 | -0.0436 | 0.9065 | 1.3992 |
| DIVERGENCE_X_DAMAGE | june | -3.7975 | -2.1785 | -0.9512 | -0.3806 | -0.1552 | 0.2463 | 1.3983 | 4.4429 | 5.6205 | -0.0000 | 1.0000 | 1.9468 |
| DIVERGENCE_X_DAMAGE | july | -2.1392 | -2.0295 | -1.2361 | -0.4149 | -0.2029 | 0.2989 | 2.5724 | 3.3336 | 3.4680 | 0.0484 | 1.0360 | 1.3765 |

- 判定: June 标准化后 max|min| / max|max| >= 5.0 => DERIVED_TAIL_WATCH (只记录; derived 项不做第二次 clip, 不 winsorize)
- DERIVED_TAIL_WATCH 因子: `DIVERGENCE_X_RECLAIM, DIVERGENCE_X_DAMAGE`

## 13. Hierarchy 检查确认

- R2 成员包含每个 derived term 的全部父 factor: PASS
- 禁止新增 interaction (OPEN_X_DIVERGENCE / DIV_X_TURNOVER / RECLAIM_SQ / SUPPLY_SQ / CLOSE_DAMAGE_SQ): 不属于 v002 模型, 未构造
- SENSITIVITY (HIGHZONE, MOM7, DAMAGE7, REGIME, POS7, TREND, DIVERGENCE_X_SUPPLY, TURNOVER_COST) 全部不在 R1/R2 core: PASS

## 14. Sensitivity 说明

- DIVERGENCE_X_SUPPLY (v001 derived): 定义保留在 v001/v002 spec, v002 降为 SENSITIVITY (当前优先验证'分歧过度''分歧后的回收''分歧是否留下真实损伤' 三个核心机制, 避免 interaction 继续膨胀)
- TURNOVER_COST = z(break_volume_ratio_vs_board_days): 字段定义核验通过 (`break_volume / mean(断板前 streak 个板日 volume)`, 见  `v004c_feature_definitions.md` 与 model table schema); 只预声明, 本版本禁止构造 DIV_X_TURNOVER, 禁止进入 R1/R2
- `break_volume_ratio_vs_board_days` (sensitivity 源字段) June: n=173 missing=0 unique=173 min=0.7672 median=1.7884 max=32.9414; July: n=160 missing=0 unique=160 min=0.0852 median=1.9486 max=13.0976
- LEGACY_SENSITIVITY (HIGHZONE, MOM7, DAMAGE7, REGIME, POS7, TREND): 全部不在 R1/R2; 历史定义保留, 不删除

## 15. 最终状态

- 最终状态: **PASS_REPAIR_STATE_X_V002**
- 判定门 (任务 §37): DIVERGENCE primitive mismatch => BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH; DIVERGENCE vs RECLAIM June |Pearson| >= 0.70 或 |Spearman| >= 0.70 => SEMANTIC_DECOUPLING_REVIEW; degenerate factor / VIF >= 10 / condition number >= 100 => REVIEW_REQUIRED; 否则 PASS_REPAIR_STATE_X_V002。
- 非阻塞记录 (不进入判定门): SEMANTIC_DECOUPLING_WATCH = 无; SHIFT_WATCH = 无; DERIVED_TAIL_WATCH = `DIVERGENCE_X_RECLAIM, DIVERGENCE_X_DAMAGE`; DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW = 否 (VIF 5-10 / kappa 30-100 同样只记录)
- 本审查只做 X 结构 / 数值稳定性 / June→July X 稳定性诊断: 未读取 Target, 未执行任何训练/预测/指标, 未用 Target7 rate / AUC / LogLoss / Brier / coefficient / Top3 参与判定
- 禁止自动修正: 即使出现 SEMANTIC_DECOUPLING_REVIEW / SEVERE / VIF >= 10 / kappa >= 100 / 退化, 本阶段也不 drop / swap / PCA / Lasso / 改权重 / 加 interaction; 修正方向只记录给人工

## 16. 结论与后续流程

- 本阶段只回答: 新 DIVERGENCE 是否与 RECLAIM 语义解耦、R1/R2 的 X 结构 是否健康、CLOSE_DAMAGE 是否仍等于旧 RESET 数学、层级项是否数值健康、 June→July X 是否发生明显漂移。
- 下一阶段必须人工审查: 语义解耦结果 / DIVERGENCE 构造合理性 / interaction 是否数值健康 / R1/R2 是否存在严重共线性 / June→July X 是否明显漂移; 只有人工批准后才能正式冻结 Repair-State v002, 进入 June development fit + frozen retrospective OOT。
- post-June hypothesis: 本模型是 June walk-forward 失败后形成的假设, June 不能再作为独立验证集; 后续流程 = X spec freeze -> June development fit -> freeze coefficients/spec -> July retrospective OOT -> August+ forward shadow (July 打开以后, 禁止再根据 July 结果修改 Repair-State v002)。


# v004c Repair-State Model v001 — 规格 + 纯 X 结构审计

- 阶段: Repair-State v001 规格冻结 + 纯 X 结构审计 (X ONLY, 不训练模型, 不读取 Target)
- 分支: research-sample-analysis
- 输入: `v004c_model_table_v001.csv` SHA256: `c0de3bc639142c3e0b14b8018ff7ca13351ba46ed8f43cba3f54b18a261bd274`
- 背景: 上一版 M0/M1/M2 June expanding-date walk-forward 结论 REJECT_NO_STABLE_MODEL; 本版把人工逐样本分析形成的新交易假设正式写成数学规格, 并做严格 Target-blind 的纯 X 审计 (post-June hypothesis: June 不能再作为 该模型的独立验证集)。

## 0. 方法 (固定, 不搜索)

- 基础 factor 只有 4 个: OPEN / DIVERGENCE / SUPPLY / RECLAIM (上一版 OPEN + RESET + HIGHZONE + LATESELL 无法区分'跌得很多但是没人接'与 '充分分歧以后重新出现承接'; 本版显式加入 RECLAIM 与条件项)
- DIVERGENCE 是旧 RESET 的重新命名和重新解释, 数学定义完全一致 (权重 -1/3, +1/3, -1/3 固定, 禁止修改); 公式函数直接复用 `v004c_factor_spec.composite_reset`
- SUPPLY 只使用 `late_day_sell_volume_ratio` 单独构造 (不重新使用旧的 HIGHZONE + LATESELL composite)
- 预声明 derived terms 只有 3 个: DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM / DIVERGENCE_X_SUPPLY (只做 mean/std 标准化, 不做第二次 clip); 禁止自动搜索 interaction
- X loader 使用显式 `usecols` 只加载 3 个标识列 (event_id, code, signal_date) + 7 个 primitive 列; `target7_daily_d2open_d3high` / D2 / D3 / future return / 旧模型输出从未被加载 (读取入口即 target-blind)
- June = **development / reference X sample** (2026-06-01 ~ 2026-06-30, 用于拟合 primitive q01/q99/mu/sigma + composite + derived mean/std)
- July = **retrospective unlabeled X-stability slice** (2026-07-01 ~ 2026-07-29, 只 apply June reference 参数; 不得用 July 重算 mean/std, 不得读取 July Target)
- transform 顺序: primitive 先 FOLD_CLIP_Z (reference 原始有限值 -> q01/q99 -> clip -> clipped mean/std -> z); composite 再 z(G); derived 最后 z(raw)。July 全程只 apply。

## 1. 样本规模

| 切片 | rows | signal dates | 日期范围 |
|---|---|---|---|
| June (development / reference X sample) | 173 | 21 | 2026-06-01 ~ 2026-06-30 |
| July (retrospective unlabeled X-stability slice) | 160 | 21 | 2026-07-01 ~ 2026-07-29 |

- 窗口外行: 0 (全部行都在 June/July 窗口内)

## 2. primitive 相关性 (June 主参考; July 仅稳定性描述)

- 最大 June Pearson pair: `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw` rho=-0.8261 (severity=SEVERE)
- 最大 June Spearman pair: `d1_high_to_close_drawdown_raw-d1_close_to_vwap_raw` rho=-0.8554 (severity=SEVERE)
- 完整 21 对明细见 `v004c_repair_state_primitive_dependence_v001.csv` (reclaim_internal 标记 RECLAIM 内部对)。

## 3. RECLAIM 内部结构审计 (d1_low_to_close_recovery vs d1_afternoon_return)

| 切片 | Pearson | Spearman |
|---|---|---|
| June | 0.5787 | 0.5414 |
| July | 0.4409 | 0.4306 |

- 目的: 判断它们是否确实可以作为同一 RECLAIM latent composite 的两个视角 (50/50 等权是否合理)。
- composite status: **COMPOSITE_SUPPORTED**
- June |Pearson| / |Spearman| 至少一项 >= 0.10: 存在同一潜变量的可辨 证据, composite 构造合理 (COMPOSITE_SUPPORTED)。

## 4. factor 构造与 June 分布 (7 factors)

- 构造: 7 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8 (degenerate 检查: 无退化)
- derived terms: raw = F_D^2 / F_D*F_Q / F_D*F_S, 只做 mean/std 标准化, 未施加第二次 clip

| factor | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPEN | -0.0000 | 1.0000 | -2.1741 | -2.1732 | -1.4013 | -0.6423 | -0.2152 | 0.5928 | 2.0157 | 2.5785 | 2.5787 | 0.5323 |
| DIVERGENCE | 0.0000 | 1.0000 | -2.1411 | -1.8466 | -1.4110 | -0.7259 | -0.0921 | 0.6937 | 1.6833 | 2.2581 | 2.6676 | 0.3490 |
| SUPPLY | -0.0000 | 1.0000 | -2.0145 | -2.0145 | -1.6577 | -0.6208 | -0.1037 | 0.5621 | 1.9502 | 2.3039 | 2.3306 | 0.3088 |
| RECLAIM | 0.0000 | 1.0000 | -2.1773 | -1.9649 | -1.5261 | -0.6864 | -0.1069 | 0.5853 | 1.6802 | 3.2483 | 3.5383 | 0.7092 |
| DIVERGENCE_SQ | 0.0000 | 1.0000 | -0.7988 | -0.7985 | -0.7942 | -0.6942 | -0.3778 | 0.3597 | 2.1914 | 3.4166 | 4.8852 | 2.1824 |
| DIVERGENCE_X_RECLAIM | 0.0000 | 1.0000 | -5.8359 | -4.1254 | -1.6600 | -0.2598 | 0.3398 | 0.6145 | 0.7898 | 1.0623 | 1.4015 | -2.7171 |
| DIVERGENCE_X_SUPPLY | 0.0000 | 1.0000 | -5.0562 | -2.9761 | -1.3868 | -0.3373 | -0.0706 | 0.4214 | 1.7786 | 2.7082 | 2.8287 | -0.7940 |

## 5. factor 相关性 (June 主参考; July 稳定性; hierarchy 分类)

- 最大 June Pearson pair: `DIVERGENCE-RECLAIM` rho=-0.7340 (severity=HIGH)
- 最大 June Spearman pair: `DIVERGENCE-RECLAIM` rho=-0.7725 (severity=HIGH)
- 完整 21 对明细见 `v004c_repair_state_factor_dependence_v001.csv` (hierarchy_expected 标记层级依赖)。

### 层级项分类 (EXPECTED_HIERARCHICAL_DEPENDENCE vs UNEXPECTED_REDUNDANCY)

- 天然层级依赖 pair (derived 与其父 factor, 高相关属预期):
  - `DIVERGENCE-DIVERGENCE_SQ`: June Pearson=0.2763 / Spearman=0.0260 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `DIVERGENCE-DIVERGENCE_X_RECLAIM`: June Pearson=0.0621 / Spearman=-0.0079 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `DIVERGENCE-DIVERGENCE_X_SUPPLY`: June Pearson=-0.1576 / Spearman=-0.1250 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `SUPPLY-DIVERGENCE_X_SUPPLY`: June Pearson=0.2156 / Spearman=0.0843 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)
  - `RECLAIM-DIVERGENCE_X_RECLAIM`: June Pearson=-0.2538 / Spearman=-0.0093 (severity=LOW) — EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)

- 非层级 SEVERE pairs: 无 (所有高相关 pair 都属层级依赖)。

## 6. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)

| factor | R1 VIF | R2 VIF |
|---|---|---|
| OPEN | 1.176 | 1.306 |
| DIVERGENCE | 2.525 | 3.542 |
| SUPPLY | 1.090 | 1.283 |
| RECLAIM | 2.396 | 3.063 |
| DIVERGENCE_SQ | — | 3.190 |
| DIVERGENCE_X_RECLAIM | — | 2.854 |
| DIVERGENCE_X_SUPPLY | — | 1.263 |

- R1 max VIF: 2.525 (factor `DIVERGENCE`); VIF >= 5: 无; VIF >= 10: 无
- R2 max VIF: 3.542 (factor `DIVERGENCE`); VIF >= 5: 无; VIF >= 10: 无
- 解释: VIF < 5 OK; 5 <= VIF < 10 WATCH; VIF >= 10 SEVERE (只解释, 禁止自动删因子; VIF/condition number 必须真实报告, 不因层级项人为放宽)

## 7. condition number (centered + unit-variance 诊断矩阵, SVD)

| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |
|---|---|---|---|---|---|
| R1 | OPEN DIVERGENCE SUPPLY RECLAIM | 18.291 | 6.237557 | 2.932 | OK |
| R2 | 7 factors (含 3 个层级项) | 18.487 | 5.116557 | 3.613 | OK |
| REPAIR_BLOCK | DIVERGENCE RECLAIM SUPPLY | 17.868 | 6.643486 | 2.690 | OK |
| CONDITIONAL_BLOCK | DIVERGENCE_SQ X_RECLAIM X_SUPPLY | 17.180 | 6.869831 | 2.501 | OK |
- 参考: kappa < 30 OK; 30-100 WATCH; >= 100 SEVERE; smallest singular ~ 0 => SEVERE / near singular

## 8. July 无标签 X 稳定性 (June reference transform 映射)

| factor | June mean | June std | July mean | July std | SMD | KS | flag |
|---|---|---|---|---|---|---|---|
| OPEN | -0.0000 | 1.0000 | -0.0994 | 1.1824 | -0.0908 | 0.1246 | — |
| DIVERGENCE | 0.0000 | 1.0000 | 0.2327 | 1.0151 | 0.2310 | 0.1226 | — |
| SUPPLY | -0.0000 | 1.0000 | 0.1047 | 1.1109 | 0.0991 | 0.0774 | — |
| RECLAIM | 0.0000 | 1.0000 | -0.2235 | 0.9769 | -0.2261 | 0.1367 | — |
| DIVERGENCE_SQ | 0.0000 | 1.0000 | 0.0675 | 1.1551 | 0.0625 | 0.0627 | — |
| DIVERGENCE_X_RECLAIM | 0.0000 | 1.0000 | -0.0268 | 0.8364 | -0.0291 | 0.0732 | — |
| DIVERGENCE_X_SUPPLY | 0.0000 | 1.0000 | -0.1106 | 1.0042 | -0.1103 | 0.1397 | — |

- 判定: |SMD| >= 0.50 或 KS >= 0.25 => SHIFT_WATCH (只记录, 禁止自动换因子)
- SHIFT_WATCH 因子: 无

## 9. derived tail 诊断 (interaction 是否被少量极端值主导)

| factor | 切片 | min | p01 | p05 | median | p95 | p99 | max | mean | std |
|---|---|---|---|---|---|---|---|---|---|---|
| DIVERGENCE_SQ | june | -0.7988 | -0.7985 | -0.7942 | -0.3778 | 2.1914 | 3.4166 | 4.8852 | 0.0000 | 1.0000 |
| DIVERGENCE_SQ | july | -0.7987 | -0.7983 | -0.7928 | -0.4135 | 2.5955 | 4.0384 | 4.3433 | 0.0675 | 1.1551 |
| DIVERGENCE_X_RECLAIM | june | -5.8359 | -4.1254 | -1.6600 | 0.3398 | 0.7898 | 1.0623 | 1.4015 | 0.0000 | 1.0000 |
| DIVERGENCE_X_RECLAIM | july | -3.2904 | -2.4565 | -1.9522 | 0.2210 | 0.7909 | 0.9642 | 1.1785 | -0.0268 | 0.8364 |
| DIVERGENCE_X_SUPPLY | june | -5.0562 | -2.9761 | -1.3868 | -0.0706 | 1.7786 | 2.7082 | 2.8287 | 0.0000 | 1.0000 |
| DIVERGENCE_X_SUPPLY | july | -4.4030 | -3.4068 | -1.6875 | -0.1275 | 1.3240 | 2.5153 | 3.1724 | -0.1106 | 1.0042 |

- 判定: June 标准化后 max|min| / max|max| >= 5.0 => DERIVED_TAIL_WATCH (只记录; derived 项不做第二次 clip, 不 winsorize)
- DERIVED_TAIL_WATCH 因子: `DIVERGENCE_X_RECLAIM, DIVERGENCE_X_SUPPLY`

## 10. Hierarchy 检查确认

- R2 成员包含每个 derived term 的全部父 factor: PASS
- 禁止新增 interaction (OPEN_X_DIVERGENCE / HIGHZONE_X_DIVERGENCE / HIGHZONE_X_RECLAIM / RECLAIM_SQ / SUPPLY_SQ): 不属于 v001 模型, 未构造
- LEGACY_SENSITIVITY (HIGHZONE, MOM7, DAMAGE7, REGIME, POS7, TREND) 全部不在 R1/R2 core: PASS

## 11. 最终状态

- 最终状态: **PASS_REPAIR_STATE_X_V001**
- 判定门 (任务 §32): degenerate factor / VIF >= 10 / condition number >= 100 => REVIEW_REQUIRED; 否则 RECLAIM 两 primitive 在 June 中 |Pearson| < 0.10 且 |Spearman| < 0.10 => RECLAIM_COMPOSITE_REVIEW (进入人工审查, 不自动拆); 否则 PASS_REPAIR_STATE_X_V001。
- 非阻塞记录 (不进入判定门): SHIFT_WATCH = 无; DERIVED_TAIL_WATCH = `DIVERGENCE_X_RECLAIM, DIVERGENCE_X_SUPPLY`; hierarchy 相关高只记 EXPECTED_HIERARCHICAL_DEPENDENCE
- 本审查只做 X 结构 / 数值稳定性 / June→July X 稳定性诊断: 未读取 Target, 未执行任何训练/预测/指标, 未用 Target7 rate / AUC / LogLoss / Brier / coefficient / Top3 参与判定
- 禁止自动修正: 即使出现 SEVERE / VIF >= 10 / kappa >= 100 / 退化, 本阶段也不 drop / swap / PCA / Lasso / 改权重 / 加 interaction; 修正方向只记录给人工

## 12. 结论与后续流程

- 本阶段只回答: R1/R2 的 X 结构是否健康、RECLAIM composite 是否合理、 层级项是否数值健康、June→July X 是否发生明显漂移。
- 下一阶段必须人工审查: RECLAIM 是否构造合理 / interaction 是否数值健康 / R1/R2 是否存在严重共线性 / June→July X 是否明显漂移; 只有人工批准后才能进入 Repair-State development fit + frozen retrospective OOT。
- post-June hypothesis: 本模型是 June walk-forward 失败后形成的假设, June 不能再作为独立验证集; 后续流程 = X spec freeze -> June development fit -> freeze coefficients/spec -> July retrospective OOT -> August+ forward shadow。


# v004c Repair-State Model v002 — June Full-Development Fit + Frozen Model

- 阶段: June full-development fit (R0/R1/R2) + 冻结模型工件 (FROZEN FOR JULY RETROSPECTIVE OOT)
- 分支: research-sample-analysis
- 输入: `v004c_model_table_v001.csv` (X-only SHA256: `f78da6a8a4436a3942b9e5df412965f00316db6adf87e9d69169314fffb57f49`; July Target 值不参与 hash, 保证资产对 July Target 变化不变)
- 三件事必须区分:
  - **Factor structure**: **FROZEN** — Repair-State v002 (人工批准 FREEZE_REPAIR_STATE_V002; 本任务未修改任何 factor / interaction / 预处理 / Logistic 超参数)
  - **June model fit**: **DEVELOPMENT_IN_SAMPLE_ONLY** — post-June hypothesis, June 不是验证集; 本阶段全部指标仅 in-sample 诊断, 不用来 选择 R1/R2, 不宣称 validated / OOF / out-of-sample / generalizes
  - **Model artifact**: **FROZEN FOR JULY RETROSPECTIVE OOT** — 下一阶段 只加载 frozen JSON apply, 禁止 refit / recalibrate / 调参

## 0. 方法 (固定, 不搜索)

- transform 完全复用 `v004c_repair_state_spec_v002.py` (fit_repair_state_transform_v2 / construct_repair_factors_v2), 不复制第二套公式; June 只 fit 一次, July 全程只 apply
- R0 = June prevalence (无 ranking 意义); R1 = 5 base factors; R2 = 8 factors (5 base + DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM / DIVERGENCE_X_DAMAGE)
- Logistic 固定: L2 (`penalty=l2`), `C=1.0`, `solver=lbfgs`, `fit_intercept=True`, `class_weight=None`, `max_iter=1000`, `random_state=None`; 禁止 GridSearch / RandomSearch / Optuna / CV tuning
- daily rank: probability 降序, tie-break `event_id` 升序 (确定性); 禁止用 Target 打破 tie; R0 概率同日恒同 => ranking = N/A
- 判定门 (§30): 只有 non-convergence / NaN-Inf / abs(coef) >= 10 / frozen JSON 无法复现 / 实现错误 / July target leakage 允许 MODEL_FREEZE_REVIEW_REQUIRED; June AUC / LogLoss / 系数方向不参与判定
- 确定性: 无时间戳 / 无随机; 两次运行 9 个资产字节级一致; July Target 随机化 / 删除不影响任何正式输出

## 1. 训练切片

- period: 2026-06-01 ~ 2026-06-30; rows: **173**; signal dates: **21**; positive: **59**; negative: **114**; base rate: **34.10%**
- Target (冻结): `target7_daily_d2open_d3high` = 1[High_D3/Open_D2 - 1 >= 0.07], 预测时点 D1 close
- 窗口外行: 0 (全部行在 June/July 窗口内)

## 2. July 边界 (硬要求)

- July X read: **YES** (Step A 只加载 ID + 8 core primitive, 仅用于 确定 June 行号与窗口完整性; July X 不参与任何 transform / fit)
- July Target read: **NO**; July Target loaded: **NO**; July Target used: **NO**
- 实现: Step B 逐行 csv 流式读取, 只有 June 行解析 Target 值; July 行 的 Target 数值从未被解析 / 保留 / 使用 (测试: July 行 Target 为非法值 不影响 fit)
- July X rows: 160 (21 dates); 只记录行数

## 3. Preprocessing (June-only fit; 复用冻结 spec)

- primitive: June 原始有限值 -> fit q01/q99 -> clip -> fit mu/sigma (clipped June) -> z; 禁止 mean/std before clipping / July fit / Target-dependent preprocessing / 全样本预处理
- composite: CLOSE_DAMAGE = z(G_C), G_C = (-z_OC + z_HC - z_VWAP)/3 (复用 composite_reset, 与旧 RESET 数学一致); RECLAIM = z(G_Q), G_Q = (z(low_to_close_recovery) + z(afternoon_return))/2 (v001 冻结)
- derived: raw = F_V^2 / F_V*F_Q / F_V*F_C -> fit June mean/std -> 标准化; **无第二次 clip** (参数无 q01/q99)
- 全部 transform 参数已冻结进 frozen JSON, 下一阶段禁止修改

## 4. R0 — Base-rate model

- probability (June prevalence): **34.10%** (59 / 173)
- intercept (logit): -0.658661
- ranking: **N/A** (同日所有候选概率相同, 无横截面排序意义)

## 5. R1 — Base Repair State (严格 5 factors + intercept)

logit(p) = α + β_O OPEN + β_V DIVERGENCE + β_C CLOSE_DAMAGE + β_S SUPPLY + β_Q RECLAIM

| factor | coefficient |
|---|---|
| INTERCEPT | -0.673046 |
| OPEN | 0.074832 |
| DIVERGENCE | -0.087949 |
| CLOSE_DAMAGE | 0.437198 |
| SUPPLY | -0.145838 |
| RECLAIM | 0.344015 |

- n_iter: 8; converged: **True**

## 6. R2 — Conditional Repair State (严格 8 factors + intercept)

logit(p) = α + β_O O + β_V V + β_C C + β_S S + β_Q Q + γ₁ V² + γ₂ VQ + γ₃ VC

| factor | coefficient |
|---|---|
| INTERCEPT | -0.676573 |
| OPEN | 0.119904 |
| DIVERGENCE | -0.156602 |
| CLOSE_DAMAGE | 0.539700 |
| SUPPLY | -0.162977 |
| RECLAIM | 0.414053 |
| DIVERGENCE_SQ | 0.278701 |
| DIVERGENCE_X_RECLAIM | -0.286307 |
| DIVERGENCE_X_DAMAGE | -0.364459 |

- n_iter: 13; converged: **True**
- 禁止加入: DIVERGENCE_X_SUPPLY / TURNOVER_COST / HIGHZONE / MOM7 / DAMAGE7 / REGIME / POS7 / TREND (全部保持 SENSITIVITY / LEGACY_SENSITIVITY)

## 7. 训练算法 (冻结, 禁止搜索)

| parameter | value |
|---|---|
| library | scikit-learn |
| estimator | LogisticRegression |
| penalty | l2 |
| C | 1.0 |
| solver | lbfgs |
| fit_intercept | True |
| class_weight | None |
| max_iter | 1000 |
| random_state | None |
- hyperparameter search: **NO** (GridSearch / RandomSearch / Optuna / CV tuning 全部禁止)

## 8. June development diagnostics — DEVELOPMENT_IN_SAMPLE_ONLY

> 本表全部指标为 June in-sample 诊断 (拟合 173 行 + 评估 173 行), 不是 validated / OOF / out-of-sample; 禁止宣称性能 proven; 不用于 选择 R1/R2。

| model | LogLoss | Brier | AUC | AP | mean p | observed | gap | Top1 | Top3 |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.641723 | 0.224732 | 0.5000 | 0.3410 | 34.10% | 34.10% | -0.0000 | NA | NA |
| R1 | 0.627730 | 0.218159 | 0.6090 | 0.4488 | 34.11% | 34.10% | 0.0000 | 0.4762 | 0.3871 |
| R2 | 0.623281 | 0.215792 | 0.6090 | 0.4920 | 34.10% | 34.10% | -0.0000 | 0.5238 | 0.3871 |

- R0 AUC = 0.5 / AP = base rate: 结构化无区分度 (同日概率恒同), 不是 模型缺陷
- Top1/Top3 是 ranking diagnostic, 不是概率模型 primary gate; Top3 denominator = sum(min(3, candidates))

## 9. Within-date development ranking (仍 DEVELOPMENT_IN_SAMPLE_ONLY)

- 只对同日同时存在至少 1 positive 与至少 1 negative 的日期计算 daily AUC (valid_auc_dates)

| model | valid dates | mean daily AUC | median daily AUC | pair-weighted within-date AUC | dates > .5 | = .5 | < .5 |
|---|---|---|---|---|---|---|---|
| R1 | 19 | 0.5788 | 0.6122 | 0.5706 | 12 | 1 | 6 |
| R2 | 19 | 0.6124 | 0.6250 | 0.5982 | 12 | 2 | 5 |

## 10. 自然候选率 (候选池属性, 与模型无关)

- pooled candidate Target rate: 34.10% (59 / 173)
- mean daily candidate Target rate: 32.95%

## 11. 系数健康与概率范围 (§21)

| model | max abs coefficient | factor | magnitude watch (>=5) | numerical review (>=10) |
|---|---|---|---|---|
| R1 | 0.6730 | INTERCEPT | — | — |
| R2 | 0.6766 | INTERCEPT | — | — |

| model | min | p01 | p05 | median | p95 | p99 | max | p<0.01 | p>0.99 |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.205691 | 0.216438 | 0.247943 | 0.322185 | 0.491247 | 0.572115 | 0.604012 | 0 | 0 |
| R2 | 0.179655 | 0.208188 | 0.240060 | 0.322413 | 0.482109 | 0.549773 | 0.611578 | 0 | 0 |

- coefficients finite: **YES** (R1/R2 全部 finite); 0 < p < 1 且全部 finite
- COEFFICIENT_MAGNITUDE_REVIEW (abs >= 5): 无 (只记录, 不自动失败)
- MODEL_NUMERICAL_REVIEW (abs >= 10): 无
- PROBABILITY_EXTREME_WATCH (p<0.01 或 p>0.99 占比 >= 5%): NO

## 12. Interaction 极端样本审计 (§20, 只诊断)

- contribution_j = coefficient_j × factor_j (R2 每样本); 禁止根据这些 样本删行 / 再 clip / winsorize / 调 C

| factor | coefficient | max abs contribution | event | signal date |
|---|---|---|---|---|
| OPEN | 0.1199 | 0.3092 | 603616_2026-06-10 | 2026-06-10 |
| DIVERGENCE | -0.1566 | 0.4497 | 603261_2026-06-22 | 2026-06-22 |
| CLOSE_DAMAGE | 0.5397 | 1.4397 | 000517_2026-06-09 | 2026-06-09 |
| SUPPLY | -0.1630 | 0.3798 | 002119_2026-06-12 | 2026-06-12 |
| RECLAIM | 0.4141 | 1.4651 | 002068_2026-06-24 | 2026-06-24 |
| DIVERGENCE_SQ | 0.2787 | 1.3488 | 603261_2026-06-22 | 2026-06-22 |
| DIVERGENCE_X_RECLAIM | -0.2863 | 1.8459 | 605580_2026-06-24 | 2026-06-24 |
| DIVERGENCE_X_DAMAGE | -0.3645 | 2.0484 | 001896_2026-06-05 | 2026-06-05 |

### Top 10 abs DIVERGENCE_X_RECLAIM contribution 行

| rank | event | signal date | factor value | contribution | abs |
|---|---|---|---|---|---|
| 1 | 605580_2026-06-24 | 2026-06-24 | 6.4472 | -1.8459 | 1.8459 |
| 2 | 002068_2026-06-24 | 2026-06-24 | 5.5360 | -1.5850 | 1.5850 |
| 3 | 001896_2026-06-05 | 2026-06-05 | -4.5572 | 1.3048 | 1.3048 |
| 4 | 000811_2026-06-23 | 2026-06-23 | 4.4105 | -1.2628 | 1.2628 |
| 5 | 603261_2026-06-22 | 2026-06-22 | 3.3558 | -0.9608 | 0.9608 |
| 6 | 600178_2026-06-12 | 2026-06-12 | -2.3803 | 0.6815 | 0.6815 |
| 7 | 002579_2026-06-03 | 2026-06-03 | -1.9925 | 0.5705 | 0.5705 |
| 8 | 000517_2026-06-09 | 2026-06-09 | -1.6409 | 0.4698 | 0.4698 |
| 9 | 600280_2026-06-03 | 2026-06-03 | 1.5641 | -0.4478 | 0.4478 |
| 10 | 603155_2026-06-30 | 2026-06-30 | 1.5014 | -0.4299 | 0.4299 |

### Top 10 abs DIVERGENCE_X_DAMAGE contribution 行

| rank | event | signal date | factor value | contribution | abs |
|---|---|---|---|---|---|
| 1 | 001896_2026-06-05 | 2026-06-05 | 5.6205 | -2.0484 | 2.0484 |
| 2 | 000517_2026-06-09 | 2026-06-09 | 4.8499 | -1.7676 | 1.7676 |
| 3 | 600228_2026-06-29 | 2026-06-29 | 4.2845 | -1.5615 | 1.5615 |
| 4 | 002068_2026-06-24 | 2026-06-24 | -3.7975 | 1.3840 | 1.3840 |
| 5 | 605580_2026-06-24 | 2026-06-24 | -3.6081 | 1.3150 | 1.3150 |
| 6 | 002584_2026-06-25 | 2026-06-25 | 3.1437 | -1.1457 | 1.1457 |
| 7 | 002141_2026-06-22 | 2026-06-22 | 3.0295 | -1.1041 | 1.1041 |
| 8 | 000823_2026-06-29 | 2026-06-29 | 1.8456 | -0.6726 | 0.6726 |
| 9 | 600178_2026-06-12 | 2026-06-12 | 1.7445 | -0.6358 | 0.6358 |
| 10 | 603757_2026-06-24 | 2026-06-24 | -1.6225 | 0.5913 | 0.5913 |

- anomalies: 无 (全部 contribution finite; 无 |contribution| >= 10)

## 13. R2 层级解释 (marginal effect; 只解释, 不选模型)

- 禁止把 β_DIVERGENCE / β_CLOSE_DAMAGE / β_RECLAIM 孤立解释成全局单调 效应; R2 中 DIVERGENCE 的边际作用:

    ∂logit(p)/∂V = β_V + 2γ₁V + γ₂Q + γ₃C

- 标准情景 (V=-1,0,+1,+2; Q=-1,0,+1; C=-1,0,+1; 不使用 Target 选择 情景; 目的: 确认模型能表达 '适度分歧最好、极端分歧可能透支、回收和 最终损伤改变分歧价值'):

| V | Q | C | ∂logit/∂V |
|---|---|---|---|
| -1.0000 | -1.0000 | -1.0000 | -0.0632 |
| -1.0000 | -1.0000 | 0.0000 | -0.4277 |
| -1.0000 | -1.0000 | 1.0000 | -0.7922 |
| -1.0000 | 0.0000 | -1.0000 | -0.3495 |
| -1.0000 | 0.0000 | 0.0000 | -0.7140 |
| -1.0000 | 0.0000 | 1.0000 | -1.0785 |
| -1.0000 | 1.0000 | -1.0000 | -0.6359 |
| -1.0000 | 1.0000 | 0.0000 | -1.0003 |
| -1.0000 | 1.0000 | 1.0000 | -1.3648 |
| 0.0000 | -1.0000 | -1.0000 | 0.4942 |
| 0.0000 | -1.0000 | 0.0000 | 0.1297 |
| 0.0000 | -1.0000 | 1.0000 | -0.2348 |
| 0.0000 | 0.0000 | -1.0000 | 0.2079 |
| 0.0000 | 0.0000 | 0.0000 | -0.1566 |
| 0.0000 | 0.0000 | 1.0000 | -0.5211 |
| 0.0000 | 1.0000 | -1.0000 | -0.0785 |
| 0.0000 | 1.0000 | 0.0000 | -0.4429 |
| 0.0000 | 1.0000 | 1.0000 | -0.8074 |
| 1.0000 | -1.0000 | -1.0000 | 1.0516 |
| 1.0000 | -1.0000 | 0.0000 | 0.6871 |
| 1.0000 | -1.0000 | 1.0000 | 0.3226 |
| 1.0000 | 0.0000 | -1.0000 | 0.7653 |
| 1.0000 | 0.0000 | 0.0000 | 0.4008 |
| 1.0000 | 0.0000 | 1.0000 | 0.0363 |
| 1.0000 | 1.0000 | -1.0000 | 0.4790 |
| 1.0000 | 1.0000 | 0.0000 | 0.1145 |
| 1.0000 | 1.0000 | 1.0000 | -0.2500 |
| 2.0000 | -1.0000 | -1.0000 | 1.6090 |
| 2.0000 | -1.0000 | 0.0000 | 1.2445 |
| 2.0000 | -1.0000 | 1.0000 | 0.8800 |
| 2.0000 | 0.0000 | -1.0000 | 1.3227 |
| 2.0000 | 0.0000 | 0.0000 | 0.9582 |
| 2.0000 | 0.0000 | 1.0000 | 0.5937 |
| 2.0000 | 1.0000 | -1.0000 | 1.0364 |
| 2.0000 | 1.0000 | 0.0000 | 0.6719 |
| 2.0000 | 1.0000 | 1.0000 | 0.3074 |

## 14. Frozen artifact (§25)

- JSON: `v004c_repair_state_frozen_model_v001.json`
- 内容: model_version / factor_spec_version / training_period / training_rows / training_dates / target_column / algorithm (固定参数) / R1+R2 (factor_order + intercept + coefficients) / primitive transforms (q01/q99/mu/sigma) / composite transforms (CLOSE_DAMAGE, RECLAIM) / derived transforms (3 个 mu/sigma) / ranking 规则 / research status (post_june_hypothesis=true, june_is_validation=false, july_target_seen=false)
- 重建验证: R1 prediction reproduction max error = 1.1102230246251565e-16; R2 = 5.551115123125783e-17 (须 < 1e-12)
- full preprocessing frozen: **YES**; coefficients frozen: **YES**; ranking frozen: **YES** (probability descending, event_id ascending)
- 禁止存储: July Target 派生统计 / future-selected thresholds

## 15. Leakage 测试 (§29, 测试覆盖)

- July target randomization invariance: **PASS** (Version B: July Target 随机化 => June transform params / R1+R2 coefficients / June predictions / frozen JSON 完全不变, 9 个资产 byte-identical)
- July target removal invariance: **PASS** (Version C: July 行 Target 删除/置空 => fit 仍完全成功, 资产 byte-identical)
- July X mutation cannot alter June coefficients: **PASS** (修改 July X 后 June coefficients 逐位相同)
- July target value 从未被解析: **PASS** (July 行 Target 为非法值不影响 任何输出)
- X-only hash: July Target 值不参与 input hash, 全部模型资产 byte-identical

## 16. July 合同 (下一阶段不可变)

- 本任务 commit 完成后, 以下全部不得因 July 结果修改: R1 factors / R2 factors / factor formulas / factor order / primitive clipping params / primitive mean/std / composite mean/std / derived mean/std / Logistic C / penalty / solver / coefficients / ranking rule / Top1/Top3 rule
- July 阶段只能: load frozen JSON -> load July X -> apply frozen transforms -> predict probabilities -> open July Target once -> evaluate
- 禁止: fit / refit / recalibrate / threshold tuning / coefficient adjustment / factor changes

## 17. 最终状态

- 最终研究状态: **FROZEN_REPAIR_STATE_MODEL_V001**
- June interpretation: development only; model selected using June: **NO**; R1 frozen: **YES**; R2 frozen: **YES** (两个都必须冻结, 由 July OOT 负责 R0 vs R1 vs R2 比较)
- Historical assets modified: **NO**
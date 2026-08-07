# v004c M0/M1/M2 Logistic expanding-date walk-forward (June 开发窗口)

- 阶段: 正式模型开发 — June expanding-date walk-forward (M0/M1/M2)
- 分支: research-sample-analysis
- 输入: `v004c_model_table_v001.csv` SHA256: `c0de3bc639142c3e0b14b8018ff7ca13351ba46ed8f43cba3f54b18a261bd274`
- June 窗口: 2026-06-01 ~ 2026-06-30 (173 rows, 21 signal dates); July 完全不读取
- Target (冻结): `target7_daily_d2open_d3high` = 1[High_D3/Open_D2 - 1 >= 0.07], 预测时点 D1 close; 只用于训练 label 与 OOF 评估
- Model selected/frozen: **NO** (最终模型由人工审查 OOF 概率质量 / M1 vs M0 / M2 vs M1 / 系数路径 / ranking 后决定)

## 0. 方法 (固定, 不搜索)

- 每个 fold 独立拟合 FOLD_CLIP_Z: training fold raw primitives -> q01/q99 (train) -> clip -> mu/sigma (clipped train) -> z -> factor 构造 -> RESET composite mean/std (train 上构造的 G) -> Logistic
- test date 严格不在 train (`signal_date < test_date`); 同一天全部候选 原子进入同一个 test fold; 禁止 random/row-level/KFold/StratifiedKFold
- Logistic 固定: L2 (`penalty=l2`), `C=1.0`, `solver=lbfgs`, `fit_intercept=True`, `class_weight=None`, `max_iter=1000`; 不搜索, 不按 fold 调整
- M0 = training-fold prevalence (非人工 constant feature 调 L2 Logistic); M1 = OPEN, RESET, HIGHZONE, LATESELL; M2 = OPEN, RESET, HIGHZONE, LATESELL, MOM7, DAMAGE7, REGIME; POS7/TREND 是 SENSITIVITY 不进 M1/M2; SUPPLY active = NO
- 启动条件固定: 首次允许 test 需此前 training `signal_dates >= 10, rows >= 70, positive >= 15, negative >= 30`; 不因结果修改
- 禁止: 自动 feature selection / C / hyperparameter / class_weight search / interaction / PCA / Lasso / ElasticNet / 树模型 / boosting / NN; 禁止根据 fold 结果改 factor
- daily rank: probability 降序, tie-break `event_id` 升序 (确定性); 禁止用 target/future return/code 表现打破 tie; M0 概率同日恒同, M0 daily rank = NOT_APPLICABLE
- 确定性: 全流程无时间戳/无随机, 两次运行 6 个资产字节级一致

## 1. Data

- June signal dates 总数: 21 (2026-06-01 ~ 2026-06-30)
- 初始 training: 12 dates (2026-06-01 ~ 2026-06-16), 72 rows (pos=20, neg=52, base rate=27.78%)
- 第一个 OOF test date: **2026-06-17**
- OOF 覆盖: 9 dates (2026-06-17 ~ 2026-06-30), 101 rows (pos=39, neg=62)
- 启动条件记录: 该 fold 之前 train dates=12 >= 10, rows=72 >= 70, pos=20 >= 15, neg=52 >= 30

### fold 汇总 (详见 `v004c_walkforward_folds_v001.csv`)

| fold | test date | train dates | train rows | pos | neg | base rate | test rows | pos | neg | M0 prob | M1 n_iter | M2 n_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_01 | 2026-06-17 | 12 | 72 | 20 | 52 | 0.278 | 17 | 12 | 5 | 0.278 | 6 | 10 |
| fold_02 | 2026-06-18 | 13 | 89 | 32 | 57 | 0.360 | 15 | 2 | 13 | 0.360 | 7 | 10 |
| fold_03 | 2026-06-22 | 14 | 104 | 34 | 70 | 0.327 | 7 | 4 | 3 | 0.327 | 6 | 10 |
| fold_04 | 2026-06-23 | 15 | 111 | 38 | 73 | 0.342 | 14 | 7 | 7 | 0.342 | 7 | 10 |
| fold_05 | 2026-06-24 | 16 | 125 | 45 | 80 | 0.360 | 15 | 3 | 12 | 0.360 | 5 | 12 |
| fold_06 | 2026-06-25 | 17 | 140 | 48 | 92 | 0.343 | 9 | 4 | 5 | 0.343 | 5 | 10 |
| fold_07 | 2026-06-26 | 18 | 149 | 52 | 97 | 0.349 | 12 | 3 | 9 | 0.349 | 6 | 11 |
| fold_08 | 2026-06-29 | 19 | 161 | 55 | 106 | 0.342 | 5 | 3 | 2 | 0.342 | 6 | 10 |
| fold_09 | 2026-06-30 | 20 | 166 | 58 | 108 | 0.349 | 7 | 1 | 6 | 0.349 | 6 | 11 |

## 2. M0 — INTERCEPT BASELINE

- OOF LogLoss: 0.697476 (probability 只做数值安全 eps=1e-15 clip)
- OOF Brier: 0.250327
- observed OOF target rate: 38.61% (39 / 101)
- mean predicted probability: 33.69%
- AUC = 0.5, AP = OOF base rate: **M0 discrimination is structurally none** (同日概率恒同, 无横截面排序意义)

## 3. M1 — D1 STRUCTURE (OPEN, RESET, HIGHZONE, LATESELL)

- OOF LogLoss: 0.733877 (M0: 0.697476)
- OOF Brier: 0.263649 (M0: 0.250327)
- AUC: 0.3854; AP: 0.3582
- mean predicted probability: 32.24%; observed rate: 38.61%; calibration gap: -0.0637
- Top1 target rate: 33.33%; Top3 target rate: 33.33%; dates with >=1 Top3 hit: 6; zero-hit dates: 3
- 系数路径 (逐 fold, 见 §7 表与 `v004c_walkforward_coefficients_v001.csv`)

## 4. M2 — INTEGRATED PATH (OPEN, RESET, HIGHZONE, LATESELL, MOM7, DAMAGE7, REGIME)

- OOF LogLoss: 0.758661 (M1: 0.733877)
- OOF Brier: 0.273493 (M1: 0.263649)
- AUC: 0.3532; AP: 0.3124
- mean predicted probability: 31.64%; observed rate: 38.61%; calibration gap: -0.0697
- Top1 target rate: 33.33%; Top3 target rate: 37.04%; dates with >=1 Top3 hit: 7; zero-hit dates: 2
- MOM7/DAMAGE7/REGIME 增量: 见 §5 比较表; 共有 factor (OPEN/RESET/HIGHZONE/LATESELL) 在 M1 vs M2 中的重构见 §7

## 5. 模型比较 (pooled OOF; delta 方向统一: LogLoss/Brier improvement = baseline - model, AUC/AP = model - baseline, positive = improvement)

| model | LogLoss | Brier | AUC | AP | mean prob | observed | gap |
|---|---|---|---|---|---|---|---|
| M0 | 0.697476 | 0.250327 | 0.5000 | 0.3861 | 33.69% | 38.61% | -0.0493 |
| M1 | 0.733877 | 0.263649 | 0.3854 | 0.3582 | 32.24% | 38.61% | -0.0637 |
| M2 | 0.758661 | 0.273493 | 0.3532 | 0.3124 | 31.64% | 38.61% | -0.0697 |

| comparison | LogLoss improvement | Brier improvement | AUC delta | AP delta |
|---|---|---|---|---|
| M1 vs M0 | -0.036400 | -0.013322 | -0.114557 | -0.027961 |
| M2 vs M1 | -0.024784 | -0.009843 | -0.032258 | -0.045764 |
| M2 vs M0 | -0.061185 | -0.023166 | -0.146816 | -0.073725 |

## 6. Ranking 诊断 (每日 Top1/Top3; Top3 不足 3 候选取全部, denominator = min(3, candidate_count))

| test date | candidates | candidate target rate | M1 Top1 | M1 Top3 hits | M1 Top3 rate | M2 Top1 | M2 Top3 hits | M2 Top3 rate |
|---|---|---|---|---|---|---|---|---|
| 2026-06-17 | 17 | 70.59% | 0 | 0 | 0.00% | 0 | 1 | 33.33% |
| 2026-06-18 | 15 | 13.33% | 0 | 0 | 0.00% | 0 | 0 | 0.00% |
| 2026-06-22 | 7 | 57.14% | 1 | 2 | 66.67% | 1 | 3 | 100.00% |
| 2026-06-23 | 14 | 50.00% | 1 | 2 | 66.67% | 1 | 2 | 66.67% |
| 2026-06-24 | 15 | 20.00% | 0 | 1 | 33.33% | 0 | 1 | 33.33% |
| 2026-06-25 | 9 | 44.44% | 0 | 1 | 33.33% | 0 | 1 | 33.33% |
| 2026-06-26 | 12 | 25.00% | 1 | 2 | 66.67% | 1 | 1 | 33.33% |
| 2026-06-29 | 5 | 60.00% | 0 | 1 | 33.33% | 0 | 1 | 33.33% |
| 2026-06-30 | 7 | 14.29% | 0 | 0 | 0.00% | 0 | 0 | 0.00% |

- **M1**: Top1 target rate=33.33% (denominator=9 dates); Top3 target rate=33.33% (denominator=sum(min(3, candidates))=27 picks); dates with >=1 Top3 hit=6; zero-hit dates=3; mean daily candidate target rate=39.42%; pooled candidate target rate=38.61%
- **M2**: Top1 target rate=33.33% (denominator=9 dates); Top3 target rate=37.04% (denominator=sum(min(3, candidates))=27 picks); dates with >=1 Top3 hit=7; zero-hit dates=2; mean daily candidate target rate=39.42%; pooled candidate target rate=38.61%
- 注: Top3 指标是 ranking diagnostic, 不是概率模型 primary gate

## 7. Coefficient stability (near-zero: |coef| < 1e-8; consistency = max(pos, neg) / fold_count, 只用于解释, 禁止自动 drop factor)

| model | factor | folds | mean | std | median | p25 | p75 | min | max | pos | neg | ~0 | sign | consistency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | HIGHZONE | 9 | 0.0218 | 0.1974 | -0.0004 | -0.1132 | 0.0860 | -0.2195 | 0.4905 | 4 | 5 | 0 | - | 0.556 |
| M1 | LATESELL | 9 | -0.2328 | 0.0532 | -0.2295 | -0.2806 | -0.1823 | -0.3095 | -0.1481 | 0 | 9 | 0 | - | 1.000 |
| M1 | OPEN | 9 | 0.1152 | 0.1694 | 0.1839 | 0.0466 | 0.2245 | -0.3171 | 0.2408 | 8 | 1 | 0 | + | 0.889 |
| M1 | RESET | 9 | 0.0973 | 0.0830 | 0.0805 | 0.0634 | 0.0999 | -0.0041 | 0.3147 | 8 | 1 | 0 | + | 0.889 |
| M1 | INTERCEPT | 9 | -0.6904 | 0.1184 | -0.6662 | -0.6799 | -0.6374 | -1.0016 | -0.5861 | 0 | 9 | 0 | - | 1.000 |
| M2 | DAMAGE7 | 9 | 0.1309 | 0.0650 | 0.1418 | 0.0710 | 0.1720 | 0.0421 | 0.2564 | 9 | 0 | 0 | + | 1.000 |
| M2 | HIGHZONE | 9 | 0.0112 | 0.1807 | -0.0175 | -0.1289 | 0.1046 | -0.2317 | 0.3844 | 4 | 5 | 0 | - | 0.556 |
| M2 | LATESELL | 9 | -0.2386 | 0.0589 | -0.2285 | -0.3004 | -0.1825 | -0.3173 | -0.1519 | 0 | 9 | 0 | - | 1.000 |
| M2 | MOM7 | 9 | 0.0769 | 0.1019 | 0.0956 | 0.0286 | 0.1452 | -0.1395 | 0.2111 | 7 | 2 | 0 | + | 0.778 |
| M2 | OPEN | 9 | 0.1092 | 0.1577 | 0.1851 | 0.0594 | 0.1943 | -0.3055 | 0.2179 | 8 | 1 | 0 | + | 0.889 |
| M2 | REGIME | 9 | 0.0394 | 0.1537 | 0.0520 | -0.0807 | 0.0793 | -0.1524 | 0.3704 | 5 | 4 | 0 | + | 0.556 |
| M2 | RESET | 9 | 0.1184 | 0.0979 | 0.1075 | 0.0708 | 0.1319 | 0.0110 | 0.3668 | 9 | 0 | 0 | + | 1.000 |
| M2 | INTERCEPT | 9 | -0.6983 | 0.1344 | -0.6516 | -0.6918 | -0.6444 | -1.0586 | -0.5743 | 0 | 9 | 0 | - | 1.000 |

### M1 系数路径 (fold -> test date)

| factor | 2026-06-17 | 2026-06-18 | 2026-06-22 | 2026-06-23 | 2026-06-24 | 2026-06-25 | 2026-06-26 | 2026-06-29 | 2026-06-30 |
|---|---|---|---|---|---|---|---|---|---|
| INTERCEPT | -1.002 | -0.592 | -0.739 | -0.673 | -0.586 | -0.666 | -0.637 | -0.680 | -0.639 |
| OPEN | -0.317 | 0.047 | 0.033 | 0.177 | 0.224 | 0.231 | 0.216 | 0.241 | 0.184 |
| RESET | 0.315 | -0.004 | 0.059 | 0.080 | 0.063 | 0.115 | 0.083 | 0.065 | 0.100 |
| HIGHZONE | 0.491 | 0.036 | 0.127 | 0.086 | -0.000 | -0.048 | -0.113 | -0.219 | -0.162 |
| LATESELL | -0.257 | -0.309 | -0.281 | -0.295 | -0.148 | -0.182 | -0.181 | -0.229 | -0.213 |

### M2 系数路径 (fold -> test date)

| factor | 2026-06-17 | 2026-06-18 | 2026-06-22 | 2026-06-23 | 2026-06-24 | 2026-06-25 | 2026-06-26 | 2026-06-29 | 2026-06-30 |
|---|---|---|---|---|---|---|---|---|---|
| INTERCEPT | -1.059 | -0.614 | -0.737 | -0.666 | -0.574 | -0.644 | -0.648 | -0.692 | -0.652 |
| OPEN | -0.305 | 0.059 | 0.054 | 0.194 | 0.193 | 0.207 | 0.185 | 0.218 | 0.179 |
| RESET | 0.367 | 0.011 | 0.021 | 0.071 | 0.108 | 0.143 | 0.132 | 0.104 | 0.110 |
| HIGHZONE | 0.384 | 0.027 | 0.187 | 0.105 | -0.018 | -0.063 | -0.129 | -0.232 | -0.162 |
| LATESELL | -0.261 | -0.317 | -0.300 | -0.314 | -0.152 | -0.183 | -0.179 | -0.229 | -0.212 |
| MOM7 | 0.211 | 0.044 | -0.139 | -0.003 | 0.181 | 0.130 | 0.145 | 0.096 | 0.029 |
| DAMAGE7 | 0.071 | 0.147 | 0.092 | 0.188 | 0.256 | 0.172 | 0.142 | 0.067 | 0.042 |
| REGIME | 0.370 | 0.177 | -0.058 | -0.081 | -0.104 | -0.152 | 0.052 | 0.072 | 0.079 |

### 稳定性分析

- sign flip: M1 (HIGHZONE)/4/9 sign flips; M1 (OPEN)/1/9 sign flips; M1 (RESET)/1/9 sign flips; M2 (HIGHZONE)/4/9 sign flips; M2 (MOM7)/2/9 sign flips; M2 (OPEN)/1/9 sign flips; M2 (REGIME)/4/9 sign flips
- convergence failure: M1: 无; M2: 无
- 数值异常: 无
- 严重不稳定 (consistency < 0.55): 无

## 8. Leakage 与确定性

- test date 严格不在 train: 每 fold `max(train.signal_date) < test.signal_date` (测试覆盖: date leakage)
- same-date atomic: 同一 signal_date 全部候选要么全 train 要么全 test (测试覆盖)
- future-X invariance: 修改未来日期 X 后更早 fold 系数/概率不变 (测试覆盖)
- future-target invariance: 修改未来日期 Target 后更早 fold fit/预测不变 (测试覆盖)
- fold transform 顺序: q01/q99 (train raw) -> clip -> clipped mean/std (测试覆盖, 人工极端值区分新旧实现)
- deterministic rebuild: 全流程两次运行 6 个资产字节级一致 (测试覆盖)

## 9. July

- Target loaded: **NO**; July predictions generated: **NO**; July metrics: **NO** (July rows 在加载后立即丢弃, 未进入任何 fold / 预测 / 指标计算; July 是 retrospective OOT, 由人工决定模型后单独执行)

## 10. Final

- 最终研究状态: **REJECT_NO_STABLE_MODEL**
- 判定依据: M1 未改善 M0 概率质量 (LogLoss/Brier 均未下降), 且 M2 未同时改善 M1 (LogLoss/Brier)
- Model selected/frozen: **NO**; 是否进入下一阶段 (refit frozen model -> final coefficients -> July retrospective OOT) 由人工审查本报告后决定
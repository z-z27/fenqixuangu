# v004c 因子模型规格 v002 (M0/M1/M2 数学结构 — 仅记录, 不拟合)

- 阶段: 多变量建模前的纯 X 侧因子结构审查产物 (v002 修订); 本规格只定义公式与未来训练算法
- 状态: factor definition freeze 候选 (由 v004c_factor_dependence_review_v002.md 结论背书)
- 注意: 这是 factor definition freeze, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## v002 修订要点 (依据 v001 纯 X 结构审查结论, 与 Target 无关)

1. SUPPLY composite 拆分: `high_zone_volume_ratio` 与 `late_day_sell_volume_ratio`
   在 June 中接近独立 (Pearson=-0.0649 / Spearman=-0.0264), 不再强制压缩成 50/50 composite;
   HIGHZONE 与 LATESELL 作为独立 factor, 未来必须由 Logistic 独立估计系数,
   禁止重新合成 SUPPLY。v001 中 SUPPLY 的历史定义只保留在 v001 报告资产。
2. POS7 移入 SENSITIVITY: 与 RESET 结构重复 (v001 June Pearson=-0.8681 / Spearman=-0.8980);
   状态 SENSITIVITY_STRUCTURAL_REDUNDANCY, 不得进入 M2 core。
3. TREND 移入 SENSITIVITY: 与 MOM7 高度相关 (v001 June Pearson=0.7770 / Spearman=0.7764)
   + June→July X shift (v001 SMD ≈ -0.63); 状态 SENSITIVITY_HORIZON_STATIONARITY,
   不得进入第一版 M2 core。

## Factor 预处理 (future walk-forward)

- 连续 factor: 每个 training fold 内拟合 clip [q01, q99] + (x - mu)/sigma
  (q01/q99/mu/sigma 只允许来自当前 training fold; 禁止全样本参数)
- REGIME: 0/1 原值, 不做 z-score; 第一版禁止 interaction
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## M0 — INTERCEPT BASELINE

    logit(p) = alpha

## M1 — D1 STRUCTURE

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_HIGHZONE
             + beta_4 * F_LATESELL

(4 factors + intercept)

经济假设: D1 开盘承接 (OPEN)、D1 价格重置 (RESET)、高位筹码堆积 (HIGHZONE)
与尾盘主动卖压 (LATESELL) 是否共同形成对未来 Target7 概率有解释力的 D1 结构。
只描述结构问题, 不预设系数方向。

## M2 — INTEGRATED PATH MODEL

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_HIGHZONE
             + beta_4 * F_LATESELL
             + gamma_1 * F_MOM7
             + gamma_2 * F_DAMAGE7
             + gamma_3 * F_REGIME

(7 factors + intercept)

M2 回答: 在 D1 结构已知后, 最近价格路径 (MOM7/DAMAGE7) 和二/三板 regime (REGIME)
是否提供增量预测信息。DAMAGE7 即使存在 SHIFT_WATCH 也继续属于 core
(与 RESET/MOM7 提供明显不同的路径信息)。

## Sensitivity factors (不进 M1/M2 core, 只做诊断)

- POS7: SENSITIVITY_STRUCTURAL_REDUNDANCY (与 RESET 严重重复, 纯 X 结构发现)
- TREND: SENSITIVITY_HORIZON_STATIONARITY (与 MOM7 高度相关 + June→July X shift)

## 未来训练算法 (只记录, 本阶段禁止执行)

- L2 Logistic Regression: C = 1.0, solver = lbfgs, fit_intercept = True,
  class_weight = None
- 目标函数:

      min_theta [ -sum_i ( y_i*log(p_i) + (1-y_i)*log(1-p_i) ) + lambda * ||theta_0||_2^2 ]

- walk-forward 的目的不是只获得一组系数, 而是产生严格时间外 OOF 预测以及逐 fold
  系数路径, 用于检验预测能力、增量价值和参数稳定性
- 训练/验证: M0/M1/M2 expanding-date walk-forward (下一阶段)

## June/July 命名约定

- June: development / reference X sample
- July: retrospective unlabeled X-stability slice
  (July 不是 blind holdout; 本阶段只做 X 分布稳定性, 不读取 July Target)

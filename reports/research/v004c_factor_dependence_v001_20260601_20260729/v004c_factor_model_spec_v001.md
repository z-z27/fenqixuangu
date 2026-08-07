# v004c 因子模型规格 v001 (M0/M1/M2 数学结构 — 仅记录, 不拟合)

- 阶段: 多变量建模前的纯 X 侧因子结构审查产物; 本规格只定义公式与未来训练算法
- 状态: factor definition freeze 候选 (由 v004c_factor_dependence_review.md 结论背书)
- 注意: 这是 factor definition freeze, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## Factor 预处理 (future walk-forward)

- 连续 factor: 每个 training fold 内拟合 clip [q01, q99] + (x - mu)/sigma
  (q01/q99/mu/sigma 只允许来自当前 training fold; 禁止全样本参数)
- REGIME: 0/1 原值, 不做 z-score
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## M0 — INTERCEPT BASELINE

    logit(p) = alpha

## M1 — D1 STRUCTURE

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_SUPPLY

经济假设: D1 开盘承接、D1 价格重置和 D1 供应压力是否共同形成对未来 Target7
概率有解释力的结构。只描述结构问题, 不预设系数方向。

## M2 — INTEGRATED PATH MODEL

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_SUPPLY
             + gamma_1 * F_MOM7
             + gamma_2 * F_DAMAGE7
             + gamma_3 * F_POS7
             + gamma_4 * F_TREND
             + gamma_5 * F_REGIME

M2 回答: 在 D1 结构已知后, 最近价格路径 (MOM7/DAMAGE7/POS7)、中期趋势
(TREND) 和二/三板 regime (REGIME) 是否提供增量预测信息。

## 未来训练算法 (只记录, 本阶段禁止执行)

- L2 Logistic Regression: C = 1.0, solver = lbfgs, fit_intercept = True,
  class_weight = None
- 目标函数:

      min_theta [ -sum_i ( y_i*log(p_i) + (1-y_i)*log(1-p_i) ) + lambda * ||theta_{-0}||_2^2 ]

- 训练/验证: M0/M1/M2 expanding-date walk-forward (下一阶段)

## June/July 命名约定

- June: development / reference X sample
- July: retrospective unlabeled X-stability slice
  (July 不是 blind holdout; 本阶段只做 X 分布稳定性, 不读取 July Target)

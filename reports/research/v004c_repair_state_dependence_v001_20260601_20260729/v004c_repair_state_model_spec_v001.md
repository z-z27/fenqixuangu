# v004c Repair-State Model v001 — 模型规格 (R0/R1/R2 数学结构 — 仅记录, 不拟合)

- 阶段: Repair-State v001 规格 + 纯 X 结构审计产物; 本规格只定义公式与未来训练算法
- 状态: 由 v004c_repair_state_review_v001.md 结论背书 (X 结构 / 数值稳定性 / June→July
  X 稳定性; 与 Target 无关)
- 注意: 这是 X spec freeze 候选, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## Post-June hypothesis 声明

> Repair-State v001 是在 June walk-forward 失败 (REJECT_NO_STABLE_MODEL) 之后形成的
> post-June hypothesis。因此 **June 不能再作为该模型的独立验证集**。
> 后续流程: X spec freeze -> June development fit -> freeze coefficients/spec
> -> July retrospective OOT -> August+ forward shadow。
> 不得声称 June 可以验证新模型。

## 交易逻辑

2板/3板连续强势 -> 首次断板 -> D1 首次明显分歧 -> 判断分歧是否充分 -> 判断卖压是否
仍持续 -> 判断低位是否重新获得承接 -> 预测 D2-D3 是否发生强修复。

## Factor 定义与预处理 (future walk-forward)

- 每个 training fold 内拟合 (FOLD_CLIP_Z 顺序: 在 fold 原始值上拟合 q01/q99 -> clip
  -> 在 clipped fold 值上拟合 mu/sigma) + (x - mu)/sigma (fold 参数; 禁止全样本参数)
- composite: 由标准化 primitive 构造 G, 在 fold 上拟合 G 的 mu/sigma -> z(G)
- derived terms: 由标准化 base factor 构造 raw (F_D^2 / F_D*F_Q / F_D*F_S), 在 fold
  上拟合 raw 的 mu/sigma -> 标准化; **不做第二次 clip**
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## R0 — BASE RATE

    logit(p) = alpha

未来实现等价于 training prevalence。

## R1 — BASE REPAIR STATE (4 factors + intercept)

    logit(p) = alpha
             + beta_O  * F_OPEN
             + beta_D  * F_DIVERGENCE
             + beta_S  * F_SUPPLY
             + beta_Q  * F_RECLAIM

## R2 — CONDITIONAL REPAIR STATE (7 factors + intercept)

    logit(p) = alpha
             + beta_O * F_OPEN
             + beta_D * F_DIVERGENCE
             + beta_S * F_SUPPLY
             + beta_Q * F_RECLAIM
             + gamma_1 * F_DIVERGENCE_SQ
             + gamma_2 * F_DIVERGENCE_X_RECLAIM
             + gamma_3 * F_DIVERGENCE_X_SUPPLY

## 预声明 derived terms (第一版 v001 只允许这 3 个)

- DIVERGENCE_SQ = z(F_D^2): 允许"分歧存在最佳区间", 而非强制单调
- DIVERGENCE_X_RECLAIM = z(F_D * F_Q): 大分歧后是否真正出现重新承接 (核心条件项)
- DIVERGENCE_X_SUPPLY = z(F_D * F_S): 充分分歧时尾盘供应是否仍持续
- 禁止自动增加: OPEN_X_DIVERGENCE / HIGHZONE_X_DIVERGENCE / HIGHZONE_X_RECLAIM /
  RECLAIM_SQ / SUPPLY_SQ (不属于当前 v001 模型)

## Hierarchy 原则 (R2 必须遵守)

- 存在 DIVERGENCE_X_RECLAIM => DIVERGENCE 与 RECLAIM 必须同时存在
- 存在 DIVERGENCE_X_SUPPLY => DIVERGENCE 与 SUPPLY 必须同时存在
- 存在 DIVERGENCE_SQ => DIVERGENCE 必须存在

## Legacy sensitivity (禁止进入 R1/R2 core)

HIGHZONE / MOM7 / DAMAGE7 / REGIME / POS7 / TREND — 统一记录为 LEGACY_SENSITIVITY;
历史定义保留在 v004c_factor_spec.py (v002 规格), 本版本不删除。

## 未来训练算法 (只记录, 本阶段禁止执行)

- sklearn LogisticRegression: penalty = l2, C = 1.0, solver = lbfgs,
  fit_intercept = True, class_weight = None, max_iter = 1000
- 目标函数 (beta = non-intercept coefficients; alpha/intercept is not penalized):

      min_{alpha, beta} [ -sum_i ( y_i*log(p_i) + (1-y_i)*log(1-p_i) ) + lambda * ||beta||_2^2 ]

- Target (未来阶段): target7_daily_d2open_d3high = 1[High_D3/Open_D2 - 1 >= 0.07];
  本阶段禁止读取该 Target

## June/July 命名约定

- June: development / reference X sample (2026-06-01 ~ 2026-06-30)
- July: retrospective unlabeled X-stability slice (2026-07-01 ~ 2026-07-29,
  只 apply June reference 参数, 不读取 July Target)

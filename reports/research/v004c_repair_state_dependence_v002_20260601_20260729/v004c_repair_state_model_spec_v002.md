# v004c Repair-State Model v002 — 模型规格 (R0/R1/R2 数学结构 — 仅记录, 不拟合)

- 阶段: Repair-State v002 语义修订 + 纯 X 结构审计产物; 本规格只定义公式与未来训练算法
- 状态: 由 v004c_repair_state_review_v002.md 结论背书 (X 结构 / 数值稳定性 / June→July
  X 稳定性; 与 Target 无关)
- 注意: 这是 X spec freeze 候选, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## Post-June hypothesis 声明

> Repair-State v002 是在 June walk-forward 失败 (REJECT_NO_STABLE_MODEL) 之后形成的
> post-June hypothesis (v001 基础上的语义修订)。因此 **June 不能再作为该模型的独立
> 验证集**。
> 后续流程: X spec freeze -> June development fit -> freeze coefficients/spec
> -> July retrospective OOT -> August+ forward shadow。
> 不得声称 June 可以验证新模型。

## 交易逻辑

2/3板连续强势 -> D1 首次断板 -> 发生分歧 -> 分歧不能不足, 也不能过度透支 ->
最终损伤必须可控 -> 尾盘供应最好衰竭 -> 分歧后需要出现真实重新承接 -> D2-D3 强修复。

三个状态 factor 是不同阶段 (过程分歧 / 最终损伤 / 弱点后回收):

- DIVERGENCE = D1 盘中实际发生过多大的首次分歧 (过程强度)
- CLOSE_DAMAGE = 这次分歧结束以后, D1 收盘最终留下多少结构损伤
- RECLAIM = D1 出现弱点以后, 资金有没有重新把价格接回来

## Factor 定义与预处理 (future walk-forward)

- 每个 training fold 内拟合 (FOLD_CLIP_Z 顺序: 在 fold 原始值上拟合 q01/q99 -> clip
  -> 在 clipped fold 值上拟合 mu/sigma) + (x - mu)/sigma (fold 参数; 禁止全样本参数)
- composite (CLOSE_DAMAGE / RECLAIM): 由标准化 primitive 构造 G, 在 fold 上拟合 G 的
  mu/sigma -> z(G)
- derived terms: 由标准化 base factor 构造 raw (F_V^2 / F_V*F_Q / F_V*F_C), 在 fold
  上拟合 raw 的 mu/sigma -> 标准化; **不做第二次 clip**
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## R0 — BASE RATE

    logit(p) = alpha

未来实现等价于 training prevalence。

## R1 — BASE REPAIR STATE (5 factors + intercept)

    logit(p) = alpha
             + beta_O * F_OPEN
             + beta_V * F_DIVERGENCE
             + beta_C * F_CLOSE_DAMAGE
             + beta_S * F_SUPPLY
             + beta_Q * F_RECLAIM

## R2 — CONDITIONAL REPAIR STATE (8 factors + intercept)

    logit(p) = alpha
             + beta_O * F_OPEN
             + beta_V * F_DIVERGENCE
             + beta_C * F_CLOSE_DAMAGE
             + beta_S * F_SUPPLY
             + beta_Q * F_RECLAIM
             + gamma_1 * F_DIVERGENCE_SQ
             + gamma_2 * F_DIVERGENCE_X_RECLAIM
             + gamma_3 * F_DIVERGENCE_X_DAMAGE

## 预声明 derived terms (v002 只允许这 3 个)

- DIVERGENCE_SQ = z(F_V^2): 分歧可能存在最佳区间 (太小 -> 浮筹释放不足; 适度 ->
  健康换手; 极端 -> 大量兑现/承接消耗/修复预算透支); 若未来 linear DIV > 0 且
  DIV_SQ < 0 可形成倒 U, 但禁止用 Target 验证方向
- DIVERGENCE_X_RECLAIM = z(F_V * F_Q): 同样大分歧后, 有真实资金重新承接与没有
  完全不同 (DIV高+Q高 = 强回收; DIV高+Q低 = 缺乏承接)
- DIVERGENCE_X_DAMAGE = z(F_V * F_C): 大分歧最终是剧烈换手还是实际留下严重结构
  破坏 (DIV高+C低 = 过程剧烈但损伤可控; DIV高+C高 = 获利盘兑现 + 承接消耗 +
  修复预算透支)
- 禁止自动增加: OPEN_X_DIVERGENCE / DIV_X_TURNOVER / RECLAIM_SQ / SUPPLY_SQ /
  CLOSE_DAMAGE_SQ (不属于当前 v002 模型)

## Hierarchy 原则 (R2 必须遵守)

- 存在 DIVERGENCE_SQ => DIVERGENCE 必须存在
- 存在 DIVERGENCE_X_RECLAIM => DIVERGENCE 与 RECLAIM 必须同时存在
- 存在 DIVERGENCE_X_DAMAGE => DIVERGENCE 与 CLOSE_DAMAGE 必须同时存在

## Sensitivity (禁止进入 R1/R2 core)

- DIVERGENCE_X_SUPPLY: v001 derived 定义保留, v002 降为 SENSITIVITY (避免
  interaction 膨胀; 优先验证'分歧过度''分歧后回收''分歧是否留下真实损伤')
- TURNOVER_COST = z(break_volume_ratio_vs_board_days): 未来机制预声明 (同样的价格
  分歧, 极端放量 => 更大筹码兑现和修复预算消耗); 本版本禁止构造 DIV_X_TURNOVER
- LEGACY_SENSITIVITY: HIGHZONE / MOM7 / DAMAGE7 / REGIME / POS7 / TREND
  (历史定义保留在 v004c_factor_spec.py, 本版本不删除)

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

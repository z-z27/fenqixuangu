# v004c 研究 — 现有模型资料清单 (existing_model_inventory)

本清单只记录路径与摘要,不修改任何文件。2026-08-03 只读调查。

## v004a(加权 logistic walk-forward)

| 项 | 路径 | 摘要 |
|---|---|---|
| 冻结策略配置 | `configs/policy_v005_v1.json` | policy_version `v005.1-frozen-20260626`;policy_id `policy_v005_v002_regime_fallback`;deployment_status `shadow_only`;research_only=true;target=target7_d2open_d3high(7.0%);selector grid_id=4, candidate_top_k=15, top_n=3;fallback gate 参数;readiness min_forward_dates=30 |
| 冻结系数(当前 daily 使用) | `configs/models/v004a_coefficients_2026-06-26.csv` | 19 行(intercept + 18 特征);model_id `logistic_v004a_weighted`;fold_index=34;train 2026-05-07..06-25;predict_date 2026-06-26;l2=0.30;positive_weight=1.5;normalized_sha256 记录于 policy json: `78fe05032ca0efd6bc931d0766f6427923ae52168a291c47c5e02145575b2117` |
| v004a 模块 | `src/v004a.py` | 特征定义(BASE_RANK_SPECS/OPTIONAL_RANK_SPECS/交互项/day buckets)、eligibility、predict_logistic、walk-forward 训练与评分 |
| v004a 研究输出 | `reports/v004a/` (grid_v1, grid_v2_scored, smoke*, smoke_scored, smoke_v1) | grid_v2_scored/v004a_scored_candidates.csv 含 06-02..06-29 全网格(l2×positive_weight×scope)评分;冻结参数行(l2=0.3, pw=1.5, walk_forward)为本数据集比较字段来源之一 |
| v004a 历史验证包 | `reports/v004a_grid_v1_review_pack.zip`, `reports/v004a_grid_v2_scored_review_pack.zip`, `reports/model_review_d2open_d3high_pack.zip` | 归档研究产物 |

## v002(manual ranking model)

| 项 | 路径 | 摘要 |
|---|---|---|
| v002 模型文件 | `reports/manual_models/ranking_model_v002_core_momentum_support.json` | model_id `ranking_model_v002_core_momentum_support`;normalized_sha256 记录于 policy json: `f4a0eaaf60341e4a9a3e729aed879c4a4d84b3dbf107282957493b252f7e07f3` |
| v001 模型文件 | `reports/manual_models/ranking_model_v001_core_momentum.json` | 早期版本,仅对照 |
| v002 daily 输出 | `reports/daily_signals/signals_2026-06-25.csv`, `signals_2026-07-01..07-31.csv` | 含 research_score/daily_rank(06-25 文件为旧格式无 rank);rank 以文件内 trade_date 为准 |
| v002 排序验证 | `src/ranking_backtest.py`, `src/daily_ranking.py` | 验证与 daily 应用模块 |
| v002 历史验证 | `reports/ranking_backtests/`, `reports/rb_v002_d2d3_top3/` | 研究产物 |

## 旧 v004b(pairwise ranking 分支,非主线)

| 项 | 路径 | 摘要 |
|---|---|---|
| v004b 模块 | `src/v004b.py` | pairwise ranking 研究分支 |
| v004b 报告 | `reports/v004b/` | smoke;smoke_target_binary_compat;smoke_use_source_features;top10_union_v002_smoke_v1;top10_v004a_only_smoke_v1;v004b1_return_gap_top10_v004a_only;v004b1_return_gap_top15_gap1_hard5;v004b1_return_gap_top15_union_no_source;v004b1_return_gap_top15_v004a_only;v004b1_target_binary_top15_v004a_only |
| v004b 验证包 | `reports/v004b_smoke_review_pack.zip` | 归档 |

## v005 / v5a

| 项 | 路径 | 摘要 |
|---|---|---|
| v005 模块 | `src/v005_set_selector.py`, `src/v005_fixed_grid_holdout.py`, `src/v005_daily_selector.py`, `src/v005_objective_sweep.py`, `src/v005_failure_attribution.py`, `src/v005_fallback_gate.py`, `src/run_daily_v005.py` | Top3 set-level 组合选择器 + fixed-grid holdout + daily shadow flow + fallback gate |
| v005 daily 输出 | `reports/daily_v005/2026-07-02..2026-07-31/` | 每日 scored candidates(冻结 06-26 折)、selection、decision、run_meta |
| fixed-grid holdout | `reports/v005_fixed_grid_holdout_2026-06-26_2026-06-30/`, `..._2026-07-01_2026-07-03/`, `..._2026-06-26_2026-07-08*/`, `..._2026-07-09_2026-07-14_forward_v1/`, `..._2026-07-15_2026-07-17_forward_v1/`, `..._2026-07-20_2026-07-22_forward_v1/`, `..._2026-07-23_2026-07-29_forward_v1/` | 冻结策略 forward 验证(不重新训练/选 grid/调 fallback) |
| fallback gate | `reports/v005_fallback_gate/` | v005_fallback_gate_daily/summary/report/replacement |
| **v5a 说明** | — | **仓库中不存在名为 "v5a" 的代码、配置、文档或报告**(全库 grep `v5a|v_5a|V5A` 无命中)。最接近的是 v005 冻结策略链(policy_v005_v1.json + src/v005_*.py + reports/v005_* + reports/daily_v005)。若 "v5a" 指 v005 冻结政策,即上表内容。 |

## 其他相关资产

| 项 | 路径 | 摘要 |
|---|---|---|
| 历史样本 | `reports/history_samples/2026-05-06_2026-06-29/history_candidates_2026-05-06_2026-06-29.csv` 等 | D1 信号全池 + target7_d2open_d3high 标签(本项目标签交叉核对基准) |
| daily v005 报告 | `reports/daily_v005/*/v005_daily_report_*.md` | 每日研究观察清单 |
| 手动模型验证包 | `reports/model_validation_2026-05-06_2026-06-29.zip` 等 | 归档 |

## 版本/哈希摘要

- 本数据集读取的冻结资产哈希: v004a 系数 sha256 `78fe05...`(policy json 记录);v002 模型 sha256 `f4a0ea...`(policy json 记录)。
- 本数据集 git commit: `9f45af1f0ab8f506fb1cd41939e2c2985d0d9671`(research-sample-analysis 分支)。
- 本任务未修改上述任何文件。

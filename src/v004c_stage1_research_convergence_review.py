"""Static evidence synthesis for the v004c Stage1 research convergence review.

This module deliberately performs no model fitting, outcome calculation, feature
screen, or external data access.  It turns already-finalized research reviews
into a normalized route inventory and a single convergence decision.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TASK_ID = "v004c_stage1_research_convergence_review_v001"
OUTPUT_DIR = Path("reports/research") / TASK_ID

CURRENT_STAGE1_BOTTLENECK = (
    "现有D1-known表征无法跨时间稳定地区分Target7与LOSS，尤其在Rank2/Rank3发生winner-loss混排。"
)
MODEL_FORM_LIMITATION = "NOT_SUPPORTED"
CURRENT_INFORMATION_LIMITATION = "SUPPORTED"
BROADER_D1_INFORMATION_LIMITATION = "[待核验]"
HIGH_VALUE_UNTESTED_INFORMATION_FAMILY = "PRE_BREAK_BOARD_DAY_SEAL_PATH"
STAGE1_RESEARCH_STATE = "ONE_NARROW_INFORMATION_AUDIT_JUSTIFIED"
NEXT_ACTION = "BLIND_PRE_BREAK_BOARD_DAY_SEAL_PATH_INFORMATION_AUDIT"
AUGUST_HOLDOUT_STATUS = "CONSUMED"

S2_FEATURES = (
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
)


def _src(relative: str) -> str:
    return f"reports/research/{relative}"


ROUTES: tuple[dict[str, str], ...] = (
    {
        "route_name": "A_18F_ARCHITECTURE_TRANSFER",
        "question": "Frozen v4a architecture是否能在v004c候选池内工作？",
        "problem": "建立current Stage1并验证旧18F架构迁移。",
        "experiment_done": "YES",
        "data_window": "2026-05-06..2026-06-30; strict June",
        "model_or_information_change": "在v004c universe内复用18F weighted L2 Logistic架构。",
        "key_result": "ARCHITECTURE_TRANSFER_SIGNAL=PARTIAL；Rank1略高于universe，但Top3低0.80pp。",
        "formal_state": "PARTIAL",
        "what_confirmed": "18F语义和训练实现可严格迁移；有少量Rank1信息。",
        "what_not_confirmed": "Top3 alpha、低机会日保护、可进入July OOT均未确认。",
        "why_stopped": "Top3与机会分层门未通过。",
        "current_authorization": "仅保留历史参考；禁止继续调18F。",
        "failure_types": "TEMPORAL_INSTABILITY|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_v4a_architecture_transfer_v001_20260506_20260630/v004c_v4a_transfer_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "ORIGINAL_V4A_DIRECT_TRANSFER",
        "question": "original v4a完整横截面先打分再交集是否更好？",
        "problem": "检查current v4c重训是否破坏原始横截面上下文。",
        "experiment_done": "NO",
        "data_window": "May-June provenance audit",
        "model_or_information_change": "无；只做provenance gate。",
        "key_result": "Archived folds均含尚未成熟标签；coverage 85.16%；实效比较未运行。",
        "formal_state": "INVALID",
        "what_confirmed": "现有archived original-v4a不满足严格label-availability contract。",
        "what_not_confirmed": "original-v4a direct transfer优劣、Top2/Top3均未验证。",
        "why_stopped": "严格provenance失败，且probe发生July access breach。",
        "current_authorization": "禁止把该路线表述为已比较；除非出现新的严格归档资产。",
        "failure_types": "LINEAGE_LIMITATION|DATA_LIMITATION",
        "evidence_sources": _src("v004c_original_v4a_direct_transfer_top2_top3_v001_20260506_20260626/v004c_original_v4a_direct_transfer_review_v001.md"),
        "conflict_note": "该实验是INVALID而不是negative model result。",
    },
    {
        "route_name": "B_REDUCED7F_CONTRIBUTION_ABLATION",
        "question": "删除legacy timing/total/interactions是否改善18F？",
        "problem": "归因旧v4a结构是否拖累v004c。",
        "experiment_done": "YES",
        "data_window": "2026-05-06..2026-07-29 mature",
        "model_or_information_change": "不重训，仅将预注册feature contribution置零。",
        "key_result": "A3跨月AUC/return同向小幅改善，但risk不一致；结论MIXED。",
        "formal_state": "MIXED",
        "what_confirmed": "legacy interactions/timing不是稳定、单调的收益来源；简化存在弱信号。",
        "what_not_confirmed": "不能仅凭ablation授权Reduced7F训练。",
        "why_stopped": "改善非单调且增加净LOSS slot。",
        "current_authorization": "ablation本身结束；只保留对7F预注册来源的证据。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED|TEMPORAL_INSTABILITY",
        "evidence_sources": _src("v004c_frozen_stage1_contribution_ablation_audit_v001_20260506_20260731/v004c_stage1_ablation_review_v001.md"),
        "conflict_note": "后续独立预注册Reduced7F temporal validation提供了更强证据。",
    },
    {
        "route_name": "B_REDUCED7F_TEMPORAL_VALIDATION",
        "question": "单一7F Logistic challenger能否跨June/July改善？",
        "problem": "减少18F legacy structure并保持chronological validation。",
        "experiment_done": "YES",
        "data_window": "May->June; May+June->July; expanding OOF",
        "model_or_information_change": "仅预注册7F；原训练权重和L2=.30。",
        "key_result": "June/July Top3 excess分别改善0.25pp/0.17pp；MODEL_STATE=SUPPORTED。",
        "formal_state": "REDUCED7F_TEMPORAL_SIGNAL_SUPPORTED",
        "what_confirmed": "7F比current18F更稳定，值得进入一次训练规格sanity。",
        "what_not_confirmed": "绝对alpha和tail改善仍有限，不等于成功模型。",
        "why_stopped": "单一challenger验证完成；不允许继续subset search。",
        "current_authorization": "作为S2形成路径的历史依据；不是当前成功模型。",
        "failure_types": "NONE",
        "evidence_sources": _src("v004c_reduced7f_stage1_temporal_validation_v001_20260506_20260731/v004c_reduced7f_review_v001.md"),
        "conflict_note": "后续S2 attribution将剩余瓶颈定为information limit；两者是阶段性递进而非冲突。",
    },
    {
        "route_name": "C_TRAINING_SPEC_SANITY",
        "question": "关闭tail bonus并降低L2能否修正7F训练规格？",
        "problem": "检查L2=.30欠拟合和高收益额外权重与7%业务目标不一致。",
        "experiment_done": "YES",
        "data_window": "May-July mature; fixed June/July temporal folds",
        "model_or_information_change": "固定S0/S1/S2/S3；最终支持S2 no-tail L2=.10。",
        "key_result": "S2训练分离与June/July方向均优于S0；TRAINING_SPEC_REPAIR_SUPPORTED。",
        "formal_state": "TRAINING_SPEC_REPAIR_SUPPORTED",
        "what_confirmed": "tail bonus语义不匹配；L2=.30过强；S2是较合理reference。",
        "what_not_confirmed": "S2仍未建立稳定正Top3 alpha，S3训练更好但forward不稳。",
        "why_stopped": "四个预注册规格完成，明确禁止继续参数搜索。",
        "current_authorization": "S2仅作reference baseline；L2/tail/positive weight不再调。",
        "failure_types": "MODEL_SPEC_FAILURE_RESOLVED|RESIDUAL_INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_reduced7f_training_spec_sanity_v001_20260506_20260731/v004c_reduced7f_training_spec_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "D_S2_REFERENCE",
        "question": "当前最合理7F规格能否作为成功Stage1？",
        "problem": "建立后续failure attribution的固定参照。",
        "experiment_done": "YES",
        "data_window": "May-July mature, 485 rows / 60 dates",
        "model_or_information_change": "7F; weighted Logistic; L2=.10; positive_weight=1.50; tail OFF。",
        "key_result": "T7-vs-LOSS pair concordance 55.14%；Top3 49个LOSS slot中25个可替换。",
        "formal_state": "REFERENCE_BASELINE_NOT_SUCCESSFUL_MODEL",
        "what_confirmed": "训练规格较S0合理，可用于一致诊断。",
        "what_not_confirmed": "未确认稳定Top3 selection alpha或winner-loss separation。",
        "why_stopped": "后续归因指向information limit。",
        "current_authorization": "仅REFERENCE BASELINE；禁止继续调参/子集。",
        "failure_types": "INFORMATION_FAILURE|TEMPORAL_INSTABILITY",
        "evidence_sources": _src("v004c_s2_winner_vs_loss_failure_attribution_v001_20260506_20260731/v004c_s2_winner_loss_failure_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "CURRENT_18F_TOP2_JULY_FORWARD",
        "question": "current18F预声明Top2能否在July延续？",
        "problem": "检验Top2 capacity与Stage1 forward alpha。",
        "experiment_done": "YES",
        "data_window": "July 23 dates / 178 rows",
        "model_or_information_change": "无；复用冻结July predictions。",
        "key_result": "Top2 capped 1.59%低于universe 1.99%；PRIMARY_FORWARD_FAILURE=STAGE1_SELECTION_ALPHA_FAILURE。",
        "formal_state": "TOP2_CAPACITY_FORWARD_SIGNAL_PARTIAL",
        "what_confirmed": "Top2相对Top3方向仍在但优势很弱。",
        "what_not_confirmed": "Stage1未击败universe，不能冻结Top2核心架构。",
        "why_stopped": "July forward gates失败并要求REOPEN_STAGE1_QUESTION。",
        "current_authorization": "仅历史forward evidence；禁止以Top2掩盖alpha failure。",
        "failure_types": "FORWARD_FAILURE|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_frozen_stage1_top2_july_forward_stress_v001_20260701_20260731/v004c_top2_july_forward_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "J_WINNER_VS_LOSS_ATTRIBUTION",
        "question": "S2没分开Target7/LOSS是组合错误还是信息不足？",
        "problem": "定位Rank2/Rank3 winner-loss confusion。",
        "experiment_done": "YES",
        "data_window": "May-July mature",
        "model_or_information_change": "无；只使用冻结S2/7F和outcomes。",
        "key_result": "321同日pairs concordance 55.14%；raw不普遍强于rank；归因INFORMATION_LIMIT_DOMINANT。",
        "formal_state": "INFORMATION_LIMIT_DOMINANT",
        "what_confirmed": "候选有winner，但7F几何/单因子稳定信息不足，LOSS获相同正贡献。",
        "what_not_confirmed": "没有证据表明换复杂组合即可解决。",
        "why_stopped": "failure attribution完成且不授权模型。",
        "current_authorization": "保留为当前核心瓶颈证据。",
        "failure_types": "INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_s2_winner_vs_loss_failure_attribution_v001_20260506_20260731/v004c_s2_winner_loss_failure_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "REPLACEABLE_LOSS_CASE_PACK",
        "question": "可替换LOSS错误案例能否整理用于机制复核？",
        "problem": "把25个可替换LOSS相关原始D1资料打包。",
        "experiment_done": "YES",
        "data_window": "May-July through 2026-07-29",
        "model_or_information_change": "无；数据准备。",
        "key_result": "CASE_PACK_STATE=READY_FOR_MECHANISM_REVIEW。",
        "formal_state": "READY_FOR_MECHANISM_REVIEW",
        "what_confirmed": "错误案例的日线、5分钟、7F、53F可追溯。",
        "what_not_confirmed": "case pack本身不证明任何普遍因子。",
        "why_stopped": "数据包准备任务完成。",
        "current_authorization": "仅机制参考；不能当作feature test。",
        "failure_types": "NONE",
        "evidence_sources": _src("v004c_s2_replaceable_loss_case_pack_v001_20260506_20260729/v004c_s2_case_pack_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "K_53F_D1_INFORMATION_GBDT",
        "question": "同一53F是否存在可泛化非线性Target7信息？",
        "problem": "区分线性表达限制与时间不稳定/信息限制。",
        "experiment_done": "YES",
        "data_window": "May-June; 319 rows / 39 dates; 29 OOF dates",
        "model_or_information_change": "固定GBDT capacity diagnostic与Ridge对照。",
        "key_result": "GBDT train AUC .882但OOF .535；Top3 2.69%低于baseline 3.00%；May/June AUC .422/.577。",
        "formal_state": "NONLINEAR_OOF_SIGNAL_PRESENT_NO",
        "what_confirmed": "53F可训练内拟合，Rank1有部分信号，但Top3不能OOF泛化。",
        "what_not_confirmed": "更复杂模型并未解决问题；不能推断永久无信息。",
        "why_stopped": "固定nonlinear OOF gate失败。",
        "current_authorization": "禁止继续model zoo；作为当前信息限制证据。",
        "failure_types": "TEMPORAL_INSTABILITY|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_upside_predictability_diagnostic_v001_20260506_20260630/v004c_upside_predictability_diagnostic_review_v001.md"),
        "conflict_note": "报告Q3称stable D1 info=YES仅基于train gate，同时PRIMARY_DIAGNOSIS=TEMPORAL_INSTABILITY；本review以后者和OOF为准。",
    },
    {
        "route_name": "TOP3_FAILURE_DECOMPOSITION",
        "question": "Top3 regret主要来自winner capture还是sub-7 repair ordering？",
        "problem": "定位最终Top3失败层。",
        "experiment_done": "YES",
        "data_window": "May-June OOF 29 dates",
        "model_or_information_change": "无；读取Ridge/GBDT OOF。",
        "key_result": "73%-80% regret来自winner capture；within-nontarget约45%-47%。",
        "formal_state": "TOP3_FAILURE_PICTURE_WINNER_CAPTURE",
        "what_confirmed": "主瓶颈是没抓到>=7% winner，repair fill是次要但真实问题。",
        "what_not_confirmed": "改目标不能自动解决winner识别。",
        "why_stopped": "归因完成。",
        "current_authorization": "保留为pipeline bottleneck证据。",
        "failure_types": "INFORMATION_FAILURE|OBJECTIVE_MISALIGNMENT_SECONDARY",
        "evidence_sources": _src("v004c_top3_failure_decomposition_v001_20260506_20260630/v004c_top3_failure_decomposition_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "G_PAIRWISE_RIDGE_53F",
        "question": "same-date pairwise Ridge能否提升upside/风险排序？",
        "problem": "用排序目标替代点式binary。",
        "experiment_done": "YES",
        "data_window": "May-June; 29 OOF dates",
        "model_or_information_change": "53F pairwise Ridge；历史lambda选择。",
        "key_result": "Upside AUC .505且Top3低于baseline；tail dev signal一度PRESENT。",
        "formal_state": "UPSIDE_READY_FOR_JULY_OOT_NO",
        "what_confirmed": "简单pairwise形式未恢复upside排序。",
        "what_not_confirmed": "tail模型未在该任务跑July。",
        "why_stopped": "upside gate失败；后续risk路线也未建立。",
        "current_authorization": "禁止重复同一53F pairwise upside搜索。",
        "failure_types": "MODEL_FORM_NOT_SUFFICIENT|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_pairwise_ridge_v001_20260506_20260630/v004c_pairwise_ridge_review_v001.md"),
        "conflict_note": "tail dev信号不能覆盖后续risk foundation/protector的ABSENT结论。",
    },
    {
        "route_name": "G_CORRECTED_BINARY_PAIRWISE",
        "question": "修正same-date binary pair定义后是否有dev signal？",
        "problem": "排除历史pair构造问题。",
        "experiment_done": "YES",
        "data_window": "May-June; 29 OOF dates",
        "model_or_information_change": "同日binary pairs，固定lambda选择规则。",
        "key_result": "AUC .4993；Top1/Top3低于baseline。",
        "formal_state": "CORRECTED_BINARY_DEV_SIGNAL_ABSENT",
        "what_confirmed": "pair定义修正没有恢复selection alpha。",
        "what_not_confirmed": "未进入July。",
        "why_stopped": "全部主要dev gates失败。",
        "current_authorization": "禁止重复。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_binary_same_date_pairwise_ridge_v002_20260506_20260630/v004c_binary_vs_repair_review_v002.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "G_RAW_REPAIR_PAIRWISE",
        "question": "学习完整raw repair ordering能否改善Top3？",
        "problem": "修复binary 7%以下信息丢失。",
        "experiment_done": "YES",
        "data_window": "May-June; 29 OOF dates",
        "model_or_information_change": "pairwise target改为raw repair strength。",
        "key_result": "Top3 2.11%低于binary 2.55%与baseline 3.00%；all-repair concordance 47.93%。",
        "formal_state": "REPAIR_OBJECTIVE_DEV_SIGNAL_ABSENT",
        "what_confirmed": "单纯换raw repair objective不能修复Top3。",
        "what_not_confirmed": "不代表任何未来目标工程都失败。",
        "why_stopped": "dev gates失败。",
        "current_authorization": "禁止重复raw repair pairwise路线。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_repair_pairwise_ridge_v001_20260506_20260630/v004c_repair_pairwise_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "G_PAIR_CAPPED7_DEVELOPMENT",
        "question": "保留sub-7 severity但不奖励>7%能否改善Top10->Top3？",
        "problem": "目标信息粒度与业务cap对齐。",
        "experiment_done": "YES",
        "data_window": "May-June strict development",
        "model_or_information_change": "固定3-predictor same-date pairwise CAPPED7。",
        "key_result": "Development Top3 3.04%，高于control 1.85%；TARGET_INFORMATION_SIGNAL=SUPPORTED。",
        "formal_state": "TARGET_INFORMATION_SIGNAL_SUPPORTED",
        "what_confirmed": "development内sub-7信息有价值；>7额外信息有害。",
        "what_not_confirmed": "不构成forward成功。",
        "why_stopped": "进入一次July forward stress。",
        "current_authorization": "仅作为PAIR_CAPPED7 forward路线的历史阶段。",
        "failure_types": "NONE",
        "evidence_sources": _src("v004c_top10_target_information_v001_20260506_20260630/v004c_top10_target_info_review_v001.md"),
        "conflict_note": "后续July将强development结果降为PARTIAL。",
    },
    {
        "route_name": "G_PAIR_CAPPED7_JULY_FORWARD",
        "question": "冻结PAIR_CAPPED7能否在July复现？",
        "problem": "验证Top10 rerank的forward alpha。",
        "experiment_done": "YES",
        "data_window": "July 23 dates / 178 rows",
        "model_or_information_change": "无；Stage1/Stage2各fit once pre-July。",
        "key_result": "Top3较control +0.26pp但仍低于universe -0.22pp；net T7 slots -1。",
        "formal_state": "FORWARD_STRESS_SIGNAL_PARTIAL",
        "what_confirmed": "部分downside改善。",
        "what_not_confirmed": "未恢复selection alpha，PRIMARY_FAILURE=STAGE2_RERANK_FAILURE。",
        "why_stopped": "forward主要门失败。",
        "current_authorization": "禁止重复/扩展PAIR_CAPPED7 Stage2。",
        "failure_types": "FORWARD_FAILURE|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_pair_capped7_july_forward_v001_20260701_20260731/v004c_july_forward_review_v001.md"),
        "conflict_note": "解决了development SUPPORTED与forward不稳定的证据演化。",
    },
    {
        "route_name": "E_BOARD2_BOARD3_STRUCTURE",
        "question": "Board3是否天然更差，或被Stage1过度提拔？",
        "problem": "解释Top3 LOSS contamination。",
        "experiment_done": "YES",
        "data_window": "May-June matured; strict 17 dates",
        "model_or_information_change": "无；结构审计。",
        "key_result": "Board3总体不天然更差，但Top3 share lift 1.82x且LOSS 53.33%；STAGE1_BOARD3_OVERPROMOTION。",
        "formal_state": "STAGE1_BOARD3_OVERPROMOTION",
        "what_confirmed": "Board3 overpromotion是局部问题。",
        "what_not_confirmed": "不能把全部Rank3/Stage1失败归咎于Board3。",
        "why_stopped": "结构只支持诊断，不支持直接排除/阈值。",
        "current_authorization": "仅参考；Board2/Board3规则不得新增。",
        "failure_types": "PARTIAL_STRUCTURAL_FAILURE",
        "evidence_sources": _src("v004c_board2_board3_structural_audit_v001_20260506_20260626/v004c_board2_board3_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "E_BOARD3_FEATURE_RESPONSE",
        "question": "哪些18F contribution在Board3中方向错位？",
        "problem": "定位overpromotion机制。",
        "experiment_done": "YES",
        "data_window": "strict May-June Board3 selected surface",
        "model_or_information_change": "无；feature-response审计。",
        "key_result": "三项贡献显示winner-harmful，但机制仅PARTIAL且样本弱。",
        "formal_state": "PARTIAL_FEATURE_MISALIGNMENT",
        "what_confirmed": "Board3存在部分mapping错位。",
        "what_not_confirmed": "不支持最小board interaction或广泛分离。",
        "why_stopped": "harmful share与独立性不足。",
        "current_authorization": "只保留机制参考。",
        "failure_types": "SELECTION_BIAS_RISK|DATA_LIMITATION",
        "evidence_sources": _src("v004c_board2_board3_stage1_feature_response_v001_20260603_20260626/v004c_board_feature_response_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "E_BOARD3_HARMFUL_ABLATION",
        "question": "删除三项Board3 harmful contributions能否安全去坏？",
        "problem": "验证贡献归因是否可转化为safe ablation。",
        "experiment_done": "YES",
        "data_window": "strict May-June 17 dates",
        "model_or_information_change": "不重训；预注册Board3 contribution ablation。",
        "key_result": "删除93.33% Board3 slots，同时删除全部4个winner；Top3 -0.11pp。",
        "formal_state": "ABLATION_MECHANISM_SIGNAL_ABSENT",
        "what_confirmed": "贡献推动Board3但不具loss selectivity。",
        "what_not_confirmed": "不能形成Board3安全抑制策略。",
        "why_stopped": "winner伤害与LOSS抑制同样强。",
        "current_authorization": "禁止重复该Board3 ablation。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED",
        "evidence_sources": _src("v004c_board3_confirmed_harmful_ablation_v001_20260603_20260626/v004c_board3_ablation_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "E_BOARD3_ONLY_MODEL",
        "question": "Board3-only冻结18F架构能否改善内部排序？",
        "problem": "解除Board2/Board3 mapping冲突。",
        "experiment_done": "YES",
        "data_window": "May-June; 8 OOF dates / 19 rows",
        "model_or_information_change": "仅训练population改为Board3。",
        "key_result": "NONLOSS AUC .692但Target7 AUC .457；BOARD3_ONLY_SIGNAL=PARTIAL。",
        "formal_state": "BOARD3_ONLY_SIGNAL_PARTIAL",
        "what_confirmed": "小样本内有risk ordering迹象。",
        "what_not_confirmed": "practical/robustness gates不足。",
        "why_stopped": "support极小且多项gate失败。",
        "current_authorization": "NO_NEW_MODEL_YET；现收敛状态禁止恢复。",
        "failure_types": "DATA_LIMITATION|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_board3_only_v4a_arch_feasibility_v001_20260506_20260626/v004c_board3_only_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "E_BOARD3_RISK_SELECTIVITY",
        "question": "locked Board3 score能否去坏留好？",
        "problem": "为有限risk veto建立信息基础。",
        "experiment_done": "YES",
        "data_window": "8 dates / 19 rows",
        "model_or_information_change": "无；locked OOF diagnostic。",
        "key_result": "bottom-half gap +25pp但winner removal 50%；bootstrap P(gap>0)=58.51%。",
        "formal_state": "BOARD3_RISK_SELECTIVITY_SIGNAL_PARTIAL",
        "what_confirmed": "存在描述性risk ordering。",
        "what_not_confirmed": "不支持threshold/veto。",
        "why_stopped": "winner-loss pair支持极弱且robustness不足。",
        "current_authorization": "NO_BOARD3_RISK_POLICY_YET；禁止恢复。",
        "failure_types": "DATA_LIMITATION|SELECTION_BIAS_RISK",
        "evidence_sources": _src("v004c_board3_oof_risk_selectivity_v001_20260612_20260625/v004c_board3_risk_selectivity_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "E_BOARD3_HISTORY_EXTENSION",
        "question": "Jan-Apr扩历史能否增加support且不污染May-June？",
        "problem": "解除Board3小样本限制。",
        "experiment_done": "YES",
        "data_window": "Jan-June; 60 OOF dates / 105 rows",
        "model_or_information_change": "冻结模型，仅扩大最早训练日期。",
        "key_result": "support大增但shared-date selectivity gap .25->0；HISTORY_EXTENSION_EFFECT=HARMFUL，DRIFTED。",
        "formal_state": "REJECT_OLD_HISTORY_EXTENSION",
        "what_confirmed": "旧历史与近期Board3 mapping不兼容。",
        "what_not_confirmed": "不能通过删月份/滚动窗口修补。",
        "why_stopped": "反污染HARMFUL gate触发。",
        "current_authorization": "禁止Jan-Jun Board3扩历史及窗口搜索。",
        "failure_types": "TEMPORAL_INSTABILITY|FORWARD_FAILURE",
        "evidence_sources": _src("v004c_board3_historical_extension_temporal_compatibility_v001_20260101_20260630/v004c_board3_historical_extension_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "F_D1_RISK_INFORMATION_FOUNDATION",
        "question": "53F是否为Top3 LOSS提供稳定且winner-safe的风险信息？",
        "problem": "为risk protector建立低层信息基础。",
        "experiment_done": "YES",
        "data_window": "strict June Top3 51 rows / 17 dates",
        "model_or_information_change": "无；53F univariate/permutation/FDR。",
        "key_result": "无feature通过foundation/FDR；RISK_INFORMATION_FOUNDATION=NOT_ESTABLISHED。",
        "formal_state": "NOT_ESTABLISHED",
        "what_confirmed": "局部风险相关存在但多重检验与日期稳定性不足。",
        "what_not_confirmed": "没有至少两项非冗余stable risk source。",
        "why_stopped": "foundation gate失败。",
        "current_authorization": "停止D1-only risk protector路线。",
        "failure_types": "SELECTION_BIAS_RISK|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_stage1_top3_risk_information_v001_20260603_20260626/v004c_stage1_top3_risk_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "F_LIMITED_RISK_PROTECTOR",
        "question": "固定三特征loss model能否veto并backfill？",
        "problem": "在不改Stage1的情况下移除LOSS。",
        "experiment_done": "YES",
        "data_window": "strict June 17 dates",
        "model_or_information_change": "固定loss logistic与0.50阈值。",
        "key_result": "AUC .405；veto=0；headroom recovered 0%。",
        "formal_state": "RISK_PROTECTOR_SIGNAL_ABSENT",
        "what_confirmed": "现有三特征risk representation不可用。",
        "what_not_confirmed": "不支持调threshold/新risk model。",
        "why_stopped": "risk identification与policy gates均失败。",
        "current_authorization": "禁止重复D1 risk protector。",
        "failure_types": "MODEL_SPEC_FAILURE|INFORMATION_FAILURE",
        "evidence_sources": _src("v004c_limited_risk_protector_v001_20260506_20260630/v004c_limited_risk_review_v001.md"),
        "conflict_note": "早期oracle complementarity STRONG仅证明headroom，不证明可识别。",
    },
    {
        "route_name": "RISK_COMPLEMENTARITY_ORACLE",
        "question": "如果能识别LOSS，risk-protector架构是否有headroom？",
        "problem": "区分架构空间与可学习信息。",
        "experiment_done": "YES",
        "data_window": "June strict Top3",
        "model_or_information_change": "oracle/permission counterfactual，无可部署model。",
        "key_result": "ARCHITECTURE_COMPLEMENTARITY_SIGNAL=STRONG；oracle backfill Top3 +1.52pp。",
        "formal_state": "ARCHITECTURE_COMPLEMENTARITY_STRONG",
        "what_confirmed": "去掉真实LOSS理论上有价值。",
        "what_not_confirmed": "没有证明D1能识别哪些是LOSS。",
        "why_stopped": "后续实际protector与information foundation均失败。",
        "current_authorization": "仅理论headroom参考。",
        "failure_types": "NONE",
        "evidence_sources": _src("v004c_stage1_risk_complementarity_v001_20260601_20260630/v004c_stage1_risk_review_v001.md"),
        "conflict_note": "与risk protector ABSENT不冲突：一个是oracle architecture，一个是identifiability。",
    },
    {
        "route_name": "V4A_TOP10_RESIDUAL_RERANKER",
        "question": "小型Stage2能否将Top10 retrieval转成Top3？",
        "problem": "利用Stage1 shortlist空间。",
        "experiment_done": "YES",
        "data_window": "May-June strict",
        "model_or_information_change": "固定residual reranker。",
        "key_result": "Top3 1.77%低于control 1.85%；RERANKER_SIGNAL=ABSENT。",
        "formal_state": "ABSENT",
        "what_confirmed": "shortlist headroom不能由该reranker稳定转化。",
        "what_not_confirmed": "不支持继续Stage2搜索。",
        "why_stopped": "实效低于control。",
        "current_authorization": "禁止重复/扩展Stage2。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED|MODEL_SPEC_FAILURE",
        "evidence_sources": _src("v004c_v4a_top10_residual_reranker_v001_20260506_20260630/v004c_v4a_top10_reranker_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "VETO_ONLY_BACKFILL",
        "question": "只用veto/backfill是否优于control？",
        "problem": "隔离rerank收益是否仅来自去坏。",
        "experiment_done": "YES",
        "data_window": "June strict",
        "model_or_information_change": "固定veto-only counterfactual。",
        "key_result": "VETO_ONLY_SIGNAL=ABSENT；Top3未超过control/universe。",
        "formal_state": "ABSENT",
        "what_confirmed": "closing-completion/veto路线不成立。",
        "what_not_confirmed": "precision改善不足以成为策略。",
        "why_stopped": "收益与风险主要门失败。",
        "current_authorization": "禁止重复。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED",
        "evidence_sources": _src("v004c_veto_only_backfill_v001_20260601_20260630/v004c_veto_only_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "OVEREXTENSION_CURVE",
        "question": "D1强度是否存在非单调过度延伸曲线？",
        "problem": "解释强票为何成为false positive。",
        "experiment_done": "YES",
        "data_window": "May-June",
        "model_or_information_change": "固定curve diagnostic。",
        "key_result": "OVEREXTENSION_CURVATURE_SIGNAL=ABSENT。",
        "formal_state": "ABSENT",
        "what_confirmed": "预注册非单调假设不成立。",
        "what_not_confirmed": "不能据此构造curve feature。",
        "why_stopped": "主要门失败。",
        "current_authorization": "禁止重复同一curve。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED",
        "evidence_sources": _src("v004c_overextension_curve_v001_20260506_20260630/v004c_overextension_curve_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "L_UNFINISHED_REPAIR_HYPOTHESIS",
        "question": "D1未完成修复状态是否稳定区分未来T7/LOSS？",
        "problem": "验证case review的D1 path机制。",
        "experiment_done": "YES",
        "data_window": "May-July mature",
        "model_or_information_change": "无；仅4个预注册已有字段。",
        "key_result": "case中明显，但完整/Rank2-6不稳定；CASE_CONDITIONAL_ONLY。",
        "formal_state": "CASE_CONDITIONAL_ONLY",
        "what_confirmed": "Rank3错误案例常偏好D1已修复更强的LOSS。",
        "what_not_confirmed": "没有字段满足full-population formal gate。",
        "why_stopped": "机制无法普遍化。",
        "current_authorization": "禁止构造unfinished-repair score。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED|TEMPORAL_INSTABILITY",
        "evidence_sources": _src("v004c_d1_unfinished_repair_information_audit_v001_20260506_20260731/v004c_unfinished_repair_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "REPAIR_STATE_NONLINEAR_INTERACTIONS",
        "question": "OPEN/DIVERGENCE/DAMAGE/SUPPLY/RECLAIM状态能否泛化？",
        "problem": "将D1 path机制构造成低维状态模型。",
        "experiment_done": "YES",
        "data_window": "June fit -> July retrospective OOT",
        "model_or_information_change": "预注册5 base + 3 interactions Logistic。",
        "key_result": "X结构健康、June可fit；July R1/R2均未过OOT，FINAL_DECISION=REJECT。",
        "formal_state": "REJECT_REPAIR_STATE_V002_OOT",
        "what_confirmed": "数值/lineage不是主要问题。",
        "what_not_confirmed": "factor-state与nonlinear interactions没有forward增量。",
        "why_stopped": "July probability/ranking/incremental gates失败。",
        "current_authorization": "禁止继续repair-state interaction扩展。",
        "failure_types": "FORWARD_FAILURE|TEMPORAL_INSTABILITY",
        "evidence_sources": _src("v004c_repair_state_july_oot_v001_202607/v004c_repair_state_july_oot_review_v001.md"),
        "conflict_note": "June in-sample/X PASS不能覆盖July OOT REJECT。",
    },
    {
        "route_name": "I_RECENCY_WEIGHTING",
        "question": "May降权、June相对2x是否更适合July？",
        "problem": "检验简单月份近因权重。",
        "experiment_done": "YES",
        "data_window": "May+June train -> July test",
        "model_or_information_change": "S2 row weight额外乘May .5 / June 1.0。",
        "key_result": "July Top3 -0.61pp、T7 -1.64pp、LOSS +3.28pp。",
        "formal_state": "RECENCY_SIGNAL_NOT_SUPPORTED",
        "what_confirmed": "月份接近不等于信息更可迁移。",
        "what_not_confirmed": "不支持继续权重网格/窗口搜索。",
        "why_stopped": "预注册权重方向全面恶化。",
        "current_authorization": "禁止simple recency weighting。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED|FORWARD_FAILURE",
        "evidence_sources": _src("v004c_recency_weighting_diagnostic_v001_20260506_20260731/v004c_recency_weighting_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "M_LOCAL_BOARD_BREAK_REGIME",
        "question": "May/June/July连板-断板-修复生态是否不同？",
        "problem": "解释月份环境变化。",
        "experiment_done": "YES",
        "data_window": "May-July mature",
        "model_or_information_change": "无；生态描述。",
        "key_result": "T7 41.60%->32.95%->25.26%；LOSS 18.63%->26.69%->35.62%；LOCAL_REGIME_DIFFERENCE_SUPPORTED。",
        "formal_state": "LOCAL_REGIME_DIFFERENCE_SUPPORTED",
        "what_confirmed": "局部生态差异真实存在。",
        "what_not_confirmed": "不证明因果或可用于ranking。",
        "why_stopped": "描述任务完成。",
        "current_authorization": "仅环境参考。",
        "failure_types": "NONE",
        "evidence_sources": _src("v004c_board_break_regime_audit_v001_202605_202607/v004c_regime_review_v001.md"),
        "conflict_note": "当时BROADER_MARKET_CONTEXT_NEEDED=NO；后续在local interaction forward失败后才重新授权broad data audit。",
    },
    {
        "route_name": "N_LOCAL_REGIME_CONDITIONAL_SIGNAL",
        "question": "local ecology能否解释stock signal时间翻转？",
        "problem": "检验signal×local-regime条件结构。",
        "experiment_done": "YES",
        "data_window": "May-July development",
        "model_or_information_change": "无；8 signals × 11 regimes预注册审计。",
        "key_result": "development内4个formal relation；最强close/VWAP×board4plus delta -0.276。",
        "formal_state": "LOCAL_REGIME_EXPLAINS_SIGNAL_FLIP",
        "what_confirmed": "development内存在条件结构。",
        "what_not_confirmed": "88选1风险与forward泛化未解决；open-close flip未解释。",
        "why_stopped": "只允许确认一个最强关系。",
        "current_authorization": "仅历史开发证据；不能直接建模。",
        "failure_types": "SELECTION_BIAS_RISK",
        "evidence_sources": _src("v004c_regime_conditional_signal_audit_v001_202605_202607/v004c_regime_conditional_review_v001.md"),
        "conflict_note": "后续selection-aware confirmation与August OOT削弱了该正面development状态。",
    },
    {
        "route_name": "O_CLOSE_VWAP_BOARD4PLUS_CONFIRMATION",
        "question": "最强单关系在88选1校正后是否仍异常？",
        "problem": "校正winner's curse。",
        "experiment_done": "YES",
        "data_window": "May-July development",
        "model_or_information_change": "无；固定单关系、threshold与permutation。",
        "key_result": "固定关系bootstrap稳定，但selection-adjusted p=.3879。",
        "formal_state": "SINGLE_RELATION_SUPPORTED_BUT_SELECTION_RISK",
        "what_confirmed": "development内关系不是1-2日驱动。",
        "what_not_confirmed": "考虑88选1后并不异常。",
        "why_stopped": "selection risk显著。",
        "current_authorization": "不找第二关系；仅因已有计划允许一次August OOT。",
        "failure_types": "SELECTION_BIAS_RISK",
        "evidence_sources": _src("v004c_close_vwap_board4plus_confirmation_v001_202605_202607/v004c_close_vwap_board4plus_confirmation_review_v001.md"),
        "conflict_note": "NONE",
    },
    {
        "route_name": "P_AUGUST_REGIME_AWARE_CHALLENGER",
        "question": "单一close/VWAP×board4plus feature能否通过August OOT？",
        "problem": "给development条件关系一次独立forward verdict。",
        "experiment_done": "YES",
        "data_window": "August mature complete-date: 60 rows / 8 dates",
        "model_or_information_change": "S2 + 唯一预注册regime-signed feature。",
        "key_result": "原关系方向反转；Top3 -0.7pp，T7 -12.5pp，LOSS/severe +4.2pp；正式样本不足。",
        "formal_state": "AUGUST_SAMPLE_INSUFFICIENT",
        "what_confirmed": "August holdout已消费；方向性结果明显negative。",
        "what_not_confirmed": "8日不足以作高置信度统计结论。",
        "why_stopped": "formal insufficient且所有方向指标不利。",
        "current_authorization": "STOP_THIS_REGIME_INTERACTION_LINE；不得测第二关系。",
        "failure_types": "DATA_LIMITATION|FORWARD_FAILURE",
        "evidence_sources": _src("v004c_regime_aware_stage1_august_oot_v001/v004c_regime_aware_stage1_august_oot_review_v001.md"),
        "conflict_note": "正式enum为INSUFFICIENT；方向性negative必须同时保留，不能写成纯未知。",
    },
    {
        "route_name": "Q_BROAD_MARKET_COVERAGE",
        "question": "现有历史数据能否重建4个broad-market context？",
        "problem": "先盲审数据覆盖而不看outcome。",
        "experiment_done": "YES",
        "data_window": "May-August context only",
        "model_or_information_change": "无；M1/M2/M3/M4 coverage。",
        "key_result": "M1-M3 75日可用；M4 0日；初始状态LINEAGE_RISK。",
        "formal_state": "BROAD_CONTEXT_DATA_LINEAGE_RISK",
        "what_confirmed": "daily return横截面可重建；历史cap/liquidity/index/limit-down不可用。",
        "what_not_confirmed": "current snapshot不能宣称true historical universe。",
        "why_stopped": "先要求lineage materiality audit。",
        "current_authorization": "coverage结论被后续sensitivity部分解决。",
        "failure_types": "LINEAGE_LIMITATION|DATA_LIMITATION",
        "evidence_sources": _src("v004c_broad_market_context_coverage_audit_v001/v004c_broad_market_context_coverage_review_v001.md"),
        "conflict_note": "后续lineage sensitivity证明M1-M3 aggregate对该近似robust；M4仍不可用。",
    },
    {
        "route_name": "R_HISTORICAL_UNIVERSE_LINEAGE_SENSITIVITY",
        "question": "universe membership近似是否实质改变M1/M2/M3？",
        "problem": "避免无必要的历史证券主数据工程。",
        "experiment_done": "YES",
        "data_window": "May-July primary; August auxiliary",
        "model_or_information_change": "无；U0/U1/U2 proxy sensitivity。",
        "key_result": "三变量Spearman >.9995、state agreement 100%；LINEAGE_ROBUST。",
        "formal_state": "BROAD_CONTEXT_LINEAGE_ROBUST",
        "what_confirmed": "形式不完美但不 materially影响aggregate strength。",
        "what_not_confirmed": "不证明市场context有alpha；M4仍缺数据。",
        "why_stopped": "lineage materiality问题已回答。",
        "current_authorization": "仅M1-M3可进入一次blind information audit。",
        "failure_types": "LINEAGE_LIMITATION_RESOLVED_FOR_M1_M2_M3",
        "evidence_sources": _src("v004c_broad_market_context_lineage_sensitivity_audit_v001/v004c_broad_market_context_lineage_sensitivity_review_v001.md"),
        "conflict_note": "解决coverage audit的LINEAGE_RISK，不覆盖M4缺失。",
    },
    {
        "route_name": "S_BROAD_MARKET_CONTEXT_INFORMATION",
        "question": "全市场强弱能否解释repair环境与S2不稳定？",
        "problem": "检验唯一broad-market信息维度。",
        "experiment_done": "YES",
        "data_window": "May-July primary; August post-hoc auxiliary",
        "model_or_information_change": "无；M1 primary，M2/M3 robustness。",
        "key_result": "candidate environment fixed-split部分支持；M1-alpha Spearman .070，0/5 temporal metrics改善。",
        "formal_state": "BROAD_MARKET_CONTEXT_PARTIALLY_SUPPORTED",
        "what_confirmed": "强市场candidate T7较高/LOSS较低是部分描述证据；M2/M3方向一致。",
        "what_not_confirmed": "不能解释S2 selection alpha或跨月signal flip；August多数方向不同。",
        "why_stopped": "只解释environment、不解释ranking核心瓶颈。",
        "current_authorization": "STOP_AND_REVIEW；禁止扩展为ranking factor family。",
        "failure_types": "HYPOTHESIS_NOT_SUPPORTED_FOR_SELECTION_ALPHA",
        "evidence_sources": _src("v004c_broad_market_context_information_audit_v001/v004c_broad_market_context_information_review_v001.md"),
        "conflict_note": "formal state PARTIAL，不能简化为完全支持或完全失败。",
    },
)


CONFIRMED_FINDINGS: tuple[dict[str, str], ...] = (
    {"finding_id": "F01", "finding": "Candidate universe本身存在明显oracle gap。", "status": "CONFIRMED", "evidence": "May-June full oracle Top3约6.1%，模型约2%-3%；July full oracle 5.45%而Stage1约1.97%。", "sources": "upside_predictability|top3_failure_decomposition|pair_capped7_july", "conflict_note": "Oracle只证明机会空间，不证明可由D1识别。"},
    {"finding_id": "F02", "finding": "Stage1主要问题不是完全没有机会，而是没有稳定抓住opportunity。", "status": "CONFIRMED", "evidence": "Top3 regret约73%-80%来自winner capture；S2有25/49可替换LOSS slots。", "sources": "top3_failure_decomposition|s2_winner_loss", "conflict_note": "NONE"},
    {"finding_id": "F03", "finding": "Rank1存在一定信息，主要breakdown在Rank1之后。", "status": "PARTIALLY_CONFIRMED", "evidence": "GBDT OOF Rank1有正excess、S2 Rank1 T7 45%；但current18F July Rank1低于universe且各期不稳定。", "sources": "upside_predictability|s2_winner_loss|top2_july", "conflict_note": "不是稳定Rank1 alpha；只能称部分信息。"},
    {"finding_id": "F04", "finding": "当前7F的Target7/LOSS separation时间稳定性不足。", "status": "CONFIRMED", "evidence": "S2 pair concordance 55.14%，仅4/7跨月同方向；Rank2/3 winner-loss混排。", "sources": "s2_winner_loss", "conflict_note": "NONE"},
    {"finding_id": "F05", "finding": "简单降低正则能改善训练拟合，但不能恢复稳定forward selection alpha。", "status": "CONFIRMED", "evidence": "S2优于S0并修复欠拟合；后续绝对forward alpha和winner-loss分离仍弱。", "sources": "training_spec_sanity|s2_winner_loss", "conflict_note": "训练规格修复有效，不等于模型成功。"},
    {"finding_id": "F06", "finding": "简单月份recency weighting失败。", "status": "CONFIRMED", "evidence": "May .5/June 1.0使July Top3 -0.61pp、T7 -1.64pp、LOSS +3.28pp。", "sources": "recency_weighting", "conflict_note": "NONE"},
    {"finding_id": "F07", "finding": "Board3不是唯一原因。", "status": "CONFIRMED", "evidence": "Board3 overpromotion存在但总体不天然更差；Board3 ablation伤winner；全体S2仍有winner-loss confusion。", "sources": "board_structure|board3_ablation|s2_winner_loss", "conflict_note": "Board3是局部放大器，不是全局解释。"},
    {"finding_id": "F08", "finding": "D1 path fields存在局部机制，但一般化不稳定。", "status": "CONFIRMED", "evidence": "case/Rank3错误结构明显，但四字段full-population均未过formal gate；repair-state July OOT被拒。", "sources": "unfinished_repair|repair_state_july", "conflict_note": "NONE"},
    {"finding_id": "F09", "finding": "local board/break regime difference真实存在。", "status": "CONFIRMED", "evidence": "May/June/July T7 41.60/32.95/25.26%，LOSS 18.63/26.69/35.62%，生态多维同步变化。", "sources": "board_break_regime", "conflict_note": "描述差异不等于因果或ranking信息。"},
    {"finding_id": "F10", "finding": "local regime不能稳定解决Stage1 ranking。", "status": "CONFIRMED", "evidence": "development关系存在但88选1 p=.3879；August方向反转且challenger全面不利。", "sources": "regime_conditional|single_relation_confirmation|august_oot", "conflict_note": "早期development formal support被selection correction和forward结果显著削弱。"},
    {"finding_id": "F11", "finding": "broad market strength影响candidate environment，但不能解释S2 selection alpha与时间不稳定。", "status": "PARTIALLY_CONFIRMED", "evidence": "fixed split T7/LOSS有6-7pp差，但continuous+split完整gate未过；M1-alpha rho=.070且0/5 temporal metrics改善。", "sources": "broad_market_information", "conflict_note": "environment部分只能称partial；alpha不支持是明确的。"},
    {"finding_id": "F12", "finding": "August regime-aware challenger正式样本不足，但方向性明显negative。", "status": "CONFIRMED", "evidence": "8 complete mature dates；关系反向；Top3 -0.7pp、T7 -12.5pp、LOSS/severe +4.2pp。", "sources": "august_oot", "conflict_note": "必须同时保留INSUFFICIENT formal enum与negative directional evidence。"},
    {"finding_id": "F13", "finding": "August holdout已经消费，不能再称fresh/untouched OOT。", "status": "CONFIRMED", "evidence": "August OOT review明确AUGUST_HOLDOUT_STATUS=CONSUMED。", "sources": "august_oot", "conflict_note": "后续只可标DEVELOPMENT/AUXILIARY/HISTORICAL。"},
    {"finding_id": "F14", "finding": "复杂模型未证明能解决现有信息不足。", "status": "CONFIRMED", "evidence": "固定GBDT train强但OOF Top3低于baseline；pairwise、repair-state interactions与rerank均未forward成立。", "sources": "upside_predictability|pairwise|repair_state_july|reranker", "conflict_note": "不能由Logistic失败推导需要模型zoo。"},
)


DO_NOT_REPEAT: tuple[dict[str, str], ...] = (
    {"route": "继续调18F legacy architecture", "formal_basis": "18F PARTIAL + July Stage1 alpha failure + ablation non-monotone", "failure_type": "INFORMATION_FAILURE|TEMPORAL_INSTABILITY", "scope": "不再调18F系数、interaction或subset", "source": "v4a_transfer|18f_ablation|top2_july"},
    {"route": "current7F继续L2/tail/positive-weight参数搜索", "formal_basis": "S2修复已完成且任务明确freeze one spec", "failure_type": "MODEL_SPEC_FAILURE_RESOLVED", "scope": "S2只作reference，不追加S4/S5", "source": "training_spec_sanity"},
    {"route": "simple recency/month weighting", "formal_basis": "RECENCY_SIGNAL_NOT_SUPPORTED", "failure_type": "HYPOTHESIS_NOT_SUPPORTED|FORWARD_FAILURE", "scope": "不搜索更多month multiplier/window", "source": "recency_weighting"},
    {"route": "53F GBDT/nonlinear model zoo", "formal_basis": "NONLINEAR_OOF_SIGNAL_PRESENT=NO", "failure_type": "TEMPORAL_INSTABILITY", "scope": "不追加GBDT/XGB/RF/NN", "source": "upside_predictability"},
    {"route": "PAIR_CAPPED7 / pairwise Stage2", "formal_basis": "July FORWARD_STRESS_SIGNAL=PARTIAL且PRIMARY_FAILURE=STAGE2_RERANK_FAILURE", "failure_type": "FORWARD_FAILURE", "scope": "不重跑objective/lambda/reranker", "source": "pair_capped7_july"},
    {"route": "raw repair-strength pairwise objective", "formal_basis": "REPAIR_OBJECTIVE_DEV_SIGNAL=ABSENT", "failure_type": "HYPOTHESIS_NOT_SUPPORTED", "scope": "不改target后重试同一53F", "source": "repair_pairwise"},
    {"route": "v4a Top10 residual reranker", "formal_basis": "RERANKER_SIGNAL=ABSENT", "failure_type": "MODEL_SPEC_FAILURE", "scope": "不追加Stage2/reranker", "source": "v4a_top10_reranker"},
    {"route": "D1 risk protector / veto/backfill", "formal_basis": "RISK_INFORMATION_FOUNDATION=NOT_ESTABLISHED; RISK_PROTECTOR_SIGNAL=ABSENT", "failure_type": "INFORMATION_FAILURE", "scope": "不调risk threshold或训练新protector", "source": "risk_information|limited_risk"},
    {"route": "Board3-only model/risk policy", "formal_basis": "Board3-only/risk均PARTIAL且不稳", "failure_type": "DATA_LIMITATION|INFORMATION_FAILURE", "scope": "不恢复Board3模型、veto或cross-board merge", "source": "board3_only|board3_risk"},
    {"route": "Jan-Jun Board3 old-history extension", "formal_basis": "HISTORY_EXTENSION_EFFECT=HARMFUL; TEMPORAL_COMPATIBILITY=DRIFTED", "failure_type": "TEMPORAL_INSTABILITY", "scope": "不删月份、不调窗口、不加recency", "source": "board3_history"},
    {"route": "Board3 confirmed-harmful contribution ablation", "formal_basis": "ABLATION_MECHANISM_SIGNAL=ABSENT", "failure_type": "HYPOTHESIS_NOT_SUPPORTED", "scope": "不重复三贡献删除/Board3压制", "source": "board3_ablation"},
    {"route": "unfinished-repair 4字段直接结构修正", "formal_basis": "CASE_CONDITIONAL_ONLY", "failure_type": "HYPOTHESIS_NOT_SUPPORTED|TEMPORAL_INSTABILITY", "scope": "不构造unfinished repair score", "source": "unfinished_repair"},
    {"route": "repair-state nonlinear interaction model", "formal_basis": "REJECT_REPAIR_STATE_V002_OOT", "failure_type": "FORWARD_FAILURE", "scope": "不扩展OPEN/DIVERGENCE/RECLAIM interactions", "source": "repair_state_july"},
    {"route": "close/VWAP × board4plus interaction", "formal_basis": "selection-adjusted risk + August方向反转/negative", "failure_type": "SELECTION_BIAS_RISK|FORWARD_FAILURE", "scope": "不改sign/cutoff后重试", "source": "single_relation_confirmation|august_oot"},
    {"route": "第二强local regime relation", "formal_basis": "最强关系未通过selection-aware/forward证据", "failure_type": "SELECTION_BIAS_RISK", "scope": "不测试88关系第二名/替代regime", "source": "single_relation_confirmation|august_oot"},
    {"route": "broad-market-strength ranking factor扩展", "formal_basis": "只部分解释candidate environment；alpha/time instability gates失败", "failure_type": "HYPOTHESIS_NOT_SUPPORTED_FOR_SELECTION_ALPHA", "scope": "不加指数/liquidity/更多market factor救假设", "source": "broad_market_information"},
    {"route": "original-v4a direct transfer旧归档比较", "formal_basis": "ORIGINAL_V4A_PROVENANCE=BLOCKED/benchmark INVALID", "failure_type": "LINEAGE_LIMITATION", "scope": "无新严格归档资产时不得重述/重跑", "source": "original_v4a_direct_transfer"},
)


INFORMATION_FAMILIES: tuple[dict[str, str], ...] = (
    {"family": "价格相对位置（MA/VWAP/日内close-low位置）", "classification": "ADEQUATELY_TESTED_BUT_UNSTABLE", "direct_test": "YES", "evidence": "18F/7F chronological；53F GBDT/Ridge；unfinished-repair与S2 pair audit", "finding": "局部方向存在，但T7-vs-LOSS跨月与forward不稳。", "data_status": "READY", "overlap_note": "核心已覆盖"},
    {"family": "趋势与hold结构", "classification": "ADEQUATELY_TESTED_BUT_UNSTABLE", "direct_test": "YES", "evidence": "7F rank_trend_hold_score；53F OOF；coefficient stability", "finding": "有时贡献，但不能稳定恢复Top3 alpha。", "data_status": "READY", "overlap_note": "核心已覆盖"},
    {"family": "主题/板块属性", "classification": "PARTIAL_INFORMATION_FOUND", "direct_test": "YES", "evidence": "rank_theme_score在7F/S2与market-conditioned fixed signal audit", "finding": "部分winner信息但不足以单独解决Rank2/3。", "data_status": "READY", "overlap_note": "静态theme score已覆盖；point-in-time peer breadth另见DATA_NOT_AVAILABLE"},
    {"family": "主动资金/成交与量能结构", "classification": "ADEQUATELY_TESTED_BUT_UNSTABLE", "direct_test": "YES", "evidence": "rank_active_money_score；53F D1_VOLUME_ACTIVITY/CHIP/LATE_DAY；risk foundation", "finding": "upside与tail方向混合，多重检验后无稳定risk foundation。", "data_status": "READY", "overlap_note": "核心已覆盖"},
    {"family": "D1 intraday path / recovery / absorption / tail", "classification": "ADEQUATELY_TESTED_BUT_UNSTABLE", "direct_test": "YES", "evidence": "53F models；4-field unfinished repair；repair-state v2 July OOT", "finding": "case机制明显但full population/forward不稳定。", "data_status": "READY", "overlap_note": "不要再扫描53F"},
    {"family": "连板层级、stock board history与recent pool path", "classification": "PARTIAL_INFORMATION_FOUND", "direct_test": "YES", "evidence": "Board2/3 structure；Board3 geometry/model；53F BOARD_HISTORY/RECENT_7D_PATH", "finding": "Board3 overpromotion真实，但专模/扩历史/ablation未形成稳定方案。", "data_status": "READY", "overlap_note": "层级计数已覆盖；不等于board-day微观封板路径"},
    {"family": "D0 timing / days_since_d0", "classification": "ADEQUATELY_TESTED_AND_WEAK", "direct_test": "YES", "evidence": "18F contribution ablation A1", "finding": "删除后Top3 membership与经济结果完全不变。", "data_status": "READY", "overlap_note": "已充分弱化"},
    {"family": "局部board/break/candidate ecology", "classification": "ADEQUATELY_TESTED_BUT_UNSTABLE", "direct_test": "YES", "evidence": "regime audit；8x11 conditional；selection-aware confirmation；August OOT", "finding": "生态差异真实、development条件关系存在，但selection risk与forward反向。", "data_status": "READY", "overlap_note": "禁止第二关系"},
    {"family": "全市场broad strength", "classification": "ADEQUATELY_TESTED_AND_WEAK", "direct_test": "YES", "evidence": "coverage+lineage sensitivity+information audit", "finding": "部分解释candidate environment，不解释S2 alpha或跨月不稳。", "data_status": "READY_M1_M2_M3", "overlap_note": "M4 size context不可用"},
    {"family": "个股价格/市值规模", "classification": "PARTIAL_INFORMATION_FOUND", "direct_test": "YES", "evidence": "rank_log_candidate_base_price在7F/S2；历史float-cap coverage=0", "finding": "价格级别已测试；point-in-time market-cap style未测试且无数据。", "data_status": "PARTIAL", "overlap_note": "不要把价格proxy当作历史市值"},
    {"family": "sub-7 repair-strength / target information", "classification": "ADEQUATELY_TESTED_BUT_UNSTABLE", "direct_test": "YES", "evidence": "Top3 decomposition；raw repair pairwise；PAIR_CAPPED7 development+July", "finding": "development内capped target有增量，July仍低于universe。", "data_status": "READY", "overlap_note": "目标路线已充分测试"},
    {"family": "断板前2/3连板日的个股级封板/回封路径质量", "classification": "NOT_DIRECTLY_TESTED", "direct_test": "NO", "evidence": "53F仅含board-history计数/board-day volume rank；现有正式实验未直接审计D0及前序板日的分钟封板路径", "finding": "机制上可能描述断板前筹码锁定、分歧与供给质量；必须先盲审数据覆盖与lineage。", "data_status": "PARTIAL", "overlap_note": "与D1断板日path和date-level board ecology本质不同"},
    {"family": "point-in-time主题同伴/板块breadth", "classification": "DATA_NOT_AVAILABLE", "direct_test": "NO", "evidence": "mechanism foundation明确group block因无point-in-time membership snapshot不可用", "finding": "不能用当前板块成员回构历史。", "data_status": "NO", "overlap_note": "不进入未测试高价值候选"},
    {"family": "集合竞价、逐笔订单簿与封单深度", "classification": "DATA_NOT_AVAILABLE", "direct_test": "NO", "evidence": "现有正式数据为daily/5m与派生字段，未见point-in-time auction/order-book source", "finding": "可能有机制，但当前数据不满足低成本/lineage条件。", "data_status": "NO", "overlap_note": "不授权外部数据获取"},
    {"family": "历史全市场size-style / liquidity / index / formal limit-down", "classification": "DATA_NOT_AVAILABLE", "direct_test": "NO", "evidence": "broad-market coverage: M4=0，liquidity/index/limit-down均不可用", "finding": "当前复杂度与数据预算不支持。", "data_status": "NO", "overlap_note": "不进入下一步"},
)


UNTESTED_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "priority_rank": "1",
        "family": HIGH_VALUE_UNTESTED_INFORMATION_FAMILY,
        "plain_name": "断板前2/3连板日的个股级封板/回封路径质量",
        "mechanism": "用D0及前序连板日的分钟路径描述封板是否顺畅、是否反复开合及板上供给质量；这些信息在D1收盘时已知，并可能影响断板后修复潜力。",
        "mechanism_clarity": "HIGH",
        "data_ready": "PARTIAL",
        "overlap_with_failed_info": "LOW",
        "expected_complexity": "LOW",
        "temporal_risk": "MEDIUM",
        "why_not_previously_tested": "既有53F/board audits侧重D1断板日路径、板数/近况计数和date-level生态；未见直接的stock-level pre-break board-day seal-path audit。",
        "authorization_boundary": "只授权blind coverage/lineage + fixed descriptive information audit；不得建模、搜索分钟窗口或阈值。",
        "eligible": "YES",
    },
)


BOTTLENECK_ROWS: tuple[dict[str, str], ...] = (
    {"item": "CURRENT_STAGE1_BOTTLENECK", "value": CURRENT_STAGE1_BOTTLENECK, "evidence": "S2 pair 55.14%；Rank2/3混排；July alpha failure；53F nonlinear OOF failure"},
    {"item": "MODEL_FORM_LIMITATION", "value": MODEL_FORM_LIMITATION, "evidence": "GBDT、pairwise、repair-state interactions和reranker均未稳定泛化；没有证据证明换更复杂model能解决。"},
    {"item": "CURRENT_INFORMATION_LIMITATION", "value": CURRENT_INFORMATION_LIMITATION, "evidence": "7F raw/rank都弱且不稳；固定53F GBDT train-OOF gap明显；winner capture是主要regret。"},
    {"item": "BROADER_D1_INFORMATION_LIMITATION", "value": BROADER_D1_INFORMATION_LIMITATION, "evidence": "高价值常见family大多覆盖，但pre-break board-day seal path未直接测试。"},
    {"item": "HIGH_VALUE_UNTESTED_INFORMATION_FAMILY", "value": HIGH_VALUE_UNTESTED_INFORMATION_FAMILY, "evidence": "D1-known、机制清晰、与D1 break-day path/date-level ecology低重叠；数据ready仅PARTIAL。"},
    {"item": "STAGE1_RESEARCH_STATE", "value": STAGE1_RESEARCH_STATE, "evidence": "只剩一个窄、盲、低复杂度information audit有资格。"},
    {"item": "NEXT_ACTION", "value": NEXT_ACTION, "evidence": "先证明历史D0/board-day minute覆盖和固定描述信息；不得训练model。"},
    {"item": "AUGUST_HOLDOUT_STATUS", "value": AUGUST_HOLDOUT_STATUS, "evidence": "regime-aware August OOT已读取并给出formal verdict。"},
    {"item": "S2_ROLE", "value": "REFERENCE_BASELINE_ONLY", "evidence": "S2是training-spec sanity的最好预注册规格，但winner-loss attribution=INFORMATION_LIMIT_DOMINANT。"},
    {"item": "S2_FEATURE_COUNT", "value": "7", "evidence": "|".join(S2_FEATURES)},
    {"item": "S2_L2", "value": "0.10", "evidence": "S2_NO_TAIL_L2_010"},
    {"item": "S2_POSITIVE_WEIGHT", "value": "1.50", "evidence": "frozen training-spec"},
    {"item": "S2_TAIL_BONUS", "value": "OFF", "evidence": "7/10/12% Target7同等tail weighting"},
    {"item": "S2_TRAINING_WINDOW", "value": "2026-05-06..2026-07-29 mature as-of 2026-08-01; 485 rows / 60 dates", "evidence": "training-spec and S2 attribution reviews"},
    {"item": "L2_TUNING_AUTHORIZED", "value": "NOT_AUTHORIZED", "evidence": "fixed S0-S3 sanity completed"},
    {"item": "MONTH_WEIGHT_TUNING_AUTHORIZED", "value": "NOT_AUTHORIZED", "evidence": "RECENCY_SIGNAL_NOT_SUPPORTED"},
    {"item": "TAIL_WEIGHT_TUNING_AUTHORIZED", "value": "NOT_AUTHORIZED", "evidence": "tail bonus mismatch already repaired"},
    {"item": "FEATURE_SUBSET_TUNING_AUTHORIZED", "value": "NOT_AUTHORIZED", "evidence": "7F preregistered; no further subset search"},
    {"item": "FACTOR_ANALYSIS_4_STATE_DISCOVERY", "value": "NOT_FOUND_IN_WORKSPACE", "evidence": "Repository-wide filename/content search on 2026-09-02 found no readable artifact; formal reviews used."},
    {"item": "PRIMARY_OBJECTIVE", "value": "Target7 = 1[D3 high / D2 open - 1 >= 7%]", "evidence": "frozen v004c contract"},
    {"item": "SECONDARY_OBJECTIVE", "value": "Preserve Target7 while controlling LOSS and severe LOSS", "evidence": "frozen v004c contract"},
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evidence_paths(routes: Iterable[Mapping[str, str]]) -> tuple[str, ...]:
    values: list[str] = []
    for row in routes:
        for value in row["evidence_sources"].split("|"):
            if value and value not in values:
                values.append(value)
    return tuple(values)


def verify_source_reviews(repo_root: Path) -> list[str]:
    missing: list[str] = []
    for relative in _evidence_paths(ROUTES):
        if not (repo_root / relative).is_file():
            missing.append(relative)
    return missing


def render_review() -> str:
    return f"""# v004c Stage1 Research Convergence Review v001

## 简单版结论

1. **现在卡在哪里？** 现有D1-known信息不能跨时间稳定地区分未来Target7与LOSS，Rank2/Rank3尤其容易把winner和LOSS混在一起。
2. **已经试过什么？** 18F/7F Logistic、训练规格、53F Ridge/GBDT、pairwise/Stage2、risk protector、Board3专模、D1 path、local regime、broad-market strength以及一次August regime-aware OOT都已正式覆盖。
3. **哪些路不用再试？** 继续调18F/7F参数、模型zoo、pairwise/PAIR_CAPPED7、risk protector、Board3-only/旧历史、recency weighting、close/VWAP×board4plus及broad-market ranking扩展均不再授权。
4. **还有没有真正没试过的？** 有且只有一个较窄候选：断板前2/3连板日自身的个股级封板/回封路径质量。它不是D1断板日path，也不是date-level板池生态；当前数据ready仅PARTIAL。
5. **下一步继续还是停？** 只授权一次blind information audit，先审coverage/lineage，再用固定描述检验；不授权任何模型。若该family也不成立，应停止当前Stage1 D1因子挖掘路线。

## 1. Frozen Goal and Pipeline

最终任务没有改变：D1收盘后，对v004c首次断板候选排序，用户关注Top3。Primary是 `Target7 = 1[D3 high / D2 open - 1 >= 7%]`；Secondary是在尽量不损失Target7的前提下控制LOSS与severe LOSS。

```text
V4C Board2/Board3 first-break candidate universe
    -> D1-known information
    -> Stage1 ranking
    -> Top3
    -> D2-open to D3-high outcome
```

当前主要问题位于 **D1-known information -> stable cross-sectional ranking**：候选池仍有oracle空间，但现有信息无法稳定把Target7推到Rank2/Rank3并把LOSS压下去。

## 2. Current Reference Baseline

S2仍然只是 `REFERENCE BASELINE`，不是successful model：

- 7F: {', '.join(S2_FEATURES)}
- weighted L2 Logistic；L2=0.10；positive_weight=1.50；tail bonus=OFF。
- development: 2026-05-06..2026-07-29 mature as-of 2026-08-01，485 rows / 60 dates。
- L2、月份权重、tail、feature subset继续调整：全部 `NOT AUTHORIZED`。

## 3. Research Route Inventory

统一路线总表见 `v004c_research_route_inventory_v001.csv`，共 {len(ROUTES)} 条。它区分了：正式negative、partial、data/provenance blocked、development-only positive和forward failure。关键证据链如下：

- 18F只获得PARTIAL，July current Stage1未击败universe。
- Reduced7F与S2修复了部分model-spec问题，但S2 T7-vs-LOSS concordance仅55.14%。
- 固定53F GBDT训练内AUC高而OOF Top3低于baseline，说明“更复杂模型”没有解决跨期泛化。
- Board3 overpromotion真实，但Board3 ablation、专模、risk tail和旧历史均未形成安全稳定方案。
- D1 path、repair-state、local regime在development有局部信息，但selection-aware或forward验证不成立。
- broad market strength只部分解释candidate environment，不解释S2 alpha与时间翻转。

## 4. Confirmed Findings

逐条裁决见 `v004c_confirmed_findings_v001.csv`。最重要的三点：

1. **Opportunity存在**：oracle gap明确；不是候选池完全没有winner。
2. **识别失败为主**：Top3 regret约73%-80%来自winner capture；S2有25/49可替换LOSS slots。
3. **信息稳定性不足**：7F、53F、local/broad context与非线性形式都没有建立稳定Rank2/Rank3 winner-loss separation。

其中“Rank1存在一定信息”和“broad market影响candidate environment”只能标 `PARTIALLY_CONFIRMED`，不能夸大成稳定alpha。

## 5. Failure Types

- `MODEL_SPEC_FAILURE`: 7F的L2=.30与legacy tail bonus；已由S2修复。
- `INFORMATION_FAILURE`: S2/53F不能稳定分开Target7与LOSS；当前主瓶颈。
- `TEMPORAL_INSTABILITY`: May/June/July的单因子与模型方向变化。
- `SELECTION_BIAS_RISK`: 88关系选最强后，close/VWAP×board4plus adjusted p=.3879。
- `FORWARD_FAILURE`: PAIR_CAPPED7、repair-state、current18F July与August challenger均未恢复稳定alpha。
- `DATA_LIMITATION`: Board3小样本；August complete mature仅8日。
- `LINEAGE_LIMITATION`: original-v4a strict provenance失败；broad-market历史universe形式不完美但M1-M3 materiality已证明robust。
- `HYPOTHESIS_NOT_SUPPORTED`: recency、overextension、veto/reranker等固定假设。

## 6. Do Not Repeat

`v004c_failed_routes_do_not_repeat_v001.csv`只收录有正式证据或明确停止合同的路线。特别是：

- 不把GBDT训练内拟合当作需要更多复杂模型的证据。
- 不把Board3局部问题当作全局解释。
- 不因local regime development关系而测试第二名关系。
- 不因broad market部分解释candidate environment而把它加进ranking。

## 7. Model vs Information

`MODEL_FORM_LIMITATION = {MODEL_FORM_LIMITATION}`。

已有信息若只是Logistic表达不够，固定GBDT、pairwise、repair-state interactions或Stage2至少应在OOF/forward中恢复一部分稳定Top3 alpha；事实没有发生。复杂模型有强train fit但弱OOF，说明不能把问题归为单纯model form。

`CURRENT_INFORMATION_LIMITATION = {CURRENT_INFORMATION_LIMITATION}`。

该判断由S2 pair concordance、raw-vs-rank审计、53F nonlinear OOF、跨月方向变化、local/broad regime结果和August方向性失败共同支持。

`BROADER_D1_INFORMATION_LIMITATION = {BROADER_D1_INFORMATION_LIMITATION}`，因为仍有一个机制上不同、尚未直接审计的pre-break board-day路径家族；在该家族完成盲审计前，不能宣称所有D1信息已穷尽。

## 8. Information Family Inventory

完整分类见 `v004c_information_family_inventory_v001.csv`。规则严格区分“字段存在”与“被直接验证”：

- 价格位置、趋势、资金/量能、D1 path、board history、local ecology、broad strength和target information均已有直接实验。
- 53F的存在本身不等于每个字段独立通过；这里只按正式family级模型/信息审计定性。
- point-in-time主题成员、auction/order-book、历史size/liquidity/index/limit-down属于 `DATA_NOT_AVAILABLE`，不进入候选清单。
- 唯一 `NOT_DIRECTLY_TESTED` 且满足低复杂度条件的是 `PRE_BREAK_BOARD_DAY_SEAL_PATH`。

## 9. The Only Authorized Gap

`HIGH_VALUE_UNTESTED_INFORMATION_FAMILY = {HIGH_VALUE_UNTESTED_INFORMATION_FAMILY}`。

中文定义：**断板前2/3连板日的个股级封板/回封路径质量**。它与既有失败路线本质不同，因为既有D1 path描述的是“断板当天发生了什么”，local ecology描述的是“当天整个板池怎样”，而该family描述的是“这只股票在进入断板日以前，连续涨停形成过程的封板质量与潜在供给”。

仅授权：固定字段定义前的blind coverage/lineage audit，以及随后一次预注册information audit。禁止模型、阈值、窗口搜索和自动字段生成。

## 10. Conflicts and Evidence Evolution

- 仓库内未找到可读的 `FACTOR_ANALYSIS_4_STATE` 最新状态总结；已执行filename/content检索。故本报告完全按正式review裁决。
- local regime conditional audit在development给出EXPLAINS，但单关系selection correction只到SUPPORTED_BUT_SELECTION_RISK，August方向反转且challenger不利；最终不能写成已解决ranking。
- broad-market coverage最初为LINEAGE_RISK，后续sensitivity证明M1-M3 aggregate robust；这是风险被量化解决，不是报告冲突。M4仍不可用。
- Reduced7F temporal SUPPORTED与后续INFORMATION_LIMIT_DOMINANT不冲突：前者是相对18F改善，后者指出绝对winner-loss separation仍不足。
- August正式enum是SAMPLE_INSUFFICIENT，但方向性结果negative；两者必须同时保留。

## 11. Final Verdict

CURRENT_STAGE1_BOTTLENECK = {CURRENT_STAGE1_BOTTLENECK}

MODEL_FORM_LIMITATION = {MODEL_FORM_LIMITATION}

CURRENT_INFORMATION_LIMITATION = {CURRENT_INFORMATION_LIMITATION}

BROADER_D1_INFORMATION_LIMITATION = {BROADER_D1_INFORMATION_LIMITATION}

HIGH_VALUE_UNTESTED_INFORMATION_FAMILY = {HIGH_VALUE_UNTESTED_INFORMATION_FAMILY}

STAGE1_RESEARCH_STATE = {STAGE1_RESEARCH_STATE}

NEXT_ACTION = {NEXT_ACTION}

AUGUST_HOLDOUT_STATUS = {AUGUST_HOLDOUT_STATUS}

后续任何August使用只能标 `DEVELOPMENT / AUXILIARY / HISTORICAL`，不能再称fresh/untouched OOT。
"""


def build_outputs(repo_root: Path) -> dict[str, Path]:
    missing = verify_source_reviews(repo_root)
    if missing:
        raise FileNotFoundError("Missing formal source reviews: " + ", ".join(missing))

    out_dir = repo_root / OUTPUT_DIR
    route_fields = tuple(ROUTES[0].keys())
    finding_fields = tuple(CONFIRMED_FINDINGS[0].keys())
    dnr_fields = tuple(DO_NOT_REPEAT[0].keys())
    family_fields = tuple(INFORMATION_FAMILIES[0].keys())
    untested_fields = tuple(UNTESTED_FAMILIES[0].keys())
    bottleneck_fields = tuple(BOTTLENECK_ROWS[0].keys())

    paths = {
        "routes": out_dir / "v004c_research_route_inventory_v001.csv",
        "findings": out_dir / "v004c_confirmed_findings_v001.csv",
        "do_not_repeat": out_dir / "v004c_failed_routes_do_not_repeat_v001.csv",
        "families": out_dir / "v004c_information_family_inventory_v001.csv",
        "untested": out_dir / "v004c_untested_information_families_v001.csv",
        "bottleneck": out_dir / "v004c_current_bottleneck_v001.csv",
        "review": out_dir / "v004c_stage1_research_convergence_review_v001.md",
    }
    _write_csv(paths["routes"], ROUTES, route_fields)
    _write_csv(paths["findings"], CONFIRMED_FINDINGS, finding_fields)
    _write_csv(paths["do_not_repeat"], DO_NOT_REPEAT, dnr_fields)
    _write_csv(paths["families"], INFORMATION_FAMILIES, family_fields)
    _write_csv(paths["untested"], UNTESTED_FAMILIES, untested_fields)
    _write_csv(paths["bottleneck"], BOTTLENECK_ROWS, bottleneck_fields)
    paths["review"].write_text(render_review(), encoding="utf-8", newline="\n")
    return paths


__all__ = [
    "AUGUST_HOLDOUT_STATUS",
    "BROADER_D1_INFORMATION_LIMITATION",
    "CURRENT_INFORMATION_LIMITATION",
    "CURRENT_STAGE1_BOTTLENECK",
    "HIGH_VALUE_UNTESTED_INFORMATION_FAMILY",
    "MODEL_FORM_LIMITATION",
    "NEXT_ACTION",
    "OUTPUT_DIR",
    "ROUTES",
    "STAGE1_RESEARCH_STATE",
    "build_outputs",
    "verify_source_reviews",
]

# -*- coding: utf-8 -*-
"""v004c 阶段2.3: 人工候选模型规格冻结 (只冻结规格与协议, 不训练不预测)。

本阶段只冻结:
    候选模型成员 (M0 截距基准 + M1/M2/M3 三个实质候选)
    特征顺序
    预处理规则 (FOLD_CLIP_Z / RAW_BINARY / FAIL_CLOSED 缺失策略)
    模型族和固定超参数 (L2 C=1.0 lbfgs, 禁止搜索)
    七月一次性评价协议 (只写协议, 不执行)
    日期级 Walk-forward 协议 (只写协议, 不运行)
    最终候选资格门与比较层级

本阶段禁止:
    训练模型 / 拟合系数 / 生成预测 / 查看七月标签 / 计算七月表现 /
    运行 Walk-forward / 根据统计结果自动选择或替换因子 / 自动 commit 或 push。

候选成员是研究人员阅读六月阶段2.2 证据后在本模块内人工固定的常量 (CANDIDATE_MODELS), 与阶段2.2 证据值
(effect_value / signed_auc_raw / bootstrap CI / flags / LODO / board 方向)
完全无关: 模型目录、特征规格、预处理规格、候选 JSON、评价协议 JSON 的构建
函数不接受任何证据输入, 因此对证据扰动逐字节不变。

无标签冗余审计只读取阶段1 snapshot 的 13 个允许列 (signal_date +
8 个源特征 + 4 个派生源列), 从不读取任何标签/结果/审计列;
派生因子按阶段2.1 冻结公式逐行重算 (不用标签, 不用跨行统计)。

输出 (正式输出目录共 15 个文件; manifest 记录其余 14 个非 manifest 文件):
    v004c_stage2_3_model_catalog_v001.csv          4 个模型目录
    v004c_stage2_3_model_feature_spec_v001.csv     M1/M2/M3 特征规格 (15 行)
    v004c_stage2_3_preprocessing_spec_v001.csv     预处理规格 (4 模型级 + 15 特征级)
    v004c_stage2_3_selected_feature_evidence_v001.csv  12 个唯一入选特征证据表
    v004c_stage2_3_candidate_redundancy_audit_v001.csv  无标签冗余审计 (30 行)
    v004c_stage2_3_candidate_boundary_v001.csv     59 行候选边界表
    v004c_stage2_3_holdout_lock_audit_v001.csv     七月锁定审计
    v004c_stage2_3_candidate_specs_v001.json       候选规格 (稳定排序, 确定性)
    v004c_stage2_3_evaluation_protocol_v001.json   评价协议 (稳定排序, 确定性)
    v004c_stage2_3_manifest.json
    v004c_stage2_3_review.md
    git_head.txt / git_status_before.txt / git_status_after.txt
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .v004c_d1_dataset import (
    DatasetValidationError,
    GitProvenance,
    _count_csv_rows,
    _repo_relative_posix,
    _run_git,
    _sha256,
    collect_git_provenance,
    validate_git_head,
)
from .v004c_factor_dictionary import TARGET_BLIND_SENSITIVE_COLUMNS

# ---------------------------------------------------------------------------
# 版本链常量 (阶段1 / 阶段2.1 / 阶段2.2 冻结)
# ---------------------------------------------------------------------------
STAGE2_3_VERSION = "v004c-stage2-3-candidate-specs-0.1"
EXPECTED_BRANCH = "research-sample-analysis"

EXPECTED_STAGE1_TAG = "v004c-d1-dataset-0.1"
EXPECTED_STAGE1_DATA_COMMIT = "a65661f4738b849a06efb4859d4100842871e971"
EXPECTED_STAGE1_GENERATOR_COMMIT = "7f4db91e06513a08de382831d401c5cc4d5ea3c4"
STAGE1_MANIFEST_AUDIT_REFERENCE = (
    "ae2a2b48b0b97390e08e81c42cadaa84d5ff76b95e847a80a66e480936ccd618")

EXPECTED_STAGE2_1_TAG = "v004c-factor-dictionary-0.1"
EXPECTED_STAGE2_1_DATA_COMMIT = "11f455dc9a6643a67250321cce68868be904cf06"
EXPECTED_STAGE2_1_CODE_COMMIT = "aa62b315dd1f792dd7657b190e41c1e992f93112"

EXPECTED_STAGE2_2_TAG = "v004c-stage2-2-univariate-0.1"
EXPECTED_STAGE2_2_DATA_COMMIT = "fba302ffc58deb7b15812e4dd2ada113ae1f3a34"
EXPECTED_STAGE2_2_CODE_COMMIT = "ee0306fc4a5f3dadd3f0a780721754a014c58c58"

TIMEZONE = "Asia/Shanghai"

DEVELOPMENT_START = "2026-06-01"
DEVELOPMENT_END = "2026-06-30"
LOCKED_START = "2026-07-01"
LOCKED_END = "2026-07-29"

EXPECTED_GATES = {
    "input_rows": 333,
    "columns": 197,
    "signal_dates": 42,
    "lineage_rows": 197,
    "dev_rows": 173,
    "dev_target_positive": 59,
    "locked_rows": 160,
    "primary_allowlist": 38,
    "sensitivity_allowlist": 21,
    "total_features": 59,
    "selected_unique_features": 12,
    "boundary_rows": 59,
    "model_count": 4,
    "substantive_model_count": 3,
    "m_feature_count": 5,
    "redundancy_rows": 30,
    "feature_spec_rows": 15,
}

# ---------------------------------------------------------------------------
# 阶段2.3 只允许从 snapshot 读取的列 (signal_date + 12 个唯一入选特征的源列)
# ---------------------------------------------------------------------------
SELECTED_RAW_FEATURES = (
    "break_open_return",
    "down_bar_volume_ratio",
    "high_zone_volume_ratio",
    "late_day_sell_volume_ratio",
    "d1_close_to_vwap_raw",
    "high_zone_amount_ratio",
    "late_day_sell_amount_ratio",
    "d1_vwap_to_close_gap",
)
SELECTED_DERIVED_SOURCE_COLUMNS = {
    "overrepair": "d1_open_to_close_return_raw",
    "ma5_overheat_10": "d1_close_to_ma5_raw",
    "break_volume_abnormality": "break_volume_ratio_vs_board_days",
    "profit_chip_ratio": "volume_above_d1_close_ratio",
}
SNAPSHOT_READ_COLUMNS = tuple(sorted(
    ("signal_date",) + SELECTED_RAW_FEATURES
    + tuple(SELECTED_DERIVED_SOURCE_COLUMNS.values())))
SIGNAL_DATE_COL = "signal_date"

# 阶段2.3 允许读取的 19 个输入文件 (任务 5.2)
ALLOWED_INPUT_FILES = {
    "stage1": ("v004c_d1_snapshot_v001.csv",
               "v004c_d1_column_lineage.csv",
               "v004c_d1_data_manifest.json"),
    "stage2_1": ("v004c_factor_dictionary_v001.csv",
                 "v004c_feature_allowlist_primary_v001.csv",
                 "v004c_feature_allowlist_sensitivity_v001.csv",
                 "v004c_canonical_feature_map_v001.csv",
                 "v004c_exact_duplicate_groups_v001.csv",
                 "v004c_near_duplicate_pairs_v001.csv",
                 "v004c_predeclared_derived_factor_spec_v001.csv",
                 "v004c_predeclared_derived_factor_audit_v001.csv",
                 "v004c_factor_dictionary_manifest.json"),
    "stage2_2": ("v004c_stage2_2_feature_structure_v001.csv",
                 "v004c_stage2_2_univariate_summary_dev_v001.csv",
                 "v004c_stage2_2_cluster_bootstrap_dev_v001.csv",
                 "v004c_stage2_2_board_subgroup_dev_v001.csv",
                 "v004c_stage2_2_distribution_shift_unlabeled_v001.csv",
                 "v004c_stage2_2_manifest.json",
                 "v004c_stage2_2_review.md"),
}
ALLOWED_INPUT_COUNT = sum(len(v) for v in ALLOWED_INPUT_FILES.values())

# 阶段2.3 从未读取的输入 (硬门: 实际 loaded_input_files 与禁止清单交集为空)
FORBIDDEN_INPUT_FILENAMES = frozenset({
    "v004c_training_d1_v001.csv",
    "v004c_d1_existing_model_audit_v001.csv",
    "v004c_d1_candidate_audit.csv",
    "v004c_d1_data_quality.csv",
    "v004c_d1_excluded_rows_v001.csv",
    "v004c_d1_trade_date_adjacency_evidence.csv",
    "v004c_d1_dataset_review.md",
})

# ---------------------------------------------------------------------------
# 固定超参数与预处理常量 (人工冻结, 禁止后续搜索)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "binary_logistic_regression"
PENALTY = "L2"
C_VALUE = 1.0
SOLVER = "lbfgs"
FIT_INTERCEPT = True
CLASS_WEIGHT = None
MAX_ITER = 5000
TOL = 1e-8
RANDOM_SEED = 20260806

FOLD_CLIP_LOWER_QUANTILE = 0.01
FOLD_CLIP_UPPER_QUANTILE = 0.99
FOLD_CLIP_DDOF = 0
FOLD_STD_FLOOR = 1e-12
BINARY_ALLOWED_VALUES = (0, 1)
MISSING_POLICY = "FAIL_CLOSED"

# 候选设计定位 (阶段2.3.1 修正): 六月开发证据用于人工设计;
# 候选规格在七月标签揭示前冻结; 运行时绝不读取 Target 列。
CANDIDATE_SELECTION_MODE = "human_fixed_after_june_development_review"
JUNE_DEVELOPMENT_EVIDENCE_USED_FOR_MANUAL_DESIGN = True
CANDIDATE_SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS = True
RUNTIME_CANDIDATE_MEMBERSHIP_USES_TARGET_COLUMNS = False
RUNTIME_MEMBERSHIP_INVARIANT_TO_TARGET_COLUMN_MUTATION = True
AUTOMATIC_TARGET_DRIVEN_SELECTION = False

# ---------------------------------------------------------------------------
# 人工固定候选模型目录 (六月开发审查后固定, 不得按运行结果增删替换)
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = (
    {
        "model_id": "M0",
        "model_name": "INTERCEPT_BASELINE",
        "role": "BASELINE",
        "feature_count": 0,
        "model_family": "intercept_prevalence_baseline",
        "penalty": "none",
        "design_purpose": "截距基准: 后续拟合时预测概率 = 训练窗口 Target7 正样本率; "
                          "不使用任何特征, 作为七月一次性评价与 Walk-forward 的对比下界。",
        "features": (),
    },
    {
        "model_id": "M1",
        "model_name": "PRIMARY_MECHANISM_CORE",
        "role": "PRIMARY_CANDIDATE",
        "feature_count": 5,
        "model_family": MODEL_FAMILY,
        "penalty": PENALTY,
        "design_purpose": "primary 跨机制核心模型: 5 个不同机制组的 primary 表达; "
                          "无缺失处理需求; 不使用稀疏二元字段; "
                          "不使用日期不稳定或单日主导字段; 作为主要紧凑候选。",
        "features": (
            {"position": 1, "feature_name": "break_open_return",
             "expected_direction": "+",
             "mechanism_group": "D1_PRICE_ACTION",
             "selection_rationale": "断板日开盘相对前收收益, 修复动能的方向性 primary 表达。"},
            {"position": 2, "feature_name": "down_bar_volume_ratio",
             "expected_direction": "+",
             "mechanism_group": "D1_VOLUME_ACTIVITY",
             "selection_rationale": "阴线量占比, 出货/承接结构的方向性 primary 表达。"},
            {"position": 3, "feature_name": "high_zone_volume_ratio",
             "expected_direction": "-",
             "mechanism_group": "D1_CHIP_DISTRIBUTION",
             "selection_rationale": "高位区成交量占比, 套牢筹码堆积的 primary 表达。"},
            {"position": 4, "feature_name": "late_day_sell_volume_ratio",
             "expected_direction": "-",
             "mechanism_group": "D1_LATE_DAY_PRESSURE",
             "selection_rationale": "尾盘卖出成交量占比, 尾盘抛压的 primary 表达。"},
            {"position": 5, "feature_name": "d1_close_to_vwap_raw",
             "expected_direction": "-",
             "mechanism_group": "D1_VWAP_POSITION",
             "selection_rationale": "收盘相对当日 VWAP 的位置, 全天持仓成本的 primary 表达。"},
        ),
    },
    {
        "model_id": "M2",
        "model_name": "SENSITIVITY_EXPRESSION_ALTERNATIVES",
        "role": "SENSITIVITY_CANDIDATE",
        "feature_count": 5,
        "model_family": MODEL_FAMILY,
        "penalty": PENALTY,
        "design_purpose": "表达替代敏感性模型: 保持与 M1 相同的主要机制结构, "
                          "保留 break_open_return 与 down_bar_volume_ratio, "
                          "替换 high_zone_volume_ratio→high_zone_amount_ratio、"
                          "late_day_sell_volume_ratio→late_day_sell_amount_ratio、"
                          "d1_close_to_vwap_raw→d1_vwap_to_close_gap; "
                          "用于判断信号是否来自机制本身, 而不是只来自 volume/amount "
                          "或正向/反向表达。M1 与 M2 的互斥表达不得放进同一模型。",
        "features": (
            {"position": 1, "feature_name": "break_open_return",
             "expected_direction": "+",
             "mechanism_group": "D1_PRICE_ACTION",
             "selection_rationale": "与 M1 相同, 保持主要机制结构不变。"},
            {"position": 2, "feature_name": "down_bar_volume_ratio",
             "expected_direction": "+",
             "mechanism_group": "D1_VOLUME_ACTIVITY",
             "selection_rationale": "与 M1 相同, 保持主要机制结构不变。"},
            {"position": 3, "feature_name": "high_zone_amount_ratio",
             "expected_direction": "-",
             "mechanism_group": "D1_CHIP_DISTRIBUTION",
             "selection_rationale": "M1 high_zone_volume_ratio 的金额替代 (互斥组 ME03_HIGH_ZONE)。"},
            {"position": 4, "feature_name": "late_day_sell_amount_ratio",
             "expected_direction": "-",
             "mechanism_group": "D1_LATE_DAY_PRESSURE",
             "selection_rationale": "M1 late_day_sell_volume_ratio 的金额替代 (互斥组 ME04_LATE_SELL)。"},
            {"position": 5, "feature_name": "d1_vwap_to_close_gap",
             "expected_direction": "+",
             "mechanism_group": "D1_VWAP_POSITION",
             "selection_rationale": "M1 d1_close_to_vwap_raw 的反向表达式替代 (互斥组 ME05_VWAP_GAP)。"},
        ),
    },
    {
        "model_id": "M3",
        "model_name": "PREDECLARED_DERIVED_MECHANISMS",
        "role": "DERIVED_SENSITIVITY_CANDIDATE",
        "feature_count": 5,
        "model_family": MODEL_FAMILY,
        "penalty": PENALTY,
        "design_purpose": "预声明派生机制模型: 检验阶段2.1 预声明派生表达是否能形成 "
                          "结构更明确的跨机制组合; 不是根据阶段2.2 表现自动挑选派生因子。",
        "features": (
            {"position": 1, "feature_name": "overrepair",
             "expected_direction": "-",
             "mechanism_group": "D1_PRICE_ACTION",
             "selection_rationale": "单日修复动能超过 3% 的截断部分 (派生自 "
                                     "d1_open_to_close_return_raw), 过度修复后回落风险。"},
            {"position": 2, "feature_name": "ma5_overheat_10",
             "expected_direction": "-",
             "mechanism_group": "D1_MA_POSITION",
             "selection_rationale": "收盘偏离 MA5 达 +10% 的过热二元项 (派生自 "
                                     "d1_close_to_ma5_raw), RAW_BINARY 处理。"},
            {"position": 3, "feature_name": "break_volume_abnormality",
             "expected_direction": "-",
             "mechanism_group": "D1_VOLUME_ACTIVITY",
             "selection_rationale": "断板日相对连板均量偏离的绝对对数幅度 (派生自 "
                                     "break_volume_ratio_vs_board_days), 双向异常。"},
            {"position": 4, "feature_name": "profit_chip_ratio",
             "expected_direction": "+",
             "mechanism_group": "D1_CHIP_DISTRIBUTION",
             "selection_rationale": "收盘价上方成交占比的补数 (派生自 "
                                     "volume_above_d1_close_ratio), 下方获利盘占比。"},
            {"position": 5, "feature_name": "late_day_sell_volume_ratio",
             "expected_direction": "-",
             "mechanism_group": "D1_LATE_DAY_PRESSURE",
             "selection_rationale": "尾盘卖出成交量占比, 与 M1 相同的尾盘抛压表达。"},
        ),
    },
)
MODEL_BY_ID = {m["model_id"]: m for m in CANDIDATE_MODELS}

# M3 派生因子 → 直接源表达式 (不得同时进入 M3)
DERIVED_DIRECT_SOURCES = dict(SELECTED_DERIVED_SOURCE_COLUMNS)

# 唯一入选特征 (首次出现在 M1/M2/M3 中的顺序)
UNIQUE_SELECTED_FEATURES = tuple(dict.fromkeys(
    f["feature_name"] for m in CANDIDATE_MODELS
    for f in m["features"]))

# 禁止作为任何模型成员的字段特征 (阶段2.3 硬门)
FORBIDDEN_FEATURE_TOKENS = (
    "recognition_score", "v004a", "v002", "v005", "existing_model_audit",
    "signal_date", "break_date", "month", "identifier", "d2_", "d3_",
    "target", "outcome", "bucket", "rank", "tail_loss", "label",
)

# AST 静态无训练审计 (任务 11): 只检查 import 节点与 Call 节点;
# 字符串常量 (协议文字、deny-list 定义、fixture 名称) 不参与判定, 不产生假阳性。
_FORBIDDEN_IMPORT_ROOTS = frozenset({
    "sklearn", "statsmodels", "xgboost", "lightgbm", "catboost",
    "torch", "tensorflow",
})
_FORBIDDEN_CALL_ATTRS = frozenset({
    "fit", "fit_predict", "predict", "predict_proba", "decision_function",
    "cross_val_score", "cross_validate",
})
_FORBIDDEN_CALL_NAMES = frozenset({
    "LogisticRegression", "GridSearchCV", "RandomizedSearchCV",
    "Lasso", "RFE", "cross_val_score", "cross_validate",
})
# 5 个证据无关构建函数的禁止证据字段 token (成员不可由证据变化驱动)
_EVIDENCE_TOKENS = ("effect_value", "signed_auc", "bootstrap", "lodo",
                    "board_direction", "flags")

_OUTPUT_ROLE = {
    "v004c_stage2_3_model_catalog_v001.csv": "model_catalog",
    "v004c_stage2_3_model_feature_spec_v001.csv": "model_feature_spec",
    "v004c_stage2_3_preprocessing_spec_v001.csv": "preprocessing_spec",
    "v004c_stage2_3_selected_feature_evidence_v001.csv": "selected_feature_evidence",
    "v004c_stage2_3_candidate_redundancy_audit_v001.csv": "candidate_redundancy_audit",
    "v004c_stage2_3_candidate_boundary_v001.csv": "candidate_boundary",
    "v004c_stage2_3_holdout_lock_audit_v001.csv": "holdout_lock_audit",
    "v004c_stage2_3_candidate_specs_v001.json": "candidate_specs",
    "v004c_stage2_3_evaluation_protocol_v001.json": "evaluation_protocol",
    "v004c_stage2_3_design_disclosure_v001.json": "design_disclosure",
    "v004c_stage2_3_manifest.json": "manifest",
    "v004c_stage2_3_review.md": "review",
    "git_head.txt": "provenance",
    "git_status_before.txt": "provenance",
    "git_status_after.txt": "provenance",
}

_STAGE2_1_INPUT_FILES = (
    "v004c_factor_dictionary_v001.csv",
    "v004c_feature_allowlist_primary_v001.csv",
    "v004c_feature_allowlist_sensitivity_v001.csv",
    "v004c_canonical_feature_map_v001.csv",
    "v004c_exact_duplicate_groups_v001.csv",
    "v004c_near_duplicate_pairs_v001.csv",
    "v004c_predeclared_derived_factor_spec_v001.csv",
    "v004c_predeclared_derived_factor_audit_v001.csv",
    "v004c_factor_dictionary_manifest.json",
)
_STAGE2_2_INPUT_FILES = (
    "v004c_stage2_2_feature_structure_v001.csv",
    "v004c_stage2_2_univariate_summary_dev_v001.csv",
    "v004c_stage2_2_cluster_bootstrap_dev_v001.csv",
    "v004c_stage2_2_board_subgroup_dev_v001.csv",
    "v004c_stage2_2_distribution_shift_unlabeled_v001.csv",
    "v004c_stage2_2_manifest.json",
    "v004c_stage2_2_review.md",
)

CATALOG_COLUMNS = (
    "model_id", "model_name", "role", "feature_count", "model_family",
    "penalty", "C", "solver", "fit_intercept", "class_weight", "max_iter",
    "tol", "random_seed", "direction_is_constraint", "design_purpose",
    "mechanism_groups",
)
FEATURE_SPEC_COLUMNS = (
    "model_id", "position", "feature_name", "source_or_derived",
    "source_column", "allowlist_tier", "mechanism_group",
    "underlying_mechanism", "analysis_type",
    "preprocess_policy", "mutual_exclusion_group_id", "expected_direction",
    "direction_is_constraint", "selection_rationale",
)
PREPROCESSING_COLUMNS = (
    "model_id", "spec_level", "feature_name", "model_family", "penalty", "C",
    "solver", "fit_intercept", "class_weight", "max_iter", "tol",
    "random_seed", "preprocess_policy", "missing_policy",
    "fold_clip_lower_quantile", "fold_clip_upper_quantile", "fold_clip_ddof",
    "fold_std_floor_fail_closed", "train_only_parameter_scope",
    "binary_allowed_values", "derived_row_wise_compute",
)
EVIDENCE_COLUMNS = (
    "feature_name", "allowlist_tier", "source_or_derived", "source_column",
    "mechanism_group", "preprocess_policy", "mutual_exclusion_group_id",
    "analysis_type", "effect_metric_name", "effect_value", "effect_direction",
    "signed_auc_raw", "lodo_sign_consistency", "lodo_max_absolute_delta",
    "board_direction_agreement_status", "board_2_effect", "board_3_effect",
    "bootstrap_valid", "bootstrap_ci_2_5", "bootstrap_ci_97_5",
    "bootstrap_ci_crosses_zero", "bootstrap_low_valid_replicates",
    "distribution_shift", "flags",
)
REDUNDANCY_COLUMNS = (
    "model_id", "position_a", "position_b", "feature_a", "feature_b",
    "dev_non_null_pair_count", "dev_spearman_rho", "locked_non_null_pair_count",
    "locked_spearman_rho", "absolute_dev_rho", "absolute_locked_rho",
    "maximum_absolute_rho", "absolute_rho_shift", "redundancy_flag",
)
BOUNDARY_COLUMNS = (
    "feature_name", "allowlist_tier", "mechanism_group", "selected_any_model",
    "selected_models", "boundary_status", "boundary_note",
)


def _empty_to_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "nat", "none", "<na>"):
        return ""
    return text


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


# 锁定审计检查名 → 中文标签 (manifest/review 使用; 避免英文 token 进入内容扫描)
_AUDIT_CHECK_LABELS = {
    "stage2_3_not_read_training_d1_table": "未读取training D1表",
    "stage2_3_not_read_existing_model_audit": "未读取existing model audit",
    "snapshot_label_columns_not_accessed": "snapshot标签列未访问",
    "july_label_columns_not_accessed": "七月标签列未访问",
    "actual_loaded_input_files_match_allowed": "实际输入文件与允许清单精确相等",
    "actual_loaded_input_files_disjoint_forbidden": "实际输入文件与禁止清单交集为空",
    "candidate_membership_fixed_after_june_development_review":
        "模型成员六月开发审查后人工固定",
    "membership_independent_of_evidence_model_catalog": "模型目录不依赖证据值",
    "membership_independent_of_evidence_model_feature_spec": "特征规格不依赖证据值",
    "membership_independent_of_evidence_preprocessing_spec": "预处理规格不依赖证据值",
    "membership_independent_of_evidence_candidate_specs_json": "候选JSON不依赖证据值",
    "membership_independent_of_evidence_evaluation_protocol_json": "协议JSON不依赖证据值",
    "no_model_fit_call": "未调用模型拟合接口",
    "no_prediction_generated": "未生成预测",
    "no_automatic_feature_selection": "无自动特征选择",
    "no_hyperparameter_search": "无超参数搜索",
    "no_threshold_search": "无阈值搜索",
    "no_walk_forward_run": "未运行Walk-forward",
    "model_count_is_4": "模型数恰好4",
    "substantive_model_count_is_3": "实质模型数恰好3",
    "m1_m2_m3_feature_count_is_5": "M1/M2/M3各5个特征",
    "candidate_boundary_rows_is_59": "59行候选边界完整",
    "selected_unique_feature_count_is_12": "12个唯一入选特征完整",
    "RUNTIME_TARGET_COLUMN_ACCESS": "运行时Target列访问=False",
    "JULY_LABEL_ACCESS": "七月标签访问=False",
    "AUTOMATIC_TARGET_DRIVEN_SELECTION": "自动Target驱动选择=False",
    "JUNE_EVIDENCE_USED_FOR_MANUAL_DESIGN_DECLARED": "六月证据用于人工设计已声明",
    "SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS": "规格在七月揭示前冻结=True",
    "JUNE_DEVELOPMENT_EVIDENCE_DISCLOSED": "六月开发证据参与人工设计已披露",
    "no_july_target_metric_in_csv_columns": "CSV列无七月标签指标",
    "no_july_target_metric_in_review": "review无七月标签指标",
    "no_july_target_metric_in_design_disclosure": "design disclosure无七月标签指标",
}


# ---------------------------------------------------------------------------
# 版本链校验 (任务一)
# ---------------------------------------------------------------------------
def validate_git_chain(git_root: Path, stage2_2_ref: str) -> dict[str, Any]:
    """校验分支 / 三个来源 tag 目标 / 阶段2.2 数据提交祖先关系 (fail closed)。"""
    try:
        stage22_target = _run_git(
            ["git", "rev-parse", "--verify", f"{stage2_2_ref}^{{}}"],
            cwd=git_root).strip()
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_not_found: source ref {stage2_2_ref!r} 无法解析 ({exc})"
        ) from exc
    validate_git_head(stage22_target)
    if stage2_2_ref != EXPECTED_STAGE2_2_TAG:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: stage2-2-ref 必须为 "
            f"{EXPECTED_STAGE2_2_TAG!r}, 实际 {stage2_2_ref!r}")
    if stage22_target != EXPECTED_STAGE2_2_DATA_COMMIT:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: tag {stage2_2_ref} 目标必须为 "
            f"{EXPECTED_STAGE2_2_DATA_COMMIT}, 实际 {stage22_target}")

    for tag, expected in ((EXPECTED_STAGE2_1_TAG, EXPECTED_STAGE2_1_DATA_COMMIT),
                          (EXPECTED_STAGE1_TAG, EXPECTED_STAGE1_DATA_COMMIT)):
        try:
            target = _run_git(
                ["git", "rev-parse", "--verify", f"{tag}^{{}}"],
                cwd=git_root).strip()
        except DatasetValidationError as exc:
            raise DatasetValidationError(
                f"source_ref_not_found: source ref {tag!r} 无法解析 ({exc})"
            ) from exc
        validate_git_head(target)
        if target != expected:
            raise DatasetValidationError(
                f"source_ref_commit_mismatch: tag {tag} 目标必须为 {expected}, "
                f"实际 {target}")

    branch = _run_git(["git", "branch", "--show-current"], cwd=git_root).strip()
    if branch != EXPECTED_BRANCH:
        raise DatasetValidationError(
            f"branch_mismatch: 分支必须为 {EXPECTED_BRANCH!r}, 实际 {branch!r}")

    try:
        _run_git(["git", "merge-base", "--is-ancestor",
                  EXPECTED_STAGE2_2_DATA_COMMIT, "HEAD"], cwd=git_root)
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: 阶段2.2数据提交 "
            f"{EXPECTED_STAGE2_2_DATA_COMMIT[:12]} 不是当前 HEAD 祖先 ({exc})"
        ) from exc

    return {
        "branch": branch,
        "stage2_2_ref": stage2_2_ref,
        "stage2_2_tag_target": stage22_target,
        "stage2_1_tag_target": EXPECTED_STAGE2_1_DATA_COMMIT,
        "stage1_tag_target": EXPECTED_STAGE1_DATA_COMMIT,
        "data_commit_ancestor": True,
    }


# ---------------------------------------------------------------------------
# 输入读取 (任务五; 只读允许列, 禁止标签)
# ---------------------------------------------------------------------------
def load_inputs(
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_2_dir: str | Path,
    snapshot_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """读取阶段1 (仅允许列) / 阶段2.1 / 阶段2.2 输入。

    snapshot 使用 usecols 物理限制为 SNAPSHOT_READ_COLUMNS,
    从读取层面保证标签/结果/审计列不可见。
    """
    d1 = Path(stage1_dir)
    d21 = Path(stage2_1_dir)
    d22 = Path(stage2_2_dir)
    for p, name in ((d1 / "v004c_d1_data_manifest.json", "阶段1 manifest"),
                    (d21 / "v004c_factor_dictionary_manifest.json", "阶段2.1 manifest"),
                    (d22 / "v004c_stage2_2_manifest.json", "阶段2.2 manifest")):
        if not p.exists():
            raise DatasetValidationError(f"{name} 不存在: {p}")
    with (d1 / "v004c_d1_data_manifest.json").open("r", encoding="utf-8") as handle:
        stage1_manifest = json.load(handle)
    with (d21 / "v004c_factor_dictionary_manifest.json").open("r", encoding="utf-8") as handle:
        stage21_manifest = json.load(handle)
    with (d22 / "v004c_stage2_2_manifest.json").open("r", encoding="utf-8") as handle:
        stage22_manifest = json.load(handle)

    # 实际打开/读取过的输入文件 (stage, basename) 对; 用于运行时输入审计
    loaded_input_files: list[tuple[str, str]] = [
        ("stage1", "v004c_d1_data_manifest.json"),
        ("stage2_1", "v004c_factor_dictionary_manifest.json"),
        ("stage2_2", "v004c_stage2_2_manifest.json"),
    ]

    if snapshot_override is not None:
        # override 提供数据, 但文件仍被 verify_inputs 验证 SHA → 计入实际读取集合
        snapshot = snapshot_override
        loaded_input_files.append(("stage1", "v004c_d1_snapshot_v001.csv"))
    else:
        snapshot = pd.read_csv(
            d1 / "v004c_d1_snapshot_v001.csv",
            usecols=list(SNAPSHOT_READ_COLUMNS),
            dtype={SIGNAL_DATE_COL: str},
            float_precision="round_trip")
        loaded_input_files.append(("stage1", "v004c_d1_snapshot_v001.csv"))
    lineage = pd.read_csv(d1 / "v004c_d1_column_lineage.csv", dtype=str)
    loaded_input_files.append(("stage1", "v004c_d1_column_lineage.csv"))

    stage21_frames: dict[str, pd.DataFrame] = {}
    for name in _STAGE2_1_INPUT_FILES:
        stage21_frames[name] = pd.read_csv(d21 / name, dtype=str)
        loaded_input_files.append(("stage2_1", name))
    stage22_frames: dict[str, pd.DataFrame] = {}
    for name in _STAGE2_2_INPUT_FILES:
        if name.endswith(".json") or name.endswith(".md"):
            continue
        stage22_frames[name] = pd.read_csv(
            d22 / name, dtype=str, float_precision="round_trip")
        loaded_input_files.append(("stage2_2", name))

    review_text = (d22 / "v004c_stage2_2_review.md").read_text(encoding="utf-8")
    loaded_input_files.append(("stage2_2", "v004c_stage2_2_review.md"))

    return {
        "snapshot": snapshot,
        "lineage": lineage,
        "stage1_manifest": stage1_manifest,
        "stage21_manifest": stage21_manifest,
        "stage22_manifest": stage22_manifest,
        "stage21_frames": stage21_frames,
        "stage22_frames": stage22_frames,
        "stage22_review": review_text,
        "loaded_input_files": frozenset(loaded_input_files),
        "paths": {
            "stage1": d1,
            "stage2_1": d21,
            "stage2_2": d22,
        },
    }


def _sha_matches(path: Path, manifest: dict, name: str) -> tuple[bool, str, str]:
    record = manifest.get("output_files", {}).get(name)
    if record is None:
        return False, "", ""
    actual = _sha256(path)
    return actual == record.get("sha256"), actual, str(record.get("sha256") or "")


def verify_inputs(
    payload: dict[str, Any],
    *,
    current_git_root: Path,
    enforce_input_sha: bool = True,
) -> dict[str, Any]:
    """验证版本链 / 结构门 / 输入 SHA 与对应 manifest 一致 (fail closed)。"""
    checks: dict[str, Any] = {}
    d1 = Path(payload["paths"]["stage1"])
    d21 = Path(payload["paths"]["stage2_1"])
    d22 = Path(payload["paths"]["stage2_2"])
    stage1_manifest = payload["stage1_manifest"]
    stage21_manifest = payload["stage21_manifest"]
    stage22_manifest = payload["stage22_manifest"]
    snapshot = payload["snapshot"]
    lineage = payload["lineage"]

    if stage1_manifest.get("dataset_version") != "v004c-d1-dataset-0.1":
        raise DatasetValidationError(
            f"stage1_manifest_version_mismatch: 阶段1 manifest dataset_version 不符: "
            f"{stage1_manifest.get('dataset_version')!r}")
    if stage21_manifest.get("dataset_version") != "v004c-factor-dictionary-0.1":
        raise DatasetValidationError(
            f"stage2_1_manifest_version_mismatch: 阶段2.1 manifest dataset_version 不符: "
            f"{stage21_manifest.get('dataset_version')!r}")
    if stage22_manifest.get("stage2_2_version") != "v004c-stage2-2-univariate-0.1":
        raise DatasetValidationError(
            f"stage2_2_manifest_version_mismatch: 阶段2.2 manifest 版本不符: "
            f"{stage22_manifest.get('stage2_2_version')!r}")
    if stage1_manifest.get("git_head") != EXPECTED_STAGE1_GENERATOR_COMMIT:
        raise DatasetValidationError(
            f"stage1_manifest_git_head_mismatch: 阶段1 manifest git_head 不符: "
            f"{stage1_manifest.get('git_head')!r}")
    if stage21_manifest.get("git_head") != EXPECTED_STAGE2_1_CODE_COMMIT:
        raise DatasetValidationError(
            f"stage2_1_manifest_git_head_mismatch: 阶段2.1 manifest git_head 不符: "
            f"{stage21_manifest.get('git_head')!r}")
    if stage22_manifest.get("git_head") != EXPECTED_STAGE2_2_CODE_COMMIT:
        raise DatasetValidationError(
            f"stage2_2_manifest_git_head_mismatch: 阶段2.2 manifest git_head 不符: "
            f"{stage22_manifest.get('git_head')!r}")

    # 目录 git_head.txt 校验
    for subdir, expected, label in (
            (d21, EXPECTED_STAGE2_1_CODE_COMMIT, "阶段2.1"),
            (d22, EXPECTED_STAGE2_2_CODE_COMMIT, "阶段2.2")):
        gh_text = (subdir / "git_head.txt").read_text(encoding="utf-8").strip()
        if gh_text != expected:
            raise DatasetValidationError(
                f"git_head_file_mismatch: {label} git_head.txt 应为 {expected}, "
                f"实际 {gh_text!r}")
    checks["stage21_git_head_file"] = EXPECTED_STAGE2_1_CODE_COMMIT
    checks["stage22_git_head_file"] = EXPECTED_STAGE2_2_CODE_COMMIT

    # 结构门
    checks["input_rows"] = int(len(snapshot))
    checks["signal_dates"] = int(snapshot[SIGNAL_DATE_COL].nunique())
    checks["lineage_rows"] = int(len(lineage))
    if checks["input_rows"] != EXPECTED_GATES["input_rows"]:
        raise DatasetValidationError(
            f"input_rows_mismatch: snapshot 行数应为 {EXPECTED_GATES['input_rows']}, "
            f"实际 {checks['input_rows']}")
    if checks["signal_dates"] != EXPECTED_GATES["signal_dates"]:
        raise DatasetValidationError(
            f"signal_dates_mismatch: 信号日数应为 {EXPECTED_GATES['signal_dates']}, "
            f"实际 {checks['signal_dates']}")
    if checks["lineage_rows"] != EXPECTED_GATES["lineage_rows"]:
        raise DatasetValidationError(
            f"lineage_rows_mismatch: lineage 行数应为 {EXPECTED_GATES['lineage_rows']}, "
            f"实际 {checks['lineage_rows']}")

    # 时间切片门 (只用 signal_date 列)
    dev = snapshot[(snapshot[SIGNAL_DATE_COL] >= DEVELOPMENT_START)
                   & (snapshot[SIGNAL_DATE_COL] <= DEVELOPMENT_END)]
    locked = snapshot[(snapshot[SIGNAL_DATE_COL] >= LOCKED_START)
                      & (snapshot[SIGNAL_DATE_COL] <= LOCKED_END)]
    checks["dev_rows"] = int(len(dev))
    checks["locked_rows"] = int(len(locked))
    if checks["dev_rows"] != EXPECTED_GATES["dev_rows"]:
        raise DatasetValidationError(
            f"dev_rows_mismatch: 六月开发集行数应为 {EXPECTED_GATES['dev_rows']}, "
            f"实际 {checks['dev_rows']}")
    if checks["locked_rows"] != EXPECTED_GATES["locked_rows"]:
        raise DatasetValidationError(
            f"locked_rows_mismatch: 七月锁定切片行数应为 {EXPECTED_GATES['locked_rows']}, "
            f"实际 {checks['locked_rows']}")

    # snapshot 读取列门: 除 signal_date 外全部必须为阶段1允许特征列
    lineage_map = {
        str(r["column_name"]): r for _, r in lineage.iterrows()}
    allowed_mask = lineage["allowed_for_future_feature_analysis"].astype(
        str).str.strip().str.lower().isin(("true", "1", "yes"))
    allowed_cols = set(lineage.loc[allowed_mask, "column_name"].astype(str))
    read_cols = set(SNAPSHOT_READ_COLUMNS) - {SIGNAL_DATE_COL}
    bad_read = read_cols - allowed_cols
    if bad_read:
        raise DatasetValidationError(
            f"forbidden_snapshot_column_read: 读取列不在阶段1允许特征列内: "
            f"{sorted(bad_read)}")
    label_role_cols = set(lineage.loc[
        lineage["column_role"].astype(str).isin(
            ("label", "outcome_audit", "existing_model_audit")),
        "column_name"].astype(str))
    overlap = read_cols & label_role_cols
    if overlap:
        raise DatasetValidationError(
            f"label_column_read: 读取列包含标签/审计列: {sorted(overlap)}")
    sens_overlap = read_cols & set(TARGET_BLIND_SENSITIVE_COLUMNS)
    if sens_overlap:
        raise DatasetValidationError(
            f"sensitive_column_read: 读取列包含敏感列: {sorted(sens_overlap)}")
    checks["snapshot_read_columns"] = sorted(SNAPSHOT_READ_COLUMNS)
    checks["snapshot_label_columns_not_read"] = True
    checks["lineage_lookup"] = len(lineage_map)

    # 禁止输入文件门
    for name in FORBIDDEN_INPUT_FILENAMES:
        if (d1 / name).exists() and enforce_input_sha:
            # 文件存在并不禁止 (冻结资产), 禁止的是读取; 读取清单不含它们
            pass

    # 输入文件 SHA 与对应 manifest 一致
    sha_checks: dict[str, Any] = {}
    stage1_files = {
        "v004c_d1_snapshot_v001.csv": "snapshot",
        "v004c_d1_column_lineage.csv": "lineage",
    }
    for name, role in stage1_files.items():
        path = d1 / name
        ok, actual, recorded = _sha_matches(path, stage1_manifest, name)
        sha_checks[name] = {
            "role": role, "stage": "stage1",
            "path": _repo_relative_posix(path, current_git_root),
            "bytes": int(path.stat().st_size),
            "sha256": actual,
            "verified": ok,
        }
        if not ok and enforce_input_sha:
            raise DatasetValidationError(
                f"input_sha_mismatch: 阶段1输入 {name} 与 manifest 记录不一致 "
                f"(record {recorded} vs local {actual})")
    man_sha = _sha256(d1 / "v004c_d1_data_manifest.json")
    audit_ok = man_sha == STAGE1_MANIFEST_AUDIT_REFERENCE
    checks["stage1_manifest_audit_reference_sha256"] = STAGE1_MANIFEST_AUDIT_REFERENCE
    checks["stage1_manifest_audit_reference_verified"] = bool(audit_ok)
    sha_checks["v004c_d1_data_manifest.json"] = {
        "role": "manifest", "stage": "stage1",
        "path": _repo_relative_posix(d1 / "v004c_d1_data_manifest.json",
                                     current_git_root),
        "bytes": int((d1 / "v004c_d1_data_manifest.json").stat().st_size),
        "sha256": man_sha,
        "verified": bool(audit_ok),
    }
    if not audit_ok and enforce_input_sha:
        raise DatasetValidationError(
            f"stage1_manifest_audit_reference_mismatch: 阶段1 manifest SHA 与审计参考 "
            f"{STAGE1_MANIFEST_AUDIT_REFERENCE} 不一致 (local={man_sha})")

    for name in _STAGE2_1_INPUT_FILES:
        if name == "v004c_factor_dictionary_manifest.json":
            continue
        path = d21 / name
        ok, actual, recorded = _sha_matches(path, stage21_manifest, name)
        sha_checks[name] = {
            "role": "input", "stage": "stage2_1",
            "path": _repo_relative_posix(path, current_git_root),
            "bytes": int(path.stat().st_size),
            "sha256": actual,
            "verified": ok,
        }
        if not ok and enforce_input_sha:
            raise DatasetValidationError(
                f"input_sha_mismatch: 阶段2.1输入 {name} 与 manifest 记录不一致 "
                f"(record {recorded} vs local {actual})")
    d21_manifest_sha = _sha256(d21 / "v004c_factor_dictionary_manifest.json")
    sha_checks["v004c_factor_dictionary_manifest.json"] = {
        "role": "manifest", "stage": "stage2_1",
        "path": _repo_relative_posix(
            d21 / "v004c_factor_dictionary_manifest.json", current_git_root),
        "bytes": int((d21 / "v004c_factor_dictionary_manifest.json").stat().st_size),
        "sha256": d21_manifest_sha,
        "verified": True,
    }

    for name in _STAGE2_2_INPUT_FILES:
        if name == "v004c_stage2_2_manifest.json":
            continue
        path = d22 / name
        ok, actual, recorded = _sha_matches(path, stage22_manifest, name)
        sha_checks[name] = {
            "role": "input", "stage": "stage2_2",
            "path": _repo_relative_posix(path, current_git_root),
            "bytes": int(path.stat().st_size),
            "sha256": actual,
            "verified": ok,
        }
        if not ok and enforce_input_sha:
            raise DatasetValidationError(
                f"input_sha_mismatch: 阶段2.2输入 {name} 与 manifest 记录不一致 "
                f"(record {recorded} vs local {actual})")
    d22_manifest_sha = _sha256(d22 / "v004c_stage2_2_manifest.json")
    sha_checks["v004c_stage2_2_manifest.json"] = {
        "role": "manifest", "stage": "stage2_2",
        "path": _repo_relative_posix(
            d22 / "v004c_stage2_2_manifest.json", current_git_root),
        "bytes": int((d22 / "v004c_stage2_2_manifest.json").stat().st_size),
        "sha256": d22_manifest_sha,
        "verified": True,
    }
    checks["input_sha_verification"] = sha_checks
    checks["input_sha_all_verified"] = all(
        v.get("verified", False) for v in sha_checks.values())
    if not checks["input_sha_all_verified"] and enforce_input_sha:
        raise DatasetValidationError(
            "input_sha_mismatch: 存在输入 SHA 与 manifest 不一致 (fail closed)")
    checks["current_git_root"] = str(current_git_root)
    return checks


# ---------------------------------------------------------------------------
# 准入辅助 (阶段2.1 allowlist 查询)
# ---------------------------------------------------------------------------
def _allowlist_lookup(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    """从阶段2.1 primary/sensitivity allowlist 构建特征准入查询表。"""
    out: dict[str, dict[str, str]] = {}
    primary = payload["stage21_frames"]["v004c_feature_allowlist_primary_v001.csv"]
    sensitivity = payload["stage21_frames"]["v004c_feature_allowlist_sensitivity_v001.csv"]
    for frame, tier in ((primary, "PRIMARY"), (sensitivity, "SENSITIVITY")):
        for _, row in frame.iterrows():
            name = _empty_to_str(row.get("feature_name"))
            out[name] = {
                "allowlist_tier": tier,
                "source_or_derived": _empty_to_str(row.get("source_or_derived")),
                "source_column": _empty_to_str(row.get("source_column")),
                "mechanism_group": _empty_to_str(row.get("mechanism_group")),
                "preprocess_policy": _empty_to_str(row.get("preprocess_policy")),
                "mutual_exclusion_group_id": _empty_to_str(
                    row.get("mutual_exclusion_group_id")),
            }
    return out


# ---------------------------------------------------------------------------
# 证据无关的固定规格构建 (任务七/八/十/十一; 不接受证据输入)
# ---------------------------------------------------------------------------
def build_model_catalog() -> pd.DataFrame:
    """模型目录 (4 行): 全部字段来自人工固定常量。"""
    rows = []
    for m in CANDIDATE_MODELS:
        rows.append({
            "model_id": m["model_id"],
            "model_name": m["model_name"],
            "role": m["role"],
            "feature_count": int(m["feature_count"]),
            "model_family": m["model_family"],
            "penalty": m["penalty"],
            "C": C_VALUE if m["model_family"] == MODEL_FAMILY else None,
            "solver": SOLVER if m["model_family"] == MODEL_FAMILY else None,
            "fit_intercept": FIT_INTERCEPT if m["model_family"] == MODEL_FAMILY else None,
            "class_weight": CLASS_WEIGHT,
            "max_iter": MAX_ITER if m["model_family"] == MODEL_FAMILY else None,
            "tol": TOL if m["model_family"] == MODEL_FAMILY else None,
            "random_seed": RANDOM_SEED if m["model_family"] == MODEL_FAMILY else None,
            "direction_is_constraint": False,
            "design_purpose": m["design_purpose"],
            "mechanism_groups": "|".join(dict.fromkeys(
                f["mechanism_group"] for f in m["features"])),
        })
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


def _hard_membership_checks(
    payload: dict[str, Any],
    allowlist: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """任务八硬门: 成员数 / 准入 / 禁止字段 / 精确重复 / 互斥组 / 派生源。"""
    checks: dict[str, Any] = {}
    summary_by_name = {
        str(r["feature_name"]): r
        for _, r in payload["stage22_frames"][
            "v004c_stage2_2_univariate_summary_dev_v001.csv"].iterrows()}
    exact_dup = payload["stage21_frames"][
        "v004c_exact_duplicate_groups_v001.csv"]
    near_dup = payload["stage21_frames"][
        "v004c_near_duplicate_pairs_v001.csv"]

    feature_names = {f["feature_name"] for m in CANDIDATE_MODELS
                     for f in m["features"]}
    if any(name not in allowlist for name in feature_names):
        raise DatasetValidationError(
            f"feature_not_in_allowlist: 存在不在任一 allowlist 的成员: "
            f"{sorted(feature_names - set(allowlist))}")
    if any(name not in summary_by_name for name in feature_names):
        raise DatasetValidationError(
            f"feature_not_in_stage2_2: 存在不在阶段2.2 59 因子中的成员: "
            f"{sorted(feature_names - set(summary_by_name))}")

    for m in CANDIDATE_MODELS:
        mid = m["model_id"]
        feats = m["features"]
        names = [f["feature_name"] for f in feats]
        if len(names) != len(set(names)):
            raise DatasetValidationError(
                f"duplicate_feature_in_model: {mid} 内部存在重复成员")
        if len(names) != m["feature_count"]:
            raise DatasetValidationError(
                f"feature_count_mismatch: {mid} 声明 {m['feature_count']} 个特征, "
                f"实际 {len(names)}")
        if mid == "M0" and names:
            raise DatasetValidationError("M0 不得包含特征")
        for f in feats:
            name = f["feature_name"]
            for token in FORBIDDEN_FEATURE_TOKENS:
                if token in name.lower():
                    raise DatasetValidationError(
                        f"forbidden_feature_name: {mid} 成员 {name} 含禁止 token "
                        f"{token!r}")
            if _empty_to_str(summary_by_name[name].get("analysis_type")) == "BUCKET":
                raise DatasetValidationError(
                    f"bucket_feature_in_model: {mid} 成员 {name} 为 BUCKET 分析类型, "
                    f"禁止进入候选模型")
        # 互斥组: 每模型每个互斥组最多一次
        me_groups = [_empty_to_str(allowlist[n]["mutual_exclusion_group_id"])
                     for n in names]
        seen: set[str] = set()
        for g in me_groups:
            if not g:
                continue
            if g in seen:
                raise DatasetValidationError(
                    f"mutual_exclusion_violation: {mid} 互斥组 {g} 出现两次 "
                    f"(成员 {names})")
            seen.add(g)
        checks[f"{mid}_mutual_exclusion_groups"] = sorted(
            g for g in me_groups if g)

    # 精确重复 / canonical+alias 共存 (阶段2.1 冻结映射)
    canonical_alias_pairs = []
    for _, r in exact_dup.iterrows():
        canonical_alias_pairs.append(
            (_empty_to_str(r.get("canonical_source_column")),
             _empty_to_str(r.get("alias_source_column"))))
    for m in CANDIDATE_MODELS:
        names = [f["feature_name"] for f in m["features"]]
        name_set = set(names)
        for canon, alias in canonical_alias_pairs:
            if not canon or not alias:
                continue
            if canon in name_set and alias in name_set:
                raise DatasetValidationError(
                    f"canonical_alias_coexistence: {m['model_id']} 同时包含 "
                    f"canonical {canon} 与 alias {alias}")
    # 阶段2.1 数值级近重复 (|spearman| >= 0.999999, 即精确重复/精确反向)
    exact_inverse_pairs = []
    for _, r in near_dup.iterrows():
        left = _empty_to_str(r.get("left_column"))
        right = _empty_to_str(r.get("right_column"))
        rho = pd.to_numeric(r.get("spearman"), errors="coerce")
        if pd.notna(rho) and abs(float(rho)) >= 0.999999:
            exact_inverse_pairs.append((left, right))
    for m in CANDIDATE_MODELS:
        names = [f["feature_name"] for f in m["features"]]
        name_set = set(names)
        for left, right in exact_inverse_pairs:
            if left in name_set and right in name_set:
                raise DatasetValidationError(
                    f"exact_duplicate_or_inverse_coexistence: {m['model_id']} "
                    f"同时包含 {left} 与 {right}")
    checks["exact_duplicate_pairs_checked"] = len(canonical_alias_pairs)
    checks["exact_inverse_pairs_checked"] = len(exact_inverse_pairs)

    # M1/M2/M3 各自的准入层级
    m1 = MODEL_BY_ID["M1"]["features"]
    m2 = MODEL_BY_ID["M2"]["features"]
    m3 = MODEL_BY_ID["M3"]["features"]
    if any(allowlist[f["feature_name"]]["allowlist_tier"] != "PRIMARY"
           for f in m1):
        raise DatasetValidationError("M1 存在非 primary 成员 (必须全部为 primary)")
    m2_sens = [f["feature_name"] for f in m2
               if allowlist[f["feature_name"]]["allowlist_tier"] != "PRIMARY"]
    if sorted(m2_sens) != sorted((
            "high_zone_amount_ratio", "late_day_sell_amount_ratio",
            "d1_vwap_to_close_gap")):
        raise DatasetValidationError(
            f"M2 的 sensitivity 替代表达固定为 3 个, 实际 {sorted(m2_sens)}")
    m3_names = [f["feature_name"] for f in m3]
    if m3_names[-1] != "late_day_sell_volume_ratio":
        raise DatasetValidationError("M3 第 5 个特征必须为 late_day_sell_volume_ratio")
    for name in m3_names[:-1]:
        row = allowlist[name]
        if row["allowlist_tier"] != "PRIMARY" or row["source_or_derived"] != "derived":
            raise DatasetValidationError(
                f"M3 派生成员 {name} 必须为冻结 derived primary")
    # M3 派生与直接源变量不得共存
    m3_set = set(m3_names)
    for dname, src in DERIVED_DIRECT_SOURCES.items():
        if dname in m3_set and src in m3_set:
            raise DatasetValidationError(
                f"derived_source_coexistence: M3 同时包含派生 {dname} 与直接源 {src}")
    checks["m1_all_primary"] = True
    checks["m2_sensitivity_alternatives"] = sorted(m2_sens)
    checks["m3_derived_primary"] = True
    return checks


def build_model_feature_spec(
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """特征规格 (15 行) + 硬门检查。

    层级/机制/预处理/互斥组全部来自阶段2.1 allowlist (冻结准入资产);
    analysis_type 来自阶段2.2 结构字段, 不读取任何证据值。
    """
    allowlist = _allowlist_lookup(payload)
    summary = payload["stage22_frames"][
        "v004c_stage2_2_univariate_summary_dev_v001.csv"]
    summary_by_name = {
        str(r["feature_name"]): r for _, r in summary.iterrows()}
    checks = _hard_membership_checks(payload, allowlist)

    rows = []
    for m in CANDIDATE_MODELS:
        for f in m["features"]:
            name = f["feature_name"]
            info = allowlist[name]
            rows.append({
                "model_id": m["model_id"],
                "position": int(f["position"]),
                "feature_name": name,
                "source_or_derived": info["source_or_derived"],
                "source_column": info["source_column"],
                "allowlist_tier": info["allowlist_tier"],
                "mechanism_group": info["mechanism_group"],
                # 派生因子 allowlist 机制组为 PREDECLARED_DERIVED;
                # underlying_mechanism 记录任务声明的底层机制
                "underlying_mechanism": f["mechanism_group"],
                "analysis_type": _empty_to_str(
                    summary_by_name[name].get("analysis_type")),
                "preprocess_policy": info["preprocess_policy"],
                "mutual_exclusion_group_id": info["mutual_exclusion_group_id"],
                "expected_direction": f["expected_direction"],
                "direction_is_constraint": False,
                "selection_rationale": f["selection_rationale"],
            })
    return pd.DataFrame(rows, columns=FEATURE_SPEC_COLUMNS), checks


def build_preprocessing_spec(catalog: pd.DataFrame,
                             feature_spec: pd.DataFrame) -> pd.DataFrame:
    """预处理规格: 4 行模型级 + 15 行特征级 (任务十/十一)。"""
    model_rows = []
    for _, r in catalog.iterrows():
        model_rows.append({
            "model_id": r["model_id"],
            "spec_level": "MODEL",
            "feature_name": "",
            "model_family": r["model_family"],
            "penalty": r["penalty"],
            "C": r["C"], "solver": r["solver"],
            "fit_intercept": r["fit_intercept"],
            "class_weight": r["class_weight"],
            "max_iter": r["max_iter"], "tol": r["tol"],
            "random_seed": r["random_seed"],
            "preprocess_policy": "",
            "missing_policy": MISSING_POLICY,
            "fold_clip_lower_quantile": FOLD_CLIP_LOWER_QUANTILE,
            "fold_clip_upper_quantile": FOLD_CLIP_UPPER_QUANTILE,
            "fold_clip_ddof": FOLD_CLIP_DDOF,
            "fold_std_floor_fail_closed": FOLD_STD_FLOOR,
            "train_only_parameter_scope": "训练窗口内非缺失值; 测试窗口不参与",
            "binary_allowed_values": "|".join(str(v) for v in BINARY_ALLOWED_VALUES),
            "derived_row_wise_compute": "逐行公式, 不用标签, 不用跨行统计",
        })
    feature_rows = []
    for _, r in feature_spec.iterrows():
        policy = r["preprocess_policy"]
        feature_rows.append({
            "model_id": r["model_id"],
            "spec_level": "FEATURE",
            "feature_name": r["feature_name"],
            "model_family": MODEL_BY_ID[r["model_id"]]["model_family"],
            "penalty": PENALTY if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else "none",
            "C": C_VALUE if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else None,
            "solver": SOLVER if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else None,
            "fit_intercept": FIT_INTERCEPT if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else None,
            "class_weight": CLASS_WEIGHT,
            "max_iter": MAX_ITER if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else None,
            "tol": TOL if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else None,
            "random_seed": RANDOM_SEED if MODEL_BY_ID[r["model_id"]]["model_family"] == MODEL_FAMILY else None,
            "preprocess_policy": policy,
            "missing_policy": MISSING_POLICY,
            "fold_clip_lower_quantile": FOLD_CLIP_LOWER_QUANTILE,
            "fold_clip_upper_quantile": FOLD_CLIP_UPPER_QUANTILE,
            "fold_clip_ddof": FOLD_CLIP_DDOF,
            "fold_std_floor_fail_closed": FOLD_STD_FLOOR,
            "train_only_parameter_scope": "训练窗口内非缺失值; 测试窗口不参与",
            "binary_allowed_values": "|".join(str(v) for v in BINARY_ALLOWED_VALUES),
            "derived_row_wise_compute": "逐行公式, 不用标签, 不用跨行统计",
        })
    return pd.DataFrame(model_rows + feature_rows, columns=PREPROCESSING_COLUMNS)


# ---------------------------------------------------------------------------
# 证据提取 (任务十九; 从阶段2.2 冻结资产原样提取, 不重算关联)
# ---------------------------------------------------------------------------
def build_selected_feature_evidence(payload: dict[str, Any]) -> pd.DataFrame:
    """12 个唯一入选特征证据表 (首次出现在 M1/M2/M3 中的顺序)。"""
    summary = payload["stage22_frames"][
        "v004c_stage2_2_univariate_summary_dev_v001.csv"]
    bootstrap = payload["stage22_frames"][
        "v004c_stage2_2_cluster_bootstrap_dev_v001.csv"]
    shift = payload["stage22_frames"][
        "v004c_stage2_2_distribution_shift_unlabeled_v001.csv"]
    summary_by_name = {str(r["feature_name"]): r for _, r in summary.iterrows()}
    bootstrap_by_name = {
        str(r["feature_name"]): r for _, r in bootstrap.iterrows()}
    shift_by_name = {str(r["feature_name"]): r for _, r in shift.iterrows()}

    rows = []
    for name in UNIQUE_SELECTED_FEATURES:
        s = summary_by_name[name]
        b = bootstrap_by_name[name]
        sh = shift_by_name[name]
        rows.append({
            "feature_name": name,
            "allowlist_tier": _empty_to_str(s.get("allowlist_tier")),
            "source_or_derived": _empty_to_str(s.get("source_or_derived")),
            "source_column": _empty_to_str(s.get("source_column")),
            "mechanism_group": _empty_to_str(s.get("mechanism_group")),
            "preprocess_policy": _empty_to_str(s.get("preprocess_policy")),
            "mutual_exclusion_group_id": _empty_to_str(
                s.get("mutual_exclusion_group_id")),
            "analysis_type": _empty_to_str(s.get("analysis_type")),
            "effect_metric_name": _empty_to_str(s.get("effect_metric_name")),
            "effect_value": s.get("effect_value"),
            "effect_direction": _empty_to_str(s.get("effect_direction")),
            "signed_auc_raw": s.get("signed_auc_raw"),
            "lodo_sign_consistency": s.get("lodo_sign_consistency"),
            "lodo_max_absolute_delta": s.get("lodo_max_absolute_delta"),
            "board_direction_agreement_status": _empty_to_str(
                s.get("board_direction_agreement_status")),
            "board_2_effect": s.get("board_2_effect"),
            "board_3_effect": s.get("board_3_effect"),
            "bootstrap_valid": b.get("bootstrap_valid"),
            "bootstrap_ci_2_5": b.get("bootstrap_ci_2_5"),
            "bootstrap_ci_97_5": b.get("bootstrap_ci_97_5"),
            "bootstrap_ci_crosses_zero": b.get("bootstrap_ci_crosses_zero"),
            "bootstrap_low_valid_replicates": b.get("bootstrap_low_valid_replicates"),
            "distribution_shift": sh.get("distribution_shift"),
            "flags": _empty_to_str(s.get("flags")),
        })
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


# ---------------------------------------------------------------------------
# 候选边界表 (任务十八; 59 行)
# ---------------------------------------------------------------------------
def build_candidate_boundary(payload: dict[str, Any]) -> pd.DataFrame:
    """59 行候选边界表; 状态只允许 SELECTED_FIXED / NOT_SELECTED_NOT_REJECTED。"""
    summary = payload["stage22_frames"][
        "v004c_stage2_2_univariate_summary_dev_v001.csv"]
    selected: dict[str, list[str]] = {}
    for m in CANDIDATE_MODELS:
        for f in m["features"]:
            selected.setdefault(f["feature_name"], []).append(m["model_id"])

    rows = []
    for _, r in summary.iterrows():
        name = str(r["feature_name"])
        chosen = selected.get(name, [])
        rows.append({
            "feature_name": name,
            "allowlist_tier": _empty_to_str(r.get("allowlist_tier")),
            "mechanism_group": _empty_to_str(r.get("mechanism_group")),
            "selected_any_model": bool(chosen),
            "selected_models": "|".join(chosen),
            "boundary_status": "SELECTED_FIXED" if chosen
            else "NOT_SELECTED_NOT_REJECTED",
            "boundary_note": "候选成员在六月开发审查后人工固定" if chosen
            else "不属于本次预声明候选设计; 不代表永久淘汰, 不得按 AUC/flags 自动填写淘汰理由",
        })
    return pd.DataFrame(rows, columns=BOUNDARY_COLUMNS)


# ---------------------------------------------------------------------------
# 无标签冗余审计 (任务十二; 只读特征值, 不读标签)
# ---------------------------------------------------------------------------
def _derive_needed_series(snapshot: pd.DataFrame) -> dict[str, pd.Series]:
    """按阶段2.1 冻结公式逐行重算 4 个入选派生因子 (不用标签)。

    二元派生项 ma5_overheat_10 必须传播源缺失 (阶段2.1 冻结
    "derived is null iff source null"): 源缺失 → 派生缺失,
    不得把缺失解释为 0/False。
    """
    out: dict[str, pd.Series] = {}
    out["overrepair"] = np.maximum(
        pd.to_numeric(snapshot["d1_open_to_close_return_raw"], errors="coerce")
        - 0.03, 0.0)
    overheat_source = pd.to_numeric(
        snapshot["d1_close_to_ma5_raw"], errors="coerce")
    out["ma5_overheat_10"] = pd.Series(
        np.where(overheat_source.isna(), np.nan,
                 (overheat_source >= 0.10).astype(float)),
        index=snapshot.index, dtype="float64")
    out["break_volume_abnormality"] = np.abs(np.log(np.maximum(
        pd.to_numeric(snapshot["break_volume_ratio_vs_board_days"],
                      errors="coerce"), 1e-6)))
    out["profit_chip_ratio"] = 1.0 - pd.to_numeric(
        snapshot["volume_above_d1_close_ratio"], errors="coerce")
    return out


def _spearman_pair(a: pd.Series, b: pd.Series) -> tuple[int, float | None]:
    mask = a.notna() & b.notna()
    count = int(mask.sum())
    if count < 2:
        return count, None
    with warnings.catch_warnings():
        # 常数输入 → scipy 报 ConstantInputWarning 且返回 nan; 在此预期内
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            rho = spearmanr(a[mask].astype(float), b[mask].astype(float))[0]
    return count, None if pd.isna(rho) else float(rho)


def build_candidate_redundancy_audit(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """30 行无标签冗余审计 (M1/M2/M3 各 10 对)。"""
    dev = snapshot[(snapshot[SIGNAL_DATE_COL] >= DEVELOPMENT_START)
                   & (snapshot[SIGNAL_DATE_COL] <= DEVELOPMENT_END)]
    locked = snapshot[(snapshot[SIGNAL_DATE_COL] >= LOCKED_START)
                      & (snapshot[SIGNAL_DATE_COL] <= LOCKED_END)]
    derived = _derive_needed_series(snapshot)

    def feature_series(name: str) -> pd.Series:
        if name in SELECTED_DERIVED_SOURCE_COLUMNS:
            return derived[name]
        return pd.to_numeric(snapshot[name], errors="coerce")

    def flag_for(max_abs: float | None) -> str:
        if max_abs is None:
            return ""
        if max_abs >= 0.999999:
            return "ABS_RHO_GE_0_999999"
        if max_abs >= 0.85:
            return "ABS_RHO_GE_0_85"
        if max_abs >= 0.70:
            return "ABS_RHO_0_70_TO_0_85"
        return "ABS_RHO_LT_0_70"

    rows = []
    for m in CANDIDATE_MODELS:
        feats = m["features"]
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                a_name = feats[i]["feature_name"]
                b_name = feats[j]["feature_name"]
                dev_count, dev_rho = _spearman_pair(
                    feature_series(a_name).reindex(dev.index),
                    feature_series(b_name).reindex(dev.index))
                locked_count, locked_rho = _spearman_pair(
                    feature_series(a_name).reindex(locked.index),
                    feature_series(b_name).reindex(locked.index))
                present = [v for v in (dev_rho, locked_rho) if v is not None]
                max_abs = max(abs(v) for v in present) if present else None
                shift = None
                if dev_rho is not None and locked_rho is not None:
                    shift = abs(dev_rho - locked_rho)
                rows.append({
                    "model_id": m["model_id"],
                    "position_a": int(feats[i]["position"]),
                    "position_b": int(feats[j]["position"]),
                    "feature_a": a_name,
                    "feature_b": b_name,
                    "dev_non_null_pair_count": dev_count,
                    "dev_spearman_rho": (None if dev_rho is None
                                         else round(dev_rho, 12)),
                    "locked_non_null_pair_count": locked_count,
                    "locked_spearman_rho": (None if locked_rho is None
                                            else round(locked_rho, 12)),
                    "absolute_dev_rho": (None if dev_rho is None
                                         else round(abs(dev_rho), 12)),
                    "absolute_locked_rho": (None if locked_rho is None
                                            else round(abs(locked_rho), 12)),
                    "maximum_absolute_rho": (None if max_abs is None
                                             else round(max_abs, 12)),
                    "absolute_rho_shift": (None if shift is None
                                           else round(shift, 12)),
                    "redundancy_flag": flag_for(max_abs),
                })
    return pd.DataFrame(rows, columns=REDUNDANCY_COLUMNS)


# ---------------------------------------------------------------------------
# 固定确定性最终候选比较算法 (阶段2.3.1 修正 6.2-6.4; 纯函数冻结定义)
# ---------------------------------------------------------------------------
FINAL_CANDIDATE_PRIORITY = {"M1": 1, "M2": 2, "M3": 3}
OOS_LOG_LOSS_TOLERANCE = 0.005
OOS_BRIER_TOLERANCE = 0.002


def select_final_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """阶段2.6 固定候选比较 (mode = global_sequential_tolerance_filter)。

    输入每条: {"model_id", "eligible", "oos_log_loss", "oos_brier",
               "july_log_loss", "feature_count"}。
    算法:
      资格集合 → OOS LL 相对全局最小值的绝对差 <= 0.005 →
      OOS Brier 相对本阶段最小值的绝对差 <= 0.002 →
      July log loss 最小 (无容差, 数值完全相同才进入下一步) →
      feature_count 最小 → 固定优先序 M1 > M2 > M3。
    全程基于全体合格候选的顺序集合过滤, 不进行任何两模型逐对比较;
    结果与输入排列顺序无关。
    """
    eligible = [c for c in candidates if bool(c.get("eligible"))]
    if not eligible:
        return {"final_status": "REJECT_NO_STABLE_MODEL",
                "selected_model_id": None,
                "stage": "eligible_empty",
                "eligible_count": 0}

    def ll(c: dict[str, Any]) -> float:
        return float(c["oos_log_loss"])

    def brier(c: dict[str, Any]) -> float:
        return float(c["oos_brier"])

    def july(c: dict[str, Any]) -> float:
        return float(c["july_log_loss"])

    best_ll = min(ll(c) for c in eligible)
    stage_a = [c for c in eligible
               if ll(c) - best_ll <= OOS_LOG_LOSS_TOLERANCE]
    best_brier = min(brier(c) for c in stage_a)
    stage_b = [c for c in stage_a
               if brier(c) - best_brier <= OOS_BRIER_TOLERANCE]
    best_july = min(july(c) for c in stage_b)
    stage_c = [c for c in stage_b if july(c) == best_july]
    min_fc = min(int(c["feature_count"]) for c in stage_c)
    stage_d = [c for c in stage_c if int(c["feature_count"]) == min_fc]
    winner = min(stage_d,
                 key=lambda c: FINAL_CANDIDATE_PRIORITY.get(c["model_id"], 99))
    return {
        "final_status": "SELECTED",
        "selected_model_id": winner["model_id"],
        "stage": {
            "stage_a": [c["model_id"] for c in stage_a],
            "stage_b": [c["model_id"] for c in stage_b],
            "stage_c": [c["model_id"] for c in stage_c],
            "stage_d": [c["model_id"] for c in stage_d],
        },
        "eligible_count": len(eligible),
        "best_oos_log_loss": best_ll,
        "best_oos_brier": best_brier,
        "best_july_log_loss": best_july,
    }


# ---------------------------------------------------------------------------
# VWAP 符号重参数化核查 (阶段2.3.1 修正 7.x; 由数据决定, 不硬编码)
# ---------------------------------------------------------------------------
VWAP_EQUIVALENCE_TOL = 1e-12


def compute_vwap_reparameterization(snapshot: pd.DataFrame) -> dict[str, Any]:
    """核查 d1_close_to_vwap_raw 与 d1_vwap_to_close_gap 的符号重参数化关系。

    在六月 (development) 与七月 (locked, 无标签) 切片分别计算:
      pair count / max_abs_sum / Pearson rho / Spearman rho /
      exact_negative_equivalence (atol=rtol=1e-12)。
    若两个切片均满足 max_abs_sum <= 1e-12 且 Pearson ≈ -1 且 Spearman ≈ -1,
    则 relationship_type = EXACT_NEGATIVE_REPARAMETERIZATION; 否则如实记录。
    """
    a = pd.to_numeric(snapshot["d1_close_to_vwap_raw"], errors="coerce")
    b = pd.to_numeric(snapshot["d1_vwap_to_close_gap"], errors="coerce")
    dates = snapshot[SIGNAL_DATE_COL]
    dev_mask = (dates >= DEVELOPMENT_START) & (dates <= DEVELOPMENT_END)
    locked_mask = (dates >= LOCKED_START) & (dates <= LOCKED_END)

    def slice_stats(mask: pd.Series) -> dict[str, Any]:
        av = a[mask]
        bv = b[mask]
        pair = av.notna() & bv.notna()
        count = int(pair.sum())
        if count < 2:
            return {
                "pair_non_null": count,
                "max_abs_sum": None,
                "pearson_rho": None,
                "spearman_rho": None,
                "exact_negative_equivalence": False,
            }
        x = av[pair].astype(float)
        y = bv[pair].astype(float)
        max_abs_sum = float(np.max(np.abs(x.to_numpy() + y.to_numpy())))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                pearson = float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])
                rho = spearmanr(x, y)[0]
        exact = (
            max_abs_sum <= VWAP_EQUIVALENCE_TOL
            and abs(pearson + 1.0) <= VWAP_EQUIVALENCE_TOL
            and abs(float(rho) + 1.0) <= VWAP_EQUIVALENCE_TOL
        )
        return {
            "pair_non_null": count,
            "max_abs_sum": max_abs_sum,
            "pearson_rho": pearson,
            "spearman_rho": None if pd.isna(rho) else float(rho),
            "exact_negative_equivalence": bool(exact),
        }

    dev = slice_stats(dev_mask)
    locked = slice_stats(locked_mask)
    exact = bool(dev["exact_negative_equivalence"]
                 and locked["exact_negative_equivalence"])
    return {
        "features": ["d1_close_to_vwap_raw", "d1_vwap_to_close_gap"],
        "tolerance": {"absolute": VWAP_EQUIVALENCE_TOL,
                      "relative": VWAP_EQUIVALENCE_TOL},
        "dev": dev,
        "locked": locked,
        "relationship_type": ("EXACT_NEGATIVE_REPARAMETERIZATION" if exact
                              else "NOT_EXACT_NEGATIVE"),
    }


def build_design_disclosure(vwap_check: dict[str, Any]) -> dict[str, Any]:
    """研究设计披露 JSON (阶段2.3.1 修正 5.3): 六月证据用于人工设计。"""
    return {
        "stage2_3_version": STAGE2_3_VERSION,
        "candidate_selection_mode": CANDIDATE_SELECTION_MODE,
        "june_role": "development_and_manual_candidate_design",
        "july_role": "locked_holdout_evaluation",
        "june_development_evidence_used_for_manual_design":
            JUNE_DEVELOPMENT_EVIDENCE_USED_FOR_MANUAL_DESIGN,
        "candidate_specs_frozen_before_july_label_access":
            CANDIDATE_SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS,
        "runtime_candidate_membership_uses_target_columns":
            RUNTIME_CANDIDATE_MEMBERSHIP_USES_TARGET_COLUMNS,
        "runtime_membership_invariant_to_target_column_mutation":
            RUNTIME_MEMBERSHIP_INVARIANT_TO_TARGET_COLUMN_MUTATION,
        "automatic_target_driven_selection": AUTOMATIC_TARGET_DRIVEN_SELECTION,
        "automatic_feature_selection": False,
        "model_training_performed": False,
        "prediction_generated": False,
        "vwap_reparameterization": vwap_check,
        "evaluation_metric_definitions": {
            "roc_auc_implementation": "sklearn.metrics.roc_auc_score",
            "pr_auc_implementation": "sklearn.metrics.average_precision_score",
            "brier_definition": "mean((y - p)^2)",
            "log_loss_probability_clip_epsilon": 1e-15,
            "calibration_model": "logit(P(Y=1)) = alpha + beta * z "
                                 "(GLM Binomial, maxiter=100, tol=1e-10)",
            "m0_calibration_slope": "NOT_APPLICABLE",
            "m0_calibration_intercept": "calibration-in-the-large "
                                        "(logit(observed_rate) - "
                                        "logit(constant_prediction))",
            "metric_status_values": [
                "OK", "METRIC_UNDEFINED", "NOT_APPLICABLE",
                "INPUT_VALIDATION_FAILURE", "NUMERICAL_FAILURE",
            ],
            "calibration_descriptive_only": True,
            "full_definitions": "v004c_stage2_3_evaluation_protocol_v001.json",
        },
        "notes": [
            "M1/M2/M3 由研究人员在阅读六月阶段2.2 单因子结构与稳定性证据后人工确定",
            "六月是开发与候选设计数据; 候选规格在任何七月标签揭示前冻结",
            "阶段2.3 运行时未读取 Target 列, 未执行自动特征搜索",
            "该披露是研究设计定位的事实记录, 不是自动选择声明",
        ],
    }


# ---------------------------------------------------------------------------
# 候选规格 JSON (任务二十一; 稳定排序, 确定性)
# ---------------------------------------------------------------------------
def build_candidate_specs_json(feature_spec: pd.DataFrame) -> dict[str, Any]:
    """候选规格完整记录 (成员/顺序/预期方向/机制/准入/预处理/互斥组/理由)。"""
    models = []
    spec_by_model: dict[str, list[dict[str, Any]]] = {}
    for _, r in feature_spec.iterrows():
        spec_by_model.setdefault(r["model_id"], []).append({
            "position": int(r["position"]),
            "feature_name": r["feature_name"],
            "expected_direction": r["expected_direction"],
            "direction_is_constraint": bool(r["direction_is_constraint"]),
            "mechanism_group": r["mechanism_group"],
            "underlying_mechanism": r["underlying_mechanism"],
            "allowlist_tier": r["allowlist_tier"],
            "preprocess_policy": r["preprocess_policy"],
            "mutual_exclusion_group_id": r["mutual_exclusion_group_id"],
            "selection_rationale": r["selection_rationale"],
        })
    for m in CANDIDATE_MODELS:
        models.append({
            "model_id": m["model_id"],
            "model_name": m["model_name"],
            "role": m["role"],
            "feature_count": int(m["feature_count"]),
            "model_family": m["model_family"],
            "penalty": m["penalty"],
            "design_purpose": m["design_purpose"],
            "features": spec_by_model.get(m["model_id"], []),
        })
    return {
        "stage2_3_version": STAGE2_3_VERSION,
        "candidate_selection_mode": CANDIDATE_SELECTION_MODE,
        "direction_is_constraint": False,
        "model_count": len(models),
        "substantive_model_count": int(sum(
            1 for m in models if m["role"] != "BASELINE")),
        "selected_unique_feature_count": len(UNIQUE_SELECTED_FEATURES),
        "unique_selected_features": list(UNIQUE_SELECTED_FEATURES),
        "models": models,
        "notes": [
            "M1/M2/M3 由研究人员在阅读六月阶段2.2 单因子结构与稳定性证据后人工确定; "
            "候选规格在七月标签揭示前冻结 (candidate_specs_frozen_before_july_label_access=true)",
            "阶段2.3 运行时未读取 Target 列, 未执行自动特征搜索 "
            "(runtime_candidate_membership_uses_target_columns=false)",
            "M1 与 M2 的互斥表达不得放进同一模型 (各自互斥组每模型最多一次)",
            "M3 不包含派生因子的直接源变量",
            "expected_direction 仅用于后续诊断, 不是系数约束 (direction_is_constraint=false)",
            "本 JSON 不包含任何系数/截距/预测概率/模型表现/七月结果",
        ],
    }


# ---------------------------------------------------------------------------
# 评价协议 JSON (任务二十二)
# ---------------------------------------------------------------------------
def build_evaluation_protocol_json() -> dict[str, Any]:
    """七月一次性评价协议 + Walk-forward 协议 + 资格门 + 比较层级。"""
    return {
        "stage2_3_version": STAGE2_3_VERSION,
        "protocol_kind": "frozen_specification_only",
        "model_family": MODEL_FAMILY,
        "penalty": PENALTY,
        "C": C_VALUE,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "class_weight": CLASS_WEIGHT,
        "random_seed": RANDOM_SEED,
        "fit_intercept": FIT_INTERCEPT,
        "intercept_baseline": {
            "model_id": "M0",
            "model_family": "intercept_prevalence_baseline",
            "penalty": "none",
            "prediction_rule": "训练窗口 Target7 正样本率",
        },
        "hyperparameter_search_forbidden": [
            "C", "penalty", "solver", "class_weight",
        ],
        "preprocessing": {
            "fold_clip_z": {
                "lower_quantile": FOLD_CLIP_LOWER_QUANTILE,
                "upper_quantile": FOLD_CLIP_UPPER_QUANTILE,
                "ddof": FOLD_CLIP_DDOF,
                "train_std_floor_fail_closed": FOLD_STD_FLOOR,
                "rule": "每个训练 fold 独立: 只用训练窗口非缺失值计算 p01/p99; "
                        "训练值与测试值裁剪到训练 p01/p99; 用裁剪后训练均值/标准差 "
                        "做 z-score; 测试窗口不参与任何参数计算; "
                        "训练标准差 <= 1e-12 时该 fold 失败, 不得填 0 或跳过特征",
                "train_only_parameters": True,
            },
            "raw_binary": {
                "allowed_values": list(BINARY_ALLOWED_VALUES),
                "rule": "保持 0/1, 不裁剪, 不标准化; 严格验证值域 {0, 1}",
                "features": ["ma5_overheat_10"],
            },
            "derived_compute": {
                "rule": "派生公式逐行计算, 不使用标签, 也不使用跨行统计; "
                        "派生完成后再执行其冻结预处理政策",
                "features": [
                    "overrepair", "ma5_overheat_10",
                    "break_volume_abnormality", "profit_chip_ratio",
                ],
            },
            "missing_policy": MISSING_POLICY,
            "missing_policy_rule": "任何训练或测试窗口出现缺失时该模型 fold 失败并记录; "
                                   "禁止均值/中位数/众数填补、缺失 indicator、删除含缺失样本",
            "forbidden_preprocessing": [
                "全样本先裁剪", "全样本先标准化", "七月参与预处理",
                "监督式分箱", "重新创建 bucket", "PCA", "特征选择",
            ],
        },
        "july_evaluation": {
            "train_range": [DEVELOPMENT_START, DEVELOPMENT_END],
            "test_range": [LOCKED_START, LOCKED_END],
            "reveal_after": "模型规格提交并打标签后才允许揭示七月结果",
            "probability_metrics": [
                "ROC-AUC", "PR-AUC", "Brier score", "log loss",
                "calibration intercept", "calibration slope",
            ],
            "calibration_rule": "校准截距与斜率只做评估, 不得重新校准预测",
            "date_ranking_metrics": [
                "date Top1 Target7 hit rate",
                "date Top3 Target7 hit rate",
                "mean Target7 count in Top3",
                "eligible signal_date count",
            ],
            "trading_threshold_rule": "不创建交易阈值; 不选择概率阈值",
            "tie_breaking": ["predicted_probability descending",
                             "stock_code ascending"],
            "candidate_less_than_3_rule": "若某日候选数少于 3, 使用该日全部候选",
            "forbidden": [
                "七月调参", "七月选变量", "七月选择概率阈值",
                "七月删除失败模型", "七月新增模型",
            ],
        },
        "walk_forward": {
            "split_rule": "按 signal_date 升序; 训练 = 当前测试日期之前的全部日期; "
                          "测试 = 下一完整 signal_date; 同一日期全部样本位于同一 fold",
            "minimum_training_gates": {
                "train_signal_dates": 10,
                "train_rows": 80,
                "train_positive": 20,
                "train_negative": 40,
            },
            "first_formal_test_date_rule": "第一个同时满足全部训练门的日期作为首个正式测试日期",
            "fold_preprocessing_rule": "每个 fold 独立: 物化派生因子、检查缺失、"
                                       "计算 p01/p99、裁剪、计算 mean/std、标准化、拟合; "
                                       "测试日期不得参与任何训练参数",
            "oos_outputs_required": [
                "每行OOS预测", "fold训练日期范围", "测试日期", "训练行数",
                "训练正负样本数", "预处理参数", "模型系数", "截距", "收敛状态",
            ],
        },
        "final_eligibility_gates": [
            "1. 七月正式拟合收敛",
            "2. 所有正式 Walk-forward fold 均收敛",
            "3. 所有预测均有限且在 [0,1]",
            "4. 七月 log loss < M0 七月 log loss",
            "5. 完整 Walk-forward OOS log loss < M0 OOS log loss",
            "6. 完整 Walk-forward OOS ROC-AUC >= 0.50",
        ],
        "rejection_state": "REJECT_NO_STABLE_MODEL",
        "rejection_rule": "如果没有候选满足全部条件, 最终状态 = REJECT_NO_STABLE_MODEL; "
                          "不得降低门槛",
        "candidate_comparison": {
            "mode": "global_sequential_tolerance_filter",
            "eligibility_first": True,
            "algorithm_note": "基于全体合格候选的顺序集合过滤, 不进行任何两模型逐对比较; "
                              "全部差值均为绝对数值差, 不是相对百分比差; "
                              "结果与候选输入排列顺序无关",
            "oos_log_loss": {
                "operation": "retain_within_absolute_difference_of_global_minimum",
                "tolerance": 0.005,
            },
            "oos_brier": {
                "operation": "retain_within_absolute_difference_of_stage_minimum",
                "tolerance": 0.002,
            },
            "july_log_loss": {
                "operation": "minimum",
                "tolerance": 0.0,
                "rule": "无额外容差; 数值完全相同才进入下一步",
            },
            "feature_count": {
                "operation": "minimum",
            },
            "fixed_model_priority": ["M1", "M2", "M3"],
            "forbidden": [
                "逐对比较模型", "相对差值", "百分比差",
                "使用七月 AUC 打破并列", "使用 Top1 或 Top3 打破并列",
                "临时调整容差", "看到结果后修改层级",
            ],
        },
        "forbidden_selection_metrics": [
            "七月 ROC-AUC 最高", "Top1 最高", "Top3 最高", "任意单一最好指标",
        ],
        "top_metrics_role": "Top1/Top3 只作为描述性业务排序指标",
        "convergence_status": {
            "definitions": [
                "CONVERGED", "MAX_ITER_REACHED", "NUMERICAL_FAILURE",
                "NON_FINITE_COEFFICIENT", "NON_FINITE_PREDICTION",
                "INPUT_VALIDATION_FAILURE",
            ],
            "failed_rule": "solver 报告未收敛, 或达到 max_iter 仍未满足求解停止条件 "
                           "→ convergence_status = FAILED; 即使预测有限, "
                           "也不得将未收敛模型标为收敛",
        },
        "coefficient_stability": {
            "required_reports": [
                "每个特征在有效 fold 中的系数", "系数符号",
                "expected direction 一致比例", "系数均值", "系数标准差",
                "系数最小值", "系数最大值",
            ],
            "expected_direction_agreement_is_gate": False,
            "refit_on_direction_mismatch_forbidden": True,
        },
        "metric_definitions": {
            "roc_auc": {
                "implementation": "sklearn.metrics.roc_auc_score",
                "positive_label": 1,
                "label_requirement": "评估标签必须同时包含 0 和 1; "
                                     "否则 metric_status = METRIC_UNDEFINED, "
                                     "metric_value = null (不得填 0.5)",
            },
            "pr_auc": {
                "implementation": "sklearn.metrics.average_precision_score",
                "metric_name": "average_precision",
                "note": "不是 precision-recall 曲线梯形积分",
                "no_positive_rule": "评估标签中没有正样本 → "
                                    "metric_status = METRIC_UNDEFINED, "
                                    "metric_value = null",
            },
            "brier": {
                "formula": "mean((y - p)^2)",
                "prediction_requirement": "预测必须有限且位于 [0,1]; "
                                          "否则模型评估失败, "
                                          "不得裁剪后掩盖非法预测",
            },
            "log_loss": {
                "epsilon": 1e-15,
                "formula": "mean(-y*log(p_clipped) - (1-y)*log(1-p_clipped)); "
                           "p_clipped = clip(p, epsilon, 1 - epsilon)",
                "clip_scope": "该裁剪只用于计算 log loss, "
                              "不得修改保存的原始预测概率",
            },
            "calibration": {
                "p_cal": "clip(p, 1e-15, 1 - 1e-15)",
                "z": "log(p_cal / (1 - p_cal))",
                "model": "logit(P(Y=1)) = alpha + beta * z",
                "implementation": "statsmodels GLM Binomial",
                "maxiter": 100,
                "tol": 1e-10,
                "penalty": "none",
                "note": "这些参数仅用于评价期校准指标估计, "
                        "不是候选预测模型超参数",
                "calibration_intercept": "alpha",
                "calibration_slope": "beta",
                "undefined_rules": [
                    "评估标签只有一个类别",
                    "预测 logit 方差 <= 1e-12",
                    "校准模型不收敛",
                    "参数非有限",
                    "数值异常",
                ],
                "m0": {
                    "calibration_slope": "NOT_APPLICABLE",
                    "calibration_intercept_rule": "calibration-in-the-large: "
                        "logit(clip(observed_rate, 1e-15, 1 - 1e-15)) - "
                        "logit(clip(predicted_rate, 1e-15, 1 - 1e-15)); "
                        "observed_rate = mean(y), predicted_rate = "
                        "constant_prediction",
                    "single_class_rule": "评估标签只有一个类别 → "
                        "M0 calibration_intercept = METRIC_UNDEFINED, "
                        "M0 calibration_slope = NOT_APPLICABLE",
                },
                "descriptive_only": "calibration metrics are descriptive only; "
                                    "不属于六条最终资格门, 不参与候选比较算法",
            },
            "metric_status_values": [
                "OK", "METRIC_UNDEFINED", "NOT_APPLICABLE",
                "INPUT_VALIDATION_FAILURE", "NUMERICAL_FAILURE",
            ],
            "no_placeholder_rule": "不得使用 0、0.5 或空字符串伪装不可计算指标",
        },
        "flags": {
            "model_training_performed": False,
            "prediction_generated": False,
            "holdout_label_access": False,
            "automatic_feature_selection": False,
            "hyperparameter_search": False,
            "threshold_search": False,
            "interaction_search": False,
        },
    }


# ---------------------------------------------------------------------------
# 锁定审计 (任务二十三)
# ---------------------------------------------------------------------------
def scan_source_for_forbidden_calls(source: str) -> dict[str, list[str]]:
    """AST 扫描源码: 返回 {"imports": [...], "calls": [...]} 禁止节点。

    只遍历 ast.Import / ast.ImportFrom / ast.Call 节点;
    字符串常量 (协议文字、deny-list 定义) 不参与判定。
    """
    import ast

    tree = ast.parse(source)
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    imports.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in _FORBIDDEN_CALL_ATTRS:
                    calls.append(f".{node.func.attr}(")
            elif isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALL_NAMES:
                    calls.append(f"{node.func.id}(")
    return {"imports": sorted(set(imports)), "calls": sorted(set(calls))}


def _module_ast_scan_checks() -> list[dict[str, str]]:
    """基于 AST 的静态无训练审计 (任务 11; 字符串常量不误报)。"""
    source = inspect.getsource(sys.modules[__name__])
    found = scan_source_for_forbidden_calls(source)
    imports = found["imports"]
    calls = found["calls"]

    def group(label: str, tokens: tuple[str, ...]) -> dict[str, str]:
        hits = [t for t in tokens if t in calls or t in imports]
        return {
            "check_name": label,
            "status": "PASS" if not hits else "FAIL",
            "details": (f"AST 发现禁止调用: {hits}" if hits
                        else "AST 无该类别禁止调用"),
        }

    return [
        group("no_model_fit_call",
              (".fit(", ".fit_predict(")),
        group("no_prediction_generated",
              (".predict(", ".predict_proba(", ".decision_function(")),
        group("no_automatic_feature_selection",
              ("cross_val_score(", "cross_validate(",
               "GridSearchCV(", "RFE(", "Lasso(")),
        group("no_hyperparameter_search",
              ("GridSearchCV(", "RandomizedSearchCV(")),
        group("no_threshold_search",
              ("cross_val_score(", "cross_validate(")),
        group("no_ml_library_import",
              ("sklearn", "statsmodels", "xgboost", "lightgbm",
               "catboost", "torch", "tensorflow")),
        group("no_logistic_instantiation", ("LogisticRegression(",)),
    ]


def _builder_invariance_checks() -> list[dict[str, str]]:
    """证据扰动不变 (任务八 8.7): 5 个构建函数源码不含证据字段 token。

    评价协议 JSON 的 activity flags 键是任务规定的协议记录, 不属于
    阶段2.2 单因子证据字段, 因此该函数使用排除 flags 的 token 集。
    """
    out = []
    builders = (
        (build_model_catalog, "model_catalog", _EVIDENCE_TOKENS),
        (build_model_feature_spec, "model_feature_spec", _EVIDENCE_TOKENS),
        (build_preprocessing_spec, "preprocessing_spec", _EVIDENCE_TOKENS),
        (build_candidate_specs_json, "candidate_specs_json", _EVIDENCE_TOKENS),
        (build_evaluation_protocol_json, "evaluation_protocol_json",
         ("effect_value", "signed_auc", "bootstrap", "lodo",
          "board_direction")),
    )
    for fn, label, tokens in builders:
        src = inspect.getsource(fn)
        hits = [t for t in tokens if t in src]
        out.append({
            "check_name": f"membership_independent_of_evidence_{label}",
            "status": "PASS" if not hits else "FAIL",
            "details": (f"构建函数含证据 token: {hits}" if hits
                        else "构建函数不引用证据字段"),
        })
    return out


def _expected_member_spec(
    allowlist: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """从冻结常量 CANDIDATE_MODELS 生成确定性规范对象 (与 feature_spec 同源可比)。"""
    out: dict[str, Any] = {}
    for m in CANDIDATE_MODELS:
        feats = []
        for f in m["features"]:
            info = allowlist[f["feature_name"]]
            feats.append({
                "feature_name": f["feature_name"],
                "expected_direction": f["expected_direction"],
                "mechanism_group": info["mechanism_group"],
                "allowlist_tier": info["allowlist_tier"],
                "preprocess_policy": info["preprocess_policy"],
            })
        out[m["model_id"]] = {
            "model_name": m["model_name"],
            "role": m["role"],
            "feature_count": int(m["feature_count"]),
            "features": feats,
        }
    return out


def _actual_member_spec(catalog: pd.DataFrame,
                        feature_spec: pd.DataFrame) -> dict[str, Any]:
    """从实际生成的规格生成同构规范对象 (逐项与期望侧比较)。"""
    out: dict[str, Any] = {}
    for _, r in catalog.iterrows():
        mid = str(r["model_id"])
        sub = feature_spec[feature_spec["model_id"] == mid].sort_values("position")
        out[mid] = {
            "model_name": str(r["model_name"]),
            "role": str(r["role"]),
            "feature_count": int(r["feature_count"]),
            "features": [
                {"feature_name": str(row["feature_name"]),
                 "expected_direction": str(row["expected_direction"]),
                 "mechanism_group": str(row["mechanism_group"]),
                 "allowlist_tier": str(row["allowlist_tier"]),
                 "preprocess_policy": str(row["preprocess_policy"])}
                for _, row in sub.iterrows()],
        }
    return out


def build_holdout_lock_audit(
    *,
    catalog: pd.DataFrame,
    feature_spec: pd.DataFrame,
    boundary: pd.DataFrame,
    evidence: pd.DataFrame,
    input_checks: dict[str, Any],
    allowlist: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """任务二十三锁定审计 (基础检查 + 源码扫描 + 证据无关检查)。"""
    rows: list[dict[str, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"check_name": name,
                     "status": "PASS" if ok else "FAIL",
                     "details": detail})

    # 实际输入文件集合 (相对 git root 的 POSIX 路径) — 运行时输入审计
    loaded = set(input_checks.get("loaded_input_files", []))
    expected_inputs = set(input_checks.get("expected_input_files", []))
    forbidden_inputs = set(input_checks.get("forbidden_input_files", []))
    add("stage2_3_not_read_training_d1_table",
        not any(str(p).endswith("v004c_training_d1_v001.csv") for p in loaded),
        "实际 loaded_input_files 不含 training D1 表")
    add("stage2_3_not_read_existing_model_audit",
        not any(str(p).endswith("v004c_d1_existing_model_audit_v001.csv")
                for p in loaded),
        "实际 loaded_input_files 不含 existing model audit")
    add("actual_loaded_input_files_match_allowed",
        loaded == expected_inputs,
        f"实际读取 {len(loaded)} 个文件与允许清单 {len(expected_inputs)} 个 "
        f"精确相等" if loaded == expected_inputs
        else f"集合差异: 多读={sorted(loaded - expected_inputs)} "
             f"少读={sorted(expected_inputs - loaded)}")
    add("actual_loaded_input_files_disjoint_forbidden",
        not (loaded & forbidden_inputs),
        "实际读取文件与禁止输入清单交集为空" if not (loaded & forbidden_inputs)
        else f"禁止文件被读取: {sorted(loaded & forbidden_inputs)}")
    read_names = set(input_checks.get("snapshot_read_columns", []))
    labels_not_read = bool(input_checks.get("snapshot_label_columns_not_read"))
    add("snapshot_label_columns_not_accessed", labels_not_read,
        "snapshot 仅读取允许特征列, 标签/审计列不可见")
    add("july_label_columns_not_accessed", labels_not_read,
        "七月标签列未访问 (读取列与标签/敏感列交集为空)")
    # 阶段2.3.1 修正 4.4: 研究设计披露类审计项
    # 两项均要求"未访问"为真 (labels_not_read=True → 访问=False → PASS)
    add("RUNTIME_TARGET_COLUMN_ACCESS", labels_not_read,
        "运行时 Target 列访问 = False (snapshot 仅读取允许特征列)")
    add("JULY_LABEL_ACCESS", labels_not_read,
        "七月标签访问 = False (读取列与标签/敏感列交集为空)")
    add("AUTOMATIC_TARGET_DRIVEN_SELECTION",
        not AUTOMATIC_TARGET_DRIVEN_SELECTION,
        f"自动 Target 驱动选择 = {AUTOMATIC_TARGET_DRIVEN_SELECTION} "
        f"(成员由人工设计固定)")
    add("JUNE_EVIDENCE_USED_FOR_MANUAL_DESIGN_DECLARED",
        JUNE_DEVELOPMENT_EVIDENCE_USED_FOR_MANUAL_DESIGN,
        "六月开发证据用于人工设计已声明 "
        f"(june_development_evidence_used_for_manual_design="
        f"{JUNE_DEVELOPMENT_EVIDENCE_USED_FOR_MANUAL_DESIGN})")
    add("SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS",
        CANDIDATE_SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS,
        "候选规格在七月标签揭示前冻结已声明 "
        f"(candidate_specs_frozen_before_july_label_access="
        f"{CANDIDATE_SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS})")

    catalog_rows = int(len(catalog))
    substantive = int((catalog["role"] != "BASELINE").sum())
    m_counts = {mid: int((feature_spec["model_id"] == mid).sum())
                for mid in ("M1", "M2", "M3")}
    # 固定成员结构级比较 (任务 5.5): 实际生成规格与冻结常量逐项相等
    expected_spec = _expected_member_spec(allowlist)
    actual_spec = _actual_member_spec(catalog, feature_spec)
    spec_equal = actual_spec == expected_spec
    add("candidate_membership_fixed_after_june_development_review",
        catalog_rows == EXPECTED_GATES["model_count"] and spec_equal,
        "实际模型规格与冻结常量逐项相等 (model_id/name/role/count/"
        "特征顺序/方向/机制/tier/预处理)" if spec_equal
        else "实际模型规格与冻结常量不一致, fail closed")
    for check in _builder_invariance_checks():
        add(check["check_name"], check["status"] == "PASS", check["details"])
    for check in _module_ast_scan_checks():
        add(check["check_name"], check["status"] == "PASS", check["details"])
    wf_funcs = sorted(
        n for n, v in vars(sys.modules[__name__]).items()
        if inspect.isfunction(v) and "walk_forward" in n.lower())
    add("no_walk_forward_run", not wf_funcs,
        "模块无任何 Walk-forward 执行函数" if not wf_funcs
        else f"存在 Walk-forward 函数: {wf_funcs}")
    add("model_count_is_4", catalog_rows == 4,
        f"模型总数 {catalog_rows}")
    add("substantive_model_count_is_3", substantive == 3,
        f"实质模型数 {substantive}")
    add("m1_m2_m3_feature_count_is_5",
        all(m_counts.get(mid) == 5 for mid in ("M1", "M2", "M3")),
        f"M1={m_counts.get('M1')} M2={m_counts.get('M2')} M3={m_counts.get('M3')}")
    add("candidate_boundary_rows_is_59", int(len(boundary)) == 59,
        f"候选边界行数 {len(boundary)}")
    add("selected_unique_feature_count_is_12",
        int(len(evidence)) == 12 and int(evidence["feature_name"].nunique()) == 12,
        f"唯一入选特征数 {len(evidence)}")
    return pd.DataFrame(rows, columns=["check_name", "status", "details"])


# 内容扫描只针对七月标签"指标" token (如 july_auc/july_log_loss 等);
# 字段名 (candidate_specs_frozen_before_july_label_access) 与角色声明
# (july_role) 是任务规定的规范字段, 不是指标, 不匹配扫描。
_JULY_METRIC_TOKENS = (
    "july_auc", "july_roc", "july_brier", "july_log_loss", "july_accuracy",
    "july_hit", "july_top1", "july_top3", "july_positive_rate",
    "july_target_rate", "july_prediction", "july_performance",
    "july_calibration", "locked_target",
)


def _no_july_metric_text_scan(text: str, what: str) -> tuple[bool, str]:
    lower = text.lower()
    bad = [t for t in _JULY_METRIC_TOKENS if t in lower]
    return (not bad), f"{what}: " + ("干净" if not bad else f"发现 {bad}")


# ---------------------------------------------------------------------------
# 输出包 (任务二十/二十七)
# ---------------------------------------------------------------------------
def _render_review(
    git_chain: dict[str, Any],
    input_checks: dict[str, Any],
    catalog: pd.DataFrame,
    feature_spec: pd.DataFrame,
    preprocessing: pd.DataFrame,
    evidence: pd.DataFrame,
    redundancy: pd.DataFrame,
    boundary: pd.DataFrame,
    audit: pd.DataFrame,
    candidate_specs: dict[str, Any],
    protocol: dict[str, Any],
    vwap_check: dict[str, Any],
    git_head: str,
) -> str:
    """阶段2.3 review (任务二十六): 人工固定说明 + 全部冻结内容。"""
    lines: list[str] = []
    add = lines.append
    add("# v004c 阶段2.3 — 人工候选模型规格冻结报告 (自动生成)")
    add("")
    add("- 规格版本: " + STAGE2_3_VERSION)
    add("- 来源: 阶段1 v004c-d1-dataset-0.1 → "
        "a65661f4738b849a06efb4859d4100842871e971 / "
        "阶段2.1 v004c-factor-dictionary-0.1 → "
        "11f455dc9a6643a67250321cce68868be904cf06 / "
        "阶段2.2 v004c-stage2-2-univariate-0.1 → "
        "fba302ffc58deb7b15812e4dd2ada113ae1f3a34")
    add(f"- 分支: {git_chain['branch']} / HEAD: {git_head}")

    add("")
    add("## 版本链与输入验证")
    add("")
    add(f"- 阶段2.2 tag 目标 = {EXPECTED_STAGE2_2_DATA_COMMIT} "
        f"({git_chain['stage2_2_tag_target'] == EXPECTED_STAGE2_2_DATA_COMMIT})")
    add(f"- 阶段2.1 tag 目标 = {EXPECTED_STAGE2_1_DATA_COMMIT} (通过)")
    add(f"- 阶段1 tag 目标 = {EXPECTED_STAGE1_DATA_COMMIT} (通过)")
    add(f"- 阶段2.2数据提交是当前 HEAD 祖先: {git_chain['data_commit_ancestor']}")
    add(f"- 阶段2.1 git_head.txt = {input_checks.get('stage21_git_head_file')} (通过)")
    add(f"- 阶段2.2 git_head.txt = {input_checks.get('stage22_git_head_file')} (通过)")
    sha_summary = "; ".join(
        f"{name}={ 'OK' if v.get('verified') else 'MISMATCH' }"
        for name, v in sorted(input_checks.get("input_sha_verification", {}).items()))
    add(f"- 输入 SHA 与对应 manifest 一致: {sha_summary}")
    add(f"- 阶段1 manifest 审计参考校验: "
        f"{input_checks.get('stage1_manifest_audit_reference_verified')}")
    add(f"- 结构门: {input_checks.get('input_rows')} 行 / "
        f"{input_checks.get('signal_dates')} 信号日")
    add(f"- snapshot 仅读取允许特征列: "
        f"{', '.join(input_checks.get('snapshot_read_columns', []))}")
    add(f"- 时间切片: 六月 {input_checks.get('dev_rows')} 行 / "
        f"七月锁定 {input_checks.get('locked_rows')} 行 (仅无标签使用)")

    add("")
    add("## 阶段2.3定位: 人工设计 + 规格先冻结")
    add("")
    add("- candidate_selection_mode = human_fixed_after_june_development_review")
    add("- M1、M2和M3由研究人员在阅读六月阶段2.2单因子结构和稳定性证据后人工确定。")
    add("- 六月是开发与候选设计数据 (development_and_manual_candidate_design)。")
    add("- 候选规格在任何七月标签揭示前冻结 "
        "(candidate_specs_frozen_before_july_label_access=true)。")
    add("- 阶段2.3运行时未读取 Target 列, 未根据 AUC、bootstrap、flags 或其它统计 "
        "自动搜索、增加、删除或替换候选成员。")
    add("- 运行时成员不变性 (runtime membership invariance): 模型目录/特征规格/"
        "预处理规格/候选 JSON/评价协议 对阶段2.2 证据字段扰动与 Target 列扰动逐字节不变")

    add("")
    add("## 固定候选模型 (M0 + 3 个实质候选, 共 4 个)")
    add("")
    for _, r in catalog.iterrows():
        add(f"- **{r['model_id']} {r['model_name']}** (role={r['role']}, "
            f"feature_count={r['feature_count']}, "
            f"family={r['model_family']}, penalty={r['penalty']})")
    add("")
    for mid in ("M1", "M2", "M3"):
        add(f"### {mid} 固定特征顺序与预期方向 (direction_is_constraint=false)")
        add("")
        add("| 位置 | 特征 | 预期方向 | 机制组 | 预处理 | 互斥组 |")
        add("|---|---|---|---|---|---|")
        sub = feature_spec[feature_spec["model_id"] == mid]
        for _, r in sub.iterrows():
            add(f"| {r['position']} | {r['feature_name']} | "
                f"{r['expected_direction']} | {r['mechanism_group']} | "
                f"{r['preprocess_policy']} | {r['mutual_exclusion_group_id']} |")
        add("")
        design = next(m["design_purpose"] for m in CANDIDATE_MODELS
                      if m["model_id"] == mid)
        add(f"设计目的: {design}")
        add("")
    add("### M1 与 M2 的表达替换关系")
    add("")
    add("| 机制组 | M1 | M2 |")
    add("|---|---|---|")
    add("| D1_PRICE_ACTION | break_open_return | break_open_return |")
    add("| D1_VOLUME_ACTIVITY | down_bar_volume_ratio | down_bar_volume_ratio |")
    add("| D1_CHIP_DISTRIBUTION | high_zone_volume_ratio | high_zone_amount_ratio |")
    add("| D1_LATE_DAY_PRESSURE | late_day_sell_volume_ratio | late_day_sell_amount_ratio |")
    add("| D1_VWAP_POSITION | d1_close_to_vwap_raw | d1_vwap_to_close_gap |")
    add("")
    add("M1 与 M2 的互斥表达不得放进同一模型; M2 用于判断信号是否来自机制本身, "
        "而不是只来自 volume/amount 或正向/反向表达。")
    add("")
    add("### M3 派生与底层机制关系")
    add("")
    add("| 位置 | 派生特征 | 冻结公式 | 直接源 | 底层机制 |")
    add("|---|---|---|---|---|")
    add("| 1 | overrepair | max(d1_open_to_close_return_raw - 0.03, 0) | "
        "d1_open_to_close_return_raw | D1_PRICE_ACTION |")
    add("| 2 | ma5_overheat_10 | 1[d1_close_to_ma5_raw >= 0.10] | "
        "d1_close_to_ma5_raw | D1_MA_POSITION |")
    add("| 3 | break_volume_abnormality | abs(log(max(break_volume_ratio_vs_board_days, 1e-6))) | "
        "break_volume_ratio_vs_board_days | D1_VOLUME_ACTIVITY |")
    add("| 4 | profit_chip_ratio | 1 - volume_above_d1_close_ratio | "
        "volume_above_d1_close_ratio | D1_CHIP_DISTRIBUTION |")
    add("| 5 | late_day_sell_volume_ratio | (源字段) | - | D1_LATE_DAY_PRESSURE |")
    add("")
    add("M3 不包含任何派生因子的直接源变量; 检验阶段2.1 预声明派生表达能否形成 "
        "结构更明确的跨机制组合, 不是根据阶段2.2 表现自动挑选派生因子。")

    add("")
    add("## 12 个唯一入选特征 (首次出现顺序)")
    add("")
    add("1. " + ", ".join(UNIQUE_SELECTED_FEATURES))
    add("")
    add("## 候选边界 (59 行)")
    add("")
    sel = int((boundary["boundary_status"] == "SELECTED_FIXED").sum())
    add(f"- 59 个阶段2.2 因子逐一登记; SELECTED_FIXED = {sel}, "
        f"NOT_SELECTED_NOT_REJECTED = {int(len(boundary) - sel)}")
    add("- 未进入 M1/M2/M3 的字段只表示不属于本次预声明候选设计, "
        "不代表永久淘汰; 禁止按 AUC 或 flags 自动填写淘汰理由")

    add("")
    add("## 无标签冗余审计 (30 行)")
    add("")
    flags = redundancy["redundancy_flag"].value_counts()
    flag_summary = "; ".join(f"{k}={int(v)}" for k, v in flags.items())
    add(f"- 数据切片: 六月 {DEVELOPMENT_START}..{DEVELOPMENT_END} / "
        f"七月锁定 {LOCKED_START}..{LOCKED_END} (只读特征值, 不读标签)")
    add(f"- 每模型 10 对, 共 {len(redundancy)} 行; 指标: dev/locked Spearman rho、"
        f"pair 计数、最大绝对相关、六月七月相关变化")
    add(f"- 冗余 flag 数量: {flag_summary or '无'}")
    if len(redundancy) and redundancy["maximum_absolute_rho"].notna().any():
        top = redundancy.loc[redundancy["maximum_absolute_rho"].idxmax()]
        add(f"- 最高相关特征对: {top['model_id']} {top['feature_a']} vs "
            f"{top['feature_b']} (|rho|={top['maximum_absolute_rho']})")
    if len(redundancy) and redundancy["absolute_rho_shift"].notna().any():
        max_shift = redundancy.loc[redundancy["absolute_rho_shift"].idxmax()]
        add(f"- 六月/七月相关性变化最大对: {max_shift['feature_a']} vs "
            f"{max_shift['feature_b']} (shift={max_shift['absolute_rho_shift']})")
    add("- 冗余指标只作描述, 不得自动修改模型成员; 高相关也保留固定候选并在此披露")

    add("")
    add("## VWAP 符号重参数化披露")
    add("")
    dev = vwap_check["dev"]
    locked = vwap_check["locked"]
    add(f"- 核查字段: d1_close_to_vwap_raw (M1) 与 d1_vwap_to_close_gap (M2); "
        f"浮点容差 atol=rtol={VWAP_EQUIVALENCE_TOL}")
    add(f"- 六月核查: pair_non_null={dev.get('pair_non_null')}, "
        f"max_abs_sum={dev.get('max_abs_sum')}, "
        f"Pearson={dev.get('pearson_rho')}, Spearman={dev.get('spearman_rho')}, "
        f"exact_negative_equivalence={dev.get('exact_negative_equivalence')}")
    add(f"- 七月无标签核查: pair_non_null={locked.get('pair_non_null')}, "
        f"max_abs_sum={locked.get('max_abs_sum')}, "
        f"Pearson={locked.get('pearson_rho')}, "
        f"Spearman={locked.get('spearman_rho')}, "
        f"exact_negative_equivalence={locked.get('exact_negative_equivalence')}")
    add(f"- relationship_type = {vwap_check['relationship_type']} (由数据判定, 未硬编码)")
    if vwap_check["relationship_type"] == "EXACT_NEGATIVE_REPARAMETERIZATION":
        add("- M2中的d1_vwap_to_close_gap与M1中的d1_close_to_vwap_raw属于符号重参数化。")
        add("- 在相同fold标准化、无方向系数约束和相同L2 Logistic下, "
            "该替换本身不增加独立信息或模型表达能力, 只改变字段和系数符号表示。")
        add("- M2与M1的实质差异主要来自: high_zone volume → amount, "
            "late_day sell volume → amount。")
        add("- 不得将 d1_vwap_to_close_gap 称为新的独立VWAP信号、独立信息源或额外模型自由度。")
    else:
        add("- 该两字段在所选切片中不构成严格负向关系, 如实记录, 不作符号重参数化声明。")

    add("")
    add("## 固定预处理")
    add("")
    add("- FOLD_CLIP_Z: 每个训练 fold 独立, 只用训练窗口非缺失值计算 p01/p99 "
        f"({FOLD_CLIP_LOWER_QUANTILE}/{FOLD_CLIP_UPPER_QUANTILE}), 训练值与测试值裁剪到"
        f"训练 p01/p99, 再用裁剪后训练均值/标准差 z-score (ddof={FOLD_CLIP_DDOF}); "
        f"测试窗口不参与任何参数计算; 训练标准差 <= {FOLD_STD_FLOOR} 时 fail closed")
    add("- RAW_BINARY: 保持 0/1, 不裁剪不标准化, 严格验证值域 {0, 1} "
        "(M3 的 ma5_overheat_10)")
    add("- 派生因子逐行计算, 不用标签, 不用跨行统计; 派生完成后执行冻结预处理政策")
    add(f"- 缺失策略 = {MISSING_POLICY}: 任何窗口出现缺失该 fold 失败并记录, "
        "禁止均值/中位数/众数填补、缺失 indicator、删除含缺失样本")
    add("- 禁止: 全样本先裁剪、全样本先标准化、七月参与预处理、监督式分箱、"
        "重新创建 bucket、PCA、特征选择")

    add("")
    add("## 固定模型族与超参数")
    add("")
    add(f"- M1/M2/M3: {MODEL_FAMILY}, penalty={PENALTY}, C={C_VALUE}, "
        f"solver={SOLVER}, fit_intercept={FIT_INTERCEPT}, class_weight={CLASS_WEIGHT}, "
        f"max_iter={MAX_ITER}, tol={TOL}, random_seed={RANDOM_SEED}")
    add("- M0: intercept_prevalence_baseline, penalty=none")
    add("- 禁止后续搜索 C / penalty / solver / class_weight / tol / max_iter; "
        "阶段2.4 和 2.5 不得根据七月或 Walk-forward 表现调参")
    add("- 收敛状态定义: CONVERGED / MAX_ITER_REACHED / NUMERICAL_FAILURE / "
        "NON_FINITE_COEFFICIENT / NON_FINITE_PREDICTION / INPUT_VALIDATION_FAILURE; "
        "solver 报告未收敛或达到 max_iter 仍未满足停止条件 → convergence_status=FAILED, "
        "即使预测有限也不得标为收敛 (本阶段只冻结定义, 不运行拟合)")

    add("")
    add("## 七月一次性评价协议 (只写协议, 不执行)")
    add("")
    add(f"- 训练 {DEVELOPMENT_START}..{DEVELOPMENT_END} / 测试 {LOCKED_START}..{LOCKED_END}")
    add("- 概率指标: ROC-AUC / PR-AUC / Brier score / log loss / "
        "calibration intercept / calibration slope (校准只评估, 不重新校准)")
    add("- 日期级排序指标: date Top1/Top3 Target7 hit rate / mean Target7 count "
        "in Top3 / eligible signal_date count; 不创建交易阈值; "
        "排序 = predicted_probability 降序 + stock_code 升序")
    add("- 模型规格提交并打标签后才允许揭示七月结果")

    add("")
    add("## 日期级 Walk-forward 协议 (只写协议, 不运行)")
    add("")
    add("- 切分: 按 signal_date 升序, 训练 = 测试日期之前的全部日期, "
        "测试 = 下一完整 signal_date, 同日全部样本同一 fold")
    add("- 最低训练门: train_signal_dates >= 10 / train_rows >= 80 / "
        "train_positive >= 20 / train_negative >= 40; 首个同时满足全部门的日期 "
        "作为首个正式测试日期")
    add("- 每 fold 独立物化派生/检查缺失/裁剪/标准化/拟合; 测试日期不参与训练参数")

    add("")
    add("## 最终候选资格门 (阶段2.6 不得临时创造规则)")
    add("")
    for gate in protocol["final_eligibility_gates"]:
        add(f"- {gate}")
    add(f"- 无候选满足全部条件 → 最终状态 = {protocol['rejection_state']}, 不得降低门槛")

    add("")
    add("## 最终候选比较算法 (global_sequential_tolerance_filter)")
    add("")
    cc = protocol["candidate_comparison"]
    add(f"- 模式: {cc['mode']}; 资格集合优先 (eligibility_first="
        f"{cc['eligibility_first']}); 基于全体合格候选的顺序集合过滤, "
        "不进行任何两模型逐对比较; 全部差值为绝对数值差")
    add(f"- Step 1 OOS log loss: {cc['oos_log_loss']['operation']}, "
        f"绝对容差 = {cc['oos_log_loss']['tolerance']}")
    add(f"- Step 2 OOS Brier: {cc['oos_brier']['operation']}, "
        f"绝对容差 = {cc['oos_brier']['tolerance']}")
    add(f"- Step 3 July log loss: {cc['july_log_loss']['operation']}, "
        f"容差 = {cc['july_log_loss']['tolerance']} "
        f"({cc['july_log_loss']['rule']})")
    add(f"- Step 4 feature_count: {cc['feature_count']['operation']}")
    add(f"- Step 5 固定模型优先序: {' > '.join(cc['fixed_model_priority'])}")
    add("- 禁止: 逐对比较模型 / 相对差值 / 百分比差 / 七月 AUC 打破并列 / "
        "Top1 或 Top3 打破并列 / 临时调整容差 / 看到结果后修改层级; "
        "Top1/Top3 只作为描述性业务排序指标")

    add("")
    add("## 阶段2.4 指标精确定义 (只冻结口径, 不执行)")
    add("")
    md = protocol["metric_definitions"]
    add(f"- ROC-AUC: implementation = {md['roc_auc']['implementation']}, "
        f"positive_label = {md['roc_auc']['positive_label']}; "
        f"评估标签必须同时包含 0 和 1, 否则 {md['roc_auc']['label_requirement'].split(';')[1].strip()}")
    add(f"- PR-AUC: implementation = {md['pr_auc']['implementation']} "
        f"(metric_name = {md['pr_auc']['metric_name']}); "
        f"{md['pr_auc']['note']}; 无正样本 → METRIC_UNDEFINED")
    add(f"- Brier score: 公式 = {md['brier']['formula']}; "
        f"{md['brier']['prediction_requirement']}")
    add(f"- Log loss: epsilon = {md['log_loss']['epsilon']}; "
        f"{md['log_loss']['clip_scope']}")
    add(f"- Calibration: {md['calibration']['model']} "
        f"(implementation = {md['calibration']['implementation']}, "
        f"maxiter = {md['calibration']['maxiter']}, tol = {md['calibration']['tol']}); "
        f"calibration_intercept = alpha, calibration_slope = beta")
    add(f"- M0 校准: calibration_slope = {md['calibration']['m0']['calibration_slope']}; "
        f"校准截距 = {md['calibration']['m0']['calibration_intercept_rule']}")
    add(f"- 校准指标 {md['calibration']['descriptive_only']}")
    add(f"- 指标状态字段: {' / '.join(md['metric_status_values'])}; "
        f"{md['no_placeholder_rule']}")

    add("")
    add("## 系数稳定性只作诊断 (阶段2.5 报告, 不作硬资格门)")
    add("")
    add("- 报告每个特征在有效 fold 中的系数、符号、expected direction 一致比例、"
        "均值/标准差/最小/最大")
    add("- expected direction 一致比例不作为阶段2.6 硬资格门; "
        "不得因系数方向不符而重拟合或删除变量")

    add("")
    add("## Target-blind 锁定测试")
    add("")
    ok_count = int((audit["status"] == "PASS").sum())
    add(f"- 七月锁定审计: {ok_count}/{len(audit)} 项 PASS")
    add("- 审计明细: v004c_stage2_3_holdout_lock_audit_v001.csv")

    add("")
    add("## 声明")
    add("")
    add("阶段2.3 只冻结候选模型规格和后续评价协议;")
    add("M1、M2和M3由研究人员在阅读六月阶段2.2单因子结构和稳定性证据后人工确定, "
        "候选规格在任何七月标签揭示前冻结;")
    add("阶段2.3运行时未读取 Target 列, 未根据 AUC、bootstrap、flags 或其它统计 "
        "自动搜索、增加、删除或替换候选成员;")
    add("没有训练模型; 没有生成预测; 没有读取七月标签; 没有运行 Walk-forward;")
    add("没有搜索超参数; 没有搜索交易阈值;")
    add("阶段1、阶段2.1 和阶段2.2 资产未被修改;")
    add("没有自动 commit 或 push。")
    return "\n".join(lines) + "\n"


def write_data_package(
    *,
    catalog: pd.DataFrame,
    feature_spec: pd.DataFrame,
    preprocessing: pd.DataFrame,
    evidence: pd.DataFrame,
    redundancy: pd.DataFrame,
    boundary: pd.DataFrame,
    audit: pd.DataFrame,
    candidate_specs: dict[str, Any],
    protocol: dict[str, Any],
    design_disclosure: dict[str, Any],
    out_dir: str | Path,
    git_head: str,
    git_status_before: str,
    output_dir_gate: dict[str, Any],
    manifest_draft: dict[str, Any],
) -> dict[str, Any]:
    """把 13 个数据/git 输出文件写入临时目录 (review/git_status_after/manifest 由 finalize 写入); 正式目录共 15 个文件, manifest 记录 14 个非 manifest 文件。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save_csv(frame: pd.DataFrame, name: str) -> Path:
        path = out / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    data_paths = {
        "v004c_stage2_3_model_catalog_v001.csv": save_csv(catalog, "v004c_stage2_3_model_catalog_v001.csv"),
        "v004c_stage2_3_model_feature_spec_v001.csv": save_csv(feature_spec, "v004c_stage2_3_model_feature_spec_v001.csv"),
        "v004c_stage2_3_preprocessing_spec_v001.csv": save_csv(preprocessing, "v004c_stage2_3_preprocessing_spec_v001.csv"),
        "v004c_stage2_3_selected_feature_evidence_v001.csv": save_csv(evidence, "v004c_stage2_3_selected_feature_evidence_v001.csv"),
        "v004c_stage2_3_candidate_redundancy_audit_v001.csv": save_csv(redundancy, "v004c_stage2_3_candidate_redundancy_audit_v001.csv"),
        "v004c_stage2_3_candidate_boundary_v001.csv": save_csv(boundary, "v004c_stage2_3_candidate_boundary_v001.csv"),
        "v004c_stage2_3_holdout_lock_audit_v001.csv": save_csv(audit, "v004c_stage2_3_holdout_lock_audit_v001.csv"),
    }
    json_paths = {
        "v004c_stage2_3_candidate_specs_v001.json": _dump_json(candidate_specs),
        "v004c_stage2_3_evaluation_protocol_v001.json": _dump_json(protocol),
        "v004c_stage2_3_design_disclosure_v001.json": _dump_json(design_disclosure),
    }
    for name, text in json_paths.items():
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
        data_paths[name] = path

    git_paths = {}
    for name, content in (("git_head.txt", git_head),
                          ("git_status_before.txt", git_status_before)):
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        git_paths[name] = path

    draft = dict(manifest_draft)
    draft["output_dir_gate"] = output_dir_gate
    return {"paths": {**data_paths, **git_paths}, "manifest_draft": draft}


def finalize_package(
    *,
    out_dir: str | Path,
    manifest_draft: dict[str, Any],
    review_text: str,
    git_provenance_after: GitProvenance | None,
    git_root: Path,
    generated_at: str | None,
) -> dict[str, Any]:
    """写入 review / git_status_after / manifest, 计算全部输出 SHA。"""
    out = Path(out_dir)
    if git_provenance_after is None:
        git_provenance_after = collect_git_provenance(git_root)
    manifest = dict(manifest_draft)
    manifest["generated_at"] = generated_at or pd.Timestamp.now(
        tz=TIMEZONE).strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["git_status_after"] = git_provenance_after.git_status
    manifest["git_dirty_after"] = bool(git_provenance_after.git_dirty)

    with (out / "git_status_after.txt").open("w", encoding="utf-8") as handle:
        handle.write(git_provenance_after.git_status)
    with (out / "v004c_stage2_3_review.md").open("w", encoding="utf-8") as handle:
        handle.write(review_text)

    output_meta = {}
    all_outputs = {name: Path(out) / name
                   for name in sorted(p.name for p in out.iterdir() if p.is_file())}
    for name, path in all_outputs.items():
        if name == "v004c_stage2_3_manifest.json":
            continue
        output_meta[name] = {
            "path": _repo_relative_posix(path, git_root),
            "rows": None if name.endswith((".txt", ".md", ".json"))
            else _count_csv_rows(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "role": _OUTPUT_ROLE.get(name, "provenance"),
        }
    manifest["output_files"] = output_meta

    manifest_path = out / "v004c_stage2_3_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(_dump_json(manifest))
    return {"paths": all_outputs, "manifest": manifest,
            "manifest_path": manifest_path}


def _write_failure_audit(
    output_dir: Path,
    *,
    failure_audit_dir: Path,
    failure_stage: str,
    exception: Exception,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_2_dir: str | Path,
    stage2_2_ref: str,
    git_head: str,
) -> None:
    """失败审计写入正式输出目录之外的 diagnostics 目录 (任务 8.x)。

    正式输出目录保持不存在或为空; 失败审计内容不含任何七月标签数据。
    """
    payload = {
        "stage": "v004c_stage2_3",
        "failure_stage": failure_stage,
        "timestamp": pd.Timestamp.now(tz=TIMEZONE).strftime(
            "%Y-%m-%dT%H:%M:%S%z"),
        "requested_output_dir": str(output_dir),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "git_head": git_head,
        "branch": EXPECTED_BRANCH,
        "input_version_anchors": {
            "stage1_commit": EXPECTED_STAGE1_DATA_COMMIT,
            "stage2_1_commit": EXPECTED_STAGE2_1_DATA_COMMIT,
            "stage2_2_commit": EXPECTED_STAGE2_2_DATA_COMMIT,
            "stage2_2_ref": stage2_2_ref,
        },
        "input_dirs": {
            "stage1_dir": str(stage1_dir),
            "stage2_1_dir": str(stage2_1_dir),
            "stage2_2_dir": str(stage2_2_dir),
        },
        "formal_output_directory_created": False,
    }
    try:
        failure_audit_dir.mkdir(parents=True, exist_ok=True)
        name = f"v004c_stage2_3_failure_{Path(output_dir).name}.json"
        (failure_audit_dir / name).write_text(
            _dump_json(payload), encoding="utf-8")
    except Exception:
        pass


def _supported_environment(git_root: Path) -> dict[str, str]:
    try:
        autocrlf = _run_git(["git", "config", "--get", "core.autocrlf"],
                            cwd=git_root).strip()
    except DatasetValidationError:
        autocrlf = ""
    try:
        git_version = _run_git(["git", "--version"], cwd=git_root).strip()
    except DatasetValidationError:
        git_version = ""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "core_autocrlf": autocrlf,
        "git_version": git_version,
    }


def run_v004c_stage2_3_candidate_specs(
    *,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_2_dir: str | Path,
    stage2_2_ref: str = EXPECTED_STAGE2_2_TAG,
    output_dir: str | Path,
    git_provenance_before: GitProvenance | None = None,
    git_provenance_after: GitProvenance | None = None,
    enforce_input_sha: bool = True,
    generated_at: str | None = None,
    snapshot_override: pd.DataFrame | None = None,
    failure_audit_dir: str | Path | None = None,
) -> dict[str, Any]:
    """阶段2.3 完整流程 (fail closed; 临时目录构建 + 原子 rename)。

    失败时正式输出目录保持不存在或为空; 失败审计写入外部
    failure_audit_dir (默认 <git_root>/reports/diagnostics/v004c_stage2_3_failures)。
    """
    target_dir = Path(output_dir)
    git_provenance_before = git_provenance_before or collect_git_provenance(Path.cwd())
    git_root = git_provenance_before.git_root
    failure_dir = (Path(failure_audit_dir) if failure_audit_dir
                   else git_root / "reports/diagnostics/v004c_stage2_3_failures")

    preexisting = []
    if target_dir.exists():
        preexisting = sorted(str(p) for p in target_dir.iterdir())
    if preexisting:
        raise DatasetValidationError(
            f"输出目录非空, 阻止冻结: {output_dir} (含 {len(preexisting)} 项; "
            f"请显式删除旧目录后重建)")

    git_chain = validate_git_chain(git_root, stage2_2_ref)

    tmp_dir = Path(f"{target_dir}.tmp_{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        result = _run_core(
            stage1_dir=stage1_dir, stage2_1_dir=stage2_1_dir,
            stage2_2_dir=stage2_2_dir, stage2_2_ref=stage2_2_ref,
            git_root=git_root, git_chain=git_chain,
            provenance_before=git_provenance_before,
            provenance_after=git_provenance_after,
            out_dir=tmp_dir, enforce_input_sha=enforce_input_sha,
            generated_at=generated_at, snapshot_override=snapshot_override,
        )
    except Exception as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            _write_failure_audit(
                target_dir, failure_audit_dir=failure_dir,
                failure_stage="build", exception=exc,
                stage1_dir=stage1_dir, stage2_1_dir=stage2_1_dir,
                stage2_2_dir=stage2_2_dir, stage2_2_ref=stage2_2_ref,
                git_head=git_provenance_before.git_head)
        except Exception:
            pass
        raise

    if target_dir.exists():
        try:
            target_dir.rmdir()
        except OSError as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise DatasetValidationError(
                f"输出目录无法清空: {target_dir}: {exc}") from exc
    os.rename(tmp_dir, target_dir)

    review_text = result["review_text"]
    manifest_draft = result["manifest_draft"]
    final = finalize_package(
        out_dir=target_dir,
        manifest_draft=manifest_draft,
        review_text=review_text,
        git_provenance_after=git_provenance_after,
        git_root=git_root,
        generated_at=generated_at,
    )
    return final


def _verify_inputs_unchanged(input_checks: dict[str, Any],
                             git_root: Path) -> dict[str, Any]:
    """运行结束时重新计算输入 SHA, 与开始记录逐文件比较 (前后一致门)。"""
    changed_by_stage: dict[str, list[str]] = {}
    for name, rec in input_checks["input_sha_verification"].items():
        stage = str(rec.get("stage", "?"))
        path = git_root / rec["path"]
        actual = _sha256(path)
        if actual != rec["sha256"]:
            changed_by_stage.setdefault(stage, []).append(name)
    return {
        "unchanged": not any(changed_by_stage.values()),
        "by_stage": {stage: {"unchanged": not files,
                             "changed_files": sorted(files)}
                     for stage, files in sorted(changed_by_stage.items())},
        "file_count": int(len(input_checks["input_sha_verification"])),
    }


def _run_core(
    *,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_2_dir: str | Path,
    stage2_2_ref: str,
    git_root: Path,
    git_chain: dict[str, Any],
    provenance_before: GitProvenance,
    provenance_after: GitProvenance | None,
    out_dir: Path,
    enforce_input_sha: bool,
    generated_at: str | None,
    snapshot_override: pd.DataFrame | None,
) -> dict[str, Any]:
    """构建全部输出到临时目录; 返回 (paths, manifest_draft, review_text)。"""
    payload = load_inputs(stage1_dir, stage2_1_dir, stage2_2_dir,
                          snapshot_override=snapshot_override)
    input_checks = verify_inputs(
        payload, current_git_root=git_root,
        enforce_input_sha=enforce_input_sha)
    # 实际输入文件集合 (相对 git root 的 POSIX 路径) — 运行时输入审计
    loaded_actual: set[str] = set()
    loaded_expected: set[str] = set()
    for stage, names in ALLOWED_INPUT_FILES.items():
        stage_dir = Path(payload["paths"][stage])
        for name in names:
            loaded_expected.add(_repo_relative_posix(stage_dir / name, git_root))
    for stage, name in payload["loaded_input_files"]:
        loaded_actual.add(_repo_relative_posix(
            Path(payload["paths"][stage]) / name, git_root))
    forbidden_actual = {
        _repo_relative_posix(Path(payload["paths"]["stage1"]) / name, git_root)
        for name in FORBIDDEN_INPUT_FILENAMES}
    input_checks["loaded_input_files"] = loaded_actual
    input_checks["expected_input_files"] = loaded_expected
    input_checks["forbidden_input_files"] = forbidden_actual
    if loaded_actual != loaded_expected:
        raise DatasetValidationError(
            "loaded_input_files_mismatch: 实际读取输入文件与允许清单不一致 "
            f"(实际 {len(loaded_actual)} vs 允许 {len(loaded_expected)}); "
            f"多读={sorted(loaded_actual - loaded_expected)} "
            f"少读={sorted(loaded_expected - loaded_actual)}")
    if loaded_actual & forbidden_actual:
        raise DatasetValidationError(
            "forbidden_input_loaded: 实际读取了禁止输入文件: "
            f"{sorted(loaded_actual & forbidden_actual)}")

    # ---- 规格构建 (证据无关) ----
    catalog = build_model_catalog()
    feature_spec, membership_checks = build_model_feature_spec(payload)
    preprocessing = build_preprocessing_spec(catalog, feature_spec)
    candidate_specs = build_candidate_specs_json(feature_spec)
    protocol = build_evaluation_protocol_json()

    # ---- 证据提取 + 边界 + 冗余审计 + VWAP 重参数化核查 ----
    evidence = build_selected_feature_evidence(payload)
    boundary = build_candidate_boundary(payload)
    redundancy = build_candidate_redundancy_audit(payload["snapshot"])
    vwap_check = compute_vwap_reparameterization(payload["snapshot"])
    design_disclosure = build_design_disclosure(vwap_check)

    # ---- 锁定审计 (基础检查; 内容检查随后追加) ----
    audit = build_holdout_lock_audit(
        catalog=catalog, feature_spec=feature_spec, boundary=boundary,
        evidence=evidence, input_checks=input_checks,
        allowlist=_allowlist_lookup(payload))

    # ---- manifest draft ----
    manifest_draft: dict[str, Any] = {
        "stage2_3_version": STAGE2_3_VERSION,
        "source_stage1_version": "v004c-d1-dataset-0.1",
        "source_stage1_commit": EXPECTED_STAGE1_DATA_COMMIT,
        "source_stage2_1_version": "v004c-factor-dictionary-0.1",
        "source_stage2_1_commit": EXPECTED_STAGE2_1_DATA_COMMIT,
        "source_stage2_2_version": EXPECTED_STAGE2_2_TAG,
        "source_stage2_2_ref": stage2_2_ref,
        "source_stage2_2_commit": EXPECTED_STAGE2_2_DATA_COMMIT,
        "source_stage2_2_code_commit": EXPECTED_STAGE2_2_CODE_COMMIT,
        "candidate_selection_mode": CANDIDATE_SELECTION_MODE,
        "model_count": 4,
        "substantive_model_count": 3,
        "selected_unique_feature_count": 12,
        "model_feature_counts": {
            "M0": 0, "M1": 5, "M2": 5, "M3": 5,
        },
        "june_development_evidence_used_for_manual_design":
            JUNE_DEVELOPMENT_EVIDENCE_USED_FOR_MANUAL_DESIGN,
        "candidate_specs_frozen_before_july_label_access":
            CANDIDATE_SPECS_FROZEN_BEFORE_JULY_LABEL_ACCESS,
        "runtime_candidate_membership_uses_target_columns":
            RUNTIME_CANDIDATE_MEMBERSHIP_USES_TARGET_COLUMNS,
        "runtime_membership_invariant_to_target_column_mutation":
            RUNTIME_MEMBERSHIP_INVARIANT_TO_TARGET_COLUMN_MUTATION,
        "automatic_target_driven_selection": AUTOMATIC_TARGET_DRIVEN_SELECTION,
        "automatic_feature_selection": False,
        "model_training_performed": False,
        "prediction_generated": False,
        "holdout_label_access": False,
        "hyperparameter_search": False,
        "threshold_search": False,
        "interaction_search": False,
        "runtime_membership_invariant_to_evidence_field_mutation": True,
        "direction_is_constraint": False,
        "development_date_range": [DEVELOPMENT_START, DEVELOPMENT_END],
        "locked_date_range": [LOCKED_START, LOCKED_END],
        "timezone": TIMEZONE,
        "research_only": True,
        "deployment_status": "research_only",
        "git_root": str(git_root),
        "git_branch": git_chain["branch"],
        "git_head": provenance_before.git_head,
        "git_status_before": provenance_before.git_status,
        "git_dirty_before": provenance_before.git_dirty,
        "frozen": False,
        "audit_complete": True,
        "supported_environment": _supported_environment(git_root),
    }
    manifest_draft["input_verification"] = {
        "input_rows": input_checks.get("input_rows"),
        "signal_dates": input_checks.get("signal_dates"),
        "lineage_rows": input_checks.get("lineage_rows"),
        "dev_rows": input_checks.get("dev_rows"),
        "locked_rows": input_checks.get("locked_rows"),
        "input_sha_all_verified": input_checks.get("input_sha_all_verified"),
        "stage1_manifest_audit_reference_verified": input_checks.get(
            "stage1_manifest_audit_reference_verified"),
    }
    manifest_draft["input_files"] = input_checks["input_sha_verification"]
    manifest_draft["input_files_unchanged_during_run"] = True

    def audit_manifest_section(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "all_pass": bool((frame["status"] == "PASS").all()),
            "check_count": int(len(frame)),
            "pass_count": int((frame["status"] == "PASS").sum()),
            "checks": [{"check_name": _AUDIT_CHECK_LABELS.get(
                str(r["check_name"]), str(r["check_name"])),
                "status": r["status"]}
                for _, r in frame.iterrows()],
        }

    manifest_draft["holdout_lock_audit"] = audit_manifest_section(audit)
    manifest_draft["catalog_counts"] = {
        "model_count": int(len(catalog)),
        "substantive_model_count": int((catalog["role"] != "BASELINE").sum()),
    }
    manifest_draft["boundary_counts"] = {
        "rows": int(len(boundary)),
        "selected_fixed": int((boundary["boundary_status"] == "SELECTED_FIXED").sum()),
        "not_selected": int(
            (boundary["boundary_status"] == "NOT_SELECTED_NOT_REJECTED").sum()),
    }
    manifest_draft["redundancy_counts"] = {
        "rows": int(len(redundancy)),
        "flags": {str(k): int(v)
                  for k, v in redundancy["redundancy_flag"].value_counts().items()},
    }
    manifest_draft["evidence_counts"] = {
        "rows": int(len(evidence)),
        "unique_features": int(evidence["feature_name"].nunique()),
    }
    manifest_draft["vwap_reparameterization"] = {
        "relationship_type": vwap_check["relationship_type"],
        "dev": {k: v for k, v in vwap_check["dev"].items()},
        "locked": {k: v for k, v in vwap_check["locked"].items()},
    }
    manifest_draft["candidate_comparison"] = {
        "mode": "global_sequential_tolerance_filter",
        "eligibility_first": True,
        "oos_log_loss_tolerance": OOS_LOG_LOSS_TOLERANCE,
        "oos_brier_tolerance": OOS_BRIER_TOLERANCE,
        "fixed_model_priority": ["M1", "M2", "M3"],
    }

    # ---- 写入 13 个输出文件 (review/manifest/git_status_after 由 finalize 写) ----
    written = write_data_package(
        catalog=catalog, feature_spec=feature_spec, preprocessing=preprocessing,
        evidence=evidence, redundancy=redundancy, boundary=boundary, audit=audit,
        candidate_specs=candidate_specs, protocol=protocol,
        design_disclosure=design_disclosure,
        out_dir=out_dir, git_head=provenance_before.git_head,
        git_status_before=provenance_before.git_status,
        output_dir_gate={"was_empty": True},
        manifest_draft=manifest_draft,
    )
    manifest_draft = written["manifest_draft"]

    # ---- 内容检查 1: 已写入 CSV 列名无七月标签指标 ----
    content_checks: list[dict[str, str]] = []
    for name in ("v004c_stage2_3_model_catalog_v001.csv",
                 "v004c_stage2_3_model_feature_spec_v001.csv",
                 "v004c_stage2_3_preprocessing_spec_v001.csv",
                 "v004c_stage2_3_selected_feature_evidence_v001.csv",
                 "v004c_stage2_3_candidate_redundancy_audit_v001.csv",
                 "v004c_stage2_3_candidate_boundary_v001.csv"):
        cols = pd.read_csv(out_dir / name, nrows=0).columns
        bad = [c for c in cols
               if "july" in c.lower() or "locked_target" in c.lower()]
        content_checks.append({
            "check_name": "no_july_target_metric_in_csv_columns",
            "status": "PASS" if not bad else "FAIL",
            "details": (f"{len(bad)} forbidden columns in {name}" if bad
                        else f"{len(cols)} columns of {name} clean"),
        })
    audit = pd.concat(
        [audit, pd.DataFrame(content_checks,
                             columns=["check_name", "status", "details"])],
        ignore_index=True)

    # ---- 内容检查 2: review 无七月指标 (先渲染含当前审计的 review) ----
    review_text = _render_review(
        git_chain=git_chain,
        input_checks=input_checks,
        catalog=catalog,
        feature_spec=feature_spec,
        preprocessing=preprocessing,
        evidence=evidence,
        redundancy=redundancy,
        boundary=boundary,
        audit=audit,
        candidate_specs=candidate_specs,
        protocol=protocol,
        vwap_check=vwap_check,
        git_head=provenance_before.git_head,
    )
    review_ok, review_detail = _no_july_metric_text_scan(review_text, "review")
    content_checks.append({
        "check_name": "no_july_target_metric_in_review",
        "status": "PASS" if review_ok else "FAIL",
        "details": review_detail,
    })
    # 阶段2.3.1 修正 4.4: 六月开发证据参与人工设计的真实披露检查
    disclosure_ok = (
        "由研究人员在阅读六月阶段2.2" in review_text
        and "候选规格在任何七月标签揭示前冻结" in review_text
        and "未读取 Target 列" in review_text)
    content_checks.append({
        "check_name": "JUNE_DEVELOPMENT_EVIDENCE_DISCLOSED",
        "status": "PASS" if disclosure_ok else "FAIL",
        "details": ("review 已披露六月开发证据参与人工设计、规格在七月揭示前冻结、"
                    "运行时未读取 Target" if disclosure_ok
                    else "review 缺少六月证据披露语句"),
    })
    disclosure_text = _dump_json(design_disclosure)
    disc_ok, disc_detail = _no_july_metric_text_scan(
        disclosure_text, "design_disclosure")
    content_checks.append({
        "check_name": "no_july_target_metric_in_design_disclosure",
        "status": "PASS" if disc_ok else "FAIL",
        "details": disc_detail,
    })
    audit = pd.concat(
        [audit, pd.DataFrame(content_checks[-3:],
                             columns=["check_name", "status", "details"])],
        ignore_index=True)
    manifest_draft["holdout_lock_audit"] = audit_manifest_section(audit)
    audit_csv = out_dir / "v004c_stage2_3_holdout_lock_audit_v001.csv"
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")

    # ---- 内容检查 3: manifest 无七月指标 ----
    manifest_text = _dump_json(manifest_draft)
    man_ok, man_detail = _no_july_metric_text_scan(manifest_text, "manifest")
    if not (review_ok and man_ok):
        raise DatasetValidationError(
            "内容扫描失败: review/manifest 含七月指标 token, fail closed")

    # ---- 输入前后一致门 (运行结束时重算输入 SHA) ----
    unchanged = _verify_inputs_unchanged(input_checks, git_root)
    if not unchanged["unchanged"]:
        raise DatasetValidationError(
            "input_changed_during_run: 输入文件在运行期间变化: "
            + str(unchanged["by_stage"]))
    manifest_draft["input_files_unchanged_during_run"] = True
    manifest_draft["input_verification"].update({
        "input_file_count": unchanged["file_count"],
        "actual_loaded_input_file_count": len(input_checks["loaded_input_files"]),
        "expected_allowed_input_file_count": len(
            input_checks["expected_input_files"]),
        "actual_matches_allowed_inputs": (
            input_checks["loaded_input_files"]
            == input_checks["expected_input_files"]),
        "stage1_files_unchanged": unchanged["by_stage"].get(
            "stage1", {}).get("unchanged", True),
        "stage2_1_files_unchanged": unchanged["by_stage"].get(
            "stage2_1", {}).get("unchanged", True),
        "stage2_2_files_unchanged": unchanged["by_stage"].get(
            "stage2_2", {}).get("unchanged", True),
    })

    # ---- 最终 review (含全部审计行) ----
    review_text = _render_review(
        git_chain=git_chain,
        input_checks=input_checks,
        catalog=catalog,
        feature_spec=feature_spec,
        preprocessing=preprocessing,
        evidence=evidence,
        redundancy=redundancy,
        boundary=boundary,
        audit=audit,
        candidate_specs=candidate_specs,
        protocol=protocol,
        vwap_check=vwap_check,
        git_head=provenance_before.git_head,
    )
    return {
        "paths": written["paths"],
        "manifest_draft": manifest_draft,
        "review_text": review_text,
        "input_checks": input_checks,
        "membership_checks": membership_checks,
        "catalog": catalog,
        "feature_spec": feature_spec,
        "evidence": evidence,
        "boundary": boundary,
        "redundancy": redundancy,
        "audit": audit,
    }

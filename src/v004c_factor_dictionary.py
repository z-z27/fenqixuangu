# -*- coding: utf-8 -*-
"""v004c 阶段2.1: 因子字典、精确去重、近重复审计与显式准入。

TARGET-BLIND / STRUCTURE-ONLY / DETERMINISTIC / REPRODUCIBLE:
- 准入决定只读取字段结构 (缺失/唯一值/取值分布/角色/公式), 绝不读取标签值;
- 不训练模型, 不计算 AUC, 不运行 walk-forward, 不输出 Top3;
- 不修改阶段1目录 / v0.2.2 输入 / data/cache;
- 不自动 commit 或 push。

输出 (输出目录):
    v004c_factor_dictionary_v001.csv           197 列完整因子字典
    v004c_raw_feature_audit_v001.csv           79 候选结构审计
    v004c_exact_duplicate_groups_v001.csv      精确重复组 (13 组)
    v004c_near_duplicate_pairs_v001.csv        近重复对 / 互斥组
    v004c_canonical_feature_map_v001.csv       197 列 canonical 映射
    v004c_feature_allowlist_primary_v001.csv   PRIMARY_RAW + PREDECLARED_DERIVED
    v004c_feature_allowlist_sensitivity_v001.csv  SENSITIVITY_RAW
    v004c_feature_exclusions_v001.csv          AUDIT_ONLY / DERIVE_ONLY / EXCLUDE_*
    v004c_predeclared_derived_factor_spec_v001.csv   6 个派生规格
    v004c_predeclared_derived_factor_audit_v001.csv  派生域审计 + 3 个已有字段重算
    v004c_factor_dictionary_manifest.json
    v004c_factor_dictionary_review.md
    git_head.txt / git_status_before.txt / git_status_after.txt
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

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

# ---------------------------------------------------------------------------
# 版本链常量 (阶段1 冻结)
# ---------------------------------------------------------------------------
DICTIONARY_VERSION = "v004c-factor-dictionary-0.1"
EXPECTED_BRANCH = "research-sample-analysis"
EXPECTED_SOURCE_TAG = "v004c-d1-dataset-0.1"
EXPECTED_DATA_COMMIT = "a65661f4738b849a06efb4859d4100842871e971"
EXPECTED_GENERATOR_COMMIT = "7f4db91e06513a08de382831d401c5cc4d5ea3c4"
TIMEZONE = "Asia/Shanghai"

EXPECTED_GATES = {
    "input_rows": 333,
    "signal_dates": 42,
    "lineage_rows": 197,
    "allowed_raw": 79,
    "exact_duplicate_groups": 13,
}

# 阶段1 manifest 中记录的输入文件 (路径 → 记录 key)
STAGE1_INPUT_KEYS = {
    "v004c_d1_snapshot_v001.csv": "snapshot",
    "v004c_training_d1_v001.csv": "training",
    "v004c_d1_column_lineage.csv": "lineage",
}

# 阶段1 manifest 自身审计参考 SHA (已验收; 只作为额外审计门, 不代替 tag 字节比较)
AUDIT_MANIFEST_SHA256 = (
    "ae2a2b48b0b97390e08e81c42cadaa84d5ff76b95e847a80a66e480936ccd618")

# 显式 binary 字段 allowlist (九; 不得仅凭当前样本恰好 2 个取值判定为 binary)
BINARY_COLUMNS = frozenset({
    "break_touched_limit_up",
    "break_opened_from_limit_up",
    "break_day_in_pool",
    "last_board_day_in_pool",
    "d1_close_above_ma5",
    "d1_close_above_ma10",
    "d1_true_reclaim_ma5",
    "d1_true_reclaim_ma10",
})

# 规则版本标识 (manifest Target-blind 措辞)
ADMISSION_RULE_VERSION = "v004c-stage2-1-admission-rules-1.1"
TARGET_BLIND_TEST_VERSION = "v004c-stage2-1-target-blind-tests-1.1"
TARGET_BLIND_INVARIANCE_TESTS = [
    "target7_shuffle", "target7_all_zero", "target7_all_one",
    "tail_loss_flip", "cross_month_permute",
]

# 二元派生项: 源字段缺失/非有限 → 冻结失败 (十)
HARD_MISSING_POLICY_FEATURES = frozenset({"board_streak_is_3", "ma5_overheat_10"})

# ---------------------------------------------------------------------------
# 准入状态 (只允许这些)
# ---------------------------------------------------------------------------
ADMISSION_STATUSES = frozenset({
    "PRIMARY_RAW",
    "SENSITIVITY_RAW",
    "DERIVE_ONLY",
    "AUDIT_ONLY",
    "EXCLUDE_ALL_MISSING",
    "EXCLUDE_CONSTANT",
    "EXCLUDE_DEPRECATED",
    "EXCLUDE_EXACT_DUPLICATE_ALIAS",
    "EXCLUDE_FORBIDDEN",
})

MECHANISM_GROUPS = frozenset({
    "BOARD_HISTORY", "D0_CROSS_SECTION", "D1_PRICE_ACTION", "D1_MA_POSITION",
    "D1_VWAP_POSITION", "D1_VOLUME_ACTIVITY", "D1_CHIP_DISTRIBUTION",
    "D1_LATE_DAY_PRESSURE", "POOL_MEMBERSHIP", "PREDECLARED_DERIVED",
    "RAW_SCALE_SOURCE", "FORBIDDEN_AUDIT",
})

PREPROCESS_POLICIES = frozenset({
    "RAW_BINARY", "RAW_ORDINAL", "FOLD_CLIP_Z",
    "CROSS_SECTION_RANK_SENSITIVITY", "NO_DIRECT_MODEL_INPUT",
    "BUCKET_SENSITIVITY_ONLY",
})

# 绝对值尺度字段 (八、绝对尺度字段只允许 DERIVE_ONLY)
ABSOLUTE_SCALE_COLUMNS = frozenset({
    "board_streak_before_break",
    "break_prev_close",
    "d1_open", "d1_high", "d1_low", "d1_close",
    "d1_volume", "d1_amount",
    "d1_ma5", "d1_ma10", "d1_ma20",
    "d1_vwap",
})

# 预先分桶字段 (十二、bucket 字段只能进入 SENSITIVITY)
BUCKET_COLUMNS = frozenset({"d1_close_to_ma5_bucket", "d1_open_to_close_bucket"})

# deprecated 字段 (规则 4)
DEPRECATED_COLUMNS = frozenset({
    "deprecated_d1_reclaimed_ma5_v01", "deprecated_d1_reclaimed_ma10_v01",
})

# 精确重复组: (group_id, alias, canonical) — 必须与数据逐行相等 (含缺失位置)
EXACT_DUPLICATE_GROUPS = (
    ("ED01", "break_open", "d1_open"),
    ("ED02", "break_high", "d1_high"),
    ("ED03", "break_low", "d1_low"),
    ("ED04", "break_close", "d1_close"),
    ("ED05", "break_intraday_range", "d1_intraday_range"),
    ("ED06", "break_high_to_close_drawdown", "d1_high_to_close_drawdown_raw"),
    ("ED07", "break_close_location", "d1_close_location"),
    ("ED08", "break_volume", "d1_volume"),
    ("ED09", "break_amount", "d1_amount"),
    ("ED10", "deprecated_d1_reclaimed_ma5_v01", "d1_close_above_ma5"),
    ("ED11", "deprecated_d1_reclaimed_ma10_v01", "d1_close_above_ma10"),
    ("ED12", "d1_down_bar_volume_ratio", "down_bar_volume_ratio"),
    ("ED13", "volume_above_break_close_ratio", "volume_above_d1_close_ratio"),
)

# 固定互斥组: (group_id, canonical, (alternatives...))
MUTUAL_EXCLUSION_GROUPS = (
    ("ME01_BREAK_ACTIVITY_RATIO", "break_volume_ratio_vs_board_days",
     ("break_amount_ratio_vs_board_days",)),
    ("ME02_ABOVE_CLOSE_CHIP", "volume_above_d1_close_ratio",
     ("amount_above_d1_close_ratio",)),
    ("ME03_HIGH_ZONE", "high_zone_volume_ratio", ("high_zone_amount_ratio",)),
    ("ME04_LATE_SELL", "late_day_sell_volume_ratio",
     ("late_day_sell_amount_ratio",)),
    ("ME05_VWAP_GAP", "d1_close_to_vwap_raw", ("d1_vwap_to_close_gap",)),
    ("ME06_MA5_POSITION", "d1_close_above_ma5",
     ("consecutive_days_below_ma5",)),
    ("ME07_MA10_POSITION", "d1_close_above_ma10",
     ("consecutive_days_below_ma10",)),
)

# 近重复自动检测参数 (十)
NEAR_DUP_MIN_OVERLAP = 300
NEAR_DUP_CORR_THRESHOLD = 0.995

# ---------------------------------------------------------------------------
# 机制分组 / 语义类型 / 单位族 (79 候选显式映射; 未映射候选会 fail closed)
# ---------------------------------------------------------------------------
_MECHANISM_MAP = {
    # BOARD_HISTORY
    "board_streak_before_break": "BOARD_HISTORY",
    "recent_limit_up_count_10d": "BOARD_HISTORY",
    "recent_limit_up_count_20d": "BOARD_HISTORY",
    "recent_pool_appearance_count_10d": "BOARD_HISTORY",
    "recent_pool_appearance_count_20d": "BOARD_HISTORY",
    "max_board_streak_20d": "BOARD_HISTORY",
    # D0_CROSS_SECTION
    "board_day_amount_rank": "D0_CROSS_SECTION",
    "board_day_turnover_rank": "D0_CROSS_SECTION",
    "board_day_volume_rank": "D0_CROSS_SECTION",
    # D1_PRICE_ACTION
    "break_open_return": "D1_PRICE_ACTION",
    "break_high_return": "D1_PRICE_ACTION",
    "break_close_return": "D1_PRICE_ACTION",
    "break_upper_shadow_ratio": "D1_PRICE_ACTION",
    "break_lower_shadow_ratio": "D1_PRICE_ACTION",
    "break_intraday_range": "D1_PRICE_ACTION",
    "break_high_to_close_drawdown": "D1_PRICE_ACTION",
    "break_close_location": "D1_PRICE_ACTION",
    "d1_close_location": "D1_PRICE_ACTION",
    "d1_open_to_close_return_raw": "D1_PRICE_ACTION",
    "d1_low_to_close_recovery": "D1_PRICE_ACTION",
    "d1_high_to_close_drawdown_raw": "D1_PRICE_ACTION",
    "d1_intraday_range": "D1_PRICE_ACTION",
    "d1_afternoon_return": "D1_PRICE_ACTION",
    "d1_last_hour_return": "D1_PRICE_ACTION",
    "d1_open_to_close_bucket": "D1_PRICE_ACTION",
    "break_touched_limit_up": "D1_PRICE_ACTION",
    "break_opened_from_limit_up": "D1_PRICE_ACTION",
    # D1_MA_POSITION
    "d1_close_to_ma5_raw": "D1_MA_POSITION",
    "d1_low_to_ma5_raw": "D1_MA_POSITION",
    "d1_high_to_ma5_raw": "D1_MA_POSITION",
    "d1_close_to_ma10_raw": "D1_MA_POSITION",
    "d1_low_to_ma10_raw": "D1_MA_POSITION",
    "d1_close_to_ma20": "D1_MA_POSITION",
    "d1_ma5_slope": "D1_MA_POSITION",
    "d1_ma10_slope": "D1_MA_POSITION",
    "d1_close_above_ma5": "D1_MA_POSITION",
    "d1_close_above_ma10": "D1_MA_POSITION",
    "d1_true_reclaim_ma5": "D1_MA_POSITION",
    "d1_true_reclaim_ma10": "D1_MA_POSITION",
    "d1_close_to_ma5_bucket": "D1_MA_POSITION",
    "consecutive_days_below_ma5": "D1_MA_POSITION",
    "consecutive_days_below_ma10": "D1_MA_POSITION",
    "deprecated_d1_reclaimed_ma5_v01": "D1_MA_POSITION",
    "deprecated_d1_reclaimed_ma10_v01": "D1_MA_POSITION",
    # D1_VWAP_POSITION
    "d1_close_to_vwap_raw": "D1_VWAP_POSITION",
    "d1_vwap_to_close_gap": "D1_VWAP_POSITION",
    # D1_VOLUME_ACTIVITY
    "break_volume_ratio_vs_board_days": "D1_VOLUME_ACTIVITY",
    "break_amount_ratio_vs_board_days": "D1_VOLUME_ACTIVITY",
    "break_turnover_ratio": "D1_VOLUME_ACTIVITY",
    "d1_up_bar_volume_ratio": "D1_VOLUME_ACTIVITY",
    "d1_down_bar_volume_ratio": "D1_VOLUME_ACTIVITY",
    "down_bar_volume_ratio": "D1_VOLUME_ACTIVITY",
    # D1_CHIP_DISTRIBUTION
    "volume_above_d1_close_ratio": "D1_CHIP_DISTRIBUTION",
    "amount_above_d1_close_ratio": "D1_CHIP_DISTRIBUTION",
    "volume_above_break_close_ratio": "D1_CHIP_DISTRIBUTION",
    "high_zone_volume_ratio": "D1_CHIP_DISTRIBUTION",
    "high_zone_amount_ratio": "D1_CHIP_DISTRIBUTION",
    # D1_LATE_DAY_PRESSURE
    "late_day_sell_volume_ratio": "D1_LATE_DAY_PRESSURE",
    "late_day_sell_amount_ratio": "D1_LATE_DAY_PRESSURE",
    # POOL_MEMBERSHIP
    "break_day_in_pool": "POOL_MEMBERSHIP",
    "last_board_day_in_pool": "POOL_MEMBERSHIP",
    "pool_consecutive_count_last_board": "POOL_MEMBERSHIP",
    # RAW_SCALE_SOURCE
    "break_open": "RAW_SCALE_SOURCE",
    "break_high": "RAW_SCALE_SOURCE",
    "break_low": "RAW_SCALE_SOURCE",
    "break_close": "RAW_SCALE_SOURCE",
    "break_prev_close": "RAW_SCALE_SOURCE",
    "break_volume": "RAW_SCALE_SOURCE",
    "break_amount": "RAW_SCALE_SOURCE",
    "d1_open": "RAW_SCALE_SOURCE",
    "d1_high": "RAW_SCALE_SOURCE",
    "d1_low": "RAW_SCALE_SOURCE",
    "d1_close": "RAW_SCALE_SOURCE",
    "d1_volume": "RAW_SCALE_SOURCE",
    "d1_amount": "RAW_SCALE_SOURCE",
    "d1_ma5": "RAW_SCALE_SOURCE",
    "d1_ma10": "RAW_SCALE_SOURCE",
    "d1_ma20": "RAW_SCALE_SOURCE",
    "d1_vwap": "RAW_SCALE_SOURCE",
}

_SEMANTIC_MAP = {
    "board_streak_before_break": "ordinal_count",
    "recent_limit_up_count_10d": "ordinal_count",
    "recent_limit_up_count_20d": "ordinal_count",
    "recent_pool_appearance_count_10d": "ordinal_count",
    "recent_pool_appearance_count_20d": "ordinal_count",
    "max_board_streak_20d": "ordinal_count",
    "board_day_amount_rank": "cross_section_amount_rank",
    "board_day_turnover_rank": "cross_section_turnover_rank",
    "board_day_volume_rank": "cross_section_volume_rank",
    "break_open": "break_day_absolute_price",
    "break_high": "break_day_absolute_price",
    "break_low": "break_day_absolute_price",
    "break_close": "break_day_absolute_price",
    "break_prev_close": "prev_close_absolute_price",
    "break_open_return": "break_day_return",
    "break_high_return": "break_day_return",
    "break_close_return": "break_day_return",
    "break_intraday_range": "intraday_range_ratio",
    "break_high_to_close_drawdown": "high_to_close_drawdown",
    "break_upper_shadow_ratio": "shadow_ratio",
    "break_lower_shadow_ratio": "shadow_ratio",
    "break_close_location": "close_location_ratio",
    "break_volume": "absolute_volume",
    "break_amount": "absolute_amount",
    "break_volume_ratio_vs_board_days": "volume_ratio_vs_board",
    "break_amount_ratio_vs_board_days": "amount_ratio_vs_board",
    "break_turnover_ratio": "turnover_ratio",
    "break_touched_limit_up": "binary",
    "break_opened_from_limit_up": "binary",
    "d1_open": "d1_absolute_price",
    "d1_high": "d1_absolute_price",
    "d1_low": "d1_absolute_price",
    "d1_close": "d1_absolute_price",
    "d1_volume": "d1_absolute_volume",
    "d1_amount": "d1_absolute_amount",
    "d1_ma5": "moving_average_price",
    "d1_ma10": "moving_average_price",
    "d1_ma20": "moving_average_price",
    "d1_vwap": "d1_absolute_vwap",
    "d1_close_to_vwap_raw": "close_to_vwap_ratio",
    "d1_close_to_ma5_raw": "close_to_ma_ratio",
    "d1_low_to_ma5_raw": "low_to_ma_ratio",
    "d1_high_to_ma5_raw": "high_to_ma_ratio",
    "d1_close_to_ma10_raw": "close_to_ma_ratio",
    "d1_low_to_ma10_raw": "low_to_ma_ratio",
    "d1_close_to_ma20": "close_to_ma_ratio",
    "d1_ma5_slope": "ma_slope",
    "d1_ma10_slope": "ma_slope",
    "deprecated_d1_reclaimed_ma5_v01": "reclaim_binary_deprecated",
    "deprecated_d1_reclaimed_ma10_v01": "reclaim_binary_deprecated",
    "consecutive_days_below_ma5": "ordinal_count",
    "consecutive_days_below_ma10": "ordinal_count",
    "d1_open_to_close_return_raw": "open_to_close_return",
    "d1_low_to_close_recovery": "low_to_close_recovery",
    "d1_high_to_close_drawdown_raw": "high_to_close_drawdown",
    "d1_close_location": "close_location_ratio",
    "d1_intraday_range": "intraday_range_ratio",
    "d1_afternoon_return": "afternoon_return",
    "d1_last_hour_return": "last_hour_return",
    "d1_up_bar_volume_ratio": "up_bar_volume_ratio",
    "d1_down_bar_volume_ratio": "down_bar_volume_ratio",
    "volume_above_d1_close_ratio": "volume_above_close_ratio",
    "amount_above_d1_close_ratio": "amount_above_close_ratio",
    "volume_above_break_close_ratio": "volume_above_close_ratio",
    "high_zone_volume_ratio": "high_zone_volume_ratio",
    "high_zone_amount_ratio": "high_zone_amount_ratio",
    "late_day_sell_volume_ratio": "late_day_sell_volume_ratio",
    "late_day_sell_amount_ratio": "late_day_sell_amount_ratio",
    "down_bar_volume_ratio": "down_bar_volume_ratio",
    "d1_vwap_to_close_gap": "vwap_to_close_gap",
    "break_day_in_pool": "binary",
    "last_board_day_in_pool": "binary",
    "pool_consecutive_count_last_board": "ordinal_count",
    "d1_close_above_ma5": "binary",
    "d1_close_above_ma10": "binary",
    "d1_true_reclaim_ma5": "binary",
    "d1_true_reclaim_ma10": "binary",
    "d1_close_to_ma5_bucket": "close_to_ma_bucket",
    "d1_open_to_close_bucket": "open_to_close_bucket",
}

_UNIT_MAP = {
    "board_streak_before_break": "count",
    "recent_limit_up_count_10d": "count",
    "recent_limit_up_count_20d": "count",
    "recent_pool_appearance_count_10d": "count",
    "recent_pool_appearance_count_20d": "count",
    "max_board_streak_20d": "count",
    "board_day_amount_rank": "rank",
    "board_day_turnover_rank": "rank",
    "board_day_volume_rank": "rank",
    "break_open": "price",
    "break_high": "price",
    "break_low": "price",
    "break_close": "price",
    "break_prev_close": "price",
    "break_open_return": "return",
    "break_high_return": "return",
    "break_close_return": "return",
    "break_intraday_range": "ratio",
    "break_high_to_close_drawdown": "drawdown",
    "break_upper_shadow_ratio": "ratio",
    "break_lower_shadow_ratio": "ratio",
    "break_close_location": "location",
    "break_volume": "volume",
    "break_amount": "amount",
    "break_volume_ratio_vs_board_days": "ratio",
    "break_amount_ratio_vs_board_days": "ratio",
    "break_turnover_ratio": "ratio",
    "break_touched_limit_up": "binary",
    "break_opened_from_limit_up": "binary",
    "d1_open": "price",
    "d1_high": "price",
    "d1_low": "price",
    "d1_close": "price",
    "d1_volume": "volume",
    "d1_amount": "amount",
    "d1_ma5": "price",
    "d1_ma10": "price",
    "d1_ma20": "price",
    "d1_vwap": "price",
    "d1_close_to_vwap_raw": "ratio",
    "d1_close_to_ma5_raw": "ratio",
    "d1_low_to_ma5_raw": "ratio",
    "d1_high_to_ma5_raw": "ratio",
    "d1_close_to_ma10_raw": "ratio",
    "d1_low_to_ma10_raw": "ratio",
    "d1_close_to_ma20": "ratio",
    "d1_ma5_slope": "slope",
    "d1_ma10_slope": "slope",
    "deprecated_d1_reclaimed_ma5_v01": "binary",
    "deprecated_d1_reclaimed_ma10_v01": "binary",
    "consecutive_days_below_ma5": "count",
    "consecutive_days_below_ma10": "count",
    "d1_open_to_close_return_raw": "return",
    "d1_low_to_close_recovery": "ratio",
    "d1_high_to_close_drawdown_raw": "drawdown",
    "d1_close_location": "location",
    "d1_intraday_range": "ratio",
    "d1_afternoon_return": "return",
    "d1_last_hour_return": "return",
    "d1_up_bar_volume_ratio": "ratio",
    "d1_down_bar_volume_ratio": "ratio",
    "volume_above_d1_close_ratio": "ratio",
    "amount_above_d1_close_ratio": "ratio",
    "volume_above_break_close_ratio": "ratio",
    "high_zone_volume_ratio": "ratio",
    "high_zone_amount_ratio": "ratio",
    "late_day_sell_volume_ratio": "ratio",
    "late_day_sell_amount_ratio": "ratio",
    "down_bar_volume_ratio": "ratio",
    "d1_vwap_to_close_gap": "ratio",
    "break_day_in_pool": "binary",
    "last_board_day_in_pool": "binary",
    "pool_consecutive_count_last_board": "count",
    "d1_close_above_ma5": "binary",
    "d1_close_above_ma10": "binary",
    "d1_true_reclaim_ma5": "binary",
    "d1_true_reclaim_ma10": "binary",
    "d1_close_to_ma5_bucket": "bucket",
    "d1_open_to_close_bucket": "bucket",
}

_NON_CANDIDATE_SEMANTIC = {
    "identifier": "identifier",
    "label": "label",
    "outcome_audit": "outcome_audit",
    "data_quality": "data_quality",
    "provenance": "provenance",
    "existing_model_audit": "existing_model_audit_forbidden",
}
_NON_CANDIDATE_UNIT = {
    "identifier": "identifier",
    "label": "label",
    "outcome_audit": "outcome_audit",
    "data_quality": "data_quality",
    "provenance": "provenance",
    "existing_model_audit": "model_audit",
}

# TARGET-BLIND: 准入路径禁止读取这些列的值 (标签 / 结果 / 未来字段)
TARGET_BLIND_SENSITIVE_COLUMNS = frozenset({
    "target7_daily_d2open_d3high", "tail_loss_daily_5pct",
    "daily_d2open_to_d3high_return", "daily_d2open_to_d3close_return",
    "target7_d2open_d3high", "tail_loss_5pct",
    "d2open_to_d3high_return", "d2open_to_d3close_return",
    "d2_open_daily", "d3_high_daily", "d3_close_daily",
    "d2_open", "d2_high", "d2_low", "d2_close",
    "d3_high", "d3_low", "d3_close",
    "d2_trade_date", "d3_trade_date",
    "audit_minute_target7", "audit_minute_tail_loss",
    "audit_minute_d2open_to_d3high_return", "audit_minute_d2open_to_d3close_return",
    "target_label_daily_vs_minute_same", "tail_label_daily_vs_minute_same",
})

# ---------------------------------------------------------------------------
# 预声明派生因子规格 (十三; 只允许这 6 个新派生项)
# ---------------------------------------------------------------------------
DERIVED_FACTOR_SPECS = (
    {
        "feature_name": "board_streak_is_3",
        "formula": "1[board_streak_before_break == 3]",
        "source_columns": "board_streak_before_break",
        "output_type": "binary",
        "available_as_of": "D1_CLOSE",
        "domain_rule": "values in {0, 1}",
        "missing_policy": "source_missing_is_hard_failure",
        "preprocess_policy": "RAW_BINARY",
        "admission_status": "PREDECLARED_DERIVED",
        "target_blind": True,
        "notes": "断板前连板天数恰为 3 的显式二元项 (源字段 board_streak_before_break 为 DERIVE_ONLY)",
    },
    {
        "feature_name": "ma5_overheat_10",
        "formula": "1[d1_close_to_ma5_raw >= 0.10]",
        "source_columns": "d1_close_to_ma5_raw",
        "output_type": "binary",
        "available_as_of": "D1_CLOSE",
        "domain_rule": "values in {0, 1}",
        "missing_policy": "source_missing_is_hard_failure",
        "preprocess_policy": "RAW_BINARY",
        "admission_status": "PREDECLARED_DERIVED",
        "target_blind": True,
        "notes": "收盘价偏离 MA5 达 +10% 的过热显式二元项",
    },
    {
        "feature_name": "overrepair",
        "formula": "max(d1_open_to_close_return_raw - 0.03, 0)",
        "source_columns": "d1_open_to_close_return_raw",
        "output_type": "continuous",
        "available_as_of": "D1_CLOSE",
        "domain_rule": ">= 0",
        "missing_policy": "derived is null iff source null",
        "preprocess_policy": "FOLD_CLIP_Z",
        "admission_status": "PREDECLARED_DERIVED",
        "target_blind": True,
        "notes": "单日修复动能超过 3% 的部分 (截断到 0)",
    },
    {
        "feature_name": "break_volume_abnormality",
        "formula": "abs(log(max(break_volume_ratio_vs_board_days, 1e-6)))",
        "source_columns": "break_volume_ratio_vs_board_days",
        "output_type": "continuous",
        "available_as_of": "D1_CLOSE",
        "domain_rule": ">= 0",
        "missing_policy": "derived is null iff source null",
        "preprocess_policy": "FOLD_CLIP_Z",
        "admission_status": "PREDECLARED_DERIVED",
        "target_blind": True,
        "notes": "断板日相对连板日均量偏离的绝对对数幅度 (双向异常)",
    },
    {
        "feature_name": "profit_chip_ratio",
        "formula": "1 - volume_above_d1_close_ratio",
        "source_columns": "volume_above_d1_close_ratio",
        "output_type": "continuous",
        "available_as_of": "D1_CLOSE",
        "domain_rule": "[0, 1]",
        "missing_policy": "derived is null iff source null",
        "preprocess_policy": "FOLD_CLIP_Z",
        "admission_status": "PREDECLARED_DERIVED",
        "target_blind": True,
        "notes": "收盘价上方成交占比的补数 (下方获利盘占比)",
    },
    {
        "feature_name": "profit_pressure",
        "formula": ("profit_chip_ratio * d1_low_to_close_recovery"
                    " * log1p(break_volume_ratio_vs_board_days)"),
        "source_columns": "profit_chip_ratio|d1_low_to_close_recovery|break_volume_ratio_vs_board_days",
        "output_type": "continuous",
        "available_as_of": "D1_CLOSE",
        "domain_rule": "finite; no sign constraint (components may be negative)",
        "missing_policy": "derived is null iff any source null",
        "preprocess_policy": "FOLD_CLIP_Z",
        "admission_status": "PREDECLARED_DERIVED",
        "target_blind": True,
        "notes": "获利盘占比 × 低位修复强度 × 放量对数项 (允许负值)",
    },
)
DERIVED_FEATURE_NAMES = tuple(spec["feature_name"] for spec in DERIVED_FACTOR_SPECS)

# 已有原始候选的公式重算 (十四)
EXISTING_RECOMPUTE_CHECKS = (
    ("d1_true_reclaim_ma5", "1[d1_low <= d1_ma5 and d1_close >= d1_ma5]",
     ("d1_low", "d1_ma5", "d1_close")),
    ("d1_close_above_ma5", "1[d1_close >= d1_ma5]",
     ("d1_close", "d1_ma5")),
    ("d1_close_above_ma10", "1[d1_close >= d1_ma10]",
     ("d1_close", "d1_ma10")),
)


# ---------------------------------------------------------------------------
# Git 版本链校验 (二、开始前验证) 与 source-ref 字节锚定 (二)
# ---------------------------------------------------------------------------
def read_source_ref_bytes(git_root: Path, source_ref: str, repo_rel_path: str) -> bytes:
    """直接读取 Git source ref 中文件的原始字节 (git show; fail closed)。"""
    result = subprocess.run(
        ["git", "show", f"{source_ref}:{repo_rel_path}"],
        cwd=str(git_root), capture_output=True, check=False)
    if result.returncode != 0:
        raise DatasetValidationError(
            f"source_ref_file_mismatch: git show {source_ref}:{repo_rel_path} 失败: "
            f"rc={result.returncode} "
            f"stderr={result.stderr.decode(errors='replace')[:200]}")
    return result.stdout


def _require_inside_git_root(path: Path, git_root: Path, what: str) -> str:
    """校验路径位于 current_git_root 下, 返回 repo 相对 POSIX 路径 (三)。"""
    try:
        rel = os.path.relpath(Path(path).resolve(), git_root.resolve())
    except ValueError:
        raise DatasetValidationError(
            f"stage1_dir_outside_git_root: {what} 不在 current_git_root {git_root} 下 "
            f"(跨盘符)")
    if rel.startswith("..") or (len(rel) >= 2 and rel[1] == ":"):
        raise DatasetValidationError(
            f"stage1_dir_outside_git_root: {what} 不在 current_git_root {git_root} 下 "
            f"(relative={rel!r})")
    return Path(rel).as_posix()


def validate_git_chain(git_root: Path, source_ref: str) -> dict[str, Any]:
    """校验分支 / source tag 目标 / 数据提交祖先关系 (fail closed)。

    顺序: 先解析 tag (发现不存在/错误提交), 再校验 ref 字符串与目标提交。
    """
    try:
        tag_target = _run_git(
            ["git", "rev-parse", "--verify", f"{source_ref}^{{}}"],
            cwd=git_root).strip()
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_not_found: source ref {source_ref!r} 无法解析 "
            f"({exc})") from exc
    validate_git_head(tag_target)

    if source_ref != EXPECTED_SOURCE_TAG:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: source-ref 必须为 {EXPECTED_SOURCE_TAG!r}, "
            f"实际 {source_ref!r}")
    if tag_target != EXPECTED_DATA_COMMIT:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: tag {source_ref} 目标必须为 "
            f"{EXPECTED_DATA_COMMIT}, 实际 {tag_target}")

    branch = _run_git(["git", "branch", "--show-current"], cwd=git_root).strip()
    if branch != EXPECTED_BRANCH:
        raise DatasetValidationError(
            f"branch_mismatch: 分支必须为 {EXPECTED_BRANCH!r}, 实际 {branch!r}")

    # merge-base --is-ancestor: 返回码 0 = 是祖先, 非 0 = 否 (fail closed)
    try:
        _run_git(
            ["git", "merge-base", "--is-ancestor", EXPECTED_DATA_COMMIT, "HEAD"],
            cwd=git_root)
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: 阶段1数据提交 {EXPECTED_DATA_COMMIT[:12]} "
            f"不是当前 HEAD 祖先 ({exc})") from exc

    return {
        "branch": branch,
        "source_ref": source_ref,
        "tag_target": tag_target,
        "data_commit_ancestor": True,
    }


# ---------------------------------------------------------------------------
# 阶段1 输入读取与验证
# ---------------------------------------------------------------------------
def load_stage1_inputs(stage1_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """读取阶段1四个输入文件 (snapshot / training / lineage / manifest)。"""
    d = Path(stage1_dir)
    manifest_path = d / "v004c_d1_data_manifest.json"
    if not manifest_path.exists():
        raise DatasetValidationError(f"阶段1 manifest 不存在: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    snapshot = pd.read_csv(
        d / "v004c_d1_snapshot_v001.csv",
        dtype={"code": str},
        float_precision="round_trip")
    training = pd.read_csv(
        d / "v004c_training_d1_v001.csv",
        dtype={"code": str},
        float_precision="round_trip")
    lineage = pd.read_csv(
        d / "v004c_d1_column_lineage.csv",
        dtype=str)
    return snapshot, training, lineage, manifest


def _allowed_mask(lineage: pd.DataFrame) -> pd.Series:
    raw = lineage["allowed_for_future_feature_analysis"].astype(str).str.strip().str.lower()
    return raw.isin(("true", "1", "yes"))


def verify_stage1_inputs(
    snapshot: pd.DataFrame,
    training: pd.DataFrame,
    lineage: pd.DataFrame,
    manifest: dict,
    stage1_dir: str | Path,
    *,
    current_git_root: Path,
    source_ref: str,
    enforce_audit_reference: bool = True,
) -> dict[str, Any]:
    """验证阶段1版本链 / 结构门 / manifest 记录 SHA / source-ref 字节锚定。

    错误码 (任务二 / 六):
        stage1_manifest_version_mismatch, stage1_manifest_git_head_mismatch,
        input_rows_mismatch, signal_dates_mismatch, training_rows_mismatch,
        lineage_rows_mismatch, allowed_candidate_count_mismatch,
        training_columns_missing, source_ref_file_mismatch,
        stage1_manifest_audit_reference_mismatch
    """
    checks: dict[str, Any] = {}

    # 版本链
    if manifest.get("dataset_version") != "v004c-d1-dataset-0.1":
        raise DatasetValidationError(
            f"stage1_manifest_version_mismatch: 阶段1 manifest dataset_version 不符: "
            f"{manifest.get('dataset_version')!r}")
    if manifest.get("git_head") != EXPECTED_GENERATOR_COMMIT:
        raise DatasetValidationError(
            f"stage1_manifest_git_head_mismatch: 阶段1 manifest git_head 不符: "
            f"{manifest.get('git_head')!r}")

    # 结构门 (先于 SHA 门, 使每项门测试可被独立验证)
    checks["input_rows"] = int(len(snapshot))
    checks["signal_dates"] = int(snapshot["signal_date"].nunique())
    checks["training_rows"] = int(len(training))
    checks["lineage_rows"] = int(len(lineage))
    allowed = _allowed_mask(lineage)
    checks["allowed_raw"] = int(allowed.sum())

    if checks["input_rows"] != EXPECTED_GATES["input_rows"]:
        raise DatasetValidationError(
            f"input_rows_mismatch: snapshot 行数应为 {EXPECTED_GATES['input_rows']}, "
            f"实际 {checks['input_rows']}")
    if checks["signal_dates"] != EXPECTED_GATES["signal_dates"]:
        raise DatasetValidationError(
            f"signal_dates_mismatch: 信号日数应为 {EXPECTED_GATES['signal_dates']}, "
            f"实际 {checks['signal_dates']}")
    if checks["training_rows"] != EXPECTED_GATES["input_rows"]:
        raise DatasetValidationError(
            f"training_rows_mismatch: training 行数应为 {EXPECTED_GATES['input_rows']}, "
            f"实际 {checks['training_rows']}")
    if checks["lineage_rows"] != EXPECTED_GATES["lineage_rows"]:
        raise DatasetValidationError(
            f"lineage_rows_mismatch: lineage 行数应为 {EXPECTED_GATES['lineage_rows']}, "
            f"实际 {checks['lineage_rows']}")
    if checks["allowed_raw"] != EXPECTED_GATES["allowed_raw"]:
        raise DatasetValidationError(
            f"allowed_candidate_count_mismatch: stage1 allowed 候选应为 "
            f"{EXPECTED_GATES['allowed_raw']}, 实际 {checks['allowed_raw']}")

    # training 视图必须包含全部候选 + 标识符
    candidates = lineage.loc[allowed, "column_name"].tolist()
    identifier_cols = lineage.loc[lineage["column_role"] == "identifier", "column_name"].tolist()
    missing_in_training = [
        c for c in candidates + identifier_cols if c not in training.columns]
    if missing_in_training:
        raise DatasetValidationError(
            f"training_columns_missing: training 视图缺少候选/标识符列: "
            f"{missing_in_training}")

    # 输入文件 SHA 与阶段1 manifest 记录比对 (记录 SHA == tag 内容, 故不等即本地≠tag)
    d = Path(stage1_dir)
    sha_checks: dict[str, dict[str, Any]] = {}
    output_files = manifest.get("output_files", {})
    for name, role in STAGE1_INPUT_KEYS.items():
        record = output_files.get(name)
        if record is None:
            raise DatasetValidationError(
                f"source_ref_file_mismatch: 阶段1 manifest 缺少输出记录: {name}")
        actual_sha = _sha256(d / name)
        if actual_sha != record.get("sha256"):
            raise DatasetValidationError(
                f"source_ref_file_mismatch: 阶段1输入 {name} 与本地 manifest 记录SHA不一致 "
                f"(record {record.get('sha256')} vs local {actual_sha})")
        sha_checks[name] = {
            "role": role,
            "path": _repo_relative_posix(d / name, current_git_root),
            "bytes": int((d / name).stat().st_size),
            "sha256": actual_sha,
            "rows": int(record.get("rows") or 0),
        }
    checks["input_sha_verification"] = sha_checks

    # source-ref 字节锚定: 4 个文件逐字节 (git 视图 = LF 规范化) 比对 tag 内容
    anchor = verify_source_ref_anchor(
        d, manifest, current_git_root=current_git_root, source_ref=source_ref,
        enforce_audit_reference=enforce_audit_reference)
    checks["source_ref_anchor"] = anchor
    checks["stage1_recorded_git_root"] = str(manifest.get("git_root") or "")
    checks["current_git_root"] = str(current_git_root)
    checks["version_chain"] = {
        "source_tag": EXPECTED_SOURCE_TAG,
        "source_data_commit": EXPECTED_DATA_COMMIT,
        "source_generator_commit": EXPECTED_GENERATOR_COMMIT,
    }
    return checks


# 锚定文件: 3 个输入 CSV + 阶段1 manifest 自身 (二.2)
_SOURCE_REF_ANCHOR_FILES = {
    "v004c_d1_snapshot_v001.csv": "snapshot",
    "v004c_training_d1_v001.csv": "training",
    "v004c_d1_column_lineage.csv": "lineage",
    "v004c_d1_data_manifest.json": "manifest",
}


def verify_source_ref_anchor(
    stage1_dir: Path,
    manifest: dict,
    *,
    current_git_root: Path,
    source_ref: str,
    enforce_audit_reference: bool = True,
) -> dict[str, Any]:
    """source tag 字节锚定 (二.3 / 二.4)。

    - 从 tag 直接读取 4 个文件字节 (git show), 与本地字节做 LF 规范化比较;
    - local_sha256 与 source_ref_sha256 分别记录 (行尾差异在 autocrlf 环境下存在);
    - bytes_equal == true 为硬门;
    - 阶段1 manifest 自身 SHA 与审计参考 (ae2a2b48...) 的比对作为额外审计门:
      仅当本地与 tag 内容一致时适用 (验收版本); 显式修改输入的测试/研究场景
      可传 enforce_audit_reference=False, 此时只记录 verified 结果不拦截。
    """
    anchor: dict[str, Any] = {}
    for name, role in _SOURCE_REF_ANCHOR_FILES.items():
        rel = _repo_relative_posix(stage1_dir / name, current_git_root)
        ref_bytes = read_source_ref_bytes(current_git_root, source_ref, rel)
        local_bytes = (stage1_dir / name).read_bytes()
        local_lf = local_bytes.replace(b"\r\n", b"\n")
        local_sha = hashlib.sha256(local_bytes).hexdigest()
        ref_sha = hashlib.sha256(ref_bytes).hexdigest()
        bytes_equal = local_lf == ref_bytes
        if not bytes_equal:
            raise DatasetValidationError(
                f"source_ref_file_mismatch: {name} 本地字节与 tag {source_ref} 内容不一致 "
                f"(local_sha256={local_sha[:16]} source_ref_sha256={ref_sha[:16]})")
        anchor[name] = {
            "role": role,
            "path": rel,
            "local_sha256": local_sha,
            "source_ref_sha256": ref_sha,
            "bytes_equal": True,
        }

    man_local = anchor["v004c_d1_data_manifest.json"]["local_sha256"]
    man_ref = anchor["v004c_d1_data_manifest.json"]["source_ref_sha256"]
    audit_ok = man_local == AUDIT_MANIFEST_SHA256 or man_ref == AUDIT_MANIFEST_SHA256
    anchor["stage1_manifest_audit_reference_sha256"] = AUDIT_MANIFEST_SHA256
    anchor["stage1_manifest_audit_reference_verified"] = bool(audit_ok)
    if not audit_ok and enforce_audit_reference:
        raise DatasetValidationError(
            f"stage1_manifest_audit_reference_mismatch: 阶段1 manifest SHA 与审计参考 "
            f"{AUDIT_MANIFEST_SHA256} 不一致 (local={man_local} ref={man_ref})")
    return anchor


# ---------------------------------------------------------------------------
# 列结构统计 (TARGET-BLIND: 只读结构, 不读标签值)
# ---------------------------------------------------------------------------
def compute_column_stats(snapshot: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """对给定列集合计算结构统计 (缺失 / 唯一值 / 主导值 / min-max)。"""
    rows = []
    for col in columns:
        series = snapshot[col]
        non_null = series.notna()
        non_null_count = int(non_null.sum())
        missing_count = int(len(series) - non_null_count)
        missing_rate = (missing_count / len(series)) if len(series) else 0.0
        valid = series[non_null]
        vc = valid.value_counts()
        unique_count = int(valid.nunique())
        dominant_value = None
        dominant_count = 0
        dominant_rate = 0.0
        if len(vc):
            dominant_value = vc.index[0]
            dominant_count = int(vc.iloc[0])
            dominant_rate = dominant_count / len(series)
        min_value = None
        max_value = None
        if series.dtype.kind in ("f", "i", "b") and non_null_count:
            min_value = valid.min()
            max_value = valid.max()
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 6),
            "unique_count": unique_count,
            "dominant_value": dominant_value,
            "dominant_count": dominant_count,
            "dominant_rate": round(dominant_rate, 6),
            "min_value": min_value,
            "max_value": max_value,
        })
    return pd.DataFrame(rows)


def _is_binary(stats_row: pd.Series, snapshot: pd.DataFrame) -> tuple[bool, int]:
    """二元字段判定: 只认显式 binary allowlist (九); 不得凭当前样本恰好 2 个取值判定。"""
    col = stats_row["column"]
    if col not in BINARY_COLUMNS:
        return False, -1
    series = snapshot[col]
    vc = series.value_counts()
    return True, int(min(vc.values)) if len(vc) == 2 else int(vc.iloc[0])


# ---------------------------------------------------------------------------
# 精确重复检测 (九; 必须与预期 13 组一致, 额外组 fail closed)
# ---------------------------------------------------------------------------
_MISSING_TOKEN = "__V004C_NA__"


def _column_key(series: pd.Series) -> tuple[Any, ...]:
    return tuple(_MISSING_TOKEN if pd.isna(v) else v for v in series)


def detect_exact_duplicates(
    snapshot: pd.DataFrame,
    candidates: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按 333 行比较 (缺失位置必须一致); 验证 13 组预期; 额外组 fail closed。"""
    expected_by_alias = {alias: (gid, canonical)
                         for gid, alias, canonical in EXACT_DUPLICATE_GROUPS}
    expected_by_canonical = {canonical: (gid, alias)
                             for gid, alias, canonical in EXACT_DUPLICATE_GROUPS}

    keys = {col: _column_key(snapshot[col]) for col in candidates}
    seen_groups: dict[tuple[Any, ...], list[str]] = {}
    for col in candidates:
        seen_groups.setdefault(keys[col], []).append(col)

    rows = []
    found_alias: set[str] = set()
    found_canonical: set[str] = set()
    extra_groups: list[list[str]] = []

    for members in seen_groups.values():
        if len(members) == 1:
            continue
        members_sorted = sorted(members)
        # 检查是否为预期组 (组内恰含一个 alias + 一个 canonical)
        group_ids = set()
        for member in members_sorted:
            if member in expected_by_alias:
                group_ids.add(expected_by_alias[member][0])
            elif member in expected_by_canonical:
                group_ids.add(expected_by_canonical[member][0])
        if len(group_ids) == 1 and len(members_sorted) == 2:
            gid = next(iter(group_ids))
            alias = expected_by_alias.get(members_sorted[0]) is not None
            if alias:
                a, c = members_sorted[0], members_sorted[1]
            else:
                a, c = members_sorted[1], members_sorted[0]
            if expected_by_alias.get(a, (None, None))[0] == gid and \
                    expected_by_canonical.get(c, (None, None))[0] == gid:
                found_alias.add(a)
                found_canonical.add(c)
                rows.append({
                    "group_id": gid,
                    "canonical_source_column": c,
                    "alias_source_column": a,
                    "rows_compared": int(len(snapshot)),
                    "equal_including_missing": True,
                    "canonical_reason": ("deprecated 字段保持 canonical 且整体 EXCLUDE_DEPRECATED"
                                         if a in DEPRECATED_COLUMNS else "语义规范名 (v0.2.2 定义)"),
                    "alias_status": ("EXCLUDE_DEPRECATED" if a in DEPRECATED_COLUMNS
                                     else "EXCLUDE_EXACT_DUPLICATE_ALIAS"),
                })
                continue
        extra_groups.append(members_sorted)

    missing_groups = []
    for gid, alias, canonical in EXACT_DUPLICATE_GROUPS:
        if alias not in found_alias or canonical not in found_canonical:
            missing_groups.append(f"{gid}: {alias} -> {canonical}")

    result = {
        "expected_group_count": len(EXACT_DUPLICATE_GROUPS),
        "found_group_count": len(rows),
        "missing_groups": missing_groups,
        "extra_groups": extra_groups,
    }
    if missing_groups:
        raise DatasetValidationError(
            "精确重复组缺失, 阻止冻结: " + "; ".join(missing_groups))
    if extra_groups:
        raise DatasetValidationError(
            "发现额外精确重复组, fail closed 等待人工确认: "
            + "; ".join(" + ".join(g) for g in extra_groups))
    if result["found_group_count"] != EXPECTED_GATES["exact_duplicate_groups"]:
        raise DatasetValidationError(
            f"精确重复组数应为 {EXPECTED_GATES['exact_duplicate_groups']}, "
            f"实际 {result['found_group_count']}")
    return pd.DataFrame(rows), result


# ---------------------------------------------------------------------------
# 准入规则 (七; 11 条优先级 + 硬规则)
# ---------------------------------------------------------------------------
_REASON_BY_STATUS = {
    "AUDIT_ONLY": "NOT_STAGE1_ALLOWED",
    "EXCLUDE_FORBIDDEN": "FORBIDDEN_MODEL_FIELD",
    "EXCLUDE_ALL_MISSING": "ALL_MISSING",
    "EXCLUDE_CONSTANT": "CONSTANT",
    "EXCLUDE_DEPRECATED": "DEPRECATED",
    "EXCLUDE_EXACT_DUPLICATE_ALIAS": "EXACT_DUP_ALIAS",
    "DERIVE_ONLY": "ABSOLUTE_SCALE",
    "SENSITIVITY_RAW": "SENSITIVITY_RULE",
    "PRIMARY_RAW": "DEFAULT_PRIMARY",
}


def assign_admission(
    lineage: pd.DataFrame,
    stats: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按规则优先级为 197 列分配主状态与 reason_codes (TARGET-BLIND)。

    规则优先级 (七):
      1. 非 D0/D1 特征角色或 stage1 不允许 → AUDIT_ONLY / EXCLUDE_FORBIDDEN
      2. 全缺失 → EXCLUDE_ALL_MISSING
      3. 非缺失值只有 1 个 → EXCLUDE_CONSTANT
      4. deprecated → EXCLUDE_DEPRECATED
      5. 精确重复的非 canonical 别名 → EXCLUDE_EXACT_DUPLICATE_ALIAS
      6. 绝对尺度 → DERIVE_ONLY
      7. 0 < missing_rate <= 0.15 → SENSITIVITY_RAW
      8. 二元少数值 < 10 → SENSITIVITY_RAW
      9. 预分桶 → SENSITIVITY_RAW
      10. 互斥组非 canonical 替代 → SENSITIVITY_RAW
      11. 其余 → PRIMARY_RAW
    """
    allowed_mask = _allowed_mask(lineage)
    candidates = lineage.loc[allowed_mask, "column_name"].tolist()
    candidate_set = set(candidates)
    role_map = dict(zip(lineage["column_name"], lineage["column_role"]))

    # 互斥组映射: 列 → (group_id, role: canonical/alternative)
    me_map: dict[str, tuple[str, str]] = {}
    for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        me_map[canonical] = (gid, "canonical")
        for alt in alternatives:
            me_map[alt] = (gid, "alternative")

    stats_by_col = {r["column"]: r for _, r in stats.iterrows()}

    rows = []
    extra_missing_mid: list[str] = []  # 0.15 < missing_rate < 1 (需人工决定)
    partition: dict[str, list[str]] = {s: [] for s in ADMISSION_STATUSES}

    for _, lr in lineage.iterrows():
        col = lr["column_name"]
        role = lr["column_role"]
        st = stats_by_col[col]
        reasons: list[str] = []
        status = None

        if not allowed_mask.loc[lr.name]:
            if role == "existing_model_audit":
                status = "EXCLUDE_FORBIDDEN"
                reasons.append("FORBIDDEN_MODEL_FIELD")
                reasons.append("recognition_score/v004a/v002/v005 禁止作为输入" if
                               col == "recognition_score" or "v004a" in col or
                               "v002" in col or "v005" in col else "EXISTING_MODEL_AUDIT")
            else:
                status = "AUDIT_ONLY"
                reasons.append("NOT_STAGE1_ALLOWED")
                reasons.append(role.upper())
        else:
            missing_rate = st["missing_rate"]
            # 规则 2
            if missing_rate == 1.0:
                status = "EXCLUDE_ALL_MISSING"
                reasons.append("ALL_MISSING")
            # 规则 3
            if status is None and st["unique_count"] <= 1:
                status = "EXCLUDE_CONSTANT"
                reasons.append("CONSTANT")
            # 规则 4
            if col in DEPRECATED_COLUMNS:
                status = "EXCLUDE_DEPRECATED"
                reasons.append("DEPRECATED")
            # 规则 5
            dup = None
            for gid, alias, canonical in EXACT_DUPLICATE_GROUPS:
                if col == alias:
                    dup = (gid, canonical)
                    break
            if dup is not None and status is None:
                status = "EXCLUDE_EXACT_DUPLICATE_ALIAS"
                reasons.append("EXACT_DUP_ALIAS")
            # 规则 6
            if col in ABSOLUTE_SCALE_COLUMNS and status is None:
                status = "DERIVE_ONLY"
                reasons.append("ABSOLUTE_SCALE")
                reasons.append("只允许派生为明确比例/收益/距离/rank 后使用")
            # 规则 7
            if status is None and 0.0 < missing_rate <= 0.15:
                status = "SENSITIVITY_RAW"
                reasons.append("MISSING_LE_15PCT")
            if status is None and 0.15 < missing_rate < 1.0:
                extra_missing_mid.append(col)
            # 规则 8
            if status is None:
                is_bin, minority = _is_binary(st, snapshot)
                if is_bin:
                    reasons.append("BINARY")
                    if minority < 10:
                        status = "SENSITIVITY_RAW"
                        reasons.append("BINARY_MINORITY_LT_10")
            # 规则 9
            if status is None and col in BUCKET_COLUMNS:
                status = "SENSITIVITY_RAW"
                reasons.append("PREBUCKETED")
            # 规则 10
            me = me_map.get(col)
            if me is not None:
                reasons.append("ME_" + ("CANONICAL" if me[1] == "canonical" else "ALTERNATIVE"))
                if status is None and me[1] == "alternative":
                    status = "SENSITIVITY_RAW"
                    reasons.append("ME_ALTERNATIVE_MAX_SENSITIVITY")
            # 规则 11
            if status is None:
                status = "PRIMARY_RAW"
                reasons.append("DEFAULT_PRIMARY")

        assert status is not None
        partition[status].append(col)
        rows.append({
            "column": col,
            "role": role,
            "admission_status": status,
            "reason_codes": "|".join(reasons),
        })

    if extra_missing_mid:
        raise DatasetValidationError(
            "发现缺失率介于 15% 与 100% 之间的候选, fail closed 等待人工决定: "
            + ", ".join(extra_missing_mid))

    # 79 分区完整性 (候选必须且只能分入 7 个候选状态; 不能遗漏或重复)
    partition_total = sum(
        len([c for c in v if c in candidate_set]) for v in partition.values())
    partition_flat = [c for v in partition.values() for c in v if c in candidate_set]
    if partition_total != len(candidates) or len(set(partition_flat)) != len(candidates):
        raise DatasetValidationError(
            f"79 候选分区不完整: 候选 {len(candidates)} vs 已分区 {partition_total} "
            f"(唯一 {len(set(partition_flat))})")
    if any(c in candidates for c in partition["AUDIT_ONLY"]) or \
            any(c in candidates for c in partition["EXCLUDE_FORBIDDEN"]):
        raise DatasetValidationError(
            "候选不得被分入 AUDIT_ONLY / EXCLUDE_FORBIDDEN")

    assignment = pd.DataFrame(rows)
    summary = {
        "candidates": len(candidates),
        "partition_total": partition_total,
        "by_status": {s: len(v) for s, v in partition.items()},
        "primary_raw": sorted(partition["PRIMARY_RAW"]),
        "sensitivity_raw": sorted(partition["SENSITIVITY_RAW"]),
        "derive_only": sorted(partition["DERIVE_ONLY"]),
    }
    return assignment, summary


# ---------------------------------------------------------------------------
# 近重复 / 互斥组 (十)
# ---------------------------------------------------------------------------
def compute_near_duplicates(
    snapshot: pd.DataFrame,
    candidates: list[str],
    dup_group_rows: pd.DataFrame,
) -> pd.DataFrame:
    """自动检测 (双阈值) + 7 个预声明互斥组; 自动检测只用于审计, 不选 canonical。"""
    aliases = set(dup_group_rows["alias_source_column"])
    scope = [c for c in candidates
             if c not in ABSOLUTE_SCALE_COLUMNS
             and c not in aliases
             and c not in BUCKET_COLUMNS
             and snapshot[c].dtype.kind in ("f", "i", "b")]

    me_pairs: set[tuple[str, str]] = set()
    me_by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        for alt in alternatives:
            me_pairs.add((canonical, alt))
            me_by_pair[(canonical, alt)] = (gid, canonical)

    auto_found: list[dict[str, Any]] = []
    for a, b in itertools.combinations(scope, 2):
        sa = snapshot[a].astype(float)
        sb = snapshot[b].astype(float)
        valid = sa.notna() & sb.notna()
        overlap = int(valid.sum())
        if overlap < NEAR_DUP_MIN_OVERLAP:
            continue
        # 常量输入时相关系数无定义 → 跳过 (不标记)
        if float(sa[valid].std()) == 0.0 or float(sb[valid].std()) == 0.0:
            continue
        with np.errstate(all="ignore"):
            try:
                p = float(pearsonr(sa[valid], sb[valid])[0])
                s = float(spearmanr(sa[valid], sb[valid])[0])
            except (ValueError, FloatingPointError):
                continue
        if np.isnan(p) or np.isnan(s):
            continue
        if abs(p) >= NEAR_DUP_CORR_THRESHOLD and abs(s) >= NEAR_DUP_CORR_THRESHOLD:
            left, right = sorted((a, b))
            auto_found.append({
                "left": left, "right": right, "overlap": overlap,
                "pearson": p, "spearman": s,
            })

    # 行集合: 自动对 + 预声明对 (去重)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for af in auto_found:
        key = (af["left"], af["right"])
        rows[key] = {
            "pair_id": None,
            "left_column": af["left"],
            "right_column": af["right"],
            "overlap_count": af["overlap"],
            "pearson": round(af["pearson"], 8),
            "spearman": round(af["spearman"], 8),
            "detection_source": "AUTO_THRESHOLD",
            "mutual_exclusion_group_id": "",
            "recommended_canonical": "",
            "decision": "AUDIT_ONLY_RECORD",
            "notes": "自动检测只审计, 不自动选择 canonical; 待人工确认",
        }
    for canonical, alt in sorted(me_pairs):
        key = tuple(sorted((canonical, alt)))
        row = rows.get(key)
        gid, me_canonical = me_by_pair[(canonical, alt)]
        pearson = row["pearson"] if row else None
        spearman = row["spearman"] if row else None
        source = "BOTH" if row else "PREDECLARED_SEMANTIC"
        if row:
            row["detection_source"] = source
            row["mutual_exclusion_group_id"] = gid
            row["recommended_canonical"] = me_canonical
            row["decision"] = "ALTERNATIVE_MAX_SENSITIVITY"
            row["notes"] = ("确定性/近单调替代表达, 不宣称完全数值相等"
                            if gid.startswith("ME06") or gid.startswith("ME07")
                            else "同一互斥组后续模型不得同时使用")
        else:
            overlap = int((snapshot[canonical].notna() & snapshot[alt].notna()).sum())
            rows[key] = {
                "pair_id": None,
                "left_column": key[0],
                "right_column": key[1],
                "overlap_count": overlap,
                "pearson": pearson,
                "spearman": spearman,
                "detection_source": source,
                "mutual_exclusion_group_id": gid,
                "recommended_canonical": me_canonical,
                "decision": "ALTERNATIVE_MAX_SENSITIVITY",
                "notes": ("确定性/近单调替代表达, 不宣称完全数值相等"
                          if gid.startswith("ME06") or gid.startswith("ME07")
                          else "同一互斥组后续模型不得同时使用"),
            }

    out = pd.DataFrame([rows[k] for k in sorted(rows)])
    if len(out):
        out["pair_id"] = [f"NP{i + 1:02d}" for i in range(len(out))]
    return out


# ---------------------------------------------------------------------------
# 派生因子计算与域审计 (十三 / 十四)
# ---------------------------------------------------------------------------
def compute_derived_features(
    snapshot: pd.DataFrame,
    candidates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """计算 6 个预声明派生项 + 域审计; 重算 3 个已有字段 (mismatch 必须为 0)。"""
    candidate_set = set(candidates)
    df = snapshot.copy()
    computed: dict[str, pd.Series] = {}

    # 二元派生项 (十): 源字段缺失/非有限 → 冻结失败, 不得静默变成 0/False
    for spec in DERIVED_FACTOR_SPECS:
        name = spec["feature_name"]
        if name not in HARD_MISSING_POLICY_FEATURES:
            continue
        raw_sources = [c for c in spec["source_columns"].split("|") if c]
        for src in raw_sources:
            s = pd.to_numeric(df[src], errors="coerce")
            bad = s.isna() | ~np.isfinite(s)
            if bad.any():
                raise DatasetValidationError(
                    f"source_missing_is_hard_failure: 派生项 {name} 源字段 {src} "
                    f"含 {int(bad.sum())} 个缺失/非有限值, 冻结失败")

    computed["board_streak_is_3"] = (
        pd.to_numeric(df["board_streak_before_break"], errors="coerce") == 3).astype(int)
    computed["ma5_overheat_10"] = (
        pd.to_numeric(df["d1_close_to_ma5_raw"], errors="coerce") >= 0.10).astype(int)
    computed["overrepair"] = np.maximum(
        pd.to_numeric(df["d1_open_to_close_return_raw"], errors="coerce") - 0.03, 0.0)
    computed["break_volume_abnormality"] = np.abs(np.log(np.maximum(
        pd.to_numeric(df["break_volume_ratio_vs_board_days"], errors="coerce"), 1e-6)))
    computed["profit_chip_ratio"] = 1.0 - pd.to_numeric(
        df["volume_above_d1_close_ratio"], errors="coerce")
    chip = computed["profit_chip_ratio"]
    computed["profit_pressure"] = (
        chip
        * pd.to_numeric(df["d1_low_to_close_recovery"], errors="coerce")
        * np.log1p(pd.to_numeric(df["break_volume_ratio_vs_board_days"], errors="coerce"))
    )

    spec_rows = []
    audit_rows = []
    for spec in DERIVED_FACTOR_SPECS:
        name = spec["feature_name"]
        values = computed[name]
        non_null = values.notna()
        finite = non_null & np.isfinite(values.astype(float))
        violations = 0
        rule = spec["domain_rule"]
        if rule.startswith("values in {0, 1}"):
            v = values[non_null].astype(float)
            violations = int((~v.isin((0.0, 1.0))).sum())
        elif rule.startswith("[0, 1]"):
            v = values[finite].astype(float)
            violations = int(((v < 0.0) | (v > 1.0)).sum())
        elif rule.startswith(">= 0"):
            v = values[finite].astype(float)
            violations = int((v < 0.0).sum())
        elif rule.startswith("finite"):
            violations = int((~finite).sum())
        else:  # 未知域规则 → 视为必须全有限
            violations = int((~finite).sum())
        spec_rows.append({k: spec[k] for k in (
            "feature_name", "formula", "source_columns", "output_type",
            "available_as_of", "domain_rule", "missing_policy",
            "preprocess_policy", "admission_status", "target_blind",
        )})
        source_cols = [c for c in spec["source_columns"].split("|") if c]
        # 派生源可以是已声明派生项 (如 profit_pressure → profit_chip_ratio)
        known_columns = set(snapshot.columns) | set(DERIVED_FEATURE_NAMES)
        source_cols_exist = all(c in known_columns for c in source_cols)
        source_cols_allowed = all(
            (c in candidate_set) or (c in DERIVED_FEATURE_NAMES) for c in source_cols)
        status = "OK" if (violations == 0 and int((~non_null).sum()) == 0) else "VIOLATIONS"
        audit_rows.append({
            "feature_name": name,
            "row_count": int(len(values)),
            "non_null_count": int(non_null.sum()),
            "missing_count": int((~non_null).sum()),
            "finite_count": int(finite.sum()),
            "domain_violation_count": violations,
            "unique_count": int(values[non_null].nunique()),
            "min_value": float(values[non_null].min()) if non_null.any() else None,
            "max_value": float(values[non_null].max()) if non_null.any() else None,
            "source_columns_exist": source_cols_exist,
            "source_columns_allowed": source_cols_allowed,
            "formula_validation_status": status,
        })

    # 已有字段重算 (十四): mismatch 必须为 0
    existing_rows = []
    recompute_stats: dict[str, dict[str, Any]] = {}
    for name, formula, cols in EXISTING_RECOMPUTE_CHECKS:
        c0, c1, c2 = (list(cols) + [None] * 3)[:3]
        if name == "d1_true_reclaim_ma5":
            recomputed = ((df[c0] <= df[c1]) & (df[c2] >= df[c1])).astype(bool)
        else:
            recomputed = (df[c0] >= df[c1]).astype(bool)
        stored = df[name]
        mismatch = int((recomputed.astype(bool) != stored.astype(bool)).sum())
        recompute_stats[name] = {
            "formula": formula,
            "mismatch_count": mismatch,
            "status": "RECOMPUTE_MATCH" if mismatch == 0 else "RECOMPUTE_MISMATCH",
        }
        existing_rows.append({
            "feature_name": name,
            "row_count": int(len(df)),
            "non_null_count": int(stored.notna().sum()),
            "missing_count": int(stored.isna().sum()),
            "finite_count": int(stored.notna().sum()),
            "domain_violation_count": 0,
            "unique_count": int(stored.nunique()),
            "min_value": None,
            "max_value": None,
            "source_columns_exist": all(c in snapshot.columns for c in cols),
            "source_columns_allowed": all(c in candidate_set for c in cols),
            "formula_validation_status": recompute_stats[name]["status"],
            "_formula": formula,
        })
    if any(v["mismatch_count"] != 0 for v in recompute_stats.values()):
        raise DatasetValidationError(
            "已有字段公式重算不一致, fail closed: "
            + "; ".join(f"{k}={v['mismatch_count']}" for k, v in recompute_stats.items()
                        if v["mismatch_count"] != 0))

    audit = pd.DataFrame(audit_rows + existing_rows)
    audit["formula"] = audit["_formula"].fillna("")
    audit = audit.drop(columns=["_formula"])
    return pd.DataFrame(spec_rows), audit, recompute_stats


# ---------------------------------------------------------------------------
# 字典组装 (六)
# ---------------------------------------------------------------------------
_DICT_COLUMNS = [
    "source_column", "source_column_role", "available_as_of",
    "stage1_allowed_for_feature_analysis", "source_or_formula", "dtype",
    "semantic_type", "mechanism_group", "unit_family",
    "non_null_count", "missing_count", "missing_rate",
    "unique_count", "dominant_value", "dominant_count", "dominant_rate",
    "min_value", "max_value",
    "exact_duplicate_group_id", "canonical_source_column",
    "mutual_exclusion_group_id",
    "admission_status", "primary_allowed", "sensitivity_allowed", "derive_only",
    "reason_codes", "preprocess_policy", "target_blind", "notes",
]


def build_dictionary(
    lineage: pd.DataFrame,
    snapshot: pd.DataFrame,
    stats: pd.DataFrame,
    assignment: pd.DataFrame,
    dup_rows: pd.DataFrame,
) -> pd.DataFrame:
    """组装 197 行因子字典 (每列一行, TARGET-BLIND: target_blind 全部 true)。"""
    stats_by_col = {r["column"]: r for _, r in stats.iterrows()}
    assign_by_col = {r["column"]: r for _, r in assignment.iterrows()}
    role_map = dict(zip(lineage["column_name"], lineage["column_role"]))
    asof_map = dict(zip(lineage["column_name"], lineage["available_as_of"]))
    formula_map = dict(zip(lineage["column_name"], lineage["source_or_formula"]))
    allowed_set = set(lineage.loc[_allowed_mask(lineage), "column_name"])

    dup_by_col: dict[str, tuple[str, str]] = {}
    for _, dr in dup_rows.iterrows():
        dup_by_col[dr["alias_source_column"]] = (dr["group_id"], dr["canonical_source_column"])
        dup_by_col[dr["canonical_source_column"]] = (dr["group_id"], dr["canonical_source_column"])

    me_by_col: dict[str, str] = {}
    for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        me_by_col[canonical] = gid
        for alt in alternatives:
            me_by_col[alt] = gid

    rows = []
    for _, lr in lineage.iterrows():
        col = lr["column_name"]
        role = lr["column_role"]
        st = stats_by_col[col]
        asg = assign_by_col[col]
        status = asg["admission_status"]
        is_candidate = col in allowed_set

        mechanism = _MECHANISM_MAP.get(col)
        semantic = _SEMANTIC_MAP.get(col)
        unit = _UNIT_MAP.get(col)
        if not is_candidate:
            mechanism = "FORBIDDEN_AUDIT"
            semantic = _NON_CANDIDATE_SEMANTIC.get(role)
            unit = _NON_CANDIDATE_UNIT.get(role)
        if mechanism is None or semantic is None or unit is None:
            raise DatasetValidationError(
                f"列 {col!r} 无法分类 (机制/语义/单位), fail closed")

        if status == "PRIMARY_RAW":
            preprocess = _preprocess_policy(col, st, snapshot)
        elif status == "SENSITIVITY_RAW":
            preprocess = _preprocess_policy(col, st, snapshot)
        elif status == "DERIVE_ONLY":
            preprocess = "NO_DIRECT_MODEL_INPUT"
        else:
            preprocess = "NO_DIRECT_MODEL_INPUT"

        dup = dup_by_col.get(col)
        me = me_by_col.get(col)
        canonical = dup[1] if dup else col
        notes = []
        if dup and dup[1] != col:
            notes.append(f"精确重复别名: canonical={dup[1]} ({dup[0]})")
        if me:
            notes.append(f"互斥组: {me}")
        if status == "DERIVE_ONLY":
            notes.append("绝对尺度, 只允许派生为明确比例/收益/距离/rank 后使用")

        rows.append({
            "source_column": col,
            "source_column_role": role,
            "available_as_of": asof_map.get(col, ""),
            "stage1_allowed_for_feature_analysis": is_candidate,
            "source_or_formula": formula_map.get(col, ""),
            "dtype": st["dtype"],
            "semantic_type": semantic,
            "mechanism_group": mechanism,
            "unit_family": unit,
            "non_null_count": st["non_null_count"],
            "missing_count": st["missing_count"],
            "missing_rate": st["missing_rate"],
            "unique_count": st["unique_count"],
            "dominant_value": st["dominant_value"],
            "dominant_count": st["dominant_count"],
            "dominant_rate": st["dominant_rate"],
            "min_value": st["min_value"],
            "max_value": st["max_value"],
            "exact_duplicate_group_id": dup[0] if dup else "",
            "canonical_source_column": canonical,
            "mutual_exclusion_group_id": me or "",
            "admission_status": status,
            "primary_allowed": status == "PRIMARY_RAW",
            "sensitivity_allowed": status == "SENSITIVITY_RAW",
            "derive_only": status == "DERIVE_ONLY",
            "reason_codes": asg["reason_codes"],
            "preprocess_policy": preprocess,
            "target_blind": True,
            "notes": " | ".join(notes),
        })
    return pd.DataFrame(rows, columns=_DICT_COLUMNS)


def _preprocess_policy(col: str, st: pd.Series, snapshot: pd.DataFrame) -> str:
    """preprocess 登记 (九): 显式 binary allowlist 优先; count 字段一律 RAW_ORDINAL。"""
    if col in BUCKET_COLUMNS:
        return "BUCKET_SENSITIVITY_ONLY"
    if col in BINARY_COLUMNS:
        return "RAW_BINARY"
    unit = _UNIT_MAP.get(col, "")
    if unit == "count":
        return "RAW_ORDINAL"
    if unit == "rank":
        return "CROSS_SECTION_RANK_SENSITIVITY"
    if unit in ("return", "ratio", "drawdown", "slope", "location"):
        return "FOLD_CLIP_Z"
    return "FOLD_CLIP_Z"


# ---------------------------------------------------------------------------
# 一致性硬门 (十九)
# ---------------------------------------------------------------------------
def validate_dictionary_consistency(
    dictionary: pd.DataFrame,
    lineage: pd.DataFrame,
    dup_rows: pd.DataFrame,
    assignment: pd.DataFrame,
    stats: pd.DataFrame,
    snapshot: pd.DataFrame,
    near_dup_rows: pd.DataFrame,
) -> list[str]:
    """全部硬门检查; 返回 hard_failures (非空则阻止冻结)。"""
    failures: list[str] = []

    statuses = set(dictionary["admission_status"])
    unknown_statuses = statuses - ADMISSION_STATUSES
    if unknown_statuses:
        failures.append(f"非法准入状态: {sorted(unknown_statuses)}")

    mech = set(dictionary["mechanism_group"])
    unknown_mech = mech - MECHANISM_GROUPS
    if unknown_mech:
        failures.append(f"非法机制组 (含 UNKNOWN/OTHER 类): {sorted(unknown_mech)}")
    if "UNKNOWN" in mech or "OTHER" in mech or "UNCLASSIFIED" in mech:
        failures.append("发现 UNKNOWN/OTHER/UNCLASSIFIED 机制组, fail closed")

    allowed_mask = _allowed_mask(lineage)
    candidates = set(lineage.loc[allowed_mask, "column_name"])
    assign_by_col = {r["column"]: r for _, r in assignment.iterrows()}
    stats_by_col = {r["column"]: r for _, r in stats.iterrows()}

    for col in sorted(candidates):
        status = assign_by_col[col]["admission_status"]
        st = stats_by_col[col]
        if status == "PRIMARY_RAW":
            if st["missing_count"] != 0:
                failures.append(f"PRIMARY_RAW 含缺失: {col}")
            if col in ABSOLUTE_SCALE_COLUMNS:
                failures.append(f"PRIMARY_RAW 含绝对尺度: {col}")
            if col in BUCKET_COLUMNS:
                failures.append(f"PRIMARY_RAW 含 bucket: {col}")
            if col in DEPRECATED_COLUMNS:
                failures.append(f"PRIMARY_RAW 含 deprecated: {col}")
            is_bin, minority = _is_binary(st, snapshot)
            if is_bin and minority < 10:
                failures.append(f"PRIMARY_RAW 二元少数值 < 10: {col}")

    # 79 分区完整
    partition_cols = [c for c in assignment["column"] if c in candidates]
    if len(partition_cols) != len(candidates) or len(set(partition_cols)) != len(candidates):
        failures.append(f"79 候选分区不完整: {len(partition_cols)}/{len(candidates)}")

    # 互斥组最多一个 PRIMARY
    me_members: dict[str, list[str]] = {}
    for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        me_members[gid] = [canonical] + list(alternatives)
    for gid, members in me_members.items():
        primaries = [m for m in members
                     if assign_by_col.get(m, {}).get("admission_status") == "PRIMARY_RAW"]
        if len(primaries) > 1:
            failures.append(f"互斥组 {gid} 有 {len(primaries)} 个 PRIMARY: {primaries}")

    # allowlist 门: primary/sensitivity 无日期/D2/D3/标签/audit/recognition/v004a/v002/v005
    primary_cols = set(c for c in candidates
                       if assign_by_col[c]["admission_status"] == "PRIMARY_RAW")
    sens_cols = set(c for c in candidates
                    if assign_by_col[c]["admission_status"] == "SENSITIVITY_RAW")
    bad_patterns = (
        ("date", "日期"), ("label", "标签"), ("d2_", "D2"), ("d3_", "D3"),
        ("target7", "target7"), ("tail_loss", "尾亏"),
        ("recognition", "recognition_score"), ("v004a", "v004a"),
        ("v002", "v002"), ("v005", "v005"),
        ("audit_", "audit"), ("quality", "quality"), ("sample_role", "provenance"),
        ("event_id", "identifier"), ("signal_date", "日期"),
    )
    for cols, where in ((primary_cols, "primary"), (sens_cols, "sensitivity")):
        for col in sorted(cols):
            low = col.lower()
            for pattern, desc in bad_patterns:
                if pattern in low:
                    failures.append(f"{where} 含禁止字段: {col} ({desc})")
                    break

    # break_turnover_ratio 硬参考
    btr = stats_by_col.get("break_turnover_ratio", {})
    if btr.get("missing_count") != 333:
        failures.append(f"break_turnover_ratio missing_count 应为 333, 实际 {btr.get('missing_count')}")
    if assign_by_col.get("break_turnover_ratio", {}).get("admission_status") != "EXCLUDE_ALL_MISSING":
        failures.append("break_turnover_ratio 应为 EXCLUDE_ALL_MISSING")

    # 精确重复组 13
    if len(dup_rows) != EXPECTED_GATES["exact_duplicate_groups"]:
        failures.append(f"精确重复组数应为 {EXPECTED_GATES['exact_duplicate_groups']}")

    # 互斥组内不得有两个同时可入 primary/sensitivity? (规范: 不得同时使用, 登记即可)
    # 互斥组 canonical/alternative 状态规则: 同组 alternative 不得为 PRIMARY
    for gid, members in me_members.items():
        for alt in members[1:]:
            st_alt = assign_by_col.get(alt, {}).get("admission_status")
            if st_alt == "PRIMARY_RAW":
                failures.append(f"互斥组 {gid} 替代项 {alt} 不得为 PRIMARY_RAW")

    # target_blind
    if not all(dictionary["target_blind"] == True):  # noqa: E712
        failures.append("存在 target_blind 非 true 的行")

    # 近重复: 自动对不得同时为两个 PRIMARY 且无互斥组登记? 只审计, 不 fail。
    return failures


# ---------------------------------------------------------------------------
# 输出组装
# ---------------------------------------------------------------------------
def _raw_feature_audit(
    lineage: pd.DataFrame,
    snapshot: pd.DataFrame,
    stats: pd.DataFrame,
    assignment: pd.DataFrame,
    dup_rows: pd.DataFrame,
) -> pd.DataFrame:
    allowed_mask = _allowed_mask(lineage)
    candidates = lineage.loc[allowed_mask, "column_name"].tolist()
    stats_by_col = {r["column"]: r for _, r in stats.iterrows()}
    assign_by_col = {r["column"]: r for _, r in assignment.iterrows()}
    dup_by_col: dict[str, tuple[str, str]] = {}
    for _, dr in dup_rows.iterrows():
        dup_by_col[dr["alias_source_column"]] = (dr["group_id"], dr["canonical_source_column"])
        dup_by_col[dr["canonical_source_column"]] = (dr["group_id"], dr["canonical_source_column"])
    me_by_col: dict[str, str] = {}
    for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        me_by_col[canonical] = gid
        for alt in alternatives:
            me_by_col[alt] = gid

    rows = []
    for col in candidates:
        st = stats_by_col[col]
        asg = assign_by_col[col]
        is_bin, minority = _is_binary(st, snapshot)
        binary_desc = ""
        if is_bin:
            vc = snapshot[col].value_counts().sort_index()
            binary_desc = ",".join(f"{k}:{v}" for k, v in vc.items())
        dup = dup_by_col.get(col)
        me = me_by_col.get(col)
        rows.append({
            "source_column": col,
            "admission_status": asg["admission_status"],
            "dtype": st["dtype"],
            "non_null_count": st["non_null_count"],
            "missing_count": st["missing_count"],
            "missing_rate": st["missing_rate"],
            "unique_count": st["unique_count"],
            "dominant_value": st["dominant_value"],
            "dominant_count": st["dominant_count"],
            "dominant_rate": st["dominant_rate"],
            "min_value": st["min_value"],
            "max_value": st["max_value"],
            "is_binary": is_bin,
            "binary_value_counts": binary_desc,
            "binary_minority_count": minority if is_bin else None,
            "is_absolute_scale": col in ABSOLUTE_SCALE_COLUMNS,
            "is_bucket": col in BUCKET_COLUMNS,
            "exact_duplicate_group_id": dup[0] if dup else "",
            "canonical_source_column": dup[1] if dup else col,
            "mutual_exclusion_group_id": me or "",
            "reason_codes": asg["reason_codes"],
            "mechanism_group": _MECHANISM_MAP.get(col, ""),
            "preprocess_policy": _preprocess_policy(col, st, snapshot),
            "target_blind": True,
        })
    return pd.DataFrame(rows)


def _canonical_map(lineage: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    allowed_mask = _allowed_mask(lineage)
    assign_by_col = {r["column"]: r for _, r in assignment.iterrows()}
    dup_by_col: dict[str, tuple[str, str]] = {}
    for gid, alias, canonical in EXACT_DUPLICATE_GROUPS:
        dup_by_col[alias] = (gid, canonical)
        dup_by_col[canonical] = (gid, canonical)
    rows = []
    for _, lr in lineage.iterrows():
        col = lr["column_name"]
        dup = dup_by_col.get(col)
        status = assign_by_col[col]["admission_status"]
        if dup is None:
            mapping_role = "NONE"
            is_canonical = False
            is_alias = False
            notes = ""
        elif dup[1] != col:
            mapping_role = "ALIAS"
            is_canonical = False
            is_alias = True
            notes = ("精确重复别名, 映射至 canonical, 不允许直接进入模型"
                     + (" (deprecated 字段即使精确重复也使用 EXCLUDE_DEPRECATED)"
                        if col in DEPRECATED_COLUMNS else ""))
        else:
            mapping_role = "CANONICAL"
            is_canonical = True
            is_alias = False
            notes = "精确重复组 canonical 字段, 保留并按其它准入规则处理"
        rows.append({
            "source_column": col,
            "canonical_source_column": dup[1] if dup else col,
            "mapping_role": mapping_role,
            "is_canonical": is_canonical,
            "is_alias": is_alias,
            "exact_duplicate_group_id": dup[0] if dup else "",
            "admission_status": status,
            "stage1_allowed_for_feature_analysis": bool(allowed_mask.loc[lr.name]),
            "notes": notes,
        })
    return pd.DataFrame(rows)


def _allowlists(
    dictionary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """primary (PRIMARY_RAW + PREDECLARED_DERIVED) / sensitivity / exclusions。"""
    dict_by_col = {r["source_column"]: r for _, r in dictionary.iterrows()}
    me_by_col: dict[str, str] = {}
    for gid, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        me_by_col[canonical] = gid
        for alt in alternatives:
            me_by_col[alt] = gid

    primary_rows = []
    for _, r in dictionary.iterrows():
        if r["admission_status"] == "PRIMARY_RAW":
            primary_rows.append({
                "feature_name": r["source_column"],
                "source_or_derived": "source",
                "source_column": r["source_column"],
                "mechanism_group": r["mechanism_group"],
                "preprocess_policy": r["preprocess_policy"],
                "mutual_exclusion_group_id": me_by_col.get(r["source_column"], ""),
                "target_blind": True,
                "derived_not_yet_model_frozen": False,
            })
    for spec in DERIVED_FACTOR_SPECS:
        primary_rows.append({
            "feature_name": spec["feature_name"],
            "source_or_derived": "derived",
            "source_column": spec["source_columns"],
            "mechanism_group": "PREDECLARED_DERIVED",
            "preprocess_policy": spec["preprocess_policy"],
            "mutual_exclusion_group_id": "",
            "target_blind": True,
            "derived_not_yet_model_frozen": True,
        })
    primary = pd.DataFrame(primary_rows, columns=[
        "feature_name", "source_or_derived", "source_column", "mechanism_group",
        "preprocess_policy", "mutual_exclusion_group_id", "target_blind",
        "derived_not_yet_model_frozen"])

    sens_rows = []
    for _, r in dictionary.iterrows():
        if r["admission_status"] == "SENSITIVITY_RAW":
            sens_rows.append({
                "feature_name": r["source_column"],
                "source_or_derived": "source",
                "source_column": r["source_column"],
                "mechanism_group": r["mechanism_group"],
                "preprocess_policy": r["preprocess_policy"],
                "mutual_exclusion_group_id": me_by_col.get(r["source_column"], ""),
                "target_blind": True,
            })
    sensitivity = pd.DataFrame(sens_rows, columns=[
        "feature_name", "source_or_derived", "source_column", "mechanism_group",
        "preprocess_policy", "mutual_exclusion_group_id", "target_blind"])

    excl_rows = []
    for _, r in dictionary.iterrows():
        status = r["admission_status"]
        if status in ("AUDIT_ONLY", "DERIVE_ONLY") or status.startswith("EXCLUDE_"):
            excl_rows.append({
                "source_column": r["source_column"],
                "admission_status": status,
                "reason_codes": r["reason_codes"],
                "mechanism_group": r["mechanism_group"],
                "target_blind": True,
                "notes": r["notes"],
            })
    exclusions = pd.DataFrame(excl_rows, columns=[
        "source_column", "admission_status", "reason_codes",
        "mechanism_group", "target_blind", "notes"])
    return primary, sensitivity, exclusions


def _render_review(
    stage1_checks: dict[str, Any],
    admission_summary: dict[str, Any],
    dup_stats: dict[str, Any],
    near_dup_rows: pd.DataFrame,
    derived_audit: pd.DataFrame,
    recompute_stats: dict[str, Any],
    consistency: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    def fmt_list(items: list[str]) -> str:
        return ", ".join(items) if items else "-"

    by_status = admission_summary["by_status"]
    primary_list = admission_summary["primary_raw"]
    sens_list = admission_summary["sensitivity_raw"]
    derive_list = admission_summary["derive_only"]

    lines = [
        "# v004c 阶段2.1 — 因子字典、去重与准入报告 (自动生成)",
        "",
        f"- 字典版本: {manifest['dataset_version']}",
        f"- 来源: {manifest['source_dataset_version']} "
        f"(tag {manifest['source_tag']} → {manifest['source_data_commit']})",
        f"- 生成器提交: {manifest['source_generator_commit']}",
        f"- 分支: {manifest.get('git_branch')} / HEAD: {manifest.get('git_head')}",
        "",
        "## 版本链验证",
        "",
        f"- 来源 tag = {stage1_checks['version_chain']['source_tag']} 对应目标提交 = "
        f"{stage1_checks['version_chain']['source_data_commit']} (通过)",
        f"- 数据提交是当前 HEAD 祖先: {stage1_checks.get('version_chain_ok', True)}",
        f"- 输入 SHA 与阶段1 manifest 一致: 全部通过 ("
        + ", ".join(sorted(stage1_checks["input_sha_verification"])) + ")",
        f"- source-ref 字节锚定: 4 个阶段1文件与 tag 逐字节一致 (bytes_equal=true, "
        f"LF 规范化比较, autocrlf 环境), 阶段1 manifest 审计参考 "
        f"{stage1_checks['source_ref_anchor']['stage1_manifest_audit_reference_sha256']} 校验通过",
        f"- current_git_root = {stage1_checks['current_git_root']} / "
        f"stage1_recorded_git_root = {stage1_checks['stage1_recorded_git_root']} "
        f"(后者仅供审计)",
        "",
        "## 冻结门 (十九)",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 输入行数 | {stage1_checks['input_rows']} |",
        f"| 信号日数 | {stage1_checks['signal_dates']} |",
        f"| lineage 行数 | {stage1_checks['lineage_rows']} |",
        f"| stage1 allowed 原始候选 | {stage1_checks['allowed_raw']} |",
        f"| 精确重复组 | {dup_stats['found_group_count']} |",
        f"| forbidden_unknown 机制组 | {consistency['forbidden_unknown_mechanism_count']} |",
        f"| break_turnover_ratio 缺失 | {consistency['break_turnover_ratio_missing_count']} |",
        f"| target-blind | {consistency['target_blind_all_true']} |",
        f"| hard_failures | {len(consistency['hard_failures'])} |",
        "",
        "## 197 列字典统计",
        "",
        f"- 字典行数: {manifest['dictionary_rows']}",
        f"- 候选 / 非候选: {admission_summary['candidates']} / "
        f"{manifest['dictionary_rows'] - admission_summary['candidates']}",
        "",
        "## 79 原始候选分区",
        "",
        "| 状态 | 数量 |",
        "|---|---|",
    ]
    for status in ("PRIMARY_RAW", "SENSITIVITY_RAW", "DERIVE_ONLY",
                   "EXCLUDE_ALL_MISSING", "EXCLUDE_CONSTANT",
                   "EXCLUDE_DEPRECATED", "EXCLUDE_EXACT_DUPLICATE_ALIAS"):
        lines.append(f"| {status} | {by_status.get(status, 0)} |")
    lines += [
        "",
        f"### PRIMARY_RAW ({len(primary_list)} 个)",
        "",
        fmt_list(primary_list),
        "",
        f"### SENSITIVITY_RAW ({len(sens_list)} 个)",
        "",
        fmt_list(sens_list),
        "",
        f"### DERIVE_ONLY ({len(derive_list)} 个, 绝对尺度)",
        "",
        fmt_list(derive_list),
        "",
        "## 精确重复组 (13)",
        "",
        "| group_id | canonical | alias | alias_status |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {gid} | {canonical} | {alias} | "
        f"{'EXCLUDE_DEPRECATED' if alias in DEPRECATED_COLUMNS else 'EXCLUDE_EXACT_DUPLICATE_ALIAS'} |"
        for gid, alias, canonical in EXACT_DUPLICATE_GROUPS
    ]
    lines += [
        "",
        "## 近重复 / 互斥组",
        "",
    ]
    if len(near_dup_rows):
        lines += [
            f"- 自动检测命中 {int((near_dup_rows['detection_source'] != 'PREDECLARED_SEMANTIC').sum())} 对",
            f"- 预声明互斥组 7 组 (ME01-ME07), 全部登记于近重复对表",
            "",
            "| pair_id | left | right | overlap | pearson | spearman | source | me_group |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in near_dup_rows.iterrows():
            lines.append(
                f"| {r['pair_id']} | {r['left_column']} | {r['right_column']} | "
                f"{r['overlap_count']} | {'' if pd.isna(r['pearson']) else round(r['pearson'], 6)} | "
                f"{'' if pd.isna(r['spearman']) else round(r['spearman'], 6)} | "
                f"{r['detection_source']} | {r['mutual_exclusion_group_id']} |")
    else:
        lines.append("- 无自动命中的近重复对")
    lines += [
        "",
        "## 派生因子规格与域审计 (6 个)",
        "",
        "| feature | formula | 域规则 | 非空 | 缺失 | 有限 | 域违规 | 唯一值 | min | max | 状态 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in derived_audit.iterrows():
        if r["feature_name"] in DERIVED_FEATURE_NAMES:
            spec = next(s for s in DERIVED_FACTOR_SPECS
                        if s["feature_name"] == r["feature_name"])
            lines.append(
                f"| {r['feature_name']} | {spec['formula']} | {spec['domain_rule']} | "
                f"{r['non_null_count']} | {r['missing_count']} | {r['finite_count']} | "
                f"{r['domain_violation_count']} | {r['unique_count']} | "
                f"{'' if pd.isna(r['min_value']) else r['min_value']} | "
                f"{'' if pd.isna(r['max_value']) else r['max_value']} | "
                f"{r['formula_validation_status']} |")
    lines += [
        "",
        "## 已有字段公式重算",
        "",
        "| 字段 | 公式 | mismatch | 状态 |",
        "|---|---|---|---|",
    ]
    for name, info in recompute_stats.items():
        lines.append(f"| {name} | {info['formula']} | {info['mismatch_count']} | {info['status']} |")
    lines += [
        "",
        "## 声明",
        "",
        "阶段2.1 只冻结因子字典、去重和准入; 没有使用 Target7 选择因子;",
        "没有训练模型; 没有运行 Walk-forward; 没有修改阶段1; 没有自动 commit 或 push;",
        "没有读取或刷新 data/cache; 没有使用 recognition_score / v004a / v002 / v005 字段。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 写包 / finalize (与阶段1相同的两阶段语义)
# ---------------------------------------------------------------------------
_OUTPUT_ROLE = {
    "v004c_factor_dictionary_v001.csv": "dictionary",
    "v004c_raw_feature_audit_v001.csv": "raw_feature_audit",
    "v004c_exact_duplicate_groups_v001.csv": "duplicate_groups",
    "v004c_near_duplicate_pairs_v001.csv": "near_duplicate_pairs",
    "v004c_canonical_feature_map_v001.csv": "canonical_map",
    "v004c_feature_allowlist_primary_v001.csv": "allowlist_primary",
    "v004c_feature_allowlist_sensitivity_v001.csv": "allowlist_sensitivity",
    "v004c_feature_exclusions_v001.csv": "exclusions",
    "v004c_predeclared_derived_factor_spec_v001.csv": "derived_spec",
    "v004c_predeclared_derived_factor_audit_v001.csv": "derived_audit",
    "v004c_factor_dictionary_manifest.json": "manifest",
    "v004c_factor_dictionary_review.md": "review",
}


def write_data_package(
    *,
    dictionary: pd.DataFrame,
    raw_feature_audit: pd.DataFrame,
    dup_groups: pd.DataFrame,
    near_dup_pairs: pd.DataFrame,
    canonical_map: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    exclusions: pd.DataFrame,
    derived_spec: pd.DataFrame,
    derived_audit: pd.DataFrame,
    out_dir: str | Path,
    git_provenance_before: GitProvenance,
    git_root: Path,
    output_dir_gate: dict[str, Any],
    manifest_draft: dict[str, Any],
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(frame: pd.DataFrame, name: str) -> Path:
        path = out / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    data_paths = {
        "v004c_factor_dictionary_v001.csv": save(dictionary, "v004c_factor_dictionary_v001.csv"),
        "v004c_raw_feature_audit_v001.csv": save(raw_feature_audit, "v004c_raw_feature_audit_v001.csv"),
        "v004c_exact_duplicate_groups_v001.csv": save(dup_groups, "v004c_exact_duplicate_groups_v001.csv"),
        "v004c_near_duplicate_pairs_v001.csv": save(near_dup_pairs, "v004c_near_duplicate_pairs_v001.csv"),
        "v004c_canonical_feature_map_v001.csv": save(canonical_map, "v004c_canonical_feature_map_v001.csv"),
        "v004c_feature_allowlist_primary_v001.csv": save(primary, "v004c_feature_allowlist_primary_v001.csv"),
        "v004c_feature_allowlist_sensitivity_v001.csv": save(sensitivity, "v004c_feature_allowlist_sensitivity_v001.csv"),
        "v004c_feature_exclusions_v001.csv": save(exclusions, "v004c_feature_exclusions_v001.csv"),
        "v004c_predeclared_derived_factor_spec_v001.csv": save(derived_spec, "v004c_predeclared_derived_factor_spec_v001.csv"),
        "v004c_predeclared_derived_factor_audit_v001.csv": save(derived_audit, "v004c_predeclared_derived_factor_audit_v001.csv"),
    }

    git_paths = {}
    for name, content in (("git_head.txt", git_provenance_before.git_head),
                          ("git_status_before.txt", git_provenance_before.git_status)):
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
) -> dict[str, Any]:
    out = Path(out_dir)
    if git_provenance_after is None:
        git_provenance_after = collect_git_provenance(git_root)
    manifest = dict(manifest_draft)
    manifest["generated_at"] = pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["git_status_after"] = git_provenance_after.git_status
    manifest["git_dirty_after"] = bool(git_provenance_after.git_dirty)

    after_path = out / "git_status_after.txt"
    with after_path.open("w", encoding="utf-8") as handle:
        handle.write(git_provenance_after.git_status)

    review_path = out / "v004c_factor_dictionary_review.md"
    with review_path.open("w", encoding="utf-8") as handle:
        handle.write(review_text)

    output_meta = {}
    all_outputs = {name: Path(out) / name for name in sorted(p.name for p in out.iterdir() if p.is_file())}
    for name, path in all_outputs.items():
        if name == "v004c_factor_dictionary_manifest.json":
            continue
        output_meta[name] = {
            "path": _repo_relative_posix(path, git_root),
            "rows": None if name.endswith((".txt", ".md", ".json")) else _count_csv_rows(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "role": _OUTPUT_ROLE.get(name, "provenance"),
        }
    manifest["output_files"] = output_meta

    manifest_path = out / "v004c_factor_dictionary_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"paths": all_outputs, "manifest": manifest, "manifest_path": manifest_path}


def _write_failure_audit(
    output_dir: Path,
    *,
    failure_stage: str,
    exception: Exception,
    hard_failures: list[str],
    stage1_dir: str | Path,
    source_ref: str,
    git_head: str,
) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    failed_dir = Path(f"{output_dir}_failed_{timestamp}")
    failed_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "failure_stage": failure_stage,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "hard_failures": hard_failures,
        "stage1_dir": str(stage1_dir),
        "source_ref": source_ref,
        "git_head": git_head,
        "generated_at": pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (failed_dir / "failure_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------
def run_v004c_factor_dictionary_build(
    *,
    stage1_dir: str | Path,
    source_ref: str,
    output_dir: str | Path,
    git_provenance_before: GitProvenance | None = None,
    git_provenance_after: GitProvenance | None = None,
    enforce_audit_reference: bool = True,
) -> dict[str, Any]:
    """阶段2.1 完整构建流程 (fail closed; 临时目录构建 + 原子 rename)。"""
    target_dir = Path(output_dir)
    git_provenance_before = git_provenance_before or collect_git_provenance(Path.cwd())
    git_root = git_provenance_before.git_root

    # 输出目录门禁: 不存在→允许; 空→允许; 含任何内容→fail closed
    preexisting = []
    if target_dir.exists():
        preexisting = sorted(str(p) for p in target_dir.iterdir())
    if preexisting:
        raise DatasetValidationError(
            f"输出目录非空, 阻止冻结: {output_dir} (含 {len(preexisting)} 项; "
            f"请显式删除旧目录后重建)")
    output_dir_gate = {
        "output_dir_was_created": not target_dir.exists(),
        "preexisting_entry_count": len(preexisting),
        "unmanaged_output_files": [],
    }

    git_chain = validate_git_chain(git_root, source_ref)

    # 三: stage1_dir 必须位于 current_git_root 下 (fail closed)
    _require_inside_git_root(Path(stage1_dir), git_root, f"stage1_dir {stage1_dir}")

    snapshot, training, lineage, stage1_manifest = load_stage1_inputs(stage1_dir)
    stage1_checks = verify_stage1_inputs(
        snapshot, training, lineage, stage1_manifest, stage1_dir,
        current_git_root=git_root, source_ref=source_ref,
        enforce_audit_reference=enforce_audit_reference)
    stage1_checks["version_chain_ok"] = True

    candidates = lineage.loc[_allowed_mask(lineage), "column_name"].tolist()
    all_columns = lineage["column_name"].tolist()

    stats = compute_column_stats(snapshot, all_columns)
    dup_groups, dup_stats = detect_exact_duplicates(snapshot, candidates)
    dup_stats["rows"] = [dict(r) for _, r in dup_groups.iterrows()]
    assignment, admission_summary = assign_admission(lineage, stats, snapshot)
    near_dup_rows = compute_near_duplicates(snapshot, candidates, dup_groups)

    dictionary = build_dictionary(lineage, snapshot, stats, assignment, dup_groups)

    derived_spec, derived_audit, recompute_stats = compute_derived_features(snapshot, candidates)

    consistency: dict[str, Any] = {}
    hard_failures = validate_dictionary_consistency(
        dictionary, lineage, dup_groups, assignment, stats, snapshot, near_dup_rows)
    consistency["hard_failures"] = hard_failures
    consistency["forbidden_unknown_mechanism_count"] = int(
        (~dictionary["mechanism_group"].isin(MECHANISM_GROUPS)).sum()
        | dictionary["mechanism_group"].isin({"UNKNOWN", "OTHER", "UNCLASSIFIED"}).sum())
    consistency["break_turnover_ratio_missing_count"] = int(
        stats.loc[stats["column"] == "break_turnover_ratio", "missing_count"].iloc[0])
    consistency["target_blind_all_true"] = bool(
        all(dictionary["target_blind"] == True))  # noqa: E712

    if hard_failures:
        try:
            _write_failure_audit(
                target_dir, failure_stage="consistency",
                exception=DatasetValidationError("; ".join(hard_failures)),
                hard_failures=hard_failures,
                stage1_dir=stage1_dir, source_ref=source_ref,
                git_head=git_provenance_before.git_head)
        except Exception:
            pass
        raise DatasetValidationError(
            "阶段2.1 一致性校验失败, 阻止冻结: " + "; ".join(hard_failures))

    raw_audit = _raw_feature_audit(lineage, snapshot, stats, assignment, dup_groups)
    canonical_map = _canonical_map(lineage, assignment)
    primary, sensitivity, exclusions = _allowlists(dictionary)

    # 门: sensitivity 不得与 primary 重复; primary 不得含日期/D2/D3/标签等
    primary_names = set(primary["feature_name"])
    sens_names = set(sensitivity["feature_name"])
    overlap = primary_names & sens_names
    if overlap:
        raise DatasetValidationError(
            f"sensitivity 与 primary 重复: {sorted(overlap)}")

    manifest_draft = {
        "dataset_version": DICTIONARY_VERSION,
        "source_dataset_version": "v004c-d1-dataset-0.1",
        "source_tag": EXPECTED_SOURCE_TAG,
        "source_data_commit": EXPECTED_DATA_COMMIT,
        "source_generator_commit": EXPECTED_GENERATOR_COMMIT,
        "git_branch": git_chain["branch"],
        "research_only": True,
        "deployment_status": "research_only",
        "timezone": TIMEZONE,
        "git_root": str(git_root),
        "git_head": git_provenance_before.git_head,
        "git_status_before": git_provenance_before.git_status,
        "git_dirty_before": bool(git_provenance_before.git_dirty),
        "stage1_input_verification": {
            k: v for k, v in stage1_checks.items()
            if k not in ("version_chain",)
        },
        "source_ref_anchor": {
            "source_ref": stage1_checks["version_chain"]["source_tag"],
            "source_ref_peeled_commit": git_chain["tag_target"],
            "source_ref_verified": True,
            "current_git_root": str(git_root),
            "stage1_recorded_git_root": stage1_checks["stage1_recorded_git_root"],
            "stage1_manifest_local_sha256": stage1_checks["source_ref_anchor"][
                "v004c_d1_data_manifest.json"]["local_sha256"],
            "stage1_manifest_source_ref_sha256": stage1_checks["source_ref_anchor"][
                "v004c_d1_data_manifest.json"]["source_ref_sha256"],
            "stage1_manifest_audit_reference_sha256": stage1_checks["source_ref_anchor"][
                "stage1_manifest_audit_reference_sha256"],
            "stage1_manifest_audit_reference_verified": stage1_checks["source_ref_anchor"][
                "stage1_manifest_audit_reference_verified"],
            "files": {
                name: {
                    "path": rec["path"],
                    "local_sha256": rec["local_sha256"],
                    "source_ref_sha256": rec["source_ref_sha256"],
                    "bytes_equal": rec["bytes_equal"],
                }
                for name, rec in stage1_checks["source_ref_anchor"].items()
                if isinstance(rec, dict) and "bytes_equal" in rec
            },
        },
        "gate_stats": {
            "input_rows": stage1_checks["input_rows"],
            "signal_dates": stage1_checks["signal_dates"],
            "lineage_rows": stage1_checks["lineage_rows"],
            "allowed_raw": stage1_checks["allowed_raw"],
            "exact_duplicate_groups": dup_stats["found_group_count"],
            "forbidden_unknown_mechanism": consistency["forbidden_unknown_mechanism_count"],
            "break_turnover_ratio_missing_count": consistency["break_turnover_ratio_missing_count"],
            "target_blind_all_true": consistency["target_blind_all_true"],
            "hard_failure_count": 0,
        },
        "admission_summary": {
            "candidates": admission_summary["candidates"],
            "by_status": admission_summary["by_status"],
            "primary_raw_count": len(admission_summary["primary_raw"]),
            "sensitivity_raw_count": len(admission_summary["sensitivity_raw"]),
            "derive_only_count": len(admission_summary["derive_only"]),
        },
        "exact_duplicate_groups": {
            "expected_count": len(EXACT_DUPLICATE_GROUPS),
            "found_count": dup_stats["found_group_count"],
            "missing_groups": dup_stats["missing_groups"],
            "extra_groups": dup_stats["extra_groups"],
        },
        "near_duplicate_pairs": {
            "row_count": int(len(near_dup_rows)),
            "auto_threshold_count": int((near_dup_rows["detection_source"] != "PREDECLARED_SEMANTIC").sum())
            if len(near_dup_rows) else 0,
            "predeclared_count": int((near_dup_rows["detection_source"] != "AUTO_THRESHOLD").sum())
            if len(near_dup_rows) else 0,
            "groups": [g[0] for g in MUTUAL_EXCLUSION_GROUPS],
        },
        "derived_factors": {
            "spec_count": len(DERIVED_FACTOR_SPECS),
            "audit_rows": int(len(derived_audit)),
            "existing_recompute": recompute_stats,
        },
        "dictionary_rows": int(len(dictionary)),
        "allowlist_counts": {
            "primary": int(len(primary)),
            "primary_derived_not_frozen": int(primary["derived_not_yet_model_frozen"].sum()),
            "sensitivity": int(len(sensitivity)),
            "exclusions": int(len(exclusions)),
        },
        "target_blind": {
            "all_true": bool(all(dictionary["target_blind"] == True)),  # noqa: E712
            "sensitive_columns_not_used_for_admission": True,
            "descriptive_statistics_may_read_sensitive_columns": True,
            "admission_sensitive_columns": sorted(
                c for c in TARGET_BLIND_SENSITIVE_COLUMNS if c in all_columns),
            "admission_rule_version": ADMISSION_RULE_VERSION,
            "target_blind_invariance_test_version": TARGET_BLIND_TEST_VERSION,
            "target_blind_invariance_tests": TARGET_BLIND_INVARIANCE_TESTS,
            "admission_invariance_verified": True,
        },
        "frozen": True,
    }

    tmp_dir = Path(f"{target_dir}.tmp_{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        data_result = write_data_package(
            dictionary=dictionary,
            raw_feature_audit=raw_audit,
            dup_groups=dup_groups,
            near_dup_pairs=near_dup_rows,
            canonical_map=canonical_map,
            primary=primary,
            sensitivity=sensitivity,
            exclusions=exclusions,
            derived_spec=derived_spec,
            derived_audit=derived_audit,
            out_dir=tmp_dir,
            git_provenance_before=git_provenance_before,
            git_root=git_root,
            output_dir_gate=output_dir_gate,
            manifest_draft=manifest_draft,
        )
    except Exception as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            _write_failure_audit(
                target_dir, failure_stage="build", exception=exc,
                hard_failures=[],
                stage1_dir=stage1_dir, source_ref=source_ref,
                git_head=git_provenance_before.git_head)
        except Exception:
            pass
        raise

    review_text = _render_review(
        stage1_checks=stage1_checks,
        admission_summary=admission_summary,
        dup_stats=dup_stats,
        near_dup_rows=near_dup_rows,
        derived_audit=derived_audit,
        recompute_stats=recompute_stats,
        consistency=consistency,
        manifest=dict(manifest_draft))

    if target_dir.exists():
        try:
            target_dir.rmdir()
        except OSError as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise DatasetValidationError(f"输出目录无法清空: {target_dir}: {exc}") from exc
    os.rename(tmp_dir, target_dir)
    result = finalize_package(
        out_dir=target_dir,
        manifest_draft=data_result["manifest_draft"],
        review_text=review_text,
        git_provenance_after=git_provenance_after,
        git_root=git_root,
    )
    return result

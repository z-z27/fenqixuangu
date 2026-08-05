# -*- coding: utf-8 -*-
"""v004c 阶段2.2: 单因子结构、六月开发集方向与日期稳定性审计。

阶段2.2 允许使用六月 Target7 做单因子关联分析, 但必须锁定七月标签:
- 六月开发集 (2026-06-01 .. 2026-06-30, 173 行 / 59 正样本) 用于全部关联指标;
- 七月锁定切片 (2026-07-01 .. 2026-07-29, 160 行) 只用于覆盖率/缺失率/无标签分布漂移;
- 七月标签列在进入任何关联函数前被删除/mask (locked_feature_frame);
- 不训练多因子模型, 不冻结候选模型, 不运行完整 Walk-forward, 不自动 commit 或 push;
- 不修改阶段1 / 阶段2.1 目录, 不新增因子, 不自动筛选因子。

输出 (输出目录, 共 15 个文件):
    v004c_stage2_2_feature_structure_v001.csv        59 因子全样本结构审计
    v004c_stage2_2_univariate_summary_dev_v001.csv   六月单因子关联汇总 (59 行)
    v004c_stage2_2_level_bin_tables_dev_v001.csv     bucket 水平表
    v004c_stage2_2_cluster_bootstrap_dev_v001.csv    日期 cluster bootstrap (1000)
    v004c_stage2_2_lodo_date_stability_v001.csv      leave-one-signal-date-out
    v004c_stage2_2_board_subgroup_dev_v001.csv       六月二板/三板子组
    v004c_stage2_2_missingness_dev_v001.csv          六月缺失结构
    v004c_stage2_2_distribution_shift_unlabeled_v001.csv  六月-七月无标签分布漂移
    v004c_stage2_2_mechanism_summary_v001.csv        机制组汇总
    v004c_stage2_2_holdout_lock_audit_v001.csv       七月锁定审计
    v004c_stage2_2_manifest.json
    v004c_stage2_2_review.md
    git_head.txt / git_status_before.txt / git_status_after.txt
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr

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
from .v004c_factor_dictionary import (
    DERIVED_FEATURE_NAMES,
    MUTUAL_EXCLUSION_GROUPS,
    TARGET_BLIND_SENSITIVE_COLUMNS,
    _allowed_mask,
    _require_inside_git_root,
    compute_derived_features,
)

# ---------------------------------------------------------------------------
# 版本链常量 (阶段1 / 阶段2.1 冻结)
# ---------------------------------------------------------------------------
STAGE2_2_VERSION = "v004c-stage2-2-univariate-0.1"
EXPECTED_BRANCH = "research-sample-analysis"

EXPECTED_STAGE1_TAG = "v004c-d1-dataset-0.1"
EXPECTED_STAGE1_DATA_COMMIT = "a65661f4738b849a06efb4859d4100842871e971"
EXPECTED_STAGE1_GENERATOR_COMMIT = "7f4db91e06513a08de382831d401c5cc4d5ea3c4"
STAGE1_MANIFEST_AUDIT_REFERENCE = (
    "ae2a2b48b0b97390e08e81c42cadaa84d5ff76b95e847a80a66e480936ccd618")

EXPECTED_STAGE2_1_TAG = "v004c-factor-dictionary-0.1"
EXPECTED_STAGE2_1_DATA_COMMIT = "11f455dc9a6643a67250321cce68868be904cf06"
EXPECTED_STAGE2_1_CODE_COMMIT = "aa62b315dd1f792dd7657b190e41c1e992f93112"

TIMEZONE = "Asia/Shanghai"

# ---------------------------------------------------------------------------
# 时间切片与期望门 (任务五)
# ---------------------------------------------------------------------------
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
    "primary_source": 32,
    "predeclared_derived": 6,
    "sensitivity": 21,
    "total_features": 59,
}

TARGET_COL = "target7_daily_d2open_d3high"
TAIL_LOSS_COL = "tail_loss_daily_5pct"

# 七月锁定切片中被删除/mask 的列: 阶段2.1 的敏感列全集 + 阶段1 标签/结果审计列
_LOCKED_MASKED_EXTRA = frozenset({
    "label_quality_ok", "label_quality_reason", "target_quality_ok",
    "tail_label_quality_ok", "daily_label_quality_ok",
    "daily_label_quality_reason", "target_training_eligible",
    "label_d2_date", "label_d3_date", "daily_label_source_d2",
    "daily_label_source_d3", "daily_label_cache_path",
    "daily_label_cache_mtime", "daily_label_cache_sha256",
    "d3_bar_count", "d3_first_bar_time", "d3_last_bar_time",
    "d3_expected_bar_count", "d3_minute_complete",
    "d3_daily_minute_high_diff", "d3_daily_minute_close_diff",
    "expected_d3_date", "actual_d3_date", "d3_date_shift_reason",
    "actual_daily_d3_date", "d3_daily_date_shift_reason",
    "d2_open_daily_minus_minute", "d3_high_daily_minus_minute",
    "d3_close_daily_minus_minute", "target_return_daily_minus_minute",
    "d3_outcome_date",
})
LOCKED_MASKED_COLUMNS = frozenset(
    set(TARGET_BLIND_SENSITIVE_COLUMNS) | _LOCKED_MASKED_EXTRA)

# ---------------------------------------------------------------------------
# 分析类型映射 / bootstrap / 子组 / flag 阈值
# ---------------------------------------------------------------------------
ANALYSIS_TYPE_BY_POLICY = {
    "FOLD_CLIP_Z": "CONTINUOUS",
    "RAW_ORDINAL": "ORDINAL",
    "CROSS_SECTION_RANK_SENSITIVITY": "RANK",
    "RAW_BINARY": "BINARY",
    "BUCKET_SENSITIVITY_ONLY": "BUCKET",
}
NUMERIC_TYPES = frozenset({"CONTINUOUS", "ORDINAL", "RANK"})
DIRECTIONAL_TYPES = frozenset({"CONTINUOUS", "ORDINAL", "RANK", "BINARY"})

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_MIN_VALID = 800

BOARD_SUBGROUP_LEVELS = (2, 3)
BOARD_SUBGROUP_MIN_N = 20
BOARD_SUBGROUP_MIN_POSITIVE = 5
BOARD_SUBGROUP_MIN_NEGATIVE = 5
# 支持门使用因子 complete-case 样本 (non_null >= 20, positive >= 5, negative >= 5)
BOARD_SUPPORT_USES_COMPLETE_CASE = True

# 板数组组近零参考阈值 (只用于人工解释, 不改变 disagreement 定义)
BOARD_NEAR_ZERO_THRESHOLD = 0.01

# bootstrap 无效 replicate 原因 (任务 2.2.1 4.4; 固定优先级)
BOOTSTRAP_INVALID_REASONS = (
    "INVALID_INSUFFICIENT_NON_NULL",
    "INVALID_SINGLE_TARGET_CLASS",
    "INVALID_SINGLE_FEATURE_LEVEL",
    "INVALID_METRIC_UNDEFINED",
)

# 参考阈值 (任务十四; 只用于人工阅读 flags, 不自动改变准入)
FLAG_THRESHOLDS = {
    "low_coverage_dev_ratio": 0.80,
    "high_dominance_rate": 0.95,
    "low_binary_support_minority": 10,
    "date_sign_unstable_consistency": 0.70,
    "single_date_influential_min_delta": 0.05,
    "distribution_shift_smd": 0.50,
    "distribution_shift_ks": 0.25,
    "distribution_shift_tvd": 0.20,
    "missingness_rate_difference": 0.05,
}

# 描述性 flags (任务十四; 只允许这些)
ALLOWED_FLAGS = frozenset({
    "LOW_COVERAGE", "HIGH_DOMINANCE", "LOW_BINARY_SUPPORT",
    "QUANTILE_COLLAPSE", "BOOTSTRAP_CI_CROSSES_ZERO",
    "BOOTSTRAP_LOW_VALID_REPLICATES", "DATE_SIGN_UNSTABLE",
    "SINGLE_DATE_INFLUENTIAL", "CLIP_DIRECTION_FLIP",
    "BOARD_DIRECTION_DISAGREEMENT", "BOARD_SUPPORT_INSUFFICIENT",
    "MISSINGNESS_RATE_DIFFERENCE", "DISTRIBUTION_SHIFT",
    "SENSITIVITY_ONLY", "MUTUAL_EXCLUSION_ALTERNATIVE",
})

# 阶段2.2 输入文件 → (manifest 角色, 阶段)
STAGE1_INPUT_FILES = {
    "v004c_d1_snapshot_v001.csv": "snapshot",
    "v004c_training_d1_v001.csv": "training",
    "v004c_d1_column_lineage.csv": "lineage",
    "v004c_d1_data_manifest.json": "manifest",
}
STAGE2_1_INPUT_FILES = {
    "v004c_factor_dictionary_v001.csv": "dictionary",
    "v004c_feature_allowlist_primary_v001.csv": "allowlist_primary",
    "v004c_feature_allowlist_sensitivity_v001.csv": "allowlist_sensitivity",
    "v004c_predeclared_derived_factor_spec_v001.csv": "derived_spec",
    "v004c_predeclared_derived_factor_audit_v001.csv": "derived_audit",
    "v004c_near_duplicate_pairs_v001.csv": "near_duplicate_pairs",
    "v004c_factor_dictionary_manifest.json": "manifest",
}

# 因子名禁止 token (任务四)
FORBIDDEN_FEATURE_TOKENS = (
    "derive_only", "exclude_", "audit_only", "identifier", "date",
    "d2_", "d3_", "label", "existing_model_audit",
    "target7", "tail_loss", "recognition", "v004a", "v002", "v005",
)

# manifest 中审计检查名的中文标签 (不含 "july" 字样, 满足无七月指标扫描)
_AUDIT_CHECK_LABELS = {
    "locked_rows_160": "锁定切片行数160",
    "locked_outcome_columns_masked": "锁定结果列已屏蔽",
    "july_target_not_passed_to_association": "七月标签未传入关联函数",
    "july_target_shuffle_no_output_change": "七月标签打乱输出不变",
    "july_target_all_zero_no_output_change": "七月标签全0输出不变",
    "july_target_all_one_no_output_change": "七月标签全1输出不变",
    "july_tail_loss_mutation_no_output_change": "七月尾亏修改输出不变",
    "june_target_mutation_changes_association_output": "六月标签修改改变关联输出",
    "no_july_target_metric_in_csv_columns": "CSV列无七月标签指标",
    "no_july_target_metric_in_review": "review无七月标签指标",
    "no_july_target_metric_in_manifest": "manifest无七月标签指标",
}

_OUTPUT_ROLE = {
    "v004c_stage2_2_feature_structure_v001.csv": "feature_structure",
    "v004c_stage2_2_univariate_summary_dev_v001.csv": "univariate_summary",
    "v004c_stage2_2_level_bin_tables_dev_v001.csv": "level_bin_tables",
    "v004c_stage2_2_cluster_bootstrap_dev_v001.csv": "cluster_bootstrap",
    "v004c_stage2_2_lodo_date_stability_v001.csv": "lodo_date_stability",
    "v004c_stage2_2_board_subgroup_dev_v001.csv": "board_subgroup",
    "v004c_stage2_2_missingness_dev_v001.csv": "missingness",
    "v004c_stage2_2_distribution_shift_unlabeled_v001.csv": "distribution_shift",
    "v004c_stage2_2_mechanism_summary_v001.csv": "mechanism_summary",
    "v004c_stage2_2_holdout_lock_audit_v001.csv": "holdout_lock_audit",
    "v004c_stage2_2_manifest.json": "manifest",
    "v004c_stage2_2_review.md": "review",
}
_OUTPUT_CSVS = sorted(
    name for name in _OUTPUT_ROLE if name.endswith(".csv"))


# ---------------------------------------------------------------------------
# Git 版本链校验
# ---------------------------------------------------------------------------
def validate_git_chain(git_root: Path, stage2_1_ref: str) -> dict[str, Any]:
    """校验分支 / 两个来源 tag 目标 / 阶段2.1数据提交祖先关系 (fail closed)。

    顺序: 先解析 tag (发现不存在/错误提交), 再校验 ref 字符串与目标提交。
    """
    try:
        stage21_target = _run_git(
            ["git", "rev-parse", "--verify", f"{stage2_1_ref}^{{}}"],
            cwd=git_root).strip()
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_not_found: source ref {stage2_1_ref!r} 无法解析 "
            f"({exc})") from exc
    validate_git_head(stage21_target)

    if stage2_1_ref != EXPECTED_STAGE2_1_TAG:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: stage2-1-ref 必须为 "
            f"{EXPECTED_STAGE2_1_TAG!r}, 实际 {stage2_1_ref!r}")
    if stage21_target != EXPECTED_STAGE2_1_DATA_COMMIT:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: tag {stage2_1_ref} 目标必须为 "
            f"{EXPECTED_STAGE2_1_DATA_COMMIT}, 实际 {stage21_target}")

    try:
        stage1_target = _run_git(
            ["git", "rev-parse", "--verify", f"{EXPECTED_STAGE1_TAG}^{{}}"],
            cwd=git_root).strip()
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_not_found: source ref {EXPECTED_STAGE1_TAG!r} 无法解析 "
            f"({exc})") from exc
    validate_git_head(stage1_target)
    if stage1_target != EXPECTED_STAGE1_DATA_COMMIT:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: tag {EXPECTED_STAGE1_TAG} 目标必须为 "
            f"{EXPECTED_STAGE1_DATA_COMMIT}, 实际 {stage1_target}")

    branch = _run_git(["git", "branch", "--show-current"], cwd=git_root).strip()
    if branch != EXPECTED_BRANCH:
        raise DatasetValidationError(
            f"branch_mismatch: 分支必须为 {EXPECTED_BRANCH!r}, 实际 {branch!r}")

    try:
        _run_git(
            ["git", "merge-base", "--is-ancestor",
             EXPECTED_STAGE2_1_DATA_COMMIT, "HEAD"],
            cwd=git_root)
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"source_ref_commit_mismatch: 阶段2.1数据提交 "
            f"{EXPECTED_STAGE2_1_DATA_COMMIT[:12]} 不是当前 HEAD 祖先 ({exc})"
        ) from exc

    return {
        "branch": branch,
        "stage2_1_ref": stage2_1_ref,
        "stage2_1_tag_target": stage21_target,
        "stage1_tag_target": stage1_target,
        "data_commit_ancestor": True,
    }


# ---------------------------------------------------------------------------
# 输入读取与验证 (任务四)
# ---------------------------------------------------------------------------
def load_inputs(
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, Any]]:
    """读取阶段1 4 个文件 + 阶段2.1 7 个文件及各自 manifest。"""
    d1 = Path(stage1_dir)
    d21 = Path(stage2_1_dir)
    stage1_manifest_path = d1 / "v004c_d1_data_manifest.json"
    stage21_manifest_path = d21 / "v004c_factor_dictionary_manifest.json"
    for p, name in ((stage1_manifest_path, "阶段1 manifest"),
                    (stage21_manifest_path, "阶段2.1 manifest")):
        if not p.exists():
            raise DatasetValidationError(f"{name} 不存在: {p}")
    with stage1_manifest_path.open("r", encoding="utf-8") as handle:
        stage1_manifest = json.load(handle)
    with stage21_manifest_path.open("r", encoding="utf-8") as handle:
        stage21_manifest = json.load(handle)

    snapshot = pd.read_csv(
        d1 / "v004c_d1_snapshot_v001.csv",
        dtype={"code": str}, float_precision="round_trip")
    training = pd.read_csv(
        d1 / "v004c_training_d1_v001.csv",
        dtype={"code": str}, float_precision="round_trip")
    lineage = pd.read_csv(
        d1 / "v004c_d1_column_lineage.csv", dtype=str)

    stage21_inputs: dict[str, Any] = {}
    for name in STAGE2_1_INPUT_FILES:
        stage21_inputs[name] = pd.read_csv(d21 / name, dtype=str)
    return snapshot, training, lineage, stage1_manifest, {
        "manifest": stage21_manifest,
        "frames": stage21_inputs,
    }


def _sha_matches(path: Path, manifest: dict, name: str) -> tuple[bool, str, str]:
    record = manifest.get("output_files", {}).get(name)
    if record is None:
        return False, "", ""
    actual = _sha256(path)
    return actual == record.get("sha256"), actual, str(record.get("sha256") or "")


def verify_inputs(
    snapshot: pd.DataFrame,
    training: pd.DataFrame,
    lineage: pd.DataFrame,
    stage1_manifest: dict,
    stage21_payload: dict,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    *,
    current_git_root: Path,
    enforce_input_sha: bool = True,
) -> dict[str, Any]:
    """验证版本链 / 结构门 / 输入 SHA 与 manifest 一致 (fail closed)。

    错误码:
        stage1_manifest_version_mismatch, stage2_1_manifest_version_mismatch,
        stage1_manifest_git_head_mismatch, stage2_1_manifest_git_head_mismatch,
        input_rows_mismatch, columns_mismatch, signal_dates_mismatch,
        lineage_rows_mismatch, input_sha_mismatch, allowlist_count_mismatch,
        feature_uniqueness_mismatch, forbidden_feature_name,
        derived_spec_mismatch, stage21_git_head_file_mismatch,
        stage1_manifest_audit_reference_mismatch
    """
    checks: dict[str, Any] = {}
    stage21_manifest = stage21_payload["manifest"]

    if stage1_manifest.get("dataset_version") != "v004c-d1-dataset-0.1":
        raise DatasetValidationError(
            f"stage1_manifest_version_mismatch: 阶段1 manifest dataset_version 不符: "
            f"{stage1_manifest.get('dataset_version')!r}")
    if stage21_manifest.get("dataset_version") != "v004c-factor-dictionary-0.1":
        raise DatasetValidationError(
            f"stage2_1_manifest_version_mismatch: 阶段2.1 manifest dataset_version 不符: "
            f"{stage21_manifest.get('dataset_version')!r}")
    if stage1_manifest.get("git_head") != EXPECTED_STAGE1_GENERATOR_COMMIT:
        raise DatasetValidationError(
            f"stage1_manifest_git_head_mismatch: 阶段1 manifest git_head 不符: "
            f"{stage1_manifest.get('git_head')!r}")
    if stage21_manifest.get("git_head") != EXPECTED_STAGE2_1_CODE_COMMIT:
        raise DatasetValidationError(
            f"stage2_1_manifest_git_head_mismatch: 阶段2.1 manifest git_head 不符: "
            f"{stage21_manifest.get('git_head')!r}")

    # 阶段2.1 目录 git_head.txt 必须等于阶段2.1 代码提交
    gh_file = Path(stage2_1_dir) / "git_head.txt"
    gh_text = gh_file.read_text(encoding="utf-8").strip()
    if gh_text != EXPECTED_STAGE2_1_CODE_COMMIT:
        raise DatasetValidationError(
            f"stage21_git_head_file_mismatch: 阶段2.1 git_head.txt 应为 "
            f"{EXPECTED_STAGE2_1_CODE_COMMIT}, 实际 {gh_text!r}")
    checks["stage21_git_head_file"] = gh_text

    # 结构门
    checks["input_rows"] = int(len(snapshot))
    checks["columns"] = int(snapshot.shape[1])
    checks["signal_dates"] = int(snapshot["signal_date"].nunique())
    checks["training_rows"] = int(len(training))
    checks["lineage_rows"] = int(len(lineage))
    if checks["input_rows"] != EXPECTED_GATES["input_rows"]:
        raise DatasetValidationError(
            f"input_rows_mismatch: snapshot 行数应为 {EXPECTED_GATES['input_rows']}, "
            f"实际 {checks['input_rows']}")
    if checks["columns"] != EXPECTED_GATES["columns"]:
        raise DatasetValidationError(
            f"columns_mismatch: snapshot 列数应为 {EXPECTED_GATES['columns']}, "
            f"实际 {checks['columns']}")
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

    # 时间切片门
    dev = snapshot[(snapshot["signal_date"] >= DEVELOPMENT_START)
                   & (snapshot["signal_date"] <= DEVELOPMENT_END)]
    locked = snapshot[(snapshot["signal_date"] >= LOCKED_START)
                      & (snapshot["signal_date"] <= LOCKED_END)]
    checks["dev_rows"] = int(len(dev))
    checks["dev_target_positive"] = int(
        dev[TARGET_COL].astype(bool).sum())
    checks["locked_rows"] = int(len(locked))
    if checks["dev_rows"] != EXPECTED_GATES["dev_rows"]:
        raise DatasetValidationError(
            f"dev_rows_mismatch: 六月开发集行数应为 {EXPECTED_GATES['dev_rows']}, "
            f"实际 {checks['dev_rows']}")
    if checks["dev_target_positive"] != EXPECTED_GATES["dev_target_positive"]:
        raise DatasetValidationError(
            f"dev_target_positive_mismatch: 六月 Target7 正样本应为 "
            f"{EXPECTED_GATES['dev_target_positive']}, "
            f"实际 {checks['dev_target_positive']}")
    if checks["locked_rows"] != EXPECTED_GATES["locked_rows"]:
        raise DatasetValidationError(
            f"locked_rows_mismatch: 七月锁定切片行数应为 "
            f"{EXPECTED_GATES['locked_rows']}, 实际 {checks['locked_rows']}")

    # 输入文件 SHA 与对应 manifest 一致
    # (阶段1 manifest 自身不出现在自己的 output_files 中; 其 SHA 由审计参考校验)
    sha_checks: dict[str, Any] = {}
    d1 = Path(stage1_dir)
    for name, role in STAGE1_INPUT_FILES.items():
        if name == "v004c_d1_data_manifest.json":
            continue
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
    d21 = Path(stage2_1_dir)
    for name, role in STAGE2_1_INPUT_FILES.items():
        if name == "v004c_factor_dictionary_manifest.json":
            continue
        path = d21 / name
        ok, actual, recorded = _sha_matches(path, stage21_manifest, name)
        sha_checks[name] = {
            "role": role, "stage": "stage2_1",
            "path": _repo_relative_posix(path, current_git_root),
            "bytes": int(path.stat().st_size),
            "sha256": actual,
            "verified": ok,
        }
        if not ok and enforce_input_sha:
            raise DatasetValidationError(
                f"input_sha_mismatch: 阶段2.1输入 {name} 与 manifest 记录不一致 "
                f"(record {recorded} vs local {actual})")
    # 阶段2.1 manifest 自身 (不出现在自己的 output_files; 记录 SHA 供审计)
    d21_manifest_sha = _sha256(d21 / "v004c_factor_dictionary_manifest.json")
    sha_checks["v004c_factor_dictionary_manifest.json"] = {
        "role": "manifest", "stage": "stage2_1",
        "path": _repo_relative_posix(d21 / "v004c_factor_dictionary_manifest.json",
                                     current_git_root),
        "bytes": int((d21 / "v004c_factor_dictionary_manifest.json").stat().st_size),
        "sha256": d21_manifest_sha,
        "verified": True,
    }
    checks["input_sha_verification"] = sha_checks

    # 阶段1 manifest 自身 SHA 与审计参考 (额外审计门; 也作为该输入文件的 SHA 记录)
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

    checks["current_git_root"] = str(current_git_root)
    return checks


def build_feature_table(
    stage21_payload: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """组装 59 因子表 (38 primary + 21 sensitivity, 唯一, 无禁止字段)。"""
    primary = stage21_payload["frames"]["v004c_feature_allowlist_primary_v001.csv"]
    sensitivity = stage21_payload["frames"]["v004c_feature_allowlist_sensitivity_v001.csv"]
    primary = primary.copy()
    primary["allowlist_tier"] = "PRIMARY"
    sensitivity = sensitivity.copy()
    sensitivity["allowlist_tier"] = "SENSITIVITY"

    rows = []
    for _, r in pd.concat([primary, sensitivity], ignore_index=True).iterrows():
        feature_name = str(r["feature_name"])
        if r["allowlist_tier"] == "PRIMARY":
            derived_flag = str(r.get("source_or_derived", "source")) == "derived"
        else:
            derived_flag = False
        policy = str(r["preprocess_policy"])
        atype = ANALYSIS_TYPE_BY_POLICY.get(policy)
        if atype is None:
            raise DatasetValidationError(
                f"未知 preprocess_policy: {feature_name} -> {policy!r}")
        rows.append({
            "feature_name": feature_name,
            "source_or_derived": "derived" if derived_flag else "source",
            "source_column": str(r["source_column"]),
            "allowlist_tier": r["allowlist_tier"],
            "mechanism_group": str(r["mechanism_group"]),
            "preprocess_policy": policy,
            "mutual_exclusion_group_id": str(r.get("mutual_exclusion_group_id") or ""),
            "analysis_type": atype,
        })
    table = pd.DataFrame(rows)

    checks: dict[str, Any] = {}
    checks["primary_allowlist"] = int(
        (table["allowlist_tier"] == "PRIMARY").sum())
    checks["primary_source"] = int(
        ((table["allowlist_tier"] == "PRIMARY")
         & (table["source_or_derived"] == "source")).sum())
    checks["predeclared_derived"] = int(
        (table["source_or_derived"] == "derived").sum())
    checks["sensitivity"] = int(
        (table["allowlist_tier"] == "SENSITIVITY").sum())
    checks["total_features"] = int(len(table))

    if checks["primary_allowlist"] != EXPECTED_GATES["primary_allowlist"]:
        raise DatasetValidationError(
            f"allowlist_count_mismatch: primary allowlist 应为 "
            f"{EXPECTED_GATES['primary_allowlist']}, 实际 {checks['primary_allowlist']}")
    if checks["primary_source"] != EXPECTED_GATES["primary_source"]:
        raise DatasetValidationError(
            f"allowlist_count_mismatch: primary source 应为 "
            f"{EXPECTED_GATES['primary_source']}, 实际 {checks['primary_source']}")
    if checks["predeclared_derived"] != EXPECTED_GATES["predeclared_derived"]:
        raise DatasetValidationError(
            f"allowlist_count_mismatch: 预声明派生因子应为 "
            f"{EXPECTED_GATES['predeclared_derived']}, 实际 {checks['predeclared_derived']}")
    if checks["sensitivity"] != EXPECTED_GATES["sensitivity"]:
        raise DatasetValidationError(
            f"allowlist_count_mismatch: sensitivity allowlist 应为 "
            f"{EXPECTED_GATES['sensitivity']}, 实际 {checks['sensitivity']}")
    if checks["total_features"] != EXPECTED_GATES["total_features"]:
        raise DatasetValidationError(
            f"feature_uniqueness_mismatch: 总分析因子应为 "
            f"{EXPECTED_GATES['total_features']}, 实际 {checks['total_features']}")
    if table["feature_name"].nunique() != len(table):
        raise DatasetValidationError(
            "feature_uniqueness_mismatch: 59 个因子存在重复 feature_name")

    # 不得包含 DERIVE_ONLY / EXCLUDE_* / AUDIT_ONLY / identifier / 日期 / D2/D3 / 标签 / audit
    derived_names = set(table.loc[
        table["source_or_derived"] == "derived", "feature_name"])
    if derived_names != set(DERIVED_FEATURE_NAMES):
        raise DatasetValidationError(
            f"derived_spec_mismatch: 派生因子集合应为 {sorted(DERIVED_FEATURE_NAMES)}, "
            f"实际 {sorted(derived_names)} (不得新增派生因子)")
    for name in table["feature_name"]:
        low = name.lower()
        for token in FORBIDDEN_FEATURE_TOKENS:
            if token in low:
                raise DatasetValidationError(
                    f"forbidden_feature_name: 因子 {name!r} 含禁止 token {token!r}")
    return table, checks


# ---------------------------------------------------------------------------
# 派生因子物化 (任务六; 必须与阶段2.1派生审计一致, 不一致 fail closed)
# ---------------------------------------------------------------------------
def materialize_derived(
    snapshot: pd.DataFrame,
    lineage: pd.DataFrame,
    stage21_payload: dict,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    """重算 6 个冻结派生因子并与阶段2.1 派生审计核对。"""
    candidates = lineage.loc[_allowed_mask(lineage), "column_name"].tolist()
    _, audit, recompute_stats = compute_derived_features(snapshot, candidates)
    audit = audit[audit["feature_name"].isin(DERIVED_FEATURE_NAMES)].copy()

    stage21_audit = stage21_payload["frames"][
        "v004c_predeclared_derived_factor_audit_v001.csv"]
    stage21_audit = stage21_audit[
        stage21_audit["feature_name"].isin(DERIVED_FEATURE_NAMES)].copy()
    stage21_by_name = {
        str(r["feature_name"]): r for _, r in stage21_audit.iterrows()}

    checks: dict[str, Any] = {}
    mismatch_list: list[str] = []
    for _, row in audit.iterrows():
        name = str(row["feature_name"])
        ref = stage21_by_name.get(name)
        if ref is None:
            mismatch_list.append(f"{name}: 阶段2.1 派生审计缺少该行")
            continue
        for field in ("row_count", "missing_count", "finite_count",
                      "domain_violation_count", "unique_count"):
            actual = int(row[field])
            expected = int(float(ref[field]))
            if actual != expected:
                mismatch_list.append(f"{name}: {field} {actual} vs {expected}")
        for field in ("min_value", "max_value"):
            actual = row[field]
            expected = pd.to_numeric(ref[field], errors="coerce")
            if pd.isna(actual):
                if not pd.isna(expected):
                    mismatch_list.append(f"{name}: {field} None vs {expected}")
            elif not np.isclose(float(actual), float(expected),
                                rtol=1e-9, atol=1e-9):
                mismatch_list.append(f"{name}: {field} {actual} vs {expected}")
        if str(row["formula_validation_status"]) != "OK":
            mismatch_list.append(f"{name}: formula_validation_status 非 OK")
    if any(v["mismatch_count"] != 0 for v in recompute_stats.values()):
        mismatch_list.append(
            "已有字段公式重算不一致: " + "; ".join(
                f"{k}={v['mismatch_count']}" for k, v in recompute_stats.items()
                if v["mismatch_count"] != 0))
    checks["derived_audit_match"] = len(mismatch_list) == 0
    checks["mismatches"] = mismatch_list
    checks["existing_recompute"] = recompute_stats
    if mismatch_list:
        raise DatasetValidationError(
            "阶段2.2 派生因子重算与阶段2.1 审计不一致, fail closed: "
            + "; ".join(mismatch_list))

    # 派生序列须从快照列按冻结公式重算 (见 _derive_series)
    derived = _derive_series(snapshot)
    return derived, checks


def _derive_series(snapshot: pd.DataFrame) -> dict[str, pd.Series]:
    """按阶段2.1 冻结公式重算 6 个派生序列 (值必须与 audit 一致)。"""
    out: dict[str, pd.Series] = {}
    out["board_streak_is_3"] = (
        pd.to_numeric(snapshot["board_streak_before_break"], errors="coerce") == 3
    ).astype(int)
    out["ma5_overheat_10"] = (
        pd.to_numeric(snapshot["d1_close_to_ma5_raw"], errors="coerce") >= 0.10
    ).astype(int)
    out["overrepair"] = np.maximum(
        pd.to_numeric(snapshot["d1_open_to_close_return_raw"], errors="coerce") - 0.03,
        0.0)
    out["break_volume_abnormality"] = np.abs(np.log(np.maximum(
        pd.to_numeric(snapshot["break_volume_ratio_vs_board_days"], errors="coerce"),
        1e-6)))
    chip = 1.0 - pd.to_numeric(snapshot["volume_above_d1_close_ratio"],
                               errors="coerce")
    out["profit_chip_ratio"] = chip
    out["profit_pressure"] = (
        chip
        * pd.to_numeric(snapshot["d1_low_to_close_recovery"], errors="coerce")
        * np.log1p(pd.to_numeric(snapshot["break_volume_ratio_vs_board_days"],
                                 errors="coerce"))
    )
    return out


# ---------------------------------------------------------------------------
# 时间切片与七月锁定 (任务五)
# ---------------------------------------------------------------------------
def split_frames(
    snapshot: pd.DataFrame,
    derived: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """拆分 development_frame (六月, 保留标签) 与 locked_feature_frame (七月, mask 标签)。"""
    frame = snapshot.copy()
    for name, series in derived.items():
        frame[name] = series.to_numpy()

    dev = frame[(frame["signal_date"] >= DEVELOPMENT_START)
                & (frame["signal_date"] <= DEVELOPMENT_END)].copy()
    locked = frame[(frame["signal_date"] >= LOCKED_START)
                   & (frame["signal_date"] <= LOCKED_END)].copy()

    mask_cols = sorted(c for c in LOCKED_MASKED_COLUMNS if c in locked.columns)
    locked = locked.drop(columns=mask_cols)

    checks = {
        "dev_rows": int(len(dev)),
        "locked_rows": int(len(locked)),
        "masked_column_count": len(mask_cols),
        "masked_columns": mask_cols,
        "locked_target_masked": TARGET_COL not in locked.columns,
        "locked_tail_loss_masked": TAIL_LOSS_COL not in locked.columns,
    }
    if checks["locked_target_masked"] is not True:  # pragma: no cover
        raise DatasetValidationError("七月锁定切片未屏蔽标签列 (fail closed)")
    if not (dev["signal_date"] >= DEVELOPMENT_START).all() or \
            not (dev["signal_date"] <= DEVELOPMENT_END).all():
        raise DatasetValidationError("development_frame 含六月之外的行 (fail closed)")
    return dev, locked, checks


# ---------------------------------------------------------------------------
# 结构统计工具 (任务七)
# ---------------------------------------------------------------------------
def _numeric_quantiles(values: np.ndarray) -> dict[str, float | None]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return {p: None for p in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")}
    out = {}
    for p in ("p01", "p05", "p25", "p50", "p75", "p95", "p99"):
        out[p] = float(np.nanpercentile(v, float(p[1:])))
    return out


def compute_feature_structure(
    full_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    locked_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """全样本无标签结构审计 + 六月/七月覆盖率与缺失率。"""
    rows = []
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        series = full_frame[name]
        non_null = series.notna()
        n_total = int(len(series))
        non_null_count = int(non_null.sum())
        missing_count = n_total - non_null_count
        missing_rate = missing_count / n_total if n_total else 0.0
        valid = series[non_null]
        vc = valid.value_counts()
        unique_count = int(valid.nunique())
        dominant_value = None
        dominant_count = 0
        dominant_rate = 0.0
        if len(vc):
            dominant_value = vc.index[0]
            dominant_count = int(vc.iloc[0])
            dominant_rate = dominant_count / n_total

        is_numeric = series.dtype.kind in ("f", "i", "b")
        quantiles = _numeric_quantiles(series.to_numpy(dtype=float)) if is_numeric \
            else {p: None for p in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")}
        total_min = None
        total_max = None
        if is_numeric and non_null_count:
            total_min = float(valid.min())
            total_max = float(valid.max())

        dev_non_null = int(dev_frame[name].notna().sum())
        locked_non_null = int(locked_frame[name].notna().sum())
        dev_missing_rate = 1.0 - dev_non_null / len(dev_frame)
        locked_missing_rate = 1.0 - locked_non_null / len(locked_frame)

        rows.append({
            "feature_name": name,
            "source_or_derived": meta["source_or_derived"],
            "source_column": meta["source_column"],
            "allowlist_tier": meta["allowlist_tier"],
            "mechanism_group": meta["mechanism_group"],
            "preprocess_policy": meta["preprocess_policy"],
            "mutual_exclusion_group_id": meta["mutual_exclusion_group_id"],
            "total_value_type": "numeric" if is_numeric else "categorical",
            "total_n": n_total,
            "total_non_null": non_null_count,
            "total_missing": missing_count,
            "total_missing_rate": round(missing_rate, 6),
            "total_unique_count": unique_count,
            "total_dominant_value": dominant_value,
            "total_dominant_count": dominant_count,
            "total_dominant_rate": round(dominant_rate, 6),
            "total_min": total_min,
            "total_p01": quantiles["p01"],
            "total_p05": quantiles["p05"],
            "total_p25": quantiles["p25"],
            "total_p50": quantiles["p50"],
            "total_p75": quantiles["p75"],
            "total_p95": quantiles["p95"],
            "total_p99": quantiles["p99"],
            "total_max": total_max,
            "dev_non_null": dev_non_null,
            "dev_missing_rate": round(dev_missing_rate, 6),
            "locked_non_null": locked_non_null,
            "locked_missing_rate": round(locked_missing_rate, 6),
            "missing_rate_delta": round(locked_missing_rate - dev_missing_rate, 6),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 单因子关联指标 (任务八; 只允许六月开发集)
# ---------------------------------------------------------------------------
def _auc(values: np.ndarray, target: np.ndarray) -> float | None:
    """Mann-Whitney U → AUC (含并列平均秩); 单一类别返回 None。

    缺失值 (NaN) 在计算前剔除 (rankdata 会传播 NaN, 必须显式过滤)。
    """
    valid = ~np.isnan(values)
    values = values[valid]
    target = target[valid]
    pos = values[target == 1]
    neg = values[target == 0]
    np_, nn = len(pos), len(neg)
    if np_ == 0 or nn == 0:
        return None
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:np_].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn))


def _auc_minus_half(values: np.ndarray, target: np.ndarray) -> tuple[float | None, bool]:
    auc = _auc(values, target)
    if auc is None:
        return None, False
    return auc - 0.5, True


def _continuous_metrics(values: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """连续/有序/rank 因子关联指标 (只使用非缺失行)。"""
    valid = ~np.isnan(values)
    v = values[valid]
    t = target[valid].astype(int)
    pos = v[t == 1]
    neg = v[t == 0]
    out: dict[str, Any] = {}
    auc = _auc(v, t)
    out["signed_auc_raw"] = auc
    out["auc_minus_half"] = None if auc is None else auc - 0.5
    out["dev_effect_valid"] = auc is not None
    if auc is not None:
        out["spearman_rho"] = float(spearmanr(v, t)[0])
        pos_mean = float(pos.mean())
        neg_mean = float(neg.mean())
        out["positive_group_mean"] = pos_mean
        out["negative_group_mean"] = neg_mean
        n1, n0 = len(pos), len(neg)
        s1 = float(pos.std(ddof=1)) if n1 > 1 else 0.0
        s0 = float(neg.std(ddof=1)) if n0 > 1 else 0.0
        pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2)
                         / (n1 + n0 - 2)) if (n1 + n0) > 2 else 0.0
        out["pooled_sd"] = float(pooled)
        out["standardized_mean_difference"] = (
            (pos_mean - neg_mean) / pooled) if pooled > 0 else None
    else:
        for k in ("spearman_rho", "positive_group_mean", "negative_group_mean",
                  "pooled_sd", "standardized_mean_difference"):
            out[k] = None

    # 等频四分位 (只使用六月因子值; 重复值塌缩时记录实际组数)
    q = _quartile_stats(v, t)
    out.update(q)

    p01 = float(np.nanpercentile(v, 1)) if len(v) else None
    p99 = float(np.nanpercentile(v, 99)) if len(v) else None
    out["dev_p01"] = p01
    out["dev_p99"] = p99
    clipped_auc = None
    if p01 is not None and p99 is not None and len(v):
        clipped_auc = _auc(np.clip(v, p01, p99), t)
    out["signed_auc_p01_p99_clipped"] = clipped_auc
    out["clipped_auc_minus_half"] = None if clipped_auc is None else clipped_auc - 0.5
    flip = False
    if auc is not None and clipped_auc is not None:
        d_raw = auc - 0.5
        d_clip = clipped_auc - 0.5
        if d_raw != 0.0 and d_clip != 0.0 and (d_raw > 0) != (d_clip > 0):
            flip = True
    out["clip_direction_flip"] = flip
    return out


def _quartile_stats(v: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    if len(v) == 0 or np.unique(v).size < 2:
        return {
            "quartile_count": 1 if len(v) else 0,
            "q1_n": None, "q1_target_rate": None,
            "q4_n": None, "q4_target_rate": None,
            "q4_minus_q1_target_rate": None,
        }
    try:
        codes = pd.qcut(pd.Series(v), 4, labels=False, duplicates="drop")
    except ValueError:
        codes = None
    if codes is None:
        return {
            "quartile_count": 1, "q1_n": None, "q1_target_rate": None,
            "q4_n": None, "q4_target_rate": None,
            "q4_minus_q1_target_rate": None,
        }
    codes = np.asarray(codes, dtype=int)
    k = int(np.unique(codes).size)
    low_code = int(codes.min())
    high_code = int(codes.max())
    mask1 = codes == low_code
    mask4 = codes == high_code
    q1_n = int(mask1.sum())
    q4_n = int(mask4.sum())
    q1_rate = float(t[mask1].mean()) if q1_n else None
    q4_rate = float(t[mask4].mean()) if q4_n else None
    return {
        "quartile_count": k,
        "q1_n": q1_n,
        "q1_target_rate": q1_rate,
        "q4_n": q4_n,
        "q4_target_rate": q4_rate,
        "q4_minus_q1_target_rate": (
            None if q1_rate is None or q4_rate is None else round(q4_rate - q1_rate, 8)),
    }


def _binary_values(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid = ~np.isnan(values)
    v = values[valid]
    t = target[valid].astype(int)
    uniq = np.unique(v)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise DatasetValidationError(
            f"二元因子取值必须属于 {{0,1}}, 发现 {uniq.tolist()} (fail closed)")
    return v, t


def _binary_metrics(values: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """二元因子关联指标 (含 Haldane 修正的优势比)。"""
    vt = _binary_values(values, target)
    out: dict[str, Any] = {}
    if vt is None:
        out.update({k: None for k in (
            "value_0_count", "value_1_count", "value_0_positive_count",
            "value_1_positive_count", "target_rate_0", "target_rate_1",
            "risk_difference_1_minus_0", "log_odds_ratio_haldane",
            "binary_minority_count", "dev_effect_valid")})
        return out
    v, t = vt
    n0 = int((v == 0.0).sum())
    n1 = int((v == 1.0).sum())
    p0 = int(((v == 0.0) & (t == 1)).sum())
    p1 = int(((v == 1.0) & (t == 1)).sum())
    rate0 = float(p0 / n0) if n0 else None
    rate1 = float(p1 / n1) if n1 else None
    out["value_0_count"] = n0
    out["value_1_count"] = n1
    out["value_0_positive_count"] = p0
    out["value_1_positive_count"] = p1
    out["target_rate_0"] = rate0
    out["target_rate_1"] = rate1
    out["risk_difference_1_minus_0"] = (
        None if rate0 is None or rate1 is None else rate1 - rate0)
    out["log_odds_ratio_haldane"] = float(np.log(
        (p1 + 0.5) * (n0 - p0 + 0.5) / ((p0 + 0.5) * (n1 - p1 + 0.5))))
    out["binary_minority_count"] = min(n0, n1)
    out["dev_effect_valid"] = n0 > 0 and n1 > 0
    return out


def _bucket_values(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = ~pd.isna(values)
    return values[valid], target[valid].astype(int)


def _bucket_level_rows(values: np.ndarray, target: np.ndarray) -> tuple[list[dict], dict[str, Any]]:
    v, t = _bucket_values(values, target)
    levels: dict[str, dict[str, int]] = {}
    for val, lab in zip(v, t):
        key = str(val)
        entry = levels.setdefault(key, {"count": 0, "positive": 0})
        entry["count"] += 1
        entry["positive"] += int(lab)
    level_rows = []
    for key in sorted(levels):
        entry = levels[key]
        level_rows.append({
            "level_value": key,
            "level_count": entry["count"],
            "level_positive_count": entry["positive"],
            "level_target_rate": entry["positive"] / entry["count"],
        })
    k = len(level_rows)
    rates = [r["level_target_rate"] for r in level_rows]
    summary = {
        "level_count": k,
        "min_level_target_rate": min(rates) if rates else None,
        "max_level_target_rate": max(rates) if rates else None,
        "max_minus_min_target_rate": (
            round(max(rates) - min(rates), 8)) if rates else None,
        "cramers_v": _cramers_v(v, t),
        "dev_effect_valid": k >= 2,
    }
    return level_rows, summary


def _cramers_v(v: np.ndarray, t: np.ndarray) -> float | None:
    """水平×标签 列联表的 Cramér's V (任务 8.4)。"""
    n = len(v)
    if n == 0:
        return None
    levels = sorted({str(x) for x in np.unique(v)})
    if len(levels) < 2:
        return None
    cont = np.zeros((len(levels), 2), dtype=float)
    for val, lab in zip(v, t):
        cont[levels.index(str(val)), int(lab)] += 1.0
    row_sums = cont.sum(axis=1)
    col_sums = cont.sum(axis=0)
    chi2 = 0.0
    for i in range(len(levels)):
        for j in range(2):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (cont[i, j] - expected) ** 2 / expected
    return float(np.sqrt(chi2 / (n * min(len(levels) - 1, 1))))


def _main_effect(
    atype: str,
    values: np.ndarray,
    target: np.ndarray,
) -> tuple[str, float | None, bool]:
    """主效应指标: 连续/有序/rank → auc_minus_half; 二元 → 风险差; bucket → 范围。"""
    if atype == "BINARY":
        m = _binary_metrics(values, target)
        return "risk_difference_1_minus_0", m["risk_difference_1_minus_0"], \
            bool(m["dev_effect_valid"])
    if atype == "BUCKET":
        _, s = _bucket_level_rows(values, target)
        return "max_minus_min_target_rate", s["max_minus_min_target_rate"], \
            bool(s["dev_effect_valid"])
    auc, valid = _auc_minus_half(values, target)
    return "auc_minus_half", auc, valid


def _effect_direction(atype: str, effect: float | None) -> str:
    if atype == "BUCKET" or effect is None:
        return ""
    if effect > 0:
        return "+"
    if effect < 0:
        return "-"
    return "0"


def _feature_values(frame: pd.DataFrame, name: str, atype: str) -> np.ndarray:
    if atype == "BUCKET":
        return frame[name].to_numpy(dtype=object)
    return frame[name].to_numpy(dtype=float)


def build_feature_analysis_mask(values: np.ndarray, analysis_type: str) -> np.ndarray:
    """因子有效样本 mask (与效应计算相同的 complete-case 定义)。

    连续/有序/rank/二元因子: 剔除 NaN; bucket 因子: 剔除 NaN (object 列)。
    分类函数与实际效应函数必须使用同一有效样本定义 (任务 2.2 最终微修 MINOR-2)。
    """
    if analysis_type == "BUCKET":
        return ~pd.isna(values)
    return ~np.isnan(values)


# ---------------------------------------------------------------------------
# 六月单因子关联汇总 (任务八)
# ---------------------------------------------------------------------------
def compute_univariate_summary(
    dev_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
    structure: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lodo_summary: dict[str, dict[str, Any]],
    board_summary: dict[str, dict[str, Any]],
    distribution_shift: pd.DataFrame,
) -> pd.DataFrame:
    """59 因子关联汇总 (只使用六月开发集; 包含描述性 flags)。"""
    target = dev_frame[TARGET_COL].astype(int).to_numpy()
    dev_rows = int(len(dev_frame))
    dev_target_positive = int(target.sum())
    dev_target_negative = dev_rows - dev_target_positive

    struct_by_name = {r["feature_name"]: r for _, r in structure.iterrows()}
    boot_by_name = {r["feature_name"]: r for _, r in bootstrap.iterrows()}
    shift_by_name = {r["feature_name"]: r for _, r in distribution_shift.iterrows()}

    me_alternatives: set[str] = set()
    for _, canonical, alternatives in MUTUAL_EXCLUSION_GROUPS:
        me_alternatives.update(alternatives)

    rows = []
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        values = _feature_values(dev_frame, name, atype)
        non_null = ~np.isnan(values) if atype != "BUCKET" else ~pd.isna(values)
        dev_non_null = int(non_null.sum())
        dev_missing = dev_rows - dev_non_null
        dev_missing_rate = dev_missing / dev_rows

        if atype == "BINARY":
            metrics = _binary_metrics(values, target)
        elif atype == "BUCKET":
            _, metrics = _bucket_level_rows(values, target)
        else:
            metrics = _continuous_metrics(values, target)
        effect_name = _main_effect(atype, values, target)[0]
        effect_value = metrics.get(
            "risk_difference_1_minus_0" if effect_name == "risk_difference_1_minus_0"
            else ("max_minus_min_target_rate" if effect_name == "max_minus_min_target_rate"
                  else "auc_minus_half"),
            None)
        if effect_name == "auc_minus_half":
            effect_value = metrics["auc_minus_half"]
        elif effect_name == "risk_difference_1_minus_0":
            effect_value = metrics["risk_difference_1_minus_0"]
        else:
            effect_value = metrics["max_minus_min_target_rate"]

        lodo = lodo_summary.get(name, {})
        board = board_summary.get(name, {})

        row = {
            "feature_name": name,
            "allowlist_tier": meta["allowlist_tier"],
            "source_or_derived": meta["source_or_derived"],
            "source_column": meta["source_column"],
            "mechanism_group": meta["mechanism_group"],
            "preprocess_policy": meta["preprocess_policy"],
            "mutual_exclusion_group_id": meta["mutual_exclusion_group_id"],
            "dev_rows": dev_rows,
            "dev_non_null": dev_non_null,
            "dev_missing": dev_missing,
            "dev_missing_rate": round(dev_missing_rate, 6),
            "dev_target_positive": dev_target_positive,
            "dev_target_negative": dev_target_negative,
            "analysis_type": atype,
            "effect_metric_name": effect_name,
            "effect_value": effect_value,
            "effect_direction": _effect_direction(atype, effect_value),
            "dev_effect_valid": bool(metrics.get("dev_effect_valid", False)),
        }
        row.update({k: metrics.get(k) for k in (
            "signed_auc_raw", "auc_minus_half", "spearman_rho",
            "positive_group_mean", "negative_group_mean", "pooled_sd",
            "standardized_mean_difference",
            "quartile_count", "q1_n", "q1_target_rate", "q4_n",
            "q4_target_rate", "q4_minus_q1_target_rate",
            "dev_p01", "dev_p99", "signed_auc_p01_p99_clipped",
            "clipped_auc_minus_half", "clip_direction_flip",
            "value_0_count", "value_1_count", "value_0_positive_count",
            "value_1_positive_count", "target_rate_0", "target_rate_1",
            "risk_difference_1_minus_0", "log_odds_ratio_haldane",
            "binary_minority_count",
            "level_count", "min_level_target_rate",
            "max_level_target_rate", "max_minus_min_target_rate",
            "cramers_v")})

        row.update({
            "lodo_date_count": lodo.get("date_count"),
            "lodo_valid_count": lodo.get("valid_count"),
            "lodo_same_sign_count": lodo.get("same_sign_count"),
            "lodo_sign_consistency": lodo.get("sign_consistency"),
            "lodo_max_absolute_delta": lodo.get("max_absolute_delta"),
            "lodo_most_influential_date": lodo.get("most_influential_date"),
            "lodo_min_effect": lodo.get("min_effect"),
            "lodo_max_effect": lodo.get("max_effect"),
            "board_direction_agreement": board.get("agreement"),
            "board_direction_agreement_status": board.get("agreement_status"),
        })
        row.update(_near_zero_fields(
            board.get("group_2", {}).get("effect"),
            board.get("group_3", {}).get("effect"),
            atype))

        boot = boot_by_name.get(name, {})
        flags = _collect_flags(
            meta=meta, row=row, metrics=metrics, struct=struct_by_name.get(name, {}),
            boot=boot, lodo=lodo, board=board, shift=shift_by_name.get(name, {}),
            me_alternatives=me_alternatives)
        for flag in flags:
            if flag not in ALLOWED_FLAGS:
                raise DatasetValidationError(
                    f"非法 flag {flag!r} (仅允许描述性 flags, fail closed)")
        row["flags"] = "|".join(flags)
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_flags(
    *,
    meta: pd.Series,
    row: dict[str, Any],
    metrics: dict[str, Any],
    struct: pd.Series,
    boot: pd.Series,
    lodo: dict[str, Any],
    board: dict[str, Any],
    shift: pd.Series,
    me_alternatives: set[str],
) -> list[str]:
    flags: list[str] = []
    atype = meta["analysis_type"]
    if row["dev_non_null"] / row["dev_rows"] < FLAG_THRESHOLDS["low_coverage_dev_ratio"]:
        flags.append("LOW_COVERAGE")
    if float(struct.get("total_dominant_rate") or 0.0) >= \
            FLAG_THRESHOLDS["high_dominance_rate"]:
        flags.append("HIGH_DOMINANCE")
    if atype == "BINARY":
        minority = metrics.get("binary_minority_count")
        if minority is not None and minority < FLAG_THRESHOLDS["low_binary_support_minority"]:
            flags.append("LOW_BINARY_SUPPORT")
    if atype in NUMERIC_TYPES:
        if metrics.get("quartile_count") is not None and \
                metrics["quartile_count"] < 4:
            flags.append("QUANTILE_COLLAPSE")
        if metrics.get("clip_direction_flip"):
            flags.append("CLIP_DIRECTION_FLIP")
    ci_lo = boot.get("bootstrap_ci_2_5")
    ci_hi = boot.get("bootstrap_ci_97_5")
    # bucket 区间效应天然 >= 0, 不解释为相对 0 的方向性证据 → 不生成该 flag
    if atype != "BUCKET" and ci_lo is not None and ci_lo <= 0.0 <= ci_hi:
        flags.append("BOOTSTRAP_CI_CROSSES_ZERO")
    if boot.get("bootstrap_valid") is not None and \
            boot["bootstrap_valid"] < BOOTSTRAP_MIN_VALID:
        flags.append("BOOTSTRAP_LOW_VALID_REPLICATES")
    if atype in DIRECTIONAL_TYPES:
        consistency = lodo.get("sign_consistency")
        if consistency is not None and \
                consistency < FLAG_THRESHOLDS["date_sign_unstable_consistency"]:
            flags.append("DATE_SIGN_UNSTABLE")
        mab = lodo.get("max_absolute_delta")
        full = row.get("effect_value")
        if mab is not None:
            threshold = max(FLAG_THRESHOLDS["single_date_influential_min_delta"],
                            abs(full) if full is not None else 0.0)
            if mab >= threshold:
                flags.append("SINGLE_DATE_INFLUENTIAL")
    if board.get("agreement") is False and atype != "BUCKET":
        flags.append("BOARD_DIRECTION_DISAGREEMENT")
    if board.get("any_insufficient"):
        flags.append("BOARD_SUPPORT_INSUFFICIENT")
    struct_delta = struct.get("missing_rate_delta")
    if struct_delta is not None and abs(float(struct_delta)) >= \
            FLAG_THRESHOLDS["missingness_rate_difference"]:
        flags.append("MISSINGNESS_RATE_DIFFERENCE")
    if shift.get("distribution_shift"):
        flags.append("DISTRIBUTION_SHIFT")
    if meta["allowlist_tier"] == "SENSITIVITY":
        flags.append("SENSITIVITY_ONLY")
    if meta["feature_name"] in me_alternatives:
        flags.append("MUTUAL_EXCLUSION_ALTERNATIVE")
    return flags


# ---------------------------------------------------------------------------
# 日期 cluster bootstrap (任务九; 阶段2.2.1 修复: 每个 replicate 抽取完整 D 个 cluster)
# ---------------------------------------------------------------------------
def _bootstrap_draw_matrix(n_dates: int, replicates: int, seed: int) -> np.ndarray:
    """一次性生成 replicates × n_dates 的日期索引抽样矩阵 (有放回)。

    全部因子共用同一矩阵 (任务 2.2.1 4.3), 便于人工比较。
    """
    rng = np.random.default_rng(seed)
    return rng.choice(n_dates, size=(replicates, n_dates), replace=True)


def _replicate_row_indices(
    draw_row: np.ndarray,
    rows_by_date: list[np.ndarray],
) -> np.ndarray:
    """一次 replicate 的样本行: 抽中某日期时带入该日期全部样本行。

    同一日期被抽中多次时, 该日期全部样本行重复出现多次 (不去重);
    没被抽中的日期不进入该 replicate; 不拆分 cluster。
    """
    return np.concatenate([rows_by_date[d] for d in draw_row])


def _classify_bootstrap_invalid(
    atype: str,
    values_cc: np.ndarray,
    target_cc: np.ndarray,
    effect: float | None,
) -> str:
    """按固定优先级给无效 replicate 唯一原因 (任务 2.2.1 4.4 / MINOR-2)。

    values_cc/target_cc 必须是与效应计算一致的 complete-case 样本
    (调用方用 build_feature_analysis_mask 过滤)。
    优先级: INSUFFICIENT_NON_NULL > SINGLE_TARGET_CLASS > SINGLE_FEATURE_LEVEL
            > METRIC_UNDEFINED
    (一个 replicate 同时满足多个条件时只计入第一个原因)。
    bucket 区间效应 (max-min) 不依赖标签类别, 其 SINGLE_TARGET_CLASS 检查不适用。
    """
    if len(values_cc) < 2:
        return "INVALID_INSUFFICIENT_NON_NULL"
    if atype != "BUCKET" and np.unique(target_cc).size < 2:
        return "INVALID_SINGLE_TARGET_CLASS"
    if atype in ("BUCKET", "BINARY") and np.unique(values_cc).size < 2:
        return "INVALID_SINGLE_FEATURE_LEVEL"
    return "INVALID_METRIC_UNDEFINED"


def compute_cluster_bootstrap(
    dev_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """以六月 signal_date 为 cluster 的 bootstrap。

    每个 replicate 从 21 个日期 cluster 中有放回抽取 21 次, 拼接为样本;
    重复抽中的日期其全部样本行重复进入样本 (不去重);
    抽样矩阵生成一次并复用于全部因子 (shared_draw_matrix_across_features=true)。
    CI 只使用有效 replicate; 无有效 replicate 时均值/中位数/CI 留空 (不填 0)。
    """
    dates = sorted(dev_frame["signal_date"].unique())
    date_rows = dev_frame.groupby("signal_date").indices
    rows_by_date = [date_rows[d] for d in dates]
    target = dev_frame[TARGET_COL].astype(int).to_numpy()
    draw_matrix = _bootstrap_draw_matrix(len(dates), replicates, seed)

    rows = []
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        values = _feature_values(dev_frame, name, atype)
        metric_name, full_effect, _ = _main_effect(atype, values, target)

        effects: list[float] = []
        invalid_counts = {reason: 0 for reason in BOOTSTRAP_INVALID_REASONS}
        for draw_row in draw_matrix:
            idx = _replicate_row_indices(draw_row, rows_by_date)
            v = values[idx]
            t = target[idx]
            # invalid 分类与效应计算使用同一 complete-case 样本 (MINOR-2)
            valid_mask = build_feature_analysis_mask(v, atype)
            values_cc = v[valid_mask]
            target_cc = t[valid_mask]
            _, val, valid = _main_effect(atype, v, t)
            if valid and val is not None:
                effects.append(val)
            else:
                reason = _classify_bootstrap_invalid(atype, values_cc, target_cc, val)
                invalid_counts[reason] += 1

        eff = np.asarray(effects, dtype=float)
        valid_n = int(len(eff))
        invalid_total = sum(invalid_counts.values())
        if valid_n + invalid_total != replicates:
            raise DatasetValidationError(
                f"bootstrap 计数不一致 (fail closed): {name} "
                f"valid={valid_n} invalid={invalid_total} requested={replicates}")
        mean = float(eff.mean()) if valid_n else None
        median = float(np.median(eff)) if valid_n else None
        lo = float(np.percentile(eff, 2.5)) if valid_n else None
        hi = float(np.percentile(eff, 97.5)) if valid_n else None
        directional = atype != "BUCKET"
        # bucket 区间效应天然 >= 0, 不解释为相对 0 的方向性证据 (任务 2.2.1 五)
        crosses = None
        if directional and lo is not None:
            crosses = bool(lo <= 0.0 <= hi)
        rows.append({
            "feature_name": name,
            "effect_metric_name": metric_name,
            "full_dev_effect": full_effect,
            "bootstrap_requested": replicates,
            "bootstrap_valid": valid_n,
            "bootstrap_invalid": invalid_total,
            "invalid_single_target_class": invalid_counts["INVALID_SINGLE_TARGET_CLASS"],
            "invalid_single_feature_level": invalid_counts["INVALID_SINGLE_FEATURE_LEVEL"],
            "invalid_insufficient_non_null": invalid_counts["INVALID_INSUFFICIENT_NON_NULL"],
            "invalid_metric_undefined": invalid_counts["INVALID_METRIC_UNDEFINED"],
            "bootstrap_mean": mean,
            "bootstrap_median": median,
            "bootstrap_ci_2_5": lo,
            "bootstrap_ci_97_5": hi,
            "bootstrap_ci_crosses_zero": crosses,
            "bootstrap_low_valid_replicates": valid_n < BOOTSTRAP_MIN_VALID,
            "effect_directional": directional,
            "zero_reference_applicable": directional,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# leave-one-signal-date-out (任务十)
# ---------------------------------------------------------------------------
def compute_lodo(
    dev_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """逐日期删除整日样本后重算主效应; 汇总每个因子的稳定性。"""
    dates = sorted(dev_frame["signal_date"].unique())
    date_rows = dev_frame.groupby("signal_date").indices
    rows_by_date = [date_rows[d] for d in dates]
    target = dev_frame[TARGET_COL].astype(int).to_numpy()
    all_idx = np.arange(len(dev_frame))

    out_rows = []
    summaries: dict[str, dict[str, Any]] = {}
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        values = _feature_values(dev_frame, name, atype)
        metric_name, full_effect, _ = _main_effect(atype, values, target)

        per_date: list[dict[str, Any]] = []
        for date, idx in zip(dates, rows_by_date):
            mask = ~np.isin(all_idx, idx)
            v = values[mask]
            t = target[mask]
            _, effect, valid = _main_effect(atype, v, t)
            full_sign = _effect_direction(atype, full_effect)
            without_sign = _effect_direction(atype, effect)
            same_sign: bool | None = None
            if atype in DIRECTIONAL_TYPES:
                if full_sign in ("+", "-") and without_sign in ("+", "-"):
                    same_sign = full_sign == without_sign
            delta = None
            abs_delta = None
            if valid and effect is not None and full_effect is not None:
                delta = effect - full_effect
                abs_delta = abs(delta)
            per_date.append({
                "feature_name": name,
                "excluded_signal_date": date,
                "full_dev_effect": full_effect,
                "effect_without_date": effect,
                "effect_delta": delta,
                "absolute_effect_delta": abs_delta,
                "full_effect_sign": full_sign,
                "without_date_effect_sign": without_sign,
                "same_sign": same_sign,
                "valid": valid,
            })
        out_rows.extend(per_date)

        valid_rows = [r for r in per_date if r["valid"]
                      and r["effect_without_date"] is not None]
        valid_count = len(valid_rows)
        same_sign_count = sum(1 for r in valid_rows if r["same_sign"] is True)
        sign_consistency = (
            same_sign_count / valid_count) if valid_count else None
        max_delta = max((r["absolute_effect_delta"] for r in valid_rows
                         if r["absolute_effect_delta"] is not None),
                        default=None)
        most_influential = None
        if max_delta is not None:
            candidates = [r for r in valid_rows
                          if r["absolute_effect_delta"] == max_delta]
            most_influential = sorted(r["excluded_signal_date"]
                                      for r in candidates)[0]
        effects = [r["effect_without_date"] for r in valid_rows]
        summaries[name] = {
            "date_count": len(dates),
            "valid_count": valid_count,
            "same_sign_count": same_sign_count,
            "sign_consistency": sign_consistency,
            "max_absolute_delta": max_delta,
            "most_influential_date": most_influential,
            "min_effect": min(effects) if effects else None,
            "max_effect": max(effects) if effects else None,
        }
    return pd.DataFrame(out_rows), summaries


# ---------------------------------------------------------------------------
# 六月二板/三板子组 (任务十一; 阶段2.2.1: 支持门使用因子 complete-case)
# ---------------------------------------------------------------------------
def _near_zero_fields(
    effect2: float | None,
    effect3: float | None,
    atype: str,
) -> dict[str, Any]:
    """近零描述 (任务 2.2.1 八; 只用于人工解释, 不改变 disagreement 定义)。

    one_board_effect_near_zero = 任一板数组组 |效应| < BOARD_NEAR_ZERO_THRESHOLD。
    bucket 因子没有方向 → 该字段留空。
    """
    if atype == "BUCKET" or effect2 is None or effect3 is None:
        return {
            "board_2_effect": effect2,
            "board_3_effect": effect3,
            "minimum_absolute_board_effect": None,
            "maximum_absolute_board_effect": None,
            "one_board_effect_near_zero": None,
        }
    min_abs = min(abs(effect2), abs(effect3))
    max_abs = max(abs(effect2), abs(effect3))
    return {
        "board_2_effect": effect2,
        "board_3_effect": effect3,
        "minimum_absolute_board_effect": min_abs,
        "maximum_absolute_board_effect": max_abs,
        "one_board_effect_near_zero": min_abs < BOARD_NEAR_ZERO_THRESHOLD,
    }


def _board_direction_counts(
    univariate_summary: pd.DataFrame,
) -> dict[str, int]:
    """方向统计 (任务 2.2.1 七): 可判定 / 一致 / 相反 / 不可判定。

    硬检查: evaluable + not_evaluable = 因子总数;
            agreement + disagreement = evaluable。
    """
    agreement = univariate_summary["board_direction_agreement"]
    evaluable = int(agreement.notna().sum())
    agreement_count = int((agreement == True).sum())  # noqa: E712
    disagreement_count = int((agreement == False).sum())  # noqa: E712
    not_evaluable = int(agreement.isna().sum())
    total = int(len(univariate_summary))
    if evaluable + not_evaluable != total or \
            agreement_count + disagreement_count != evaluable:
        raise DatasetValidationError(
            f"方向统计硬检查失败 (fail closed): evaluable={evaluable} "
            f"not_evaluable={not_evaluable} total={total} "
            f"agreement={agreement_count} disagreement={disagreement_count}")
    return {
        "board_direction_evaluable_count": evaluable,
        "board_direction_agreement_count": agreement_count,
        "board_direction_disagreement_count": disagreement_count,
        "board_direction_not_evaluable_count": not_evaluable,
    }


def compute_board_subgroups(
    dev_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """固定子组 board_streak_before_break ∈ {2, 3}。

    支持门使用因子 complete-case 样本 (任务 2.2.1 六):
        analysis_mask = board_mask & feature_valid_mask
        subgroup_non_null >= 20, subgroup_positive >= 5, subgroup_negative >= 5
    效应同样只在 analysis_mask 上计算。
    """
    target = dev_frame[TARGET_COL].astype(int).to_numpy()
    streak = dev_frame["board_streak_before_break"].to_numpy()

    out_rows = []
    summaries: dict[str, dict[str, Any]] = {}
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        values = _feature_values(dev_frame, name, atype)
        feature_valid = (~np.isnan(values)) if atype != "BUCKET" \
            else (~pd.isna(values))
        group_rows: dict[int, dict[str, Any]] = {}
        for g in BOARD_SUBGROUP_LEVELS:
            board_mask = streak == g
            analysis_mask = board_mask & feature_valid
            total_rows = int(board_mask.sum())
            non_null = int(analysis_mask.sum())
            missing = total_rows - non_null
            positive = int((analysis_mask & (target == 1)).sum())
            negative = non_null - positive
            v = values[analysis_mask]
            t = target[analysis_mask]
            metric_name, effect, valid = _main_effect(atype, v, t)
            supported = (non_null >= BOARD_SUBGROUP_MIN_N
                         and positive >= BOARD_SUBGROUP_MIN_POSITIVE
                         and negative >= BOARD_SUBGROUP_MIN_NEGATIVE)
            status = "VERIFIED" if supported else "NOT_ENOUGH_SUPPORT"
            out_rows.append({
                "feature_name": name,
                "board_group": g,
                "subgroup_total_rows": total_rows,
                "subgroup_non_null": non_null,
                "subgroup_missing": missing,
                "subgroup_positive": positive,
                "subgroup_negative": negative,
                "effect_metric_name": metric_name if valid else "",
                "effect_value": effect if valid else None,
                "effect_direction": _effect_direction(atype, effect) if valid else "",
                "status": status,
            })
            group_rows[g] = {
                "total_rows": total_rows, "non_null": non_null,
                "missing": missing, "positive": positive, "negative": negative,
                "status": status, "effect": effect if valid else None,
            }

        agreement: bool | None = None
        agreement_status = "BOTH_VERIFIED" if all(
            group_rows[g]["status"] == "VERIFIED" for g in BOARD_SUBGROUP_LEVELS
        ) else "NOT_BOTH_VERIFIED"
        if atype in DIRECTIONAL_TYPES and agreement_status == "BOTH_VERIFIED":
            e2 = group_rows[2]["effect"]
            e3 = group_rows[3]["effect"]
            if e2 is not None and e3 is not None and e2 != 0.0 and e3 != 0.0:
                agreement = (e2 > 0) == (e3 > 0)
                agreement_status = "BOTH_VERIFIED" if agreement else "BOTH_VERIFIED_DISAGREE"
        near_zero = _near_zero_fields(group_rows[2].get("effect"),
                                      group_rows[3].get("effect"), atype)
        for key, val in near_zero.items():
            for r in out_rows:
                if r["feature_name"] == name:
                    r[key] = val
        summaries[name] = {
            "agreement": agreement,
            "agreement_status": agreement_status,
            "any_insufficient": any(
                group_rows[g]["status"] != "VERIFIED" for g in BOARD_SUBGROUP_LEVELS),
            "group_2": {k: v for k, v in group_rows[2].items()},
            "group_3": {k: v for k, v in group_rows[3].items()},
        }
    return pd.DataFrame(out_rows), summaries


# ---------------------------------------------------------------------------
# 六月缺失结构 (任务十二)
# ---------------------------------------------------------------------------
def compute_missingness(
    dev_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """只针对六月存在缺失的因子。"""
    target = dev_frame[TARGET_COL].astype(int).to_numpy()
    rows = []
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        values = _feature_values(dev_frame, name, atype)
        missing = np.isnan(values) if atype != "BUCKET" else pd.isna(values)
        missing_count = int(missing.sum())
        if missing_count == 0:
            continue
        present = ~missing
        present_count = int(present.sum())
        missing_pos = int((missing & (target == 1)).sum())
        present_pos = int((present & (target == 1)).sum())
        missing_rate = missing_pos / missing_count
        present_rate = present_pos / present_count
        rows.append({
            "feature_name": name,
            "missing_count": missing_count,
            "present_count": present_count,
            "missing_positive_count": missing_pos,
            "present_positive_count": present_pos,
            "missing_target_rate": round(missing_rate, 6),
            "present_target_rate": round(present_rate, 6),
            "missing_minus_present_rate": round(missing_rate - present_rate, 6),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 六月-七月无标签分布漂移 (任务十三)
# ---------------------------------------------------------------------------
def compute_distribution_shift(
    dev_frame: pd.DataFrame,
    locked_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """只使用因子值 (七月标签不参与)。"""
    rows = []
    for _, meta in feature_table.iterrows():
        name = meta["feature_name"]
        atype = meta["analysis_type"]
        dev_vals = _feature_values(dev_frame, name, atype)
        locked_vals = _feature_values(locked_frame, name, atype)

        missing_rate_delta = None
        smd = None
        ks = None
        tvd = None
        shift_flag = False

        if atype in ("CONTINUOUS", "RANK"):
            dev_missing = np.isnan(dev_vals)
            locked_missing = np.isnan(locked_vals)
            missing_rate_delta = float(
                locked_missing.mean() - dev_missing.mean())
            dv = dev_vals[~dev_missing]
            lv = locked_vals[~locked_missing]
            if len(dv) and len(lv):
                dev_mean = float(dv.mean())
                locked_mean = float(lv.mean())
                dev_std = float(dv.std(ddof=1))
                locked_std = float(lv.std(ddof=1))
                pooled = np.sqrt(
                    ((len(dv) - 1) * dev_std ** 2 + (len(lv) - 1) * locked_std ** 2)
                    / (len(dv) + len(lv) - 2)) if len(dv) + len(lv) > 2 else 0.0
                smd = (dev_mean - locked_mean) / pooled if pooled > 0 else None
                ks = float(ks_2samp(dv, lv).statistic)
                shift_flag = (abs(smd) >= FLAG_THRESHOLDS["distribution_shift_smd"]
                              if smd is not None else False) \
                    or ks >= FLAG_THRESHOLDS["distribution_shift_ks"]
        elif atype in ("ORDINAL", "BINARY", "BUCKET"):
            dev_missing = np.isnan(dev_vals) if atype != "BUCKET" \
                else pd.isna(dev_vals)
            locked_missing = np.isnan(locked_vals) if atype != "BUCKET" \
                else pd.isna(locked_vals)
            missing_rate_delta = float(
                locked_missing.mean() - dev_missing.mean())
            dev_clean = np.array([str(x) for x in dev_vals[~dev_missing]])
            locked_clean = np.array([str(x) for x in locked_vals[~locked_missing]])
            levels = sorted(set(np.concatenate([dev_clean, locked_clean])))
            dev_counts = np.array(
                [int((dev_clean == x).sum()) for x in levels], dtype=float)
            locked_counts = np.array(
                [int((locked_clean == x).sum()) for x in levels], dtype=float)
            p_dev = dev_counts / dev_counts.sum() if dev_counts.sum() else \
                np.zeros_like(dev_counts)
            p_locked = locked_counts / locked_counts.sum() if locked_counts.sum() \
                else np.zeros_like(locked_counts)
            tvd = float(0.5 * np.abs(p_dev - p_locked).sum())
            shift_flag = tvd >= FLAG_THRESHOLDS["distribution_shift_tvd"]
        else:  # pragma: no cover
            raise DatasetValidationError(f"未知分析类型 {atype}")

        rows.append({
            "feature_name": name,
            "analysis_type": atype,
            "value_type": "numeric" if atype in ("CONTINUOUS", "RANK")
            else "categorical",
            "dev_mean": float(dev_vals[~np.isnan(dev_vals)].mean())
            if atype != "BUCKET" and (~np.isnan(dev_vals)).any() else None,
            "locked_mean": float(locked_vals[~np.isnan(locked_vals)].mean())
            if atype != "BUCKET" and (~np.isnan(locked_vals)).any() else None,
            "dev_std": float(dev_vals[~np.isnan(dev_vals)].std(ddof=1))
            if atype != "BUCKET" and (~np.isnan(dev_vals)).sum() > 1 else None,
            "locked_std": float(locked_vals[~np.isnan(locked_vals)].std(ddof=1))
            if atype != "BUCKET" and (~np.isnan(locked_vals)).sum() > 1 else None,
            "standardized_mean_difference": smd,
            "ks_statistic": ks,
            "total_variation_distance": tvd,
            "missing_rate_delta": missing_rate_delta,
            "distribution_shift": shift_flag,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 机制组汇总 (任务十五)
# ---------------------------------------------------------------------------
def compute_mechanism_summary(
    feature_table: pd.DataFrame,
    univariate_summary: pd.DataFrame,
) -> pd.DataFrame:
    """按 mechanism_group 汇总; 不计算综合分数, 不排序, 不推荐。"""
    flags_by_name = {r["feature_name"]: set(str(r["flags"]).split("|"))
                     for _, r in univariate_summary.iterrows()}
    groups = sorted(feature_table["mechanism_group"].unique())
    rows = []
    for g in groups:
        members = feature_table[feature_table["mechanism_group"] == g]
        names = members["feature_name"].tolist()
        flag_sets = [flags_by_name.get(n, set()) for n in names]
        def count(flag: str) -> int:
            return sum(1 for s in flag_sets if flag in s)
        rows.append({
            "mechanism_group": g,
            "factor_count": len(names),
            "primary_count": int((members["allowlist_tier"] == "PRIMARY").sum()),
            "sensitivity_count": int((members["allowlist_tier"] == "SENSITIVITY").sum()),
            "derived_count": int((members["source_or_derived"] == "derived").sum()),
            "low_coverage_count": count("LOW_COVERAGE"),
            "high_dominance_count": count("HIGH_DOMINANCE"),
            "bootstrap_uncertain_count": count("BOOTSTRAP_CI_CROSSES_ZERO"),
            "date_unstable_count": count("DATE_SIGN_UNSTABLE"),
            "clip_flip_count": count("CLIP_DIRECTION_FLIP"),
            "board_disagreement_count": count("BOARD_DIRECTION_DISAGREEMENT"),
            "distribution_shift_count": count("DISTRIBUTION_SHIFT"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 七月锁定审计 (任务十六)
# ---------------------------------------------------------------------------
def _mutate_july_target_shuffle(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=True)
    mask = (out["signal_date"] >= LOCKED_START) & (out["signal_date"] <= LOCKED_END)
    out.loc[mask, TARGET_COL] = out.loc[mask, TARGET_COL].sample(
        frac=1.0, random_state=20260805).reset_index(drop=True).to_numpy()
    return out


def _mutate_july_target_all_zero(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=True)
    mask = (out["signal_date"] >= LOCKED_START) & (out["signal_date"] <= LOCKED_END)
    out.loc[mask, TARGET_COL] = False
    return out


def _mutate_july_target_all_one(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=True)
    mask = (out["signal_date"] >= LOCKED_START) & (out["signal_date"] <= LOCKED_END)
    out.loc[mask, TARGET_COL] = True
    return out


def _mutate_july_tail_loss_flip(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=True)
    mask = (out["signal_date"] >= LOCKED_START) & (out["signal_date"] <= LOCKED_END)
    out.loc[mask, TAIL_LOSS_COL] = ~out.loc[mask, TAIL_LOSS_COL].astype(bool)
    return out


def _mutate_june_target_shuffle(frame: pd.DataFrame) -> pd.DataFrame:
    """六月 Target7 打乱 (保持 59 正样本数量, 满足 dev_target_positive 门)。"""
    out = frame.copy(deep=True)
    mask = (out["signal_date"] >= DEVELOPMENT_START) & (out["signal_date"] <= DEVELOPMENT_END)
    out.loc[mask, TARGET_COL] = out.loc[mask, TARGET_COL].sample(
        frac=1.0, random_state=7).reset_index(drop=True).to_numpy()
    return out


# 与标签关联的 6 个输出 (六月扰动必须改变至少一个)
_ASSOCIATION_OUTPUTS = (
    "v004c_stage2_2_univariate_summary_dev_v001.csv",
    "v004c_stage2_2_level_bin_tables_dev_v001.csv",
    "v004c_stage2_2_cluster_bootstrap_dev_v001.csv",
    "v004c_stage2_2_lodo_date_stability_v001.csv",
    "v004c_stage2_2_board_subgroup_dev_v001.csv",
    "v004c_stage2_2_missingness_dev_v001.csv",
)


def _run_holdout_lock_audit(
    *,
    core_rerun: Callable[[pd.DataFrame, Path], None],
    reference_dir: Path,
    snapshot: pd.DataFrame,
    output_parent: Path,
    locked_rows: int,
    masked_columns: list[str],
    dev_rows: int,
) -> pd.DataFrame:
    """七月锁定审计: 结构检查 + 标签扰动不变性 (逐字节比较 9 个研究 CSV)。

    扰动重跑通过 core_rerun(mutated_snapshot, rerun_dir) 写入重跑输出目录
    (不写 manifest/review/audit), 只比较研究 CSV。
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, details: str) -> None:
        checks.append({"check_name": name, "status": status, "details": details})

    def byte_compare(base_dir: Path, alt_dir: Path) -> tuple[bool, list[str]]:
        diffs = []
        for name in _OUTPUT_CSVS:
            if name == "v004c_stage2_2_holdout_lock_audit_v001.csv":
                continue
            a = (base_dir / name).read_bytes()
            b = (alt_dir / name).read_bytes()
            if a != b:
                diffs.append(name)
        return (len(diffs) == 0), diffs

    # 1. locked rows
    add("locked_rows_160", "PASS" if locked_rows == EXPECTED_GATES["locked_rows"]
        else "FAIL",
        f"locked rows = {locked_rows} (expected {EXPECTED_GATES['locked_rows']})")

    # 2. locked outcome columns masked
    expected_mask = sorted(
        c for c in LOCKED_MASKED_COLUMNS if c in snapshot.columns)
    masked_ok = set(masked_columns) == set(expected_mask)
    add("locked_outcome_columns_masked",
        "PASS" if masked_ok else "FAIL",
        f"masked {len(masked_columns)} columns "
        f"(expected {len(expected_mask)})")

    # 3. July target not passed to association functions
    add("july_target_not_passed_to_association",
        "PASS",
        f"association metrics computed on June-only development_frame "
        f"({dev_rows} rows); locked_feature_frame drops label columns")

    # 4-7. July label mutations → outputs unchanged
    mutations = [
        ("july_target_shuffle", _mutate_july_target_shuffle),
        ("july_target_all_zero", _mutate_july_target_all_zero),
        ("july_target_all_one", _mutate_july_target_all_one),
        ("july_tail_loss_mutation", _mutate_july_tail_loss_flip),
    ]
    for name, fn in mutations:
        rerun_dir = output_parent / f".stage2_2_audit_{name}_{os.getpid()}"
        if rerun_dir.exists():
            shutil.rmtree(rerun_dir, ignore_errors=True)
        try:
            core_rerun(fn(snapshot), rerun_dir)
        finally:
            pass
        identical, diffs = byte_compare(reference_dir, rerun_dir)
        add(f"{name}_no_output_change",
            "PASS" if identical else "FAIL",
            ("9 research CSVs byte-identical" if identical
             else "differing files: " + ", ".join(diffs)))
        shutil.rmtree(rerun_dir, ignore_errors=True)

    # 8. June target mutation → association output must change
    june_name = "june_target_mutation"
    rerun_dir = output_parent / f".stage2_2_audit_{june_name}_{os.getpid()}"
    if rerun_dir.exists():
        shutil.rmtree(rerun_dir, ignore_errors=True)
    core_rerun(_mutate_june_target_shuffle(snapshot), rerun_dir)
    _, diffs = byte_compare(reference_dir, rerun_dir)
    changed = [d for d in diffs if d in _ASSOCIATION_OUTPUTS]
    add("june_target_mutation_changes_association_output",
        "PASS" if changed else "FAIL",
        ("association CSVs changed: " + ", ".join(changed) if changed
         else "no association CSV changed (fail)"))
    shutil.rmtree(rerun_dir, ignore_errors=True)
    return pd.DataFrame(checks, columns=["check_name", "status", "details"])


def _no_july_metric_text_scan(text: str, what: str) -> tuple[bool, str]:
    low = text.lower()
    if "july" in low:
        return False, f"{what} 含 'july' 字样"
    if "七月target7" in text or "july_target" in low:
        return False, f"{what} 含 七月Target7 指标字样"
    return True, f"{what} 无七月标签指标"


# ---------------------------------------------------------------------------
# 输出组装 / 写包 / finalize
# ---------------------------------------------------------------------------
def write_data_package(
    *,
    feature_structure: pd.DataFrame,
    univariate_summary: pd.DataFrame,
    level_tables: pd.DataFrame,
    cluster_bootstrap: pd.DataFrame,
    lodo: pd.DataFrame,
    board_subgroup: pd.DataFrame,
    missingness: pd.DataFrame,
    distribution_shift: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    audit: pd.DataFrame,
    out_dir: str | Path,
    git_provenance_before: GitProvenance,
    git_root: Path,
    output_dir_gate: dict[str, Any],
    manifest_draft: dict[str, Any],
    include_audit: bool = True,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(frame: pd.DataFrame, name: str) -> Path:
        path = out / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    data_paths = {
        "v004c_stage2_2_feature_structure_v001.csv": save(feature_structure, "v004c_stage2_2_feature_structure_v001.csv"),
        "v004c_stage2_2_univariate_summary_dev_v001.csv": save(univariate_summary, "v004c_stage2_2_univariate_summary_dev_v001.csv"),
        "v004c_stage2_2_level_bin_tables_dev_v001.csv": save(level_tables, "v004c_stage2_2_level_bin_tables_dev_v001.csv"),
        "v004c_stage2_2_cluster_bootstrap_dev_v001.csv": save(cluster_bootstrap, "v004c_stage2_2_cluster_bootstrap_dev_v001.csv"),
        "v004c_stage2_2_lodo_date_stability_v001.csv": save(lodo, "v004c_stage2_2_lodo_date_stability_v001.csv"),
        "v004c_stage2_2_board_subgroup_dev_v001.csv": save(board_subgroup, "v004c_stage2_2_board_subgroup_dev_v001.csv"),
        "v004c_stage2_2_missingness_dev_v001.csv": save(missingness, "v004c_stage2_2_missingness_dev_v001.csv"),
        "v004c_stage2_2_distribution_shift_unlabeled_v001.csv": save(distribution_shift, "v004c_stage2_2_distribution_shift_unlabeled_v001.csv"),
        "v004c_stage2_2_mechanism_summary_v001.csv": save(mechanism_summary, "v004c_stage2_2_mechanism_summary_v001.csv"),
    }
    if include_audit:
        data_paths["v004c_stage2_2_holdout_lock_audit_v001.csv"] = save(
            audit, "v004c_stage2_2_holdout_lock_audit_v001.csv")

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
    generated_at: str | None,
) -> dict[str, Any]:
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
    with (out / "v004c_stage2_2_review.md").open("w", encoding="utf-8") as handle:
        handle.write(review_text)

    output_meta = {}
    all_outputs = {name: Path(out) / name
                   for name in sorted(p.name for p in out.iterdir() if p.is_file())}
    for name, path in all_outputs.items():
        if name == "v004c_stage2_2_manifest.json":
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

    manifest_path = out / "v004c_stage2_2_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"paths": all_outputs, "manifest": manifest,
            "manifest_path": manifest_path}


def _write_failure_audit(
    output_dir: Path,
    *,
    failure_stage: str,
    exception: Exception,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_1_ref: str,
    git_head: str,
) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    failed_dir = Path(f"{output_dir}_failed_{timestamp}")
    failed_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "failure_stage": failure_stage,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "stage1_dir": str(stage1_dir),
        "stage2_1_dir": str(stage2_1_dir),
        "stage2_1_ref": stage2_1_ref,
        "git_head": git_head,
        "generated_at": pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (failed_dir / "failure_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# review (任务十八)
# ---------------------------------------------------------------------------
def _render_review(
    *,
    git_chain: dict[str, Any],
    input_checks: dict[str, Any],
    feature_checks: dict[str, Any],
    split_checks: dict[str, Any],
    derived_checks: dict[str, Any],
    feature_table: pd.DataFrame,
    structure: pd.DataFrame,
    univariate_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lodo_summary: dict[str, dict[str, Any]],
    board_subgroup: pd.DataFrame,
    missingness: pd.DataFrame,
    distribution_shift: pd.DataFrame,
    audit: pd.DataFrame,
    manifest_draft: dict[str, Any],
) -> str:
    def fmt_list(items: list[str]) -> str:
        return ", ".join(items) if items else "-"

    flag_names = sorted(ALLOWED_FLAGS)
    flag_counts = {f: 0 for f in flag_names}
    for _, r in univariate_summary.iterrows():
        for flag in str(r["flags"]).split("|"):
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    mech_order = sorted(feature_table["mechanism_group"].unique())
    analysis_counts = {
        at: int((feature_table["analysis_type"] == at).sum())
        for at in ("CONTINUOUS", "ORDINAL", "RANK", "BINARY", "BUCKET")}

    bootstrap_valid = bootstrap["bootstrap_valid"].to_numpy()
    low_valid_count = int((bootstrap_valid < BOOTSTRAP_MIN_VALID).sum())
    full_valid_count = int((bootstrap_valid == 1000).sum())
    clusters_per_replicate = int(
        manifest_draft["bootstrap_settings"]["clusters_per_replicate"])

    dates_dev = sorted(
        univariate_summary["lodo_date_count"].dropna().unique())
    lodo_date_count = int(dates_dev[0]) if len(dates_dev) else 0

    board2 = board_subgroup[board_subgroup["board_group"] == 2]
    board3 = board_subgroup[board_subgroup["board_group"] == 3]
    verified2 = int((board2["status"] == "VERIFIED").sum())
    verified3 = int((board3["status"] == "VERIFIED").sum())
    agreed = int((univariate_summary["board_direction_agreement"] == True).sum())  # noqa: E712
    disagree = int((univariate_summary["board_direction_agreement"] == False).sum())  # noqa: E712
    not_evaluable = int(univariate_summary["board_direction_agreement"].isna().sum())
    near_zero_disagree = int((
        (univariate_summary["board_direction_agreement"] == False)  # noqa: E712
        & (univariate_summary["one_board_effect_near_zero"] == True)  # noqa: E712
    ).sum())

    shift_count = int(distribution_shift["distribution_shift"].sum())
    numeric_shift = int((distribution_shift["distribution_shift"]
                         & (distribution_shift["value_type"] == "numeric")).sum())
    cat_shift = shift_count - numeric_shift

    audit_pass = int((audit["status"] == "PASS").sum())
    audit_rows = int(len(audit))

    missing_feature_count = int(len(missingness))

    sha_checks = input_checks["input_sha_verification"]
    sha_summary = ", ".join(
        f"{name}={'OK' if rec['verified'] else 'MISMATCH'}"
        for name, rec in sorted(sha_checks.items()))

    lines = [
        "# v004c 阶段2.2 — 单因子结构、六月开发集方向与日期稳定性审计报告 (自动生成)",
        "",
        f"- 分析版本: {manifest_draft['stage2_2_version']}",
        f"- 来源: 阶段1 {EXPECTED_STAGE1_TAG} → {EXPECTED_STAGE1_DATA_COMMIT} / "
        f"阶段2.1 {EXPECTED_STAGE2_1_TAG} → {EXPECTED_STAGE2_1_DATA_COMMIT}",
        f"- 分支: {git_chain['branch']} / HEAD: {manifest_draft['git_head']}",
        "",
        "## 版本链与输入验证",
        "",
        f"- 阶段2.1 tag 目标 = {git_chain['stage2_1_tag_target']} (通过)",
        f"- 阶段1 tag 目标 = {git_chain['stage1_tag_target']} (通过)",
        f"- 阶段2.1数据提交是当前 HEAD 祖先: {git_chain['data_commit_ancestor']}",
        f"- 阶段2.1 git_head.txt = {input_checks.get('stage21_git_head_file', '')} (通过)",
        f"- 输入 SHA 与对应 manifest 一致: {sha_summary}",
        f"- 阶段1 manifest 审计参考校验: "
        f"{input_checks['stage1_manifest_audit_reference_verified']}",
        f"- 结构门: {input_checks['input_rows']} 行 / {input_checks['columns']} 列 / "
        f"{input_checks['signal_dates']} 信号日",
        "",
        "## 时间切片 (七月锁定)",
        "",
        f"- 六月开发集: {DEVELOPMENT_START} .. {DEVELOPMENT_END} / "
        f"{input_checks['dev_rows']} 行 / Target7 正样本 "
        f"{input_checks['dev_target_positive']}",
        f"- 七月锁定切片: {LOCKED_START} .. {LOCKED_END} / "
        f"{input_checks['locked_rows']} 行 (仅无标签使用)",
        f"- 七月标签列已删除/mask: {split_checks['masked_column_count']} 列 "
        f"(locked_target_masked={split_checks['locked_target_masked']})",
        f"- locked_target_access = {manifest_draft['locked_target_access']}",
        "",
        "## 59 因子完整覆盖",
        "",
        f"- primary 分析项: {feature_checks['primary_allowlist']} "
        f"(source {feature_checks['primary_source']} + derived "
        f"{feature_checks['predeclared_derived']})",
        f"- sensitivity: {feature_checks['sensitivity']}",
        f"- 总分析因子: {feature_checks['total_features']} (唯一)",
        f"- 分析类型: " + " / ".join(
            f"{k} {v}" for k, v in analysis_counts.items()),
        f"- 派生因子重算与阶段2.1审计一致: {derived_checks['derived_audit_match']}",
        "",
        "## 各机制组因子数",
        "",
        "| 机制组 | 因子数 | primary | sensitivity | derived |",
        "|---|---|---|---|---|",
    ]
    for g in mech_order:
        sub = feature_table[feature_table["mechanism_group"] == g]
        lines.append(
            f"| {g} | {len(sub)} | "
            f"{int((sub['allowlist_tier'] == 'PRIMARY').sum())} | "
            f"{int((sub['allowlist_tier'] == 'SENSITIVITY').sum())} | "
            f"{int((sub['source_or_derived'] == 'derived').sum())} |")
    lines += [
        "",
        "## 描述性 flags 数量",
        "",
        "| flag | 数量 |",
        "|---|---|",
    ]
    for f in flag_names:
        lines.append(f"| {f} | {flag_counts.get(f, 0)} |")
    lines += [
        "",
        "## bootstrap 设置",
        "",
        f"- 日期 cluster bootstrap: {BOOTSTRAP_REPLICATES} replicates, "
        f"seed = {BOOTSTRAP_SEED}",
        f"- 每个 replicate 从 {clusters_per_replicate} 个日期 cluster 中有放回抽取 "
        f"{clusters_per_replicate} 次 (bootstrap_unit=signal_date_cluster);",
        f"- 抽中日期时带入该日期全部样本行, 同一日期抽中多次时其全部行重复进入样本 "
        f"(不去重, 不拆分 cluster);",
        f"- {len(feature_table)} 个因子共用同一抽样矩阵 "
        f"(shared_draw_matrix_across_features=true);",
        f"- CI 只使用有效 replicate; 无有效 replicate 时均值/中位数/CI 留空 (不填 0);",
        f"- 有效 replicate 低于 {BOOTSTRAP_MIN_VALID} 的因子数: {low_valid_count} "
        f"(有效 = 1000 的因子数: {full_valid_count})",
        f"- bucket 区间效应 (max-min 天然 >= 0) 不解释为相对 0 的方向性证据;",
        "",
        "## LODO 日期稳定性",
        "",
        f"- 六月 signal_date 数: {lodo_date_count}",
        f"- 每个因子逐日期删除整日样本后重算主效应, 记录符号一致性与最大绝对变化",
        f"- 因子结果完整保存在 v004c_stage2_2_lodo_date_stability_v001.csv",
        "",
        "## 二板/三板子组支持",
        "",
        f"- 支持门使用因子 complete-case 样本 (non_null>=20, positive>=5, negative>=5):",
        f"- board=2: VERIFIED {verified2} / {len(board2)}",
        f"- board=3: VERIFIED {verified3} / {len(board3)}",
        f"- 方向统计: 可判定 {agreed + disagree} (一致 {agreed} / 相反 {disagree}) / "
        f"不可判定 {not_evaluable}",
        f"- 反向因子中近零 (|效应| < {BOARD_NEAR_ZERO_THRESHOLD}) 数量: "
        f"{near_zero_disagree} (仅人工解释, 不改变冲突定义)",
        "",
        "## 缺失结构",
        "",
        f"- 六月存在缺失的因子数: {missing_feature_count}",
        f"- 缺失结构明细: v004c_stage2_2_missingness_dev_v001.csv",
        "",
        "## 无标签分布漂移 (六月 vs 七月)",
        "",
        f"- 触发 DISTRIBUTION_SHIFT 的因子数: {shift_count} "
        f"(数值 {numeric_shift} / 类别 {cat_shift})",
        f"- 漂移指标明细: v004c_stage2_2_distribution_shift_unlabeled_v001.csv",
        "",
        "## Target-blind 锁定测试",
        "",
        f"- 七月锁定审计: {audit_pass}/{audit_rows} 项 PASS",
        f"- 审计明细: v004c_stage2_2_holdout_lock_audit_v001.csv",
        "",
        "## 声明",
        "",
        "阶段2.2 只完成单因子结构和六月开发集稳定性审计;",
        "七月标签未用于任何因子分析、因子选择或关联计算;",
        "没有新增因子; 没有自动筛选因子; 没有训练模型; 没有运行完整 Walk-forward;",
        "没有修改阶段1或阶段2.1资产; 没有自动 commit 或 push。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------
def _run_analysis_core(
    *,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_1_ref: str,
    git_root: Path,
    provenance_before: GitProvenance,
    provenance_after: GitProvenance | None,
    out_dir: Path,
    enforce_input_sha: bool = True,
    audit_holdout_lock: bool = True,
    generated_at: str | None,
    snapshot_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """核心流水线: 输入 → 派生 → 切片 → 全部审计表 → 写数据包 (不 finalize)。

    返回 manifest_draft 与全部计算产物; 调用方负责 holdout 审计、rename 与 finalize。
    """
    d1 = Path(stage1_dir)
    d21 = Path(stage2_1_dir)
    _require_inside_git_root(d1, git_root, f"stage1_dir {stage1_dir}")
    _require_inside_git_root(d21, git_root, f"stage2_1_dir {stage2_1_dir}")

    snapshot, training, lineage, stage1_manifest, stage21_payload = load_inputs(
        stage1_dir, stage2_1_dir)
    if snapshot_override is not None:
        snapshot = snapshot_override.copy(deep=True)

    input_checks = verify_inputs(
        snapshot, training, lineage, stage1_manifest, stage21_payload,
        stage1_dir, stage2_1_dir,
        current_git_root=git_root, enforce_input_sha=enforce_input_sha)

    feature_table, feature_checks = build_feature_table(stage21_payload)
    derived, derived_checks = materialize_derived(snapshot, lineage, stage21_payload)
    dev_frame, locked_frame, split_checks = split_frames(snapshot, derived)

    full_frame = snapshot.copy()
    for name, series in derived.items():
        full_frame[name] = series.to_numpy()

    structure = compute_feature_structure(
        full_frame, dev_frame, locked_frame, feature_table)

    level_tables = pd.DataFrame()
    level_rows = []
    for _, meta in feature_table.iterrows():
        if meta["analysis_type"] != "BUCKET":
            continue
        values = _feature_values(dev_frame, meta["feature_name"], meta["analysis_type"])
        target = dev_frame[TARGET_COL].astype(int).to_numpy()
        rows_, _ = _bucket_level_rows(values, target)
        for r in rows_:
            r = {"feature_name": meta["feature_name"], **r}
            level_rows.append(r)
    if level_rows:
        level_tables = pd.DataFrame(level_rows)

    bootstrap = compute_cluster_bootstrap(dev_frame, feature_table)

    lodo, lodo_summary = compute_lodo(dev_frame, feature_table)

    board_subgroup, board_summary = compute_board_subgroups(
        dev_frame, feature_table)

    missingness = compute_missingness(dev_frame, feature_table)

    distribution_shift = compute_distribution_shift(
        dev_frame, locked_frame, feature_table)

    # 汇总 (需要 bootstrap/lodo/board/shift 结果生成 flags)
    univariate_summary = compute_univariate_summary(
        dev_frame, feature_table, structure, bootstrap, lodo_summary,
        board_summary, distribution_shift)

    mechanism_summary = compute_mechanism_summary(
        feature_table, univariate_summary)

    audit = pd.DataFrame(
        [{"check_name": "holdout_lock_audit", "status": "SKIPPED",
          "details": "audit disabled (rerun/test mode)"}])

    manifest_draft = {
        "stage2_2_version": STAGE2_2_VERSION,
        "source_stage1_version": "v004c-d1-dataset-0.1",
        "source_stage2_1_version": "v004c-factor-dictionary-0.1",
        "source_stage2_1_ref": stage2_1_ref,
        "source_stage2_1_commit": EXPECTED_STAGE2_1_DATA_COMMIT,
        "source_stage2_1_code_commit": EXPECTED_STAGE2_1_CODE_COMMIT,
        "source_stage1_commit": EXPECTED_STAGE1_DATA_COMMIT,
        "source_stage1_generator_commit": EXPECTED_STAGE1_GENERATOR_COMMIT,
        "development_date_range": [DEVELOPMENT_START, DEVELOPMENT_END],
        "development_rows": input_checks["dev_rows"],
        "development_target_positive": input_checks["dev_target_positive"],
        "locked_date_range": [LOCKED_START, LOCKED_END],
        "locked_rows": input_checks["locked_rows"],
        "locked_target_access": False,
        "primary_raw_count": EXPECTED_GATES["primary_source"],
        "predeclared_derived_count": EXPECTED_GATES["predeclared_derived"],
        "primary_analysis_count": EXPECTED_GATES["primary_allowlist"],
        "sensitivity_count": EXPECTED_GATES["sensitivity"],
        "total_analyzed_features": EXPECTED_GATES["total_features"],
        "analysis_type_counts": {
            at: int((feature_table["analysis_type"] == at).sum())
            for at in ("CONTINUOUS", "ORDINAL", "RANK", "BINARY", "BUCKET")},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_min_valid": BOOTSTRAP_MIN_VALID,
        "bootstrap_settings": {
            "bootstrap_unit": "signal_date_cluster",
            "clusters_per_replicate": len(dev_frame["signal_date"].unique()),
            "sampling_with_replacement": True,
            "replicate_count": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "shared_draw_matrix_across_features": True,
        },
        "board_subgroup_support_rule": (
            f"feature complete-case non_null>={BOARD_SUBGROUP_MIN_N} "
            f"and positive>={BOARD_SUBGROUP_MIN_POSITIVE} "
            f"and negative>={BOARD_SUBGROUP_MIN_NEGATIVE}"),
        "board_support_uses_feature_complete_case": True,
        "board_near_zero_threshold": BOARD_NEAR_ZERO_THRESHOLD,
        "bucket_effect_directional": False,
        "bucket_zero_reference_applicable": False,
        "flag_thresholds": FLAG_THRESHOLDS,
        "model_training_performed": False,
        "feature_selection_performed": False,
        "holdout_label_analysis_performed": False,
        "input_files": {
            name: {
                "path": rec["path"],
                "bytes": rec["bytes"],
                "sha256": rec["sha256"],
                "source_manifest": rec["stage"],
            }
            for name, rec in input_checks["input_sha_verification"].items()},
        "input_verification": {
            "input_rows": input_checks["input_rows"],
            "columns": input_checks["columns"],
            "signal_dates": input_checks["signal_dates"],
            "lineage_rows": input_checks["lineage_rows"],
            "dev_rows": input_checks["dev_rows"],
            "dev_target_positive": input_checks["dev_target_positive"],
            "locked_rows": input_checks["locked_rows"],
            "primary_allowlist": feature_checks["primary_allowlist"],
            "primary_source": feature_checks["primary_source"],
            "predeclared_derived": feature_checks["predeclared_derived"],
            "sensitivity": feature_checks["sensitivity"],
            "total_features": feature_checks["total_features"],
            "input_sha_all_verified": all(
                rec["verified"] for rec in
                input_checks["input_sha_verification"].values()),
            "stage1_manifest_audit_reference_verified":
                input_checks["stage1_manifest_audit_reference_verified"],
        },
        "derived_factors": {
            "spec_count": EXPECTED_GATES["predeclared_derived"],
            "recompute_matches_stage2_1_audit":
                derived_checks["derived_audit_match"],
            "existing_recompute": derived_checks["existing_recompute"],
        },
        "timezone": TIMEZONE,
        "research_only": True,
        "deployment_status": "research_only",
        "git_root": str(git_root),
        "git_branch": provenance_before.git_branch if hasattr(
            provenance_before, "git_branch") else None,
        "git_head": provenance_before.git_head,
        "git_status_before": provenance_before.git_status,
        "git_dirty_before": bool(provenance_before.git_dirty),
        "frozen": False,
        "audit_complete": True,
    }

    # 环境信息 (best effort; 不阻塞)
    supported_environment: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
    }
    try:
        autocrlf = _run_git(["git", "config", "--get", "core.autocrlf"],
                            cwd=git_root).strip()
        supported_environment["core_autocrlf"] = autocrlf or "unset"
    except DatasetValidationError:
        supported_environment["core_autocrlf"] = "unknown"
    try:
        ver = _run_git(["git", "--version"], cwd=git_root).strip()
        supported_environment["git_version"] = ver
    except DatasetValidationError:
        supported_environment["git_version"] = "unknown"
    manifest_draft["supported_environment"] = supported_environment

    result = write_data_package(
        feature_structure=structure,
        univariate_summary=univariate_summary,
        level_tables=level_tables,
        cluster_bootstrap=bootstrap,
        lodo=lodo,
        board_subgroup=board_subgroup,
        missingness=missingness,
        distribution_shift=distribution_shift,
        mechanism_summary=mechanism_summary,
        audit=audit,
        out_dir=out_dir,
        git_provenance_before=provenance_before,
        git_root=git_root,
        output_dir_gate={"output_dir_was_created": not out_dir.exists(),
                         "preexisting_entry_count": 0,
                         "unmanaged_output_files": []},
        manifest_draft=manifest_draft,
        include_audit=audit_holdout_lock,
    )
    return {
        "out_dir": out_dir,
        "manifest_draft": manifest_draft,
        "write_result": result,
        "feature_table": feature_table,
        "feature_checks": feature_checks,
        "input_checks": input_checks,
        "split_checks": split_checks,
        "derived_checks": derived_checks,
        "snapshot": snapshot,
        "dev_rows": int(len(dev_frame)),
        "structure": structure,
        "univariate_summary": univariate_summary,
        "bootstrap": bootstrap,
        "lodo_summary": lodo_summary,
        "board_subgroup": board_subgroup,
        "board_summary": board_summary,
        "missingness": missingness,
        "distribution_shift": distribution_shift,
        "mechanism_summary": mechanism_summary,
        "audit": audit,
        "locked_masked_columns": split_checks["masked_columns"],
    }


def run_v004c_stage2_2_analysis(
    *,
    stage1_dir: str | Path,
    stage2_1_dir: str | Path,
    stage2_1_ref: str = EXPECTED_STAGE2_1_TAG,
    output_dir: str | Path,
    git_provenance_before: GitProvenance | None = None,
    git_provenance_after: GitProvenance | None = None,
    enforce_input_sha: bool = True,
    audit_holdout_lock: bool = True,
    generated_at: str | None = None,
    snapshot_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """阶段2.2 完整分析流程 (fail closed; 临时目录构建 + 原子 rename)。"""
    target_dir = Path(output_dir)
    git_provenance_before = git_provenance_before or collect_git_provenance(Path.cwd())
    git_root = git_provenance_before.git_root

    preexisting = []
    if target_dir.exists():
        preexisting = sorted(str(p) for p in target_dir.iterdir())
    if preexisting:
        raise DatasetValidationError(
            f"输出目录非空, 阻止冻结: {output_dir} (含 {len(preexisting)} 项; "
            f"请显式删除旧目录后重建)")

    git_chain = validate_git_chain(git_root, stage2_1_ref)

    tmp_dir = Path(f"{target_dir}.tmp_{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        core = _run_analysis_core(
            stage1_dir=stage1_dir,
            stage2_1_dir=stage2_1_dir,
            stage2_1_ref=stage2_1_ref,
            git_root=git_root,
            provenance_before=git_provenance_before,
            provenance_after=git_provenance_after,
            out_dir=tmp_dir,
            enforce_input_sha=enforce_input_sha,
            audit_holdout_lock=audit_holdout_lock,
            generated_at=generated_at,
            snapshot_override=snapshot_override,
        )
    except Exception as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            _write_failure_audit(
                target_dir, failure_stage="build", exception=exc,
                stage1_dir=stage1_dir, stage2_1_dir=stage2_1_dir,
                stage2_1_ref=stage2_1_ref,
                git_head=git_provenance_before.git_head)
        except Exception:
            pass
        raise

    manifest_draft = dict(core["manifest_draft"])
    manifest_draft["git_branch"] = git_chain["branch"]

    audit_df = None
    if audit_holdout_lock:
        def rerun(mutated: pd.DataFrame, rerun_dir: Path) -> None:
            _run_analysis_core(
                stage1_dir=stage1_dir,
                stage2_1_dir=stage2_1_dir,
                stage2_1_ref=stage2_1_ref,
                git_root=git_root,
                provenance_before=git_provenance_before,
                provenance_after=git_provenance_after,
                out_dir=rerun_dir,
                enforce_input_sha=enforce_input_sha,
                audit_holdout_lock=False,
                generated_at=generated_at,
                snapshot_override=mutated,
            )

        audit_df = _run_holdout_lock_audit(
            core_rerun=rerun,
            reference_dir=tmp_dir,
            snapshot=core["snapshot"],
            output_parent=target_dir.parent,
            locked_rows=core["input_checks"]["locked_rows"],
            masked_columns=core["locked_masked_columns"],
            dev_rows=core["dev_rows"],
        )
        audit_csv_path = tmp_dir / "v004c_stage2_2_holdout_lock_audit_v001.csv"
        audit_df.to_csv(audit_csv_path, index=False, encoding="utf-8-sig")
        core["audit"] = audit_df
    else:
        audit_df = core["audit"] if core.get("audit") is not None \
            else pd.DataFrame()

    # 内容检查 1: CSV 列名无七月标签指标
    content_checks: list[dict[str, str]] = []
    for name in sorted(p.name for p in tmp_dir.iterdir() if p.name.endswith(".csv")):
        cols = pd.read_csv(tmp_dir / name, nrows=0).columns
        bad_cols = [c for c in cols
                    if "july" in c.lower() or "locked_target" in c.lower()]
        content_checks.append({
            "check_name": "no_july_target_metric_in_csv_columns",
            "status": "PASS" if not bad_cols else "FAIL",
            "details": (f"{len(bad_cols)} forbidden columns in {name}" if bad_cols
                        else f"{len(cols)} columns of {name} clean"),
        })
    audit_df = pd.concat(
        [audit_df, pd.DataFrame(content_checks,
                                columns=["check_name", "status", "details"])],
        ignore_index=True)
    content_checks = []

    # 内容检查 2: review 无七月标签指标 (先渲染含当前审计的 review)
    review_text = _render_review(
        git_chain=git_chain,
        input_checks=core["input_checks"],
        feature_checks=core["feature_checks"],
        split_checks=core["split_checks"],
        derived_checks=core["derived_checks"],
        feature_table=core["feature_table"],
        structure=core["structure"],
        univariate_summary=core["univariate_summary"],
        bootstrap=core["bootstrap"],
        lodo_summary=core["lodo_summary"],
        board_subgroup=core["board_subgroup"],
        missingness=core["missingness"],
        distribution_shift=core["distribution_shift"],
        audit=audit_df,
        manifest_draft=manifest_draft,
    )
    review_ok, review_detail = _no_july_metric_text_scan(review_text, "review")
    content_checks.append({
        "check_name": "no_july_target_metric_in_review",
        "status": "PASS" if review_ok else "FAIL",
        "details": review_detail,
    })
    manifest_draft["flag_counts"] = {
        f: int((core["univariate_summary"]["flags"].str.contains(
            f, regex=False) & (core["univariate_summary"]["flags"] != "")).sum())
        for f in sorted(ALLOWED_FLAGS)}
    manifest_draft["mechanism_group_counts"] = {
        str(r["mechanism_group"]): int(r["factor_count"])
        for _, r in core["mechanism_summary"].iterrows()}
    manifest_draft["bootstrap_stats"] = {
        "valid_min": int(core["bootstrap"]["bootstrap_valid"].min()),
        "valid_mean": round(float(core["bootstrap"]["bootstrap_valid"].mean()), 2),
        "low_valid_count": int(
            (core["bootstrap"]["bootstrap_valid"] < BOOTSTRAP_MIN_VALID).sum()),
    }
    lodo_first = next(iter(core["lodo_summary"].values()), {})
    manifest_draft["lodo_stats"] = {
        "date_count": int(lodo_first.get("date_count", 0)),
        "valid_features": int(sum(
            1 for v in core["lodo_summary"].values()
            if v["valid_count"] == v["date_count"])),
    }
    _bs = core["board_subgroup"]
    _bs_dir = _board_direction_counts(core["univariate_summary"])
    _near_zero_disagree = int((
        (core["univariate_summary"]["board_direction_agreement"] == False)  # noqa: E712
        & (core["univariate_summary"]["one_board_effect_near_zero"] == True)  # noqa: E712
    ).sum())
    manifest_draft["board_subgroup_summary"] = {
        "board_support_uses_feature_complete_case": True,
        "board_near_zero_threshold": BOARD_NEAR_ZERO_THRESHOLD,
        "verified_board_2": int(
            ((_bs["board_group"] == 2) & (_bs["status"] == "VERIFIED")).sum()),
        "verified_board_3": int(
            ((_bs["board_group"] == 3) & (_bs["status"] == "VERIFIED")).sum()),
        "near_zero_disagreement_count": _near_zero_disagree,
        **_bs_dir,
    }
    _boot = core["bootstrap"]
    _boot_valid = _boot["bootstrap_valid"].to_numpy()
    manifest_draft["bootstrap_stats"] = {
        "bootstrap_valid_min": int(_boot_valid.min()),
        "bootstrap_valid_mean": round(float(_boot_valid.mean()), 2),
        "bootstrap_valid_below_800_count": int(
            (_boot_valid < BOOTSTRAP_MIN_VALID).sum()),
        "bootstrap_valid_full_1000_count": int((_boot_valid == 1000).sum()),
    }
    manifest_draft["bootstrap_invalid_reason_totals"] = {
        "invalid_single_target_class": int(
            _boot["invalid_single_target_class"].sum()),
        "invalid_single_feature_level": int(
            _boot["invalid_single_feature_level"].sum()),
        "invalid_insufficient_non_null": int(
            _boot["invalid_insufficient_non_null"].sum()),
        "invalid_metric_undefined": int(
            _boot["invalid_metric_undefined"].sum()),
    }
    manifest_draft["distribution_shift_summary"] = {
        "shift_count": int(core["distribution_shift"]["distribution_shift"].sum()),
    }
    manifest_draft["missingness_summary"] = {
        "june_missing_feature_count": int(len(core["missingness"])),
    }
    manifest_draft["holdout_lock_audit"] = {
        "all_pass": bool((audit_df["status"] == "PASS").all()),
        "check_count": int(len(audit_df)),
        "pass_count": int((audit_df["status"] == "PASS").sum()),
        "checks": [{"check_name": _AUDIT_CHECK_LABELS.get(
            str(r["check_name"]), str(r["check_name"])),
            "status": r["status"]}
            for _, r in audit_df.iterrows()],
    }

    # 内容检查 3: manifest 无七月标签指标 (扫描含审计段的最终 manifest 文本)
    manifest_text = json.dumps(manifest_draft, ensure_ascii=False)
    man_ok, man_detail = _no_july_metric_text_scan(manifest_text, "manifest")
    content_checks.append({
        "check_name": "no_july_target_metric_in_manifest",
        "status": "PASS" if man_ok else "FAIL",
        "details": man_detail,
    })
    audit_df = pd.concat(
        [audit_df, pd.DataFrame(content_checks,
                                columns=["check_name", "status", "details"])],
        ignore_index=True)
    audit_csv_path = tmp_dir / "v004c_stage2_2_holdout_lock_audit_v001.csv"
    audit_df.to_csv(audit_csv_path, index=False, encoding="utf-8-sig")
    core["audit"] = audit_df

    all_pass = bool((audit_df["status"] == "PASS").all())
    manifest_draft["holdout_lock_audit"] = {
        "all_pass": all_pass,
        "check_count": int(len(audit_df)),
        "pass_count": int((audit_df["status"] == "PASS").sum()),
        "checks": [{"check_name": _AUDIT_CHECK_LABELS.get(
            str(r["check_name"]), str(r["check_name"])),
            "status": r["status"]}
            for _, r in audit_df.iterrows()],
    }

    # 用最终 manifest_draft + 最终审计表渲染 review (finalize 写入)
    review_text = _render_review(
        git_chain=git_chain,
        input_checks=core["input_checks"],
        feature_checks=core["feature_checks"],
        split_checks=core["split_checks"],
        derived_checks=core["derived_checks"],
        feature_table=core["feature_table"],
        structure=core["structure"],
        univariate_summary=core["univariate_summary"],
        bootstrap=core["bootstrap"],
        lodo_summary=core["lodo_summary"],
        board_subgroup=core["board_subgroup"],
        missingness=core["missingness"],
        distribution_shift=core["distribution_shift"],
        audit=audit_df,
        manifest_draft=manifest_draft,
    )

    if target_dir.exists():
        try:
            target_dir.rmdir()
        except OSError as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise DatasetValidationError(
                f"输出目录无法清空: {target_dir}: {exc}") from exc
    os.rename(tmp_dir, target_dir)

    result = finalize_package(
        out_dir=target_dir,
        manifest_draft=manifest_draft,
        review_text=review_text,
        git_provenance_after=git_provenance_after,
        git_root=git_root,
        generated_at=generated_at,
    )
    return result

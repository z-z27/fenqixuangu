"""v004c 阶段1: D1 首次断板训练数据冻结与审计 (正式模块, 阶段1.1收尾修复)

统一交易时序 (本模块的规范命名):
    D0 = 最后一个成功连板日 (二板或三板完成日)
    D1 = D0 之后的首次断板日 (旧数据中的 stage_group=="d0" / days_since_break==0)
    D1 收盘后 = v004c 生成推荐排行
    D2 早盘 = 用户结合集合竞价决定是否买入
    D3 盘中 = 检验相对 D2 开盘是否达到 +7%

正式标签:
    target7 = 1[D3_high_daily / D2_open_daily - 1 >= 0.07]

阶段1.1 收尾修复:
- 严格二元字段规范 normalize_binary_series (显式映射, 非法值阻止冻结);
- Git provenance fail closed (真实 git root / 40位hex HEAD / 失败阻止冻结);
- 股票代码严格 ^\\d{6}$ (不再自动补零);
- 缓存非法日期阻止冻结 (raw/valid/invalid/duplicate 计数进入 provenance);
- 输出目录门禁 (不存在→创建, 空→允许, 非空→fail closed);
- git_status_after 在全部非 manifest 输出写完后采集 (明确语义);
- 交易日邻接证据表 v004c_d1_trade_date_adjacency_evidence.csv (含 D3 前日期序列 SHA);
- 临时目录构建 + 原子 rename + 失败审计目录;
- 消除 ResourceWarning (全部文本/JSON/CSV 读取使用 context manager)。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_VERSION = "v004c-d1-dataset-0.1"
SOURCE_DATASET_VERSION = "v004c-dataset-0.2.2"
CANDIDATE_DEFINITION = "two_or_three_board_first_break_d1"
LABEL_DEFINITION = "daily_d2open_to_d3high_target7"
ADJUSTMENT = "none"
TIMEZONE = "Asia/Shanghai"

TARGET_THRESHOLD = 0.07
STAGE_D0_VALUE = "d0"
STAGE_POST_VALUE = "post"
ALLOWED_BOARD_STREAKS = (2, 3)

GIT_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CODE_PATTERN = re.compile(r"^\d{6}$")

# 源 manifest 期望值 (冻结门参考; 新 manifest 记录的 source 版本取自已验证的源 manifest)
EXPECTED_SOURCE_MANIFEST = {
    "dataset_version": SOURCE_DATASET_VERSION,
    "candidate_definition_version": "break-repair-0.1",
    "label_definition_version": "v004c-daily-label-0.2",
    "adjustment": "none",
}
SOURCE_TRAINING_FILE_KEY = "v004c_training_primary_v022.csv"

# 需要严格二元验证的字段 (D1 集合内必须无缺失)
STRICT_BINARY_COLUMNS = (
    "target7_daily_d2open_d3high",
    "tail_loss_daily_5pct",
    "daily_label_quality_ok",
    "target_training_eligible",
    "tail_training_eligible",
    "d2_minute_complete",
    "d3_minute_complete",
)

_BINARY_TRUE_VALUES = {True, 1, 1.0, "1", "true"}
_BINARY_FALSE_VALUES = {False, 0, 0.0, "0", "false"}


class DatasetValidationError(RuntimeError):
    """数据集校验失败; 任何硬性失败都阻止数据冻结。"""


# ---------------------------------------------------------------------------
# 严格二元字段规范
# ---------------------------------------------------------------------------
def normalize_binary_series(
    series: pd.Series,
    *,
    column_name: str,
    allow_missing: bool = False,
) -> pd.Series:
    """严格二元字段规范 (显式映射, 禁止 Python 隐式 truthiness)。

    允许: True/False, 1/0, 1.0/0.0, "true"/"false"/"1"/"0" (strip + 大小写不敏感)。
    禁止: 2, -1, 0.5, "yes", "no", "Y", "N", "", 空白, NaN, ±inf 及任何其它值。
    非法值抛 DatasetValidationError (含字段名/非法值/行数/示例索引)。
    """
    out: list[Any] = []
    invalid: list[tuple[int, Any]] = []
    for idx, raw in series.items():
        value: Any = None
        if isinstance(raw, bool):
            value = bool(raw)
        elif raw is None or (isinstance(raw, float) and np.isnan(raw)):
            if allow_missing:
                value = np.nan
            else:
                invalid.append((idx, raw))
        elif isinstance(raw, (int, np.integer)):
            if raw in (0, 1):
                value = bool(raw)
            else:
                invalid.append((idx, raw))
        elif isinstance(raw, (float, np.floating)):
            if raw in (0.0, 1.0):
                value = bool(raw)
            else:
                invalid.append((idx, raw))
        elif isinstance(raw, str):
            text = raw.strip().lower()
            if text in _BINARY_TRUE_VALUES:
                value = True
            elif text in _BINARY_FALSE_VALUES:
                value = False
            else:
                invalid.append((idx, raw))
        else:
            invalid.append((idx, raw))
        out.append(value)
    if invalid:
        examples = [str(v) for _, v in invalid[:5]]
        raise DatasetValidationError(
            f"字段 {column_name} 含非法二元值: 非法行数={len(invalid)}, "
            f"示例值={examples}, 示例索引={[i for i, _ in invalid[:5]]}")
    result = pd.Series(out, index=series.index, dtype=object)
    if allow_missing:
        result = result.where(result.notna(), pd.NA)
        return result
    return result.astype(bool)


# ---------------------------------------------------------------------------
# Git provenance (fail closed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GitProvenance:
    git_root: Path
    git_head: str
    git_status: str
    git_dirty: bool


def _run_git(args: list[str], cwd: Path) -> str:
    import subprocess

    result = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DatasetValidationError(
            f"Git 命令失败 ({' '.join(args)}): rc={result.returncode} "
            f"stderr={result.stderr.strip()[:200]}")
    return result.stdout


def validate_git_head(head: str) -> str:
    """校验 git HEAD 严格格式 ^[0-9a-f]{40}$; 空/长度错误/非 hex 均失败。"""
    if not isinstance(head, str) or not GIT_HEAD_PATTERN.fullmatch(head):
        raise DatasetValidationError(f"非法 git HEAD: {head!r} (须为 40 位 hex)")
    return head


def collect_git_provenance(repo_root: Path) -> GitProvenance:
    """严格采集 Git provenance (fail closed)。

    - 先 git rev-parse --show-toplevel 取得真实 Git 根目录;
    - HEAD 必须匹配 ^[0-9a-f]{40}$;
    - 任何 Git 失败 / 非 Git 目录 / 空或非法 HEAD 都阻止冻结。
    """
    root_text = _run_git(["git", "rev-parse", "--show-toplevel"], cwd=repo_root).strip()
    if not root_text:
        raise DatasetValidationError("Git 根目录无法解析 (rev-parse --show-toplevel 为空)")
    git_root = Path(root_text)

    head = _run_git(["git", "rev-parse", "HEAD"], cwd=repo_root).strip()
    validate_git_head(head)
    status = _run_git(["git", "status", "--short"], cwd=repo_root)
    dirty = bool(status.strip())
    return GitProvenance(git_root=git_root, git_head=head, git_status=status, git_dirty=dirty)


def _repo_relative_posix(path: str | Path, git_root: Path) -> str:
    """基于真实 git_root 的仓库相对 POSIX 路径 (跨盘符时回退为绝对 POSIX)。"""
    resolved = Path(path).resolve()
    try:
        rel = os.path.relpath(resolved, git_root.resolve())
    except ValueError:
        return resolved.as_posix()
    return Path(rel).as_posix()


# ---------------------------------------------------------------------------
# 读取与 schema 发现
# ---------------------------------------------------------------------------
def read_v004c_source(csv_path: str | Path, manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, dict | None]:
    """按 v004c 规范读取输入 CSV (code 严格六位, round-trip 浮点) 与 manifest。"""
    frame = pd.read_csv(
        csv_path,
        dtype={"code": str},
        float_precision="round_trip",
    )
    if "code" in frame.columns:
        frame["code"] = _validate_strict_codes(frame["code"])
    manifest = None
    if manifest_path is not None:
        try:
            with Path(manifest_path).open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            raise DatasetValidationError(f"源 manifest 读取/解析失败: {manifest_path}: {exc}") from exc
    return frame, manifest


def _validate_strict_codes(codes: pd.Series) -> pd.Series:
    """股票代码严格 ^\\d{6}$: 不自动补零, 首尾空格直接失败。"""
    out = []
    for idx, raw in codes.items():
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raise DatasetValidationError(f"股票代码缺失 (None/NaN), 行 {idx}")
        text = str(raw)
        if not CODE_PATTERN.fullmatch(text):
            raise DatasetValidationError(
                f"股票代码格式非法: {raw!r}, 行 {idx} (须为恰好 6 位数字, 不自动补零)")
        out.append(text)
    return pd.Series(out, index=codes.index, dtype=str)


def validate_source_manifest(manifest: dict | None, csv_path: str | Path) -> dict:
    """硬校验源 manifest (fail closed); 返回已验证 manifest。"""
    if manifest is None:
        raise DatasetValidationError("源 manifest 缺失, 阻止冻结")
    for key, expected in EXPECTED_SOURCE_MANIFEST.items():
        actual = manifest.get(key)
        if actual != expected:
            raise DatasetValidationError(
                f"源 manifest 字段 {key} 不符: expected={expected!r} actual={actual!r}")
    output_files = manifest.get("output_files") or {}
    entry = output_files.get(SOURCE_TRAINING_FILE_KEY) or {}
    recorded_sha = str(entry.get("sha256") or "")
    if not recorded_sha:
        raise DatasetValidationError(f"源 manifest 缺少 {SOURCE_TRAINING_FILE_KEY} 的 sha256")
    actual_sha = _sha256(csv_path)
    if actual_sha != recorded_sha:
        raise DatasetValidationError(
            f"输入 CSV SHA256 不符: manifest={recorded_sha} actual={actual_sha}")
    return manifest


def discover_v004c_schema(frame: pd.DataFrame) -> dict[str, Any]:
    """发现输入 schema。"""
    columns = set(frame.columns)
    schema: dict[str, Any] = {
        "stage_group_col": "stage_group" if "stage_group" in columns else None,
        "days_since_break_col": "days_since_break" if "days_since_break" in columns else None,
        "signal_date_col": "signal_date" if "signal_date" in columns else None,
        "break_date_col": "break_date" if "break_date" in columns else None,
        "board_streak_col": "board_streak_before_break" if "board_streak_before_break" in columns else None,
        "d0_last_board_date_col": "last_board_date" if "last_board_date" in columns else None,
        "d2_entry_date_col": "label_d2_date" if "label_d2_date" in columns else ("expected_d2_date" if "expected_d2_date" in columns else None),
        "d3_outcome_date_col": "label_d3_date" if "label_d3_date" in columns else ("expected_d3_date" if "expected_d3_date" in columns else None),
        "d2_open_col": "d2_open_daily" if "d2_open_daily" in columns else None,
        "d3_high_col": "d3_high_daily" if "d3_high_daily" in columns else None,
        "d3_close_col": "d3_close_daily" if "d3_close_daily" in columns else None,
        "target_col": "target7_daily_d2open_d3high" if "target7_daily_d2open_d3high" in columns else None,
        "daily_label_quality_col": "daily_label_quality_ok" if "daily_label_quality_ok" in columns else None,
        "column_count": int(len(frame.columns)),
        "row_count": int(len(frame)),
        "missing_required_identity": sorted(
            c for c in ("event_id", "code") if c not in columns
        ),
    }
    return schema


# ---------------------------------------------------------------------------
# 列血缘分类 (阶段1.1 未变, 保持上次修复)
# ---------------------------------------------------------------------------
_IDENTIFIER_COLS = {
    "event_id", "code", "name",
    "signal_date", "break_date", "last_board_date", "d0_last_board_date",
    "d1_break_date", "d1_signal_date",
}

_D0_FEATURE_COLS = {
    "last_board_day_in_pool",
    "pool_consecutive_count_last_board",
    "board_day_amount_rank", "board_day_turnover_rank", "board_day_volume_rank",
}

_D1_FEATURE_COLS = {
    "board_streak_before_break",
    "recent_limit_up_count_10d", "recent_limit_up_count_20d",
    "recent_pool_appearance_count_10d", "recent_pool_appearance_count_20d",
    "max_board_streak_20d",
    "break_open", "break_high", "break_low", "break_close", "break_prev_close",
    "break_open_return", "break_high_return", "break_close_return",
    "break_intraday_range", "break_high_to_close_drawdown",
    "break_upper_shadow_ratio", "break_lower_shadow_ratio", "break_close_location",
    "break_volume", "break_amount",
    "break_volume_ratio_vs_board_days", "break_amount_ratio_vs_board_days",
    "break_turnover_ratio", "break_touched_limit_up", "break_opened_from_limit_up",
    "break_day_in_pool",
    "d1_open", "d1_high", "d1_low", "d1_close", "d1_volume", "d1_amount",
    "d1_ma5", "d1_ma10", "d1_ma20",
    "d1_close_to_ma5_raw", "d1_low_to_ma5_raw", "d1_high_to_ma5_raw",
    "d1_close_to_ma10_raw", "d1_low_to_ma10_raw", "d1_close_to_ma20",
    "d1_ma5_slope", "d1_ma10_slope",
    "deprecated_d1_reclaimed_ma5_v01", "deprecated_d1_reclaimed_ma10_v01",
    "consecutive_days_below_ma5", "consecutive_days_below_ma10",
    "d1_open_to_close_return_raw", "d1_low_to_close_recovery",
    "d1_high_to_close_drawdown_raw", "d1_close_location",
    "d1_vwap", "d1_close_to_vwap_raw", "d1_intraday_range",
    "d1_afternoon_return", "d1_last_hour_return",
    "d1_up_bar_volume_ratio", "d1_down_bar_volume_ratio",
    "volume_above_d1_close_ratio", "amount_above_d1_close_ratio",
    "volume_above_break_close_ratio",
    "high_zone_volume_ratio", "high_zone_amount_ratio",
    "late_day_sell_volume_ratio", "late_day_sell_amount_ratio",
    "down_bar_volume_ratio", "d1_vwap_to_close_gap",
    "d1_close_above_ma5", "d1_close_above_ma10",
    "d1_true_reclaim_ma5", "d1_true_reclaim_ma10",
    "d1_close_to_ma5_bucket", "d1_open_to_close_bucket",
}

_LABEL_COLS = {
    "d2_open_daily", "d3_high_daily", "d3_close_daily",
    "daily_d2open_to_d3high_return", "daily_d2open_to_d3close_return",
    "target7_daily_d2open_d3high", "tail_loss_daily_5pct",
}

_OUTCOME_AUDIT_COLS = {
    "d2_trade_date", "d3_trade_date",
    "d2_open", "d2_high", "d2_low", "d2_close",
    "d3_high", "d3_low", "d3_close",
    "d2open_to_d3high_return", "d2open_to_d3close_return",
    "target7_d2open_d3high", "tail_loss_5pct",
    "audit_minute_d2_open", "audit_minute_d3_high", "audit_minute_d3_close",
    "audit_minute_target7", "audit_minute_tail_loss",
    "audit_minute_d2open_to_d3high_return", "audit_minute_d2open_to_d3close_return",
    "expected_d2_date", "expected_d3_date",
    "actual_d2_date", "actual_d3_date",
    "actual_daily_d2_date", "actual_daily_d3_date",
    "label_d2_date", "label_d3_date",
    "d2_entry_date", "d3_outcome_date",
    "target_label_daily_vs_minute_same", "tail_label_daily_vs_minute_same",
    "d2_open_daily_minus_minute", "d3_high_daily_minus_minute",
    "d3_close_daily_minus_minute",
    "target_return_daily_minus_minute", "close_return_daily_minus_minute",
}

_EXISTING_MODEL_AUDIT_COLS = {
    "is_v004a_scorable", "v004a_probability", "v004a_rank",
    "is_v004a_top3", "is_v004a_top10", "is_v004a_top15", "v004a_score_source",
    "is_v002_scorable", "v002_rank",
    "is_v002_top3", "is_v002_top10", "is_v002_top15",
    "recognition_score",
}

_DATA_QUALITY_D1_CLOSE = {
    "d1_bar_count", "d1_first_bar_time", "d1_last_bar_time",
    "d1_expected_bar_count", "d1_minute_complete", "d1_factor_quality_ok",
}
_DATA_QUALITY_POST_D3 = {
    "d2_bar_count", "d3_bar_count",
    "d2_first_bar_time", "d3_first_bar_time",
    "d2_last_bar_time", "d3_last_bar_time",
    "d2_expected_bar_count", "d3_expected_bar_count",
    "d2_minute_complete", "d3_minute_complete",
    "future_data_status",
    "target_training_eligible", "tail_training_eligible",
    "daily_label_source_d2", "daily_label_source_d3",
    "label_quality_ok", "label_quality_reason",
    "target_quality_ok", "tail_label_quality_ok",
    "daily_label_quality_ok", "daily_label_quality_reason",
    "d2_daily_minute_open_diff", "d3_daily_minute_high_diff", "d3_daily_minute_close_diff",
    "suspension_proof_status", "suspension_proof_status_daily",
    "d2_date_shift_reason", "d3_date_shift_reason",
    "d2_daily_date_shift_reason", "d3_daily_date_shift_reason",
}
_DATA_QUALITY_BUILD_TIME = {
    "daily_label_cache_path", "daily_label_cache_mtime", "daily_label_cache_sha256",
}

_PROVENANCE_COLS = {
    "stage_group", "post_day", "training_stage_feature",
    "days_since_break", "days_since_last_limit_up", "repair_attempt_count",
    "sample_role", "sample_role_reason", "sample_role_v021", "sample_role_v022",
    "event_observation_count", "signal_date_candidate_count", "proposed_training_weight",
}

_AVAILABLE_AS_OF = {
    "identifier": "D1_CLOSE",
    "d0_feature": "D0_CLOSE",
    "d1_feature": "D1_CLOSE",
    "label": "D3_CLOSE",
    "outcome_audit": "POST_D3",
    "existing_model_audit": "D1_CLOSE",
    "data_quality": "POST_D3",
    "provenance": "D1_CLOSE",
    "forbidden_unknown": "UNKNOWN",
}

_KEY_FORMULAS = {
    "event_id": "code + '_' + break_date (v0.2.2 来源, 未改变)",
    "board_streak_before_break": "断板前连续涨停天数 ∈ {2,3} (日线相邻交易日判定, v0.2.2 来源)",
    "last_board_date": "D0 最后连板日 (断板前最后一个交易日, v0.2.2 来源)",
    "break_date": "D1 断板日 = 连续 2/3 板后第一个非涨停交易日 (v0.2.2 来源)",
    "signal_date": "D1 信号日 = 断板日 (v0.2.2 来源)",
    "d2_trade_date": "标签口径 D2 交易日 (5min 序列, 分钟标签审计用)",
    "d3_trade_date": "标签口径 D3 交易日 (5min 序列, 分钟标签审计用)",
    "d2_open_daily": "标签: D2 日线 open (未复权 daily OHLC)",
    "d3_high_daily": "标签: D3 日线 high (未复权 daily OHLC)",
    "daily_d2open_to_d3high_return": "标签: d3_high_daily / d2_open_daily - 1",
    "target7_daily_d2open_d3high": "标签: daily_d2open_to_d3high_return >= 0.07 (严格二元)",
    "daily_label_quality_ok": "D1/D2/D3 日线标签质量门 (日期可解析且价格有效, 严格二元)",
}


def classify_column_lineage(frame: pd.DataFrame, schema: dict | None = None) -> pd.DataFrame:
    """对每列分类列血缘。"""
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if column in _IDENTIFIER_COLS:
            role = "identifier"
        elif column in _D0_FEATURE_COLS:
            role = "d0_feature"
        elif column in _D1_FEATURE_COLS:
            role = "d1_feature"
        elif column in _LABEL_COLS:
            role = "label"
        elif column in _OUTCOME_AUDIT_COLS:
            role = "outcome_audit"
        elif column in _EXISTING_MODEL_AUDIT_COLS or column.startswith("existing_model"):
            role = "existing_model_audit"
        elif column in _DATA_QUALITY_D1_CLOSE:
            role = "data_quality"
            available = "D1_CLOSE"
        elif column in _DATA_QUALITY_POST_D3:
            role = "data_quality"
            available = "POST_D3"
        elif column in _DATA_QUALITY_BUILD_TIME:
            role = "data_quality"
            available = "BUILD_TIME"
        elif column in _PROVENANCE_COLS:
            role = "provenance"
        else:
            role = "forbidden_unknown"
            available = "UNKNOWN"
        if role == "data_quality":
            allowed = False
        else:
            allowed = role in ("d0_feature", "d1_feature")
            available = _AVAILABLE_AS_OF.get(role, "UNKNOWN")
        rows.append({
            "column_name": column,
            "column_role": role,
            "available_as_of": available,
            "source_or_formula": _column_source_formula(column, role),
            "allowed_for_future_feature_analysis": allowed,
            "reason": _column_reason(column, role),
        })
    return pd.DataFrame(rows)


def _column_source_formula(column: str, role: str) -> str:
    if column in _KEY_FORMULAS:
        return _KEY_FORMULAS[column]
    return (f"copied unchanged from source v0.2.2 column {column}")


def _column_reason(column: str, role: str) -> str:
    if role == "forbidden_unknown":
        return "产生时点无法确认; 不得默认进入后续因子池, 需人工确认"
    if role in ("label", "outcome_audit"):
        return "包含 D2/D3 未来信息, 不得作为 D1 特征"
    if role == "existing_model_audit":
        return "现有模型结果只用于独立增量审计, 不得成为 v004c 输入因子"
    if role == "identifier":
        return "身份或绝对日期字段, 不作为特征"
    if role in ("provenance", "data_quality"):
        return "溯源/质量元数据, 不作为特征"
    return "D1 时点已知, 允许进入未来因子分析候选"


# ---------------------------------------------------------------------------
# D1 首次断板筛选
# ---------------------------------------------------------------------------
def select_d1_first_break_rows(frame: pd.DataFrame, schema: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """筛选 D1 首次断板观察行, 其余行带 exclusion_reason 输出。"""
    schema = schema or discover_v004c_schema(frame)
    stage_col = schema.get("stage_group_col")
    dsb_col = schema.get("days_since_break_col")

    stage_d0 = None
    if stage_col:
        stage_d0 = frame[stage_col].astype(str).eq(STAGE_D0_VALUE)
    if dsb_col:
        dsb_d0 = pd.to_numeric(frame[dsb_col], errors="coerce").eq(0)
        if stage_d0 is not None and not stage_d0.eq(dsb_d0).all():
            mismatches = int((stage_d0 != dsb_d0).sum())
            raise DatasetValidationError(
                f"stage_group 与 days_since_break 语义不一致: {mismatches} 行")
        stage_d0 = dsb_d0 if stage_d0 is None else stage_d0
    if stage_d0 is None:
        raise DatasetValidationError("缺少 stage_group 或 days_since_break 阶段字段")

    frame = frame.copy()
    frame["_is_d1"] = stage_d0.fillna(False)
    frame["_exclusion_reason"] = ""
    frame.loc[~frame["_is_d1"], "_exclusion_reason"] = "not_d1_first_break_observation"

    d1 = frame[frame["_is_d1"]].copy()
    rest = frame[~frame["_is_d1"]].copy()

    if schema.get("signal_date_col") and schema.get("break_date_col"):
        signal_mismatch = int(
            (d1[schema["signal_date_col"]].astype(str) != d1[schema["break_date_col"]].astype(str)).sum())
        if signal_mismatch:
            raise DatasetValidationError(
                f"D1 行 signal_date != break_date: {signal_mismatch} 行, 阻止冻结")

    def exclude_mask(mask: pd.Series, reason: str) -> None:
        nonlocal d1, rest
        sel = d1[mask]
        if sel.empty:
            return
        sel = sel.copy()
        sel["_exclusion_reason"] = reason
        rest = pd.concat([rest, sel], ignore_index=True)
        d1 = d1[~mask]

    if schema.get("board_streak_col"):
        streak = pd.to_numeric(d1[schema["board_streak_col"]], errors="coerce")
        exclude_mask(~streak.isin(ALLOWED_BOARD_STREAKS), "board_streak_not_two_or_three")
    if schema.get("daily_label_quality_col"):
        exclude_mask(d1[schema["daily_label_quality_col"]].fillna(False).astype(bool) == False,  # noqa: E712
                     "daily_label_quality_failed")
    if schema.get("target_col"):
        exclude_mask(d1[schema["target_col"]].isna(), "target7_missing")
    exclude_mask(d1["event_id"].isna() | (d1["event_id"].astype(str).str.strip() == ""),
                 "event_id_missing")

    d1 = d1.drop(columns=["_is_d1", "_exclusion_reason"]).reset_index(drop=True)
    rest = rest.drop(columns=["_is_d1"]).rename(
        columns={"_exclusion_reason": "exclusion_reason"}).reset_index(drop=True)
    return d1, rest


# ---------------------------------------------------------------------------
# 规范时间字段
# ---------------------------------------------------------------------------
def build_canonical_time_fields(d1: pd.DataFrame, schema: dict | None = None) -> pd.DataFrame:
    schema = schema or discover_v004c_schema(d1)
    out = d1.copy()
    out["d1_break_date"] = d1["break_date"].astype(str)
    out["d1_signal_date"] = d1["signal_date"].astype(str)
    d0_col = schema.get("d0_last_board_date_col")
    out["d0_last_board_date"] = d1[d0_col].astype(str) if (d0_col and d0_col in d1.columns) else None
    d2_col = schema.get("d2_entry_date_col")
    out["d2_entry_date"] = d1[d2_col].astype(str) if (d2_col and d2_col in d1.columns) else None
    d3_col = schema.get("d3_outcome_date_col")
    out["d3_outcome_date"] = d1[d3_col].astype(str) if (d3_col and d3_col in d1.columns) else None
    return out


# ---------------------------------------------------------------------------
# 交易日邻接验证 (冻结硬条件)
# ---------------------------------------------------------------------------
def read_daily_cache_series(cache_dir: str | Path, code: str) -> tuple[list[str], dict[str, Any]]:
    """读取单只股票的未复权日线缓存日期序列。

    严格日期校验: 原始行数 / 有效日期数 / 非法日期数 / 重复日期数;
    任何非法或空白日期、任何重复日期都阻止缓存通过。
    """
    path = Path(cache_dir) / f"{code}_daily.pkl"
    provenance: dict[str, Any] = {"path": path.as_posix(), "exists": path.exists()}
    if not path.exists():
        raise DatasetValidationError(f"日线缓存缺失: {code} -> {path}")
    try:
        with path.open("rb") as handle:
            frame = pd.read_pickle(handle)
    except Exception as exc:
        raise DatasetValidationError(f"日线缓存读取失败: {code}: {exc}") from exc
    if frame is None or frame.empty or "date" not in frame.columns:
        raise DatasetValidationError(f"日线缓存无有效日期列: {code}")
    raw_row_count = int(len(frame))
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    valid = parsed.notna()
    valid_date_count = int(valid.sum())
    invalid_date_count = raw_row_count - valid_date_count
    if invalid_date_count:
        raise DatasetValidationError(
            f"日线缓存含非法日期: {code} 非法日期数={invalid_date_count} "
            f"(示例: {[str(v) for v in frame.loc[~valid, 'date'].head(3).tolist()]})")
    dates_str = parsed.dt.strftime("%Y-%m-%d")
    duplicate_date_count = int(len(dates_str) - int(dates_str.nunique()))
    if duplicate_date_count:
        raise DatasetValidationError(f"日线缓存日期重复: {code} 重复数={duplicate_date_count}")
    series = sorted(dates_str.tolist())
    provenance.update({
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
        "raw_row_count": raw_row_count,
        "valid_date_count": valid_date_count,
        "invalid_date_count": invalid_date_count,
        "duplicate_date_count": duplicate_date_count,
        "min_date": series[0] if series else "",
        "max_date": series[-1] if series else "",
    })
    return series, provenance


def _date_sequence_sha(dates: list[str], through_date: str) -> str:
    """截至 D3 的规范交易日序列 SHA256 (未来缓存追加不影响历史证据复核)。"""
    seq = "\n".join(d for d in dates if d <= through_date)
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def verify_trade_date_adjacency(
    d1: pd.DataFrame,
    cache_dir: str | Path,
    schema: dict | None = None,
) -> dict[str, Any]:
    """逐股验证 D0→D1→D2→D3 为缓存实际交易日序列中的相邻交易日; 生成证据表。"""
    schema = schema or discover_v004c_schema(d1)
    cache_root = Path(cache_dir)
    if not cache_root.exists():
        raise DatasetValidationError(f"日线缓存目录缺失: {cache_dir}")

    code_cache: dict[str, tuple[list[str], dict[str, Any]]] = {}
    for code in sorted(d1["code"].unique()):
        try:
            code_cache[code] = read_daily_cache_series(cache_root, code)
        except DatasetValidationError as exc:
            code_cache[code] = ([], {"path": (cache_root / f"{code}_daily.pkl").as_posix(),
                                     "exists": False, "error": str(exc)})

    per_row = []
    verified = mismatch = not_verified = 0
    for _, row in d1.iterrows():
        dates, prov = code_cache.get(row["code"], ([], {}))
        status = "verified"
        reason = ""
        if not dates:
            status = "not_verified"
            reason = prov.get("error", "cache_unavailable")
        else:
            d0v, d1v, d2v, d3v = (str(row["d0_last_board_date"]), str(row["d1_break_date"]),
                                  str(row["d2_entry_date"]), str(row["d3_outcome_date"]))
            required = [d0v, d1v, d2v, d3v]
            if any(not d or d == "None" for d in required):
                status = "not_verified"
                reason = "required_date_missing"
            elif not all(d in dates for d in required):
                status = "not_verified"
                reason = "required_date_not_in_cache"
            else:
                before_d1 = [x for x in dates if x < d1v]
                after_d1 = [x for x in dates if x > d1v]
                after_d2 = [x for x in dates if x > d2v]
                ok = (before_d1 and before_d1[-1] == d0v
                      and after_d1 and after_d1[0] == d2v
                      and after_d2 and after_d2[0] == d3v)
                if not ok:
                    status = "mismatch"
                    reason = "trade_date_adjacency_violation"
        per_row.append({
            "event_id": row["event_id"], "code": row["code"],
            "d0_last_board_date": row["d0_last_board_date"],
            "d1_break_date": row["d1_break_date"],
            "d2_entry_date": row["d2_entry_date"],
            "d3_outcome_date": row["d3_outcome_date"],
            "daily_cache_relative_path": prov.get("path", ""),
            "daily_cache_sha256": prov.get("sha256", ""),
            "date_sequence_through_d3_sha256": (
                _date_sequence_sha(dates, str(row["d3_outcome_date"])) if dates else ""),
            "adjacency_verified": status == "verified",
            "adjacency_status": status, "adjacency_reason": reason,
        })
        if status == "verified":
            verified += 1
        elif status == "mismatch":
            mismatch += 1
        else:
            not_verified += 1

    return {
        "verified_rows": verified,
        "mismatch_rows": mismatch,
        "not_verified_rows": not_verified,
        "cache_stock_count": int(len(code_cache)),
        "cache_file_count": int(sum(1 for _, p in code_cache.values() if p.get("exists"))),
        "per_row": per_row,
        "cache_files": {code: prov for code, (_, prov) in code_cache.items()},
    }


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def validate_d1_dataset(
    d1: pd.DataFrame,
    lineage: pd.DataFrame,
    schema: dict | None = None,
    daily_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """D1 数据集全部校验; 返回校验结果 (硬性失败置 hard_failures 非空)。"""
    schema = schema or discover_v004c_schema(d1)
    checks: dict[str, Any] = {}
    hard_failures: list[str] = []

    checks["input_rows"] = int(schema.get("row_count", len(d1)))
    checks["selected_d1_rows"] = int(len(d1))
    checks["unique_event_id"] = int(d1["event_id"].nunique())
    checks["unique_signal_dates"] = int(d1["signal_date"].nunique())
    checks["duplicate_event_id_rows"] = int(d1["event_id"].duplicated().sum())
    checks["duplicate_code_break_date_rows"] = int(d1.duplicated(["code", "break_date"]).sum())
    checks["duplicate_code_signal_date_rows"] = int(d1.duplicated(["code", "signal_date"]).sum())
    for name, count in (("duplicate_event_id", checks["duplicate_event_id_rows"]),
                        ("duplicate_code_break_date", checks["duplicate_code_break_date_rows"]),
                        ("duplicate_code_signal_date", checks["duplicate_code_signal_date_rows"])):
        if count:
            hard_failures.append(f"{name}={count}")

    # 严格二元字段验证 (显式映射; 非法值带字段/值/行数/示例)
    binary_ok: dict[str, pd.Series] = {}
    for column in STRICT_BINARY_COLUMNS:
        if column not in d1.columns:
            hard_failures.append(f"binary_field_missing:{column}")
            continue
        try:
            binary_ok[column] = normalize_binary_series(
                d1[column], column_name=column, allow_missing=False)
        except DatasetValidationError as exc:
            detail = str(exc)
            invalid_mask = ~d1[column].isin(_BINARY_TRUE_VALUES | _BINARY_FALSE_VALUES)
            sample_ids = d1.loc[invalid_mask, "event_id"].head(3).tolist()
            sample_codes = d1.loc[invalid_mask, "code"].head(3).tolist()
            hard_failures.append(
                f"binary_field_invalid:{column}|{detail}|示例event_id={sample_ids}|示例code={sample_codes}")

    target = binary_ok.get("target7_daily_d2open_d3high")
    if target is not None:
        checks["target7_positive_count"] = int(target.sum())
        checks["target7_rate"] = float(target.mean()) if len(target) else None
    else:
        checks["target7_positive_count"] = None
        checks["target7_rate"] = None
    if target is None or target.isna().any():
        hard_failures.append("target7_missing_or_invalid_in_selected")
    if "daily_label_quality_ok" in binary_ok:
        lq = binary_ok["daily_label_quality_ok"]
        checks["daily_label_quality_false_count"] = int((~lq).sum())
        if lq.isna().any() or (~lq).sum():
            hard_failures.append("daily_label_quality_false_or_missing_in_selected")
    else:
        checks["daily_label_quality_false_count"] = None
        hard_failures.append("daily_label_quality_missing_in_selected")

    for month in ("2026-06", "2026-07"):
        m = d1[d1["signal_date"].astype(str).str.startswith(month)]
        mt = target.loc[m.index] if target is not None else pd.Series(dtype=bool)
        checks[f"{month}_rows"] = int(len(m))
        checks[f"{month}_positives"] = int(mt.sum()) if len(mt) else 0
        checks[f"{month}_rate"] = float(mt.mean()) if len(mt) else None

    streak_col = schema.get("board_streak_col") or "board_streak_before_break"
    streak = pd.to_numeric(d1.get(streak_col), errors="coerce")
    for s in ALLOWED_BOARD_STREAKS:
        m = streak.eq(s)
        mt = target[m] if target is not None else pd.Series(dtype=bool)
        checks[f"board_streak_{s}_rows"] = int(m.sum())
        checks[f"board_streak_{s}_positives"] = int(mt.sum()) if len(mt) else 0
        checks[f"board_streak_{s}_rate"] = float(mt.mean()) if m.sum() else None
    checks["board_streak_other_count"] = int(streak[~streak.isin(ALLOWED_BOARD_STREAKS)].count())
    if checks["board_streak_other_count"]:
        hard_failures.append(f"board_streak_other_count={checks['board_streak_other_count']}")

    time_fail_rows = 0
    for _, row in d1.iterrows():
        d0v, d1v, d2v, d3v = (row["d0_last_board_date"], row["d1_break_date"],
                              row["d2_entry_date"], row["d3_outcome_date"])
        if not (d0v and d1v and d2v and d3v):
            time_fail_rows += 1
            continue
        if not (str(d0v) < str(d1v) < str(d2v) < str(d3v)):
            time_fail_rows += 1
    checks["time_relation_fail_rows"] = time_fail_rows
    checks["signal_equals_break_fail_rows"] = int(
        (d1["signal_date"].astype(str) != d1["break_date"].astype(str)).sum())
    if time_fail_rows or checks["signal_equals_break_fail_rows"]:
        hard_failures.append("time_relation_or_signal_break_mismatch")

    if daily_cache_dir is None:
        hard_failures.append("daily_cache_dir_required")
        checks["d2_d3_adjacency"] = {"verified_rows": 0, "mismatch_rows": len(d1),
                                     "not_verified_rows": len(d1), "note": "cache_dir 未提供"}
    else:
        adjacency = verify_trade_date_adjacency(d1, daily_cache_dir, schema)
        checks["d2_d3_adjacency"] = {k: adjacency[k] for k in
                                     ("verified_rows", "mismatch_rows", "not_verified_rows",
                                      "cache_stock_count", "cache_file_count")}
        checks["adjacency_per_row"] = adjacency["per_row"]
        checks["daily_cache_provenance"] = adjacency
        if (adjacency["verified_rows"] != len(d1)
                or adjacency["mismatch_rows"]
                or adjacency["not_verified_rows"]):
            hard_failures.append(
                f"trade_date_adjacency: verified={adjacency['verified_rows']} "
                f"mismatch={adjacency['mismatch_rows']} not_verified={adjacency['not_verified_rows']}")
            for code, prov in adjacency.get("cache_files", {}).items():
                if not prov.get("exists") or prov.get("error"):
                    hard_failures.append(f"cache_error:{code}:{prov.get('error', 'cache_unavailable')}")

    d2_open = pd.to_numeric(d1.get(schema.get("d2_open_col") or "d2_open_daily"), errors="coerce")
    d3_high = pd.to_numeric(d1.get(schema.get("d3_high_col") or "d3_high_daily"), errors="coerce")
    d3_close = pd.to_numeric(d1.get(schema.get("d3_close_col") or "d3_close_daily"), errors="coerce")
    bad_price = 0
    for name, series in (("d2_open", d2_open), ("d3_high", d3_high), ("d3_close", d3_close)):
        invalid = series.isna() | ~np.isfinite(series) | (series <= 0)
        checks[f"{name}_invalid_count"] = int(invalid.sum())
        bad_price += int(invalid.sum())
    if bad_price:
        hard_failures.append(f"invalid_label_price_count={bad_price}")

    stored = pd.to_numeric(d1.get("daily_d2open_to_d3high_return"), errors="coerce")
    recomputed = d3_high / d2_open - 1.0
    diff = (recomputed - stored).abs()
    checks["max_return_diff"] = None if diff.isna().all() else float(diff.max())
    recomputed_target = recomputed >= TARGET_THRESHOLD
    if target is not None:
        mismatch_mask = recomputed_target.fillna(False) != target.fillna(False)
        checks["target7_recompute_mismatch_count"] = int(mismatch_mask.sum())
        checks["target7_recompute_mismatch_rows"] = d1.loc[mismatch_mask, "event_id"].tolist()
    else:
        checks["target7_recompute_mismatch_count"] = 0
        checks["target7_recompute_mismatch_rows"] = []
    if checks["target7_recompute_mismatch_count"]:
        hard_failures.append("target7_recompute_mismatch")

    leak = lineage[
        lineage["column_role"].isin(("label", "outcome_audit", "existing_model_audit"))
        & (lineage["allowed_for_future_feature_analysis"] == True)  # noqa: E712
    ]
    checks["leakage_violation_columns"] = leak["column_name"].tolist()
    checks["forbidden_unknown_columns"] = lineage.loc[
        lineage["column_role"] == "forbidden_unknown", "column_name"].tolist()
    if leak["column_name"].nunique():
        hard_failures.append("leakage_violation_in_lineage")

    checks["hard_failures"] = hard_failures
    checks["frozen"] = not hard_failures
    return checks


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _render_review(validation: dict[str, Any], manifest: dict[str, Any]) -> str:
    """由结构化统计自动生成 review (核心数字不手工填写)。"""
    v = validation
    adj = v.get("d2_d3_adjacency", {})
    lines = [
        "# v004c 阶段1 — D1 首次断板训练数据冻结与审计报告 (自动生成)",
        "",
        f"- 数据集版本: {manifest['dataset_version']}",
        f"- 来源数据集: {manifest['source_dataset_version']} (已验证源 manifest)",
        f"- 候选定义: {manifest['candidate_definition']}",
        f"- 标签定义: {manifest['label_definition']}",
        f"- adjustment: {manifest['adjustment']}",
        f"- 生成时间: {manifest['generated_at']} ({manifest['timezone']})",
        f"- git root: {manifest.get('git_root')}",
        f"- git HEAD: {manifest.get('git_head')}",
        "",
        "## 统一交易时序",
        "",
        "D0 = 最后一个成功连板日; D1 = 首次断板日 (旧数据 stage_group=='d0' / days_since_break==0);",
        "D2 = D1 后下一有效交易日 (早盘买入); D3 = D2 后下一有效交易日 (盘中检验 +7%)。",
        "",
        "## git_status_after 语义",
        "",
        "git_status_after = 本次构建所有非 manifest 输出文件写完后的 Git 状态",
        f"(git_dirty_before={manifest.get('git_dirty_before')}, "
        f"git_dirty_after={manifest.get('git_dirty_after')})。",
        "",
        "## 冻结统计",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 输入行数 | {v.get('input_rows')} |",
        f"| D1 输出行数 | {v.get('selected_d1_rows')} |",
        f"| 唯一 event_id | {v.get('unique_event_id')} |",
        f"| 唯一信号日 | {v.get('unique_signal_dates')} |",
        f"| Target7 正样本 | {v.get('target7_positive_count')} ({v.get('target7_rate'):.6f}) |",
        f"| 六月 / 七月 | {v.get('2026-06_rows')}/{v.get('2026-07_rows')} 行, "
        f"正样本 {v.get('2026-06_positives')}/{v.get('2026-07_positives')} |",
        f"| 二板 / 三板 | {v.get('board_streak_2_rows')}/{v.get('board_streak_3_rows')} 行, "
        f"正样本 {v.get('board_streak_2_positives')}/{v.get('board_streak_3_positives')} |",
        f"| 重复 (event/code-break/code-signal) | {v.get('duplicate_event_id_rows')}/"
        f"{v.get('duplicate_code_break_date_rows')}/{v.get('duplicate_code_signal_date_rows')} |",
        f"| 时间关系失败行 | {v.get('time_relation_fail_rows')} |",
        f"| Target7 重算不一致 | {v.get('target7_recompute_mismatch_count')} (最大收益差异 {v.get('max_return_diff')}) |",
        f"| 交易日邻接 verified/mismatch/not_verified | {adj.get('verified_rows')}/"
        f"{adj.get('mismatch_rows')}/{adj.get('not_verified_rows')} |",
        f"| 日线缓存股票数 / 文件数 | {adj.get('cache_stock_count')}/{adj.get('cache_file_count')} |",
        f"| 泄漏违规列 | {len(v.get('leakage_violation_columns', []))} |",
        f"| forbidden_unknown 列 | {len(v.get('forbidden_unknown_columns', []))} |",
        f"| 冻结状态 | {'FROZEN' if manifest.get('frozen') else 'NOT_FROZEN'} |",
        "",
        "## 声明",
        "",
        "本阶段只冻结 D1 训练数据; 未筛选因子、未训练模型、未运行 walk-forward、",
        "未修改 v004a/v005.1 冻结逻辑; 未自动 commit 或 push。",
        "",
    ]
    return "\n".join(lines)


def write_data_package(
    *,
    snapshot: pd.DataFrame,
    training: pd.DataFrame,
    existing_model_audit: pd.DataFrame,
    excluded: pd.DataFrame,
    data_quality: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    lineage: pd.DataFrame,
    adjacency_evidence: pd.DataFrame,
    validation: dict[str, Any],
    out_dir: str | Path,
    input_file: str | Path,
    input_manifest: str | Path | None,
    source_manifest: dict,
    git_provenance_before: GitProvenance,
    git_root: Path,
    output_dir_gate: dict[str, Any],
    daily_cache_dir: str | Path | None,
    daily_cache_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """阶段1: 写入数据文件 (CSV + git_head + git_status_before) 到临时目录。

    git_status_after 与最终 manifest 由 finalize_package 在原子 rename 后写入,
    保证 git_status_after 反映全部非 manifest 输出文件在正式位置的 Git 状态。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(frame: pd.DataFrame, name: str) -> Path:
        path = out / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    data_paths = {
        "v004c_d1_snapshot_v001.csv": save(snapshot, "v004c_d1_snapshot_v001.csv"),
        "v004c_training_d1_v001.csv": save(training, "v004c_training_d1_v001.csv"),
        "v004c_d1_existing_model_audit_v001.csv": save(existing_model_audit, "v004c_d1_existing_model_audit_v001.csv"),
        "v004c_d1_excluded_rows_v001.csv": save(excluded, "v004c_d1_excluded_rows_v001.csv"),
        "v004c_d1_data_quality.csv": save(data_quality, "v004c_d1_data_quality.csv"),
        "v004c_d1_candidate_audit.csv": save(candidate_audit, "v004c_d1_candidate_audit.csv"),
        "v004c_d1_column_lineage.csv": save(lineage, "v004c_d1_column_lineage.csv"),
        "v004c_d1_trade_date_adjacency_evidence.csv": save(adjacency_evidence,
                                                           "v004c_d1_trade_date_adjacency_evidence.csv"),
    }

    git_paths = {}
    for name, content in (("git_head.txt", git_provenance_before.git_head),
                          ("git_status_before.txt", git_provenance_before.git_status)):
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        git_paths[name] = path

    input_meta = {
        "path": _repo_relative_posix(input_file, git_root),
        "bytes": int(Path(input_file).stat().st_size),
        "sha256": _sha256(input_file),
    }
    input_manifest_meta = None
    if input_manifest and Path(input_manifest).exists():
        input_manifest_meta = {
            "path": _repo_relative_posix(input_manifest, git_root),
            "bytes": int(Path(input_manifest).stat().st_size),
            "sha256": _sha256(input_manifest),
        }

    cache_meta = None
    if daily_cache_dir is not None and daily_cache_provenance:
        cache_meta = {
            "daily_cache_dir": str(daily_cache_dir),
            "stock_count": int(daily_cache_provenance.get("cache_stock_count", 0)),
            "file_count": int(daily_cache_provenance.get("cache_file_count", 0)),
            "files": daily_cache_provenance.get("cache_files", {}),
        }

    manifest_draft = {
        "dataset_version": DATASET_VERSION,
        "source_dataset_version": source_manifest.get("dataset_version"),
        "source_candidate_definition_version": source_manifest.get("candidate_definition_version"),
        "source_label_definition_version": source_manifest.get("label_definition_version"),
        "source_adjustment": source_manifest.get("adjustment"),
        "candidate_definition": CANDIDATE_DEFINITION,
        "label_definition": LABEL_DEFINITION,
        "adjustment": ADJUSTMENT,
        "research_only": True,
        "deployment_status": "research_only",
        "timezone": TIMEZONE,
        "git_root": str(git_root),
        "git_head": git_provenance_before.git_head,
        "git_status_before": git_provenance_before.git_status,
        "git_dirty_before": bool(git_provenance_before.git_dirty),
        "output_dir_gate": output_dir_gate,
        "input_file": input_meta,
        "input_manifest": input_manifest_meta,
        "daily_cache": cache_meta,
        "row_stats": {
            "input_rows": validation.get("input_rows"),
            "selected_d1_rows": validation.get("selected_d1_rows"),
            "unique_event_id": validation.get("unique_event_id"),
            "unique_signal_dates": validation.get("unique_signal_dates"),
            "target7_positive_count": validation.get("target7_positive_count"),
            "target7_rate": validation.get("target7_rate"),
            "signal_date_min": str(snapshot["signal_date"].min()),
            "signal_date_max": str(snapshot["signal_date"].max()),
        },
        "column_role_counts": lineage["column_role"].value_counts().to_dict(),
        "validation": {k: v for k, v in validation.items()
                       if k not in ("hard_failures", "target7_recompute_mismatch_rows",
                                    "adjacency_per_row", "daily_cache_provenance")},
        "hard_failures": validation.get("hard_failures", []),
        "frozen": bool(validation.get("frozen", False)),
    }
    return {"paths": {**data_paths, **git_paths}, "manifest_draft": manifest_draft}


def finalize_package(
    *,
    out_dir: str | Path,
    manifest_draft: dict[str, Any],
    validation: dict[str, Any],
    git_provenance_after: GitProvenance | None,
    git_root: Path,
) -> dict[str, Any]:
    """阶段2 (原子 rename 后): 采集 git_status_after, 写 git_status_after.txt / review / 最终 manifest。

    保证 git_status_after = 全部非 manifest 输出文件在正式位置写完后 的 Git 状态,
    且 manifest 记录后不再修改任何被记录的非 manifest 文件。
    """
    out = Path(out_dir)
    if git_provenance_after is None:
        git_provenance_after = collect_git_provenance(git_root)
    manifest = dict(manifest_draft)
    manifest["generated_at"] = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["git_status_after"] = git_provenance_after.git_status
    manifest["git_dirty_after"] = bool(git_provenance_after.git_dirty)

    after_path = out / "git_status_after.txt"
    with after_path.open("w", encoding="utf-8") as handle:
        handle.write(git_provenance_after.git_status)

    review_path = out / "v004c_d1_dataset_review.md"
    with review_path.open("w", encoding="utf-8") as handle:
        handle.write(_render_review(validation, manifest))

    output_meta = {}
    all_outputs = {name: Path(out) / name for name in sorted(p.name for p in out.iterdir() if p.is_file())}
    for name, path in all_outputs.items():
        if name == "v004c_d1_data_manifest.json":
            continue
        output_meta[name] = {
            "path": _repo_relative_posix(path, git_root),
            "rows": None if name.endswith((".txt", ".md")) else _count_csv_rows(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "role": _output_role(name),
        }
    manifest["output_files"] = output_meta

    manifest_path = out / "v004c_d1_data_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"paths": all_outputs, "manifest": manifest, "manifest_path": manifest_path}


def _output_role(name: str) -> str:
    if name == "v004c_training_d1_v001.csv":
        return "training"
    if name == "v004c_d1_snapshot_v001.csv":
        return "snapshot"
    if name == "v004c_d1_existing_model_audit_v001.csv":
        return "existing_model_audit"
    if name == "v004c_d1_excluded_rows_v001.csv":
        return "excluded_audit"
    if name == "v004c_d1_data_quality.csv":
        return "quality_audit"
    if name == "v004c_d1_candidate_audit.csv":
        return "candidate_audit"
    if name == "v004c_d1_column_lineage.csv":
        return "lineage"
    if name == "v004c_d1_trade_date_adjacency_evidence.csv":
        return "provenance"
    if name == "v004c_d1_dataset_review.md":
        return "review"
    if name.startswith("git_"):
        return "provenance"
    return "provenance"


def _write_failure_audit(
    output_dir: Path,
    *,
    failure_stage: str,
    exception: Exception,
    hard_failures: list[str],
    input_file: str | Path,
    input_manifest: str | Path | None,
    daily_cache_dir: str | Path | None,
    git_head: str,
) -> None:
    """将失败审计写入 <output_dir>_failed_<timestamp>/ (不污染正式输出目录)。"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    failed_dir = Path(f"{output_dir}_failed_{timestamp}")
    failed_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "failure_stage": failure_stage,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "hard_failures": hard_failures,
        "input_file": str(input_file),
        "input_manifest": str(input_manifest),
        "daily_cache_dir": str(daily_cache_dir),
        "git_head": git_head,
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (failed_dir / "failure_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------
def run_v004c_d1_dataset_build(
    *,
    input_file: str | Path,
    input_manifest: str | Path | None = None,
    output_dir: str | Path,
    daily_cache_dir: str | Path,
    expected_stats: dict[str, int] | None = None,
    git_provenance_before: GitProvenance | None = None,
    git_provenance_after: GitProvenance | None = None,
) -> dict[str, Any]:
    """v004c 阶段1 完整构建流程 (fail closed; 临时目录构建 + 原子 rename)。"""
    if daily_cache_dir is None:
        raise DatasetValidationError("daily_cache_dir 必须提供 (交易日邻接验证为冻结硬条件)")

    target_dir = Path(output_dir)
    git_provenance_before = git_provenance_before or collect_git_provenance(Path.cwd())
    git_root = git_provenance_before.git_root

    # 输出目录门禁: 不存在→创建允许; 空→允许; 含任何文件/子目录→fail closed
    preexisting = []
    if target_dir.exists():
        preexisting = sorted(str(p) for p in target_dir.iterdir())
    if preexisting:
        raise DatasetValidationError(
            f"输出目录非空, 阻止冻结: {output_dir} (含 {len(preexisting)} 项; "
            f"不得自动删除未知文件, 请显式删除旧目录后重建)")
    output_dir_gate = {
        "output_dir_was_created": not target_dir.exists(),
        "preexisting_entry_count": len(preexisting),
        "unmanaged_output_files": [],
    }

    frame, source_manifest = read_v004c_source(input_file, input_manifest)
    source_manifest = validate_source_manifest(source_manifest, input_file)
    schema = discover_v004c_schema(frame)

    d1, excluded = select_d1_first_break_rows(frame, schema)
    d1 = build_canonical_time_fields(d1, schema)
    lineage = classify_column_lineage(d1, schema)

    if expected_stats:
        actual = {
            "input_rows": len(frame),
            "d1_rows": int((frame.get("stage_group", "").astype(str) == STAGE_D0_VALUE).sum()),
            "post_rows": int((frame.get("stage_group", "").astype(str) == STAGE_POST_VALUE).sum()),
            "d0_target7": int(pd.to_numeric(
                frame.loc[frame.get("stage_group", "").astype(str) == STAGE_D0_VALUE,
                          schema.get("target_col") or "target7_daily_d2open_d3high"],
                errors="coerce").fillna(False).astype(bool).sum()),
            "signal_dates": int(d1["signal_date"].nunique()),
        }
        mismatches = [f"{key}: expected={expected} actual={actual.get(key)}"
                      for key, expected in expected_stats.items()
                      if actual.get(key) is not None and actual.get(key) != int(expected)]
        if mismatches:
            raise DatasetValidationError("参考统计不一致, 阻止冻结: " + "; ".join(mismatches))

    validation = validate_d1_dataset(d1, lineage, schema, daily_cache_dir=daily_cache_dir)
    hard_failures = validation["hard_failures"]
    if hard_failures:
        try:
            _write_failure_audit(
                target_dir, failure_stage="validation",
                exception=DatasetValidationError("; ".join(hard_failures)),
                hard_failures=hard_failures,
                input_file=input_file, input_manifest=input_manifest,
                daily_cache_dir=daily_cache_dir,
                git_head=git_provenance_before.git_head)
        except Exception:
            pass
        raise DatasetValidationError(
            "D1 数据集校验失败, 阻止冻结: " + "; ".join(hard_failures))

    # 临时目录构建 (成功前不污染正式输出目录)
    tmp_dir = Path(f"{target_dir}.tmp_{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        allowed_roles = ("identifier", "d0_feature", "d1_feature", "label", "data_quality", "provenance")
        training_cols = lineage.loc[lineage["column_role"].isin(allowed_roles), "column_name"].tolist()
        training = d1[[c for c in training_cols if c in d1.columns]].copy()

        model_cols = lineage.loc[lineage["column_role"] == "existing_model_audit", "column_name"].tolist()
        existing_model_audit = d1[["event_id", "code", "name", "signal_date", "break_date"]
                                  + [c for c in model_cols if c in d1.columns]].copy()

        snapshot = d1.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)
        training = training.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)
        existing_model_audit = existing_model_audit.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)

        data_quality_rows = [
            {"metric": k, "value": (v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False))}
            for k, v in validation.items()
            if k not in ("hard_failures", "adjacency_per_row", "daily_cache_provenance",
                         "target7_recompute_mismatch_rows")
        ]
        data_quality_rows.append({"metric": "hard_failures",
                                  "value": json.dumps(hard_failures, ensure_ascii=False)})
        data_quality = pd.DataFrame(data_quality_rows)

        target = pd.to_numeric(d1.get(schema.get("target_col") or "target7_daily_d2open_d3high"), errors="coerce")
        d2_open = pd.to_numeric(d1.get(schema.get("d2_open_col") or "d2_open_daily"), errors="coerce")
        d3_high = pd.to_numeric(d1.get(schema.get("d3_high_col") or "d3_high_daily"), errors="coerce")
        stored = pd.to_numeric(d1.get("daily_d2open_to_d3high_return"), errors="coerce")
        recomputed = d3_high / d2_open - 1.0
        adj_map = {r["event_id"]: r for r in validation.get("adjacency_per_row", [])}
        cand_rows = []
        for _, row in d1.iterrows():
            adj = adj_map.get(row["event_id"], {})
            cand_rows.append({
                "event_id": row["event_id"], "code": row["code"], "signal_date": row["signal_date"],
                "d0_last_board_date": row["d0_last_board_date"], "d1_break_date": row["d1_break_date"],
                "d2_entry_date": row["d2_entry_date"], "d3_outcome_date": row["d3_outcome_date"],
                "signal_equals_break": str(row["signal_date"]) == str(row["break_date"]),
                "time_order_ok": bool(row["d0_last_board_date"] and row["d2_entry_date"] and row["d3_outcome_date"]
                                      and str(row["d0_last_board_date"]) < str(row["d1_break_date"])
                                      < str(row["d2_entry_date"]) < str(row["d3_outcome_date"])),
                "adjacency_status": adj.get("adjacency_status", ""),
                "adjacency_reason": adj.get("adjacency_reason", ""),
                "stored_return": stored.get(row.name, None),
                "recomputed_return": recomputed.get(row.name, None),
                "target7": bool(target.get(row.name, False)),
            })
        candidate_audit = pd.DataFrame(cand_rows)
        adjacency_evidence = pd.DataFrame(validation.get("adjacency_per_row", []))

        excluded_out = excluded[["event_id", "code", "signal_date", "break_date",
                                 "stage_group", "days_since_break", "exclusion_reason"]].copy()
        excluded_out = excluded_out.sort_values(["signal_date", "code"]).reset_index(drop=True)

        data_result = write_data_package(
            snapshot=snapshot,
            training=training,
            existing_model_audit=existing_model_audit,
            excluded=excluded_out,
            data_quality=data_quality,
            candidate_audit=candidate_audit,
            lineage=lineage,
            adjacency_evidence=adjacency_evidence,
            validation=validation,
            out_dir=tmp_dir,
            input_file=input_file,
            input_manifest=input_manifest,
            source_manifest=source_manifest,
            git_provenance_before=git_provenance_before,
            git_root=git_root,
            output_dir_gate=output_dir_gate,
            daily_cache_dir=daily_cache_dir,
            daily_cache_provenance=validation.get("daily_cache_provenance"),
        )
        data_result["manifest_draft"]  # noqa: B018  (草稿在 finalize 中补全)
        result = None
    except Exception as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            _write_failure_audit(
                target_dir, failure_stage="build",
                exception=exc,
                hard_failures=validation.get("hard_failures", []) if "validation" in locals() else [],
                input_file=input_file, input_manifest=input_manifest,
                daily_cache_dir=daily_cache_dir,
                git_head=git_provenance_before.git_head)
        except Exception:
            pass
        raise

    # 原子 rename (目标为空目录时先移除), 然后 finalize (采集 git_status_after + 最终 manifest)
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
        validation=validation,
        git_provenance_after=git_provenance_after,
        git_root=git_root,
    )
    return result

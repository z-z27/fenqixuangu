"""Two-phase audit of the preregistered pre-break board-day seal-path family.

Phase A is deliberately blind to outcomes and frozen S2 scores/ranks.  It uses
only candidate identity/dates plus the existing historical limit-up cache.  It
writes and hashes the Phase-A decision before Phase B is allowed to load any
outcome.  Daily-derived limit-up rows never receive inferred time/open-board
semantics: their seal-path inputs remain missing.

No model is fitted, no feature/window/threshold is searched, and August is not
part of the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TASK_NAME = "v004c_pre_break_board_day_seal_path_information_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
DEVELOPMENT_START = "2026-05-06"
DEVELOPMENT_END = "2026-07-29"
TRAINING_ASOF = "2026-08-01"
AUGUST_HOLDOUT_STATUS = "CONSUMED"

BRIDGE_RELATIVE_PATH = (
    "reports/research/v004c_stage1_18f_bridge_analysis_v001_20260506_20260731/"
    "v004c_stage1_18f_bridge_candidates.csv"
)
POOL_CACHE_RELATIVE_DIR = "data/cache/limit_ups"
DAILY_CACHE_RELATIVE_DIR = "data/cache/daily"
POOL_LINEAGE_START = "2026-04-01"

FIXED_FIELDS = (
    "F1_D0_FINAL_SEAL_MINUTE",
    "F2_D0_OPEN_BOARD_COUNT",
    "F3_D0_RESEAL_DELAY_MINUTE",
    "F4_FINAL_SEAL_DETERIORATION",
    "F5_OPEN_BOARD_DETERIORATION",
)
FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "F1_D0_FINAL_SEAL_MINUTE": {
        "formula": "minutes(D0.final_limit_up_time - 09:30)",
        "parents": "D0.final_limit_up_time",
        "quality_direction": "LOWER_IS_STRONGER",
    },
    "F2_D0_OPEN_BOARD_COUNT": {
        "formula": "D0.open_board_count",
        "parents": "D0.open_board_count",
        "quality_direction": "LOWER_IS_STRONGER",
    },
    "F3_D0_RESEAL_DELAY_MINUTE": {
        "formula": "minutes(D0.final_limit_up_time - D0.limit_up_time)",
        "parents": "D0.final_limit_up_time;D0.limit_up_time",
        "quality_direction": "LOWER_IS_STRONGER",
    },
    "F4_FINAL_SEAL_DETERIORATION": {
        "formula": "D0 final-seal minute - PREV_BOARD_DAY final-seal minute",
        "parents": "D0.final_limit_up_time;PREV_BOARD_DAY.final_limit_up_time",
        "quality_direction": "LOWER_IS_STRONGER",
    },
    "F5_OPEN_BOARD_DETERIORATION": {
        "formula": "D0 open_board_count - PREV_BOARD_DAY open_board_count",
        "parents": "D0.open_board_count;PREV_BOARD_DAY.open_board_count",
        "quality_direction": "LOWER_IS_STRONGER",
    },
}

VALID_POOL_SOURCE = "akshare_zt_pool_em"
DERIVED_FALLBACK_SOURCE = "daily_limitup_derived"
OVERALL_COVERAGE_GATE = 0.80
MONTH_COVERAGE_GATE = 0.70
BOARD_COVERAGE_GATE = 0.65
FAMILY_READY_FIELD_MINIMUM = 3

PERMUTATION_SEED = 20260905
PERMUTATION_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_REPETITIONS = 10_000

# Fixed interpretation gates used only if Phase A authorizes Phase B.
MATERIAL_DEVIATION = 0.07
HEAD_MATERIAL_DEVIATION = 0.05
RESCUE_MATERIAL_DEVIATION = 0.05
SELECTION_ADJUSTED_P_GATE = 0.10
MIN_PAIR_DATES = 10

OUTPUT_FILENAMES = (
    "v004c_prebreak_seal_path_source_lineage_v001.csv",
    "v004c_prebreak_seal_path_coverage_v001.csv",
    "v004c_prebreak_seal_path_row_eligibility_v001.csv",
    "v004c_prebreak_seal_path_phase_a_lock_v001.csv",
    "v004c_prebreak_seal_path_values_v001.csv",
    "v004c_prebreak_seal_path_pair_concordance_v001.csv",
    "v004c_prebreak_seal_path_monthly_stability_v001.csv",
    "v004c_prebreak_seal_path_rank2_6_v001.csv",
    "v004c_prebreak_seal_path_s2_error_rescue_v001.csv",
    "v004c_prebreak_seal_path_selection_permutation_v001.csv",
    "v004c_prebreak_seal_path_bootstrap_v001.csv",
    "v004c_prebreak_seal_path_missingness_v001.csv",
    "v004c_pre_break_board_day_seal_path_information_review_v001.md",
)

PHASE_A_USECOLS = (
    "event_id",
    "code",
    "signal_date",
    "d0_date",
    "d1_date",
    "board_group",
    "label_available_date",
)
FORBIDDEN_PHASE_A_COLUMNS = frozenset(
    {
        "target7",
        "positive_non_target",
        "loss",
        "severe_loss",
        "raw_repair_return",
        "capped_return_7",
        "stage1_score",
        "stage1_rank",
        "s2_score",
        "s2_rank",
    }
)


@dataclass(frozen=True)
class AuditState:
    data_state: str
    information_state: str
    broader_information_limitation: str
    stage1_research_state: str
    next_action: str
    phase_b_run: bool


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _month_label(value: str) -> str:
    return {"2026-05": "MAY", "2026-06": "JUNE", "2026-07": "JULY"}.get(
        str(value)[:7], str(value)[:7]
    )


def _normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    return digits[-6:].zfill(6)


def parse_pool_time(value: Any) -> tuple[float, float, str]:
    """Return minutes after 09:30, seconds since midnight, and parse status."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return math.nan, math.nan, "MISSING"
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"[^0-9]", "", text)
    if len(text) == 5:
        text = "0" + text
    if len(text) != 6:
        return math.nan, math.nan, "UNPARSEABLE"
    hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:6])
    if hour > 23 or minute > 59 or second > 59:
        return math.nan, math.nan, "UNPARSEABLE"
    seconds = float(hour * 3600 + minute * 60 + second)
    return (seconds - (9 * 3600 + 30 * 60)) / 60.0, seconds, "VALID"


def _safe_numeric(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else math.nan


def load_phase_a_population(root: str | Path) -> pd.DataFrame:
    """Load only identity/date columns; never materialize outcomes or S2 scores."""
    root_path = Path(root).resolve()
    path = root_path / BRIDGE_RELATIVE_PATH
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=list(PHASE_A_USECOLS),
        dtype={"event_id": str, "code": str},
    )
    if FORBIDDEN_PHASE_A_COLUMNS.intersection(frame.columns):
        raise RuntimeError("FATAL: Phase A loaded a forbidden outcome/model column")
    for column in ("signal_date", "d0_date", "d1_date", "label_available_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.strftime("%Y-%m-%d")
    frame["code"] = frame["code"].map(_normalize_code)
    frame = frame[
        frame["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & frame["label_available_date"].lt(TRAINING_ASOF)
    ].copy()
    frame["month"] = frame["signal_date"].map(_month_label)
    frame = frame.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)
    if (len(frame), frame["signal_date"].nunique()) != (485, 60):
        raise RuntimeError(
            f"FATAL: frozen mature population parity failed: {len(frame)} rows / "
            f"{frame['signal_date'].nunique()} dates"
        )
    if frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in frozen Phase A population")
    if frame["signal_date"].ge(TRAINING_ASOF).any():
        raise RuntimeError("FATAL: August signal date reached Phase A")
    return frame


def load_pool_cache(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    root_path = Path(root).resolve()
    pool_dir = root_path / POOL_CACHE_RELATIVE_DIR
    files = sorted(pool_dir.glob("*_limitups.pkl"))
    if not files:
        raise FileNotFoundError(pool_dir)
    all_rows: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    trade_dates: list[str] = []
    keep_columns = (
        "trade_date",
        "code",
        "limit_up_time",
        "final_limit_up_time",
        "open_board_count",
        "seal_amount",
        "amount",
        "consecutive_limit_up_count",
        "source",
    )
    for path in files:
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})_limitups\.pkl$", path.name)
        if not date_match:
            continue
        date_text = date_match.group(1)
        if date_text < POOL_LINEAGE_START or date_text > DEVELOPMENT_END:
            continue
        trade_dates.append(date_text)
        frame = pd.read_pickle(path)
        for column in keep_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[list(keep_columns)].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["code"] = frame["code"].map(_normalize_code)
        frame["cache_path"] = str(path.relative_to(root_path)).replace("\\", "/")
        frame["cache_file_date"] = date_text
        source_counts = frame["source"].fillna("UNKNOWN").astype(str).value_counts()
        for source, count in source_counts.items():
            part = frame[frame["source"].fillna("UNKNOWN").astype(str).eq(source)]
            file_rows.append(
                {
                    "record_type": "CACHE_FILE_SOURCE",
                    "cache_date": date_text,
                    "cache_path": str(path.relative_to(root_path)).replace("\\", "/"),
                    "source": source,
                    "rows": int(count),
                    "limit_up_time_valid_rows": int(
                        part["limit_up_time"].fillna("").astype(str).str.strip().ne("").sum()
                    ),
                    "final_limit_up_time_valid_rows": int(
                        part["final_limit_up_time"].fillna("").astype(str).str.strip().ne("").sum()
                    ),
                    "open_board_count_valid_rows": int(
                        pd.to_numeric(part["open_board_count"], errors="coerce").notna().sum()
                    ),
                    "seal_amount_valid_rows": int(
                        pd.to_numeric(part["seal_amount"], errors="coerce").notna().sum()
                    ),
                    "amount_valid_rows": int(
                        pd.to_numeric(part["amount"], errors="coerce").notna().sum()
                    ),
                    "field_semantics": (
                        "HISTORICAL_POOL_EXACT"
                        if source == VALID_POOL_SOURCE
                        else "DAILY_DERIVED_NO_SEAL_PATH"
                        if source == DERIVED_FALLBACK_SOURCE
                        else "UNKNOWN_SOURCE_SEMANTICS"
                    ),
                    "eligible_for_f1_f5": "YES" if source == VALID_POOL_SOURCE else "NO",
                    "d1_close_known": "YES",
                    "uses_d1_d2_d3_outcome": "NO",
                    "notes": "",
                }
            )
        all_rows.append(frame)
    pool = pd.concat(all_rows, ignore_index=True)
    pool["duplicate_date_code"] = pool.duplicated(["trade_date", "code"], keep=False)
    lineage = pd.DataFrame(file_rows).sort_values(
        ["cache_date", "source"], kind="mergesort"
    ).reset_index(drop=True)
    return pool, lineage, sorted(set(trade_dates))


def _pool_lookup(pool: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pool.sort_values(["trade_date", "code", "cache_path"], kind="mergesort").to_dict("records"):
        result.setdefault((str(row["trade_date"]), str(row["code"])), []).append(row)
    return result


def _previous_trade_date(trade_dates: Sequence[str], d0_date: str) -> str:
    eligible = [value for value in trade_dates if value < d0_date]
    return eligible[-1] if eligible else ""


def build_prev_board_day_map(
    root: str | Path, population: pd.DataFrame
) -> dict[str, dict[str, str]]:
    """Recover PREV_BOARD_DAY from the stock's own cached daily chronology.

    Only the date/source columns are read.  Values on or after D0 are never
    used to choose the previous board day.
    """
    root_path = Path(root).resolve()
    result: dict[str, dict[str, str]] = {}
    cached_dates: dict[str, tuple[list[str], dict[str, str], str]] = {}
    for event in population.to_dict("records"):
        code = str(event["code"])
        if code not in cached_dates:
            path = root_path / DAILY_CACHE_RELATIVE_DIR / f"{code}_daily.pkl"
            if not path.is_file():
                cached_dates[code] = ([], {}, str(path.relative_to(root_path)).replace("\\", "/"))
            else:
                frame = pd.read_pickle(path)
                if "date" not in frame.columns:
                    dates: list[str] = []
                    source_by_date: dict[str, str] = {}
                else:
                    normalized = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    dates = sorted(normalized.dropna().unique().tolist())
                    if "source" in frame.columns:
                        source_by_date = {
                            str(date): str(source)
                            for date, source in zip(normalized, frame["source"], strict=False)
                            if pd.notna(date)
                        }
                    else:
                        source_by_date = {}
                cached_dates[code] = (
                    dates,
                    source_by_date,
                    str(path.relative_to(root_path)).replace("\\", "/"),
                )
        dates, source_by_date, relative_path = cached_dates[code]
        d0_date = str(event["d0_date"])
        prior = [date for date in dates if date < d0_date]
        prev_date = prior[-1] if prior else ""
        result[str(event["event_id"])] = {
            "prev_board_day": prev_date,
            "daily_cache_path": relative_path,
            "daily_cache_source_on_prev": source_by_date.get(prev_date, ""),
            "d0_present_in_daily_cache": "YES" if d0_date in dates else "NO",
        }
    return result


def _extract_pool_side(rows: list[dict[str, Any]] | None, prefix: str) -> dict[str, Any]:
    rows = rows or []
    if len(rows) != 1:
        return {
            f"{prefix}_row_found": int(bool(rows)),
            f"{prefix}_row_count": len(rows),
            f"{prefix}_source": "" if not rows else ";".join(sorted({str(r.get('source', '')) for r in rows})),
            f"{prefix}_cache_path": "" if not rows else ";".join(sorted({str(r.get('cache_path', '')) for r in rows})),
            f"{prefix}_source_semantics_valid": 0,
            f"{prefix}_limit_up_time": "",
            f"{prefix}_final_limit_up_time": "",
            f"{prefix}_open_board_count": math.nan,
            f"{prefix}_seal_amount": math.nan,
            f"{prefix}_amount": math.nan,
            f"{prefix}_consecutive_limit_up_count": math.nan,
            f"{prefix}_first_minute": math.nan,
            f"{prefix}_first_seconds": math.nan,
            f"{prefix}_first_time_parse_status": "ROW_MISSING" if not rows else "AMBIGUOUS_ROW",
            f"{prefix}_final_minute": math.nan,
            f"{prefix}_final_seconds": math.nan,
            f"{prefix}_final_time_parse_status": "ROW_MISSING" if not rows else "AMBIGUOUS_ROW",
        }
    row = rows[0]
    first_minute, first_seconds, first_status = parse_pool_time(row.get("limit_up_time"))
    final_minute, final_seconds, final_status = parse_pool_time(row.get("final_limit_up_time"))
    return {
        f"{prefix}_row_found": 1,
        f"{prefix}_row_count": 1,
        f"{prefix}_source": str(row.get("source", "")),
        f"{prefix}_cache_path": str(row.get("cache_path", "")),
        f"{prefix}_source_semantics_valid": int(str(row.get("source", "")) == VALID_POOL_SOURCE),
        f"{prefix}_limit_up_time": str(row.get("limit_up_time", "")),
        f"{prefix}_final_limit_up_time": str(row.get("final_limit_up_time", "")),
        f"{prefix}_open_board_count": _safe_numeric(row.get("open_board_count")),
        f"{prefix}_seal_amount": _safe_numeric(row.get("seal_amount")),
        f"{prefix}_amount": _safe_numeric(row.get("amount")),
        f"{prefix}_consecutive_limit_up_count": _safe_numeric(
            row.get("consecutive_limit_up_count")
        ),
        f"{prefix}_first_minute": first_minute,
        f"{prefix}_first_seconds": first_seconds,
        f"{prefix}_first_time_parse_status": first_status,
        f"{prefix}_final_minute": final_minute,
        f"{prefix}_final_seconds": final_seconds,
        f"{prefix}_final_time_parse_status": final_status,
    }


def build_row_eligibility(
    population: pd.DataFrame,
    pool: pd.DataFrame,
    trade_dates: Sequence[str],
    prev_board_days: Mapping[str, Mapping[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = _pool_lookup(pool)
    rows: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for event in population.to_dict("records"):
        prev_meta = prev_board_days[str(event["event_id"])]
        prev_date = str(prev_meta["prev_board_day"])
        global_prev_date = _previous_trade_date(trade_dates, str(event["d0_date"]))
        d0 = _extract_pool_side(lookup.get((str(event["d0_date"]), str(event["code"]))), "d0")
        prev = _extract_pool_side(lookup.get((prev_date, str(event["code"]))), "prev")
        expected_streak = 2 if str(event["board_group"]) == "BOARD2" else 3
        d0_before_d1 = int(str(event["d0_date"]) < str(event["d1_date"]))
        prev_before_d0 = int(bool(prev_date) and prev_date < str(event["d0_date"]))
        d0_streak_match = int(
            np.isfinite(d0["d0_consecutive_limit_up_count"])
            and int(d0["d0_consecutive_limit_up_count"]) == expected_streak
        )
        prev_streak_match = int(
            np.isfinite(prev["prev_consecutive_limit_up_count"])
            and int(prev["prev_consecutive_limit_up_count"]) == expected_streak - 1
        )
        d0_time_order_valid = int(
            d0["d0_first_time_parse_status"] == "VALID"
            and d0["d0_final_time_parse_status"] == "VALID"
            and float(d0["d0_final_seconds"]) >= float(d0["d0_first_seconds"])
        )
        prev_time_order_valid = int(
            prev["prev_first_time_parse_status"] == "VALID"
            and prev["prev_final_time_parse_status"] == "VALID"
            and float(prev["prev_final_seconds"]) >= float(prev["prev_first_seconds"])
        )
        d0_open_valid = int(
            np.isfinite(d0["d0_open_board_count"])
            and float(d0["d0_open_board_count"]) >= 0
        )
        prev_open_valid = int(
            np.isfinite(prev["prev_open_board_count"])
            and float(prev["prev_open_board_count"]) >= 0
        )
        # The authoritative candidate table fixes BOARD2/BOARD3.  Pool-native
        # consecutive_limit_up_count is a cross-check only: the repository has
        # already documented that its counting convention can differ from the
        # canonical daily-adjacency candidate definition.  Requiring exact
        # equality here would silently replace candidate lineage with pool
        # metadata semantics.  Actual same-code pool rows on D0 and the prior
        # market trading day are required for the last-two-board chain.
        chain_valid = int(
            d0["d0_row_count"] == 1
            and prev["prev_row_count"] == 1
            and d0_before_d1
            and prev_before_d0
        )
        f1_valid = int(
            chain_valid
            and d0["d0_source_semantics_valid"]
            and d0["d0_final_time_parse_status"] == "VALID"
        )
        f2_valid = int(
            chain_valid and d0["d0_source_semantics_valid"] and d0_open_valid
        )
        f3_valid = int(
            chain_valid and d0["d0_source_semantics_valid"] and d0_time_order_valid
        )
        f4_valid = int(
            chain_valid
            and d0["d0_source_semantics_valid"]
            and prev["prev_source_semantics_valid"]
            and d0["d0_final_time_parse_status"] == "VALID"
            and prev["prev_final_time_parse_status"] == "VALID"
        )
        f5_valid = int(
            chain_valid
            and d0["d0_source_semantics_valid"]
            and prev["prev_source_semantics_valid"]
            and d0_open_valid
            and prev_open_valid
        )
        violations: list[str] = []
        if not d0_before_d1:
            violations.append("D0_NOT_BEFORE_D1")
        if not prev_before_d0:
            violations.append("PREV_NOT_BEFORE_D0")
        if d0["d0_row_count"] != 1:
            violations.append("D0_ROW_MISSING_OR_AMBIGUOUS")
        if prev["prev_row_count"] != 1:
            violations.append("PREV_ROW_MISSING_OR_AMBIGUOUS")
        if d0["d0_row_count"] == 1 and not d0_streak_match:
            violations.append("D0_POOL_STREAK_CROSSCHECK_MISMATCH")
        if prev["prev_row_count"] == 1 and not prev_streak_match:
            violations.append("PREV_POOL_STREAK_CROSSCHECK_MISMATCH")
        if d0["d0_source_semantics_valid"] and not d0_time_order_valid:
            violations.append("D0_TIME_INVALID_OR_FINAL_BEFORE_FIRST")
        if prev["prev_source_semantics_valid"] and not prev_time_order_valid:
            violations.append("PREV_TIME_INVALID_OR_FINAL_BEFORE_FIRST")
        if d0["d0_source_semantics_valid"] and not d0_open_valid:
            violations.append("D0_OPEN_BOARD_COUNT_INVALID")
        if prev["prev_source_semantics_valid"] and not prev_open_valid:
            violations.append("PREV_OPEN_BOARD_COUNT_INVALID")
        lineage_violation = int(any(
            item not in {
                "D0_ROW_MISSING_OR_AMBIGUOUS",
                "PREV_ROW_MISSING_OR_AMBIGUOUS",
                "D0_POOL_STREAK_CROSSCHECK_MISMATCH",
                "PREV_POOL_STREAK_CROSSCHECK_MISMATCH",
            }
            for item in violations
        ))
        base = {
            "event_id": str(event["event_id"]),
            "code": str(event["code"]),
            "signal_date": str(event["signal_date"]),
            "month": str(event["month"]),
            "d0_date": str(event["d0_date"]),
            "prev_board_day": prev_date,
            "global_previous_pool_date": global_prev_date,
            "prev_matches_global_pool_calendar": int(bool(prev_date) and prev_date == global_prev_date),
            "daily_cache_path": str(prev_meta["daily_cache_path"]),
            "daily_cache_source_on_prev": str(prev_meta["daily_cache_source_on_prev"]),
            "d0_present_in_daily_cache": str(prev_meta["d0_present_in_daily_cache"]),
            "d1_date": str(event["d1_date"]),
            "board_group": str(event["board_group"]),
            "expected_streak": expected_streak,
            **d0,
            **prev,
            "d0_before_d1": d0_before_d1,
            "prev_before_d0": prev_before_d0,
            "d0_streak_match": d0_streak_match,
            "prev_streak_match": prev_streak_match,
            "chain_valid": chain_valid,
            "lineage_violation": lineage_violation,
            "lineage_notes": ";".join(violations),
            "F1_valid": f1_valid,
            "F2_valid": f2_valid,
            "F3_valid": f3_valid,
            "F4_valid": f4_valid,
            "F5_valid": f5_valid,
        }
        f1 = float(d0["d0_final_minute"]) if f1_valid else math.nan
        f2 = float(d0["d0_open_board_count"]) if f2_valid else math.nan
        f3 = (
            (float(d0["d0_final_seconds"]) - float(d0["d0_first_seconds"])) / 60.0
            if f3_valid
            else math.nan
        )
        f4 = (
            float(d0["d0_final_minute"]) - float(prev["prev_final_minute"])
            if f4_valid
            else math.nan
        )
        f5 = (
            float(d0["d0_open_board_count"]) - float(prev["prev_open_board_count"])
            if f5_valid
            else math.nan
        )
        rows.append(base)
        values.append(
            {
                "event_id": str(event["event_id"]),
                "code": str(event["code"]),
                "signal_date": str(event["signal_date"]),
                "month": str(event["month"]),
                "d0_date": str(event["d0_date"]),
                "prev_board_day": prev_date,
                "d1_date": str(event["d1_date"]),
                "board_group": str(event["board_group"]),
                "F1_D0_FINAL_SEAL_MINUTE": f1,
                "F2_D0_OPEN_BOARD_COUNT": f2,
                "F3_D0_RESEAL_DELAY_MINUTE": f3,
                "F4_FINAL_SEAL_DETERIORATION": f4,
                "F5_OPEN_BOARD_DETERIORATION": f5,
            }
        )
    eligibility = pd.DataFrame(rows).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    value_frame = pd.DataFrame(values).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if FORBIDDEN_PHASE_A_COLUMNS.intersection(eligibility.columns) or FORBIDDEN_PHASE_A_COLUMNS.intersection(value_frame.columns):
        raise RuntimeError("FATAL: Phase A output contains outcome/model columns")
    return eligibility, value_frame


def build_source_lineage(
    cache_lineage: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.DataFrame:
    source_rows: list[dict[str, Any]] = []
    for position, prefix in (("D0", "d0"), ("PREV_BOARD_DAY", "prev")):
        counts = eligibility[f"{prefix}_source"].replace("", "MISSING").value_counts()
        for source, count in counts.items():
            part = eligibility[eligibility[f"{prefix}_source"].replace("", "MISSING").eq(source)]
            source_rows.append(
                {
                    "record_type": "CANDIDATE_SOURCE_COMPOSITION",
                    "cache_date": "",
                    "cache_path": ";".join(sorted(part[f"{prefix}_cache_path"].dropna().astype(str).unique())),
                    "source": source,
                    "rows": int(count),
                    "limit_up_time_valid_rows": int(part[f"{prefix}_first_time_parse_status"].eq("VALID").sum()),
                    "final_limit_up_time_valid_rows": int(part[f"{prefix}_final_time_parse_status"].eq("VALID").sum()),
                    "open_board_count_valid_rows": int(pd.to_numeric(part[f"{prefix}_open_board_count"], errors="coerce").notna().sum()),
                    "seal_amount_valid_rows": int(pd.to_numeric(part[f"{prefix}_seal_amount"], errors="coerce").notna().sum()),
                    "amount_valid_rows": int(pd.to_numeric(part[f"{prefix}_amount"], errors="coerce").notna().sum()),
                    "field_semantics": (
                        "HISTORICAL_POOL_EXACT"
                        if source == VALID_POOL_SOURCE
                        else "DAILY_DERIVED_NO_SEAL_PATH"
                        if source == DERIVED_FALLBACK_SOURCE
                        else "MISSING_OR_UNKNOWN"
                    ),
                    "eligible_for_f1_f5": "YES" if source == VALID_POOL_SOURCE else "NO",
                    "d1_close_known": "YES",
                    "uses_d1_d2_d3_outcome": "NO",
                    "notes": f"position={position}",
                }
            )
    definitions = []
    for field in FIXED_FIELDS:
        meta = FIELD_DEFINITIONS[field]
        definitions.append(
            {
                "record_type": "FIELD_LINEAGE",
                "cache_date": "",
                "cache_path": POOL_CACHE_RELATIVE_DIR,
                "source": "HISTORICAL_LIMIT_UP_POOL_ONLY",
                "rows": 0,
                "limit_up_time_valid_rows": 0,
                "final_limit_up_time_valid_rows": 0,
                "open_board_count_valid_rows": 0,
                "seal_amount_valid_rows": 0,
                "amount_valid_rows": 0,
                "field_semantics": f"{field}: {meta['formula']}",
                "eligible_for_f1_f5": "PREDECLARED",
                "d1_close_known": "YES; source date <= D0 < D1",
                "uses_d1_d2_d3_outcome": "NO",
                "notes": f"parents={meta['parents']};quality_direction={meta['quality_direction']}",
            }
        )
    return pd.concat(
        [cache_lineage, pd.DataFrame(source_rows), pd.DataFrame(definitions)],
        ignore_index=True,
    )


def build_coverage(eligibility: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    major_lineage = eligibility["lineage_violation"].astype(bool)
    for index, field in enumerate(FIXED_FIELDS, start=1):
        valid_col = f"F{index}_valid"
        valid = eligibility[valid_col].astype(bool)
        def rate(mask: pd.Series) -> float:
            return float(valid[mask].mean()) if int(mask.sum()) else math.nan
        month_rates = {
            month: rate(eligibility["month"].eq(month)) for month in ("MAY", "JUNE", "JULY")
        }
        board_rates = {
            board: rate(eligibility["board_group"].eq(board)) for board in ("BOARD2", "BOARD3")
        }
        relevant_lineage = int((major_lineage & valid).sum())
        ready = bool(
            valid.mean() >= OVERALL_COVERAGE_GATE
            and all(value >= MONTH_COVERAGE_GATE for value in month_rates.values())
            and all(value >= BOARD_COVERAGE_GATE for value in board_rates.values())
            and relevant_lineage == 0
        )
        rows.append(
            {
                "field": field,
                "formula": FIELD_DEFINITIONS[field]["formula"],
                "quality_direction": "LOWER_IS_STRONGER",
                "total_rows": len(eligibility),
                "valid_rows": int(valid.sum()),
                "coverage_rate": float(valid.mean()),
                "valid_dates": int(eligibility.loc[valid, "signal_date"].nunique()),
                "total_dates": int(eligibility["signal_date"].nunique()),
                "may_valid_rows": int((valid & eligibility["month"].eq("MAY")).sum()),
                "may_total_rows": int(eligibility["month"].eq("MAY").sum()),
                "may_coverage": month_rates["MAY"],
                "june_valid_rows": int((valid & eligibility["month"].eq("JUNE")).sum()),
                "june_total_rows": int(eligibility["month"].eq("JUNE").sum()),
                "june_coverage": month_rates["JUNE"],
                "july_valid_rows": int((valid & eligibility["month"].eq("JULY")).sum()),
                "july_total_rows": int(eligibility["month"].eq("JULY").sum()),
                "july_coverage": month_rates["JULY"],
                "board2_valid_rows": int((valid & eligibility["board_group"].eq("BOARD2")).sum()),
                "board2_total_rows": int(eligibility["board_group"].eq("BOARD2").sum()),
                "board2_coverage": board_rates["BOARD2"],
                "board3_valid_rows": int((valid & eligibility["board_group"].eq("BOARD3")).sum()),
                "board3_total_rows": int(eligibility["board_group"].eq("BOARD3").sum()),
                "board3_coverage": board_rates["BOARD3"],
                "major_lineage_violations_on_valid_rows": relevant_lineage,
                "overall_gate": f">={OVERALL_COVERAGE_GATE:.0%}",
                "monthly_gate": f">={MONTH_COVERAGE_GATE:.0%} each",
                "board_gate": f">={BOARD_COVERAGE_GATE:.0%} each",
                "field_data_ready": "YES" if ready else "NO",
                "failure_reasons": ";".join(
                    reason
                    for condition, reason in (
                        (valid.mean() < OVERALL_COVERAGE_GATE, "OVERALL_COVERAGE_BELOW_80PCT"),
                        (month_rates["MAY"] < MONTH_COVERAGE_GATE, "MAY_COVERAGE_BELOW_70PCT"),
                        (month_rates["JUNE"] < MONTH_COVERAGE_GATE, "JUNE_COVERAGE_BELOW_70PCT"),
                        (month_rates["JULY"] < MONTH_COVERAGE_GATE, "JULY_COVERAGE_BELOW_70PCT"),
                        (board_rates["BOARD2"] < BOARD_COVERAGE_GATE, "BOARD2_COVERAGE_BELOW_65PCT"),
                        (board_rates["BOARD3"] < BOARD_COVERAGE_GATE, "BOARD3_COVERAGE_BELOW_65PCT"),
                        (relevant_lineage > 0, "MAJOR_LINEAGE_VIOLATION"),
                    )
                    if condition
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_data_state(coverage: pd.DataFrame) -> tuple[str, tuple[str, ...]]:
    eligible = tuple(
        coverage.loc[coverage["field_data_ready"].eq("YES"), "field"].astype(str)
    )
    if len(eligible) == len(FIXED_FIELDS):
        return "READY", eligible
    if len(eligible) >= FAMILY_READY_FIELD_MINIMUM:
        return "PARTIAL", eligible
    return "NOT_READY", eligible


def build_phase_a_lock(
    population: pd.DataFrame,
    coverage: pd.DataFrame,
    source_lineage: pd.DataFrame,
    data_state: str,
    eligible_fields: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    payload = {
        "task": TASK_NAME,
        "phase": "PHASE_A_BLIND_COVERAGE_LINEAGE",
        "fixed_fields": list(FIXED_FIELDS),
        "candidate_rows": population[list(PHASE_A_USECOLS)].to_dict("records"),
        "coverage": coverage.to_dict("records"),
        "source_composition": source_lineage[
            source_lineage["record_type"].eq("CANDIDATE_SOURCE_COMPOSITION")
        ].to_dict("records"),
        "data_state": data_state,
        "eligible_fields": list(eligible_fields),
        "dropped_fields": [field for field in FIXED_FIELDS if field not in eligible_fields],
        "outcome_accessed": "NO",
        "model_score_or_rank_accessed": "NO",
    }
    lock_hash = _json_hash(payload)
    rows: list[dict[str, Any]] = [
        {
            "record_type": "LOCK_METADATA",
            "key": "PHASE_A_LOCK_SHA256",
            "value": lock_hash,
            "event_id": "",
            "signal_date": "",
            "details": f"data_state={data_state};eligible_fields={len(eligible_fields)}/5",
            "phase_a_lock_sha256": lock_hash,
        },
        {
            "record_type": "LOCK_METADATA",
            "key": "OUTCOME_ACCESSED_BEFORE_LOCK",
            "value": "NO",
            "event_id": "",
            "signal_date": "",
            "details": "Phase A read_csv(usecols=identity/date only)",
            "phase_a_lock_sha256": lock_hash,
        },
    ]
    for field in FIXED_FIELDS:
        rows.append(
            {
                "record_type": "FIELD_DECISION",
                "key": field,
                "value": "ELIGIBLE" if field in eligible_fields else "DROPPED_BY_PREDECLARED_GATE",
                "event_id": "",
                "signal_date": "",
                "details": str(
                    coverage.loc[coverage["field"].eq(field), "failure_reasons"].iloc[0]
                ),
                "phase_a_lock_sha256": lock_hash,
            }
        )
    for row in population[["event_id", "signal_date"]].to_dict("records"):
        rows.append(
            {
                "record_type": "CANDIDATE_ID",
                "key": "",
                "value": "",
                "event_id": row["event_id"],
                "signal_date": row["signal_date"],
                "details": "FROZEN_PHASE_A_POPULATION",
                "phase_a_lock_sha256": lock_hash,
            }
        )
    return pd.DataFrame(rows), lock_hash


def run_phase_a(root: str | Path) -> dict[str, Any]:
    population = load_phase_a_population(root)
    pool, cache_lineage, trade_dates = load_pool_cache(root)
    prev_board_days = build_prev_board_day_map(root, population)
    eligibility, values = build_row_eligibility(
        population, pool, trade_dates, prev_board_days
    )
    source_lineage = build_source_lineage(cache_lineage, eligibility)
    coverage = build_coverage(eligibility)
    data_state, eligible_fields = classify_data_state(coverage)
    phase_a_lock, lock_hash = build_phase_a_lock(
        population, coverage, source_lineage, data_state, eligible_fields
    )
    return {
        "population": population,
        "source_lineage": source_lineage,
        "coverage": coverage,
        "eligibility": eligibility,
        "values": values,
        "phase_a_lock": phase_a_lock,
        "phase_a_lock_sha256": lock_hash,
        "data_state": data_state,
        "eligible_fields": eligible_fields,
        "outcome_accessed": "NO",
    }


def _pair_table(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in frame.groupby("signal_date", sort=True):
        winners = day[day["target7"].astype(bool)].sort_values("event_id", kind="mergesort")
        losses = day[day["loss"].astype(bool)].sort_values("event_id", kind="mergesort")
        for winner in winners.itertuples(index=False):
            for loss in losses.itertuples(index=False):
                winner_value = float(getattr(winner, field))
                loss_value = float(getattr(loss, field))
                raw_diff = winner_value - loss_value
                quality_diff = loss_value - winner_value  # lower raw value is preregistered stronger
                rows.append(
                    {
                        "field": field,
                        "signal_date": str(signal_date),
                        "month": _month_label(str(signal_date)),
                        "target7_event_id": str(winner.event_id),
                        "loss_event_id": str(loss.event_id),
                        "target7_raw_value": winner_value,
                        "loss_raw_value": loss_value,
                        "raw_difference_t7_minus_loss": raw_diff,
                        "quality_difference_t7_minus_loss": quality_diff,
                        "raw_concordance": 1.0 if raw_diff > 0 else 0.0 if raw_diff < 0 else 0.5,
                        "quality_oriented_concordance": 1.0 if quality_diff > 0 else 0.0 if quality_diff < 0 else 0.5,
                        "target7_s2_score": float(winner.s2_score),
                        "loss_s2_score": float(loss.s2_score),
                        "s2_wrong_pair": int(float(winner.s2_score) <= float(loss.s2_score)),
                    }
                )
    return pd.DataFrame(rows)


def _summarize_pairs(pairs: pd.DataFrame, field: str, period: str, scope: str) -> dict[str, Any]:
    if period == "POOLED":
        part = pairs
    else:
        part = pairs[pairs["month"].eq(period)]
    return {
        "field": field,
        "period": period,
        "scope": scope,
        "pair_count": len(part),
        "date_count": int(part["signal_date"].nunique()) if len(part) else 0,
        "raw_concordance": float(part["raw_concordance"].mean()) if len(part) else math.nan,
        "quality_oriented_concordance": float(part["quality_oriented_concordance"].mean()) if len(part) else math.nan,
        "quality_direction": "LOWER_IS_STRONGER",
        "audit_status": "RUN",
    }


def load_phase_b_population(root: str | Path, phase_a: Mapping[str, Any]) -> pd.DataFrame:
    """Open frozen outcomes/S2 only after a persisted Phase-A lock authorizes it."""
    from .v004c_s2_winner_vs_loss_failure_attribution import load_frozen_s2

    if len(phase_a["eligible_fields"]) < FAMILY_READY_FIELD_MINIMUM:
        raise RuntimeError("FATAL: Phase B attempted despite failed family coverage gate")
    if not phase_a.get("phase_a_lock_persisted", False):
        raise RuntimeError("FATAL: Phase B attempted before Phase-A lock persistence")
    scored, _, audit = load_frozen_s2(Path(root).resolve())
    scored = scored[
        scored["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & scored["label_available_date"].lt(TRAINING_ASOF)
    ].copy()
    if (len(scored), scored["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: Phase B frozen S2 population parity failed")
    if int(audit.get("model_refits", 0)) != 0:
        raise RuntimeError("FATAL: Phase B source unexpectedly refit a model")
    values = phase_a["values"]
    merged = scored.merge(values, on=["event_id", "code", "signal_date", "d0_date", "d1_date", "board_group"], how="inner", validate="one_to_one")
    if len(merged) != 485:
        raise RuntimeError("FATAL: Phase A/B event identity mismatch")
    if merged["signal_date"].ge(TRAINING_ASOF).any():
        raise RuntimeError("FATAL: August signal row reached Phase B")
    return merged.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def _direction_state(month_values: Sequence[float]) -> str:
    finite = [value for value in month_values if np.isfinite(value)]
    if len(finite) != 3:
        return "INSUFFICIENT_MONTHS"
    if max(finite) > 0.55 and min(finite) < 0.45:
        return "TEMPORAL_DIRECTION_FLIP"
    if all(value > 0.5 for value in finite):
        return "STABLE_PREREGISTERED_QUALITY"
    if all(value < 0.5 for value in finite):
        return "STABLE_INVERSE_INFORMATION"
    return "WEAK_OR_MIXED"


def _bootstrap_from_pairs(
    pairs: pd.DataFrame, metric_col: str, seed: int
) -> tuple[float, float, float, float, float]:
    if pairs.empty:
        return (math.nan,) * 5
    daily = pairs.groupby("signal_date")[metric_col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(daily), size=(BOOTSTRAP_REPETITIONS, len(daily)))
    sums = daily["sum"].to_numpy(float)[draws].sum(axis=1)
    counts = daily["count"].to_numpy(float)[draws].sum(axis=1)
    estimates = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return (
        float(pairs[metric_col].mean()),
        float(np.nanmedian(estimates)),
        float(np.nanpercentile(estimates, 2.5)),
        float(np.nanpercentile(estimates, 97.5)),
        float(np.nanmean(estimates > 0.5)),
    )


def _selection_permutation(
    frame: pd.DataFrame, eligible_fields: Sequence[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    binary = frame[frame["target7"].astype(bool) | frame["loss"].astype(bool)].copy()
    observed: dict[str, float] = {}
    for field in eligible_fields:
        pairs = _pair_table(binary.dropna(subset=[field]), field)
        observed[field] = float(pairs["quality_oriented_concordance"].mean()) if len(pairs) else math.nan
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, day in binary.groupby("signal_date", sort=True):
        labels = day["target7"].to_numpy(int)
        values = day[list(eligible_fields)].to_numpy(float)
        groups.append((labels, values))
    rng = np.random.default_rng(PERMUTATION_SEED)
    max_abs = np.zeros(PERMUTATION_REPETITIONS, dtype=float)
    for repetition in range(PERMUTATION_REPETITIONS):
        sums = np.zeros(len(eligible_fields), dtype=float)
        counts = np.zeros(len(eligible_fields), dtype=float)
        for labels, values in groups:
            permuted = rng.permutation(labels)
            winner_idx = np.flatnonzero(permuted == 1)
            loss_idx = np.flatnonzero(permuted == 0)
            if not len(winner_idx) or not len(loss_idx):
                continue
            for field_index in range(len(eligible_fields)):
                winner_values = values[winner_idx, field_index]
                loss_values = values[loss_idx, field_index]
                valid_w = winner_values[np.isfinite(winner_values)]
                valid_l = loss_values[np.isfinite(loss_values)]
                if not len(valid_w) or not len(valid_l):
                    continue
                differences = valid_l[:, None] - valid_w[None, :]
                sums[field_index] += float((differences > 0).sum() + 0.5 * (differences == 0).sum())
                counts[field_index] += differences.size
        concordance = np.divide(sums, counts, out=np.full_like(sums, 0.5), where=counts > 0)
        max_abs[repetition] = float(np.max(np.abs(concordance - 0.5)))
    rows = []
    for field in eligible_fields:
        deviation = abs(observed[field] - 0.5) if np.isfinite(observed[field]) else math.nan
        rows.append(
            {
                "field": field,
                "observed_quality_concordance": observed[field],
                "observed_abs_deviation_from_0_5": deviation,
                "eligible_field_count": len(eligible_fields),
                "permutations": PERMUTATION_REPETITIONS,
                "seed": PERMUTATION_SEED,
                "sampling_rule": "WITHIN_SIGNAL_DATE_LABEL_PERMUTATION_PRESERVE_CLASS_COUNTS",
                "family_selection_adjusted_p": float(np.mean(max_abs >= deviation)) if np.isfinite(deviation) else math.nan,
                "null_max_abs_p50": float(np.percentile(max_abs, 50)),
                "null_max_abs_p95": float(np.percentile(max_abs, 95)),
                "audit_status": "RUN",
            }
        )
    return pd.DataFrame(rows), max_abs


def build_phase_b(root: str | Path, phase_a: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    frame = load_phase_b_population(root, phase_a)
    eligible_fields = tuple(phase_a["eligible_fields"])
    full_pairs: dict[str, pd.DataFrame] = {}
    pair_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for field_index, field in enumerate(eligible_fields):
        field_frame = frame.dropna(subset=[field]).copy()
        pairs = _pair_table(field_frame, field)
        full_pairs[field] = pairs
        for period in ("POOLED", "MAY", "JUNE", "JULY"):
            pair_rows.append(_summarize_pairs(pairs, field, period, "ALL_CANDIDATES"))
        month_values = [
            next(
                row["quality_oriented_concordance"]
                for row in pair_rows
                if row["field"] == field and row["period"] == month and row["scope"] == "ALL_CANDIDATES"
            )
            for month in ("MAY", "JUNE", "JULY")
        ]
        monthly_rows.append(
            {
                "field": field,
                "may_quality_concordance": month_values[0],
                "june_quality_concordance": month_values[1],
                "july_quality_concordance": month_values[2],
                "minimum": float(np.nanmin(month_values)),
                "maximum": float(np.nanmax(month_values)),
                "range": float(np.nanmax(month_values) - np.nanmin(month_values)),
                "direction_state": _direction_state(month_values),
                "quality_direction": "LOWER_IS_STRONGER",
                "audit_status": "RUN",
            }
        )
        head = field_frame[field_frame["s2_rank"].between(2, 6)].copy()
        head_pairs = _pair_table(head, field)
        for period in ("POOLED", "MAY", "JUNE", "JULY"):
            rank_rows.append(_summarize_pairs(head_pairs, field, period, "S2_RANK2_6"))
        wrong = pairs[pairs["s2_wrong_pair"].eq(1)].copy()
        for period in ("POOLED", "MAY", "JUNE", "JULY"):
            part = wrong if period == "POOLED" else wrong[wrong["month"].eq(period)]
            rescue_rows.append(
                {
                    "field": field,
                    "period": period,
                    "wrong_pair_count": len(part),
                    "date_count": int(part["signal_date"].nunique()) if len(part) else 0,
                    "error_rescue_rate": float(part["quality_oriented_concordance"].mean()) if len(part) else math.nan,
                    "quality_direction": "LOWER_IS_STRONGER",
                    "audit_status": "RUN",
                }
            )
        for endpoint, source_pairs in (("PAIR_CONCORDANCE", pairs), ("S2_WRONG_PAIR_RESCUE", wrong)):
            metric = "quality_oriented_concordance"
            observed, median, p025, p975, probability = _bootstrap_from_pairs(
                source_pairs, metric, BOOTSTRAP_SEED + field_index * 10 + (1 if endpoint.endswith("RESCUE") else 0)
            )
            bootstrap_rows.append(
                {
                    "field": field,
                    "endpoint": endpoint,
                    "observed": observed,
                    "bootstrap_median": median,
                    "p2_5": p025,
                    "p97_5": p975,
                    "p_gt_0_5": probability,
                    "sampling_unit": "SIGNAL_DATE",
                    "repetitions": BOOTSTRAP_REPETITIONS,
                    "seed": BOOTSTRAP_SEED + field_index * 10 + (1 if endpoint.endswith("RESCUE") else 0),
                    "audit_status": "RUN",
                }
            )
    permutation, _ = _selection_permutation(frame, eligible_fields)
    missing_rows: list[dict[str, Any]] = []
    for field in FIXED_FIELDS:
        for completeness, mask in (
            ("COMPLETE", frame[field].notna()), ("INCOMPLETE", frame[field].isna())
        ):
            part = frame[mask]
            missing_rows.append(
                {
                    "scope": field,
                    "completeness": completeness,
                    "rows": len(part),
                    "dates": int(part["signal_date"].nunique()),
                    "target7_rate": float(part["target7"].mean()) if len(part) else math.nan,
                    "loss_rate": float(part["loss"].mean()) if len(part) else math.nan,
                    "may_share": float(part["month"].eq("MAY").mean()) if len(part) else math.nan,
                    "june_share": float(part["month"].eq("JUNE").mean()) if len(part) else math.nan,
                    "july_share": float(part["month"].eq("JULY").mean()) if len(part) else math.nan,
                    "board2_share": float(part["board_group"].eq("BOARD2").mean()) if len(part) else math.nan,
                    "board3_share": float(part["board_group"].eq("BOARD3").mean()) if len(part) else math.nan,
                    "audit_status": "RUN_POST_PHASE_A_LOCK",
                }
            )
    complete_all = frame[list(eligible_fields)].notna().all(axis=1)
    for completeness, mask in (("COMPLETE", complete_all), ("INCOMPLETE", ~complete_all)):
        part = frame[mask]
        missing_rows.append(
            {
                "scope": "ALL_ELIGIBLE_FIELDS",
                "completeness": completeness,
                "rows": len(part),
                "dates": int(part["signal_date"].nunique()),
                "target7_rate": float(part["target7"].mean()) if len(part) else math.nan,
                "loss_rate": float(part["loss"].mean()) if len(part) else math.nan,
                "may_share": float(part["month"].eq("MAY").mean()) if len(part) else math.nan,
                "june_share": float(part["month"].eq("JUNE").mean()) if len(part) else math.nan,
                "july_share": float(part["month"].eq("JULY").mean()) if len(part) else math.nan,
                "board2_share": float(part["board_group"].eq("BOARD2").mean()) if len(part) else math.nan,
                "board3_share": float(part["board_group"].eq("BOARD3").mean()) if len(part) else math.nan,
                "audit_status": "RUN_POST_PHASE_A_LOCK",
            }
        )
    return {
        "pair_concordance": pd.DataFrame(pair_rows),
        "monthly_stability": pd.DataFrame(monthly_rows),
        "rank2_6": pd.DataFrame(rank_rows),
        "error_rescue": pd.DataFrame(rescue_rows),
        "selection_permutation": permutation,
        "bootstrap": pd.DataFrame(bootstrap_rows),
        "missingness": pd.DataFrame(missing_rows),
    }


def _placeholder_phase_b_tables() -> dict[str, pd.DataFrame]:
    reason = "NOT_RUN_PHASE_A_FAMILY_GATE_FAILED"
    return {
        "pair_concordance": pd.DataFrame([{
            "field": "", "period": "", "scope": "", "pair_count": 0, "date_count": 0,
            "raw_concordance": math.nan, "quality_oriented_concordance": math.nan,
            "quality_direction": "LOWER_IS_STRONGER", "audit_status": reason,
        }]),
        "monthly_stability": pd.DataFrame([{
            "field": "", "may_quality_concordance": math.nan, "june_quality_concordance": math.nan,
            "july_quality_concordance": math.nan, "minimum": math.nan, "maximum": math.nan,
            "range": math.nan, "direction_state": "NOT_TESTED", "quality_direction": "LOWER_IS_STRONGER",
            "audit_status": reason,
        }]),
        "rank2_6": pd.DataFrame([{
            "field": "", "period": "", "scope": "S2_RANK2_6", "pair_count": 0, "date_count": 0,
            "raw_concordance": math.nan, "quality_oriented_concordance": math.nan,
            "quality_direction": "LOWER_IS_STRONGER", "audit_status": reason,
        }]),
        "error_rescue": pd.DataFrame([{
            "field": "", "period": "", "wrong_pair_count": 0, "date_count": 0,
            "error_rescue_rate": math.nan, "quality_direction": "LOWER_IS_STRONGER", "audit_status": reason,
        }]),
        "selection_permutation": pd.DataFrame([{
            "field": "", "observed_quality_concordance": math.nan,
            "observed_abs_deviation_from_0_5": math.nan, "eligible_field_count": 0,
            "permutations": 0, "seed": PERMUTATION_SEED,
            "sampling_rule": "NOT_RUN", "family_selection_adjusted_p": math.nan,
            "null_max_abs_p50": math.nan, "null_max_abs_p95": math.nan, "audit_status": reason,
        }]),
        "bootstrap": pd.DataFrame([{
            "field": "", "endpoint": "", "observed": math.nan, "bootstrap_median": math.nan,
            "p2_5": math.nan, "p97_5": math.nan, "p_gt_0_5": math.nan,
            "sampling_unit": "SIGNAL_DATE", "repetitions": 0, "seed": BOOTSTRAP_SEED,
            "audit_status": reason,
        }]),
        "missingness": pd.DataFrame([{
            "scope": "", "completeness": "", "rows": 0, "dates": 0,
            "target7_rate": math.nan, "loss_rate": math.nan, "may_share": math.nan,
            "june_share": math.nan, "july_share": math.nan, "board2_share": math.nan,
            "board3_share": math.nan, "audit_status": reason,
        }]),
    }


def determine_information_state(
    data_state: str, eligible_fields: Sequence[str], phase_b: Mapping[str, pd.DataFrame]
) -> AuditState:
    if len(eligible_fields) < FAMILY_READY_FIELD_MINIMUM:
        return AuditState(
            data_state="NOT_READY",
            information_state="NOT_TESTED_DUE_TO_DATA",
            broader_information_limitation="[待核验]",
            stage1_research_state="STOP_AND_REVIEW",
            next_action="STOP_AND_REVIEW",
            phase_b_run=False,
        )
    pairs = phase_b["pair_concordance"]
    monthly = phase_b["monthly_stability"]
    head = phase_b["rank2_6"]
    rescue = phase_b["error_rescue"]
    permutation = phase_b["selection_permutation"]
    bootstrap = phase_b["bootstrap"]
    qualifying: list[str] = []
    partial: list[str] = []
    for field in eligible_fields:
        pooled = pairs[(pairs["field"].eq(field)) & pairs["period"].eq("POOLED")].iloc[0]
        stability = monthly[monthly["field"].eq(field)].iloc[0]
        head_row = head[(head["field"].eq(field)) & head["period"].eq("POOLED")].iloc[0]
        rescue_row = rescue[(rescue["field"].eq(field)) & rescue["period"].eq("POOLED")].iloc[0]
        perm_row = permutation[permutation["field"].eq(field)].iloc[0]
        boot_row = bootstrap[(bootstrap["field"].eq(field)) & bootstrap["endpoint"].eq("PAIR_CONCORDANCE")].iloc[0]
        direction = 1 if pooled["quality_oriented_concordance"] > 0.5 else -1
        head_same = (head_row["quality_oriented_concordance"] - 0.5) * direction >= HEAD_MATERIAL_DEVIATION
        rescue_same = (rescue_row["error_rescue_rate"] - 0.5) * direction >= RESCUE_MATERIAL_DEVIATION
        material = abs(pooled["quality_oriented_concordance"] - 0.5) >= MATERIAL_DEVIATION
        stable = stability["direction_state"] in {
            "STABLE_PREREGISTERED_QUALITY", "STABLE_INVERSE_INFORMATION"
        }
        bootstrap_support = not (boot_row["p2_5"] <= 0.5 <= boot_row["p97_5"])
        selection_support = perm_row["family_selection_adjusted_p"] <= SELECTION_ADJUSTED_P_GATE
        date_support = int(pooled["date_count"]) >= MIN_PAIR_DATES
        if material and stable and head_same and rescue_same and bootstrap_support and selection_support and date_support:
            qualifying.append(field)
        elif material or head_same or rescue_same:
            partial.append(field)
    if qualifying:
        return AuditState(
            data_state=data_state,
            information_state="PREBREAK_SEAL_PATH_INFORMATION_SUPPORTED",
            broader_information_limitation="[待核验]",
            stage1_research_state="ONE_CONFIRMATION_EXPERIMENT_JUSTIFIED",
            next_action="DESIGN_SINGLE_SEAL_PATH_CONFIRMATION",
            phase_b_run=True,
        )
    if partial:
        return AuditState(
            data_state=data_state,
            information_state="PREBREAK_SEAL_PATH_INFORMATION_PARTIAL",
            broader_information_limitation="[待核验]",
            stage1_research_state="STOP_AND_REVIEW",
            next_action="STOP_AND_REVIEW",
            phase_b_run=True,
        )
    return AuditState(
        data_state=data_state,
        information_state="PREBREAK_SEAL_PATH_INFORMATION_NOT_SUPPORTED",
        broader_information_limitation="SUPPORTED",
        stage1_research_state="CURRENT_D1_INFORMATION_RESEARCH_EXHAUSTED",
        next_action="STOP_STAGE1_FACTOR_MINING_AND_REVIEW_PIPELINE",
        phase_b_run=True,
    )


def render_review(phase_a: Mapping[str, Any], state: AuditState) -> str:
    coverage = phase_a["coverage"]
    ready_count = int(coverage["field_data_ready"].eq("YES").sum())
    coverage_lines = "\n".join(
        f"- {row.field}: overall {row.coverage_rate:.2%}; May {row.may_coverage:.2%}; "
        f"June {row.june_coverage:.2%}; July {row.july_coverage:.2%}; ready={row.field_data_ready}."
        for row in coverage.itertuples(index=False)
    )
    if not state.phase_b_run:
        answers = (
            ("Q1. 现有历史数据能否可靠看到以前两板怎么封？",
             "不能覆盖完整开发期。真实 historical pool 的封板时间/开板次数主要出现在部分 June 和 July；May 使用 daily-derived fallback，这些字段按规则全部缺失。"),
            ("Q2. Target7 与 LOSS 在封板路径上是否有区别？",
             f"未测试。Phase A 只有 {ready_count}/5 字段通过门槛，family gate 失败，因此没有读取 outcome。"),
            ("Q3. May / June / July 是否稳定？", "未测试；不能用缺失结构替代时间稳定性结论。"),
            ("Q4. S2 排错时 seal path 能否救回？", "未测试；S2 score/rank 在 Phase A 未读取。"),
            ("Q5. 这是新信息还是另一个不稳定关系？", "目前只能确认它是语义上新的 family，不能确认其信息效果。"),
            ("Q6. 这个 family 值不值得继续？", "现有数据不足以直接继续信息判断；先停止并评审数据可用性，不能补零或临时用5分钟重建。"),
        )
    else:
        answers = (
            ("Q1. 现有历史数据能否可靠看到以前两板怎么封？", f"可以进入固定信息审计；{ready_count}/5 字段通过预注册coverage/lineage门。"),
            ("Q2. Target7 与 LOSS 在封板路径上是否有区别？", state.information_state),
            ("Q3. May / June / July 是否稳定？", "见 monthly stability 表；正式状态已同时要求跨月不翻转。"),
            ("Q4. S2 排错时 seal path 能否救回？", "见 S2 error rescue 表；未修改S2。"),
            ("Q5. 这是新信息还是另一个不稳定关系？", state.information_state),
            ("Q6. 这个 family 值不值得继续？", state.next_action),
        )
    lines = [
        "# v004c Pre-break Board-day Seal-path Information Audit v001",
        "",
        "## 简单版",
        "",
    ]
    for question, answer in answers:
        lines.extend([f"### {question}", "", answer, ""])
    lines.extend(
        [
            "## Phase A — Blind Coverage / Lineage",
            "",
            f"Frozen population: 485 rows / 60 dates, {DEVELOPMENT_START} through {DEVELOPMENT_END}.",
            "",
            "Phase A used only event identity/date/board columns and historical limit-up pool fields. "
            "It did not load target, return, S2 score, or S2 rank.",
            "",
            coverage_lines,
            "",
            f"Fields ready: {ready_count}/5; eligible={';'.join(phase_a['eligible_fields']) or 'NONE'}.",
            "",
            f"PHASE_A_LOCK_SHA256 = {phase_a['phase_a_lock_sha256']}",
            "",
            "`daily_limitup_derived` rows retain MISSING time/open-board semantics; no zero-fill, "
            "15:00 substitution, OHLC inference, 5-minute repair, or cross-date fill was used.",
            "",
            "`seal_amount` and `amount` are coverage notes only and never enter the five-field audit.",
            "",
            "## Phase B — Fixed Information Audit",
            "",
            "RUN" if state.phase_b_run else "NOT RUN: Phase A family coverage gate failed before outcome access.",
            "",
            "## Final State",
            "",
            f"PREBREAK_SEAL_PATH_DATA_STATE = {state.data_state}",
            "",
            f"PREBREAK_SEAL_PATH_INFORMATION_STATE = {state.information_state}",
            "",
            f"BROADER_D1_INFORMATION_LIMITATION = {state.broader_information_limitation}",
            "",
            f"STAGE1_D1_INFORMATION_RESEARCH_STATE = {state.stage1_research_state}",
            "",
            "MODEL_TRAINED = NO",
            "",
            "FEATURE_SEARCH = NO",
            "",
            "WINDOW_SEARCH = NO",
            "",
            "AUGUST_HOLDOUT_STATUS = CONSUMED",
            "",
            "AUGUST_USED_AS_FRESH_OOT = NO",
            "",
            f"NEXT_ACTION = {state.next_action}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_phase_a_outputs(output_dir: Path, phase_a: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = (
        (OUTPUT_FILENAMES[0], phase_a["source_lineage"]),
        (OUTPUT_FILENAMES[1], phase_a["coverage"]),
        (OUTPUT_FILENAMES[2], phase_a["eligibility"]),
        (OUTPUT_FILENAMES[3], phase_a["phase_a_lock"]),
        (OUTPUT_FILENAMES[4], phase_a["values"]),
    )
    for name, frame in tables:
        (output_dir / name).write_bytes(_csv_bytes(frame))


def write_outputs(root: str | Path, output_dir: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    target = Path(output_dir).resolve() if output_dir is not None else (
        root_path / "reports" / "research" / OUTPUT_DIRNAME
    )
    phase_a = run_phase_a(root_path)
    _write_phase_a_outputs(target, phase_a)
    # The persisted lock is the only authorization to open outcomes/S2.
    lock_path = target / OUTPUT_FILENAMES[3]
    if not lock_path.is_file() or phase_a["phase_a_lock_sha256"] not in lock_path.read_text(encoding="utf-8-sig"):
        raise RuntimeError("FATAL: Phase-A lock persistence check failed")
    phase_a = dict(phase_a)
    phase_a["phase_a_lock_persisted"] = True
    if len(phase_a["eligible_fields"]) >= FAMILY_READY_FIELD_MINIMUM:
        phase_b = build_phase_b(root_path, phase_a)
        provisional = determine_information_state(phase_a["data_state"], phase_a["eligible_fields"], phase_b)
        state = AuditState(**{**provisional.__dict__, "phase_b_run": True})
        outcome_accessed = "YES_AFTER_PHASE_A_LOCK"
    else:
        phase_b = _placeholder_phase_b_tables()
        state = determine_information_state(phase_a["data_state"], phase_a["eligible_fields"], phase_b)
        outcome_accessed = "NO"
    phase_b_tables = (
        (OUTPUT_FILENAMES[5], phase_b["pair_concordance"]),
        (OUTPUT_FILENAMES[6], phase_b["monthly_stability"]),
        (OUTPUT_FILENAMES[7], phase_b["rank2_6"]),
        (OUTPUT_FILENAMES[8], phase_b["error_rescue"]),
        (OUTPUT_FILENAMES[9], phase_b["selection_permutation"]),
        (OUTPUT_FILENAMES[10], phase_b["bootstrap"]),
        (OUTPUT_FILENAMES[11], phase_b["missingness"]),
    )
    for name, frame in phase_b_tables:
        (target / name).write_bytes(_csv_bytes(frame))
    (target / OUTPUT_FILENAMES[12]).write_text(render_review(phase_a, state), encoding="utf-8", newline="\n")
    output_hashes = {
        name: hashlib.sha256((target / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }
    audit = {
        "rows": len(phase_a["population"]),
        "dates": int(phase_a["population"]["signal_date"].nunique()),
        "fixed_field_count": len(FIXED_FIELDS),
        "field_data_ready_count": len(phase_a["eligible_fields"]),
        "eligible_fields": list(phase_a["eligible_fields"]),
        "phase_a_lock_sha256": phase_a["phase_a_lock_sha256"],
        "phase_a_outcome_accessed": "NO",
        "phase_a_model_score_or_rank_accessed": "NO",
        "phase_b_run": state.phase_b_run,
        "outcome_accessed": outcome_accessed,
        "model_trained": "NO",
        "feature_search": "NO",
        "window_search": "NO",
        "august_used_as_fresh_oot": "NO",
        "data_state": state.data_state,
        "information_state": state.information_state,
        "broader_d1_information_limitation": state.broader_information_limitation,
        "stage1_research_state": state.stage1_research_state,
        "next_action": state.next_action,
        "output_sha256": output_hashes,
    }
    return target, audit


__all__ = [
    "AUGUST_HOLDOUT_STATUS",
    "BOOTSTRAP_REPETITIONS",
    "FAMILY_READY_FIELD_MINIMUM",
    "FIXED_FIELDS",
    "OUTPUT_FILENAMES",
    "PERMUTATION_REPETITIONS",
    "build_coverage",
    "build_row_eligibility",
    "classify_data_state",
    "load_phase_a_population",
    "parse_pool_time",
    "run_phase_a",
    "write_outputs",
]

"""Blind data/engineering feasibility audit for the v004d auction pipeline.

The audit is deliberately blind to forward outcomes.  Candidate identity is
loaded through an explicit column projection, historical cache inspection is
limited to D2 official-open and intraday timestamp availability, and no model
or BUY/PASS rule is created.

The repository does not contain a historical 09:25 auction feed.  Daily open
is retained as a price-equivalent lineage reference only; it is never used to
invent historical auction volume or amount.  The live collector is provided
for a future real trading-day 09:25--09:30 validation, but the offline audit
cannot mark that test as passed.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TASK_NAME = "v004d_auction_data_feasibility_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
AUDIT_ASOF = "2026-09-06"
TIMEZONE = "Asia/Shanghai"
PRIMARY_HISTORY_START = "2026-05-01"
PRIMARY_HISTORY_END = "2026-07-31"
AUGUST_AVAILABILITY_END = "2026-08-31"

BRIDGE_POPULATION_ARTIFACT = (
    "reports/research/v004c_stage1_18f_bridge_analysis_v001_20260506_20260731/"
    "v004c_stage1_18f_bridge_candidates.csv"
)
INPUT_COLUMNS = (
    "event_id", "code", "signal_date", "d1_date", "d2_date",
    "board_group", "candidate_count",
)
FORBIDDEN_INPUT_COLUMNS = (
    "target7", "loss", "severe_loss", "capped_return_7", "raw_repair_return",
    "d3_high", "stage1_score", "stage1_rank",
)

HISTORICAL_CORE_COVERAGE_GATE = 0.90
HISTORICAL_MONTH_COVERAGE_GATE = 0.85
LIVE_COMPLETENESS_GATE = 0.95
LIVE_P95_SECONDS_GATE = 120.0
SCALE_SIZES = (5, 10, 15, 20, 30)

OUTPUT_FILENAMES = (
    "v004d_auction_source_inventory_v001.csv",
    "v004d_auction_field_semantics_v001.csv",
    "v004d_auction_historical_coverage_v001.csv",
    "v004d_auction_price_lineage_v001.csv",
    "v004d_auction_volume_amount_semantics_v001.csv",
    "v004d_auction_5m_semantics_v001.csv",
    "v004d_auction_historical_values_sample_v001.csv",
    "v004d_auction_live_latency_v001.csv",
    "v004d_auction_scale_stress_v001.csv",
    "v004d_auction_pipeline_timing_v001.csv",
    "v004d_auction_source_shift_v001.csv",
    "v004d_auction_data_feasibility_review_v001.md",
)

LIVE_COLUMNS = (
    "code", "auction_price", "auction_volume", "auction_amount",
    "bid1_price", "bid1_volume", "ask1_price", "ask1_volume",
    "source_timestamp",
)


class LiveWindowError(RuntimeError):
    """Raised when a live audit is attempted outside the fixed auction window."""


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator or not np.isfinite(denominator):
        return math.nan
    return float(numerator / denominator)


def _month(value: str) -> str:
    return str(value)[:7]


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_blind_candidate_universe(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load candidate identity only; outcome/model columns are never materialized."""

    if set(INPUT_COLUMNS) & set(FORBIDDEN_INPUT_COLUMNS):
        raise RuntimeError("FATAL: blind candidate projection contains forbidden columns")
    root_path = Path(root).resolve()
    path = root_path / BRIDGE_POPULATION_ARTIFACT
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        usecols=list(INPUT_COLUMNS),
        dtype={"event_id": str, "code": str},
        encoding="utf-8-sig",
    )
    if set(frame.columns) != set(INPUT_COLUMNS):
        raise RuntimeError("FATAL: projected candidate schema mismatch")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in ("signal_date", "d1_date", "d2_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.strftime("%Y-%m-%d")
    frame["board_group"] = frame["board_group"].astype(str).str.upper()
    if not frame["board_group"].isin(("BOARD2", "BOARD3")).all():
        raise RuntimeError("FATAL: candidate universe contains non-Board2/Board3 rows")
    frame["candidate_count"] = pd.to_numeric(frame["candidate_count"], errors="raise").astype(int)
    frame["month"] = frame["signal_date"].map(_month)
    frame["history_scope"] = np.where(
        frame["d2_date"].between(PRIMARY_HISTORY_START, PRIMARY_HISTORY_END),
        "PRIMARY_MAY_JULY_D2",
        np.where(
            frame["d2_date"].between("2026-08-01", AUGUST_AVAILABILITY_END),
            "AUGUST_AVAILABILITY_ONLY",
            "OUTSIDE_AUDIT_WINDOW",
        ),
    )
    frame = frame.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate candidate event_id")
    if (len(frame), frame["signal_date"].nunique()) != (497, 62):
        raise RuntimeError("FATAL: authoritative May-July universe is not 497 rows / 62 dates")
    by_date = frame.groupby("signal_date")["event_id"].transform("size").astype(int)
    if not by_date.eq(frame["candidate_count"]).all():
        raise RuntimeError("FATAL: candidate_count parity failed")
    if not frame["d1_date"].eq(frame["signal_date"]).all():
        raise RuntimeError("FATAL: D1/signal_date identity failed")
    audit = {
        "input_artifact": BRIDGE_POPULATION_ARTIFACT,
        "input_column_projection": "|".join(INPUT_COLUMNS),
        "all_candidate_rows": len(frame),
        "all_candidate_dates": int(frame["signal_date"].nunique()),
        "primary_candidate_d2_rows": int(frame["history_scope"].eq("PRIMARY_MAY_JULY_D2").sum()),
        "august_availability_rows": int(frame["history_scope"].eq("AUGUST_AVAILABILITY_ONLY").sum()),
        "duplicate_event_id": int(frame["event_id"].duplicated().sum()),
        "outcome_accessed": "NO",
        "model_trained": "NO",
    }
    return frame, audit


def _read_daily_row(root: Path, code: str, trade_date: str) -> dict[str, Any]:
    candidates = (
        (root / "data/cache/daily_unadjusted" / f"{code}_daily.pkl", "daily_unadjusted"),
        (root / "data/cache/daily" / f"{code}_daily.pkl", "daily"),
    )
    errors: list[str] = []
    for path, cache_kind in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_pickle(path)
            if not isinstance(frame, pd.DataFrame) or "date" not in frame.columns:
                errors.append(f"{cache_kind}:INVALID_FRAME")
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            rows = frame.loc[dates.eq(trade_date)].copy()
            if rows.empty:
                errors.append(f"{cache_kind}:DATE_MISSING")
                continue
            row = rows.iloc[-1]
            return {
                "daily_row_available": 1,
                "official_d2_open": pd.to_numeric(pd.Series([row.get("open")]), errors="coerce").iloc[0],
                "official_d2_volume": pd.to_numeric(pd.Series([row.get("volume")]), errors="coerce").iloc[0],
                "official_d2_amount": pd.to_numeric(pd.Series([row.get("amount")]), errors="coerce").iloc[0],
                "daily_source": str(row.get("source", "")),
                "daily_cache_kind": cache_kind,
                "daily_cache_path": _rel(root, path),
                "daily_error": "",
            }
        except Exception as exc:  # pragma: no cover - emitted into audit rows
            errors.append(f"{cache_kind}:{type(exc).__name__}")
    return {
        "daily_row_available": 0,
        "official_d2_open": math.nan,
        "official_d2_volume": math.nan,
        "official_d2_amount": math.nan,
        "daily_source": "",
        "daily_cache_kind": "",
        "daily_cache_path": "",
        "daily_error": "|".join(errors) or "NO_DAILY_CACHE",
    }


def _minute_cache_audit(root: Path, code: str, trade_date: str) -> dict[str, Any]:
    candidates = (
        (root / "data/cache/minute_5m" / f"{code}_5min.pkl", "minute_5m"),
        (root / "data/cache/baostock_5m" / f"{code}_5min.pkl", "baostock_5m"),
    )
    errors: list[str] = []
    for path, cache_kind in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_pickle(path)
            if not isinstance(frame, pd.DataFrame) or "trade_date" not in frame.columns:
                errors.append(f"{cache_kind}:INVALID_FRAME")
                continue
            dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            day = frame.loc[dates.eq(trade_date)].copy()
            if day.empty:
                errors.append(f"{cache_kind}:DATE_MISSING")
                continue
            timestamps = pd.to_datetime(day.get("datetime"), errors="coerce")
            times = timestamps.dt.strftime("%H:%M:%S")
            preopen = times.lt("09:30:00")
            at_0925 = times.eq("09:25:00")
            return {
                "minute_row_available": 1,
                "minute_cache_kind": cache_kind,
                "minute_cache_path": _rel(root, path),
                "minute_source": "|".join(sorted(set(day.get("source", pd.Series(dtype=str)).astype(str)))) if "source" in day else "",
                "minute_bar_count": int(len(day)),
                "minute_first_time": str(times.min()),
                "minute_last_time": str(times.max()),
                "preopen_bar_count": int(preopen.sum()),
                "bar_0925_count": int(at_0925.sum()),
                "minute_error": "",
            }
        except Exception as exc:  # pragma: no cover
            errors.append(f"{cache_kind}:{type(exc).__name__}")
    return {
        "minute_row_available": 0,
        "minute_cache_kind": "",
        "minute_cache_path": "",
        "minute_source": "",
        "minute_bar_count": 0,
        "minute_first_time": "",
        "minute_last_time": "",
        "preopen_bar_count": 0,
        "bar_0925_count": 0,
        "minute_error": "|".join(errors) or "NO_5M_CACHE",
    }


def build_historical_audit(
    root: str | Path, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit historical core auction availability without inventing volume/amount."""

    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for item in candidates.itertuples(index=False):
        daily = _read_daily_row(root_path, item.code, item.d2_date)
        minute = _minute_cache_audit(root_path, item.code, item.d2_date)
        official_open_ok = bool(
            daily["daily_row_available"] and pd.notna(daily["official_d2_open"])
            and float(daily["official_d2_open"]) > 0
        )
        # Daily open is an official price-equivalent reference, not an
        # independently archived 09:25 snapshot.  No historical auction
        # volume/amount exists in the repository.
        primary = item.history_scope == "PRIMARY_MAY_JULY_D2"
        row = {
            "record_type": "CANDIDATE",
            "event_id": item.event_id,
            "code": item.code,
            "signal_date": item.signal_date,
            "d1_date": item.d1_date,
            "d2_date": item.d2_date,
            "month": item.month,
            "board_group": item.board_group,
            "history_scope": item.history_scope,
            "in_primary_history_gate": int(primary),
            **daily,
            **minute,
            "auction_price_equivalent_available": int(official_open_ok),
            "independent_auction_price_snapshot_available": 0,
            "auction_volume_available": 0,
            "auction_amount_available": 0,
            "historical_core_fields_complete": 0,
            "historical_auction_source": "NONE",
            "missing_reason": "NO_HISTORICAL_0925_AUCTION_VOLUME_AMOUNT",
        }
        rows.append(row)
    detail = pd.DataFrame(rows).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    primary = detail[detail["in_primary_history_gate"].eq(1)].copy()
    summary_rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str, pd.DataFrame]] = [("OVERALL", "ALL", primary)]
    group_specs += [
        ("MONTH", month, primary[primary["month"].eq(month)])
        for month in ("2026-05", "2026-06", "2026-07")
    ]
    group_specs += [
        ("BOARD", board, primary[primary["board_group"].eq(board)])
        for board in ("BOARD2", "BOARD3")
    ]
    for dimension, value, part in group_specs:
        summary_rows.append({
            "record_type": "SUMMARY",
            "summary_dimension": dimension,
            "summary_value": value,
            "rows": len(part),
            "signal_dates": int(part["signal_date"].nunique()),
            "d2_dates": int(part["d2_date"].nunique()),
            "auction_price_equivalent_coverage": float(part["auction_price_equivalent_available"].mean()),
            "independent_auction_price_coverage": float(part["independent_auction_price_snapshot_available"].mean()),
            "auction_volume_coverage": float(part["auction_volume_available"].mean()),
            "auction_amount_coverage": float(part["auction_amount_available"].mean()),
            "historical_core_three_coverage": float(part["historical_core_fields_complete"].mean()),
            "coverage_gate": HISTORICAL_CORE_COVERAGE_GATE if dimension != "MONTH" else HISTORICAL_MONTH_COVERAGE_GATE,
            "gate_pass": int(
                part["historical_core_fields_complete"].mean()
                >= (HISTORICAL_CORE_COVERAGE_GATE if dimension != "MONTH" else HISTORICAL_MONTH_COVERAGE_GATE)
            ),
        })
    summary = pd.DataFrame(summary_rows)

    lineage = detail[[
        "event_id", "code", "signal_date", "d2_date", "month", "board_group",
        "history_scope", "official_d2_open", "daily_source", "daily_cache_kind",
        "daily_cache_path", "independent_auction_price_snapshot_available",
    ]].copy()
    lineage["historical_auction_price"] = math.nan
    lineage["price_difference"] = math.nan
    lineage["exact_match"] = math.nan
    lineage["within_0_01"] = math.nan
    lineage["comparison_status"] = "NOT_TESTABLE_NO_INDEPENDENT_HISTORICAL_AUCTION_PRICE"
    lineage["aggregate_exact_match_rate"] = math.nan
    lineage["aggregate_within_0_01_rate"] = math.nan
    lineage["aggregate_max_difference"] = math.nan
    lineage["lineage_state"] = "AUCTION_PRICE_LINEAGE_RISK"

    audit = {
        "primary_rows": len(primary),
        "primary_signal_dates": int(primary["signal_date"].nunique()),
        "primary_d2_dates": int(primary["d2_date"].nunique()),
        "price_equivalent_coverage": float(primary["auction_price_equivalent_available"].mean()),
        "independent_auction_price_coverage": 0.0,
        "auction_volume_coverage": 0.0,
        "auction_amount_coverage": 0.0,
        "historical_core_coverage": 0.0,
        "historical_state": "NOT_READY",
        "price_lineage_state": "AUCTION_PRICE_LINEAGE_RISK",
        "earliest_5m_time": str(primary.loc[primary["minute_row_available"].eq(1), "minute_first_time"].min()),
        "historical_0925_bar_rows": int(primary["bar_0925_count"].gt(0).sum()),
        "historical_preopen_bar_rows": int(primary["preopen_bar_count"].gt(0).sum()),
    }
    return detail, summary, lineage, audit


def build_source_inventory() -> pd.DataFrame:
    """Static inventory plus the explicit 2026-09-06 non-trading-day probe."""

    rows = [
        {
            "source_name": "EASTMONEY_SPOT_BATCH_REPO",
            "interface": "MarketDataProvider._fetch_spot_eastmoney / push2 clist",
            "historical_support": "NO",
            "realtime_support": "PROSPECTIVE_YES",
            "earliest_historical_date": "NOT_APPLICABLE",
            "intraday_update_time": "CURRENT_SNAPSHOT; 09:25 BEHAVIOR NOT VALIDATED",
            "candidate_batch_support": "YES_FULL_MARKET_THEN_FILTER",
            "credentials_required": "NO",
            "rate_limit": "UNDOCUMENTED; repository client throttles requests",
            "available_fields": "open|latest_price|volume|amount|high|low|prev_close; no L1-L5 in repo wrapper",
            "data_semantics": "Current quote cumulative fields; 09:25 final-auction meaning not yet proven",
            "observed_latency_ms": math.nan,
            "probe_context": "NOT_PROBED_IN_REAL_AUCTION_WINDOW",
            "status": "NEEDS_REAL_TRADING_DAY_VALIDATION",
        },
        {
            "source_name": "AKSHARE_EASTMONEY_PRE_MIN",
            "interface": "akshare.stock_zh_a_hist_pre_min_em / push2 trends2 iscr=1 ndays=1",
            "historical_support": "NO_MAY_JULY_REPLAY; CURRENT/RECENT_SESSION_ONLY",
            "realtime_support": "PROSPECTIVE_YES_SINGLE_CODE",
            "earliest_historical_date": "NOT_PARAMETERIZED",
            "intraday_update_time": "Rows labelled 09:15 through 09:30; live stabilization unverified",
            "candidate_batch_support": "NO_SINGLE_CODE",
            "credentials_required": "NO",
            "rate_limit": "UNDOCUMENTED",
            "available_fields": "time|open|close|high|low|volume|amount|latest/reference-like price",
            "data_semantics": "Supplier pre-market minute state; not proven ordinary trades or final matched-only flow",
            "observed_latency_ms": 13803.626,
            "probe_context": "2026-09-06 Sunday query returned 16 rows for 2026-09-04; repeat was intermittent",
            "status": "LIVE_SEMANTICS_AND_RELIABILITY_UNRESOLVED",
        },
        {
            "source_name": "AKSHARE_EASTMONEY_BID_ASK",
            "interface": "akshare.stock_bid_ask_em / push2 stock/get",
            "historical_support": "NO",
            "realtime_support": "PROSPECTIVE_YES_SINGLE_CODE",
            "earliest_historical_date": "NOT_APPLICABLE",
            "intraday_update_time": "CURRENT_SNAPSHOT; source timestamp not exposed by wrapper",
            "candidate_batch_support": "NO_SINGLE_CODE",
            "credentials_required": "NO",
            "rate_limit": "UNDOCUMENTED",
            "available_fields": "bid1-5 price/volume|ask1-5 price/volume|current quote|volume|amount|open|prev_close",
            "data_semantics": "Current order-book/quote snapshot; auction matched/unmatched semantics not exposed",
            "observed_latency_ms": 2367.233,
            "probe_context": "2026-09-06 Sunday probe returned JSON decode error",
            "status": "UNRELIABLE_OFF_HOURS; NEEDS_REAL_WINDOW_VALIDATION",
        },
        {
            "source_name": "SINA_5M_REPOSITORY",
            "interface": "MarketDataProvider.fetch_5min_history(source=sina_5m)",
            "historical_support": "YES_CONTINUOUS_SESSION_ONLY",
            "realtime_support": "NOT_DESIGNED_FOR_0925",
            "earliest_historical_date": "CACHE_DEPENDENT",
            "intraday_update_time": "First normalized bar 09:35",
            "candidate_batch_support": "NO_SINGLE_CODE",
            "credentials_required": "NO",
            "rate_limit": "Repository RequestClient throttle",
            "available_fields": "09:35-15:00 OHLCV/amount",
            "data_semantics": "Ordinary 5m continuous-auction bars; first bar mixes 09:30-09:35 and is not auction-only",
            "observed_latency_ms": math.nan,
            "probe_context": "Local cache inspection",
            "status": "NOT_AN_OPENING_AUCTION_SOURCE",
        },
        {
            "source_name": "BAOSTOCK_5M_REPOSITORY",
            "interface": "MarketDataProvider.fetch_5min_history(source=baostock_5m)",
            "historical_support": "YES_CONTINUOUS_SESSION_ONLY",
            "realtime_support": "NO",
            "earliest_historical_date": "CACHE/SERVICE_DEPENDENT",
            "intraday_update_time": "First normalized bar 09:35",
            "candidate_batch_support": "NO_SINGLE_CODE",
            "credentials_required": "NO",
            "rate_limit": "Service dependent",
            "available_fields": "09:35-15:00 OHLCV/amount",
            "data_semantics": "Ordinary 5m continuous-auction bars; no 09:25 auction-only record",
            "observed_latency_ms": math.nan,
            "probe_context": "Local cache inspection",
            "status": "NOT_AN_OPENING_AUCTION_SOURCE",
        },
        {
            "source_name": "DAILY_UNADJUSTED_CACHE",
            "interface": "data/cache/daily_unadjusted/<code>_daily.pkl",
            "historical_support": "YES_DAILY_OPEN_ONLY_FOR_AUCTION_PURPOSE",
            "realtime_support": "NO_0925_DELIVERY_CONTRACT",
            "earliest_historical_date": "CACHE_DEPENDENT",
            "intraday_update_time": "END-OF-DAY/HISTORICAL",
            "candidate_batch_support": "LOCAL_BATCH",
            "credentials_required": "NO",
            "rate_limit": "NOT_APPLICABLE",
            "available_fields": "official daily open plus full-day OHLCV; amount often missing",
            "data_semantics": "D2 official open is price-equivalent reference only; full-day volume/amount are not auction volume/amount",
            "observed_latency_ms": 0.0,
            "probe_context": "Local cache inspection",
            "status": "PRICE_REFERENCE_ONLY",
        },
    ]
    return pd.DataFrame(rows)


def build_field_semantics() -> pd.DataFrame:
    definitions = [
        ("auction_price", "DAILY_OPEN_EQUIVALENT_ONLY", "PROSPECTIVE_SPOT_OPEN", "Final 09:25 clearing price should equal official open, but no independent historical auction snapshot exists", "PARTIAL"),
        ("auction_pct_change", "DERIVABLE_FROM_PRICE_AND_D1_CLOSE_FOR_QA", "DERIVABLE", "QA-only arithmetic; no threshold or predictive use in this audit", "PARTIAL"),
        ("auction_volume", "NOT_AVAILABLE", "PROSPECTIVE_CUMULATIVE_QUOTE_FIELD", "Not archived; live field must be proven as 09:25 matched volume rather than virtual/cumulative supplier value", "UNRESOLVED"),
        ("auction_amount", "NOT_AVAILABLE", "PROSPECTIVE_CUMULATIVE_QUOTE_FIELD", "Not archived; live field must be proven as 09:25 matched amount", "UNRESOLVED"),
        ("bid1_price", "NOT_AVAILABLE", "AKSHARE_BID_ASK_PROSPECTIVE", "Current level-1 price; wrapper has no source timestamp", "UNRESOLVED"),
        ("bid1_volume", "NOT_AVAILABLE", "AKSHARE_BID_ASK_PROSPECTIVE", "Current level-1 volume; unmatched-buy interpretation not established", "UNRESOLVED"),
        ("ask1_price", "NOT_AVAILABLE", "AKSHARE_BID_ASK_PROSPECTIVE", "Current level-1 price; wrapper has no source timestamp", "UNRESOLVED"),
        ("ask1_volume", "NOT_AVAILABLE", "AKSHARE_BID_ASK_PROSPECTIVE", "Current level-1 volume; unmatched-sell interpretation not established", "UNRESOLVED"),
        ("bid2_to_bid5", "NOT_AVAILABLE", "AKSHARE_BID_ASK_PROSPECTIVE", "Level 2-5 quote ladder; current-only and no historical replay", "UNRESOLVED"),
        ("ask2_to_ask5", "NOT_AVAILABLE", "AKSHARE_BID_ASK_PROSPECTIVE", "Level 2-5 quote ladder; current-only and no historical replay", "UNRESOLVED"),
        ("matched_volume", "NOT_AVAILABLE", "NOT_EXPLICITLY_EXPOSED", "Must not be inferred from general quote volume", "NOT_AVAILABLE"),
        ("unmatched_buy_volume", "NOT_AVAILABLE", "NOT_EXPLICITLY_EXPOSED", "Must not be inferred from bid1 volume", "NOT_AVAILABLE"),
        ("unmatched_sell_volume", "NOT_AVAILABLE", "NOT_EXPLICITLY_EXPOSED", "Must not be inferred from ask1 volume", "NOT_AVAILABLE"),
        ("reference_or_virtual_match_price", "NOT_AVAILABLE", "PRE_MIN_LAST_COLUMN_SEMANTICS_UNCONFIRMED", "AKShare labels the eighth pre-min field as latest price; virtual-match semantics are not documented in repository code", "UNRESOLVED"),
        ("state_0915_0920", "NOT_HISTORICALLY_AVAILABLE", "PRE_MIN_1M_ROWS_PROSPECTIVE", "Order cancellation is permitted; rows cannot be treated as ordinary trades", "UNRESOLVED"),
        ("state_0920_0925", "NOT_HISTORICALLY_AVAILABLE", "PRE_MIN_1M_ROWS_PROSPECTIVE", "Order cancellation prohibited by market phase; supplier row aggregation semantics still need validation", "UNRESOLVED"),
    ]
    return pd.DataFrame(definitions, columns=[
        "field", "historical_availability", "live_availability", "semantics", "semantic_state"
    ])


def build_volume_amount_semantics() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source": "DAILY_UNADJUSTED_CACHE", "field": "volume|amount",
            "historical_or_live": "HISTORICAL", "available": "FULL_DAY_ONLY",
            "is_0925_final_match": "NO", "is_virtual_or_cumulative": "FULL_DAY_CUMULATIVE",
            "semantic_state": "CLEAR_NOT_USABLE", "evidence": "Daily row represents the full trading day",
        },
        {
            "source": "SINA_OR_BAOSTOCK_5M", "field": "09:35 volume|amount",
            "historical_or_live": "HISTORICAL", "available": "YES_WHERE_CACHED",
            "is_0925_final_match": "NO", "is_virtual_or_cumulative": "09:30-09:35 BAR_INCLUDING_CONTINUOUS_TRADING",
            "semantic_state": "CLEAR_NOT_USABLE", "evidence": "Earliest normalized bar timestamp is 09:35",
        },
        {
            "source": "AKSHARE_EASTMONEY_PRE_MIN", "field": "volume|amount",
            "historical_or_live": "LIVE_PROSPECTIVE", "available": "CURRENT_SESSION_ONLY",
            "is_0925_final_match": "UNPROVEN", "is_virtual_or_cumulative": "UNRESOLVED",
            "semantic_state": "SEMANTICS_UNRESOLVED", "evidence": "Wrapper exposes fields but no explicit matched-only contract",
        },
        {
            "source": "EASTMONEY_SPOT_OR_BID_ASK", "field": "volume|amount",
            "historical_or_live": "LIVE_PROSPECTIVE", "available": "CURRENT_SNAPSHOT",
            "is_0925_final_match": "UNPROVEN", "is_virtual_or_cumulative": "GENERAL_QUOTE_CUMULATIVE",
            "semantic_state": "SEMANTICS_UNRESOLVED", "evidence": "Must be sampled after 09:25 and reconciled; source timestamp absent",
        },
    ])


def build_5m_semantics(historical_audit: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source": "SINA_5M_REPOSITORY", "native_interval": "5m",
            "timestamp_range": "09:35-15:00", "has_0915_0920_record": 0,
            "has_0920_0925_record": 0, "ohlc_semantics": "CONTINUOUS_TRADING_BAR",
            "volume_semantics": "BAR_VOLUME_INCLUDING_09:30-09:35",
            "real_trades": "YES_AFTER_0930", "snapshot_aggregation": "NO_AUCTION_STATE_RECORD",
            "historical_coverage_rows": historical_audit["primary_rows"],
            "live_availability": "NOT_FOR_0925", "state": "NOT_USABLE",
        },
        {
            "source": "BAOSTOCK_5M_REPOSITORY", "native_interval": "5m",
            "timestamp_range": "09:35-15:00", "has_0915_0920_record": 0,
            "has_0920_0925_record": 0, "ohlc_semantics": "CONTINUOUS_TRADING_BAR",
            "volume_semantics": "BAR_VOLUME_INCLUDING_09:30-09:35",
            "real_trades": "YES_AFTER_0930", "snapshot_aggregation": "NO_AUCTION_STATE_RECORD",
            "historical_coverage_rows": historical_audit["primary_rows"],
            "live_availability": "NO_0925", "state": "NOT_USABLE",
        },
        {
            "source": "AKSHARE_EASTMONEY_PRE_MIN", "native_interval": "1m supplier state",
            "timestamp_range": "09:15-09:30 current/recent session", "has_0915_0920_record": 1,
            "has_0920_0925_record": 1, "ohlc_semantics": "NOT_PROVEN_AS_ORDINARY_TRADE_OHLC",
            "volume_semantics": "VIRTUAL_OR_CUMULATIVE_SEMANTICS_UNRESOLVED",
            "real_trades": "NO_DURING_PREOPEN_EXCEPT_FINAL_MATCH", "snapshot_aggregation": "LIKELY_SUPPLIER_STATE_SERIES; UNPROVEN",
            "historical_coverage_rows": 0, "live_availability": "PROSPECTIVE_SINGLE_CODE",
            "state": "SEMANTICALLY_LIMITED",
        },
    ])


def normalize_live_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically normalize a live source response for timing/QA only."""

    result = frame.copy()
    for column in LIVE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan if column != "code" else ""
    result = result[list(LIVE_COLUMNS)]
    result["code"] = result["code"].astype(str).str.zfill(6)
    for column in LIVE_COLUMNS[1:-1]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source_timestamp"] = result["source_timestamp"].astype(str)
    return result.sort_values("code", kind="mergesort").drop_duplicates(
        "code", keep="last"
    ).reset_index(drop=True)


def calculate_basic_live_fields(
    candidates: pd.DataFrame, snapshot: pd.DataFrame
) -> pd.DataFrame:
    """Perform the fixed Stage-B feasibility calculations, without prediction."""

    base = candidates[["code", "d1_close", "d1_volume"]].copy()
    base["code"] = base["code"].astype(str).str.zfill(6)
    joined = base.merge(normalize_live_snapshot(snapshot), on="code", how="left", validate="one_to_one")
    joined["auction_pct_from_d1_close"] = joined["auction_price"] / joined["d1_close"] - 1.0
    joined["auction_volume_ratio"] = joined["auction_volume"] / joined["d1_volume"]
    denominator = joined["bid1_volume"] + joined["ask1_volume"]
    joined["basic_imbalance"] = np.where(
        denominator.gt(0),
        (joined["bid1_volume"] - joined["ask1_volume"]) / denominator,
        np.nan,
    )
    joined["core_fields_complete"] = joined[[
        "auction_price", "auction_volume", "auction_amount"
    ]].notna().all(axis=1).astype(int)
    joined["missing_core_fields"] = joined.apply(
        lambda row: "|".join(
            field for field in ("auction_price", "auction_volume", "auction_amount")
            if pd.isna(row[field])
        ), axis=1,
    )
    return joined.sort_values("code", kind="mergesort").reset_index(drop=True)


def _fixture_candidates(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "code": [f"{i + 1:06d}" for i in range(n)],
        "d1_close": np.linspace(8.0, 18.0, n),
        "d1_volume": np.linspace(1_000_000, 3_000_000, n),
    })


def _fixture_snapshot(n: int) -> pd.DataFrame:
    codes = [f"{i + 1:06d}" for i in range(n)]
    return pd.DataFrame({
        "code": codes,
        "auction_price": np.linspace(8.08, 18.18, n),
        "auction_volume": np.linspace(50_000, 250_000, n),
        "auction_amount": np.linspace(500_000, 4_000_000, n),
        "bid1_price": np.linspace(8.07, 18.17, n),
        "bid1_volume": np.linspace(20_000, 100_000, n),
        "ask1_price": np.linspace(8.09, 18.19, n),
        "ask1_volume": np.linspace(15_000, 120_000, n),
        "source_timestamp": ["DRY_RUN_FIXTURE"] * n,
    })


def build_dry_run_timing(repetitions: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Offline scale test.  Fetch latency is intentionally not simulated."""

    detail_rows: list[dict[str, Any]] = []
    for n in SCALE_SIZES:
        candidates = _fixture_candidates(n)
        snapshot = _fixture_snapshot(n)
        for repetition in range(1, repetitions + 1):
            fetch_ms = 0.0
            start = time.perf_counter_ns()
            normalized = normalize_live_snapshot(snapshot)
            parse_end = time.perf_counter_ns()
            calculated = calculate_basic_live_fields(candidates, normalized)
            calc_end = time.perf_counter_ns()
            detail_rows.append({
                "test_mode": "OFFLINE_SYNTHETIC_DRY_RUN_NOT_LIVE",
                "candidate_count": n,
                "repetition": repetition,
                "request_mode": "BATCH_FIXTURE",
                "fetch_time_ms": fetch_ms,
                "parse_time_ms": (parse_end - start) / 1_000_000,
                "feature_calc_time_ms": (calc_end - parse_end) / 1_000_000,
                "total_pipeline_time_ms": (calc_end - start) / 1_000_000,
                "success_count": int(calculated["core_fields_complete"].sum()),
                "missing_count": int(calculated["core_fields_complete"].eq(0).sum()),
                "stale_count": math.nan,
                "error_count": 0,
                "retry_count": 0,
                "live_gate_eligible": 0,
            })
    detail = pd.DataFrame(detail_rows)
    summaries: list[dict[str, Any]] = []
    for n, part in detail.groupby("candidate_count", sort=True):
        summaries.append({
            "test_mode": "OFFLINE_SYNTHETIC_DRY_RUN_NOT_LIVE",
            "candidate_count": int(n),
            "repetitions": len(part),
            "fetch_time_ms_median": float(part["fetch_time_ms"].median()),
            "parse_time_ms_median": float(part["parse_time_ms"].median()),
            "feature_calc_time_ms_median": float(part["feature_calc_time_ms"].median()),
            "total_pipeline_time_ms_median": float(part["total_pipeline_time_ms"].median()),
            "total_pipeline_time_ms_p95": float(part["total_pipeline_time_ms"].quantile(.95)),
            "core_field_completeness": float(part["success_count"].sum() / (len(part) * int(n))),
            "interpretation": "Parser/join/calculation capacity only; excludes network fetch and cannot satisfy live gate",
        })
    return pd.DataFrame(summaries), detail


def _validate_live_window(now: datetime) -> None:
    local = now.astimezone(ZoneInfo(TIMEZONE))
    if local.weekday() >= 5 or not (clock_time(9, 25) <= local.time() <= clock_time(9, 30)):
        raise LiveWindowError(
            "real live latency audit is only valid on a weekday between 09:25:00 and 09:30:00 Asia/Shanghai"
        )


@dataclass(frozen=True)
class LiveCollectionResult:
    normalized: pd.DataFrame
    metrics: Mapping[str, Any]


def collect_live_snapshot(
    candidates: pd.DataFrame,
    fetcher: Callable[[Sequence[str]], pd.DataFrame],
    *,
    now: datetime,
) -> LiveCollectionResult:
    """Real-window collector entry point with timestamp/latency logging.

    ``fetcher`` must return the ``LIVE_COLUMNS`` contract.  This function is
    intentionally source-agnostic so the primary and at most one fallback can
    be validated without changing downstream semantics.
    """

    _validate_live_window(now)
    codes = sorted(candidates["code"].astype(str).str.zfill(6).unique())
    requested = datetime.now(ZoneInfo(TIMEZONE))
    start = time.perf_counter_ns()
    raw = fetcher(codes)
    responded = datetime.now(ZoneInfo(TIMEZONE))
    fetch_ms = (time.perf_counter_ns() - start) / 1_000_000
    parse_start = time.perf_counter_ns()
    normalized = normalize_live_snapshot(raw)
    parse_ms = (time.perf_counter_ns() - parse_start) / 1_000_000
    found = set(normalized["code"])
    missing = len(set(codes) - found)
    complete = int(normalized[["auction_price", "auction_volume", "auction_amount"]].notna().all(axis=1).sum())
    metrics = {
        "request_timestamp": requested.isoformat(),
        "response_timestamp": responded.isoformat(),
        "latency_ms": fetch_ms,
        "parse_time_ms": parse_ms,
        "candidate_count": len(codes),
        "success_count": complete,
        "missing_count": missing + (len(normalized) - complete),
        "stale_count": math.nan,
        "error_count": 0,
        "retry_count": 0,
        "stale_check_status": "UNRESOLVED_UNLESS_SOURCE_TIMESTAMP_IS_PROVEN",
    }
    return LiveCollectionResult(normalized=normalized, metrics=metrics)


def build_live_pending() -> pd.DataFrame:
    return pd.DataFrame([{
        "test_mode": "PENDING_REAL_TRADING_DAY_VALIDATION",
        "audit_asof": AUDIT_ASOF,
        "request_timestamp": "",
        "response_timestamp": "",
        "latency_ms": math.nan,
        "candidate_count": math.nan,
        "success_count": math.nan,
        "missing_count": math.nan,
        "stale_count": math.nan,
        "error_count": math.nan,
        "retry_count": math.nan,
        "whole_pool_complete_timestamp": "",
        "earliest_reliable_live_data_time": "PENDING_REAL_TRADING_DAY_VALIDATION",
        "live_core_field_completeness": math.nan,
        "p95_total_pipeline_time_seconds": math.nan,
        "live_gate_pass": 0,
        "reason": "Audit date is Sunday; no 09:25-09:30 real trading-window observation was fabricated",
    }])


def build_source_shift() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "field": "auction_price", "historical_source": "DAILY_UNADJUSTED_OFFICIAL_OPEN_REFERENCE",
            "live_source": "EASTMONEY_SPOT_OPEN_PROSPECTIVE", "historical_semantics": "OFFICIAL_D2_OPEN",
            "live_semantics": "CURRENT_OPEN_FIELD_AT_0925_NOT_YET_VALIDATED", "semantic_equivalence": "UNPROVEN",
            "source_shift": "UNKNOWN",
        },
        {
            "field": "auction_volume", "historical_source": "NONE",
            "live_source": "EASTMONEY_PRE_MIN_OR_SPOT_PROSPECTIVE", "historical_semantics": "NOT_AVAILABLE",
            "live_semantics": "MATCHED_VS_VIRTUAL_OR_CUMULATIVE_UNRESOLVED", "semantic_equivalence": "NOT_COMPARABLE",
            "source_shift": "UNKNOWN",
        },
        {
            "field": "auction_amount", "historical_source": "NONE",
            "live_source": "EASTMONEY_PRE_MIN_OR_SPOT_PROSPECTIVE", "historical_semantics": "NOT_AVAILABLE",
            "live_semantics": "MATCHED_VS_VIRTUAL_OR_CUMULATIVE_UNRESOLVED", "semantic_equivalence": "NOT_COMPARABLE",
            "source_shift": "UNKNOWN",
        },
    ])


def _historical_sample(detail: pd.DataFrame) -> pd.DataFrame:
    primary = detail[detail["in_primary_history_gate"].eq(1)].copy()
    pieces: list[pd.DataFrame] = []
    for (month, board), part in primary.groupby(["month", "board_group"], sort=True):
        pieces.append(part.head(5))
    cols = [
        "event_id", "code", "signal_date", "d2_date", "month", "board_group",
        "official_d2_open", "daily_source", "auction_price_equivalent_available",
        "independent_auction_price_snapshot_available", "auction_volume_available",
        "auction_amount_available", "historical_core_fields_complete", "minute_first_time",
        "preopen_bar_count", "bar_0925_count", "missing_reason",
    ]
    return pd.concat(pieces, ignore_index=True)[cols]


def _review(context: Mapping[str, Any]) -> str:
    history = context["history"]
    dry = context["dry"]
    primary_rows = history["primary_rows"]
    price_pct = history["price_equivalent_coverage"] * 100
    return f"""# v004d Auction Data Feasibility Audit v001

## 先说人话

1. **历史上不能完整看到 D2 09:25 集合竞价结果。** 现有缓存可从 D2 日线 open 找到开盘价等价参考，但没有独立 09:25 snapshot，也没有可分离的竞价成交量和成交额。
2. **价格参考覆盖 {price_pct:.2f}%，但核心三字段覆盖 0%。** 主窗口共有 {primary_rows} 个 candidate-D2；auction volume/amount 都不能从全日量额或 09:35 bar 伪造。
3. **现有 5 分钟 K 不是集合竞价 K。** Sina/BaoStock 的第一根规范化 bar 是 09:35，已经混入 09:30 后连续竞价。
4. AKShare/Eastmoney 的盘前接口能返回当前/最近交易日 09:15--09:30 状态，但不能回放 May--July，量额究竟是虚拟撮合、累计状态还是最终撮合量仍未被仓库语义证明。
5. 今天是周日，**真实 09:25--09:30 latency 未测试**。实现了 collector、时间戳日志与 N=5/10/15/20/30 dry-run；dry-run P95 最慢约 {dry['max_p95_ms']:.3f} ms，但它不含网络 fetch，不能代替 live PASS。
6. 因历史核心数据缺失，当前 auction 路线还没有资格进入 outcome information audit。

## Q1. 历史上到底能不能看到 D2 09:25 结果？

只能看到官方 D2 open 的价格等价参考；看不到独立保存的 09:25 price snapshot、auction volume 和 auction amount，因此答案是 **不能完整看到**。

## Q2. 价格、成交量、成交额是否完整可靠？

- official-open price reference：{price_pct:.2f}% coverage；但没有独立 auction feed，无法做独立 exact-match lineage 验证。
- auction volume：0% historical coverage。
- auction amount：0% historical coverage。
- 三字段同时完整：0%，未达到 overall 90% / monthly 85% gate。

## Q3. 这些量是真实集合竞价结果，还是普通 5 分钟 K 的伪解释？

日线 open 只保留作最终开盘价参考；全日 volume/amount 与 09:35 首根 bar 都明确不是 auction-only 数据，未作伪解释。当前盘前接口量额语义仍是 `UNRESOLVED`。

## Q4. 两根 5 分钟竞价数据有没有明确含义？

仓库没有原生 09:15--09:20、09:20--09:25 两根 auction 5m 记录。当前 AKShare 接口提供 1 分钟 supplier state，不是普通成交 OHLCV，也不支持 May--July 回放。因此 `AUCTION_5M_DATA_STATE = NOT_AVAILABLE`；不强迫保留。

## Q5. 实盘最早什么时候稳定可获取？

`PENDING_REAL_TRADING_DAY_VALIDATION`。非交易日 probe 不能回答 09:25 后稳定时点。

## Q6. 5/10/20/30 只需要多久？

本轮只完成 parser/join/basic-calculation dry-run；最大规模 P95 为 {dry['max_p95_ms']:.3f} ms。网络 fetch 未计入，详见 scale/timing CSV，不能用该数声明实盘通过。

## Q7. 09:30 前有没有足够时间？

尚不能判断。需要真实交易日 09:25--09:30 记录 source availability、完整率、stale 与 P95 total time；正式 gate 是 P95 <=120s 且核心字段完整率 >=95%。

## Q8. 历史研究与未来实盘能否使用同语义数据？

不能证明。历史只有 official-open price reference，volume/amount 缺失；live prospective source 是 Eastmoney current snapshot，`HISTORICAL_LIVE_SOURCE_SHIFT = UNKNOWN`。

## Q9. 是否有资格进入下一阶段？

没有。问题不是计算速度，而是历史核心数据缺失和量额语义未确认。

## 数据纪律

- Candidate universe：完整 Board2/Board3 first-break，S2 不做 hard filter。
- 输入 CSV 使用显式 identity/date/board 列投影；outcome、future return、model score/rank 未载入。
- 未训练模型、未搜索 feature、未创建 BUY/PASS、entry 仍为 D2 open。
- Sunday probe 只验证接口可达性/解析：盘前接口一次成功（13.804s，返回 2026-09-04 的 16 行），一次复测失败；bid/ask probe 失败。均不属于 live latency evidence。

## Formal result

V004D_CANDIDATE_UNIVERSE = BOARD2_BOARD3_FIRST_BREAK_ALL_CANDIDATES

D1_HARD_FILTER = NO

HISTORICAL_AUCTION_DATA_STATE = NOT_READY

LIVE_AUCTION_DATA_STATE = PENDING_REAL_TRADING_DAY_VALIDATION

AUCTION_FIELD_SEMANTICS_STATE = UNRESOLVED

AUCTION_5M_DATA_STATE = NOT_AVAILABLE

HISTORICAL_LIVE_SOURCE_SHIFT = UNKNOWN

CORE_FIELDS_AVAILABLE = OFFICIAL_D2_OPEN_PRICE_REFERENCE_ONLY

EARLIEST_RELIABLE_LIVE_DATA_TIME = PENDING_REAL_TRADING_DAY_VALIDATION

P95_TOTAL_PIPELINE_TIME = NOT_MEASURED_LIVE

V004D_AUCTION_DATA_FEASIBILITY_STATE = V004D_AUCTION_DATA_NOT_FEASIBLE

OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

BUY_PASS_RULE_CREATED = NO

ENTRY_PRICE_DEFINITION = D2_OPEN

NEXT_ACTION = STOP_AUCTION_PIPELINE
"""


def build_outputs(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    root_path = Path(root).resolve()
    candidates, candidate_audit = load_blind_candidate_universe(root_path)
    detail, coverage_summary, price_lineage, history_audit = build_historical_audit(
        root_path, candidates
    )
    scale_summary, timing_detail = build_dry_run_timing()
    inventory = build_source_inventory()
    field_semantics = build_field_semantics()
    volume_semantics = build_volume_amount_semantics()
    five_min = build_5m_semantics(history_audit)
    live_pending = build_live_pending()
    source_shift = build_source_shift()
    sample = _historical_sample(detail)

    # Put summary and row-level detail in one machine-readable coverage file.
    union_columns = list(coverage_summary.columns) + [
        column for column in detail.columns if column not in coverage_summary.columns
    ]
    historical_coverage = pd.concat(
        [coverage_summary.reindex(columns=union_columns), detail.reindex(columns=union_columns)],
        ignore_index=True,
    )

    context = {
        "candidate": candidate_audit,
        "history": history_audit,
        "dry": {
            "max_p95_ms": float(scale_summary["total_pipeline_time_ms_p95"].max()),
        },
        "formal": {
            "historical_state": "NOT_READY",
            "live_state": "PENDING_REAL_TRADING_DAY_VALIDATION",
            "semantics_state": "UNRESOLVED",
            "auction_5m_state": "NOT_AVAILABLE",
            "source_shift": "UNKNOWN",
            "feasibility_state": "V004D_AUCTION_DATA_NOT_FEASIBLE",
            "next_action": "STOP_AUCTION_PIPELINE",
        },
        "outcome_accessed": "NO",
        "model_trained": "NO",
        "feature_search": "NO",
        "buy_pass_rule_created": "NO",
        "external_source_probe_context": "EXISTING_EASTMONEY_AKSHARE_ONLY_NON_TRADING_DAY",
    }
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(inventory),
        OUTPUT_FILENAMES[1]: _csv_bytes(field_semantics),
        OUTPUT_FILENAMES[2]: _csv_bytes(historical_coverage),
        OUTPUT_FILENAMES[3]: _csv_bytes(price_lineage),
        OUTPUT_FILENAMES[4]: _csv_bytes(volume_semantics),
        OUTPUT_FILENAMES[5]: _csv_bytes(five_min),
        OUTPUT_FILENAMES[6]: _csv_bytes(sample),
        OUTPUT_FILENAMES[7]: _csv_bytes(live_pending),
        OUTPUT_FILENAMES[8]: _csv_bytes(scale_summary),
        OUTPUT_FILENAMES[9]: _csv_bytes(timing_detail),
        OUTPUT_FILENAMES[10]: _csv_bytes(source_shift),
        OUTPUT_FILENAMES[11]: _review(context).encode("utf-8"),
    }
    if tuple(outputs) != OUTPUT_FILENAMES:
        raise RuntimeError("FATAL: output manifest mismatch")
    return outputs, context


def run(root: str | Path) -> tuple[Path, dict[str, Any], dict[str, str]]:
    root_path = Path(root).resolve()
    outputs, context = build_outputs(root_path)
    output_dir = root_path / "reports/research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, payload in outputs.items():
        path = output_dir / name
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return output_dir, context, hashes

"""Blind coverage/lineage audit for conservative 5-minute seal-path proxies.

This module deliberately reads only the frozen candidate identity lock, local
5-minute caches, unadjusted daily caches, and historical limit-up-pool rows
used for reconstruction sanity checks.  It never loads outcomes, model scores,
or model ranks.

The three preregistered proxies are:

* F1: first 5-minute LOCKED bar of the final uninterrupted LOCKED suffix;
* F2: a conservative reopen flag (a non-LOCKED bar after a LOCKED bar);
* F3: D0 F1 trading-minute coordinate minus PREV_BOARD_DAY F1 coordinate.

They are coarse 5-minute observations, not tick-level first/final seal times or
true open-board counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TASK_NAME = "v004c_5m_conservative_seal_path_coverage_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
DEVELOPMENT_START = "2026-05-06"
DEVELOPMENT_END = "2026-07-29"

IDENTITY_LOCK_RELATIVE_PATH = (
    "reports/research/v004c_pre_break_board_day_seal_path_information_audit_v001/"
    "v004c_prebreak_seal_path_row_eligibility_v001.csv"
)

IDENTITY_COLUMNS = (
    "event_id",
    "code",
    "signal_date",
    "month",
    "d0_date",
    "prev_board_day",
    "d1_date",
    "board_group",
)

FORBIDDEN_INPUT_COLUMNS = frozenset(
    {
        "target7",
        "loss",
        "severe_loss",
        "capped_return",
        "capped_return_7",
        "raw_repair_return",
        "stage1_score",
        "stage1_rank",
        "s2_score",
        "s2_rank",
        "model_prediction",
    }
)

EXPECTED_BAR_CLOCKS = tuple(
    [f"{minute // 60:02d}:{minute % 60:02d}:00" for minute in range(9 * 60 + 35, 11 * 60 + 31, 5)]
    + [f"{minute // 60:02d}:{minute % 60:02d}:00" for minute in range(13 * 60 + 5, 15 * 60 + 1, 5)]
)
EXPECTED_BAR_COUNT = 48

# Repository-wide main-board price-comparison tolerance.  It is one RMB cent
# plus a floating-point epsilon; the audit does not tune it.
PRICE_TOLERANCE = 0.011
LIMIT_RATIO = Decimal("1.10")

SOURCE_SPECS = (
    {
        "cache_kind": "baostock_5m",
        "folder": "data/cache/baostock_5m",
        "accepted_sources": frozenset({"baostock_5m"}),
    },
    {
        "cache_kind": "minute_5m",
        "folder": "data/cache/minute_5m",
        "accepted_sources": frozenset(
            {
                "sina_5m",
                "sina_5m_closing_auction_repaired",
                "sina_5m_daily_aggregate_repaired",
            }
        ),
    },
)

MISSING_REASONS = (
    "NO_5M_DATA",
    "PARTIAL_5M_DAY",
    "INVALID_OHLC",
    "LIMIT_PRICE_UNAVAILABLE",
    "LIMIT_PRICE_MISMATCH",
    "NO_FULL_LOCK_BAR",
    "PREV_BOARD_DAY_MISSING",
    "D0_MISSING",
    "SOURCE_SEMANTICS_RISK",
    "OTHER_EXPLICIT",
)

F1_OVERALL_GATE = 0.80
F1_MONTH_GATE = 0.70
F1_BOARD_GATE = 0.65
F2_OVERALL_GATE = 0.80
F2_MONTH_GATE = 0.70
F2_BOARD_GATE = 0.65
F3_OVERALL_GATE = 0.70
F3_MONTH_GATE = 0.60
F3_BOARD_GATE = 0.55

F3_PARTIAL_OVERALL_FLOOR = 0.60
F3_PARTIAL_MONTH_FLOOR = 0.50

RELIABLE_POOL_SOURCE = "akshare_zt_pool_em"

OUTPUT_FILENAMES = (
    "v004c_5m_seal_path_lineage_v001.csv",
    "v004c_5m_seal_path_bar_quality_v001.csv",
    "v004c_5m_seal_path_reconstruction_values_v001.csv",
    "v004c_5m_seal_path_coverage_v001.csv",
    "v004c_5m_seal_path_missing_reasons_v001.csv",
    "v004c_5m_seal_path_source_mix_v001.csv",
    "v004c_5m_vs_limitup_final_seal_validation_v001.csv",
    "v004c_5m_vs_limitup_reopen_validation_v001.csv",
    "v004c_5m_seal_path_phase_a_lock_v001.csv",
    "v004c_5m_conservative_seal_path_coverage_review_v001.md",
)


@dataclass(frozen=True)
class DayAudit:
    code: str
    board_date: str
    cache_kind: str
    cache_path: str
    cache_sources: str
    adjustment: str
    interval: str
    bar_count: int
    first_bar_time: str
    last_bar_time: str
    duplicate_bar_count: int
    timestamp_order_valid: int
    expected_grid_match: int
    ohlc_valid: int
    volume_amount_valid: int
    source_semantics_valid: int
    daily_cache_kind: str
    daily_cache_path: str
    daily_source: str
    previous_trade_date: str
    previous_close: float
    daily_close: float
    daily_high: float
    limit_price: float
    limit_price_valid: int
    daily_minute_price_match: int
    day_complete: int
    locked_bar_count: int
    touched_mixed_bar_count: int
    off_limit_bar_count: int
    uncertain_bar_count: int
    f1_valid: int
    f1_clock_time: str
    f1_trading_minute: float
    stable_suffix_bars: int
    late_single_bar_lock: int
    f2_valid: int
    confirmed_reopen: float
    first_locked_clock_time: str
    bar_state_sequence: str
    bar_state_sha256: str
    missing_reason: str
    rejected_source_notes: str


def _normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    return digits[-6:].zfill(6)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig"))


def round_main_board_limit_price(previous_close: float) -> float:
    """Round previous close * 1.10 to the RMB-cent tick, half up."""
    return float(
        (Decimal(str(float(previous_close))) * LIMIT_RATIO).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


def trading_minute_from_clock(value: Any) -> float:
    """Map a clock time to effective trading minutes since 09:30.

    Lunch is removed: 13:05 maps to 125 rather than 215.
    Pre-open auction times map to zero for cross-validation only.
    """
    parsed = pd.to_datetime(str(value), format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        return math.nan
    minute = int(parsed.hour) * 60 + int(parsed.minute) + int(parsed.second) / 60.0
    if minute <= 9 * 60 + 30:
        return 0.0
    if minute <= 11 * 60 + 30:
        return minute - (9 * 60 + 30)
    if minute < 13 * 60:
        return 120.0
    return 120.0 + minute - 13 * 60


def _bar_bucket_end(value: Any) -> str:
    """Return the conservative 5m bar end containing a pool timestamp."""
    parsed = pd.to_datetime(str(value), format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        return ""
    minute = int(parsed.hour) * 60 + int(parsed.minute)
    second = int(parsed.second)
    if minute <= 9 * 60 + 35:
        return "09:35:00"
    if minute <= 11 * 60 + 30:
        offset = minute - (9 * 60 + 30)
        end_offset = int(math.ceil((offset + second / 60.0) / 5.0) * 5)
        end_minute = min(11 * 60 + 30, 9 * 60 + 30 + end_offset)
    elif minute < 13 * 60 + 5:
        end_minute = 13 * 60 + 5
    else:
        offset = minute - 13 * 60
        end_offset = int(math.ceil((offset + second / 60.0) / 5.0) * 5)
        end_minute = min(15 * 60, 13 * 60 + end_offset)
    return f"{end_minute // 60:02d}:{end_minute % 60:02d}:00"


def _expected_first_full_bar_end_after(value: Any) -> str:
    """Expected end time of the first complete 5m bar after a tick seal.

    F1 cannot normally equal the bar containing the tick-level final seal: that
    bar may include pre-seal trades below the limit.  The next complete bar is
    the apples-to-apples conservative expectation.
    """
    parsed = pd.to_datetime(str(value), format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        return ""
    minute = int(parsed.hour) * 60 + int(parsed.minute)
    second = int(parsed.second)
    if minute <= 9 * 60 + 30:
        return "09:35:00"
    if minute <= 11 * 60 + 30:
        offset = minute - (9 * 60 + 30) + second / 60.0
        containing_end = 9 * 60 + 30 + int(math.ceil(offset / 5.0) * 5)
        next_end = containing_end + 5
        if next_end <= 11 * 60 + 30:
            return f"{next_end // 60:02d}:{next_end % 60:02d}:00"
        return "13:05:00"
    if minute < 13 * 60:
        return "13:05:00"
    offset = minute - 13 * 60 + second / 60.0
    containing_end = 13 * 60 + int(math.ceil(offset / 5.0) * 5)
    next_end = containing_end + 5
    if next_end > 15 * 60:
        return ""
    return f"{next_end // 60:02d}:{next_end % 60:02d}:00"


def load_frozen_identity(root: str | Path) -> pd.DataFrame:
    root_path = Path(root).resolve()
    path = root_path / IDENTITY_LOCK_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    forbidden = sorted(FORBIDDEN_INPUT_COLUMNS.intersection(header.columns))
    if forbidden:
        raise RuntimeError(f"identity lock contains forbidden fields: {forbidden}")
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=list(IDENTITY_COLUMNS),
        dtype={"event_id": str, "code": str},
    )
    frame["code"] = frame["code"].map(_normalize_code)
    for column in ("signal_date", "d0_date", "prev_board_day", "d1_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.strftime("%Y-%m-%d")
    frame = frame[
        frame["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
    ].copy()
    frame = frame.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)
    if (len(frame), frame["signal_date"].nunique()) != (485, 60):
        raise RuntimeError(
            f"frozen identity parity failed: {len(frame)} rows / "
            f"{frame['signal_date'].nunique()} dates"
        )
    if frame["event_id"].duplicated().any():
        raise RuntimeError("duplicate event_id in frozen identity lock")
    if set(frame["board_group"].astype(str)) - {"BOARD2", "BOARD3"}:
        raise RuntimeError("unexpected board_group in frozen identity lock")
    if not (frame["d0_date"] < frame["d1_date"]).all():
        raise RuntimeError("D0 must precede D1")
    if not (frame["prev_board_day"] < frame["d0_date"]).all():
        raise RuntimeError("PREV_BOARD_DAY must precede D0")
    return frame


def _prepare_minute_day(frame: pd.DataFrame, board_date: str) -> pd.DataFrame:
    data = frame.copy()
    if "trade_date" not in data.columns and "datetime" in data.columns:
        data["trade_date"] = pd.to_datetime(data["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["trade_date"] = data["trade_date"].astype(str)
    data = data[data["trade_date"].eq(board_date)].copy()
    if data.empty:
        return data
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    if "time" not in data.columns:
        data["time"] = data["datetime"].dt.strftime("%H:%M:%S")
    else:
        raw_time = data["time"].astype(str).str.strip()
        normalized_time = pd.to_datetime(raw_time, format="%H:%M:%S", errors="coerce")
        fallback = data["datetime"].dt.strftime("%H:%M:%S")
        data["time"] = normalized_time.dt.strftime("%H:%M:%S").fillna(fallback)
    return data.sort_values("datetime", kind="mergesort").reset_index(drop=True)


def _validate_minute_source(
    frame: pd.DataFrame,
    accepted_sources: frozenset[str],
) -> dict[str, Any]:
    # Some older canonical ``minute_5m`` cache files predate the redundant
    # ``interval`` marker.  Their accepted source, ``adjust=none`` marker and
    # exact 48-point 5-minute clock grid prove the same semantics.  Missing an
    # interval metadata column therefore remains visible in lineage but is not
    # itself a reason to discard otherwise complete bars.
    required = {"datetime", "trade_date", "time", "open", "high", "low", "close", "volume", "amount", "source", "adjust"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {
            "source_semantics_valid": 0,
            "timestamp_order_valid": 0,
            "duplicate_bar_count": 0,
            "expected_grid_match": 0,
            "ohlc_valid": 0,
            "volume_amount_valid": 0,
            "reason": "SOURCE_SEMANTICS_RISK",
            "notes": f"missing_columns={','.join(missing_columns)}",
        }
    dt = pd.to_datetime(frame["datetime"], errors="coerce")
    duplicate_count = int(dt.duplicated().sum()) if not dt.isna().all() else 0
    order_valid = int(not dt.isna().any() and dt.is_monotonic_increasing and duplicate_count == 0)
    clocks = tuple(frame["time"].astype(str).tolist())
    grid_match = int(clocks == EXPECTED_BAR_CLOCKS)
    numeric = frame[["open", "high", "low", "close", "volume", "amount"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_price = np.isfinite(numeric[["open", "high", "low", "close"]]).all(axis=1)
    positive_price = (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
    high_ok = numeric["high"].ge(numeric[["open", "close", "low"]].max(axis=1))
    low_ok = numeric["low"].le(numeric[["open", "close", "high"]].min(axis=1))
    ohlc_valid = int((finite_price & positive_price & high_ok & low_ok).all())
    volume_amount_valid = int(
        np.isfinite(numeric[["volume", "amount"]]).all(axis=1).all()
        and (numeric[["volume", "amount"]] >= 0).all(axis=1).all()
    )
    sources = set(frame["source"].astype(str))
    interval_values = set(frame["interval"].astype(str)) if "interval" in frame.columns else set()
    interval_valid = not interval_values or interval_values == {"5m"}
    source_valid = int(
        bool(sources)
        and sources.issubset(accepted_sources)
        and set(frame["adjust"].astype(str)) == {"none"}
        and interval_valid
    )
    if not source_valid:
        reason = "SOURCE_SEMANTICS_RISK"
    elif len(frame) != EXPECTED_BAR_COUNT or not grid_match or not order_valid:
        reason = "PARTIAL_5M_DAY"
    elif not ohlc_valid:
        reason = "INVALID_OHLC"
    elif not volume_amount_valid:
        reason = "SOURCE_SEMANTICS_RISK"
    else:
        reason = ""
    return {
        "source_semantics_valid": source_valid,
        "timestamp_order_valid": order_valid,
        "duplicate_bar_count": duplicate_count,
        "expected_grid_match": grid_match,
        "ohlc_valid": ohlc_valid,
        "volume_amount_valid": volume_amount_valid,
        "reason": reason,
        "notes": "",
    }


class DailyPriceStore:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[tuple[str, str], tuple[pd.DataFrame, Path] | None] = {}

    def _load(self, code: str, kind: str) -> tuple[pd.DataFrame, Path] | None:
        key = (code, kind)
        if key in self._cache:
            return self._cache[key]
        path = self.root / "data/cache" / kind / f"{code}_daily.pkl"
        if not path.is_file():
            self._cache[key] = None
            return None
        frame = pd.read_pickle(path).copy()
        if "date" not in frame.columns:
            self._cache[key] = None
            return None
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date"]).sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
        self._cache[key] = (frame, path)
        return self._cache[key]

    def lookup(self, code: str, board_date: str) -> dict[str, Any]:
        for kind in ("daily_unadjusted", "daily"):
            loaded = self._load(code, kind)
            if loaded is None:
                continue
            frame, path = loaded
            matches = frame.index[frame["date"].eq(board_date)].tolist()
            if not matches:
                continue
            idx = int(matches[-1])
            if idx <= 0:
                continue
            today = frame.iloc[idx]
            previous = frame.iloc[idx - 1]
            values = pd.to_numeric(
                pd.Series(
                    {
                        "previous_close": previous.get("close"),
                        "daily_close": today.get("close"),
                        "daily_high": today.get("high"),
                        "daily_low": today.get("low"),
                        "daily_open": today.get("open"),
                    }
                ),
                errors="coerce",
            )
            return {
                "daily_cache_kind": kind,
                "daily_cache_path": _relative(path, self.root),
                "daily_source": str(today.get("source", "")),
                "previous_trade_date": str(previous["date"]),
                **{key: float(value) if pd.notna(value) else math.nan for key, value in values.items()},
            }
        return {
            "daily_cache_kind": "",
            "daily_cache_path": "",
            "daily_source": "",
            "previous_trade_date": "",
            "previous_close": math.nan,
            "daily_close": math.nan,
            "daily_high": math.nan,
            "daily_low": math.nan,
            "daily_open": math.nan,
        }


class MinuteStore:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[tuple[str, str], tuple[pd.DataFrame, Path] | None] = {}

    def _load(self, code: str, spec: Mapping[str, Any]) -> tuple[pd.DataFrame, Path] | None:
        key = (code, str(spec["cache_kind"]))
        if key in self._cache:
            return self._cache[key]
        path = self.root / str(spec["folder"]) / f"{code}_5min.pkl"
        if not path.is_file():
            self._cache[key] = None
            return None
        frame = pd.read_pickle(path)
        self._cache[key] = (frame, path)
        return self._cache[key]

    def candidates(self, code: str, board_date: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for spec in SOURCE_SPECS:
            loaded = self._load(code, spec)
            if loaded is None:
                continue
            frame, path = loaded
            day = _prepare_minute_day(frame, board_date)
            if day.empty:
                continue
            qa = _validate_minute_source(day, spec["accepted_sources"])
            result.append(
                {
                    "cache_kind": str(spec["cache_kind"]),
                    "cache_path": _relative(path, self.root),
                    "frame": day,
                    "qa": qa,
                }
            )
        return result


def classify_bar_states(frame: pd.DataFrame, limit_price: float) -> pd.Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    locked = low.ge(limit_price - PRICE_TOLERANCE) & high.le(limit_price + PRICE_TOLERANCE)
    touched = high.ge(limit_price - PRICE_TOLERANCE) & low.lt(limit_price - PRICE_TOLERANCE)
    values = np.select(
        [locked, touched, high.lt(limit_price - PRICE_TOLERANCE)],
        ["LOCKED", "TOUCHED_MIXED", "OFF_LIMIT"],
        default="UNCERTAIN",
    )
    return pd.Series(values, index=frame.index, dtype="object")


def reconstruct_day(
    root: Path,
    code: str,
    board_date: str,
    minute_store: MinuteStore,
    daily_store: DailyPriceStore,
) -> DayAudit:
    source_candidates = minute_store.candidates(code, board_date)
    daily = daily_store.lookup(code, board_date)
    previous_close = float(daily["previous_close"])
    limit_price = (
        round_main_board_limit_price(previous_close)
        if np.isfinite(previous_close) and previous_close > 0
        else math.nan
    )

    rejected_notes: list[str] = []
    chosen: dict[str, Any] | None = None
    chosen_price_valid = 0
    chosen_daily_minute_match = 0
    for candidate in source_candidates:
        frame = candidate["frame"]
        qa = candidate["qa"]
        if qa["reason"]:
            rejected_notes.append(f"{candidate['cache_kind']}:{qa['reason']}")
            continue
        if not np.isfinite(limit_price):
            rejected_notes.append(f"{candidate['cache_kind']}:LIMIT_PRICE_UNAVAILABLE")
            continue
        prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        daily_price_ready = all(
            np.isfinite(float(daily[key])) for key in ("daily_close", "daily_high")
        )
        daily_limit_match = bool(
            daily_price_ready
            and abs(float(daily["daily_close"]) - limit_price) <= PRICE_TOLERANCE
            and abs(float(daily["daily_high"]) - limit_price) <= PRICE_TOLERANCE
        )
        minute_limit_match = bool(
            abs(float(prices["high"].max()) - limit_price) <= PRICE_TOLERANCE
            and abs(float(prices["close"].iloc[-1]) - limit_price) <= PRICE_TOLERANCE
            and float(prices["high"].max()) <= limit_price + PRICE_TOLERANCE
        )
        daily_minute_match = bool(
            daily_price_ready
            and abs(float(prices["high"].max()) - float(daily["daily_high"])) <= PRICE_TOLERANCE
            and abs(float(prices["close"].iloc[-1]) - float(daily["daily_close"])) <= PRICE_TOLERANCE
        )
        price_valid = int(daily_limit_match and minute_limit_match and daily_minute_match)
        if not price_valid:
            rejected_notes.append(f"{candidate['cache_kind']}:LIMIT_PRICE_MISMATCH")
            continue
        chosen = candidate
        chosen_price_valid = price_valid
        chosen_daily_minute_match = int(daily_minute_match)
        break

    if chosen is None:
        if not source_candidates:
            reason = "NO_5M_DATA"
            base = None
        else:
            base = source_candidates[0]
            reasons = [str(item["qa"]["reason"]) for item in source_candidates if item["qa"]["reason"]]
            if reasons:
                reason = reasons[0]
            elif not np.isfinite(limit_price):
                reason = "LIMIT_PRICE_UNAVAILABLE"
            else:
                reason = "LIMIT_PRICE_MISMATCH"
        frame = pd.DataFrame() if base is None else base["frame"]
        qa = {
            "duplicate_bar_count": 0,
            "timestamp_order_valid": 0,
            "expected_grid_match": 0,
            "ohlc_valid": 0,
            "volume_amount_valid": 0,
            "source_semantics_valid": 0,
        } if base is None else base["qa"]
        return DayAudit(
            code=code,
            board_date=board_date,
            cache_kind="" if base is None else str(base["cache_kind"]),
            cache_path="" if base is None else str(base["cache_path"]),
            cache_sources="" if frame.empty or "source" not in frame else ";".join(sorted(set(frame["source"].astype(str)))),
            adjustment="" if frame.empty or "adjust" not in frame else ";".join(sorted(set(frame["adjust"].astype(str)))),
            interval=("" if frame.empty else (";".join(sorted(set(frame["interval"].astype(str)))) if "interval" in frame else "5m_grid_inferred")),
            bar_count=len(frame),
            first_bar_time="" if frame.empty else str(frame["time"].iloc[0]),
            last_bar_time="" if frame.empty else str(frame["time"].iloc[-1]),
            duplicate_bar_count=int(qa["duplicate_bar_count"]),
            timestamp_order_valid=int(qa["timestamp_order_valid"]),
            expected_grid_match=int(qa["expected_grid_match"]),
            ohlc_valid=int(qa["ohlc_valid"]),
            volume_amount_valid=int(qa["volume_amount_valid"]),
            source_semantics_valid=int(qa["source_semantics_valid"]),
            daily_cache_kind=str(daily["daily_cache_kind"]),
            daily_cache_path=str(daily["daily_cache_path"]),
            daily_source=str(daily["daily_source"]),
            previous_trade_date=str(daily["previous_trade_date"]),
            previous_close=previous_close,
            daily_close=float(daily["daily_close"]),
            daily_high=float(daily["daily_high"]),
            limit_price=limit_price,
            limit_price_valid=0,
            daily_minute_price_match=0,
            day_complete=0,
            locked_bar_count=0,
            touched_mixed_bar_count=0,
            off_limit_bar_count=0,
            uncertain_bar_count=0,
            f1_valid=0,
            f1_clock_time="",
            f1_trading_minute=math.nan,
            stable_suffix_bars=0,
            late_single_bar_lock=0,
            f2_valid=0,
            confirmed_reopen=math.nan,
            first_locked_clock_time="",
            bar_state_sequence="",
            bar_state_sha256="",
            missing_reason=reason if reason in MISSING_REASONS else "OTHER_EXPLICIT",
            rejected_source_notes=";".join(rejected_notes),
        )

    frame = chosen["frame"].copy()
    qa = chosen["qa"]
    states = classify_bar_states(frame, limit_price)
    frame["bar_state"] = states
    state_sequence = ";".join(
        f"{clock}={state}"
        for clock, state in zip(frame["time"].astype(str), states.astype(str), strict=True)
    )
    locked_indices = np.flatnonzero(states.eq("LOCKED").to_numpy())
    f1_valid = 0
    f1_clock = ""
    f1_minute = math.nan
    suffix_bars = 0
    late_single = 0
    if len(locked_indices) and states.iloc[-1] == "LOCKED":
        start = len(states) - 1
        while start > 0 and states.iloc[start - 1] == "LOCKED":
            start -= 1
        suffix_bars = len(states) - start
        f1_clock = str(frame["time"].iloc[start])
        f1_minute = trading_minute_from_clock(f1_clock)
        f1_valid = 1
        late_single = int(suffix_bars == 1)
    first_locked_clock = ""
    f2_valid = 0
    confirmed_reopen = math.nan
    if len(locked_indices):
        first_locked = int(locked_indices[0])
        first_locked_clock = str(frame["time"].iloc[first_locked])
        confirmed_reopen = float((states.iloc[first_locked + 1 :] != "LOCKED").any())
        f2_valid = 1
    reason = "" if f1_valid and f2_valid else "NO_FULL_LOCK_BAR"
    return DayAudit(
        code=code,
        board_date=board_date,
        cache_kind=str(chosen["cache_kind"]),
        cache_path=str(chosen["cache_path"]),
        cache_sources=";".join(sorted(set(frame["source"].astype(str)))),
        adjustment=";".join(sorted(set(frame["adjust"].astype(str)))),
        interval=(";".join(sorted(set(frame["interval"].astype(str)))) if "interval" in frame else "5m_grid_inferred"),
        bar_count=len(frame),
        first_bar_time=str(frame["time"].iloc[0]),
        last_bar_time=str(frame["time"].iloc[-1]),
        duplicate_bar_count=int(qa["duplicate_bar_count"]),
        timestamp_order_valid=int(qa["timestamp_order_valid"]),
        expected_grid_match=int(qa["expected_grid_match"]),
        ohlc_valid=int(qa["ohlc_valid"]),
        volume_amount_valid=int(qa["volume_amount_valid"]),
        source_semantics_valid=int(qa["source_semantics_valid"]),
        daily_cache_kind=str(daily["daily_cache_kind"]),
        daily_cache_path=str(daily["daily_cache_path"]),
        daily_source=str(daily["daily_source"]),
        previous_trade_date=str(daily["previous_trade_date"]),
        previous_close=previous_close,
        daily_close=float(daily["daily_close"]),
        daily_high=float(daily["daily_high"]),
        limit_price=limit_price,
        limit_price_valid=chosen_price_valid,
        daily_minute_price_match=chosen_daily_minute_match,
        day_complete=1,
        locked_bar_count=int(states.eq("LOCKED").sum()),
        touched_mixed_bar_count=int(states.eq("TOUCHED_MIXED").sum()),
        off_limit_bar_count=int(states.eq("OFF_LIMIT").sum()),
        uncertain_bar_count=int(states.eq("UNCERTAIN").sum()),
        f1_valid=f1_valid,
        f1_clock_time=f1_clock,
        f1_trading_minute=f1_minute,
        stable_suffix_bars=suffix_bars,
        late_single_bar_lock=late_single,
        f2_valid=f2_valid,
        confirmed_reopen=confirmed_reopen,
        first_locked_clock_time=first_locked_clock,
        bar_state_sequence=state_sequence,
        bar_state_sha256=_sha256_bytes(state_sequence.encode("utf-8")),
        missing_reason=reason,
        rejected_source_notes=";".join(rejected_notes),
    )


def build_day_audits(root: str | Path, population: pd.DataFrame) -> pd.DataFrame:
    root_path = Path(root).resolve()
    minute_store = MinuteStore(root_path)
    daily_store = DailyPriceStore(root_path)
    unique_days = sorted(
        {
            (str(row.code), str(board_date))
            for row in population.itertuples()
            for board_date in (row.d0_date, row.prev_board_day)
        }
    )
    rows = [
        reconstruct_day(root_path, code, board_date, minute_store, daily_store).__dict__
        for code, board_date in unique_days
    ]
    return pd.DataFrame(rows).sort_values(["board_date", "code"], kind="mergesort").reset_index(drop=True)


def build_candidate_outputs(
    population: pd.DataFrame, day_audits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = {
        (str(row.code), str(row.board_date)): row._asdict()
        for row in day_audits.itertuples(index=False)
    }
    lineage_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for event in population.to_dict("records"):
        side_values: dict[str, dict[str, Any]] = {}
        for side, date_column in (("D0", "d0_date"), ("PREV_BOARD_DAY", "prev_board_day")):
            audit = dict(lookup[(str(event["code"]), str(event[date_column]))])
            side_values[side] = audit
            lineage_rows.append(
                {
                    "event_id": str(event["event_id"]),
                    "code": str(event["code"]),
                    "signal_date": str(event["signal_date"]),
                    "month": str(event["month"]),
                    "board_group": str(event["board_group"]),
                    "d1_date": str(event["d1_date"]),
                    "d0_date": str(event["d0_date"]),
                    "prev_board_day": str(event["prev_board_day"]),
                    "expected_streak": 2 if str(event["board_group"]) == "BOARD2" else 3,
                    "board_day_role": side,
                    "board_day_before_d1": int(str(audit["board_date"]) < str(event["d1_date"])),
                    "last_two_board_chain_role_valid": int(
                        (side == "D0" and str(audit["board_date"]) == str(event["d0_date"]))
                        or (side == "PREV_BOARD_DAY" and str(audit["board_date"]) == str(event["prev_board_day"]))
                    ),
                    **audit,
                }
            )
        d0 = side_values["D0"]
        prev = side_values["PREV_BOARD_DAY"]
        f3_valid = int(bool(d0["f1_valid"]) and bool(prev["f1_valid"]))
        if f3_valid:
            f3 = float(d0["f1_trading_minute"]) - float(prev["f1_trading_minute"])
            f3_reason = ""
        else:
            f3 = math.nan
            if not d0["f1_valid"]:
                f3_reason = "D0_MISSING"
            elif not prev["f1_valid"]:
                f3_reason = "PREV_BOARD_DAY_MISSING"
            else:
                f3_reason = "OTHER_EXPLICIT"
        value_rows.append(
            {
                "event_id": str(event["event_id"]),
                "code": str(event["code"]),
                "signal_date": str(event["signal_date"]),
                "month": str(event["month"]),
                "board_group": str(event["board_group"]),
                "d1_date": str(event["d1_date"]),
                "d0_date": str(event["d0_date"]),
                "prev_board_day": str(event["prev_board_day"]),
                "d0_cache_kind": str(d0["cache_kind"]),
                "prev_cache_kind": str(prev["cache_kind"]),
                "d0_limit_price": d0["limit_price"],
                "prev_limit_price": prev["limit_price"],
                "d0_f1_valid": int(d0["f1_valid"]),
                "d0_final_confirmed_stable_lock_clock": str(d0["f1_clock_time"]),
                "d0_final_confirmed_stable_lock_minute": d0["f1_trading_minute"],
                "d0_stable_suffix_bars": int(d0["stable_suffix_bars"]),
                "d0_late_single_bar_lock": int(d0["late_single_bar_lock"]),
                "d0_f2_valid": int(d0["f2_valid"]),
                "d0_confirmed_reopen": d0["confirmed_reopen"],
                "prev_f1_valid": int(prev["f1_valid"]),
                "prev_final_confirmed_stable_lock_clock": str(prev["f1_clock_time"]),
                "prev_final_confirmed_stable_lock_minute": prev["f1_trading_minute"],
                "prev_stable_suffix_bars": int(prev["stable_suffix_bars"]),
                "prev_late_single_bar_lock": int(prev["late_single_bar_lock"]),
                "prev_f2_valid": int(prev["f2_valid"]),
                "prev_confirmed_reopen": prev["confirmed_reopen"],
                "F1_FINAL_CONFIRMED_STABLE_LOCK_TIME": d0["f1_trading_minute"] if d0["f1_valid"] else math.nan,
                "F1_valid": int(d0["f1_valid"]),
                "F1_missing_reason": "" if d0["f1_valid"] else str(d0["missing_reason"]),
                "F2_CONFIRMED_REOPEN": d0["confirmed_reopen"] if d0["f2_valid"] else math.nan,
                "F2_valid": int(d0["f2_valid"]),
                "F2_missing_reason": "" if d0["f2_valid"] else str(d0["missing_reason"]),
                "F3_D0_VS_PREV_LOCK_DETERIORATION": f3,
                "F3_valid": f3_valid,
                "F3_missing_reason": f3_reason,
            }
        )
    lineage = pd.DataFrame(lineage_rows).sort_values(
        ["signal_date", "event_id", "board_day_role"], kind="mergesort"
    ).reset_index(drop=True)
    values = pd.DataFrame(value_rows).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    return lineage, values


def _coverage_row(
    values: pd.DataFrame,
    variable: str,
    valid_column: str,
    scope_type: str,
    scope_value: str,
) -> dict[str, Any]:
    if scope_type == "OVERALL":
        part = values
    elif scope_type == "MONTH":
        part = values[values["month"].eq(scope_value)]
    elif scope_type == "BOARD":
        part = values[values["board_group"].eq(scope_value)]
    else:
        raise ValueError(scope_type)
    valid = int(pd.to_numeric(part[valid_column], errors="coerce").fillna(0).astype(int).sum())
    total = len(part)
    return {
        "variable": variable,
        "scope_type": scope_type,
        "scope_value": scope_value,
        "candidate_rows": total,
        "valid_rows": valid,
        "coverage_rate": valid / total if total else math.nan,
        "valid_signal_dates": int(part.loc[part[valid_column].eq(1), "signal_date"].nunique()),
        "total_signal_dates": int(part["signal_date"].nunique()),
    }


def _field_ready(coverage: pd.DataFrame, variable: str) -> bool:
    rows = coverage[coverage["variable"].eq(variable)].set_index(["scope_type", "scope_value"])
    gates = (
        ("OVERALL", "ALL", F3_OVERALL_GATE if variable == "F3" else F1_OVERALL_GATE),
        ("MONTH", "MAY", F3_MONTH_GATE if variable == "F3" else F1_MONTH_GATE),
        ("MONTH", "JUNE", F3_MONTH_GATE if variable == "F3" else F1_MONTH_GATE),
        ("MONTH", "JULY", F3_MONTH_GATE if variable == "F3" else F1_MONTH_GATE),
        ("BOARD", "BOARD2", F3_BOARD_GATE if variable == "F3" else F1_BOARD_GATE),
        ("BOARD", "BOARD3", F3_BOARD_GATE if variable == "F3" else F1_BOARD_GATE),
    )
    return all(float(rows.loc[(kind, value), "coverage_rate"]) >= gate for kind, value, gate in gates)


def build_coverage(values: pd.DataFrame, lineage: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    for variable, valid_column in (("F1", "F1_valid"), ("F2", "F2_valid"), ("F3", "F3_valid")):
        rows.append(_coverage_row(values, variable, valid_column, "OVERALL", "ALL"))
        for month in ("MAY", "JUNE", "JULY"):
            rows.append(_coverage_row(values, variable, valid_column, "MONTH", month))
        for board in ("BOARD2", "BOARD3"):
            rows.append(_coverage_row(values, variable, valid_column, "BOARD", board))
    # Required side-level reconstruction availability for F1/F2, both overall
    # and by month/board.  These rows make the D0/PREV coverage explicit rather
    # than hiding the weaker side inside F3.
    for variable, valid_column in (("F1_SIDE", "f1_valid"), ("F2_SIDE", "f2_valid")):
        for side in ("D0", "PREV_BOARD_DAY"):
            part = lineage[lineage["board_day_role"].eq(side)]
            for scope_type, scope_value, scoped in (
                ("BOARD_DAY_ROLE", side, part),
                ("BOARD_DAY_ROLE_MONTH", f"{side}:MAY", part[part["month"].eq("MAY")]),
                ("BOARD_DAY_ROLE_MONTH", f"{side}:JUNE", part[part["month"].eq("JUNE")]),
                ("BOARD_DAY_ROLE_MONTH", f"{side}:JULY", part[part["month"].eq("JULY")]),
                ("BOARD_DAY_ROLE_BOARD", f"{side}:BOARD2", part[part["board_group"].eq("BOARD2")]),
                ("BOARD_DAY_ROLE_BOARD", f"{side}:BOARD3", part[part["board_group"].eq("BOARD3")]),
            ):
                valid = int(scoped[valid_column].sum())
                rows.append(
                    {
                        "variable": variable,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "candidate_rows": len(scoped),
                        "valid_rows": valid,
                        "coverage_rate": valid / len(scoped) if len(scoped) else math.nan,
                        "valid_signal_dates": int(scoped.loc[scoped[valid_column].eq(1), "signal_date"].nunique()),
                        "total_signal_dates": int(scoped["signal_date"].nunique()),
                    }
                )
    coverage = pd.DataFrame(rows)
    ready = {field: _field_ready(coverage, field) for field in ("F1", "F2", "F3")}
    coverage["field_data_ready"] = coverage["variable"].map(
        {"F1": "YES" if ready["F1"] else "NO", "F2": "YES" if ready["F2"] else "NO", "F3": "YES" if ready["F3"] else "NO"}
    ).fillna("NOT_APPLICABLE")
    return coverage, ready


def determine_data_state(coverage: pd.DataFrame, ready: Mapping[str, bool]) -> str:
    if all(ready.values()):
        return "5M_SEAL_PATH_DATA_READY"
    if ready.get("F1") and ready.get("F2"):
        f3 = coverage[
            coverage["variable"].eq("F3")
            & coverage["scope_type"].isin(["OVERALL", "MONTH"])
        ].set_index(["scope_type", "scope_value"])["coverage_rate"]
        if (
            float(f3.loc[("OVERALL", "ALL")]) >= F3_PARTIAL_OVERALL_FLOOR
            and all(float(f3.loc[("MONTH", month)]) >= F3_PARTIAL_MONTH_FLOOR for month in ("MAY", "JUNE", "JULY"))
        ):
            return "5M_SEAL_PATH_DATA_PARTIAL"
    return "5M_SEAL_PATH_DATA_NOT_READY"


def build_missing_reasons(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, valid_col, reason_col in (
        ("F1", "F1_valid", "F1_missing_reason"),
        ("F2", "F2_valid", "F2_missing_reason"),
        ("F3", "F3_valid", "F3_missing_reason"),
    ):
        missing = values[values[valid_col].ne(1)].copy()
        if missing.empty:
            continue
        grouped = missing.groupby(["month", "board_group", reason_col], dropna=False).size()
        for (month, board, reason), count in grouped.items():
            reason_text = str(reason) if str(reason) in MISSING_REASONS else "OTHER_EXPLICIT"
            rows.append(
                {
                    "variable": variable,
                    "month": month,
                    "board_group": board,
                    "missing_reason": reason_text,
                    "rows": int(count),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["variable", "month", "board_group", "missing_reason"], kind="mergesort"
    ).reset_index(drop=True)


def build_source_mix(lineage: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    grouped = (
        lineage.groupby(["month", "board_day_role", "cache_kind", "cache_sources"], dropna=False)
        .agg(rows=("event_id", "size"), valid_days=("day_complete", "sum"), f1_valid=("f1_valid", "sum"), f2_valid=("f2_valid", "sum"))
        .reset_index()
    )
    valid = lineage[lineage["day_complete"].eq(1)].copy()
    dominant: dict[str, str] = {}
    for month in ("MAY", "JUNE", "JULY"):
        counts = valid[valid["month"].eq(month)]["cache_kind"].value_counts()
        dominant[month] = "" if counts.empty else str(counts.index[0])
    shift = len({value for value in dominant.values() if value}) > 1
    summary = pd.DataFrame(
        [
            {
                "month": "SUMMARY",
                "board_day_role": "ALL",
                "cache_kind": "TEMPORAL_SOURCE_SHIFT",
                "cache_sources": "YES" if shift else "NO",
                "rows": len(valid),
                "valid_days": len(valid),
                "f1_valid": int(valid["f1_valid"].sum()),
                "f2_valid": int(valid["f2_valid"].sum()),
                "dominant_source_by_month": json.dumps(dominant, sort_keys=True),
            }
        ]
    )
    grouped["dominant_source_by_month"] = ""
    return pd.concat([grouped, summary], ignore_index=True), shift


def _parse_pool_clock(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value).replace(".0", ""))
    if len(digits) == 5:
        digits = "0" + digits
    if len(digits) != 6:
        return ""
    hour, minute, second = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    if hour > 23 or minute > 59 or second > 59:
        return ""
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def load_reliable_limitup_rows(root: str | Path, desired_days: set[tuple[str, str]]) -> pd.DataFrame:
    root_path = Path(root).resolve()
    rows: list[pd.DataFrame] = []
    for path in sorted((root_path / "data/cache/limit_ups").glob("*_limitups.pkl")):
        date_text = path.name[:10]
        if date_text < "2026-04-01" or date_text > DEVELOPMENT_END:
            continue
        frame = pd.read_pickle(path)
        needed = {"trade_date", "code", "final_limit_up_time", "open_board_count", "source"}
        if not needed.issubset(frame.columns):
            continue
        part = frame[list(needed)].copy()
        part["trade_date"] = part["trade_date"].astype(str)
        part["code"] = part["code"].map(_normalize_code)
        part = part[
            part["source"].astype(str).eq(RELIABLE_POOL_SOURCE)
            & part.apply(lambda row: (str(row["code"]), str(row["trade_date"])) in desired_days, axis=1)
        ].copy()
        if not part.empty:
            part["pool_cache_path"] = _relative(path, root_path)
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["trade_date", "code", "final_limit_up_time", "open_board_count", "source", "pool_cache_path"])
    result = pd.concat(rows, ignore_index=True)
    return result.drop_duplicates(["trade_date", "code"], keep="last")


def build_pool_validations(
    lineage: pd.DataFrame, pool: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = lineage[lineage["day_complete"].eq(1)].copy()
    merged = base.merge(
        pool,
        left_on=["board_date", "code"],
        right_on=["trade_date", "code"],
        how="inner",
        validate="many_to_one",
    )
    final_rows: list[dict[str, Any]] = []
    reopen_rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        pool_clock = _parse_pool_clock(row.get("final_limit_up_time"))
        pool_minute = trading_minute_from_clock(pool_clock) if pool_clock else math.nan
        f1_minute = float(row["f1_trading_minute"]) if row["f1_valid"] else math.nan
        abs_diff = abs(f1_minute - pool_minute) if np.isfinite(f1_minute) and np.isfinite(pool_minute) else math.nan
        final_rows.append(
            {
                "record_type": "PAIR",
                "event_id": row["event_id"],
                "signal_date": row["signal_date"],
                "month": row["month"],
                "board_day_role": row["board_day_role"],
                "board_date": row["board_date"],
                "code": row["code"],
                "cache_kind": row["cache_kind"],
                "f1_clock_time": row["f1_clock_time"],
                "f1_trading_minute": f1_minute,
                "pool_final_limit_up_time": pool_clock,
                "pool_final_trading_minute": pool_minute,
                "absolute_trading_minute_difference": abs_diff,
                "pool_5m_bucket_end": _bar_bucket_end(pool_clock) if pool_clock else "",
                "exact_same_5m_bucket": int(bool(pool_clock) and row["f1_clock_time"] == _bar_bucket_end(pool_clock)),
                "expected_first_full_bar_end": _expected_first_full_bar_end_after(pool_clock) if pool_clock else "",
                "matches_expected_first_full_bar": int(bool(pool_clock) and row["f1_clock_time"] == _expected_first_full_bar_end_after(pool_clock)),
                "within_5min": int(np.isfinite(abs_diff) and abs_diff <= 5.0),
                "within_10min": int(np.isfinite(abs_diff) and abs_diff <= 10.0),
                "pool_source": row["source"],
                "pool_cache_path": row["pool_cache_path"],
            }
        )
        pool_open = pd.to_numeric(pd.Series([row.get("open_board_count")]), errors="coerce").iloc[0]
        f2 = float(row["confirmed_reopen"]) if row["f2_valid"] else math.nan
        reopen_rows.append(
            {
                "record_type": "PAIR",
                "event_id": row["event_id"],
                "signal_date": row["signal_date"],
                "month": row["month"],
                "board_day_role": row["board_day_role"],
                "board_date": row["board_date"],
                "code": row["code"],
                "cache_kind": row["cache_kind"],
                "confirmed_reopen": f2,
                "pool_open_board_count": float(pool_open) if pd.notna(pool_open) else math.nan,
                "pool_any_open_board": int(pd.notna(pool_open) and float(pool_open) > 0),
                "agreement": int(pd.notna(pool_open) and np.isfinite(f2) and int(f2) == int(float(pool_open) > 0)),
                "pool_source": row["source"],
                "pool_cache_path": row["pool_cache_path"],
            }
        )
    final = pd.DataFrame(final_rows)
    reopen = pd.DataFrame(reopen_rows)
    return final, reopen


def _validation_summaries(final: pd.DataFrame, reopen: pd.DataFrame) -> dict[str, Any]:
    valid_final = final[
        pd.to_numeric(final.get("absolute_trading_minute_difference"), errors="coerce").notna()
    ] if not final.empty else pd.DataFrame()
    if valid_final.empty:
        final_stats = {
            "final_pair_count": 0,
            "final_median_abs_diff": math.nan,
            "final_p90_abs_diff": math.nan,
            "final_exact_bucket_rate": math.nan,
            "final_expected_full_bar_match_rate": math.nan,
            "final_within_5m_rate": math.nan,
            "final_within_10m_rate": math.nan,
        }
    else:
        final_stats = {
            "final_pair_count": len(valid_final),
            "final_median_abs_diff": float(valid_final["absolute_trading_minute_difference"].median()),
            "final_p90_abs_diff": float(valid_final["absolute_trading_minute_difference"].quantile(0.90)),
            "final_exact_bucket_rate": float(valid_final["exact_same_5m_bucket"].mean()),
            "final_expected_full_bar_match_rate": float(valid_final["matches_expected_first_full_bar"].mean()),
            "final_within_5m_rate": float(valid_final["within_5min"].mean()),
            "final_within_10m_rate": float(valid_final["within_10min"].mean()),
        }
    valid_reopen = reopen[
        pd.to_numeric(reopen.get("confirmed_reopen"), errors="coerce").notna()
        & pd.to_numeric(reopen.get("pool_open_board_count"), errors="coerce").notna()
    ] if not reopen.empty else pd.DataFrame()
    if valid_reopen.empty:
        reopen_stats = {
            "reopen_pair_count": 0,
            "f2_precision_vs_pool_open": math.nan,
            "f2_observable_rate_given_pool_open": math.nan,
            "reopen_agreement_rate": math.nan,
        }
    else:
        f2_positive = valid_reopen[valid_reopen["confirmed_reopen"].eq(1)]
        pool_positive = valid_reopen[valid_reopen["pool_any_open_board"].eq(1)]
        reopen_stats = {
            "reopen_pair_count": len(valid_reopen),
            "f2_precision_vs_pool_open": float(f2_positive["pool_any_open_board"].mean()) if len(f2_positive) else math.nan,
            "f2_observable_rate_given_pool_open": float(pool_positive["confirmed_reopen"].mean()) if len(pool_positive) else math.nan,
            "reopen_agreement_rate": float(valid_reopen["agreement"].mean()),
        }
    return {**final_stats, **reopen_stats}


def add_final_validation_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    summaries: list[dict[str, Any]] = []
    for month, part in [("ALL", frame), *[(month, frame[frame["month"].eq(month)]) for month in ("MAY", "JUNE", "JULY")]]:
        valid = part[pd.to_numeric(part["absolute_trading_minute_difference"], errors="coerce").notna()]
        summaries.append(
            {
                "record_type": "SUMMARY",
                "month": month,
                "validation_pair_count": len(valid),
                "median_absolute_trading_minute_difference": float(valid["absolute_trading_minute_difference"].median()) if len(valid) else math.nan,
                "p90_absolute_trading_minute_difference": float(valid["absolute_trading_minute_difference"].quantile(0.90)) if len(valid) else math.nan,
                "exact_same_containing_5m_bucket_rate": float(valid["exact_same_5m_bucket"].mean()) if len(valid) else math.nan,
                "expected_first_full_bar_match_rate": float(valid["matches_expected_first_full_bar"].mean()) if len(valid) else math.nan,
                "within_5min_rate": float(valid["within_5min"].mean()) if len(valid) else math.nan,
                "within_10min_rate": float(valid["within_10min"].mean()) if len(valid) else math.nan,
            }
        )
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def add_reopen_validation_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    summaries: list[dict[str, Any]] = []
    for month, part in [("ALL", frame), *[(month, frame[frame["month"].eq(month)]) for month in ("MAY", "JUNE", "JULY")]]:
        valid = part[
            pd.to_numeric(part["confirmed_reopen"], errors="coerce").notna()
            & pd.to_numeric(part["pool_open_board_count"], errors="coerce").notna()
        ]
        positive = valid[valid["confirmed_reopen"].eq(1)]
        pool_positive = valid[valid["pool_any_open_board"].eq(1)]
        summaries.append(
            {
                "record_type": "SUMMARY",
                "month": month,
                "validation_pair_count": len(valid),
                "f2_precision_pool_open_positive": float(positive["pool_any_open_board"].mean()) if len(positive) else math.nan,
                "f2_observable_rate_given_pool_open": float(pool_positive["confirmed_reopen"].mean()) if len(pool_positive) else math.nan,
                "agreement_rate": float(valid["agreement"].mean()) if len(valid) else math.nan,
                "pool_no_open_f2_zero": int(((valid["pool_any_open_board"] == 0) & (valid["confirmed_reopen"] == 0)).sum()),
                "pool_no_open_f2_one": int(((valid["pool_any_open_board"] == 0) & (valid["confirmed_reopen"] == 1)).sum()),
                "pool_open_f2_zero": int(((valid["pool_any_open_board"] == 1) & (valid["confirmed_reopen"] == 0)).sum()),
                "pool_open_f2_one": int(((valid["pool_any_open_board"] == 1) & (valid["confirmed_reopen"] == 1)).sum()),
            }
        )
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def build_phase_lock(
    population: pd.DataFrame,
    lineage: pd.DataFrame,
    values: pd.DataFrame,
    coverage: pd.DataFrame,
    ready: Mapping[str, bool],
    data_state: str,
    temporal_source_shift: bool,
) -> pd.DataFrame:
    protocol = {
        "task": TASK_NAME,
        "population_window": [DEVELOPMENT_START, DEVELOPMENT_END],
        "expected_grid": EXPECTED_BAR_CLOCKS,
        "price_tolerance": PRICE_TOLERANCE,
        "limit_price": "ROUND_HALF_UP(previous_unadjusted_close*1.10, RMB cent)",
        "source_priority": [item["cache_kind"] for item in SOURCE_SPECS],
        "legacy_interval_metadata": "missing interval marker accepted only with accepted source, adjust=none, and exact fixed 48-bar grid",
        "bar_states": {
            "LOCKED": "low>=L-tolerance and high<=L+tolerance",
            "TOUCHED_MIXED": "high>=L-tolerance and low<L-tolerance",
            "OFF_LIMIT": "high<L-tolerance",
        },
        "F1": "earliest LOCKED bar of final all-LOCKED suffix",
        "F2": "non-LOCKED bar after at least one LOCKED bar",
        "F3": "D0 F1 trading minute - PREV_BOARD_DAY F1 trading minute",
        "coverage_gates": {
            "F1_F2": {"overall": 0.80, "month": 0.70, "board": 0.65},
            "F3": {"overall": 0.70, "month": 0.60, "board": 0.55},
        },
    }
    rows = [
        ("task_name", TASK_NAME),
        ("blind_to_outcome", "YES"),
        ("candidate_rows", len(population)),
        ("signal_dates", population["signal_date"].nunique()),
        ("identity_lock_sha256", _frame_hash(population[list(IDENTITY_COLUMNS)])),
        ("protocol_sha256", _sha256_bytes(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8"))),
        ("lineage_sha256", _frame_hash(lineage)),
        ("reconstruction_values_sha256", _frame_hash(values)),
        ("coverage_sha256", _frame_hash(coverage)),
        ("F1_FINAL_STABLE_LOCK_DATA", "READY" if ready["F1"] else "NOT_READY"),
        ("F2_CONFIRMED_REOPEN_DATA", "READY" if ready["F2"] else "NOT_READY"),
        ("F3_LOCK_DETERIORATION_DATA", "READY" if ready["F3"] else "NOT_READY"),
        ("TEMPORAL_SOURCE_SHIFT", "YES" if temporal_source_shift else "NO"),
        ("LIMIT_PRICE_LINEAGE", "PASS" if int(lineage["limit_price_valid"].sum()) > 0 and not ((lineage["day_complete"].eq(1)) & lineage["limit_price_valid"].ne(1)).any() else "FAIL"),
        ("5M_SEAL_PATH_DATA_STATE", data_state),
        ("OUTCOME_ACCESSED", "NO"),
        ("MODEL_TRAINED", "NO"),
        ("FEATURE_SEARCH", "NO"),
        ("WINDOW_SEARCH", "NO"),
        ("EXTERNAL_DATA_FETCHED", "NO"),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def _pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def render_review(
    population: pd.DataFrame,
    lineage: pd.DataFrame,
    coverage: pd.DataFrame,
    ready: Mapping[str, bool],
    data_state: str,
    temporal_source_shift: bool,
    validation: Mapping[str, Any],
    phase_lock: pd.DataFrame,
) -> str:
    cov = coverage.set_index(["variable", "scope_type", "scope_value"])["coverage_rate"]
    next_action = {
        "5M_SEAL_PATH_DATA_READY": "BLIND_5M_SEAL_PATH_INFORMATION_AUDIT",
        "5M_SEAL_PATH_DATA_PARTIAL": "STOP_AND_REVIEW",
        "5M_SEAL_PATH_DATA_NOT_READY": "STOP_STAGE1_FACTOR_MINING_AND_REVIEW_PIPELINE",
    }[data_state]
    limit_state = str(phase_lock.set_index("key").loc["LIMIT_PRICE_LINEAGE", "value"])
    dominant = {}
    valid_lineage = lineage[lineage["day_complete"].eq(1)]
    for month in ("MAY", "JUNE", "JULY"):
        counts = valid_lineage[valid_lineage["month"].eq(month)]["cache_kind"].value_counts()
        dominant[month] = "无有效来源" if counts.empty else str(counts.index[0])
    lines = [
        "# v004c 5m Conservative Seal-Path Coverage Audit v001",
        "",
        "## 简单结论",
        "",
        f"Q1. May / June / July 的5分钟K够不够？  F1月度覆盖为 May {_pct(float(cov.loc[('F1','MONTH','MAY')]))}、June {_pct(float(cov.loc[('F1','MONTH','JUNE')]))}、July {_pct(float(cov.loc[('F1','MONTH','JULY')]))}；最终状态见下方。",
        "",
        f"Q2. 涨停价能否可靠确定？  {limit_state}。使用未复权前收盘价 × 1.10、人民币分位 ROUND_HALF_UP，并与日线及5分钟最高/收盘价交叉核对。",
        "",
        f"Q3. 能否稳定识别最终稳定锁板时间？  {'可以，达到预注册门槛。' if ready['F1'] else '不能，未达到预注册门槛。'}",
        "",
        f"Q4. 能否保守识别明确破板？  {'可以，达到预注册门槛。' if ready['F2'] else '不能，未达到预注册门槛。'} 该字段只是5分钟可观察下界，不是真实炸板次数。",
        "",
        f"Q5. 能否比较D0和前一板谁更晚稳定锁板？  {'可以，达到预注册门槛。' if ready['F3'] else '不能，双日有效覆盖未达到门槛。'}",
        "",
        f"Q6. 与可靠历史涨停池交叉验证是否合理？  共有 {validation['final_pair_count']} 个F1可比板日；匹配‘真实final seal之后第一根完整5分钟bar’的比例 {_pct(float(validation['final_expected_full_bar_match_rate']))}，10分钟内比例 {_pct(float(validation['final_within_10m_rate']))}。F2=1 对原始 open_board_count>0 的精确率 {_pct(float(validation['f2_precision_vs_pool_open']))}。",
        "",
        "F1通常比逐笔 final seal 晚5–10分钟是定义使然：包含真实封板时刻的那根5分钟K仍可能含有封板前的低价成交；F1要求下一根完整K从头到尾都锁在涨停价。",
        "",
        f"Q7. 是否存在明显月份 source shift？  {'是。' if temporal_source_shift else '否。'} 主来源：May={dominant['MAY']}，June={dominant['JUNE']}，July={dominant['JULY']}。",
        "",
        f"Q8. 是否有资格进入 outcome information audit？  {'有；本轮仍未读取任何 outcome。' if data_state == '5M_SEAL_PATH_DATA_READY' else '没有；本轮停止在数据/lineage层。'}",
        "",
        "## 审计边界",
        "",
        f"- 冻结 population：{len(population)} rows / {population['signal_date'].nunique()} signal dates（{DEVELOPMENT_START} 至 {DEVELOPMENT_END}）。",
        "- 只读取 candidate identity、D0/PREV_BOARD_DAY、本地5分钟缓存、未复权日线和历史涨停池路径字段。",
        "- 未读取 Target7、LOSS、收益、S2 score/rank 或未来标签；未联网、未训练、未搜索窗口/阈值。",
        f"- 固定 tolerance：{PRICE_TOLERANCE:.3f}；完整日固定为48根（09:35..11:30，13:05..15:00）。",
        "- F1/F2/F3 是保守5分钟 proxy，不能称为真实 final_limit_up_time/open_board_count。",
        "",
        "## Coverage",
        "",
        "| field | overall | May | June | July | Board2 | Board3 | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for field in ("F1", "F2", "F3"):
        lines.append(
            f"| {field} | {_pct(float(cov.loc[(field,'OVERALL','ALL')]))} | "
            f"{_pct(float(cov.loc[(field,'MONTH','MAY')]))} | "
            f"{_pct(float(cov.loc[(field,'MONTH','JUNE')]))} | "
            f"{_pct(float(cov.loc[(field,'MONTH','JULY')]))} | "
            f"{_pct(float(cov.loc[(field,'BOARD','BOARD2')]))} | "
            f"{_pct(float(cov.loc[(field,'BOARD','BOARD3')]))} | "
            f"{'YES' if ready[field] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## 最终状态",
            "",
            f"F1_FINAL_STABLE_LOCK_DATA = {'READY' if ready['F1'] else 'NOT_READY'}",
            "",
            f"F2_CONFIRMED_REOPEN_DATA = {'READY' if ready['F2'] else 'NOT_READY'}",
            "",
            f"F3_LOCK_DETERIORATION_DATA = {'READY' if ready['F3'] else 'NOT_READY'}",
            "",
            f"TEMPORAL_SOURCE_SHIFT = {'YES' if temporal_source_shift else 'NO'}",
            "",
            f"LIMIT_PRICE_LINEAGE = {limit_state}",
            "",
            f"5M_SEAL_PATH_DATA_STATE = {data_state}",
            "",
            "OUTCOME_ACCESSED = NO",
            "",
            "MODEL_TRAINED = NO",
            "",
            "FEATURE_SEARCH = NO",
            "",
            "WINDOW_SEARCH = NO",
            "",
            "EXTERNAL_DATA_FETCHED = NO",
            "",
            f"NEXT_ACTION = {next_action}",
            "",
        ]
    )
    if data_state == "5M_SEAL_PATH_DATA_NOT_READY":
        lines.extend(
            [
                "CURRENT_D1_INFORMATION_RESEARCH = PRACTICALLY_EXHAUSTED",
                "",
                "BROADER_D1_INFORMATION_LIMITATION = [待核验]",
                "",
            ]
        )
    return "\n".join(lines)


def run_audit(root: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output_path = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root_path / "reports/research" / OUTPUT_DIRNAME
    )
    population = load_frozen_identity(root_path)
    day_audits = build_day_audits(root_path, population)
    lineage, values = build_candidate_outputs(population, day_audits)
    coverage, ready = build_coverage(values, lineage)
    data_state = determine_data_state(coverage, ready)
    missing = build_missing_reasons(values)
    source_mix, temporal_source_shift = build_source_mix(lineage)
    desired_days = set(zip(day_audits["code"].astype(str), day_audits["board_date"].astype(str)))
    pool = load_reliable_limitup_rows(root_path, desired_days)
    final_validation, reopen_validation = build_pool_validations(lineage, pool)
    validation = _validation_summaries(final_validation, reopen_validation)
    phase_lock = build_phase_lock(
        population,
        lineage,
        values,
        coverage,
        ready,
        data_state,
        temporal_source_shift,
    )
    review = render_review(
        population,
        lineage,
        coverage,
        ready,
        data_state,
        temporal_source_shift,
        validation,
        phase_lock,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    outputs: Sequence[tuple[str, pd.DataFrame | str]] = (
        (OUTPUT_FILENAMES[0], lineage),
        (OUTPUT_FILENAMES[1], day_audits),
        (OUTPUT_FILENAMES[2], values),
        (OUTPUT_FILENAMES[3], coverage),
        (OUTPUT_FILENAMES[4], missing),
        (OUTPUT_FILENAMES[5], source_mix),
        (OUTPUT_FILENAMES[6], add_final_validation_summaries(final_validation)),
        (OUTPUT_FILENAMES[7], add_reopen_validation_summaries(reopen_validation)),
        (OUTPUT_FILENAMES[8], phase_lock),
        (OUTPUT_FILENAMES[9], review),
    )
    for filename, payload in outputs:
        path = output_path / filename
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
        else:
            path.write_text(payload, encoding="utf-8")
    return {
        "output_dir": output_path,
        "population": population,
        "lineage": lineage,
        "bar_quality": day_audits,
        "values": values,
        "coverage": coverage,
        "missing": missing,
        "source_mix": source_mix,
        "final_validation": final_validation,
        "reopen_validation": reopen_validation,
        "phase_lock": phase_lock,
        "review": review,
        "ready": ready,
        "data_state": data_state,
        "temporal_source_shift": temporal_source_shift,
        "validation": validation,
    }


__all__ = [
    "EXPECTED_BAR_CLOCKS",
    "PRICE_TOLERANCE",
    "build_coverage",
    "classify_bar_states",
    "determine_data_state",
    "load_frozen_identity",
    "reconstruct_day",
    "round_main_board_limit_price",
    "run_audit",
    "trading_minute_from_clock",
]

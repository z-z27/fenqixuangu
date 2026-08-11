"""Recover frozen v004c Jan-Apr historical inputs from BaoStock.

This tool is deliberately data-only.  It does not fit a model and it does not
read any July outcome.  The recovery order is:

1. recover unadjusted daily history for the complete cached main-board
   universe (with pre-January feature context);
2. derive canonical daily limit-up pools with the existing loader helpers;
3. derive the two/three-board first-break candidate identity table; and
4. recover BaoStock 5-minute bars only for those candidate codes.

The tool is resumable: existing daily and 5-minute cache rows are merged by
their natural timestamp keys and regenerated limit-up pools are deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import Baostock5mCache  # noqa: E402
from src.code_utils import is_main_board_code, normalize_stock_code  # noqa: E402
from src.config import get_data_config  # noqa: E402
from src.data_sources import (  # noqa: E402
    BAOSTOCK_5M_SOURCE,
    NORMALIZED_5M_INTERVAL,
    MarketDataProvider,
    load_baostock,
    login_baostock,
    logout_baostock,
    to_baostock_code,
    validate_normalized_5m_frame,
)
from src.loaders import (  # noqa: E402
    MarketDataService,
    _count_consecutive_main_board_limit_ups,
    _derive_limit_up_row_from_daily,
    _is_main_board_limit_up_day,
    _merge_daily_frames,
    _prepare_daily_limitup_scan_frame,
)


DAILY_CONTEXT_START = "2025-06-01"
DAILY_CONTEXT_FIRST_TRADE = "2025-06-03"
DAILY_END = "2026-06-30"
POOL_START = "2025-12-20"
POOL_END = "2026-04-30"
CANDIDATE_START = "2026-01-01"
CANDIDATE_END = "2026-04-30"
JULY_SENTINEL = "2026-07-01"

RECOVERY_ROOT = ROOT / "data" / "cache" / "v004c_board3_historical_extension_v001"
DAILY_AUDIT_PATH = RECOVERY_ROOT / "daily_recovery_audit.csv"
POOL_AUDIT_PATH = RECOVERY_ROOT / "pool_recovery_audit.csv"
CANDIDATE_PATH = RECOVERY_ROOT / "jan_apr_authoritative_candidates.csv"
MINUTE_AUDIT_PATH = RECOVERY_ROOT / "minute_recovery_audit.csv"
MANIFEST_PATH = RECOVERY_ROOT / "recovery_manifest.json"


def _normalise_daily(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"])
    for column in (
        "open", "high", "low", "close", "preclose", "volume", "amount", "turn",
    ):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "turnover_rate" not in out.columns:
        out["turnover_rate"] = out["turn"]
    else:
        out["turnover_rate"] = pd.to_numeric(out["turnover_rate"], errors="coerce")
        out["turnover_rate"] = out["turnover_rate"].fillna(out["turn"])
    out["code"] = normalize_stock_code(code)
    return out.sort_values("date", kind="mergesort").drop_duplicates(
        "date", keep="last"
    ).reset_index(drop=True)


def _query_baostock_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    bs = load_baostock()
    result = bs.query_history_k_data_plus(
        to_baostock_code(code),
        "date,open,high,low,close,preclose,volume,amount,turn",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if str(result.error_code) != "0":
        raise RuntimeError(
            f"BaoStock daily query failed {code}: {result.error_code} {result.error_msg}"
        )
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    if not rows:
        return pd.DataFrame()
    return _normalise_daily(pd.DataFrame(rows, columns=result.fields), code)


def _coverage(frame: pd.DataFrame | None) -> tuple[str, str, int]:
    if frame is None or frame.empty or "date" not in frame.columns:
        return "", "", 0
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return "", "", 0
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"), int(len(dates))


def _load_universe(service: MarketDataService) -> pd.DataFrame:
    universe = service.universe_cache.read("eastmoney_main_board")
    if universe is None or universe.empty:
        raise RuntimeError("cached eastmoney_main_board universe is unavailable")
    universe = universe.copy()
    universe["code"] = universe["code"].map(normalize_stock_code)
    universe = universe[universe["code"].map(is_main_board_code)]
    return universe.sort_values("code", kind="mergesort").drop_duplicates(
        "code", keep="last"
    ).reset_index(drop=True)


def recover_daily(service: MarketDataService, universe: pd.DataFrame, force: bool) -> pd.DataFrame:
    """Recover full-universe daily input in one BaoStock login lifecycle."""
    rows: list[dict[str, Any]] = []
    pending: list[str] = []
    total = int(len(universe))
    for scan_index, code in enumerate(universe["code"].astype(str), 1):
        cached = service.daily_unadjusted_cache.read(code)
        start, end, count = _coverage(cached)
        covered = bool(
            start and end and start <= DAILY_CONTEXT_FIRST_TRADE and end >= DAILY_END
        )
        if covered and not force:
            rows.append({
                "code": code, "status": "reused", "date_min": start,
                "date_max": end, "rows": count, "error": "",
            })
        else:
            # Only inspect/merge the second cache when unadjusted history is
            # actually incomplete.  This keeps repeat runs cheap and avoids
            # rewriting thousands of already-complete files.
            legacy = service.daily_cache.read(code)
            if legacy is not None and not legacy.empty:
                merged = _merge_daily_frames(cached, _normalise_daily(legacy, code))
                merged_start, merged_end, merged_count = _coverage(merged)
                if len(merged) != (0 if cached is None else len(cached)):
                    service.daily_unadjusted_cache.write(code, merged)
                cached = merged
                start, end, count = merged_start, merged_end, merged_count
                covered = bool(
                    start and end and start <= DAILY_CONTEXT_FIRST_TRADE and end >= DAILY_END
                )
            if covered and not force:
                rows.append({
                    "code": code, "status": "copied", "date_min": start,
                    "date_max": end, "rows": count, "error": "",
                })
            else:
                pending.append(code)
        if scan_index % 250 == 0 or scan_index == total:
            print(
                f"[daily-scan] {scan_index}/{total} pending={len(pending)}",
                flush=True,
            )

    if pending:
        login_baostock()
        try:
            for index, code in enumerate(pending, 1):
                # BaoStock may expire a long-lived session after many queries.
                # Refresh proactively; an explicit one-time retry below also
                # handles an earlier server-side expiry without losing the run.
                if index > 1 and (index - 1) % 500 == 0:
                    logout_baostock()
                    login_baostock()
                cached = service.daily_unadjusted_cache.read(code)
                try:
                    try:
                        fetched = _query_baostock_daily(code, DAILY_CONTEXT_START, DAILY_END)
                    except RuntimeError as first_error:
                        if "10001001" not in str(first_error) and "用户未登录" not in str(first_error):
                            raise
                        logout_baostock()
                        login_baostock()
                        fetched = _query_baostock_daily(code, DAILY_CONTEXT_START, DAILY_END)
                    if fetched.empty:
                        status = "empty"
                        merged = cached if cached is not None else pd.DataFrame()
                    else:
                        merged = _merge_daily_frames(cached, fetched)
                        service.daily_unadjusted_cache.write(code, merged)
                        status = "fetched"
                    start, end, count = _coverage(merged)
                    rows.append({
                        "code": code, "status": status, "date_min": start,
                        "date_max": end, "rows": count, "error": "",
                    })
                except Exception as exc:  # fail is recorded; audit decides completeness
                    start, end, count = _coverage(cached)
                    rows.append({
                        "code": code, "status": "failed", "date_min": start,
                        "date_max": end, "rows": count,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                if index % 100 == 0 or index == len(pending):
                    print(f"[daily] {index}/{len(pending)}", flush=True)
        finally:
            logout_baostock()

    audit = pd.DataFrame(rows).sort_values("code", kind="mergesort").reset_index(drop=True)
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(DAILY_AUDIT_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    return audit


def _trade_dates(start_date: str, end_date: str) -> list[str]:
    login_baostock()
    try:
        bs = load_baostock()
        result = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if str(result.error_code) != "0":
            raise RuntimeError(
                f"BaoStock trade-calendar query failed: {result.error_code} {result.error_msg}"
            )
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
    finally:
        logout_baostock()
    return [row[0] for row in rows if len(row) >= 2 and row[1] == "1"]


def derive_pools_and_candidates(
    service: MarketDataService,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive pools and first-break candidates with existing loader semantics."""
    pool_dates = set(_trade_dates(POOL_START, POOL_END))
    pool_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []

    stock_by_code = {
        str(row["code"]): row for _, row in universe.iterrows()
    }
    for index, code in enumerate(sorted(stock_by_code), 1):
        stock = stock_by_code[code]
        daily = service.daily_unadjusted_cache.read(code)
        prepared = _prepare_daily_limitup_scan_frame(daily)
        if prepared.empty:
            continue
        for day_index in range(1, len(prepared)):
            date_text = str(prepared.iloc[day_index]["date"])
            if date_text > POOL_END:
                break
            if date_text in pool_dates and _is_main_board_limit_up_day(prepared, day_index):
                row = _derive_limit_up_row_from_daily(prepared, stock, date_text)
                if row is not None:
                    pool_rows[date_text].append(row)

            if not (CANDIDATE_START <= date_text <= CANDIDATE_END):
                continue
            current = prepared.iloc[day_index]
            # BaoStock can emit suspended calendar rows with stale OHLC and
            # null volume.  Such a row is not a tradable first-break signal
            # date and has no D1 five-minute tape; reject it before candidate
            # identity construction instead of treating the source-correct
            # absence of intraday bars as missing data.
            if any(
                pd.isna(current.get(column)) or float(current.get(column)) <= 0
                for column in ("open", "high", "low", "close", "volume")
            ):
                continue
            if _is_main_board_limit_up_day(prepared, day_index):
                continue
            if not _is_main_board_limit_up_day(prepared, day_index - 1):
                continue
            streak = _count_consecutive_main_board_limit_ups(prepared, day_index - 1)
            if streak not in (2, 3):
                continue
            d2_date = str(prepared.iloc[day_index + 1]["date"]) if day_index + 1 < len(prepared) else ""
            d3_date = str(prepared.iloc[day_index + 2]["date"]) if day_index + 2 < len(prepared) else ""
            candidates.append({
                "event_id": f"{code}_{date_text}",
                "code": code,
                "name": "" if pd.isna(stock.get("name", "")) else str(stock.get("name", "")),
                "signal_date": date_text,
                "break_date": date_text,
                "d0_date": str(prepared.iloc[day_index - 1]["date"]),
                "d2_date": d2_date,
                "d3_date": d3_date,
                "label_available_date": d3_date,
                "board_streak_before_break": int(streak),
                "candidate_source": "daily_unadjusted:first_break_after_exact_2_or_3_limitups",
            })
        if index % 250 == 0 or index == len(stock_by_code):
            print(f"[derive] {index}/{len(stock_by_code)}", flush=True)

    pool_audit_rows: list[dict[str, Any]] = []
    for date_text in sorted(pool_dates):
        frame = pd.DataFrame(pool_rows.get(date_text, []))
        if frame.empty:
            pool_audit_rows.append({
                "trade_date": date_text, "rows": 0, "status": "empty", "error": "",
            })
            continue
        frame = frame.sort_values(["trade_date", "code"], kind="mergesort").drop_duplicates(
            ["trade_date", "code"], keep="last"
        ).reset_index(drop=True)
        service.limit_up_cache.write(date_text, frame)
        pool_audit_rows.append({
            "trade_date": date_text, "rows": int(len(frame)), "status": "written", "error": "",
        })

    pool_audit = pd.DataFrame(pool_audit_rows)
    candidate_frame = pd.DataFrame(candidates)
    if not candidate_frame.empty:
        candidate_frame = candidate_frame.sort_values(
            ["signal_date", "event_id"], kind="mergesort"
        ).drop_duplicates("event_id", keep="last").reset_index(drop=True)
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    pool_audit.to_csv(POOL_AUDIT_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    candidate_frame.to_csv(CANDIDATE_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    return pool_audit, candidate_frame


def _minute_window_by_code(candidates: pd.DataFrame) -> dict[str, tuple[str, str]]:
    windows: dict[str, tuple[str, str]] = {}
    for code, rows in candidates.groupby("code", sort=True):
        dates = pd.to_datetime(rows["signal_date"], errors="raise")
        # 40 trading days of frozen minute context fit comfortably in 90
        # calendar days.  Fetching one broad per-code window preserves all
        # timestamps and avoids per-event network requests.
        start = (dates.min() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        end = dates.max().strftime("%Y-%m-%d")
        windows[str(code)] = (start, end)
    return windows


def recover_minutes(candidates: pd.DataFrame, force: bool) -> pd.DataFrame:
    config = get_data_config()
    provider = MarketDataProvider(config)
    cache = Baostock5mCache(config.cache_dir / "baostock_5m", suffix="5min")
    cache.validator = validate_normalized_5m_frame
    rows: list[dict[str, Any]] = []
    windows = _minute_window_by_code(candidates)
    pending: list[tuple[str, str, str]] = []
    for code, (start, end) in windows.items():
        existing = cache.read(code)
        covered = False
        if existing is not None and not existing.empty:
            dates = pd.to_datetime(existing["datetime"], errors="coerce").dropna()
            covered = bool(
                not dates.empty
                and dates.min().strftime("%Y-%m-%d") <= start
                and dates.max().strftime("%Y-%m-%d") >= end
            )
        if covered and not force:
            rows.append({
                "code": code, "status": "reused", "window_start": start,
                "window_end": end, "rows": int(len(existing)), "error": "",
            })
        else:
            pending.append((code, start, end))

    if pending:
        login_baostock()
        try:
            for index, (code, start, end) in enumerate(pending, 1):
                if index > 1 and (index - 1) % 200 == 0:
                    logout_baostock()
                    login_baostock()
                existing = cache.read(code)
                try:
                    try:
                        fetched, _ = provider.fetch_5min_history(
                            code,
                            f"{start} 09:00:00",
                            f"{end} 15:30:00",
                            adjust="none",
                            source=BAOSTOCK_5M_SOURCE,
                        )
                    except RuntimeError as first_error:
                        if "10001001" not in str(first_error) and "用户未登录" not in str(first_error):
                            raise
                        logout_baostock()
                        login_baostock()
                        fetched, _ = provider.fetch_5min_history(
                            code,
                            f"{start} 09:00:00",
                            f"{end} 15:30:00",
                            adjust="none",
                            source=BAOSTOCK_5M_SOURCE,
                        )
                    validate_normalized_5m_frame(fetched)
                    merged = fetched
                    if existing is not None and not existing.empty:
                        merged = pd.concat([existing, fetched], ignore_index=True)
                        merged = merged.drop_duplicates(["code", "datetime"], keep="last")
                        merged = merged.sort_values("datetime", kind="mergesort").reset_index(drop=True)
                    cache.write(
                        code,
                        merged,
                        meta=cache.build_meta(
                            code,
                            merged,
                            source=BAOSTOCK_5M_SOURCE,
                            adjustment="none",
                            interval=NORMALIZED_5M_INTERVAL,
                            fetch_timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                    rows.append({
                        "code": code, "status": "fetched", "window_start": start,
                        "window_end": end, "rows": int(len(merged)), "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "code": code, "status": "failed", "window_start": start,
                        "window_end": end,
                        "rows": 0 if existing is None else int(len(existing)),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                if index % 25 == 0 or index == len(pending):
                    print(f"[minute] {index}/{len(pending)}", flush=True)
        finally:
            logout_baostock()

    audit = pd.DataFrame(rows).sort_values("code", kind="mergesort").reset_index(drop=True)
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(MINUTE_AUDIT_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    return audit


def _write_manifest(
    daily: pd.DataFrame | None,
    pools: pd.DataFrame | None,
    candidates: pd.DataFrame | None,
    minutes: pd.DataFrame | None,
) -> None:
    manifest = {
        "daily_context_start": DAILY_CONTEXT_START,
        "daily_end": DAILY_END,
        "pool_start": POOL_START,
        "pool_end": POOL_END,
        "candidate_start": CANDIDATE_START,
        "candidate_end": CANDIDATE_END,
        "july_result_rows_accessed": 0,
        "daily_audit_rows": None if daily is None else int(len(daily)),
        "pool_audit_rows": None if pools is None else int(len(pools)),
        "candidate_rows": None if candidates is None else int(len(candidates)),
        "candidate_dates": None if candidates is None or candidates.empty else int(candidates["signal_date"].nunique()),
        "board2_rows": None if candidates is None or candidates.empty else int(
            pd.to_numeric(candidates["board_streak_before_break"]).eq(2).sum()
        ),
        "board3_rows": None if candidates is None or candidates.empty else int(
            pd.to_numeric(candidates["board_streak_before_break"]).eq(3).sum()
        ),
        "minute_audit_rows": None if minutes is None else int(len(minutes)),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("daily", "pools", "minutes", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if DAILY_END >= JULY_SENTINEL or CANDIDATE_END >= JULY_SENTINEL:
        raise RuntimeError("July sentinel violated by recovery date constants")
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    service = MarketDataService(get_data_config())
    universe = _load_universe(service)
    print(f"[recovery] main-board universe={len(universe)}", flush=True)

    daily_audit: pd.DataFrame | None = None
    pool_audit: pd.DataFrame | None = None
    candidates: pd.DataFrame | None = None
    minute_audit: pd.DataFrame | None = None

    if args.stage in ("daily", "all"):
        daily_audit = recover_daily(service, universe, force=args.force)
    if args.stage in ("pools", "all"):
        pool_audit, candidates = derive_pools_and_candidates(service, universe)
    elif CANDIDATE_PATH.exists():
        candidates = pd.read_csv(CANDIDATE_PATH, encoding="utf-8-sig", dtype={"code": str})
        candidates["code"] = candidates["code"].astype(str).str.zfill(6)
    if args.stage in ("minutes", "all"):
        if candidates is None or candidates.empty:
            raise RuntimeError("candidate identity table is unavailable; run --stage pools first")
        minute_audit = recover_minutes(candidates, force=args.force)

    _write_manifest(daily_audit, pool_audit, candidates, minute_audit)
    if candidates is not None:
        print(
            "[recovery] candidates "
            f"rows={len(candidates)} dates={candidates['signal_date'].nunique()} "
            f"board2={pd.to_numeric(candidates['board_streak_before_break']).eq(2).sum()} "
            f"board3={pd.to_numeric(candidates['board_streak_before_break']).eq(3).sum()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_data_config
from .indicators import enrich_5min_indicators
from .loaders import DataQualityError, MarketDataService
from .report import write_data_quality_reports, write_signal_reports
from .signal_engine import Signal, generate_signal


DEFAULT_TOP_N = 3
DEFAULT_TARGET_RETURN_PCT = 7.0
HISTORY_HORIZONS = (2, 3, 5, 10)


def simulate_d2_execution(
    signal_row: pd.Series,
    minute_d2: pd.DataFrame,
    reclaim_minutes: int = 30,
    price_mode: str = "confirmation_close",
) -> dict:
    """Minimal D2 execution simulator using only intraday rows up to trigger time."""
    if minute_d2.empty:
        return {"executed": False, "reason": "D2 minute data is empty"}

    invalid_price = signal_row.get("invalid_price")
    low_absorb_min = signal_row.get("low_absorb_min")
    low_absorb_max = signal_row.get("low_absorb_max")
    rows = minute_d2.sort_values("datetime").reset_index(drop=True)
    break_started_at = None
    for _, row in rows.iterrows():
        row_time = pd.Timestamp(row["datetime"])
        low = float(row["low"])
        close = float(row["close"])
        vwap = row.get("intraday_vwap")
        if pd.notna(invalid_price):
            invalid = float(invalid_price)
            if low < invalid and close < invalid and break_started_at is None:
                break_started_at = row_time
            elif close >= invalid:
                break_started_at = None

            if break_started_at is not None and row_time - break_started_at > pd.Timedelta(minutes=reclaim_minutes):
                return {
                    "executed": False,
                    "reason": f"跌破失效位且 {reclaim_minutes} 分钟内未收回",
                    "time": str(row["datetime"]),
                }
        in_zone = (
            pd.notna(low_absorb_min)
            and pd.notna(low_absorb_max)
            and low <= float(low_absorb_max)
            and close >= float(low_absorb_min)
        )
        if in_zone and pd.notna(vwap) and close >= float(vwap):
            zone_buy_price = float(low_absorb_max)
            return {
                "executed": True,
                "reason": "回踩低吸区并站回 VWAP",
                "time": str(row["datetime"]),
                "price": zone_buy_price if price_mode == "zone_max" else close,
                "zone_buy_price": zone_buy_price,
                "confirmation_price": close,
            }
    return {"executed": False, "reason": "D2 未触发低吸验证"}


def run_top3_signal_backtest(
    signals_file: str | Path,
    top_n: int = DEFAULT_TOP_N,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    include_small: bool = False,
    fetch_through_date: str | None = None,
    days: int | None = None,
    force_refresh: bool = False,
    entry_price_mode: str = "zone_max",
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    signals = pd.read_csv(signals_file, dtype={"code": str})
    if signals.empty:
        raise RuntimeError("signals file is empty")

    selected = select_top_signals(signals, top_n=top_n, include_small=include_small)
    service = MarketDataService()
    if fetch_through_date and not selected.empty:
        _prefetch_selected_bars(
            selected,
            service=service,
            end_date=fetch_through_date,
            days=days,
            force_refresh=force_refresh,
        )
    rows = [
        evaluate_top_signal(
            row,
            service=service,
            target_return_pct=target_return_pct,
            entry_price_mode=entry_price_mode,
        )
        for _, row in selected.iterrows()
    ]
    trades = pd.DataFrame(rows)
    summary = build_top3_summary(trades, top_n=top_n, target_return_pct=target_return_pct)
    trade_date = _signals_trade_date(signals)
    output_dir = get_data_config().reports_dir / "backtest_results"
    csv_path, md_path = write_top3_backtest_reports(trades, summary, output_dir, trade_date)
    return trades, summary, csv_path, md_path


def run_full_history_backtest(
    start_date: str,
    end_date: str,
    top_n: int = DEFAULT_TOP_N,
    hold_days: int = 10,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    stop_loss_pct: float = 3.0,
    lookback_days: int = 5,
    signal_days: int | None = None,
    eval_days: int | None = None,
    max_codes: int | None = None,
    force_refresh: bool = False,
    include_all_allowed: bool = False,
    include_small: bool = False,
    entry_price_mode: str = "zone_max",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path, Path, Path]:
    service = MarketDataService()
    reports_dir = get_data_config().reports_dir
    run_root = reports_dir / "backtest_runs" / f"{start_date}_{end_date}"
    signal_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    future_fetch_rows: list[dict[str, Any]] = []

    for requested_date in _iter_weekdays(start_date, end_date):
        run_row = _empty_full_run_row(requested_date)
        try:
            pool = service.collect_limit_ups(
                trade_date=requested_date,
                lookback_days=lookback_days,
                force_refresh=force_refresh,
                write_processed=False,
            )
            actual_date = _latest_trade_date_from_pool(pool)
            run_row["actual_signal_date"] = actual_date
            run_row["limitup_rows"] = int(len(pool))
            if actual_date != requested_date:
                run_row["status"] = "skipped"
                run_row["error"] = "exact signal date limit-up pool missing"
                run_rows.append(run_row)
                continue

            signals, quality_rows = build_signals_for_pool(
                service=service,
                pool=pool,
                as_of_date=actual_date,
                days=signal_days,
                max_codes=max_codes,
                force_refresh=force_refresh,
            )
            signals_csv, signals_md = write_signal_reports(
                signals,
                run_root / "daily_signals",
                trade_date=actual_date,
            )
            quality_csv, quality_md = write_data_quality_reports(
                quality_rows,
                run_root / "data_quality",
                trade_date=actual_date,
            )
            frame = _signals_to_frame(signals)
            if not frame.empty:
                frame["signal_file_date"] = actual_date
                frame["source_signal_file"] = str(signals_csv)
                signal_frames.append(frame)

            future_end_date = _future_end_date(actual_date, hold_days)
            future_fetch = prefetch_future_bars_for_signals(
                frame,
                service=service,
                signal_date=actual_date,
                end_date=future_end_date,
                days=eval_days,
                force_refresh=force_refresh,
            )
            future_fetch_rows.extend(future_fetch["rows"])
            quality_counts = (
                pd.Series([row.get("status") for row in quality_rows]).value_counts()
                if quality_rows
                else pd.Series(dtype=int)
            )
            run_row.update(
                {
                    "status": "generated",
                    "signal_rows": int(len(frame)),
                    "quality_rows": int(len(quality_rows)),
                    "quality_ok": int(quality_counts.get("ok", 0)),
                    "quality_failed": int(quality_counts.get("failed", 0)),
                    "future_fetch_end_date": future_end_date,
                    "future_fetch_attempted": int(future_fetch["attempted"]),
                    "future_fetch_ok": int(future_fetch["ok"]),
                    "future_fetch_failed": int(future_fetch["failed"]),
                    "signals_csv": str(signals_csv),
                    "signals_markdown": str(signals_md),
                    "quality_csv": str(quality_csv),
                    "quality_markdown": str(quality_md),
                }
            )
        except Exception as exc:
            run_row["status"] = "failed"
            run_row["error"] = str(exc)
        run_rows.append(run_row)

    all_signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    if all_signals.empty:
        trades = pd.DataFrame()
    else:
        ranked = prepare_history_rankings(
            all_signals,
            top_n=top_n,
            include_small=include_small,
            include_all_allowed=include_all_allowed,
        )
        minute_cache: dict[str, pd.DataFrame | None] = {}
        rows = [
            evaluate_history_signal(
                row,
                service=service,
                minute_cache=minute_cache,
                hold_days=hold_days,
                target_return_pct=target_return_pct,
                stop_loss_pct=stop_loss_pct,
                entry_price_mode=entry_price_mode,
                top_n=top_n,
                include_all_allowed=include_all_allowed,
            )
            for _, row in ranked.iterrows()
        ]
        trades = pd.DataFrame(rows)

    summary = build_history_summary(
        trades,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        hold_days=hold_days,
        target_return_pct=target_return_pct,
        stop_loss_pct=stop_loss_pct,
        include_all_allowed=include_all_allowed,
        include_small=include_small,
        entry_price_mode=entry_price_mode,
    )
    factor_stats = build_history_factor_stats(trades)
    output_dir = run_root / "backtest_results"
    trade_csv, summary_csv, factor_csv, md_path = write_history_backtest_reports(
        trades,
        summary,
        factor_stats,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
    )
    future_fetch_log = pd.DataFrame(future_fetch_rows)
    future_fetch_csv = write_future_fetch_log(
        future_fetch_log,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
    )
    run_log = pd.DataFrame(run_rows)
    run_log_csv, run_log_md = write_full_backtest_run_log(
        run_log,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
    )
    return (
        trades,
        summary,
        factor_stats,
        run_log,
        trade_csv,
        summary_csv,
        factor_csv,
        md_path,
        run_log_csv,
        run_log_md,
        future_fetch_csv,
    )


def run_history_backtest(
    signals_dir: str | Path,
    start_date: str,
    end_date: str,
    top_n: int = DEFAULT_TOP_N,
    hold_days: int = 10,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    stop_loss_pct: float = 3.0,
    include_all_allowed: bool = False,
    include_small: bool = False,
    entry_price_mode: str = "zone_max",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path]:
    signals = load_history_signal_files(signals_dir, start_date=start_date, end_date=end_date)
    ranked = prepare_history_rankings(
        signals,
        top_n=top_n,
        include_small=include_small,
        include_all_allowed=include_all_allowed,
    )
    service = MarketDataService()
    minute_cache: dict[str, pd.DataFrame | None] = {}
    rows = [
        evaluate_history_signal(
            row,
            service=service,
            minute_cache=minute_cache,
            hold_days=hold_days,
            target_return_pct=target_return_pct,
            stop_loss_pct=stop_loss_pct,
            entry_price_mode=entry_price_mode,
            top_n=top_n,
            include_all_allowed=include_all_allowed,
        )
        for _, row in ranked.iterrows()
    ]
    trades = pd.DataFrame(rows)
    summary = build_history_summary(
        trades,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        hold_days=hold_days,
        target_return_pct=target_return_pct,
        stop_loss_pct=stop_loss_pct,
        include_all_allowed=include_all_allowed,
        include_small=include_small,
        entry_price_mode=entry_price_mode,
    )
    factor_stats = build_history_factor_stats(trades)
    output_dir = get_data_config().reports_dir / "backtest_results"
    trade_csv, summary_csv, factor_csv, md_path = write_history_backtest_reports(
        trades,
        summary,
        factor_stats,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
    )
    return trades, summary, factor_stats, trade_csv, summary_csv, factor_csv, md_path


def build_signals_for_pool(
    service: MarketDataService,
    pool: pd.DataFrame,
    as_of_date: str,
    days: int | None = None,
    max_codes: int | None = None,
    force_refresh: bool = False,
) -> tuple[list[Signal], list[dict[str, Any]]]:
    codes = pool["code"].dropna().astype(str).drop_duplicates().tolist()
    if max_codes:
        codes = codes[: int(max_codes)]
    signals: list[Signal] = []
    quality_rows: list[dict[str, Any]] = []
    for code in codes:
        code_pool = pool[pool["code"].astype(str) == code].sort_values("trade_date")
        if code_pool.empty:
            continue
        d0_date = str(code_pool["trade_date"].iloc[-1])
        if d0_date > as_of_date:
            continue
        name = str(code_pool.iloc[-1].get("name") or "")
        try:
            bars = service.get_stock_bars(code, days=days, end_date=as_of_date, force_refresh=force_refresh)
            quality = dict(bars.quality)
            quality.update({"name": name, "trade_date": as_of_date, "d0_date": d0_date})
            quality_rows.append(quality)
            signals.append(generate_signal(code, name, bars.daily, bars.minute_5m, pool, d0_date=d0_date))
        except Exception as exc:
            if isinstance(exc, DataQualityError):
                quality = dict(exc.quality)
                quality.update({"name": name, "trade_date": as_of_date, "d0_date": d0_date, "error": str(exc)})
                quality_rows.append(quality)
            else:
                quality_rows.append(_failed_quality_row(code, name, as_of_date, d0_date, exc))
    return signals, quality_rows


def prefetch_future_bars_for_signals(
    signals: pd.DataFrame,
    service: MarketDataService,
    signal_date: str,
    end_date: str,
    days: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    if signals is None or signals.empty or "code" not in signals.columns:
        return {"attempted": 0, "ok": 0, "failed": 0, "rows": []}
    attempted = 0
    ok = 0
    failed = 0
    rows: list[dict[str, Any]] = []
    eval_days = int(days or max(get_data_config().default_5min_days, 15))
    for code in signals["code"].astype(str).str.zfill(6).drop_duplicates():
        attempted += 1
        try:
            service.get_stock_bars(code, days=eval_days, end_date=end_date, force_refresh=force_refresh)
            ok += 1
            rows.append(
                {
                    "signal_date": signal_date,
                    "code": code,
                    "future_fetch_end_date": end_date,
                    "eval_days": eval_days,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            failed += 1
            rows.append(
                {
                    "signal_date": signal_date,
                    "code": code,
                    "future_fetch_end_date": end_date,
                    "eval_days": eval_days,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return {"attempted": attempted, "ok": ok, "failed": failed, "rows": rows}


def write_full_backtest_run_log(
    run_log: pd.DataFrame,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_date}_{end_date}"
    csv_path = output_dir / f"history_run_log_{suffix}.csv"
    md_path = output_dir / f"history_run_log_{suffix}.md"
    run_log.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_full_backtest_run_log_markdown(run_log, start_date, end_date), encoding="utf-8")
    return csv_path, md_path


def write_future_fetch_log(
    future_fetch_log: pd.DataFrame,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"history_future_fetch_{start_date}_{end_date}.csv"
    future_fetch_log.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def build_full_backtest_run_log_markdown(run_log: pd.DataFrame, start_date: str, end_date: str) -> str:
    lines = [f"# Full Backtest Run Log {start_date} to {end_date}", ""]
    if run_log.empty:
        lines.append("No run rows.")
        return "\n".join(lines)
    counts = run_log["status"].fillna("").replace("", "unknown").value_counts()
    lines.extend(["## Summary", ""])
    for status, count in counts.items():
        lines.append(f"- {status}: **{int(count)}**")
    lines.append("")
    lines.extend(
        [
            "## Dates",
            "",
            "| requested | actual | status | limitups | signals | quality ok | quality failed | future ok | future failed | error |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in run_log.iterrows():
        lines.append(
            "| {requested} | {actual} | {status} | {limitups} | {signals} | {quality_ok} | {quality_failed} | {future_ok} | {future_failed} | {error} |".format(
                requested=row.get("requested_signal_date", ""),
                actual=row.get("actual_signal_date", ""),
                status=row.get("status", ""),
                limitups=int(row.get("limitup_rows") or 0),
                signals=int(row.get("signal_rows") or 0),
                quality_ok=int(row.get("quality_ok") or 0),
                quality_failed=int(row.get("quality_failed") or 0),
                future_ok=int(row.get("future_fetch_ok") or 0),
                future_failed=int(row.get("future_fetch_failed") or 0),
                error=str(row.get("error") or "")[:160],
            )
        )
    return "\n".join(lines)


def load_history_signal_files(signals_dir: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    root = Path(signals_dir)
    if not root.exists():
        raise RuntimeError(f"signals dir does not exist: {root}")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("signals_*.csv")):
        file_date = _date_from_signal_filename(path)
        if not file_date:
            continue
        file_ts = pd.Timestamp(file_date)
        if file_ts < start_ts or file_ts > end_ts:
            continue
        frame = pd.read_csv(path, dtype={"code": str})
        if frame.empty:
            continue
        if "trade_date" not in frame.columns:
            frame["trade_date"] = file_date
        frame["signal_file_date"] = file_date
        frame["source_signal_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"no signal files found in {root} from {start_date} to {end_date}")
    result = pd.concat(frames, ignore_index=True)
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result = result[result["trade_date"].notna()].copy()
    return result.sort_values(["trade_date", "code"]).reset_index(drop=True)


def prepare_history_rankings(
    signals: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    include_small: bool = False,
    include_all_allowed: bool = False,
) -> pd.DataFrame:
    frame = signals.copy().reset_index(drop=True)
    frame["__row_id"] = range(len(frame))
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["allowed_bool"] = frame.get("allowed", False).astype(str).str.lower().isin({"true", "1", "yes"})
    for column in (
        "total_score",
        "graph_quality_score",
        "theme_score",
        "days_since_d0",
        "consecutive_boards",
        "low_absorb_min",
        "low_absorb_max",
        "invalid_price",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    allowed_types = {"D2_LOW_ABSORB"}
    if include_small or include_all_allowed:
        allowed_types.add("D2_WATCH_OR_SMALL")
    frame["eligible_for_trade"] = frame["allowed_bool"] & frame["signal_type"].astype(str).isin(allowed_types)
    eligible = frame[frame["eligible_for_trade"]].copy()
    if eligible.empty:
        frame["daily_rank"] = pd.NA
        frame["selected_by_topn"] = False
        frame["selected_for_execution"] = False
        return frame.drop(columns=["__row_id"])

    eligible = eligible.sort_values(
        [
            "trade_date",
            "total_score",
            "graph_quality_score",
            "code",
        ],
        ascending=[True, False, False, True],
    )
    eligible["daily_rank"] = eligible.groupby("trade_date").cumcount() + 1
    rank_map = eligible[["__row_id", "daily_rank"]]
    frame = frame.merge(rank_map, on="__row_id", how="left")
    frame["selected_by_topn"] = frame["eligible_for_trade"] & (pd.to_numeric(frame["daily_rank"], errors="coerce") <= int(top_n))
    frame["selected_for_execution"] = frame["eligible_for_trade"] & (
        bool(include_all_allowed) | frame["selected_by_topn"].fillna(False).astype(bool)
    )
    return frame.drop(columns=["__row_id"]).sort_values(["trade_date", "code"]).reset_index(drop=True)


def evaluate_history_signal(
    signal_row: pd.Series,
    service: MarketDataService,
    minute_cache: dict[str, pd.DataFrame | None],
    hold_days: int,
    target_return_pct: float,
    stop_loss_pct: float,
    entry_price_mode: str,
    top_n: int,
    include_all_allowed: bool,
) -> dict[str, Any]:
    code = str(signal_row["code"]).zfill(6)
    signal_date = str(signal_row["trade_date"])
    base_price = _candidate_base_price(signal_row, service)
    result = _base_history_result(
        signal_row,
        code=code,
        signal_date=signal_date,
        base_price=base_price,
        hold_days=hold_days,
        target_return_pct=target_return_pct,
        stop_loss_pct=stop_loss_pct,
        entry_price_mode=entry_price_mode,
        top_n=top_n,
        include_all_allowed=include_all_allowed,
    )
    signal_file_date = str(signal_row.get("signal_file_date", ""))
    if signal_file_date and signal_file_date != signal_date:
        result["data_reason"] = "signal date does not match signal file date"
        result["failure_reason"] = "data_issue"
        return result

    minute = _read_minute_from_cache(code, service, minute_cache)
    if minute is None or minute.empty:
        result["data_reason"] = "missing 5m cache"
        result["failure_reason"] = "data_issue"
        return result
    if "trade_date" not in minute.columns or "datetime" not in minute.columns:
        result["data_reason"] = "5m cache missing date columns"
        result["failure_reason"] = "data_issue"
        return result

    minute = _normalise_minute_frame(minute)
    future_dates = _future_trade_dates(minute, signal_date)
    result["future_trade_days_available"] = len(future_dates)
    if not future_dates:
        result["data_reason"] = "missing future 5m data"
        result["failure_reason"] = "data_issue"
        return result

    candidate_metrics = _path_metrics_by_horizon(
        minute,
        future_dates=future_dates,
        base_price=base_price,
        prefix="candidate",
        hold_days=hold_days,
    )
    result.update(candidate_metrics)
    if base_price is not None:
        result["candidate_evaluable"] = True

    if not bool(signal_row.get("selected_for_execution")):
        result["failure_reason"] = ""
        return result

    d2_date = future_dates[0]
    minute_d2 = minute[minute["trade_date"].astype(str) == d2_date].copy()
    if minute_d2.empty:
        result["data_reason"] = "D2 5m rows empty"
        result["failure_reason"] = "data_issue"
        return result
    minute_d2 = enrich_5min_indicators(minute_d2)
    execution = simulate_d2_execution(signal_row, minute_d2, price_mode=entry_price_mode)
    result["execution_date"] = d2_date
    result["executed"] = bool(execution.get("executed"))
    result["execution_reason"] = execution.get("reason", "")
    if not execution.get("executed"):
        result["evaluable"] = True
        result["data_reason"] = "not triggered"
        result["failure_reason"] = _classify_not_triggered(execution, result, target_return_pct)
        return result

    buy_time = str(execution.get("time", ""))
    buy_price = _to_float(execution.get("price"))
    if buy_price is None or buy_price <= 0:
        result["evaluable"] = True
        result["data_reason"] = "invalid execution price"
        result["failure_reason"] = "data_issue"
        return result

    after_buy = _after_buy_rows(minute, future_dates=future_dates, buy_time=buy_time, hold_days=hold_days)
    if after_buy.empty:
        result["evaluable"] = True
        result["data_reason"] = "no bars after buy"
        result["failure_reason"] = "data_issue"
        return result

    execution_metrics = _path_metrics_by_horizon(
        minute,
        future_dates=future_dates,
        base_price=buy_price,
        prefix="",
        hold_days=hold_days,
        start_time=buy_time,
    )
    outcome = _first_outcome(after_buy, buy_price, target_return_pct=target_return_pct, stop_loss_pct=stop_loss_pct)
    result.update(execution_metrics)
    result.update(
        {
            "evaluable": True,
            "executed": True,
            "buy_time": buy_time,
            "buy_price": buy_price,
            "zone_buy_price": execution.get("zone_buy_price"),
            "confirmation_price": execution.get("confirmation_price"),
            "target_hit": outcome["target_hit"],
            "stop_hit": outcome["stop_hit"],
            "first_outcome": outcome["first_outcome"],
            "first_outcome_time": outcome["first_outcome_time"],
            "data_reason": "",
            "failure_reason": _classify_executed_failure(outcome, result),
        }
    )
    return result


def build_history_summary(
    trades: pd.DataFrame,
    start_date: str,
    end_date: str,
    top_n: int,
    hold_days: int,
    target_return_pct: float,
    stop_loss_pct: float,
    include_all_allowed: bool,
    include_small: bool,
    entry_price_mode: str,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "top_n": int(top_n),
                    "record_count": 0,
                }
            ]
        )
    selected = trades[trades["selected_for_execution"].fillna(False).astype(bool)].copy()
    evaluable = selected[selected["evaluable"].fillna(False).astype(bool)].copy()
    executed = evaluable[evaluable["executed"].fillna(False).astype(bool)].copy()
    target_hit_count = int(executed["target_hit"].fillna(False).astype(bool).sum()) if not executed.empty else 0
    stop_hit_count = int(executed["stop_hit"].fillna(False).astype(bool).sum()) if not executed.empty else 0
    rows = [
        {
            "start_date": start_date,
            "end_date": end_date,
            "top_n": int(top_n),
            "hold_days": int(hold_days),
            "target_return_pct": float(target_return_pct),
            "stop_loss_pct": float(stop_loss_pct),
            "include_all_allowed": bool(include_all_allowed),
            "include_small": bool(include_small),
            "entry_price_mode": entry_price_mode,
            "signal_file_days": int(trades["signal_file_date"].dropna().nunique()) if "signal_file_date" in trades else 0,
            "signal_row_date_count": int(trades["signal_date"].dropna().nunique()) if "signal_date" in trades else 0,
            "signal_date_mismatch_count": int((trades["signal_date_matches_file"].fillna(False).astype(bool) == False).sum())
            if "signal_date_matches_file" in trades
            else 0,
            "record_count": int(len(trades)),
            "allowed_count": int(trades["allowed_bool"].fillna(False).astype(bool).sum()),
            "eligible_count": int(trades["eligible_for_trade"].fillna(False).astype(bool).sum()),
            "selected_count": int(len(selected)),
            "candidate_evaluable_count": int(trades["candidate_evaluable"].fillna(False).astype(bool).sum()),
            "evaluable_count": int(len(evaluable)),
            "executed_count": int(len(executed)),
            "execution_rate": _safe_rate(len(executed), len(evaluable)),
            "target_hit_count": target_hit_count,
            "target_hit_rate": _safe_rate(target_hit_count, len(executed)),
            "stop_hit_count": stop_hit_count,
            "stop_hit_rate": _safe_rate(stop_hit_count, len(executed)),
            "avg_d2_max_return_pct": _mean_or_none(executed, "d2_max_return_pct"),
            "avg_d3_max_return_pct": _mean_or_none(executed, "d3_max_return_pct"),
            "avg_d5_max_return_pct": _mean_or_none(executed, "d5_max_return_pct"),
            "avg_d10_max_return_pct": _mean_or_none(executed, "d10_max_return_pct"),
            "avg_d5_close_return_pct": _mean_or_none(executed, "d5_close_return_pct"),
            "avg_d10_close_return_pct": _mean_or_none(executed, "d10_close_return_pct"),
            "avg_d10_max_drawdown_pct": _mean_or_none(executed, "d10_max_drawdown_pct"),
            "median_d5_max_return_pct": _median_or_none(executed, "d5_max_return_pct"),
            "median_d10_max_return_pct": _median_or_none(executed, "d10_max_return_pct"),
            "candidate_avg_d5_max_return_pct": _mean_or_none(trades, "candidate_d5_max_return_pct"),
            "candidate_avg_d10_max_return_pct": _mean_or_none(trades, "candidate_d10_max_return_pct"),
            "not_triggered_count": int((selected["failure_reason"].astype(str) == "not_triggered").sum()) if not selected.empty else 0,
            "break_invalid_count": int((selected["failure_reason"].astype(str) == "break_invalid").sum()) if not selected.empty else 0,
            "data_issue_count": int((selected["failure_reason"].astype(str) == "data_issue").sum()) if not selected.empty else 0,
        }
    ]
    return pd.DataFrame(rows)


def build_history_factor_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    specs = [
        ("daily_rank", "rank"),
        ("days_since_d0", "days_since_d0"),
        ("signal_type", "category"),
        ("position_level", "category"),
        ("support_type", "category"),
        ("total_score", "score"),
        ("graph_quality_score", "score"),
        ("support_score", "score"),
        ("active_money_score", "score"),
        ("theme_score", "score"),
        ("low_absorb_width_pct", "width"),
        ("invalid_distance_pct", "distance"),
    ]
    rows: list[dict[str, Any]] = []
    for factor, kind in specs:
        if factor not in trades.columns:
            continue
        bucketed = trades.copy()
        bucketed["factor_bucket"] = _bucket_factor(bucketed[factor], kind)
        for bucket, group in bucketed.groupby("factor_bucket", dropna=False):
            selected = group[group["selected_for_execution"].fillna(False).astype(bool)]
            executed = selected[selected["executed"].fillna(False).astype(bool)]
            target_hits = int(executed["target_hit"].fillna(False).astype(bool).sum()) if not executed.empty else 0
            rows.append(
                {
                    "factor": factor,
                    "bucket": "" if pd.isna(bucket) else str(bucket),
                    "record_count": int(len(group)),
                    "selected_count": int(len(selected)),
                    "executed_count": int(len(executed)),
                    "execution_rate": _safe_rate(len(executed), len(selected)),
                    "target_hit_count": target_hits,
                    "target_hit_rate": _safe_rate(target_hits, len(executed)),
                    "avg_candidate_d5_max_return_pct": _mean_or_none(group, "candidate_d5_max_return_pct"),
                    "avg_candidate_d10_max_return_pct": _mean_or_none(group, "candidate_d10_max_return_pct"),
                    "avg_d5_max_return_pct": _mean_or_none(executed, "d5_max_return_pct"),
                    "avg_d5_close_return_pct": _mean_or_none(executed, "d5_close_return_pct"),
                    "avg_d10_max_return_pct": _mean_or_none(executed, "d10_max_return_pct"),
                    "avg_d10_close_return_pct": _mean_or_none(executed, "d10_close_return_pct"),
                    "avg_d10_max_drawdown_pct": _mean_or_none(executed, "d10_max_drawdown_pct"),
                }
            )
    return pd.DataFrame(rows).sort_values(["factor", "bucket"]).reset_index(drop=True)


def write_history_backtest_reports(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    factor_stats: pd.DataFrame,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_date}_{end_date}"
    trade_csv = output_dir / f"history_trades_{suffix}.csv"
    summary_csv = output_dir / f"history_summary_{suffix}.csv"
    factor_csv = output_dir / f"history_factor_stats_{suffix}.csv"
    md_path = output_dir / f"history_review_{suffix}.md"
    trades.to_csv(trade_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    factor_stats.to_csv(factor_csv, index=False, encoding="utf-8-sig")
    md_path.write_text(build_history_review_markdown(trades, summary, factor_stats, start_date, end_date), encoding="utf-8")
    return trade_csv, summary_csv, factor_csv, md_path


def build_history_review_markdown(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    factor_stats: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> str:
    lines = [f"# History Backtest {start_date} to {end_date}", ""]
    if summary.empty:
        lines.append("No summary.")
        return "\n".join(lines)
    item = summary.iloc[0]
    lines.extend(
        [
            "## Summary",
            "",
            f"- signal files: **{int(item.get('signal_file_days', 0))}**",
            f"- signal row dates: **{int(item.get('signal_row_date_count', 0))}**",
            f"- signal date mismatches: **{int(item.get('signal_date_mismatch_count', 0))}**",
            f"- records: **{int(item.get('record_count', 0))}**",
            f"- eligible: **{int(item.get('eligible_count', 0))}**",
            f"- selected: **{int(item.get('selected_count', 0))}**",
            f"- evaluable: **{int(item.get('evaluable_count', 0))}**",
            f"- executed: **{int(item.get('executed_count', 0))}**",
            f"- execution rate: **{_format_pct(item.get('execution_rate'))}**",
            f"- target hit rate: **{_format_pct(item.get('target_hit_rate'))}**",
            f"- stop hit rate: **{_format_pct(item.get('stop_hit_rate'))}**",
            f"- avg D5 max return: **{_format_number(item.get('avg_d5_max_return_pct'))}%**",
            f"- avg D10 max return: **{_format_number(item.get('avg_d10_max_return_pct'))}%**",
            f"- avg candidate D10 max return: **{_format_number(item.get('candidate_avg_d10_max_return_pct'))}%**",
            "",
        ]
    )
    if not trades.empty:
        lines.extend(["## Failure Reasons", ""])
        selected = trades[trades["selected_for_execution"].fillna(False).astype(bool)].copy()
        if selected.empty:
            lines.append("No selected records.")
        else:
            counts = selected["failure_reason"].fillna("").replace("", "none").value_counts()
            for reason, count in counts.items():
                lines.append(f"- {reason}: **{int(count)}**")
        lines.append("")
    if not factor_stats.empty:
        lines.extend(
            [
                "## Factor Snapshot",
                "",
                "| factor | bucket | records | selected | executed | target hit rate | avg D10 max% | avg candidate D10 max% |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        snapshot = factor_stats[factor_stats["factor"].isin(["daily_rank", "days_since_d0", "total_score", "support_type"])].head(40)
        for _, row in snapshot.iterrows():
            lines.append(
                "| {factor} | {bucket} | {records} | {selected} | {executed} | {hit} | {d10} | {candidate} |".format(
                    factor=row.get("factor", ""),
                    bucket=row.get("bucket", ""),
                    records=int(row.get("record_count", 0)),
                    selected=int(row.get("selected_count", 0)),
                    executed=int(row.get("executed_count", 0)),
                    hit=_format_pct(row.get("target_hit_rate")),
                    d10=_format_number(row.get("avg_d10_max_return_pct")),
                    candidate=_format_number(row.get("avg_candidate_d10_max_return_pct")),
                )
            )
    return "\n".join(lines)


def _base_history_result(
    signal_row: pd.Series,
    code: str,
    signal_date: str,
    base_price: float | None,
    hold_days: int,
    target_return_pct: float,
    stop_loss_pct: float,
    entry_price_mode: str,
    top_n: int,
    include_all_allowed: bool,
) -> dict[str, Any]:
    low_absorb_min = _to_float(signal_row.get("low_absorb_min"))
    low_absorb_max = _to_float(signal_row.get("low_absorb_max"))
    invalid_price = _to_float(signal_row.get("invalid_price"))
    return {
        "signal_date": signal_date,
        "signal_file_date": signal_row.get("signal_file_date", ""),
        "signal_date_matches_file": bool(not signal_row.get("signal_file_date") or signal_row.get("signal_file_date") == signal_date),
        "source_signal_file": signal_row.get("source_signal_file", ""),
        "code": code,
        "name": signal_row.get("name", ""),
        "d0_date": signal_row.get("d0_date", ""),
        "days_since_d0": _to_float(signal_row.get("days_since_d0")),
        "consecutive_boards": _to_float(signal_row.get("consecutive_boards")),
        "signal_type": signal_row.get("signal_type", ""),
        "allowed_bool": bool(signal_row.get("allowed_bool", False)),
        "eligible_for_trade": bool(signal_row.get("eligible_for_trade", False)),
        "selected_by_topn": bool(signal_row.get("selected_by_topn", False)),
        "selected_for_execution": bool(signal_row.get("selected_for_execution", False)),
        "daily_rank": _to_float(signal_row.get("daily_rank")),
        "rank_method": "baseline_total",
        "top_n": int(top_n),
        "include_all_allowed": bool(include_all_allowed),
        "position_level": signal_row.get("position_level", ""),
        "total_score": _to_float(signal_row.get("total_score")),
        "graph_quality_score": _to_float(signal_row.get("graph_quality_score")),
        "support_score": _to_float(signal_row.get("support_score")),
        "active_money_score": _to_float(signal_row.get("active_money_score")),
        "theme_score": _to_float(signal_row.get("theme_score")),
        "support_type": signal_row.get("support_type", ""),
        "low_absorb_min": low_absorb_min,
        "low_absorb_max": low_absorb_max,
        "invalid_price": invalid_price,
        "low_absorb_width_pct": _zone_width_pct(low_absorb_min, low_absorb_max),
        "invalid_distance_pct": _invalid_distance_pct(invalid_price, low_absorb_max),
        "candidate_base_price": base_price,
        "candidate_evaluable": False,
        "future_trade_days_available": 0,
        "execution_date": "",
        "executed": False,
        "buy_time": "",
        "buy_price": None,
        "zone_buy_price": None,
        "confirmation_price": None,
        "entry_price_mode": entry_price_mode,
        "execution_reason": "",
        "evaluable": False,
        "target_return_pct": float(target_return_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "target_hit": False,
        "stop_hit": False,
        "first_outcome": "",
        "first_outcome_time": "",
        "data_reason": "",
        "failure_reason": "",
        "reasons": signal_row.get("reasons", ""),
        "key_zones_json": signal_row.get("key_zones_json", ""),
        "hold_days": int(hold_days),
    } | _empty_horizon_metrics()


def _empty_horizon_metrics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix in ("candidate", ""):
        for horizon in HISTORY_HORIZONS:
            label = f"{prefix}_d{horizon}" if prefix else f"d{horizon}"
            result[f"{label}_max_return_pct"] = None
            result[f"{label}_close_return_pct"] = None
            result[f"{label}_max_drawdown_pct"] = None
    return result


def _path_metrics_by_horizon(
    minute: pd.DataFrame,
    future_dates: list[str],
    base_price: float | None,
    prefix: str,
    hold_days: int,
    start_time: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if base_price is None or base_price <= 0:
        return metrics
    start_ts = pd.Timestamp(start_time) if start_time else None
    for horizon in HISTORY_HORIZONS:
        window_days = max(1, min(int(hold_days), int(horizon) - 1))
        rows = minute[minute["trade_date"].astype(str).isin(future_dates[:window_days])].copy()
        if start_ts is not None:
            rows = rows[pd.to_datetime(rows["datetime"], errors="coerce") >= start_ts].copy()
        if rows.empty:
            continue
        high = pd.to_numeric(rows["high"], errors="coerce").max()
        low = pd.to_numeric(rows["low"], errors="coerce").min()
        close_values = pd.to_numeric(rows.sort_values("datetime")["close"], errors="coerce").dropna()
        if pd.isna(high) or pd.isna(low) or close_values.empty:
            continue
        close = float(close_values.iloc[-1])
        label = f"{prefix}_d{horizon}" if prefix else f"d{horizon}"
        metrics[f"{label}_max_return_pct"] = (float(high) / base_price - 1.0) * 100.0
        metrics[f"{label}_close_return_pct"] = (close / base_price - 1.0) * 100.0
        metrics[f"{label}_max_drawdown_pct"] = (float(low) / base_price - 1.0) * 100.0
    return metrics


def _first_outcome(
    after_buy: pd.DataFrame,
    buy_price: float,
    target_return_pct: float,
    stop_loss_pct: float,
) -> dict[str, Any]:
    target_price = buy_price * (1.0 + float(target_return_pct) / 100.0)
    stop_price = buy_price * (1.0 - float(stop_loss_pct) / 100.0)
    target_hit = False
    stop_hit = False
    first_outcome = "none"
    first_time = ""
    for _, row in after_buy.sort_values("datetime").iterrows():
        high = _to_float(row.get("high"))
        low = _to_float(row.get("low"))
        current_time = str(row.get("datetime", ""))
        high_hit = high is not None and high >= target_price
        low_hit = low is not None and low <= stop_price
        target_hit = target_hit or high_hit
        stop_hit = stop_hit or low_hit
        if first_outcome == "none" and (high_hit or low_hit):
            if high_hit and low_hit:
                first_outcome = "target_and_stop_same_bar"
            elif high_hit:
                first_outcome = "target_first"
            else:
                first_outcome = "stop_first"
            first_time = current_time
            break
    if first_outcome != "none":
        remaining = after_buy[pd.to_datetime(after_buy["datetime"], errors="coerce") > pd.Timestamp(first_time)].copy()
        if not remaining.empty:
            target_hit = target_hit or bool((pd.to_numeric(remaining["high"], errors="coerce") >= target_price).any())
            stop_hit = stop_hit or bool((pd.to_numeric(remaining["low"], errors="coerce") <= stop_price).any())
    return {
        "target_hit": bool(target_hit),
        "stop_hit": bool(stop_hit),
        "first_outcome": first_outcome,
        "first_outcome_time": first_time,
    }


def _classify_not_triggered(execution: dict[str, Any], result: dict[str, Any], target_return_pct: float) -> str:
    reason = str(execution.get("reason", ""))
    if "跌破失效位" in reason:
        return "break_invalid"
    candidate_d10 = _to_float(result.get("candidate_d10_max_return_pct"))
    candidate_d5 = _to_float(result.get("candidate_d5_max_return_pct"))
    candidate_best = max([value for value in (candidate_d5, candidate_d10) if value is not None], default=None)
    if candidate_best is not None and candidate_best >= float(target_return_pct):
        return "zone_too_low"
    return "not_triggered"


def _classify_executed_failure(outcome: dict[str, Any], result: dict[str, Any]) -> str:
    if outcome.get("first_outcome") == "target_first":
        return ""
    if outcome.get("first_outcome") == "target_and_stop_same_bar":
        return ""
    if outcome.get("first_outcome") == "stop_first":
        return "stop_hit"
    if outcome.get("target_hit"):
        return ""
    d5_close = _to_float(result.get("d5_close_return_pct"))
    d10_close = _to_float(result.get("d10_close_return_pct"))
    if (d5_close is not None and d5_close < 0) or (d10_close is not None and d10_close < 0):
        return "weak_repair"
    return "unknown"


def _after_buy_rows(
    minute: pd.DataFrame,
    future_dates: list[str],
    buy_time: str,
    hold_days: int,
) -> pd.DataFrame:
    rows = minute[minute["trade_date"].astype(str).isin(future_dates[: max(1, int(hold_days))])].copy()
    if rows.empty:
        return rows
    return rows[pd.to_datetime(rows["datetime"], errors="coerce") >= pd.Timestamp(buy_time)].sort_values("datetime")


def _candidate_base_price(signal_row: pd.Series, service: MarketDataService) -> float | None:
    zones = _parse_key_zones(signal_row.get("key_zones_json"))
    for key in ("d1_close", "close"):
        value = _to_float(zones.get(key))
        if value is not None and value > 0:
            return value
    code = str(signal_row.get("code", "")).zfill(6)
    signal_date = str(signal_row.get("trade_date", ""))
    daily = service.daily_cache.read(code)
    if daily is None or daily.empty or "date" not in daily.columns or "close" not in daily.columns:
        return None
    dates = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    matched = daily[dates == signal_date]
    if matched.empty:
        return None
    return _to_float(matched.iloc[-1].get("close"))


def _parse_key_zones(raw: Any) -> dict[str, Any]:
    if raw is None or pd.isna(raw):
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_minute_from_cache(
    code: str,
    service: MarketDataService,
    minute_cache: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    if code not in minute_cache:
        minute_cache[code] = service.minute_cache.read(code)
    return minute_cache[code]


def _normalise_minute_frame(minute: pd.DataFrame) -> pd.DataFrame:
    result = minute.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    return result[result["datetime"].notna() & result["trade_date"].notna()].sort_values("datetime").reset_index(drop=True)


def _future_trade_dates(minute: pd.DataFrame, signal_date: str) -> list[str]:
    dates = sorted(minute["trade_date"].dropna().astype(str).unique().tolist())
    return [date for date in dates if date > signal_date]


def _date_from_signal_filename(path: Path) -> str:
    stem = path.stem
    prefix = "signals_"
    if not stem.startswith(prefix):
        return ""
    text = stem[len(prefix) :]
    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _bucket_factor(values: pd.Series, kind: str) -> pd.Series:
    if kind == "category":
        return values.fillna("")
    if kind == "rank":
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.map(lambda value: "" if pd.isna(value) else str(int(value)) if value <= 3 else ">3")
    if kind == "days_since_d0":
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.map(lambda value: "" if pd.isna(value) else str(int(value)) if value <= 3 else ">3")
    numeric = pd.to_numeric(values, errors="coerce")
    if kind == "score":
        return pd.cut(numeric, bins=[-0.001, 50, 60, 70, 80, 1000], labels=["<=50", "50-60", "60-70", "70-80", ">80"])
    if kind == "width":
        return pd.cut(numeric, bins=[-1000, 1, 2, 3, 5, 1000], labels=["<=1", "1-2", "2-3", "3-5", ">5"])
    if kind == "distance":
        return pd.cut(numeric, bins=[-1000, 1, 2, 3, 5, 1000], labels=["<=1", "1-2", "2-3", "3-5", ">5"])
    return values.fillna("")


def _zone_width_pct(low_absorb_min: float | None, low_absorb_max: float | None) -> float | None:
    if low_absorb_min is None or low_absorb_max is None or low_absorb_min <= 0:
        return None
    return (low_absorb_max / low_absorb_min - 1.0) * 100.0


def _invalid_distance_pct(invalid_price: float | None, low_absorb_max: float | None) -> float | None:
    if invalid_price is None or low_absorb_max is None or low_absorb_max <= 0:
        return None
    return (low_absorb_max / invalid_price - 1.0) * 100.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_full_run_row(requested_date: str) -> dict[str, Any]:
    return {
        "requested_signal_date": requested_date,
        "actual_signal_date": "",
        "status": "started",
        "limitup_rows": 0,
        "signal_rows": 0,
        "quality_rows": 0,
        "quality_ok": 0,
        "quality_failed": 0,
        "future_fetch_end_date": "",
        "future_fetch_attempted": 0,
        "future_fetch_ok": 0,
        "future_fetch_failed": 0,
        "signals_csv": "",
        "signals_markdown": "",
        "quality_csv": "",
        "quality_markdown": "",
        "error": "",
    }


def _iter_weekdays(start_date: str, end_date: str) -> list[str]:
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    return [date.strftime("%Y-%m-%d") for date in dates if date.weekday() < 5]


def _latest_trade_date_from_pool(pool: pd.DataFrame) -> str:
    if pool is None or pool.empty or "trade_date" not in pool.columns:
        return ""
    return str(pool["trade_date"].dropna().astype(str).max())


def _signals_to_frame(signals: list[Signal]) -> pd.DataFrame:
    return pd.DataFrame([signal.to_dict() for signal in signals])


def _future_end_date(signal_date: str, hold_days: int) -> str:
    buffer_days = max(14, int(hold_days * 2.5) + 7)
    raw_end = pd.Timestamp(signal_date) + pd.Timedelta(days=buffer_days)
    latest_possible = pd.Timestamp(_latest_possible_market_date())
    if latest_possible > pd.Timestamp(signal_date):
        raw_end = min(raw_end, latest_possible)
    return raw_end.strftime("%Y-%m-%d")


def _latest_possible_market_date() -> str:
    current = pd.Timestamp.now().normalize()
    while current.weekday() >= 5:
        current -= pd.Timedelta(days=1)
    return current.strftime("%Y-%m-%d")


def _failed_quality_row(code: str, name: str, trade_date: str, d0_date: str, exc: Exception) -> dict[str, Any]:
    return {
        "code": str(code).zfill(6),
        "name": name,
        "trade_date": trade_date,
        "d0_date": d0_date,
        "status": "failed",
        "daily_source": "",
        "minute_source": "",
        "from_cache": False,
        "daily_rows": 0,
        "daily_history_rows": 0,
        "daily_required_days": 0,
        "minute_rows": 0,
        "minute_trade_days": 0,
        "minute_required_days": 0,
        "daily_start": "",
        "daily_end": "",
        "minute_start": "",
        "minute_end": "",
        "daily_ma_coverage_ok": False,
        "missing_latest_daily_ma": "",
        "missing_daily_amount_count": 0,
        "missing_daily_volume_count": 0,
        "missing_minute_amount_count": 0,
        "missing_minute_volume_count": 0,
        "zero_minute_volume_count": 0,
        "daily_minute_close_matched_days": 0,
        "daily_minute_close_max_abs_diff": None,
        "daily_minute_close_check_ok": False,
        "warnings": "",
        "error": str(exc),
    }


def _prefetch_selected_bars(
    selected: pd.DataFrame,
    service: MarketDataService,
    end_date: str,
    days: int | None,
    force_refresh: bool,
) -> None:
    for code in selected["code"].astype(str).str.zfill(6).drop_duplicates():
        try:
            service.get_stock_bars(code, days=days, end_date=end_date, force_refresh=force_refresh)
        except Exception:
            # The backtest row will still record missing or unusable D2 data.
            continue


def select_top_signals(signals: pd.DataFrame, top_n: int = DEFAULT_TOP_N, include_small: bool = False) -> pd.DataFrame:
    frame = signals.copy()
    frame["allowed"] = frame["allowed"].astype(str).str.lower().isin({"true", "1", "yes"})
    eligible = frame[frame["allowed"]].copy()
    if not include_small:
        eligible = eligible[eligible["signal_type"].astype(str) == "D2_LOW_ABSORB"].copy()
    if eligible.empty:
        return eligible

    eligible["position_priority"] = eligible["position_level"].map({"normal": 0, "small": 1}).fillna(9)
    for column in ("total_score", "graph_quality_score", "support_score", "active_money_score"):
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    eligible = eligible.sort_values(
        [
            "trade_date",
            "position_priority",
            "total_score",
            "graph_quality_score",
            "support_score",
            "active_money_score",
        ],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)
    eligible["daily_rank"] = eligible.groupby("trade_date").cumcount() + 1
    return eligible[eligible["daily_rank"] <= int(top_n)].reset_index(drop=True)


def evaluate_top_signal(
    signal_row: pd.Series,
    service: MarketDataService,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    entry_price_mode: str = "zone_max",
) -> dict[str, Any]:
    code = str(signal_row["code"]).zfill(6)
    signal_date = str(signal_row["trade_date"])
    result = {
        "signal_date": signal_date,
        "code": code,
        "name": signal_row.get("name", ""),
        "daily_rank": int(signal_row.get("daily_rank", 0)),
        "signal_type": signal_row.get("signal_type", ""),
        "position_level": signal_row.get("position_level", ""),
        "total_score": signal_row.get("total_score"),
        "graph_quality_score": signal_row.get("graph_quality_score"),
        "support_score": signal_row.get("support_score"),
        "active_money_score": signal_row.get("active_money_score"),
        "low_absorb_min": signal_row.get("low_absorb_min"),
        "low_absorb_max": signal_row.get("low_absorb_max"),
        "invalid_price": signal_row.get("invalid_price"),
        "d2_date": "",
        "executed": False,
        "buy_time": "",
        "buy_price": None,
        "confirmation_price": None,
        "entry_price_mode": entry_price_mode,
        "execution_reason": "",
        "d2_max_return_pct": None,
        "d2_close_return_pct": None,
        "target_return_pct": float(target_return_pct),
        "target_hit": False,
        "evaluable": False,
        "data_reason": "",
    }
    minute = service.minute_cache.read(code)
    if minute is None or minute.empty:
        result["data_reason"] = "missing 5m cache"
        return result
    if "trade_date" not in minute.columns or "datetime" not in minute.columns:
        result["data_reason"] = "5m cache missing date columns"
        return result

    d2_date = _next_trade_date(minute, signal_date)
    if not d2_date:
        result["data_reason"] = "missing D2 5m data"
        return result
    minute_d2 = minute[minute["trade_date"].astype(str) == d2_date].copy()
    if minute_d2.empty:
        result["data_reason"] = "D2 5m rows empty"
        return result

    minute_d2 = enrich_5min_indicators(minute_d2)
    execution = simulate_d2_execution(signal_row, minute_d2, price_mode=entry_price_mode)
    result["d2_date"] = d2_date
    result["executed"] = bool(execution.get("executed"))
    result["execution_reason"] = execution.get("reason", "")
    if not execution.get("executed"):
        result["data_reason"] = "not triggered"
        result["evaluable"] = True
        return result

    buy_time = str(execution.get("time", ""))
    buy_price = float(execution.get("price"))
    after_buy = minute_d2[pd.to_datetime(minute_d2["datetime"], errors="coerce") >= pd.Timestamp(buy_time)].copy()
    if after_buy.empty:
        result["data_reason"] = "no bars after buy"
        result["evaluable"] = True
        return result

    max_high = float(pd.to_numeric(after_buy["high"], errors="coerce").max())
    close_price = float(pd.to_numeric(after_buy["close"], errors="coerce").iloc[-1])
    max_return_pct = (max_high / buy_price - 1.0) * 100.0
    close_return_pct = (close_price / buy_price - 1.0) * 100.0
    result.update(
        {
            "buy_time": buy_time,
            "buy_price": buy_price,
            "confirmation_price": execution.get("confirmation_price"),
            "d2_max_return_pct": max_return_pct,
            "d2_close_return_pct": close_return_pct,
            "target_hit": bool(max_return_pct >= float(target_return_pct)),
            "evaluable": True,
            "data_reason": "",
        }
    )
    return result


def build_top3_summary(
    trades: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "top_n": int(top_n),
                    "target_return_pct": float(target_return_pct),
                    "selected_count": 0,
                }
            ]
        )
    evaluable = trades[trades["evaluable"].fillna(False).astype(bool)].copy()
    executed = evaluable[evaluable["executed"].fillna(False).astype(bool)].copy()
    target_hit_count = int(executed["target_hit"].fillna(False).astype(bool).sum()) if not executed.empty else 0
    return pd.DataFrame(
        [
            {
                "top_n": int(top_n),
                "target_return_pct": float(target_return_pct),
                "selected_count": int(len(trades)),
                "evaluable_count": int(len(evaluable)),
                "executed_count": int(len(executed)),
                "execution_rate": _safe_rate(len(executed), len(evaluable)),
                "target_hit_count": target_hit_count,
                "target_hit_rate": _safe_rate(target_hit_count, len(executed)),
                "avg_d2_max_return_pct": _mean_or_none(executed, "d2_max_return_pct"),
                "avg_d2_close_return_pct": _mean_or_none(executed, "d2_close_return_pct"),
                "median_d2_max_return_pct": _median_or_none(executed, "d2_max_return_pct"),
                "median_d2_close_return_pct": _median_or_none(executed, "d2_close_return_pct"),
            }
        ]
    )


def write_top3_backtest_reports(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    trade_date: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"top3_backtest_{trade_date}.csv"
    md_path = output_dir / f"top3_backtest_{trade_date}.md"
    trades.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_top3_backtest_markdown(trades, summary, trade_date), encoding="utf-8")
    return csv_path, md_path


def build_top3_backtest_markdown(trades: pd.DataFrame, summary: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} Top3 Backtest", ""]
    if summary.empty:
        lines.append("No summary.")
        return "\n".join(lines)
    item = summary.iloc[0]
    lines.extend(
        [
            "## Summary",
            "",
            f"- top_n: **{int(item.get('top_n', 0))}**",
            f"- target return: **{float(item.get('target_return_pct', 0)):.2f}%**",
            f"- selected: **{int(item.get('selected_count', 0))}**",
            f"- evaluable: **{int(item.get('evaluable_count', 0))}**",
            f"- executed: **{int(item.get('executed_count', 0))}**",
            f"- execution rate: **{_format_pct(item.get('execution_rate'))}**",
            f"- target hit rate: **{_format_pct(item.get('target_hit_rate'))}**",
            f"- avg D2 max return: **{_format_number(item.get('avg_d2_max_return_pct'))}%**",
            f"- avg D2 close return: **{_format_number(item.get('avg_d2_close_return_pct'))}%**",
            "",
        ]
    )
    if trades.empty:
        lines.append("No selected trades.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Selected",
            "",
            "| date | rank | code | name | score | executed | buy | confirm | max% | close% | hit | reason |",
            "|---|---:|---|---|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in trades.iterrows():
        lines.append(
            "| {date} | {rank} | {code} | {name} | {score} | {executed} | {buy} | {confirm} | {max_ret} | {close_ret} | {hit} | {reason} |".format(
                date=row.get("signal_date", ""),
                rank=row.get("daily_rank", ""),
                code=row.get("code", ""),
                name=row.get("name", ""),
                score=_format_number(row.get("total_score")),
                executed=str(bool(row.get("executed"))),
                buy=_format_number(row.get("buy_price")),
                confirm=_format_number(row.get("confirmation_price")),
                max_ret=_format_number(row.get("d2_max_return_pct")),
                close_ret=_format_number(row.get("d2_close_return_pct")),
                hit=str(bool(row.get("target_hit"))),
                reason=str(row.get("execution_reason") or row.get("data_reason") or "")[:120],
            )
        )
    return "\n".join(lines)


def _next_trade_date(minute: pd.DataFrame, signal_date: str) -> str:
    dates = sorted(pd.to_datetime(minute["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    for date in dates:
        if date > signal_date:
            return date
    return ""


def _signals_trade_date(signals: pd.DataFrame) -> str:
    if "trade_date" in signals.columns and signals["trade_date"].notna().any():
        return str(signals["trade_date"].dropna().max())
    return pd.Timestamp.now().strftime("%Y-%m-%d")


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _median_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"

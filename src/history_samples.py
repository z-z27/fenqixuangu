from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .backtester import (
    _candidate_base_price,
    _future_end_date,
    _future_trade_dates,
    _invalid_distance_pct,
    _iter_weekdays,
    _latest_trade_date_from_pool,
    _normalise_minute_frame,
    _path_metrics_by_horizon,
    _read_minute_from_cache,
    _signals_to_frame,
    _to_float,
    _zone_width_pct,
    build_signals_for_pool,
    prefetch_future_bars_for_signals,
)
from .config import get_data_config
from .loaders import MarketDataService
from .report import write_data_quality_reports, write_signal_reports


DEFAULT_TARGET_RETURN_PCT = 7.0
DEFAULT_SECONDARY_TARGET_RETURN_PCT = 10.0

FORBIDDEN_HISTORY_SAMPLE_COLUMNS = {
    "executed",
    "selected_for_execution",
    "selected_by_topn",
    "buy_price",
    "zone_buy_price",
    "confirmation_price",
    "execution_date",
    "buy_time",
    "entry_price_mode",
    "execution_reason",
    "target_hit",
    "stop_hit",
    "first_outcome",
    "first_outcome_time",
    "failure_reason",
    "d3_realized_return_pct",
    "d3_sell_reason",
}

HISTORY_CANDIDATE_COLUMNS = [
    "signal_date",
    "code",
    "name",
    "d0_date",
    "days_since_d0",
    "consecutive_boards",
    "signal_type",
    "allowed_bool",
    "eligible_for_trade",
    "total_score",
    "graph_quality_score",
    "active_money_score",
    "active_cooling_score",
    "support_score",
    "theme_score",
    "trend_hold_score",
    "entry_width_score",
    "d1_low_ma10_pct",
    "d1_close_ma10_pct",
    "d1_close_vwap_pct",
    "low_absorb_width_pct",
    "invalid_distance_pct",
    "support_type",
    "low_absorb_min",
    "low_absorb_max",
    "invalid_price",
    "candidate_base_price",
    "candidate_evaluable",
    "future_trade_days_available",
    "d2_trade_date",
    "d3_trade_date",
    "d2_open_price",
    "d3_high_price",
    "d3_close_price",
    "d2open_d3high_return_pct",
    "d2open_d3close_return_pct",
    "candidate_d2_max_return_pct",
    "candidate_d2_close_return_pct",
    "candidate_d2_max_drawdown_pct",
    "candidate_d3_max_return_pct",
    "candidate_d3_close_return_pct",
    "candidate_d3_max_drawdown_pct",
    "candidate_d5_max_return_pct",
    "candidate_d5_close_return_pct",
    "candidate_d5_max_drawdown_pct",
    "candidate_d10_max_return_pct",
    "candidate_d10_close_return_pct",
    "candidate_d10_max_drawdown_pct",
    "target7",
    "target10",
    "target7_d2open_d3high",
    "target7_d2open_d3close",
    "reasons",
    "key_zones_json",
]


def run_history_sample_generation(
    start_date: str,
    end_date: str,
    lookback_days: int = 5,
    signal_days: int | None = None,
    eval_days: int | None = None,
    max_codes: int | None = None,
    force_refresh: bool = False,
    workers: int = 6,
    hold_days: int = 10,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    secondary_target_return_pct: float = DEFAULT_SECONDARY_TARGET_RETURN_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    """Generate clean historical candidate samples without execution backtest fields.

    Each output row is one D1-night candidate. The row contains D1-known factors
    and future candidate labels such as candidate_d3_max_return_pct and
    target7_d2open_d3high. This layer does not emit execution-only fields.
    """
    service = MarketDataService()
    run_root = get_data_config().reports_dir / "history_samples" / f"{start_date}_{end_date}"
    candidate_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    future_fetch_rows: list[dict[str, Any]] = []
    minute_cache: dict[str, pd.DataFrame | None] = {}
    missing_limitup_dates: set[str] = set()
    requested_dates = _iter_weekdays(start_date, end_date)
    total_dates = len(requested_dates)

    for date_index, requested_date in enumerate(requested_dates, 1):
        run_row = _empty_generation_row(requested_date)
        print(f"[history-samples] {date_index}/{total_dates} start {requested_date}", flush=True)
        try:
            pool = _collect_limitups_for_history_sample(
                service=service,
                requested_date=requested_date,
                start_date=start_date,
                lookback_days=lookback_days,
                force_refresh=force_refresh,
                workers=workers,
                missing_dates=missing_limitup_dates,
            )
            actual_date = _latest_trade_date_from_pool(pool)
            run_row["actual_signal_date"] = actual_date
            run_row["limitup_rows"] = int(len(pool))
            print(
                f"[history-samples] {requested_date} limitups={len(pool)} actual={actual_date}",
                flush=True,
            )
            if actual_date != requested_date:
                run_row["status"] = "skipped"
                run_row["error"] = "exact signal date limit-up pool missing"
                run_rows.append(run_row)
                print(
                    f"[history-samples] {date_index}/{total_dates} skipped {requested_date}: actual={actual_date}",
                    flush=True,
                )
                continue

            signals, quality_rows = build_signals_for_pool(
                service=service,
                pool=pool,
                as_of_date=actual_date,
                days=signal_days,
                max_codes=max_codes,
                force_refresh=force_refresh,
            )
            quality_counts = (
                pd.Series([row.get("status") for row in quality_rows]).value_counts()
                if quality_rows
                else pd.Series(dtype=int)
            )
            quality_ok = int(quality_counts.get("ok", 0))
            quality_failed = int(quality_counts.get("failed", 0))
            print(
                f"[history-samples] {requested_date} signals={len(signals)} "
                f"quality_ok={quality_ok} quality_failed={quality_failed}",
                flush=True,
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
            signal_frame = _signals_to_frame(signals)
            if not signal_frame.empty:
                signal_frame["signal_file_date"] = actual_date
                signal_frame["source_signal_file"] = str(signals_csv)

            future_end_date = _future_end_date(actual_date, hold_days)
            future_fetch = prefetch_future_bars_for_signals(
                signal_frame,
                service=service,
                signal_date=actual_date,
                end_date=future_end_date,
                days=eval_days,
                force_refresh=force_refresh,
            )
            future_fetch_rows.extend(future_fetch["rows"])
            print(
                f"[history-samples] {requested_date} future_fetch "
                f"attempted={future_fetch['attempted']} ok={future_fetch['ok']} failed={future_fetch['failed']} "
                f"end={future_end_date}",
                flush=True,
            )
            for code in signal_frame.get("code", pd.Series(dtype=str)).astype(str).str.zfill(6).drop_duplicates():
                minute_cache.pop(code, None)

            for _, signal_row in signal_frame.iterrows():
                candidate_rows.append(
                    evaluate_history_candidate_only(
                        signal_row,
                        service=service,
                        minute_cache=minute_cache,
                        hold_days=hold_days,
                        target_return_pct=target_return_pct,
                        secondary_target_return_pct=secondary_target_return_pct,
                    )
                )
            print(
                f"[history-samples] {requested_date} evaluated_candidates={len(signal_frame)} "
                f"total_candidates={len(candidate_rows)}",
                flush=True,
            )

            run_row.update(
                {
                    "status": "generated",
                    "signal_rows": int(len(signal_frame)),
                    "candidate_rows": int(len(signal_frame)),
                    "quality_rows": int(len(quality_rows)),
                    "quality_ok": quality_ok,
                    "quality_failed": quality_failed,
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
            print(f"[history-samples] {date_index}/{total_dates} failed {requested_date}: {exc}", flush=True)
        run_rows.append(run_row)
        print(f"[history-samples] {date_index}/{total_dates} done {requested_date} status={run_row['status']}", flush=True)

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        candidates = pd.DataFrame(columns=HISTORY_CANDIDATE_COLUMNS)
    else:
        candidates = _normalise_history_candidate_columns(candidates)
    _assert_no_forbidden_sample_columns(candidates)

    summary = build_history_candidate_summary(
        candidates,
        start_date=start_date,
        end_date=end_date,
        hold_days=hold_days,
        target_return_pct=target_return_pct,
        secondary_target_return_pct=secondary_target_return_pct,
    )
    run_log = pd.DataFrame(run_rows)
    future_fetch_log = pd.DataFrame(future_fetch_rows)

    candidates_csv, summary_csv, run_log_csv, future_fetch_csv, markdown_path = write_history_sample_reports(
        candidates=candidates,
        summary=summary,
        run_log=run_log,
        future_fetch_log=future_fetch_log,
        output_dir=run_root,
        start_date=start_date,
        end_date=end_date,
    )
    return candidates, summary, run_log, future_fetch_log, candidates_csv, summary_csv, run_log_csv, future_fetch_csv, markdown_path


def _collect_limitups_for_history_sample(
    service: MarketDataService,
    requested_date: str,
    start_date: str,
    lookback_days: int,
    force_refresh: bool,
    workers: int,
    missing_dates: set[str] | None = None,
) -> pd.DataFrame:
    """Collect lookback limit-up pools without scanning pre-window holidays."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    known_missing = missing_dates if missing_dates is not None else set()
    anchor = pd.Timestamp(requested_date)
    start_ts = pd.Timestamp(start_date)

    for offset in range(max(1, int(lookback_days))):
        current = anchor - pd.Timedelta(days=offset)
        if current.weekday() >= 5:
            continue
        date_text = current.strftime("%Y-%m-%d")
        cached = None if force_refresh else service.limit_up_cache.read(date_text)
        if cached is not None and not cached.empty:
            frames.append(cached.copy())
            continue
        if date_text in known_missing and not force_refresh:
            print(f"[history-samples] skip known missing limit-up date {date_text}", flush=True)
            continue
        if current < start_ts and not force_refresh:
            print(f"[history-samples] skip pre-window missing limit-up cache {date_text}", flush=True)
            continue
        if not force_refresh and _cached_daily_proves_non_trading(service, date_text):
            known_missing.add(date_text)
            print(f"[history-samples] skip non-trading date from cached daily data {date_text}", flush=True)
            continue

        try:
            frame = service.collect_limit_ups(
                trade_date=date_text,
                lookback_days=1,
                force_refresh=force_refresh,
                write_processed=False,
                workers=workers,
            )
        except Exception as exc:
            message = str(exc)
            if "date_seen=0" in message and not force_refresh:
                known_missing.add(date_text)
                print(f"[history-samples] skip non-trading date after daily scan {date_text}", flush=True)
                continue
            errors.append(f"{date_text}: {message}")
            if offset == 0:
                raise
            continue
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        suffix = "" if not errors else ": " + " | ".join(errors[:5])
        raise RuntimeError(f"no limit-up data collected for history sample lookback ending {requested_date}{suffix}")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["trade_date", "code"]).drop_duplicates(["trade_date", "code"], keep="last")
    return result.reset_index(drop=True)


def _cached_daily_proves_non_trading(service: MarketDataService, date_text: str, sample_size: int = 50) -> bool:
    target = pd.Timestamp(date_text)
    covered = 0
    for cache in (service.daily_unadjusted_cache, service.daily_cache):
        paths = sorted(cache.root.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                frame = pd.read_pickle(path)
            except Exception:
                continue
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
            if dates.empty or dates.max() < target:
                continue
            covered += 1
            if bool((dates == target).any()):
                return False
            if covered >= int(sample_size):
                return True
    return covered >= 10


def evaluate_history_candidate_only(
    signal_row: pd.Series,
    service: MarketDataService,
    minute_cache: dict[str, pd.DataFrame | None],
    hold_days: int,
    target_return_pct: float,
    secondary_target_return_pct: float,
) -> dict[str, Any]:
    code = str(signal_row.get("code", "")).zfill(6)
    signal_date = str(signal_row.get("trade_date", signal_row.get("signal_date", "")))
    low_absorb_min = _to_float(signal_row.get("low_absorb_min"))
    low_absorb_max = _to_float(signal_row.get("low_absorb_max"))
    invalid_price = _to_float(signal_row.get("invalid_price"))
    allowed_bool = _bool_from_value(signal_row.get("allowed", signal_row.get("allowed_bool", False)))
    signal_type = str(signal_row.get("signal_type", ""))
    result: dict[str, Any] = {
        "signal_date": signal_date,
        "code": code,
        "name": signal_row.get("name", ""),
        "d0_date": signal_row.get("d0_date", ""),
        "days_since_d0": _to_float(signal_row.get("days_since_d0")),
        "consecutive_boards": _to_float(signal_row.get("consecutive_boards")),
        "signal_type": signal_type,
        "allowed_bool": allowed_bool,
        "eligible_for_trade": bool(allowed_bool and signal_type == "D2_LOW_ABSORB"),
        "total_score": _to_float(signal_row.get("total_score")),
        "graph_quality_score": _to_float(signal_row.get("graph_quality_score")),
        "active_money_score": _to_float(signal_row.get("active_money_score")),
        "active_cooling_score": _to_float(signal_row.get("active_cooling_score")),
        "support_score": _to_float(signal_row.get("support_score")),
        "theme_score": _to_float(signal_row.get("theme_score")),
        "trend_hold_score": _to_float(signal_row.get("trend_hold_score")),
        "entry_width_score": _to_float(signal_row.get("entry_width_score")),
        "d1_low_ma10_pct": _to_float(signal_row.get("d1_low_ma10_pct")),
        "d1_close_ma10_pct": _to_float(signal_row.get("d1_close_ma10_pct")),
        "d1_close_vwap_pct": _to_float(signal_row.get("d1_close_vwap_pct")),
        "low_absorb_width_pct": _zone_width_pct(low_absorb_min, low_absorb_max),
        "invalid_distance_pct": _invalid_distance_pct(invalid_price, low_absorb_max),
        "support_type": signal_row.get("support_type", ""),
        "low_absorb_min": low_absorb_min,
        "low_absorb_max": low_absorb_max,
        "invalid_price": invalid_price,
        "candidate_base_price": None,
        "candidate_evaluable": False,
        "future_trade_days_available": 0,
        "reasons": signal_row.get("reasons", ""),
        "key_zones_json": signal_row.get("key_zones_json", ""),
    }
    result.update(_empty_candidate_metrics())

    base_price = _candidate_base_price(signal_row, service)
    result["candidate_base_price"] = base_price
    minute = _read_minute_from_cache(code, service, minute_cache)
    if minute is None or minute.empty or "trade_date" not in minute.columns or "datetime" not in minute.columns:
        return _finalise_targets(result, target_return_pct, secondary_target_return_pct)

    minute = _normalise_minute_frame(minute)
    future_dates = _future_trade_dates(minute, signal_date)
    result["future_trade_days_available"] = len(future_dates)
    if not future_dates:
        return _finalise_targets(result, target_return_pct, secondary_target_return_pct)

    result.update(_d2open_d3_metrics(code, service, minute, future_dates))
    candidate_metrics = _path_metrics_by_horizon(
        minute,
        future_dates=future_dates,
        base_price=base_price,
        prefix="candidate",
        hold_days=hold_days,
    )
    result.update(candidate_metrics)
    if base_price is not None and _to_float(result.get("candidate_d3_max_return_pct")) is not None:
        result["candidate_evaluable"] = True
    return _finalise_targets(result, target_return_pct, secondary_target_return_pct)


def build_history_candidate_summary(
    candidates: pd.DataFrame,
    start_date: str,
    end_date: str,
    hold_days: int,
    target_return_pct: float,
    secondary_target_return_pct: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            [
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "hold_days": int(hold_days),
                    "target_return_pct": float(target_return_pct),
                    "secondary_target_return_pct": float(secondary_target_return_pct),
                    "record_count": 0,
                }
            ]
        )
    target7 = candidates["target7"].fillna(False).astype(bool) if "target7" in candidates.columns else pd.Series(dtype=bool)
    target10 = candidates["target10"].fillna(False).astype(bool) if "target10" in candidates.columns else pd.Series(dtype=bool)
    d2open_d3high = (
        candidates["target7_d2open_d3high"].fillna(False).astype(bool)
        if "target7_d2open_d3high" in candidates.columns
        else pd.Series(dtype=bool)
    )
    d2open_d3close = (
        candidates["target7_d2open_d3close"].fillna(False).astype(bool)
        if "target7_d2open_d3close" in candidates.columns
        else pd.Series(dtype=bool)
    )
    evaluable = candidates["candidate_evaluable"].fillna(False).astype(bool) if "candidate_evaluable" in candidates.columns else pd.Series(dtype=bool)
    eligible = candidates["eligible_for_trade"].fillna(False).astype(bool) if "eligible_for_trade" in candidates.columns else pd.Series(dtype=bool)
    d2open_d3high_evaluable_count = _numeric_notna_count(candidates, "d2open_d3high_return_pct")
    d2open_d3close_evaluable_count = _numeric_notna_count(candidates, "d2open_d3close_return_pct")
    return pd.DataFrame(
        [
            {
                "start_date": start_date,
                "end_date": end_date,
                "hold_days": int(hold_days),
                "target_return_pct": float(target_return_pct),
                "secondary_target_return_pct": float(secondary_target_return_pct),
                "record_count": int(len(candidates)),
                "date_count": int(candidates["signal_date"].dropna().nunique()) if "signal_date" in candidates else 0,
                "eligible_count": int(eligible.sum()) if len(eligible) else 0,
                "candidate_evaluable_count": int(evaluable.sum()) if len(evaluable) else 0,
                "candidate_target7_count": int(target7.sum()) if len(target7) else 0,
                "candidate_target7_rate": _safe_rate(int(target7.sum()), int(evaluable.sum())) if len(evaluable) else None,
                "candidate_target10_count": int(target10.sum()) if len(target10) else 0,
                "candidate_target10_rate": _safe_rate(int(target10.sum()), int(evaluable.sum())) if len(evaluable) else None,
                "d2open_d3high_evaluable_count": d2open_d3high_evaluable_count,
                "target7_d2open_d3high_count": int(d2open_d3high.sum()) if len(d2open_d3high) else 0,
                "target7_d2open_d3high_rate": _safe_rate(int(d2open_d3high.sum()), d2open_d3high_evaluable_count),
                "d2open_d3close_evaluable_count": d2open_d3close_evaluable_count,
                "target7_d2open_d3close_count": int(d2open_d3close.sum()) if len(d2open_d3close) else 0,
                "target7_d2open_d3close_rate": _safe_rate(int(d2open_d3close.sum()), d2open_d3close_evaluable_count),
                "candidate_avg_d3_max_return_pct": _mean_or_none(candidates, "candidate_d3_max_return_pct"),
                "avg_d2open_d3high_return_pct": _mean_or_none(candidates, "d2open_d3high_return_pct"),
                "avg_d2open_d3close_return_pct": _mean_or_none(candidates, "d2open_d3close_return_pct"),
                "candidate_avg_d5_max_return_pct": _mean_or_none(candidates, "candidate_d5_max_return_pct"),
                "candidate_avg_d10_max_return_pct": _mean_or_none(candidates, "candidate_d10_max_return_pct"),
            }
        ]
    )


def write_history_sample_reports(
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    run_log: pd.DataFrame,
    future_fetch_log: pd.DataFrame,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_date}_{end_date}"
    candidates_csv = output_dir / f"history_candidates_{suffix}.csv"
    summary_csv = output_dir / f"history_candidates_summary_{suffix}.csv"
    run_log_csv = output_dir / f"history_generation_log_{suffix}.csv"
    future_fetch_csv = output_dir / f"history_future_fetch_{suffix}.csv"
    markdown_path = output_dir / f"history_candidates_review_{suffix}.md"
    candidates.to_csv(candidates_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    run_log.to_csv(run_log_csv, index=False, encoding="utf-8-sig")
    future_fetch_log.to_csv(future_fetch_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(build_history_candidates_markdown(candidates, summary, run_log, start_date, end_date), encoding="utf-8")
    return candidates_csv, summary_csv, run_log_csv, future_fetch_csv, markdown_path


def build_history_candidates_markdown(
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    run_log: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> str:
    lines = [f"# History Candidates {start_date} to {end_date}", ""]
    if summary.empty:
        lines.append("No summary.")
        return "\n".join(lines)
    item = summary.iloc[0]
    lines.extend(
        [
            "## Summary",
            "",
            f"- records: **{int(item.get('record_count', 0))}**",
            f"- dates: **{int(item.get('date_count', 0))}**",
            f"- eligible: **{int(item.get('eligible_count', 0))}**",
            f"- candidate evaluable: **{int(item.get('candidate_evaluable_count', 0))}**",
            f"- legacy target7 rate (D1 close proxy, D2+D3 high window): **{_format_pct(item.get('candidate_target7_rate'))}**",
            f"- candidate target10 rate: **{_format_pct(item.get('candidate_target10_rate'))}**",
            f"- D2 open -> D3 high target7 rate: **{_format_pct(item.get('target7_d2open_d3high_rate'))}**",
            f"- D2 open -> D3 close target7 rate: **{_format_pct(item.get('target7_d2open_d3close_rate'))}**",
            f"- avg candidate D3 max return: **{_format_number(item.get('candidate_avg_d3_max_return_pct'))}%**",
            f"- avg D2 open -> D3 high return: **{_format_number(item.get('avg_d2open_d3high_return_pct'))}%**",
            f"- avg D2 open -> D3 close return: **{_format_number(item.get('avg_d2open_d3close_return_pct'))}%**",
            "",
        ]
    )
    if not run_log.empty and "status" in run_log.columns:
        lines.extend(["## Generation Log", ""])
        for status, count in run_log["status"].fillna("unknown").value_counts().items():
            lines.append(f"- {status}: **{int(count)}**")
        lines.append("")
    if not candidates.empty:
        preview_cols = [
            "signal_date",
            "code",
            "name",
            "eligible_for_trade",
            "total_score",
            "candidate_d3_max_return_pct",
            "target7",
            "d2open_d3high_return_pct",
            "target7_d2open_d3high",
        ]
        preview_cols = [column for column in preview_cols if column in candidates.columns]
        lines.extend(
            [
                "## Candidate Preview",
                "",
                "| date | code | name | eligible | total | legacy d3 max% | legacy target7 | d2open d3high% | d2open d3high target7 |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in candidates[preview_cols].head(30).iterrows():
            lines.append(
                "| {date} | {code} | {name} | {eligible} | {score} | {d3} | {target7} | {d2d3} | {d2d3_target} |".format(
                    date=row.get("signal_date", ""),
                    code=row.get("code", ""),
                    name=row.get("name", ""),
                    eligible=row.get("eligible_for_trade", ""),
                    score=_format_number(row.get("total_score")),
                    d3=_format_number(row.get("candidate_d3_max_return_pct")),
                    target7=row.get("target7", ""),
                    d2d3=_format_number(row.get("d2open_d3high_return_pct")),
                    d2d3_target=row.get("target7_d2open_d3high", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This output is a clean historical candidate sample. It contains D1-known factors and future candidate labels only.",
            "Legacy target7 is a D1-close proxy target based on the D2+D3 high window. target7_d2open_d3high uses D2 open as the entry reference and D3 intraday high as the exit opportunity.",
            "It intentionally excludes execution-only fields such as executed, buy_price, target_hit, and stop_hit.",
        ]
    )
    return "\n".join(lines)


def _empty_candidate_metrics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(
        {
            "d2_trade_date": None,
            "d3_trade_date": None,
            "d2_open_price": None,
            "d3_high_price": None,
            "d3_close_price": None,
            "d2open_d3high_return_pct": None,
            "d2open_d3close_return_pct": None,
        }
    )
    for horizon in (2, 3, 5, 10):
        label = f"candidate_d{horizon}"
        result[f"{label}_max_return_pct"] = None
        result[f"{label}_close_return_pct"] = None
        result[f"{label}_max_drawdown_pct"] = None
    return result


def _d2open_d3_metrics(
    code: str,
    service: MarketDataService,
    minute: pd.DataFrame,
    future_dates: list[str],
) -> dict[str, Any]:
    result = {
        "d2_trade_date": None,
        "d3_trade_date": None,
        "d2_open_price": None,
        "d3_high_price": None,
        "d3_close_price": None,
        "d2open_d3high_return_pct": None,
        "d2open_d3close_return_pct": None,
    }
    if len(future_dates) < 2:
        return result

    d2_date = str(future_dates[0])
    d3_date = str(future_dates[1])
    result["d2_trade_date"] = d2_date
    result["d3_trade_date"] = d3_date

    d2_rows = minute[minute["trade_date"].astype(str) == d2_date].sort_values("datetime")
    d3_rows = minute[minute["trade_date"].astype(str) == d3_date].sort_values("datetime")
    d2_open = _first_numeric(d2_rows, "open")
    if d2_open is None:
        d2_open = _daily_price_for_date(code, service, d2_date, "open")
    d3_high = _max_numeric(d3_rows, "high")
    d3_close = _last_numeric(d3_rows, "close")

    result["d2_open_price"] = d2_open
    result["d3_high_price"] = d3_high
    result["d3_close_price"] = d3_close
    if d2_open is None or d2_open <= 0:
        return result
    if d3_high is not None:
        result["d2open_d3high_return_pct"] = (d3_high / d2_open - 1.0) * 100.0
    if d3_close is not None:
        result["d2open_d3close_return_pct"] = (d3_close / d2_open - 1.0) * 100.0
    return result


def _finalise_targets(result: dict[str, Any], target_return_pct: float, secondary_target_return_pct: float) -> dict[str, Any]:
    d3_max = _to_float(result.get("candidate_d3_max_return_pct"))
    result["target7"] = bool(d3_max is not None and d3_max >= float(target_return_pct))
    result["target10"] = bool(d3_max is not None and d3_max >= float(secondary_target_return_pct))
    d2open_d3high = _to_float(result.get("d2open_d3high_return_pct"))
    d2open_d3close = _to_float(result.get("d2open_d3close_return_pct"))
    result["target7_d2open_d3high"] = bool(d2open_d3high is not None and d2open_d3high >= float(target_return_pct))
    result["target7_d2open_d3close"] = bool(d2open_d3close is not None and d2open_d3close >= float(target_return_pct))
    return result


def _normalise_history_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in HISTORY_CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[HISTORY_CANDIDATE_COLUMNS].sort_values(["signal_date", "code"]).reset_index(drop=True)


def _assert_no_forbidden_sample_columns(frame: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_HISTORY_SAMPLE_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise RuntimeError(f"history sample contains execution-only columns: {forbidden}")


def _empty_generation_row(requested_date: str) -> dict[str, Any]:
    return {
        "requested_signal_date": requested_date,
        "actual_signal_date": "",
        "status": "started",
        "limitup_rows": 0,
        "signal_rows": 0,
        "candidate_rows": 0,
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


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _numeric_notna_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").notna().sum())


def _first_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _last_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _max_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _daily_price_for_date(code: str, service: MarketDataService, trade_date: str, column: str) -> float | None:
    daily = service.daily_cache.read(code)
    if daily is None or daily.empty or "date" not in daily.columns or column not in daily.columns:
        return None
    dates = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    matched = daily[dates == str(trade_date)]
    if matched.empty:
        return None
    return _to_float(matched.iloc[-1].get(column))


def _format_pct(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.2%}"


def _format_number(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.2f}"

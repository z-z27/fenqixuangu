from __future__ import annotations

import json
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
from .provenance import runtime_provenance
from .report import write_data_quality_reports, write_signal_reports
from .universe_audit import (
    UNIVERSE_SNAPSHOT_MODES,
    UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
    StageSnapshot,
    UniverseAuditError,
    UniverseSnapshotError,
    canonical_rows_sha256,
    count_duplicate_keys,
    create_or_verify_snapshot,
    normalize_code_series,
    require_requested_signal_date_match,
    require_unique_keys,
    stage_identity,
    write_json,
)
from .v004a import annotate_v004a_input_eligibility


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
    "execution_model_version",
    "execution_bar_resolution",
    "confirmation_bar_excluded",
    "same_bar_policy",
    "transaction_costs_included",
    "slippage_included",
    "execution_reason",
    "target_hit",
    "stop_hit",
    "first_outcome",
    "first_outcome_time",
    "same_bar_ambiguous",
    "failure_reason",
    "d3_realized_return_pct",
    "d3_sell_reason",
}

HISTORY_CANDIDATE_COLUMNS = [
    "requested_signal_date",
    "signal_date",
    "code",
    "name",
    "d0_date",
    "days_since_d0",
    "consecutive_boards",
    "signal_type",
    "allowed_bool",
    "eligible_for_trade",
    "v004a_scorable_bool",
    "v004a_exclusion_reason",
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

HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS = [
    "requested_signal_date",
    "actual_signal_date",
    "stage",
    "member_key",
    "source_trade_date",
    "code",
    "name",
    "d0_date",
    "included_bool",
    "exclusion_reason",
]

RAW_SOURCE_ROW_COLUMNS = (
    "requested_signal_date",
    "source_trade_date",
    "code",
    "name",
    "market",
    "latest_price",
    "pct_chg",
    "amount",
    "turnover_rate",
    "float_market_cap",
    "total_market_cap",
    "industry",
    "limit_up_time",
    "final_limit_up_time",
    "open_board_count",
    "seal_amount",
    "consecutive_limit_up_count",
    "source",
    "open",
    "high",
    "low",
    "prev_close",
    "limit_price",
    "limitup_basis",
    "limitup_price_tolerance",
    "limitup_min_pct_chg",
)

SIGNAL_ROW_COLUMNS = (
    "requested_signal_date",
    "actual_signal_date",
    "code",
    "name",
    "d0_date",
    "days_since_d0",
    "consecutive_boards",
    "signal_type",
    "allowed_bool",
    "position_level",
    "total_score",
    "graph_quality_score",
    "active_money_score",
    "active_cooling_score",
    "support_score",
    "theme_score",
    "trend_hold_score",
    "entry_width_score",
    "low_absorb_width_pct",
    "invalid_distance_pct",
    "d1_low_ma10_pct",
    "d1_close_ma10_pct",
    "d1_close_vwap_pct",
    "support_type",
    "low_absorb_min",
    "low_absorb_max",
    "invalid_price",
    "key_zones_json",
    "reasons",
)

SCORABLE_ROW_COLUMNS = tuple(HISTORY_CANDIDATE_COLUMNS)


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
    universe_snapshot_mode: str = "create-or-verify",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    """Generate clean historical candidate samples without execution backtest fields.

    Each output row is one D1-night candidate. The row contains D1-known factors
    and future candidate labels such as candidate_d3_max_return_pct and
    target7_d2open_d3high. This layer does not emit execution-only fields.
    """
    if universe_snapshot_mode not in UNIVERSE_SNAPSHOT_MODES:
        raise ValueError(
            f"unsupported universe_snapshot_mode={universe_snapshot_mode!r}; expected one of {UNIVERSE_SNAPSHOT_MODES}"
        )
    service = MarketDataService()
    data_config = get_data_config()
    run_root = data_config.reports_dir / "history_samples" / f"{start_date}_{end_date}"
    run_root.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    future_fetch_rows: list[dict[str, Any]] = []
    universe_audit_rows: list[dict[str, Any]] = []
    universe_membership_frames: list[pd.DataFrame] = []
    minute_cache: dict[str, pd.DataFrame | None] = {}
    missing_limitup_dates: set[str] = set()
    requested_dates = _iter_weekdays(start_date, end_date)
    total_dates = len(requested_dates)

    for date_index, requested_date in enumerate(requested_dates, 1):
        run_row = _empty_generation_row(requested_date)
        run_row["universe_snapshot_mode"] = universe_snapshot_mode
        audit_row = _empty_universe_audit_row(requested_date)
        audit_appended = False
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
            raw_source = _standardize_raw_source_pool(pool, requested_date)
            raw_identity = stage_identity(_raw_stage_snapshot(raw_source))
            _apply_stage_identity(audit_row, "raw_source", raw_identity)
            audit_row["duplicate_raw_key_count"] = count_duplicate_keys(
                raw_source, ("source_trade_date", "code")
            )
            require_unique_keys(raw_source, ("source_trade_date", "code"), "raw_source_pool")
            actual_date = _latest_trade_date_from_pool(pool)
            run_row["actual_signal_date"] = actual_date
            run_row["limitup_rows"] = int(len(pool))
            audit_row["actual_signal_date"] = actual_date
            print(
                f"[history-samples] {requested_date} limitups={len(pool)} actual={actual_date}",
                flush=True,
            )
            if actual_date != requested_date:
                run_row["status"] = "skipped"
                run_row["error"] = "exact signal date limit-up pool missing"
                audit_row["generation_status"] = "skipped"
                audit_row["snapshot_status"] = "SKIPPED_NO_EXACT_SIGNAL_DATE"
                universe_membership_frames.append(
                    _build_history_universe_membership(
                        requested_date=requested_date,
                        actual_date=actual_date,
                        raw_source=raw_source,
                        signal_pool=pd.DataFrame(),
                        candidates=pd.DataFrame(),
                    )
                )
                universe_audit_rows.append(audit_row)
                audit_appended = True
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
            audit_row["quality_ok"] = quality_ok
            audit_row["quality_failed"] = quality_failed
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
            signal_frame = _standardize_signal_pool(signal_frame, requested_date, actual_date)
            audit_row["duplicate_signal_key_count"] = count_duplicate_keys(
                signal_frame, ("requested_signal_date", "code")
            )
            audit_row["signal_date_mismatch_count"] = int(
                signal_frame["requested_signal_date"].fillna("").astype(str).ne(
                    signal_frame["actual_signal_date"].fillna("").astype(str)
                ).sum()
            )
            require_unique_keys(signal_frame, ("requested_signal_date", "code"), "signal_pool")
            require_requested_signal_date_match(
                signal_frame.rename(columns={"actual_signal_date": "signal_date"})
            )
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
            for code in signal_frame.get("code", pd.Series(dtype=str)).astype(str):
                minute_cache.pop(code, None)

            day_candidate_rows: list[dict[str, Any]] = []
            for _, signal_row in signal_frame.iterrows():
                day_candidate_rows.append(
                    evaluate_history_candidate_only(
                        signal_row,
                        service=service,
                        minute_cache=minute_cache,
                        hold_days=hold_days,
                        target_return_pct=target_return_pct,
                        secondary_target_return_pct=secondary_target_return_pct,
                        requested_signal_date=requested_date,
                    )
                )
            day_candidates = annotate_v004a_input_eligibility(pd.DataFrame(day_candidate_rows))
            audit_row["signal_date_mismatch_count"] = _signal_date_mismatch_count(day_candidates)
            audit_row["duplicate_candidate_key_count"] = count_duplicate_keys(
                day_candidates, ("requested_signal_date", "code")
            )
            require_requested_signal_date_match(day_candidates)
            require_unique_keys(
                day_candidates,
                ("requested_signal_date", "code"),
                "history candidates",
            )
            stages = _build_history_stage_snapshots(
                requested_date=requested_date,
                actual_date=actual_date,
                raw_source=raw_source,
                signal_pool=signal_frame,
                candidates=day_candidates,
            )
            for stage_name, prefix in (
                ("signal_pool", "signal"),
                ("eligible_pool", "eligible"),
                ("scorable_pool", "scorable"),
            ):
                _apply_stage_identity(audit_row, prefix, stage_identity(stages[stage_name]))
            membership = _build_history_universe_membership(
                requested_date=requested_date,
                actual_date=actual_date,
                raw_source=raw_source,
                signal_pool=signal_frame,
                candidates=day_candidates,
            )
            try:
                snapshot_status, canonical_manifest_path, _ = create_or_verify_snapshot(
                    requested_signal_date=requested_date,
                    stages=stages,
                    snapshot_root=data_config.snapshot_dir,
                    run_output_dir=run_root,
                    mode=universe_snapshot_mode,
                )
            except UniverseSnapshotError as exc:
                audit_row["generation_status"] = "failed"
                audit_row["snapshot_status"] = exc.status
                audit_row["canonical_manifest_path"] = str(exc.canonical_manifest_path)
                universe_membership_frames.append(membership)
                universe_audit_rows.append(audit_row)
                audit_appended = True
                raise
            audit_row["generation_status"] = "generated"
            audit_row["snapshot_status"] = snapshot_status
            audit_row["canonical_manifest_path"] = str(canonical_manifest_path)
            universe_membership_frames.append(membership)
            universe_audit_rows.append(audit_row)
            audit_appended = True
            candidate_rows.extend(day_candidates.to_dict(orient="records"))
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
                    "universe_snapshot_mode": universe_snapshot_mode,
                    "snapshot_status": snapshot_status,
                    "canonical_manifest_path": str(canonical_manifest_path),
                }
            )
        except UniverseAuditError as exc:
            run_row["status"] = "failed"
            run_row["error"] = str(exc)
            run_row["signal_date_mismatch"] = "signal_date_mismatch" in str(exc)
            if not audit_appended:
                audit_row["generation_status"] = "failed"
                audit_row["snapshot_status"] = getattr(exc, "status", "INTEGRITY_FAILURE")
                audit_row["canonical_manifest_path"] = str(
                    getattr(exc, "canonical_manifest_path", "")
                )
                universe_audit_rows.append(audit_row)
            run_rows.append(run_row)
            _write_partial_history_universe_failure(
                run_root=run_root,
                start_date=start_date,
                end_date=end_date,
                run_rows=run_rows,
                audit_rows=universe_audit_rows,
                membership_frames=universe_membership_frames,
            )
            print(f"[history-samples] {date_index}/{total_dates} failed {requested_date}: {exc}", flush=True)
            raise
        except Exception as exc:
            run_row["status"] = "failed"
            run_row["error"] = str(exc)
            if not audit_appended:
                audit_row["generation_status"] = "failed"
                audit_row["snapshot_status"] = "GENERATION_FAILED"
                universe_audit_rows.append(audit_row)
                audit_appended = True
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
    universe_audit = pd.DataFrame(universe_audit_rows)
    universe_membership = _combine_membership_frames(universe_membership_frames)

    candidates_csv, summary_csv, run_log_csv, future_fetch_csv, markdown_path = write_history_sample_reports(
        candidates=candidates,
        summary=summary,
        run_log=run_log,
        future_fetch_log=future_fetch_log,
        output_dir=run_root,
        start_date=start_date,
        end_date=end_date,
        universe_audit=universe_audit,
        universe_snapshot_mode=universe_snapshot_mode,
    )
    _write_history_universe_outputs(
        output_dir=run_root,
        start_date=start_date,
        end_date=end_date,
        lookback_days=int(lookback_days),
        signal_days=signal_days,
        eval_days=eval_days,
        hold_days=int(hold_days),
        target_return_pct=float(target_return_pct),
        universe_snapshot_mode=universe_snapshot_mode,
        candidates=candidates,
        universe_audit=universe_audit,
        universe_membership=universe_membership,
    )
    return candidates, summary, run_log, future_fetch_log, candidates_csv, summary_csv, run_log_csv, future_fetch_csv, markdown_path


def _standardize_raw_source_pool(pool: pd.DataFrame, requested_date: str) -> pd.DataFrame:
    frame = pool.copy().reset_index(drop=True)
    missing = [column for column in ("trade_date", "code") if column not in frame.columns]
    if missing:
        raise UniverseAuditError(f"raw_source_pool missing columns: {missing}")
    frame["requested_signal_date"] = str(requested_date)
    frame["source_trade_date"] = frame["trade_date"].astype(str)
    frame["code"] = normalize_code_series(frame["code"])
    frame = frame[frame["code"].ne("")].copy()
    for column in RAW_SOURCE_ROW_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[list(RAW_SOURCE_ROW_COLUMNS)].sort_values(
        ["source_trade_date", "code"], kind="mergesort"
    ).reset_index(drop=True)


def _standardize_signal_pool(
    signals: pd.DataFrame,
    requested_date: str,
    actual_date: str,
) -> pd.DataFrame:
    frame = signals.copy().reset_index(drop=True)
    if "code" not in frame.columns:
        frame["code"] = pd.Series(dtype=str)
    frame["requested_signal_date"] = str(requested_date)
    frame["actual_signal_date"] = (
        frame["trade_date"].fillna("").astype(str)
        if "trade_date" in frame.columns
        else str(actual_date)
    )
    if "trade_date" not in frame.columns:
        frame["trade_date"] = frame["actual_signal_date"]
    frame["code"] = normalize_code_series(frame["code"])
    allowed_source = frame["allowed"] if "allowed" in frame.columns else frame.get("allowed_bool", False)
    if isinstance(allowed_source, pd.Series):
        frame["allowed_bool"] = allowed_source.map(_bool_from_value)
    else:
        frame["allowed_bool"] = bool(allowed_source)
    for column in SIGNAL_ROW_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.sort_values(["requested_signal_date", "code"], kind="mergesort").reset_index(drop=True)


def _raw_stage_snapshot(raw_source: pd.DataFrame) -> StageSnapshot:
    return StageSnapshot(
        stage="raw_source_pool",
        frame=raw_source,
        key_columns=("source_trade_date", "code"),
        row_columns=RAW_SOURCE_ROW_COLUMNS,
    )


def _build_history_stage_snapshots(
    requested_date: str,
    actual_date: str,
    raw_source: pd.DataFrame,
    signal_pool: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, StageSnapshot]:
    del requested_date, actual_date
    eligible_mask = (
        signal_pool.get("allowed_bool", pd.Series(False, index=signal_pool.index)).fillna(False).astype(bool)
        & signal_pool.get("signal_type", pd.Series("", index=signal_pool.index)).fillna("").astype(str).eq("D2_LOW_ABSORB")
    )
    scorable_mask = candidates.get(
        "v004a_scorable_bool", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(bool)
    return {
        "raw_source_pool": _raw_stage_snapshot(raw_source),
        "signal_pool": StageSnapshot(
            stage="signal_pool",
            frame=signal_pool,
            key_columns=("requested_signal_date", "code"),
            row_columns=SIGNAL_ROW_COLUMNS,
        ),
        "eligible_pool": StageSnapshot(
            stage="eligible_pool",
            frame=signal_pool.loc[eligible_mask].copy(),
            key_columns=("requested_signal_date", "code"),
            row_columns=SIGNAL_ROW_COLUMNS,
        ),
        "scorable_pool": StageSnapshot(
            stage="scorable_pool",
            frame=candidates.loc[scorable_mask].copy(),
            key_columns=("requested_signal_date", "code"),
            row_columns=SCORABLE_ROW_COLUMNS,
        ),
    }


def _build_history_universe_membership(
    requested_date: str,
    actual_date: str,
    raw_source: pd.DataFrame,
    signal_pool: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in raw_source.iterrows():
        source_date = str(row.get("source_trade_date", ""))
        code = str(row.get("code", ""))
        rows.append(
            _membership_row(
                requested_date,
                actual_date,
                "raw_source_pool",
                f"{source_date}|{code}",
                source_date,
                row,
                True,
                "",
            )
        )
    for _, row in signal_pool.iterrows():
        code = str(row.get("code", ""))
        key = f"{requested_date}|{code}"
        rows.append(_membership_row(requested_date, actual_date, "signal_pool", key, "", row, True, ""))
        eligible = bool(_bool_from_value(row.get("allowed_bool"))) and str(row.get("signal_type", "")) == "D2_LOW_ABSORB"
        rows.append(
            _membership_row(
                requested_date,
                actual_date,
                "eligible_pool",
                key,
                "",
                row,
                eligible,
                "" if eligible else "not_eligible",
            )
        )
    for _, row in candidates.iterrows():
        code = str(row.get("code", ""))
        included = bool(_bool_from_value(row.get("v004a_scorable_bool")))
        rows.append(
            _membership_row(
                requested_date,
                actual_date,
                "scorable_pool",
                f"{requested_date}|{code}",
                "",
                row,
                included,
                str(row.get("v004a_exclusion_reason", "")),
            )
        )
    frame = pd.DataFrame(rows, columns=HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS)
    return frame


def _membership_row(
    requested_date: str,
    actual_date: str,
    stage: str,
    member_key: str,
    source_trade_date: str,
    row: pd.Series,
    included: bool,
    exclusion_reason: str,
) -> dict[str, Any]:
    return {
        "requested_signal_date": str(requested_date),
        "actual_signal_date": str(actual_date),
        "stage": stage,
        "member_key": member_key,
        "source_trade_date": source_trade_date,
        "code": str(row.get("code", "")),
        "name": row.get("name", ""),
        "d0_date": row.get("d0_date", ""),
        "included_bool": bool(included),
        "exclusion_reason": exclusion_reason,
    }


def _empty_universe_audit_row(requested_date: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "requested_signal_date": str(requested_date),
        "actual_signal_date": "",
        "generation_status": "started",
        "quality_ok": 0,
        "quality_failed": 0,
        "signal_date_mismatch_count": 0,
        "duplicate_raw_key_count": 0,
        "duplicate_signal_key_count": 0,
        "duplicate_candidate_key_count": 0,
        "snapshot_status": "",
        "canonical_manifest_path": "",
    }
    for prefix in ("raw_source", "signal", "eligible", "scorable"):
        row[f"{prefix}_row_count"] = 0
        row[f"{prefix}_code_count"] = 0
        row[f"{prefix}_code_set_sha256"] = ""
        row[f"{prefix}_key_set_sha256"] = ""
        row[f"{prefix}_rows_sha256"] = ""
    return row


def _apply_stage_identity(row: dict[str, Any], prefix: str, identity: dict[str, Any]) -> None:
    row[f"{prefix}_row_count"] = int(identity["row_count"])
    row[f"{prefix}_code_count"] = int(identity["code_count"])
    row[f"{prefix}_code_set_sha256"] = str(identity["code_set_sha256"])
    row[f"{prefix}_key_set_sha256"] = str(identity["key_set_sha256"])
    row[f"{prefix}_rows_sha256"] = str(identity["rows_sha256"])


def _signal_date_mismatch_count(frame: pd.DataFrame) -> int:
    if frame.empty or "requested_signal_date" not in frame.columns or "signal_date" not in frame.columns:
        return 0
    return int(
        frame["requested_signal_date"].fillna("").astype(str).ne(
            frame["signal_date"].fillna("").astype(str)
        ).sum()
    )


def _combine_membership_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    stage_order = {stage: index for index, stage in enumerate(("raw_source_pool", "signal_pool", "eligible_pool", "scorable_pool"))}
    combined["__stage_order"] = combined["stage"].map(stage_order).fillna(999)
    combined = combined.sort_values(
        ["requested_signal_date", "__stage_order", "member_key"], kind="mergesort"
    ).drop(columns=["__stage_order"])
    return combined[HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS].reset_index(drop=True)


def _write_partial_history_universe_failure(
    run_root: Path,
    start_date: str,
    end_date: str,
    run_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    membership_frames: list[pd.DataFrame],
) -> None:
    suffix = f"{start_date}_{end_date}"
    pd.DataFrame(run_rows).to_csv(
        run_root / f"history_generation_log_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    pd.DataFrame(audit_rows).to_csv(
        run_root / f"history_universe_audit_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    _combine_membership_frames(membership_frames).to_csv(
        run_root / f"history_universe_membership_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def _write_history_universe_outputs(
    output_dir: Path,
    start_date: str,
    end_date: str,
    lookback_days: int,
    signal_days: int | None,
    eval_days: int | None,
    hold_days: int,
    target_return_pct: float,
    universe_snapshot_mode: str,
    candidates: pd.DataFrame,
    universe_audit: pd.DataFrame,
    universe_membership: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    suffix = f"{start_date}_{end_date}"
    audit_path = output_dir / f"history_universe_audit_{suffix}.csv"
    membership_path = output_dir / f"history_universe_membership_{suffix}.csv"
    manifest_path = output_dir / f"history_universe_manifest_{suffix}.json"
    universe_audit.to_csv(audit_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    universe_membership.to_csv(
        membership_path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    membership_hash = canonical_rows_sha256(
        universe_membership,
        HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS,
        ("requested_signal_date", "stage", "member_key"),
    )
    candidates_hash = canonical_rows_sha256(
        candidates,
        HISTORY_CANDIDATE_COLUMNS,
        ("requested_signal_date", "code"),
    )
    manifest = {
        "universe_snapshot_schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "lookback_days": int(lookback_days),
        "signal_days": signal_days,
        "eval_days": eval_days,
        "hold_days": int(hold_days),
        "target_return_pct": float(target_return_pct),
        "universe_snapshot_mode": universe_snapshot_mode,
        "membership_file": str(membership_path),
        "membership_canonical_rows_sha256": membership_hash,
        "history_candidates_canonical_rows_sha256": candidates_hash,
        "dates": _records_for_json(universe_audit),
        "runtime_provenance": runtime_provenance(),
    }
    write_json(manifest_path, manifest)
    return audit_path, membership_path, manifest_path


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


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
    result["code"] = normalize_code_series(result.get("code", pd.Series(dtype=object)))
    result["trade_date"] = result.get("trade_date", "").astype(str)
    result = result.sort_values(["trade_date", "code"], kind="mergesort")
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
    requested_signal_date: str | None = None,
) -> dict[str, Any]:
    code = normalize_code_series(pd.Series([signal_row.get("code", "")])).iloc[0]
    signal_date = str(
        signal_row.get(
            "trade_date",
            signal_row.get("actual_signal_date", signal_row.get("signal_date", "")),
        )
    )
    requested_date = str(
        requested_signal_date
        if requested_signal_date is not None
        else signal_row.get("requested_signal_date", signal_date)
    )
    if requested_date != signal_date:
        raise UniverseAuditError(
            "signal_date_mismatch: "
            f"requested_signal_date={requested_date}, signal_date={signal_date}, code={code}"
        )
    low_absorb_min = _to_float(signal_row.get("low_absorb_min"))
    low_absorb_max = _to_float(signal_row.get("low_absorb_max"))
    invalid_price = _to_float(signal_row.get("invalid_price"))
    allowed_bool = _bool_from_value(signal_row.get("allowed", signal_row.get("allowed_bool", False)))
    signal_type = str(signal_row.get("signal_type", ""))
    result: dict[str, Any] = {
        "requested_signal_date": requested_date,
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
    universe_audit: pd.DataFrame | None = None,
    universe_snapshot_mode: str = "off",
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
    markdown_path.write_text(
        build_history_candidates_markdown(
            candidates,
            summary,
            run_log,
            start_date,
            end_date,
            universe_audit=universe_audit,
            universe_snapshot_mode=universe_snapshot_mode,
        ),
        encoding="utf-8",
    )
    return candidates_csv, summary_csv, run_log_csv, future_fetch_csv, markdown_path


def build_history_candidates_markdown(
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    run_log: pd.DataFrame,
    start_date: str,
    end_date: str,
    universe_audit: pd.DataFrame | None = None,
    universe_snapshot_mode: str = "off",
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
    lines.extend(["## Universe Reproducibility", ""])
    lines.append(f"- snapshot mode: **{universe_snapshot_mode}**")
    audit = universe_audit if universe_audit is not None else pd.DataFrame()
    if audit.empty:
        lines.append("- universe audit: **not available**")
    else:
        columns = [
            "requested_signal_date",
            "generation_status",
            "raw_source_code_count",
            "signal_code_count",
            "eligible_code_count",
            "scorable_code_count",
            "duplicate_raw_key_count",
            "duplicate_signal_key_count",
            "duplicate_candidate_key_count",
            "signal_date_mismatch_count",
            "snapshot_status",
        ]
        columns = [column for column in columns if column in audit.columns]
        lines.append("")
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in audit[columns].iterrows():
            lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
        lines.extend(["", "### Per-stage hashes", ""])
        hash_columns = [
            "requested_signal_date",
            "raw_source_rows_sha256",
            "signal_rows_sha256",
            "eligible_rows_sha256",
            "scorable_rows_sha256",
        ]
        hash_columns = [column for column in hash_columns if column in audit.columns]
        lines.append("| " + " | ".join(hash_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(hash_columns)) + " |")
        for _, row in audit[hash_columns].iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in hash_columns) + " |")
    lines.extend(
        [
            "",
            "A matching candidate-universe hash proves only that the audited member sets and selected row fields are reproducible.",
            "`cache_snapshot_complete=False` still means the complete daily and 5-minute market-data cache is not frozen.",
            "Changing the candidate universe changes cross-sectional percentile features; runs with different universe hashes must not be compared directly.",
            "Canonical snapshots are never overwritten automatically when a mismatch is found.",
            "",
        ]
    )
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
    frame = annotate_v004a_input_eligibility(candidates)
    if "requested_signal_date" not in frame.columns:
        raise UniverseAuditError("history candidates missing requested_signal_date")
    require_requested_signal_date_match(frame)
    require_unique_keys(frame, ("requested_signal_date", "code"), "history candidates")
    for column in HISTORY_CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[HISTORY_CANDIDATE_COLUMNS].sort_values(
        ["requested_signal_date", "code"], kind="mergesort"
    ).reset_index(drop=True)


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
        "universe_snapshot_mode": "",
        "snapshot_status": "",
        "canonical_manifest_path": "",
        "signal_date_mismatch": False,
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

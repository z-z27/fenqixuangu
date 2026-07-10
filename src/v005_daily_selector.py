from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtester import _candidate_base_price, build_signals_for_pool
from .config import get_data_config
from .daily_ranking import apply_daily_research_ranking, load_ranking_model
from .loaders import MarketDataService
from .policy_config import frozen_policy_input_checks, get_default_policy, normalized_sha256, validate_model_dates_for_signal_date
from .provenance import file_sha256, runtime_provenance
from .report import write_data_quality_reports, write_signal_reports
from .ranking_backtest import validate_ranking_model
from .signal_engine import Signal
from .v004a import (
    BASE_INTERACTION_SPECS,
    BASE_RANK_SPECS,
    DAY_BUCKET_COLUMNS,
    DEFAULT_CLOSE_RETURN_COLUMN,
    DEFAULT_HIGH_RETURN_COLUMN,
    DEFAULT_TARGET_COLUMN,
    REALIZED_RETURN_KIND,
    TARGET_METRIC_KIND,
    MODEL_ID_V004A,
    SCOPE_WALK_FORWARD,
    _score_logistic_frame,
    _score_manual_and_hand_models,
    add_scored_model_rank,
    build_scored_candidates_output,
)
from .v005_fixed_grid_holdout import load_fixed_v004a_beta
from .v005_fallback_gate import PRIMARY_POLICY, ctx_for_codes, is_policy_fallback, is_risk_ticket, norm, parse_codes
from .v005_set_selector import (
    DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_CANDIDATE_TOP_K as RESEARCH_DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
    DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
    DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
    DEFAULT_EXTREME_PRICE_PENALTY_GRID,
    DEFAULT_EXTREME_VWAP_PENALTY_GRID,
    DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    DEFAULT_TOP_N as RESEARCH_DEFAULT_TOP_N,
    DEFAULT_V004A_L2 as RESEARCH_DEFAULT_V004A_L2,
    DEFAULT_V004A_POSITIVE_WEIGHT as RESEARCH_DEFAULT_V004A_POSITIVE_WEIGHT,
    GRID_PARAM_COLUMNS,
    V002_MODEL_ID,
    V004A_MODEL_ID,
    build_candidate_pool,
    build_combo_candidates,
    build_rule_grid,
    explode_selected_combos,
    prepare_scored_candidates,
    score_combos,
    select_best_combo_by_date,
)

DEFAULT_POLICY = get_default_policy()
DEFAULT_COEFFICIENTS_FILE = DEFAULT_POLICY.coefficients_path
DEFAULT_RANKING_MODEL_FILE = DEFAULT_POLICY.ranking_model_path
DEFAULT_OUTPUT_ROOT = Path("reports/daily_v005")
DEFAULT_GRID_ID = DEFAULT_POLICY.grid_id
DEFAULT_COEFFICIENT_PREDICT_DATE = DEFAULT_POLICY.coefficient_predict_date
DEFAULT_CANDIDATE_TOP_K = DEFAULT_POLICY.candidate_top_k
DEFAULT_TOP_N = DEFAULT_POLICY.top_n
DEFAULT_V004A_L2 = DEFAULT_POLICY.v004a_l2
DEFAULT_V004A_POSITIVE_WEIGHT = DEFAULT_POLICY.v004a_positive_weight

if (
    RESEARCH_DEFAULT_CANDIDATE_TOP_K != DEFAULT_CANDIDATE_TOP_K
    or RESEARCH_DEFAULT_TOP_N != DEFAULT_TOP_N
    or RESEARCH_DEFAULT_V004A_L2 != DEFAULT_V004A_L2
    or RESEARCH_DEFAULT_V004A_POSITIVE_WEIGHT != DEFAULT_V004A_POSITIVE_WEIGHT
):
    raise RuntimeError("frozen policy manifest disagrees with v005 research defaults")

STRATEGY_PRIMARY = PRIMARY_POLICY
STRATEGY_BASELINE = "baseline_v005_fixed_grid"
STRATEGY_V002 = "v002_top3_control"
STRATEGY_V004A = "v004a_top3_control"

SELECTION_COLUMNS = [
    "policy_version",
    "deployment_status",
    "research_only",
    "selection_intent",
    "manual_review_required",
    "transaction_costs_included",
    "liquidity_execution_validated",
    "frozen_policy_inputs_verified",
    "v002_source_model_id",
    "strategy_role",
    "strategy",
    "is_primary_buy",
    "buy_priority",
    "display_order",
    "signal_date",
    "action",
    "fallback_triggered",
    "selected_grid_id",
    "daily_rank",
    "code",
    "name",
    "v004a_model_rank",
    "v002_model_rank",
    "v004a_score",
    "v002_score",
    "rank_total_score",
    "rank_log_candidate_base_price",
    "rank_d1_close_vwap_pct",
    "inter_close_low",
    "extreme_price",
    "extreme_vwap",
    "extreme_close_low",
    "low_absorb_min",
    "low_absorb_max",
    "invalid_price",
    "candidate_base_price",
    "reasons",
]

DECISION_COLUMNS = [
    "signal_date",
    "policy_version",
    "deployment_status",
    "research_only",
    "metric_scope",
    "manual_review_required",
    "transaction_costs_included",
    "liquidity_execution_validated",
    "frozen_policy_inputs_verified",
    "matches_frozen_manifest",
    "v002_source_model_id",
    "final_strategy",
    "action",
    "fallback_triggered",
    "research_watchlist_codes",
    "primary_buy_codes",
    "v005_baseline_codes",
    "v002_codes",
    "v004a_codes",
    "selected_grid_id",
    "gate_v002_extreme_vwap_count",
    "gate_v002_extreme_close_low_count",
    "gate_v005_avg_v002_rank",
    "gate_v005_has_risk_ticket",
    "candidate_pool_count",
    "combo_candidate_count",
]


def run_v005_daily_selector(
    signals: list[Signal] | pd.DataFrame,
    output_dir: str | Path | None = None,
    coefficients_file: str | Path = DEFAULT_COEFFICIENTS_FILE,
    grid_id: int = DEFAULT_GRID_ID,
    coefficient_predict_date: str = DEFAULT_COEFFICIENT_PREDICT_DATE,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
    ranking_model_file: str | Path = DEFAULT_RANKING_MODEL_FILE,
    source_signals_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    signal_frame = signals_to_frame(signals)
    if signal_frame.empty:
        raise RuntimeError("no signals for v005 daily selector")
    signal_date = require_single_signal_date(signal_frame)
    signal_frame, ranking_meta = apply_daily_research_ranking(
        signal_frame,
        model_file=ranking_model_file,
        top_n=int(top_n),
    )
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_ROOT / signal_date
    out_dir.mkdir(parents=True, exist_ok=True)

    live_features, feature_info = prepare_live_v004a_features(signal_frame)
    beta, feature_columns, coefficient_meta = load_fixed_v004a_beta(
        coefficients_file=Path(coefficients_file),
        coefficient_predict_date=str(coefficient_predict_date),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    validate_model_dates_for_signal_date(
        predict_date=str(coefficient_meta["coefficient_predict_date"]),
        train_end=str(coefficient_meta["coefficient_train_end"]),
        signal_date=signal_date,
    )
    policy_meta = build_runtime_policy_meta(
        coefficients_file=Path(coefficients_file),
        coefficient_meta=coefficient_meta,
        grid_id=int(grid_id),
        top_n=int(top_n),
        candidate_top_k=int(candidate_top_k),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        ranking_model_file=Path(ranking_model_file),
        ranking_model_id=str(ranking_meta["model_id"]),
    )
    missing_features = [column for column in feature_columns if column not in live_features.columns]
    if missing_features:
        raise RuntimeError(f"live signals missing v004a coefficient features: {missing_features}")

    scored = score_live_candidates(
        live_features=live_features,
        beta=beta,
        feature_columns=feature_columns,
        feature_info=feature_info,
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        ranking_model_file=Path(ranking_model_file),
    )
    scored_path = out_dir / f"v005_daily_scored_candidates_{signal_date}.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    prepared_scored = prepare_scored_candidates(scored_path)

    candidate_pool = build_candidate_pool(
        prepared_scored,
        candidate_top_k=int(candidate_top_k),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        v002_model_id=str(ranking_meta["model_id"]),
    )
    if len(candidate_pool) < int(top_n):
        raise RuntimeError(f"v005 daily candidate pool has {len(candidate_pool)} rows; at least top_n={top_n} are required")
    combo_candidates = build_combo_candidates(candidate_pool, top_n=int(top_n))
    if combo_candidates.empty:
        raise RuntimeError("v005 daily selector produced no valid TopN combinations")
    grid = build_default_grid()
    fixed_params = select_grid_params(grid, grid_id=int(grid_id))
    selected_combos = select_best_combo_by_date(score_combos(combo_candidates, fixed_params))
    if selected_combos.empty:
        raise RuntimeError("v005 daily selector did not select a combination")
    baseline_top3 = explode_selected_combos(selected_combos, candidate_pool, top_n=int(top_n))

    decisions, selections = build_daily_policy_outputs(
        selected_combos=selected_combos,
        candidate_pool=candidate_pool,
        top_n=int(top_n),
        policy_meta=policy_meta,
    )
    decisions["candidate_pool_count"] = int(len(candidate_pool))
    decisions["combo_candidate_count"] = int(len(combo_candidates))

    decision_path = out_dir / f"v005_daily_decision_{signal_date}.csv"
    selection_path = out_dir / f"v005_daily_selection_{signal_date}.csv"
    combo_path = out_dir / f"v005_daily_selected_combos_{signal_date}.csv"
    baseline_path = out_dir / f"v005_daily_baseline_top3_{signal_date}.csv"
    report_path = out_dir / f"v005_daily_report_{signal_date}.md"

    decisions.to_csv(decision_path, index=False, encoding="utf-8-sig")
    selections.to_csv(selection_path, index=False, encoding="utf-8-sig")
    selected_combos.to_csv(combo_path, index=False, encoding="utf-8-sig")
    baseline_top3.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_daily_report(
            signal_date=signal_date,
            output_dir=out_dir,
            scored_path=scored_path,
            coefficients_file=Path(coefficients_file),
            coefficient_meta=coefficient_meta,
            grid_id=int(grid_id),
            grid_params=fixed_params,
            decisions=decisions,
            selections=selections,
            policy_meta=policy_meta,
        ),
        encoding="utf-8",
    )
    source_signals_path = Path(source_signals_file) if source_signals_file else None
    selector_meta = pd.DataFrame(
        [
            {
                **DEFAULT_POLICY.provenance(),
                **runtime_provenance(),
                **policy_meta,
                "signal_date": signal_date,
                "source_signals_file": str(source_signals_path) if source_signals_path else "in_memory",
                "source_signals_sha256": (
                    file_sha256(source_signals_path) if source_signals_path and source_signals_path.is_file() else ""
                ),
                "scored_candidates_sha256": file_sha256(scored_path),
                "decision_sha256": file_sha256(decision_path),
                "selection_sha256": file_sha256(selection_path),
                "cache_snapshot_complete": False,
            }
        ]
    )
    selector_meta.to_csv(out_dir / f"v005_daily_selector_meta_{signal_date}.csv", index=False, encoding="utf-8-sig")
    return decisions, selections, selected_combos, scored, report_path


def run_v005_daily_from_market(
    date: str | None = None,
    lookback_days: int = 5,
    days: int | None = None,
    max_codes: int | None = None,
    force_refresh: bool = False,
    workers: int = 6,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ranking_model: str | Path = DEFAULT_RANKING_MODEL_FILE,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    coefficients_file: str | Path = DEFAULT_COEFFICIENTS_FILE,
    coefficient_predict_date: str = DEFAULT_COEFFICIENT_PREDICT_DATE,
    grid_id: int = DEFAULT_GRID_ID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    service = MarketDataService()
    pool = service.collect_limit_ups(
        trade_date=date,
        lookback_days=int(lookback_days),
        force_refresh=force_refresh,
        write_processed=True,
        workers=int(workers),
    )
    signal_date = latest_trade_date(pool)
    signals, quality_rows = build_signals_for_pool(
        service=service,
        pool=pool,
        as_of_date=signal_date,
        days=days,
        max_codes=max_codes,
        force_refresh=force_refresh,
    )
    ranked_signals, ranking_meta = apply_daily_research_ranking(signals, model_file=ranking_model, top_n=int(top_n))
    out_root = Path(output_root)
    signal_csv, signal_md = write_signal_reports(ranked_signals, get_data_config().reports_dir / "daily_signals", trade_date=signal_date)
    quality_csv, quality_md = write_data_quality_reports(quality_rows, get_data_config().reports_dir / "data_quality", trade_date=signal_date)
    daily_output_dir = out_root / signal_date
    decisions, selections, combos, scored, report_path = run_v005_daily_selector(
        ranked_signals,
        output_dir=daily_output_dir,
        coefficients_file=coefficients_file,
        grid_id=int(grid_id),
        coefficient_predict_date=coefficient_predict_date,
        top_n=int(top_n),
        candidate_top_k=int(candidate_top_k),
        ranking_model_file=ranking_model,
        source_signals_file=signal_csv,
    )
    runtime_meta = runtime_provenance()
    decision_meta = decisions.iloc[0].to_dict() if not decisions.empty else {}
    ranking_model_path = Path(ranking_model)
    coefficients_path = Path(coefficients_file)
    meta = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "policy_version": decision_meta.get("policy_version", ""),
                "deployment_status": decision_meta.get("deployment_status", ""),
                "research_only": True,
                "metric_scope": decision_meta.get("metric_scope", DEFAULT_POLICY.metric_scope),
                "manual_review_required": True,
                "transaction_costs_included": False,
                "liquidity_execution_validated": False,
                "frozen_policy_inputs_verified": bool(decision_meta.get("frozen_policy_inputs_verified", False)),
                "matches_frozen_manifest": bool(decision_meta.get("matches_frozen_manifest", False)),
                "v002_source_model_id": decision_meta.get("v002_source_model_id", ranking_meta.get("model_id", "")),
                "limitup_rows": int(len(pool)),
                "signals": int(len(ranked_signals)),
                "quality_rows": int(len(quality_rows)),
                "ranking_model_id": ranking_meta.get("model_id", ""),
                "ranking_model_path": ranking_meta.get("model_path", ""),
                "coefficient_predict_date": str(coefficient_predict_date),
                "grid_id": int(grid_id),
                "candidate_top_k": int(candidate_top_k),
                "top_n": int(top_n),
                "v2_signals_csv": str(signal_csv),
                "v2_signals_md": str(signal_md),
                "quality_csv": str(quality_csv),
                "quality_md": str(quality_md),
                "v005_report": str(report_path),
                "signal_csv_sha256": file_sha256(signal_csv),
                "quality_csv_sha256": file_sha256(quality_csv),
                "ranking_model_sha256": file_sha256(ranking_model_path) if ranking_model_path.is_file() else "missing",
                "ranking_model_normalized_sha256": normalized_sha256(ranking_model_path) if ranking_model_path.is_file() else "missing",
                "coefficients_normalized_sha256": normalized_sha256(coefficients_path) if coefficients_path.is_file() else "missing",
                "policy_manifest_sha256": file_sha256(DEFAULT_POLICY.manifest_path),
                "policy_manifest_normalized_sha256": DEFAULT_POLICY.manifest_sha256,
                "target_return_pct": DEFAULT_POLICY.target_return_pct,
                "min_forward_dates": DEFAULT_POLICY.min_forward_dates,
                "market_data_sources": "akshare,eastmoney,tencent,sina",
                "cache_snapshot_complete": False,
                **runtime_meta,
            }
        ]
    )
    meta_path = daily_output_dir / f"v005_daily_run_meta_{signal_date}.csv"
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    return decisions, selections, combos, scored, report_path, signal_csv, quality_csv


def signals_to_frame(signals: list[Signal] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(signals, pd.DataFrame):
        frame = signals.copy()
    else:
        frame = pd.DataFrame([signal.to_dict() for signal in signals])
    if frame.empty:
        return frame
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "trade_date" not in frame.columns and "signal_date" in frame.columns:
        frame["trade_date"] = frame["signal_date"].astype(str)
    return frame.reset_index(drop=True)


def prepare_live_v004a_features(signals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = signals.copy()
    required = [
        "trade_date",
        "code",
        "signal_type",
        "total_score",
        "graph_quality_score",
        "theme_score",
        "trend_hold_score",
        "d1_close_ma10_pct",
        "d1_low_ma10_pct",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"v005 daily signals missing required columns: {missing}")
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "allowed_bool" in frame.columns:
        allowed = bool_series(frame["allowed_bool"])
    else:
        allowed = bool_series(frame.get("allowed", False))
    frame["allowed_bool"] = allowed
    frame["eligible_for_trade"] = allowed & frame["signal_type"].astype(str).eq("D2_LOW_ABSORB")

    service = MarketDataService()
    if "candidate_base_price" not in frame.columns:
        frame["candidate_base_price"] = np.nan
    missing_base = pd.to_numeric(frame["candidate_base_price"], errors="coerce").isna() | (pd.to_numeric(frame["candidate_base_price"], errors="coerce") <= 0)
    if missing_base.any():
        frame.loc[missing_base, "candidate_base_price"] = [
            _candidate_base_price(row, service) for _, row in frame[missing_base].iterrows()
        ]

    if "days_since_d0" not in frame.columns:
        frame["days_since_d0"] = np.nan
    if "active_money_score" not in frame.columns:
        frame["active_money_score"] = np.nan
    if "d1_close_vwap_pct" not in frame.columns:
        frame["d1_close_vwap_pct"] = np.nan
    if "graph_quality_score" not in frame.columns:
        frame["graph_quality_score"] = 0.0
    if "name" not in frame.columns:
        frame["name"] = ""
    if "reasons" not in frame.columns:
        frame["reasons"] = ""

    numeric_columns = [
        "candidate_base_price",
        "d1_close_ma10_pct",
        "d1_low_ma10_pct",
        "trend_hold_score",
        "total_score",
        "theme_score",
        "days_since_d0",
        "active_money_score",
        "d1_close_vwap_pct",
        "low_absorb_min",
        "low_absorb_max",
        "invalid_price",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    eligible = frame[
        frame["eligible_for_trade"].fillna(False).astype(bool)
        & frame["candidate_base_price"].notna()
        & (pd.to_numeric(frame["candidate_base_price"], errors="coerce") > 0)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no eligible v005 daily rows after allowed/signal_type/base-price filters")

    eligible["log_candidate_base_price"] = np.log(pd.to_numeric(eligible["candidate_base_price"], errors="coerce"))
    rank_specs = [*BASE_RANK_SPECS]
    optional_specs = []
    if "active_money_score" in eligible.columns:
        optional_specs.append(("active_money_score", "rank_active_money_score"))
    if "d1_close_vwap_pct" in eligible.columns:
        optional_specs.append(("d1_close_vwap_pct", "rank_d1_close_vwap_pct"))
    rank_specs.extend(optional_specs)

    for raw_column, rank_column in rank_specs:
        if raw_column not in eligible.columns:
            eligible[raw_column] = np.nan
        eligible[rank_column] = (
            eligible.groupby("signal_date", dropna=False)[raw_column]
            .transform(lambda values: pd.to_numeric(values, errors="coerce").rank(pct=True, method="average"))
            .astype(float)
            .fillna(0.5)
        )

    days_values = pd.to_numeric(eligible["days_since_d0"], errors="coerce")
    eligible["days_since_d0_le1"] = (days_values <= 1).fillna(False).astype(int)
    eligible["days_since_d0_eq2"] = (days_values == 2).fillna(False).astype(int)
    eligible["days_since_d0_ge3"] = (days_values >= 3).fillna(False).astype(int)

    interaction_columns: list[str] = []
    for interaction_name, inputs, kind in BASE_INTERACTION_SPECS:
        left, right = inputs
        if left not in eligible.columns or right not in eligible.columns:
            continue
        if kind == "product":
            eligible[interaction_name] = eligible[left].astype(float) * eligible[right].astype(float)
        elif kind == "spread":
            eligible[interaction_name] = eligible[left].astype(float) - eligible[right].astype(float)
        else:
            raise RuntimeError(f"unsupported v004a interaction kind: {kind}")
        interaction_columns.append(interaction_name)

    v004a_feature_columns = [
        *[rank_column for _, rank_column in BASE_RANK_SPECS],
        *[rank_column for _, rank_column in optional_specs],
        *interaction_columns,
        *DAY_BUCKET_COLUMNS,
    ]
    for column in v004a_feature_columns:
        eligible[column] = pd.to_numeric(eligible.get(column, 0.0), errors="coerce").fillna(0.0).astype(float)

    eligible[DEFAULT_TARGET_COLUMN] = False
    eligible[DEFAULT_HIGH_RETURN_COLUMN] = np.nan
    eligible[DEFAULT_CLOSE_RETURN_COLUMN] = np.nan
    eligible["realized_return_pct"] = np.nan
    eligible["outcome_labels_available"] = False
    eligible["target_metric_kind"] = TARGET_METRIC_KIND
    eligible["realized_return_kind"] = REALIZED_RETURN_KIND

    feature_info = {
        "v004a_feature_columns": v004a_feature_columns,
        "feature_set": ",".join(v004a_feature_columns),
        "interaction_set": ",".join(interaction_columns) if interaction_columns else "none",
        "interaction_columns": interaction_columns,
        "missing_optional_columns": [],
    }
    return eligible.sort_values(["signal_date", "code"]).reset_index(drop=True), feature_info


def score_live_candidates(
    live_features: pd.DataFrame,
    beta: np.ndarray,
    feature_columns: list[str],
    feature_info: dict[str, Any],
    v004a_l2: float,
    v004a_positive_weight: float,
    ranking_model_file: str | Path = DEFAULT_RANKING_MODEL_FILE,
) -> pd.DataFrame:
    manual_models = _load_manual_models_safe()
    ranking_model, ranking_meta = load_ranking_model(ranking_model_file)
    validate_ranking_model(ranking_model, live_features.columns)
    manual_models.pop(V002_MODEL_ID, None)
    manual_models[str(ranking_meta["model_id"])] = ranking_model
    scored_frames = _score_manual_and_hand_models(live_features, manual_models)
    scored_frames.append(
        _score_logistic_frame(
            live_features,
            beta=beta,
            feature_columns=feature_columns,
            model_id=MODEL_ID_V004A,
            l2=float(v004a_l2),
            positive_weight=float(v004a_positive_weight),
            feature_set=",".join(feature_columns),
            interaction_set=feature_info.get("interaction_set", "none"),
        )
    )
    scored = add_scored_model_rank(pd.concat(scored_frames, ignore_index=True))
    output = build_scored_candidates_output(scored, feature_info)
    for column, value in (
        (DEFAULT_TARGET_COLUMN, False),
        (DEFAULT_HIGH_RETURN_COLUMN, np.nan),
        (DEFAULT_CLOSE_RETURN_COLUMN, np.nan),
        ("realized_return_pct", np.nan),
    ):
        if column not in output.columns:
            output[column] = value
    output.attrs["v002_source_model_id"] = str(ranking_meta["model_id"])
    return output


def _load_manual_models_safe() -> dict[str, dict[str, Any]]:
    # Import lazily to keep daily selector startup simple and avoid exposing v004a internals at module import time.
    from .v004a import _load_manual_models

    return _load_manual_models()


def build_runtime_policy_meta(
    coefficients_file: Path = DEFAULT_COEFFICIENTS_FILE,
    coefficient_meta: dict[str, Any] | None = None,
    grid_id: int = DEFAULT_GRID_ID,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
    ranking_model_file: Path = DEFAULT_RANKING_MODEL_FILE,
    ranking_model_id: str | None = None,
    target_return_pct: float = DEFAULT_POLICY.target_return_pct,
    min_forward_dates: int = DEFAULT_POLICY.min_forward_dates,
) -> dict[str, Any]:
    coefficient_meta = coefficient_meta or {
        "coefficient_predict_date": DEFAULT_POLICY.coefficient_predict_date,
        "coefficient_train_end": DEFAULT_POLICY.coefficient_train_end,
    }
    coefficients_path = Path(coefficients_file)
    ranking_model_path = Path(ranking_model_file)
    if ranking_model_id is None:
        _, ranking_meta = load_ranking_model(ranking_model_path)
        ranking_model_id = str(ranking_meta["model_id"])
    checks = frozen_policy_input_checks(
        DEFAULT_POLICY,
        coefficients_file=coefficients_path,
        coefficient_predict_date=str(coefficient_meta.get("coefficient_predict_date", "")),
        coefficient_train_end=str(coefficient_meta.get("coefficient_train_end", "")),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        ranking_model_file=ranking_model_path,
        ranking_model_id=str(ranking_model_id),
        target_column=DEFAULT_POLICY.target_column,
        target_return_pct=float(target_return_pct),
        grid_id=int(grid_id),
        candidate_top_k=int(candidate_top_k),
        top_n=int(top_n),
        min_forward_dates=int(min_forward_dates),
    )
    matches = all(checks.values())
    coefficients_hash = normalized_sha256(coefficients_path) if coefficients_path.is_file() else ""
    ranking_model_hash = normalized_sha256(ranking_model_path) if ranking_model_path.is_file() else ""
    return {
        "policy_version": DEFAULT_POLICY.policy_version if matches else f"{DEFAULT_POLICY.policy_version}+custom_override",
        "policy_manifest": str(DEFAULT_POLICY.manifest_path),
        "deployment_status": DEFAULT_POLICY.deployment_status if matches else "custom_research_only",
        "research_only": True,
        "metric_scope": DEFAULT_POLICY.metric_scope,
        "manual_review_required": True,
        "transaction_costs_included": False,
        "liquidity_execution_validated": False,
        "frozen_policy_inputs_verified": bool(matches),
        "matches_frozen_manifest": bool(matches),
        "policy_manifest_normalized_sha256": DEFAULT_POLICY.manifest_sha256,
        "coefficients_file": str(coefficients_path),
        "coefficients_normalized_sha256": coefficients_hash,
        "coefficient_predict_date": str(coefficient_meta.get("coefficient_predict_date", "")),
        "coefficient_train_end": str(coefficient_meta.get("coefficient_train_end", "")),
        "v004a_l2": float(v004a_l2),
        "v004a_positive_weight": float(v004a_positive_weight),
        "ranking_model_path": str(ranking_model_path),
        "ranking_model_normalized_sha256": ranking_model_hash,
        "ranking_model_id": str(ranking_model_id),
        "v002_source_model_id": str(ranking_model_id),
        "observed_ranking_model_ids": str(ranking_model_id),
        "target_return_pct": float(target_return_pct),
        "grid_id": int(grid_id),
        "candidate_top_k": int(candidate_top_k),
        "top_n": int(top_n),
        "min_forward_dates": int(min_forward_dates),
        "failed_frozen_policy_checks": ",".join(key for key, passed in checks.items() if not passed),
    }


def build_daily_policy_outputs(
    selected_combos: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    top_n: int,
    policy_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_meta = policy_meta or build_runtime_policy_meta()
    decision_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    for _, combo in selected_combos.sort_values("signal_date").iterrows():
        date = str(combo["signal_date"])
        day_pool = candidate_pool[candidate_pool["signal_date"].astype(str) == date].copy()
        baseline_codes = parse_codes(combo["codes"])
        v002_codes = day_pool.sort_values(["v002_model_rank", "code"]).head(int(top_n))["code"].astype(str).tolist()
        v004a_codes = day_pool.sort_values(["v004a_model_rank", "code"]).head(int(top_n))["code"].astype(str).tolist()
        v005_ctx = ctx_for_codes(day_pool, baseline_codes)
        v002_top = ctx_for_codes(day_pool, v002_codes)
        risk = v005_ctx.apply(is_risk_ticket, axis=1) if not v005_ctx.empty else pd.Series(dtype=bool)
        gate_base = pd.Series(
            {
                "v005_avg_v002_rank": mean(v005_ctx["v002_model_rank"]) if not v005_ctx.empty else np.nan,
                "v005_has_risk_ticket": bool(risk.any()) if len(risk) else False,
                "v002_extreme_vwap_count": int(v002_top["extreme_vwap"].astype(bool).sum()) if not v002_top.empty else 0,
                "v002_extreme_close_low_count": int(v002_top["extreme_close_low"].astype(bool).sum()) if not v002_top.empty else 0,
            }
        )
        triggered = bool(is_policy_fallback(gate_base))
        action = "fallback_to_v002_regime_policy" if triggered else "keep_v005_fixed_grid"
        primary_codes = v002_codes if triggered else baseline_codes
        grid_id = int(combo["grid_id"])

        decision_rows.append(
            {
                "signal_date": date,
                "policy_version": policy_meta["policy_version"],
                "deployment_status": policy_meta["deployment_status"],
                "research_only": True,
                "metric_scope": policy_meta["metric_scope"],
                "manual_review_required": True,
                "transaction_costs_included": False,
                "liquidity_execution_validated": False,
                "frozen_policy_inputs_verified": bool(policy_meta["frozen_policy_inputs_verified"]),
                "matches_frozen_manifest": bool(policy_meta["matches_frozen_manifest"]),
                "v002_source_model_id": policy_meta["v002_source_model_id"],
                "final_strategy": STRATEGY_PRIMARY,
                "action": action,
                "fallback_triggered": triggered,
                "research_watchlist_codes": ",".join(norm(primary_codes)),
                "primary_buy_codes": ",".join(norm(primary_codes)),
                "v005_baseline_codes": ",".join(norm(baseline_codes)),
                "v002_codes": ",".join(norm(v002_codes)),
                "v004a_codes": ",".join(norm(v004a_codes)),
                "selected_grid_id": grid_id,
                "gate_v002_extreme_vwap_count": int(gate_base["v002_extreme_vwap_count"]),
                "gate_v002_extreme_close_low_count": int(gate_base["v002_extreme_close_low_count"]),
                "gate_v005_avg_v002_rank": float(gate_base["v005_avg_v002_rank"]),
                "gate_v005_has_risk_ticket": bool(gate_base["v005_has_risk_ticket"]),
            }
        )
        selection_frames.append(
            selection_rows(
                day_pool,
                codes=primary_codes,
                signal_date=date,
                strategy=STRATEGY_PRIMARY,
                role="primary_buy",
                action=action,
                triggered=triggered,
                grid_id=grid_id,
                display_order=1,
                is_primary=True,
                policy_meta=policy_meta,
            )
        )
        selection_frames.append(
            selection_rows(day_pool, baseline_codes, date, STRATEGY_BASELINE, "baseline_control", "control_v005_baseline", False, grid_id, 2, False, policy_meta)
        )
        selection_frames.append(
            selection_rows(day_pool, v002_codes, date, STRATEGY_V002, "fallback_source_control", "control_v002_top3", False, grid_id, 3, False, policy_meta)
        )
        selection_frames.append(
            selection_rows(day_pool, v004a_codes, date, STRATEGY_V004A, "model_control", "control_v004a_top3", False, grid_id, 4, False, policy_meta)
        )
    decisions = pd.DataFrame(decision_rows)
    if decisions.empty:
        decisions = pd.DataFrame(columns=DECISION_COLUMNS)
    else:
        decisions = decisions[[column for column in DECISION_COLUMNS if column in decisions.columns]]
    selections = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame(columns=SELECTION_COLUMNS)
    if not selections.empty:
        selections = selections[[column for column in SELECTION_COLUMNS if column in selections.columns]].sort_values(
            ["signal_date", "display_order", "daily_rank", "code"]
        )
    return decisions, selections


def selection_rows(
    day_pool: pd.DataFrame,
    codes: list[str],
    signal_date: str,
    strategy: str,
    role: str,
    action: str,
    triggered: bool,
    grid_id: int,
    display_order: int,
    is_primary: bool,
    policy_meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    policy_meta = policy_meta or build_runtime_policy_meta()
    selected = ctx_for_codes(day_pool, codes).copy()
    if selected.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    selected["policy_version"] = policy_meta["policy_version"]
    selected["deployment_status"] = policy_meta["deployment_status"]
    selected["research_only"] = True
    selected["selection_intent"] = "research_watchlist" if is_primary else "research_control"
    selected["manual_review_required"] = True
    selected["transaction_costs_included"] = False
    selected["liquidity_execution_validated"] = False
    selected["frozen_policy_inputs_verified"] = bool(policy_meta["frozen_policy_inputs_verified"])
    selected["v002_source_model_id"] = policy_meta["v002_source_model_id"]
    selected["strategy_role"] = role
    selected["strategy"] = strategy
    selected["is_primary_buy"] = bool(is_primary)
    selected["buy_priority"] = np.arange(1, len(selected) + 1) if is_primary else pd.NA
    selected["display_order"] = int(display_order)
    selected["signal_date"] = str(signal_date)
    selected["action"] = action
    selected["fallback_triggered"] = bool(triggered)
    selected["selected_grid_id"] = int(grid_id)
    selected["daily_rank"] = np.arange(1, len(selected) + 1)
    return selected[[column for column in SELECTION_COLUMNS if column in selected.columns]]


def build_default_grid() -> pd.DataFrame:
    return build_rule_grid(
        min_total_rank_weight_grid=DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
        avg_total_rank_weight_grid=DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
        contains_v004a_top3_bonus_grid=DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
        contains_v002_top3_bonus_grid=DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
        extreme_vwap_penalty_grid=DEFAULT_EXTREME_VWAP_PENALTY_GRID,
        extreme_close_low_penalty_grid=DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
        extreme_price_penalty_grid=DEFAULT_EXTREME_PRICE_PENALTY_GRID,
        rank_dispersion_weight_grid=DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    )


def select_grid_params(grid: pd.DataFrame, grid_id: int) -> pd.Series:
    row = grid[pd.to_numeric(grid["grid_id"], errors="coerce") == int(grid_id)].copy()
    if row.empty:
        raise RuntimeError(f"grid_id={grid_id} not found in default v005 grid")
    return row.iloc[0]


def build_daily_report(
    signal_date: str,
    output_dir: Path,
    scored_path: Path,
    coefficients_file: Path,
    coefficient_meta: dict[str, Any],
    grid_id: int,
    grid_params: pd.Series,
    decisions: pd.DataFrame,
    selections: pd.DataFrame,
    policy_meta: dict[str, Any] | None = None,
) -> str:
    policy_meta = policy_meta or build_runtime_policy_meta()
    lines = [
        "# v005 daily research watchlist",
        "",
        "## Safety status",
        "",
        "**RESEARCH ONLY / SHADOW FLOW. This is not an order list or evidence of deployable profitability. Manual review is required.**",
        "",
        "This daily inference does not use future target or realized-return labels. The historical target is an opportunity proxy based on D2 open to D3 high, not executable net PnL.",
        "Transaction costs, slippage, queue priority, one-price limit-up liquidity, suspensions, and announcements are not validated by this output.",
        "The legacy `primary_buy_*` CSV fields are retained for compatibility; interpret them only as a research watchlist.",
        "",
        "## Configuration",
        "",
        f"- signal_date: `{signal_date}`",
        f"- policy_version: `{policy_meta['policy_version']}`",
        f"- deployment_status: `{policy_meta['deployment_status']}`",
        f"- matches_frozen_manifest: `{policy_meta['matches_frozen_manifest']}`",
        f"- frozen_policy_inputs_verified: `{policy_meta['frozen_policy_inputs_verified']}`",
        f"- policy_manifest: `{policy_meta['policy_manifest']}`",
        f"- metric_scope: `{policy_meta['metric_scope']}`",
        f"- ranking_model_normalized_sha256: `{policy_meta['ranking_model_normalized_sha256']}`",
        f"- v002_source_model_id: `{policy_meta['v002_source_model_id']}`",
        f"- observed_ranking_model_ids: `{policy_meta['observed_ranking_model_ids']}`",
        f"- output_dir: `{output_dir}`",
        f"- scored_candidates: `{scored_path}`",
        f"- coefficients_file: `{coefficients_file}`",
        f"- coefficients_normalized_sha256: `{policy_meta['coefficients_normalized_sha256']}`",
        f"- coefficient_predict_date: `{coefficient_meta.get('coefficient_predict_date', '')}`",
        f"- fixed grid_id: `{grid_id}`",
        "",
        "## Research watchlist",
        "",
    ]
    primary = selections[selections["is_primary_buy"].fillna(False).astype(bool)].copy() if not selections.empty else pd.DataFrame()
    if primary.empty:
        lines.append("_No research watchlist rows._")
    else:
        lines.extend(md_table(primary, ["buy_priority", "code", "name", "strategy", "action", "v004a_model_rank", "v002_model_rank", "v004a_score", "v002_score"]))
    lines.extend(["", "## Decision", ""])
    lines.extend(md_table(decisions, DECISION_COLUMNS))
    lines.extend(["", "## Controls", ""])
    controls = selections[~selections["is_primary_buy"].fillna(False).astype(bool)].copy() if not selections.empty else pd.DataFrame()
    lines.extend(md_table(controls, ["strategy_role", "strategy", "daily_rank", "code", "name", "v004a_model_rank", "v002_model_rank", "v004a_score", "v002_score"]))
    lines.extend(["", "## Fixed grid params", ""])
    grid_df = pd.DataFrame([{column: grid_params[column] for column in GRID_PARAM_COLUMNS if column in grid_params.index}])
    lines.extend(md_table(grid_df, GRID_PARAM_COLUMNS))
    return "\n".join(lines)


def require_single_signal_date(frame: pd.DataFrame) -> str:
    if "trade_date" in frame.columns:
        date_column = "trade_date"
    elif "signal_date" in frame.columns:
        date_column = "signal_date"
    else:
        dates: list[str] = []
        raise RuntimeError(f"v005 daily selector requires exactly one signal_date; found {dates}")
    dates = sorted(frame[date_column].dropna().astype(str).unique().tolist())
    if len(dates) != 1:
        raise RuntimeError(f"v005 daily selector requires exactly one signal_date; found {dates}")
    return dates[0]


def latest_trade_date(pool: pd.DataFrame) -> str:
    if pool is None or pool.empty or "trade_date" not in pool.columns:
        return pd.Timestamp.now().strftime("%Y-%m-%d")
    return str(pool["trade_date"].dropna().astype(str).max())


def bool_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        return value.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})
    return pd.Series(bool(value))


def mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    return np.nan if series.empty else float(series.mean())


def md_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df is None or df.empty:
        return ["_No rows._"]
    use = [column for column in columns if column in df.columns]
    rows = ["| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for _, row in df[use].iterrows():
        rows.append("| " + " | ".join(fmt(row[column]) for column in use) + " |")
    return rows


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed-grid v005 daily selector shadow flow.")
    parser.add_argument("--signals-file", default=None, help="Existing daily signals CSV. If omitted, collect limit-ups and build signals first.")
    parser.add_argument("--date", default=None, help="Signal date for market collection; default today/latest available.")
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--days", type=int, default=None, help="Bars days passed to signal generation.")
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--ranking-model", default=str(DEFAULT_RANKING_MODEL_FILE))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--coefficients-file", default=str(DEFAULT_COEFFICIENTS_FILE))
    parser.add_argument("--coefficient-predict-date", default=DEFAULT_COEFFICIENT_PREDICT_DATE)
    parser.add_argument("--grid-id", type=int, default=DEFAULT_GRID_ID)
    args = parser.parse_args(argv)

    if args.signals_file:
        signals = pd.read_csv(args.signals_file, dtype={"code": str})
        signal_date = require_single_signal_date(signals_to_frame(signals))
        output_dir = Path(args.output_root) / signal_date
        decisions, selections, combos, scored, report = run_v005_daily_selector(
            signals,
            output_dir=output_dir,
            coefficients_file=args.coefficients_file,
            grid_id=args.grid_id,
            coefficient_predict_date=args.coefficient_predict_date,
            top_n=args.top_n,
            candidate_top_k=args.candidate_top_k,
            ranking_model_file=args.ranking_model,
            source_signals_file=args.signals_file,
        )
        print(f"signals file: {args.signals_file}")
        print(f"decision rows: {len(decisions)}")
        print(f"selection rows: {len(selections)}")
        print(f"combo rows: {len(combos)}")
        print(f"scored rows: {len(scored)}")
        print("status: RESEARCH ONLY / SHADOW FLOW; manual review required")
        print(f"markdown: {report}")
        return 0

    decisions, selections, combos, scored, report, signal_csv, quality_csv = run_v005_daily_from_market(
        date=args.date,
        lookback_days=args.lookback_days,
        days=args.days,
        max_codes=args.max_codes,
        force_refresh=args.force_refresh,
        workers=args.workers,
        output_root=args.output_root,
        ranking_model=args.ranking_model,
        top_n=args.top_n,
        candidate_top_k=args.candidate_top_k,
        coefficients_file=args.coefficients_file,
        coefficient_predict_date=args.coefficient_predict_date,
        grid_id=args.grid_id,
    )
    print(f"decision rows: {len(decisions)}")
    print(f"selection rows: {len(selections)}")
    print(f"combo rows: {len(combos)}")
    print(f"scored rows: {len(scored)}")
    print("status: RESEARCH ONLY / SHADOW FLOW; manual review required")
    print(f"v2 signals csv: {signal_csv}")
    print(f"quality csv: {quality_csv}")
    print(f"markdown: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

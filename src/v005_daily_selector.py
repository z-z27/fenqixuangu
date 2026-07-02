from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtester import _candidate_base_price, build_signals_for_pool
from .config import get_data_config
from .daily_ranking import DEFAULT_DAILY_RANKING_MODEL, DEFAULT_DAILY_TOP_N, apply_daily_research_ranking
from .loaders import MarketDataService
from .report import write_data_quality_reports, write_signal_reports
from .signal_engine import Signal
from .v004a import (
    BASE_INTERACTION_SPECS,
    BASE_RANK_SPECS,
    DAY_BUCKET_COLUMNS,
    DEFAULT_CLOSE_RETURN_COLUMN,
    DEFAULT_HIGH_RETURN_COLUMN,
    DEFAULT_TARGET_COLUMN,
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
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
    DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
    DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
    DEFAULT_EXTREME_PRICE_PENALTY_GRID,
    DEFAULT_EXTREME_VWAP_PENALTY_GRID,
    DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    DEFAULT_TOP_N,
    DEFAULT_V004A_L2,
    DEFAULT_V004A_POSITIVE_WEIGHT,
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

DEFAULT_COEFFICIENTS_FILE = Path("reports/v004a/grid_v2_scored/v004a_coefficients.csv")
DEFAULT_OUTPUT_ROOT = Path("reports/daily_v005")
DEFAULT_GRID_ID = 4

STRATEGY_PRIMARY = PRIMARY_POLICY
STRATEGY_BASELINE = "baseline_v005_fixed_grid"
STRATEGY_V002 = "v002_top3_control"
STRATEGY_V004A = "v004a_top3_control"

SELECTION_COLUMNS = [
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
    "final_strategy",
    "action",
    "fallback_triggered",
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
    coefficient_predict_date: str = "2026-06-26",
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    signal_frame = signals_to_frame(signals)
    if signal_frame.empty:
        raise RuntimeError("no signals for v005 daily selector")
    signal_date = infer_signal_date(signal_frame)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_ROOT / signal_date
    out_dir.mkdir(parents=True, exist_ok=True)

    live_features, feature_info = prepare_live_v004a_features(signal_frame)
    beta, feature_columns, coefficient_meta = load_fixed_v004a_beta(
        coefficients_file=Path(coefficients_file),
        coefficient_predict_date=str(coefficient_predict_date),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
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
    )
    scored_path = out_dir / f"v005_daily_scored_candidates_{signal_date}.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    prepared_scored = prepare_scored_candidates(scored_path)

    candidate_pool = build_candidate_pool(
        prepared_scored,
        candidate_top_k=int(candidate_top_k),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    combo_candidates = build_combo_candidates(candidate_pool, top_n=int(top_n))
    grid = build_default_grid()
    fixed_params = select_grid_params(grid, grid_id=int(grid_id))
    selected_combos = select_best_combo_by_date(score_combos(combo_candidates, fixed_params))
    baseline_top3 = explode_selected_combos(selected_combos, candidate_pool, top_n=int(top_n))

    decisions, selections = build_daily_policy_outputs(
        selected_combos=selected_combos,
        candidate_pool=candidate_pool,
        top_n=int(top_n),
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
        ),
        encoding="utf-8",
    )
    return decisions, selections, selected_combos, scored, report_path


def run_v005_daily_from_market(
    date: str | None = None,
    lookback_days: int = 5,
    days: int | None = None,
    max_codes: int | None = None,
    force_refresh: bool = False,
    workers: int = 6,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ranking_model: str | Path = DEFAULT_DAILY_RANKING_MODEL,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    coefficients_file: str | Path = DEFAULT_COEFFICIENTS_FILE,
    coefficient_predict_date: str = "2026-06-26",
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
    )
    meta = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "limitup_rows": int(len(pool)),
                "signals": int(len(ranked_signals)),
                "quality_rows": int(len(quality_rows)),
                "ranking_model_id": ranking_meta.get("model_id", ""),
                "ranking_model_path": ranking_meta.get("model_path", ""),
                "v2_signals_csv": str(signal_csv),
                "v2_signals_md": str(signal_md),
                "quality_csv": str(quality_csv),
                "quality_md": str(quality_md),
                "v005_report": str(report_path),
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
) -> pd.DataFrame:
    scored_frames = _score_manual_and_hand_models(live_features, _load_manual_models_safe())
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
    return output


def _load_manual_models_safe() -> dict[str, dict[str, Any]]:
    # Import lazily to keep daily selector startup simple and avoid exposing v004a internals at module import time.
    from .v004a import _load_manual_models

    return _load_manual_models()


def build_daily_policy_outputs(selected_combos: pd.DataFrame, candidate_pool: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "final_strategy": STRATEGY_PRIMARY,
                "action": action,
                "fallback_triggered": triggered,
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
            )
        )
        selection_frames.append(
            selection_rows(day_pool, baseline_codes, date, STRATEGY_BASELINE, "baseline_control", "control_v005_baseline", False, grid_id, 2, False)
        )
        selection_frames.append(
            selection_rows(day_pool, v002_codes, date, STRATEGY_V002, "fallback_source_control", "control_v002_top3", False, grid_id, 3, False)
        )
        selection_frames.append(
            selection_rows(day_pool, v004a_codes, date, STRATEGY_V004A, "model_control", "control_v004a_top3", False, grid_id, 4, False)
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
) -> pd.DataFrame:
    selected = ctx_for_codes(day_pool, codes).copy()
    if selected.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
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
) -> str:
    lines = [
        "# v005 daily selector",
        "",
        "## Scope",
        "",
        "Daily shadow-flow output for the fixed-grid v005 policy. It does not use future target or realized-return labels.",
        "The primary buy list is always `policy_v005_v002_regime_fallback`; all other rows are controls.",
        "",
        "## Configuration",
        "",
        f"- signal_date: `{signal_date}`",
        f"- output_dir: `{output_dir}`",
        f"- scored_candidates: `{scored_path}`",
        f"- coefficients_file: `{coefficients_file}`",
        f"- coefficient_predict_date: `{coefficient_meta.get('coefficient_predict_date', '')}`",
        f"- fixed grid_id: `{grid_id}`",
        "",
        "## Primary buy list",
        "",
    ]
    primary = selections[selections["is_primary_buy"].fillna(False).astype(bool)].copy() if not selections.empty else pd.DataFrame()
    if primary.empty:
        lines.append("_No primary buy rows._")
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


def infer_signal_date(frame: pd.DataFrame) -> str:
    if "trade_date" in frame.columns and frame["trade_date"].notna().any():
        return str(frame["trade_date"].dropna().astype(str).max())
    if "signal_date" in frame.columns and frame["signal_date"].notna().any():
        return str(frame["signal_date"].dropna().astype(str).max())
    return pd.Timestamp.now().strftime("%Y-%m-%d")


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
    parser.add_argument("--ranking-model", default=str(DEFAULT_DAILY_RANKING_MODEL))
    parser.add_argument("--top-n", type=int, default=DEFAULT_DAILY_TOP_N)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--coefficients-file", default=str(DEFAULT_COEFFICIENTS_FILE))
    parser.add_argument("--coefficient-predict-date", default="2026-06-26")
    parser.add_argument("--grid-id", type=int, default=DEFAULT_GRID_ID)
    args = parser.parse_args(argv)

    if args.signals_file:
        signals = pd.read_csv(args.signals_file, dtype={"code": str})
        signal_date = infer_signal_date(signals)
        output_dir = Path(args.output_root) / signal_date
        decisions, selections, combos, scored, report = run_v005_daily_selector(
            signals,
            output_dir=output_dir,
            coefficients_file=args.coefficients_file,
            grid_id=args.grid_id,
            coefficient_predict_date=args.coefficient_predict_date,
            top_n=args.top_n,
            candidate_top_k=args.candidate_top_k,
        )
        print(f"signals file: {args.signals_file}")
        print(f"decision rows: {len(decisions)}")
        print(f"selection rows: {len(selections)}")
        print(f"combo rows: {len(combos)}")
        print(f"scored rows: {len(scored)}")
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
    print(f"v2 signals csv: {signal_csv}")
    print(f"quality csv: {quality_csv}")
    print(f"markdown: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

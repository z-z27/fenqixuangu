"""Broad-market-strength information audit for the frozen v004c S2 ranker.

The primary May-July conclusion is built first from the frozen S2 score and the
lineage-audited U0 market context.  Only after that conclusion is fixed does
the module open the archived August OOT population for a post-hoc auxiliary
description.  The module never fits a model or changes a score/rank.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_s2_winner_vs_loss_failure_attribution import load_frozen_s2


TASK_NAME = "v004c_broad_market_context_information_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
PRIMARY_START = "2026-05-01"
PRIMARY_END = "2026-07-31"
AUGUST_START = "2026-08-01"
AUGUST_END = "2026-08-31"

CONTEXT_ARTIFACT = (
    "reports/research/v004c_broad_market_context_lineage_sensitivity_audit_v001/"
    "v004c_context_lineage_daily_values_v001.csv"
)
AUGUST_POPULATION_ARTIFACT = (
    "reports/research/v004c_regime_aware_stage1_august_oot_v001/"
    "v004c_august_oot_population_v001.csv"
)

PRIMARY_CONTEXT_VARIABLE = "market_up_ratio"
CONTEXT_VARIABLES = (
    "market_up_ratio",
    "market_median_return",
    "market_tail_balance",
)
FIXED_STOCK_SIGNALS = (
    "rank_theme_score",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
)
MARKET_STATES = ("WEAK_MARKET", "STRONG_MARKET")
MONTHS = ("2026-05", "2026-06", "2026-07")

# Interpretation gates are fixed in source before the data are evaluated.
ENVIRONMENT_RHO_MATERIAL = 0.20
RATE_DIFFERENCE_MATERIAL = 0.05
RETURN_DIFFERENCE_MATERIAL = 0.005
ALPHA_RHO_MATERIAL = 0.20
ALPHA_DIFFERENCE_MATERIAL = 0.005
POSITIVE_ALPHA_RATE_DIFFERENCE_MATERIAL = 0.10
PAIR_TEMPORAL_RANGE_REDUCTION_MATERIAL = 0.05
ALPHA_TEMPORAL_RANGE_REDUCTION_MATERIAL = 0.005
MIN_PRIMARY_DATES = 45

OUTPUT_FILENAMES = (
    "v004c_market_context_candidate_environment_v001.csv",
    "v004c_market_context_s2_daily_performance_v001.csv",
    "v004c_market_context_selection_alpha_v001.csv",
    "v004c_market_context_rankwise_v001.csv",
    "v004c_market_context_s2_pair_separation_v001.csv",
    "v004c_market_context_fixed_signal_pairs_v001.csv",
    "v004c_market_context_month_conditioned_v001.csv",
    "v004c_market_context_robustness_m2_m3_v001.csv",
    "v004c_market_context_august_auxiliary_v001.csv",
    "v004c_broad_market_context_information_review_v001.md",
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator or not np.isfinite(denominator):
        return math.nan
    return float(numerator / denominator)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def _direction(value: float, neutral: float = 0.0) -> str:
    if not np.isfinite(value):
        return "NA"
    if value > neutral:
        return "POSITIVE"
    if value < neutral:
        return "NEGATIVE"
    return "TIE"


def _market_state(value: float, threshold: float) -> str:
    return "WEAK_MARKET" if float(value) <= float(threshold) else "STRONG_MARKET"


def load_context(root: str | Path) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    path = Path(root).resolve() / CONTEXT_ARTIFACT
    frame = pd.read_csv(path, dtype={"trade_date": str}, encoding="utf-8-sig")
    required = {
        "trade_date", "analysis_scope", "universe_variant", *CONTEXT_VARIABLES,
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("FATAL: lineage-audited market context schema mismatch")
    u0 = frame[
        frame["universe_variant"].eq("U0_CURRENT_SNAPSHOT_REFERENCE")
    ].copy()
    if u0["trade_date"].duplicated().any():
        raise RuntimeError("FATAL: duplicate U0 context date")
    primary = u0[
        u0["analysis_scope"].eq("PRIMARY_MAY_JULY")
        & u0["trade_date"].between(PRIMARY_START, PRIMARY_END)
    ].copy()
    auxiliary = u0[
        u0["analysis_scope"].eq("AUGUST_AUXILIARY")
        & u0["trade_date"].between(AUGUST_START, AUGUST_END)
    ].copy()
    if len(primary) != 62 or len(auxiliary) != 13:
        raise RuntimeError(
            f"FATAL: context date parity failed: primary={len(primary)}, august={len(auxiliary)}"
        )
    if primary[list(CONTEXT_VARIABLES)].isna().any().any():
        raise RuntimeError("FATAL: primary context contains missing values")
    threshold = float(primary[PRIMARY_CONTEXT_VARIABLE].median())
    primary["market_state"] = primary[PRIMARY_CONTEXT_VARIABLE].map(
        lambda value: _market_state(float(value), threshold)
    )
    auxiliary["market_state"] = auxiliary[PRIMARY_CONTEXT_VARIABLE].map(
        lambda value: _market_state(float(value), threshold)
    )
    primary = primary.rename(columns={"trade_date": "signal_date"})
    auxiliary = auxiliary.rename(columns={"trade_date": "signal_date"})
    return (
        pd.concat([primary, auxiliary], ignore_index=True),
        threshold,
        {
            "context_artifact": CONTEXT_ARTIFACT,
            "primary_context_dates": len(primary),
            "august_context_dates": len(auxiliary),
            "market_up_ratio_fixed_median": threshold,
            "threshold_source": "ALL_62_MAY_JULY_FULL_QUALITY_DATES",
        },
    )


def load_primary_population(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[str, Any]]:
    """Load frozen S2 and attach U0 context; no model fitting occurs."""
    scored, _, s2_audit = load_frozen_s2(root)
    context, threshold, context_audit = load_context(root)
    primary_context = context[
        context["analysis_scope"].eq("PRIMARY_MAY_JULY")
    ].copy()
    merged = scored.merge(
        primary_context[
            ["signal_date", *CONTEXT_VARIABLES, "market_state", "universe_count"]
        ],
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    if merged[list(CONTEXT_VARIABLES) + ["market_state"]].isna().any().any():
        missing = sorted(
            merged.loc[merged["market_state"].isna(), "signal_date"].astype(str).unique()
        )
        raise RuntimeError(f"FATAL: candidate dates missing market context: {missing}")
    if (len(merged), merged["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: frozen primary candidate parity failed")
    if merged["signal_date"].ge(AUGUST_START).any():
        raise RuntimeError("FATAL: August signal-date row reached primary analysis")
    if merged["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate primary event_id")
    audit = {
        **context_audit,
        **{f"s2_{key}": value for key, value in s2_audit.items()},
        "primary_candidate_rows": len(merged),
        "primary_candidate_dates": merged["signal_date"].nunique(),
        "model_trained": "NO",
        "new_context_variable_added": "NO",
    }
    return merged.sort_values(["signal_date", "s2_rank", "event_id"]), context, threshold, audit


def _daily_performance(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in population.groupby("signal_date", sort=True):
        day = day.sort_values(["s2_rank", "event_id"], kind="mergesort")
        top3 = day[day["s2_rank"].le(3)].copy()
        total_target7 = float(day["target7"].sum())
        top3_target7 = float(top3["target7"].sum())
        row: dict[str, Any] = {
            "signal_date": str(signal_date),
            "month": str(signal_date)[:7],
            "market_state": day["market_state"].iloc[0],
            "market_up_ratio_fixed_median": math.nan,
            "market_up_ratio": float(day["market_up_ratio"].iloc[0]),
            "market_median_return": float(day["market_median_return"].iloc[0]),
            "market_tail_balance": float(day["market_tail_balance"].iloc[0]),
            "candidate_count": len(day),
            "board2_count": int(day["board_group"].eq("BOARD2").sum()),
            "board3_count": int(day["board_group"].eq("BOARD3").sum()),
            "candidate_target7_count": int(day["target7"].sum()),
            "candidate_target7_rate": float(day["target7"].mean()),
            "candidate_loss_count": int(day["loss"].sum()),
            "candidate_loss_rate": float(day["loss"].mean()),
            "candidate_severe_loss_count": int(day["severe_loss"].sum()),
            "candidate_severe_loss_rate": float(day["severe_loss"].mean()),
            "candidate_capped_return": float(day["capped_return_7"].mean()),
            "top3_rows": len(top3),
            "top3_target7_count": int(top3["target7"].sum()),
            "top3_target7_rate": float(top3["target7"].mean()),
            "top3_loss_count": int(top3["loss"].sum()),
            "top3_loss_rate": float(top3["loss"].mean()),
            "top3_severe_loss_count": int(top3["severe_loss"].sum()),
            "top3_severe_loss_rate": float(top3["severe_loss"].mean()),
            "top3_capped_return": float(top3["capped_return_7"].mean()),
            "daily_selection_alpha": float(
                top3["capped_return_7"].mean() - day["capped_return_7"].mean()
            ),
            "positive_selection_alpha": bool(
                top3["capped_return_7"].mean() > day["capped_return_7"].mean()
            ),
            "winner_capture": _safe_ratio(top3_target7, total_target7),
        }
        for rank in (1, 2, 3):
            selected = day[day["s2_rank"].eq(rank)]
            prefix = f"rank{rank}"
            row[f"{prefix}_present"] = bool(len(selected))
            for metric in ("target7", "loss", "severe_loss", "capped_return_7"):
                out = "capped_return" if metric == "capped_return_7" else f"{metric}_rate"
                row[f"{prefix}_{out}"] = (
                    float(selected[metric].mean()) if len(selected) else math.nan
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_date", kind="mergesort").reset_index(drop=True)


ENVIRONMENT_METRICS = (
    "candidate_target7_rate",
    "candidate_loss_rate",
    "candidate_severe_loss_rate",
    "candidate_capped_return",
)


def build_candidate_environment(daily: pd.DataFrame, threshold: float) -> pd.DataFrame:
    daily_rows = daily[
        [
            "signal_date", "month", "market_state", *CONTEXT_VARIABLES,
            "candidate_count", "board2_count", "board3_count",
            "candidate_target7_count", "candidate_target7_rate",
            "candidate_loss_count", "candidate_loss_rate",
            "candidate_severe_loss_count", "candidate_severe_loss_rate",
            "candidate_capped_return",
        ]
    ].copy()
    daily_rows.insert(0, "record_type", "DAILY")
    daily_rows["market_up_ratio_fixed_median"] = threshold
    summary_rows: list[dict[str, Any]] = []
    for state in MARKET_STATES:
        part = daily[daily["market_state"].eq(state)]
        row: dict[str, Any] = {
            "record_type": "STATE_SUMMARY",
            "market_state": state,
            "state_date_count": len(part),
            "market_up_ratio_fixed_median": threshold,
        }
        for metric in ENVIRONMENT_METRICS:
            row[f"{metric}_state_mean"] = float(part[metric].mean())
            row[f"{metric}_state_median"] = float(part[metric].median())
        summary_rows.append(row)
    weak = daily[daily["market_state"].eq("WEAK_MARKET")]
    strong = daily[daily["market_state"].eq("STRONG_MARKET")]
    contrast: dict[str, Any] = {
        "record_type": "STRONG_MINUS_WEAK",
        "market_state": "STRONG_MINUS_WEAK",
        "state_date_count": len(daily),
        "market_up_ratio_fixed_median": threshold,
    }
    for metric in ENVIRONMENT_METRICS:
        contrast[f"{metric}_state_mean"] = float(strong[metric].mean() - weak[metric].mean())
        contrast[f"{metric}_state_median"] = float(strong[metric].median() - weak[metric].median())
    return pd.concat([daily_rows, pd.DataFrame(summary_rows + [contrast])], ignore_index=True)


def _pair_daily(
    population: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for signal_date, day in population.groupby("signal_date", sort=True):
        target = day[day["target7"].eq(1)]
        loss = day[day["loss"].eq(1)]
        daily_values: list[float] = []
        for t_row in target.itertuples(index=False):
            for l_row in loss.itertuples(index=False):
                t_value = float(getattr(t_row, value_column))
                l_value = float(getattr(l_row, value_column))
                concordance = 1.0 if t_value > l_value else 0.0 if t_value < l_value else 0.5
                daily_values.append(concordance)
                pair_rows.append({
                    "signal_date": str(signal_date),
                    "month": str(signal_date)[:7],
                    "market_state": day["market_state"].iloc[0],
                    "value_column": value_column,
                    "target7_event_id": str(t_row.event_id),
                    "loss_event_id": str(l_row.event_id),
                    "target7_value": t_value,
                    "loss_value": l_value,
                    "target7_minus_loss": t_value - l_value,
                    "t7_high_concordance": concordance,
                })
        if daily_values:
            daily_rows.append({
                "signal_date": str(signal_date),
                "month": str(signal_date)[:7],
                "market_state": day["market_state"].iloc[0],
                "value_column": value_column,
                "pair_count": len(daily_values),
                "daily_pair_concordance": float(np.mean(daily_values)),
            })
    return pd.DataFrame(pair_rows), pd.DataFrame(daily_rows)


def _pair_summary(
    pairs: pd.DataFrame,
    daily_pairs: pd.DataFrame,
    period: str,
    state: str,
) -> dict[str, Any]:
    mask = pd.Series(True, index=pairs.index)
    daily_mask = pd.Series(True, index=daily_pairs.index)
    if period != "ALL_MAY_JULY":
        month = {"MAY": "2026-05", "JUNE": "2026-06", "JULY": "2026-07"}[period]
        mask &= pairs["month"].eq(month)
        daily_mask &= daily_pairs["month"].eq(month)
    if state != "ALL_MARKET":
        mask &= pairs["market_state"].eq(state)
        daily_mask &= daily_pairs["market_state"].eq(state)
    part = pairs[mask]
    days = daily_pairs[daily_mask]
    return {
        "period": period,
        "market_state": state,
        "date_count": int(days["signal_date"].nunique()) if len(days) else 0,
        "pair_count": len(part),
        "within_date_pair_concordance": (
            float(days["daily_pair_concordance"].mean()) if len(days) else math.nan
        ),
        "pooled_pair_concordance": (
            float(part["t7_high_concordance"].mean()) if len(part) else math.nan
        ),
        "mean_target7_value": float(part["target7_value"].mean()) if len(part) else math.nan,
        "mean_loss_value": float(part["loss_value"].mean()) if len(part) else math.nan,
        "median_target7_value": float(part["target7_value"].median()) if len(part) else math.nan,
        "median_loss_value": float(part["loss_value"].median()) if len(part) else math.nan,
    }


def build_pair_tables(population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    pair_map: dict[str, pd.DataFrame] = {}
    daily_map: dict[str, pd.DataFrame] = {}
    all_columns = ("s2_score", *FIXED_STOCK_SIGNALS)
    for column in all_columns:
        pair_map[column], daily_map[column] = _pair_daily(population, column)
    periods = ("ALL_MAY_JULY", "MAY", "JUNE", "JULY")
    states = ("ALL_MARKET", *MARKET_STATES)
    s2_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for period in periods:
        for state in states:
            s2_rows.append(_pair_summary(pair_map["s2_score"], daily_map["s2_score"], period, state))
            for signal in FIXED_STOCK_SIGNALS:
                row = _pair_summary(pair_map[signal], daily_map[signal], period, state)
                row["stock_signal"] = signal
                signal_rows.append(row)
    return pd.DataFrame(s2_rows), pd.DataFrame(signal_rows), daily_map


def build_rankwise(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in ("ALL_MARKET", *MARKET_STATES):
        part = population if state == "ALL_MARKET" else population[population["market_state"].eq(state)]
        for rank in (1, 2, 3):
            selected = part[part["s2_rank"].eq(rank)]
            rows.append({
                "market_state": state,
                "rank": rank,
                "date_count": selected["signal_date"].nunique(),
                "rows": len(selected),
                "target7_count": int(selected["target7"].sum()),
                "target7_rate": float(selected["target7"].mean()) if len(selected) else math.nan,
                "loss_count": int(selected["loss"].sum()),
                "loss_rate": float(selected["loss"].mean()) if len(selected) else math.nan,
                "severe_loss_count": int(selected["severe_loss"].sum()),
                "severe_loss_rate": float(selected["severe_loss"].mean()) if len(selected) else math.nan,
                "capped_return": float(selected["capped_return_7"].mean()) if len(selected) else math.nan,
            })
    return pd.DataFrame(rows)


def build_selection_alpha(daily: pd.DataFrame, threshold: float) -> pd.DataFrame:
    daily_rows = daily[
        [
            "signal_date", "month", "market_state", *CONTEXT_VARIABLES,
            "candidate_capped_return", "top3_capped_return",
            "daily_selection_alpha", "positive_selection_alpha",
        ]
    ].copy()
    daily_rows.insert(0, "record_type", "DAILY")
    daily_rows["market_up_ratio_fixed_median"] = threshold
    rows: list[dict[str, Any]] = []
    for state in MARKET_STATES:
        part = daily[daily["market_state"].eq(state)]
        rows.append({
            "record_type": "STATE_SUMMARY",
            "market_state": state,
            "state_date_count": len(part),
            "daily_selection_alpha_state_mean": float(part["daily_selection_alpha"].mean()),
            "daily_selection_alpha_state_median": float(part["daily_selection_alpha"].median()),
            "positive_selection_alpha_date_rate": float(part["positive_selection_alpha"].mean()),
            "market_up_ratio_fixed_median": threshold,
        })
    weak = daily[daily["market_state"].eq("WEAK_MARKET")]
    strong = daily[daily["market_state"].eq("STRONG_MARKET")]
    rows.append({
        "record_type": "STRONG_MINUS_WEAK",
        "market_state": "STRONG_MINUS_WEAK",
        "state_date_count": len(daily),
        "daily_selection_alpha_state_mean": float(
            strong["daily_selection_alpha"].mean() - weak["daily_selection_alpha"].mean()
        ),
        "daily_selection_alpha_state_median": float(
            strong["daily_selection_alpha"].median() - weak["daily_selection_alpha"].median()
        ),
        "positive_selection_alpha_date_rate": float(
            strong["positive_selection_alpha"].mean() - weak["positive_selection_alpha"].mean()
        ),
        "market_up_ratio_fixed_median": threshold,
    })
    for variable in CONTEXT_VARIABLES:
        rows.append({
            "record_type": "CONTINUOUS_CORRELATION",
            "context_variable": variable,
            "state_date_count": len(daily),
            "context_vs_selection_alpha_spearman": _spearman(
                daily[variable], daily["daily_selection_alpha"]
            ),
            "market_up_ratio_fixed_median": threshold,
        })
    return pd.concat([daily_rows, pd.DataFrame(rows)], ignore_index=True)


ROBUSTNESS_METRICS = (
    "candidate_target7_rate",
    "candidate_loss_rate",
    "candidate_capped_return",
    "top3_target7_rate",
    "top3_loss_rate",
    "top3_capped_return",
    "daily_selection_alpha",
    "winner_capture",
)


def build_robustness(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    m1_directions: dict[str, str] = {}
    for variable in CONTEXT_VARIABLES:
        for metric in ROBUSTNESS_METRICS:
            value = _spearman(daily[variable], daily[metric])
            if variable == PRIMARY_CONTEXT_VARIABLE:
                m1_directions[metric] = _direction(value)
            rows.append({
                "record_type": "CONTINUOUS_SPEARMAN",
                "context_variable": variable,
                "outcome_metric": metric,
                "date_count": int(daily[[variable, metric]].dropna().shape[0]),
                "spearman_rho": value,
                "direction": _direction(value),
            })
    frame = pd.DataFrame(rows)
    for metric in ROBUSTNESS_METRICS:
        mask = frame["outcome_metric"].eq(metric)
        frame.loc[mask, "m1_direction"] = m1_directions[metric]
        frame.loc[mask, "direction_matches_m1"] = frame.loc[mask, "direction"].eq(
            m1_directions[metric]
        )
    checks = frame[
        frame["context_variable"].isin(["market_median_return", "market_tail_balance"])
    ]
    match_rate = float(checks["direction_matches_m1"].mean())
    summary = pd.DataFrame([{
        "record_type": "ROBUSTNESS_SUMMARY",
        "context_variable": "M2_M3_VS_M1",
        "outcome_metric": "ALL_FIXED_METRICS",
        "date_count": len(daily),
        "direction_match_rate": match_rate,
        "broad_market_strength_result_robust": bool(match_rate >= .75),
        "robustness_label": (
            "BROAD_MARKET_STRENGTH_RESULT_ROBUST"
            if match_rate >= .75 else "VARIABLE_SPECIFIC_RESULT"
        ),
    }])
    return pd.concat([frame, summary], ignore_index=True)


def build_month_conditioned(
    daily: pd.DataFrame,
    s2_pairs: pd.DataFrame,
    signal_pairs: pd.DataFrame,
    primary_context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month in MONTHS:
        context_month = primary_context[
            primary_context["signal_date"].astype(str).str[:7].eq(month)
        ]
        rows.append({
            "record_type": "CONTEXT_MONTH_DISTRIBUTION",
            "month": month,
            "market_state": "ALL_CONTEXT_DATES",
            "context_date_count": len(context_month),
            "weak_context_date_count": int(context_month["market_state"].eq("WEAK_MARKET").sum()),
            "strong_context_date_count": int(context_month["market_state"].eq("STRONG_MARKET").sum()),
            "market_up_ratio_mean": float(context_month["market_up_ratio"].mean()),
            "market_up_ratio_median": float(context_month["market_up_ratio"].median()),
            "market_up_ratio_p25": float(context_month["market_up_ratio"].quantile(.25)),
            "market_up_ratio_p75": float(context_month["market_up_ratio"].quantile(.75)),
            "market_up_ratio_min": float(context_month["market_up_ratio"].min()),
            "market_up_ratio_max": float(context_month["market_up_ratio"].max()),
        })
    for month in MONTHS:
        month_name = {"2026-05": "MAY", "2026-06": "JUNE", "2026-07": "JULY"}[month]
        for state in ("ALL_MARKET", *MARKET_STATES):
            part = daily[daily["month"].eq(month)]
            if state != "ALL_MARKET":
                part = part[part["market_state"].eq(state)]
            s2 = s2_pairs[
                s2_pairs["period"].eq(month_name)
                & s2_pairs["market_state"].eq(state)
            ]
            row: dict[str, Any] = {
                "record_type": "MONTH_STATE",
                "month": month,
                "market_state": state,
                "date_count": len(part),
                "market_up_ratio_mean": float(part["market_up_ratio"].mean()) if len(part) else math.nan,
                "market_up_ratio_median": float(part["market_up_ratio"].median()) if len(part) else math.nan,
                "selection_alpha_mean": float(part["daily_selection_alpha"].mean()) if len(part) else math.nan,
                "selection_alpha_median": float(part["daily_selection_alpha"].median()) if len(part) else math.nan,
                "positive_alpha_date_rate": float(part["positive_selection_alpha"].mean()) if len(part) else math.nan,
                "s2_pair_count": int(s2["pair_count"].iloc[0]) if len(s2) else 0,
                "s2_pair_concordance": float(s2["within_date_pair_concordance"].iloc[0]) if len(s2) else math.nan,
            }
            for signal in FIXED_STOCK_SIGNALS:
                value = signal_pairs[
                    signal_pairs["period"].eq(month_name)
                    & signal_pairs["market_state"].eq(state)
                    & signal_pairs["stock_signal"].eq(signal)
                ]
                row[f"{signal}_pair_count"] = int(value["pair_count"].iloc[0]) if len(value) else 0
                row[f"{signal}_concordance"] = (
                    float(value["within_date_pair_concordance"].iloc[0]) if len(value) else math.nan
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    stability_rows: list[dict[str, Any]] = []
    metrics = (
        "selection_alpha_mean",
        "s2_pair_concordance",
        *(f"{signal}_concordance" for signal in FIXED_STOCK_SIGNALS),
    )
    improved_count = 0
    for metric in metrics:
        all_values = frame[frame["market_state"].eq("ALL_MARKET")][metric].dropna()
        weak_values = frame[frame["market_state"].eq("WEAK_MARKET")][metric].dropna()
        strong_values = frame[frame["market_state"].eq("STRONG_MARKET")][metric].dropna()
        baseline_range = float(all_values.max() - all_values.min()) if len(all_values) >= 2 else math.nan
        weak_range = float(weak_values.max() - weak_values.min()) if len(weak_values) >= 2 else math.nan
        strong_range = float(strong_values.max() - strong_values.min()) if len(strong_values) >= 2 else math.nan
        usable = [value for value in (weak_range, strong_range) if np.isfinite(value)]
        conditioned_range = float(np.mean(usable)) if usable else math.nan
        reduction = baseline_range - conditioned_range if np.isfinite(conditioned_range) else math.nan
        material = (
            ALPHA_TEMPORAL_RANGE_REDUCTION_MATERIAL
            if metric == "selection_alpha_mean" else PAIR_TEMPORAL_RANGE_REDUCTION_MATERIAL
        )
        improved = bool(np.isfinite(reduction) and reduction >= material)
        improved_count += int(improved)
        stability_rows.append({
            "record_type": "TEMPORAL_STABILITY",
            "temporal_metric": metric,
            "unconditioned_month_range": baseline_range,
            "weak_state_month_range": weak_range,
            "strong_state_month_range": strong_range,
            "mean_conditioned_month_range": conditioned_range,
            "month_range_reduction": reduction,
            "material_reduction_threshold": material,
            "conditioning_improves_temporal_stability": improved,
        })
    stability = pd.DataFrame(stability_rows)
    combined = pd.concat([frame, stability], ignore_index=True)
    return combined, {
        "temporal_metric_count": len(metrics),
        "temporal_metrics_improved": improved_count,
        "market_state_temporal_explanation_supported": bool(improved_count >= 2),
    }


def _monthly_direction_breadth(daily: pd.DataFrame, metric: str) -> tuple[int, str]:
    directions: list[str] = []
    for month in MONTHS:
        part = daily[daily["month"].eq(month)]
        rho = _spearman(part[PRIMARY_CONTEXT_VARIABLE], part[metric])
        directions.append(_direction(rho))
    usable = [value for value in directions if value in {"POSITIVE", "NEGATIVE"}]
    if not usable:
        return 0, "NA"
    counts = {direction: usable.count(direction) for direction in set(usable)}
    dominant = sorted(counts, key=lambda key: (-counts[key], key))[0]
    return counts[dominant], dominant


def decide_primary_state(
    daily: pd.DataFrame,
    robustness: pd.DataFrame,
    temporal_audit: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    weak = daily[daily["market_state"].eq("WEAK_MARKET")]
    strong = daily[daily["market_state"].eq("STRONG_MARKET")]
    correlations = {
        metric: _spearman(daily[PRIMARY_CONTEXT_VARIABLE], daily[metric])
        for metric in (
            "candidate_target7_rate", "candidate_loss_rate", "candidate_capped_return",
            "daily_selection_alpha",
        )
    }
    differences = {
        metric: float(strong[metric].mean() - weak[metric].mean())
        for metric in (
            "candidate_target7_rate", "candidate_loss_rate", "candidate_capped_return",
            "daily_selection_alpha", "positive_selection_alpha",
        )
    }
    expected_environment_directions = (
        correlations["candidate_target7_rate"] > 0,
        correlations["candidate_loss_rate"] < 0,
        correlations["candidate_capped_return"] > 0,
        differences["candidate_target7_rate"] > 0,
        differences["candidate_loss_rate"] < 0,
        differences["candidate_capped_return"] > 0,
    )
    material_environment_effects = (
        abs(correlations["candidate_target7_rate"]) >= ENVIRONMENT_RHO_MATERIAL,
        abs(correlations["candidate_loss_rate"]) >= ENVIRONMENT_RHO_MATERIAL,
        abs(correlations["candidate_capped_return"]) >= ENVIRONMENT_RHO_MATERIAL,
        abs(differences["candidate_target7_rate"]) >= RATE_DIFFERENCE_MATERIAL,
        abs(differences["candidate_loss_rate"]) >= RATE_DIFFERENCE_MATERIAL,
        abs(differences["candidate_capped_return"]) >= RETURN_DIFFERENCE_MATERIAL,
    )
    environment_supported = bool(
        sum(expected_environment_directions) >= 5
        and sum(material_environment_effects) >= 4
    )
    state_split_material = (
        abs(differences["candidate_target7_rate"]) >= RATE_DIFFERENCE_MATERIAL,
        abs(differences["candidate_loss_rate"]) >= RATE_DIFFERENCE_MATERIAL,
        abs(differences["candidate_capped_return"]) >= RETURN_DIFFERENCE_MATERIAL,
    )
    # A directional fixed-split result without a material continuous relation
    # is partial environment evidence, matching the preregistered semantic
    # category "only explains candidate environment".
    environment_partial = bool(
        sum(expected_environment_directions) >= 5
        and sum(state_split_material) >= 2
    )
    alpha_rho = correlations["daily_selection_alpha"]
    alpha_difference = differences["daily_selection_alpha"]
    alpha_direction_coherent = (
        (alpha_rho > 0 and alpha_difference > 0)
        or (alpha_rho < 0 and alpha_difference < 0)
    )
    alpha_supported = bool(
        alpha_direction_coherent
        and abs(alpha_rho) >= ALPHA_RHO_MATERIAL
        and abs(alpha_difference) >= ALPHA_DIFFERENCE_MATERIAL
        and abs(differences["positive_selection_alpha"]) >= POSITIVE_ALPHA_RATE_DIFFERENCE_MATERIAL
    )
    breadth: dict[str, tuple[int, str]] = {
        metric: _monthly_direction_breadth(daily, metric)
        for metric in (
            "candidate_target7_rate", "candidate_loss_rate", "candidate_capped_return",
            "daily_selection_alpha",
        )
    }
    not_single_month = bool(
        sum(count >= 2 for count, _ in breadth.values()) >= 3
    )
    robust_row = robustness[robustness["record_type"].eq("ROBUSTNESS_SUMMARY")].iloc[0]
    strength_robust = bool(robust_row["broad_market_strength_result_robust"])
    temporal_supported = bool(temporal_audit["market_state_temporal_explanation_supported"])
    if len(daily) < MIN_PRIMARY_DATES:
        state = "BROAD_MARKET_CONTEXT_INCONCLUSIVE"
        next_action = "STOP_AND_REVIEW"
    elif (
        environment_supported
        and strength_robust
        and (alpha_supported or temporal_supported)
        and not_single_month
    ):
        state = "BROAD_MARKET_CONTEXT_INFORMATION_SUPPORTED"
        next_action = "SINGLE_CONTEXT_CONFIRMATION_DESIGN"
    elif environment_partial or alpha_supported or temporal_supported:
        state = "BROAD_MARKET_CONTEXT_PARTIALLY_SUPPORTED"
        next_action = "STOP_AND_REVIEW"
    else:
        state = "BROAD_MARKET_CONTEXT_NOT_SUPPORTED"
        next_action = "STOP_BROAD_MARKET_CONTEXT_LINE"
    evidence = {
        **{f"rho_{key}": value for key, value in correlations.items()},
        **{f"strong_minus_weak_{key}": value for key, value in differences.items()},
        "environment_supported": environment_supported,
        "candidate_environment_partial_evidence": environment_partial,
        "selection_alpha_market_dependent": alpha_supported,
        "market_state_temporal_explanation_supported": temporal_supported,
        "not_single_month": not_single_month,
        "m2_m3_direction_robust": strength_robust,
        "monthly_direction_breadth": breadth,
    }
    return state, next_action, evidence


def load_august_auxiliary(
    root: str | Path,
    context: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(root).resolve() / AUGUST_POPULATION_ARTIFACT
    population = pd.read_csv(
        path, dtype={"event_id": str, "code": str, "signal_date": str}, encoding="utf-8-sig"
    )
    required = {
        "event_id", "signal_date", "board_group", "baseline_score", "baseline_rank",
        "target7", "positive_non_target", "loss", "severe_loss", "capped_return_7",
        "mature_at_evaluation_asof", "eligible_for_oot_evaluation", *FIXED_STOCK_SIGNALS,
    }
    if not required.issubset(population.columns):
        raise RuntimeError("FATAL: archived August baseline population schema mismatch")
    for column in ("mature_at_evaluation_asof", "eligible_for_oot_evaluation"):
        population[column] = population[column].astype(str).str.lower().eq("true")
    auxiliary_context = context[context["analysis_scope"].eq("AUGUST_AUXILIARY")].copy()
    context_dates = set(auxiliary_context["signal_date"].astype(str))
    eligible = population[
        population["mature_at_evaluation_asof"]
        & population["eligible_for_oot_evaluation"]
        & population["signal_date"].isin(context_dates)
    ].copy()
    eligible = eligible.rename(columns={"baseline_score": "s2_score", "baseline_rank": "s2_rank"})
    eligible = eligible.merge(
        auxiliary_context[
            ["signal_date", *CONTEXT_VARIABLES, "market_state", "universe_count"]
        ],
        on="signal_date", how="left", validate="many_to_one",
    )
    if len(eligible) and eligible[list(CONTEXT_VARIABLES) + ["market_state"]].isna().any().any():
        raise RuntimeError("FATAL: August auxiliary context merge failed")
    if eligible["signal_date"].lt(AUGUST_START).any():
        raise RuntimeError("FATAL: non-August row reached August auxiliary analysis")
    if eligible["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate August auxiliary event_id")
    return eligible.sort_values(["signal_date", "s2_rank", "event_id"]), {
        "august_role": "POST_HOC_AUXILIARY_ONLY",
        "august_context_dates": len(context_dates),
        "august_candidate_dates_with_mature_outcomes": eligible["signal_date"].nunique(),
        "august_candidate_rows_with_mature_outcomes": len(eligible),
        "fixed_primary_threshold_reused": threshold,
        "august_used_for_primary_state": "NO",
    }


def build_august_output(
    population: pd.DataFrame,
    context: pd.DataFrame,
    threshold: float,
    primary_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    auxiliary_context = context[context["analysis_scope"].eq("AUGUST_AUXILIARY")]
    for row in auxiliary_context.sort_values("signal_date").itertuples(index=False):
        rows.append({
            "section": "CONTEXT_DATE",
            "signal_date": row.signal_date,
            "market_state": row.market_state,
            "market_up_ratio_fixed_median": threshold,
            "market_up_ratio": row.market_up_ratio,
            "market_median_return": row.market_median_return,
            "market_tail_balance": row.market_tail_balance,
            "august_role": "POST_HOC_AUXILIARY_ONLY",
        })
    if population.empty:
        return pd.DataFrame(rows), {"august_direction_summary": "NO_MATURE_CANDIDATE_OVERLAP"}
    daily = _daily_performance(population)
    for row in daily.itertuples(index=False):
        values = row._asdict()
        values.update({"section": "DAILY_PERFORMANCE", "august_role": "POST_HOC_AUXILIARY_ONLY"})
        rows.append(values)
    for state in MARKET_STATES:
        part = daily[daily["market_state"].eq(state)]
        rows.append({
            "section": "STATE_SUMMARY",
            "market_state": state,
            "date_count": len(part),
            "candidate_target7_rate": float(part["candidate_target7_rate"].mean()) if len(part) else math.nan,
            "candidate_loss_rate": float(part["candidate_loss_rate"].mean()) if len(part) else math.nan,
            "candidate_capped_return": float(part["candidate_capped_return"].mean()) if len(part) else math.nan,
            "top3_capped_return": float(part["top3_capped_return"].mean()) if len(part) else math.nan,
            "daily_selection_alpha": float(part["daily_selection_alpha"].mean()) if len(part) else math.nan,
            "top3_target7_rate": float(part["top3_target7_rate"].mean()) if len(part) else math.nan,
            "top3_loss_rate": float(part["top3_loss_rate"].mean()) if len(part) else math.nan,
            "august_role": "POST_HOC_AUXILIARY_ONLY",
        })
    august_s2, august_signals, _ = build_pair_tables(population)
    for row in august_s2[
        august_s2["period"].eq("ALL_MAY_JULY")
        & august_s2["market_state"].isin(MARKET_STATES)
    ].itertuples(index=False):
        values = row._asdict()
        values.update({"section": "S2_PAIR_SEPARATION", "august_role": "POST_HOC_AUXILIARY_ONLY"})
        rows.append(values)
    for row in august_signals[
        august_signals["period"].eq("ALL_MAY_JULY")
        & august_signals["market_state"].isin(MARKET_STATES)
    ].itertuples(index=False):
        values = row._asdict()
        values.update({"section": "FIXED_SIGNAL_PAIR", "august_role": "POST_HOC_AUXILIARY_ONLY"})
        rows.append(values)
    comparison_metrics = (
        "candidate_target7_rate", "candidate_loss_rate", "candidate_capped_return",
        "daily_selection_alpha",
    )
    direction_matches: list[bool] = []
    for metric in comparison_metrics:
        primary_rho = _spearman(primary_daily[PRIMARY_CONTEXT_VARIABLE], primary_daily[metric])
        august_rho = _spearman(daily[PRIMARY_CONTEXT_VARIABLE], daily[metric])
        match = _direction(primary_rho) == _direction(august_rho)
        direction_matches.append(match)
        rows.append({
            "section": "DIRECTION_CHECK",
            "context_variable": PRIMARY_CONTEXT_VARIABLE,
            "outcome_metric": metric,
            "primary_spearman": primary_rho,
            "august_spearman": august_rho,
            "direction_consistent": match,
            "august_role": "POST_HOC_AUXILIARY_ONLY",
        })
    summary = {
        "august_direction_checks": len(direction_matches),
        "august_direction_matches": sum(direction_matches),
        "august_direction_match_rate": float(np.mean(direction_matches)),
        "august_direction_summary": (
            "BROADLY_SAME_DIRECTION" if np.mean(direction_matches) >= .75
            else "MIXED_DIRECTION" if np.mean(direction_matches) >= .50
            else "MOSTLY_DIFFERENT_DIRECTION"
        ),
    }
    rows.append({
        "section": "AUXILIARY_SUMMARY",
        "august_role": "POST_HOC_AUXILIARY_ONLY",
        **summary,
    })
    return pd.DataFrame(rows), summary


def render_review(context: Mapping[str, Any]) -> str:
    evidence = context["evidence"]
    daily = context["daily"]
    rankwise = context["rankwise"]
    s2_pairs = context["s2_pairs"]
    signal_pairs = context["signal_pairs"]
    weak = daily[daily["market_state"].eq("WEAK_MARKET")]
    strong = daily[daily["market_state"].eq("STRONG_MARKET")]
    rank_lines = []
    for rank in (1, 2, 3):
        low = rankwise[(rankwise["market_state"].eq("WEAK_MARKET")) & rankwise["rank"].eq(rank)].iloc[0]
        high = rankwise[(rankwise["market_state"].eq("STRONG_MARKET")) & rankwise["rank"].eq(rank)].iloc[0]
        rank_lines.append(
            f"Rank{rank}: WEAK T7/LOSS/capped={low.target7_rate:.2%}/{low.loss_rate:.2%}/{low.capped_return:.2%}; "
            f"STRONG={high.target7_rate:.2%}/{high.loss_rate:.2%}/{high.capped_return:.2%}."
        )
    s2_low = s2_pairs[
        s2_pairs["period"].eq("ALL_MAY_JULY") & s2_pairs["market_state"].eq("WEAK_MARKET")
    ].iloc[0]
    s2_high = s2_pairs[
        s2_pairs["period"].eq("ALL_MAY_JULY") & s2_pairs["market_state"].eq("STRONG_MARKET")
    ].iloc[0]
    signal_lines = []
    for signal in FIXED_STOCK_SIGNALS:
        low = signal_pairs[
            signal_pairs["period"].eq("ALL_MAY_JULY")
            & signal_pairs["market_state"].eq("WEAK_MARKET")
            & signal_pairs["stock_signal"].eq(signal)
        ].iloc[0]
        high = signal_pairs[
            signal_pairs["period"].eq("ALL_MAY_JULY")
            & signal_pairs["market_state"].eq("STRONG_MARKET")
            & signal_pairs["stock_signal"].eq(signal)
        ].iloc[0]
        signal_lines.append(
            f"{signal}: WEAK={low.within_date_pair_concordance:.3f}, "
            f"STRONG={high.within_date_pair_concordance:.3f}."
        )
    robustness_row = context["robustness"][
        context["robustness"]["record_type"].eq("ROBUSTNESS_SUMMARY")
    ].iloc[0]
    august_summary = context["august_summary"]
    state = context["state"]
    next_action = context["next_action"]
    return "\n".join([
        "# v004c Broad Market Context Information Audit v001", "",
        "本审计只使用固定 M1/M2/M3、冻结 S2 分数与三项预注册 stock signals；模型拟合次数为 0。主结论先在 May-July 冻结，August 仅后验辅助描述。", "",
        "## Q1. 市场整体强时，v004c candidate 是否更容易修复？", "",
        f"WEAK vs STRONG 的 candidate T7 为 {weak.candidate_target7_rate.mean():.2%} vs {strong.candidate_target7_rate.mean():.2%}，"
        f"LOSS 为 {weak.candidate_loss_rate.mean():.2%} vs {strong.candidate_loss_rate.mean():.2%}，"
        f"capped 为 {weak.candidate_capped_return.mean():.2%} vs {strong.candidate_capped_return.mean():.2%}。"
        f"完整 continuous+split environment gate={'SUPPORTED' if evidence['environment_supported'] else 'NOT_SUPPORTED'}；"
        f"fixed-split partial evidence={'YES' if evidence['candidate_environment_partial_evidence'] else 'NO'}。", "",
        "## Q2. 市场强弱是否能解释 S2 什么时候有效、什么时候失效？", "",
        f"Selection alpha WEAK/STRONG 均值为 {weak.daily_selection_alpha.mean():.2%}/{strong.daily_selection_alpha.mean():.2%}；"
        f"M1 与 alpha Spearman={evidence['rho_daily_selection_alpha']:.3f}。"
        f"预注册 alpha-dependence gate={'SUPPORTED' if evidence['selection_alpha_market_dependent'] else 'NOT_SUPPORTED'}。", "",
        "## Q3. Rank2 / Rank3 confusion 是否集中在某种市场环境？", "",
        *[f"- {line}" for line in rank_lines], "",
        "## Q4. S2 Target7-vs-LOSS separation 在强/弱市场中是否不同？", "",
        f"WEAK={s2_low.within_date_pair_concordance:.3f} ({int(s2_low.pair_count)} pairs)，"
        f"STRONG={s2_high.within_date_pair_concordance:.3f} ({int(s2_high.pair_count)} pairs)。", "",
        "## Q5. theme / active_money / close_vwap 的方向不稳定是否被市场状态解释？", "",
        *[f"- {line}" for line in signal_lines], "",
        "## Q6. 按 market state 条件化后，May / June / July 是否更一致？", "",
        f"{context['temporal_audit']['temporal_metrics_improved']}/{context['temporal_audit']['temporal_metric_count']} 个预注册 temporal metrics 达到固定的 range-reduction 门；"
        f"formal flag={'YES' if evidence['market_state_temporal_explanation_supported'] else 'NO'}。逐月/分状态数值见 month-conditioned 表。", "",
        "## Q7. M2/M3 是否支持 M1 的主要结论？", "",
        f"{robustness_row.robustness_label}；固定指标方向匹配率={robustness_row.direction_match_rate:.2%}。未按结果选择变量。", "",
        "## Q8. August 辅助结果是否同方向？", "",
        f"{august_summary['august_direction_summary']}；{august_summary.get('august_direction_matches', 0)}/{august_summary.get('august_direction_checks', 0)} 个固定方向检查一致。"
        "该结果不改变 May-July 状态与授权判断。", "",
        f"BROAD_MARKET_CONTEXT_INFORMATION_STATE = {state}", "",
        "PRIMARY_CONTEXT_VARIABLE = MARKET_UP_RATIO", "",
        "AUGUST_ROLE = POST_HOC_AUXILIARY_ONLY", "",
        "MODEL_TRAINED = NO", "",
        "NEW_CONTEXT_VARIABLE_ADDED = NO", "",
        f"NEXT_ACTION = {next_action}", "",
    ])


def analyze_primary(root: str | Path) -> dict[str, Any]:
    population, context, threshold, audit = load_primary_population(root)
    daily = _daily_performance(population)
    daily["market_up_ratio_fixed_median"] = threshold
    candidate_environment = build_candidate_environment(daily, threshold)
    rankwise = build_rankwise(population)
    selection_alpha = build_selection_alpha(daily, threshold)
    s2_pairs, signal_pairs, _ = build_pair_tables(population)
    robustness = build_robustness(daily)
    month_conditioned, temporal_audit = build_month_conditioned(
        daily,
        s2_pairs,
        signal_pairs,
        context[context["analysis_scope"].eq("PRIMARY_MAY_JULY")],
    )
    state, next_action, evidence = decide_primary_state(
        daily, robustness, temporal_audit
    )
    return {
        "root": Path(root).resolve(),
        "population": population,
        "context": context,
        "threshold": threshold,
        "audit": audit,
        "daily": daily,
        "candidate_environment": candidate_environment,
        "rankwise": rankwise,
        "selection_alpha": selection_alpha,
        "s2_pairs": s2_pairs,
        "signal_pairs": signal_pairs,
        "robustness": robustness,
        "month_conditioned": month_conditioned,
        "temporal_audit": temporal_audit,
        "state": state,
        "next_action": next_action,
        "evidence": evidence,
    }


def analyze(root: str | Path) -> dict[str, Any]:
    # Freeze all May-July outputs and the formal state before opening August.
    context = analyze_primary(root)
    frozen_state = str(context["state"])
    frozen_next_action = str(context["next_action"])
    august_population, august_audit = load_august_auxiliary(
        root, context["context"], float(context["threshold"])
    )
    august_output, august_summary = build_august_output(
        august_population,
        context["context"],
        float(context["threshold"]),
        context["daily"],
    )
    if context["state"] != frozen_state or context["next_action"] != frozen_next_action:
        raise RuntimeError("FATAL: August altered frozen May-July decision")
    context["august_population"] = august_population
    context["august_output"] = august_output
    context["august_audit"] = august_audit
    context["august_summary"] = august_summary
    context["review"] = render_review(context)
    return context


def build_outputs(context: Mapping[str, Any]) -> dict[str, bytes]:
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(context["candidate_environment"]),
        OUTPUT_FILENAMES[1]: _csv_bytes(context["daily"]),
        OUTPUT_FILENAMES[2]: _csv_bytes(context["selection_alpha"]),
        OUTPUT_FILENAMES[3]: _csv_bytes(context["rankwise"]),
        OUTPUT_FILENAMES[4]: _csv_bytes(context["s2_pairs"]),
        OUTPUT_FILENAMES[5]: _csv_bytes(context["signal_pairs"]),
        OUTPUT_FILENAMES[6]: _csv_bytes(context["month_conditioned"]),
        OUTPUT_FILENAMES[7]: _csv_bytes(context["robustness"]),
        OUTPUT_FILENAMES[8]: _csv_bytes(context["august_output"]),
        OUTPUT_FILENAMES[9]: str(context["review"]).encode("utf-8"),
    }
    if tuple(outputs) != OUTPUT_FILENAMES:
        raise RuntimeError("FATAL: output manifest mismatch")
    return outputs


def write_outputs(context: Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
    target = Path(output_dir) if output_dir is not None else (
        context["root"] / "reports" / "research" / OUTPUT_DIRNAME
    )
    first = build_outputs(context)
    second = build_outputs(context)
    if first != second:
        raise RuntimeError("FATAL: deterministic output build failed")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        temporary = target / f"{name}.tmp"
        temporary.write_bytes(payload)
        temporary.replace(target / name)
    return target


def output_sha256s(output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }

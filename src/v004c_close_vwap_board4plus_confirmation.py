"""Confirm one frozen v004c regime-conditional information relationship.

This module confirms only ``d1_close_to_vwap_raw x board4plus_count``.  It
reconstructs the relationship from the mature May-July population, keeps the
prior all-60-date median split fixed, and corrects the original 8 x 11
selection process with a within-date label permutation maximum statistic.
It never fits a model, changes a score/rank, searches a threshold, or accesses
an August signal-date outcome.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.v004c_regime_conditional_signal_audit import (
    ANALYSIS_END,
    AUGUST_ASOF,
    MONTHS,
    POPULATION_AUTHORITY,
    REGIME_AUTHORITY,
    REGIME_VARIABLES,
    STOCK_SIGNALS,
    build_regime_splits,
    load_inputs,
)


TASK_NAME = "v004c_close_vwap_board4plus_confirmation_v001"
OUTPUT_DIRNAME = "v004c_close_vwap_board4plus_confirmation_v001_202605_202607"

FIXED_SIGNAL = "d1_close_to_vwap_raw"
FIXED_REGIME = "board4plus_count"
FIXED_REGIME_SOURCE = "four_plus_board_limit_up_count"
FIXED_THRESHOLD_EXPECTED = 1.0

PRIOR_SUMMARY_AUTHORITY = (
    "reports/research/"
    "v004c_regime_conditional_signal_audit_v001_202605_202607/"
    "v004c_signal_regime_conditional_summary_v001.csv"
)
PRIOR_EXPECTED_LOW = 0.5910884353741496
PRIOR_EXPECTED_HIGH = 0.31497493734335835
PRIOR_EXPECTED_DELTA = -0.2761134980307912
REPRODUCTION_TOLERANCE = 1e-12

PERMUTATION_SEED = 20260828
PERMUTATION_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_RESAMPLES = 20_000

# Fixed confirmation gates.  These are declared in code, not selected from
# confirmation results.
CLEAR_LOW_MIN = 0.55
CLEAR_HIGH_MAX = 0.45
CLEAR_ABS_DELTA_MIN = 0.15
SELECTION_ADJUSTED_ALPHA = 0.05
BOOTSTRAP_DIRECTION_PROBABILITY_MIN = 0.95
RANK2_6_MIN_PAIRS_PER_STATE = 10
RANK2_6_MIN_DATES_PER_STATE = 3

OUTPUT_FILENAMES = (
    "v004c_close_vwap_board4plus_reproduction_v001.csv",
    "v004c_close_vwap_board4plus_monthly_v001.csv",
    "v004c_close_vwap_board4plus_lomo_v001.csv",
    "v004c_close_vwap_board4plus_rank2_6_v001.csv",
    "v004c_close_vwap_board4plus_rankwise_v001.csv",
    "v004c_close_vwap_board4plus_bootstrap_v001.csv",
    "v004c_close_vwap_board4plus_selection_permutation_v001.csv",
    "v004c_close_vwap_board4plus_leave_one_date_v001.csv",
    "v004c_close_vwap_board4plus_lineage_v001.csv",
    "v004c_close_vwap_board4plus_confirmation_review_v001.md",
)


def assert_contract() -> None:
    if FIXED_SIGNAL not in STOCK_SIGNALS:
        raise RuntimeError("FATAL: frozen stock signal not in prior eight-signal manifest")
    if FIXED_REGIME not in REGIME_VARIABLES:
        raise RuntimeError("FATAL: frozen regime not in prior eleven-regime manifest")
    if REGIME_VARIABLES[FIXED_REGIME][0] != FIXED_REGIME_SOURCE:
        raise RuntimeError("FATAL: frozen board4plus source mapping changed")
    if len(STOCK_SIGNALS) * len(REGIME_VARIABLES) != 88:
        raise RuntimeError("FATAL: original selection family is no longer exactly 88")
    if ANALYSIS_END >= AUGUST_ASOF:
        raise RuntimeError("FATAL: analysis boundary reached August")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _fixed_pair_table(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in population.groupby("signal_date", sort=True):
        targets = day[day["target7"].astype(bool)].sort_values("event_id", kind="mergesort")
        losses = day[day["loss"].astype(bool)].sort_values("event_id", kind="mergesort")
        for target in targets.itertuples(index=False):
            for loss in losses.itertuples(index=False):
                t_value = float(getattr(target, FIXED_SIGNAL))
                l_value = float(getattr(loss, FIXED_SIGNAL))
                difference = t_value - l_value
                rows.append({
                    "signal_date": str(signal_date),
                    "month": str(target.month),
                    "target_event_id": str(target.event_id),
                    "loss_event_id": str(loss.event_id),
                    "target_signal_value": t_value,
                    "loss_signal_value": l_value,
                    "signal_difference_t7_minus_loss": difference,
                    "t7_high_concordance": (
                        1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
                    ),
                })
    return pd.DataFrame(rows).sort_values(
        ["signal_date", "target_event_id", "loss_event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _pair_stats(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {
            "informative_date_count": 0,
            "pair_count": 0,
            "within_date_pair_concordance": math.nan,
            "pooled_pair_concordance": math.nan,
            "mean_pair_difference": math.nan,
            "median_pair_difference": math.nan,
        }
    daily = pairs.groupby("signal_date", sort=True)["t7_high_concordance"].mean()
    return {
        "informative_date_count": int(len(daily)),
        "pair_count": int(len(pairs)),
        "within_date_pair_concordance": float(daily.mean()),
        "pooled_pair_concordance": float(pairs["t7_high_concordance"].mean()),
        "mean_pair_difference": float(pairs["signal_difference_t7_minus_loss"].mean()),
        "median_pair_difference": float(pairs["signal_difference_t7_minus_loss"].median()),
    }


def _candidate_stats(frame: pd.DataFrame) -> dict[str, Any]:
    targets = frame[frame["target7"].astype(bool)][FIXED_SIGNAL]
    losses = frame[frame["loss"].astype(bool)][FIXED_SIGNAL]
    return {
        "target7_rows": int(len(targets)),
        "loss_rows": int(len(losses)),
        "mean_signal_target7": float(targets.mean()) if len(targets) else math.nan,
        "mean_signal_loss": float(losses.mean()) if len(losses) else math.nan,
        "median_signal_target7": float(targets.median()) if len(targets) else math.nan,
        "median_signal_loss": float(losses.median()) if len(losses) else math.nan,
    }


def _attach_fixed_state(
    population: pd.DataFrame, daily: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_long, split_info = build_regime_splits(daily)
    fixed = split_long[split_long["regime_variable"].eq(FIXED_REGIME)].copy()
    info = split_info[FIXED_REGIME]
    if abs(float(info["threshold"]) - FIXED_THRESHOLD_EXPECTED) > REPRODUCTION_TOLERANCE:
        raise RuntimeError("FATAL: frozen board4plus all-60-date median changed")
    state = fixed[["signal_date", "regime_value", "regime_state"]]
    enriched = population.merge(state, on="signal_date", validate="many_to_one")
    return enriched, fixed, info


def _pairs_with_state(population: pd.DataFrame, fixed_split: pd.DataFrame) -> pd.DataFrame:
    pairs = _fixed_pair_table(population)
    return pairs.merge(
        fixed_split[["signal_date", "regime_value", "regime_state"]],
        on="signal_date",
        validate="many_to_one",
    )


def build_reproduction(
    population: pd.DataFrame,
    fixed_split: pd.DataFrame,
    root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    pairs = _pairs_with_state(population, fixed_split)
    rows: list[dict[str, Any]] = []
    state_values: dict[str, float] = {}
    for state in ("LOW_REGIME", "HIGH_REGIME"):
        dates = fixed_split.loc[fixed_split["regime_state"].eq(state), "signal_date"]
        subset = population[population["signal_date"].isin(dates)]
        stats = _pair_stats(pairs[pairs["regime_state"].eq(state)])
        candidate = _candidate_stats(subset)
        state_values[state] = stats["within_date_pair_concordance"]
        rows.append({
            "stock_signal": FIXED_SIGNAL,
            "regime_variable": FIXED_REGIME,
            "fixed_split_rule": "LOW_LE_ALL_60_DATE_MEDIAN__HIGH_GT",
            "fixed_median": FIXED_THRESHOLD_EXPECTED,
            "regime_state": state,
            "regime_date_count": int(len(dates)),
            **candidate,
            **stats,
        })
    delta = state_values["HIGH_REGIME"] - state_values["LOW_REGIME"]
    for row in rows:
        row["delta_high_minus_low"] = delta

    prior = pd.read_csv(
        Path(root) / PRIOR_SUMMARY_AUTHORITY,
        encoding="utf-8-sig",
    )
    authority = prior[
        prior["stock_signal"].eq(FIXED_SIGNAL)
        & prior["regime_variable"].eq(FIXED_REGIME)
    ]
    if len(authority) != 1:
        raise RuntimeError("FATAL: prior frozen relationship authority missing or duplicated")
    authority_row = authority.iloc[0]
    expected = {
        "low": float(authority_row["low_within_date_pair_concordance"]),
        "high": float(authority_row["high_within_date_pair_concordance"]),
        "delta": float(authority_row["conditional_concordance_delta_high_minus_low"]),
    }
    max_error = max(
        abs(state_values["LOW_REGIME"] - expected["low"]),
        abs(state_values["HIGH_REGIME"] - expected["high"]),
        abs(delta - expected["delta"]),
        abs(expected["low"] - PRIOR_EXPECTED_LOW),
        abs(expected["high"] - PRIOR_EXPECTED_HIGH),
        abs(expected["delta"] - PRIOR_EXPECTED_DELTA),
    )
    passed = bool(max_error <= REPRODUCTION_TOLERANCE)
    for row in rows:
        row.update({
            "prior_low_concordance": expected["low"],
            "prior_high_concordance": expected["high"],
            "prior_delta_high_minus_low": expected["delta"],
            "reproduction_max_abs_error": max_error,
            "reproduction_state": "PASS" if passed else "REPRODUCTION_FAIL",
        })
    audit = {
        "reproduction_pass": passed,
        "reproduction_max_abs_error": max_error,
        "low_concordance": state_values["LOW_REGIME"],
        "high_concordance": state_values["HIGH_REGIME"],
        "delta": delta,
        "pair_count": len(pairs),
    }
    return pd.DataFrame(rows), audit, pairs


def build_monthly(
    population: pd.DataFrame, fixed_split: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month in MONTHS:
        for state in ("LOW_REGIME", "HIGH_REGIME"):
            dates = fixed_split.loc[
                fixed_split["month"].eq(month) & fixed_split["regime_state"].eq(state),
                "signal_date",
            ]
            subset = population[population["signal_date"].isin(dates)]
            pair_subset = pairs[
                pairs["month"].eq(month) & pairs["regime_state"].eq(state)
            ]
            rows.append({
                "month": month,
                "regime_state": state,
                "fixed_median": FIXED_THRESHOLD_EXPECTED,
                "state_signal_date_count": int(len(dates)),
                **_candidate_stats(subset),
                **_pair_stats(pair_subset),
            })
    return pd.DataFrame(rows)


def build_lomo(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    months = tuple(MONTHS)
    for omitted in months:
        kept = [month for month in months if month != omitted]
        subset = pairs[pairs["month"].isin(kept)]
        state_stats = {
            state: _pair_stats(subset[subset["regime_state"].eq(state)])
            for state in ("LOW_REGIME", "HIGH_REGIME")
        }
        delta = (
            state_stats["HIGH_REGIME"]["within_date_pair_concordance"]
            - state_stats["LOW_REGIME"]["within_date_pair_concordance"]
        )
        rows.append({
            "omitted_month": omitted,
            "included_months": "+".join(kept),
            "low_informative_dates": state_stats["LOW_REGIME"]["informative_date_count"],
            "low_pairs": state_stats["LOW_REGIME"]["pair_count"],
            "low_concordance": state_stats["LOW_REGIME"]["within_date_pair_concordance"],
            "high_informative_dates": state_stats["HIGH_REGIME"]["informative_date_count"],
            "high_pairs": state_stats["HIGH_REGIME"]["pair_count"],
            "high_concordance": state_stats["HIGH_REGIME"]["within_date_pair_concordance"],
            "delta_high_minus_low": delta,
            "same_direction_as_full": bool(delta < 0),
        })
    return pd.DataFrame(rows)


def build_rank2_6(
    population: pd.DataFrame, fixed_split: pd.DataFrame
) -> pd.DataFrame:
    head = population[population["s2_rank"].between(2, 6)].copy()
    pairs = _pairs_with_state(head, fixed_split)
    rows: list[dict[str, Any]] = []
    for period in ("POOLED", *MONTHS):
        period_pairs = pairs if period == "POOLED" else pairs[pairs["month"].eq(period)]
        state_stats: dict[str, dict[str, Any]] = {}
        for state in ("LOW_REGIME", "HIGH_REGIME"):
            stats = _pair_stats(period_pairs[period_pairs["regime_state"].eq(state)])
            state_stats[state] = stats
            rows.append({
                "period": period,
                "rank_region": "S2_RANK2_6_FIXED",
                "regime_state": state,
                **stats,
            })
        delta = (
            state_stats["HIGH_REGIME"]["within_date_pair_concordance"]
            - state_stats["LOW_REGIME"]["within_date_pair_concordance"]
        )
        for row in rows[-2:]:
            row["delta_high_minus_low"] = delta
    return pd.DataFrame(rows)


def build_rankwise(population_with_state: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ("POOLED", *MONTHS):
        period_frame = (
            population_with_state
            if period == "POOLED"
            else population_with_state[population_with_state["month"].eq(period)]
        )
        for state in ("LOW_REGIME", "HIGH_REGIME"):
            state_frame = period_frame[period_frame["regime_state"].eq(state)]
            for rank in (1, 2, 3):
                frame = state_frame[state_frame["s2_rank"].eq(rank)]
                n = len(frame)
                rows.append({
                    "period": period,
                    "regime_state": state,
                    "s2_rank": rank,
                    "rows": n,
                    "signal_dates": int(frame["signal_date"].nunique()),
                    "target7_count": int(frame["target7"].sum()),
                    "target7_rate": float(frame["target7"].mean()) if n else math.nan,
                    "loss_count": int(frame["loss"].sum()),
                    "loss_rate": float(frame["loss"].mean()) if n else math.nan,
                    "pnt_count": int(frame["positive_non_target"].sum()),
                    "pnt_rate": float(frame["positive_non_target"].mean()) if n else math.nan,
                    "mean_capped_return_7": float(frame["capped_return_7"].mean()) if n else math.nan,
                })
    return pd.DataFrame(rows)


def build_bootstrap(
    pairs: pd.DataFrame,
    all_dates: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    dates = np.asarray(sorted(map(str, all_dates)))
    daily = pairs.groupby(["signal_date", "regime_state"])["t7_high_concordance"].mean().unstack()
    low_map = daily.get("LOW_REGIME", pd.Series(dtype=float)).to_dict()
    high_map = daily.get("HIGH_REGIME", pd.Series(dtype=float)).to_dict()
    low = np.asarray([low_map.get(date, math.nan) for date in dates], dtype=float)
    high = np.asarray([high_map.get(date, math.nan) for date in dates], dtype=float)
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, len(dates), size=(resamples, len(dates)))

    def sampled_mean(values: np.ndarray) -> np.ndarray:
        sampled = values[schedule]
        valid_counts = np.sum(np.isfinite(sampled), axis=1)
        return np.divide(
            np.nansum(sampled, axis=1),
            valid_counts,
            out=np.full(resamples, np.nan),
            where=valid_counts > 0,
        )

    delta = sampled_mean(high) - sampled_mean(low)
    valid = delta[np.isfinite(delta)]
    observed = float(np.nanmean(high) - np.nanmean(low))
    return pd.DataFrame([{
        "stock_signal": FIXED_SIGNAL,
        "regime_variable": FIXED_REGIME,
        "sampling_unit": "SIGNAL_DATE",
        "seed": seed,
        "resamples": resamples,
        "valid_resamples": len(valid),
        "observed_delta_high_minus_low": observed,
        "bootstrap_delta_p2_5": float(np.quantile(valid, .025)),
        "bootstrap_delta_p50": float(np.quantile(valid, .50)),
        "bootstrap_delta_p97_5": float(np.quantile(valid, .975)),
        "P_delta_lt_zero": float(np.mean(valid < 0)),
        "P_delta_gt_zero": float(np.mean(valid > 0)),
        "fixed_relation_only": "YES",
    }])


def _random_target_mask(
    rng: np.random.Generator, resamples: int, row_count: int, target_count: int
) -> np.ndarray:
    """Choose exactly target_count identities per row without replacement."""
    random_key = rng.random((resamples, row_count))
    chosen = np.argpartition(random_key, kth=target_count - 1, axis=1)[:, :target_count]
    mask = np.zeros((resamples, row_count), dtype=bool)
    mask[np.arange(resamples)[:, None], chosen] = True
    return mask


def build_selection_permutation(
    population: pd.DataFrame,
    daily: pd.DataFrame,
    observed_abs_delta: float,
    *,
    resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Westfall-Young-style max statistic over the fixed prior 8 x 11 family."""
    dates = np.asarray(sorted(population["signal_date"].unique()))
    signal_count = len(STOCK_SIGNALS)
    daily_concordance = np.full((resamples, signal_count, len(dates)), np.nan)
    rng = np.random.default_rng(seed)
    count_checks: list[bool] = []

    for date_index, signal_date in enumerate(dates):
        day = population[
            population["signal_date"].eq(signal_date)
            & (population["target7"].astype(bool) | population["loss"].astype(bool))
        ].sort_values("event_id", kind="mergesort")
        target_count = int(day["target7"].sum())
        loss_count = int(day["loss"].sum())
        row_count = len(day)
        if target_count == 0 or loss_count == 0:
            continue
        target_mask = _random_target_mask(rng, resamples, row_count, target_count)
        count_checks.append(bool(np.all(target_mask.sum(axis=1) == target_count)))
        for signal_index, signal in enumerate(STOCK_SIGNALS):
            ranks = day[signal].rank(method="average", ascending=True).to_numpy(float)
            rank_sum = target_mask @ ranks
            auc = (
                rank_sum - target_count * (target_count + 1) / 2.0
            ) / (target_count * loss_count)
            daily_concordance[:, signal_index, date_index] = auc

    split_long, _ = build_regime_splits(daily)
    max_abs_delta = np.zeros(resamples, dtype=float)
    relation_count = 0
    for signal_index, _signal in enumerate(STOCK_SIGNALS):
        values = daily_concordance[:, signal_index, :]
        for regime_variable in REGIME_VARIABLES:
            state = split_long[split_long["regime_variable"].eq(regime_variable)].set_index(
                "signal_date"
            )["regime_state"]
            low_mask = np.asarray([state[date] == "LOW_REGIME" for date in dates])
            high_mask = ~low_mask
            low = np.nanmean(values[:, low_mask], axis=1)
            high = np.nanmean(values[:, high_mask], axis=1)
            delta = np.abs(high - low)
            max_abs_delta = np.maximum(max_abs_delta, delta)
            relation_count += 1
    if relation_count != 88:
        raise RuntimeError("FATAL: permutation maximum did not cover exactly 88 relationships")
    exceeds = max_abs_delta + 1e-15 >= observed_abs_delta
    count_exceeds = int(exceeds.sum())
    p_raw = float(np.mean(exceeds))
    p_plus_one = float((count_exceeds + 1) / (resamples + 1))
    output = pd.DataFrame({
        "permutation_index": np.arange(1, resamples + 1, dtype=int),
        "seed": seed,
        "sampling_unit": "SIGNAL_DATE",
        "permutation_scope": "WITHIN_DATE_T7_LOSS_IDENTITIES_ONLY",
        "family_relationship_count": 88,
        "max_abs_delta_across_88": max_abs_delta,
        "frozen_relation_observed_abs_delta": observed_abs_delta,
        "max_abs_delta_ge_observed": exceeds,
        "selection_adjusted_p_raw": p_raw,
        "selection_adjusted_p_plus_one": p_plus_one,
        "same_date_class_counts_preserved": "YES",
    })
    audit = {
        "selection_adjusted_p": p_plus_one,
        "selection_adjusted_p_raw": p_raw,
        "permutation_exceed_count": count_exceeds,
        "permutation_null_p50": float(np.quantile(max_abs_delta, .50)),
        "permutation_null_p90": float(np.quantile(max_abs_delta, .90)),
        "permutation_null_p95": float(np.quantile(max_abs_delta, .95)),
        "permutation_null_p99": float(np.quantile(max_abs_delta, .99)),
        "permutation_counts_preserved": bool(all(count_checks)),
        "permutation_relationship_count": relation_count,
    }
    return output, audit


def build_leave_one_date(
    pairs: pd.DataFrame, full_delta: float, all_dates: Sequence[str], fixed_split: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Include all 60 development dates, even if no T7-vs-LOSS pair exists on a
    # date; removing a non-informative date correctly has zero influence.
    population_dates = sorted(set(map(str, all_dates)))
    state_by_date = fixed_split.set_index("signal_date")["regime_state"]
    for omitted_date in population_dates:
        subset = pairs[~pairs["signal_date"].eq(omitted_date)]
        low = _pair_stats(subset[subset["regime_state"].eq("LOW_REGIME")])
        high = _pair_stats(subset[subset["regime_state"].eq("HIGH_REGIME")])
        delta = high["within_date_pair_concordance"] - low["within_date_pair_concordance"]
        rows.append({
            "omitted_signal_date": omitted_date,
            "omitted_regime_state": state_by_date.loc[omitted_date],
            "low_informative_dates": low["informative_date_count"],
            "low_concordance": low["within_date_pair_concordance"],
            "high_informative_dates": high["informative_date_count"],
            "high_concordance": high["within_date_pair_concordance"],
            "delta_high_minus_low": delta,
            "delta_change_vs_full": delta - full_delta,
            "absolute_delta_change_vs_full": abs(delta - full_delta),
            "sign_flip_vs_full": bool(np.sign(delta) != np.sign(full_delta)),
        })
    frame = pd.DataFrame(rows)
    frame["influence_rank"] = frame["absolute_delta_change_vs_full"].rank(
        method="first", ascending=False
    ).astype(int)
    return frame.sort_values("omitted_signal_date", kind="mergesort").reset_index(drop=True)


def build_lineage() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "field_type": "STOCK_SIGNAL",
            "audit_name": FIXED_SIGNAL,
            "source_column": FIXED_SIGNAL,
            "source_artifact": POPULATION_AUTHORITY,
            "formula_or_definition": "d1_close / d1_vwap - 1",
            "raw_dependencies": "d1_close;d1_vwap",
            "source_semantics": "existing D1 5-minute VWAP-position field",
            "information_available_at": "D1_CLOSE",
            "contains_d2_or_d3_information": "NO",
            "depends_on_future_outcome": "NO",
            "transform_in_confirmation": "NONE",
        },
        {
            "field_type": "LOCAL_REGIME_VARIABLE",
            "audit_name": FIXED_REGIME,
            "source_column": FIXED_REGIME_SOURCE,
            "source_artifact": REGIME_AUTHORITY,
            "formula_or_definition": "count(consecutive_limit_up_count >= 4) in D1 final-limit-up pool",
            "raw_dependencies": "D1 final-limit-up pool;consecutive_limit_up_count",
            "source_semantics": "existing main-board final-limit-up pool ecology",
            "information_available_at": "D1_CLOSE",
            "contains_d2_or_d3_information": "NO",
            "depends_on_future_outcome": "NO",
            "transform_in_confirmation": "LOW <= fixed all-60-date median 1; HIGH > 1",
        },
        {
            "field_type": "REGIME_SPLIT_RULE",
            "audit_name": "board4plus_fixed_median_split",
            "source_column": FIXED_REGIME_SOURCE,
            "source_artifact": PRIOR_SUMMARY_AUTHORITY,
            "formula_or_definition": "median over all 60 development signal dates = 1",
            "raw_dependencies": FIXED_REGIME,
            "source_semantics": "exact prior preregistered split reused without monthly recalculation",
            "information_available_at": "AUDIT_FIXED_FROM_DEVELOPMENT",
            "contains_d2_or_d3_information": "NO",
            "depends_on_future_outcome": "NO",
            "transform_in_confirmation": "NO_THRESHOLD_SEARCH",
        },
        {
            "field_type": "SELECTION_CORRECTION_FAMILY",
            "audit_name": "prior_8x11_family",
            "source_column": "8 frozen stock signals x 11 frozen local regimes",
            "source_artifact": PRIOR_SUMMARY_AUTHORITY,
            "formula_or_definition": "max(abs(HIGH concordance - LOW concordance)) across exactly 88",
            "raw_dependencies": "prior fixed manifests",
            "source_semantics": "selection-aware within-date label permutation maximum statistic",
            "information_available_at": "AUDIT_ONLY",
            "contains_d2_or_d3_information": "LABELS_PERMUTED_ONLY_FOR_NULL",
            "depends_on_future_outcome": "NO_AUGUST",
            "transform_in_confirmation": "NO_SECOND_RELATION_REPORTED_OR_SELECTED",
        },
    ])


def decide_state(
    reproduction_audit: Mapping[str, Any],
    lomo: pd.DataFrame,
    rank2_6: pd.DataFrame,
    bootstrap: pd.DataFrame,
    permutation_audit: Mapping[str, Any],
    leave_one_date: pd.DataFrame,
) -> tuple[str, dict[str, bool]]:
    low = float(reproduction_audit["low_concordance"])
    high = float(reproduction_audit["high_concordance"])
    delta = float(reproduction_audit["delta"])
    pooled_rank = rank2_6[rank2_6["period"].eq("POOLED")].set_index("regime_state")
    rank_low = float(pooled_rank.loc["LOW_REGIME", "within_date_pair_concordance"])
    rank_high = float(pooled_rank.loc["HIGH_REGIME", "within_date_pair_concordance"])
    rank_low_pairs = int(pooled_rank.loc["LOW_REGIME", "pair_count"])
    rank_high_pairs = int(pooled_rank.loc["HIGH_REGIME", "pair_count"])
    rank_low_dates = int(pooled_rank.loc["LOW_REGIME", "informative_date_count"])
    rank_high_dates = int(pooled_rank.loc["HIGH_REGIME", "informative_date_count"])
    boot = bootstrap.iloc[0]
    gates = {
        "reproduction": bool(reproduction_audit["reproduction_pass"]),
        "clear_conditional_direction": bool(
            low >= CLEAR_LOW_MIN and high <= CLEAR_HIGH_MAX and delta <= -CLEAR_ABS_DELTA_MIN
        ),
        "lomo_majority_same_direction": bool((lomo["delta_high_minus_low"] < 0).sum() >= 2),
        "lomo_all_same_direction": bool((lomo["delta_high_minus_low"] < 0).all()),
        "rank2_6_same_direction": bool(
            rank_low > .5
            and rank_high < .5
            and rank_low_pairs >= RANK2_6_MIN_PAIRS_PER_STATE
            and rank_high_pairs >= RANK2_6_MIN_PAIRS_PER_STATE
            and rank_low_dates >= RANK2_6_MIN_DATES_PER_STATE
            and rank_high_dates >= RANK2_6_MIN_DATES_PER_STATE
        ),
        "bootstrap_support": bool(
            float(boot["P_delta_lt_zero"]) >= BOOTSTRAP_DIRECTION_PROBABILITY_MIN
            and float(boot["bootstrap_delta_p97_5"]) < 0
        ),
        "selection_adjusted_support": bool(
            float(permutation_audit["selection_adjusted_p"]) <= SELECTION_ADJUSTED_ALPHA
        ),
        "not_one_or_two_dates_driven": bool(
            int(leave_one_date["sign_flip_vs_full"].sum()) == 0
            and float(leave_one_date["delta_high_minus_low"].max()) < 0
        ),
    }
    fixed_stable = all(
        gates[name]
        for name in (
            "reproduction", "clear_conditional_direction",
            "lomo_majority_same_direction", "rank2_6_same_direction",
            "bootstrap_support", "not_one_or_two_dates_driven",
        )
    )
    if all(gates.values()):
        state = "SINGLE_RELATION_CONFIRMED"
    elif fixed_stable and not gates["selection_adjusted_support"]:
        state = "SINGLE_RELATION_SUPPORTED_BUT_SELECTION_RISK"
    elif gates["reproduction"] and gates["clear_conditional_direction"] and (
        not gates["lomo_majority_same_direction"]
        or not gates["rank2_6_same_direction"]
        or not gates["not_one_or_two_dates_driven"]
    ):
        state = "SINGLE_RELATION_UNSTABLE"
    else:
        state = "SINGLE_RELATION_NOT_SUPPORTED"
    return state, gates


def _fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def build_review(
    reproduction_audit: Mapping[str, Any],
    monthly: pd.DataFrame,
    lomo: pd.DataFrame,
    rank2_6: pd.DataFrame,
    bootstrap: pd.DataFrame,
    permutation_audit: Mapping[str, Any],
    leave_one_date: pd.DataFrame,
    state: str,
    gates: Mapping[str, bool],
) -> str:
    month_lines: list[str] = []
    for month in MONTHS:
        part = monthly[monthly["month"].eq(month)].set_index("regime_state")
        month_lines.append(
            f"{month}: LOW {_fmt(part.loc['LOW_REGIME', 'within_date_pair_concordance'])}; "
            f"HIGH {_fmt(part.loc['HIGH_REGIME', 'within_date_pair_concordance'])}."
        )
    rank = rank2_6[rank2_6["period"].eq("POOLED")].set_index("regime_state")
    top_dates = leave_one_date.nsmallest(5, "influence_rank")
    top_text = "; ".join(
        f"{row.omitted_signal_date} (|change|={_fmt(row.absolute_delta_change_vs_full)})"
        for row in top_dates.itertuples(index=False)
    )
    boot = bootstrap.iloc[0]
    lines = [
        "# v004c Close/VWAP × Board4plus Confirmation v001",
        "",
        "## Frozen Contract",
        "",
        "- Single stock signal: d1_close_to_vwap_raw.",
        "- Single regime: board4plus_count; fixed all-60-date median = 1; LOW <= 1, HIGH > 1.",
        "- No model, feature/threshold search, score/rank change, second relation, or August outcome.",
        "",
        "## Q1. Can the prior 0.591 vs 0.315 relationship be reproduced exactly?",
        "",
        f"- {'YES' if reproduction_audit['reproduction_pass'] else 'NO'}. LOW {_fmt(reproduction_audit['low_concordance'])}; HIGH {_fmt(reproduction_audit['high_concordance'])}; delta {_fmt(reproduction_audit['delta'])}; max parity error {reproduction_audit['reproduction_max_abs_error']:.3g}.",
        "",
        "## Direction in Plain Language",
        "",
        "- LOW board4plus: Target7 names tend to close higher versus D1 VWAP than LOSS names.",
        "- HIGH board4plus: the direction reverses; Target7 names tend to close lower versus D1 VWAP than LOSS names.",
        "",
        "## Q2. Is the relationship visible in May, June, and July?",
        "",
        "- " + " ".join(month_lines),
        "",
        "## Q3. Does it remain after omitting any one month?",
        "",
        f"- {'YES' if gates['lomo_all_same_direction'] else 'PARTIAL'}. HIGH-minus-LOW deltas: "
        + "; ".join(
            f"omit {row.omitted_month} {_fmt(row.delta_high_minus_low)}"
            for row in lomo.itertuples(index=False)
        )
        + ".",
        "",
        "## Q4. Does it remain in the fixed S2 Rank2-6 error region?",
        "",
        f"- {'YES' if gates['rank2_6_same_direction'] else 'NO'}. LOW {_fmt(rank.loc['LOW_REGIME', 'within_date_pair_concordance'])}; HIGH {_fmt(rank.loc['HIGH_REGIME', 'within_date_pair_concordance'])}; delta {_fmt(rank.loc['LOW_REGIME', 'delta_high_minus_low'])}.",
        "",
        "## Q5. Is the effect driven by only a few dates?",
        "",
        f"- {'NO' if gates['not_one_or_two_dates_driven'] else 'YES/UNCERTAIN'}. LODO delta range [{_fmt(leave_one_date['delta_high_minus_low'].min())}, {_fmt(leave_one_date['delta_high_minus_low'].max())}], sign flips {int(leave_one_date['sign_flip_vs_full'].sum())}. Largest five date influences: {top_text}.",
        "",
        "## Q6. After correcting for selecting the strongest of 88 relationships, is it still abnormal?",
        "",
        f"- {'YES' if gates['selection_adjusted_support'] else 'NO'}. Selection-adjusted permutation p = {permutation_audit['selection_adjusted_p']:.4f}; null p95 = {_fmt(permutation_audit['permutation_null_p95'])}; observed |delta| = {_fmt(abs(reproduction_audit['delta']))}.",
        "",
        "## Fixed-Relation Bootstrap",
        "",
        f"- Delta median {_fmt(boot['bootstrap_delta_p50'])}; 95% CI [{_fmt(boot['bootstrap_delta_p2_5'])}, {_fmt(boot['bootstrap_delta_p97_5'])}]; P(delta < 0) = {_fmt(boot['P_delta_lt_zero'])}.",
        "",
        f"SINGLE_RELATION_STATE = {state}",
        "",
        "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO",
        "",
        "NEXT_ACTION = STOP_AND_REVIEW",
        "",
    ]
    return "\n".join(lines)


def build_outputs(
    root: str | Path,
    *,
    permutation_resamples: int = PERMUTATION_RESAMPLES,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_contract()
    population, daily, input_audit = load_inputs(root)
    enriched, fixed_split, split_info = _attach_fixed_state(population, daily)
    reproduction, reproduction_audit, pairs = build_reproduction(
        population, fixed_split, root
    )
    if not reproduction_audit["reproduction_pass"]:
        review = (
            "# v004c Close/VWAP × Board4plus Confirmation v001\n\n"
            "CONFIRMATION_STATE = REPRODUCTION_FAIL\n\n"
            "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO\n\n"
            "NEXT_ACTION = STOP_AND_REVIEW\n"
        )
        outputs = {
            OUTPUT_FILENAMES[0]: _csv_bytes(reproduction),
            OUTPUT_FILENAMES[-1]: review.encode("utf-8"),
        }
        return outputs, {
            **input_audit,
            **reproduction_audit,
            "confirmation_state": "REPRODUCTION_FAIL",
            "august_signal_outcome_accessed": "NO",
        }

    monthly = build_monthly(population, fixed_split, pairs)
    lomo = build_lomo(pairs)
    rank2_6 = build_rank2_6(population, fixed_split)
    rankwise = build_rankwise(enriched)
    bootstrap = build_bootstrap(
        pairs,
        population["signal_date"].unique(),
        resamples=bootstrap_resamples,
    )
    permutation, permutation_audit = build_selection_permutation(
        population,
        daily,
        abs(reproduction_audit["delta"]),
        resamples=permutation_resamples,
    )
    leave_one_date = build_leave_one_date(
        pairs,
        reproduction_audit["delta"],
        population["signal_date"].unique(),
        fixed_split,
    )
    lineage = build_lineage()
    state, gates = decide_state(
        reproduction_audit, lomo, rank2_6, bootstrap,
        permutation_audit, leave_one_date,
    )
    review = build_review(
        reproduction_audit, monthly, lomo, rank2_6, bootstrap,
        permutation_audit, leave_one_date, state, gates,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(reproduction),
        OUTPUT_FILENAMES[1]: _csv_bytes(monthly),
        OUTPUT_FILENAMES[2]: _csv_bytes(lomo),
        OUTPUT_FILENAMES[3]: _csv_bytes(rank2_6),
        OUTPUT_FILENAMES[4]: _csv_bytes(rankwise),
        OUTPUT_FILENAMES[5]: _csv_bytes(bootstrap),
        OUTPUT_FILENAMES[6]: _csv_bytes(permutation),
        OUTPUT_FILENAMES[7]: _csv_bytes(leave_one_date),
        OUTPUT_FILENAMES[8]: _csv_bytes(lineage),
        OUTPUT_FILENAMES[9]: review.encode("utf-8"),
    }
    audit = {
        **input_audit,
        **reproduction_audit,
        **permutation_audit,
        **{f"gate_{key}": value for key, value in gates.items()},
        "fixed_signal": FIXED_SIGNAL,
        "fixed_regime": FIXED_REGIME,
        "fixed_regime_threshold": float(split_info["threshold"]),
        "prior_family_relationship_count": 88,
        "permutation_resamples": permutation_resamples,
        "bootstrap_resamples": bootstrap_resamples,
        "lomo_same_direction_count": int((lomo["delta_high_minus_low"] < 0).sum()),
        "leave_one_date_runs": len(leave_one_date),
        "leave_one_date_sign_flips": int(leave_one_date["sign_flip_vs_full"].sum()),
        "single_relation_state": state,
        "next_action": "STOP_AND_REVIEW",
        "august_signal_outcome_accessed": "NO",
        "model_fits": 0,
        "threshold_searches": 0,
        "second_relations_selected": 0,
        "output_file_count": len(outputs),
    }
    return outputs, audit


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, audit = build_outputs(root_path)
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        (output_dir / filename).write_bytes(payload)
    return output_dir, audit

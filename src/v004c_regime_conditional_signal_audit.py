"""Conditional information audit for fixed v004c stock signals and local ecology.

The audit is deliberately descriptive.  It reuses the locked mature S2
population and the prior board-break daily ecology, applies one fixed median
split per preregistered regime variable, and computes same-date Target7-vs-
LOSS concordance.  It never fits a model, changes a score/rank, searches a
threshold/grouping, constructs an interaction feature, or reads an August
signal-date outcome.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TASK_NAME = "v004c_regime_conditional_signal_audit_v001"
OUTPUT_DIRNAME = "v004c_regime_conditional_signal_audit_v001_202605_202607"
ANALYSIS_START = "2026-05-06"
ANALYSIS_END = "2026-07-29"
AUGUST_ASOF = "2026-08-01"
EXPECTED_ROWS = 485
EXPECTED_DATES = 60

BOOTSTRAP_SEED = 20260827
BOOTSTRAP_RESAMPLES = 20_000

# Fixed interpretation/support rules.  They are not optimized from the data.
MATERIAL_HIGH = 0.55
MATERIAL_LOW = 0.45
MATERIAL_CONDITIONAL_GAP = 0.15
MIN_SPLIT_DATES = 10
MIN_STATE_INFORMATIVE_DATES = 5
MIN_STATE_PAIRS = 20
MIN_MONTH_INFORMATIVE_DATES = 2
MIN_MONTH_PAIRS = 5
MIN_RANK_STATE_DATES = 3
MIN_RANK_STATE_PAIRS = 10
MIN_RANK_CONDITIONAL_GAP = 0.10
MONTH_PROXY_CRAMERS_V = 0.70

MONTHS: Mapping[str, str] = {
    "MAY": "2026-05",
    "JUNE": "2026-06",
    "JULY": "2026-07",
}

STOCK_SIGNALS = (
    "rank_theme_score",
    "rank_active_money_score",
    "rank_log_candidate_base_price",
    "rank_trend_hold_score",
    "d1_open_to_close_return_raw",
    "d1_afternoon_return",
    "d1_close_to_vwap_raw",
    "d1_low_to_close_recovery",
)

RANK_SIGNAL_COLUMNS = STOCK_SIGNALS[:4]
PATH_SIGNAL_COLUMNS = STOCK_SIGNALS[4:]

# audit_name -> (source_column, semantics)
REGIME_VARIABLES: Mapping[str, tuple[str, str]] = {
    "final_limit_up_pool_count": (
        "limit_up_count",
        "existing main-board final-limit-up pool count",
    ),
    "board2_count": (
        "two_board_limit_up_count",
        "two-board names in existing final-limit-up pool",
    ),
    "board3_count": (
        "three_board_limit_up_count",
        "three-board names in existing final-limit-up pool",
    ),
    "board4plus_count": (
        "four_plus_board_limit_up_count",
        "four-or-more-board names in existing final-limit-up pool",
    ),
    "max_board_height": (
        "highest_board_height",
        "highest consecutive-board height in existing final-limit-up pool",
    ),
    "promotion_rate": (
        "promotion_success_rate",
        "existing final-limit-up pool continuation rate from prior pool date",
    ),
    "first_break_candidate_count": (
        "candidate_count",
        "authoritative v004c Board2/Board3 first-break candidate count",
    ),
    "candidate_cohort_d1_afternoon_median": (
        "d1_afternoon_return__median",
        "same-date v004c candidate-cohort median D1 afternoon return",
    ),
    "candidate_cohort_d1_close_vwap_median": (
        "d1_close_to_vwap_raw__median",
        "same-date v004c candidate-cohort median D1 close-to-VWAP return",
    ),
    "candidate_cohort_d1_open_close_median": (
        "d1_open_to_close_return_raw__median",
        "same-date v004c candidate-cohort median D1 open-to-close return",
    ),
    "candidate_cohort_d1_low_close_recovery_median": (
        "d1_low_to_close_recovery__median",
        "same-date v004c candidate-cohort median D1 low-to-close recovery",
    ),
}

POPULATION_AUTHORITY = (
    "reports/research/"
    "v004c_d1_unfinished_repair_information_audit_v001_20260506_20260731/"
    "v004c_unfinished_repair_population_v001.csv"
)
BRIDGE_AUTHORITY = (
    "reports/research/"
    "v004c_stage1_18f_bridge_analysis_v001_20260506_20260731/"
    "v004c_stage1_18f_bridge_candidates.csv"
)
REGIME_AUTHORITY = (
    "reports/research/"
    "v004c_board_break_regime_audit_v001_202605_202607/"
    "v004c_regime_daily_ecology_v001.csv"
)

OUTPUT_FILENAMES = (
    "v004c_regime_variable_month_summary_v001.csv",
    "v004c_signal_regime_conditional_pairs_v001.csv",
    "v004c_signal_regime_conditional_summary_v001.csv",
    "v004c_signal_regime_month_comparison_v001.csv",
    "v004c_open_close_regime_explanation_v001.csv",
    "v004c_rank2_6_regime_conditional_v001.csv",
    "v004c_regime_conditional_bootstrap_v001.csv",
    "v004c_regime_conditional_review_v001.md",
    "v004c_regime_conditional_lineage_v001.csv",
)


def assert_contract() -> None:
    if len(STOCK_SIGNALS) != 8 or len(set(STOCK_SIGNALS)) != 8:
        raise RuntimeError("FATAL: fixed eight-signal manifest changed")
    if len(REGIME_VARIABLES) != 11 or len(set(REGIME_VARIABLES)) != 11:
        raise RuntimeError("FATAL: fixed eleven-regime manifest changed")
    if ANALYSIS_END >= AUGUST_ASOF:
        raise RuntimeError("FATAL: signal-date boundary reached August")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _month_label(signal_date: str) -> str:
    prefix = str(signal_date)[:7]
    for label, month in MONTHS.items():
        if prefix == month:
            return label
    raise ValueError(f"date outside fixed audit months: {signal_date}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"event_id": str, "code": str, "signal_date": str},
    )


def load_inputs(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and parity-check the mature stock and date-level authorities."""
    assert_contract()
    root_path = Path(root).resolve()
    population = _read_csv(root_path / POPULATION_AUTHORITY)
    bridge = _read_csv(root_path / BRIDGE_AUTHORITY)
    daily = _read_csv(root_path / REGIME_AUTHORITY)

    population["signal_date"] = population["signal_date"].astype(str)
    population["label_available_date"] = population["label_available_date"].astype(str)
    bridge["signal_date"] = bridge["signal_date"].astype(str)
    bridge["label_available_date"] = bridge["label_available_date"].astype(str)
    bridge = bridge[bridge["label_available_date"].lt(AUGUST_ASOF)].copy()

    left_ids = population[["event_id", "signal_date", "code"]].sort_values(
        "event_id", kind="mergesort"
    ).reset_index(drop=True)
    right_ids = bridge[["event_id", "signal_date", "code"]].sort_values(
        "event_id", kind="mergesort"
    ).reset_index(drop=True)
    if len(left_ids) != len(right_ids) or not left_ids.astype(str).equals(right_ids.astype(str)):
        raise RuntimeError("FATAL: mature S2 population differs from mature bridge authority")

    feature_source = bridge[["event_id", *RANK_SIGNAL_COLUMNS]].copy()
    population = population.merge(feature_source, on="event_id", validate="one_to_one")
    population["month"] = population["signal_date"].map(_month_label)

    if (len(population), population["signal_date"].nunique()) != (
        EXPECTED_ROWS,
        EXPECTED_DATES,
    ):
        raise RuntimeError("FATAL: mature May-July population parity failed")
    if population["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id")
    if population["signal_date"].min() != ANALYSIS_START or population["signal_date"].max() != ANALYSIS_END:
        raise RuntimeError("FATAL: mature analysis date range changed")
    if population["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal date reached stock population")
    if population["label_available_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: immature/August-available outcome reached population")
    if population[list(STOCK_SIGNALS)].isna().any().any():
        raise RuntimeError("FATAL: fixed stock signal values are incomplete")

    daily["signal_date"] = daily["signal_date"].astype(str)
    daily["month"] = daily["signal_date"].map(_month_label)
    if (len(daily), daily["signal_date"].nunique()) != (EXPECTED_DATES, EXPECTED_DATES):
        raise RuntimeError("FATAL: prior daily regime authority parity failed")
    if set(daily["signal_date"]) != set(population["signal_date"]):
        raise RuntimeError("FATAL: daily regime dates differ from mature S2 dates")
    source_columns = [source for source, _ in REGIME_VARIABLES.values()]
    if any(column not in daily.columns for column in source_columns):
        missing = [column for column in source_columns if column not in daily.columns]
        raise RuntimeError(f"FATAL: prior regime columns missing: {missing}")
    daily[source_columns] = daily[source_columns].apply(pd.to_numeric, errors="coerce")
    if daily[source_columns].isna().any().any():
        raise RuntimeError("FATAL: fixed local regime variables are incomplete")
    if daily["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal date reached daily regime authority")

    # Candidate count is the same frozen first-break universe in both assets.
    population_counts = population.groupby("signal_date").size()
    daily_counts = daily.set_index("signal_date")["candidate_count"].astype(int)
    if not population_counts.sort_index().equals(daily_counts.sort_index()):
        raise RuntimeError("FATAL: daily ecology candidate counts differ from mature population")

    population = population.sort_values(
        ["signal_date", "s2_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    daily = daily.sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    audit = {
        "rows": len(population),
        "dates": population["signal_date"].nunique(),
        "stock_signal_count": len(STOCK_SIGNALS),
        "regime_variable_count": len(REGIME_VARIABLES),
        "duplicate_event_id_count": int(population["event_id"].duplicated().sum()),
        "missing_stock_signal_cells": int(population[list(STOCK_SIGNALS)].isna().sum().sum()),
        "missing_regime_cells": int(daily[source_columns].isna().sum().sum()),
        "max_signal_date": population["signal_date"].max(),
        "max_label_available_date": population["label_available_date"].max(),
        "model_fits": 0,
        "threshold_searches": 0,
        "new_features": 0,
        "august_signal_outcome_accessed": "NO",
    }
    return population, daily, audit


def _cramers_v(month: pd.Series, state: pd.Series) -> float:
    table = pd.crosstab(month, state)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return math.nan
    observed = table.to_numpy(float)
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True)
    expected /= observed.sum()
    valid = expected > 0
    chi2 = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    denom = observed.sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else math.nan


def build_regime_splits(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Apply exactly one all-60-date median split to each fixed variable."""
    rows: list[dict[str, Any]] = []
    info: dict[str, dict[str, Any]] = {}
    for regime_variable, (source_column, semantics) in REGIME_VARIABLES.items():
        values = pd.to_numeric(daily[source_column], errors="raise")
        threshold = float(values.median())
        states = np.where(values.le(threshold), "LOW_REGIME", "HIGH_REGIME")
        low_dates = int(np.sum(states == "LOW_REGIME"))
        high_dates = int(np.sum(states == "HIGH_REGIME"))
        unique_values = int(values.nunique(dropna=True))
        usable = bool(
            unique_values >= 2
            and low_dates >= MIN_SPLIT_DATES
            and high_dates >= MIN_SPLIT_DATES
        )
        status = "USABLE_FIXED_MEDIAN_SPLIT" if usable else "REGIME_SPLIT_NOT_USABLE"
        temp = pd.DataFrame({
            "signal_date": daily["signal_date"].astype(str),
            "month": daily["month"].astype(str),
            "regime_state": states,
        })
        cramers_v = _cramers_v(temp["month"], temp["regime_state"])
        proxy = bool(np.isfinite(cramers_v) and cramers_v >= MONTH_PROXY_CRAMERS_V)
        month_counts = pd.crosstab(temp["month"], temp["regime_state"])
        represented_months = {
            state: int((month_counts.get(state, pd.Series(dtype=float)) > 0).sum())
            for state in ("LOW_REGIME", "HIGH_REGIME")
        }
        info[regime_variable] = {
            "source_column": source_column,
            "semantics": semantics,
            "threshold": threshold,
            "unique_values": unique_values,
            "median_tie_dates": int(values.eq(threshold).sum()),
            "low_dates": low_dates,
            "high_dates": high_dates,
            "split_status": status,
            "split_usable": usable,
            "month_regime_cramers_v": cramers_v,
            "regime_proxy_for_month_only": proxy,
            "low_represented_months": represented_months["LOW_REGIME"],
            "high_represented_months": represented_months["HIGH_REGIME"],
        }
        for row, value, state in zip(daily.itertuples(index=False), values, states):
            rows.append({
                "signal_date": str(row.signal_date),
                "month": str(row.month),
                "regime_variable": regime_variable,
                "regime_source_column": source_column,
                "regime_semantics": semantics,
                "regime_value": float(value),
                "split_threshold_all_60_dates": threshold,
                "regime_state": state,
                "split_status": status,
            })
    long = pd.DataFrame(rows).sort_values(
        ["regime_variable", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)
    return long, info


def build_regime_month_summary(
    daily: pd.DataFrame,
    split_info: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime_variable, (source_column, semantics) in REGIME_VARIABLES.items():
        output: dict[str, Any] = {
            "regime_variable": regime_variable,
            "source_column": source_column,
            "source_artifact": REGIME_AUTHORITY,
            "semantics": semantics,
            "information_available_at": "D1_CLOSE",
        }
        for label in MONTHS:
            values = pd.to_numeric(daily.loc[daily["month"].eq(label), source_column])
            prefix = label.lower()
            output.update({
                f"{prefix}_dates": len(values),
                f"{prefix}_mean": float(values.mean()),
                f"{prefix}_median": float(values.median()),
                f"{prefix}_p25": float(values.quantile(.25)),
                f"{prefix}_p75": float(values.quantile(.75)),
                f"{prefix}_std": float(values.std(ddof=1)),
            })
        output.update({
            "may_vs_june_mean_difference": output["may_mean"] - output["june_mean"],
            "june_vs_july_mean_difference": output["june_mean"] - output["july_mean"],
            "may_vs_july_mean_difference": output["may_mean"] - output["july_mean"],
            "may_vs_june_median_difference": output["may_median"] - output["june_median"],
            "june_vs_july_median_difference": output["june_median"] - output["july_median"],
            "may_vs_july_median_difference": output["may_median"] - output["july_median"],
            **split_info[regime_variable],
        })
        rows.append(output)
    return pd.DataFrame(rows)


def build_signal_pairs(population: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create all unordered cross-class same-date Target7-vs-LOSS pairs."""
    tables: dict[str, pd.DataFrame] = {}
    for signal in STOCK_SIGNALS:
        rows: list[dict[str, Any]] = []
        for signal_date, day in population.groupby("signal_date", sort=True):
            targets = day[day["target7"].astype(bool)].sort_values("event_id", kind="mergesort")
            losses = day[day["loss"].astype(bool)].sort_values("event_id", kind="mergesort")
            for target in targets.itertuples(index=False):
                for loss in losses.itertuples(index=False):
                    target_value = float(getattr(target, signal))
                    loss_value = float(getattr(loss, signal))
                    difference = target_value - loss_value
                    rows.append({
                        "stock_signal": signal,
                        "signal_date": str(signal_date),
                        "month": _month_label(str(signal_date)),
                        "target_event_id": str(target.event_id),
                        "target_code": str(target.code).zfill(6),
                        "target_s2_rank": int(target.s2_rank),
                        "target_signal_value": target_value,
                        "loss_event_id": str(loss.event_id),
                        "loss_code": str(loss.code).zfill(6),
                        "loss_s2_rank": int(loss.s2_rank),
                        "loss_signal_value": loss_value,
                        "signal_difference_t7_minus_loss": difference,
                        "raw_direction": (
                            "T7_HIGH" if difference > 0 else "T7_LOW" if difference < 0 else "TIE"
                        ),
                        "t7_high_concordance": (
                            1.0 if difference > 0 else 0.0 if difference < 0 else .5
                        ),
                    })
        tables[signal] = pd.DataFrame(rows)
    pair_counts = {len(frame) for frame in tables.values()}
    if len(pair_counts) != 1:
        raise RuntimeError("FATAL: fixed complete stock signals produced different pair support")
    return tables


def build_conditional_pairs(
    signal_pairs: Mapping[str, pd.DataFrame],
    regime_splits: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    regime_columns = [
        "signal_date", "regime_variable", "regime_source_column",
        "regime_value", "split_threshold_all_60_dates", "regime_state", "split_status",
    ]
    for signal in STOCK_SIGNALS:
        for regime_variable in REGIME_VARIABLES:
            states = regime_splits[
                regime_splits["regime_variable"].eq(regime_variable)
            ][regime_columns]
            merged = signal_pairs[signal].merge(states, on="signal_date", validate="many_to_one")
            rows.append(merged)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["stock_signal", "regime_variable", "signal_date", "target_event_id", "loss_event_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected = len(STOCK_SIGNALS) * len(REGIME_VARIABLES) * len(next(iter(signal_pairs.values())))
    if len(result) != expected:
        raise RuntimeError("FATAL: conditional pair matrix shape mismatch")
    if not result["signal_date"].eq(result["signal_date"]).all():
        raise RuntimeError("FATAL: invalid pair date")
    return result


def _direction(concordance: float) -> str:
    if not np.isfinite(concordance):
        return "NA"
    if concordance > .5:
        return "T7_HIGH"
    if concordance < .5:
        return "T7_LOW"
    return "TIE"


def _pair_stats(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {
            "informative_date_count": 0,
            "pair_count": 0,
            "within_date_pair_concordance": math.nan,
            "pooled_pair_concordance": math.nan,
            "mean_pair_difference": math.nan,
            "median_pair_difference": math.nan,
            "raw_direction": "NA",
        }
    daily = pairs.groupby("signal_date", sort=True)["t7_high_concordance"].mean()
    concordance = float(daily.mean())
    return {
        "informative_date_count": int(len(daily)),
        "pair_count": int(len(pairs)),
        "within_date_pair_concordance": concordance,
        "pooled_pair_concordance": float(pairs["t7_high_concordance"].mean()),
        "mean_pair_difference": float(pairs["signal_difference_t7_minus_loss"].mean()),
        "median_pair_difference": float(pairs["signal_difference_t7_minus_loss"].median()),
        "raw_direction": _direction(concordance),
    }


def _candidate_stats(frame: pd.DataFrame, signal: str) -> dict[str, Any]:
    targets = pd.to_numeric(frame.loc[frame["target7"].astype(bool), signal], errors="coerce")
    losses = pd.to_numeric(frame.loc[frame["loss"].astype(bool), signal], errors="coerce")
    return {
        "target_rows": int(len(targets)),
        "loss_rows": int(len(losses)),
        "mean_signal_target7": float(targets.mean()) if len(targets) else math.nan,
        "mean_signal_loss": float(losses.mean()) if len(losses) else math.nan,
        "median_signal_target7": float(targets.median()) if len(targets) else math.nan,
        "median_signal_loss": float(losses.median()) if len(losses) else math.nan,
    }


def build_month_comparison(
    population: pd.DataFrame,
    conditional_pairs: pd.DataFrame,
    regime_splits: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in STOCK_SIGNALS:
        for regime_variable in REGIME_VARIABLES:
            split = regime_splits[regime_splits["regime_variable"].eq(regime_variable)]
            tagged = population.merge(
                split[["signal_date", "regime_state"]], on="signal_date", validate="many_to_one"
            )
            relation_pairs = conditional_pairs[
                conditional_pairs["stock_signal"].eq(signal)
                & conditional_pairs["regime_variable"].eq(regime_variable)
            ]
            threshold = float(split["split_threshold_all_60_dates"].iloc[0])
            status = str(split["split_status"].iloc[0])
            for month_label in MONTHS:
                month_pairs = relation_pairs[relation_pairs["month"].eq(month_label)]
                month_tagged = tagged[tagged["month"].eq(month_label)]
                month_dates = int(split[split["month"].eq(month_label)]["signal_date"].nunique())
                for state in ("ALL", "LOW_REGIME", "HIGH_REGIME"):
                    if state == "ALL":
                        pairs = month_pairs
                        candidates = month_tagged
                        state_dates = month_dates
                    else:
                        pairs = month_pairs[month_pairs["regime_state"].eq(state)]
                        candidates = month_tagged[month_tagged["regime_state"].eq(state)]
                        state_dates = int(
                            split[
                                split["month"].eq(month_label)
                                & split["regime_state"].eq(state)
                            ]["signal_date"].nunique()
                        )
                    rows.append({
                        "stock_signal": signal,
                        "regime_variable": regime_variable,
                        "month": month_label,
                        "regime_state": state,
                        "split_threshold_all_60_dates": threshold,
                        "split_status": status,
                        "month_signal_dates": month_dates,
                        "state_signal_dates": state_dates,
                        "state_date_share_in_month": state_dates / month_dates if month_dates else math.nan,
                        **_pair_stats(pairs),
                        **_candidate_stats(candidates, signal),
                    })
    return pd.DataFrame(rows).sort_values(
        ["stock_signal", "regime_variable", "month", "regime_state"],
        kind="mergesort",
    ).reset_index(drop=True)


def _month_support_count(
    month_comparison: pd.DataFrame,
    signal: str,
    regime_variable: str,
    state: str,
    pooled_direction: str,
) -> int:
    part = month_comparison[
        month_comparison["stock_signal"].eq(signal)
        & month_comparison["regime_variable"].eq(regime_variable)
        & month_comparison["regime_state"].eq(state)
    ]
    valid = part[
        part["informative_date_count"].ge(MIN_MONTH_INFORMATIVE_DATES)
        & part["pair_count"].ge(MIN_MONTH_PAIRS)
    ]
    return int(valid["raw_direction"].eq(pooled_direction).sum())


def build_conditional_summary(
    population: pd.DataFrame,
    conditional_pairs: pd.DataFrame,
    regime_splits: pd.DataFrame,
    split_info: Mapping[str, Mapping[str, Any]],
    month_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in STOCK_SIGNALS:
        for regime_variable in REGIME_VARIABLES:
            relation_pairs = conditional_pairs[
                conditional_pairs["stock_signal"].eq(signal)
                & conditional_pairs["regime_variable"].eq(regime_variable)
            ]
            split = regime_splits[regime_splits["regime_variable"].eq(regime_variable)]
            tagged = population.merge(
                split[["signal_date", "regime_state"]], on="signal_date", validate="many_to_one"
            )
            state_stats: dict[str, dict[str, Any]] = {}
            for state in ("LOW_REGIME", "HIGH_REGIME"):
                pairs = relation_pairs[relation_pairs["regime_state"].eq(state)]
                candidates = tagged[tagged["regime_state"].eq(state)]
                state_stats[state] = {
                    "regime_date_count": int(split["regime_state"].eq(state).sum()),
                    **_pair_stats(pairs),
                    **_candidate_stats(candidates, signal),
                }
            low = state_stats["LOW_REGIME"]
            high = state_stats["HIGH_REGIME"]
            low_c = float(low["within_date_pair_concordance"])
            high_c = float(high["within_date_pair_concordance"])
            delta = high_c - low_c if np.isfinite(low_c) and np.isfinite(high_c) else math.nan
            usable_support = bool(
                split_info[regime_variable]["split_usable"]
                and low["informative_date_count"] >= MIN_STATE_INFORMATIVE_DATES
                and high["informative_date_count"] >= MIN_STATE_INFORMATIVE_DATES
                and low["pair_count"] >= MIN_STATE_PAIRS
                and high["pair_count"] >= MIN_STATE_PAIRS
            )
            flip = bool(
                usable_support
                and (
                    (low_c >= MATERIAL_HIGH and high_c <= MATERIAL_LOW)
                    or (low_c <= MATERIAL_LOW and high_c >= MATERIAL_HIGH)
                )
                and abs(delta) >= MATERIAL_CONDITIONAL_GAP
            )
            low_support = _month_support_count(
                month_comparison, signal, regime_variable, "LOW_REGIME", str(low["raw_direction"])
            )
            high_support = _month_support_count(
                month_comparison, signal, regime_variable, "HIGH_REGIME", str(high["raw_direction"])
            )
            month_confounded = bool(flip and (low_support < 2 or high_support < 2))
            monthly_all = month_comparison[
                month_comparison["stock_signal"].eq(signal)
                & month_comparison["regime_variable"].eq(regime_variable)
                & month_comparison["regime_state"].eq("ALL")
            ]["within_date_pair_concordance"].dropna()
            month_range = float(monthly_all.max() - monthly_all.min()) if len(monthly_all) else math.nan
            proxy = bool(split_info[regime_variable]["regime_proxy_for_month_only"])
            rows.append({
                "stock_signal": signal,
                "regime_variable": regime_variable,
                "regime_source_column": split_info[regime_variable]["source_column"],
                "split_threshold_all_60_dates": split_info[regime_variable]["threshold"],
                "split_status": split_info[regime_variable]["split_status"],
                "month_regime_cramers_v": split_info[regime_variable]["month_regime_cramers_v"],
                "regime_proxy_for_month_only": proxy,
                **{f"low_{key}": value for key, value in low.items()},
                **{f"high_{key}": value for key, value in high.items()},
                "conditional_concordance_delta_high_minus_low": delta,
                "absolute_conditional_concordance_difference": abs(delta) if np.isfinite(delta) else math.nan,
                "monthly_all_concordance_range": month_range,
                "low_direction_month_support_count": low_support,
                "high_direction_month_support_count": high_support,
                "same_regime_cross_month_direction_consistency": low_support >= 2 and high_support >= 2,
                "conditional_flip_candidate": flip,
                "month_confounded": month_confounded,
                "regime_more_explanatory_than_month_label": bool(
                    flip
                    and not month_confounded
                    and not proxy
                    and np.isfinite(month_range)
                    and abs(delta) >= month_range
                ),
            })
    return pd.DataFrame(rows).sort_values(
        ["absolute_conditional_concordance_difference", "stock_signal", "regime_variable"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_rank2_6_summary(
    population: pd.DataFrame,
    regime_splits: pd.DataFrame,
    full_summary: pd.DataFrame,
) -> pd.DataFrame:
    rank_population = population[population["s2_rank"].between(2, 6)].copy()
    rank_pairs = build_signal_pairs(rank_population)
    conditional = build_conditional_pairs(rank_pairs, regime_splits)
    rows: list[dict[str, Any]] = []
    full_index = full_summary.set_index(["stock_signal", "regime_variable"])
    for signal in STOCK_SIGNALS:
        for regime_variable in REGIME_VARIABLES:
            relation = conditional[
                conditional["stock_signal"].eq(signal)
                & conditional["regime_variable"].eq(regime_variable)
            ]
            low = _pair_stats(relation[relation["regime_state"].eq("LOW_REGIME")])
            high = _pair_stats(relation[relation["regime_state"].eq("HIGH_REGIME")])
            low_c = float(low["within_date_pair_concordance"])
            high_c = float(high["within_date_pair_concordance"])
            delta = high_c - low_c if np.isfinite(low_c) and np.isfinite(high_c) else math.nan
            full = full_index.loc[(signal, regime_variable)]
            same_direction = bool(
                low["raw_direction"] == full["low_raw_direction"]
                and high["raw_direction"] == full["high_raw_direction"]
                and low["raw_direction"] not in {"NA", "TIE"}
                and high["raw_direction"] not in {"NA", "TIE"}
            )
            support = bool(
                low["informative_date_count"] >= MIN_RANK_STATE_DATES
                and high["informative_date_count"] >= MIN_RANK_STATE_DATES
                and low["pair_count"] >= MIN_RANK_STATE_PAIRS
                and high["pair_count"] >= MIN_RANK_STATE_PAIRS
            )
            rows.append({
                "stock_signal": signal,
                "regime_variable": regime_variable,
                "rank_region": "S2_RANK2_6_FIXED",
                **{f"low_{key}": value for key, value in low.items()},
                **{f"high_{key}": value for key, value in high.items()},
                "conditional_concordance_delta_high_minus_low": delta,
                "absolute_conditional_concordance_difference": abs(delta) if np.isfinite(delta) else math.nan,
                "same_direction_as_full_population": same_direction,
                "rank2_6_support_adequate": support,
                "rank2_6_same_conditional_structure": bool(
                    support and same_direction and np.isfinite(delta) and abs(delta) >= MIN_RANK_CONDITIONAL_GAP
                ),
            })
    return pd.DataFrame(rows).sort_values(
        ["absolute_conditional_concordance_difference", "stock_signal", "regime_variable"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def attach_rank_support(
    full_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> pd.DataFrame:
    keep = rank_summary[[
        "stock_signal", "regime_variable",
        "rank2_6_same_conditional_structure",
        "rank2_6_support_adequate",
        "conditional_concordance_delta_high_minus_low",
        "low_within_date_pair_concordance",
        "high_within_date_pair_concordance",
        "low_pair_count", "high_pair_count",
        "low_informative_date_count", "high_informative_date_count",
    ]].rename(columns={
        "conditional_concordance_delta_high_minus_low": "rank2_6_conditional_delta_high_minus_low",
        "low_within_date_pair_concordance": "rank2_6_low_concordance",
        "high_within_date_pair_concordance": "rank2_6_high_concordance",
        "low_pair_count": "rank2_6_low_pairs",
        "high_pair_count": "rank2_6_high_pairs",
        "low_informative_date_count": "rank2_6_low_dates",
        "high_informative_date_count": "rank2_6_high_dates",
    })
    result = full_summary.merge(keep, on=["stock_signal", "regime_variable"], validate="one_to_one")
    result["formal_local_regime_explanation"] = (
        result["conditional_flip_candidate"].astype(bool)
        & ~result["month_confounded"].astype(bool)
        & ~result["regime_proxy_for_month_only"].astype(bool)
        & result["same_regime_cross_month_direction_consistency"].astype(bool)
        & result["rank2_6_same_conditional_structure"].astype(bool)
    )
    return result.sort_values(
        ["absolute_conditional_concordance_difference", "stock_signal", "regime_variable"],
        ascending=[False, True, True], kind="mergesort",
    ).reset_index(drop=True)


def build_open_close_explanation(
    regime_splits: pd.DataFrame,
    month_comparison: pd.DataFrame,
    full_summary: pd.DataFrame,
) -> pd.DataFrame:
    signal = "d1_open_to_close_return_raw"
    rows: list[dict[str, Any]] = []
    relation_index = full_summary.set_index(["stock_signal", "regime_variable"])
    for regime_variable in REGIME_VARIABLES:
        split = regime_splits[regime_splits["regime_variable"].eq(regime_variable)]
        relation = relation_index.loc[(signal, regime_variable)]
        output: dict[str, Any] = {
            "stock_signal": signal,
            "regime_variable": regime_variable,
            "regime_source_column": relation["regime_source_column"],
            "split_threshold_all_60_dates": relation["split_threshold_all_60_dates"],
            "split_status": relation["split_status"],
            "conditional_flip_candidate": relation["conditional_flip_candidate"],
            "month_confounded": relation["month_confounded"],
            "regime_proxy_for_month_only": relation["regime_proxy_for_month_only"],
            "rank2_6_same_conditional_structure": relation["rank2_6_same_conditional_structure"],
            "formal_local_regime_explanation": relation["formal_local_regime_explanation"],
        }
        same_state_material_flip = False
        for month_label in MONTHS:
            month_key = month_label.lower()
            month_dates = split[split["month"].eq(month_label)]
            for state in ("LOW_REGIME", "HIGH_REGIME"):
                state_key = "low" if state == "LOW_REGIME" else "high"
                state_dates = int(month_dates["regime_state"].eq(state).sum())
                part = month_comparison[
                    month_comparison["stock_signal"].eq(signal)
                    & month_comparison["regime_variable"].eq(regime_variable)
                    & month_comparison["month"].eq(month_label)
                    & month_comparison["regime_state"].eq(state)
                ].iloc[0]
                output[f"{month_key}_{state_key}_date_share"] = (
                    state_dates / len(month_dates) if len(month_dates) else math.nan
                )
                output[f"{month_key}_{state_key}_concordance"] = part["within_date_pair_concordance"]
                output[f"{month_key}_{state_key}_pairs"] = part["pair_count"]
                output[f"{month_key}_{state_key}_informative_dates"] = part["informative_date_count"]
        for state_key in ("low", "high"):
            values = [output[f"{month}_{state_key}_concordance"] for month in ("may", "june", "july")]
            finite = [float(value) for value in values if np.isfinite(value)]
            if any(value >= MATERIAL_HIGH for value in finite) and any(value <= MATERIAL_LOW for value in finite):
                same_state_material_flip = True
        output["same_regime_state_still_materially_flips_across_months"] = same_state_material_flip
        output["open_close_flip_explained"] = bool(
            relation["formal_local_regime_explanation"] and not same_state_material_flip
        )
        output["open_close_local_regime_result"] = (
            "LOCAL_REGIME_DOES_NOT_EXPLAIN_OPEN_CLOSE_FLIP"
            if same_state_material_flip
            else "LOCAL_REGIME_EXPLAINS_OPEN_CLOSE_FLIP"
            if bool(relation["formal_local_regime_explanation"])
            else "NO_COMPLETE_OPEN_CLOSE_EXPLANATION"
        )
        rows.append(output)
    return pd.DataFrame(rows).sort_values(
        ["formal_local_regime_explanation", "regime_variable"],
        ascending=[False, True], kind="mergesort",
    ).reset_index(drop=True)


def build_bootstrap(
    conditional_pairs: pd.DataFrame,
    full_summary: pd.DataFrame,
    all_dates: Sequence[str],
) -> pd.DataFrame:
    eligible = full_summary[full_summary["split_status"].eq("USABLE_FIXED_MEDIAN_SPLIT")].copy()
    eligible = eligible.sort_values(
        ["absolute_conditional_concordance_difference", "stock_signal", "regime_variable"],
        ascending=[False, True, True], kind="mergesort",
    ).head(3)
    dates = np.asarray(sorted(map(str, all_dates)), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    schedule = rng.integers(0, len(dates), size=(BOOTSTRAP_RESAMPLES, len(dates)))
    sampled_dates = dates[schedule]
    rows: list[dict[str, Any]] = []
    for report_rank, relation in enumerate(eligible.itertuples(index=False), start=1):
        pairs = conditional_pairs[
            conditional_pairs["stock_signal"].eq(relation.stock_signal)
            & conditional_pairs["regime_variable"].eq(relation.regime_variable)
        ]
        daily = pairs.groupby(["signal_date", "regime_state"])["t7_high_concordance"].mean().unstack()
        low_map = daily.get("LOW_REGIME", pd.Series(dtype=float)).to_dict()
        high_map = daily.get("HIGH_REGIME", pd.Series(dtype=float)).to_dict()
        low_values = np.asarray([low_map.get(date, math.nan) for date in dates], dtype=float)
        high_values = np.asarray([high_map.get(date, math.nan) for date in dates], dtype=float)
        sampled_low = low_values[schedule]
        sampled_high = high_values[schedule]
        low_counts = np.sum(np.isfinite(sampled_low), axis=1)
        high_counts = np.sum(np.isfinite(sampled_high), axis=1)
        low_means = np.divide(
            np.nansum(sampled_low, axis=1), low_counts,
            out=np.full(BOOTSTRAP_RESAMPLES, np.nan), where=low_counts > 0,
        )
        high_means = np.divide(
            np.nansum(sampled_high, axis=1), high_counts,
            out=np.full(BOOTSTRAP_RESAMPLES, np.nan), where=high_counts > 0,
        )
        delta = high_means - low_means
        valid = delta[np.isfinite(delta)]
        rows.append({
            "report_rank_by_abs_conditional_difference": report_rank,
            "stock_signal": relation.stock_signal,
            "regime_variable": relation.regime_variable,
            "selection_rule": "TOP3_USABLE_BY_ABS_HIGH_MINUS_LOW_CONCORDANCE_ONLY",
            "conditional_flip_candidate": relation.conditional_flip_candidate,
            "formal_local_regime_explanation": relation.formal_local_regime_explanation,
            "sampling_unit": "SIGNAL_DATE",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "valid_resamples": len(valid),
            "low_concordance": relation.low_within_date_pair_concordance,
            "high_concordance": relation.high_within_date_pair_concordance,
            "delta_high_minus_low": relation.conditional_concordance_delta_high_minus_low,
            "delta_p2_5": float(np.quantile(valid, .025)) if len(valid) else math.nan,
            "delta_p50": float(np.quantile(valid, .50)) if len(valid) else math.nan,
            "delta_p97_5": float(np.quantile(valid, .975)) if len(valid) else math.nan,
            "P_delta_gt_zero": float(np.mean(valid > 0)) if len(valid) else math.nan,
            "P_delta_lt_zero": float(np.mean(valid < 0)) if len(valid) else math.nan,
            "low_informative_dates": relation.low_informative_date_count,
            "high_informative_dates": relation.high_informative_date_count,
            "low_pairs": relation.low_pair_count,
            "high_pairs": relation.high_pair_count,
            "sampled_date_schedule_hash_basis": "sorted 60 signal dates",
        })
    return pd.DataFrame(rows)


def build_lineage() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in STOCK_SIGNALS:
        rows.append({
            "field_type": "STOCK_SIGNAL",
            "audit_name": signal,
            "source_column": signal,
            "source_artifact": BRIDGE_AUTHORITY if signal in RANK_SIGNAL_COLUMNS else POPULATION_AUTHORITY,
            "source_semantics": (
                "existing frozen Stage1 daily cross-sectional rank feature"
                if signal in RANK_SIGNAL_COLUMNS
                else "existing audited D1-close path field"
            ),
            "information_available_at": "D1_CLOSE",
            "daily_cross_sectional": signal in RANK_SIGNAL_COLUMNS,
            "transform_in_this_audit": "NONE",
            "used_to_fit_or_change_model": "NO",
        })
    for regime_variable, (source_column, semantics) in REGIME_VARIABLES.items():
        rows.append({
            "field_type": "LOCAL_REGIME_VARIABLE",
            "audit_name": regime_variable,
            "source_column": source_column,
            "source_artifact": REGIME_AUTHORITY,
            "source_semantics": semantics,
            "information_available_at": "D1_CLOSE",
            "daily_cross_sectional": True,
            "transform_in_this_audit": "FIXED_ALL_60_DATE_MEDIAN_SPLIT_LOW_LE_HIGH_GT",
            "used_to_fit_or_change_model": "NO",
        })
    return pd.DataFrame(rows)


def decide_state(summary: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    explainers = summary[summary["formal_local_regime_explanation"].astype(bool)]
    partial = summary[
        summary["conditional_flip_candidate"].astype(bool)
        | summary["absolute_conditional_concordance_difference"].ge(MATERIAL_CONDITIONAL_GAP)
    ]
    usable_regimes = summary.loc[
        summary["split_status"].eq("USABLE_FIXED_MEDIAN_SPLIT"), "regime_variable"
    ].nunique()
    if len(explainers):
        state = "LOCAL_REGIME_EXPLAINS_SIGNAL_FLIP"
    elif len(partial):
        state = "LOCAL_REGIME_PARTIALLY_EXPLAINS_SIGNAL_FLIP"
    elif usable_regimes >= 8:
        state = "LOCAL_REGIME_DOES_NOT_EXPLAIN_SIGNAL_FLIP"
    else:
        state = "INCONCLUSIVE"
    broader = {
        "LOCAL_REGIME_EXPLAINS_SIGNAL_FLIP": "NO_FOR_NOW",
        "LOCAL_REGIME_DOES_NOT_EXPLAIN_SIGNAL_FLIP": "YES_TO_AUDIT",
    }.get(state, "UNCERTAIN")
    audit = {
        "formal_explanation_count": len(explainers),
        "conditional_flip_candidate_count": int(summary["conditional_flip_candidate"].sum()),
        "month_confounded_candidate_count": int(
            (summary["conditional_flip_candidate"].astype(bool) & summary["month_confounded"].astype(bool)).sum()
        ),
        "month_proxy_relationship_count": int(summary["regime_proxy_for_month_only"].sum()),
        "rank2_6_supported_explanation_count": int(summary["rank2_6_same_conditional_structure"].sum()),
        "usable_regime_variable_count": int(usable_regimes),
        "top_explainers": explainers[["stock_signal", "regime_variable"]].to_dict("records"),
    }
    return state, broader, audit


def _fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def build_review(
    summary: pd.DataFrame,
    open_close: pd.DataFrame,
    bootstrap: pd.DataFrame,
    state: str,
    broader: str,
    decision_audit: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> str:
    top = summary.sort_values(
        ["absolute_conditional_concordance_difference", "stock_signal", "regime_variable"],
        ascending=[False, True, True], kind="mergesort",
    ).iloc[0]
    formal = summary[summary["formal_local_regime_explanation"].astype(bool)]
    most_explanatory = (
        f"{formal.iloc[0]['regime_variable']} for {formal.iloc[0]['stock_signal']}"
        if len(formal)
        else f"NONE; largest descriptive split was {top['regime_variable']} for {top['stock_signal']}"
    )
    open_supported = open_close[open_close["open_close_flip_explained"].astype(bool)]
    open_same_flip = int(open_close["same_regime_state_still_materially_flips_across_months"].sum())
    rank_supported = int(summary["rank2_6_same_conditional_structure"].sum())
    lines = [
        "# v004c Regime-Conditional Signal Audit v001",
        "",
        "## Contract",
        "",
        f"- Fixed matrix: {len(STOCK_SIGNALS)} existing stock signals × {len(REGIME_VARIABLES)} existing D1-close local regime variables.",
        "- One preregistered split only: all-60-date median; LOW <= median, HIGH > median.",
        "- Same-date Target7-vs-LOSS information audit only; no model, feature/threshold/group search, interaction construction, score/rank change, or August outcome.",
        f"- Mature population: {input_audit['rows']} rows / {input_audit['dates']} dates; max signal date {input_audit['max_signal_date']}.",
        "",
        "## Q1. Can local board/break ecology explain why the same signal changes direction across months?",
        "",
        f"- {state}. Formal explanation relationships: {decision_audit['formal_explanation_count']}; preregistered conditional-flip candidates: {decision_audit['conditional_flip_candidate_count']}.",
        "",
        "## Q2. Which local regime variable most clearly explains a signal flip?",
        "",
        f"- {most_explanatory}. Largest absolute HIGH-minus-LOW concordance gap: {_fmt(top['absolute_conditional_concordance_difference'])}.",
        "",
        "## Q3. Is the relationship a regime condition or only a May/June/July proxy?",
        "",
        f"- Formal non-month-proxy explanations: {len(formal)}. Month-confounded flip candidates: {decision_audit['month_confounded_candidate_count']}; month-proxy relationships by fixed Cramer's-V gate: {decision_audit['month_proxy_relationship_count']}.",
        "",
        "## Q4. Can local ecology explain the May vs June/July d1_open_to_close_return_raw flip?",
        "",
        f"- {'YES' if len(open_supported) else 'NO'}. Formal open/close explanations: {len(open_supported)}; regime splits where the same state still materially flips by month: {open_same_flip}.",
        "",
        "## Q5. Does the conditional structure remain in the fixed S2 Rank2-6 error region?",
        "",
        f"- {'YES' if rank_supported else 'NO'}. Relationships with the same supported conditional direction in Rank2-6: {rank_supported}.",
        "",
        "## Q6. Is local board/break ecology sufficient for now?",
        "",
        f"- BROADER_MARKET_CONTEXT_NEEDED = {broader}.",
        "",
        "## Reported Bootstrap Relationships",
        "",
        "- " + (
            "; ".join(
                f"#{int(row.report_rank_by_abs_conditional_difference)} {row.stock_signal} × {row.regime_variable}: delta {_fmt(row.delta_high_minus_low)}, 95% CI [{_fmt(row.delta_p2_5)}, {_fmt(row.delta_p97_5)}]"
                for row in bootstrap.itertuples(index=False)
            )
            if len(bootstrap) else "NONE"
        ),
        "",
        f"REGIME_CONDITIONAL_STATE = {state}",
        "",
        f"BROADER_MARKET_CONTEXT_NEEDED = {broader}",
        "",
        "NEXT_ACTION = STOP_AND_REVIEW",
        "",
        "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO",
        "",
    ]
    return "\n".join(lines)


def build_outputs(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    population, daily, input_audit = load_inputs(root)
    regime_splits, split_info = build_regime_splits(daily)
    variable_month = build_regime_month_summary(daily, split_info)
    signal_pairs = build_signal_pairs(population)
    conditional_pairs = build_conditional_pairs(signal_pairs, regime_splits)
    month_comparison = build_month_comparison(population, conditional_pairs, regime_splits)
    conditional_summary = build_conditional_summary(
        population, conditional_pairs, regime_splits, split_info, month_comparison
    )
    rank2_6 = build_rank2_6_summary(population, regime_splits, conditional_summary)
    conditional_summary = attach_rank_support(conditional_summary, rank2_6)
    open_close = build_open_close_explanation(regime_splits, month_comparison, conditional_summary)
    bootstrap = build_bootstrap(
        conditional_pairs, conditional_summary, population["signal_date"].unique()
    )
    lineage = build_lineage()
    state, broader, decision_audit = decide_state(conditional_summary)
    review = build_review(
        conditional_summary, open_close, bootstrap, state, broader, decision_audit, input_audit
    )

    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(variable_month),
        OUTPUT_FILENAMES[1]: _csv_bytes(conditional_pairs),
        OUTPUT_FILENAMES[2]: _csv_bytes(conditional_summary),
        OUTPUT_FILENAMES[3]: _csv_bytes(month_comparison),
        OUTPUT_FILENAMES[4]: _csv_bytes(open_close),
        OUTPUT_FILENAMES[5]: _csv_bytes(rank2_6),
        OUTPUT_FILENAMES[6]: _csv_bytes(bootstrap),
        OUTPUT_FILENAMES[7]: review.encode("utf-8"),
        OUTPUT_FILENAMES[8]: _csv_bytes(lineage),
    }
    audit = {
        **input_audit,
        **decision_audit,
        "regime_conditional_state": state,
        "broader_market_context_needed": broader,
        "next_action": "STOP_AND_REVIEW",
        "output_file_count": len(outputs),
        "conditional_relationship_count": len(conditional_summary),
        "conditional_pair_rows": len(conditional_pairs),
        "rank2_6_relationship_count": len(rank2_6),
        "bootstrap_relationship_count": len(bootstrap),
        "lineage_rows": len(lineage),
    }
    return outputs, audit


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, audit = build_outputs(root_path)
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
    return output_dir, audit

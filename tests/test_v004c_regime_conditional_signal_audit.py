from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_regime_conditional_signal_audit import (
    AUGUST_ASOF,
    MATERIAL_HIGH,
    MATERIAL_LOW,
    OUTPUT_FILENAMES,
    REGIME_VARIABLES,
    STOCK_SIGNALS,
    _cramers_v,
    build_conditional_pairs,
    build_outputs,
    build_regime_splits,
    build_signal_pairs,
    load_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_manifests_and_mature_population_no_august():
    assert STOCK_SIGNALS == (
        "rank_theme_score",
        "rank_active_money_score",
        "rank_log_candidate_base_price",
        "rank_trend_hold_score",
        "d1_open_to_close_return_raw",
        "d1_afternoon_return",
        "d1_close_to_vwap_raw",
        "d1_low_to_close_recovery",
    )
    assert tuple(REGIME_VARIABLES) == (
        "final_limit_up_pool_count",
        "board2_count",
        "board3_count",
        "board4plus_count",
        "max_board_height",
        "promotion_rate",
        "first_break_candidate_count",
        "candidate_cohort_d1_afternoon_median",
        "candidate_cohort_d1_close_vwap_median",
        "candidate_cohort_d1_open_close_median",
        "candidate_cohort_d1_low_close_recovery_median",
    )
    population, daily, audit = load_inputs(ROOT)
    assert (len(population), population["signal_date"].nunique()) == (485, 60)
    assert population.groupby("month").size().to_dict() == {
        "MAY": 146, "JUNE": 173, "JULY": 166,
    }
    assert population["event_id"].is_unique
    assert population["signal_date"].max() == "2026-07-29"
    assert population["signal_date"].lt(AUGUST_ASOF).all()
    assert population["label_available_date"].lt(AUGUST_ASOF).all()
    assert not population[list(STOCK_SIGNALS)].isna().any().any()
    assert len(daily) == 60
    assert audit["model_fits"] == 0
    assert audit["threshold_searches"] == 0
    assert audit["august_signal_outcome_accessed"] == "NO"


def test_regime_split_is_exact_single_median_low_le_high_gt():
    _, daily, _ = load_inputs(ROOT)
    long, info = build_regime_splits(daily)
    assert len(long) == 60 * 11
    for regime_variable, details in info.items():
        part = long[long["regime_variable"].eq(regime_variable)]
        threshold = details["threshold"]
        assert part.loc[part["regime_state"].eq("LOW_REGIME"), "regime_value"].le(threshold).all()
        assert part.loc[part["regime_state"].eq("HIGH_REGIME"), "regime_value"].gt(threshold).all()
        assert part["split_threshold_all_60_dates"].nunique() == 1
        assert details["low_dates"] + details["high_dates"] == 60


def test_pairs_are_same_date_and_fixed_full_matrix():
    population, daily, _ = load_inputs(ROOT)
    splits, _ = build_regime_splits(daily)
    signal_pairs = build_signal_pairs(population)
    counts = {len(frame) for frame in signal_pairs.values()}
    assert len(counts) == 1
    assert next(iter(counts)) > 0
    conditional = build_conditional_pairs(signal_pairs, splits)
    assert conditional.groupby(["stock_signal", "regime_variable"]).size().nunique() == 1
    assert conditional[["stock_signal", "regime_variable"]].drop_duplicates().shape[0] == 8 * 11
    assert conditional["signal_date"].isin(population["signal_date"]).all()
    assert conditional["target_event_id"].isin(population["event_id"]).all()
    assert conditional["loss_event_id"].isin(population["event_id"]).all()
    event_dates = population.set_index("event_id")["signal_date"]
    assert np.array_equal(
        conditional["signal_date"].to_numpy(),
        conditional["target_event_id"].map(event_dates).to_numpy(),
    )
    assert np.array_equal(
        conditional["signal_date"].to_numpy(),
        conditional["loss_event_id"].map(event_dates).to_numpy(),
    )


def test_direction_and_month_proxy_helpers_are_fixed():
    # Perfect association is month-proxy evidence; balanced states are not.
    month = pd.Series(["MAY"] * 4 + ["JUNE"] * 4 + ["JULY"] * 4)
    perfect = pd.Series(["LOW"] * 4 + ["HIGH"] * 8)
    mixed = pd.Series(["LOW", "HIGH"] * 6)
    assert _cramers_v(month, perfect) > .7
    assert _cramers_v(month, mixed) < .1
    assert MATERIAL_HIGH == .55
    assert MATERIAL_LOW == .45


def test_outputs_are_deterministic_complete_and_stop_only():
    first, audit = build_outputs(ROOT)
    second, second_audit = build_outputs(ROOT)
    assert first == second
    assert audit == second_audit
    assert tuple(first) == OUTPUT_FILENAMES
    assert audit["output_file_count"] == 9
    assert audit["conditional_relationship_count"] == 8 * 11
    assert audit["rank2_6_relationship_count"] == 8 * 11
    assert audit["bootstrap_relationship_count"] <= 3
    assert audit["lineage_rows"] == 8 + 11
    assert audit["regime_conditional_state"] in {
        "LOCAL_REGIME_EXPLAINS_SIGNAL_FLIP",
        "LOCAL_REGIME_PARTIALLY_EXPLAINS_SIGNAL_FLIP",
        "LOCAL_REGIME_DOES_NOT_EXPLAIN_SIGNAL_FLIP",
        "INCONCLUSIVE",
    }
    assert audit["broader_market_context_needed"] in {
        "NO_FOR_NOW", "YES_TO_AUDIT", "UNCERTAIN",
    }
    review = first["v004c_regime_conditional_review_v001.md"].decode("utf-8")
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review
    assert "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO" in review
    summary = pd.read_csv(
        pd.io.common.BytesIO(first["v004c_signal_regime_conditional_summary_v001.csv"]),
        encoding="utf-8-sig",
    )
    assert len(summary) == 88
    assert {
        "conditional_flip_candidate", "month_confounded",
        "regime_proxy_for_month_only", "rank2_6_same_conditional_structure",
        "formal_local_regime_explanation",
    }.issubset(summary.columns)
    open_close = pd.read_csv(
        pd.io.common.BytesIO(first["v004c_open_close_regime_explanation_v001.csv"]),
        encoding="utf-8-sig",
    )
    same_state_flip = open_close["same_regime_state_still_materially_flips_across_months"].astype(bool)
    assert open_close.loc[same_state_flip, "open_close_flip_explained"].eq(False).all()
    assert open_close.loc[same_state_flip, "open_close_local_regime_result"].eq(
        "LOCAL_REGIME_DOES_NOT_EXPLAIN_OPEN_CLOSE_FLIP"
    ).all()


def test_module_has_no_learner_or_search_path():
    text = (ROOT / "src/v004c_regime_conditional_signal_audit.py").read_text(encoding="utf-8")
    forbidden = (
        "fit_logistic", "GradientBoosting", "RandomForest", "XGBClassifier",
        "LogisticRegression", "threshold_grid", "feature_subset", "requests.",
        "akshare",
    )
    assert not any(token in text for token in forbidden)

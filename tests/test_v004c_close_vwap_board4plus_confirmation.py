from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_close_vwap_board4plus_confirmation import (
    BOOTSTRAP_RESAMPLES,
    FIXED_REGIME,
    FIXED_SIGNAL,
    FIXED_THRESHOLD_EXPECTED,
    OUTPUT_FILENAMES,
    PERMUTATION_RESAMPLES,
    PRIOR_EXPECTED_DELTA,
    PRIOR_EXPECTED_HIGH,
    PRIOR_EXPECTED_LOW,
    _attach_fixed_state,
    _random_target_mask,
    build_outputs,
    build_reproduction,
    build_selection_permutation,
)
from src.v004c_regime_conditional_signal_audit import (
    REGIME_VARIABLES,
    STOCK_SIGNALS,
    load_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_single_relation_and_prior_selection_family():
    assert FIXED_SIGNAL == "d1_close_to_vwap_raw"
    assert FIXED_REGIME == "board4plus_count"
    assert FIXED_THRESHOLD_EXPECTED == 1.0
    assert len(STOCK_SIGNALS) == 8
    assert len(REGIME_VARIABLES) == 11
    assert len(STOCK_SIGNALS) * len(REGIME_VARIABLES) == 88
    assert PERMUTATION_RESAMPLES >= 10_000
    assert BOOTSTRAP_RESAMPLES >= 10_000


def test_population_split_and_exact_reproduction():
    population, daily, audit = load_inputs(ROOT)
    assert (len(population), population["signal_date"].nunique()) == (485, 60)
    assert population["signal_date"].max() == "2026-07-29"
    assert population["label_available_date"].lt("2026-08-01").all()
    _, fixed_split, split_info = _attach_fixed_state(population, daily)
    assert split_info["threshold"] == 1.0
    assert fixed_split.groupby("regime_state").size().to_dict() == {
        "HIGH_REGIME": 27,
        "LOW_REGIME": 33,
    }
    reproduction, result, _ = build_reproduction(population, fixed_split, ROOT)
    assert result["reproduction_pass"] is True
    assert np.isclose(result["low_concordance"], PRIOR_EXPECTED_LOW, atol=1e-12)
    assert np.isclose(result["high_concordance"], PRIOR_EXPECTED_HIGH, atol=1e-12)
    assert np.isclose(result["delta"], PRIOR_EXPECTED_DELTA, atol=1e-12)
    indexed = reproduction.set_index("regime_state")
    assert indexed.loc["LOW_REGIME", "pair_count"] == 170
    assert indexed.loc["HIGH_REGIME", "pair_count"] == 151


def test_random_permutation_preserves_within_date_class_count():
    mask = _random_target_mask(np.random.default_rng(7), 200, 9, 3)
    assert mask.shape == (200, 9)
    assert np.all(mask.sum(axis=1) == 3)


def test_selection_permutation_is_deterministic_and_exact_88_max_family():
    population, daily, _ = load_inputs(ROOT)
    first, audit = build_selection_permutation(
        population, daily, abs(PRIOR_EXPECTED_DELTA), resamples=100, seed=123
    )
    second, second_audit = build_selection_permutation(
        population, daily, abs(PRIOR_EXPECTED_DELTA), resamples=100, seed=123
    )
    pd.testing.assert_frame_equal(first, second)
    assert audit == second_audit
    assert first["family_relationship_count"].eq(88).all()
    assert first["same_date_class_counts_preserved"].eq("YES").all()
    assert audit["permutation_counts_preserved"] is True
    assert 0 <= audit["selection_adjusted_p"] <= 1


def test_all_outputs_are_deterministic_and_stop_only():
    first, audit = build_outputs(
        ROOT, permutation_resamples=200, bootstrap_resamples=200
    )
    second, second_audit = build_outputs(
        ROOT, permutation_resamples=200, bootstrap_resamples=200
    )
    assert first == second
    assert audit == second_audit
    assert tuple(first) == OUTPUT_FILENAMES
    assert audit["output_file_count"] == 10
    assert audit["model_fits"] == 0
    assert audit["threshold_searches"] == 0
    assert audit["second_relations_selected"] == 0
    assert audit["august_signal_outcome_accessed"] == "NO"
    assert audit["single_relation_state"] in {
        "SINGLE_RELATION_CONFIRMED",
        "SINGLE_RELATION_SUPPORTED_BUT_SELECTION_RISK",
        "SINGLE_RELATION_UNSTABLE",
        "SINGLE_RELATION_NOT_SUPPORTED",
    }
    review = first[OUTPUT_FILENAMES[-1]].decode("utf-8")
    assert "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO" in review
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review


def test_no_learner_threshold_search_or_august_data_path():
    text = (ROOT / "src/v004c_close_vwap_board4plus_confirmation.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "fit_logistic", "LogisticRegression", "GradientBoosting", "RandomForest",
        "XGBClassifier", "threshold_grid", "feature_subset", "2026-08-",
        "board3_count as alternative", "max_board_height as alternative",
    )
    assert not any(token in text for token in forbidden)


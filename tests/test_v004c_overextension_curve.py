from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004a import DEFAULT_TARGET_COLUMN
from src.v004c_overextension_curve import (
    CONTROL_MODEL_ID,
    CONTROL_PREDICTOR_COUNT,
    CURVE_FEATURE_COLUMNS,
    CURVE_MODEL_ID,
    CURVE_PREDICTOR_COUNT,
    EXTRA_SCALING,
    FEATURE_SELECTION,
    HYPERPARAMETER_SEARCH,
    INITIAL_TRAIN_DATES,
    L2,
    NEW_RAW_FEATURES,
    POSITIVE_WEIGHT,
    QUADRATIC_COLUMNS,
    QUADRATIC_SPECS,
    USES_BOARD3,
    USES_MARKET_REGIME,
    USES_PAIRWISE,
    USES_PCA,
    USES_SAME_TIER,
    USES_THEME_EXTENSION,
    add_quadratic_terms,
    assert_curve_contract,
    build_outputs,
    load_frozen_opportunity_buckets,
    valid_turning_point,
)
from src.v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    assert_authoritative_universe,
    load_authoritative_input,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark():
    outputs, context = build_outputs(ROOT)
    return outputs, context


def test_authoritative_universe_exact_and_july_sentinel() -> None:
    frame = load_authoritative_input(ROOT)
    assert_authoritative_universe(frame)
    assert len(frame) == 319
    assert frame["signal_date"].nunique() == 39
    assert int(frame["signal_date"].str.startswith("2026-05").sum()) == 146
    assert frame.loc[
        frame["signal_date"].str.startswith("2026-05"), "signal_date"
    ].nunique() == 18
    assert int(frame["signal_date"].str.startswith("2026-06").sum()) == 173
    assert frame.loc[
        frame["signal_date"].str.startswith("2026-06"), "signal_date"
    ].nunique() == 21
    assert int((frame["board_streak_before_break"] == 2).sum()) == 261
    assert int((frame["board_streak_before_break"] == 3).sum()) == 58
    changed = frame.copy()
    changed.loc[0, "signal_date"] = "2026-07-01"
    with pytest.raises(RuntimeError, match="July sentinel"):
        assert_authoritative_universe(changed)


def test_exact_feature_contract_has_only_five_predeclared_squares() -> None:
    assert_curve_contract()
    assert len(FROZEN_FEATURE_COLUMNS) == CONTROL_PREDICTOR_COUNT == 18
    assert QUADRATIC_COLUMNS == [
        "quad_close_ma10",
        "quad_low_ma10",
        "quad_trend_hold",
        "quad_close_vwap",
        "quad_log_base_price",
    ]
    assert CURVE_FEATURE_COLUMNS == [*FROZEN_FEATURE_COLUMNS, *QUADRATIC_COLUMNS]
    assert len(CURVE_FEATURE_COLUMNS) == CURVE_PREDICTOR_COUNT == 23
    assert "rank_total_score_squared" not in CURVE_FEATURE_COLUMNS


def test_quadratic_formula_is_uncentered_x_squared() -> None:
    row = {column: 0.5 for column in FROZEN_FEATURE_COLUMNS}
    row.update({
        "rank_d1_close_ma10_pct": 0.10,
        "rank_d1_low_ma10_pct": 0.25,
        "rank_trend_hold_score": 0.50,
        "rank_d1_close_vwap_pct": 0.75,
        "rank_log_candidate_base_price": 1.00,
    })
    result = add_quadratic_terms(pd.DataFrame([row])).iloc[0]
    for quadratic, linear, _ in QUADRATIC_SPECS:
        assert result[quadratic] == pytest.approx(result[linear] ** 2)
    assert result["quad_close_ma10"] == pytest.approx(0.01)


def test_frozen_training_controls_and_no_extra_logic() -> None:
    assert L2 == pytest.approx(0.30)
    assert POSITIVE_WEIGHT == pytest.approx(1.50)
    assert INITIAL_TRAIN_DATES == 18
    assert HYPERPARAMETER_SEARCH is False
    assert FEATURE_SELECTION is False
    assert NEW_RAW_FEATURES is False
    assert EXTRA_SCALING is False
    assert USES_PCA is False
    assert USES_PAIRWISE is False
    assert USES_BOARD3 is False
    assert USES_MARKET_REGIME is False
    assert USES_THEME_EXTENSION is False
    assert USES_SAME_TIER is False


def test_turning_point_formula_and_invalid_handling() -> None:
    assert valid_turning_point(1.0, -1.0) == pytest.approx(0.5)
    assert np.isnan(valid_turning_point(1.0, 1.0))
    assert np.isnan(valid_turning_point(-1.0, -1.0))
    assert np.isnan(valid_turning_point(3.0, -1.0))
    assert np.isnan(valid_turning_point(np.nan, -1.0))


def test_control_parity_is_exact_and_precedes_curve(benchmark) -> None:
    _, context = benchmark
    parity = context["control_parity"]
    assert parity["identity_exact"] is True
    assert parity["score_exact_at_frozen_precision"] is True
    assert parity["rank_exact"] is True
    assert parity["metric_exact_at_frozen_precision"] is True
    assert parity["pass"] is True
    assert parity["previous"]["Rank1"] == pytest.approx(0.0283930560164)
    assert parity["previous"]["Rank2"] == pytest.approx(0.0320306491097)
    assert parity["previous"]["Rank3"] == pytest.approx(-0.00302540178647)
    assert parity["previous"]["Top2"] == pytest.approx(0.0302118525631)
    assert parity["previous"]["Top3"] == pytest.approx(0.01853903678)
    assert parity["previous"]["cross_threshold"] == pytest.approx(0.487123907143)


def test_warmup_june_folds_and_chronology(benchmark) -> None:
    _, context = benchmark
    folds = context["coefficients"]
    assert len(folds) == 21
    assert folds.iloc[0]["train_dates"] == 18
    assert folds.iloc[-1]["train_dates"] == 38
    assert folds["test_date"].str.startswith("2026-06").all()
    assert folds["train_before_test"].all()
    assert (folds["train_end"] < folds["test_date"]).all()
    assert folds["feature_count"].eq(23).all()
    assert folds["l2"].eq(0.30).all()
    assert folds["positive_weight"].eq(1.50).all()


def test_oof_identity_scores_and_ranks_are_deterministic(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    assert len(oof) == 173
    assert oof["signal_date"].nunique() == 21
    assert oof["control_score"].between(0, 1).all()
    assert oof["curve_score"].between(0, 1).all()
    assert oof["rank_delta"].equals(oof["curve_rank"] - oof["control_rank"])
    for _, day in oof.groupby("signal_date", sort=True):
        assert sorted(day["control_rank"].tolist()) == list(range(1, len(day) + 1))
        assert sorted(day["curve_rank"].tolist()) == list(range(1, len(day) + 1))
        expected = day.sort_values(
            ["curve_score", "event_id"],
            ascending=[False, True],
            kind="mergesort",
        )["event_id"].tolist()
        actual = day.sort_values("curve_rank")["event_id"].tolist()
        assert actual == expected


def test_quadratics_are_audit_visible_but_no_audit_count_is_a_predictor(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    assert set(QUADRATIC_COLUMNS).issubset(oof.columns)
    assert oof["extreme_strength_count"].between(0, 5).all()
    assert "extreme_strength_count" not in CURVE_FEATURE_COLUMNS


def test_opportunity_bucket_assignment_reuses_frozen_transfer(benchmark) -> None:
    _, context = benchmark
    mapping = load_frozen_opportunity_buckets(
        ROOT, context["curve_oof"]["signal_date"].unique()
    )
    assert mapping == context["bucket_map"]
    assert list(mapping.values()).count("LOW") == 7
    assert list(mapping.values()).count("MID") == 7
    assert list(mapping.values()).count("HIGH") == 7
    assert mapping["2026-06-03"] == "LOW"
    assert mapping["2026-06-23"] == "HIGH"


def test_coefficient_and_curvature_audit_contract(benchmark) -> None:
    _, context = benchmark
    coefficients = context["coefficients"]
    audit = context["curvature_audit"]
    assert len(coefficients) == 21
    assert len(audit) == 5
    assert set(audit["feature"]) == set(QUADRATIC_COLUMNS)
    for quadratic, linear, slug in QUADRATIC_SPECS:
        assert f"beta_{linear}" in coefficients
        assert f"beta_{quadratic}" in coefficients
        assert f"linear_beta_{slug}" in coefficients
        assert f"quadratic_beta_{slug}" in coefficients
        assert f"turning_point_{slug}" in coefficients


def test_sample_weight_and_target_are_frozen(benchmark) -> None:
    _, context = benchmark
    samples = context["samples"]
    assert "tail_weight" in samples
    expected_tail = np.select(
        [samples["raw_repair_return"].ge(0.12), samples["raw_repair_return"].ge(0.10)],
        [2.0, 1.5],
        default=1.0,
    )
    assert np.array_equal(samples["tail_weight"].to_numpy(float), expected_tail)
    assert np.array_equal(
        samples[DEFAULT_TARGET_COLUMN].astype(int).to_numpy(),
        samples["raw_repair_return"].ge(0.07).astype(int).to_numpy(),
    )


def test_outcome_changes_cannot_change_control_or_curve_x(benchmark) -> None:
    _, context = benchmark
    samples = context["samples"].sort_values("event_id").reset_index(drop=True)
    changed = samples.copy()
    changed["raw_repair_return"] *= -1
    changed["target7"] = 1 - changed["target7"]
    changed["capped_opportunity_return_7"] = np.minimum(
        changed["raw_repair_return"], 0.07
    )
    original_x = add_quadratic_terms(samples)[CURVE_FEATURE_COLUMNS].to_numpy(float)
    changed_x = add_quadratic_terms(changed)[CURVE_FEATURE_COLUMNS].to_numpy(float)
    assert np.array_equal(original_x, changed_x)


def test_output_set_and_byte_determinism(benchmark) -> None:
    first, _ = benchmark
    second, _ = build_outputs(ROOT)
    assert first.keys() == second.keys()
    assert len(first) == 7
    assert all(first[name] == second[name] for name in first)


def test_formal_status_is_closed_vocabulary(benchmark) -> None:
    _, context = benchmark
    status = context["status"]
    assert status["overextension_curvature_signal"] in {"STRONG", "PARTIAL", "ABSENT"}
    assert status["non_monotonic_model_hypothesis"] in {
        "SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"
    }
    assert status["ready_for_july_oot"] in {"YES", "NO"}
    assert context["summaries"][CONTROL_MODEL_ID]["date_count"] == 21
    assert context["summaries"][CURVE_MODEL_ID]["date_count"] == 21


def test_fixed_benchmark_regression_and_stop_decision(benchmark) -> None:
    _, context = benchmark
    control = context["summaries"][CONTROL_MODEL_ID]
    curve = context["summaries"][CURVE_MODEL_ID]
    assert control["Rank3_mean"] == pytest.approx(-0.0030254017864726535)
    assert control["Top3_mean"] == pytest.approx(0.01853903678000977)
    assert curve["Rank1_mean"] == pytest.approx(0.0319291093776)
    assert curve["Rank2_mean"] == pytest.approx(0.0358410753056)
    assert curve["Rank3_mean"] == pytest.approx(-0.0107392053214)
    assert curve["Top2_mean"] == pytest.approx(0.0338850923416)
    assert curve["Top3_mean"] == pytest.approx(0.01853903678000977)
    assert curve["cross_threshold_concordance"] == pytest.approx(0.48553809738)
    assert context["curvature_audit"]["negative_fold_pct"].eq(0.0).all()
    assert context["status"]["overextension_curvature_signal"] == "ABSENT"
    assert context["status"]["non_monotonic_model_hypothesis"] == "NOT_SUPPORTED"
    assert context["status"]["ready_for_july_oot"] == "NO"

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import src.v004c_reduced7f_training_spec_sanity as audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "research" / audit.OUTPUT_DIRNAME


def test_exact_four_specs_and_preregistered_seven_features_only() -> None:
    audit.assert_experiment_contract()
    assert audit.SPEC_ORDER == (
        "S0_CURRENT_CONTROL",
        "S1_NO_TAIL_L2_030",
        "S2_NO_TAIL_L2_010",
        "S3_NO_TAIL_L2_003",
    )
    assert tuple(audit.REDUCED_FEATURE_COLUMNS) == (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    assert [audit.SPECIFICATIONS[name]["l2"] for name in audit.SPEC_ORDER] == [.30, .30, .10, .03]
    source = inspect.getsource(audit)
    assert source.count("fit_logistic_l2_weighted(") == 1
    for forbidden in ("GradientBoosting", "RandomForest", "XGBoost", "pairwise", "reranker"):
        assert forbidden.lower() not in source.lower()


def test_bridge_maturity_and_no_august_signal_date_outcomes() -> None:
    frame, _ = audit.load_bridge(ROOT)
    mature = frame.label_available_date.lt("2026-08-01")
    assert (len(frame), frame.signal_date.nunique()) == (497, 62)
    assert (int(mature.sum()), frame.loc[mature, "signal_date"].nunique()) == (485, 60)
    assert not frame.signal_date.ge("2026-08-01").any()
    outcomes = ["target7", "positive_non_target", "loss", "severe_loss", "raw_repair_return", "capped_return_7"]
    assert frame.loc[~mature, outcomes].isna().all().all()


def test_tail_weight_and_date_weight_semantics_are_exact() -> None:
    rows = pd.DataFrame({
        "signal_date": ["2026-05-01"] * 4,
        "target7": [0, 1, 1, 1],
        "raw_repair_return": [.01, .07, .10, .12],
    })
    _, s0 = audit.build_sample_weight(rows, "S0_CURRENT_CONTROL")
    _, s1 = audit.build_sample_weight(rows, "S1_NO_TAIL_L2_030")
    assert np.allclose(s0, [.25, .375, .5625, .75])
    assert np.allclose(s1, [.25, .375, .375, .375])


def test_s0_reproduces_previously_frozen_reduced7f_exactly() -> None:
    context = audit.analyze(ROOT)
    previous = __import__("json").loads((ROOT / audit.PREVIOUS_REDUCED_SPEC).read_text(encoding="utf-8"))
    expected = np.asarray([previous["intercept"]] + [previous["coefficients"][f] for f in audit.REDUCED_FEATURE_COLUMNS])
    assert np.allclose(context["betas"][audit.CONTROL_SPEC], expected, rtol=0, atol=1e-12)


def test_training_fit_metrics_and_auc_orientation() -> None:
    context = audit.analyze(ROOT)
    fit = context["training_fit"].set_index("spec")
    assert set(fit.index) == set(audit.SPEC_ORDER)
    assert (fit.feature_count == 7).all()
    assert (fit.fitted_weighted_logloss < fit.weighted_constant_only_logloss).all()
    assert (fit.logloss_improvement > 0).all()
    assert fit.loc["S2_NO_TAIL_L2_010", "coefficient_l2_norm"] > fit.loc["S1_NO_TAIL_L2_030", "coefficient_l2_norm"]
    assert fit.loc["S2_NO_TAIL_L2_010", "auc_target7_vs_loss"] > fit.loc[audit.CONTROL_SPEC, "auc_target7_vs_loss"]


def test_fixed_and_expanding_temporal_contract_has_zero_leakage() -> None:
    context = audit.analyze(ROOT)
    leakage = context["leakage"]
    assert leakage["self_or_future_signal_leakage_rows"] == 0
    assert leakage["current_or_future_label_leakage_rows"] == 0
    assert leakage["august_signal_date_rows_accessed"] == 0
    predictions = context["temporal_predictions"]
    assert set(predictions.spec) == set(audit.SPEC_ORDER)
    identity_counts = predictions.groupby(["prediction_set", "signal_date", "event_id"]).spec.nunique()
    assert identity_counts.eq(4).all()
    fold_b = predictions[predictions.prediction_set.eq("FOLD_B_MAY_JUNE_TO_JULY")]
    assert fold_b.train_rows.nunique() == 1
    assert fold_b.train_dates.nunique() == 1
    assert fold_b.train_end_signal_date.nunique() == 1
    assert fold_b.max_train_label_available_date.nunique() == 1


def test_rankwise_oracle_and_top3_metrics_are_date_equal() -> None:
    context = audit.analyze(ROOT)
    rankwise = context["rankwise"]
    assert set(rankwise.selection) == {"UNIVERSE", "RANK1", "RANK2", "RANK3", "TOP3"}
    assert set(rankwise.population_scope) == {"ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY"}
    pooled = rankwise[(rankwise.period == "MAY_JUNE_JULY_MATURE") & (rankwise.spec == audit.CONTROL_SPEC) & (rankwise.population_scope == "ALL_CANDIDATES")]
    assert pooled.dates.unique().tolist() == [60]
    assert rankwise[rankwise.population_scope.eq("BOARD3_ONLY")].support_warning.eq("LOW_N").any()
    oracle = context["oracle"]
    assert (oracle.oracle_top3_capped_return >= oracle.actual_top3_capped_return - 1e-15).all()
    assert np.allclose(oracle.model_recovered_gap, oracle.actual_top3_capped_return - oracle.universe_capped_return)


def test_temporal_summary_contains_complete_rank1_rank2_rank3_and_top3() -> None:
    context = audit.analyze(ROOT)
    summary = context["temporal_summary"]
    assert set(summary.period) == set(audit.TEMPORAL_PERIOD_ORDER)
    assert set(summary.population_scope) == {"ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY"}
    assert summary[summary.population_scope.eq("BOARD3_ONLY")].support_warning.eq("LOW_N").any()
    for prefix in ("rank1", "rank2", "rank3"):
        for metric in ("capped_return", "target7_rate", "loss_rate"):
            assert f"{prefix}_{metric}" in summary.columns
    assert {"top3_capped_return", "top3_target7_rate", "top3_loss_rate", "top3_excess_vs_universe"}.issubset(summary.columns)


def test_paired_daily_bootstrap_samples_signal_dates_deterministically() -> None:
    first = audit.analyze(ROOT)["paired"]
    second = audit.analyze(ROOT)["paired"]
    pd.testing.assert_frame_equal(first, second)
    summary = first[first.row_type.eq("SUMMARY")]
    assert set(summary.challenger) == set(audit.CHALLENGER_SPECS)
    assert set(summary.control) == {audit.CONTROL_SPEC}
    assert set(summary.bootstrap_resamples.astype(int)) == {20_000}
    assert set(summary.bootstrap_seed.astype(int)) == {audit.BOOTSTRAP_SEED}


def test_outputs_exist_and_review_has_exactly_eight_questions() -> None:
    for name in audit.OUTPUT_FILENAMES:
        assert (OUTPUT_DIR / name).is_file()
    review = (OUTPUT_DIR / audit.OUTPUT_FILENAMES[-1]).read_text(encoding="utf-8")
    assert sum(review.count(f"## Q{i}.") for i in range(1, 9)) == 8
    assert "## Q9." not in review
    assert "S2_NO_TAIL_L2_010" in review
    assert "TRAINING_SPEC_STATE = TRAINING_SPEC_REPAIR_SUPPORTED" in review
    assert "NEXT_ACTION = FREEZE_ONE_REPAIRED_7F_SPEC" in review

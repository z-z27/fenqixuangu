from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import src.v004c_recency_weighting_diagnostic as audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "research" / audit.OUTPUT_DIRNAME


def test_exact_two_versions_and_frozen_s2_contract() -> None:
    audit.assert_experiment_contract()
    assert audit.VERSION_ORDER == ("HISTORICAL_EQUAL", "RECENT_PRIORITY_2X")
    assert audit.VERSION_MULTIPLIERS == {
        "HISTORICAL_EQUAL": {5: 1.0, 6: 1.0},
        "RECENT_PRIORITY_2X": {5: 0.5, 6: 1.0},
    }
    assert audit.L2 == .10
    assert audit.POSITIVE_WEIGHT == 1.50
    assert tuple(audit.REDUCED_FEATURE_COLUMNS) == (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    source = inspect.getsource(audit)
    assert source.count("fit_logistic_l2_weighted(") == 1
    for forbidden in (
        "GradientBoosting", "RandomForest", "XGBoost", "pairwise", "reranker",
        "threshold search", "August model",
    ):
        assert forbidden.lower() not in source.lower()


def test_strict_training_and_july_population_boundaries() -> None:
    frame, _ = audit.load_bridge(ROOT)
    train, july = audit.build_populations(frame)
    assert (len(train), train.signal_date.nunique()) == (307, 37)
    assert train.signal_date.max() == "2026-06-26"
    assert train.label_available_date.max() == "2026-06-30"
    assert (len(july), july.signal_date.nunique()) == (166, 21)
    assert july.signal_date.min() == "2026-07-01"
    assert july.signal_date.max() == "2026-07-29"
    assert july.label_available_date.lt("2026-08-01").all()
    assert not frame.signal_date.ge("2026-08-01").any()


def test_month_multiplier_only_multiplies_existing_s2_weight() -> None:
    frame, _ = audit.load_bridge(ROOT)
    train, _ = audit.build_populations(frame)
    equal, equal_audit = audit.build_version_weight(train, "HISTORICAL_EQUAL")
    recent, recent_audit = audit.build_version_weight(train, "RECENT_PRIORITY_2X")
    months = pd.to_datetime(train.signal_date).dt.month.to_numpy()
    assert np.allclose(recent[months == 5], equal[months == 5] * .5)
    assert np.allclose(recent[months == 6], equal[months == 6])
    assert set(equal_audit.month_multiplier) == {1.0}
    assert set(recent_audit.loc[recent_audit.month.eq(5), "month_multiplier"]) == {.5}
    assert set(recent_audit.loc[recent_audit.month.eq(6), "month_multiplier"]) == {1.0}


def test_equal_version_reproduces_archived_s2_fold_b_exactly() -> None:
    context = audit.analyze(ROOT)
    assert context["parity"]["archived_equal_score_max_abs_error"] <= 1e-12
    assert context["parity"]["archived_equal_coefficient_max_abs_error"] <= 1e-12
    assert context["parity"]["archived_equal_rank_mismatch_count"] == 0


def test_prediction_lock_is_outcome_immutable() -> None:
    frame, _ = audit.load_bridge(ROOT)
    train, july = audit.build_populations(frame)
    betas, _ = audit.fit_versions(train)
    first = audit.build_prediction_lock(july, betas)
    perturbed = july.copy()
    perturbed["target7"] = 1 - perturbed["target7"]
    perturbed["loss"] = 1 - perturbed["loss"]
    perturbed["raw_repair_return"] = -999.0
    perturbed["capped_return_7"] = -999.0
    second = audit.build_prediction_lock(perturbed, betas)
    pd.testing.assert_frame_equal(first, second)


def test_rank_rule_and_complete_july_identity() -> None:
    context = audit.analyze(ROOT)
    lock = context["prediction_lock"]
    assert set(lock.version) == set(audit.VERSION_ORDER)
    assert lock.groupby("version").size().eq(166).all()
    assert lock.groupby(["version", "signal_date"]).model_rank.min().eq(1).all()
    for _, day in lock.groupby(["version", "signal_date"]):
        ordered = day.sort_values(["model_score", "event_id"], ascending=[False, True], kind="mergesort")
        assert ordered.model_rank.astype(int).tolist() == list(range(1, len(day) + 1))


def test_outputs_cover_required_metrics_and_rank_changes() -> None:
    context = audit.analyze(ROOT)
    summary = context["summary"]
    assert set(summary.version) == set(audit.VERSION_ORDER)
    required = {
        "top1_average_capped_return", "top2_average_capped_return",
        "top3_average_capped_return", "top3_target7_rate", "top3_loss_rate",
        "top3_severe_loss_rate", "top3_all_hit_rate", "zero_hit_day_rate",
        "winner_capture_mean_eligible_dates", "universe_capped_return",
        "worst_daily_top3_capped_return", "negative_dates",
    }
    assert required.issubset(summary.columns)
    rankwise = context["rankwise"]
    assert set(rankwise.selection) == {"RANK1", "RANK2", "RANK3", "TOP1", "TOP2", "TOP3"}
    changes = context["rank_changes"]
    assert len(changes) == 166
    assert {"equal_rank", "recent_priority_rank", "rank_delta_recent_minus_equal", "entered_top3", "left_top3"}.issubset(changes.columns)


def test_no_july_refit_or_model_selection_and_review_contract() -> None:
    source = inspect.getsource(audit)
    assert "EXPANDING" not in source
    assert "JULY_MODEL_SELECTION = NO" in source
    context = audit.analyze(ROOT)
    review = audit.render_review(context, context["state"], context["diagnostics"])
    assert sum(review.count(f"## Q{i}.") for i in range(1, 6)) == 5
    assert "## Q6." not in review
    assert "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO" in review
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review
    assert context["state"] in {
        "RECENCY_SIGNAL_SUPPORTED", "RECENCY_SIGNAL_NOT_SUPPORTED", "MIXED", "INCONCLUSIVE"
    }


def test_deterministic_output_bytes_and_required_files_exist_after_run() -> None:
    context = audit.analyze(ROOT)
    assert audit.build_outputs(context) == audit.build_outputs(context)
    for name in audit.OUTPUT_FILENAMES:
        assert (OUTPUT_DIR / name).is_file()


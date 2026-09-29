from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.v004c_reduced7f_stage1_temporal_validation as audit
from src.v004c_v4a_architecture_transfer import FROZEN_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "research" / audit.OUTPUT_DIRNAME


def test_single_preregistered_challenger_and_exact_seven_features() -> None:
    assert audit.REDUCED_FEATURE_COLUMNS == (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    assert len(audit.REDUCED_FEATURE_COLUMNS) == 7
    assert set(audit.REDUCED_FEATURE_COLUMNS).issubset(FROZEN_FEATURE_COLUMNS)
    assert tuple(audit.MODEL_FEATURES) == (
        audit.BASELINE_MODEL, audit.CHALLENGER_MODEL,
    )
    source = inspect.getsource(audit)
    assert source.count("fit_logistic_l2_weighted(") == 1
    assert "GradientBoosting" not in source
    assert "RandomForest" not in source
    assert "XGBoost" not in source
    assert "pairwise" not in source.lower()


def test_bridge_and_mature_population_parity_without_august_signal_outcome() -> None:
    frame, spec = audit.load_bridge(ROOT)
    assert (len(frame), frame.signal_date.nunique()) == (497, 62)
    mature = frame.label_available_date.lt("2026-08-01")
    assert (int(mature.sum()), frame.loc[mature, "signal_date"].nunique()) == (485, 60)
    assert not frame.signal_date.ge("2026-08-01").any()
    outcomes = [
        "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
    ]
    assert frame.loc[~mature, outcomes].isna().all().all()
    assert spec["features"] == list(FROZEN_FEATURE_COLUMNS)


def test_current_baseline_is_reused_and_never_fit() -> None:
    context = audit.analyze(ROOT, "test-commit")
    leakage = context["leakage"]
    assert leakage["current18f_fit_count"] == 0
    assert (leakage["june_rows"], leakage["june_dates"]) == (155, 17)
    assert (leakage["july_mature_rows"], leakage["july_mature_dates"]) == (166, 21)
    predictions = context["predictions"]
    june = pd.read_csv(ROOT / audit.STRICT_JUNE_PATH, encoding="utf-8-sig", dtype={"event_id": str})
    observed = predictions[
        predictions.fold_type.eq("FOLD_A_MAY_TO_JUNE")
        & predictions.model.eq(audit.BASELINE_MODEL)
    ][["event_id", "model_score", "model_rank"]]
    joined = observed.merge(
        june[["event_id", "stage1_score", "stage1_rank"]], on="event_id", validate="one_to_one"
    )
    assert np.allclose(joined.model_score, joined.stage1_score, rtol=0, atol=1e-12)
    assert np.array_equal(joined.model_rank.to_numpy(int), joined.stage1_rank.to_numpy(int))


def test_fixed_fold_training_contract_and_no_temporal_leakage() -> None:
    context = audit.analyze(ROOT, "test-commit")
    coefficients = context["coefficients"]
    reduced = coefficients[
        coefficients.model.eq(audit.CHALLENGER_MODEL)
        & coefficients.feature.eq("__INTERCEPT__")
    ]
    fold_a = reduced[reduced.fold_type.eq("FOLD_A_MAY_TO_JUNE")].iloc[0]
    fold_b = reduced[reduced.fold_type.eq("FOLD_B_MAY_JUNE_TO_JULY")].iloc[0]
    assert (fold_a.train_rows, fold_a.train_dates) == (146, 18)
    assert fold_a.train_end_signal_date == "2026-05-29"
    assert fold_a.max_train_label_available_date == "2026-06-02"
    assert (fold_b.train_rows, fold_b.train_dates) == (307, 37)
    assert fold_b.train_end_signal_date == "2026-06-26"
    assert fold_b.max_train_label_available_date == "2026-06-30"
    assert context["leakage"]["self_or_future_signal_leakage_rows"] == 0
    assert context["leakage"]["current_test_label_leakage_rows"] == 0
    assert context["leakage"]["august_signal_rows_accessed"] == 0


def test_expanding_oof_uses_18_date_gate_and_records_seven_coefficients() -> None:
    context = audit.analyze(ROOT, "test-commit")
    predictions = context["predictions"]
    expanding = predictions[
        predictions.fold_type.eq("EXPANDING_OOF")
        & predictions.model.eq(audit.CHALLENGER_MODEL)
    ]
    assert expanding.signal_date.nunique() == 38
    assert expanding.signal_date.min() == "2026-06-03"
    assert expanding.train_dates.min() >= audit.MIN_TRAIN_SIGNAL_DATES
    coefficients = context["coefficients"]
    reduced = coefficients[
        coefficients.fold_type.eq("EXPANDING_OOF")
        & coefficients.model.eq(audit.CHALLENGER_MODEL)
        & coefficients.feature.ne("__INTERCEPT__")
    ]
    assert set(reduced.feature) == set(audit.REDUCED_FEATURE_COLUMNS)
    assert reduced.groupby("test_date").size().eq(7).all()
    assert reduced.coefficient_median.notna().all()
    assert reduced.coefficient_iqr.notna().all()
    assert reduced.sign_flip_count.notna().all()


def test_date_equal_metrics_and_bootstrap_by_date() -> None:
    context = audit.analyze(ROOT, "test-commit")
    daily = context["daily"]
    row = audit.summary_row(context, "FOLD_A_MAY_TO_JUNE", audit.CHALLENGER_MODEL)
    selected = daily[
        daily.fold_type.eq("FOLD_A_MAY_TO_JUNE")
        & daily.population_scope.eq("ALL_CANDIDATES")
        & daily.model.eq(audit.CHALLENGER_MODEL)
    ]
    assert np.isclose(row.target7_within_date_auc, selected.target7_within_date_auc.mean())
    assert (daily.top5_selected_rows <= daily.candidate_count).all()
    bootstrap = context["bootstrap"]
    assert set(bootstrap.sampling_unit) == {"signal_date"}
    assert set(bootstrap.resamples) == {20_000}
    assert set(bootstrap.challenger) == {audit.CHALLENGER_MODEL}
    assert set(bootstrap.baseline) == {audit.BASELINE_MODEL}


def test_board_outputs_and_board3_low_n_warning() -> None:
    context = audit.analyze(ROOT, "test-commit")
    summary = context["summary"]
    assert set(summary.population_scope) == set(audit.SCOPE_ORDER)
    board3 = summary[summary.population_scope.eq("BOARD3_ONLY")]
    assert board3.support_warning.eq("LOW_N").all()


def test_final_fit_is_after_validation_and_exact_mature_population() -> None:
    context = audit.analyze(ROOT, "test-commit")
    spec = context["final_spec"]
    assert spec["feature_order"] == list(audit.REDUCED_FEATURE_COLUMNS)
    assert (spec["training_rows"], spec["training_dates"]) == (485, 60)
    assert spec["training_asof"] == "2026-08-01"
    assert spec["max_label_available_date"] == "2026-07-31"
    assert spec["august_signal_date_outcome_rows_accessed"] == 0
    assert spec["in_sample_performance_used_as_evidence"] is False
    assert spec["L2"] == .30 and spec["positive_weight"] == 1.5


def test_outputs_exist_and_review_has_exactly_seven_questions() -> None:
    for name in audit.OUTPUT_FILENAMES:
        assert (OUTPUT_DIR / name).is_file()
    review = (OUTPUT_DIR / audit.OUTPUT_FILENAMES[-1]).read_text(encoding="utf-8")
    assert sum(review.count(f"## Q{i}.") for i in range(1, 8)) == 7
    assert "## Q8." not in review
    assert "MODEL_STATE = REDUCED7F_TEMPORAL_SIGNAL_SUPPORTED" in review
    assert "NEXT_ACTION = FREEZE_REDUCED7F_FOR_AUGUST_HOLDOUT" in review
    spec = json.loads((OUTPUT_DIR / audit.OUTPUT_FILENAMES[5]).read_text(encoding="utf-8"))
    assert spec["feature_count"] == 7

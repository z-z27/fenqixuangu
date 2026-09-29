from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import src.v004c_s2_winner_vs_loss_failure_attribution as audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "research" / audit.OUTPUT_DIRNAME


def test_frozen_s2_contract_and_no_learner_calls() -> None:
    audit.assert_experiment_contract()
    assert tuple(audit.REDUCED_FEATURE_COLUMNS) == (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    assert (audit.S2_L2, audit.S2_POSITIVE_WEIGHT, audit.S2_TAIL_WEIGHTING) == (.10, 1.50, "NONE")
    source = inspect.getsource(audit).lower()
    for forbidden in (
        "fit_logistic_l2_weighted", "fit_logistic(", "gradientboosting",
        "randomforest", "xgboost", "pairwise", "reranker",
    ):
        assert forbidden not in source


def test_population_and_archived_s2_score_reconstruction() -> None:
    scored, beta, evidence = audit.load_frozen_s2(ROOT)
    assert (len(scored), scored.signal_date.nunique()) == (485, 60)
    assert len(beta) == 8
    assert evidence["model_refits"] == 0
    assert evidence["august_signal_date_outcome_rows_accessed"] == 0
    assert evidence["score_reconstruction_max_summary_error"] <= 1e-12
    assert not scored.signal_date.ge("2026-08-01").any()
    assert scored.groupby("signal_date").s2_rank.apply(
        lambda rank: sorted(rank.tolist()) == list(range(1, len(rank) + 1))
    ).all()


def test_outputs_are_complete_and_deterministic() -> None:
    context = audit.analyze(ROOT)
    first = audit.build_outputs(context)
    second = audit.build_outputs(context)
    assert first == second
    assert tuple(first) == audit.OUTPUT_FILENAMES
    assert len(first) == 11


def test_replaceable_loss_uses_one_for_one_capacity() -> None:
    frame = pd.DataFrame({
        "signal_date": ["2026-05-06"] * 6,
        "event_id": [f"e{i}" for i in range(6)],
        "s2_score": [.9, .8, .7, .6, .5, .4],
        "s2_rank": [1, 2, 3, 4, 5, 6],
        "loss": [1, 1, 0, 0, 0, 0],
        "target7": [0, 0, 0, 1, 0, 0],
    })
    result = audit.build_replaceable_loss(frame)
    daily = result[result.row_type.eq("DAILY")].iloc[0]
    assert daily.top3_loss_slots == 2
    assert daily.loss_slots_with_any_outside_target7 == 2
    assert daily.theoretically_replaceable_loss_slots == 1
    assert daily.theoretical_replaceable_ratio == .5


def test_same_date_pairs_never_cross_dates_and_ties_are_half() -> None:
    frame = pd.DataFrame({
        "signal_date": ["2026-05-06"] * 2 + ["2026-05-07"] * 2,
        "event_id": ["t1", "l1", "t2", "l2"],
        "code": ["1", "2", "3", "4"],
        "board_group": ["BOARD2"] * 4,
        "target7": [1, 0, 1, 0],
        "loss": [0, 1, 0, 1],
        "s2_score": [.5, .5, .6, .4],
    })
    result = audit.build_pairs(frame)
    pairs = result[result.row_type.eq("PAIR")]
    assert len(pairs) == 2
    assert set(zip(pairs.target7_event_id, pairs.loss_event_id)) == {("t1", "l1"), ("t2", "l2")}
    pooled = result[(result.row_type == "SUMMARY") & (result.period == "MAY_JUNE_JULY_MATURE")].iloc[0]
    assert pooled.t7_loss_pair_concordance == .75


def test_feature_audit_never_reorients_per_month_or_endpoint() -> None:
    context = audit.analyze(ROOT)
    info = context["feature_information"]
    assert set(info.feature) == set(audit.REDUCED_FEATURE_COLUMNS)
    assert set(info.period) == set(audit.PERIODS)
    assert set(info.population_scope) == set(audit.SCOPES)
    assert set(info.direction).issubset(
        {"HIGHER_FAVORS_TARGET7", "LOWER_FAVORS_TARGET7", "NEAR_RANDOM"}
    )
    assert set(info.direction_stability).issubset(
        {"STABLE_POSITIVE", "STABLE_NEGATIVE", "SIGN_FLIP", "NEAR_RANDOM"}
    )


def test_raw_vs_rank_uses_only_existing_parent_columns() -> None:
    context = audit.analyze(ROOT)
    raw = context["raw_vs_rank"]
    assert set(raw.feature) == set(audit.REDUCED_FEATURE_COLUMNS)
    assert set(raw.raw_parent) == set(audit.RAW_PARENT_COLUMNS.values())
    assert set(raw.information_state).issubset({
        "RAW_INFORMATION_PRESENT_BUT_RANK_TRANSFORM_LOST",
        "RAW_INFORMATION_ABSENT",
        "RAW_AND_RANK_SIMILAR",
    })


def test_contributions_reconstruct_s2_logit_exactly() -> None:
    context = audit.analyze(ROOT)
    scored = context["scored"]
    beta = context["beta"]
    contribution_sum = scored[list(audit.REDUCED_FEATURE_COLUMNS)].to_numpy(float) @ beta[1:]
    assert np.allclose(beta[0] + contribution_sum, scored.s2_logit, rtol=0, atol=1e-14)


def test_hard_loss_and_missed_winner_semantics() -> None:
    context = audit.analyze(ROOT)
    hard = context["hard_losses"]
    detail = hard[hard.row_type.eq("LOSS_DETAIL")]
    assert set(detail.loss_classification) <= {
        "HARD_FALSE_POSITIVE_LOSS", "UNAVOIDABLE_TOP3_LOSS"
    }
    hard_fp = detail[detail.loss_classification.eq("HARD_FALSE_POSITIVE_LOSS")]
    assert hard_fp.lower_ranked_target7_count.gt(0).all()
    missed = context["missed_target7"]
    missed_detail = missed[missed.row_type.eq("TARGET7_DETAIL")]
    assert missed_detail.s2_rank.gt(3).all()
    assert set(missed_detail.rank_bucket) <= {"4-5", "6-7", "8-10", ">10"}


def test_geometry_is_descriptive_fixed_7f_only() -> None:
    geometry = audit.analyze(ROOT)["geometry"]
    assert geometry.feature_count.eq(7).all()
    assert set(geometry.standardization) == {"WITHIN_PERIOD_T7_AND_LOSS_ZSCORE"}
    finite = geometry.nearest_opposite_overlap_rate.dropna()
    assert finite.between(0, 1).all()


def test_formal_state_and_review_contract() -> None:
    context = audit.analyze(ROOT)
    assert context["state"] in {
        "MODEL_FORMULATION_FAILURE_DOMINANT", "INFORMATION_LIMIT_DOMINANT", "MIXED"
    }
    review = audit.render_review(context, context["state"])
    assert sum(review.count(f"## Q{i}.") for i in range(1, 9)) == 8
    assert "## Q9." not in review
    assert f"WINNER_LOSS_FAILURE_ATTRIBUTION = {context['state']}" in review
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review


def test_written_outputs_exist() -> None:
    for name in audit.OUTPUT_FILENAMES:
        assert (OUTPUT_DIR / name).is_file()


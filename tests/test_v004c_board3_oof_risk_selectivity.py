from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.v004c_board3_oof_risk_selectivity as diagnostic
from src.v004c_v4a_architecture_transfer import dataframe_csv_bytes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    first = diagnostic.build_outputs(ROOT)
    second = diagnostic.build_outputs(ROOT)
    assert first[0] == second[0]
    return first


def test_starting_and_scope_contract():
    assert diagnostic.EXPECTED_STARTING_HEAD == "15c0000ef6a72f5a80f104c01a0527e918efef82"
    diagnostic.assert_diagnostic_contract()
    assert diagnostic.POST_HOC_OOF_RISK_SELECTIVITY_DIAGNOSTIC is True
    assert diagnostic.TAIL_DEFINITIONS == ("BOTTOM_HALF", "WORST_ONE")
    assert diagnostic.BOTTOM_HALF_CUTOFF == .50
    assert not any((
        diagnostic.NEW_MODEL, diagnostic.MODEL_REFIT, diagnostic.NEW_FEATURE,
        diagnostic.NEW_TARGET, diagnostic.NEW_SCORE, diagnostic.THRESHOLD_SEARCH,
        diagnostic.VETO_POLICY, diagnostic.BACKFILL, diagnostic.BOARD2_MODIFICATION,
        diagnostic.CROSS_BOARD_COMBINATION,
    ))
    assert diagnostic.JULY_RESULT_ROWS_ACCESSED == 0


def test_prediction_lock_sha256_and_score_immutability(built):
    _, context = built
    lock_path = ROOT / diagnostic.SOURCE_DIR / diagnostic.LOCK_FILENAME
    assert sha256(lock_path.read_bytes()).hexdigest() == diagnostic.EXPECTED_LOCK_SHA256
    assert context["lock_audit"]["observed_lock_sha256"] == diagnostic.EXPECTED_LOCK_SHA256
    assert context["outcome_perturbation"] == "PASS"
    assert diagnostic.outcome_perturbation_is_immutable(context["population"])


def test_common_population_and_outcomes_exact(built):
    population = built[1]["population"]
    assert sorted(population.signal_date.unique()) == diagnostic.EXPECTED_COMMON_DATES
    assert population.signal_date.nunique() == 8
    assert len(population) == 19
    assert population.event_id.nunique() == 19
    assert int(population.target7.sum()) == 5
    assert int(population.loss.sum()) == 6
    assert int(population.positive_non_target.sum()) == 8
    assert int(population.nonloss.sum()) == 13
    assert (population.loss == population.raw_repair_return.lt(0).astype(int)).all()
    assert (population.target7 == population.raw_repair_return.ge(.07).astype(int)).all()
    assert np.allclose(population.capped_return_7, np.minimum(population.raw_repair_return, .07))


def test_candidate_count_and_relative_population_parity(built):
    context = built[1]
    counts = context["lock_audit"]["candidate_counts"]
    assert counts == {
        "min": 1, "median": 2.0, "mean": 2.375, "max": 8,
        "dates_1": 3, "dates_2": 4, "dates_ge4": 1,
    }
    eligible = context["eligible"]
    assert eligible.signal_date.nunique() == 5
    assert len(eligible) == 16
    assert int(eligible.target7.sum()) == 4
    assert int(eligible.loss.sum()) == 4
    assert int(eligible.positive_non_target.sum()) == 8
    assert eligible.board3_candidate_count.ge(2).all()


def test_singleton_risk_surface_is_ineligible():
    frame = pd.DataFrame({
        "event_id": ["e1"], "signal_date": ["d"], "code": ["000001"],
        "board3_candidate_count": [1], "board3_only_score": [.7],
        "board3_only_internal_rank": [1],
    })
    result = diagnostic.derive_relative_risk_surface(frame).iloc[0]
    assert bool(result.relative_rank_eligible) is False
    assert np.isnan(result.risk_rank_percentile)
    assert bool(result.bottom_half_flag) is False
    assert bool(result.worst_one_flag) is False


def test_bottom_half_and_worst_one_mechanics():
    frame = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(1, 5)],
        "signal_date": ["d"] * 4, "code": [f"{i:06d}" for i in range(1, 5)],
        "board3_candidate_count": [4] * 4,
        "board3_only_score": [.9, .8, .7, .6],
        "board3_only_internal_rank": [1, 2, 3, 4],
    })
    result = diagnostic.derive_relative_risk_surface(frame)
    assert result.risk_rank_percentile.tolist() == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert result.loc[result.bottom_half_flag, "board3_only_internal_rank"].tolist() == [3, 4]
    assert result.loc[result.worst_one_flag, "board3_only_internal_rank"].tolist() == [4]
    two = frame.iloc[:2].copy()
    two["board3_candidate_count"] = 2
    two_result = diagnostic.derive_relative_risk_surface(two)
    assert two_result.risk_rank_percentile.tolist() == [0, 1]
    assert two_result.bottom_half_flag.tolist() == [False, True]


def test_primary_tail_selectivity_exact(built):
    summaries = built[1]["tail_summary"]
    bottom = summaries["BOTTOM_HALF"]
    worst = summaries["WORST_ONE"]
    assert bottom["flagged_rows"] == 8
    assert (bottom["flagged_loss"], bottom["flagged_pnt"], bottom["flagged_target7"]) == (3, 3, 2)
    assert bottom["loss_capture_rate"] == pytest.approx(.75)
    assert bottom["pnt_removal_rate"] == pytest.approx(.375)
    assert bottom["winner_removal_rate"] == pytest.approx(.50)
    assert bottom["selectivity_gap"] == pytest.approx(.25)
    assert (bottom["unflagged_loss"], bottom["unflagged_pnt"], bottom["unflagged_target7"]) == (1, 5, 2)
    assert worst["flagged_rows"] == 5
    assert (worst["flagged_loss"], worst["flagged_pnt"], worst["flagged_target7"]) == (2, 1, 2)
    assert worst["selectivity_gap"] == pytest.approx(0)


def test_selectivity_arithmetic_good_and_bad():
    base = pd.DataFrame({
        "signal_date": ["d"] * 10,
        "loss": [1] * 4 + [0] * 6,
        "target7": [0] * 4 + [1] * 4 + [0] * 2,
        "positive_non_target": [0] * 8 + [1] * 2,
        "flag": [True, True, True, False, True, False, False, False, True, False],
    })
    good = diagnostic.tail_selectivity(base, "flag", "SYNTHETIC")
    assert good["loss_capture_rate"] == .75
    assert good["winner_removal_rate"] == .25
    assert good["pnt_removal_rate"] == .50
    assert good["selectivity_gap"] == .50
    base["flag"] = [True, True, False, False, True, True, True, False, False, False]
    bad = diagnostic.tail_selectivity(base, "flag", "SYNTHETIC")
    assert bad["loss_capture_rate"] == .50
    assert bad["winner_removal_rate"] == .75
    assert bad["selectivity_gap"] == -.25


def test_known_auc_and_pair_parity(built):
    context = built[1]
    controls = context["controls"]
    pairs = context["pairs"]
    assert controls["unified_target7_auc"] == pytest.approx(.4714285714285714)
    assert controls["board3_target7_auc"] == pytest.approx(.45714285714285713)
    assert controls["unified_nonloss_auc"] == pytest.approx(3 / 13)
    assert controls["board3_nonloss_auc"] == pytest.approx(9 / 13)
    assert pairs["unified_nonloss"]["informative_pairs"] == 14
    assert pairs["unified_nonloss"]["concordance"] == pytest.approx(3 / 14)
    assert pairs["board3_nonloss"]["informative_pairs"] == 14
    assert pairs["board3_nonloss"]["concordance"] == pytest.approx(13 / 14)
    assert context["winner_loss_pair_support_very_weak"] is True


def test_same_date_pairs_never_cross_dates_and_ties_half():
    frame = pd.DataFrame({
        "signal_date": ["a", "a", "b", "b"], "score": [.5, .5, .1, .9],
        "positive": [1, 0, 1, 0], "negative": [0, 1, 0, 1],
    })
    summary = diagnostic.cross_class_pair_summary(frame, "score", "positive", "negative")
    assert summary["informative_dates"] == 2
    assert summary["informative_pairs"] == 2
    assert summary["ties"] == 1
    assert summary["concordance"] == .25


def test_curve_has_all_steps_and_no_optimization_fields(built):
    curve = built[1]["curve"]
    assert len(curve) == len(built[1]["eligible"]) == 16
    assert curve.step.tolist() == list(range(1, 17))
    assert curve.removed_rows.tolist() == list(range(1, 17))
    forbidden = {"optimal", "best", "selected_threshold", "score_threshold"}
    assert not forbidden.intersection(curve.columns)
    assert built[1]["curve_summary"]["no_threshold_selected"] is True


def test_decision_surface_is_exact_unified_top3_subset(built):
    context = built[1]
    expected = set(context["population"].loc[
        context["population"].unified_stage1_top3, "event_id"
    ])
    observed = set(context["decision_surface"].event_id)
    assert observed == expected
    summary = context["decision_surface_summary"]
    assert summary["rows"] == len(expected)
    assert summary["rows"] == 10
    assert summary["support"] == "ADEQUATE"


def test_bootstrap_samples_dates_and_lodo_never_refits(built):
    context = built[1]
    dates, counts = diagnostic.date_bootstrap_counts(context["population"].signal_date)
    assert dates == diagnostic.EXPECTED_COMMON_DATES
    assert counts.shape == (20_000, 8)
    assert (counts.sum(axis=1) == 8).all()
    assert context["bootstrap"]["BOTTOM_HALF_SELECTIVITY_GAP"]["valid_resamples"] <= 20_000
    assert context["lodo"]["BOTTOM_HALF_SELECTIVITY_GAP"]["valid_resamples"] == 8


def test_formal_gates_and_partial_stop(built):
    decision = built[1]["decision"]
    assert decision["gates"] == {
        "A": True, "B": True, "C": False, "D": True, "E": False,
        "F": True, "G": False, "H": False, "I": False, "J": True, "K": True,
    }
    assert decision["decision_surface_gate"] is True
    assert decision["signal"] == "PARTIAL"
    assert decision["next_action"] == "NO_BOARD3_RISK_POLICY_YET"


def test_no_model_search_veto_backfill_board2_or_july_path(built):
    source = inspect.getsource(diagnostic)
    forbidden_calls = (
        "fit_logistic_l2_weighted(", "fit_logistic(", "GradientBoosting(",
        "RandomForest(", "XGBoost(", "GridSearch(",
    )
    assert all(token not in source for token in forbidden_calls)
    assert "optimal_fraction" not in source
    assert "best_threshold" not in source
    assert built[1]["july_result_rows_accessed"] == 0
    assert len(diagnostic.TAIL_DEFINITIONS) == 2


def test_output_contract_and_double_build_byte_identity(built):
    outputs, context = built
    assert set(outputs) == set(diagnostic.OUTPUT_FILENAMES)
    assert outputs[diagnostic.OUTPUT_FILENAMES[0]] == dataframe_csv_bytes(
        context["population"][[
            "event_id", "signal_date", "code", "board3_candidate_count",
            "board3_only_score", "board3_only_internal_rank", "risk_rank_percentile",
            "unified_stage1_score", "unified_stage1_rank", "unified_stage1_top3",
            "target7", "loss", "positive_non_target", "nonloss", "raw_repair_return",
            "capped_return_7", "relative_rank_eligible", "bottom_half_flag", "worst_one_flag",
        ]]
    )
    assert context["decision"]["signal"] in {
        "STRONG", "PARTIAL", "ABSENT", "INSUFFICIENT_EVIDENCE",
    }

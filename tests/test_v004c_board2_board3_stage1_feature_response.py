from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import pytest

import src.v004c_board2_board3_stage1_feature_response as audit
from src.v004c_limited_risk_protector import (
    EXPECTED_STAGE1_PARITY,
    EXPECTED_STRICT_DATES,
)
from src.v004c_stage1_top3_risk_information import benjamini_hochberg, binary_auc


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    outputs, context = audit.build_outputs(ROOT)
    return outputs, context


def test_starting_frozen_contract():
    assert audit.EXPECTED_STARTING_HEAD == "e58c0da22553f598aaf5874b614b1ee5ae2e1949"
    audit.assert_audit_contract()
    assert audit.MODEL_ID == "V4A_ARCH_TRANSFER_V4C"
    assert audit.MODEL_FAMILY == "WEIGHTED_L2_LOGISTIC"
    assert audit.L2 == 0.30
    assert audit.POSITIVE_WEIGHT == 1.50
    assert audit.INITIAL_TRAIN_DATES == 18


def test_strict_population_and_board_parity(built):
    _, context = built
    frame = context["population"]
    assert sorted(frame["signal_date"].unique()) == EXPECTED_STRICT_DATES
    assert frame["signal_date"].nunique() == 17
    assert len(frame) == 155
    assert frame["event_id"].nunique() == 155
    b2 = frame[frame["board_group"].eq("BOARD2")]
    b3 = frame[frame["board_group"].eq("BOARD3")]
    assert (len(b2), int(b2.target7.sum()), int(b2.loss.sum())) == (130, 44, 33)
    assert (len(b3), int(b3.target7.sum()), int(b3.loss.sum())) == (25, 9, 8)
    b2_top3 = b2[b2.stage1_rank.le(3)]
    b3_top3 = b3[b3.stage1_rank.le(3)]
    assert (len(b2_top3), int(b2_top3.target7.sum()), int(b2_top3.loss.sum())) == (36, 12, 8)
    assert (len(b3_top3), int(b3_top3.target7.sum()), int(b3_top3.loss.sum())) == (15, 4, 8)
    assert b2_top3.capped_return_7.mean() == pytest.approx(0.026824, abs=5e-7)
    assert b3_top3.capped_return_7.mean() == pytest.approx(0.014147, abs=5e-7)


def test_stage1_parity_exact(built):
    parity = built[1]["reconstruction"]["stage1_parity"]
    for metric, expected in EXPECTED_STAGE1_PARITY.items():
        assert parity[metric] == pytest.approx(expected, abs=1e-11)


def test_exact_18_features_and_no_extension():
    assert audit.FROZEN_FEATURE_COLUMNS == audit.EXPECTED_FEATURE_COLUMNS
    assert len(audit.FROZEN_FEATURE_COLUMNS) == 18
    forbidden = {
        "board_streak_is_3", "max_board_streak_20d",
        "recent_7d_limit_up_count", "closing_completion_gap",
    }
    assert forbidden.isdisjoint(audit.FROZEN_FEATURE_COLUMNS)


def test_contribution_reconstruction_unit():
    contributions, logits = audit.contribution_decomposition(
        np.array([[0.2, 0.5]]), np.array([2.0, -1.0]), -1.0
    )
    assert contributions[0].tolist() == pytest.approx([0.4, -0.5])
    assert logits.tolist() == pytest.approx([-1.1])


def test_real_logit_score_and_daily_closure(built):
    context = built[1]
    metrics = context["reconstruction"]
    assert metrics["max_logit_reconstruction_error"] <= 1e-10
    assert metrics["max_score_reconstruction_error"] <= 1e-10
    assert context["daily_summary"]["max_contribution_closure_error"] <= 1e-10


def test_temporal_and_july_gates(built):
    audit_frame = built[1]["fold_audit"]
    assert audit_frame.self_label_leakage_rows.sum() == 0
    assert audit_frame.current_test_leakage_rows.sum() == 0
    assert audit_frame.july_rows_accessed.sum() == 0
    assert (audit_frame.latest_label_available_date < audit_frame.test_date).all()
    assert built[1]["july_result_rows_accessed"] == 0


@pytest.mark.parametrize(
    "b2,b3,expected",
    [
        (0.62, 0.38, "BOARD3_INVERTED"),
        (0.55, 0.45, "BOARD3_INVERTED"),
        (0.55, 0.56, "HEALTHY_SHARED"),
        (0.55, 0.50, "BOARD3_DEGRADED"),
        (0.50, 0.50, "SHARED_WEAK"),
        (0.45, 0.45, "SHARED_BAD"),
    ],
)
def test_response_category_fixed_boundaries(b2, b3, expected):
    assert audit.response_category(b2, b3) == expected


def test_auc_is_not_flipped():
    labels = [0, 0, 1, 1]
    scores = [4.0, 3.0, 2.0, 1.0]
    assert binary_auc(labels, scores) == 0.0


def test_same_date_pairs_and_ties():
    frame = pd.DataFrame({
        "signal_date": ["d1"] * 4,
        "nonloss": [1, 1, 0, 0],
        "target7": [1, 0, 0, 0],
        "score": [2.0, 1.0, 1.0, 0.0],
    })
    result = audit.same_date_pair_summary(frame, "score", "NONLOSS")
    assert result["informative_pairs"] == 4
    assert result["concordant"] == 3
    assert result["ties"] == 1
    assert result["pair_concordance"] == pytest.approx(0.875)


def test_bootstrap_samples_dates_not_rows():
    frame = pd.DataFrame({
        "signal_date": ["a", "a", "b"],
        "event_id": ["1", "2", "3"],
    })
    dates, counts, weights = audit.date_bootstrap_weights(frame, resamples=20, seed=7)
    assert dates == ["a", "b"]
    assert counts.shape == (20, 2)
    assert (counts.sum(axis=1) == 2).all()
    assert np.array_equal(weights[:, 0], weights[:, 1])


def test_permutation_within_date_preserves_board_counts():
    frame = pd.DataFrame({
        "signal_date": ["a"] * 6 + ["b"] * 6,
        "board_group": ["BOARD2"] * 4 + ["BOARD3"] * 2
        + ["BOARD2"] * 4 + ["BOARD3"] * 2,
    })
    permutations = audit.permute_board_labels(frame, permutations=100, seed=9)
    assert (permutations[:, :6].sum(axis=1) == 2).all()
    assert (permutations[:, 6:].sum(axis=1) == 2).all()


def test_one_sided_permutation_and_bh_family_exact(built):
    context = built[1]
    permutation = context["permutation"]
    assert len(permutation) == 18
    assert permutation.interaction_p.between(0, 1).all()
    assert permutation.interaction_q.between(0, 1).all()
    assert permutation.target7_interaction_p.between(0, 1).all()
    assert np.allclose(
        permutation.interaction_q,
        benjamini_hochberg(permutation.interaction_p),
    )


def test_response_inversion_gate_exact():
    row = {
        "board2_nonloss_auc": 0.55,
        "board3_nonloss_auc": 0.45,
        "delta_nonloss_auc": -0.15,
        "mean_daily_contribution_gap": 0.001,
        "bootstrap_P_delta_nonloss_lt0": 0.90,
        "bootstrap_P_gap_gt0": 0.80,
        "lodo_delta_nonloss_negative_pct": 0.80,
        "interaction_q": 0.10,
    }
    assert audit.response_inversion_gate(row)
    assert not audit.response_inversion_gate({**row, "mean_daily_contribution_gap": 0.0})


def test_output_schema_and_primary_bh_only(built):
    outputs, context = built
    assert set(outputs) == set(audit.OUTPUT_FILENAMES)
    assert len(context["univariate"]) == 18
    assert set(context["univariate"].feature) == set(audit.FROZEN_FEATURE_COLUMNS)
    assert set(context["pairs"].endpoint) == {"NONLOSS", "TARGET7"}
    assert len(context["daily"]) == 17 * 18
    close = context["univariate"].set_index("feature").loc["rank_d1_close_ma10_pct"]
    assert close.same_date_feature_gap_positive_pct == pytest.approx(12 / 14)
    target_rows = context["robustness"][
        context["robustness"]["metric"].eq("DELTA_TARGET7_AUC")
        & context["robustness"]["section"].eq("PERMUTATION")
    ]
    assert len(target_rows) == 18
    assert target_rows.q_value.isna().all()


def test_no_model_training_beyond_frozen_reconstruction_and_no_policy():
    source = inspect.getsource(audit)
    assert source.count("_fit_stage1_snapshot(") == 1
    for forbidden in (
        "fit_pairwise_weighted_ridge(", "fit_logistic_l2_weighted(",
        "board3_penalty", "exclude_board3", "board3_veto",
    ):
        assert forbidden not in source

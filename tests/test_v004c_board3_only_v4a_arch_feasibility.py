from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import pytest

import src.v004c_board3_only_v4a_arch_feasibility as bench
from src.v004a import DEFAULT_HIGH_RETURN_COLUMN, DEFAULT_TARGET_COLUMN, build_training_sample_weight


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    return bench.build_outputs(ROOT)


def test_starting_and_frozen_contract():
    assert bench.EXPECTED_STARTING_HEAD == "5fa911907532bf03f9b4144f2263199dd9432384"
    bench.assert_benchmark_contract()
    assert len(bench.FROZEN_FEATURE_COLUMNS) == 18
    assert bench.L2 == .30
    assert bench.POSITIVE_WEIGHT == 1.5
    assert bench.MIN_TRAIN_BOARD3_SIGNAL_DATES == 18
    assert not any((bench.FEATURE_SELECTION, bench.HYPERPARAMETER_SEARCH,
                    bench.MODEL_ZOO, bench.TARGET_SEARCH, bench.CROSS_BOARD_COMBINATION))


def test_authoritative_and_matured_population_parity(built):
    context = built[1]
    assert context["source_audit"]["rows"] == 319
    assert context["source_audit"]["dates"] == 39
    assert context["source_audit"]["board2"] == 261
    assert context["source_audit"]["board3"] == 58
    board3 = context["matured_board3"]
    assert len(board3) == 55
    assert board3.signal_date.nunique() == 28
    assert (board3.label_available_date < "2026-07-01").all()


def test_full_date_rank_semantics_before_board3_filter():
    rows = []
    for index in range(10):
        row = {"event_id": f"e{index}", "signal_date": "2026-01-01",
               "board_streak_before_break": 3 if index >= 8 else 2}
        row.update({feature: index / 9 for feature in bench.FROZEN_FEATURE_COLUMNS})
        rows.append(row)
    full = pd.DataFrame(rows)
    board3 = bench.select_board3_after_full_date_transform(full)
    assert board3[bench.FROZEN_FEATURE_COLUMNS[0]].tolist() == pytest.approx([8 / 9, 1.0])
    assert board3[bench.FROZEN_FEATURE_COLUMNS[0]].tolist() != [0.0, 1.0]


def test_weighting_and_temporal_gates():
    frame = pd.DataFrame({
        "signal_date": ["d", "d", "d"], DEFAULT_TARGET_COLUMN: [True, False, True],
        "tail_weight": [2.0, 1.0, 1.5], DEFAULT_HIGH_RETURN_COLUMN: [12.0, 0.0, 10.0],
    })
    weights = build_training_sample_weight(frame, positive_weight=1.5)
    assert weights.tolist() == pytest.approx([1.0, 1 / 3, .75])
    assert bench.fold_is_eligible(17, 2)[0] is False
    assert bench.fold_is_eligible(18, 2)[0] is True
    assert bench.fold_is_eligible(18, 1) == (False, "UNFITTABLE_SINGLE_CLASS")


def test_oof_availability_and_common_identity(built):
    context = built[1]
    frame = context["oof"]
    availability = context["availability"]
    assert len(availability["common_dates"]) >= 6
    assert len(frame) >= 12
    assert sorted(frame.signal_date.unique()) == availability["common_dates"]
    assert availability["board2_training_rows"] == 0
    assert availability["self_label_leakage_rows"] == 0
    assert availability["current_test_leakage_rows"] == 0
    assert set(frame.event_id).issubset(set(context["prediction_lock"].event_id))


def test_board3_only_fits_and_prediction_lock(built):
    outputs, context = built
    lock = context["prediction_lock"]
    assert not bench.OUTCOME_COLUMNS.intersection(lock.columns)
    assert bench.prediction_lock_sha256(lock) == context["prediction_lock_sha256"]
    assert outputs[bench.OUTPUT_FILENAMES[2]] == bench.dataframe_csv_bytes(lock)
    assert (context["fold_audit"].board2_training_rows == 0).all()
    assert (context["fold_audit"].feature_count == 18).all()
    assert (context["fold_audit"].l2 == .30).all()
    assert (context["fold_audit"].positive_weight == 1.5).all()


def test_outcome_perturbation_cannot_change_prediction_lock(built):
    predictions = built[1]["prediction_lock"].copy()
    original = bench.build_prediction_lock(bench.prediction_inputs(predictions))
    for column in bench.OUTCOME_COLUMNS:
        predictions[column] = np.arange(len(predictions), dtype=float) * 999.0
    perturbed = bench.build_prediction_lock(bench.prediction_inputs(predictions))
    assert bench.dataframe_csv_bytes(original) == bench.dataframe_csv_bytes(perturbed)


def test_auc_and_same_date_pair_semantics():
    frame = pd.DataFrame({
        "signal_date": ["d", "d", "d"], "score": [.9, .4, .4],
        "nonloss": [1, 0, 0], "target7": [1, 0, 0], "loss": [0, 1, 1],
    })
    result = bench.same_date_pair_summary(frame, "score", "nonloss")
    assert result["informative_pairs"] == 2
    assert result["concordance"] == 1.0
    tied = frame.copy()
    tied["score"] = .5
    assert bench.same_date_pair_summary(tied, "score", "nonloss")["concordance"] == .5


def test_internal_rank_percentile_and_topk_date_equal(built):
    frame = built[1]["oof"]
    for _, day in frame.groupby("signal_date"):
        expected = day.sort_values(["board3_only_score", "event_id"], ascending=[False, True])
        assert expected.board3_only_internal_rank.tolist() == list(range(1, len(day) + 1))
        assert len(day[day.board3_only_internal_rank.le(min(2, len(day)))]) == min(2, len(day))
    assert bench.rank_percentile(1, 1) == 0
    assert bench.rank_percentile(2, 3) == .5


def test_bootstrap_and_lodo_are_date_based(built):
    context = built[1]
    assert all(values["valid"] <= 20_000 for values in context["bootstrap"].values())
    assert context["bootstrap"]["RANK1_CAPPED_DELTA"]["valid"] == 20_000
    dates = context["oof"].signal_date.nunique()
    assert context["lodo"]["RANK1_CAPPED_DELTA"]["valid"] == dates
    rng = np.random.default_rng(bench.BOOTSTRAP_SEED)
    draws = rng.integers(0, dates, size=(bench.BOOTSTRAP_RESAMPLES, dates))
    assert draws.shape == (20_000, dates)


def test_no_search_cross_board_or_july_expansion(built):
    context = built[1]
    assert context["july_result_rows_accessed"] == 0
    assert context["outcome_perturbation"] == "PASS"
    source = inspect.getsource(bench)
    assert "GridSearch" not in source
    assert "RandomForest" not in source
    assert "XGBoost" not in source
    assert "LightGBM" not in source
    assert context["decision"]["signal"] in {
        "STRONG", "PARTIAL", "ABSENT", "INSUFFICIENT_EVIDENCE",
    }


def test_output_contract(built):
    outputs, context = built
    assert set(outputs) == set(bench.OUTPUT_FILENAMES)
    assert len(context["coefficients"].feature.unique()) == 19
    assert context["decision"]["next_action"] in {
        "BOARD3_MINIMAL_REPRESENTATION_AUDIT", "STOP_BOARD3_D1_MODEL_ROUTE",
        "NO_NEW_MODEL_YET", "INSUFFICIENT_DATA",
    }

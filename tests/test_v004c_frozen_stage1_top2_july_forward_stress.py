from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.v004c_frozen_stage1_top2_july_forward_stress as stress
from src.v004c_v4a_architecture_transfer import dataframe_csv_bytes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    first = stress.build_outputs(ROOT)
    second = stress.build_outputs(ROOT)
    assert first[0] == second[0]
    return first


def test_starting_scope_and_frozen_contract():
    assert stress.EXPECTED_STARTING_HEAD == "b59f169fe5c7b597e8ec99c5644a32f5e697016a"
    stress.assert_frozen_contract()
    assert stress.EVALUATION_MODE == "CHRONOLOGICAL_FORWARD_STRESS"
    assert stress.SEMI_OOT and not stress.PRISTINE_OOT
    assert stress.MODEL_ID == "V4A_ARCH_TRANSFER_V4C"
    assert stress.MODEL_FAMILY == "WEIGHTED_L2_LOGISTIC"
    assert len(stress.FROZEN_FEATURE_COLUMNS) == 18
    assert stress.L2 == .30 and stress.POSITIVE_WEIGHT == 1.50
    assert not any((stress.NEW_MODEL, stress.NEW_FEATURE, stress.HYPERPARAMETER_SEARCH,
                    stress.STAGE2, stress.BOARD3_REPAIR, stress.TOPK_SEARCH,
                    stress.THRESHOLD_SEARCH, stress.MODEL_SWITCH))


def test_existing_prediction_provenance_and_no_july_training(built):
    context = built[1]
    provenance = context["provenance"]
    source = ROOT / stress.SOURCE_DIR / stress.SOURCE_LOCK
    assert sha256(source.read_bytes()).hexdigest() == stress.EXPECTED_SOURCE_LOCK_SHA256
    assert provenance["prediction_source"] == "REUSED_EXISTING"
    assert provenance["fit_count"] == 0 and provenance["july_refits"] == 0
    audit = provenance["training_audit"]
    assert audit["training_snapshot_date"] == "2026-07-01"
    assert audit["stage1_training_rows"] == "307"
    assert audit["eligible_historical_dates"] == "37"
    assert audit["latest_eligible_signal_date"] == "2026-06-26"
    assert audit["latest_label_available_date"] == "2026-06-30"
    assert audit["july_training_rows"] == "0"


def test_prediction_lock_before_outcomes_and_immutable(built):
    outputs, context = built
    lock = context["lock"]
    assert list(lock.columns) == stress.LOCK_COLUMNS
    assert not stress.OUTCOME_COLUMNS.intersection(lock.columns)
    assert stress.prediction_lock_sha256(lock) == context["provenance"]["prediction_lock_sha256"]
    assert stress.outcome_perturbation_is_immutable(context["population"])
    assert outputs[stress.OUTPUT_FILENAMES[0]] == dataframe_csv_bytes(lock)


def test_july_authoritative_population_and_outcomes(built):
    population = built[1]["population"]
    assert len(population) == 178
    assert population.signal_date.nunique() == 23
    assert population.signal_date.min() == "2026-07-01"
    assert population.signal_date.max() == "2026-07-31"
    assert int(population.board_group.eq("BOARD2").sum()) == 155
    assert int(population.board_group.eq("BOARD3").sum()) == 23
    assert (population.loss == population.raw_repair_return.lt(0).astype(int)).all()
    assert (population.severe_loss == population.raw_repair_return.le(-.05).astype(int)).all()
    assert (population.target7 == population.raw_repair_return.ge(.07).astype(int)).all()
    np.testing.assert_allclose(
        population.capped_return_7,
        np.minimum(population.raw_repair_return, .07), rtol=0, atol=1e-12,
    )


def test_top2_top3_and_equal_weights_exact(built):
    context = built[1]
    population = context["population"]
    daily = context["daily"]
    for date, day in population.groupby("signal_date"):
        ordered = day.sort_values(["stage1_rank", "event_id"], kind="mergesort")
        row = daily[daily.signal_date.eq(date)].iloc[0]
        assert row.top2_raw == pytest.approx(ordered.head(2).raw_repair_return.mean())
        assert row.top2_capped == pytest.approx(ordered.head(2).capped_return_7.mean())
        if len(ordered) >= 3:
            assert row.top3_raw == pytest.approx(ordered.head(3).raw_repair_return.mean())
            assert row.top3_capped == pytest.approx(ordered.head(3).capped_return_7.mean())
    assert context["topk"][2]["selected_rows"] == 2 * context["topk"][2]["dates"]
    assert context["topk"][3]["selected_rows"] == 3 * context["topk"][3]["dates"]


def test_rank3_marginal_exact():
    daily = pd.DataFrame({
        "signal_date": ["d"], "rank1_code": ["a"], "rank2_code": ["b"],
        "rank3_code": ["c"], "rank1_capped": [.07], "rank2_capped": [.05],
        "rank3_capped": [-.06], "rank3_raw": [-.06], "rank3_target7": [0],
        "rank3_loss": [1], "top2_capped": [.06], "top3_capped": [.02],
    })
    table, summary = stress.build_rank3_marginal(daily)
    assert table.iloc[0].rank3_marginal == pytest.approx(-.04)
    assert summary["mean"] == pytest.approx(-.04)


def test_rank_forward_value_boundaries():
    established = {
        "dates": 10, "capped_mean": .03, "universe_capped_mean": .02,
        "loss_rate": .20, "universe_loss_rate": .30,
        "target7_rate": .40, "universe_target7_rate": .30,
    }
    assert stress.forward_value(established) == "ESTABLISHED"
    absent = established | {"capped_mean": .01, "loss_rate": .40, "target7_rate": .20}
    assert stress.forward_value(absent) == "ABSENT"


def test_bootstrap_dates_and_lodo_no_refit(built):
    context = built[1]
    assert stress.BOOTSTRAP_RESAMPLES == 20_000
    assert context["bootstrap"]["TOP2_MINUS_TOP3_CAPPED"]["valid_resamples"] == 20_000
    assert context["lodo"]["TOP2_MINUS_UNIVERSE_CAPPED"]["valid_resamples"] == 23
    source = inspect.getsource(stress)
    forbidden = (
        "fit_logistic_l2_weighted(", "fit_logistic(", "GradientBoosting(",
        "RandomForest(", "XGBoost(", "GridSearch(",
    )
    assert all(token not in source for token in forbidden)


def test_formal_gate_is_predeclared_and_valid(built):
    decision = built[1]["decision"]
    assert set(decision["gates"]) == set("ABCDEFGHIJ")
    assert decision["gates"] == {
        "A": True, "B": False, "C": False, "D": True, "E": True,
        "F": True, "G": True, "H": True, "I": False, "J": True,
    }
    assert built[1]["rank_values"] == {1: "PARTIAL", 2: "PARTIAL", 3: "PARTIAL"}
    assert decision["signal"] == "PARTIAL"
    assert decision["primary_failure"] in {
        "UNIVERSE_OPPORTUNITY_COLLAPSE", "STAGE1_SELECTION_ALPHA_FAILURE",
        "RANK1_FAILURE", "RANK2_FAILURE", "TOP2_CAPACITY_FAILURE",
        "TAIL_RISK_FAILURE", "MIXED_FAILURE", "INSUFFICIENT", "NONE",
    }
    assert decision["primary_failure"] == "STAGE1_SELECTION_ALPHA_FAILURE"
    assert decision["architecture_state"] == "REOPEN_STAGE1_QUESTION"


def test_outputs_complete_and_byte_deterministic(built):
    outputs, context = built
    assert set(outputs) == set(stress.OUTPUT_FILENAMES)
    assert all(outputs.values())
    assert context["outcome_perturbation"] == "PASS"

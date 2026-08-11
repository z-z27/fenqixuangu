from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_stage1_risk_complementarity import (
    JULY_RESULT_ROWS_ACCESSED,
    JULY_SENTINEL,
    MIN_STAGE1_TRAIN_DATES,
    NEW_FEATURES,
    NEW_MODEL_DEVELOPED,
    OUTPUT_FILENAMES,
    POLICY_CAPPED,
    POLICY_STAGE1,
    THRESHOLD_SEARCH,
    add_policy_ranks,
    assert_diagnostic_contract,
    build_outputs,
    build_swap_attribution,
    capped7_utility,
    loss_permission_ids,
    stage1_loss_veto_ids,
    winner_protection_ids,
)
from src.v004c_top10_target_information import PAIR_FEATURE_COLUMNS, PAIR_L2
from src.v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2 as STAGE1_L2,
    POSITIVE_WEIGHT as STAGE1_POSITIVE_WEIGHT,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diagnostic_result() -> tuple[dict[str, bytes], dict]:
    return build_outputs(ROOT)


def _day_fixture() -> pd.DataFrame:
    rows = [
        ("A", 1, 4, 4, 1, 0.08),
        ("B", 2, 3, 3, 0, 0.03),
        ("C", 3, 5, 5, 0, -0.06),
        ("D", 4, 1, 1, 0, 0.02),
        ("E", 5, 2, 2, 1, 0.10),
        ("F", 6, 6, 6, 0, -0.02),
    ]
    frame = pd.DataFrame(rows, columns=[
        "event_id", "stage1_rank", "final_capped_rank", "capped_pair_rank",
        "target7", "raw_repair_return",
    ])
    frame["signal_date"] = "2026-06-10"
    frame["code"] = [f"{index:06d}" for index in range(len(frame))]
    frame["stage1_top10"] = True
    frame["candidate_count"] = len(frame)
    frame["capped_opportunity_return_7"] = np.minimum(frame["raw_repair_return"], 0.07)
    frame["stage1_score"] = np.linspace(0.9, 0.4, len(frame))
    frame["capped_pair_score"] = np.linspace(0.4, 0.9, len(frame))
    frame["label_available_date"] = "2026-06-09"
    frame["closing_completion_gap"] = 0.5
    frame["strength_x_gap"] = 0.4
    frame["outcome_state"] = np.where(
        frame["target7"].eq(1), "TARGET7",
        np.where(frame["raw_repair_return"].ge(0), "POSITIVE_NON_TARGET", "LOSS"),
    )
    return frame


def test_frozen_contract_and_no_development_scope() -> None:
    assert_diagnostic_contract()
    assert len(FROZEN_FEATURE_COLUMNS) == 18
    assert STAGE1_L2 == 0.30
    assert STAGE1_POSITIVE_WEIGHT == 1.50
    assert PAIR_L2 == 0.30
    assert PAIR_FEATURE_COLUMNS == [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]
    assert MIN_STAGE1_TRAIN_DATES == 18
    assert JULY_RESULT_ROWS_ACCESSED == 0
    assert not NEW_MODEL_DEVELOPED and not NEW_FEATURES and not THRESHOLD_SEARCH


def test_winner_protection_keeps_demoted_stage1_winner() -> None:
    day = _day_fixture()
    assert winner_protection_ids(day) == ["A", "D", "E"]


def test_loss_permission_only_executes_actual_loss_demotion() -> None:
    day = _day_fixture()
    assert loss_permission_ids(day) == ["A", "B", "D"]


def test_stage1_backfill_and_oracle_backfill_are_distinct() -> None:
    day = _day_fixture()
    assert stage1_loss_veto_ids(day, oracle_backfill=False) == ["A", "B", "D"]
    assert stage1_loss_veto_ids(day, oracle_backfill=True) == ["A", "B", "E"]


def test_negative_return_is_not_floored() -> None:
    values = np.asarray([0.20, 0.06, -0.12])
    np.testing.assert_allclose(capped7_utility(values), [0.07, 0.06, -0.12])


def test_strict_temporal_reconstruction_and_no_july(diagnostic_result) -> None:
    outputs, context = diagnostic_result
    assert set(outputs) == set(OUTPUT_FILENAMES)
    assert context["strict_dates"] == 17
    assert context["unavailable_dates"] == [
        "2026-06-01", "2026-06-02", "2026-06-29", "2026-06-30"
    ]
    assert context["self_label_leakage_rows"] == 0
    assert context["current_test_leakage_rows"] == 0
    assert context["july_result_rows_accessed"] == 0
    oof = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[0]]), encoding="utf-8-sig")
    temporal = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[1]]), encoding="utf-8-sig")
    assert oof["signal_date"].astype(str).lt(JULY_SENTINEL).all()
    assert oof["label_available_date"].astype(str).lt(JULY_SENTINEL).all()
    available = temporal[temporal["strict_date_available"].astype(str).str.lower().eq("true")]
    assert available["matured_training_dates"].ge(18).all()
    assert (
        available["latest_training_label_available_date"].astype(str)
        < available["test_date"].astype(str)
    ).all()
    assert int(temporal["self_label_leakage_rows"].sum()) == 0
    assert int(temporal["current_test_leakage_rows"].sum()) == 0
    assert int(temporal["july_rows_accessed"].sum()) == 0


def test_swap_attribution_closes_exactly(diagnostic_result) -> None:
    outputs, context = diagnostic_result
    assert context["attribution"]["closure"] == "PASS"
    assert np.isclose(
        context["attribution"]["attributed_total"],
        context["attribution"]["observed_delta"],
        rtol=0.0,
        atol=1e-10,
    )
    swaps = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[2]]), encoding="utf-8-sig")
    assert np.isclose(
        swaps["overall_mean_contribution"].sum(),
        context["attribution"]["observed_delta"],
        rtol=0.0,
        atol=1e-10,
    )


def test_policy_ranking_is_deterministic() -> None:
    day = _day_fixture()
    ranked1 = add_policy_ranks(day)
    ranked2 = add_policy_ranks(day.sample(frac=1.0, random_state=7))
    columns = [column for column in ranked1 if column.startswith("rank_")]
    left = ranked1.sort_values("event_id")[columns].reset_index(drop=True)
    right = ranked2.sort_values("event_id")[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_strict_observed_policies_are_not_legacy_outputs(diagnostic_result) -> None:
    _, context = diagnostic_result
    practical = context["practical"]
    stage1 = practical[
        practical["scope"].eq("COMBINED") & practical["policy"].eq(POLICY_STAGE1)
    ].iloc[0]
    capped = practical[
        practical["scope"].eq("COMBINED") & practical["policy"].eq(POLICY_CAPPED)
    ].iloc[0]
    assert np.isfinite(float(stage1["Top3_mean"]))
    assert np.isfinite(float(capped["Top3_mean"]))
    assert not (
        np.isclose(float(stage1["Top3_mean"]), 0.018539, atol=1e-6)
        and np.isclose(float(capped["Top3_mean"]), 0.030443, atol=1e-6)
    )


def test_outputs_are_nonempty(diagnostic_result) -> None:
    outputs, _ = diagnostic_result
    assert all(payload for payload in outputs.values())

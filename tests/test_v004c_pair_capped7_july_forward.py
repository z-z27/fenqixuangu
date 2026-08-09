from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_pair_capped7_july_forward import (
    AUGUST_SIGNAL_SENTINEL,
    FROZEN_DEVELOPMENT_COMMIT,
    FORWARD_STRESS_CLASS,
    JULY_END,
    JULY_REFIT,
    JULY_START,
    NEW_FEATURES,
    OUTCOME_COLUMNS,
    PAIR_FEATURE_COLUMNS,
    PAIR_L2,
    PREDICTION_COLUMNS,
    STAGE1_L2,
    STAGE1_POSITIVE_WEIGHT,
    STAGE2_INTERCEPT,
    STAGE2_TARGET,
    TOP_K,
    TRAINING_ASOF_DATE,
    TUNING,
    build_forward_outputs,
    build_prediction_lock,
    capped7_utility,
    label_is_available,
    prediction_lock_sha256,
)
from src.v004c_v4a_architecture_transfer import FROZEN_FEATURE_COLUMNS
from src.v004c_v4a_top10_reranker import RESIDUAL_RAW_FIELDS


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def forward_result() -> tuple[dict[str, bytes], dict]:
    return build_forward_outputs(ROOT)


def _prediction_fixture(rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "event_id": [f"E{i:03d}" for i in range(rows)],
        "signal_date": ["2026-07-01"] * rows,
        "code": [f"{i:06d}" for i in range(rows)],
        "board_streak_before_break": [2] * rows,
    })
    for feature in FROZEN_FEATURE_COLUMNS:
        frame[feature] = rng.uniform(0.0, 1.0, rows)
    frame["d1_high_to_close_drawdown_raw"] = np.linspace(0.0, 0.10, rows)
    frame["d1_close_location"] = np.linspace(0.1, 0.9, rows)
    frame["d1_low_to_close_recovery"] = np.linspace(0.0, 0.08, rows)
    frame["d1_open_to_close_return_raw"] = np.linspace(-0.05, 0.05, rows)
    return frame


def test_frozen_constants_match_development_commit() -> None:
    assert FROZEN_DEVELOPMENT_COMMIT == "c6af1289f66f8c2cad610b9564573607ce57ac14"
    assert FORWARD_STRESS_CLASS == "SEMI_OOT_CHRONOLOGICAL"
    assert STAGE1_L2 == 0.30
    assert STAGE1_POSITIVE_WEIGHT == 1.50
    assert PAIR_L2 == 0.30
    assert STAGE2_TARGET == "CAPPED7"
    assert STAGE2_INTERCEPT == 0.0
    assert TOP_K == 10
    assert PAIR_FEATURE_COLUMNS == [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]
    assert len(FROZEN_FEATURE_COLUMNS) == 18
    assert not NEW_FEATURES and not TUNING and not JULY_REFIT


def test_capped_target_is_upper_cap_only() -> None:
    values = np.asarray([0.20, 0.08, 0.06, 0.02, -0.05, -0.12])
    actual = capped7_utility(values)
    expected = np.asarray([0.07, 0.07, 0.06, 0.02, -0.05, -0.12])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_label_maturity_is_strictly_before_asof() -> None:
    assert label_is_available("2026-06-30", TRAINING_ASOF_DATE)
    assert not label_is_available("2026-07-01", TRAINING_ASOF_DATE)
    assert not label_is_available("2026-07-02", TRAINING_ASOF_DATE)
    assert not label_is_available(None, TRAINING_ASOF_DATE)


def test_prediction_lock_contains_no_outcome_and_is_outcome_independent() -> None:
    frame = _prediction_fixture()
    stage1_beta = np.zeros(len(FROZEN_FEATURE_COLUMNS) + 1)
    stage2_beta = np.zeros(len(PAIR_FEATURE_COLUMNS))
    first = build_prediction_lock(frame, stage1_beta, stage2_beta)
    perturbed = frame.copy()
    for column, value in {
        "target7": 1,
        "d2_open": -999.0,
        "d3_high": 999.0,
        "raw_return": -123.0,
        "capped_return": 456.0,
    }.items():
        perturbed[column] = value
    second = build_prediction_lock(perturbed, stage1_beta, stage2_beta)
    assert not OUTCOME_COLUMNS.intersection(first.columns)
    assert list(first.columns) == PREDICTION_COLUMNS
    assert prediction_lock_sha256(first) == prediction_lock_sha256(second)


def test_top10_restriction_and_tie_break() -> None:
    frame = _prediction_fixture()
    stage1_beta = np.zeros(len(FROZEN_FEATURE_COLUMNS) + 1)
    stage2_beta = np.zeros(len(PAIR_FEATURE_COLUMNS))
    lock = build_prediction_lock(frame, stage1_beta, stage2_beta)
    assert int(lock["stage1_top10"].sum()) == 10
    outside = lock[~lock["stage1_top10"]]
    assert outside["capped_pair_score"].isna().all()
    assert outside["capped_pair_rank"].isna().all()
    top10 = lock[lock["stage1_top10"]]
    assert top10.sort_values("stage1_rank")["capped_pair_rank"].tolist() == list(range(1, 11))


def test_forward_universe_and_prediction_lock(forward_result) -> None:
    outputs, context = forward_result
    assert context["july_rows"] == 178
    assert context["july_dates"] == 23
    assert context["board2"] == 155
    assert context["board3"] == 23
    assert context["outcome_missing"] == 0
    assert context["outcome_complete"] == 178
    assert context["outcome_perturbation"] == "PASS"
    assert len(context["prediction_lock_sha256"]) == 64
    lock = pd.read_csv(
        BytesIO(outputs["v004c_july_forward_prediction_lock_v001.csv"]),
        encoding="utf-8-sig",
    )
    assert set(lock["signal_date"].astype(str)).issubset(
        set(pd.date_range(JULY_START, JULY_END).strftime("%Y-%m-%d"))
    )
    assert not set(OUTCOME_COLUMNS).intersection(lock.columns)
    assert not lock["signal_date"].astype(str).ge(AUGUST_SIGNAL_SENTINEL).any()


def test_training_snapshot_is_mature_and_fitted_once(forward_result) -> None:
    _, context = forward_result
    audit = context["training_audit"]
    assert audit["eligible_historical_rows"] == 307
    assert audit["eligible_historical_dates"] == 37
    assert audit["latest_eligible_signal_date"] == "2026-06-26"
    assert audit["latest_label_available_date"] == "2026-06-30"
    assert audit["self_label_leakage_rows"] == 0
    assert audit["july_training_rows"] == 0
    assert audit["august_training_rows"] == 0
    assert context["maturity"]["june_immature_label_rows"] > 0
    assert context["maturity"]["june_folds_with_immature_labels"] > 0


def test_august_is_outcome_only(forward_result) -> None:
    outputs, context = forward_result
    assert context["august_outcome_only"] is True
    evaluated = pd.read_csv(
        BytesIO(outputs["v004c_july_forward_oof_v001.csv"]),
        encoding="utf-8-sig",
    )
    assert not evaluated["signal_date"].astype(str).ge(AUGUST_SIGNAL_SENTINEL).any()
    assert evaluated["d3_date"].astype(str).ge(AUGUST_SIGNAL_SENTINEL).any()


def test_stage2_formula_and_top10_prediction_complete(forward_result) -> None:
    outputs, _ = forward_result
    lock = pd.read_csv(
        BytesIO(outputs["v004c_july_forward_prediction_lock_v001.csv"]),
        encoding="utf-8-sig",
    )
    top10 = lock[lock["stage1_top10"].astype(str).str.lower().eq("true")]
    np.testing.assert_allclose(
        top10["strength_x_gap"],
        top10["stage1_strength"] * top10["closing_completion_gap"],
        rtol=0.0,
        atol=1e-11,
    )
    assert top10["capped_pair_score"].notna().all()
    assert top10["capped_pair_rank"].notna().all()


def test_outputs_are_complete_and_deterministic(forward_result) -> None:
    outputs, _ = forward_result
    assert set(outputs) == {
        "v004c_july_forward_prediction_lock_v001.csv",
        "v004c_july_forward_training_audit_v001.csv",
        "v004c_july_forward_oof_v001.csv",
        "v004c_july_forward_daily_v001.csv",
        "v004c_july_forward_practical_v001.csv",
        "v004c_july_forward_membership_v001.csv",
        "v004c_july_forward_review_v001.md",
    }
    assert all(value for value in outputs.values())

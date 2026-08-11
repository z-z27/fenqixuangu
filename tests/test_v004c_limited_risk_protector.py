from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_limited_risk_protector import (
    EXPECTED_STRICT_DATES,
    EXPECTED_UNAVAILABLE_DATES,
    JULY_RESULT_ROWS_ACCESSED,
    MODEL_ID,
    OUTPUT_FILENAMES,
    RECURSIVE_VETO,
    RISK_CLASS_WEIGHTING,
    RISK_FEATURE_COLUMNS,
    RISK_FEATURE_SELECTION,
    RISK_L2,
    RISK_THRESHOLD,
    RISK_THRESHOLD_SEARCH,
    RISK_TAIL_WEIGHTING,
    apply_limited_policy,
    assert_risk_contract,
    build_outputs,
    build_veto_attribution,
    fixed_risk_veto,
    limited_veto_selection,
    loss_target,
    prediction_outcome_independence,
    risk_date_sample_weight,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def risk_result() -> tuple[dict[str, bytes], dict]:
    return build_outputs(ROOT)


def _policy_fixture(probabilities: list[float], candidates: int = 5) -> pd.DataFrame:
    rows = []
    for rank in range(1, candidates + 1):
        rows.append({
            "event_id": chr(64 + rank),
            "signal_date": "2026-06-10",
            "code": f"{rank:06d}",
            "candidate_count": candidates,
            "stage1_rank": rank,
            "stage1_strength": 1.0 - (rank - 1) / max(candidates - 1, 1),
            "closing_completion_gap": 0.5,
            "strength_x_gap": 0.5 * (1.0 - (rank - 1) / max(candidates - 1, 1)),
            "risk_p_loss": probabilities[rank - 1] if rank <= 3 else np.nan,
            "raw_repair_return": [0.08, 0.02, -0.04, 0.07, -0.06][rank - 1],
            "target7": int([0.08, 0.02, -0.04, 0.07, -0.06][rank - 1] >= 0.07),
            "capped_opportunity_return_7": min([0.08, 0.02, -0.04, 0.07, -0.06][rank - 1], 0.07),
        })
    return pd.DataFrame(rows)


def test_frozen_risk_contract() -> None:
    assert_risk_contract()
    assert RISK_FEATURE_COLUMNS == [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]
    assert RISK_L2 == 0.30
    assert RISK_THRESHOLD == 0.50
    assert not RISK_CLASS_WEIGHTING
    assert not RISK_TAIL_WEIGHTING
    assert not RISK_THRESHOLD_SEARCH
    assert not RISK_FEATURE_SELECTION
    assert not RECURSIVE_VETO
    assert JULY_RESULT_ROWS_ACCESSED == 0


def test_loss_target_boundary_is_strictly_below_zero() -> None:
    values = np.asarray([0.08, 0.02, 0.0, -0.0001, -0.12])
    np.testing.assert_array_equal(loss_target(values), [0, 0, 0, 1, 1])


def test_fixed_threshold_boundary() -> None:
    assert fixed_risk_veto(0.499999) is False
    assert fixed_risk_veto(0.500000) is True
    assert fixed_risk_veto(0.800000) is True


def test_nonrecursive_single_veto_uses_stage1_backfill() -> None:
    day = _policy_fixture([0.80, 0.10, 0.20, 0.99, 0.10])
    selected, executed, backfilled = limited_veto_selection(day)
    assert selected == ["B", "C", "D"]
    assert executed == ["A"]
    assert backfilled == ["D"]


def test_multiple_veto_preserves_survivor_then_rank4_rank5() -> None:
    day = _policy_fixture([0.80, 0.70, 0.10, 0.99, 0.99])
    selected, executed, backfilled = limited_veto_selection(day)
    assert selected == ["C", "D", "E"]
    assert executed == ["A", "B"]
    assert backfilled == ["D", "E"]


def test_no_replacement_restores_flagged_member() -> None:
    day = _policy_fixture([0.80, 0.10, 0.20], candidates=3)
    selected, executed, backfilled = limited_veto_selection(day)
    assert selected == ["B", "C", "A"]
    assert executed == []
    assert backfilled == []


def test_date_equal_weighting() -> None:
    meta = pd.DataFrame({
        "signal_date": ["d1", "d1", "d1", "d2", "d2"],
        "event_id": ["a", "b", "c", "d", "e"],
    })
    weight = risk_date_sample_weight(meta)
    assert np.isclose(weight[:3].sum(), 1.0)
    assert np.isclose(weight[3:].sum(), 1.0)


def test_prediction_is_outcome_independent() -> None:
    day = _policy_fixture([0.1, 0.1, 0.1, 0.1, 0.1])
    beta = np.asarray([-0.2, 0.1, 0.2, -0.1])
    assert prediction_outcome_independence(day, beta)


def test_strict_temporal_parity_and_training_scope(risk_result) -> None:
    outputs, context = risk_result
    assert set(outputs) == set(OUTPUT_FILENAMES)
    assert context["strict_dates"] == EXPECTED_STRICT_DATES
    assert context["unavailable_dates"] == EXPECTED_UNAVAILABLE_DATES
    assert context["self_label_leakage_rows"] == 0
    assert context["current_test_leakage_rows"] == 0
    assert context["july_result_rows_accessed"] == 0
    assert context["stage1_parity"]
    assert context["training"]["all_rows_stage1_top3"]
    assert context["training"]["outcome_perturbation_pass"]
    oof = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[0]]), encoding="utf-8-sig")
    audit = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[1]]), encoding="utf-8-sig")
    assert oof["signal_date"].astype(str).lt("2026-07-01").all()
    assert oof.loc[oof["original_stage1_top3"], "risk_p_loss"].notna().all()
    assert oof.loc[~oof["original_stage1_top3"], "risk_p_loss"].isna().all()
    assert (~oof.loc[~oof["original_stage1_top3"], "risk_veto"]).all()
    assert audit["feature_count"].eq(3).all()
    assert audit["l2"].eq(0.30).all()
    assert audit["threshold"].eq(0.50).all()
    assert audit["july_rows_accessed"].eq(0).all()


def test_final_policy_size_and_no_recursive_veto(risk_result) -> None:
    outputs, _ = risk_result
    oof = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[0]]), encoding="utf-8-sig")
    for _, day in oof.groupby("signal_date", sort=True):
        assert int(day["final_limited_risk_rank"].le(min(3, len(day))).sum()) == min(3, len(day))
        backfilled = day[
            day["final_limited_risk_rank"].le(3) & ~day["original_stage1_top3"]
        ].sort_values("stage1_rank")
        if len(backfilled):
            assert backfilled["stage1_rank"].tolist() == list(
                range(4, 4 + len(backfilled))
            )


def test_attribution_closes(risk_result) -> None:
    outputs, context = risk_result
    assert context["accounting"]["closure"] == "PASS"
    assert np.isclose(
        context["accounting"]["attributed_total"],
        context["accounting"]["observed_delta"],
        rtol=0.0,
        atol=1e-10,
    )
    attribution = pd.read_csv(BytesIO(outputs[OUTPUT_FILENAMES[3]]), encoding="utf-8-sig")
    if len(attribution):
        assert np.isclose(
            attribution["overall_mean_contribution"].sum(),
            context["accounting"]["observed_delta"],
            rtol=0.0,
            atol=1e-10,
        )


def test_apply_policy_is_deterministic_under_row_shuffle() -> None:
    day = _policy_fixture([0.80, 0.10, 0.20, 0.99, 0.10])
    beta = np.asarray([0.0, 0.0, 0.0, 0.0])
    first = apply_limited_policy(day, beta).sort_values("event_id")
    second = apply_limited_policy(day.sample(frac=1.0, random_state=17), beta).sort_values("event_id")
    pd.testing.assert_series_equal(
        first["final_limited_risk_rank"].reset_index(drop=True),
        second["final_limited_risk_rank"].reset_index(drop=True),
    )


def test_outputs_are_nonempty(risk_result) -> None:
    outputs, context = risk_result
    assert all(payload for payload in outputs.values())
    assert context["decision"]["risk_protector_signal"] in {
        "STRONG", "PARTIAL", "ABSENT"
    }
    assert MODEL_ID in set(context["practical"]["policy"])

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.v004c_d1_coarse_retrieval_capacity_audit import (
    ALL_KS,
    OUTPUT_FILENAMES,
    RETRIEVAL_KS,
    _capacity_row,
    _random_topk_metrics,
    analyze,
    assert_contract,
    build_outputs,
    load_frozen_population,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def full_context():
    return analyze(ROOT)


def _toy() -> pd.DataFrame:
    return pd.DataFrame([
        {"event_id": "a", "signal_date": "2026-05-06", "month": "MAY", "s2_rank": 1, "target7": 1, "loss": 0, "severe_loss": 0},
        {"event_id": "b", "signal_date": "2026-05-06", "month": "MAY", "s2_rank": 2, "target7": 0, "loss": 1, "severe_loss": 1},
        {"event_id": "c", "signal_date": "2026-05-06", "month": "MAY", "s2_rank": 3, "target7": 1, "loss": 0, "severe_loss": 0},
        {"event_id": "d", "signal_date": "2026-05-07", "month": "MAY", "s2_rank": 1, "target7": 0, "loss": 0, "severe_loss": 0},
        {"event_id": "e", "signal_date": "2026-05-07", "month": "MAY", "s2_rank": 2, "target7": 1, "loss": 0, "severe_loss": 0},
    ])


def test_contract_has_only_preregistered_k_values() -> None:
    assert_contract()
    assert ALL_KS == (3, 4, 5, 6, 7)
    assert RETRIEVAL_KS == (4, 5, 6, 7)


def test_capacity_uses_effective_k_and_date_macro_recall() -> None:
    row = _capacity_row(_toy(), 1, "POOLED")
    assert row["target7_row_recall"] == pytest.approx(1 / 3)
    assert row["date_macro_target7_recall"] == pytest.approx(0.25)
    assert row["winner_presence_hit_rate"] == pytest.approx(0.5)
    assert row["all_winners_retained_rate"] == pytest.approx(0.0)


def test_random_baseline_is_deterministic_and_bounded() -> None:
    a = _random_topk_metrics(_toy(), (1, 2), repetitions=200, seed=17)
    b = _random_topk_metrics(_toy(), (1, 2), repetitions=200, seed=17)
    for metric in a:
        assert (a[metric] == b[metric]).all()
        assert ((a[metric] >= 0) & (a[metric] <= 1)).all()


def test_real_population_parity_and_no_august() -> None:
    frame, audit = load_frozen_population(ROOT)
    assert (len(frame), frame["signal_date"].nunique()) == (485, 60)
    assert frame["signal_date"].max() == "2026-07-29"
    assert frame["label_available_date"].lt("2026-08-01").all()
    assert audit["model_trained"] == "NO"
    assert audit["august_used"] == "NO"
    assert audit["frozen_rank_mismatch_count"] == 0
    assert audit["frozen_score_parity_max_abs_error"] <= 1e-12


def test_actual_decision_does_not_invent_an_operating_point(full_context) -> None:
    assert full_context["decision"]["state"] in {
        "D1_COARSE_RETRIEVAL_SUPPORTED",
        "D1_COARSE_RETRIEVAL_PARTIAL",
        "D1_COARSE_RETRIEVAL_NOT_SUPPORTED",
    }
    if full_context["decision"]["state"] != "D1_COARSE_RETRIEVAL_SUPPORTED":
        assert full_context["decision"]["recommended_k"] is None


def test_outputs_are_exact_and_exclude_august(full_context) -> None:
    outputs = build_outputs(full_context)
    assert set(outputs) == set(OUTPUT_FILENAMES)
    assert len(outputs) == 11
    for name, payload in outputs.items():
        text = payload.decode("utf-8-sig")
        assert "2026-08-" not in text, name
    assert full_context["audit"]["model_trained"] == "NO"
    assert full_context["audit"]["feature_search"] == "NO"
    assert full_context["audit"]["parameter_search"] == "NO"


def test_full_analysis_is_deterministic(full_context) -> None:
    assert build_outputs(full_context) == build_outputs(full_context)


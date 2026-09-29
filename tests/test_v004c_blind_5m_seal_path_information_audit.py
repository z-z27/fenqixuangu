from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.v004c_blind_5m_seal_path_information_audit import (
    F1,
    F2,
    F3,
    FIXED_PROXIES,
    OUTPUT_FILENAMES,
    _make_pairs,
    _monthly_stability,
    analyze,
    assert_contract,
    build_outputs,
    load_analysis_population,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def full_context():
    return analyze(ROOT)


def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "t", "signal_date": "2026-05-06", "month": "MAY", "board_group": "BOARD2", "d0_source": "BAOSTOCK", "target7": 1, "loss": 0, "s2_score": .4, "s2_rank": 4, F1: 20.0, F2: 0.0, F3: -5.0, "F1_valid": 1, "F2_valid": 1, "F3_valid": 1},
            {"event_id": "l", "signal_date": "2026-05-06", "month": "MAY", "board_group": "BOARD2", "d0_source": "BAOSTOCK", "target7": 0, "loss": 1, "s2_score": .5, "s2_rank": 2, F1: 40.0, F2: 1.0, F3: 10.0, "F1_valid": 1, "F2_valid": 1, "F3_valid": 1},
        ]
    )


def test_contract_is_exact_three_frozen_proxies() -> None:
    assert_contract()
    assert FIXED_PROXIES == (F1, F2, F3)


def test_continuous_concordance_and_wrong_pair_are_exact() -> None:
    pairs = _make_pairs(_toy(), F1)
    assert len(pairs) == 1
    assert pairs.iloc[0]["quality_concordance_value"] == 1.0
    assert pairs.iloc[0]["s2_wrong_pair"] == 1


def test_f2_concordance_excludes_ties() -> None:
    pairs = _make_pairs(_toy(), F2)
    assert pairs.iloc[0]["quality_better"] == 1
    assert pairs.iloc[0]["quality_tie"] == 0


def test_direction_flip_rule_is_predeclared() -> None:
    summary = pd.DataFrame(
        [
            {"feature": F1, "period": "MAY", "quality_oriented_concordance": .60},
            {"feature": F1, "period": "JUNE", "quality_oriented_concordance": .58},
            {"feature": F1, "period": "JULY", "quality_oriented_concordance": .40},
            {"feature": F2, "period": "MAY", "quality_oriented_concordance": .50},
            {"feature": F2, "period": "JUNE", "quality_oriented_concordance": .50},
            {"feature": F2, "period": "JULY", "quality_oriented_concordance": .50},
            {"feature": F3, "period": "MAY", "quality_oriented_concordance": .50},
            {"feature": F3, "period": "JUNE", "quality_oriented_concordance": .50},
            {"feature": F3, "period": "JULY", "quality_oriented_concordance": .50},
        ]
    )
    result = _monthly_stability(summary)
    row = result[result["feature"].eq(F1)].iloc[0]
    assert row["temporal_direction_flip"] == "YES"
    assert row["source_effect_risk"] == "TEMPORAL_OR_SOURCE_CONFOUNDING"


def test_real_population_parity_and_boundaries() -> None:
    frame, audit = load_analysis_population(ROOT)
    assert (len(frame), frame["signal_date"].nunique()) == (485, 60)
    assert frame["signal_date"].max() == "2026-07-29"
    assert frame["label_available_date"].lt("2026-08-01").all()
    assert audit["model_trained"] == "NO"
    assert audit["model_refits"] == 0


def test_outputs_exclude_return_magnitude_and_august(full_context) -> None:
    outputs = build_outputs(full_context)
    assert set(outputs) == set(OUTPUT_FILENAMES)
    forbidden = ("raw_repair_return", "capped_return")
    for name, payload in outputs.items():
        text = payload.decode("utf-8-sig")
        assert all(token not in text for token in forbidden), name
    population = full_context["analysis_population"]
    assert population["signal_date"].max() == "2026-07-29"


def test_full_analysis_is_deterministic(full_context) -> None:
    assert build_outputs(full_context) == build_outputs(full_context)
    assert full_context["audit"]["model_refits"] == 0
    assert full_context["audit"]["august_used_in_primary_analysis"] == "NO"

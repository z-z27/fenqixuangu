from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_pre_break_board_day_seal_path_information_audit import (
    FAMILY_READY_FIELD_MINIMUM,
    FIXED_FIELDS,
    OUTPUT_FILENAMES,
    _pair_table,
    classify_data_state,
    parse_pool_time,
    run_phase_a,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_five_field_contract() -> None:
    assert FIXED_FIELDS == (
        "F1_D0_FINAL_SEAL_MINUTE",
        "F2_D0_OPEN_BOARD_COUNT",
        "F3_D0_RESEAL_DELAY_MINUTE",
        "F4_FINAL_SEAL_DETERIORATION",
        "F5_OPEN_BOARD_DETERIORATION",
    )
    assert FAMILY_READY_FIELD_MINIMUM == 3


def test_pool_time_parser_is_strict_and_uses_0930_origin() -> None:
    minute, seconds, status = parse_pool_time("092505")
    assert status == "VALID"
    assert np.isclose(minute, -4.916666666666667)
    assert seconds == 9 * 3600 + 25 * 60 + 5
    assert parse_pool_time("145251")[2] == "VALID"
    assert parse_pool_time("")[2] == "UNPARSEABLE"
    assert parse_pool_time(None)[2] == "MISSING"
    assert parse_pool_time("250000")[2] == "UNPARSEABLE"


def test_phase_a_is_blind_and_reproduces_frozen_population() -> None:
    phase_a = run_phase_a(ROOT)
    population = phase_a["population"]
    assert (len(population), population["signal_date"].nunique()) == (485, 60)
    assert population.groupby("month").size().to_dict() == {
        "MAY": 146, "JUNE": 173, "JULY": 166,
    }
    forbidden = {
        "target7", "loss", "severe_loss", "raw_repair_return", "capped_return_7",
        "stage1_score", "stage1_rank", "s2_score", "s2_rank",
    }
    assert not forbidden.intersection(population.columns)
    assert not forbidden.intersection(phase_a["eligibility"].columns)
    assert not forbidden.intersection(phase_a["values"].columns)
    assert phase_a["outcome_accessed"] == "NO"
    assert len(phase_a["phase_a_lock_sha256"]) == 64


def test_daily_derived_rows_remain_missing_and_family_gate_stops() -> None:
    phase_a = run_phase_a(ROOT)
    may = phase_a["eligibility"][phase_a["eligibility"]["month"].eq("MAY")]
    assert may["d0_source"].eq("daily_limitup_derived").all()
    assert may[["F1_valid", "F2_valid", "F3_valid", "F4_valid", "F5_valid"]].to_numpy().sum() == 0
    assert phase_a["data_state"] == "NOT_READY"
    assert len(phase_a["eligible_fields"]) < 3
    assert phase_a["coverage"]["field_data_ready"].eq("NO").all()
    # Pool-native streak counts are cross-checks, not replacements for the
    # authoritative Board2/Board3 candidate definition.
    mismatches = phase_a["eligibility"]["lineage_notes"].str.contains(
        "POOL_STREAK_CROSSCHECK_MISMATCH", na=False
    )
    assert mismatches.any()
    assert phase_a["eligibility"].loc[mismatches, "lineage_violation"].eq(0).all()


def test_data_state_gate_is_exact() -> None:
    frame = pd.DataFrame({
        "field": FIXED_FIELDS,
        "field_data_ready": ["YES", "YES", "YES", "NO", "NO"],
    })
    state, eligible = classify_data_state(frame)
    assert state == "PARTIAL"
    assert eligible == FIXED_FIELDS[:3]
    frame["field_data_ready"] = "YES"
    assert classify_data_state(frame)[0] == "READY"
    frame["field_data_ready"] = ["YES", "YES", "NO", "NO", "NO"]
    assert classify_data_state(frame)[0] == "NOT_READY"


def test_quality_orientation_is_frozen_lower_is_stronger() -> None:
    frame = pd.DataFrame([
        {"signal_date": "2026-06-01", "event_id": "T", "target7": 1, "loss": 0, "s2_score": .4, "x": 2.0},
        {"signal_date": "2026-06-01", "event_id": "L", "target7": 0, "loss": 1, "s2_score": .5, "x": 5.0},
    ])
    pairs = _pair_table(frame, "x")
    assert len(pairs) == 1
    assert pairs.iloc[0]["raw_concordance"] == 0.0
    assert pairs.iloc[0]["quality_oriented_concordance"] == 1.0
    assert pairs.iloc[0]["s2_wrong_pair"] == 1


def test_outputs_stop_before_phase_b_and_are_deterministic() -> None:
    output_dir, audit = write_outputs(ROOT)
    first = {name: (output_dir / name).read_bytes() for name in OUTPUT_FILENAMES}
    second_dir, second_audit = write_outputs(ROOT)
    second = {name: (second_dir / name).read_bytes() for name in OUTPUT_FILENAMES}
    assert first == second
    assert audit["output_sha256"] == second_audit["output_sha256"]
    assert set(first) == set(OUTPUT_FILENAMES)
    assert audit["data_state"] == "NOT_READY"
    assert audit["information_state"] == "NOT_TESTED_DUE_TO_DATA"
    assert audit["phase_b_run"] is False
    assert audit["outcome_accessed"] == "NO"
    assert audit["model_trained"] == "NO"
    review = first[OUTPUT_FILENAMES[-1]].decode("utf-8")
    assert "PREBREAK_SEAL_PATH_DATA_STATE = NOT_READY" in review
    assert "PREBREAK_SEAL_PATH_INFORMATION_STATE = NOT_TESTED_DUE_TO_DATA" in review
    assert "Phase A 只有 0/5 字段通过门槛" in review
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review


def test_module_contains_no_learner_or_feature_window_search() -> None:
    text = (ROOT / "src/v004c_pre_break_board_day_seal_path_information_audit.py").read_text(encoding="utf-8")
    forbidden = (
        "fit_logistic", "GradientBoostingClassifier(", "RandomForestClassifier(",
        "XGBClassifier(", "PCA(", "KMeans(", "seal_amount / amount",
    )
    assert not any(token in text for token in forbidden)

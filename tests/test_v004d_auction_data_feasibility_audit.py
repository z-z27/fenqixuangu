from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.v004d_auction_data_feasibility_audit import (
    FORBIDDEN_INPUT_COLUMNS,
    INPUT_COLUMNS,
    OUTPUT_FILENAMES,
    SCALE_SIZES,
    LiveWindowError,
    build_dry_run_timing,
    build_outputs,
    calculate_basic_live_fields,
    collect_live_snapshot,
    load_blind_candidate_universe,
)


ROOT = Path(__file__).resolve().parents[1]


def test_blind_projection_and_authoritative_universe():
    assert not (set(INPUT_COLUMNS) & set(FORBIDDEN_INPUT_COLUMNS))
    frame, audit = load_blind_candidate_universe(ROOT)
    assert len(frame) == 497
    assert frame["signal_date"].nunique() == 62
    assert audit["primary_candidate_d2_rows"] == 492
    assert audit["august_availability_rows"] == 5
    assert set(frame["board_group"]) == {"BOARD2", "BOARD3"}
    assert audit["outcome_accessed"] == "NO"


def test_basic_live_calculation_is_deterministic_and_not_a_model():
    candidates = pd.DataFrame({
        "code": ["000001", "600001"],
        "d1_close": [10.0, 20.0],
        "d1_volume": [1_000.0, 2_000.0],
    })
    snapshot = pd.DataFrame({
        "code": ["600001", "000001"],
        "auction_price": [21.0, 10.5],
        "auction_volume": [200.0, 100.0],
        "auction_amount": [4_200.0, 1_050.0],
        "bid1_price": [20.9, 10.4],
        "bid1_volume": [80.0, 60.0],
        "ask1_price": [21.1, 10.6],
        "ask1_volume": [20.0, 40.0],
        "source_timestamp": ["x", "x"],
    })
    result = calculate_basic_live_fields(candidates, snapshot)
    assert result["code"].tolist() == ["000001", "600001"]
    assert np.allclose(result["auction_pct_from_d1_close"], [.05, .05])
    assert np.allclose(result["auction_volume_ratio"], [.1, .1])
    assert result["core_fields_complete"].eq(1).all()


def test_live_collector_rejects_non_window_and_logs_real_window_with_fake_source():
    candidates = pd.DataFrame({"code": ["000001"]})
    with pytest.raises(LiveWindowError):
        collect_live_snapshot(
            candidates,
            lambda _: pd.DataFrame(),
            now=datetime(2026, 9, 6, 9, 26, tzinfo=ZoneInfo("Asia/Shanghai")),
        )

    def fake(_: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "code": ["000001"], "auction_price": [10.0],
            "auction_volume": [100.0], "auction_amount": [1000.0],
            "bid1_price": [9.99], "bid1_volume": [10.0],
            "ask1_price": [10.01], "ask1_volume": [12.0],
            "source_timestamp": ["2026-09-07T09:25:10+08:00"],
        })

    result = collect_live_snapshot(
        candidates, fake,
        now=datetime(2026, 9, 7, 9, 26, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert len(result.normalized) == 1
    assert result.metrics["success_count"] == 1
    assert result.metrics["candidate_count"] == 1


def test_fixed_scale_dry_run_only_tests_parser_capacity():
    summary, detail = build_dry_run_timing(repetitions=2)
    assert tuple(summary["candidate_count"]) == SCALE_SIZES
    assert set(detail["live_gate_eligible"]) == {0}
    assert summary["core_field_completeness"].eq(1).all()
    assert summary["interpretation"].str.contains("excludes network fetch").all()


def test_real_outputs_are_complete_blind_and_fail_closed():
    outputs, context = build_outputs(ROOT)
    assert tuple(outputs) == OUTPUT_FILENAMES
    assert len(outputs) == 12
    assert context["history"]["historical_core_coverage"] == 0
    assert context["history"]["historical_state"] == "NOT_READY"
    assert context["formal"]["live_state"] == "PENDING_REAL_TRADING_DAY_VALIDATION"
    assert context["formal"]["semantics_state"] == "UNRESOLVED"
    assert context["formal"]["feasibility_state"] == "V004D_AUCTION_DATA_NOT_FEASIBLE"
    assert context["formal"]["next_action"] == "STOP_AUCTION_PIPELINE"
    assert context["outcome_accessed"] == "NO"
    assert context["model_trained"] == "NO"
    review = outputs[OUTPUT_FILENAMES[-1]].decode("utf-8")
    assert "OUTCOME_ACCESSED = NO" in review
    assert "MODEL_TRAINED = NO" in review
    assert "BUY_PASS_RULE_CREATED = NO" in review
    assert "P95_TOTAL_PIPELINE_TIME = NOT_MEASURED_LIVE" in review


def test_no_prohibited_model_or_outcome_materialization():
    text = (ROOT / "src/v004d_auction_data_feasibility_audit.py").read_text(
        encoding="utf-8"
    )
    forbidden_code = (
        "LogisticRegression", "GradientBoosting", "RandomForest", "XGBClassifier",
        ".fit(", "predict_proba", "BUY_PASS_SCORE",
    )
    assert not any(token in text for token in forbidden_code)


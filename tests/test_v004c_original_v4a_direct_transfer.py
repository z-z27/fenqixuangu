from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_original_v4a_direct_transfer import (
    EXPECTED_STARTING_HEAD,
    FEATURE_COLUMNS,
    HYPERPARAMETER_SEARCH,
    JULY_RESULT_ROWS_ACCESSED_MINIMUM,
    NEW_FEATURE,
    NEW_MODEL,
    POSITIVE_WEIGHT,
    L2,
    STAGE2,
    BOARD3_REPAIR,
    audit_temporal_contract,
    build_outputs,
    rank_scores_after_scoring,
    top2_top3_values,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    return build_outputs(ROOT)


def test_starting_contract_and_frozen_scope() -> None:
    assert EXPECTED_STARTING_HEAD == "9fc73b5e2fbbc2ad8cd773f7ffec8b40c380a165"
    assert len(FEATURE_COLUMNS) == 18
    assert L2 == 0.30
    assert POSITIVE_WEIGHT == 1.50
    assert not any((NEW_MODEL, NEW_FEATURE, HYPERPARAMETER_SEARCH, STAGE2, BOARD3_REPAIR))
    assert JULY_RESULT_ROWS_ACCESSED_MINIMUM == 1


def test_original_universe_rank_is_preserved_before_intersection() -> None:
    full = pd.DataFrame({
        "signal_date": ["2026-06-03"] * 10,
        "event_id": [f"e{i:02d}" for i in range(10)],
        "score": np.arange(10, dtype=float),
        "is_v4c": [False, True, False, False, True, False, False, False, True, False],
        "frozen_rank_feature": np.linspace(0, 1, 10),
    })
    full["rank_full"] = rank_scores_after_scoring(full, "score")
    intersection = full[full["is_v4c"]].copy()
    intersection["rank_v4c"] = rank_scores_after_scoring(intersection, "score")
    assert intersection["frozen_rank_feature"].tolist() == pytest.approx([1 / 9, 4 / 9, 8 / 9])
    assert intersection["rank_full"].tolist() == [9, 6, 2]
    assert intersection["rank_v4c"].tolist() == [3, 2, 1]


def test_top2_top3_marginal_definition() -> None:
    top2, top3, marginal = top2_top3_values([0.07, 0.05, -0.06])
    assert top2 == pytest.approx(0.06)
    assert top3 == pytest.approx(0.02)
    assert marginal == pytest.approx(-0.04)


def test_temporal_audit_rejects_unmatured_labels() -> None:
    samples = pd.DataFrame({
        "signal_date": ["2026-06-01", "2026-06-02"],
        "d3_trade_date": ["2026-06-04", "2026-06-05"],
    })
    folds = pd.DataFrame({
        "predict_date": ["2026-06-03"],
        "fold_index": [18],
        "train_start": ["2026-06-01"],
        "train_end": ["2026-06-02"],
    })
    audit = audit_temporal_contract(samples, folds)
    assert audit.loc[0, "current_or_future_label_rows"] == 2
    assert not bool(audit.loc[0, "strict_point_in_time_pass"])


def test_real_provenance_is_blocked_before_outcome_benchmark(built) -> None:
    outputs, context = built
    assert context["provenance"] == "BLOCKED"
    assert context["folds"] == 17
    assert context["leaking_folds"] == 17
    assert context["min_leak_rows"] > 0
    assert context["direct_transfer_signal"] == "INVALID"
    assert context["final_trading_capacity_signal"] == "INVALID"
    assert context["july_result_rows_accessed"].startswith(">=1")
    assert context["july_used_for_model_or_outcome_conclusion"] is False
    assert "v004c_original_v4a_temporal_audit_v001.csv" in outputs
    assert "v004c_original_v4a_rankwise_v001.csv" not in outputs
    assert "v004c_original_v4a_topk_v001.csv" not in outputs
    assert "v004c_original_v4a_robustness_v001.csv" not in outputs


def test_v4c_and_archived_score_coverage_parity(built) -> None:
    _, context = built
    assert context["v4c_rows"] == 319
    assert context["v4c_dates"] == 39
    assert context["matured_rows"] == 307
    assert context["matured_dates"] == 37
    assert context["strict_rows"] == 155
    assert context["strict_dates"] == 17
    assert context["covered_rows"] == 132
    assert context["covered_dates"] == 16
    assert context["coverage"] == pytest.approx(132 / 155)
    assert context["zero_coverage_dates"] == ["2026-06-22"]


def test_output_locks_have_no_outcome_columns(built) -> None:
    outputs, _ = built
    for name in (
        "v004c_original_v4a_full_score_lock_v001.csv",
        "v004c_original_v4a_v4c_intersection_v001.csv",
    ):
        header = outputs[name].decode("utf-8-sig").splitlines()[0].lower()
        for forbidden in ("target7", "loss", "raw_repair_return", "capped_return"):
            assert forbidden not in header


def test_deterministic_outputs(built) -> None:
    outputs, context = built
    outputs2, context2 = build_outputs(ROOT)
    assert outputs == outputs2
    assert context == context2

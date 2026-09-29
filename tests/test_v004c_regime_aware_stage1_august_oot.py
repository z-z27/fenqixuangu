from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_regime_aware_stage1_august_oot import (
    BOARD4PLUS_CUTOFF,
    CHALLENGER_FEATURE_COLUMNS,
    EVALUATION_ASOF,
    L2,
    MODEL_ORDER,
    OUTPUT_FILENAMES,
    PAIR_ORDER_PARITY_MIN,
    POSITIVE_WEIGHT,
    REDUCED_FEATURE_COLUMNS,
    REGIME_FEATURE,
    S2_SPEC,
    TRAINING_ASOF,
    _complete_date_eligibility,
    build_feature_parity,
    load_development,
    prediction_lock_sha256,
    score_and_lock,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_two_model_contract_and_single_feature():
    assert MODEL_ORDER == ("BASELINE_S2", "CHALLENGER_REGIME_SIGNED")
    assert S2_SPEC == "S2_NO_TAIL_L2_010"
    assert L2 == .10
    assert POSITIVE_WEIGHT == 1.50
    assert BOARD4PLUS_CUTOFF == 1.0
    assert CHALLENGER_FEATURE_COLUMNS == (*REDUCED_FEATURE_COLUMNS, REGIME_FEATURE)
    assert REGIME_FEATURE == "regime_signed_close_vwap_rank"
    assert TRAINING_ASOF == "2026-08-01"
    assert EVALUATION_ASOF == "2026-09-01"
    assert len(OUTPUT_FILENAMES) == 13


def test_development_parity_and_raw_rank_pair_order_gate():
    train, _ = load_development(ROOT)
    assert (len(train), train["signal_date"].nunique()) == (485, 60)
    assert train["signal_date"].max() == "2026-07-29"
    assert train["label_available_date"].lt(TRAINING_ASOF).all()
    assert np.array_equal(
        train[REGIME_FEATURE].to_numpy(float),
        train["regime_sign"].to_numpy(float)
        * train["rank_d1_close_vwap_pct"].to_numpy(float),
    )
    parity, audit = build_feature_parity(train)
    assert len(parity) > 0
    assert audit["pair_order_parity_rate"] >= PAIR_ORDER_PARITY_MIN
    assert audit["pair_order_parity_pass"] is True


def test_score_lock_is_outcome_free_and_outcome_perturbation_invariant():
    synthetic = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "signal_date": ["2026-08-03"] * 2 + ["2026-08-04"] * 2,
        "code": ["000001", "000002", "000003", "000004"],
        "board_group": ["BOARD2"] * 4,
        "candidate_count": [2] * 4,
        "board_streak_before_break": [2] * 4,
        "d1_close_to_vwap_raw": [.01, -.01, .02, -.02],
        "board4plus_count": [1, 1, 2, 2],
        "high_board4plus": [False, False, True, True],
        "regime_sign": [1., 1., -1., -1.],
    })
    for index, feature in enumerate(REDUCED_FEATURE_COLUMNS):
        synthetic[feature] = np.asarray([.1, .9, .2, .8]) + index * .001
    synthetic[REGIME_FEATURE] = synthetic["regime_sign"] * synthetic["rank_d1_close_vwap_pct"]
    dates = pd.DataFrame({
        "event_id": synthetic["event_id"],
        "d2_date": ["2026-08-04"] * 4,
        "d3_date": ["2026-08-05"] * 4,
        "label_available_date": ["2026-08-05"] * 4,
        "mature_at_evaluation_asof": [True] * 4,
    })
    beta = {
        "BASELINE_S2": np.arange(8, dtype=float) / 10,
        "CHALLENGER_REGIME_SIGNED": np.arange(9, dtype=float) / 10,
    }
    first = score_and_lock(synthetic, dates, beta)
    changed = synthetic.copy()
    changed["target7"] = [1, 0, 1, 0]
    changed["raw_repair_return"] = [9., -9., 8., -8.]
    second = score_and_lock(changed.drop(columns=["target7", "raw_repair_return"]), dates, beta)
    pd.testing.assert_frame_equal(first, second)
    assert not {"target7", "raw_repair_return", "loss"}.intersection(first.columns)
    assert prediction_lock_sha256(first) == prediction_lock_sha256(second)


def test_topk_evaluation_requires_complete_date_maturity():
    frame = pd.DataFrame({
        "signal_date": ["2026-08-03", "2026-08-03", "2026-08-04", "2026-08-04"],
        "mature_at_evaluation_asof": [True, False, True, True],
    })
    assert _complete_date_eligibility(frame).tolist() == [False, False, True, True]


def test_module_has_no_search_or_alternative_relation_path():
    text = (ROOT / "src/v004c_regime_aware_stage1_august_oot.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "GradientBoosting", "RandomForest", "XGBClassifier", "threshold_grid",
        "feature_subset", "board3_count", "max_board_height", "second_best",
        "positive_weight_grid", "l2_grid",
    )
    assert not any(token in text for token in forbidden)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004a import (
    DEFAULT_CLOSE_RETURN_COLUMN,
    DEFAULT_HIGH_RETURN_COLUMN,
    DEFAULT_TARGET_COLUMN,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
    prepare_v004a_samples,
)
from src.v004c_v4a_architecture_transfer import (
    EXTRA_SCALING,
    FEATURE_SELECTION,
    FROZEN_FEATURE_COLUMNS,
    HYPERPARAMETER_SEARCH,
    INITIAL_TRAIN_DATES,
    L2,
    MODEL_FAMILY,
    NEW_FEATURES,
    POSITIVE_WEIGHT,
    RAW_INPUT_COLUMNS,
    USES_PAIRWISE,
    USES_PCA,
    _sigmoid,
    assert_authoritative_universe,
    assert_frozen_architecture_contract,
    build_outputs,
    load_authoritative_input,
    prepare_transfer_samples,
)
from src.v004c_mechanism_foundation import load_outcomes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark():
    outputs, context = build_outputs(ROOT)
    return outputs, context


def _synthetic_raw() -> pd.DataFrame:
    values = [
        ("a", "2026-05-01", "000001", 10.0, 1.0, 2.0, 60.0, 50.0, 40.0, 1.0, 50.0, 1.0),
        ("b", "2026-05-01", "000002", 20.0, 1.0, 4.0, 70.0, 60.0, 50.0, 2.0, 60.0, 2.0),
        ("c", "2026-05-01", "000003", 40.0, np.nan, 6.0, 80.0, 70.0, 60.0, 3.0, 70.0, 3.0),
    ]
    columns = [
        "event_id", "signal_date", "code", "candidate_base_price",
        "d1_close_ma10_pct", "d1_low_ma10_pct", "trend_hold_score",
        "total_score", "theme_score", "days_since_d0", "active_money_score",
        "d1_close_vwap_pct",
    ]
    frame = pd.DataFrame(values, columns=columns)
    frame["eligible_for_trade"] = True
    frame[DEFAULT_TARGET_COLUMN] = [False, True, True]
    frame[DEFAULT_HIGH_RETURN_COLUMN] = [9.99, 10.0, 12.0]
    frame[DEFAULT_CLOSE_RETURN_COLUMN] = [0.0, 0.0, 0.0]
    return frame


def test_authoritative_identity_319_39_and_no_july() -> None:
    frame = load_authoritative_input(ROOT)
    assert_authoritative_universe(frame)
    assert len(frame) == 319
    assert frame["signal_date"].nunique() == 39
    assert int(frame["signal_date"].str.startswith("2026-05").sum()) == 146
    assert int(frame["signal_date"].str.startswith("2026-06").sum()) == 173
    assert int((frame["board_streak_before_break"] == 2).sum()) == 261
    assert int((frame["board_streak_before_break"] == 3).sum()) == 58
    assert frame["signal_date"].max() == "2026-06-30"


def test_july_sentinel_is_fatal() -> None:
    frame = load_authoritative_input(ROOT)
    changed = frame.copy()
    changed.loc[0, "signal_date"] = "2026-07-01"
    with pytest.raises(RuntimeError, match="July sentinel"):
        assert_authoritative_universe(changed)


def test_exact_frozen_feature_contract() -> None:
    assert_frozen_architecture_contract()
    assert len(FROZEN_FEATURE_COLUMNS) == 18
    assert FROZEN_FEATURE_COLUMNS == [
        "rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct",
        "rank_trend_hold_score", "rank_total_score", "rank_theme_score",
        "rank_days_since_d0", "rank_log_candidate_base_price",
        "rank_active_money_score", "rank_d1_close_vwap_pct",
        "inter_close_low", "inter_close_trend", "inter_total_trend",
        "inter_total_active", "inter_low_active", "spread_close_low",
        "days_since_d0_le1", "days_since_d0_eq2", "days_since_d0_ge3",
    ]


def test_daily_percentile_rank_average_tie_and_missing_fallback() -> None:
    prepared, info, _ = prepare_v004a_samples(_synthetic_raw())
    assert info["v004a_feature_columns"] == FROZEN_FEATURE_COLUMNS
    by_code = prepared.set_index("code")
    assert by_code.loc["000001", "rank_d1_close_ma10_pct"] == pytest.approx(0.75)
    assert by_code.loc["000002", "rank_d1_close_ma10_pct"] == pytest.approx(0.75)
    assert by_code.loc["000003", "rank_d1_close_ma10_pct"] == pytest.approx(0.5)
    assert by_code.loc["000001", "rank_d1_low_ma10_pct"] == pytest.approx(1 / 3)
    assert by_code.loc["000003", "rank_d1_low_ma10_pct"] == pytest.approx(1.0)


def test_log_candidate_price_and_rank() -> None:
    prepared, _, _ = prepare_v004a_samples(_synthetic_raw())
    by_code = prepared.set_index("code")
    assert by_code.loc["000001", "log_candidate_base_price"] == pytest.approx(np.log(10.0))
    assert by_code.loc["000003", "rank_log_candidate_base_price"] == pytest.approx(1.0)


def test_six_interaction_formulas_are_exact() -> None:
    prepared, _, _ = prepare_v004a_samples(_synthetic_raw())
    row = prepared.set_index("code").loc["000002"]
    assert row["inter_close_low"] == pytest.approx(
        row["rank_d1_close_ma10_pct"] * row["rank_d1_low_ma10_pct"]
    )
    assert row["inter_close_trend"] == pytest.approx(
        row["rank_d1_close_ma10_pct"] * row["rank_trend_hold_score"]
    )
    assert row["inter_total_trend"] == pytest.approx(
        row["rank_total_score"] * row["rank_trend_hold_score"]
    )
    assert row["inter_total_active"] == pytest.approx(
        row["rank_total_score"] * row["rank_active_money_score"]
    )
    assert row["inter_low_active"] == pytest.approx(
        row["rank_d1_low_ma10_pct"] * row["rank_active_money_score"]
    )
    assert row["spread_close_low"] == pytest.approx(
        row["rank_d1_close_ma10_pct"] - row["rank_d1_low_ma10_pct"]
    )


def test_day_buckets_are_exact() -> None:
    prepared, _, _ = prepare_v004a_samples(_synthetic_raw())
    buckets = prepared.set_index("code")[[
        "days_since_d0_le1", "days_since_d0_eq2", "days_since_d0_ge3"
    ]]
    assert buckets.loc["000001"].tolist() == [1.0, 0.0, 0.0]
    assert buckets.loc["000002"].tolist() == [0.0, 1.0, 0.0]
    assert buckets.loc["000003"].tolist() == [0.0, 0.0, 1.0]


def test_tail_weight_thresholds_exact() -> None:
    prepared, _, _ = prepare_v004a_samples(_synthetic_raw())
    assert prepared.sort_values("code")["tail_weight"].tolist() == [1.0, 1.5, 2.0]


def test_sample_weight_parity_date_class_tail() -> None:
    prepared, _, _ = prepare_v004a_samples(_synthetic_raw())
    weight = build_training_sample_weight(prepared, positive_weight=POSITIVE_WEIGHT)
    expected = np.array([
        1 / 3,
        (1 / 3) * 1.5 * 1.5,
        (1 / 3) * 1.5 * 2.0,
    ])
    assert np.allclose(weight, expected)
    assert weight.sum() != pytest.approx(1.0)


def test_logistic_prediction_parity() -> None:
    prepared, _, _ = prepare_v004a_samples(_synthetic_raw())
    x = prepared[FROZEN_FEATURE_COLUMNS].to_numpy(float)
    y = prepared[DEFAULT_TARGET_COLUMN].astype(int).to_numpy(float)
    weight = build_training_sample_weight(prepared, POSITIVE_WEIGHT)
    beta = fit_logistic_l2_weighted(x, y, l2=L2, sample_weight=weight)
    adapter_prediction = _sigmoid(beta[0] + x @ beta[1:])
    direct_prediction = 1.0 / (1.0 + np.exp(-np.clip(beta[0] + x @ beta[1:], -35, 35)))
    assert np.allclose(adapter_prediction, direct_prediction, rtol=0.0, atol=1e-15)


def test_frozen_model_controls() -> None:
    assert MODEL_FAMILY == "WEIGHTED_L2_LOGISTIC"
    assert L2 == pytest.approx(0.30)
    assert POSITIVE_WEIGHT == pytest.approx(1.50)
    assert INITIAL_TRAIN_DATES == 18
    assert HYPERPARAMETER_SEARCH is False
    assert FEATURE_SELECTION is False
    assert EXTRA_SCALING is False
    assert USES_PCA is False
    assert USES_PAIRWISE is False
    assert NEW_FEATURES is False


def test_feature_gate_and_provenance(benchmark) -> None:
    _, context = benchmark
    assert context["feature_gate"]["pass"] is True
    assert context["feature_gate"]["frozen_rows"] == 304
    assert context["feature_gate"]["canonical_reconstructed_rows"] == 15
    assert context["feature_gate"]["july_accessed"] is False
    assert len(context["raw"]) == 319
    assert set(context["provenance"]["feature"]) == set(RAW_INPUT_COLUMNS)
    assert context["provenance"]["exact_semantic_match"].eq("YES").all()
    assert context["provenance"]["D1_safe"].eq("YES").all()


def test_eighteen_date_warmup_and_june_21_folds(benchmark) -> None:
    _, context = benchmark
    folds = context["fold_audit"]
    assert len(folds) == 21
    assert folds.iloc[0]["train_dates"] == 18
    assert folds.iloc[0]["train_start"].startswith("2026-05")
    assert folds.iloc[0]["train_end"].startswith("2026-05")
    assert folds["test_date"].str.startswith("2026-06").all()
    assert folds["train_before_test"].all()
    assert folds.iloc[-1]["train_dates"] == 38


def test_oof_identity_score_and_rank_are_deterministic(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    assert len(oof) == 173
    assert oof["signal_date"].nunique() == 21
    assert oof["model_score"].between(0.0, 1.0).all()
    for _, day in oof.groupby("signal_date"):
        assert sorted(day["model_rank"].tolist()) == list(range(1, len(day) + 1))
        expected = day.sort_values(
            ["model_score", "event_id"], ascending=[False, True], kind="mergesort"
        )["event_id"].tolist()
        actual = day.sort_values("model_rank")["event_id"].tolist()
        assert actual == expected


def test_test_outcome_changes_do_not_change_x(benchmark) -> None:
    _, context = benchmark
    base = context["base"]
    outcomes = load_outcomes(ROOT, base)
    original = prepare_transfer_samples(context["raw"], outcomes)
    changed = outcomes.copy()
    changed["raw_repair_return"] = -changed["raw_repair_return"]
    changed["target7"] = 1 - changed["target7"]
    changed["capped_opportunity_return_7"] = np.minimum(
        changed["raw_repair_return"], 0.07
    )
    modified = prepare_transfer_samples(context["raw"], changed)
    left = original.sort_values("event_id")[FROZEN_FEATURE_COLUMNS].to_numpy(float)
    right = modified.sort_values("event_id")[FROZEN_FEATURE_COLUMNS].to_numpy(float)
    assert np.array_equal(left, right)


def test_pipeline_outputs_are_byte_deterministic(benchmark) -> None:
    first, _ = benchmark
    second, _ = build_outputs(ROOT)
    assert first.keys() == second.keys()
    assert all(first[name] == second[name] for name in first)


def test_regression_metrics_and_stop_status(benchmark) -> None:
    _, context = benchmark
    summary = context["summaries"]["V4A_ARCH_TRANSFER_V4C"]
    assert summary["Rank1_mean"] == pytest.approx(0.028393056016408322)
    assert summary["Top3_mean"] == pytest.approx(0.01853903678000977)
    assert summary["Rank1_beat_universe"] == pytest.approx(12 / 21)
    assert summary["Top3_beat_universe"] == pytest.approx(8 / 20)
    assert context["status"] == {
        "architecture_transfer_signal": "PARTIAL",
        "previous_route_misdesigned": "INCONCLUSIVE",
        "ready_for_july_oot": "NO",
    }

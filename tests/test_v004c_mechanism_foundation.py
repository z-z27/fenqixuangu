from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_mechanism_foundation import (
    BLOCK_COLUMNS,
    FORBIDDEN_X_COLUMNS,
    MAX_SIGNAL_DATE,
    MODEL_RAW_COLUMNS,
    STAGE_BLOCKS,
    add_repair_rank_target,
    aggregate_multiple_groups,
    assert_authoritative_universe,
    assert_no_july_signal_dates,
    compute_group_snapshot_features,
    compute_intraday_path,
    compute_same_tier_features,
    cross_sectional_rank_frame,
    date_equal_weights,
    fit_block_transform,
    mechanism_x_fingerprint,
    percentile_rank,
    permute_target_within_date,
)


ROOT = Path(__file__).resolve().parents[1]
DEV_CSV = (
    ROOT
    / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
    / "v004c_baostock_d1_dev_v002.csv"
)


def test_authoritative_universe_identity_contract():
    frame = pd.read_csv(DEV_CSV, dtype={"code": str})
    assert_authoritative_universe(frame)
    assert len(frame) == 319
    assert frame["signal_date"].nunique() == 39
    assert frame["signal_date"].astype(str).max() == MAX_SIGNAL_DATE


def test_repair_rank_target_preserves_below_and_above_7pct_order():
    frame = pd.DataFrame({
        "signal_date": ["2026-05-06"] * 4,
        "raw_repair_return": [-0.05, 0.02, 0.06, 0.10],
    })
    result = add_repair_rank_target(frame)
    assert result["repair_rank_target"].tolist() == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )
    assert result["repair_rank_target"].is_monotonic_increasing


def test_percentile_rank_uses_average_ties_and_small_sample_formula():
    ranked = percentile_rank(pd.Series([1.0, 2.0, 2.0, 4.0]))
    assert ranked.tolist() == pytest.approx([0.125, 0.5, 0.5, 0.875])
    frame = pd.DataFrame({
        "signal_date": ["d1", "d1", "d2"],
        "x": [3.0, 3.0, 7.0],
    })
    transformed = cross_sectional_rank_frame(frame, ["x"])
    assert transformed["x"].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_same_tier_synthetic_contract_uses_only_d0_and_d1():
    cohort = pd.DataFrame({
        "code": ["A", "B", "C"],
        "d0_close": [10.0, 10.0, 10.0],
        "d2_return": [99.0, -99.0, 42.0],
    })
    state = pd.DataFrame({
        "code": ["A", "B", "C"],
        "d1_close": [11.0, 10.2, 9.8],
        "is_advance": [True, False, False],
        "d3_return": [-88.0, 88.0, 0.0],
    })
    result = compute_same_tier_features("B", cohort, state)
    assert result["tier_cohort_size"] == 3
    assert result["tier_peer_count_ex_self"] == 2
    assert result["tier_peer_advance_count_d1"] == 1
    assert result["tier_peer_break_count_d1"] == 1
    assert result["tier_peer_advance_rate_d1"] == pytest.approx(0.5)
    assert result["tier_peer_break_rate_d1"] == pytest.approx(0.5)

    changed = cohort.copy()
    changed["d2_return"] *= -1000
    changed_state = state.copy()
    changed_state["d3_return"] *= -1000
    assert compute_same_tier_features("B", changed, changed_state) == result


def test_group_synthetic_breadth_and_deterministic_multiple_group_aggregation():
    group = pd.DataFrame({
        "is_limit_up": [True, True, False, False, False],
        "is_multi_board": [True, False, False, False, False],
        "d1_return": [0.10, 0.10, 0.01, -0.02, 0.00],
    })
    features = compute_group_snapshot_features(group)
    assert features["group_member_count"] == 5
    assert features["group_limit_up_count_d1"] == 2
    assert features["group_limit_up_breadth_d1"] == pytest.approx(0.4)
    groups = pd.DataFrame({
        "group_limit_up_breadth_d1": [0.4, 0.2],
        "group_member_mean_d1_return": [0.03, -0.01],
    })
    aggregate = aggregate_multiple_groups(groups, groups.columns)
    assert aggregate["max_group_limit_up_breadth_d1"] == pytest.approx(0.4)
    assert aggregate["mean_group_limit_up_breadth_d1"] == pytest.approx(0.3)


def test_intraday_path_uses_first_low_and_includes_low_bar_volume():
    lows = np.full(48, 10.0)
    lows[10] = 8.0
    lows[20] = 8.0
    minute = pd.DataFrame({
        "time": [f"bar{i:02d}" for i in range(48)],
        "low": lows,
        "volume": np.ones(48),
    })
    result = compute_intraday_path(minute)
    assert result["intraday_low_time_fraction"] == pytest.approx(10 / 47)
    assert result["post_low_volume_share"] == pytest.approx(38 / 48)


def test_intraday_path_requires_exactly_48_bars():
    with pytest.raises(RuntimeError, match="expected 48"):
        compute_intraday_path(pd.DataFrame({
            "time": list(range(47)), "low": np.ones(47), "volume": np.ones(47)
        }))


def test_july_sentinel_is_fatal():
    assert_no_july_signal_dates(pd.DataFrame({"signal_date": ["2026-06-30"]}))
    with pytest.raises(RuntimeError, match="2026-07-01"):
        assert_no_july_signal_dates(pd.DataFrame({"signal_date": ["2026-07-01"]}))


def test_no_lookahead_outcome_mutation_cannot_change_mechanism_x():
    row = {
        "event_id": "e1",
        "code": "000001",
        "signal_date": "2026-05-06",
        "board_streak_before_break": 2,
        **{column: 0.25 for column in MODEL_RAW_COLUMNS},
    }
    frame = pd.DataFrame([row])
    before = mechanism_x_fingerprint(frame)
    changed = frame.copy()
    for column in FORBIDDEN_X_COLUMNS:
        changed[column] = 999.0
    after = mechanism_x_fingerprint(changed)
    assert before == after


def test_pca_fit_is_train_only_and_has_deterministic_sign():
    columns = BLOCK_COLUMNS["EXISTING_TREND_POSITION"]
    rng = np.random.default_rng(27)
    train = pd.DataFrame(rng.normal(size=(20, len(columns))), columns=columns)
    first = fit_block_transform(train, columns)
    test = pd.DataFrame(rng.normal(size=(3, len(columns))), columns=columns)
    test.iloc[:, :] = 10_000
    second = fit_block_transform(train, columns)
    assert first.loading.tolist() == pytest.approx(second.loading.tolist())
    largest = int(np.argmax(np.abs(first.loading)))
    assert first.loading[largest] >= 0


def test_date_equal_weights_give_each_date_total_weight_one():
    dates = ["d1", "d1", "d1", "d2", "d2", "d3"]
    weights = date_equal_weights(dates)
    totals = pd.DataFrame({"date": dates, "weight": weights}).groupby("date")["weight"].sum()
    assert totals.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_predictor_degree_of_freedom_and_forbidden_contract():
    assert set(STAGE_BLOCKS) == {"M0", "M1"}
    assert len(STAGE_BLOCKS["M0"]) + 1 == 4
    assert len(STAGE_BLOCKS["M1"]) + 1 == 5
    assert len(STAGE_BLOCKS["M1"]) + 1 <= 8
    assert not FORBIDDEN_X_COLUMNS.intersection(MODEL_RAW_COLUMNS)


def test_single_permutation_is_deterministic_and_within_date():
    frame = pd.DataFrame({
        "signal_date": ["d1"] * 4 + ["d2"] * 3,
        "repair_rank_target": [0.1, 0.3, 0.6, 0.9, 0.2, 0.5, 0.8],
    })
    first = permute_target_within_date(frame)
    second = permute_target_within_date(frame)
    assert first["permuted_repair_rank_target"].tolist() == second[
        "permuted_repair_rank_target"
    ].tolist()
    for date, group in first.groupby("signal_date"):
        expected = sorted(frame.loc[frame["signal_date"].eq(date), "repair_rank_target"])
        actual = sorted(group["permuted_repair_rank_target"])
        assert actual == expected

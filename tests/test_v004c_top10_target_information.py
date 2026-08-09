from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004a import DEFAULT_TARGET_COLUMN
from src.v004c_v4a_architecture_transfer import FROZEN_FEATURE_COLUMNS
from src.v004c_top10_target_information import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FEATURE_SELECTION,
    HYPERPARAMETER_SEARCH,
    MODEL_IDS,
    NEW_FEATURES,
    OUTPUT_FILENAMES,
    PAIR_FEATURE_COLUMNS,
    PAIR_L2,
    POSITIVE_WEIGHTING,
    TARGET_VARIANTS,
    TAIL_WEIGHTING,
    aggregate_pair_information,
    assert_experiment_contract,
    build_outputs,
    build_same_date_pairs,
    encode_target_information,
    fit_pairwise_weighted_ridge,
)
from src.v004c_v4a_top10_reranker import STAGE2_FEATURE_COLUMNS, score_stage1_date


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark():
    outputs, context = build_outputs(ROOT)
    return outputs, context


def _pair_fixture(raw_values: list[float]) -> pd.DataFrame:
    rows = []
    for index, raw in enumerate(raw_values):
        rows.append({
            "signal_date": "2026-05-01",
            "event_id": f"e{index}",
            "target7": int(raw >= 0.07),
            "raw_repair_return": raw,
            "stage1_strength": 1.0 - index * 0.1,
            "closing_completion_gap": 0.2 + index * 0.1,
            "strength_x_gap": (1.0 - index * 0.1) * (0.2 + index * 0.1),
        })
    return pd.DataFrame(rows)


def _synthetic_stage1_samples() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]
    for date_index, date in enumerate(dates):
        for row_index in range(4):
            row: dict[str, object] = {
                "event_id": f"{date}_{row_index}",
                "signal_date": date,
                "code": f"{date_index * 10 + row_index:06d}",
                "target7": int((date_index + row_index) % 2 == 0),
                "tail_weight": 1.0,
            }
            row[DEFAULT_TARGET_COLUMN] = bool(row["target7"])
            for feature_index, feature in enumerate(FROZEN_FEATURE_COLUMNS):
                row[feature] = 0.05 + 0.01 * feature_index + 0.03 * row_index
            rows.append(row)
    return pd.DataFrame(rows)


def test_authoritative_identity_and_no_july(benchmark) -> None:
    _, context = benchmark
    base = context["base"]
    assert len(base) == 319
    assert base["signal_date"].nunique() == 39
    assert int(base["signal_date"].str.startswith("2026-05").sum()) == 146
    assert int(base["signal_date"].str.startswith("2026-06").sum()) == 173
    assert int((base["board_streak_before_break"] == 2).sum()) == 261
    assert int((base["board_streak_before_break"] == 3).sum()) == 58
    assert base["signal_date"].max() == "2026-06-30"
    assert context["oof"]["signal_date"].max() == "2026-06-30"


def test_frozen_stage1_parity_and_june_control(benchmark) -> None:
    _, context = benchmark
    assert context["parity"] == {
        "identity_exact": True,
        "score_exact_at_frozen_precision": True,
        "rank_exact": True,
        "pass": True,
    }
    control = context["summaries"]["CONTROL_V4A"]
    assert control["Rank1_mean"] == pytest.approx(0.028393, abs=5e-7)
    assert control["Rank2_mean"] == pytest.approx(0.032031, abs=5e-7)
    assert control["Rank3_mean"] == pytest.approx(-0.003025, abs=5e-7)
    assert control["Top2_mean"] == pytest.approx(0.030212, abs=5e-7)
    assert control["Top3_mean"] == pytest.approx(0.018539, abs=5e-7)


def test_target_encodings_exact_examples() -> None:
    result = encode_target_information(np.asarray([0.20, 0.06, -0.12]))
    assert result["utility_binary7"].tolist() == [1.0, 0.0, 0.0]
    assert result["utility_capped7"].tolist() == [0.07, 0.06, -0.12]
    assert result["utility_raw"].tolist() == [0.20, 0.06, -0.12]
    assert result.loc[2, "utility_capped7"] < 0.0


def test_pair_target_sub7_severity_example() -> None:
    pairs = build_same_date_pairs(_pair_fixture([0.06, -0.12]))
    assert len(pairs) == 1
    assert pairs.iloc[0]["pair_y_binary7"] == pytest.approx(0.0)
    assert pairs.iloc[0]["pair_y_capped7"] == pytest.approx(0.18)
    assert pairs.iloc[0]["pair_y_raw"] == pytest.approx(0.18)


def test_pair_target_above7_information_example() -> None:
    pairs = build_same_date_pairs(_pair_fixture([0.20, 0.08]))
    assert len(pairs) == 1
    assert pairs.iloc[0]["pair_y_binary7"] == pytest.approx(0.0)
    assert pairs.iloc[0]["pair_y_capped7"] == pytest.approx(0.0)
    assert pairs.iloc[0]["pair_y_raw"] == pytest.approx(0.12)


def test_pair_orientation_keys_x_and_weights_are_single_and_deterministic() -> None:
    fixture = _pair_fixture([0.08, -0.02, 0.06]).iloc[[2, 0, 1]].copy()
    pairs = build_same_date_pairs(fixture)
    assert pairs[["left_event_id", "right_event_id"]].values.tolist() == [
        ["e0", "e1"], ["e0", "e2"], ["e1", "e2"]
    ]
    assert pairs["pair_key"].is_unique
    assert pairs["pair_weight"].tolist() == pytest.approx([1 / 3] * 3)
    assert pairs["pair_weight"].sum() == pytest.approx(1.0)
    expected = fixture.set_index("event_id")
    first = pairs.iloc[0]
    for feature in PAIR_FEATURE_COLUMNS:
        assert first[f"x_{feature}"] == pytest.approx(
            expected.loc["e0", feature] - expected.loc["e1", feature]
        )


def test_weighted_ridge_has_no_intercept_and_fixed_l2() -> None:
    x = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.asarray([1.0, -1.0])
    w = np.asarray([0.5, 0.5])
    beta = fit_pairwise_weighted_ridge(x, y, w)
    expected = np.linalg.solve(x.T @ (w[:, None] * x) + 0.3 * np.eye(3), x.T @ (w * y))
    assert np.allclose(beta, expected)
    assert len(beta) == 3
    assert PAIR_L2 == pytest.approx(0.30)


def test_nested_crossfit_has_identical_pair_contract_and_no_leakage(benchmark) -> None:
    _, context = benchmark
    audit = context["fold_audit"]
    assert len(audit) == 21
    assert audit["test_date"].str.startswith("2026-06").all()
    assert audit.iloc[0]["history_dates"] == 18
    assert audit.iloc[-1]["history_dates"] == 38
    assert audit["historical_top10_dates"].equals(audit["history_dates"])
    assert audit["binary_pair_count"].equals(audit["pair_count"])
    assert audit["capped_pair_count"].equals(audit["pair_count"])
    assert audit["raw_pair_count"].equals(audit["pair_count"])
    assert audit["pair_keys_identical"].all()
    assert audit["pair_x_identical"].all()
    assert audit["pair_weights_identical"].all()
    assert audit["pair_weight_date_sum_min"].eq(1.0).all()
    assert audit["pair_weight_date_sum_max"].eq(1.0).all()
    assert audit["self_label_leakage_rows"].eq(0).all()
    assert audit["current_test_leakage_rows"].eq(0).all()
    assert context["self_label_leakage_rows"] == 0
    assert context["current_test_date_leakage_rows"] == 0
    assert context["gates_valid"] is True


def test_meta_date_label_flip_cannot_change_own_stage1_retrieval() -> None:
    samples = _synthetic_stage1_samples()
    meta_date = "2026-05-02"
    base_dates = ["2026-05-01", "2026-05-03", "2026-05-04"]
    original, _ = score_stage1_date(samples, base_dates, meta_date, "2026-06-01")
    changed = samples.copy()
    mask = changed["signal_date"].eq(meta_date)
    changed.loc[mask, "target7"] = 1 - changed.loc[mask, "target7"]
    changed.loc[mask, DEFAULT_TARGET_COLUMN] = changed.loc[mask, "target7"].astype(bool)
    flipped, _ = score_stage1_date(changed, base_dates, meta_date, "2026-06-01")
    columns = ["stage1_score", "stage1_rank", "stage1_top10", "stage1_strength"]
    assert np.array_equal(original[columns].to_numpy(), flipped[columns].to_numpy())


def test_exact_three_predictors_same_learner_and_no_search() -> None:
    assert_experiment_contract()
    assert PAIR_FEATURE_COLUMNS == [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]
    assert PAIR_FEATURE_COLUMNS == STAGE2_FEATURE_COLUMNS
    assert TARGET_VARIANTS == ("BINARY7", "CAPPED7", "RAW_RETURN")
    assert MODEL_IDS == ("PAIR_BINARY7", "PAIR_CAPPED7", "PAIR_RAW")
    assert HYPERPARAMETER_SEARCH is False
    assert FEATURE_SELECTION is False
    assert NEW_FEATURES is False
    assert POSITIVE_WEIGHTING is False
    assert TAIL_WEIGHTING is False


def test_top10_restriction_and_tie_break(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    for score_column, rank_column in (
        ("binary_pair_score", "binary_pair_rank"),
        ("capped_pair_score", "capped_pair_rank"),
        ("raw_pair_score", "raw_pair_rank"),
    ):
        outside = oof[~oof["stage1_top10"]]
        assert outside[score_column].isna().all()
        assert outside[rank_column].isna().all()
        for _, day in oof[oof["stage1_top10"]].groupby("signal_date", sort=True):
            expected = day.sort_values(
                [score_column, "stage1_rank", "event_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )["event_id"].tolist()
            actual = day.sort_values(rank_column)["event_id"].tolist()
            assert actual == expected


def test_pair_information_counts_are_coherent(benchmark) -> None:
    _, context = benchmark
    info = context["pair_information"]
    assert info["training_pair_total"] == (
        info["binary_cross_threshold_pairs"] + info["binary_same_class_pairs"]
    )
    assert info["binary_same_class_pairs"] == (
        info["within_non_target_pairs"] + info["within_target7_pairs"]
    )
    assert info == aggregate_pair_information(context["pair_audit"])
    assert info["binary_tied_separated_by_raw"] >= info["binary_tied_separated_by_capped"]


def test_output_contract_and_determinism(benchmark) -> None:
    first, _ = benchmark
    second, _ = build_outputs(ROOT)
    assert tuple(first) == OUTPUT_FILENAMES
    assert len(first) == 6
    assert all(first[name] == second[name] for name in first)
    assert BOOTSTRAP_SEED == 20260810
    assert BOOTSTRAP_RESAMPLES == 20_000

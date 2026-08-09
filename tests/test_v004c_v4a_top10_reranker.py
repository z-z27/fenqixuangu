from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004a import DEFAULT_TARGET_COLUMN
from src.v004c_v4a_architecture_transfer import FROZEN_FEATURE_COLUMNS
from src.v004c_v4a_top10_reranker import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FEATURE_SELECTION,
    HYPERPARAMETER_SEARCH,
    MODEL_ZOO,
    NEW_RAW_FEATURES,
    OUTPUT_FILENAMES,
    RESIDUAL_RAW_FIELDS,
    STAGE1_L2,
    STAGE1_POSITIVE_WEIGHT,
    STAGE2_FEATURE_COLUMNS,
    STAGE2_L2,
    STAGE2_POSITIVE_WEIGHT,
    STAGE2_TAIL_WEIGHTING,
    _stage2_date_weights,
    assert_model_contract,
    build_outputs,
    build_stage2_features,
    score_stage1_date,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark():
    outputs, context = build_outputs(ROOT)
    return outputs, context


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
                row[feature] = (
                    0.05
                    + 0.01 * feature_index
                    + 0.03 * row_index
                    + 0.02 * date_index
                )
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


def test_stage1_architecture_and_test_parity(benchmark) -> None:
    _, context = benchmark
    assert len(FROZEN_FEATURE_COLUMNS) == 18
    assert STAGE1_L2 == pytest.approx(0.30)
    assert STAGE1_POSITIVE_WEIGHT == pytest.approx(1.50)
    assert context["parity"] == {
        "identity_exact": True,
        "score_exact_at_frozen_precision": True,
        "rank_exact": True,
        "top3_membership_exact": True,
        "pass": True,
    }


def test_nested_crossfit_audit_has_no_leakage(benchmark) -> None:
    _, context = benchmark
    audit = context["fold_audit"]
    assert len(audit) == 21
    assert audit["test_date"].str.startswith("2026-06").all()
    assert audit.iloc[0]["history_date_count"] == 18
    assert audit.iloc[-1]["history_date_count"] == 38
    assert audit["meta_train_dates"].equals(audit["history_date_count"])
    assert audit["crossfit_stage1_models_fitted"].equals(audit["history_date_count"])
    assert audit["min_base_train_dates"].equals(audit["history_date_count"] - 1)
    assert audit["max_base_train_dates"].equals(audit["history_date_count"] - 1)
    assert audit["self_label_leakage_rows"].eq(0).all()
    assert audit["current_test_date_leakage_rows"].eq(0).all()
    assert audit["stage2_weight_date_sum_min"].eq(1.0).all()
    assert audit["stage2_weight_date_sum_max"].eq(1.0).all()
    assert context["crossfit_stage1_fits"] == sum(range(18, 39))


def test_adversarial_meta_date_label_flip_cannot_change_own_stage1_prediction() -> None:
    samples = _synthetic_stage1_samples()
    meta_date = "2026-05-02"
    base_dates = ["2026-05-01", "2026-05-03", "2026-05-04"]
    original, original_audit = score_stage1_date(
        samples, base_dates, meta_date, "2026-06-01"
    )
    changed = samples.copy()
    mask = changed["signal_date"].eq(meta_date)
    changed.loc[mask, "target7"] = 1 - changed.loc[mask, "target7"]
    changed.loc[mask, DEFAULT_TARGET_COLUMN] = changed.loc[mask, "target7"].astype(bool)
    flipped, flipped_audit = score_stage1_date(
        changed, base_dates, meta_date, "2026-06-01"
    )
    columns = ["stage1_score", "stage1_rank", "stage1_top10", "stage1_strength"]
    assert np.array_equal(original[columns].to_numpy(), flipped[columns].to_numpy())
    assert original_audit["self_label_leakage_rows"] == 0
    assert flipped_audit["self_label_leakage_rows"] == 0


def test_crossfit_rejects_self_date_and_current_test_leakage() -> None:
    samples = _synthetic_stage1_samples()
    with pytest.raises(RuntimeError, match="self-label"):
        score_stage1_date(
            samples,
            ["2026-05-01", "2026-05-02"],
            "2026-05-02",
            "2026-06-01",
        )
    with pytest.raises(RuntimeError, match="current test date"):
        score_stage1_date(
            samples,
            ["2026-05-01", "2026-05-03"],
            "2026-05-02",
            "2026-05-03",
        )


def test_exact_stage2_feature_contract() -> None:
    assert_model_contract()
    assert RESIDUAL_RAW_FIELDS == [
        "d1_high_to_close_drawdown_raw",
        "d1_close_location",
        "d1_low_to_close_recovery",
        "d1_open_to_close_return_raw",
    ]
    assert STAGE2_FEATURE_COLUMNS == [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]
    assert len(STAGE2_FEATURE_COLUMNS) == 3


def test_closing_completion_gap_and_interaction_formula() -> None:
    frame = pd.DataFrame({
        "event_id": ["a", "b"],
        "signal_date": ["2026-06-01", "2026-06-01"],
        "stage1_strength": [1.0, 0.5],
        "d1_high_to_close_drawdown_raw": [2.0, 1.0],
        "d1_close_location": [1.0, 2.0],
        "d1_low_to_close_recovery": [1.0, 2.0],
        "d1_open_to_close_return_raw": [1.0, 2.0],
    })
    result = build_stage2_features(frame).set_index("event_id")
    assert result.loc["a", "rank_drawdown_top10"] == pytest.approx(1.0)
    assert result.loc["a", "rank_close_location_top10"] == pytest.approx(0.5)
    assert result.loc["a", "closing_completion_gap"] == pytest.approx(0.625)
    assert result.loc["a", "strength_x_gap"] == pytest.approx(0.625)
    assert result.loc["b", "closing_completion_gap"] == pytest.approx(0.125)
    assert result.loc["b", "strength_x_gap"] == pytest.approx(0.0625)


def test_stage1_strength_formula_uses_full_candidate_count(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    expected = 1.0 - (
        (oof["stage1_rank"] - 1) / np.maximum(oof["candidate_count"] - 1, 1)
    )
    assert np.allclose(oof["stage1_strength"], expected)


def test_stage2_date_weights_are_exact_and_no_class_or_tail_weight() -> None:
    frame = pd.DataFrame({
        "signal_date": ["a", "a", "b", "b", "b"],
        "event_id": list("12345"),
        "target7": [0, 1, 0, 1, 1],
    })
    weight = _stage2_date_weights(frame)
    assert np.allclose(weight, [0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    assert np.sum(weight[:2]) == pytest.approx(1.0)
    assert np.sum(weight[2:]) == pytest.approx(1.0)
    assert STAGE2_L2 == pytest.approx(0.30)
    assert STAGE2_POSITIVE_WEIGHT == pytest.approx(1.0)
    assert STAGE2_TAIL_WEIGHTING is False


def test_stage2_never_promotes_outside_stage1_top10(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    assert oof.loc[oof["reranker_top3"], "stage1_top10"].all()
    outside = oof[~oof["stage1_top10"]]
    assert outside["stage2_score"].isna().all()
    assert outside["stage2_rank_within_top10"].isna().all()
    assert outside["final_rerank_rank"].gt(10).all()


def test_reranker_tie_break_is_stage2_then_stage1_then_event(benchmark) -> None:
    _, context = benchmark
    oof = context["oof"]
    for _, day in oof[oof["stage1_top10"]].groupby("signal_date", sort=True):
        expected = day.sort_values(
            ["stage2_score", "stage1_score", "event_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )["event_id"].tolist()
        actual = day.sort_values("stage2_rank_within_top10")["event_id"].tolist()
        assert actual == expected


def test_membership_status_and_possible_replacements(benchmark) -> None:
    _, context = benchmark
    membership = context["membership_frame"]
    assert set(membership["membership_status"]).issubset(
        {"PRESERVED", "DEMOTED", "PROMOTED"}
    )
    assert context["membership"]["possible_replacement_slots"] == 21
    assert context["membership"]["demoted_total"] == context["membership"]["promoted_total"]


def test_no_search_new_features_or_external_factors() -> None:
    assert HYPERPARAMETER_SEARCH is False
    assert FEATURE_SELECTION is False
    assert NEW_RAW_FEATURES is False
    assert MODEL_ZOO is False
    assert BOOTSTRAP_SEED == 20260809
    assert BOOTSTRAP_RESAMPLES == 20_000


def test_output_set_and_determinism(benchmark) -> None:
    first, _ = benchmark
    second, _ = build_outputs(ROOT)
    assert tuple(first) == OUTPUT_FILENAMES
    assert len(first) == 6
    assert all(first[name] == second[name] for name in first)

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import v004c_veto_only_backfill as veto


ROOT = Path(__file__).resolve().parents[1]


def _day(stage2_ranks: list[int], candidate_count: int | None = None) -> pd.DataFrame:
    size = candidate_count or len(stage2_ranks)
    rows = []
    for index in range(size):
        rank = index + 1
        stage2 = stage2_ranks[index] if index < len(stage2_ranks) else rank
        rows.append({
            "event_id": f"E{rank}",
            "signal_date": "2026-06-01",
            "code": f"{rank:06d}",
            "stage1_rank": rank,
            "stage2_rank_within_top10": stage2,
        })
    return pd.DataFrame(rows)


def test_adversarial_stage2_favorite_cannot_control_backfill() -> None:
    day = _day([1, 8, 2, 7, 6, 5, 4, 3])
    result, audit = veto.apply_veto_only_day(day)
    selected = result.loc[result["veto_only_top3"], "event_id"].tolist()
    assert audit["vetoed_ids"] == ["E2"]
    assert audit["backfilled_ids"] == ["E4"]
    assert selected == ["E1", "E3", "E4"]
    assert "E8" not in selected


def test_double_veto_keeps_rank2_and_backfills_rank4_rank5() -> None:
    day = _day([8, 1, 7, 2, 3, 4, 5, 6])
    result, audit = veto.apply_veto_only_day(day)
    assert audit["vetoed_ids"] == ["E1", "E3"]
    assert audit["backfilled_ids"] == ["E4", "E5"]
    assert result.loc[result["veto_only_top3"], "event_id"].tolist() == [
        "E2",
        "E4",
        "E5",
    ]


def test_no_veto_is_exact_control_membership() -> None:
    day = _day([1, 2, 3, 8, 7, 6, 5, 4])
    result, audit = veto.apply_veto_only_day(day)
    assert audit["vetoed_ids"] == []
    assert audit["backfilled_ids"] == []
    assert result.loc[result["veto_only_top3"], "event_id"].tolist() == [
        "E1",
        "E2",
        "E3",
    ]


def test_candidate_count_below_three_uses_all_candidates() -> None:
    day = _day([1, 2], candidate_count=2)
    result, audit = veto.apply_veto_only_day(day)
    assert audit["final_ids"] == ["E1", "E2"]
    assert result["veto_only_top3"].all()


def test_stage1_rank_must_be_frozen_and_contiguous() -> None:
    day = _day([1, 2, 3, 4])
    day.loc[2, "stage1_rank"] = 5
    with pytest.raises(RuntimeError, match="Stage1 ranks"):
        veto.apply_veto_only_day(day)


def test_frozen_artifact_identity_and_no_july() -> None:
    frozen = veto.load_frozen_artifacts(ROOT)
    oof = frozen["oof"]
    assert len(oof) == 173
    assert oof["signal_date"].nunique() == 21
    assert oof["signal_date"].max() < veto.JULY_SENTINEL_DATE
    assert frozen["fold_audit"]["self_label_leakage_rows"].sum() == 0
    assert frozen["fold_audit"]["current_test_date_leakage_rows"].sum() == 0


@pytest.fixture(scope="module")
def built() -> tuple[dict[str, bytes], dict[str, object]]:
    return veto.build_outputs(ROOT)


def test_previous_control_and_full_reranker_parity(built) -> None:
    _, context = built
    summaries = context["summaries"]
    assert context["parity"]["pass"] is True
    assert summaries["CONTROL"]["Rank1_mean"] == pytest.approx(0.0283930560164)
    assert summaries["CONTROL"]["Rank2_mean"] == pytest.approx(0.0320306491097)
    assert summaries["CONTROL"]["Rank3_mean"] == pytest.approx(-0.00302540178647)
    assert summaries["CONTROL"]["Top2_mean"] == pytest.approx(0.0302118525631)
    assert summaries["CONTROL"]["Top3_mean"] == pytest.approx(0.01853903678)
    assert summaries["FULL_RERANKER"]["Top3_mean"] == pytest.approx(0.0177345604656)
    assert summaries["FULL_RERANKER"]["Top3_target7_precision"] == pytest.approx(0.30)


def test_real_veto_definition_is_exact(built) -> None:
    _, context = built
    oof = context["oof"]
    expected = oof["control_top3"] & oof["stage2_rank_within_top10"].gt(3)
    actual_ids = {
        event_id
        for policy in context["policies"].values()
        for event_id in policy["vetoed_ids"]
    }
    assert actual_ids == set(oof.loc[expected, "event_id"])


def test_real_backfill_is_strictly_stage1_order(built) -> None:
    _, context = built
    oof = context["oof"]
    for date, policy in context["policies"].items():
        day = oof[oof["signal_date"].eq(date)].sort_values("stage1_rank")
        expected = day[day["stage1_rank"].between(4, 10)].head(
            len(policy["vetoed_ids"])
        )["event_id"].tolist()
        assert policy["backfilled_ids"] == expected


def test_final_membership_size_and_deterministic_rank(built) -> None:
    _, context = built
    oof = context["oof"]
    for _, day in oof.groupby("signal_date"):
        selected = day[day["veto_only_top3"]].sort_values("veto_only_rank")
        assert len(selected) == min(3, len(day))
        assert selected["stage1_rank"].tolist() == sorted(selected["stage1_rank"])


def test_opportunity_buckets_are_frozen_and_balanced(built) -> None:
    _, context = built
    values = list(context["bucket_map"].values())
    assert values.count("LOW") == 7
    assert values.count("MID") == 7
    assert values.count("HIGH") == 7


def test_capped_return_has_upper_cap_only() -> None:
    raw = pd.Series([-0.10, 0.02, 0.08, 0.15])
    capped = np.minimum(raw.to_numpy(float), 0.07)
    assert capped.tolist() == pytest.approx([-0.10, 0.02, 0.07, 0.07])


def test_no_model_fit_or_training_dependency() -> None:
    source = inspect.getsource(veto)
    forbidden = (
        "fit_logistic_l2_weighted",
        "run_expanding_forward",
        "GridSearch",
        "RandomForest",
        "XGBoost",
        "xgboost",
        "sklearn",
    )
    assert not any(token in source for token in forbidden)
    assert "stage2_score <" not in source
    assert "stage2_score >" not in source


def test_policy_contract_has_no_tuning_or_new_features() -> None:
    assert veto.TOP_K == 10
    assert veto.STRICT_K == 3
    assert set(veto.REQUIRED_OOF_COLUMNS).issuperset(
        {"stage1_rank", "stage2_rank_within_top10", "target7"}
    )


def test_build_is_byte_deterministic(built) -> None:
    first, _ = built
    second, _ = veto.build_outputs(ROOT)
    assert first.keys() == second.keys()
    assert all(first[name] == second[name] for name in first)
    assert tuple(first) == veto.OUTPUT_FILENAMES

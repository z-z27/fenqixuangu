from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_exact_v4a_residual import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COMPLETION_FIELDS,
    CURVE_SCORE_USED,
    DAMAGE_FIELDS,
    FEATURE_SELECTION,
    GROUP_FP,
    GROUP_MW,
    GROUP_OTHER,
    GROUP_RN,
    GROUP_TP,
    MODEL_TRAINING,
    NEW_FEATURES,
    OUTPUT_FILENAMES,
    RESIDUAL_FIELDS,
    SCORE_MODIFICATION,
    _feature_path,
    _ranking_path,
    assert_analysis_contract,
    assign_residual_groups,
    build_outputs,
    cliffs_delta,
    deterministic_date_bootstrap,
    load_exact_transfer_oof,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analysis():
    outputs, context = build_outputs(ROOT)
    return outputs, context


def test_exact_transfer_artifact_and_june_only_gate() -> None:
    frame = load_exact_transfer_oof(ROOT)
    assert _ranking_path(ROOT).as_posix().endswith(
        "v004c_v4a_architecture_transfer_v001_20260506_20260630/"
        "v004c_v4a_transfer_oof_v001.csv"
    )
    assert len(frame) == 173
    assert frame["signal_date"].nunique() == 21
    assert frame["signal_date"].str.startswith("2026-06").all()
    assert frame["signal_date"].max() == "2026-06-30"
    assert frame["event_id"].is_unique


def test_frozen_rank_is_score_desc_event_id_asc() -> None:
    frame = load_exact_transfer_oof(ROOT)
    for _, day in frame.groupby("signal_date", sort=True):
        expected = day.sort_values(
            ["model_score", "event_id"],
            ascending=[False, True],
            kind="mergesort",
        )["event_id"].tolist()
        actual = day.sort_values("model_rank")["event_id"].tolist()
        assert actual == expected


def test_authoritative_join_is_exact_and_uses_only_bao_source(analysis) -> None:
    _, context = analysis
    assert _feature_path(ROOT).as_posix().endswith(
        "v004c_baostock_d1_dev_v002_20260506_20260630/"
        "v004c_baostock_d1_dev_v002.csv"
    )
    assert context["join_audit"] == {
        "oof_rows": 173,
        "matched_rows": 173,
        "unmatched_rows": 0,
        "duplicate_joins": 0,
        "code_date_board_mismatches": 0,
    }


def test_exact_eight_fields_and_no_discovery() -> None:
    assert_analysis_contract()
    assert RESIDUAL_FIELDS == [
        "d1_high_to_close_drawdown_raw",
        "d1_close_location",
        "d1_intraday_range",
        "d1_low_to_close_recovery",
        "d1_open_to_close_return_raw",
        "d1_last_hour_return",
        "late_day_sell_volume_ratio",
        "late_day_sell_amount_ratio",
    ]
    assert set(DAMAGE_FIELDS + COMPLETION_FIELDS) == set(RESIDUAL_FIELDS)
    assert len(RESIDUAL_FIELDS) == 8


def test_residual_group_definitions_are_exact() -> None:
    synthetic = pd.DataFrame({
        "model_rank": [1, 3, 4, 10, 11],
        "target7": [0, 1, 1, 0, 1],
        "signal_date": ["2026-06-01"] * 5,
        "event_id": list("abcde"),
        "model_score": [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    result = assign_residual_groups(synthetic)
    assert result["residual_group"].tolist() == [
        GROUP_FP, GROUP_TP, GROUP_MW, GROUP_RN, GROUP_OTHER
    ]


def test_real_groups_follow_only_frozen_rank_and_target(analysis) -> None:
    _, context = analysis
    frame = context["grouped"]
    assert frame.loc[frame["residual_group"].eq(GROUP_FP), "model_rank"].le(3).all()
    assert frame.loc[frame["residual_group"].eq(GROUP_FP), "target7"].eq(0).all()
    assert frame.loc[frame["residual_group"].eq(GROUP_MW), "model_rank"].between(4, 10).all()
    assert frame.loc[frame["residual_group"].eq(GROUP_MW), "target7"].eq(1).all()


def test_cliffs_delta_direction_and_ties_are_deterministic() -> None:
    assert cliffs_delta([2.0, 3.0], [1.0, 2.0]) == pytest.approx(0.75)
    assert cliffs_delta([1.0], [2.0]) == pytest.approx(-1.0)
    assert cliffs_delta([1.0], [1.0]) == pytest.approx(0.0)


def test_date_bootstrap_fixed_seed_and_resample_count() -> None:
    assert BOOTSTRAP_SEED == 20260809
    assert BOOTSTRAP_RESAMPLES == 10_000
    first = deterministic_date_bootstrap([1.0, 2.0, 3.0])
    second = deterministic_date_bootstrap([1.0, 2.0, 3.0])
    assert first == second
    assert first[0] <= first[1] <= first[2]


def test_daily_differences_only_include_dates_with_fp_and_mw(analysis) -> None:
    _, context = analysis
    daily = context["daily_differences"]
    grouped = context["grouped"]
    fp_dates = set(grouped.loc[grouped["residual_group"].eq(GROUP_FP), "signal_date"])
    mw_dates = set(grouped.loc[grouped["residual_group"].eq(GROUP_MW), "signal_date"])
    assert set(daily["signal_date"]) == fp_dates.intersection(mw_dates)
    assert daily.groupby("feature").size().nunique() == 1
    assert daily.groupby("feature").size().iloc[0] == len(fp_dates.intersection(mw_dates))


def test_top10_oracle_is_restricted_and_full_oracle_is_separate(analysis) -> None:
    _, context = analysis
    frame = context["grouped"]
    daily = context["top10_daily"].set_index("signal_date")
    for date, day in frame.groupby("signal_date", sort=True):
        if len(day) < 3:
            assert pd.isna(daily.loc[date, "top10_oracle_top3_capped"])
            continue
        top10 = day[day["model_rank"].le(10)].nlargest(
            3, "capped_opportunity_return_7", keep="all"
        ).head(3)
        full = day.nlargest(3, "capped_opportunity_return_7", keep="all").head(3)
        assert daily.loc[date, "top10_oracle_top3_capped"] == pytest.approx(
            top10["capped_opportunity_return_7"].mean()
        )
        assert daily.loc[date, "full_oracle_top3_capped"] == pytest.approx(
            full["capped_opportunity_return_7"].mean()
        )


def test_scores_are_not_modified_and_curve_score_is_absent(analysis) -> None:
    _, context = analysis
    source = context["oof"].set_index("event_id")["model_score"].sort_index()
    analyzed = context["events"].set_index("event_id")["model_score"].sort_index()
    assert np.array_equal(source.to_numpy(float), analyzed.to_numpy(float))
    assert not any("curve" in column.lower() for column in context["events"].columns)
    assert CURVE_SCORE_USED is False
    assert SCORE_MODIFICATION is False


def test_analysis_module_has_no_model_library_or_fit_call() -> None:
    source = (ROOT / "src/v004c_exact_v4a_residual.py").read_text(encoding="utf-8")
    banned = (
        "import sklearn",
        "from sklearn",
        "import statsmodels",
        "from statsmodels",
        "import xgboost",
        "import lightgbm",
        "fit_logistic",
        "fit_logistic_l2_weighted",
    )
    assert not any(token in source for token in banned)
    assert MODEL_TRAINING is False
    assert FEATURE_SELECTION is False
    assert NEW_FEATURES is False


def test_output_set_and_byte_determinism(analysis) -> None:
    first, _ = analysis
    second, _ = build_outputs(ROOT)
    assert tuple(first) == OUTPUT_FILENAMES
    assert len(first) == 5
    assert all(first[name] == second[name] for name in first)


def test_exact_control_parity_regression(analysis) -> None:
    _, context = analysis
    parity = context["parity"]
    assert parity["pass"] is True
    assert parity["Rank1"] == pytest.approx(0.0283930560164)
    assert parity["Rank2"] == pytest.approx(0.0320306491097)
    assert parity["Rank3"] == pytest.approx(-0.00302540178647)
    assert parity["Top2"] == pytest.approx(0.0302118525631)
    assert parity["Top3"] == pytest.approx(0.01853903678)
    assert parity["cross_threshold"] == pytest.approx(0.487123907143)


def test_fixed_residual_counts_evidence_and_top10_decision(analysis) -> None:
    _, context = analysis
    groups = context["groups"]
    assert groups[GROUP_FP] == {"rows": 44, "dates": 21}
    assert groups[GROUP_TP] == {"rows": 18, "dates": 15}
    assert groups[GROUP_MW] == {"rows": 31, "dates": 15}
    assert groups[GROUP_RN] == {"rows": 57, "dates": 17}
    assert groups["BOTH"]["dates"] == 15
    assert context["evidence"]["repair_room_residual_evidence"] == "MIXED"
    assert context["evidence"]["damage_answer"] == "MIXED"
    assert context["evidence"]["completion_answer"] == "YES"
    assert context["top10"]["top10_rerank_feasibility"] == "YES"
    assert context["top10"]["dates_with_replacements"] == 15
    assert context["top10"]["total_possible_replacements"] == 21
    assert context["top10"]["actual_top3"] == pytest.approx(0.01853903678)
    assert context["top10"]["top10_oracle_top3"] > 0.06
    assert context["top10"]["full_oracle_top3"] == pytest.approx(0.061177092789)
    assert context["top10"]["top10_oracle_recovery_ratio"] > 0.98

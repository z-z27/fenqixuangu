from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_board2_board3_structural_audit import (
    BOARD_GROUPS,
    EXPECTED_MATURED_DATES,
    EXPECTED_MATURED_ROWS,
    EXPECTED_RAW_BOARD2,
    EXPECTED_RAW_BOARD3,
    EXPECTED_RAW_DATES,
    EXPECTED_RAW_ROWS,
    JULY_RESULT_ROWS_ACCESSED,
    POPULATION_A_BOOTSTRAP_RESAMPLES,
    POPULATION_A_PERMUTATIONS,
    STRICT_BOOTSTRAP_RESAMPLES,
    _bootstrap_draws,
    board_group,
    build_outputs,
    capped_return_7,
    composition_lift,
    loss_from_raw,
    population_a_same_date,
    selection_rate,
    stage1_rank_percentile,
    stratified_board_permutation,
    target7_from_raw,
)
from src.v004c_limited_risk_protector import EXPECTED_STRICT_DATES


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit_result():
    return build_outputs(ROOT)


def test_contract_constants_and_no_july_access():
    assert (EXPECTED_RAW_ROWS, EXPECTED_RAW_DATES) == (319, 39)
    assert (EXPECTED_RAW_BOARD2, EXPECTED_RAW_BOARD3) == (261, 58)
    assert (EXPECTED_MATURED_ROWS, EXPECTED_MATURED_DATES) == (307, 37)
    assert POPULATION_A_BOOTSTRAP_RESAMPLES == 20_000
    assert POPULATION_A_PERMUTATIONS == 20_000
    assert STRICT_BOOTSTRAP_RESAMPLES == 20_000
    assert JULY_RESULT_ROWS_ACCESSED == 0


def test_outcome_boundaries_and_upper_cap_only():
    raw = np.array([-0.0001, 0.0, 0.0699, 0.07, 0.20, -0.12])
    assert loss_from_raw(raw).tolist() == [1, 0, 0, 0, 0, 1]
    assert target7_from_raw(raw).tolist() == [0, 0, 0, 1, 1, 0]
    assert capped_return_7(raw).tolist() == pytest.approx(
        [-0.0001, 0.0, 0.0699, 0.07, 0.07, -0.12]
    )


def test_board_identity_is_exact():
    assert board_group(2) == "BOARD2"
    assert board_group(3) == "BOARD3"
    with pytest.raises(ValueError):
        board_group(1)
    with pytest.raises(ValueError):
        board_group(4)


def test_population_a_exact_identity_and_maturity(audit_result):
    _, context = audit_result
    matured = context["population_a_same_date"]
    summary = context["population_a_summary"]
    assert context["matured_rows"] == 307
    assert context["matured_dates"] == 37
    assert summary["BOARD2"]["rows"] + summary["BOARD3"]["rows"] == 307
    assert context["raw_audit"] == {
        "rows": 319, "dates": 39, "board2": 261, "board3": 58,
    }
    # Same-date summary is derived only after the exact matured population gate.
    assert len(matured) == context["population_a_same_date_summary"]["mixed_dates"]
    assert context["july_result_rows_accessed"] == 0


def test_strict_date_set_and_stage1_parity(audit_result):
    _, context = audit_result
    strict = context["strict"]
    parity = context["stage1_parity"]
    assert sorted(strict["signal_date"].unique()) == EXPECTED_STRICT_DATES
    assert strict["stage1_top3"].sum() == 51
    assert parity["Rank1_mean"] == pytest.approx(0.0298285616089, abs=1e-11)
    assert parity["Rank2_mean"] == pytest.approx(0.0368040074256, abs=1e-11)
    assert parity["Rank3_mean"] == pytest.approx(0.00265478297043, abs=1e-11)
    assert parity["Top3_mean"] == pytest.approx(0.0230957840016, abs=1e-11)
    assert parity["Top3_target7_precision"] == pytest.approx(0.313725490196, abs=1e-11)
    assert parity["winner_capture"] == pytest.approx(0.477777777778, abs=1e-11)
    assert parity["Top3_negative_date_rate"] == pytest.approx(0.235294117647, abs=1e-11)
    assert parity["Top3_worst"] == pytest.approx(-0.0676878130753, abs=1e-11)


def test_strict_membership_and_rank_percentile_are_exact(audit_result):
    _, context = audit_result
    strict = context["strict"]
    assert (strict["stage1_top10"] == strict["stage1_rank"].le(
        np.minimum(strict["candidate_count"], 10)
    )).all()
    assert (strict["stage1_top5"] == strict["stage1_rank"].le(
        np.minimum(strict["candidate_count"], 5)
    )).all()
    assert (strict["stage1_top3"] == strict["stage1_rank"].le(
        np.minimum(strict["candidate_count"], 3)
    )).all()
    expected = (strict["stage1_rank"] - 1) / np.maximum(strict["candidate_count"] - 1, 1)
    assert strict["stage1_rank_percentile"].to_numpy() == pytest.approx(expected)
    assert stage1_rank_percentile(1, 1) == 0.0
    assert stage1_rank_percentile(3, 5) == 0.5


def test_funnel_composition_and_selection_rate_fixture():
    # 8 candidates: 6 Board2 / 2 Board3; Top3 has 2 / 1.
    assert composition_lift(1, 3, 2, 8) == pytest.approx(4 / 3)
    b2_rate = selection_rate(2, 6)
    b3_rate = selection_rate(1, 2)
    assert b2_rate == pytest.approx(1 / 3)
    assert b3_rate == pytest.approx(1 / 2)
    assert b3_rate / b2_rate == pytest.approx(1.5)


def test_same_date_comparison_loss_gap_fixture():
    fixture = pd.DataFrame({
        "signal_date": ["A"] * 6,
        "board_group": ["BOARD2"] * 4 + ["BOARD3"] * 2,
        "loss": [0, 1, 0, 0, 1, 1],
        "target7": [0, 0, 1, 0, 0, 0],
        "capped_return_7": [0.01, -0.02, 0.07, 0.02, -0.03, -0.04],
    })
    daily, summary = population_a_same_date(fixture)
    assert len(daily) == 1
    assert daily.iloc[0]["loss_gap"] == pytest.approx(0.75)
    assert summary["loss"]["positive"] == 1
    assert summary["loss"]["direction_consistency"] == 1.0


def test_stratified_board_permutation_preserves_board_counts_per_date():
    fixture = pd.DataFrame({
        "signal_date": ["A"] * 3 + ["B"] * 4,
        "board_group": ["BOARD2", "BOARD2", "BOARD3",
                        "BOARD2", "BOARD3", "BOARD3", "BOARD2"],
    })
    labels = stratified_board_permutation(fixture, permutations=250, seed=20260812)
    assert labels.shape == (250, 7)
    assert np.all(labels[:, :3].sum(axis=1) == 1)
    assert np.all(labels[:, 3:].sum(axis=1) == 2)


def test_date_bootstrap_draws_dates_not_rows():
    counts = _bootstrap_draws(7, 100, 123)
    assert counts.shape == (100, 7)
    assert np.all(counts.sum(axis=1) == 7)


def test_board_groups_and_funnel_counts_are_exhaustive(audit_result):
    _, context = audit_result
    strict = context["strict"]
    composition = context["composition_summary"]
    assert set(strict["board_group"]) == set(BOARD_GROUPS)
    assert composition["candidate_board2"] + composition["candidate_board3"] == len(strict)
    for level, column in (("TOP10", "stage1_top10"), ("TOP5", "stage1_top5"), ("TOP3", "stage1_top3")):
        row = composition[level]
        assert row["board2"] + row["board3"] == int(strict[column].sum())


def test_bucket_assignments_are_frozen_and_small_samples_flagged(audit_result):
    _, context = audit_result
    buckets = context["buckets"]
    assert set(buckets["opportunity_bucket"]) == {"LOW", "MID", "HIGH"}
    assert set(buckets["board_group"]) == set(BOARD_GROUPS)
    assert (buckets["sample_warning"].eq("SMALL_SAMPLE") == buckets["top3_slots"].lt(5)).all()


def test_bootstrap_and_permutation_are_complete_and_deterministic(audit_result):
    _, context = audit_result
    for summary in context["population_a_bootstrap"].values():
        assert summary["valid_resamples"] == 20_000
    for summary in context["strict_bootstrap"].values():
        assert 0 < summary["valid_resamples"] <= 20_000
    assert all(0 <= value <= 1 for value in context["population_a_permutation"].values())


def test_no_model_fit_feature_search_or_board_policy_in_module():
    source = (ROOT / "src/v004c_board2_board3_structural_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {name for name in calls if name.startswith("fit_")}
    assert "sklearn" not in source
    assert "xgboost" not in source.lower()
    assert "exclude_board3" not in source
    assert "board3_penalty" not in source


def test_outputs_and_formal_status(audit_result):
    outputs, context = audit_result
    assert len(outputs) == 7
    assert all(payload for payload in outputs.values())
    assert context["decision"]["mode"] in {
        "INHERENT_BOARD3_RISK", "STAGE1_BOARD3_OVERPROMOTION", "BOTH",
        "NO_CLEAR_DIFFERENCE", "INSUFFICIENT_EVIDENCE",
    }
    assert context["decision"]["separation_warranted"] == (
        context["decision"]["mode"] in {
            "INHERENT_BOARD3_RISK", "STAGE1_BOARD3_OVERPROMOTION", "BOTH",
        }
    )

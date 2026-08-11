from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_board3_winner_information_geometry import (
    BOOTSTRAP_RESAMPLES,
    CONTRACT_FEATURE_NAMES,
    EXPECTED_GROUP_COUNTS,
    EXPECTED_STARTING_HEAD,
    FEATURE_CONTRACT,
    FROZEN_FEATURE_COLUMNS,
    JULY_RESULT_ROWS_ACCESSED,
    LOW_LEVEL_FAMILY,
    PERMUTATIONS,
    STAGE1_FAMILY,
    _same_date_pairs,
    apply_gates,
    benjamini_hochberg,
    build_outputs,
    classify_outcome,
    date_bootstrap_counts,
    ordinal_median_monotonic,
    stratified_winner_permutations,
    winner_sign_from_auc,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit_result():
    return build_outputs(ROOT)


def test_starting_contract_and_feature_families_are_exact():
    assert EXPECTED_STARTING_HEAD == "7e18d8059f83108651198bef5aa02c3b681db9d0"
    assert len(CONTRACT_FEATURE_NAMES) == len(set(CONTRACT_FEATURE_NAMES)) == 53
    assert len(FROZEN_FEATURE_COLUMNS) == len(set(FROZEN_FEATURE_COLUMNS)) == 18
    assert set(CONTRACT_FEATURE_NAMES).isdisjoint(FROZEN_FEATURE_COLUMNS)
    assert pd.Series([row["semantic_group"] for row in FEATURE_CONTRACT]).value_counts().to_dict() == EXPECTED_GROUP_COUNTS
    assert all(row["available_as_of"] in {"D0_CLOSE", "D1_CLOSE"} for row in FEATURE_CONTRACT)


def test_authoritative_and_matured_board3_population_parity(audit_result):
    _, context = audit_result
    audit = context["population_audit"]
    source = audit["source_audit"]
    population = context["population"]
    assert (source["rows"], source["dates"], source["board2"], source["board3"]) == (319, 39, 261, 58)
    assert (audit["matured_rows"], audit["matured_dates"]) == (307, 37)
    assert (audit["rows"], audit["dates"]) == (55, 28)
    assert (audit["target7"], audit["loss"], audit["positive_non_target"], audit["nonloss"]) == (23, 13, 19, 42)
    assert population["event_id"].nunique() == 55
    assert population["label_available_date"].astype(str).lt("2026-07-01").all()
    assert JULY_RESULT_ROWS_ACCESSED == context["july_result_rows_accessed"] == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(-.0001, "LOSS"), (0.0, "POSITIVE_NON_TARGET"), (.069999, "POSITIVE_NON_TARGET"), (.07, "TARGET7")],
)
def test_outcome_boundaries_are_exact(raw, expected):
    assert classify_outcome(raw) == expected


def test_capped_return_is_upper_cap_only(audit_result):
    _, context = audit_result
    population = context["population"]
    expected = np.minimum(population["raw_repair_return"].to_numpy(float), .07)
    assert np.array_equal(population["capped_return_7"].to_numpy(float), expected)
    assert population.loc[population["raw_repair_return"].lt(0), "capped_return_7"].lt(0).all()


def test_winner_orientation_is_derived_once_and_reused():
    assert winner_sign_from_auc(.30) == -1
    assert winner_sign_from_auc(.50) == 1
    fixture = pd.DataFrame({
        "signal_date": ["A"] * 4,
        "raw_repair_return": [.08, .09, .01, -.02],
        "target7": [1, 1, 0, 0], "positive_non_target": [0, 0, 1, 0],
        "loss": [0, 0, 0, 1], "nonloss": [1, 1, 1, 0],
    })
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    oriented = -raw
    _, summary = _same_date_pairs(fixture, oriented, "PNT_VS_LOSS")
    # The same -1 orientation makes PNT beat LOSS here; no endpoint re-flip occurs.
    assert summary["pairs"] == 1
    assert summary["concordance"] == 1.0


def test_ordinal_median_contract():
    assert ordinal_median_monotonic(.2, .5, .8)
    assert not ordinal_median_monotonic(.2, .8, .5)
    assert ordinal_median_monotonic(.2, .2, .8)


def test_same_date_pairs_never_cross_dates_and_ties_are_half():
    fixture = pd.DataFrame({
        "signal_date": ["A", "A", "B", "B"],
        "raw_repair_return": [.08, .02, .09, .03],
        "target7": [1, 0, 1, 0], "positive_non_target": [0, 1, 0, 1],
        "loss": [0, 0, 0, 0], "nonloss": [1, 1, 1, 1],
    })
    daily, summary = _same_date_pairs(fixture, np.ones(4), "T7_VS_PNT")
    assert len(daily) == summary["dates"] == 2
    assert summary["pairs"] == summary["ties"] == 2
    assert summary["concordance"] == .5


def test_bootstrap_samples_dates_not_rows_with_fixed_contract():
    dates, counts = date_bootstrap_counts(["A", "A", "B", "C"], resamples=100, seed=7)
    assert dates == ["A", "B", "C"]
    assert counts.shape == (100, 3)
    assert np.all(counts.sum(axis=1) == 3)
    assert BOOTSTRAP_RESAMPLES == 20_000


def test_permutation_is_nonloss_only_and_preserves_each_date_t7_count():
    fixture = pd.DataFrame({
        "signal_date": ["A"] * 6 + ["B"] * 4,
        "nonloss": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        "target7": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    })
    permuted = stratified_winner_permutations(fixture, permutations=250, seed=9)
    # Filtered NONLOSS order: A has 5 rows/2 T7, B has 4 rows/1 T7.
    assert permuted.shape == (250, 9)
    assert np.all(permuted[:, :5].sum(axis=1) == 2)
    assert np.all(permuted[:, 5:].sum(axis=1) == 1)
    assert PERMUTATIONS == 20_000


def _gate_row(**changes):
    row = {
        "coverage": 1.0, "unique_values": 10,
        "AUC_oriented_T7_vs_PNT": .66, "Cliff_T7_vs_PNT": .32,
        "AUC_oriented_T7_vs_LOSS": .66, "AUC_oriented_PNT_vs_LOSS": .56,
        "NONLOSS_raw_spearman": .21, "NONLOSS_raw_pair_concordance": .58,
        "date_mixed_count": 5, "date_direction_consistency": .65,
        "bootstrap_P_cliff_gt0": .90, "LODO_cliff_positive_pct": .80,
        "BH_q": .10, "T7_PNT_pairs": 10, "T7_PNT_pair_concordance": .60,
        "ordinal_median_monotonic": True,
    }
    row.update(changes)
    return row


def test_winner_and_ordinal_gates_are_exact():
    passed = apply_gates(pd.DataFrame([_gate_row()])).iloc[0]
    assert passed["winner_pass"]
    assert passed["ordinal_winner_pass"]
    assert passed["geometry_class"] == "ORDINAL"
    failed = apply_gates(pd.DataFrame([_gate_row(AUC_oriented_T7_vs_PNT=.649999)])).iloc[0]
    assert not failed["winner_pass"]
    assert "PRIMARY_AUC" in failed["failed_gates"]


def test_bh_families_are_separate_and_exact(audit_result):
    _, context = audit_result
    table = context["univariate"]
    assert len(table[table["family"].eq(LOW_LEVEL_FAMILY)]) == 53
    assert len(table[table["family"].eq(STAGE1_FAMILY)]) == 18
    assert table["BH_q"].between(0, 1).all()
    assert benjamini_hochberg([.01, .04, .03]).tolist() == pytest.approx([.03, .04, .04])


def test_manifest_and_outputs_have_no_new_features_or_july(audit_result):
    outputs, context = audit_result
    manifest = context["manifest"]
    assert set(manifest.loc[manifest["family"].eq(LOW_LEVEL_FAMILY), "feature"]) == set(CONTRACT_FEATURE_NAMES)
    assert set(manifest.loc[manifest["family"].eq(STAGE1_FAMILY), "feature"]) == set(FROZEN_FEATURE_COLUMNS)
    assert manifest["D1_safe"].all()
    assert len(outputs) == 8 and all(outputs.values())


def test_no_model_fit_target_search_or_threshold_sensitivity_path():
    source = (ROOT / "src/v004c_board3_winner_information_geometry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {name for name in called if name.startswith("fit_")}
    lowered = source.lower()
    assert "gradientboosting" not in lowered
    assert "randomforest" not in lowered
    assert "xgboost" not in lowered
    assert "target5" not in lowered and "target6" not in lowered and "target8" not in lowered and "target10" not in lowered


def test_orientation_is_fixed_in_bootstrap_and_lodo_outputs(audit_result):
    _, context = audit_result
    table = context["univariate"]
    assert set(table["winner_sign"].unique()).issubset({-1, 1})
    assert table["orientation_source"].eq("FULL_DEVELOPMENT_T7_VS_PNT").all()
    assert context["decision"]["foundation"] in {"ESTABLISHED", "PARTIAL", "NOT_ESTABLISHED"}

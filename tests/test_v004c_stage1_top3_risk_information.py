from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_limited_risk_protector import EXPECTED_STRICT_DATES
from src.v004c_stage1_top3_risk_information import (
    BOOTSTRAP_RESAMPLES,
    CONTRACT_FEATURE_NAMES,
    EXPECTED_GROUP_COUNTS,
    FEATURE_CONTRACT,
    FORBIDDEN_SCREEN_COLUMNS,
    JULY_RESULT_ROWS_ACCESSED,
    PERMUTATIONS,
    REDUNDANCY_THRESHOLD,
    RISK_FEATURE_COLUMNS,
    _lodo,
    apply_evidence_gates,
    benjamini_hochberg,
    binary_auc,
    build_outputs,
    date_bootstrap_counts,
    diagnostic_risk_sign,
    stratified_permutation_labels,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit_result():
    return build_outputs(ROOT)


def test_frozen_contract_is_exactly_53_d1_safe_features_in_expected_groups():
    assert len(CONTRACT_FEATURE_NAMES) == 53
    assert len(set(CONTRACT_FEATURE_NAMES)) == 53
    assert {row["feature_name"] for row in FEATURE_CONTRACT} == set(CONTRACT_FEATURE_NAMES)
    counts = pd.Series([row["semantic_group"] for row in FEATURE_CONTRACT]).value_counts()
    assert counts.to_dict() == EXPECTED_GROUP_COUNTS
    assert all(row["available_as_of"] in {"D0_CLOSE", "D1_CLOSE"}
               for row in FEATURE_CONTRACT)
    assert not FORBIDDEN_SCREEN_COLUMNS.intersection(CONTRACT_FEATURE_NAMES)
    assert set(RISK_FEATURE_COLUMNS).isdisjoint(CONTRACT_FEATURE_NAMES)


def test_primary_population_and_stage1_parity_are_exact(audit_result):
    _, context = audit_result
    population = context["population"]
    metrics = context["population_metrics"]
    assert sorted(population["signal_date"].unique()) == EXPECTED_STRICT_DATES
    assert len(population) == population["event_id"].nunique() == 51
    assert metrics["loss"] == 16
    assert metrics["nonloss"] == 35
    assert metrics["target7"] == 16
    assert metrics["positive_non_target"] == 19
    assert metrics["parity"] is True
    assert metrics["top3"] == pytest.approx(0.0230957840016, abs=1e-11)
    assert metrics["precision"] == pytest.approx(0.313725490196, abs=1e-11)
    assert metrics["negative_date_rate"] == pytest.approx(0.235294117647, abs=1e-11)
    assert metrics["worst"] == pytest.approx(-0.0676878130753, abs=1e-11)


def test_population_has_exact_feature_and_control_columns_without_july(audit_result):
    _, context = audit_result
    population = context["population"]
    assert set(CONTRACT_FEATURE_NAMES).issubset(population.columns)
    assert set(RISK_FEATURE_COLUMNS).issubset(population.columns)
    assert (population["signal_date"] < "2026-07-01").all()
    assert JULY_RESULT_ROWS_ACCESSED == 0
    assert context["july_result_rows_accessed"] == 0


def test_manifest_reports_exact_d1_safe_coverage_and_controls_are_outside_family(audit_result):
    _, context = audit_result
    manifest = context["manifest"]
    univariate = context["univariate"]
    assert len(manifest) == 53
    assert manifest["d1_safe"].all()
    assert set(univariate.loc[univariate["is_low_level_53"], "feature"]) == set(
        CONTRACT_FEATURE_NAMES
    )
    assert set(univariate.loc[univariate["is_control_representation"], "feature"]) == set(
        RISK_FEATURE_COLUMNS
    )
    assert univariate["is_low_level_53"].sum() == 53


def test_module_contains_no_model_fit_or_threshold_trading_policy():
    source = (ROOT / "src/v004c_stage1_top3_risk_information.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {name for name in called if name.startswith("fit_")}
    assert "sklearn" not in source
    assert "xgboost" not in source.lower()
    assert "lightgbm" not in source.lower()


def test_orientation_rule_and_same_sign_target7_comparison():
    labels = np.array([1, 1, 0, 0])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    raw_auc = binary_auc(labels, values)
    assert raw_auc == pytest.approx(0.0)
    sign = diagnostic_risk_sign(raw_auc)
    assert sign == -1
    assert binary_auc(labels, sign * values) == pytest.approx(1.0)
    # The Target7 safety comparison must retain -1 rather than optimizing again.
    target_labels = np.array([1, 1, 0, 0])
    target_values = np.array([0.0, 1.0, -2.0, -3.0])
    assert binary_auc(target_labels, sign * target_values) == pytest.approx(0.0)


def test_stratified_permutation_preserves_each_date_loss_count():
    fixture = pd.DataFrame({
        "signal_date": ["A"] * 3 + ["B"] * 4,
        "loss_target": [1, 0, 0, 1, 1, 0, 0],
    })
    permuted = stratified_permutation_labels(fixture, permutations=250, seed=20260812)
    assert permuted.shape == (250, 7)
    assert np.all(permuted[:, :3].sum(axis=1) == 1)
    assert np.all(permuted[:, 3:].sum(axis=1) == 2)


def test_bootstrap_resamples_dates_not_rows_and_uses_fixed_contract():
    dates, counts = date_bootstrap_counts(["A", "A", "B", "C"], resamples=100, seed=7)
    assert dates == ["A", "B", "C"]
    assert counts.shape == (100, 3)
    assert np.all(counts.sum(axis=1) == 3)
    assert BOOTSTRAP_RESAMPLES == 20_000
    assert PERMUTATIONS == 10_000


def _passing_row(**changes):
    row = {
        "feature": "x",
        "group": "D1_PRICE_ACTION",
        "coverage": 1.0,
        "oriented_auc_loss_nonloss": 0.61,
        "raw_cliff_delta": -0.31,
        "oriented_auc_loss_target7": 0.61,
        "mixed_dates": 5,
        "date_direction_consistency": 0.65,
        "bootstrap_cliff_positive_probability": 0.90,
        "lodo_cliff_positive_pct": 0.80,
        "permutation_p": 0.01,
        "is_low_level_53": True,
        "is_control_representation": False,
    }
    row.update(changes)
    return row


def test_foundation_gate_is_exact_and_coverage_48_of_51_fails():
    passed = apply_evidence_gates(pd.DataFrame([_passing_row()])).iloc[0]
    assert passed["foundation_pass"]
    coverage = 48 / 51
    assert coverage == pytest.approx(0.941176470588)
    failed = apply_evidence_gates(pd.DataFrame([
        _passing_row(coverage=coverage)
    ])).iloc[0]
    assert failed["coverage_fail"]
    assert not failed["foundation_pass"]
    assert "COVERAGE" in failed["failed_gates"]


def test_exceptional_gate_is_exact():
    row = _passing_row(
        oriented_auc_loss_nonloss=0.70,
        oriented_auc_loss_target7=0.70,
        raw_cliff_delta=0.45,
        date_direction_consistency=0.70,
        bootstrap_cliff_positive_probability=0.95,
        lodo_cliff_positive_pct=0.90,
        permutation_p=0.01,
    )
    result = apply_evidence_gates(pd.DataFrame([row])).iloc[0]
    assert result["foundation_exceptional"]
    assert result["evidence_status"] == "FOUNDATION_EXCEPTIONAL"


def test_benjamini_hochberg_monotone_correction():
    q = benjamini_hochberg([0.01, 0.04, 0.03])
    assert q.tolist() == pytest.approx([0.03, 0.04, 0.04])


def test_lodo_keeps_full_sample_orientation_without_reflipping():
    rows = []
    for i, date in enumerate(EXPECTED_STRICT_DATES):
        rows.extend([
            {"signal_date": date, "loss_target": 1, "x": float(i)},
            {"signal_date": date, "loss_target": 0, "x": float(i + 1)},
            {"signal_date": date, "loss_target": 0, "x": float(i + 2)},
        ])
    frame = pd.DataFrame(rows)
    result = _lodo(frame, "x", risk_sign=-1)
    expected = []
    for date in EXPECTED_STRICT_DATES:
        fold = frame[frame["signal_date"].ne(date)]
        expected.append(binary_auc(fold["loss_target"], -fold["x"]))
    assert result["lodo_auc_min"] == pytest.approx(min(expected))
    assert result["lodo_auc_median"] == pytest.approx(np.median(expected))
    assert result["lodo_auc_gt_half_pct"] == pytest.approx(
        np.mean(np.asarray(expected) > 0.5)
    )


def test_date_stability_and_redundancy_contracts(audit_result):
    _, context = audit_result
    stability = context["date_stability"]
    redundancy = context["redundancy"]
    assert set(stability["signal_date"]).issubset(EXPECTED_STRICT_DATES)
    assert (stability["risk_sign"].abs() == 1).all()
    if not redundancy.empty:
        assert (redundancy["near_redundant"] ==
                redundancy["abs_rho"].ge(REDUNDANCY_THRESHOLD)).all()


def test_outputs_are_exactly_seven_and_formal_status_is_valid(audit_result):
    outputs, context = audit_result
    assert len(outputs) == 7
    assert all(payload for payload in outputs.values())
    assert context["decision"]["status"] in {
        "ESTABLISHED", "PARTIAL", "NOT_ESTABLISHED"
    }
    assert context["decision"]["experiment_warranted"] == (
        context["decision"]["status"] == "ESTABLISHED"
    )


def test_no_trading_counterfactual_or_new_feature_output(audit_result):
    _, context = audit_result
    allowed = set(CONTRACT_FEATURE_NAMES) | set(RISK_FEATURE_COLUMNS)
    assert set(context["univariate"]["feature"]) == allowed
    assert "top3_return_if_veto" not in context["univariate"].columns
    assert "threshold" not in context["univariate"].columns

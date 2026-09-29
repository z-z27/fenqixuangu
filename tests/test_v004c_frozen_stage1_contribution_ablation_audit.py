from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.v004c_frozen_stage1_contribution_ablation_audit as audit
from src.v004c_stage1_18f_bridge_analysis import FROZEN_COEFFICIENTS
from src.v004c_v4a_architecture_transfer import FROZEN_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = ROOT / "reports" / "research" / audit.BRIDGE_DIRNAME
OUTPUT_DIR = ROOT / "reports" / "research" / audit.OUTPUT_DIRNAME


def _bridge() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(
        BRIDGE_DIR / audit.BRIDGE_CANDIDATES,
        encoding="utf-8-sig",
        dtype={"event_id": str, "code": str},
    )
    spec = json.loads((BRIDGE_DIR / audit.BRIDGE_SPEC).read_text(encoding="utf-8"))
    return frame, spec


def test_contract_contains_only_four_predeclared_versions_and_no_learner() -> None:
    assert tuple(audit.ABLATION_VERSIONS) == (
        "A0_ORIGINAL_18F",
        "A1_REMOVE_D0_TIMING",
        "A2_REMOVE_D0_TIMING_AND_TOTAL_FAMILY",
        "A3_REDUCED_BASE_ONLY",
    )
    assert audit.ABLATION_VERSIONS["A1_REMOVE_D0_TIMING"] == audit.D0_TIMING_FEATURES
    assert len(audit.ABLATION_VERSIONS["A2_REMOVE_D0_TIMING_AND_TOTAL_FAMILY"]) == 7
    assert set(FROZEN_FEATURE_COLUMNS).difference(
        audit.ABLATION_VERSIONS["A3_REDUCED_BASE_ONLY"]
    ) == set(audit.REDUCED_BASE_FEATURES)
    assert len(audit.ABLATION_VERSIONS["A3_REDUCED_BASE_ONLY"]) == 11
    source = inspect.getsource(audit)
    forbidden = (
        "fit_logistic_l2_weighted(",
        "fit_logistic(",
        "GradientBoosting",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "pairwise fit",
    )
    assert not any(token in source for token in forbidden)


def test_a0_exact_parity_and_july_locked_identity() -> None:
    context = audit.analyze(ROOT)
    parity = context["parity"]
    assert parity["rows"] == 497
    assert parity["dates"] == 62
    assert parity["duplicate_event_id_count"] == 0
    assert parity["missing_18f_cell_count"] == 0
    assert parity["reconstructed_score_max_abs_error"] <= 1e-12
    assert parity["reconstructed_rank_mismatch_count"] == 0
    assert parity["july_locked_population_identity_mismatch_count"] == 0
    assert (parity["july_rows"], parity["july_dates"]) == (178, 23)


def test_ablation_is_exact_contribution_zeroing_without_refit() -> None:
    frame, spec = _bridge()
    a0, _ = audit.build_a0_and_parity(frame, spec, ROOT)
    scored = audit.build_ablation_scores(a0, spec)
    for model, features in audit.ABLATION_VERSIONS.items():
        prefix = model.split("_", 1)[0]
        if prefix == "A0":
            continue
        expected = sum(
            scored[feature].to_numpy(float) * float(spec["coefficients"][feature])
            for feature in features
        )
        assert np.allclose(
            scored[f"{prefix}_removed_contribution"], expected, rtol=0.0, atol=1e-14
        )
        assert np.allclose(
            scored[f"{prefix}_logit"], scored["A0_logit"] - expected,
            rtol=0.0, atol=1e-14,
        )
        expected_score = 1.0 / (
            1.0 + np.exp(-np.clip(scored[f"{prefix}_logit"], -35.0, 35.0))
        )
        assert np.allclose(scored[f"{prefix}_score"], expected_score, rtol=0.0, atol=1e-15)
    assert spec["coefficients"] == dict(FROZEN_COEFFICIENTS)


def test_deterministic_ranking_uses_score_desc_event_id_ascending() -> None:
    synthetic = pd.DataFrame({
        "signal_date": ["2026-05-06"] * 4,
        "event_id": ["B", "A", "D", "C"],
        "score": [0.9, 0.9, 0.2, 0.5],
    })
    ranked = audit._rank_scores(synthetic, "score", "rank")
    assert ranked.sort_values("rank")["event_id"].tolist() == ["A", "B", "C", "D"]


def test_mature_only_period_excludes_entire_july_30_31_dates() -> None:
    context = audit.analyze(ROOT)
    daily = context["daily"]
    main = daily[
        daily["population_scope"].eq("ALL_CANDIDATES")
        & daily["model"].eq("A0_ORIGINAL_18F")
    ]
    assert set(main.loc[main["outcome_evaluation_eligible"].eq("NO"), "signal_date"]) == {
        "2026-07-30", "2026-07-31"
    }
    mature = audit._period_daily(main, "JULY_MATURE_ONLY")
    assert mature["signal_date"].nunique() == 21
    assert not mature["signal_date"].isin(("2026-07-30", "2026-07-31")).any()
    scored = context["scored"]
    august_label_rows = scored["label_available_date"].ge("2026-08-01")
    assert scored.loc[august_label_rows, audit.OUTCOME_COLUMNS].isna().all().all()
    assert not scored["signal_date"].ge("2026-08-01").any()


def test_daily_auc_is_date_equal_and_recall_uses_actual_k() -> None:
    context = audit.analyze(ROOT)
    daily = context["daily"]
    may_a0 = daily[
        daily["population_scope"].eq("ALL_CANDIDATES")
        & daily["model"].eq("A0_ORIGINAL_18F")
        & daily["month"].eq("2026-05")
    ]
    summary = context["summary"]
    row = summary[
        summary["period"].eq("MAY")
        & summary["population_scope"].eq("ALL_CANDIDATES")
        & summary["model"].eq("A0_ORIGINAL_18F")
    ].iloc[0]
    assert np.isclose(
        row["target7_within_date_auc"], may_a0["target7_within_date_auc"].mean()
    )
    assert (daily["top5_selected_rows"] <= daily["candidate_count"]).all()
    assert "recall_at_10" not in daily.columns


def test_membership_accounting_and_date_bootstrap_are_exact() -> None:
    context = audit.analyze(ROOT)
    membership = context["membership"]
    for challenger, group in membership.groupby("challenger"):
        for _, day in group.groupby("signal_date"):
            assert int(day["change_type"].eq("PROMOTED").sum()) == int(
                day["change_type"].eq("DEMOTED").sum()
            )
    bootstrap = context["bootstrap"]
    assert set(bootstrap["sampling_unit"]) == {"signal_date"}
    assert set(bootstrap["resamples"]) == {20_000}
    assert set(bootstrap["challenger"]) == set(audit.MODEL_ORDER[1:])
    assert len(bootstrap) == len(audit.PERIOD_ORDER) * 3


def test_days_since_d0_parity_is_record_only_and_frozen_inputs_unchanged() -> None:
    context = audit.analyze(ROOT)
    parity = context["days_parity"]
    assert list(parity.columns) == [
        "event_id", "signal_date", "code", "d0_date", "d1_date",
        "raw_days_since_d0", "canonical_calendar_days", "match",
    ]
    assert context["days_audit"] == {"mismatch_rows": 5, "mismatch_dates": 4}
    bridge, _ = _bridge()
    compared = context["scored"][["event_id", "raw__days_since_d0"]].merge(
        bridge[["event_id", "raw__days_since_d0"]],
        on="event_id", suffixes=("_audit", "_bridge"), validate="one_to_one",
    )
    assert compared["raw__days_since_d0_audit"].equals(
        compared["raw__days_since_d0_bridge"]
    )


def test_outputs_exist_and_review_has_only_six_questions() -> None:
    for name in audit.OUTPUT_FILENAMES:
        assert (OUTPUT_DIR / name).is_file()
    review = (OUTPUT_DIR / audit.OUTPUT_FILENAMES[-1]).read_text(encoding="utf-8")
    assert sum(review.count(f"## Q{i}.") for i in range(1, 7)) == 6
    assert "NEXT_ACTION = INSUFFICIENT_EVIDENCE" in review
    assert "A4" not in review and "August" not in review

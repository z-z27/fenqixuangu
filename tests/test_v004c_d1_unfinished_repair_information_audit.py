from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_d1_unfinished_repair_information_audit import (
    AUGUST_ASOF,
    BOOTSTRAP_RESAMPLES,
    FEATURES,
    PERIODS,
    RANK_REGIONS,
    _direction_stability,
    _pair_rows,
    build_outputs,
    build_population,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_four_field_contract_and_population_parity():
    assert FEATURES == (
        "d1_afternoon_return",
        "d1_open_to_close_return_raw",
        "d1_close_to_vwap_raw",
        "d1_low_to_close_recovery",
    )
    population, audit = build_population(ROOT)
    assert (len(population), population["signal_date"].nunique()) == (485, 60)
    assert population.groupby("month").size().to_dict() == {
        "2026-05": 146, "2026-06": 173, "2026-07": 166,
    }
    assert population["event_id"].is_unique
    assert not population[list(FEATURES)].isna().any().any()
    assert population["signal_date"].max() == "2026-07-29"
    assert population["label_available_date"].lt(AUGUST_ASOF).all()
    assert audit["case_pack_semantic_parity_max_abs_error"] <= 1e-12
    assert audit["reconstructed_event_count"] == 6
    assert audit["model_refits"] == 0
    assert audit["august_signal_outcome_accessed"] == "NO"


def test_same_date_pair_semantics_and_ties():
    frame = pd.DataFrame([
        {"signal_date": "2026-06-01", "event_id": "T1", "code": "1", "s2_rank": 1, "target7": 1, "loss": 0, "x": 2.0},
        {"signal_date": "2026-06-01", "event_id": "L1", "code": "2", "s2_rank": 2, "target7": 0, "loss": 1, "x": 1.0},
        {"signal_date": "2026-06-01", "event_id": "L2", "code": "3", "s2_rank": 3, "target7": 0, "loss": 1, "x": 2.0},
        {"signal_date": "2026-06-02", "event_id": "T2", "code": "4", "s2_rank": 1, "target7": 1, "loss": 0, "x": 0.0},
        {"signal_date": "2026-06-02", "event_id": "L3", "code": "5", "s2_rank": 2, "target7": 0, "loss": 1, "x": 3.0},
    ])
    pairs = _pair_rows(frame, "x", "target7", "loss", "T7_VS_LOSS")
    assert len(pairs) == 3
    assert set(pairs["signal_date"]) == {"2026-06-01", "2026-06-02"}
    assert np.allclose(pairs["raw_concordance"], [1.0, .5, 0.0])
    assert np.allclose(pairs["unfinished_low_concordance"], [0.0, .5, 1.0])
    assert not any(
        row.positive_event_id == "T1" and row.negative_event_id == "L3"
        for row in pairs.itertuples(index=False)
    )


def test_direction_stability_requires_monthly_sign_consistency():
    assert _direction_stability([.55, .60, .51]) == "STABLE_T7_HIGH"
    assert _direction_stability([.45, .40, .49]) == "STABLE_T7_LOW"
    assert _direction_stability([.55, .45, .55]) == "TEMPORALLY_UNSTABLE"
    assert _direction_stability([.50, .50, .50]) == "NEAR_RANDOM"


def test_fixed_rank_regions_only():
    assert RANK_REGIONS == {
        "RANK1_3": (1, 3), "RANK2_6": (2, 6), "RANK4_7": (4, 7)
    }


def test_complete_outputs_deterministic_and_no_model_search():
    first, audit = build_outputs(ROOT)
    second, second_audit = build_outputs(ROOT)
    assert first == second
    assert audit == second_audit
    assert set(first) == {
        "v004c_unfinished_repair_population_v001.csv",
        "v004c_unfinished_repair_feature_summary_v001.csv",
        "v004c_unfinished_repair_t7_loss_pairs_v001.csv",
        "v004c_unfinished_repair_rank_region_summary_v001.csv",
        "v004c_unfinished_repair_rank3_failure_v001.csv",
        "v004c_unfinished_repair_case_vs_population_v001.csv",
        "v004c_unfinished_repair_bootstrap_v001.csv",
        "v004c_unfinished_repair_review_v001.md",
    }
    assert audit["four_feature_count"] == 4
    assert audit["model_refits"] == 0
    assert audit["next_action"] == "STOP_AND_REVIEW"
    assert audit["unfinished_repair_state"] in {
        "UNFINISHED_REPAIR_INFORMATION_SUPPORTED",
        "CASE_CONDITIONAL_ONLY",
        "UNFINISHED_REPAIR_INFORMATION_NOT_SUPPORTED",
        "MIXED",
    }
    bootstrap = pd.read_csv(
        pd.io.common.BytesIO(first["v004c_unfinished_repair_bootstrap_v001.csv"]),
        encoding="utf-8-sig",
    )
    assert set(bootstrap["period"]) == set(PERIODS)
    assert bootstrap["sampling_unit"].eq("SIGNAL_DATE").all()
    assert bootstrap["resamples"].eq(BOOTSTRAP_RESAMPLES).all()
    review = first["v004c_unfinished_repair_review_v001.md"].decode("utf-8")
    assert "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO" in review


def test_module_contains_no_learner_or_search_calls():
    text = (ROOT / "src/v004c_d1_unfinished_repair_information_audit.py").read_text(encoding="utf-8")
    forbidden = (
        "fit_logistic_l2_weighted(", "fit_logistic(", "GradientBoostingClassifier(",
        "RandomForestClassifier(", "XGBClassifier(", "PCA(", "KMeans(",
    )
    assert not any(token in text for token in forbidden)


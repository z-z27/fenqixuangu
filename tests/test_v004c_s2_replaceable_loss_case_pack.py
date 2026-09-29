from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_pairwise_feature_contract import CONTRACT_FEATURE_NAMES
from src.v004c_s2_replaceable_loss_case_pack import (
    MAX_SIGNAL_DATE,
    PATH_DESCRIPTOR_COLUMNS,
    build_case_pairs,
    build_lineage,
    build_outputs,
    describe_path,
)
from src.v004c_s2_winner_vs_loss_failure_attribution import load_frozen_s2


ROOT = Path(__file__).resolve().parents[1]


def test_contract_and_case_identity():
    scored, _, audit = load_frozen_s2(ROOT)
    pairs, pair_audit = build_case_pairs(scored, ROOT)
    assert audit["model_refits"] == 0
    assert pairs["signal_date"].max() <= MAX_SIGNAL_DATE
    assert pairs["loss_event_id"].nunique() == 25
    assert pair_audit["theoretical_one_to_one_replaceable_slots"] == 25
    assert len(pairs) == 25
    assert not pairs.duplicated(["loss_event_id", "target_event_id"]).any()
    assert set(pairs["pair_type"]) == {"HIGHEST_RANKED_OUTSIDE_TARGET7"}
    assert pairs["also_nearest_score_outside_target7"].eq(1).all()
    assert pairs["loss_s2_rank"].le(3).all()
    assert pairs["target_s2_rank"].gt(3).all()
    assert pairs["loss_raw_repair_return"].lt(0).all()
    assert pairs["target_raw_repair_return"].ge(0.07).all()


def test_descriptor_formulas_on_synthetic_day():
    timestamps = pd.date_range("2026-05-06 09:35", periods=48, freq="5min")
    frame = pd.DataFrame({
        "datetime": timestamps,
        "open": np.linspace(100, 104.7, 48),
        "high": np.linspace(100.5, 105.2, 48),
        "low": np.linspace(99.5, 104.2, 48),
        "close": np.linspace(100.1, 104.8, 48),
        "volume": np.ones(48) * 100,
        "amount": np.linspace(10010, 10480, 48),
    })
    # Put the afternoon on the canonical exchange clock.
    frame.loc[24:, "datetime"] = pd.date_range("2026-05-06 13:05", periods=24, freq="5min")
    result = describe_path(frame)
    assert set(result) == set(PATH_DESCRIPTOR_COLUMNS)
    assert result["d1_open"] == 100.0
    assert np.isclose(result["low_to_close_recovery_pct"], 104.8 / 99.5 - 1)
    assert result["bars_after_daily_low"] == 47
    assert 0 <= result["morning_amount_share"] <= 1
    assert 0 <= result["bars_close_above_intraday_vwap_share"] <= 1


def test_lineage_is_exact_existing_53():
    lineage = build_lineage()
    assert lineage["feature_name"].tolist() == CONTRACT_FEATURE_NAMES
    assert len(lineage) == 53
    assert lineage["available_at_D1_close"].all()
    assert lineage["existing_historical_audit_used"].all()


def test_full_build_integrity_and_no_august():
    outputs, metrics = build_outputs(ROOT)
    assert metrics["s2_model_refits"] == 0
    assert metrics["august_signal_outcome_accessed"] == "NO"
    assert metrics["case_pack_state"] == "READY_FOR_MECHANISM_REVIEW"
    assert metrics["integrity_failures"] == 0
    assert metrics["chart_count"] == 25
    assert metrics["minute_rows"] == 25 * 2 * 48
    assert "v004c_s2_case_pack_review_v001.md" in outputs
    assert sum(name.startswith("case_charts/") for name in outputs) == 25
    assert b"AUGUST_SIGNAL_OUTCOME_ACCESSED = NO" in outputs["v004c_s2_case_pack_review_v001.md"]


def test_deterministic_core_outputs():
    first, _ = build_outputs(ROOT)
    second, _ = build_outputs(ROOT)
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], name


def test_no_learner_or_search_tokens_in_module():
    text = (ROOT / "src/v004c_s2_replaceable_loss_case_pack.py").read_text(encoding="utf-8")
    forbidden_calls = (
        "fit_logistic_l2_weighted(", "fit_logistic(", "GradientBoostingClassifier(",
        "RandomForestClassifier(", "XGBClassifier(", "PCA(", "KMeans(",
    )
    assert not any(token in text for token in forbidden_calls)
    assert 'MAX_SIGNAL_DATE = "2026-07-29"' in text
    assert 'august_signal_outcome_accessed": "NO"' in text

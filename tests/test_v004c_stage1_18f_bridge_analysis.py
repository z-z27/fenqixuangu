from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.v004c_stage1_18f_bridge_analysis as bridge
from src.v004c_v4a_architecture_transfer import FROZEN_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "reports/research/"
    "v004c_stage1_18f_bridge_analysis_v001_20260506_20260731"
)


def _candidates() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "v004c_stage1_18f_bridge_candidates.csv",
        encoding="utf-8-sig",
        dtype={"code": str},
    )


def test_frozen_contract_and_analysis_has_no_learner_path() -> None:
    assert list(bridge.FROZEN_COEFFICIENTS) == FROZEN_FEATURE_COLUMNS
    assert len(FROZEN_FEATURE_COLUMNS) == 18
    source = inspect.getsource(bridge)
    forbidden_calls = (
        "fit_logistic_l2_weighted(",
        "_fit_stage1_snapshot(",
        "GradientBoosting",
        "RandomForest",
        "XGBoost",
    )
    assert not any(token in source for token in forbidden_calls)


def test_feature_lineage_exact_order_and_schema() -> None:
    lineage = pd.read_csv(
        OUTPUT_DIR / "v004c_stage1_feature_lineage.csv",
        encoding="utf-8-sig",
    )
    assert lineage["feature_name"].tolist() == FROZEN_FEATURE_COLUMNS
    assert len(lineage) == 18
    assert set(lineage["daily_cross_sectional"]) <= {"YES", "NO"}
    assert set(lineage["interaction"]) <= {"YES", "NO"}


def test_daily_rank_semantics_use_the_complete_daily_candidate_universe() -> None:
    frame = _candidates()
    parent_map = {
        "rank_d1_close_ma10_pct": "raw__d1_close_ma10_pct",
        "rank_d1_low_ma10_pct": "raw__d1_low_ma10_pct",
        "rank_trend_hold_score": "raw__trend_hold_score",
        "rank_total_score": "raw__total_score",
        "rank_theme_score": "raw__theme_score",
        "rank_days_since_d0": "raw__days_since_d0",
        "rank_log_candidate_base_price": "raw__log_candidate_base_price",
        "rank_active_money_score": "raw__active_money_score",
        "rank_d1_close_vwap_pct": "raw__d1_close_vwap_pct",
    }
    for feature, raw_column in parent_map.items():
        expected = frame.groupby("signal_date", sort=True)[raw_column].transform(
            lambda values: pd.to_numeric(values, errors="coerce").rank(
                pct=True, method="average"
            )
        ).fillna(0.5)
        assert np.allclose(frame[feature], expected, rtol=0.0, atol=1e-12)

    assert np.allclose(
        frame["inter_close_low"],
        frame["rank_d1_close_ma10_pct"] * frame["rank_d1_low_ma10_pct"],
        rtol=0.0, atol=2e-12,
    )
    assert np.allclose(
        frame["spread_close_low"],
        frame["rank_d1_close_ma10_pct"] - frame["rank_d1_low_ma10_pct"],
        rtol=0.0, atol=2e-12,
    )


def test_complete_population_and_july_locked_parity() -> None:
    frame = _candidates()
    assert len(frame) == 497
    assert frame["signal_date"].nunique() == 62
    month = frame["signal_date"].str.slice(0, 7)
    assert (int((month == "2026-05").sum()), frame.loc[month == "2026-05", "signal_date"].nunique()) == (146, 18)
    assert (int((month == "2026-06").sum()), frame.loc[month == "2026-06", "signal_date"].nunique()) == (173, 21)
    assert (int((month == "2026-07").sum()), frame.loc[month == "2026-07", "signal_date"].nunique()) == (178, 23)
    assert frame["event_id"].duplicated().sum() == 0
    assert frame[FROZEN_FEATURE_COLUMNS].isna().sum().sum() == 0

    july = frame[month == "2026-07"].copy()
    lock = pd.read_csv(
        ROOT / bridge.JULY_LOCK_RELATIVE_PATH,
        usecols=["event_id", "stage1_score", "stage1_rank", "candidate_count"],
        encoding="utf-8-sig",
    )
    checked = july.merge(lock, on="event_id", suffixes=("_package", "_lock"), validate="one_to_one")
    assert set(july["event_id"]) == set(lock["event_id"])
    assert np.max(np.abs(checked["stage1_score_package"] - checked["stage1_score_lock"])) <= 1e-12
    assert checked["stage1_rank_package"].equals(checked["stage1_rank_lock"])
    assert checked["candidate_count_package"].equals(checked["candidate_count_lock"])


def test_score_reconstructs_from_exported_spec() -> None:
    frame = _candidates()
    spec = json.loads(
        (OUTPUT_DIR / "v004c_stage1_model_spec.json").read_text(encoding="utf-8")
    )
    beta = np.asarray([spec["coefficients"][name] for name in spec["features"]])
    logit = float(spec["intercept"]) + frame[spec["features"]].to_numpy(float) @ beta
    score = 1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0)))
    assert np.max(np.abs(score - frame["stage1_score"].to_numpy(float))) <= 1e-12
    assert spec["new_model_trained"] is False
    assert spec["feature_search"] is False
    assert spec["hyperparameter_search"] is False


def test_august_outcomes_are_neither_accessed_nor_exported() -> None:
    frame = _candidates()
    august_labels = frame["label_available_date"].ge("2026-08-01")
    outcome_columns = [
        "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
    ]
    assert august_labels.any()
    assert frame.loc[august_labels, outcome_columns].isna().all().all()
    assert frame.loc[~august_labels, outcome_columns].notna().all().all()
    integrity = pd.read_csv(
        OUTPUT_DIR / "v004c_stage1_bridge_integrity.csv",
        encoding="utf-8-sig",
    )
    august_access = integrity[
        integrity["metric"].eq("august_outcome_rows_accessed")
    ]
    assert len(august_access) == 1
    assert float(august_access.iloc[0]["value"]) == 0.0
    assert not integrity["status"].eq("FAIL").any()

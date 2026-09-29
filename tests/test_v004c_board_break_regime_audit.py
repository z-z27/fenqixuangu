from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v004c_board_break_regime_audit import (
    AUGUST_ASOF,
    FAILED_BOARD_STATUS,
    OUTPUT_FILENAMES,
    PATH_FEATURES,
    _oracle_mean,
    build_board_ecology,
    build_candidate_pool,
    build_outputs,
    build_recent_matured_history,
    load_population,
)


ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_mature_population_and_no_august():
    population, audit = load_population(ROOT)
    assert (len(population), population["signal_date"].nunique()) == (485, 60)
    assert population.groupby("month").size().to_dict() == {
        "MAY": 146, "JUNE": 173, "JULY": 166,
    }
    assert population["event_id"].is_unique
    assert population["signal_date"].max() == "2026-07-29"
    assert population["signal_date"].lt(AUGUST_ASOF).all()
    assert population["label_available_date"].lt(AUGUST_ASOF).all()
    assert not population[list(PATH_FEATURES)].isna().any().any()
    assert audit["model_fits"] == 0
    assert audit["external_data_connections"] == 0
    assert audit["august_signal_outcome_accessed"] == "NO"


def test_candidate_pool_counts_and_oracle_are_exact():
    population, _ = load_population(ROOT)
    candidate = build_candidate_pool(population)
    assert len(candidate) == 60
    assert int(candidate["candidate_count"].sum()) == 485
    assert int(candidate["target7_count"].sum()) == 159
    assert int(candidate["loss_count"].sum()) == 131
    assert int(candidate["severe_loss_count"].sum()) == 27
    synthetic = pd.DataFrame({"capped_return_7": [-.03, .02, .07, .05]})
    assert _oracle_mean(synthetic, 1) == .07
    assert _oracle_mean(synthetic, 2) == pytest.approx(.06)
    assert _oracle_mean(synthetic, 3) == pytest.approx(np.mean([.07, .05, .02]))


def test_existing_board_cache_coverage_and_failed_board_not_fabricated():
    population, _ = load_population(ROOT)
    ecology = build_board_ecology(ROOT, population["signal_date"].unique())
    assert len(ecology) == 60
    assert ecology["data_status"].eq("AVAILABLE_EXISTING_CACHE").all()
    assert ecology["limit_up_count"].gt(0).all()
    assert ecology["highest_board_height"].ge(1).all()
    assert ecology["failed_limit_up_count"].isna().all()
    assert ecology["failed_limit_up_rate"].isna().all()
    assert ecology["failed_limit_up_status"].eq(FAILED_BOARD_STATUS).all()
    assert ecology["pool_semantics"].eq("EXISTING_MAIN_BOARD_FINAL_LIMIT_UP_POOL").all()


def test_recent_five_history_uses_only_strictly_matured_past_rows():
    frame = pd.DataFrame([
        {"signal_date": f"2026-05-{day:02d}", "label_available_date": f"2026-05-{day + 2:02d}",
         "target7": day % 2, "loss": (day + 1) % 2, "capped_return_7": day / 1000}
        for day in range(1, 9)
    ])
    result = build_recent_matured_history(frame)
    for row in result.itertuples(index=False):
        if row.recent5_status == "AVAILABLE_STRICT_MATURED":
            assert row.history_max_label_available_date < row.signal_date
            assert row.strict_no_future_label
            assert row.recent5_matured_date_count == 5


def test_outputs_are_exact_deterministic_and_descriptive_only():
    first, audit = build_outputs(ROOT)
    second, second_audit = build_outputs(ROOT)
    assert first == second
    assert audit == second_audit
    assert tuple(first) == OUTPUT_FILENAMES
    assert audit["output_file_count"] == 7
    assert audit["board_ecology_missing_dates"] == 0
    assert audit["local_regime_state"] in {
        "LOCAL_REGIME_DIFFERENCE_SUPPORTED",
        "LOCAL_REGIME_DIFFERENCE_WEAK",
        "LOCAL_REGIME_DIFFERENCE_NOT_SUPPORTED",
    }
    assert audit["broader_market_context_needed"] in {"YES", "NO", "UNCERTAIN"}
    review = first["v004c_regime_review_v001.md"].decode("utf-8")
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review
    assert "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO" in review
    summary = pd.read_csv(
        pd.io.common.BytesIO(first["v004c_regime_month_summary_v001.csv"]),
        encoding="utf-8-sig",
    )
    assert {"metric", "may", "june", "july", "change_direction"}.issubset(summary)
    assert summary["metric"].is_unique


def test_module_has_no_learner_search_or_network_fetch():
    text = (ROOT / "src/v004c_board_break_regime_audit.py").read_text(encoding="utf-8")
    forbidden = (
        "fit_logistic", "GradientBoosting", "RandomForest", "XGBClassifier",
        "pairwise", "requests.", "akshare", "fetch_limit_up_pool(",
    )
    assert not any(token in text for token in forbidden)

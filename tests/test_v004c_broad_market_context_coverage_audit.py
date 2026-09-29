from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_broad_market_context_coverage_audit import (
    CONTEXT_VARIABLES,
    OUTPUT_FILENAMES,
    RAW_SPOT_FIELDS,
    build_daily_context,
    build_outputs,
    build_redundancy,
    build_source_field_status,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_inputs():
    universe = pd.DataFrame({
        "code": ["000001", "000002", "600001", "600002"],
        "name": ["A", "B", "C", "D"],
        "float_market_cap": [1e9, 2e9, 3e9, 4e9],
        "total_market_cap": [2e9, 3e9, 4e9, 5e9],
    })
    listing = pd.DataFrame(columns=["code", "listing_date"])
    rows = []
    for code, pct in zip(universe["code"], [4.0, 1.0, -2.0, -5.0]):
        rows.append({
            "date": "2026-05-06", "code": code, "pct_chg": pct,
            "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
            "amount": np.nan, "change": pct, "amplitude": 0,
            "turnover_rate": np.nan, "source": "synthetic",
            "cache_kind": "daily_unadjusted", "in_current_universe": True,
        })
    return universe, listing, pd.DataFrame(rows)


def test_fixed_four_variable_manifest_and_spot_contract():
    assert CONTEXT_VARIABLES == (
        "market_up_ratio", "market_median_return", "market_tail_balance",
        "size_style_spread",
    )
    assert RAW_SPOT_FIELDS["pct_chg"][0] == "f3"
    assert RAW_SPOT_FIELDS["float_market_cap"][0] == "f21"
    assert len(CONTEXT_VARIABLES) == 4


def test_daily_values_use_fixed_definitions_and_m4_fails_closed():
    universe, listing, history = _synthetic_inputs()
    coverage, values = build_daily_context(universe, listing, history)
    assert len(coverage) == len(values) == 1
    row = values.iloc[0]
    assert row["market_up_ratio"] == .5
    assert row["market_median_return"] == -.5
    assert row["big_up_ratio"] == .25
    assert row["big_down_ratio"] == .25
    assert row["market_tail_balance"] == 0
    assert pd.isna(row["size_style_spread"])
    assert coverage.iloc[0]["m4_status"] == "SIZE_CONTEXT_DATE_INCOMPLETE"


def test_source_normalize_loss_is_reported_not_hidden():
    _, _, history = _synthetic_inputs()
    status = build_source_field_status(history).set_index("field")
    assert status.loc["pct_chg", "raw_source_available"]
    assert status.loc["pct_chg", "normalize_stock_universe_action"] == "DROPPED"
    assert status.loc["float_market_cap", "normalize_stock_universe_action"] == "PRESERVED"
    assert not status.loc["float_market_cap", "historical_daily_schema_available"]


def test_redundancy_is_only_fixed_four_by_four():
    frame = pd.DataFrame({
        "market_up_ratio": np.linspace(.2, .8, 10),
        "market_median_return": np.linspace(-2, 2, 10),
        "market_tail_balance": np.linspace(-.2, .2, 10),
        "size_style_spread": [np.nan] * 10,
    })
    result = build_redundancy(frame)
    assert len(result) == 16
    assert set(result["variable_1"]) == set(CONTEXT_VARIABLES)
    assert set(result["variable_2"]) == set(CONTEXT_VARIABLES)


def test_real_outputs_are_deterministic_blind_and_complete():
    first, audit = build_outputs(ROOT)
    second, second_audit = build_outputs(ROOT)
    assert first == second
    assert audit == second_audit
    assert tuple(first) == OUTPUT_FILENAMES
    assert audit["output_file_count"] == 9
    assert audit["external_data_fetches"] == 0
    assert audit["label_or_outcome_accessed"] == "NO"
    assert audit["model_trained"] == "NO"
    assert audit["new_feature_search"] == "NO"
    assert audit["m4_full_quality_dates"] == 0
    review = first["v004c_broad_market_context_coverage_review_v001.md"].decode("utf-8")
    assert "LABEL_OR_OUTCOME_ACCESSED = NO" in review
    assert "MODEL_TRAINED = NO" in review
    assert "NEW_FEATURE_SEARCH = NO" in review


def test_module_has_no_external_fetch_or_research_result_input():
    text = (ROOT / "src/v004c_broad_market_context_coverage_audit.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "requests.", "fetch_stock_universe(", "fetch_daily_history(",
        "reports/research/", "GradientBoosting", "RandomForest", "XGBClassifier",
    )
    assert not any(token in text for token in forbidden)


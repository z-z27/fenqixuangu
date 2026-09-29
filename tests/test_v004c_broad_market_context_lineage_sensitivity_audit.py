from __future__ import annotations

from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd

from src.v004c_broad_market_context_lineage_sensitivity_audit import (
    CONTEXT_VARIABLES,
    OUTPUT_FILENAMES,
    UNIVERSE_VARIANTS,
    _context_values,
    build_daily_differences,
    build_membership_difference,
    build_outputs,
    build_state_agreement,
    pools_from_history,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_inputs():
    universe = pd.DataFrame({
        "code": ["000001", "000002", "600001"],
        "name": ["A", "B", "C"],
        "float_market_cap": [1.0, 2.0, 3.0],
        "total_market_cap": [2.0, 3.0, 4.0],
    })
    listing = pd.DataFrame({"code": ["600001"], "listing_date": ["2026-06-01"]})
    rows = []
    for date in ("2026-05-06", "2026-06-02"):
        for code, pct in (
            ("000001", 4.0), ("000002", -4.0), ("600001", 1.0), ("600999", -1.0)
        ):
            rows.append({
                "date": date, "code": code, "pct_chg": pct,
                "open": 1.0, "high": 1.1, "low": .9, "close": 1.0,
                "volume": 1.0, "amount": np.nan, "change": pct,
                "amplitude": 0.0, "turnover_rate": np.nan,
                "source": "synthetic", "cache_kind": "daily_unadjusted",
                "in_current_universe": code in set(universe["code"]),
            })
    return universe, listing, pd.DataFrame(rows)


def test_fixed_manifest_has_only_three_context_variables_and_three_variants():
    assert CONTEXT_VARIABLES == (
        "market_up_ratio", "market_median_return", "market_tail_balance"
    )
    assert UNIVERSE_VARIANTS == (
        "U0_CURRENT_SNAPSHOT_REFERENCE",
        "U1_OBSERVED_DAILY_UNIVERSE",
        "U2_CURRENT_INTERSECTION_STRICT",
    )
    assert len(OUTPUT_FILENAMES) == 11


def test_context_definitions_keep_fixed_three_percent_tail_cutoffs():
    frame = pd.DataFrame({"pct_chg": [4.0, 1.0, -2.0, -5.0]})
    values = _context_values(frame)
    assert values["market_up_ratio"] == .5
    assert values["market_median_return"] == -.5
    assert values["market_tail_balance"] == 0.0


def test_u1_is_observed_pool_and_u0_u2_use_current_snapshot_contract():
    universe, listing, history = _synthetic_inputs()
    pools = pools_from_history(
        universe, listing, history, ["2026-05-06", "2026-06-02"]
    )
    assert set(pools[("2026-05-06", UNIVERSE_VARIANTS[0])]["code"]) == {"000001", "000002"}
    assert set(pools[("2026-05-06", UNIVERSE_VARIANTS[1])]["code"]) == {
        "000001", "000002", "600001", "600999"
    }
    assert set(pools[("2026-05-06", UNIVERSE_VARIANTS[2])]["code"]) == {"000001", "000002"}
    assert set(pools[("2026-06-02", UNIVERSE_VARIANTS[0])]["code"]) == {
        "000001", "000002", "600001"
    }


def test_membership_difference_is_symmetric_union_ratio():
    universe, listing, history = _synthetic_inputs()
    dates = ["2026-05-06"]
    pools = pools_from_history(universe, listing, history, dates)
    counts = pd.DataFrame({"trade_date": dates})
    result = build_membership_difference(counts, pools)
    row = result[
        result["left_variant"].eq(UNIVERSE_VARIANTS[0])
        & result["right_variant"].eq(UNIVERSE_VARIANTS[1])
    ].iloc[0]
    assert row["symmetric_difference_count"] == 2
    assert row["union_count"] == 4
    assert row["membership_difference_ratio"] == .5


def test_daily_difference_units_and_state_thresholds_are_fixed():
    rows = []
    for date, u0, u1, u2 in (
        ("2026-05-06", (.50, 0.10, .02), (.51, 0.12, .01), (.50, 0.10, .02)),
        ("2026-05-07", (.49, -0.10, -.02), (.48, -0.08, -.01), (.49, -0.10, -.02)),
    ):
        for variant, values in zip(UNIVERSE_VARIANTS, (u0, u1, u2)):
            rows.append({
                "trade_date": date, "month": date[:7],
                "analysis_scope": "PRIMARY_MAY_JULY",
                "universe_variant": variant,
                "market_up_ratio": values[0],
                "market_median_return": values[1],
                "market_tail_balance": values[2],
            })
    frame = pd.DataFrame(rows)
    differences = build_daily_differences(frame)
    m1 = differences[
        differences["context_variable"].eq("market_up_ratio")
        & differences["comparison"].str.startswith(UNIVERSE_VARIANTS[1])
    ].iloc[0]
    assert np.isclose(m1["signed_difference_pp"], 1.0)
    agreements = build_state_agreement(frame)
    pooled = agreements[
        agreements["period"].eq("MAY_JULY_POOLED")
        & agreements["context_variable"].eq("market_up_ratio")
        & agreements["left_variant"].eq(UNIVERSE_VARIANTS[0])
        & agreements["right_variant"].eq(UNIVERSE_VARIANTS[1])
    ].iloc[0]
    assert pooled["state_agreement_rate"] == .5


def test_real_outputs_reproduce_u0_and_are_deterministic_and_blind():
    first, audit = build_outputs(ROOT)
    second, second_audit = build_outputs(ROOT)
    assert first == second
    assert audit == second_audit
    assert tuple(first) == OUTPUT_FILENAMES
    assert audit["output_file_count"] == 11
    assert audit["reference_reproduction_pass"]
    assert audit["u0_reference_mismatch_dates"] == 0
    assert audit["u0_reference_max_abs_value_error"] <= 1e-12
    assert audit["u0_reference_max_count_error"] == 0
    assert audit["primary_dates"] == 62
    assert audit["august_auxiliary_dates"] == 13
    assert audit["label_or_outcome_accessed"] == "NO"
    assert audit["model_trained"] == "NO"
    assert audit["external_data_fetched"] == "NO"
    forbidden_columns = {
        "target7", "loss", "capped_return", "capped_return_7",
        "raw_repair_return", "stage1_score", "stage1_rank",
        "model_score", "model_rank",
    }
    for name, payload in first.items():
        if name.endswith(".csv"):
            columns = set(pd.read_csv(BytesIO(payload), nrows=0).columns.str.lower())
            assert not columns.intersection(forbidden_columns)
    review = first[OUTPUT_FILENAMES[-1]].decode("utf-8")
    assert "LABEL_OR_OUTCOME_ACCESSED = NO" in review
    assert "MODEL_TRAINED = NO" in review
    assert "EXTERNAL_DATA_FETCHED = NO" in review


def test_module_has_no_forbidden_data_or_external_fetch_calls():
    source = (ROOT / "src/v004c_broad_market_context_lineage_sensitivity_audit.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = (
        "requests.", "fetch_stock_universe(", "fetch_daily_history(",
        "gradientboosting", "randomforest", "xgbclassifier",
        "board4plus", "size_style_spread",
    )
    assert not any(token in source for token in forbidden)

from __future__ import annotations

import ast
import io
from pathlib import Path

import pandas as pd
import pytest

from src import v004c_broad_market_context_information_audit as audit


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def primary() -> dict:
    return audit.analyze_primary(ROOT)


@pytest.fixture(scope="module")
def full(primary: dict) -> dict:
    # The production function independently rebuilds/fixes primary state before
    # it opens the post-hoc August artifact.
    return audit.analyze(ROOT)


def test_contract_is_exact_and_bounded() -> None:
    assert audit.PRIMARY_CONTEXT_VARIABLE == "market_up_ratio"
    assert audit.CONTEXT_VARIABLES == (
        "market_up_ratio", "market_median_return", "market_tail_balance",
    )
    assert audit.FIXED_STOCK_SIGNALS == (
        "rank_theme_score", "rank_active_money_score", "rank_d1_close_vwap_pct",
    )
    assert audit.MARKET_STATES == ("WEAK_MARKET", "STRONG_MARKET")
    assert len(audit.OUTPUT_FILENAMES) == 10


def test_no_model_fit_call_in_module() -> None:
    tree = ast.parse((ROOT / "src" / "v004c_broad_market_context_information_audit.py").read_text(encoding="utf-8"))
    called = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
    assert not [name for name in called if name == "fit" or name.startswith("fit_")]


def test_primary_population_context_and_threshold_parity(primary: dict) -> None:
    population = primary["population"]
    context = primary["context"]
    primary_context = context[context["analysis_scope"].eq("PRIMARY_MAY_JULY")]
    assert (len(population), population["signal_date"].nunique()) == (485, 60)
    assert len(primary_context) == 62
    assert population["signal_date"].lt("2026-08-01").all()
    assert not population["event_id"].duplicated().any()
    assert primary["audit"]["s2_model_refits"] == 0
    assert primary["threshold"] == pytest.approx(
        primary_context["market_up_ratio"].median(), abs=1e-15
    )
    expected_state = population["market_up_ratio"].map(
        lambda value: "WEAK_MARKET" if value <= primary["threshold"] else "STRONG_MARKET"
    )
    assert expected_state.equals(population["market_state"])


def test_pair_concordance_is_date_equal_with_half_credit_ties() -> None:
    frame = pd.DataFrame([
        {"signal_date": "2026-05-01", "month": "2026-05", "market_state": "WEAK_MARKET", "event_id": "t1", "target7": 1, "loss": 0, "x": .9},
        {"signal_date": "2026-05-01", "month": "2026-05", "market_state": "WEAK_MARKET", "event_id": "t2", "target7": 1, "loss": 0, "x": .1},
        {"signal_date": "2026-05-01", "month": "2026-05", "market_state": "WEAK_MARKET", "event_id": "l1", "target7": 0, "loss": 1, "x": .5},
        {"signal_date": "2026-05-02", "month": "2026-05", "market_state": "STRONG_MARKET", "event_id": "t3", "target7": 1, "loss": 0, "x": .4},
        {"signal_date": "2026-05-02", "month": "2026-05", "market_state": "STRONG_MARKET", "event_id": "l2", "target7": 0, "loss": 1, "x": .2},
    ])
    pairs, daily = audit._pair_daily(frame, "x")
    summary = audit._pair_summary(pairs, daily, "ALL_MAY_JULY", "ALL_MARKET")
    assert summary["pair_count"] == 3
    assert summary["within_date_pair_concordance"] == pytest.approx(.75)
    assert summary["pooled_pair_concordance"] == pytest.approx(2 / 3)


def test_daily_selection_alpha_is_same_date_top3_minus_universe() -> None:
    rows = []
    for rank, capped in enumerate((.07, .04, -.02, -.06), start=1):
        rows.append({
            "signal_date": "2026-05-06", "event_id": f"e{rank}", "s2_rank": rank,
            "board_group": "BOARD2", "target7": int(capped >= .07),
            "loss": int(capped < 0), "severe_loss": int(capped <= -.05),
            "capped_return_7": capped, "market_state": "STRONG_MARKET",
            "market_up_ratio": .6, "market_median_return": .5,
            "market_tail_balance": .1,
        })
    daily = audit._daily_performance(pd.DataFrame(rows)).iloc[0]
    assert daily.top3_capped_return == pytest.approx((.07 + .04 - .02) / 3)
    assert daily.candidate_capped_return == pytest.approx((.07 + .04 - .02 - .06) / 4)
    assert daily.daily_selection_alpha == pytest.approx(.0225)


def test_primary_state_is_fixed_before_august(primary: dict, full: dict) -> None:
    assert primary["state"] == "BROAD_MARKET_CONTEXT_PARTIALLY_SUPPORTED"
    assert primary["next_action"] == "STOP_AND_REVIEW"
    assert full["state"] == primary["state"]
    assert full["next_action"] == primary["next_action"]
    assert full["august_audit"]["august_role"] == "POST_HOC_AUXILIARY_ONLY"
    assert full["august_audit"]["august_context_dates"] == 13
    assert full["august_audit"]["august_used_for_primary_state"] == "NO"


def test_outputs_are_exact_parseable_and_deterministic(full: dict) -> None:
    first = audit.build_outputs(full)
    second = audit.build_outputs(full)
    assert first == second
    assert tuple(first) == audit.OUTPUT_FILENAMES
    assert len(first) == 10
    for name, payload in first.items():
        assert payload
        if name.endswith(".csv"):
            parsed = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
            assert not parsed.columns.duplicated().any()


def test_review_has_required_formal_contract(full: dict) -> None:
    review = full["review"]
    assert "BROAD_MARKET_CONTEXT_INFORMATION_STATE = BROAD_MARKET_CONTEXT_PARTIALLY_SUPPORTED" in review
    assert "PRIMARY_CONTEXT_VARIABLE = MARKET_UP_RATIO" in review
    assert "AUGUST_ROLE = POST_HOC_AUXILIARY_ONLY" in review
    assert "MODEL_TRAINED = NO" in review
    assert "NEW_CONTEXT_VARIABLE_ADDED = NO" in review
    assert "NEXT_ACTION = STOP_AND_REVIEW" in review

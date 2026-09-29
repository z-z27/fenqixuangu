from __future__ import annotations

import csv
from pathlib import Path

from src.v004c_stage1_research_convergence_review import (
    AUGUST_HOLDOUT_STATUS,
    BROADER_D1_INFORMATION_LIMITATION,
    CURRENT_INFORMATION_LIMITATION,
    HIGH_VALUE_UNTESTED_INFORMATION_FAMILY,
    MODEL_FORM_LIMITATION,
    NEXT_ACTION,
    ROUTES,
    STAGE1_RESEARCH_STATE,
    build_outputs,
    verify_source_reviews,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_formal_convergence_state_is_exact() -> None:
    assert MODEL_FORM_LIMITATION == "NOT_SUPPORTED"
    assert CURRENT_INFORMATION_LIMITATION == "SUPPORTED"
    assert BROADER_D1_INFORMATION_LIMITATION == "[待核验]"
    assert HIGH_VALUE_UNTESTED_INFORMATION_FAMILY == "PRE_BREAK_BOARD_DAY_SEAL_PATH"
    assert STAGE1_RESEARCH_STATE == "ONE_NARROW_INFORMATION_AUDIT_JUSTIFIED"
    assert NEXT_ACTION == "BLIND_PRE_BREAK_BOARD_DAY_SEAL_PATH_INFORMATION_AUDIT"
    assert AUGUST_HOLDOUT_STATUS == "CONSUMED"


def test_route_inventory_covers_preregistered_routes_and_extras() -> None:
    names = {row["route_name"] for row in ROUTES}
    required = {
        "A_18F_ARCHITECTURE_TRANSFER",
        "B_REDUCED7F_CONTRIBUTION_ABLATION",
        "C_TRAINING_SPEC_SANITY",
        "D_S2_REFERENCE",
        "E_BOARD2_BOARD3_STRUCTURE",
        "F_D1_RISK_INFORMATION_FOUNDATION",
        "G_PAIRWISE_RIDGE_53F",
        "G_PAIR_CAPPED7_JULY_FORWARD",
        "I_RECENCY_WEIGHTING",
        "J_WINNER_VS_LOSS_ATTRIBUTION",
        "K_53F_D1_INFORMATION_GBDT",
        "L_UNFINISHED_REPAIR_HYPOTHESIS",
        "M_LOCAL_BOARD_BREAK_REGIME",
        "N_LOCAL_REGIME_CONDITIONAL_SIGNAL",
        "O_CLOSE_VWAP_BOARD4PLUS_CONFIRMATION",
        "P_AUGUST_REGIME_AWARE_CHALLENGER",
        "Q_BROAD_MARKET_COVERAGE",
        "R_HISTORICAL_UNIVERSE_LINEAGE_SENSITIVITY",
        "S_BROAD_MARKET_CONTEXT_INFORMATION",
        "REPAIR_STATE_NONLINEAR_INTERACTIONS",
        "CURRENT_18F_TOP2_JULY_FORWARD",
    }
    assert required <= names
    assert len(ROUTES) >= 30


def test_every_route_has_existing_formal_source() -> None:
    assert verify_source_reviews(ROOT) == []
    assert all(row["experiment_done"] in {"YES", "NO"} for row in ROUTES)
    assert all(row["current_authorization"] for row in ROUTES)


def test_outputs_have_exact_manifest_and_parse() -> None:
    paths = build_outputs(ROOT)
    assert set(paths) == {
        "routes",
        "findings",
        "do_not_repeat",
        "families",
        "untested",
        "bottleneck",
        "review",
    }
    for name, path in paths.items():
        assert path.is_file() and path.stat().st_size > 0, name
    assert len(_read(paths["routes"])) == len(ROUTES)
    assert len(_read(paths["untested"])) == 1


def test_only_one_untested_family_is_authorized() -> None:
    out = build_outputs(ROOT)
    rows = _read(out["untested"])
    assert [row["family"] for row in rows] == ["PRE_BREAK_BOARD_DAY_SEAL_PATH"]
    assert rows[0]["eligible"] == "YES"
    assert rows[0]["authorization_boundary"].startswith("只授权blind")


def test_review_contains_required_plain_language_and_final_fields() -> None:
    path = build_outputs(ROOT)["review"]
    text = path.read_text(encoding="utf-8")
    assert text.index("## 简单版结论") < text.index("## 1. Frozen Goal and Pipeline")
    assert "FACTOR_ANALYSIS_4_STATE" in text
    assert "AUGUST_HOLDOUT_STATUS = CONSUMED" in text
    assert "MODEL_FORM_LIMITATION = NOT_SUPPORTED" in text
    assert "CURRENT_INFORMATION_LIMITATION = SUPPORTED" in text
    assert "BROADER_D1_INFORMATION_LIMITATION = [待核验]" in text
    assert "STAGE1_RESEARCH_STATE = ONE_NARROW_INFORMATION_AUDIT_JUSTIFIED" in text
    assert "NEXT_ACTION = BLIND_PRE_BREAK_BOARD_DAY_SEAL_PATH_INFORMATION_AUDIT" in text


def test_no_forbidden_model_or_feature_search_language_in_authorization() -> None:
    text = build_outputs(ROOT)["review"].read_text(encoding="utf-8")
    assert "不授权任何模型" in text
    assert "禁止模型、阈值、窗口搜索和自动字段生成" in text
    assert "若该family也不成立，应停止当前Stage1 D1因子挖掘路线" in text

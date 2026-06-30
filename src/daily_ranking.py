from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .ranking_backtest import score_candidates, validate_ranking_model
from .signal_engine import Signal


DEFAULT_DAILY_RANKING_MODEL = Path("reports/manual_models/ranking_model_v002_core_momentum_support.json")
DEFAULT_DAILY_TOP_N = 3


def apply_daily_research_ranking(
    signals: list[Signal] | pd.DataFrame,
    model_file: str | Path = DEFAULT_DAILY_RANKING_MODEL,
    top_n: int = DEFAULT_DAILY_TOP_N,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score daily signal output with the selected research ranking model.

    This is intentionally a daily-output layer. Historical sample generation
    keeps writing raw Signal rows and does not call this function.
    """
    model_path = Path(model_file)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model_id = str(model.get("model_id", model_path.stem))
    score_column = str(model.get("score_column", "research_score"))
    meta = {"model_id": model_id, "model_path": str(model_path), "top_n": int(top_n)}

    frame = _signals_to_frame(signals)
    frame["ranking_model_id"] = model_id
    frame["model_topn"] = int(top_n)
    frame["research_score"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame["daily_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if frame.empty:
        return frame, meta

    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    validate_ranking_model(model, frame.columns)
    scored = score_candidates(frame, model)
    if score_column not in scored.columns:
        raise RuntimeError(f"ranking model did not produce score column: {score_column}")
    scored["research_score"] = pd.to_numeric(scored[score_column], errors="coerce")
    scored["ranking_model_id"] = model_id
    scored["model_topn"] = int(top_n)
    scored["daily_rank"] = _daily_rank_series(scored)
    return scored, meta


def _daily_rank_series(frame: pd.DataFrame) -> pd.Series:
    ranks = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    allowed = _bool_series(frame.get("allowed", False), frame.index)
    signal_type = frame.get("signal_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    eligible_mask = allowed & signal_type.eq("D2_LOW_ABSORB")
    eligible = frame[eligible_mask].copy()
    if eligible.empty:
        return ranks

    if "trade_date" not in eligible.columns:
        eligible["trade_date"] = ""
    if "graph_quality_score" not in eligible.columns:
        eligible["graph_quality_score"] = 0.0
    if "code" not in eligible.columns:
        eligible["code"] = ""
    eligible["__research_score"] = pd.to_numeric(eligible["research_score"], errors="coerce").fillna(float("-inf"))
    eligible["__graph_quality_score"] = pd.to_numeric(eligible["graph_quality_score"], errors="coerce").fillna(0.0)
    eligible["__rank_trade_date"] = eligible["trade_date"].fillna("").astype(str)
    eligible["__code"] = eligible["code"].fillna("").astype(str)
    eligible = eligible.sort_values(
        ["__rank_trade_date", "__research_score", "__graph_quality_score", "__code"],
        ascending=[True, False, False, True],
    )
    ranks.loc[eligible.index] = (eligible.groupby("__rank_trade_date").cumcount() + 1).astype("int64").to_numpy()
    return ranks


def _signals_to_frame(signals: list[Signal] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(signals, pd.DataFrame):
        return signals.copy()
    return pd.DataFrame([signal.to_dict() for signal in signals])


def _bool_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        return value.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return pd.Series(bool(value), index=index)

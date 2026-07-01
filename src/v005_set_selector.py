from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_FILE = Path("reports/v004a/grid_v2_scored/v004a_scored_candidates.csv")
DEFAULT_OUTPUT_DIR = Path("reports/v005_set_selector")
DEFAULT_TOP_N = 3
DEFAULT_CANDIDATE_TOP_K = 15
DEFAULT_V004A_L2 = 0.30
DEFAULT_V004A_POSITIVE_WEIGHT = 1.5

TARGET_COLUMN = "target7_d2open_d3high"
HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
REALIZED_RETURN_COLUMN = "realized_return_pct"
CLOSE_RETURN_COLUMN = "d2open_d3close_return_pct"

V004A_MODEL_ID = "logistic_v004a_weighted"
V002_MODEL_ID = "ranking_model_v002_core_momentum_support"
SCOPE = "walk_forward"

# Keep the first v005 grid deliberately small. The goal is set-level attribution
# and low-degree rule discovery, not another high-variance ML search.
DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID = "0.5,1.0,1.5"
DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID = "0.0,0.5"
DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID = "0.0,0.05"
DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID = "0.0,0.05"
DEFAULT_EXTREME_VWAP_PENALTY_GRID = "0.05,0.10,0.15"
DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID = "0.05,0.10,0.15"
DEFAULT_EXTREME_PRICE_PENALTY_GRID = "0.0,0.05"
DEFAULT_RANK_DISPERSION_WEIGHT_GRID = "0.0,0.02"

NUMERIC_COLUMNS = [
    "l2",
    "positive_weight",
    "model_score",
    "model_rank",
    HIGH_RETURN_COLUMN,
    CLOSE_RETURN_COLUMN,
    REALIZED_RETURN_COLUMN,
    "candidate_base_price",
    "rank_log_candidate_base_price",
    "rank_d1_close_vwap_pct",
    "inter_close_low",
    "rank_total_score",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_active_money_score",
    "graph_quality_score",
]

COMBO_FEATURE_COLUMNS = [
    "signal_date",
    "codes",
    "hit_count",
    "all_hit",
    "avg_high_return",
    "avg_realized_return",
    "avg_v004a_rank",
    "min_v004a_rank",
    "max_v004a_rank",
    "avg_v002_rank",
    "min_v002_rank",
    "max_v002_rank",
    "rank_dispersion_v004a",
    "rank_dispersion_v002",
    "rank_dispersion_norm",
    "contains_v004a_top3",
    "contains_v002_top3",
    "all_in_v004a_top10",
    "all_in_v002_top10",
    "extreme_price_count",
    "extreme_vwap_count",
    "extreme_close_low_count",
    "avg_total_rank",
    "min_total_rank",
    "avg_price_rank",
    "max_price_rank",
    "avg_vwap_rank",
    "max_vwap_rank",
    "avg_close_low",
    "max_close_low",
]

GRID_PARAM_COLUMNS = [
    "grid_id",
    "min_total_rank_weight",
    "avg_total_rank_weight",
    "contains_v004a_top3_bonus",
    "contains_v002_top3_bonus",
    "extreme_vwap_penalty",
    "extreme_close_low_penalty",
    "extreme_price_penalty",
    "rank_dispersion_weight",
]

SUMMARY_COLUMNS = [
    *GRID_PARAM_COLUMNS,
    "date_count",
    "selected_ticket_count",
    "top3_target_rate",
    "top3_all_hit_rate",
    "hit_count_0_days",
    "hit_count_1_days",
    "hit_count_2_days",
    "hit_count_3_days",
    "rank1_hit_rate",
    "rank2_hit_rate",
    "rank3_hit_rate",
    "avg_top3_high_return",
    "avg_top3_realized_return",
    "late10_date_count",
    "late10_top3_target_rate",
    "late10_top3_all_hit_rate",
    "late10_hit_count_0_days",
    "late10_hit_count_3_days",
    "late10_avg_top3_realized_return",
]


def run_v005_set_selector(
    scored_file: str | Path = DEFAULT_SCORED_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
    min_total_rank_weight_grid: str = DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
    avg_total_rank_weight_grid: str = DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
    contains_v004a_top3_bonus_grid: str = DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
    contains_v002_top3_bonus_grid: str = DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
    extreme_vwap_penalty_grid: str = DEFAULT_EXTREME_VWAP_PENALTY_GRID,
    extreme_close_low_penalty_grid: str = DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
    extreme_price_penalty_grid: str = DEFAULT_EXTREME_PRICE_PENALTY_GRID,
    rank_dispersion_weight_grid: str = DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    scored_path = Path(scored_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = prepare_scored_candidates(scored_path)
    candidate_pool = build_candidate_pool(
        scored,
        candidate_top_k=int(candidate_top_k),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    combo_candidates = build_combo_candidates(candidate_pool, top_n=int(top_n))
    grid = build_rule_grid(
        min_total_rank_weight_grid=min_total_rank_weight_grid,
        avg_total_rank_weight_grid=avg_total_rank_weight_grid,
        contains_v004a_top3_bonus_grid=contains_v004a_top3_bonus_grid,
        contains_v002_top3_bonus_grid=contains_v002_top3_bonus_grid,
        extreme_vwap_penalty_grid=extreme_vwap_penalty_grid,
        extreme_close_low_penalty_grid=extreme_close_low_penalty_grid,
        extreme_price_penalty_grid=extreme_price_penalty_grid,
        rank_dispersion_weight_grid=rank_dispersion_weight_grid,
    )
    selected_combos, grid_summary = evaluate_rule_grid(combo_candidates, candidate_pool, grid, top_n=int(top_n))
    best_grid_id = select_best_grid_id(grid_summary)
    daily_top3 = build_daily_top3_for_grid(selected_combos, candidate_pool, best_grid_id=best_grid_id, top_n=int(top_n))

    combo_candidates_csv = out_dir / "v005_combo_candidates.csv"
    selected_combos_csv = out_dir / "v005_selected_combos.csv"
    grid_summary_csv = out_dir / "v005_grid_summary.csv"
    daily_top3_csv = out_dir / "v005_daily_top3.csv"
    report_path = out_dir / "v005_report.md"

    combo_candidates.to_csv(combo_candidates_csv, index=False, encoding="utf-8-sig")
    selected_combos.to_csv(selected_combos_csv, index=False, encoding="utf-8-sig")
    grid_summary.to_csv(grid_summary_csv, index=False, encoding="utf-8-sig")
    daily_top3.to_csv(daily_top3_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            scored_path=scored_path,
            output_dir=out_dir,
            top_n=int(top_n),
            candidate_top_k=int(candidate_top_k),
            v004a_l2=float(v004a_l2),
            v004a_positive_weight=float(v004a_positive_weight),
            grid=grid,
            combo_candidates=combo_candidates,
            selected_combos=selected_combos,
            grid_summary=grid_summary,
            daily_top3=daily_top3,
            best_grid_id=best_grid_id,
        ),
        encoding="utf-8",
    )
    return grid_summary, daily_top3, selected_combos, combo_candidates, report_path


def prepare_scored_candidates(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"code": str})
    required = ["model_id", "evaluation_scope", "signal_date", "code", "model_score", "model_rank", TARGET_COLUMN, HIGH_RETURN_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"v005 input missing required columns: {missing}")
    frame = raw.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)
    frame["evaluation_scope"] = frame["evaluation_scope"].astype(str)
    frame[TARGET_COLUMN] = _bool_series(frame[TARGET_COLUMN])
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    return frame


def build_candidate_pool(scored: pd.DataFrame, candidate_top_k: int, v004a_l2: float, v004a_positive_weight: float) -> pd.DataFrame:
    v004a_mask = (
        (scored["model_id"] == V004A_MODEL_ID)
        & (scored["evaluation_scope"] == SCOPE)
        & (pd.to_numeric(scored["l2"], errors="coerce").sub(float(v004a_l2)).abs() <= 1e-9)
        & (pd.to_numeric(scored["positive_weight"], errors="coerce").sub(float(v004a_positive_weight)).abs() <= 1e-9)
    )
    v004a = scored[v004a_mask].copy()
    if v004a.empty:
        raise RuntimeError(f"no v004a rows found for l2={v004a_l2:g}, positive_weight={v004a_positive_weight:g}")
    v004a = v004a.rename(columns={"model_score": "v004a_score", "model_rank": "v004a_model_rank"})
    v004a["v004a_model_rank"] = pd.to_numeric(v004a["v004a_model_rank"], errors="coerce")
    v004a_top = v004a[v004a["v004a_model_rank"] <= int(candidate_top_k)].copy()
    if v004a_top.empty:
        raise RuntimeError(f"no v004a Top{candidate_top_k} rows found")

    v002 = scored[(scored["model_id"] == V002_MODEL_ID) & (scored["evaluation_scope"] == SCOPE)].copy()
    if v002.empty:
        raise RuntimeError(f"no v002 rows found for model_id={V002_MODEL_ID}, scope={SCOPE}")
    v002 = v002[["signal_date", "code", "model_score", "model_rank"]].rename(
        columns={"model_score": "v002_score", "model_rank": "v002_model_rank"}
    )

    pool = v004a_top.merge(v002, on=["signal_date", "code"], how="left")
    pool["v002_model_rank"] = pd.to_numeric(pool["v002_model_rank"], errors="coerce").fillna(999999.0)
    pool["v002_score"] = pd.to_numeric(pool["v002_score"], errors="coerce")
    pool["v004a_model_rank"] = pd.to_numeric(pool["v004a_model_rank"], errors="coerce")
    pool["v004a_rank_tranche"] = pool["v004a_model_rank"].map(_rank_tranche)
    pool["v002_rank_tranche"] = pool["v002_model_rank"].map(_rank_tranche)
    pool["extreme_price"] = pd.to_numeric(pool["rank_log_candidate_base_price"], errors="coerce") >= 0.85
    pool["extreme_vwap"] = pd.to_numeric(pool["rank_d1_close_vwap_pct"], errors="coerce") >= 0.85
    pool["extreme_close_low"] = pd.to_numeric(pool["inter_close_low"], errors="coerce") >= 0.90
    pool["near_miss_5_7"] = (~pool[TARGET_COLUMN].astype(bool)) & pd.to_numeric(pool[HIGH_RETURN_COLUMN], errors="coerce").between(5.0, 7.0, inclusive="left")
    pool = pool.sort_values(["signal_date", "v004a_model_rank", "code"]).reset_index(drop=True)
    return pool


def build_combo_candidates(candidate_pool: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, group in candidate_pool.groupby("signal_date", dropna=False):
        group = group.sort_values(["v004a_model_rank", "code"]).reset_index(drop=True)
        if len(group) < int(top_n):
            continue
        for combo_index, indexes in enumerate(itertools.combinations(range(len(group)), int(top_n)), start=1):
            combo = group.iloc[list(indexes)].copy()
            rows.append(_combo_row(str(signal_date), int(combo_index), combo, top_n=int(top_n)))
    if not rows:
        raise RuntimeError("no v005 combo candidates were generated")
    combo_frame = pd.DataFrame(rows)
    for column in COMBO_FEATURE_COLUMNS:
        if column not in combo_frame.columns:
            combo_frame[column] = np.nan
    return combo_frame[COMBO_FEATURE_COLUMNS]


def _combo_row(signal_date: str, combo_index: int, combo: pd.DataFrame, top_n: int) -> dict[str, Any]:
    v004a_rank = pd.to_numeric(combo["v004a_model_rank"], errors="coerce")
    v002_rank = pd.to_numeric(combo["v002_model_rank"], errors="coerce")
    rank_dispersion_v004a = _max(v004a_rank) - _min(v004a_rank)
    rank_dispersion_v002 = _max(v002_rank) - _min(v002_rank)
    hit_count = int(combo[TARGET_COLUMN].astype(bool).sum())
    return {
        "signal_date": signal_date,
        "combo_index": int(combo_index),
        "codes": ",".join(combo["code"].astype(str).tolist()),
        "hit_count": hit_count,
        "all_hit": bool(hit_count == int(top_n)),
        "avg_high_return": _mean(combo[HIGH_RETURN_COLUMN]),
        "avg_realized_return": _mean(combo[REALIZED_RETURN_COLUMN]),
        "avg_v004a_rank": _mean(v004a_rank),
        "min_v004a_rank": _min(v004a_rank),
        "max_v004a_rank": _max(v004a_rank),
        "avg_v002_rank": _mean(v002_rank),
        "min_v002_rank": _min(v002_rank),
        "max_v002_rank": _max(v002_rank),
        "rank_dispersion_v004a": rank_dispersion_v004a,
        "rank_dispersion_v002": rank_dispersion_v002,
        "rank_dispersion_norm": rank_dispersion_v004a / max(float(top_n), 1.0),
        "contains_v004a_top3": bool((v004a_rank <= 3).any()),
        "contains_v002_top3": bool((v002_rank <= 3).any()),
        "all_in_v004a_top10": bool((v004a_rank <= 10).all()),
        "all_in_v002_top10": bool((v002_rank <= 10).all()),
        "extreme_price_count": int(combo["extreme_price"].astype(bool).sum()),
        "extreme_vwap_count": int(combo["extreme_vwap"].astype(bool).sum()),
        "extreme_close_low_count": int(combo["extreme_close_low"].astype(bool).sum()),
        "avg_total_rank": _mean(combo["rank_total_score"]),
        "min_total_rank": _min(combo["rank_total_score"]),
        "avg_price_rank": _mean(combo["rank_log_candidate_base_price"]),
        "max_price_rank": _max(combo["rank_log_candidate_base_price"]),
        "avg_vwap_rank": _mean(combo["rank_d1_close_vwap_pct"]),
        "max_vwap_rank": _max(combo["rank_d1_close_vwap_pct"]),
        "avg_close_low": _mean(combo["inter_close_low"]),
        "max_close_low": _max(combo["inter_close_low"]),
    }


def build_rule_grid(
    min_total_rank_weight_grid: str,
    avg_total_rank_weight_grid: str,
    contains_v004a_top3_bonus_grid: str,
    contains_v002_top3_bonus_grid: str,
    extreme_vwap_penalty_grid: str,
    extreme_close_low_penalty_grid: str,
    extreme_price_penalty_grid: str,
    rank_dispersion_weight_grid: str,
) -> pd.DataFrame:
    grids = [
        _parse_float_grid(min_total_rank_weight_grid),
        _parse_float_grid(avg_total_rank_weight_grid),
        _parse_float_grid(contains_v004a_top3_bonus_grid),
        _parse_float_grid(contains_v002_top3_bonus_grid),
        _parse_float_grid(extreme_vwap_penalty_grid),
        _parse_float_grid(extreme_close_low_penalty_grid),
        _parse_float_grid(extreme_price_penalty_grid),
        _parse_float_grid(rank_dispersion_weight_grid),
    ]
    rows: list[dict[str, Any]] = []
    for grid_id, values in enumerate(itertools.product(*grids), start=1):
        rows.append(
            {
                "grid_id": int(grid_id),
                "min_total_rank_weight": float(values[0]),
                "avg_total_rank_weight": float(values[1]),
                "contains_v004a_top3_bonus": float(values[2]),
                "contains_v002_top3_bonus": float(values[3]),
                "extreme_vwap_penalty": float(values[4]),
                "extreme_close_low_penalty": float(values[5]),
                "extreme_price_penalty": float(values[6]),
                "rank_dispersion_weight": float(values[7]),
            }
        )
    if not rows:
        raise RuntimeError("empty v005 rule grid")
    return pd.DataFrame(rows)


def evaluate_rule_grid(combo_candidates: pd.DataFrame, candidate_pool: pd.DataFrame, grid: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    late10_dates = sorted(combo_candidates["signal_date"].dropna().astype(str).unique().tolist())[-10:]
    for _, params in grid.iterrows():
        scored = score_combos(combo_candidates, params)
        selected = select_best_combo_by_date(scored)
        selected_rows.append(selected)
        summary_rows.append(summarize_selected_grid(selected, candidate_pool, params, late10_dates=late10_dates, top_n=int(top_n)))
    selected_combos = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    grid_summary = pd.DataFrame(summary_rows)
    grid_summary = grid_summary.sort_values(
        [
            "top3_all_hit_rate",
            "hit_count_0_days",
            "top3_target_rate",
            "avg_top3_realized_return",
            "late10_top3_all_hit_rate",
            "late10_avg_top3_realized_return",
        ],
        ascending=[False, True, False, False, False, False],
    ).reset_index(drop=True)
    return selected_combos, grid_summary[SUMMARY_COLUMNS]


def score_combos(combo_candidates: pd.DataFrame, params: pd.Series) -> pd.DataFrame:
    scored = combo_candidates.copy()
    scored["grid_id"] = int(params["grid_id"])
    for column in GRID_PARAM_COLUMNS:
        if column == "grid_id":
            continue
        scored[column] = float(params[column])
    scored["combo_score"] = (
        float(params["min_total_rank_weight"]) * pd.to_numeric(scored["min_total_rank"], errors="coerce").fillna(0.0)
        + float(params["avg_total_rank_weight"]) * pd.to_numeric(scored["avg_total_rank"], errors="coerce").fillna(0.0)
        + float(params["contains_v004a_top3_bonus"]) * scored["contains_v004a_top3"].astype(float)
        + float(params["contains_v002_top3_bonus"]) * scored["contains_v002_top3"].astype(float)
        + float(params["rank_dispersion_weight"]) * pd.to_numeric(scored["rank_dispersion_norm"], errors="coerce").fillna(0.0)
        - float(params["extreme_vwap_penalty"]) * pd.to_numeric(scored["extreme_vwap_count"], errors="coerce").fillna(0.0)
        - float(params["extreme_close_low_penalty"]) * pd.to_numeric(scored["extreme_close_low_count"], errors="coerce").fillna(0.0)
        - float(params["extreme_price_penalty"]) * pd.to_numeric(scored["extreme_price_count"], errors="coerce").fillna(0.0)
    )
    return scored


def select_best_combo_by_date(scored: pd.DataFrame) -> pd.DataFrame:
    selected = (
        scored.sort_values(
            ["signal_date", "combo_score", "min_total_rank", "avg_total_rank", "avg_v004a_rank", "codes"],
            ascending=[True, False, False, False, True, True],
        )
        .groupby("signal_date", as_index=False, dropna=False)
        .head(1)
        .copy()
    )
    return selected.reset_index(drop=True)


def summarize_selected_grid(selected: pd.DataFrame, candidate_pool: pd.DataFrame, params: pd.Series, late10_dates: list[str], top_n: int) -> dict[str, Any]:
    exploded = explode_selected_combos(selected, candidate_pool, top_n=int(top_n))
    daily = selected.copy()
    row: dict[str, Any] = {column: params[column] for column in GRID_PARAM_COLUMNS}
    row.update(_summary_metrics(daily, exploded, prefix=""))
    late_daily = daily[daily["signal_date"].astype(str).isin(late10_dates)].copy()
    late_exploded = exploded[exploded["signal_date"].astype(str).isin(late10_dates)].copy()
    row.update(_summary_metrics(late_daily, late_exploded, prefix="late10_"))
    return row


def _summary_metrics(daily: pd.DataFrame, exploded: pd.DataFrame, prefix: str) -> dict[str, Any]:
    targets = exploded[TARGET_COLUMN].astype(bool) if not exploded.empty else pd.Series(dtype=bool)
    hit_count = pd.to_numeric(daily["hit_count"], errors="coerce").fillna(0) if not daily.empty else pd.Series(dtype=float)
    result = {
        f"{prefix}date_count": int(daily["signal_date"].nunique()) if not daily.empty else 0,
        f"{prefix}selected_ticket_count": int(len(exploded)),
        f"{prefix}top3_target_rate": _safe_rate(int(targets.sum()), int(len(targets))),
        f"{prefix}top3_all_hit_rate": _safe_rate(int(daily["all_hit"].astype(bool).sum()), int(len(daily))) if not daily.empty else np.nan,
        f"{prefix}hit_count_0_days": int((hit_count == 0).sum()),
        f"{prefix}hit_count_1_days": int((hit_count == 1).sum()),
        f"{prefix}hit_count_2_days": int((hit_count == 2).sum()),
        f"{prefix}hit_count_3_days": int((hit_count == 3).sum()),
        f"{prefix}avg_top3_high_return": _mean(daily["avg_high_return"]) if not daily.empty else np.nan,
        f"{prefix}avg_top3_realized_return": _mean(daily["avg_realized_return"]) if not daily.empty else np.nan,
    }
    if prefix == "":
        for rank in range(1, 4):
            rank_rows = exploded[pd.to_numeric(exploded["daily_rank"], errors="coerce") == rank] if not exploded.empty else pd.DataFrame()
            result[f"rank{rank}_hit_rate"] = _safe_rate(int(rank_rows[TARGET_COLUMN].astype(bool).sum()), int(len(rank_rows))) if not rank_rows.empty else np.nan
    return result


def explode_selected_combos(selected: pd.DataFrame, candidate_pool: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    key_pool = candidate_pool.copy()
    for _, combo in selected.iterrows():
        codes = [code.strip().zfill(6) for code in str(combo["codes"]).split(",") if code.strip()]
        if not codes:
            continue
        date_pool = key_pool[(key_pool["signal_date"].astype(str) == str(combo["signal_date"])) & (key_pool["code"].astype(str).isin(codes))].copy()
        date_pool["grid_id"] = int(combo["grid_id"])
        date_pool["combo_score"] = float(combo["combo_score"])
        date_pool = date_pool.sort_values(["v004a_model_rank", "v002_model_rank", "code"]).head(int(top_n)).copy()
        date_pool["daily_rank"] = np.arange(1, len(date_pool) + 1)
        rows.append(date_pool)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    keep = [
        "grid_id",
        "signal_date",
        "daily_rank",
        "code",
        TARGET_COLUMN,
        HIGH_RETURN_COLUMN,
        REALIZED_RETURN_COLUMN,
        "combo_score",
        "v004a_model_rank",
        "v002_model_rank",
        "v004a_score",
        "v002_score",
        "rank_total_score",
        "rank_log_candidate_base_price",
        "rank_d1_close_vwap_pct",
        "inter_close_low",
        "extreme_price",
        "extreme_vwap",
        "extreme_close_low",
        "near_miss_5_7",
    ]
    return result[[column for column in keep if column in result.columns]]


def build_daily_top3_for_grid(selected_combos: pd.DataFrame, candidate_pool: pd.DataFrame, best_grid_id: int, top_n: int) -> pd.DataFrame:
    selected = selected_combos[pd.to_numeric(selected_combos["grid_id"], errors="coerce") == int(best_grid_id)].copy()
    exploded = explode_selected_combos(selected, candidate_pool, top_n=int(top_n))
    if exploded.empty:
        return pd.DataFrame()
    return exploded.sort_values(["signal_date", "daily_rank", "code"]).reset_index(drop=True)


def select_best_grid_id(grid_summary: pd.DataFrame) -> int:
    if grid_summary.empty:
        raise RuntimeError("empty v005 grid summary")
    return int(grid_summary.iloc[0]["grid_id"])


def build_report(
    scored_path: Path,
    output_dir: Path,
    top_n: int,
    candidate_top_k: int,
    v004a_l2: float,
    v004a_positive_weight: float,
    grid: pd.DataFrame,
    combo_candidates: pd.DataFrame,
    selected_combos: pd.DataFrame,
    grid_summary: pd.DataFrame,
    daily_top3: pd.DataFrame,
    best_grid_id: int,
) -> str:
    best_summary = grid_summary[grid_summary["grid_id"] == int(best_grid_id)].copy()
    best_selected = selected_combos[selected_combos["grid_id"] == int(best_grid_id)].copy()
    lines = [
        "# v005 set-level Top3 selector research",
        "",
        "## Scope",
        "",
        "v005 is research-only. It does not train an ML model, does not write a ranking_model JSON, and does not connect to run-daily.",
        "It treats v004a TopK as the candidate pool, enumerates Top3 combinations, and evaluates low-degree set-level scoring rules.",
        "The grid selection is in-sample research attribution, not a production model selection procedure.",
        "",
        "## Configuration",
        "",
        f"- scored candidates file: `{scored_path}`",
        f"- output dir: `{output_dir}`",
        f"- top_n: `{top_n}`",
        f"- candidate_top_k: `{candidate_top_k}`",
        f"- v004a l2 / positive_weight: `{v004a_l2:g}` / `{v004a_positive_weight:g}`",
        f"- grid size: `{len(grid)}`",
        f"- combo candidates: `{len(combo_candidates)}`",
        f"- selected combo rows across grid: `{len(selected_combos)}`",
        f"- best grid id: `{best_grid_id}`",
        "",
        "## Best grid summary",
        "",
    ]
    lines.extend(_markdown_table(best_summary, SUMMARY_COLUMNS))
    lines.extend(["", "## Top 20 grid summary", ""])
    lines.extend(_markdown_table(grid_summary.head(20), SUMMARY_COLUMNS))
    lines.extend(["", "## Best selected combos", ""])
    lines.extend(
        _markdown_table(
            best_selected,
            [
                "grid_id",
                "signal_date",
                "codes",
                "combo_score",
                "hit_count",
                "all_hit",
                "avg_high_return",
                "avg_realized_return",
                "avg_v004a_rank",
                "avg_v002_rank",
                "extreme_price_count",
                "extreme_vwap_count",
                "extreme_close_low_count",
                "min_total_rank",
                "avg_total_rank",
            ],
        )
    )
    lines.extend(["", "## Best daily Top3 rows", ""])
    lines.extend(
        _markdown_table(
            daily_top3,
            [
                "signal_date",
                "daily_rank",
                "code",
                TARGET_COLUMN,
                HIGH_RETURN_COLUMN,
                REALIZED_RETURN_COLUMN,
                "v004a_model_rank",
                "v002_model_rank",
                "rank_total_score",
                "rank_log_candidate_base_price",
                "rank_d1_close_vwap_pct",
                "inter_close_low",
                "extreme_price",
                "extreme_vwap",
                "extreme_close_low",
                "near_miss_5_7",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Research interpretation checklist",
            "",
            "- Treat a high in-sample all-hit rate as a hypothesis, not proof of deployable edge.",
            "- Compare full-window and late10 metrics; a rule that only wins full-window but fails late10 is likely unstable.",
            "- Inspect whether selected combos reduce excessive extreme VWAP / close-low exposure without destroying rank1 quality.",
            "- If set-level rules beat v002 and v004a on all-hit while keeping 0-hit days low, the next step is walk-forward set-level validation.",
        ]
    )
    return "\n".join(lines)


def _parse_float_grid(text: str) -> list[float]:
    values = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise RuntimeError(f"empty float grid: {text!r}")
    return values


def _rank_tranche(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "missing"
    if number <= 1:
        return "rank_1"
    if number <= 3:
        return "rank_2_3"
    if number <= 5:
        return "rank_4_5"
    if number <= 10:
        return "rank_6_10"
    if number <= 15:
        return "rank_11_15"
    if number <= 30:
        return "rank_16_30"
    return "rank_31_plus"


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def _safe_rate(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def _mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return np.nan
    return float(series.mean())


def _min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return np.nan
    return float(series.min())


def _max(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return np.nan
    return float(series.max())


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame is None or frame.empty:
        return ["_No rows._"]
    usable = [column for column in columns if column in frame.columns]
    if not usable:
        return ["_No columns._"]
    table = frame[usable].copy()
    for column in table.columns:
        table[column] = table[column].map(_format_markdown_value)
    lines = [
        "| " + " | ".join(usable) + " |",
        "| " + " | ".join(["---"] * len(usable)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in usable) + " |")
    return lines


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    return text.replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run research-only v005 set-level Top3 selector grid.")
    parser.add_argument("--scored-file", default=str(DEFAULT_SCORED_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    parser.add_argument("--min-total-rank-weight-grid", default=DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID)
    parser.add_argument("--avg-total-rank-weight-grid", default=DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID)
    parser.add_argument("--contains-v004a-top3-bonus-grid", default=DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID)
    parser.add_argument("--contains-v002-top3-bonus-grid", default=DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID)
    parser.add_argument("--extreme-vwap-penalty-grid", default=DEFAULT_EXTREME_VWAP_PENALTY_GRID)
    parser.add_argument("--extreme-close-low-penalty-grid", default=DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID)
    parser.add_argument("--extreme-price-penalty-grid", default=DEFAULT_EXTREME_PRICE_PENALTY_GRID)
    parser.add_argument("--rank-dispersion-weight-grid", default=DEFAULT_RANK_DISPERSION_WEIGHT_GRID)
    args = parser.parse_args(argv)

    grid_summary, daily_top3, selected_combos, combo_candidates, report_path = run_v005_set_selector(
        scored_file=args.scored_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        candidate_top_k=args.candidate_top_k,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
        min_total_rank_weight_grid=args.min_total_rank_weight_grid,
        avg_total_rank_weight_grid=args.avg_total_rank_weight_grid,
        contains_v004a_top3_bonus_grid=args.contains_v004a_top3_bonus_grid,
        contains_v002_top3_bonus_grid=args.contains_v002_top3_bonus_grid,
        extreme_vwap_penalty_grid=args.extreme_vwap_penalty_grid,
        extreme_close_low_penalty_grid=args.extreme_close_low_penalty_grid,
        extreme_price_penalty_grid=args.extreme_price_penalty_grid,
        rank_dispersion_weight_grid=args.rank_dispersion_weight_grid,
    )
    best_grid_id = int(grid_summary.iloc[0]["grid_id"]) if not grid_summary.empty else -1
    print(f"grid rows: {len(grid_summary)}")
    print(f"combo candidate rows: {len(combo_candidates)}")
    print(f"selected combo rows: {len(selected_combos)}")
    print(f"best daily top3 rows: {len(daily_top3)}")
    print(f"best grid id: {best_grid_id}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

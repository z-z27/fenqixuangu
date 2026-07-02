from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_FILE = Path("reports/v004a/grid_v2_scored/v004a_scored_candidates.csv")
DEFAULT_V005_DIR = Path("reports/v005_set_selector")
DEFAULT_OUTPUT_DIR = Path("reports/v005_failure_attribution")
DEFAULT_OBJECTIVE = "realized_then_all_hit"
DEFAULT_TOP_N = 3
DEFAULT_V004A_L2 = 0.30
DEFAULT_V004A_POSITIVE_WEIGHT = 1.5
DEFAULT_REALIZED_GAP_THRESHOLD = 0.50

TARGET_COLUMN = "target7_d2open_d3high"
HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
REALIZED_RETURN_COLUMN = "realized_return_pct"
V004A_MODEL_ID = "logistic_v004a_weighted"
V002_MODEL_ID = "ranking_model_v002_core_momentum_support"
SCOPE = "walk_forward"

NUMERIC_COLUMNS = [
    "l2",
    "positive_weight",
    "model_score",
    "model_rank",
    HIGH_RETURN_COLUMN,
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

CONTEXT_FEATURE_COLUMNS = [
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

FAILURE_DATE_COLUMNS = [
    "signal_date",
    "objective",
    "selected_grid_id",
    "failure_reason",
    "v005_codes",
    "v005_hit_count",
    "v005_all_hit",
    "v005_avg_high_return",
    "v005_avg_realized_return",
    "v002_codes",
    "v002_hit_count",
    "v002_all_hit",
    "v002_avg_high_return",
    "v002_avg_realized_return",
    "v004a_codes",
    "v004a_hit_count",
    "v004a_all_hit",
    "v004a_avg_high_return",
    "v004a_avg_realized_return",
    "hit_delta_vs_v002",
    "realized_delta_vs_v002",
    "hit_delta_vs_v004a",
    "realized_delta_vs_v004a",
]

REPLACEMENT_COLUMNS = [
    "signal_date",
    "objective",
    "selected_grid_id",
    "selection_bucket",
    "code",
    "in_v005",
    "in_v002",
    "in_v004a",
    TARGET_COLUMN,
    HIGH_RETURN_COLUMN,
    REALIZED_RETURN_COLUMN,
    "v004a_model_rank",
    "v002_model_rank",
    "v004a_score",
    "v002_score",
    *CONTEXT_FEATURE_COLUMNS,
    "extreme_price",
    "extreme_vwap",
    "extreme_close_low",
    "near_miss_5_7",
]

PROFILE_COLUMNS = [
    "profile_group",
    "rows",
    "date_count",
    "target_rate",
    "avg_high_return",
    "avg_realized_return",
    "avg_v004a_rank",
    "avg_v002_rank",
    "avg_rank_total_score",
    "avg_price_rank",
    "avg_vwap_rank",
    "avg_close_low",
    "extreme_price_rate",
    "extreme_vwap_rate",
    "extreme_close_low_rate",
    "near_miss_5_7_rate",
]


def run_failure_attribution(
    scored_file: str | Path = DEFAULT_SCORED_FILE,
    v005_dir: str | Path = DEFAULT_V005_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    objective: str = DEFAULT_OBJECTIVE,
    top_n: int = DEFAULT_TOP_N,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
    realized_gap_threshold: float = DEFAULT_REALIZED_GAP_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    scored_path = Path(scored_file)
    v005_path = Path(v005_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = prepare_scored_candidates(scored_path)
    context = build_cross_model_context(scored, v004a_l2=float(v004a_l2), v004a_positive_weight=float(v004a_positive_weight))
    v002_top = select_model_topn(context, model_label="v002", rank_column="v002_model_rank", top_n=int(top_n))
    v004a_top = select_model_topn(context, model_label="v004a", rank_column="v004a_model_rank", top_n=int(top_n))
    selected_combos = load_selected_combos(v005_path / "v005_selected_combos.csv")
    history = load_objective_history(v005_path, objective=objective)
    v005_selected = build_v005_selected_from_history(history, selected_combos)

    daily = build_daily_failure_frame(
        v005_selected=v005_selected,
        v002_top=v002_top,
        v004a_top=v004a_top,
        objective=objective,
        top_n=int(top_n),
        realized_gap_threshold=float(realized_gap_threshold),
    )
    failure_dates = daily[daily["failure_reason"].astype(str) != ""].copy().reset_index(drop=True)
    replacements = build_replacement_rows(failure_dates, context, objective=objective)
    profile = build_failure_profile(failure_dates, replacements, daily)

    daily_csv = out_dir / "v005_failure_dates.csv"
    replacement_csv = out_dir / "v005_vs_v002_failure_replacement.csv"
    profile_csv = out_dir / "v005_failure_candidate_profile.csv"
    report_path = out_dir / "v005_failure_report.md"

    failure_dates[FAILURE_DATE_COLUMNS].to_csv(daily_csv, index=False, encoding="utf-8-sig")
    replacements[REPLACEMENT_COLUMNS].to_csv(replacement_csv, index=False, encoding="utf-8-sig")
    profile[PROFILE_COLUMNS].to_csv(profile_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            scored_path=scored_path,
            v005_path=v005_path,
            out_dir=out_dir,
            objective=objective,
            top_n=int(top_n),
            realized_gap_threshold=float(realized_gap_threshold),
            daily=daily,
            failure_dates=failure_dates,
            replacements=replacements,
            profile=profile,
        ),
        encoding="utf-8",
    )
    return failure_dates, replacements, profile, report_path


def prepare_scored_candidates(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"code": str})
    required = ["model_id", "evaluation_scope", "signal_date", "code", "model_score", "model_rank", TARGET_COLUMN, HIGH_RETURN_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"scored candidates missing required columns: {missing}")
    frame = raw.copy()
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["model_id"] = frame["model_id"].astype(str)
    frame["evaluation_scope"] = frame["evaluation_scope"].astype(str)
    frame[TARGET_COLUMN] = _bool_series(frame[TARGET_COLUMN])
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    return frame


def build_cross_model_context(scored: pd.DataFrame, v004a_l2: float, v004a_positive_weight: float) -> pd.DataFrame:
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

    v002 = scored[(scored["model_id"] == V002_MODEL_ID) & (scored["evaluation_scope"] == SCOPE)].copy()
    if v002.empty:
        raise RuntimeError(f"no v002 rows found for model_id={V002_MODEL_ID}, scope={SCOPE}")
    v002 = v002[["signal_date", "code", "model_score", "model_rank"]].rename(
        columns={"model_score": "v002_score", "model_rank": "v002_model_rank"}
    )

    context = v004a.merge(v002, on=["signal_date", "code"], how="left")
    context["v004a_model_rank"] = pd.to_numeric(context["v004a_model_rank"], errors="coerce")
    context["v002_model_rank"] = pd.to_numeric(context["v002_model_rank"], errors="coerce")
    context["extreme_price"] = pd.to_numeric(context["rank_log_candidate_base_price"], errors="coerce") >= 0.85
    context["extreme_vwap"] = pd.to_numeric(context["rank_d1_close_vwap_pct"], errors="coerce") >= 0.85
    context["extreme_close_low"] = pd.to_numeric(context["inter_close_low"], errors="coerce") >= 0.90
    context["near_miss_5_7"] = (~context[TARGET_COLUMN].astype(bool)) & pd.to_numeric(context[HIGH_RETURN_COLUMN], errors="coerce").between(5.0, 7.0, inclusive="left")
    return context.sort_values(["signal_date", "v004a_model_rank", "code"]).reset_index(drop=True)


def load_selected_combos(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"missing {path}; run `python -m src.v005_set_selector` first")
    frame = pd.read_csv(path, dtype={"codes": str})
    required = ["grid_id", "signal_date", "codes", "hit_count", "all_hit", "avg_high_return", "avg_realized_return"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["grid_id"]).copy()
    frame["grid_id"] = frame["grid_id"].astype(int)
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["codes"] = frame["codes"].astype(str)
    frame["hit_count"] = pd.to_numeric(frame["hit_count"], errors="coerce").fillna(0).astype(int)
    frame["all_hit"] = _bool_series(frame["all_hit"])
    frame["avg_high_return"] = pd.to_numeric(frame["avg_high_return"], errors="coerce")
    frame["avg_realized_return"] = pd.to_numeric(frame["avg_realized_return"], errors="coerce")
    return frame


def load_objective_history(v005_dir: Path, objective: str) -> pd.DataFrame:
    objective_history_path = v005_dir / "v005_wf_objective_history.csv"
    wf_history_path = v005_dir / "v005_wf_grid_history.csv"
    if objective_history_path.exists():
        history = pd.read_csv(objective_history_path)
        required = ["selection_objective", "validation_date", "selected_grid_id"]
        missing = [column for column in required if column not in history.columns]
        if missing:
            raise RuntimeError(f"{objective_history_path} missing required columns: {missing}")
        history = history[history["selection_objective"].astype(str) == str(objective)].copy()
        if history.empty:
            available = sorted(pd.read_csv(objective_history_path)["selection_objective"].astype(str).unique().tolist())
            raise RuntimeError(f"objective={objective!r} not found in {objective_history_path}; available={available}")
        history = history.rename(columns={"validation_date": "signal_date", "selected_grid_id": "grid_id"})
    elif wf_history_path.exists():
        history = pd.read_csv(wf_history_path)
        required = ["validation_date", "selected_grid_id"]
        missing = [column for column in required if column not in history.columns]
        if missing:
            raise RuntimeError(f"{wf_history_path} missing required columns: {missing}")
        history = history.rename(columns={"validation_date": "signal_date", "selected_grid_id": "grid_id"})
        history["selection_objective"] = "v005_wf_grid_history"
    else:
        raise RuntimeError(f"missing objective history in {v005_dir}; run objective sweep first")
    history["signal_date"] = history["signal_date"].astype(str)
    history["grid_id"] = pd.to_numeric(history["grid_id"], errors="coerce").astype("Int64")
    history = history.dropna(subset=["grid_id"]).copy()
    history["grid_id"] = history["grid_id"].astype(int)
    return history.sort_values("signal_date").reset_index(drop=True)


def build_v005_selected_from_history(history: pd.DataFrame, selected_combos: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, item in history.iterrows():
        date = str(item["signal_date"])
        grid_id = int(item["grid_id"])
        match = selected_combos[(selected_combos["signal_date"] == date) & (selected_combos["grid_id"] == grid_id)].copy()
        if match.empty:
            raise RuntimeError(f"missing v005 selected combo for date={date}, grid_id={grid_id}")
        row = match.iloc[0].copy()
        row["selected_grid_id"] = grid_id
        rows.append(row)
    if not rows:
        raise RuntimeError("empty v005 selected history")
    return pd.DataFrame(rows).reset_index(drop=True)


def select_model_topn(context: pd.DataFrame, model_label: str, rank_column: str, top_n: int) -> pd.DataFrame:
    selected = (
        context.sort_values(["signal_date", rank_column, "code"], ascending=[True, True, True])
        .groupby("signal_date", as_index=False, dropna=False)
        .head(int(top_n))
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for date, group in selected.groupby("signal_date", dropna=False):
        rows.append(selection_summary_row(str(date), group, prefix=model_label, top_n=int(top_n)))
    return pd.DataFrame(rows)


def selection_summary_row(signal_date: str, group: pd.DataFrame, prefix: str, top_n: int) -> dict[str, Any]:
    hit_count = int(group[TARGET_COLUMN].astype(bool).sum())
    return {
        "signal_date": signal_date,
        f"{prefix}_codes": ",".join(group["code"].astype(str).tolist()),
        f"{prefix}_hit_count": hit_count,
        f"{prefix}_all_hit": bool(hit_count == int(top_n)),
        f"{prefix}_avg_high_return": _mean(group[HIGH_RETURN_COLUMN]),
        f"{prefix}_avg_realized_return": _mean(group[REALIZED_RETURN_COLUMN]),
    }


def build_daily_failure_frame(
    v005_selected: pd.DataFrame,
    v002_top: pd.DataFrame,
    v004a_top: pd.DataFrame,
    objective: str,
    top_n: int,
    realized_gap_threshold: float,
) -> pd.DataFrame:
    daily = v005_selected.copy()
    daily = daily.rename(
        columns={
            "codes": "v005_codes",
            "hit_count": "v005_hit_count",
            "all_hit": "v005_all_hit",
            "avg_high_return": "v005_avg_high_return",
            "avg_realized_return": "v005_avg_realized_return",
        }
    )
    daily["selected_grid_id"] = pd.to_numeric(daily.get("selected_grid_id", daily["grid_id"]), errors="coerce").astype(int)
    daily["objective"] = objective
    daily = daily.merge(v002_top, on="signal_date", how="left").merge(v004a_top, on="signal_date", how="left")
    daily["v002_hit_count"] = pd.to_numeric(daily["v002_hit_count"], errors="coerce").fillna(0).astype(int)
    daily["v004a_hit_count"] = pd.to_numeric(daily["v004a_hit_count"], errors="coerce").fillna(0).astype(int)
    daily["hit_delta_vs_v002"] = pd.to_numeric(daily["v005_hit_count"], errors="coerce") - pd.to_numeric(daily["v002_hit_count"], errors="coerce")
    daily["realized_delta_vs_v002"] = pd.to_numeric(daily["v005_avg_realized_return"], errors="coerce") - pd.to_numeric(daily["v002_avg_realized_return"], errors="coerce")
    daily["hit_delta_vs_v004a"] = pd.to_numeric(daily["v005_hit_count"], errors="coerce") - pd.to_numeric(daily["v004a_hit_count"], errors="coerce")
    daily["realized_delta_vs_v004a"] = pd.to_numeric(daily["v005_avg_realized_return"], errors="coerce") - pd.to_numeric(daily["v004a_avg_realized_return"], errors="coerce")
    daily["failure_reason"] = daily.apply(lambda row: failure_reason(row, realized_gap_threshold=float(realized_gap_threshold), top_n=int(top_n)), axis=1)
    return daily.sort_values("signal_date").reset_index(drop=True)


def failure_reason(row: pd.Series, realized_gap_threshold: float, top_n: int) -> str:
    reasons: list[str] = []
    v005_hit = int(row.get("v005_hit_count", 0))
    v002_hit = int(row.get("v002_hit_count", 0))
    v005_realized = pd.to_numeric(pd.Series([row.get("v005_avg_realized_return", np.nan)]), errors="coerce").iloc[0]
    v002_realized = pd.to_numeric(pd.Series([row.get("v002_avg_realized_return", np.nan)]), errors="coerce").iloc[0]
    if v005_hit == 0:
        reasons.append("v005_zero_hit")
    if v005_hit < v002_hit:
        reasons.append("v005_hit_less_than_v002")
    if v002_hit == int(top_n) and v005_hit < int(top_n):
        reasons.append("missed_v002_all_hit_day")
    if pd.notna(v005_realized) and v005_realized < 0:
        reasons.append("v005_negative_realized")
    if pd.notna(v005_realized) and pd.notna(v002_realized) and v005_realized < v002_realized - float(realized_gap_threshold):
        reasons.append("v005_realized_lags_v002")
    return ";".join(reasons)


def build_replacement_rows(failure_dates: pd.DataFrame, context: pd.DataFrame, objective: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, day in failure_dates.iterrows():
        date = str(day["signal_date"])
        v005_codes = parse_codes(day.get("v005_codes", ""))
        v002_codes = parse_codes(day.get("v002_codes", ""))
        v004a_codes = parse_codes(day.get("v004a_codes", ""))
        all_codes = sorted(v005_codes | v002_codes)
        date_context = context[context["signal_date"].astype(str) == date].copy()
        for code in all_codes:
            item = date_context[date_context["code"].astype(str) == code].head(1)
            if item.empty:
                row = blank_replacement_row(date, code)
            else:
                row = item.iloc[0].to_dict()
            in_v005 = code in v005_codes
            in_v002 = code in v002_codes
            in_v004a = code in v004a_codes
            if in_v005 and in_v002:
                bucket = "overlap"
            elif in_v005:
                bucket = "v005_only"
            elif in_v002:
                bucket = "v002_only"
            else:
                bucket = "other"
            rows.append(
                {
                    "signal_date": date,
                    "objective": objective,
                    "selected_grid_id": int(day["selected_grid_id"]),
                    "selection_bucket": bucket,
                    "code": code,
                    "in_v005": bool(in_v005),
                    "in_v002": bool(in_v002),
                    "in_v004a": bool(in_v004a),
                    TARGET_COLUMN: bool(row.get(TARGET_COLUMN, False)),
                    HIGH_RETURN_COLUMN: row.get(HIGH_RETURN_COLUMN, np.nan),
                    REALIZED_RETURN_COLUMN: row.get(REALIZED_RETURN_COLUMN, np.nan),
                    "v004a_model_rank": row.get("v004a_model_rank", np.nan),
                    "v002_model_rank": row.get("v002_model_rank", np.nan),
                    "v004a_score": row.get("v004a_score", np.nan),
                    "v002_score": row.get("v002_score", np.nan),
                    **{column: row.get(column, np.nan) for column in CONTEXT_FEATURE_COLUMNS},
                    "extreme_price": bool(row.get("extreme_price", False)),
                    "extreme_vwap": bool(row.get("extreme_vwap", False)),
                    "extreme_close_low": bool(row.get("extreme_close_low", False)),
                    "near_miss_5_7": bool(row.get("near_miss_5_7", False)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=REPLACEMENT_COLUMNS)
    frame = pd.DataFrame(rows)
    return frame[REPLACEMENT_COLUMNS].sort_values(["signal_date", "selection_bucket", "code"]).reset_index(drop=True)


def blank_replacement_row(signal_date: str, code: str) -> dict[str, Any]:
    row: dict[str, Any] = {"signal_date": signal_date, "code": code, TARGET_COLUMN: False}
    for column in [HIGH_RETURN_COLUMN, REALIZED_RETURN_COLUMN, "v004a_model_rank", "v002_model_rank", "v004a_score", "v002_score", *CONTEXT_FEATURE_COLUMNS]:
        row[column] = np.nan
    row["extreme_price"] = False
    row["extreme_vwap"] = False
    row["extreme_close_low"] = False
    row["near_miss_5_7"] = False
    return row


def build_failure_profile(failure_dates: pd.DataFrame, replacements: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not replacements.empty:
        for bucket, group in replacements.groupby("selection_bucket", dropna=False):
            rows.append(profile_row(f"replacement:{bucket}", group))
        for label, group in replacements.groupby("target_label", dropna=False) if "target_label" in replacements.columns else []:
            rows.append(profile_row(f"target:{label}", group))
        rows.append(profile_row("replacement:all_failure_rows", replacements))
        v005_false = replacements[(replacements["in_v005"].astype(bool)) & (~replacements[TARGET_COLUMN].astype(bool))].copy()
        v002_true = replacements[(replacements["in_v002"].astype(bool)) & (replacements[TARGET_COLUMN].astype(bool))].copy()
        if not v005_false.empty:
            rows.append(profile_row("v005_selected_false", v005_false))
        if not v002_true.empty:
            rows.append(profile_row("v002_selected_true", v002_true))
    if not failure_dates.empty:
        rows.append(daily_failure_profile_row("daily:failure_dates", failure_dates))
    non_failure = daily[daily["failure_reason"].astype(str) == ""].copy()
    if not non_failure.empty:
        rows.append(daily_failure_profile_row("daily:non_failure_dates", non_failure))
    if not rows:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    return pd.DataFrame(rows)[PROFILE_COLUMNS]


def profile_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    target = frame[TARGET_COLUMN].astype(bool) if TARGET_COLUMN in frame.columns else pd.Series(dtype=bool)
    return {
        "profile_group": name,
        "rows": int(len(frame)),
        "date_count": int(frame["signal_date"].nunique()) if "signal_date" in frame.columns else 0,
        "target_rate": _safe_rate(int(target.sum()), int(len(target))),
        "avg_high_return": _mean(frame[HIGH_RETURN_COLUMN]) if HIGH_RETURN_COLUMN in frame.columns else np.nan,
        "avg_realized_return": _mean(frame[REALIZED_RETURN_COLUMN]) if REALIZED_RETURN_COLUMN in frame.columns else np.nan,
        "avg_v004a_rank": _mean(frame["v004a_model_rank"]) if "v004a_model_rank" in frame.columns else np.nan,
        "avg_v002_rank": _mean(frame["v002_model_rank"]) if "v002_model_rank" in frame.columns else np.nan,
        "avg_rank_total_score": _mean(frame["rank_total_score"]) if "rank_total_score" in frame.columns else np.nan,
        "avg_price_rank": _mean(frame["rank_log_candidate_base_price"]) if "rank_log_candidate_base_price" in frame.columns else np.nan,
        "avg_vwap_rank": _mean(frame["rank_d1_close_vwap_pct"]) if "rank_d1_close_vwap_pct" in frame.columns else np.nan,
        "avg_close_low": _mean(frame["inter_close_low"]) if "inter_close_low" in frame.columns else np.nan,
        "extreme_price_rate": _safe_rate(int(frame["extreme_price"].astype(bool).sum()), int(len(frame))) if "extreme_price" in frame.columns else np.nan,
        "extreme_vwap_rate": _safe_rate(int(frame["extreme_vwap"].astype(bool).sum()), int(len(frame))) if "extreme_vwap" in frame.columns else np.nan,
        "extreme_close_low_rate": _safe_rate(int(frame["extreme_close_low"].astype(bool).sum()), int(len(frame))) if "extreme_close_low" in frame.columns else np.nan,
        "near_miss_5_7_rate": _safe_rate(int(frame["near_miss_5_7"].astype(bool).sum()), int(len(frame))) if "near_miss_5_7" in frame.columns else np.nan,
    }


def daily_failure_profile_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "profile_group": name,
        "rows": int(len(frame)),
        "date_count": int(frame["signal_date"].nunique()),
        "target_rate": _mean(pd.to_numeric(frame["v005_hit_count"], errors="coerce") / 3.0),
        "avg_high_return": _mean(frame["v005_avg_high_return"]),
        "avg_realized_return": _mean(frame["v005_avg_realized_return"]),
        "avg_v004a_rank": np.nan,
        "avg_v002_rank": np.nan,
        "avg_rank_total_score": np.nan,
        "avg_price_rank": np.nan,
        "avg_vwap_rank": np.nan,
        "avg_close_low": np.nan,
        "extreme_price_rate": np.nan,
        "extreme_vwap_rate": np.nan,
        "extreme_close_low_rate": np.nan,
        "near_miss_5_7_rate": np.nan,
    }


def build_report(
    scored_path: Path,
    v005_path: Path,
    out_dir: Path,
    objective: str,
    top_n: int,
    realized_gap_threshold: float,
    daily: pd.DataFrame,
    failure_dates: pd.DataFrame,
    replacements: pd.DataFrame,
    profile: pd.DataFrame,
) -> str:
    lines = [
        "# v005 failure attribution",
        "",
        "## Scope",
        "",
        "This research-only diagnostic compares v005 walk-forward selections with v002 and v004a Top3 selections.",
        "It is intended to identify failure dates, replacement candidates, and possible fallback/downside-guard conditions.",
        "",
        "## Configuration",
        "",
        f"- scored file: `{scored_path}`",
        f"- v005 dir: `{v005_path}`",
        f"- output dir: `{out_dir}`",
        f"- objective: `{objective}`",
        f"- top_n: `{top_n}`",
        f"- realized_gap_threshold: `{realized_gap_threshold:g}`",
        f"- validation dates: `{len(daily)}`",
        f"- failure dates: `{len(failure_dates)}`",
        f"- replacement rows: `{len(replacements)}`",
        "",
        "## Failure dates",
        "",
    ]
    lines.extend(_markdown_table(failure_dates, FAILURE_DATE_COLUMNS))
    lines.extend(["", "## Candidate profile", ""])
    lines.extend(_markdown_table(profile, PROFILE_COLUMNS))
    lines.extend(["", "## v005 vs v002 replacement rows", ""])
    lines.extend(_markdown_table(replacements, REPLACEMENT_COLUMNS))
    lines.extend(
        [
            "",
            "## Research interpretation checklist",
            "",
            "- Check whether v005-only false positives have systematic price/VWAP/close-low or total-rank patterns.",
            "- Check whether v002-only true positives cluster in a regime that v005 underweights.",
            "- If v005 failures are mostly negative-realized days, add downside guard diagnostics before expanding the rule grid.",
            "- If failures are mostly v002 all-hit days, test a v002 fallback gate rather than more single-score reranking.",
        ]
    )
    return "\n".join(lines)


def parse_codes(text: Any) -> set[str]:
    if pd.isna(text):
        return set()
    return {part.strip().zfill(6) for part in str(text).split(",") if part.strip()}


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
    parser = argparse.ArgumentParser(description="Run v005 failure attribution against v002/v004a baselines.")
    parser.add_argument("--scored-file", default=str(DEFAULT_SCORED_FILE))
    parser.add_argument("--v005-dir", default=str(DEFAULT_V005_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    parser.add_argument("--realized-gap-threshold", type=float, default=DEFAULT_REALIZED_GAP_THRESHOLD)
    args = parser.parse_args(argv)

    failure_dates, replacements, profile, report_path = run_failure_attribution(
        scored_file=args.scored_file,
        v005_dir=args.v005_dir,
        output_dir=args.output_dir,
        objective=args.objective,
        top_n=args.top_n,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
        realized_gap_threshold=args.realized_gap_threshold,
    )
    print(f"failure date rows: {len(failure_dates)}")
    print(f"replacement rows: {len(replacements)}")
    print(f"profile rows: {len(profile)}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

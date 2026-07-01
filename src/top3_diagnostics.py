from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_FILE = Path("reports/v004a/grid_v2_scored/v004a_scored_candidates.csv")
DEFAULT_OUTPUT_DIR = Path("reports/top3_diagnostics")
DEFAULT_TOP_N = 3
DEFAULT_CANDIDATE_TOP_K = 15
DEFAULT_V004A_L2 = 0.30
DEFAULT_V004A_POSITIVE_WEIGHT = 1.5

TARGET_COLUMN = "target7_d2open_d3high"
HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
CLOSE_RETURN_COLUMN = "d2open_d3close_return_pct"
REALIZED_RETURN_COLUMN = "realized_return_pct"

V004A_MODEL_ID = "logistic_v004a_weighted"
V002_MODEL_ID = "ranking_model_v002_core_momentum_support"
SCOPE = "walk_forward"

# Diagnostic-only columns. They are never used as deployable daily features here.
DIAGNOSTIC_FEATURE_COLUMNS = [
    "candidate_base_price",
    "log_candidate_base_price",
    "rank_log_candidate_base_price",
    "d1_close_vwap_pct",
    "rank_d1_close_vwap_pct",
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_total_score",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_active_money_score",
    "inter_close_low",
    "inter_close_trend",
    "inter_total_trend",
    "inter_total_active",
    "inter_low_active",
    "spread_close_low",
    "days_since_d0",
    "rank_days_since_d0",
    "graph_quality_score",
    "total_score",
    "trend_hold_score",
    "theme_score",
    "active_money_score",
]

MODEL_SUMMARY_COLUMNS = [
    "model_label",
    "model_id",
    "row_filter",
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
]

DAILY_COLUMNS = [
    "model_label",
    "signal_date",
    "selected_count",
    "hit_count",
    "all_hit",
    "top3_target_rate",
    "avg_top3_high_return",
    "avg_top3_realized_return",
    "top3_codes",
    "top3_returns",
    "top3_model_ranks",
]

TRANCHE_BINS = [
    (1, 1, "rank_1"),
    (2, 3, "rank_2_3"),
    (4, 5, "rank_4_5"),
    (6, 10, "rank_6_10"),
    (11, 15, "rank_11_15"),
    (16, 30, "rank_16_30"),
    (31, 999999, "rank_31_plus"),
]


def run_top3_diagnostics(
    scored_file: str | Path = DEFAULT_SCORED_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    scored_path = Path(scored_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_scored_candidates(scored_path)
    v004a = select_v004a_frame(frame, v004a_l2=float(v004a_l2), v004a_positive_weight=float(v004a_positive_weight))
    v002 = select_model_frame(frame, V002_MODEL_ID, model_label="v002")

    model_frames = {
        "v004a_best": v004a,
        "v002": v002,
    }

    daily_model_comparison, model_summary = build_model_comparison(model_frames, top_n=int(top_n))
    replacement = build_v002_vs_v004a_replacement(v004a, v002, top_n=int(top_n))
    false_profile = build_false_positive_profile(v004a, v002, top_n=int(top_n))
    tranche = build_rank_tranche_hit_rate(v004a, v002, candidate_top_k=int(candidate_top_k))
    combo_profile = build_set_level_combo_profile(v004a, v002, top_n=int(top_n), candidate_top_k=int(candidate_top_k))

    model_summary_csv = out_dir / "model_summary.csv"
    daily_csv = out_dir / "daily_model_comparison.csv"
    replacement_csv = out_dir / "v002_vs_v004a_replacement.csv"
    false_csv = out_dir / "false_positive_profile.csv"
    tranche_csv = out_dir / "rank_tranche_hit_rate.csv"
    combo_csv = out_dir / "set_level_combo_profile.csv"
    report_path = out_dir / "top3_diagnostics_report.md"

    model_summary.to_csv(model_summary_csv, index=False, encoding="utf-8-sig")
    daily_model_comparison.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    replacement.to_csv(replacement_csv, index=False, encoding="utf-8-sig")
    false_profile.to_csv(false_csv, index=False, encoding="utf-8-sig")
    tranche.to_csv(tranche_csv, index=False, encoding="utf-8-sig")
    combo_profile.to_csv(combo_csv, index=False, encoding="utf-8-sig")

    report_path.write_text(
        build_report(
            scored_path=scored_path,
            output_dir=out_dir,
            top_n=int(top_n),
            candidate_top_k=int(candidate_top_k),
            v004a_l2=float(v004a_l2),
            v004a_positive_weight=float(v004a_positive_weight),
            model_summary=model_summary,
            daily_model_comparison=daily_model_comparison,
            replacement=replacement,
            false_profile=false_profile,
            tranche=tranche,
            combo_profile=combo_profile,
        ),
        encoding="utf-8",
    )

    return model_summary, daily_model_comparison, replacement, false_profile, tranche, combo_profile, report_path


def prepare_scored_candidates(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"code": str})
    required = [
        "model_id",
        "evaluation_scope",
        "signal_date",
        "code",
        "model_score",
        "model_rank",
        TARGET_COLUMN,
        HIGH_RETURN_COLUMN,
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"scored candidates missing required columns: {missing}")

    frame = raw.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)
    frame["evaluation_scope"] = frame["evaluation_scope"].astype(str)
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="coerce")
    frame["model_rank"] = pd.to_numeric(frame["model_rank"], errors="coerce")
    frame[TARGET_COLUMN] = _bool_series(frame[TARGET_COLUMN])
    for column in [HIGH_RETURN_COLUMN, CLOSE_RETURN_COLUMN, REALIZED_RETURN_COLUMN, "l2", "positive_weight", *DIAGNOSTIC_FEATURE_COLUMNS]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if CLOSE_RETURN_COLUMN not in frame.columns:
        frame[CLOSE_RETURN_COLUMN] = np.nan
    if REALIZED_RETURN_COLUMN not in frame.columns:
        frame[REALIZED_RETURN_COLUMN] = np.nan
    for column in DIAGNOSTIC_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def select_v004a_frame(frame: pd.DataFrame, v004a_l2: float, v004a_positive_weight: float) -> pd.DataFrame:
    l2 = pd.to_numeric(frame.get("l2"), errors="coerce")
    positive_weight = pd.to_numeric(frame.get("positive_weight"), errors="coerce")
    mask = (
        (frame["model_id"] == V004A_MODEL_ID)
        & (frame["evaluation_scope"] == SCOPE)
        & (l2.sub(float(v004a_l2)).abs() <= 1e-9)
        & (positive_weight.sub(float(v004a_positive_weight)).abs() <= 1e-9)
    )
    result = frame[mask].copy()
    if result.empty:
        raise RuntimeError(f"no v004a rows found for l2={v004a_l2:g}, positive_weight={v004a_positive_weight:g}")
    result["model_label"] = "v004a_best"
    result["row_filter"] = f"{V004A_MODEL_ID}, l2={v004a_l2:g}, positive_weight={v004a_positive_weight:g}, scope={SCOPE}"
    result = result.rename(columns={"model_rank": "v004a_model_rank", "model_score": "v004a_score"})
    # Keep generic names for shared helper functions.
    result["model_rank"] = result["v004a_model_rank"]
    result["model_score"] = result["v004a_score"]
    return result


def select_model_frame(frame: pd.DataFrame, model_id: str, model_label: str) -> pd.DataFrame:
    result = frame[(frame["model_id"] == model_id) & (frame["evaluation_scope"] == SCOPE)].copy()
    if result.empty:
        raise RuntimeError(f"no scored rows found for model_id={model_id}, scope={SCOPE}")
    result["model_label"] = model_label
    result["row_filter"] = f"{model_id}, scope={SCOPE}"
    if model_label == "v002":
        result = result.rename(columns={"model_rank": "v002_model_rank", "model_score": "v002_score"})
        result["model_rank"] = result["v002_model_rank"]
        result["model_score"] = result["v002_score"]
    return result


def build_model_comparison(model_frames: dict[str, pd.DataFrame], top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, Any]] = []
    top_rows: list[pd.DataFrame] = []
    for label, frame in model_frames.items():
        for signal_date, group in frame.groupby("signal_date", dropna=False):
            selected = _select_topn(group, top_n=top_n).copy()
            selected["model_label"] = label
            selected["daily_rank"] = np.arange(1, len(selected) + 1)
            top_rows.append(selected)
            daily_rows.append(_daily_row(label, str(signal_date), selected, top_n=top_n))
    daily = pd.DataFrame(daily_rows)
    selected_all = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    summary = pd.DataFrame([_summary_row(label, daily[daily["model_label"] == label], selected_all[selected_all["model_label"] == label]) for label in model_frames])
    return daily[DAILY_COLUMNS], summary[MODEL_SUMMARY_COLUMNS]


def build_v002_vs_v004a_replacement(v004a: pd.DataFrame, v002: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    merged_context = _build_cross_model_context(v004a, v002)
    dates = sorted(set(v004a["signal_date"].dropna().astype(str)) & set(v002["signal_date"].dropna().astype(str)))
    for signal_date in dates:
        v004a_top = _select_topn(v004a[v004a["signal_date"] == signal_date], top_n)
        v002_top = _select_topn(v002[v002["signal_date"] == signal_date], top_n)
        v004a_codes = set(v004a_top["code"].astype(str))
        v002_codes = set(v002_top["code"].astype(str))
        context = merged_context[merged_context["signal_date"] == signal_date].copy()
        selected_codes = v004a_codes | v002_codes
        context = context[context["code"].isin(selected_codes)].copy()
        if context.empty:
            continue
        context["selection_group"] = context["code"].apply(
            lambda code: "overlap" if code in v004a_codes and code in v002_codes else ("v004a_only" if code in v004a_codes else "v002_only")
        )
        context["v004a_selected"] = context["code"].isin(v004a_codes)
        context["v002_selected"] = context["code"].isin(v002_codes)
        rows.append(context)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    ordered = [
        "signal_date",
        "code",
        "selection_group",
        TARGET_COLUMN,
        HIGH_RETURN_COLUMN,
        REALIZED_RETURN_COLUMN,
        "v004a_model_rank",
        "v002_model_rank",
        "v004a_score",
        "v002_score",
        *[column for column in DIAGNOSTIC_FEATURE_COLUMNS if column in result.columns],
    ]
    return result[[column for column in ordered if column in result.columns]].sort_values(["signal_date", "selection_group", "code"])


def build_false_positive_profile(v004a: pd.DataFrame, v002: pd.DataFrame, top_n: int) -> pd.DataFrame:
    context = _build_cross_model_context(v004a, v002)
    rows: list[dict[str, Any]] = []
    for label, frame in {"v004a_best": v004a, "v002": v002}.items():
        selected_rows = []
        for _, group in frame.groupby("signal_date", dropna=False):
            selected_rows.append(_select_topn(group, top_n=top_n))
        selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
        selected_context = context.merge(selected[["signal_date", "code"]].drop_duplicates(), on=["signal_date", "code"], how="inner")
        for bucket_name, bucket in {
            "selected_all": selected_context,
            "selected_true_positive": selected_context[selected_context[TARGET_COLUMN].astype(bool)],
            "selected_false_positive": selected_context[~selected_context[TARGET_COLUMN].astype(bool)],
            "selected_near_miss_5_7": selected_context[(~selected_context[TARGET_COLUMN].astype(bool)) & (pd.to_numeric(selected_context[HIGH_RETURN_COLUMN], errors="coerce").between(5.0, 7.0, inclusive="left"))],
            "selected_hard_v004a_fp": selected_context[(~selected_context[TARGET_COLUMN].astype(bool)) & (pd.to_numeric(selected_context["v004a_model_rank"], errors="coerce") <= 3)],
            "selected_hard_v002_fp": selected_context[(~selected_context[TARGET_COLUMN].astype(bool)) & (pd.to_numeric(selected_context["v002_model_rank"], errors="coerce") <= 3)],
        }.items():
            rows.append(_profile_row(model_label=label, bucket=bucket_name, frame=bucket))
    return pd.DataFrame(rows)


def build_rank_tranche_hit_rate(v004a: pd.DataFrame, v002: pd.DataFrame, candidate_top_k: int) -> pd.DataFrame:
    context = _build_cross_model_context(v004a, v002)
    rows: list[dict[str, Any]] = []
    for rank_column, model_label in [("v004a_model_rank", "v004a_rank"), ("v002_model_rank", "v002_rank")]:
        values = pd.to_numeric(context[rank_column], errors="coerce")
        ranked = context[values <= int(candidate_top_k)].copy()
        for low, high, label in TRANCHE_BINS:
            bucket = ranked[(pd.to_numeric(ranked[rank_column], errors="coerce") >= low) & (pd.to_numeric(ranked[rank_column], errors="coerce") <= high)].copy()
            if bucket.empty:
                continue
            rows.append(
                {
                    "rank_source": model_label,
                    "rank_column": rank_column,
                    "tranche": label,
                    "rank_low": low,
                    "rank_high": high,
                    "row_count": int(len(bucket)),
                    "date_count": int(bucket["signal_date"].nunique()),
                    "target_rate": _safe_rate(int(bucket[TARGET_COLUMN].astype(bool).sum()), int(len(bucket))),
                    "avg_high_return": _mean(bucket[HIGH_RETURN_COLUMN]),
                    "avg_realized_return": _mean(bucket[REALIZED_RETURN_COLUMN]),
                    "avg_candidate_base_price": _mean(bucket["candidate_base_price"]),
                    "avg_rank_log_candidate_base_price": _mean(bucket["rank_log_candidate_base_price"]),
                    "avg_rank_d1_close_vwap_pct": _mean(bucket["rank_d1_close_vwap_pct"]),
                    "avg_inter_close_low": _mean(bucket["inter_close_low"]),
                }
            )
    return pd.DataFrame(rows)


def build_set_level_combo_profile(v004a: pd.DataFrame, v002: pd.DataFrame, top_n: int, candidate_top_k: int) -> pd.DataFrame:
    context = _build_cross_model_context(v004a, v002)
    rows: list[dict[str, Any]] = []
    candidate_pool = context[pd.to_numeric(context["v004a_model_rank"], errors="coerce") <= int(candidate_top_k)].copy()
    for signal_date, group in candidate_pool.groupby("signal_date", dropna=False):
        group = group.sort_values(["v004a_model_rank", "code"]).reset_index(drop=True)
        if len(group) < int(top_n):
            continue
        combo_rows = []
        for combo_indexes in itertools.combinations(range(len(group)), int(top_n)):
            combo = group.iloc[list(combo_indexes)].copy()
            hit_count = int(combo[TARGET_COLUMN].astype(bool).sum())
            combo_rows.append(
                {
                    "signal_date": str(signal_date),
                    "codes": ",".join(combo["code"].astype(str).tolist()),
                    "hit_count": hit_count,
                    "all_hit": bool(hit_count == int(top_n)),
                    "avg_high_return": _mean(combo[HIGH_RETURN_COLUMN]),
                    "avg_realized_return": _mean(combo[REALIZED_RETURN_COLUMN]),
                    "min_v004a_rank": _mean_min(combo["v004a_model_rank"]),
                    "max_v004a_rank": _mean_max(combo["v004a_model_rank"]),
                    "avg_v004a_rank": _mean(combo["v004a_model_rank"]),
                    "min_v002_rank": _mean_min(combo["v002_model_rank"]),
                    "max_v002_rank": _mean_max(combo["v002_model_rank"]),
                    "avg_v002_rank": _mean(combo["v002_model_rank"]),
                    "rank_dispersion_v004a": _mean_max(combo["v004a_model_rank"]) - _mean_min(combo["v004a_model_rank"]),
                    "rank_dispersion_v002": _mean_max(combo["v002_model_rank"]) - _mean_min(combo["v002_model_rank"]),
                    "contains_v004a_top3": bool((pd.to_numeric(combo["v004a_model_rank"], errors="coerce") <= 3).any()),
                    "contains_v002_top3": bool((pd.to_numeric(combo["v002_model_rank"], errors="coerce") <= 3).any()),
                    "all_in_v004a_top10": bool((pd.to_numeric(combo["v004a_model_rank"], errors="coerce") <= 10).all()),
                    "all_in_v002_top10": bool((pd.to_numeric(combo["v002_model_rank"], errors="coerce") <= 10).all()),
                    "extreme_price_count": int((pd.to_numeric(combo["rank_log_candidate_base_price"], errors="coerce") >= 0.85).sum()),
                    "extreme_vwap_count": int((pd.to_numeric(combo["rank_d1_close_vwap_pct"], errors="coerce") >= 0.85).sum()),
                    "extreme_close_low_count": int((pd.to_numeric(combo["inter_close_low"], errors="coerce") >= 0.90).sum()),
                    "avg_total_rank": _mean(combo["rank_total_score"]),
                    "min_total_rank": _mean_min(combo["rank_total_score"]),
                    "avg_price_rank": _mean(combo["rank_log_candidate_base_price"]),
                    "avg_vwap_rank": _mean(combo["rank_d1_close_vwap_pct"]),
                    "avg_close_low": _mean(combo["inter_close_low"]),
                }
            )
        combo_frame = pd.DataFrame(combo_rows)
        if combo_frame.empty:
            continue
        all_hit = combo_frame[combo_frame["all_hit"].astype(bool)]
        not_all_hit = combo_frame[~combo_frame["all_hit"].astype(bool)]
        rows.append(_combo_summary_row(str(signal_date), "all_combos", combo_frame))
        rows.append(_combo_summary_row(str(signal_date), "all_hit_combos", all_hit))
        rows.append(_combo_summary_row(str(signal_date), "not_all_hit_combos", not_all_hit))
    return pd.DataFrame(rows)


def _build_cross_model_context(v004a: pd.DataFrame, v002: pd.DataFrame) -> pd.DataFrame:
    v004a_cols = [
        "signal_date",
        "code",
        TARGET_COLUMN,
        HIGH_RETURN_COLUMN,
        CLOSE_RETURN_COLUMN,
        REALIZED_RETURN_COLUMN,
        "v004a_model_rank",
        "v004a_score",
        *DIAGNOSTIC_FEATURE_COLUMNS,
    ]
    left = v004a[[column for column in v004a_cols if column in v004a.columns]].drop_duplicates(["signal_date", "code"]).copy()
    right = v002[["signal_date", "code", "v002_model_rank", "v002_score"]].drop_duplicates(["signal_date", "code"]).copy()
    context = left.merge(right, on=["signal_date", "code"], how="left")
    context["v002_model_rank"] = pd.to_numeric(context["v002_model_rank"], errors="coerce")
    context["v004a_model_rank"] = pd.to_numeric(context["v004a_model_rank"], errors="coerce")
    return context


def _select_topn(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["model_rank"] = pd.to_numeric(result["model_rank"], errors="coerce")
    result["model_score"] = pd.to_numeric(result["model_score"], errors="coerce")
    result = result.sort_values(["model_rank", "model_score", "code"], ascending=[True, False, True])
    return result.head(int(top_n)).copy()


def _daily_row(label: str, signal_date: str, selected: pd.DataFrame, top_n: int) -> dict[str, Any]:
    target = selected[TARGET_COLUMN].astype(bool) if not selected.empty else pd.Series(dtype=bool)
    hit_count = int(target.sum())
    return {
        "model_label": label,
        "signal_date": str(signal_date),
        "selected_count": int(len(selected)),
        "hit_count": hit_count,
        "all_hit": bool(len(selected) == int(top_n) and hit_count == int(top_n)),
        "top3_target_rate": _safe_rate(hit_count, int(len(selected))),
        "avg_top3_high_return": _mean(selected[HIGH_RETURN_COLUMN]) if not selected.empty else np.nan,
        "avg_top3_realized_return": _mean(selected[REALIZED_RETURN_COLUMN]) if not selected.empty else np.nan,
        "top3_codes": ",".join(selected["code"].astype(str).tolist()) if not selected.empty else "",
        "top3_returns": ",".join(f"{float(value):.2f}" for value in pd.to_numeric(selected[HIGH_RETURN_COLUMN], errors="coerce")) if not selected.empty else "",
        "top3_model_ranks": ",".join(f"{int(value)}" for value in pd.to_numeric(selected["model_rank"], errors="coerce").fillna(0)) if not selected.empty else "",
    }


def _summary_row(label: str, daily: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    targets = selected[TARGET_COLUMN].astype(bool) if not selected.empty else pd.Series(dtype=bool)
    row = {
        "model_label": label,
        "model_id": _first_value(selected.get("model_id")),
        "row_filter": _first_value(selected.get("row_filter")),
        "date_count": int(daily["signal_date"].nunique()) if not daily.empty else 0,
        "selected_ticket_count": int(len(selected)),
        "top3_target_rate": _safe_rate(int(targets.sum()), int(len(targets))),
        "top3_all_hit_rate": _safe_rate(int(daily["all_hit"].astype(bool).sum()), int(len(daily))) if not daily.empty else np.nan,
        "hit_count_0_days": int((pd.to_numeric(daily["hit_count"], errors="coerce").fillna(0) == 0).sum()) if not daily.empty else 0,
        "hit_count_1_days": int((pd.to_numeric(daily["hit_count"], errors="coerce").fillna(0) == 1).sum()) if not daily.empty else 0,
        "hit_count_2_days": int((pd.to_numeric(daily["hit_count"], errors="coerce").fillna(0) == 2).sum()) if not daily.empty else 0,
        "hit_count_3_days": int((pd.to_numeric(daily["hit_count"], errors="coerce").fillna(0) == 3).sum()) if not daily.empty else 0,
        "avg_top3_high_return": _mean(daily["avg_top3_high_return"]) if not daily.empty else np.nan,
        "avg_top3_realized_return": _mean(daily["avg_top3_realized_return"]) if not daily.empty else np.nan,
    }
    for rank in range(1, 4):
        rank_rows = selected[pd.to_numeric(selected.get("daily_rank"), errors="coerce") == rank] if not selected.empty else pd.DataFrame()
        row[f"rank{rank}_hit_rate"] = _safe_rate(int(rank_rows[TARGET_COLUMN].astype(bool).sum()), int(len(rank_rows))) if not rank_rows.empty else np.nan
    return row


def _profile_row(model_label: str, bucket: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "model_label": model_label,
        "bucket": bucket,
        "row_count": int(len(frame)),
        "date_count": int(frame["signal_date"].nunique()) if "signal_date" in frame.columns and not frame.empty else 0,
        "target_rate": _safe_rate(int(frame[TARGET_COLUMN].astype(bool).sum()), int(len(frame))) if not frame.empty else np.nan,
        "avg_high_return": _mean(frame[HIGH_RETURN_COLUMN]) if not frame.empty else np.nan,
        "avg_realized_return": _mean(frame[REALIZED_RETURN_COLUMN]) if not frame.empty else np.nan,
        "avg_v004a_rank": _mean(frame["v004a_model_rank"]) if "v004a_model_rank" in frame.columns else np.nan,
        "avg_v002_rank": _mean(frame["v002_model_rank"]) if "v002_model_rank" in frame.columns else np.nan,
        "avg_candidate_base_price": _mean(frame["candidate_base_price"]) if "candidate_base_price" in frame.columns else np.nan,
        "avg_rank_log_candidate_base_price": _mean(frame["rank_log_candidate_base_price"]) if "rank_log_candidate_base_price" in frame.columns else np.nan,
        "avg_rank_d1_close_vwap_pct": _mean(frame["rank_d1_close_vwap_pct"]) if "rank_d1_close_vwap_pct" in frame.columns else np.nan,
        "avg_rank_d1_close_ma10_pct": _mean(frame["rank_d1_close_ma10_pct"]) if "rank_d1_close_ma10_pct" in frame.columns else np.nan,
        "avg_rank_d1_low_ma10_pct": _mean(frame["rank_d1_low_ma10_pct"]) if "rank_d1_low_ma10_pct" in frame.columns else np.nan,
        "avg_inter_close_low": _mean(frame["inter_close_low"]) if "inter_close_low" in frame.columns else np.nan,
        "extreme_price_rate": _safe_rate(int((pd.to_numeric(frame.get("rank_log_candidate_base_price"), errors="coerce") >= 0.85).sum()), int(len(frame))) if not frame.empty else np.nan,
        "extreme_vwap_rate": _safe_rate(int((pd.to_numeric(frame.get("rank_d1_close_vwap_pct"), errors="coerce") >= 0.85).sum()), int(len(frame))) if not frame.empty else np.nan,
        "near_miss_5_7_rate": _safe_rate(
            int(((~frame[TARGET_COLUMN].astype(bool)) & (pd.to_numeric(frame[HIGH_RETURN_COLUMN], errors="coerce").between(5.0, 7.0, inclusive="left"))).sum()),
            int(len(frame)),
        )
        if not frame.empty
        else np.nan,
    }


def _combo_summary_row(signal_date: str, bucket: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "signal_date": signal_date,
        "bucket": bucket,
        "combo_count": int(len(frame)),
        "all_hit_rate": _safe_rate(int(frame["all_hit"].astype(bool).sum()), int(len(frame))) if not frame.empty else np.nan,
        "avg_hit_count": _mean(frame["hit_count"]) if not frame.empty else np.nan,
        "avg_high_return": _mean(frame["avg_high_return"]) if not frame.empty else np.nan,
        "avg_realized_return": _mean(frame["avg_realized_return"]) if not frame.empty else np.nan,
        "avg_v004a_rank": _mean(frame["avg_v004a_rank"]) if not frame.empty else np.nan,
        "avg_v002_rank": _mean(frame["avg_v002_rank"]) if not frame.empty else np.nan,
        "avg_rank_dispersion_v004a": _mean(frame["rank_dispersion_v004a"]) if not frame.empty else np.nan,
        "avg_rank_dispersion_v002": _mean(frame["rank_dispersion_v002"]) if not frame.empty else np.nan,
        "contains_v004a_top3_rate": _safe_rate(int(frame["contains_v004a_top3"].astype(bool).sum()), int(len(frame))) if not frame.empty else np.nan,
        "contains_v002_top3_rate": _safe_rate(int(frame["contains_v002_top3"].astype(bool).sum()), int(len(frame))) if not frame.empty else np.nan,
        "all_in_v004a_top10_rate": _safe_rate(int(frame["all_in_v004a_top10"].astype(bool).sum()), int(len(frame))) if not frame.empty else np.nan,
        "all_in_v002_top10_rate": _safe_rate(int(frame["all_in_v002_top10"].astype(bool).sum()), int(len(frame))) if not frame.empty else np.nan,
        "avg_extreme_price_count": _mean(frame["extreme_price_count"]) if not frame.empty else np.nan,
        "avg_extreme_vwap_count": _mean(frame["extreme_vwap_count"]) if not frame.empty else np.nan,
        "avg_extreme_close_low_count": _mean(frame["extreme_close_low_count"]) if not frame.empty else np.nan,
        "avg_total_rank": _mean(frame["avg_total_rank"]) if not frame.empty else np.nan,
        "min_total_rank": _mean(frame["min_total_rank"]) if not frame.empty else np.nan,
        "avg_price_rank": _mean(frame["avg_price_rank"]) if not frame.empty else np.nan,
        "avg_vwap_rank": _mean(frame["avg_vwap_rank"]) if not frame.empty else np.nan,
        "avg_close_low": _mean(frame["avg_close_low"]) if not frame.empty else np.nan,
    }


def build_report(
    scored_path: Path,
    output_dir: Path,
    top_n: int,
    candidate_top_k: int,
    v004a_l2: float,
    v004a_positive_weight: float,
    model_summary: pd.DataFrame,
    daily_model_comparison: pd.DataFrame,
    replacement: pd.DataFrame,
    false_profile: pd.DataFrame,
    tranche: pd.DataFrame,
    combo_profile: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Top3 attribution diagnostics",
        "",
        "## Scope",
        "",
        "This is a research-only attribution report. It does not train a model, does not write a ranking model JSON, and does not connect to run-daily.",
        "",
        "## Configuration",
        "",
        f"- scored file: `{scored_path}`",
        f"- output dir: `{output_dir}`",
        f"- top_n: `{top_n}`",
        f"- candidate_top_k: `{candidate_top_k}`",
        f"- v004a l2 / positive_weight: `{v004a_l2:g}` / `{v004a_positive_weight:g}`",
        "",
        "## Model summary",
        "",
    ]
    lines.extend(_markdown_table(model_summary, MODEL_SUMMARY_COLUMNS))
    lines.extend(["", "## Daily model comparison", ""])
    lines.extend(_markdown_table(daily_model_comparison, DAILY_COLUMNS))
    lines.extend(["", "## v002 vs v004a replacement profile", ""])
    replacement_summary = (
        replacement.groupby("selection_group", dropna=False)
        .agg(
            row_count=("code", "count"),
            target_rate=(TARGET_COLUMN, lambda values: _safe_rate(int(pd.Series(values).astype(bool).sum()), len(values))),
            avg_high_return=(HIGH_RETURN_COLUMN, _mean),
            avg_realized_return=(REALIZED_RETURN_COLUMN, _mean),
            avg_v004a_rank=("v004a_model_rank", _mean),
            avg_v002_rank=("v002_model_rank", _mean),
            avg_price=("candidate_base_price", _mean),
            avg_price_rank=("rank_log_candidate_base_price", _mean),
            avg_vwap_rank=("rank_d1_close_vwap_pct", _mean),
            avg_total_rank=("rank_total_score", _mean),
        )
        .reset_index()
        if not replacement.empty
        else pd.DataFrame()
    )
    lines.extend(_markdown_table(replacement_summary, list(replacement_summary.columns)))
    lines.extend(["", "## False-positive profile", ""])
    lines.extend(_markdown_table(false_profile, list(false_profile.columns)))
    lines.extend(["", "## Rank tranche hit rate", ""])
    lines.extend(_markdown_table(tranche, list(tranche.columns)))
    lines.extend(["", "## Set-level combo profile preview", ""])
    lines.extend(_markdown_table(combo_profile.head(60), list(combo_profile.columns)))
    lines.extend(
        [
            "",
            "## Research interpretation checklist",
            "",
            "- Check whether v002-only selections have lower extreme-price exposure than v004a-only selections.",
            "- Check whether v004a rank 2/3 deteriorates relative to rank 1.",
            "- Check whether all-hit combos are concentrated in rank-dispersed groups instead of pure top-ranked groups.",
            "- If all-hit combos show stable group-level structure, prefer a set-level Top3 selector over another single-score reranker.",
        ]
    )
    return "\n".join(lines)


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


def _mean_min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return np.nan
    return float(series.min())


def _mean_max(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return np.nan
    return float(series.max())


def _first_value(values: Any) -> str:
    if values is None:
        return ""
    series = pd.Series(values).dropna()
    if series.empty:
        return ""
    return str(series.iloc[0])


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
    parser = argparse.ArgumentParser(description="Run research-only Top3 model attribution diagnostics.")
    parser.add_argument("--scored-file", default=str(DEFAULT_SCORED_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    args = parser.parse_args(argv)

    model_summary, daily, replacement, false_profile, tranche, combo_profile, report_path = run_top3_diagnostics(
        scored_file=args.scored_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        candidate_top_k=args.candidate_top_k,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
    )
    print(f"model summary rows: {len(model_summary)}")
    print(f"daily comparison rows: {len(daily)}")
    print(f"replacement rows: {len(replacement)}")
    print(f"false profile rows: {len(false_profile)}")
    print(f"rank tranche rows: {len(tranche)}")
    print(f"combo profile rows: {len(combo_profile)}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_FILE = Path("reports/v004a/grid_v2_scored/v004a_scored_candidates.csv")
DEFAULT_V005_DIR = Path("reports/v005_set_selector")
DEFAULT_OUTPUT_DIR = Path("reports/v005_fallback_gate")
DEFAULT_OBJECTIVE = "realized_then_all_hit"
DEFAULT_TOP_N = 3
DEFAULT_V004A_L2 = 0.30
DEFAULT_V004A_POSITIVE_WEIGHT = 1.5

TARGET_COLUMN = "target7_d2open_d3high"
HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
REALIZED_RETURN_COLUMN = "realized_return_pct"
V004A_MODEL_ID = "logistic_v004a_weighted"
V002_MODEL_ID = "ranking_model_v002_core_momentum_support"
SCOPE = "walk_forward"

STRATEGIES = (
    "baseline_v005_realized",
    "fallback_v002_extreme_confirm",
    "fallback_v002_extreme_confirm_v005_weak",
    "fallback_v002_extreme_confirm_plus_risk",
    "risk_replace_only",
    "fallback_plus_risk_replace",
)

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
    "v004a_model_rank",
    "v002_model_rank",
    "v004a_score",
    "v002_score",
    "candidate_base_price",
    "rank_log_candidate_base_price",
    "rank_d1_close_vwap_pct",
    "inter_close_low",
    "rank_total_score",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_active_money_score",
    "graph_quality_score",
    "extreme_price",
    "extreme_vwap",
    "extreme_close_low",
    "near_miss_5_7",
]

DAILY_COLUMNS = [
    "strategy",
    "signal_date",
    "action",
    "selected_codes",
    "source_codes",
    "selected_grid_id",
    "hit_count",
    "all_hit",
    "avg_high_return",
    "avg_realized_return",
    "rank1_hit",
    "rank2_hit",
    "rank3_hit",
    "v002_codes",
    "v002_hit_count",
    "v002_all_hit",
    "v002_avg_realized_return",
    "v004a_codes",
    "v004a_hit_count",
    "v004a_all_hit",
    "v004a_avg_realized_return",
    "baseline_v005_codes",
    "baseline_v005_hit_count",
    "baseline_v005_all_hit",
    "baseline_v005_avg_realized_return",
    "gate_v002_extreme_vwap_count",
    "gate_v002_extreme_close_low_count",
    "gate_v005_avg_v002_rank",
    "gate_v005_has_risk_ticket",
    "gate_triggered",
]

SUMMARY_COLUMNS = [
    "strategy",
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
    "fallback_days",
    "risk_replace_days",
    "changed_days",
    "changed_from_baseline_days",
    "negative_realized_days",
    "v002_all_hit_captured_days",
]

REPLACEMENT_COLUMNS = [
    "strategy",
    "signal_date",
    "action",
    "selection_bucket",
    "code",
    "in_final",
    "in_baseline_v005",
    "in_v002",
    "in_v004a",
    TARGET_COLUMN,
    HIGH_RETURN_COLUMN,
    REALIZED_RETURN_COLUMN,
    *CONTEXT_FEATURE_COLUMNS,
]


def run_fallback_gate(
    scored_file: str | Path = DEFAULT_SCORED_FILE,
    v005_dir: str | Path = DEFAULT_V005_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    objective: str = DEFAULT_OBJECTIVE,
    top_n: int = DEFAULT_TOP_N,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    scored_path = Path(scored_file)
    v005_path = Path(v005_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = build_cross_model_context(
        prepare_scored_candidates(scored_path),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    selected_combos = load_selected_combos(v005_path / "v005_selected_combos.csv")
    objective_history = load_objective_history(v005_path, objective=objective)
    baseline = build_v005_baseline(objective_history, selected_combos, context, top_n=int(top_n))
    v002_top = build_model_topn(context, model_label="v002", rank_column="v002_model_rank", top_n=int(top_n))
    v004a_top = build_model_topn(context, model_label="v004a", rank_column="v004a_model_rank", top_n=int(top_n))

    daily_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for _, baseline_day in baseline.iterrows():
            date = str(baseline_day["signal_date"])
            day_context = context[context["signal_date"].astype(str) == date].copy()
            v002_day = v002_top[v002_top["signal_date"].astype(str) == date]
            v004a_day = v004a_top[v004a_top["signal_date"].astype(str) == date]
            if v002_day.empty:
                raise RuntimeError(f"missing v002 topN rows for date={date}")
            if v004a_day.empty:
                raise RuntimeError(f"missing v004a topN rows for date={date}")

            decision = apply_strategy(
                strategy=strategy,
                baseline_day=baseline_day,
                v002_day=v002_day.iloc[0],
                day_context=day_context,
                top_n=int(top_n),
            )
            selected_context = context_for_codes(day_context, decision["selected_codes"])
            metrics = summarize_codes(date, selected_context, selected_codes=decision["selected_codes"], top_n=int(top_n))
            daily_rows.append(
                {
                    "strategy": strategy,
                    "signal_date": date,
                    "action": decision["action"],
                    "selected_codes": ",".join(decision["selected_codes"]),
                    "source_codes": ",".join(decision["source_codes"]),
                    "selected_grid_id": int(baseline_day["selected_grid_id"]),
                    **metrics,
                    "v002_codes": v002_day.iloc[0]["codes"],
                    "v002_hit_count": int(v002_day.iloc[0]["hit_count"]),
                    "v002_all_hit": bool(v002_day.iloc[0]["all_hit"]),
                    "v002_avg_realized_return": float(v002_day.iloc[0]["avg_realized_return"]),
                    "v004a_codes": v004a_day.iloc[0]["codes"],
                    "v004a_hit_count": int(v004a_day.iloc[0]["hit_count"]),
                    "v004a_all_hit": bool(v004a_day.iloc[0]["all_hit"]),
                    "v004a_avg_realized_return": float(v004a_day.iloc[0]["avg_realized_return"]),
                    "baseline_v005_codes": baseline_day["codes"],
                    "baseline_v005_hit_count": int(baseline_day["hit_count"]),
                    "baseline_v005_all_hit": bool(baseline_day["all_hit"]),
                    "baseline_v005_avg_realized_return": float(baseline_day["avg_realized_return"]),
                    "gate_v002_extreme_vwap_count": int(baseline_day["v002_extreme_vwap_count"]),
                    "gate_v002_extreme_close_low_count": int(baseline_day["v002_extreme_close_low_count"]),
                    "gate_v005_avg_v002_rank": float(baseline_day["v005_avg_v002_rank"]),
                    "gate_v005_has_risk_ticket": bool(baseline_day["v005_has_risk_ticket"]),
                    "gate_triggered": bool(decision["gate_triggered"]),
                }
            )
            replacement_rows.extend(
                build_replacement_rows(
                    strategy=strategy,
                    signal_date=date,
                    action=decision["action"],
                    final_codes=decision["selected_codes"],
                    baseline_codes=parse_codes(baseline_day["codes"]),
                    v002_codes=parse_codes(v002_day.iloc[0]["codes"]),
                    v004a_codes=parse_codes(v004a_day.iloc[0]["codes"]),
                    day_context=day_context,
                )
            )

    daily = pd.DataFrame(daily_rows)
    summary = summarize_strategy_daily(daily, top_n=int(top_n))
    replacements = pd.DataFrame(replacement_rows)
    if replacements.empty:
        replacements = pd.DataFrame(columns=REPLACEMENT_COLUMNS)
    else:
        replacements = replacements[REPLACEMENT_COLUMNS].sort_values(["strategy", "signal_date", "selection_bucket", "code"]).reset_index(drop=True)

    summary_csv = out_dir / "v005_fallback_gate_summary.csv"
    daily_csv = out_dir / "v005_fallback_gate_daily.csv"
    replacement_csv = out_dir / "v005_fallback_gate_replacement.csv"
    report_path = out_dir / "v005_fallback_gate_report.md"

    summary[SUMMARY_COLUMNS].to_csv(summary_csv, index=False, encoding="utf-8-sig")
    daily[DAILY_COLUMNS].to_csv(daily_csv, index=False, encoding="utf-8-sig")
    replacements[REPLACEMENT_COLUMNS].to_csv(replacement_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            scored_path=scored_path,
            v005_path=v005_path,
            out_dir=out_dir,
            objective=objective,
            top_n=int(top_n),
            summary=summary,
            daily=daily,
            replacements=replacements,
        ),
        encoding="utf-8",
    )
    return summary[SUMMARY_COLUMNS], daily[DAILY_COLUMNS], replacements[REPLACEMENT_COLUMNS], report_path


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


def build_v005_baseline(history: pd.DataFrame, selected_combos: pd.DataFrame, context: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in history.iterrows():
        date = str(item["signal_date"])
        grid_id = int(item["grid_id"])
        selected = selected_combos[(selected_combos["signal_date"] == date) & (selected_combos["grid_id"] == grid_id)].copy()
        if selected.empty:
            raise RuntimeError(f"missing v005 selected combo for date={date}, grid_id={grid_id}")
        selected_row = selected.iloc[0]
        codes = parse_codes(selected_row["codes"])
        day_context = context[context["signal_date"].astype(str) == date].copy()
        v005_context = context_for_codes(day_context, codes)
        v002_top_context = day_context.sort_values(["v002_model_rank", "code"], ascending=[True, True]).head(int(top_n)).copy()
        risk_mask = v005_context.apply(is_risk_ticket, axis=1) if not v005_context.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "signal_date": date,
                "selected_grid_id": grid_id,
                "codes": ",".join(codes),
                "hit_count": int(selected_row["hit_count"]),
                "all_hit": bool(selected_row["all_hit"]),
                "avg_high_return": float(selected_row["avg_high_return"]),
                "avg_realized_return": float(selected_row["avg_realized_return"]),
                "v005_avg_v002_rank": _mean(v005_context["v002_model_rank"]) if not v005_context.empty else np.nan,
                "v005_has_risk_ticket": bool(risk_mask.any()) if len(risk_mask) else False,
                "v002_extreme_vwap_count": int(v002_top_context["extreme_vwap"].astype(bool).sum()),
                "v002_extreme_close_low_count": int(v002_top_context["extreme_close_low"].astype(bool).sum()),
                "v002_avg_v002_rank": _mean(v002_top_context["v002_model_rank"]),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def build_model_topn(context: pd.DataFrame, model_label: str, rank_column: str, top_n: int) -> pd.DataFrame:
    selected = (
        context.sort_values(["signal_date", rank_column, "code"], ascending=[True, True, True])
        .groupby("signal_date", as_index=False, dropna=False)
        .head(int(top_n))
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for date, group in selected.groupby("signal_date", dropna=False):
        metrics = summarize_codes(str(date), group, selected_codes=group["code"].astype(str).tolist(), top_n=int(top_n))
        rows.append({"signal_date": str(date), "codes": ",".join(group["code"].astype(str).tolist()), **metrics})
    return pd.DataFrame(rows)


def apply_strategy(
    strategy: str,
    baseline_day: pd.Series,
    v002_day: pd.Series,
    day_context: pd.DataFrame,
    top_n: int,
) -> dict[str, Any]:
    baseline_codes = parse_codes(baseline_day["codes"])
    v002_codes = parse_codes(v002_day["codes"])
    extreme_confirm = (
        int(baseline_day["v002_extreme_vwap_count"]) >= 2
        and int(baseline_day["v002_extreme_close_low_count"]) >= 2
    )
    v005_weak = pd.notna(baseline_day["v005_avg_v002_rank"]) and float(baseline_day["v005_avg_v002_rank"]) >= 12.0
    has_risk = bool(baseline_day["v005_has_risk_ticket"])

    if strategy == "baseline_v005_realized":
        return make_decision(baseline_codes, baseline_codes, "baseline", gate_triggered=False)
    if strategy == "fallback_v002_extreme_confirm":
        if extreme_confirm:
            return make_decision(v002_codes, v002_codes, "fallback_to_v002_extreme_confirm", gate_triggered=True)
        return make_decision(baseline_codes, baseline_codes, "keep_v005", gate_triggered=False)
    if strategy == "fallback_v002_extreme_confirm_v005_weak":
        if extreme_confirm and v005_weak:
            return make_decision(v002_codes, v002_codes, "fallback_to_v002_extreme_confirm_v005_weak", gate_triggered=True)
        return make_decision(baseline_codes, baseline_codes, "keep_v005", gate_triggered=False)
    if strategy == "fallback_v002_extreme_confirm_plus_risk":
        if extreme_confirm and has_risk:
            return make_decision(v002_codes, v002_codes, "fallback_to_v002_extreme_confirm_plus_risk", gate_triggered=True)
        return make_decision(baseline_codes, baseline_codes, "keep_v005", gate_triggered=False)
    if strategy == "risk_replace_only":
        replaced = replace_risk_tickets(baseline_codes, v002_codes, day_context, top_n=top_n)
        return make_decision(replaced, baseline_codes, "risk_replace_only" if replaced != baseline_codes else "keep_v005", gate_triggered=replaced != baseline_codes)
    if strategy == "fallback_plus_risk_replace":
        if extreme_confirm and v005_weak:
            return make_decision(v002_codes, v002_codes, "fallback_to_v002_extreme_confirm_v005_weak", gate_triggered=True)
        replaced = replace_risk_tickets(baseline_codes, v002_codes, day_context, top_n=top_n)
        return make_decision(replaced, baseline_codes, "risk_replace_only" if replaced != baseline_codes else "keep_v005", gate_triggered=replaced != baseline_codes)
    raise RuntimeError(f"unsupported strategy: {strategy}")


def make_decision(selected_codes: list[str] | set[str], source_codes: list[str] | set[str], action: str, gate_triggered: bool) -> dict[str, Any]:
    return {
        "selected_codes": normalize_code_order(selected_codes),
        "source_codes": normalize_code_order(source_codes),
        "action": action,
        "gate_triggered": bool(gate_triggered),
    }


def replace_risk_tickets(v005_codes: list[str] | set[str], v002_codes: list[str] | set[str], day_context: pd.DataFrame, top_n: int) -> list[str]:
    current = normalize_code_order(v005_codes)
    v002_ordered = normalize_code_order(v002_codes)
    current_context = context_for_codes(day_context, current)
    risk_codes = set(current_context[current_context.apply(is_risk_ticket, axis=1)]["code"].astype(str).tolist()) if not current_context.empty else set()
    if not risk_codes:
        return current
    replacement_pool = [code for code in v002_ordered if code not in current]
    output: list[str] = []
    replacements = iter(replacement_pool)
    for code in current:
        if code in risk_codes:
            replacement = next(replacements, None)
            if replacement is None:
                output.append(code)
            else:
                output.append(replacement)
        else:
            output.append(code)
    return normalize_code_order(output)[: int(top_n)]


def is_risk_ticket(row: pd.Series) -> bool:
    return (
        pd.notna(row.get("v002_model_rank", np.nan))
        and float(row.get("v002_model_rank", np.nan)) >= 20.0
        and pd.notna(row.get("rank_log_candidate_base_price", np.nan))
        and float(row.get("rank_log_candidate_base_price", np.nan)) >= 0.90
        and pd.notna(row.get("rank_d1_close_vwap_pct", np.nan))
        and float(row.get("rank_d1_close_vwap_pct", np.nan)) < 0.60
        and pd.notna(row.get("inter_close_low", np.nan))
        and float(row.get("inter_close_low", np.nan)) < 0.85
    )


def summarize_codes(signal_date: str, frame: pd.DataFrame, selected_codes: list[str] | set[str], top_n: int) -> dict[str, Any]:
    selected = context_for_codes(frame, selected_codes)
    targets = selected[TARGET_COLUMN].astype(bool).tolist() if not selected.empty else []
    hit_count = int(sum(targets))
    return {
        "hit_count": hit_count,
        "all_hit": bool(hit_count == int(top_n)),
        "avg_high_return": _mean(selected[HIGH_RETURN_COLUMN]) if not selected.empty else np.nan,
        "avg_realized_return": _mean(selected[REALIZED_RETURN_COLUMN]) if not selected.empty else np.nan,
        "rank1_hit": bool(targets[0]) if len(targets) >= 1 else False,
        "rank2_hit": bool(targets[1]) if len(targets) >= 2 else False,
        "rank3_hit": bool(targets[2]) if len(targets) >= 3 else False,
    }


def summarize_strategy_daily(daily: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, group in daily.groupby("strategy", dropna=False):
        hit_count = pd.to_numeric(group["hit_count"], errors="coerce").fillna(0).astype(int)
        rows.append(
            {
                "strategy": str(strategy),
                "date_count": int(group["signal_date"].nunique()),
                "selected_ticket_count": int(len(group) * int(top_n)),
                "top3_target_rate": _safe_rate(int(hit_count.sum()), int(len(group) * int(top_n))),
                "top3_all_hit_rate": _safe_rate(int(group["all_hit"].astype(bool).sum()), int(len(group))),
                "hit_count_0_days": int((hit_count == 0).sum()),
                "hit_count_1_days": int((hit_count == 1).sum()),
                "hit_count_2_days": int((hit_count == 2).sum()),
                "hit_count_3_days": int((hit_count == 3).sum()),
                "rank1_hit_rate": _safe_rate(int(group["rank1_hit"].astype(bool).sum()), int(len(group))),
                "rank2_hit_rate": _safe_rate(int(group["rank2_hit"].astype(bool).sum()), int(len(group))),
                "rank3_hit_rate": _safe_rate(int(group["rank3_hit"].astype(bool).sum()), int(len(group))),
                "avg_top3_high_return": _mean(group["avg_high_return"]),
                "avg_top3_realized_return": _mean(group["avg_realized_return"]),
                "fallback_days": int(group["action"].astype(str).str.contains("fallback").sum()),
                "risk_replace_days": int(group["action"].astype(str).str.contains("risk_replace").sum()),
                "changed_days": int(group["gate_triggered"].astype(bool).sum()),
                "changed_from_baseline_days": int((group["selected_codes"].astype(str) != group["baseline_v005_codes"].astype(str)).sum()),
                "negative_realized_days": int((pd.to_numeric(group["avg_realized_return"], errors="coerce") < 0).sum()),
                "v002_all_hit_captured_days": int((group["v002_all_hit"].astype(bool) & group["all_hit"].astype(bool)).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["top3_all_hit_rate", "hit_count_0_days", "avg_top3_realized_return", "top3_target_rate"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def build_replacement_rows(
    strategy: str,
    signal_date: str,
    action: str,
    final_codes: list[str],
    baseline_codes: set[str],
    v002_codes: set[str],
    v004a_codes: set[str],
    day_context: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_codes = sorted(set(final_codes) | set(baseline_codes) | set(v002_codes))
    for code in all_codes:
        item = day_context[day_context["code"].astype(str) == code].head(1)
        row = blank_context_row(signal_date, code) if item.empty else item.iloc[0].to_dict()
        in_final = code in set(final_codes)
        in_baseline = code in baseline_codes
        in_v002 = code in v002_codes
        in_v004a = code in v004a_codes
        if in_final and in_baseline and in_v002:
            bucket = "final_baseline_v002_overlap"
        elif in_final and in_baseline:
            bucket = "final_baseline_only"
        elif in_final and in_v002:
            bucket = "final_v002_replacement"
        elif in_final:
            bucket = "final_only"
        elif in_baseline:
            bucket = "dropped_baseline"
        elif in_v002:
            bucket = "unused_v002"
        else:
            bucket = "other"
        rows.append(
            {
                "strategy": strategy,
                "signal_date": signal_date,
                "action": action,
                "selection_bucket": bucket,
                "code": code,
                "in_final": bool(in_final),
                "in_baseline_v005": bool(in_baseline),
                "in_v002": bool(in_v002),
                "in_v004a": bool(in_v004a),
                TARGET_COLUMN: bool(row.get(TARGET_COLUMN, False)),
                HIGH_RETURN_COLUMN: row.get(HIGH_RETURN_COLUMN, np.nan),
                REALIZED_RETURN_COLUMN: row.get(REALIZED_RETURN_COLUMN, np.nan),
                **{column: row.get(column, np.nan) for column in CONTEXT_FEATURE_COLUMNS},
            }
        )
    return rows


def blank_context_row(signal_date: str, code: str) -> dict[str, Any]:
    row: dict[str, Any] = {"signal_date": signal_date, "code": code, TARGET_COLUMN: False}
    for column in [HIGH_RETURN_COLUMN, REALIZED_RETURN_COLUMN, *CONTEXT_FEATURE_COLUMNS]:
        row[column] = False if column.startswith("extreme_") or column == "near_miss_5_7" else np.nan
    return row


def context_for_codes(frame: pd.DataFrame, codes: list[str] | set[str]) -> pd.DataFrame:
    ordered = normalize_code_order(codes)
    if not ordered:
        return frame.iloc[0:0].copy()
    result = frame[frame["code"].astype(str).isin(ordered)].copy()
    order_map = {code: index for index, code in enumerate(ordered)}
    result["_selection_order"] = result["code"].map(order_map)
    return result.sort_values("_selection_order").drop(columns=["_selection_order"]).reset_index(drop=True)


def normalize_code_order(codes: list[str] | set[str]) -> list[str]:
    if isinstance(codes, set):
        return sorted(str(code).zfill(6) for code in codes)
    output: list[str] = []
    for code in codes:
        value = str(code).strip().zfill(6)
        if value and value not in output:
            output.append(value)
    return output


def parse_codes(text: Any) -> set[str]:
    if pd.isna(text):
        return set()
    return {part.strip().zfill(6) for part in str(text).split(",") if part.strip()}


def build_report(
    scored_path: Path,
    v005_path: Path,
    out_dir: Path,
    objective: str,
    top_n: int,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    replacements: pd.DataFrame,
) -> str:
    lines = [
        "# v005 fallback gate diagnostics",
        "",
        "## Scope",
        "",
        "This research-only diagnostic compares a v005 objective baseline with small v002 fallback and downside replacement gates.",
        "It does not change the v005 selector, does not train a model, and does not touch daily production logic.",
        "",
        "## Configuration",
        "",
        f"- scored file: `{scored_path}`",
        f"- v005 dir: `{v005_path}`",
        f"- output dir: `{out_dir}`",
        f"- baseline objective: `{objective}`",
        f"- top_n: `{top_n}`",
        "",
        "## Strategy definitions",
        "",
        "- `baseline_v005_realized`: keep v005 `realized_then_all_hit` selections.",
        "- `fallback_v002_extreme_confirm`: fallback to v002 if v002 Top3 has at least 2 extreme VWAP and 2 extreme close-low names.",
        "- `fallback_v002_extreme_confirm_v005_weak`: same as above, but only if v005 selected names have average v002 rank >= 12.",
        "- `fallback_v002_extreme_confirm_plus_risk`: same extreme confirmation, but only if v005 has at least one catastrophic risk ticket.",
        "- `risk_replace_only`: replace catastrophic v005 tickets with v002 Top3 names not already selected.",
        "- `fallback_plus_risk_replace`: fallback by the cautious v005-weak gate first, otherwise apply risk replacement.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(_markdown_table(summary, SUMMARY_COLUMNS))
    lines.extend(["", "## Daily selections", ""])
    lines.extend(_markdown_table(daily, DAILY_COLUMNS))
    lines.extend(["", "## Replacement rows", ""])
    lines.extend(_markdown_table(replacements, REPLACEMENT_COLUMNS))
    lines.extend(
        [
            "",
            "## Research checklist",
            "",
            "- A useful gate must keep top3_all_hit_rate >= the v005 baseline.",
            "- It should keep hit_count_0_days <= the v005 baseline.",
            "- It should improve avg_top3_realized_return above the v002 reference if possible.",
            "- It must not achieve gains only by repairing one date while damaging existing v005 all-hit dates.",
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
    parser = argparse.ArgumentParser(description="Run v005 fallback gate diagnostics.")
    parser.add_argument("--scored-file", default=str(DEFAULT_SCORED_FILE))
    parser.add_argument("--v005-dir", default=str(DEFAULT_V005_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    args = parser.parse_args(argv)

    summary, daily, replacements, report_path = run_fallback_gate(
        scored_file=args.scored_file,
        v005_dir=args.v005_dir,
        output_dir=args.output_dir,
        objective=args.objective,
        top_n=args.top_n,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
    )
    print(f"strategy rows: {len(summary)}")
    print(f"daily rows: {len(daily)}")
    print(f"replacement rows: {len(replacements)}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best strategy: {best['strategy']}")
        print(f"best top3_all_hit_rate: {float(best['top3_all_hit_rate']):.4f}")
        print(f"best avg_realized_return: {float(best['avg_top3_realized_return']):.4f}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

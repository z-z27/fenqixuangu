from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SCORED_FILE = Path("reports/v004a/grid_v2_scored/v004a_scored_candidates.csv")
DEFAULT_V005_DIR = Path("reports/v005_set_selector")
DEFAULT_OUTPUT_DIR = Path("reports/v005_fallback_gate")
DEFAULT_OBJECTIVE = "realized_then_all_hit"
DEFAULT_TOP_N = 3
DEFAULT_V004A_L2 = 0.30
DEFAULT_V004A_POSITIVE_WEIGHT = 1.5

TARGET = "target7_d2open_d3high"
HIGH = "d2open_d3high_return_pct"
REAL = "realized_return_pct"
V004A_MODEL_ID = "logistic_v004a_weighted"
V002_MODEL_ID = "ranking_model_v002_core_momentum_support"
SCOPE = "walk_forward"

PRIMARY_POLICY = "policy_v005_v002_regime_fallback"

STRATEGIES = [
    "baseline_v005_realized",
    PRIMARY_POLICY,
    "risk_replace_only",
]

SUMMARY_COLUMNS = [
    "strategy", "date_count", "selected_ticket_count", "top3_target_rate", "top3_all_hit_rate",
    "hit_count_0_days", "hit_count_1_days", "hit_count_2_days", "hit_count_3_days",
    "rank1_hit_rate", "rank2_hit_rate", "rank3_hit_rate", "avg_top3_high_return",
    "avg_top3_realized_return", "fallback_days", "risk_replace_days", "changed_days",
    "changed_from_baseline_days", "negative_realized_days", "v002_all_hit_captured_days",
]

DAILY_COLUMNS = [
    "strategy", "signal_date", "action", "selected_codes", "source_codes", "selected_grid_id",
    "hit_count", "all_hit", "avg_high_return", "avg_realized_return", "rank1_hit", "rank2_hit",
    "rank3_hit", "v002_codes", "v002_hit_count", "v002_all_hit", "v002_avg_realized_return",
    "v004a_codes", "v004a_hit_count", "v004a_all_hit", "v004a_avg_realized_return",
    "baseline_v005_codes", "baseline_v005_hit_count", "baseline_v005_all_hit",
    "baseline_v005_avg_realized_return", "gate_v002_extreme_vwap_count",
    "gate_v002_extreme_close_low_count", "gate_v005_avg_v002_rank", "gate_v005_has_risk_ticket",
    "gate_triggered",
]

REPLACEMENT_COLUMNS = [
    "strategy", "signal_date", "action", "selection_bucket", "code", "in_final", "in_baseline_v005",
    "in_v002", "in_v004a", TARGET, HIGH, REAL, "v004a_model_rank", "v002_model_rank",
    "v004a_score", "v002_score", "rank_log_candidate_base_price", "rank_d1_close_vwap_pct",
    "inter_close_low", "rank_total_score", "extreme_price", "extreme_vwap", "extreme_close_low",
    "near_miss_5_7",
]


def run_fallback_gate(
    scored_file=DEFAULT_SCORED_FILE,
    v005_dir=DEFAULT_V005_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    objective=DEFAULT_OBJECTIVE,
    top_n=DEFAULT_TOP_N,
    v004a_l2=DEFAULT_V004A_L2,
    v004a_positive_weight=DEFAULT_V004A_POSITIVE_WEIGHT,
):
    scored_file = Path(scored_file)
    v005_dir = Path(v005_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(scored_file, v004a_l2, v004a_positive_weight)
    combos = load_combos(v005_dir / "v005_selected_combos.csv")
    hist = load_history(v005_dir, objective)
    baseline = build_baseline(hist, combos, ctx, top_n)
    v002 = build_topn(ctx, "v002_model_rank", top_n)
    v004a = build_topn(ctx, "v004a_model_rank", top_n)

    daily_rows, replacement_rows = [], []
    for strategy in STRATEGIES:
        for _, base in baseline.iterrows():
            date = str(base["signal_date"])
            day_ctx = ctx[ctx["signal_date"] == date].copy()
            v002_day = v002[v002["signal_date"] == date].iloc[0]
            v004a_day = v004a[v004a["signal_date"] == date].iloc[0]
            decision = decide(strategy, base, parse_codes(v002_day["codes"]), day_ctx, top_n)
            selected = ctx_for_codes(day_ctx, decision["selected_codes"])
            met = metrics(selected, decision["selected_codes"], top_n)
            daily_rows.append({
                "strategy": strategy, "signal_date": date, "action": decision["action"],
                "selected_codes": ",".join(decision["selected_codes"]),
                "source_codes": ",".join(decision["source_codes"]),
                "selected_grid_id": int(base["selected_grid_id"]), **met,
                "v002_codes": v002_day["codes"], "v002_hit_count": int(v002_day["hit_count"]),
                "v002_all_hit": bool(v002_day["all_hit"]),
                "v002_avg_realized_return": float(v002_day["avg_realized_return"]),
                "v004a_codes": v004a_day["codes"], "v004a_hit_count": int(v004a_day["hit_count"]),
                "v004a_all_hit": bool(v004a_day["all_hit"]),
                "v004a_avg_realized_return": float(v004a_day["avg_realized_return"]),
                "baseline_v005_codes": base["codes"], "baseline_v005_hit_count": int(base["hit_count"]),
                "baseline_v005_all_hit": bool(base["all_hit"]),
                "baseline_v005_avg_realized_return": float(base["avg_realized_return"]),
                "gate_v002_extreme_vwap_count": int(base["v002_extreme_vwap_count"]),
                "gate_v002_extreme_close_low_count": int(base["v002_extreme_close_low_count"]),
                "gate_v005_avg_v002_rank": float(base["v005_avg_v002_rank"]),
                "gate_v005_has_risk_ticket": bool(base["v005_has_risk_ticket"]),
                "gate_triggered": bool(decision["gate_triggered"]),
            })
            replacement_rows.extend(replacement_detail(
                strategy, date, decision["action"], decision["selected_codes"],
                parse_codes(base["codes"]), parse_codes(v002_day["codes"]), parse_codes(v004a_day["codes"]), day_ctx,
            ))

    daily = pd.DataFrame(daily_rows)[DAILY_COLUMNS]
    summary = summarize(daily, top_n)[SUMMARY_COLUMNS]
    repl = pd.DataFrame(replacement_rows)
    if repl.empty:
        repl = pd.DataFrame(columns=REPLACEMENT_COLUMNS)
    else:
        repl = repl[REPLACEMENT_COLUMNS].sort_values(["strategy", "signal_date", "selection_bucket", "code"])

    summary.to_csv(output_dir / "v005_fallback_gate_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "v005_fallback_gate_daily.csv", index=False, encoding="utf-8-sig")
    repl.to_csv(output_dir / "v005_fallback_gate_replacement.csv", index=False, encoding="utf-8-sig")
    report_path = output_dir / "v005_fallback_gate_report.md"
    report_path.write_text(make_report(scored_file, v005_dir, output_dir, objective, summary, daily, repl), encoding="utf-8")
    return summary, daily, repl, report_path


def build_context(path, l2, positive_weight):
    if not Path(path).exists():
        raise RuntimeError(f"missing scored file: {path}")
    df = pd.read_csv(path, dtype={"code": str})
    required = ["model_id", "evaluation_scope", "signal_date", "code", "model_score", "model_rank", TARGET, HIGH]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"scored candidates missing columns: {missing}")
    df = df.copy()
    df["signal_date"] = df["signal_date"].astype(str)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df[TARGET] = bool_series(df[TARGET])
    for col in ["l2", "positive_weight", "model_score", "model_rank", HIGH, REAL, "rank_log_candidate_base_price",
                "rank_d1_close_vwap_pct", "inter_close_low", "rank_total_score"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    v4 = df[(df["model_id"].astype(str) == V004A_MODEL_ID) & (df["evaluation_scope"].astype(str) == SCOPE)
            & (df["l2"].sub(float(l2)).abs() <= 1e-9)
            & (df["positive_weight"].sub(float(positive_weight)).abs() <= 1e-9)].copy()
    v2 = df[(df["model_id"].astype(str) == V002_MODEL_ID) & (df["evaluation_scope"].astype(str) == SCOPE)].copy()
    if v4.empty:
        raise RuntimeError("no v004a rows found for configured l2/positive_weight")
    if v2.empty:
        raise RuntimeError("no v002 rows found")
    v4 = v4.rename(columns={"model_score": "v004a_score", "model_rank": "v004a_model_rank"})
    v2 = v2[["signal_date", "code", "model_score", "model_rank"]].rename(columns={"model_score": "v002_score", "model_rank": "v002_model_rank"})
    out = v4.merge(v2, on=["signal_date", "code"], how="left")
    out["v004a_model_rank"] = pd.to_numeric(out["v004a_model_rank"], errors="coerce")
    out["v002_model_rank"] = pd.to_numeric(out["v002_model_rank"], errors="coerce")
    out["extreme_price"] = out["rank_log_candidate_base_price"] >= 0.85
    out["extreme_vwap"] = out["rank_d1_close_vwap_pct"] >= 0.85
    out["extreme_close_low"] = out["inter_close_low"] >= 0.90
    out["near_miss_5_7"] = (~out[TARGET].astype(bool)) & pd.to_numeric(out[HIGH], errors="coerce").between(5.0, 7.0, inclusive="left")
    return out.sort_values(["signal_date", "v004a_model_rank", "code"]).reset_index(drop=True)


def load_combos(path):
    if not Path(path).exists():
        raise RuntimeError(f"missing {path}; run python -m src.v005_set_selector first")
    df = pd.read_csv(path, dtype={"codes": str})
    for c in ["grid_id", "signal_date", "codes", "hit_count", "all_hit", "avg_high_return", "avg_realized_return"]:
        if c not in df.columns:
            raise RuntimeError(f"{path} missing column: {c}")
    df["signal_date"] = df["signal_date"].astype(str)
    df["grid_id"] = pd.to_numeric(df["grid_id"], errors="coerce").astype(int)
    df["hit_count"] = pd.to_numeric(df["hit_count"], errors="coerce").fillna(0).astype(int)
    df["all_hit"] = bool_series(df["all_hit"])
    df["avg_high_return"] = pd.to_numeric(df["avg_high_return"], errors="coerce")
    df["avg_realized_return"] = pd.to_numeric(df["avg_realized_return"], errors="coerce")
    return df


def load_history(v005_dir, objective):
    objective_path = Path(v005_dir) / "v005_wf_objective_history.csv"
    wf_path = Path(v005_dir) / "v005_wf_grid_history.csv"
    if objective_path.exists():
        h = pd.read_csv(objective_path)
        h = h[h["selection_objective"].astype(str) == str(objective)].copy()
        if h.empty:
            raise RuntimeError(f"objective {objective!r} not found in {objective_path}")
        h = h.rename(columns={"validation_date": "signal_date", "selected_grid_id": "grid_id"})
    elif wf_path.exists():
        h = pd.read_csv(wf_path).rename(columns={"validation_date": "signal_date", "selected_grid_id": "grid_id"})
    else:
        raise RuntimeError(f"missing walk-forward history in {v005_dir}")
    h["signal_date"] = h["signal_date"].astype(str)
    h["grid_id"] = pd.to_numeric(h["grid_id"], errors="coerce").astype(int)
    return h.sort_values("signal_date")


def build_baseline(history, combos, ctx, top_n):
    rows = []
    for _, h in history.iterrows():
        date, gid = str(h["signal_date"]), int(h["grid_id"])
        row = combos[(combos["signal_date"] == date) & (combos["grid_id"] == gid)]
        if row.empty:
            raise RuntimeError(f"missing combo for date={date}, grid_id={gid}")
        row = row.iloc[0]
        codes = parse_codes(row["codes"])
        day = ctx[ctx["signal_date"] == date].copy()
        v005_ctx = ctx_for_codes(day, codes)
        v002_top = day.sort_values(["v002_model_rank", "code"]).head(top_n)
        risk = v005_ctx.apply(is_risk_ticket, axis=1) if not v005_ctx.empty else pd.Series(dtype=bool)
        rows.append({
            "signal_date": date, "selected_grid_id": gid, "codes": ",".join(codes),
            "hit_count": int(row["hit_count"]), "all_hit": bool(row["all_hit"]),
            "avg_high_return": float(row["avg_high_return"]), "avg_realized_return": float(row["avg_realized_return"]),
            "v005_avg_v002_rank": mean(v005_ctx["v002_model_rank"]) if not v005_ctx.empty else np.nan,
            "v005_has_risk_ticket": bool(risk.any()) if len(risk) else False,
            "v002_extreme_vwap_count": int(v002_top["extreme_vwap"].astype(bool).sum()),
            "v002_extreme_close_low_count": int(v002_top["extreme_close_low"].astype(bool).sum()),
        })
    return pd.DataFrame(rows)


def build_topn(ctx, rank_col, top_n):
    rows = []
    chosen = ctx.sort_values(["signal_date", rank_col, "code"]).groupby("signal_date", as_index=False).head(top_n)
    for date, g in chosen.groupby("signal_date"):
        codes = g["code"].astype(str).tolist()
        rows.append({"signal_date": str(date), "codes": ",".join(codes), **metrics(g, codes, top_n)})
    return pd.DataFrame(rows)


def is_policy_fallback(base):
    extreme_confirm = int(base["v002_extreme_vwap_count"]) >= 2 and int(base["v002_extreme_close_low_count"]) >= 2
    close_low_dominant = int(base["v002_extreme_close_low_count"]) >= 3
    v005_weak = pd.notna(base["v005_avg_v002_rank"]) and float(base["v005_avg_v002_rank"]) >= 12
    return (extreme_confirm or close_low_dominant) and v005_weak


def decide(strategy, base, v002_codes, day_ctx, top_n):
    baseline = parse_codes(base["codes"])
    if strategy == "baseline_v005_realized":
        return decision(baseline, baseline, "baseline", False)
    if strategy == PRIMARY_POLICY:
        if is_policy_fallback(base):
            return decision(v002_codes, v002_codes, "fallback_to_v002_regime_policy", True)
        return decision(baseline, baseline, "keep_v005", False)
    if strategy == "risk_replace_only":
        out = replace_risk(baseline, v002_codes, day_ctx, top_n)
        return decision(out, baseline, "risk_replace_only" if out != baseline else "keep_v005", out != baseline)
    raise RuntimeError(f"unknown strategy: {strategy}")


def decision(selected, source, action, triggered):
    return {"selected_codes": norm(selected), "source_codes": norm(source), "action": action, "gate_triggered": bool(triggered)}


def replace_risk(v005_codes, v002_codes, day_ctx, top_n):
    current, v002_order = norm(v005_codes), norm(v002_codes)
    cur_ctx = ctx_for_codes(day_ctx, current)
    risks = set(cur_ctx[cur_ctx.apply(is_risk_ticket, axis=1)]["code"].astype(str).tolist()) if not cur_ctx.empty else set()
    if not risks:
        return current
    repl_iter = iter([c for c in v002_order if c not in current])
    out = []
    for c in current:
        out.append(next(repl_iter, c) if c in risks else c)
    return norm(out)[:top_n]


def is_risk_ticket(row):
    return (pd.notna(row.get("v002_model_rank", np.nan)) and float(row.get("v002_model_rank", np.nan)) >= 20
            and pd.notna(row.get("rank_log_candidate_base_price", np.nan)) and float(row.get("rank_log_candidate_base_price", np.nan)) >= 0.90
            and pd.notna(row.get("rank_d1_close_vwap_pct", np.nan)) and float(row.get("rank_d1_close_vwap_pct", np.nan)) < 0.60
            and pd.notna(row.get("inter_close_low", np.nan)) and float(row.get("inter_close_low", np.nan)) < 0.85)


def metrics(frame, selected_codes, top_n):
    s = ctx_for_codes(frame, selected_codes)
    targets = s[TARGET].astype(bool).tolist() if not s.empty else []
    hit = int(sum(targets))
    return {
        "hit_count": hit, "all_hit": bool(hit == top_n),
        "avg_high_return": mean(s[HIGH]) if not s.empty else np.nan,
        "avg_realized_return": mean(s[REAL]) if not s.empty else np.nan,
        "rank1_hit": bool(targets[0]) if len(targets) > 0 else False,
        "rank2_hit": bool(targets[1]) if len(targets) > 1 else False,
        "rank3_hit": bool(targets[2]) if len(targets) > 2 else False,
    }


def summarize(daily, top_n):
    rows = []
    for strategy, g in daily.groupby("strategy"):
        hits = pd.to_numeric(g["hit_count"], errors="coerce").fillna(0).astype(int)
        rows.append({
            "strategy": strategy, "date_count": int(g["signal_date"].nunique()),
            "selected_ticket_count": int(len(g) * top_n),
            "top3_target_rate": rate(int(hits.sum()), int(len(g) * top_n)),
            "top3_all_hit_rate": rate(int(g["all_hit"].astype(bool).sum()), len(g)),
            "hit_count_0_days": int((hits == 0).sum()), "hit_count_1_days": int((hits == 1).sum()),
            "hit_count_2_days": int((hits == 2).sum()), "hit_count_3_days": int((hits == 3).sum()),
            "rank1_hit_rate": rate(int(g["rank1_hit"].astype(bool).sum()), len(g)),
            "rank2_hit_rate": rate(int(g["rank2_hit"].astype(bool).sum()), len(g)),
            "rank3_hit_rate": rate(int(g["rank3_hit"].astype(bool).sum()), len(g)),
            "avg_top3_high_return": mean(g["avg_high_return"]),
            "avg_top3_realized_return": mean(g["avg_realized_return"]),
            "fallback_days": int(g["action"].astype(str).str.contains("fallback").sum()),
            "risk_replace_days": int(g["action"].astype(str).str.contains("risk_replace").sum()),
            "changed_days": int(g["gate_triggered"].astype(bool).sum()),
            "changed_from_baseline_days": int((g["selected_codes"].astype(str) != g["baseline_v005_codes"].astype(str)).sum()),
            "negative_realized_days": int((pd.to_numeric(g["avg_realized_return"], errors="coerce") < 0).sum()),
            "v002_all_hit_captured_days": int((g["v002_all_hit"].astype(bool) & g["all_hit"].astype(bool)).sum()),
        })
    return pd.DataFrame(rows).sort_values(["top3_all_hit_rate", "hit_count_0_days", "avg_top3_realized_return", "top3_target_rate"], ascending=[False, True, False, False])


def replacement_detail(strategy, date, action, final, baseline, v002, v004a, day_ctx):
    final_s, base_s, v002_s, v004a_s = map(set, [final, baseline, v002, v004a])
    rows = []
    for code in sorted(final_s | base_s | v002_s):
        item = day_ctx[day_ctx["code"].astype(str) == code].head(1)
        row = {} if item.empty else item.iloc[0].to_dict()
        inf, inb, inv2, inv4 = code in final_s, code in base_s, code in v002_s, code in v004a_s
        bucket = "final_baseline_v002_overlap" if inf and inb and inv2 else "final_baseline_only" if inf and inb else "final_v002_replacement" if inf and inv2 else "final_only" if inf else "dropped_baseline" if inb else "unused_v002" if inv2 else "other"
        rows.append({
            "strategy": strategy, "signal_date": date, "action": action, "selection_bucket": bucket, "code": code,
            "in_final": inf, "in_baseline_v005": inb, "in_v002": inv2, "in_v004a": inv4,
            TARGET: bool(row.get(TARGET, False)), HIGH: row.get(HIGH, np.nan), REAL: row.get(REAL, np.nan),
            "v004a_model_rank": row.get("v004a_model_rank", np.nan), "v002_model_rank": row.get("v002_model_rank", np.nan),
            "v004a_score": row.get("v004a_score", np.nan), "v002_score": row.get("v002_score", np.nan),
            "rank_log_candidate_base_price": row.get("rank_log_candidate_base_price", np.nan),
            "rank_d1_close_vwap_pct": row.get("rank_d1_close_vwap_pct", np.nan), "inter_close_low": row.get("inter_close_low", np.nan),
            "rank_total_score": row.get("rank_total_score", np.nan), "extreme_price": bool(row.get("extreme_price", False)),
            "extreme_vwap": bool(row.get("extreme_vwap", False)), "extreme_close_low": bool(row.get("extreme_close_low", False)),
            "near_miss_5_7": bool(row.get("near_miss_5_7", False)),
        })
    return rows


def ctx_for_codes(frame, codes):
    ordered = norm(codes)
    out = frame[frame["code"].astype(str).isin(ordered)].copy()
    order = {c: i for i, c in enumerate(ordered)}
    if out.empty:
        return out
    out["_order"] = out["code"].map(order)
    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def parse_codes(x):
    if isinstance(x, (list, tuple, set)):
        return norm(list(x))
    if pd.isna(x):
        return []
    return norm([p for p in str(x).split(",") if p.strip()])


def norm(codes):
    out = []
    for c in codes:
        c = str(c).strip().zfill(6)
        if c and c not in out:
            out.append(c)
    return out


def bool_series(s):
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def rate(n, d):
    return np.nan if d <= 0 else float(n) / float(d)


def mean(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    s = s[np.isfinite(s)]
    return np.nan if s.empty else float(s.mean())


def make_report(scored, v005_dir, out_dir, objective, summary, daily, repl):
    lines = [
        "# v005 fallback gate diagnostics", "", "## Scope", "",
        "Converged research-only diagnostic for the v005 state-aware fallback policy.", "",
        "## Configuration", "", f"- scored file: `{scored}`", f"- v005 dir: `{v005_dir}`",
        f"- output dir: `{out_dir}`", f"- baseline objective: `{objective}`", f"- primary policy: `{PRIMARY_POLICY}`", "",
        "## Converged policy", "",
        "Use v005 `realized_then_all_hit` by default. Fallback to v002 Top3 only when v002 shows either:",
        "", "1. extreme confirmation: `v002_extreme_vwap_count >= 2` and `v002_extreme_close_low_count >= 2`; or",
        "2. close-low dominance: `v002_extreme_close_low_count >= 3`;",
        "", "and v005 is weak versus v002: `v005_avg_v002_rank >= 12`.", "",
        "## Strategy definitions", "",
        "- `baseline_v005_realized`: keep v005 realized objective selections.",
        f"- `{PRIMARY_POLICY}`: clean candidate policy; fallback by the converged v002 regime rule only.",
        "- `risk_replace_only`: only replace catastrophic risk tickets with v002 names; retained as a control.", "",
        "## Summary", "",
    ]
    lines += md_table(summary, SUMMARY_COLUMNS)
    lines += ["", "## Daily selections", ""] + md_table(daily, DAILY_COLUMNS)
    lines += ["", "## Replacement rows", ""] + md_table(repl, REPLACEMENT_COLUMNS)
    return "\n".join(lines)


def md_table(df, cols):
    if df is None or df.empty:
        return ["_No rows._"]
    use = [c for c in cols if c in df.columns]
    rows = ["| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for _, r in df[use].iterrows():
        rows.append("| " + " | ".join(fmt(r[c]) for c in use) + " |")
    return rows


def fmt(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4f}"
    return str(x).replace("|", "\\|")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run v005 fallback gate diagnostics.")
    parser.add_argument("--scored-file", default=str(DEFAULT_SCORED_FILE))
    parser.add_argument("--v005-dir", default=str(DEFAULT_V005_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    args = parser.parse_args(argv)
    summary, daily, repl, report = run_fallback_gate(
        args.scored_file, args.v005_dir, args.output_dir, args.objective, args.top_n,
        args.v004a_l2, args.v004a_positive_weight,
    )
    print(f"strategy rows: {len(summary)}")
    print(f"daily rows: {len(daily)}")
    print(f"replacement rows: {len(repl)}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best strategy: {best['strategy']}")
        print(f"best top3_all_hit_rate: {float(best['top3_all_hit_rate']):.4f}")
        print(f"best avg_realized_return: {float(best['avg_top3_realized_return']):.4f}")
    print(f"markdown: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

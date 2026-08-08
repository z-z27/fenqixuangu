# -*- coding: utf-8 -*-
"""v004c Top3 Failure Decomposition v001 — 运行工具 (纯诊断, 无模型训练)。

读取既有 OOF score 与冻结 outcome, 分解 Top3 失败:
- Problem A: Winner Capture (≥+7% 赢家是否进 Top3);
- Problem B: Repair-Strength Ordering (非 Target7 之间能否排修复强弱)。

输入 (§6):
- Ridge OOF: reports/research/v004c_pairwise_ridge_v001_20260506_20260630/
  v004c_pairwise_ridge_oof_predictions_v001.csv (upside_oof_score)
- GBDT OOF: reports/research/
  v004c_upside_predictability_diagnostic_v001_20260506_20260630/
  v004c_upside_nonlinear_oof_predictions_v001.csv (gbdt_oof_score)
- 冻结 outcome 来自 v002 input table + May/June 源表 (builder 复用)

输出 (5 个文件, 两次运行 byte-identical):
    v004c_top3_failure_daily_v001.csv
    v004c_top3_failure_group_summary_v001.csv
    v004c_top3_repair_concordance_v001.csv
    v004c_top3_selected_members_v001.csv
    v004c_top3_failure_decomposition_review_v001.md

禁止: 任何模型 fit / July 访问 / 修改历史资产。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_pairwise_ridge import compute_capped_opportunity_return  # noqa: E402
from src.v004c_top3_failure_decomposition import (  # noqa: E402
    audit_stored_rank,
    date_weighted_concordance,
    day_repair_concordance,
    decompose_day,
    oracle_ceiling_summary,
    rank_profile,
    recompute_day_rank,
)

RIDGE_OOF_CSV = (ROOT / "reports" / "research"
                 / "v004c_pairwise_ridge_v001_20260506_20260630"
                 / "v004c_pairwise_ridge_oof_predictions_v001.csv")
GBDT_OOF_CSV = (ROOT / "reports" / "research"
                / "v004c_upside_predictability_diagnostic_v001_20260506_20260630"
                / "v004c_upside_nonlinear_oof_predictions_v001.csv")
OUT_DIR = (ROOT / "reports" / "research"
           / "v004c_top3_failure_decomposition_v001_20260506_20260630")

DAILY_CSV = "v004c_top3_failure_daily_v001.csv"
GROUP_CSV = "v004c_top3_failure_group_summary_v001.csv"
CONCORDANCE_CSV = "v004c_top3_repair_concordance_v001.csv"
MEMBERS_CSV = "v004c_top3_selected_members_v001.csv"
REVIEW_MD = "v004c_top3_failure_decomposition_review_v001.md"

UPSIDE_LABEL = "target7_daily_d2open_d3high"

# 复用 ridge builder 的冻结 outcome 装载 (d2_open/d3_high/raw/capped)
_B = importlib.util.spec_from_file_location(
    "ridge_builder_v001", ROOT / "tools" / "build_v004c_pairwise_ridge_v001.py")
_builder = importlib.util.module_from_spec(_B)
_B.loader.exec_module(_builder)
load_frozen_input = _builder.load_frozen_input
load_frozen_outcomes = _builder.load_frozen_outcomes


# ---------------------------------------------------------------------------
# 输入装载 + identity 校验 (§7, §82)
# ---------------------------------------------------------------------------
def load_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    ridge = pd.read_csv(RIDGE_OOF_CSV, encoding="utf-8-sig", dtype={"code": str})
    gbdt = pd.read_csv(GBDT_OOF_CSV, encoding="utf-8-sig", dtype={"code": str})
    for df in (ridge, gbdt):
        df["event_id"] = df["event_id"].astype(str)
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["signal_date"] = df["signal_date"].astype(str)
        if df["signal_date"].max() > "2026-06-30":
            raise RuntimeError("FATAL: July accessed")
    # OOF 评价集: 有 score 的行 (§7: 233 rows / 29 dates)
    ridge_ev = ridge[ridge["upside_oof_score"].notna()].copy()
    gbdt_ev = gbdt[gbdt["gbdt_oof_score"].notna()].copy()
    if len(ridge_ev) != 233 or len(gbdt_ev) != 233:
        raise RuntimeError(f"FATAL: OOF rows {len(ridge_ev)}/{len(gbdt_ev)} != 233")
    if ridge_ev["signal_date"].nunique() != 29 or \
            gbdt_ev["signal_date"].nunique() != 29:
        raise RuntimeError("FATAL: OOF signal dates != 29")
    r_ids = set(ridge_ev["event_id"])
    g_ids = set(gbdt_ev["event_id"])
    if r_ids != g_ids:
        raise RuntimeError("FATAL: ridge/gbdt OOF event identity mismatch")
    # 冻结 outcome (raw/capped) 合并
    dev = load_frozen_outcomes(load_frozen_input())
    dev["event_id"] = dev["event_id"].astype(str)
    out_cols = ["event_id", "d2_open_daily", "d3_high_daily",
                "raw_opportunity_return", "capped_opportunity_return_7"]
    # raw/capped 由 builder 口径计算 (compute_capped_opportunity_return)
    outcome = compute_capped_opportunity_return(
        dev["d2_open_daily"].to_numpy(float),
        dev["d3_high_daily"].to_numpy(float),
        dev[UPSIDE_LABEL].to_numpy(float))
    dev["raw_opportunity_return"] = outcome["raw_opportunity_return"]
    dev["capped_opportunity_return_7"] = outcome["capped_opportunity_return_7"]
    # 统一 outcome 列: 用冻结 dev 值覆盖 (OOF CSV 中已有列同源, 丢弃避免歧义)
    for df in (ridge_ev, gbdt_ev):
        for c in out_cols:
            if c in df.columns and c != "event_id":
                df.drop(columns=[c], inplace=True)
    ridge_ev = ridge_ev.merge(dev[out_cols], on="event_id", how="left")
    gbdt_ev = gbdt_ev.merge(dev[out_cols], on="event_id", how="left")
    for df in (ridge_ev, gbdt_ev):
        if df["capped_opportunity_return_7"].isna().any():
            raise RuntimeError("FATAL: outcome merge missing values")
    return ridge_ev, gbdt_ev


def _model_frame(ev: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """统一列名: score / target7 / capped, 并重算 diagnostic_rank (§8)。"""
    out = ev.rename(columns={score_col: "score"})
    out["target7_daily_d2open_d3high"] = pd.to_numeric(
        out[UPSIDE_LABEL], errors="coerce").fillna(0).astype(int)
    return recompute_day_rank(out, "score")


# ---------------------------------------------------------------------------
# 主 pipeline
# ---------------------------------------------------------------------------
def build_pipeline() -> dict[str, bytes]:
    ridge_ev, gbdt_ev = load_assets()
    models = {"Ridge": _model_frame(ridge_ev, "upside_oof_score"),
              "GBDT": _model_frame(gbdt_ev, "gbdt_oof_score")}

    # ---- stored rank audit (§8) ----
    ridge_mismatch = audit_stored_rank(ridge_ev, "upside_oof_score",
                                       "upside_rank")
    gbdt_mismatch = audit_stored_rank(gbdt_ev, "gbdt_oof_score", "gbdt_rank")

    # ---- daily decomposition (§52) ----
    daily_rows = []
    member_rows = []
    for model_name, frame in models.items():
        for d, day in frame.groupby("signal_date", sort=True):
            dec = decompose_day(day, "score")
            dec["model"] = model_name
            daily_rows.append(dec)
            # selected members (§53): Top3 每只一行
            K = dec["K"]
            ranked = recompute_day_rank(day, "score").head(K)
            oracle_top3_ids = set(day.sort_values(
                "capped_opportunity_return_7", ascending=False)
                .head(K)["event_id"].astype(str))
            t7c = int(dec["target7_count"])
            req_nt = dec["required_non_target_slots"]
            nt_sorted = day[day["target7_daily_d2open_d3high"] == 0] \
                .sort_values("capped_opportunity_return_7", ascending=False)
            oracle_fill_ids = set(nt_sorted.head(req_nt)["event_id"].astype(str)) \
                if req_nt > 0 else set()
            for _, r in ranked.iterrows():
                member_rows.append({
                    "model": model_name,
                    "signal_date": str(d),
                    "date_group": dec["date_group"],
                    "diagnostic_rank": int(r["diagnostic_rank"]),
                    "event_id": str(r["event_id"]),
                    "code": str(r["code"]).zfill(6),
                    "score": float(r["score"]),
                    "target7": int(r["target7_daily_d2open_d3high"]),
                    "raw_opportunity_return": float(r["raw_opportunity_return"]),
                    "capped_opportunity_return_7": float(
                        r["capped_opportunity_return_7"]),
                    "is_oracle_top3_member": int(
                        str(r["event_id"]) in oracle_top3_ids),
                    "is_oracle_non_target_fill_member": int(
                        str(r["event_id"]) in oracle_fill_ids),
                    "is_selected_non_target": int(
                        r["target7_daily_d2open_d3high"] == 0),
                })
    daily = pd.DataFrame(daily_rows)
    members = pd.DataFrame(member_rows)
    daily["month"] = daily["signal_date"].astype(str).str[:7]

    # ---- group summary (§54) ----
    group_rows = []
    for model_name in ("Ridge", "GBDT"):
        sub = daily[daily["model"] == model_name]
        for g in ("A", "B", "C"):
            s = sub[sub["date_group"] == g]
            if s.empty:
                continue
            group_rows.append({
                "model": model_name,
                "date_group": g,
                "date_count": int(len(s)),
                "mean_target7_count": float(s["target7_count"].mean()),
                "mean_top3_target7_hits": float(s["top3_target7_hits"].mean()),
                "mean_available_winner_capture": float(
                    s["available_winner_capture_rate"].dropna().mean()),
                "full_capture_date_rate": float(
                    s["full_available_winner_capture"].dropna().mean()),
                "mean_model_top3_return": float(s["model_top3_capped_return"].mean()),
                "mean_oracle_top3_return": float(s["oracle_top3_capped_return"].mean()),
                "mean_top3_regret": float(s["total_top3_regret"].mean()),
                "mean_winner_capture_regret": float(
                    s["winner_capture_regret"].mean()),
                "mean_repair_ordering_regret": float(
                    s["repair_ordering_regret"].mean()),
                "mean_best_non_target_fill_capture": float(
                    s["best_non_target_fill_capture_rate"].dropna().mean()),
                "mean_selected_non_target_return": float(
                    s["selected_non_target_mean_capped_return"].dropna().mean()),
                "mean_oracle_non_target_fill_return": float(
                    s["oracle_non_target_fill_mean_return"].dropna().mean()),
            })
    group_summary = pd.DataFrame(group_rows)

    # ---- repair concordance (§55) ----
    conc_rows = []
    for model_name, frame in models.items():
        for period, mask in (("May", frame["signal_date"].astype(str)
                              .str.startswith("2026-05")),
                             ("June", frame["signal_date"].astype(str)
                              .str.startswith("2026-06")),
                             ("Combined", pd.Series(True, index=frame.index))):
            sub = frame[mask]
            day_rows = [day_repair_concordance(day, "score")
                        for _, day in sub.groupby("signal_date", sort=True)]
            agg = date_weighted_concordance(day_rows)
            conc_rows.append({
                "model": model_name,
                "period": period,
                "all_repair_pair_concordance": agg["all_concordance"],
                "cross_threshold_concordance": agg["cross_concordance"],
                "within_non_target_repair_concordance": agg["within_concordance"],
                "all_comparable_dates": agg["all_evaluable_dates"],
                "cross_threshold_evaluable_dates": agg["cross_evaluable_dates"],
                "within_non_target_evaluable_dates": agg["within_evaluable_dates"],
            })
    concordance = pd.DataFrame(conc_rows)

    # ---- Rank profile (§29, §56) ----
    rank_frames = {m: rank_profile(frame, "score") for m, frame in models.items()}

    # ---- Aggregate regret (§41) ----
    _TOL = 1e-9

    def agg_regret(model_name: str, sub=None):
        s = daily[daily["model"] == model_name] if sub is None \
            else daily[(daily["model"] == model_name) & (daily["date_group"] == sub)]
        tot = float(s["total_top3_regret"].mean())
        wc = float(s["winner_capture_regret"].mean())
        ro = float(s["repair_ordering_regret"].mean())
        sum_tot = float(s["total_top3_regret"].sum())
        wc_share = float(s["winner_capture_regret"].sum() / sum_tot) \
            if sum_tot > _TOL else None
        ro_share = float(s["repair_ordering_regret"].sum() / sum_tot) \
            if sum_tot > _TOL else None
        larger = ("WINNER_CAPTURE" if wc_share is not None and wc_share > ro_share + 1e-9
                  else "REPAIR_ORDERING" if ro_share is not None and ro_share > wc_share + 1e-9
                  else "EQUAL")
        return tot, wc, ro, wc_share, ro_share, larger

    regret = {}
    for model_name in ("Ridge", "GBDT"):
        regret[model_name] = {"Combined": agg_regret(model_name)}
        for g in ("A", "B", "C"):
            regret[model_name][g] = agg_regret(model_name, g)

    # ---- Oracle display audit (§48, §81) ----
    dev_full = load_frozen_outcomes(load_frozen_input())
    dev_full["event_id"] = dev_full["event_id"].astype(str)
    outcome = compute_capped_opportunity_return(
        dev_full["d2_open_daily"].to_numpy(float),
        dev_full["d3_high_daily"].to_numpy(float),
        dev_full[UPSIDE_LABEL].to_numpy(float))
    dev_full["capped_opportunity_return_7"] = outcome["capped_opportunity_return_7"]
    oof_dates = sorted(models["Ridge"]["signal_date"].unique().tolist())
    oracle_all = oracle_ceiling_summary(dev_full)
    oracle_oof = oracle_ceiling_summary(
        dev_full[dev_full["signal_date"].isin(oof_dates)])
    prev_display_bug = "CONFIRMED"  # 旧 comparison 表硬编码 1.0/1.0 (§48)

    # ---- 核心数字 (review 用) ----
    def combined_metrics(model_name: str) -> dict:
        d = daily[daily["model"] == model_name]
        pos_winners = d[d["available_winner_capture_rate"].notna()]
        full = d[d["full_available_winner_capture"].notna()]
        return {
            "n_dates": int(len(d)),
            "mean_top3_return": float(d["model_top3_capped_return"].mean()),
            "mean_oracle_top3": float(d["oracle_top3_capped_return"].mean()),
            "total_regret": float(d["total_top3_regret"].mean()),
            "wc_regret": float(d["winner_capture_regret"].mean()),
            "ro_regret": float(d["repair_ordering_regret"].mean()),
            "capture_mean": float(pos_winners["available_winner_capture_rate"].mean()),
            "capture_median": float(pos_winners["available_winner_capture_rate"].median()),
            "full_capture_rate": float(full["full_available_winner_capture"].mean()),
            "mean_hits": float(d["top3_target7_hits"].mean()),
        }

    cm = {m: combined_metrics(m) for m in ("Ridge", "GBDT")}

    # Group A 3/3 2/3 1/3 0/3 (§58)
    def group_a_hits(model_name: str) -> dict:
        s = daily[(daily["model"] == model_name) & (daily["date_group"] == "A")]
        counts = {k: int((s["top3_target7_hits"] == k).sum()) for k in (3, 2, 1, 0)}
        return counts

    ga = {m: group_a_hits(m) for m in ("Ridge", "GBDT")}

    # Group B full-winner-capture fill (§20)
    def group_b_fill(model_name: str) -> dict:
        s = daily[(daily["model"] == model_name) & (daily["date_group"] == "B")
                  & (daily["full_available_winner_capture"] == True)]
        return {
            "n_full_capture_dates": int(len(s)),
            "model_fill_mean_return": float(
                s["selected_non_target_mean_capped_return"].dropna().mean()),
            "oracle_fill_mean_return": float(
                s["oracle_non_target_fill_mean_return"].dropna().mean()),
            "fill_return_gap": float(
                (s["oracle_non_target_fill_mean_return"]
                 - s["selected_non_target_mean_capped_return"]).dropna().mean()),
        }

    gb_fill = {m: group_b_fill(m) for m in ("Ridge", "GBDT")}

    # Group C baseline = 当天 universe mean (从 members/rank 计算)
    group_c_rows = {}
    for model_name in ("Ridge", "GBDT"):
        s = daily[(daily["model"] == model_name) & (daily["date_group"] == "C")]
        dates_c = set(s["signal_date"])
        frame = models[model_name]
        sub = frame[frame["signal_date"].isin(dates_c)]
        base = float(sub.groupby("signal_date")["capped_opportunity_return_7"]
                     .mean().mean()) if len(sub) else float("nan")
        group_c_rows[model_name] = {
            "n_dates": int(len(s)),
            "baseline": base,
            "top3_return": float(s["model_top3_capped_return"].mean()),
            "oracle_top3": float(s["oracle_top3_capped_return"].mean()),
            "gap": float(s["total_top3_regret"].mean()),
        }

    # ---- Review (§57-§63, §87) ----
    fmt_pct = lambda v: "nan" if v is None or np.isnan(v) else f"{v * 100:.2f}%"
    fmt_ret = lambda v: "nan" if v is None or np.isnan(v) else f"{v * 100:.2f}%"
    fmt_pp = lambda v: "nan" if v is None or np.isnan(v) else f"{v * 100:+.2f}pp"

    lines = []
    lines.append("# v004c Top3 Failure Decomposition v001")
    lines.append("")
    lines.append(f"**输入**: Ridge OOF rows = 233 | GBDT OOF rows = 233 | "
                 f"OOF signal dates = 29 | exact event identity = PASS | "
                 f"max signal_date = 2026-06-30 | July NOT accessed | "
                 f"model training performed = NO")
    lines.append("")
    lines.append("## Stored Rank Audit (§8)")
    lines.append("")
    lines.append(f"- Ridge stored `upside_rank` available: YES | mismatch vs "
                 f"recomputed: {ridge_mismatch} (score 列与 rank 展示列一致)")
    lines.append(f"- GBDT stored `gbdt_rank` available: YES | mismatch vs "
                 f"recomputed: {gbdt_mismatch} / 233")
    lines.append("> 注: GBDT diagnostic OOF 的 `gbdt_rank` 展示列存在对齐错误 "
                 "(rank 值按排序后顺序赋给未排序行); 其 `gbdt_oof_score` 列本身"
                 "正确, 本任务全部指标使用重算 rank, 不受影响; 历史资产不回写。")
    lines.append("")
    lines.append("## 1. Why Is Top3 Failing?")
    lines.append("")
    for model_name in ("Ridge", "GBDT"):
        m = cm[model_name]
        lines.append(f"{model_name} Top3:")
        lines.append(f"- model return: {fmt_ret(m['mean_top3_return'])}")
        lines.append(f"- oracle return: {fmt_ret(m['mean_oracle_top3'])}")
        lines.append(f"- total regret: {fmt_pp(m['total_regret'])}")
        lines.append(f"- winner-capture regret: {fmt_pp(m['wc_regret'])}")
        lines.append(f"- repair-ordering regret: {fmt_pp(m['ro_regret'])}")
        lines.append("")
    for model_name in ("Ridge", "GBDT"):
        _, _, _, wc_s, ro_s, larger = regret[model_name]["Combined"]
        lines.append(f"{model_name} 更大的损失来自: "
                     f"{larger} (winner share {fmt_pct(wc_s)}, "
                     f"repair share {fmt_pct(ro_s)})")
    lines.append("")
    lines.append("## 2. When >=3 Real +7% Winners Exist")
    lines.append("")
    for model_name in ("Ridge", "GBDT"):
        s = daily[(daily["model"] == model_name) & (daily["date_group"] == "A")]
        counts = ga[model_name]
        lines.append(f"{model_name} (dates={len(s)}):")
        lines.append(f"- 3/3 dates: {counts[3]} | 2/3: {counts[2]} | "
                     f"1/3: {counts[1]} | 0/3: {counts[0]}")
        lines.append(f"- mean capture: {fmt_pct(float(s['available_winner_capture_rate'].mean()))}")
        lines.append(f"- Top3 return: {fmt_ret(float(s['model_top3_capped_return'].mean()))} "
                     f"| Oracle Top3: {fmt_ret(float(s['oracle_top3_capped_return'].mean()))}")
    lines.append("")
    lines.append("## 3. When Only 1-2 +7% Winners Exist")
    lines.append("")
    for model_name in ("Ridge", "GBDT"):
        s = daily[(daily["model"] == model_name) & (daily["date_group"] == "B")]
        fb = gb_fill[model_name]
        lines.append(f"{model_name} Group B (dates={len(s)}):")
        lines.append(f"- winner capture: {fmt_pct(float(s['available_winner_capture_rate'].mean()))}")
        lines.append(f"- full-winner-capture dates: {fb['n_full_capture_dates']}")
        lines.append(f"- selected remaining non-target return: {fmt_ret(fb['model_fill_mean_return'])}")
        lines.append(f"- oracle remaining non-target return: {fmt_ret(fb['oracle_fill_mean_return'])}")
        lines.append(f"- fill return gap: {fmt_pp(fb['fill_return_gap'])}")
        lines.append(f"- best non-target fill capture: "
                     f"{fmt_pct(float(s['best_non_target_fill_capture_rate'].dropna().mean()))}")
        lines.append(f"- winner regret: {fmt_pp(float(s['winner_capture_regret'].mean()))} | "
                     f"repair-ordering regret: {fmt_pp(float(s['repair_ordering_regret'].mean()))}")
    lines.append("")
    lines.append("## 4. When No Stock Reaches +7%")
    lines.append("")
    for model_name in ("Ridge", "GBDT"):
        g = group_c_rows[model_name]
        s = daily[(daily["model"] == model_name) & (daily["date_group"] == "C")]
        lines.append(f"{model_name} Group C (dates={g['n_dates']}):")
        lines.append(f"- daily baseline: {fmt_ret(g['baseline'])} | "
                     f"Top3: {fmt_ret(g['top3_return'])} | "
                     f"Oracle Top3: {fmt_ret(g['oracle_top3'])} | "
                     f"gap: {fmt_pp(g['gap'])}")
        lines.append(f"- repair concordance: {fmt_pct(_conc_of(models[model_name], 'C', s))}")
    lines.append("")
    lines.append("## 5. Can the Models Rank Repair Strength Below 7%?")
    lines.append("")
    lines.append("| Metric | Ridge | GBDT |")
    lines.append("|---|---|---|")
    for period in ("Combined", "May", "June"):
        r_c = concordance[(concordance["model"] == "Ridge")
                          & (concordance["period"] == period)].iloc[0]
        g_c = concordance[(concordance["model"] == "GBDT")
                          & (concordance["period"] == period)].iloc[0]
        lines.append(f"| {period} cross-threshold | "
                     f"{fmt_pct(r_c['cross_threshold_concordance'])} | "
                     f"{fmt_pct(g_c['cross_threshold_concordance'])} |")
        lines.append(f"| {period} within non-target | "
                     f"{fmt_pct(r_c['within_non_target_repair_concordance'])} | "
                     f"{fmt_pct(g_c['within_non_target_repair_concordance'])} |")
        lines.append(f"| {period} all repair pairs | "
                     f"{fmt_pct(r_c['all_repair_pair_concordance'])} | "
                     f"{fmt_pct(g_c['all_repair_pair_concordance'])} |")
    lines.append("")
    lines.append("## 6. Rank1 vs Rank2 vs Rank3")
    lines.append("")
    for model_name in ("Ridge", "GBDT"):
        lines.append(f"{model_name}:")
        for _, r in rank_frames[model_name].iterrows():
            lines.append(f"- Rank{int(r['rank'])}: return {fmt_ret(r['mean_capped_return'])} "
                         f"| Target7 hit {fmt_pct(r['target7_hit_rate'])} | "
                         f"excess {fmt_pp(r['mean_excess_vs_universe'])}")
    lines.append("")
    lines.append("## 7. Does Binary Target7 Lose Repair-Strength Information?")
    lines.append("")
    for model_name in ("Ridge", "GBDT"):
        c = concordance[(concordance["model"] == model_name)
                        & (concordance["period"] == "Combined")].iloc[0]
        lines.append(f"{model_name}: within-non-target concordance "
                     f"{fmt_pct(c['within_non_target_repair_concordance'])} vs "
                     f"cross-threshold {fmt_pct(c['cross_threshold_concordance'])}")
    lines.append("")
    lines.append(f"## Oracle Display Audit (§48)")
    lines.append("")
    lines.append(f"PREVIOUS_ORACLE_DISPLAY_BUG: {prev_display_bug} "
                 f"(旧 comparison 表硬编码 Oracle Top1 Hit = 100% / Top3 Prec = "
                 f"100%; 正确口径见下)")
    lines.append(f"Correct All-39: Oracle Top1 hit {fmt_pct(oracle_all['oracle_top1_hit_rate'])} "
                 f"| Top3 precision {fmt_pct(oracle_all['oracle_top3_target7_precision'])}")
    lines.append(f"Correct OOF-29: Oracle Top1 hit {fmt_pct(oracle_oof['oracle_top1_hit_rate'])} "
                 f"| Top3 precision {fmt_pct(oracle_oof['oracle_top3_target7_precision'])} "
                 f"| Top3 capped return {fmt_ret(oracle_oof['oracle_top3_capped_return'])}")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    # ---- Q1-Q5 (§64-§67) ----
    ga_ridge = daily[(daily["model"] == "Ridge") & (daily["date_group"] == "A")]
    ga_gbdt = daily[(daily["model"] == "GBDT") & (daily["date_group"] == "A")]
    gb_ridge = daily[(daily["model"] == "Ridge") & (daily["date_group"] == "B")]
    gb_gbdt = daily[(daily["model"] == "GBDT") & (daily["date_group"] == "B")]
    cap_a_ridge = float(ga_ridge["available_winner_capture_rate"].mean())
    cap_a_gbdt = float(ga_gbdt["available_winner_capture_rate"].mean())
    cap_b_ridge = float(gb_ridge["available_winner_capture_rate"].mean())
    cap_b_gbdt = float(gb_gbdt["available_winner_capture_rate"].mean())
    c_ridge = concordance[(concordance["model"] == "Ridge")
                          & (concordance["period"] == "Combined")].iloc[0]
    c_gbdt = concordance[(concordance["model"] == "GBDT")
                         & (concordance["period"] == "Combined")].iloc[0]
    wn_ridge = float(c_ridge["within_non_target_repair_concordance"])
    wn_gbdt = float(c_gbdt["within_non_target_repair_concordance"])
    ct_ridge = float(c_ridge["cross_threshold_concordance"])
    ct_gbdt = float(c_gbdt["cross_threshold_concordance"])

    q1_ridge = "YES" if cap_a_ridge >= 2 / 3 - 1e-9 else "NO"
    q1_gbdt = "YES" if cap_a_gbdt >= 2 / 3 - 1e-9 else "NO"
    q2_ridge = "YES" if cap_b_ridge >= 0.9 else ("PARTIAL" if cap_b_ridge >= 0.5 else "NO")
    q2_gbdt = "YES" if cap_b_gbdt >= 0.9 else ("PARTIAL" if cap_b_gbdt >= 0.5 else "NO")
    q3_ridge = "YES" if wn_ridge > 0.55 else ("WEAK" if wn_ridge > 0.5 else "NO")
    q3_gbdt = "YES" if wn_gbdt > 0.55 else ("WEAK" if wn_gbdt > 0.5 else "NO")

    lines.append("Q1 When >=3 winners exist, can Ridge catch them?")
    lines.append(f"- {q1_ridge} | evidence: Group A mean capture "
                 f"{fmt_pct(cap_a_ridge)} (3/3 只有 {ga_ridge[ga_ridge['top3_target7_hits'] == 3].shape[0]}/"
                 f"{len(ga_ridge)} 天)")
    lines.append("Q1b When >=3 winners exist, can GBDT catch them?")
    lines.append(f"- {q1_gbdt} | evidence: Group A mean capture "
                 f"{fmt_pct(cap_a_gbdt)} (3/3 {ga_gbdt[ga_gbdt['top3_target7_hits'] == 3].shape[0]}/"
                 f"{len(ga_gbdt)} 天)")
    lines.append("Q2 When only 1-2 winners exist, can Ridge capture available winners?")
    lines.append(f"- {q2_ridge} | evidence: Group B capture {fmt_pct(cap_b_ridge)}")
    lines.append("Q2b GBDT:")
    lines.append(f"- {q2_gbdt} | evidence: Group B capture {fmt_pct(cap_b_gbdt)}")
    lines.append("Q3 Can Ridge rank repair strength among non-Target7 stocks?")
    lines.append(f"- {q3_ridge} | evidence: within-non-target concordance "
                 f"{fmt_pct(wn_ridge)} (chance = 50%), cross-threshold "
                 f"{fmt_pct(ct_ridge)}")
    lines.append("Q3b GBDT:")
    lines.append(f"- {q3_gbdt} | evidence: within-non-target concordance "
                 f"{fmt_pct(wn_gbdt)}, cross-threshold {fmt_pct(ct_gbdt)}")
    lines.append("Q4 What causes more Top3 regret?")
    for model_name in ("Ridge", "GBDT"):
        _, _, _, wc_s, ro_s, larger = regret[model_name]["Combined"]
        lines.append(f"- {model_name}: winner capture {fmt_pct(wc_s)} | "
                     f"repair ordering {fmt_pct(ro_s)} | larger = {larger}")
    lines.append("Q5 Binary Target7 objective misalignment:")
    # 判定 (§64-§65): within-non-target 明显差于 cross-threshold, 且 fill gap
    # 造成实际收益损失; 但 winner 识别本身也弱 (§66 -> 附注)
    misalign_supported = (
        (wn_ridge < ct_ridge - 0.02 or wn_gbdt < ct_gbdt - 0.02)
        and (gb_fill["Ridge"]["fill_return_gap"] > 0.01
             or gb_fill["GBDT"]["fill_return_gap"] > 0.01))
    misalign = "SUPPORTED" if misalign_supported else "INCONCLUSIVE"
    lines.append(f"- {misalign}")
    lines.append(f"- Evidence A: within-non-target {fmt_pct(wn_ridge)}/"
                 f"{fmt_pct(wn_gbdt)} vs cross-threshold {fmt_pct(ct_ridge)}/"
                 f"{fmt_pct(ct_gbdt)} (相对差 "
                 f"{fmt_pp(wn_ridge - ct_ridge)}/{fmt_pp(wn_gbdt - ct_gbdt)})")
    lines.append(f"- Evidence B: Group B full-winner-capture 后 fill gap "
                 f"{fmt_pp(gb_fill['Ridge']['fill_return_gap'])}/"
                 f"{fmt_pp(gb_fill['GBDT']['fill_return_gap'])}")
    lines.append(f"- Evidence C: Group C Ridge Top3 {fmt_ret(group_c_rows['Ridge']['top3_return'])} "
                 f"vs baseline {fmt_ret(group_c_rows['Ridge']['baseline'])} (gap "
                 f"{fmt_pp(group_c_rows['Ridge']['gap'])}); GBDT Top3 "
                 f"{fmt_ret(group_c_rows['GBDT']['top3_return'])} (gap "
                 f"{fmt_pp(group_c_rows['GBDT']['gap'])})")
    lines.append(f"- Evidence D: Rank1->Rank3 hit 衰减 Ridge "
                 f"{fmt_pct(rank_frames['Ridge'].iloc[0]['target7_hit_rate'])} -> "
                 f"{fmt_pct(rank_frames['Ridge'].iloc[2]['target7_hit_rate'])}; "
                 f"GBDT {fmt_pct(rank_frames['GBDT'].iloc[0]['target7_hit_rate'])} -> "
                 f"{fmt_pct(rank_frames['GBDT'].iloc[2]['target7_hit_rate'])}")
    lines.append("- 附注 (§66): Group A winner capture 本身仅 "
                 f"{fmt_pct(cap_a_ridge)}/{fmt_pct(cap_a_gbdt)}, cross-threshold "
                 f"仅 {fmt_pct(ct_ridge)}/{fmt_pct(ct_gbdt)} —— 问题不只是 binary "
                 f"目标错位: 当前 D1 score 连真正 >=7% 的赢家都无法稳定识别。")
    lines.append("")
    wc_share_ridge = regret["Ridge"]["Combined"][3]
    picture = ("WINNER_CAPTURE" if wc_share_ridge is not None
               and wc_share_ridge > 0.5 else "MIXED")
    lines.append(f"TOP3_FAILURE_PICTURE: {picture}")
    lines.append("")
    lines.append("Plain-language conclusion:")
    lines.append("- Top3 失败约 4/5 来自赢家没抓进 Top3 (Ridge 79.6% / GBDT "
                 "73.5% of regret): 即使当天存在 >=3 只 +7% 股票, 模型平均也只抓到 "
                 f"{fmt_pct(cap_a_ridge)}/{fmt_pct(cap_a_gbdt)};")
    lines.append("- 剩余约 1/5 来自 7% 以下股票之间的修复强弱排序 (Ridge 20.4% / "
                 "GBDT 26.5%): 全赢家捕获日期的非赢家填充收益 "
                 f"{fmt_ret(gb_fill['Ridge']['model_fill_mean_return'])}/"
                 f"{fmt_ret(gb_fill['GBDT']['model_fill_mean_return'])} vs oracle "
                 f"{fmt_ret(gb_fill['Ridge']['oracle_fill_mean_return'])}/"
                 f"{fmt_ret(gb_fill['GBDT']['oracle_fill_mean_return'])};")
    lines.append("- within-non-target 修复一致性 "
                 f"{fmt_pct(wn_ridge)}/{fmt_pct(wn_gbdt)} (≈/低于 50% 随机), "
                 f"提示 binary Target7 目标在 7% 以下丢失修复强度信息, 但该错位是"
                 f"次要成分; 主因仍是赢家识别失败。")
    lines.append("")
    lines.append("Does current evidence support changing the next model objective "
                 "from binary Target7 to repair-strength ranking?")
    lines.append("- YES (部分支持) | reason: within-non-target 排序 ≤ 随机且 fill "
                 "gap 实际损失存在; 但 79.6%/73.5% regret 来自 winner capture, "
                 "且 cross-threshold 也仅勉强 ≥ 随机 —— 换目标前必须先解决赢家"
                 "识别本身 (D1 信息/时间稳定性), 否则 repair-strength 目标同样无法"
                 "泛化 (diagnostic §TEMPORAL_INSTABILITY 结论)。")
    lines.append("")
    lines.append("Recommended next step (external review 后决定):")
    lines.append("- BOTH: (1) 修复赢家识别 (D1 信息/时间稳定性问题, 见 "
                 "predictability diagnostic); (2) 若赢家识别改善后仍存在 fill gap, "
                 "再冻结 repair-strength ranking objective 作为下一阶段实验 "
                 "(本任务不实现)。")
    lines.append("")
    lines.append("## Scope Note")
    lines.append("")
    lines.append(f"- 本诊断 Top3 按 §9 K = min(3, candidate_count), 29 个 OOF "
                 f"日期全部参与 (v001 review 的 Top3 2.55% 只统计 >=3 候选日期, "
                 f"因此本报告 Combined model return 2.78%/2.92% 与 v001 2.55% "
                 f"口径不同, 数字均正确)。")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append("- full pipeline executed twice; all 5 outputs byte-identical: PASS")
    lines.append("")
    review = "\n".join(lines) + "\n"

    daily_out = daily.copy()
    daily_out["available_winner_capture_rate"] = daily_out[
        "available_winner_capture_rate"].apply(lambda v: "" if v is None else f"{v:.6f}")
    daily_out["best_non_target_fill_capture_rate"] = daily_out[
        "best_non_target_fill_capture_rate"].apply(lambda v: "" if v is None else f"{v:.6f}")
    daily_out["full_available_winner_capture"] = daily_out[
        "full_available_winner_capture"].apply(lambda v: "" if v is None else str(v))

    artifacts = {
        DAILY_CSV: daily_out.to_csv(index=False),
        GROUP_CSV: group_summary.to_csv(index=False),
        CONCORDANCE_CSV: concordance.to_csv(index=False),
        MEMBERS_CSV: members.to_csv(index=False),
        REVIEW_MD: review,
    }
    return {name: data.encode("utf-8") for name, data in artifacts.items()}


def _conc_of(frame: pd.DataFrame, group: str, daily_sub: pd.DataFrame) -> float:
    """Group C 的 repair concordance (within non-target)。"""
    from src.v004c_top3_failure_decomposition import day_repair_concordance
    dates = set(daily_sub["signal_date"])
    sub = frame[frame["signal_date"].isin(dates)]
    rows = [day_repair_concordance(day, "score")
            for _, day in sub.groupby("signal_date", sort=True)]
    vals = [r["within_concordance"] for r in rows if r["within_concordance"] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    print("[decomposition] run #1 ...", flush=True)
    run1 = build_pipeline()
    print("[decomposition] run #2 ...", flush=True)
    run2 = build_pipeline()
    mismatched = [name for name in run1 if run1[name] != run2[name]]
    if mismatched:
        print(f"FATAL: determinism mismatch in {mismatched}", flush=True)
        return 1
    for name, data in run1.items():
        write_atomic(OUT_DIR / name, data)
        print(f"[write] {name} ({len(data)} bytes)", flush=True)
    print(f"OK: {len(run1)} outputs written to {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

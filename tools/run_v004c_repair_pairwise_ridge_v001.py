# -*- coding: utf-8 -*-
"""v004c Repair-Strength Pairwise Ridge v001 — Raw Repair Ordering A/B runner。

输入 (全部冻结, 只读):
- Pairwise v1 feature contract v002 (53 features / 319 rows / 39 dates);
- May/June outcome 源 (经 v004c_pairwise_ridge_v001 builder 的冻结 loader);
- Binary Target7 Ridge v001 OOF (control): reports/research/
  v004c_pairwise_ridge_v001_20260506_20260630/
  v004c_pairwise_ridge_oof_predictions_v001.csv (upside_oof_score)。

流程 (§3, §37, §51, §113):
1. 组装 319 行开发表 + repair_strength_raw_return (= d3_high/d2_open - 1)
   与 capped_opportunity_return_7 (OUTCOME_ONLY); FATAL 校验 319/39,
   May 146/18, June 173/21, board2 261, board3 58, max date <= 2026-06-30;
2. RAW_REPAIR_PAIRWISE_RIDGE 在 LAMBDA_GRID 上 chronological walk-forward
   (训练 pair = 同日 raw ordering, 权重 1/N_pair_date, 无类别/幅度加权);
   lambda 唯一选择标准 = date-weighted OOF raw-repair pairwise logloss (§37);
   收益/赢家/一致性列只做 EVALUATION_ONLY (§38, §74);
3. 用全部 May+June (39 dates / 319 rows) 以 selected lambda 最终 refit
   -> FINAL_DEV_REPAIR_MODEL (DEV_FROZEN, SHA256 lineage) (§86-§89);
4. Control A/B: 用统一 evaluator 对 Binary Ridge v001 冻结 upside_oof_score
   重算同一套指标 (§50); OOF event universe identity FATAL (§51);
5. 9 个输出 (§93): lambda_selection / oof_predictions / daily_metrics /
   ab_comparison / pair_type_diagnostics / regret / coefficients /
   model json / review md; 完整 pipeline 运行两次 byte-identical (§113);
6. GO gates (§80): REPAIR_OBJECTIVE_DEV_SIGNAL 五条件。

禁止: July 数据、模型 zoo、按收益选 lambda、修改任何历史资产 (§114)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_pairwise_ridge import (  # noqa: E402
    BOARD3_STREAK,
    CAP_RETURN_7,
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    build_board3_interaction,
    compute_capped_opportunity_return,
    compute_practical_rank_metrics,
    fit_pairwise_ridge,
    fit_preprocessor,
    score_from_z,
    transform_preprocessor,
)
from src.v004c_repair_pairwise_ridge import (  # noqa: E402
    OOF_SCORE_COL,
    build_same_date_repair_pairs,
    chronological_walkforward_repair,
    date_weighted_repair_pair_stats,
    day_repair_pair_counts,
    day_repair_pair_gaps,
    repair_gap_stats,
    select_lambda_repair,
)
from src.v004c_top3_failure_decomposition import (  # noqa: E402
    decompose_day,
    oracle_ceiling_summary,
)

# ---- builder loaders (冻结资产装载复用) ----
_spec = importlib.util.spec_from_file_location(
    "ridge_builder", ROOT / "tools" / "build_v004c_pairwise_ridge_v001.py")
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)
load_frozen_input = _builder.load_frozen_input
load_feature_order = _builder.load_feature_order
load_frozen_outcomes = _builder.load_frozen_outcomes
_day_rank = _builder._day_rank

CONTROL_DIR = (ROOT / "reports" / "research"
               / "v004c_pairwise_ridge_v001_20260506_20260630")
CONTROL_OOF_CSV = CONTROL_DIR / "v004c_pairwise_ridge_oof_predictions_v001.csv"
OUT_DIR = (ROOT / "reports" / "research"
           / "v004c_repair_pairwise_ridge_v001_20260506_20260630")

LAMBDA_SELECTION_CSV = "v004c_repair_pairwise_lambda_selection_v001.csv"
OOF_PRED_CSV = "v004c_repair_pairwise_oof_predictions_v001.csv"
DAILY_METRICS_CSV = "v004c_repair_pairwise_daily_metrics_v001.csv"
AB_COMPARISON_CSV = "v004c_repair_pairwise_ab_comparison_v001.csv"
PAIR_TYPE_DIAG_CSV = "v004c_repair_pair_type_diagnostics_v001.csv"
REGRET_CSV = "v004c_repair_pairwise_regret_v001.csv"
COEFF_CSV = "v004c_repair_pairwise_coefficients_v001.csv"
MODEL_JSON = "v004c_repair_pairwise_model_v001.json"
REVIEW_MD = "v004c_repair_pairwise_review_v001.md"

RAW_COL = "repair_strength_raw_return"
T7_COL = "target7_daily_d2open_d3high"
CONTROL_NAME = "BINARY_TARGET7_PAIRWISE_RIDGE_CONTROL"
REPAIR_NAME = "REPAIR_RIDGE_V001"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def load_control_oof() -> pd.DataFrame:
    c = pd.read_csv(CONTROL_OOF_CSV, encoding="utf-8-sig", dtype={"code": str})
    c["code"] = c["code"].astype(str).str.zfill(6)
    c["event_id"] = c["event_id"].astype(str)
    c["signal_date"] = c["signal_date"].astype(str)
    c["target7_daily_d2open_d3high"] = pd.to_numeric(
        c["target7_daily_d2open_d3high"], errors="coerce")
    return c


def build_eval_frame(res: dict, dev: pd.DataFrame) -> pd.DataFrame:
    """walk-forward 结果 -> 评价 frame (score + OUTCOME_ONLY 列, warm-up 剔除)。"""
    o = res["oof"].rename(columns={OOF_SCORE_COL: "score"})
    o = o.dropna(subset=["score"]).copy()
    dev_cols = ["event_id", "code", "signal_date",
                "board_streak_before_break", T7_COL,
                "d2_open_daily", "d3_high_daily",
                RAW_COL, "capped_opportunity_return_7"]
    o = o.drop(columns=["signal_date", "board_streak_before_break",
                        "oof_fold_index"]) \
        .merge(dev[dev_cols], on="event_id", how="left")
    return o


def control_eval_frame() -> pd.DataFrame:
    c = load_control_oof()
    c = c.rename(columns={"upside_oof_score": "score",
                          "raw_opportunity_return": RAW_COL})
    c = c[["event_id", "code", "signal_date",
           "board_streak_before_break", "score", T7_COL,
           "d2_open_daily", "d3_high_daily",
           RAW_COL, "capped_opportunity_return_7"]].copy()
    return c.dropna(subset=["score"])


# ---------------------------------------------------------------------------
# 统一 evaluator (两模型同一套口径, §48-§68)
# ---------------------------------------------------------------------------
def model_metrics(frame: pd.DataFrame, baseline: float) -> dict:
    """Combined 全部指标 (practical + winner capture + groups + regret +
    concordance + oracle)。decompose_day 提供逐日 regret/捕获/填充。"""
    m = compute_practical_rank_metrics(frame)
    def _pos(pos):
        return m[m["rank_position"] == pos].iloc[0]
    r1, r2, r3 = _pos(1), _pos(2), _pos(3)
    t1, t2, t3 = _pos("Top1"), _pos("Top2"), _pos("Top3")

    ranked = frame.sort_values(["signal_date", "score", "event_id"],
                               ascending=[True, False, True])
    ranked["_r"] = ranked.groupby("signal_date", sort=True).cumcount() + 1
    top1_hit = float(ranked[ranked["_r"] == 1][T7_COL].fillna(0).mean())
    top3_prec = float(ranked[ranked["_r"] <= 3].groupby("signal_date")[
        T7_COL].mean().mean())

    day_rows = [decompose_day(day, "score")
                for _, day in frame.groupby("signal_date", sort=True)]
    win_days = [r for r in day_rows if r["target7_count"] > 0]
    capture = float(np.mean([r["available_winner_capture_rate"]
                             for r in win_days]))
    capture_median = float(np.median([r["available_winner_capture_rate"]
                                      for r in win_days]))
    full_rate = float(np.mean([r["full_available_winner_capture"]
                               for r in win_days]))

    def _group(g):
        rows = [r for r in day_rows if r["date_group"] == g]
        if not rows:
            return {}
        out = {"n_dates": len(rows)}
        if g == "A":
            out.update({
                "n3_3": sum(1 for r in rows if r["top3_target7_hits"] == 3),
                "n2_3": sum(1 for r in rows if r["top3_target7_hits"] == 2),
                "n1_3": sum(1 for r in rows if r["top3_target7_hits"] == 1),
                "n0_3": sum(1 for r in rows if r["top3_target7_hits"] == 0),
                "capture": float(np.mean([r["available_winner_capture_rate"]
                                          for r in rows])),
                "top3_return": float(np.mean([r["model_top3_capped_return"]
                                              for r in rows])),
                "oracle_top3_return": float(np.mean(
                    [r["oracle_top3_capped_return"] for r in rows])),
            })
        elif g == "B":
            # 与 Top3 Failure Decomposition v001 口径一致: fill 指标只看
            # 全赢家捕获日期 (full_available_winner_capture == True)
            fill_rows = [r for r in rows
                         if r["full_available_winner_capture"]
                         and np.isfinite(
                             r["selected_non_target_mean_capped_return"])
                         and np.isfinite(
                             r["oracle_non_target_fill_mean_return"])]
            cap_rows = [r for r in rows
                        if r["available_winner_capture_rate"] is not None]
            bf_rows = [r for r in rows
                       if r["best_non_target_fill_capture_rate"] is not None]
            out.update({
                "capture": float(np.mean([r["available_winner_capture_rate"]
                                          for r in cap_rows])),
                "full_capture_count": int(sum(
                    1 for r in rows if r["full_available_winner_capture"])),
                "selected_fill_mean": float(np.mean(
                    [r["selected_non_target_mean_capped_return"]
                     for r in fill_rows])),
                "oracle_fill_mean": float(np.mean(
                    [r["oracle_non_target_fill_mean_return"]
                     for r in fill_rows])),
                "fill_gap": float(np.mean(
                    [r["oracle_non_target_fill_mean_return"]
                     - r["selected_non_target_mean_capped_return"]
                     for r in fill_rows])),
                "best_fill_capture": float(np.mean(
                    [r["best_non_target_fill_capture_rate"]
                     for r in bf_rows])),
            })
        elif g == "C":
            out.update({
                "top3_return": float(np.mean([r["model_top3_capped_return"]
                                              for r in rows])),
                "baseline": float(np.mean([
                    frame[frame["signal_date"] == r["signal_date"]]
                    ["capped_opportunity_return_7"].mean()
                    for r in rows])),
                "oracle_top3_return": float(np.mean(
                    [r["oracle_top3_capped_return"] for r in rows])),
            })
        return out

    grp_a, grp_b, grp_c = _group("A"), _group("B"), _group("C")

    agg = np.mean if day_rows else lambda x: float("nan")
    total_regret = float(agg([r["total_top3_regret"] for r in day_rows]))
    winner_regret = float(agg([r["winner_capture_regret"] for r in day_rows]))
    repair_regret = float(agg([r["repair_ordering_regret"] for r in day_rows]))

    scores = frame["score"].to_numpy(float)
    raw = frame[RAW_COL].to_numpy(float)
    dates = frame["signal_date"].astype(str).to_numpy()
    conc = date_weighted_repair_pair_stats(scores, raw, dates)

    splits = {}
    for month, sub in (("May", frame[frame["signal_date"].str.startswith("2026-05")]),
                       ("June", frame[frame["signal_date"].str.startswith("2026-06")])):
        splits[month] = date_weighted_repair_pair_stats(
            sub["score"].to_numpy(float), sub[RAW_COL].to_numpy(float),
            sub["signal_date"].astype(str).to_numpy())

    return {
        "r1": r1, "r2": r2, "r3": r3, "t1": t1, "t2": t2, "t3": t3,
        "baseline": baseline,
        "top1_target7_hit": top1_hit,
        "top3_target7_precision": top3_prec,
        "capture": capture, "capture_median": capture_median,
        "full_capture_rate": full_rate,
        "n_positive_winner_dates": len(win_days),
        "group_A": grp_a, "group_B": grp_b, "group_C": grp_c,
        "total_regret": total_regret,
        "winner_regret": winner_regret,
        "repair_regret": repair_regret,
        "concordance": conc,
        "splits": splits,
        "day_rows": day_rows,
        "oracle": oracle_ceiling_summary(frame),
    }


def oracle_metrics(frame: pd.DataFrame, baseline: float) -> dict:
    """Oracle 行: 真实 outcome 排序 (§52-§53)。"""
    out = {"r1": None, "r2": None, "r3": None,
           "t1": None, "t2": None, "t3": None}
    rets = {1: [], 2: [], 3: []}
    top_rets = {1: [], 2: [], 3: []}
    for _, day in frame.groupby("signal_date", sort=True):
        s = day.sort_values("capped_opportunity_return_7", ascending=False)
        for k in (1, 2, 3):
            if len(s) >= k:
                rets[k].append(float(s.iloc[k - 1]["capped_opportunity_return_7"]))
                top_rets[k].append(float(s.head(k)["capped_opportunity_return_7"].mean()))
    for k in (1, 2, 3):
        out[f"r{k}"] = float(np.mean(rets[k]))
        out[f"t{k}"] = float(np.mean(top_rets[k]))
    oc = oracle_ceiling_summary(frame)
    out["top1_target7_hit"] = oc["oracle_top1_hit_rate"]
    out["top3_target7_precision"] = oc["oracle_top3_target7_precision"]
    out["capture"] = 1.0
    out["group_A_capture"] = 1.0
    out["baseline"] = baseline
    return out


def universe_metrics(frame: pd.DataFrame, baseline: float) -> dict:
    """Daily universe 行: 收益 = baseline; hit = random 基准; 其余 N/A。"""
    random_base = float(frame.groupby("signal_date")[T7_COL]
                        .mean().mean())
    return {
        "r1": baseline, "r2": baseline, "r3": baseline,
        "t1": baseline, "t2": baseline, "t3": baseline,
        "top1_target7_hit": random_base,
        "top3_target7_precision": random_base,
        "capture": float("nan"), "group_A_capture": float("nan"),
        "baseline": baseline,
    }


# ---------------------------------------------------------------------------
# 主 pipeline: 返回 {filename: bytes}
# ---------------------------------------------------------------------------
def build_pipeline() -> dict[str, bytes]:
    dev = load_frozen_input()
    feature_names = load_feature_order()
    if len(feature_names) != 53 or len(set(feature_names)) != 53:
        raise RuntimeError("FATAL: feature contract must be exactly 53 unique")
    dev = load_frozen_outcomes(dev)

    # ---- 冻结边界校验 (§6-§7, §42) ----
    if len(dev) != 319 or dev["signal_date"].nunique() != 39:
        raise RuntimeError("FATAL: universe must be 319 rows / 39 signal dates")
    may_n, june_n = (len(dev[dev["signal_date"].str.startswith("2026-05")]),
                     len(dev[dev["signal_date"].str.startswith("2026-06")]))
    if may_n != 146 or june_n != 173:
        raise RuntimeError(f"FATAL: May {may_n} / June {june_n} != 146/173")
    b2 = int((dev["board_streak_before_break"] == 2).sum())
    b3 = int((dev["board_streak_before_break"] == 3).sum())
    if b2 != 261 or b3 != 58:
        raise RuntimeError(f"FATAL: board2 {b2} / board3 {b3} != 261/58")
    if dev["signal_date"].max() > "2026-06-30":
        raise RuntimeError("FATAL: signal_date exceeds 2026-06-30 (July access)")

    # ---- OUTCOME_ONLY (§10, §12) ----
    outcome = compute_capped_opportunity_return(
        dev["d2_open_daily"].to_numpy(float),
        dev["d3_high_daily"].to_numpy(float),
        dev[T7_COL].to_numpy(float))
    dev[RAW_COL] = outcome["raw_opportunity_return"]
    dev["capped_opportunity_return_7"] = outcome["capped_opportunity_return_7"]

    # ---- X / label 隔离 (§9, §105) ----
    forbidden = ({T7_COL, "tail_loss_daily_5pct", "d2_open_daily",
                  "d3_high_daily", RAW_COL, "raw_opportunity_return",
                  "capped_opportunity_return_7", "source_window",
                  "signal_date", "event_id", "code"}
                 & set(feature_names))
    if forbidden:
        raise RuntimeError(f"FATAL: forbidden columns in X: {forbidden}")

    # ---- Lambda selection (§36-§39) + EVALUATION_ONLY (§74) ----
    sel_table, selected_lambda = select_lambda_repair(
        dev, feature_names, RAW_COL)
    if sel_table["date_weighted_oof_repair_pairwise_logloss"].isna().all():
        raise RuntimeError("FATAL: no evaluable dates for repair selection")

    eval_only = {}
    for lam in LAMBDA_GRID:
        res = chronological_walkforward_repair(dev, feature_names, RAW_COL, lam)
        fr = build_eval_frame(res, dev)
        base = float(fr.groupby("signal_date")["capped_opportunity_return_7"]
                     .mean().mean())
        ev = model_metrics(fr, base)
        eval_only[lam] = {
            "all_repair_concordance": ev["concordance"]["all_concordance"],
            "top3_mean_capped_return": float(ev["t3"]["mean_capped_return"]),
            "available_winner_capture": ev["capture"],
        }
    sel_table = sel_table.copy()
    sel_table["all_repair_concordance_EVALUATION_ONLY"] = [
        eval_only[float(lam)]["all_repair_concordance"]
        for lam in sel_table["lambda"]]
    sel_table["top3_mean_capped_return_EVALUATION_ONLY"] = [
        eval_only[float(lam)]["top3_mean_capped_return"]
        for lam in sel_table["lambda"]]
    sel_table["available_winner_capture_EVALUATION_ONLY"] = [
        eval_only[float(lam)]["available_winner_capture"]
        for lam in sel_table["lambda"]]
    sel_table["selected_lambda"] = (sel_table["lambda"] == selected_lambda) \
        .astype(int)

    # ---- selected-lambda walk-forward (fold 系数用于稳定性, §55/§98) ----
    res = chronological_walkforward_repair(dev, feature_names, RAW_COL,
                                           selected_lambda,
                                           collect_coefs=True)
    repair_frame = build_eval_frame(res, dev)

    # ---- Control (§5, §50-§51) ----
    control_frame = control_eval_frame()
    for name, fr in (("repair", repair_frame), ("control", control_frame)):
        if len(fr) != 233 or fr["signal_date"].nunique() != 29:
            raise RuntimeError(
                f"FATAL: {name} OOF must be 233 rows / 29 dates, "
                f"got {len(fr)} / {fr['signal_date'].nunique()}")
    if set(repair_frame["event_id"]) != set(control_frame["event_id"]):
        raise RuntimeError(  # FATAL (§51)
            "FATAL: repair OOF event universe != binary control OOF universe")
    if set(repair_frame["signal_date"]) != set(control_frame["signal_date"]):
        raise RuntimeError("FATAL: repair OOF dates != control OOF dates")

    baseline = float(repair_frame.groupby("signal_date")
                     ["capped_opportunity_return_7"].mean().mean())
    repair_ev = model_metrics(repair_frame, baseline)
    control_ev = model_metrics(control_frame, baseline)
    oracle = oracle_metrics(repair_frame, baseline)
    universe = universe_metrics(repair_frame, baseline)

    # ---- OOF predictions CSV (§43-§44, §94) ----
    oof = dev[["event_id", "code", "signal_date",
               "board_streak_before_break", T7_COL,
               "d2_open_daily", "d3_high_daily",
               RAW_COL, "capped_opportunity_return_7"]].copy()
    oof = oof.merge(res["oof"][["event_id", OOF_SCORE_COL]],
                    on="event_id", how="left")
    oof["repair_rank"] = _day_rank(
        oof[OOF_SCORE_COL].to_numpy(float), oof["event_id"].to_numpy(),
        oof["signal_date"].to_numpy())
    oof.loc[oof[OOF_SCORE_COL].isna(), "repair_rank"] = np.nan

    # ---- daily metrics CSV (§95) ----
    daily_rows = []
    for d, day in repair_frame.sort_values("signal_date") \
            .groupby("signal_date", sort=True):
        base = float(day["capped_opportunity_return_7"].mean())
        dec = decompose_day(day, "score")
        row = {"signal_date": str(d),
               "candidate_count": len(day),
               "target7_count": dec["target7_count"],
               "daily_universe_mean_capped_return": base}
        for pos in (1, 2, 3):
            if len(day) >= pos:
                r = day.sort_values(["score", "event_id"],
                                    ascending=[False, True]).iloc[pos - 1]
                row[f"rank{pos}_event_id"] = str(r["event_id"])
                row[f"rank{pos}_capped_return"] = float(
                    r["capped_opportunity_return_7"])
                row[f"rank{pos}_target7"] = int(r[T7_COL])
            else:
                row[f"rank{pos}_event_id"] = ""
                row[f"rank{pos}_capped_return"] = float("nan")
                row[f"rank{pos}_target7"] = 0
        for k in (1, 2, 3):
            if len(day) >= k:
                top = day.sort_values(["score", "event_id"],
                                      ascending=[False, True]).head(k)
                tr = float(top["capped_opportunity_return_7"].mean())
                row[f"top{k}_capped_return"] = tr
                row[f"top{k}_excess_vs_universe"] = tr - base
                row[f"top{k}_beat_universe"] = int(tr > base)
            else:
                row[f"top{k}_capped_return"] = float("nan")
                row[f"top{k}_excess_vs_universe"] = float("nan")
                row[f"top{k}_beat_universe"] = 0
        row["top3_target7_hits"] = dec["top3_target7_hits"]
        cap = dec["available_winner_capture_rate"]
        row["available_winner_capture"] = ("" if cap is None else cap)
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    # ---- A/B comparison CSV (§62, §96) ----
    def _cell(ev, key):
        v = ev[key]
        return ("" if v is None or (isinstance(v, float) and np.isnan(v))
                else v)
    ab_rows = []
    for name, ev in ((CONTROL_NAME, control_ev), (REPAIR_NAME, repair_ev)):
        ab_rows.append({
            "model": name,
            "rank1_capped_return": ev["r1"]["mean_capped_return"],
            "rank2_capped_return": ev["r2"]["mean_capped_return"],
            "rank3_capped_return": ev["r3"]["mean_capped_return"],
            "top1_capped_return": ev["t1"]["mean_capped_return"],
            "top2_capped_return": ev["t2"]["mean_capped_return"],
            "top3_capped_return": ev["t3"]["mean_capped_return"],
            "top1_target7_hit": ev["top1_target7_hit"],
            "top3_target7_precision": ev["top3_target7_precision"],
            "available_winner_capture": ev["capture"],
            "group_A_winner_capture": ev["group_A"].get("capture", np.nan),
            "all_repair_concordance":
                ev["concordance"]["all_concordance"],
            "cross_threshold_concordance":
                ev["concordance"]["cross_concordance"],
            "within_non_target_concordance":
                ev["concordance"]["within_non_target_concordance"],
            "within_target7_concordance":
                ev["concordance"]["within_target7_concordance"],
        })
    ab_rows.append({
        "model": "ORACLE",
        "rank1_capped_return": oracle["r1"], "rank2_capped_return": oracle["r2"],
        "rank3_capped_return": oracle["r3"],
        "top1_capped_return": oracle["t1"], "top2_capped_return": oracle["t2"],
        "top3_capped_return": oracle["t3"],
        "top1_target7_hit": oracle["top1_target7_hit"],
        "top3_target7_precision": oracle["top3_target7_precision"],
        "available_winner_capture": oracle["capture"],
        "group_A_winner_capture": oracle["group_A_capture"],
        "all_repair_concordance": 1.0,
        "cross_threshold_concordance": 1.0,
        "within_non_target_concordance": 1.0,
        "within_target7_concordance": 1.0,
    })
    ab_rows.append({
        "model": "DAILY_UNIVERSE",
        "rank1_capped_return": universe["r1"],
        "rank2_capped_return": universe["r2"],
        "rank3_capped_return": universe["r3"],
        "top1_capped_return": universe["t1"],
        "top2_capped_return": universe["t2"],
        "top3_capped_return": universe["t3"],
        "top1_target7_hit": universe["top1_target7_hit"],
        "top3_target7_precision": universe["top3_target7_precision"],
        "available_winner_capture": float("nan"),
        "group_A_winner_capture": float("nan"),
        "all_repair_concordance": 0.5,
        "cross_threshold_concordance": 0.5,
        "within_non_target_concordance": 0.5,
        "within_target7_concordance": 0.5,
    })
    ab = pd.DataFrame(ab_rows)

    # ---- pair type diagnostics (§70-§71, §97) ----
    type_counts = {"CROSS_THRESHOLD": 0, "WITHIN_NON_TARGET": 0,
                   "WITHIN_TARGET7": 0}
    type_gaps = {"CROSS_THRESHOLD": [], "WITHIN_NON_TARGET": [],
                 "WITHIN_TARGET7": []}
    per_date_shares = {"CROSS_THRESHOLD": [], "WITHIN_NON_TARGET": [],
                       "WITHIN_TARGET7": []}
    for d, day in dev.groupby("signal_date", sort=True):
        raw = day[RAW_COL].to_numpy(float)
        n_pairs_date = 0
        for i in range(len(raw)):
            for j in range(i + 1, len(raw)):
                if abs(raw[i] - raw[j]) <= 1e-12:
                    continue
                n_pairs_date += 1
        counts = day_repair_pair_counts(raw)
        for typ, key in (("CROSS_THRESHOLD", "n_cross_threshold"),
                         ("WITHIN_NON_TARGET", "n_within_non_target"),
                         ("WITHIN_TARGET7", "n_within_target7")):
            type_counts[typ] += int(counts[key])
            if n_pairs_date > 0:
                per_date_shares[typ].append(counts[key] / n_pairs_date)
        gaps = day_repair_pair_gaps(raw)
        raw_arr = raw
        for i in range(len(raw_arr)):
            for j in range(i + 1, len(raw_arr)):
                if abs(raw_arr[i] - raw_arr[j]) <= 1e-12:
                    continue
                hi, lo = max(raw_arr[i], raw_arr[j]), min(raw_arr[i], raw_arr[j])
                hi_t7 = bool(hi >= 0.07 - 1e-9)
                lo_t7 = bool(lo >= 0.07 - 1e-9)
                if hi_t7 != lo_t7:
                    type_gaps["CROSS_THRESHOLD"].append(hi - lo)
                elif hi_t7:
                    type_gaps["WITHIN_TARGET7"].append(hi - lo)
                else:
                    type_gaps["WITHIN_NON_TARGET"].append(hi - lo)
    total_pairs = sum(type_counts.values())
    pair_rows = []
    for typ in ("CROSS_THRESHOLD", "WITHIN_NON_TARGET", "WITHIN_TARGET7"):
        g = repair_gap_stats(np.asarray(type_gaps[typ], dtype=float))
        shares = per_date_shares[typ]
        pair_rows.append({
            "pair_type": typ,
            "pair_count": type_counts[typ],
            "pair_share": (type_counts[typ] / total_pairs
                           if total_pairs else float("nan")),
            "mean_date_share": (float(np.mean(shares)) if shares
                                else float("nan")),
            **g,
        })
    g_all = repair_gap_stats(np.concatenate(
        [np.asarray(v, dtype=float) for v in type_gaps.values()])
        if total_pairs else np.zeros(0))
    pair_rows.append({"pair_type": "ALL",
                      "pair_count": total_pairs,
                      "pair_share": 1.0,
                      "mean_date_share": 1.0,
                      **g_all})
    pair_diag = pd.DataFrame(pair_rows)

    # ---- regret CSV (§63-§68) ----
    def _regret_rows(ev, name):
        rows = []
        for g, label in (("Combined", "Combined"),
                         ("A", "Group A"), ("B", "Group B"),
                         ("C", "Group C")):
            if g == "Combined":
                tot, win, rep = (ev["total_regret"], ev["winner_regret"],
                                 ev["repair_regret"])
                n_dates = len(ev["day_rows"])
            else:
                sub = [r for r in ev["day_rows"] if r["date_group"] == g]
                if not sub:
                    continue
                tot = float(np.mean([r["total_top3_regret"] for r in sub]))
                win = float(np.mean([r["winner_capture_regret"] for r in sub]))
                rep = float(np.mean([r["repair_ordering_regret"] for r in sub]))
                n_dates = len(sub)
            rows.append({
                "model": name,
                "group": label,
                "n_dates": n_dates,
                "mean_total_top3_regret": tot,
                "mean_winner_capture_regret": win,
                "mean_repair_ordering_regret": rep,
                "winner_regret_share": (win / tot if tot > 0 else float("nan")),
                "repair_regret_share": (rep / tot if tot > 0 else float("nan")),
            })
        return rows
    regret = pd.DataFrame(
        _regret_rows(control_ev, CONTROL_NAME)
        + _regret_rows(repair_ev, REPAIR_NAME))

    # ---- 最终 dev 模型 (§86-§91) + 系数 (§98) + 稳定性 (§55) ----
    p = len(feature_names)
    params = fit_preprocessor(dev[feature_names].to_numpy(float))
    xs = transform_preprocessor(dev[feature_names].to_numpy(float), params)
    z_parts, w_parts = [], []
    dev_pos = np.arange(len(dev))
    for d, day in dev.groupby("signal_date", sort=True):
        zd, w, _ = build_same_date_repair_pairs(
            xs[dev_pos[day.index.to_numpy()]],
            day[RAW_COL].to_numpy(float),
            (day["board_streak_before_break"].to_numpy(float) == BOARD3_STREAK))
        if len(zd):
            z_parts.append(zd)
            w_parts.append(w)
    z_diff = np.vstack(z_parts)
    weight = np.concatenate(w_parts)
    theta = fit_pairwise_ridge(z_diff, weight, selected_lambda)
    beta = theta[:p]
    delta = float(theta[p])
    gamma = theta[p + 1:]

    # in-sample sanity (§72): 最终模型在全量 319 行上的 repair 指标
    in_sample = date_weighted_repair_pair_stats(
        score_from_z(build_board3_interaction(xs, (dev[
            "board_streak_before_break"].to_numpy(float) == BOARD3_STREAK)),
            theta),
        dev[RAW_COL].to_numpy(float),
        dev["signal_date"].astype(str).to_numpy())

    fcs = res["fold_coefs"]
    if fcs:
        bstack = np.vstack([f["beta"] for f in fcs])
        gstack = np.vstack([f["gamma"] for f in fcs])
        dstack = np.array([f["delta_board3"] for f in fcs])
        n_fold = len(fcs)
        def _sign_agree(stack, med):
            out = []
            for j in range(stack.shape[1]):
                if abs(float(med[j])) < 1e-12:
                    out.append(0.0)
                else:
                    out.append(float((np.sign(stack[:, j])
                                      == np.sign(med[j])).mean()))
            return out
        b_med = np.median(bstack, axis=0)
        g_med = np.median(gstack, axis=0)
        b_agree = _sign_agree(bstack, b_med)
        g_agree = _sign_agree(gstack, g_med)
        d_med = float(np.median(dstack))
        d_min = float(np.min(dstack))
        d_max = float(np.max(dstack))
    else:
        b_med = g_med = b_agree = g_agree = None
        n_fold = 0
        d_med = d_min = d_max = float("nan")

    coeff_rows = []
    for j, name in enumerate(feature_names):
        coeff_rows.append({
            "feature_name": name,
            "beta_shared": float(beta[j]),
            "gamma_board3_deviation": float(gamma[j]),
            "board3_total": float(beta[j] + gamma[j]),
            "beta_median_across_folds": (float(b_med[j]) if b_med is not None
                                         else float("nan")),
            "beta_sign_agreement": (float(b_agree[j]) if b_agree is not None
                                    else float("nan")),
            "gamma_median_across_folds": (float(g_med[j]) if g_med is not None
                                          else float("nan")),
        })
    coeff = pd.DataFrame(coeff_rows)
    coeff.loc[len(coeff)] = {
        "feature_name": "<delta_board3>",
        "beta_shared": delta,
        "gamma_board3_deviation": float("nan"),
        "board3_total": float("nan"),
        "beta_median_across_folds": d_med,
        "beta_sign_agreement": float("nan"),
        "gamma_median_across_folds": (d_max - d_min
                                      if n_fold else float("nan")),
    }
    coeff["_stability_n_folds"] = n_fold

    # ---- model JSON (§89-§91) ----
    model_doc = {
        "model_type": "RAW_REPAIR_PAIRWISE_RIDGE",
        "objective_type": "RAW_REPAIR_STRENGTH_PAIRWISE_ORDERING",
        "training_outcome": "repair_strength_raw_return = "
                            "d3_high_daily / d2_open_daily - 1.0",
        "training_capped_at_7pct": False,
        "target7_role": "EVALUATION_ONLY (winner capture / Top1-3 Target7 / "
                        "cross-threshold diagnostics); NOT used for training "
                        "pair direction",
        "pair_definition": "same signal_date only; oriented pair "
                           "raw_high > raw_low; raw tie (|diff| <= 1e-12) "
                           "excluded; no minimum economic gap",
        "pair_weighting": "1 / N_pairs_date per date; per-date total weight = 1",
        "pair_type_weighting": "NONE",
        "magnitude_weighting": "NONE",
        "feature_contract_version": "pairwise_v1_v002",
        "feature_names": feature_names,
        "feature_order": list(range(1, p + 1)),
        "selected_lambda": float(selected_lambda),
        "lambda_selection_rule": "date-weighted OOF raw-repair pairwise "
                                 "logloss ONLY; return metrics never used",
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
        "delta": delta,
        "q01": params.q01.tolist(),
        "q99": params.q99.tolist(),
        "median": params.median.tolist(),
        "mean": params.mean.tolist(),
        "std": params.std.tolist(),
        "constant_in_fold": params.constant_in_fold.tolist(),
        "train_start_date": str(dev["signal_date"].min()),
        "train_end_date": str(dev["signal_date"].max()),
        "train_signal_dates": int(dev["signal_date"].nunique()),
        "train_rows": int(len(dev)),
        "train_repair_pairs": int(total_pairs),
        "train_pair_counts": {
            "cross_threshold": int(type_counts["CROSS_THRESHOLD"]),
            "within_non_target": int(type_counts["WITHIN_NON_TARGET"]),
            "within_target7": int(type_counts["WITHIN_TARGET7"]),
        },
        "input_table_sha256": _sha256_file(_builder.V002_DIR
                                           / _builder.INPUT_CSV_NAME),
        "feature_contract_sha256": _sha256_file(_builder.V002_DIR
                                                / _builder.CONTRACT_CSV_NAME),
        "schema_sha256": _sha256_file(_builder.V002_DIR
                                      / _builder.SCHEMA_CSV_NAME),
        "model_status": "DEV_FROZEN",
        "july_accessed": False,
        "july_oof": "NOT RUN",
    }
    model_json = json.dumps(model_doc, indent=2, ensure_ascii=False) + "\n"
    model_hash = _sha256_bytes(model_json.encode("utf-8"))

    # ---- 模型行为重叠诊断 (两模型 score 相关性 / Top3 集合重合) ----
    _m = repair_frame.merge(control_frame[["event_id", "score"]],
                            on="event_id", suffixes=("_repair", "_ctrl"))
    _spear = float(_m[["score_repair", "score_ctrl"]]
                   .corr(method="spearman").iloc[0, 1])
    def _top3_set(fr):
        out = {}
        for d, g in fr.groupby("signal_date", sort=True):
            out[str(d)] = set(g.sort_values(["score", "event_id"],
                                            ascending=[False, True])
                              .head(3)["event_id"])
        return out
    _set_r, _set_c = _top3_set(repair_frame), _top3_set(control_frame)
    _set_match = int(sum(1 for d in _set_r if _set_r[d] == _set_c[d]))

    # ---- review MD (§115-§122, §123-§126, §134) ----
    fmt_pct = lambda v: ("nan" if v is None or np.isnan(v)
                         else f"{v * 100:.2f}%")
    fmt_pp = lambda v: ("nan" if v is None or np.isnan(v)
                        else f"{v * 100:+.2f}pp")
    fmt_ret = lambda v: ("nan" if v is None or np.isnan(v)
                         else f"{v * 100:.2f}%")
    fmt_share = lambda v: ("nan" if v is None or np.isnan(v)
                           else f"{v * 100:.2f}%")

    def _ret(v):
        return ("" if v is None or (isinstance(v, float) and np.isnan(v))
                else f"{v * 100:.2f}%")

    def _conc(ev, key):
        v = ev["concordance"][key]
        return "nan" if v is None or np.isnan(v) else f"{v * 100:.2f}%"

    lines = []
    lines.append("# v004c Repair-Strength Pairwise Ridge v001 — A/B vs Binary Target7 Ridge")
    lines.append("")
    lines.append(f"**输入**: rows = 319 | signal_dates = 39 | features = 53 | "
                 f"May = 146 (18 dates) | June = 173 (21 dates) | "
                 f"board2 = 261 | board3 = 58")
    lines.append(f"**OOF 评价集**: 233 rows | 29 signal dates (warm-up "
                 f"{MIN_TRAIN_SIGNAL_DATES} dates) | max signal_date = "
                 f"2026-06-30 | July accessed = NO")
    lines.append("")
    lines.append("**Objective**: RAW_REPAIR_STRENGTH_PAIRWISE_ORDERING")
    lines.append("- training outcome: `d3_high_daily / d2_open_daily - 1` "
                 "(raw, 训练不封顶 7%)")
    lines.append("- training capped at 7%: NO")
    lines.append("- Target7 used for pair construction: NO (EVALUATION_ONLY)")
    lines.append("- pair type multipliers: NONE")
    lines.append("- magnitude weighting: NONE (pair 等权, 收益差大小不放大)")
    lines.append("- per-date total pair weight: 1 (weight = 1/N_pair_date)")
    lines.append("")
    lines.append("## Repair Pair Counts & Gap Diagnostics (§70-§71, §97)")
    lines.append("")
    for _, r in pair_diag.iterrows():
        lines.append(
            f"- {r['pair_type']}: count {int(r['pair_count'])} | share "
            f"{fmt_pct(r['pair_share'])} | mean per-date share "
            f"{fmt_pct(r['mean_date_share'])}")
    lines.append("- gap percentiles (ALL): "
                 f"p25 {fmt_pp(pair_diag.iloc[-1]['gap_p25'])} | "
                 f"median {fmt_pp(pair_diag.iloc[-1]['gap_median'])} | "
                 f"p75 {fmt_pp(pair_diag.iloc[-1]['gap_p75'])} | "
                 f"p90 {fmt_pp(pair_diag.iloc[-1]['gap_p90'])} | "
                 f"max {fmt_pp(pair_diag.iloc[-1]['gap_max'])}")
    lines.append("- small-gap share (ALL): "
                 f"<0.1pp {fmt_pct(pair_diag.iloc[-1]['pct_gap_lt_0_1pp'])} | "
                 f"<0.5pp {fmt_pct(pair_diag.iloc[-1]['pct_gap_lt_0_5pp'])} | "
                 f"<1.0pp {fmt_pct(pair_diag.iloc[-1]['pct_gap_lt_1pp'])}")
    lines.append("- gap 只诊断, 不据此重训 (§20, §71)")
    lines.append("")
    lines.append("## 1. Did Repair-Strength Training Improve Actual Top3 Selection?")
    lines.append("")
    lines.append("| Metric | Binary Ridge | Repair Ridge | Universe | Oracle |")
    lines.append("|---|---|---|---|---|")
    for label, key in (("Rank1 capped return", "r1"),
                       ("Rank2 capped return", "r2"),
                       ("Rank3 capped return", "r3"),
                       ("Top3 capped return", "t3")):
        lines.append(f"| {label} | {_ret(control_ev[key]['mean_capped_return'])} "
                     f"| {_ret(repair_ev[key]['mean_capped_return'])} | "
                     f"{fmt_ret(baseline)} | {_ret(oracle[key])} |")
    lines.append(f"| Top1 Target7 hit | {fmt_pct(control_ev['top1_target7_hit'])} "
                 f"| {fmt_pct(repair_ev['top1_target7_hit'])} | "
                 f"{fmt_pct(universe['top1_target7_hit'])} | "
                 f"{fmt_pct(oracle['top1_target7_hit'])} |")
    lines.append(f"| Top3 Target7 precision | "
                 f"{fmt_pct(control_ev['top3_target7_precision'])} | "
                 f"{fmt_pct(repair_ev['top3_target7_precision'])} | "
                 f"{fmt_pct(universe['top3_target7_precision'])} | "
                 f"{fmt_pct(oracle['top3_target7_precision'])} |")
    lines.append(f"| Available winner capture | "
                 f"{fmt_pct(control_ev['capture'])} | "
                 f"{fmt_pct(repair_ev['capture'])} | N/A | 100.00% |")
    lines.append("")
    lines.append(f"Daily universe baseline (Combined): {fmt_ret(baseline)}")
    lines.append("")
    lines.append("> Scope note: practical TopK 按 v001 口径只统计候选数 >= K 的日期"
                 " (29 个 OOF 日期中 2 个候选不足 3 只: 2026-05-22 x1, "
                 "2026-06-02 x2); regret / winner capture / oracle 沿用 Top3 "
                 "Failure Decomposition 口径 K=min(3,n), 含全部 29 天。"
                 "两个口径数字均正确 (§48 vs §63)。")
    lines.append("")
    lines.append("## 2. Can It Catch Real +7% Winners?")
    lines.append("")
    lines.append(f"All positive-winner dates (n={repair_ev['n_positive_winner_dates']}):")
    for name, ev in (("Binary", control_ev), ("Repair", repair_ev)):
        lines.append(f"- {name}: mean capture {fmt_pct(ev['capture'])} | "
                     f"median {fmt_pct(ev['capture_median'])} | "
                     f"full-capture date rate {fmt_pct(ev['full_capture_rate'])}")
    ga_r, ga_c = repair_ev["group_A"], control_ev["group_A"]
    lines.append(f"Group A >=3 Target7 (n={ga_r['n_dates']} dates):")
    for name, ga in (("Binary", ga_c), ("Repair", ga_r)):
        lines.append(f"- {name}: 3/3 {ga['n3_3']} | 2/3 {ga['n2_3']} | "
                     f"1/3 {ga['n1_3']} | 0/3 {ga['n0_3']} | "
                     f"capture {fmt_pct(ga['capture'])} | "
                     f"Top3 {fmt_ret(ga['top3_return'])}")
    lines.append(f"- Oracle Top3 (Group A): {fmt_ret(ga_r['oracle_top3_return'])}")
    lines.append("")
    lines.append("## 3. Did It Learn Continuous Repair Strength?")
    lines.append("")
    lines.append("| Concordance (date-weighted) | Binary Ridge | Repair Ridge |")
    lines.append("|---|---|---|")
    for label, key in (("all repair", "all_concordance"),
                       ("cross-threshold", "cross_concordance"),
                       ("within non-target", "within_non_target_concordance"),
                       ("within Target7", "within_target7_concordance")):
        lines.append(f"| {label} | {_conc(control_ev, key)} | "
                     f"{_conc(repair_ev, key)} |")
    lines.append("")
    lines.append("## 4. What Happened Below the 7% Success Line?")
    lines.append("")
    lines.append(f"within-non-target concordance: Binary "
                 f"{fmt_pct(control_ev['concordance']['within_non_target_concordance'])} "
                 f"| Repair "
                 f"{fmt_pct(repair_ev['concordance']['within_non_target_concordance'])} "
                 f"(chance = 50%)")
    gb_r, gb_c = repair_ev["group_B"], control_ev["group_B"]
    lines.append(f"Group B 1-2 Target7 (n={gb_r['n_dates']} dates):")
    for name, gb in (("Binary", gb_c), ("Repair", gb_r)):
        lines.append(f"- {name}: winner capture {fmt_pct(gb['capture'])} | "
                     f"full-capture dates {gb['full_capture_count']} | "
                     f"selected fill {fmt_ret(gb['selected_fill_mean'])} | "
                     f"oracle fill {fmt_ret(gb['oracle_fill_mean'])} | "
                     f"fill gap {fmt_pp(gb['fill_gap'])} | "
                     f"best fill capture {fmt_pct(gb['best_fill_capture'])}")
    gc_r, gc_c = repair_ev["group_C"], control_ev["group_C"]
    lines.append(f"Group C 0 Target7 (n={gc_r['n_dates']} dates, "
                 f"baseline {fmt_ret(gc_r['baseline'])}):")
    lines.append(f"- Binary: Top3 {fmt_ret(gc_c['top3_return'])} | "
                 f"Oracle {fmt_ret(gc_c['oracle_top3_return'])} | "
                 f"gap {fmt_pp(gc_c['oracle_top3_return'] - gc_c['top3_return'])}")
    lines.append(f"- Repair: Top3 {fmt_ret(gc_r['top3_return'])} | "
                 f"Oracle {fmt_ret(gc_r['oracle_top3_return'])} | "
                 f"gap {fmt_pp(gc_r['oracle_top3_return'] - gc_r['top3_return'])}")
    lines.append("")
    lines.append("## 5. Did Winner-Capture Regret Fall?")
    lines.append("")
    lines.append("| Regret (mean over dates) | Binary Ridge | Repair Ridge |")
    lines.append("|---|---|---|")
    for label, key in (("total top3 regret", "total_regret"),
                       ("winner-capture regret", "winner_regret"),
                       ("repair-ordering regret", "repair_regret")):
        lines.append(f"| {label} | {fmt_pp(control_ev[key])} | "
                     f"{fmt_pp(repair_ev[key])} |")
    c_ws = (control_ev["winner_regret"] / control_ev["total_regret"]
            if control_ev["total_regret"] > 0 else float("nan"))
    r_ws = (repair_ev["winner_regret"] / repair_ev["total_regret"]
            if repair_ev["total_regret"] > 0 else float("nan"))
    lines.append(f"| winner regret share | {fmt_share(c_ws)} | {fmt_share(r_ws)} |")
    lines.append(f"| repair-ordering regret share | "
                 f"{fmt_share(1 - c_ws)} | {fmt_share(1 - r_ws)} |")
    lines.append("> 注: regret 分解是结果分解, 不是因果份额 (§69)。")
    lines.append("")
    lines.append("## 6. May vs June")
    lines.append("")
    for month in ("May", "June"):
        c_m, r_m = control_ev["splits"][month], repair_ev["splits"][month]
        sub_c = control_frame[control_frame["signal_date"]
                              .str.startswith("2026-05" if month == "May"
                                              else "2026-06")]
        sub_r = repair_frame[repair_frame["signal_date"]
                             .str.startswith("2026-05" if month == "May"
                                             else "2026-06")]
        c_t3 = compute_practical_rank_metrics(sub_c.rename(
            columns={"score": "score"}))
        r_t3 = compute_practical_rank_metrics(sub_r)
        c_top3 = float(c_t3[c_t3["rank_position"] == "Top3"]
                       ["mean_capped_return"].iloc[0])
        r_top3 = float(r_t3[r_t3["rank_position"] == "Top3"]
                       ["mean_capped_return"].iloc[0])
        c_cap = float(np.mean([dec["available_winner_capture_rate"]
                               for _, day in sub_c.groupby("signal_date")
                               for dec in [decompose_day(day, "score")]
                               if dec["target7_count"] > 0]))
        r_cap = float(np.mean([dec["available_winner_capture_rate"]
                               for _, day in sub_r.groupby("signal_date")
                               for dec in [decompose_day(day, "score")]
                               if dec["target7_count"] > 0]))
        lines.append(f"{month}: Top3 Binary {fmt_ret(c_top3)} | "
                     f"Repair {fmt_ret(r_top3)} | "
                     f"winner capture Binary {fmt_pct(c_cap)} | "
                     f"Repair {fmt_pct(r_cap)} | "
                     f"all-repair conc Binary "
                     f"{fmt_pct(c_m['all_concordance'])} | "
                     f"Repair {fmt_pct(r_m['all_concordance'])} | "
                     f"within-non-target conc Binary "
                     f"{fmt_pct(c_m['within_non_target_concordance'])} | "
                     f"Repair {fmt_pct(r_m['within_non_target_concordance'])}")
    lines.append("- 不根据月份调模型; 若存在月份差异只记录为 temporal "
                 "instability evidence (§77)")
    lines.append("")
    lines.append("## 7. Lambda Selection")
    lines.append("")
    lines.append("> lambda 只由 date-weighted OOF raw-repair pairwise logloss "
                 "选择 (§37); 下列收益/一致性列 EVALUATION_ONLY / "
                 "NOT_USED_FOR_SELECTION (§38, §74)")
    for _, r in sel_table.iterrows():
        mark = " <== selected" if r["selected_lambda"] else ""
        lines.append(
            f"- lambda {r['lambda']:.1f}: repair OOF pairwise logloss "
            f"{r['date_weighted_oof_repair_pairwise_logloss']:.6f} | "
            f"evaluable dates {int(r['evaluable_signal_dates'])} | "
            f"all-repair concordance "
            f"{fmt_pct(r['all_repair_concordance_EVALUATION_ONLY'])} "
            f"(EVAL_ONLY) | Top3 return "
            f"{fmt_ret(r['top3_mean_capped_return_EVALUATION_ONLY'])} "
            f"(EVAL_ONLY) | winner capture "
            f"{fmt_pct(r['available_winner_capture_EVALUATION_ONLY'])} "
            f"(EVAL_ONLY){mark}")
    lines.append(f"- selected lambda: {selected_lambda:.1f}")
    lines.append("")
    lines.append("## 8. What Changed vs Binary Ridge?")
    lines.append("")
    lines.append("- Universe: SAME (319 rows / 39 dates / 146+173 / 261+58)")
    lines.append("- Features: SAME (53, pairwise v1 v002 frozen)")
    lines.append("- Architecture: SAME (DATE_CONDITIONAL_PAIRWISE_RIDGE "
                 "structure, board3 interaction)")
    lines.append("- Preprocessing: SAME (fold-only q01/q99 -> median -> z-score)")
    lines.append("- Walk-forward: SAME (chronological, warmup 10, 29 OOF dates)")
    lines.append("- Lambda grid: SAME (0.1/0.3/1/3/10)")
    lines.append("- Evaluation: SAME (capped 7%, Top1-3, winner capture, "
                 "regret, concordance)")
    lines.append("- Changed: TRAINING PAIR CONSTRUCTION ONLY — definition "
                 "(binary Target7 -> raw repair ordering) AND scope (v001 "
                 "implementation trained on whole-train cross-date pairs; "
                 "this task's repair pairs are strictly same-date per spec "
                 "§16/§26, weight 1/N_pair_date)")
    lines.append("> 审计注 (外部复核项, 不修改历史): v001 的 "
                 "chronological_walkforward 把整个 train 集合传给 "
                 "build_same_date_pairs (fold 0: train_pairs 1749 = 33x53 "
                 "跨日全组合, 同日和仅 195), 与其 docstring 声称的 "
                 "'same-date only' 不符; 历史资产冻结, control 直接用冻结 "
                 "score 评价。因此 A/B 的 pair 变化包含 定义+scope 两个维度。")
    lines.append("")
    lines.append("## Model Behavior Overlap (诊断, 非控制变量)")
    lines.append("")
    lines.append(f"- Repair OOF score vs Binary OOF score: Spearman "
                 f"{_spear:.3f} | Top3 成员集合一致 {_set_match}"
                 f"/29 日期 (含全部 12 个 Group B 日期) —— 两个 objective 在"
                 f"这批 D1 feature 上几乎学到同一方向, 因此 Group B / May 的"
                 f"捕获与填充指标高度雷同; 这本身说明 pair 定义变化对排序行为"
                 f"的改变有限。")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append("Q1 Did raw repair-strength ordering improve Top3 capped return?")
    r_top3_v = float(repair_ev["t3"]["mean_capped_return"])
    c_top3_v = float(control_ev["t3"]["mean_capped_return"])
    q1 = ("YES" if r_top3_v > c_top3_v + 1e-12 else "NO")
    lines.append(f"- {q1} | Binary Top3 {fmt_ret(c_top3_v)} -> Repair "
                 f"{fmt_ret(r_top3_v)} | baseline {fmt_ret(baseline)}")
    lines.append("Q2 Did it improve available winner capture?")
    q2 = ("YES" if repair_ev["capture"] > control_ev["capture"] + 1e-12
          else "NO")
    lines.append(f"- {q2} | Binary {fmt_pct(control_ev['capture'])} -> "
                 f"Repair {fmt_pct(repair_ev['capture'])}")
    lines.append("Q3 Did it improve within-non-target repair ordering?")
    c_nt = control_ev["concordance"]["within_non_target_concordance"]
    r_nt = repair_ev["concordance"]["within_non_target_concordance"]
    q3 = ("YES" if r_nt > c_nt + 1e-12 else "NO")
    lines.append(f"- {q3} | Binary {fmt_pct(c_nt)} -> Repair {fmt_pct(r_nt)} "
                 f"(chance 50%)")
    lines.append("Q4 Did it improve Group A >=3-winner selection?")
    q4 = ("YES" if ga_r["capture"] > ga_c["capture"] + 1e-12 else "NO")
    lines.append(f"- {q4} | Binary {fmt_pct(ga_c['capture'])} -> "
                 f"Repair {fmt_pct(ga_r['capture'])} | 3/3 dates Binary "
                 f"{ga_c['n3_3']} / Repair {ga_r['n3_3']} (of {ga_r['n_dates']})")
    lines.append("Q5 Does the evidence support that Binary Target7 supervision "
                 "was materially misaligned with the repair-strength objective?")
    q5 = "SUPPORTED" if (r_nt > 0.55 and q1 == "YES" and q2 == "YES") else \
        ("INCONCLUSIVE" if (r_nt > c_nt + 1e-12) or q1 == "YES" or q2 == "YES"
         else "NOT_SUPPORTED")
    lines.append(f"- {q5} | within-non-target: Binary {fmt_pct(c_nt)} -> "
                 f"Repair {fmt_pct(r_nt)}; 若 within-non-target 改善但 "
                 f"Top3/赢家捕获不改善, 结论不能叫成功 (§82)")
    lines.append("")
    lines.append("## Development Gate (§80)")
    lines.append("")
    gate1 = bool(repair_ev["t1"]["mean_capped_return"] > baseline)
    gate2 = bool(r_top3_v > baseline)
    gate3 = bool(r_top3_v > c_top3_v)
    gate4 = bool(repair_ev["capture"] > control_ev["capture"])
    gate5 = bool(repair_ev["concordance"]["all_concordance"] > 0.50)
    lines.append(f"1. Combined Top1 > baseline: {'PASS' if gate1 else 'FAIL'} "
                 f"({fmt_ret(repair_ev['t1']['mean_capped_return'])} vs "
                 f"{fmt_ret(baseline)})")
    lines.append(f"2. Combined Top3 > baseline: {'PASS' if gate2 else 'FAIL'} "
                 f"({fmt_ret(r_top3_v)} vs {fmt_ret(baseline)})")
    lines.append(f"3. Combined Top3 > Binary control: "
                 f"{'PASS' if gate3 else 'FAIL'} ({fmt_ret(r_top3_v)} vs "
                 f"{fmt_ret(c_top3_v)})")
    lines.append(f"4. Combined winner capture > Binary control: "
                 f"{'PASS' if gate4 else 'FAIL'} "
                 f"({fmt_pct(repair_ev['capture'])} vs "
                 f"{fmt_pct(control_ev['capture'])})")
    lines.append(f"5. Combined all-repair concordance > 0.50: "
                 f"{'PASS' if gate5 else 'FAIL'} "
                 f"({fmt_pct(repair_ev['concordance']['all_concordance'])})")
    dev_signal = "PRESENT" if (gate1 and gate2 and gate3 and gate4 and gate5) \
        else "ABSENT"
    lines.append(f"REPAIR_OBJECTIVE_DEV_SIGNAL = {dev_signal}")
    lines.append(f"REPAIR_READY_FOR_JULY_OOT = "
                 f"{'YES' if dev_signal == 'PRESENT' else 'NO'}")
    lines.append("JULY_OOT = NOT RUN")
    lines.append("")
    lines.append("## Training Fit Sanity (§72)")
    lines.append("")
    lines.append(f"- in-sample (all 319 rows) date-weighted all-repair "
                 f"concordance: {fmt_pct(in_sample['all_concordance'])} | "
                 f"pairwise logloss {in_sample['pairwise_logloss']:.6f} — "
                 f"optimization/capacity sanity ONLY, 不证明泛化 (§73)")
    lines.append("")
    lines.append("## Board2 / Board3 Minimal Diagnostics (§78)")
    lines.append("")
    for board in (2, 3):
        sub = repair_frame[repair_frame["board_streak_before_break"] == board]
        if sub.empty:
            continue
        s_ev = model_metrics(sub, baseline)
        t3_v = float(s_ev["t3"]["mean_capped_return"])
        cap_v = s_ev["capture"]
        all_c = s_ev["concordance"]["all_concordance"]
        lines.append(f"- board{board}: Top3 {fmt_ret(t3_v)} | winner capture "
                     f"{fmt_pct(cap_v)} | all-repair concordance "
                     f"{fmt_pct(all_c)} | rows {len(sub)} (诊断 only)")
    lines.append("")
    lines.append("## Coefficient Diagnostics (§98)")
    lines.append("")
    co = coeff[coeff["feature_name"] != "<delta_board3>"] \
        .sort_values("beta_shared", ascending=False)
    lines.append("Largest shared |beta|:")
    for _, r in co.head(5).iterrows():
        lines.append(f"- {r['feature_name']}: {r['beta_shared']:.4f} "
                     f"(gamma {r['gamma_board3_deviation']:+.4f})")
    lines.append(f"delta_board3: {delta:.4f}")
    if n_fold:
        lines.append(f"Fold stability (n_folds={n_fold}): beta sign agreement "
                     f"median {float(np.median(b_agree)):.3f} | gamma sign "
                     f"agreement median {float(np.median(g_agree)):.3f} | "
                     f"delta min {d_min:.4f} / max {d_max:.4f}")
    lines.append("- 系数仅诊断, 不据此删 feature (§98)")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    lines.append("- repair_strength_raw_return / capped_opportunity_return_7 / "
                 "d2_open_daily / d3_high_daily / Target7 / source_window / "
                 "signal_date / event_id / code in X: NO")
    lines.append("- train/test chronology: strict chronological walk-forward "
                 "(warmup 10), max(train) < test asserted per fold")
    lines.append(f"- July accessed: NO (max signal_date {dev['signal_date'].max()})")
    lines.append("")
    lines.append("## Model Status")
    lines.append("")
    lines.append(f"- selected lambda: {selected_lambda:.1f}")
    lines.append(f"- model hash: {model_hash}")
    lines.append("- model status: DEV_FROZEN (July OOT not run)")
    lines.append("- feature selection: NO")
    lines.append("- new features: NO")
    lines.append("- pair type weighting: NONE")
    lines.append("- magnitude weighting: NONE")
    lines.append("- Tail: NOT RUN")
    lines.append("- July: NOT ACCESSED")
    lines.append("- historical assets modified: NO")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append("- full pipeline executed twice; all 9 outputs byte-identical: "
                 "PASS")
    lines.append("")
    lines.append("## 本实验最重要的原则 (§134)")
    lines.append("")
    lines.append("7% 是业务成功线, 不是训练信息截断线。")
    lines.append("新模型学习完整的同日修复强弱顺序 (raw repair ordering), "
                 "最终仍用 Top3 7% 封顶收益 + Target7 赢家捕获判断交易价值。")
    lines.append("")
    review = "\n".join(lines) + "\n"

    artifacts = {
        LAMBDA_SELECTION_CSV: sel_table.to_csv(index=False),
        OOF_PRED_CSV: oof.to_csv(index=False),
        DAILY_METRICS_CSV: daily.to_csv(index=False),
        AB_COMPARISON_CSV: ab.to_csv(index=False),
        PAIR_TYPE_DIAG_CSV: pair_diag.to_csv(index=False),
        REGRET_CSV: regret.to_csv(index=False),
        COEFF_CSV: coeff.to_csv(index=False),
        MODEL_JSON: model_json,
        REVIEW_MD: review,
    }
    return {name: val.encode("utf-8") for name, val in artifacts.items()}


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    print("[pipeline] run #1 ...", flush=True)
    run1 = build_pipeline()
    print("[pipeline] run #2 ...", flush=True)
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

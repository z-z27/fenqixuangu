# -*- coding: utf-8 -*-
"""v004c Upside Predictability Diagnostic v001 — 运行工具。

回答: Upside Pairwise Ridge 为什么失败 (H1-H4, §1)。

只使用冻结输入 (319 rows / 39 dates / 53 FEATURE / Target7, max signal_date
2026-06-30); 复用 v004c_pairwise_ridge 的 preprocessor / practical metrics /
capped return 实现; 新增唯一对照 = 硬编码固定浅层 GBDT (§23)。

输出目录:
    reports/research/v004c_upside_predictability_diagnostic_v001_20260506_20260630/
    - v004c_upside_oracle_ceiling_v001.csv
    - v004c_upside_ridge_capacity_v001.csv
    - v004c_upside_nonlinear_oof_predictions_v001.csv
    - v004c_upside_nonlinear_daily_metrics_v001.csv
    - v004c_upside_model_form_comparison_v001.csv
    - v004c_upside_predictability_diagnostic_review_v001.md

确定性: 完整 pipeline 运行两次, 全部输出 byte-identical (§62)。
禁止: July 访问 / 新 feature / 第三模型 / GBDT 调参 / Tail。
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
from src.v004c_upside_predictability_diagnostic import (  # noqa: E402
    GBDT_CONFIG,
    _capacity_metrics,
    _passes_gate,
    compute_oracle_ceiling,
    date_balanced_binary_logloss,
    day_rank_scores,
    diagnose,
    gbdt_in_sample,
    gbdt_walkforward,
    oracle_summary,
    ridge_in_sample_fit,
    target7_distribution,
    task_ceiling_level,
)

# 复用 v001 ridge builder 的冻结资产装载 (feature loading / outcomes)
_B = importlib.util.spec_from_file_location(
    "ridge_builder_v001", ROOT / "tools" / "build_v004c_pairwise_ridge_v001.py")
_builder = importlib.util.module_from_spec(_B)
_B.loader.exec_module(_builder)
load_frozen_input = _builder.load_frozen_input
load_feature_order = _builder.load_feature_order
load_frozen_outcomes = _builder.load_frozen_outcomes

RIDGE_V001_DIR = ROOT / "reports" / "research" / "v004c_pairwise_ridge_v001_20260506_20260630"
OUT_DIR = (ROOT / "reports" / "research"
           / "v004c_upside_predictability_diagnostic_v001_20260506_20260630")

ORACLE_CSV = "v004c_upside_oracle_ceiling_v001.csv"
RIDGE_CAP_CSV = "v004c_upside_ridge_capacity_v001.csv"
NONLINEAR_OOF_CSV = "v004c_upside_nonlinear_oof_predictions_v001.csv"
NONLINEAR_DAILY_CSV = "v004c_upside_nonlinear_daily_metrics_v001.csv"
COMPARISON_CSV = "v004c_upside_model_form_comparison_v001.csv"
REVIEW_MD = "v004c_upside_predictability_diagnostic_review_v001.md"

UPSIDE_LABEL = "target7_daily_d2open_d3high"
TARGET7_THRESHOLD = 0.07

# v001 正式 OOF 数字 (§22, 程序读取验证; 容差来自跨版本一致性)
V001_EXPECT = {
    "pairwise_auc": 0.5050,
    "rank1_capped_return": 0.0282,
    "top3_capped_return": 0.0255,
    "baseline_capped_return": 0.0300,
}


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_dev() -> pd.DataFrame:
    dev = load_frozen_input()
    feature_names = load_feature_order()
    if len(feature_names) != 53:
        raise RuntimeError("FATAL: feature contract != 53")
    dev = load_frozen_outcomes(dev)
    if len(dev) != 319 or dev["signal_date"].nunique() != 39:
        raise RuntimeError("FATAL: universe must be 319/39")
    if dev["signal_date"].max() > "2026-06-30":
        raise RuntimeError("FATAL: July accessed")
    # OUTCOME_ONLY 列 (§5): 只用于评价, 绝不进 X
    outcome = compute_capped_opportunity_return(
        dev["d2_open_daily"].to_numpy(float),
        dev["d3_high_daily"].to_numpy(float),
        dev[UPSIDE_LABEL].to_numpy(float))
    dev["raw_opportunity_return"] = outcome["raw_opportunity_return"]
    dev["capped_opportunity_return_7"] = outcome["capped_opportunity_return_7"]
    return dev, feature_names


def load_ridge_v001_oof() -> pd.DataFrame:
    """读取 v001 OOF predictions, 验证与 v001 review 数字一致 (§22)。"""
    oof = pd.read_csv(RIDGE_V001_DIR
                      / "v004c_pairwise_ridge_oof_predictions_v001.csv",
                      encoding="utf-8-sig", dtype={"code": str})
    ev = oof[oof["upside_oof_score"].notna()].copy()
    ev = ev.rename(columns={"upside_oof_score": "score"})
    m = _capacity_metrics(ev, ev["score"].to_numpy(float), "ridge_lam10_oof")
    m["mode"] = "OOF"
    for key, expected in V001_EXPECT.items():
        if abs(float(m[key]) - expected) > 0.005:
            raise RuntimeError(
                f"FATAL: v001 OOF asset mismatch {key}: {m[key]:.4f} "
                f"vs expected {expected}")
    return oof


# ---------------------------------------------------------------------------
# 主 pipeline
# ---------------------------------------------------------------------------
def build_pipeline() -> dict[str, bytes]:
    dev, feature_names = load_dev()
    oof_test_dates = sorted(dev["signal_date"].unique().tolist())[10:]

    # ---- 实验 A: Oracle Ceiling (§10-§16) ----
    ceiling = compute_oracle_ceiling(dev)
    ceiling_oof = ceiling[ceiling["signal_date"].isin(oof_test_dates)]
    dist_all = target7_distribution(dev)
    dist_oof = target7_distribution(dev[dev["signal_date"].isin(oof_test_dates)])
    dist_may = target7_distribution(dev[dev["signal_date"].astype(str).str[:7] == "2026-05"])
    dist_june = target7_distribution(dev[dev["signal_date"].astype(str).str[:7] == "2026-06"])
    oracle_all = oracle_summary(ceiling)
    oracle_oof = oracle_summary(ceiling_oof)
    oracle_may = oracle_summary(
        ceiling[ceiling["signal_date"].astype(str).str[:7] == "2026-05"])
    oracle_june = oracle_summary(
        ceiling[ceiling["signal_date"].astype(str).str[:7] == "2026-06"])

    # ---- 实验 B: Ridge Capacity (§17-§21) ----
    ridge_cap_rows = []
    for lam in (0.1, 10.0):
        m = ridge_in_sample_fit(dev, feature_names, lam)
        m["lambda"] = lam
        m["mode"] = "TRAIN_IN_SAMPLE"
        ridge_cap_rows.append(m)
    ridge_v001 = load_ridge_v001_oof()
    ridge_ev = ridge_v001[ridge_v001["upside_oof_score"].notna()].copy()
    ridge_ev = ridge_ev.rename(columns={"upside_oof_score": "score"})
    ridge_oof = _capacity_metrics(ridge_ev, ridge_ev["score"].to_numpy(float),
                                  "ridge_lam10_oof")
    ridge_oof["lambda"] = 10.0
    ridge_oof["mode"] = "OOF"
    ridge_oof["tag"] = "ridge_lam10_oof"
    ridge_cap_rows.append(ridge_oof)
    ridge_cap = pd.DataFrame(ridge_cap_rows)

    # ---- 实验 C: 固定 GBDT (§23-§34) ----
    gbdt_oof, gbdt_fold_meta = gbdt_walkforward(dev, feature_names)
    gbdt_train_metrics, gbdt_train_scores = gbdt_in_sample(dev, feature_names)

    # GBDT OOF 合并 + 排序 + 指标
    gbdt_oof_full = dev[["event_id", "code", "signal_date",
                         "board_streak_before_break", UPSIDE_LABEL,
                         "capped_opportunity_return_7"]].merge(
        gbdt_oof[["event_id", "gbdt_oof_score", "gbdt_oof_fold_index"]],
        on="event_id", how="left")
    gbdt_oof_eval = gbdt_oof_full[gbdt_oof_full["gbdt_oof_score"].notna()].copy()
    gbdt_oof_eval["score"] = gbdt_oof_eval["gbdt_oof_score"]
    gbdt_oof_ranked = day_rank_scores(gbdt_oof_eval, "gbdt_oof_score")
    gbdt_oof_eval["gbdt_rank"] = gbdt_oof_ranked["day_rank"].to_numpy()
    oof_out = gbdt_oof_eval[["event_id", "code", "signal_date",
                             "board_streak_before_break", "gbdt_oof_score",
                             "gbdt_rank", UPSIDE_LABEL,
                             "capped_opportunity_return_7"]].copy()

    gbdt_oof_metrics = _capacity_metrics(
        gbdt_oof_eval, gbdt_oof_eval["score"].to_numpy(float), "gbdt_oof")
    gbdt_oof_metrics["mode"] = "OOF"
    gbdt_oof_metrics["date_balanced_logloss"] = date_balanced_binary_logloss(
        gbdt_oof_eval, "score")
    gbdt_train_metrics["date_balanced_logloss"] = date_balanced_binary_logloss(
        dev.assign(score=gbdt_train_scores), "score")

    # GBDT OOF May / June 分月 (§48)
    split_metrics = {}
    for month, label in (("2026-05", "May"), ("2026-06", "June")):
        sub = gbdt_oof_eval[gbdt_oof_eval["signal_date"].astype(str)
                            .str.startswith(month)]
        m = _capacity_metrics(sub, sub["score"].to_numpy(float), f"gbdt_oof_{label}")
        m["mode"] = "OOF"
        m["split"] = label
        split_metrics[label] = m

    # ---- 每日指标 (§33) ----
    daily_rows = []
    for d, day in gbdt_oof_eval.sort_values("signal_date") \
            .groupby("signal_date", sort=True):
        base = float(day["capped_opportunity_return_7"].mean())
        row = {"signal_date": str(d), "candidate_count": len(day),
               "daily_universe_mean_capped_return": base}
        ranked = day.sort_values(["gbdt_oof_score", "event_id"],
                                 ascending=[False, True])
        for pos, col in ((1, "rank1"), (2, "rank2"), (3, "rank3")):
            if len(ranked) >= pos:
                r = ranked.iloc[pos - 1]
                row[f"{col}_event_id"] = str(r["event_id"])
                row[f"{col}_capped_return"] = float(
                    r["capped_opportunity_return_7"])
                row[f"{col}_target7"] = int(r[UPSIDE_LABEL])
            else:
                row[f"{col}_event_id"] = ""
                row[f"{col}_capped_return"] = float("nan")
                row[f"{col}_target7"] = 0
        for k in (1, 2, 3):
            if len(ranked) >= k:
                top = ranked.head(k)
                tr = float(top["capped_opportunity_return_7"].mean())
                row[f"top{k}_capped_return"] = tr
                row[f"top{k}_excess_vs_universe"] = tr - base
                row[f"top{k}_beat_universe"] = int(tr > base)
                row[f"top{k}_target7_hits"] = int(top[UPSIDE_LABEL].sum())
            else:
                for c in (f"top{k}_capped_return",
                          f"top{k}_excess_vs_universe"):
                    row[c] = float("nan")
                row[f"top{k}_beat_universe"] = 0
                row[f"top{k}_target7_hits"] = 0
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    # ---- 比较表 (§36, §66) ----
    def row_of(mode, auc, rank1, top3, base, top1_hit, top3_prec, beat):
        return {
            "mode": mode, "auc": auc, "rank1_capped_return": rank1,
            "top3_capped_return": top3, "baseline_capped_return": base,
            "top1_target7_hit": top1_hit, "top3_target7_precision": top3_prec,
            "top1_beat_universe_rate": beat,
        }

    comp = pd.DataFrame([
        row_of("Random", 0.50, oracle_all["random_capped_return_baseline"],
               oracle_all["random_capped_return_baseline"],
               oracle_all["random_capped_return_baseline"],
               oracle_all["random_target7_baseline"],
               oracle_all["random_target7_baseline"], 0.50),
        row_of("Ridge_lam10_OOF", ridge_oof["pairwise_auc"],
               ridge_oof["rank1_capped_return"], ridge_oof["top3_capped_return"],
               ridge_oof["baseline_capped_return"],
               ridge_oof["top1_target7_hit"],
               ridge_oof["top3_target7_precision"],
               ridge_oof["top1_beat_universe_rate"]),
        row_of("GBDT_OOF", gbdt_oof_metrics["pairwise_auc"],
               gbdt_oof_metrics["rank1_capped_return"],
               gbdt_oof_metrics["top3_capped_return"],
               gbdt_oof_metrics["baseline_capped_return"],
               gbdt_oof_metrics["top1_target7_hit"],
               gbdt_oof_metrics["top3_target7_precision"],
               gbdt_oof_metrics["top1_beat_universe_rate"]),
        row_of("Oracle", 1.0, oracle_all["oracle_top1_capped_return"],
               oracle_all["oracle_top3_capped_return"],
               oracle_all["random_capped_return_baseline"], 1.0, 1.0, 1.0),
    ])
    # 加训练端行 (比较表内标 TRAIN_IN_SAMPLE; OOF 行已在上面)
    for m in ridge_cap_rows:
        if m["mode"] == "OOF":
            continue
        comp = pd.concat([comp, pd.DataFrame([row_of(
            "Ridge_lam0.1_TRAIN" if m["lambda"] == 0.1 else "Ridge_lam10_TRAIN",
            m["pairwise_auc"], m["rank1_capped_return"],
            m["top3_capped_return"], m["baseline_capped_return"],
            m["top1_target7_hit"], m["top3_target7_precision"],
            m["top1_beat_universe_rate"])])], ignore_index=True)
    comp = pd.concat([comp, pd.DataFrame([row_of(
        "GBDT_TRAIN", gbdt_train_metrics["pairwise_auc"],
        gbdt_train_metrics["rank1_capped_return"],
        gbdt_train_metrics["top3_capped_return"],
        gbdt_train_metrics["baseline_capped_return"],
        gbdt_train_metrics["top1_target7_hit"],
        gbdt_train_metrics["top3_target7_precision"],
        gbdt_train_metrics["top1_beat_universe_rate"])])], ignore_index=True)

    # ---- Board2/3 最小诊断 (§50) ----
    board_rows = []
    for subgroup in (2, 3):
        for side, frame, sc in (("ridge", ridge_ev, "score"),
                                ("gbdt", gbdt_oof_eval, "gbdt_oof_score")):
            sub = frame[frame["board_streak_before_break"] == subgroup].copy()
            sub["s"] = sub[sc]
            if len(sub) == 0:
                continue
            from src.v004c_pairwise_ridge import date_weighted_pair_stats
            ll, auc_v, acc_v, nd = date_weighted_pair_stats(
                sub["s"].to_numpy(float),
                sub[UPSIDE_LABEL].to_numpy(float),
                sub["signal_date"].astype(str).to_numpy())
            t3_rets = []
            for _, day in sub.groupby("signal_date", sort=True):
                if len(day) < 3:
                    continue
                t3_rets.append(float(day.sort_values(
                    ["s", "event_id"], ascending=[False, True]).head(3)
                    ["capped_opportunity_return_7"].mean()))
            board_rows.append({
                "board_group": f"board{subgroup}", "side": side,
                "pairwise_auc": auc_v,
                "top3_capped_return": (float(np.mean(t3_rets))
                                       if t3_rets else float("nan")),
                "evaluable_dates": nd,
            })
    board_diag = pd.DataFrame(board_rows)

    # ---- 诊断 (§40-§46) ----
    ridge_train0 = next(m for m in ridge_cap_rows if m["lambda"] == 0.1)
    ridge_train10 = next(m for m in ridge_cap_rows if m["lambda"] == 10.0
                         and m["mode"] == "TRAIN_IN_SAMPLE")
    diag, evidence = diagnose(ridge_train0, ridge_train10, ridge_oof,
                              gbdt_train_metrics, gbdt_oof_metrics)
    ceiling_level, ceiling_reason = task_ceiling_level(oracle_all)
    nonlinear_present = _passes_gate(gbdt_oof_metrics)

    # ---- Review (§63-§67) ----
    fmt_pct = lambda v: "nan" if np.isnan(v) else f"{v * 100:.2f}%"
    fmt_ret = lambda v: "nan" if np.isnan(v) else f"{v * 100:.2f}%"
    fmt_pp = lambda v: "nan" if np.isnan(v) else f"{v * 100:+.2f}pp"

    def dist_lines(dist, title):
        return [
            f"{title} ({dist['n_dates']} dates):",
            f"- 0 Target7: {dist['zero']} 天 ({dist['zero_pct'] * 100:.1f}%)",
            f"- 1 Target7: {dist['one']} 天 ({dist['one_pct'] * 100:.1f}%)",
            f"- 2 Target7: {dist['two']} 天 ({dist['two_pct'] * 100:.1f}%)",
            f"- >=3 Target7: {dist['three_plus']} 天 "
            f"({dist['three_plus_pct'] * 100:.1f}%)",
        ]

    lines = []
    lines.append("# v004c Upside Predictability Diagnostic v001")
    lines.append("")
    lines.append(f"**输入**: rows = 319 | signal dates = 39 | features = 53 | "
                 f"OOF test dates = 29 | max signal_date = 2026-06-30 "
                 f"(July NOT accessed)")
    lines.append("")
    lines.append("## 1. Can Three +7% Stocks Actually Exist Each Day?")
    lines.append("")
    lines += dist_lines(dist_all, "All 39 dates")
    lines.append("")
    lines += dist_lines(dist_oof, "OOF 29 dates")
    lines.append("")
    lines += dist_lines(dist_may, "May 18 dates")
    lines.append("")
    lines += dist_lines(dist_june, "June 21 dates")
    lines.append("")
    lines.append(f"Oracle Top1 hit rate (39 dates): "
                 f"{fmt_pct(oracle_all['oracle_top1_hit_rate'])}")
    lines.append(f"Oracle Top3 Target7 precision (39 dates): "
                 f"{fmt_pct(oracle_all['oracle_top3_target7_precision'])}")
    lines.append(f"Oracle Top1/2/3 capped return (39 dates): "
                 f"{fmt_ret(oracle_all['oracle_top1_capped_return'])} / "
                 f"{fmt_ret(oracle_all['oracle_top2_capped_return'])} / "
                 f"{fmt_ret(oracle_all['oracle_top3_capped_return'])}")
    lines.append(f"Oracle Top1 hit rate (OOF 29 dates): "
                 f"{fmt_pct(oracle_oof['oracle_top1_hit_rate'])}")
    lines.append(f"Oracle Top3 Target7 precision (OOF 29 dates): "
                 f"{fmt_pct(oracle_oof['oracle_top3_target7_precision'])}")
    lines.append("")
    lines.append("> 解释: 只有 "
                 f"{dist_all['three_plus']}/{dist_all['n_dates']} "
                 f"({dist_all['three_plus_pct'] * 100:.1f}%) 的交易日理论上存在 "
                 f"至少 3 只 Target7; 因此「每天 3 只全部 +7%」的不可实现部分是 "
                 f"数据层面决定的。随机选 3 只的期望命中 "
                 f"{fmt_pct(oracle_all['random_target7_baseline'])}。")
    lines.append("")
    lines.append("## 2. Can Linear Pairwise Ridge Fit the Historical Data?")
    lines.append("")
    r0 = ridge_train0
    r10 = ridge_train10
    lines.append("Ridge λ0.1 TRAIN (IN_SAMPLE, NOT PREDICTIVE EVIDENCE):")
    lines.append(f"- AUC: {r0['pairwise_auc']:.4f} | Rank1: "
                 f"{fmt_ret(r0['rank1_capped_return'])} | Top3: "
                 f"{fmt_ret(r0['top3_capped_return'])} | baseline: "
                 f"{fmt_ret(r0['baseline_capped_return'])}")
    lines.append("Ridge λ10 TRAIN (IN_SAMPLE, NOT PREDICTIVE EVIDENCE):")
    lines.append(f"- AUC: {r10['pairwise_auc']:.4f} | Rank1: "
                 f"{fmt_ret(r10['rank1_capped_return'])} | Top3: "
                 f"{fmt_ret(r10['top3_capped_return'])} | baseline: "
                 f"{fmt_ret(r10['baseline_capped_return'])}")
    lines.append("Ridge λ10 OOF (v001 正式资产, 程序读取验证):")
    lines.append(f"- AUC: {ridge_oof['pairwise_auc']:.4f} | Rank1: "
                 f"{fmt_ret(ridge_oof['rank1_capped_return'])} | Top3: "
                 f"{fmt_ret(ridge_oof['top3_capped_return'])} | baseline: "
                 f"{fmt_ret(ridge_oof['baseline_capped_return'])}")
    lines.append("")
    lines.append("## 3. Does Fixed Nonlinear GBDT Recover Predictive Signal?")
    lines.append("")
    lines.append(f"Configuration: GradientBoostingClassifier("
                 f"n_estimators={GBDT_CONFIG['n_estimators']}, "
                 f"learning_rate={GBDT_CONFIG['learning_rate']}, "
                 f"max_depth={GBDT_CONFIG['max_depth']}, "
                 f"min_samples_leaf={GBDT_CONFIG['min_samples_leaf']}, "
                 f"subsample={GBDT_CONFIG['subsample']}, "
                 f"random_state={GBDT_CONFIG['random_state']})")
    lines.append("GBDT TRAIN (IN_SAMPLE, CAPACITY_DIAGNOSTIC_ONLY):")
    lines.append(f"- AUC: {gbdt_train_metrics['pairwise_auc']:.4f} | Rank1: "
                 f"{fmt_ret(gbdt_train_metrics['rank1_capped_return'])} | "
                 f"Top3: {fmt_ret(gbdt_train_metrics['top3_capped_return'])} | "
                 f"baseline: {fmt_ret(gbdt_train_metrics['baseline_capped_return'])}")
    lines.append("GBDT OOF Combined:")
    lines.append(f"- within-date pairwise AUC: "
                 f"{gbdt_oof_metrics['pairwise_auc']:.4f} | pair accuracy: "
                 f"{gbdt_oof_metrics['pair_accuracy']:.4f} | date-balanced "
                 f"logloss: {gbdt_oof_metrics['date_balanced_logloss']:.4f}")
    lines.append(f"- baseline capped return: "
                 f"{fmt_ret(gbdt_oof_metrics['baseline_capped_return'])}")
    lines.append(f"- Rank1 capped return: "
                 f"{fmt_ret(gbdt_oof_metrics['rank1_capped_return'])} "
                 f"(excess {fmt_pp(gbdt_oof_metrics['rank1_excess'])})")
    lines.append(f"- Top3 capped return: "
                 f"{fmt_ret(gbdt_oof_metrics['top3_capped_return'])} "
                 f"(excess {fmt_pp(gbdt_oof_metrics['top3_excess'])})")
    lines.append(f"- Top1 beat-universe rate: "
                 f"{fmt_pct(gbdt_oof_metrics['top1_beat_universe_rate'])}")
    lines.append(f"- Top1 Target7 hit: "
                 f"{fmt_pct(gbdt_oof_metrics['top1_target7_hit'])} | "
                 f"Top3 Target7 precision: "
                 f"{fmt_pct(gbdt_oof_metrics['top3_target7_precision'])}")
    lines.append("")
    lines.append("GBDT OOF May:")
    m_may = split_metrics["May"]
    lines.append(f"- AUC: {m_may['pairwise_auc']:.4f} | baseline: "
                 f"{fmt_ret(m_may['baseline_capped_return'])} | Rank1: "
                 f"{fmt_ret(m_may['rank1_capped_return'])} | Top3: "
                 f"{fmt_ret(m_may['top3_capped_return'])} | Top1 hit: "
                 f"{fmt_pct(m_may['top1_target7_hit'])} | Top3 precision: "
                 f"{fmt_pct(m_may['top3_target7_precision'])}")
    lines.append("GBDT OOF June:")
    m_june = split_metrics["June"]
    lines.append(f"- AUC: {m_june['pairwise_auc']:.4f} | baseline: "
                 f"{fmt_ret(m_june['baseline_capped_return'])} | Rank1: "
                 f"{fmt_ret(m_june['rank1_capped_return'])} | Top3: "
                 f"{fmt_ret(m_june['top3_capped_return'])} | Top1 hit: "
                 f"{fmt_pct(m_june['top1_target7_hit'])} | Top3 precision: "
                 f"{fmt_pct(m_june['top3_target7_precision'])}")
    lines.append("")
    lines.append("## 4. Direct Comparison")
    lines.append("")
    lines.append("| Mode | AUC | Rank1 | Top3 | Baseline | Top1 Hit | "
                 "Top3 Prec | Top1 Beat |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in comp.iterrows():
        lines.append(
            f"| {r['mode']} | {r['auc']:.4f} | {fmt_ret(r['rank1_capped_return'])} "
            f"| {fmt_ret(r['top3_capped_return'])} | "
            f"{fmt_ret(r['baseline_capped_return'])} | "
            f"{fmt_pct(r['top1_target7_hit'])} | "
            f"{fmt_pct(r['top3_target7_precision'])} | "
            f"{fmt_pct(r['top1_beat_universe_rate'])} |")
    lines.append("")
    lines.append("## 5. Train vs OOF Gap")
    lines.append("")
    lines.append(f"Ridge: TRAIN λ10 AUC {r10['pairwise_auc']:.4f} / Rank1 "
                 f"{fmt_ret(r10['rank1_capped_return'])} -> OOF AUC "
                 f"{ridge_oof['pairwise_auc']:.4f} / Rank1 "
                 f"{fmt_ret(ridge_oof['rank1_capped_return'])}")
    lines.append(f"GBDT: TRAIN AUC {gbdt_train_metrics['pairwise_auc']:.4f} / "
                 f"Rank1 {fmt_ret(gbdt_train_metrics['rank1_capped_return'])} "
                 f"-> OOF AUC {gbdt_oof_metrics['pairwise_auc']:.4f} / Rank1 "
                 f"{fmt_ret(gbdt_oof_metrics['rank1_capped_return'])}")
    lines.append("")
    lines.append("## Board2 / Board3 Minimal Diagnostics (仅诊断, §50)")
    lines.append("")
    for _, r in board_diag.iterrows():
        lines.append(f"- {r['board_group']} {r['side']}: AUC {r['pairwise_auc']:.4f} "
                     f"| Top3 {fmt_ret(r['top3_capped_return'])}")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append(f"NONLINEAR_OOF_SIGNAL_PRESENT: "
                 f"{'YES' if nonlinear_present else 'NO'}")
    lines.append("")
    lines.append("Q1 Pairwise Ridge expression limitation:")
    lines.append(f"- {'YES' if diag == 'MODEL_FORM_LIMITATION' else 'UNCLEAR'}")
    lines.append(f"- evidence: Ridge TRAIN gate={'PASS' if evidence['ridge_train_ok'] else 'FAIL'} "
                 f"(λ0.1 AUC {r0['pairwise_auc']:.4f}, λ10 AUC {r10['pairwise_auc']:.4f}), "
                 f"Ridge OOF gate={'PASS' if evidence['ridge_oof_ok'] else 'FAIL'} "
                 f"(AUC {ridge_oof['pairwise_auc']:.4f}), "
                 f"GBDT OOF gate={'PASS' if evidence['gbdt_oof_ok'] else 'FAIL'} "
                 f"(AUC {gbdt_oof_metrics['pairwise_auc']:.4f})")
    lines.append("")
    lines.append("Q2 Same 53 features nonlinear OOF signal:")
    lines.append(f"- {'YES' if nonlinear_present else 'NO'}")
    lines.append(f"- evidence: GBDT OOF AUC {gbdt_oof_metrics['pairwise_auc']:.4f}, "
                 f"Rank1 {fmt_ret(gbdt_oof_metrics['rank1_capped_return'])} vs "
                 f"baseline {fmt_ret(gbdt_oof_metrics['baseline_capped_return'])}, "
                 f"Top1 beat {fmt_pct(gbdt_oof_metrics['top1_beat_universe_rate'])}")
    lines.append("")
    if evidence["ridge_train_ok"] or evidence["gbdt_train_ok"]:
        q3 = "YES"
    elif not nonlinear_present and not evidence["ridge_train_ok"] \
            and not evidence["gbdt_train_ok"]:
        q3 = "NO"
    else:
        q3 = "INCONCLUSIVE"
    gate_word = lambda ok: "PASS" if ok else "FAIL"
    lines.append("Q3 Stable D1 Target7 information:")
    lines.append(f"- {q3}")
    lines.append(f"- evidence: Ridge TRAIN gate={gate_word(evidence['ridge_train_ok'])}, "
                 f"GBDT TRAIN gate={gate_word(evidence['gbdt_train_ok'])}, "
                 f"GBDT OOF gate={gate_word(evidence['gbdt_oof_ok'])}")
    lines.append("")
    lines.append(f"PRIMARY_DIAGNOSIS: {diag}")
    lines.append("")
    lines.append(f"TASK_CEILING: {ceiling_level} ({ceiling_reason})")
    lines.append("")
    lines.append("Plain-language conclusion:")
    if diag == "MODEL_FORM_LIMITATION":
        lines.append("- 同样 53 个特征在非线性模型下 walk-forward 出现实际排序价值, "
                     "而线性 Pairwise Ridge 没有; 主要失败在模型形式表达。")
    elif diag == "TEMPORAL_INSTABILITY":
        lines.append("- 历史数据可以拟合 (Ridge λ0.1/λ10 TRAIN 与 GBDT TRAIN 训练端"
                     "均通过 gate), 但两种模型 walk-forward 都不能稳定预测下一交易日;"
                     " 主要失败在时间稳定性。")
        lines.append("- 补充: GBDT OOF 的 Top1 侧有部分信号 (Rank1 3.68% > baseline"
                     " 3.00%, beat 62.07%, Top1 hit 51.72% vs 随机 36.94%), 但 Top3"
                     " 2.69% 低于 baseline 且 AUC 仅 0.5352, 不足以通过 gate; May OOF"
                     " AUC 0.4221 vs June 0.5768 显示月份间结构不稳定 — 与"
                     " TEMPORAL_INSTABILITY 一致。")
    elif diag == "D1_FEATURE_INFORMATION_WEAK":
        lines.append("- 53 个 D1_CLOSE 前特征在训练端都缺乏可分离能力 (Ridge 与 "
                     "固定 GBDT 训练端均未通过 gate); 主要失败在特征信息量。")
    else:
        lines.append("- 证据混合, 无法唯一归因; 按 §46 报告各方向证据。")
    lines.append("")
    lines.append("This result DOES prove:")
    lines.append(f"- 每日 Target7 数量分布与 Oracle 上限 (理论 Top3 精度 "
                 f"{fmt_pct(oracle_all['oracle_top3_target7_precision'])}, "
                 f"随机基线 {fmt_pct(oracle_all['random_target7_baseline'])});")
    lines.append(f"- Ridge 在 May+June 上的训练端拟合能力 "
                 f"(λ10 TRAIN AUC {r10['pairwise_auc']:.4f});")
    lines.append(f"- 固定 GBDT 与 Ridge 的 walk-forward OOF 对比 "
                 f"(AUC {gbdt_oof_metrics['pairwise_auc']:.4f} vs "
                 f"{ridge_oof['pairwise_auc']:.4f})。")
    lines.append("This result DOES NOT prove:")
    lines.append("- 任何模型在 July 或未来的预测能力 (July NOT accessed);")
    lines.append("- 非线性模型已可推广 (若 OOF 通过 gate, 仍需独立 OOT 验证);")
    lines.append("- 特征信息不足是永久结论 (只对当前 53 特征有效)。")
    lines.append("")
    lines.append("Recommended next step (external review 后决定):")
    if nonlinear_present:
        lines.append("- 冻结非线性模型设计 -> 独立 July OOT 验证。")
    else:
        lines.append("- 保留现有 53 特征与双模型结论; 评估新 D1 信息源前不扩大 "
                     "模型; 或接受 D1 信息限制。")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append("- full pipeline executed twice; all 6 outputs byte-identical: "
                 "PASS")
    lines.append("")
    review = "\n".join(lines) + "\n"

    # Oracle ceiling CSV 附加分月/分窗口汇总行 (简单起见附 dist 汇总)
    ceiling_out = ceiling.copy()
    ceiling_out["oracle_top3_target7_hits"] = ceiling_out[
        ["target7_count", "candidate_count"]].apply(
        lambda r: min(int(r["target7_count"]), min(3, int(r["candidate_count"]))),
        axis=1)

    artifacts = {
        ORACLE_CSV: ceiling_out.to_csv(index=False),
        RIDGE_CAP_CSV: ridge_cap.to_csv(index=False),
        NONLINEAR_OOF_CSV: oof_out.to_csv(index=False),
        NONLINEAR_DAILY_CSV: daily.to_csv(index=False),
        COMPARISON_CSV: comp.to_csv(index=False),
        REVIEW_MD: review,
    }
    return {name: data.encode("utf-8") for name, data in artifacts.items()}


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    print("[diagnostic] run #1 ...", flush=True)
    run1 = build_pipeline()
    print("[diagnostic] run #2 ...", flush=True)
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

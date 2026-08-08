# -*- coding: utf-8 -*-
"""v004c Top3 Failure Decomposition Diagnostic v001 — 赢家捕获 vs 修复强度排序。

本模块只做: read / merge / rank / diagnose / report。
**禁止模型训练** (§3, §72): 不 import sklearn, 不调用任何 fit / lambda
选择 / preprocessing 重算。所有分析基于既有 OOF score 与冻结 outcome。

目标: 把 Top3 失败拆成两个不同问题 (§1):
- Problem A WINNER CAPTURE: 真正 >=+7% 的赢家有没有进 Top3;
- Problem B REPAIR-STRENGTH ORDERING: 不足 3 只 Target7 时, 剩余位置
  能否把修复更强的非 Target7 股票排在前面。

核心数学 (§24-§41):
- repair-strength pairwise concordance (all / cross-threshold /
  within-non-target), 日期等权;
- Top3 regret 恒等式: total = winner_capture + repair_ordering;
- winner-fixed counterfactual: 把漏掉的 Target7 (capped=7%) 补回 Top3,
  分离两类 regret;
- Group A (>=3 赢家): repair_ordering 应为 0 (赢家补齐即达 Oracle);
- Group C (0 赢家): winner_capture_regret = 0, 全部 regret 来自
  repair_ordering。

全部输入 max signal_date = 2026-06-30 (July 隔离, §5)。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

CAP_RETURN_7 = 0.07
OOF_SLOT_K = 3
_TOL = 1e-9


# ---------------------------------------------------------------------------
# Rank 重算 (§8) 与存储 rank 审计
# ---------------------------------------------------------------------------
def recompute_day_rank(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """signal_date 内 score 降序, event_id 升序 (§8, §80)。

    对乱序输入也给出确定的 Top3。
    """
    out = df.copy()
    out = out.sort_values(["signal_date", score_col, "event_id"],
                          ascending=[True, False, True]).reset_index(drop=True)
    out["diagnostic_rank"] = out.groupby("signal_date", sort=True).cumcount() + 1
    return out


def audit_stored_rank(df: pd.DataFrame, score_col: str,
                      stored_col: str) -> int | str:
    """stored rank vs recomputed diagnostic_rank 不一致数; 列缺失 -> 字符串。"""
    if stored_col not in df.columns:
        return "NOT_AVAILABLE"
    ranked = recompute_day_rank(df, score_col)
    mism = int((ranked[stored_col].to_numpy(dtype=float)
                != ranked["diagnostic_rank"].to_numpy(dtype=float)).sum())
    return mism


# ---------------------------------------------------------------------------
# 每日基本事实 (§9-§10, §16-§18)
# ---------------------------------------------------------------------------
def date_group(target7_count: int) -> str:
    if target7_count >= 3:
        return "A"
    if target7_count >= 1:
        return "B"
    return "C"


def decompose_day(day: pd.DataFrame, score_col: str) -> dict:
    """单日完整分解 (winner capture + fill quality + regret 恒等式)。

    day 必须含: signal_date / event_id / code / score(经 score_col) /
    target7_daily_d2open_d3high / capped_opportunity_return_7。
    返回全部 §52 字段。
    """
    day = day.sort_values([score_col, "event_id"], ascending=[False, True])
    n = int(len(day))
    K = min(OOF_SLOT_K, n)
    t7 = pd.to_numeric(day["target7_daily_d2open_d3high"],
                       errors="coerce").fillna(0).astype(int).to_numpy()
    capped = day["capped_opportunity_return_7"].to_numpy(float)
    t7_count = int(t7.sum())
    avail = min(K, t7_count)
    group = date_group(t7_count)

    model_top3 = day.head(K)
    oracle_top3 = day.sort_values("capped_opportunity_return_7",
                                  ascending=False).head(K)
    hits = int(model_top3["target7_daily_d2open_d3high"]
               .fillna(0).astype(int).sum())
    capture = hits / avail if avail > 0 else None
    full_capture = bool(hits == avail) if avail > 0 else None
    req_non_target = K - avail
    model_top3_return = float(model_top3["capped_opportunity_return_7"].mean())
    oracle_top3_return = float(oracle_top3["capped_opportunity_return_7"].mean())

    # 非 Target7 成员 (§17-§19)
    model_nt = model_top3[model_top3["target7_daily_d2open_d3high"]
                          .fillna(0).astype(int) == 0]
    selected_nt_count = int(len(model_nt))
    selected_nt_mean = (float(model_nt["capped_opportunity_return_7"].mean())
                        if len(model_nt) else float("nan"))
    non_target = day[day["target7_daily_d2open_d3high"].fillna(0)
                     .astype(int) == 0].sort_values(
        "capped_opportunity_return_7", ascending=False)
    oracle_fill = non_target.head(req_non_target)
    oracle_fill_mean = (float(oracle_fill["capped_opportunity_return_7"].mean())
                        if req_non_target > 0 else float("nan"))
    best_fill_capture = (
        len(set(model_nt["event_id"].astype(str))
            & set(oracle_fill["event_id"].astype(str))) / req_non_target
        if req_non_target > 0 else None)

    # ---- Regret 分解 (§31-§35) ----
    missed = avail - hits
    if missed > 0:
        # 把 Top3 中 capped 最低的 missed 只非 Target7 替换为 7%
        worst_nt = model_nt.sort_values("capped_opportunity_return_7").head(missed)
        fixed = model_top3.copy()
        fixed.loc[worst_nt.index, "capped_opportunity_return_7"] = CAP_RETURN_7
        winner_fixed_return = float(
            fixed["capped_opportunity_return_7"].mean())
    else:
        winner_fixed_return = model_top3_return

    total_regret = oracle_top3_return - model_top3_return
    winner_capture_regret = winner_fixed_return - model_top3_return
    repair_ordering_regret = oracle_top3_return - winner_fixed_return
    # 恒等式 (§36) 与非负 (§37)
    if abs(total_regret - (winner_capture_regret + repair_ordering_regret)) > _TOL:
        raise RuntimeError(f"FATAL regret identity violated: {day.iloc[0]['signal_date']}")
    if min(total_regret, winner_capture_regret, repair_ordering_regret) < -_TOL:
        raise RuntimeError(f"FATAL negative regret: {day.iloc[0]['signal_date']}")

    return {
        "signal_date": str(day.iloc[0]["signal_date"]),
        "month": str(day.iloc[0]["signal_date"])[:7],
        "candidate_count": n,
        "target7_count": t7_count,
        "date_group": group,
        "K": K,
        "available_target7_slots": avail,
        "top3_target7_hits": hits,
        "available_winner_capture_rate": capture,
        "full_available_winner_capture": full_capture,
        "required_non_target_slots": req_non_target,
        "model_top3_capped_return": model_top3_return,
        "oracle_top3_capped_return": oracle_top3_return,
        "selected_non_target_count": selected_nt_count,
        "selected_non_target_mean_capped_return": selected_nt_mean,
        "oracle_non_target_fill_mean_return": oracle_fill_mean,
        "best_non_target_fill_capture_rate": best_fill_capture,
        "missed_target7_winners": missed,
        "winner_fixed_top3_return": winner_fixed_return,
        "total_top3_regret": total_regret,
        "winner_capture_regret": winner_capture_regret,
        "repair_ordering_regret": repair_ordering_regret,
    }


# ---------------------------------------------------------------------------
# Repair-strength concordance (§23-§28)
# ---------------------------------------------------------------------------
def day_repair_concordance(day: pd.DataFrame, score_col: str) -> dict:
    """单日三个 concordance (日期内 capped 不同才参与, §24-§26)。

    返回 (all, cross, within) 各 (concordance, pair_count) 与可评估标记。
    """
    scores = day[score_col].to_numpy(float)
    capped = day["capped_opportunity_return_7"].to_numpy(float)
    t7 = day["target7_daily_d2open_d3high"].fillna(0).astype(int).to_numpy()
    n = len(day)
    all_corr, all_n = 0.0, 0
    cross_corr, cross_n = 0.0, 0
    within_corr, within_n = 0.0, 0
    for i, j in itertools.combinations(range(n), 2):
        if capped[i] == capped[j]:
            continue  # return tie: 不参与 (§24, §79)
        real_better = 1 if capped[i] > capped[j] else -1
        diff = scores[i] - scores[j]
        if diff > 0:
            corr = 1.0 if real_better == 1 else 0.0
        elif diff < 0:
            corr = 1.0 if real_better == -1 else 0.0
        else:
            corr = 0.5  # score tie: 0.5 credit (§78)
        all_corr += corr
        all_n += 1
        if t7[i] != t7[j]:
            cross_corr += corr
            cross_n += 1
        else:
            within_corr += corr
            within_n += 1
    return {
        "all_concordance": (all_corr / all_n) if all_n else None,
        "all_pairs": all_n,
        "cross_concordance": (cross_corr / cross_n) if cross_n else None,
        "cross_pairs": cross_n,
        "within_concordance": (within_corr / within_n) if within_n else None,
        "within_pairs": within_n,
    }


def date_weighted_concordance(day_rows: list[dict]) -> dict:
    """日期等权平均 (§25): 每个可评估日期先算 concordance, 再等权。"""
    out = {}
    for key in ("all_concordance", "cross_concordance", "within_concordance"):
        vals = [r[key] for r in day_rows if r[key] is not None]
        out[key] = float(np.mean(vals)) if vals else float("nan")
        out[key.replace("_concordance", "_evaluable_dates")] = len(vals)
    return out


# ---------------------------------------------------------------------------
# Rank profile (§29-§30, §56)
# ---------------------------------------------------------------------------
def rank_profile(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Rank1/2/3: capped return / Target7 hit / excess vs universe (日期等权)。"""
    ranked = recompute_day_rank(df, score_col)
    ranked["_day_mean"] = ranked.groupby("signal_date")[
        "capped_opportunity_return_7"].transform("mean")
    rows = []
    for pos in (1, 2, 3):
        sub = ranked[ranked["diagnostic_rank"] == pos]
        if sub.empty:
            rows.append({"rank": pos, "n_dates": 0, "mean_capped_return": np.nan,
                         "target7_hit_rate": np.nan, "mean_excess_vs_universe": np.nan})
            continue
        rows.append({
            "rank": pos,
            "n_dates": int(len(sub)),
            "mean_capped_return": float(sub["capped_opportunity_return_7"].mean()),
            "target7_hit_rate": float(sub["target7_daily_d2open_d3high"]
                                      .fillna(0).astype(int).mean()),
            "mean_excess_vs_universe": float(
                (sub["capped_opportunity_return_7"] - sub["_day_mean"]).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Oracle ceiling 真实口径 (§48, §81) — 复用 diagnostic module 口径
# ---------------------------------------------------------------------------
def oracle_ceiling_summary(df: pd.DataFrame) -> dict:
    """Oracle Top1 hit = mean(target7_count>=1); Top3 precision =
    mean(min(target7_count, K)/K)。禁止硬编码 1.0 (§81)。"""
    rows = []
    for d, day in df.groupby("signal_date", sort=True):
        n = len(day)
        K = min(OOF_SLOT_K, n)
        t7 = int(pd.to_numeric(day["target7_daily_d2open_d3high"],
                               errors="coerce").fillna(0).astype(int).sum())
        rows.append({"signal_date": str(d), "n": n, "t7": t7, "K": K})
    f = pd.DataFrame(rows)
    top1_hit = float((f["t7"] >= 1).mean())
    top3_prec = float((f[["t7", "K"]].apply(
        lambda r: min(int(r["t7"]), int(r["K"])) / int(r["K"]), axis=1)).mean())
    top3_ret = float(df.groupby("signal_date")[
        "capped_opportunity_return_7"].apply(
        lambda s: float(np.sort(s.to_numpy(float))[::-1][:min(3, len(s))].mean())
    ).mean())
    return {"oracle_top1_hit_rate": top1_hit,
            "oracle_top3_target7_precision": top3_prec,
            "oracle_top3_capped_return": top3_ret}

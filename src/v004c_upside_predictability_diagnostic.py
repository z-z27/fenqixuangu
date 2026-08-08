# -*- coding: utf-8 -*-
"""v004c Upside Predictability Diagnostic v001 — 模型形式 vs 特征信息诊断。

目的: 用数据区分 Upside Pairwise Ridge 失败的四种可能 (§1):

  H1 MODEL_FORM_LIMITATION         线性形式无法表达可预测信息
  H2 TEMPORAL_INSTABILITY          历史可拟合但不能泛化到下一交易日
  H3 D1_FEATURE_INFORMATION_WEAK   53 个 D1_CLOSE 前特征本身信息不足
  H4 TASK_CEILING_LIMITED          每日 Target7 数量本身不足以支撑 Top3

实验设计 (全部只用冻结 319 rows / 39 dates / 53 FEATURE / Target7):
- 实验 A: Oracle Ceiling — 每天实际 Target7 数量分布 + 理论上限
  (oracle top1/top3 hit 与 capped return, 绝不用于训练);
- 实验 B: Ridge Capacity — 全量 319 行 in-sample fit (λ=0.1, λ=10),
  CAPACITY_DIAGNOSTIC_ONLY, 与已有 λ10 OOF (v001 正式资产, 程序读取验证);
- 实验 C: 固定浅层 GBDT (GradientBoostingClassifier, 硬编码配置) —
  与 Ridge 完全相同的 fold-only preprocessing + 时间升序 walk-forward
  (warmup 10), row weight = 1/N_t (每训练日期总权重 1), 只做
  NONLINEAR_INFORMATION_PROBE。

复用 src/v004c_pairwise_ridge.py 的全部数学: preprocessor / practical
metrics / date-weighted pairwise stats / capped return。禁止新模型、禁止
新 feature、禁止 July 访问。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v004c_pairwise_ridge import (
    BOARD3_STREAK,
    MIN_TRAIN_SIGNAL_DATES,
    build_board3_interaction,
    build_same_date_pairs,
    compute_capped_opportunity_return,
    compute_practical_rank_metrics,
    date_weighted_pair_stats,
    fit_pairwise_ridge,
    fit_preprocessor,
    score_from_z,
    transform_preprocessor,
)

# ---------------------------------------------------------------------------
# 固定非线性对照配置 (§23) — 硬编码, 禁止修改
# ---------------------------------------------------------------------------
GBDT_CONFIG = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_leaf": 10,
    "subsample": 1.0,
    "random_state": 20260809,
}


def make_gbdt():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**GBDT_CONFIG)


# ---------------------------------------------------------------------------
# 通用: 每日 rank (score 降序, event_id 升序) / Top 命中指标
# ---------------------------------------------------------------------------
def day_rank_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """按 signal_date 分组给 score 排名 (score 降序, event_id 升序 tie-break)。

    返回带 day_rank 列的副本。
    """
    out = df.sort_values(["signal_date", score_col, "event_id"],
                         ascending=[True, False, True]).reset_index(drop=True)
    out["day_rank"] = out.groupby("signal_date", sort=True).cumcount() + 1
    return out


def top_hit_metrics(df: pd.DataFrame, score_col: str,
                    label_col: str = "target7_daily_d2open_d3high"):
    """按日期等权的 Top1 Target7 hit / Top3 Target7 precision (§33)。"""
    ranked = day_rank_scores(df, score_col)
    top1 = ranked[ranked["day_rank"] == 1]
    top1_hit = float(top1[label_col].mean())
    top3_prec = float(ranked.groupby("signal_date").apply(
        lambda g: float(g[g["day_rank"] <= 3][label_col].mean()),
        include_groups=False).mean())
    return top1_hit, top3_prec


def date_balanced_binary_logloss(df: pd.DataFrame, score_col: str,
                                 label_col: str = "target7_daily_d2open_d3high"):
    """每日期平均二元交叉熵, 再对所有日期等权平均 (§34)。"""
    p = np.clip(df[score_col].to_numpy(float), 1e-12, 1.0 - 1e-12)
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    df = df.copy()
    df["_bce"] = bce
    return float(df.groupby("signal_date")["_bce"].mean().mean())


# ---------------------------------------------------------------------------
# 实验 A: Oracle Ceiling (§10-§16)
# ---------------------------------------------------------------------------
def compute_oracle_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """每日候选数 / Target7 数 / Oracle Top1-3 / Random baseline (§16)。

    Oracle 只用于显示数据集理论上限, 绝不用于训练。
    """
    rows = []
    for d, day in df.groupby("signal_date", sort=True):
        n = len(day)
        t7 = int(pd.to_numeric(day["target7_daily_d2open_d3high"],
                               errors="coerce").fillna(0).astype(int).sum())
        rets = day["capped_opportunity_return_7"].to_numpy(float)
        top_vals = np.sort(rets)[::-1]
        k = min(3, n)
        row = {
            "signal_date": str(d),
            "candidate_count": n,
            "target7_count": t7,
            "target7_rate": t7 / n,
            "oracle_top1_hit": int(t7 >= 1),
            "oracle_top3_target7_hits": min(t7, k),
            "oracle_top3_precision": min(t7, k) / k,
            "oracle_top1_capped_return": float(top_vals[0]),
            "oracle_top2_capped_return": float(top_vals[1]) if n >= 2 else np.nan,
            "oracle_top3_capped_return": float(top_vals[:k].mean()),
            "random_target7_rate": t7 / n,
            "random_capped_return": float(rets.mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def target7_distribution(df: pd.DataFrame) -> dict:
    """0 / 1 / 2 / >=3 只 Target7 的日期数与占比。"""
    per_date = df.groupby("signal_date")["target7_daily_d2open_d3high"].apply(
        lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).astype(int).sum()))
    n = len(per_date)
    return {
        "n_dates": n,
        "zero": int((per_date == 0).sum()),
        "one": int((per_date == 1).sum()),
        "two": int((per_date == 2).sum()),
        "three_plus": int((per_date >= 3).sum()),
        "zero_pct": float((per_date == 0).mean()),
        "one_pct": float((per_date == 1).mean()),
        "two_pct": float((per_date == 2).mean()),
        "three_plus_pct": float((per_date >= 3).mean()),
    }


def oracle_summary(ceiling: pd.DataFrame) -> dict:
    """跨日期等权的 Oracle / Random 汇总 (§13-§16)。"""
    return {
        "oracle_top1_hit_rate": float(ceiling["oracle_top1_hit"].mean()),
        "oracle_top3_target7_precision": float(
            ceiling["oracle_top3_precision"].mean()),
        "oracle_top1_capped_return": float(
            ceiling["oracle_top1_capped_return"].mean()),
        "oracle_top2_capped_return": float(
            ceiling["oracle_top2_capped_return"].mean()),
        "oracle_top3_capped_return": float(
            ceiling["oracle_top3_capped_return"].mean()),
        "random_target7_baseline": float(ceiling["random_target7_rate"].mean()),
        "random_capped_return_baseline": float(
            ceiling["random_capped_return"].mean()),
    }


# ---------------------------------------------------------------------------
# 实验 B: Ridge Capacity (§17-§21)
# ---------------------------------------------------------------------------
def ridge_in_sample_fit(df: pd.DataFrame, feature_names: list[str],
                        lam: float, label_col: str = "target7_daily_d2open_d3high"):
    """全量 319 行 in-sample fit (CAPACITY_DIAGNOSTIC_ONLY, §19)。

    允许看到全部 May+June label; 不是 OOF, 不是泛化证明。
    """
    x_all = df[feature_names].to_numpy(float)
    y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    b_all = (df["board_streak_before_break"].to_numpy(float) == BOARD3_STREAK)
    params = fit_preprocessor(x_all)
    xs = transform_preprocessor(x_all, params)
    z_diff, weight = build_same_date_pairs(xs, y_all, b_all)
    theta = fit_pairwise_ridge(z_diff, weight, float(lam))
    scores = score_from_z(build_board3_interaction(xs, b_all), theta)
    return _capacity_metrics(df, scores, f"ridge_in_sample_lambda_{lam}")


def _capacity_metrics(df: pd.DataFrame, scores: np.ndarray, tag: str) -> dict:
    """in-sample / OOF 通用指标块 (AUC + 实用收益 + Target7 命中)。"""
    df = df.copy()
    df["score"] = scores
    ll, auc_v, acc_v, nd = date_weighted_pair_stats(
        df["score"].to_numpy(float),
        pd.to_numeric(df["target7_daily_d2open_d3high"], errors="coerce")
        .to_numpy(float),
        df["signal_date"].astype(str).to_numpy())
    m = compute_practical_rank_metrics(df)
    def _get(pos):
        r = m[m["rank_position"] == pos]
        return r.iloc[0]
    r1, t3 = _get(1), _get("Top3")
    top1_hit, top3_prec = top_hit_metrics(df, "score")
    baseline = float(df.groupby("signal_date")
                     ["capped_opportunity_return_7"].mean().mean())
    return {
        "tag": tag,
        "mode": "TRAIN_IN_SAMPLE",
        "pairwise_auc": auc_v,
        "pair_accuracy": acc_v,
        "pairwise_logloss": ll,
        "baseline_capped_return": baseline,
        "rank1_capped_return": float(r1["mean_capped_return"]),
        "rank1_excess": float(r1["mean_excess_vs_daily_universe"]),
        "top3_capped_return": float(t3["mean_capped_return"]),
        "top3_excess": float(t3["mean_excess_vs_daily_universe"]),
        "top1_beat_universe_rate": float(r1["beat_daily_universe_rate"]),
        "top1_target7_hit": top1_hit,
        "top3_target7_precision": top3_prec,
    }


# ---------------------------------------------------------------------------
# 实验 C: 固定浅层 GBDT (§23-§34)
# ---------------------------------------------------------------------------
def gbdt_walkforward(df: pd.DataFrame, feature_names: list[str],
                     label_col: str = "target7_daily_d2open_d3high"):
    """时间升序 walk-forward (warmup 10, 与 Ridge v001 完全一致 §31)。

    - 每 fold: train = signal_date < test_date (硬断言 max(train) < test);
    - fold-only preprocessing 与 Ridge 完全相同 (fit_preprocessor/
      transform_preprocessor);
    - row weight = 1 / N_t (每训练日期总权重 1, §30);
    - GBDT 固定配置 (§23), predict_proba(Target7=1) 作为 score。
    """
    df = df.sort_values("signal_date").reset_index(drop=True)
    dates = sorted(df["signal_date"].unique().tolist())
    if len(dates) <= MIN_TRAIN_SIGNAL_DATES:
        raise RuntimeError("not enough signal dates for warmup")

    x_all = df[feature_names].to_numpy(float)
    y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    oof_score = np.full(len(df), np.nan, dtype=float)
    oof_fold = np.full(len(df), -1, dtype=int)
    fold_rows = []
    for fold_i, test_date in enumerate(dates[MIN_TRAIN_SIGNAL_DATES:]):
        train_mask = df["signal_date"] < test_date
        test_mask = df["signal_date"] == test_date
        train_dates_used = sorted(set(df.loc[train_mask, "signal_date"]))
        if train_dates_used and train_dates_used[-1] >= str(test_date):
            raise RuntimeError(  # FATAL (§60)
                f"gbdt walk-forward chronology violated: max train "
                f"{train_dates_used[-1]} >= test {test_date}")
        params = fit_preprocessor(x_all[train_mask])
        train_xs = transform_preprocessor(x_all[train_mask], params)
        test_xs = transform_preprocessor(x_all[test_mask], params)
        # date-balanced row weights (§30): 每训练日期总权重 = 1
        weights = 1.0 / df.loc[train_mask].groupby("signal_date") \
            ["event_id"].transform("count").to_numpy(float)
        train_y = y_all[train_mask]
        if int(np.isfinite(train_y).sum()) < 2 or float(np.nanmax(train_y)) <= 0 \
                or float(np.nanmin(train_y)) >= 1:
            raise RuntimeError("gbdt train has no positive/negative samples")
        clf = make_gbdt()
        clf.fit(train_xs, train_y, sample_weight=weights)
        oof_score[test_mask] = clf.predict_proba(test_xs)[:, 1]
        oof_fold[test_mask] = fold_i
        fold_rows.append({
            "fold_index": fold_i,
            "test_signal_date": str(test_date),
            "train_signal_dates": len(train_dates_used),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        })

    result = pd.DataFrame({
        "event_id": df["event_id"].astype(str).tolist(),
        "signal_date": df["signal_date"].astype(str).tolist(),
        "board_streak_before_break": df["board_streak_before_break"]
        .astype(int).tolist(),
        "gbdt_oof_score": oof_score,
        "gbdt_oof_fold_index": oof_fold,
    })
    return result, pd.DataFrame(fold_rows)


def gbdt_in_sample(df: pd.DataFrame, feature_names: list[str],
                   label_col: str = "target7_daily_d2open_d3high"):
    """全量 319 行 fit 一次同样固定 GBDT (CAPACITY_DIAGNOSTIC_ONLY §35)。"""
    x_all = df[feature_names].to_numpy(float)
    y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(float)
    params = fit_preprocessor(x_all)
    xs = transform_preprocessor(x_all, params)
    weights = 1.0 / df.groupby("signal_date")["event_id"] \
        .transform("count").to_numpy(float)
    clf = make_gbdt()
    clf.fit(xs, y_all, sample_weight=weights)
    scores = clf.predict_proba(xs)[:, 1]
    metrics = _capacity_metrics(df, scores, "gbdt_in_sample")
    metrics["mode"] = "TRAIN_IN_SAMPLE"
    return metrics, scores


# ---------------------------------------------------------------------------
# 诊断判定 (§40-§46)
# ---------------------------------------------------------------------------
def _passes_gate(m: dict) -> bool:
    """同一 GO gate (§45 上一阶段, §40): Top1/Rank1 > baseline,
    Top3 > baseline, beat > 0.50, pairwise AUC > 0.50。"""
    return bool(
        m["rank1_capped_return"] > m["baseline_capped_return"] + 1e-12
        and m["top3_capped_return"] > m["baseline_capped_return"] + 1e-12
        and m["top1_beat_universe_rate"] > 0.50
        and m["pairwise_auc"] > 0.50)


def diagnose(ridge_train0: dict, ridge_train10: dict, ridge_oof: dict,
             gbdt_train: dict, gbdt_oof: dict) -> tuple[str, dict]:
    """按 §43-§46 规则选择 PRIMARY_DIAGNOSIS (只用冻结 GO gate 作阈值)。"""
    ridge_train_ok = _passes_gate(ridge_train0) or _passes_gate(ridge_train10)
    ridge_oof_ok = _passes_gate(ridge_oof)
    gbdt_train_ok = _passes_gate(gbdt_train)
    gbdt_oof_ok = _passes_gate(gbdt_oof)
    evidence = {
        "ridge_train_ok": ridge_train_ok,
        "ridge_oof_ok": ridge_oof_ok,
        "gbdt_train_ok": gbdt_train_ok,
        "gbdt_oof_ok": gbdt_oof_ok,
    }
    if gbdt_oof_ok and not ridge_oof_ok:
        diag = "MODEL_FORM_LIMITATION"  # §43: 非线性 OOF 通过 gate
    elif (ridge_train_ok or gbdt_train_ok) and not gbdt_oof_ok:
        diag = "TEMPORAL_INSTABILITY"   # §44: 历史可拟合, 未来不能
    elif not ridge_train_ok and not gbdt_train_ok and not gbdt_oof_ok:
        diag = "D1_FEATURE_INFORMATION_WEAK"  # §45: 训练端都弱
    else:
        diag = "MIXED_EVIDENCE"
    return diag, evidence


def task_ceiling_level(oracle: dict) -> tuple[str, str]:
    """TASK_CEILING 分级 (§42): 用 Oracle Top3 Target7 precision 数值表达。

    规则 (透明陈述, 非 magic): >=80% -> HIGH, >=50% -> MODERATE, else LOW。
    """
    p = oracle["oracle_top3_target7_precision"]
    if p >= 0.80:
        return "HIGH", f"oracle top3 precision {p:.1%}"
    if p >= 0.50:
        return "MODERATE", f"oracle top3 precision {p:.1%}"
    return "LOW", f"oracle top3 precision {p:.1%}"

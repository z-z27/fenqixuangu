# -*- coding: utf-8 -*-
"""v004c Repair-Strength Pairwise Ridge v001 — Raw Repair Ordering Objective。

模型名: RAW_REPAIR_PAIRWISE_RIDGE

唯一改变的核心变量 (§3): TRAINING PAIR DEFINITION
- Binary Target7 pair (>=7% positive vs <7% negative) ->
- Raw Repair-Strength pair: 同日任意两只股票, raw return 高者在左,
  raw return 低者在右; 训练目标 score_high > score_low (§16-§17)。

本模块只提供 repair pair 构造 / walk-forward / lambda 选择 / 评价数学,
所有其他组件 (preprocessing / board3 interaction / pairwise ridge 求解器 /
capped opportunity return / practical rank metrics) 直接复用
src/v004c_pairwise_ridge.py (§127)。

Pair 构造纪律 (§18-§26):
- raw return tie (|raw_i - raw_j| <= 1e-12) -> 不生成 pair (§18);
- 不设置 minimum economic gap (§19);
- 三种 pair type (CROSS_THRESHOLD / WITHIN_NON_TARGET / WITHIN_TARGET7)
  一律无类别权重: 同日全部有效 pair 同权 = 1 / N_pair_t (§21-§24, §25);
- 严格 same signal_date pair, 禁止跨日 (§26);
- Target7 不参与 pair direction (§11): t7 只用于评价。

数值口径:
- t7 side 判定 raw >= 0.07 - 1e-9, 与 compute_capped_opportunity_return
  (src/v004c_pairwise_ridge.py) 完全一致;
- repair pairwise logloss = mean(logaddexp(0, -(score_high - score_low)));
- lambda 唯一选择标准 = date-weighted OOF raw-repair pairwise logloss (§37),
  禁止收益参与选择 (§38); tie -> 更大 lambda (§39)。

July 隔离: 本模块不读取任何数据; 所有输入由 runner 以冻结资产传入,
max signal_date <= 2026-06-30 由 runner FATAL 断言 (§42)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v004c_pairwise_ridge import (  # noqa: E402
    BOARD3_STREAK,
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    TARGET7_THRESHOLD,
    build_board3_interaction,
    fit_pairwise_ridge,
    fit_preprocessor,
    score_from_z,
    transform_preprocessor,
)

RAW_TIE_EPS = 1e-12
T7_SIDE_EPS = 1e-9
OOF_SCORE_COL = "repair_oof_score"


def _is_t7_side(raw: np.ndarray) -> np.ndarray:
    """Target7 side 判定 (与 compute_capped_opportunity_return 同口径)。"""
    return np.asarray(raw, dtype=float) >= TARGET7_THRESHOLD - T7_SIDE_EPS


# ---------------------------------------------------------------------------
# Repair pair 构造 (§16-§26)
# ---------------------------------------------------------------------------
def build_same_date_repair_pairs(day_x: np.ndarray, day_raw_return: np.ndarray,
                                 day_board3: np.ndarray):
    """同一天 raw repair ordering pairs (§16-§25)。

    对当天任意两只股票 i, j: raw_i > raw_j -> oriented pair (i, j)
    (high 在左), z_diff = z_high - z_low; 权重 = 1 / N_pairs_date,
    使该日总训练权重 = 1 (§25)。raw tie (<=1e-12) 不生成 pair (§18)。

    返回 (z_diff, weight, meta):
    - z_diff: (N, 2p+1) board3 interaction 差;
    - weight: (N,) 全部 1/N;
    - meta: {n_pairs, n_cross_threshold, n_within_non_target,
             n_within_target7} (§21 分类)。
    """
    x = np.asarray(day_x, dtype=float)
    raw = np.asarray(day_raw_return, dtype=float)
    b = np.asarray(day_board3, dtype=float)
    if x.ndim != 2:
        raise RuntimeError("day_x must be 2D")
    n = int(len(x))
    z = build_board3_interaction(x, b)
    hi_idx: list[int] = []
    lo_idx: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(raw[i] - raw[j]) <= RAW_TIE_EPS:
                continue  # §18: tie -> no pair
            if raw[i] > raw[j]:
                hi_idx.append(i)
                lo_idx.append(j)
            else:
                hi_idx.append(j)
                lo_idx.append(i)
    if not hi_idx:
        return (np.zeros((0, z.shape[1]), dtype=float),
                np.zeros(0, dtype=float),
                {"n_pairs": 0, "n_cross_threshold": 0,
                 "n_within_non_target": 0, "n_within_target7": 0})
    z_diff = z[hi_idx] - z[lo_idx]
    weight = np.full(len(hi_idx), 1.0 / len(hi_idx), dtype=float)
    hi_t7 = _is_t7_side(raw[hi_idx])
    lo_t7 = _is_t7_side(raw[lo_idx])
    meta = {
        "n_pairs": int(len(hi_idx)),
        "n_cross_threshold": int((hi_t7 & ~lo_t7).sum()),
        "n_within_non_target": int((~hi_t7 & ~lo_t7).sum()),
        "n_within_target7": int((hi_t7 & lo_t7).sum()),
    }
    return z_diff, weight, meta


def day_repair_pair_gaps(raw_return: np.ndarray) -> np.ndarray:
    """单日全部 repair pair 的 raw gap (tie 排除), 诊断用 (§20)。"""
    raw = np.asarray(raw_return, dtype=float)
    gaps = []
    n = len(raw)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(raw[i] - raw[j]) <= RAW_TIE_EPS:
                continue
            gaps.append(abs(raw[i] - raw[j]))
    return np.asarray(gaps, dtype=float)


def repair_gap_stats(gaps: np.ndarray) -> dict:
    """gap 分布 (§20): p25/median/p75/p90/max + 小 gap 比例。"""
    g = np.asarray(gaps, dtype=float)
    if g.size == 0:
        return {"gap_p25": float("nan"), "gap_median": float("nan"),
                "gap_p75": float("nan"), "gap_p90": float("nan"),
                "gap_max": float("nan"),
                "pct_gap_lt_0_1pp": float("nan"),
                "pct_gap_lt_0_5pp": float("nan"),
                "pct_gap_lt_1pp": float("nan")}
    return {
        "gap_p25": float(np.quantile(g, 0.25)),
        "gap_median": float(np.quantile(g, 0.50)),
        "gap_p75": float(np.quantile(g, 0.75)),
        "gap_p90": float(np.quantile(g, 0.90)),
        "gap_max": float(g.max()),
        "pct_gap_lt_0_1pp": float((g < 0.001).mean()),
        "pct_gap_lt_0_5pp": float((g < 0.005).mean()),
        "pct_gap_lt_1pp": float((g < 0.01).mean()),
    }


def day_repair_pair_counts(raw_return: np.ndarray) -> dict:
    """单日 pair type 计数 (诊断用, 不构造特征矩阵)。"""
    raw = np.asarray(raw_return, dtype=float)
    n = len(raw)
    n_cross = n_nt = n_t7 = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(raw[i] - raw[j]) <= RAW_TIE_EPS:
                continue
            t7_i = bool(_is_t7_side(raw[i]))
            t7_j = bool(_is_t7_side(raw[j]))
            if t7_i != t7_j:
                n_cross += 1
            elif t7_i:
                n_t7 += 1
            else:
                n_nt += 1
    return {"n_cross_threshold": n_cross,
            "n_within_non_target": n_nt,
            "n_within_target7": n_t7}


# ---------------------------------------------------------------------------
# Repair pairwise 指标 (logloss + 4-type concordance, §37/§59-§61)
# ---------------------------------------------------------------------------
def day_repair_pair_stats(day_score: np.ndarray, day_raw: np.ndarray):
    """单日 repair pairwise logloss + 4-type concordance。

    concordance (§60): raw 高者 score 更高 -> 1, 更低 -> 0, score tie ->
    0.5; raw tie -> exclude。日期内 pair 全部计入。
    返回 None 当: score 非有限 (warmup) 或有效 pair 数 = 0。
    """
    score = np.asarray(day_score, dtype=float)
    raw = np.asarray(day_raw, dtype=float)
    if not np.isfinite(score).all():
        return None
    n = int(len(score))
    ll_sum = 0.0
    all_corr = 0.0
    cross_corr = cross_n = 0.0
    nt_corr = nt_n = 0.0
    t7_corr = t7_n = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(raw[i] - raw[j]) <= RAW_TIE_EPS:
                continue
            if raw[i] > raw[j]:
                hi, lo = i, j
            else:
                hi, lo = j, i
            d = float(score[hi] - score[lo])
            ll_sum += float(np.logaddexp(0.0, -d))
            corr = 1.0 if d > 0 else (0.0 if d < 0 else 0.5)
            all_corr += corr
            n_pairs += 1
            hi_t7 = bool(_is_t7_side(raw[hi]))
            lo_t7 = bool(_is_t7_side(raw[lo]))
            if hi_t7 != lo_t7:
                cross_corr += corr
                cross_n += 1
            elif hi_t7:
                t7_corr += corr
                t7_n += 1
            else:
                nt_corr += corr
                nt_n += 1
    if n_pairs == 0:
        return None
    return {
        "pairwise_logloss": float(ll_sum / n_pairs),
        "n_pairs": int(n_pairs),
        "all_concordance": float(all_corr / n_pairs),
        "n_cross": int(cross_n),
        "cross_concordance": (float(cross_corr / cross_n)
                              if cross_n else None),
        "n_within_non_target": int(nt_n),
        "within_non_target_concordance": (float(nt_corr / nt_n)
                                          if nt_n else None),
        "n_within_target7": int(t7_n),
        "within_target7_concordance": (float(t7_corr / t7_n)
                                       if t7_n else None),
    }


def date_weighted_repair_pair_stats(oof_score: np.ndarray, oof_raw: np.ndarray,
                                    date_idx: np.ndarray) -> dict:
    """日期等权聚合 (§37, §61): 每个可评估日期先算, 再对所有可评估日期等权。

    跳过: score 非有限日期 (warmup), 有效 pair = 0 的日期。
    返回 {pairwise_logloss, all_concordance, cross_concordance,
          within_non_target_concordance, within_target7_concordance,
          evaluable_dates}。
    """
    logs: list[float] = []
    all_c: list[float] = []
    cross_c: list[float] = []
    nt_c: list[float] = []
    t7_c: list[float] = []
    for d in sorted(set(date_idx.tolist())):
        mask = date_idx == d
        st = day_repair_pair_stats(oof_score[mask], oof_raw[mask])
        if st is None:
            continue
        logs.append(st["pairwise_logloss"])
        all_c.append(st["all_concordance"])
        if st["cross_concordance"] is not None:
            cross_c.append(st["cross_concordance"])
        if st["within_non_target_concordance"] is not None:
            nt_c.append(st["within_non_target_concordance"])
        if st["within_target7_concordance"] is not None:
            t7_c.append(st["within_target7_concordance"])
    def _mean(vals):
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "pairwise_logloss": _mean(logs),
        "all_concordance": _mean(all_c),
        "cross_concordance": _mean(cross_c),
        "within_non_target_concordance": _mean(nt_c),
        "within_target7_concordance": _mean(t7_c),
        "evaluable_dates": int(len(logs)),
    }


# ---------------------------------------------------------------------------
# Chronological walk-forward (repair pairs) (§40-§44)
# ---------------------------------------------------------------------------
def chronological_walkforward_repair(
    df: pd.DataFrame,
    feature_names: list[str],
    raw_return_col: str,
    lam: float,
    collect_coefs: bool = False,
) -> dict:
    """严格时间升序 walk-forward; warm-up 前 10 个 signal_date (§7, §40)。

    与 Binary v001 的唯一差别 (§3): 训练 pair 为每 signal_date 内
    build_same_date_repair_pairs (同日, raw ordering, 权重 1/N_pair_date);
    禁止跨日 pair (§26)。OOF 评价为 repair pairwise logloss + 4-type
    concordance (同日的 raw ordering)。

    返回 dict: oof 行 (event_id 对齐, repair_oof_score),
    fold_meta (含 train_pairs / 类型计数 / test repair 指标),
    fold_coefs (可选)。
    """
    df = df.sort_values("signal_date").reset_index(drop=True)
    dates = sorted(df["signal_date"].unique().tolist())
    if len(dates) <= MIN_TRAIN_SIGNAL_DATES:
        raise RuntimeError(
            f"not enough signal dates for warmup: {len(dates)} <= "
            f"{MIN_TRAIN_SIGNAL_DATES}")

    x_all = df[feature_names].to_numpy(dtype=float)
    raw_all = pd.to_numeric(df[raw_return_col],
                            errors="coerce").to_numpy(dtype=float)
    b_all = (df["board_streak_before_break"].to_numpy(dtype=float)
             == BOARD3_STREAK)
    event_ids = df["event_id"].astype(str).tolist()
    signal_dates = df["signal_date"].astype(str).tolist()

    oof_score = np.full(len(df), np.nan, dtype=float)
    oof_fold = np.full(len(df), -1, dtype=int)
    fold_rows: list[dict] = []
    fold_coefs: list[dict] = []
    p = len(feature_names)
    for fold_i, test_date in enumerate(dates[MIN_TRAIN_SIGNAL_DATES:]):
        train_mask = (df["signal_date"] < test_date).to_numpy()
        test_mask = (df["signal_date"] == test_date).to_numpy()
        train_dates_used = sorted(set(signal_dates[i]
                                      for i in np.where(train_mask)[0]))
        if train_dates_used and train_dates_used[-1] >= str(test_date):
            raise RuntimeError(  # FATAL (§40)
                f"walk-forward chronology violated: max train "
                f"{train_dates_used[-1]} >= test {test_date}")

        params = fit_preprocessor(x_all[train_mask])
        train_xs = transform_preprocessor(x_all[train_mask], params)
        raw_tr = raw_all[train_mask]
        b_tr = b_all[train_mask]
        tr_dates = np.asarray(signal_dates, dtype=object)[train_mask]

        # ---- 每 signal_date 独立 repair pairs (§16, §26) ----
        z_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []
        train_pair_types = {"n_cross_threshold": 0, "n_within_non_target": 0,
                            "n_within_target7": 0}
        for td in sorted(set(tr_dates.tolist())):
            d_mask = tr_dates == td
            zd, w, meta = build_same_date_repair_pairs(
                train_xs[d_mask], raw_tr[d_mask], b_tr[d_mask])
            if len(zd):
                z_parts.append(zd)
                w_parts.append(w)
            for k in train_pair_types:
                train_pair_types[k] += int(meta[k])
        if z_parts:
            z_diff = np.vstack(z_parts)
            weight = np.concatenate(w_parts)
        else:
            z_diff = np.zeros((0, 2 * p + 1), dtype=float)
            weight = np.zeros(0, dtype=float)

        theta = fit_pairwise_ridge(z_diff, weight, float(lam))
        test_xs = transform_preprocessor(x_all[test_mask], params)
        test_z = build_board3_interaction(test_xs, b_all[test_mask])
        scores = score_from_z(test_z, theta)
        oof_score[test_mask] = scores
        oof_fold[test_mask] = fold_i

        # ---- test 日 repair 指标 (lambda 选择用, §37) ----
        tst = day_repair_pair_stats(
            scores, raw_all[test_mask])
        fold_rows.append({
            "fold_index": fold_i,
            "test_signal_date": str(test_date),
            "train_signal_dates": len(train_dates_used),
            "train_rows": int(train_mask.sum()),
            "train_pairs": int(len(z_diff)),
            "train_cross_threshold_pairs": train_pair_types["n_cross_threshold"],
            "train_within_non_target_pairs": train_pair_types["n_within_non_target"],
            "train_within_target7_pairs": train_pair_types["n_within_target7"],
            "test_rows": int(test_mask.sum()),
            "test_repair_pairwise_logloss": (tst["pairwise_logloss"]
                                             if tst is not None else float("nan")),
            "test_all_repair_concordance": (tst["all_concordance"]
                                            if tst is not None else float("nan")),
            "test_cross_concordance": (tst["cross_concordance"]
                                       if tst is not None else float("nan")),
            "test_within_non_target_concordance": (
                tst["within_non_target_concordance"]
                if tst is not None else float("nan")),
            "test_within_target7_concordance": (
                tst["within_target7_concordance"]
                if tst is not None else float("nan")),
            "test_pair_evaluable": int(tst is not None),
            "constant_in_fold_features": int(
                np.asarray(params.constant_in_fold).sum()),
        })
        if collect_coefs:
            beta = theta[:p]
            gamma = theta[p + 1:]
            delta = float(theta[p])
            fold_coefs.append({
                "test_signal_date": str(test_date),
                "delta_board3": delta,
                "beta": beta.tolist(),
                "gamma": gamma.tolist(),
            })

    result = pd.DataFrame({
        "event_id": event_ids,
        "signal_date": signal_dates,
        "board_streak_before_break":
            df["board_streak_before_break"].astype(int).tolist(),
        OOF_SCORE_COL: oof_score,
        "oof_fold_index": oof_fold,
    })
    return {"oof": result,
            "fold_meta": pd.DataFrame(fold_rows),
            "fold_coefs": fold_coefs}


# ---------------------------------------------------------------------------
# Lambda selection (§36-§39)
# ---------------------------------------------------------------------------
def select_lambda_repair(df: pd.DataFrame, feature_names: list[str],
                         raw_return_col: str):
    """对 LAMBDA_GRID 全跑 repair walk-forward; 唯一选择标准 = date-weighted
    OOF raw-repair pairwise logloss (§37); tie -> 更大 lambda (§39)。

    禁止任何收益指标参与选择 (§38)。
    返回 (table, selected_lambda)。
    """
    rows: list[dict] = []
    for lam in LAMBDA_GRID:
        res = chronological_walkforward_repair(df, feature_names,
                                               raw_return_col, lam)
        scores = res["oof"][OOF_SCORE_COL].to_numpy(dtype=float)
        raw = pd.to_numeric(df[raw_return_col], errors="coerce").to_numpy(dtype=float)
        dates = df["signal_date"].astype(str).to_numpy()
        agg = date_weighted_repair_pair_stats(scores, raw, dates)
        rows.append({
            "lambda": float(lam),
            "date_weighted_oof_repair_pairwise_logloss": agg["pairwise_logloss"],
            "evaluable_signal_dates": agg["evaluable_dates"],
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["date_weighted_oof_repair_pairwise_logloss", "lambda"],
        ascending=[True, False]).iloc[0]
    return table, float(best["lambda"])

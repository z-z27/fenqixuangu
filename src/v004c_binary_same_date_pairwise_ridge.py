# -*- coding: utf-8 -*-
"""v004c Corrected Same-Date Binary Pairwise Ridge v002 — 严格同日 Binary control。

模型名: CORRECTED_BINARY_SAME_DATE_PAIRWISE_RIDGE_V002

背景 (§1-§4): v004c Pairwise Ridge v001 的 chronological_walkforward 把整个
train 集合一次性传入 build_same_date_pairs (该函数无日期参数), 实际构造
all-historical-positive x all-historical-negative 的 CROSS-DATE pairs
(某 fold: 1749 = 33 x 53, 而真正同日和为 195)。因此历史 v001 不能作为严格
同日 Binary control。本模块提供严格同日版本的 Binary 训练 (per signal_date
分组构造 pairs), 用于与 SAME_DATE_RAW_REPAIR (repair v001) 进行 objective A/B。

Pair 构造纪律 (§11-§17):
- build_same_date_binary_pairs 只接收一天的数据; caller 必须按 signal_date
  分组调用 (§11-§12);
- 每天: positive_t (Target7=1) x negative_t (Target7=0), 禁止跨日期配对 (§12);
- 单类日期 -> 0 binary training pairs, 不跨日期补 pair (§13);
- 每日期总权重 = 1: weight = 1 / (N_pair_t) = 1 / (N_pos_t * N_neg_t) (§14);
- 不可 pair 日期贡献 0 权重 (§15);
- 禁止为 pair 数量对齐而抽样/补 pair (§17)。

与 Repair v001 的共同点 (§18-§23, §28): 同一 preprocessing / board3
interaction / pairwise ridge solver / lambda grid / chronological walk-forward
(warmup 10, 29 OOF dates) / 评价口径。唯一主要差别: TRAINING OBJECTIVE
(binary Target7 boundary vs raw repair ordering)。

本模块不读取任何数据; max signal_date <= 2026-06-30 由 runner FATAL 断言 (§32)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v004c_pairwise_ridge import (  # noqa: E402
    BOARD3_STREAK,
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    _same_date_pair_stats,
    build_board3_interaction,
    date_weighted_pair_stats,
    fit_pairwise_ridge,
    fit_preprocessor,
    score_from_z,
    transform_preprocessor,
)

OOF_SCORE_COL = "binary_same_date_oof_score"
_WEIGHT_TOL = 1e-9


# ---------------------------------------------------------------------------
# Same-date binary pair builder (§11-§15)
# ---------------------------------------------------------------------------
def build_same_date_binary_pairs(day_x: np.ndarray, day_y: np.ndarray,
                                 day_board3: np.ndarray):
    """同一天 Target7 positive x negative 的全组合对 (§12, §14)。

    本函数只接受一天的数据。positive = y==1, negative = y==0;
    每对权重 = 1 / (N_pos * N_neg) -> 该日总训练权重 = 1。
    单类日期返回空数组 (§13)。

    返回 (z_diff, weight): z_diff (N, 2p+1) board3 interaction 差。
    """
    x = np.asarray(day_x, dtype=float)
    y = np.asarray(day_y, dtype=float).astype(int)
    b = np.asarray(day_board3, dtype=float)
    if x.ndim != 2:
        raise RuntimeError("day_x must be 2D")
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.zeros((0, 2 * x.shape[1] + 1), dtype=float), \
            np.zeros(0, dtype=float)
    z = build_board3_interaction(x, b)
    z_diff = (z[pos][:, None, :] - z[neg][None, :, :]).reshape(-1, z.shape[1])
    weight = np.full(len(z_diff), 1.0 / (n_pos * n_neg), dtype=float)
    return z_diff, weight


# ---------------------------------------------------------------------------
# Chronological walk-forward (per-date binary pairs) (§28-§30)
# ---------------------------------------------------------------------------
def chronological_walkforward_binary_same_date(
    df: pd.DataFrame,
    feature_names: list[str],
    label_col: str,
    lam: float,
    collect_coefs: bool = False,
) -> dict:
    """严格时间升序 walk-forward; 训练 pair 每 signal_date 独立构造 (§29)。

    与 v001 (src/v004c_pairwise_ridge.py chronological_walkforward) 的唯一
    实现差别: 训练 pair 按日期分组 (per train_date 调用
    build_same_date_binary_pairs), 禁止跨日 pair (§12, §29-§30)。

    fold 权重审计 (§58): 若 E = 该 fold 的 binary-pair-evaluable train dates
    (当天同时存在 positive 和 negative), 则 sum(all pair weights) 必须 == E
    (每日期权重和=1); 违反 FATAL。

    返回 dict: oof 行 (event_id 对齐, binary_same_date_oof_score),
    fold_meta (含 train pair 审计列), fold_coefs (可选)。
    """
    df = df.sort_values("signal_date").reset_index(drop=True)
    dates = sorted(df["signal_date"].unique().tolist())
    if len(dates) <= MIN_TRAIN_SIGNAL_DATES:
        raise RuntimeError(
            f"not enough signal dates for warmup: {len(dates)} <= "
            f"{MIN_TRAIN_SIGNAL_DATES}")

    x_all = df[feature_names].to_numpy(dtype=float)
    y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
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
            raise RuntimeError(  # FATAL (§28)
                f"walk-forward chronology violated: max train "
                f"{train_dates_used[-1]} >= test {test_date}")

        params = fit_preprocessor(x_all[train_mask])
        train_xs = transform_preprocessor(x_all[train_mask], params)
        y_tr = y_all[train_mask]
        b_tr = b_all[train_mask]
        tr_dates = np.asarray(signal_dates, dtype=object)[train_mask]

        # ---- 每 signal_date 独立 binary pairs (§29) ----
        z_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []
        evaluable_dates = 0
        single_class_dates = 0
        n_pairs = 0
        for td in sorted(set(tr_dates.tolist())):
            d_mask = tr_dates == td
            day_y = y_tr[d_mask]
            n_pos_d = int((day_y == 1).sum())
            n_neg_d = int((day_y == 0).sum())
            if n_pos_d == 0 or n_neg_d == 0:
                single_class_dates += 1
                continue
            evaluable_dates += 1
            zd, w = build_same_date_binary_pairs(
                train_xs[d_mask], day_y, b_tr[d_mask])
            if len(zd):
                z_parts.append(zd)
                w_parts.append(w)
                n_pairs += int(len(zd))
        if z_parts:
            z_diff = np.vstack(z_parts)
            weight = np.concatenate(w_parts)
        else:
            z_diff = np.zeros((0, 2 * p + 1), dtype=float)
            weight = np.zeros(0, dtype=float)

        # §58: 每日期权重和=1 -> sum(weight) == evaluable train dates
        if abs(float(weight.sum()) - float(evaluable_dates)) > _WEIGHT_TOL:
            raise RuntimeError(  # FATAL (§58)
                f"FATAL fold weight audit: sum(weight) {float(weight.sum())} "
                f"!= evaluable train dates {evaluable_dates}")

        theta = fit_pairwise_ridge(z_diff, weight, float(lam))
        test_xs = transform_preprocessor(x_all[test_mask], params)
        test_z = build_board3_interaction(test_xs, b_all[test_mask])
        scores = score_from_z(test_z, theta)
        oof_score[test_mask] = scores
        oof_fold[test_mask] = fold_i

        # ---- test 日 binary pairwise 指标 (§24-§25) ----
        ll, auc_v, acc_v, _ = _same_date_pair_stats(
            scores, y_all[test_mask])
        fold_rows.append({
            "fold_index": fold_i,
            "test_signal_date": str(test_date),
            "train_signal_dates": len(train_dates_used),
            "train_rows": int(train_mask.sum()),
            "train_binary_pair_evaluable_dates": evaluable_dates,
            "train_single_class_dates": single_class_dates,
            "train_same_date_pairs": n_pairs,
            "train_total_weight": float(weight.sum()),
            "train_cross_date_pairs": 0,
            "test_rows": int(test_mask.sum()),
            "test_binary_pairwise_logloss": ll,
            "test_binary_pairwise_auc": auc_v,
            "test_binary_pairwise_accuracy": acc_v,
            "test_binary_evaluable": int(isinstance(ll, (float, np.floating))),
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
# Lambda selection (§23-§27)
# ---------------------------------------------------------------------------
def select_lambda_binary_same_date(df: pd.DataFrame, feature_names: list[str],
                                   label_col: str):
    """对 LAMBDA_GRID 全跑 walk-forward; 唯一选择标准 = date-weighted OOF
    same-date binary pairwise logloss (§24); tie -> 更大 lambda (§27)。

    单类 test date (NOT_EVALUABLE) 不进入 mean (§25)。禁止收益参与选择 (§26)。
    返回 (table, selected_lambda)。
    """
    rows: list[dict] = []
    for lam in LAMBDA_GRID:
        res = chronological_walkforward_binary_same_date(
            df, feature_names, label_col, lam)
        scores = res["oof"][OOF_SCORE_COL].to_numpy(dtype=float)
        y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
        dates = df["signal_date"].astype(str).to_numpy()
        logloss, auc_v, acc_v, ndates = date_weighted_pair_stats(scores, y, dates)
        rows.append({
            "lambda": float(lam),
            "date_weighted_oof_binary_pairwise_logloss": logloss,
            "date_weighted_oof_binary_pairwise_auc": auc_v,
            "date_weighted_oof_binary_pair_accuracy": acc_v,
            "evaluable_signal_dates": ndates,
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["date_weighted_oof_binary_pairwise_logloss", "lambda"],
        ascending=[True, False]).iloc[0]
    return table, float(best["lambda"])

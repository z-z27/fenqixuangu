# -*- coding: utf-8 -*-
"""v004c Date-Conditional Pairwise Ridge — 开发期核心模型 (v001)。

模型名: DATE_CONDITIONAL_PAIRWISE_RIDGE

核心原理 (§7-§11, §17-§19):
- 只生成 same-date pairwise 比较 (positive vs negative, 同日候选之间);
- score_i = x_i · beta + b_i * delta + b_i * (x_i · gamma),
  b_i = 1 当 board_streak_before_break == 3, 否则 0;
- 统一 lambda 约束 ||beta||^2 + delta^2 + ||gamma||^2;
- fold-only preprocessing: train q01/q99 clip -> train median impute (含
  structural missing) -> train mean/std z-score; test 只用 train 参数;
- board3 interaction 在 preprocessing 之后构造 (scaled_feature * b), 不重复
  clip / 不再标准化;
- chronological walk-forward: train = signal_date < test_date, warmup 10 dates;
- pair weight = 1 / (N_pos_t * N_neg_t) -> 每个可 pair 日期总训练权重 = 1;
- lambda 唯一选择标准: date-weighted OOF pairwise logloss (禁止用收益选 lambda)。

数值稳定性: pairwise logistic loss 用 np.logaddexp(0, -d); 求解器为阻尼
Newton (参照 src/v004b.py fit_pairwise_logistic_l2 的数学 kernel)。

本模块只做模型/评价数学, 不含任何 July 数据读取; 所有输入均由 builder
(tools/build_v004c_pairwise_ridge_v001.py) 以冻结 v002 资产传入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 预声明常量 (§12, §21, §29, §27)
# ---------------------------------------------------------------------------
LAMBDA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
MIN_TRAIN_SIGNAL_DATES = 10
TARGET7_THRESHOLD = 0.07
CAP_RETURN_7 = 0.07
BOARD3_STREAK = 3
NOT_EVALUABLE_SINGLE_CLASS = "NOT_EVALUABLE_SINGLE_CLASS"
_TOL = 1e-9


# ---------------------------------------------------------------------------
# Fold-only preprocessing (§17-§18)
# ---------------------------------------------------------------------------
class PreprocParams:
    """train-only 拟合出的 preprocessing 参数 (确定性, 可序列化)。"""

    __slots__ = ("q01", "q99", "median", "mean", "std", "constant_in_fold")

    def __init__(self, q01, q99, median, mean, std, constant_in_fold):
        self.q01 = np.asarray(q01, dtype=float)
        self.q99 = np.asarray(q99, dtype=float)
        self.median = np.asarray(median, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.constant_in_fold = np.asarray(constant_in_fold, dtype=bool)

    def to_dict(self) -> dict:
        return {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "constant_in_fold": self.constant_in_fold.tolist(),
        }


def fit_preprocessor(train_x: np.ndarray, clip_low: float = 0.01,
                     clip_high: float = 0.99) -> PreprocParams:
    """Step 1-4: 只使用 training 观测值拟合 q01/q99 -> median -> mean/std。"""
    x = np.asarray(train_x, dtype=float)
    n_feat = x.shape[1]
    q01 = np.full(n_feat, np.nan, dtype=float)
    q99 = np.full(n_feat, np.nan, dtype=float)
    median = np.full(n_feat, 0.0, dtype=float)
    mean = np.full(n_feat, 0.0, dtype=float)
    std = np.full(n_feat, 1.0, dtype=float)
    const = np.zeros(n_feat, dtype=bool)
    for j in range(n_feat):
        col = x[:, j]
        obs = col[np.isfinite(col)]
        if obs.size == 0:
            # 整列在 train 中全 NaN (不可能退化的极端情形): 中性占位, 保持确定
            q01[j], q99[j], median[j], mean[j], std[j] = 0.0, 0.0, 0.0, 0.0, 1.0
            const[j] = True
            continue
        q01[j] = float(np.quantile(obs, clip_low))
        q99[j] = float(np.quantile(obs, clip_high))
        # Step 2: clip (NaN 保持 NaN)
        clipped = np.clip(col, q01[j], q99[j])
        # Step 3: train clipped median 用于 impute (含 structural missing)
        if np.isfinite(clipped).any():
            median[j] = float(np.median(clipped[np.isfinite(clipped)]))
        else:
            median[j] = 0.0
        imputed = np.where(np.isfinite(clipped), clipped, median[j])
        # Step 4: train imputed mean/std
        mean[j] = float(imputed.mean())
        var = float(np.mean((imputed - mean[j]) ** 2))
        std[j] = float(np.sqrt(var)) if var > 0.0 else 0.0
        if std[j] == 0.0:
            const[j] = True
            std[j] = 1.0  # 占位 (transform 时该列恒 0, 不除 0)
    return PreprocParams(q01, q99, median, mean, std, const)


def transform_preprocessor(x: np.ndarray, params: PreprocParams) -> np.ndarray:
    """Step 2-4 应用于 train/test: 用 train 参数 clip -> impute -> z-score。"""
    x = np.asarray(x, dtype=float)
    out = np.clip(x, params.q01, params.q99)
    out = np.where(np.isfinite(out), out, params.median)
    out = (out - params.mean) / params.std
    out[:, params.constant_in_fold] = 0.0  # §18: fold 内 constant -> scaled = 0
    return out


# ---------------------------------------------------------------------------
# board3 interaction 构造 (§19) 与 score (§8)
# ---------------------------------------------------------------------------
def build_board3_interaction(scaled_x: np.ndarray, board3: np.ndarray) -> np.ndarray:
    """z_i = concat([scaled_x_i, b_i, b_i * scaled_x_i]) (长度 2p+1)。

    b_i 来自 board_streak_before_break == 3 (identity 列, 非 FEATURE)。
    interaction 不单独 clip / 不再标准化。
    """
    scaled_x = np.asarray(scaled_x, dtype=float)
    b = np.asarray(board3, dtype=float).reshape(-1, 1)
    return np.hstack([scaled_x, b, b * scaled_x])


def score_from_z(z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """score = z · theta。"""
    return np.asarray(z, dtype=float) @ np.asarray(theta, dtype=float)


def build_same_date_pairs(day_x: np.ndarray, day_y: np.ndarray,
                          day_board3: np.ndarray):
    """同一天 positive vs negative 的全组合对。

    返回 (z_diff, weight): 每对权重 = 1 / (N_pos * N_neg),
    使该日总训练权重 = 1 (§13)。单类日期返回空数组。
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
        return np.zeros((0, 2 * x.shape[1] + 1), dtype=float), np.zeros(0, dtype=float)
    z = build_board3_interaction(x, b)
    z_diff = (z[pos][:, None, :] - z[neg][None, :, :]).reshape(-1, z.shape[1])
    weight = np.full(len(z_diff), 1.0 / (n_pos * n_neg), dtype=float)
    return z_diff, weight


# ---------------------------------------------------------------------------
# Pairwise logistic loss + Ridge (§10-§11)
# ---------------------------------------------------------------------------
def _sigmoid(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    pos = values >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_pos = np.exp(values[~pos])
    out[~pos] = exp_pos / (1.0 + exp_pos)
    return out


def pairwise_ridge_objective(theta: np.ndarray, z_diff: np.ndarray,
                             weight: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """pairwise logistic loss + ridge; 返回 (loss, analytic gradient)。"""
    theta = np.asarray(theta, dtype=float)
    z_diff = np.asarray(z_diff, dtype=float)
    w = np.asarray(weight, dtype=float)
    wsum = float(w.sum())
    if wsum <= 0:
        raise RuntimeError("pairwise weight sum must be positive")
    margin = z_diff @ theta
    p = _sigmoid(margin)
    grad = (z_diff.T @ (w * (p - 1.0))) / wsum + float(lam) * theta
    loss = float(np.sum(w * np.logaddexp(0.0, -margin)) / wsum)
    loss += 0.5 * float(lam) * float(theta @ theta)
    return loss, grad


def fit_pairwise_ridge(z_diff: np.ndarray, weight: np.ndarray, lam: float,
                       max_iter: int = 200, tol: float = 1e-9) -> np.ndarray:
    """阻尼 Newton 求解 pairwise logistic + ridge (参照 v004b kernel)。

    theta 结构: [beta (p), delta (1), gamma (p)]。
    """
    z_diff = np.asarray(z_diff, dtype=float)
    if z_diff.ndim != 2:
        raise RuntimeError("z_diff must be 2D")
    weight = np.asarray(weight, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    if z_diff.shape[0] == 0:
        return np.zeros(z_diff.shape[1], dtype=float)
    wsum = float(weight.sum())
    if wsum <= 0:
        raise RuntimeError("pairwise sample_weight sum must be positive")

    lam = float(lam)
    theta = np.zeros(z_diff.shape[1], dtype=float)
    prev_loss, _ = pairwise_ridge_objective(theta, z_diff, weight, lam)
    for _ in range(int(max_iter)):
        margin = z_diff @ theta
        p = _sigmoid(margin)
        grad = (z_diff.T @ (weight * (p - 1.0))) / wsum + lam * theta
        variance = weight * p * (1.0 - p)
        hessian = (z_diff.T * variance) @ z_diff / wsum
        hessian[np.diag_indices_from(hessian)] += lam
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        scale = 1.0
        loss = prev_loss
        while scale >= 1e-4:
            candidate = theta - scale * step
            loss, _ = pairwise_ridge_objective(candidate, z_diff, weight, lam)
            if loss <= prev_loss:
                theta = candidate
                break
            scale *= 0.5
        else:
            break
        if abs(prev_loss - loss) < float(tol):
            break
        prev_loss = loss
    return theta


# ---------------------------------------------------------------------------
# 同日期对指标 (§23, §42, §44)
# ---------------------------------------------------------------------------
def _same_date_pair_stats(day_score: np.ndarray, day_y: np.ndarray):
    """单日 pairwise logloss / AUC / accuracy。

    单类日期: 返回 (NOT_EVALUABLE_SINGLE_CLASS, ...) 全部 None。
    """
    score = np.asarray(day_score, dtype=float)
    y = np.asarray(day_y, dtype=float).astype(int)
    if not np.isfinite(score).all():
        return NOT_EVALUABLE_SINGLE_CLASS, None, None, None
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return NOT_EVALUABLE_SINGLE_CLASS, None, None, None
    s_pos = score[pos]
    s_neg = score[neg]
    n_pair = n_pos * n_neg
    diffs = s_pos[:, None] - s_neg[None, :]
    flat = diffs.reshape(-1)
    logloss = float(np.logaddexp(0.0, -flat).mean())
    acc = float((flat > 0.0).mean())
    # AUC: 单日 rank-based (pos 排在 neg 上面的概率)
    auc = float((diffs > 0.0).mean() + 0.5 * (diffs == 0.0).mean())
    return logloss, auc, acc, int(n_pair)


def date_weighted_pair_stats(oof_score: np.ndarray, oof_y: np.ndarray,
                             date_idx: np.ndarray) -> tuple[float, float, float, int]:
    """跨日等权: 每个可评估日期先算平均, 再对所有可评估日期等权平均 (§23)。

    单类日期跳过 (NOT_EVALUABLE_SINGLE_CLASS); 无可评估日期时全部 NaN。
    返回 (logloss, auc, accuracy, evaluable_dates)。
    """
    logs, aucs, accs, ndates = [], [], [], 0
    for d in sorted(set(date_idx.tolist())):
        mask = date_idx == d
        day_score = oof_score[mask]
        if not np.isfinite(day_score).all():
            continue  # 无 OOF score 的日期 (warm-up) 跳过
        ll, auc_v, acc_v, _ = _same_date_pair_stats(day_score, oof_y[mask])
        if not isinstance(ll, (float, np.floating)):
            continue  # 单类日期不可评估
        logs.append(ll)
        aucs.append(auc_v)
        accs.append(acc_v)
        ndates += 1
    if ndates == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.mean(logs)), float(np.mean(aucs)),
            float(np.mean(accs)), ndates)


# ---------------------------------------------------------------------------
# Chronological walk-forward (§20-§22)
# ---------------------------------------------------------------------------
def chronological_walkforward(
    df: pd.DataFrame,
    feature_names: list[str],
    label_col: str,
    lam: float,
    collect_coefs: bool = False,
) -> dict:
    """严格时间升序 walk-forward; 前 10 个 signal_date 为 warm-up。

    train = signal_date < test_date (硬断言 max(train) < test_date, 违反 FATAL);
    test = signal_date == test_date; 每个 test 日: fit preprocessor on train,
    transform train -> same-date pairs -> fit -> predict test scores。
    返回 dict: oof 行 (event_id 对齐)、fold 元数据、fold 系数 (可选)。
    """
    df = df.sort_values("signal_date").reset_index(drop=True)
    dates = sorted(df["signal_date"].unique().tolist())
    if len(dates) <= MIN_TRAIN_SIGNAL_DATES:
        raise RuntimeError(
            f"not enough signal dates for warmup: {len(dates)} <= "
            f"{MIN_TRAIN_SIGNAL_DATES}")

    x_all = df[feature_names].to_numpy(dtype=float)
    y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
    b_all = (df["board_streak_before_break"].to_numpy(dtype=float) == BOARD3_STREAK)
    event_ids = df["event_id"].astype(str).tolist()
    signal_dates = df["signal_date"].astype(str).tolist()

    oof_score = np.full(len(df), np.nan, dtype=float)
    oof_fold = np.full(len(df), -1, dtype=int)
    fold_rows: list[dict] = []
    fold_coefs: list[dict] = []
    p = len(feature_names)
    for fold_i, test_date in enumerate(dates[MIN_TRAIN_SIGNAL_DATES:]):
        train_mask = df["signal_date"] < test_date
        test_mask = df["signal_date"] == test_date
        train_dates_used = sorted(set(signal_dates[i] for i in np.where(train_mask)[0]))
        if train_dates_used and train_dates_used[-1] >= str(test_date):
            raise RuntimeError(  # FATAL (§22)
                f"walk-forward chronology violated: max train {train_dates_used[-1]} "
                f">= test {test_date}")
        train_x = x_all[train_mask]
        train_y = y_all[train_mask]
        test_x = x_all[test_mask]

        params = fit_preprocessor(train_x)
        train_xs = transform_preprocessor(train_x, params)
        z_diff, weight = build_same_date_pairs(
            train_xs, train_y, b_all[train_mask])
        theta = fit_pairwise_ridge(z_diff, weight, float(lam))
        test_xs = transform_preprocessor(test_x, params)
        test_z = build_board3_interaction(test_xs, b_all[test_mask])
        scores = score_from_z(test_z, theta)
        oof_score[test_mask] = scores
        oof_fold[test_mask] = fold_i

        # fold 元数据
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        n_pos = int(pd.to_numeric(pd.Series(train_y), errors="coerce").eq(1).sum())
        n_neg = int(pd.to_numeric(pd.Series(train_y), errors="coerce").eq(0).sum())
        n_pairs = int(len(z_diff))
        ll, auc_v, acc_v, nd = date_weighted_pair_stats(
            scores, y_all[test_mask],
            np.asarray(signal_dates, dtype=object)[test_mask])
        fold_rows.append({
            "fold_index": fold_i,
            "test_signal_date": str(test_date),
            "train_signal_dates": len(train_dates_used),
            "train_rows": n_train,
            "train_positive": n_pos,
            "train_negative": n_neg,
            "train_pairs": n_pairs,
            "test_rows": n_test,
            "test_pairwise_logloss": ll,
            "test_pairwise_auc": auc_v,
            "test_pairwise_accuracy": acc_v,
            "test_evaluable_dates": nd,
            "constant_in_fold_features": int(np.asarray(params.constant_in_fold).sum()),
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
        "board_streak_before_break": df["board_streak_before_break"].astype(int).tolist(),
        f"{label_col}_oof_score": oof_score,
        "oof_fold_index": oof_fold,
    })
    return {
        "oof": result,
        "fold_meta": pd.DataFrame(fold_rows),
        "fold_coefs": fold_coefs,
    }


# ---------------------------------------------------------------------------
# Lambda selection (§23-§25)
# ---------------------------------------------------------------------------
def select_lambda(df: pd.DataFrame, feature_names: list[str], label_col: str):
    """对 LAMBDA_GRID 全跑 walk-forward; 主标准 = date-weighted OOF pairwise
    logloss; tie -> 更大 lambda (§25)。禁止收益参与选择。
    """
    rows: list[dict] = []
    for lam in LAMBDA_GRID:
        res = chronological_walkforward(df, feature_names, label_col, lam)
        scores = res["oof"][f"{label_col}_oof_score"].to_numpy(dtype=float)
        y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)
        dates = df["signal_date"].astype(str).to_numpy()
        logloss, auc_v, acc_v, ndates = date_weighted_pair_stats(scores, y, dates)
        rows.append({
            "lambda": float(lam),
            "date_weighted_oof_pairwise_logloss": logloss,
            "date_weighted_oof_pairwise_auc": auc_v,
            "date_weighted_oof_pair_accuracy": acc_v,
            "evaluable_signal_dates": ndates,
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["date_weighted_oof_pairwise_logloss", "lambda"],
        ascending=[True, False]).iloc[0]
    return table, float(best["lambda"])


# ---------------------------------------------------------------------------
# Outcome: capped opportunity return (§27-§30)
# ---------------------------------------------------------------------------
def compute_capped_opportunity_return(d2_open: np.ndarray, d3_high: np.ndarray,
                                      target7: np.ndarray) -> pd.DataFrame:
    """raw = d3_high/d2_open - 1; capped = min(raw, 0.07) (负收益不截断)。

    硬断言 target7 == (raw >= 0.07) (§29), 不一致 FATAL。
    返回值只用于 OOF evaluation / historical report (OUTCOME_ONLY)。
    """
    d2_open = np.asarray(d2_open, dtype=float)
    d3_high = np.asarray(d3_high, dtype=float)
    raw = d3_high / d2_open - 1.0
    capped = np.minimum(raw, CAP_RETURN_7)
    t7 = np.asarray(target7, dtype=float)
    valid = np.isfinite(raw) & np.isfinite(t7)
    if not np.allclose((raw[valid] >= TARGET7_THRESHOLD - 1e-9).astype(float),
                       t7[valid], atol=1e-9):
        n_bad = int((np.abs((raw[valid] >= TARGET7_THRESHOLD - 1e-9).astype(float)
                            - t7[valid]) > 1e-9).sum())
        raise RuntimeError(  # FATAL (§29)
            f"target7-return consistency violated for {n_bad} rows")
    return pd.DataFrame({
        "raw_opportunity_return": raw,
        "capped_opportunity_return_7": capped,
    })


# ---------------------------------------------------------------------------
# Practical ranking metrics (§32-§39)
# ---------------------------------------------------------------------------
def compute_practical_rank_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    """每日 Rank1-5 / Top1-3 等权机会收益 + daily universe baseline + beat rate。

    跨日期再按日期等权 (§35)。每天候选不足某 rank 则该日不进该 rank 统计
    (§33)。返回 rank-position 汇总表。
    """
    rows: list[dict] = []
    for pos in (1, 2, 3, 4, 5):
        rets, bases, n_dates = [], [], 0
        for _, day in oof_df.groupby("signal_date", sort=True):
            if len(day) < pos:
                continue
            base = float(day["capped_opportunity_return_7"].mean())
            rank_row = day.sort_values(["score", "event_id"],
                                       ascending=[False, True]).iloc[pos - 1]
            rets.append(float(rank_row["capped_opportunity_return_7"]))
            bases.append(base)
            n_dates += 1
        rets = np.asarray(rets, dtype=float)
        bases = np.asarray(bases, dtype=float)
        rows.append({
            "rank_position": pos,
            "number_of_dates_available": n_dates,
            "mean_capped_return": float(rets.mean()) if n_dates else float("nan"),
            "median_capped_return": float(np.median(rets)) if n_dates else float("nan"),
            "positive_return_rate": float((rets > 0).mean()) if n_dates else float("nan"),
            "mean_excess_vs_daily_universe": float((rets - bases).mean()) if n_dates else float("nan"),
            "beat_daily_universe_rate": float((rets > bases).mean()) if n_dates else float("nan"),
        })
    for k in (1, 2, 3):
        top_rets, bases, n_dates = [], [], 0
        for _, day in oof_df.groupby("signal_date", sort=True):
            if len(day) < k:
                continue
            base = float(day["capped_opportunity_return_7"].mean())
            top = day.sort_values(["score", "event_id"],
                                  ascending=[False, True]).head(k)
            top_rets.append(float(top["capped_opportunity_return_7"].mean()))
            bases.append(base)
            n_dates += 1
        top_rets = np.asarray(top_rets, dtype=float)
        bases = np.asarray(bases, dtype=float)
        rows.append({
            "rank_position": f"Top{k}",
            "number_of_dates_available": n_dates,
            "mean_capped_return": float(top_rets.mean()) if n_dates else float("nan"),
            "median_capped_return": float(np.median(top_rets)) if n_dates else float("nan"),
            "positive_return_rate": float((top_rets > 0).mean()) if n_dates else float("nan"),
            "mean_excess_vs_daily_universe": float((top_rets - bases).mean()) if n_dates else float("nan"),
            "beat_daily_universe_rate": float((top_rets > bases).mean()) if n_dates else float("nan"),
        })
    return pd.DataFrame(rows)

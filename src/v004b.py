from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_data_config
from .logistic_v003 import _bool_series, _markdown_table, _mean, _safe_rate


DEFAULT_SCORED_FILE = Path("reports/v004a/grid_v2_scored/v004a_scored_candidates.csv")
DEFAULT_TOP_N = 3
DEFAULT_INITIAL_TRAIN_DAYS = 8
DEFAULT_CANDIDATE_TOP_K = 10
DEFAULT_V004A_L2 = 0.30
DEFAULT_V004A_POSITIVE_WEIGHT = 1.5
DEFAULT_PAIRWISE_L2 = 0.10
DEFAULT_PAIRWISE_MODE = "target_binary"
DEFAULT_RETURN_GAP_PCT = 3.0
DEFAULT_HARD_NEGATIVE_WEIGHT = 1.0
PAIRWISE_MODES = {"target_binary", "return_gap"}

TARGET_COLUMN = "target7_d2open_d3high"
HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
CLOSE_RETURN_COLUMN = "d2open_d3close_return_pct"
REALIZED_RETURN_COLUMN = "realized_return_pct"

MODEL_ID = "pairwise_v004b1_linear_ranker"
SCOPE = "walk_forward"
V004A_MODEL_ID = "logistic_v004a_weighted"
V002_MODEL_ID = "ranking_model_v002_core_momentum_support"

BASE_FEATURE_COLUMNS = [
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_total_score",
    "rank_theme_score",
    "rank_days_since_d0",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
    "inter_close_low",
    "inter_close_trend",
    "inter_total_trend",
    "inter_total_active",
    "inter_low_active",
    "spread_close_low",
    "days_since_d0_le1",
    "days_since_d0_eq2",
    "days_since_d0_ge3",
    "v004a_score_rank_pct",
    "v004a_rank_inverse",
    "v002_score_rank_pct",
    "v002_rank_inverse",
]

NONLINEAR_FEATURE_COLUMNS = [
    "extreme_vwap",
    "extreme_price",
    "extreme_close_low",
    "extreme_close_ma10",
    "extreme_low_ma10",
    "mid_vwap_55_85",
    "mid_price_55_85",
    "mid_close_low_55_90",
]

SOURCE_FEATURE_COLUMNS = [
    "in_v004a_top10",
    "in_v002_top10",
]


def run_v004b_research(
    scored_file: str | Path = DEFAULT_SCORED_FILE,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
    pairwise_l2: float = DEFAULT_PAIRWISE_L2,
    include_v002_top10: bool = False,
    use_source_features: bool = False,
    pairwise_mode: str = DEFAULT_PAIRWISE_MODE,
    return_gap_pct: float = DEFAULT_RETURN_GAP_PCT,
    hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    _validate_config(pairwise_mode=pairwise_mode, return_gap_pct=return_gap_pct, hard_negative_weight=hard_negative_weight)
    scored_path = Path(scored_file)
    raw = pd.read_csv(scored_path, dtype={"code": str})
    candidate_union, feature_columns, data_quality = prepare_candidate_union(
        raw,
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        candidate_top_k=int(candidate_top_k),
        include_v002_top10=bool(include_v002_top10),
        use_source_features=bool(use_source_features),
    )
    dates = sorted(candidate_union["signal_date"].dropna().astype(str).unique().tolist())
    if len(dates) <= int(initial_train_days):
        raise RuntimeError(f"v004b walk-forward requires more than {initial_train_days} signal_date values")

    top3_rows, daily_top3, coefficients = build_walk_forward_predictions(
        candidate_union=candidate_union,
        feature_columns=feature_columns,
        dates=dates,
        top_n=int(top_n),
        initial_train_days=int(initial_train_days),
        pairwise_l2=float(pairwise_l2),
        pairwise_mode=str(pairwise_mode),
        return_gap_pct=float(return_gap_pct),
        hard_negative_weight=float(hard_negative_weight),
    )
    summary = build_summary(daily_top3, top3_rows)
    for frame in (summary, daily_top3, top3_rows, candidate_union, coefficients, data_quality):
        _add_run_meta(
            frame,
            top_n=int(top_n),
            initial_train_days=int(initial_train_days),
            candidate_top_k=int(candidate_top_k),
            v004a_l2=float(v004a_l2),
            v004a_positive_weight=float(v004a_positive_weight),
            pairwise_l2=float(pairwise_l2),
            include_v002_top10=bool(include_v002_top10),
            use_source_features=bool(use_source_features),
            pairwise_mode=str(pairwise_mode),
            return_gap_pct=float(return_gap_pct),
            hard_negative_weight=float(hard_negative_weight),
        )

    out_dir = Path(output_dir) if output_dir else get_data_config().reports_dir / "v004b" / _suffix_from_scored_path(scored_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "v004b_summary.csv"
    daily_csv = out_dir / "v004b_daily_top3.csv"
    top3_rows_csv = out_dir / "v004b_top3_rows.csv"
    candidate_union_csv = out_dir / "v004b_candidate_union.csv"
    coefficients_csv = out_dir / "v004b_coefficients.csv"
    data_quality_csv = out_dir / "v004b_data_quality.csv"
    report_path = out_dir / "v004b_report.md"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    daily_top3.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    top3_rows.to_csv(top3_rows_csv, index=False, encoding="utf-8-sig")
    candidate_union.to_csv(candidate_union_csv, index=False, encoding="utf-8-sig")
    coefficients.to_csv(coefficients_csv, index=False, encoding="utf-8-sig")
    data_quality.to_csv(data_quality_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            scored_path=scored_path,
            output_dir=out_dir,
            feature_columns=feature_columns,
            data_quality=data_quality,
            summary=summary,
            daily_top3=daily_top3,
            top3_rows=top3_rows,
        ),
        encoding="utf-8",
    )
    return summary, daily_top3, top3_rows, candidate_union, coefficients, report_path


def prepare_candidate_union(
    raw: pd.DataFrame,
    v004a_l2: float,
    v004a_positive_weight: float,
    candidate_top_k: int,
    include_v002_top10: bool,
    use_source_features: bool,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    required = [
        "model_id",
        "evaluation_scope",
        "l2",
        "positive_weight",
        "signal_date",
        "code",
        "model_score",
        "model_rank",
        TARGET_COLUMN,
        HIGH_RETURN_COLUMN,
        CLOSE_RETURN_COLUMN,
        REALIZED_RETURN_COLUMN,
        "eligible_for_trade",
        "graph_quality_score",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"v004b input missing required columns: {missing}")

    frame = raw.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)
    frame["evaluation_scope"] = frame["evaluation_scope"].astype(str)
    frame["l2_numeric"] = pd.to_numeric(frame["l2"], errors="coerce")
    frame["positive_weight_numeric"] = pd.to_numeric(frame["positive_weight"], errors="coerce")
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="coerce")
    frame["model_probability"] = pd.to_numeric(frame.get("model_probability"), errors="coerce") if "model_probability" in frame.columns else np.nan
    frame["model_rank"] = pd.to_numeric(frame["model_rank"], errors="coerce")
    frame["graph_quality_score"] = pd.to_numeric(frame["graph_quality_score"], errors="coerce").fillna(0.0)
    frame["eligible_for_trade"] = _bool_series(frame["eligible_for_trade"])
    frame[TARGET_COLUMN] = _bool_series(frame[TARGET_COLUMN])
    for column in (HIGH_RETURN_COLUMN, CLOSE_RETURN_COLUMN, REALIZED_RETURN_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    v004a_mask = (
        (frame["model_id"] == V004A_MODEL_ID)
        & (frame["evaluation_scope"] == SCOPE)
        & (frame["l2_numeric"].sub(float(v004a_l2)).abs() <= 1e-9)
        & (frame["positive_weight_numeric"].sub(float(v004a_positive_weight)).abs() <= 1e-9)
    )
    v004a_all = frame[v004a_mask].copy()
    if v004a_all.empty:
        raise RuntimeError(
            f"no v004a scored candidates found for l2={v004a_l2:g}, positive_weight={v004a_positive_weight:g}"
        )
    v004a_top = v004a_all[v004a_all["model_rank"] <= int(candidate_top_k)].copy()
    if v004a_top.empty:
        raise RuntimeError(f"no v004a Top{candidate_top_k} rows found")

    union_keys = v004a_top[["signal_date", "code"]].drop_duplicates().copy()
    union_keys["in_v004a_top10"] = 1
    union_keys["in_v002_top10"] = 0

    v002_all = frame[(frame["model_id"] == V002_MODEL_ID) & (frame["evaluation_scope"] == SCOPE)].copy()
    v002_top = pd.DataFrame(columns=["signal_date", "code"])
    if include_v002_top10:
        if v002_all.empty:
            raise RuntimeError("include_v002_top10 was requested, but no v002 scored candidates were found")
        v002_top = v002_all[v002_all["model_rank"] <= int(candidate_top_k)].copy()
        if v002_top.empty:
            raise RuntimeError(f"include_v002_top10 was requested, but no v002 Top{candidate_top_k} rows were found")
        v002_keys = v002_top[["signal_date", "code"]].drop_duplicates().copy()
        v002_keys["in_v004a_top10"] = 0
        v002_keys["in_v002_top10"] = 1
        union_keys = pd.concat([union_keys, v002_keys], ignore_index=True)
        union_keys = (
            union_keys.groupby(["signal_date", "code"], as_index=False)
            .agg({"in_v004a_top10": "max", "in_v002_top10": "max"})
        )

    base = v004a_all.drop_duplicates(["signal_date", "code"]).copy()
    union = union_keys.merge(base, on=["signal_date", "code"], how="left", suffixes=("", "_base"))
    if union["model_id"].isna().any():
        missing_count = int(union["model_id"].isna().sum())
        raise RuntimeError(f"candidate_union has {missing_count} rows without matching v004a scored features")

    union = union.rename(
        columns={
            "model_score": "v004a_score",
            "model_probability": "v004a_probability",
            "model_rank": "v004a_model_rank",
        }
    )
    union["source"] = np.where(
        (union["in_v004a_top10"].astype(int) == 1) & (union["in_v002_top10"].astype(int) == 1),
        "both",
        np.where(union["in_v004a_top10"].astype(int) == 1, "v004a_top10", "v002_top10"),
    )

    if not v002_all.empty:
        v002_scores = v002_all[["signal_date", "code", "model_score", "model_rank"]].rename(
            columns={"model_score": "v002_score", "model_rank": "v002_model_rank"}
        )
        union = union.merge(v002_scores, on=["signal_date", "code"], how="left")
    else:
        union["v002_score"] = np.nan
        union["v002_model_rank"] = np.nan

    union["v004a_score_rank_pct"] = _rank_pct_by_date(union, "v004a_score")
    union["v002_score_rank_pct"] = _rank_pct_by_date(union, "v002_score")
    union["v004a_rank_inverse"] = _inverse_rank(union["v004a_model_rank"])
    union["v002_rank_inverse"] = _inverse_rank(union["v002_model_rank"])
    union["in_v004a_top10"] = pd.to_numeric(union["in_v004a_top10"], errors="coerce").fillna(0).astype(int)
    union["in_v002_top10"] = pd.to_numeric(union["in_v002_top10"], errors="coerce").fillna(0).astype(int)
    union["hard_negative"] = (
        (pd.to_numeric(union["v004a_model_rank"], errors="coerce") <= 3)
        & (~union[TARGET_COLUMN].astype(bool))
    )
    add_nonlinear_features(union)

    feature_columns = [*BASE_FEATURE_COLUMNS, *NONLINEAR_FEATURE_COLUMNS]
    if use_source_features:
        feature_columns.extend(SOURCE_FEATURE_COLUMNS)
    missing_feature_columns = [column for column in feature_columns if column not in union.columns]
    for column in missing_feature_columns:
        union[column] = 0.0
    for column in feature_columns:
        union[column] = pd.to_numeric(union[column], errors="coerce").fillna(0.0).astype(float)
    if not feature_columns:
        raise RuntimeError("v004b has no usable feature columns")

    union = union.sort_values(["signal_date", "v004a_model_rank", "code"]).reset_index(drop=True)
    data_quality = pd.DataFrame(
        [
            {
                "raw_rows": int(len(raw)),
                "v004a_scored_rows": int(len(v004a_all)),
                "v004a_topk_rows": int(len(v004a_top)),
                "v002_scored_rows": int(len(v002_all)),
                "v002_topk_rows": int(len(v002_top)) if include_v002_top10 else 0,
                "candidate_union_rows": int(len(union)),
                "candidate_union_dates": int(union["signal_date"].nunique()),
                "include_v002_top10": bool(include_v002_top10),
                "use_source_features": bool(use_source_features),
                "feature_columns": ",".join(feature_columns),
                "base_feature_columns": ",".join(BASE_FEATURE_COLUMNS),
                "nonlinear_feature_columns": ",".join(NONLINEAR_FEATURE_COLUMNS),
                "source_feature_columns": ",".join(SOURCE_FEATURE_COLUMNS),
                "missing_feature_columns_filled_zero": ",".join(missing_feature_columns),
            }
        ]
    )
    return union, feature_columns, data_quality


def add_nonlinear_features(frame: pd.DataFrame) -> None:
    vwap = _feature_series(frame, "rank_d1_close_vwap_pct")
    price = _feature_series(frame, "rank_log_candidate_base_price")
    close_low = _feature_series(frame, "inter_close_low")
    close_ma10 = _feature_series(frame, "rank_d1_close_ma10_pct")
    low_ma10 = _feature_series(frame, "rank_d1_low_ma10_pct")
    frame["extreme_vwap"] = np.maximum(vwap - 0.85, 0.0)
    frame["extreme_price"] = np.maximum(price - 0.85, 0.0)
    frame["extreme_close_low"] = np.maximum(close_low - 0.90, 0.0)
    frame["extreme_close_ma10"] = np.maximum(close_ma10 - 0.90, 0.0)
    frame["extreme_low_ma10"] = np.maximum(low_ma10 - 0.90, 0.0)
    frame["mid_vwap_55_85"] = np.clip(vwap, 0.55, 0.85) - 0.55
    frame["mid_price_55_85"] = np.clip(price, 0.55, 0.85) - 0.55
    frame["mid_close_low_55_90"] = np.clip(close_low, 0.55, 0.90) - 0.55


def build_walk_forward_predictions(
    candidate_union: pd.DataFrame,
    feature_columns: list[str],
    dates: list[str],
    top_n: int,
    initial_train_days: int,
    pairwise_l2: float,
    pairwise_mode: str,
    return_gap_pct: float,
    hard_negative_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    coefficient_rows: list[pd.DataFrame] = []

    for fold_index in range(int(initial_train_days), len(dates)):
        train_dates = dates[:fold_index]
        predict_date = dates[fold_index]
        train = candidate_union[candidate_union["signal_date"].isin(train_dates)].copy()
        predict = candidate_union[candidate_union["signal_date"] == predict_date].copy()
        x_pairs, pair_weight, pair_count_by_date = build_pairwise_training_data(
            train,
            feature_columns,
            pairwise_mode=pairwise_mode,
            return_gap_pct=float(return_gap_pct),
            hard_negative_weight=float(hard_negative_weight),
        )
        beta = fit_pairwise_logistic_l2(x_pairs, sample_weight=pair_weight, l2=float(pairwise_l2))
        coefficient_rows.append(
            build_coefficients_frame(
                beta=beta,
                feature_columns=feature_columns,
                fold_index=fold_index,
                train_dates=train_dates,
                predict_date=predict_date,
                pair_count=int(len(x_pairs)),
                pair_count_by_date=pair_count_by_date,
            )
        )
        ranked = score_and_rank_predict_date(predict, feature_columns, beta)
        selected = ranked.head(int(top_n)).copy()
        top_rows.append(selected)
        daily_rows.append(build_daily_row(selected, predict, predict_date, top_n=int(top_n)))

    top3_rows = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    daily_top3 = pd.DataFrame(daily_rows)
    coefficients = pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame()
    return top3_rows, daily_top3, coefficients


def build_pairwise_training_data(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    pairwise_mode: str = DEFAULT_PAIRWISE_MODE,
    return_gap_pct: float = DEFAULT_RETURN_GAP_PCT,
    hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    _validate_config(pairwise_mode=pairwise_mode, return_gap_pct=return_gap_pct, hard_negative_weight=hard_negative_weight)
    pair_frames: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    pair_count_by_date: dict[str, int] = {}
    for signal_date, group in train.groupby("signal_date", dropna=False):
        if pairwise_mode == "target_binary":
            diffs, raw_weight = _target_binary_pairs(group, feature_columns, hard_negative_weight=float(hard_negative_weight))
        elif pairwise_mode == "return_gap":
            diffs, raw_weight = _return_gap_pairs(
                group,
                feature_columns,
                return_gap_pct=float(return_gap_pct),
                hard_negative_weight=float(hard_negative_weight),
            )
        else:
            raise RuntimeError(f"unsupported pairwise_mode={pairwise_mode}")
        pair_count = int(len(diffs))
        if pair_count <= 0:
            pair_count_by_date[str(signal_date)] = 0
            continue
        raw_weight = np.asarray(raw_weight, dtype=float)
        raw_weight = np.where(np.isfinite(raw_weight) & (raw_weight > 0), raw_weight, 0.0)
        raw_sum = float(raw_weight.sum())
        if raw_sum <= 0:
            pair_count_by_date[str(signal_date)] = 0
            continue
        pair_frames.append(diffs)
        pair_weights.append(raw_weight / raw_sum)
        pair_count_by_date[str(signal_date)] = pair_count
    if not pair_frames:
        return np.zeros((0, len(feature_columns)), dtype=float), np.zeros(0, dtype=float), pair_count_by_date
    return np.vstack(pair_frames), np.concatenate(pair_weights), pair_count_by_date


def _target_binary_pairs(group: pd.DataFrame, feature_columns: list[str], hard_negative_weight: float) -> tuple[np.ndarray, np.ndarray]:
    positives = group[group[TARGET_COLUMN].astype(bool)]
    negatives = group[~group[TARGET_COLUMN].astype(bool)]
    if positives.empty or negatives.empty:
        return np.zeros((0, len(feature_columns)), dtype=float), np.zeros(0, dtype=float)
    pos_x = positives[feature_columns].to_numpy(dtype=float)
    neg_x = negatives[feature_columns].to_numpy(dtype=float)
    diffs = (pos_x[:, None, :] - neg_x[None, :, :]).reshape(-1, len(feature_columns))
    negative_weight = np.where(negatives["hard_negative"].astype(bool).to_numpy(), float(hard_negative_weight), 1.0)
    weights = np.tile(negative_weight, len(positives))
    return diffs, weights.astype(float)


def _return_gap_pairs(
    group: pd.DataFrame,
    feature_columns: list[str],
    return_gap_pct: float,
    hard_negative_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    evaluable = group[pd.to_numeric(group[HIGH_RETURN_COLUMN], errors="coerce").notna()].copy()
    if len(evaluable) < 2:
        return np.zeros((0, len(feature_columns)), dtype=float), np.zeros(0, dtype=float)
    features = evaluable[feature_columns].to_numpy(dtype=float)
    returns = pd.to_numeric(evaluable[HIGH_RETURN_COLUMN], errors="coerce").to_numpy(dtype=float)
    hard_negative = evaluable["hard_negative"].astype(bool).to_numpy()
    diffs: list[np.ndarray] = []
    weights: list[float] = []
    for left in range(len(evaluable) - 1):
        for right in range(left + 1, len(evaluable)):
            gap = float(returns[left] - returns[right])
            if gap >= float(return_gap_pct):
                positive_idx, negative_idx = left, right
            elif -gap >= float(return_gap_pct):
                positive_idx, negative_idx = right, left
            else:
                continue
            diffs.append(features[positive_idx] - features[negative_idx])
            weight = float(hard_negative_weight) if bool(hard_negative[negative_idx]) else 1.0
            weights.append(weight)
    if not diffs:
        return np.zeros((0, len(feature_columns)), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(diffs).astype(float), np.asarray(weights, dtype=float)


def fit_pairwise_logistic_l2(
    x_diff: np.ndarray,
    sample_weight: np.ndarray,
    l2: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    x_diff = np.asarray(x_diff, dtype=float)
    if x_diff.ndim != 2:
        raise RuntimeError("x_diff must be a 2D array")
    if len(x_diff) == 0:
        return np.zeros(x_diff.shape[1], dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    weight_sum = float(weight.sum())
    if weight_sum <= 0:
        raise RuntimeError("pairwise sample_weight sum must be positive")

    beta = np.zeros(x_diff.shape[1], dtype=float)
    reg = np.full_like(beta, float(l2))
    previous_loss = _pairwise_loss(x_diff, beta, reg, weight)
    for _ in range(int(max_iter)):
        margin = x_diff @ beta
        p = _sigmoid(margin)
        gradient = (x_diff.T @ (weight * (p - 1.0))) / weight_sum + reg * beta
        variance = weight * p * (1.0 - p)
        hessian = (x_diff.T * variance) @ x_diff / weight_sum
        hessian[np.diag_indices_from(hessian)] += reg
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        scale = 1.0
        loss = previous_loss
        while scale >= 1e-4:
            candidate = beta - scale * step
            loss = _pairwise_loss(x_diff, candidate, reg, weight)
            if loss <= previous_loss:
                beta = candidate
                break
            scale *= 0.5
        else:
            break
        if abs(previous_loss - loss) < float(tol):
            break
        previous_loss = loss
    return beta


def score_and_rank_predict_date(predict: pd.DataFrame, feature_columns: list[str], beta: np.ndarray) -> pd.DataFrame:
    result = predict.copy()
    result["v004b_score"] = result[feature_columns].to_numpy(dtype=float) @ beta
    result["model_id"] = MODEL_ID
    result["evaluation_scope"] = SCOPE
    result = result.sort_values(["v004b_score", "graph_quality_score", "code"], ascending=[False, False, True]).reset_index(drop=True)
    result["daily_rank"] = np.arange(1, len(result) + 1)
    return result


def build_daily_row(selected: pd.DataFrame, candidate_pool: pd.DataFrame, signal_date: str, top_n: int) -> dict[str, Any]:
    target = selected[TARGET_COLUMN].astype(bool) if not selected.empty else pd.Series(dtype=bool)
    hit_count = int(target.sum())
    pool_target = candidate_pool[TARGET_COLUMN].astype(bool) if not candidate_pool.empty else pd.Series(dtype=bool)
    candidate_pool_hit_count = int(pool_target.sum())
    realized = pd.to_numeric(selected.get(REALIZED_RETURN_COLUMN), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    high_return = pd.to_numeric(selected.get(HIGH_RETURN_COLUMN), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    return {
        "signal_date": str(signal_date),
        "model_id": MODEL_ID,
        "evaluation_scope": SCOPE,
        "candidate_pool_size": int(len(candidate_pool)),
        "candidate_pool_hit_count": candidate_pool_hit_count,
        "candidate_pool_all3_possible": bool(candidate_pool_hit_count >= int(top_n)),
        "selected_count": int(len(selected)),
        "hit_count": hit_count,
        "all_hit": bool(len(selected) == int(top_n) and hit_count == int(top_n)),
        "top3_target_rate": _safe_rate(hit_count, int(len(selected))),
        "avg_top3_high_return": _mean(high_return),
        "avg_top3_realized_return": _mean(realized),
        "top3_codes": ",".join(selected["code"].astype(str).tolist()) if not selected.empty else "",
        "top3_scores": ",".join(f"{float(value):.6f}" for value in pd.to_numeric(selected.get("v004b_score"), errors="coerce")) if not selected.empty else "",
    }


def build_summary(daily_top3: pd.DataFrame, top3_rows: pd.DataFrame) -> pd.DataFrame:
    if daily_top3.empty:
        return pd.DataFrame()
    targets = top3_rows[TARGET_COLUMN].astype(bool) if not top3_rows.empty else pd.Series(dtype=bool)
    row: dict[str, Any] = {
        "model_id": MODEL_ID,
        "evaluation_scope": SCOPE,
        "date_count": int(daily_top3["signal_date"].nunique()),
        "selected_ticket_count": int(len(top3_rows)),
        "top3_target_rate": _safe_rate(int(targets.sum()), int(len(targets))),
        "top3_all_hit_rate": _safe_rate(int(daily_top3["all_hit"].astype(bool).sum()), int(len(daily_top3))),
        "hit_count_0_days": int((pd.to_numeric(daily_top3["hit_count"], errors="coerce").fillna(0) == 0).sum()),
        "hit_count_1_days": int((pd.to_numeric(daily_top3["hit_count"], errors="coerce").fillna(0) == 1).sum()),
        "hit_count_2_days": int((pd.to_numeric(daily_top3["hit_count"], errors="coerce").fillna(0) == 2).sum()),
        "hit_count_3_days": int((pd.to_numeric(daily_top3["hit_count"], errors="coerce").fillna(0) == 3).sum()),
        "avg_top3_realized_return": _mean(pd.to_numeric(daily_top3["avg_top3_realized_return"], errors="coerce")),
        "candidate_pool_all3_possible_days": int(daily_top3["candidate_pool_all3_possible"].astype(bool).sum())
        if "candidate_pool_all3_possible" in daily_top3.columns
        else 0,
        "candidate_pool_all3_impossible_days": int((~daily_top3["candidate_pool_all3_possible"].astype(bool)).sum())
        if "candidate_pool_all3_possible" in daily_top3.columns
        else 0,
    }
    for rank in (1, 2, 3):
        rank_rows = top3_rows[pd.to_numeric(top3_rows.get("daily_rank"), errors="coerce") == rank] if not top3_rows.empty else pd.DataFrame()
        rank_target = rank_rows[TARGET_COLUMN].astype(bool) if not rank_rows.empty else pd.Series(dtype=bool)
        row[f"rank{rank}_hit_rate"] = _safe_rate(int(rank_target.sum()), int(len(rank_target)))
    return pd.DataFrame([row])


def build_coefficients_frame(
    beta: np.ndarray,
    feature_columns: list[str],
    fold_index: int,
    train_dates: list[str],
    predict_date: str,
    pair_count: int,
    pair_count_by_date: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for feature, coefficient in zip(feature_columns, beta):
        rows.append(
            {
                "model_id": MODEL_ID,
                "evaluation_scope": SCOPE,
                "fold_index": int(fold_index),
                "train_start": train_dates[0] if train_dates else "",
                "train_end": train_dates[-1] if train_dates else "",
                "predict_date": predict_date,
                "pair_count": int(pair_count),
                "pair_date_count": int(sum(1 for count in pair_count_by_date.values() if count > 0)),
                "term": feature,
                "coefficient": float(coefficient),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    scored_path: Path,
    output_dir: Path,
    feature_columns: list[str],
    data_quality: pd.DataFrame,
    summary: pd.DataFrame,
    daily_top3: pd.DataFrame,
    top3_rows: pd.DataFrame,
) -> str:
    lines = ["# v004b pairwise ranker research", ""]
    lines.extend(
        [
            "## Scope",
            "",
            "v004b is research-only.",
            "It does not replace run-daily and is not connected to daily ranking.",
            "It trains only on past signal_date candidate_union rows in each walk-forward fold.",
            "",
            "## Inputs",
            "",
            f"- scored candidates file: `{scored_path}`",
            f"- output dir: `{output_dir}`",
            f"- feature columns: `{', '.join(feature_columns)}`",
            "",
            "## Configuration",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            data_quality,
            [
                "pairwise_mode",
                "return_gap_pct",
                "hard_negative_weight",
                "use_source_features",
                "candidate_top_k",
                "include_v002_top10",
                "v004a_l2",
                "v004a_positive_weight",
                "pairwise_l2",
                "initial_train_days",
                "top_n",
                "feature_columns",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Data Quality",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            data_quality,
            [
                "raw_rows",
                "v004a_scored_rows",
                "v004a_topk_rows",
                "v002_scored_rows",
                "v002_topk_rows",
                "candidate_union_rows",
                "candidate_union_dates",
                "include_v002_top10",
            ],
        )
    )
    lines.extend(["", "## Summary", ""])
    lines.extend(
        _markdown_table(
            summary,
            [
                "model_id",
                "evaluation_scope",
                "top3_target_rate",
                "top3_all_hit_rate",
                "hit_count_0_days",
                "hit_count_1_days",
                "hit_count_2_days",
                "hit_count_3_days",
                "rank1_hit_rate",
                "rank2_hit_rate",
                "rank3_hit_rate",
                "avg_top3_realized_return",
                "candidate_pool_all3_possible_days",
                "candidate_pool_all3_impossible_days",
            ],
        )
    )
    lines.extend(["", "## Daily Top3", ""])
    lines.extend(
        _markdown_table(
            daily_top3,
            [
                "signal_date",
                "candidate_pool_size",
                "candidate_pool_hit_count",
                "candidate_pool_all3_possible",
                "selected_count",
                "hit_count",
                "all_hit",
                "avg_top3_realized_return",
                "top3_codes",
                "top3_scores",
            ],
        )
    )
    lines.extend(["", "## Top3 Rows Preview", ""])
    lines.extend(
        _markdown_table(
            top3_rows.head(80),
            ["signal_date", "daily_rank", "code", "v004b_score", TARGET_COLUMN, HIGH_RETURN_COLUMN, REALIZED_RETURN_COLUMN, "source"],
        )
    )
    return "\n".join(lines)


def _add_run_meta(
    frame: pd.DataFrame,
    top_n: int,
    initial_train_days: int,
    candidate_top_k: int,
    v004a_l2: float,
    v004a_positive_weight: float,
    pairwise_l2: float,
    include_v002_top10: bool,
    use_source_features: bool,
    pairwise_mode: str,
    return_gap_pct: float,
    hard_negative_weight: float,
) -> None:
    if frame.empty:
        return
    frame["top_n"] = int(top_n)
    frame["initial_train_days"] = int(initial_train_days)
    frame["candidate_top_k"] = int(candidate_top_k)
    frame["v004a_l2"] = float(v004a_l2)
    frame["v004a_positive_weight"] = float(v004a_positive_weight)
    frame["pairwise_l2"] = float(pairwise_l2)
    frame["include_v002_top10"] = bool(include_v002_top10)
    frame["use_source_features"] = bool(use_source_features)
    frame["pairwise_mode"] = str(pairwise_mode)
    frame["return_gap_pct"] = float(return_gap_pct)
    frame["hard_negative_weight"] = float(hard_negative_weight)


def _validate_config(pairwise_mode: str, return_gap_pct: float, hard_negative_weight: float) -> None:
    if str(pairwise_mode) not in PAIRWISE_MODES:
        raise RuntimeError(f"unsupported pairwise_mode={pairwise_mode}; supported={sorted(PAIRWISE_MODES)}")
    if not math.isfinite(float(return_gap_pct)) or float(return_gap_pct) < 0:
        raise RuntimeError(f"return_gap_pct must be finite and non-negative: {return_gap_pct}")
    if not math.isfinite(float(hard_negative_weight)) or float(hard_negative_weight) <= 0:
        raise RuntimeError(f"hard_negative_weight must be finite and positive: {hard_negative_weight}")


def _feature_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _rank_pct_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return (
        frame.groupby("signal_date", dropna=False)[column]
        .transform(lambda values: pd.to_numeric(values, errors="coerce").rank(pct=True, method="average"))
        .astype(float)
        .fillna(0.0)
    )


def _inverse_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (1.0 / values.where(values > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _pairwise_loss(x_diff: np.ndarray, beta: np.ndarray, reg: np.ndarray, sample_weight: np.ndarray) -> float:
    margin = x_diff @ beta
    weight_sum = float(sample_weight.sum())
    loss = float(np.sum(sample_weight * np.logaddexp(0.0, -margin)) / weight_sum)
    penalty = 0.5 * float(np.sum(reg * beta * beta))
    return loss + penalty


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _suffix_from_scored_path(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if parent and parent not in {"", "."}:
        return parent
    return stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v004b research-only pairwise walk-forward ranker.")
    parser.add_argument("--scored-file", default=str(DEFAULT_SCORED_FILE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    parser.add_argument("--pairwise-l2", type=float, default=DEFAULT_PAIRWISE_L2)
    parser.add_argument("--include-v002-top10", action="store_true")
    parser.add_argument("--use-source-features", action="store_true")
    parser.add_argument("--pairwise-mode", choices=sorted(PAIRWISE_MODES), default=DEFAULT_PAIRWISE_MODE)
    parser.add_argument("--return-gap-pct", type=float, default=DEFAULT_RETURN_GAP_PCT)
    parser.add_argument("--hard-negative-weight", type=float, default=DEFAULT_HARD_NEGATIVE_WEIGHT)
    args = parser.parse_args(argv)
    summary, daily_top3, top3_rows, candidate_union, coefficients, report_path = run_v004b_research(
        scored_file=args.scored_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        initial_train_days=args.initial_train_days,
        candidate_top_k=args.candidate_top_k,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
        pairwise_l2=args.pairwise_l2,
        include_v002_top10=args.include_v002_top10,
        use_source_features=args.use_source_features,
        pairwise_mode=args.pairwise_mode,
        return_gap_pct=args.return_gap_pct,
        hard_negative_weight=args.hard_negative_weight,
    )
    print(f"summary rows: {len(summary)}")
    print(f"daily rows: {len(daily_top3)}")
    print(f"top3 rows: {len(top3_rows)}")
    print(f"candidate union rows: {len(candidate_union)}")
    print(f"coefficients rows: {len(coefficients)}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

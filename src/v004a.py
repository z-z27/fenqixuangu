from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_data_config
from .logistic_v003 import (
    BASELINE_MODELS,
    DEFAULT_SAMPLES_FILE,
    HAND_SCORE_WEIGHTS,
    _auc,
    _bool_series,
    _logloss,
    _markdown_table,
    _mean,
    _safe_rate,
)
from .ranking_backtest import score_candidates


DEFAULT_TARGET_COLUMN = "target7_d2open_d3high"
DEFAULT_HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
DEFAULT_CLOSE_RETURN_COLUMN = "d2open_d3close_return_pct"
DEFAULT_TARGET_RETURN_PCT = 7.0
DEFAULT_TOP_N = 3
DEFAULT_INITIAL_TRAIN_DAYS = 18
DEFAULT_L2_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_POSITIVE_WEIGHT_GRID = (1.0, 1.5, 2.0, 3.0)
DEFAULT_THRESHOLD_GRID = (0.40, 0.45, 0.50, 0.55)

MODEL_ID_V003_BASELINE = "logistic_v003_unweighted_baseline"
MODEL_ID_V004A = "logistic_v004a_weighted"
SCOPE_WALK_FORWARD = "walk_forward"

BASE_RANK_SPECS = [
    ("d1_close_ma10_pct", "rank_d1_close_ma10_pct"),
    ("d1_low_ma10_pct", "rank_d1_low_ma10_pct"),
    ("trend_hold_score", "rank_trend_hold_score"),
    ("total_score", "rank_total_score"),
    ("theme_score", "rank_theme_score"),
    ("days_since_d0", "rank_days_since_d0"),
    ("log_candidate_base_price", "rank_log_candidate_base_price"),
]

OPTIONAL_RANK_SPECS = [
    ("active_money_score", "rank_active_money_score"),
    ("d1_close_vwap_pct", "rank_d1_close_vwap_pct"),
]

V003_FEATURE_COLUMNS = [feature for _, feature in BASE_RANK_SPECS]

DAY_BUCKET_COLUMNS = [
    "days_since_d0_le1",
    "days_since_d0_eq2",
    "days_since_d0_ge3",
]

BASE_INTERACTION_SPECS = [
    ("inter_close_low", ("rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct"), "product"),
    ("inter_close_trend", ("rank_d1_close_ma10_pct", "rank_trend_hold_score"), "product"),
    ("inter_total_trend", ("rank_total_score", "rank_trend_hold_score"), "product"),
    ("inter_total_active", ("rank_total_score", "rank_active_money_score"), "product"),
    ("inter_low_active", ("rank_d1_low_ma10_pct", "rank_active_money_score"), "product"),
    ("spread_close_low", ("rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct"), "spread"),
]

META_COLUMNS = [
    "model_id",
    "evaluation_scope",
    "l2",
    "positive_weight",
    "selection_policy",
    "threshold",
    "feature_set",
    "interaction_set",
]

SCORE_META_COLUMNS = [
    "model_id",
    "evaluation_scope",
    "l2",
    "positive_weight",
    "feature_set",
    "interaction_set",
]


def run_v004a_research(
    samples_file: str | Path = DEFAULT_SAMPLES_FILE,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    l2_grid: str | list[float] | tuple[float, ...] | None = None,
    positive_weight_grid: str | list[float] | tuple[float, ...] | None = None,
    threshold_grid: str | list[float] | tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    _validate_target_return_pct(target_return_pct)
    samples_path = Path(samples_file)
    raw = pd.read_csv(samples_path, dtype={"code": str})
    samples, feature_info, data_quality = prepare_v004a_samples(raw, target_return_pct=float(target_return_pct))
    dates = sorted(samples["signal_date"].dropna().astype(str).unique().tolist())
    if len(dates) <= int(initial_train_days):
        raise RuntimeError(f"v004a walk-forward requires more than {initial_train_days} signal_date values")

    l2_values = parse_float_grid(l2_grid, DEFAULT_L2_GRID, "l2-grid", positive=True)
    positive_values = parse_float_grid(positive_weight_grid, DEFAULT_POSITIVE_WEIGHT_GRID, "positive-weight-grid", positive=True)
    threshold_values = parse_float_grid(threshold_grid, DEFAULT_THRESHOLD_GRID, "threshold-grid", positive=False)
    out_dir = Path(output_dir) if output_dir else get_data_config().reports_dir / "v004a" / _suffix_from_samples_path(samples_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored, coefficients = build_walk_forward_scores(
        samples=samples,
        dates=dates,
        initial_train_days=int(initial_train_days),
        l2_values=l2_values,
        positive_weight_values=positive_values,
        feature_info=feature_info,
    )
    per_date = build_per_date_topk_outcomes(
        scored=scored,
        top_n=int(top_n),
        thresholds=threshold_values,
        target_return_pct=float(target_return_pct),
    )
    top3_combo = build_top3_combo_summary(
        per_date,
        top_n=int(top_n),
        target_return_pct=float(target_return_pct),
    )
    rankwise = build_rankwise_summary(per_date)
    comparison = build_grid_comparison(scored=scored, top3_combo=top3_combo, rankwise=rankwise)
    scored = add_scored_model_rank(scored)
    scored_output = build_scored_candidates_output(scored, feature_info)

    data_quality = data_quality.copy()
    data_quality["l2_grid"] = ",".join(_format_grid_value(value) for value in l2_values)
    data_quality["positive_weight_grid"] = ",".join(_format_grid_value(value) for value in positive_values)
    data_quality["threshold_grid"] = ",".join(_format_grid_value(value) for value in threshold_values)
    data_quality["initial_train_days"] = int(initial_train_days)
    data_quality["top_n"] = int(top_n)
    _ensure_meta_columns(data_quality, fill={
        "model_id": MODEL_ID_V004A,
        "evaluation_scope": SCOPE_WALK_FORWARD,
        "l2": "grid",
        "positive_weight": "grid",
        "selection_policy": "grid",
        "threshold": "grid",
        "feature_set": feature_info["feature_set"],
        "interaction_set": feature_info["interaction_set"],
    })

    comparison_csv = out_dir / "v004a_grid_comparison.csv"
    rankwise_csv = out_dir / "v004a_rankwise_summary.csv"
    top3_combo_csv = out_dir / "v004a_top3_combo_summary.csv"
    per_date_csv = out_dir / "v004a_per_date_topk_outcomes.csv"
    coefficients_csv = out_dir / "v004a_coefficients.csv"
    data_quality_csv = out_dir / "v004a_data_quality.csv"
    scored_csv = out_dir / "v004a_scored_candidates.csv"
    report_path = out_dir / "v004a_report.md"

    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    rankwise.to_csv(rankwise_csv, index=False, encoding="utf-8-sig")
    top3_combo.to_csv(top3_combo_csv, index=False, encoding="utf-8-sig")
    per_date.to_csv(per_date_csv, index=False, encoding="utf-8-sig")
    coefficients.to_csv(coefficients_csv, index=False, encoding="utf-8-sig")
    data_quality.to_csv(data_quality_csv, index=False, encoding="utf-8-sig")
    scored_output.to_csv(scored_csv, index=False, encoding="utf-8-sig")
    print(f"scored candidates csv: {scored_csv}")
    report_path.write_text(
        build_v004a_report(
            samples_path=samples_path,
            output_dir=out_dir,
            samples=samples,
            feature_info=feature_info,
            top_n=int(top_n),
            initial_train_days=int(initial_train_days),
            target_return_pct=float(target_return_pct),
            l2_values=l2_values,
            positive_weight_values=positive_values,
            threshold_values=threshold_values,
            comparison=comparison,
            rankwise=rankwise,
            top3_combo=top3_combo,
            data_quality=data_quality,
        ),
        encoding="utf-8",
    )
    return comparison, rankwise, top3_combo, per_date, coefficients, data_quality, scored, report_path


def prepare_v004a_samples(raw: pd.DataFrame, target_return_pct: float = DEFAULT_TARGET_RETURN_PCT) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    _validate_target_return_pct(target_return_pct)
    required = [
        "signal_date",
        "code",
        "eligible_for_trade",
        DEFAULT_TARGET_COLUMN,
        DEFAULT_HIGH_RETURN_COLUMN,
        DEFAULT_CLOSE_RETURN_COLUMN,
        "candidate_base_price",
        "d1_close_ma10_pct",
        "d1_low_ma10_pct",
        "trend_hold_score",
        "total_score",
        "theme_score",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"v004a input missing required columns: {missing}")

    frame = raw.copy()
    raw_rows = int(len(frame))
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = frame["signal_date"].astype(str)
    if "graph_quality_score" not in frame.columns:
        frame["graph_quality_score"] = 0.0
    frame["graph_quality_score"] = pd.to_numeric(frame["graph_quality_score"], errors="coerce").fillna(0.0)
    frame["eligible_for_trade"] = _bool_series(frame["eligible_for_trade"])
    frame[DEFAULT_TARGET_COLUMN] = _bool_series(frame[DEFAULT_TARGET_COLUMN])

    days_since_missing_column = "days_since_d0" not in frame.columns
    if days_since_missing_column:
        frame["days_since_d0"] = np.nan

    present_optional_specs = [(raw_column, rank_column) for raw_column, rank_column in OPTIONAL_RANK_SPECS if raw_column in frame.columns]
    missing_optional_columns = [raw_column for raw_column, _ in OPTIONAL_RANK_SPECS if raw_column not in frame.columns]

    numeric_columns = [
        DEFAULT_HIGH_RETURN_COLUMN,
        DEFAULT_CLOSE_RETURN_COLUMN,
        "candidate_base_price",
        *[raw_column for raw_column, _ in BASE_RANK_SPECS if raw_column != "log_candidate_base_price"],
        *[raw_column for raw_column, _ in present_optional_specs],
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    eligible_mask = frame["eligible_for_trade"].fillna(False).astype(bool)
    high_notna = frame[DEFAULT_HIGH_RETURN_COLUMN].notna()
    close_notna = frame[DEFAULT_CLOSE_RETURN_COLUMN].notna()
    base_price_positive = frame["candidate_base_price"].notna() & (frame["candidate_base_price"] > 0)
    filtered = frame[eligible_mask & high_notna & close_notna & base_price_positive].copy()
    if filtered.empty:
        raise RuntimeError("no v004a rows remain after eligible/return/base-price filters")

    filtered["log_candidate_base_price"] = np.log(filtered["candidate_base_price"].astype(float))
    rank_specs = [*BASE_RANK_SPECS, *present_optional_specs]
    for raw_column, rank_column in rank_specs:
        filtered[rank_column] = (
            filtered.groupby("signal_date", dropna=False)[raw_column]
            .transform(lambda values: pd.to_numeric(values, errors="coerce").rank(pct=True, method="average"))
            .astype(float)
            .fillna(0.5)
        )

    days_values = pd.to_numeric(filtered["days_since_d0"], errors="coerce")
    filtered["days_since_d0_le1"] = (days_values <= 1).fillna(False).astype(int)
    filtered["days_since_d0_eq2"] = (days_values == 2).fillna(False).astype(int)
    filtered["days_since_d0_ge3"] = (days_values >= 3).fillna(False).astype(int)

    interaction_columns: list[str] = []
    for interaction_name, inputs, kind in BASE_INTERACTION_SPECS:
        left, right = inputs
        if left not in filtered.columns or right not in filtered.columns:
            continue
        if kind == "product":
            filtered[interaction_name] = filtered[left].astype(float) * filtered[right].astype(float)
        elif kind == "spread":
            filtered[interaction_name] = filtered[left].astype(float) - filtered[right].astype(float)
        else:
            raise RuntimeError(f"unsupported v004a interaction kind: {kind}")
        interaction_columns.append(interaction_name)

    v004a_feature_columns = [
        *V003_FEATURE_COLUMNS,
        *[rank_column for _, rank_column in present_optional_specs],
        *interaction_columns,
        *DAY_BUCKET_COLUMNS,
    ]
    for column in v004a_feature_columns:
        if column not in filtered.columns:
            filtered[column] = 0.0
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce").fillna(0.0).astype(float)

    filtered["realized_return_pct"] = np.where(
        filtered[DEFAULT_TARGET_COLUMN].astype(bool),
        float(target_return_pct),
        pd.to_numeric(filtered[DEFAULT_CLOSE_RETURN_COLUMN], errors="coerce"),
    )
    filtered["tail_weight"] = np.select(
        [
            pd.to_numeric(filtered[DEFAULT_HIGH_RETURN_COLUMN], errors="coerce") >= 12.0,
            pd.to_numeric(filtered[DEFAULT_HIGH_RETURN_COLUMN], errors="coerce") >= 10.0,
        ],
        [2.0, 1.5],
        default=1.0,
    ).astype(float)

    feature_set = ",".join(v004a_feature_columns)
    interaction_set = ",".join(interaction_columns) if interaction_columns else "none"
    feature_info = {
        "v003_feature_columns": V003_FEATURE_COLUMNS,
        "v004a_feature_columns": v004a_feature_columns,
        "optional_rank_columns": [rank_column for _, rank_column in present_optional_specs],
        "missing_optional_columns": missing_optional_columns,
        "interaction_columns": interaction_columns,
        "feature_set": feature_set,
        "interaction_set": interaction_set,
    }
    data_quality = pd.DataFrame(
        [
            {
                "raw_rows": raw_rows,
                "eligible_rows_before_return_filter": int(eligible_mask.sum()),
                "high_return_notna_rows": int((eligible_mask & high_notna).sum()),
                "close_return_notna_rows": int((eligible_mask & close_notna).sum()),
                "base_price_positive_rows": int((eligible_mask & high_notna & close_notna & base_price_positive).sum()),
                "final_rows": int(len(filtered)),
                "close_return_missing_count": int((eligible_mask & high_notna & ~close_notna).sum()),
                "target_return_pct": float(target_return_pct),
                "target_column": DEFAULT_TARGET_COLUMN,
                "high_return_column": DEFAULT_HIGH_RETURN_COLUMN,
                "close_return_column": DEFAULT_CLOSE_RETURN_COLUMN,
                "optional_rank_columns_present": ",".join([raw_column for raw_column, _ in present_optional_specs]),
                "optional_rank_columns_missing": ",".join(missing_optional_columns),
                "days_since_d0_missing_column": bool(days_since_missing_column),
                "days_since_d0_missing_count": int(pd.to_numeric(filtered["days_since_d0"], errors="coerce").isna().sum()),
                "v003_feature_columns": ",".join(V003_FEATURE_COLUMNS),
                "v004a_feature_columns": feature_set,
                "interaction_columns": interaction_set,
            }
        ]
    )
    return filtered.sort_values(["signal_date", "code"]).reset_index(drop=True), feature_info, data_quality


def build_walk_forward_scores(
    samples: pd.DataFrame,
    dates: list[str],
    initial_train_days: int,
    l2_values: list[float],
    positive_weight_values: list[float],
    feature_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    manual_models = _load_manual_models()

    for fold_index in range(int(initial_train_days), len(dates)):
        train_dates = dates[:fold_index]
        predict_date = dates[fold_index]
        train = samples[samples["signal_date"].isin(train_dates)].copy()
        predict = samples[samples["signal_date"] == predict_date].copy()
        if train.empty or predict.empty:
            continue

        scored_frames.extend(_score_manual_and_hand_models(predict, manual_models))

        for l2 in l2_values:
            beta_v003 = fit_logistic_l2_weighted(
                train[V003_FEATURE_COLUMNS].to_numpy(dtype=float),
                train[DEFAULT_TARGET_COLUMN].astype(bool).astype(int).to_numpy(dtype=float),
                l2=float(l2),
                sample_weight=None,
            )
            scored_frames.append(
                _score_logistic_frame(
                    predict,
                    beta=beta_v003,
                    feature_columns=V003_FEATURE_COLUMNS,
                    model_id=MODEL_ID_V003_BASELINE,
                    l2=float(l2),
                    positive_weight=1.0,
                    feature_set=",".join(V003_FEATURE_COLUMNS),
                    interaction_set="none",
                )
            )
            coefficient_frames.append(
                _build_coefficients_frame(
                    beta=beta_v003,
                    feature_columns=V003_FEATURE_COLUMNS,
                    model_id=MODEL_ID_V003_BASELINE,
                    l2=float(l2),
                    positive_weight=1.0,
                    feature_set=",".join(V003_FEATURE_COLUMNS),
                    interaction_set="none",
                    fold_index=fold_index,
                    train_dates=train_dates,
                    predict_date=predict_date,
                )
            )

            for positive_weight in positive_weight_values:
                sample_weight = build_training_sample_weight(train, positive_weight=float(positive_weight))
                beta_v004a = fit_logistic_l2_weighted(
                    train[feature_info["v004a_feature_columns"]].to_numpy(dtype=float),
                    train[DEFAULT_TARGET_COLUMN].astype(bool).astype(int).to_numpy(dtype=float),
                    l2=float(l2),
                    sample_weight=sample_weight,
                )
                scored_frames.append(
                    _score_logistic_frame(
                        predict,
                        beta=beta_v004a,
                        feature_columns=feature_info["v004a_feature_columns"],
                        model_id=MODEL_ID_V004A,
                        l2=float(l2),
                        positive_weight=float(positive_weight),
                        feature_set=feature_info["feature_set"],
                        interaction_set=feature_info["interaction_set"],
                    )
                )
                coefficient_frames.append(
                    _build_coefficients_frame(
                        beta=beta_v004a,
                        feature_columns=feature_info["v004a_feature_columns"],
                        model_id=MODEL_ID_V004A,
                        l2=float(l2),
                        positive_weight=float(positive_weight),
                        feature_set=feature_info["feature_set"],
                        interaction_set=feature_info["interaction_set"],
                        fold_index=fold_index,
                        train_dates=train_dates,
                        predict_date=predict_date,
                    )
                )

    if not scored_frames:
        raise RuntimeError("v004a walk-forward produced no scored rows")
    scored = pd.concat(scored_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    return scored, coefficients


def build_training_sample_weight(train: pd.DataFrame, positive_weight: float) -> np.ndarray:
    counts = train.groupby("signal_date", dropna=False)["signal_date"].transform("count").astype(float)
    date_weight = 1.0 / counts.replace(0, np.nan)
    target = train[DEFAULT_TARGET_COLUMN].astype(bool)
    class_weight = np.where(target, float(positive_weight), 1.0)
    tail_weight = pd.to_numeric(train["tail_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    weight = date_weight.to_numpy(dtype=float) * class_weight.astype(float) * tail_weight
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    if float(weight.sum()) <= 0:
        raise RuntimeError("sample_weight sum is zero")
    return weight.astype(float)


def fit_logistic_l2_weighted(
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if sample_weight is None:
        weight = np.ones(len(y), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
    if len(x) != len(y) or len(y) != len(weight):
        raise RuntimeError("x, y and sample_weight lengths must match")
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    weight_sum = float(weight.sum())
    if weight_sum <= 0:
        raise RuntimeError("sample_weight sum must be positive")

    x_aug = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x_aug.shape[1], dtype=float)
    weighted_positive_rate = float(np.sum(weight * y) / weight_sum)
    if weighted_positive_rate <= 1e-6 or weighted_positive_rate >= 1.0 - 1e-6 or len(np.unique(y)) < 2:
        p = min(max(weighted_positive_rate, 1e-6), 1.0 - 1e-6)
        beta[0] = math.log(p / (1.0 - p))
        return beta

    reg = np.zeros_like(beta)
    reg[1:] = float(l2)
    previous_loss = _weighted_logistic_loss(x_aug, y, beta, reg, weight)
    for _ in range(int(max_iter)):
        p = _sigmoid(x_aug @ beta)
        gradient = (x_aug.T @ (weight * (p - y))) / weight_sum + reg * beta
        weighted_variance = weight * p * (1.0 - p)
        hessian = (x_aug.T * weighted_variance) @ x_aug / weight_sum
        hessian[np.diag_indices_from(hessian)] += reg
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        scale = 1.0
        loss = previous_loss
        while scale >= 1e-4:
            candidate = beta - scale * step
            loss = _weighted_logistic_loss(x_aug, y, candidate, reg, weight)
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


def build_per_date_topk_outcomes(
    scored: pd.DataFrame,
    top_n: int,
    thresholds: list[float],
    target_return_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in scored.groupby(SCORE_META_COLUMNS, dropna=False):
        meta = dict(zip(SCORE_META_COLUMNS, group_key))
        for signal_date, date_group in group.groupby("signal_date", dropna=False):
            ranked = _rank_scored_candidates(date_group)
            force_top = ranked.head(int(top_n)).copy()
            rows.extend(
                _selection_rows(
                    selected=force_top,
                    meta=meta,
                    signal_date=str(signal_date),
                    top_n=int(top_n),
                    selection_policy="force_top3",
                    threshold=np.nan,
                    target_return_pct=float(target_return_pct),
                )
            )
            if ranked["model_probability"].notna().any():
                for threshold in thresholds:
                    allowed = ranked[pd.to_numeric(ranked["model_probability"], errors="coerce") >= float(threshold)].head(int(top_n)).copy()
                    rows.extend(
                        _selection_rows(
                            selected=allowed,
                            meta=meta,
                            signal_date=str(signal_date),
                            top_n=int(top_n),
                            selection_policy="allow_skip",
                            threshold=float(threshold),
                            target_return_pct=float(target_return_pct),
                        )
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[*META_COLUMNS, "signal_date"])
    return result.sort_values(
        ["model_id", "l2", "positive_weight", "selection_policy", "threshold", "signal_date", "rank"],
        na_position="last",
    ).reset_index(drop=True)


def build_top3_combo_summary(per_date: pd.DataFrame, top_n: int, target_return_pct: float) -> pd.DataFrame:
    if per_date.empty:
        return pd.DataFrame()
    daily = _daily_rows_from_per_date(per_date)
    rows: list[dict[str, Any]] = []
    for group_key, group in daily.groupby(META_COLUMNS, dropna=False):
        meta = dict(zip(META_COLUMNS, group_key))
        selected_ticket_count = int(pd.to_numeric(group["selected_count"], errors="coerce").fillna(0).sum())
        hit_count_total = int(pd.to_numeric(group["hit_count"], errors="coerce").fillna(0).sum())
        date_count = int(len(group))
        selected_count_series = pd.to_numeric(group["selected_count"], errors="coerce").fillna(0)
        hit_count_series = pd.to_numeric(group["hit_count"], errors="coerce").fillna(0)
        no_trade_days = int((selected_count_series == 0).sum())
        underfilled_days = int((selected_count_series < int(top_n)).sum())
        avg_selected = _mean(pd.to_numeric(group["selected_count"], errors="coerce"))
        row = {
            **meta,
            "date_count": date_count,
            "selected_ticket_count": selected_ticket_count,
            "top3_target_rate": _safe_rate(hit_count_total, selected_ticket_count),
            "top3_all_hit_rate": _safe_rate(int(group["all_hit"].astype(bool).sum()), date_count),
            "hit_count_0_days": int((hit_count_series == 0).sum()),
            "hit_count_1_days": int((hit_count_series == 1).sum()),
            "hit_count_2_days": int((hit_count_series == 2).sum()),
            "hit_count_3_days": int((hit_count_series == 3).sum()),
            "avg_top3_high_return": _mean(pd.to_numeric(group["avg_high_return"], errors="coerce")),
            "avg_top3_realized_return": _mean(pd.to_numeric(group["avg_realized_return"], errors="coerce")),
            "portfolio_realized_positive_rate": _safe_rate(int(group["portfolio_realized_positive"].astype(bool).sum()), date_count),
            "portfolio_realized_hit7_rate": _safe_rate(int(group["portfolio_realized_hit7"].astype(bool).sum()), date_count),
            "selected_count_per_day": avg_selected,
            "avg_selected_count_per_day": avg_selected,
            "skip_day_rate": _safe_rate(underfilled_days, date_count),
            "no_trade_days": no_trade_days,
            "no_trade_day_rate": _safe_rate(no_trade_days, date_count),
            "underfilled_days": underfilled_days,
            "underfilled_day_rate": _safe_rate(underfilled_days, date_count),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(META_COLUMNS).reset_index(drop=True)


def build_rankwise_summary(per_date: pd.DataFrame) -> pd.DataFrame:
    if per_date.empty:
        return pd.DataFrame()
    selected = per_date[pd.to_numeric(per_date["rank"], errors="coerce").notna()].copy()
    rows: list[dict[str, Any]] = []
    for group_key, group in per_date.groupby(META_COLUMNS, dropna=False):
        meta = dict(zip(META_COLUMNS, group_key))
        selected_group = selected
        for column, value in meta.items():
            if pd.isna(value):
                selected_group = selected_group[selected_group[column].isna()]
            else:
                selected_group = selected_group[selected_group[column] == value]
        row: dict[str, Any] = {**meta}
        for rank in (1, 2, 3):
            rank_rows = selected_group[pd.to_numeric(selected_group["rank"], errors="coerce") == rank]
            target = rank_rows[DEFAULT_TARGET_COLUMN].astype(bool) if not rank_rows.empty else pd.Series(dtype=bool)
            row[f"rank{rank}_count"] = int(len(rank_rows))
            row[f"rank{rank}_hit_rate"] = _safe_rate(int(target.sum()), int(len(target)))
            row[f"rank{rank}_avg_return"] = _mean(pd.to_numeric(rank_rows[DEFAULT_HIGH_RETURN_COLUMN], errors="coerce"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(META_COLUMNS).reset_index(drop=True)


def build_grid_comparison(scored: pd.DataFrame, top3_combo: pd.DataFrame, rankwise: pd.DataFrame) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []
    for group_key, group in scored.groupby(SCORE_META_COLUMNS, dropna=False):
        meta = dict(zip(SCORE_META_COLUMNS, group_key))
        target = group[DEFAULT_TARGET_COLUMN].astype(bool)
        score = pd.to_numeric(group["model_score"], errors="coerce")
        probability = pd.to_numeric(group["model_probability"], errors="coerce")
        candidate_rows.append(
            {
                **meta,
                "candidate_date_count": int(group["signal_date"].nunique()),
                "candidate_count": int(len(group)),
                "candidate_target_count": int(target.sum()),
                "candidate_target_rate": _safe_rate(int(target.sum()), int(len(group))),
                "auc": _auc(target.astype(int), score),
                "logloss": _logloss(target.astype(int), probability) if probability.notna().any() else None,
            }
        )
    candidate_summary = pd.DataFrame(candidate_rows)
    if top3_combo.empty:
        return pd.DataFrame()
    merged = top3_combo.merge(
        rankwise,
        on=META_COLUMNS,
        how="left",
        suffixes=("", "_rankwise"),
    )
    merged = merged.merge(
        candidate_summary,
        on=SCORE_META_COLUMNS,
        how="left",
    )
    preferred = [
        *META_COLUMNS,
        "candidate_date_count",
        "candidate_count",
        "candidate_target_rate",
        "auc",
        "logloss",
        "top3_target_rate",
        "top3_all_hit_rate",
        "avg_top3_high_return",
        "avg_top3_realized_return",
        "portfolio_realized_positive_rate",
        "portfolio_realized_hit7_rate",
        "hit_count_0_days",
        "hit_count_1_days",
        "hit_count_2_days",
        "hit_count_3_days",
        "rank1_hit_rate",
        "rank2_hit_rate",
        "rank3_hit_rate",
        "rank1_avg_return",
        "rank2_avg_return",
        "rank3_avg_return",
        "selected_count_per_day",
        "avg_selected_count_per_day",
        "skip_day_rate",
        "no_trade_days",
        "no_trade_day_rate",
        "underfilled_days",
        "underfilled_day_rate",
    ]
    remaining = [column for column in merged.columns if column not in preferred]
    return merged[[column for column in preferred if column in merged.columns] + remaining].sort_values(META_COLUMNS).reset_index(drop=True)


def add_scored_model_rank(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        result = scored.copy()
        result["model_rank"] = pd.Series(dtype="Int64")
        return result
    result = scored.copy()
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["signal_date"] = result["signal_date"].astype(str)
    result["model_score"] = pd.to_numeric(result["model_score"], errors="coerce")
    if "graph_quality_score" not in result.columns:
        result["graph_quality_score"] = 0.0
    result["graph_quality_score"] = pd.to_numeric(result["graph_quality_score"], errors="coerce").fillna(0.0)
    sort_columns = [*SCORE_META_COLUMNS, "signal_date", "model_score", "graph_quality_score", "code"]
    ascending = [True] * (len(SCORE_META_COLUMNS) + 1) + [False, False, True]
    result = result.sort_values(sort_columns, ascending=ascending, na_position="last").copy()
    result["model_rank"] = result.groupby([*SCORE_META_COLUMNS, "signal_date"], dropna=False).cumcount() + 1
    return result.reset_index(drop=True)


def build_scored_candidates_output(scored: pd.DataFrame, feature_info: dict[str, Any]) -> pd.DataFrame:
    priority_columns = [
        "model_id",
        "evaluation_scope",
        "l2",
        "positive_weight",
        "feature_set",
        "interaction_set",
        "signal_date",
        "code",
        "model_score",
        "model_probability",
        "model_rank",
        DEFAULT_TARGET_COLUMN,
        DEFAULT_HIGH_RETURN_COLUMN,
        DEFAULT_CLOSE_RETURN_COLUMN,
        "realized_return_pct",
        "eligible_for_trade",
        "graph_quality_score",
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_total_score",
        "rank_theme_score",
        "rank_days_since_d0",
        "rank_log_candidate_base_price",
    ]
    for column in ("rank_active_money_score", "rank_d1_close_vwap_pct"):
        if column in scored.columns:
            priority_columns.append(column)
    for column in feature_info.get("interaction_columns", []) or []:
        if column in scored.columns and column not in priority_columns:
            priority_columns.append(column)
    for column in DAY_BUCKET_COLUMNS:
        if column in scored.columns and column not in priority_columns:
            priority_columns.append(column)
    output_columns = [column for column in priority_columns if column in scored.columns]
    output_columns.extend(column for column in scored.columns if column not in output_columns)
    return scored[output_columns].copy()


def build_v004a_report(
    samples_path: Path,
    output_dir: Path,
    samples: pd.DataFrame,
    feature_info: dict[str, Any],
    top_n: int,
    initial_train_days: int,
    target_return_pct: float,
    l2_values: list[float],
    positive_weight_values: list[float],
    threshold_values: list[float],
    comparison: pd.DataFrame,
    rankwise: pd.DataFrame,
    top3_combo: pd.DataFrame,
    data_quality: pd.DataFrame,
) -> str:
    lines = ["# v004a weighted logistic research", ""]
    lines.extend(
        [
            "## Scope",
            "",
            "v004a is research-only.",
            "It does not replace run-daily.",
            "It is not connected to daily ranking.",
            "It does not remove v001 or v002.",
            "Random split is not allowed; this runner uses walk-forward validation only.",
            "The primary decision metrics are walk_forward Top3, realized return, and rank-wise metrics, not train AUC.",
            "Full scored candidate ranks are written to v004a_scored_candidates.csv.",
            "",
            "## Inputs",
            "",
            f"- samples file: `{samples_path}`",
            f"- output dir: `{output_dir}`",
            f"- rows after filters: **{len(samples)}**",
            f"- signal dates: **{samples['signal_date'].nunique()}**",
            f"- top_n: **{int(top_n)}**",
            f"- initial_train_days: **{int(initial_train_days)}**",
            f"- target column: **{DEFAULT_TARGET_COLUMN}**",
            f"- high return column: **{DEFAULT_HIGH_RETURN_COLUMN}**",
            f"- close return column: **{DEFAULT_CLOSE_RETURN_COLUMN}**",
            f"- target_return_pct: **{float(target_return_pct):.4f}**",
            f"- l2 grid: **{', '.join(_format_grid_value(value) for value in l2_values)}**",
            f"- positive weight grid: **{', '.join(_format_grid_value(value) for value in positive_weight_values)}**",
            f"- threshold grid: **{', '.join(_format_grid_value(value) for value in threshold_values)}**",
            "",
            "## Feature Set",
            "",
            f"- v003 baseline features: `{', '.join(V003_FEATURE_COLUMNS)}`",
            f"- v004a features: `{feature_info['feature_set']}`",
            f"- interactions: `{feature_info['interaction_set']}`",
            f"- missing optional raw columns: `{', '.join(feature_info['missing_optional_columns']) or 'none'}`",
            "",
            "## Data Quality",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            data_quality,
            [
                *META_COLUMNS,
                "raw_rows",
                "eligible_rows_before_return_filter",
                "high_return_notna_rows",
                "close_return_notna_rows",
                "base_price_positive_rows",
                "final_rows",
                "close_return_missing_count",
                "days_since_d0_missing_column",
                "days_since_d0_missing_count",
            ],
        )
    )
    lines.extend(["", "## Best Walk-Forward Rows", ""])
    best = comparison.sort_values(
        ["avg_top3_realized_return", "top3_target_rate", "rank1_hit_rate"],
        ascending=[False, False, False],
        na_position="last",
    ).head(30)
    lines.extend(
        _markdown_table(
            best,
            [
                *META_COLUMNS,
                "top3_target_rate",
                "top3_all_hit_rate",
                "avg_top3_high_return",
                "avg_top3_realized_return",
                "portfolio_realized_positive_rate",
                "portfolio_realized_hit7_rate",
                "rank1_hit_rate",
                "rank2_hit_rate",
                "rank3_hit_rate",
                "avg_selected_count_per_day",
                "skip_day_rate",
                "no_trade_days",
                "no_trade_day_rate",
                "underfilled_days",
                "underfilled_day_rate",
            ],
        )
    )
    lines.extend(["", "## Rank Wise Summary", ""])
    lines.extend(
        _markdown_table(
            rankwise.sort_values(["model_id", "l2", "positive_weight", "selection_policy", "threshold"]).head(80),
            [
                *META_COLUMNS,
                "rank1_hit_rate",
                "rank2_hit_rate",
                "rank3_hit_rate",
                "rank1_avg_return",
                "rank2_avg_return",
                "rank3_avg_return",
            ],
        )
    )
    lines.extend(["", "## Top3 Combination Summary", ""])
    lines.extend(
        _markdown_table(
            top3_combo.sort_values(["model_id", "l2", "positive_weight", "selection_policy", "threshold"]).head(80),
            [
                *META_COLUMNS,
                "top3_target_rate",
                "top3_all_hit_rate",
                "avg_top3_high_return",
                "avg_top3_realized_return",
                "portfolio_realized_positive_rate",
                "portfolio_realized_hit7_rate",
                "avg_selected_count_per_day",
                "skip_day_rate",
                "no_trade_days",
                "no_trade_day_rate",
                "underfilled_days",
                "underfilled_day_rate",
            ],
        )
    )
    return "\n".join(lines)


def _load_manual_models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for model_id, path in BASELINE_MODELS.items():
        if path.exists():
            models[model_id] = json.loads(path.read_text(encoding="utf-8"))
    return models


def _score_manual_and_hand_models(predict: pd.DataFrame, manual_models: dict[str, dict[str, Any]]) -> list[pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    for model_id, model in manual_models.items():
        scored = score_candidates(predict.copy(), model)
        scored_frames.append(
            _score_frame(
                scored,
                model_id=model_id,
                score_column=str(model.get("score_column", "research_score")),
                l2=np.nan,
                positive_weight=np.nan,
                feature_set="manual_json_features",
                interaction_set="none",
            )
        )
    hand = predict.copy()
    hand["score_v003_hand"] = sum(float(weight) * hand[column].astype(float) for column, weight in HAND_SCORE_WEIGHTS.items())
    scored_frames.append(
        _score_frame(
            hand,
            model_id="score_v003_hand",
            score_column="score_v003_hand",
            l2=np.nan,
            positive_weight=np.nan,
            feature_set=",".join(HAND_SCORE_WEIGHTS.keys()),
            interaction_set="none",
        )
    )
    return scored_frames


def _score_logistic_frame(
    frame: pd.DataFrame,
    beta: np.ndarray,
    feature_columns: list[str],
    model_id: str,
    l2: float,
    positive_weight: float,
    feature_set: str,
    interaction_set: str,
) -> pd.DataFrame:
    result = frame.copy()
    probability = predict_logistic(beta, result[feature_columns].to_numpy(dtype=float))
    result["model_probability"] = probability
    result["model_score"] = probability
    result["model_id"] = model_id
    result["evaluation_scope"] = SCOPE_WALK_FORWARD
    result["l2"] = float(l2)
    result["positive_weight"] = float(positive_weight)
    result["feature_set"] = feature_set
    result["interaction_set"] = interaction_set
    return result


def _score_frame(
    frame: pd.DataFrame,
    model_id: str,
    score_column: str,
    l2: float,
    positive_weight: float,
    feature_set: str,
    interaction_set: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["model_id"] = model_id
    result["evaluation_scope"] = SCOPE_WALK_FORWARD
    result["l2"] = l2
    result["positive_weight"] = positive_weight
    result["model_score"] = pd.to_numeric(result[score_column], errors="coerce")
    result["model_probability"] = np.nan
    result["feature_set"] = feature_set
    result["interaction_set"] = interaction_set
    return result


def _rank_scored_candidates(group: pd.DataFrame) -> pd.DataFrame:
    ranked = group.copy()
    ranked["model_score"] = pd.to_numeric(ranked["model_score"], errors="coerce").fillna(float("-inf"))
    ranked["graph_quality_score"] = pd.to_numeric(ranked.get("graph_quality_score", 0.0), errors="coerce").fillna(0.0)
    ranked = ranked.sort_values(["model_score", "graph_quality_score", "code"], ascending=[False, False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _selection_rows(
    selected: pd.DataFrame,
    meta: dict[str, Any],
    signal_date: str,
    top_n: int,
    selection_policy: str,
    threshold: float,
    target_return_pct: float,
) -> list[dict[str, Any]]:
    selected_count = int(len(selected))
    targets = selected[DEFAULT_TARGET_COLUMN].astype(bool) if selected_count else pd.Series(dtype=bool)
    hit_count = int(targets.sum()) if selected_count else 0
    all_hit = bool(selected_count == int(top_n) and hit_count == int(top_n))
    high_return = pd.to_numeric(selected.get(DEFAULT_HIGH_RETURN_COLUMN), errors="coerce") if selected_count else pd.Series(dtype=float)
    realized = pd.to_numeric(selected.get("realized_return_pct"), errors="coerce") if selected_count else pd.Series(dtype=float)
    avg_high_return = _mean(high_return)
    avg_realized_return = _mean(realized)
    base = {
        **meta,
        "selection_policy": selection_policy,
        "threshold": threshold,
        "signal_date": signal_date,
        "selected_count": selected_count,
        "hit_count": hit_count,
        "all_hit": all_hit,
        "avg_high_return": avg_high_return,
        "avg_realized_return": avg_realized_return,
        "portfolio_realized_positive": bool(avg_realized_return is not None and avg_realized_return > 0.0),
        "portfolio_realized_hit7": bool(avg_realized_return is not None and avg_realized_return >= float(target_return_pct)),
    }
    if selected_count <= 0:
        return [
            {
                **base,
                "rank": np.nan,
                "code": "",
                "model_score": np.nan,
                "model_probability": np.nan,
                DEFAULT_TARGET_COLUMN: False,
                DEFAULT_HIGH_RETURN_COLUMN: np.nan,
                DEFAULT_CLOSE_RETURN_COLUMN: np.nan,
                "realized_return_pct": np.nan,
            }
        ]
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        rows.append(
            {
                **base,
                "rank": int(row.get("rank")),
                "code": str(row.get("code")),
                "model_score": float(row.get("model_score")),
                "model_probability": _float_or_nan(row.get("model_probability")),
                DEFAULT_TARGET_COLUMN: bool(row.get(DEFAULT_TARGET_COLUMN)),
                DEFAULT_HIGH_RETURN_COLUMN: _float_or_nan(row.get(DEFAULT_HIGH_RETURN_COLUMN)),
                DEFAULT_CLOSE_RETURN_COLUMN: _float_or_nan(row.get(DEFAULT_CLOSE_RETURN_COLUMN)),
                "realized_return_pct": _float_or_nan(row.get("realized_return_pct")),
            }
        )
    return rows


def _daily_rows_from_per_date(per_date: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *META_COLUMNS,
        "signal_date",
        "selected_count",
        "hit_count",
        "all_hit",
        "avg_high_return",
        "avg_realized_return",
        "portfolio_realized_positive",
        "portfolio_realized_hit7",
    ]
    return per_date[columns].drop_duplicates(subset=[*META_COLUMNS, "signal_date"]).reset_index(drop=True)


def _build_coefficients_frame(
    beta: np.ndarray,
    feature_columns: list[str],
    model_id: str,
    l2: float,
    positive_weight: float,
    feature_set: str,
    interaction_set: str,
    fold_index: int,
    train_dates: list[str],
    predict_date: str,
) -> pd.DataFrame:
    rows = [
        {
            "model_id": model_id,
            "evaluation_scope": SCOPE_WALK_FORWARD,
            "l2": float(l2),
            "positive_weight": float(positive_weight),
            "selection_policy": "training",
            "threshold": np.nan,
            "feature_set": feature_set,
            "interaction_set": interaction_set,
            "fold_index": int(fold_index),
            "train_start": train_dates[0] if train_dates else "",
            "train_end": train_dates[-1] if train_dates else "",
            "predict_date": predict_date,
            "term": "intercept",
            "coefficient": float(beta[0]),
        }
    ]
    for feature, coefficient in zip(feature_columns, beta[1:]):
        rows.append(
            {
                "model_id": model_id,
                "evaluation_scope": SCOPE_WALK_FORWARD,
                "l2": float(l2),
                "positive_weight": float(positive_weight),
                "selection_policy": "training",
                "threshold": np.nan,
                "feature_set": feature_set,
                "interaction_set": interaction_set,
                "fold_index": int(fold_index),
                "train_start": train_dates[0] if train_dates else "",
                "train_end": train_dates[-1] if train_dates else "",
                "predict_date": predict_date,
                "term": feature,
                "coefficient": float(coefficient),
            }
        )
    return pd.DataFrame(rows)


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    return _sigmoid(x_aug @ beta)


def _weighted_logistic_loss(x_aug: np.ndarray, y: np.ndarray, beta: np.ndarray, reg: np.ndarray, sample_weight: np.ndarray) -> float:
    weight_sum = float(sample_weight.sum())
    z = x_aug @ beta
    loss = float(np.sum(sample_weight * (np.logaddexp(0.0, z) - y * z)) / weight_sum)
    penalty = 0.5 * float(np.sum(reg * beta * beta))
    return loss + penalty


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _ensure_meta_columns(frame: pd.DataFrame, fill: dict[str, Any]) -> None:
    for column in META_COLUMNS:
        if column not in frame.columns:
            frame[column] = fill.get(column, np.nan)


def parse_float_grid(raw: str | list[float] | tuple[float, ...] | None, default: tuple[float, ...], name: str, positive: bool) -> list[float]:
    if raw is None:
        values = [float(value) for value in default]
    elif isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    if not values:
        raise RuntimeError(f"{name} is empty")
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        if positive and value <= 0:
            raise RuntimeError(f"{name} values must be positive: {value}")
        if not math.isfinite(value):
            raise RuntimeError(f"{name} values must be finite: {value}")
        if name == "threshold-grid" and not (0.0 <= value <= 1.0):
            raise RuntimeError(f"{name} values must be between 0.0 and 1.0: {value}")
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _validate_target_return_pct(target_return_pct: float) -> None:
    if abs(float(target_return_pct) - DEFAULT_TARGET_RETURN_PCT) > 1e-9:
        raise RuntimeError("v004a currently supports only target7_d2open_d3high with target_return_pct=7.0")


def _suffix_from_samples_path(path: Path) -> str:
    stem = path.stem
    prefix = "history_candidates_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _format_grid_value(value: float) -> str:
    return f"{float(value):g}"


def _float_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v004a research-only weighted logistic walk-forward validation.")
    parser.add_argument("--samples-file", default=str(DEFAULT_SAMPLES_FILE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    parser.add_argument("--target-return-pct", type=float, default=DEFAULT_TARGET_RETURN_PCT)
    parser.add_argument("--l2-grid", default=",".join(_format_grid_value(value) for value in DEFAULT_L2_GRID))
    parser.add_argument("--positive-weight-grid", default=",".join(_format_grid_value(value) for value in DEFAULT_POSITIVE_WEIGHT_GRID))
    parser.add_argument("--threshold-grid", default=",".join(_format_grid_value(value) for value in DEFAULT_THRESHOLD_GRID))
    args = parser.parse_args(argv)
    comparison, rankwise, top3_combo, per_date, coefficients, data_quality, scored, report_path = run_v004a_research(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        initial_train_days=args.initial_train_days,
        target_return_pct=args.target_return_pct,
        l2_grid=args.l2_grid,
        positive_weight_grid=args.positive_weight_grid,
        threshold_grid=args.threshold_grid,
    )
    print(f"comparison rows: {len(comparison)}")
    print(f"rankwise rows: {len(rankwise)}")
    print(f"top3 combo rows: {len(top3_combo)}")
    print(f"per-date rows: {len(per_date)}")
    print(f"coefficients rows: {len(coefficients)}")
    print(f"data quality rows: {len(data_quality)}")
    print(f"scored candidate rows: {len(scored)}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

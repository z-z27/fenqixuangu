from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_data_config
from .ranking_backtest import score_candidates


DEFAULT_SAMPLES_FILE = Path(
    "reports/history_samples/2026-05-06_2026-06-29/history_candidates_2026-05-06_2026-06-29_dedup.csv"
)
DEFAULT_TARGET_COLUMN = "target7_d2open_d3high"
DEFAULT_HIGH_RETURN_COLUMN = "d2open_d3high_return_pct"
DEFAULT_CLOSE_RETURN_COLUMN = "d2open_d3close_return_pct"
DEFAULT_TOP_N = 3
DEFAULT_INITIAL_TRAIN_DAYS = 18
DEFAULT_L2 = 1.0
DEFAULT_TARGET_RETURN_PCT = 7.0

RAW_FEATURE_COLUMNS = [
    "d1_close_ma10_pct",
    "d1_low_ma10_pct",
    "trend_hold_score",
    "total_score",
    "theme_score",
    "days_since_d0",
    "log_candidate_base_price",
]

MODEL_FEATURE_COLUMNS = [
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_total_score",
    "rank_theme_score",
    "rank_days_since_d0",
    "rank_log_candidate_base_price",
]

HAND_SCORE_WEIGHTS = {
    "rank_d1_close_ma10_pct": 0.27,
    "rank_d1_low_ma10_pct": 0.25,
    "rank_trend_hold_score": 0.23,
    "rank_total_score": 0.18,
    "rank_theme_score": 0.05,
    "rank_days_since_d0": -0.04,
    "rank_log_candidate_base_price": 0.04,
}

BASELINE_MODELS = {
    "ranking_model_v001_core_momentum": Path("reports/manual_models/ranking_model_v001_core_momentum.json"),
    "ranking_model_v002_core_momentum_support": Path("reports/manual_models/ranking_model_v002_core_momentum_support.json"),
}


def run_logistic_v003_research(
    samples_file: str | Path = DEFAULT_SAMPLES_FILE,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    l2: float = DEFAULT_L2,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    samples_path = Path(samples_file)
    raw = pd.read_csv(samples_path, dtype={"code": str})
    samples, data_quality = prepare_samples(raw, target_return_pct=float(target_return_pct))
    data_quality["l2"] = float(l2)
    dates = sorted(samples["signal_date"].dropna().astype(str).unique().tolist())
    if len(dates) < 3:
        raise RuntimeError("logistic_v003 requires at least 3 signal_date values")
    split_index = max(1, min(len(dates) - 1, int(math.floor(len(dates) * 0.70))))
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]

    out_dir = Path(output_dir) if output_dir else get_data_config().reports_dir / "logistic_v003" / _suffix_from_samples_path(samples_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_frame = samples[samples["signal_date"].isin(train_dates)].copy()
    test_frame = samples[samples["signal_date"].isin(test_dates)].copy()
    beta = fit_logistic_l2(
        train_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
        train_frame[DEFAULT_TARGET_COLUMN].astype(bool).astype(int).to_numpy(dtype=float),
        l2=float(l2),
    )
    coefficients = build_coefficients(beta, l2=float(l2), train_dates=train_dates, test_dates=test_dates)

    scored_train_test = []
    for scope, frame in (("date_split_train", train_frame), ("date_split_test", test_frame)):
        scored_train_test.extend(score_all_models(frame, beta=beta, scope=scope))
    scored_train_test_frame = pd.concat(scored_train_test, ignore_index=True) if scored_train_test else pd.DataFrame()

    walk_forward_scored = build_walk_forward_predictions(
        samples=samples,
        dates=dates,
        initial_train_days=int(initial_train_days),
        l2=float(l2),
    )
    all_scored = pd.concat([scored_train_test_frame, walk_forward_scored], ignore_index=True)
    daily_top3, top3_rows = build_daily_topn_outputs(all_scored, top_n=int(top_n), target_return_pct=float(target_return_pct))
    train_test_summary = build_model_summary(scored_train_test_frame, daily_top3, top3_rows)
    walk_forward_summary = build_model_summary(walk_forward_scored, daily_top3, top3_rows)
    rankwise = build_rankwise_summary(top3_rows)
    top3_combo = build_top3_combo_summary(daily_top3, top3_rows)
    comparison = pd.concat([train_test_summary, walk_forward_summary], ignore_index=True)
    for frame in (train_test_summary, walk_forward_summary, rankwise, top3_combo, comparison):
        if not frame.empty:
            frame["l2"] = float(l2)
            frame["target_return_pct"] = float(target_return_pct)

    coefficients_csv = out_dir / "logistic_v003_coefficients.csv"
    train_test_csv = out_dir / "logistic_v003_train_test_summary.csv"
    walk_forward_csv = out_dir / "logistic_v003_walk_forward_summary.csv"
    daily_top3_csv = out_dir / "logistic_v003_daily_top3.csv"
    top3_rows_csv = out_dir / "logistic_v003_top3_rows.csv"
    rankwise_csv = out_dir / "logistic_v003_rankwise_summary.csv"
    top3_combo_csv = out_dir / "logistic_v003_top3_combo_summary.csv"
    comparison_csv = out_dir / "logistic_v003_comparison_summary.csv"
    data_quality_csv = out_dir / "logistic_v003_data_quality.csv"
    model_json = out_dir / "logistic_v003_model.json"
    markdown_path = out_dir / "logistic_v003_comparison_report.md"

    coefficients.to_csv(coefficients_csv, index=False, encoding="utf-8-sig")
    train_test_summary.to_csv(train_test_csv, index=False, encoding="utf-8-sig")
    walk_forward_summary.to_csv(walk_forward_csv, index=False, encoding="utf-8-sig")
    daily_top3.to_csv(daily_top3_csv, index=False, encoding="utf-8-sig")
    top3_rows.to_csv(top3_rows_csv, index=False, encoding="utf-8-sig")
    rankwise.to_csv(rankwise_csv, index=False, encoding="utf-8-sig")
    top3_combo.to_csv(top3_combo_csv, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    data_quality.to_csv(data_quality_csv, index=False, encoding="utf-8-sig")
    model_json.write_text(
        json.dumps(
            {
                "model_id": "logistic_v003_d2open_d3high",
                "model_type": "l2_logistic_regression",
                "target_column": DEFAULT_TARGET_COLUMN,
                "high_return_column": DEFAULT_HIGH_RETURN_COLUMN,
                "close_return_column": DEFAULT_CLOSE_RETURN_COLUMN,
                "target_return_pct": float(target_return_pct),
                "feature_columns": MODEL_FEATURE_COLUMNS,
                "raw_feature_columns": RAW_FEATURE_COLUMNS,
                "l2": float(l2),
                "research_only": True,
                "daily_ranking_compatible": False,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "intercept": float(beta[0]),
                "coefficients": {feature: float(value) for feature, value in zip(MODEL_FEATURE_COLUMNS, beta[1:])},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_markdown_report(
            samples_path=samples_path,
            output_dir=out_dir,
            samples=samples,
            train_dates=train_dates,
            test_dates=test_dates,
            initial_train_days=int(initial_train_days),
            l2=float(l2),
            target_return_pct=float(target_return_pct),
            coefficients=coefficients,
            comparison=comparison,
            rankwise=rankwise,
            top3_combo=top3_combo,
            daily_top3=daily_top3,
            data_quality=data_quality,
        ),
        encoding="utf-8",
    )
    return coefficients, comparison, rankwise, top3_combo, daily_top3, data_quality, markdown_path


def run_logistic_v003_l2_grid(
    samples_file: str | Path = DEFAULT_SAMPLES_FILE,
    output_dir: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    l2_grid: str | list[float] | tuple[float, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    samples_path = Path(samples_file)
    values = parse_l2_grid(l2_grid)
    if not values:
        raise RuntimeError("l2-grid is empty")
    root_dir = Path(output_dir) if output_dir else get_data_config().reports_dir / "logistic_v003" / f"{_suffix_from_samples_path(samples_path)}_l2_grid"
    root_dir.mkdir(parents=True, exist_ok=True)

    comparison_frames: list[pd.DataFrame] = []
    rankwise_frames: list[pd.DataFrame] = []
    top3_combo_frames: list[pd.DataFrame] = []
    for value in values:
        subdir = root_dir / _l2_dir_name(float(value))
        _, comparison, rankwise, top3_combo, _, _, _ = run_logistic_v003_research(
            samples_file=samples_path,
            output_dir=subdir,
            top_n=top_n,
            initial_train_days=initial_train_days,
            l2=float(value),
            target_return_pct=target_return_pct,
        )
        comparison_frames.append(comparison.assign(l2=float(value), l2_dir=subdir.name))
        rankwise_frames.append(rankwise.assign(l2=float(value), l2_dir=subdir.name))
        top3_combo_frames.append(top3_combo.assign(l2=float(value), l2_dir=subdir.name))

    comparison_all = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    rankwise_all = pd.concat(rankwise_frames, ignore_index=True) if rankwise_frames else pd.DataFrame()
    top3_combo_all = pd.concat(top3_combo_frames, ignore_index=True) if top3_combo_frames else pd.DataFrame()
    comparison_csv = root_dir / "logistic_v003_l2_grid_comparison.csv"
    rankwise_csv = root_dir / "logistic_v003_l2_grid_rankwise.csv"
    top3_combo_csv = root_dir / "logistic_v003_l2_grid_top3_combo.csv"
    comparison_all.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    rankwise_all.to_csv(rankwise_csv, index=False, encoding="utf-8-sig")
    top3_combo_all.to_csv(top3_combo_csv, index=False, encoding="utf-8-sig")
    return comparison_all, rankwise_all, top3_combo_all, comparison_csv, rankwise_csv, top3_combo_csv


def prepare_samples(raw: pd.DataFrame, target_return_pct: float = DEFAULT_TARGET_RETURN_PCT) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "signal_date",
        "code",
        "eligible_for_trade",
        DEFAULT_TARGET_COLUMN,
        DEFAULT_HIGH_RETURN_COLUMN,
        DEFAULT_CLOSE_RETURN_COLUMN,
        "candidate_base_price",
        *[column for column in RAW_FEATURE_COLUMNS if column != "log_candidate_base_price"],
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"logistic_v003 input missing required columns: {missing}")

    frame = raw.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["eligible_for_trade"] = _bool_series(frame["eligible_for_trade"])
    frame[DEFAULT_TARGET_COLUMN] = _bool_series(frame[DEFAULT_TARGET_COLUMN])
    numeric_columns = [
        DEFAULT_HIGH_RETURN_COLUMN,
        DEFAULT_CLOSE_RETURN_COLUMN,
        "candidate_base_price",
        *[column for column in RAW_FEATURE_COLUMNS if column != "log_candidate_base_price"],
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    eligible_mask = frame["eligible_for_trade"].fillna(False).astype(bool)
    high_notna = frame[DEFAULT_HIGH_RETURN_COLUMN].notna()
    close_notna = frame[DEFAULT_CLOSE_RETURN_COLUMN].notna()
    frame = frame[eligible_mask & high_notna & close_notna].copy()
    frame = frame[frame["candidate_base_price"].notna() & (frame["candidate_base_price"] > 0)].copy()
    data_quality = pd.DataFrame(
        [
            {
                "raw_rows": int(len(raw)),
                "eligible_rows_before_return_filter": int(eligible_mask.sum()),
                "high_return_notna_rows": int((eligible_mask & high_notna).sum()),
                "close_return_notna_rows": int((eligible_mask & close_notna).sum()),
                "final_rows": int(len(frame)),
                "close_return_missing_count": int((eligible_mask & high_notna & ~close_notna).sum()),
                "target_return_pct": float(target_return_pct),
            }
        ]
    )
    if frame.empty:
        raise RuntimeError("no logistic_v003 rows remain after eligible/return/base-price filters")

    frame["log_candidate_base_price"] = np.log(frame["candidate_base_price"].astype(float))
    for raw_column, feature_column in zip(RAW_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS):
        frame[feature_column] = (
            frame.groupby("signal_date", dropna=False)[raw_column]
            .transform(lambda values: pd.to_numeric(values, errors="coerce").rank(pct=True, method="average"))
            .astype(float)
            .fillna(0.5)
        )
    frame["realized_return_pct"] = np.where(
        frame[DEFAULT_TARGET_COLUMN].astype(bool),
        float(target_return_pct),
        pd.to_numeric(frame[DEFAULT_CLOSE_RETURN_COLUMN], errors="coerce"),
    )
    return frame.sort_values(["signal_date", "code"]).reset_index(drop=True), data_quality


def score_all_models(frame: pd.DataFrame, beta: np.ndarray | None, scope: str) -> list[pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    for model_id, model_path in BASELINE_MODELS.items():
        if model_path.exists():
            model = json.loads(model_path.read_text(encoding="utf-8"))
            scored = score_candidates(frame.copy(), model)
            scored_frames.append(_score_frame(scored, model_id=model_id, scope=scope, score_column="research_score"))
    hand = frame.copy()
    hand["score_v003_hand"] = sum(float(weight) * hand[column].astype(float) for column, weight in HAND_SCORE_WEIGHTS.items())
    scored_frames.append(_score_frame(hand, model_id="score_v003_hand", scope=scope, score_column="score_v003_hand"))
    if beta is not None:
        logistic = frame.copy()
        logistic["logistic_v003_probability"] = predict_logistic(
            beta,
            logistic[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
        )
        scored_frames.append(
            _score_frame(
                logistic,
                model_id="logistic_v003",
                scope=scope,
                score_column="logistic_v003_probability",
                probability_column="logistic_v003_probability",
            )
        )
    return scored_frames


def build_walk_forward_predictions(
    samples: pd.DataFrame,
    dates: list[str],
    initial_train_days: int,
    l2: float,
) -> pd.DataFrame:
    if len(dates) <= int(initial_train_days):
        raise RuntimeError(f"walk-forward requires more than {initial_train_days} signal_date values")
    scored: list[pd.DataFrame] = []
    for index in range(int(initial_train_days), len(dates)):
        train_dates = dates[:index]
        predict_date = dates[index]
        train = samples[samples["signal_date"].isin(train_dates)].copy()
        predict = samples[samples["signal_date"] == predict_date].copy()
        if train.empty or predict.empty:
            continue
        beta = fit_logistic_l2(
            train[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
            train[DEFAULT_TARGET_COLUMN].astype(bool).astype(int).to_numpy(dtype=float),
            l2=float(l2),
        )
        scored.extend(score_all_models(predict, beta=beta, scope="walk_forward"))
    if not scored:
        return pd.DataFrame()
    return pd.concat(scored, ignore_index=True)


def fit_logistic_l2(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = DEFAULT_L2,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x_aug.shape[1], dtype=float)
    if len(np.unique(y)) < 2:
        p = min(max(float(y.mean()), 1e-6), 1.0 - 1e-6)
        beta[0] = math.log(p / (1.0 - p))
        return beta

    reg = np.zeros_like(beta)
    reg[1:] = float(l2)
    previous_loss = _logistic_loss(x_aug, y, beta, reg)
    for _ in range(int(max_iter)):
        p = _sigmoid(x_aug @ beta)
        gradient = (x_aug.T @ (p - y)) / len(y) + reg * beta
        weights = p * (1.0 - p)
        hessian = (x_aug.T * weights) @ x_aug / len(y)
        hessian[np.diag_indices_from(hessian)] += reg
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        scale = 1.0
        while scale >= 1e-4:
            candidate = beta - scale * step
            loss = _logistic_loss(x_aug, y, candidate, reg)
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


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    return _sigmoid(x_aug @ beta)


def build_daily_topn_outputs(scored: pd.DataFrame, top_n: int, target_return_pct: float = DEFAULT_TARGET_RETURN_PCT) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    for (scope, model_id, signal_date), group in scored.groupby(["evaluation_scope", "model_id", "signal_date"], dropna=False):
        ranked = group.sort_values(["model_score", "graph_quality_score", "code"], ascending=[False, False, True]).copy()
        ranked["daily_rank"] = np.arange(1, len(ranked) + 1)
        top = ranked.head(int(top_n)).copy()
        rows.append(top)
        hit_count = int(top[DEFAULT_TARGET_COLUMN].astype(bool).sum())
        realized = pd.to_numeric(top["realized_return_pct"], errors="coerce")
        high_return = pd.to_numeric(top[DEFAULT_HIGH_RETURN_COLUMN], errors="coerce")
        daily_rows.append(
            {
                "evaluation_scope": scope,
                "model_id": model_id,
                "signal_date": signal_date,
                "top_n": int(top_n),
                "topn_count": int(len(top)),
                "topn_codes": ",".join(top["code"].astype(str).tolist()),
                "topn_scores": ",".join(f"{float(value):.6f}" for value in pd.to_numeric(top["model_score"], errors="coerce")),
                "hit_count": hit_count,
                "any_hit": bool(hit_count > 0),
                "all_hit": bool(len(top) == int(top_n) and hit_count == int(top_n)),
                "top3_target_rate": _safe_rate(hit_count, int(len(top))),
                "avg_top3_high_return": _mean(high_return),
                "avg_top3_realized_return": _mean(realized),
                "portfolio_realized_positive": bool(_mean(realized) is not None and (_mean(realized) or 0.0) > 0.0),
                "portfolio_realized_hit7": bool(_mean(realized) is not None and (_mean(realized) or 0.0) >= float(target_return_pct)),
            }
        )
    top3_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.DataFrame(daily_rows), top3_rows.reset_index(drop=True)


def build_model_summary(scored: pd.DataFrame, daily_top3: pd.DataFrame, top3_rows: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    combo = build_top3_combo_summary(daily_top3, top3_rows)
    rows: list[dict[str, Any]] = []
    for (scope, model_id), group in scored.groupby(["evaluation_scope", "model_id"], dropna=False):
        target = group[DEFAULT_TARGET_COLUMN].astype(bool)
        score = pd.to_numeric(group["model_score"], errors="coerce")
        probability = pd.to_numeric(group.get("model_probability"), errors="coerce") if "model_probability" in group.columns else pd.Series(dtype=float)
        row = {
            "evaluation_scope": scope,
            "model_id": model_id,
            "date_count": int(group["signal_date"].nunique()),
            "candidate_count": int(len(group)),
            "target_count": int(target.sum()),
            "target_rate": _safe_rate(int(target.sum()), int(len(target))),
            "score_mean": _mean(score),
            "auc": _auc(target.astype(int), score),
            "logloss": _logloss(target.astype(int), probability) if model_id == "logistic_v003" else None,
        }
        match = combo[(combo["evaluation_scope"] == scope) & (combo["model_id"] == model_id)]
        if not match.empty:
            row.update(match.iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation_scope", "model_id"]).reset_index(drop=True)


def build_rankwise_summary(top3_rows: pd.DataFrame) -> pd.DataFrame:
    if top3_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (scope, model_id), group in top3_rows.groupby(["evaluation_scope", "model_id"], dropna=False):
        row: dict[str, Any] = {"evaluation_scope": scope, "model_id": model_id}
        for rank in (1, 2, 3):
            rank_rows = group[group["daily_rank"] == rank]
            target = rank_rows[DEFAULT_TARGET_COLUMN].astype(bool) if not rank_rows.empty else pd.Series(dtype=bool)
            row[f"rank{rank}_count"] = int(len(rank_rows))
            row[f"rank{rank}_hit_rate"] = _safe_rate(int(target.sum()), int(len(target)))
            row[f"rank{rank}_avg_return"] = _mean(pd.to_numeric(rank_rows.get(DEFAULT_HIGH_RETURN_COLUMN), errors="coerce"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation_scope", "model_id"]).reset_index(drop=True)


def build_top3_combo_summary(daily_top3: pd.DataFrame, top3_rows: pd.DataFrame) -> pd.DataFrame:
    if daily_top3.empty or top3_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (scope, model_id), daily_group in daily_top3.groupby(["evaluation_scope", "model_id"], dropna=False):
        row_group = top3_rows[(top3_rows["evaluation_scope"] == scope) & (top3_rows["model_id"] == model_id)]
        targets = row_group[DEFAULT_TARGET_COLUMN].astype(bool)
        row = {
            "evaluation_scope": scope,
            "model_id": model_id,
            "date_count": int(daily_group["signal_date"].nunique()),
            "selected_ticket_count": int(len(row_group)),
            "top3_target_rate": _safe_rate(int(targets.sum()), int(len(targets))),
            "top3_all_hit_rate": _safe_rate(int(daily_group["all_hit"].astype(bool).sum()), int(len(daily_group))),
            "hit_count_0_days": int((daily_group["hit_count"] == 0).sum()),
            "hit_count_1_days": int((daily_group["hit_count"] == 1).sum()),
            "hit_count_2_days": int((daily_group["hit_count"] == 2).sum()),
            "hit_count_3_days": int((daily_group["hit_count"] == 3).sum()),
            "avg_top3_high_return": _mean(pd.to_numeric(daily_group["avg_top3_high_return"], errors="coerce")),
            "avg_top3_realized_return": _mean(pd.to_numeric(daily_group["avg_top3_realized_return"], errors="coerce")),
            "portfolio_realized_positive_rate": _safe_rate(
                int(daily_group["portfolio_realized_positive"].astype(bool).sum()),
                int(len(daily_group)),
            ),
            "portfolio_realized_hit7_rate": _safe_rate(
                int(daily_group["portfolio_realized_hit7"].astype(bool).sum()),
                int(len(daily_group)),
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation_scope", "model_id"]).reset_index(drop=True)


def build_coefficients(beta: np.ndarray, l2: float, train_dates: list[str], test_dates: list[str]) -> pd.DataFrame:
    rows = [
        {
            "model_id": "logistic_v003",
            "term": "intercept",
            "coefficient": float(beta[0]),
            "l2": float(l2),
            "train_start": train_dates[0] if train_dates else "",
            "train_end": train_dates[-1] if train_dates else "",
            "test_start": test_dates[0] if test_dates else "",
            "test_end": test_dates[-1] if test_dates else "",
        }
    ]
    for feature, value in zip(MODEL_FEATURE_COLUMNS, beta[1:]):
        rows.append(
            {
                "model_id": "logistic_v003",
                "term": feature,
                "coefficient": float(value),
                "l2": float(l2),
                "train_start": train_dates[0] if train_dates else "",
                "train_end": train_dates[-1] if train_dates else "",
                "test_start": test_dates[0] if test_dates else "",
                "test_end": test_dates[-1] if test_dates else "",
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(
    samples_path: Path,
    output_dir: Path,
    samples: pd.DataFrame,
    train_dates: list[str],
    test_dates: list[str],
    initial_train_days: int,
    l2: float,
    target_return_pct: float,
    coefficients: pd.DataFrame,
    comparison: pd.DataFrame,
    rankwise: pd.DataFrame,
    top3_combo: pd.DataFrame,
    daily_top3: pd.DataFrame,
    data_quality: pd.DataFrame,
) -> str:
    lines = ["# logistic_v003 comparison", ""]
    lines.extend(
        [
            "## Scope",
            "",
            "This is a research-only training run. It does not replace run-daily or the current daily ranking model.",
            "logistic_v003_model.json is research-only and is not yet compatible with the existing daily_ranking/ranking_backtest model scorer.",
            "Do not use it directly as a replacement ranking model for run-daily at this stage.",
            f"- samples file: `{samples_path}`",
            f"- output dir: `{output_dir}`",
            f"- rows after filters: **{len(samples)}**",
            f"- signal dates: **{samples['signal_date'].nunique()}**",
            f"- target column: **{DEFAULT_TARGET_COLUMN}**",
            f"- target return pct: **{float(target_return_pct):.4f}**",
            f"- high return column: **{DEFAULT_HIGH_RETURN_COLUMN}**",
            f"- close return column: **{DEFAULT_CLOSE_RETURN_COLUMN}**",
            f"- date split: train **{train_dates[0]} to {train_dates[-1]}** ({len(train_dates)} dates), test **{test_dates[0]} to {test_dates[-1]}** ({len(test_dates)} dates)",
            f"- walk-forward initial train days: **{int(initial_train_days)}**",
            f"- logistic L2: **{float(l2):.4f}**",
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
                "eligible_rows_before_return_filter",
                "high_return_notna_rows",
                "close_return_notna_rows",
                "final_rows",
                "close_return_missing_count",
                "target_return_pct",
                "l2",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Coefficients",
            "",
            "| term | coefficient |",
            "|---|---:|",
        ]
    )
    for _, row in coefficients.iterrows():
        lines.append(f"| {row.get('term', '')} | {_format_number(row.get('coefficient'))} |")
    lines.extend(["", "## Comparison Summary", ""])
    lines.extend(_markdown_table(comparison, ["evaluation_scope", "model_id", "date_count", "candidate_count", "target_rate", "auc", "logloss", "top3_target_rate", "portfolio_realized_positive_rate", "portfolio_realized_hit7_rate"]))
    lines.extend(["", "## Rank Wise", ""])
    lines.extend(_markdown_table(rankwise, ["evaluation_scope", "model_id", "rank1_hit_rate", "rank2_hit_rate", "rank3_hit_rate", "rank1_avg_return", "rank2_avg_return", "rank3_avg_return"]))
    lines.extend(["", "## Top3 Combination", ""])
    lines.extend(_markdown_table(top3_combo, ["evaluation_scope", "model_id", "top3_target_rate", "top3_all_hit_rate", "hit_count_0_days", "hit_count_1_days", "hit_count_2_days", "hit_count_3_days", "avg_top3_high_return", "avg_top3_realized_return", "portfolio_realized_positive_rate", "portfolio_realized_hit7_rate"]))
    lines.extend(["", "## Daily Top3 Preview", ""])
    preview = daily_top3.sort_values(["evaluation_scope", "model_id", "signal_date"]).head(80)
    lines.extend(_markdown_table(preview, ["evaluation_scope", "model_id", "signal_date", "topn_codes", "hit_count", "avg_top3_high_return", "avg_top3_realized_return"]))
    return "\n".join(lines)


def _score_frame(
    frame: pd.DataFrame,
    model_id: str,
    scope: str,
    score_column: str,
    probability_column: str | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    result["model_id"] = model_id
    result["evaluation_scope"] = scope
    result["model_score"] = pd.to_numeric(result[score_column], errors="coerce")
    if probability_column and probability_column in result.columns:
        result["model_probability"] = pd.to_numeric(result[probability_column], errors="coerce")
    else:
        result["model_probability"] = pd.NA
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _logistic_loss(x_aug: np.ndarray, y: np.ndarray, beta: np.ndarray, reg: np.ndarray) -> float:
    z = x_aug @ beta
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    penalty = 0.5 * float(np.sum(reg * beta * beta))
    return float(loss + penalty)


def _auc(target: pd.Series, score: pd.Series) -> float | None:
    frame = pd.DataFrame({"target": target.astype(int), "score": pd.to_numeric(score, errors="coerce")}).dropna()
    positives = int(frame["target"].sum())
    negatives = int(len(frame) - positives)
    if positives <= 0 or negatives <= 0:
        return None
    ranks = frame["score"].rank(method="average")
    rank_sum_positive = float(ranks[frame["target"] == 1].sum())
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / float(positives * negatives)


def _logloss(target: pd.Series, probability: pd.Series) -> float | None:
    frame = pd.DataFrame({"target": target.astype(int), "probability": pd.to_numeric(probability, errors="coerce")}).dropna()
    if frame.empty:
        return None
    p = np.clip(frame["probability"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    y = frame["target"].to_numpy(dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.4f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["No rows."]
    cols = [column for column in columns if column in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame[cols].head(80).iterrows():
        values = []
        for column in cols:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(_format_number(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _suffix_from_samples_path(path: Path) -> str:
    stem = path.stem
    prefix = "history_candidates_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def parse_l2_grid(raw: str | list[float] | tuple[float, ...] | None) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        values = [float(part) for part in parts]
    else:
        values = [float(value) for value in raw]
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value <= 0:
            raise RuntimeError(f"l2 values must be positive: {value}")
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _l2_dir_name(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"l2_{text}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and validate logistic_v003 research ranking model.")
    parser.add_argument("--samples-file", default=str(DEFAULT_SAMPLES_FILE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--l2-grid", default=None)
    parser.add_argument("--target-return-pct", type=float, default=DEFAULT_TARGET_RETURN_PCT)
    args = parser.parse_args(argv)
    if args.l2_grid:
        comparison, rankwise, top3_combo, comparison_csv, rankwise_csv, top3_combo_csv = run_logistic_v003_l2_grid(
            samples_file=args.samples_file,
            output_dir=args.output_dir,
            top_n=args.top_n,
            initial_train_days=args.initial_train_days,
            target_return_pct=args.target_return_pct,
            l2_grid=args.l2_grid,
        )
        print(f"grid comparison rows: {len(comparison)}")
        print(f"grid rankwise rows: {len(rankwise)}")
        print(f"grid top3 combo rows: {len(top3_combo)}")
        print(f"grid comparison csv: {comparison_csv}")
        print(f"grid rankwise csv: {rankwise_csv}")
        print(f"grid top3 combo csv: {top3_combo_csv}")
        return 0
    coefficients, comparison, rankwise, top3_combo, daily_top3, data_quality, markdown_path = run_logistic_v003_research(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        initial_train_days=args.initial_train_days,
        l2=args.l2,
        target_return_pct=args.target_return_pct,
    )
    print(f"coefficients: {len(coefficients)}")
    print(f"comparison rows: {len(comparison)}")
    print(f"rankwise rows: {len(rankwise)}")
    print(f"top3 combo rows: {len(top3_combo)}")
    print(f"daily top3 rows: {len(daily_top3)}")
    print(f"data quality rows: {len(data_quality)}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

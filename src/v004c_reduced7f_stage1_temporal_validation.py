"""Temporal validation for the one preregistered reduced-7F Stage1 challenger.

This module deliberately exposes only two weighted-L2 logistic specifications:
the frozen current 18F representation and the preregistered reduced 7F
representation.  It performs no feature, hyperparameter, target, threshold, or
Top-K search and never reads an August signal-date outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004a import (
    DEFAULT_TARGET_COLUMN,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
)
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2,
    MODEL_FAMILY,
    POSITIVE_WEIGHT,
)


MODEL_ID = "REDUCED7F_WEIGHTED_LOGISTIC"
BASELINE_MODEL = "CURRENT_FROZEN_STAGE1_18F"
CHALLENGER_MODEL = "REDUCED7F_WEIGHTED_LOGISTIC"
MODEL_ORDER = (BASELINE_MODEL, CHALLENGER_MODEL)

REDUCED_FEATURE_COLUMNS = (
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
)
MODEL_FEATURES: Mapping[str, tuple[str, ...]] = {
    BASELINE_MODEL: tuple(FROZEN_FEATURE_COLUMNS),
    CHALLENGER_MODEL: REDUCED_FEATURE_COLUMNS,
}

BRIDGE_DIRNAME = "v004c_stage1_18f_bridge_analysis_v001_20260506_20260731"
BRIDGE_CANDIDATES = "v004c_stage1_18f_bridge_candidates.csv"
BRIDGE_SPEC = "v004c_stage1_model_spec.json"
OUTPUT_DIRNAME = "v004c_reduced7f_stage1_temporal_validation_v001_20260506_20260731"
OUTPUT_FILENAMES = (
    "v004c_reduced7f_fold_predictions_v001.csv",
    "v004c_reduced7f_daily_metrics_v001.csv",
    "v004c_reduced7f_period_summary_v001.csv",
    "v004c_reduced7f_paired_bootstrap_v001.csv",
    "v004c_reduced7f_fold_coefficients_v001.csv",
    "v004c_reduced7f_final_model_spec_v001.json",
    "v004c_reduced7f_review_v001.md",
)

DEVELOPMENT_START = "2026-05-06"
DEVELOPMENT_END = "2026-07-31"
AUGUST_ASOF = "2026-08-01"
FOLD_A_TRAIN_START = "2026-05-06"
FOLD_A_TRAIN_END_EXCLUSIVE = "2026-06-01"
FOLD_A_TEST_START = "2026-06-01"
FOLD_A_TEST_END_EXCLUSIVE = "2026-07-01"
FOLD_B_TRAIN_START = "2026-05-06"
FOLD_B_TRAIN_END_EXCLUSIVE = "2026-07-01"
FOLD_B_TEST_START = "2026-07-01"
FOLD_B_TEST_END_EXCLUSIVE = "2026-08-01"

BOOTSTRAP_SEED = 20260827
BOOTSTRAP_RESAMPLES = 20_000
MIN_TRAIN_SIGNAL_DATES = 18
SCOPE_ORDER = ("ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY")
PERIOD_ORDER = (
    "FOLD_A_MAY_TO_JUNE",
    "FOLD_B_MAY_JUNE_TO_JULY",
    "EXPANDING_JUNE",
    "EXPANDING_JULY_MATURE_ONLY",
    "EXPANDING_ALL_MATURE",
)

STRICT_JUNE_PATH = (
    "reports/research/v004c_stage1_risk_complementarity_v001_20260601_20260630/"
    "v004c_stage1_risk_strict_oof_v001.csv"
)
STRICT_JUNE_COEFFICIENT_PATH = (
    "reports/research/v004c_board2_board3_stage1_feature_response_v001_20260603_20260626/"
    "v004c_board_feature_response_coefficients_v001.csv"
)
STRICT_JUNE_AUDIT_PATH = (
    "reports/research/v004c_stage1_risk_complementarity_v001_20260601_20260630/"
    "v004c_stage1_risk_temporal_audit_v001.csv"
)
JULY_LOCK_PATH = (
    "reports/research/v004c_frozen_stage1_top2_july_forward_stress_v001_20260701_20260731/"
    "v004c_top2_july_prediction_lock_v001.csv"
)


def assert_experiment_contract() -> None:
    expected = (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    if REDUCED_FEATURE_COLUMNS != expected or len(REDUCED_FEATURE_COLUMNS) != 7:
        raise RuntimeError("FATAL: preregistered reduced 7F contract changed")
    if tuple(MODEL_FEATURES) != MODEL_ORDER:
        raise RuntimeError("FATAL: model zoo detected")
    if set(REDUCED_FEATURE_COLUMNS).difference(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: reduced model contains a new feature")
    if L2 != 0.30 or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen weighted-logistic hyperparameters changed")
    if MIN_TRAIN_SIGNAL_DATES != 18:
        raise RuntimeError("FATAL: frozen initial train-date gate changed")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _binary_auc(target: Sequence[float], score: Sequence[float]) -> float:
    y = np.asarray(target, dtype=float)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid].astype(int)
    s = s[valid]
    positive = y == 1
    negative = y == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy(float)
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _cosine(left: np.ndarray | None, right: np.ndarray) -> float:
    if left is None:
        return math.nan
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else math.nan


def load_bridge(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(root).resolve()
    bridge_dir = root_path / "reports" / "research" / BRIDGE_DIRNAME
    frame = pd.read_csv(
        bridge_dir / BRIDGE_CANDIDATES,
        encoding="utf-8-sig",
        dtype={"event_id": str, "code": str},
    )
    spec = json.loads((bridge_dir / BRIDGE_SPEC).read_text(encoding="utf-8"))
    for column in ("event_id", "code", "signal_date", "label_available_date"):
        frame[column] = frame[column].astype(str)
    frame["code"] = frame["code"].str.zfill(6)
    assert_experiment_contract()
    if spec.get("features") != list(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: bridge does not contain exact frozen 18F order")
    if str(spec.get("model_id")) != "V4A_ARCH_TRANSFER_V4C":
        raise RuntimeError("FATAL: bridge model identity changed")
    if len(frame) != 497 or frame["signal_date"].nunique() != 62:
        raise RuntimeError("FATAL: bridge population parity failed")
    if frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in bridge")
    if frame[FROZEN_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("FATAL: frozen feature missingness detected")
    if frame["signal_date"].min() != DEVELOPMENT_START or frame["signal_date"].max() != DEVELOPMENT_END:
        raise RuntimeError("FATAL: development date boundary changed")
    if frame["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal-date row accessed")
    mature = frame["label_available_date"].lt(AUGUST_ASOF)
    outcome_columns = [
        "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
    ]
    if frame.loc[mature, outcome_columns].isna().any().any():
        raise RuntimeError("FATAL: mature outcome is missing")
    if frame.loc[~mature, outcome_columns].notna().any().any():
        raise RuntimeError("FATAL: post-asof outcome was read or exported")
    if (int(mature.sum()), int(frame.loc[mature, "signal_date"].nunique())) != (485, 60):
        raise RuntimeError("FATAL: mature development parity failed")
    frame["board_group"] = frame["board_group"].astype(str)
    return frame.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True), spec


def _training_adapter(train: pd.DataFrame) -> pd.DataFrame:
    result = train.copy()
    result[DEFAULT_TARGET_COLUMN] = result["target7"].astype(bool)
    raw = pd.to_numeric(result["raw_repair_return"], errors="raise")
    result["tail_weight"] = np.select(
        [raw.ge(0.12), raw.ge(0.10)], [2.0, 1.5], default=1.0
    ).astype(float)
    return result


def fit_reduced7f(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = REDUCED_FEATURE_COLUMNS
    if train.empty:
        raise RuntimeError("FATAL: empty temporal training sample")
    if train[list(features)].isna().any().any():
        raise RuntimeError("FATAL: missing feature reached learner")
    adapter = _training_adapter(train)
    weight = build_training_sample_weight(adapter, positive_weight=POSITIVE_WEIGHT)
    beta = fit_logistic_l2_weighted(
        adapter[list(features)].to_numpy(float),
        adapter[DEFAULT_TARGET_COLUMN].astype(int).to_numpy(float),
        l2=L2,
        sample_weight=weight,
    )
    return beta, weight


def _rank_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    ranked: list[pd.DataFrame] = []
    for _, day in frame.groupby(["fold_type", "model", "signal_date"], sort=True):
        part = day.sort_values(
            ["model_score", "event_id"], ascending=[False, True], kind="mergesort"
        ).copy()
        part["model_rank"] = np.arange(1, len(part) + 1)
        ranked.append(part)
    return pd.concat(ranked, ignore_index=True).sort_values(
        ["fold_type", "model", "signal_date", "model_rank", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    fold_type: str,
    fold_id: str,
    test_date: str,
    previous: Mapping[str, np.ndarray | None],
    models: Sequence[str] = (CHALLENGER_MODEL,),
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, np.ndarray]]:
    if train.empty or test.empty:
        raise RuntimeError(f"FATAL: empty train/test in {fold_id}")
    if not train["label_available_date"].lt(test_date).all():
        raise RuntimeError(f"FATAL: label leakage in {fold_id}")
    if not train["signal_date"].lt(test_date).all():
        raise RuntimeError(f"FATAL: future signal row in {fold_id}")
    predictions: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []
    current: dict[str, np.ndarray] = {}
    train_dates = sorted(train["signal_date"].unique())
    max_label = str(train["label_available_date"].max())
    if tuple(models) != (CHALLENGER_MODEL,):
        raise RuntimeError("FATAL: only the single preregistered challenger may be fit")
    for model in models:
        features = MODEL_FEATURES[model]
        if tuple(features) != REDUCED_FEATURE_COLUMNS:
            raise RuntimeError("FATAL: a non-preregistered challenger reached the learner")
        beta, sample_weight = fit_reduced7f(train)
        score = _sigmoid(beta[0] + test[list(features)].to_numpy(float) @ beta[1:])
        predicted = test[[
            "event_id", "signal_date", "code", "board_group", "candidate_count",
            "label_available_date", "target7", "positive_non_target", "loss",
            "severe_loss", "raw_repair_return", "capped_return_7",
        ]].copy()
        predicted.insert(0, "fold_id", fold_id)
        predicted.insert(0, "fold_type", fold_type)
        predicted.insert(2, "model", model)
        predicted["model_score"] = score
        predicted["train_rows"] = len(train)
        predicted["train_dates"] = len(train_dates)
        predicted["train_end_signal_date"] = train_dates[-1]
        predicted["max_train_label_available_date"] = max_label
        predictions.append(predicted)
        coef = np.asarray(beta[1:], dtype=float)
        current[model] = coef
        adjacent = _cosine(previous.get(model), coef) if fold_type == "EXPANDING_OOF" else math.nan
        common = {
            "fold_type": fold_type,
            "fold_id": fold_id,
            "test_date": test_date,
            "model": model,
            "feature_count": len(features),
            "train_rows": len(train),
            "train_dates": len(train_dates),
            "train_start_signal_date": train_dates[0],
            "train_end_signal_date": train_dates[-1],
            "max_train_label_available_date": max_label,
            "train_target7_rate": float(train["target7"].mean()),
            "weighted_target7_rate": float(
                np.sum(sample_weight * train["target7"].to_numpy(float)) / sample_weight.sum()
            ),
            "test_rows": len(test),
            "test_target7_rate": float(test["target7"].mean()),
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "adjacent_coefficient_cosine": adjacent,
        }
        coefficients.append({**common, "feature": "__INTERCEPT__", "coefficient": float(beta[0]), "coefficient_sign": int(np.sign(beta[0]))})
        coefficients.extend(
            {**common, "feature": feature, "coefficient": float(value), "coefficient_sign": int(np.sign(value))}
            for feature, value in zip(features, coef)
        )
    return predictions, coefficients, current


def _load_frozen_baseline(
    root: Path, frame: pd.DataFrame, spec: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reuse frozen Current18F predictions and coefficients without fitting it."""
    dtype = {"event_id": str, "signal_date": str, "code": str}
    june = pd.read_csv(root / STRICT_JUNE_PATH, encoding="utf-8-sig", dtype=dtype)
    july = pd.read_csv(root / JULY_LOCK_PATH, encoding="utf-8-sig", dtype=dtype)
    june_audit = pd.read_csv(root / STRICT_JUNE_AUDIT_PATH, encoding="utf-8-sig")
    expected_june_dates = (
        "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08",
        "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
        "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18",
        "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
        "2026-06-26",
    )
    if tuple(sorted(june["signal_date"].unique())) != expected_june_dates or len(june) != 155:
        raise RuntimeError("FATAL: strict June frozen baseline parity failed")
    june_audit["test_date"] = june_audit["test_date"].astype(str)
    june_audit = june_audit[june_audit["strict_date_available"].astype(str).str.lower().eq("true")].copy()
    if tuple(sorted(june_audit["test_date"])) != expected_june_dates:
        raise RuntimeError("FATAL: strict June fold-audit parity failed")
    audit_by_date = june_audit.set_index("test_date")
    if len(july) != 178 or july["signal_date"].nunique() != 23:
        raise RuntimeError("FATAL: July frozen baseline parity failed")
    if june["event_id"].duplicated().any() or july["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate frozen baseline event")

    outcome_columns = [
        "event_id", "label_available_date", "target7", "positive_non_target",
        "loss", "severe_loss", "raw_repair_return", "capped_return_7",
        "board_group", "candidate_count", "code", "signal_date",
    ]
    mature = frame[frame["label_available_date"].lt(AUGUST_ASOF)][outcome_columns].copy()
    mature_ids = set(mature["event_id"])
    july = july[july["event_id"].isin(mature_ids)].copy()
    if len(july) != 166 or july["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: July mature-only frozen baseline parity failed")

    def attach(source: pd.DataFrame, score: str, rank: str, fold_type: str) -> pd.DataFrame:
        identity = source[["event_id", score, rank]].copy()
        merged = identity.merge(mature, on="event_id", how="left", validate="one_to_one")
        if merged[outcome_columns[1:]].isna().any().any():
            raise RuntimeError(f"FATAL: frozen baseline identity/outcome mismatch: {fold_type}")
        merged = merged.rename(columns={score: "model_score", rank: "model_rank"})
        merged.insert(0, "fold_id", merged["signal_date"].map(lambda date: f"{fold_type}__{date}"))
        merged.insert(0, "fold_type", fold_type)
        merged.insert(2, "model", BASELINE_MODEL)
        if fold_type == "FOLD_A_MAY_TO_JUNE":
            merged["train_rows"] = merged["signal_date"].map(
                audit_by_date["matured_training_rows"].astype(int)
            )
            merged["train_dates"] = merged["signal_date"].map(
                audit_by_date["matured_training_dates"].astype(int)
            )
            merged["train_end_signal_date"] = merged["signal_date"].map(
                audit_by_date["latest_training_signal_date"].astype(str)
            )
            merged["max_train_label_available_date"] = merged["signal_date"].map(
                audit_by_date["latest_training_label_available_date"].astype(str)
            )
        else:
            merged["train_rows"] = 307
            merged["train_dates"] = 37
            merged["train_end_signal_date"] = "2026-06-26"
            merged["max_train_label_available_date"] = "2026-06-30"
        return merged

    june_base = attach(june, "stage1_score", "stage1_rank", "FOLD_A_MAY_TO_JUNE")
    july_base = attach(july, "stage1_score", "stage1_rank", "FOLD_B_MAY_JUNE_TO_JULY")
    expanding = pd.concat([
        june_base.assign(
            fold_type="EXPANDING_OOF",
            fold_id=june_base["signal_date"].map(lambda date: f"EXPANDING_OOF__{date}"),
        ),
        july_base.assign(
            fold_type="EXPANDING_OOF",
            fold_id=july_base["signal_date"].map(lambda date: f"EXPANDING_OOF__{date}"),
        ),
    ], ignore_index=True)
    predictions = pd.concat([june_base, july_base, expanding], ignore_index=True)

    june_coef = pd.read_csv(root / STRICT_JUNE_COEFFICIENT_PATH, encoding="utf-8-sig")
    if set(june_coef["feature"]) != set(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: archived strict 18F coefficient manifest changed")
    coefficient_rows: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    for test_date in expected_june_dates:
        fold = june_coef[june_coef["test_date"].astype(str).eq(test_date)].set_index("feature")
        fold_audit = audit_by_date.loc[test_date]
        vector = np.asarray([float(fold.loc[feature, "beta"]) for feature in FROZEN_FEATURE_COLUMNS])
        intercept = float(fold["intercept"].iloc[0])
        common = {
            "fold_type": "EXPANDING_OOF", "fold_id": f"EXPANDING_OOF__{test_date}",
            "test_date": test_date, "model": BASELINE_MODEL, "feature_count": 18,
            "train_rows": int(fold_audit["matured_training_rows"]),
            "train_dates": int(fold_audit["matured_training_dates"]),
            "train_start_signal_date": "2026-05-06",
            "train_end_signal_date": str(fold_audit["latest_training_signal_date"]),
            "max_train_label_available_date": str(fold_audit["latest_training_label_available_date"]),
            "train_target7_rate": np.nan, "weighted_target7_rate": np.nan,
            "test_rows": int((june["signal_date"] == test_date).sum()),
            "test_target7_rate": float(june.loc[june["signal_date"] == test_date, "target7"].mean()),
            "l2": L2, "positive_weight": POSITIVE_WEIGHT,
            "adjacent_coefficient_cosine": _cosine(previous, vector),
        }
        coefficient_rows.append({**common, "feature": "__INTERCEPT__", "coefficient": intercept, "coefficient_sign": int(np.sign(intercept))})
        coefficient_rows.extend(
            {**common, "feature": feature, "coefficient": float(value), "coefficient_sign": int(np.sign(value))}
            for feature, value in zip(FROZEN_FEATURE_COLUMNS, vector)
        )
        previous = vector
    frozen_vector = np.asarray([float(spec["coefficients"][f]) for f in FROZEN_FEATURE_COLUMNS])
    for test_date in sorted(july["signal_date"].unique()):
        common = {
            "fold_type": "EXPANDING_OOF", "fold_id": f"EXPANDING_OOF__{test_date}",
            "test_date": test_date, "model": BASELINE_MODEL, "feature_count": 18,
            "train_rows": 307, "train_dates": 37,
            "train_start_signal_date": "2026-05-06", "train_end_signal_date": "2026-06-26",
            "max_train_label_available_date": "2026-06-30",
            "train_target7_rate": np.nan, "weighted_target7_rate": np.nan,
            "test_rows": int((july["signal_date"] == test_date).sum()),
            "test_target7_rate": float(mature.loc[mature["signal_date"] == test_date, "target7"].mean()),
            "l2": L2, "positive_weight": POSITIVE_WEIGHT,
            "adjacent_coefficient_cosine": _cosine(previous, frozen_vector),
        }
        coefficient_rows.append({**common, "feature": "__INTERCEPT__", "coefficient": float(spec["intercept"]), "coefficient_sign": int(np.sign(spec["intercept"]))})
        coefficient_rows.extend(
            {**common, "feature": feature, "coefficient": float(value), "coefficient_sign": int(np.sign(value))}
            for feature, value in zip(FROZEN_FEATURE_COLUMNS, frozen_vector)
        )
        previous = frozen_vector
    return predictions, pd.DataFrame(coefficient_rows), {
        "june_rows": len(june), "june_dates": len(expected_june_dates),
        "july_mature_rows": len(july), "july_mature_dates": july["signal_date"].nunique(),
        "current18f_fit_count": 0,
    }


def build_temporal_predictions(
    root: Path, frame: pd.DataFrame, spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mature = frame[frame["label_available_date"].lt(AUGUST_ASOF)].copy()
    baseline_predictions, baseline_coefficients, baseline_audit = _load_frozen_baseline(root, frame, spec)
    prediction_frames: list[pd.DataFrame] = [baseline_predictions]
    coefficient_rows: list[dict[str, Any]] = baseline_coefficients.to_dict("records")

    fixed_contracts = (
        (
            "FOLD_A_MAY_TO_JUNE", FOLD_A_TRAIN_START, FOLD_A_TRAIN_END_EXCLUSIVE,
            FOLD_A_TEST_START, FOLD_A_TEST_END_EXCLUSIVE,
        ),
        (
            "FOLD_B_MAY_JUNE_TO_JULY", FOLD_B_TRAIN_START, FOLD_B_TRAIN_END_EXCLUSIVE,
            FOLD_B_TEST_START, FOLD_B_TEST_END_EXCLUSIVE,
        ),
    )
    for fold_type, train_start, train_end, test_start, test_end in fixed_contracts:
        baseline_fold = baseline_predictions[baseline_predictions["fold_type"].eq(fold_type)]
        test_dates = sorted(baseline_fold["signal_date"].unique())
        if not test_dates:
            raise RuntimeError(f"FATAL: no frozen baseline dates for {fold_type}")
        earliest_test_date = test_dates[0]
        train = mature[
            mature["signal_date"].ge(train_start)
            & mature["signal_date"].lt(train_end)
            & mature["label_available_date"].lt(earliest_test_date)
        ].copy()
        test = mature[mature["signal_date"].isin(test_dates)].copy()
        if train["signal_date"].nunique() < MIN_TRAIN_SIGNAL_DATES:
            raise RuntimeError(f"FATAL: fixed temporal fold below 18-date gate: {fold_type}")
        fold_id = fold_type
        predictions, coefficients, _ = _fit_and_score(
            train, test, fold_type=fold_type, fold_id=fold_id,
            test_date=earliest_test_date, previous={CHALLENGER_MODEL: None},
        )
        prediction_frames.extend(predictions)
        coefficient_rows.extend(coefficients)

    previous = {CHALLENGER_MODEL: None}
    all_dates = sorted(
        baseline_predictions.loc[baseline_predictions["fold_type"].eq("EXPANDING_OOF"), "signal_date"].unique()
    )
    for test_date in all_dates:
        train = mature[
            mature["signal_date"].lt(test_date)
            & mature["label_available_date"].lt(test_date)
        ].copy()
        if train.empty:
            continue
        if train["signal_date"].nunique() < MIN_TRAIN_SIGNAL_DATES:
            continue
        test = mature[mature["signal_date"].eq(test_date)].copy()
        fold_id = f"EXPANDING_OOF__{test_date}"
        predictions, coefficients, current = _fit_and_score(
            train, test, fold_type="EXPANDING_OOF", fold_id=fold_id,
            test_date=test_date, previous=previous,
        )
        prediction_frames.extend(predictions)
        coefficient_rows.extend(coefficients)
        previous = current

    predictions = _rank_prediction(pd.concat(prediction_frames, ignore_index=True))
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["fold_type", "test_date", "model", "feature"], kind="mergesort"
    ).reset_index(drop=True)
    concrete_train_end = pd.to_datetime(
        predictions["train_end_signal_date"], format="%Y-%m-%d", errors="coerce"
    )
    concrete_max_label = pd.to_datetime(
        predictions["max_train_label_available_date"], format="%Y-%m-%d", errors="coerce"
    )
    prediction_date = pd.to_datetime(predictions["signal_date"], errors="raise")
    leakage = {
        "prediction_rows": len(predictions),
        "self_or_future_signal_leakage_rows": int(
            (concrete_train_end.notna() & concrete_train_end.ge(prediction_date)).sum()
        ),
        "current_test_label_leakage_rows": int(
            (concrete_max_label.notna() & concrete_max_label.ge(prediction_date)).sum()
        ),
        "august_signal_rows_accessed": int(predictions["signal_date"].ge(AUGUST_ASOF).sum()),
        **baseline_audit,
    }
    violation_keys = (
        "self_or_future_signal_leakage_rows",
        "current_test_label_leakage_rows",
        "august_signal_rows_accessed",
        "current18f_fit_count",
    )
    if any(leakage[key] for key in violation_keys):
        raise RuntimeError(f"FATAL: temporal leakage audit failed: {leakage}")
    return predictions, coefficients, leakage


def _scope_filter(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "ALL_CANDIDATES":
        return frame
    if scope == "BOARD2_ONLY":
        return frame[frame["board_group"].eq("BOARD2")]
    if scope == "BOARD3_ONLY":
        return frame[frame["board_group"].eq("BOARD3")]
    raise ValueError(scope)


def build_daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold_type, model, date), group in predictions.groupby(
        ["fold_type", "model", "signal_date"], sort=True
    ):
        for scope in SCOPE_ORDER:
            day = _scope_filter(group, scope)
            if day.empty:
                continue
            ranked = day.sort_values(
                ["model_score", "event_id"], ascending=[False, True], kind="mergesort"
            ).copy()
            ranked["scope_rank"] = np.arange(1, len(ranked) + 1)
            universe_capped = float(ranked["capped_return_7"].mean())
            row: dict[str, Any] = {
                "fold_type": fold_type,
                "model": model,
                "signal_date": date,
                "month": date[:7],
                "population_scope": scope,
                "candidate_count": len(ranked),
                "universe_target7_rate": float(ranked["target7"].mean()),
                "universe_loss_rate": float(ranked["loss"].mean()),
                "universe_capped_return": universe_capped,
                "target7_within_date_auc": _binary_auc(ranked["target7"], ranked["model_score"]),
            }
            winners = int(ranked["target7"].sum())
            for k in (1, 2, 3, 5):
                selected = ranked.head(min(k, len(ranked)))
                row[f"top{k}_selected_rows"] = len(selected)
                row[f"top{k}_target7_rate"] = float(selected["target7"].mean())
                row[f"top{k}_loss_rate"] = float(selected["loss"].mean())
                row[f"top{k}_severe_loss_rate"] = float(selected["severe_loss"].mean())
                row[f"top{k}_capped_return"] = float(selected["capped_return_7"].mean())
                row[f"top{k}_excess_vs_universe"] = row[f"top{k}_capped_return"] - universe_capped
                row[f"recall_at_{k}"] = float(selected["target7"].sum() / winners) if winners else math.nan
            row["top3_negative_date"] = int(row["top3_capped_return"] < 0)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["fold_type", "signal_date", "population_scope", "model"], kind="mergesort"
    ).reset_index(drop=True)


def _period_filter(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "FOLD_A_MAY_TO_JUNE":
        return daily[daily["fold_type"].eq("FOLD_A_MAY_TO_JUNE")]
    if period == "FOLD_B_MAY_JUNE_TO_JULY":
        return daily[daily["fold_type"].eq("FOLD_B_MAY_JUNE_TO_JULY")]
    expanding = daily[daily["fold_type"].eq("EXPANDING_OOF")]
    if period == "EXPANDING_JUNE":
        return expanding[expanding["month"].eq("2026-06")]
    if period == "EXPANDING_JULY_MATURE_ONLY":
        return expanding[expanding["month"].eq("2026-07")]
    if period == "EXPANDING_ALL_MATURE":
        return expanding
    raise ValueError(period)


def build_period_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIOD_ORDER:
        period_frame = _period_filter(daily, period)
        for scope in SCOPE_ORDER:
            for model in MODEL_ORDER:
                part = period_frame[
                    period_frame["population_scope"].eq(scope)
                    & period_frame["model"].eq(model)
                ]
                if part.empty:
                    continue
                row: dict[str, Any] = {
                    "period": period,
                    "population_scope": scope,
                    "model": model,
                    "feature_count": len(MODEL_FEATURES[model]),
                    "dates": int(part["signal_date"].nunique()),
                    "candidate_rows": int(part["candidate_count"].sum()),
                    "target7_within_date_auc": float(part["target7_within_date_auc"].mean()),
                    "eligible_auc_dates": int(part["target7_within_date_auc"].notna().sum()),
                    "universe_target7_rate": float(part["universe_target7_rate"].mean()),
                    "universe_loss_rate": float(part["universe_loss_rate"].mean()),
                    "universe_capped_return": float(part["universe_capped_return"].mean()),
                }
                for k in (1, 2, 3):
                    row[f"top{k}_target7_rate"] = float(part[f"top{k}_target7_rate"].mean())
                    row[f"top{k}_loss_rate"] = float(part[f"top{k}_loss_rate"].mean())
                    row[f"top{k}_severe_loss_rate"] = float(part[f"top{k}_severe_loss_rate"].mean())
                    row[f"top{k}_capped_return"] = float(part[f"top{k}_capped_return"].mean())
                    row[f"top{k}_excess_vs_universe"] = float(part[f"top{k}_excess_vs_universe"].mean())
                row["top3_negative_date_rate"] = float(part["top3_negative_date"].mean())
                row["worst_daily_top3_capped_return"] = float(part["top3_capped_return"].min())
                row["top3_loss_excess"] = row["top3_loss_rate"] - row["universe_loss_rate"]
                for k in (1, 2, 3, 5):
                    valid = part[f"recall_at_{k}"].notna()
                    row[f"recall_at_{k}"] = float(part.loc[valid, f"recall_at_{k}"].mean()) if valid.any() else math.nan
                    row[f"recall_at_{k}_eligible_dates"] = int(valid.sum())
                row["support_warning"] = (
                    "LOW_N" if scope == "BOARD3_ONLY" and (row["dates"] < 5 or row["eligible_auc_dates"] < 5) else ""
                )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["period", "population_scope", "model"], kind="mergesort"
    ).reset_index(drop=True)


def build_paired_bootstrap(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_index, period in enumerate(PERIOD_ORDER):
        part = _period_filter(daily, period)
        part = part[part["population_scope"].eq("ALL_CANDIDATES")]
        pivot = part.pivot(index="signal_date", columns="model", values="top3_capped_return").dropna()
        if pivot.empty:
            continue
        delta = pivot[CHALLENGER_MODEL].to_numpy(float) - pivot[BASELINE_MODEL].to_numpy(float)
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, period_index]))
        draws = rng.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
        boot = delta[draws].mean(axis=1)
        rows.append({
            "period": period,
            "challenger": CHALLENGER_MODEL,
            "baseline": BASELINE_MODEL,
            "sampling_unit": "signal_date",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "dates": len(delta),
            "estimate_mean": float(delta.mean()),
            "observed_median": float(np.median(delta)),
            "positive_days": int((delta > 0).sum()),
            "negative_days": int((delta < 0).sum()),
            "zero_days": int((delta == 0).sum()),
            "bootstrap_p2_5": float(np.quantile(boot, 0.025)),
            "bootstrap_p50": float(np.quantile(boot, 0.50)),
            "bootstrap_p97_5": float(np.quantile(boot, 0.975)),
            "bootstrap_probability_delta_gt_0": float((boot > 0).mean()),
        })
    return pd.DataFrame(rows)


def add_coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    result = coefficients.copy()
    result["coefficient_median"] = math.nan
    result["coefficient_iqr"] = math.nan
    result["sign_flip_count"] = math.nan
    expanding = result[
        result["fold_type"].eq("EXPANDING_OOF")
        & result["feature"].ne("__INTERCEPT__")
    ]
    for (model, feature), group in expanding.groupby(["model", "feature"], sort=True):
        ordered = group.sort_values("test_date", kind="mergesort")
        values = ordered["coefficient"].to_numpy(float)
        signs = np.sign(values)
        flips = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        mask = result["model"].eq(model) & result["feature"].eq(feature)
        result.loc[mask, "coefficient_median"] = float(np.median(values))
        result.loc[mask, "coefficient_iqr"] = float(np.quantile(values, .75) - np.quantile(values, .25))
        result.loc[mask, "sign_flip_count"] = flips
    return result


def fit_final_reduced(frame: pd.DataFrame, source_commit: str) -> dict[str, Any]:
    train = frame[frame["label_available_date"].lt(AUGUST_ASOF)].copy()
    beta, weight = fit_reduced7f(train)
    dates = sorted(train["signal_date"].unique())
    return {
        "model_id": "FINAL_REDUCED7F_STAGE1",
        "model_family": MODEL_FAMILY,
        "purpose": "FROZEN_COEFFICIENTS_FOR_POSSIBLE_LATER_AUGUST_FINAL_HOLDOUT_ONLY",
        "in_sample_performance_used_as_evidence": False,
        "git_commit": source_commit,
        "feature_count": 7,
        "feature_order": list(REDUCED_FEATURE_COLUMNS),
        "intercept": float(beta[0]),
        "coefficients": {feature: float(value) for feature, value in zip(REDUCED_FEATURE_COLUMNS, beta[1:])},
        "L2": L2,
        "positive_weight": POSITIVE_WEIGHT,
        "standardization": "NONE",
        "PCA": "NONE",
        "missing_policy": "reuse exact bridge frozen features; no missing 7F accepted by learner",
        "sample_weight_rules": {
            "date_weight": "1 / candidate rows on historical signal_date",
            "positive_class_weight": "1.5 when target7 else 1.0",
            "tail_weight": {
                "raw_repair_return_gte_0.12": 2.0,
                "raw_repair_return_gte_0.10_and_lt_0.12": 1.5,
                "otherwise": 1.0,
            },
            "formula": "date_weight * positive_class_weight * tail_weight",
            "sum_weight": float(weight.sum()),
        },
        "target": "target7 = int(raw_repair_return >= 0.07)",
        "training_rows": len(train),
        "training_dates": len(dates),
        "training_start_signal_date": dates[0],
        "training_end_signal_date": dates[-1],
        "training_asof": AUGUST_ASOF,
        "label_availability_rule": "label_available_date < 2026-08-01",
        "max_label_available_date": str(train["label_available_date"].max()),
        "august_signal_date_outcome_rows_accessed": 0,
    }


def analyze(root: str | Path, source_commit: str = "UNKNOWN") -> dict[str, Any]:
    frame, bridge_spec = load_bridge(root)
    root_path = Path(root).resolve()
    predictions, raw_coefficients, leakage = build_temporal_predictions(root_path, frame, bridge_spec)
    coefficients = add_coefficient_stability(raw_coefficients)
    daily = build_daily_metrics(predictions)
    summary = build_period_summary(daily)
    bootstrap = build_paired_bootstrap(daily)
    final_spec = fit_final_reduced(frame, source_commit)
    return {
        "root": Path(root).resolve(),
        "frame": frame,
        "bridge_spec": bridge_spec,
        "predictions": predictions,
        "coefficients": coefficients,
        "leakage": leakage,
        "daily": daily,
        "summary": summary,
        "bootstrap": bootstrap,
        "final_spec": final_spec,
    }


def summary_row(context: Mapping[str, Any], period: str, model: str, scope: str = "ALL_CANDIDATES") -> pd.Series:
    rows = context["summary"]
    match = rows[
        rows["period"].eq(period)
        & rows["model"].eq(model)
        & rows["population_scope"].eq(scope)
    ]
    if len(match) != 1:
        raise RuntimeError(f"FATAL: missing/duplicate summary row {period}/{scope}/{model}")
    return match.iloc[0]


def _pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4%}"


def render_review(
    context: Mapping[str, Any], model_state: str, next_action: str,
    answers: Mapping[str, str],
) -> str:
    allowed_states = {
        "REDUCED7F_TEMPORAL_SIGNAL_SUPPORTED",
        "REDUCED7F_TEMPORAL_SIGNAL_NOT_SUPPORTED",
        "MIXED",
    }
    action_map = {
        "REDUCED7F_TEMPORAL_SIGNAL_SUPPORTED": "FREEZE_REDUCED7F_FOR_AUGUST_HOLDOUT",
        "REDUCED7F_TEMPORAL_SIGNAL_NOT_SUPPORTED": "STOP_STAGE1_MODEL_EXPANSION",
        "MIXED": "STOP_AND_REVIEW",
    }
    if model_state not in allowed_states or next_action != action_map[model_state]:
        raise RuntimeError("FATAL: review state/action contract violated")
    if set(answers) != {f"Q{i}" for i in range(1, 8)}:
        raise RuntimeError("FATAL: review must answer exactly Q1-Q7")
    lines = ["# v004c Reduced7F Stage1 Temporal Validation v001", ""]
    questions = (
        "Reduced7F 是否严格只用了预注册 7F？",
        "所有 temporal fold 是否严格遵守 label_available_date < test signal_date？",
        "May→June 是否优于/至少不差于 Current18F？",
        "May+June→July 是否优于/至少不差于 Current18F？",
        "Expanding OOF 是否支持稳定 selection alpha？",
        "Reduced7F 是否改善 upside ranking 而没有明显恶化 risk？",
        "是否有资格冻结并进行一次 August final holdout？",
    )
    for index, question in enumerate(questions, 1):
        lines.extend([f"## Q{index}. {question}", "", answers[f"Q{index}"], ""])
    lines.extend([
        "", f"MODEL_STATE = {model_state}", "", f"NEXT_ACTION = {next_action}", "",
    ])
    return "\n".join(lines)


def build_outputs(
    context: Mapping[str, Any], model_state: str, next_action: str,
    answers: Mapping[str, str],
) -> dict[str, bytes]:
    review = render_review(context, model_state, next_action, answers)
    return {
        OUTPUT_FILENAMES[0]: _csv_bytes(context["predictions"]),
        OUTPUT_FILENAMES[1]: _csv_bytes(context["daily"]),
        OUTPUT_FILENAMES[2]: _csv_bytes(context["summary"]),
        OUTPUT_FILENAMES[3]: _csv_bytes(context["bootstrap"]),
        OUTPUT_FILENAMES[4]: _csv_bytes(context["coefficients"]),
        OUTPUT_FILENAMES[5]: _json_bytes(context["final_spec"]),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }


def write_outputs(
    context: Mapping[str, Any], model_state: str, next_action: str,
    answers: Mapping[str, str], output_dir: str | Path | None = None,
) -> Path:
    target = Path(output_dir) if output_dir is not None else (
        context["root"] / "reports" / "research" / OUTPUT_DIRNAME
    )
    first = build_outputs(context, model_state, next_action, answers)
    second = build_outputs(context, model_state, next_action, answers)
    if first != second:
        raise RuntimeError("FATAL: deterministic output build failed")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (target / name).write_bytes(payload)
    return target


def output_sha256s(output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }

"""Four-spec training sanity audit for the frozen Reduced7F representation.

The experiment is intentionally closed: exactly S0/S1/S2/S3 are fit with the
same seven features and the repository's canonical weighted-L2 logistic
learner.  The only allowed changes are the preregistered tail-weight switch and
L2 values.  No August signal-date outcome is read.
"""

from __future__ import annotations

import hashlib
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
from .v004c_reduced7f_stage1_temporal_validation import (
    AUGUST_ASOF,
    BRIDGE_DIRNAME,
    REDUCED_FEATURE_COLUMNS,
    _binary_auc,
    _sigmoid,
    load_bridge,
)
from .v004c_v4a_architecture_transfer import MODEL_FAMILY, POSITIVE_WEIGHT


TASK_NAME = "v004c_reduced7f_training_spec_sanity_v001"
OUTPUT_DIRNAME = "v004c_reduced7f_training_spec_sanity_v001_20260506_20260731"
OUTPUT_FILENAMES = (
    "v004c_reduced7f_spec_training_fit_v001.csv",
    "v004c_reduced7f_spec_rankwise_v001.csv",
    "v004c_reduced7f_spec_oracle_gap_v001.csv",
    "v004c_reduced7f_spec_fold_predictions_v001.csv",
    "v004c_reduced7f_spec_temporal_summary_v001.csv",
    "v004c_reduced7f_spec_coefficients_v001.csv",
    "v004c_reduced7f_spec_paired_daily_v001.csv",
    "v004c_reduced7f_training_spec_review_v001.md",
)

SPECIFICATIONS: Mapping[str, Mapping[str, Any]] = {
    "S0_CURRENT_CONTROL": {"l2": 0.30, "tail_weighting": "LEGACY"},
    "S1_NO_TAIL_L2_030": {"l2": 0.30, "tail_weighting": "NONE"},
    "S2_NO_TAIL_L2_010": {"l2": 0.10, "tail_weighting": "NONE"},
    "S3_NO_TAIL_L2_003": {"l2": 0.03, "tail_weighting": "NONE"},
}
SPEC_ORDER = tuple(SPECIFICATIONS)
CONTROL_SPEC = SPEC_ORDER[0]
CHALLENGER_SPECS = SPEC_ORDER[1:]

FULL_PERIODS: Mapping[str, tuple[str, str]] = {
    "MAY": ("2026-05-01", "2026-06-01"),
    "JUNE": ("2026-06-01", "2026-07-01"),
    "JULY_MATURE_ONLY": ("2026-07-01", "2026-08-01"),
    "MAY_JUNE_JULY_MATURE": ("2026-05-01", "2026-08-01"),
}
TEMPORAL_PERIOD_ORDER = (
    "FOLD_A_MAY_TO_JUNE",
    "FOLD_B_MAY_JUNE_TO_JULY",
    "EXPANDING_JUNE",
    "EXPANDING_JULY_MATURE_ONLY",
    "EXPANDING_ALL_MATURE",
)

MIN_TRAIN_SIGNAL_DATES = 18
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_RESAMPLES = 20_000
PREVIOUS_REDUCED_SPEC = (
    "reports/research/v004c_reduced7f_stage1_temporal_validation_v001_20260506_20260731/"
    "v004c_reduced7f_final_model_spec_v001.json"
)


def assert_experiment_contract() -> None:
    if len(REDUCED_FEATURE_COLUMNS) != 7:
        raise RuntimeError("FATAL: Reduced7F feature count changed")
    if SPEC_ORDER != (
        "S0_CURRENT_CONTROL",
        "S1_NO_TAIL_L2_030",
        "S2_NO_TAIL_L2_010",
        "S3_NO_TAIL_L2_003",
    ):
        raise RuntimeError("FATAL: specification set changed")
    expected = ((0.30, "LEGACY"), (0.30, "NONE"), (0.10, "NONE"), (0.03, "NONE"))
    observed = tuple(
        (float(SPECIFICATIONS[name]["l2"]), str(SPECIFICATIONS[name]["tail_weighting"]))
        for name in SPEC_ORDER
    )
    if observed != expected or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: preregistered training specification changed")
    if any(float(spec["l2"]) < 0.03 for spec in SPECIFICATIONS.values()):
        raise RuntimeError("FATAL: forbidden L2 reached experiment")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def build_sample_weight(train: pd.DataFrame, spec_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Adapt bridge outcomes to the exact canonical sample-weight helper."""
    assert_experiment_contract()
    if spec_name not in SPECIFICATIONS:
        raise RuntimeError(f"FATAL: unknown/extra specification: {spec_name}")
    adapter = train.copy()
    adapter[DEFAULT_TARGET_COLUMN] = adapter["target7"].astype(bool)
    if SPECIFICATIONS[spec_name]["tail_weighting"] == "LEGACY":
        raw = pd.to_numeric(adapter["raw_repair_return"], errors="raise")
        adapter["tail_weight"] = np.select(
            [raw.ge(0.12), raw.ge(0.10)], [2.0, 1.5], default=1.0
        ).astype(float)
    else:
        adapter["tail_weight"] = 1.0
    weights = build_training_sample_weight(adapter, positive_weight=POSITIVE_WEIGHT)
    return adapter, weights


def fit_spec(train: pd.DataFrame, spec_name: str) -> tuple[np.ndarray, np.ndarray]:
    if train.empty or train[list(REDUCED_FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("FATAL: invalid training population")
    adapter, weights = build_sample_weight(train, spec_name)
    beta = fit_logistic_l2_weighted(
        adapter[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float),
        adapter[DEFAULT_TARGET_COLUMN].astype(int).to_numpy(float),
        l2=float(SPECIFICATIONS[spec_name]["l2"]),
        sample_weight=weights,
    )
    return np.asarray(beta, dtype=float), weights


def _weighted_logloss(y: np.ndarray, logits: np.ndarray, weight: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    logits = np.asarray(logits, dtype=float)
    weight = np.asarray(weight, dtype=float)
    return float(np.sum(weight * (np.logaddexp(0.0, logits) - y * logits)) / weight.sum())


def _auc_between(frame: pd.DataFrame, positive: str, negative: str, score: str) -> float:
    selected = frame[frame["outcome_class"].isin([positive, negative])].copy()
    if selected.empty:
        return math.nan
    target = selected["outcome_class"].eq(positive).astype(int)
    return _binary_auc(target, selected[score])


def _within_date_auc(frame: pd.DataFrame, score_column: str = "model_score") -> tuple[float, int]:
    values: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        value = _binary_auc(day["target7"], day[score_column])
        if np.isfinite(value):
            values.append(float(value))
    return (float(np.mean(values)) if values else math.nan, len(values))


def _rank_scores(frame: pd.DataFrame, score_column: str = "model_score") -> pd.DataFrame:
    ranked: list[pd.DataFrame] = []
    grouping = [column for column in ("prediction_set", "spec", "signal_date") if column in frame]
    for _, day in frame.groupby(grouping, sort=True):
        part = day.sort_values(
            [score_column, "event_id"], ascending=[False, True], kind="mergesort"
        ).copy()
        part["model_rank"] = np.arange(1, len(part) + 1)
        ranked.append(part)
    if not ranked:
        return frame.assign(model_rank=pd.Series(dtype=int))
    return pd.concat(ranked, ignore_index=True).sort_values(
        grouping + ["model_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _outcome_class(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [frame["target7"].astype(bool), frame["loss"].astype(bool)],
            ["TARGET7", "LOSS"],
            default="PNT",
        ),
        index=frame.index,
    )


def build_full_development_fits(
    root: Path, frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    mature = frame[frame["label_available_date"].lt(AUGUST_ASOF)].copy()
    mature["outcome_class"] = _outcome_class(mature)
    if (len(mature), mature["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: mature development parity failed")
    scored_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    betas: dict[str, np.ndarray] = {}
    x = mature[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float)
    y = mature["target7"].to_numpy(float)
    for spec_name in SPEC_ORDER:
        beta, weight = fit_spec(mature, spec_name)
        betas[spec_name] = beta
        logits = beta[0] + x @ beta[1:]
        scores = _sigmoid(logits)
        weighted_rate = float(np.sum(weight * y) / weight.sum())
        constant_logit = math.log(weighted_rate / (1.0 - weighted_rate))
        constant_loss = _weighted_logloss(y, np.full(len(y), constant_logit), weight)
        fitted_loss = _weighted_logloss(y, logits, weight)
        scored = mature.copy()
        scored.insert(0, "spec", spec_name)
        scored.insert(0, "prediction_set", "FULL_DEVELOPMENT_IN_SAMPLE")
        scored["model_logit"] = logits
        scored["model_score"] = scores
        scored_frames.append(scored)
        scored_metric = scored.copy()
        fit_rows.append({
            "spec": spec_name,
            "training_rows": len(mature),
            "training_dates": mature["signal_date"].nunique(),
            "training_asof": AUGUST_ASOF,
            "feature_count": len(REDUCED_FEATURE_COLUMNS),
            "l2": float(SPECIFICATIONS[spec_name]["l2"]),
            "positive_weight": POSITIVE_WEIGHT,
            "tail_weighting": SPECIFICATIONS[spec_name]["tail_weighting"],
            "intercept": float(beta[0]),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_std": float(scores.std(ddof=0)),
            "weighted_constant_only_logloss": constant_loss,
            "fitted_weighted_logloss": fitted_loss,
            "logloss_improvement": constant_loss - fitted_loss,
            "coefficient_l2_norm": float(np.linalg.norm(beta[1:])),
            "auc_target7_vs_non_target": _binary_auc(scored_metric["target7"], scores),
            "within_date_target7_auc": _within_date_auc(scored_metric)[0],
            "within_date_auc_dates": _within_date_auc(scored_metric)[1],
            "auc_target7_vs_loss": _auc_between(scored_metric, "TARGET7", "LOSS", "model_score"),
            "auc_target7_vs_pnt": _auc_between(scored_metric, "TARGET7", "PNT", "model_score"),
            "auc_pnt_vs_loss": _auc_between(scored_metric, "PNT", "LOSS", "model_score"),
            "mean_score_target7": float(scored_metric.loc[scored_metric["outcome_class"].eq("TARGET7"), "model_score"].mean()),
            "mean_score_pnt": float(scored_metric.loc[scored_metric["outcome_class"].eq("PNT"), "model_score"].mean()),
            "mean_score_loss": float(scored_metric.loc[scored_metric["outcome_class"].eq("LOSS"), "model_score"].mean()),
            "mean_score_severe_loss": float(scored_metric.loc[scored_metric["severe_loss"].astype(bool), "model_score"].mean()),
            "score_mean_order": "PENDING",
            "legacy_tail_rows_1_5": int((pd.to_numeric(mature["raw_repair_return"]) >= .10).sum() - (pd.to_numeric(mature["raw_repair_return"]) >= .12).sum()) if spec_name == CONTROL_SPEC else 0,
            "legacy_tail_rows_2_0": int((pd.to_numeric(mature["raw_repair_return"]) >= .12).sum()) if spec_name == CONTROL_SPEC else 0,
        })
        fit_rows[-1]["score_mean_order"] = _score_mean_order(fit_rows[-1])
        common = {
            "stage": "FULL_DEVELOPMENT_IN_SAMPLE",
            "prediction_set": "FULL_DEVELOPMENT_IN_SAMPLE",
            "fold_id": "FULL_DEVELOPMENT_ASOF_2026_08_01",
            "test_date": "",
            "spec": spec_name,
            "l2": float(SPECIFICATIONS[spec_name]["l2"]),
            "positive_weight": POSITIVE_WEIGHT,
            "tail_weighting": SPECIFICATIONS[spec_name]["tail_weighting"],
            "train_rows": len(mature),
            "train_dates": mature["signal_date"].nunique(),
            "train_end_signal_date": str(mature["signal_date"].max()),
            "max_train_label_available_date": str(mature["label_available_date"].max()),
        }
        coefficient_rows.append({**common, "feature": "__INTERCEPT__", "coefficient": float(beta[0])})
        coefficient_rows.extend(
            {**common, "feature": feature, "coefficient": float(value)}
            for feature, value in zip(REDUCED_FEATURE_COLUMNS, beta[1:])
        )
    scored_all = _rank_scores(pd.concat(scored_frames, ignore_index=True))
    fit_table = pd.DataFrame(fit_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    _assert_s0_previous_parity(root, betas[CONTROL_SPEC])
    return scored_all, fit_table, coefficients, betas


def _score_mean_order(row: Mapping[str, Any]) -> str:
    t7, pnt, loss = (float(row[f"mean_score_{name}"]) for name in ("target7", "pnt", "loss"))
    if t7 > pnt > loss:
        return "TARGET7_GT_PNT_GT_LOSS"
    if t7 > loss > pnt:
        return "TARGET7_GT_LOSS_GT_PNT"
    if loss >= t7 > pnt:
        return "LOSS_GE_TARGET7_GT_PNT"
    return "MIXED"


def _assert_s0_previous_parity(root: Path, beta: np.ndarray) -> None:
    import json

    previous = json.loads((root / PREVIOUS_REDUCED_SPEC).read_text(encoding="utf-8"))
    archived = np.asarray(
        [previous["intercept"]] + [previous["coefficients"][f] for f in REDUCED_FEATURE_COLUMNS],
        dtype=float,
    )
    if not np.allclose(beta, archived, rtol=0, atol=1e-12):
        error = float(np.max(np.abs(beta - archived)))
        raise RuntimeError(f"FATAL: S0 does not reproduce frozen Reduced7F ({error})")


def _period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame[frame["signal_date"].ge(start) & frame["signal_date"].lt(end)].copy()


def _scope_filter(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "ALL_CANDIDATES":
        return frame
    if scope == "BOARD2_ONLY":
        return frame[frame["board_group"].eq("BOARD2")]
    if scope == "BOARD3_ONLY":
        return frame[frame["board_group"].eq("BOARD3")]
    raise ValueError(scope)


def _selection_daily(frame: pd.DataFrame, rank: int | None = None, topk: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("signal_date", sort=True):
        ordered = day.sort_values(["model_rank", "event_id"], kind="mergesort")
        if rank is not None:
            selected = ordered[ordered["model_rank"].eq(rank)]
        elif topk is not None:
            selected = ordered.head(min(topk, len(ordered)))
        else:
            selected = ordered
        if selected.empty:
            continue
        rows.append({
            "signal_date": date,
            "selected_rows": len(selected),
            "target7_rate": float(selected["target7"].mean()),
            "loss_rate": float(selected["loss"].mean()),
            "severe_loss_rate": float(selected["severe_loss"].mean()),
            "capped_return": float(selected["capped_return_7"].mean()),
        })
    return pd.DataFrame(rows)


def build_training_rankwise(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, (start, end) in FULL_PERIODS.items():
        for spec_name in SPEC_ORDER:
            all_part = _period_slice(scored[scored["spec"].eq(spec_name)], start, end)
            for scope in ("ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY"):
                part = _scope_filter(all_part, scope)
                if part.empty:
                    continue
                # Board-only outputs must rank the board subset internally.
                part = _rank_scores(part.drop(columns=["model_rank"], errors="ignore"))
                universe = _selection_daily(part)
                base = {
                    "period": period,
                    "spec": spec_name,
                    "population_scope": scope,
                    "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                    "dates": int(universe["signal_date"].nunique()),
                    "candidate_rows": len(part),
                    "candidate_count_per_day_mean": float(part.groupby("signal_date").size().mean()),
                    "candidate_count_per_day_min": int(part.groupby("signal_date").size().min()),
                    "candidate_count_per_day_max": int(part.groupby("signal_date").size().max()),
                    "universe_target7_rate": float(universe["target7_rate"].mean()),
                    "universe_loss_rate": float(universe["loss_rate"].mean()),
                    "universe_capped_return": float(universe["capped_return"].mean()),
                }
                selections = (
                    ("UNIVERSE", None, None),
                    ("RANK1", 1, None),
                    ("RANK2", 2, None),
                    ("RANK3", 3, None),
                    ("TOP3", None, 3),
                )
                for selection, rank, topk in selections:
                    daily = _selection_daily(part, rank=rank, topk=topk)
                    if daily.empty:
                        continue
                    rows.append({
                        **base,
                        "selection": selection,
                        "selection_dates": int(daily["signal_date"].nunique()),
                        "selected_rows": int(daily["selected_rows"].sum()),
                        "target7_rate": float(daily["target7_rate"].mean()),
                        "loss_rate": float(daily["loss_rate"].mean()),
                        "severe_loss_rate": float(daily["severe_loss_rate"].mean()),
                        "capped_return": float(daily["capped_return"].mean()),
                        "excess_vs_universe": float(daily["capped_return"].mean() - base["universe_capped_return"]),
                        "negative_date_rate": float((daily["capped_return"] < 0).mean()),
                        "worst_daily_capped_return": float(daily["capped_return"].min()),
                    })
    return pd.DataFrame(rows).sort_values(["period", "spec", "selection"], kind="mergesort").reset_index(drop=True)


def build_oracle_gap(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, (start, end) in FULL_PERIODS.items():
        for spec_name in SPEC_ORDER:
            part = _period_slice(scored[scored["spec"].eq(spec_name)], start, end)
            daily: list[dict[str, float]] = []
            for _, day in part.groupby("signal_date", sort=True):
                model = day.sort_values(["model_rank", "event_id"], kind="mergesort").head(min(3, len(day)))
                oracle = day.sort_values(
                    ["capped_return_7", "event_id"], ascending=[False, True], kind="mergesort"
                ).head(min(3, len(day)))
                daily.append({
                    "universe": float(day["capped_return_7"].mean()),
                    "actual": float(model["capped_return_7"].mean()),
                    "oracle": float(oracle["capped_return_7"].mean()),
                })
            values = pd.DataFrame(daily)
            universe = float(values["universe"].mean())
            actual = float(values["actual"].mean())
            oracle = float(values["oracle"].mean())
            available = oracle - universe
            recovered = actual - universe
            rows.append({
                "period": period,
                "spec": spec_name,
                "dates": len(values),
                "universe_capped_return": universe,
                "actual_top3_capped_return": actual,
                "oracle_top3_capped_return": oracle,
                "available_oracle_gap": available,
                "model_recovered_gap": recovered,
                "recovery_ratio": recovered / available if abs(available) > 1e-15 else math.nan,
            })
    return pd.DataFrame(rows).sort_values(["period", "spec"], kind="mergesort").reset_index(drop=True)


def _fit_score_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prediction_set: str,
    fold_id: str,
    test_date: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    if train.empty or test.empty:
        raise RuntimeError(f"FATAL: empty temporal fold {fold_id}")
    if not train["signal_date"].lt(test_date).all():
        raise RuntimeError(f"FATAL: future signal leakage in {fold_id}")
    if not train["label_available_date"].lt(test_date).all():
        raise RuntimeError(f"FATAL: label leakage in {fold_id}")
    outputs: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []
    for spec_name in SPEC_ORDER:
        beta, _ = fit_spec(train, spec_name)
        scored = test[[
            "event_id", "signal_date", "code", "board_group", "candidate_count",
            "label_available_date", "target7", "positive_non_target", "loss",
            "severe_loss", "raw_repair_return", "capped_return_7", "outcome_class",
        ]].copy()
        scored.insert(0, "fold_id", fold_id)
        scored.insert(0, "spec", spec_name)
        scored.insert(0, "prediction_set", prediction_set)
        scored["model_logit"] = beta[0] + test[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float) @ beta[1:]
        scored["model_score"] = _sigmoid(scored["model_logit"].to_numpy(float))
        scored["train_rows"] = len(train)
        scored["train_dates"] = train["signal_date"].nunique()
        scored["train_end_signal_date"] = str(train["signal_date"].max())
        scored["max_train_label_available_date"] = str(train["label_available_date"].max())
        outputs.append(scored)
        common = {
            "stage": "TEMPORAL_VALIDATION",
            "prediction_set": prediction_set,
            "fold_id": fold_id,
            "test_date": test_date,
            "spec": spec_name,
            "l2": float(SPECIFICATIONS[spec_name]["l2"]),
            "positive_weight": POSITIVE_WEIGHT,
            "tail_weighting": SPECIFICATIONS[spec_name]["tail_weighting"],
            "train_rows": len(train),
            "train_dates": train["signal_date"].nunique(),
            "train_end_signal_date": str(train["signal_date"].max()),
            "max_train_label_available_date": str(train["label_available_date"].max()),
        }
        coefficients.append({**common, "feature": "__INTERCEPT__", "coefficient": float(beta[0])})
        coefficients.extend(
            {**common, "feature": feature, "coefficient": float(value)}
            for feature, value in zip(REDUCED_FEATURE_COLUMNS, beta[1:])
        )
    return outputs, coefficients


def build_temporal_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    mature = frame[frame["label_available_date"].lt(AUGUST_ASOF)].copy()
    mature["outcome_class"] = _outcome_class(mature)
    predictions: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []

    # Fold A: each June date uses only May history matured strictly before it.
    june_dates = sorted(mature.loc[mature["signal_date"].between("2026-06-01", "2026-06-30"), "signal_date"].unique())
    fold_a_dates: list[str] = []
    for test_date in june_dates:
        train = mature[
            mature["signal_date"].lt("2026-06-01")
            & mature["label_available_date"].lt(test_date)
        ].copy()
        if train["signal_date"].nunique() < MIN_TRAIN_SIGNAL_DATES:
            continue
        test = mature[mature["signal_date"].eq(test_date)].copy()
        out, coef = _fit_score_fold(
            train, test, "FOLD_A_MAY_TO_JUNE", f"FOLD_A__{test_date}", test_date
        )
        predictions.extend(out)
        coefficients.extend(coef)
        fold_a_dates.append(test_date)

    # Fold B: one fixed May+June snapshot as of the first mature July test date.
    july_dates = sorted(mature.loc[mature["signal_date"].between("2026-07-01", "2026-07-31"), "signal_date"].unique())
    if not july_dates:
        raise RuntimeError("FATAL: no mature July validation dates")
    first_july = july_dates[0]
    fold_b_train = mature[
        mature["signal_date"].lt("2026-07-01")
        & mature["label_available_date"].lt(first_july)
    ].copy()
    fold_b_test = mature[mature["signal_date"].isin(july_dates)].copy()
    out, coef = _fit_score_fold(
        fold_b_train, fold_b_test, "FOLD_B_MAY_JUNE_TO_JULY",
        "FOLD_B__FIXED_ASOF_2026_07_01", first_july,
    )
    predictions.extend(out)
    coefficients.extend(coef)

    # Auxiliary chronological expanding OOF, kept separate from fixed folds.
    expanding_dates: list[str] = []
    for test_date in sorted(mature.loc[mature["signal_date"].ge("2026-06-01"), "signal_date"].unique()):
        train = mature[
            mature["signal_date"].lt(test_date)
            & mature["label_available_date"].lt(test_date)
        ].copy()
        if train["signal_date"].nunique() < MIN_TRAIN_SIGNAL_DATES:
            continue
        test = mature[mature["signal_date"].eq(test_date)].copy()
        out, coef = _fit_score_fold(
            train, test, "EXPANDING_OOF", f"EXPANDING__{test_date}", test_date
        )
        predictions.extend(out)
        coefficients.extend(coef)
        expanding_dates.append(test_date)

    result = _rank_scores(pd.concat(predictions, ignore_index=True))
    coefficient_table = pd.DataFrame(coefficients)
    prediction_dates = pd.to_datetime(result["signal_date"], errors="raise")
    train_end = pd.to_datetime(result["train_end_signal_date"], errors="raise")
    max_label = pd.to_datetime(result["max_train_label_available_date"], errors="raise")
    audit = {
        "self_or_future_signal_leakage_rows": int((train_end >= prediction_dates).sum()),
        "current_or_future_label_leakage_rows": int((max_label >= prediction_dates).sum()),
        "august_signal_date_rows_accessed": int(result["signal_date"].ge(AUGUST_ASOF).sum()),
        "fold_a_dates": len(fold_a_dates),
        "fold_b_dates": len(july_dates),
        "expanding_dates": len(expanding_dates),
    }
    if any(audit[key] for key in (
        "self_or_future_signal_leakage_rows",
        "current_or_future_label_leakage_rows",
        "august_signal_date_rows_accessed",
    )):
        raise RuntimeError(f"FATAL: temporal leakage audit failed: {audit}")
    return result, coefficient_table, audit


def _prediction_period(predictions: pd.DataFrame, period: str) -> pd.DataFrame:
    if period in ("FOLD_A_MAY_TO_JUNE", "FOLD_B_MAY_JUNE_TO_JULY"):
        return predictions[predictions["prediction_set"].eq(period)]
    expanding = predictions[predictions["prediction_set"].eq("EXPANDING_OOF")]
    if period == "EXPANDING_JUNE":
        return expanding[expanding["signal_date"].str.startswith("2026-06")]
    if period == "EXPANDING_JULY_MATURE_ONLY":
        return expanding[expanding["signal_date"].str.startswith("2026-07")]
    if period == "EXPANDING_ALL_MATURE":
        return expanding
    raise ValueError(period)


def _daily_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (spec_name, date), day in frame.groupby(["spec", "signal_date"], sort=True):
        ordered = day.sort_values(["model_rank", "event_id"], kind="mergesort")
        universe = float(ordered["capped_return_7"].mean())
        row: dict[str, Any] = {
            "spec": spec_name,
            "signal_date": date,
            "candidate_count": len(ordered),
            "universe_capped_return": universe,
            "universe_target7_rate": float(ordered["target7"].mean()),
            "universe_loss_rate": float(ordered["loss"].mean()),
            "within_date_target7_auc": _binary_auc(ordered["target7"], ordered["model_score"]),
        }
        for rank in (1, 2, 3):
            selected = ordered[ordered["model_rank"].eq(rank)]
            row[f"rank{rank}_capped_return"] = float(selected["capped_return_7"].mean()) if not selected.empty else math.nan
            row[f"rank{rank}_target7_rate"] = float(selected["target7"].mean()) if not selected.empty else math.nan
            row[f"rank{rank}_loss_rate"] = float(selected["loss"].mean()) if not selected.empty else math.nan
        top3 = ordered.head(min(3, len(ordered)))
        row.update({
            "top3_selected_rows": len(top3),
            "top3_capped_return": float(top3["capped_return_7"].mean()),
            "top3_target7_rate": float(top3["target7"].mean()),
            "top3_loss_rate": float(top3["loss"].mean()),
            "top3_severe_loss_rate": float(top3["severe_loss"].mean()),
            "top3_excess_vs_universe": float(top3["capped_return_7"].mean() - universe),
            "top3_negative_date": int(float(top3["capped_return_7"].mean()) < 0),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_temporal_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in TEMPORAL_PERIOD_ORDER:
        part = _prediction_period(predictions, period)
        for spec_name in SPEC_ORDER:
            all_scored = part[part["spec"].eq(spec_name)].copy()
            for scope in ("ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY"):
                scored = _scope_filter(all_scored, scope)
                if scored.empty:
                    continue
                scored = _rank_scores(scored.drop(columns=["model_rank"], errors="ignore"))
                daily = _daily_temporal(scored)
                within, eligible = _within_date_auc(scored)
                row: dict[str, Any] = {
                    "period": period,
                    "spec": spec_name,
                    "population_scope": scope,
                    "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                    "dates": int(scored["signal_date"].nunique()),
                    "candidate_rows": len(scored),
                    "within_date_target7_auc": within,
                    "eligible_auc_dates": eligible,
                    "pooled_target7_auc": _binary_auc(scored["target7"], scored["model_score"]),
                    "target7_vs_loss_auc": _auc_between(scored, "TARGET7", "LOSS", "model_score"),
                    "universe_capped_return": float(daily["universe_capped_return"].mean()),
                    "universe_target7_rate": float(daily["universe_target7_rate"].mean()),
                    "universe_loss_rate": float(daily["universe_loss_rate"].mean()),
                }
                for rank in (1, 2, 3):
                    row[f"rank{rank}_capped_return"] = float(daily[f"rank{rank}_capped_return"].mean())
                    row[f"rank{rank}_target7_rate"] = float(daily[f"rank{rank}_target7_rate"].mean())
                    row[f"rank{rank}_loss_rate"] = float(daily[f"rank{rank}_loss_rate"].mean())
                row.update({
                    "top3_capped_return": float(daily["top3_capped_return"].mean()),
                    "top3_target7_rate": float(daily["top3_target7_rate"].mean()),
                    "top3_loss_rate": float(daily["top3_loss_rate"].mean()),
                    "top3_severe_loss_rate": float(daily["top3_severe_loss_rate"].mean()),
                    "top3_excess_vs_universe": float(daily["top3_excess_vs_universe"].mean()),
                    "top3_negative_date_rate": float(daily["top3_negative_date"].mean()),
                    "worst_daily_top3_capped_return": float(daily["top3_capped_return"].min()),
                })
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["period", "population_scope", "spec"], kind="mergesort"
    ).reset_index(drop=True)


def build_paired_daily(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_index, period in enumerate(TEMPORAL_PERIOD_ORDER):
        daily = _daily_temporal(_prediction_period(predictions, period))
        pivot = daily.pivot(index="signal_date", columns="spec", values="top3_capped_return")
        if CONTROL_SPEC not in pivot:
            continue
        for spec_index, challenger in enumerate(CHALLENGER_SPECS):
            common = pivot[[CONTROL_SPEC, challenger]].dropna()
            delta = common[challenger] - common[CONTROL_SPEC]
            for date, value in delta.items():
                rows.append({
                    "row_type": "DAILY",
                    "period": period,
                    "challenger": challenger,
                    "control": CONTROL_SPEC,
                    "signal_date": date,
                    "control_top3_capped": float(common.loc[date, CONTROL_SPEC]),
                    "challenger_top3_capped": float(common.loc[date, challenger]),
                    "delta": float(value),
                    "dates": math.nan,
                    "mean_delta": math.nan,
                    "median_delta": math.nan,
                    "positive_days": math.nan,
                    "negative_days": math.nan,
                    "zero_days": math.nan,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_p2_5": math.nan,
                    "bootstrap_p50": math.nan,
                    "bootstrap_p97_5": math.nan,
                    "bootstrap_probability_delta_gt_0": math.nan,
                })
            values = delta.to_numpy(float)
            rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, period_index, spec_index]))
            draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
            boot = values[draws].mean(axis=1)
            rows.append({
                "row_type": "SUMMARY",
                "period": period,
                "challenger": challenger,
                "control": CONTROL_SPEC,
                "signal_date": "",
                "control_top3_capped": float(common[CONTROL_SPEC].mean()),
                "challenger_top3_capped": float(common[challenger].mean()),
                "delta": float(values.mean()),
                "dates": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "positive_days": int((values > 0).sum()),
                "negative_days": int((values < 0).sum()),
                "zero_days": int((values == 0).sum()),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_p2_5": float(np.quantile(boot, .025)),
                "bootstrap_p50": float(np.quantile(boot, .50)),
                "bootstrap_p97_5": float(np.quantile(boot, .975)),
                "bootstrap_probability_delta_gt_0": float((boot > 0).mean()),
            })
    return pd.DataFrame(rows)


def analyze(root: str | Path) -> dict[str, Any]:
    assert_experiment_contract()
    root_path = Path(root).resolve()
    frame, bridge_spec = load_bridge(root_path)
    full_scored, training_fit, full_coefficients, betas = build_full_development_fits(root_path, frame)
    rankwise = build_training_rankwise(full_scored)
    oracle = build_oracle_gap(full_scored)
    temporal_predictions, temporal_coefficients, leakage = build_temporal_predictions(frame)
    temporal_summary = build_temporal_summary(temporal_predictions)
    paired = build_paired_daily(temporal_predictions)
    coefficients = pd.concat([full_coefficients, temporal_coefficients], ignore_index=True)
    return {
        "root": root_path,
        "frame": frame,
        "bridge_spec": bridge_spec,
        "full_scored": full_scored,
        "training_fit": training_fit,
        "rankwise": rankwise,
        "oracle": oracle,
        "temporal_predictions": temporal_predictions,
        "temporal_summary": temporal_summary,
        "coefficients": coefficients,
        "paired": paired,
        "leakage": leakage,
        "betas": betas,
    }


def render_review(
    answers: Mapping[str, str], training_spec_state: str, next_action: str,
) -> str:
    mapping = {
        "TRAINING_SPEC_REPAIR_SUPPORTED": "FREEZE_ONE_REPAIRED_7F_SPEC",
        "TRAINING_SPEC_REPAIR_NOT_SUPPORTED": "STOP_PARAMETER_REPAIR_AND_REASSESS_INFORMATION",
        "MIXED": "STOP_AND_REVIEW",
    }
    if training_spec_state not in mapping or next_action != mapping[training_spec_state]:
        raise RuntimeError("FATAL: invalid decision/action mapping")
    if set(answers) != {f"Q{i}" for i in range(1, 9)}:
        raise RuntimeError("FATAL: review must answer exactly Q1-Q8")
    questions = (
        "Current S0 是否存在明显训练内欠拟合？",
        "Tail bonus 是否与‘>=7%即强修复’业务定义存在不一致？",
        "关闭 tail bonus 后实际结果是否改善？",
        "L2=.30 是否过强？",
        "S2/S3 是否恢复 Target7 vs LOSS separation？",
        "Rank1/Rank2/Rank3 的问题有没有改善？",
        "May→June 与 May+June→July 是否支持同一个训练规格？",
        "当前问题主要属于 TRAINING_SPEC / INFORMATION_LIMIT / MIXED？",
    )
    lines = ["# v004c Reduced7F Training Spec Sanity v001", ""]
    for index, question in enumerate(questions, 1):
        lines.extend([f"## Q{index}. {question}", "", answers[f"Q{index}"], ""])
    lines.extend([
        f"TRAINING_SPEC_STATE = {training_spec_state}",
        "",
        f"NEXT_ACTION = {next_action}",
        "",
    ])
    return "\n".join(lines)


def build_outputs(
    context: Mapping[str, Any], answers: Mapping[str, str],
    training_spec_state: str, next_action: str,
) -> dict[str, bytes]:
    review = render_review(answers, training_spec_state, next_action)
    return {
        OUTPUT_FILENAMES[0]: _csv_bytes(context["training_fit"]),
        OUTPUT_FILENAMES[1]: _csv_bytes(context["rankwise"]),
        OUTPUT_FILENAMES[2]: _csv_bytes(context["oracle"]),
        OUTPUT_FILENAMES[3]: _csv_bytes(context["temporal_predictions"]),
        OUTPUT_FILENAMES[4]: _csv_bytes(context["temporal_summary"]),
        OUTPUT_FILENAMES[5]: _csv_bytes(context["coefficients"]),
        OUTPUT_FILENAMES[6]: _csv_bytes(context["paired"]),
        OUTPUT_FILENAMES[7]: review.encode("utf-8"),
    }


def write_outputs(
    context: Mapping[str, Any], answers: Mapping[str, str],
    training_spec_state: str, next_action: str,
    output_dir: str | Path | None = None,
) -> Path:
    target = Path(output_dir) if output_dir is not None else (
        context["root"] / "reports" / "research" / OUTPUT_DIRNAME
    )
    first = build_outputs(context, answers, training_spec_state, next_action)
    second = build_outputs(context, answers, training_spec_state, next_action)
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

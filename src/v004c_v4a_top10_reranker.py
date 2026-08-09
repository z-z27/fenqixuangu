from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004a import (
    DEFAULT_TARGET_COLUMN,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
)
from .v004c_exact_v4a_residual import load_exact_transfer_oof
from .v004c_mechanism_foundation import load_outcomes
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    INITIAL_TRAIN_DATES,
    JULY_SENTINEL_DATE,
    L2 as STAGE1_L2,
    MODEL_ID as STAGE1_MODEL_ID,
    POSITIVE_WEIGHT as STAGE1_POSITIVE_WEIGHT,
    _sigmoid,
    build_exact_raw_features,
    dataframe_csv_bytes,
    exact_tie_rows,
    load_authoritative_input,
    prepare_transfer_samples,
    run_expanding_forward,
)


MODEL_ID = "V4A_TOP10_CLOSING_COMPLETION_RERANKER_V4C"
STAGE2_L2 = 0.30
STAGE2_POSITIVE_WEIGHT = 1.0
STAGE2_TAIL_WEIGHTING = False
TOP_K = 10
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_RESAMPLES = 20_000
HYPERPARAMETER_SEARCH = False
FEATURE_SELECTION = False
NEW_RAW_FEATURES = False
EXTERNAL_FACTORS = False
MODEL_ZOO = False

RESIDUAL_RAW_FIELDS = [
    "d1_high_to_close_drawdown_raw",
    "d1_close_location",
    "d1_low_to_close_recovery",
    "d1_open_to_close_return_raw",
]
RESIDUAL_RANK_FIELDS = [
    "rank_drawdown_top10",
    "rank_close_location_top10",
    "rank_recovery_top10",
    "rank_open_close_top10",
]
STAGE2_FEATURE_COLUMNS = [
    "stage1_strength",
    "closing_completion_gap",
    "strength_x_gap",
]

OUTPUT_FILENAMES = (
    "v004c_v4a_top10_reranker_oof_v001.csv",
    "v004c_v4a_top10_reranker_fold_audit_v001.csv",
    "v004c_v4a_top10_reranker_coefficients_v001.csv",
    "v004c_v4a_top10_reranker_membership_v001.csv",
    "v004c_v4a_top10_reranker_practical_v001.csv",
    "v004c_v4a_top10_reranker_review_v001.md",
)


def assert_model_contract() -> None:
    if len(FROZEN_FEATURE_COLUMNS) != 18:
        raise RuntimeError("FATAL: frozen Stage1 must retain 18 predictors")
    if STAGE1_L2 != 0.30 or STAGE1_POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen Stage1 hyperparameters changed")
    if INITIAL_TRAIN_DATES != 18:
        raise RuntimeError("FATAL: Stage1 warmup changed")
    if RESIDUAL_RAW_FIELDS != [
        "d1_high_to_close_drawdown_raw",
        "d1_close_location",
        "d1_low_to_close_recovery",
        "d1_open_to_close_return_raw",
    ]:
        raise RuntimeError("FATAL: Stage2 raw completion inputs changed")
    if STAGE2_FEATURE_COLUMNS != [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]:
        raise RuntimeError("FATAL: Stage2 must contain exactly three predictors")
    if STAGE2_L2 != 0.30 or STAGE2_POSITIVE_WEIGHT != 1.0:
        raise RuntimeError("FATAL: Stage2 hyperparameters changed")
    if STAGE2_TAIL_WEIGHTING:
        raise RuntimeError("FATAL: Stage2 tail weighting is forbidden")
    if HYPERPARAMETER_SEARCH or FEATURE_SELECTION or NEW_RAW_FEATURES:
        raise RuntimeError("FATAL: Stage2 search/selection/new raw features are forbidden")


def prepare_reranker_samples(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assert_model_contract()
    base = load_authoritative_input(root)
    raw, _, _, feature_gate = build_exact_raw_features(root, base)
    if not feature_gate["pass"]:
        raise RuntimeError("FATAL: exact frozen v4a feature gate failed")
    outcomes = load_outcomes(root, base)
    samples = prepare_transfer_samples(raw, outcomes)
    authority = base[[
        "event_id", "signal_date", "code", *RESIDUAL_RAW_FIELDS
    ]].copy()
    for feature in RESIDUAL_RAW_FIELDS:
        authority[feature] = pd.to_numeric(authority[feature], errors="coerce")
    joined = samples.merge(
        authority,
        on="event_id",
        how="left",
        suffixes=("", "_authority"),
        validate="one_to_one",
    )
    code_mismatch = joined["code"].astype(str).str.zfill(6).ne(
        joined["code_authority"].astype(str).str.zfill(6)
    )
    date_mismatch = joined["signal_date"].astype(str).ne(
        joined["signal_date_authority"].astype(str)
    )
    if bool(code_mismatch.any() or date_mismatch.any()) or len(joined) != 319:
        raise RuntimeError("FATAL: Stage2 authoritative join mismatch")
    joined = joined.drop(columns=["code_authority", "signal_date_authority"])
    if bool(joined["signal_date"].astype(str).ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July sentinel triggered")
    return (
        joined.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True),
        base,
        feature_gate,
    )


def _fit_stage1(train: pd.DataFrame) -> np.ndarray:
    weights = build_training_sample_weight(
        train, positive_weight=STAGE1_POSITIVE_WEIGHT
    )
    return fit_logistic_l2_weighted(
        train[FROZEN_FEATURE_COLUMNS].to_numpy(float),
        train[DEFAULT_TARGET_COLUMN].astype(bool).astype(float).to_numpy(),
        l2=STAGE1_L2,
        sample_weight=weights,
    )


def score_stage1_date(
    samples: pd.DataFrame,
    base_training_dates: Sequence[str],
    score_date: str,
    current_test_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    training_dates = sorted(str(date) for date in base_training_dates)
    if score_date in training_dates:
        raise RuntimeError("FATAL: self-label Stage1 training date detected")
    if not training_dates or max(training_dates) >= current_test_date:
        raise RuntimeError("FATAL: current test date leaked into crossfit history")
    train = samples[samples["signal_date"].isin(training_dates)].copy()
    scored = samples[samples["signal_date"].eq(score_date)].copy()
    if train.empty or scored.empty:
        raise RuntimeError("FATAL: Stage1 crossfit train/score rows unavailable")
    beta = _fit_stage1(train)
    score = _sigmoid(
        beta[0] + scored[FROZEN_FEATURE_COLUMNS].to_numpy(float) @ beta[1:]
    )
    scored["stage1_score"] = score
    scored = scored.sort_values(
        ["stage1_score", "event_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scored["stage1_rank"] = np.arange(1, len(scored) + 1)
    scored["candidate_count"] = len(scored)
    scored["stage1_top10"] = scored["stage1_rank"].le(min(TOP_K, len(scored)))
    scored["stage1_strength"] = 1.0 - (
        (scored["stage1_rank"] - 1)
        / max(int(scored["candidate_count"].iloc[0]) - 1, 1)
    )
    audit = {
        "score_date": score_date,
        "base_train_dates": len(training_dates),
        "self_label_leakage_rows": int(
            len(scored) if score_date in training_dates else 0
        ),
        "current_test_date_leakage": int(
            len(scored) if max(training_dates) >= current_test_date else 0
        ),
    }
    return scored, audit


def build_stage2_features(stage1_top10: pd.DataFrame) -> pd.DataFrame:
    if stage1_top10.empty:
        raise RuntimeError("FATAL: empty Stage1 Top10")
    result = stage1_top10.copy()
    if result["signal_date"].nunique() != 1:
        raise RuntimeError("FATAL: Stage2 rank transform must be single-date")
    rank_specs = list(zip(RESIDUAL_RAW_FIELDS, RESIDUAL_RANK_FIELDS))
    for raw, ranked in rank_specs:
        # Reuse frozen-v4a's neutral percentile-rank fallback for structurally
        # undefined values (one zero-range D1 close-location row in authority).
        result[ranked] = result[raw].rank(method="average", pct=True).fillna(0.5)
    result["closing_completion_gap"] = (
        result["rank_drawdown_top10"]
        + (1.0 - result["rank_close_location_top10"])
        + (1.0 - result["rank_recovery_top10"])
        + (1.0 - result["rank_open_close_top10"])
    ) / 4.0
    result["strength_x_gap"] = (
        result["stage1_strength"] * result["closing_completion_gap"]
    )
    if result[STAGE2_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("FATAL: Stage2 predictor missing")
    return result


def _format_frozen(value: float) -> str:
    return format(float(value), ".12g")


def validate_stage1_control_parity(
    rebuilt: pd.DataFrame, frozen: pd.DataFrame
) -> dict[str, Any]:
    left = frozen.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)
    right = rebuilt.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)
    identity = left[["event_id", "signal_date", "code"]].equals(
        right[["event_id", "signal_date", "code"]]
    )
    score = left["model_score"].map(_format_frozen).tolist() == right[
        "model_score"
    ].map(_format_frozen).tolist()
    rank = left["model_rank"].astype(int).tolist() == right["model_rank"].astype(int).tolist()
    membership = left["model_rank"].le(3).tolist() == right["model_rank"].le(3).tolist()
    passed = bool(identity and score and rank and membership)
    result = {
        "identity_exact": identity,
        "score_exact_at_frozen_precision": score,
        "rank_exact": rank,
        "top3_membership_exact": membership,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(f"FATAL: Stage1 control parity failed: {result}")
    return result


def _stage2_date_weights(meta: pd.DataFrame) -> np.ndarray:
    counts = meta.groupby("signal_date", sort=True)["event_id"].transform("size")
    return (1.0 / counts.to_numpy(float)).astype(float)


def _coefficient_cosine(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return np.nan
    denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
    return float(previous @ current / denominator) if denominator > 0.0 else np.nan


def run_nested_reranker(
    samples: pd.DataFrame,
    control_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(samples["signal_date"].astype(str).unique())
    if len(dates) != 39 or not all(date.startswith("2026-05") for date in dates[:18]):
        raise RuntimeError("FATAL: May-18/June-21 split unavailable")
    if not all(date.startswith("2026-06") for date in dates[18:]):
        raise RuntimeError("FATAL: Stage2 test scope must be June")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    previous_coefficient: np.ndarray | None = None

    for fold_index in range(INITIAL_TRAIN_DATES, len(dates)):
        test_date = dates[fold_index]
        history_dates = dates[:fold_index]
        if max(history_dates) >= test_date:
            raise RuntimeError("FATAL: current test date in Stage2 history")
        meta_parts: list[pd.DataFrame] = []
        self_leakage_rows = 0
        test_date_leakage_rows = 0
        base_train_counts: list[int] = []
        for meta_date in history_dates:
            base_dates = [date for date in history_dates if date != meta_date]
            scored, audit = score_stage1_date(
                samples, base_dates, meta_date, test_date
            )
            self_leakage_rows += int(audit["self_label_leakage_rows"])
            test_date_leakage_rows += int(audit["current_test_date_leakage"])
            base_train_counts.append(int(audit["base_train_dates"]))
            meta_parts.append(build_stage2_features(scored[scored["stage1_top10"]].copy()))
        meta = pd.concat(meta_parts, ignore_index=True)
        if meta["signal_date"].nunique() != len(history_dates):
            raise RuntimeError("FATAL: crossfit meta date coverage mismatch")
        if self_leakage_rows or test_date_leakage_rows:
            raise RuntimeError("FATAL: nested crossfit leakage audit failed")
        stage2_weight = _stage2_date_weights(meta)
        beta = fit_logistic_l2_weighted(
            meta[STAGE2_FEATURE_COLUMNS].to_numpy(float),
            meta["target7"].astype(float).to_numpy(),
            l2=STAGE2_L2,
            sample_weight=stage2_weight,
        )

        test_samples = samples[samples["signal_date"].eq(test_date)].copy()
        test_control = control_oof[control_oof["signal_date"].eq(test_date)][
            ["event_id", "model_score", "model_rank"]
        ].rename(columns={
            "model_score": "stage1_score",
            "model_rank": "stage1_rank",
        })
        test_full = test_samples.merge(
            test_control, on="event_id", how="left", validate="one_to_one"
        )
        if test_full[["stage1_score", "stage1_rank"]].isna().any().any():
            raise RuntimeError("FATAL: Stage1 test score missing")
        test_full["stage1_rank"] = test_full["stage1_rank"].astype(int)
        test_full["candidate_count"] = len(test_full)
        test_full["stage1_top10"] = test_full["stage1_rank"].le(
            min(TOP_K, len(test_full))
        )
        test_full["stage1_strength"] = 1.0 - (
            (test_full["stage1_rank"] - 1) / max(len(test_full) - 1, 1)
        )
        test_top10 = build_stage2_features(
            test_full[test_full["stage1_top10"]].copy()
        )
        coefficient = np.asarray(beta[1:], dtype=float)
        stage2_score = _sigmoid(
            beta[0] + test_top10[STAGE2_FEATURE_COLUMNS].to_numpy(float) @ coefficient
        )
        test_top10["stage2_score"] = stage2_score
        test_top10 = test_top10.sort_values(
            ["stage2_score", "stage1_score", "event_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        test_top10["stage2_rank_within_top10"] = np.arange(1, len(test_top10) + 1)

        stage2_columns = [
            "event_id",
            *RESIDUAL_RANK_FIELDS,
            "closing_completion_gap",
            "strength_x_gap",
            "stage2_score",
            "stage2_rank_within_top10",
        ]
        predicted = test_full.merge(
            test_top10[stage2_columns], on="event_id", how="left", validate="one_to_one"
        )
        predicted["final_rerank_rank"] = predicted["stage1_rank"]
        selected_mask = predicted["stage1_top10"]
        predicted.loc[selected_mask, "final_rerank_rank"] = predicted.loc[
            selected_mask, "stage2_rank_within_top10"
        ]
        predicted["final_rerank_rank"] = predicted["final_rerank_rank"].astype(int)
        predicted["control_top3"] = predicted["stage1_rank"].le(3)
        predicted["reranker_top3"] = predicted["final_rerank_rank"].le(3)
        predictions.append(predicted)

        fold_rows.append({
            "test_date": test_date,
            "history_date_count": len(history_dates),
            "meta_train_rows": len(meta),
            "meta_train_dates": meta["signal_date"].nunique(),
            "meta_target7_rate": float(meta["target7"].mean()),
            "crossfit_stage1_models_fitted": len(history_dates),
            "min_base_train_dates": min(base_train_counts),
            "max_base_train_dates": max(base_train_counts),
            "stage2_weight_sum": float(stage2_weight.sum()),
            "stage2_weight_date_sum_min": float(
                pd.Series(stage2_weight, index=meta.index).groupby(meta["signal_date"].to_numpy()).sum().min()
            ),
            "stage2_weight_date_sum_max": float(
                pd.Series(stage2_weight, index=meta.index).groupby(meta["signal_date"].to_numpy()).sum().max()
            ),
            "self_label_leakage_rows": self_leakage_rows,
            "current_test_date_leakage_rows": test_date_leakage_rows,
            "stage2_score_min": float(stage2_score.min()),
            "stage2_score_max": float(stage2_score.max()),
            "stage2_score_std": float(stage2_score.std(ddof=0)),
            "stage2_exact_tie_rows": exact_tie_rows(pd.Series(stage2_score)),
            "stage1_top10_rows": len(test_top10),
            "stage2_predictor_count": len(STAGE2_FEATURE_COLUMNS),
            "stage2_l2": STAGE2_L2,
            "stage2_positive_weight": STAGE2_POSITIVE_WEIGHT,
            "stage2_tail_weighting": "NONE",
        })
        coefficient_rows.append({
            "test_date": test_date,
            "intercept": float(beta[0]),
            "beta_stage1_strength": float(coefficient[0]),
            "beta_closing_completion_gap": float(coefficient[1]),
            "beta_strength_x_gap": float(coefficient[2]),
            "adjacent_coefficient_cosine": _coefficient_cosine(
                previous_coefficient, coefficient
            ),
            "purpose": "AUDIT_ONLY",
        })
        previous_coefficient = coefficient

    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(oof) != 173 or oof["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: reranker OOF identity changed")
    return oof, pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def oof_output(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id", "signal_date", "code", "board_streak_before_break",
        "candidate_count", "stage1_score", "stage1_rank", "stage1_top10",
        "stage1_strength", *RESIDUAL_RAW_FIELDS, *RESIDUAL_RANK_FIELDS,
        "closing_completion_gap", "strength_x_gap", "stage2_score",
        "stage2_rank_within_top10", "final_rerank_rank", "control_top3",
        "reranker_top3", "target7", "raw_repair_return",
        "capped_opportunity_return_7",
    ]
    result = frame[columns].rename(
        columns={"board_streak_before_break": "board"}
    )
    return result.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _selected_daily(
    frame: pd.DataFrame, rank_column: str, kind: str, k: int, strict: bool = True
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("signal_date", sort=True):
        if strict and len(day) < k:
            continue
        if kind == "rank":
            if len(day) < k:
                continue
            selected = day[day[rank_column].eq(k)]
        else:
            selected = day[day[rank_column].le(min(k, len(day)))]
        selected_return = float(selected["capped_opportunity_return_7"].mean())
        baseline = float(day["capped_opportunity_return_7"].mean())
        rows.append({
            "signal_date": date,
            "selected_return": selected_return,
            "baseline": baseline,
            "target7_precision": float(selected["target7"].mean()),
            "positive": float(selected_return > 0.0),
            "negative": float(selected_return < 0.0),
            "beat": float(selected_return > baseline),
            "selected_codes": "|".join(selected.sort_values(rank_column)["code"].astype(str)),
        })
    return pd.DataFrame(rows)


def _winner_capture(frame: pd.DataFrame, rank_column: str, k: int = 3) -> float:
    values: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        if len(day) < k:
            continue
        target_count = int(day["target7"].sum())
        if target_count <= 0:
            continue
        selected = day[day[rank_column].le(k)]
        values.append(float(selected["target7"].sum() / min(k, target_count)))
    return float(np.mean(values)) if values else np.nan


def strategy_summary(frame: pd.DataFrame, rank_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date_count": int(frame["signal_date"].nunique()),
        "universe_target7_rate": float(
            frame.groupby("signal_date", sort=True)["target7"].mean().mean()
        ),
        "winner_capture": _winner_capture(frame, rank_column, 3),
    }
    specs = (
        ("Rank1", "rank", 1), ("Rank2", "rank", 2), ("Rank3", "rank", 3),
        ("Top1", "top", 1), ("Top2", "top", 2), ("Top3", "top", 3),
    )
    for label, kind, k in specs:
        daily = _selected_daily(frame, rank_column, kind, k, strict=True)
        result[f"{label}_mean"] = float(daily["selected_return"].mean())
        result[f"{label}_median"] = float(daily["selected_return"].median())
        result[f"{label}_p25"] = float(daily["selected_return"].quantile(0.25))
        result[f"{label}_worst"] = float(daily["selected_return"].min())
        result[f"{label}_baseline"] = float(daily["baseline"].mean())
        result[f"{label}_excess"] = result[f"{label}_mean"] - result[f"{label}_baseline"]
        result[f"{label}_positive_date_rate"] = float(daily["positive"].mean())
        result[f"{label}_negative_date_rate"] = float(daily["negative"].mean())
        result[f"{label}_beat_universe"] = float(daily["beat"].mean())
        result[f"{label}_target7_precision"] = float(daily["target7_precision"].mean())
        result[f"{label}_dates"] = int(len(daily))
    top3_daily = _selected_daily(frame, rank_column, "top", 3, strict=True)
    result["Top3_days_le_minus_3"] = int(top3_daily["selected_return"].le(-0.03).sum())
    result["Top3_days_le_minus_5"] = int(top3_daily["selected_return"].le(-0.05).sum())
    return result


def _oracle_rank(frame: pd.DataFrame, top10_only: bool) -> pd.Series:
    ranks = pd.Series(index=frame.index, dtype=int)
    for _, day in frame.groupby("signal_date", sort=True):
        pool = day[day["stage1_rank"].le(TOP_K)] if top10_only else day
        ordered_pool = pool.sort_values(
            ["capped_opportunity_return_7", "event_id"],
            ascending=[False, True], kind="mergesort"
        )
        outside = day[~day.index.isin(ordered_pool.index)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        ordered = pd.concat([ordered_pool, outside])
        ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return ranks.astype(int)


def practical_table(
    oof: pd.DataFrame,
    bucket_map: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    evaluated = oof.copy()
    evaluated["top10_oracle_rank"] = _oracle_rank(evaluated, True)
    evaluated["full_oracle_rank"] = _oracle_rank(evaluated, False)
    rank_columns = {
        "CONTROL": "stage1_rank",
        "RERANKER": "final_rerank_rank",
        "TOP10_ORACLE": "top10_oracle_rank",
        "FULL_ORACLE": "full_oracle_rank",
    }
    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for scope in ("COMBINED", "LOW", "MID", "HIGH"):
        dates = (
            sorted(evaluated["signal_date"].unique())
            if scope == "COMBINED"
            else sorted(date for date, bucket in bucket_map.items() if bucket == scope)
        )
        subset = evaluated[evaluated["signal_date"].isin(dates)]
        for model, rank_column in rank_columns.items():
            summary = strategy_summary(subset, rank_column)
            if scope == "COMBINED":
                summaries[model] = summary
            row: dict[str, Any] = {"scope": scope, "model": model}
            for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3"):
                for metric in (
                    "mean", "median", "p25", "worst", "baseline", "excess",
                    "positive_date_rate", "negative_date_rate", "beat_universe",
                    "target7_precision", "dates",
                ):
                    row[f"{label}_{metric}"] = summary[f"{label}_{metric}"]
            row["winner_capture"] = summary["winner_capture"]
            row["universe_target7_rate"] = summary["universe_target7_rate"]
            row["Top3_days_le_minus_3"] = summary["Top3_days_le_minus_3"]
            row["Top3_days_le_minus_5"] = summary["Top3_days_le_minus_5"]
            rows.append(row)
        control = strategy_summary(subset, "stage1_rank")
        universe_row: dict[str, Any] = {
            "scope": scope,
            "model": "DAILY_UNIVERSE",
            "winner_capture": np.nan,
            "universe_target7_rate": control["universe_target7_rate"],
            "Top3_days_le_minus_3": np.nan,
            "Top3_days_le_minus_5": np.nan,
        }
        for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3"):
            baseline = control[f"{label}_baseline"]
            universe_row[f"{label}_mean"] = baseline
            universe_row[f"{label}_median"] = np.nan
            universe_row[f"{label}_p25"] = np.nan
            universe_row[f"{label}_worst"] = np.nan
            universe_row[f"{label}_baseline"] = baseline
            universe_row[f"{label}_excess"] = 0.0
            universe_row[f"{label}_positive_date_rate"] = np.nan
            universe_row[f"{label}_negative_date_rate"] = np.nan
            universe_row[f"{label}_beat_universe"] = np.nan
            universe_row[f"{label}_target7_precision"] = control["universe_target7_rate"]
            universe_row[f"{label}_dates"] = control[f"{label}_dates"]
        rows.append(universe_row)
    return pd.DataFrame(rows), summaries


def load_opportunity_buckets(root: Path, dates: Iterable[str]) -> dict[str, str]:
    path = (
        root
        / "reports/research/v004c_v4a_architecture_transfer_v001_20260506_20260630"
        / "v004c_v4a_transfer_opportunity_terciles_v001.csv"
    )
    source = pd.read_csv(path, encoding="utf-8-sig", usecols=["opportunity_bucket", "dates"])
    mapping: dict[str, str] = {}
    for row in source.itertuples(index=False):
        for date in str(row.dates).split("|"):
            if date >= JULY_SENTINEL_DATE or date in mapping:
                raise RuntimeError("FATAL: invalid opportunity bucket date")
            mapping[date] = str(row.opportunity_bucket)
    if set(mapping) != set(dates) or any(list(mapping.values()).count(bucket) != 7 for bucket in ("LOW", "MID", "HIGH")):
        raise RuntimeError("FATAL: frozen opportunity bucket parity failed")
    return mapping


def membership_audit(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    union = oof[oof["control_top3"] | oof["reranker_top3"]].copy()
    union["membership_status"] = np.select(
        [
            union["control_top3"] & union["reranker_top3"],
            union["control_top3"] & ~union["reranker_top3"],
            ~union["control_top3"] & union["reranker_top3"],
        ],
        ["PRESERVED", "DEMOTED", "PROMOTED"],
        default="INVALID",
    )
    demoted = union[union["membership_status"].eq("DEMOTED")]
    promoted = union[union["membership_status"].eq("PROMOTED")]
    preserved = union[union["membership_status"].eq("PRESERVED")]
    changed_dates = int(
        union.loc[union["membership_status"].ne("PRESERVED"), "signal_date"].nunique()
    )
    successful = 0
    possible = 0
    for _, day in oof.groupby("signal_date", sort=True):
        day_demoted = day[day["control_top3"] & ~day["reranker_top3"]]
        day_promoted = day[~day["control_top3"] & day["reranker_top3"]]
        successful += min(
            int(day_demoted["target7"].eq(0).sum()),
            int(day_promoted["target7"].eq(1).sum()),
        )
        fp = int((day["stage1_rank"].le(3) & day["target7"].eq(0)).sum())
        mw = int((day["stage1_rank"].between(4, 10) & day["target7"].eq(1)).sum())
        possible += min(fp, mw)
    demoted_total = len(demoted)
    promoted_total = len(promoted)
    stats = {
        "changed_dates": changed_dates,
        "unchanged_dates": int(oof["signal_date"].nunique() - changed_dates),
        "preserved": int(len(preserved)),
        "demoted_total": int(demoted_total),
        "demoted_target7": int(demoted["target7"].eq(1).sum()),
        "demoted_non_target": int(demoted["target7"].eq(0).sum()),
        "demoted_loss": int(demoted["raw_repair_return"].lt(0).sum()),
        "demoted_severe_loss": int(demoted["raw_repair_return"].le(-0.05).sum()),
        "promoted_total": int(promoted_total),
        "promoted_target7": int(promoted["target7"].eq(1).sum()),
        "promoted_non_target": int(promoted["target7"].eq(0).sum()),
        "promoted_loss": int(promoted["raw_repair_return"].lt(0).sum()),
        "false_positive_veto_rate": (
            float(demoted["target7"].eq(0).sum() / demoted_total) if demoted_total else np.nan
        ),
        "promotion_target7_rate": (
            float(promoted["target7"].eq(1).sum() / promoted_total) if promoted_total else np.nan
        ),
        "successful_replacement_slots": int(successful),
        "failed_replacement_slots": int(demoted_total - successful),
        "net_target7_slots_gained": int(
            promoted["target7"].sum() - demoted["target7"].sum()
        ),
        "possible_replacement_slots": int(possible),
        "replacement_capture": float(successful / possible) if possible else np.nan,
    }
    output = union[[
        "signal_date", "event_id", "code", "stage1_rank", "final_rerank_rank",
        "membership_status", "target7", "raw_repair_return",
        "capped_opportunity_return_7",
    ]].rename(columns={
        "stage1_rank": "control_rank", "final_rerank_rank": "reranker_rank"
    }).sort_values(["signal_date", "membership_status", "event_id"], kind="mergesort")
    return output.reset_index(drop=True), stats


def paired_daily_robustness(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, day in oof.groupby("signal_date", sort=True):
        k = min(3, len(day))
        control = float(day.loc[day["stage1_rank"].le(k), "capped_opportunity_return_7"].mean())
        reranker = float(day.loc[day["final_rerank_rank"].le(k), "capped_opportunity_return_7"].mean())
        rows.append({
            "signal_date": date,
            "control_top3": control,
            "reranker_top3": reranker,
            "daily_delta": reranker - control,
        })
    daily = pd.DataFrame(rows)
    delta = daily["daily_delta"].to_numpy(float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw = generator.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
    boot = delta[draw].mean(axis=1)
    bootstrap = np.quantile(boot, [0.025, 0.50, 0.975])
    lodo = np.asarray([np.delete(delta, index).mean() for index in range(len(delta))])
    best = daily.sort_values(["daily_delta", "signal_date"], ascending=[False, True], kind="mergesort").head(3)
    worst = daily.sort_values(["daily_delta", "signal_date"], ascending=[True, True], kind="mergesort").head(3)
    summary = {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "positive_dates": int((delta > 0).sum()),
        "negative_dates": int((delta < 0).sum()),
        "zero_dates": int((delta == 0).sum()),
        "bootstrap_p025": float(bootstrap[0]),
        "bootstrap_p50": float(bootstrap[1]),
        "bootstrap_p975": float(bootstrap[2]),
        "bootstrap_probability_positive": float((boot > 0).mean()),
        "lodo_positive_pct": float((lodo > 0).mean()),
        "lodo_min": float(lodo.min()),
        "lodo_median": float(np.median(lodo)),
        "lodo_max": float(lodo.max()),
        "best_dates": best.to_dict("records"),
        "worst_dates": worst.to_dict("records"),
        "paired_date_count": len(delta),
    }
    return daily, summary


def coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in STAGE2_FEATURE_COLUMNS:
        values = coefficients[f"beta_{feature}"].astype(float)
        rows.append({
            "feature": feature,
            "median_beta": float(values.median()),
            "mean_beta": float(values.mean()),
            "positive_fold_pct": float(values.gt(0).mean()),
            "negative_fold_pct": float(values.lt(0).mean()),
            "p25_beta": float(values.quantile(0.25)),
            "p75_beta": float(values.quantile(0.75)),
            "iqr_beta": float(values.quantile(0.75) - values.quantile(0.25)),
        })
    return pd.DataFrame(rows)


def _scope_summary(practical: pd.DataFrame, scope: str, model: str) -> pd.Series:
    row = practical[practical["scope"].eq(scope) & practical["model"].eq(model)]
    if len(row) != 1:
        raise RuntimeError(f"FATAL: practical row unavailable {scope}/{model}")
    return row.iloc[0]


def formal_status(
    summaries: Mapping[str, Mapping[str, Any]],
    practical: pd.DataFrame,
    membership: Mapping[str, Any],
    robustness: Mapping[str, Any],
    leakage_rows: int,
    parity_pass: bool,
) -> dict[str, Any]:
    control = summaries["CONTROL"]
    reranker = summaries["RERANKER"]
    low_c = _scope_summary(practical, "LOW", "CONTROL")
    low_r = _scope_summary(practical, "LOW", "RERANKER")
    high_c = _scope_summary(practical, "HIGH", "CONTROL")
    high_r = _scope_summary(practical, "HIGH", "RERANKER")
    top3_delta = float(reranker["Top3_mean"] - control["Top3_mean"])
    low_delta = float(low_r["Top3_excess"] - low_c["Top3_excess"])
    q3 = bool(reranker["Top3_mean"] > reranker["Top3_baseline"])
    q4 = bool(top3_delta >= 0.005)
    q5 = bool(
        reranker["Top3_target7_precision"] > control["Top3_target7_precision"]
        and reranker["Top3_target7_precision"] > reranker["universe_target7_rate"]
    )
    q6 = bool(
        low_delta >= 0.005
        and low_r["Top3_negative_date_rate"] <= low_c["Top3_negative_date_rate"]
    )
    q7 = bool(high_r["Top3_mean"] >= high_c["Top3_mean"] - 0.005)
    q8 = bool(
        reranker["Top3_worst"] >= control["Top3_worst"] - 0.01
        and reranker["Top3_negative_date_rate"]
        <= control["Top3_negative_date_rate"] + 0.05
    )
    q9 = bool(
        membership["promotion_target7_rate"] > control["Top3_target7_precision"]
        and membership["demoted_non_target"] > membership["demoted_target7"]
        and membership["net_target7_slots_gained"] > 0
    )
    q10 = bool(
        robustness["positive_dates"] > robustness["negative_dates"]
        and robustness["bootstrap_probability_positive"] >= 0.70
        and robustness["lodo_positive_pct"] >= 0.80
    )
    beat_gate = bool(reranker["Top3_beat_universe"] > 0.50)
    invalid = bool(not parity_pass or leakage_rows > 0)
    strong = bool(
        q3 and q4 and q5 and beat_gate and q6 and q7 and q8 and q9 and q10
    )
    if invalid:
        signal = "INVALID"
    elif strong:
        signal = "STRONG"
    elif (
        top3_delta <= 0.0
        or reranker["Top3_target7_precision"] <= control["Top3_target7_precision"]
        or low_delta <= 0.0
        or membership["promotion_target7_rate"] <= control["Top3_target7_precision"]
    ):
        signal = "ABSENT"
    else:
        signal = "PARTIAL"
    return {
        "Q1": parity_pass,
        "Q2": leakage_rows == 0,
        "Q3": q3,
        "Q4": q4,
        "Q5": q5,
        "Q6": q6,
        "Q7": q7,
        "Q8": q8,
        "Q9": q9,
        "Q10": q10,
        "top3_delta": top3_delta,
        "low_top3_excess_delta": low_delta,
        "reranker_signal": signal,
        "forward_stress_candidate": "YES" if signal == "STRONG" else "NO",
    }


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _num(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def build_review(
    practical: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, Any]],
    membership: Mapping[str, Any],
    robustness: Mapping[str, Any],
    fold_audit: pd.DataFrame,
    coefficient_audit: pd.DataFrame,
    parity: Mapping[str, Any],
    status: Mapping[str, Any],
) -> str:
    combined_rows = {
        model: _scope_summary(practical, "COMBINED", model)
        for model in ("DAILY_UNIVERSE", "CONTROL", "RERANKER", "TOP10_ORACLE")
    }
    lines = [
        "# Did the Two-Stage Reranker Produce a Better Tradable Top3?",
        "",
    ]
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Rank2", "Rank3", "Top2", "Top3", "Top3 Excess vs Universe", "Top3 Precision", "Beat-Universe", "Negative Days", "Worst"],
        [[model, _pct(row.Rank1_mean), _pct(row.Rank2_mean), _pct(row.Rank3_mean), _pct(row.Top2_mean), _pct(row.Top3_mean), _pct(row.Top3_excess), _pct(row.Top3_target7_precision), _pct(row.Top3_beat_universe), _pct(row.Top3_negative_date_rate), _pct(row.Top3_worst)] for model, row in combined_rows.items()],
    ))
    lines.extend([
        "",
        "# Did It Actually Replace the Wrong Stocks?",
        "",
        f"- Changed / unchanged dates: **{membership['changed_dates']} / {membership['unchanged_dates']}**",
        f"- Demoted false positives / winners: **{membership['demoted_non_target']} / {membership['demoted_target7']}**",
        f"- Promoted winners / false positives: **{membership['promoted_target7']} / {membership['promoted_non_target']}**",
        f"- Promotion Target7 rate: **{_pct(membership['promotion_target7_rate'])}**",
        f"- False-positive veto rate: **{_pct(membership['false_positive_veto_rate'])}**",
        f"- Net Target7 slots gained: **{membership['net_target7_slots_gained']}**",
        f"- Successful replacement slots / possible: **{membership['successful_replacement_slots']} / {membership['possible_replacement_slots']}**",
        f"- Replacement capture: **{_pct(membership['replacement_capture'])}**",
        "",
        "# Did Downside Improve Without Killing Upside?",
        "",
    ])
    opportunity_rows: list[list[str]] = []
    for scope in ("LOW", "MID", "HIGH"):
        for model in ("CONTROL", "RERANKER"):
            row = _scope_summary(practical, scope, model)
            opportunity_rows.append([scope, model, _pct(row.Rank1_mean), _pct(row.Top2_mean), _pct(row.Top3_mean), _pct(row.Top3_excess), _pct(row.Top3_negative_date_rate), _pct(row.Top3_worst)])
    lines.extend(_markdown_table(
        ["Bucket", "Model", "Rank1", "Top2", "Top3", "Top3 Excess", "Negative Rate", "Worst"], opportunity_rows,
    ))
    lines.extend([
        "",
        "# Was the Result Broad or Driven by One Date?",
        "",
        f"- Daily delta mean / median: **{_pct(robustness['mean'])} / {_pct(robustness['median'])}**",
        f"- Positive / negative / zero dates: **{robustness['positive_dates']} / {robustness['negative_dates']} / {robustness['zero_dates']}**",
        f"- Bootstrap 95%: **[{_pct(robustness['bootstrap_p025'])}, {_pct(robustness['bootstrap_p975'])}]**",
        f"- P(delta > 0): **{_pct(robustness['bootstrap_probability_positive'])}**",
        f"- Leave-one-date-out positive: **{_pct(robustness['lodo_positive_pct'])}**",
        "- Best 3 improvement dates: " + ", ".join(f"{row['signal_date']} ({_pct(row['daily_delta'])})" for row in robustness["best_dates"]),
        "- Worst 3 deterioration dates: " + ", ".join(f"{row['signal_date']} ({_pct(row['daily_delta'])})" for row in robustness["worst_dates"]),
        "",
        "# Was Stage2 Truly Leakage-Free?",
        "",
        f"- SELF_LABEL_LEAKAGE_ROWS = **{int(fold_audit['self_label_leakage_rows'].sum())}**",
        f"- Current test date leakage rows = **{int(fold_audit['current_test_date_leakage_rows'].sum())}**",
        "- July accessed = **NO**",
        f"- Stage1 control parity = **{'PASS' if parity['pass'] else 'FAIL'}**",
        "- Historical Stage1 meta predictions use H−{s}; Stage2 labels are applied only after each date's self-excluded Stage1 Top10 is formed.",
        "",
        "## Stage2 Coefficient Audit",
        "",
    ])
    lines.extend(_markdown_table(
        ["Feature", "Median Beta", "Positive Fold %", "Negative Fold %", "IQR"],
        [[row.feature, _num(row.median_beta, 6), _pct(row.positive_fold_pct), _pct(row.negative_fold_pct), _num(row.iqr_beta, 6)] for row in coefficient_audit.itertuples()],
    ))
    lines.extend([
        "",
        "## Oracle Gap",
        "",
        f"- CONTROL Top3: **{_pct(summaries['CONTROL']['Top3_mean'])}**",
        f"- RERANKER Top3: **{_pct(summaries['RERANKER']['Top3_mean'])}**",
        f"- Top10 Oracle Top3: **{_pct(summaries['TOP10_ORACLE']['Top3_mean'])}**",
        f"- Full Oracle Top3: **{_pct(summaries['FULL_ORACLE']['Top3_mean'])}**",
        f"- Reranker recovery ratio: **{_pct((summaries['RERANKER']['Top3_mean'] - summaries['CONTROL']['Top3_mean']) / (summaries['TOP10_ORACLE']['Top3_mean'] - summaries['CONTROL']['Top3_mean'])) if summaries['TOP10_ORACLE']['Top3_mean'] > summaries['CONTROL']['Top3_mean'] else 'NA'}**",
        "",
        "## Formal Decision",
        "",
    ])
    for number in range(1, 11):
        lines.append(f"- Q{number}: **{'YES' if status[f'Q{number}'] else 'NO'}**")
    lines.extend([
        "",
        f"RERANKER_SIGNAL: **{status['reranker_signal']}**",
        "",
        f"FORWARD_STRESS_CANDIDATE: **{status['forward_stress_candidate']}**",
        "",
        "- Production ready: **NO**",
        "- July accessed: **NO**",
        "- Feature selection: **NO**",
        "- Hyperparameter search: **NO**",
    ])
    if status["reranker_signal"] == "STRONG":
        lines.extend(["", "The frozen-v4a Top10 + tiny closing-completion reranker established a meaningful June chronological practical edge and is eligible for a separate forward stress test."])
    elif status["reranker_signal"] == "PARTIAL":
        lines.extend(["", "The tiny residual reranker produced some practical improvement, but did not satisfy every trading-candidate gate."])
    elif status["reranker_signal"] == "ABSENT":
        lines.extend(["", "This tiny residual reranker did not convert v4a Top10 retrieval strength into a reliable tradable Top3 edge."])
    return "\n".join(lines) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_model_contract()
    samples, base, feature_gate = prepare_reranker_samples(root)
    control_oof, _, _ = run_expanding_forward(samples)
    frozen = load_exact_transfer_oof(root)
    parity = validate_stage1_control_parity(control_oof, frozen)
    reranker_oof, fold_audit, coefficients = run_nested_reranker(samples, control_oof)
    output_oof = oof_output(reranker_oof)
    bucket_map = load_opportunity_buckets(root, output_oof["signal_date"].unique())
    practical, summaries = practical_table(output_oof, bucket_map)
    membership_frame, membership = membership_audit(output_oof)
    paired_daily, robustness = paired_daily_robustness(output_oof)
    coefficient_audit = coefficient_summary(coefficients)
    leakage_rows = int(
        fold_audit["self_label_leakage_rows"].sum()
        + fold_audit["current_test_date_leakage_rows"].sum()
    )
    status = formal_status(
        summaries, practical, membership, robustness, leakage_rows, parity["pass"]
    )
    review = build_review(
        practical, summaries, membership, robustness, fold_audit,
        coefficient_audit, parity, status,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(output_oof),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(fold_audit),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(membership_frame),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[5]: review.encode("utf-8"),
    }
    context = {
        "samples": samples,
        "base": base,
        "feature_gate": feature_gate,
        "control_oof": control_oof,
        "frozen_oof": frozen,
        "parity": parity,
        "oof": output_oof,
        "fold_audit": fold_audit,
        "coefficients": coefficients,
        "coefficient_audit": coefficient_audit,
        "membership_frame": membership_frame,
        "membership": membership,
        "practical": practical,
        "summaries": summaries,
        "bucket_map": bucket_map,
        "paired_daily": paired_daily,
        "robustness": robustness,
        "status": status,
        "historical_meta_rows": int(fold_audit["meta_train_rows"].sum()),
        "crossfit_stage1_fits": int(fold_audit["crossfit_stage1_models_fitted"].sum()),
        "self_label_leakage_rows": int(fold_audit["self_label_leakage_rows"].sum()),
        "current_test_date_leakage_rows": int(fold_audit["current_test_date_leakage_rows"].sum()),
        "output_sha256": {name: sha256_bytes(value) for name, value in outputs.items()},
    }
    return outputs, context


def run_v004c_v4a_top10_reranker(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first.keys() != second.keys() or any(first[name] != second[name] for name in first):
        raise RuntimeError("FATAL: deterministic reranker rebuild failed")
    output_dir = (
        root_path
        / "reports/research/v004c_v4a_top10_residual_reranker_v001_20260506_20260630"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (output_dir / name).write_bytes(first[name])
    context["deterministic_rebuild"] = "PASS"
    context["output_dir"] = output_dir
    return output_dir, context

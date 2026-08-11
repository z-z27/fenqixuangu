from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v004a import (
    DEFAULT_TARGET_COLUMN,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
)
from src.v004c_board2_board3_structural_audit import (
    EXPECTED_RAW_BOARD2,
    EXPECTED_RAW_BOARD3,
    EXPECTED_RAW_DATES,
    EXPECTED_RAW_ROWS,
    EXPECTED_STRICT_DATES,
    load_strict_funnel,
)
from src.v004c_pair_capped7_july_forward import (
    _sigmoid,
)
from src.v004c_stage1_risk_complementarity import prepare_diagnostic_samples
from src.v004c_stage1_top3_risk_information import binary_auc
from src.v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2,
    MODEL_FAMILY,
    MODEL_ID,
    POSITIVE_WEIGHT,
    dataframe_csv_bytes,
)


TASK_NAME = "BOARD3-ONLY FROZEN-V4A-ARCHITECTURE FEASIBILITY BENCHMARK"
EXPECTED_STARTING_HEAD = "5fa911907532bf03f9b4144f2263199dd9432384"
MODEL_NAME = "BOARD3_ONLY_V4A_ARCH_V001"
EXPERIMENT_TYPE = "BOARD3_ONLY_FROZEN_REPRESENTATION_WEIGHTED_L2_LOGISTIC"

DEVELOPMENT_ARCHITECTURE_FEASIBILITY = True
PRISTINE_OOT = False
BOARD3_INTERNAL_RANKING_ONLY = True
ONLY_CHANGED_DIMENSION = "TRAINING POPULATION"
FEATURE_SELECTION = False
HYPERPARAMETER_SEARCH = False
MODEL_ZOO = False
TARGET_SEARCH = False
CROSS_BOARD_COMBINATION = False
JULY_RESULT_ROWS_ACCESSED = 0
JULY_USED_FOR_MODEL_SELECTION = False
JULY_USED_FOR_FEATURE_SELECTION = False

MIN_TRAIN_BOARD3_SIGNAL_DATES = 18
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_RESAMPLES = 20_000
JULY_SENTINEL = "2026-07-01"

EXPECTED_MATURED_ROWS = 307
EXPECTED_MATURED_DATES = 37
EXPECTED_MATURED_BOARD3_ROWS = 55
EXPECTED_MATURED_BOARD3_DATES = 28

OUTCOME_COLUMNS = frozenset({
    "target7", "loss", "nonloss", "raw_repair_return",
    "capped_return_7", "capped_opportunity_return_7", "d2_date",
    "d3_date", "d2_open_daily", "d3_high_daily", "label_available_date",
})

OUTPUT_FILENAMES = (
    "v004c_board3_only_fold_audit_v001.csv",
    "v004c_board3_only_coefficients_v001.csv",
    "v004c_board3_only_prediction_lock_v001.csv",
    "v004c_board3_only_oof_v001.csv",
    "v004c_board3_only_pairwise_diagnostics_v001.csv",
    "v004c_board3_only_daily_v001.csv",
    "v004c_board3_only_practical_v001.csv",
    "v004c_board3_only_robustness_v001.csv",
    "v004c_board3_only_review_v001.md",
)


def assert_benchmark_contract() -> None:
    if MODEL_ID != "V4A_ARCH_TRANSFER_V4C":
        raise RuntimeError("FATAL: frozen Stage1 model id changed")
    if MODEL_FAMILY != "WEIGHTED_L2_LOGISTIC":
        raise RuntimeError("FATAL: frozen learner changed")
    if L2 != 0.30 or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen Stage1 hyperparameters changed")
    if len(FROZEN_FEATURE_COLUMNS) != 18 or len(set(FROZEN_FEATURE_COLUMNS)) != 18:
        raise RuntimeError("FATAL: frozen feature contract changed")
    if MIN_TRAIN_BOARD3_SIGNAL_DATES != 18:
        raise RuntimeError("FATAL: Board3 minimum-history contract changed")
    if any((FEATURE_SELECTION, HYPERPARAMETER_SEARCH, MODEL_ZOO, TARGET_SEARCH,
            CROSS_BOARD_COMBINATION)):
        raise RuntimeError("FATAL: unauthorized experiment dimension enabled")
    if JULY_RESULT_ROWS_ACCESSED != 0:
        raise RuntimeError("FATAL: July results accessed")


def select_board3_after_full_date_transform(full_x: pd.DataFrame) -> pd.DataFrame:
    """Filter only after the frozen full-date percentile-rank transform."""
    required = {"event_id", "signal_date", "board_streak_before_break", *FROZEN_FEATURE_COLUMNS}
    if not required.issubset(full_x.columns):
        raise RuntimeError("FATAL: full-date frozen feature matrix incomplete")
    selected = full_x[pd.to_numeric(full_x["board_streak_before_break"]).eq(3)].copy()
    return selected.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def fold_is_eligible(train_dates: int, target_classes: int) -> tuple[bool, str]:
    if int(train_dates) < MIN_TRAIN_BOARD3_SIGNAL_DATES:
        return False, f"MATURED_BOARD3_DATES_{int(train_dates)}_LT_18"
    if int(target_classes) < 2:
        return False, "UNFITTABLE_SINGLE_CLASS"
    return True, ""


def assign_internal_rank(frame: pd.DataFrame, score_column: str, rank_column: str) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["signal_date", score_column, "event_id"],
        ascending=[True, False, True], kind="mergesort",
    ).copy()
    ranked[rank_column] = ranked.groupby("signal_date", sort=True).cumcount() + 1
    return ranked.sort_values(["signal_date", rank_column, "event_id"], kind="mergesort").reset_index(drop=True)


def rank_percentile(rank: float, count: float) -> float:
    return float((float(rank) - 1.0) / max(float(count) - 1.0, 1.0))


def _fit_frozen(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if pd.to_numeric(train["board_streak_before_break"]).ne(3).any():
        raise RuntimeError("FATAL: Board2 row entered Board3-only fit")
    weights = build_training_sample_weight(train, positive_weight=POSITIVE_WEIGHT)
    beta = fit_logistic_l2_weighted(
        train[FROZEN_FEATURE_COLUMNS].to_numpy(float),
        train[DEFAULT_TARGET_COLUMN].astype(float).to_numpy(),
        l2=L2, sample_weight=weights,
    )
    return beta, weights


def _fit_unified(train: pd.DataFrame) -> np.ndarray:
    weights = build_training_sample_weight(train, positive_weight=POSITIVE_WEIGHT)
    return fit_logistic_l2_weighted(
        train[FROZEN_FEATURE_COLUMNS].to_numpy(float),
        train[DEFAULT_TARGET_COLUMN].astype(float).to_numpy(),
        l2=L2, sample_weight=weights,
    )


def _score_no_outcomes(test_x: pd.DataFrame, beta: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = [
        "event_id", "signal_date", "code", "board_streak_before_break",
        *FROZEN_FEATURE_COLUMNS,
    ]
    scored = test_x[columns].copy()
    scored[f"{prefix}_logit"] = beta[0] + scored[FROZEN_FEATURE_COLUMNS].to_numpy(float) @ beta[1:]
    scored[f"{prefix}_score"] = _sigmoid(scored[f"{prefix}_logit"].to_numpy(float))
    return scored


def build_chronological_oof(
    base_x: pd.DataFrame,
    samples: pd.DataFrame,
    strict_unified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit one frozen Board3-only and one frozen unified comparator per eligible date."""
    assert_benchmark_contract()
    board3_x = select_board3_after_full_date_transform(base_x)
    board3_samples = samples[pd.to_numeric(samples["board_streak_before_break"]).eq(3)].copy()
    test_dates = sorted(board3_samples["signal_date"].astype(str).unique())
    strict_dates = set(EXPECTED_STRICT_DATES)
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    eligible_dates: list[str] = []
    single_class_dates: list[str] = []

    for test_date in test_dates:
        train_b3 = board3_samples[board3_samples["label_available_date"].lt(test_date)].copy()
        train_dates = int(train_b3["signal_date"].nunique())
        classes = int(train_b3[DEFAULT_TARGET_COLUMN].nunique()) if len(train_b3) else 0
        eligible, reason = fold_is_eligible(train_dates, classes)
        test_x = board3_x[board3_x["signal_date"].eq(test_date)].copy()
        if bool(train_b3["label_available_date"].ge(test_date).any()):
            raise RuntimeError("FATAL: future label entered Board3-only fit")
        if bool(train_b3["signal_date"].eq(test_date).any()):
            raise RuntimeError("FATAL: current test label entered Board3-only fit")
        if bool(train_b3["signal_date"].ge(JULY_SENTINEL).any()):
            raise RuntimeError("FATAL: July row entered Board3-only fit")
        base_audit: dict[str, Any] = {
            "test_date": test_date, "eligible": eligible,
            "unavailable_reason": reason, "train_rows": int(len(train_b3)),
            "train_dates": train_dates,
            "train_target7": int(train_b3["target7"].sum()) if len(train_b3) else 0,
            "train_non_target": int(len(train_b3) - train_b3["target7"].sum()) if len(train_b3) else 0,
            "train_loss": int(train_b3["raw_repair_return"].lt(0).sum()) if len(train_b3) else 0,
            "train_nonloss": int(train_b3["raw_repair_return"].ge(0).sum()) if len(train_b3) else 0,
            "latest_train_signal_date": str(train_b3["signal_date"].max()) if len(train_b3) else "",
            "latest_label_available_date": str(train_b3["label_available_date"].max()) if len(train_b3) else "",
            "single_class": classes < 2,
            "self_label_leakage_rows": 0,
            "current_test_leakage_rows": 0,
            "board2_training_rows": int(pd.to_numeric(train_b3["board_streak_before_break"]).eq(2).sum()),
            "feature_count": len(FROZEN_FEATURE_COLUMNS), "l2": L2,
            "positive_weight": POSITIVE_WEIGHT, "weighted_target7_prevalence": np.nan,
            "beta_l2_norm": np.nan, "max_abs_beta": np.nan,
            "prediction_score_mean": np.nan, "prediction_score_std": np.nan,
            "prediction_score_min": np.nan, "prediction_score_max": np.nan,
            "test_board3_rows": int(len(test_x)), "july_rows_accessed": 0,
        }
        if not eligible:
            if reason == "UNFITTABLE_SINGLE_CLASS":
                single_class_dates.append(test_date)
            audit_rows.append(base_audit)
            continue

        beta_b3, weights = _fit_frozen(train_b3)
        eligible_dates.append(test_date)
        scored_b3 = _score_no_outcomes(test_x, beta_b3, "board3_only")
        scored_b3 = assign_internal_rank(scored_b3, "board3_only_score", "board3_only_internal_rank")
        scored_b3["board3_candidate_count"] = len(scored_b3)

        train_unified = samples[samples["label_available_date"].lt(test_date)].copy()
        if bool(train_unified["label_available_date"].ge(test_date).any()):
            raise RuntimeError("FATAL: future label entered unified comparator fit")
        beta_unified = _fit_unified(train_unified)
        unified = _score_no_outcomes(test_x, beta_unified, "unified_stage1")
        unified = assign_internal_rank(unified, "unified_stage1_score", "unified_board3_internal_rank")
        scored = scored_b3.merge(
            unified[["event_id", "unified_stage1_logit", "unified_stage1_score",
                     "unified_board3_internal_rank"]],
            on="event_id", validate="one_to_one",
        )
        if test_date in strict_dates:
            expected = strict_unified[
                strict_unified["signal_date"].eq(test_date)
                & strict_unified["board_group"].eq("BOARD3")
            ][["event_id", "stage1_score"]]
            checked = scored.merge(expected, on="event_id", validate="one_to_one")
            if len(checked) != len(scored) or not np.allclose(
                checked["unified_stage1_score"], checked["stage1_score"], rtol=0.0, atol=1e-10,
            ):
                raise RuntimeError("FATAL: unified strict score parity failed")
        prediction_parts.append(scored)

        y = train_b3[DEFAULT_TARGET_COLUMN].astype(float).to_numpy()
        base_audit.update({
            "weighted_target7_prevalence": float(np.sum(weights * y) / np.sum(weights)),
            "beta_l2_norm": float(np.linalg.norm(beta_b3[1:])),
            "max_abs_beta": float(np.max(np.abs(beta_b3[1:]))),
            "prediction_score_mean": float(scored["board3_only_score"].mean()),
            "prediction_score_std": float(scored["board3_only_score"].std(ddof=0)),
            "prediction_score_min": float(scored["board3_only_score"].min()),
            "prediction_score_max": float(scored["board3_only_score"].max()),
        })
        audit_rows.append(base_audit)
        for index, feature in enumerate(["INTERCEPT", *FROZEN_FEATURE_COLUMNS]):
            b3_value = float(beta_b3[index])
            unified_value = float(beta_unified[index])
            coefficient_rows.append({
                "test_date": test_date, "feature": feature,
                "board3_only_beta": b3_value, "unified_beta": unified_value,
                "beta_delta": b3_value - unified_value,
                "board3_only_sign": int(np.sign(b3_value)),
                "unified_sign": int(np.sign(unified_value)),
                "same_sign": bool(np.sign(b3_value) == np.sign(unified_value)),
            })

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["signal_date", "board3_only_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    audits = pd.DataFrame(audit_rows).sort_values("test_date", kind="mergesort").reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["test_date", "feature"], kind="mergesort"
    ).reset_index(drop=True)
    common_dates = sorted(set(eligible_dates).intersection(strict_dates))
    predictions = predictions[predictions["signal_date"].isin(common_dates)].copy().reset_index(drop=True)
    if predictions.empty:
        raise RuntimeError("FATAL: no common Board3 evaluation rows")
    metrics = {
        "eligible_dates": eligible_dates, "single_class_dates": single_class_dates,
        "common_dates": common_dates,
        "self_label_leakage_rows": int(audits["self_label_leakage_rows"].sum()),
        "current_test_leakage_rows": int(audits["current_test_leakage_rows"].sum()),
        "board2_training_rows": int(audits["board2_training_rows"].sum()),
    }
    return predictions, audits, coefficients, metrics


def build_prediction_lock(predictions: pd.DataFrame) -> pd.DataFrame:
    if OUTCOME_COLUMNS.intersection(predictions.columns):
        raise RuntimeError("FATAL: outcome column entered prediction lock")
    columns = [
        "event_id", "signal_date", "code", "board3_candidate_count",
        "unified_stage1_score", "unified_board3_internal_rank",
        "board3_only_logit", "board3_only_score", "board3_only_internal_rank",
        *FROZEN_FEATURE_COLUMNS,
    ]
    lock = predictions[columns].copy().sort_values(
        ["signal_date", "board3_only_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    for rank in ("unified_board3_internal_rank", "board3_only_internal_rank", "board3_candidate_count"):
        lock[rank] = pd.to_numeric(lock[rank]).astype(int)
    return lock


def prediction_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the outcome-free prediction surface used by the lock builder."""
    columns = [
        "event_id", "signal_date", "code", "board3_candidate_count",
        "unified_stage1_score", "unified_board3_internal_rank",
        "board3_only_logit", "board3_only_score", "board3_only_internal_rank",
        *FROZEN_FEATURE_COLUMNS,
    ]
    return frame[columns].copy()


def prediction_lock_sha256(frame: pd.DataFrame) -> str:
    return sha256(dataframe_csv_bytes(frame)).hexdigest()


def attach_outcomes(lock: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    outcomes = samples[[
        "event_id", "target7", "raw_repair_return", "capped_opportunity_return_7",
    ]].copy().rename(columns={"capped_opportunity_return_7": "capped_return_7"})
    outcomes["target7"] = outcomes["target7"].astype(int)
    outcomes["loss"] = outcomes["raw_repair_return"].lt(0).astype(int)
    outcomes["nonloss"] = 1 - outcomes["loss"]
    evaluated = lock.merge(outcomes, on="event_id", how="left", validate="one_to_one")
    if evaluated[["target7", "loss", "raw_repair_return", "capped_return_7"]].isna().any().any():
        raise RuntimeError("FATAL: common Board3 outcome join incomplete")
    evaluated["unified_rank_percentile"] = [
        rank_percentile(rank, count) for rank, count in zip(
            evaluated["unified_board3_internal_rank"], evaluated["board3_candidate_count"]
        )
    ]
    evaluated["board3_only_rank_percentile"] = [
        rank_percentile(rank, count) for rank, count in zip(
            evaluated["board3_only_internal_rank"], evaluated["board3_candidate_count"]
        )
    ]
    return evaluated.sort_values(
        ["signal_date", "board3_only_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def same_date_pair_summary(
    frame: pd.DataFrame, score_column: str, positive_column: str,
) -> dict[str, Any]:
    informative_dates = 0
    concordant = discordant = ties = 0
    for _, day in frame.groupby("signal_date", sort=True):
        positive = day[day[positive_column].eq(1)][score_column].to_numpy(float)
        negative = day[day[positive_column].eq(0)][score_column].to_numpy(float)
        if not len(positive) or not len(negative):
            continue
        informative_dates += 1
        comparisons = positive[:, None] - negative[None, :]
        concordant += int((comparisons > 0).sum())
        discordant += int((comparisons < 0).sum())
        ties += int((comparisons == 0).sum())
    pairs = concordant + discordant + ties
    return {
        "informative_dates": informative_dates, "informative_pairs": pairs,
        "concordant": concordant, "discordant": discordant, "ties": ties,
        "concordance": float((concordant + .5 * ties) / pairs) if pairs else np.nan,
        "pair_support_weak": pairs < 10,
    }


def winner_loss_pair_summary(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    informative_dates = 0
    concordant = discordant = ties = 0
    for _, day in frame.groupby("signal_date", sort=True):
        winner = day[day["target7"].eq(1)][score_column].to_numpy(float)
        loss = day[day["loss"].eq(1)][score_column].to_numpy(float)
        if not len(winner) or not len(loss):
            continue
        informative_dates += 1
        comparisons = winner[:, None] - loss[None, :]
        concordant += int((comparisons > 0).sum())
        discordant += int((comparisons < 0).sum())
        ties += int((comparisons == 0).sum())
    pairs = concordant + discordant + ties
    return {
        "informative_dates": informative_dates, "informative_pairs": pairs,
        "concordant": concordant, "discordant": discordant, "ties": ties,
        "concordance": float((concordant + .5 * ties) / pairs) if pairs else np.nan,
        "pair_support_weak": pairs < 10,
    }


def build_pair_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    models = {
        "UNIFIED": "unified_stage1_score",
        "BOARD3_ONLY": "board3_only_score",
    }
    endpoints = {
        "TARGET7_VS_NONTARGET": "target7",
        "NONLOSS_VS_LOSS": "nonloss",
    }
    for model, score in models.items():
        summary[model] = {}
        for endpoint, label in endpoints.items():
            result = same_date_pair_summary(frame, score, label)
            summary[model][endpoint] = result
            rows.append({"model": model, "endpoint": endpoint, **result})
        result = winner_loss_pair_summary(frame, score)
        summary[model]["TARGET7_VS_LOSS"] = result
        rows.append({"model": model, "endpoint": "TARGET7_VS_LOSS", **result})
    return pd.DataFrame(rows), summary


def _selected_daily(frame: pd.DataFrame, rank_column: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    for date, day in frame.groupby("signal_date", sort=True):
        selected = day[day[rank_column].le(min(k, len(day)))].copy()
        selected_parts.append(selected)
        daily_rows.append({
            "signal_date": date, "target7_rate": float(selected["target7"].mean()),
            "loss_rate": float(selected["loss"].mean()),
            "capped_mean": float(selected["capped_return_7"].mean()),
            "raw_mean": float(selected["raw_repair_return"].mean()),
        })
    return pd.DataFrame(daily_rows), pd.concat(selected_parts, ignore_index=True)


def practical_summary(frame: pd.DataFrame, rank_column: str | None, k: int | None) -> dict[str, Any]:
    if rank_column is None:
        daily = frame.groupby("signal_date", sort=True).agg(
            target7_rate=("target7", "mean"), loss_rate=("loss", "mean"),
            capped_mean=("capped_return_7", "mean"), raw_mean=("raw_repair_return", "mean"),
        ).reset_index()
        selected = frame.copy()
    else:
        assert k is not None
        daily, selected = _selected_daily(frame, rank_column, k)
    return {
        "rows_slots": int(len(selected)), "dates": int(len(daily)),
        "target7_rate": float(daily["target7_rate"].mean()),
        "loss_rate": float(daily["loss_rate"].mean()),
        "mean_capped": float(daily["capped_mean"].mean()),
        "median_capped": float(daily["capped_mean"].median()),
        "p25_capped": float(daily["capped_mean"].quantile(.25)),
        "worst_capped": float(daily["capped_mean"].min()),
        "raw_mean": float(selected["raw_repair_return"].mean()),
        "raw_median": float(selected["raw_repair_return"].median()),
        "worst_raw": float(selected["raw_repair_return"].min()),
    }


def build_practical(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = (
        ("BOARD3_UNIVERSE", None, None),
        ("UNIFIED_BOARD3_RANK1", "unified_board3_internal_rank", 1),
        ("BOARD3_ONLY_RANK1", "board3_only_internal_rank", 1),
        ("UNIFIED_BOARD3_TOP2", "unified_board3_internal_rank", 2),
        ("BOARD3_ONLY_TOP2", "board3_only_internal_rank", 2),
        ("UNIFIED_BOARD3_TOP3", "unified_board3_internal_rank", 3),
        ("BOARD3_ONLY_TOP3", "board3_only_internal_rank", 3),
    )
    rows = []
    summary: dict[str, Any] = {}
    for policy, rank, k in specs:
        values = practical_summary(frame, rank, k)
        summary[policy] = values
        rows.append({"policy": policy, **values})
    return pd.DataFrame(rows), summary


def _rank_group(frame: pd.DataFrame, rank_column: str, label: str) -> dict[str, float]:
    selected = frame[frame[label].eq(1)]
    percentile_column = (
        "unified_rank_percentile"
        if rank_column == "unified_board3_internal_rank"
        else "board3_only_rank_percentile"
    )
    return {
        "mean_rank": float(selected[rank_column].mean()),
        "median_rank": float(selected[rank_column].median()),
        "mean_rank_percentile": float(selected[percentile_column].mean()),
        "median_rank_percentile": float(selected[percentile_column].median()),
    }


def rank_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    unified_winner = _rank_group(frame, "unified_board3_internal_rank", "target7")
    board3_winner = _rank_group(frame, "board3_only_internal_rank", "target7")
    unified_loss = _rank_group(frame, "unified_board3_internal_rank", "loss")
    board3_loss = _rank_group(frame, "board3_only_internal_rank", "loss")
    return {
        "UNIFIED_TARGET7": unified_winner, "BOARD3_ONLY_TARGET7": board3_winner,
        "UNIFIED_LOSS": unified_loss, "BOARD3_ONLY_LOSS": board3_loss,
        "unified_selectivity": unified_loss["median_rank_percentile"] - unified_winner["median_rank_percentile"],
        "board3_only_selectivity": board3_loss["median_rank_percentile"] - board3_winner["median_rank_percentile"],
    }


def _state(row: pd.Series) -> str:
    if int(row["target7"]) == 1:
        return "TARGET7"
    if int(row["loss"]) == 1:
        return "LOSS"
    return "POSITIVE_NON_TARGET"


def build_daily_and_transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for date, day in frame.groupby("signal_date", sort=True):
        u = day.sort_values(["unified_board3_internal_rank", "event_id"], kind="mergesort")
        b = day.sort_values(["board3_only_internal_rank", "event_id"], kind="mergesort")
        old = u.iloc[0]
        new = b.iloc[0]
        u2 = u.head(min(2, len(u)))
        b2 = b.head(min(2, len(b)))
        daily_rows.append({
            "signal_date": date, "board3_candidate_count": len(day),
            "unified_rank1_code": old.code, "board3_only_rank1_code": new.code,
            "unified_rank1_target7": int(old.target7), "board3_only_rank1_target7": int(new.target7),
            "unified_rank1_loss": int(old.loss), "board3_only_rank1_loss": int(new.loss),
            "unified_rank1_capped": float(old.capped_return_7),
            "board3_only_rank1_capped": float(new.capped_return_7),
            "rank1_delta": float(new.capped_return_7 - old.capped_return_7),
            "rank1_target7_delta": int(new.target7 - old.target7),
            "rank1_loss_delta": int(new.loss - old.loss),
            "unified_top2_capped": float(u2.capped_return_7.mean()),
            "board3_only_top2_capped": float(b2.capped_return_7.mean()),
            "top2_delta": float(b2.capped_return_7.mean() - u2.capped_return_7.mean()),
        })
        if str(old.event_id) != str(new.event_id):
            category = f"{_state(old)} -> {_state(new)}"
            category_counts[category] = category_counts.get(category, 0) + 1
            transition_rows.append({
                "signal_date": date, "old_event_id": old.event_id, "old_code": old.code,
                "new_event_id": new.event_id, "new_code": new.code,
                "old_state": _state(old), "new_state": _state(new),
                "old_target7": int(old.target7), "new_target7": int(new.target7),
                "old_loss": int(old.loss), "new_loss": int(new.loss),
                "old_raw_return": float(old.raw_repair_return),
                "new_raw_return": float(new.raw_repair_return),
                "capped_delta": float(new.capped_return_7 - old.capped_return_7),
                "transition": category,
            })
    daily = pd.DataFrame(daily_rows)
    transitions = pd.DataFrame(transition_rows)
    summary = {
        "changed_dates": int(len(transitions)),
        "unified_loss_replaced": int(transitions["old_loss"].sum()) if len(transitions) else 0,
        "unified_target7_replaced": int(transitions["old_target7"].sum()) if len(transitions) else 0,
        "net_target7": int((transitions["new_target7"] - transitions["old_target7"]).sum()) if len(transitions) else 0,
        "net_loss_removed": int((transitions["old_loss"] - transitions["new_loss"]).sum()) if len(transitions) else 0,
        "categories": category_counts,
    }
    return daily, transitions, summary


def _summary(values: Sequence[float], direction: str = "positive") -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"valid": 0, "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan,
                "direction_probability": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
    probability = float(np.mean(array > 0)) if direction == "positive" else float(np.mean(array < 0))
    return {
        "valid": int(len(array)), "p2.5": float(np.quantile(array, .025)),
        "p50": float(np.quantile(array, .5)), "p97.5": float(np.quantile(array, .975)),
        "direction_probability": probability, "min": float(array.min()),
        "median": float(np.median(array)), "max": float(array.max()),
    }


def ranking_auc(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "unified_target7": binary_auc(frame["target7"], frame["unified_stage1_score"]),
        "board3_target7": binary_auc(frame["target7"], frame["board3_only_score"]),
        "unified_nonloss": binary_auc(frame["nonloss"], frame["unified_stage1_score"]),
        "board3_nonloss": binary_auc(frame["nonloss"], frame["board3_only_score"]),
    }


def build_robustness(frame: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dates = sorted(frame["signal_date"].unique())
    grouped = {date: frame[frame["signal_date"].eq(date)] for date in dates}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(dates), size=(BOOTSTRAP_RESAMPLES, len(dates)))
    bootstrap_values: dict[str, list[float]] = {
        "TARGET7_AUC_DELTA": [], "NONLOSS_AUC_DELTA": [],
        "RANK1_CAPPED_DELTA": [], "RANK1_TARGET7_DELTA": [],
        "RANK1_LOSS_DELTA": [], "TOP2_CAPPED_DELTA": [],
    }
    daily_indexed = daily.set_index("signal_date")
    for draw in draws:
        sampled_dates = [dates[index] for index in draw]
        sampled = pd.concat([grouped[date] for date in sampled_dates], ignore_index=True)
        auc = ranking_auc(sampled)
        bootstrap_values["TARGET7_AUC_DELTA"].append(auc["board3_target7"] - auc["unified_target7"])
        bootstrap_values["NONLOSS_AUC_DELTA"].append(auc["board3_nonloss"] - auc["unified_nonloss"])
        rows = daily_indexed.loc[sampled_dates]
        bootstrap_values["RANK1_CAPPED_DELTA"].append(float(rows["rank1_delta"].mean()))
        bootstrap_values["RANK1_TARGET7_DELTA"].append(float(rows["rank1_target7_delta"].mean()))
        bootstrap_values["RANK1_LOSS_DELTA"].append(float(rows["rank1_loss_delta"].mean()))
        bootstrap_values["TOP2_CAPPED_DELTA"].append(float(rows["top2_delta"].mean()))

    directions = {"RANK1_LOSS_DELTA": "negative"}
    bootstrap = {
        metric: _summary(values, directions.get(metric, "positive"))
        for metric, values in bootstrap_values.items()
    }
    lodo_values = {metric: [] for metric in bootstrap_values}
    for omitted in dates:
        kept = frame[~frame["signal_date"].eq(omitted)]
        auc = ranking_auc(kept)
        lodo_values["TARGET7_AUC_DELTA"].append(auc["board3_target7"] - auc["unified_target7"])
        lodo_values["NONLOSS_AUC_DELTA"].append(auc["board3_nonloss"] - auc["unified_nonloss"])
        kept_daily = daily[~daily["signal_date"].eq(omitted)]
        lodo_values["RANK1_CAPPED_DELTA"].append(float(kept_daily["rank1_delta"].mean()))
        lodo_values["RANK1_TARGET7_DELTA"].append(float(kept_daily["rank1_target7_delta"].mean()))
        lodo_values["RANK1_LOSS_DELTA"].append(float(kept_daily["rank1_loss_delta"].mean()))
        lodo_values["TOP2_CAPPED_DELTA"].append(float(kept_daily["top2_delta"].mean()))
    lodo = {
        metric: _summary(values, directions.get(metric, "positive"))
        for metric, values in lodo_values.items()
    }
    rows = []
    for section, values in (("BOOTSTRAP", bootstrap), ("LODO", lodo)):
        for metric, result in values.items():
            rows.append({"section": section, "metric": metric, **result})
    return pd.DataFrame(rows), bootstrap, lodo


def coefficient_stability(coefficients: pd.DataFrame, audits: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for feature in FROZEN_FEATURE_COLUMNS:
        current = coefficients[coefficients["feature"].eq(feature)]
        values = current["board3_only_beta"].to_numpy(float)
        unified = current["unified_beta"].to_numpy(float)
        positive = float(np.mean(values > 0))
        negative = float(np.mean(values < 0))
        rows.append({
            "feature": feature, "unified_median_beta": float(np.median(unified)),
            "board3_only_mean_beta": float(np.mean(values)),
            "board3_only_median_beta": float(np.median(values)),
            "board3_only_min_beta": float(np.min(values)),
            "board3_only_max_beta": float(np.max(values)),
            "beta_delta": float(np.median(values) - np.median(unified)),
            "positive_fold_pct": positive, "negative_fold_pct": negative,
            "zero_fold_pct": float(np.mean(values == 0)),
            "sign_consistency": max(positive, negative),
            "same_median_sign": bool(np.sign(np.median(values)) == np.sign(np.median(unified))),
        })
    summary_frame = pd.DataFrame(rows)
    valid_audits = audits[audits["eligible"]]
    median_norm = float(valid_audits["beta_l2_norm"].median())
    max_norm = float(valid_audits["beta_l2_norm"].max())
    return {
        "table": summary_frame,
        "coefficient_instability_warning": int(summary_frame["sign_consistency"].lt(.70).sum()) >= 6,
        "extreme_coefficient_fold": bool(max_norm > 3.0 * median_norm),
        "prediction_collapse": bool(frame["board3_only_score"].std(ddof=0) < .01),
        "median_beta_l2_norm": median_norm, "max_beta_l2_norm": max_norm,
    }


def candidate_count_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.groupby("signal_date", sort=True).size().astype(int)
    return {
        "dates": int(len(counts)), "min": int(counts.min()),
        "median": float(counts.median()), "mean": float(counts.mean()),
        "max": int(counts.max()), "dates_1": int(counts.eq(1).sum()),
        "dates_2": int(counts.eq(2).sum()), "dates_3": int(counts.eq(3).sum()),
        "dates_ge4": int(counts.ge(4).sum()),
    }


def formal_decision(context: Mapping[str, Any]) -> dict[str, Any]:
    frame = context["oof"]
    auc = context["ranking_auc"]
    practical = context["practical_summary"]
    pair = context["pair_summary"]
    ranks = context["rank_diagnostics"]
    daily = context["daily"]
    bootstrap = context["bootstrap"]
    stability = context["coefficient_stability"]
    u1, b1 = practical["UNIFIED_BOARD3_RANK1"], practical["BOARD3_ONLY_RANK1"]
    u2, b2 = practical["UNIFIED_BOARD3_TOP2"], practical["BOARD3_ONLY_TOP2"]
    direct_u = pair["UNIFIED"]["TARGET7_VS_LOSS"]
    direct_b = pair["BOARD3_ONLY"]["TARGET7_VS_LOSS"]
    direct_support = direct_b["informative_pairs"] >= 10
    gates = {
        "A": frame["signal_date"].nunique() >= 6 and len(frame) >= 12,
        "B": auc["board3_target7"] >= .60 and auc["board3_target7"] - auc["unified_target7"] >= .05,
        "C": auc["board3_nonloss"] >= .55 and auc["board3_nonloss"] - auc["unified_nonloss"] >= .15,
        "D": (not direct_support) or (
            direct_b["concordance"] >= .60
            and direct_b["concordance"] - direct_u["concordance"] >= .10
        ),
        "E": b1["target7_rate"] >= u1["target7_rate"],
        "F": b1["loss_rate"] <= u1["loss_rate"] - .10,
        "G": b1["mean_capped"] >= u1["mean_capped"] + .005,
        "H": b2["mean_capped"] >= u2["mean_capped"] and b2["target7_rate"] >= u2["target7_rate"] - .05,
        "I": ranks["board3_only_selectivity"] > ranks["unified_selectivity"],
        "J": int(daily["rank1_delta"].gt(0).sum()) > int(daily["rank1_delta"].lt(0).sum()),
        "K": bootstrap["RANK1_CAPPED_DELTA"]["direction_probability"] >= .65
             and bootstrap["NONLOSS_AUC_DELTA"]["direction_probability"] >= .75,
        "L": not stability["prediction_collapse"] and not stability["extreme_coefficient_fold"],
    }
    if not gates["A"]:
        signal = "INSUFFICIENT_EVIDENCE"
    elif all(gates.values()):
        signal = "STRONG"
    else:
        meaningful = (
            (auc["board3_nonloss"] > .50 and auc["board3_nonloss"] - auc["unified_nonloss"] >= .10)
            or (auc["board3_target7"] >= .55 and auc["board3_target7"] - auc["unified_target7"] >= .05)
            or (b1["mean_capped"] > u1["mean_capped"] and b1["loss_rate"] < u1["loss_rate"])
            or b2["mean_capped"] > u2["mean_capped"]
        )
        signal = "PARTIAL" if meaningful else "ABSENT"
    action = {
        "STRONG": "BOARD3_MINIMAL_REPRESENTATION_AUDIT",
        "PARTIAL": "NO_NEW_MODEL_YET", "ABSENT": "STOP_BOARD3_D1_MODEL_ROUTE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_DATA", "INVALID": "INVALID",
    }[signal]
    return {"gates": gates, "signal": signal, "next_action": action,
            "direct_pair_support": direct_support}


def _pct(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 6) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    population = context["matured_board3"]
    frame = context["oof"]
    audit = context["fold_audit"]
    auc = context["ranking_auc"]
    practical = context["practical_summary"]
    pairs = context["pair_summary"]
    ranks = context["rank_diagnostics"]
    stability = context["coefficient_stability"]
    decision = context["decision"]
    daily = context["daily"]
    lines = [
        "# v004c Board3-Only Frozen-V4A-Architecture Feasibility Benchmark v001", "",
        "## 1. Experimental Contract", "",
        "- Development architecture feasibility: **YES**; pristine OOT: **NO**.",
        "- Board3 internal ranking only. The only changed dimension is the training population.",
        "- No feature, target, learner, weighting, hyperparameter, model-zoo, cross-board merge, or July-result access.", "",
        "## 2. Matured Board3 Population", "",
        f"- Matured pre-July: **307 rows / 37 dates**; Board3: **{len(population)} rows / {population.signal_date.nunique()} dates**.",
        f"- Board3 Target7: **{int(population.target7.sum())}**; LOSS: **{int(population.raw_repair_return.lt(0).sum())}**.", "",
        "## 3. Chronological OOF Availability", "",
        f"- Board3-only eligible dates: **{', '.join(context['availability']['eligible_dates'])}**.",
        f"- Common evaluation: **{frame.signal_date.nunique()} dates / {len(frame)} rows**.",
        f"- Unfittable single-class dates: **{len(context['availability']['single_class_dates'])}**.", "",
        "## 4. Common Evaluation Population", "",
        f"- Target7 rows: **{int(frame.target7.sum())}**; LOSS rows: **{int(frame.loss.sum())}**.",
        f"- Board3 candidates/date: min **{context['candidate_counts']['min']}**, median **{_num(context['candidate_counts']['median'], 2)}**, max **{context['candidate_counts']['max']}**.", "",
        "## 5. Frozen 18-Feature Parity", "",
        f"- Model: `{MODEL_ID}` / `{MODEL_FAMILY}`; 18 exact features; L2 `{L2:.2f}`; positive weight `{POSITIVE_WEIGHT:.2f}`.",
        "- Percentile-rank features were constructed on the complete daily V4C universe before Board3 filtering.", "",
        "## 6. Training / Leakage Audit", "",
        f"- Self leakage: **{context['availability']['self_label_leakage_rows']}**; current-test leakage: **{context['availability']['current_test_leakage_rows']}**; Board2 training rows: **{context['availability']['board2_training_rows']}**; July rows: **0**.",
        f"- Prediction lock SHA256: `{context['prediction_lock_sha256']}`; outcome perturbation: **PASS**.", "",
        "## 7. Coefficient Stability", "",
        f"- Instability warning: **{'YES' if stability['coefficient_instability_warning'] else 'NO'}**; extreme fold: **{'YES' if stability['extreme_coefficient_fold'] else 'NO'}**; prediction collapse: **{'YES' if stability['prediction_collapse'] else 'NO'}**.",
        f"- Median / max beta L2 norm: **{_num(stability['median_beta_l2_norm'])} / {_num(stability['max_beta_l2_norm'])}**.", "",
        "## 8. Unified vs Board3-Only Coefficient Changes", "",
        "| Feature | Unified median | Board3-only median | Delta | Same sign |", "|---|---:|---:|---:|:---:|",
    ]
    for row in stability["table"].itertuples(index=False):
        lines.append(f"| {row.feature} | {_num(row.unified_median_beta)} | {_num(row.board3_only_median_beta)} | {_num(row.beta_delta)} | {'YES' if row.same_median_sign else 'NO'} |")
    lines += ["", "## 9. Target7 Ranking", "",
        f"- Unified AUC: **{_num(auc['unified_target7'], 4)}**; Board3-only AUC: **{_num(auc['board3_target7'], 4)}**; delta: **{_num(auc['board3_target7'] - auc['unified_target7'], 4)}**.", "",
        "## 10. NONLOSS Ranking", "",
        f"- Unified AUC: **{_num(auc['unified_nonloss'], 4)}**; Board3-only AUC: **{_num(auc['board3_nonloss'], 4)}**; delta: **{_num(auc['board3_nonloss'] - auc['unified_nonloss'], 4)}**.", "",
        "## 11. Same-Date Winner / Loss Ordering", "",
    ]
    for endpoint in ("TARGET7_VS_NONTARGET", "NONLOSS_VS_LOSS", "TARGET7_VS_LOSS"):
        u, b = pairs["UNIFIED"][endpoint], pairs["BOARD3_ONLY"][endpoint]
        lines.append(f"- {endpoint}: {u['informative_pairs']} pairs; unified **{_num(u['concordance'], 4)}**, Board3-only **{_num(b['concordance'], 4)}**, delta **{_num(b['concordance'] - u['concordance'], 4)}**.")
    lines += ["", "## 12. Rank1 / Top2 Practical Performance", "",
        "| Policy | Target7 | LOSS | Mean capped | Median capped | Worst |", "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ("BOARD3_UNIVERSE", "UNIFIED_BOARD3_RANK1", "BOARD3_ONLY_RANK1", "UNIFIED_BOARD3_TOP2", "BOARD3_ONLY_TOP2", "UNIFIED_BOARD3_TOP3", "BOARD3_ONLY_TOP3"):
        row = practical[policy]
        lines.append(f"| {policy} | {_pct(row['target7_rate'])} | {_pct(row['loss_rate'])} | {_pct(row['mean_capped'])} | {_pct(row['median_capped'])} | {_pct(row['worst_capped'])} |")
    lines += ["", "## 13. Winner and Loss Rank Movement", "",
        f"- Unified / Board3-only rank selectivity: **{_num(ranks['unified_selectivity'], 4)} / {_num(ranks['board3_only_selectivity'], 4)}**.",
        f"- Rank1 changed dates: **{context['transition_summary']['changed_dates']}**; net Target7: **{context['transition_summary']['net_target7']}**; net losses removed: **{context['transition_summary']['net_loss_removed']}**.", "",
        "## 14. Bootstrap / LODO", "",
        f"- Rank1 daily delta: mean **{_pct(daily.rank1_delta.mean())}**, positive/negative/zero **{int(daily.rank1_delta.gt(0).sum())}/{int(daily.rank1_delta.lt(0).sum())}/{int(daily.rank1_delta.eq(0).sum())}**.",
        f"- Bootstrap P(Rank1 capped delta > 0): **{_pct(context['bootstrap']['RANK1_CAPPED_DELTA']['direction_probability'])}**.",
        f"- Bootstrap P(NONLOSS AUC delta > 0): **{_pct(context['bootstrap']['NONLOSS_AUC_DELTA']['direction_probability'])}**.", "",
        "## 15. Feasibility Decision", "",
        "| Gate | Pass |", "|---|:---:|",
    ]
    for gate, passed in decision["gates"].items():
        lines.append(f"| {gate} | {'YES' if passed else 'NO'} |")
    lines += ["",
        f"- `BOARD3_ONLY_SIGNAL = {decision['signal']}`",
        f"- `NEXT_BOARD3_ACTION = {decision['next_action']}`",
        "- This benchmark evaluates Board3-internal mapping feasibility only. It does not calibrate or merge Board2 and Board3 scores.",
        "- No July result was accessed, and this small May/June development sample is not production evidence.",
    ]
    if decision["signal"] == "ABSENT":
        lines.append("- Recommended action: stop the current Board3 D1-only model-development route using the frozen v4a representation. Do not automatically try GBDT, pairwise models, new targets, or another feature screen.")
    elif decision["signal"] == "STRONG":
        lines.append("- Recommended action: in a separate May/June-only task, conduct a Board3 minimal-representation audit; do not access July yet.")
    elif decision["signal"] == "PARTIAL":
        lines.append("- Recommended action: preserve the result and do not start another model or tune this specification yet.")
    else:
        lines.append("- Recommended action: preserve the descriptive audit; support is insufficient without lowering the frozen history gate.")
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_benchmark_contract()
    base_x, samples, _, source_audit = prepare_diagnostic_samples(root)
    # prepare_diagnostic_samples constructs the frozen X matrix on every full
    # date before select_board3_after_full_date_transform is ever called.
    strict, strict_parity = load_strict_funnel(root)
    if source_audit != {
        "rows": EXPECTED_RAW_ROWS, "dates": EXPECTED_RAW_DATES,
        "may_rows": 146, "may_dates": 18, "june_rows": 173, "june_dates": 21,
        "board2": EXPECTED_RAW_BOARD2, "board3": EXPECTED_RAW_BOARD3,
        "july_result_rows_accessed": 0,
    }:
        raise RuntimeError("FATAL: authoritative universe parity changed")
    if len(samples) != EXPECTED_MATURED_ROWS or samples["signal_date"].nunique() != EXPECTED_MATURED_DATES:
        raise RuntimeError("FATAL: matured pre-July population changed")
    matured_board3 = samples[pd.to_numeric(samples["board_streak_before_break"]).eq(3)].copy()
    if len(matured_board3) != EXPECTED_MATURED_BOARD3_ROWS or matured_board3["signal_date"].nunique() != EXPECTED_MATURED_BOARD3_DATES:
        raise RuntimeError("FATAL: matured Board3 population must be 55 rows / 28 dates")
    if bool(matured_board3["label_available_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: immature Board3 outcome entered benchmark")

    predictions, fold_audit, coefficients, availability = build_chronological_oof(
        base_x, samples, strict
    )
    lock = build_prediction_lock(prediction_inputs(predictions))
    lock_hash = prediction_lock_sha256(lock)
    perturbed = predictions.copy()
    rng = np.random.default_rng(20260817)
    for column in OUTCOME_COLUMNS:
        perturbed[column] = rng.normal(1_000.0, 100.0, len(perturbed))
    if dataframe_csv_bytes(build_prediction_lock(prediction_inputs(perturbed))) != dataframe_csv_bytes(lock):
        raise RuntimeError("FATAL: outcome perturbation changed prediction lock")
    oof = attach_outcomes(lock, samples)
    if sorted(oof["signal_date"].unique()) != availability["common_dates"]:
        raise RuntimeError("FATAL: common-date intersection changed")
    if not oof["event_id"].isin(strict["event_id"]).all():
        raise RuntimeError("FATAL: unified comparator identity mismatch")

    ranking = ranking_auc(oof)
    pair_table, pair_summary = build_pair_diagnostics(oof)
    practical, practical_summary = build_practical(oof)
    ranks = rank_diagnostics(oof)
    daily, transitions, transition_summary = build_daily_and_transitions(oof)
    robustness, bootstrap, lodo = build_robustness(oof, daily)
    stability = coefficient_stability(coefficients, fold_audit, oof)
    counts = candidate_count_summary(oof)
    context: dict[str, Any] = {
        "source_audit": source_audit, "strict_parity": strict_parity,
        "matured_board3": matured_board3, "fold_audit": fold_audit,
        "coefficients": coefficients, "prediction_lock": lock,
        "prediction_lock_sha256": lock_hash, "oof": oof,
        "ranking_auc": ranking, "pair_table": pair_table,
        "pair_summary": pair_summary, "practical": practical,
        "practical_summary": practical_summary, "rank_diagnostics": ranks,
        "daily": daily, "transitions": transitions,
        "transition_summary": transition_summary, "robustness": robustness,
        "bootstrap": bootstrap, "lodo": lodo,
        "coefficient_stability": stability, "candidate_counts": counts,
        "availability": availability, "outcome_perturbation": "PASS",
        "july_result_rows_accessed": 0,
    }
    context["decision"] = formal_decision(context)
    review = render_review(context)
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(fold_audit),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(lock),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(oof),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(pair_table),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[6]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[7]: dataframe_csv_bytes(robustness),
        OUTPUT_FILENAMES[8]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_board3_only_v4a_arch_feasibility(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path / "reports/research/v004c_board3_only_v4a_arch_feasibility_v001_20260506_20260626"
    )
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first != second:
        mismatched = [name for name in first if first[name] != second[name]]
        raise RuntimeError(f"FATAL: deterministic rebuild failed: {mismatched}")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (target / name).write_bytes(payload)
    context["deterministic_rebuild"] = "PASS"
    return target, context

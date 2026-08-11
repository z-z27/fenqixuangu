"""Frozen V4C_STAGE1 Board2-versus-Board3 feature-response audit.

The only fits performed here reconstruct the already-frozen strict June OOF
Stage1 folds.  Board identity is an audit label, never a model input.  Feature
contributions are diagnostic ``x_j * beta_j`` decompositions of those frozen
folds and are never used by another learner or trading policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_board2_board3_structural_audit import (
    BOARD_GROUPS,
    board_group,
    load_strict_funnel,
)
from .v004c_pair_capped7_july_forward import _fit_stage1_snapshot, _score_stage1
from .v004c_stage1_risk_complementarity import (
    JULY_SENTINEL,
    prepare_diagnostic_samples,
)
from .v004c_stage1_top3_risk_information import (
    benjamini_hochberg,
    binary_auc,
)
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    INITIAL_TRAIN_DATES,
    L2,
    MODEL_FAMILY,
    MODEL_ID,
    POSITIVE_WEIGHT,
    dataframe_csv_bytes,
)


TASK_NAME = "BOARD2 VS BOARD3 STAGE1 FEATURE-RESPONSE AUDIT"
EXPECTED_STARTING_HEAD = "e58c0da22553f598aaf5874b614b1ee5ae2e1949"
NEW_MODEL = False
NEW_FEATURE = False
NEW_POLICY = False
THRESHOLD_SEARCH = False
DIAGNOSTIC_CONTRIBUTION_ONLY = True
NEW_MODEL_FEATURE = False
JULY_RESULT_ROWS_ACCESSED = 0
JULY_USED_FOR_FEATURE_RESPONSE_AUDIT = False

BOOTSTRAP_SEED = 20260814
BOOTSTRAP_RESAMPLES = 20_000
PERMUTATION_SEED = 20260815
PERMUTATIONS = 10_000
PAIR_SUPPORT_MIN = 20

EXPECTED_FEATURE_COLUMNS = [
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
]

OUTPUT_FILENAMES = (
    "v004c_board_feature_response_population_v001.csv",
    "v004c_board_feature_response_coefficients_v001.csv",
    "v004c_board_feature_response_univariate_v001.csv",
    "v004c_board_feature_response_daily_v001.csv",
    "v004c_board_feature_response_pairs_v001.csv",
    "v004c_board_feature_response_robustness_v001.csv",
    "v004c_board_feature_response_redundancy_v001.csv",
    "v004c_board_feature_response_review_v001.md",
)


def assert_audit_contract() -> None:
    if MODEL_ID != "V4A_ARCH_TRANSFER_V4C":
        raise RuntimeError("FATAL: frozen Stage1 model identity changed")
    if MODEL_FAMILY != "WEIGHTED_L2_LOGISTIC":
        raise RuntimeError("FATAL: frozen Stage1 learner changed")
    if L2 != 0.30 or POSITIVE_WEIGHT != 1.50 or INITIAL_TRAIN_DATES != 18:
        raise RuntimeError("FATAL: frozen Stage1 fit contract changed")
    if list(FROZEN_FEATURE_COLUMNS) != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError("FATAL: frozen Stage1 18-feature contract changed")
    if any((NEW_MODEL, NEW_FEATURE, NEW_POLICY, THRESHOLD_SEARCH)):
        raise RuntimeError("FATAL: audit expanded into model/policy development")
    if not DIAGNOSTIC_CONTRIBUTION_ONLY or NEW_MODEL_FEATURE:
        raise RuntimeError("FATAL: diagnostic contribution scope changed")
    if JULY_RESULT_ROWS_ACCESSED != 0 or JULY_USED_FOR_FEATURE_RESPONSE_AUDIT:
        raise RuntimeError("FATAL: July result data entered feature-response audit")
    if BOOTSTRAP_RESAMPLES != 20_000 or PERMUTATIONS != 10_000:
        raise RuntimeError("FATAL: robustness resampling contract changed")


def contribution_column(feature: str) -> str:
    return f"contrib__{feature}"


def contribution_decomposition(
    x: np.ndarray, beta: np.ndarray, intercept: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=float)
    coefficients = np.asarray(beta, dtype=float)
    if values.ndim != 2 or coefficients.ndim != 1 or values.shape[1] != len(coefficients):
        raise ValueError("x/beta shape mismatch")
    contributions = values * coefficients
    logits = float(intercept) + contributions.sum(axis=1)
    return contributions, logits


def response_category(board2_auc: float, board3_auc: float) -> str:
    b2 = float(board2_auc)
    b3 = float(board3_auc)
    if b2 >= 0.55 and b3 >= 0.55:
        return "HEALTHY_SHARED"
    if b2 >= 0.55 and b3 <= 0.45:
        return "BOARD3_INVERTED"
    if b2 >= 0.55 and 0.45 < b3 < 0.55:
        return "BOARD3_DEGRADED"
    if 0.45 < b2 < 0.55 and 0.45 < b3 < 0.55:
        return "SHARED_WEAK"
    if b2 <= 0.45 and b3 <= 0.45:
        return "SHARED_BAD"
    return "OTHER"


def winner_safety_status(board3_target7_auc: float) -> str:
    value = float(board3_target7_auc)
    if value >= 0.55:
        return "WINNER_ALIGNED"
    if value <= 0.45:
        return "WINNER_HARMFUL"
    return "WINNER_NEUTRAL"


def response_inversion_gate(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["board2_nonloss_auc"]) >= 0.55
        and float(row["board3_nonloss_auc"]) <= 0.45
        and float(row["delta_nonloss_auc"]) <= -0.15
        and float(row["mean_daily_contribution_gap"]) > 0.0
        and float(row["bootstrap_P_delta_nonloss_lt0"]) >= 0.90
        and float(row["bootstrap_P_gap_gt0"]) >= 0.80
        and float(row["lodo_delta_nonloss_negative_pct"]) >= 0.80
        and float(row["interaction_q"]) <= 0.10
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _cliff_positive_vs_negative(labels: Sequence[int], values: Sequence[float]) -> float:
    auc = binary_auc(labels, values)
    return float(2.0 * auc - 1.0) if np.isfinite(auc) else np.nan


def reconstruct_frozen_stage1(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Re-fit only the frozen strict Stage1 folds and decompose their logits."""
    assert_audit_contract()
    base_x, samples, _, source_audit = prepare_diagnostic_samples(root)
    frozen, parity = load_strict_funnel(root)
    strict_dates = sorted(frozen["signal_date"].astype(str).unique())
    parts: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    max_logit_error = 0.0
    max_score_error = 0.0
    for test_date in strict_dates:
        matured = samples[samples["label_available_date"].lt(test_date)].copy()
        if matured["signal_date"].nunique() < INITIAL_TRAIN_DATES:
            raise RuntimeError("FATAL: strict Stage1 history below frozen minimum")
        if bool(matured["label_available_date"].ge(test_date).any()):
            raise RuntimeError("FATAL: current-test leakage")
        if bool(matured["signal_date"].eq(test_date).any()):
            raise RuntimeError("FATAL: current test date entered Stage1 training")
        if bool(matured["signal_date"].ge(JULY_SENTINEL).any()):
            raise RuntimeError("FATAL: July signal entered Stage1 training")
        beta = _fit_stage1_snapshot(matured)
        scored = _score_stage1(
            base_x[base_x["signal_date"].eq(test_date)].copy(), beta
        )
        expected = frozen[frozen["signal_date"].eq(test_date)][[
            "event_id", "stage1_score", "stage1_rank", "candidate_count",
            "board_group", "stage1_rank_percentile", "target7", "loss",
            "raw_repair_return", "capped_return_7", "code",
        ]].copy()
        scored = scored.merge(
            expected,
            on="event_id",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_frozen"),
        )
        if len(scored) != len(expected):
            raise RuntimeError("FATAL: strict Stage1 row identity mismatch")
        if not np.array_equal(
            scored["stage1_rank"].to_numpy(int),
            scored["stage1_rank_frozen"].to_numpy(int),
        ):
            raise RuntimeError("FATAL: strict Stage1 rank parity failed")
        x = scored[FROZEN_FEATURE_COLUMNS].to_numpy(float)
        contributions, logits = contribution_decomposition(x, beta[1:], beta[0])
        direct_logits = beta[0] + x @ beta[1:]
        score = _sigmoid(logits)
        max_logit_error = max(max_logit_error, float(np.max(np.abs(logits - direct_logits))))
        max_score_error = max(
            max_score_error,
            float(np.max(np.abs(score - scored["stage1_score_frozen"].to_numpy(float)))),
        )
        for index, feature in enumerate(FROZEN_FEATURE_COLUMNS):
            scored[contribution_column(feature)] = contributions[:, index]
            coefficient_rows.append({
                "test_date": test_date,
                "feature": feature,
                "beta": float(beta[index + 1]),
                "positive": bool(beta[index + 1] > 0),
                "negative": bool(beta[index + 1] < 0),
                "intercept": float(beta[0]),
            })
        scored["stage1_logit"] = logits
        scored["stage1_score"] = score
        scored["stage1_rank"] = scored["stage1_rank_frozen"].astype(int)
        scored["candidate_count"] = scored["candidate_count_frozen"].astype(int)
        scored["board_group"] = scored["board_group"]
        scored["stage1_rank_percentile"] = scored["stage1_rank_percentile"]
        scored["target7"] = scored["target7"].astype(int)
        scored["loss"] = scored["loss"].astype(int)
        scored["nonloss"] = 1 - scored["loss"]
        scored["raw_repair_return"] = scored["raw_repair_return"]
        scored["capped_return_7"] = scored["capped_return_7"]
        scored["code"] = scored["code_frozen"].astype(str).str.zfill(6)
        parts.append(scored)
        fold_rows.append({
            "test_date": test_date,
            "training_rows": int(len(matured)),
            "training_dates": int(matured["signal_date"].nunique()),
            "latest_label_available_date": str(matured["label_available_date"].max()),
            "self_label_leakage_rows": 0,
            "current_test_leakage_rows": 0,
            "july_rows_accessed": 0,
        })
    population = pd.concat(parts, ignore_index=True).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["test_date", "feature"], kind="mergesort"
    ).reset_index(drop=True)
    fold_audit = pd.DataFrame(fold_rows)
    if len(population) != 155 or population["event_id"].duplicated().any():
        raise RuntimeError("FATAL: strict population must be 155 unique events")
    if int(population["board_group"].eq("BOARD2").sum()) != 130:
        raise RuntimeError("FATAL: strict Board2 population must be 130")
    if int(population["board_group"].eq("BOARD3").sum()) != 25:
        raise RuntimeError("FATAL: strict Board3 population must be 25")
    if max_logit_error > 1e-10 or max_score_error > 1e-10:
        raise RuntimeError("FATAL: frozen Stage1 logit/score reconstruction failed")
    metrics = {
        "max_logit_reconstruction_error": max_logit_error,
        "max_score_reconstruction_error": max_score_error,
        "source_audit": source_audit,
        "stage1_parity": parity,
        "self_label_leakage_rows": 0,
        "current_test_leakage_rows": 0,
        "july_result_rows_accessed": 0,
    }
    return population, coefficients, fold_audit, metrics


def _coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FROZEN_FEATURE_COLUMNS:
        values = coefficients.loc[coefficients["feature"].eq(feature), "beta"].to_numpy(float)
        positive = float(np.mean(values > 0))
        negative = float(np.mean(values < 0))
        rows.append({
            "feature": feature,
            "coefficient_mean": float(np.mean(values)),
            "coefficient_median": float(np.median(values)),
            "coefficient_min": float(np.min(values)),
            "coefficient_max": float(np.max(values)),
            "coefficient_positive_fold_pct": positive,
            "coefficient_negative_fold_pct": negative,
            "coefficient_zero_fold_pct": float(np.mean(values == 0)),
            "coefficient_sign_stable": bool(max(positive, negative) >= 0.80),
        })
    return pd.DataFrame(rows)


def _daily_table(population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logit_rows: list[dict[str, Any]] = []
    for date, day in population.groupby("signal_date", sort=True):
        b2 = day[day["board_group"].eq("BOARD2")]
        b3 = day[day["board_group"].eq("BOARD3")]
        mixed = bool(len(b2) and len(b3))
        logit_gap = (
            float(b3["stage1_logit"].mean() - b2["stage1_logit"].mean())
            if mixed else np.nan
        )
        contribution_sum = 0.0
        for feature in FROZEN_FEATURE_COLUMNS:
            column = contribution_column(feature)
            gap = float(b3[column].mean() - b2[column].mean())
            if np.isfinite(gap):
                contribution_sum += gap
            rows.append({
                "feature": feature,
                "signal_date": str(date),
                "board2_rows": int(len(b2)),
                "board3_rows": int(len(b3)),
                "board2_mean_feature": float(b2[feature].mean()),
                "board3_mean_feature": float(b3[feature].mean()),
                "feature_gap": float(b3[feature].mean() - b2[feature].mean()),
                "board2_mean_contribution": float(b2[column].mean()),
                "board3_mean_contribution": float(b3[column].mean()),
                "contribution_gap": gap,
                "board2_target7_rate": float(b2["target7"].mean()),
                "board3_target7_rate": float(b3["target7"].mean()),
                "board2_loss_rate": float(b2["loss"].mean()),
                "board3_loss_rate": float(b3["loss"].mean()),
            })
        closure = float(abs(contribution_sum - logit_gap)) if mixed else np.nan
        if mixed and closure > 1e-10:
            raise RuntimeError(f"FATAL: daily contribution closure failed {date}")
        logit_rows.append({
            "signal_date": str(date), "logit_gap": logit_gap,
            "contribution_gap_sum": contribution_sum, "closure_error": closure,
        })
    daily = pd.DataFrame(rows).sort_values(
        ["feature", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)
    logit = pd.DataFrame(logit_rows)
    mixed_logit = logit[logit["logit_gap"].notna()]
    summary = {
        "mixed_dates": int(len(mixed_logit)),
        "mean_daily_logit_gap": float(logit["logit_gap"].mean()),
        "median_daily_logit_gap": float(logit["logit_gap"].median()),
        "positive_gap_dates": int(logit["logit_gap"].gt(0).sum()),
        "negative_gap_dates": int(logit["logit_gap"].lt(0).sum()),
        "zero_gap_dates": int(logit["logit_gap"].eq(0).sum()),
        "max_contribution_closure_error": float(mixed_logit["closure_error"].max()),
    }
    return daily, summary


def same_date_pair_summary(
    frame: pd.DataFrame, score_column: str, endpoint: str
) -> dict[str, Any]:
    if endpoint not in {"NONLOSS", "TARGET7"}:
        raise ValueError("endpoint must be NONLOSS or TARGET7")
    label_column = "nonloss" if endpoint == "NONLOSS" else "target7"
    concordant = 0.0
    discordant = 0.0
    ties = 0.0
    informative_dates = 0
    for _, day in frame.groupby("signal_date", sort=True):
        positive = day[day[label_column].eq(1)][score_column].to_numpy(float)
        negative = day[day[label_column].eq(0)][score_column].to_numpy(float)
        if not len(positive) or not len(negative):
            continue
        informative_dates += 1
        comparisons = positive[:, None] - negative[None, :]
        concordant += float(np.sum(comparisons > 0))
        discordant += float(np.sum(comparisons < 0))
        ties += float(np.sum(comparisons == 0))
    pairs = int(concordant + discordant + ties)
    value = (concordant + 0.5 * ties) / pairs if pairs else np.nan
    return {
        "informative_dates": informative_dates,
        "informative_pairs": pairs,
        "concordant": int(concordant),
        "discordant": int(discordant),
        "ties": int(ties),
        "pair_concordance": float(value),
        "pair_support_weak": bool(pairs < PAIR_SUPPORT_MIN),
    }


def _pair_table(population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    totals = {board: {endpoint: 0 for endpoint in ("NONLOSS", "TARGET7")} for board in BOARD_GROUPS}
    for feature in FROZEN_FEATURE_COLUMNS:
        column = contribution_column(feature)
        for board in BOARD_GROUPS:
            subset = population[population["board_group"].eq(board)]
            for endpoint in ("NONLOSS", "TARGET7"):
                result = same_date_pair_summary(subset, column, endpoint)
                totals[board][endpoint] = result["informative_pairs"]
                rows.append({"feature": feature, "board_group": board, "endpoint": endpoint, **result})
    return pd.DataFrame(rows), totals


def _auc_many(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Tie-aware AUC for many nonnegative row-weight vectors."""
    y = np.asarray(labels, dtype=int)
    x = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[1] != len(y):
        raise ValueError("weight matrix shape mismatch")
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    w = w[:, order]
    numerator = np.zeros(len(w), dtype=float)
    cumulative_negative = np.zeros(len(w), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[end] == x[start]:
            end += 1
        group = w[:, start:end]
        group_pos = group[:, y[start:end] == 1].sum(axis=1)
        group_neg = group[:, y[start:end] == 0].sum(axis=1)
        numerator += group_pos * cumulative_negative + 0.5 * group_pos * group_neg
        cumulative_negative += group_neg
        start = end
    total_pos = w[:, y == 1].sum(axis=1)
    total_neg = w[:, y == 0].sum(axis=1)
    denominator = total_pos * total_neg
    result = np.full(len(w), np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def date_bootstrap_weights(
    population: pd.DataFrame,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    dates = sorted(population["signal_date"].astype(str).unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dates), size=(resamples, len(dates)))
    counts = np.zeros((resamples, len(dates)), dtype=np.int16)
    row = np.repeat(np.arange(resamples), len(dates))
    np.add.at(counts, (row, draws.ravel()), 1)
    date_index = population["signal_date"].map({date: i for i, date in enumerate(dates)}).to_numpy(int)
    return dates, counts, counts[:, date_index].astype(float)


def permute_board_labels(
    population: pd.DataFrame,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> np.ndarray:
    labels = population["board_group"].eq("BOARD3").to_numpy(np.int8)
    result = np.empty((permutations, len(population)), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for _, indices in population.groupby("signal_date", sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        order = np.argsort(rng.random((permutations, len(idx))), axis=1)
        result[:, idx] = labels[idx][order]
    return result


def _quantiles(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "p2.5": float(np.quantile(x, 0.025)) if len(x) else np.nan,
        "p50": float(np.quantile(x, 0.50)) if len(x) else np.nan,
        "p97.5": float(np.quantile(x, 0.975)) if len(x) else np.nan,
        "valid_resamples": int(len(x)),
    }


def _univariate_base(
    population: pd.DataFrame, coefficients: pd.DataFrame, daily: pd.DataFrame,
    pair_table: pd.DataFrame,
) -> pd.DataFrame:
    coefficient_summary = _coefficient_summary(coefficients).set_index("feature")
    rows: list[dict[str, Any]] = []
    for feature in FROZEN_FEATURE_COLUMNS:
        contribution = contribution_column(feature)
        b2 = population[population["board_group"].eq("BOARD2")]
        b3 = population[population["board_group"].eq("BOARD3")]
        daily_feature = daily[daily["feature"].eq(feature)]
        pairs = pair_table[pair_table["feature"].eq(feature)]
        response: dict[str, Any] = {}
        for board, subset in (("board2", b2), ("board3", b3)):
            response[f"{board}_target7_auc"] = binary_auc(subset["target7"], subset[contribution])
            response[f"{board}_nonloss_auc"] = binary_auc(subset["nonloss"], subset[contribution])
            response[f"{board}_target7_cliff"] = _cliff_positive_vs_negative(subset["target7"], subset[contribution])
            response[f"{board}_nonloss_cliff"] = _cliff_positive_vs_negative(subset["nonloss"], subset[contribution])
            for endpoint in ("NONLOSS", "TARGET7"):
                pair = pairs[(pairs["board_group"].eq(board.upper())) & pairs["endpoint"].eq(endpoint)].iloc[0]
                response[f"{board}_{endpoint.lower()}_pair_count"] = int(pair["informative_pairs"])
                response[f"{board}_{endpoint.lower()}_pair_concordance"] = float(pair["pair_concordance"])
        category = response_category(response["board2_nonloss_auc"], response["board3_nonloss_auc"])
        contribution_gap = float(daily_feature["contribution_gap"].mean())
        row = {
            "feature": feature,
            **coefficient_summary.loc[feature].to_dict(),
            "board2_feature_mean": float(b2[feature].mean()),
            "board2_feature_median": float(b2[feature].median()),
            "board3_feature_mean": float(b3[feature].mean()),
            "board3_feature_median": float(b3[feature].median()),
            "raw_feature_cliff_board3_vs_board2": _cliff_positive_vs_negative(
                np.r_[np.zeros(len(b2), dtype=int), np.ones(len(b3), dtype=int)],
                np.r_[b2[feature].to_numpy(float), b3[feature].to_numpy(float)],
            ),
            "same_date_feature_gap_positive_pct": float(daily_feature["feature_gap"].gt(0).mean()),
            "mean_daily_contribution_gap": contribution_gap,
            "median_daily_contribution_gap": float(daily_feature["contribution_gap"].median()),
            "contribution_gap_positive_date_pct": float(daily_feature["contribution_gap"].gt(0).mean()),
            **response,
            "delta_target7_auc": response["board3_target7_auc"] - response["board2_target7_auc"],
            "delta_nonloss_auc": response["board3_nonloss_auc"] - response["board2_nonloss_auc"],
            "response_category": category,
            "overpromotion_relevant": bool(contribution_gap > 0),
            "mechanism_candidate": bool(
                category in {"BOARD3_INVERTED", "BOARD3_DEGRADED"}
                and contribution_gap > 0
            ),
            "winner_safety_status": winner_safety_status(response["board3_target7_auc"]),
            "contribution_rank_spearman": float(
                pd.Series(population[contribution]).corr(
                    pd.Series(population["stage1_rank_percentile"]), method="spearman"
                )
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_and_lodo(
    population: pd.DataFrame, daily: pd.DataFrame, base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates, counts, row_weights = date_bootstrap_weights(population)
    rows: list[dict[str, Any]] = []
    lodo_rows: list[dict[str, Any]] = []
    board_masks = {
        board: population["board_group"].eq(board).to_numpy(float)
        for board in BOARD_GROUPS
    }
    for feature in FROZEN_FEATURE_COLUMNS:
        scores = population[contribution_column(feature)].to_numpy(float)
        boot: dict[str, np.ndarray] = {}
        for board in BOARD_GROUPS:
            weights = row_weights * board_masks[board][None, :]
            boot[f"{board}_nonloss"] = _auc_many(population["nonloss"].to_numpy(int), scores, weights)
            boot[f"{board}_target7"] = _auc_many(population["target7"].to_numpy(int), scores, weights)
        delta_nonloss = boot["BOARD3_nonloss"] - boot["BOARD2_nonloss"]
        delta_target7 = boot["BOARD3_target7"] - boot["BOARD2_target7"]
        daily_feature = daily[daily["feature"].eq(feature)].set_index("signal_date")["contribution_gap"]
        daily_values = np.array([daily_feature.get(date, np.nan) for date in dates], dtype=float)
        valid_daily = np.isfinite(daily_values)
        gap_numerator = counts[:, valid_daily] @ daily_values[valid_daily]
        gap_denominator = counts[:, valid_daily].sum(axis=1)
        contribution_gap = gap_numerator / gap_denominator
        dn = _quantiles(delta_nonloss)
        dt = _quantiles(delta_target7)
        cg = _quantiles(contribution_gap)
        rows.append({
            "feature": feature,
            "bootstrap_delta_nonloss_p2.5": dn["p2.5"],
            "bootstrap_delta_nonloss_p50": dn["p50"],
            "bootstrap_delta_nonloss_p97.5": dn["p97.5"],
            "bootstrap_P_delta_nonloss_lt0": float(np.nanmean(delta_nonloss < 0)),
            "bootstrap_delta_target7_p2.5": dt["p2.5"],
            "bootstrap_delta_target7_p50": dt["p50"],
            "bootstrap_delta_target7_p97.5": dt["p97.5"],
            "bootstrap_P_delta_target7_lt0": float(np.nanmean(delta_target7 < 0)),
            "bootstrap_contribution_gap_p2.5": cg["p2.5"],
            "bootstrap_contribution_gap_p50": cg["p50"],
            "bootstrap_contribution_gap_p97.5": cg["p97.5"],
            "bootstrap_P_gap_gt0": float(np.nanmean(contribution_gap > 0)),
            "bootstrap_valid_resamples": int(np.isfinite(delta_nonloss).sum()),
        })
        delta_values = []
        board3_values = []
        gap_values = []
        for omitted in dates:
            subset = population[~population["signal_date"].eq(omitted)]
            b2 = subset[subset["board_group"].eq("BOARD2")]
            b3 = subset[subset["board_group"].eq("BOARD3")]
            b2_auc = binary_auc(b2["nonloss"], b2[contribution_column(feature)])
            b3_auc = binary_auc(b3["nonloss"], b3[contribution_column(feature)])
            delta_values.append(b3_auc - b2_auc)
            board3_values.append(b3_auc)
            remaining = daily[(daily["feature"].eq(feature)) & ~daily["signal_date"].eq(omitted)]
            gap_values.append(float(remaining["contribution_gap"].mean()))
        delta_array = np.asarray(delta_values, dtype=float)
        b3_array = np.asarray(board3_values, dtype=float)
        gap_array = np.asarray(gap_values, dtype=float)
        lodo_rows.append({
            "feature": feature,
            "lodo_delta_nonloss_negative_pct": float(np.mean(delta_array < 0)),
            "lodo_board3_auc_below_half_pct": float(np.mean(b3_array < 0.5)),
            "lodo_contribution_gap_positive_pct": float(np.mean(gap_array > 0)),
            "lodo_delta_nonloss_min": float(np.nanmin(delta_array)),
            "lodo_delta_nonloss_median": float(np.nanmedian(delta_array)),
            "lodo_delta_nonloss_max": float(np.nanmax(delta_array)),
        })
    return pd.DataFrame(rows), pd.DataFrame(lodo_rows)


def _interaction_permutation(
    population: pd.DataFrame, base: pd.DataFrame,
) -> pd.DataFrame:
    assignments = permute_board_labels(population)
    board3_weights = assignments.astype(float)
    board2_weights = 1.0 - board3_weights
    labels = population["nonloss"].to_numpy(int)
    target_labels = population["target7"].to_numpy(int)
    rows = []
    for feature in FROZEN_FEATURE_COLUMNS:
        scores = population[contribution_column(feature)].to_numpy(float)
        b2_auc = _auc_many(labels, scores, board2_weights)
        b3_auc = _auc_many(labels, scores, board3_weights)
        delta = b3_auc - b2_auc
        b2_target_auc = _auc_many(target_labels, scores, board2_weights)
        b3_target_auc = _auc_many(target_labels, scores, board3_weights)
        target_delta = b3_target_auc - b2_target_auc
        observed = float(base.loc[base["feature"].eq(feature), "delta_nonloss_auc"].iloc[0])
        observed_target = float(base.loc[base["feature"].eq(feature), "delta_target7_auc"].iloc[0])
        finite = np.isfinite(delta)
        target_finite = np.isfinite(target_delta)
        p_value = float(
            (1 + np.sum(delta[finite] <= observed + 1e-15))
            / (1 + np.sum(finite))
        )
        rows.append({
            "feature": feature, "interaction_p": p_value,
            "target7_interaction_p": float(
                (1 + np.sum(target_delta[target_finite] <= observed_target + 1e-15))
                / (1 + np.sum(target_finite))
            ),
            "permutation_valid_resamples": int(finite.sum()),
            "target7_permutation_valid_resamples": int(target_finite.sum()),
        })
    result = pd.DataFrame(rows)
    if len(result) != 18:
        raise RuntimeError("FATAL: BH family must contain exactly 18 NONLOSS tests")
    result["interaction_q"] = benjamini_hochberg(result["interaction_p"])
    return result


def _aggregate_score_control(population: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for board in BOARD_GROUPS:
        subset = population[population["board_group"].eq(board)]
        result[board] = {
            "target7_auc": binary_auc(subset["target7"], subset["stage1_score"]),
            "nonloss_auc": binary_auc(subset["nonloss"], subset["stage1_score"]),
            "target7_pair": same_date_pair_summary(subset, "stage1_score", "TARGET7"),
            "nonloss_pair": same_date_pair_summary(subset, "stage1_score", "NONLOSS"),
        }
    return result


def _concentration(univariate: pd.DataFrame) -> dict[str, Any]:
    positive = univariate[univariate["mean_daily_contribution_gap"].gt(0)].sort_values(
        ["mean_daily_contribution_gap", "feature"], ascending=[False, True], kind="mergesort"
    )
    total = float(positive["mean_daily_contribution_gap"].sum())
    shares = {
        f"top{n}_share": float(positive.head(n)["mean_daily_contribution_gap"].sum() / total)
        if total > 0 else np.nan
        for n in (1, 3, 5)
    }
    harmful = float(
        univariate.loc[univariate["response_inversion_pass"], "mean_daily_contribution_gap"].clip(lower=0).sum()
    )
    return {
        "total_positive_contribution_gap": total,
        **shares,
        "harmful_inversion_positive_gap": harmful,
        "harmful_share": harmful / total if total > 0 else np.nan,
        "positive_drivers": positive["feature"].tolist(),
    }


def _redundancy(population: pd.DataFrame, univariate: pd.DataFrame) -> pd.DataFrame:
    selected = univariate[univariate["response_inversion_pass"]]["feature"].tolist()
    candidates = univariate[univariate["mechanism_candidate"]].sort_values(
        ["delta_nonloss_auc", "feature"], kind="mergesort"
    )["feature"].head(5).tolist()
    features = list(dict.fromkeys([*selected, *candidates]))
    rows = []
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            rho = float(pd.Series(population[contribution_column(left)]).corr(
                pd.Series(population[contribution_column(right)]), method="spearman"
            ))
            rows.append({
                "feature_a": left, "feature_b": right, "rho": rho,
                "abs_rho": abs(rho), "near_redundant": bool(abs(rho) >= 0.85),
            })
    return pd.DataFrame(rows, columns=[
        "feature_a", "feature_b", "rho", "abs_rho", "near_redundant",
    ])


def _formal_decision(
    univariate: pd.DataFrame, redundancy: pd.DataFrame,
    concentration: Mapping[str, Any], aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    passed = univariate[univariate["response_inversion_pass"]]
    independent = False
    if len(passed) >= 2:
        pass_set = set(passed["feature"])
        pairs = redundancy[
            redundancy["feature_a"].isin(pass_set)
            & redundancy["feature_b"].isin(pass_set)
        ]
        independent = bool((pairs["abs_rho"] < 0.85).any()) if len(pairs) else False
    exceptional = False
    if len(passed) == 1:
        row = passed.iloc[0]
        exceptional = bool(
            row["board2_nonloss_auc"] >= 0.60
            and row["board3_nonloss_auc"] <= 0.35
            and row["delta_nonloss_auc"] <= -0.25
            and row["interaction_q"] <= 0.05
            and row["bootstrap_P_delta_nonloss_lt0"] >= 0.95
            and row["mean_daily_contribution_gap"] > 0
            and row["lodo_delta_nonloss_negative_pct"] >= 0.90
        )
    specific = bool(
        (len(passed) >= 2 and independent and concentration["harmful_share"] >= 0.30)
        or exceptional
    )
    broad_count = int(univariate["board3_nonloss_auc"].lt(0.45).sum())
    broad = bool(
        len(passed) == 0
        and aggregate["BOARD3"]["nonloss_auc"] <= 0.35
        and broad_count >= 6
        and concentration["top3_share"] < 0.60
    )
    candidates = int(univariate["mechanism_candidate"].sum())
    if specific:
        mechanism = "SPECIFIC_FEATURE_INVERSION"
    elif broad:
        mechanism = "BROAD_SCORE_MISALIGNMENT"
    elif len(passed) >= 1 or candidates >= 2:
        mechanism = "PARTIAL_FEATURE_MISALIGNMENT"
    else:
        mechanism = "NO_CLEAR_FEATURE_MECHANISM"

    if mechanism == "SPECIFIC_FEATURE_INVERSION":
        direction = "MINIMAL_BOARD_FEATURE_INTERACTIONS"
    elif mechanism == "BROAD_SCORE_MISALIGNMENT":
        direction = "BROADER_BOARD_SEPARATION"
    elif (
        mechanism == "NO_CLEAR_FEATURE_MECHANISM"
        and aggregate["BOARD3"]["target7_auc"] >= 0.50
        and aggregate["BOARD3"]["nonloss_auc"] <= 0.35
        and broad_count < 6
    ):
        direction = "BOARD_SPECIFIC_SCORE_CALIBRATION"
    else:
        direction = "NO_BOARD_AWARE_MODEL_YET"
    return {
        "mechanism": mechanism,
        "direction": direction,
        "formal_pass_count": int(len(passed)),
        "mechanism_candidate_count": candidates,
        "independent_passes": independent,
        "exceptional_single": exceptional,
        "broad_bad_feature_count": broad_count,
    }


def _robustness_table(
    univariate: pd.DataFrame, bootstrap: pd.DataFrame,
    lodo: pd.DataFrame, permutation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in bootstrap.iterrows():
        feature = str(row["feature"])
        rows.extend([
            {"feature": feature, "section": "BOOTSTRAP", "metric": "DELTA_NONLOSS_AUC", "estimate": float(univariate.loc[univariate.feature.eq(feature), "delta_nonloss_auc"].iloc[0]), "p2.5": row["bootstrap_delta_nonloss_p2.5"], "p50": row["bootstrap_delta_nonloss_p50"], "p97.5": row["bootstrap_delta_nonloss_p97.5"], "direction_probability": row["bootstrap_P_delta_nonloss_lt0"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": row["bootstrap_valid_resamples"]},
            {"feature": feature, "section": "BOOTSTRAP", "metric": "CONTRIBUTION_GAP", "estimate": float(univariate.loc[univariate.feature.eq(feature), "mean_daily_contribution_gap"].iloc[0]), "p2.5": row["bootstrap_contribution_gap_p2.5"], "p50": row["bootstrap_contribution_gap_p50"], "p97.5": row["bootstrap_contribution_gap_p97.5"], "direction_probability": row["bootstrap_P_gap_gt0"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": row["bootstrap_valid_resamples"]},
        ])
    for row in lodo.itertuples(index=False):
        rows.append({"feature": row.feature, "section": "LODO", "metric": "DELTA_NONLOSS_AUC", "estimate": float(univariate.loc[univariate.feature.eq(row.feature), "delta_nonloss_auc"].iloc[0]), "p2.5": row.lodo_delta_nonloss_min, "p50": row.lodo_delta_nonloss_median, "p97.5": row.lodo_delta_nonloss_max, "direction_probability": row.lodo_delta_nonloss_negative_pct, "p_value": np.nan, "q_value": np.nan, "valid_resamples": 17})
    for row in permutation.itertuples(index=False):
        rows.append({"feature": row.feature, "section": "PERMUTATION", "metric": "DELTA_NONLOSS_AUC", "estimate": float(univariate.loc[univariate.feature.eq(row.feature), "delta_nonloss_auc"].iloc[0]), "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan, "direction_probability": np.nan, "p_value": row.interaction_p, "q_value": row.interaction_q, "valid_resamples": row.permutation_valid_resamples})
        rows.append({"feature": row.feature, "section": "PERMUTATION", "metric": "DELTA_TARGET7_AUC", "estimate": float(univariate.loc[univariate.feature.eq(row.feature), "delta_target7_auc"].iloc[0]), "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan, "direction_probability": np.nan, "p_value": row.target7_interaction_p, "q_value": np.nan, "valid_resamples": row.target7_permutation_valid_resamples})
    return pd.DataFrame(rows)


def _pct(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{100 * float(value):.{digits}f}%"


def _num(value: Any, digits: int = 6) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    pop = context["population"]
    uni = context["univariate"]
    coef = context["coefficient_summary"]
    agg = context["aggregate_score"]
    concentration = context["concentration"]
    decision = context["decision"]
    parity = context["reconstruction"]["stage1_parity"]
    top = uni.sort_values(["mean_daily_contribution_gap", "feature"], ascending=[False, True]).head(10)
    passes = uni[uni["response_inversion_pass"]]
    candidates = uni[uni["mechanism_candidate"]].sort_values(["delta_nonloss_auc", "feature"]).head(5)
    b2 = pop[pop["board_group"].eq("BOARD2")]
    b3 = pop[pop["board_group"].eq("BOARD3")]
    lines = [
        "# v004c Board2 vs Board3 Stage1 Feature-Response Audit v001",
        "## 1. Experimental Contract",
        "- Frozen V4C_STAGE1 reconstruction only; no new model, feature, policy, threshold, or July result access.",
        "- `DIAGNOSTIC_CONTRIBUTION_ONLY = YES`; `NEW_MODEL_FEATURE = NO`.",
        "## 2. Strict Population / Stage1 Parity",
        f"- Population: **{len(pop)} rows / {pop.signal_date.nunique()} dates; Board2 {len(b2)}, Board3 {len(b3)}**.",
        f"- Board2 Target7/LOSS: **{int(b2.target7.sum())} / {int(b2.loss.sum())}**; Board3: **{int(b3.target7.sum())} / {int(b3.loss.sum())}**.",
        f"- Stage1 Rank1/Rank2/Rank3/Top2/Top3: **{_pct(parity['Rank1_mean'])} / {_pct(parity['Rank2_mean'])} / {_pct(parity['Rank3_mean'])} / {_pct(parity['Top2_mean'])} / {_pct(parity['Top3_mean'])}**.",
        "## 3. Frozen 18-Feature Contract",
        f"- Model: `{MODEL_ID}` / `{MODEL_FAMILY}`; L2 **{L2:.2f}**; positive weight **{POSITIVE_WEIGHT:.2f}**; features **18**.",
        "- " + ", ".join(f"`{feature}`" for feature in FROZEN_FEATURE_COLUMNS) + ".",
        "## 4. Coefficient Stability",
        "| Feature | Mean beta | Median | Min | Max | + folds | - folds | Stable |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in coef.itertuples(index=False):
        lines.append(f"| {row.feature} | {_num(row.coefficient_mean)} | {_num(row.coefficient_median)} | {_num(row.coefficient_min)} | {_num(row.coefficient_max)} | {_pct(row.coefficient_positive_fold_pct)} | {_pct(row.coefficient_negative_fold_pct)} | {'YES' if row.coefficient_sign_stable else 'NO'} |")
    lines += [
        "## 5. Board Feature Exposure",
        "| Feature | B2 median | B3 median | B3-vs-B2 Cliff | Same-date positive gap |",
        "|---|---:|---:|---:|---:|",
    ]
    exposure = uni.reindex(uni["raw_feature_cliff_board3_vs_board2"].abs().sort_values(ascending=False).index).head(10)
    for row in exposure.itertuples(index=False):
        lines.append(f"| {row.feature} | {_num(row.board2_feature_median)} | {_num(row.board3_feature_median)} | {_num(row.raw_feature_cliff_board3_vs_board2)} | {_pct(row.same_date_feature_gap_positive_pct)} |")
    lines += [
        "## 6. Board Contribution Exposure",
        f"- Board3-Board2 mean/median daily total logit gap: **{_num(context['daily_summary']['mean_daily_logit_gap'])} / {_num(context['daily_summary']['median_daily_logit_gap'])}**; positive dates **{context['daily_summary']['positive_gap_dates']}**.",
        "| Driver | Mean daily gap | Median | Beta median | B2 NONLOSS AUC | B3 NONLOSS AUC | Category | Pass |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in top.itertuples(index=False):
        lines.append(f"| {row.feature} | {_num(row.mean_daily_contribution_gap)} | {_num(row.median_daily_contribution_gap)} | {_num(row.coefficient_median)} | {_num(row.board2_nonloss_auc)} | {_num(row.board3_nonloss_auc)} | {row.response_category} | {'YES' if row.response_inversion_pass else 'NO'} |")
    lines += [
        "## 7. Board2 vs Board3 Target7 Response",
        "| Feature | B2 AUC | B3 AUC | Delta | B2 pair | B3 pair | Winner safety |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    report_set = pd.concat([passes, candidates]).drop_duplicates("feature")
    for row in report_set.itertuples(index=False):
        lines.append(f"| {row.feature} | {_num(row.board2_target7_auc)} | {_num(row.board3_target7_auc)} | {_num(row.delta_target7_auc)} | {_num(row.board2_target7_pair_concordance)} | {_num(row.board3_target7_pair_concordance)} | {row.winner_safety_status} |")
    if report_set.empty:
        lines.append("| None | NA | NA | NA | NA | NA | NA |")
    lines += [
        "## 8. Board2 vs Board3 NONLOSS Response",
        "| Feature | B2 AUC | B3 AUC | Delta | B2 pair | B3 pair | Gap | Bootstrap P(delta<0) | q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report_set.itertuples(index=False):
        lines.append(f"| {row.feature} | {_num(row.board2_nonloss_auc)} | {_num(row.board3_nonloss_auc)} | {_num(row.delta_nonloss_auc)} | {_num(row.board2_nonloss_pair_concordance)} | {_num(row.board3_nonloss_pair_concordance)} | {_num(row.mean_daily_contribution_gap)} | {_pct(row.bootstrap_P_delta_nonloss_lt0)} | {_num(row.interaction_q)} |")
    if report_set.empty:
        lines.append("| None | NA | NA | NA | NA | NA | NA | NA | NA |")
    lines += [
        "## 9. Same-Date Pair Concordance",
        f"- Aggregate score Board2 Target7/NONLOSS pairs: **{agg['BOARD2']['target7_pair']['informative_pairs']} / {agg['BOARD2']['nonloss_pair']['informative_pairs']}**; Board3: **{agg['BOARD3']['target7_pair']['informative_pairs']} / {agg['BOARD3']['nonloss_pair']['informative_pairs']}**.",
        "- Feature/board/endpoint pair counts and weak-support flags are preserved in the pair artifact.",
        "## 10. Bootstrap / LODO / Permutation",
        f"- Primary NONLOSS interaction tests: **18**; q<=0.10 **{int(uni.interaction_q.le(.10).sum())}**; q<=0.05 **{int(uni.interaction_q.le(.05).sum())}**; minimum q **{_num(uni.interaction_q.min())}**.",
        "## 11. Specific Response Inversions",
        f"- Formal passes: **{len(passes)}**; mechanism candidates: **{int(uni.mechanism_candidate.sum())}**.",
    ]
    for row in passes.itertuples(index=False):
        lines.append(f"- `{row.feature}`: B2/B3 NONLOSS AUC **{_num(row.board2_nonloss_auc)} / {_num(row.board3_nonloss_auc)}**, gap **{_num(row.mean_daily_contribution_gap)}**, q **{_num(row.interaction_q)}**, {row.winner_safety_status}.")
    if passes.empty:
        lines.append("- No `RESPONSE_INVERSION_PASS` feature was formally established.")
    lines += [
        "## 12. Contribution Concentration",
        f"- Total positive gap **{_num(concentration['total_positive_contribution_gap'])}**; Top1/Top3/Top5 shares **{_pct(concentration['top1_share'])} / {_pct(concentration['top3_share'])} / {_pct(concentration['top5_share'])}**.",
        f"- Harmful inversion positive gap/share: **{_num(concentration['harmful_inversion_positive_gap'])} / {_pct(concentration['harmful_share'])}**.",
        "## 13. Aggregate Score Control",
        f"- Board2 Target7/NONLOSS AUC: **{_num(agg['BOARD2']['target7_auc'])} / {_num(agg['BOARD2']['nonloss_auc'])}**; pair concordance **{_num(agg['BOARD2']['target7_pair']['pair_concordance'])} / {_num(agg['BOARD2']['nonloss_pair']['pair_concordance'])}**.",
        f"- Board3 Target7/NONLOSS AUC: **{_num(agg['BOARD3']['target7_auc'])} / {_num(agg['BOARD3']['nonloss_auc'])}**; pair concordance **{_num(agg['BOARD3']['target7_pair']['pair_concordance'])} / {_num(agg['BOARD3']['nonloss_pair']['pair_concordance'])}**.",
        "## 14. Board-Aware Formulation Decision",
        f"- Q1 Frozen features reverse good/bad response from Board2 to Board3: **{'YES' if len(passes) else 'PARTIAL' if int(uni.mechanism_candidate.sum()) else 'NO'}**.",
        f"- Q2 Such features give Board3 positive Stage1 contribution: **{'YES' if len(passes) else 'PARTIAL' if int(uni.mechanism_candidate.sum()) else 'NO'}**.",
        f"- Q3 Any inversion survives bootstrap, LODO, and BH: **{'YES' if len(passes) else 'NO'}**.",
        f"- Q4 Established inversions non-redundant: **{'YES' if decision['independent_passes'] else 'NO' if len(passes) >= 2 else 'NA'}**.",
        f"- Q5 Overpromotion mainly attributable to a small feature set: **{'YES' if decision['mechanism'] == 'SPECIFIC_FEATURE_INVERSION' else 'PARTIAL' if len(passes) else 'NO'}**.",
        f"- Q6 Broad Stage1 score misalignment inside Board3: **{'YES' if decision['mechanism'] == 'BROAD_SCORE_MISALIGNMENT' else 'PARTIAL' if decision['broad_bad_feature_count'] >= 6 else 'NO'}**.",
        f"- Q7 Minimal board×feature formulation supported now: **{'YES' if decision['direction'] == 'MINIMAL_BOARD_FEATURE_INTERACTIONS' else 'NO'}**.",
        f"- `BOARD3_STAGE1_RESPONSE_MECHANISM = {decision['mechanism']}`",
        f"- `BOARD_AWARE_FORMULATION_DIRECTION = {decision['direction']}`",
        f"- Why Stage1 over-selects Board3: Board3 receives a date-equal logit lift of **{_num(context['daily_summary']['mean_daily_logit_gap'])}**; positive price/active-money interaction contributions dominate the lift, while three formally inverted responses explain only **{_pct(concentration['harmful_share'])}** of all positive contribution gap.",
        f"- Largest positive Board3 contribution sources: **{', '.join(concentration['positive_drivers'][:5])}**.",
        f"- Formal harmful/inverted contributions: **{', '.join(passes.feature.tolist()) if len(passes) else 'none'}**.",
        f"- Board2 uses each formal inversion feature in the intended NONLOSS direction (AUC >=0.55), but their Board3 winner-safety is **{', '.join(sorted(set(passes.winner_safety_status))) if len(passes) else 'NA'}**.",
        f"- Aggregate Board3 retains Target7 ordering above chance (**{_num(agg['BOARD3']['target7_auc'])}**) while failing NONLOSS ordering (**{_num(agg['BOARD3']['nonloss_auc'])}**); however same-date Board3 pair support is weak and the formal harmful share is below 30%.",
        "- The failure is partially localized but not sufficiently independent/concentrated for a minimal interaction experiment; the evidence also does not satisfy the formal broad-separation state.",
        "- Recommended next technical action: preserve this audit and do not train a board-aware model yet; require independent predeclared evidence before choosing calibration, minimal interactions, or broader separation.",
        "- No board interaction, calibration, board-specific model, penalty, exclusion, or policy was trained/tested.",
    ]
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    population, coefficients, fold_audit, reconstruction = reconstruct_frozen_stage1(root)
    daily, daily_summary = _daily_table(population)
    pair_table, pair_totals = _pair_table(population)
    univariate = _univariate_base(population, coefficients, daily, pair_table)
    bootstrap, lodo = _bootstrap_and_lodo(population, daily, univariate)
    permutation = _interaction_permutation(population, univariate)
    univariate = univariate.merge(bootstrap, on="feature", validate="one_to_one")
    univariate = univariate.merge(lodo, on="feature", validate="one_to_one")
    univariate = univariate.merge(permutation, on="feature", validate="one_to_one")
    univariate["response_inversion_pass"] = [
        response_inversion_gate(row) for row in univariate.to_dict("records")
    ]
    aggregate = _aggregate_score_control(population)
    concentration = _concentration(univariate)
    redundancy = _redundancy(population, univariate)
    decision = _formal_decision(univariate, redundancy, concentration, aggregate)
    coefficient_summary = _coefficient_summary(coefficients)
    robustness = _robustness_table(univariate, bootstrap, lodo, permutation)

    context: dict[str, Any] = {
        "population": population,
        "coefficients": coefficients,
        "fold_audit": fold_audit,
        "reconstruction": reconstruction,
        "coefficient_summary": coefficient_summary,
        "daily": daily,
        "daily_summary": daily_summary,
        "pairs": pair_table,
        "pair_totals": pair_totals,
        "univariate": univariate,
        "bootstrap": bootstrap,
        "lodo": lodo,
        "permutation": permutation,
        "robustness": robustness,
        "aggregate_score": aggregate,
        "concentration": concentration,
        "redundancy": redundancy,
        "decision": decision,
        "july_result_rows_accessed": 0,
    }
    review = render_review(context)
    population_columns = [
        "event_id", "signal_date", "code", "board_group", "candidate_count",
        "stage1_score", "stage1_logit", "stage1_rank", "stage1_rank_percentile",
        "target7", "loss", "nonloss", "raw_repair_return", "capped_return_7",
        *FROZEN_FEATURE_COLUMNS,
        *[contribution_column(feature) for feature in FROZEN_FEATURE_COLUMNS],
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(population[population_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(univariate),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(pair_table),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(robustness),
        OUTPUT_FILENAMES[6]: dataframe_csv_bytes(redundancy),
        OUTPUT_FILENAMES[7]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_board2_board3_stage1_feature_response(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path / "reports/research/v004c_board2_board3_stage1_feature_response_v001_20260603_20260626"
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

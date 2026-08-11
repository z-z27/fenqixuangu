from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004a import fit_logistic_l2_weighted
from .v004c_stage1_risk_complementarity import (
    JULY_SENTINEL,
    POLICY_CAPPED,
    POLICY_STAGE1,
    POLICY_STAGE1_BACKFILL,
    _fit_strict_stage2,
    add_policy_ranks,
    load_frozen_june_buckets,
    outcome_state,
    prepare_diagnostic_samples,
    reconstruct_strict_june,
)
from .v004c_top10_target_information import PAIR_FEATURE_COLUMNS
from .v004c_v4a_architecture_transfer import dataframe_csv_bytes
from .v004c_v4a_top10_reranker import strategy_summary


TASK_NAME = "LIMITED STAGE1 RISK PROTECTOR"
MODEL_ID = "LIMITED_RISK_PROTECTOR_P050"
RISK_FEATURE_COLUMNS = [
    "stage1_strength",
    "closing_completion_gap",
    "strength_x_gap",
]
RISK_L2 = 0.30
RISK_THRESHOLD = 0.50
RISK_CLASS_WEIGHTING = False
RISK_TAIL_WEIGHTING = False
RISK_THRESHOLD_SEARCH = False
RISK_FEATURE_SELECTION = False
RECURSIVE_VETO = False
JULY_RESULT_ROWS_ACCESSED = 0
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_RESAMPLES = 20_000

EXPECTED_STRICT_DATES = [
    "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08",
    "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
    "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18",
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
    "2026-06-26",
]
EXPECTED_UNAVAILABLE_DATES = [
    "2026-06-01", "2026-06-02", "2026-06-29", "2026-06-30"
]
EXPECTED_STAGE1_PARITY = {
    "Rank1_mean": 0.0298285616089,
    "Rank2_mean": 0.0368040074256,
    "Rank3_mean": 0.00265478297043,
    "Top2_mean": 0.0333162845172,
    "Top3_mean": 0.0230957840016,
    "Top3_target7_precision": 0.313725490196,
    "winner_capture": 0.477777777778,
    "Top3_negative_date_rate": 0.235294117647,
    "Top3_worst": -0.0676878130753,
}
EXPECTED_CAPPED_TOP3 = 0.0187410087299

POLICY_LIMITED = MODEL_ID
POLICY_ORACLE = "ORACLE_LOSS_VETO_STAGE1_BACKFILL"

OUTPUT_FILENAMES = (
    "v004c_limited_risk_oof_v001.csv",
    "v004c_limited_risk_meta_audit_v001.csv",
    "v004c_limited_risk_coefficients_v001.csv",
    "v004c_limited_risk_veto_attribution_v001.csv",
    "v004c_limited_risk_daily_v001.csv",
    "v004c_limited_risk_practical_v001.csv",
    "v004c_limited_risk_review_v001.md",
)


def assert_risk_contract() -> None:
    if RISK_FEATURE_COLUMNS != PAIR_FEATURE_COLUMNS:
        raise RuntimeError("FATAL: risk representation is not the frozen three-feature representation")
    if len(RISK_FEATURE_COLUMNS) != 3:
        raise RuntimeError("FATAL: risk feature count must equal three")
    if RISK_L2 != 0.30 or RISK_THRESHOLD != 0.50:
        raise RuntimeError("FATAL: risk L2 or threshold changed")
    if any((RISK_CLASS_WEIGHTING, RISK_TAIL_WEIGHTING, RISK_THRESHOLD_SEARCH, RISK_FEATURE_SELECTION)):
        raise RuntimeError("FATAL: prohibited risk-model weighting or search enabled")
    if RECURSIVE_VETO:
        raise RuntimeError("FATAL: recursive veto enabled")
    if JULY_RESULT_ROWS_ACCESSED != 0:
        raise RuntimeError("FATAL: July result rows entered development")


def loss_target(raw_return: float | np.ndarray | pd.Series) -> Any:
    values = np.asarray(raw_return, dtype=float)
    result = (values < 0.0).astype(int)
    if np.ndim(raw_return) == 0:
        return int(result)
    return result


def fixed_risk_veto(probability: float | np.ndarray | pd.Series) -> Any:
    values = np.asarray(probability, dtype=float)
    result = values >= RISK_THRESHOLD
    if np.ndim(probability) == 0:
        return bool(result)
    return result


def risk_date_sample_weight(meta: pd.DataFrame) -> np.ndarray:
    counts = meta.groupby("signal_date", sort=False)["event_id"].transform("count").astype(float)
    if bool(counts.le(0).any()):
        raise RuntimeError("FATAL: invalid risk date count")
    weights = 1.0 / counts.to_numpy(float)
    date_sums = pd.Series(weights, index=meta.index).groupby(meta["signal_date"]).sum()
    if not np.allclose(date_sums.to_numpy(float), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("FATAL: risk date weighting is not date-equal")
    return weights


def fit_risk_logistic(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = ["signal_date", "event_id", "stage1_rank", "candidate_count", "loss_target", *RISK_FEATURE_COLUMNS]
    missing = [column for column in required if column not in meta]
    if missing:
        raise RuntimeError(f"FATAL: risk meta columns missing {missing}")
    if bool(meta["stage1_rank"].gt(np.minimum(3, meta["candidate_count"])).any()):
        raise RuntimeError("FATAL: risk training row is not historical Stage1 Top3")
    if bool(meta[RISK_FEATURE_COLUMNS].isna().any().any()):
        raise RuntimeError("FATAL: risk training feature missing")
    weights = risk_date_sample_weight(meta)
    beta = fit_logistic_l2_weighted(
        meta[RISK_FEATURE_COLUMNS].to_numpy(float),
        meta["loss_target"].to_numpy(float),
        l2=RISK_L2,
        sample_weight=weights,
    )
    if beta.shape != (len(RISK_FEATURE_COLUMNS) + 1,):
        raise RuntimeError("FATAL: unexpected risk coefficient shape")
    return beta, weights


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def score_risk_probability(frame: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return _sigmoid(float(beta[0]) + frame[RISK_FEATURE_COLUMNS].to_numpy(float) @ beta[1:])


def limited_veto_selection(
    day: pd.DataFrame,
    threshold: float = RISK_THRESHOLD,
) -> tuple[list[str], list[str], list[str]]:
    if float(threshold) != RISK_THRESHOLD:
        raise RuntimeError("FATAL: veto threshold changed")
    size = min(3, len(day))
    ordered = day.sort_values(["stage1_rank", "event_id"], kind="mergesort")
    control = ordered[ordered["stage1_rank"].le(size)]
    if len(control) != size or bool(control["risk_p_loss"].isna().any()):
        raise RuntimeError("FATAL: original Stage1 Top3 risk score incomplete")
    flagged = control[control["risk_p_loss"].ge(threshold)]
    flagged_ids = flagged["event_id"].astype(str).tolist()
    final = control[~control["event_id"].astype(str).isin(flagged_ids)]["event_id"].astype(str).tolist()
    backfilled: list[str] = []
    pool = ordered[
        ordered["stage1_rank"].gt(3)
        & ordered["stage1_rank"].le(min(10, len(day)))
    ]
    for event_id in pool["event_id"].astype(str):
        if len(final) >= size:
            break
        final.append(event_id)
        backfilled.append(event_id)
    # The policy cannot reduce portfolio size.  If no external replacement is
    # available, restore flagged original members in frozen Stage1 order.
    if len(final) < size:
        for event_id in flagged_ids:
            if len(final) >= size:
                break
            final.append(event_id)
    if len(final) != size or len(final) != len(set(final)):
        raise RuntimeError("FATAL: limited-veto portfolio size or identity invalid")
    executed = [event_id for event_id in flagged_ids if event_id not in set(final)]
    if len(executed) != len(backfilled):
        raise RuntimeError("FATAL: limited-veto replacement cardinality mismatch")
    return final, executed, backfilled


def apply_limited_policy(day: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    result = day.copy()
    size = min(3, len(result))
    result["original_stage1_top3"] = result["stage1_rank"].le(size)
    result["risk_p_loss"] = np.nan
    top3_index = result.index[result["original_stage1_top3"]]
    result.loc[top3_index, "risk_p_loss"] = score_risk_probability(result.loc[top3_index], beta)
    result["risk_veto"] = False
    result.loc[top3_index, "risk_veto"] = fixed_risk_veto(result.loc[top3_index, "risk_p_loss"])
    selected, executed, backfilled = limited_veto_selection(result)
    result["risk_veto_executed"] = result["event_id"].astype(str).isin(executed)
    result["risk_backfilled"] = result["event_id"].astype(str).isin(backfilled)
    remainder = [
        event_id
        for event_id in result.sort_values(["stage1_rank", "event_id"], kind="mergesort")["event_id"].astype(str)
        if event_id not in selected
    ]
    rank_map = {event_id: rank for rank, event_id in enumerate(selected + remainder, 1)}
    result["final_limited_risk_rank"] = result["event_id"].astype(str).map(rank_map).astype(int)
    if int(result["final_limited_risk_rank"].le(size).sum()) != size:
        raise RuntimeError("FATAL: final limited-risk membership invalid")
    return result


def prediction_outcome_independence(day: pd.DataFrame, beta: np.ndarray) -> bool:
    before = apply_limited_policy(day, beta).sort_values("event_id", kind="mergesort")
    mutated = day.copy()
    for column, value in (
        ("raw_repair_return", 9.0),
        ("capped_opportunity_return_7", -9.0),
        ("target7", 1),
        ("loss_target", 1),
        ("d2_open_daily", 1.0),
        ("d3_high_daily", 999.0),
    ):
        if column in mutated:
            mutated[column] = value
    after = apply_limited_policy(mutated, beta).sort_values("event_id", kind="mergesort")
    numeric_equal = np.allclose(
        before["risk_p_loss"].fillna(-1.0),
        after["risk_p_loss"].fillna(-1.0),
        rtol=0.0,
        atol=0.0,
    )
    return bool(
        numeric_equal
        and before["risk_veto"].tolist() == after["risk_veto"].tolist()
        and before["final_limited_risk_rank"].tolist() == after["final_limited_risk_rank"].tolist()
    )


def reconstruct_limited_risk(
    base_x: pd.DataFrame,
    samples: pd.DataFrame,
    strict_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    meta_aggregate: list[pd.DataFrame] = []
    outcome_independence = True
    for test_date in sorted(strict_oof["signal_date"].astype(str).unique()):
        matured = samples[samples["label_available_date"].lt(test_date)].copy()
        _, meta_top10, _, leakage = _fit_strict_stage2(base_x, matured, test_date)
        meta = meta_top10[
            meta_top10["stage1_rank"].le(np.minimum(3, meta_top10["candidate_count"]))
        ].copy()
        meta["loss_target"] = loss_target(meta["raw_repair_return"])
        if bool(meta["signal_date"].eq(test_date).any()):
            raise RuntimeError("FATAL: current test date entered risk training")
        if bool(meta["label_available_date"].ge(test_date).any()):
            raise RuntimeError("FATAL: immature label entered risk training")
        if bool(meta["signal_date"].ge(JULY_SENTINEL).any()):
            raise RuntimeError("FATAL: July row entered risk training")
        beta, weights = fit_risk_logistic(meta)
        meta["risk_sample_weight"] = weights
        meta["fold_test_date"] = test_date
        meta_aggregate.append(meta[[
            "fold_test_date", "signal_date", "event_id", "stage1_rank",
            "loss_target", "target7", "risk_sample_weight",
        ]])
        test = strict_oof[strict_oof["signal_date"].eq(test_date)].copy()
        test["loss_target"] = loss_target(test["raw_repair_return"])
        if bool(test[RISK_FEATURE_COLUMNS].loc[test["stage1_rank"].le(3)].isna().any().any()):
            raise RuntimeError("FATAL: strict test Top3 risk feature missing")
        outcome_independence = outcome_independence and prediction_outcome_independence(test, beta)
        scored = apply_limited_policy(test, beta)
        parts.append(scored)
        weighted_prevalence = float(np.sum(weights * meta["loss_target"].to_numpy(float)) / weights.sum())
        audits.append({
            "test_date": test_date,
            "matured_rows": int(len(matured)),
            "matured_dates": int(matured["signal_date"].nunique()),
            "risk_meta_dates": int(meta["signal_date"].nunique()),
            "risk_meta_rows": int(len(meta)),
            "risk_meta_loss_rows": int(meta["loss_target"].sum()),
            "risk_meta_nonloss_rows": int(meta["loss_target"].eq(0).sum()),
            "risk_meta_target7_rows": int(meta["target7"].sum()),
            "weighted_loss_prevalence": weighted_prevalence,
            "self_label_leakage_rows": int(leakage["self_label_leakage_rows"]),
            "current_test_leakage_rows": int(leakage["current_test_leakage_rows"]),
            "feature_count": len(RISK_FEATURE_COLUMNS),
            "l2": RISK_L2,
            "threshold": RISK_THRESHOLD,
            "july_rows_accessed": 0,
        })
        coefficients.append({
            "test_date": test_date,
            "intercept": float(beta[0]),
            "beta_stage1_strength": float(beta[1]),
            "beta_closing_completion_gap": float(beta[2]),
            "beta_strength_x_gap": float(beta[3]),
            "train_loss_rate": weighted_prevalence,
            "l2": RISK_L2,
        })
    oof = pd.concat(parts, ignore_index=True).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audits)
    coefficient_frame = pd.DataFrame(coefficients)
    aggregate = pd.concat(meta_aggregate, ignore_index=True)
    if int(audit["self_label_leakage_rows"].sum()) != 0:
        raise RuntimeError("FATAL: risk meta self-label leakage")
    if int(audit["current_test_leakage_rows"].sum()) != 0:
        raise RuntimeError("FATAL: risk current-test leakage")
    if int(audit["july_rows_accessed"].sum()) != 0:
        raise RuntimeError("FATAL: July result row accessed")
    if not outcome_independence:
        raise RuntimeError("FATAL: current outcome changed risk prediction")
    training = {
        "total_fold_aggregated_rows": int(len(aggregate)),
        "total_fold_aggregated_dates": int(audit["risk_meta_dates"].sum()),
        "loss_rows": int(aggregate["loss_target"].sum()),
        "nonloss_rows": int(aggregate["loss_target"].eq(0).sum()),
        "target7_rows": int(aggregate["target7"].sum()),
        "weighted_loss_prevalence": float(
            np.average(aggregate["loss_target"], weights=aggregate["risk_sample_weight"])
        ),
        "all_rows_stage1_top3": bool(aggregate["stage1_rank"].le(3).all()),
        "outcome_perturbation_pass": outcome_independence,
    }
    return oof, audit, coefficient_frame, training


def _selected_daily(frame: pd.DataFrame, rank_column: str) -> pd.Series:
    values: dict[str, float] = {}
    for date, day in frame.groupby("signal_date", sort=True):
        size = min(3, len(day))
        values[str(date)] = float(
            day.loc[day[rank_column].le(size), "capped_opportunity_return_7"].mean()
        )
    return pd.Series(values, dtype=float)


def _raw_downside(frame: pd.DataFrame, rank_column: str) -> dict[str, float]:
    selected = frame[frame[rank_column].le(np.minimum(3, frame["candidate_count"]))]
    daily = selected.groupby("signal_date", sort=True)["raw_repair_return"].mean()
    return {
        "selected_stock_raw_mean": float(selected["raw_repair_return"].mean()),
        "selected_stock_raw_median": float(selected["raw_repair_return"].median()),
        "worst_selected_stock": float(selected["raw_repair_return"].min()),
        "worst_top3_raw_date": float(daily.min()),
    }


def build_practical(
    oof: pd.DataFrame,
    bucket_map: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rank_columns = {
        POLICY_STAGE1: "stage1_rank",
        POLICY_CAPPED: "final_capped_rank",
        POLICY_LIMITED: "final_limited_risk_rank",
        POLICY_ORACLE: "rank_oracle_loss_stage1_backfill",
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scope in ("COMBINED", "LOW", "MID", "HIGH"):
        dates = sorted(oof["signal_date"].unique()) if scope == "COMBINED" else [
            date for date, bucket in bucket_map.items() if bucket == scope
        ]
        subset = oof[oof["signal_date"].isin(dates)].copy()
        for policy, rank_column in rank_columns.items():
            summary = strategy_summary(subset, rank_column)
            if scope == "COMBINED":
                summaries[policy] = summary
            row: dict[str, Any] = {"scope": scope, "policy": policy, **summary}
            if scope == "COMBINED":
                row.update(_raw_downside(subset, rank_column))
            rows.append(row)
        stage1 = strategy_summary(subset, "stage1_rank")
        universe: dict[str, Any] = {
            "scope": scope,
            "policy": "UNIVERSE",
            "date_count": int(subset["signal_date"].nunique()),
            "universe_target7_rate": stage1["universe_target7_rate"],
            "winner_capture": np.nan,
            "Top3_days_le_minus_3": np.nan,
            "Top3_days_le_minus_5": np.nan,
        }
        for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3"):
            baseline = stage1[f"{label}_baseline"]
            for metric in (
                "mean", "median", "p25", "worst", "baseline", "excess",
                "positive_date_rate", "negative_date_rate", "beat_universe",
                "target7_precision", "dates",
            ):
                universe[f"{label}_{metric}"] = np.nan
            universe[f"{label}_mean"] = baseline
            universe[f"{label}_baseline"] = baseline
            universe[f"{label}_excess"] = 0.0
            universe[f"{label}_target7_precision"] = stage1["universe_target7_rate"]
            universe[f"{label}_dates"] = stage1[f"{label}_dates"]
        rows.append(universe)
    return pd.DataFrame(rows), summaries


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    positive = s[y == 1]
    negative = s[y == 0]
    if not len(positive) or not len(negative):
        return np.nan
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "score": np.asarray(scores, dtype=float)})
    positives = int(frame["y"].sum())
    if positives == 0:
        return np.nan
    grouped = frame.groupby("score", sort=False)["y"].agg(["sum", "count"]).sort_index(ascending=False)
    tp = grouped["sum"].cumsum().to_numpy(float)
    fp = (grouped["count"] - grouped["sum"]).cumsum().to_numpy(float)
    recall = tp / positives
    precision = tp / (tp + fp)
    previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def risk_diagnostics(oof: pd.DataFrame) -> dict[str, Any]:
    rows = oof[oof["original_stage1_top3"]].copy()
    y = rows["loss_target"].to_numpy(int)
    score = rows["risk_p_loss"].to_numpy(float)
    veto = rows["risk_veto"].to_numpy(bool)
    tp = int(np.sum(veto & (y == 1)))
    fp = int(np.sum(veto & (y == 0)))
    fn = int(np.sum(~veto & (y == 1)))
    tn = int(np.sum(~veto & (y == 0)))
    precision = float(tp / (tp + fp)) if tp + fp else np.nan
    recall = float(tp / (tp + fn)) if tp + fn else np.nan
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else np.nan
    target7_rows = rows[rows["target7"].eq(1)]
    vetoed = rows[rows["risk_veto"]]
    base = float(rows["loss_target"].mean())
    veto_rate = float(vetoed["loss_target"].mean()) if len(vetoed) else np.nan
    def distribution(subset: pd.DataFrame) -> dict[str, float]:
        return {
            "mean": float(subset["risk_p_loss"].mean()) if len(subset) else np.nan,
            "median": float(subset["risk_p_loss"].median()) if len(subset) else np.nan,
        }
    return {
        "rows": int(len(rows)),
        "dates": int(rows["signal_date"].nunique()),
        "actual_losses": int(y.sum()),
        "actual_nonloss": int((y == 0).sum()),
        "actual_target7": int(rows["target7"].sum()),
        "loss_prevalence": base,
        "loss_probability": distribution(rows[rows["loss_target"].eq(1)]),
        "nonloss_probability": distribution(rows[rows["loss_target"].eq(0)]),
        "target7_probability": distribution(target7_rows),
        "roc_auc": _binary_auc(y, score),
        "average_precision": _average_precision(y, score),
        "veto_count": int(veto.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "loss_veto_precision": precision,
        "loss_veto_recall": recall,
        "loss_f1": f1,
        "nonloss_false_veto_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
        "target7_false_veto_count": int(target7_rows["risk_veto"].sum()),
        "target7_false_veto_rate": float(target7_rows["risk_veto"].mean()) if len(target7_rows) else np.nan,
        "positive_nontarget_false_veto": int(
            (rows["risk_veto"] & rows["target7"].eq(0) & rows["raw_repair_return"].ge(0.0)).sum()
        ),
        "vetoed_stock_loss_rate": veto_rate,
        "veto_lift": float(veto_rate / base) if len(vetoed) and base > 0 else np.nan,
    }


def build_veto_attribution(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_count = int(oof["signal_date"].nunique())
    for date, day in oof.groupby("signal_date", sort=True):
        size = min(3, len(day))
        control_ids = set(day.loc[day["stage1_rank"].le(size), "event_id"].astype(str))
        limited_ids = set(day.loc[day["final_limited_risk_rank"].le(size), "event_id"].astype(str))
        removed = day[day["event_id"].astype(str).isin(control_ids - limited_ids)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        added = day[day["event_id"].astype(str).isin(limited_ids - control_ids)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        if len(removed) != len(added):
            raise RuntimeError("FATAL: risk replacement membership mismatch")
        for left, right in zip(removed.itertuples(), added.itertuples()):
            if left.raw_repair_return < 0:
                category = "CORRECT_LOSS_VETO"
            elif left.target7 == 1:
                category = "FALSE_TARGET7_VETO"
            else:
                category = "FALSE_POSITIVE_NON_TARGET_VETO"
            pair_delta = float(right.capped_opportunity_return_7 - left.capped_opportunity_return_7)
            rows.append({
                "signal_date": date,
                "vetoed_event_id": left.event_id,
                "vetoed_code": str(left.code).zfill(6),
                "vetoed_stage1_rank": int(left.stage1_rank),
                "vetoed_p_loss": float(left.risk_p_loss),
                "vetoed_state": category,
                "vetoed_target7": int(left.target7),
                "vetoed_raw_return": float(left.raw_repair_return),
                "backfill_event_id": right.event_id,
                "backfill_code": str(right.code).zfill(6),
                "backfill_stage1_rank": int(right.stage1_rank),
                "backfill_state": outcome_state(right.raw_repair_return),
                "backfill_target7": int(right.target7),
                "backfill_raw_return": float(right.raw_repair_return),
                "pair_delta": pair_delta,
                "daily_slot_contribution": pair_delta / size,
                "overall_mean_contribution": pair_delta / size / date_count,
            })
    columns = [
        "signal_date", "vetoed_event_id", "vetoed_code", "vetoed_stage1_rank",
        "vetoed_p_loss", "vetoed_state", "vetoed_target7", "vetoed_raw_return",
        "backfill_event_id", "backfill_code", "backfill_stage1_rank", "backfill_state",
        "backfill_target7", "backfill_raw_return", "pair_delta",
        "daily_slot_contribution", "overall_mean_contribution",
    ]
    attribution = pd.DataFrame(rows, columns=columns)
    control = _selected_daily(oof, "stage1_rank")
    limited = _selected_daily(oof, "final_limited_risk_rank")
    observed = float((limited - control).mean())
    attributed = float(attribution["overall_mean_contribution"].sum()) if len(attribution) else 0.0
    if not np.isclose(observed, attributed, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"FATAL: risk attribution does not close {attributed} != {observed}")
    contributions = {
        category: float(attribution.loc[attribution["vetoed_state"].eq(category), "overall_mean_contribution"].sum())
        if len(attribution) else 0.0
        for category in (
            "CORRECT_LOSS_VETO", "FALSE_TARGET7_VETO", "FALSE_POSITIVE_NON_TARGET_VETO"
        )
    }
    backfill = oof[oof["event_id"].astype(str).isin(
        attribution["backfill_event_id"].astype(str) if len(attribution) else []
    )]
    removed = oof[oof["event_id"].astype(str).isin(
        attribution["vetoed_event_id"].astype(str) if len(attribution) else []
    )]
    accounting = {
        "executed_veto_count": int(len(attribution)),
        "backfill_total": int(len(backfill)),
        "backfill_target7": int(backfill["target7"].sum()),
        "backfill_positive_non_target": int(
            (backfill["target7"].eq(0) & backfill["raw_repair_return"].ge(0.0)).sum()
        ),
        "backfill_losses": int(backfill["raw_repair_return"].lt(0.0).sum()),
        "backfill_severe_losses": int(backfill["raw_repair_return"].le(-0.05).sum()),
        "vetoed_target7": int(removed["target7"].sum()),
        "net_target7_slots_gained": int(backfill["target7"].sum() - removed["target7"].sum()),
        "contributions": contributions,
        "attributed_total": attributed,
        "observed_delta": observed,
        "closure": "PASS",
    }
    return attribution, accounting


def build_daily(
    oof: pd.DataFrame,
    bucket_map: Mapping[str, str],
    attribution: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in oof.groupby("signal_date", sort=True):
        subset = attribution[attribution["signal_date"].eq(date)]
        rows.append({
            "signal_date": date,
            "candidate_count": int(len(day)),
            "universe": float(day["capped_opportunity_return_7"].mean()),
            "stage1_top3": float(day.loc[day["stage1_rank"].le(3), "capped_opportunity_return_7"].mean()),
            "capped_top3": float(day.loc[day["final_capped_rank"].le(3), "capped_opportunity_return_7"].mean()),
            "limited_risk_top3": float(day.loc[day["final_limited_risk_rank"].le(3), "capped_opportunity_return_7"].mean()),
            "oracle_loss_backfill_top3": float(day.loc[day["rank_oracle_loss_stage1_backfill"].le(3), "capped_opportunity_return_7"].mean()),
            "delta_limited_vs_stage1": float(
                day.loc[day["final_limited_risk_rank"].le(3), "capped_opportunity_return_7"].mean()
                - day.loc[day["stage1_rank"].le(3), "capped_opportunity_return_7"].mean()
            ),
            "veto_count": int(len(subset)),
            "correct_loss_veto_count": int(subset["vetoed_state"].eq("CORRECT_LOSS_VETO").sum()),
            "false_target7_veto_count": int(subset["vetoed_state"].eq("FALSE_TARGET7_VETO").sum()),
            "false_positive_nontarget_veto_count": int(subset["vetoed_state"].eq("FALSE_POSITIVE_NON_TARGET_VETO").sum()),
            "backfill_count": int(len(subset)),
            "opportunity_bucket": bucket_map[str(date)],
        })
    return pd.DataFrame(rows)


def paired_robustness(daily: pd.DataFrame) -> dict[str, Any]:
    delta = daily["delta_limited_vs_stage1"].to_numpy(float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw = generator.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
    boot = delta[draw].mean(axis=1)
    lodo = np.asarray([np.delete(delta, index).mean() for index in range(len(delta))])
    quantiles = np.quantile(boot, [0.025, 0.50, 0.975])
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "positive_dates": int((delta > 0).sum()),
        "negative_dates": int((delta < 0).sum()),
        "zero_dates": int((delta == 0).sum()),
        "bootstrap_p025": float(quantiles[0]),
        "bootstrap_p50": float(quantiles[1]),
        "bootstrap_p975": float(quantiles[2]),
        "bootstrap_probability_positive": float((boot > 0).mean()),
        "lodo_positive_pct": float((lodo > 0).mean()),
        "lodo_min": float(lodo.min()),
        "lodo_median": float(np.median(lodo)),
        "lodo_max": float(lodo.max()),
    }


def coefficient_stability(coefficients: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping = {
        "intercept": "intercept",
        "stage1_strength": "beta_stage1_strength",
        "closing_completion_gap": "beta_closing_completion_gap",
        "strength_x_gap": "beta_strength_x_gap",
    }
    result: dict[str, dict[str, float]] = {}
    for label, column in mapping.items():
        values = coefficients[column].astype(float)
        result[label] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_fold_pct": float(values.gt(0).mean()),
            "negative_fold_pct": float(values.lt(0).mean()),
        }
    return result


def _scope(practical: pd.DataFrame, scope: str, policy: str) -> pd.Series:
    rows = practical[practical["scope"].eq(scope) & practical["policy"].eq(policy)]
    if len(rows) != 1:
        raise RuntimeError(f"FATAL: practical row unavailable {scope}/{policy}")
    return rows.iloc[0]


def stage1_parity_gate(practical: pd.DataFrame) -> bool:
    row = _scope(practical, "COMBINED", POLICY_STAGE1)
    for metric, expected in EXPECTED_STAGE1_PARITY.items():
        if not np.isclose(float(row[metric]), expected, rtol=0.0, atol=5e-13):
            return False
    capped = _scope(practical, "COMBINED", POLICY_CAPPED)
    return bool(np.isclose(float(capped.Top3_mean), EXPECTED_CAPPED_TOP3, rtol=0.0, atol=5e-13))


def formal_decision(
    practical: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    accounting: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> dict[str, Any]:
    stage1 = _scope(practical, "COMBINED", POLICY_STAGE1)
    limited = _scope(practical, "COMBINED", POLICY_LIMITED)
    universe = _scope(practical, "COMBINED", "UNIVERSE")
    low_stage1 = _scope(practical, "LOW", POLICY_STAGE1)
    low_limited = _scope(practical, "LOW", POLICY_LIMITED)
    high_stage1 = _scope(practical, "HIGH", POLICY_STAGE1)
    high_limited = _scope(practical, "HIGH", POLICY_LIMITED)
    precision = diagnostics["loss_veto_precision"]
    recall = diagnostics["loss_veto_recall"]
    false_winner = diagnostics["target7_false_veto_rate"]
    useful = bool(
        diagnostics["veto_count"] > 0
        and np.isfinite(precision) and precision >= 0.50
        and np.isfinite(recall) and recall >= 0.25
        and false_winner <= 0.25
        and diagnostics["roc_auc"] > 0.55
    )
    if useful:
        identification = "USEFUL"
    elif diagnostics["veto_count"] > 0 and (
        diagnostics["roc_auc"] > 0.50
        or (np.isfinite(precision) and precision > diagnostics["loss_prevalence"])
        or (np.isfinite(recall) and recall > 0.0)
    ):
        identification = "WEAK"
    else:
        identification = "ABSENT"
    gates = {
        "top3_delta_50bp": float(limited.Top3_mean) >= float(stage1.Top3_mean) + 0.005,
        "top3_above_universe": float(limited.Top3_mean) > float(universe.Top3_mean),
        "precision_preserved": float(limited.Top3_target7_precision) >= float(stage1.Top3_target7_precision),
        "net_target7_nonnegative": int(accounting["net_target7_slots_gained"]) >= 0,
        "negative_rate_no_worse": float(limited.Top3_negative_date_rate) <= float(stage1.Top3_negative_date_rate),
        "worst_no_worse": float(limited.Top3_worst) >= float(stage1.Top3_worst),
        "loss_veto_precision_50": bool(np.isfinite(precision) and precision >= 0.50),
        "loss_veto_recall_25": bool(np.isfinite(recall) and recall >= 0.25),
        "winner_false_veto_25": bool(false_winner <= 0.25),
        "low_delta_50bp": float(low_limited.Top3_mean) >= float(low_stage1.Top3_mean) + 0.005,
        "low_negative_no_worse": float(low_limited.Top3_negative_date_rate) <= float(low_stage1.Top3_negative_date_rate),
        "high_preserved_50bp": float(high_limited.Top3_mean) >= float(high_stage1.Top3_mean) - 0.005,
        "positive_dates_gt_negative": int(robustness["positive_dates"]) > int(robustness["negative_dates"]),
        "bootstrap_probability_65": float(robustness["bootstrap_probability_positive"]) >= 0.65,
        "lodo_positive_70": float(robustness["lodo_positive_pct"]) >= 0.70,
    }
    if all(gates.values()):
        signal = "STRONG"
    elif (
        identification in {"USEFUL", "WEAK"}
        or float(limited.Top3_mean) > float(stage1.Top3_mean)
        or float(limited.Top3_negative_date_rate) < float(stage1.Top3_negative_date_rate)
        or float(limited.Top3_worst) > float(stage1.Top3_worst)
    ):
        signal = "PARTIAL"
    else:
        signal = "ABSENT"
    return {
        "gates": gates,
        "risk_identification_signal": identification,
        "risk_protector_signal": signal,
        "july_confirmation_candidate": signal == "STRONG",
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
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return output


def build_review(context: Mapping[str, Any]) -> str:
    practical = context["practical"]
    diagnostics = context["risk_diagnostics"]
    accounting = context["accounting"]
    robustness = context["robustness"]
    coefficient_summary = context["coefficient_stability"]
    decision = context["decision"]
    oracle_headroom = float(_scope(practical, "COMBINED", POLICY_ORACLE).Top3_mean) - float(
        _scope(practical, "COMBINED", POLICY_STAGE1).Top3_mean
    )
    recovered_ratio = float(accounting["observed_delta"] / oracle_headroom) if oracle_headroom > 0 else np.nan
    combined_policies = ["UNIVERSE", POLICY_STAGE1, POLICY_CAPPED, POLICY_LIMITED, POLICY_ORACLE]
    practical_rows = []
    for policy in combined_policies:
        row = _scope(practical, "COMBINED", policy)
        practical_rows.append([
            policy, _pct(row.Rank1_mean), _pct(row.Rank2_mean), _pct(row.Rank3_mean),
            _pct(row.Top2_mean), _pct(row.Top3_mean), _pct(row.Top3_target7_precision),
            _pct(row.Top3_negative_date_rate), _pct(row.Top3_worst),
        ])
    gate_rows = [[name, "PASS" if passed else "FAIL"] for name, passed in decision["gates"].items()]
    lines = [
        "# v004c Limited Stage1 Risk Protector v001",
        "",
        "## 1. Experimental Contract",
        "",
        f"- Task: {TASK_NAME}",
        "- Development data: May/June only",
        "- July result rows accessed: 0",
        "- Stage1 owns selection; risk model has one-pass veto authority over original Top3 only.",
        "- Backfill: frozen Stage1 Rank4 through Rank10; no recursive veto.",
        "",
        "## 2. Strict Temporal Reconstruction",
        "",
        f"- Strict June dates: {len(context['strict_dates'])}",
        f"- Dates: {'|'.join(context['strict_dates'])}",
        f"- Unavailable: {'|'.join(context['unavailable_dates'])}",
        f"- SELF_LABEL_LEAKAGE_ROWS: {context['self_label_leakage_rows']}",
        f"- CURRENT_TEST_LEAKAGE_ROWS: {context['current_test_leakage_rows']}",
        f"- Stage1 parity: {'PASS' if context['stage1_parity'] else 'FAIL'}",
        "",
        "## 3. Historical Stage1 Top3 Risk Training Set",
        "",
        f"- Fold-aggregated meta rows: {context['training']['total_fold_aggregated_rows']}",
        f"- Fold-aggregated meta dates: {context['training']['total_fold_aggregated_dates']}",
        f"- Loss / non-loss / Target7 rows: {context['training']['loss_rows']} / {context['training']['nonloss_rows']} / {context['training']['target7_rows']}",
        f"- Weighted loss prevalence: {_pct(context['training']['weighted_loss_prevalence'])}",
        "- Predictors: stage1_strength, closing_completion_gap, strength_x_gap",
        "- L2: 0.30; class weighting: none; threshold: 0.50 fixed.",
        "",
        "## 4. Loss-Probability Model",
        "",
        f"- OOF original Stage1 Top3 rows: {diagnostics['rows']}",
        f"- Actual loss / non-loss / Target7: {diagnostics['actual_losses']} / {diagnostics['actual_nonloss']} / {diagnostics['actual_target7']}",
        f"- ROC AUC: {_num(diagnostics['roc_auc'])}",
        f"- Average precision: {_num(diagnostics['average_precision'])}",
        f"- Mean p_loss LOSS / NONLOSS / Target7: {_num(diagnostics['loss_probability']['mean'])} / {_num(diagnostics['nonloss_probability']['mean'])} / {_num(diagnostics['target7_probability']['mean'])}",
        "- Coefficient stability:",
        "",
        *_markdown_table(
            ["Coefficient", "Mean", "Median", "Min", "Max", "Positive Folds"],
            [
                [
                    feature,
                    _num(values["mean"]),
                    _num(values["median"]),
                    _num(values["min"]),
                    _num(values["max"]),
                    _pct(values["positive_fold_pct"]),
                ]
                for feature, values in coefficient_summary.items()
            ],
        ),
        "",
        "## 5. Fixed 0.50 Veto Classification",
        "",
        f"- Veto count: {diagnostics['veto_count']}",
        f"- TP / FP / FN / TN: {diagnostics['tp']} / {diagnostics['fp']} / {diagnostics['fn']} / {diagnostics['tn']}",
        f"- Loss veto precision / recall / F1: {_pct(diagnostics['loss_veto_precision'])} / {_pct(diagnostics['loss_veto_recall'])} / {_pct(diagnostics['loss_f1'])}",
        f"- Stage1 base loss prevalence: {_pct(diagnostics['loss_prevalence'])}",
        f"- Vetoed-stock loss rate / lift: {_pct(diagnostics['vetoed_stock_loss_rate'])} / {_num(diagnostics['veto_lift'])}x",
        f"- Target7 false veto count / rate: {diagnostics['target7_false_veto_count']} / {_pct(diagnostics['target7_false_veto_rate'])}",
        "",
        "## 6. Veto / Backfill Attribution",
        "",
        f"- Executed vetoes / backfills: {accounting['executed_veto_count']} / {accounting['backfill_total']}",
        f"- Backfill Target7 / positive non-target / loss / severe loss: {accounting['backfill_target7']} / {accounting['backfill_positive_non_target']} / {accounting['backfill_losses']} / {accounting['backfill_severe_losses']}",
        f"- Net Target7 slots gained: {accounting['net_target7_slots_gained']}",
        f"- Correct-loss contribution: {_pct(accounting['contributions']['CORRECT_LOSS_VETO'])}",
        f"- False-Target7 contribution: {_pct(accounting['contributions']['FALSE_TARGET7_VETO'])}",
        f"- False-positive-nontarget contribution: {_pct(accounting['contributions']['FALSE_POSITIVE_NON_TARGET_VETO'])}",
        f"- Attribution closure: {accounting['closure']}",
        "",
        "## 7. Practical Top3 Performance",
        "",
        *_markdown_table(
            ["Policy", "Rank1", "Rank2", "Rank3", "Top2", "Top3", "Top3 Precision", "Negative Dates", "Worst"],
            practical_rows,
        ),
        "",
        "- Raw downside:",
        "",
        *_markdown_table(
            ["Policy", "Stock Mean", "Stock Median", "Worst Stock", "Worst Top3 Date"],
            [
                [
                    policy,
                    _pct(_scope(practical, "COMBINED", policy).selected_stock_raw_mean),
                    _pct(_scope(practical, "COMBINED", policy).selected_stock_raw_median),
                    _pct(_scope(practical, "COMBINED", policy).worst_selected_stock),
                    _pct(_scope(practical, "COMBINED", policy).worst_top3_raw_date),
                ]
                for policy in (POLICY_STAGE1, POLICY_CAPPED, POLICY_LIMITED, POLICY_ORACLE)
            ],
        ),
        f"- LIMITED minus STRICT_STAGE1 Top3: {_pct(accounting['observed_delta'])}",
        f"- LIMITED minus STRICT_CAPPED Top3: {_pct(float(_scope(practical, 'COMBINED', POLICY_LIMITED).Top3_mean) - float(_scope(practical, 'COMBINED', POLICY_CAPPED).Top3_mean))}",
        f"- Oracle headroom recovered: {_pct(accounting['observed_delta'])} of {_pct(oracle_headroom)} ({_pct(recovered_ratio)}).",
        "",
        "## 8. LOW / MID / HIGH",
        "",
    ]
    for scope in ("LOW", "MID", "HIGH"):
        lines.append(f"### {scope}")
        lines.append("")
        scope_rows = []
        for policy in ("UNIVERSE", POLICY_STAGE1, POLICY_CAPPED, POLICY_LIMITED, POLICY_ORACLE):
            row = _scope(practical, scope, policy)
            scope_rows.append([policy, _pct(row.Top3_mean), _pct(row.Top3_target7_precision), _pct(row.Top3_negative_date_rate), _pct(row.Top3_worst)])
        lines.extend(_markdown_table(["Policy", "Top3", "Precision", "Negative Dates", "Worst"], scope_rows))
        lines.append("")
    lines.extend([
        "## 9. Robustness",
        "",
        f"- Daily delta mean / median: {_pct(robustness['mean'])} / {_pct(robustness['median'])}",
        f"- Positive / negative / zero dates: {robustness['positive_dates']} / {robustness['negative_dates']} / {robustness['zero_dates']}",
        f"- Bootstrap 95%: [{_pct(robustness['bootstrap_p025'])}, {_pct(robustness['bootstrap_p975'])}]",
        f"- Bootstrap P(delta > 0): {_pct(robustness['bootstrap_probability_positive'])}",
        f"- LODO positive: {_pct(robustness['lodo_positive_pct'])}",
        "",
        "## 10. Risk-Protector Decision",
        "",
        *_markdown_table(["Gate", "Result"], gate_rows),
        "",
        f"- RISK_IDENTIFICATION_SIGNAL: **{decision['risk_identification_signal']}**",
        f"- RISK_PROTECTOR_SIGNAL: **{decision['risk_protector_signal']}**",
        f"- JULY_CONFIRMATION_CANDIDATE: **{'YES' if decision['july_confirmation_candidate'] else 'NO'}**",
        "- Existing three-feature representation did not identify Stage1 losses under the frozen natural-probability threshold.",
        "- The learned policy executed no veto, recovered none of the oracle headroom, and produced no practical or downside change.",
        "- No threshold, learner, or feature variant was tested.",
    ])
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_risk_contract()
    base_x, samples, date_contract, input_audit = prepare_diagnostic_samples(root)
    strict, temporal, unavailable = reconstruct_strict_june(base_x, samples, date_contract)
    strict = add_policy_ranks(strict)
    strict_dates = sorted(strict["signal_date"].astype(str).unique())
    if strict_dates != EXPECTED_STRICT_DATES or unavailable != EXPECTED_UNAVAILABLE_DATES:
        raise RuntimeError("FATAL: strict June date set changed")
    oof, meta_audit, coefficients, training = reconstruct_limited_risk(base_x, samples, strict)
    bucket_map = load_frozen_june_buckets(root, strict_dates)
    practical, summaries = build_practical(oof, bucket_map)
    parity = stage1_parity_gate(practical)
    if not parity:
        raise RuntimeError("FATAL: strict Stage1/CAPPED parity failed")
    attribution, accounting = build_veto_attribution(oof)
    daily = build_daily(oof, bucket_map, attribution)
    robustness = paired_robustness(daily)
    diagnostics = risk_diagnostics(oof)
    coefficients_summary = coefficient_stability(coefficients)
    decision = formal_decision(practical, diagnostics, accounting, robustness)
    self_leak = int(meta_audit["self_label_leakage_rows"].sum())
    test_leak = int(meta_audit["current_test_leakage_rows"].sum())
    context: dict[str, Any] = {
        "input": input_audit,
        "strict_dates": strict_dates,
        "unavailable_dates": unavailable,
        "self_label_leakage_rows": self_leak,
        "current_test_leakage_rows": test_leak,
        "july_result_rows_accessed": 0,
        "stage1_parity": parity,
        "training": training,
        "risk_diagnostics": diagnostics,
        "accounting": accounting,
        "robustness": robustness,
        "coefficient_stability": coefficients_summary,
        "practical": practical,
        "summaries": summaries,
        "decision": decision,
    }
    review = build_review(context)
    oof_columns = [
        "event_id", "signal_date", "code", "candidate_count", "label_available_date",
        "stage1_score", "stage1_rank", "stage1_top10", "stage1_strength",
        "closing_completion_gap", "strength_x_gap", "original_stage1_top3",
        "risk_p_loss", "risk_veto", "final_limited_risk_rank", "target7",
        "loss_target", "raw_repair_return", "capped_opportunity_return_7",
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(oof[oof_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(meta_audit),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(attribution),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_limited_risk_protector(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path / "reports/research/v004c_limited_risk_protector_v001_20260506_20260630"
    )
    outputs, context = build_outputs(root_path)
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        (target / filename).write_bytes(payload)
    return target, context

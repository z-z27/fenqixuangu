"""Strict-June information-foundation audit for V4C_STAGE1 Top3 risk.

This module is deliberately descriptive.  It fits no model, creates no feature,
tests no trading threshold, and reads no July result artifact.  The screened
feature family is the frozen 53-column D1-safe pairwise-v1 contract; the audit
population is the unique 51-row strict June OOF Stage1 Top3 artifact.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_limited_risk_protector import (
    EXPECTED_STAGE1_PARITY,
    EXPECTED_STRICT_DATES,
    RISK_FEATURE_COLUMNS,
)
from .v004c_pairwise_feature_contract import (
    CONTRACT_FEATURE_NAMES,
    FEATURE_CONTRACT,
    assert_contract_integrity,
)
from .v004c_v4a_architecture_transfer import dataframe_csv_bytes


TASK_NAME = "STAGE1 TOP3 RISK INFORMATION FOUNDATION AUDIT"
JULY_RESULT_ROWS_ACCESSED = 0
JULY_USED_FOR_FEATURE_SCREEN = False
NEW_MODEL = False
NEW_FEATURES = False
THRESHOLD_SEARCH = False
TRANSFORMATION_SEARCH = False
POST_HOC_DEVELOPMENT_ORIENTATION = True
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_RESAMPLES = 20_000
PERMUTATION_SEED = 20260812
PERMUTATIONS = 10_000
REDUNDANCY_THRESHOLD = 0.85

GROUP_ORDER = (
    "BOARD_HISTORY",
    "D0_CROSS_SECTION",
    "D1_PRICE_ACTION",
    "D1_VOLUME_ACTIVITY",
    "D1_CHIP_DISTRIBUTION",
    "D1_LATE_DAY_PRESSURE",
    "D1_VWAP_POSITION",
    "D1_MA_POSITION",
    "POOL_MEMBERSHIP",
    "RECENT_7D_PATH",
)
EXPECTED_GROUP_COUNTS = {
    "BOARD_HISTORY": 6,
    "D0_CROSS_SECTION": 1,
    "D1_PRICE_ACTION": 14,
    "D1_VOLUME_ACTIVITY": 3,
    "D1_CHIP_DISTRIBUTION": 4,
    "D1_LATE_DAY_PRESSURE": 2,
    "D1_VWAP_POSITION": 1,
    "D1_MA_POSITION": 15,
    "POOL_MEMBERSHIP": 3,
    "RECENT_7D_PATH": 4,
}

OUTPUT_FILENAMES = (
    "v004c_stage1_top3_risk_population_v001.csv",
    "v004c_stage1_top3_risk_feature_manifest_v001.csv",
    "v004c_stage1_top3_risk_univariate_v001.csv",
    "v004c_stage1_top3_risk_date_stability_v001.csv",
    "v004c_stage1_top3_risk_redundancy_v001.csv",
    "v004c_stage1_top3_risk_context_v001.csv",
    "v004c_stage1_top3_risk_review_v001.md",
)

FORBIDDEN_SCREEN_COLUMNS = frozenset({
    "target7", "loss_target", "raw_repair_return",
    "capped_opportunity_return_7", "d2_open_daily", "d3_high_daily",
    "label_available_date", "capped_pair_score", "capped_pair_rank",
    "risk_p_loss", "risk_veto",
})


def assert_audit_contract() -> None:
    """Fail closed if the predeclared no-model audit contract drifts."""
    assert_contract_integrity()
    if len(CONTRACT_FEATURE_NAMES) != 53 or len(set(CONTRACT_FEATURE_NAMES)) != 53:
        raise RuntimeError("FATAL: low-level feature count must equal 53")
    counts = Counter(row["semantic_group"] for row in FEATURE_CONTRACT)
    if dict(counts) != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"FATAL: feature group counts changed: {dict(counts)}")
    if any(row["available_as_of"] not in {"D0_CLOSE", "D1_CLOSE"}
           for row in FEATURE_CONTRACT):
        raise RuntimeError("FATAL: non-D1-safe feature entered the contract")
    if FORBIDDEN_SCREEN_COLUMNS.intersection(CONTRACT_FEATURE_NAMES):
        raise RuntimeError("FATAL: outcome/model field entered low-level screen")
    if any((NEW_MODEL, NEW_FEATURES, THRESHOLD_SEARCH, TRANSFORMATION_SEARCH)):
        raise RuntimeError("FATAL: this task permits no model, feature, threshold, or transform search")
    if JULY_RESULT_ROWS_ACCESSED != 0 or JULY_USED_FOR_FEATURE_SCREEN:
        raise RuntimeError("FATAL: July results are prohibited")
    if BOOTSTRAP_RESAMPLES != 20_000 or PERMUTATIONS != 10_000:
        raise RuntimeError("FATAL: resampling contract changed")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    if mapped.isna().any():
        raise RuntimeError("FATAL: invalid boolean value")
    return mapped.astype(bool)


def _source_paths(root: Path) -> tuple[Path, Path]:
    oof = (
        root / "reports/research/v004c_limited_risk_protector_v001_20260506_20260630"
        / "v004c_limited_risk_oof_v001.csv"
    )
    features = (
        root / "reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
        / "v004c_pairwise_v1_input_table_v002.csv"
    )
    if not oof.is_file() or not features.is_file():
        raise RuntimeError("FATAL: frozen strict OOF or 53-feature authority unavailable")
    return oof, features


def load_population(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join the frozen strict OOF identities to the frozen 53-feature table."""
    assert_audit_contract()
    oof_path, feature_path = _source_paths(root)
    oof = pd.read_csv(oof_path, dtype={"event_id": str, "code": str})
    if oof["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in frozen strict OOF")
    original = _as_bool(oof["original_stage1_top3"])
    top3 = oof.loc[original].copy()
    top3["signal_date"] = top3["signal_date"].astype(str)
    top3["code"] = top3["code"].astype(str).str.zfill(6)
    if (top3["signal_date"] >= "2026-07-01").any():
        raise RuntimeError("FATAL: July row entered primary population")

    usecols = [
        "event_id", "code", "signal_date", "board_streak_before_break",
        *CONTRACT_FEATURE_NAMES,
    ]
    authority = pd.read_csv(
        feature_path, usecols=usecols, dtype={"event_id": str, "code": str}
    )
    if authority["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in feature authority")
    authority["code"] = authority["code"].astype(str).str.zfill(6)
    authority["signal_date"] = authority["signal_date"].astype(str)
    merged = top3.merge(
        authority,
        on="event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        raise RuntimeError(f"FATAL: unmatched Stage1 Top3 events: {unmatched}")
    for field in ("code", "signal_date"):
        if merged[field].astype(str).ne(merged[f"{field}_feature"].astype(str)).any():
            raise RuntimeError(f"FATAL: {field} mismatch in feature join")
    merged = merged.drop(columns=["_merge", "code_feature", "signal_date_feature"])
    merged = merged.rename(columns={"board_streak_before_break": "board_streak_oof"})
    merged = merged.rename(columns={"board_streak_before_break_feature": "board_streak_before_break"})
    if "board_streak_before_break" not in merged:
        # pandas only adds suffix when the column overlaps; retain the authority column.
        merged["board_streak_before_break"] = merged.pop("board_streak_oof")
    else:
        if pd.to_numeric(merged["board_streak_oof"]).ne(
            pd.to_numeric(merged["board_streak_before_break"])
        ).any():
            raise RuntimeError("FATAL: board mismatch in feature join")
        merged = merged.drop(columns="board_streak_oof")

    merged["loss_target"] = pd.to_numeric(merged["raw_repair_return"]).lt(0).astype(int)
    merged["target7"] = pd.to_numeric(merged["target7"]).astype(int)
    merged["outcome_state"] = np.select(
        [merged["loss_target"].eq(1), merged["target7"].eq(1)],
        ["LOSS", "TARGET7"],
        default="POSITIVE_NON_TARGET",
    )
    merged = merged.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    dates = sorted(merged["signal_date"].unique())
    counts = merged["outcome_state"].value_counts().to_dict()
    if dates != EXPECTED_STRICT_DATES or len(merged) != 51:
        raise RuntimeError("FATAL: strict June population identity changed")
    if merged["event_id"].nunique() != 51:
        raise RuntimeError("FATAL: strict population event_id is not unique")
    expected_counts = {"LOSS": 16, "TARGET7": 16, "POSITIVE_NON_TARGET": 19}
    if any(int(counts.get(key, 0)) != value for key, value in expected_counts.items()):
        raise RuntimeError(f"FATAL: strict population labels changed: {counts}")

    daily = merged.groupby("signal_date", sort=True)["capped_opportunity_return_7"].mean()
    metrics = {
        "rows": len(merged),
        "dates": len(dates),
        "unique_event_ids": merged["event_id"].nunique(),
        "loss": int(merged["loss_target"].sum()),
        "nonloss": int(merged["loss_target"].eq(0).sum()),
        "target7": int(merged["target7"].sum()),
        "positive_non_target": int(merged["outcome_state"].eq("POSITIVE_NON_TARGET").sum()),
        "top3": float(daily.mean()),
        "precision": float(merged["target7"].mean()),
        "negative_date_rate": float(daily.lt(0).mean()),
        "worst": float(daily.min()),
        "duplicate_event_id": 0,
        "unmatched": 0,
        "extra_outcome_driven_selection": 0,
    }
    parity_keys = {
        "top3": "Top3_mean",
        "precision": "Top3_target7_precision",
        "negative_date_rate": "Top3_negative_date_rate",
        "worst": "Top3_worst",
    }
    metrics["parity"] = all(
        np.isclose(metrics[key], EXPECTED_STAGE1_PARITY[frozen], rtol=0.0, atol=1e-11)
        for key, frozen in parity_keys.items()
    )
    if not metrics["parity"]:
        raise RuntimeError(f"FATAL: strict Stage1 Top3 parity failed: {metrics}")

    manifest_rows = []
    for spec in FEATURE_CONTRACT:
        name = spec["feature_name"]
        values = pd.to_numeric(merged[name], errors="coerce")
        manifest_rows.append({
            "feature_order": int(CONTRACT_FEATURE_NAMES.index(name) + 1),
            "feature_name": name,
            "feature_group": spec["semantic_group"],
            "d1_safe": spec["available_as_of"] in {"D0_CLOSE", "D1_CLOSE"},
            "available_as_of": spec["available_as_of"],
            "dtype": spec["dtype"],
            "nonmissing_rows": int(values.notna().sum()),
            "unique_values": int(values.nunique(dropna=True)),
            "coverage": float(values.notna().mean()),
        })
    manifest = pd.DataFrame(manifest_rows)
    return merged, manifest, metrics


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware empirical AUC with LOSS (1) as the positive class."""
    y = np.asarray(labels, dtype=float)
    x = np.asarray(scores, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid].astype(int)
    x = x[valid]
    pos = x[y == 1]
    neg = x[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).astype(float)
    comparisons += 0.5 * (pos[:, None] == neg[None, :])
    return float(comparisons.mean())


def cliffs_delta(loss_values: Sequence[float], comparison_values: Sequence[float]) -> float:
    loss = np.asarray(loss_values, dtype=float)
    other = np.asarray(comparison_values, dtype=float)
    loss = loss[np.isfinite(loss)]
    other = other[np.isfinite(other)]
    if not len(loss) or not len(other):
        return float("nan")
    comp = (loss[:, None] > other[None, :]).astype(float)
    comp -= (loss[:, None] < other[None, :]).astype(float)
    return float(comp.mean())


def diagnostic_risk_sign(raw_auc: float) -> int:
    if not np.isfinite(raw_auc):
        raise ValueError("raw AUC must be finite")
    return 1 if raw_auc >= 0.5 else -1


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values with monotone correction."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite one-dimensional [0,1] array")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    raw = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def stratified_permutation_labels(
    frame: pd.DataFrame,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> np.ndarray:
    """Shuffle LOSS labels independently inside each date, preserving counts."""
    labels = frame["loss_target"].to_numpy(int)
    result = np.empty((permutations, len(frame)), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for _, index in frame.groupby("signal_date", sort=True).indices.items():
        idx = np.asarray(index, dtype=int)
        random_order = np.argsort(rng.random((permutations, len(idx))), axis=1)
        result[:, idx] = labels[idx][random_order]
    return result


def date_bootstrap_counts(
    dates: Sequence[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(map(str, dates)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    counts = np.zeros((resamples, len(unique)), dtype=np.int16)
    rows = np.repeat(np.arange(resamples), len(unique))
    np.add.at(counts, (rows, draws.ravel()), 1)
    return unique, counts


def _summary_stats(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "p25": float(x.quantile(0.25)) if len(x) else np.nan,
        "p75": float(x.quantile(0.75)) if len(x) else np.nan,
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }


def _permutation_p(
    values: np.ndarray,
    labels: np.ndarray,
    permuted_labels: np.ndarray,
    observed_auc: float,
) -> float:
    valid = np.isfinite(values)
    x = values[valid]
    y_perm = permuted_labels[:, valid].astype(float)
    ranks = pd.Series(x).rank(method="average").to_numpy(float)
    n_pos = y_perm.sum(axis=1)
    n_neg = len(x) - n_pos
    rank_sum = y_perm @ ranks
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    perm_distance = np.abs(auc - 0.5)
    observed = abs(float(observed_auc) - 0.5)
    exceed = int(np.sum(np.isfinite(perm_distance) & (perm_distance >= observed - 1e-15)))
    return float((1 + exceed) / (1 + len(permuted_labels)))


def _bootstrap_auc(
    frame: pd.DataFrame,
    oriented: np.ndarray,
    unique_dates: Sequence[str],
    counts: np.ndarray,
) -> np.ndarray:
    date_to_index = {date: i for i, date in enumerate(unique_dates)}
    date_index = frame["signal_date"].astype(str).map(date_to_index).to_numpy(int)
    y = frame["loss_target"].to_numpy(int)
    valid = np.isfinite(oriented)
    n_dates = len(unique_dates)
    loss_counts = np.zeros(n_dates, dtype=float)
    nonloss_counts = np.zeros(n_dates, dtype=float)
    numerator = np.zeros((n_dates, n_dates), dtype=float)
    for left in range(n_dates):
        loss = oriented[valid & (y == 1) & (date_index == left)]
        loss_counts[left] = len(loss)
        for right in range(n_dates):
            nonloss = oriented[valid & (y == 0) & (date_index == right)]
            if left == 0:
                nonloss_counts[right] = len(nonloss)
            if len(loss) and len(nonloss):
                comp = (loss[:, None] > nonloss[None, :]).astype(float)
                comp += 0.5 * (loss[:, None] == nonloss[None, :])
                numerator[left, right] = float(comp.sum())
    numer = np.einsum("bi,ij,bj->b", counts, numerator, counts, optimize=True)
    loss_n = counts @ loss_counts
    nonloss_n = counts @ nonloss_counts
    denom = loss_n * nonloss_n
    auc = np.full(len(counts), np.nan, dtype=float)
    valid_boot = denom > 0
    auc[valid_boot] = numer[valid_boot] / denom[valid_boot]
    return auc


def _date_stability(
    frame: pd.DataFrame, feature: str, group: str, risk_sign: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for date, day in frame.groupby("signal_date", sort=True):
        loss = pd.to_numeric(day.loc[day["loss_target"].eq(1), feature], errors="coerce").dropna()
        nonloss = pd.to_numeric(day.loc[day["loss_target"].eq(0), feature], errors="coerce").dropna()
        if not len(loss) or not len(nonloss):
            continue
        raw = float(loss.median() - nonloss.median())
        oriented = float(risk_sign * raw)
        rows.append({
            "feature": feature,
            "group": group,
            "signal_date": str(date),
            "loss_n": len(loss),
            "nonloss_n": len(nonloss),
            "loss_median": float(loss.median()),
            "nonloss_median": float(nonloss.median()),
            "raw_daily_diff": raw,
            "risk_sign": risk_sign,
            "oriented_daily_diff": oriented,
            "direction_match": oriented > 0,
        })
    daily = pd.DataFrame(rows, columns=[
        "feature", "group", "signal_date", "loss_n", "nonloss_n",
        "loss_median", "nonloss_median", "raw_daily_diff", "risk_sign",
        "oriented_daily_diff", "direction_match",
    ])
    if daily.empty:
        return daily, {
            "mixed_dates": 0, "positive": 0, "negative": 0, "zero": 0,
            "consistency": np.nan, "median_diff": np.nan,
        }
    diff = daily["oriented_daily_diff"].to_numpy(float)
    return daily, {
        "mixed_dates": len(diff),
        "positive": int((diff > 0).sum()),
        "negative": int((diff < 0).sum()),
        "zero": int((diff == 0).sum()),
        "consistency": float((diff > 0).mean()),
        "median_diff": float(np.median(diff)),
    }


def _lodo(frame: pd.DataFrame, feature: str, risk_sign: int) -> dict[str, float]:
    aucs: list[float] = []
    cliffs: list[float] = []
    for date in EXPECTED_STRICT_DATES:
        fold = frame[frame["signal_date"].ne(date)]
        x = pd.to_numeric(fold[feature], errors="coerce").to_numpy(float) * risk_sign
        y = fold["loss_target"].to_numpy(int)
        aucs.append(binary_auc(y, x))
        cliffs.append(cliffs_delta(x[y == 1], x[y == 0]))
    auc_array = np.asarray(aucs, dtype=float)
    cliff_array = np.asarray(cliffs, dtype=float)
    return {
        "lodo_auc_gt_half_pct": float(np.sum(auc_array > 0.5) / len(EXPECTED_STRICT_DATES)),
        "lodo_auc_min": float(np.nanmin(auc_array)),
        "lodo_auc_median": float(np.nanmedian(auc_array)),
        "lodo_auc_max": float(np.nanmax(auc_array)),
        "lodo_cliff_positive_pct": float(np.sum(cliff_array > 0) / len(EXPECTED_STRICT_DATES)),
        "lodo_cliff_min": float(np.nanmin(cliff_array)),
        "lodo_cliff_median": float(np.nanmedian(cliff_array)),
        "lodo_cliff_max": float(np.nanmax(cliff_array)),
    }


def analyze_feature(
    frame: pd.DataFrame,
    feature: str,
    group: str,
    is_low_level: bool,
    permuted_labels: np.ndarray,
    bootstrap_dates: Sequence[str],
    bootstrap_counts: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    values = pd.to_numeric(frame[feature], errors="coerce")
    y = frame["loss_target"].to_numpy(int)
    raw_auc = binary_auc(y, values)
    sign = diagnostic_risk_sign(raw_auc)
    oriented = values.to_numpy(float) * sign
    loss = values[frame["loss_target"].eq(1)]
    nonloss = values[frame["loss_target"].eq(0)]
    target7 = values[frame["target7"].eq(1)]
    positive = values[frame["outcome_state"].eq("POSITIVE_NON_TARGET")]
    raw_cliff = cliffs_delta(loss, nonloss)
    oriented_auc = binary_auc(y, oriented)
    target_labels = np.r_[np.ones(len(loss)), np.zeros(len(target7))]
    target_scores = np.r_[loss.to_numpy(float) * sign, target7.to_numpy(float) * sign]
    positive_labels = np.r_[np.ones(len(loss)), np.zeros(len(positive))]
    positive_scores = np.r_[loss.to_numpy(float) * sign, positive.to_numpy(float) * sign]
    daily, daily_summary = _date_stability(frame, feature, group, sign)
    lodo = _lodo(frame, feature, sign)
    boot_auc = _bootstrap_auc(frame, oriented, bootstrap_dates, bootstrap_counts)
    boot_auc = boot_auc[np.isfinite(boot_auc)]
    boot_cliff = 2.0 * boot_auc - 1.0

    row: dict[str, Any] = {
        "feature": feature,
        "group": group,
        "coverage": float(values.notna().mean()),
        "nonmissing_rows": int(values.notna().sum()),
        "missing_rows": int(values.isna().sum()),
        "unique_values": int(values.nunique(dropna=True)),
        "raw_median_diff": float(loss.median() - nonloss.median()),
        "raw_cliff_delta": raw_cliff,
        "raw_auc_loss_nonloss": raw_auc,
        "risk_sign": sign,
        "oriented_auc_loss_nonloss": oriented_auc,
        "oriented_auc_loss_target7": binary_auc(target_labels, target_scores),
        "oriented_auc_loss_positive_non_target": binary_auc(positive_labels, positive_scores),
        "oriented_loss_minus_target7_median": float(sign * (loss.median() - target7.median())),
        "mixed_dates": daily_summary["mixed_dates"],
        "positive_direction_dates": daily_summary["positive"],
        "negative_direction_dates": daily_summary["negative"],
        "zero_direction_dates": daily_summary["zero"],
        "date_direction_consistency": daily_summary["consistency"],
        "median_daily_oriented_diff": daily_summary["median_diff"],
        "bootstrap_cliff_p2_5": float(np.quantile(boot_cliff, 0.025)),
        "bootstrap_cliff_p50": float(np.quantile(boot_cliff, 0.50)),
        "bootstrap_cliff_p97_5": float(np.quantile(boot_cliff, 0.975)),
        "bootstrap_cliff_positive_probability": float((boot_cliff > 0).mean()),
        "bootstrap_auc_p2_5": float(np.quantile(boot_auc, 0.025)),
        "bootstrap_auc_p50": float(np.quantile(boot_auc, 0.50)),
        "bootstrap_auc_p97_5": float(np.quantile(boot_auc, 0.975)),
        "bootstrap_auc_gt_half_probability": float((boot_auc > 0.5).mean()),
        "permutation_p": _permutation_p(
            values.to_numpy(float), y, permuted_labels, raw_auc
        ),
        "BH_q": np.nan,
        "is_low_level_53": bool(is_low_level),
        "is_control_representation": not bool(is_low_level),
        "post_hoc_development_orientation": POST_HOC_DEVELOPMENT_ORIENTATION,
    }
    for label, subset in (
        ("loss", loss), ("nonloss", nonloss), ("target7", target7),
        ("positive_non_target", positive),
    ):
        for stat, value in _summary_stats(subset).items():
            row[f"{label}_{stat}"] = value
    row.update(lodo)
    return row, daily


def _failed_gates(row: Mapping[str, Any]) -> list[str]:
    gates = [
        (float(row["coverage"]) >= 0.95, "COVERAGE"),
        (float(row["oriented_auc_loss_nonloss"]) >= 0.60, "PRIMARY_AUC"),
        (abs(float(row["raw_cliff_delta"])) >= 0.30, "CLIFF"),
        (float(row["oriented_auc_loss_target7"]) >= 0.60, "WINNER_SAFETY"),
        (int(row["mixed_dates"]) >= 5, "DATE_SUPPORT"),
        (float(row["date_direction_consistency"]) >= 0.65, "DATE_CONSISTENCY"),
        (float(row["bootstrap_cliff_positive_probability"]) >= 0.90, "BOOTSTRAP"),
        (float(row["lodo_cliff_positive_pct"]) >= 0.80, "LODO"),
        (float(row["BH_q"]) <= 0.10, "BH_FDR"),
    ]
    return [name for passed, name in gates if not passed]


def apply_evidence_gates(univariate: pd.DataFrame) -> pd.DataFrame:
    result = univariate.copy()
    low = result["is_low_level_53"].astype(bool)
    result.loc[low, "BH_q"] = benjamini_hochberg(
        result.loc[low, "permutation_p"].to_numpy(float)
    )
    result["coverage_fail"] = result["coverage"].lt(0.95)
    result["date_support_weak"] = result["mixed_dates"].lt(5)
    result["foundation_pass"] = (
        low
        & result["coverage"].ge(0.95)
        & result["oriented_auc_loss_nonloss"].ge(0.60)
        & result["raw_cliff_delta"].abs().ge(0.30)
        & result["oriented_auc_loss_target7"].ge(0.60)
        & result["mixed_dates"].ge(5)
        & result["date_direction_consistency"].ge(0.65)
        & result["bootstrap_cliff_positive_probability"].ge(0.90)
        & result["lodo_cliff_positive_pct"].ge(0.80)
        & result["BH_q"].le(0.10)
    )
    result["foundation_exceptional"] = (
        low
        & result["oriented_auc_loss_nonloss"].ge(0.70)
        & result["oriented_auc_loss_target7"].ge(0.70)
        & result["raw_cliff_delta"].abs().ge(0.45)
        & result["date_direction_consistency"].ge(0.70)
        & result["bootstrap_cliff_positive_probability"].ge(0.95)
        & result["lodo_cliff_positive_pct"].ge(0.90)
        & result["BH_q"].le(0.05)
    )
    result["evidence_status"] = np.select(
        [result["foundation_exceptional"], result["foundation_pass"]],
        ["FOUNDATION_EXCEPTIONAL", "FOUNDATION_PASS"],
        default="NO_PASS",
    )
    failed = []
    for row in result.to_dict("records"):
        if row["is_low_level_53"]:
            failed.append("|".join(_failed_gates(row)))
        else:
            failed.append("CONTROL_NOT_IN_GATE")
    result["failed_gates"] = failed
    order = {name: i for i, name in enumerate(CONTRACT_FEATURE_NAMES)}
    result["_order"] = result["feature"].map(order).fillna(10_000)
    return result.sort_values(
        ["is_control_representation", "_order", "feature"],
        kind="mergesort",
    ).drop(columns="_order").reset_index(drop=True)


def build_redundancy(population: pd.DataFrame, univariate: pd.DataFrame) -> pd.DataFrame:
    selected = univariate.loc[
        univariate["foundation_pass"] | univariate["foundation_exceptional"], "feature"
    ].tolist()
    rows = []
    groups = univariate.set_index("feature")["group"].to_dict()
    for i, left in enumerate(selected):
        for right in selected[i + 1:]:
            rho = float(pd.to_numeric(population[left], errors="coerce").corr(
                pd.to_numeric(population[right], errors="coerce"), method="spearman"
            ))
            rows.append({
                "feature_a": left, "group_a": groups[left],
                "feature_b": right, "group_b": groups[right],
                "spearman_rho": rho, "abs_rho": abs(rho),
                "near_redundant": abs(rho) >= REDUNDANCY_THRESHOLD,
            })
    return pd.DataFrame(rows, columns=[
        "feature_a", "group_a", "feature_b", "group_b",
        "spearman_rho", "abs_rho", "near_redundant",
    ])


def _context_row(kind: str, value: str, subset: pd.DataFrame) -> dict[str, Any]:
    return {
        "context_type": kind,
        "context_value": value,
        "rows": len(subset),
        "loss_count": int(subset["loss_target"].sum()),
        "loss_rate": float(subset["loss_target"].mean()) if len(subset) else np.nan,
        "target7_count": int(subset["target7"].sum()),
        "target7_rate": float(subset["target7"].mean()) if len(subset) else np.nan,
        "mean_raw_return": float(subset["raw_repair_return"].mean()) if len(subset) else np.nan,
    }


def build_context(
    population: pd.DataFrame, manifest: pd.DataFrame, univariate: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank in (1, 2, 3):
        rows.append(_context_row(
            "STAGE1_RANK", f"Rank{rank}", population[population["stage1_rank"].eq(rank)]
        ))
    for board in (2, 3):
        rows.append(_context_row(
            "BOARD", f"board{board}",
            population[pd.to_numeric(population["board_streak_before_break"]).eq(board)],
        ))
    count = pd.to_numeric(population["candidate_count"])
    for label, mask in (
        ("<5", count.lt(5)), ("5-9", count.between(5, 9)), (">=10", count.ge(10)),
    ):
        rows.append(_context_row("CANDIDATE_COUNT_BUCKET", label, population[mask]))
    low = univariate[univariate["is_low_level_53"]]
    for group in GROUP_ORDER:
        subset = low[low["group"].eq(group)]
        row = {
            "context_type": "FEATURE_GROUP",
            "context_value": group,
            "rows": len(subset),
            "loss_count": np.nan,
            "loss_rate": np.nan,
            "target7_count": np.nan,
            "target7_rate": np.nan,
            "mean_raw_return": np.nan,
            "coverage_pass_count": int(subset["coverage"].ge(0.95).sum()),
            "foundation_pass_count": int(subset["foundation_pass"].sum()),
            "exceptional_count": int(subset["foundation_exceptional"].sum()),
            "best_oriented_auc_loss_nonloss": float(subset["oriented_auc_loss_nonloss"].max()),
            "best_oriented_auc_loss_target7": float(subset["oriented_auc_loss_target7"].max()),
            "best_absolute_cliff": float(subset["raw_cliff_delta"].abs().max()),
            "best_BH_q": float(subset["BH_q"].min()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["context_type", "context_value"], kind="mergesort"
    ).reset_index(drop=True)


def formal_decision(
    univariate: pd.DataFrame, redundancy: pd.DataFrame
) -> dict[str, Any]:
    low = univariate[univariate["is_low_level_53"]].copy()
    passed = low[low["foundation_pass"] | low["foundation_exceptional"]]
    exceptional = low[low["foundation_exceptional"]]
    groups = passed["group"].nunique()
    independent_pair = bool((redundancy["abs_rho"] < REDUNDANCY_THRESHOLD).any()) \
        if not redundancy.empty else False
    route_a = len(passed) >= 2 and groups >= 2 and independent_pair
    route_b = len(exceptional) >= 1
    if route_a or route_b:
        status = "ESTABLISHED"
    elif len(passed):
        status = "PARTIAL"
    else:
        status = "NOT_ESTABLISHED"

    primary_candidates = low[
        low["oriented_auc_loss_nonloss"].ge(0.60)
        & low["raw_cliff_delta"].abs().ge(0.30)
    ]
    safe_candidates = primary_candidates[
        primary_candidates["oriented_auc_loss_target7"].ge(0.60)
    ]
    stable_candidates = safe_candidates[
        safe_candidates["mixed_dates"].ge(5)
        & safe_candidates["date_direction_consistency"].ge(0.65)
        & safe_candidates["bootstrap_cliff_positive_probability"].ge(0.90)
        & safe_candidates["lodo_cliff_positive_pct"].ge(0.80)
    ]
    fdr_candidates = stable_candidates[stable_candidates["BH_q"].le(0.10)]
    questions = {
        "q1": "YES" if len(primary_candidates) else (
            "PARTIAL" if low["oriented_auc_loss_nonloss"].ge(0.60).any() else "NO"
        ),
        "q2": "YES" if len(safe_candidates) else (
            "PARTIAL" if low["oriented_auc_loss_target7"].ge(0.60).any() else "NO"
        ),
        "q3": "YES" if len(stable_candidates) else (
            "PARTIAL" if low["date_direction_consistency"].ge(0.65).any() else "NO"
        ),
        "q4": "YES" if len(fdr_candidates) else (
            "PARTIAL" if low["BH_q"].le(0.10).any() else "NO"
        ),
        "q5": "YES" if route_a else "NO",
        "q6": "YES" if status == "ESTABLISHED" else (
            "INCONCLUSIVE" if len(primary_candidates) else "NO"
        ),
    }
    return {
        "status": status,
        "experiment_warranted": status == "ESTABLISHED",
        "pass_count": len(passed),
        "exceptional_count": len(exceptional),
        "pass_groups": groups,
        "independent_pair": independent_pair,
        "questions": questions,
    }


def _pct(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def _num(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def build_review(context: Mapping[str, Any]) -> str:
    population = context["population"]
    manifest = context["manifest"]
    univariate = context["univariate"]
    redundancy = context["redundancy"]
    context_table = context["context_table"]
    parity = context["population_metrics"]
    decision = context["decision"]
    controls = univariate[univariate["is_control_representation"]]
    low = univariate[univariate["is_low_level_53"]]
    passed = low[low["foundation_pass"] | low["foundation_exceptional"]]
    nonpass = low[~(low["foundation_pass"] | low["foundation_exceptional"])].sort_values(
        ["oriented_auc_loss_nonloss", "oriented_auc_loss_target7", "feature"],
        ascending=[False, False, True], kind="mergesort",
    ).head(10)
    groups = context_table[context_table["context_type"].eq("FEATURE_GROUP")]
    rank_context = context_table[context_table["context_type"].eq("STAGE1_RANK")]
    board_context = context_table[context_table["context_type"].eq("BOARD")]
    candidate_context = context_table[
        context_table["context_type"].eq("CANDIDATE_COUNT_BUCKET")
    ]
    pass_names = passed["feature"].tolist()
    losses = population[population["loss_target"].eq(1)]
    winners = population[population["target7"].eq(1)]
    pass_percentiles = {
        name: pd.to_numeric(population[name], errors="coerce").rank(
            method="average", pct=True
        )
        for name in pass_names
    }

    lines = [
        "# v004c Stage1 Top3 Risk Information Foundation Audit v001",
        "",
        "## 1. Experimental Contract",
        "",
        "- Scope: strict June V4C_STAGE1 original Top3, information audit only.",
        "- New model/features/thresholds/transformations: NO / NO / NO / NO.",
        "- July result rows accessed: 0.",
        "- Orientation is post-hoc June development diagnostics only; it is not a trading rule.",
        "",
        "## 2. Strict Stage1 Top3 Population",
        "",
        f"- Dates: {parity['dates']}; rows: {parity['rows']}; unique event_id: {parity['unique_event_ids']}.",
        f"- LOSS: {parity['loss']}; NONLOSS: {parity['nonloss']}; TARGET7: {parity['target7']}; POSITIVE_NON_TARGET: {parity['positive_non_target']}.",
        f"- Top3: {_pct(parity['top3'])}; precision: {_pct(parity['precision'])}; negative-date rate: {_pct(parity['negative_date_rate'])}; worst: {_pct(parity['worst'])}.",
        "- Population parity: PASS.",
        "",
        "## 3. Existing 53-Feature Manifest",
        "",
        f"- Exact low-level count: {len(manifest)}; all D1-safe: {'YES' if manifest['d1_safe'].all() else 'NO'}.",
        f"- Coverage failures (<95%): {int(manifest['coverage'].lt(0.95).sum())}.",
    ]
    lines.extend(_table(
        ["Group", "Count", "Coverage pass"],
        ([group, EXPECTED_GROUP_COUNTS[group], int(manifest.loc[
            manifest["feature_group"].eq(group), "coverage"
        ].ge(0.95).sum())] for group in GROUP_ORDER),
    ))
    lines.extend(["", "## 4. Failed 3-Feature Control Representation", ""])
    lines.extend(_table(
        ["Feature", "Raw AUC", "Oriented AUC", "LOSS/T7 AUC", "Cliff", "Date %", "Boot P", "LODO %"],
        ([
            row.feature, _num(row.raw_auc_loss_nonloss),
            _num(row.oriented_auc_loss_nonloss), _num(row.oriented_auc_loss_target7),
            _num(row.raw_cliff_delta), _pct(row.date_direction_consistency),
            _num(row.bootstrap_cliff_positive_probability),
            _pct(row.lodo_cliff_positive_pct),
        ] for row in controls.itertuples()),
    ))
    lines.extend(["", "## 5. LOSS vs NONLOSS Univariate Evidence", ""])
    if passed.empty:
        lines.append("No FOUNDATION_PASS or FOUNDATION_EXCEPTIONAL feature.")
    else:
        lines.extend(_table(
            ["Feature", "Group", "AUC", "LOSS/T7 AUC", "Cliff", "Date %", "p", "q", "Status"],
            ([
                row.feature, row.group, _num(row.oriented_auc_loss_nonloss),
                _num(row.oriented_auc_loss_target7), _num(row.raw_cliff_delta),
                _pct(row.date_direction_consistency), _num(row.permutation_p),
                _num(row.BH_q), row.evidence_status,
            ] for row in passed.itertuples()),
        ))
    lines.extend(["", "Strongest non-pass signals (ranked descriptively by oriented LOSS/NONLOSS AUC):", ""])
    lines.extend(_table(
        ["Feature", "Group", "AUC", "LOSS/T7", "Cliff", "Date %", "Boot P", "LODO %", "p", "q", "Failed"],
        ([
            row.feature, row.group, _num(row.oriented_auc_loss_nonloss),
            _num(row.oriented_auc_loss_target7), _num(row.raw_cliff_delta),
            _pct(row.date_direction_consistency),
            _num(row.bootstrap_cliff_positive_probability),
            _pct(row.lodo_cliff_positive_pct), _num(row.permutation_p),
            _num(row.BH_q), row.failed_gates,
        ] for row in nonpass.itertuples()),
    ))
    lines.extend([
        "", "## 6. LOSS vs Target7 Winner-Safety Evidence", "",
        f"Features with oriented LOSS/TARGET7 AUC >=0.60: {int(low['oriented_auc_loss_target7'].ge(0.60).sum())}.",
        "The same full-sample LOSS/NONLOSS risk sign is used; no Target7-specific reorientation is permitted.",
        "", "## 7. Date Stability and LODO", "",
        f"Features passing mixed-date support and >=65% direction consistency: {int((low['mixed_dates'].ge(5) & low['date_direction_consistency'].ge(0.65)).sum())}.",
        f"Features with LODO Cliff-positive >=80%: {int(low['lodo_cliff_positive_pct'].ge(0.80).sum())}.",
        "", "## 8. Date-Block Bootstrap", "",
        f"Date resamples: {BOOTSTRAP_RESAMPLES}; seed: {BOOTSTRAP_SEED}; unit: signal_date block.",
        f"Features with P(oriented Cliff>0) >=0.90: {int(low['bootstrap_cliff_positive_probability'].ge(0.90).sum())}.",
        "", "## 9. Stratified Permutation / FDR", "",
        f"Permutations: {PERMUTATIONS}; seed: {PERMUTATION_SEED}; labels shuffled within signal_date.",
        f"Features with BH q<=0.10: {int(low['BH_q'].le(0.10).sum())}.",
        "", "## 10. Feature-Group Summary", "",
    ])
    lines.extend(_table(
        ["Group", "Features", "Pass", "Exceptional", "Best AUC", "Best LOSS/T7", "Best abs Cliff", "Best q"],
        ([
            row.context_value, int(row.rows), int(row.foundation_pass_count),
            int(row.exceptional_count), _num(row.best_oriented_auc_loss_nonloss),
            _num(row.best_oriented_auc_loss_target7), _num(row.best_absolute_cliff),
            _num(row.best_BH_q),
        ] for row in groups.itertuples()),
    ))
    lines.extend(["", "## 11. Redundancy", ""])
    lines.append(f"Passing features: {', '.join(pass_names) if pass_names else 'NONE'}.")
    if redundancy.empty:
        lines.append("No passing-feature pair available for redundancy assessment.")
    else:
        lines.extend(_table(
            ["A", "B", "Spearman", "Near redundant"],
            ([row.feature_a, row.feature_b, _num(row.spearman_rho), row.near_redundant]
             for row in redundancy.itertuples()),
        ))
    lines.extend(["", "## 12. Rank / Board / Candidate-Count Context", ""])
    lines.extend(_table(
        ["Context", "Rows", "LOSS rate", "Target7 rate", "Mean raw"],
        ([f"{row.context_type}:{row.context_value}", int(row.rows),
          _pct(row.loss_rate), _pct(row.target7_rate), _pct(row.mean_raw_return)]
         for row in pd.concat([rank_context, board_context, candidate_context]).itertuples()),
    ))
    severe_headers = ["event_id", "date", "code", "rank", "raw", "<=-5%", *pass_names]
    lines.extend(["", "All LOSS rows (the <=-5% flag is descriptive only):", ""])
    lines.extend(_table(
        severe_headers,
        ([
            row.event_id, row.signal_date, row.code, int(row.stage1_rank),
            _pct(row.raw_repair_return), row.raw_repair_return <= -0.05,
            *(_num(getattr(row, name)) for name in pass_names),
        ] for row in losses.itertuples()),
    ))
    lines.extend(["", "Target7 winner-safety rows:", ""])
    winner_headers = [
        "event_id", "date", "code", "rank", "raw",
        *(label for name in pass_names for label in (name, f"{name}_population_pct")),
    ]
    lines.extend(_table(
        winner_headers,
        ([
            row.event_id, row.signal_date, row.code, int(row.stage1_rank),
            _pct(row.raw_repair_return),
            *(value for name in pass_names for value in (
                _num(getattr(row, name)),
                _pct(pass_percentiles[name].loc[row.Index]),
            )),
        ] for row in winners.itertuples()),
    ))
    lines.extend([
        "", "## 13. Risk Information Foundation Decision", "",
        f"- Q1 material LOSS/NONLOSS separation: **{decision['questions']['q1']}**.",
        f"- Q2 winner-safe separation: **{decision['questions']['q2']}**.",
        f"- Q3 stable across dates: **{decision['questions']['q3']}**.",
        f"- Q4 survives permutation/FDR: **{decision['questions']['q4']}**.",
        f"- Q5 at least two non-redundant sources: **{decision['questions']['q5']}**.",
        f"- Q6 prior 3-feature representation discarded established low-level information: **{decision['questions']['q6']}**.",
        f"- RISK_INFORMATION_FOUNDATION: **{decision['status']}**.",
        f"- LIMITED_RISK_REPRESENTATION_EXPERIMENT_WARRANTED: **{'YES' if decision['experiment_warranted'] else 'NO'}**.",
        "- No risk model, veto threshold, feature transform, or trading counterfactual was tested.",
    ])
    if decision["status"] == "ESTABLISHED":
        lines.append(
            "- Recommended next action: in a separate May/June-only task, freeze a minimal representation drawn only from predeclared passing evidence and test one limited Stage1 Top3 risk protector; do not access July first."
        )
    elif decision["status"] == "PARTIAL":
        lines.append(
            "- Recommended next action: preserve the audit; do not train another protector because evidence is insufficiently independent or stable."
        )
    else:
        lines.append(
            "- Recommended next action: stop the existing D1-only risk-protector route; oracle headroom remains theoretical, but this audited feature set did not establish a stable information foundation."
        )
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    population, manifest, population_metrics = load_population(root)
    permuted = stratified_permutation_labels(population)
    boot_dates, boot_counts = date_bootstrap_counts(population["signal_date"])
    rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    groups = {row["feature_name"]: row["semantic_group"] for row in FEATURE_CONTRACT}
    for feature in CONTRACT_FEATURE_NAMES:
        row, daily = analyze_feature(
            population, feature, groups[feature], True,
            permuted, boot_dates, boot_counts,
        )
        rows.append(row)
        daily_frames.append(daily)
    for feature in RISK_FEATURE_COLUMNS:
        row, daily = analyze_feature(
            population, feature, "CONTROL_REPRESENTATION", False,
            permuted, boot_dates, boot_counts,
        )
        rows.append(row)
        daily_frames.append(daily)
    univariate = apply_evidence_gates(pd.DataFrame(rows))
    daily_stability = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["feature", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)
    redundancy = build_redundancy(population, univariate)
    context_table = build_context(population, manifest, univariate)
    decision = formal_decision(univariate, redundancy)
    context: dict[str, Any] = {
        "population": population,
        "manifest": manifest,
        "population_metrics": population_metrics,
        "univariate": univariate,
        "date_stability": daily_stability,
        "redundancy": redundancy,
        "context_table": context_table,
        "decision": decision,
        "july_result_rows_accessed": 0,
    }
    review = build_review(context)
    population_columns = [
        "event_id", "signal_date", "code", "stage1_rank", "stage1_score",
        "candidate_count", "board_streak_before_break", "raw_repair_return",
        "outcome_state", "loss_target", "target7", *CONTRACT_FEATURE_NAMES,
        *RISK_FEATURE_COLUMNS,
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(population[population_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(manifest),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(univariate),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(daily_stability),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(redundancy),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(context_table),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    return outputs, context


def write_outputs(
    root: Path, outputs: Mapping[str, bytes], output_dir: Path | None = None
) -> Path:
    target = output_dir or (
        root / "reports/research/v004c_stage1_top3_risk_information_v001_20260603_20260626"
    )
    target.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILENAMES:
        (target / filename).write_bytes(outputs[filename])
    return target


def run_v004c_stage1_top3_risk_information(
    root: str | Path, output_dir: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    outputs, context = build_outputs(root_path)
    return write_outputs(
        root_path, outputs, Path(output_dir) if output_dir is not None else None
    ), context

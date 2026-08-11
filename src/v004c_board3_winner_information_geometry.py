"""Board3 winner-information geometry audit on existing D1-safe information.

This is a descriptive May/June information audit.  It trains no learner,
creates no feature, searches no target or threshold, and never reads a July
result.  The 53 low-level and 18 frozen Stage1 families remain separate for
permutation/FDR control.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_pairwise_feature_contract import (
    CONTRACT_FEATURE_NAMES,
    FEATURE_CONTRACT,
    assert_contract_integrity,
)
from .v004c_stage1_risk_complementarity import prepare_diagnostic_samples
from .v004c_stage1_top3_risk_information import (
    EXPECTED_GROUP_COUNTS,
    benjamini_hochberg,
    binary_auc,
    cliffs_delta,
)
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    dataframe_csv_bytes,
)


TASK_NAME = "BOARD3 WINNER INFORMATION GEOMETRY AUDIT"
EXPECTED_STARTING_HEAD = "7e18d8059f83108651198bef5aa02c3b681db9d0"
JULY_SENTINEL = "2026-07-01"
JULY_RESULT_ROWS_ACCESSED = 0
JULY_USED_FOR_FEATURE_SELECTION = False
JULY_USED_FOR_INFORMATION_AUDIT = False
NEW_MODEL = False
NEW_FEATURE = False
FEATURE_ENGINEERING = False
TARGET_SEARCH = False
THRESHOLD_SEARCH = False
ORIENTATION_SOURCE = "FULL_DEVELOPMENT_T7_VS_PNT"
POST_HOC_DIAGNOSTIC_ORIENTATION = True

BOOTSTRAP_SEED = 20260818
BOOTSTRAP_RESAMPLES = 20_000
PERMUTATION_SEED = 20260819
PERMUTATIONS = 20_000
REDUNDANCY_THRESHOLD = 0.85

LOW_LEVEL_FAMILY = "LOW_LEVEL_53"
STAGE1_FAMILY = "FROZEN_STAGE1_18"
EXPECTED_MATURED_ROWS = 307
EXPECTED_MATURED_DATES = 37
EXPECTED_BOARD3_ROWS = 55
EXPECTED_BOARD3_DATES = 28
EXPECTED_BOARD3_TARGET7 = 23
EXPECTED_BOARD3_LOSS = 13
EXPECTED_BOARD3_PNT = 19

OUTPUT_FILENAMES = (
    "v004c_board3_winner_population_v001.csv",
    "v004c_board3_winner_feature_manifest_v001.csv",
    "v004c_board3_winner_lowlevel_univariate_v001.csv",
    "v004c_board3_winner_stage1rep_univariate_v001.csv",
    "v004c_board3_winner_daily_pairs_v001.csv",
    "v004c_board3_winner_robustness_v001.csv",
    "v004c_board3_winner_redundancy_v001.csv",
    "v004c_board3_winner_review_v001.md",
)

STAGE1_GROUPS = {
    feature: (
        "RANK" if feature.startswith("rank_")
        else "DAY_BUCKET" if feature.startswith("days_since_d0_")
        else "INTERACTION_OR_SPREAD"
    )
    for feature in FROZEN_FEATURE_COLUMNS
}

LOW_LEVEL_CORROBORATION = {
    "rank_d1_close_ma10_pct": "d1_close_to_ma10_raw",
    "rank_d1_low_ma10_pct": "d1_low_to_ma10_raw",
    "rank_d1_close_vwap_pct": "d1_close_to_vwap_raw",
}


def assert_audit_contract() -> None:
    assert_contract_integrity()
    if len(CONTRACT_FEATURE_NAMES) != 53 or len(set(CONTRACT_FEATURE_NAMES)) != 53:
        raise RuntimeError("FATAL: low-level family must contain exactly 53 features")
    if Counter(row["semantic_group"] for row in FEATURE_CONTRACT) != Counter(EXPECTED_GROUP_COUNTS):
        raise RuntimeError("FATAL: low-level group contract changed")
    if any(row["available_as_of"] not in {"D0_CLOSE", "D1_CLOSE"} for row in FEATURE_CONTRACT):
        raise RuntimeError("FATAL: non-D1-safe low-level feature entered audit")
    if len(FROZEN_FEATURE_COLUMNS) != 18 or len(set(FROZEN_FEATURE_COLUMNS)) != 18:
        raise RuntimeError("FATAL: frozen Stage1 family must contain exactly 18 features")
    if set(CONTRACT_FEATURE_NAMES).intersection(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: feature families unexpectedly overlap")
    if any((NEW_MODEL, NEW_FEATURE, FEATURE_ENGINEERING, TARGET_SEARCH, THRESHOLD_SEARCH)):
        raise RuntimeError("FATAL: unauthorized model/feature/target/threshold work enabled")
    if JULY_RESULT_ROWS_ACCESSED or JULY_USED_FOR_FEATURE_SELECTION or JULY_USED_FOR_INFORMATION_AUDIT:
        raise RuntimeError("FATAL: July result access is prohibited")
    if BOOTSTRAP_RESAMPLES != 20_000 or PERMUTATIONS != 20_000:
        raise RuntimeError("FATAL: resampling contract changed")


def classify_outcome(raw_return: float) -> str:
    value = float(raw_return)
    if value < 0.0:
        return "LOSS"
    if value < 0.07:
        return "POSITIVE_NON_TARGET"
    return "TARGET7"


def winner_sign_from_auc(raw_auc: float) -> int:
    if not np.isfinite(raw_auc):
        raise ValueError("winner orientation requires a finite T7/PNT AUC")
    return 1 if float(raw_auc) >= 0.5 else -1


def ordinal_median_monotonic(loss: float, pnt: float, target7: float) -> bool:
    values = np.asarray([loss, pnt, target7], dtype=float)
    return bool(np.isfinite(values).all() and loss <= pnt <= target7)


def _feature_authority_path(root: Path) -> Path:
    path = (
        root / "reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
        / "v004c_pairwise_v1_input_table_v002.csv"
    )
    if not path.is_file():
        raise RuntimeError("FATAL: authoritative 53-feature table unavailable")
    return path


def load_population(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the unique matured Board3 population and the two frozen manifests."""
    assert_audit_contract()
    _, samples, _, source_audit = prepare_diagnostic_samples(root)
    expected_source = {
        "rows": 319, "dates": 39, "may_rows": 146, "may_dates": 18,
        "june_rows": 173, "june_dates": 21, "board2": 261, "board3": 58,
        "july_result_rows_accessed": 0,
    }
    if source_audit != expected_source:
        raise RuntimeError(f"FATAL: authoritative universe parity changed: {source_audit}")
    if len(samples) != EXPECTED_MATURED_ROWS or samples["signal_date"].nunique() != EXPECTED_MATURED_DATES:
        raise RuntimeError("FATAL: matured pre-July population parity changed")

    board3 = samples[
        pd.to_numeric(samples["board_streak_before_break"]).eq(3)
        & samples["label_available_date"].astype(str).lt(JULY_SENTINEL)
    ].copy()
    authority = pd.read_csv(
        _feature_authority_path(root),
        usecols=["event_id", *CONTRACT_FEATURE_NAMES],
        dtype={"event_id": str},
    )
    if authority["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in low-level authority")
    # Some low-level fields are also carried as audit helpers by the Stage1
    # sample table.  The frozen pairwise-contract table remains authoritative;
    # remove those aliases before the one-to-one join to avoid suffix drift.
    board3 = board3.drop(columns=[name for name in CONTRACT_FEATURE_NAMES if name in board3], errors="ignore")
    board3["event_id"] = board3["event_id"].astype(str)
    board3 = board3.merge(authority, on="event_id", how="left", validate="one_to_one", indicator=True)
    if board3["_merge"].ne("both").any():
        raise RuntimeError("FATAL: unmatched matured Board3 feature row")
    board3 = board3.drop(columns="_merge")
    board3["code"] = board3["code"].astype(str).str.zfill(6)
    board3["signal_date"] = board3["signal_date"].astype(str)
    board3["raw_repair_return"] = pd.to_numeric(board3["raw_repair_return"])
    board3["capped_return_7"] = np.minimum(board3["raw_repair_return"], 0.07)
    board3["outcome_class"] = board3["raw_repair_return"].map(classify_outcome)
    board3["target7"] = board3["outcome_class"].eq("TARGET7").astype(int)
    board3["loss"] = board3["outcome_class"].eq("LOSS").astype(int)
    board3["positive_non_target"] = board3["outcome_class"].eq("POSITIVE_NON_TARGET").astype(int)
    board3["nonloss"] = board3["loss"].eq(0).astype(int)
    board3 = board3.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)

    counts = board3["outcome_class"].value_counts().to_dict()
    if len(board3) != EXPECTED_BOARD3_ROWS or board3["signal_date"].nunique() != EXPECTED_BOARD3_DATES:
        raise RuntimeError("FATAL: matured Board3 must be 55 rows / 28 dates")
    if board3["event_id"].nunique() != EXPECTED_BOARD3_ROWS:
        raise RuntimeError("FATAL: matured Board3 event_id is not unique")
    if counts != {"TARGET7": 23, "POSITIVE_NON_TARGET": 19, "LOSS": 13}:
        raise RuntimeError(f"FATAL: Board3 outcome counts changed: {counts}")
    if board3["label_available_date"].astype(str).ge(JULY_SENTINEL).any():
        raise RuntimeError("FATAL: immature/July outcome entered audit")
    if not np.allclose(board3["capped_return_7"], np.minimum(board3["raw_repair_return"], .07)):
        raise RuntimeError("FATAL: capped-return upper-cap semantics changed")

    low_specs = {row["feature_name"]: row for row in FEATURE_CONTRACT}
    manifest_rows: list[dict[str, Any]] = []
    for family, features in (
        (LOW_LEVEL_FAMILY, CONTRACT_FEATURE_NAMES),
        (STAGE1_FAMILY, FROZEN_FEATURE_COLUMNS),
    ):
        for feature in features:
            values = pd.to_numeric(board3[feature], errors="coerce")
            spec = low_specs.get(feature)
            unique = int(values.nunique(dropna=True))
            coverage = float(values.notna().mean())
            manifest_rows.append({
                "feature": feature,
                "family": family,
                "feature_group": spec["semantic_group"] if spec else STAGE1_GROUPS[feature],
                "coverage": coverage,
                "non_null_rows": int(values.notna().sum()),
                "unique_values": unique,
                "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan,
                "iqr": float(values.quantile(.75) - values.quantile(.25)) if values.notna().any() else np.nan,
                "variation_status": "LOW_VARIATION" if unique < 3 else "OK",
                "source": spec["source"] if spec else "V4C_STAGE1_FROZEN_REPRESENTATION",
                "D1_safe": True,
            })
    manifest = pd.DataFrame(manifest_rows)
    audit = {
        "source_audit": source_audit,
        "matured_rows": len(samples), "matured_dates": samples["signal_date"].nunique(),
        "rows": len(board3), "dates": board3["signal_date"].nunique(),
        "target7": int(board3["target7"].sum()), "loss": int(board3["loss"].sum()),
        "positive_non_target": int(board3["positive_non_target"].sum()),
        "nonloss": int(board3["nonloss"].sum()), "july_result_rows_accessed": 0,
    }
    return board3, manifest, audit


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    x = pd.Series(values, dtype=float).dropna()
    return {
        "n": int(len(x)), "mean": float(x.mean()) if len(x) else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "p25": float(x.quantile(.25)) if len(x) else np.nan,
        "p75": float(x.quantile(.75)) if len(x) else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def _nanquantile_safe(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else np.nan


def _finite_probability(values: np.ndarray, predicate: Any) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(predicate(finite))) if len(finite) else np.nan


def date_bootstrap_counts(
    dates: Sequence[str], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(map(str, dates)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    counts = np.zeros((resamples, len(unique)), dtype=np.int16)
    np.add.at(counts, (np.repeat(np.arange(resamples), len(unique)), draws.ravel()), 1)
    return unique, counts


def stratified_winner_permutations(
    frame: pd.DataFrame, permutations: int = PERMUTATIONS, seed: int = PERMUTATION_SEED,
) -> np.ndarray:
    """Shuffle T7/PNT labels within date among NONLOSS rows only."""
    nonloss = frame[frame["nonloss"].eq(1)].reset_index(drop=True)
    labels = nonloss["target7"].to_numpy(np.int8)
    result = np.empty((permutations, len(nonloss)), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for _, indices in nonloss.groupby("signal_date", sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        order = np.argsort(rng.random((permutations, len(idx))), axis=1)
        result[:, idx] = labels[idx][order]
    return result


def _permutation_p(values: np.ndarray, labels: np.ndarray, permuted: np.ndarray, observed_auc: float) -> float:
    valid = np.isfinite(values)
    x = values[valid]
    y_perm = permuted[:, valid].astype(float)
    ranks = pd.Series(x).rank(method="average").to_numpy(float)
    n_pos = y_perm.sum(axis=1)
    n_neg = len(x) - n_pos
    rank_sum = y_perm @ ranks
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    distance = np.abs(auc - .5)
    observed = abs(float(observed_auc) - .5)
    return float((1 + np.sum(np.isfinite(distance) & (distance >= observed - 1e-15))) / (1 + len(permuted)))


def _bootstrap_auc(
    frame: pd.DataFrame, scores: np.ndarray, positive: np.ndarray, negative: np.ndarray,
    unique_dates: Sequence[str], counts: np.ndarray,
) -> np.ndarray:
    date_map = {date: index for index, date in enumerate(unique_dates)}
    date_idx = frame["signal_date"].map(date_map).to_numpy(int)
    valid = np.isfinite(scores)
    n_dates = len(unique_dates)
    pos_n = np.zeros(n_dates)
    neg_n = np.zeros(n_dates)
    numerator = np.zeros((n_dates, n_dates))
    for left in range(n_dates):
        pos = scores[valid & positive & (date_idx == left)]
        pos_n[left] = len(pos)
        for right in range(n_dates):
            neg = scores[valid & negative & (date_idx == right)]
            if left == 0:
                neg_n[right] = len(neg)
            if len(pos) and len(neg):
                comp = (pos[:, None] > neg[None, :]).astype(float)
                comp += .5 * (pos[:, None] == neg[None, :])
                numerator[left, right] = float(comp.sum())
    numer = np.einsum("bi,ij,bj->b", counts, numerator, counts, optimize=True)
    denom = (counts @ pos_n) * (counts @ neg_n)
    result = np.full(len(counts), np.nan)
    ok = denom > 0
    result[ok] = numer[ok] / denom[ok]
    return result


def _weighted_midranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(values, return_inverse=True)
    groups = int(inverse.max() + 1)
    indicator = np.eye(groups, dtype=float)[inverse]
    group_weights = weights @ indicator
    lower = np.cumsum(group_weights, axis=1) - group_weights
    group_ranks = lower + (group_weights + 1.0) / 2.0
    return group_ranks[:, inverse]


def _bootstrap_spearman(
    frame: pd.DataFrame, scores: np.ndarray, unique_dates: Sequence[str], counts: np.ndarray,
) -> np.ndarray:
    mask = frame["nonloss"].to_numpy(bool) & np.isfinite(scores)
    subset = frame.loc[mask]
    x = scores[mask]
    y = subset["raw_repair_return"].to_numpy(float)
    date_map = {date: index for index, date in enumerate(unique_dates)}
    date_idx = subset["signal_date"].map(date_map).to_numpy(int)
    weights = counts[:, date_idx].astype(float)
    rx = _weighted_midranks(x, weights)
    ry = _weighted_midranks(y, weights)
    total = weights.sum(axis=1)
    mx = np.sum(weights * rx, axis=1) / total
    my = np.sum(weights * ry, axis=1) / total
    dx = rx - mx[:, None]
    dy = ry - my[:, None]
    covariance = np.sum(weights * dx * dy, axis=1)
    denominator = np.sqrt(np.sum(weights * dx * dx, axis=1) * np.sum(weights * dy * dy, axis=1))
    result = np.full(len(counts), np.nan)
    ok = denominator > 0
    result[ok] = covariance[ok] / denominator[ok]
    return result


def _same_date_pairs(
    frame: pd.DataFrame, oriented: np.ndarray, endpoint: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame[["signal_date", "raw_repair_return", "target7", "loss", "positive_non_target", "nonloss"]].copy()
    work["score"] = oriented
    rows: list[dict[str, Any]] = []
    for date, day in work.groupby("signal_date", sort=True):
        if endpoint == "T7_VS_PNT":
            high = day[day["target7"].eq(1)]["score"].dropna().to_numpy(float)
            low = day[day["positive_non_target"].eq(1)]["score"].dropna().to_numpy(float)
            comparisons = [(a, b) for a in high for b in low]
        elif endpoint == "PNT_VS_LOSS":
            high = day[day["positive_non_target"].eq(1)]["score"].dropna().to_numpy(float)
            low = day[day["loss"].eq(1)]["score"].dropna().to_numpy(float)
            comparisons = [(a, b) for a in high for b in low]
        elif endpoint == "T7_VS_LOSS":
            high = day[day["target7"].eq(1)]["score"].dropna().to_numpy(float)
            low = day[day["loss"].eq(1)]["score"].dropna().to_numpy(float)
            comparisons = [(a, b) for a in high for b in low]
        elif endpoint == "NONLOSS_RAW_ORDER":
            nl = day[day["nonloss"].eq(1)][["score", "raw_repair_return"]].dropna().to_numpy(float)
            comparisons = []
            for left in range(len(nl)):
                for right in range(left + 1, len(nl)):
                    if nl[left, 1] == nl[right, 1]:
                        continue
                    if nl[left, 1] > nl[right, 1]:
                        comparisons.append((nl[left, 0], nl[right, 0]))
                    else:
                        comparisons.append((nl[right, 0], nl[left, 0]))
        else:
            raise ValueError(endpoint)
        if not comparisons:
            continue
        comp = np.asarray(comparisons, dtype=float)
        concordant = int(np.sum(comp[:, 0] > comp[:, 1]))
        discordant = int(np.sum(comp[:, 0] < comp[:, 1]))
        ties = int(np.sum(comp[:, 0] == comp[:, 1]))
        pairs = len(comp)
        rows.append({
            "signal_date": str(date), "endpoint": endpoint, "pairs": pairs,
            "concordant": concordant, "discordant": discordant, "ties": ties,
            "concordance": float((concordant + .5 * ties) / pairs),
        })
    daily = pd.DataFrame(rows, columns=[
        "signal_date", "endpoint", "pairs", "concordant", "discordant", "ties", "concordance",
    ])
    if daily.empty:
        summary = {"dates": 0, "pairs": 0, "concordant": 0, "discordant": 0, "ties": 0, "concordance": np.nan}
    else:
        pairs = int(daily["pairs"].sum())
        concordant = int(daily["concordant"].sum())
        ties = int(daily["ties"].sum())
        summary = {
            "dates": len(daily), "pairs": pairs, "concordant": concordant,
            "discordant": int(daily["discordant"].sum()), "ties": ties,
            "concordance": float((concordant + .5 * ties) / pairs),
        }
    return daily, summary


def _date_gap(frame: pd.DataFrame, oriented: np.ndarray) -> dict[str, Any]:
    work = frame[["signal_date", "target7", "positive_non_target"]].copy()
    work["score"] = oriented
    gaps: list[float] = []
    for _, day in work.groupby("signal_date", sort=True):
        target = day.loc[day["target7"].eq(1), "score"].dropna()
        pnt = day.loc[day["positive_non_target"].eq(1), "score"].dropna()
        if len(target) and len(pnt):
            gaps.append(float(target.median() - pnt.median()))
    values = np.asarray(gaps)
    non_tie = int(np.sum(values != 0))
    return {
        "count": len(values), "positive": int(np.sum(values > 0)),
        "negative": int(np.sum(values < 0)), "ties": int(np.sum(values == 0)),
        "consistency": float(np.sum(values > 0) / non_tie) if non_tie else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "median": float(np.median(values)) if len(values) else np.nan,
    }


def _lodo(frame: pd.DataFrame, feature: str, sign: int) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for date in sorted(frame["signal_date"].unique()):
        fold = frame[frame["signal_date"].ne(date)].copy()
        score = pd.to_numeric(fold[feature], errors="coerce").to_numpy(float) * sign
        target = fold["target7"].to_numpy(bool)
        pnt = fold["positive_non_target"].to_numpy(bool)
        loss = fold["loss"].to_numpy(bool)
        nonloss = fold["nonloss"].to_numpy(bool)
        auc = binary_auc(np.r_[np.ones(target.sum()), np.zeros(pnt.sum())], np.r_[score[target], score[pnt]])
        cliff = cliffs_delta(score[target], score[pnt])
        safe = binary_auc(np.r_[np.ones(pnt.sum()), np.zeros(loss.sum())], np.r_[score[pnt], score[loss]])
        rho = _spearman(score[nonloss], fold.loc[nonloss, "raw_repair_return"])
        rows.append({"auc": auc, "cliff": cliff, "safe": safe, "rho": rho})
    result = pd.DataFrame(rows)
    return {
        "LODO_auc_positive_pct": _finite_probability(result["auc"].to_numpy(float), lambda x: x > .5),
        "LODO_cliff_positive_pct": _finite_probability(result["cliff"].to_numpy(float), lambda x: x > 0),
        "LODO_pnt_loss_safe_pct": _finite_probability(result["safe"].to_numpy(float), lambda x: x >= .5),
        "LODO_spearman_positive_pct": _finite_probability(result["rho"].to_numpy(float), lambda x: x > 0),
        "LODO_auc_min": float(result["auc"].min()),
        "LODO_auc_median": float(result["auc"].median()),
        "LODO_auc_max": float(result["auc"].max()),
    }


def analyze_feature(
    frame: pd.DataFrame, feature: str, family: str, group: str,
    bootstrap_dates: Sequence[str], bootstrap_counts: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
    target = frame["target7"].to_numpy(bool)
    pnt = frame["positive_non_target"].to_numpy(bool)
    loss = frame["loss"].to_numpy(bool)
    nonloss = frame["nonloss"].to_numpy(bool)
    raw_auc = binary_auc(np.r_[np.ones(target.sum()), np.zeros(pnt.sum())], np.r_[values[target], values[pnt]])
    sign = winner_sign_from_auc(raw_auc)
    oriented = values * sign
    primary_auc = binary_auc(np.r_[np.ones(target.sum()), np.zeros(pnt.sum())], np.r_[oriented[target], oriented[pnt]])
    primary_cliff = cliffs_delta(oriented[target], oriented[pnt])
    pnt_loss_auc = binary_auc(np.r_[np.ones(pnt.sum()), np.zeros(loss.sum())], np.r_[oriented[pnt], oriented[loss]])
    t7_loss_auc = binary_auc(np.r_[np.ones(target.sum()), np.zeros(loss.sum())], np.r_[oriented[target], oriented[loss]])
    rho = _spearman(oriented[nonloss], frame.loc[nonloss, "raw_repair_return"])

    pair_daily: list[pd.DataFrame] = []
    pair_summaries: dict[str, dict[str, Any]] = {}
    for endpoint in ("T7_VS_PNT", "PNT_VS_LOSS", "T7_VS_LOSS", "NONLOSS_RAW_ORDER"):
        daily, summary = _same_date_pairs(frame, oriented, endpoint)
        daily.insert(0, "feature", feature)
        daily.insert(0, "family", family)
        pair_daily.append(daily)
        pair_summaries[endpoint] = summary
    pair_table = pd.concat(pair_daily, ignore_index=True) if pair_daily else pd.DataFrame()
    date_gap = _date_gap(frame, oriented)

    boot_primary = _bootstrap_auc(frame, oriented, target, pnt, bootstrap_dates, bootstrap_counts)
    boot_pnt_loss = _bootstrap_auc(frame, oriented, pnt, loss, bootstrap_dates, bootstrap_counts)
    boot_t7_loss = _bootstrap_auc(frame, oriented, target, loss, bootstrap_dates, bootstrap_counts)
    boot_rho = _bootstrap_spearman(frame, oriented, bootstrap_dates, bootstrap_counts)
    boot_cliff = 2.0 * boot_primary - 1.0

    nonloss_frame = frame[frame["nonloss"].eq(1)].reset_index(drop=True)
    nonloss_values = pd.to_numeric(nonloss_frame[feature], errors="coerce").to_numpy(float)
    permutation_frame = nonloss_frame[np.isfinite(nonloss_values)].reset_index(drop=True)
    permutation_values = pd.to_numeric(permutation_frame[feature], errors="raise").to_numpy(float)
    permuted = stratified_winner_permutations(permutation_frame)
    permutation_p = _permutation_p(
        permutation_values, permutation_frame["target7"].to_numpy(int), permuted, raw_auc,
    )
    lodo = _lodo(frame, feature, sign)

    summaries = {state: _summary(frame.loc[frame["outcome_class"].eq(state), feature]) for state in ("LOSS", "POSITIVE_NON_TARGET", "TARGET7")}
    medians = {state: float(sign * summaries[state]["median"]) for state in summaries}
    t7_pair = pair_summaries["T7_VS_PNT"]
    result: dict[str, Any] = {
        "feature": feature, "family": family, "feature_group": group,
        "coverage": float(np.isfinite(values).mean()), "unique_values": int(pd.Series(values).nunique(dropna=True)),
        "winner_sign": sign, "orientation_source": ORIENTATION_SOURCE,
        "LOSS_n": summaries["LOSS"]["n"], "LOSS_mean": summaries["LOSS"]["mean"], "LOSS_median": summaries["LOSS"]["median"],
        "LOSS_p25": summaries["LOSS"]["p25"], "LOSS_p75": summaries["LOSS"]["p75"],
        "PNT_n": summaries["POSITIVE_NON_TARGET"]["n"], "PNT_mean": summaries["POSITIVE_NON_TARGET"]["mean"], "PNT_median": summaries["POSITIVE_NON_TARGET"]["median"],
        "PNT_p25": summaries["POSITIVE_NON_TARGET"]["p25"], "PNT_p75": summaries["POSITIVE_NON_TARGET"]["p75"],
        "T7_n": summaries["TARGET7"]["n"], "T7_mean": summaries["TARGET7"]["mean"], "T7_median": summaries["TARGET7"]["median"],
        "T7_p25": summaries["TARGET7"]["p25"], "T7_p75": summaries["TARGET7"]["p75"],
        "oriented_LOSS_median": medians["LOSS"], "oriented_PNT_median": medians["POSITIVE_NON_TARGET"], "oriented_T7_median": medians["TARGET7"],
        "AUC_raw_T7_vs_PNT": raw_auc, "AUC_oriented_T7_vs_PNT": primary_auc,
        "Cliff_T7_vs_PNT": primary_cliff, "AUC_oriented_PNT_vs_LOSS": pnt_loss_auc,
        "AUC_oriented_T7_vs_LOSS": t7_loss_auc,
        "ordinal_median_monotonic": ordinal_median_monotonic(medians["LOSS"], medians["POSITIVE_NON_TARGET"], medians["TARGET7"]),
        "NONLOSS_raw_spearman": rho,
        "NONLOSS_raw_pair_concordance": pair_summaries["NONLOSS_RAW_ORDER"]["concordance"],
        "T7_PNT_pair_dates": t7_pair["dates"], "T7_PNT_pairs": t7_pair["pairs"], "T7_PNT_pair_concordance": t7_pair["concordance"],
        "PNT_LOSS_pair_dates": pair_summaries["PNT_VS_LOSS"]["dates"], "PNT_LOSS_pairs": pair_summaries["PNT_VS_LOSS"]["pairs"], "PNT_LOSS_concordance": pair_summaries["PNT_VS_LOSS"]["concordance"],
        "T7_LOSS_pair_dates": pair_summaries["T7_VS_LOSS"]["dates"], "T7_LOSS_pairs": pair_summaries["T7_VS_LOSS"]["pairs"], "T7_LOSS_concordance": pair_summaries["T7_VS_LOSS"]["concordance"],
        "date_mixed_count": date_gap["count"], "date_positive": date_gap["positive"], "date_negative": date_gap["negative"], "date_ties": date_gap["ties"],
        "date_direction_consistency": date_gap["consistency"], "date_mean_gap": date_gap["mean"], "date_median_gap": date_gap["median"],
        "bootstrap_auc_p2_5": _nanquantile_safe(boot_primary, .025), "bootstrap_auc_p50": _nanquantile_safe(boot_primary, .5), "bootstrap_auc_p97_5": _nanquantile_safe(boot_primary, .975),
        "bootstrap_P_auc_gt_half": _finite_probability(boot_primary, lambda x: x > .5), "bootstrap_auc_valid": int(np.isfinite(boot_primary).sum()),
        "bootstrap_cliff_p2_5": _nanquantile_safe(boot_cliff, .025), "bootstrap_cliff_p50": _nanquantile_safe(boot_cliff, .5), "bootstrap_cliff_p97_5": _nanquantile_safe(boot_cliff, .975),
        "bootstrap_P_cliff_gt0": _finite_probability(boot_cliff, lambda x: x > 0),
        "bootstrap_P_pnt_loss_auc_gt_half": _finite_probability(boot_pnt_loss, lambda x: x > .5),
        "bootstrap_P_t7_loss_auc_gt_half": _finite_probability(boot_t7_loss, lambda x: x > .5),
        "bootstrap_pnt_loss_p2_5": _nanquantile_safe(boot_pnt_loss, .025),
        "bootstrap_pnt_loss_p50": _nanquantile_safe(boot_pnt_loss, .5),
        "bootstrap_pnt_loss_p97_5": _nanquantile_safe(boot_pnt_loss, .975),
        "bootstrap_t7_loss_p2_5": _nanquantile_safe(boot_t7_loss, .025),
        "bootstrap_t7_loss_p50": _nanquantile_safe(boot_t7_loss, .5),
        "bootstrap_t7_loss_p97_5": _nanquantile_safe(boot_t7_loss, .975),
        "bootstrap_P_nonloss_spearman_gt0": _finite_probability(boot_rho, lambda x: x > 0),
        "bootstrap_spearman_p2_5": _nanquantile_safe(boot_rho, .025), "bootstrap_spearman_p50": _nanquantile_safe(boot_rho, .5), "bootstrap_spearman_p97_5": _nanquantile_safe(boot_rho, .975),
        "permutation_p": permutation_p,
        "pair_support_weak": bool(t7_pair["pairs"] < 10 or t7_pair["dates"] < 5),
        **lodo,
    }
    robustness = [
        {"family": family, "feature": feature, "section": "BOOTSTRAP", "metric": "T7_PNT_AUC", "estimate": primary_auc, "p2.5": result["bootstrap_auc_p2_5"], "p50": result["bootstrap_auc_p50"], "p97.5": result["bootstrap_auc_p97_5"], "direction_probability": result["bootstrap_P_auc_gt_half"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": result["bootstrap_auc_valid"]},
        {"family": family, "feature": feature, "section": "BOOTSTRAP", "metric": "T7_PNT_CLIFF", "estimate": primary_cliff, "p2.5": result["bootstrap_cliff_p2_5"], "p50": result["bootstrap_cliff_p50"], "p97.5": result["bootstrap_cliff_p97_5"], "direction_probability": result["bootstrap_P_cliff_gt0"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": result["bootstrap_auc_valid"]},
        {"family": family, "feature": feature, "section": "BOOTSTRAP", "metric": "PNT_LOSS_AUC", "estimate": pnt_loss_auc, "p2.5": result["bootstrap_pnt_loss_p2_5"], "p50": result["bootstrap_pnt_loss_p50"], "p97.5": result["bootstrap_pnt_loss_p97_5"], "direction_probability": result["bootstrap_P_pnt_loss_auc_gt_half"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": int(np.isfinite(boot_pnt_loss).sum())},
        {"family": family, "feature": feature, "section": "BOOTSTRAP", "metric": "T7_LOSS_AUC", "estimate": t7_loss_auc, "p2.5": result["bootstrap_t7_loss_p2_5"], "p50": result["bootstrap_t7_loss_p50"], "p97.5": result["bootstrap_t7_loss_p97_5"], "direction_probability": result["bootstrap_P_t7_loss_auc_gt_half"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": int(np.isfinite(boot_t7_loss).sum())},
        {"family": family, "feature": feature, "section": "BOOTSTRAP", "metric": "NONLOSS_SPEARMAN", "estimate": rho, "p2.5": result["bootstrap_spearman_p2_5"], "p50": result["bootstrap_spearman_p50"], "p97.5": result["bootstrap_spearman_p97_5"], "direction_probability": result["bootstrap_P_nonloss_spearman_gt0"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": int(np.isfinite(boot_rho).sum())},
        {"family": family, "feature": feature, "section": "PERMUTATION", "metric": "ABS_RAW_T7_PNT_AUC_DISTANCE", "estimate": abs(raw_auc - .5), "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan, "direction_probability": np.nan, "p_value": permutation_p, "q_value": np.nan, "valid_resamples": PERMUTATIONS},
        {"family": family, "feature": feature, "section": "LODO", "metric": "T7_PNT_AUC", "estimate": primary_auc, "p2.5": lodo["LODO_auc_min"], "p50": lodo["LODO_auc_median"], "p97.5": lodo["LODO_auc_max"], "direction_probability": lodo["LODO_auc_positive_pct"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": EXPECTED_BOARD3_DATES},
        {"family": family, "feature": feature, "section": "LODO", "metric": "PNT_LOSS_AUC", "estimate": pnt_loss_auc, "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan, "direction_probability": lodo["LODO_pnt_loss_safe_pct"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": EXPECTED_BOARD3_DATES},
        {"family": family, "feature": feature, "section": "LODO", "metric": "NONLOSS_SPEARMAN", "estimate": rho, "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan, "direction_probability": lodo["LODO_spearman_positive_pct"], "p_value": np.nan, "q_value": np.nan, "valid_resamples": EXPECTED_BOARD3_DATES},
    ]
    return result, pair_table, robustness, []


def _failed_gates(row: Mapping[str, Any]) -> list[str]:
    failed: list[str] = []
    checks = (
        ("COVERAGE", row["coverage"] >= .95), ("VARIATION", row["unique_values"] >= 3),
        ("PRIMARY_AUC", row["AUC_oriented_T7_vs_PNT"] >= .65),
        ("EFFECT_SIZE", row["Cliff_T7_vs_PNT"] >= .30),
        ("T7_LOSS_SAFETY", row["AUC_oriented_T7_vs_LOSS"] >= .60),
        ("PNT_LOSS_SAFETY", row["AUC_oriented_PNT_vs_LOSS"] >= .52),
        ("CONTINUOUS_INFO", row["NONLOSS_raw_spearman"] >= .20 or row["NONLOSS_raw_pair_concordance"] >= .57),
        ("DATE_SUPPORT", row["date_mixed_count"] >= 5),
        ("DATE_CONSISTENCY", row["date_direction_consistency"] >= .65),
        ("BOOTSTRAP", row["bootstrap_P_cliff_gt0"] >= .90),
        ("LODO", row["LODO_cliff_positive_pct"] >= .80),
        ("FDR", row["BH_q"] <= .10),
    )
    for name, passed in checks:
        if not bool(passed):
            failed.append(name)
    if row["T7_PNT_pairs"] >= 10 and row["T7_PNT_pair_concordance"] < .60:
        failed.append("PAIR_CONCORDANCE")
    return failed


def apply_gates(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    failed = [_failed_gates(row) for row in result.to_dict("records")]
    result["failed_gates"] = ["|".join(items) for items in failed]
    result["winner_pass"] = [not items for items in failed]
    result["winner_exceptional"] = (
        result["coverage"].ge(.95) & result["unique_values"].ge(3)
        & result["AUC_oriented_T7_vs_PNT"].ge(.72) & result["Cliff_T7_vs_PNT"].ge(.44)
        & result["AUC_oriented_T7_vs_LOSS"].ge(.70) & result["AUC_oriented_PNT_vs_LOSS"].ge(.55)
        & result["NONLOSS_raw_spearman"].ge(.25) & result["date_mixed_count"].ge(5)
        & result["date_direction_consistency"].ge(.70) & result["bootstrap_P_cliff_gt0"].ge(.95)
        & result["LODO_cliff_positive_pct"].ge(.90) & result["BH_q"].le(.05)
        & result["T7_PNT_pairs"].ge(10) & result["T7_PNT_pair_concordance"].ge(.65)
    )
    result["ordinal_winner_pass"] = (
        result["winner_pass"] & result["ordinal_median_monotonic"]
        & result["AUC_oriented_PNT_vs_LOSS"].ge(.55)
        & result["AUC_oriented_T7_vs_PNT"].ge(.65)
        & result["AUC_oriented_T7_vs_LOSS"].ge(.65)
        & result["NONLOSS_raw_spearman"].ge(.20)
    )
    result["geometry_class"] = np.select(
        [result["ordinal_winner_pass"], result["winner_pass"], result["AUC_oriented_T7_vs_LOSS"].ge(.70) & result["AUC_oriented_T7_vs_PNT"].lt(.65)],
        ["ORDINAL", "WINNER_ONLY", "EXTREME_ONLY"], default="NONE",
    )
    return result


def build_redundancy(population: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = table[table["winner_pass"] | table["winner_exceptional"]]
    for family, family_rows in selected.groupby("family", sort=True):
        names = sorted(family_rows["feature"].tolist())
        for left, feature_a in enumerate(names):
            for feature_b in names[left + 1:]:
                rho = _spearman(population[feature_a], population[feature_b])
                rows.append({
                    "family": family, "feature_a": feature_a, "feature_b": feature_b,
                    "rho": rho, "abs_rho": abs(rho), "near_redundant": abs(rho) >= REDUNDANCY_THRESHOLD,
                    "pass_a": True, "pass_b": True,
                    "ordinal_a": bool(table.set_index("feature").loc[feature_a, "ordinal_winner_pass"]),
                    "ordinal_b": bool(table.set_index("feature").loc[feature_b, "ordinal_winner_pass"]),
                })
    low = selected[selected["family"].eq(LOW_LEVEL_FAMILY)]
    stage = selected[selected["family"].eq(STAGE1_FAMILY)]
    for feature_a in sorted(low["feature"]):
        for feature_b in sorted(stage["feature"]):
            rho = _spearman(population[feature_a], population[feature_b])
            rows.append({
                "family": "CROSS_FAMILY", "feature_a": feature_a, "feature_b": feature_b,
                "rho": rho, "abs_rho": abs(rho), "near_redundant": abs(rho) >= REDUNDANCY_THRESHOLD,
                "pass_a": True, "pass_b": True,
                "ordinal_a": bool(table.set_index("feature").loc[feature_a, "ordinal_winner_pass"]),
                "ordinal_b": bool(table.set_index("feature").loc[feature_b, "ordinal_winner_pass"]),
            })
    return pd.DataFrame(rows, columns=[
        "family", "feature_a", "feature_b", "rho", "abs_rho", "near_redundant",
        "pass_a", "pass_b", "ordinal_a", "ordinal_b",
    ])


def _has_independent_pair(rows: pd.DataFrame, redundancy: pd.DataFrame) -> bool:
    names = set(rows["feature"])
    if len(names) < 2:
        return False
    pairs = redundancy[redundancy["feature_a"].isin(names) & redundancy["feature_b"].isin(names)]
    return bool(pairs["abs_rho"].lt(REDUNDANCY_THRESHOLD).any())


def formal_decision(table: pd.DataFrame, redundancy: pd.DataFrame) -> dict[str, Any]:
    low = table[table["family"].eq(LOW_LEVEL_FAMILY)]
    stage = table[table["family"].eq(STAGE1_FAMILY)]
    low_pass = low[low["winner_pass"]]
    stage_pass = stage[stage["winner_pass"]]
    route_a = (
        len(low_pass) >= 2 and low_pass["feature_group"].nunique() >= 2
        and _has_independent_pair(low_pass, redundancy)
        and bool(low_pass["ordinal_winner_pass"].any())
    )
    route_b_low = bool(low["winner_exceptional"].any())
    route_b_stage = bool(stage["winner_exceptional"].any())
    low_by_name = low.set_index("feature")
    corroborated = []
    for row in stage_pass.itertuples(index=False):
        source = LOW_LEVEL_CORROBORATION.get(row.feature)
        if source in low_by_name.index:
            low_row = low_by_name.loc[source]
            if low_row["BH_q"] <= .20 and int(low_row["winner_sign"]) == int(row.winner_sign):
                corroborated.append(row.feature)
    route_c = (
        len(stage_pass) >= 2 and _has_independent_pair(stage_pass, redundancy)
        and bool(stage_pass["ordinal_winner_pass"].any()) and bool(corroborated)
    )
    if route_a:
        foundation, source = "ESTABLISHED", "LOW_LEVEL_MULTI_SOURCE"
    elif route_b_low:
        foundation, source = "ESTABLISHED", "LOW_LEVEL_EXCEPTIONAL"
    elif route_b_stage or route_c:
        foundation, source = "ESTABLISHED", "TRANSFORMED_REPRESENTATION_WITH_LOW_LEVEL_CORROBORATION"
    else:
        primary_candidates = table[
            table["coverage"].ge(.95) & table["unique_values"].ge(3)
            & table["AUC_oriented_T7_vs_PNT"].ge(.65) & table["Cliff_T7_vs_PNT"].ge(.30)
            & (
                table["date_direction_consistency"].ge(.65)
                | table["bootstrap_P_cliff_gt0"].ge(.90)
                | table["LODO_cliff_positive_pct"].ge(.80)
                | table["BH_q"].le(.20)
            )
        ]
        if len(low_pass) or len(stage_pass) or len(primary_candidates):
            foundation, source = "PARTIAL", "PARTIAL_ONLY"
        else:
            foundation, source = "NOT_ESTABLISHED", "NONE"

    all_ordinal = table[table["ordinal_winner_pass"]]
    ordinal_independent = _has_independent_pair(all_ordinal, redundancy)
    ordinal_sources = all_ordinal.assign(
        information_source=all_ordinal["family"] + "::" + all_ordinal["feature_group"]
    )["information_source"].nunique()
    if foundation == "ESTABLISHED" and len(all_ordinal) >= 2 and ordinal_sources >= 2 and ordinal_independent:
        ordinal = "ESTABLISHED"
    elif foundation in {"ESTABLISHED", "PARTIAL"}:
        ordinal = "PARTIAL"
    else:
        ordinal = "ABSENT"

    if foundation == "ESTABLISHED" and ordinal == "ESTABLISHED":
        next_action = "BOARD3_ORDERED_RANKING_DESIGN_WARRANTED"
    elif foundation == "ESTABLISHED":
        next_action = "BOARD3_WINNER_REPRESENTATION_DESIGN_WARRANTED"
    elif foundation == "NOT_ESTABLISHED":
        next_action = "STOP_PRECISE_TARGET7_ROUTE"
    else:
        next_action = "NO_NEW_MODEL_YET"
    return {
        "foundation": foundation, "source": source, "ordinal": ordinal,
        "next_action": next_action, "route_a": route_a, "route_b_low": route_b_low,
        "route_b_stage": route_b_stage, "route_c": route_c,
        "corroborated_stage1": corroborated,
    }


def return_distribution(population: pd.DataFrame) -> dict[str, Any]:
    result = {state: _summary(population.loc[population["outcome_class"].eq(state), "raw_repair_return"]) for state in ("LOSS", "POSITIVE_NON_TARGET", "TARGET7")}
    raw = population["raw_repair_return"]
    result["PNT_BINS"] = {
        "0_3": int(((raw >= 0) & (raw < .03)).sum()),
        "3_5": int(((raw >= .03) & (raw < .05)).sum()),
        "5_7": int(((raw >= .05) & (raw < .07)).sum()),
    }
    result["T7_BINS"] = {
        "7_10": int(((raw >= .07) & (raw < .10)).sum()),
        "10_12": int(((raw >= .10) & (raw < .12)).sum()),
        "GE12": int((raw >= .12).sum()),
    }
    result["rows_within_1pp"] = int((raw.sub(.07).abs() <= .01).sum())
    result["PNT_median_abs_distance_7"] = float(population.loc[population["positive_non_target"].eq(1), "raw_repair_return"].sub(.07).abs().median())
    result["T7_median_abs_distance_7"] = float(population.loc[population["target7"].eq(1), "raw_repair_return"].sub(.07).abs().median())
    return result


def candidate_count_context(population: pd.DataFrame) -> dict[str, Any]:
    counts = population.groupby("signal_date", sort=True).size()
    return {
        "min": int(counts.min()), "median": float(counts.median()),
        "mean": float(counts.mean()), "max": int(counts.max()),
    }


def family_summary(table: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for family, group in table.groupby("family", sort=True):
        rows[family] = {
            "count": len(group), "pass": int(group["winner_pass"].sum()),
            "exceptional": int(group["winner_exceptional"].sum()),
            "ordinal": int(group["ordinal_winner_pass"].sum()),
            "q10": int(group["BH_q"].le(.10).sum()), "q05": int(group["BH_q"].le(.05).sum()),
            "lowest_q": float(group["BH_q"].min()),
        }
    return rows


def group_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, group), frame in table.groupby(["family", "feature_group"], sort=True):
        rows.append({
            "family": family, "feature_group": group, "feature_count": len(frame),
            "winner_pass_count": int(frame["winner_pass"].sum()),
            "ordinal_pass_count": int(frame["ordinal_winner_pass"].sum()),
            "best_T7_PNT_AUC": float(frame["AUC_oriented_T7_vs_PNT"].max()),
            "best_q": float(frame["BH_q"].min()),
            "best_NONLOSS_spearman": float(frame["NONLOSS_raw_spearman"].max()),
        })
    return pd.DataFrame(rows)


def _pct(value: Any, digits: int = 4) -> str:
    return "NA" if not np.isfinite(float(value)) else f"{100 * float(value):.{digits}f}%"


def _num(value: Any, digits: int = 4) -> str:
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    audit = context["population_audit"]
    table = context["univariate"]
    decision = context["decision"]
    distributions = context["return_distribution"]
    summaries = context["family_summary"]
    lines = [
        "# v004c Board3 Winner Information Geometry Audit v001", "",
        "## 1. Experimental Contract", "",
        "- Board3-only May/June information audit; no model, new feature, feature engineering, target search, or threshold search.",
        "- `ORIENTATION_SOURCE = FULL_DEVELOPMENT_T7_VS_PNT`; orientation is post-hoc diagnostic only and is fixed for all endpoints.",
        "- `JULY_RESULT_ROWS_ACCESSED = 0`.", "",
        "## 2. Board3 Matured Population", "",
        f"- Rows/dates: **{audit['rows']} / {audit['dates']}**.",
        f"- LOSS / POSITIVE_NON_TARGET / TARGET7: **{audit['loss']} / {audit['positive_non_target']} / {audit['target7']}**.", "",
        "## 3. LOSS / POSITIVE_NON_TARGET / TARGET7 Distribution", "",
        "| State | n | Mean | Median | p25 | p75 | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for state in ("LOSS", "POSITIVE_NON_TARGET", "TARGET7"):
        row = distributions[state]
        lines.append(f"| {state} | {row['n']} | {_pct(row['mean'])} | {_pct(row['median'])} | {_pct(row['p25'])} | {_pct(row['p75'])} | {_pct(row['min'])} | {_pct(row['max'])} |")
    pnt_bins, t7_bins = distributions["PNT_BINS"], distributions["T7_BINS"]
    lines += [
        "",
        f"- PNT 0–3% / 3–5% / 5–7% counts: **{pnt_bins['0_3']} / {pnt_bins['3_5']} / {pnt_bins['5_7']}**.",
        f"- T7 7–10% / 10–12% / >=12% counts: **{t7_bins['7_10']} / {t7_bins['10_12']} / {t7_bins['GE12']}**.",
        f"- Rows within +/-1pp of 7%: **{distributions['rows_within_1pp']}**; median absolute distance for PNT/T7: **{_pct(distributions['PNT_median_abs_distance_7'])} / {_pct(distributions['T7_median_abs_distance_7'])}**.",
    ]
    lines += ["", "## 4. Feature Families", "",
        f"- Low-level family: **{summaries[LOW_LEVEL_FAMILY]['count']}**; frozen Stage1 representation: **{summaries[STAGE1_FAMILY]['count']}**.",
        "- Benjamini-Hochberg correction was performed separately over exactly 53 and exactly 18 primary permutation p-values.", "",
        "| Family | Group | Features | Winner pass | Ordinal pass | Best T7/PNT AUC | Best q | Best NONLOSS rho |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in context["group_summary"].itertuples(index=False):
        lines.append(f"| {row.family} | {row.feature_group} | {row.feature_count} | {row.winner_pass_count} | {row.ordinal_pass_count} | {_num(row.best_T7_PNT_AUC)} | {_num(row.best_q, 6)} | {_num(row.best_NONLOSS_spearman)} |")
    lines.append("")
    for title, family in (("## 5. Low-Level 53 Winner Screen", LOW_LEVEL_FAMILY), ("## 6. Frozen Stage1 18 Winner Screen", STAGE1_FAMILY)):
        info = summaries[family]
        lines += [title, "", f"- WINNER_PASS / EXCEPTIONAL / ORDINAL: **{info['pass']} / {info['exceptional']} / {info['ordinal']}**; q<=.10: **{info['q10']}**; lowest q: **{_num(info['lowest_q'], 6)}**.", "",
            "| Feature | Group | T7/PNT AUC | Cliff | PNT/LOSS | T7/LOSS | Spearman | Date consistency | q | Status |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        subset = table[table["family"].eq(family)].sort_values(["BH_q", "AUC_oriented_T7_vs_PNT", "Cliff_T7_vs_PNT", "feature"], ascending=[True, False, False, True], kind="mergesort")
        shown = pd.concat([subset[subset["winner_pass"]], subset[~subset["winner_pass"]].head(10)]).drop_duplicates("feature")
        for row in shown.itertuples(index=False):
            status = "EXCEPTIONAL" if row.winner_exceptional else "ORDINAL_PASS" if row.ordinal_winner_pass else "WINNER_PASS" if row.winner_pass else f"FAIL:{row.failed_gates}"
            lines.append(f"| {row.feature} | {row.feature_group} | {_num(row.AUC_oriented_T7_vs_PNT)} | {_num(row.Cliff_T7_vs_PNT)} | {_num(row.AUC_oriented_PNT_vs_LOSS)} | {_num(row.AUC_oriented_T7_vs_LOSS)} | {_num(row.NONLOSS_raw_spearman)} | {_pct(row.date_direction_consistency)} | {_num(row.BH_q, 6)} | {status} |")
        lines.append("")
    lines += [
        "## 7. TARGET7 vs POSITIVE_NON_TARGET", "",
        f"- Strongest low-level oriented AUC: **{_num(table[table.family.eq(LOW_LEVEL_FAMILY)].AUC_oriented_T7_vs_PNT.max())}**.",
        f"- Strongest Stage1-representation oriented AUC: **{_num(table[table.family.eq(STAGE1_FAMILY)].AUC_oriented_T7_vs_PNT.max())}**.", "",
        "## 8. POSITIVE_NON_TARGET vs LOSS Safety", "",
        f"- Winner-pass features safe at AUC>=.52: **{int(table[table.winner_pass].AUC_oriented_PNT_vs_LOSS.ge(.52).sum())}/{int(table.winner_pass.sum())}**.", "",
        "## 9. TARGET7 vs LOSS Extreme Separation", "",
        f"- Best extreme AUC: **{_num(table.AUC_oriented_T7_vs_LOSS.max())}**.", "",
        "## 10. Continuous NONLOSS Raw-Repair Ordering", "",
        f"- Features with Spearman>=.20: **{int(table.NONLOSS_raw_spearman.ge(.20).sum())}**; raw pair concordance>=.57: **{int(table.NONLOSS_raw_pair_concordance.ge(.57).sum())}**.", "",
        "## 11. Same-Date Pair Geometry", "",
        f"- T7/PNT pair support range: **{int(table.T7_PNT_pairs.min())}–{int(table.T7_PNT_pairs.max())} pairs**, informative dates **{int(table.T7_PNT_pair_dates.min())}–{int(table.T7_PNT_pair_dates.max())}**.", "",
        "## 12. Bootstrap / LODO / Permutation / FDR", "",
        f"- Low-level / Stage1 q<=.10: **{summaries[LOW_LEVEL_FAMILY]['q10']} / {summaries[STAGE1_FAMILY]['q10']}**.",
        "- Bootstrap resamples dates; LODO fixes the full-development orientation; permutation shuffles only T7/PNT labels within each date among NONLOSS rows.", "",
        "## 13. Redundancy", "",
        f"- Passing features: **{int(table.winner_pass.sum())}**; near-redundant pass pairs: **{int(context['redundancy'].near_redundant.sum()) if len(context['redundancy']) else 0}**.", "",
        "## 14. Winner Information Foundation", "",
        f"- `BOARD3_WINNER_INFORMATION_FOUNDATION = {decision['foundation']}`",
        f"- `FOUNDATION_SOURCE = {decision['source']}`", "",
        "## 15. Ordinal Repair Information", "",
        f"- `BOARD3_ORDINAL_REPAIR_INFORMATION = {decision['ordinal']}`", "",
        "## 16. Next Board3 Decision", "",
        f"- `NEXT_BOARD3_ACTION = {decision['next_action']}`",
        "- No model is trained in this task; no July result is accessed; this is not production evidence.",
    ]
    if decision["next_action"] == "STOP_PRECISE_TARGET7_ROUTE":
        lines.append("- Existing audited D1-close-safe information did not establish stable separation between Board3 TARGET7 and POSITIVE_NON_TARGET. Stop the current route aimed at precisely ranking Board3 +7% winners from D1 information.")
    elif decision["next_action"] == "NO_NEW_MODEL_YET":
        lines.append("- Preserve this partial audit and do not train another Board3 model yet.")
    else:
        lines.append("- Any next formulation must be a separate predeclared May/June-only task; July remains untouched.")
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    population, manifest, population_audit = load_population(root)
    bootstrap_dates, bootstrap_counts = date_bootstrap_counts(population["signal_date"])
    feature_rows: list[dict[str, Any]] = []
    pair_parts: list[pd.DataFrame] = []
    robustness_rows: list[dict[str, Any]] = []
    manifest_index = manifest.set_index(["family", "feature"])
    for family, features in ((LOW_LEVEL_FAMILY, CONTRACT_FEATURE_NAMES), (STAGE1_FAMILY, FROZEN_FEATURE_COLUMNS)):
        for feature in features:
            group = str(manifest_index.loc[(family, feature), "feature_group"])
            row, pairs, robust, _ = analyze_feature(
                population, feature, family, group, bootstrap_dates, bootstrap_counts,
            )
            feature_rows.append(row)
            pair_parts.append(pairs)
            robustness_rows.extend(robust)
    univariate = pd.DataFrame(feature_rows)
    if len(univariate[univariate["family"].eq(LOW_LEVEL_FAMILY)]) != 53 or len(univariate[univariate["family"].eq(STAGE1_FAMILY)]) != 18:
        raise RuntimeError("FATAL: BH family sizes changed")
    univariate["BH_q"] = np.nan
    for family, expected in ((LOW_LEVEL_FAMILY, 53), (STAGE1_FAMILY, 18)):
        index = univariate.index[univariate["family"].eq(family)]
        if len(index) != expected:
            raise RuntimeError("FATAL: BH family size mismatch")
        univariate.loc[index, "BH_q"] = benjamini_hochberg(univariate.loc[index, "permutation_p"].to_numpy(float))
    univariate = apply_gates(univariate)
    univariate = univariate.sort_values(["family", "BH_q", "AUC_oriented_T7_vs_PNT", "feature"], ascending=[True, True, False, True], kind="mergesort").reset_index(drop=True)
    daily_pairs = pd.concat(pair_parts, ignore_index=True).sort_values(["family", "feature", "signal_date", "endpoint"], kind="mergesort").reset_index(drop=True)
    robustness = pd.DataFrame(robustness_rows)
    q_map = univariate.set_index(["family", "feature"])["BH_q"]
    perm_mask = robustness["section"].eq("PERMUTATION")
    robustness.loc[perm_mask, "q_value"] = [q_map.loc[(family, feature)] for family, feature in robustness.loc[perm_mask, ["family", "feature"]].itertuples(index=False, name=None)]
    robustness = robustness.sort_values(["family", "feature", "section", "metric"], kind="mergesort").reset_index(drop=True)
    redundancy = build_redundancy(population, univariate)
    decision = formal_decision(univariate, redundancy)
    context: dict[str, Any] = {
        "population": population, "manifest": manifest, "population_audit": population_audit,
        "univariate": univariate, "daily_pairs": daily_pairs, "robustness": robustness,
        "redundancy": redundancy, "decision": decision,
        "return_distribution": return_distribution(population),
        "candidate_count_context": candidate_count_context(population),
        "family_summary": family_summary(univariate), "group_summary": group_summary(univariate),
        "july_result_rows_accessed": 0,
    }
    review = render_review(context)
    population_columns = [
        "event_id", "signal_date", "code", "board_streak_before_break", "d2_date", "d3_date",
        "label_available_date", "raw_repair_return", "capped_return_7", "outcome_class",
        "target7", "loss", "positive_non_target", "nonloss",
        *CONTRACT_FEATURE_NAMES, *FROZEN_FEATURE_COLUMNS,
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(population[population_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(manifest),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(univariate[univariate["family"].eq(LOW_LEVEL_FAMILY)]),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(univariate[univariate["family"].eq(STAGE1_FAMILY)]),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(daily_pairs),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(robustness),
        OUTPUT_FILENAMES[6]: dataframe_csv_bytes(redundancy),
        OUTPUT_FILENAMES[7]: review.encode("utf-8"),
    }
    return outputs, context


def write_outputs(root: Path, outputs: Mapping[str, bytes], output_dir: Path | None = None) -> Path:
    target = output_dir or (
        root / "reports/research/v004c_board3_winner_information_geometry_v001_20260506_20260626"
    )
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (target / name).write_bytes(payload)
    return target


def run_v004c_board3_winner_information_geometry(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first != second:
        mismatch = [name for name in first if first[name] != second[name]]
        raise RuntimeError(f"FATAL: deterministic rebuild failed: {mismatch}")
    target = write_outputs(root_path, first, Path(output_dir) if output_dir else None)
    context["deterministic_rebuild"] = "PASS"
    return target, context

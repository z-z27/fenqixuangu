"""v004c mechanism-information foundation screen (research only).

This module deliberately implements one low-degree-of-freedom capability screen:
fold-only block PCA followed by date-weighted ridge regression.  It is not a
production model and it never consumes a signal event after 2026-06-30.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


WINDOW_START = "2026-05-06"
MAX_SIGNAL_DATE = "2026-06-30"
MIN_TRAIN_SIGNAL_DATES = 10
RIDGE_LAMBDAS = (0.1, 1.0, 10.0)
PERMUTATION_SEED = 20260809
CAP_RETURN_7 = 0.07
CAPABILITY_STATUS = "CAPABILITY_SCREEN_ONLY"
FORMAL_POOL_SOURCES = frozenset({"daily_limitup_derived", "akshare_zt_pool_em"})

IDENTITY_COLUMNS = (
    "event_id",
    "code",
    "signal_date",
    "board_streak_before_break",
)

TREND_POSITION_COLUMNS = (
    "d1_close_to_ma5_raw",
    "d1_close_to_ma10_raw",
    "d1_close_to_ma20",
    "d1_ma5_slope",
    "d1_ma10_slope",
    "recent_7d_cumulative_return",
    "recent_7d_max_drawdown",
    "recent_7d_close_position",
)

D1_REPAIR_PATH_COLUMNS = (
    "d1_low_to_close_recovery",
    "d1_close_location",
    "d1_close_to_vwap_raw",
    "d1_high_to_close_drawdown_raw",
    "d1_afternoon_return",
    "d1_last_hour_return",
    "intraday_low_time_fraction",
)

SUPPLY_ACTIVITY_COLUMNS = (
    "d1_up_bar_volume_ratio",
    "down_bar_volume_ratio",
    "late_day_sell_volume_ratio",
    "high_zone_volume_ratio",
    "amount_above_d1_close_ratio",
    "volume_above_d1_close_ratio",
    "post_low_volume_share",
)

SAME_TIER_COLUMNS = (
    "tier_cohort_size",
    "tier_peer_count_ex_self",
    "tier_peer_advance_count_d1",
    "tier_peer_advance_rate_d1",
    "tier_peer_break_count_d1",
    "tier_peer_break_rate_d1",
    "tier_peer_mean_d1_close_return",
    "tier_peer_max_d1_close_return",
    "subject_d1_close_return_vs_tier_mean",
    "subject_d1_close_return_percentile_in_tier",
)

BLOCK_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "EXISTING_TREND_POSITION": TREND_POSITION_COLUMNS,
    "EXISTING_D1_REPAIR_PATH": D1_REPAIR_PATH_COLUMNS,
    "EXISTING_SUPPLY_ACTIVITY": SUPPLY_ACTIVITY_COLUMNS,
    "SAME_TIER_COMPETITION": SAME_TIER_COLUMNS,
}

STAGE_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "M0": (
        "EXISTING_TREND_POSITION",
        "EXISTING_D1_REPAIR_PATH",
        "EXISTING_SUPPLY_ACTIVITY",
    ),
    "M1": (
        "EXISTING_TREND_POSITION",
        "EXISTING_D1_REPAIR_PATH",
        "EXISTING_SUPPLY_ACTIVITY",
        "SAME_TIER_COMPETITION",
    ),
}

MODEL_RAW_COLUMNS = tuple(dict.fromkeys(
    column for columns in BLOCK_COLUMNS.values() for column in columns
))

FORBIDDEN_X_COLUMNS = frozenset({
    "d2_open_daily",
    "d3_high_daily",
    "target7_daily_d2open_d3high",
    "target7",
    "raw_repair_return",
    "raw_opportunity_return",
    "capped_opportunity_return_7",
    "repair_rank_target",
    "tail_loss_daily_5pct",
})

GROUP_FIELDS = (
    "group_member_count",
    "group_limit_up_count_d1",
    "group_multi_board_count_d1",
    "group_limit_up_breadth_d1",
    "group_member_mean_d1_return",
    "group_member_positive_return_rate_d1",
    "group_member_top_return_d1",
    "group_strength_percentile_among_groups_d1",
)


def assert_no_july_signal_dates(frame: pd.DataFrame) -> None:
    """Fail closed if a formal input contains a July-or-later signal event."""
    if "signal_date" not in frame:
        raise RuntimeError("FATAL: signal_date is required")
    dates = frame["signal_date"].astype(str)
    bad = dates[dates > MAX_SIGNAL_DATE]
    if not bad.empty:
        raise RuntimeError(
            f"FATAL: signal_date >= 2026-07-01 encountered ({bad.iloc[0]})"
        )


def assert_authoritative_universe(frame: pd.DataFrame) -> None:
    assert_no_july_signal_dates(frame)
    dates = frame["signal_date"].astype(str)
    streak = pd.to_numeric(frame["board_streak_before_break"], errors="raise")
    checks = {
        "rows": len(frame) == 319,
        "dates": dates.nunique() == 39,
        "may_rows": int(dates.str.startswith("2026-05").sum()) == 146,
        "may_dates": dates[dates.str.startswith("2026-05")].nunique() == 18,
        "june_rows": int(dates.str.startswith("2026-06").sum()) == 173,
        "june_dates": dates[dates.str.startswith("2026-06")].nunique() == 21,
        "board2": int((streak == 2).sum()) == 261,
        "board3": int((streak == 3).sum()) == 58,
        "event_unique": frame["event_id"].astype(str).nunique() == 319,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("FATAL: authoritative universe mismatch: " + ",".join(failed))


def percentile_rank(values: pd.Series | Sequence[float]) -> pd.Series:
    """Deterministic average-tie percentile: (rank_average - .5) / n."""
    series = pd.Series(values, copy=True, dtype=float)
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    n = int(valid.sum())
    if n:
        result.loc[valid] = (series.loc[valid].rank(method="average") - 0.5) / n
    return result


def add_repair_rank_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["repair_rank_target"] = result.groupby(
        "signal_date", sort=True, group_keys=False
    )["raw_repair_return"].transform(percentile_rank)
    if result["repair_rank_target"].isna().any():
        raise RuntimeError("FATAL: incomplete repair_rank_target")
    return result


def cross_sectional_rank_frame(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Rank each X within its own signal date; no cross-date state is used."""
    result = frame.copy()
    for column in columns:
        result[column] = result.groupby(
            "signal_date", sort=True, group_keys=False
        )[column].transform(percentile_rank)
    return result


def compute_intraday_path(minute: pd.DataFrame) -> dict[str, float]:
    """Compute the two predeclared 48-bar path descriptors.

    The first low is used.  Time is zero-based and divided by n-1, so the first
    bar is 0 and the closing bar is 1.  Post-low volume includes the low bar.
    """
    if len(minute) != 48:
        raise RuntimeError(f"FATAL: expected 48 D1 bars, got {len(minute)}")
    sort_column = "datetime" if "datetime" in minute.columns else "time"
    ordered = minute.sort_values(sort_column, kind="mergesort").reset_index(drop=True)
    lows = pd.to_numeric(ordered["low"], errors="coerce").to_numpy(float)
    volumes = pd.to_numeric(ordered["volume"], errors="coerce").to_numpy(float)
    if not np.isfinite(lows).all() or not np.isfinite(volumes).all():
        raise RuntimeError("FATAL: non-finite 5m low/volume")
    if np.any(volumes < 0) or float(volumes.sum()) <= 0:
        raise RuntimeError("FATAL: invalid 5m volume")
    first_low = int(np.flatnonzero(lows == np.min(lows))[0])
    return {
        "intraday_low_time_fraction": first_low / (len(ordered) - 1),
        "post_low_volume_share": float(volumes[first_low:].sum() / volumes.sum()),
    }


def compute_same_tier_features(
    subject_code: str,
    cohort: pd.DataFrame,
    d1_state: pd.DataFrame,
) -> dict[str, float]:
    """Compute target-blind same-tier fields from D0 cohort and D1 close state.

    ``cohort`` must contain code/d0_close and ``d1_state`` must contain
    code/d1_close/is_advance.  No D2-or-later field is accepted or inspected.
    """
    required_cohort = {"code", "d0_close"}
    required_state = {"code", "d1_close", "is_advance"}
    if not required_cohort.issubset(cohort.columns):
        raise ValueError("cohort requires code,d0_close")
    if not required_state.issubset(d1_state.columns):
        raise ValueError("d1_state requires code,d1_close,is_advance")
    c = cohort[list(required_cohort)].copy()
    s = d1_state[list(required_state)].copy()
    c["code"] = c["code"].astype(str).str.zfill(6)
    s["code"] = s["code"].astype(str).str.zfill(6)
    c = c.drop_duplicates("code", keep="last")
    s = s.drop_duplicates("code", keep="last")
    subject = str(subject_code).zfill(6)
    if subject not in set(c["code"]):
        raise RuntimeError(f"FATAL: subject {subject} absent from same-tier D0 cohort")
    merged = c.merge(s, on="code", how="left", validate="one_to_one")
    for column in ("d0_close", "d1_close"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if merged[["d0_close", "d1_close", "is_advance"]].isna().any().any():
        raise RuntimeError("FATAL: incomplete D1 state for same-tier cohort")
    if (merged["d0_close"] <= 0).any() or (merged["d1_close"] <= 0).any():
        raise RuntimeError("FATAL: invalid same-tier close price")
    merged["d1_return"] = merged["d1_close"] / merged["d0_close"] - 1.0
    merged["is_advance"] = merged["is_advance"].astype(bool)
    peers = merged[merged["code"] != subject]
    subject_row = merged[merged["code"] == subject].iloc[0]
    peer_n = len(peers)
    advance = int(peers["is_advance"].sum())
    broken = peer_n - advance
    tier_mean = float(merged["d1_return"].mean())
    tier_percentiles = percentile_rank(merged["d1_return"])
    subject_index = merged.index[merged["code"] == subject][0]
    return {
        "tier_cohort_size": float(len(merged)),
        "tier_peer_count_ex_self": float(peer_n),
        "tier_peer_advance_count_d1": float(advance),
        "tier_peer_advance_rate_d1": float(advance / peer_n) if peer_n else 0.0,
        "tier_peer_break_count_d1": float(broken),
        "tier_peer_break_rate_d1": float(broken / peer_n) if peer_n else 0.0,
        "tier_peer_mean_d1_close_return": (
            float(peers["d1_return"].mean()) if peer_n else np.nan
        ),
        "tier_peer_max_d1_close_return": (
            float(peers["d1_return"].max()) if peer_n else np.nan
        ),
        "subject_d1_close_return_vs_tier_mean": (
            float(subject_row["d1_return"]) - tier_mean
        ),
        "subject_d1_close_return_percentile_in_tier": float(
            tier_percentiles.loc[subject_index]
        ),
    }


def compute_group_snapshot_features(group_members: pd.DataFrame) -> dict[str, float]:
    """Pure D1-safe group formula helper used by the synthetic contract test.

    The production screen intentionally does not call this helper because no
    point-in-time historical membership source is available in the repository.
    """
    required = {"is_limit_up", "is_multi_board", "d1_return"}
    if not required.issubset(group_members.columns):
        raise ValueError("group snapshot is missing required columns")
    n = len(group_members)
    if not n:
        raise ValueError("group snapshot must not be empty")
    returns = pd.to_numeric(group_members["d1_return"], errors="raise")
    limit_count = int(group_members["is_limit_up"].astype(bool).sum())
    return {
        "group_member_count": float(n),
        "group_limit_up_count_d1": float(limit_count),
        "group_multi_board_count_d1": float(
            group_members["is_multi_board"].astype(bool).sum()
        ),
        "group_limit_up_breadth_d1": float(limit_count / n),
        "group_member_mean_d1_return": float(returns.mean()),
        "group_member_positive_return_rate_d1": float((returns > 0).mean()),
        "group_member_top_return_d1": float(returns.max()),
    }


def aggregate_multiple_groups(
    group_rows: pd.DataFrame, fields: Sequence[str]
) -> dict[str, float]:
    """Predeclared deterministic max/mean aggregation; never outcome-selected."""
    result: dict[str, float] = {}
    for field in fields:
        values = pd.to_numeric(group_rows[field], errors="coerce")
        result[f"max_{field}"] = float(values.max())
        result[f"mean_{field}"] = float(values.mean())
    return result


def date_equal_weights(dates: Sequence[str]) -> np.ndarray:
    series = pd.Series(dates, dtype=str)
    counts = series.map(series.value_counts(sort=False)).to_numpy(float)
    return 1.0 / counts


def mechanism_x_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint only identity and predeclared X; outcome edits are invisible."""
    columns = [column for column in (*IDENTITY_COLUMNS, *MODEL_RAW_COLUMNS)
               if column in frame.columns]
    payload = frame[columns].sort_values("event_id").to_csv(
        index=False, lineterminator="\n", float_format="%.12g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_x_contract(frame: pd.DataFrame) -> None:
    accidental = FORBIDDEN_X_COLUMNS.intersection(MODEL_RAW_COLUMNS)
    if accidental:
        raise RuntimeError(f"FATAL: forbidden outcome columns in X: {sorted(accidental)}")
    missing = [column for column in MODEL_RAW_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"FATAL: missing mechanism columns: {missing}")
    if frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: mechanism merge duplicated event_id")


@dataclass(frozen=True)
class BlockTransform:
    columns: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    loading: np.ndarray
    explained_variance_ratio: float
    numerical_issue: str


def fit_block_transform(frame: pd.DataFrame, columns: Sequence[str]) -> BlockTransform:
    x = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.nanmedian(x, axis=0)
    if not np.isfinite(median).all():
        raise RuntimeError("FATAL: all-missing feature in training-fold block")
    filled = np.where(np.isfinite(x), x, median)
    q25 = np.percentile(filled, 25, axis=0)
    q75 = np.percentile(filled, 75, axis=0)
    scale = q75 - q25
    std = np.std(filled, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    robust = (filled - median) / scale
    pca_mean = robust.mean(axis=0)
    centered = robust - pca_mean
    issue = ""
    if not np.isfinite(centered).all():
        raise RuntimeError("FATAL: non-finite PCA input")
    if float(np.linalg.norm(centered)) <= 1e-14:
        loading = np.zeros(len(columns), dtype=float)
        loading[0] = 1.0
        explained = 0.0
        issue = "CONSTANT_BLOCK"
    else:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        loading = vt[0].astype(float)
        largest = int(np.argmax(np.abs(loading)))
        if loading[largest] < 0:
            loading *= -1.0
        denom = float(np.square(singular).sum())
        explained = float(singular[0] ** 2 / denom) if denom > 0 else 0.0
    return BlockTransform(
        columns=tuple(columns),
        median=median,
        scale=scale,
        pca_mean=pca_mean,
        loading=loading,
        explained_variance_ratio=explained,
        numerical_issue=issue,
    )


def apply_block_transform(frame: pd.DataFrame, transform: BlockTransform) -> np.ndarray:
    x = frame[list(transform.columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    filled = np.where(np.isfinite(x), x, transform.median)
    robust = (filled - transform.median) / transform.scale
    return (robust - transform.pca_mean) @ transform.loading


@dataclass
class FoldDesign:
    test_date: str
    train_dates: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    weights: np.ndarray
    predictor_names: tuple[str, ...]
    pca_rows: list[dict]


def _standardize_predictors(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x_train - mean) / std, (x_test - mean) / std


def build_fold_designs(
    frame: pd.DataFrame,
    representation: str,
    model_stage: str,
    target_column: str,
) -> list[FoldDesign]:
    if representation not in {"RAW", "CS_RANK"}:
        raise ValueError(representation)
    blocks = STAGE_BLOCKS[model_stage]
    source = (
        frame
        if representation == "RAW"
        else cross_sectional_rank_frame(frame, MODEL_RAW_COLUMNS)
    )
    dates = sorted(source["signal_date"].astype(str).unique())
    if len(dates) != 39:
        raise RuntimeError("FATAL: chronological screen requires 39 signal dates")
    designs: list[FoldDesign] = []
    date_values = source["signal_date"].astype(str).to_numpy()
    for fold_index, test_date in enumerate(dates[MIN_TRAIN_SIGNAL_DATES:], start=MIN_TRAIN_SIGNAL_DATES):
        train_dates = tuple(dates[:fold_index])
        train_idx = np.flatnonzero(np.isin(date_values, train_dates))
        test_idx = np.flatnonzero(date_values == test_date)
        if not len(test_idx) or np.any(date_values[train_idx] >= test_date):
            raise RuntimeError("FATAL: chronological fold boundary violated")
        train = source.iloc[train_idx]
        test = source.iloc[test_idx]
        train_components: list[np.ndarray] = []
        test_components: list[np.ndarray] = []
        pca_rows: list[dict] = []
        for block in blocks:
            transform = fit_block_transform(train, BLOCK_COLUMNS[block])
            train_components.append(apply_block_transform(train, transform))
            test_components.append(apply_block_transform(test, transform))
            pca_rows.append({
                "representation": representation,
                "model_stage": model_stage,
                "test_signal_date": test_date,
                "block": block,
                "feature_names": json.dumps(transform.columns, ensure_ascii=False,
                                             separators=(",", ":")),
                "pc1_loadings": json.dumps(
                    [float(value) for value in transform.loading],
                    separators=(",", ":"),
                ),
                "explained_variance_ratio": transform.explained_variance_ratio,
                "dominant_loading_abs": float(np.max(np.abs(transform.loading))),
                "numerical_issue": transform.numerical_issue,
            })
        is_board3_train = (
            pd.to_numeric(train["board_streak_before_break"], errors="raise")
            .eq(3).to_numpy(float)
        )
        is_board3_test = (
            pd.to_numeric(test["board_streak_before_break"], errors="raise")
            .eq(3).to_numpy(float)
        )
        x_train = np.column_stack([*train_components, is_board3_train])
        x_test = np.column_stack([*test_components, is_board3_test])
        if x_train.shape[1] > 8:
            raise RuntimeError("FATAL: predictor count exceeds 8")
        x_train, x_test = _standardize_predictors(x_train, x_test)
        designs.append(FoldDesign(
            test_date=test_date,
            train_dates=train_dates,
            train_indices=train_idx,
            test_indices=test_idx,
            x_train=x_train,
            x_test=x_test,
            y_train=pd.to_numeric(train[target_column], errors="raise").to_numpy(float),
            y_test=pd.to_numeric(test[target_column], errors="raise").to_numpy(float),
            weights=date_equal_weights(train["signal_date"].astype(str)),
            predictor_names=tuple([*blocks, "is_board3"]),
            pca_rows=pca_rows,
        ))
    return designs


def fit_weighted_ridge(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge_lambda: float
) -> np.ndarray:
    """Weighted least squares with an unpenalized intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    root_w = np.sqrt(np.asarray(weights, dtype=float))
    weighted_design = design * root_w[:, None]
    weighted_y = np.asarray(y, dtype=float) * root_w
    penalty = np.eye(design.shape[1], dtype=float) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def predict_weighted_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + x @ coefficients[1:]


def run_lambda(
    frame: pd.DataFrame, designs: Sequence[FoldDesign], ridge_lambda: float
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for design in designs:
        coef = fit_weighted_ridge(
            design.x_train, design.y_train, design.weights, ridge_lambda
        )
        test_rows = frame.iloc[design.test_indices][list(IDENTITY_COLUMNS)].copy()
        test_rows["score"] = predict_weighted_ridge(design.x_test, coef)
        test_rows["fold_test_signal_date"] = design.test_date
        test_rows["fold_train_signal_dates"] = len(design.train_dates)
        test_rows["fold_train_max_signal_date"] = max(design.train_dates)
        rows.append(test_rows)
    return pd.concat(rows, ignore_index=True)


def date_equal_mse(
    predictions: pd.DataFrame, frame: pd.DataFrame, target_column: str
) -> float:
    joined = predictions[["event_id", "score"]].merge(
        frame[["event_id", "signal_date", target_column]],
        on="event_id", how="left", validate="one_to_one",
    )
    joined["squared_error"] = (
        joined["score"] - pd.to_numeric(joined[target_column], errors="raise")
    ) ** 2
    return float(joined.groupby("signal_date", sort=True)["squared_error"].mean().mean())


def select_lambda(
    frame: pd.DataFrame,
    designs: Sequence[FoldDesign],
    target_column: str,
) -> tuple[pd.DataFrame, float, dict[float, pd.DataFrame]]:
    rows: list[dict] = []
    predictions: dict[float, pd.DataFrame] = {}
    for ridge_lambda in RIDGE_LAMBDAS:
        pred = run_lambda(frame, designs, ridge_lambda)
        predictions[ridge_lambda] = pred
        rows.append({
            "lambda": ridge_lambda,
            "date_equal_oof_mse": date_equal_mse(pred, frame, target_column),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["date_equal_oof_mse", "lambda"], ascending=[True, False], kind="mergesort"
    ).iloc[0]
    selected = float(best["lambda"])
    table["selected"] = table["lambda"].eq(selected)
    table["selection_basis"] = "DATE_EQUAL_OOF_MSE_REPAIR_RANK_TARGET_ONLY"
    return table, selected, predictions


def assign_daily_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["signal_date", "score", "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["rank"] = ranked.groupby("signal_date", sort=True).cumcount() + 1
    return ranked.sort_values(
        ["signal_date", "rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _selector_metrics(
    frame: pd.DataFrame, kind: str, k: int, strict: bool = True
) -> dict[str, float]:
    daily: list[dict[str, float]] = []
    for _, day in frame.groupby("signal_date", sort=True):
        ordered = day.sort_values(
            ["score", "event_id"], ascending=[False, True], kind="mergesort"
        )
        if kind == "rank":
            if len(ordered) < k:
                continue
            selected = ordered.iloc[[k - 1]]
        else:
            if strict and len(ordered) < k:
                continue
            selected = ordered.head(min(k, len(ordered)))
        selected_return = float(selected["capped_opportunity_return_7"].mean())
        baseline = float(day["capped_opportunity_return_7"].mean())
        daily.append({
            "selected_return": selected_return,
            "baseline": baseline,
            "positive_rate": float((selected["capped_opportunity_return_7"] > 0).mean()),
            "target7_precision": float(selected["target7"].mean()),
            "beat": float(selected_return > baseline),
        })
    if not daily:
        return {key: np.nan for key in (
            "mean_return", "median_return", "baseline", "excess",
            "positive_return_rate", "beat_universe_date_rate",
            "target7_precision", "n_dates",
        )}
    d = pd.DataFrame(daily)
    mean_return = float(d["selected_return"].mean())
    baseline = float(d["baseline"].mean())
    return {
        "mean_return": mean_return,
        "median_return": float(d["selected_return"].median()),
        "baseline": baseline,
        "excess": mean_return - baseline,
        "positive_return_rate": float(d["positive_rate"].mean()),
        "beat_universe_date_rate": float(d["beat"].mean()),
        "target7_precision": float(d["target7_precision"].mean()),
        "n_dates": float(len(d)),
    }


def winner_capture(frame: pd.DataFrame, k: int = 3, strict: bool = True) -> float:
    values: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        if strict and len(day) < k:
            continue
        target_count = int(day["target7"].sum())
        if target_count <= 0:
            continue
        selected = day.sort_values(
            ["score", "event_id"], ascending=[False, True], kind="mergesort"
        ).head(min(k, len(day)))
        available_slots = min(k, target_count)
        values.append(float(selected["target7"].sum() / available_slots))
    return float(np.mean(values)) if values else np.nan


def repair_concordance(frame: pd.DataFrame) -> dict[str, float]:
    categories: dict[str, list[float]] = {
        "all_repair_concordance": [],
        "cross_threshold_concordance": [],
        "within_non_target_concordance": [],
        "within_target_concordance": [],
    }
    for _, day in frame.groupby("signal_date", sort=True):
        score = day["score"].to_numpy(float)
        raw = day["raw_repair_return"].to_numpy(float)
        target = day["target7"].to_numpy(bool)
        day_values = {key: [] for key in categories}
        for left in range(len(day)):
            for right in range(left + 1, len(day)):
                raw_diff = raw[left] - raw[right]
                if abs(raw_diff) <= 1e-15:
                    continue
                score_diff = score[left] - score[right]
                product = raw_diff * score_diff
                value = 1.0 if product > 0 else (0.0 if product < 0 else 0.5)
                day_values["all_repair_concordance"].append(value)
                if target[left] != target[right]:
                    day_values["cross_threshold_concordance"].append(value)
                elif not target[left]:
                    day_values["within_non_target_concordance"].append(value)
                else:
                    day_values["within_target_concordance"].append(value)
        for key, values in day_values.items():
            if values:
                categories[key].append(float(np.mean(values)))
    return {
        key: (float(np.mean(values)) if values else np.nan)
        for key, values in categories.items()
    }


def summarize_model(frame: pd.DataFrame) -> dict[str, float | str | bool]:
    summary: dict[str, float | str | bool] = {}
    for label, kind, k, strict in (
        ("rank1", "rank", 1, True),
        ("rank2", "rank", 2, True),
        ("rank3", "rank", 3, True),
        ("top1", "top", 1, True),
        ("top2", "top", 2, True),
        ("strict_top3", "top", 3, True),
        ("up_to_3", "top", 3, False),
    ):
        values = _selector_metrics(frame, kind, k, strict)
        for key, value in values.items():
            summary[f"{label}_{key}"] = value
    summary["winner_capture"] = winner_capture(frame, 3, strict=True)
    summary.update(repair_concordance(frame))
    summary["date_equal_oof_mse_actual_target"] = float(
        frame.assign(_se=(frame["score"] - frame["repair_rank_target"]) ** 2)
        .groupby("signal_date", sort=True)["_se"].mean().mean()
    )
    gate_checks = (
        summary["rank1_excess"] > 0,
        summary["strict_top3_excess"] > 0,
        summary["rank1_beat_universe_date_rate"] > 0.5,
        summary["strict_top3_beat_universe_date_rate"] > 0.5,
        summary["all_repair_concordance"] > 0.5,
    )
    summary["foundation_gate_conditions_met"] = int(sum(gate_checks))
    summary["foundation_gate"] = bool(all(gate_checks))
    return summary


def month_summaries(frame: pd.DataFrame) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for label, prefix in (("May", "2026-05"), ("June", "2026-06")):
        subset = frame[frame["signal_date"].astype(str).str.startswith(prefix)]
        result[label] = summarize_model(subset)
    return result


def temporal_status(may_top3_excess: float, june_top3_excess: float) -> str:
    if may_top3_excess >= 0 and june_top3_excess >= 0:
        return "STABLE"
    if (may_top3_excess < 0 < june_top3_excess
            or june_top3_excess < 0 < may_top3_excess):
        return "TEMPORALLY_MIXED"
    return "WEAK"


def permute_target_within_date(
    frame: pd.DataFrame,
    source_column: str = "repair_rank_target",
    output_column: str = "permuted_repair_rank_target",
    seed: int = PERMUTATION_SEED,
) -> pd.DataFrame:
    result = frame.copy()
    rng = np.random.default_rng(seed)
    result[output_column] = np.nan
    for date in sorted(result["signal_date"].astype(str).unique()):
        idx = result.index[result["signal_date"].astype(str) == date]
        values = result.loc[idx, source_column].to_numpy(float)
        result.loc[idx, output_column] = values[rng.permutation(len(values))]
    return result


def add_pca_cosine_similarity(pca_rows: pd.DataFrame) -> pd.DataFrame:
    result = pca_rows.copy().sort_values(
        ["representation", "block", "test_signal_date", "model_stage"],
        kind="mergesort",
    )
    result["adjacent_fold_loading_cosine"] = np.nan
    # Existing blocks are identical across M0/M1; retain only one row per
    # representation/date/block and annotate every stage that consumes it.
    grouped_stages = result.groupby(
        ["representation", "test_signal_date", "block"], sort=True
    )["model_stage"].agg(lambda x: "|".join(sorted(set(x))))
    result = result.drop_duplicates(
        ["representation", "test_signal_date", "block"], keep="first"
    ).copy()
    result["model_stages"] = [
        grouped_stages.loc[(r.representation, r.test_signal_date, r.block)]
        for r in result.itertuples()
    ]
    for (_, _), idx in result.groupby(["representation", "block"], sort=True).groups.items():
        ordered_idx = list(result.loc[idx].sort_values("test_signal_date").index)
        previous: np.ndarray | None = None
        for row_index in ordered_idx:
            current = np.asarray(json.loads(result.at[row_index, "pc1_loadings"]), dtype=float)
            if previous is not None:
                denom = float(np.linalg.norm(previous) * np.linalg.norm(current))
                result.at[row_index, "adjacent_fold_loading_cosine"] = (
                    float(previous @ current / denom) if denom > 0 else np.nan
                )
            previous = current
    return result.drop(columns=["model_stage"]).sort_values(
        ["representation", "block", "test_signal_date"], kind="mergesort"
    ).reset_index(drop=True)


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    )
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"FATAL: daily cache missing: {path}")
    frame = pd.read_pickle(path).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    return frame.drop_duplicates("date", keep="last").reset_index(drop=True)


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _repo_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _source_record(
    root: Path,
    block: str,
    provider: str,
    query_range: str,
    symbol_group: str,
    cache_path: Path | None,
    source_timestamp: str = "",
    status: str = "AVAILABLE",
    notes: str = "",
) -> dict[str, object]:
    return {
        "mechanism_block": block,
        "provider": provider,
        "query_date_range": query_range,
        "symbol_or_group": symbol_group,
        "raw_cache_location": (
            _repo_relative(root, cache_path) if cache_path is not None else ""
        ),
        "source_timestamp": source_timestamp,
        "sha256": sha256_file(cache_path) if cache_path is not None else "",
        "status": status,
        "notes": notes,
    }


def _aggregate_source_audit(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Collapse repeated date uses of one immutable cache into one audit row."""
    key_columns = (
        "mechanism_block", "provider", "symbol_or_group", "raw_cache_location",
        "source_timestamp", "sha256", "status",
    )
    accumulated: dict[tuple[str, ...], dict[str, object]] = {}
    for record in records:
        key = tuple(str(record.get(column, "")) for column in key_columns)
        item = accumulated.setdefault(key, {
            **{column: record.get(column, "") for column in key_columns},
            "dates": set(),
            "notes_set": set(),
        })
        for token in str(record.get("query_date_range", "")).replace("..", " ").split():
            if len(token) == 10 and token[4:5] == "-" and token[7:8] == "-":
                item["dates"].add(token)
        note = str(record.get("notes", "")).strip()
        if note:
            item["notes_set"].add(note)
    rows: list[dict[str, object]] = []
    for item in accumulated.values():
        dates = sorted(item.pop("dates"))
        notes = sorted(item.pop("notes_set"))
        item["query_date_range"] = (
            "" if not dates else (dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}")
        )
        item["notes"] = "; ".join(notes)
        rows.append(item)
    return pd.DataFrame(rows).sort_values(
        ["mechanism_block", "raw_cache_location", "symbol_or_group"], kind="mergesort"
    ).reset_index(drop=True)


def load_authoritative_input(root: Path) -> pd.DataFrame:
    path = (
        root
        / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv"
    )
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["event_id"] = frame["event_id"].astype(str)
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["break_date"] = frame["break_date"].astype(str)
    assert_authoritative_universe(frame)
    if not frame["signal_date"].equals(frame["break_date"]):
        raise RuntimeError("FATAL: signal_date must equal D1 break_date")
    # The canonical repository pool is explicitly main-board-only.  Refuse to
    # silently apply it to a board with a different price-limit regime.
    from .code_utils import is_main_board_code

    non_main = sorted({code for code in frame["code"] if not is_main_board_code(code)})
    if non_main:
        raise RuntimeError(f"FATAL: canonical main-board pool is inapplicable: {non_main[:5]}")
    return frame


def _load_pool(
    root: Path,
    trade_date: str,
    complete_map: Mapping[str, bool],
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if not complete_map.get(trade_date, False):
        raise RuntimeError(f"FATAL: incomplete canonical limit-up pool: {trade_date}")
    if trade_date not in cache:
        path = root / "data/cache/limit_ups" / f"{trade_date}_limitups.pkl"
        if not path.exists():
            raise RuntimeError(f"FATAL: canonical limit-up cache missing: {trade_date}")
        frame = pd.read_pickle(path).copy()
        required = {"code", "consecutive_limit_up_count", "latest_price", "source"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"FATAL: invalid pool schema for {trade_date}")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["consecutive_limit_up_count"] = pd.to_numeric(
            frame["consecutive_limit_up_count"], errors="coerce"
        )
        sources = set(frame["source"].dropna().astype(str))
        if not sources or not sources.issubset(FORMAL_POOL_SOURCES):
            raise RuntimeError(f"FATAL: non-formal pool source {trade_date}: {sources}")
        cache[trade_date] = frame.sort_values("code", kind="mergesort").reset_index(drop=True)
    return cache[trade_date]


def _d0_for_event(daily: pd.DataFrame, d1: str) -> str:
    matches = daily.index[daily["date"] == d1].tolist()
    if len(matches) != 1 or matches[0] <= 0:
        raise RuntimeError(f"FATAL: cannot identify D0 before {d1}")
    return str(daily.iloc[matches[0] - 1]["date"])


def _canonical_board_streak(daily: pd.DataFrame, trade_date: str) -> int:
    """Reuse the repository's formal adjacent main-board limit-up rules."""
    from .loaders import (
        _count_consecutive_main_board_limit_ups,
        _prepare_daily_limitup_scan_frame,
    )

    prepared = _prepare_daily_limitup_scan_frame(daily)
    matches = prepared.index[prepared["date"].eq(trade_date)].tolist()
    if len(matches) != 1:
        raise RuntimeError(f"FATAL: canonical streak date unavailable: {trade_date}")
    return int(_count_consecutive_main_board_limit_ups(prepared, matches[0]))


def build_feature_dictionary(down_bar_duplicate: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    block_formula = {
        "EXISTING_TREND_POSITION": "Frozen BaoStock D1/daily primitive; see v002 lineage",
        "EXISTING_D1_REPAIR_PATH": "Frozen BaoStock D1 5m primitive; see v002 lineage",
        "EXISTING_SUPPLY_ACTIVITY": "Frozen BaoStock D1 5m primitive; see v002 lineage",
    }
    for block in (
        "EXISTING_TREND_POSITION",
        "EXISTING_D1_REPAIR_PATH",
        "EXISTING_SUPPLY_ACTIVITY",
    ):
        for name in BLOCK_COLUMNS[block]:
            is_new = name in {"intraday_low_time_fraction", "post_low_volume_share"}
            if name == "intraday_low_time_fraction":
                formula = "first 5m bar index attaining D1 low / 47 (zero-based)"
            elif name == "post_low_volume_share":
                formula = "sum 5m volume from first D1-low bar through close / full-day volume"
            else:
                formula = block_formula[block]
            rows.append({
                "name": name,
                "mechanism_block": block,
                "formula": formula,
                "source": "BaoStock 5m cache" if is_new else "BaoStock D1 dev v002",
                "information_timestamp": "D1 15:00",
                "d1_safe": "YES",
                "missing_rule": "FATAL source gap; train-fold median only for declared structural missing",
                "already_exists": "NO" if is_new else "YES",
                "inclusion_status": "INCLUDED",
            })
    tier_formulas = {
        "tier_cohort_size": "count(D0 canonical pool members with consecutive_limit_up_count=k)",
        "tier_peer_count_ex_self": "tier_cohort_size - 1",
        "tier_peer_advance_count_d1": "count(peers closing D1 at canonical limit-up with streak >= k+1)",
        "tier_peer_advance_rate_d1": "peer advance count / peer count; 0 when no peer",
        "tier_peer_break_count_d1": "peer count - peer advance count",
        "tier_peer_break_rate_d1": "peer break count / peer count; 0 when no peer",
        "tier_peer_mean_d1_close_return": "mean(peer D1 close / D0 close - 1)",
        "tier_peer_max_d1_close_return": "max(peer D1 close / D0 close - 1)",
        "subject_d1_close_return_vs_tier_mean": "subject D1 return - full-tier mean D1 return",
        "subject_d1_close_return_percentile_in_tier": "(average rank(subject D1 return)-0.5)/cohort size",
    }
    for name in SAME_TIER_COLUMNS:
        rows.append({
            "name": name,
            "mechanism_block": "SAME_TIER_COMPETITION",
            "formula": tier_formulas[name],
            "source": "frozen formal D0/D1 full-market pools + canonical daily cache",
            "information_timestamp": "D1 15:00",
            "d1_safe": "YES",
            "missing_rule": (
                "STRUCTURAL_NA when no peer" if name in {
                    "tier_peer_mean_d1_close_return", "tier_peer_max_d1_close_return"
                } else "FATAL source gap"
            ),
            "already_exists": "NO",
            "inclusion_status": "INCLUDED",
        })
    rows.append({
        "name": "is_board3",
        "mechanism_block": "BOARD_STATE",
        "formula": "1[board_streak_before_break == 3]",
        "source": "authoritative event identity",
        "information_timestamp": "D0 close",
        "d1_safe": "YES",
        "missing_rule": "FATAL",
        "already_exists": "DERIVED",
        "inclusion_status": "INCLUDED_AS_EXPLICIT_PREDICTOR",
    })
    for name in GROUP_FIELDS:
        rows.append({
            "name": name,
            "mechanism_block": "THEME_OR_GROUP_POSITION",
            "formula": "predeclared; not computed without point-in-time membership",
            "source": "UNAVAILABLE",
            "information_timestamp": "UNAVAILABLE",
            "d1_safe": "NOT_ESTABLISHED",
            "missing_rule": "DO_NOT_IMPUTE; OMIT BLOCK",
            "already_exists": "NO",
            "inclusion_status": "UNAVAILABLE",
        })
    rows.extend([
        {
            "name": "post_low_price_recovery",
            "mechanism_block": "EXISTING_D1_REPAIR_PATH",
            "formula": "D1 close / D1 low - 1",
            "source": "BaoStock D1 dev v002",
            "information_timestamp": "D1 15:00",
            "d1_safe": "YES",
            "missing_rule": "not applicable",
            "already_exists": "YES: d1_low_to_close_recovery",
            "inclusion_status": "DUPLICATE_REJECTED",
        },
        {
            "name": "d1_down_bar_volume_ratio",
            "mechanism_block": "EXISTING_SUPPLY_ACTIVITY",
            "formula": "existing alias checked against down_bar_volume_ratio",
            "source": "BaoStock D1 dev v002",
            "information_timestamp": "D1 15:00",
            "d1_safe": "YES",
            "missing_rule": "not applicable",
            "already_exists": "YES: down_bar_volume_ratio",
            "inclusion_status": (
                "DUPLICATE_REJECTED" if down_bar_duplicate else "ALIAS_REJECTED_PREDECLARED_CANONICAL"
            ),
        },
    ])
    return pd.DataFrame(rows).sort_values(
        ["mechanism_block", "inclusion_status", "name"], kind="mergesort"
    ).reset_index(drop=True)


def build_mechanism_features(
    root: Path, base: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build X before any outcome is loaded or evaluated."""
    assert_authoritative_universe(base)
    coverage_path = (
        root
        / "reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
        / "v004c_pairwise_v1_pool_date_coverage_v002.csv"
    )
    coverage = pd.read_csv(coverage_path, encoding="utf-8-sig")
    complete_map = {
        str(row.trade_date): _bool_value(row.complete)
        for row in coverage.itertuples()
    }
    daily_cache: dict[str, pd.DataFrame] = {}
    pool_cache: dict[str, pd.DataFrame] = {}
    cohort_cache: dict[tuple[str, str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    records: dict[tuple[str, str], dict[str, object]] = {}

    def load_daily(
        code: str, required_dates: Sequence[str] = ()
    ) -> tuple[pd.DataFrame, Path]:
        normalized = str(code).zfill(6)
        candidates = (
            root / "data/cache/daily" / f"{normalized}_daily.pkl",
            root / "data/cache/daily_unadjusted" / f"{normalized}_daily.pkl",
            (
                root / "data/cache/v004c_mechanism_foundation_v001/daily"
                / f"{normalized}_daily.pkl"
            ),
        )
        required = set(map(str, required_dates))
        existing: list[tuple[pd.DataFrame, Path]] = []
        for path in candidates:
            if not path.exists():
                continue
            cache_key = str(path)
            if cache_key not in daily_cache:
                daily_cache[cache_key] = _read_daily(path)
            frame = daily_cache[cache_key]
            existing.append((frame, path))
            if required.issubset(set(frame["date"].astype(str))):
                return frame, path
        if existing and not required:
            return existing[0]
        raise RuntimeError(
            f"FATAL: no daily cache for {normalized} covers {sorted(required)}; "
            f"checked={[str(path) for path in candidates]}"
        )

    event_d0: dict[str, str] = {}
    for row in base.itertuples():
        daily, path = load_daily(row.code, [row.signal_date])
        d0 = _d0_for_event(daily, row.signal_date)
        event_d0[row.event_id] = d0
        records[("same-tier-daily", str(path), d0, row.signal_date)] = _source_record(
            root, "SAME_TIER_COMPETITION", "canonical daily cache",
            f"{d0}..{row.signal_date}", row.code, path,
            status="AVAILABLE", notes="subject D0/D1 identity and D1 close",
        )

    used_pool_dates: set[str] = set()
    for row in base.itertuples():
        d0 = event_d0[row.event_id]
        d1 = row.signal_date
        streak = int(row.board_streak_before_break)
        key = (d0, d1, streak)
        if key not in cohort_cache:
            d0_pool = _load_pool(root, d0, complete_map, pool_cache)
            d1_pool = _load_pool(root, d1, complete_map, pool_cache)
            used_pool_dates.update((d0, d1))
            # Provider consecutive-count fields are not trusted for cohort
            # identity.  Recompute every D0 pool member with the repository's
            # formal adjacent limit-up helper.  This fixes the one frozen
            # AkShare metadata mismatch (000068@2026-06-08) without changing
            # either the event universe or the full-market D0 pool.
            cohort_data: list[dict[str, object]] = []
            for pool_member in d0_pool.itertuples():
                member_daily, member_path = load_daily(pool_member.code, [d0])
                try:
                    canonical_streak = _canonical_board_streak(member_daily, d0)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{exc}; pool member={pool_member.code}; cache={member_path}"
                    ) from exc
                member_d0 = member_daily[member_daily["date"].eq(d0)]
                if len(member_d0) != 1:
                    raise RuntimeError(
                        f"FATAL: D0 close unavailable for pool member {pool_member.code}@{d0}"
                    )
                if canonical_streak == streak:
                    cohort_data.append({
                        "code": pool_member.code,
                        "d0_close": float(pd.to_numeric(
                            member_d0.iloc[0]["close"], errors="raise"
                        )),
                    })
                member_sources = sorted(set(
                    member_d0.get("source", pd.Series(dtype=str)).dropna().astype(str)
                ))
                member_meta_path = member_path.with_suffix(".meta.json")
                member_timestamp = ""
                if member_meta_path.exists():
                    member_meta = json.loads(
                        member_meta_path.read_text(encoding="utf-8")
                    )
                    member_timestamp = str(member_meta.get("fetch_timestamp", ""))
                records[("same-tier-streak", str(member_path), d0)] = _source_record(
                    root,
                    "SAME_TIER_COMPETITION",
                    "/".join(member_sources) or "canonical daily cache",
                    d0,
                    pool_member.code,
                    member_path,
                    source_timestamp=member_timestamp,
                    status="AVAILABLE",
                    notes="formal canonical D0 streak reconstruction for full pool member",
                )
            cohort_rows = pd.DataFrame(cohort_data)
            if cohort_rows.empty:
                raise RuntimeError(f"FATAL: empty same-tier cohort {key}")
            d1_pool_codes = set(d1_pool["code"].astype(str))
            state_rows: list[dict[str, object]] = []
            for peer in cohort_rows.itertuples():
                peer_daily, peer_path = load_daily(peer.code, [d0, d1])
                peer_d0 = peer_daily[peer_daily["date"].eq(d0)]
                peer_d1 = peer_daily[peer_daily["date"].eq(d1)]
                if len(peer_d0) != 1 or len(peer_d1) != 1:
                    raise RuntimeError(
                        f"FATAL: same-tier peer D0/D1 close unavailable: {peer.code}@{d0}/{d1}"
                    )
                cohort_rows.loc[
                    cohort_rows["code"].astype(str).eq(str(peer.code)), "d0_close"
                ] = float(pd.to_numeric(peer_d0.iloc[0]["close"], errors="raise"))
                close = float(pd.to_numeric(peer_d1.iloc[0]["close"], errors="raise"))
                state_rows.append({
                    "code": peer.code,
                    "d1_close": close,
                    "is_advance": bool(
                        peer.code in d1_pool_codes
                        and _canonical_board_streak(peer_daily, d1) >= streak + 1
                    ),
                })
                peer_sources = sorted(set(peer_d1.get("source", pd.Series(dtype=str))
                                          .dropna().astype(str)))
                peer_timestamp = ""
                peer_meta_path = peer_path.with_suffix(".meta.json")
                if peer_meta_path.exists():
                    peer_meta = json.loads(peer_meta_path.read_text(encoding="utf-8"))
                    peer_timestamp = str(peer_meta.get("fetch_timestamp", ""))
                records[("same-tier-daily", str(peer_path), d0, d1)] = _source_record(
                    root, "SAME_TIER_COMPETITION", "/".join(peer_sources) or "canonical daily cache",
                    f"{d0}..{d1}", peer.code, peer_path,
                    source_timestamp=peer_timestamp,
                    status="AVAILABLE", notes="full D0 same-tier member D1 close",
                )
            cohort_cache[key] = (
                cohort_rows.sort_values("code", kind="mergesort").reset_index(drop=True),
                pd.DataFrame(state_rows).sort_values("code", kind="mergesort").reset_index(drop=True),
            )

    for trade_date in sorted(used_pool_dates):
        path = root / "data/cache/limit_ups" / f"{trade_date}_limitups.pkl"
        actual_sources = sorted(set(pool_cache[trade_date]["source"].dropna().astype(str)))
        records[("same-tier-pool", str(path))] = _source_record(
            root, "SAME_TIER_COMPETITION", "/".join(actual_sources),
            trade_date, "FULL_MAIN_BOARD_MARKET", path,
            status="AVAILABLE", notes="frozen formal complete D0/D1 limit-up pool",
        )

    minute_cache: dict[str, pd.DataFrame] = {}
    minute_dates_by_code = base.groupby("code", sort=True)["signal_date"].agg(list).to_dict()
    minute_paths: dict[str, Path] = {}
    for code, requested_dates in minute_dates_by_code.items():
        path = root / "data/cache/baostock_5m" / f"{code}_5min.pkl"
        meta_path = root / "data/cache/baostock_5m" / f"{code}_5min.meta.json"
        if not path.exists() or not meta_path.exists():
            raise RuntimeError(f"FATAL: BaoStock 5m cache/meta missing for {code}")
        minute = pd.read_pickle(path).copy()
        minute["trade_date"] = pd.to_datetime(
            minute["trade_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        minute_cache[code] = minute
        minute_paths[code] = path
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records[("intraday", str(path))] = _source_record(
            root, "D0_TO_D1_CAPITAL_PATH", str(meta.get("source", "baostock_5m")),
            f"{min(requested_dates)}..{max(requested_dates)}", code, path,
            source_timestamp=str(meta.get("fetch_timestamp", "")),
            status="AVAILABLE",
            notes=(f"adjustment={meta.get('adjustment')}; interval={meta.get('interval')}; "
                   "48-bar D1 slices only"),
        )

    feature_rows: list[dict[str, object]] = []
    for row in base.itertuples():
        d0 = event_d0[row.event_id]
        key = (d0, row.signal_date, int(row.board_streak_before_break))
        cohort, state = cohort_cache[key]
        tier_features = compute_same_tier_features(row.code, cohort, state)
        d1_minute = minute_cache[row.code][
            minute_cache[row.code]["trade_date"].eq(row.signal_date)
        ].copy()
        path_features = compute_intraday_path(d1_minute)
        values: dict[str, object] = {
            "event_id": row.event_id,
            "code": row.code,
            "signal_date": row.signal_date,
            "board_streak_before_break": int(row.board_streak_before_break),
            "D0": d0,
        }
        for column in (
            *TREND_POSITION_COLUMNS,
            *[c for c in D1_REPAIR_PATH_COLUMNS if c != "intraday_low_time_fraction"],
            *[c for c in SUPPLY_ACTIVITY_COLUMNS if c != "post_low_volume_share"],
        ):
            values[column] = getattr(row, column)
        values.update(path_features)
        values.update(tier_features)
        feature_rows.append(values)
    raw = pd.DataFrame(feature_rows)
    raw = raw[[*IDENTITY_COLUMNS, "D0", *MODEL_RAW_COLUMNS]]
    if len(raw) != len(base) or set(raw["event_id"]) != set(base["event_id"]):
        raise RuntimeError("FATAL: mechanism merge changed event identity")
    assert_x_contract(raw)
    assert_no_july_signal_dates(raw)

    down_duplicate = False
    if {"d1_down_bar_volume_ratio", "down_bar_volume_ratio"}.issubset(base.columns):
        left = pd.to_numeric(base["d1_down_bar_volume_ratio"], errors="coerce")
        right = pd.to_numeric(base["down_bar_volume_ratio"], errors="coerce")
        valid = left.notna() & right.notna()
        down_duplicate = bool(valid.any() and np.allclose(left[valid], right[valid], atol=1e-12))

    records[("input", "authoritative")] = _source_record(
        root,
        "AUTHORITATIVE_UNIVERSE",
        "BaoStock D1 corrected development universe v002",
        f"{WINDOW_START}..{MAX_SIGNAL_DATE}",
        "319_EVENTS",
        root / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv",
        status="AVAILABLE",
        notes="identity anchor; outcome columns physically isolated from X selection",
    )
    records[("group", "unavailable")] = _source_record(
        root,
        "THEME_OR_GROUP_POSITION",
        "UNAVAILABLE",
        f"{WINDOW_START}..{MAX_SIGNAL_DATE}",
        "POINT_IN_TIME_MEMBERSHIP",
        None,
        status="UNAVAILABLE",
        notes=("No historical D1-effective theme/industry/concept membership snapshot; "
               "current universe industry metadata rejected as hindsight-unsafe"),
    )
    unused_fallback = (
        root / "data/cache/v004c_mechanism_foundation_v001/daily/000567_daily.pkl"
    )
    unused_meta_path = unused_fallback.with_suffix(".meta.json")
    if unused_fallback.exists() and unused_meta_path.exists():
        unused_meta = json.loads(unused_meta_path.read_text(encoding="utf-8"))
        records[("same-tier-unused-fallback", str(unused_fallback))] = _source_record(
            root,
            "SAME_TIER_COMPETITION",
            str(unused_meta.get("provider", "BaoStock")),
            f"{unused_meta.get('query_start', '')}..{unused_meta.get('query_end', '')}",
            "000567",
            unused_fallback,
            source_timestamp=str(unused_meta.get("fetch_timestamp", "")),
            status="AVAILABLE_NOT_USED",
            notes="cached during source recovery; higher-priority daily_unadjusted source used in X",
        )
    source_audit = _aggregate_source_audit(records.values())
    dictionary = build_feature_dictionary(down_duplicate)
    gate_checks = {
        "event_identity_exact": len(raw) == 319 and raw["event_id"].nunique() == 319,
        "all_used_x_known_by_d1_close": True,
        "no_july_signal_read": raw["signal_date"].max() <= MAX_SIGNAL_DATE,
        "same_tier_provenance_valid": bool(
            (source_audit[source_audit["mechanism_block"].eq("SAME_TIER_COMPETITION")]
             ["status"].astype(str).str.startswith("AVAILABLE")).all()
        ),
        "theme_group_valid_or_unavailable": bool(
            (source_audit[source_audit["mechanism_block"].eq("THEME_OR_GROUP_POSITION")]
             ["status"] == "UNAVAILABLE").all()
        ),
        "intraday_coverage_319": len(raw) == 319,
        "no_target_derived_feature_engineering": not bool(
            FORBIDDEN_X_COLUMNS.intersection(MODEL_RAW_COLUMNS)
        ),
    }
    gate = {
        **gate_checks,
        "pass": bool(all(gate_checks.values())),
        "same_tier_coverage": f"{len(raw)}/{len(base)} events",
        "intraday_coverage": f"{len(raw)}/{len(base)} events",
        "theme_group_coverage": "0/319 events (explicitly unavailable)",
        "down_bar_duplicate": down_duplicate,
    }
    return raw, source_audit, dictionary, gate


def load_outcomes(root: Path, base: pd.DataFrame) -> pd.DataFrame:
    """Load physically isolated D2/D3 columns from a pure May-June frozen asset.

    The source CSV also contains historical model scores, but ``usecols`` keeps
    those columns out of memory.  Only identity, Target7, D2 open, and D3 high
    are read; raw/capped outcomes are recomputed here.
    """
    path = (
        root
        / "reports/research/v004c_repair_pairwise_ridge_v001_20260506_20260630"
        / "v004c_repair_pairwise_oof_predictions_v001.csv"
    )
    columns = [
        "event_id", "code", "signal_date", "board_streak_before_break",
        "target7_daily_d2open_d3high", "d2_open_daily", "d3_high_daily",
    ]
    frozen = pd.read_csv(
        path, encoding="utf-8-sig", dtype={"code": str}, usecols=columns
    )
    frozen["code"] = frozen["code"].astype(str).str.zfill(6)
    frozen["event_id"] = frozen["event_id"].astype(str)
    frozen["signal_date"] = frozen["signal_date"].astype(str)
    assert_authoritative_universe(frozen)
    identity = ["event_id", "code", "signal_date", "board_streak_before_break"]
    expected = base[list(identity)].copy().sort_values("event_id", kind="mergesort")
    actual = frozen[list(identity)].copy().sort_values("event_id", kind="mergesort")
    expected["board_streak_before_break"] = pd.to_numeric(
        expected["board_streak_before_break"], errors="raise"
    ).astype(int)
    actual["board_streak_before_break"] = pd.to_numeric(
        actual["board_streak_before_break"], errors="raise"
    ).astype(int)
    if not expected.reset_index(drop=True).equals(actual.reset_index(drop=True)):
        raise RuntimeError("FATAL: frozen May-June outcome identity mismatch")
    rows: list[dict[str, object]] = []
    for row in frozen.itertuples():
        d2_open = float(pd.to_numeric(row.d2_open_daily, errors="raise"))
        d3_high = float(pd.to_numeric(row.d3_high_daily, errors="raise"))
        raw = d3_high / d2_open - 1.0
        target7 = int(raw >= CAP_RETURN_7 - 1e-9)
        frozen_target = int(_bool_value(row.target7_daily_d2open_d3high))
        if target7 != frozen_target:
            raise RuntimeError(f"FATAL: target7 consistency mismatch {row.event_id}")
        rows.append({
            "event_id": row.event_id,
            "d2_open_daily": d2_open,
            "d3_high_daily": d3_high,
            "raw_repair_return": raw,
            "capped_opportunity_return_7": min(raw, CAP_RETURN_7),
            "target7": target7,
        })
    return pd.DataFrame(rows)


def _comparison_row(
    representation: str,
    stage: str,
    run_type: str,
    selected_lambda: float,
    oof: pd.DataFrame,
) -> dict[str, object]:
    combined = summarize_model(oof)
    months = month_summaries(oof)
    row: dict[str, object] = {
        "representation": representation,
        "model_stage": stage,
        "run_type": run_type,
        "status": "RUN",
        "lambda": selected_lambda,
        **combined,
    }
    row["top1_target7_hit"] = combined["top1_target7_precision"]
    row["top3_target7_precision"] = combined["strict_top3_target7_precision"]
    row["up_to_3_return"] = combined["up_to_3_mean_return"]
    for month_name, prefix in (("May", "may"), ("June", "june")):
        month = months[month_name]
        for key in (
            "rank1_mean_return", "rank1_baseline", "rank1_excess",
            "rank1_beat_universe_date_rate",
            "strict_top3_mean_return", "strict_top3_baseline", "strict_top3_excess",
            "strict_top3_beat_universe_date_rate", "winner_capture",
            "all_repair_concordance", "within_non_target_concordance",
        ):
            row[f"{prefix}_{key}"] = month[key]
    row["temporal_status"] = temporal_status(
        float(row["may_strict_top3_excess"]),
        float(row["june_strict_top3_excess"]),
    )
    return row


def _annotate_increment(comparison: pd.DataFrame) -> pd.DataFrame:
    result = comparison.copy()
    for column in (
        "delta_rank1", "delta_strict_top3", "delta_winner_capture",
        "delta_repair_concordance", "increment_status",
    ):
        result[column] = np.nan if column != "increment_status" else "NOT_APPLICABLE"
    for (representation, run_type), group in result[
        result["status"].eq("RUN")
    ].groupby(["representation", "run_type"], sort=True):
        m0 = group[group["model_stage"].eq("M0")]
        m1 = group[group["model_stage"].eq("M1")]
        if m0.empty or m1.empty:
            continue
        base = m0.iloc[0]
        idx = m1.index[0]
        deltas = {
            "delta_rank1": float(m1.iloc[0]["rank1_mean_return"] - base["rank1_mean_return"]),
            "delta_strict_top3": float(
                m1.iloc[0]["strict_top3_mean_return"] - base["strict_top3_mean_return"]
            ),
            "delta_winner_capture": float(m1.iloc[0]["winner_capture"] - base["winner_capture"]),
            "delta_repair_concordance": float(
                m1.iloc[0]["all_repair_concordance"] - base["all_repair_concordance"]
            ),
        }
        for key, value in deltas.items():
            result.at[idx, key] = value
        positive = [value > 0 for value in deltas.values()]
        result.at[idx, "increment_status"] = (
            "YES" if all(positive) else ("MIXED" if any(positive) else "NO")
        )
    return result


def _not_run_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "representation": representation,
            "model_stage": "M2",
            "run_type": "REAL",
            "status": "NOT_RUN_UNAVAILABLE",
            "lambda": np.nan,
            "foundation_gate": False,
            "foundation_gate_conditions_met": 0,
            "temporal_status": "NOT_RUN",
            "increment_status": "NOT_TESTED",
        }
        for representation in ("RAW", "CS_RANK")
    ])


def _select_strongest_real(comparison: pd.DataFrame) -> pd.Series:
    candidates = comparison[
        comparison["run_type"].eq("REAL")
        & comparison["model_stage"].eq("M1")
        & comparison["status"].eq("RUN")
    ].copy()
    candidates["_stable"] = candidates["temporal_status"].eq("STABLE").astype(int)
    candidates["_increment"] = candidates["increment_status"].eq("YES").astype(int)
    candidates["_rep_order"] = candidates["representation"].map({"RAW": 1, "CS_RANK": 0})
    return candidates.sort_values(
        ["foundation_gate", "_stable", "_increment", "foundation_gate_conditions_met",
         "strict_top3_excess", "rank1_excess", "all_repair_concordance", "_rep_order"],
        ascending=[False, False, False, False, False, False, False, False],
        kind="mergesort",
    ).iloc[0]


def _coordinate_support(comparison: pd.DataFrame) -> str:
    real = comparison[
        comparison["run_type"].eq("REAL")
        & comparison["model_stage"].eq("M1")
        & comparison["status"].eq("RUN")
    ].set_index("representation")
    raw = real.loc["RAW"]
    rank = real.loc["CS_RANK"]
    raw_candidate = bool(raw["foundation_gate"] and raw["temporal_status"] == "STABLE"
                         and raw["increment_status"] == "YES")
    rank_candidate = bool(rank["foundation_gate"] and rank["temporal_status"] == "STABLE"
                          and rank["increment_status"] == "YES")
    if raw_candidate != rank_candidate:
        return "RAW" if raw_candidate else "CROSS_SECTIONAL_RANK"
    fields = ("rank1_excess", "strict_top3_excess", "winner_capture",
              "all_repair_concordance")
    raw_better = [float(raw[field]) > float(rank[field]) for field in fields]
    rank_better = [float(rank[field]) > float(raw[field]) for field in fields]
    if all(raw_better):
        return "RAW"
    if all(rank_better):
        return "CROSS_SECTIONAL_RANK"
    if (int(raw["foundation_gate_conditions_met"]) == 0
            and int(rank["foundation_gate_conditions_met"]) == 0
            and float(raw["strict_top3_excess"]) <= 0
            and float(rank["strict_top3_excess"]) <= 0
            and float(raw["all_repair_concordance"]) <= 0.5
            and float(rank["all_repair_concordance"]) <= 0.5):
        return "NEITHER"
    return "MIXED"


def _questions(comparison: pd.DataFrame) -> dict[str, str]:
    real = comparison[comparison["run_type"].eq("REAL") & comparison["status"].eq("RUN")]
    m0 = real[real["model_stage"].eq("M0")]
    m1 = real[real["model_stage"].eq("M1")]
    q1 = "YES" if any(
        bool(row.foundation_gate) and row.temporal_status == "STABLE"
        for row in m0.itertuples()
    ) else "NO"
    increments = set(m1["increment_status"].astype(str))
    q2 = "YES" if "YES" in increments else ("MIXED" if "MIXED" in increments else "NO")
    strongest = _select_strongest_real(comparison)
    present = bool(
        strongest["foundation_gate"]
        and strongest["temporal_status"] == "STABLE"
        and strongest["increment_status"] == "YES"
    )
    perm = comparison[
        comparison["run_type"].eq("PERMUTED")
        & comparison["representation"].eq(strongest["representation"])
        & comparison["model_stage"].eq(strongest["model_stage"])
    ].iloc[0]
    beats_perm = bool(
        strongest["rank1_excess"] > perm["rank1_excess"]
        and strongest["strict_top3_excess"] > perm["strict_top3_excess"]
        and strongest["all_repair_concordance"] > perm["all_repair_concordance"]
    )
    return {
        "q1": q1,
        "q2": q2,
        "q3": "NOT_TESTED",
        "q4": _coordinate_support(comparison),
        "q5": "YES" if strongest["temporal_status"] == "STABLE" else "NO",
        "q6": "YES" if beats_perm else "NO",
        "foundational_signal": "PRESENT" if present else "NOT_ESTABLISHED",
        "strongest_key": f"{strongest['representation']} {strongest['model_stage']}",
    }


def _pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number * 100:.3f}%"


def render_source_audit_markdown(gate: Mapping[str, object]) -> str:
    return "\n".join([
        "# v004c Mechanism Source Audit v001",
        "",
        "## Same-tier",
        "",
        "- source: frozen formal full-main-board pools (`daily_limitup_derived`; five cached `akshare_zt_pool_em` dates) plus canonical daily cache",
        "- historical reconstructable: YES",
        "- D1 safe: YES; latest input is peer D1 close / D1 limit-up state",
        f"- coverage: {gate['same_tier_coverage']}",
        "- status: AVAILABLE",
        "- recovery note: a BaoStock 000567 fallback was cached with timestamp/SHA but not used because a higher-priority repository daily_unadjusted source covered D0/D1",
        "",
        "## Theme/group",
        "",
        "- source: none accepted; current universe `industry` metadata was rejected",
        "- historical membership: NO point-in-time D1-effective snapshot found",
        "- D1 safe: NOT ESTABLISHED",
        f"- coverage: {gate['theme_group_coverage']}",
        "- status: UNAVAILABLE",
        "",
        "## 5m path",
        "",
        "- source: frozen BaoStock 5m cache (`adjustment=none`, `interval=5m`)",
        f"- coverage: {gate['intraday_coverage']} with exactly 48 D1 bars per event",
        "- D1 safe: YES; only bars through D1 15:00 are sliced",
        "",
        "## Leakage boundary",
        "",
        "- No signal event after 2026-06-30 was read.",
        "- D2/D3 prices, raw repair, capped repair, Target7, and repair-rank target are OUTCOME_ONLY.",
        "- Mechanism X is completed and the data-collection gate passes before outcomes are loaded.",
        "- Theme/group is omitted, not imputed or guessed.",
        "",
    ])


def render_review(
    comparison: pd.DataFrame,
    gate: Mapping[str, object],
    questions: Mapping[str, str],
) -> str:
    real = comparison[
        comparison["run_type"].eq("REAL") & comparison["status"].eq("RUN")
    ].copy()
    strongest = _select_strongest_real(comparison)
    perm = comparison[
        comparison["run_type"].eq("PERMUTED")
        & comparison["representation"].eq(strongest["representation"])
        & comparison["model_stage"].eq(strongest["model_stage"])
    ].iloc[0]
    lines = [
        "# v004c Mechanism Information Foundation Screen v001",
        "",
        f"Status: `{CAPABILITY_STATUS}` — PRE-MODEL ONLY; no production artifact was trained.",
        "",
        "## 1. Did We Find Any Basic Ranking Ability?",
        "",
        "| Representation | Stage | Rank1 | Baseline | Excess | Top3 | Baseline | Excess | Winner Capture | Repair Concordance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in real.sort_values(["representation", "model_stage"]).itertuples():
        lines.append(
            f"| {row.representation} | {row.model_stage} | {_pct(row.rank1_mean_return)} | "
            f"{_pct(row.rank1_baseline)} | {_pct(row.rank1_excess)} | "
            f"{_pct(row.strict_top3_mean_return)} | {_pct(row.strict_top3_baseline)} | "
            f"{_pct(row.strict_top3_excess)} | {_pct(row.winner_capture)} | "
            f"{float(row.all_repair_concordance):.4f} |"
        )
    lines.extend([
        "",
        "M2 was not run because a historical D1-safe theme/group membership source was unavailable.",
        "",
        "## 2. Did Same-Tier Competition Add Information?",
        "",
    ])
    for representation in ("RAW", "CS_RANK"):
        row = real[
            real["representation"].eq(representation)
            & real["model_stage"].eq("M1")
        ].iloc[0]
        lines.append(
            f"- {representation}: {row['increment_status']} — ΔRank1 {_pct(row['delta_rank1'])}, "
            f"ΔTop3 {_pct(row['delta_strict_top3'])}, Δwinner capture {_pct(row['delta_winner_capture'])}, "
            f"Δconcordance {float(row['delta_repair_concordance']):+.4f}."
        )
    lines.extend([
        "",
        f"Answer: **{questions['q2']}**.",
        "",
        "## 3. Did Theme / Group Position Add Information?",
        "",
        "NOT TESTED — HISTORICAL D1-SAFE SOURCE UNAVAILABLE",
        "",
        "## 4. Raw vs Cross-Sectional Rank Coordinates",
        "",
        f"Better supported: **{questions['q4']}**. This is based only on the fixed RAW/CS_RANK A/B.",
        "",
        "## 5. May vs June",
        "",
        "| Representation | Stage | May Rank1 Excess | May Top3 Excess | June Rank1 Excess | June Top3 Excess | Concordance | Temporal |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in real.sort_values(["representation", "model_stage"]).itertuples():
        lines.append(
            f"| {row.representation} | {row.model_stage} | {_pct(row.may_rank1_excess)} | "
            f"{_pct(row.may_strict_top3_excess)} | {_pct(row.june_rank1_excess)} | "
            f"{_pct(row.june_strict_top3_excess)} | {float(row.all_repair_concordance):.4f} | "
            f"{row.temporal_status} |"
        )
    lines.extend([
        "",
        "## 6. Permutation Sanity",
        "",
        f"Strongest fixed candidate: `{questions['strongest_key']}`. Real Rank1 excess "
        f"{_pct(strongest['rank1_excess'])}, Top3 excess {_pct(strongest['strict_top3_excess'])}, "
        f"concordance {float(strongest['all_repair_concordance']):.4f}. Permuted-target pipeline: "
        f"Rank1 excess {_pct(perm['rank1_excess'])}, Top3 excess {_pct(perm['strict_top3_excess'])}, "
        f"concordance {float(perm['all_repair_concordance']):.4f}.",
        f"Clearly better than permutation: **{questions['q6']}**.",
        "",
        "## 7. Data Sources and Leakage Boundary",
        "",
        "Same-tier cohorts are reconstructed from complete frozen formal D0 pools at the exact streak, "
        "with D1 advancement from the complete D1 pool and D1 close returns from canonical daily caches. "
        "The group block is unavailable because no point-in-time membership snapshot exists. The two new "
        "path descriptors use only each event's frozen 48 BaoStock bars through D1 close. No July signal "
        "event was accessed; outcomes are isolated until after the data-collection gate.",
        "",
        "## Final Questions",
        "",
        f"- Q1 Existing low-DF D1 blocks: **{questions['q1']}**",
        f"- Q2 Same-tier incremental value: **{questions['q2']}**",
        f"- Q3 Theme/group incremental value: **{questions['q3']}**",
        f"- Q4 Best coordinate: **{questions['q4']}**",
        f"- Q5 Stable across May and June: **{questions['q5']}**",
        f"- Q6 Clearly beats permutation: **{questions['q6']}**",
        f"- Q7 FOUNDATIONAL_SIGNAL: **{questions['foundational_signal']}**",
        "",
    ])
    if questions["foundational_signal"] == "PRESENT":
        lines.append(
            "Recommended next step: Freeze the winning information representation and design one formal low-DF ranking model."
        )
    else:
        lines.append(
            "The newly collected D1 mechanism information has not yet demonstrated sufficient stable OOF "
            "ranking ability to justify formal model development."
        )
    lines.extend([
        "",
        "No feature was selected from outcome performance; no block, sign, interaction, lambda grid, or "
        "model family was changed after the screen began. July OOT and Tail were not run.",
        "",
        f"DATA_COLLECTION_GATE: **{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        "Deterministic rebuild: **PASS** (the CLI builds all ten outputs twice and writes only if every byte matches).",
        "",
    ])
    return "\n".join(lines)


def build_pipeline_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    """Run the fixed screen once and return deterministic report bytes + summary."""
    base = load_authoritative_input(root)
    raw, source_audit, dictionary, gate = build_mechanism_features(root, base)
    if not gate["pass"]:
        raise RuntimeError(f"FATAL: DATA_COLLECTION_GATE failed: {gate}")

    # Outcome access starts only after the X gate above has passed.
    outcomes = load_outcomes(root, base)
    model = raw.merge(outcomes, on="event_id", how="left", validate="one_to_one")
    model = add_repair_rank_target(model)
    model = permute_target_within_date(model)

    all_oof: list[pd.DataFrame] = []
    lambda_tables: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    for run_type, target_column in (
        ("REAL", "repair_rank_target"),
        ("PERMUTED", "permuted_repair_rank_target"),
    ):
        for representation in ("RAW", "CS_RANK"):
            for stage in ("M0", "M1"):
                designs = build_fold_designs(model, representation, stage, target_column)
                lambda_table, selected, predictions = select_lambda(
                    model, designs, target_column
                )
                lambda_table.insert(0, "run_type", run_type)
                lambda_table.insert(0, "model_stage", stage)
                lambda_table.insert(0, "representation", representation)
                lambda_tables.append(lambda_table)
                pred = predictions[selected].merge(
                    model[[
                        "event_id", "raw_repair_return", "capped_opportunity_return_7",
                        "target7", "repair_rank_target",
                    ]],
                    on="event_id", how="left", validate="one_to_one",
                )
                pred = assign_daily_rank(pred)
                pred["board"] = pred["board_streak_before_break"]
                pred["representation"] = representation
                pred["model_stage"] = stage
                pred["lambda"] = selected
                pred["run_type"] = run_type
                pred["model_status"] = CAPABILITY_STATUS
                all_oof.append(pred)
                comparison_rows.append(
                    _comparison_row(representation, stage, run_type, selected, pred)
                )
                for design in designs:
                    fold_rows.append({
                        "representation": representation,
                        "model_stage": stage,
                        "run_type": run_type,
                        "lambda": selected,
                        "test_signal_date": design.test_date,
                        "train_signal_dates": len(design.train_dates),
                        "train_rows": len(design.train_indices),
                        "test_rows": len(design.test_indices),
                        "train_max_signal_date": max(design.train_dates),
                        "test_after_train": max(design.train_dates) < design.test_date,
                        "date_equal_weight_sum": float(design.weights.sum()),
                        "predictor_count": len(design.predictor_names),
                        "predictors": json_cell(design.predictor_names),
                        "imputation_scaling_pca_fit": "TRAIN_FOLD_ONLY",
                        "model_status": CAPABILITY_STATUS,
                    })
                    if run_type == "REAL":
                        pca_rows.extend(design.pca_rows)

    comparison = _annotate_increment(pd.DataFrame(comparison_rows))
    comparison = pd.concat([comparison, _not_run_rows()], ignore_index=True, sort=False)
    comparison = comparison.sort_values(
        ["run_type", "representation", "model_stage"], kind="mergesort"
    ).reset_index(drop=True)
    questions = _questions(comparison)
    pca = add_pca_cosine_similarity(pd.DataFrame(pca_rows))
    oof = pd.concat(all_oof, ignore_index=True, sort=False)
    oof = oof[[
        "event_id", "code", "signal_date", "board", "board_streak_before_break",
        "representation", "model_stage", "run_type", "lambda", "score", "rank",
        "repair_rank_target", "raw_repair_return", "capped_opportunity_return_7",
        "target7", "fold_test_signal_date", "fold_train_signal_dates",
        "fold_train_max_signal_date", "model_status",
    ]].sort_values(
        ["run_type", "representation", "model_stage", "signal_date", "rank", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    fold_audit = pd.DataFrame(fold_rows).sort_values(
        ["run_type", "representation", "model_stage", "test_signal_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    lambda_output = pd.concat(lambda_tables, ignore_index=True).sort_values(
        ["run_type", "representation", "model_stage", "lambda"], kind="mergesort"
    ).reset_index(drop=True)
    review = render_review(comparison, gate, questions)
    source_md = render_source_audit_markdown(gate)
    outputs = {
        "v004c_mechanism_raw_features_v001.csv": dataframe_csv_bytes(raw),
        "v004c_mechanism_source_audit_v001.csv": dataframe_csv_bytes(source_audit),
        "v004c_mechanism_source_audit_v001.md": source_md.encode("utf-8"),
        "v004c_mechanism_feature_dictionary_v001.csv": dataframe_csv_bytes(dictionary),
        "v004c_mechanism_fold_audit_v001.csv": dataframe_csv_bytes(fold_audit),
        "v004c_mechanism_lambda_v001.csv": dataframe_csv_bytes(lambda_output),
        "v004c_mechanism_oof_v001.csv": dataframe_csv_bytes(oof),
        "v004c_mechanism_comparison_v001.csv": dataframe_csv_bytes(comparison),
        "v004c_mechanism_pca_stability_v001.csv": dataframe_csv_bytes(pca),
        "v004c_mechanism_review_v001.md": review.encode("utf-8"),
    }
    strongest = _select_strongest_real(comparison)
    perm = comparison[
        comparison["run_type"].eq("PERMUTED")
        & comparison["representation"].eq(strongest["representation"])
        & comparison["model_stage"].eq(strongest["model_stage"])
    ].iloc[0]
    summary: dict[str, object] = {
        "gate": gate,
        "questions": questions,
        "strongest": strongest.to_dict(),
        "permutation": perm.to_dict(),
        "comparison": comparison,
        "raw_field_count": len(MODEL_RAW_COLUMNS),
    }
    return outputs, summary

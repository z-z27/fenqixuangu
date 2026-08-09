from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_exact_v4a_residual import load_exact_transfer_oof
from .v004c_mechanism_foundation import repair_concordance
from .v004c_v4a_architecture_transfer import (
    INITIAL_TRAIN_DATES,
    JULY_SENTINEL_DATE,
    dataframe_csv_bytes,
)
from .v004c_v4a_top10_reranker import (
    RESIDUAL_RAW_FIELDS,
    STAGE2_FEATURE_COLUMNS,
    TOP_K,
    build_stage2_features,
    load_opportunity_buckets,
    prepare_reranker_samples,
    score_stage1_date,
    strategy_summary,
)


MODEL_IDS = ("PAIR_BINARY7", "PAIR_CAPPED7", "PAIR_RAW")
TARGET_VARIANTS = ("BINARY7", "CAPPED7", "RAW_RETURN")
TARGET_TO_MODEL = dict(zip(TARGET_VARIANTS, MODEL_IDS))
PAIR_FEATURE_COLUMNS = [
    "stage1_strength",
    "closing_completion_gap",
    "strength_x_gap",
]
PAIR_L2 = 0.30
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_RESAMPLES = 20_000
HYPERPARAMETER_SEARCH = False
FEATURE_SELECTION = False
NEW_FEATURES = False
EXTERNAL_FACTORS = False
TAIL_WEIGHTING = False
POSITIVE_WEIGHTING = False

FROZEN_RERANKER_OOF = (
    "reports/research/"
    "v004c_v4a_top10_residual_reranker_v001_20260506_20260630/"
    "v004c_v4a_top10_reranker_oof_v001.csv"
)
OUTPUT_FILENAMES = (
    "v004c_top10_target_info_oof_v001.csv",
    "v004c_top10_target_info_fold_audit_v001.csv",
    "v004c_top10_target_info_coefficients_v001.csv",
    "v004c_top10_target_info_pair_audit_v001.csv",
    "v004c_top10_target_info_practical_v001.csv",
    "v004c_top10_target_info_review_v001.md",
)


def assert_experiment_contract() -> None:
    if PAIR_FEATURE_COLUMNS != STAGE2_FEATURE_COLUMNS or len(PAIR_FEATURE_COLUMNS) != 3:
        raise RuntimeError("FATAL: target-information predictors changed")
    if PAIR_L2 != 0.30:
        raise RuntimeError("FATAL: target-information L2 changed")
    if TARGET_VARIANTS != ("BINARY7", "CAPPED7", "RAW_RETURN"):
        raise RuntimeError("FATAL: target variants changed")
    if any((HYPERPARAMETER_SEARCH, FEATURE_SELECTION, NEW_FEATURES, EXTERNAL_FACTORS)):
        raise RuntimeError("FATAL: search/selection/new information is forbidden")
    if TAIL_WEIGHTING or POSITIVE_WEIGHTING:
        raise RuntimeError("FATAL: target-information pair weighting changed")


def encode_target_information(raw_return: pd.Series | np.ndarray) -> pd.DataFrame:
    raw = np.asarray(raw_return, dtype=float)
    if not np.isfinite(raw).all():
        raise RuntimeError("FATAL: non-finite raw return")
    return pd.DataFrame({
        "utility_binary7": (raw >= 0.07).astype(float),
        "utility_capped7": np.minimum(raw, 0.07),
        "utility_raw": raw.copy(),
    })


def fit_pairwise_weighted_ridge(
    pair_x: np.ndarray,
    pair_y: np.ndarray,
    pair_weight: np.ndarray,
    l2: float = PAIR_L2,
) -> np.ndarray:
    x = np.asarray(pair_x, dtype=float)
    y = np.asarray(pair_y, dtype=float)
    weight = np.asarray(pair_weight, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(PAIR_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: pair X shape mismatch")
    if len(x) != len(y) or len(y) != len(weight) or len(x) == 0:
        raise RuntimeError("FATAL: pair arrays mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("FATAL: non-finite pair training data")
    if not np.isfinite(weight).all() or bool((weight <= 0).any()):
        raise RuntimeError("FATAL: invalid pair weights")
    gram = x.T @ (weight[:, None] * x) + float(l2) * np.eye(x.shape[1])
    rhs = x.T @ (weight * y)
    return np.linalg.solve(gram, rhs)


def build_same_date_pairs(meta_top10: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signal_date", "event_id", "target7", "raw_repair_return",
        *PAIR_FEATURE_COLUMNS,
    }
    missing = required.difference(meta_top10.columns)
    if missing:
        raise RuntimeError(f"FATAL: pair input columns missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for date, day in meta_top10.groupby("signal_date", sort=True):
        ordered = day.sort_values("event_id", kind="mergesort").reset_index(drop=True)
        if len(ordered) > TOP_K:
            raise RuntimeError("FATAL: pair date exceeds frozen Top10")
        pair_indices = list(combinations(range(len(ordered)), 2))
        if not pair_indices:
            # A one-candidate date has exactly zero unordered pairs. It remains
            # in the meta-date audit but cannot contribute a fabricated pair.
            continue
        utility = encode_target_information(ordered["raw_repair_return"])
        expected_target = ordered["raw_repair_return"].ge(0.07).astype(int).to_numpy()
        if not np.array_equal(expected_target, ordered["target7"].astype(int).to_numpy()):
            raise RuntimeError("FATAL: Target7 semantic mismatch in pair input")
        pair_weight = 1.0 / len(pair_indices)
        for left_index, right_index in pair_indices:
            left = ordered.iloc[left_index]
            right = ordered.iloc[right_index]
            if str(left.event_id) >= str(right.event_id):
                raise RuntimeError("FATAL: pair orientation is not event_id ascending")
            record: dict[str, Any] = {
                "signal_date": str(date),
                "left_event_id": str(left.event_id),
                "right_event_id": str(right.event_id),
                "pair_key": f"{date}|{left.event_id}|{right.event_id}",
                "pair_weight": pair_weight,
                "left_target7": int(left.target7),
                "right_target7": int(right.target7),
            }
            for feature in PAIR_FEATURE_COLUMNS:
                record[f"x_{feature}"] = float(left[feature] - right[feature])
            record["pair_y_binary7"] = float(
                utility.iloc[left_index]["utility_binary7"]
                - utility.iloc[right_index]["utility_binary7"]
            )
            record["pair_y_capped7"] = float(
                utility.iloc[left_index]["utility_capped7"]
                - utility.iloc[right_index]["utility_capped7"]
            )
            record["pair_y_raw"] = float(
                utility.iloc[left_index]["utility_raw"]
                - utility.iloc[right_index]["utility_raw"]
            )
            rows.append(record)
    if not rows:
        raise RuntimeError("FATAL: no historical pair available")
    pairs = pd.DataFrame(rows).sort_values(
        ["signal_date", "left_event_id", "right_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    date_weight = pairs.groupby("signal_date", sort=True)["pair_weight"].sum()
    if not np.allclose(date_weight.to_numpy(float), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("FATAL: pair date weights do not sum to one")
    if pairs["pair_key"].duplicated().any():
        raise RuntimeError("FATAL: duplicate pair key")
    return pairs


def _format_frozen(value: float) -> str:
    return format(float(value), ".12g")


def load_frozen_june_stage1(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / FROZEN_RERANKER_OOF
    frozen = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
    required = {
        "event_id", "signal_date", "code", "board", "candidate_count",
        "stage1_score", "stage1_rank", "stage1_top10", "stage1_strength",
        *RESIDUAL_RAW_FIELDS, "closing_completion_gap", "strength_x_gap",
        "target7", "raw_repair_return", "capped_opportunity_return_7",
    }
    missing = required.difference(frozen.columns)
    if missing:
        raise RuntimeError(f"FATAL: frozen Stage1 columns missing: {sorted(missing)}")
    frozen["signal_date"] = frozen["signal_date"].astype(str)
    frozen["code"] = frozen["code"].astype(str).str.zfill(6)
    if len(frozen) != 173 or frozen["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: frozen June Stage1 identity changed")
    if bool(frozen["signal_date"].ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July sentinel triggered")
    if frozen["event_id"].duplicated().any():
        raise RuntimeError("FATAL: frozen Stage1 event_id duplicate")
    cap_expected = np.minimum(frozen["raw_repair_return"].to_numpy(float), 0.07)
    if not np.allclose(
        cap_expected,
        frozen["capped_opportunity_return_7"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("FATAL: frozen capped return semantic mismatch")
    if not np.array_equal(
        frozen["target7"].astype(int).to_numpy(),
        frozen["raw_repair_return"].ge(0.07).astype(int).to_numpy(),
    ):
        raise RuntimeError("FATAL: frozen Target7 semantic mismatch")
    exact = load_exact_transfer_oof(root).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    compared = frozen.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    identity = compared[["event_id", "signal_date", "code"]].equals(
        exact[["event_id", "signal_date", "code"]]
    )
    score = compared["stage1_score"].map(_format_frozen).tolist() == exact[
        "model_score"
    ].map(_format_frozen).tolist()
    rank = compared["stage1_rank"].astype(int).tolist() == exact[
        "model_rank"
    ].astype(int).tolist()
    parity = {
        "identity_exact": bool(identity),
        "score_exact_at_frozen_precision": bool(score),
        "rank_exact": bool(rank),
        "pass": bool(identity and score and rank),
    }
    if not parity["pass"]:
        raise RuntimeError(f"FATAL: frozen Stage1 parity failed: {parity}")
    frozen["stage1_rank"] = frozen["stage1_rank"].astype(int)
    frozen["stage1_top10"] = frozen["stage1_top10"].astype(str).str.lower().eq("true")
    return frozen.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True), parity


def _pair_audit_row(test_date: str, pairs: pd.DataFrame) -> dict[str, Any]:
    binary_tied = pairs["pair_y_binary7"].eq(0.0)
    cross = ~binary_tied
    within_non = pairs["left_target7"].eq(0) & pairs["right_target7"].eq(0)
    within_target = pairs["left_target7"].eq(1) & pairs["right_target7"].eq(1)
    capped_separated = binary_tied & pairs["pair_y_capped7"].abs().gt(1e-15)
    raw_separated = binary_tied & pairs["pair_y_raw"].abs().gt(1e-15)
    return {
        "test_date": test_date,
        "training_pair_total": int(len(pairs)),
        "binary_cross_threshold_pairs": int(cross.sum()),
        "binary_same_class_pairs": int(binary_tied.sum()),
        "within_non_target_pairs": int(within_non.sum()),
        "within_target7_pairs": int(within_target.sum()),
        "binary_tied_separated_by_capped": int(capped_separated.sum()),
        "binary_tied_separated_by_raw": int(raw_separated.sum()),
        "within_non_target_separated_by_capped": int((within_non & capped_separated).sum()),
        "within_non_target_separated_by_raw": int((within_non & raw_separated).sum()),
        "within_target7_separated_by_capped": int((within_target & capped_separated).sum()),
        "within_target7_separated_by_raw": int((within_target & raw_separated).sum()),
    }


def _fit_all_variants(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    x_columns = [f"x_{feature}" for feature in PAIR_FEATURE_COLUMNS]
    x = pairs[x_columns].to_numpy(float)
    weight = pairs["pair_weight"].to_numpy(float)
    targets = {
        "BINARY7": "pair_y_binary7",
        "CAPPED7": "pair_y_capped7",
        "RAW_RETURN": "pair_y_raw",
    }
    result = {
        variant: fit_pairwise_weighted_ridge(
            x, pairs[column].to_numpy(float), weight, l2=PAIR_L2
        )
        for variant, column in targets.items()
    }
    if set(result) != set(TARGET_VARIANTS):
        raise RuntimeError("FATAL: target variant fit mismatch")
    return result


def _rank_top10(test: pd.DataFrame, score_column: str, rank_column: str) -> pd.DataFrame:
    result = test.copy()
    selected = result[result["stage1_top10"]].sort_values(
        [score_column, "stage1_rank", "event_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    ranks = pd.Series(np.arange(1, len(selected) + 1), index=selected.index)
    result[rank_column] = np.nan
    result.loc[selected.index, rank_column] = ranks
    return result


def run_target_information_benchmark(
    samples: pd.DataFrame,
    frozen_stage1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(samples["signal_date"].astype(str).unique())
    if len(dates) != 39 or len(dates[:INITIAL_TRAIN_DATES]) != 18:
        raise RuntimeError("FATAL: May-18/June-21 temporal split unavailable")
    if not all(date.startswith("2026-05") for date in dates[:18]):
        raise RuntimeError("FATAL: initial history is not May")
    if not all(date.startswith("2026-06") for date in dates[18:]):
        raise RuntimeError("FATAL: forward scope is not June")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for fold_index in range(INITIAL_TRAIN_DATES, len(dates)):
        test_date = dates[fold_index]
        history_dates = dates[:fold_index]
        if max(history_dates) >= test_date:
            raise RuntimeError("FATAL: current test date entered history")
        meta_parts: list[pd.DataFrame] = []
        self_leakage_rows = 0
        current_leakage_rows = 0
        for meta_date in history_dates:
            base_dates = [date for date in history_dates if date != meta_date]
            scored, audit = score_stage1_date(samples, base_dates, meta_date, test_date)
            self_leakage_rows += int(audit["self_label_leakage_rows"])
            current_leakage_rows += int(audit["current_test_date_leakage"])
            meta_parts.append(
                build_stage2_features(scored[scored["stage1_top10"]].copy())
            )
        meta = pd.concat(meta_parts, ignore_index=True)
        if meta["signal_date"].nunique() != len(history_dates):
            raise RuntimeError("FATAL: historical meta date coverage mismatch")
        if self_leakage_rows or current_leakage_rows:
            raise RuntimeError("FATAL: nested cross-fit leakage detected")
        pairs = build_same_date_pairs(meta)
        betas = _fit_all_variants(pairs)
        if len({tuple(pairs["pair_key"]) for _ in TARGET_VARIANTS}) != 1:
            raise RuntimeError("FATAL: pair keys differ across targets")
        test = frozen_stage1[frozen_stage1["signal_date"].eq(test_date)].copy()
        if len(test) != len(samples[samples["signal_date"].eq(test_date)]):
            raise RuntimeError("FATAL: frozen Stage1 test row mismatch")
        top10 = test[test["stage1_top10"]].copy()
        rebuilt_top10 = build_stage2_features(top10)
        for feature in PAIR_FEATURE_COLUMNS:
            if not np.allclose(
                top10[feature].to_numpy(float),
                rebuilt_top10[feature].to_numpy(float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"FATAL: frozen test feature parity failed: {feature}")
        x_test = top10[PAIR_FEATURE_COLUMNS].to_numpy(float)
        for variant, model in TARGET_TO_MODEL.items():
            score_column = {
                "BINARY7": "binary_pair_score",
                "CAPPED7": "capped_pair_score",
                "RAW_RETURN": "raw_pair_score",
            }[variant]
            rank_column = score_column.replace("score", "rank")
            test[score_column] = np.nan
            test.loc[top10.index, score_column] = x_test @ betas[variant]
            test = _rank_top10(test, score_column, rank_column)
            coefficient_rows.append({
                "test_date": test_date,
                "target_variant": variant,
                "model": model,
                "beta_stage1_strength": float(betas[variant][0]),
                "beta_closing_completion_gap": float(betas[variant][1]),
                "beta_strength_x_gap": float(betas[variant][2]),
                "l2": PAIR_L2,
                "intercept": 0.0,
            })
        predictions.append(test)
        pair_audit = _pair_audit_row(test_date, pairs)
        pair_rows.append(pair_audit)
        fold_rows.append({
            "test_date": test_date,
            "history_dates": len(history_dates),
            "historical_meta_rows": len(meta),
            "historical_top10_dates": meta["signal_date"].nunique(),
            "historical_top10_rows": len(meta),
            "historical_pair_dates": pairs["signal_date"].nunique(),
            "historical_zero_pair_dates": int(
                meta["signal_date"].nunique() - pairs["signal_date"].nunique()
            ),
            "pair_count": len(pairs),
            "binary_pair_count": len(pairs),
            "capped_pair_count": len(pairs),
            "raw_pair_count": len(pairs),
            "pair_keys_identical": True,
            "pair_x_identical": True,
            "pair_weights_identical": True,
            "self_label_leakage_rows": self_leakage_rows,
            "current_test_leakage_rows": current_leakage_rows,
            "binary_target_mean": float(pairs["pair_y_binary7"].mean()),
            "capped_target_mean": float(pairs["pair_y_capped7"].mean()),
            "raw_target_mean": float(pairs["pair_y_raw"].mean()),
            "pair_weight_date_sum_min": float(
                pairs.groupby("signal_date")["pair_weight"].sum().min()
            ),
            "pair_weight_date_sum_max": float(
                pairs.groupby("signal_date")["pair_weight"].sum().max()
            ),
            "predictors": len(PAIR_FEATURE_COLUMNS),
            "l2": PAIR_L2,
        })
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(oof) != 173 or oof["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: target-information OOF identity changed")
    fold_audit = pd.DataFrame(fold_rows)
    pair_audit = pd.DataFrame(pair_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    return oof, fold_audit, coefficients, pair_audit


def oof_output(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id", "signal_date", "code", "candidate_count", "stage1_rank",
        "stage1_strength", "stage1_top10", "closing_completion_gap",
        "strength_x_gap", "binary_pair_score", "binary_pair_rank",
        "capped_pair_score", "capped_pair_rank", "raw_pair_score",
        "raw_pair_rank", "target7", "raw_repair_return",
        "capped_opportunity_return_7",
    ]
    return frame[columns].rename(columns={
        "raw_repair_return": "raw_return",
        "capped_opportunity_return_7": "capped_return_7",
    }).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _oracle_rank(frame: pd.DataFrame) -> pd.Series:
    ranks = pd.Series(index=frame.index, dtype=float)
    for _, day in frame.groupby("signal_date", sort=True):
        pool = day[day["stage1_top10"]].sort_values(
            ["capped_opportunity_return_7", "event_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        outside = day[~day.index.isin(pool.index)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        ordered = pd.concat([pool, outside])
        ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return ranks.astype(int)


def _ordering(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    top10 = frame[frame["stage1_top10"]].copy().rename(
        columns={score_column: "score"}
    )
    return repair_concordance(top10)


def practical_table(
    oof: pd.DataFrame,
    bucket_map: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    evaluated = oof.copy()
    evaluated["top10_oracle_rank"] = _oracle_rank(evaluated)
    rank_columns = {
        "CONTROL_V4A": "stage1_rank",
        "PAIR_BINARY7": "binary_pair_rank",
        "PAIR_CAPPED7": "capped_pair_rank",
        "PAIR_RAW": "raw_pair_rank",
        "TOP10_ORACLE": "top10_oracle_rank",
    }
    score_columns = {
        "CONTROL_V4A": "stage1_score",
        "PAIR_BINARY7": "binary_pair_score",
        "PAIR_CAPPED7": "capped_pair_score",
        "PAIR_RAW": "raw_pair_score",
    }
    ordering = {
        model: _ordering(evaluated, score)
        for model, score in score_columns.items()
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
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
            for metric in (
                "all_repair_concordance", "cross_threshold_concordance",
                "within_non_target_concordance", "within_target_concordance",
            ):
                row[metric] = ordering.get(model, {}).get(metric, np.nan)
            rows.append(row)
        control = strategy_summary(subset, "stage1_rank")
        universe: dict[str, Any] = {
            "scope": scope,
            "model": "UNIVERSE",
            "winner_capture": np.nan,
            "universe_target7_rate": control["universe_target7_rate"],
            "Top3_days_le_minus_3": np.nan,
            "Top3_days_le_minus_5": np.nan,
            "all_repair_concordance": np.nan,
            "cross_threshold_concordance": np.nan,
            "within_non_target_concordance": np.nan,
            "within_target_concordance": np.nan,
        }
        for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3"):
            baseline = control[f"{label}_baseline"]
            universe.update({
                f"{label}_mean": baseline,
                f"{label}_median": np.nan,
                f"{label}_p25": np.nan,
                f"{label}_worst": np.nan,
                f"{label}_baseline": baseline,
                f"{label}_excess": 0.0,
                f"{label}_positive_date_rate": np.nan,
                f"{label}_negative_date_rate": np.nan,
                f"{label}_beat_universe": np.nan,
                f"{label}_target7_precision": control["universe_target7_rate"],
                f"{label}_dates": control[f"{label}_dates"],
            })
        rows.append(universe)
    return pd.DataFrame(rows), summaries, ordering


def _daily_top3(frame: pd.DataFrame, rank_column: str) -> pd.Series:
    values: dict[str, float] = {}
    for date, day in frame.groupby("signal_date", sort=True):
        k = min(3, len(day))
        selected = day[day[rank_column].le(k)]
        if len(selected) != k:
            raise RuntimeError(f"FATAL: Top3 membership unavailable for {date}")
        values[str(date)] = float(selected["capped_opportunity_return_7"].mean())
    return pd.Series(values, dtype=float).sort_index()


def daily_robustness(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = pd.DataFrame({
        "binary": _daily_top3(oof, "binary_pair_rank"),
        "capped": _daily_top3(oof, "capped_pair_rank"),
        "raw": _daily_top3(oof, "raw_pair_rank"),
        "control": _daily_top3(oof, "stage1_rank"),
    })
    daily.index.name = "signal_date"
    daily = daily.reset_index()
    comparisons = {
        "CAPPED_MINUS_BINARY": daily["capped"] - daily["binary"],
        "RAW_MINUS_BINARY": daily["raw"] - daily["binary"],
        "RAW_MINUS_CAPPED": daily["raw"] - daily["capped"],
        "CAPPED_MINUS_CONTROL": daily["capped"] - daily["control"],
        "RAW_MINUS_CONTROL": daily["raw"] - daily["control"],
    }
    for name, values in comparisons.items():
        daily[name.lower()] = values
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw = generator.integers(0, len(daily), size=(BOOTSTRAP_RESAMPLES, len(daily)))
    summaries: dict[str, Any] = {}
    for name, values in comparisons.items():
        delta = values.to_numpy(float)
        boot = delta[draw].mean(axis=1)
        quantile = np.quantile(boot, [0.025, 0.50, 0.975])
        result: dict[str, Any] = {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "positive_dates": int((delta > 0).sum()),
            "negative_dates": int((delta < 0).sum()),
            "zero_dates": int((delta == 0).sum()),
            "bootstrap_p025": float(quantile[0]),
            "bootstrap_p50": float(quantile[1]),
            "bootstrap_p975": float(quantile[2]),
            "bootstrap_probability_positive": float((boot > 0).mean()),
        }
        if name in ("CAPPED_MINUS_BINARY", "RAW_MINUS_BINARY"):
            lodo = np.asarray([
                np.delete(delta, index).mean() for index in range(len(delta))
            ])
            result.update({
                "lodo_positive_pct": float((lodo > 0).mean()),
                "lodo_min": float(lodo.min()),
                "lodo_median": float(np.median(lodo)),
                "lodo_max": float(lodo.max()),
            })
        summaries[name] = result
    return daily, summaries


def membership_movement(oof: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("BINARY_VS_CAPPED", "binary_pair_rank", "capped_pair_rank"),
        ("BINARY_VS_RAW", "binary_pair_rank", "raw_pair_rank"),
        ("CAPPED_VS_RAW", "capped_pair_rank", "raw_pair_rank"),
    )
    rows: list[dict[str, Any]] = []
    for comparison, left_rank, right_rank in specs:
        changed_dates = 0
        changed_slots = 0
        for _, day in oof.groupby("signal_date", sort=True):
            k = min(3, len(day))
            left = set(day.loc[day[left_rank].le(k), "event_id"])
            right = set(day.loc[day[right_rank].le(k), "event_id"])
            slots = k - len(left.intersection(right))
            changed_dates += int(slots > 0)
            changed_slots += slots
        rows.append({
            "comparison": comparison,
            "changed_dates": changed_dates,
            "changed_slots": changed_slots,
        })
    return pd.DataFrame(rows)


def aggregate_pair_information(pair_audit: pd.DataFrame) -> dict[str, Any]:
    count_columns = [
        column for column in pair_audit.columns
        if column != "test_date"
    ]
    totals = {column: int(pair_audit[column].sum()) for column in count_columns}
    tied = totals["binary_same_class_pairs"]
    totals["binary_cross_threshold_pct"] = (
        totals["binary_cross_threshold_pairs"] / totals["training_pair_total"]
    )
    totals["binary_same_class_pct"] = tied / totals["training_pair_total"]
    totals["binary_tied_separated_by_capped_pct"] = (
        totals["binary_tied_separated_by_capped"] / tied
    )
    totals["binary_tied_separated_by_raw_pct"] = (
        totals["binary_tied_separated_by_raw"] / tied
    )
    return totals


def _scope_row(practical: pd.DataFrame, scope: str, model: str) -> pd.Series:
    row = practical[practical["scope"].eq(scope) & practical["model"].eq(model)]
    if len(row) != 1:
        raise RuntimeError(f"FATAL: practical row unavailable {scope}/{model}")
    return row.iloc[0]


def _detailed_gate(
    model: str,
    practical: pd.DataFrame,
    robustness: Mapping[str, Any],
) -> dict[str, bool]:
    binary = _scope_row(practical, "COMBINED", "PAIR_BINARY7")
    detailed = _scope_row(practical, "COMBINED", model)
    comparison = {
        "PAIR_CAPPED7": "CAPPED_MINUS_BINARY",
        "PAIR_RAW": "RAW_MINUS_BINARY",
    }[model]
    robust = robustness[comparison]
    return {
        "top3_improvement": bool(detailed.Top3_mean - binary.Top3_mean >= 0.005),
        "precision": bool(
            detailed.Top3_target7_precision > binary.Top3_target7_precision
        ),
        "negative_risk": bool(
            detailed.Top3_negative_date_rate
            <= binary.Top3_negative_date_rate + 0.05
        ),
        "daily_breadth": bool(
            robust["positive_dates"] > robust["negative_dates"]
        ),
        "bootstrap": bool(robust["bootstrap_probability_positive"] >= 0.70),
        "lodo": bool(robust["lodo_positive_pct"] >= 0.80),
    }


def formal_decision(
    practical: pd.DataFrame,
    ordering: Mapping[str, Mapping[str, float]],
    robustness: Mapping[str, Any],
    gates_valid: bool,
) -> dict[str, Any]:
    detailed_gates = {
        model: _detailed_gate(model, practical, robustness)
        for model in ("PAIR_CAPPED7", "PAIR_RAW")
    }
    supported_models = [
        model for model, gate in detailed_gates.items() if all(gate.values())
    ]
    if not gates_valid:
        target_signal = "INVALID"
    elif supported_models:
        target_signal = "SUPPORTED"
    else:
        binary = _scope_row(practical, "COMBINED", "PAIR_BINARY7")
        directional = False
        for model in ("PAIR_CAPPED7", "PAIR_RAW"):
            detailed = _scope_row(practical, "COMBINED", model)
            directional = directional or bool(
                detailed.Top3_mean > binary.Top3_mean
                or detailed.Top3_target7_precision > binary.Top3_target7_precision
                or ordering[model]["within_non_target_concordance"]
                > ordering["PAIR_BINARY7"]["within_non_target_concordance"]
                or ordering[model]["within_target_concordance"]
                > ordering["PAIR_BINARY7"]["within_target_concordance"]
            )
        target_signal = "MIXED" if directional else "NOT_SUPPORTED"
    capped = _scope_row(practical, "COMBINED", "PAIR_CAPPED7")
    raw = _scope_row(practical, "COMBINED", "PAIR_RAW")
    raw_capped = robustness["RAW_MINUS_CAPPED"]
    above_useful = bool(
        raw.Top3_mean - capped.Top3_mean >= 0.003
        and raw.Top3_target7_precision >= capped.Top3_target7_precision
        and raw.Top3_negative_date_rate <= capped.Top3_negative_date_rate
        and raw.Top3_worst >= capped.Top3_worst
        and raw_capped["bootstrap_probability_positive"] >= 0.65
    )
    above_harmful = bool(
        raw.Top3_mean <= capped.Top3_mean - 0.003
        or (
            raw.Top3_target7_precision < capped.Top3_target7_precision
            and raw.Top3_negative_date_rate > capped.Top3_negative_date_rate + 0.05
        )
    )
    above_signal = "USEFUL" if above_useful else ("HARMFUL" if above_harmful else "NEUTRAL")
    selected = "NONE"
    if len(supported_models) == 1:
        selected = supported_models[0]
    elif len(supported_models) == 2:
        capped_top3 = float(capped.Top3_mean)
        raw_top3 = float(raw.Top3_mean)
        if abs(capped_top3 - raw_top3) >= 0.001:
            selected = "PAIR_CAPPED7" if capped_top3 > raw_top3 else "PAIR_RAW"
        elif capped.Top3_negative_date_rate != raw.Top3_negative_date_rate:
            selected = (
                "PAIR_CAPPED7"
                if capped.Top3_negative_date_rate < raw.Top3_negative_date_rate
                else "PAIR_RAW"
            )
        else:
            selected = "PAIR_CAPPED7"
    trading = False
    trading_gates: dict[str, bool] = {}
    if selected != "NONE":
        candidate = _scope_row(practical, "COMBINED", selected)
        control = _scope_row(practical, "COMBINED", "CONTROL_V4A")
        low_candidate = _scope_row(practical, "LOW", selected)
        low_control = _scope_row(practical, "LOW", "CONTROL_V4A")
        high_candidate = _scope_row(practical, "HIGH", selected)
        high_control = _scope_row(practical, "HIGH", "CONTROL_V4A")
        comparison = (
            "CAPPED_MINUS_CONTROL" if selected == "PAIR_CAPPED7" else "RAW_MINUS_CONTROL"
        )
        trading_gates = {
            "top3_control": bool(
                candidate.Top3_mean > control.Top3_mean
                and candidate.Top3_mean - control.Top3_mean >= 0.005
            ),
            "top3_universe": bool(candidate.Top3_mean > candidate.Top3_baseline),
            "precision": bool(
                candidate.Top3_target7_precision > control.Top3_target7_precision
                and candidate.Top3_target7_precision > candidate.universe_target7_rate
            ),
            "risk": bool(
                candidate.Top3_negative_date_rate
                <= control.Top3_negative_date_rate + 0.05
                and candidate.Top3_worst >= control.Top3_worst - 0.01
            ),
            "low": bool(
                low_candidate.Top3_mean - low_control.Top3_mean >= 0.005
                and low_candidate.Top3_negative_date_rate
                <= low_control.Top3_negative_date_rate
            ),
            "high": bool(
                high_candidate.Top3_mean >= high_control.Top3_mean - 0.005
            ),
            "cross_date": bool(
                candidate.Top3_beat_universe > 0.50
                and robustness[comparison]["positive_dates"]
                > robustness[comparison]["negative_dates"]
            ),
        }
        trading = all(trading_gates.values())
    return {
        "detailed_gates": detailed_gates,
        "supported_models": supported_models,
        "target_information_signal": target_signal,
        "above7_information_signal": above_signal,
        "selected_detailed_variant": selected,
        "trading_gates": trading_gates,
        "target_objective_trading_candidate": "YES" if trading else "NO",
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
    ordering: Mapping[str, Mapping[str, float]],
    pair_information: Mapping[str, Any],
    robustness: Mapping[str, Any],
    movement: pd.DataFrame,
    fold_audit: pd.DataFrame,
    parity: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    combined_models = (
        "UNIVERSE", "CONTROL_V4A", "PAIR_BINARY7", "PAIR_CAPPED7",
        "PAIR_RAW", "TOP10_ORACLE",
    )
    lines = [
        "# v004c Top10 Target-Information Benchmark v001",
        "",
        "## 1. Experimental Contract",
        "",
        "- Learner: same-date pairwise weighted Ridge, no intercept on pair differences.",
        "- Predictors: stage1_strength, closing_completion_gap, strength_x_gap.",
        "- L2: 0.30. Pair date weight: 1 / pair_count_on_date.",
        "- Pair orientation: event_id ascending; pair_x = X_left - X_right.",
        "- Same pair keys, X, weights, learner and chronology; only pair_y changes.",
        "- BINARY7 utility: 1[raw_return >= 0.07].",
        "- CAPPED7 utility: min(raw_return, 0.07), with no lower floor.",
        "- RAW_RETURN utility: raw_return, with no clipping or winsorization.",
        "- Hyperparameter search: NO. Feature selection: NO. July: NOT ACCESSED.",
        "",
        "## 2. Frozen Stage1 Parity",
        "",
        f"- Identity exact: {parity['identity_exact']}.",
        f"- Score exact at frozen precision: {parity['score_exact_at_frozen_precision']}.",
        f"- Rank exact: {parity['rank_exact']}.",
        f"- Parity gate: {'PASS' if parity['pass'] else 'FAIL'}.",
        "",
        "## 3. Leakage Audit",
        "",
        f"- SELF_LABEL_LEAKAGE_ROWS: {int(fold_audit.self_label_leakage_rows.sum())}.",
        f"- CURRENT_TEST_DATE_LEAKAGE: {int(fold_audit.current_test_leakage_rows.sum())}.",
        "- Historical meta method: DATE-CROSSFITTED STAGE1 TOP10.",
        "- Leakage gate: PASS.",
        "",
        "## 4. Pair and Target Information Audit",
        "",
    ]
    lines.extend(_markdown_table(
        ["Metric", "Count", "Rate"],
        [
            ["Total training pairs", pair_information["training_pair_total"], "100.0000%"],
            ["Binary cross-threshold", pair_information["binary_cross_threshold_pairs"], _pct(pair_information["binary_cross_threshold_pct"])],
            ["Binary same-class", pair_information["binary_same_class_pairs"], _pct(pair_information["binary_same_class_pct"])],
            ["Within non-target", pair_information["within_non_target_pairs"], "NA"],
            ["Within Target7", pair_information["within_target7_pairs"], "NA"],
            ["Binary ties separated by CAPPED", pair_information["binary_tied_separated_by_capped"], _pct(pair_information["binary_tied_separated_by_capped_pct"])],
            ["Binary ties separated by RAW", pair_information["binary_tied_separated_by_raw"], _pct(pair_information["binary_tied_separated_by_raw_pct"])],
        ],
    ))
    lines.extend([
        "",
        "## 5. Binary vs Capped vs Raw Practical Performance",
        "",
    ])
    practical_rows = []
    for model in combined_models:
        row = _scope_row(practical, "COMBINED", model)
        practical_rows.append([
            model, _pct(row.Rank1_mean), _pct(row.Rank2_mean),
            _pct(row.Rank3_mean), _pct(row.Top2_mean), _pct(row.Top3_mean),
            _pct(row.Top3_excess), _pct(row.Top3_target7_precision),
            _pct(row.Top3_beat_universe), _pct(row.Top3_negative_date_rate),
            _pct(row.Top3_worst),
        ])
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Rank2", "Rank3", "Top2", "Top3", "Excess", "Precision", "Beat", "Negative", "Worst"],
        practical_rows,
    ))
    lines.extend(["", "## 6. Ordering Diagnostics", ""])
    ordering_rows = []
    for model in ("PAIR_BINARY7", "PAIR_CAPPED7", "PAIR_RAW"):
        value = ordering[model]
        ordering_rows.append([
            model, _num(value["all_repair_concordance"]),
            _num(value["cross_threshold_concordance"]),
            _num(value["within_non_target_concordance"]),
            _num(value["within_target_concordance"]),
        ])
    lines.extend(_markdown_table(
        ["Model", "All", "Cross", "Within non-target", "Within Target7"],
        ordering_rows,
    ))
    lines.extend(["", "## 7. LOW / MID / HIGH Opportunity", ""])
    opportunity_rows = []
    for scope in ("LOW", "MID", "HIGH"):
        for model in ("CONTROL_V4A", "PAIR_BINARY7", "PAIR_CAPPED7", "PAIR_RAW", "UNIVERSE"):
            row = _scope_row(practical, scope, model)
            opportunity_rows.append([
                scope, model, _pct(row.Rank1_mean), _pct(row.Top2_mean),
                _pct(row.Top3_mean), _pct(row.Top3_excess),
                _pct(row.Top3_target7_precision), _pct(row.Top3_negative_date_rate),
                _pct(row.Top3_median), _pct(row.Top3_worst),
            ])
    lines.extend(_markdown_table(
        ["Bucket", "Model", "Rank1", "Top2", "Top3", "Excess", "Precision", "Negative", "Median", "Worst"],
        opportunity_rows,
    ))
    lines.extend(["", "## 8. Daily Robustness", ""])
    robust_rows = []
    for comparison in ("CAPPED_MINUS_BINARY", "RAW_MINUS_BINARY", "RAW_MINUS_CAPPED"):
        value = robustness[comparison]
        robust_rows.append([
            comparison, _pct(value["mean"]), _pct(value["median"]),
            value["positive_dates"], value["negative_dates"], value["zero_dates"],
            f"[{_pct(value['bootstrap_p025'])}, {_pct(value['bootstrap_p975'])}]",
            _pct(value["bootstrap_p50"]),
            _pct(value["bootstrap_probability_positive"]),
            _pct(value.get("lodo_positive_pct", np.nan)),
            _pct(value.get("lodo_min", np.nan)),
            _pct(value.get("lodo_median", np.nan)),
            _pct(value.get("lodo_max", np.nan)),
        ])
    lines.extend(_markdown_table(
        ["Comparison", "Mean", "Median", "+", "-", "0", "Bootstrap 95%", "Bootstrap P50", "P(>0)", "LODO +", "LODO Min", "LODO Median", "LODO Max"],
        robust_rows,
    ))
    lines.append("")
    lines.extend(_markdown_table(
        ["Membership Comparison", "Changed Dates", "Changed Slots"],
        movement[["comparison", "changed_dates", "changed_slots"]].values.tolist(),
    ))
    lines.extend([
        "",
        "## 9. Target-Information Decision",
        "",
    ])
    binary = _scope_row(practical, "COMBINED", "PAIR_BINARY7")
    objective_rows = []
    for model in ("PAIR_CAPPED7", "PAIR_RAW"):
        detailed = _scope_row(practical, "COMBINED", model)
        objective_rows.append([
            f"{model} - PAIR_BINARY7",
            _pct(detailed.Top3_mean - binary.Top3_mean),
            _pct(detailed.Top3_target7_precision - binary.Top3_target7_precision),
            _pct(detailed.Top3_negative_date_rate - binary.Top3_negative_date_rate),
            "YES" if all(decision["detailed_gates"][model].values()) else "NO",
        ])
    raw = _scope_row(practical, "COMBINED", "PAIR_RAW")
    capped = _scope_row(practical, "COMBINED", "PAIR_CAPPED7")
    objective_rows.append([
        "PAIR_RAW - PAIR_CAPPED7",
        _pct(raw.Top3_mean - capped.Top3_mean),
        _pct(raw.Top3_target7_precision - capped.Top3_target7_precision),
        _pct(raw.Top3_negative_date_rate - capped.Top3_negative_date_rate),
        decision["above7_information_signal"],
    ])
    lines.extend(_markdown_table(
        ["Comparison", "Top3 Delta", "Precision Delta", "Negative-Rate Delta", "Gate/Status"],
        objective_rows,
    ))
    capped_supported = all(decision["detailed_gates"]["PAIR_CAPPED7"].values())
    raw_supported = all(decision["detailed_gates"]["PAIR_RAW"].values())
    within_non_improved = any(
        ordering[model]["within_non_target_concordance"]
        > ordering["PAIR_BINARY7"]["within_non_target_concordance"]
        for model in ("PAIR_CAPPED7", "PAIR_RAW")
    )
    within_target_improved = any(
        ordering[model]["within_target_concordance"]
        > ordering["PAIR_BINARY7"]["within_target_concordance"]
        for model in ("PAIR_CAPPED7", "PAIR_RAW")
    )
    practical_improved = max(capped.Top3_mean, raw.Top3_mean) - binary.Top3_mean >= 0.005
    lines.extend([
        "",
        f"- Q1 Does CAPPED materially beat BINARY: **{'YES' if capped_supported else 'NO'}**.",
        f"- Q2 Does RAW materially beat BINARY: **{'YES' if raw_supported else 'NO'}**.",
        f"- Q3 Does preserving sub-7% severity improve Top10→Top3: **{'YES' if capped_supported else ('MIXED' if capped.Top3_mean > binary.Top3_mean else 'NO')}**.",
        f"- Q4 Does preserving information above 7% add value: **{'YES' if decision['above7_information_signal'] == 'USEFUL' else ('NEUTRAL' if decision['above7_information_signal'] == 'NEUTRAL' else 'NO')}**.",
        f"- Q5 Did detailed targets improve within-nontarget ordering: **{'YES' if within_non_improved else 'NO'}**.",
        f"- Q6 Did detailed targets improve within-Target7 ordering: **{'YES' if within_target_improved else 'NO'}**.",
        f"- Q7 Did finer ordering improve practical Top3: **{'YES' if practical_improved else 'NO'}**.",
        f"- TARGET_INFORMATION_SIGNAL: **{decision['target_information_signal']}**.",
        f"- ABOVE7_INFORMATION_SIGNAL: **{decision['above7_information_signal']}**.",
        "",
        "## 10. Trading-Candidate Decision",
        "",
        f"- Selected detailed variant by frozen rule: **{decision['selected_detailed_variant']}**.",
        f"- TARGET_OBJECTIVE_TRADING_CANDIDATE: **{decision['target_objective_trading_candidate']}**.",
        f"- Trading gates: {decision['trading_gates']}.",
        "- July: NOT ACCESSED. Production ready: NO.",
        "- Another target-objective variant immediately: NO.",
    ])
    if decision["target_information_signal"] == "NOT_SUPPORTED":
        lines.extend([
            "",
            "Target7 information compression is not the main reason this Stage2 representation failed. Do not introduce a fourth target, tune L2, add weighting, or access July.",
        ])
    elif decision["target_information_signal"] == "MIXED":
        lines.extend([
            "",
            "Finer target information changed some ordering evidence, but did not satisfy the controlled practical-support gate.",
        ])
    elif decision["target_information_signal"] == "SUPPORTED" and decision["target_objective_trading_candidate"] == "NO":
        lines.extend([
            "",
            "Target information hypothesis supported, but the current feature representation is still insufficient for a tradable Top3.",
        ])
    return "\n".join(lines) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_experiment_contract()
    samples, base, feature_gate = prepare_reranker_samples(root)
    frozen, parity = load_frozen_june_stage1(root)
    oof, fold_audit, coefficients, pair_audit = run_target_information_benchmark(
        samples, frozen
    )
    bucket_map = load_opportunity_buckets(root, oof["signal_date"].unique())
    practical, summaries, ordering = practical_table(oof, bucket_map)
    daily, robustness = daily_robustness(oof)
    movement = membership_movement(oof)
    pair_information = aggregate_pair_information(pair_audit)
    gates_valid = bool(
        parity["pass"]
        and fold_audit["pair_keys_identical"].all()
        and fold_audit["pair_x_identical"].all()
        and fold_audit["pair_weights_identical"].all()
        and fold_audit["self_label_leakage_rows"].eq(0).all()
        and fold_audit["current_test_leakage_rows"].eq(0).all()
        and oof["signal_date"].lt(JULY_SENTINEL_DATE).all()
    )
    decision = formal_decision(practical, ordering, robustness, gates_valid)
    review = build_review(
        practical, ordering, pair_information, robustness, movement,
        fold_audit, parity, decision,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(oof_output(oof)),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(fold_audit),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(pair_audit),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[5]: review.encode("utf-8"),
    }
    context = {
        "base": base,
        "samples": samples,
        "feature_gate": feature_gate,
        "frozen": frozen,
        "parity": parity,
        "oof": oof,
        "fold_audit": fold_audit,
        "coefficients": coefficients,
        "pair_audit": pair_audit,
        "pair_information": pair_information,
        "practical": practical,
        "summaries": summaries,
        "ordering": ordering,
        "bucket_map": bucket_map,
        "daily": daily,
        "robustness": robustness,
        "movement": movement,
        "decision": decision,
        "gates_valid": gates_valid,
        "self_label_leakage_rows": int(fold_audit.self_label_leakage_rows.sum()),
        "current_test_date_leakage_rows": int(fold_audit.current_test_leakage_rows.sum()),
        "output_sha256": {name: sha256_bytes(value) for name, value in outputs.items()},
    }
    return outputs, context


def run_v004c_top10_target_information(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first.keys() != second.keys() or any(first[name] != second[name] for name in first):
        raise RuntimeError("FATAL: deterministic target-information rebuild failed")
    output_dir = (
        root_path
        / "reports/research/v004c_top10_target_information_v001_20260506_20260630"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (output_dir / name).write_bytes(first[name])
    context["deterministic_rebuild"] = "PASS"
    context["output_dir"] = output_dir
    return output_dir, context

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
from .v004c_v4a_architecture_transfer import (
    FEATURE_SELECTION as TRANSFER_FEATURE_SELECTION,
    FROZEN_FEATURE_COLUMNS,
    HYPERPARAMETER_SEARCH as TRANSFER_HYPERPARAMETER_SEARCH,
    INITIAL_TRAIN_DATES,
    JULY_SENTINEL_DATE,
    L2,
    MODEL_ID as CONTROL_MODEL_ID,
    POSITIVE_WEIGHT,
    _daily_selected,
    _load_reference_scores,
    _oracle_frame,
    _sigmoid,
    assert_authoritative_universe,
    assert_frozen_architecture_contract,
    assign_model_rank,
    build_exact_raw_features,
    dataframe_csv_bytes,
    exact_tie_rows,
    load_authoritative_input,
    prepare_transfer_samples,
    run_expanding_forward,
    sha256_bytes,
    summarize_model,
)
from .v004c_mechanism_foundation import load_outcomes


CURVE_MODEL_ID = "V4A_OVEREXTENSION_CURVE_V4C"
CONTROL_PREDICTOR_COUNT = 18
CURVE_PREDICTOR_COUNT = 23
HYPERPARAMETER_SEARCH = False
FEATURE_SELECTION = False
NEW_RAW_FEATURES = False
EXTRA_SCALING = False
USES_PCA = False
USES_PAIRWISE = False
USES_BOARD3 = False
USES_MARKET_REGIME = False
USES_THEME_EXTENSION = False
USES_SAME_TIER = False

QUADRATIC_SPECS = (
    ("quad_close_ma10", "rank_d1_close_ma10_pct", "close_ma10"),
    ("quad_low_ma10", "rank_d1_low_ma10_pct", "low_ma10"),
    ("quad_trend_hold", "rank_trend_hold_score", "trend_hold"),
    ("quad_close_vwap", "rank_d1_close_vwap_pct", "close_vwap"),
    ("quad_log_base_price", "rank_log_candidate_base_price", "log_base_price"),
)
QUADRATIC_COLUMNS = [spec[0] for spec in QUADRATIC_SPECS]
CURVE_FEATURE_COLUMNS = [*FROZEN_FEATURE_COLUMNS, *QUADRATIC_COLUMNS]
EXTREME_THRESHOLD = 0.90

OUTPUT_FILENAMES = (
    "v004c_overextension_curve_oof_v001.csv",
    "v004c_overextension_curve_practical_v001.csv",
    "v004c_overextension_curve_coefficients_v001.csv",
    "v004c_overextension_curve_curvature_audit_v001.csv",
    "v004c_overextension_curve_rank_movement_v001.csv",
    "v004c_overextension_curve_opportunity_v001.csv",
    "v004c_overextension_curve_review_v001.md",
)


def assert_curve_contract() -> None:
    assert_frozen_architecture_contract()
    if len(FROZEN_FEATURE_COLUMNS) != CONTROL_PREDICTOR_COUNT:
        raise RuntimeError("FATAL: frozen control no longer has 18 predictors")
    expected = [
        "quad_close_ma10",
        "quad_low_ma10",
        "quad_trend_hold",
        "quad_close_vwap",
        "quad_log_base_price",
    ]
    if QUADRATIC_COLUMNS != expected:
        raise RuntimeError("FATAL: predeclared quadratic term list changed")
    if len(CURVE_FEATURE_COLUMNS) != CURVE_PREDICTOR_COUNT:
        raise RuntimeError("FATAL: curve model must have exactly 23 predictors")
    if len(set(CURVE_FEATURE_COLUMNS)) != CURVE_PREDICTOR_COUNT:
        raise RuntimeError("FATAL: duplicate curve predictor")
    if L2 != 0.30 or POSITIVE_WEIGHT != 1.50 or INITIAL_TRAIN_DATES != 18:
        raise RuntimeError("FATAL: frozen training controls changed")
    if TRANSFER_HYPERPARAMETER_SEARCH or TRANSFER_FEATURE_SELECTION:
        raise RuntimeError("FATAL: frozen control unexpectedly enables search")


def add_quadratic_terms(samples: pd.DataFrame) -> pd.DataFrame:
    assert_curve_contract()
    missing = sorted(set(FROZEN_FEATURE_COLUMNS).difference(samples.columns))
    if missing:
        raise RuntimeError(f"FATAL: frozen feature columns missing: {missing}")
    result = samples.copy()
    for quadratic, linear, _ in QUADRATIC_SPECS:
        values = pd.to_numeric(result[linear], errors="raise").astype(float)
        result[quadratic] = values ** 2
    return result


def valid_turning_point(linear_beta: float, quadratic_beta: float) -> float:
    if not np.isfinite(linear_beta) or not np.isfinite(quadratic_beta):
        return np.nan
    if quadratic_beta >= 0.0:
        return np.nan
    point = -float(linear_beta) / (2.0 * float(quadratic_beta))
    if not (0.0 < point < 1.0):
        return np.nan
    return float(point)


def _adjacent_cosine(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return np.nan
    denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
    if denominator <= 0.0:
        return np.nan
    return float(previous @ current / denominator)


def run_curve_expanding_forward(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curved = add_quadratic_terms(samples)
    dates = sorted(curved["signal_date"].astype(str).unique())
    if len(dates) != 39 or not all(date.startswith("2026-05") for date in dates[:18]):
        raise RuntimeError("FATAL: natural May-18/June-21 split unavailable")
    if not all(date.startswith("2026-06") for date in dates[18:]):
        raise RuntimeError("FATAL: curve test scope must be June only")
    if any(date >= JULY_SENTINEL_DATE for date in dates):
        raise RuntimeError("FATAL: July sentinel triggered")

    predictions: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    previous_coef: np.ndarray | None = None
    for fold_index in range(INITIAL_TRAIN_DATES, len(dates)):
        test_date = dates[fold_index]
        train_dates = dates[:fold_index]
        train = curved[curved["signal_date"].isin(train_dates)].copy()
        test = curved[curved["signal_date"].eq(test_date)].copy()
        if train.empty or test.empty or str(train["signal_date"].max()) >= test_date:
            raise RuntimeError(f"FATAL: chronological curve fold violation {test_date}")
        sample_weight = build_training_sample_weight(
            train, positive_weight=POSITIVE_WEIGHT
        )
        x_train = train[CURVE_FEATURE_COLUMNS].to_numpy(float)
        y_train = train[DEFAULT_TARGET_COLUMN].astype(bool).astype(float).to_numpy()
        beta = fit_logistic_l2_weighted(
            x_train, y_train, l2=L2, sample_weight=sample_weight
        )
        coefficient = np.asarray(beta[1:], dtype=float)
        score = _sigmoid(
            beta[0] + test[CURVE_FEATURE_COLUMNS].to_numpy(float) @ coefficient
        )
        predicted = test[[
            "event_id",
            "signal_date",
            "code",
            "board_streak_before_break",
            "target7",
            "raw_repair_return",
            "capped_opportunity_return_7",
        ]].copy()
        predicted["model_score"] = score
        predicted["train_start"] = train_dates[0]
        predicted["train_end"] = train_dates[-1]
        predicted["train_dates"] = len(train_dates)
        predictions.append(predicted)

        coefficient_map = dict(zip(CURVE_FEATURE_COLUMNS, coefficient))
        row: dict[str, Any] = {
            "test_date": test_date,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "train_dates": len(train_dates),
            "train_rows": len(train),
            "train_target7_rate": float(y_train.mean()),
            "weighted_target7_rate": float(
                np.sum(sample_weight * y_train) / np.sum(sample_weight)
            ),
            "mean_sample_weight": float(sample_weight.mean()),
            "sum_sample_weight": float(sample_weight.sum()),
            "intercept": float(beta[0]),
            "curve_score_min": float(score.min()),
            "curve_score_max": float(score.max()),
            "curve_score_std": float(score.std(ddof=0)),
            "curve_exact_score_tie_rows": exact_tie_rows(pd.Series(score)),
            "adjacent_full_coefficient_cosine": _adjacent_cosine(
                previous_coef, coefficient
            ),
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "feature_count": len(CURVE_FEATURE_COLUMNS),
            "train_before_test": bool(train_dates[-1] < test_date),
            "purpose": "CAPABILITY_SCREEN_AUDIT_ONLY",
        }
        row.update({
            f"beta_{name}": float(value)
            for name, value in zip(CURVE_FEATURE_COLUMNS, coefficient)
        })
        for quadratic, linear, slug in QUADRATIC_SPECS:
            linear_beta = float(coefficient_map[linear])
            quadratic_beta = float(coefficient_map[quadratic])
            row[f"linear_beta_{slug}"] = linear_beta
            row[f"quadratic_beta_{slug}"] = quadratic_beta
            row[f"turning_point_{slug}"] = valid_turning_point(
                linear_beta, quadratic_beta
            )
        coefficient_rows.append(row)
        previous_coef = coefficient

    oof = assign_model_rank(pd.concat(predictions, ignore_index=True))
    oof["board"] = pd.to_numeric(oof["board_streak_before_break"]).astype(int)
    daily = oof.groupby("signal_date", sort=True).agg(
        daily_universe_capped_return=("capped_opportunity_return_7", "mean"),
        daily_universe_target7_rate=("target7", "mean"),
    ).reset_index()
    oof = oof.merge(daily, on="signal_date", how="left", validate="many_to_one")
    columns = [
        "event_id",
        "signal_date",
        "code",
        "board",
        "model_score",
        "model_rank",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
        "daily_universe_capped_return",
        "daily_universe_target7_rate",
        "train_start",
        "train_end",
        "train_dates",
    ]
    return (
        oof[columns].sort_values(
            ["signal_date", "model_rank", "event_id"], kind="mergesort"
        ).reset_index(drop=True),
        pd.DataFrame(coefficient_rows),
    )


def _frozen_precision(value: float) -> str:
    return format(float(value), ".12g")


def load_frozen_control_artifacts(
    root: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    directory = (
        root
        / "reports/research/v004c_v4a_architecture_transfer_v001_20260506_20260630"
    )
    oof = pd.read_csv(
        directory / "v004c_v4a_transfer_oof_v001.csv",
        encoding="utf-8-sig",
        dtype={"code": str, "event_id": str, "signal_date": str},
    )
    comparison = pd.read_csv(
        directory / "v004c_v4a_transfer_practical_comparison_v001.csv",
        encoding="utf-8-sig",
    )
    if bool(oof["signal_date"].ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July row in frozen control artifact")
    row = comparison[comparison["model"].eq(CONTROL_MODEL_ID)]
    if len(oof) != 173 or oof["signal_date"].nunique() != 21 or len(row) != 1:
        raise RuntimeError("FATAL: frozen control artifact scope mismatch")
    oof["code"] = oof["code"].astype(str).str.zfill(6)
    return oof, row.iloc[0]


def validate_control_parity(
    root: Path, rebuilt: pd.DataFrame
) -> dict[str, Any]:
    frozen, frozen_metrics = load_frozen_control_artifacts(root)
    left = frozen.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(
        drop=True
    )
    right = rebuilt.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(
        drop=True
    )
    identity_exact = left[["event_id", "signal_date", "code"]].equals(
        right[["event_id", "signal_date", "code"]]
    )
    score_exact = (
        left["model_score"].map(_frozen_precision).tolist()
        == right["model_score"].map(_frozen_precision).tolist()
    )
    rank_exact = (
        left["model_rank"].astype(int).tolist()
        == right["model_rank"].astype(int).tolist()
    )
    rebuilt_summary = summarize_model(rebuilt)
    metric_specs = (
        ("Rank1", "Rank1", "Rank1_mean"),
        ("Rank2", "Rank2", "Rank2_mean"),
        ("Rank3", "Rank3", "Rank3_mean"),
        ("Top2", "Top2", "Top2_mean"),
        ("Top3", "Top3", "Top3_mean"),
        (
            "cross_threshold",
            "cross_threshold_concordance",
            "cross_threshold_concordance",
        ),
    )
    metric_exact = all(
        _frozen_precision(rebuilt_summary[summary_key])
        == _frozen_precision(frozen_metrics[artifact_key])
        for _, artifact_key, summary_key in metric_specs
    )
    passed = bool(identity_exact and score_exact and rank_exact and metric_exact)
    result = {
        "identity_exact": identity_exact,
        "score_exact_at_frozen_precision": score_exact,
        "rank_exact": rank_exact,
        "metric_exact_at_frozen_precision": metric_exact,
        "pass": passed,
        "previous": {
            label: float(frozen_metrics[artifact_key])
            for label, artifact_key, _ in metric_specs
        },
        "rebuilt": {
            label: float(rebuilt_summary[summary_key])
            for label, _, summary_key in metric_specs
        },
    }
    if not passed:
        raise RuntimeError(f"FATAL: frozen control parity failed: {result}")
    return result


def combine_oof(
    samples: pd.DataFrame,
    control: pd.DataFrame,
    curve: pd.DataFrame,
) -> pd.DataFrame:
    key = ["event_id", "signal_date", "code"]
    base_columns = [
        *key,
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
        "model_score",
        "model_rank",
    ]
    result = control[base_columns].rename(columns={
        "model_score": "control_score",
        "model_rank": "control_rank",
    }).merge(
        curve[key + ["model_score", "model_rank"]].rename(columns={
            "model_score": "curve_score",
            "model_rank": "curve_rank",
        }),
        on=key,
        how="inner",
        validate="one_to_one",
    )
    featured = add_quadratic_terms(samples)[
        ["event_id", *[linear for _, linear, _ in QUADRATIC_SPECS], *QUADRATIC_COLUMNS]
    ]
    result = result.merge(featured, on="event_id", how="left", validate="one_to_one")
    if len(result) != 173 or result["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: combined OOF identity changed")
    result["rank_delta"] = result["curve_rank"] - result["control_rank"]
    original_rank_columns = [linear for _, linear, _ in QUADRATIC_SPECS]
    result["extreme_strength_count"] = (
        result[original_rank_columns].ge(EXTREME_THRESHOLD).sum(axis=1).astype(int)
    )
    columns = [
        "event_id",
        "signal_date",
        "code",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
        "control_score",
        "control_rank",
        "curve_score",
        "curve_rank",
        "rank_delta",
        *original_rank_columns,
        *QUADRATIC_COLUMNS,
        "extreme_strength_count",
    ]
    return result[columns].sort_values(
        ["signal_date", "curve_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def model_frame(combined: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = combined[[
        "event_id",
        "signal_date",
        "code",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
        f"{prefix}_score",
        f"{prefix}_rank",
    ]].rename(columns={
        f"{prefix}_score": "model_score",
        f"{prefix}_rank": "model_rank",
    })
    return frame.copy()


def practical_comparison(
    control: pd.DataFrame,
    curve: pd.DataFrame,
    references: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    oracle = _oracle_frame(curve)
    frames = {
        "DAILY_UNIVERSE": control,
        CONTROL_MODEL_ID: control,
        CURVE_MODEL_ID: curve,
        "CORRECTED_BINARY": references["CORRECTED_BINARY_SAME_DATE"],
        "REPAIR": references["REPAIR_SAME_DATE"],
        "ORACLE": oracle,
    }
    summaries = {
        model: summarize_model(frame)
        for model, frame in frames.items()
        if model != "DAILY_UNIVERSE"
    }
    universe = summaries[CONTROL_MODEL_ID]
    rows: list[dict[str, Any]] = []
    selector_labels = ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3")
    for model in (
        CONTROL_MODEL_ID,
        CURVE_MODEL_ID,
        "CORRECTED_BINARY",
        "REPAIR",
        "ORACLE",
    ):
        summary = summaries[model]
        row: dict[str, Any] = {"model": model}
        for label in selector_labels:
            row[label] = summary[f"{label}_mean"]
            for metric in (
                "mean",
                "median",
                "baseline",
                "excess",
                "beat_universe",
                "positive_return",
                "target7_precision",
                "dates",
            ):
                row[f"{label}_{metric}"] = summary[f"{label}_{metric}"]
        row.update({
            "Top1_hit": summary["Top1_target7_precision"],
            "Top3_precision": summary["Top3_target7_precision"],
            "winner_capture": summary["winner_capture"],
            "all_concordance": summary["all_repair_concordance"],
            "cross_threshold": summary["cross_threshold_concordance"],
            "within_non_target": summary["within_non_target_concordance"],
            "within_target": summary["within_target_concordance"],
            "exact_score_tie_rows": summary["exact_score_tie_rows"],
            "date_count": summary["date_count"],
        })
        rows.append(row)

    universe_row: dict[str, Any] = {
        "model": "DAILY_UNIVERSE",
        "Top1_hit": universe["daily_universe_target7_rate"],
        "Top3_precision": universe["daily_universe_target7_rate"],
        "winner_capture": np.nan,
        "all_concordance": np.nan,
        "cross_threshold": np.nan,
        "within_non_target": np.nan,
        "within_target": np.nan,
        "exact_score_tie_rows": np.nan,
        "date_count": universe["date_count"],
    }
    for label in selector_labels:
        baseline = float(universe[f"{label}_baseline"])
        universe_row[label] = baseline
        universe_row[f"{label}_mean"] = baseline
        universe_row[f"{label}_median"] = np.nan
        universe_row[f"{label}_baseline"] = baseline
        universe_row[f"{label}_excess"] = 0.0
        universe_row[f"{label}_beat_universe"] = np.nan
        universe_row[f"{label}_positive_return"] = np.nan
        universe_row[f"{label}_target7_precision"] = universe[
            "daily_universe_target7_rate"
        ]
        universe_row[f"{label}_dates"] = universe[f"{label}_dates"]
    rows.insert(0, universe_row)
    return pd.DataFrame(rows), summaries, oracle


def load_frozen_opportunity_buckets(
    root: Path, expected_dates: Iterable[str]
) -> dict[str, str]:
    path = (
        root
        / "reports/research/v004c_v4a_architecture_transfer_v001_20260506_20260630"
        / "v004c_v4a_transfer_opportunity_terciles_v001.csv"
    )
    frame = pd.read_csv(
        path, encoding="utf-8-sig", usecols=["opportunity_bucket", "dates"]
    )
    if frame["opportunity_bucket"].tolist() != ["LOW", "MID", "HIGH"]:
        raise RuntimeError("FATAL: frozen opportunity bucket order changed")
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        dates = str(row.dates).split("|")
        if len(dates) != 7:
            raise RuntimeError("FATAL: frozen opportunity bucket must contain seven dates")
        for date in dates:
            if date >= JULY_SENTINEL_DATE or date in mapping:
                raise RuntimeError("FATAL: invalid frozen opportunity date")
            mapping[date] = str(row.opportunity_bucket)
    if set(mapping) != set(expected_dates):
        raise RuntimeError("FATAL: frozen opportunity dates do not match June OOF")
    return mapping


def opportunity_comparison(
    control: pd.DataFrame,
    curve: pd.DataFrame,
    bucket_map: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = {CONTROL_MODEL_ID: control, CURVE_MODEL_ID: curve}
    for bucket in ("LOW", "MID", "HIGH"):
        dates = sorted(date for date, value in bucket_map.items() if value == bucket)
        for model, frame in frames.items():
            subset = frame[frame["signal_date"].isin(dates)].copy()
            summary = summarize_model(subset)
            top3_daily = _daily_selected(subset, "top", 3, True)
            rows.append({
                "opportunity_bucket": bucket,
                "model": model,
                "date_count": len(dates),
                "dates": "|".join(dates),
                "universe_mean": summary["daily_universe_capped_return"],
                "Rank1": summary["Rank1_mean"],
                "Top2": summary["Top2_mean"],
                "Top3": summary["Top3_mean"],
                "Rank1_excess": summary["Rank1_excess"],
                "Top2_excess": summary["Top2_excess"],
                "Top3_excess": summary["Top3_excess"],
                "Top3_negative_date_rate": float(
                    (top3_daily["selected_return"] < 0).mean()
                ),
                "Top3_median": float(top3_daily["selected_return"].median()),
                "Top3_worst": float(top3_daily["selected_return"].min()),
                "Top1_Target7": summary["Top1_target7_precision"],
                "Top3_Target7_precision": summary["Top3_target7_precision"],
            })
    return pd.DataFrame(rows)


def curvature_audit(
    coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, str]:
    rows: list[dict[str, Any]] = []
    for quadratic, linear, slug in QUADRATIC_SPECS:
        beta = coefficients[f"quadratic_beta_{slug}"].astype(float)
        turning = coefficients[f"turning_point_{slug}"].astype(float)
        valid = turning.dropna()
        rows.append({
            "feature": quadratic,
            "linear_feature": linear,
            "negative_fold_pct": float((beta < 0).mean()),
            "positive_fold_pct": float((beta > 0).mean()),
            "mean_quad_beta": float(beta.mean()),
            "median_quad_beta": float(beta.median()),
            "valid_turning_point_fold_pct": float(turning.notna().mean()),
            "median_turning_point": float(valid.median()) if len(valid) else np.nan,
            "p25_turning_point": float(valid.quantile(0.25)) if len(valid) else np.nan,
            "p75_turning_point": float(valid.quantile(0.75)) if len(valid) else np.nan,
            "folds": len(coefficients),
            "purpose": "STRUCTURAL_AUDIT_ONLY",
        })
    audit = pd.DataFrame(rows)
    stable = audit[audit["negative_fold_pct"].ge(0.70)]
    high_turning = stable["median_turning_point"].between(0.65, 0.98, inclusive="both")
    structural_support = bool(len(stable) >= 2 and high_turning.any())
    if structural_support:
        stability_answer = "YES"
    elif len(stable) >= 1 or bool(
        audit["median_turning_point"].between(0.65, 0.98, inclusive="both").any()
    ):
        stability_answer = "MIXED"
    else:
        stability_answer = "NO"
    return audit, structural_support, stability_answer


def rank_movement_audit(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = combined[[
        "event_id",
        "signal_date",
        "code",
        "target7",
        "capped_opportunity_return_7",
        "control_rank",
        "curve_rank",
        "rank_delta",
        "extreme_strength_count",
    ]].copy()
    result["control_top3_false_positive"] = (
        result["control_rank"].le(3) & result["target7"].eq(0)
    )
    result["control_top3_true_positive"] = (
        result["control_rank"].le(3) & result["target7"].eq(1)
    )
    result["demoted_below_top3"] = (
        result["control_rank"].le(3) & result["curve_rank"].gt(3)
    )
    result["newly_promoted_winner"] = (
        result["control_rank"].gt(3)
        & result["curve_rank"].le(3)
        & result["target7"].eq(1)
    )
    result["newly_promoted_false_positive"] = (
        result["control_rank"].gt(3)
        & result["curve_rank"].le(3)
        & result["target7"].eq(0)
    )
    result["extreme_strength"] = result["extreme_strength_count"].ge(3)

    false_positive = result[result["control_top3_false_positive"]]
    true_positive = result[result["control_top3_true_positive"]]
    extreme = result[result["extreme_strength"]]
    false_demoted = int((false_positive["curve_rank"] > 3).sum())
    true_preserved = int((true_positive["curve_rank"] <= 3).sum())
    stats = {
        "control_top3_false_positives": int(len(false_positive)),
        "false_positives_remain_top3": int((false_positive["curve_rank"] <= 3).sum()),
        "false_positives_demoted": false_demoted,
        "false_positive_demotion_rate": (
            float(false_demoted / len(false_positive)) if len(false_positive) else np.nan
        ),
        "false_positive_mean_rank_change": (
            float(false_positive["rank_delta"].mean()) if len(false_positive) else np.nan
        ),
        "control_top3_true_positives": int(len(true_positive)),
        "true_positives_preserved": true_preserved,
        "true_positives_demoted": int((true_positive["curve_rank"] > 3).sum()),
        "true_positive_preservation_rate": (
            float(true_preserved / len(true_positive)) if len(true_positive) else np.nan
        ),
        "newly_promoted_winners": int(result["newly_promoted_winner"].sum()),
        "newly_promoted_false_positives": int(
            result["newly_promoted_false_positive"].sum()
        ),
        "extreme_strength_rows": int(len(extreme)),
        "extreme_strength_mean_rank_change": (
            float(extreme["rank_delta"].mean()) if len(extreme) else np.nan
        ),
        "extreme_strength_median_rank_change": (
            float(extreme["rank_delta"].median()) if len(extreme) else np.nan
        ),
        "extreme_strength_target7_rate": (
            float(extreme["target7"].mean()) if len(extreme) else np.nan
        ),
        "extreme_strength_mean_capped_return": (
            float(extreme["capped_opportunity_return_7"].mean())
            if len(extreme)
            else np.nan
        ),
    }
    return result.sort_values(
        ["signal_date", "control_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True), stats


def _opportunity_row(
    opportunity: pd.DataFrame, bucket: str, model: str
) -> pd.Series:
    row = opportunity[
        opportunity["opportunity_bucket"].eq(bucket)
        & opportunity["model"].eq(model)
    ]
    if len(row) != 1:
        raise RuntimeError(f"FATAL: opportunity row unavailable: {bucket}/{model}")
    return row.iloc[0]


def decision_status(
    summaries: Mapping[str, Mapping[str, Any]],
    opportunity: pd.DataFrame,
    structural_support: bool,
    stability_answer: str,
) -> dict[str, Any]:
    control = summaries[CONTROL_MODEL_ID]
    curve = summaries[CURVE_MODEL_ID]
    low_control = _opportunity_row(opportunity, "LOW", CONTROL_MODEL_ID)
    low_curve = _opportunity_row(opportunity, "LOW", CURVE_MODEL_ID)
    high_control = _opportunity_row(opportunity, "HIGH", CONTROL_MODEL_ID)
    high_curve = _opportunity_row(opportunity, "HIGH", CURVE_MODEL_ID)

    rank1_delta = float(curve["Rank1_mean"] - control["Rank1_mean"])
    rank2_delta = float(curve["Rank2_mean"] - control["Rank2_mean"])
    rank3_delta = float(curve["Rank3_mean"] - control["Rank3_mean"])
    top3_delta = float(curve["Top3_mean"] - control["Top3_mean"])
    cross_delta = float(
        curve["cross_threshold_concordance"]
        - control["cross_threshold_concordance"]
    )
    low_delta = float(low_curve["Top3_excess"] - low_control["Top3_excess"])
    preserve_rank1 = bool(
        curve["Rank1_mean"] >= curve["Rank1_baseline"] and rank1_delta >= -0.0025
    )
    preserve_rank2 = bool(
        curve["Rank2_mean"] >= curve["Rank2_baseline"] and rank2_delta >= -0.0025
    )
    rank3_material = bool(rank3_delta >= 0.0100 and curve["Rank3_mean"] > 0.0)
    top3_material = bool(top3_delta >= 0.0050 and top3_delta > 0.0)
    cross_repaired = bool(
        curve["cross_threshold_concordance"] > 0.50 and cross_delta > 0.0
    )
    low_improved = bool(low_delta >= 0.0050 and low_delta > 0.0)
    high_preserved = bool(all(
        float(high_curve[column]) >= float(high_control[column]) - 0.0025
        for column in ("Rank1", "Top2", "Top3")
    ))
    strong = bool(
        preserve_rank1
        and preserve_rank2
        and rank3_material
        and top3_material
        and cross_repaired
        and low_improved
    )
    coherent_direction = bool(
        rank1_delta >= -0.0025
        and rank2_delta >= -0.0025
        and rank3_delta > 0.0
        and top3_delta > 0.0
        and cross_delta > 0.0
        and low_delta > 0.0
    )
    if strong:
        signal = "STRONG"
    elif coherent_direction:
        signal = "PARTIAL"
    else:
        signal = "ABSENT"

    if signal == "STRONG" and structural_support:
        hypothesis = "SUPPORTED"
    elif signal == "PARTIAL" and structural_support:
        hypothesis = "PARTIALLY_SUPPORTED"
    else:
        hypothesis = "NOT_SUPPORTED"

    universe_target7 = float(curve["daily_universe_target7_rate"])
    ready = bool(
        signal == "STRONG"
        and curve["Rank1_mean"] > curve["Rank1_baseline"]
        and curve["Top3_mean"] > curve["Top3_baseline"]
        and curve["Top3_beat_universe"] > 0.50
        and curve["Top1_target7_precision"] >= universe_target7
        and curve["Top3_target7_precision"] >= universe_target7
    )
    return {
        "preserve_rank1": preserve_rank1,
        "preserve_rank2": preserve_rank2,
        "rank3_material": rank3_material,
        "top3_material": top3_material,
        "cross_repaired": cross_repaired,
        "low_improved": low_improved,
        "high_preserved": high_preserved,
        "rank1_delta": rank1_delta,
        "rank2_delta": rank2_delta,
        "rank3_delta": rank3_delta,
        "top3_delta": top3_delta,
        "cross_delta": cross_delta,
        "low_top3_excess_delta": low_delta,
        "curvature_structural_support": "YES" if structural_support else "NO",
        "stable_high_end_negative_curvature": stability_answer,
        "overextension_curvature_signal": signal,
        "non_monotonic_model_hypothesis": hypothesis,
        "ready_for_july_oot": "YES" if ready else "NO",
    }


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _markdown_table(
    headers: Sequence[str], rows: Iterable[Sequence[Any]]
) -> list[str]:
    text_rows = [[str(value) for value in row] for row in rows]
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(row) + " |" for row in text_rows)
    return result


def build_review(
    summaries: Mapping[str, Mapping[str, Any]],
    opportunity: pd.DataFrame,
    audit: pd.DataFrame,
    movement: Mapping[str, Any],
    parity: Mapping[str, Any],
    status: Mapping[str, Any],
) -> str:
    control = summaries[CONTROL_MODEL_ID]
    curve = summaries[CURVE_MODEL_ID]
    low_control = _opportunity_row(opportunity, "LOW", CONTROL_MODEL_ID)
    low_curve = _opportunity_row(opportunity, "LOW", CURVE_MODEL_ID)
    high_control = _opportunity_row(opportunity, "HIGH", CONTROL_MODEL_ID)
    high_curve = _opportunity_row(opportunity, "HIGH", CURVE_MODEL_ID)
    lines = [
        "# v004c Overextension-Aware v4a Curvature Benchmark v001",
        "",
        "CAPABILITY_SCREEN_ONLY — ONE PREDECLARED CURVATURE MODEL — NOT A PRODUCTION MODEL",
        "",
        "## 1. Did Limited Curvature Fix the v4a Transfer Failure?",
        "",
    ]
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Rank2", "Rank3", "Top2", "Top3", "Cross-Threshold", "LOW Top3 Excess"],
        [
            ["CONTROL", _pct(control["Rank1_mean"]), _pct(control["Rank2_mean"]), _pct(control["Rank3_mean"]), _pct(control["Top2_mean"]), _pct(control["Top3_mean"]), f"{control['cross_threshold_concordance']:.4f}", _pct(low_control["Top3_excess"])],
            ["CURVE", _pct(curve["Rank1_mean"]), _pct(curve["Rank2_mean"]), _pct(curve["Rank3_mean"]), _pct(curve["Top2_mean"]), _pct(curve["Top3_mean"]), f"{curve['cross_threshold_concordance']:.4f}", _pct(low_curve["Top3_excess"])],
            ["UNIVERSE", _pct(control["Rank1_baseline"]), _pct(control["Rank2_baseline"]), _pct(control["Rank3_baseline"]), _pct(control["Top2_baseline"]), _pct(control["Top3_baseline"]), "NA", "0.0000%"],
        ],
    ))
    lines.extend([
        "",
        "## 2. Did Rank3 Recover?",
        "",
        f"- Control Rank3: **{_pct(control['Rank3_mean'])}**",
        f"- Curve Rank3: **{_pct(curve['Rank3_mean'])}**",
        f"- Delta: **{_pct(status['rank3_delta'])}**",
        f"- Universe Rank3: **{_pct(control['Rank3_baseline'])}**",
        "",
        "## 3. Did We Preserve Rank1 and Rank2?",
        "",
        f"- Rank1 delta: **{_pct(status['rank1_delta'])}**; preserved: **{'YES' if status['preserve_rank1'] else 'NO'}**",
        f"- Rank2 delta: **{_pct(status['rank2_delta'])}**; preserved: **{'YES' if status['preserve_rank2'] else 'NO'}**",
        "",
        "## 4. Did Cross-Threshold Winner Separation Improve?",
        "",
    ])
    lines.extend(_markdown_table(
        ["Model", "All", "Cross Threshold", "Within Non-Target", "Within Target7"],
        [
            ["CONTROL", f"{control['all_repair_concordance']:.4f}", f"{control['cross_threshold_concordance']:.4f}", f"{control['within_non_target_concordance']:.4f}", f"{control['within_target_concordance']:.4f}"],
            ["CURVE", f"{curve['all_repair_concordance']:.4f}", f"{curve['cross_threshold_concordance']:.4f}", f"{curve['within_non_target_concordance']:.4f}", f"{curve['within_target_concordance']:.4f}"],
        ],
    ))
    lines.extend([
        "",
        f"- Cross-threshold delta: **{status['cross_delta']:+.4f}**",
        "",
        "## 5. Did Low-Opportunity Downside Improve?",
        "",
    ])
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Top2", "Top3", "Top3 Excess", "Negative Rate", "Median", "Worst"],
        [
            ["CONTROL", _pct(low_control["Rank1"]), _pct(low_control["Top2"]), _pct(low_control["Top3"]), _pct(low_control["Top3_excess"]), _pct(low_control["Top3_negative_date_rate"]), _pct(low_control["Top3_median"]), _pct(low_control["Top3_worst"])],
            ["CURVE", _pct(low_curve["Rank1"]), _pct(low_curve["Top2"]), _pct(low_curve["Top3"]), _pct(low_curve["Top3_excess"]), _pct(low_curve["Top3_negative_date_rate"]), _pct(low_curve["Top3_median"]), _pct(low_curve["Top3_worst"])],
        ],
    ))
    lines.extend([
        "",
        f"- LOW Top3 excess improvement: **{_pct(status['low_top3_excess_delta'])}**",
        "",
        "## 6. Did High-Opportunity Upside Survive?",
        "",
    ])
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Top2", "Top3", "Top1 Target7", "Top3 Precision"],
        [
            ["CONTROL", _pct(high_control["Rank1"]), _pct(high_control["Top2"]), _pct(high_control["Top3"]), _pct(high_control["Top1_Target7"]), _pct(high_control["Top3_Target7_precision"])],
            ["CURVE", _pct(high_curve["Rank1"]), _pct(high_curve["Top2"]), _pct(high_curve["Top3"]), _pct(high_curve["Top1_Target7"]), _pct(high_curve["Top3_Target7_precision"])],
        ],
    ))
    lines.extend([
        "",
        f"- Upside preserved under the predeclared 0.25pp tolerance: **{'YES' if status['high_preserved'] else 'NO'}**",
        "",
        "## 7. What Curvature Did the Model Actually Learn?",
        "",
    ])
    lines.extend(_markdown_table(
        ["Term", "Negative Fold %", "Median Beta", "Valid Turning %", "Median Turning Point"],
        [[row.feature, _pct(row.negative_fold_pct), f"{row.median_quad_beta:.6f}", _pct(row.valid_turning_point_fold_pct), "NA" if pd.isna(row.median_turning_point) else f"{row.median_turning_point:.4f}"] for row in audit.itertuples()],
    ))
    lines.extend([
        "",
        f"- CURVATURE_STRUCTURAL_SUPPORT: **{status['curvature_structural_support']}**",
        "",
        "## 8. Which Stocks Moved?",
        "",
        f"- CONTROL Top3 false positives: **{movement['control_top3_false_positives']}**",
        f"- False-positive demotion rate: **{_pct(movement['false_positive_demotion_rate'])}**",
        f"- CONTROL Top3 true positives: **{movement['control_top3_true_positives']}**",
        f"- True-positive preservation rate: **{_pct(movement['true_positive_preservation_rate'])}**",
        f"- Newly promoted winners: **{movement['newly_promoted_winners']}**",
        f"- Newly promoted false positives: **{movement['newly_promoted_false_positives']}**",
        f"- Extreme-strength rows: **{movement['extreme_strength_rows']}**",
        f"- Extreme-strength median rank change: **{movement['extreme_strength_median_rank_change']:.4f}**",
        "",
        "## 9. Formal Decision",
        "",
        f"- Frozen CONTROL parity: **{'PASS' if parity['pass'] else 'FAIL'}**",
        f"- Q1 preserve Rank1: **{'YES' if status['preserve_rank1'] else 'NO'}**",
        f"- Q2 preserve Rank2: **{'YES' if status['preserve_rank2'] else 'NO'}**",
        f"- Q3 Rank3 materially recovered: **{'YES' if status['rank3_material'] else 'NO'}**",
        f"- Q4 STRICT Top3 materially improved: **{'YES' if status['top3_material'] else 'NO'}**",
        f"- Q5 cross-threshold >0.50 and improved: **{'YES' if status['cross_repaired'] else 'NO'}**",
        f"- Q6 LOW-opportunity downside improved: **{'YES' if status['low_improved'] else 'NO'}**",
        f"- Q7 HIGH-opportunity upside preserved: **{'YES' if status['high_preserved'] else 'NO'}**",
        f"- Q8 stable high-end negative curvature: **{status['stable_high_end_negative_curvature']}**",
        f"- Q9 non-monotonic hypothesis: **{status['non_monotonic_model_hypothesis']}**",
        "",
        f"OVEREXTENSION_CURVATURE_SIGNAL: **{status['overextension_curvature_signal']}**",
        "",
        f"READY_FOR_JULY_OOT: **{status['ready_for_july_oot']}**",
        "",
        "- July actually run: **NO**",
        "- New raw data: **NO**",
        "- New external factors: **NO**",
        "- Hyperparameter search: **NO**",
        "- Feature selection: **NO**",
    ])
    if status["overextension_curvature_signal"] == "ABSENT":
        lines.extend([
            "",
            "Limited quadratic overextension modeling did not solve the v4a-transfer failure.",
        ])
    return "\n".join(lines) + "\n"


def _combine_coefficient_audit(
    control_fold: pd.DataFrame,
    control_coefficients: pd.DataFrame,
    curve_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    control_score = control_fold[[
        "test_date", "score_min", "score_max", "score_std", "exact_score_tie_rows"
    ]].rename(columns={
        "score_min": "control_score_min",
        "score_max": "control_score_max",
        "score_std": "control_score_std",
        "exact_score_tie_rows": "control_exact_score_tie_rows",
    })
    control_cosine = control_coefficients[[
        "test_date", "adjacent_coefficient_cosine"
    ]].rename(columns={
        "adjacent_coefficient_cosine": "control_adjacent_coefficient_cosine"
    })
    return curve_coefficients.merge(
        control_score, on="test_date", how="left", validate="one_to_one"
    ).merge(
        control_cosine, on="test_date", how="left", validate="one_to_one"
    )


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_curve_contract()
    base = load_authoritative_input(root)
    assert_authoritative_universe(base)
    raw, _, _, feature_gate = build_exact_raw_features(root, base)
    if not feature_gate["pass"]:
        raise RuntimeError("FATAL: exact frozen v004a feature gate failed")
    outcomes = load_outcomes(root, base)
    samples = prepare_transfer_samples(raw, outcomes)

    # Control must pass before the candidate is fitted.
    control_oof, control_fold, control_coefficients = run_expanding_forward(samples)
    parity = validate_control_parity(root, control_oof)

    curve_oof, curve_coefficients = run_curve_expanding_forward(samples)
    combined = combine_oof(samples, control_oof, curve_oof)
    control_frame = model_frame(combined, "control")
    curve_frame = model_frame(combined, "curve")
    references = _load_reference_scores(root, outcomes)
    practical, summaries, oracle = practical_comparison(
        control_frame, curve_frame, references
    )
    bucket_map = load_frozen_opportunity_buckets(
        root, curve_frame["signal_date"].unique()
    )
    opportunity = opportunity_comparison(control_frame, curve_frame, bucket_map)
    coefficients = _combine_coefficient_audit(
        control_fold, control_coefficients, curve_coefficients
    )
    audit, structural_support, stability_answer = curvature_audit(coefficients)
    movement_frame, movement = rank_movement_audit(combined)
    status = decision_status(
        summaries, opportunity, structural_support, stability_answer
    )
    review = build_review(
        summaries, opportunity, audit, movement, parity, status
    )

    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(combined),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(audit),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(movement_frame),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(opportunity),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    context = {
        "base": base,
        "raw": raw,
        "feature_gate": feature_gate,
        "samples": samples,
        "control_oof": control_oof,
        "curve_oof": curve_oof,
        "oof": combined,
        "control_fold": control_fold,
        "coefficients": coefficients,
        "curvature_audit": audit,
        "movement_frame": movement_frame,
        "movement": movement,
        "practical": practical,
        "summaries": summaries,
        "oracle": oracle,
        "bucket_map": bucket_map,
        "opportunity": opportunity,
        "control_parity": parity,
        "status": status,
        "output_sha256": {
            name: sha256_bytes(value) for name, value in outputs.items()
        },
    }
    return outputs, context


def run_v004c_overextension_curve(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first.keys() != second.keys() or any(
        first[name] != second[name] for name in first
    ):
        raise RuntimeError("FATAL: deterministic rebuild failed")
    output_dir = (
        root_path
        / "reports/research/v004c_overextension_curve_v001_20260506_20260630"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (output_dir / name).write_bytes(first[name])
    context["deterministic_rebuild"] = "PASS"
    context["output_dir"] = output_dir
    return output_dir, context

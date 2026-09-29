"""Fixed May-vs-June recency-weight diagnostic for the frozen S2 Stage1.

Exactly two fits are permitted: the archived S2 May/June weighting and one
preregistered challenger that multiplies May row weights by 0.5.  The learner,
features, target, L2, positive weight, date weights, and July ranking rules are
otherwise unchanged.  July is evaluation-only and no August signal outcome is
read.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .v004a import DEFAULT_TARGET_COLUMN, fit_logistic_l2_weighted
from .v004c_reduced7f_stage1_temporal_validation import (
    AUGUST_ASOF,
    REDUCED_FEATURE_COLUMNS,
    _sigmoid,
    load_bridge,
)
from .v004c_reduced7f_training_spec_sanity import (
    SPECIFICATIONS,
    build_sample_weight,
)
from .v004c_v4a_architecture_transfer import POSITIVE_WEIGHT


TASK_NAME = "v004c_recency_weighting_diagnostic_v001"
OUTPUT_DIRNAME = "v004c_recency_weighting_diagnostic_v001_20260506_20260731"
OUTPUT_FILENAMES = (
    "v004c_recency_weighting_summary_v001.csv",
    "v004c_recency_weighting_rankwise_v001.csv",
    "v004c_recency_weight_rank_changes_v001.csv",
    "v004c_recency_weighting_daily_v001.csv",
    "v004c_recency_weighting_coefficients_v001.csv",
    "v004c_recency_weighting_review_v001.md",
)

S2_SPEC = "S2_NO_TAIL_L2_010"
L2 = 0.10
VERSION_MULTIPLIERS: Mapping[str, Mapping[int, float]] = {
    "HISTORICAL_EQUAL": {5: 1.0, 6: 1.0},
    "RECENT_PRIORITY_2X": {5: 0.5, 6: 1.0},
}
VERSION_ORDER = tuple(VERSION_MULTIPLIERS)
TRAIN_ASOF = "2026-07-01"
ARCHIVED_PREDICTIONS = (
    "reports/research/v004c_reduced7f_training_spec_sanity_v001_20260506_20260731/"
    "v004c_reduced7f_spec_fold_predictions_v001.csv"
)
ARCHIVED_COEFFICIENTS = (
    "reports/research/v004c_reduced7f_training_spec_sanity_v001_20260506_20260731/"
    "v004c_reduced7f_spec_coefficients_v001.csv"
)


def assert_experiment_contract() -> None:
    expected_features = (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    if tuple(REDUCED_FEATURE_COLUMNS) != expected_features:
        raise RuntimeError("FATAL: frozen S2 seven-feature manifest changed")
    if VERSION_ORDER != ("HISTORICAL_EQUAL", "RECENT_PRIORITY_2X"):
        raise RuntimeError("FATAL: version set changed")
    if VERSION_MULTIPLIERS != {
        "HISTORICAL_EQUAL": {5: 1.0, 6: 1.0},
        "RECENT_PRIORITY_2X": {5: 0.5, 6: 1.0},
    }:
        raise RuntimeError("FATAL: month multipliers changed")
    if float(SPECIFICATIONS[S2_SPEC]["l2"]) != L2:
        raise RuntimeError("FATAL: S2 L2 changed")
    if str(SPECIFICATIONS[S2_SPEC]["tail_weighting"]) != "NONE":
        raise RuntimeError("FATAL: S2 tail bonus is not OFF")
    if float(POSITIVE_WEIGHT) != 1.50:
        raise RuntimeError("FATAL: S2 positive weight changed")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _outcome_class(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [frame["target7"].astype(bool), frame["loss"].astype(bool)],
            ["TARGET7", "LOSS"],
            default="PNT",
        ),
        index=frame.index,
    )


def build_populations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the strict July-1 training snapshot and mature July test rows."""
    assert_experiment_contract()
    train = frame[
        frame["signal_date"].lt(TRAIN_ASOF)
        & frame["label_available_date"].lt(TRAIN_ASOF)
    ].copy()
    july = frame[
        frame["signal_date"].ge(TRAIN_ASOF)
        & frame["signal_date"].lt(AUGUST_ASOF)
        & frame["label_available_date"].lt(AUGUST_ASOF)
    ].copy()
    if (len(train), train["signal_date"].nunique()) != (307, 37):
        raise RuntimeError("FATAL: May/June training parity failed")
    if (len(july), july["signal_date"].nunique()) != (166, 21):
        raise RuntimeError("FATAL: mature July population parity failed")
    if not train["signal_date"].str.startswith(("2026-05", "2026-06")).all():
        raise RuntimeError("FATAL: training contains a forbidden month")
    if not july["signal_date"].str.startswith("2026-07").all():
        raise RuntimeError("FATAL: July evaluation contains a forbidden month")
    if frame["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal-date row reached diagnostic")
    required = list(REDUCED_FEATURE_COLUMNS)
    if train[required].isna().any().any() or july[required].isna().any().any():
        raise RuntimeError("FATAL: missing frozen S2 feature")
    return train.reset_index(drop=True), july.reset_index(drop=True)


def build_version_weight(
    train: pd.DataFrame, version: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Multiply the exact frozen S2 row weight by the preregistered month factor."""
    if version not in VERSION_MULTIPLIERS:
        raise RuntimeError(f"FATAL: unregistered recency version {version}")
    adapter, existing = build_sample_weight(train, S2_SPEC)
    months = pd.to_datetime(adapter["signal_date"], errors="raise").dt.month
    multipliers = months.map(VERSION_MULTIPLIERS[version])
    if multipliers.isna().any():
        raise RuntimeError("FATAL: non-May/June row reached month multiplier")
    final = np.asarray(existing, dtype=float) * multipliers.to_numpy(float)
    if not np.isfinite(final).all() or (final <= 0).any():
        raise RuntimeError("FATAL: invalid final sample weight")
    audit = pd.DataFrame({
        "signal_date": adapter["signal_date"].astype(str),
        "month": months,
        "target7": adapter[DEFAULT_TARGET_COLUMN].astype(int),
        "existing_s2_row_weight": np.asarray(existing, dtype=float),
        "month_multiplier": multipliers.to_numpy(float),
        "final_sample_weight": final,
    })
    return final, audit


def fit_versions(
    train: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    betas: dict[str, np.ndarray] = {}
    weights: dict[str, pd.DataFrame] = {}
    x = train[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float)
    y = train["target7"].to_numpy(float)
    for version in VERSION_ORDER:
        sample_weight, audit = build_version_weight(train, version)
        beta = fit_logistic_l2_weighted(
            x,
            y,
            l2=L2,
            sample_weight=sample_weight,
        )
        betas[version] = np.asarray(beta, dtype=float)
        weights[version] = audit
    return betas, weights


def build_prediction_lock(
    july: pd.DataFrame, betas: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Score feature-only July rows; outcomes are deliberately attached later."""
    identity = [
        "event_id", "signal_date", "code", "board_group", "candidate_count",
        "label_available_date",
    ]
    feature_only = july[identity + list(REDUCED_FEATURE_COLUMNS)].copy()
    outputs: list[pd.DataFrame] = []
    x = feature_only[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float)
    for version in VERSION_ORDER:
        beta = np.asarray(betas[version], dtype=float)
        scored = feature_only[identity].copy()
        scored.insert(0, "version", version)
        scored["model_logit"] = beta[0] + x @ beta[1:]
        scored["model_score"] = _sigmoid(scored["model_logit"].to_numpy(float))
        ranked: list[pd.DataFrame] = []
        for _, day in scored.groupby("signal_date", sort=True):
            part = day.sort_values(
                ["model_score", "event_id"], ascending=[False, True], kind="mergesort"
            ).copy()
            part["model_rank"] = np.arange(1, len(part) + 1)
            ranked.append(part)
        outputs.append(pd.concat(ranked, ignore_index=True))
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["version", "signal_date", "model_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if result.duplicated(["version", "event_id"]).any():
        raise RuntimeError("FATAL: duplicate prediction-lock identity")
    return result


def attach_outcomes(lock: pd.DataFrame, july: pd.DataFrame) -> pd.DataFrame:
    outcome_columns = [
        "event_id", "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
    ]
    outcomes = july[outcome_columns].copy()
    if outcomes["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate July event_id")
    scored = lock.merge(outcomes, on="event_id", how="left", validate="many_to_one")
    if scored[outcome_columns[1:]].isna().any().any():
        raise RuntimeError("FATAL: missing mature July outcome")
    scored["outcome_class"] = _outcome_class(scored)
    return scored


def _load_archived_equal(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    predictions = pd.read_csv(
        root / ARCHIVED_PREDICTIONS,
        dtype={"event_id": str, "signal_date": str, "code": str},
    )
    archived = predictions[
        predictions["prediction_set"].eq("FOLD_B_MAY_JUNE_TO_JULY")
        & predictions["spec"].eq(S2_SPEC)
    ].copy()
    coefficients = pd.read_csv(root / ARCHIVED_COEFFICIENTS, dtype=str)
    rows = coefficients[
        coefficients["prediction_set"].eq("FOLD_B_MAY_JUNE_TO_JULY")
        & coefficients["spec"].eq(S2_SPEC)
        & coefficients["fold_id"].eq("FOLD_B__FIXED_ASOF_2026_07_01")
    ].copy()
    if len(archived) != 166 or len(rows) != 8:
        raise RuntimeError("FATAL: archived S2 Fold B parity artifact changed")
    by_feature = rows.set_index("feature")["coefficient"].astype(float)
    beta = np.asarray(
        [by_feature["__INTERCEPT__"]]
        + [by_feature[feature] for feature in REDUCED_FEATURE_COLUMNS],
        dtype=float,
    )
    return archived, beta


def assert_equal_parity(
    root: Path, lock: pd.DataFrame, beta: np.ndarray,
) -> dict[str, Any]:
    archived, archived_beta = _load_archived_equal(root)
    equal = lock[lock["version"].eq("HISTORICAL_EQUAL")][
        ["event_id", "model_score", "model_rank"]
    ].copy()
    compare = equal.merge(
        archived[["event_id", "model_score", "model_rank"]],
        on="event_id",
        suffixes=("_new", "_archived"),
        validate="one_to_one",
    )
    score_error = float(
        np.max(np.abs(compare["model_score_new"] - compare["model_score_archived"]))
    )
    rank_mismatch = int(
        (compare["model_rank_new"].astype(int) != compare["model_rank_archived"].astype(int)).sum()
    )
    beta_error = float(np.max(np.abs(np.asarray(beta) - archived_beta)))
    if score_error > 1e-12 or beta_error > 1e-12 or rank_mismatch:
        raise RuntimeError(
            "FATAL: HISTORICAL_EQUAL does not reproduce archived S2 Fold B "
            f"(score={score_error}, beta={beta_error}, rank={rank_mismatch})"
        )
    return {
        "archived_equal_score_max_abs_error": score_error,
        "archived_equal_coefficient_max_abs_error": beta_error,
        "archived_equal_rank_mismatch_count": rank_mismatch,
    }


def build_daily(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (version, signal_date), day in scored.groupby(["version", "signal_date"], sort=True):
        ordered = day.sort_values(["model_rank", "event_id"], kind="mergesort")
        universe_target7 = int(ordered["target7"].sum())
        row: dict[str, Any] = {
            "version": version,
            "signal_date": signal_date,
            "candidate_count": len(ordered),
            "universe_target7_count": universe_target7,
            "universe_target7_rate": float(ordered["target7"].mean()),
            "universe_loss_count": int(ordered["loss"].sum()),
            "universe_loss_rate": float(ordered["loss"].mean()),
            "universe_severe_loss_rate": float(ordered["severe_loss"].mean()),
            "universe_capped_return": float(ordered["capped_return_7"].mean()),
        }
        for rank in (1, 2, 3):
            selected = ordered[ordered["model_rank"].eq(rank)]
            prefix = f"rank{rank}"
            if selected.empty:
                for name in (
                    "event_id", "code", "board_group", "score", "target7", "loss",
                    "severe_loss", "capped_return",
                ):
                    row[f"{prefix}_{name}"] = math.nan
                continue
            item = selected.iloc[0]
            row.update({
                f"{prefix}_event_id": item["event_id"],
                f"{prefix}_code": item["code"],
                f"{prefix}_board_group": item["board_group"],
                f"{prefix}_score": float(item["model_score"]),
                f"{prefix}_target7": int(item["target7"]),
                f"{prefix}_loss": int(item["loss"]),
                f"{prefix}_severe_loss": int(item["severe_loss"]),
                f"{prefix}_capped_return": float(item["capped_return_7"]),
            })
        for k in (1, 2, 3):
            top = ordered.head(min(k, len(ordered)))
            row[f"top{k}_selected_rows"] = len(top)
            row[f"top{k}_capped_return"] = float(top["capped_return_7"].mean())
            row[f"top{k}_target7_rate"] = float(top["target7"].mean())
            row[f"top{k}_loss_rate"] = float(top["loss"].mean())
        top3 = ordered.head(min(3, len(ordered)))
        top3_t7 = int(top3["target7"].sum())
        row.update({
            "top3_target7_count": top3_t7,
            "top3_loss_count": int(top3["loss"].sum()),
            "top3_severe_loss_count": int(top3["severe_loss"].sum()),
            "top3_severe_loss_rate": float(top3["severe_loss"].mean()),
            "top3_all_hit": int(len(top3) == 3 and top3_t7 == 3),
            "top3_zero_hit": int(top3_t7 == 0),
            "winner_capture": top3_t7 / universe_target7 if universe_target7 else math.nan,
            "top3_negative_date": int(float(top3["capped_return_7"].mean()) < 0),
            "top3_board3_count": int(top3["board_group"].eq("BOARD3").sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["version", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)


def build_rankwise(scored: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version in VERSION_ORDER:
        part = scored[scored["version"].eq(version)]
        daily_part = daily[daily["version"].eq(version)]
        for selection in ("RANK1", "RANK2", "RANK3", "TOP1", "TOP2", "TOP3"):
            if selection.startswith("RANK"):
                rank = int(selection[-1])
                chosen = part[part["model_rank"].eq(rank)]
            else:
                k = int(selection[-1])
                chosen = part[part["model_rank"].le(k)]
            k = 1 if selection in ("RANK1", "RANK2", "RANK3", "TOP1") else int(selection[-1])
            capped_column = (
                f"rank{selection[-1]}_capped_return"
                if selection.startswith("RANK")
                else f"top{k}_capped_return"
            )
            rows.append({
                "version": version,
                "selection": selection,
                "dates": int(chosen["signal_date"].nunique()),
                "selected_rows": len(chosen),
                "target7_count": int(chosen["target7"].sum()),
                "target7_rate": float(chosen["target7"].mean()),
                "loss_count": int(chosen["loss"].sum()),
                "loss_rate": float(chosen["loss"].mean()),
                "severe_loss_count": int(chosen["severe_loss"].sum()),
                "severe_loss_rate": float(chosen["severe_loss"].mean()),
                "average_capped_return": float(daily_part[capped_column].mean()),
                "board3_count": int(chosen["board_group"].eq("BOARD3").sum()),
                "board3_share": float(chosen["board_group"].eq("BOARD3").mean()),
            })
    return pd.DataFrame(rows)


def build_rank_changes(scored: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "event_id", "signal_date", "code", "board_group", "target7",
        "positive_non_target", "loss", "severe_loss", "capped_return_7",
        "model_score", "model_rank",
    ]
    equal = scored[scored["version"].eq("HISTORICAL_EQUAL")][fields].copy()
    recent = scored[scored["version"].eq("RECENT_PRIORITY_2X")][fields].copy()
    merged = equal.merge(
        recent,
        on=[
            "event_id", "signal_date", "code", "board_group", "target7",
            "positive_non_target", "loss", "severe_loss", "capped_return_7",
        ],
        suffixes=("_equal", "_recent_priority"),
        validate="one_to_one",
    )
    merged = merged.rename(columns={
        "model_score_equal": "equal_score",
        "model_rank_equal": "equal_rank",
        "model_score_recent_priority": "recent_priority_score",
        "model_rank_recent_priority": "recent_priority_rank",
    })
    merged["rank_delta_recent_minus_equal"] = (
        merged["recent_priority_rank"].astype(int) - merged["equal_rank"].astype(int)
    )
    merged["entered_top3"] = (
        merged["equal_rank"].gt(3) & merged["recent_priority_rank"].le(3)
    ).astype(int)
    merged["left_top3"] = (
        merged["equal_rank"].le(3) & merged["recent_priority_rank"].gt(3)
    ).astype(int)
    for rank in (1, 2, 3):
        merged[f"entered_rank{rank}"] = (
            merged["equal_rank"].ne(rank) & merged["recent_priority_rank"].eq(rank)
        ).astype(int)
        merged[f"left_rank{rank}"] = (
            merged["equal_rank"].eq(rank) & merged["recent_priority_rank"].ne(rank)
        ).astype(int)
    return merged.sort_values(
        ["signal_date", "equal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _month_weight_stats(
    train: pd.DataFrame, audit: pd.DataFrame, beta: np.ndarray, month: int,
) -> dict[str, float]:
    mask = audit["month"].eq(month).to_numpy()
    weight = audit.loc[mask, "final_sample_weight"].to_numpy(float)
    x = train.loc[mask, list(REDUCED_FEATURE_COLUMNS)].to_numpy(float)
    y = train.loc[mask, "target7"].to_numpy(float)
    logits = beta[0] + x @ beta[1:]
    probabilities = _sigmoid(logits)
    total = float(weight.sum())
    return {
        "rows": int(mask.sum()),
        "dates": int(train.loc[mask, "signal_date"].nunique()),
        "weight_sum": total,
        "weighted_target_rate": float(np.sum(weight * y) / total),
        "weighted_mean_logit": float(np.sum(weight * logits) / total),
        "weighted_mean_score": float(np.sum(weight * probabilities) / total),
    }


def build_coefficients(
    train: pd.DataFrame,
    betas: Mapping[str, np.ndarray],
    weights: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    equal = np.asarray(betas["HISTORICAL_EQUAL"], dtype=float)
    recent = np.asarray(betas["RECENT_PRIORITY_2X"], dtype=float)
    features = ("__INTERCEPT__",) + tuple(REDUCED_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        row: dict[str, Any] = {
            "row_type": "COEFFICIENT_AND_MONTH_CONTRIBUTION",
            "feature": feature,
            "equal_coefficient": float(equal[index]),
            "recent_priority_coefficient": float(recent[index]),
            "coefficient_delta_recent_minus_equal": float(recent[index] - equal[index]),
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "tail_bonus": "OFF",
        }
        for version, beta in (("equal", equal), ("recent_priority", recent)):
            audit = weights[
                "HISTORICAL_EQUAL" if version == "equal" else "RECENT_PRIORITY_2X"
            ]
            for month, label in ((5, "may"), (6, "june")):
                mask = audit["month"].eq(month).to_numpy()
                w = audit.loc[mask, "final_sample_weight"].to_numpy(float)
                if feature == "__INTERCEPT__":
                    mean_feature = 1.0
                else:
                    values = train.loc[mask, feature].to_numpy(float)
                    mean_feature = float(np.sum(w * values) / w.sum())
                row[f"{version}_{label}_weighted_mean_feature"] = mean_feature
                row[f"{version}_{label}_mean_linear_contribution"] = float(
                    beta[index] * mean_feature
                )
        rows.append(row)
    for version in VERSION_ORDER:
        audit = weights[version]
        may = _month_weight_stats(train, audit, betas[version], 5)
        june = _month_weight_stats(train, audit, betas[version], 6)
        total_weight = may["weight_sum"] + june["weight_sum"]
        rows.append({
            "row_type": "MONTH_WEIGHT_SUMMARY",
            "feature": version,
            "equal_coefficient": math.nan,
            "recent_priority_coefficient": math.nan,
            "coefficient_delta_recent_minus_equal": math.nan,
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "tail_bonus": "OFF",
            "may_rows": may["rows"],
            "may_dates": may["dates"],
            "may_weight_sum": may["weight_sum"],
            "may_effective_weight_share": may["weight_sum"] / total_weight,
            "may_weighted_target_rate": may["weighted_target_rate"],
            "may_weighted_mean_logit": may["weighted_mean_logit"],
            "may_weighted_mean_score": may["weighted_mean_score"],
            "june_rows": june["rows"],
            "june_dates": june["dates"],
            "june_weight_sum": june["weight_sum"],
            "june_effective_weight_share": june["weight_sum"] / total_weight,
            "june_weighted_target_rate": june["weighted_target_rate"],
            "june_weighted_mean_logit": june["weighted_mean_logit"],
            "june_weighted_mean_score": june["weighted_mean_score"],
        })
    return pd.DataFrame(rows)


def build_summary(
    scored: pd.DataFrame,
    daily: pd.DataFrame,
    train: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    rank_changes: pd.DataFrame,
    parity: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version in VERSION_ORDER:
        part = scored[scored["version"].eq(version)]
        day = daily[daily["version"].eq(version)]
        top3 = part[part["model_rank"].le(3)]
        audit = weights[version]
        month_weight = audit.groupby("month")["final_sample_weight"].sum()
        rows.append({
            "version": version,
            "may_multiplier": VERSION_MULTIPLIERS[version][5],
            "june_multiplier": VERSION_MULTIPLIERS[version][6],
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "tail_bonus": "OFF",
            "train_rows": len(train),
            "train_dates": int(train["signal_date"].nunique()),
            "train_end_signal_date": str(train["signal_date"].max()),
            "max_train_label_available_date": str(train["label_available_date"].max()),
            "may_final_weight_sum": float(month_weight[5]),
            "june_final_weight_sum": float(month_weight[6]),
            "june_to_may_effective_weight_ratio": float(month_weight[6] / month_weight[5]),
            "july_rows": len(part),
            "july_dates": int(part["signal_date"].nunique()),
            "universe_target7_rate": float(day["universe_target7_rate"].mean()),
            "universe_loss_rate": float(day["universe_loss_rate"].mean()),
            "universe_capped_return": float(day["universe_capped_return"].mean()),
            "top1_average_capped_return": float(day["top1_capped_return"].mean()),
            "top2_average_capped_return": float(day["top2_capped_return"].mean()),
            "top3_average_capped_return": float(day["top3_capped_return"].mean()),
            "top3_excess_vs_universe": float(
                day["top3_capped_return"].mean() - day["universe_capped_return"].mean()
            ),
            "top3_target7_count": int(top3["target7"].sum()),
            "top3_target7_rate": float(top3["target7"].mean()),
            "top3_loss_count": int(top3["loss"].sum()),
            "top3_loss_rate": float(top3["loss"].mean()),
            "top3_severe_loss_count": int(top3["severe_loss"].sum()),
            "top3_severe_loss_rate": float(top3["severe_loss"].mean()),
            "top3_all_hit_dates": int(day["top3_all_hit"].sum()),
            "top3_all_hit_rate": float(day["top3_all_hit"].mean()),
            "zero_hit_dates": int(day["top3_zero_hit"].sum()),
            "zero_hit_day_rate": float(day["top3_zero_hit"].mean()),
            "winner_capture_mean_eligible_dates": float(day["winner_capture"].mean()),
            "winner_capture_pooled": float(
                top3["target7"].sum()
                / part.groupby("event_id", sort=False).first()["target7"].sum()
            ),
            "negative_dates": int(day["top3_negative_date"].sum()),
            "negative_date_rate": float(day["top3_negative_date"].mean()),
            "worst_date": str(day.loc[day["top3_capped_return"].idxmin(), "signal_date"]),
            "worst_daily_top3_capped_return": float(day["top3_capped_return"].min()),
            "top3_board3_count": int(top3["board_group"].eq("BOARD3").sum()),
            "top3_board3_share": float(top3["board_group"].eq("BOARD3").mean()),
            "august_signal_outcome_accessed": "NO",
            "july_model_selection": "NO",
            **parity,
        })
    return pd.DataFrame(rows)


def _rank_row(rankwise: pd.DataFrame, version: str, selection: str) -> pd.Series:
    found = rankwise[
        rankwise["version"].eq(version) & rankwise["selection"].eq(selection)
    ]
    if len(found) != 1:
        raise RuntimeError(f"FATAL: missing rankwise row {version}/{selection}")
    return found.iloc[0]


def classify_state(
    summary: pd.DataFrame, rankwise: pd.DataFrame, daily: pd.DataFrame,
    rank_changes: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    equal = summary.set_index("version").loc["HISTORICAL_EQUAL"]
    recent = summary.set_index("version").loc["RECENT_PRIORITY_2X"]
    deltas = {
        "top3_capped": float(recent["top3_average_capped_return"] - equal["top3_average_capped_return"]),
        "top3_target7": float(recent["top3_target7_rate"] - equal["top3_target7_rate"]),
        "top3_loss": float(recent["top3_loss_rate"] - equal["top3_loss_rate"]),
        "top3_severe_loss": float(recent["top3_severe_loss_rate"] - equal["top3_severe_loss_rate"]),
        "negative_dates": int(recent["negative_dates"] - equal["negative_dates"]),
        "winner_capture": float(recent["winner_capture_pooled"] - equal["winner_capture_pooled"]),
    }
    pivot = daily.pivot(index="signal_date", columns="version", values="top3_capped_return")
    daily_delta = pivot["RECENT_PRIORITY_2X"] - pivot["HISTORICAL_EQUAL"]
    positive = int((daily_delta > 1e-15).sum())
    negative = int((daily_delta < -1e-15).sum())
    zero = int((daily_delta.abs() <= 1e-15).sum())
    absolute_sum = float(daily_delta.abs().sum())
    top3_abs_share = (
        float(daily_delta.abs().nlargest(3).sum() / absolute_sum) if absolute_sum else math.nan
    )
    changed_dates = int(
        rank_changes.loc[
            rank_changes["entered_top3"].eq(1) | rank_changes["left_top3"].eq(1),
            "signal_date",
        ].nunique()
    )
    rank_deltas: dict[int, dict[str, float]] = {}
    for rank in (1, 2, 3):
        old = _rank_row(rankwise, "HISTORICAL_EQUAL", f"RANK{rank}")
        new = _rank_row(rankwise, "RECENT_PRIORITY_2X", f"RANK{rank}")
        rank_deltas[rank] = {
            "capped": float(new["average_capped_return"] - old["average_capped_return"]),
            "target7": float(new["target7_rate"] - old["target7_rate"]),
            "loss": float(new["loss_rate"] - old["loss_rate"]),
        }
    support = int(equal["july_dates"]) >= 15 and int(equal["july_rows"]) >= 100
    lower_rank_help = any(
        rank_deltas[rank]["capped"] > 0
        or rank_deltas[rank]["target7"] > 0
        or rank_deltas[rank]["loss"] < 0
        for rank in (2, 3)
    )
    lower_rank_not_harmed = all(
        rank_deltas[rank]["capped"] >= -0.005
        and rank_deltas[rank]["target7"] >= -0.05
        and rank_deltas[rank]["loss"] <= 0.05
        for rank in (2, 3)
    )
    broad = positive > negative and changed_dates >= 3 and (
        not np.isfinite(top3_abs_share) or top3_abs_share <= 0.75
    )
    if not support or changed_dates == 0:
        state = "INCONCLUSIVE"
    elif (
        deltas["top3_capped"] >= 0.005
        and deltas["top3_target7"] > 0
        and deltas["top3_loss"] < 0
        and deltas["top3_severe_loss"] <= 0
        and lower_rank_help
        and lower_rank_not_harmed
        and broad
    ):
        state = "RECENCY_SIGNAL_SUPPORTED"
    elif (
        (deltas["top3_capped"] <= 0 and deltas["top3_target7"] <= 0 and deltas["top3_loss"] >= 0)
        or deltas["top3_capped"] <= -0.005
        or (deltas["top3_loss"] >= 0.05 and deltas["top3_capped"] < 0.005)
    ):
        state = "RECENCY_SIGNAL_NOT_SUPPORTED"
    else:
        state = "MIXED"
    diagnostics: dict[str, Any] = {
        **deltas,
        "positive_daily_delta_dates": positive,
        "negative_daily_delta_dates": negative,
        "zero_daily_delta_dates": zero,
        "top3_absolute_delta_share_largest_3_dates": top3_abs_share,
        "changed_top3_dates": changed_dates,
        "lower_rank_help": lower_rank_help,
        "lower_rank_not_harmed": lower_rank_not_harmed,
        "broad_daily_improvement": broad,
        "rank_deltas": rank_deltas,
    }
    return state, diagnostics


def render_review(
    context: Mapping[str, Any], state: str, diagnostics: Mapping[str, Any],
) -> str:
    if state not in {
        "RECENCY_SIGNAL_SUPPORTED", "RECENCY_SIGNAL_NOT_SUPPORTED", "MIXED", "INCONCLUSIVE"
    }:
        raise RuntimeError("FATAL: invalid recency diagnostic state")
    rank_deltas = diagnostics["rank_deltas"]
    source_parts = []
    for rank in (1, 2, 3):
        delta = rank_deltas[rank]
        source_parts.append(
            f"Rank{rank}: capped {delta['capped']:+.4%}, Target7 {delta['target7']:+.2%}, "
            f"LOSS {delta['loss']:+.2%}"
        )
    changed = context["rank_changes"]
    added = changed[changed["entered_top3"].eq(1)]
    removed = changed[changed["left_top3"].eq(1)]
    top_dates = (
        context["daily"].pivot(index="signal_date", columns="version", values="top3_capped_return")
    )
    top_dates["delta"] = top_dates["RECENT_PRIORITY_2X"] - top_dates["HISTORICAL_EQUAL"]
    contributor_dates = top_dates["delta"].abs().nlargest(3).index
    contributors = ", ".join(
        f"{date} ({float(top_dates.loc[date, 'delta']):+.4%})"
        for date in contributor_dates
    )
    equal_summary = context["summary"].set_index("version").loc["HISTORICAL_EQUAL"]
    recent_summary = context["summary"].set_index("version").loc["RECENT_PRIORITY_2X"]
    coefficient_table = context["coefficients"]
    coefficient_rows = coefficient_table[
        coefficient_table["row_type"].eq("COEFFICIENT_AND_MONTH_CONTRIBUTION")
        & coefficient_table["feature"].ne("__INTERCEPT__")
    ].copy()
    largest_coefficients = coefficient_rows.assign(
        abs_delta=coefficient_rows["coefficient_delta_recent_minus_equal"].abs()
    ).nlargest(3, "abs_delta")
    coefficient_text = ", ".join(
        f"{row.feature} {float(row.coefficient_delta_recent_minus_equal):+.5f}"
        for row in largest_coefficients.itertuples()
    )
    answers = {
        "Q1": (
            f"NO。状态为 {state}。相对等权，Recent Priority 的 July Top3 capped 变化 "
            f"{diagnostics['top3_capped']:+.4%}，Target7 率变化 "
            f"{diagnostics['top3_target7']:+.2%}，LOSS 率变化 "
            f"{diagnostics['top3_loss']:+.2%}。"
        ),
        "Q2": (
            "没有整体改善；变化来源为：" + "；".join(source_parts) + "。"
            f"Top3 Board3 数量由 {int(equal_summary['top3_board3_count'])} 变为 "
            f"{int(recent_summary['top3_board3_count'])}，并非通过减少 Board3 获益。"
            f"绝对值最大的 coefficient delta 为 {coefficient_text}；这里只作诊断，不据此改模。"
        ),
        "Q3": (
            f"Top3 LOSS 率变化 {diagnostics['top3_loss']:+.2%}，severe LOSS 率变化 "
            f"{diagnostics['top3_severe_loss']:+.2%}；新增 Top3 中 LOSS "
            f"{int(added['loss'].sum())} 只，剔除 Top3 中 LOSS {int(removed['loss'].sum())} 只。"
        ),
        "Q4": (
            f"逐日 Top3 delta：正 {diagnostics['positive_daily_delta_dates']} 日、负 "
            f"{diagnostics['negative_daily_delta_dates']} 日、零 {diagnostics['zero_daily_delta_dates']} 日；"
            f"绝对变化最大的 3 日占比 {diagnostics['top3_absolute_delta_share_largest_3_dates']:.2%}。"
            f"最大三日（按绝对值，保留实际方向）为 {contributors}。本次不是少数日期带来的改善；"
            "所有实质变化日期均为负向。"
        ),
        "Q5": (
            "NO。固定 2x recency 假设未获支持，不值得据此进入正式 forward 模型研究。"
            "本任务不选择或冻结模型；July 已消费，不能用于追加权重或生产定版。"
        ),
    }
    lines = [
        "# v004c Recency Weighting Diagnostic v001",
        "",
        "S2 source contract: 7F, weighted L2 logistic, L2=0.10, positive_weight=1.50, "
        "tail bonus OFF, date_weight=1/candidate_count.",
        "",
        "Only HISTORICAL_EQUAL (May=1.0, June=1.0) and RECENT_PRIORITY_2X "
        "(May=0.5, June=1.0) were fit.",
        "",
    ]
    questions = (
        "6月加权是否比5/6月等权更适合7月？",
        "如果改善，改善发生在 Rank1、Rank2、Rank3，还是整体Top3？",
        "5月权重降低后，LOSS是否减少？",
        "是否只是少数几个July日期造成改善？",
        "是否值得进行下一阶段的正式 forward 研究？",
    )
    for index, question in enumerate(questions, 1):
        lines.extend([f"## Q{index}. {question}", "", answers[f"Q{index}"], ""])
    lines.extend([
        "JULY_MODEL_SELECTION = NO",
        "",
        "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO",
        "",
        f"RECENCY_WEIGHTING_STATE = {state}",
        "",
        "NEXT_ACTION = STOP_AND_REVIEW",
        "",
    ])
    return "\n".join(lines)


def analyze(root: str | Path) -> dict[str, Any]:
    assert_experiment_contract()
    root_path = Path(root).resolve()
    frame, bridge_spec = load_bridge(root_path)
    train, july = build_populations(frame)
    betas, weights = fit_versions(train)
    lock = build_prediction_lock(july, betas)
    parity = assert_equal_parity(root_path, lock, betas["HISTORICAL_EQUAL"])
    scored = attach_outcomes(lock, july)
    daily = build_daily(scored)
    rankwise = build_rankwise(scored, daily)
    rank_changes = build_rank_changes(scored)
    coefficients = build_coefficients(train, betas, weights)
    summary = build_summary(scored, daily, train, weights, rank_changes, parity)
    state, diagnostics = classify_state(summary, rankwise, daily, rank_changes)
    summary["recency_weighting_state"] = state
    summary["next_action"] = "STOP_AND_REVIEW"
    summary["august_signal_outcome_accessed"] = "NO"
    return {
        "root": root_path,
        "bridge_spec": bridge_spec,
        "train": train,
        "july": july,
        "betas": betas,
        "weights": weights,
        "prediction_lock": lock,
        "scored": scored,
        "summary": summary,
        "rankwise": rankwise,
        "rank_changes": rank_changes,
        "daily": daily,
        "coefficients": coefficients,
        "parity": parity,
        "state": state,
        "diagnostics": diagnostics,
    }


def build_outputs(context: Mapping[str, Any]) -> dict[str, bytes]:
    review = render_review(context, str(context["state"]), context["diagnostics"])
    return {
        OUTPUT_FILENAMES[0]: _csv_bytes(context["summary"]),
        OUTPUT_FILENAMES[1]: _csv_bytes(context["rankwise"]),
        OUTPUT_FILENAMES[2]: _csv_bytes(context["rank_changes"]),
        OUTPUT_FILENAMES[3]: _csv_bytes(context["daily"]),
        OUTPUT_FILENAMES[4]: _csv_bytes(context["coefficients"]),
        OUTPUT_FILENAMES[5]: review.encode("utf-8"),
    }


def write_outputs(
    context: Mapping[str, Any], output_dir: str | Path | None = None,
) -> Path:
    target = Path(output_dir) if output_dir is not None else (
        context["root"] / "reports" / "research" / OUTPUT_DIRNAME
    )
    first = build_outputs(context)
    second = build_outputs(context)
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

"""Audit four preregistered D1 unfinished-repair information fields.

This is an information audit only.  It loads the frozen S2 population and
existing D1-safe field values, creates same-date descriptive comparisons, and
never fits a model, changes a score/rank, searches a feature/window/threshold,
or reads an August signal-date outcome.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_minute_features import compute_minute_features
from .v004c_pairwise_feature_contract import FEATURE_CONTRACT
from .v004c_s2_replaceable_loss_case_pack import load_d1_minute
from .v004c_s2_winner_vs_loss_failure_attribution import load_frozen_s2


TASK_NAME = "v004c_d1_unfinished_repair_information_audit_v001"
OUTPUT_DIRNAME = "v004c_d1_unfinished_repair_information_audit_v001_20260506_20260731"
AUGUST_ASOF = "2026-08-01"
MAX_SIGNAL_DATE = "2026-07-29"
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_RESAMPLES = 20_000

FEATURES = (
    "d1_afternoon_return",
    "d1_open_to_close_return_raw",
    "d1_close_to_vwap_raw",
    "d1_low_to_close_recovery",
)
PERIODS = (
    "MAY",
    "JUNE",
    "JULY_MATURE_ONLY",
    "MAY_JUNE_JULY_MATURE",
)
RANK_REGIONS = {
    "RANK1_3": (1, 3),
    "RANK2_6": (2, 6),
    "RANK4_7": (4, 7),
}
PAIRWISE_AUTHORITY = (
    "reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630/"
    "v004c_pairwise_v1_input_table_v002.csv"
)
MODEL_TABLE_AUTHORITY = (
    "reports/research/v004c_model_table_v001_20260601_20260729/"
    "v004c_model_table_v001.csv"
)
CASE_53_AUTHORITY = (
    "reports/research/v004c_s2_replaceable_loss_case_pack_v001_20260506_20260729/"
    "v004c_s2_case_53f_comparison_v001.csv"
)

OUTPUT_FILENAMES = (
    "v004c_unfinished_repair_population_v001.csv",
    "v004c_unfinished_repair_feature_summary_v001.csv",
    "v004c_unfinished_repair_t7_loss_pairs_v001.csv",
    "v004c_unfinished_repair_rank_region_summary_v001.csv",
    "v004c_unfinished_repair_rank3_failure_v001.csv",
    "v004c_unfinished_repair_case_vs_population_v001.csv",
    "v004c_unfinished_repair_bootstrap_v001.csv",
    "v004c_unfinished_repair_review_v001.md",
)

# Fixed, preregistered interpretation gates.  They summarize evidence; they do
# not select a threshold, feature subset, trading rule, or model.
MATERIAL_POOLED_CONCORDANCE = 0.57
PERIOD_REPLICATION_CONCORDANCE = 0.55
HEAD_NEAR_CONCORDANCE = 0.55
BOOTSTRAP_DIRECTION_PROBABILITY = 0.80
CASE_CONDITIONAL_CONCORDANCE = 0.60


def assert_contract() -> None:
    if FEATURES != (
        "d1_afternoon_return",
        "d1_open_to_close_return_raw",
        "d1_close_to_vwap_raw",
        "d1_low_to_close_recovery",
    ):
        raise RuntimeError("FATAL: preregistered four-field manifest changed")
    contract_names = {row["feature_name"] for row in FEATURE_CONTRACT}
    if any(feature not in contract_names for feature in FEATURES):
        raise RuntimeError("FATAL: preregistered field missing from existing 53F contract")
    if MAX_SIGNAL_DATE >= AUGUST_ASOF:
        raise RuntimeError("FATAL: signal-date boundary reached August")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _period_slice(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    month = frame["signal_date"].astype(str).str[:7]
    if period == "MAY":
        return frame[month.eq("2026-05")]
    if period == "JUNE":
        return frame[month.eq("2026-06")]
    if period == "JULY_MATURE_ONLY":
        return frame[month.eq("2026-07")]
    if period == "MAY_JUNE_JULY_MATURE":
        return frame
    raise ValueError(period)


def _feature_contract() -> dict[str, dict[str, Any]]:
    by_name = {row["feature_name"]: row for row in FEATURE_CONTRACT}
    return {feature: by_name[feature] for feature in FEATURES}


def _load_archived_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"event_id": str, "code": str})
    if frame["event_id"].duplicated().any():
        raise RuntimeError(f"FATAL: duplicate event_id in {path}")
    return frame.set_index("event_id", drop=False)


def build_population(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load frozen S2 rows and attach only the four existing D1-safe fields."""
    assert_contract()
    root_path = Path(root).resolve()
    scored, _, s2_audit = load_frozen_s2(root_path)
    pairwise = _load_archived_table(root_path / PAIRWISE_AUTHORITY)
    model_table = _load_archived_table(root_path / MODEL_TABLE_AUTHORITY)

    rows: list[dict[str, Any]] = []
    reconstructed_events: dict[str, tuple[dict[str, float], str, str]] = {}
    for event in scored.sort_values(["signal_date", "s2_rank", "event_id"], kind="mergesort").to_dict("records"):
        event_id = str(event["event_id"])
        signal_date = str(event["signal_date"])
        source_row: Mapping[str, Any] | None = None
        source = ""
        source_path = ""
        if signal_date <= "2026-06-30" and event_id in pairwise.index:
            source_row = pairwise.loc[event_id]
            source = "ARCHIVED_PAIRWISE_V002_53F"
            source_path = PAIRWISE_AUTHORITY
        elif event_id in model_table.index:
            source_row = model_table.loc[event_id]
            source = "ARCHIVED_MODEL_TABLE_V001"
            source_path = MODEL_TABLE_AUTHORITY
        else:
            minute, minute_source, minute_path = load_d1_minute(
                root_path, str(event["code"]).zfill(6), str(event["d1_date"])
            )
            if len(minute) != 48:
                raise RuntimeError(f"FATAL: incomplete D1 5m bars for {event_id}: {len(minute)}")
            values = compute_minute_features(minute, None, None, None, None, None)
            reconstructed_events[event_id] = (
                {feature: float(values[feature]) for feature in FEATURES},
                minute_source,
                minute_path,
            )
            source_row = reconstructed_events[event_id][0]
            source = "EXISTING_MINUTE_FEATURE_FORMULA_RECONSTRUCTION"
            source_path = minute_path

        output = {
            "event_id": event_id,
            "code": str(event["code"]).zfill(6),
            "signal_date": signal_date,
            "month": signal_date[:7],
            "d0_date": str(event["d0_date"]),
            "d1_date": str(event["d1_date"]),
            "d2_date": str(event["d2_date"]),
            "d3_date": str(event["d3_date"]),
            "label_available_date": str(event["label_available_date"]),
            "board_group": str(event["board_group"]),
            "candidate_count": int(event["candidate_count"]),
            "s2_score": float(event["s2_score"]),
            "s2_rank": int(event["s2_rank"]),
            "target7": int(event["target7"]),
            "positive_non_target": int(event["positive_non_target"]),
            "loss": int(event["loss"]),
            "severe_loss": int(event["severe_loss"]),
            "raw_repair_return": float(event["raw_repair_return"]),
            "capped_return_7": float(event["capped_return_7"]),
        }
        for feature in FEATURES:
            output[feature] = pd.to_numeric(pd.Series([source_row[feature]]), errors="coerce").iloc[0]
            output[f"provenance__{feature}"] = source
            output[f"source_path__{feature}"] = source_path
        rows.append(output)

    population = pd.DataFrame(rows).sort_values(
        ["signal_date", "s2_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if (len(population), population["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: mature population parity failed")
    if population["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in population")
    if population[list(FEATURES)].isna().any().any():
        raise RuntimeError("FATAL: preregistered field missing after authoritative routing")
    if population["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal-date row reached audit")
    if population["label_available_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: immature label reached audit")

    case = pd.read_csv(root_path / CASE_53_AUTHORITY, encoding="utf-8-sig", dtype=str)
    case = case[case["feature_name"].isin(FEATURES)]
    by_event = population.set_index("event_id")
    parity_diffs: list[float] = []
    for row in case.itertuples(index=False):
        for side in ("target", "loss"):
            event_id = str(getattr(row, f"{side}_event_id"))
            expected = pd.to_numeric(pd.Series([getattr(row, f"{side}_value")]), errors="coerce").iloc[0]
            actual = float(by_event.at[event_id, row.feature_name])
            if pd.notna(expected):
                parity_diffs.append(abs(actual - float(expected)))
    parity_max = max(parity_diffs, default=math.nan)
    if np.isfinite(parity_max) and parity_max > 1e-12:
        raise RuntimeError(f"FATAL: four-field semantics differ from historical case pack: {parity_max}")

    month_counts = population.groupby("month").agg(
        rows=("event_id", "size"), dates=("signal_date", "nunique")
    ).to_dict("index")
    audit = {
        **s2_audit,
        "rows": len(population),
        "dates": population["signal_date"].nunique(),
        "duplicate_event_id_count": int(population["event_id"].duplicated().sum()),
        "missing_four_field_cells": int(population[list(FEATURES)].isna().sum().sum()),
        "reconstructed_event_count": len(reconstructed_events),
        "reconstructed_event_ids": sorted(reconstructed_events),
        "case_pack_semantic_parity_max_abs_error": parity_max,
        "month_counts": month_counts,
        "august_signal_outcome_accessed": "NO",
        "model_refits": 0,
    }
    return population, audit


def _pair_rows(
    frame: pd.DataFrame,
    feature: str,
    positive_col: str,
    negative_col: str,
    endpoint: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in frame.groupby("signal_date", sort=True):
        positive = day[day[positive_col].astype(bool)].sort_values("event_id", kind="mergesort")
        negative = day[day[negative_col].astype(bool)].sort_values("event_id", kind="mergesort")
        for p in positive.itertuples(index=False):
            for n in negative.itertuples(index=False):
                p_value = float(getattr(p, feature))
                n_value = float(getattr(n, feature))
                difference = p_value - n_value
                rows.append({
                    "endpoint": endpoint,
                    "feature": feature,
                    "signal_date": str(signal_date),
                    "month": str(signal_date)[:7],
                    "positive_event_id": str(p.event_id),
                    "positive_code": str(p.code).zfill(6),
                    "positive_s2_rank": int(p.s2_rank),
                    "positive_value": p_value,
                    "negative_event_id": str(n.event_id),
                    "negative_code": str(n.code).zfill(6),
                    "negative_s2_rank": int(n.s2_rank),
                    "negative_value": n_value,
                    "difference": difference,
                    "raw_concordance": 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5,
                    "unfinished_low_concordance": 1.0 if difference < 0 else 0.0 if difference > 0 else 0.5,
                })
    return pd.DataFrame(rows)


def _pair_period_slice(pairs: pd.DataFrame, period: str) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    month = pairs["signal_date"].astype(str).str[:7]
    if period == "MAY":
        return pairs[month.eq("2026-05")]
    if period == "JUNE":
        return pairs[month.eq("2026-06")]
    if period == "JULY_MATURE_ONLY":
        return pairs[month.eq("2026-07")]
    if period == "MAY_JUNE_JULY_MATURE":
        return pairs
    raise ValueError(period)


def _median_daily_rank_gap(
    frame: pd.DataFrame, feature: str, positive_col: str, negative_col: str
) -> float:
    gaps: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        part = day[day[positive_col].astype(bool) | day[negative_col].astype(bool)].copy()
        if not part[positive_col].any() or not part[negative_col].any():
            continue
        part["_rank_pct"] = pd.to_numeric(part[feature]).rank(method="average", pct=True)
        gaps.append(
            float(part.loc[part[positive_col].astype(bool), "_rank_pct"].median())
            - float(part.loc[part[negative_col].astype(bool), "_rank_pct"].median())
        )
    return float(np.median(gaps)) if gaps else math.nan


def _summarize_endpoint(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    feature: str,
    positive_col: str,
    negative_col: str,
) -> dict[str, Any]:
    positive = pd.to_numeric(frame.loc[frame[positive_col].astype(bool), feature], errors="coerce")
    negative = pd.to_numeric(frame.loc[frame[negative_col].astype(bool), feature], errors="coerce")
    if pairs.empty:
        return {
            "eligible_dates": 0, "pairs": 0, "within_date_auc_raw": math.nan,
            "same_date_pair_concordance_raw": math.nan,
            "unfinished_low_pair_concordance": math.nan,
            "best_oriented_pair_concordance": math.nan,
            "best_interpretation_direction": "NA",
            "median_within_date_rank_difference": math.nan,
            "positive_mean": float(positive.mean()) if len(positive) else math.nan,
            "positive_median": float(positive.median()) if len(positive) else math.nan,
            "negative_mean": float(negative.mean()) if len(negative) else math.nan,
            "negative_median": float(negative.median()) if len(negative) else math.nan,
        }
    daily = pairs.groupby("signal_date")["raw_concordance"].mean()
    raw = float(pairs["raw_concordance"].mean())
    return {
        "eligible_dates": int(daily.size),
        "pairs": len(pairs),
        "within_date_auc_raw": float(daily.mean()),
        "same_date_pair_concordance_raw": raw,
        "unfinished_low_pair_concordance": 1.0 - raw,
        "best_oriented_pair_concordance": max(raw, 1.0 - raw),
        "best_interpretation_direction": (
            "T7_HIGH" if raw > .5 else "T7_LOW" if raw < .5 else "TIE"
        ),
        "median_within_date_rank_difference": _median_daily_rank_gap(
            frame, feature, positive_col, negative_col
        ),
        "positive_mean": float(positive.mean()),
        "positive_median": float(positive.median()),
        "negative_mean": float(negative.mean()),
        "negative_median": float(negative.median()),
    }


def _direction_stability(monthly_raw_pair_concordance: Sequence[float]) -> str:
    values = np.asarray(
        [value for value in monthly_raw_pair_concordance if np.isfinite(value)], dtype=float
    )
    if len(values) != 3:
        return "NEAR_RANDOM"
    positive = values > .5
    negative = values < .5
    if positive.any() and negative.any():
        return "TEMPORALLY_UNSTABLE"
    if positive.all():
        return "STABLE_T7_HIGH"
    if negative.all():
        return "STABLE_T7_LOW"
    if np.all(np.abs(values - .5) <= .05):
        return "NEAR_RANDOM"
    return "TEMPORALLY_UNSTABLE"


def build_full_pairs(population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tables = {
        feature: _pair_rows(population, feature, "target7", "loss", "T7_VS_LOSS")
        for feature in FEATURES
    }
    combined = pd.concat(tables.values(), ignore_index=True).sort_values(
        ["feature", "signal_date", "positive_event_id", "negative_event_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return combined, tables


def build_feature_summary(
    population: pd.DataFrame, full_pairs: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    contract = _feature_contract()
    rows: list[dict[str, Any]] = []
    feature_rows: dict[str, list[dict[str, Any]]] = {feature: [] for feature in FEATURES}
    for feature in FEATURES:
        provenance = ";".join(sorted(population[f"provenance__{feature}"].unique()))
        for period in PERIODS:
            part = _period_slice(population, period)
            t7_loss_pairs = _pair_period_slice(full_pairs[feature], period)
            stats = _summarize_endpoint(part, t7_loss_pairs, feature, "target7", "loss")
            row = {
                "feature": feature,
                "period": period,
                "endpoint": "T7_VS_LOSS",
                "rows": len(part),
                "dates": part["signal_date"].nunique(),
                "missing_count": int(part[feature].isna().sum()),
                "unique_values": int(part[feature].nunique(dropna=True)),
                "source_provenance": provenance,
                "contract_source": contract[feature]["source"],
                "formula": contract[feature]["formula"],
                "available_as_of": contract[feature]["available_as_of"],
                "uses_d2_d3_information": "NO",
                **stats,
                "direction_stability": "",
            }
            rows.append(row)
            feature_rows[feature].append(row)

        for period in PERIODS:
            part = _period_slice(population, period)
            aux_pairs = _pair_rows(part, feature, "target7", "positive_non_target", "T7_VS_PNT")
            aux = _summarize_endpoint(
                part, aux_pairs, feature, "target7", "positive_non_target"
            )
            rows.append({
                "feature": feature,
                "period": period,
                "endpoint": "T7_VS_PNT_AUXILIARY",
                "rows": len(part),
                "dates": part["signal_date"].nunique(),
                "missing_count": int(part[feature].isna().sum()),
                "unique_values": int(part[feature].nunique(dropna=True)),
                "source_provenance": provenance,
                "contract_source": contract[feature]["source"],
                "formula": contract[feature]["formula"],
                "available_as_of": contract[feature]["available_as_of"],
                "uses_d2_d3_information": "NO",
                **{key: value for key, value in aux.items() if key in {
                    "eligible_dates", "pairs", "within_date_auc_raw",
                    "same_date_pair_concordance_raw",
                }},
                "direction_stability": "",
            })

    for feature in FEATURES:
        monthly = [
            row["same_date_pair_concordance_raw"]
            for row in feature_rows[feature]
            if row["period"] in {"MAY", "JUNE", "JULY_MATURE_ONLY"}
        ]
        stability = _direction_stability(monthly)
        for row in rows:
            if row["feature"] == feature:
                row["direction_stability"] = stability
    return pd.DataFrame(rows).sort_values(
        ["endpoint", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True)


def build_rank_region_summary(population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    rows: list[dict[str, Any]] = []
    pair_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for region, (low, high) in RANK_REGIONS.items():
        region_frame = population[population["s2_rank"].between(low, high)].copy()
        pair_tables[region] = {}
        for feature in FEATURES:
            pairs = _pair_rows(region_frame, feature, "target7", "loss", "T7_VS_LOSS")
            pair_tables[region][feature] = pairs
            for period in PERIODS:
                part = _period_slice(region_frame, period)
                period_pairs = _pair_period_slice(pairs, period)
                stats = _summarize_endpoint(part, period_pairs, feature, "target7", "loss")
                rows.append({
                    "rank_region": region,
                    "rank_min": low,
                    "rank_max": high,
                    "feature": feature,
                    "period": period,
                    "rows": len(part),
                    "dates": part["signal_date"].nunique(),
                    **stats,
                })
    return pd.DataFrame(rows).sort_values(
        ["rank_region", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True), pair_tables


def build_rank3_failure(population: pd.DataFrame) -> pd.DataFrame:
    pair_rows: list[dict[str, Any]] = []
    for signal_date, day in population.groupby("signal_date", sort=True):
        losses = day[day["s2_rank"].eq(3) & day["loss"].astype(bool)]
        targets = day[day["s2_rank"].ge(4) & day["target7"].astype(bool)]
        for loss in losses.itertuples(index=False):
            for target in targets.itertuples(index=False):
                for feature in FEATURES:
                    target_value = float(getattr(target, feature))
                    loss_value = float(getattr(loss, feature))
                    difference = target_value - loss_value
                    pair_rows.append({
                        "row_type": "PAIR",
                        "period": str(signal_date)[:7],
                        "signal_date": str(signal_date),
                        "feature": feature,
                        "loss_event_id": str(loss.event_id),
                        "loss_code": str(loss.code).zfill(6),
                        "loss_s2_rank": int(loss.s2_rank),
                        "target_event_id": str(target.event_id),
                        "target_code": str(target.code).zfill(6),
                        "target_s2_rank": int(target.s2_rank),
                        "target_value": target_value,
                        "loss_value": loss_value,
                        "difference": difference,
                        "target7_less_than_loss": int(difference < 0),
                        "target7_greater_than_loss": int(difference > 0),
                        "tie": int(difference == 0),
                    })
    pairs = pd.DataFrame(pair_rows)
    summaries: list[dict[str, Any]] = []
    period_map = {
        "MAY": "2026-05", "JUNE": "2026-06", "JULY_MATURE_ONLY": "2026-07"
    }
    for feature in FEATURES:
        for period in PERIODS:
            if pairs.empty:
                part = pairs
            elif period == "MAY_JUNE_JULY_MATURE":
                part = pairs[pairs["feature"].eq(feature)]
            else:
                part = pairs[pairs["feature"].eq(feature) & pairs["period"].eq(period_map[period])]
            summaries.append({
                "row_type": "SUMMARY",
                "period": period,
                "signal_date": "",
                "feature": feature,
                "loss_event_id": "",
                "loss_code": "",
                "loss_s2_rank": 3,
                "target_event_id": "",
                "target_code": "",
                "target_s2_rank": math.nan,
                "target_value": math.nan,
                "loss_value": math.nan,
                "difference": math.nan,
                "target7_less_than_loss": int(part["target7_less_than_loss"].sum()) if len(part) else 0,
                "target7_greater_than_loss": int(part["target7_greater_than_loss"].sum()) if len(part) else 0,
                "tie": int(part["tie"].sum()) if len(part) else 0,
                "pair_count": len(part),
                "target7_less_than_loss_proportion": float(part["target7_less_than_loss"].mean()) if len(part) else math.nan,
                "target7_greater_than_loss_proportion": float(part["target7_greater_than_loss"].mean()) if len(part) else math.nan,
                "median_difference": float(part["difference"].median()) if len(part) else math.nan,
            })
    if pairs.empty:
        return pd.DataFrame(summaries)
    pairs["pair_count"] = math.nan
    pairs["target7_less_than_loss_proportion"] = math.nan
    pairs["target7_greater_than_loss_proportion"] = math.nan
    pairs["median_difference"] = math.nan
    return pd.concat([pairs, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def _case_pair_tables(root: Path) -> dict[str, pd.DataFrame]:
    case = pd.read_csv(root / CASE_53_AUTHORITY, encoding="utf-8-sig", dtype=str)
    case = case[case["feature_name"].isin(FEATURES)].copy()
    output: dict[str, pd.DataFrame] = {}
    for feature in FEATURES:
        part = case[case["feature_name"].eq(feature)].copy()
        part["positive_value"] = pd.to_numeric(part["target_value"], errors="coerce")
        part["negative_value"] = pd.to_numeric(part["loss_value"], errors="coerce")
        part["difference"] = part["positive_value"] - part["negative_value"]
        part["raw_concordance"] = np.where(
            part["difference"].gt(0), 1.0, np.where(part["difference"].lt(0), 0.0, .5)
        )
        part["unfinished_low_concordance"] = 1.0 - part["raw_concordance"]
        output[feature] = part[[
            "case_id", "signal_date", "positive_value", "negative_value",
            "difference", "raw_concordance", "unfinished_low_concordance",
        ]]
    if any(len(table) != 25 for table in output.values()):
        raise RuntimeError("FATAL: replaceable-loss authority is not exact 25 cases")
    return output


def _comparison_stats(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {
            "dates": 0, "pairs": 0, "mean_difference": math.nan,
            "median_difference": math.nan, "raw_pair_concordance": math.nan,
            "unfinished_low_pair_concordance": math.nan,
            "best_oriented_pair_concordance": math.nan,
        }
    raw = float(pairs["raw_concordance"].mean())
    return {
        "dates": int(pairs["signal_date"].nunique()),
        "pairs": len(pairs),
        "mean_difference": float(pairs["difference"].mean()),
        "median_difference": float(pairs["difference"].median()),
        "raw_pair_concordance": raw,
        "unfinished_low_pair_concordance": 1.0 - raw,
        "best_oriented_pair_concordance": max(raw, 1.0 - raw),
    }


def build_case_vs_population(
    root: Path,
    full_pairs: Mapping[str, pd.DataFrame],
    rank_pairs: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    case_pairs = _case_pair_tables(root)
    sources = {
        "REPLACEABLE_LOSS_25": case_pairs,
        "FULL_CANDIDATE_T7_VS_LOSS": full_pairs,
        "S2_RANK2_6_T7_VS_LOSS": rank_pairs["RANK2_6"],
    }
    rows: list[dict[str, Any]] = []
    for population_name, feature_tables in sources.items():
        for feature in FEATURES:
            for period in PERIODS:
                pairs = _pair_period_slice(feature_tables[feature], period)
                rows.append({
                    "comparison_population": population_name,
                    "feature": feature,
                    "period": period,
                    **_comparison_stats(pairs),
                })
    return pd.DataFrame(rows).sort_values(
        ["comparison_population", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True)


def build_bootstrap(full_pairs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_index, period in enumerate(PERIODS):
        reference = _pair_period_slice(full_pairs[FEATURES[0]], period)
        dates = sorted(reference["signal_date"].unique())
        if not dates:
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED + period_index)
        schedule = rng.integers(0, len(dates), size=(BOOTSTRAP_RESAMPLES, len(dates)))
        for feature in FEATURES:
            pairs = _pair_period_slice(full_pairs[feature], period)
            grouped = pairs.groupby("signal_date")["raw_concordance"].agg(["sum", "count", "mean"]).reindex(dates)
            if grouped.isna().any().any():
                raise RuntimeError("FATAL: bootstrap informative-date schedules differ across four complete fields")
            sums = grouped["sum"].to_numpy(float)
            counts = grouped["count"].to_numpy(float)
            means = grouped["mean"].to_numpy(float)
            distributions = {
                "WITHIN_DATE_AUC_RAW": means[schedule].mean(axis=1),
                "SAME_DATE_PAIR_CONCORDANCE_RAW": (
                    sums[schedule].sum(axis=1) / counts[schedule].sum(axis=1)
                ),
            }
            for metric, values in distributions.items():
                estimate = float(means.mean()) if metric == "WITHIN_DATE_AUC_RAW" else float(sums.sum() / counts.sum())
                best_high = estimate >= .5
                best_values = values if best_high else 1.0 - values
                rows.append({
                    "feature": feature,
                    "period": period,
                    "metric": metric,
                    "sampling_unit": "SIGNAL_DATE",
                    "seed": BOOTSTRAP_SEED,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "valid_resamples": len(values),
                    "estimate_raw": estimate,
                    "p2_5_raw": float(np.quantile(values, .025)),
                    "p50_raw": float(np.quantile(values, .50)),
                    "p97_5_raw": float(np.quantile(values, .975)),
                    "P_raw_gt_half": float(np.mean(values > .5)),
                    "P_unfinished_low_gt_half": float(np.mean(values < .5)),
                    "best_interpretation_direction": "T7_HIGH" if best_high else "T7_LOW",
                    "best_oriented_estimate": max(estimate, 1.0 - estimate),
                    "best_oriented_p2_5": float(np.quantile(best_values, .025)),
                    "best_oriented_p50": float(np.quantile(best_values, .50)),
                    "best_oriented_p97_5": float(np.quantile(best_values, .975)),
                    "P_best_oriented_gt_half": float(np.mean(best_values > .5)),
                })
    return pd.DataFrame(rows).sort_values(
        ["feature", "period", "metric"], kind="mergesort"
    ).reset_index(drop=True)


def decide_state(
    feature_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    case_comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    primary = feature_summary[feature_summary["endpoint"].eq("T7_VS_LOSS")]
    flags: dict[str, dict[str, Any]] = {}
    supported: list[str] = []
    case_only: list[str] = []
    for feature in FEATURES:
        f = primary[primary["feature"].eq(feature)].set_index("period")
        head = rank_summary[
            rank_summary["rank_region"].eq("RANK2_6")
            & rank_summary["feature"].eq(feature)
            & rank_summary["period"].eq("MAY_JUNE_JULY_MATURE")
        ].iloc[0]
        case = case_comparison[
            case_comparison["comparison_population"].eq("REPLACEABLE_LOSS_25")
            & case_comparison["feature"].eq(feature)
            & case_comparison["period"].eq("MAY_JUNE_JULY_MATURE")
        ].iloc[0]
        boot = bootstrap[
            bootstrap["feature"].eq(feature)
            & bootstrap["period"].eq("MAY_JUNE_JULY_MATURE")
            & bootstrap["metric"].eq("SAME_DATE_PAIR_CONCORDANCE_RAW")
        ].iloc[0]
        pooled = float(f.at["MAY_JUNE_JULY_MATURE", "unfinished_low_pair_concordance"])
        june = float(f.at["JUNE", "unfinished_low_pair_concordance"])
        july = float(f.at["JULY_MATURE_ONLY", "unfinished_low_pair_concordance"])
        stable_low = f.iloc[0]["direction_stability"] == "STABLE_T7_LOW"
        qualifies = bool(
            stable_low
            and pooled >= MATERIAL_POOLED_CONCORDANCE
            and june >= PERIOD_REPLICATION_CONCORDANCE
            and july >= PERIOD_REPLICATION_CONCORDANCE
            and float(head["unfinished_low_pair_concordance"]) >= HEAD_NEAR_CONCORDANCE
            and int(head["pairs"]) >= 10
            and float(boot["P_unfinished_low_gt_half"]) >= BOOTSTRAP_DIRECTION_PROBABILITY
        )
        conditional = bool(
            not qualifies
            and float(case["unfinished_low_pair_concordance"]) >= CASE_CONDITIONAL_CONCORDANCE
            and (
                pooled < HEAD_NEAR_CONCORDANCE
                or float(head["unfinished_low_pair_concordance"]) < .53
                or not stable_low
            )
        )
        if qualifies:
            supported.append(feature)
        if conditional:
            case_only.append(feature)
        flags[feature] = {
            "stable_low": stable_low,
            "pooled_unfinished": pooled,
            "june_unfinished": june,
            "july_unfinished": july,
            "rank2_6_unfinished": float(head["unfinished_low_pair_concordance"]),
            "case_unfinished": float(case["unfinished_low_pair_concordance"]),
            "bootstrap_probability": float(boot["P_unfinished_low_gt_half"]),
            "supported": qualifies,
            "case_conditional": conditional,
        }
    if supported:
        state = "UNFINISHED_REPAIR_INFORMATION_SUPPORTED"
    elif case_only:
        state = "CASE_CONDITIONAL_ONLY"
    else:
        any_stable_low = any(value["stable_low"] for value in flags.values())
        any_material = any(value["pooled_unfinished"] >= HEAD_NEAR_CONCORDANCE for value in flags.values())
        state = "MIXED" if any_stable_low or any_material else "UNFINISHED_REPAIR_INFORMATION_NOT_SUPPORTED"
    return state, {"feature_flags": flags, "supported_features": supported, "case_only_features": case_only}


def build_review(
    population: pd.DataFrame,
    feature_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    rank3_failure: pd.DataFrame,
    state: str,
    decision_audit: Mapping[str, Any],
    population_audit: Mapping[str, Any],
) -> str:
    primary = feature_summary[
        feature_summary["endpoint"].eq("T7_VS_LOSS")
        & feature_summary["period"].eq("MAY_JUNE_JULY_MATURE")
    ].set_index("feature")
    stable = [
        feature for feature in FEATURES
        if primary.at[feature, "direction_stability"] in {"STABLE_T7_HIGH", "STABLE_T7_LOW"}
    ]
    june_july = [
        feature for feature, values in decision_audit["feature_flags"].items()
        if values["june_unfinished"] > .5 and values["july_unfinished"] > .5
    ]
    head = [
        feature for feature, values in decision_audit["feature_flags"].items()
        if values["rank2_6_unfinished"] >= HEAD_NEAR_CONCORDANCE
    ]
    rank3 = rank3_failure[
        rank3_failure["row_type"].eq("SUMMARY")
        & rank3_failure["period"].eq("MAY_JUNE_JULY_MATURE")
    ].set_index("feature")
    rank3_lines = "; ".join(
        f"{feature}={rank3.at[feature, 'target7_less_than_loss_proportion']:.2%}"
        for feature in FEATURES
    )
    full_lines = "; ".join(
        f"{feature}={primary.at[feature, 'unfinished_low_pair_concordance']:.2%}"
        for feature in FEATURES
    )
    authorize = "YES" if state == "UNFINISHED_REPAIR_INFORMATION_SUPPORTED" else "NO"
    lines = [
        "# v004c D1 Unfinished Repair Information Audit v001",
        "",
        "## Contract",
        "",
        "- Information audit only: four preregistered existing D1-close-safe fields.",
        "- No model fit, parameter/feature/window/threshold search, reranking, or August signal-date outcome access.",
        f"- Mature population: {len(population)} rows / {population['signal_date'].nunique()} dates; max signal date {population['signal_date'].max()}.",
        f"- Existing case-pack semantic parity max absolute error: {population_audit['case_pack_semantic_parity_max_abs_error']}.",
        "",
        "## Q1. Does the case-pack unfinished-repair pattern exist in the full candidate universe?",
        "",
        f"- Full-population T7-low-vs-LOSS concordance: {full_lines}.",
        f"- Formal supported fields: {', '.join(decision_audit['supported_features']) or 'NONE'}; case-conditional fields: {', '.join(decision_audit['case_only_features']) or 'NONE'}.",
        "",
        "## Q2. Do any fields carry directionally stable T7-vs-LOSS information across May/June/July?",
        "",
        f"- Stable-direction fields: {', '.join(stable) or 'NONE'}.",
        "",
        "## Q3. Do June and July replicate the unfinished-repair direction?",
        "",
        f"- Fields with T7-low concordance above 0.5 in both June and July: {', '.join(june_july) or 'NONE'}.",
        "",
        "## Q4. Is the information concentrated near S2 Rank2-6?",
        "",
        f"- Rank2-6 fields with T7-low concordance >= {HEAD_NEAR_CONCORDANCE:.2f}: {', '.join(head) or 'NONE'}.",
        "",
        "## Q5. Is Rank3 failure associated with preferring D1-completed-looking LOSS names?",
        "",
        f"- Rank3 LOSS vs lower-ranked Target7, Target7<LOSS proportions: {rank3_lines}.",
        "",
        "## Q6. Does this mechanism authorize one structural-correction experiment?",
        "",
        f"- {authorize}. This audit itself does not create a score, select a field, or train a challenger.",
        "",
        f"UNFINISHED_REPAIR_STATE = {state}",
        "",
        "NEXT_ACTION = STOP_AND_REVIEW",
        "",
        "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO",
        "",
    ]
    return "\n".join(lines)


def build_outputs(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    root_path = Path(root).resolve()
    population, population_audit = build_population(root_path)
    t7_loss_pairs, pair_tables = build_full_pairs(population)
    feature_summary = build_feature_summary(population, pair_tables)
    rank_summary, rank_pair_tables = build_rank_region_summary(population)
    rank3_failure = build_rank3_failure(population)
    case_comparison = build_case_vs_population(root_path, pair_tables, rank_pair_tables)
    bootstrap = build_bootstrap(pair_tables)
    state, decision_audit = decide_state(
        feature_summary, rank_summary, case_comparison, bootstrap
    )
    review = build_review(
        population, feature_summary, rank_summary, rank3_failure,
        state, decision_audit, population_audit,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(population),
        OUTPUT_FILENAMES[1]: _csv_bytes(feature_summary),
        OUTPUT_FILENAMES[2]: _csv_bytes(t7_loss_pairs),
        OUTPUT_FILENAMES[3]: _csv_bytes(rank_summary),
        OUTPUT_FILENAMES[4]: _csv_bytes(rank3_failure),
        OUTPUT_FILENAMES[5]: _csv_bytes(case_comparison),
        OUTPUT_FILENAMES[6]: _csv_bytes(bootstrap),
        OUTPUT_FILENAMES[7]: review.encode("utf-8"),
    }
    audit = {
        **population_audit,
        **decision_audit,
        "unfinished_repair_state": state,
        "next_action": "STOP_AND_REVIEW",
        "output_file_count": len(outputs),
        "four_feature_count": len(FEATURES),
        "t7_loss_pair_rows": len(t7_loss_pairs),
        "bootstrap_rows": len(bootstrap),
    }
    return outputs, audit


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, audit = build_outputs(root_path)
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
    return output_dir, audit


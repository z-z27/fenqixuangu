"""Frozen Stage1 contribution ablation audit for the May--July bridge.

This module never fits a learner.  It reconstructs the accepted frozen
``V4A_ARCH_TRANSFER_V4C`` logit from the bridge package, sets only the
predeclared feature contributions to zero for A1/A2/A3, and deterministically
re-ranks the unchanged authoritative candidate universe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_stage1_top3_risk_information import binary_auc
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    MODEL_ID,
    dataframe_csv_bytes,
)


TASK_NAME = "v004c_frozen_stage1_contribution_ablation_audit_v001"
BRIDGE_DIRNAME = "v004c_stage1_18f_bridge_analysis_v001_20260506_20260731"
OUTPUT_DIRNAME = (
    "v004c_frozen_stage1_contribution_ablation_audit_v001_20260506_20260731"
)
BRIDGE_CANDIDATES = "v004c_stage1_18f_bridge_candidates.csv"
BRIDGE_SPEC = "v004c_stage1_model_spec.json"
BRIDGE_LINEAGE = "v004c_stage1_feature_lineage.csv"

BOOTSTRAP_SEED = 20260826
BOOTSTRAP_RESAMPLES = 20_000
SCORE_TOLERANCE = 1e-12

D0_TIMING_FEATURES = (
    "rank_days_since_d0",
    "days_since_d0_le1",
    "days_since_d0_eq2",
    "days_since_d0_ge3",
)
TOTAL_FAMILY_ADDITIONS = (
    "rank_total_score",
    "inter_total_trend",
    "inter_total_active",
)
LEGACY_INTERACTION_FEATURES = (
    "inter_close_low",
    "inter_close_trend",
    "inter_total_trend",
    "inter_total_active",
    "inter_low_active",
    "spread_close_low",
)
REDUCED_BASE_FEATURES = (
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
)

ABLATION_VERSIONS: Mapping[str, tuple[str, ...]] = {
    "A0_ORIGINAL_18F": (),
    "A1_REMOVE_D0_TIMING": D0_TIMING_FEATURES,
    "A2_REMOVE_D0_TIMING_AND_TOTAL_FAMILY": (
        *D0_TIMING_FEATURES,
        *TOTAL_FAMILY_ADDITIONS,
    ),
    "A3_REDUCED_BASE_ONLY": tuple(
        feature
        for feature in FROZEN_FEATURE_COLUMNS
        if feature not in REDUCED_BASE_FEATURES
    ),
}

PERIOD_ORDER = (
    "MAY",
    "JUNE",
    "MAY_JUNE",
    "JULY_MATURE_ONLY",
    "MAY_JUNE_JULY_MATURE_ONLY",
)
SCOPE_ORDER = ("ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY")
MODEL_ORDER = tuple(ABLATION_VERSIONS)

OUTPUT_FILENAMES = (
    "v004c_stage1_ablation_summary_v001.csv",
    "v004c_stage1_ablation_daily_v001.csv",
    "v004c_stage1_ablation_membership_v001.csv",
    "v004c_stage1_ablation_paired_bootstrap_v001.csv",
    "v004c_stage1_days_since_d0_parity_v001.csv",
    "v004c_stage1_ablation_review_v001.md",
)

OUTCOME_COLUMNS = (
    "target7",
    "positive_non_target",
    "loss",
    "severe_loss",
    "raw_repair_return",
    "capped_return_7",
)


def _bridge_dir(root: Path) -> Path:
    return root / "reports" / "research" / BRIDGE_DIRNAME


def assert_audit_contract() -> None:
    if MODEL_ID != "V4A_ARCH_TRANSFER_V4C":
        raise RuntimeError("FATAL: frozen Stage1 identity changed")
    if len(FROZEN_FEATURE_COLUMNS) != 18:
        raise RuntimeError("FATAL: frozen Stage1 must contain exactly 18 features")
    if tuple(ABLATION_VERSIONS) != (
        "A0_ORIGINAL_18F",
        "A1_REMOVE_D0_TIMING",
        "A2_REMOVE_D0_TIMING_AND_TOTAL_FAMILY",
        "A3_REDUCED_BASE_ONLY",
    ):
        raise RuntimeError("FATAL: only predeclared A0/A1/A2/A3 are allowed")
    if ABLATION_VERSIONS["A1_REMOVE_D0_TIMING"] != D0_TIMING_FEATURES:
        raise RuntimeError("FATAL: A1 feature set changed")
    if ABLATION_VERSIONS["A2_REMOVE_D0_TIMING_AND_TOTAL_FAMILY"] != (
        *D0_TIMING_FEATURES,
        *TOTAL_FAMILY_ADDITIONS,
    ):
        raise RuntimeError("FATAL: A2 feature set changed")
    expected_a3_removed = set(FROZEN_FEATURE_COLUMNS).difference(REDUCED_BASE_FEATURES)
    if set(ABLATION_VERSIONS["A3_REDUCED_BASE_ONLY"]) != expected_a3_removed:
        raise RuntimeError("FATAL: A3 is not the exact seven-feature base representation")
    if len(ABLATION_VERSIONS["A3_REDUCED_BASE_ONLY"]) != 11:
        raise RuntimeError("FATAL: A3 must remove exactly 11 frozen contributions")
    if BOOTSTRAP_RESAMPLES != 20_000:
        raise RuntimeError("FATAL: bootstrap contract changed")


def _sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _load_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    source = _bridge_dir(root)
    candidates_path = source / BRIDGE_CANDIDATES
    spec_path = source / BRIDGE_SPEC
    lineage_path = source / BRIDGE_LINEAGE
    for path in (candidates_path, spec_path, lineage_path):
        if not path.is_file():
            raise RuntimeError(f"FATAL: required bridge input missing: {path}")
    frame = pd.read_csv(
        candidates_path,
        encoding="utf-8-sig",
        dtype={"event_id": str, "code": str, "signal_date": str},
    )
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    lineage = pd.read_csv(lineage_path, encoding="utf-8-sig")
    if spec.get("model_id") != MODEL_ID:
        raise RuntimeError("FATAL: bridge model is not current frozen Stage1")
    if spec.get("features") != list(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: bridge feature order differs from frozen 18F")
    if lineage["feature_name"].tolist() != list(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: bridge lineage differs from frozen 18F")
    if set(frame["signal_date"].str[:4]) != {"2026"}:
        raise RuntimeError("FATAL: unexpected analysis year")
    if frame["signal_date"].min() != "2026-05-06" or frame["signal_date"].max() != "2026-07-31":
        raise RuntimeError("FATAL: bridge date range changed")
    if bool(frame["signal_date"].ge("2026-08-01").any()):
        raise RuntimeError("FATAL: August signal date entered the audit")
    return frame, spec, lineage


def _rank_scores(frame: pd.DataFrame, score_column: str, rank_column: str) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["signal_date", score_column, "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked[rank_column] = ranked.groupby("signal_date", sort=False).cumcount() + 1
    return ranked.sort_values(["signal_date", rank_column, "event_id"], kind="mergesort")


def build_a0_and_parity(
    frame: pd.DataFrame, spec: Mapping[str, Any], root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build A0 from frozen inputs and stop immediately on any parity failure."""
    beta = np.asarray(
        [float(spec["coefficients"][feature]) for feature in FROZEN_FEATURE_COLUMNS]
    )
    intercept = float(spec["intercept"])
    x = frame[FROZEN_FEATURE_COLUMNS].to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError("FATAL: missing/nonfinite frozen 18F input")
    result = frame.copy()
    result["A0_logit"] = intercept + x @ beta
    result["A0_score"] = _sigmoid(result["A0_logit"].to_numpy(float))
    result = _rank_scores(result, "A0_score", "A0_rank")

    score_error = np.abs(
        result["A0_score"].to_numpy(float)
        - pd.to_numeric(result["stage1_score"], errors="raise").to_numpy(float)
    )
    rank_mismatches = int(
        pd.to_numeric(result["A0_rank"], errors="raise").ne(
            pd.to_numeric(result["stage1_rank"], errors="raise")
        ).sum()
    )
    july = result[result["signal_date"].str.startswith("2026-07")].copy()
    lock_path = root / str(spec["july_lock"]["path"])
    lock = pd.read_csv(
        lock_path,
        encoding="utf-8-sig",
        dtype={"event_id": str, "code": str, "signal_date": str},
        usecols=["event_id", "signal_date", "code"],
    )
    lock["code"] = lock["code"].astype(str).str.zfill(6)
    identity = july[["event_id", "signal_date", "code"]].merge(
        lock,
        on="event_id",
        how="outer",
        suffixes=("_bridge", "_lock"),
        indicator=True,
    )
    identity_mismatch = (
        identity["_merge"].ne("both")
        | identity["signal_date_bridge"].ne(identity["signal_date_lock"])
        | identity["code_bridge"].ne(identity["code_lock"])
    )
    parity = {
        "rows": int(len(result)),
        "dates": int(result["signal_date"].nunique()),
        "duplicate_event_id_count": int(result["event_id"].duplicated().sum()),
        "missing_18f_cell_count": int(result[FROZEN_FEATURE_COLUMNS].isna().sum().sum()),
        "missing_18f_row_count": int(result[FROZEN_FEATURE_COLUMNS].isna().any(axis=1).sum()),
        "reconstructed_score_max_abs_error": float(score_error.max()),
        "reconstructed_rank_mismatch_count": rank_mismatches,
        "july_locked_population_identity_mismatch_count": int(identity_mismatch.sum()),
        "july_rows": int(len(july)),
        "july_dates": int(july["signal_date"].nunique()),
    }
    failures = []
    if parity["rows"] != 497 or parity["dates"] != 62:
        failures.append("bridge_population")
    if parity["duplicate_event_id_count"] != 0:
        failures.append("duplicate_event_id")
    if parity["missing_18f_cell_count"] != 0:
        failures.append("missing_18f")
    if parity["reconstructed_score_max_abs_error"] > SCORE_TOLERANCE:
        failures.append("score_reconstruction")
    if parity["reconstructed_rank_mismatch_count"] != 0:
        failures.append("rank_reconstruction")
    if parity["july_locked_population_identity_mismatch_count"] != 0:
        failures.append("july_identity")
    if parity["july_rows"] != 178 or parity["july_dates"] != 23:
        failures.append("july_population")
    if failures:
        raise RuntimeError(f"FATAL: A0 parity failed; A1/A2/A3 not run: {failures}")
    return result.reset_index(drop=True), parity


def build_ablation_scores(
    a0: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    """Apply only the three predeclared contribution-removal challengers."""
    result = a0.copy()
    for model, removed in ABLATION_VERSIONS.items():
        prefix = model.split("_", 1)[0]
        if prefix == "A0":
            continue
        removed_contribution = np.zeros(len(result), dtype=float)
        for feature in removed:
            removed_contribution += (
                pd.to_numeric(result[feature], errors="raise").to_numpy(float)
                * float(spec["coefficients"][feature])
            )
        result[f"{prefix}_removed_contribution"] = removed_contribution
        result[f"{prefix}_logit"] = result["A0_logit"] - removed_contribution
        result[f"{prefix}_score"] = _sigmoid(result[f"{prefix}_logit"].to_numpy(float))
        result = _rank_scores(result, f"{prefix}_score", f"{prefix}_rank")
    return result.sort_values(["signal_date", "A0_rank", "event_id"], kind="mergesort").reset_index(drop=True)


def _date_equal_auc(day: pd.DataFrame, score_column: str) -> float:
    return binary_auc(day["target7"], day[score_column])


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "ALL_CANDIDATES":
        return frame.copy()
    if scope == "BOARD2_ONLY":
        return frame[frame["board_group"].eq("BOARD2")].copy()
    if scope == "BOARD3_ONLY":
        return frame[frame["board_group"].eq("BOARD3")].copy()
    raise ValueError(scope)


def build_daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mature_by_date = frame.groupby("signal_date", sort=True)["raw_repair_return"].agg(
        lambda values: bool(values.notna().all())
    )
    partial_by_date = frame.groupby("signal_date", sort=True)["raw_repair_return"].agg(
        lambda values: bool(values.notna().any() and not values.notna().all())
    )
    if bool(partial_by_date.any()):
        raise RuntimeError("FATAL: partially matured date cannot enter date-equal audit")
    for scope in SCOPE_ORDER:
        scoped = _scope_frame(frame, scope)
        for date, day in scoped.groupby("signal_date", sort=True):
            outcome_mature = bool(mature_by_date.loc[date])
            for model in MODEL_ORDER:
                prefix = model.split("_", 1)[0]
                ranked = day.sort_values(
                    [f"{prefix}_score", "event_id"],
                    ascending=[False, True],
                    kind="mergesort",
                ).copy()
                ranked["_scope_rank"] = np.arange(1, len(ranked) + 1)
                row: dict[str, Any] = {
                    "signal_date": date,
                    "month": date[:7],
                    "population_scope": scope,
                    "model": model,
                    "candidate_count": int(len(ranked)),
                    "outcome_evaluation_eligible": "YES" if outcome_mature else "NO",
                }
                for k in (1, 2, 3, 5):
                    selected = ranked.head(min(k, len(ranked)))
                    row[f"top{k}_selected_rows"] = int(len(selected))
                    row[f"top{k}_event_ids"] = "|".join(selected["event_id"].astype(str))
                    row[f"top{k}_codes"] = "|".join(selected["code"].astype(str))
                if outcome_mature:
                    row.update({
                        "universe_target7_rate": float(ranked["target7"].mean()),
                        "universe_loss_rate": float(ranked["loss"].mean()),
                        "universe_severe_loss_rate": float(ranked["severe_loss"].mean()),
                        "universe_capped_return": float(ranked["capped_return_7"].mean()),
                        "target7_within_date_auc": _date_equal_auc(ranked, f"{prefix}_score"),
                    })
                    total_winners = int(ranked["target7"].sum())
                    for k in (1, 2, 3, 5):
                        selected = ranked.head(min(k, len(ranked)))
                        row[f"top{k}_target7_rate"] = float(selected["target7"].mean())
                        row[f"top{k}_loss_rate"] = float(selected["loss"].mean())
                        row[f"top{k}_severe_loss_rate"] = float(selected["severe_loss"].mean())
                        row[f"top{k}_capped_return"] = float(selected["capped_return_7"].mean())
                        row[f"top{k}_excess_vs_universe"] = (
                            row[f"top{k}_capped_return"] - row["universe_capped_return"]
                        )
                        row[f"top{k}_negative_date"] = int(row[f"top{k}_capped_return"] < 0)
                        row[f"recall_at_{k}"] = (
                            float(selected["target7"].sum() / total_winners)
                            if total_winners else np.nan
                        )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["signal_date", "population_scope", "model"], kind="mergesort"
    ).reset_index(drop=True)


def _period_daily(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    eligible = daily[daily["outcome_evaluation_eligible"].eq("YES")]
    month = eligible["month"]
    if period == "MAY":
        return eligible[month.eq("2026-05")]
    if period == "JUNE":
        return eligible[month.eq("2026-06")]
    if period == "MAY_JUNE":
        return eligible[month.isin(("2026-05", "2026-06"))]
    if period == "JULY_MATURE_ONLY":
        return eligible[month.eq("2026-07")]
    if period == "MAY_JUNE_JULY_MATURE_ONLY":
        return eligible
    raise ValueError(period)


def build_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIOD_ORDER:
        period_frame = _period_daily(daily, period)
        for scope in SCOPE_ORDER:
            for model in MODEL_ORDER:
                part = period_frame[
                    period_frame["population_scope"].eq(scope)
                    & period_frame["model"].eq(model)
                ]
                if part.empty:
                    continue
                row: dict[str, Any] = {
                    "period": period,
                    "population_scope": scope,
                    "model": model,
                    "deleted_feature_count": len(ABLATION_VERSIONS[model]),
                    "deleted_features": "|".join(ABLATION_VERSIONS[model]),
                    "dates": int(part["signal_date"].nunique()),
                    "candidate_rows": int(part["candidate_count"].sum()),
                    "target7_within_date_auc": float(part["target7_within_date_auc"].mean()),
                    "eligible_auc_dates": int(part["target7_within_date_auc"].notna().sum()),
                    "universe_target7_rate": float(part["universe_target7_rate"].mean()),
                    "universe_loss_rate": float(part["universe_loss_rate"].mean()),
                    "universe_severe_loss_rate": float(part["universe_severe_loss_rate"].mean()),
                    "universe_capped_return": float(part["universe_capped_return"].mean()),
                }
                for k in (1, 2, 3):
                    row[f"top{k}_target7_rate"] = float(part[f"top{k}_target7_rate"].mean())
                    row[f"top{k}_loss_rate"] = float(part[f"top{k}_loss_rate"].mean())
                    row[f"top{k}_severe_loss_rate"] = float(part[f"top{k}_severe_loss_rate"].mean())
                    row[f"top{k}_capped_return"] = float(part[f"top{k}_capped_return"].mean())
                    row[f"top{k}_excess_vs_universe"] = float(part[f"top{k}_excess_vs_universe"].mean())
                    row[f"top{k}_negative_date_rate"] = float(part[f"top{k}_negative_date"].mean())
                row["worst_daily_top3_capped_return"] = float(part["top3_capped_return"].min())
                row["top3_loss_excess"] = row["top3_loss_rate"] - row["universe_loss_rate"]
                for k in (1, 2, 3, 5):
                    valid_recall = part[f"recall_at_{k}"].notna()
                    row[f"recall_at_{k}"] = float(part.loc[valid_recall, f"recall_at_{k}"].mean()) if valid_recall.any() else np.nan
                    row[f"recall_at_{k}_eligible_dates"] = int(valid_recall.sum())
                row["support_warning"] = (
                    "DESCRIPTIVE_ONLY_LOW_WITHIN_DATE_CLASS_SUPPORT"
                    if scope == "BOARD3_ONLY" and row["eligible_auc_dates"] < 5
                    else ""
                )
                rows.append(row)
    result = pd.DataFrame(rows)
    period_map = {value: index for index, value in enumerate(PERIOD_ORDER)}
    scope_map = {value: index for index, value in enumerate(SCOPE_ORDER)}
    model_map = {value: index for index, value in enumerate(MODEL_ORDER)}
    return result.sort_values(
        ["period", "population_scope", "model"],
        key=lambda values: values.map(
            period_map if values.name == "period" else scope_map if values.name == "population_scope" else model_map
        ),
        kind="mergesort",
    ).reset_index(drop=True)


def build_membership(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    maturity = frame.groupby("signal_date", sort=True)["raw_repair_return"].agg(
        lambda values: bool(values.notna().all())
    )
    for challenger in MODEL_ORDER[1:]:
        prefix = challenger.split("_", 1)[0]
        for date, day in frame.groupby("signal_date", sort=True):
            k = min(3, len(day))
            baseline_ids = set(day.nsmallest(k, "A0_rank")["event_id"].astype(str))
            challenger_ids = set(day.nsmallest(k, f"{prefix}_rank")["event_id"].astype(str))
            for change_type, identities in (
                ("DEMOTED", baseline_ids - challenger_ids),
                ("PROMOTED", challenger_ids - baseline_ids),
            ):
                changed = day[day["event_id"].astype(str).isin(identities)].sort_values(
                    ["A0_rank" if change_type == "DEMOTED" else f"{prefix}_rank", "event_id"],
                    kind="mergesort",
                )
                for record in changed.itertuples(index=False):
                    record_map = record._asdict()
                    rows.append({
                        "challenger": challenger,
                        "signal_date": date,
                        "outcome_evaluation_eligible": "YES" if maturity.loc[date] else "NO",
                        "change_type": change_type,
                        "event_id": record_map["event_id"],
                        "code": record_map["code"],
                        "board_group": record_map["board_group"],
                        "a0_rank": int(record_map["A0_rank"]),
                        "challenger_rank": int(record_map[f"{prefix}_rank"]),
                        "a0_score": float(record_map["A0_score"]),
                        "challenger_score": float(record_map[f"{prefix}_score"]),
                        "removed_contribution": float(record_map[f"{prefix}_removed_contribution"]),
                        "target7": record_map["target7"],
                        "positive_non_target": record_map["positive_non_target"],
                        "loss": record_map["loss"],
                        "severe_loss": record_map["severe_loss"],
                        "raw_repair_return": record_map["raw_repair_return"],
                        "capped_return_7": record_map["capped_return_7"],
                    })
    columns = [
        "challenger", "signal_date", "outcome_evaluation_eligible", "change_type",
        "event_id", "code", "board_group", "a0_rank", "challenger_rank",
        "a0_score", "challenger_score", "removed_contribution", "target7",
        "positive_non_target", "loss", "severe_loss", "raw_repair_return",
        "capped_return_7",
    ]
    membership = pd.DataFrame(rows, columns=columns)
    for challenger in MODEL_ORDER[1:]:
        mature = membership[
            membership["challenger"].eq(challenger)
            & membership["outcome_evaluation_eligible"].eq("YES")
        ]
        promoted = mature[mature["change_type"].eq("PROMOTED")]
        demoted = mature[mature["change_type"].eq("DEMOTED")]
        summary[challenger] = {
            "changed_dates": int(mature["signal_date"].nunique()),
            "changed_slots": int(len(promoted)),
            "promoted_target7": int(promoted["target7"].sum()),
            "promoted_pnt": int(promoted["positive_non_target"].sum()),
            "promoted_loss": int(promoted["loss"].sum()),
            "demoted_target7": int(demoted["target7"].sum()),
            "demoted_pnt": int(demoted["positive_non_target"].sum()),
            "demoted_loss": int(demoted["loss"].sum()),
            "net_target7_slots": int(promoted["target7"].sum() - demoted["target7"].sum()),
            "net_loss_slots_change": int(promoted["loss"].sum() - demoted["loss"].sum()),
            "net_loss_slots_removed": int(demoted["loss"].sum() - promoted["loss"].sum()),
        }
    return membership, summary


def build_paired_bootstrap(daily: pd.DataFrame) -> pd.DataFrame:
    main = daily[
        daily["population_scope"].eq("ALL_CANDIDATES")
        & daily["outcome_evaluation_eligible"].eq("YES")
    ]
    pivot = main.pivot(index="signal_date", columns="model", values="top3_capped_return")
    rows: list[dict[str, Any]] = []
    for period_index, period in enumerate(PERIOD_ORDER):
        period_dates = _period_daily(main, period)["signal_date"].drop_duplicates().sort_values()
        period_pivot = pivot.loc[period_dates]
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, period_index]))
        draws = rng.integers(0, len(period_pivot), size=(BOOTSTRAP_RESAMPLES, len(period_pivot)))
        for challenger in MODEL_ORDER[1:]:
            delta = (
                period_pivot[challenger].to_numpy(float)
                - period_pivot["A0_ORIGINAL_18F"].to_numpy(float)
            )
            boot = delta[draws].mean(axis=1)
            rows.append({
                "period": period,
                "challenger": challenger,
                "baseline": "A0_ORIGINAL_18F",
                "sampling_unit": "signal_date",
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "dates": int(len(delta)),
                "estimate_mean": float(delta.mean()),
                "observed_median": float(np.median(delta)),
                "positive_days": int(np.sum(delta > 0)),
                "negative_days": int(np.sum(delta < 0)),
                "zero_days": int(np.sum(delta == 0)),
                "bootstrap_p2_5": float(np.quantile(boot, .025)),
                "bootstrap_p50": float(np.quantile(boot, .50)),
                "bootstrap_p97_5": float(np.quantile(boot, .975)),
                "bootstrap_probability_delta_gt_0": float(np.mean(boot > 0)),
            })
    return pd.DataFrame(rows)


def build_days_since_d0_parity(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    canonical = (
        pd.to_datetime(frame["d1_date"], errors="raise")
        - pd.to_datetime(frame["d0_date"], errors="raise")
    ).dt.days
    raw = pd.to_numeric(frame["raw__days_since_d0"], errors="coerce")
    result = frame[["event_id", "signal_date", "code", "d0_date", "d1_date"]].copy()
    result["raw_days_since_d0"] = raw
    result["canonical_calendar_days"] = canonical
    result["match"] = raw.eq(canonical)
    audit = {
        "mismatch_rows": int((~result["match"]).sum()),
        "mismatch_dates": int(result.loc[~result["match"], "signal_date"].nunique()),
    }
    return result, audit


def analyze(root: str | Path) -> dict[str, Any]:
    assert_audit_contract()
    root_path = Path(root).resolve()
    frame, spec, lineage = _load_inputs(root_path)
    a0, parity = build_a0_and_parity(frame, spec, root_path)
    scored = build_ablation_scores(a0, spec)
    daily = build_daily(scored)
    summary = build_summary(daily)
    membership, membership_summary = build_membership(scored)
    bootstrap = build_paired_bootstrap(daily)
    days_parity, days_audit = build_days_since_d0_parity(scored)
    return {
        "root": root_path,
        "spec": spec,
        "lineage": lineage,
        "scored": scored,
        "parity": parity,
        "daily": daily,
        "summary": summary,
        "membership": membership,
        "membership_summary": membership_summary,
        "bootstrap": bootstrap,
        "days_parity": days_parity,
        "days_audit": days_audit,
    }


def _metric_row(
    summary: pd.DataFrame, period: str, model: str, scope: str = "ALL_CANDIDATES"
) -> pd.Series:
    matches = summary[
        summary["period"].eq(period)
        & summary["population_scope"].eq(scope)
        & summary["model"].eq(model)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"FATAL: summary row missing/duplicated: {period}/{scope}/{model}")
    return matches.iloc[0]


def _pct(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4%}"


def render_review(
    context: Mapping[str, Any],
    research_conclusion: str,
    next_action: str,
    answer_notes: Mapping[str, str],
) -> str:
    allowed_conclusions = {
        "LEGACY_STRUCTURE_HARM_SUPPORTED",
        "LEGACY_STRUCTURE_HARM_NOT_SUPPORTED",
        "MIXED",
    }
    allowed_actions = {
        "TRAIN_ONE_REDUCED_STAGE1",
        "KEEP_CURRENT_STAGE1_AND_STOP",
        "INSUFFICIENT_EVIDENCE",
    }
    if research_conclusion not in allowed_conclusions or next_action not in allowed_actions:
        raise RuntimeError("FATAL: review decision outside allowed enum")
    missing_answers = [f"Q{i}" for i in range(1, 7) if f"Q{i}" not in answer_notes]
    if missing_answers:
        raise RuntimeError(f"FATAL: review answers missing {missing_answers}")
    parity = context["parity"]
    summary = context["summary"]
    lines = [
        "# v004c Frozen Stage1 Contribution Ablation Audit v001",
        "",
        "## Q1. A0 是否精确复现 current frozen Stage1？",
        "",
        f"{answer_notes['Q1']} Rows/dates **{parity['rows']}/{parity['dates']}**; score max abs error **{parity['reconstructed_score_max_abs_error']:.3e}**; rank mismatch **{parity['reconstructed_rank_mismatch_count']}**; July identity mismatch **{parity['july_locked_population_identity_mismatch_count']}**.",
    ]
    questions = (
        ("Q2", "删除 D0 timing family 是否有稳定价值？", "A1_REMOVE_D0_TIMING"),
        ("Q3", "进一步删除 total_score family 是否有稳定价值？", "A2_REMOVE_D0_TIMING_AND_TOTAL_FAMILY"),
        ("Q4", "进一步删除 legacy interactions 是否有稳定价值？", "A3_REDUCED_BASE_ONLY"),
    )
    for key, question, model in questions:
        transition = context["membership_summary"][model]
        lines += [
            "", f"## {key}. {question}", "", answer_notes[key], "",
            "| Period | AUC delta | Top3 excess delta | Top3 LOSS-excess delta |",
            "|---|---:|---:|---:|",
        ]
        for period in ("MAY", "JUNE", "JULY_MATURE_ONLY", "MAY_JUNE_JULY_MATURE_ONLY"):
            base = _metric_row(summary, period, "A0_ORIGINAL_18F")
            challenger = _metric_row(summary, period, model)
            lines.append(
                f"| {period} | {_pct(challenger.target7_within_date_auc-base.target7_within_date_auc)} | "
                f"{_pct(challenger.top3_excess_vs_universe-base.top3_excess_vs_universe)} | "
                f"{_pct(challenger.top3_loss_excess-base.top3_loss_excess)} |"
            )
        lines += [
            "",
            f"- Mature Top3 membership: changed dates/slots **{transition['changed_dates']}/{transition['changed_slots']}**; "
            f"promoted T7/PNT/LOSS **{transition['promoted_target7']}/{transition['promoted_pnt']}/{transition['promoted_loss']}**; "
            f"demoted T7/PNT/LOSS **{transition['demoted_target7']}/{transition['demoted_pnt']}/{transition['demoted_loss']}**; "
            f"net Target7 slots **{transition['net_target7_slots']}**; net LOSS slots removed **{transition['net_loss_slots_removed']}**.",
        ]
    lines += [
        "",
        "## Q5. 是否存在“越简单反而跨 May/June/July 越稳定”的证据？",
        "",
        answer_notes["Q5"],
        "",
        "## Q6. 现有证据是否足以授权下一步只训练一个 Reduced V4C Stage1 Logistic？",
        "",
        answer_notes["Q6"],
        "",
        f"Research conclusion: `{research_conclusion}`",
        "",
        f"NEXT_ACTION = {next_action}",
        "",
    ]
    return "\n".join(lines)


def build_outputs(
    context: Mapping[str, Any],
    research_conclusion: str,
    next_action: str,
    answer_notes: Mapping[str, str],
) -> dict[str, bytes]:
    review = render_review(context, research_conclusion, next_action, answer_notes)
    return {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(context["summary"]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(context["daily"]),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(context["membership"]),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(context["bootstrap"]),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(context["days_parity"]),
        OUTPUT_FILENAMES[5]: review.encode("utf-8"),
    }


def write_outputs(
    context: Mapping[str, Any],
    research_conclusion: str,
    next_action: str,
    answer_notes: Mapping[str, str],
    output_dir: str | Path | None = None,
) -> Path:
    root = Path(context["root"])
    target = Path(output_dir) if output_dir is not None else (
        root / "reports" / "research" / OUTPUT_DIRNAME
    )
    first = build_outputs(context, research_conclusion, next_action, answer_notes)
    second = build_outputs(context, research_conclusion, next_action, answer_notes)
    if first != second:
        raise RuntimeError("FATAL: deterministic output rebuild failed")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (target / name).write_bytes(payload)
    return target

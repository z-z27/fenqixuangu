from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v004c_board2_board3_structural_audit import load_strict_funnel
from src.v004c_board3_only_v4a_arch_feasibility import same_date_pair_summary
from src.v004c_stage1_top3_risk_information import binary_auc
from src.v004c_v4a_architecture_transfer import dataframe_csv_bytes


TASK_NAME = "BOARD3-ONLY OOF RISK SELECTIVITY DIAGNOSTIC"
EXPECTED_STARTING_HEAD = "15c0000ef6a72f5a80f104c01a0527e918efef82"
POST_HOC_OOF_RISK_SELECTIVITY_DIAGNOSTIC = True

NEW_MODEL = False
MODEL_REFIT = False
NEW_FEATURE = False
NEW_TARGET = False
NEW_SCORE = False
THRESHOLD_SEARCH = False
VETO_POLICY = False
BACKFILL = False
BOARD2_MODIFICATION = False
CROSS_BOARD_COMBINATION = False
JULY_RESULT_ROWS_ACCESSED = 0

EXPECTED_LOCK_SHA256 = "2f85e79738dbbedf19382db0da5f1136de02e84c69557ea5c85341bcf62fbd6b"
EXPECTED_COMMON_DATES = [
    "2026-06-12", "2026-06-16", "2026-06-17", "2026-06-18",
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
]
EXPECTED_COMMON_ROWS = 19
EXPECTED_TARGET7 = 5
EXPECTED_LOSS = 6
EXPECTED_PNT = 8
EXPECTED_NONLOSS = 13

BOTTOM_HALF_CUTOFF = 0.50
TAIL_DEFINITIONS = ("BOTTOM_HALF", "WORST_ONE")
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_RESAMPLES = 20_000

SOURCE_DIR = (
    "reports/research/"
    "v004c_board3_only_v4a_arch_feasibility_v001_20260506_20260626"
)
LOCK_FILENAME = "v004c_board3_only_prediction_lock_v001.csv"
OOF_FILENAME = "v004c_board3_only_oof_v001.csv"

OUTPUT_FILENAMES = (
    "v004c_board3_risk_selectivity_population_v001.csv",
    "v004c_board3_risk_selectivity_daily_v001.csv",
    "v004c_board3_risk_selectivity_tail_v001.csv",
    "v004c_board3_risk_selectivity_curve_v001.csv",
    "v004c_board3_risk_selectivity_decision_surface_v001.csv",
    "v004c_board3_risk_selectivity_robustness_v001.csv",
    "v004c_board3_risk_selectivity_review_v001.md",
)

RISK_SURFACE_COLUMNS = (
    "event_id", "signal_date", "code", "board3_candidate_count",
    "board3_only_score", "board3_only_internal_rank",
)


def assert_diagnostic_contract() -> None:
    if not POST_HOC_OOF_RISK_SELECTIVITY_DIAGNOSTIC:
        raise RuntimeError("FATAL: post-hoc diagnostic disclosure changed")
    if any((NEW_MODEL, MODEL_REFIT, NEW_FEATURE, NEW_TARGET, NEW_SCORE,
            THRESHOLD_SEARCH, VETO_POLICY, BACKFILL, BOARD2_MODIFICATION,
            CROSS_BOARD_COMBINATION)):
        raise RuntimeError("FATAL: unauthorized model/policy/search dimension enabled")
    if JULY_RESULT_ROWS_ACCESSED != 0:
        raise RuntimeError("FATAL: July result access is prohibited")
    if BOTTOM_HALF_CUTOFF != 0.50 or TAIL_DEFINITIONS != ("BOTTOM_HALF", "WORST_ONE"):
        raise RuntimeError("FATAL: fixed tail diagnostic contract changed")
    if BOOTSTRAP_SEED != 20260820 or BOOTSTRAP_RESAMPLES != 20_000:
        raise RuntimeError("FATAL: robustness contract changed")


def _source_path(root: Path, filename: str) -> Path:
    path = root / SOURCE_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"FATAL: locked Board3-only artifact unavailable: {path}")
    return path


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def derive_relative_risk_surface(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the two predeclared tails from locked, outcome-free rank fields."""
    missing = sorted(set(RISK_SURFACE_COLUMNS).difference(frame.columns))
    if missing:
        raise RuntimeError(f"FATAL: risk surface missing locked fields: {missing}")
    surface = frame[list(RISK_SURFACE_COLUMNS)].copy()
    surface["board3_candidate_count"] = pd.to_numeric(
        surface["board3_candidate_count"]
    ).astype(int)
    surface["board3_only_internal_rank"] = pd.to_numeric(
        surface["board3_only_internal_rank"]
    ).astype(int)
    surface["board3_only_score"] = pd.to_numeric(surface["board3_only_score"])
    surface["relative_rank_eligible"] = surface["board3_candidate_count"].ge(2)
    surface["risk_rank_percentile"] = np.where(
        surface["relative_rank_eligible"],
        (surface["board3_only_internal_rank"] - 1)
        / (surface["board3_candidate_count"] - 1),
        np.nan,
    )
    surface["bottom_half_flag"] = (
        surface["relative_rank_eligible"]
        & surface["risk_rank_percentile"].ge(BOTTOM_HALF_CUTOFF)
    )
    surface["worst_one_flag"] = (
        surface["relative_rank_eligible"]
        & surface["board3_only_internal_rank"].eq(surface["board3_candidate_count"])
    )
    return surface.sort_values(
        ["signal_date", "board3_only_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_locked_identity(lock: pd.DataFrame, oof: pd.DataFrame) -> None:
    left = lock[list(RISK_SURFACE_COLUMNS)].copy().sort_values("event_id").reset_index(drop=True)
    right = oof[list(RISK_SURFACE_COLUMNS)].copy().sort_values("event_id").reset_index(drop=True)
    if dataframe_csv_bytes(left) != dataframe_csv_bytes(right):
        raise RuntimeError("FATAL: OOF score/rank surface differs from prediction lock")


def load_locked_population(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the existing prediction lock and its attached OOF outcomes; never fit."""
    assert_diagnostic_contract()
    lock_path = _source_path(root, LOCK_FILENAME)
    oof_path = _source_path(root, OOF_FILENAME)
    observed_hash = sha256(lock_path.read_bytes()).hexdigest()
    if observed_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError(
            f"FATAL: prediction lock SHA256 mismatch: {observed_hash} != {EXPECTED_LOCK_SHA256}"
        )
    dtype = {"event_id": str, "code": str, "signal_date": str}
    lock = pd.read_csv(lock_path, encoding="utf-8-sig", dtype=dtype)
    oof = pd.read_csv(oof_path, encoding="utf-8-sig", dtype=dtype)
    if lock["event_id"].duplicated().any() or oof["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in locked Board3-only artifacts")
    _validate_locked_identity(lock, oof)

    surface = derive_relative_risk_surface(lock)
    outcome_columns = [
        "event_id", "unified_stage1_score", "unified_board3_internal_rank",
        "target7", "loss", "nonloss", "raw_repair_return", "capped_return_7",
    ]
    population = surface.merge(
        oof[outcome_columns], on="event_id", how="left", validate="one_to_one"
    )
    if population[outcome_columns[1:]].isna().any().any():
        raise RuntimeError("FATAL: locked OOF outcome join incomplete")
    population["target7"] = pd.to_numeric(population["target7"]).astype(int)
    population["loss"] = pd.to_numeric(population["loss"]).astype(int)
    population["nonloss"] = pd.to_numeric(population["nonloss"]).astype(int)
    population["raw_repair_return"] = pd.to_numeric(population["raw_repair_return"])
    population["capped_return_7"] = pd.to_numeric(population["capped_return_7"])
    population["positive_non_target"] = (
        population["raw_repair_return"].ge(0)
        & population["raw_repair_return"].lt(.07)
    ).astype(int)
    population["code"] = population["code"].astype(str).str.zfill(6)
    population["risk_score"] = -population["board3_only_score"]

    strict, _ = load_strict_funnel(root)
    strict_join = strict[["event_id", "stage1_score", "stage1_rank", "stage1_top3"]].copy()
    population = population.merge(strict_join, on="event_id", validate="one_to_one")
    if not np.allclose(
        population["unified_stage1_score"], population["stage1_score"],
        rtol=0.0, atol=1e-10,
    ):
        raise RuntimeError("FATAL: unified strict Stage1 score mismatch")
    population = population.rename(columns={
        "stage1_rank": "unified_stage1_rank",
        "stage1_top3": "unified_stage1_top3",
    }).drop(columns="stage1_score")
    population["unified_stage1_top3"] = population["unified_stage1_top3"].astype(bool)

    population = population.sort_values(
        ["signal_date", "board3_only_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    dates = sorted(population["signal_date"].unique())
    outcomes = {
        "target7": int(population["target7"].sum()),
        "loss": int(population["loss"].sum()),
        "positive_non_target": int(population["positive_non_target"].sum()),
        "nonloss": int(population["nonloss"].sum()),
    }
    expected_outcomes = {
        "target7": EXPECTED_TARGET7, "loss": EXPECTED_LOSS,
        "positive_non_target": EXPECTED_PNT, "nonloss": EXPECTED_NONLOSS,
    }
    if dates != EXPECTED_COMMON_DATES or len(population) != EXPECTED_COMMON_ROWS:
        raise RuntimeError("FATAL: common Board3 OOF date/row parity changed")
    if outcomes != expected_outcomes:
        raise RuntimeError(f"FATAL: common Board3 OOF outcomes changed: {outcomes}")
    if not (population["loss"].eq(population["raw_repair_return"].lt(0).astype(int))).all():
        raise RuntimeError("FATAL: LOSS semantics changed")
    if not (population["target7"].eq(population["raw_repair_return"].ge(.07).astype(int))).all():
        raise RuntimeError("FATAL: Target7 semantics changed")
    if not np.allclose(
        population["capped_return_7"], np.minimum(population["raw_repair_return"], .07),
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("FATAL: capped-return semantics changed")
    actual_counts = population.groupby("signal_date", sort=True).size()
    declared_counts = population.groupby("signal_date", sort=True)["board3_candidate_count"].first()
    if not actual_counts.equals(declared_counts.astype(int)):
        raise RuntimeError("FATAL: Board3 candidate-count parity changed")
    count_audit = {
        "min": int(actual_counts.min()), "median": float(actual_counts.median()),
        "mean": float(actual_counts.mean()), "max": int(actual_counts.max()),
        "dates_1": int(actual_counts.eq(1).sum()), "dates_2": int(actual_counts.eq(2).sum()),
        "dates_ge4": int(actual_counts.ge(4).sum()),
    }
    if count_audit != {
        "min": 1, "median": 2.0, "mean": 2.375, "max": 8,
        "dates_1": 3, "dates_2": 4, "dates_ge4": 1,
    }:
        raise RuntimeError(f"FATAL: candidate-count parity changed: {count_audit}")
    return population, {
        "expected_lock_sha256": EXPECTED_LOCK_SHA256,
        "observed_lock_sha256": observed_hash,
        "outcome_perturbation": "PASS", "candidate_counts": count_audit,
        "july_result_rows_accessed": 0,
    }


def outcome_perturbation_is_immutable(population: pd.DataFrame) -> bool:
    original = derive_relative_risk_surface(population)
    perturbed = population.copy()
    rng = np.random.default_rng(20260820)
    for column in ("target7", "loss", "positive_non_target", "nonloss",
                   "raw_repair_return", "capped_return_7"):
        perturbed[column] = rng.normal(999.0, 100.0, len(perturbed))
    rebuilt = derive_relative_risk_surface(perturbed)
    return dataframe_csv_bytes(original) == dataframe_csv_bytes(rebuilt)


def cross_class_pair_summary(
    frame: pd.DataFrame, score_column: str, positive_column: str, negative_column: str,
) -> dict[str, Any]:
    informative_dates = concordant = discordant = ties = 0
    for _, day in frame.groupby("signal_date", sort=True):
        positive = day.loc[day[positive_column].eq(1), score_column].to_numpy(float)
        negative = day.loc[day[negative_column].eq(1), score_column].to_numpy(float)
        if not len(positive) or not len(negative):
            continue
        informative_dates += 1
        comparison = positive[:, None] - negative[None, :]
        concordant += int((comparison > 0).sum())
        discordant += int((comparison < 0).sum())
        ties += int((comparison == 0).sum())
    pairs = concordant + discordant + ties
    return {
        "informative_dates": informative_dates, "informative_pairs": pairs,
        "concordant": concordant, "discordant": discordant, "ties": ties,
        "concordance": _safe_ratio(concordant + .5 * ties, pairs),
    }


def build_pair_diagnostics(population: pd.DataFrame) -> dict[str, Any]:
    unified = same_date_pair_summary(population, "unified_stage1_score", "nonloss")
    board3 = same_date_pair_summary(population, "board3_only_score", "nonloss")
    pnt_loss = cross_class_pair_summary(
        population, "board3_only_score", "positive_non_target", "loss"
    )
    target_loss = cross_class_pair_summary(
        population, "board3_only_score", "target7", "loss"
    )
    return {
        "unified_nonloss": unified, "board3_nonloss": board3,
        "pnt_loss": pnt_loss, "target7_loss": target_loss,
    }


def _series_geometry(values: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "mean": float(series.mean()) if len(series) else np.nan,
        "median": float(series.median()) if len(series) else np.nan,
        "p25": float(series.quantile(.25)) if len(series) else np.nan,
        "p75": float(series.quantile(.75)) if len(series) else np.nan,
    }


def build_absolute_score_geometry(population: pd.DataFrame) -> dict[str, Any]:
    groups = {
        "LOSS": population[population["loss"].eq(1)],
        "PNT": population[population["positive_non_target"].eq(1)],
        "TARGET7": population[population["target7"].eq(1)],
    }
    return {
        name: _series_geometry(group["board3_only_score"])
        for name, group in groups.items()
    }


def build_risk_geometry(eligible: pd.DataFrame) -> dict[str, Any]:
    groups = {
        "LOSS": eligible[eligible["loss"].eq(1)],
        "PNT": eligible[eligible["positive_non_target"].eq(1)],
        "TARGET7": eligible[eligible["target7"].eq(1)],
    }
    result = {
        name: _series_geometry(group["risk_rank_percentile"])
        for name, group in groups.items()
    }
    medians = {name: values["median"] for name, values in result.items()}
    if not all(np.isfinite(value) for value in medians.values()):
        order = "INSUFFICIENT"
    elif medians["LOSS"] > max(medians["PNT"], medians["TARGET7"]):
        order = "LOSS_RISKIEST"
    elif medians["TARGET7"] > max(medians["LOSS"], medians["PNT"]):
        order = "TARGET7_RISKIEST"
    else:
        order = "MIXED"
    favorable = unfavorable = ties = 0
    for _, day in eligible.groupby("signal_date", sort=True):
        loss = day.loc[day["loss"].eq(1), "risk_rank_percentile"]
        nonloss = day.loc[day["nonloss"].eq(1), "risk_rank_percentile"]
        if loss.empty or nonloss.empty:
            continue
        difference = float(loss.median() - nonloss.median())
        favorable += int(difference > 0)
        unfavorable += int(difference < 0)
        ties += int(difference == 0)
    result["risk_median_order"] = order
    result["date_direction"] = {
        "favorable": favorable, "unfavorable": unfavorable, "ties": ties,
        "consistency": _safe_ratio(favorable, favorable + unfavorable),
    }
    return result


def tail_selectivity(
    eligible: pd.DataFrame, flag_column: str, name: str,
) -> dict[str, Any]:
    flagged = eligible[eligible[flag_column].astype(bool)]
    eligible_counts = {
        "loss": int(eligible["loss"].sum()),
        "pnt": int(eligible["positive_non_target"].sum()),
        "target7": int(eligible["target7"].sum()),
    }
    flagged_counts = {
        "loss": int(flagged["loss"].sum()),
        "pnt": int(flagged["positive_non_target"].sum()),
        "target7": int(flagged["target7"].sum()),
    }
    unflagged_counts = {
        key: eligible_counts[key] - flagged_counts[key] for key in eligible_counts
    }
    eligible_rows = len(eligible)
    flagged_rows = len(flagged)
    unflagged_rows = eligible_rows - flagged_rows
    eligible_rates = {
        key: _safe_ratio(value, eligible_rows) for key, value in eligible_counts.items()
    }
    flagged_rates = {
        key: _safe_ratio(value, flagged_rows) for key, value in flagged_counts.items()
    }
    unflagged_rates = {
        key: _safe_ratio(value, unflagged_rows) for key, value in unflagged_counts.items()
    }
    loss_capture = _safe_ratio(flagged_counts["loss"], eligible_counts["loss"])
    pnt_removal = _safe_ratio(flagged_counts["pnt"], eligible_counts["pnt"])
    winner_removal = _safe_ratio(flagged_counts["target7"], eligible_counts["target7"])
    return {
        "tail": name, "eligible_dates": int(eligible["signal_date"].nunique()),
        "eligible_rows": int(eligible_rows), "flagged_rows": int(flagged_rows),
        "unflagged_rows": int(unflagged_rows),
        "flagged_loss": flagged_counts["loss"], "flagged_pnt": flagged_counts["pnt"],
        "flagged_target7": flagged_counts["target7"],
        "unflagged_loss": unflagged_counts["loss"],
        "unflagged_pnt": unflagged_counts["pnt"],
        "unflagged_target7": unflagged_counts["target7"],
        "flagged_loss_rate": flagged_rates["loss"],
        "flagged_pnt_rate": flagged_rates["pnt"],
        "flagged_target7_rate": flagged_rates["target7"],
        "unflagged_loss_rate": unflagged_rates["loss"],
        "unflagged_pnt_rate": unflagged_rates["pnt"],
        "unflagged_target7_rate": unflagged_rates["target7"],
        "eligible_loss_rate": eligible_rates["loss"],
        "eligible_pnt_rate": eligible_rates["pnt"],
        "eligible_target7_rate": eligible_rates["target7"],
        "loss_enrichment": flagged_rates["loss"] - eligible_rates["loss"],
        "loss_relative_risk": _safe_ratio(flagged_rates["loss"], eligible_rates["loss"]),
        "loss_capture_rate": loss_capture, "pnt_removal_rate": pnt_removal,
        "winner_removal_rate": winner_removal,
        "selectivity_gap": loss_capture - winner_removal,
    }


def build_selectivity_curve(eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = eligible.sort_values(
        ["risk_rank_percentile", "signal_date", "event_id"],
        ascending=[False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    total_loss = int(ordered["loss"].sum())
    total_pnt = int(ordered["positive_non_target"].sum())
    total_target7 = int(ordered["target7"].sum())
    rows: list[dict[str, Any]] = []
    loss_removed = pnt_removed = target7_removed = 0
    for index, row in ordered.iterrows():
        loss_removed += int(row["loss"])
        pnt_removed += int(row["positive_non_target"])
        target7_removed += int(row["target7"])
        loss_capture = _safe_ratio(loss_removed, total_loss)
        pnt_removal = _safe_ratio(pnt_removed, total_pnt)
        winner_removal = _safe_ratio(target7_removed, total_target7)
        rows.append({
            "step": index + 1, "removed_rows": index + 1,
            "removed_fraction": (index + 1) / len(ordered),
            "loss_removed": loss_removed, "pnt_removed": pnt_removed,
            "target7_removed": target7_removed,
            "loss_capture_rate": loss_capture, "pnt_removal_rate": pnt_removal,
            "winner_removal_rate": winner_removal,
            "selectivity_gap": loss_capture - winner_removal,
        })
    curve = pd.DataFrame(rows)
    mean_gap = float(curve["selectivity_gap"].mean()) if len(curve) else np.nan
    dominance = float(curve["selectivity_gap"].gt(0).mean()) if len(curve) else np.nan
    if np.isfinite(dominance) and dominance >= .70 and mean_gap > 0:
        pattern = "LOSS_DOMINANT"
    elif np.isfinite(mean_gap) and mean_gap < 0:
        pattern = "WINNER_HARMFUL"
    else:
        pattern = "MIXED"
    return curve, {
        "removal_steps": len(curve), "mean_selectivity_gap": mean_gap,
        "selectivity_dominance_rate": dominance, "early_removal_pattern": pattern,
        "no_threshold_selected": True,
    }


def _outcome_name(row: pd.Series) -> str:
    if int(row["loss"]):
        return "LOSS"
    if int(row["target7"]):
        return "TARGET7"
    return "PNT"


def build_daily(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in population.groupby("signal_date", sort=True):
        highest = day.sort_values(
            ["board3_only_internal_rank", "event_id"], kind="mergesort"
        ).iloc[0]
        lowest = day.sort_values(
            ["board3_only_internal_rank", "event_id"], ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        bottom = day[day["bottom_half_flag"]]
        upper = day[day["relative_rank_eligible"] & ~day["bottom_half_flag"]]
        rows.append({
            "signal_date": date, "candidate_count": len(day),
            "target7": int(day["target7"].sum()),
            "pnt": int(day["positive_non_target"].sum()), "loss": int(day["loss"].sum()),
            "highest_score_code": highest["code"],
            "highest_score_outcome": _outcome_name(highest),
            "lowest_score_code": lowest["code"],
            "lowest_score_outcome": _outcome_name(lowest),
            "bottom_half_rows": len(bottom), "bottom_half_loss": int(bottom["loss"].sum()),
            "bottom_half_pnt": int(bottom["positive_non_target"].sum()),
            "bottom_half_target7": int(bottom["target7"].sum()),
            "upper_half_rows": len(upper), "upper_half_loss": int(upper["loss"].sum()),
            "upper_half_pnt": int(upper["positive_non_target"].sum()),
            "upper_half_target7": int(upper["target7"].sum()),
        })
    return pd.DataFrame(rows)


def build_decision_surface(population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    surface = population[population["unified_stage1_top3"]].copy().sort_values(
        ["signal_date", "unified_stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    t7_loss = surface[surface["target7"].eq(1) | surface["loss"].eq(1)]
    nonloss_auc = binary_auc(surface["nonloss"], surface["board3_only_score"])
    target_loss_auc = binary_auc(t7_loss["target7"], t7_loss["board3_only_score"])
    support = (
        "ADEQUATE" if len(surface) >= 8 and surface["loss"].sum() >= 2
        and surface["nonloss"].sum() >= 2 else "WEAK"
    )
    medians = {
        "loss": float(surface.loc[surface["loss"].eq(1), "board3_only_score"].median()),
        "pnt": float(surface.loc[surface["positive_non_target"].eq(1), "board3_only_score"].median()),
        "target7": float(surface.loc[surface["target7"].eq(1), "board3_only_score"].median()),
    }
    corroboration = (
        "NOT_ESTABLISHED" if support == "WEAK"
        else ("YES" if np.isfinite(nonloss_auc) and nonloss_auc > .55 else "NO")
    )
    summary = {
        "rows": len(surface), "dates": surface["signal_date"].nunique(),
        "target7": int(surface["target7"].sum()),
        "pnt": int(surface["positive_non_target"].sum()),
        "loss": int(surface["loss"].sum()), "score_medians": medians,
        "nonloss_auc": nonloss_auc, "target7_loss_auc": target_loss_auc,
        "support": support, "corroboration": corroboration,
    }
    columns = [
        "event_id", "signal_date", "code", "board3_only_score",
        "board3_only_internal_rank", "risk_rank_percentile", "target7", "loss",
        "positive_non_target", "raw_repair_return",
    ]
    return surface[columns], summary


def date_bootstrap_counts(
    dates: Sequence[str], resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(map(str, dates)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    counts = np.zeros((resamples, len(unique)), dtype=np.int16)
    np.add.at(counts, (np.repeat(np.arange(resamples), len(unique)), draws.ravel()), 1)
    return unique, counts


def _weighted_auc_vector(
    frame: pd.DataFrame, label: str, score: str, dates: list[str], counts: np.ndarray,
) -> np.ndarray:
    date_lookup = {date: index for index, date in enumerate(dates)}
    row_date = np.asarray([date_lookup[str(value)] for value in frame["signal_date"]], dtype=int)
    weights = counts[:, row_date].astype(float)
    labels = frame[label].to_numpy(int)
    scores = frame[score].to_numpy(float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    wp = weights[:, positive]
    wn = weights[:, negative]
    comparison = (
        (scores[positive, None] > scores[None, negative]).astype(float)
        + .5 * (scores[positive, None] == scores[None, negative]).astype(float)
    )
    numerator = np.einsum("ri,ij,rj->r", wp, comparison, wn, optimize=True)
    denominator = wp.sum(axis=1) * wn.sum(axis=1)
    return np.divide(
        numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0,
    )


def _pair_date_components(
    frame: pd.DataFrame, dates: list[str], score: str, positive: str, negative: str,
) -> tuple[np.ndarray, np.ndarray]:
    numerators = np.zeros(len(dates), dtype=float)
    denominators = np.zeros(len(dates), dtype=float)
    for index, date in enumerate(dates):
        day = frame[frame["signal_date"].eq(date)]
        pos = day.loc[day[positive].eq(1), score].to_numpy(float)
        neg = day.loc[day[negative].eq(1), score].to_numpy(float)
        if not len(pos) or not len(neg):
            continue
        comparison = pos[:, None] - neg[None, :]
        numerators[index] = float((comparison > 0).sum() + .5 * (comparison == 0).sum())
        denominators[index] = float(comparison.size)
    return numerators, denominators


def _date_count_vector(frame: pd.DataFrame, dates: list[str], column: str) -> np.ndarray:
    return np.asarray([
        float(frame.loc[frame["signal_date"].eq(date), column].sum()) for date in dates
    ])


def _capture_vector(
    counts: np.ndarray, frame: pd.DataFrame, dates: list[str],
    flag: str, outcome: str,
) -> np.ndarray:
    total = counts @ _date_count_vector(frame, dates, outcome)
    flagged_frame = frame[frame[flag]]
    flagged = counts @ _date_count_vector(flagged_frame, dates, outcome)
    return np.divide(flagged, total, out=np.full(len(counts), np.nan), where=total > 0)


def _metric_summary(
    values: Sequence[float], estimate: float, predicate: Any,
) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "estimate": float(estimate), "valid_resamples": int(len(finite)),
        "p2.5": float(np.quantile(finite, .025)) if len(finite) else np.nan,
        "p50": float(np.quantile(finite, .50)) if len(finite) else np.nan,
        "p97.5": float(np.quantile(finite, .975)) if len(finite) else np.nan,
        "direction_probability": float(np.mean(predicate(finite))) if len(finite) else np.nan,
    }


def _point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    eligible = frame[frame["relative_rank_eligible"]]
    bottom = tail_selectivity(eligible, "bottom_half_flag", "BOTTOM_HALF")
    worst = tail_selectivity(eligible, "worst_one_flag", "WORST_ONE")
    pair = same_date_pair_summary(frame, "board3_only_score", "nonloss")
    return {
        "NONLOSS_AUC": binary_auc(frame["nonloss"], frame["board3_only_score"]),
        "PAIR_CONCORDANCE": pair["concordance"],
        "BOTTOM_HALF_LOSS_CAPTURE": bottom["loss_capture_rate"],
        "BOTTOM_HALF_WINNER_REMOVAL": bottom["winner_removal_rate"],
        "BOTTOM_HALF_SELECTIVITY_GAP": bottom["selectivity_gap"],
        "WORST_ONE_SELECTIVITY_GAP": worst["selectivity_gap"],
    }


def build_robustness(
    population: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dates, counts = date_bootstrap_counts(population["signal_date"])
    point = _point_metrics(population)
    nonloss_auc = _weighted_auc_vector(
        population, "nonloss", "board3_only_score", dates, counts
    )
    pair_num, pair_den = _pair_date_components(
        population, dates, "board3_only_score", "nonloss", "loss"
    )
    pair_den_draw = counts @ pair_den
    pair_values = np.divide(
        counts @ pair_num, pair_den_draw,
        out=np.full(len(counts), np.nan), where=pair_den_draw > 0,
    )
    bottom_loss = _capture_vector(
        counts, population, dates, "bottom_half_flag", "loss"
    )
    bottom_winner = _capture_vector(
        counts, population, dates, "bottom_half_flag", "target7"
    )
    worst_loss = _capture_vector(counts, population, dates, "worst_one_flag", "loss")
    worst_winner = _capture_vector(
        counts, population, dates, "worst_one_flag", "target7"
    )
    bootstrap_values = {
        "NONLOSS_AUC": nonloss_auc,
        "PAIR_CONCORDANCE": pair_values,
        "BOTTOM_HALF_LOSS_CAPTURE": bottom_loss,
        "BOTTOM_HALF_WINNER_REMOVAL": bottom_winner,
        "BOTTOM_HALF_SELECTIVITY_GAP": bottom_loss - bottom_winner,
        "WORST_ONE_SELECTIVITY_GAP": worst_loss - worst_winner,
    }
    predicates = {
        "NONLOSS_AUC": lambda x: x > .5,
        "PAIR_CONCORDANCE": lambda x: x > .5,
        "BOTTOM_HALF_LOSS_CAPTURE": lambda x: x >= .60,
        "BOTTOM_HALF_WINNER_REMOVAL": lambda x: x <= .40,
        "BOTTOM_HALF_SELECTIVITY_GAP": lambda x: x > 0,
        "WORST_ONE_SELECTIVITY_GAP": lambda x: x > 0,
    }
    bootstrap = {
        metric: _metric_summary(values, point[metric], predicates[metric])
        for metric, values in bootstrap_values.items()
    }

    lodo_values = {metric: [] for metric in point}
    for omitted in dates:
        kept = population[~population["signal_date"].eq(omitted)]
        current = _point_metrics(kept)
        for metric, value in current.items():
            lodo_values[metric].append(value)
    lodo = {
        metric: _metric_summary(values, point[metric], predicates[metric])
        for metric, values in lodo_values.items()
    }
    rows = []
    for section, summaries in (("BOOTSTRAP", bootstrap), ("LODO", lodo)):
        for metric, summary in summaries.items():
            rows.append({"section": section, "metric": metric, **summary})
    return pd.DataFrame(rows), bootstrap, lodo


def formal_decision(context: Mapping[str, Any]) -> dict[str, Any]:
    population = context["population"]
    eligible = context["eligible"]
    controls = context["controls"]
    pairs = context["pairs"]
    bottom = context["tail_summary"]["BOTTOM_HALF"]
    worst = context["tail_summary"]["WORST_ONE"]
    curve = context["curve_summary"]
    bootstrap = context["bootstrap"]
    lodo = context["lodo"]
    decision_surface = context["decision_surface_summary"]
    gates = {
        "A": (
            len(population) == 19 and population["signal_date"].nunique() == 8
            and len(eligible) >= 10 and eligible["signal_date"].nunique() >= 5
        ),
        "B": (
            controls["board3_nonloss_auc"] >= .65
            and pairs["board3_nonloss"]["informative_pairs"] >= 10
            and pairs["board3_nonloss"]["concordance"] >= .75
        ),
        "C": bottom["flagged_loss_rate"] >= bottom["eligible_loss_rate"] + .15,
        "D": bottom["loss_capture_rate"] >= .60,
        "E": bottom["winner_removal_rate"] <= .40,
        "F": bottom["selectivity_gap"] >= .25,
        "G": (
            worst["selectivity_gap"] > 0
            and worst["flagged_loss"] > worst["flagged_target7"]
        ),
        "H": curve["selectivity_dominance_rate"] >= .70,
        "I": bootstrap["BOTTOM_HALF_SELECTIVITY_GAP"]["direction_probability"] >= .80,
        "J": lodo["BOTTOM_HALF_SELECTIVITY_GAP"]["direction_probability"] >= .75,
        "K": (
            bottom["loss_capture_rate"] > bottom["pnt_removal_rate"]
            or bottom["flagged_loss_rate"] > bottom["flagged_pnt_rate"]
        ),
    }
    decision_surface_gate: bool | None
    if decision_surface["support"] == "WEAK":
        decision_surface_gate = None
    else:
        decision_surface_gate = bool(decision_surface["nonloss_auc"] > .55)
    support = gates["A"]
    strong = all(gates.values()) and decision_surface_gate is not False
    if not support:
        signal = "INSUFFICIENT_EVIDENCE"
    elif strong:
        signal = "STRONG"
    elif (
        bottom["selectivity_gap"] <= 0
        or bottom["loss_enrichment"] <= 0
        or bottom["loss_capture_rate"] <= bottom["winner_removal_rate"]
    ):
        signal = "ABSENT"
    else:
        signal = "PARTIAL"
    action = {
        "STRONG": "LIMITED_BOARD3_RISK_POLICY_DESIGN_WARRANTED",
        "PARTIAL": "NO_BOARD3_RISK_POLICY_YET",
        "ABSENT": "STOP_BOARD3_RISK_ROUTE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_DATA",
        "INVALID": "INVALID",
    }[signal]
    return {
        "gates": gates, "decision_surface_gate": decision_surface_gate,
        "signal": signal, "next_action": action,
    }


def _pct(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    population = context["population"]
    eligible = context["eligible"]
    controls = context["controls"]
    pairs = context["pairs"]
    geometry = context["geometry"]
    absolute = context["absolute_score_geometry"]
    bottom = context["tail_summary"]["BOTTOM_HALF"]
    worst = context["tail_summary"]["WORST_ONE"]
    curve = context["curve_summary"]
    ds = context["decision_surface_summary"]
    decision = context["decision"]
    lines = [
        "# v004c Board3-Only OOF Risk Selectivity Diagnostic v001", "",
        "## 1. Experimental Contract", "",
        "- Post-hoc OOF risk-selectivity diagnostic: **YES**.",
        "- Existing locked score only; no fit, new model, feature, score, threshold search, veto, backfill, Board2 modification, cross-board combination, or July result access.", "",
        "## 2. Existing OOF Prediction Lock", "",
        f"- Expected / observed SHA256: `{EXPECTED_LOCK_SHA256}` / `{context['lock_audit']['observed_lock_sha256']}`.",
        "- Prediction-lock parity and outcome perturbation: **PASS**.", "",
        "## 3. Population / Support", "",
        f"- Population A: **{population.signal_date.nunique()} dates / {len(population)} rows**; Target7/PNT/LOSS: **{int(population.target7.sum())}/{int(population.positive_non_target.sum())}/{int(population.loss.sum())}**.",
        f"- Population B: **{eligible.signal_date.nunique()} dates / {len(eligible)} rows**; Target7/PNT/LOSS: **{int(eligible.target7.sum())}/{int(eligible.positive_non_target.sum())}/{int(eligible.loss.sum())}**.", "",
        "## 4. Known Risk-Ordering Controls", "",
        f"- Unified / Board3-only NONLOSS AUC: **{_num(controls['unified_nonloss_auc'])} / {_num(controls['board3_nonloss_auc'])}**.",
        f"- Same-date NONLOSS-vs-LOSS: **{pairs['board3_nonloss']['informative_pairs']} pairs**; unified / Board3-only concordance **{_num(pairs['unified_nonloss']['concordance'])} / {_num(pairs['board3_nonloss']['concordance'])}**.", "",
        "- Locked Board3-only score by outcome (mean / median / P25 / P75):",
        f"  - LOSS: **{_num(absolute['LOSS']['mean'])} / {_num(absolute['LOSS']['median'])} / {_num(absolute['LOSS']['p25'])} / {_num(absolute['LOSS']['p75'])}**.",
        f"  - PNT: **{_num(absolute['PNT']['mean'])} / {_num(absolute['PNT']['median'])} / {_num(absolute['PNT']['p25'])} / {_num(absolute['PNT']['p75'])}**.",
        f"  - Target7: **{_num(absolute['TARGET7']['mean'])} / {_num(absolute['TARGET7']['median'])} / {_num(absolute['TARGET7']['p25'])} / {_num(absolute['TARGET7']['p75'])}**.", "",
        "## 5. LOSS / PNT / TARGET7 Risk-Rank Geometry", "",
        "| Outcome | Mean risk pct | Median | P25 | P75 |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("LOSS", "PNT", "TARGET7"):
        row = geometry[name]
        lines.append(f"| {name} | {_pct(row['mean'])} | {_pct(row['median'])} | {_pct(row['p25'])} | {_pct(row['p75'])} |")
    lines += [
        "", f"- `RISK_MEDIAN_ORDER = {geometry['risk_median_order']}`.",
        f"- Mixed-date favorable/unfavorable/tie: **{geometry['date_direction']['favorable']}/{geometry['date_direction']['unfavorable']}/{geometry['date_direction']['ties']}**.", "",
        "## 6. Bottom-Half Tail Selectivity", "",
        f"- Flagged **{bottom['flagged_rows']}**: LOSS/PNT/Target7 **{bottom['flagged_loss']}/{bottom['flagged_pnt']}/{bottom['flagged_target7']}**.",
        f"- Unflagged **{bottom['unflagged_rows']}**: LOSS/PNT/Target7 **{bottom['unflagged_loss']}/{bottom['unflagged_pnt']}/{bottom['unflagged_target7']}**.",
        f"- LOSS enrichment **{_pct(bottom['loss_enrichment'])}**; loss capture **{_pct(bottom['loss_capture_rate'])}**; winner removal **{_pct(bottom['winner_removal_rate'])}**; selectivity gap **{_pct(bottom['selectivity_gap'])}**.", "",
        "## 7. Worst-One Corroboration", "",
        f"- Flagged **{worst['flagged_rows']}**: LOSS/PNT/Target7 **{worst['flagged_loss']}/{worst['flagged_pnt']}/{worst['flagged_target7']}**.",
        f"- Loss capture **{_pct(worst['loss_capture_rate'])}**; winner removal **{_pct(worst['winner_removal_rate'])}**; selectivity gap **{_pct(worst['selectivity_gap'])}**.", "",
        "## 8. Cumulative Loss-Capture vs Winner-Removal Curve", "",
        f"- Steps **{curve['removal_steps']}**; mean selectivity gap **{_pct(curve['mean_selectivity_gap'])}**; dominance **{_pct(curve['selectivity_dominance_rate'])}**.",
        f"- Descriptive pattern: **{curve['early_removal_pattern']}**. No cutoff was selected.", "",
        "## 9. Same-Date Pair Selectivity", "",
        f"- NONLOSS-vs-LOSS: **{pairs['board3_nonloss']['informative_dates']} dates / {pairs['board3_nonloss']['informative_pairs']} pairs / {_num(pairs['board3_nonloss']['concordance'])}**.",
        f"- PNT-vs-LOSS: **{pairs['pnt_loss']['informative_dates']} dates / {pairs['pnt_loss']['informative_pairs']} pairs / {_num(pairs['pnt_loss']['concordance'])}**.",
        f"- Target7-vs-LOSS: **{pairs['target7_loss']['informative_dates']} dates / {pairs['target7_loss']['informative_pairs']} pairs / {_num(pairs['target7_loss']['concordance'])}**.", "",
        f"- `WINNER_LOSS_PAIR_SUPPORT_VERY_WEAK = {'YES' if context['winner_loss_pair_support_very_weak'] else 'NO'}`.", "",
        "## 10. Unified Stage1 Top3 Decision Surface", "",
        f"- Rows/dates: **{ds['rows']}/{ds['dates']}**; Target7/PNT/LOSS **{ds['target7']}/{ds['pnt']}/{ds['loss']}**.",
        f"- NONLOSS AUC **{_num(ds['nonloss_auc'])}**; Target7-vs-LOSS AUC **{_num(ds['target7_loss_auc'])}**; support **{ds['support']}**; corroboration **{ds['corroboration']}**.", "",
        "## 11. Bootstrap / LODO", "",
        f"- Bootstrap P(bottom-half gap > 0): **{_pct(context['bootstrap']['BOTTOM_HALF_SELECTIVITY_GAP']['direction_probability'])}**; 95% interval **[{_pct(context['bootstrap']['BOTTOM_HALF_SELECTIVITY_GAP']['p2.5'])}, {_pct(context['bootstrap']['BOTTOM_HALF_SELECTIVITY_GAP']['p97.5'])}]**.",
        f"- LODO bottom-half positive gap: **{_pct(context['lodo']['BOTTOM_HALF_SELECTIVITY_GAP']['direction_probability'])}**.", "",
        "## 12. Risk-Selectivity Decision", "",
        "| Gate | Pass |", "|---|:---:|",
    ]
    for gate, passed in decision["gates"].items():
        lines.append(f"| {gate} | {'YES' if passed else 'NO'} |")
    lines += [
        "", f"- Decision-surface gate: **{'WEAK_SUPPORT' if decision['decision_surface_gate'] is None else ('YES' if decision['decision_surface_gate'] else 'NO')}**.",
        f"- `BOARD3_RISK_SELECTIVITY_SIGNAL = {decision['signal']}`", "",
        "## 13. Next Architecture Decision", "",
        f"- `NEXT_BOARD3_RISK_ACTION = {decision['next_action']}`",
        "- No threshold was selected, no veto/backfill or combined Top3 was simulated, and Board2 was not modified.",
    ]
    if decision["signal"] == "STRONG":
        lines.append("- Recommended action: in a separate May/June-only task, design one limited Board3 negative-authority risk policy while preserving Stage1 selection and Stage1-order replacement.")
    elif decision["signal"] == "PARTIAL":
        lines.append("- Recommended action: preserve this diagnostic; do not design a Board3 risk policy or select a threshold yet.")
    elif decision["signal"] == "ABSENT":
        lines.append("- Recommended action: stop the Board3 risk route; do not train another risk model.")
    else:
        lines.append("- Recommended action: preserve the descriptive audit; relative-rank support is insufficient.")
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    population, lock_audit = load_locked_population(root)
    if not outcome_perturbation_is_immutable(population):
        raise RuntimeError("FATAL: outcome perturbation changed locked risk surface")
    eligible = population[population["relative_rank_eligible"]].copy()
    controls = {
        "unified_target7_auc": binary_auc(population["target7"], population["unified_stage1_score"]),
        "board3_target7_auc": binary_auc(population["target7"], population["board3_only_score"]),
        "unified_nonloss_auc": binary_auc(population["nonloss"], population["unified_stage1_score"]),
        "board3_nonloss_auc": binary_auc(population["nonloss"], population["board3_only_score"]),
    }
    pairs = build_pair_diagnostics(population)
    if not np.isclose(controls["unified_target7_auc"], .4714285714285714, atol=1e-12):
        raise RuntimeError("FATAL: unified Target7 AUC parity changed")
    if not np.isclose(controls["board3_target7_auc"], .45714285714285713, atol=1e-12):
        raise RuntimeError("FATAL: Board3-only Target7 AUC parity changed")
    if not np.isclose(controls["unified_nonloss_auc"], .23076923076923078, atol=1e-12):
        raise RuntimeError("FATAL: unified NONLOSS AUC parity changed")
    if not np.isclose(controls["board3_nonloss_auc"], .6923076923076923, atol=1e-12):
        raise RuntimeError("FATAL: Board3-only NONLOSS AUC parity changed")
    if pairs["board3_nonloss"]["informative_pairs"] != 14:
        raise RuntimeError("FATAL: NONLOSS-vs-LOSS pair support changed")
    if not np.isclose(pairs["unified_nonloss"]["concordance"], 3 / 14, atol=1e-12):
        raise RuntimeError("FATAL: unified pair concordance parity changed")
    if not np.isclose(pairs["board3_nonloss"]["concordance"], 13 / 14, atol=1e-12):
        raise RuntimeError("FATAL: Board3-only pair concordance parity changed")

    geometry = build_risk_geometry(eligible)
    absolute_score_geometry = build_absolute_score_geometry(population)
    bottom = tail_selectivity(eligible, "bottom_half_flag", "BOTTOM_HALF")
    worst = tail_selectivity(eligible, "worst_one_flag", "WORST_ONE")
    tail_table = pd.DataFrame([bottom, worst])
    curve, curve_summary = build_selectivity_curve(eligible)
    daily = build_daily(population)
    decision_surface, decision_surface_summary = build_decision_surface(population)
    robustness, bootstrap, lodo = build_robustness(population)
    context: dict[str, Any] = {
        "population": population, "eligible": eligible, "lock_audit": lock_audit,
        "controls": controls, "pairs": pairs, "geometry": geometry,
        "absolute_score_geometry": absolute_score_geometry,
        "tail_summary": {"BOTTOM_HALF": bottom, "WORST_ONE": worst},
        "tail_table": tail_table, "curve": curve, "curve_summary": curve_summary,
        "daily": daily, "decision_surface": decision_surface,
        "decision_surface_summary": decision_surface_summary,
        "robustness": robustness, "bootstrap": bootstrap, "lodo": lodo,
        "outcome_perturbation": "PASS", "july_result_rows_accessed": 0,
        "winner_loss_pair_support_very_weak": pairs["target7_loss"]["informative_pairs"] < 5,
    }
    context["decision"] = formal_decision(context)
    review = render_review(context)
    population_columns = [
        "event_id", "signal_date", "code", "board3_candidate_count",
        "board3_only_score", "board3_only_internal_rank", "risk_rank_percentile",
        "unified_stage1_score", "unified_stage1_rank", "unified_stage1_top3",
        "target7", "loss", "positive_non_target", "nonloss", "raw_repair_return",
        "capped_return_7", "relative_rank_eligible", "bottom_half_flag", "worst_one_flag",
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(population[population_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(tail_table),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(curve),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(decision_surface),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(robustness),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_board3_oof_risk_selectivity(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path
        / "reports/research/v004c_board3_oof_risk_selectivity_v001_20260612_20260625"
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

"""Frozen-S2 Target7-vs-LOSS failure-attribution audit.

This module never fits a model.  It reconstructs the already saved
``S2_NO_TAIL_L2_010`` full-development score from its archived intercept and
seven coefficients, then performs only the preregistered descriptive
diagnostics.  August signal-date outcomes are neither loaded nor exported.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_reduced7f_stage1_temporal_validation import (
    AUGUST_ASOF,
    REDUCED_FEATURE_COLUMNS,
    _binary_auc,
    _sigmoid,
    load_bridge,
)


TASK_NAME = "v004c_s2_winner_vs_loss_failure_attribution_v001"
OUTPUT_DIRNAME = "v004c_s2_winner_vs_loss_failure_attribution_v001_20260506_20260731"
OUTPUT_FILENAMES = (
    "v004c_s2_rankwise_failure_summary_v001.csv",
    "v004c_s2_replaceable_loss_audit_v001.csv",
    "v004c_s2_t7_loss_pairs_v001.csv",
    "v004c_s2_hard_false_positive_loss_v001.csv",
    "v004c_s2_missed_target7_v001.csv",
    "v004c_s2_feature_t7_loss_information_v001.csv",
    "v004c_s2_raw_vs_rank_information_v001.csv",
    "v004c_s2_contribution_attribution_v001.csv",
    "v004c_s2_wrong_pair_feature_gap_v001.csv",
    "v004c_s2_geometry_overlap_v001.csv",
    "v004c_s2_winner_loss_failure_review_v001.md",
)

S2_SPEC = "S2_NO_TAIL_L2_010"
S2_L2 = 0.10
S2_POSITIVE_WEIGHT = 1.50
S2_TAIL_WEIGHTING = "NONE"
S2_ARTIFACT_DIR = (
    "reports/research/"
    "v004c_reduced7f_training_spec_sanity_v001_20260506_20260731"
)
S2_COEFFICIENT_ARTIFACT = f"{S2_ARTIFACT_DIR}/v004c_reduced7f_spec_coefficients_v001.csv"
S2_FIT_ARTIFACT = f"{S2_ARTIFACT_DIR}/v004c_reduced7f_spec_training_fit_v001.csv"
S2_TEMPORAL_PREDICTION_ARTIFACT = (
    f"{S2_ARTIFACT_DIR}/v004c_reduced7f_spec_fold_predictions_v001.csv"
)

RAW_PARENT_COLUMNS: Mapping[str, str] = {
    "rank_d1_close_ma10_pct": "raw__d1_close_ma10_pct",
    "rank_d1_low_ma10_pct": "raw__d1_low_ma10_pct",
    "rank_trend_hold_score": "raw__trend_hold_score",
    "rank_theme_score": "raw__theme_score",
    "rank_log_candidate_base_price": "raw__candidate_base_price",
    "rank_active_money_score": "raw__active_money_score",
    "rank_d1_close_vwap_pct": "raw__d1_close_vwap_pct",
}

PERIODS: Mapping[str, tuple[str, str]] = {
    "MAY": ("2026-05-01", "2026-06-01"),
    "JUNE": ("2026-06-01", "2026-07-01"),
    "JULY_MATURE_ONLY": ("2026-07-01", "2026-08-01"),
    "MAY_JUNE_JULY_MATURE": ("2026-05-01", "2026-08-01"),
}
MONTH_PERIODS = ("MAY", "JUNE", "JULY_MATURE_ONLY")
SCOPES = ("ALL_CANDIDATES", "BOARD2_ONLY", "BOARD3_ONLY")


def assert_experiment_contract() -> None:
    expected = (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    )
    if tuple(REDUCED_FEATURE_COLUMNS) != expected:
        raise RuntimeError("FATAL: frozen S2 seven-feature order changed")
    if tuple(RAW_PARENT_COLUMNS) != expected:
        raise RuntimeError("FATAL: raw-parent audit is not exact 7F")
    if (S2_L2, S2_POSITIVE_WEIGHT, S2_TAIL_WEIGHTING) != (0.10, 1.50, "NONE"):
        raise RuntimeError("FATAL: frozen S2 training specification changed")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _period_slice(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    return frame[frame["signal_date"].ge(start) & frame["signal_date"].lt(end)].copy()


def _scope_slice(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "ALL_CANDIDATES":
        return frame.copy()
    if scope == "BOARD2_ONLY":
        return frame[frame["board_group"].eq("BOARD2")].copy()
    if scope == "BOARD3_ONLY":
        return frame[frame["board_group"].eq("BOARD3")].copy()
    raise ValueError(scope)


def _outcome_class(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [frame["target7"].astype(bool), frame["loss"].astype(bool)],
            ["TARGET7", "LOSS"],
            default="PNT",
        ),
        index=frame.index,
    )


def _rank_s2(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, day in frame.groupby("signal_date", sort=True):
        ordered = day.sort_values(
            ["s2_score", "event_id"], ascending=[False, True], kind="mergesort"
        ).copy()
        ordered["s2_rank"] = np.arange(1, len(ordered) + 1)
        pieces.append(ordered)
    if not pieces:
        return frame.assign(s2_rank=pd.Series(dtype=int))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["signal_date", "s2_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def load_frozen_s2(root: str | Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load bridge rows and saved S2 coefficients; never invoke a learner."""
    assert_experiment_contract()
    root_path = Path(root).resolve()
    bridge, _ = load_bridge(root_path)
    mature = bridge[bridge["label_available_date"].lt(AUGUST_ASOF)].copy()
    if (len(mature), mature["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: mature May-July population parity failed")
    if mature["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal-date row reached diagnostic")
    required_outcomes = (
        "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
    )
    if mature[list(required_outcomes)].isna().any().any():
        raise RuntimeError("FATAL: mature outcome missing")
    if mature[list(REDUCED_FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("FATAL: frozen S2 feature missing")
    raw_parent_coverage = {
        column: float(mature[column].notna().mean())
        for column in RAW_PARENT_COLUMNS.values()
    }

    coefficients = pd.read_csv(root_path / S2_COEFFICIENT_ARTIFACT, encoding="utf-8-sig")
    frozen = coefficients[
        coefficients["stage"].eq("FULL_DEVELOPMENT_IN_SAMPLE")
        & coefficients["prediction_set"].eq("FULL_DEVELOPMENT_IN_SAMPLE")
        & coefficients["spec"].eq(S2_SPEC)
    ].copy()
    expected_rows = {"__INTERCEPT__", *REDUCED_FEATURE_COLUMNS}
    if len(frozen) != 8 or set(frozen["feature"]) != expected_rows:
        raise RuntimeError("FATAL: archived S2 coefficient manifest mismatch")
    if not frozen["l2"].eq(S2_L2).all():
        raise RuntimeError("FATAL: archived S2 L2 mismatch")
    if not frozen["positive_weight"].eq(S2_POSITIVE_WEIGHT).all():
        raise RuntimeError("FATAL: archived S2 positive weight mismatch")
    if not frozen["tail_weighting"].eq(S2_TAIL_WEIGHTING).all():
        raise RuntimeError("FATAL: archived S2 tail setting mismatch")
    by_feature = frozen.set_index("feature")["coefficient"]
    beta = np.asarray(
        [float(by_feature.loc["__INTERCEPT__"])]
        + [float(by_feature.loc[name]) for name in REDUCED_FEATURE_COLUMNS],
        dtype=float,
    )
    x = mature[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float)
    mature["s2_logit"] = beta[0] + x @ beta[1:]
    mature["s2_score"] = _sigmoid(mature["s2_logit"].to_numpy(float))
    mature["outcome_class"] = _outcome_class(mature)
    mature = _rank_s2(mature)

    fit = pd.read_csv(root_path / S2_FIT_ARTIFACT, encoding="utf-8-sig")
    fit_row = fit[fit["spec"].eq(S2_SPEC)]
    if len(fit_row) != 1:
        raise RuntimeError("FATAL: archived S2 fit identity missing")
    fit_row = fit_row.iloc[0]
    score_checks = {
        "min": abs(float(mature["s2_score"].min()) - float(fit_row["score_min"])),
        "max": abs(float(mature["s2_score"].max()) - float(fit_row["score_max"])),
        "std": abs(float(mature["s2_score"].std(ddof=0)) - float(fit_row["score_std"])),
    }
    if max(score_checks.values()) > 1e-12:
        raise RuntimeError(f"FATAL: S2 score reconstruction parity failed: {score_checks}")

    temporal = pd.read_csv(
        root_path / S2_TEMPORAL_PREDICTION_ARTIFACT,
        encoding="utf-8-sig", dtype={"event_id": str, "code": str},
    )
    temporal_s2 = temporal[temporal["spec"].eq(S2_SPEC)]
    if set(temporal_s2["prediction_set"]) != {
        "FOLD_A_MAY_TO_JUNE", "FOLD_B_MAY_JUNE_TO_JULY", "EXPANDING_OOF"
    }:
        raise RuntimeError("FATAL: archived temporal S2 prediction identity mismatch")
    audit = {
        "rows": len(mature),
        "dates": mature["signal_date"].nunique(),
        "august_signal_date_outcome_rows_accessed": 0,
        "model_refits": 0,
        "score_reconstruction_max_summary_error": max(score_checks.values()),
        "coefficient_artifact": S2_COEFFICIENT_ARTIFACT,
        "temporal_prediction_artifact": S2_TEMPORAL_PREDICTION_ARTIFACT,
        "temporal_s2_rows": len(temporal_s2),
        "raw_parent_coverage": raw_parent_coverage,
    }
    return mature, beta, audit


def _daily_universe(day: pd.DataFrame) -> dict[str, float]:
    return {
        "target7_rate": float(day["target7"].mean()),
        "pnt_rate": float(day["positive_non_target"].mean()),
        "loss_rate": float(day["loss"].mean()),
        "severe_loss_rate": float(day["severe_loss"].mean()),
        "capped_return": float(day["capped_return_7"].mean()),
    }


def build_rankwise_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        period_frame = _period_slice(scored, period)
        for scope in SCOPES:
            part = _scope_slice(period_frame, scope)
            if part.empty:
                continue
            part = _rank_s2(part.drop(columns=["s2_rank"], errors="ignore"))
            daily_base = [_daily_universe(day) for _, day in part.groupby("signal_date", sort=True)]
            base = pd.DataFrame(daily_base)
            counts = part.groupby("signal_date").size()
            common: dict[str, Any] = {
                "period": period,
                "population_scope": scope,
                "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                "dates": part["signal_date"].nunique(),
                "candidate_rows": len(part),
                "candidate_count_per_day_mean": float(counts.mean()),
                "candidate_count_per_day_min": int(counts.min()),
                "candidate_count_per_day_max": int(counts.max()),
                "universe_target7_rate": float(base["target7_rate"].mean()),
                "universe_pnt_rate": float(base["pnt_rate"].mean()),
                "universe_loss_rate": float(base["loss_rate"].mean()),
                "universe_severe_loss_rate": float(base["severe_loss_rate"].mean()),
                "universe_capped_return": float(base["capped_return"].mean()),
            }
            daily_oracle: list[float] = []
            for _, day in part.groupby("signal_date", sort=True):
                oracle = day.sort_values(
                    ["capped_return_7", "event_id"],
                    ascending=[False, True], kind="mergesort",
                ).head(min(3, len(day)))
                daily_oracle.append(float(oracle["capped_return_7"].mean()))
            oracle_top3 = float(np.mean(daily_oracle))
            for selection in ("UNIVERSE", "RANK1", "RANK2", "RANK3", "TOP3"):
                daily: list[dict[str, Any]] = []
                for date, day in part.groupby("signal_date", sort=True):
                    ordered = day.sort_values(["s2_rank", "event_id"], kind="mergesort")
                    if selection == "UNIVERSE":
                        chosen = ordered
                    elif selection.startswith("RANK"):
                        chosen = ordered[ordered["s2_rank"].eq(int(selection[-1]))]
                    else:
                        chosen = ordered.head(min(3, len(ordered)))
                    if chosen.empty:
                        continue
                    daily.append({
                        "signal_date": date,
                        "rows": len(chosen),
                        "target7_rate": float(chosen["target7"].mean()),
                        "pnt_rate": float(chosen["positive_non_target"].mean()),
                        "loss_rate": float(chosen["loss"].mean()),
                        "severe_loss_rate": float(chosen["severe_loss"].mean()),
                        "capped_return": float(chosen["capped_return_7"].mean()),
                    })
                values = pd.DataFrame(daily)
                rows.append({
                    **common,
                    "selection": selection,
                    "selection_dates": values["signal_date"].nunique(),
                    "selected_rows": int(values["rows"].sum()),
                    "target7_rate": float(values["target7_rate"].mean()),
                    "pnt_rate": float(values["pnt_rate"].mean()),
                    "loss_rate": float(values["loss_rate"].mean()),
                    "severe_loss_rate": float(values["severe_loss_rate"].mean()),
                    "capped_return": float(values["capped_return"].mean()),
                    "same_date_universe_capped": float(base["capped_return"].mean()),
                    "excess_vs_universe": float(values["capped_return"].mean() - base["capped_return"].mean()),
                    "oracle_top3_capped": oracle_top3 if selection == "TOP3" else math.nan,
                    "oracle_gap_remaining": oracle_top3 - float(values["capped_return"].mean()) if selection == "TOP3" else math.nan,
                })
    return pd.DataFrame(rows).sort_values(
        ["period", "population_scope", "selection"], kind="mergesort"
    ).reset_index(drop=True)


def build_replaceable_loss(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for date, day in scored.groupby("signal_date", sort=True):
        ordered = day.sort_values(["s2_rank", "event_id"], kind="mergesort")
        top3 = ordered.head(min(3, len(ordered)))
        losses = top3[top3["loss"].astype(bool)]
        outside = ordered[ordered["s2_rank"].gt(3) & ordered["target7"].astype(bool)]
        loss_with_any = int(len(losses) if len(outside) else 0)
        theoretical = int(min(len(losses), len(outside)))
        month = str(date)[:7]
        daily_rows.append({
            "row_type": "DAILY",
            "period": month,
            "signal_date": date,
            "candidate_count": len(ordered),
            "top3_loss_slots": len(losses),
            "outside_target7_available": len(outside),
            "loss_slots_with_any_outside_target7": loss_with_any,
            "theoretically_replaceable_loss_slots": theoretical,
            "theoretical_replaceable_ratio": theoretical / len(losses) if len(losses) else math.nan,
            "date_has_replaceable_loss": int(theoretical > 0),
        })
    daily = pd.DataFrame(daily_rows)
    rows.extend(daily_rows)
    summaries = {
        "MAY": daily[daily["period"].eq("2026-05")],
        "JUNE": daily[daily["period"].eq("2026-06")],
        "JULY_MATURE_ONLY": daily[daily["period"].eq("2026-07")],
        "MAY_JUNE_JULY_MATURE": daily,
    }
    for period, part in summaries.items():
        total_losses = int(part["top3_loss_slots"].sum())
        theoretical = int(part["theoretically_replaceable_loss_slots"].sum())
        rows.append({
            "row_type": "SUMMARY",
            "period": period,
            "signal_date": "",
            "candidate_count": int(part["candidate_count"].sum()),
            "top3_loss_slots": total_losses,
            "outside_target7_available": int(part["outside_target7_available"].sum()),
            "loss_slots_with_any_outside_target7": int(part["loss_slots_with_any_outside_target7"].sum()),
            "theoretically_replaceable_loss_slots": theoretical,
            "theoretical_replaceable_ratio": theoretical / total_losses if total_losses else math.nan,
            "date_has_replaceable_loss": int(part["date_has_replaceable_loss"].sum()),
        })
    return pd.DataFrame(rows)


def _pair_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in scored.groupby("signal_date", sort=True):
        winners = day[day["target7"].astype(bool)].sort_values("event_id", kind="mergesort")
        losses = day[day["loss"].astype(bool)].sort_values("event_id", kind="mergesort")
        for _, winner in winners.iterrows():
            for _, loss in losses.iterrows():
                margin = float(winner["s2_score"] - loss["s2_score"])
                rows.append({
                    "row_type": "PAIR",
                    "period": str(date)[:7],
                    "signal_date": date,
                    "target7_event_id": winner["event_id"],
                    "loss_event_id": loss["event_id"],
                    "target7_code": winner["code"],
                    "loss_code": loss["code"],
                    "target7_board_group": winner["board_group"],
                    "loss_board_group": loss["board_group"],
                    "both_board2": int(winner["board_group"] == loss["board_group"] == "BOARD2"),
                    "both_board3": int(winner["board_group"] == loss["board_group"] == "BOARD3"),
                    "target7_s2_score": float(winner["s2_score"]),
                    "loss_s2_score": float(loss["s2_score"]),
                    "score_margin": margin,
                    "s2_pair_correct": int(margin > 0),
                    "s2_pair_wrong": int(margin < 0),
                    "s2_pair_tie": int(margin == 0),
                    "total_pairs": math.nan,
                    "correct_pair_rate": math.nan,
                    "wrong_pair_rate": math.nan,
                    "tie_rate": math.nan,
                    "mean_score_margin": math.nan,
                    "median_score_margin": math.nan,
                    "t7_loss_pair_concordance": math.nan,
                })
    return pd.DataFrame(rows)


def build_pairs(scored: pd.DataFrame) -> pd.DataFrame:
    pairs = _pair_rows(scored)
    rows = pairs.to_dict("records")
    period_parts = {
        "MAY": pairs[pairs["period"].eq("2026-05")],
        "JUNE": pairs[pairs["period"].eq("2026-06")],
        "JULY_MATURE_ONLY": pairs[pairs["period"].eq("2026-07")],
        "MAY_JUNE_JULY_MATURE": pairs,
    }
    for period, part in period_parts.items():
        concordance = float((part["s2_pair_correct"] + .5 * part["s2_pair_tie"]).mean()) if len(part) else math.nan
        rows.append({
            "row_type": "SUMMARY", "period": period, "signal_date": "",
            "target7_event_id": "", "loss_event_id": "", "target7_code": "", "loss_code": "",
            "target7_board_group": "", "loss_board_group": "", "both_board2": math.nan,
            "both_board3": math.nan, "target7_s2_score": math.nan, "loss_s2_score": math.nan,
            "score_margin": math.nan, "s2_pair_correct": int(part["s2_pair_correct"].sum()),
            "s2_pair_wrong": int(part["s2_pair_wrong"].sum()), "s2_pair_tie": int(part["s2_pair_tie"].sum()),
            "total_pairs": len(part),
            "correct_pair_rate": float(part["s2_pair_correct"].mean()) if len(part) else math.nan,
            "wrong_pair_rate": float(part["s2_pair_wrong"].mean()) if len(part) else math.nan,
            "tie_rate": float(part["s2_pair_tie"].mean()) if len(part) else math.nan,
            "mean_score_margin": float(part["score_margin"].mean()) if len(part) else math.nan,
            "median_score_margin": float(part["score_margin"].median()) if len(part) else math.nan,
            "t7_loss_pair_concordance": concordance,
        })
    return pd.DataFrame(rows)


def build_hard_losses(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for date, day in scored.groupby("signal_date", sort=True):
        ordered = day.sort_values(["s2_rank", "event_id"], kind="mergesort")
        outside_t7 = ordered[ordered["s2_rank"].gt(3) & ordered["target7"].astype(bool)]
        top_losses = ordered[
            ordered["s2_rank"].le(3) & ordered["loss"].astype(bool)
        ].sort_values(["s2_rank", "event_id"], ascending=[False, True], kind="mergesort")
        available_winners = outside_t7.sort_values(["s2_rank", "event_id"], kind="mergesort")
        for loss_position, (_, loss) in enumerate(top_losses.iterrows()):
            lower = outside_t7[outside_t7["s2_rank"].gt(loss["s2_rank"])]
            best = available_winners.iloc[[loss_position]] if loss_position < len(available_winners) else available_winners.iloc[0:0]
            if best.empty:
                best_id, best_rank, best_score, margin = "", math.nan, math.nan, math.nan
            else:
                candidate = best.iloc[0]
                best_id = candidate["event_id"]
                best_rank = int(candidate["s2_rank"])
                best_score = float(candidate["s2_score"])
                margin = best_score - float(loss["s2_score"])
            detail.append({
                "row_type": "LOSS_DETAIL", "period": str(date)[:7], "signal_date": date,
                "event_id": loss["event_id"], "code": loss["code"], "board_group": loss["board_group"],
                "s2_rank": int(loss["s2_rank"]), "s2_score": float(loss["s2_score"]),
                "raw_repair_return": float(loss["raw_repair_return"]),
                "capped_return_7": float(loss["capped_return_7"]), "severe_loss": int(loss["severe_loss"]),
                "outside_target7_count": len(outside_t7), "lower_ranked_target7_count": len(lower),
                "best_lower_target7_event_id": best_id, "best_lower_target7_rank": best_rank,
                "best_lower_target7_score": best_score, "target7_minus_loss_score_margin": margin,
                "loss_classification": "HARD_FALSE_POSITIVE_LOSS" if not best.empty else "UNAVOIDABLE_TOP3_LOSS",
                "count": math.nan, "severe_loss_count": math.nan, "mean_raw_return": math.nan,
                "mean_capped_return": math.nan, "mean_s2_score": math.nan,
            })
    rows.extend(detail)
    data = pd.DataFrame(detail)
    period_parts = {
        "MAY": data[data["period"].eq("2026-05")], "JUNE": data[data["period"].eq("2026-06")],
        "JULY_MATURE_ONLY": data[data["period"].eq("2026-07")],
        "MAY_JUNE_JULY_MATURE": data,
    }
    for period, part in period_parts.items():
        for rank in (1, 2, 3):
            for classification in ("ALL_TOP3_LOSS", "HARD_FALSE_POSITIVE_LOSS", "UNAVOIDABLE_TOP3_LOSS"):
                subset = part[part["s2_rank"].eq(rank)]
                if classification != "ALL_TOP3_LOSS":
                    subset = subset[subset["loss_classification"].eq(classification)]
                rows.append({
                    "row_type": "SUMMARY", "period": period, "signal_date": "", "event_id": "", "code": "",
                    "board_group": "", "s2_rank": rank, "s2_score": math.nan, "raw_repair_return": math.nan,
                    "capped_return_7": math.nan, "severe_loss": math.nan, "outside_target7_count": math.nan,
                    "lower_ranked_target7_count": math.nan, "best_lower_target7_event_id": "",
                    "best_lower_target7_rank": math.nan, "best_lower_target7_score": math.nan,
                    "target7_minus_loss_score_margin": math.nan, "loss_classification": classification,
                    "count": len(subset), "severe_loss_count": int(subset["severe_loss"].sum()) if len(subset) else 0,
                    "mean_raw_return": float(subset["raw_repair_return"].mean()) if len(subset) else math.nan,
                    "mean_capped_return": float(subset["capped_return_7"].mean()) if len(subset) else math.nan,
                    "mean_s2_score": float(subset["s2_score"].mean()) if len(subset) else math.nan,
                })
    return pd.DataFrame(rows)


def _miss_bucket(rank: int) -> str:
    if rank <= 5:
        return "4-5"
    if rank <= 7:
        return "6-7"
    if rank <= 10:
        return "8-10"
    return ">10"


def build_missed_target7(scored: pd.DataFrame) -> pd.DataFrame:
    detail: list[dict[str, Any]] = []
    for date, day in scored.groupby("signal_date", sort=True):
        ordered = day.sort_values(["s2_rank", "event_id"], kind="mergesort")
        top3_loss = ordered[ordered["s2_rank"].le(3) & ordered["loss"].astype(bool)]
        for _, winner in ordered[ordered["s2_rank"].gt(3) & ordered["target7"].astype(bool)].iterrows():
            loss_above = top3_loss[top3_loss["s2_rank"].lt(winner["s2_rank"])]
            detail.append({
                "row_type": "TARGET7_DETAIL", "period": str(date)[:7], "signal_date": date,
                "event_id": winner["event_id"], "code": winner["code"], "board_group": winner["board_group"],
                "s2_rank": int(winner["s2_rank"]), "rank_bucket": _miss_bucket(int(winner["s2_rank"])),
                "s2_score": float(winner["s2_score"]), "raw_repair_return": float(winner["raw_repair_return"]),
                "capped_return_7": float(winner["capped_return_7"]), "top3_contains_loss": int(len(top3_loss) > 0),
                "loss_above_count": len(loss_above), "missed_winner_with_loss_above": int(len(loss_above) > 0),
                "count": math.nan, "proportion": math.nan, "mean_capped_return": math.nan,
            })
    rows = list(detail)
    data = pd.DataFrame(detail)
    period_parts = {
        "MAY": data[data["period"].eq("2026-05")], "JUNE": data[data["period"].eq("2026-06")],
        "JULY_MATURE_ONLY": data[data["period"].eq("2026-07")],
        "MAY_JUNE_JULY_MATURE": data,
    }
    for period, part in period_parts.items():
        total = len(part)
        for bucket in ("4-5", "6-7", "8-10", ">10"):
            subset = part[part["rank_bucket"].eq(bucket)]
            rows.append({
                "row_type": "SUMMARY", "period": period, "signal_date": "", "event_id": "", "code": "",
                "board_group": "", "s2_rank": math.nan, "rank_bucket": bucket, "s2_score": math.nan,
                "raw_repair_return": math.nan, "capped_return_7": math.nan, "top3_contains_loss": math.nan,
                "loss_above_count": math.nan,
                "missed_winner_with_loss_above": int(subset["missed_winner_with_loss_above"].sum()) if len(subset) else 0,
                "count": len(subset), "proportion": len(subset) / total if total else math.nan,
                "mean_capped_return": float(subset["capped_return_7"].mean()) if len(subset) else math.nan,
            })
    return pd.DataFrame(rows)


def _feature_pair_stats(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    date_aucs: list[float] = []
    pair_scores: list[float] = []
    daily_gaps: list[float] = []
    pairs = 0
    for _, day in frame.groupby("signal_date", sort=True):
        winners = pd.to_numeric(day.loc[day["target7"].astype(bool), feature], errors="coerce").dropna()
        losses = pd.to_numeric(day.loc[day["loss"].astype(bool), feature], errors="coerce").dropna()
        if winners.empty or losses.empty:
            continue
        y = np.r_[np.ones(len(winners)), np.zeros(len(losses))]
        s = np.r_[winners.to_numpy(float), losses.to_numpy(float)]
        date_aucs.append(_binary_auc(y, s))
        rank_pct = pd.Series(s).rank(method="average", pct=True).to_numpy(float)
        daily_gaps.append(float(np.median(rank_pct[:len(winners)]) - np.median(rank_pct[len(winners):])))
        for winner in winners:
            diffs = winner - losses.to_numpy(float)
            pair_scores.extend(np.where(diffs > 0, 1.0, np.where(diffs < 0, 0.0, .5)).tolist())
            pairs += len(diffs)
    within = float(np.mean(date_aucs)) if date_aucs else math.nan
    return {
        "within_date_auc": within,
        "eligible_dates": len(date_aucs),
        "pairs": pairs,
        "pair_concordance": float(np.mean(pair_scores)) if pair_scores else math.nan,
        "median_within_date_rank_difference": float(np.median(daily_gaps)) if daily_gaps else math.nan,
        "direction": (
            "HIGHER_FAVORS_TARGET7" if np.isfinite(within) and within > .525
            else "LOWER_FAVORS_TARGET7" if np.isfinite(within) and within < .475
            else "NEAR_RANDOM"
        ),
    }


def _direction_stability(monthly: Sequence[float]) -> str:
    values = np.asarray([value for value in monthly if np.isfinite(value)], dtype=float)
    if len(values) < 2:
        return "NEAR_RANDOM"
    positive = values > .525
    negative = values < .475
    if positive.any() and negative.any():
        return "SIGN_FLIP"
    if (values >= .50).all() and positive.sum() >= 2:
        return "STABLE_POSITIVE"
    if (values <= .50).all() and negative.sum() >= 2:
        return "STABLE_NEGATIVE"
    return "NEAR_RANDOM"


def build_feature_information(scored: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for feature_index, feature in enumerate(REDUCED_FEATURE_COLUMNS):
            feature_rows: list[dict[str, Any]] = []
            for period in PERIODS:
                part = _scope_slice(_period_slice(scored, period), scope)
                stats = _feature_pair_stats(part, feature)
                feature_rows.append({
                    "feature": feature, "coefficient": float(beta[feature_index + 1]),
                    "period": period, "population_scope": scope,
                    "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                    "target7_rows": int(part["target7"].sum()), "loss_rows": int(part["loss"].sum()),
                    **stats,
                })
            monthly = [row["within_date_auc"] for row in feature_rows if row["period"] in MONTH_PERIODS]
            stability = _direction_stability(monthly)
            for row in feature_rows:
                row["direction_stability"] = stability
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["population_scope", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True)


def build_raw_vs_rank(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for rank_feature, raw_feature in RAW_PARENT_COLUMNS.items():
            for period in PERIODS:
                part = _scope_slice(_period_slice(scored, period), scope)
                rank_stats = _feature_pair_stats(part, rank_feature)
                raw_stats = _feature_pair_stats(part, raw_feature)
                raw_non_null = int(part[raw_feature].notna().sum())
                raw_coverage = raw_non_null / len(part) if len(part) else math.nan
                raw_strength = abs(float(raw_stats["within_date_auc"]) - .5) if np.isfinite(raw_stats["within_date_auc"]) else math.nan
                rank_strength = abs(float(rank_stats["within_date_auc"]) - .5) if np.isfinite(rank_stats["within_date_auc"]) else math.nan
                if np.isfinite(raw_strength) and raw_strength < .05:
                    state = "RAW_INFORMATION_ABSENT"
                elif np.isfinite(raw_strength) and raw_strength >= .08 and raw_strength - rank_strength >= .03:
                    state = "RAW_INFORMATION_PRESENT_BUT_RANK_TRANSFORM_LOST"
                else:
                    state = "RAW_AND_RANK_SIMILAR"
                rows.append({
                    "feature": rank_feature, "raw_parent": raw_feature, "period": period,
                    "population_scope": scope, "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                    "raw_within_date_auc": raw_stats["within_date_auc"],
                    "rank_within_date_auc": rank_stats["within_date_auc"],
                    "raw_minus_rank_auc": float(raw_stats["within_date_auc"] - rank_stats["within_date_auc"])
                    if np.isfinite(raw_stats["within_date_auc"]) and np.isfinite(rank_stats["within_date_auc"]) else math.nan,
                    "raw_pair_concordance": raw_stats["pair_concordance"],
                    "rank_pair_concordance": rank_stats["pair_concordance"],
                    "raw_non_null_rows": raw_non_null,
                    "raw_coverage": raw_coverage,
                    "raw_eligible_dates": raw_stats["eligible_dates"],
                    "rank_eligible_dates": rank_stats["eligible_dates"],
                    "information_state": state,
                })
    return pd.DataFrame(rows).sort_values(
        ["population_scope", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True)


def build_contributions(scored: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for period in PERIODS:
            part = _scope_slice(_period_slice(scored, period), scope)
            for index, feature in enumerate(REDUCED_FEATURE_COLUMNS):
                contribution = float(beta[index + 1]) * pd.to_numeric(part[feature], errors="raise")
                local = part.assign(_contribution=contribution)
                groups: dict[str, pd.Series] = {
                    name: local.loc[local["outcome_class"].eq(name), "_contribution"]
                    for name in ("TARGET7", "PNT", "LOSS")
                }
                rows.append({
                    "feature": feature, "coefficient": float(beta[index + 1]), "period": period,
                    "population_scope": scope, "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                    "target7_n": len(groups["TARGET7"]), "target7_mean_contribution": float(groups["TARGET7"].mean()),
                    "target7_median_contribution": float(groups["TARGET7"].median()),
                    "pnt_n": len(groups["PNT"]), "pnt_mean_contribution": float(groups["PNT"].mean()),
                    "pnt_median_contribution": float(groups["PNT"].median()),
                    "loss_n": len(groups["LOSS"]), "loss_mean_contribution": float(groups["LOSS"].mean()),
                    "loss_median_contribution": float(groups["LOSS"].median()),
                    "target7_minus_loss_mean_gap": float(groups["TARGET7"].mean() - groups["LOSS"].mean()),
                    "target7_minus_loss_median_gap": float(groups["TARGET7"].median() - groups["LOSS"].median()),
                    "loss_contribution_exceeds_target7": int(groups["LOSS"].mean() > groups["TARGET7"].mean()),
                })
    return pd.DataFrame(rows).sort_values(
        ["population_scope", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True)


def build_wrong_pair_gaps(scored: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    pairs = _pair_rows(scored)
    wrong = pairs[pairs["s2_pair_wrong"].eq(1)].copy()
    indexed = scored.set_index("event_id")
    if not indexed.index.is_unique:
        raise RuntimeError("FATAL: duplicate event_id reached wrong-pair audit")
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        if scope == "ALL_CANDIDATES":
            scope_pairs = wrong
        elif scope == "BOARD2_ONLY":
            scope_pairs = wrong[wrong["both_board2"].eq(1)]
        else:
            scope_pairs = wrong[wrong["both_board3"].eq(1)]
        for period in PERIODS:
            if period == "MAY_JUNE_JULY_MATURE":
                part = scope_pairs
            else:
                month = {"MAY": "2026-05", "JUNE": "2026-06", "JULY_MATURE_ONLY": "2026-07"}[period]
                part = scope_pairs[scope_pairs["period"].eq(month)]
            for index, feature in enumerate(REDUCED_FEATURE_COLUMNS):
                differences: list[float] = []
                contribution_gaps: list[float] = []
                for _, pair in part.iterrows():
                    difference = float(indexed.loc[pair["target7_event_id"], feature] - indexed.loc[pair["loss_event_id"], feature])
                    differences.append(difference)
                    contribution_gaps.append(float(beta[index + 1]) * difference)
                values = np.asarray(differences, dtype=float)
                contrib = np.asarray(contribution_gaps, dtype=float)
                rows.append({
                    "feature": feature, "coefficient": float(beta[index + 1]), "period": period,
                    "population_scope": scope, "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                    "wrong_pairs": len(values), "mean_target7_minus_loss_feature": float(values.mean()) if len(values) else math.nan,
                    "median_target7_minus_loss_feature": float(np.median(values)) if len(values) else math.nan,
                    "positive_feature_difference_proportion": float((values > 0).mean()) if len(values) else math.nan,
                    "mean_target7_minus_loss_contribution": float(contrib.mean()) if len(contrib) else math.nan,
                    "positive_contribution_gap_proportion": float((contrib > 0).mean()) if len(contrib) else math.nan,
                })
    return pd.DataFrame(rows).sort_values(
        ["population_scope", "feature", "period"], kind="mergesort"
    ).reset_index(drop=True)


def _nearest_distances(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    same = np.full(len(values), np.nan)
    opposite = np.full(len(values), np.nan)
    for index in range(len(values)):
        same_mask = (labels == labels[index]) & (np.arange(len(values)) != index)
        opposite_mask = labels != labels[index]
        if same_mask.any():
            same[index] = float(distance[index, same_mask].min())
        if opposite_mask.any():
            opposite[index] = float(distance[index, opposite_mask].min())
    return same, opposite


def build_geometry(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for period in PERIODS:
            part = _scope_slice(_period_slice(scored, period), scope)
            part = part[part["outcome_class"].isin(["TARGET7", "LOSS"])].copy()
            winners = int(part["target7"].sum())
            losses = int(part["loss"].sum())
            if not winners or not losses:
                metrics = {key: math.nan for key in (
                    "centroid_distance", "target7_within_dispersion", "loss_within_dispersion",
                    "pooled_within_dispersion", "between_within_ratio", "mean_nearest_same_distance",
                    "mean_nearest_opposite_distance", "median_nearest_opposite_distance", "nearest_opposite_overlap_rate",
                )}
            else:
                x = part[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float)
                mean = x.mean(axis=0)
                std = x.std(axis=0, ddof=0)
                std[std <= 1e-15] = 1.0
                z = (x - mean) / std
                labels = part["target7"].to_numpy(int)
                t7_centroid = z[labels == 1].mean(axis=0)
                loss_centroid = z[labels == 0].mean(axis=0)
                centroid_distance = float(np.linalg.norm(t7_centroid - loss_centroid))
                t7_disp = float(np.linalg.norm(z[labels == 1] - t7_centroid, axis=1).mean())
                loss_disp = float(np.linalg.norm(z[labels == 0] - loss_centroid, axis=1).mean())
                pooled = float((t7_disp * winners + loss_disp * losses) / (winners + losses))
                same, opposite = _nearest_distances(z, labels)
                valid_overlap = np.isfinite(same) & np.isfinite(opposite)
                metrics = {
                    "centroid_distance": centroid_distance,
                    "target7_within_dispersion": t7_disp,
                    "loss_within_dispersion": loss_disp,
                    "pooled_within_dispersion": pooled,
                    "between_within_ratio": centroid_distance / pooled if pooled > 0 else math.nan,
                    "mean_nearest_same_distance": float(np.nanmean(same)),
                    "mean_nearest_opposite_distance": float(np.nanmean(opposite)),
                    "median_nearest_opposite_distance": float(np.nanmedian(opposite)),
                    "nearest_opposite_overlap_rate": float((opposite[valid_overlap] <= same[valid_overlap]).mean()) if valid_overlap.any() else math.nan,
                }
            rows.append({
                "period": period, "population_scope": scope,
                "support_warning": "LOW_N" if scope == "BOARD3_ONLY" else "",
                "target7_rows": winners, "loss_rows": losses, "feature_count": 7,
                "standardization": "WITHIN_PERIOD_T7_AND_LOSS_ZSCORE",
                **metrics,
            })
    return pd.DataFrame(rows).sort_values(["population_scope", "period"], kind="mergesort").reset_index(drop=True)


def _summary_row(table: pd.DataFrame, period: str, scope: str, selection: str) -> pd.Series:
    match = table[
        table["period"].eq(period)
        & table["population_scope"].eq(scope)
        & table["selection"].eq(selection)
    ]
    if len(match) != 1:
        raise RuntimeError(f"FATAL: missing summary {period}/{scope}/{selection}")
    return match.iloc[0]


def determine_attribution(context: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    feature = context["feature_information"]
    pooled = feature[
        feature["period"].eq("MAY_JUNE_JULY_MATURE")
        & feature["population_scope"].eq("ALL_CANDIDATES")
    ]
    stable = pooled[pooled["direction_stability"].isin(["STABLE_POSITIVE", "STABLE_NEGATIVE"])]
    stable_informative = stable[(stable["within_date_auc"] - .5).abs().ge(.08)]
    strongest_oriented = float((pooled["within_date_auc"] - .5).abs().max() + .5)
    pair_summary = context["pairs"]
    pair_row = pair_summary[
        pair_summary["row_type"].eq("SUMMARY")
        & pair_summary["period"].eq("MAY_JUNE_JULY_MATURE")
    ].iloc[0]
    concordance = float(pair_row["t7_loss_pair_concordance"])
    replace = context["replaceable"]
    replace_row = replace[
        replace["row_type"].eq("SUMMARY")
        & replace["period"].eq("MAY_JUNE_JULY_MATURE")
    ].iloc[0]
    replace_ratio = float(replace_row["theoretical_replaceable_ratio"])
    raw = context["raw_vs_rank"]
    raw_pooled = raw[
        raw["period"].eq("MAY_JUNE_JULY_MATURE")
        & raw["population_scope"].eq("ALL_CANDIDATES")
    ]
    raw_informative = int((raw_pooled["raw_within_date_auc"].sub(.5).abs() >= .08).sum())
    geometry = context["geometry"]
    geom = geometry[
        geometry["period"].eq("MAY_JUNE_JULY_MATURE")
        & geometry["population_scope"].eq("ALL_CANDIDATES")
    ].iloc[0]
    overlap = float(geom["nearest_opposite_overlap_rate"])
    if (
        len(stable_informative) >= 2 and raw_informative >= 2
        and concordance < .55 and replace_ratio >= .30 and overlap < .50
    ):
        state = "MODEL_FORMULATION_FAILURE_DOMINANT"
    elif (
        len(stable_informative) == 0 and strongest_oriented < .58
        and raw_informative == 0 and overlap >= .50
    ):
        state = "INFORMATION_LIMIT_DOMINANT"
    else:
        state = "MIXED"
    evidence = {
        "stable_informative_feature_count": len(stable_informative),
        "raw_informative_feature_count": raw_informative,
        "strongest_oriented_feature_auc": strongest_oriented,
        "s2_pair_concordance": concordance,
        "replaceable_loss_ratio": replace_ratio,
        "nearest_opposite_overlap_rate": overlap,
    }
    return state, evidence


def render_review(context: Mapping[str, Any], state: str) -> str:
    if state not in {"MODEL_FORMULATION_FAILURE_DOMINANT", "INFORMATION_LIMIT_DOMINANT", "MIXED"}:
        raise RuntimeError("FATAL: invalid attribution state")
    rankwise = context["rankwise"]
    pooled_universe = _summary_row(rankwise, "MAY_JUNE_JULY_MATURE", "ALL_CANDIDATES", "UNIVERSE")
    rank1 = _summary_row(rankwise, "MAY_JUNE_JULY_MATURE", "ALL_CANDIDATES", "RANK1")
    rank2 = _summary_row(rankwise, "MAY_JUNE_JULY_MATURE", "ALL_CANDIDATES", "RANK2")
    rank3 = _summary_row(rankwise, "MAY_JUNE_JULY_MATURE", "ALL_CANDIDATES", "RANK3")
    replace = context["replaceable"]
    replace_row = replace[
        replace["row_type"].eq("SUMMARY") & replace["period"].eq("MAY_JUNE_JULY_MATURE")
    ].iloc[0]
    pair_row = context["pairs"][
        context["pairs"]["row_type"].eq("SUMMARY")
        & context["pairs"]["period"].eq("MAY_JUNE_JULY_MATURE")
    ].iloc[0]
    info = context["feature_information"]
    pooled_info = info[
        info["period"].eq("MAY_JUNE_JULY_MATURE")
        & info["population_scope"].eq("ALL_CANDIDATES")
    ]
    stable = pooled_info[pooled_info["direction_stability"].isin(["STABLE_POSITIVE", "STABLE_NEGATIVE"])]
    raw = context["raw_vs_rank"]
    raw_pooled = raw[
        raw["period"].eq("MAY_JUNE_JULY_MATURE")
        & raw["population_scope"].eq("ALL_CANDIDATES")
    ]
    raw_lost = int(raw_pooled["information_state"].eq("RAW_INFORMATION_PRESENT_BUT_RANK_TRANSFORM_LOST").sum())
    contrib = context["contributions"]
    contrib_pooled = contrib[
        contrib["period"].eq("MAY_JUNE_JULY_MATURE")
        & contrib["population_scope"].eq("ALL_CANDIDATES")
    ]
    loss_favored = contrib_pooled.loc[
        contrib_pooled["loss_contribution_exceeds_target7"].eq(1), "feature"
    ].tolist()
    questions = (
        ("Q1. 候选池里是否有足够 Target7 可供排序？",
         f"有。485 行、60 个日期的成熟候选池中，date-equal Target7 基准率为 {pooled_universe['universe_target7_rate']:.2%}；漏在 Top3 外的 Target7 与可替换 LOSS 已按日逐项列出。"),
        ("Q2. Rank1、Rank2、Rank3分别出了什么问题？",
         f"Rank1/2/3 的 Target7 率分别为 {rank1['target7_rate']:.2%}/{rank2['target7_rate']:.2%}/{rank3['target7_rate']:.2%}，LOSS 率分别为 {rank1['loss_rate']:.2%}/{rank2['loss_rate']:.2%}/{rank3['loss_rate']:.2%}；Rank2/3 的 winner 与 LOSS 混排是主要失败结构。"),
        ("Q3. Top3中的LOSS有多少其实可以被当天漏掉的Target7替换？",
         f"按一只漏选 Target7 最多替换一个 LOSS 的容量约束，{int(replace_row['theoretically_replaceable_loss_slots'])}/{int(replace_row['top3_loss_slots'])} 个 LOSS 槽可理论替换（{replace_row['theoretical_replaceable_ratio']:.2%}），发生在 {int(replace_row['date_has_replaceable_loss'])} 个日期。"),
        ("Q4. S2 对 Target7 vs LOSS 的同日排序正确率是多少？",
         f"共 {int(pair_row['total_pairs'])} 个同日 T7–LOSS pairs，正确率/含 tie 的 concordance 为 {pair_row['t7_loss_pair_concordance']:.2%}，错误率为 {pair_row['wrong_pair_rate']:.2%}。"),
        ("Q5. 现有7F本身是否包含稳定的 Target7-vs-LOSS 信息？",
         f"只有 {len(stable)}/7 个 feature 达到跨月同方向标记；逐月与 pooled 的原始方向、pair 支持均已保留，不能把 pooled 单点当作稳定信息。"),
        ("Q6. raw信息比 daily-rank 信息是否明显更强？",
         f"没有普遍更强。7 个 pooled raw/rank 对照中，{raw_lost} 个被标为 RAW_INFORMATION_PRESENT_BUT_RANK_TRANSFORM_LOST；其余为 raw 信息不足或 raw/rank 相近。"),
        ("Q7. S2为什么会把 LOSS 排得这么高？",
         "固定正系数组合使部分 LOSS 同时获得多项正贡献；pooled 中 LOSS 平均贡献高于 Target7 的 feature 为："
         + (", ".join(loss_favored) if loss_favored else "无")
         + "。错误 pair 的逐 feature 与 contribution gap 已单列，未据此改模。"),
        ("Q8. 当前主要瓶颈到底是 MODEL_FORMULATION、INFORMATION_LIMIT 还是 MIXED？",
         f"{state}。证据同时考虑跨月单因子方向、raw/rank 对照、固定 S2 pair concordance、可替换 LOSS 与 7F 几何重叠；本任务不授权后续模型。"),
    )
    lines = ["# v004c S2 Winner vs Loss Failure Attribution v001", "",
             "本报告是 development failure attribution；S2 分数由已保存系数重建，模型拟合次数为 0。", ""]
    for heading, answer in questions:
        lines.extend([f"## {heading}", "", answer, ""])
    lines.extend([
        f"WINNER_LOSS_FAILURE_ATTRIBUTION = {state}", "",
        "NEXT_ACTION = STOP_AND_REVIEW", "",
    ])
    return "\n".join(lines)


def analyze(root: str | Path) -> dict[str, Any]:
    scored, beta, audit = load_frozen_s2(root)
    context: dict[str, Any] = {
        "root": Path(root).resolve(), "scored": scored, "beta": beta, "audit": audit,
    }
    context["rankwise"] = build_rankwise_summary(scored)
    context["replaceable"] = build_replaceable_loss(scored)
    context["pairs"] = build_pairs(scored)
    context["hard_losses"] = build_hard_losses(scored)
    context["missed_target7"] = build_missed_target7(scored)
    context["feature_information"] = build_feature_information(scored, beta)
    context["raw_vs_rank"] = build_raw_vs_rank(scored)
    context["contributions"] = build_contributions(scored, beta)
    context["wrong_pair_gaps"] = build_wrong_pair_gaps(scored, beta)
    context["geometry"] = build_geometry(scored)
    state, evidence = determine_attribution(context)
    context["state"] = state
    context["evidence"] = evidence
    return context


def build_outputs(context: Mapping[str, Any]) -> dict[str, bytes]:
    review = render_review(context, str(context["state"]))
    return {
        OUTPUT_FILENAMES[0]: _csv_bytes(context["rankwise"]),
        OUTPUT_FILENAMES[1]: _csv_bytes(context["replaceable"]),
        OUTPUT_FILENAMES[2]: _csv_bytes(context["pairs"]),
        OUTPUT_FILENAMES[3]: _csv_bytes(context["hard_losses"]),
        OUTPUT_FILENAMES[4]: _csv_bytes(context["missed_target7"]),
        OUTPUT_FILENAMES[5]: _csv_bytes(context["feature_information"]),
        OUTPUT_FILENAMES[6]: _csv_bytes(context["raw_vs_rank"]),
        OUTPUT_FILENAMES[7]: _csv_bytes(context["contributions"]),
        OUTPUT_FILENAMES[8]: _csv_bytes(context["wrong_pair_gaps"]),
        OUTPUT_FILENAMES[9]: _csv_bytes(context["geometry"]),
        OUTPUT_FILENAMES[10]: review.encode("utf-8"),
    }


def write_outputs(context: Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
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

"""Post-hoc June probe of one predeclared Board3 contribution ablation.

The frozen V4C_STAGE1 folds are reconstructed unchanged.  No coefficient is
re-estimated for the counterfactual: for Board3 rows only, the contributions
of the three features formally established by the preceding response audit
are set to zero.  Board2 logits are bit-for-bit unchanged and all candidates
are then deterministically re-ranked by the counterfactual score.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_board2_board3_stage1_feature_response import (
    EXPECTED_FEATURE_COLUMNS,
    contribution_column,
    reconstruct_frozen_stage1,
)
from .v004c_limited_risk_protector import EXPECTED_STAGE1_PARITY, EXPECTED_STRICT_DATES
from .v004c_stage1_risk_complementarity import load_frozen_june_buckets, outcome_state
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    INITIAL_TRAIN_DATES,
    L2,
    MODEL_FAMILY,
    MODEL_ID,
    POSITIVE_WEIGHT,
    dataframe_csv_bytes,
)
from .v004c_v4a_top10_reranker import strategy_summary


TASK_NAME = "BOARD3 CONFIRMED-HARMFUL CONTRIBUTION ABLATION"
EXPECTED_STARTING_HEAD = "80383949465a725d483810e706bcc6441361b8fe"
POST_HOC_JUNE_MECHANISM_PROBE = True
NEW_MODEL = False
NEW_FEATURE = False
NEW_POLICY = False
MODEL_REFIT_FOR_ABLATION = False
NEW_COEFFICIENTS = False
SUBSET_SEARCH = False
SCALE_SEARCH = False
SIGN_REVERSAL = False
JULY_RESULT_ROWS_ACCESSED = 0
JULY_USED_FOR_ABLATION_DESIGN = False

ABLATION_FEATURES = (
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "days_since_d0_le1",
)
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_RESAMPLES = 20_000
OUTCOME_COLUMNS = frozenset({
    "raw_repair_return", "capped_return_7", "capped_opportunity_return_7",
    "target7", "loss", "nonloss", "d2_date", "d3_date", "d2_open_daily",
    "d3_high_daily", "label_available_date",
})
OUTPUT_FILENAMES = (
    "v004c_board3_ablation_prediction_lock_v001.csv",
    "v004c_board3_ablation_population_v001.csv",
    "v004c_board3_ablation_daily_v001.csv",
    "v004c_board3_ablation_membership_v001.csv",
    "v004c_board3_ablation_attribution_v001.csv",
    "v004c_board3_ablation_practical_v001.csv",
    "v004c_board3_ablation_review_v001.md",
)


def assert_ablation_contract() -> None:
    if MODEL_ID != "V4A_ARCH_TRANSFER_V4C" or MODEL_FAMILY != "WEIGHTED_L2_LOGISTIC":
        raise RuntimeError("FATAL: frozen Stage1 identity changed")
    if L2 != 0.30 or POSITIVE_WEIGHT != 1.50 or INITIAL_TRAIN_DATES != 18:
        raise RuntimeError("FATAL: frozen Stage1 fit contract changed")
    if list(FROZEN_FEATURE_COLUMNS) != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError("FATAL: frozen 18-feature contract changed")
    if ABLATION_FEATURES != (
        "rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct", "days_since_d0_le1",
    ):
        raise RuntimeError("FATAL: ablation feature set changed")
    if len(ABLATION_FEATURES) != 3:
        raise RuntimeError("FATAL: ablation must contain exactly three features")
    if any((NEW_MODEL, NEW_FEATURE, NEW_POLICY, MODEL_REFIT_FOR_ABLATION,
            NEW_COEFFICIENTS, SUBSET_SEARCH, SCALE_SEARCH, SIGN_REVERSAL)):
        raise RuntimeError("FATAL: mechanism probe expanded beyond the one ablation")
    if JULY_RESULT_ROWS_ACCESSED != 0 or JULY_USED_FOR_ABLATION_DESIGN:
        raise RuntimeError("FATAL: July result data entered the ablation")
    if BOOTSTRAP_RESAMPLES != 20_000:
        raise RuntimeError("FATAL: bootstrap contract changed")


def _sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    result = np.empty_like(x)
    nonnegative = x >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    result[~nonnegative] = exp_x / (1.0 + exp_x)
    return result


def prediction_inputs(population: pd.DataFrame) -> pd.DataFrame:
    """Strip all outcomes before any counterfactual score or rank is built."""
    columns = [
        "event_id", "signal_date", "code", "board_group", "candidate_count",
        "stage1_score", "stage1_logit", "stage1_rank",
        *[contribution_column(feature) for feature in ABLATION_FEATURES],
    ]
    missing = sorted(set(columns).difference(population.columns))
    if missing:
        raise RuntimeError(f"FATAL: prediction inputs missing {missing}")
    result = population[columns].copy()
    if OUTCOME_COLUMNS.intersection(result.columns):
        raise RuntimeError("FATAL: outcome entered prediction inputs")
    return result.sort_values(["signal_date", "stage1_rank", "event_id"], kind="mergesort").reset_index(drop=True)


def build_prediction_lock(inputs: pd.DataFrame) -> pd.DataFrame:
    if OUTCOME_COLUMNS.intersection(inputs.columns):
        raise RuntimeError("FATAL: prediction phase received outcome columns")
    required = {
        "event_id", "signal_date", "code", "board_group", "candidate_count",
        "stage1_score", "stage1_logit", "stage1_rank",
        *[contribution_column(feature) for feature in ABLATION_FEATURES],
    }
    missing = sorted(required.difference(inputs.columns))
    if missing:
        raise RuntimeError(f"FATAL: prediction lock input missing {missing}")
    frame = inputs.copy()
    if len(frame) != 155 or frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: prediction lock must contain 155 unique events")
    if sorted(frame["signal_date"].astype(str).unique()) != EXPECTED_STRICT_DATES:
        raise RuntimeError("FATAL: strict date set changed")
    board3 = frame["board_group"].eq("BOARD3")
    if int((~board3).sum()) != 130 or int(board3.sum()) != 25:
        raise RuntimeError("FATAL: Board2/Board3 population changed")
    contrib_columns = [contribution_column(feature) for feature in ABLATION_FEATURES]
    frame = frame.rename(columns={
        "stage1_score": "original_score", "stage1_logit": "original_logit",
        "stage1_rank": "original_rank",
        **{column: f"contrib_{feature}" for column, feature in zip(contrib_columns, ABLATION_FEATURES)},
    })
    removed_columns = [f"contrib_{feature}" for feature in ABLATION_FEATURES]
    frame["ablation_removed_contribution"] = frame[removed_columns].sum(axis=1)
    frame["ablated_logit"] = frame["original_logit"]
    frame.loc[board3, "ablated_logit"] = (
        frame.loc[board3, "original_logit"]
        - frame.loc[board3, "ablation_removed_contribution"]
    )
    frame["ablated_score"] = _sigmoid(frame["ablated_logit"].to_numpy(float))
    frame = frame.sort_values(
        ["signal_date", "ablated_score", "event_id"],
        ascending=[True, False, True], kind="mergesort",
    )
    frame["ablated_rank"] = frame.groupby("signal_date", sort=False).cumcount() + 1
    for prefix, rank in (("original", "original_rank"), ("ablated", "ablated_rank")):
        for k in (10, 5, 3):
            frame[f"{prefix}_top{k}"] = frame[rank].le(np.minimum(frame["candidate_count"], k))
    board2 = frame["board_group"].eq("BOARD2")
    if not np.allclose(
        frame.loc[board2, "ablated_logit"], frame.loc[board2, "original_logit"],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("FATAL: Board2 logit changed")
    if not np.allclose(
        frame.loc[board2, "ablated_score"], frame.loc[board2, "original_score"],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("FATAL: Board2 score changed")
    board3_delta = frame.loc[~board2, "original_logit"] - frame.loc[~board2, "ablated_logit"]
    if not np.allclose(
        board3_delta, frame.loc[~board2, "ablation_removed_contribution"],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("FATAL: Board3 logit ablation is not exact")
    expected_score = _sigmoid(frame["ablated_logit"].to_numpy(float))
    if not np.allclose(expected_score, frame["ablated_score"], rtol=0.0, atol=1e-15):
        raise RuntimeError("FATAL: ablated score is not sigmoid(logit)")
    columns = [
        "event_id", "signal_date", "code", "board_group", "candidate_count",
        "original_score", "original_rank", "original_logit",
        *removed_columns, "ablation_removed_contribution", "ablated_logit",
        "ablated_score", "ablated_rank",
        "original_top10", "original_top5", "original_top3",
        "ablated_top10", "ablated_top5", "ablated_top3",
    ]
    return frame[columns].sort_values(
        ["signal_date", "ablated_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def prediction_lock_sha256(frame: pd.DataFrame) -> str:
    if OUTCOME_COLUMNS.intersection(frame.columns):
        raise RuntimeError("FATAL: prediction lock contains outcome columns")
    return hashlib.sha256(dataframe_csv_bytes(frame)).hexdigest()


def attach_outcomes(lock: pd.DataFrame, population: pd.DataFrame, bucket_map: Mapping[str, str]) -> pd.DataFrame:
    outcome = population[[
        "event_id", "target7", "loss", "raw_repair_return", "capped_return_7",
    ]].copy()
    if outcome["event_id"].duplicated().any() or lock["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate evaluation join key")
    result = lock.merge(outcome, on="event_id", how="left", validate="one_to_one")
    if len(result) != len(lock) or result[["target7", "loss", "raw_repair_return", "capped_return_7"]].isna().any().any():
        raise RuntimeError("FATAL: incomplete outcome join")
    result["target7"] = result["target7"].astype(int)
    result["loss"] = result["loss"].astype(int)
    result["capped_opportunity_return_7"] = result["capped_return_7"]
    result["opportunity_bucket"] = result["signal_date"].map(bucket_map)
    if result["opportunity_bucket"].isna().any():
        raise RuntimeError("FATAL: frozen bucket join failed")
    return result


def _raw_downside(frame: pd.DataFrame, rank_column: str) -> dict[str, float]:
    selected = frame[frame[rank_column].le(np.minimum(3, frame["candidate_count"]))]
    daily = selected.groupby("signal_date", sort=True)["raw_repair_return"].mean()
    return {
        "selected_stock_raw_mean": float(selected["raw_repair_return"].mean()),
        "selected_stock_raw_median": float(selected["raw_repair_return"].median()),
        "worst_selected_stock": float(selected["raw_repair_return"].min()),
        "worst_top3_raw_date": float(daily.min()),
    }


def _slot_quality(frame: pd.DataFrame, rank_column: str, board: str | None = None) -> dict[str, Any]:
    selected = frame[frame[rank_column].le(np.minimum(3, frame["candidate_count"]))]
    if board is not None:
        selected = selected[selected["board_group"].eq(board)]
    return {
        "slots": int(len(selected)),
        "target7_count": int(selected["target7"].sum()),
        "target7_rate": float(selected["target7"].mean()) if len(selected) else np.nan,
        "loss_count": int(selected["loss"].sum()),
        "loss_rate": float(selected["loss"].mean()) if len(selected) else np.nan,
        "severe_loss_count": int(selected["raw_repair_return"].le(-0.05).sum()),
        "mean_capped": float(selected["capped_return_7"].mean()) if len(selected) else np.nan,
        "median_capped": float(selected["capped_return_7"].median()) if len(selected) else np.nan,
        "worst_capped": float(selected["capped_return_7"].min()) if len(selected) else np.nan,
    }


def _practical(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rank_columns = {"STRICT_STAGE1": "original_rank", "BOARD3_CONFIRMED_HARMFUL_ABLATION": "ablated_rank"}
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    scopes = {"COMBINED": sorted(frame["signal_date"].unique())}
    scopes.update({bucket: sorted(frame.loc[frame["opportunity_bucket"].eq(bucket), "signal_date"].unique()) for bucket in ("LOW", "MID", "HIGH")})
    for scope, dates in scopes.items():
        subset = frame[frame["signal_date"].isin(dates)]
        for policy, rank_column in rank_columns.items():
            metrics = strategy_summary(subset, rank_column)
            if scope == "COMBINED":
                metrics.update(_raw_downside(subset, rank_column))
                summaries[policy] = metrics
            rows.append({"scope": scope, "record_type": "POLICY", "name": policy, **metrics})
        baseline = strategy_summary(subset, "original_rank")
        rows.append({
            "scope": scope, "record_type": "POLICY", "name": "UNIVERSE",
            "date_count": len(dates), "Top3_mean": baseline["Top3_baseline"],
            "universe_target7_rate": baseline["universe_target7_rate"],
        })
        for board in ("BOARD2", "BOARD3"):
            for label, rank_column in (("ORIGINAL", "original_rank"), ("ABLATED", "ablated_rank")):
                quality = _slot_quality(subset, rank_column, board)
                rows.append({
                    "scope": scope, "record_type": "BOARD_TOP3_SLOTS",
                    "name": f"{board}_{label}_TOP3_SLOTS", **quality,
                })
                if scope == "COMBINED":
                    summaries[f"{board}_{label}"] = quality
    return pd.DataFrame(rows), summaries


def _funnel_summary(frame: pd.DataFrame) -> dict[str, Any]:
    candidate_board3_share = float(frame["board_group"].eq("BOARD3").mean())
    result: dict[str, Any] = {"candidate_board3_share": candidate_board3_share}
    for prefix in ("original", "ablated"):
        result[prefix] = {}
        for k in (10, 5, 3):
            selected = frame[frame[f"{prefix}_top{k}"]]
            b3 = int(selected["board_group"].eq("BOARD3").sum())
            b2 = len(selected) - b3
            share = float(b3 / len(selected))
            result[prefix][f"top{k}"] = {
                "total": int(len(selected)), "board2": int(b2), "board3": b3,
                "board3_share": share,
                "board3_composition_lift": share / candidate_board3_share,
                "board3_selection_rate": b3 / 25.0,
                "board2_selection_rate": b2 / 130.0,
            }
    return result


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, day in frame.groupby("signal_date", sort=True):
        original = day[day["original_top3"]]
        ablated = day[day["ablated_top3"]]
        row: dict[str, Any] = {
            "signal_date": date, "candidate_count": len(day),
            "board2_candidates": int(day["board_group"].eq("BOARD2").sum()),
            "board3_candidates": int(day["board_group"].eq("BOARD3").sum()),
            "opportunity_bucket": str(day["opportunity_bucket"].iloc[0]),
        }
        for k in (10, 5, 3):
            row[f"original_board3_top{k}"] = int((day[f"original_top{k}"] & day["board_group"].eq("BOARD3")).sum())
            row[f"ablated_board3_top{k}"] = int((day[f"ablated_top{k}"] & day["board_group"].eq("BOARD3")).sum())
        row.update({
            "original_top3_return": float(original["capped_return_7"].mean()),
            "ablated_top3_return": float(ablated["capped_return_7"].mean()),
            "delta": float(ablated["capped_return_7"].mean() - original["capped_return_7"].mean()),
            "original_top3_target7": int(original["target7"].sum()),
            "ablated_top3_target7": int(ablated["target7"].sum()),
            "original_top3_loss": int(original["loss"].sum()),
            "ablated_top3_loss": int(ablated["loss"].sum()),
        })
        for prefix, selected in (("original", original), ("ablated", ablated)):
            b3 = selected[selected["board_group"].eq("BOARD3")]
            row[f"{prefix}_board3_top3_target7"] = int(b3["target7"].sum())
            row[f"{prefix}_board3_top3_loss"] = int(b3["loss"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _membership_and_attribution(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    membership_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    date_count = int(frame["signal_date"].nunique())
    for date, day in frame.groupby("signal_date", sort=True):
        original_ids = set(day.loc[day["original_top3"], "event_id"].astype(str))
        ablated_ids = set(day.loc[day["ablated_top3"], "event_id"].astype(str))
        demoted = day[day["event_id"].astype(str).isin(original_ids - ablated_ids)].sort_values(
            ["original_rank", "event_id"], kind="mergesort"
        )
        promoted = day[day["event_id"].astype(str).isin(ablated_ids - original_ids)].sort_values(
            ["ablated_rank", "event_id"], kind="mergesort"
        )
        if len(demoted) != len(promoted):
            raise RuntimeError("FATAL: changed membership cardinality mismatch")
        for change_type, changed in (("DEMOTED", demoted), ("PROMOTED", promoted)):
            for row in changed.itertuples(index=False):
                membership_rows.append({
                    "signal_date": date, "change_type": change_type,
                    "event_id": row.event_id, "code": row.code,
                    "board_group": row.board_group, "original_rank": int(row.original_rank),
                    "ablated_rank": int(row.ablated_rank), "original_score": float(row.original_score),
                    "ablated_score": float(row.ablated_score),
                    "ablation_removed_contribution": float(row.ablation_removed_contribution),
                    "target7": int(row.target7), "loss": int(row.loss),
                    "raw_repair_return": float(row.raw_repair_return),
                    "capped_return_7": float(row.capped_return_7),
                })
        k = min(3, len(day))
        for left, right in zip(demoted.itertuples(index=False), promoted.itertuples(index=False)):
            pair_delta = float(right.capped_return_7 - left.capped_return_7)
            state = outcome_state(float(left.raw_repair_return))
            attribution_rows.append({
                "signal_date": date,
                "demoted_code": left.code, "demoted_board": left.board_group,
                "demoted_original_rank": int(left.original_rank), "demoted_ablated_rank": int(left.ablated_rank),
                "demoted_target7": int(left.target7), "demoted_loss": int(left.loss),
                "demoted_raw": float(left.raw_repair_return), "demoted_capped": float(left.capped_return_7),
                "promoted_code": right.code, "promoted_board": right.board_group,
                "promoted_original_rank": int(right.original_rank), "promoted_ablated_rank": int(right.ablated_rank),
                "promoted_target7": int(right.target7), "promoted_loss": int(right.loss),
                "promoted_raw": float(right.raw_repair_return), "promoted_capped": float(right.capped_return_7),
                "pair_delta": pair_delta, "slot_contribution": pair_delta / k,
                "overall_mean_contribution": pair_delta / k / date_count,
                "attribution_category": f"DEMOTED_{left.board_group}_{state}",
            })
    membership_columns = [
        "signal_date", "change_type", "event_id", "code", "board_group",
        "original_rank", "ablated_rank", "original_score", "ablated_score",
        "ablation_removed_contribution", "target7", "loss", "raw_repair_return",
        "capped_return_7",
    ]
    attribution_columns = [
        "signal_date", "demoted_code", "demoted_board", "demoted_original_rank",
        "demoted_ablated_rank", "demoted_target7", "demoted_loss", "demoted_raw",
        "demoted_capped", "promoted_code", "promoted_board", "promoted_original_rank",
        "promoted_ablated_rank", "promoted_target7", "promoted_loss", "promoted_raw",
        "promoted_capped", "pair_delta", "slot_contribution", "overall_mean_contribution",
        "attribution_category",
    ]
    membership = pd.DataFrame(membership_rows, columns=membership_columns)
    attribution = pd.DataFrame(attribution_rows, columns=attribution_columns)
    original_daily = frame[frame["original_top3"]].groupby("signal_date")["capped_return_7"].mean()
    ablated_daily = frame[frame["ablated_top3"]].groupby("signal_date")["capped_return_7"].mean()
    observed = float((ablated_daily - original_daily).mean())
    attributed = float(attribution["overall_mean_contribution"].sum()) if len(attribution) else 0.0
    if not np.isclose(observed, attributed, rtol=0.0, atol=1e-10):
        raise RuntimeError("FATAL: ablation attribution does not close")
    summary = {
        "changed_dates": int(membership["signal_date"].nunique()) if len(membership) else 0,
        "changed_slots": int(membership["change_type"].eq("DEMOTED").sum()) if len(membership) else 0,
        "observed": observed, "attributed": attributed,
    }
    return membership, attribution, summary


def _rank_movement(frame: pd.DataFrame) -> dict[str, Any]:
    b3 = frame[frame["board_group"].eq("BOARD3")].copy()
    b3["rank_change"] = b3["ablated_rank"] - b3["original_rank"]
    states = {
        "LOSS": b3["loss"].eq(1),
        "TARGET7": b3["target7"].eq(1),
        "POSITIVE_NON_TARGET": b3["loss"].eq(0) & b3["target7"].eq(0),
    }
    result: dict[str, Any] = {}
    for state, mask in states.items():
        values = b3.loc[mask, "rank_change"].to_numpy(float)
        result[state] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)), "p75": float(np.quantile(values, 0.75)),
        }
    result["median_loss_minus_target7"] = result["LOSS"]["median"] - result["TARGET7"]["median"]
    improved = worsened = ties = informative_dates = 0
    for _, day in b3.groupby("signal_date", sort=True):
        losses = day[day["loss"].eq(1)]
        winners = day[day["target7"].eq(1)]
        if not len(losses) or not len(winners):
            continue
        informative_dates += 1
        for winner in winners.itertuples(index=False):
            for loss in losses.itertuples(index=False):
                original = winner.original_rank - loss.original_rank
                ablated = winner.ablated_rank - loss.ablated_rank
                if ablated < original:
                    improved += 1
                elif ablated > original:
                    worsened += 1
                else:
                    ties += 1
    pairs = improved + worsened + ties
    result["pair"] = {
        "informative_dates": informative_dates, "pairs": pairs, "improved": improved,
        "worsened": worsened, "ties": ties,
        "improvement_rate": improved / pairs if pairs else np.nan,
    }
    return result


def _bootstrap_and_lodo(frame: pd.DataFrame, daily: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    dates = sorted(frame["signal_date"].unique())
    date_index = {date: index for index, date in enumerate(dates)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(dates), size=(BOOTSTRAP_RESAMPLES, len(dates)))
    counts = np.zeros((BOOTSTRAP_RESAMPLES, len(dates)), dtype=np.int16)
    np.add.at(counts, (np.repeat(np.arange(BOOTSTRAP_RESAMPLES), len(dates)), draws.ravel()), 1)
    deltas = daily.set_index("signal_date").loc[dates, "delta"].to_numpy(float)
    overall = (counts @ deltas) / counts.sum(axis=1)

    def board_arrays(prefix: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        selected = frame[frame[f"{prefix}_top3"] & frame["board_group"].eq("BOARD3")]
        grouped = selected.groupby("signal_date", sort=True)
        numerator = np.zeros(len(dates), dtype=float)
        denominator = np.zeros(len(dates), dtype=float)
        for date, part in grouped:
            idx = date_index[str(date)]
            numerator[idx] = float(part[metric].sum())
            denominator[idx] = float(len(part))
        return numerator, denominator

    boot_board: dict[str, np.ndarray] = {}
    for metric in ("loss", "capped_return_7"):
        values = {}
        for prefix in ("original", "ablated"):
            numerator, denominator = board_arrays(prefix, metric)
            num = counts @ numerator
            den = counts @ denominator
            result = np.full(BOOTSTRAP_RESAMPLES, np.nan)
            valid = den > 0
            result[valid] = num[valid] / den[valid]
            values[prefix] = result
        boot_board[metric] = values["ablated"] - values["original"]

    def quantiles(values: np.ndarray, direction: str) -> dict[str, Any]:
        valid = values[np.isfinite(values)]
        return {
            "valid": int(len(valid)), "p2.5": float(np.quantile(valid, .025)),
            "p50": float(np.quantile(valid, .50)), "p97.5": float(np.quantile(valid, .975)),
            "direction_probability": float(np.mean(valid > 0)) if direction == "positive" else float(np.mean(valid < 0)),
        }

    bootstrap = {
        "overall": quantiles(overall, "positive"),
        "board3_loss": quantiles(boot_board["loss"], "negative"),
        "board3_capped": quantiles(boot_board["capped_return_7"], "positive"),
    }
    lodo_values = {"overall": [], "board3_loss": [], "board3_capped": []}
    for omitted in dates:
        subset = frame[~frame["signal_date"].eq(omitted)]
        subset_daily = daily[~daily["signal_date"].eq(omitted)]
        lodo_values["overall"].append(float(subset_daily["delta"].mean()))
        for metric, key in (("loss", "board3_loss"), ("capped_return_7", "board3_capped")):
            original = subset[subset["original_top3"] & subset["board_group"].eq("BOARD3")][metric].mean()
            ablated = subset[subset["ablated_top3"] & subset["board_group"].eq("BOARD3")][metric].mean()
            lodo_values[key].append(float(ablated - original))

    def lodo_summary(values: list[float], direction: str) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        valid = array[np.isfinite(array)]
        favorable = valid > 0 if direction == "positive" else valid < 0
        return {
            "valid": int(len(valid)),
            "favorable_pct": float(np.mean(favorable)) if len(valid) else np.nan,
            "min": float(np.min(valid)) if len(valid) else np.nan,
            "median": float(np.median(valid)) if len(valid) else np.nan,
            "max": float(np.max(valid)) if len(valid) else np.nan,
        }
    lodo = {
        "overall": lodo_summary(lodo_values["overall"], "positive"),
        "board3_loss": lodo_summary(lodo_values["board3_loss"], "negative"),
        "board3_capped": lodo_summary(lodo_values["board3_capped"], "positive"),
    }
    return bootstrap, lodo


def _transition_summary(frame: pd.DataFrame, membership: pd.DataFrame) -> dict[str, Any]:
    demoted = membership[membership["change_type"].eq("DEMOTED")]
    promoted = membership[membership["change_type"].eq("PROMOTED")]
    original_b3 = frame[frame["original_top3"] & frame["board_group"].eq("BOARD3")]
    retained_b3 = frame[frame["original_top3"] & frame["ablated_top3"] & frame["board_group"].eq("BOARD3")]
    demoted_b3 = demoted[demoted["board_group"].eq("BOARD3")]
    winner_total = int(original_b3["target7"].sum())
    loss_total = int(original_b3["loss"].sum())
    winner_demoted = int(demoted_b3["target7"].sum())
    loss_demoted = int(demoted_b3["loss"].sum())
    winner_removal = winner_demoted / winner_total if winner_total else np.nan
    loss_removal = loss_demoted / loss_total if loss_total else np.nan
    return {
        "demoted_board3": int(demoted["board_group"].eq("BOARD3").sum()),
        "demoted_board2": int(demoted["board_group"].eq("BOARD2").sum()),
        "promoted_board3": int(promoted["board_group"].eq("BOARD3").sum()),
        "promoted_board2": int(promoted["board_group"].eq("BOARD2").sum()),
        "demoted_target7": int(demoted["target7"].sum()), "demoted_loss": int(demoted["loss"].sum()),
        "promoted_target7": int(promoted["target7"].sum()), "promoted_loss": int(promoted["loss"].sum()),
        "net_target7_slots": int(promoted["target7"].sum() - demoted["target7"].sum()),
        "net_loss_slots_removed": int(demoted["loss"].sum() - promoted["loss"].sum()),
        "board3_winner_total": winner_total, "board3_winner_retained": int(retained_b3["target7"].sum()),
        "board3_winner_demoted": winner_demoted, "board3_winner_removal_rate": winner_removal,
        "board3_loss_total": loss_total, "board3_loss_retained": int(retained_b3["loss"].sum()),
        "board3_loss_demoted": loss_demoted, "board3_loss_removal_rate": loss_removal,
        "selectivity_gap": loss_removal - winner_removal,
    }


def _decision(context: Mapping[str, Any]) -> dict[str, Any]:
    practical = context["practical_summary"]
    original = practical["STRICT_STAGE1"]
    ablated = practical["BOARD3_CONFIRMED_HARMFUL_ABLATION"]
    b3_original = practical["BOARD3_ORIGINAL"]
    b3_ablated = practical["BOARD3_ABLATED"]
    transition = context["transition_summary"]
    daily = context["daily"]
    bootstrap = context["bootstrap"]
    funnel = context["funnel"]
    gates = {
        "A": b3_ablated["loss_rate"] <= b3_original["loss_rate"] - .15,
        "B": transition["board3_loss_demoted"] >= 3,
        "C": transition["board3_winner_demoted"] <= 1,
        "D": transition["selectivity_gap"] >= .25,
        "E": b3_ablated["mean_capped"] >= b3_original["mean_capped"] + .0075,
        "F": (b3_ablated["target7_rate"] >= b3_original["target7_rate"]
              or b3_ablated["target7_count"] >= b3_original["target7_count"] - 1),
        "G": ablated["Top3_mean"] >= original["Top3_mean"] + .005,
        "H": (ablated["Top3_target7_precision"] >= original["Top3_target7_precision"]
              and ablated["winner_capture"] >= original["winner_capture"] - .05),
        "I": (ablated["Top3_negative_date_rate"] <= original["Top3_negative_date_rate"]
              and ablated["Top3_worst"] >= original["Top3_worst"]),
        "J": int(daily["delta"].gt(0).sum()) > int(daily["delta"].lt(0).sum()),
        "K": (bootstrap["overall"]["direction_probability"] >= .65
              and bootstrap["board3_loss"]["direction_probability"] >= .80),
        "L": b3_ablated["slots"] >= 5,
    }
    if all(gates.values()):
        signal = "STRONG"
        direction = "MINIMAL_BOARD_INTERACTION_EXPERIMENT"
    else:
        meaningful = bool(
            (b3_ablated["loss_rate"] < b3_original["loss_rate"] and b3_ablated["mean_capped"] > b3_original["mean_capped"])
            or ablated["Top3_mean"] > original["Top3_mean"]
        )
        absent = bool(
            b3_ablated["loss_rate"] >= b3_original["loss_rate"]
            or b3_ablated["mean_capped"] <= b3_original["mean_capped"]
            or ablated["Top3_mean"] <= original["Top3_mean"]
            or transition["selectivity_gap"] <= 0
        )
        signal = "ABSENT" if absent and not meaningful else "PARTIAL"
        direction = "BROADER_BOARD_SEPARATION" if signal == "ABSENT" else "NO_BOARD_MODEL_YET"
    slot_reduction = b3_original["slots"] - b3_ablated["slots"]
    return {
        "gates": gates, "signal": signal, "direction": direction,
        "board3_slot_reduction": slot_reduction,
        "board3_slot_reduction_pct": slot_reduction / b3_original["slots"],
        "board3_suppression_high": bool(slot_reduction / b3_original["slots"] >= .50),
        "board3_effectively_excluded": bool(b3_ablated["slots"] == 0),
    }


def _pct(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{100 * float(value):.{digits}f}%"


def _num(value: Any, digits: int = 6) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    practical = context["practical_summary"]
    original = practical["STRICT_STAGE1"]
    ablated = practical["BOARD3_CONFIRMED_HARMFUL_ABLATION"]
    b3o, b3a = practical["BOARD3_ORIGINAL"], practical["BOARD3_ABLATED"]
    decision = context["decision"]
    transition = context["transition_summary"]
    movement = context["rank_movement"]
    score = context["score_movement"]
    funnel = context["funnel"]
    membership = context["membership_summary"]
    bootstrap = context["bootstrap"]
    lodo = context["lodo"]
    attribution = context["attribution"]
    lines = [
        "# v004c Board3 Confirmed-Harmful Contribution Ablation v001",
        "## 1. Experimental Contract",
        "- Post-hoc June mechanism probe; frozen V4C_STAGE1 only; exactly one predeclared three-feature contribution ablation; no model, feature, subset, scale, policy, or July-result access.",
        "## 2. Strict Stage1 Parity",
        f"- Population: **155 rows / 17 dates; Board2 130, Board3 25**. Stage1 Top3 **{_pct(original['Top3_mean'])}**, precision **{_pct(original['Top3_target7_precision'])}**.",
        "## 3. Predeclared Three-Feature Ablation",
        "- Board3 only: `rank_d1_close_ma10_pct`, `rank_d1_low_ma10_pct`, `days_since_d0_le1`; each frozen contribution is set exactly to zero, including negative contributions. Board2 is unchanged.",
        "## 4. Prediction Lock / Leakage Audit",
        f"- SHA256: `{context['prediction_lock_sha256']}`; outcome perturbation PASS; self/current/July rows **0 / 0 / 0**.",
        "## 5. Board3 Score Movement",
        f"- Mean original/ablated logit **{_num(score['original_logit_mean'])} / {_num(score['ablated_logit_mean'])}**.",
        f"- Removed contribution mean/median/p25/p75/min/max: **{_num(score['reduction_mean'])} / {_num(score['reduction_median'])} / {_num(score['reduction_p25'])} / {_num(score['reduction_p75'])} / {_num(score['reduction_min'])} / {_num(score['reduction_max'])}**.",
        "| Feature | Mean removed | Median | Positive rows | Negative rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature in ABLATION_FEATURES:
        stats = score["features"][feature]
        lines.append(f"| {feature} | {_num(stats['mean'])} | {_num(stats['median'])} | {_pct(stats['positive_row_pct'])} | {_pct(stats['negative_row_pct'])} |")
    lines += [
        "## 6. Funnel Composition Change",
        f"- Candidate Board3 share: **{_pct(funnel['candidate_board3_share'])}**.",
        "| Ranking | Level | B2 | B3 | B3 share | B3 composition lift | B3 selection rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for prefix in ("original", "ablated"):
        for k in (10, 5, 3):
            row = funnel[prefix][f"top{k}"]
            lines.append(f"| {prefix.upper()} | Top{k} | {row['board2']} | {row['board3']} | {_pct(row['board3_share'])} | {_num(row['board3_composition_lift'])} | {_pct(row['board3_selection_rate'])} |")
    lines += [
        f"- Board3 Top3 slot reduction: **{decision['board3_slot_reduction']} / {_pct(decision['board3_slot_reduction_pct'])}**; suppression high **{'YES' if decision['board3_suppression_high'] else 'NO'}**; effectively excluded **{'YES' if decision['board3_effectively_excluded'] else 'NO'}**.",
        "## 7. Board3 Rank Movement by Outcome",
        "| State | Mean rank change | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|",
    ]
    for state in ("LOSS", "TARGET7", "POSITIVE_NON_TARGET"):
        row = movement[state]
        lines.append(f"| {state} | {_num(row['mean'])} | {_num(row['median'])} | {_num(row['p25'])} | {_num(row['p75'])} |")
    pair = movement["pair"]
    lines += [
        f"- Median LOSS-minus-Target7 rank-change gap: **{_num(movement['median_loss_minus_target7'])}**.",
        f"- Same-date Board3 LOSS-vs-Target7: dates **{pair['informative_dates']}**, pairs **{pair['pairs']}**, improved/worsened/ties **{pair['improved']}/{pair['worsened']}/{pair['ties']}**, improvement rate **{_pct(pair['improvement_rate'])}**.",
        "## 8. Top3 Membership Transitions",
        f"- Changed dates/slots **{membership['changed_dates']} / {membership['changed_slots']}**.",
        f"- Demoted Board3/Board2 **{transition['demoted_board3']}/{transition['demoted_board2']}**; promoted Board3/Board2 **{transition['promoted_board3']}/{transition['promoted_board2']}**.",
        f"- Demoted Target7/LOSS **{transition['demoted_target7']}/{transition['demoted_loss']}**; promoted Target7/LOSS **{transition['promoted_target7']}/{transition['promoted_loss']}**; net Target7 **{transition['net_target7_slots']}**, net LOSS removed **{transition['net_loss_slots_removed']}**.",
        "## 9. Board3 Loss Removal vs Winner Removal",
        f"- Board3 loss removal **{transition['board3_loss_demoted']}/{transition['board3_loss_total']} ({_pct(transition['board3_loss_removal_rate'])})**; winner removal **{transition['board3_winner_demoted']}/{transition['board3_winner_total']} ({_pct(transition['board3_winner_removal_rate'])})**; selectivity gap **{_pct(transition['selectivity_gap'])}**.",
        "## 10. Overall Practical Performance",
        "| Policy | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Precision | Capture | Negative dates | Worst | <=-3 | <=-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| STRICT_STAGE1 | {_pct(original['Rank1_mean'])} | {_pct(original['Rank2_mean'])} | {_pct(original['Rank3_mean'])} | {_pct(original['Top2_mean'])} | {_pct(original['Top3_mean'])} | {_pct(original['Top3_target7_precision'])} | {_pct(original['winner_capture'])} | {_pct(original['Top3_negative_date_rate'])} | {_pct(original['Top3_worst'])} | {original['Top3_days_le_minus_3']} | {original['Top3_days_le_minus_5']} |",
        f"| BOARD3_ABLATION | {_pct(ablated['Rank1_mean'])} | {_pct(ablated['Rank2_mean'])} | {_pct(ablated['Rank3_mean'])} | {_pct(ablated['Top2_mean'])} | {_pct(ablated['Top3_mean'])} | {_pct(ablated['Top3_target7_precision'])} | {_pct(ablated['winner_capture'])} | {_pct(ablated['Top3_negative_date_rate'])} | {_pct(ablated['Top3_worst'])} | {ablated['Top3_days_le_minus_3']} | {ablated['Top3_days_le_minus_5']} |",
        f"- Stage1/Ablation Top3 **{_pct(original['Top3_mean'])} / {_pct(ablated['Top3_mean'])}**; delta **{_pct(ablated['Top3_mean'] - original['Top3_mean'])}**; precision **{_pct(original['Top3_target7_precision'])} / {_pct(ablated['Top3_target7_precision'])}**.",
        f"- Board3 Top3 loss **{_pct(b3o['loss_rate'])} → {_pct(b3a['loss_rate'])}**; mean capped **{_pct(b3o['mean_capped'])} → {_pct(b3a['mean_capped'])}**; Target7 count **{b3o['target7_count']} → {b3a['target7_count']}**.",
        f"- Board2 Top3 slots/Target7/LOSS/mean capped: **{practical['BOARD2_ORIGINAL']['slots']}/{practical['BOARD2_ORIGINAL']['target7_count']}/{practical['BOARD2_ORIGINAL']['loss_count']}/{_pct(practical['BOARD2_ORIGINAL']['mean_capped'])} → {practical['BOARD2_ABLATED']['slots']}/{practical['BOARD2_ABLATED']['target7_count']}/{practical['BOARD2_ABLATED']['loss_count']}/{_pct(practical['BOARD2_ABLATED']['mean_capped'])}**.",
        f"- Raw downside Stage1/Ablation selected-stock mean **{_pct(original['selected_stock_raw_mean'])}/{_pct(ablated['selected_stock_raw_mean'])}**, median **{_pct(original['selected_stock_raw_median'])}/{_pct(ablated['selected_stock_raw_median'])}**, worst stock **{_pct(original['worst_selected_stock'])}/{_pct(ablated['worst_selected_stock'])}**, worst Top3 date **{_pct(original['worst_top3_raw_date'])}/{_pct(ablated['worst_top3_raw_date'])}**.",
        "## 11. LOW / MID / HIGH",
    ]
    for bucket in ("LOW", "MID", "HIGH"):
        rows = context["practical"]
        o = rows[(rows.scope.eq(bucket)) & rows.name.eq("STRICT_STAGE1")].iloc[0]
        a = rows[(rows.scope.eq(bucket)) & rows.name.eq("BOARD3_CONFIRMED_HARMFUL_ABLATION")].iloc[0]
        b3o_row = rows[(rows.scope.eq(bucket)) & rows.name.eq("BOARD3_ORIGINAL_TOP3_SLOTS")].iloc[0]
        b3a_row = rows[(rows.scope.eq(bucket)) & rows.name.eq("BOARD3_ABLATED_TOP3_SLOTS")].iloc[0]
        lines.append(f"- {bucket}: Stage1/Ablation Top3 **{_pct(o.Top3_mean)} / {_pct(a.Top3_mean)}** (delta **{_pct(a.Top3_mean-o.Top3_mean)}**); Board3 slots/Target7/LOSS/mean capped **{int(b3o_row.slots)}/{int(b3o_row.target7_count)}/{int(b3o_row.loss_count)}/{_pct(b3o_row.mean_capped)} → {int(b3a_row.slots)}/{int(b3a_row.target7_count)}/{int(b3a_row.loss_count)}/{_pct(b3a_row.mean_capped)}**; {'SMALL_SAMPLE' if int(b3a_row.slots) < 5 else ''}.")
    lines += [
        "## 12. Daily Robustness / Bootstrap / LODO",
        f"- Daily delta mean/median **{_pct(context['daily'].delta.mean())}/{_pct(context['daily'].delta.median())}**; positive/negative/zero **{int(context['daily'].delta.gt(0).sum())}/{int(context['daily'].delta.lt(0).sum())}/{int(context['daily'].delta.eq(0).sum())}**.",
        f"- Bootstrap overall delta p2.5/p50/p97.5/P>0 **{_pct(bootstrap['overall']['p2.5'])}/{_pct(bootstrap['overall']['p50'])}/{_pct(bootstrap['overall']['p97.5'])}/{_pct(bootstrap['overall']['direction_probability'])}**.",
        f"- Bootstrap Board3 LOSS delta valid/p2.5/p50/p97.5/P<0 **{bootstrap['board3_loss']['valid']}/{_pct(bootstrap['board3_loss']['p2.5'])}/{_pct(bootstrap['board3_loss']['p50'])}/{_pct(bootstrap['board3_loss']['p97.5'])}/{_pct(bootstrap['board3_loss']['direction_probability'])}**.",
        f"- Bootstrap Board3 capped delta valid/p2.5/p50/p97.5/P>0 **{bootstrap['board3_capped']['valid']}/{_pct(bootstrap['board3_capped']['p2.5'])}/{_pct(bootstrap['board3_capped']['p50'])}/{_pct(bootstrap['board3_capped']['p97.5'])}/{_pct(bootstrap['board3_capped']['direction_probability'])}**.",
        f"- LODO overall valid/favorable/min/median/max **{lodo['overall']['valid']}/{_pct(lodo['overall']['favorable_pct'])}/{_pct(lodo['overall']['min'])}/{_pct(lodo['overall']['median'])}/{_pct(lodo['overall']['max'])}**.",
        f"- LODO Board3 LOSS valid/favorable/min/median/max **{lodo['board3_loss']['valid']}/{_pct(lodo['board3_loss']['favorable_pct'])}/{_pct(lodo['board3_loss']['min'])}/{_pct(lodo['board3_loss']['median'])}/{_pct(lodo['board3_loss']['max'])}**; capped **{lodo['board3_capped']['valid']}/{_pct(lodo['board3_capped']['favorable_pct'])}/{_pct(lodo['board3_capped']['min'])}/{_pct(lodo['board3_capped']['median'])}/{_pct(lodo['board3_capped']['max'])}**.",
        "## 13. Return Attribution",
        f"- Attributed/observed Top3 delta **{_pct(membership['attributed'])} / {_pct(membership['observed'])}**; closure PASS.",
        "| Demoted category | Count | Contribution |",
        "|---|---:|---:|",
    ]
    for category in (
        "DEMOTED_BOARD3_LOSS", "DEMOTED_BOARD3_TARGET7", "DEMOTED_BOARD3_POSITIVE_NON_TARGET",
        "DEMOTED_BOARD2_LOSS", "DEMOTED_BOARD2_TARGET7", "DEMOTED_BOARD2_POSITIVE_NON_TARGET",
    ):
        subset = attribution[attribution["attribution_category"].eq(category)]
        lines.append(f"| {category} | {len(subset)} | {_pct(subset['overall_mean_contribution'].sum() if len(subset) else 0.0)} |")
    lines += [
        "## 14. Mechanism Decision",
        "| Gate | Pass |",
        "|---|---|",
    ]
    for gate, passed in decision["gates"].items():
        lines.append(f"| {gate} | {'YES' if passed else 'NO'} |")
    lines += [
        f"- `ABLATION_MECHANISM_SIGNAL = {decision['signal']}`",
        f"- `NEXT_BOARD_ARCHITECTURE_DIRECTION = {decision['direction']}`",
        f"- The ablation {'selectively reduced' if transition['selectivity_gap'] > 0 else 'did not selectively reduce'} Board3 loss slots; it removed **{transition['board3_loss_demoted']}/8 losses and {transition['board3_winner_demoted']}/4 winners**, leaving one Board3 Top3 slot, which was a loss. Board3 slot suppression was **{_pct(decision['board3_slot_reduction_pct'])}**.",
        f"- Overall Top3 fell **{_pct(ablated['Top3_mean'] - original['Top3_mean'])}**. The three contributions explain broad Board3 promotion pressure, but not a safe loss-selective mechanism; their removal suppresses winners at least as aggressively as losses.",
        "- Recommended next technical action: preserve this negative probe and, in a separate task only, consider broader Board2/Board3 formulation separation. Do not test another ablation, subset, scale, or minimal interaction now.",
        "- This is same-June mechanism plausibility only, not validation or forward evidence. No second ablation or board-aware model is authorized here.",
    ]
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_ablation_contract()
    population, _, fold_audit, reconstruction = reconstruct_frozen_stage1(root)
    inputs = prediction_inputs(population)
    lock = build_prediction_lock(inputs)
    lock_hash = prediction_lock_sha256(lock)
    perturbed = population.copy()
    rng = np.random.default_rng(20260816)
    for column in ("raw_repair_return", "capped_return_7", "target7", "loss", "nonloss"):
        perturbed[column] = rng.normal(1_000.0, 100.0, len(perturbed))
    perturb_lock = build_prediction_lock(prediction_inputs(perturbed))
    if dataframe_csv_bytes(lock) != dataframe_csv_bytes(perturb_lock):
        raise RuntimeError("FATAL: outcome perturbation changed ablation prediction")
    bucket_map = load_frozen_june_buckets(root, EXPECTED_STRICT_DATES)
    evaluated = attach_outcomes(lock, population, bucket_map)
    parity = strategy_summary(evaluated, "original_rank")
    for metric, expected in EXPECTED_STAGE1_PARITY.items():
        if not np.isclose(parity[metric], expected, rtol=0.0, atol=1e-11):
            raise RuntimeError(f"FATAL: strict Stage1 parity failed for {metric}")
    b3 = evaluated[evaluated["board_group"].eq("BOARD3")]
    b3_original = b3[b3["original_top3"]]
    if (len(b3_original), int(b3_original.target7.sum()), int(b3_original.loss.sum())) != (15, 4, 8):
        raise RuntimeError("FATAL: original Board3 Top3 parity failed")
    if not np.isclose(b3_original.capped_return_7.mean(), .014147, atol=5e-7):
        raise RuntimeError("FATAL: original Board3 Top3 return parity failed")
    if not np.isclose(b3_original.capped_return_7.median(), -.007092, atol=5e-7):
        raise RuntimeError("FATAL: original Board3 Top3 median parity failed")
    if not np.isclose(b3_original.capped_return_7.min(), -.044457, atol=5e-7):
        raise RuntimeError("FATAL: original Board3 Top3 worst parity failed")
    practical, practical_summary = _practical(evaluated)
    funnel = _funnel_summary(evaluated)
    if (
        funnel["original"]["top10"]["total"] != 132
        or funnel["original"]["top10"]["board3"] != 24
        or funnel["original"]["top5"]["total"] != 81
        or funnel["original"]["top5"]["board3"] != 19
        or funnel["original"]["top3"]["total"] != 51
        or funnel["original"]["top3"]["board3"] != 15
    ):
        raise RuntimeError("FATAL: original funnel composition parity failed")
    daily = _daily(evaluated)
    membership, attribution, membership_summary = _membership_and_attribution(evaluated)
    transition_summary = _transition_summary(evaluated, membership)
    rank_movement = _rank_movement(evaluated)
    bootstrap, lodo = _bootstrap_and_lodo(evaluated, daily)
    board3_lock = lock[lock["board_group"].eq("BOARD3")]
    score_movement: dict[str, Any] = {
        "original_logit_mean": float(board3_lock.original_logit.mean()),
        "ablated_logit_mean": float(board3_lock.ablated_logit.mean()),
        "reduction_mean": float(board3_lock.ablation_removed_contribution.mean()),
        "reduction_median": float(board3_lock.ablation_removed_contribution.median()),
        "reduction_p25": float(board3_lock.ablation_removed_contribution.quantile(.25)),
        "reduction_p75": float(board3_lock.ablation_removed_contribution.quantile(.75)),
        "reduction_min": float(board3_lock.ablation_removed_contribution.min()),
        "reduction_max": float(board3_lock.ablation_removed_contribution.max()),
        "features": {},
    }
    for feature in ABLATION_FEATURES:
        values = board3_lock[f"contrib_{feature}"]
        score_movement["features"][feature] = {
            "mean": float(values.mean()), "median": float(values.median()),
            "positive_row_pct": float(values.gt(0).mean()),
            "negative_row_pct": float(values.lt(0).mean()),
        }
    context: dict[str, Any] = {
        "prediction_lock": lock, "prediction_lock_sha256": lock_hash,
        "stage1_source_population": population,
        "population": evaluated, "fold_audit": fold_audit,
        "reconstruction": reconstruction, "stage1_parity": parity,
        "score_movement": score_movement, "funnel": funnel,
        "daily": daily, "membership": membership, "attribution": attribution,
        "membership_summary": membership_summary,
        "transition_summary": transition_summary, "rank_movement": rank_movement,
        "practical": practical, "practical_summary": practical_summary,
        "bootstrap": bootstrap, "lodo": lodo,
        "self_label_leakage_rows": int(fold_audit.self_label_leakage_rows.sum()),
        "current_test_leakage_rows": int(fold_audit.current_test_leakage_rows.sum()),
        "july_result_rows_accessed": 0, "outcome_perturbation": "PASS",
    }
    context["decision"] = _decision(context)
    review = render_review(context)
    population_columns = [
        *lock.columns, "target7", "loss", "raw_repair_return", "capped_return_7",
        "opportunity_bucket",
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(lock),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(evaluated[population_columns]),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(membership),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(attribution),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_board3_confirmed_harmful_ablation(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path / "reports/research/v004c_board3_confirmed_harmful_ablation_v001_20260603_20260626"
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

"""v004c Board3 historical-extension and temporal-compatibility audit.

Only the earliest allowed historical training date differs between the locked
RECENT_ONLY comparator and EXTENDED_HISTORY.  The candidate definition,
full-date feature ranks, frozen 18-feature representation, weighted L2
logistic learner, target, weights, 18-date gate, and risk-tail diagnostics are
unchanged.  July outcome access is prohibited.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v004a import DEFAULT_TARGET_COLUMN
from src.v004c_board3_only_v4a_arch_feasibility import (
    _fit_frozen,
    _score_no_outcomes,
    assign_internal_rank,
    same_date_pair_summary,
)
from src.v004c_board3_oof_risk_selectivity import (
    EXPECTED_COMMON_DATES,
    EXPECTED_LOCK_SHA256,
    _capture_vector,
    _date_count_vector,
    _metric_summary,
    _pair_date_components,
    _point_metrics,
    _weighted_auc_vector,
    build_selectivity_curve,
    date_bootstrap_counts,
    derive_relative_risk_surface,
    load_locked_population,
    tail_selectivity,
)
from src.v004c_pair_capped7_july_forward import (
    attach_training_outcomes,
    derive_trade_dates,
    load_mature_training_outcomes,
    prepare_frozen_x,
)
from src.v004c_stage1_risk_complementarity import prepare_diagnostic_samples
from src.v004c_stage1_top3_risk_information import binary_auc
from src.v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2,
    MODEL_FAMILY,
    POSITIVE_WEIGHT,
    RAW_INPUT_COLUMNS,
    dataframe_csv_bytes,
    reconstruct_canonical_raw_row,
)


TASK_NAME = "BOARD3 HISTORICAL EXTENSION & TEMPORAL COMPATIBILITY AUDIT"
EXPECTED_STARTING_HEAD = "2445eea52e8e4999bb6f122a1cc34b0cb0c9ea3a"
HISTORICAL_BACKCAST_REPLICATION = True
PRISTINE_OOT = False
NEW_FEATURE = False
NEW_LEARNER = False
HYPERPARAMETER_SEARCH = False
WINDOW_SEARCH = False
MONTH_EXCLUSION = False
RISK_THRESHOLD_SEARCH = False
VETO_BACKFILL = False
JULY_RESULT_ROWS_ACCESSED = 0

RECENT_START = "2026-05-01"
EXTENDED_START = "2026-01-01"
JULY_SENTINEL = "2026-07-01"
MIN_TRAIN_BOARD3_SIGNAL_DATES = 18
COMPAT_BOOTSTRAP_SEED = 20260821
EXPANDED_BOOTSTRAP_SEED = 20260822
BOOTSTRAP_RESAMPLES = 20_000

RECOVERY_REL = Path("data/cache/v004c_board3_historical_extension_v001")
CANDIDATE_FILENAME = "jan_apr_authoritative_candidates.csv"
DAILY_AUDIT_FILENAME = "daily_recovery_audit.csv"
POOL_AUDIT_FILENAME = "pool_recovery_audit.csv"
MINUTE_AUDIT_FILENAME = "minute_recovery_audit.csv"

OUTPUT_FILENAMES = (
    "v004c_board3_historical_data_coverage_v001.csv",
    "v004c_board3_historical_population_monthly_v001.csv",
    "v004c_board3_historical_oof_predictions_v001.csv",
    "v004c_board3_historical_oof_monthly_v001.csv",
    "v004c_board3_historical_risk_selectivity_v001.csv",
    "v004c_board3_history_compatibility_shared_dates_v001.csv",
    "v004c_board3_history_compatibility_coefficients_v001.csv",
    "v004c_board3_history_compatibility_robustness_v001.csv",
    "v004c_board3_temporal_drift_v001.csv",
    "v004c_board3_historical_extension_review_v001.md",
)


def assert_contract() -> None:
    if MODEL_FAMILY != "WEIGHTED_L2_LOGISTIC" or L2 != .30 or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen Board3 learner contract changed")
    if len(FROZEN_FEATURE_COLUMNS) != 18 or len(set(FROZEN_FEATURE_COLUMNS)) != 18:
        raise RuntimeError("FATAL: frozen 18-feature contract changed")
    if MIN_TRAIN_BOARD3_SIGNAL_DATES != 18:
        raise RuntimeError("FATAL: minimum Board3 history gate changed")
    if any((NEW_FEATURE, NEW_LEARNER, HYPERPARAMETER_SEARCH, WINDOW_SEARCH,
            MONTH_EXCLUSION, RISK_THRESHOLD_SEARCH, VETO_BACKFILL)):
        raise RuntimeError("FATAL: unauthorized experiment dimension enabled")
    if JULY_RESULT_ROWS_ACCESSED != 0:
        raise RuntimeError("FATAL: July result access is prohibited")


def extended_fold_is_eligible(train_dates: int, target_classes: int) -> tuple[bool, str]:
    if int(train_dates) < MIN_TRAIN_BOARD3_SIGNAL_DATES:
        return False, f"MATURED_BOARD3_DATES_{int(train_dates)}_LT_18"
    if int(target_classes) < 2:
        return False, "UNFITTABLE_SINGLE_CLASS"
    return True, ""


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"FATAL: required historical recovery artifact missing: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_recovery_audits(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recovery = root / RECOVERY_REL
    daily = _read_csv(recovery / DAILY_AUDIT_FILENAME, dtype={"code": str})
    pools = _read_csv(recovery / POOL_AUDIT_FILENAME)
    minute = _read_csv(recovery / MINUTE_AUDIT_FILENAME, dtype={"code": str})
    return daily, pools, minute


def load_historical_candidates(root: Path) -> pd.DataFrame:
    frame = _read_csv(
        root / RECOVERY_REL / CANDIDATE_FILENAME,
        dtype={"event_id": str, "code": str, "signal_date": str},
    )
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["board_streak_before_break"] = pd.to_numeric(
        frame["board_streak_before_break"], errors="raise"
    ).astype(int)
    if not frame["signal_date"].between(EXTENDED_START, "2026-04-30").all():
        raise RuntimeError("FATAL: historical candidates escaped Jan-Apr")
    if not frame["board_streak_before_break"].isin([2, 3]).all():
        raise RuntimeError("FATAL: historical board definition changed")
    if frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate historical event_id")
    return frame.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def build_historical_raw_features(
    root: Path, candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the frozen raw representation, failing closed by event."""
    raw_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, event in enumerate(candidates.to_dict("records"), 1):
        try:
            raw, audit = reconstruct_canonical_raw_row(root, event)
            raw.update({
                "event_id": str(event["event_id"]),
                "board_streak_before_break": int(event["board_streak_before_break"]),
            })
            raw_rows.append(raw)
            audit_rows.append({**audit, "status": "PASS", "error": ""})
        except Exception as exc:
            audit_rows.append({
                "event_id": str(event["event_id"]), "signal_date": str(event["signal_date"]),
                "code": str(event["code"]).zfill(6), "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            })
        if index % 50 == 0 or index == len(candidates):
            print(f"[historical-features] {index}/{len(candidates)}", flush=True)
    raw = pd.DataFrame(raw_rows)
    audit = pd.DataFrame(audit_rows).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if raw.empty:
        raise RuntimeError("FATAL: no Jan-Apr raw feature row reconstructed")
    raw = raw.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)
    if raw["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate historical raw event_id")
    if bool(raw["signal_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: July signal entered historical feature reconstruction")
    missing = sorted(set(RAW_INPUT_COLUMNS).difference(raw.columns))
    if missing:
        raise RuntimeError(f"FATAL: historical frozen raw inputs missing: {missing}")
    return raw, audit


def prepare_historical_samples(
    root: Path, candidates: pd.DataFrame, raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full-date rank first, Board3 filtering later."""
    raw_identity = raw[["event_id"]]
    eligible_candidates = candidates.merge(raw_identity, on="event_id", validate="one_to_one")
    frozen_x = prepare_frozen_x(raw)
    # Date-only maturity and price outcomes remain isolated until X is locked.
    dates = derive_trade_dates(root, eligible_candidates)
    if bool(dates["label_available_date"].fillna("").ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: Jan-Apr candidate unexpectedly matures in July")
    outcomes = load_mature_training_outcomes(root, eligible_candidates, dates)
    samples = attach_training_outcomes(frozen_x, outcomes)
    samples["loss"] = samples["raw_repair_return"].lt(0).astype(int)
    samples["nonloss"] = 1 - samples["loss"]
    samples["positive_non_target"] = (
        samples["raw_repair_return"].ge(0) & samples["raw_repair_return"].lt(.07)
    ).astype(int)
    samples["capped_return_7"] = np.minimum(samples["raw_repair_return"], .07)
    if bool(samples["label_available_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: July outcome entered historical samples")
    return frozen_x, samples, dates


def _complete_may_june_samples(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_x, samples, _dates, audit = prepare_diagnostic_samples(root)
    samples = samples.copy()
    samples["loss"] = samples["raw_repair_return"].lt(0).astype(int)
    samples["nonloss"] = 1 - samples["loss"]
    samples["positive_non_target"] = (
        samples["raw_repair_return"].ge(0) & samples["raw_repair_return"].lt(.07)
    ).astype(int)
    samples["capped_return_7"] = np.minimum(samples["raw_repair_return"], .07)
    return base_x, samples, audit


def _score_extended_folds(
    all_x: pd.DataFrame, all_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    board3_x = all_x[pd.to_numeric(all_x["board_streak_before_break"]).eq(3)].copy()
    board3_samples = all_samples[
        pd.to_numeric(all_samples["board_streak_before_break"]).eq(3)
    ].copy()
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for test_date in sorted(board3_samples["signal_date"].astype(str).unique()):
        train = board3_samples[
            board3_samples["signal_date"].ge(EXTENDED_START)
            & board3_samples["label_available_date"].lt(test_date)
        ].copy()
        train_dates = int(train["signal_date"].nunique())
        classes = int(train[DEFAULT_TARGET_COLUMN].nunique()) if len(train) else 0
        eligible, unavailable_reason = extended_fold_is_eligible(train_dates, classes)
        test_x = board3_x[board3_x["signal_date"].eq(test_date)].copy()
        if bool(pd.to_numeric(train["board_streak_before_break"]).ne(3).any()):
            raise RuntimeError("FATAL: Board2 row entered extended Board3 fit")
        if bool(train["label_available_date"].ge(test_date).any()):
            raise RuntimeError("FATAL: future label entered extended fit")
        fold = {
            "test_date": test_date, "eligible": eligible,
            "unavailable_reason": unavailable_reason,
            "train_rows": int(len(train)), "train_dates": train_dates,
            "train_target7": int(train["target7"].sum()) if len(train) else 0,
            "train_pnt": int(train["positive_non_target"].sum()) if len(train) else 0,
            "train_loss": int(train["loss"].sum()) if len(train) else 0,
            "single_class": classes < 2,
            "earliest_train_date": str(train["signal_date"].min()) if len(train) else "",
            "latest_train_signal_date": str(train["signal_date"].max()) if len(train) else "",
            "latest_label_available_date": str(train["label_available_date"].max()) if len(train) else "",
            "self_label_leakage_rows": int(train["signal_date"].eq(test_date).sum()),
            "current_test_leakage_rows": int(train["label_available_date"].ge(test_date).sum()),
            "board2_training_rows": int(pd.to_numeric(train["board_streak_before_break"]).eq(2).sum()),
            "test_board3_rows": int(len(test_x)), "feature_count": 18,
            "l2": L2, "positive_weight": POSITIVE_WEIGHT,
            "beta_l2_norm": np.nan, "max_abs_beta": np.nan,
        }
        if not eligible:
            fold_rows.append(fold)
            continue
        beta, _weights = _fit_frozen(train)
        scored = _score_no_outcomes(test_x, beta, "extended")
        scored = assign_internal_rank(scored, "extended_score", "extended_internal_rank")
        scored["board3_candidate_count"] = int(len(scored))
        scored["train_rows"] = int(len(train))
        scored["train_dates"] = train_dates
        scored["earliest_train_date"] = str(train["signal_date"].min())
        scored["latest_train_signal_date"] = str(train["signal_date"].max())
        scored["latest_label_available_date"] = str(train["label_available_date"].max())
        predictions.append(scored)
        fold["beta_l2_norm"] = float(np.linalg.norm(beta[1:]))
        fold["max_abs_beta"] = float(np.max(np.abs(beta[1:])))
        fold_rows.append(fold)
        for index, feature in enumerate(["INTERCEPT", *FROZEN_FEATURE_COLUMNS]):
            coefficient_rows.append({
                "test_date": test_date, "feature": feature,
                "extended_beta": float(beta[index]),
            })
    if not predictions:
        raise RuntimeError("FATAL: no extended Board3 OOF fold eligible")
    prediction = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "extended_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    return prediction, pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def build_extended_prediction_lock(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id", "signal_date", "code", "board3_candidate_count",
        *FROZEN_FEATURE_COLUMNS, "extended_logit", "extended_score",
        "extended_internal_rank", "train_rows", "train_dates", "earliest_train_date",
        "latest_train_signal_date", "latest_label_available_date",
    ]
    lock = predictions[columns].copy().sort_values(
        ["signal_date", "extended_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    forbidden = {"target7", "loss", "raw_repair_return", "capped_return_7"}
    if forbidden.intersection(lock.columns):
        raise RuntimeError("FATAL: outcome entered extended prediction lock")
    return lock


def prediction_lock_sha256(lock: pd.DataFrame) -> str:
    return hashlib.sha256(dataframe_csv_bytes(lock)).hexdigest()


def attach_extended_outcomes(lock: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id", "target7", "loss", "nonloss", "positive_non_target",
        "raw_repair_return", "capped_return_7", "label_available_date",
    ]
    frame = lock.merge(samples[columns], on="event_id", validate="one_to_one")
    surface = derive_relative_risk_surface(frame.rename(columns={
        "extended_score": "board3_only_score",
        "extended_internal_rank": "board3_only_internal_rank",
    }))
    frame = frame.merge(
        surface[["event_id", "relative_rank_eligible", "risk_rank_percentile",
                 "bottom_half_flag", "worst_one_flag"]],
        on="event_id", validate="one_to_one",
    )
    return frame.sort_values(
        ["signal_date", "extended_internal_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _risk_view(frame: pd.DataFrame, score: str, rank: str) -> pd.DataFrame:
    renamed = frame.rename(columns={score: "board3_only_score", rank: "board3_only_internal_rank"}).copy()
    required = [
        "event_id", "signal_date", "code", "board3_candidate_count",
        "board3_only_score", "board3_only_internal_rank", "target7", "loss",
        "nonloss", "positive_non_target", "raw_repair_return", "capped_return_7",
    ]
    surface = derive_relative_risk_surface(renamed[required])
    return surface.merge(
        renamed[["event_id", "target7", "loss", "nonloss", "positive_non_target",
                 "raw_repair_return", "capped_return_7"]],
        on="event_id", validate="one_to_one",
    )


def summarize_risk(population: pd.DataFrame) -> dict[str, Any]:
    eligible = population[population["relative_rank_eligible"]].copy()
    bottom = tail_selectivity(eligible, "bottom_half_flag", "BOTTOM_HALF")
    worst = tail_selectivity(eligible, "worst_one_flag", "WORST_ONE")
    pair = same_date_pair_summary(population, "board3_only_score", "nonloss")
    curve, curve_summary = build_selectivity_curve(eligible)
    return {
        "dates": int(population["signal_date"].nunique()), "rows": int(len(population)),
        "target7": int(population["target7"].sum()),
        "pnt": int(population["positive_non_target"].sum()),
        "loss": int(population["loss"].sum()),
        "nonloss_auc": binary_auc(population["nonloss"], population["board3_only_score"]),
        "pair": pair, "eligible": eligible, "bottom": bottom, "worst": worst,
        "curve": curve, "curve_summary": curve_summary,
    }


def risk_robustness(
    population: pd.DataFrame, seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dates, counts = date_bootstrap_counts(
        population["signal_date"], resamples=BOOTSTRAP_RESAMPLES, seed=seed
    )
    point = _point_metrics(population)
    auc = _weighted_auc_vector(population, "nonloss", "board3_only_score", dates, counts)
    pair_num, pair_den = _pair_date_components(
        population, dates, "board3_only_score", "nonloss", "loss"
    )
    pair_den_draw = counts @ pair_den
    pair_values = np.divide(
        counts @ pair_num, pair_den_draw,
        out=np.full(len(counts), np.nan), where=pair_den_draw > 0,
    )
    bottom_loss = _capture_vector(counts, population, dates, "bottom_half_flag", "loss")
    bottom_winner = _capture_vector(counts, population, dates, "bottom_half_flag", "target7")
    worst_loss = _capture_vector(counts, population, dates, "worst_one_flag", "loss")
    worst_winner = _capture_vector(counts, population, dates, "worst_one_flag", "target7")
    values = {
        "NONLOSS_AUC": auc, "PAIR_CONCORDANCE": pair_values,
        "BOTTOM_HALF_LOSS_CAPTURE": bottom_loss,
        "BOTTOM_HALF_WINNER_REMOVAL": bottom_winner,
        "BOTTOM_HALF_SELECTIVITY_GAP": bottom_loss - bottom_winner,
        "WORST_ONE_SELECTIVITY_GAP": worst_loss - worst_winner,
    }
    predicates = {
        "NONLOSS_AUC": lambda x: x > .5, "PAIR_CONCORDANCE": lambda x: x > .5,
        "BOTTOM_HALF_LOSS_CAPTURE": lambda x: x >= .60,
        "BOTTOM_HALF_WINNER_REMOVAL": lambda x: x <= .40,
        "BOTTOM_HALF_SELECTIVITY_GAP": lambda x: x > 0,
        "WORST_ONE_SELECTIVITY_GAP": lambda x: x > 0,
    }
    bootstrap = {
        metric: _metric_summary(value, point[metric], predicates[metric])
        for metric, value in values.items()
    }
    lodo_values = {metric: [] for metric in point}
    for omitted in dates:
        current = _point_metrics(population[~population["signal_date"].eq(omitted)])
        for metric, value in current.items():
            lodo_values[metric].append(value)
    lodo = {
        metric: _metric_summary(value, point[metric], predicates[metric])
        for metric, value in lodo_values.items()
    }
    rows = [
        {"section": section, "metric": metric, **summary}
        for section, summaries in (("BOOTSTRAP", bootstrap), ("LODO", lodo))
        for metric, summary in summaries.items()
    ]
    return pd.DataFrame(rows), bootstrap, lodo


def formal_expanded_risk(
    risk: Mapping[str, Any], bootstrap: Mapping[str, Any], lodo: Mapping[str, Any],
) -> dict[str, Any]:
    bottom, worst, pair, curve = (
        risk["bottom"], risk["worst"], risk["pair"], risk["curve_summary"]
    )
    gates = {
        "A": risk["dates"] >= 8 and risk["rows"] >= 19
             and risk["eligible"]["signal_date"].nunique() >= 5
             and len(risk["eligible"]) >= 10,
        "B": risk["nonloss_auc"] >= .65 and pair["informative_pairs"] >= 10
             and pair["concordance"] >= .75,
        "C": bottom["flagged_loss_rate"] >= bottom["eligible_loss_rate"] + .15,
        "D": bottom["loss_capture_rate"] >= .60,
        "E": bottom["winner_removal_rate"] <= .40,
        "F": bottom["selectivity_gap"] >= .25,
        "G": worst["selectivity_gap"] > 0 and worst["flagged_loss"] > worst["flagged_target7"],
        "H": curve["selectivity_dominance_rate"] >= .70,
        "I": bootstrap["BOTTOM_HALF_SELECTIVITY_GAP"]["direction_probability"] >= .80,
        "J": lodo["BOTTOM_HALF_SELECTIVITY_GAP"]["direction_probability"] >= .75,
        "K": bottom["loss_capture_rate"] > bottom["pnt_removal_rate"]
             or bottom["flagged_loss_rate"] > bottom["flagged_pnt_rate"],
    }
    if not gates["A"]:
        signal = "INSUFFICIENT_EVIDENCE"
    elif all(gates.values()):
        signal = "STRONG"
    elif bottom["selectivity_gap"] <= 0 or bottom["loss_enrichment"] <= 0:
        signal = "ABSENT"
    else:
        signal = "PARTIAL"
    return {"gates": gates, "signal": signal}


def _daily_selected(frame: pd.DataFrame, rank: str, k: int) -> dict[str, float]:
    daily: list[dict[str, float]] = []
    for _, day in frame.groupby("signal_date", sort=True):
        selected = day.sort_values([rank, "event_id"], kind="mergesort").head(min(k, len(day)))
        daily.append({
            "target7": float(selected["target7"].mean()),
            "loss": float(selected["loss"].mean()),
            "capped": float(selected["capped_return_7"].mean()),
        })
    values = pd.DataFrame(daily)
    return {
        "target7": float(values["target7"].mean()),
        "loss": float(values["loss"].mean()),
        "capped": float(values["capped"].mean()),
    }


def shared_date_compatibility(
    root: Path, extended: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    recent, lock_audit = load_locked_population(root)
    if lock_audit["observed_lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise RuntimeError("FATAL: recent locked comparator hash changed")
    shared = extended[extended["signal_date"].isin(EXPECTED_COMMON_DATES)].copy()
    expected_ids = sorted(recent["event_id"].astype(str))
    if sorted(shared["event_id"].astype(str)) != expected_ids:
        raise RuntimeError("FATAL: shared-date extended rows differ from locked 19-row comparator")
    recent_columns = [
        "event_id", "signal_date", "code", "board3_candidate_count", "board3_only_score",
        "board3_only_internal_rank", "target7", "loss", "nonloss",
        "positive_non_target", "raw_repair_return", "capped_return_7",
    ]
    joined = recent[recent_columns].rename(columns={
        "board3_only_score": "recent_score",
        "board3_only_internal_rank": "recent_rank",
    }).merge(
        shared[["event_id", "extended_score", "extended_internal_rank"]],
        on="event_id", validate="one_to_one",
    )
    joined["rank_delta"] = joined["extended_internal_rank"] - joined["recent_rank"]
    joined["outcome"] = np.where(
        joined["loss"].eq(1), "LOSS",
        np.where(joined["target7"].eq(1), "TARGET7", "PNT"),
    )
    recent_risk = _risk_view(joined.rename(columns={
        "recent_score": "score", "recent_rank": "rank",
    }), "score", "rank")
    extended_risk = _risk_view(joined.rename(columns={
        "extended_score": "score", "extended_internal_rank": "rank",
    }), "score", "rank")
    rr, er = summarize_risk(recent_risk), summarize_risk(extended_risk)
    summary = {
        "recent": rr, "extended": er,
        "recent_target7_auc": binary_auc(joined["target7"], joined["recent_score"]),
        "extended_target7_auc": binary_auc(joined["target7"], joined["extended_score"]),
        "recent_rank1": _daily_selected(joined, "recent_rank", 1),
        "extended_rank1": _daily_selected(joined, "extended_internal_rank", 1),
        "recent_top2": _daily_selected(joined, "recent_rank", 2),
        "extended_top2": _daily_selected(joined, "extended_internal_rank", 2),
    }
    ranks = joined.groupby("outcome", sort=True)["rank_delta"].agg(["mean", "median"]).reset_index()
    return joined.sort_values(["signal_date", "event_id"], kind="mergesort"), summary, ranks


def compatibility_robustness(shared: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dates = sorted(shared["signal_date"].unique())
    dates, counts = date_bootstrap_counts(
        dates, resamples=BOOTSTRAP_RESAMPLES, seed=COMPAT_BOOTSTRAP_SEED
    )

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        recent_risk = summarize_risk(_risk_view(
            frame.rename(columns={"recent_score": "score", "recent_rank": "rank"}),
            "score", "rank",
        ))
        extended_risk = summarize_risk(_risk_view(
            frame.rename(columns={"extended_score": "score", "extended_internal_rank": "rank"}),
            "score", "rank",
        ))
        return {
            "NONLOSS_AUC_DELTA": extended_risk["nonloss_auc"] - recent_risk["nonloss_auc"],
            "PAIR_CONCORDANCE_DELTA": extended_risk["pair"]["concordance"] - recent_risk["pair"]["concordance"],
            "LOSS_CAPTURE_DELTA": extended_risk["bottom"]["loss_capture_rate"] - recent_risk["bottom"]["loss_capture_rate"],
            "WINNER_REMOVAL_DELTA": extended_risk["bottom"]["winner_removal_rate"] - recent_risk["bottom"]["winner_removal_rate"],
            "SELECTIVITY_GAP_DELTA": extended_risk["bottom"]["selectivity_gap"] - recent_risk["bottom"]["selectivity_gap"],
            "RANK1_CAPPED_DELTA": _daily_selected(frame, "extended_internal_rank", 1)["capped"] - _daily_selected(frame, "recent_rank", 1)["capped"],
            "TOP2_CAPPED_DELTA": _daily_selected(frame, "extended_internal_rank", 2)["capped"] - _daily_selected(frame, "recent_rank", 2)["capped"],
        }

    point = metrics(shared)
    recent_auc = _weighted_auc_vector(shared, "nonloss", "recent_score", dates, counts)
    extended_auc = _weighted_auc_vector(shared, "nonloss", "extended_score", dates, counts)
    recent_pair_num, recent_pair_den = _pair_date_components(
        shared, dates, "recent_score", "nonloss", "loss"
    )
    extended_pair_num, extended_pair_den = _pair_date_components(
        shared, dates, "extended_score", "nonloss", "loss"
    )

    recent_surface = _risk_view(
        shared.rename(columns={"recent_score": "score", "recent_rank": "rank"}),
        "score", "rank",
    )
    extended_surface = _risk_view(
        shared.rename(columns={"extended_score": "score", "extended_internal_rank": "rank"}),
        "score", "rank",
    )
    recent_loss = _capture_vector(
        counts, recent_surface, dates, "bottom_half_flag", "loss"
    )
    extended_loss = _capture_vector(
        counts, extended_surface, dates, "bottom_half_flag", "loss"
    )
    recent_winner = _capture_vector(
        counts, recent_surface, dates, "bottom_half_flag", "target7"
    )
    extended_winner = _capture_vector(
        counts, extended_surface, dates, "bottom_half_flag", "target7"
    )

    def daily_capped_delta(k: int) -> np.ndarray:
        delta_by_date: list[float] = []
        for date in dates:
            day = shared[shared["signal_date"].eq(date)]
            recent_selected = day.sort_values(
                ["recent_rank", "event_id"], kind="mergesort"
            ).head(min(k, len(day)))
            extended_selected = day.sort_values(
                ["extended_internal_rank", "event_id"], kind="mergesort"
            ).head(min(k, len(day)))
            delta_by_date.append(float(
                extended_selected["capped_return_7"].mean()
                - recent_selected["capped_return_7"].mean()
            ))
        vector = np.asarray(delta_by_date, dtype=float)
        denominator = counts.sum(axis=1)
        return np.divide(
            counts @ vector, denominator,
            out=np.full(len(counts), np.nan), where=denominator > 0,
        )

    recent_pair_draw_den = counts @ recent_pair_den
    extended_pair_draw_den = counts @ extended_pair_den
    recent_pair = np.divide(
        counts @ recent_pair_num, recent_pair_draw_den,
        out=np.full(len(counts), np.nan), where=recent_pair_draw_den > 0,
    )
    extended_pair = np.divide(
        counts @ extended_pair_num, extended_pair_draw_den,
        out=np.full(len(counts), np.nan), where=extended_pair_draw_den > 0,
    )
    values = {
        "NONLOSS_AUC_DELTA": extended_auc - recent_auc,
        "PAIR_CONCORDANCE_DELTA": extended_pair - recent_pair,
        "LOSS_CAPTURE_DELTA": extended_loss - recent_loss,
        "WINNER_REMOVAL_DELTA": extended_winner - recent_winner,
        "SELECTIVITY_GAP_DELTA": (
            extended_loss - extended_winner - (recent_loss - recent_winner)
        ),
        "RANK1_CAPPED_DELTA": daily_capped_delta(1),
        "TOP2_CAPPED_DELTA": daily_capped_delta(2),
    }
    predicates = {
        metric: (lambda x: x <= 0) if metric == "WINNER_REMOVAL_DELTA" else (lambda x: x > 0)
        for metric in point
    }
    bootstrap = {
        metric: _metric_summary(value, point[metric], predicates[metric])
        for metric, value in values.items()
    }
    lodo_values = {metric: [] for metric in point}
    for omitted in dates:
        current = metrics(shared[~shared["signal_date"].eq(omitted)])
        for metric, value in current.items():
            lodo_values[metric].append(value)
    lodo = {
        metric: _metric_summary(value, point[metric], predicates[metric])
        for metric, value in lodo_values.items()
    }
    rows = [
        {"section": section, "metric": metric, **summary}
        for section, summaries in (("COMPAT_BOOTSTRAP", bootstrap), ("COMPAT_LODO", lodo))
        for metric, summary in summaries.items()
    ]
    return pd.DataFrame(rows), bootstrap, lodo


def history_extension_effect(
    compatibility: Mapping[str, Any], bootstrap: Mapping[str, Any],
) -> str:
    recent, extended = compatibility["recent"], compatibility["extended"]
    auc_delta = extended["nonloss_auc"] - recent["nonloss_auc"]
    pair_delta = extended["pair"]["concordance"] - recent["pair"]["concordance"]
    gap_delta = extended["bottom"]["selectivity_gap"] - recent["bottom"]["selectivity_gap"]
    winner_delta = extended["bottom"]["winner_removal_rate"] - recent["bottom"]["winner_removal_rate"]
    loss_delta = extended["bottom"]["loss_capture_rate"] - recent["bottom"]["loss_capture_rate"]
    harmful = (
        auc_delta <= -.10 or pair_delta <= -.10 or gap_delta <= -.15
        or (winner_delta >= .15 and loss_delta < .15)
    )
    if harmful:
        return "HARMFUL"
    supportive_main = (
        auc_delta >= 0 and pair_delta >= -.05 and gap_delta >= 0 and winner_delta <= 0
        and (loss_delta >= .10 or winner_delta <= -.10 or gap_delta >= .10 or auc_delta >= .05)
    )
    supportive_robust = (
        bootstrap["SELECTIVITY_GAP_DELTA"]["direction_probability"] >= .60
        or bootstrap["NONLOSS_AUC_DELTA"]["direction_probability"] >= .65
    )
    if supportive_main and supportive_robust:
        return "SUPPORTIVE"
    if abs(auc_delta) < .05 and abs(gap_delta) < .10 and abs(winner_delta) < .10:
        return "NEUTRAL"
    return "MIXED"


def coefficient_compatibility(
    root: Path, extended_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / (
        "reports/research/v004c_board3_only_v4a_arch_feasibility_v001_20260506_20260626/"
        "v004c_board3_only_coefficients_v001.csv"
    )
    recent = _read_csv(path, dtype={"test_date": str})
    recent = recent[recent["test_date"].isin(EXPECTED_COMMON_DATES)][
        ["test_date", "feature", "board3_only_beta"]
    ].rename(columns={"board3_only_beta": "recent_beta"})
    extended = extended_coefficients[
        extended_coefficients["test_date"].isin(EXPECTED_COMMON_DATES)
    ]
    joined = recent.merge(extended, on=["test_date", "feature"], validate="one_to_one")
    joined["beta_delta"] = joined["extended_beta"] - joined["recent_beta"]
    joined["same_sign"] = np.sign(joined["extended_beta"]) == np.sign(joined["recent_beta"])
    fold_rows: list[dict[str, Any]] = []
    for date, day in joined[joined["feature"].ne("INTERCEPT")].groupby("test_date", sort=True):
        a, b = day["recent_beta"].to_numpy(float), day["extended_beta"].to_numpy(float)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        fold_rows.append({
            "test_date": date,
            "cosine_similarity": float(np.dot(a, b) / denominator) if denominator else np.nan,
            "l2_distance": float(np.linalg.norm(b - a)),
            "sign_flips": int((np.sign(a) != np.sign(b)).sum()),
        })
    folds = pd.DataFrame(fold_rows)
    return joined, {
        "folds": folds,
        "median_cosine": float(folds["cosine_similarity"].median()),
        "min_cosine": float(folds["cosine_similarity"].min()),
        "max_cosine": float(folds["cosine_similarity"].max()),
        "median_sign_flips": float(folds["sign_flips"].median()),
    }


def period_risk(population: pd.DataFrame, before_may: bool) -> dict[str, Any]:
    mask = population["signal_date"].lt(RECENT_START)
    selected = population[mask if before_may else ~mask].copy()
    if selected.empty:
        return {"dates": 0, "rows": 0, "nonloss_auc": np.nan, "pair": {},
                "bottom": {}, "worst": {}}
    return summarize_risk(selected)


def temporal_compatibility(
    early: Mapping[str, Any], bridge: Mapping[str, Any], coefficients: pd.DataFrame,
) -> tuple[str, float, int]:
    early_coeff = coefficients[
        coefficients["test_date"].lt(RECENT_START)
        & coefficients["feature"].ne("INTERCEPT")
    ].groupby("feature")["extended_beta"].median().reindex(FROZEN_FEATURE_COLUMNS)
    bridge_coeff = coefficients[
        coefficients["test_date"].ge(RECENT_START)
        & coefficients["feature"].ne("INTERCEPT")
    ].groupby("feature")["extended_beta"].median().reindex(FROZEN_FEATURE_COLUMNS)
    if early_coeff.isna().any() or bridge_coeff.isna().any():
        return "INSUFFICIENT_EVIDENCE", np.nan, 0
    a, b = early_coeff.to_numpy(float), bridge_coeff.to_numpy(float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denominator) if denominator else np.nan
    disagreements = int((np.sign(a) != np.sign(b)).sum())
    early_gap = early["bottom"]["selectivity_gap"]
    bridge_gap = bridge["bottom"]["selectivity_gap"]
    drifted = (
        (early["nonloss_auc"] > .60 and bridge["nonloss_auc"] < .45)
        or (bridge["nonloss_auc"] > .60 and early["nonloss_auc"] < .45)
        or (np.sign(early_gap) != np.sign(bridge_gap) and abs(early_gap - bridge_gap) >= .15)
        or cosine < .40
    )
    stable = (
        early["nonloss_auc"] > .55 and bridge["nonloss_auc"] > .55
        and early_gap > 0 and bridge_gap > 0 and cosine >= .75
    )
    return ("DRIFTED" if drifted else "STABLE" if stable else "PARTIAL"), cosine, disagreements


def monthly_population(
    candidate_identity: pd.DataFrame, samples: pd.DataFrame, oof: pd.DataFrame,
) -> pd.DataFrame:
    board3 = samples[pd.to_numeric(samples["board_streak_before_break"]).eq(3)].copy()
    rows: list[dict[str, Any]] = []
    for month in pd.period_range("2026-01", "2026-06", freq="M").astype(str):
        universe = candidate_identity[candidate_identity["signal_date"].str.startswith(month)]
        pop = board3[board3["signal_date"].str.startswith(month)]
        pred = oof[oof["signal_date"].str.startswith(month)]
        risk = summarize_risk(_risk_view(pred, "extended_score", "extended_internal_rank")) if len(pred) else None
        rows.append({
            "month": month, "candidate_dates": int(universe["signal_date"].nunique()),
            "all_candidates": int(len(universe)),
            "board2_candidates": int(pd.to_numeric(universe["board_streak_before_break"]).eq(2).sum()),
            "board3_candidates": int(pd.to_numeric(universe["board_streak_before_break"]).eq(3).sum()),
            "matured_board3_rows": int(len(pop)), "target7": int(pop["target7"].sum()),
            "pnt": int(pop["positive_non_target"].sum()), "loss": int(pop["loss"].sum()),
            "target7_rate": float(pop["target7"].mean()) if len(pop) else np.nan,
            "loss_rate": float(pop["loss"].mean()) if len(pop) else np.nan,
            "raw_mean": float(pop["raw_repair_return"].mean()) if len(pop) else np.nan,
            "raw_median": float(pop["raw_repair_return"].median()) if len(pop) else np.nan,
            "capped_mean": float(pop["capped_return_7"].mean()) if len(pop) else np.nan,
            "worst_raw": float(pop["raw_repair_return"].min()) if len(pop) else np.nan,
            "oof_dates": 0 if risk is None else risk["dates"],
            "oof_rows": 0 if risk is None else risk["rows"],
            "nonloss_auc": np.nan if risk is None else risk["nonloss_auc"],
            "pair_concordance": np.nan if risk is None else risk["pair"]["concordance"],
            "bottom_loss_capture": np.nan if risk is None else risk["bottom"]["loss_capture_rate"],
            "bottom_winner_removal": np.nan if risk is None else risk["bottom"]["winner_removal_rate"],
            "bottom_selectivity_gap": np.nan if risk is None else risk["bottom"]["selectivity_gap"],
            "month_support_weak": bool(len(pred) < 10),
        })
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    risk = context["expanded_risk"]
    recent, extended = context["compatibility"]["recent"], context["compatibility"]["extended"]
    lines = [
        "# v004c Board3 Historical Extension & Temporal Compatibility Audit v001", "",
        "## 1. Frozen Experimental Contract", "",
        "- HISTORICAL_BACKCAST_REPLICATION: YES", "- PRISTINE_OOT: NO",
        "- Frozen features / learner / L2 / weighting: PASS", "- July outcome rows accessed: 0", "",
        "## 2. Historical Data Coverage", "",
        f"- data status: {context['data_status']}",
        f"- raw candidates / reconstructed: {context['candidate_rows']} / {context['reconstructed_rows']}",
        f"- missing critical inputs: {context['missing_critical_inputs']}", "",
        "## 3. Existing May-Jun Parity", "",
        "- 319 rows / 39 dates / Board2 261 / Board3 58: PASS", "",
        "## 4. Jan-Jun Authoritative Candidate Population", "",
        f"- matured Board3 rows / dates: {context['board3_rows']} / {context['board3_dates']}", "",
        "## 5. Monthly Board3 Opportunity Base Rates", "",
        context["monthly"].to_markdown(index=False), "",
        "## 6. Extended Chronological OOF Coverage", "",
        f"- first eligible date: {context['first_eligible_date']}",
        f"- dates / rows: {risk['dates']} / {risk['rows']}", "",
        "## 7. Expanded Board3 Ranking / Risk Selectivity", "",
        f"- NONLOSS AUC: {_fmt(risk['nonloss_auc'])}",
        f"- pair concordance: {_fmt(risk['pair']['concordance'])}",
        f"- bottom-half loss capture / winner removal / gap: "
        f"{_fmt(risk['bottom']['loss_capture_rate'])} / {_fmt(risk['bottom']['winner_removal_rate'])} / {_fmt(risk['bottom']['selectivity_gap'])}",
        f"- formal signal: {context['expanded_signal']}", "",
        "## 8. Early Historical Backcast Replication", "",
        f"- dates / rows: {context['early']['dates']} / {context['early']['rows']}",
        f"- NONLOSS AUC: {_fmt(context['early']['nonloss_auc'])}", "",
        "## 9. May-Jun Bridge Performance", "",
        f"- dates / rows: {context['bridge']['dates']} / {context['bridge']['rows']}",
        f"- NONLOSS AUC: {_fmt(context['bridge']['nonloss_auc'])}", "",
        "## 10. Recent-Only vs Extended-History Shared-Date Comparison", "",
        f"- recent NONLOSS AUC / gap: {_fmt(recent['nonloss_auc'])} / {_fmt(recent['bottom']['selectivity_gap'])}",
        f"- extended NONLOSS AUC / gap: {_fmt(extended['nonloss_auc'])} / {_fmt(extended['bottom']['selectivity_gap'])}",
        f"- HISTORY_EXTENSION_EFFECT: {context['history_effect']}", "",
        "## 11. Coefficient Compatibility", "",
        f"- median shared-fold cosine: {_fmt(context['coefficient_summary']['median_cosine'])}", "",
        "## 12. Bootstrap / LODO Compatibility", "",
        f"- P(extended selectivity gap > recent): {_fmt(context['compat_bootstrap']['SELECTIVITY_GAP_DELTA']['direction_probability'])}", "",
        "## 13. Temporal Drift Diagnostic", "",
        f"- median-vector cosine: {_fmt(context['temporal_cosine'])}",
        f"- BOARD3_TEMPORAL_COMPATIBILITY: {context['temporal_compatibility']}", "",
        "## 14. Historical Extension Decision", "",
        f"- BOARD3_HISTORICAL_EXTENSION_CONCLUSION: {context['conclusion']}", "",
        "## 15. Next Board3 Action", "",
        f"- NEXT_BOARD3_ACTION: {context['next_action']}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return dataframe_csv_bytes(frame)


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_contract()
    daily_audit, pool_audit, minute_audit = load_recovery_audits(root)
    candidates = load_historical_candidates(root)
    historical_raw, feature_audit = build_historical_raw_features(root, candidates)
    historical_x, historical_samples, _historical_dates = prepare_historical_samples(
        root, candidates, historical_raw
    )
    may_june_x, may_june_samples, may_june_audit = _complete_may_june_samples(root)
    if may_june_audit != {
        "rows": 319, "dates": 39, "may_rows": 146, "may_dates": 18,
        "june_rows": 173, "june_dates": 21, "board2": 261, "board3": 58,
        "july_result_rows_accessed": 0,
    }:
        raise RuntimeError("FATAL: existing May-Jun parity changed")
    all_x = pd.concat([historical_x, may_june_x], ignore_index=True).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    all_samples = pd.concat([historical_samples, may_june_samples], ignore_index=True).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if bool(all_samples["label_available_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: July outcome entered Jan-Jun samples")

    prediction_inputs, folds, coefficients = _score_extended_folds(all_x, all_samples)
    lock = build_extended_prediction_lock(prediction_inputs)
    lock_sha = prediction_lock_sha256(lock)
    oof = attach_extended_outcomes(lock, all_samples)
    risk_population = _risk_view(oof, "extended_score", "extended_internal_rank")
    expanded_risk = summarize_risk(risk_population)
    expanded_robustness, expanded_bootstrap, expanded_lodo = risk_robustness(
        risk_population, EXPANDED_BOOTSTRAP_SEED
    )
    expanded_decision = formal_expanded_risk(expanded_risk, expanded_bootstrap, expanded_lodo)

    shared, compatibility, rank_movements = shared_date_compatibility(root, oof)
    compat_robustness, compat_bootstrap, compat_lodo = compatibility_robustness(shared)
    history_effect = history_extension_effect(compatibility, compat_bootstrap)
    coefficient_rows, coefficient_summary = coefficient_compatibility(root, coefficients)
    early = period_risk(risk_population, before_may=True)
    bridge = period_risk(risk_population, before_may=False)
    temporal_state, temporal_cosine, temporal_sign_disagreements = temporal_compatibility(
        early, bridge, coefficients
    )
    candidate_identity = pd.concat([
        candidates[["event_id", "signal_date", "code", "board_streak_before_break"]],
        may_june_x[["event_id", "signal_date", "code", "board_streak_before_break"]],
    ], ignore_index=True).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    monthly = monthly_population(candidate_identity, all_samples, oof)

    historical_board3 = historical_samples[
        pd.to_numeric(historical_samples["board_streak_before_break"]).eq(3)
    ]
    combined_board3 = all_samples[
        pd.to_numeric(all_samples["board_streak_before_break"]).eq(3)
    ]
    coverage_failures = feature_audit[feature_audit["status"].eq("FAIL")]
    data_status = "COMPLETE" if coverage_failures.empty else "PARTIAL_COVERAGE"
    support_increased = expanded_risk["dates"] >= 16 or expanded_risk["rows"] >= 38
    jan_apr_increment_ok = (
        historical_board3["signal_date"].nunique() >= 10 and len(historical_board3) >= 20
    )
    if not jan_apr_increment_ok or not support_increased:
        conclusion = "INSUFFICIENT_EXTENSION"
    elif history_effect == "HARMFUL":
        conclusion = "REJECT_OLD_HISTORY_EXTENSION"
    elif data_status == "COMPLETE" and history_effect in {"SUPPORTIVE", "NEUTRAL"} \
            and temporal_state in {"STABLE", "PARTIAL"}:
        conclusion = "USEFUL_AND_COMPATIBLE"
    elif history_effect != "HARMFUL":
        conclusion = "USEFUL_BUT_PARTIAL"
    else:
        conclusion = "INVALID"
    expanded_signal = expanded_decision["signal"]
    if conclusion == "USEFUL_AND_COMPATIBLE" and expanded_signal == "STRONG":
        next_action = "FREEZE_EXTENDED_BOARD3_FOR_JULY_STRESS"
    elif conclusion == "REJECT_OLD_HISTORY_EXTENSION":
        next_action = "REJECT_EXTENDED_HISTORY"
    elif not jan_apr_increment_ok:
        next_action = "INSUFFICIENT_HISTORY"
    elif expanded_signal == "ABSENT":
        next_action = "STOP_BOARD3_RISK_ROUTE"
    else:
        next_action = "NO_RISK_POLICY_YET"

    coverage = pd.concat([
        daily_audit.assign(section="DAILY"),
        pool_audit.assign(section="LIMIT_UP_POOL"),
        minute_audit.assign(section="BAOSTOCK_5M"),
        feature_audit.assign(section="FEATURE_RECONSTRUCTION"),
    ], ignore_index=True, sort=False)
    risk_rows = pd.DataFrame([
        {"period": "EXPANDED", "metric": "NONLOSS_AUC", "value": expanded_risk["nonloss_auc"]},
        {"period": "EXPANDED", "metric": "PAIR_CONCORDANCE", "value": expanded_risk["pair"]["concordance"]},
        {"period": "EXPANDED", "metric": "BOTTOM_LOSS_CAPTURE", "value": expanded_risk["bottom"]["loss_capture_rate"]},
        {"period": "EXPANDED", "metric": "BOTTOM_WINNER_REMOVAL", "value": expanded_risk["bottom"]["winner_removal_rate"]},
        {"period": "EXPANDED", "metric": "BOTTOM_SELECTIVITY_GAP", "value": expanded_risk["bottom"]["selectivity_gap"]},
        {"period": "EXPANDED", "metric": "WORST_ONE_SELECTIVITY_GAP", "value": expanded_risk["worst"]["selectivity_gap"]},
    ])
    temporal = pd.DataFrame([
        {"metric": "EARLY_NONLOSS_AUC", "value": early["nonloss_auc"]},
        {"metric": "BRIDGE_NONLOSS_AUC", "value": bridge["nonloss_auc"]},
        {"metric": "EARLY_SELECTIVITY_GAP", "value": early["bottom"]["selectivity_gap"]},
        {"metric": "BRIDGE_SELECTIVITY_GAP", "value": bridge["bottom"]["selectivity_gap"]},
        {"metric": "COEFFICIENT_MEDIAN_VECTOR_COSINE", "value": temporal_cosine},
        {"metric": "COEFFICIENT_SIGN_DISAGREEMENTS", "value": temporal_sign_disagreements},
    ])
    robustness = pd.concat([compat_robustness, expanded_robustness], ignore_index=True)
    coefficient_output = coefficient_rows.merge(
        coefficient_summary["folds"], on="test_date", how="left", validate="many_to_one"
    )
    context: dict[str, Any] = {
        "data_status": data_status,
        "candidate_rows": int(len(candidates)), "reconstructed_rows": int(len(historical_raw)),
        "missing_critical_inputs": int(len(coverage_failures)),
        "board3_rows": int(len(combined_board3)),
        "board3_dates": int(combined_board3["signal_date"].nunique()),
        "historical_board3_rows": int(len(historical_board3)),
        "historical_board3_dates": int(historical_board3["signal_date"].nunique()),
        "total_candidates": int(len(candidate_identity)),
        "candidate_dates": int(candidate_identity["signal_date"].nunique()),
        "board2_candidates": int(pd.to_numeric(candidate_identity["board_streak_before_break"]).eq(2).sum()),
        "board3_candidates": int(pd.to_numeric(candidate_identity["board_streak_before_break"]).eq(3).sum()),
        "first_eligible_date": str(oof["signal_date"].min()),
        "extended_lock_sha256": lock_sha, "expanded_risk": expanded_risk,
        "expanded_bootstrap": expanded_bootstrap, "expanded_lodo": expanded_lodo,
        "expanded_signal": expanded_signal, "expanded_gates": expanded_decision["gates"],
        "early": early, "bridge": bridge, "compatibility": compatibility,
        "compat_bootstrap": compat_bootstrap, "compat_lodo": compat_lodo,
        "history_effect": history_effect, "coefficient_summary": coefficient_summary,
        "temporal_compatibility": temporal_state, "temporal_cosine": temporal_cosine,
        "temporal_sign_disagreements": temporal_sign_disagreements,
        "conclusion": conclusion, "next_action": next_action,
        "monthly": monthly, "rank_movements": rank_movements,
        "support_increased": support_increased, "july_result_rows_accessed": 0,
        "may_june_parity": may_june_audit,
    }
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(coverage),
        OUTPUT_FILENAMES[1]: _csv_bytes(monthly),
        OUTPUT_FILENAMES[2]: _csv_bytes(oof),
        OUTPUT_FILENAMES[3]: _csv_bytes(monthly[[
            "month", "oof_dates", "oof_rows", "nonloss_auc", "pair_concordance",
            "bottom_loss_capture", "bottom_winner_removal", "bottom_selectivity_gap",
            "month_support_weak",
        ]]),
        OUTPUT_FILENAMES[4]: _csv_bytes(risk_rows),
        OUTPUT_FILENAMES[5]: _csv_bytes(shared),
        OUTPUT_FILENAMES[6]: _csv_bytes(coefficient_output),
        OUTPUT_FILENAMES[7]: _csv_bytes(robustness),
        OUTPUT_FILENAMES[8]: _csv_bytes(temporal),
        OUTPUT_FILENAMES[9]: render_review(context).encode("utf-8"),
        "v004c_board3_historical_oof_prediction_lock_v001.csv": _csv_bytes(lock),
        "v004c_board3_historical_fold_audit_v001.csv": _csv_bytes(folds),
    }
    return outputs, context


def run_v004c_board3_historical_extension(root: Path, output_dir: Path) -> dict[str, Any]:
    outputs, context = build_outputs(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        (output_dir / filename).write_bytes(payload)
    return context

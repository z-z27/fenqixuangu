"""One-shot August OOT verdict for one frozen regime-aware Stage1 challenger.

The prediction phase uses the exact mature May-July S2 training population,
fits exactly two weighted-L2 logistic models, and locks all August scores and
ranks before any August outcome is attached.  The evaluation phase consumes
only August labels available before 2026-09-01.  No tuning, alternative
relation, threshold search, August refit, or post-verdict model is exposed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import get_data_config, get_strategy_config
from src.indicators import enrich_5min_indicators, enrich_daily_indicators
from src.loaders import (
    _is_main_board_limit_up_day,
    _keep_recent_trade_days,
    _merge_minute_amount,
)
from src.legacy_calendar_semantics import generate_legacy_calendar_signal
from src.v004a import (
    DEFAULT_TARGET_COLUMN,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
)
from src.v004c_pair_capped7_july_forward import (
    _load_name_map,
    _normalise_daily,
    _read_daily,
    _read_daily_through,
    _read_minute_through,
    prepare_frozen_x,
)
from src.v004c_reduced7f_stage1_temporal_validation import (
    REDUCED_FEATURE_COLUMNS,
    load_bridge,
)
from src.v004c_reduced7f_training_spec_sanity import (
    SPECIFICATIONS,
    build_sample_weight,
    fit_spec,
)


TASK_NAME = "v004c_regime_aware_stage1_august_oot_v001"
OUTPUT_DIRNAME = "v004c_regime_aware_stage1_august_oot_v001"

TRAINING_ASOF = "2026-08-01"
EVALUATION_ASOF = "2026-09-01"
AUGUST_START = "2026-08-01"
AUGUST_END = "2026-08-31"
TRAINING_ROWS_EXPECTED = 485
TRAINING_DATES_EXPECTED = 60
S2_SPEC = "S2_NO_TAIL_L2_010"
L2 = 0.10
POSITIVE_WEIGHT = 1.50
BOARD4PLUS_CUTOFF = 1.0
PAIR_ORDER_PARITY_MIN = 0.99

BASELINE_MODEL = "BASELINE_S2"
CHALLENGER_MODEL = "CHALLENGER_REGIME_SIGNED"
MODEL_ORDER = (BASELINE_MODEL, CHALLENGER_MODEL)
REGIME_FEATURE = "regime_signed_close_vwap_rank"
CHALLENGER_FEATURE_COLUMNS = (*REDUCED_FEATURE_COLUMNS, REGIME_FEATURE)

BOOTSTRAP_SEED = 20260901
BOOTSTRAP_RESAMPLES = 10_000

DEVELOPMENT_PATH_AUTHORITY = (
    "reports/research/"
    "v004c_d1_unfinished_repair_information_audit_v001_20260506_20260731/"
    "v004c_unfinished_repair_population_v001.csv"
)
DEVELOPMENT_REGIME_AUTHORITY = (
    "reports/research/v004c_board_break_regime_audit_v001_202605_202607/"
    "v004c_regime_daily_ecology_v001.csv"
)

OUTPUT_FILENAMES = (
    "v004c_august_prediction_lock_v001.csv",
    "v004c_august_oot_population_v001.csv",
    "v004c_august_regime_hypothesis_check_v001.csv",
    "v004c_august_baseline_challenger_summary_v001.csv",
    "v004c_august_rankwise_v001.csv",
    "v004c_august_daily_paired_v001.csv",
    "v004c_august_membership_changes_v001.csv",
    "v004c_august_regime_split_performance_v001.csv",
    "v004c_august_paired_bootstrap_v001.csv",
    "v004c_august_lodo_v001.csv",
    "v004c_august_model_coefficients_v001.csv",
    "v004c_august_feature_parity_v001.csv",
    "v004c_regime_aware_stage1_august_oot_review_v001.md",
)

OUTCOME_COLUMNS = {
    "target7", "positive_non_target", "loss", "severe_loss",
    "raw_repair_return", "capped_return_7", "d2_open_daily", "d3_high_daily",
}


def assert_contract() -> None:
    if tuple(REDUCED_FEATURE_COLUMNS) != (
        "rank_d1_close_ma10_pct",
        "rank_d1_low_ma10_pct",
        "rank_trend_hold_score",
        "rank_theme_score",
        "rank_log_candidate_base_price",
        "rank_active_money_score",
        "rank_d1_close_vwap_pct",
    ):
        raise RuntimeError("FATAL: frozen S2 7F manifest changed")
    if (
        float(SPECIFICATIONS[S2_SPEC]["l2"]) != L2
        or str(SPECIFICATIONS[S2_SPEC]["tail_weighting"]) != "NONE"
    ):
        raise RuntimeError("FATAL: frozen S2 training specification changed")
    if MODEL_ORDER != (BASELINE_MODEL, CHALLENGER_MODEL):
        raise RuntimeError("FATAL: model count/identity changed")
    if CHALLENGER_FEATURE_COLUMNS[:-1] != tuple(REDUCED_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: challenger changed frozen S2 features")
    if CHALLENGER_FEATURE_COLUMNS[-1] != REGIME_FEATURE:
        raise RuntimeError("FATAL: challenger contains a second new feature")
    if BOARD4PLUS_CUTOFF != 1.0 or L2 != 0.10 or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: preregistered threshold/hyperparameter changed")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -35.0, 35.0)))


def _rank(frame: pd.DataFrame, score: str, rank_name: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, day in frame.groupby("signal_date", sort=True):
        part = day.sort_values(
            [score, "event_id"], ascending=[False, True], kind="mergesort"
        ).copy()
        part[rank_name] = np.arange(1, len(part) + 1, dtype=int)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["signal_date", rank_name, "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _board_group(streak: pd.Series) -> pd.Series:
    return pd.to_numeric(streak, errors="raise").map({2: "BOARD2", 3: "BOARD3"})


def build_august_universe(root: str | Path) -> pd.DataFrame:
    """Rebuild August v004c identity from daily prices without reading outcomes."""
    root_path = Path(root).resolve()
    names = _load_name_map(root_path)
    rows: list[dict[str, Any]] = []
    for path in sorted((root_path / "data/cache/daily").glob("*_daily.pkl")):
        code = path.name[:6]
        daily = _normalise_daily(pd.read_pickle(path))
        daily = daily[daily["date"].le(AUGUST_END)].reset_index(drop=True)
        if len(daily) < 3:
            continue
        flags = np.zeros(len(daily), dtype=bool)
        for index in range(1, len(daily)):
            flags[index] = _is_main_board_limit_up_day(daily, index)
        for index in range(1, len(daily)):
            signal_date = str(daily.iloc[index]["date"])
            if not (AUGUST_START <= signal_date <= AUGUST_END):
                continue
            if flags[index] or not flags[index - 1]:
                continue
            cursor = index - 1
            streak = 0
            while cursor >= 0 and flags[cursor]:
                streak += 1
                cursor -= 1
            if streak not in (2, 3):
                continue
            rows.append({
                "event_id": f"{code}_{signal_date}",
                "code": code,
                "name": names.get(code, ""),
                "signal_date": signal_date,
                "break_date": signal_date,
                "board_streak_before_break": streak,
            })
    universe = pd.DataFrame(rows).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if universe.empty or universe["event_id"].duplicated().any():
        raise RuntimeError("FATAL: August authoritative identity unavailable or duplicated")
    if not universe["board_streak_before_break"].isin([2, 3]).all():
        raise RuntimeError("FATAL: non-v004c board streak entered August universe")
    if not universe["signal_date"].between(AUGUST_START, AUGUST_END).all():
        raise RuntimeError("FATAL: non-August signal date entered OOT universe")
    universe["board_group"] = _board_group(universe["board_streak_before_break"])
    universe["candidate_count"] = universe.groupby("signal_date")["event_id"].transform("size")
    return universe


def _load_august_five_day_pool(root: Path, signal_date: str) -> pd.DataFrame:
    anchor = pd.Timestamp(signal_date)
    frames: list[pd.DataFrame] = []
    for offset in range(5):
        date = (anchor - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        if date > signal_date:
            raise RuntimeError("FATAL: future limit-up pool requested")
        path = root / "data/cache/limit_ups" / f"{date}_limitups.pkl"
        if not path.exists():
            continue
        frame = pd.read_pickle(path).copy()
        required = {"trade_date", "code", "industry"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"FATAL: limit-up pool schema incomplete: {path}")
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame[frame["trade_date"].eq(date)].copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"FATAL: D1-safe five-day pool unavailable for {signal_date}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "code"], kind="mergesort"
    ).drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True)


def _board4plus_count(root: Path, signal_date: str) -> int:
    path = root / "data/cache/limit_ups" / f"{signal_date}_limitups.pkl"
    if not path.exists():
        raise RuntimeError(f"FATAL: D1 final-limit-up pool missing: {signal_date}")
    frame = pd.read_pickle(path)
    streak = pd.to_numeric(frame.get("consecutive_limit_up_count"), errors="coerce")
    if len(streak) != len(frame) or streak.isna().any():
        raise RuntimeError(f"FATAL: board streak unavailable in pool: {signal_date}")
    return int(streak.ge(4).sum())


def reconstruct_august_raw_row(root: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    code = str(event["code"]).zfill(6)
    signal_date = str(event["signal_date"])
    if not (AUGUST_START <= signal_date <= AUGUST_END):
        raise RuntimeError("FATAL: feature reconstruction outside August")
    daily_raw = _read_daily_through(root, code, signal_date)
    minute_raw = _read_minute_through(root, code, signal_date)
    pool = _load_august_five_day_pool(root, signal_date)
    if len(daily_raw) < 2:
        raise RuntimeError(f"FATAL: D0 unavailable for {event['event_id']}")
    d0 = str(daily_raw.iloc[-2]["date"])
    config = get_data_config()
    merged_daily = _merge_minute_amount(daily_raw, minute_raw)
    daily_history = _keep_recent_trade_days(
        merged_daily, "date", config.daily_history_days
    )
    daily_recent = _keep_recent_trade_days(
        merged_daily, "date", config.default_5min_days
    )
    minute_recent = _keep_recent_trade_days(
        minute_raw, "trade_date", config.default_5min_days
    )
    daily = enrich_daily_indicators(daily_recent, full_daily=daily_history)
    minute = enrich_5min_indicators(minute_recent)
    signal = generate_legacy_calendar_signal(
        code=code,
        name=str(event.get("name", "")),
        daily=daily,
        minute=minute,
        limit_up_pool=pool,
        config=get_strategy_config(),
        d0_date=d0,
    )
    d1 = daily_raw.iloc[-1]
    high = float(d1["high"])
    low = float(d1["low"])
    close = float(d1["close"])
    open_ = float(d1["open"])
    close_location = np.nan if high == low else (close - low) / (high - low)
    raw_vwap = float(signal.d1_close_vwap_pct) / 100.0
    return {
        "event_id": str(event["event_id"]),
        "signal_date": signal_date,
        "code": code,
        "board_streak_before_break": int(event["board_streak_before_break"]),
        "d1_close_ma10_pct": signal.d1_close_ma10_pct,
        "d1_low_ma10_pct": signal.d1_low_ma10_pct,
        "trend_hold_score": signal.trend_hold_score,
        "total_score": signal.total_score,
        "theme_score": signal.theme_score,
        "days_since_d0": signal.days_since_d0,
        "candidate_base_price": close,
        "active_money_score": signal.active_money_score,
        "d1_close_vwap_pct": signal.d1_close_vwap_pct,
        "d1_close_to_vwap_raw": raw_vwap,
        "d1_high_to_close_drawdown_raw": (high - close) / high,
        "d1_close_location": close_location,
        "d1_low_to_close_recovery": close / low - 1.0,
        "d1_open_to_close_return_raw": close / open_ - 1.0,
    }


def build_august_features(root: str | Path, universe: pd.DataFrame) -> pd.DataFrame:
    root_path = Path(root).resolve()
    raw = pd.DataFrame([
        reconstruct_august_raw_row(root_path, event)
        for event in universe.to_dict("records")
    ]).sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)
    if len(raw) != len(universe) or raw["event_id"].nunique() != len(universe):
        raise RuntimeError("FATAL: August feature identity changed")
    x = prepare_frozen_x(raw)
    keep = ["event_id", *REDUCED_FEATURE_COLUMNS]
    result = universe.merge(raw[["event_id", "d1_close_to_vwap_raw"]], on="event_id", validate="one_to_one")
    result = result.merge(x[keep], on="event_id", validate="one_to_one")
    counts = {
        date: _board4plus_count(root_path, date)
        for date in sorted(result["signal_date"].unique())
    }
    result["board4plus_count"] = result["signal_date"].map(counts).astype(int)
    result["high_board4plus"] = result["board4plus_count"].gt(BOARD4PLUS_CUTOFF)
    result["regime_sign"] = np.where(result["high_board4plus"], -1.0, 1.0)
    result[REGIME_FEATURE] = (
        result["regime_sign"] * result["rank_d1_close_vwap_pct"]
    )
    if result[list(CHALLENGER_FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("FATAL: August frozen/challenger feature missing")
    return result.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def load_development(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root_path = Path(root).resolve()
    bridge, _ = load_bridge(root_path)
    train = bridge[bridge["label_available_date"].lt(TRAINING_ASOF)].copy()
    if (len(train), train["signal_date"].nunique()) != (
        TRAINING_ROWS_EXPECTED, TRAINING_DATES_EXPECTED,
    ):
        raise RuntimeError("FATAL: mature May-July training parity failed")
    raw = pd.read_csv(
        root_path / DEVELOPMENT_PATH_AUTHORITY,
        encoding="utf-8-sig",
        dtype={"event_id": str, "signal_date": str},
    )[["event_id", "d1_close_to_vwap_raw"]]
    ecology = pd.read_csv(
        root_path / DEVELOPMENT_REGIME_AUTHORITY,
        encoding="utf-8-sig",
        dtype={"signal_date": str},
    )[["signal_date", "four_plus_board_limit_up_count"]]
    train = train.merge(raw, on="event_id", validate="one_to_one")
    train = train.merge(ecology, on="signal_date", validate="many_to_one")
    train["board4plus_count"] = pd.to_numeric(
        train["four_plus_board_limit_up_count"], errors="raise"
    ).astype(int)
    train["high_board4plus"] = train["board4plus_count"].gt(BOARD4PLUS_CUTOFF)
    train["regime_sign"] = np.where(train["high_board4plus"], -1.0, 1.0)
    train[REGIME_FEATURE] = train["regime_sign"] * train["rank_d1_close_vwap_pct"]
    if train[list(CHALLENGER_FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("FATAL: development challenger feature missing")
    return train.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True), bridge


def build_feature_parity(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_date, day in train.groupby("signal_date", sort=True):
        targets = day[day["target7"].astype(bool)].sort_values("event_id", kind="mergesort")
        losses = day[day["loss"].astype(bool)].sort_values("event_id", kind="mergesort")
        for target in targets.itertuples(index=False):
            for loss in losses.itertuples(index=False):
                raw_diff = float(target.d1_close_to_vwap_raw - loss.d1_close_to_vwap_raw)
                rank_diff = float(target.rank_d1_close_vwap_pct - loss.rank_d1_close_vwap_pct)
                raw_order = int(np.sign(raw_diff))
                rank_order = int(np.sign(rank_diff))
                rows.append({
                    "signal_date": str(signal_date),
                    "target_event_id": str(target.event_id),
                    "loss_event_id": str(loss.event_id),
                    "target_raw_d1_close_to_vwap": float(target.d1_close_to_vwap_raw),
                    "loss_raw_d1_close_to_vwap": float(loss.d1_close_to_vwap_raw),
                    "raw_pair_difference": raw_diff,
                    "target_rank_d1_close_vwap_pct": float(target.rank_d1_close_vwap_pct),
                    "loss_rank_d1_close_vwap_pct": float(loss.rank_d1_close_vwap_pct),
                    "rank_pair_difference": rank_diff,
                    "raw_order": raw_order,
                    "rank_order": rank_order,
                    "pair_order_match": raw_order == rank_order,
                })
    parity = pd.DataFrame(rows).sort_values(
        ["signal_date", "target_event_id", "loss_event_id"], kind="mergesort"
    ).reset_index(drop=True)
    rate = float(parity["pair_order_match"].mean()) if len(parity) else math.nan
    audit = {
        "pair_count": len(parity),
        "pair_order_parity_rate": rate,
        "pair_order_parity_pass": bool(np.isfinite(rate) and rate >= PAIR_ORDER_PARITY_MIN),
        "minimum_required": PAIR_ORDER_PARITY_MIN,
    }
    return parity, audit


def derive_label_dates(root: str | Path, events: pd.DataFrame) -> pd.DataFrame:
    """Read only the date column to determine point-in-time maturity."""
    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        daily = _read_daily(root_path, str(event.code).zfill(6))[["date"]].copy()
        indices = daily.index[daily["date"].eq(str(event.signal_date))].tolist()
        if len(indices) != 1:
            raise RuntimeError(f"FATAL: signal daily date mismatch {event.event_id}")
        index = indices[0]
        d2 = str(daily.iloc[index + 1]["date"]) if index + 1 < len(daily) else None
        d3 = str(daily.iloc[index + 2]["date"]) if index + 2 < len(daily) else None
        mature = bool(d3 is not None and d3 < EVALUATION_ASOF)
        rows.append({
            "event_id": str(event.event_id),
            "d2_date": d2,
            "d3_date": d3,
            "label_available_date": d3,
            "mature_at_evaluation_asof": mature,
        })
    return pd.DataFrame(rows)


def _gradient_inf_norm(
    frame: pd.DataFrame,
    features: Sequence[str],
    beta: np.ndarray,
    weight: np.ndarray,
) -> float:
    x = frame[list(features)].to_numpy(float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    y = frame["target7"].to_numpy(float)
    p = _sigmoid(x_aug @ beta)
    reg = np.zeros_like(beta)
    reg[1:] = L2
    gradient = x_aug.T @ (weight * (p - y)) / weight.sum() + reg * beta
    return float(np.max(np.abs(gradient)))


def fit_two_models(train: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    baseline_beta, baseline_weight = fit_spec(train, S2_SPEC)
    adapter, challenger_weight = build_sample_weight(train, S2_SPEC)
    challenger_beta = fit_logistic_l2_weighted(
        adapter[list(CHALLENGER_FEATURE_COLUMNS)].to_numpy(float),
        adapter[DEFAULT_TARGET_COLUMN].astype(int).to_numpy(float),
        l2=L2,
        sample_weight=challenger_weight,
    )
    beta = {
        BASELINE_MODEL: np.asarray(baseline_beta, dtype=float),
        CHALLENGER_MODEL: np.asarray(challenger_beta, dtype=float),
    }
    rows: list[dict[str, Any]] = []
    gradients: dict[str, float] = {}
    for model, features, weights in (
        (BASELINE_MODEL, REDUCED_FEATURE_COLUMNS, baseline_weight),
        (CHALLENGER_MODEL, CHALLENGER_FEATURE_COLUMNS, challenger_weight),
    ):
        gradient = _gradient_inf_norm(train, features, beta[model], weights)
        gradients[model] = gradient
        rows.append({
            "model": model,
            "term": "INTERCEPT",
            "feature_order": 0,
            "coefficient": float(beta[model][0]),
            "feature_count": len(features),
            "training_rows": len(train),
            "training_dates": train["signal_date"].nunique(),
            "training_signal_start": train["signal_date"].min(),
            "training_signal_end": train["signal_date"].max(),
            "max_label_available_date": train["label_available_date"].max(),
            "training_asof": TRAINING_ASOF,
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "tail_bonus": "OFF",
            "optimizer_gradient_inf_norm": gradient,
            "optimizer_convergence": "PASS" if gradient <= 1e-6 else "CHECK",
        })
        for index, feature in enumerate(features, start=1):
            row = rows[-1].copy()
            row.update({
                "term": feature,
                "feature_order": index,
                "coefficient": float(beta[model][index]),
            })
            rows.append(row)
    audit = {
        "fit_count": 2,
        "baseline_gradient_inf_norm": gradients[BASELINE_MODEL],
        "challenger_gradient_inf_norm": gradients[CHALLENGER_MODEL],
        "both_optimizer_converged": bool(max(gradients.values()) <= 1e-6),
        "same_training_weight": bool(np.array_equal(baseline_weight, challenger_weight)),
        "training_weight_sum": float(baseline_weight.sum()),
    }
    return beta, pd.DataFrame(rows), audit


def score_and_lock(
    august: pd.DataFrame,
    label_dates: pd.DataFrame,
    beta: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    lock = august.merge(label_dates, on="event_id", validate="one_to_one")
    lock["baseline_score"] = _sigmoid(
        beta[BASELINE_MODEL][0]
        + lock[list(REDUCED_FEATURE_COLUMNS)].to_numpy(float) @ beta[BASELINE_MODEL][1:]
    )
    lock["challenger_score"] = _sigmoid(
        beta[CHALLENGER_MODEL][0]
        + lock[list(CHALLENGER_FEATURE_COLUMNS)].to_numpy(float) @ beta[CHALLENGER_MODEL][1:]
    )
    lock = _rank(lock, "baseline_score", "baseline_rank")
    lock = _rank(lock, "challenger_score", "challenger_rank")
    for prefix in ("baseline", "challenger"):
        for k in (1, 2, 3):
            lock[f"{prefix}_top{k}"] = lock[f"{prefix}_rank"].le(k)
    if OUTCOME_COLUMNS.intersection(lock.columns):
        raise RuntimeError("FATAL: outcome reached August prediction lock")
    return lock.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def _complete_date_eligibility(frame: pd.DataFrame) -> pd.Series:
    """Require every candidate on a date to have an available label.

    TopK and same-date universe metrics are not defined on a date where an
    immature candidate has been removed from the frozen ranking population.
    """
    row_mature = frame["mature_at_evaluation_asof"].astype(bool)
    return row_mature.groupby(frame["signal_date"]).transform("all").astype(bool)


def prediction_lock_sha256(lock: pd.DataFrame) -> str:
    return hashlib.sha256(_csv_bytes(lock)).hexdigest()


def build_prediction_phase(root: str | Path) -> dict[str, Any]:
    """Feature/fit/score phase.  No August outcome price is read."""
    assert_contract()
    train, _ = load_development(root)
    parity, parity_audit = build_feature_parity(train)
    if not parity_audit["pair_order_parity_pass"]:
        return {
            "parity": parity,
            "parity_audit": parity_audit,
            "model_test_state": "FEATURE_REPRESENTATION_MISMATCH",
        }
    universe = build_august_universe(root)
    august = build_august_features(root, universe)
    label_dates = derive_label_dates(root, august)
    beta, coefficients, fit_audit = fit_two_models(train)
    lock = score_and_lock(august, label_dates, beta)
    complete_date = _complete_date_eligibility(lock)
    audit = {
        **parity_audit,
        **fit_audit,
        "training_rows": len(train),
        "training_dates": train["signal_date"].nunique(),
        "training_signal_end": train["signal_date"].max(),
        "training_max_label_available_date": train["label_available_date"].max(),
        "august_total_rows": len(lock),
        "august_total_signal_dates": lock["signal_date"].nunique(),
        "august_first_signal_date": lock["signal_date"].min(),
        "august_last_signal_date": lock["signal_date"].max(),
        "august_row_level_label_available_rows": int(lock["mature_at_evaluation_asof"].sum()),
        "august_row_level_label_unavailable_rows": int((~lock["mature_at_evaluation_asof"]).sum()),
        "august_mature_rows": int(complete_date.sum()),
        "august_immature_rows": int((~complete_date).sum()),
        "august_mature_dates": int(lock.loc[complete_date, "signal_date"].nunique()),
        "august_immature_dates": int(lock.loc[~complete_date, "signal_date"].nunique()),
        "prediction_lock_sha256": prediction_lock_sha256(lock),
        "august_outcome_accessed": "NO_IN_PREDICTION_PHASE",
    }
    return {
        "train": train,
        "parity": parity,
        "parity_audit": parity_audit,
        "universe": universe,
        "august_features": august,
        "label_dates": label_dates,
        "beta": beta,
        "coefficients": coefficients,
        "lock": lock,
        "audit": audit,
        "model_test_state": "PREDICTION_LOCK_READY",
    }


def load_august_mature_outcomes_after_lock(
    root: str | Path, lock: pd.DataFrame
) -> pd.DataFrame:
    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    mature = lock[_complete_date_eligibility(lock)]
    for row in mature.itertuples(index=False):
        if not (AUGUST_START <= row.signal_date <= AUGUST_END):
            raise RuntimeError("FATAL: non-August event reached outcome attach")
        if not (str(row.label_available_date) < EVALUATION_ASOF):
            raise RuntimeError("FATAL: immature August label reached evaluation")
        daily = _read_daily(root_path, str(row.code).zfill(6))
        d2 = daily[daily["date"].eq(str(row.d2_date))]
        d3 = daily[daily["date"].eq(str(row.d3_date))]
        if len(d2) != 1 or len(d3) != 1:
            raise RuntimeError(f"FATAL: mature outcome price missing {row.event_id}")
        d2_open = float(d2.iloc[0]["open"])
        d3_high = float(d3.iloc[0]["high"])
        raw = d3_high / d2_open - 1.0
        rows.append({
            "event_id": str(row.event_id),
            "d2_open_daily": d2_open,
            "d3_high_daily": d3_high,
            "raw_repair_return": raw,
            "capped_return_7": min(raw, 0.07),
            "target7": int(raw >= 0.07),
            "positive_non_target": int(0 <= raw < 0.07),
            "loss": int(raw < 0),
            "severe_loss": int(raw <= -0.05),
        })
    outcomes = pd.DataFrame(rows)
    if outcomes["event_id"].duplicated().any() or len(outcomes) != len(mature):
        raise RuntimeError("FATAL: mature August outcome identity mismatch")
    return outcomes


def attach_mature_outcomes(lock: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    population = lock.merge(outcomes, on="event_id", how="left", validate="one_to_one")
    population["eligible_for_oot_evaluation"] = _complete_date_eligibility(population)
    mature = population["eligible_for_oot_evaluation"].astype(bool)
    required = list(OUTCOME_COLUMNS)
    if population.loc[mature, required].isna().any().any():
        raise RuntimeError("FATAL: mature August outcome incomplete")
    if population.loc[~mature, required].notna().any().any():
        raise RuntimeError("FATAL: immature August outcome was accessed/exported")
    population["outcome_class"] = np.select(
        [population["target7"].eq(1), population["loss"].eq(1)],
        ["TARGET7", "LOSS"],
        default="PNT",
    )
    population.loc[~mature, "outcome_class"] = "IMMATURE"
    return population.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def _pair_stats(frame: pd.DataFrame) -> dict[str, Any]:
    daily_values: list[float] = []
    pair_count = 0
    for _, day in frame.groupby("signal_date", sort=True):
        target = day[day["target7"].eq(1)]["d1_close_to_vwap_raw"].to_numpy(float)
        loss = day[day["loss"].eq(1)]["d1_close_to_vwap_raw"].to_numpy(float)
        if not len(target) or not len(loss):
            continue
        difference = target[:, None] - loss[None, :]
        daily_values.append(float(np.mean(np.where(difference > 0, 1.0, np.where(difference < 0, 0.0, .5)))))
        pair_count += int(difference.size)
    return {
        "informative_date_count": len(daily_values),
        "pair_count": pair_count,
        "within_date_pair_concordance": float(np.mean(daily_values)) if daily_values else math.nan,
    }


def build_august_hypothesis(mature: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values: dict[str, float] = {}
    for state, mask in (
        ("LOW_REGIME", mature["board4plus_count"].le(BOARD4PLUS_CUTOFF)),
        ("HIGH_REGIME", mature["board4plus_count"].gt(BOARD4PLUS_CUTOFF)),
    ):
        part = mature[mask]
        stats = _pair_stats(part)
        values[state] = stats["within_date_pair_concordance"]
        target = part[part["target7"].eq(1)]["d1_close_to_vwap_raw"]
        loss = part[part["loss"].eq(1)]["d1_close_to_vwap_raw"]
        rows.append({
            "regime_state": state,
            "fixed_cutoff": BOARD4PLUS_CUTOFF,
            "signal_date_count": part["signal_date"].nunique(),
            "target7_rows": len(target),
            "loss_rows": len(loss),
            **stats,
            "mean_signal_target7": float(target.mean()) if len(target) else math.nan,
            "mean_signal_loss": float(loss.mean()) if len(loss) else math.nan,
            "median_signal_target7": float(target.median()) if len(target) else math.nan,
            "median_signal_loss": float(loss.median()) if len(loss) else math.nan,
        })
    delta = values["HIGH_REGIME"] - values["LOW_REGIME"]
    for row in rows:
        row["delta_high_minus_low"] = delta
    direction = bool(
        np.isfinite(values["LOW_REGIME"])
        and np.isfinite(values["HIGH_REGIME"])
        and values["LOW_REGIME"] > .5
        and delta <= -.10
    )
    audit = {
        "august_low_concordance": values["LOW_REGIME"],
        "august_high_concordance": values["HIGH_REGIME"],
        "august_hypothesis_delta": delta,
        "august_hypothesis_direction_consistent": direction,
        "august_total_t7_loss_pairs": int(sum(row["pair_count"] for row in rows)),
    }
    return pd.DataFrame(rows), audit


def _daily_universe(mature: pd.DataFrame) -> pd.DataFrame:
    return mature.groupby("signal_date", as_index=False).agg(
        candidate_count=("event_id", "size"),
        universe_target7=("target7", "sum"),
        universe_loss=("loss", "sum"),
        universe_severe_loss=("severe_loss", "sum"),
        universe_capped=("capped_return_7", "mean"),
    )


def _oracle_return(mature: pd.DataFrame, k: int) -> float:
    values: list[float] = []
    for _, day in mature.groupby("signal_date", sort=True):
        selected = day.sort_values(
            ["capped_return_7", "event_id"], ascending=[False, True], kind="mergesort"
        ).head(k)
        values.append(float(selected["capped_return_7"].mean()))
    return float(np.mean(values)) if values else math.nan


def _model_long(mature: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, score, rank in (
        (BASELINE_MODEL, "baseline_score", "baseline_rank"),
        (CHALLENGER_MODEL, "challenger_score", "challenger_rank"),
    ):
        frame = mature.copy()
        frame["model"] = model
        frame["model_score"] = frame[score]
        frame["model_rank"] = frame[rank].astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _selection_daily(model_frame: pd.DataFrame, k: int) -> pd.DataFrame:
    selected = model_frame[model_frame["model_rank"].le(k)].copy()
    return selected.groupby("signal_date", as_index=False).agg(
        selected_rows=("event_id", "size"),
        target7_slots=("target7", "sum"),
        loss_slots=("loss", "sum"),
        severe_loss_slots=("severe_loss", "sum"),
        capped_return=("capped_return_7", "mean"),
        raw_return=("raw_repair_return", "mean"),
    )


def _selection_summary(
    model_frame: pd.DataFrame, universe_daily: pd.DataFrame, k: int
) -> dict[str, Any]:
    selected = model_frame[model_frame["model_rank"].le(k)]
    daily = _selection_daily(model_frame, k).merge(
        universe_daily, on="signal_date", validate="one_to_one"
    )
    winner_dates = daily["universe_target7"].gt(0)
    capture = (
        daily.loc[winner_dates, "target7_slots"]
        / daily.loc[winner_dates, "universe_target7"]
    )
    return {
        "dates": daily["signal_date"].nunique(),
        "selected_rows": len(selected),
        "target7_rate": float(selected["target7"].mean()),
        "loss_rate": float(selected["loss"].mean()),
        "severe_loss_rate": float(selected["severe_loss"].mean()),
        "mean_daily_capped_return": float(daily["capped_return"].mean()),
        "mean_daily_raw_return": float(daily["raw_return"].mean()),
        "excess_vs_same_date_universe": float((daily["capped_return"] - daily["universe_capped"]).mean()),
        "negative_date_rate": float(daily["capped_return"].lt(0).mean()),
        "zero_hit_date_rate": float(daily["target7_slots"].eq(0).mean()),
        "all_hit_date_rate": float((daily["selected_rows"].eq(k) & daily["target7_slots"].eq(k)).mean()),
        "worst_daily_capped_return": float(daily["capped_return"].min()),
        "winner_capture": float(capture.mean()) if len(capture) else math.nan,
    }


def build_summary_and_rankwise(
    mature: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    universe_daily = _daily_universe(mature)
    universe = {
        "model": "UNIVERSE",
        "selection": "ALL_CANDIDATES",
        "dates": mature["signal_date"].nunique(),
        "selected_rows": len(mature),
        "target7_rate": float(mature["target7"].mean()),
        "loss_rate": float(mature["loss"].mean()),
        "severe_loss_rate": float(mature["severe_loss"].mean()),
        "mean_daily_capped_return": float(universe_daily["universe_capped"].mean()),
        "mean_daily_raw_return": float(mature.groupby("signal_date")["raw_repair_return"].mean().mean()),
        "excess_vs_same_date_universe": 0.0,
        "negative_date_rate": float(universe_daily["universe_capped"].lt(0).mean()),
        "zero_hit_date_rate": float(universe_daily["universe_target7"].eq(0).mean()),
        "all_hit_date_rate": math.nan,
        "worst_daily_capped_return": float(universe_daily["universe_capped"].min()),
        "winner_capture": 1.0,
        "oracle_top1_capped": _oracle_return(mature, 1),
        "oracle_top2_capped": _oracle_return(mature, 2),
        "oracle_top3_capped": _oracle_return(mature, 3),
    }
    rows: list[dict[str, Any]] = [universe]
    rank_rows: list[dict[str, Any]] = []
    long = _model_long(mature)
    for model in MODEL_ORDER:
        model_frame = long[long["model"].eq(model)]
        for k in (1, 2, 3):
            rows.append({
                "model": model,
                "selection": f"TOP{k}",
                **_selection_summary(model_frame, universe_daily, k),
                "oracle_top1_capped": math.nan,
                "oracle_top2_capped": math.nan,
                "oracle_top3_capped": math.nan,
            })
            rank = model_frame[model_frame["model_rank"].eq(k)]
            rank_rows.append({
                "model": model,
                "rank": k,
                "rows": len(rank),
                "target7_count": int(rank["target7"].sum()),
                "target7_rate": float(rank["target7"].mean()) if len(rank) else math.nan,
                "loss_count": int(rank["loss"].sum()),
                "loss_rate": float(rank["loss"].mean()) if len(rank) else math.nan,
                "severe_loss_count": int(rank["severe_loss"].sum()),
                "severe_loss_rate": float(rank["severe_loss"].mean()) if len(rank) else math.nan,
                "mean_capped_return": float(rank["capped_return_7"].mean()) if len(rank) else math.nan,
                "mean_raw_return": float(rank["raw_repair_return"].mean()) if len(rank) else math.nan,
            })
    summary = pd.DataFrame(rows)
    rankwise = pd.DataFrame(rank_rows)
    audit = {
        "universe_target7_rate": universe["target7_rate"],
        "universe_loss_rate": universe["loss_rate"],
        "universe_severe_loss_rate": universe["severe_loss_rate"],
        "universe_capped_return": universe["mean_daily_capped_return"],
        "oracle_top1": universe["oracle_top1_capped"],
        "oracle_top2": universe["oracle_top2_capped"],
        "oracle_top3": universe["oracle_top3_capped"],
    }
    return summary, rankwise, audit


def build_daily_paired(mature: pd.DataFrame) -> pd.DataFrame:
    universe = _daily_universe(mature)
    long = _model_long(mature)
    baseline = _selection_daily(long[long["model"].eq(BASELINE_MODEL)], 3)
    challenger = _selection_daily(long[long["model"].eq(CHALLENGER_MODEL)], 3)
    baseline = baseline.add_prefix("baseline_").rename(columns={"baseline_signal_date": "signal_date"})
    challenger = challenger.add_prefix("challenger_").rename(columns={"challenger_signal_date": "signal_date"})
    daily = universe.merge(baseline, on="signal_date").merge(challenger, on="signal_date")
    daily["challenger_minus_baseline_capped"] = (
        daily["challenger_capped_return"] - daily["baseline_capped_return"]
    )
    for prefix, rank_column in (("baseline", "baseline_rank"), ("challenger", "challenger_rank")):
        for rank in (1, 2, 3):
            mapping = mature[mature[rank_column].eq(rank)].set_index("signal_date")
            daily[f"{prefix}_rank{rank}_event_id"] = daily["signal_date"].map(mapping["event_id"])
            daily[f"{prefix}_rank{rank}_code"] = daily["signal_date"].map(mapping["code"])
    daily["board4plus_count"] = daily["signal_date"].map(
        mature.groupby("signal_date")["board4plus_count"].first()
    )
    daily["regime_state"] = np.where(
        daily["board4plus_count"].le(BOARD4PLUS_CUTOFF), "LOW_REGIME", "HIGH_REGIME"
    )
    return daily.sort_values("signal_date", kind="mergesort").reset_index(drop=True)


def build_membership_changes(mature: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in mature.groupby("signal_date", sort=True):
        base = day[day["baseline_rank"].le(3)].sort_values("baseline_rank")
        challenger = day[day["challenger_rank"].le(3)].sort_values("challenger_rank")
        base_ids = set(base["event_id"])
        challenger_ids = set(challenger["event_id"])
        rows.append({
            "signal_date": signal_date,
            "record_type": "DAILY_MEMBERSHIP",
            "event_id": "",
            "code": "",
            "baseline_rank": math.nan,
            "challenger_rank": math.nan,
            "baseline_top3_event_ids": "|".join(base["event_id"]),
            "challenger_top3_event_ids": "|".join(challenger["event_id"]),
            "newly_entered_top3": "|".join(sorted(challenger_ids - base_ids)),
            "removed_from_top3": "|".join(sorted(base_ids - challenger_ids)),
            "target7": math.nan,
            "positive_non_target": math.nan,
            "loss": math.nan,
            "severe_loss": math.nan,
            "outcome_class": "",
        })
        for change, identities in (
            ("ENTERED_TOP3", challenger_ids - base_ids),
            ("REMOVED_FROM_TOP3", base_ids - challenger_ids),
        ):
            for event_id in sorted(identities):
                event = day[day["event_id"].eq(event_id)].iloc[0]
                rows.append({
                    "signal_date": signal_date,
                    "record_type": change,
                    "event_id": event_id,
                    "code": event["code"],
                    "baseline_rank": event["baseline_rank"],
                    "challenger_rank": event["challenger_rank"],
                    "baseline_top3_event_ids": "",
                    "challenger_top3_event_ids": "",
                    "newly_entered_top3": "",
                    "removed_from_top3": "",
                    "target7": event["target7"],
                    "positive_non_target": event["positive_non_target"],
                    "loss": event["loss"],
                    "severe_loss": event["severe_loss"],
                    "outcome_class": event["outcome_class"],
                })
    return pd.DataFrame(rows)


def build_regime_performance(mature: pd.DataFrame) -> pd.DataFrame:
    universe_daily = _daily_universe(mature)
    long = _model_long(mature)
    rows: list[dict[str, Any]] = []
    for state, date_mask in (
        ("LOW_REGIME", mature.groupby("signal_date")["board4plus_count"].first().le(BOARD4PLUS_CUTOFF)),
        ("HIGH_REGIME", mature.groupby("signal_date")["board4plus_count"].first().gt(BOARD4PLUS_CUTOFF)),
    ):
        dates = date_mask[date_mask].index
        state_universe = universe_daily[universe_daily["signal_date"].isin(dates)]
        for model in MODEL_ORDER:
            model_frame = long[long["model"].eq(model) & long["signal_date"].isin(dates)]
            for label, k, exact in (("RANK1", 1, True), ("RANK2", 2, True), ("RANK3", 3, True), ("TOP3", 3, False)):
                selected = (
                    model_frame[model_frame["model_rank"].eq(k)]
                    if exact else model_frame[model_frame["model_rank"].le(k)]
                )
                daily = selected.groupby("signal_date", as_index=False).agg(
                    capped_return=("capped_return_7", "mean")
                )
                rows.append({
                    "regime_state": state,
                    "model": model,
                    "selection": label,
                    "regime_signal_dates": len(dates),
                    "rows": len(selected),
                    "target7_rate": float(selected["target7"].mean()) if len(selected) else math.nan,
                    "loss_rate": float(selected["loss"].mean()) if len(selected) else math.nan,
                    "severe_loss_rate": float(selected["severe_loss"].mean()) if len(selected) else math.nan,
                    "mean_daily_capped_return": float(daily["capped_return"].mean()) if len(daily) else math.nan,
                    "same_date_universe_capped": float(state_universe["universe_capped"].mean()) if len(state_universe) else math.nan,
                })
    return pd.DataFrame(rows)


def build_bootstrap(daily: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(daily)
    schedule = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    metrics = {
        "TOP3_CAPPED_DELTA": daily["challenger_minus_baseline_capped"].to_numpy(float),
        "TARGET7_RATE_DELTA": (
            daily["challenger_target7_slots"] / daily["challenger_selected_rows"]
            - daily["baseline_target7_slots"] / daily["baseline_selected_rows"]
        ).to_numpy(float),
        "LOSS_RATE_DELTA": (
            daily["challenger_loss_slots"] / daily["challenger_selected_rows"]
            - daily["baseline_loss_slots"] / daily["baseline_selected_rows"]
        ).to_numpy(float),
    }
    rows: list[dict[str, Any]] = []
    for metric, values in metrics.items():
        boot = values[schedule].mean(axis=1)
        rows.append({
            "metric": metric,
            "sampling_unit": "SIGNAL_DATE",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "estimate": float(values.mean()),
            "bootstrap_p2_5": float(np.quantile(boot, .025)),
            "bootstrap_p50": float(np.quantile(boot, .50)),
            "bootstrap_p97_5": float(np.quantile(boot, .975)),
            "P_delta_gt_zero": float(np.mean(boot > 0)),
            "P_delta_lt_zero": float(np.mean(boot < 0)),
        })
    return pd.DataFrame(rows)


def build_lodo(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    full = float(daily["challenger_minus_baseline_capped"].mean())
    rows: list[dict[str, Any]] = []
    for date in daily["signal_date"]:
        value = float(daily.loc[~daily["signal_date"].eq(date), "challenger_minus_baseline_capped"].mean())
        rows.append({
            "omitted_signal_date": date,
            "full_mean_delta": full,
            "lodo_mean_delta": value,
            "delta_change": value - full,
            "absolute_delta_change": abs(value - full),
            "sign_flip_vs_full": bool(np.sign(value) != np.sign(full)),
        })
    output = pd.DataFrame(rows)
    output["influence_rank"] = output["absolute_delta_change"].rank(
        method="first", ascending=False
    ).astype(int)
    positive = daily.sort_values("challenger_minus_baseline_capped", ascending=False)
    remaining_after_top2 = positive.iloc[2:]["challenger_minus_baseline_capped"].mean() if len(positive) > 2 else math.nan
    concentrated = bool(full > 0 and np.isfinite(remaining_after_top2) and remaining_after_top2 <= 0)
    audit = {
        "lodo_delta_min": float(output["lodo_mean_delta"].min()),
        "lodo_delta_max": float(output["lodo_mean_delta"].max()),
        "lodo_delta_median": float(output["lodo_mean_delta"].median()),
        "lodo_sign_flips": int(output["sign_flip_vs_full"].sum()),
        "delta_after_removing_two_largest_positive_dates": float(remaining_after_top2),
        "date_concentrated": concentrated,
    }
    return output.sort_values("omitted_signal_date").reset_index(drop=True), audit


def _summary_row(summary: pd.DataFrame, model: str, selection: str) -> pd.Series:
    row = summary[summary["model"].eq(model) & summary["selection"].eq(selection)]
    if len(row) != 1:
        raise RuntimeError(f"FATAL: summary row missing {model}/{selection}")
    return row.iloc[0]


def decide_state(
    mature: pd.DataFrame,
    hypothesis: pd.DataFrame,
    hypothesis_audit: Mapping[str, Any],
    summary: pd.DataFrame,
    rankwise: pd.DataFrame,
    lodo_audit: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    low = hypothesis[hypothesis["regime_state"].eq("LOW_REGIME")].iloc[0]
    high = hypothesis[hypothesis["regime_state"].eq("HIGH_REGIME")].iloc[0]
    insufficient = bool(
        mature["signal_date"].nunique() < 10
        or int(low["pair_count"] + high["pair_count"]) < 20
        or int(low["pair_count"]) < 5
        or int(high["pair_count"]) < 5
    )
    baseline = _summary_row(summary, BASELINE_MODEL, "TOP3")
    challenger = _summary_row(summary, CHALLENGER_MODEL, "TOP3")
    universe = _summary_row(summary, "UNIVERSE", "ALL_CANDIDATES")
    rank = rankwise.pivot(index="rank", columns="model")
    cap_delta = float(challenger["mean_daily_capped_return"] - baseline["mean_daily_capped_return"])
    t7_delta = float(challenger["target7_rate"] - baseline["target7_rate"])
    loss_delta = float(challenger["loss_rate"] - baseline["loss_rate"])
    severe_delta = float(challenger["severe_loss_rate"] - baseline["severe_loss_rate"])
    capture_delta = float(challenger["winner_capture"] - baseline["winner_capture"])
    rank2_cap = float(rank.loc[2, ("mean_capped_return", CHALLENGER_MODEL)] - rank.loc[2, ("mean_capped_return", BASELINE_MODEL)])
    rank3_cap = float(rank.loc[3, ("mean_capped_return", CHALLENGER_MODEL)] - rank.loc[3, ("mean_capped_return", BASELINE_MODEL)])
    rank2_t7 = float(rank.loc[2, ("target7_rate", CHALLENGER_MODEL)] - rank.loc[2, ("target7_rate", BASELINE_MODEL)])
    rank3_t7 = float(rank.loc[3, ("target7_rate", CHALLENGER_MODEL)] - rank.loc[3, ("target7_rate", BASELINE_MODEL)])
    head_improves = bool(
        (rank2_cap > 0 and rank2_t7 >= 0 and rank3_cap >= -.01)
        or (rank3_cap > 0 and rank3_t7 >= 0 and rank2_cap >= -.01)
    )
    gates = {
        "sample_adequate": not insufficient,
        "raw_hypothesis_direction": bool(hypothesis_audit["august_hypothesis_direction_consistent"]),
        "challenger_top3_better": cap_delta > 0,
        "challenger_top3_beats_universe": float(challenger["mean_daily_capped_return"]) > float(universe["mean_daily_capped_return"]),
        "target7_increases": t7_delta > 0,
        "loss_not_worse": loss_delta <= 0,
        "severe_loss_not_worse": severe_delta <= 0,
        "rank2_or_rank3_improves_without_other_collapse": head_improves,
        "winner_capture_not_down": capture_delta >= 0,
        "not_date_concentrated": not bool(lodo_audit["date_concentrated"]),
    }
    if insufficient:
        state = "AUGUST_SAMPLE_INSUFFICIENT"
        next_action = "STOP_AND_REVIEW"
    elif all(gates.values()):
        state = "REGIME_AWARE_STAGE1_FORWARD_SUPPORTED"
        next_action = "FREEZE_CHALLENGER_FOR_SEPTEMBER_CONFIRMATION"
    elif (
        (cap_delta <= 0 and t7_delta <= 0)
        or (not gates["raw_hypothesis_direction"] and cap_delta <= 0)
        or loss_delta > .10
        or severe_delta > .10
        or (rank2_cap <= 0 and rank3_cap <= 0)
    ):
        state = "REGIME_AWARE_STAGE1_FORWARD_NOT_SUPPORTED"
        next_action = "STOP_THIS_REGIME_INTERACTION_LINE"
    else:
        state = "REGIME_AWARE_STAGE1_MIXED"
        next_action = "STOP_AND_REVIEW"
    audit = {
        **{f"gate_{key}": value for key, value in gates.items()},
        "top3_capped_delta": cap_delta,
        "top3_target7_rate_delta": t7_delta,
        "top3_loss_rate_delta": loss_delta,
        "top3_severe_loss_rate_delta": severe_delta,
        "winner_capture_delta": capture_delta,
        "rank2_capped_delta": rank2_cap,
        "rank3_capped_delta": rank3_cap,
        "rank2_target7_rate_delta": rank2_t7,
        "rank3_target7_rate_delta": rank3_t7,
    }
    return state, next_action, audit


def _fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def build_review(
    audit: Mapping[str, Any], summary: pd.DataFrame, rankwise: pd.DataFrame,
    hypothesis: pd.DataFrame, bootstrap: pd.DataFrame, state: str, next_action: str,
) -> str:
    universe = _summary_row(summary, "UNIVERSE", "ALL_CANDIDATES")
    base = _summary_row(summary, BASELINE_MODEL, "TOP3")
    challenger = _summary_row(summary, CHALLENGER_MODEL, "TOP3")
    low = hypothesis[hypothesis["regime_state"].eq("LOW_REGIME")].iloc[0]
    high = hypothesis[hypothesis["regime_state"].eq("HIGH_REGIME")].iloc[0]
    boot = bootstrap[bootstrap["metric"].eq("TOP3_CAPPED_DELTA")].iloc[0]
    rank = rankwise.pivot(index="rank", columns="model")
    lines = [
        "# v004c Regime-Aware Stage1 August OOT v001",
        "",
        "## Q1. August candidate pool本身难不难？",
        "",
        f"- Universe Target7 {_fmt(universe['target7_rate'])}, LOSS {_fmt(universe['loss_rate'])}, capped {_fmt(universe['mean_daily_capped_return'])}; complete-date mature {audit['august_mature_rows']} rows / {audit['august_mature_dates']} dates (row-level labels available: {audit['august_row_level_label_available_rows']}).",
        "",
        "## Q2. close/VWAP × board4plus 条件关系在August是否继续存在？",
        "",
        f"- LOW {_fmt(low['within_date_pair_concordance'])}; HIGH {_fmt(high['within_date_pair_concordance'])}; delta {_fmt(low['delta_high_minus_low'])}; direction consistent = {audit['august_hypothesis_direction_consistent']}.",
        "",
        "## Q3. Baseline S2的August Rank1/Rank2/Rank3/Top3表现怎样？",
        "",
        "- " + "; ".join(
            f"Rank{rank_no} capped {_fmt(rank.loc[rank_no, ('mean_capped_return', BASELINE_MODEL)])}, T7 {_fmt(rank.loc[rank_no, ('target7_rate', BASELINE_MODEL)])}, LOSS {_fmt(rank.loc[rank_no, ('loss_rate', BASELINE_MODEL)])}"
            for rank_no in (1, 2, 3)
        ) + f"; Top3 capped {_fmt(base['mean_daily_capped_return'])}.",
        "",
        "## Q4. Challenger相比Baseline改善了什么？",
        "",
        f"- Top3 capped delta {_fmt(audit['top3_capped_delta'])}; Rank2 delta {_fmt(audit['rank2_capped_delta'])}; Rank3 delta {_fmt(audit['rank3_capped_delta'])}; paired bootstrap 95% CI [{_fmt(boot['bootstrap_p2_5'])}, {_fmt(boot['bootstrap_p97_5'])}].",
        "",
        "## Q5. Target7增加了吗？",
        "",
        f"- {'YES' if audit['top3_target7_rate_delta'] > 0 else 'NO'}. Delta {_fmt(audit['top3_target7_rate_delta'])}.",
        "",
        "## Q6. LOSS减少了吗？",
        "",
        f"- {'YES' if audit['top3_loss_rate_delta'] < 0 else 'NO'}. LOSS delta {_fmt(audit['top3_loss_rate_delta'])}; severe LOSS delta {_fmt(audit['top3_severe_loss_rate_delta'])}.",
        "",
        "## Q7. Rank2/Rank3 winner-loss confusion改善了吗？",
        "",
        f"- {'YES' if audit['gate_rank2_or_rank3_improves_without_other_collapse'] else 'NO/MIXED'}.",
        "",
        "## Q8. 改善是不是少数日期造成？",
        "",
        (
            f"- NOT_APPLICABLE: challenger没有总体改善；paired dates为 "
            f"{audit['paired_positive_dates']} positive / {audit['paired_negative_dates']} negative / "
            f"{audit['paired_zero_dates']} zero."
            if audit["top3_capped_delta"] <= 0
            else f"- {'YES' if audit['date_concentrated'] else 'NO'}. LODO sign flips {audit['lodo_sign_flips']}."
        ),
        "",
        "## Q9. 单一regime interaction是否通过August forward test？",
        "",
        f"- {state}.",
        "",
        f"REGIME_AWARE_STAGE1_STATE = {state}",
        "",
        "AUGUST_HOLDOUT_STATUS = CONSUMED",
        "",
        f"NEXT_ACTION = {next_action}",
        "",
        f"EVALUATION_ASOF = {EVALUATION_ASOF}",
        "",
    ]
    return "\n".join(lines)


def evaluate_after_physical_lock(
    root: str | Path, phase: Mapping[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Consume mature August outcomes only after the runner writes the lock."""
    lock = phase["lock"]
    outcomes = load_august_mature_outcomes_after_lock(root, lock)
    population = attach_mature_outcomes(lock, outcomes)
    mature = population[population["eligible_for_oot_evaluation"].astype(bool)].copy()
    hypothesis, hypothesis_audit = build_august_hypothesis(mature)
    summary, rankwise, universe_audit = build_summary_and_rankwise(mature)
    daily = build_daily_paired(mature)
    membership = build_membership_changes(mature)
    regime_performance = build_regime_performance(mature)
    bootstrap = build_bootstrap(daily)
    lodo, lodo_audit = build_lodo(daily)
    state, next_action, decision_audit = decide_state(
        mature, hypothesis, hypothesis_audit, summary, rankwise, lodo_audit
    )
    audit = {
        **phase["audit"],
        **hypothesis_audit,
        **universe_audit,
        **lodo_audit,
        **decision_audit,
        "evaluation_asof": EVALUATION_ASOF,
        "regime_aware_stage1_state": state,
        "august_holdout_status": "CONSUMED",
        "next_action": next_action,
        "august_outcome_accessed": "YES_MATURE_ONLY_AFTER_PHYSICAL_LOCK",
        "august_immature_outcome_rows_accessed": 0,
        "model_fit_count": 2,
        "august_refit_count": 0,
        "threshold_search_count": 0,
        "alternative_relation_count": 0,
        "paired_positive_dates": int(daily["challenger_minus_baseline_capped"].gt(0).sum()),
        "paired_negative_dates": int(daily["challenger_minus_baseline_capped"].lt(0).sum()),
        "paired_zero_dates": int(daily["challenger_minus_baseline_capped"].eq(0).sum()),
    }
    review = build_review(audit, summary, rankwise, hypothesis, bootstrap, state, next_action)
    outputs = {
        OUTPUT_FILENAMES[1]: _csv_bytes(population),
        OUTPUT_FILENAMES[2]: _csv_bytes(hypothesis),
        OUTPUT_FILENAMES[3]: _csv_bytes(summary),
        OUTPUT_FILENAMES[4]: _csv_bytes(rankwise),
        OUTPUT_FILENAMES[5]: _csv_bytes(daily),
        OUTPUT_FILENAMES[6]: _csv_bytes(membership),
        OUTPUT_FILENAMES[7]: _csv_bytes(regime_performance),
        OUTPUT_FILENAMES[8]: _csv_bytes(bootstrap),
        OUTPUT_FILENAMES[9]: _csv_bytes(lodo),
        OUTPUT_FILENAMES[12]: review.encode("utf-8"),
    }
    return outputs, audit


def rebuild_evaluation_from_consumed_artifacts(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Repair aggregate outputs without fitting or reading an outcome source.

    This is intentionally limited to an already-consumed output directory. It
    enforces complete signal-date maturity using the physically locked ranks
    and the outcomes already present in the population artifact.
    """
    root_path = Path(root).resolve()
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    lock_path = output_dir / OUTPUT_FILENAMES[0]
    population_path = output_dir / OUTPUT_FILENAMES[1]
    coefficient_path = output_dir / OUTPUT_FILENAMES[10]
    parity_path = output_dir / OUTPUT_FILENAMES[11]
    for path in (lock_path, population_path, coefficient_path, parity_path):
        if not path.exists():
            raise RuntimeError(f"FATAL: consumed artifact missing: {path}")

    dtype = {"event_id": str, "signal_date": str, "code": str}
    lock = pd.read_csv(lock_path, encoding="utf-8-sig", dtype=dtype)
    population = pd.read_csv(population_path, encoding="utf-8-sig", dtype=dtype)
    coefficients = pd.read_csv(coefficient_path, encoding="utf-8-sig")
    parity = pd.read_csv(parity_path, encoding="utf-8-sig")
    if len(lock) != len(population) or set(lock["event_id"]) != set(population["event_id"]):
        raise RuntimeError("FATAL: consumed population does not match prediction lock")

    population["mature_at_evaluation_asof"] = population[
        "mature_at_evaluation_asof"
    ].astype(bool)
    population["eligible_for_oot_evaluation"] = _complete_date_eligibility(population)
    eligible = population["eligible_for_oot_evaluation"].astype(bool)
    required = list(OUTCOME_COLUMNS)
    if population.loc[eligible, required].isna().any().any():
        raise RuntimeError("FATAL: complete-date consumed outcomes are incomplete")
    population.loc[~eligible, required] = np.nan
    population["outcome_class"] = np.select(
        [population["target7"].eq(1), population["loss"].eq(1)],
        ["TARGET7", "LOSS"],
        default="PNT",
    )
    population.loc[~eligible, "outcome_class"] = "IMMATURE"
    mature = population[eligible].copy()

    hypothesis, hypothesis_audit = build_august_hypothesis(mature)
    summary, rankwise, universe_audit = build_summary_and_rankwise(mature)
    daily = build_daily_paired(mature)
    membership = build_membership_changes(mature)
    regime_performance = build_regime_performance(mature)
    bootstrap = build_bootstrap(daily)
    lodo, lodo_audit = build_lodo(daily)

    pair_match = parity["pair_order_match"].astype(str).str.lower().eq("true")
    baseline_meta = coefficients[coefficients["model"].eq(BASELINE_MODEL)].iloc[0]
    challenger_meta = coefficients[coefficients["model"].eq(CHALLENGER_MODEL)].iloc[0]
    audit: dict[str, Any] = {
        "pair_count": len(parity),
        "pair_order_parity_rate": float(pair_match.mean()),
        "pair_order_parity_pass": bool(pair_match.mean() >= PAIR_ORDER_PARITY_MIN),
        "minimum_required": PAIR_ORDER_PARITY_MIN,
        "fit_count": 2,
        "model_fit_count": 2,
        "same_training_weight": True,
        "baseline_gradient_inf_norm": float(baseline_meta["optimizer_gradient_inf_norm"]),
        "challenger_gradient_inf_norm": float(challenger_meta["optimizer_gradient_inf_norm"]),
        "both_optimizer_converged": bool(
            str(baseline_meta["optimizer_convergence"]) == "PASS"
            and str(challenger_meta["optimizer_convergence"]) == "PASS"
        ),
        "training_rows": int(baseline_meta["training_rows"]),
        "training_dates": int(baseline_meta["training_dates"]),
        "training_signal_end": str(baseline_meta["training_signal_end"]),
        "training_max_label_available_date": str(baseline_meta["max_label_available_date"]),
        "august_total_rows": len(population),
        "august_total_signal_dates": population["signal_date"].nunique(),
        "august_first_signal_date": population["signal_date"].min(),
        "august_last_signal_date": population["signal_date"].max(),
        "august_row_level_label_available_rows": int(population["mature_at_evaluation_asof"].sum()),
        "august_row_level_label_unavailable_rows": int((~population["mature_at_evaluation_asof"]).sum()),
        "august_mature_rows": int(eligible.sum()),
        "august_immature_rows": int((~eligible).sum()),
        "august_mature_dates": int(population.loc[eligible, "signal_date"].nunique()),
        "august_immature_dates": int(population.loc[~eligible, "signal_date"].nunique()),
        "prediction_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "evaluation_asof": EVALUATION_ASOF,
        "august_outcome_accessed": "NO_ADDITIONAL_ACCESS_REUSED_CONSUMED_ARTIFACT",
        "august_immature_outcome_rows_accessed": 0,
        "august_refit_count": 0,
        "threshold_search_count": 0,
        "alternative_relation_count": 0,
        "complete_date_evaluation_rebuild": True,
    }
    audit.update(hypothesis_audit)
    audit.update(universe_audit)
    audit.update(lodo_audit)
    audit.update({
        "paired_positive_dates": int(daily["challenger_minus_baseline_capped"].gt(0).sum()),
        "paired_negative_dates": int(daily["challenger_minus_baseline_capped"].lt(0).sum()),
        "paired_zero_dates": int(daily["challenger_minus_baseline_capped"].eq(0).sum()),
    })
    state, next_action, decision_audit = decide_state(
        mature, hypothesis, hypothesis_audit, summary, rankwise, lodo_audit
    )
    audit.update(decision_audit)
    audit.update({
        "regime_aware_stage1_state": state,
        "august_holdout_status": "CONSUMED",
        "next_action": next_action,
    })
    review = build_review(audit, summary, rankwise, hypothesis, bootstrap, state, next_action)
    outputs = {
        OUTPUT_FILENAMES[1]: _csv_bytes(population),
        OUTPUT_FILENAMES[2]: _csv_bytes(hypothesis),
        OUTPUT_FILENAMES[3]: _csv_bytes(summary),
        OUTPUT_FILENAMES[4]: _csv_bytes(rankwise),
        OUTPUT_FILENAMES[5]: _csv_bytes(daily),
        OUTPUT_FILENAMES[6]: _csv_bytes(membership),
        OUTPUT_FILENAMES[7]: _csv_bytes(regime_performance),
        OUTPUT_FILENAMES[8]: _csv_bytes(bootstrap),
        OUTPUT_FILENAMES[9]: _csv_bytes(lodo),
        OUTPUT_FILENAMES[12]: review.encode("utf-8"),
    }
    for filename, payload in outputs.items():
        (output_dir / filename).write_bytes(payload)
    return output_dir, audit


def run_one_shot(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = build_prediction_phase(root_path)
    parity_path = output_dir / OUTPUT_FILENAMES[11]
    parity_path.write_bytes(_csv_bytes(phase["parity"]))
    if phase["model_test_state"] == "FEATURE_REPRESENTATION_MISMATCH":
        review = (
            "# v004c Regime-Aware Stage1 August OOT v001\n\n"
            "MODEL_TEST_STATE = FEATURE_REPRESENTATION_MISMATCH\n\n"
            "AUGUST_HOLDOUT_STATUS = NOT_CONSUMED\n\n"
            "NEXT_ACTION = STOP_AND_REVIEW\n"
        )
        (output_dir / OUTPUT_FILENAMES[12]).write_text(review, encoding="utf-8")
        return output_dir, {
            **phase["parity_audit"],
            "model_test_state": "FEATURE_REPRESENTATION_MISMATCH",
            "august_holdout_status": "NOT_CONSUMED",
        }

    # Physical prediction lock and coefficient artifact are persisted before
    # the first August outcome price is read.
    lock_path = output_dir / OUTPUT_FILENAMES[0]
    lock_path.write_bytes(_csv_bytes(phase["lock"]))
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != phase["audit"]["prediction_lock_sha256"]:
        raise RuntimeError("FATAL: physical August prediction lock hash mismatch")
    (output_dir / OUTPUT_FILENAMES[10]).write_bytes(_csv_bytes(phase["coefficients"]))
    outputs, audit = evaluate_after_physical_lock(root_path, phase)
    for filename, payload in outputs.items():
        (output_dir / filename).write_bytes(payload)
    return output_dir, audit

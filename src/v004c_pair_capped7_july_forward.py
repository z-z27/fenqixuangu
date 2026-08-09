from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import get_data_config, get_strategy_config
from .indicators import enrich_5min_indicators, enrich_daily_indicators
from .loaders import (
    _filter_to_end_date,
    _is_main_board_limit_up_day,
    _keep_recent_trade_days,
    _merge_minute_amount,
)
from .signal_engine import generate_signal
from .v004a import (
    DEFAULT_CLOSE_RETURN_COLUMN,
    DEFAULT_HIGH_RETURN_COLUMN,
    DEFAULT_TARGET_COLUMN,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
    prepare_v004a_samples,
)
from .v004c_top10_target_information import (
    PAIR_FEATURE_COLUMNS,
    PAIR_L2,
    build_same_date_pairs,
    fit_pairwise_weighted_ridge,
)
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2 as STAGE1_L2,
    POSITIVE_WEIGHT as STAGE1_POSITIVE_WEIGHT,
    RAW_INPUT_COLUMNS,
    _sigmoid,
    assert_authoritative_universe,
    assert_frozen_architecture_contract,
    build_exact_raw_features,
    dataframe_csv_bytes,
)
from .v004c_v4a_top10_reranker import (
    RESIDUAL_RAW_FIELDS,
    STAGE2_FEATURE_COLUMNS,
    TOP_K,
    build_stage2_features,
    strategy_summary,
)


FROZEN_DEVELOPMENT_COMMIT = "c6af1289f66f8c2cad610b9564573607ce57ac14"
FORWARD_STRESS_CLASS = "SEMI_OOT_CHRONOLOGICAL"
TRAINING_ASOF_DATE = "2026-07-01"
JULY_START = "2026-07-01"
JULY_END = "2026-07-31"
AUGUST_SIGNAL_SENTINEL = "2026-08-01"
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_RESAMPLES = 20_000
STAGE2_TARGET = "CAPPED7"
STAGE2_INTERCEPT = 0.0
NEW_FEATURES = False
TUNING = False
JULY_REFIT = False

OUTPUT_FILENAMES = (
    "v004c_july_forward_prediction_lock_v001.csv",
    "v004c_july_forward_training_audit_v001.csv",
    "v004c_july_forward_oof_v001.csv",
    "v004c_july_forward_daily_v001.csv",
    "v004c_july_forward_practical_v001.csv",
    "v004c_july_forward_membership_v001.csv",
    "v004c_july_forward_review_v001.md",
)

PREDICTION_COLUMNS = [
    "event_id", "signal_date", "code", "board", "candidate_count",
    "stage1_score", "stage1_rank", "stage1_top10", "stage1_strength",
    *RESIDUAL_RAW_FIELDS,
    "closing_completion_gap", "strength_x_gap",
    "capped_pair_score", "capped_pair_rank",
]
OUTCOME_COLUMNS = {
    "d2_date", "d3_date", "d2_open_daily", "d3_high_daily",
    "target7", "raw_return", "capped_return_7",
}


def assert_frozen_forward_contract() -> None:
    assert_frozen_architecture_contract()
    if len(FROZEN_FEATURE_COLUMNS) != 18:
        raise RuntimeError("FATAL: frozen Stage1 feature count changed")
    if STAGE1_L2 != 0.30 or STAGE1_POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen Stage1 hyperparameters changed")
    if PAIR_FEATURE_COLUMNS != STAGE2_FEATURE_COLUMNS or PAIR_FEATURE_COLUMNS != [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]:
        raise RuntimeError("FATAL: frozen Stage2 predictors changed")
    if PAIR_L2 != 0.30 or STAGE2_TARGET != "CAPPED7" or STAGE2_INTERCEPT != 0.0:
        raise RuntimeError("FATAL: frozen PAIR_CAPPED7 specification changed")
    if TOP_K != 10 or NEW_FEATURES or TUNING or JULY_REFIT:
        raise RuntimeError("FATAL: forward-only contract changed")


def capped7_utility(raw_return: float | np.ndarray | pd.Series) -> Any:
    return np.minimum(raw_return, 0.07)


def label_is_available(label_available_date: str | None, asof_date: str) -> bool:
    return bool(label_available_date is not None and str(label_available_date) < asof_date)


def load_authoritative_d1_safe_base(root: Path) -> pd.DataFrame:
    """Read identity and D1 fields only; no D2/D3/target column is loaded."""
    path = (
        root / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv"
    )
    columns = [
        "event_id", "code", "name", "signal_date", "break_date",
        "board_streak_before_break", *RESIDUAL_RAW_FIELDS,
    ]
    frame = pd.read_csv(
        path, usecols=columns, encoding="utf-8-sig", dtype={"code": str}
    )
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in ("event_id", "signal_date", "break_date"):
        frame[column] = frame[column].astype(str)
    assert_authoritative_universe(frame)
    return frame.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _normalise_daily(raw: pd.DataFrame) -> pd.DataFrame:
    daily = raw.copy()
    if "date" not in daily.columns:
        raise RuntimeError("FATAL: daily cache has no date column")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily = daily.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column not in daily.columns:
            daily[column] = np.nan
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    return daily


def _daily_path(root: Path, code: str) -> Path:
    for folder in ("daily", "daily_unadjusted"):
        path = root / "data/cache" / folder / f"{code}_daily.pkl"
        if path.exists():
            return path
    raise RuntimeError(f"FATAL: canonical daily cache missing for {code}")


def _read_daily(root: Path, code: str) -> pd.DataFrame:
    return _normalise_daily(pd.read_pickle(_daily_path(root, code)))


def _read_daily_through(root: Path, code: str, end_date: str) -> pd.DataFrame:
    daily = _read_daily(root, code)
    clipped = daily[daily["date"].le(end_date)].copy()
    if clipped.empty or str(clipped.iloc[-1]["date"]) != end_date:
        raise RuntimeError(f"FATAL: daily D1 unavailable for {code}@{end_date}")
    if bool(clipped["date"].gt(end_date).any()):
        raise RuntimeError("FATAL: post-D1 daily row survived prediction clip")
    return clipped.reset_index(drop=True)


def _load_name_map(root: Path) -> dict[str, str]:
    path = root / "data/cache/universe/eastmoney_main_board_universe.pkl"
    frame = pd.read_pickle(path)
    return dict(zip(
        frame["code"].astype(str).str.zfill(6), frame["name"].fillna("").astype(str)
    ))


def build_july_universe(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild identity only; no D2/D3/outcome field is read or consulted."""
    names = _load_name_map(root)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "data/cache/daily").glob("*_daily.pkl")):
        code = path.name[:6]
        daily = _normalise_daily(pd.read_pickle(path))
        if len(daily) < 3:
            continue
        flags = np.zeros(len(daily), dtype=bool)
        for index in range(1, len(daily)):
            flags[index] = _is_main_board_limit_up_day(daily, index)
        for index in range(1, len(daily)):
            signal_date = str(daily.iloc[index]["date"])
            if not (JULY_START <= signal_date <= JULY_END):
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
        raise RuntimeError("FATAL: July identity unavailable or duplicated")
    if bool(universe["signal_date"].ge(AUGUST_SIGNAL_SENTINEL).any()):
        raise RuntimeError("FATAL: August signal candidate entered July universe")
    if not universe["board_streak_before_break"].isin([2, 3]).all():
        raise RuntimeError("FATAL: non-v004c board streak entered July universe")

    # Identity-only parity against the previous D1 snapshot.  That snapshot was
    # a training-quality table and legitimately omits several true identities;
    # it may not be used to delete them in this forward universe.
    snapshot_path = (
        root / "reports/research/v004c_d1_dataset_v001_20260601_20260729"
        / "v004c_d1_snapshot_v001.csv"
    )
    snapshot = pd.read_csv(
        snapshot_path,
        usecols=["event_id", "signal_date", "board_streak_before_break"],
        encoding="utf-8-sig",
    )
    snapshot = snapshot[snapshot["signal_date"].astype(str).str.startswith("2026-07")]
    through_29 = universe[universe["signal_date"].le("2026-07-29")]
    snapshot_ids = set(snapshot["event_id"].astype(str))
    rebuilt_ids = set(through_29["event_id"].astype(str))
    if snapshot_ids.difference(rebuilt_ids):
        raise RuntimeError("FATAL: rebuilt July identity lost a canonical snapshot event")
    audit = {
        "snapshot_identity_rows": int(len(snapshot)),
        "rebuilt_through_july29_rows": int(len(through_29)),
        "snapshot_missing_from_rebuild": int(len(snapshot_ids - rebuilt_ids)),
        "identity_only_additions_through_july29": int(len(rebuilt_ids - snapshot_ids)),
        "july30_31_rows": int(universe["signal_date"].ge("2026-07-30").sum()),
    }
    return universe, audit


def _read_minute_through(root: Path, code: str, signal_date: str) -> pd.DataFrame:
    path = root / "data/cache/minute_5m" / f"{code}_5min.pkl"
    if not path.exists():
        raise RuntimeError(f"FATAL: canonical 5m cache missing for {code}")
    minute = pd.read_pickle(path).copy()
    if "trade_date" not in minute.columns:
        minute["trade_date"] = pd.to_datetime(
            minute["datetime"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    else:
        minute["trade_date"] = pd.to_datetime(
            minute["trade_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    minute = minute.dropna(subset=["trade_date"])
    minute = minute[minute["trade_date"].le(signal_date)].copy()
    day = minute[minute["trade_date"].eq(signal_date)]
    if len(day) != 48:
        raise RuntimeError(
            f"FATAL: expected 48 canonical D1 bars for {code}@{signal_date}, got {len(day)}"
        )
    if bool(minute["trade_date"].gt(signal_date).any()):
        raise RuntimeError("FATAL: post-D1 minute row survived prediction clip")
    return minute.sort_values("datetime", kind="mergesort").reset_index(drop=True)


def _load_five_day_pool(root: Path, signal_date: str) -> pd.DataFrame:
    anchor = pd.Timestamp(signal_date)
    frames: list[pd.DataFrame] = []
    for offset in range(5):
        date = (anchor - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        if date > signal_date or date >= AUGUST_SIGNAL_SENTINEL:
            raise RuntimeError("FATAL: non-D1-safe pool date requested")
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
        raise RuntimeError(f"FATAL: five-day limit-up pool unavailable for {signal_date}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "code"], kind="mergesort"
    ).drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True)


def reconstruct_forward_raw_row(root: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    code = str(event["code"]).zfill(6)
    signal_date = str(event["signal_date"])
    if not (JULY_START <= signal_date <= JULY_END):
        raise RuntimeError("FATAL: forward raw reconstruction outside July")
    daily_raw = _read_daily_through(root, code, signal_date)
    minute_raw = _read_minute_through(root, code, signal_date)
    pool = _load_five_day_pool(root, signal_date)
    if len(daily_raw) < 2:
        raise RuntimeError(f"FATAL: D0 unavailable for {code}@{signal_date}")
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
    signal = generate_signal(
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
        "d1_high_to_close_drawdown_raw": (high - close) / high,
        "d1_close_location": close_location,
        "d1_low_to_close_recovery": close / low - 1.0,
        "d1_open_to_close_return_raw": close / open_ - 1.0,
    }


def build_july_d1_safe_features(root: Path, universe: pd.DataFrame) -> pd.DataFrame:
    rows = [
        reconstruct_forward_raw_row(root, event)
        for event in universe.to_dict("records")
    ]
    frame = pd.DataFrame(rows).sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(frame) != len(universe) or frame["event_id"].nunique() != len(universe):
        raise RuntimeError("FATAL: July feature identity changed")
    if frame["candidate_base_price"].isna().any() or frame["candidate_base_price"].le(0).any():
        raise RuntimeError("FATAL: July candidate base price invalid")
    return frame


def prepare_frozen_x(raw: pd.DataFrame) -> pd.DataFrame:
    """Build X with outcome-neutral placeholders; only D1 raw values can affect X."""
    adapter = raw.copy()
    adapter["eligible_for_trade"] = True
    adapter[DEFAULT_TARGET_COLUMN] = False
    adapter[DEFAULT_HIGH_RETURN_COLUMN] = 0.0
    adapter[DEFAULT_CLOSE_RETURN_COLUMN] = 0.0
    prepared, info, _ = prepare_v004a_samples(adapter, target_return_pct=7.0)
    if info["v004a_feature_columns"] != FROZEN_FEATURE_COLUMNS:
        raise RuntimeError("FATAL: frozen 18-feature transform changed")
    if len(prepared) != len(raw) or prepared["event_id"].nunique() != len(raw):
        raise RuntimeError("FATAL: frozen X transform changed universe")
    return prepared.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def derive_trade_dates(root: Path, events: pd.DataFrame) -> pd.DataFrame:
    """Date-only maturity table. Outcome prices are not returned or inspected."""
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        daily = _read_daily(root, str(event.code).zfill(6))
        indices = daily.index[daily["date"].eq(str(event.signal_date))].tolist()
        if len(indices) != 1:
            raise RuntimeError(f"FATAL: signal daily row mismatch {event.event_id}")
        index = indices[0]
        d2 = str(daily.iloc[index + 1]["date"]) if index + 1 < len(daily) else None
        d3 = str(daily.iloc[index + 2]["date"]) if index + 2 < len(daily) else None
        rows.append({
            "event_id": str(event.event_id),
            "signal_date": str(event.signal_date),
            "d2_date": d2,
            "d3_date": d3,
            "label_available_date": d3,
        })
    return pd.DataFrame(rows)


def load_mature_training_outcomes(
    root: Path, events: pd.DataFrame, dates: pd.DataFrame
) -> pd.DataFrame:
    joined = events[["event_id", "code", "signal_date"]].merge(
        dates, on=["event_id", "signal_date"], validate="one_to_one"
    )
    eligible_mask = joined["label_available_date"].map(
        lambda value: label_is_available(
            None if pd.isna(value) else str(value), TRAINING_ASOF_DATE
        )
    )
    eligible = joined[eligible_mask].copy()
    rows: list[dict[str, Any]] = []
    for row in eligible.itertuples(index=False):
        daily = _read_daily(root, str(row.code).zfill(6))
        d2 = daily[daily["date"].eq(row.d2_date)]
        d3 = daily[daily["date"].eq(row.d3_date)]
        if len(d2) != 1 or len(d3) != 1:
            raise RuntimeError(f"FATAL: mature outcome daily row mismatch {row.event_id}")
        d2_open = float(d2.iloc[0]["open"])
        d3_high = float(d3.iloc[0]["high"])
        raw_return = d3_high / d2_open - 1.0
        rows.append({
            "event_id": row.event_id,
            "d2_date": row.d2_date,
            "d3_date": row.d3_date,
            "label_available_date": row.label_available_date,
            "d2_open_daily": d2_open,
            "d3_high_daily": d3_high,
            "raw_repair_return": raw_return,
            "capped_opportunity_return_7": float(capped7_utility(raw_return)),
            "target7": int(raw_return >= 0.07),
        })
    outcomes = pd.DataFrame(rows)
    if outcomes.empty or bool(outcomes["label_available_date"].ge(TRAINING_ASOF_DATE).any()):
        raise RuntimeError("FATAL: immature label entered pre-July training")
    return outcomes


def attach_training_outcomes(x_all: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    samples = x_all.merge(outcomes, on="event_id", how="inner", validate="one_to_one")
    samples[DEFAULT_TARGET_COLUMN] = samples["target7"].astype(bool)
    samples[DEFAULT_HIGH_RETURN_COLUMN] = samples["raw_repair_return"] * 100.0
    samples[DEFAULT_CLOSE_RETURN_COLUMN] = 0.0
    samples["tail_weight"] = np.select(
        [
            samples[DEFAULT_HIGH_RETURN_COLUMN].ge(12.0),
            samples[DEFAULT_HIGH_RETURN_COLUMN].ge(10.0),
        ],
        [2.0, 1.5],
        default=1.0,
    )
    return samples.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _fit_stage1_snapshot(samples: pd.DataFrame) -> np.ndarray:
    weights = build_training_sample_weight(
        samples, positive_weight=STAGE1_POSITIVE_WEIGHT
    )
    return fit_logistic_l2_weighted(
        samples[FROZEN_FEATURE_COLUMNS].to_numpy(float),
        samples[DEFAULT_TARGET_COLUMN].astype(float).to_numpy(),
        l2=STAGE1_L2,
        sample_weight=weights,
    )


def _score_stage1(
    score_rows: pd.DataFrame, beta: np.ndarray
) -> pd.DataFrame:
    scored = score_rows.copy()
    scored["stage1_score"] = _sigmoid(
        beta[0] + scored[FROZEN_FEATURE_COLUMNS].to_numpy(float) @ beta[1:]
    )
    scored = scored.sort_values(
        ["signal_date", "stage1_score", "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scored["stage1_rank"] = scored.groupby("signal_date", sort=True).cumcount() + 1
    scored["candidate_count"] = scored.groupby("signal_date", sort=True)[
        "event_id"
    ].transform("size")
    scored["stage1_top10"] = scored["stage1_rank"].le(
        np.minimum(TOP_K, scored["candidate_count"])
    )
    scored["stage1_strength"] = 1.0 - (
        (scored["stage1_rank"] - 1)
        / np.maximum(scored["candidate_count"] - 1, 1)
    )
    return scored


def build_pre_july_snapshot(
    base_x: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    eligible_dates = sorted(samples["signal_date"].astype(str).unique())
    meta_parts: list[pd.DataFrame] = []
    crossfit_fits = 0
    self_label_leakage_rows = 0
    for meta_date in eligible_dates:
        base_train = samples[~samples["signal_date"].eq(meta_date)].copy()
        if base_train.empty or bool(base_train["signal_date"].eq(meta_date).any()):
            raise RuntimeError("FATAL: self-label Stage1 meta fit")
        beta = _fit_stage1_snapshot(base_train)
        score_rows = base_x[base_x["signal_date"].eq(meta_date)].copy()
        scored = _score_stage1(score_rows, beta)
        top10 = build_stage2_features(scored[scored["stage1_top10"]].copy())
        labels = samples[[
            "event_id", "target7", "raw_repair_return",
            "capped_opportunity_return_7", "label_available_date",
        ]]
        top10 = top10.merge(labels, on="event_id", how="inner", validate="one_to_one")
        if bool(top10["label_available_date"].ge(TRAINING_ASOF_DATE).any()):
            raise RuntimeError("FATAL: immature Stage2 meta label")
        meta_parts.append(top10)
        crossfit_fits += 1
    meta = pd.concat(meta_parts, ignore_index=True)
    pairs = build_same_date_pairs(meta)
    x_columns = [f"x_{feature}" for feature in PAIR_FEATURE_COLUMNS]
    stage2_beta = fit_pairwise_weighted_ridge(
        pairs[x_columns].to_numpy(float),
        pairs["pair_y_capped7"].to_numpy(float),
        pairs["pair_weight"].to_numpy(float),
        l2=PAIR_L2,
    )
    stage1_beta = _fit_stage1_snapshot(samples)
    audit = {
        "eligible_historical_rows": int(len(samples)),
        "eligible_historical_dates": int(samples["signal_date"].nunique()),
        "latest_eligible_signal_date": str(samples["signal_date"].max()),
        "latest_label_available_date": str(samples["label_available_date"].max()),
        "stage1_training_rows": int(len(samples)),
        "stage2_meta_dates": int(meta["signal_date"].nunique()),
        "stage2_meta_rows": int(len(meta)),
        "stage2_pair_count": int(len(pairs)),
        "crossfit_stage1_models_fitted": crossfit_fits,
        "self_label_leakage_rows": self_label_leakage_rows,
        "july_training_rows": int(samples["signal_date"].ge(JULY_START).sum()),
        "august_training_rows": int(samples["signal_date"].ge(AUGUST_SIGNAL_SENTINEL).sum()),
    }
    if audit["july_training_rows"] or audit["august_training_rows"]:
        raise RuntimeError("FATAL: post-June row entered frozen training snapshot")
    return stage1_beta, stage2_beta, meta, audit


def build_prediction_lock(
    july_x: pd.DataFrame,
    stage1_beta: np.ndarray,
    stage2_beta: np.ndarray,
) -> pd.DataFrame:
    scored = _score_stage1(july_x, stage1_beta)
    scored["closing_completion_gap"] = np.nan
    scored["strength_x_gap"] = np.nan
    scored["capped_pair_score"] = np.nan
    scored["capped_pair_rank"] = np.nan
    parts: list[pd.DataFrame] = []
    for _, day in scored.groupby("signal_date", sort=True):
        top10 = build_stage2_features(day[day["stage1_top10"]].copy())
        top10["capped_pair_score"] = (
            top10[PAIR_FEATURE_COLUMNS].to_numpy(float) @ stage2_beta
        )
        top10 = top10.sort_values(
            ["capped_pair_score", "stage1_rank", "event_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        top10["capped_pair_rank"] = np.arange(1, len(top10) + 1)
        parts.append(top10[[
            "event_id", "closing_completion_gap", "strength_x_gap",
            "capped_pair_score", "capped_pair_rank",
        ]])
    values = pd.concat(parts, ignore_index=True)
    scored = scored.drop(columns=[
        "closing_completion_gap", "strength_x_gap",
        "capped_pair_score", "capped_pair_rank",
    ]).merge(values, on="event_id", how="left", validate="one_to_one")
    scored["board"] = scored["board_streak_before_break"].astype(int)
    output = scored[PREDICTION_COLUMNS].sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if OUTCOME_COLUMNS.intersection(output.columns):
        raise RuntimeError("FATAL: outcome column entered prediction lock")
    if output.loc[output["stage1_top10"], [
        "capped_pair_score", "capped_pair_rank"
    ]].isna().any().any():
        raise RuntimeError("FATAL: July Top10 prediction incomplete")
    return output


def prediction_lock_bytes(lock: pd.DataFrame) -> bytes:
    unexpected = OUTCOME_COLUMNS.intersection(lock.columns)
    if unexpected:
        raise RuntimeError(f"FATAL: prediction lock contains outcome: {sorted(unexpected)}")
    return dataframe_csv_bytes(lock[PREDICTION_COLUMNS])


def prediction_lock_sha256(lock: pd.DataFrame) -> str:
    return hashlib.sha256(prediction_lock_bytes(lock)).hexdigest()


def load_july_outcomes_after_lock(root: Path, lock: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in lock[["event_id", "signal_date", "code"]].itertuples(index=False):
        if not (JULY_START <= row.signal_date <= JULY_END):
            raise RuntimeError("FATAL: non-July prediction entered evaluation")
        daily = _read_daily(root, str(row.code).zfill(6))
        indices = daily.index[daily["date"].eq(row.signal_date)].tolist()
        if len(indices) != 1 or indices[0] + 2 >= len(daily):
            rows.append({"event_id": row.event_id})
            continue
        index = indices[0]
        d2 = daily.iloc[index + 1]
        d3 = daily.iloc[index + 2]
        d2_date = str(d2["date"])
        d3_date = str(d3["date"])
        d2_open = float(d2["open"])
        d3_high = float(d3["high"])
        if not (row.signal_date < d2_date < d3_date):
            raise RuntimeError(f"FATAL: D1/D2/D3 order invalid {row.event_id}")
        raw_return = d3_high / d2_open - 1.0
        rows.append({
            "event_id": row.event_id,
            "d2_date": d2_date,
            "d3_date": d3_date,
            "d2_open_daily": d2_open,
            "d3_high_daily": d3_high,
            "target7": int(raw_return >= 0.07),
            "raw_return": raw_return,
            "capped_return_7": float(capped7_utility(raw_return)),
        })
    outcomes = pd.DataFrame(rows)
    if outcomes["event_id"].duplicated().any() or len(outcomes) != len(lock):
        raise RuntimeError("FATAL: outcome identity duplicated")
    return outcomes


def join_evaluation(lock: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    evaluated = lock.merge(outcomes, on="event_id", how="left", validate="one_to_one")
    required = [
        "d2_date", "d3_date", "d2_open_daily", "d3_high_daily",
        "target7", "raw_return", "capped_return_7",
    ]
    if evaluated[required].isna().any().any():
        missing_mask = evaluated[required].isna().any(axis=1)
        missing_ids = evaluated.loc[missing_mask, "event_id"].astype(str).tolist()
        raise RuntimeError(
            f"FATAL: July outcome incomplete for {len(missing_ids)} candidates: {missing_ids}"
        )
    expected_raw = evaluated["d3_high_daily"] / evaluated["d2_open_daily"] - 1.0
    if not np.allclose(expected_raw, evaluated["raw_return"], rtol=0.0, atol=1e-12):
        raise RuntimeError("FATAL: July raw outcome semantic mismatch")
    expected_cap = capped7_utility(evaluated["raw_return"].to_numpy(float))
    if not np.allclose(expected_cap, evaluated["capped_return_7"], rtol=0.0, atol=1e-12):
        raise RuntimeError("FATAL: July capped outcome semantic mismatch")
    evaluated["raw_repair_return"] = evaluated["raw_return"]
    evaluated["capped_opportunity_return_7"] = evaluated["capped_return_7"]
    evaluated["target7"] = evaluated["target7"].astype(int)

    # Complete the reranker rank: Top10 by frozen Stage2, outside Top10 by Stage1.
    final_rank = pd.Series(index=evaluated.index, dtype=int)
    for _, day in evaluated.groupby("signal_date", sort=True):
        inside = day[day["stage1_top10"]].sort_values(
            ["capped_pair_rank", "stage1_rank", "event_id"], kind="mergesort"
        )
        outside = day[~day["stage1_top10"]].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        ordered = pd.concat([inside, outside])
        final_rank.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    evaluated["final_capped_rank"] = final_rank.astype(int)
    evaluated["control_top3"] = evaluated["stage1_rank"].le(
        np.minimum(3, evaluated["candidate_count"])
    )
    evaluated["capped_top3"] = evaluated["final_capped_rank"].le(
        np.minimum(3, evaluated["candidate_count"])
    )
    return evaluated.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _oracle_rank(frame: pd.DataFrame, top10_only: bool) -> pd.Series:
    ranks = pd.Series(index=frame.index, dtype=int)
    for _, day in frame.groupby("signal_date", sort=True):
        pool = day[day["stage1_top10"]] if top10_only else day
        ordered_pool = pool.sort_values(
            ["capped_opportunity_return_7", "event_id"],
            ascending=[False, True], kind="mergesort"
        )
        outside = day[~day.index.isin(ordered_pool.index)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        ordered = pd.concat([ordered_pool, outside])
        ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return ranks.astype(int)


def build_opportunity_buckets(frame: pd.DataFrame) -> dict[str, str]:
    daily = frame.groupby("signal_date", sort=True)[
        "capped_opportunity_return_7"
    ].mean().reset_index(name="opportunity")
    daily = daily.sort_values(
        ["opportunity", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)
    indices = np.array_split(np.arange(len(daily)), 3)
    mapping: dict[str, str] = {}
    for bucket, index in zip(("LOW", "MID", "HIGH"), indices):
        for row_index in index:
            mapping[str(daily.iloc[int(row_index)]["signal_date"])] = bucket
    if set(mapping) != set(frame["signal_date"].astype(str).unique()):
        raise RuntimeError("FATAL: July opportunity bucket coverage mismatch")
    return mapping


def practical_table(
    evaluated: pd.DataFrame, bucket_map: Mapping[str, str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = evaluated.copy()
    frame["top10_oracle_rank"] = _oracle_rank(frame, True)
    frame["full_oracle_rank"] = _oracle_rank(frame, False)
    rank_columns = {
        "CONTROL_V4A": "stage1_rank",
        "PAIR_CAPPED7": "final_capped_rank",
        "TOP10_ORACLE": "top10_oracle_rank",
        "FULL_ORACLE": "full_oracle_rank",
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scope in ("COMBINED", "LOW", "MID", "HIGH"):
        dates = sorted(frame["signal_date"].unique()) if scope == "COMBINED" else [
            date for date, bucket in bucket_map.items() if bucket == scope
        ]
        subset = frame[frame["signal_date"].isin(dates)]
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
            top3_daily = _top3_daily(subset, rank_column, "capped_opportunity_return_7")
            row["Top3_days_le_minus_10"] = int(top3_daily.le(-0.10).sum())
            rows.append(row)
        control = strategy_summary(subset, "stage1_rank")
        universe = {
            "scope": scope,
            "model": "UNIVERSE",
            "winner_capture": np.nan,
            "universe_target7_rate": control["universe_target7_rate"],
            "Top3_days_le_minus_3": np.nan,
            "Top3_days_le_minus_5": np.nan,
            "Top3_days_le_minus_10": np.nan,
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
    return pd.DataFrame(rows), summaries


def _top3_daily(frame: pd.DataFrame, rank_column: str, value_column: str) -> pd.Series:
    values: dict[str, float] = {}
    for date, day in frame.groupby("signal_date", sort=True):
        k = min(3, len(day))
        selected = day[day[rank_column].le(k)]
        if len(selected) != k:
            raise RuntimeError(f"FATAL: Top3 membership incomplete {date}")
        values[str(date)] = float(selected[value_column].mean())
    return pd.Series(values, dtype=float).sort_index()


def build_daily_output(
    evaluated: pd.DataFrame, bucket_map: Mapping[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in evaluated.groupby("signal_date", sort=True):
        control = day.sort_values(["stage1_rank", "event_id"], kind="mergesort")
        capped = day.sort_values(["final_capped_rank", "event_id"], kind="mergesort")
        k2, k3 = min(2, len(day)), min(3, len(day))
        c3 = control.head(k3)
        p3 = capped.head(k3)
        rows.append({
            "signal_date": date,
            "candidate_count": len(day),
            "universe_capped_mean": float(day["capped_return_7"].mean()),
            "universe_target7_rate": float(day["target7"].mean()),
            "control_rank1": float(control.iloc[0]["capped_return_7"]),
            "control_rank2": float(control.iloc[1]["capped_return_7"]) if len(day) >= 2 else np.nan,
            "control_rank3": float(control.iloc[2]["capped_return_7"]) if len(day) >= 3 else np.nan,
            "control_top2": float(control.head(k2)["capped_return_7"].mean()),
            "control_top3": float(c3["capped_return_7"].mean()),
            "capped_rank1": float(capped.iloc[0]["capped_return_7"]),
            "capped_rank2": float(capped.iloc[1]["capped_return_7"]) if len(day) >= 2 else np.nan,
            "capped_rank3": float(capped.iloc[2]["capped_return_7"]) if len(day) >= 3 else np.nan,
            "capped_top2": float(capped.head(k2)["capped_return_7"].mean()),
            "capped_top3": float(p3["capped_return_7"].mean()),
            "delta": float(p3["capped_return_7"].mean() - c3["capped_return_7"].mean()),
            "control_target7_count": int(c3["target7"].sum()),
            "capped_target7_count": int(p3["target7"].sum()),
            "opportunity_bucket": bucket_map[str(date)],
        })
    return pd.DataFrame(rows)


def daily_robustness(daily: pd.DataFrame) -> dict[str, Any]:
    delta = daily["delta"].to_numpy(float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw = generator.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
    boot = delta[draw].mean(axis=1)
    quantile = np.quantile(boot, [0.025, 0.50, 0.975])
    lodo = np.asarray([np.delete(delta, index).mean() for index in range(len(delta))])
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "positive_dates": int((delta > 0).sum()),
        "negative_dates": int((delta < 0).sum()),
        "zero_dates": int((delta == 0).sum()),
        "bootstrap_p025": float(quantile[0]),
        "bootstrap_p50": float(quantile[1]),
        "bootstrap_p975": float(quantile[2]),
        "bootstrap_probability_positive": float((boot > 0).mean()),
        "lodo_positive_pct": float((lodo > 0).mean()),
        "lodo_min": float(lodo.min()),
        "lodo_median": float(np.median(lodo)),
        "lodo_max": float(lodo.max()),
    }


def retrieval_audit(evaluated: pd.DataFrame) -> dict[str, Any]:
    total_targets = int(evaluated["target7"].sum())
    result: dict[str, Any] = {}
    for k in (3, 5, 7, 10):
        selected = int(evaluated.loc[evaluated["stage1_rank"].le(k), "target7"].sum())
        result[f"target7_recall_at_{k}"] = selected / total_targets if total_targets else np.nan
        captures: list[float] = []
        for _, day in evaluated.groupby("signal_date", sort=True):
            targets = int(day["target7"].sum())
            if targets <= 0:
                continue
            slots = min(k, targets)
            captures.append(float(day.loc[day["stage1_rank"].le(k), "target7"].sum() / slots))
        result[f"winner_capture_at_{k}"] = float(np.mean(captures)) if captures else np.nan
    return result


def membership_audit(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    union = evaluated[evaluated["control_top3"] | evaluated["capped_top3"]].copy()
    union["membership_status"] = np.select(
        [
            union["control_top3"] & union["capped_top3"],
            union["control_top3"] & ~union["capped_top3"],
            ~union["control_top3"] & union["capped_top3"],
        ],
        ["PRESERVED", "DEMOTED", "PROMOTED"],
        default="INVALID",
    )
    demoted = union[union["membership_status"].eq("DEMOTED")]
    promoted = union[union["membership_status"].eq("PROMOTED")]
    stats = {
        "changed_dates": int(union.loc[
            union["membership_status"].ne("PRESERVED"), "signal_date"
        ].nunique()),
        "changed_slots": int(len(promoted)),
        "demoted_total": int(len(demoted)),
        "demoted_target7": int(demoted["target7"].sum()),
        "demoted_non_target": int(demoted["target7"].eq(0).sum()),
        "demoted_losses": int(demoted["raw_return"].lt(0).sum()),
        "promoted_total": int(len(promoted)),
        "promoted_target7": int(promoted["target7"].sum()),
        "promoted_non_target": int(promoted["target7"].eq(0).sum()),
        "promoted_losses": int(promoted["raw_return"].lt(0).sum()),
        "promoted_severe_losses": int(promoted["raw_return"].le(-0.05).sum()),
        "net_target7_slots_gained": int(promoted["target7"].sum() - demoted["target7"].sum()),
    }
    output = union[[
        "signal_date", "event_id", "code", "stage1_rank", "final_capped_rank",
        "membership_status", "target7", "raw_return", "capped_return_7",
    ]].sort_values(["signal_date", "membership_status", "event_id"], kind="mergesort")
    return output.reset_index(drop=True), stats


def june_maturity_audit(base: pd.DataFrame, dates: pd.DataFrame) -> dict[str, int]:
    joined = base[["event_id", "signal_date"]].merge(
        dates[["event_id", "label_available_date"]], on="event_id", validate="one_to_one"
    )
    june_dates = sorted(joined.loc[
        joined["signal_date"].astype(str).str.startswith("2026-06"), "signal_date"
    ].unique())
    rows = 0
    folds = 0
    for test_date in june_dates:
        immature = joined[
            joined["signal_date"].lt(test_date)
            & joined["label_available_date"].notna()
            & joined["label_available_date"].ge(test_date)
        ]
        rows += len(immature)
        folds += int(not immature.empty)
    return {
        "june_immature_label_rows": int(rows),
        "june_folds_with_immature_labels": int(folds),
    }


def _scope_row(practical: pd.DataFrame, scope: str, model: str) -> pd.Series:
    row = practical[practical["scope"].eq(scope) & practical["model"].eq(model)]
    if len(row) != 1:
        raise RuntimeError(f"FATAL: practical row unavailable {scope}/{model}")
    return row.iloc[0]


def formal_decision(
    practical: pd.DataFrame,
    robustness: Mapping[str, Any],
    valid: bool,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    control = _scope_row(practical, "COMBINED", "CONTROL_V4A")
    capped = _scope_row(practical, "COMBINED", "PAIR_CAPPED7")
    low_control = _scope_row(practical, "LOW", "CONTROL_V4A")
    low_capped = _scope_row(practical, "LOW", "PAIR_CAPPED7")
    high_control = _scope_row(practical, "HIGH", "CONTROL_V4A")
    high_capped = _scope_row(practical, "HIGH", "PAIR_CAPPED7")
    gates = {
        "top3_control": bool(capped.Top3_mean > control.Top3_mean),
        "top3_universe": bool(capped.Top3_mean > capped.Top3_baseline),
        "delta": bool(capped.Top3_mean - control.Top3_mean >= 0.005),
        "precision_control": bool(capped.Top3_target7_precision > control.Top3_target7_precision),
        "precision_universe": bool(capped.Top3_target7_precision > capped.universe_target7_rate),
        "beat_universe": bool(capped.Top3_beat_universe > 0.50),
        "daily_breadth": bool(robustness["positive_dates"] > robustness["negative_dates"]),
        "negative_rate": bool(capped.Top3_negative_date_rate <= control.Top3_negative_date_rate),
        "worst": bool(capped.Top3_worst >= control.Top3_worst - 0.01),
        "days_minus5": bool(capped.Top3_days_le_minus_5 <= control.Top3_days_le_minus_5),
        "low_return": bool(low_capped.Top3_mean - low_control.Top3_mean >= 0.005),
        "low_negative": bool(low_capped.Top3_negative_date_rate <= low_control.Top3_negative_date_rate),
        "high_upside": bool(high_capped.Top3_mean >= high_control.Top3_mean - 0.005),
        "bootstrap": bool(robustness["bootstrap_probability_positive"] >= 0.65),
        "lodo": bool(robustness["lodo_positive_pct"] >= 0.70),
    }
    if not valid:
        signal = "INVALID"
    elif all(gates.values()):
        signal = "STRONG"
    elif capped.Top3_mean > control.Top3_mean or (
        capped.Top3_negative_date_rate < control.Top3_negative_date_rate
    ) or low_capped.Top3_mean > low_control.Top3_mean:
        signal = "PARTIAL"
    else:
        signal = "ABSENT"

    top10 = _scope_row(practical, "COMBINED", "TOP10_ORACLE")
    full = _scope_row(practical, "COMBINED", "FULL_ORACLE")
    universe = float(control.Top3_baseline)
    denominator = float(full.Top3_mean - universe)
    retrieval_ratio = (
        float((top10.Top3_mean - universe) / denominator) if denominator > 0 else np.nan
    )
    retrieval_useful = bool(
        retrieval.get("target7_recall_at_10", 0.0) >= 0.70
        and (pd.isna(retrieval_ratio) or retrieval_ratio >= 0.80)
    )
    if signal == "STRONG":
        attribution = "NONE"
    elif retrieval_useful:
        attribution = "STAGE2_RERANK_FAILURE"
    elif capped.Top3_mean <= control.Top3_mean:
        attribution = "BOTH"
    elif len(practical) == 0:
        attribution = "INSUFFICIENT_FORWARD_EVIDENCE"
    else:
        attribution = "STAGE1_RETRIEVAL_FAILURE"
    return {
        "signal": signal,
        "gates": gates,
        "primary_failure_attribution": attribution,
        "retrieval_oracle_ratio": retrieval_ratio,
        "retrieval_useful": retrieval_useful,
    }


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _num(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def render_review(context: Mapping[str, Any]) -> str:
    practical = context["practical"]
    control = _scope_row(practical, "COMBINED", "CONTROL_V4A")
    capped = _scope_row(practical, "COMBINED", "PAIR_CAPPED7")
    top10 = _scope_row(practical, "COMBINED", "TOP10_ORACLE")
    full = _scope_row(practical, "COMBINED", "FULL_ORACLE")
    low_c = _scope_row(practical, "LOW", "CONTROL_V4A")
    low_p = _scope_row(practical, "LOW", "PAIR_CAPPED7")
    high_c = _scope_row(practical, "HIGH", "CONTROL_V4A")
    high_p = _scope_row(practical, "HIGH", "PAIR_CAPPED7")
    audit = context["training_audit"]
    robust = context["robustness"]
    membership = context["membership_stats"]
    decision = context["decision"]
    retrieval = context["retrieval"]
    raw = context["raw_downside"]
    lines = [
        "# v004c PAIR_CAPPED7 July Chronological Forward Stress v001",
        "",
        "## 1. Frozen Specification",
        "",
        f"- Frozen development commit: `{FROZEN_DEVELOPMENT_COMMIT}`",
        f"- Forward class: `{FORWARD_STRESS_CLASS}`",
        "- Stage1: frozen 18-feature weighted L2 Logistic; L2=0.30; positive weight=1.50; original tail/date weights.",
        "- Stage2: frozen same-date pairwise weighted Ridge; CAPPED7; 3 predictors; L2=0.30; intercept=0; Top10 only.",
        "- July model selected before outcome evaluation: **YES**; tuning/new features/July refit: **NO**.",
        "",
        "## 2. Temporal / Label Availability Audit",
        "",
        f"- Label available date: D3 date; training rule: `label_available_date < {TRAINING_ASOF_DATE}`.",
        f"- June immature label rows across original folds: **{context['maturity']['june_immature_label_rows']}**; affected folds: **{context['maturity']['june_folds_with_immature_labels']}**.",
        f"- June temporal caveat: **{'YES' if context['maturity']['june_immature_label_rows'] else 'NO'}**.",
        "",
        "## 3. Pre-July Training Snapshot",
        "",
        f"- Eligible rows/dates: **{audit['eligible_historical_rows']} / {audit['eligible_historical_dates']}**.",
        f"- Latest signal / label date: **{audit['latest_eligible_signal_date']} / {audit['latest_label_available_date']}**.",
        f"- Stage1 fitted once; Stage2 fitted once; meta rows/pairs: **{audit['stage2_meta_rows']} / {audit['stage2_pair_count']}**.",
        f"- Self-label leakage / July training / August training: **{audit['self_label_leakage_rows']} / {audit['july_training_rows']} / {audit['august_training_rows']}**.",
        "",
        "## 4. Prediction Lock and Outcome Independence",
        "",
        f"- Prediction rows/dates: **{context['july_rows']} / {context['july_dates']}**.",
        f"- SHA256: `{context['prediction_lock_sha256']}`.",
        f"- Outcome columns present: **NO**; perturbation test: **{context['outcome_perturbation']}**; outcome influence: **0**.",
        "",
        "## 5. July Universe and Data Coverage",
        "",
        f"- Rows/dates: **{context['july_rows']} / {context['july_dates']}**; Board2/Board3: **{context['board2']} / {context['board3']}**.",
        f"- Candidate count min/median/max: **{context['candidate_min']} / {_num(context['candidate_median'], 1)} / {context['candidate_max']}**.",
        f"- Outcome complete/missing: **{context['outcome_complete']} / {context['outcome_missing']}**; August only for July outcome completion: **{'YES' if context['august_outcome_only'] else 'NO'}**.",
        "",
        "## 6. CONTROL vs PAIR_CAPPED7",
        "",
        "| Model | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Excess | Top1 Hit | Top3 Precision | Beat Universe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| UNIVERSE | {_pct(control.Rank1_baseline)} | {_pct(control.Rank2_baseline)} | {_pct(control.Rank3_baseline)} | {_pct(control.Top2_baseline)} | {_pct(control.Top3_baseline)} | 0.0000% | {_pct(control.universe_target7_rate)} | {_pct(control.universe_target7_rate)} | NA |",
        f"| CONTROL_V4A | {_pct(control.Rank1_mean)} | {_pct(control.Rank2_mean)} | {_pct(control.Rank3_mean)} | {_pct(control.Top2_mean)} | {_pct(control.Top3_mean)} | {_pct(control.Top3_excess)} | {_pct(control.Top1_target7_precision)} | {_pct(control.Top3_target7_precision)} | {_pct(control.Top3_beat_universe)} |",
        f"| PAIR_CAPPED7 | {_pct(capped.Rank1_mean)} | {_pct(capped.Rank2_mean)} | {_pct(capped.Rank3_mean)} | {_pct(capped.Top2_mean)} | {_pct(capped.Top3_mean)} | {_pct(capped.Top3_excess)} | {_pct(capped.Top1_target7_precision)} | {_pct(capped.Top3_target7_precision)} | {_pct(capped.Top3_beat_universe)} |",
        f"| TOP10_ORACLE | {_pct(top10.Rank1_mean)} | {_pct(top10.Rank2_mean)} | {_pct(top10.Rank3_mean)} | {_pct(top10.Top2_mean)} | {_pct(top10.Top3_mean)} | {_pct(top10.Top3_excess)} | {_pct(top10.Top1_target7_precision)} | {_pct(top10.Top3_target7_precision)} | {_pct(top10.Top3_beat_universe)} |",
        f"| FULL_ORACLE | {_pct(full.Rank1_mean)} | {_pct(full.Rank2_mean)} | {_pct(full.Rank3_mean)} | {_pct(full.Top2_mean)} | {_pct(full.Top3_mean)} | {_pct(full.Top3_excess)} | {_pct(full.Top1_target7_precision)} | {_pct(full.Top3_target7_precision)} | {_pct(full.Top3_beat_universe)} |",
        f"- CAPPED - CONTROL: **{_pct(capped.Top3_mean - control.Top3_mean)}**; CAPPED - Universe: **{_pct(capped.Top3_excess)}**.",
        "",
        "## 7. Downside Risk",
        "",
        "| Model | Mean | Median | P25 | Worst | Positive Dates | Negative Dates | <=-3% | <=-5% | <=-10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| CONTROL | {_pct(control.Top3_mean)} | {_pct(control.Top3_median)} | {_pct(control.Top3_p25)} | {_pct(control.Top3_worst)} | {_pct(control.Top3_positive_date_rate)} | {_pct(control.Top3_negative_date_rate)} | {int(control.Top3_days_le_minus_3)} | {int(control.Top3_days_le_minus_5)} | {int(control.Top3_days_le_minus_10)} |",
        f"| PAIR_CAPPED7 | {_pct(capped.Top3_mean)} | {_pct(capped.Top3_median)} | {_pct(capped.Top3_p25)} | {_pct(capped.Top3_worst)} | {_pct(capped.Top3_positive_date_rate)} | {_pct(capped.Top3_negative_date_rate)} | {int(capped.Top3_days_le_minus_3)} | {int(capped.Top3_days_le_minus_5)} | {int(capped.Top3_days_le_minus_10)} |",
        f"- Raw selected-stock mean/median/worst: CONTROL **{_pct(raw['control_stock_mean'])} / {_pct(raw['control_stock_median'])} / {_pct(raw['control_worst_stock'])}**; CAPPED **{_pct(raw['capped_stock_mean'])} / {_pct(raw['capped_stock_median'])} / {_pct(raw['capped_worst_stock'])}**.",
        "",
        "## 8. LOW / MID / HIGH Opportunity",
        "",
        f"- LOW Top3: Universe **{_pct(low_c.Top3_baseline)}**; CONTROL **{_pct(low_c.Top3_mean)}**; CAPPED **{_pct(low_p.Top3_mean)}**; delta **{_pct(low_p.Top3_mean-low_c.Top3_mean)}**; negative dates **{_pct(low_c.Top3_negative_date_rate)} → {_pct(low_p.Top3_negative_date_rate)}**.",
        f"- MID Top3: CONTROL **{_pct(_scope_row(practical, 'MID', 'CONTROL_V4A').Top3_mean)}**; CAPPED **{_pct(_scope_row(practical, 'MID', 'PAIR_CAPPED7').Top3_mean)}**.",
        f"- HIGH Top3: CONTROL **{_pct(high_c.Top3_mean)}**; CAPPED **{_pct(high_p.Top3_mean)}**; delta **{_pct(high_p.Top3_mean-high_c.Top3_mean)}**; precision **{_pct(high_c.Top3_target7_precision)} → {_pct(high_p.Top3_target7_precision)}**.",
        "",
        "## 9. Stage1 Retrieval and Oracle Gap",
        "",
        f"- Target7 recall@3/@5/@7/@10: **{_pct(retrieval['target7_recall_at_3'])} / {_pct(retrieval['target7_recall_at_5'])} / {_pct(retrieval['target7_recall_at_7'])} / {_pct(retrieval['target7_recall_at_10'])}**.",
        f"- Date-equal winner capture@3/@5/@10: **{_pct(retrieval['winner_capture_at_3'])} / {_pct(retrieval['winner_capture_at_5'])} / {_pct(retrieval['winner_capture_at_10'])}**.",
        f"- Top10 oracle / full oracle / retrieval oracle ratio: **{_pct(top10.Top3_mean)} / {_pct(full.Top3_mean)} / {_pct(decision['retrieval_oracle_ratio'])}**.",
        f"- Available rerank gap / recovered gap / recovery ratio: **{_pct(top10.Top3_mean-control.Top3_mean)} / {_pct(capped.Top3_mean-control.Top3_mean)} / {_pct(context['reranker_recovery_ratio'])}**.",
        "",
        "## 10. Membership Changes",
        "",
        f"- Changed dates/slots: **{membership['changed_dates']} / {membership['changed_slots']}**.",
        f"- Demoted Target7/non-target: **{membership['demoted_target7']} / {membership['demoted_non_target']}**; promoted Target7/non-target/loss/severe: **{membership['promoted_target7']} / {membership['promoted_non_target']} / {membership['promoted_losses']} / {membership['promoted_severe_losses']}**.",
        f"- Net Target7 slots gained: **{membership['net_target7_slots_gained']}**.",
        "",
        "## 11. Daily Robustness",
        "",
        f"- Daily delta mean/median; positive/negative/zero: **{_pct(robust['mean'])} / {_pct(robust['median'])}; {robust['positive_dates']} / {robust['negative_dates']} / {robust['zero_dates']}**.",
        f"- Bootstrap p2.5/p50/p97.5; P(delta>0): **{_pct(robust['bootstrap_p025'])} / {_pct(robust['bootstrap_p50'])} / {_pct(robust['bootstrap_p975'])}; {_pct(robust['bootstrap_probability_positive'])}**.",
        f"- LODO positive; min/median/max: **{_pct(robust['lodo_positive_pct'])}; {_pct(robust['lodo_min'])} / {_pct(robust['lodo_median'])} / {_pct(robust['lodo_max'])}**.",
        "",
        "## 12. Forward-Stress Decision",
        "",
        f"- Gates: " + ", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in decision['gates'].items()) + ".",
        f"- `FORWARD_STRESS_SIGNAL = {decision['signal']}`",
        f"- `PRIMARY_FAILURE_ATTRIBUTION = {decision['primary_failure_attribution']}`",
        "- June development reference: CONTROL 1.8539%; CAPPED 3.0443%; delta +1.1904pp.",
        f"- July: CONTROL {_pct(control.Top3_mean)}; CAPPED {_pct(capped.Top3_mean)}; delta {_pct(capped.Top3_mean-control.Top3_mean)}.",
        "- No June/July retuning, no alternative target, no production model, no deployment action.",
    ]
    return "\n".join(lines) + "\n"


def _training_audit_frame(
    audit: Mapping[str, Any], maturity: Mapping[str, Any], lock_sha: str,
    identity_audit: Mapping[str, Any],
) -> pd.DataFrame:
    values = {
        "training_snapshot_date": TRAINING_ASOF_DATE,
        **audit,
        **maturity,
        **identity_audit,
        "stage1_fitted_count": 1,
        "stage2_fitted_count": 1,
        "prediction_lock_sha256": lock_sha,
        "label_availability_rule": "label_available_date < 2026-07-01",
        "forward_stress_class": FORWARD_STRESS_CLASS,
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in values.items()])


def build_forward_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_frozen_forward_contract()

    # Phase A: D1-safe features + pre-July matured labels only.
    base = load_authoritative_d1_safe_base(root)
    base_raw, _, _, feature_gate = build_exact_raw_features(root, base)
    if not feature_gate["pass"]:
        raise RuntimeError("FATAL: frozen May-June raw feature gate failed")
    base_x = prepare_frozen_x(base_raw)
    base_residual = base[["event_id", *RESIDUAL_RAW_FIELDS]].copy()
    for feature in RESIDUAL_RAW_FIELDS:
        base_residual[feature] = pd.to_numeric(base_residual[feature], errors="coerce")
    base_x = base_x.merge(
        base_residual, on="event_id", how="left", validate="one_to_one"
    )
    base_dates = derive_trade_dates(root, base)
    maturity = june_maturity_audit(base, base_dates)
    mature_outcomes = load_mature_training_outcomes(root, base, base_dates)
    samples = attach_training_outcomes(base_x, mature_outcomes)

    july_universe, identity_audit = build_july_universe(root)
    july_raw = build_july_d1_safe_features(root, july_universe)
    july_x = prepare_frozen_x(july_raw)
    if not set(RESIDUAL_RAW_FIELDS).issubset(july_x.columns):
        raise RuntimeError("FATAL: July residual inputs lost during frozen X transform")
    stage1_beta, stage2_beta, meta, snapshot_audit = build_pre_july_snapshot(
        base_x, samples
    )
    lock = build_prediction_lock(july_x, stage1_beta, stage2_beta)
    lock_bytes = prediction_lock_bytes(lock)
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()

    # Adversarial outcome columns are ignored by the prediction contract.
    perturbed = july_x.copy()
    rng = np.random.default_rng(20260810)
    for column in ("target7", "d2_open", "d3_high", "raw_return", "capped_return"):
        perturbed[column] = rng.normal(0.0, 1000.0, len(perturbed))
    perturbed_lock = build_prediction_lock(perturbed, stage1_beta, stage2_beta)
    perturbation_pass = prediction_lock_sha256(perturbed_lock) == lock_sha
    if not perturbation_pass:
        raise RuntimeError("FATAL: July outcome perturbation changed prediction lock")

    # Phase B begins only after prediction bytes and hash exist in memory.
    outcomes = load_july_outcomes_after_lock(root, lock)
    evaluated = join_evaluation(lock, outcomes)
    bucket_map = build_opportunity_buckets(evaluated)
    practical, summaries = practical_table(evaluated, bucket_map)
    daily = build_daily_output(evaluated, bucket_map)
    robustness = daily_robustness(daily)
    retrieval = retrieval_audit(evaluated)
    membership, membership_stats = membership_audit(evaluated)
    decision = formal_decision(practical, robustness, True, retrieval)

    control = _scope_row(practical, "COMBINED", "CONTROL_V4A")
    capped = _scope_row(practical, "COMBINED", "PAIR_CAPPED7")
    top10 = _scope_row(practical, "COMBINED", "TOP10_ORACLE")
    rerank_denominator = float(top10.Top3_mean - control.Top3_mean)
    reranker_recovery = (
        float((capped.Top3_mean - control.Top3_mean) / rerank_denominator)
        if rerank_denominator > 0 else np.nan
    )
    control_selected = evaluated[evaluated["control_top3"]]
    capped_selected = evaluated[evaluated["capped_top3"]]
    raw_downside = {
        "control_stock_mean": float(control_selected["raw_return"].mean()),
        "control_stock_median": float(control_selected["raw_return"].median()),
        "control_worst_stock": float(control_selected["raw_return"].min()),
        "control_worst_date": float(_top3_daily(evaluated, "stage1_rank", "raw_return").min()),
        "capped_stock_mean": float(capped_selected["raw_return"].mean()),
        "capped_stock_median": float(capped_selected["raw_return"].median()),
        "capped_worst_stock": float(capped_selected["raw_return"].min()),
        "capped_worst_date": float(_top3_daily(evaluated, "final_capped_rank", "raw_return").min()),
    }
    august_outcome_only = bool(
        evaluated["d2_date"].ge(AUGUST_SIGNAL_SENTINEL).any()
        or evaluated["d3_date"].ge(AUGUST_SIGNAL_SENTINEL).any()
    ) and not bool(evaluated["signal_date"].ge(AUGUST_SIGNAL_SENTINEL).any())
    context: dict[str, Any] = {
        "training_audit": snapshot_audit,
        "maturity": maturity,
        "identity_audit": identity_audit,
        "prediction_lock_sha256": lock_sha,
        "outcome_perturbation": "PASS",
        "july_rows": len(lock),
        "july_dates": lock["signal_date"].nunique(),
        "board2": int(lock["board"].eq(2).sum()),
        "board3": int(lock["board"].eq(3).sum()),
        "candidate_min": int(lock.groupby("signal_date").size().min()),
        "candidate_median": float(lock.groupby("signal_date").size().median()),
        "candidate_max": int(lock.groupby("signal_date").size().max()),
        "outcome_complete": int(outcomes.dropna(subset=["d2_date", "d3_date"]).shape[0]),
        "outcome_missing": int(outcomes[["d2_date", "d3_date"]].isna().any(axis=1).sum()),
        "august_outcome_only": august_outcome_only,
        "practical": practical,
        "summaries": summaries,
        "daily": daily,
        "robustness": robustness,
        "retrieval": retrieval,
        "membership_stats": membership_stats,
        "decision": decision,
        "raw_downside": raw_downside,
        "reranker_recovery_ratio": reranker_recovery,
    }
    review = render_review(context)
    training_audit = _training_audit_frame(
        snapshot_audit, maturity, lock_sha, identity_audit
    )
    oof_columns = PREDICTION_COLUMNS + [
        "d2_date", "d3_date", "d2_open_daily", "d3_high_daily",
        "target7", "raw_return", "capped_return_7", "final_capped_rank",
        "control_top3", "capped_top3",
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: lock_bytes,
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(training_audit),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(evaluated[oof_columns]),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(membership),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_pair_capped7_july_forward(
    root: Path,
    output_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    target = output_dir or (
        root / "reports/research/v004c_pair_capped7_july_forward_v001_20260701_20260731"
    )
    first_outputs, first_context = build_forward_outputs(root)
    second_outputs, _ = build_forward_outputs(root)
    if first_outputs != second_outputs:
        mismatched = [name for name in first_outputs if first_outputs[name] != second_outputs[name]]
        raise RuntimeError(f"FATAL: deterministic rebuild failed: {mismatched}")
    target.mkdir(parents=True, exist_ok=True)
    for name, content in first_outputs.items():
        (target / name).write_bytes(content)
    first_context["deterministic_rebuild"] = "PASS"
    return target, first_context

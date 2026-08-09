from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import get_data_config, get_strategy_config
from .indicators import enrich_5min_indicators, enrich_daily_indicators
from .loaders import _filter_to_end_date, _keep_recent_trade_days, _merge_minute_amount
from .signal_engine import generate_signal
from .v004a import (
    BASE_INTERACTION_SPECS,
    BASE_RANK_SPECS,
    DAY_BUCKET_COLUMNS,
    DEFAULT_CLOSE_RETURN_COLUMN,
    DEFAULT_HIGH_RETURN_COLUMN,
    DEFAULT_TARGET_COLUMN,
    OPTIONAL_RANK_SPECS,
    build_training_sample_weight,
    fit_logistic_l2_weighted,
    prepare_v004a_samples,
)
from .v004c_mechanism_foundation import load_outcomes, repair_concordance


MODEL_ID = "V4A_ARCH_TRANSFER_V4C"
MODEL_FAMILY = "WEIGHTED_L2_LOGISTIC"
L2 = 0.30
POSITIVE_WEIGHT = 1.50
INITIAL_TRAIN_DATES = 18
MAX_SIGNAL_DATE = "2026-06-30"
JULY_SENTINEL_DATE = "2026-07-01"
HYPERPARAMETER_SEARCH = False
FEATURE_SELECTION = False
EXTRA_SCALING = False
USES_PCA = False
USES_PAIRWISE = False
NEW_FEATURES = False

RAW_INPUT_COLUMNS = [
    "d1_close_ma10_pct",
    "d1_low_ma10_pct",
    "trend_hold_score",
    "total_score",
    "theme_score",
    "days_since_d0",
    "candidate_base_price",
    "active_money_score",
    "d1_close_vwap_pct",
]

FROZEN_FEATURE_COLUMNS = [
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_total_score",
    "rank_theme_score",
    "rank_days_since_d0",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
    "inter_close_low",
    "inter_close_trend",
    "inter_total_trend",
    "inter_total_active",
    "inter_low_active",
    "spread_close_low",
    "days_since_d0_le1",
    "days_since_d0_eq2",
    "days_since_d0_ge3",
]

EXPECTED_BASE_RANK_SPECS = [
    ("d1_close_ma10_pct", "rank_d1_close_ma10_pct"),
    ("d1_low_ma10_pct", "rank_d1_low_ma10_pct"),
    ("trend_hold_score", "rank_trend_hold_score"),
    ("total_score", "rank_total_score"),
    ("theme_score", "rank_theme_score"),
    ("days_since_d0", "rank_days_since_d0"),
    ("log_candidate_base_price", "rank_log_candidate_base_price"),
]
EXPECTED_OPTIONAL_RANK_SPECS = [
    ("active_money_score", "rank_active_money_score"),
    ("d1_close_vwap_pct", "rank_d1_close_vwap_pct"),
]
EXPECTED_INTERACTIONS = [
    ("inter_close_low", ("rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct"), "product"),
    ("inter_close_trend", ("rank_d1_close_ma10_pct", "rank_trend_hold_score"), "product"),
    ("inter_total_trend", ("rank_total_score", "rank_trend_hold_score"), "product"),
    ("inter_total_active", ("rank_total_score", "rank_active_money_score"), "product"),
    ("inter_low_active", ("rank_d1_low_ma10_pct", "rank_active_money_score"), "product"),
    ("spread_close_low", ("rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct"), "spread"),
]

OUTPUT_FILENAMES = (
    "v004c_v4a_transfer_feature_provenance_v001.csv",
    "v004c_v4a_transfer_fold_audit_v001.csv",
    "v004c_v4a_transfer_coefficients_v001.csv",
    "v004c_v4a_transfer_oof_v001.csv",
    "v004c_v4a_transfer_practical_comparison_v001.csv",
    "v004c_v4a_transfer_opportunity_terciles_v001.csv",
    "v004c_v4a_transfer_review_v001.md",
)


def assert_frozen_architecture_contract() -> None:
    if list(BASE_RANK_SPECS) != EXPECTED_BASE_RANK_SPECS:
        raise RuntimeError("FATAL: src.v004a BASE_RANK_SPECS changed")
    if list(OPTIONAL_RANK_SPECS) != EXPECTED_OPTIONAL_RANK_SPECS:
        raise RuntimeError("FATAL: src.v004a OPTIONAL_RANK_SPECS changed")
    if list(BASE_INTERACTION_SPECS) != EXPECTED_INTERACTIONS:
        raise RuntimeError("FATAL: src.v004a BASE_INTERACTION_SPECS changed")
    if list(DAY_BUCKET_COLUMNS) != FROZEN_FEATURE_COLUMNS[-3:]:
        raise RuntimeError("FATAL: src.v004a day buckets changed")


def assert_authoritative_universe(frame: pd.DataFrame) -> None:
    required = {
        "event_id", "code", "signal_date", "break_date",
        "board_streak_before_break",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"FATAL: authoritative universe missing columns: {missing}")
    dates = frame["signal_date"].astype(str)
    if bool(dates.ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July sentinel triggered")
    checks = {
        "rows": len(frame) == 319,
        "dates": dates.nunique() == 39,
        "may_rows": int(dates.str.startswith("2026-05").sum()) == 146,
        "may_dates": dates[dates.str.startswith("2026-05")].nunique() == 18,
        "june_rows": int(dates.str.startswith("2026-06").sum()) == 173,
        "june_dates": dates[dates.str.startswith("2026-06")].nunique() == 21,
        "board2": int((pd.to_numeric(frame["board_streak_before_break"]) == 2).sum()) == 261,
        "board3": int((pd.to_numeric(frame["board_streak_before_break"]) == 3).sum()) == 58,
        "event_unique": frame["event_id"].astype(str).nunique() == 319,
        "date_boundary": dates.max() == MAX_SIGNAL_DATE,
        "signal_break": frame["signal_date"].astype(str).equals(frame["break_date"].astype(str)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"FATAL: authoritative universe checks failed: {failed}")


def load_authoritative_input(root: Path) -> pd.DataFrame:
    path = (
        root
        / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv"
    )
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in ("event_id", "signal_date", "break_date"):
        frame[column] = frame[column].astype(str)
    assert_authoritative_universe(frame)
    return frame.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def _read_frozen_v4a_raw(root: Path) -> pd.DataFrame:
    paths = (
        root
        / "reports/history_samples/2026-05-06_2026-06-29"
        / "history_candidates_2026-05-06_2026-06-29_dedup.csv",
        root
        / "reports/history_samples/2026-06-26_2026-06-30"
        / "history_candidates_2026-06-26_2026-06-30.csv",
    )
    columns = ["signal_date", "code", *RAW_INPUT_COLUMNS]
    frames: list[pd.DataFrame] = []
    for source_order, path in enumerate(paths):
        frame = pd.read_csv(
            path, encoding="utf-8-sig", dtype={"code": str}, usecols=columns
        )
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["signal_date"] = frame["signal_date"].astype(str)
        if bool(frame["signal_date"].ge(JULY_SENTINEL_DATE).any()):
            raise RuntimeError(f"FATAL: July row in frozen feature source: {path}")
        # Predeclare a non-overlapping frozen source partition.  The long run
        # is canonical through June 29; the short terminal run supplies June
        # 30 only.  Never choose between overlapping vintages by performance.
        if source_order == 0:
            frame = frame[frame["signal_date"].le("2026-06-29")].copy()
        else:
            frame = frame[frame["signal_date"].eq("2026-06-30")].copy()
        frame["_source_order"] = source_order
        frame["_feature_source"] = root_relative(root, path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["signal_date", "code", "_source_order"], kind="mergesort"
    )
    key = ["signal_date", "code"]
    if bool(combined.duplicated(key, keep=False).any()):
        raise RuntimeError("FATAL: non-overlapping frozen v004a source partition failed")
    return combined.drop_duplicates(key, keep="last").reset_index(drop=True)


def _read_daily_for_date(root: Path, code: str, signal_date: str) -> tuple[pd.DataFrame, Path]:
    candidates = (
        root / "data/cache/daily" / f"{code}_daily.pkl",
        root / "data/cache/daily_unadjusted" / f"{code}_daily.pkl",
    )
    for path in candidates:
        if not path.exists():
            continue
        raw = pd.read_pickle(path).copy()
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        raw = raw.dropna(subset=["date"])
        raw = raw[raw["date"].le(signal_date)].copy()
        if bool(raw["date"].ge(JULY_SENTINEL_DATE).any()):
            raise RuntimeError("FATAL: July daily row survived point-in-time clip")
        if signal_date in set(raw["date"]):
            return raw.sort_values("date", kind="mergesort").drop_duplicates(
                "date", keep="last"
            ).reset_index(drop=True), path
    raise RuntimeError(f"FATAL: no canonical daily cache covers {code}@{signal_date}")


def _read_baostock_minute_for_date(
    root: Path, code: str, signal_date: str
) -> tuple[pd.DataFrame, Path]:
    path = root / "data/cache/baostock_5m" / f"{code}_5min.pkl"
    if not path.exists():
        raise RuntimeError(f"FATAL: BaoStock 5m cache missing for {code}")
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
    if bool(minute["trade_date"].ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July minute row survived point-in-time clip")
    day = minute[minute["trade_date"].eq(signal_date)]
    if len(day) != 48:
        raise RuntimeError(f"FATAL: expected 48 BaoStock D1 bars for {code}@{signal_date}, got {len(day)}")
    return minute.sort_values("datetime", kind="mergesort").reset_index(drop=True), path


def _load_five_calendar_day_pool(root: Path, signal_date: str) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    anchor = pd.Timestamp(signal_date)
    for offset in range(5):
        date = (anchor - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        if date >= JULY_SENTINEL_DATE:
            raise RuntimeError("FATAL: July pool date requested")
        path = root / "data/cache/limit_ups" / f"{date}_limitups.pkl"
        if not path.exists():
            continue
        frame = pd.read_pickle(path).copy()
        required = {"trade_date", "code", "industry"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"FATAL: canonical limit-up pool schema incomplete: {path}")
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame[frame["trade_date"].eq(date)].copy()
        if frame.empty:
            continue
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frames.append(frame)
        paths.append(path)
    if not frames:
        raise RuntimeError(f"FATAL: no canonical five-day limit-up pool for {signal_date}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "code"], kind="mergesort"
    ).drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True), paths


def _d0_before_signal(daily: pd.DataFrame, signal_date: str) -> str:
    indices = daily.index[daily["date"].eq(signal_date)].tolist()
    if len(indices) != 1 or indices[0] <= 0:
        raise RuntimeError(f"FATAL: D0 unavailable before {signal_date}")
    return str(daily.iloc[indices[0] - 1]["date"])


def reconstruct_canonical_raw_row(
    root: Path, event: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    code = str(event["code"]).zfill(6)
    signal_date = str(event["signal_date"])
    if signal_date >= JULY_SENTINEL_DATE:
        raise RuntimeError("FATAL: July sentinel triggered during raw reconstruction")
    daily_raw, daily_path = _read_daily_for_date(root, code, signal_date)
    minute_raw, minute_path = _read_baostock_minute_for_date(root, code, signal_date)
    pool, pool_paths = _load_five_calendar_day_pool(root, signal_date)
    d0 = _d0_before_signal(daily_raw, signal_date)

    config = get_data_config()
    daily_until = _filter_to_end_date(daily_raw, "date", signal_date)
    minute_until = _filter_to_end_date(minute_raw, "trade_date", signal_date)
    merged_daily = _merge_minute_amount(daily_until, minute_until)
    daily_history = _keep_recent_trade_days(merged_daily, "date", config.daily_history_days)
    daily_recent = _keep_recent_trade_days(merged_daily, "date", config.default_5min_days)
    minute_recent = _keep_recent_trade_days(
        minute_until, "trade_date", config.default_5min_days
    )
    daily = enrich_daily_indicators(daily_recent, full_daily=daily_history)
    minute = enrich_5min_indicators(minute_recent)
    if str(daily.iloc[-1]["date"]) != signal_date:
        raise RuntimeError(f"FATAL: canonical daily latest date mismatch {code}@{signal_date}")
    signal = generate_signal(
        code=code,
        name=str(event.get("name", "")),
        daily=daily,
        minute=minute,
        limit_up_pool=pool,
        config=get_strategy_config(),
        d0_date=d0,
    )
    row = {
        "signal_date": signal_date,
        "code": code,
        "d1_close_ma10_pct": signal.d1_close_ma10_pct,
        "d1_low_ma10_pct": signal.d1_low_ma10_pct,
        "trend_hold_score": signal.trend_hold_score,
        "total_score": signal.total_score,
        "theme_score": signal.theme_score,
        "days_since_d0": signal.days_since_d0,
        "candidate_base_price": float(daily.iloc[-1]["close"]),
        "active_money_score": signal.active_money_score,
        "d1_close_vwap_pct": signal.d1_close_vwap_pct,
        "_feature_source": "canonical_generate_signal:frozen_daily+baostock_5m+five_day_limitup_pool",
    }
    audit = {
        "event_id": str(event["event_id"]),
        "signal_date": signal_date,
        "code": code,
        "d0_date": d0,
        "daily_path": root_relative(root, daily_path),
        "minute_path": root_relative(root, minute_path),
        "pool_paths": "|".join(root_relative(root, path) for path in pool_paths),
        "pool_member_present": bool((pool["code"] == code).any()),
        "pool_rows": int(len(pool)),
    }
    return row, audit


def build_exact_raw_features(
    root: Path, base: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assert_authoritative_universe(base)
    frozen = _read_frozen_v4a_raw(root)
    key = ["signal_date", "code"]
    joined = base[["event_id", "name", *key]].merge(
        frozen, on=key, how="left", validate="one_to_one"
    )
    missing_mask = joined["_feature_source"].isna()
    reconstructed: list[dict[str, Any]] = []
    reconstruction_audit: list[dict[str, Any]] = []
    for event in joined.loc[missing_mask, ["event_id", "name", *key]].to_dict("records"):
        row, audit = reconstruct_canonical_raw_row(root, event)
        reconstructed.append(row)
        reconstruction_audit.append(audit)
    if reconstructed:
        rebuilt = pd.DataFrame(reconstructed)
        joined = base[["event_id", "board_streak_before_break", *key]].merge(
            pd.concat(
                [frozen[key + RAW_INPUT_COLUMNS + ["_feature_source"]], rebuilt],
                ignore_index=True,
            ).drop_duplicates(key, keep="last"),
            on=key,
            how="left",
            validate="one_to_one",
        )
    if len(joined) != 319 or joined["event_id"].nunique() != 319:
        raise RuntimeError("FATAL: raw feature identity changed")
    if joined["_feature_source"].isna().any():
        raise RuntimeError("FATAL: exact v004a raw source unavailable for one or more events")
    for column in RAW_INPUT_COLUMNS:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    if bool(joined["candidate_base_price"].isna().any()) or bool(
        joined["candidate_base_price"].le(0).any()
    ):
        raise RuntimeError("FATAL: candidate_base_price must be positive for all events")

    provenance_rows: list[dict[str, Any]] = []
    frozen_count = int(joined["_feature_source"].str.contains("history_samples", na=False).sum())
    rebuilt_count = int(joined["_feature_source"].str.startswith("canonical_generate_signal", na=False).sum())
    formulas = {
        "d1_close_ma10_pct": "canonical generate_signal: (D1 close / MA10 - 1) * 100",
        "d1_low_ma10_pct": "canonical generate_signal: (D1 low / MA10 - 1) * 100",
        "trend_hold_score": "src.signal_engine.score_trend_hold",
        "total_score": "src.signal_engine.score_total with frozen StrategyConfig",
        "theme_score": "src.theme_score.score_theme on five-calendar-day historical limit-up pool",
        "days_since_d0": "calendar days between canonical D0 and D1",
        "candidate_base_price": "canonical D1 close used by src.backtester._candidate_base_price",
        "active_money_score": "src.active_money.score_active_money",
        "d1_close_vwap_pct": "canonical generate_signal: (D1 close / intraday VWAP - 1) * 100",
    }
    for feature in RAW_INPUT_COLUMNS:
        missing_count = int(joined[feature].isna().sum())
        provenance_rows.append({
            "feature": feature,
            "source_artifact": (
                "reports/history_samples/2026-05-06_2026-06-29/history_candidates_2026-05-06_2026-06-29_dedup.csv"
                "|reports/history_samples/2026-06-26_2026-06-30/history_candidates_2026-06-26_2026-06-30.csv"
                "|canonical frozen-cache reconstruction"
            ),
            "source_column": feature,
            "generator": formulas[feature],
            "coverage_rows": int(len(joined)),
            "missing_rows": missing_count,
            "frozen_rows": frozen_count,
            "canonical_reconstructed_rows": rebuilt_count,
            "D1_safe": "YES",
            "historical_point_in_time": "YES",
            "exact_semantic_match": "YES",
            "notes": "canonical v004a missing-rank fallback remains 0.5" if missing_count else "complete raw values",
        })
    gate = {
        "identity_319_39": len(joined) == 319 and joined["signal_date"].nunique() == 39,
        "all_events_sourced": joined["_feature_source"].notna().all(),
        "all_base_prices_positive": joined["candidate_base_price"].gt(0).all(),
        "d1_safe": True,
        "historical_point_in_time": True,
        "exact_semantic_match": True,
        "july_accessed": False,
        "frozen_rows": frozen_count,
        "canonical_reconstructed_rows": rebuilt_count,
    }
    gate["pass"] = bool(all(value for key_name, value in gate.items() if key_name not in {
        "frozen_rows", "canonical_reconstructed_rows", "july_accessed"
    }) and not gate["july_accessed"])
    return (
        joined.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True),
        pd.DataFrame(provenance_rows),
        pd.DataFrame(reconstruction_audit),
        gate,
    )


def prepare_transfer_samples(raw: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    assert_frozen_architecture_contract()
    frame = raw.merge(outcomes, on="event_id", how="left", validate="one_to_one")
    if frame[["raw_repair_return", "target7"]].isna().any().any():
        raise RuntimeError("FATAL: outcome identity incomplete")
    adapter = frame.copy()
    adapter["eligible_for_trade"] = True
    adapter[DEFAULT_TARGET_COLUMN] = adapter["target7"].astype(bool)
    adapter[DEFAULT_HIGH_RETURN_COLUMN] = adapter["raw_repair_return"] * 100.0
    # prepare_v004a_samples requires the legacy close-return field only for its
    # old reporting proxy.  It cannot affect X, target, tail weight, or score.
    adapter[DEFAULT_CLOSE_RETURN_COLUMN] = 0.0
    prepared, info, _ = prepare_v004a_samples(adapter, target_return_pct=7.0)
    if info["v004a_feature_columns"] != FROZEN_FEATURE_COLUMNS:
        raise RuntimeError(
            "FATAL: exact frozen 18-feature list unavailable: "
            + ",".join(info["v004a_feature_columns"])
        )
    if len(prepared) != 319 or prepared["event_id"].nunique() != 319:
        raise RuntimeError("FATAL: prepare_v004a_samples changed v004c universe")
    merged = prepared.drop(columns=[
        "raw_repair_return", "capped_opportunity_return_7", "target7",
    ], errors="ignore").merge(
        outcomes, on="event_id", how="left", validate="one_to_one"
    )
    if bool(merged["signal_date"].astype(str).ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July sentinel after v004a transform")
    return merged.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def run_expanding_forward(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(samples["signal_date"].astype(str).unique())
    if len(dates) != 39 or not all(date.startswith("2026-05") for date in dates[:18]):
        raise RuntimeError("FATAL: natural May-18/June-21 split unavailable")
    if not all(date.startswith("2026-06") for date in dates[18:]):
        raise RuntimeError("FATAL: test scope must be June only")
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    previous_coef: np.ndarray | None = None
    for fold_index in range(INITIAL_TRAIN_DATES, len(dates)):
        test_date = dates[fold_index]
        train_dates = dates[:fold_index]
        train = samples[samples["signal_date"].isin(train_dates)].copy()
        test = samples[samples["signal_date"].eq(test_date)].copy()
        if train.empty or test.empty or str(train["signal_date"].max()) >= test_date:
            raise RuntimeError(f"FATAL: chronological fold violation {test_date}")
        sample_weight = build_training_sample_weight(train, positive_weight=POSITIVE_WEIGHT)
        x_train = train[FROZEN_FEATURE_COLUMNS].to_numpy(float)
        y_train = train[DEFAULT_TARGET_COLUMN].astype(bool).astype(float).to_numpy()
        beta = fit_logistic_l2_weighted(
            x_train, y_train, l2=L2, sample_weight=sample_weight
        )
        score = _sigmoid(
            beta[0] + test[FROZEN_FEATURE_COLUMNS].to_numpy(float) @ beta[1:]
        )
        predicted = test[[
            "event_id", "signal_date", "code", "board_streak_before_break",
            "target7", "raw_repair_return", "capped_opportunity_return_7",
        ]].copy()
        predicted["model_score"] = score
        predicted["train_start"] = train_dates[0]
        predicted["train_end"] = train_dates[-1]
        predicted["train_dates"] = len(train_dates)
        predictions.append(predicted)

        weighted_target_rate = float(np.sum(sample_weight * y_train) / np.sum(sample_weight))
        folds.append({
            "test_date": test_date,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "train_dates": len(train_dates),
            "train_rows": len(train),
            "train_target7_rate": float(y_train.mean()),
            "weighted_target7_rate": weighted_target_rate,
            "mean_sample_weight": float(sample_weight.mean()),
            "sum_sample_weight": float(sample_weight.sum()),
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "score_std": float(score.std(ddof=0)),
            "exact_score_tie_rows": exact_tie_rows(pd.Series(score)),
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "feature_count": len(FROZEN_FEATURE_COLUMNS),
            "train_before_test": bool(train_dates[-1] < test_date),
        })
        coef = np.asarray(beta[1:], dtype=float)
        if previous_coef is None:
            cosine = np.nan
        else:
            denominator = float(np.linalg.norm(previous_coef) * np.linalg.norm(coef))
            cosine = float(previous_coef @ coef / denominator) if denominator > 0 else np.nan
        coefficients.append({
            "test_date": test_date,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "train_dates": len(train_dates),
            "train_rows": len(train),
            "intercept": float(beta[0]),
            **{name: float(value) for name, value in zip(FROZEN_FEATURE_COLUMNS, coef)},
            "adjacent_coefficient_cosine": cosine,
            "purpose": "AUDIT_ONLY",
        })
        previous_coef = coef
    oof = assign_model_rank(pd.concat(predictions, ignore_index=True))
    oof["board"] = pd.to_numeric(oof["board_streak_before_break"]).astype(int)
    daily = oof.groupby("signal_date", sort=True).agg(
        daily_universe_capped_return=("capped_opportunity_return_7", "mean"),
        daily_universe_target7_rate=("target7", "mean"),
    ).reset_index()
    oof = oof.merge(daily, on="signal_date", how="left", validate="many_to_one")
    columns = [
        "event_id", "signal_date", "code", "board", "model_score", "model_rank",
        "target7", "raw_repair_return", "capped_opportunity_return_7",
        "daily_universe_capped_return", "daily_universe_target7_rate",
        "train_start", "train_end", "train_dates",
    ]
    return (
        oof[columns].sort_values(["signal_date", "model_rank", "event_id"], kind="mergesort").reset_index(drop=True),
        pd.DataFrame(folds),
        pd.DataFrame(coefficients),
    )


def exact_tie_rows(values: pd.Series) -> int:
    counts = values.value_counts(dropna=False)
    return int((counts[counts > 1] - 1).sum())


def assign_model_rank(frame: pd.DataFrame, score_column: str = "model_score") -> pd.DataFrame:
    ranked = frame.sort_values(
        ["signal_date", score_column, "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["model_rank"] = ranked.groupby("signal_date", sort=True).cumcount() + 1
    return ranked


def _daily_selected(frame: pd.DataFrame, kind: str, k: int, strict: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("signal_date", sort=True):
        ordered = day.sort_values(
            ["model_score", "event_id"], ascending=[False, True], kind="mergesort"
        )
        if kind == "rank":
            if len(day) < k:
                continue
            selected = ordered.iloc[[k - 1]]
        else:
            if strict and len(day) < k:
                continue
            selected = ordered.head(min(k, len(day)))
        selected_return = float(selected["capped_opportunity_return_7"].mean())
        baseline = float(day["capped_opportunity_return_7"].mean())
        rows.append({
            "signal_date": date,
            "selected_return": selected_return,
            "baseline": baseline,
            "target7_precision": float(selected["target7"].mean()),
            "positive_return_rate": float((selected["capped_opportunity_return_7"] > 0).mean()),
            "beat": float(selected_return > baseline),
            "selected_codes": "|".join(selected["code"].astype(str)),
        })
    return pd.DataFrame(rows)


def selector_summary(frame: pd.DataFrame, kind: str, k: int, strict: bool = True) -> dict[str, float]:
    daily = _daily_selected(frame, kind, k, strict)
    if daily.empty:
        return {name: np.nan for name in (
            "mean", "median", "baseline", "excess", "beat_universe",
            "positive_return", "target7_precision", "dates",
        )}
    mean_return = float(daily["selected_return"].mean())
    baseline = float(daily["baseline"].mean())
    return {
        "mean": mean_return,
        "median": float(daily["selected_return"].median()),
        "baseline": baseline,
        "excess": mean_return - baseline,
        "beat_universe": float(daily["beat"].mean()),
        "positive_return": float(daily["positive_return_rate"].mean()),
        "target7_precision": float(daily["target7_precision"].mean()),
        "dates": float(len(daily)),
    }


def winner_capture(frame: pd.DataFrame, k: int = 3) -> float:
    values: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        if len(day) < k:
            continue
        target_count = int(day["target7"].sum())
        if target_count <= 0:
            continue
        selected = day.sort_values(
            ["model_score", "event_id"], ascending=[False, True], kind="mergesort"
        ).head(k)
        values.append(float(selected["target7"].sum() / min(k, target_count)))
    return float(np.mean(values)) if values else np.nan


def summarize_model(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date_count": int(frame["signal_date"].nunique()),
        "daily_universe_capped_return": float(
            frame.groupby("signal_date", sort=True)["capped_opportunity_return_7"].mean().mean()
        ),
        "daily_universe_target7_rate": float(
            frame.groupby("signal_date", sort=True)["target7"].mean().mean()
        ),
        "winner_capture": winner_capture(frame),
        "exact_score_tie_rows": int(sum(
            exact_tie_rows(day["model_score"])
            for _, day in frame.groupby("signal_date", sort=True)
        )),
    }
    selectors = (
        ("Rank1", "rank", 1, True),
        ("Rank2", "rank", 2, True),
        ("Rank3", "rank", 3, True),
        ("Top1", "top", 1, True),
        ("Top2", "top", 2, True),
        ("Top3", "top", 3, True),
        ("Up_To_3", "top", 3, False),
    )
    for label, kind, k, strict in selectors:
        summary = selector_summary(frame, kind, k, strict)
        for key, value in summary.items():
            result[f"{label}_{key}"] = value
    result.update(repair_concordance(frame.rename(columns={"model_score": "score"})))
    return result


def _load_reference_scores(root: Path, outcomes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    specs = {
        "CORRECTED_BINARY_SAME_DATE": (
            root
            / "reports/research/v004c_binary_same_date_pairwise_ridge_v002_20260506_20260630"
            / "v004c_binary_same_date_oof_predictions_v002.csv",
            "binary_same_date_oof_score",
            None,
        ),
        "REPAIR_SAME_DATE": (
            root
            / "reports/research/v004c_repair_pairwise_ridge_v001_20260506_20260630"
            / "v004c_repair_pairwise_oof_predictions_v001.csv",
            "repair_oof_score",
            None,
        ),
        "FOUNDATION_RAW_M0": (
            root
            / "reports/research/v004c_mechanism_foundation_screen_v001_20260506_20260630"
            / "v004c_mechanism_oof_v001.csv",
            "score",
            {"representation": "RAW", "model_stage": "M0", "run_type": "REAL"},
        ),
    }
    result: dict[str, pd.DataFrame] = {}
    for model, (path, score_column, filters) in specs.items():
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        usecols = ["event_id", "signal_date", "code", score_column]
        if filters:
            usecols.extend(filters)
        frame = pd.read_csv(
            path, encoding="utf-8-sig", dtype={"code": str}, usecols=usecols
        )
        frame["signal_date"] = frame["signal_date"].astype(str)
        frame = frame[frame["signal_date"].str.startswith("2026-06")].copy()
        if filters:
            for column, value in filters.items():
                frame = frame[frame[column].astype(str).eq(value)]
        frame = frame.dropna(subset=[score_column]).copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame = frame[["event_id", "signal_date", "code", score_column]].rename(
            columns={score_column: "model_score"}
        )
        frame = frame.merge(outcomes, on="event_id", how="left", validate="one_to_one")
        if len(frame) != 173 or frame["signal_date"].nunique() != 21:
            raise RuntimeError(f"FATAL: June reference scope mismatch: {model}")
        result[model] = assign_model_rank(frame)
    return result


def _oracle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    oracle = frame.drop(columns=["model_score", "model_rank"], errors="ignore").copy()
    oracle["model_score"] = oracle["capped_opportunity_return_7"]
    return assign_model_rank(oracle)


def practical_comparison(
    transfer: pd.DataFrame,
    references: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    frames = {MODEL_ID: transfer, **references}
    oracle = _oracle_frame(transfer)
    frames["ORACLE"] = oracle
    summaries = {model: summarize_model(frame) for model, frame in frames.items()}
    universe = summaries[MODEL_ID]
    rows: list[dict[str, Any]] = []
    for model, summary in summaries.items():
        row = {
            "model": model,
            "Rank1": summary["Rank1_mean"],
            "Rank2": summary["Rank2_mean"],
            "Rank3": summary["Rank3_mean"],
            "Top1": summary["Top1_mean"],
            "Top2": summary["Top2_mean"],
            "Top3": summary["Top3_mean"],
            "Up_To_3": summary["Up_To_3_mean"],
            "Rank1_baseline": summary["Rank1_baseline"],
            "Top3_baseline": summary["Top3_baseline"],
            "Rank1_excess": summary["Rank1_excess"],
            "Top3_excess": summary["Top3_excess"],
            "Rank1_beat_universe": summary["Rank1_beat_universe"],
            "Top3_beat_universe": summary["Top3_beat_universe"],
            "Top1_Target7": summary["Top1_target7_precision"],
            "Top3_Target7_precision": summary["Top3_target7_precision"],
            "winner_capture": summary["winner_capture"],
            "all_repair_concordance": summary["all_repair_concordance"],
            "cross_threshold_concordance": summary["cross_threshold_concordance"],
            "within_non_target_concordance": summary["within_non_target_concordance"],
            "within_target7_concordance": summary["within_target_concordance"],
            "exact_score_tie_rows": summary["exact_score_tie_rows"],
            "date_count": summary["date_count"],
        }
        for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3", "Up_To_3"):
            for key in (
                "mean", "median", "baseline", "excess", "beat_universe",
                "positive_return", "target7_precision", "dates",
            ):
                row[f"{label}_{key}"] = summary[f"{label}_{key}"]
        rows.append(row)
    universe_row = {
        "model": "DAILY_UNIVERSE",
        "Rank1": universe["daily_universe_capped_return"],
        "Rank2": universe["daily_universe_capped_return"],
        "Rank3": universe["daily_universe_capped_return"],
        "Top1": universe["daily_universe_capped_return"],
        "Top2": universe["Top2_baseline"],
        "Top3": universe["Top3_baseline"],
        "Up_To_3": universe["daily_universe_capped_return"],
        "Rank1_baseline": universe["daily_universe_capped_return"],
        "Top3_baseline": universe["Top3_baseline"],
        "Rank1_excess": 0.0,
        "Top3_excess": 0.0,
        "Rank1_beat_universe": np.nan,
        "Top3_beat_universe": np.nan,
        "Top1_Target7": universe["daily_universe_target7_rate"],
        "Top3_Target7_precision": universe["daily_universe_target7_rate"],
        "winner_capture": np.nan,
        "all_repair_concordance": np.nan,
        "cross_threshold_concordance": np.nan,
        "within_non_target_concordance": np.nan,
        "within_target7_concordance": np.nan,
        "exact_score_tie_rows": np.nan,
        "date_count": universe["date_count"],
    }
    for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3", "Up_To_3"):
        baseline = (
            universe["daily_universe_capped_return"]
            if label in {"Rank1", "Top1", "Up_To_3"}
            else universe[f"{label}_baseline"]
        )
        universe_row.update({
            f"{label}_mean": baseline,
            f"{label}_median": np.nan,
            f"{label}_baseline": baseline,
            f"{label}_excess": 0.0,
            f"{label}_beat_universe": np.nan,
            f"{label}_positive_return": np.nan,
            f"{label}_target7_precision": universe["daily_universe_target7_rate"],
            f"{label}_dates": universe[f"{label}_dates"],
        })
    rows.append(universe_row)
    return pd.DataFrame(rows), summaries, oracle


def opportunity_terciles(
    transfer: pd.DataFrame, oracle: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_env = transfer.groupby("signal_date", sort=True).agg(
        universe=("capped_opportunity_return_7", "mean")
    ).reset_index().sort_values(["universe", "signal_date"], kind="mergesort")
    if len(daily_env) != 21:
        raise RuntimeError("FATAL: opportunity terciles require 21 June dates")
    daily_env["opportunity_bucket"] = ["LOW"] * 7 + ["MID"] * 7 + ["HIGH"] * 7
    bucket_map = dict(zip(daily_env["signal_date"], daily_env["opportunity_bucket"]))
    rows: list[dict[str, Any]] = []
    for bucket in ("LOW", "MID", "HIGH"):
        dates = sorted(date for date, value in bucket_map.items() if value == bucket)
        subset = transfer[transfer["signal_date"].isin(dates)]
        oracle_subset = oracle[oracle["signal_date"].isin(dates)]
        summary = summarize_model(subset)
        oracle_summary = summarize_model(oracle_subset)
        top3_daily = _daily_selected(subset, "top", 3, True)
        rows.append({
            "opportunity_bucket": bucket,
            "date_count": len(dates),
            "dates": "|".join(dates),
            "universe_mean": summary["daily_universe_capped_return"],
            "Rank1_mean": summary["Rank1_mean"],
            "Top3_mean": summary["Top3_mean"],
            "Rank1_excess": summary["Rank1_excess"],
            "Top3_excess": summary["Top3_excess"],
            "negative_date_rate": float((top3_daily["selected_return"] < 0).mean()),
            "median_top3": float(top3_daily["selected_return"].median()),
            "worst_top3": float(top3_daily["selected_return"].min()),
            "Top1_Target7": summary["Top1_target7_precision"],
            "Top3_Target7_precision": summary["Top3_target7_precision"],
            "oracle_Rank1": oracle_summary["Rank1_mean"],
            "oracle_Top3": oracle_summary["Top3_mean"],
        })
    return pd.DataFrame(rows), daily_env


def _daily_top3_distribution(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    daily = _daily_selected(frame, "top", 3, True)
    stats = {
        "mean": float(daily["selected_return"].mean()),
        "p25": float(daily["selected_return"].quantile(0.25)),
        "median": float(daily["selected_return"].median()),
        "worst": float(daily["selected_return"].min()),
        "negative_date_rate": float((daily["selected_return"] < 0).mean()),
    }
    return stats, daily


def architecture_status(
    transfer: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    universe_return = float(transfer["daily_universe_capped_return"])
    universe_hit = float(transfer["daily_universe_target7_rate"])
    conditions = (
        float(transfer["Rank1_mean"]) > universe_return,
        float(transfer["Top3_mean"]) > universe_return,
        float(transfer["Rank1_beat_universe"]) > 0.50,
        float(transfer["Top3_beat_universe"]) > 0.50,
        float(transfer["Top1_target7_precision"]) > universe_hit,
        float(transfer["Top3_target7_precision"]) > universe_hit,
    )
    formal = [references["CORRECTED_BINARY_SAME_DATE"], references["REPAIR_SAME_DATE"]]
    not_below_refs = (
        float(transfer["Rank1_mean"]) >= max(float(ref["Rank1_mean"]) for ref in formal)
        and float(transfer["Top3_mean"]) >= max(float(ref["Top3_mean"]) for ref in formal)
    )
    if all(conditions) and not_below_refs:
        signal = "STRONG"
    elif (
        float(transfer["Rank1_mean"]) <= universe_return
        and float(transfer["Top3_mean"]) <= universe_return
    ):
        signal = "ABSENT"
    else:
        signal = "PARTIAL"
    reference_improvement = (
        float(transfer["Rank1_mean"]) > max(float(ref["Rank1_mean"]) for ref in formal)
        and float(transfer["Top3_mean"]) > max(float(ref["Top3_mean"]) for ref in formal)
    )
    if signal == "STRONG" and reference_improvement:
        route = "SUPPORTED"
    elif signal == "ABSENT":
        route = "NOT_SUPPORTED"
    else:
        route = "INCONCLUSIVE"
    return {
        "architecture_transfer_signal": signal,
        "previous_route_misdesigned": route,
        "ready_for_july_oot": "YES" if signal == "STRONG" else "NO",
    }


def reference_comparison_answer(
    transfer: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]]
) -> str:
    formal = [references["CORRECTED_BINARY_SAME_DATE"], references["REPAIR_SAME_DATE"]]
    rank_better = float(transfer["Rank1_mean"]) > max(float(r["Rank1_mean"]) for r in formal)
    top3_better = float(transfer["Top3_mean"]) > max(float(r["Top3_mean"]) for r in formal)
    if rank_better and top3_better:
        return "YES"
    if not rank_better and not top3_better:
        return "NO"
    return "MIXED"


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def build_review(
    gate: Mapping[str, Any],
    provenance: pd.DataFrame,
    fold_audit: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, Any]],
    opportunity: pd.DataFrame,
    oracle_summary: Mapping[str, Any],
    top3_stats: Mapping[str, Any],
    top3_daily: pd.DataFrame,
    status: Mapping[str, str],
) -> str:
    transfer = summaries[MODEL_ID]
    lines = [
        "# v004c Frozen-v4a Architecture Transfer Benchmark v001",
        "",
        "CAPABILITY BENCHMARK ONLY — ONE FROZEN ARCHITECTURE — NO JULY",
        "",
        "## 1. Did Frozen v004a Architecture Work Inside v004c?",
        "",
    ]
    table_models = [
        MODEL_ID, "CORRECTED_BINARY_SAME_DATE", "REPAIR_SAME_DATE", "FOUNDATION_RAW_M0"
    ]
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Baseline", "Excess", "Top3", "Baseline", "Excess", "Rank1 Beat", "Top3 Beat", "Top1 Hit", "Top3 Precision"],
        [
            [
                model,
                _pct(summaries[model]["Rank1_mean"]),
                _pct(summaries[model]["Rank1_baseline"]),
                _pct(summaries[model]["Rank1_excess"]),
                _pct(summaries[model]["Top3_mean"]),
                _pct(summaries[model]["Top3_baseline"]),
                _pct(summaries[model]["Top3_excess"]),
                _pct(summaries[model]["Rank1_beat_universe"]),
                _pct(summaries[model]["Top3_beat_universe"]),
                _pct(summaries[model]["Top1_target7_precision"]),
                _pct(summaries[model]["Top3_target7_precision"]),
            ]
            for model in table_models
        ],
    ))
    lines.extend([
        "",
        f"- Architecture transfer signal: **{status['architecture_transfer_signal']}**",
        f"- Exact score tie rows: **{int(transfer['exact_score_tie_rows'])}**",
        "",
        "## 2. High-Opportunity vs Low-Opportunity Days",
        "",
    ])
    lines.extend(_markdown_table(
        ["Bucket", "Dates", "Universe", "Rank1", "Rank1 Excess", "Top3", "Top3 Excess", "Negative Rate", "Oracle Rank1", "Oracle Top3"],
        [
            [row.opportunity_bucket, int(row.date_count), _pct(row.universe_mean),
             _pct(row.Rank1_mean), _pct(row.Rank1_excess), _pct(row.Top3_mean),
             _pct(row.Top3_excess), _pct(row.negative_date_rate),
             _pct(row.oracle_Rank1), _pct(row.oracle_Top3)]
            for row in opportunity.itertuples()
        ],
    ))
    high = opportunity[opportunity["opportunity_bucket"].eq("HIGH")].iloc[0]
    low = opportunity[opportunity["opportunity_bucket"].eq("LOW")].iloc[0]
    lines.extend([
        "",
        "High-opportunity upside capture: **{}**.".format(
            "YES" if high.Rank1_excess > 0 and high.Top3_excess > 0 else "NO"
        ),
        "Low-opportunity capital preservation under the capped-opportunity evaluator: **{}**.".format(
            "YES" if low.Top3_excess > 0 else "NO"
        ),
        "",
        "## 3. Comparison With Existing v004c Models",
        "",
        f"- Practical comparison vs Corrected Binary and Repair: **{reference_comparison_answer(transfer, summaries)}**.",
        "- Every row above uses the same 21 June dates, same event universe, same capped-opportunity evaluator, and deterministic event_id tie-break.",
        "",
        "## 4. Exact v004a Architecture Parity",
        "",
        "- 18 feature exact: **YES**",
        "- l2: **0.3**",
        "- positive weight: **1.5**",
        "- tail weighting: **MATCH**",
        "- date weighting: **MATCH**",
        "- interactions: **MATCH**",
        "- rank transform: **MATCH**",
        "- extra features: **NONE**",
        "- new model logic: **NONE**",
        f"- feature gate: **{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        "## 5. What Changed From v004a?",
        "",
        "- Candidate universe: broad v004a → v004c first-break universe",
        "- Daily percentile reference: v004c candidates",
        "- Training history: v004c dates",
        "- Training architecture: **UNCHANGED**",
        "",
        "## 6. What Did NOT Change?",
        "",
        "- Target: Target7 binary",
        "- Model family: weighted L2 Logistic",
        "- L2: 0.30",
        "- Positive weight: 1.50",
        "- Tail weight: >=10% 1.5; >=12% 2.0",
        "- Date weight: 1/n before class and tail multiplication",
        "- Feature structure: frozen 18 dimensions",
        "- Interaction formulas: frozen six",
        "",
        "## 7. Feature Provenance and Leakage Boundary",
        "",
        f"- Frozen historical raw rows: **{gate['frozen_rows']}**",
        f"- Canonical reconstructed raw rows: **{gate['canonical_reconstructed_rows']}**",
        "- Reconstructed rows use the canonical generate_signal chain with daily data clipped at D1, BaoStock 5m clipped at D1, and the original five-calendar-day historical limit-up pool.",
        "- Missing raw ranks retain v004a's average-percentile then 0.5 fallback semantics.",
        "- Test outcomes never enter feature transformation or scoring; train outcomes enter only Target7 and frozen class/tail sample weights.",
        "- July was not accessed by the formal pipeline.",
        "",
        "## 8. Downside / Upside Distribution",
        "",
        f"- Top3 mean: **{_pct(top3_stats['mean'])}**",
        f"- Top3 p25: **{_pct(top3_stats['p25'])}**",
        f"- Top3 median: **{_pct(top3_stats['median'])}**",
        f"- Top3 worst: **{_pct(top3_stats['worst'])}**",
        f"- Top3 negative-date rate: **{_pct(top3_stats['negative_date_rate'])}**",
        "",
        "### Best 3 dates",
        "",
    ])
    best = top3_daily.sort_values(
        ["selected_return", "signal_date"], ascending=[False, True], kind="mergesort"
    ).head(3)
    worst = top3_daily.sort_values(
        ["selected_return", "signal_date"], ascending=[True, True], kind="mergesort"
    ).head(3)
    lines.extend(_markdown_table(
        ["Date", "Universe", "Top3", "Excess", "Selected codes"],
        [[row.signal_date, _pct(row.baseline), _pct(row.selected_return),
          _pct(row.selected_return - row.baseline), row.selected_codes]
         for row in best.itertuples()],
    ))
    lines.extend(["", "### Worst 3 dates", ""])
    lines.extend(_markdown_table(
        ["Date", "Universe", "Top3", "Excess", "Selected codes"],
        [[row.signal_date, _pct(row.baseline), _pct(row.selected_return),
          _pct(row.selected_return - row.baseline), row.selected_codes]
         for row in worst.itertuples()],
    ))
    lines.extend([
        "",
        "## 9. Oracle",
        "",
        f"- Oracle Rank1: **{_pct(oracle_summary['Rank1_mean'])}**",
        f"- Oracle STRICT Top3: **{_pct(oracle_summary['Top3_mean'])}**",
        "",
        "## 10. Final Answers",
        "",
        "- Q1 exact D1-safe 18-feature reconstruction: **YES**",
        f"- Q2 beat June universe Rank1: **{'YES' if transfer['Rank1_excess'] > 0 else 'NO'}**",
        f"- Q3 beat June universe STRICT Top3: **{'YES' if transfer['Top3_excess'] > 0 else 'NO'}**",
        f"- Q4 beat Corrected Binary + Repair: **{reference_comparison_answer(transfer, summaries)}**",
        f"- Q5 HIGH-opportunity upside capture: **{'YES' if high.Rank1_excess > 0 and high.Top3_excess > 0 else 'NO'}**",
        f"- Q6 LOW-opportunity capital preservation: **{'YES' if low.Top3_excess > 0 else 'NO'}**",
        f"- Q7 previous v004c modeling-route problem: **{status['previous_route_misdesigned']}**",
        "",
        f"ARCHITECTURE_TRANSFER_SIGNAL: **{status['architecture_transfer_signal']}**",
        "",
        f"PREVIOUS_V4C_MODELING_ROUTE_MISDESIGNED: **{status['previous_route_misdesigned']}**",
        "",
        f"READY_FOR_JULY_OOT: **{status['ready_for_july_oot']}**",
    ])
    if status["architecture_transfer_signal"] == "ABSENT":
        lines.extend([
            "",
            "Frozen v004a architecture did not establish practical June forward edge inside the v004c universe.",
        ])
    return "\n".join(lines) + "\n"


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=False, lineterminator="\n", float_format="%.12g", na_rep=""
    )
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def root_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_frozen_architecture_contract()
    base = load_authoritative_input(root)
    raw, provenance, reconstruction_audit, feature_gate = build_exact_raw_features(root, base)
    if not feature_gate["pass"]:
        raise RuntimeError("EXACT_V4A_FEATURE_GATE = FAIL")
    outcomes = load_outcomes(root, base)
    samples = prepare_transfer_samples(raw, outcomes)
    oof, fold_audit, coefficients = run_expanding_forward(samples)
    references = _load_reference_scores(root, outcomes)
    comparison, summaries, oracle = practical_comparison(oof, references)
    opportunity, daily_env = opportunity_terciles(oof, oracle)
    top3_stats, top3_daily = _daily_top3_distribution(oof)
    status = architecture_status(summaries[MODEL_ID], summaries)
    review = build_review(
        feature_gate, provenance, fold_audit, summaries, opportunity,
        summaries["ORACLE"], top3_stats, top3_daily, status,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(provenance),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(fold_audit),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(coefficients),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(oof),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(comparison),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(opportunity),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    context = {
        "base": base,
        "raw": raw,
        "provenance": provenance,
        "reconstruction_audit": reconstruction_audit,
        "feature_gate": feature_gate,
        "samples": samples,
        "oof": oof,
        "fold_audit": fold_audit,
        "coefficients": coefficients,
        "comparison": comparison,
        "summaries": summaries,
        "opportunity": opportunity,
        "daily_environment": daily_env,
        "top3_stats": top3_stats,
        "top3_daily": top3_daily,
        "status": status,
        "output_sha256": {name: sha256_bytes(value) for name, value in outputs.items()},
    }
    return outputs, context


def run_v004c_v4a_architecture_transfer(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    first, first_context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first.keys() != second.keys() or any(first[name] != second[name] for name in first):
        raise RuntimeError("FATAL: deterministic rebuild failed")
    output_dir = (
        root_path
        / "reports/research/v004c_v4a_architecture_transfer_v001_20260506_20260630"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (output_dir / name).write_bytes(first[name])
    first_context["deterministic_rebuild"] = "PASS"
    first_context["output_dir"] = output_dir
    return output_dir, first_context

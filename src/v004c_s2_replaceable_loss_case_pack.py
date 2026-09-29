"""Prepare the frozen-S2 replaceable-loss case pack.

This module is data preparation only.  It never fits a model, searches a
feature/window/threshold, or reads an August signal-date outcome.  The case
identity and S2 scores come from the already frozen S2 failure-attribution
audit.  The 53-field family is the existing pairwise-v1 D1-safe contract;
archived values are preserved and only archive gaps are filled with the same
repository formulas and local D1-safe caches.
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .v004c_minute_features import compute_minute_features, intraday_vwap
from .v004c_model_table import compute_recent_7d_features
from .v004c_pairwise_feature_contract import CONTRACT_FEATURE_NAMES, FEATURE_CONTRACT
from .v004c_s2_winner_vs_loss_failure_attribution import (
    RAW_PARENT_COLUMNS,
    load_frozen_s2,
)


TASK_NAME = "v004c_s2_replaceable_loss_case_pack_v001"
OUTPUT_DIRNAME = "v004c_s2_replaceable_loss_case_pack_v001_20260506_20260729"
MAX_SIGNAL_DATE = "2026-07-29"
AUGUST_ASOF = "2026-08-01"
OUTPUT_FILENAMES = (
    "v004c_s2_case_pairs_v001.csv",
    "v004c_s2_case_daily_history_v001.csv",
    "v004c_s2_case_d1_5min_v001.csv",
    "v004c_s2_case_path_descriptors_v001.csv",
    "v004c_s2_case_7f_comparison_v001.csv",
    "v004c_s2_case_53f_comparison_v001.csv",
    "v004c_s2_case_feature_lineage_v001.csv",
    "v004c_s2_case_pair_comparison_wide_v001.csv",
    "v004c_s2_case_pack_integrity_v001.csv",
    "v004c_s2_case_pack_review_v001.md",
)
PAIR_TYPES = (
    "HIGHEST_RANKED_OUTSIDE_TARGET7",
    "NEAREST_SCORE_OUTSIDE_TARGET7",
)
PATH_DESCRIPTOR_COLUMNS = (
    "d1_open",
    "d1_high",
    "d1_low",
    "d1_close",
    "d1_vwap_full_day",
    "time_of_day_high",
    "time_of_day_low",
    "close_position_in_day_range",
    "low_to_close_recovery_pct",
    "high_to_close_drawdown_pct",
    "morning_return",
    "afternoon_return",
    "last_30m_return",
    "last_60m_return",
    "morning_amount_share",
    "afternoon_amount_share",
    "last_30m_amount_share",
    "bars_close_above_intraday_vwap_share",
    "bars_after_daily_low",
    "return_from_daily_low_to_close",
)

PAIRWISE_AUTHORITY = (
    "reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630/"
    "v004c_pairwise_v1_input_table_v002.csv"
)
MODEL_TABLE_AUTHORITY = (
    "reports/research/v004c_model_table_v001_20260601_20260729/"
    "v004c_model_table_v001.csv"
)
D1_SNAPSHOT_AUTHORITY = (
    "reports/research/v004c_d1_dataset_v001_20260601_20260729/"
    "v004c_d1_snapshot_v001.csv"
)
HARD_LOSS_AUTHORITY = (
    "reports/research/v004c_s2_winner_vs_loss_failure_attribution_v001_20260506_20260731/"
    "v004c_s2_hard_false_positive_loss_v001.csv"
)
REPLACEABLE_AUTHORITY = (
    "reports/research/v004c_s2_winner_vs_loss_failure_attribution_v001_20260506_20260731/"
    "v004c_s2_replaceable_loss_audit_v001.csv"
)


def assert_contract() -> None:
    if len(CONTRACT_FEATURE_NAMES) != 53 or len(set(CONTRACT_FEATURE_NAMES)) != 53:
        raise RuntimeError("FATAL: existing D1-safe feature contract is not exact 53")
    if len(RAW_PARENT_COLUMNS) != 7:
        raise RuntimeError("FATAL: frozen S2 7F parent contract changed")
    if MAX_SIGNAL_DATE >= AUGUST_ASOF:
        raise RuntimeError("FATAL: case-pack boundary reached August")
    if len(PATH_DESCRIPTOR_COLUMNS) != 20:
        raise RuntimeError("FATAL: path descriptors changed from preregistration")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _safe_div(a: Any, b: Any) -> float:
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError):
        return math.nan
    return af / bf if np.isfinite(af) and np.isfinite(bf) and bf != 0 else math.nan


def _event_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_event_id", f"{prefix}_code", f"{prefix}_board_group",
        f"{prefix}_s2_rank", f"{prefix}_s2_score",
        f"{prefix}_raw_repair_return", f"{prefix}_capped_return",
    ]


def build_case_pairs(scored: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the two fixed pair rules; de-duplicate when A and B are identical."""
    frame = scored[
        scored["signal_date"].le(MAX_SIGNAL_DATE)
        & scored["label_available_date"].lt(AUGUST_ASOF)
    ].copy()
    if frame["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal-date row reached case definition")

    hard = pd.read_csv(root / HARD_LOSS_AUTHORITY, encoding="utf-8-sig", dtype=str)
    hard = hard[
        hard["row_type"].eq("LOSS_DETAIL")
        & hard["loss_classification"].eq("HARD_FALSE_POSITIVE_LOSS")
        & hard["signal_date"].le(MAX_SIGNAL_DATE)
    ]
    authoritative_loss_ids = set(hard["event_id"].astype(str))

    rows: list[dict[str, Any]] = []
    loss_index = 0
    for signal_date, day in frame.groupby("signal_date", sort=True):
        outside = day[
            day["s2_rank"].gt(3) & day["target7"].astype(bool)
        ].sort_values(["s2_rank", "event_id"], kind="mergesort")
        if outside.empty:
            continue
        losses = day[
            day["s2_rank"].le(3)
            & day["loss"].astype(bool)
            & day["event_id"].astype(str).isin(authoritative_loss_ids)
        ].sort_values(["s2_rank", "event_id"], kind="mergesort")
        for _, loss in losses.iterrows():
            loss_index += 1
            loss_case_id = f"LOSSCASE_{loss_index:03d}"
            highest = outside.iloc[0]
            nearest = outside.assign(
                _distance=(outside["s2_score"] - float(loss["s2_score"])).abs()
            ).sort_values(["_distance", "s2_rank", "event_id"], kind="mergesort").iloc[0]
            selections = [("HIGHEST_RANKED_OUTSIDE_TARGET7", highest)]
            if str(nearest["event_id"]) != str(highest["event_id"]):
                selections.append(("NEAREST_SCORE_OUTSIDE_TARGET7", nearest))
            for selection_index, (pair_type, target) in enumerate(selections, 1):
                pair_suffix = "A" if selection_index == 1 else "B"
                row = {
                    "case_id": f"{loss_case_id}_{pair_suffix}",
                    "loss_case_id": loss_case_id,
                    "signal_date": str(signal_date),
                    "month": str(signal_date)[:7],
                    "pair_type": pair_type,
                    "also_nearest_score_outside_target7": int(
                        str(target["event_id"]) == str(nearest["event_id"])
                    ),
                    "outside_target7_count": int(len(outside)),
                    "loss_event_id": str(loss["event_id"]),
                    "loss_code": str(loss["code"]).zfill(6),
                    "loss_board_group": str(loss["board_group"]),
                    "loss_s2_rank": int(loss["s2_rank"]),
                    "loss_s2_score": float(loss["s2_score"]),
                    "loss_raw_repair_return": float(loss["raw_repair_return"]),
                    "loss_capped_return": float(loss["capped_return_7"]),
                    "loss_severe_loss": int(loss["severe_loss"]),
                    "loss_d0_date": str(loss["d0_date"]),
                    "loss_d1_date": str(loss["d1_date"]),
                    "target_event_id": str(target["event_id"]),
                    "target_code": str(target["code"]).zfill(6),
                    "target_board_group": str(target["board_group"]),
                    "target_s2_rank": int(target["s2_rank"]),
                    "target_s2_score": float(target["s2_score"]),
                    "target_raw_repair_return": float(target["raw_repair_return"]),
                    "target_capped_return": float(target["capped_return_7"]),
                    "target_d0_date": str(target["d0_date"]),
                    "target_d1_date": str(target["d1_date"]),
                }
                row["score_margin"] = row["target_s2_score"] - row["loss_s2_score"]
                row["rank_gap"] = row["target_s2_rank"] - row["loss_s2_rank"]
                rows.append(row)
    pairs = pd.DataFrame(rows).sort_values(
        ["signal_date", "loss_s2_rank", "loss_event_id", "case_id"], kind="mergesort"
    ).reset_index(drop=True)
    if pairs["case_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate case_id")
    if pairs.duplicated(["loss_event_id", "target_event_id"]).any():
        raise RuntimeError("FATAL: duplicate LOSS-Target7 pair")

    if set(pairs["loss_event_id"]) != set(hard["event_id"]):
        raise RuntimeError("FATAL: fixed case identities differ from archived hard-loss audit")
    replacement = pd.read_csv(root / REPLACEABLE_AUTHORITY, encoding="utf-8-sig")
    replacement = replacement[
        replacement["row_type"].eq("DAILY")
        & replacement["signal_date"].astype(str).le(MAX_SIGNAL_DATE)
    ]
    audit = {
        "pair_count": int(len(pairs)),
        "unique_loss_count": int(pairs["loss_event_id"].nunique()),
        "unique_target_count": int(pairs["target_event_id"].nunique()),
        "archived_hard_loss_count": int(len(hard)),
        "theoretical_one_to_one_replaceable_slots": int(
            pd.to_numeric(replacement["theoretically_replaceable_loss_slots"]).sum()
        ),
        "duplicate_pair_count": 0,
    }
    return pairs, audit


def _normalize_minute(frame: pd.DataFrame, code: str, date: str) -> pd.DataFrame:
    day = frame.copy()
    if "trade_date" not in day:
        day["trade_date"] = pd.to_datetime(day["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        day["trade_date"] = pd.to_datetime(day["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    day["datetime"] = pd.to_datetime(day["datetime"], errors="coerce")
    day = day[day["trade_date"].eq(date)].copy()
    day["code"] = str(code).zfill(6)
    day = day.sort_values("datetime", kind="mergesort").drop_duplicates("datetime", keep="last")
    core = ["open", "high", "low", "close", "volume", "amount"]
    for column in core:
        day[column] = pd.to_numeric(day[column], errors="coerce")
    if day[core + ["datetime"]].isna().any().any():
        raise RuntimeError(f"FATAL: invalid D1 5m bar {code}@{date}")
    return day.reset_index(drop=True)


def load_d1_minute(root: Path, code: str, date: str) -> tuple[pd.DataFrame, str, str]:
    """Prefer local BaoStock D1; use the existing Sina cache when BaoStock lacks D1."""
    attempts = (
        ("baostock_5m", root / "data/cache/baostock_5m" / f"{code}_5min.pkl"),
        ("sina_5m", root / "data/cache/minute_5m" / f"{code}_5min.pkl"),
    )
    for source, path in attempts:
        if not path.is_file():
            continue
        day = _normalize_minute(pd.read_pickle(path), code, date)
        if len(day) == 48:
            return day, source, path.relative_to(root).as_posix()
    raise RuntimeError(f"FATAL: no complete 48-bar D1 source for {code}@{date}")


def _normalize_daily(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily = daily.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)
    daily["code"] = str(code).zfill(6)
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover_rate"):
        if column not in daily:
            daily[column] = math.nan
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["ma5"] = daily["close"].rolling(5, min_periods=5).mean()
    daily["ma10"] = daily["close"].rolling(10, min_periods=10).mean()
    daily["ma20"] = daily["close"].rolling(20, min_periods=20).mean()
    daily["vwap_daily"] = np.where(
        daily["amount"].notna() & daily["volume"].gt(0),
        daily["amount"] / daily["volume"],
        np.nan,
    )
    rng = daily["high"] - daily["low"]
    daily["close_position"] = np.where(rng.ne(0), (daily["close"] - daily["low"]) / rng, np.nan)
    if "amount_ratio" not in daily:
        daily["amount_ratio"] = math.nan
    return daily


def load_daily_full(root: Path, code: str, d1_date: str) -> tuple[pd.DataFrame, str]:
    for folder in ("daily", "daily_unadjusted"):
        path = root / "data/cache" / folder / f"{code}_daily.pkl"
        if not path.is_file():
            continue
        daily = _normalize_daily(pd.read_pickle(path), code)
        daily = daily[daily["date"].le(d1_date)].copy()
        if d1_date in set(daily["date"]):
            return daily, path.relative_to(root).as_posix()
    raise RuntimeError(f"FATAL: no daily history covers {code}@{d1_date}")


def build_daily_history(pairs: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for pair in pairs.itertuples(index=False):
        for role in ("LOSS", "TARGET7"):
            prefix = "loss" if role == "LOSS" else "target"
            event_id = str(getattr(pair, f"{prefix}_event_id"))
            code = str(getattr(pair, f"{prefix}_code"))
            d0 = str(getattr(pair, f"{prefix}_d0_date"))
            d1 = str(getattr(pair, f"{prefix}_d1_date"))
            daily, source_path = load_daily_full(root, code, d1)
            cache[event_id] = daily
            indices = daily.index[daily["date"].eq(d0)].tolist()
            if len(indices) != 1:
                raise RuntimeError(f"FATAL: D0 absent/duplicate in daily history {event_id}")
            start_index = max(0, indices[0] - 10)
            end_indices = daily.index[daily["date"].eq(d1)].tolist()
            if len(end_indices) != 1 or end_indices[0] < indices[0]:
                raise RuntimeError(f"FATAL: D1 absent/before D0 in daily history {event_id}")
            history = daily.iloc[start_index : end_indices[0] + 1].copy()
            if int(history["date"].lt(d0).sum()) < 10:
                raise RuntimeError(f"FATAL: fewer than 10 trading days before D0 for {event_id}")
            for row in history.to_dict("records"):
                rows.append({
                    "case_id": pair.case_id,
                    "loss_case_id": pair.loss_case_id,
                    "role": role,
                    "event_id": event_id,
                    "code": code,
                    "signal_date": pair.signal_date,
                    "d0_date": d0,
                    "d1_date": d1,
                    "history_source_path": source_path,
                    **row,
                })
    frame = pd.DataFrame(rows)
    order = [
        "case_id", "loss_case_id", "role", "event_id", "code", "signal_date",
        "d0_date", "d1_date", "history_source_path", "date", "market", "open",
        "high", "low", "close", "volume", "amount", "turnover_rate", "vwap_daily",
        "ma5", "ma10", "ma20", "amount_ratio", "close_position", "pct_chg",
        "change", "amplitude", "source",
    ]
    for column in order:
        if column not in frame:
            frame[column] = math.nan
    return frame[order].sort_values(["case_id", "role", "date"], kind="mergesort"), cache


def _cumulative_vwap(day: pd.DataFrame) -> pd.Series:
    amount = pd.to_numeric(day["amount"], errors="coerce").cumsum()
    volume = pd.to_numeric(day["volume"], errors="coerce").cumsum().replace(0, np.nan)
    raw = amount / volume
    hand = amount / (volume * 100.0)
    close = pd.to_numeric(day["close"], errors="coerce")
    return raw.mask(raw.gt(close * 5.0), hand)


def describe_path(day: pd.DataFrame) -> dict[str, Any]:
    ordered = day.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    high_index = pd.to_numeric(ordered["high"]).idxmax()
    low_index = pd.to_numeric(ordered["low"]).idxmin()
    high = float(ordered.loc[high_index, "high"])
    low = float(ordered.loc[low_index, "low"])
    close = float(last["close"])
    open_ = float(first["open"])
    rng = high - low
    clock = ordered["datetime"].dt.time
    morning = ordered[clock < pd.Timestamp("13:00:00").time()]
    afternoon = ordered[clock >= pd.Timestamp("13:00:00").time()]
    last30 = ordered[clock > pd.Timestamp("14:30:00").time()]
    last60 = ordered[clock > pd.Timestamp("14:00:00").time()]
    amount = pd.to_numeric(ordered["amount"], errors="coerce")
    total_amount = float(amount.sum())
    cumulative_vwap = _cumulative_vwap(ordered)
    return {
        "d1_open": open_,
        "d1_high": high,
        "d1_low": low,
        "d1_close": close,
        "d1_vwap_full_day": float(intraday_vwap(ordered)),
        "time_of_day_high": ordered.loc[high_index, "datetime"].strftime("%H:%M:%S"),
        "time_of_day_low": ordered.loc[low_index, "datetime"].strftime("%H:%M:%S"),
        "close_position_in_day_range": (close - low) / rng if rng else math.nan,
        "low_to_close_recovery_pct": close / low - 1.0 if low else math.nan,
        "high_to_close_drawdown_pct": close / high - 1.0 if high else math.nan,
        "morning_return": float(morning.iloc[-1]["close"]) / open_ - 1.0,
        "afternoon_return": close / float(afternoon.iloc[0]["open"]) - 1.0,
        "last_30m_return": close / float(last30.iloc[0]["open"]) - 1.0,
        "last_60m_return": close / float(last60.iloc[0]["open"]) - 1.0,
        "morning_amount_share": float(pd.to_numeric(morning["amount"]).sum()) / total_amount,
        "afternoon_amount_share": float(pd.to_numeric(afternoon["amount"]).sum()) / total_amount,
        "last_30m_amount_share": float(pd.to_numeric(last30["amount"]).sum()) / total_amount,
        "bars_close_above_intraday_vwap_share": float(
            pd.to_numeric(ordered["close"]).gt(cumulative_vwap).mean()
        ),
        "bars_after_daily_low": int(len(ordered) - low_index - 1),
        "return_from_daily_low_to_close": close / low - 1.0 if low else math.nan,
    }


def build_minute_and_descriptors(
    pairs: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    minute_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for pair in pairs.itertuples(index=False):
        for role in ("LOSS", "TARGET7"):
            prefix = "loss" if role == "LOSS" else "target"
            event_id = str(getattr(pair, f"{prefix}_event_id"))
            code = str(getattr(pair, f"{prefix}_code"))
            d1 = str(getattr(pair, f"{prefix}_d1_date"))
            if event_id not in cache:
                day, source, source_path = load_d1_minute(root, code, d1)
                day = day.copy()
                day["minute_source_used"] = source
                day["minute_source_path"] = source_path
                cache[event_id] = day
            day = cache[event_id]
            for row in day.to_dict("records"):
                minute_rows.append({
                    "case_id": pair.case_id,
                    "loss_case_id": pair.loss_case_id,
                    "role": role,
                    "event_id": event_id,
                    "signal_date": pair.signal_date,
                    **row,
                })
            descriptor_rows.append({
                "case_id": pair.case_id,
                "loss_case_id": pair.loss_case_id,
                "role": role,
                "event_id": event_id,
                "code": code,
                "signal_date": pair.signal_date,
                "minute_source_used": str(day.iloc[0]["minute_source_used"]),
                "bar_count": int(len(day)),
                "latest_timestamp": day["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
                **describe_path(day),
            })
    minute = pd.DataFrame(minute_rows).sort_values(
        ["case_id", "role", "datetime"], kind="mergesort"
    ).reset_index(drop=True)
    descriptors = pd.DataFrame(descriptor_rows).sort_values(
        ["case_id", "role"], kind="mergesort"
    ).reset_index(drop=True)
    return minute, descriptors, cache


def _load_archived_53(root: Path) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], str]]:
    values: dict[str, dict[str, float]] = {}
    provenance: dict[tuple[str, str], str] = {}
    pairwise_path = root / PAIRWISE_AUTHORITY
    pairwise = pd.read_csv(pairwise_path, encoding="utf-8-sig", dtype={"event_id": str, "code": str})
    for row in pairwise.to_dict("records"):
        event_id = str(row["event_id"])
        values[event_id] = {}
        for feature in CONTRACT_FEATURE_NAMES:
            value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
            values[event_id][feature] = float(value) if pd.notna(value) else math.nan
            provenance[(event_id, feature)] = PAIRWISE_AUTHORITY

    model = pd.read_csv(root / MODEL_TABLE_AUTHORITY, encoding="utf-8-sig", dtype={"event_id": str})
    snapshot = pd.read_csv(root / D1_SNAPSHOT_AUTHORITY, encoding="utf-8-sig", dtype={"event_id": str})
    if model["event_id"].duplicated().any() or snapshot["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in current D1 authorities")
    model_by = model.set_index("event_id")
    snapshot_by = snapshot.set_index("event_id")
    for event_id in sorted(set(model_by.index) | set(snapshot_by.index)):
        if event_id in values:
            continue
        values[event_id] = {}
        for feature in CONTRACT_FEATURE_NAMES:
            selected = math.nan
            selected_path = ""
            if feature in model_by.columns and event_id in model_by.index:
                selected = pd.to_numeric(pd.Series([model_by.at[event_id, feature]]), errors="coerce").iloc[0]
                if pd.notna(selected):
                    selected_path = MODEL_TABLE_AUTHORITY
            if pd.isna(selected) and feature in snapshot_by.columns and event_id in snapshot_by.index:
                selected = pd.to_numeric(pd.Series([snapshot_by.at[event_id, feature]]), errors="coerce").iloc[0]
                if pd.notna(selected):
                    selected_path = D1_SNAPSHOT_AUTHORITY
            values[event_id][feature] = float(selected) if pd.notna(selected) else math.nan
            if selected_path:
                provenance[(event_id, feature)] = selected_path
    return values, provenance


def _limit_up_flags(daily: pd.DataFrame) -> np.ndarray:
    close = pd.to_numeric(daily["close"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(daily["high"], errors="coerce").to_numpy(float)
    flags = np.zeros(len(daily), dtype=bool)
    for index in range(1, len(daily)):
        prior = close[index - 1]
        limit = math.floor(prior * 1.10 * 100.0 + 0.5) / 100.0
        flags[index] = bool(
            np.isfinite(close[index]) and np.isfinite(high[index])
            and close[index] >= limit - 0.011 and high[index] >= limit - 0.011
        )
    return flags


def _pool_by_date(root: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for path in sorted((root / "data/cache/limit_ups").glob("*_limitups.pkl")):
        date = path.name.replace("_limitups.pkl", "")
        frame = pd.read_pickle(path).copy()
        if frame.empty or "code" not in frame:
            continue
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        result[date] = frame
    return result


def _volume_lookup(root: Path) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for folder in ("daily_unadjusted", "daily"):
        for path in (root / "data/cache" / folder).glob("*_daily.pkl"):
            code = path.name.replace("_daily.pkl", "")
            try:
                frame = pd.read_pickle(path)
            except Exception:
                continue
            if frame is None or frame.empty or "date" not in frame or "volume" not in frame:
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            volumes = pd.to_numeric(frame["volume"], errors="coerce")
            for date, value in zip(dates, volumes):
                if pd.notna(date) and pd.notna(value):
                    lookup.setdefault(str(date), {})[code] = float(value)
    return lookup


def _pool_features_offline(
    event: Mapping[str, Any], daily: pd.DataFrame, pool: Mapping[str, pd.DataFrame],
    volumes: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    code = str(event["code"]).zfill(6)
    d0 = str(event["d0_date"])
    d1 = str(event["d1_date"])
    dates = daily["date"].tolist()
    idx = dates.index(d1)
    result: dict[str, float] = {}
    for n, feature in ((10, "recent_pool_appearance_count_10d"), (20, "recent_pool_appearance_count_20d")):
        window = dates[max(0, idx - n + 1) : idx + 1]
        if len(window) < n or any(date not in pool for date in window):
            result[feature] = math.nan
        else:
            result[feature] = float(sum(code in set(pool[date]["code"]) for date in window))
    for date, feature in ((d1, "break_day_in_pool"), (d0, "last_board_day_in_pool")):
        result[feature] = float(code in set(pool[date]["code"])) if date in pool else math.nan
    if d0 in pool:
        sub = pool[d0][pool[d0]["code"].eq(code)]
        value = sub.iloc[0].get("consecutive_limit_up_count") if len(sub) else math.nan
        result["pool_consecutive_count_last_board"] = float(value) if pd.notna(value) else math.nan
        members = sorted(set(pool[d0]["code"]))
        day_volumes = {member: volumes.get(d0, {}).get(member) for member in members}
        if code in day_volumes and all(value is not None for value in day_volumes.values()):
            ranks = pd.Series(day_volumes, dtype=float).rank(method="min", ascending=False)
            result["board_day_volume_rank"] = float(ranks.loc[code])
        else:
            result["board_day_volume_rank"] = math.nan
    else:
        result["pool_consecutive_count_last_board"] = math.nan
        result["board_day_volume_rank"] = math.nan
    return result


def reconstruct_existing_53(
    event: Mapping[str, Any], daily: pd.DataFrame, minute: pd.DataFrame,
    pool: Mapping[str, pd.DataFrame], volumes: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Apply only formulas already frozen in FEATURE_CONTRACT."""
    d1_date = str(event["d1_date"])
    d0_date = str(event["d0_date"])
    d1_idx = daily.index[daily["date"].eq(d1_date)].tolist()[0]
    d0_idx = daily.index[daily["date"].eq(d0_date)].tolist()[0]
    d1 = daily.loc[d1_idx]
    d0 = daily.loc[d0_idx]
    streak = 3 if str(event["board_group"]) == "BOARD3" else 2
    flags = _limit_up_flags(daily)
    lo10, lo20 = max(0, d1_idx - 9), max(0, d1_idx - 19)
    max_streak = current = 0
    for flag in flags[lo20 : d1_idx + 1]:
        current = current + 1 if flag else 0
        max_streak = max(max_streak, current)
    out: dict[str, float] = {
        "board_streak_is_3": float(streak == 3),
        "max_board_streak_20d": float(max_streak),
        "recent_limit_up_count_10d": float(flags[lo10 : d1_idx + 1].sum()),
        "recent_limit_up_count_20d": float(flags[lo20 : d1_idx + 1].sum()),
        "break_open_return": _safe_div(d1["open"], d0["close"]) - 1.0,
        "break_high_return": _safe_div(d1["high"], d0["close"]) - 1.0,
        "break_close_return": _safe_div(d1["close"], d0["close"]) - 1.0,
    }
    limit_price = math.floor(float(d0["close"]) * 1.10 * 100.0 + 0.5) / 100.0
    out["break_touched_limit_up"] = float(float(d1["high"]) >= limit_price - 0.011)
    out["break_opened_from_limit_up"] = float(float(d1["open"]) >= limit_price - 0.011)
    daily_range = float(d1["high"] - d1["low"])
    out["break_upper_shadow_ratio"] = (
        (float(d1["high"]) - max(float(d1["open"]), float(d1["close"]))) / daily_range
        if daily_range else math.nan
    )
    out["break_lower_shadow_ratio"] = (
        (min(float(d1["open"]), float(d1["close"])) - float(d1["low"])) / daily_range
        if daily_range else math.nan
    )
    board_rows = daily.iloc[d1_idx - streak : d1_idx]
    out["break_volume_ratio_vs_board_days"] = _safe_div(
        d1["volume"], pd.to_numeric(board_rows["volume"], errors="coerce").mean()
    )
    minute_features = compute_minute_features(
        minute,
        float(d1["close"]), float(d1["high"]), float(d1["low"]),
        float(d0["close"]), float(d1["close"]),
    )
    for feature in (
        "d1_high_to_close_drawdown_raw", "d1_low_to_close_recovery",
        "d1_open_to_close_return_raw", "d1_close_location", "d1_intraday_range",
        "d1_afternoon_return", "d1_last_hour_return", "d1_up_bar_volume_ratio",
        "down_bar_volume_ratio", "volume_above_d1_close_ratio",
        "amount_above_d1_close_ratio", "high_zone_volume_ratio",
        "high_zone_amount_ratio", "late_day_sell_volume_ratio",
        "late_day_sell_amount_ratio", "d1_close_to_vwap_raw",
    ):
        out[feature] = float(minute_features.get(feature, math.nan))
    for window in (5, 10, 20):
        ma = float(d1[f"ma{window}"])
        if window in (5, 10):
            out[f"d1_close_to_ma{window}_raw"] = _safe_div(d1["close"], ma) - 1.0
            out[f"d1_low_to_ma{window}_raw"] = _safe_div(d1["low"], ma) - 1.0
        if window == 5:
            out["d1_high_to_ma5_raw"] = _safe_div(d1["high"], ma) - 1.0
        if window == 20:
            out["d1_close_to_ma20"] = _safe_div(d1["close"], ma) - 1.0
        prior_ma = float(daily.loc[d1_idx - 1, f"ma{window}"])
        out[f"d1_ma{window}_slope"] = _safe_div(ma, prior_ma) - 1.0
    out["d1_close_above_ma5"] = float(float(d1["close"]) >= float(d1["ma5"]))
    out["d1_close_above_ma10"] = float(float(d1["close"]) >= float(d1["ma10"]))
    out["d1_true_reclaim_ma5"] = float(float(d1["low"]) <= float(d1["ma5"]) <= float(d1["close"]))
    out["d1_true_reclaim_ma10"] = float(float(d1["low"]) <= float(d1["ma10"]) <= float(d1["close"]))
    for window in (5, 10):
        count = 0
        idx = d1_idx
        while idx >= 0 and pd.notna(daily.loc[idx, f"ma{window}"]) and float(daily.loc[idx, "close"]) < float(daily.loc[idx, f"ma{window}"]):
            count += 1
            idx -= 1
        out[f"consecutive_days_below_ma{window}"] = float(count)
    recent = compute_recent_7d_features(daily, d1_date)
    out.update({key: float(value) for key, value in recent.items()})
    out["recent_7d_limit_up_count"] = float(flags[max(0, d1_idx - 6) : d1_idx + 1].sum())
    out.update(_pool_features_offline(event, daily, pool, volumes))
    missing = [feature for feature in CONTRACT_FEATURE_NAMES if feature not in out]
    if missing:
        raise RuntimeError(f"FATAL: existing 53F formula reconstruction omitted {missing}")
    return out


def build_53_values(
    pairs: pd.DataFrame, daily_cache: Mapping[str, pd.DataFrame],
    minute_cache: Mapping[str, pd.DataFrame], root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    archived, provenance = _load_archived_53(root)
    pool = _pool_by_date(root)
    volumes = _volume_lookup(root)
    event_rows: dict[str, dict[str, Any]] = {}
    for pair in pairs.itertuples(index=False):
        for role in ("LOSS", "TARGET7"):
            prefix = "loss" if role == "LOSS" else "target"
            event_id = str(getattr(pair, f"{prefix}_event_id"))
            event_rows[event_id] = {
                "event_id": event_id,
                "code": str(getattr(pair, f"{prefix}_code")),
                "board_group": str(getattr(pair, f"{prefix}_board_group")),
                "d0_date": str(getattr(pair, f"{prefix}_d0_date")),
                "d1_date": str(getattr(pair, f"{prefix}_d1_date")),
            }
    filled = 0
    parity_diffs: list[tuple[float, str, str, float, float]] = []
    for event_id, event in sorted(event_rows.items()):
        reconstructed = reconstruct_existing_53(
            event, daily_cache[event_id], minute_cache[event_id], pool, volumes
        )
        existing = archived.setdefault(event_id, {})
        for feature in CONTRACT_FEATURE_NAMES:
            value = existing.get(feature, math.nan)
            if pd.notna(value) and pd.notna(reconstructed[feature]):
                parity_diffs.append((
                    abs(float(value) - float(reconstructed[feature])),
                    event_id,
                    feature,
                    float(value),
                    float(reconstructed[feature]),
                ))
            if pd.isna(value):
                existing[feature] = reconstructed[feature]
                provenance[(event_id, feature)] = "EXISTING_53F_FORMULA_RECONSTRUCTION_FROM_LOCAL_D1_SAFE_CACHES"
                filled += 1

    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        for feature_order, feature in enumerate(CONTRACT_FEATURE_NAMES, 1):
            target_value = archived[pair.target_event_id][feature]
            loss_value = archived[pair.loss_event_id][feature]
            rows.append({
                "case_id": pair.case_id,
                "loss_case_id": pair.loss_case_id,
                "signal_date": pair.signal_date,
                "pair_type": pair.pair_type,
                "feature_order": feature_order,
                "feature_name": feature,
                "target_event_id": pair.target_event_id,
                "loss_event_id": pair.loss_event_id,
                "target_value": target_value,
                "loss_value": loss_value,
                "difference": target_value - loss_value if pd.notna(target_value) and pd.notna(loss_value) else math.nan,
                "target_value_provenance": provenance.get((pair.target_event_id, feature), ""),
                "loss_value_provenance": provenance.get((pair.loss_event_id, feature), ""),
            })
    comparison = pd.DataFrame(rows)
    max_parity = max(parity_diffs, default=(math.nan, "", "", math.nan, math.nan))
    audit = {
        "reconstructed_missing_feature_cells": filled,
        "missing_feature_cells_after_fill": int(comparison[["target_value", "loss_value"]].isna().sum().sum()),
        "archive_vs_formula_max_abs_difference": max_parity[0],
        "archive_vs_formula_max_difference_event": max_parity[1],
        "archive_vs_formula_max_difference_feature": max_parity[2],
        "archive_vs_formula_max_difference_archived": max_parity[3],
        "archive_vs_formula_max_difference_reconstructed": max_parity[4],
        "provenance_blank_cells": int(
            comparison[["target_value_provenance", "loss_value_provenance"]].eq("").sum().sum()
        ),
    }
    return comparison, audit


def build_7f_comparison(pairs: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    by_event = scored.set_index("event_id")
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        for order, feature in enumerate(RAW_PARENT_COLUMNS, 1):
            target = float(by_event.at[pair.target_event_id, feature])
            loss = float(by_event.at[pair.loss_event_id, feature])
            rows.append({
                "case_id": pair.case_id, "loss_case_id": pair.loss_case_id,
                "signal_date": pair.signal_date, "pair_type": pair.pair_type,
                "value_family": "MODEL_7F", "feature_order": order,
                "feature_name": feature, "target_value": target, "loss_value": loss,
                "difference": target - loss,
            })
            parent = RAW_PARENT_COLUMNS[feature]
            target_raw = pd.to_numeric(pd.Series([by_event.at[pair.target_event_id, parent]]), errors="coerce").iloc[0]
            loss_raw = pd.to_numeric(pd.Series([by_event.at[pair.loss_event_id, parent]]), errors="coerce").iloc[0]
            rows.append({
                "case_id": pair.case_id, "loss_case_id": pair.loss_case_id,
                "signal_date": pair.signal_date, "pair_type": pair.pair_type,
                "value_family": "RAW_PARENT", "feature_order": order,
                "feature_name": parent.replace("raw__", ""),
                "target_value": target_raw, "loss_value": loss_raw,
                "difference": target_raw - loss_raw if pd.notna(target_raw) and pd.notna(loss_raw) else math.nan,
            })
    return pd.DataFrame(rows).sort_values(
        ["case_id", "feature_order", "value_family"], kind="mergesort"
    ).reset_index(drop=True)


def build_lineage() -> pd.DataFrame:
    intraday_features = {
        "d1_high_to_close_drawdown_raw", "d1_low_to_close_recovery",
        "d1_open_to_close_return_raw", "d1_close_location", "d1_intraday_range",
        "d1_afternoon_return", "d1_last_hour_return", "d1_up_bar_volume_ratio",
        "down_bar_volume_ratio", "volume_above_d1_close_ratio",
        "amount_above_d1_close_ratio", "high_zone_volume_ratio",
        "high_zone_amount_ratio", "late_day_sell_volume_ratio",
        "late_day_sell_amount_ratio", "d1_close_to_vwap_raw",
    }
    rows = []
    for order, spec in enumerate(FEATURE_CONTRACT, 1):
        name = spec["feature_name"]
        if name in intraday_features:
            cadence = "INTRADAY_5M"
        elif spec["semantic_group"] == "D0_CROSS_SECTION":
            cadence = "DAILY_CROSS_SECTION"
        else:
            cadence = "DAILY_OR_POOL"
        rows.append({
            "feature_order": order,
            "feature_name": name,
            "feature_group": spec["semantic_group"],
            "source": spec["source"],
            "raw_or_derived": "DERIVED_EXISTING",
            "daily_or_intraday": cadence,
            "formula": spec["formula"],
            "raw_dependencies": ";".join(spec["raw_dependencies"]),
            "available_at_D1_close": spec["available_as_of"] in {"D0_CLOSE", "D1_CLOSE"},
            "available_as_of": spec["available_as_of"],
            "existing_historical_audit_used": True,
            "historical_audit": "v004c Stage1 Top3 Risk Information Foundation Audit",
        })
    return pd.DataFrame(rows)


def _descriptor_comparison(pairs: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    by_key = descriptors.set_index(["case_id", "role"])
    rows: list[dict[str, Any]] = []
    numeric = [name for name in PATH_DESCRIPTOR_COLUMNS if not name.startswith("time_of_day")]
    for pair in pairs.itertuples(index=False):
        target = by_key.loc[(pair.case_id, "TARGET7")]
        loss = by_key.loc[(pair.case_id, "LOSS")]
        for feature in numeric:
            rows.append({
                "case_id": pair.case_id,
                "feature_name": feature,
                "target_value": target[feature],
                "loss_value": loss[feature],
                "difference": float(target[feature]) - float(loss[feature]),
            })
    return pd.DataFrame(rows)


def build_wide(
    pairs: pd.DataFrame, seven: pd.DataFrame, fifty_three: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    records = {
        str(row.case_id): dict(row._asdict())
        for row in pairs.itertuples(index=False)
    }
    extra_columns: list[str] = []
    tables = (
        (seven, "s2_or_raw"),
        (fifty_three, "existing53"),
        (_descriptor_comparison(pairs, descriptors), "path"),
    )
    for table, family in tables:
        for row in table.itertuples(index=False):
            feature = str(row.feature_name)
            if table is seven:
                feature = f"{str(row.value_family).lower()}__{feature}"
            for side in ("target", "loss", "difference"):
                column = f"{family}__{side}__{feature}"
                if column not in extra_columns:
                    extra_columns.append(column)
                records[str(row.case_id)][column] = getattr(
                    row, f"{side}_value", getattr(row, side, math.nan)
                )
    columns = pairs.columns.tolist() + extra_columns
    return pd.DataFrame(list(records.values()), columns=columns).sort_values(
        "case_id", kind="mergesort"
    ).reset_index(drop=True)


def _chart_bytes(case: pd.Series, minute: pd.DataFrame) -> bytes:
    selected = minute[minute["case_id"].eq(case["case_id"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    all_values: list[float] = []
    for axis, role, color in zip(axes, ("LOSS", "TARGET7"), ("#C2410C", "#047857")):
        day = selected[selected["role"].eq(role)].sort_values("datetime", kind="mergesort")
        open_ = float(day.iloc[0]["open"])
        normalized = pd.to_numeric(day["close"]) / open_ * 100.0
        vwap = _cumulative_vwap(day) / open_ * 100.0
        x = np.arange(len(day))
        axis.plot(x, normalized, color=color, linewidth=1.8, label="5m close")
        axis.plot(x, vwap, color="#2563EB", linewidth=1.2, linestyle="--", label="cum VWAP")
        high_index = pd.to_numeric(day["high"]).idxmax()
        low_index = pd.to_numeric(day["low"]).idxmin()
        high_pos = day.index.get_loc(high_index)
        low_pos = day.index.get_loc(low_index)
        high_value = float(day.loc[high_index, "high"]) / open_ * 100.0
        low_value = float(day.loc[low_index, "low"]) / open_ * 100.0
        close_value = float(day.iloc[-1]["close"]) / open_ * 100.0
        axis.scatter([high_pos, low_pos, len(day) - 1], [high_value, low_value, close_value], s=28, zorder=4)
        axis.annotate("H", (high_pos, high_value), xytext=(0, 6), textcoords="offset points")
        axis.annotate("L", (low_pos, low_value), xytext=(0, -12), textcoords="offset points")
        axis.annotate("C", (len(day) - 1, close_value), xytext=(4, 0), textcoords="offset points")
        code = case["loss_code"] if role == "LOSS" else case["target_code"]
        rank = case["loss_s2_rank"] if role == "LOSS" else case["target_s2_rank"]
        axis.set_title(f"{role}  {code}  S2 rank {int(rank)}")
        axis.set_ylabel("D1 open = 100")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
        all_values.extend(normalized.tolist())
        all_values.extend(vwap.dropna().tolist())
        all_values.extend([high_value, low_value])
    pad = max(0.5, (max(all_values) - min(all_values)) * 0.08)
    for axis in axes:
        axis.set_ylim(min(all_values) - pad, max(all_values) + pad)
    bottom = selected[selected["role"].eq("TARGET7")].sort_values("datetime")
    ticks = list(range(0, len(bottom), 8)) + [len(bottom) - 1]
    axes[-1].set_xticks(sorted(set(ticks)))
    axes[-1].set_xticklabels([bottom.iloc[i]["datetime"].strftime("%H:%M") for i in sorted(set(ticks))])
    axes[-1].set_xlabel("D1 time")
    fig.suptitle(f"{case['case_id']}  {case['signal_date']}  normalized D1 paths", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, metadata={"Software": TASK_NAME})
    plt.close(fig)
    return buffer.getvalue()


def build_charts(pairs: pd.DataFrame, minute: pd.DataFrame) -> dict[str, bytes]:
    return {
        f"case_charts/{row.case_id}_{row.signal_date}_{row.loss_code}_vs_{row.target_code}.png":
        _chart_bytes(pd.Series(row._asdict()), minute)
        for row in pairs.itertuples(index=False)
    }


def _integrity_rows(
    pairs: pd.DataFrame, minute: pd.DataFrame, daily: pd.DataFrame,
    fifty_three: pd.DataFrame, pair_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any], chart_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(section: str, check: str, actual: Any, expected: Any, status: bool, details: str = "") -> None:
        rows.append({
            "section": section, "check": check, "actual": actual, "expected": expected,
            "status": "PASS" if status else "FAIL", "details": details,
        })
    add("CASE", "pair_count", len(pairs), pair_audit["pair_count"], len(pairs) == pair_audit["pair_count"])
    add("CASE", "unique_loss_count", pairs["loss_event_id"].nunique(), 25, pairs["loss_event_id"].nunique() == 25,
        "Authoritative archived HARD_FALSE_POSITIVE_LOSS identities")
    add("CASE", "theoretical_one_to_one_replaceable_slots", pair_audit["theoretical_one_to_one_replaceable_slots"], 25,
        pair_audit["theoretical_one_to_one_replaceable_slots"] == 25)
    add("CASE", "unique_target_count", pairs["target_event_id"].nunique(), pair_audit["unique_target_count"], True)
    add("CASE", "duplicate_pair_count", int(pairs.duplicated(["loss_event_id", "target_event_id"]).sum()), 0,
        not pairs.duplicated(["loss_event_id", "target_event_id"]).any())
    for month in ("2026-05", "2026-06", "2026-07"):
        count = int(pairs.loc[pairs["month"].eq(month), "loss_event_id"].nunique())
        add("CASE", f"{month}_loss_cases", count, "descriptive", True)
    for rank in (1, 2, 3):
        count = int(pairs.loc[pairs["loss_s2_rank"].eq(rank), "loss_event_id"].nunique())
        add("CASE", f"rank{rank}_loss_cases", count, "descriptive", True)
    for board in ("BOARD2", "BOARD3"):
        count = int(pairs.loc[pairs["loss_board_group"].eq(board), "loss_event_id"].nunique())
        add("CASE", f"{board}_loss_cases", count, "descriptive", True)
    add("CASE", "chart_count", chart_count, len(pairs), chart_count == len(pairs))
    add("DAILY", "rows", len(daily), ">=20 rows per pair", len(daily) >= len(pairs) * 2 * 11)
    for (case_id, role), part in minute.groupby(["case_id", "role"], sort=True):
        latest = pd.to_datetime(part["datetime"]).max().strftime("%Y-%m-%d %H:%M:%S")
        d1 = str(part["signal_date"].iloc[0])
        add("5MIN", f"{case_id}:{role}:bar_count", len(part), 48, len(part) == 48, f"latest={latest}")
        add("5MIN", f"{case_id}:{role}:D2_D3_bar_intrusion", int(part["trade_date"].astype(str).ne(d1).sum()), 0,
            part["trade_date"].astype(str).eq(d1).all())
    add("53F", "expected_feature_count", fifty_three["feature_name"].nunique(), 53,
        fifty_three["feature_name"].nunique() == 53)
    add("53F", "rows", len(fifty_three), len(pairs) * 53, len(fifty_three) == len(pairs) * 53)
    add("53F", "missing_feature_cells", feature_audit["missing_feature_cells_after_fill"], 0,
        feature_audit["missing_feature_cells_after_fill"] == 0)
    add(
        "53F", "formula_reconstructed_cells",
        feature_audit["reconstructed_missing_feature_cells"], "descriptive", True,
        "Archived values were retained wherever present",
    )
    add("53F", "provenance_blank_cells", feature_audit["provenance_blank_cells"], 0,
        feature_audit["provenance_blank_cells"] == 0)
    add(
        "53F", "archive_vs_formula_max_difference",
        feature_audit["archive_vs_formula_max_abs_difference"], "descriptive; archived value retained", True,
        (
            f"event={feature_audit['archive_vs_formula_max_difference_event']}; "
            f"feature={feature_audit['archive_vs_formula_max_difference_feature']}; "
            f"archived={feature_audit['archive_vs_formula_max_difference_archived']}; "
            f"reconstructed={feature_audit['archive_vs_formula_max_difference_reconstructed']}"
        ),
    )
    add("BOUNDARY", "AUGUST_SIGNAL_OUTCOME_ACCESSED", "NO", "NO", True)
    add("BOUNDARY", "max_signal_date", pairs["signal_date"].max(), MAX_SIGNAL_DATE,
        pairs["signal_date"].max() <= MAX_SIGNAL_DATE)
    return pd.DataFrame(rows)


def _path_field_names(lineage: pd.DataFrame) -> list[str]:
    mask = lineage["daily_or_intraday"].eq("INTRADAY_5M")
    return lineage.loc[mask, "feature_name"].tolist()


def build_review(
    pairs: pd.DataFrame, minute: pd.DataFrame, fifty_three: pd.DataFrame,
    lineage: pd.DataFrame, integrity: pd.DataFrame, pair_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> str:
    failures = integrity[integrity["status"].eq("FAIL")]
    state = "READY_FOR_MECHANISM_REVIEW" if failures.empty else "DATA_INCOMPLETE"
    action = "MANUAL_MECHANISM_REVIEW" if failures.empty else "FIX_DATA_PACKAGE"
    month_counts = pairs.groupby("month")["loss_event_id"].nunique().to_dict()
    rank_counts = pairs.groupby("loss_s2_rank")["loss_event_id"].nunique().to_dict()
    board_counts = pairs.groupby("loss_board_group")["loss_event_id"].nunique().to_dict()
    source_counts = minute.groupby("event_id")["minute_source_used"].first().value_counts().to_dict()
    path_fields = _path_field_names(lineage)
    lines = [
        "# v004c S2 Replaceable-Loss Case Pack v001",
        "",
        "## Contract",
        "",
        "- Data preparation only; no model fit, parameter search, feature search, or August signal-date outcome access.",
        "- Signal-date boundary: 2026-05-06 through 2026-07-29.",
        "- Pair rules: highest-ranked outside-Top3 Target7 and nearest-score outside-Top3 Target7; duplicate A/B selections retained once.",
        "",
        "## Q1. How many replaceable LOSS cases were prepared?",
        "",
        f"- Unique HARD_FALSE_POSITIVE_LOSS cases: {pairs['loss_event_id'].nunique()}.",
        f"- Pair rows after fixed-rule A/B de-duplication: {len(pairs)}.",
        f"- Archived one-to-one replacement capacity: {pair_audit['theoretical_one_to_one_replaceable_slots']} slots; the case identities match this authoritative attribution exactly.",
        f"- May / June / July cases: {month_counts.get('2026-05', 0)} / {month_counts.get('2026-06', 0)} / {month_counts.get('2026-07', 0)}.",
        "",
        "## Q2. Where are the errors concentrated?",
        "",
        f"- Rank1 / Rank2 / Rank3 LOSS cases: {rank_counts.get(1, 0)} / {rank_counts.get(2, 0)} / {rank_counts.get(3, 0)}.",
        f"- Board2 / Board3 LOSS cases: {board_counts.get('BOARD2', 0)} / {board_counts.get('BOARD3', 0)}.",
        "",
        "## Q3. Are D1 daily, 5-minute, 7F, and 53F data complete?",
        "",
        f"- D1 5-minute source use by unique event: {source_counts}.",
        f"- Every case side has 48 D1 bars: {'YES' if not integrity[(integrity['section'].eq('5MIN')) & (integrity['status'].eq('FAIL'))].shape[0] else 'NO'}.",
        f"- 53F feature count: {fifty_three['feature_name'].nunique()}; missing target/loss value cells: {int(fifty_three[['target_value','loss_value']].isna().sum().sum())}.",
        f"- Formula-reconstructed 53F cells: {feature_audit['reconstructed_missing_feature_cells']}; every such cell carries explicit provenance and no archived value was overwritten.",
        (
            "- Largest archived-vs-local reconstruction difference on overlap: "
            f"{feature_audit['archive_vs_formula_max_abs_difference']} at "
            f"{feature_audit['archive_vs_formula_max_difference_event']} / "
            f"{feature_audit['archive_vs_formula_max_difference_feature']} "
            f"({feature_audit['archive_vs_formula_max_difference_archived']} archived vs "
            f"{feature_audit['archive_vs_formula_max_difference_reconstructed']} reconstructed); "
            "the archived value was retained."
        ),
        "- Frozen S2 7F and their existing raw parents are included in the 7F comparison table.",
        "",
        "## Q4. Which existing 53F fields contain intraday/path semantics?",
        "",
        "- " + ", ".join(path_fields) + ".",
        "- These names and semantics are listed only as coverage metadata; no predictive-effect ranking was performed.",
        "",
        "## Q5. Are there missing-data or time-boundary issues?",
        "",
        f"- Integrity failures: {len(failures)}.",
        "- 5-minute exports contain D1 only; no D2/D3 bar is included.",
        "- AUGUST_SIGNAL_OUTCOME_ACCESSED = NO.",
        "",
        f"CASE_PACK_STATE = {state}",
        "",
        f"NEXT_ACTION = {action}",
        "",
    ]
    if len(failures):
        lines.extend(["## Integrity Failures", ""] + [
            f"- {row.section}/{row.check}: actual={row.actual}, expected={row.expected}"
            for row in failures.itertuples(index=False)
        ])
    return "\n".join(lines)


def build_outputs(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_contract()
    root_path = Path(root).resolve()
    scored, _, frozen_audit = load_frozen_s2(root_path)
    scored = scored[
        scored["signal_date"].le(MAX_SIGNAL_DATE)
        & scored["label_available_date"].lt(AUGUST_ASOF)
    ].copy()
    pairs, pair_audit = build_case_pairs(scored, root_path)
    daily, daily_cache = build_daily_history(pairs, root_path)
    minute, descriptors, minute_cache = build_minute_and_descriptors(pairs, root_path)
    seven = build_7f_comparison(pairs, scored)
    fifty_three, feature_audit = build_53_values(
        pairs, daily_cache, minute_cache, root_path
    )
    lineage = build_lineage()
    wide = build_wide(pairs, seven, fifty_three, descriptors)
    charts = build_charts(pairs, minute)
    integrity = _integrity_rows(
        pairs, minute, daily, fifty_three, pair_audit, feature_audit, len(charts)
    )
    review = build_review(
        pairs, minute, fifty_three, lineage, integrity, pair_audit, feature_audit
    )
    outputs: dict[str, bytes] = {
        OUTPUT_FILENAMES[0]: _csv_bytes(pairs),
        OUTPUT_FILENAMES[1]: _csv_bytes(daily),
        OUTPUT_FILENAMES[2]: _csv_bytes(minute),
        OUTPUT_FILENAMES[3]: _csv_bytes(descriptors),
        OUTPUT_FILENAMES[4]: _csv_bytes(seven),
        OUTPUT_FILENAMES[5]: _csv_bytes(fifty_three),
        OUTPUT_FILENAMES[6]: _csv_bytes(lineage),
        OUTPUT_FILENAMES[7]: _csv_bytes(wide),
        OUTPUT_FILENAMES[8]: _csv_bytes(integrity),
        OUTPUT_FILENAMES[9]: review.encode("utf-8"),
        **charts,
    }
    metrics = {
        **pair_audit,
        **feature_audit,
        "output_file_count": len(outputs),
        "chart_count": len(charts),
        "minute_rows": len(minute),
        "daily_history_rows": len(daily),
        "month_loss_counts": pairs.groupby("month")["loss_event_id"].nunique().to_dict(),
        "rank_loss_counts": pairs.groupby("loss_s2_rank")["loss_event_id"].nunique().to_dict(),
        "board_loss_counts": pairs.groupby("loss_board_group")["loss_event_id"].nunique().to_dict(),
        "minute_sources": minute.groupby("event_id")["minute_source_used"].first().value_counts().to_dict(),
        "integrity_failures": int(integrity["status"].eq("FAIL").sum()),
        "case_pack_state": "READY_FOR_MECHANISM_REVIEW" if not integrity["status"].eq("FAIL").any() else "DATA_INCOMPLETE",
        "next_action": "MANUAL_MECHANISM_REVIEW" if not integrity["status"].eq("FAIL").any() else "FIX_DATA_PACKAGE",
        "august_signal_outcome_accessed": "NO",
        "s2_model_refits": int(frozen_audit["model_refits"]),
    }
    return outputs, metrics


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, metrics = build_outputs(root_path)
    output_dir = root_path / "reports/research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, content in outputs.items():
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return output_dir, metrics

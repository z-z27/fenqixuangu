from __future__ import annotations

import pandas as pd


def enrich_daily_indicators(daily: pd.DataFrame, full_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    base = (full_daily if full_daily is not None else daily).copy().sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base["prev_close"] = base["close"].shift(1)
    for period in (5, 10, 20, 30):
        base[f"ma{period}"] = base["close"].rolling(period, min_periods=period).mean()
        base[f"amount_ma{period}"] = base["amount"].rolling(period, min_periods=period).mean()
    base["close_position"] = safe_divide(base["close"] - base["low"], base["high"] - base["low"])
    base["amplitude_calc"] = safe_divide(base["high"] - base["low"], base["prev_close"]) * 100
    base["amount_ratio"] = safe_divide(base["amount"], base["amount_ma10"])
    base["ma5_distance"] = safe_divide(base["close"], base["ma5"]) - 1
    base["ma10_distance"] = safe_divide(base["close"], base["ma10"]) - 1
    base["ma20_distance"] = safe_divide(base["close"], base["ma20"]) - 1
    base["vwap_daily"] = infer_vwap(base["amount"], base["volume"], base["close"])
    if full_daily is not None:
        base = base[base["date"].isin(daily["date"])].sort_values("date").reset_index(drop=True)
    return base


def enrich_5min_indicators(minute: pd.DataFrame) -> pd.DataFrame:
    result = minute.copy().sort_values("datetime").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["cum_amount"] = result.groupby("trade_date")["amount"].cumsum()
    result["cum_volume"] = result.groupby("trade_date")["volume"].cumsum()
    result["intraday_vwap"] = infer_vwap(result["cum_amount"], result["cum_volume"], result["close"])
    result["bar_return"] = result.groupby("trade_date")["close"].pct_change()
    result["is_above_vwap"] = result["close"] >= result["intraday_vwap"]
    day_open = result.groupby("trade_date")["open"].transform("first")
    result["day_open"] = day_open
    result["is_reclaim_open"] = result["close"] >= day_open
    return result


def build_key_zones(daily: pd.DataFrame, minute: pd.DataFrame | None = None) -> dict[str, float | None]:
    if daily.empty:
        return {}
    d1 = daily.iloc[-1]
    zones: dict[str, float | None] = {
        "d1_low": as_float(d1.get("low")),
        "d1_open": as_float(d1.get("open")),
        "d1_close": as_float(d1.get("close")),
        "d1_vwap": as_float(d1.get("vwap_daily")),
        "prev_close": as_float(d1.get("prev_close")),
        "ma5": as_float(d1.get("ma5")),
        "ma10": as_float(d1.get("ma10")),
    }
    if minute is not None and not minute.empty:
        recent_date = str(minute["trade_date"].max())
        day_rows = minute[minute["trade_date"].astype(str) == recent_date]
        if not day_rows.empty and "intraday_vwap" in day_rows.columns:
            zones["d1_intraday_vwap"] = as_float(day_rows["intraday_vwap"].dropna().iloc[-1]) if day_rows["intraday_vwap"].notna().any() else None
            zones["d1_first_5m_low"] = as_float(day_rows["low"].iloc[0])

    # D2 low-absorb should cluster around the D1 defended cost zone.
    # Far lower moving averages are background references, not executable buy zones.
    anchor = first_valid_number(zones.get("d1_low"), zones.get("d1_first_5m_low"), zones.get("prev_close"))
    support_keys = (
        "d1_low",
        "d1_first_5m_low",
        "d1_open",
        "prev_close",
        "d1_vwap",
        "d1_intraday_vwap",
        "ma5",
    )
    support_values = valid_numbers(zones.get(key) for key in support_keys)
    if anchor is not None and support_values:
        lower_bound = anchor * 0.985
        upper_bound = anchor * 1.05
        clustered = [value for value in support_values if lower_bound <= value <= upper_bound]
        if not clustered:
            clustered = [anchor]
        zones["low_absorb_min"] = min(clustered)
        zones["low_absorb_max"] = max(clustered)
        zones["invalid_price"] = zones["low_absorb_min"]
    else:
        zones["low_absorb_min"] = None
        zones["low_absorb_max"] = None
        zones["invalid_price"] = None
    return zones


def safe_divide(a, b):
    return a / b.replace(0, pd.NA) if hasattr(b, "replace") else a / b if b else pd.NA


def infer_vwap(amount: pd.Series, volume: pd.Series, reference_price: pd.Series) -> pd.Series:
    """Infer whether volume is shares or hands, then compute price-level VWAP."""
    raw = safe_divide(amount, volume)
    hand_adjusted = safe_divide(amount, volume * 100)
    use_hand_unit = raw > reference_price * 5
    return raw.mask(use_hand_unit, hand_adjusted)


def as_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def first_valid_number(*values) -> float | None:
    for value in values:
        number = as_float(value)
        if number is not None:
            return number
    return None


def valid_numbers(values) -> list[float]:
    return [number for number in (as_float(value) for value in values) if number is not None]

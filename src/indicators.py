from __future__ import annotations

import pandas as pd


def enrich_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy().sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["prev_close"] = result["close"].shift(1)
    for period in (5, 10, 20, 30):
        result[f"ma{period}"] = result["close"].rolling(period, min_periods=period).mean()
        result[f"amount_ma{period}"] = result["amount"].rolling(period, min_periods=period).mean()
    result["close_position"] = safe_divide(result["close"] - result["low"], result["high"] - result["low"])
    result["amplitude_calc"] = safe_divide(result["high"] - result["low"], result["prev_close"]) * 100
    result["amount_ratio"] = safe_divide(result["amount"], result["amount_ma10"])
    result["ma5_distance"] = safe_divide(result["close"], result["ma5"]) - 1
    result["ma10_distance"] = safe_divide(result["close"], result["ma10"]) - 1
    result["ma20_distance"] = safe_divide(result["close"], result["ma20"]) - 1
    result["vwap_daily"] = infer_vwap(result["amount"], result["volume"], result["close"])
    return result


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
    support_keys = ("d1_low", "d1_open", "prev_close", "ma5", "ma10", "d1_first_5m_low")
    support_values = [
        zones.get(key)
        for key in support_keys
        if zones.get(key) is not None and pd.notna(zones.get(key))
    ]
    if support_values:
        anchor = min(support_values)
        clustered = [value for value in support_values if value <= anchor * 1.05]
        zones["low_absorb_min"] = min(clustered)
        zones["low_absorb_max"] = max(clustered)
    else:
        zones["low_absorb_min"] = None
        zones["low_absorb_max"] = None
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

from __future__ import annotations

import pandas as pd


def simulate_d2_execution(signal_row: pd.Series, minute_d2: pd.DataFrame) -> dict:
    """Minimal D2 execution simulator using only intraday rows up to trigger time."""
    if minute_d2.empty:
        return {"executed": False, "reason": "D2 minute data is empty"}

    invalid_price = signal_row.get("invalid_price")
    low_absorb_min = signal_row.get("low_absorb_min")
    low_absorb_max = signal_row.get("low_absorb_max")
    rows = minute_d2.sort_values("datetime").reset_index(drop=True)
    for _, row in rows.iterrows():
        low = float(row["low"])
        close = float(row["close"])
        vwap = row.get("intraday_vwap")
        if pd.notna(invalid_price) and low < float(invalid_price) and close < float(invalid_price):
            return {
                "executed": False,
                "reason": "跌破失效位且未收回",
                "time": str(row["datetime"]),
            }
        in_zone = (
            pd.notna(low_absorb_min)
            and pd.notna(low_absorb_max)
            and low <= float(low_absorb_max)
            and close >= float(low_absorb_min)
        )
        if in_zone and pd.notna(vwap) and close >= float(vwap):
            return {
                "executed": True,
                "reason": "回踩低吸区并站回 VWAP",
                "time": str(row["datetime"]),
                "price": close,
            }
    return {"executed": False, "reason": "D2 未触发低吸验证"}

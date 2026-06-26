from __future__ import annotations

import pandas as pd


def score_active_money(daily: pd.DataFrame, minute: pd.DataFrame) -> tuple[float, list[str]]:
    if daily is None or daily.empty:
        return 0.0, ["日线样本为空"]

    d1 = daily.iloc[-1]
    score = 40.0
    reasons: list[str] = []

    raw_ar = d1.get("amount_ratio")
    amount_ratio = float(raw_ar) if pd.notna(raw_ar) else 0.0
    raw_amp = d1.get("amplitude_calc", d1.get("amplitude"))
    amplitude = float(raw_amp) if pd.notna(raw_amp) else 0.0

    if amount_ratio >= 2.0:
        score += 22
        reasons.append("D1 成交额显著放大")
    elif amount_ratio >= 1.3:
        score += 12
        reasons.append("D1 成交额温和放大")
    else:
        score -= 8
        reasons.append("D1 成交额放大不足")

    if amplitude >= 8:
        score += 14
        reasons.append("D1 振幅明显放大")
    elif amplitude >= 5:
        score += 8
        reasons.append("D1 有一定振幅")

    if minute is not None and not minute.empty:
        recent_date = str(minute["trade_date"].max())
        day_rows = minute[minute["trade_date"].astype(str) == recent_date]
        if not day_rows.empty:
            flips = _vwap_flips(day_rows)
            if flips >= 3:
                score += 10
                reasons.append("分时围绕 VWAP 反复争夺")
            if day_rows["close"].iloc[-1] >= day_rows["intraday_vwap"].iloc[-1]:
                score += 8
                reasons.append("尾盘站在日内 VWAP 上方")
            if day_rows["bar_return"].abs().max(skipna=True) >= 0.025:
                score += 8
                reasons.append("分时有急拉急杀波动")

    return max(0.0, min(100.0, score)), reasons


def _vwap_flips(day_rows: pd.DataFrame) -> int:
    if "is_above_vwap" not in day_rows.columns:
        return 0
    series = day_rows["is_above_vwap"].dropna().astype(bool)
    if series.empty:
        return 0
    return int((series != series.shift(1)).sum())

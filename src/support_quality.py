from __future__ import annotations

import pandas as pd


def score_support_quality(daily: pd.DataFrame, minute: pd.DataFrame) -> tuple[float, str, list[str]]:
    if daily is None or daily.empty:
        return 0.0, "C", ["日线样本为空"]

    d1 = daily.iloc[-1]
    score = 45.0
    support_type = "B"
    reasons: list[str] = []

    close_position = float(d1.get("close_position", 0) or 0)
    vwap = d1.get("vwap_daily")
    close = d1.get("close")
    low = d1.get("low")
    prev_low = daily["low"].iloc[-2] if len(daily) >= 2 else None

    if prev_low is not None and pd.notna(prev_low):
        if float(low) >= float(prev_low):
            score += 10
            reasons.append("D1 未跌破前一日低点")
        elif float(close) > float(prev_low):
            score += 8
            reasons.append("D1 跌破前低后收回")
        else:
            score -= 18
            reasons.append("D1 跌破前低且未收回")

    if pd.notna(vwap) and pd.notna(close):
        if float(close) >= float(vwap):
            score += 14
            reasons.append("D1 收盘站上日 VWAP")
        else:
            score -= 8
            reasons.append("D1 收盘低于日 VWAP")

    if close_position >= 0.65:
        score += 12
        reasons.append("D1 收盘位置较强")
    elif close_position <= 0.25:
        score -= 18
        reasons.append("D1 收盘接近全天低位")

    if minute is not None and not minute.empty:
        recent_date = str(minute["trade_date"].max())
        day_rows = minute[minute["trade_date"].astype(str) == recent_date]
        if not day_rows.empty and "intraday_vwap" in day_rows.columns:
            last = day_rows.iloc[-1]
            above_vwap_ratio = float(day_rows["is_above_vwap"].mean()) if "is_above_vwap" in day_rows else 0.0
            if last["close"] >= last["intraday_vwap"] and above_vwap_ratio >= 0.45:
                score += 10
                reasons.append("D1 分时主动修复到 VWAP 上方")
            elif above_vwap_ratio < 0.25:
                score -= 8
                reasons.append("D1 多数时间压在 VWAP 下方")

    if score >= 65:
        support_type = "A"
    elif score < 45:
        support_type = "C"

    return max(0.0, min(100.0, score)), support_type, reasons

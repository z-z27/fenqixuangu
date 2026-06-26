from __future__ import annotations

import pandas as pd


def score_graph_quality(daily: pd.DataFrame) -> tuple[float, list[str]]:
    if daily is None or daily.empty or len(daily) < 20:
        return 0.0, ["日线样本不足，无法判断图形趋势"]

    df = daily.sort_values("date").reset_index(drop=True).copy()
    recent = df.tail(10)
    score = 50.0
    reasons: list[str] = []

    if recent["high"].iloc[-1] >= recent["high"].iloc[:5].max():
        score += 10
        reasons.append("近 10 日高点保持抬高")
    else:
        score -= 8
        reasons.append("近 10 日高点未能继续抬高")

    if recent["low"].tail(5).min() >= recent["low"].head(5).min():
        score += 10
        reasons.append("近 10 日低点没有明显下移")
    else:
        score -= 10
        reasons.append("近 10 日低点下移")

    last = df.iloc[-1]
    if last.get("ma5") > last.get("ma10") > last.get("ma20"):
        score += 12
        reasons.append("均线多头排列")
    elif last.get("close") < last.get("ma5") and last.get("close") < last.get("ma10"):
        score -= 12
        reasons.append("收盘跌破短期均线")

    raw_cp = last.get("close_position")
    close_position = float(raw_cp) if pd.notna(raw_cp) else 0.0
    if close_position >= 0.7:
        score += 8
        reasons.append("D1 收盘位置强")
    elif close_position <= 0.3:
        score -= 12
        reasons.append("D1 收盘接近低位")

    upper_shadow_ratio = _upper_shadow_ratio(df.tail(5))
    if upper_shadow_ratio >= 0.55:
        score -= 10
        reasons.append("近期上影线压力偏重")

    raw_ar = last.get("amount_ratio")
    amount_ratio = float(raw_ar) if pd.notna(raw_ar) else 0.0
    high_volume_fail = amount_ratio >= 2.0 and close_position <= 0.25
    if high_volume_fail:
        score -= 18
        reasons.append("D1 有高位巨量失败风险")

    return max(0.0, min(100.0, score)), reasons


def _upper_shadow_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    body_high = frame[["open", "close"]].max(axis=1)
    total_range = (frame["high"] - frame["low"]).replace(0, pd.NA)
    ratio = ((frame["high"] - body_high) / total_range).fillna(0)
    return float((ratio > 0.45).mean())

"""v004c minute-dependent feature computation — EXACT reproduction of the
official formulas (同一正式公式, 不改变数学定义).

Source: reports/research/v004c_dataset_20260506_20260729/_scratch/
        build_v004c_dataset.py  (section E "D1 修复质量 (5min)" and
        section F "对手盘与成交压力", lines ~479-492 intraday_vwap,
        ~670-737 feature rows)

These helpers are copied 1:1 (plus None-safe handling identical to the
script's _ret/_safe_div/_first_numeric/_last_numeric/_max_numeric/
_min_numeric). The official script executes a full build at import time,
so it cannot be imported directly; formulas are reproduced here with the
same arithmetic. The identical copy (with the same formulas) was previously
anchor-validated against frozen model outputs (max|diff| = 5e-7) in the
Sina -> BaoStock source replay task.
"""
from __future__ import annotations

import pandas as pd


def _to_float(v) -> float | None:
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ret(v) -> float | None:
    v = _to_float(v)
    return None if v is None else v - 1.0


def _safe_div(a, b) -> float | None:
    a = _to_float(a)
    b = _to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _first_numeric(frame: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return _to_float(s.iloc[0]) if len(s) else None


def _last_numeric(frame: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return _to_float(s.iloc[-1]) if len(s) else None


def _max_numeric(frame: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return _to_float(s.max()) if len(s) else None


def _min_numeric(frame: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return _to_float(s.min()) if len(s) else None


def intraday_vwap(day_m: pd.DataFrame) -> float | None:
    """Official intraday_vwap: cum_amount/cum_volume with the unit-heuristic
    (raw > ref*5 => treat volume as lots, hand = cum_a/(cum_v*100)).
    Copied from build_v004c_dataset.py lines 479-492."""
    if day_m.empty:
        return None
    amount = pd.to_numeric(day_m["amount"], errors="coerce")
    volume = pd.to_numeric(day_m["volume"], errors="coerce")
    close = pd.to_numeric(day_m["close"], errors="coerce")
    cum_a = amount.cumsum()
    cum_v = volume.cumsum()
    ref = close
    raw = cum_a / cum_v.replace(0, pd.NA)
    hand = cum_a / (cum_v * 100).replace(0, pd.NA)
    use_hand = raw > ref * 5
    vwap_series = raw.mask(use_hand, hand).dropna()
    return float(vwap_series.iloc[-1]) if len(vwap_series) else None


def compute_minute_features(
    day_m: pd.DataFrame,
    daily_close: float | None,
    daily_high: float | None,
    daily_low: float | None,
    prev_close_d1: float | None,
    break_close: float | None,
) -> dict:
    """Reproduces build_v004c_dataset.py rows (E/F sections) for ONE D1 day.

    day_m: 5m bars of the signal_date, columns open/high/low/close/volume/
           amount/time (HH:MM:SS), sorted by datetime.
    daily_close/high/low: CANONICAL DAILY values (same reference for both
           sources, so they cannot confound the source comparison).
    """
    out: dict = {}
    if day_m is None or day_m.empty:
        return out
    o_c = _first_numeric(day_m, "open")
    h_c = _max_numeric(day_m, "high")
    l_c = _min_numeric(day_m, "low")
    c_c = _last_numeric(day_m, "close")
    out["d1_open_to_close_return_raw"] = _ret(_safe_div(c_c, o_c))
    out["d1_low_to_close_recovery"] = _safe_div(c_c - l_c, l_c) if (c_c is not None and l_c) else None
    out["d1_high_to_close_drawdown_raw"] = _safe_div(h_c - c_c, h_c) if (h_c is not None and c_c is not None) else None
    out["d1_close_location"] = (
        _safe_div(c_c - l_c, h_c - l_c) if (h_c is not None and l_c is not None and h_c != l_c) else None
    )
    vwap = intraday_vwap(day_m)
    out["d1_vwap"] = vwap
    out["d1_close_to_vwap_raw"] = _ret(_safe_div(c_c, vwap))
    out["d1_intraday_range"] = _safe_div(h_c - l_c, prev_close_d1) if (h_c is not None and l_c is not None) else None
    t_col = pd.to_datetime(day_m["time"], format="%H:%M:%S", errors="coerce").dt.time
    morning = day_m[t_col < pd.Timestamp("13:00:00").time()]
    morning_close = _last_numeric(morning, "close")
    out["d1_afternoon_return"] = _ret(_safe_div(c_c, morning_close))
    hour14 = day_m[t_col <= pd.Timestamp("14:00:00").time()]
    hour14_close = _last_numeric(hour14, "close")
    out["d1_last_hour_return"] = _ret(_safe_div(c_c, hour14_close))
    tot_vol = float(pd.to_numeric(day_m["volume"], errors="coerce").sum())
    tot_amt = float(pd.to_numeric(day_m["amount"], errors="coerce").sum())
    if tot_vol and tot_vol > 0:
        up_vol = float(pd.to_numeric(day_m.loc[day_m["close"] > day_m["open"], "volume"], errors="coerce").sum())
        dn_vol = float(pd.to_numeric(day_m.loc[day_m["close"] < day_m["open"], "volume"], errors="coerce").sum())
        out["d1_up_bar_volume_ratio"] = up_vol / tot_vol
        out["d1_down_bar_volume_ratio"] = dn_vol / tot_vol
    else:
        out["d1_up_bar_volume_ratio"] = None
        out["d1_down_bar_volume_ratio"] = None
    d1_close_v2 = daily_close
    brk_close = break_close
    d1_high_v = daily_high
    d1_low_v = daily_low
    high_zone_thr = d1_low_v + 0.7 * (d1_high_v - d1_low_v) if (d1_low_v is not None and d1_high_v is not None) else None
    vol_s = pd.to_numeric(day_m["volume"], errors="coerce")
    amt_s = pd.to_numeric(day_m["amount"], errors="coerce")
    cl_s = pd.to_numeric(day_m["close"], errors="coerce")
    op_s = pd.to_numeric(day_m["open"], errors="coerce")
    hi_s = pd.to_numeric(day_m["high"], errors="coerce")
    lo_s = pd.to_numeric(day_m["low"], errors="coerce")
    typ = (hi_s + lo_s + cl_s) / 3.0
    t_afternoon = t_col >= pd.Timestamp("14:00:00").time()
    if tot_vol and tot_vol > 0:
        out["volume_above_d1_close_ratio"] = (
            float(vol_s[cl_s >= d1_close_v2].sum()) / tot_vol if d1_close_v2 is not None else None
        )
        out["volume_above_break_close_ratio"] = (
            float(vol_s[cl_s >= brk_close].sum()) / tot_vol if brk_close is not None else None
        )
        out["high_zone_volume_ratio"] = (
            float(vol_s[typ >= high_zone_thr].sum()) / tot_vol if high_zone_thr is not None else None
        )
        out["late_day_sell_volume_ratio"] = float(vol_s[t_afternoon & (cl_s < op_s)].sum()) / tot_vol
        out["down_bar_volume_ratio"] = float(vol_s[cl_s < op_s].sum()) / tot_vol
    else:
        for k in ("volume_above_d1_close_ratio", "volume_above_break_close_ratio",
                  "high_zone_volume_ratio", "late_day_sell_volume_ratio", "down_bar_volume_ratio"):
            out[k] = None
    if tot_amt and tot_amt > 0:
        out["amount_above_d1_close_ratio"] = (
            float(amt_s[cl_s >= d1_close_v2].sum()) / tot_amt if d1_close_v2 is not None else None
        )
        out["high_zone_amount_ratio"] = (
            float(amt_s[typ >= high_zone_thr].sum()) / tot_amt if high_zone_thr is not None else None
        )
        out["late_day_sell_amount_ratio"] = float(amt_s[t_afternoon & (cl_s < op_s)].sum()) / tot_amt
    else:
        for k in ("amount_above_d1_close_ratio", "high_zone_amount_ratio", "late_day_sell_amount_ratio"):
            out[k] = None
    out["d1_vwap_to_close_gap"] = _safe_div(vwap - c_c, c_c) if vwap is not None and c_c is not None else None
    out["d1_amount"] = tot_amt if tot_amt and tot_amt > 0 else None
    out["d1_volume"] = tot_vol if tot_vol and tot_vol > 0 else None
    return out


MINUTE_DEPENDENT_FEATURES = (
    "d1_open_to_close_return_raw", "d1_low_to_close_recovery",
    "d1_high_to_close_drawdown_raw", "d1_close_location",
    "d1_vwap", "d1_close_to_vwap_raw", "d1_intraday_range",
    "d1_afternoon_return", "d1_last_hour_return",
    "d1_up_bar_volume_ratio", "d1_down_bar_volume_ratio",
    "volume_above_d1_close_ratio", "amount_above_d1_close_ratio",
    "volume_above_break_close_ratio", "high_zone_volume_ratio",
    "high_zone_amount_ratio", "late_day_sell_volume_ratio",
    "late_day_sell_amount_ratio", "down_bar_volume_ratio",
    "d1_vwap_to_close_gap", "d1_amount", "d1_volume",
)

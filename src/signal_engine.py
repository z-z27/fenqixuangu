from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any

import pandas as pd

from .active_money import score_active_money
from .config import StrategyConfig, get_strategy_config
from .graph_quality import score_graph_quality
from .indicators import build_key_zones
from .support_quality import score_support_quality
from .theme_score import score_theme


@dataclass
class Signal:
    trade_date: str
    code: str
    name: str
    d0_date: str
    days_since_d0: int
    consecutive_boards: int
    signal_type: str
    allowed: bool
    position_level: str
    total_score: float
    graph_quality_score: float
    active_money_score: float
    active_cooling_score: float
    support_score: float
    theme_score: float
    trend_hold_score: float
    entry_width_score: float
    low_absorb_width_pct: float | None
    invalid_distance_pct: float | None
    d1_low_ma10_pct: float | None
    d1_close_ma10_pct: float | None
    d1_close_vwap_pct: float | None
    support_type: str
    low_absorb_min: float | None
    low_absorb_max: float | None
    invalid_price: float | None
    key_zones_json: str
    reasons: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_signal(
    code: str,
    name: str,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    limit_up_pool: pd.DataFrame,
    config: StrategyConfig | None = None,
    d0_date: str = "",
) -> Signal:
    cfg = config or get_strategy_config()
    trade_date = str(daily.iloc[-1]["date"]) if daily is not None and not daily.empty else ""

    days_since_d0 = 0
    if d0_date and trade_date:
        days_since_d0 = (pd.Timestamp(trade_date) - pd.Timestamp(d0_date)).days

    consecutive_boards = _count_consecutive_boards(limit_up_pool, code, d0_date)

    graph_score, graph_reasons = score_graph_quality(daily)
    active_score, active_reasons = score_active_money(daily, minute)
    support_score, support_type, support_reasons = score_support_quality(daily, minute)
    theme_score_value, theme_reasons = score_theme(limit_up_pool, code)
    zones = build_key_zones(daily, minute)

    width_pct = _low_absorb_width_pct(zones)
    invalid_distance_pct = _invalid_distance_pct(zones)
    d1_low_ma10_pct = _pct_distance(zones.get("d1_low"), zones.get("ma10"))
    d1_close_ma10_pct = _pct_distance(zones.get("d1_close"), zones.get("ma10"))
    d1_close_vwap_pct = _pct_distance(
        zones.get("d1_close"),
        zones.get("d1_intraday_vwap") or zones.get("d1_vwap"),
    )

    active_cooling = score_active_cooling(active_score)
    trend_hold = score_trend_hold(zones)
    entry_width = score_entry_width(width_pct)
    total = score_total(
        cfg=cfg,
        graph_score=graph_score,
        trend_hold_score=trend_hold,
        active_cooling_score=active_cooling,
        entry_width_score=entry_width,
        theme_score=theme_score_value,
        support_score=support_score,
    )

    hard_blocks = []
    if days_since_d0 <= 0 and d0_date:
        hard_blocks.append("今日仍涨停，等待首次分歧日")
    if days_since_d0 > 3:
        hard_blocks.append(f"涨停后 {days_since_d0} 天，分歧时效已过")

    high_volume_fail = _is_high_volume_fail(daily, cfg)
    if high_volume_fail:
        hard_blocks.append("高位巨量失败风险")

    if hard_blocks:
        allowed = False
        position = "zero"
        signal_type = "WATCH_ONLY"
    else:
        allowed = True
        position = "normal"
        signal_type = "D2_LOW_ABSORB"

    all_reasons = []
    for label, items in (
        ("graph", graph_reasons),
        ("active_raw", active_reasons),
        ("support", support_reasons),
        ("theme", theme_reasons),
    ):
        all_reasons.extend([f"{label}: {item}" for item in items])
    all_reasons.extend(
        [
            f"factor: active_cooling_score={active_cooling:.2f}",
            f"factor: trend_hold_score={trend_hold:.2f}",
            f"factor: entry_width_score={entry_width:.2f}",
            f"factor: low_absorb_width_pct={_format_optional(width_pct)}",
            f"factor: invalid_distance_pct={_format_optional(invalid_distance_pct)}",
            f"factor: d1_low_ma10_pct={_format_optional(d1_low_ma10_pct)}",
            f"factor: d1_close_ma10_pct={_format_optional(d1_close_ma10_pct)}",
            f"factor: d1_close_vwap_pct={_format_optional(d1_close_vwap_pct)}",
        ]
    )
    all_reasons.extend([f"hard_filter: {item}" for item in hard_blocks])

    invalid_price = zones.get("invalid_price") or zones.get("d1_low") or zones.get("ma5")
    return Signal(
        trade_date=trade_date,
        code=code,
        name=name,
        d0_date=d0_date,
        days_since_d0=days_since_d0,
        consecutive_boards=consecutive_boards,
        signal_type=signal_type,
        allowed=allowed,
        position_level=position,
        total_score=round(float(total), 2),
        graph_quality_score=round(float(graph_score), 2),
        active_money_score=round(float(active_score), 2),
        active_cooling_score=round(float(active_cooling), 2),
        support_score=round(float(support_score), 2),
        theme_score=round(float(theme_score_value), 2),
        trend_hold_score=round(float(trend_hold), 2),
        entry_width_score=round(float(entry_width), 2),
        low_absorb_width_pct=_round_optional(width_pct),
        invalid_distance_pct=_round_optional(invalid_distance_pct),
        d1_low_ma10_pct=_round_optional(d1_low_ma10_pct),
        d1_close_ma10_pct=_round_optional(d1_close_ma10_pct),
        d1_close_vwap_pct=_round_optional(d1_close_vwap_pct),
        support_type=support_type,
        low_absorb_min=zones.get("low_absorb_min"),
        low_absorb_max=zones.get("low_absorb_max"),
        invalid_price=invalid_price,
        key_zones_json=json.dumps(zones, ensure_ascii=False),
        reasons="; ".join(all_reasons),
    )


def score_total(
    *,
    cfg: StrategyConfig,
    graph_score: float,
    trend_hold_score: float,
    active_cooling_score: float,
    entry_width_score: float,
    theme_score: float,
    support_score: float,
) -> float:
    """Single clean D1 ranking score used by daily usage and ranking validation."""
    return (
        float(trend_hold_score) * cfg.trend_hold_weight
        + float(graph_score) * cfg.graph_quality_weight
        + float(active_cooling_score) * cfg.active_cooling_weight
        + float(entry_width_score) * cfg.entry_width_weight
        + float(theme_score) * cfg.theme_weight
        + float(support_score) * cfg.support_weight
    )


def score_active_cooling(active_score: float) -> float:
    """Convert raw active-money score into a cooling score.

    Raw activity is useful, but overheated activity should not automatically
    dominate the daily Top3 ranking.
    """
    score = _to_float(active_score)
    if score is None:
        return 60.0
    if score < 50:
        return 65.0
    if score < 70:
        return 90.0
    if score < 80:
        return 85.0
    if score < 85:
        return 70.0
    if score < 90:
        return 50.0
    return 30.0


def score_entry_width(width_pct: float | None) -> float:
    """Score D1 low-absorb width / invalid-distance tightness."""
    if width_pct is None:
        return 50.0
    if width_pct <= 2.0:
        return 90.0
    if width_pct <= 3.5:
        return 65.0
    if width_pct <= 4.5:
        return 60.0
    if width_pct <= 5.0:
        return 45.0
    return 25.0


def score_trend_hold(zones: dict[str, Any]) -> float:
    """Score whether D1 divergence still holds the MA10 trend structure."""
    low_ma10_pct = _pct_distance(zones.get("d1_low"), zones.get("ma10"))
    close_ma10_pct = _pct_distance(zones.get("d1_close"), zones.get("ma10"))

    if low_ma10_pct is not None:
        if low_ma10_pct < 0.0:
            return 40.0
        if low_ma10_pct < 3.0:
            return 55.0
        if low_ma10_pct < 6.0:
            return 65.0
        if low_ma10_pct < 10.0:
            return 75.0
        if low_ma10_pct < 20.0:
            return 95.0
        return 90.0

    if close_ma10_pct is not None and close_ma10_pct >= 0.0:
        return 60.0
    return 40.0


def _low_absorb_width_pct(zones: dict[str, Any]) -> float | None:
    low_min = _to_float(zones.get("low_absorb_min"))
    low_max = _to_float(zones.get("low_absorb_max"))
    base = _to_float(zones.get("prev_close")) or _to_float(zones.get("d1_close"))
    if low_min is None or low_max is None or base is None or base <= 0:
        return None
    return max(0.0, (low_max - low_min) / base * 100.0)


def _invalid_distance_pct(zones: dict[str, Any]) -> float | None:
    invalid = _to_float(zones.get("invalid_price"))
    close = _to_float(zones.get("d1_close"))
    if invalid is None or close is None or close <= 0:
        return None
    return max(0.0, (close - invalid) / close * 100.0)


def _pct_distance(value: Any, base: Any) -> float | None:
    value_float = _to_float(value)
    base_float = _to_float(base)
    if value_float is None or base_float is None or base_float == 0:
        return None
    return (value_float / base_float - 1.0) * 100.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{float(value):.2f}"


def _count_consecutive_boards(limit_up_pool: pd.DataFrame, code: str, d0_date: str) -> int:
    if not d0_date or limit_up_pool is None or limit_up_pool.empty:
        return 0
    code_pool = limit_up_pool[limit_up_pool["code"].astype(str) == code]
    dates = sorted(code_pool["trade_date"].dropna().astype(str).unique().tolist())
    if not dates or d0_date not in dates:
        return 0
    d0_idx = dates.index(d0_date)
    count = 1
    for i in range(d0_idx - 1, -1, -1):
        gap = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
        if gap <= 2:
            count += 1
        else:
            break
    return count


def _is_high_volume_fail(daily: pd.DataFrame, cfg: StrategyConfig) -> bool:
    if daily is None or daily.empty:
        return False
    d1 = daily.iloc[-1]
    raw_ar = d1.get("amount_ratio")
    amount_ratio = float(raw_ar) if pd.notna(raw_ar) else 0.0
    raw_cp = d1.get("close_position")
    close_position = float(raw_cp) if pd.notna(raw_cp) else 0.0
    vwap = d1.get("vwap_daily")
    close = d1.get("close")
    vwap_dev = 0.0
    if pd.notna(vwap) and pd.notna(close) and float(vwap) != 0:
        vwap_dev = float(close) / float(vwap) - 1
    return (
        amount_ratio >= 2.0
        and close_position <= cfg.high_volume_fail_close_pos
        and vwap_dev <= cfg.high_volume_fail_vwap_dev
    )

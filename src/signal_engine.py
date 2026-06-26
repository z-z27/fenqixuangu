from __future__ import annotations

from dataclasses import dataclass, asdict
import json

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
    support_score: float
    theme_score: float
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
    total = (
        graph_score * cfg.graph_quality_weight
        + active_score * cfg.active_money_weight
        + support_score * cfg.support_weight
        + theme_score_value * cfg.theme_weight
    )
    zones = build_key_zones(daily, minute)

    hard_blocks = []
    if days_since_d0 <= 0 and d0_date:
        hard_blocks.append("今日仍涨停，等待分歧日")
    if days_since_d0 > 3:
        hard_blocks.append(f"涨停后 {days_since_d0} 天，分歧时效已过")
    if graph_score < cfg.min_graph_quality_trade:
        hard_blocks.append("图形趋势分不足")
    if active_score < cfg.min_active_money:
        hard_blocks.append("活跃资金分不足")
    if support_score < cfg.weak_support_min:
        hard_blocks.append("承接分过低")
    if support_type == "C":
        hard_blocks.append("无效承接")

    high_volume_fail = _is_high_volume_fail(daily, cfg)
    if high_volume_fail:
        hard_blocks.append("高位巨量失败风险")

    allowed = False
    if hard_blocks:
        position = "zero"
        signal_type = "WATCH_ONLY"
    elif support_type == "A" and support_score >= cfg.min_support_trade and total >= 70:
        allowed = True
        position = "normal"
        signal_type = "D2_LOW_ABSORB"
    elif support_score >= cfg.weak_support_min:
        allowed = True
        position = "small"
        signal_type = "D2_WATCH_OR_SMALL"
    else:
        position = "zero"
        signal_type = "WATCH_ONLY"

    all_reasons = []
    for label, items in (
        ("graph", graph_reasons),
        ("active", active_reasons),
        ("support", support_reasons),
        ("theme", theme_reasons),
    ):
        all_reasons.extend([f"{label}: {item}" for item in items])
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
        support_score=round(float(support_score), 2),
        theme_score=round(float(theme_score_value), 2),
        support_type=support_type,
        low_absorb_min=zones.get("low_absorb_min"),
        low_absorb_max=zones.get("low_absorb_max"),
        invalid_price=invalid_price,
        key_zones_json=json.dumps(zones, ensure_ascii=False),
        reasons="; ".join(all_reasons),
    )


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

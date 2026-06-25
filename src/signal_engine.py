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
) -> Signal:
    cfg = config or get_strategy_config()
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

    allowed = not hard_blocks and support_score >= cfg.min_support_trade
    if allowed and support_type == "A" and total >= 70:
        position = "normal"
        signal_type = "D2_LOW_ABSORB"
    elif not hard_blocks and support_type == "B":
        position = "small"
        signal_type = "D2_WATCH_OR_SMALL"
    else:
        position = "zero"
        signal_type = "WATCH_ONLY"
        allowed = False

    all_reasons = []
    for label, items in (
        ("graph", graph_reasons),
        ("active", active_reasons),
        ("support", support_reasons),
        ("theme", theme_reasons),
    ):
        all_reasons.extend([f"{label}: {item}" for item in items])
    all_reasons.extend([f"hard_filter: {item}" for item in hard_blocks])

    trade_date = str(daily.iloc[-1]["date"]) if daily is not None and not daily.empty else ""
    invalid_price = zones.get("d1_low") or zones.get("ma5")
    return Signal(
        trade_date=trade_date,
        code=code,
        name=name,
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


def _is_high_volume_fail(daily: pd.DataFrame, cfg: StrategyConfig) -> bool:
    if daily is None or daily.empty:
        return False
    d1 = daily.iloc[-1]
    amount_ratio = float(d1.get("amount_ratio", 0) or 0)
    close_position = float(d1.get("close_position", 0) or 0)
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

"""Isolated replay of frozen results produced before the trading-session repair.

Only frozen V4C reconstruction scripts should call this module. New signals,
historical samples, and backtests must use ``signal_engine.generate_signal``.
"""

from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .signal_engine import Signal, generate_signal


class _LegacyCalendarDayDistance:
    """The old D0 age formula, retained solely for historical reproducibility."""

    def session_distance(self, start: str, end: str) -> int:
        return int((pd.Timestamp(end) - pd.Timestamp(start)).days)


def _legacy_consecutive_boards(limit_up_pool: pd.DataFrame, code: str, d0_date: str) -> int:
    if not d0_date or limit_up_pool is None or limit_up_pool.empty:
        return 0
    code_pool = limit_up_pool[limit_up_pool["code"].astype(str) == str(code)]
    dates = sorted(code_pool["trade_date"].dropna().astype(str).unique().tolist())
    if d0_date not in dates:
        return 0
    index = dates.index(d0_date)
    count = 1
    for earlier in range(index - 1, -1, -1):
        gap = int((pd.Timestamp(dates[earlier + 1]) - pd.Timestamp(dates[earlier])).days)
        if gap > 2:
            break
        count += 1
    return count


def generate_legacy_calendar_signal(
    code: str,
    name: str,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    limit_up_pool: pd.DataFrame,
    config: StrategyConfig | None = None,
    d0_date: str = "",
) -> Signal:
    """Reproduce the old calendar-day signal fields for frozen baseline audits."""
    return generate_signal(
        code=code,
        name=name,
        daily=daily,
        minute=minute,
        limit_up_pool=limit_up_pool,
        config=config,
        d0_date=d0_date,
        trading_calendar=_LegacyCalendarDayDistance(),  # type: ignore[arg-type]
        consecutive_boards_override=_legacy_consecutive_boards(limit_up_pool, code, d0_date),
    )

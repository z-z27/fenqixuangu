from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .logistic_v003 import _mean, _safe_rate


def validate_threshold_value(name: str, value: float) -> None:
    if name == "threshold-grid" and not (0.0 <= value <= 1.0):
        raise RuntimeError(f"{name} values must be between 0.0 and 1.0: {value}")


def build_top3_combo_summary_with_underfill(per_date: pd.DataFrame, meta_columns: list[str], target_return_pct: float, top_n: int) -> pd.DataFrame:
    if per_date.empty:
        return pd.DataFrame()
    columns = [
        *meta_columns,
        "signal_date",
        "selected_count",
        "hit_count",
        "all_hit",
        "avg_high_return",
        "avg_realized_return",
        "portfolio_realized_positive",
        "portfolio_realized_hit7",
    ]
    daily = per_date[columns].drop_duplicates(subset=[*meta_columns, "signal_date"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for group_key, group in daily.groupby(meta_columns, dropna=False):
        meta = dict(zip(meta_columns, group_key))
        selected_count = pd.to_numeric(group["selected_count"], errors="coerce").fillna(0)
        hit_count = pd.to_numeric(group["hit_count"], errors="coerce").fillna(0)
        date_count = int(len(group))
        selected_ticket_count = int(selected_count.sum())
        hit_count_total = int(hit_count.sum())
        no_trade_days = int((selected_count == 0).sum())
        underfilled_days = int((selected_count < int(top_n)).sum())
        avg_selected = _mean(selected_count)
        rows.append(
            {
                **meta,
                "date_count": date_count,
                "selected_ticket_count": selected_ticket_count,
                "top3_target_rate": _safe_rate(hit_count_total, selected_ticket_count),
                "top3_all_hit_rate": _safe_rate(int(group["all_hit"].astype(bool).sum()), date_count),
                "hit_count_0_days": int((hit_count == 0).sum()),
                "hit_count_1_days": int((hit_count == 1).sum()),
                "hit_count_2_days": int((hit_count == 2).sum()),
                "hit_count_3_days": int((hit_count == 3).sum()),
                "avg_top3_high_return": _mean(pd.to_numeric(group["avg_high_return"], errors="coerce")),
                "avg_top3_realized_return": _mean(pd.to_numeric(group["avg_realized_return"], errors="coerce")),
                "portfolio_realized_positive_rate": _safe_rate(int(group["portfolio_realized_positive"].astype(bool).sum()), date_count),
                "portfolio_realized_hit7_rate": _safe_rate(int(group["portfolio_realized_hit7"].astype(bool).sum()), date_count),
                "selected_count_per_day": avg_selected,
                "avg_selected_count_per_day": avg_selected,
                "skip_day_rate": _safe_rate(underfilled_days, date_count),
                "no_trade_days": no_trade_days,
                "no_trade_day_rate": _safe_rate(no_trade_days, date_count),
                "underfilled_days": underfilled_days,
                "underfilled_day_rate": _safe_rate(underfilled_days, date_count),
            }
        )
    return pd.DataFrame(rows).sort_values(meta_columns).reset_index(drop=True)

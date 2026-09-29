from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from src.v004c_5m_conservative_seal_path_coverage_audit import (
    EXPECTED_BAR_CLOCKS,
    PRICE_TOLERANCE,
    DailyPriceStore,
    MinuteStore,
    build_candidate_outputs,
    build_coverage,
    classify_bar_states,
    determine_data_state,
    reconstruct_day,
    round_main_board_limit_price,
    trading_minute_from_clock,
    _validate_minute_source,
)


def _full_day(code: str, date: str, states: list[str], limit: float = 11.0) -> pd.DataFrame:
    rows = []
    for clock, state in zip(EXPECTED_BAR_CLOCKS, states, strict=True):
        if state == "LOCKED":
            low = high = open_price = close = limit
        elif state == "TOUCHED_MIXED":
            low, high, open_price, close = limit - 0.10, limit, limit - 0.05, limit
        else:
            low, high, open_price, close = limit - 0.20, limit - 0.10, limit - 0.15, limit - 0.12
        rows.append(
            {
                "datetime": pd.Timestamp(f"{date} {clock}"),
                "trade_date": date,
                "time": clock,
                "code": code,
                "market": "SH",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0,
                "amount": 1000.0,
                "pct_chg": math.nan,
                "change": math.nan,
                "amplitude": math.nan,
                "turnover_rate": math.nan,
                "source": "baostock_5m",
                "adjust": "none",
                "interval": "5m",
            }
        )
    return pd.DataFrame(rows)


class ConservativeSealPathTest(unittest.TestCase):
    def test_limit_price_uses_cent_half_up(self):
        self.assertEqual(round_main_board_limit_price(9.15), 10.07)
        self.assertEqual(round_main_board_limit_price(10.05), 11.06)

    def test_trading_minute_removes_lunch(self):
        self.assertEqual(trading_minute_from_clock("09:35:00"), 5.0)
        self.assertEqual(trading_minute_from_clock("11:30:00"), 120.0)
        self.assertEqual(trading_minute_from_clock("13:05:00"), 125.0)
        self.assertEqual(trading_minute_from_clock("15:00:00"), 240.0)

    def test_bar_states_are_conservative(self):
        frame = pd.DataFrame(
            {
                "high": [10.0, 10.0, 9.98],
                "low": [10.0, 9.8, 9.7],
            }
        )
        states = classify_bar_states(frame, 10.0).tolist()
        self.assertEqual(states, ["LOCKED", "TOUCHED_MIXED", "OFF_LIMIT"])
        self.assertEqual(PRICE_TOLERANCE, 0.011)

    def test_legacy_cache_without_interval_marker_uses_exact_grid_proof(self):
        frame = _full_day("600000", "2026-05-05", ["LOCKED"] * 48).drop(columns=["interval"])
        frame["source"] = "sina_5m"
        qa = _validate_minute_source(frame, frozenset({"sina_5m"}))
        self.assertEqual(qa["source_semantics_valid"], 1)
        self.assertEqual(qa["expected_grid_match"], 1)
        self.assertEqual(qa["reason"], "")

    def test_final_suffix_and_confirmed_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/cache/baostock_5m").mkdir(parents=True)
            (root / "data/cache/daily_unadjusted").mkdir(parents=True)
            states = ["OFF_LIMIT"] * 10 + ["LOCKED"] * 5 + ["TOUCHED_MIXED"] + ["OFF_LIMIT"] * 2 + ["LOCKED"] * 30
            minute = _full_day("600000", "2026-05-05", states)
            minute.to_pickle(root / "data/cache/baostock_5m/600000_5min.pkl")
            daily = pd.DataFrame(
                [
                    {"date": "2026-05-04", "close": 10.0, "high": 10.1, "low": 9.9, "open": 10.0, "source": "sina_daily"},
                    {"date": "2026-05-05", "close": 11.0, "high": 11.0, "low": 10.7, "open": 10.8, "source": "sina_daily"},
                ]
            )
            daily.to_pickle(root / "data/cache/daily_unadjusted/600000_daily.pkl")
            audit = reconstruct_day(root, "600000", "2026-05-05", MinuteStore(root), DailyPriceStore(root))
            self.assertEqual(audit.day_complete, 1)
            self.assertEqual(audit.f1_valid, 1)
            self.assertEqual(audit.stable_suffix_bars, 30)
            self.assertEqual(audit.f1_clock_time, EXPECTED_BAR_CLOCKS[18])
            self.assertEqual(audit.f2_valid, 1)
            self.assertEqual(audit.confirmed_reopen, 1.0)

    def test_touched_only_has_no_f1_or_f2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/cache/baostock_5m").mkdir(parents=True)
            (root / "data/cache/daily_unadjusted").mkdir(parents=True)
            states = ["OFF_LIMIT"] * 47 + ["TOUCHED_MIXED"]
            minute = _full_day("600000", "2026-05-05", states)
            # Board-day close must remain at limit for price-lineage validity.
            minute.loc[minute.index[-1], "close"] = 11.0
            minute.to_pickle(root / "data/cache/baostock_5m/600000_5min.pkl")
            daily = pd.DataFrame(
                [
                    {"date": "2026-05-04", "close": 10.0, "high": 10.1, "low": 9.9, "open": 10.0, "source": "sina_daily"},
                    {"date": "2026-05-05", "close": 11.0, "high": 11.0, "low": 10.7, "open": 10.8, "source": "sina_daily"},
                ]
            )
            daily.to_pickle(root / "data/cache/daily_unadjusted/600000_daily.pkl")
            audit = reconstruct_day(root, "600000", "2026-05-05", MinuteStore(root), DailyPriceStore(root))
            self.assertEqual(audit.day_complete, 1)
            self.assertEqual(audit.f1_valid, 0)
            self.assertEqual(audit.f2_valid, 0)
            self.assertEqual(audit.missing_reason, "NO_FULL_LOCK_BAR")

    def test_incomplete_day_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/cache/baostock_5m").mkdir(parents=True)
            (root / "data/cache/daily_unadjusted").mkdir(parents=True)
            _full_day("600000", "2026-05-05", ["LOCKED"] * 48).iloc[:-1].to_pickle(
                root / "data/cache/baostock_5m/600000_5min.pkl"
            )
            pd.DataFrame(
                [
                    {"date": "2026-05-04", "close": 10.0, "high": 10.1, "low": 9.9, "open": 10.0, "source": "sina_daily"},
                    {"date": "2026-05-05", "close": 11.0, "high": 11.0, "low": 11.0, "open": 11.0, "source": "sina_daily"},
                ]
            ).to_pickle(root / "data/cache/daily_unadjusted/600000_daily.pkl")
            audit = reconstruct_day(root, "600000", "2026-05-05", MinuteStore(root), DailyPriceStore(root))
            self.assertEqual(audit.day_complete, 0)
            self.assertEqual(audit.missing_reason, "PARTIAL_5M_DAY")

    def test_f3_is_d0_minus_previous_trading_minutes(self):
        population = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "code": "600000",
                    "signal_date": "2026-05-06",
                    "month": "MAY",
                    "board_group": "BOARD2",
                    "d1_date": "2026-05-06",
                    "d0_date": "2026-05-05",
                    "prev_board_day": "2026-05-04",
                }
            ]
        )
        common = {
            "cache_kind": "baostock_5m", "cache_path": "x", "cache_sources": "baostock_5m",
            "adjustment": "none", "interval": "5m", "bar_count": 48, "first_bar_time": "09:35:00",
            "last_bar_time": "15:00:00", "duplicate_bar_count": 0, "timestamp_order_valid": 1,
            "expected_grid_match": 1, "ohlc_valid": 1, "volume_amount_valid": 1,
            "source_semantics_valid": 1, "daily_cache_kind": "daily_unadjusted", "daily_cache_path": "d",
            "daily_source": "sina_daily", "previous_trade_date": "x", "previous_close": 10.0,
            "daily_close": 11.0, "daily_high": 11.0, "limit_price": 11.0, "limit_price_valid": 1,
            "daily_minute_price_match": 1, "day_complete": 1, "locked_bar_count": 10,
            "touched_mixed_bar_count": 0, "off_limit_bar_count": 38, "uncertain_bar_count": 0,
            "f1_valid": 1, "stable_suffix_bars": 10, "late_single_bar_lock": 0, "f2_valid": 1,
            "confirmed_reopen": 0.0, "first_locked_clock_time": "10:00:00", "missing_reason": "",
            "rejected_source_notes": "",
        }
        days = pd.DataFrame(
            [
                {"code": "600000", "board_date": "2026-05-05", "f1_clock_time": "14:00:00", "f1_trading_minute": 180.0, **common},
                {"code": "600000", "board_date": "2026-05-04", "f1_clock_time": "11:00:00", "f1_trading_minute": 90.0, **common},
            ]
        )
        _, values = build_candidate_outputs(population, days)
        self.assertEqual(float(values.iloc[0]["F3_D0_VS_PREV_LOCK_DETERIORATION"]), 90.0)

    def test_state_gate_requires_all_three_for_ready(self):
        rows = []
        for variable, rate in (("F1", 1.0), ("F2", 1.0), ("F3", 0.65)):
            rows.append({"variable": variable, "scope_type": "OVERALL", "scope_value": "ALL", "coverage_rate": rate})
            for month in ("MAY", "JUNE", "JULY"):
                rows.append({"variable": variable, "scope_type": "MONTH", "scope_value": month, "coverage_rate": 0.55 if variable == "F3" else 1.0})
            for board in ("BOARD2", "BOARD3"):
                rows.append({"variable": variable, "scope_type": "BOARD", "scope_value": board, "coverage_rate": 0.60 if variable == "F3" else 1.0})
        coverage = pd.DataFrame(rows)
        self.assertEqual(determine_data_state(coverage, {"F1": True, "F2": True, "F3": False}), "5M_SEAL_PATH_DATA_PARTIAL")
        self.assertEqual(determine_data_state(coverage, {"F1": False, "F2": True, "F3": False}), "5M_SEAL_PATH_DATA_NOT_READY")


if __name__ == "__main__":
    unittest.main()

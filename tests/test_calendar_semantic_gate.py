"""CORRECTED_DATASET_GENERATION_GATE (Gate A) —— 离线确定性检查。

这个网关只回答一个问题: 在 ``trading_sessions_v1`` 语义下, 生成一份 corrected
historical dataset 所需的**每一处日期决策**, 是否都能被权威日历证明, 且没有任何
一处退回 weekday 近似或「本地缓存里没有 = 休市」的近似?

检查项对应 Phase 0.5 §8 的九条:

    A1 日历权威性 (覆盖声明 / 无 weekday 近似 / 无缓存缺席推断)
    A2 session 语义 (session_distance, previous/next_trading_day,
       are_consecutive_sessions, trading_days 互相一致)
    A3 D0 年龄按 market session
    A4 板连续性跨周末 / 单日假期 / 长假
    A5 历史信号日枚举不等于 weekday≈交易日
    A6 未来 horizon D2/D3/D5/D10 按 market session 身份
    A7 缺 session -> target=null, candidate_evaluable=False
    A8 null target 在 v004a 训练前被排除
    A9 provenance 写入 ``calendar_semantics_version=trading_sessions_v1``

全部离线、确定性: 日历通过 ``fetchers`` 注入, 行情通过假 provider / 假缓存注入,
不触网、不写 corrected namespace、不训练任何模型。

日历夹具与 ``tests/test_trading_calendar.py`` 使用同一份权威交易日表
(2026-09-25 中秋休市, 10-01..10-07 国庆休市, **10-08 是交易日**)。
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

from src.backtester import (
    _exact_consecutive_boards,
    _future_trade_dates,
    _input_calendar_semantics_namespace,
    _iter_trading_days,
    _next_trade_date,
    _path_metrics_by_horizon,
    _covered_limit_up_codes,
    build_signals_for_pool,
)
from src.config import DataConfig
from src.history_samples import (
    _write_history_universe_outputs,
    evaluate_history_candidate_only,
)
from src.legacy_calendar_semantics import (
    _LegacyCalendarDayDistance,
    _legacy_consecutive_boards,
)
from src.trading_calendar import (
    CALENDAR_SEMANTICS_VERSION,
    CACHED_DAILY_SOURCE,
    CalendarFetch,
    TradingCalendar,
    TradingCalendarError,
)
from src.v004a import annotate_v004a_input_eligibility, prepare_v004a_samples

# 权威交易日表 (2026-09-01..2026-10-16), 与 test_trading_calendar 同一份。
_GATE_SESSIONS = [
    "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
    "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
    "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
    "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
    "2026-09-28", "2026-09-29", "2026-09-30",
    "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13",
    "2026-10-14", "2026-10-15", "2026-10-16",
]
_GATE_COVERAGE = (_GATE_SESSIONS[0], _GATE_SESSIONS[-1])

# 修复前被误当成交易日的非交易日 (全部是 weekday)。
_HOLIDAYS = ["2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05",
             "2026-10-06", "2026-10-07"]


def _config(root: Path) -> DataConfig:
    config = DataConfig(
        raw_dir=root / "raw",
        cache_dir=root / "cache",
        processed_dir=root / "processed",
        reports_dir=root / "reports",
        snapshot_dir=root / "snapshots",
    )
    config.ensure_directories()
    return config


def _calendar(config: DataConfig, *, coverage=_GATE_COVERAGE, dates=None) -> TradingCalendar:
    """注入一份**声明了覆盖区间**的权威日历 (仅正面/负面证据都可给出)。"""
    payload = CalendarFetch(
        dates=tuple(_GATE_SESSIONS if dates is None else dates),
        coverage_start=None if coverage is None else coverage[0],
        coverage_end=None if coverage is None else coverage[1],
    )
    return TradingCalendar(
        config, sources=("gate_source",), fetchers={"gate_source": lambda: payload}
    )


def _signal(signal_date: str, code: str = "600000") -> pd.Series:
    return pd.Series({
        "trade_date": signal_date,
        "requested_signal_date": signal_date,
        "code": code,
        "allowed": True,
        "signal_type": "D2_LOW_ABSORB",
        "key_zones_json": json.dumps({"d1_close": 10.0}),
    })


def _minute(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": [row[0] for row in rows],
        "datetime": [f"{row[0]} 09:35:00" for row in rows],
        "open": [row[1] for row in rows],
        "high": [row[2] for row in rows],
        "low": [row[3] for row in rows],
        "close": [row[4] for row in rows],
    })


def _service(calendar: TradingCalendar, *, minute=None, daily=None) -> mock.Mock:
    service = mock.Mock(trading_calendar=calendar)
    service.minute_cache.read.return_value = minute
    if daily is None:
        service.daily_cache.read.return_value = None
    else:
        service.daily_cache.read.return_value = pd.DataFrame({"date": list(daily)})
    return service


class GateACalendarAuthorityTest(unittest.TestCase):
    """A1 —— 权威性: 覆盖声明 / 无 weekday 近似 / 无缓存缺席推断。"""

    def test_gate_a1a_every_date_in_the_rebuild_range_is_provable(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            for date in pd.date_range(_GATE_COVERAGE[0], _GATE_COVERAGE[1]).strftime("%Y-%m-%d"):
                expected = date in set(_GATE_SESSIONS)
                self.assertEqual(calendar.is_trading_day(date), expected, date)
            self.assertEqual(calendar.coverage, _GATE_COVERAGE)

    def test_gate_a1b_no_weekday_approximation(self) -> None:
        """A1 的核心: 覆盖区间内的 weekday 休市日必须判 CLOSED。"""
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            for holiday in _HOLIDAYS:
                self.assertLess(pd.Timestamp(holiday).weekday(), 5, holiday)
                self.assertFalse(calendar.is_trading_day(holiday), holiday)
            # 长假后第一个交易日是 10-08 —— 旧的 5 自然日/工作日近似都到不了这一天。
            self.assertTrue(calendar.is_trading_day("2026-10-08"))

    def test_gate_a1c_without_coverage_the_gate_fails_closed(self) -> None:
        """没有覆盖声明就没有 CLOSED 权 —— 必须报错, 不能退回近似。"""
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)), coverage=None)
            for date in ("2026-09-28", "2026-09-25", "2026-10-01"):
                with self.subTest(date=date):
                    with self.assertRaises(TradingCalendarError):
                        calendar.is_trading_day(date)
            self.assertFalse(calendar.closure_authority)

    def test_gate_a1d_local_daily_cache_absence_cannot_close_a_date(self) -> None:
        """A1 §C01: 本地日线缓存只能给正面 OPEN 证据。"""
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            daily_dir = config.cache_dir / "daily"
            daily_dir.mkdir(parents=True, exist_ok=True)
            for index in range(60):
                pd.DataFrame({"date": list(_GATE_SESSIONS)}).to_pickle(
                    daily_dir / f"{600000 + index}_daily.pkl"
                )
            # 1) 不能被选为日历源。
            with self.assertRaises(TradingCalendarError) as ctx:
                TradingCalendar(config, sources=(CACHED_DAILY_SOURCE,)).is_trading_day("2026-09-28")
            self.assertIn(CACHED_DAILY_SOURCE, str(ctx.exception))
            # 2) 但作为诊断它仍然能报出「哪几天确实开过市」。
            diagnostic = TradingCalendar(config)
            self.assertIn("2026-09-24", diagnostic.cached_daily_open_sessions("2026-09-24", "2026-09-30"))
            self.assertNotIn("2026-09-25", diagnostic.cached_daily_open_sessions("2026-09-24", "2026-09-30"))


class GateASessionSemanticsTest(unittest.TestCase):
    """A2 —— 五个 session 原语互相一致。"""

    def test_gate_a2_session_primitives_are_mutually_consistent(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            sessions = calendar.trading_days(_GATE_COVERAGE[0], _GATE_COVERAGE[1])
            self.assertEqual(sessions, _GATE_SESSIONS)

            # is_trading_day 与枚举一致。
            for date in pd.date_range(_GATE_COVERAGE[0], _GATE_COVERAGE[1]).strftime("%Y-%m-%d"):
                self.assertEqual(calendar.is_trading_day(date), date in set(sessions), date)

            # previous/next 与枚举顺序一致。
            for earlier, later in zip(sessions, sessions[1:]):
                self.assertEqual(calendar.next_trading_day(earlier), later)
                self.assertEqual(calendar.previous_trading_day(later), earlier)
                self.assertTrue(calendar.are_consecutive_sessions(earlier, later))
                self.assertEqual(calendar.session_distance(earlier, later), 1)

            # session_distance == 枚举下标差, are_consecutive_sessions <=> 距离 1。
            for index, start in enumerate(sessions):
                for offset in (1, 2, 3, len(sessions) - 1 - index):
                    if index + offset >= len(sessions):
                        continue
                    end = sessions[index + offset]
                    with self.subTest(start=start, end=end):
                        self.assertEqual(calendar.session_distance(start, end), offset)
                        self.assertEqual(
                            calendar.are_consecutive_sessions(start, end), offset == 1
                        )

            # 非交易日不是任何 session 的前/后一个交易日。
            self.assertEqual(calendar.previous_trading_day("2026-10-08"), "2026-09-30")
            self.assertEqual(calendar.next_trading_day("2026-09-30"), "2026-10-08")


class GateAD0AgeTest(unittest.TestCase):
    """A3 —— D0 年龄按 market session。"""

    def test_gate_a3_d0_age_uses_market_sessions(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            for signal_date, d0_date, expected in (
                ("2026-09-21", "2026-09-18", 1),   # Friday -> Monday
                ("2026-09-21", "2026-09-17", 2),   # Thursday -> Monday
                ("2026-09-21", "2026-09-16", 3),   # Wednesday -> Monday
                ("2026-09-28", "2026-09-24", 1),   # 中秋 + 周末
                ("2026-10-08", "2026-09-30", 1),   # 国庆长假后第一个交易日
                ("2026-09-30", "2026-09-30", 0),
            ):
                with self.subTest(signal_date=signal_date, d0_date=d0_date):
                    self.assertEqual(
                        calendar.session_distance(d0_date, signal_date), expected
                    )
            # 自然日公式会把这些全算错 —— 逐条钉住差异。
            legacy = _LegacyCalendarDayDistance()
            self.assertEqual(legacy.session_distance("2026-09-30", "2026-10-08"), 8)
            self.assertEqual(calendar.session_distance("2026-09-30", "2026-10-08"), 1)


class GateABoardContinuityTest(unittest.TestCase):
    """A4 —— 板连续性跨周末 / 单日假期 / 长假。"""

    def _pool(self, dates: list[str], covered: list[str], code: str = "600000") -> pd.DataFrame:
        pool = pd.DataFrame({"code": [code] * len(dates), "trade_date": dates})
        pool.attrs["trading_dates_covered"] = tuple(covered)
        return pool

    def test_gate_a4_weekend_and_single_holiday(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            service = mock.Mock(trading_calendar=calendar)
            service.collect_limit_ups.return_value = pd.DataFrame(columns=["trade_date", "code"])

            # 周五 + 周一 = 2 板。
            friday_monday = self._pool(
                ["2026-09-18", "2026-09-21"], ["2026-09-17", "2026-09-18", "2026-09-21"]
            )
            self.assertEqual(
                _exact_consecutive_boards(
                    service, "600000", "2026-09-21", _covered_limit_up_codes(friday_monday)
                ),
                2,
            )
            # 周五 + 周一 + 周二 = 3 板。
            three = self._pool(
                ["2026-09-18", "2026-09-21", "2026-09-22"],
                ["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22"],
            )
            self.assertEqual(
                _exact_consecutive_boards(
                    service, "600000", "2026-09-22", _covered_limit_up_codes(three)
                ),
                3,
            )
            # 单日假期 (中秋 09-25): 09-24 与 09-28 仍然是连续 session。
            mid_autumn = self._pool(
                ["2026-09-24", "2026-09-28"],
                ["2026-09-23", "2026-09-24", "2026-09-28"],
            )
            self.assertEqual(
                _exact_consecutive_boards(
                    service, "600000", "2026-09-28", _covered_limit_up_codes(mid_autumn)
                ),
                2,
            )

    def test_gate_a4_long_holiday(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            service = mock.Mock(trading_calendar=calendar)
            service.collect_limit_ups.return_value = pd.DataFrame(columns=["trade_date", "code"])

            # 长假前最后一个交易日 + 长假后第一个交易日 = 2 板 (10-08, 不是 10-09)。
            long_holiday = self._pool(
                ["2026-09-30", "2026-10-08"],
                ["2026-09-29", "2026-09-30", "2026-10-08"],
            )
            self.assertEqual(
                _exact_consecutive_boards(
                    service, "600000", "2026-10-08", _covered_limit_up_codes(long_holiday)
                ),
                2,
            )
            # 09-30 与 10-09 之间隔着 10-08 -> 不连续, 不能算 2 板。
            skipped = self._pool(
                ["2026-09-30", "2026-10-09"],
                ["2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09"],
            )
            self.assertEqual(
                _exact_consecutive_boards(
                    service, "600000", "2026-10-09", _covered_limit_up_codes(skipped)
                ),
                1,
            )

    def test_gate_a4_legacy_formula_would_disagree(self) -> None:
        """钉住差异: 自然日 gap<=2 会把长假切断, 也会把 09-30/10-09 误连。"""
        long_holiday = pd.DataFrame({
            "code": ["600000", "600000"], "trade_date": ["2026-09-30", "2026-10-08"],
        })
        self.assertEqual(_legacy_consecutive_boards(long_holiday, "600000", "2026-10-08"), 1)


class GateASignalDateEnumerationTest(unittest.TestCase):
    """A5 —— 历史信号日枚举不等于 weekday≈交易日。"""

    def test_gate_a5_signal_dates_are_sessions_not_weekdays(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            start, end = "2026-09-21", "2026-10-09"
            enumerated = _iter_trading_days(start, end, calendar)
            weekdays = pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()

            self.assertEqual(enumerated, [d for d in _GATE_SESSIONS if start <= d <= end])
            self.assertNotEqual(enumerated, weekdays)
            # 差集必须恰好是那批 weekday 休市日, 一条不多一条不少。
            self.assertEqual(
                sorted(set(weekdays) - set(enumerated)),
                [d for d in _HOLIDAYS if start <= d <= end],
            )
            # 长假后第一个交易日必须被枚举到 —— 否则整段 10 月样本都会错位。
            self.assertIn("2026-10-08", enumerated)
            self.assertNotIn("2026-10-01", enumerated)


class GateAFutureHorizonTest(unittest.TestCase):
    """A6 —— 未来 horizon D2/D3/D5/D10 按 market session 身份。"""

    def test_gate_a6_horizons_keep_market_session_identity(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            signal_date = "2026-09-18"
            # 个股分钟线缺 09-21 (D2) 的 bar。
            minute = _minute([
                ("2026-09-18", 10.0, 10.0, 10.0, 10.0),
                ("2026-09-22", 11.0, 12.0, 10.5, 11.5),
            ])
            future = _future_trade_dates(minute, signal_date, calendar)
            # D2 仍然是 09-21 —— 补齐到最后一个已观测日期为止的市场 session。
            self.assertEqual(
                future, [d for d in _GATE_SESSIONS if signal_date < d <= "2026-09-22"]
            )
            self.assertEqual(future[0], "2026-09-21")
            self.assertEqual(_next_trade_date(minute, signal_date, calendar), "2026-09-21")

            # D2/D3/D5/D10 的日期身份来自日历, 与个股有没有 bar 无关。
            d_dates = {"D1": "2026-09-21", "D2": "2026-09-22", "D3": "2026-09-23",
                       "D5": "2026-09-28", "D10": "2026-10-12"}
            for label, expected in d_dates.items():
                with self.subTest(horizon=label):
                    index = {"D1": 0, "D2": 1, "D3": 2, "D5": 4, "D10": 9}[label]
                    self.assertEqual(
                        _iter_trading_days("2026-09-21", "2026-10-16", calendar)[index],
                        expected,
                    )
            # _path_metrics_by_horizon 按 (2,3,5,10) 出键, 跨长假也照样能取到 D10。
            full = _minute([
                (d, 10.0, 11.0, 9.5, 10.5)
                for d in _GATE_SESSIONS if "2026-09-18" < d <= "2026-10-09"
            ])
            future_full = _future_trade_dates(full, signal_date, calendar)
            self.assertEqual(
                future_full, [d for d in _GATE_SESSIONS if "2026-09-18" < d <= "2026-10-09"]
            )
            # 最后一个有 bar 的 session 是 10-09; horizon 身份完全由日历位置决定,
            # 缺 bar 不会把后面的 session 提前顶上来。
            self.assertEqual(future_full[-1], "2026-10-09")
            metrics = _path_metrics_by_horizon(
                full, future_full, 10.0, prefix="candidate", hold_days=10
            )
            self.assertEqual(
                {k for k in metrics if k.endswith("_max_return_pct")},
                {f"candidate_d{h}_max_return_pct" for h in (2, 3, 5, 10)},
            )
            # 缺 D2 时, 所有 horizon 的累计收益一起消失 —— 不允许用后面的 bar 顶替。
            sparse_metrics = _path_metrics_by_horizon(
                _minute([
                    (d, 10.0, 11.0, 9.5, 10.5)
                    for d in _GATE_SESSIONS if "2026-09-21" < d <= "2026-10-09"
                ]),
                future_full, 10.0, prefix="candidate", hold_days=10,
            )
            self.assertEqual(sparse_metrics, {})


class GateALabelCensoringTest(unittest.TestCase):
    """A7 —— 缺 session -> target=null, candidate_evaluable=False。"""

    def test_gate_a7_missing_session_nulls_the_target(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            # 日线跨越 09-21 却没有这一天的 bar -> 个股停牌。
            suspended = _service(
                calendar,
                minute=_minute([
                    ("2026-09-18", 10.0, 10.0, 10.0, 10.0),
                    ("2026-09-22", 11.0, 12.0, 10.5, 11.5),
                ]),
                daily=["2026-09-18", "2026-09-22"],
            )
            censored = evaluate_history_candidate_only(
                _signal("2026-09-18"), suspended, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertEqual(
                censored["label_evaluation_reason"],
                "missing_minute_session:2026-09-21(suspended)",
            )
            self.assertFalse(censored["candidate_evaluable"])
            for field in ("target7", "target10", "target7_d2open_d3high",
                          "target7_d2open_d3close", "d2open_d3high_return_pct",
                          "candidate_d2_max_return_pct", "candidate_d3_max_return_pct",
                          "candidate_d5_max_return_pct", "candidate_d10_max_return_pct"):
                self.assertIsNone(censored[field], field)
            self.assertEqual((censored["d2_trade_date"], censored["d3_trade_date"]),
                             ("2026-09-21", "2026-09-22"))

    def test_gate_a7b_provider_gap_is_distinguished_from_suspension(self) -> None:
        """两种 censoring 都不产生标签, 但 reason 必须能区分。"""
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            gap = _service(
                calendar,
                minute=_minute([
                    ("2026-09-18", 10.0, 10.0, 10.0, 10.0),
                    ("2026-09-22", 11.0, 12.0, 10.5, 11.5),
                ]),
                daily=["2026-09-18", "2026-09-21", "2026-09-22"],  # 当天有成交
            )
            result = evaluate_history_candidate_only(
                _signal("2026-09-18"), gap, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertEqual(
                result["label_evaluation_reason"],
                "missing_minute_session:2026-09-21(provider_gap)",
            )
            self.assertIsNone(result["target7"])
            self.assertFalse(result["candidate_evaluable"])

            # 日线缓存没有跨越这一天 -> 证据不足, reason 不加标签。
            unproven = _service(
                calendar,
                minute=_minute([
                    ("2026-09-18", 10.0, 10.0, 10.0, 10.0),
                    ("2026-09-22", 11.0, 12.0, 10.5, 11.5),
                ]),
                daily=["2026-09-18"],
            )
            unknown = evaluate_history_candidate_only(
                _signal("2026-09-18"), unproven, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertEqual(
                unknown["label_evaluation_reason"], "missing_minute_session:2026-09-21"
            )
            self.assertIsNone(unknown["target7"])

    def test_gate_a7c_complete_sessions_are_required_to_produce_a_label(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            service = _service(
                calendar,
                minute=_minute([
                    ("2026-09-18", 10.0, 10.0, 10.0, 10.0),
                    ("2026-09-21", 10.0, 10.5, 9.8, 10.2),
                    ("2026-09-22", 11.0, 12.0, 10.5, 11.5),
                ]),
                daily=["2026-09-18", "2026-09-21", "2026-09-22"],
            )
            result = evaluate_history_candidate_only(
                _signal("2026-09-18"), service, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertEqual(result["label_evaluation_reason"], "")
            self.assertTrue(result["candidate_evaluable"])
            self.assertIsNotNone(result["target7"])


class GateATrainingExclusionTest(unittest.TestCase):
    """A8 —— null target 在 v004a 训练前被排除。"""

    def test_gate_a8_null_targets_are_excluded_before_training(self) -> None:
        frame = pd.DataFrame([
            {"signal_date": "2026-09-18", "code": "600000", "eligible_for_trade": True,
             "target7_d2open_d3high": None, "d2open_d3high_return_pct": 20.0,
             "d2open_d3close_return_pct": 15.0, "candidate_base_price": 10.0,
             "d1_close_ma10_pct": 1.0, "d1_low_ma10_pct": 1.0, "trend_hold_score": 50.0,
             "total_score": 50.0, "theme_score": 50.0},
            {"signal_date": "2026-09-18", "code": "600001", "eligible_for_trade": True,
             "target7_d2open_d3high": False, "d2open_d3high_return_pct": 1.0,
             "d2open_d3close_return_pct": 0.5, "candidate_base_price": 10.0,
             "d1_close_ma10_pct": 1.0, "d1_low_ma10_pct": 1.0, "trend_hold_score": 50.0,
             "total_score": 50.0, "theme_score": 50.0},
        ])
        audited = annotate_v004a_input_eligibility(frame)
        self.assertIn("missing_target_label", audited.loc[0, "v004a_exclusion_reason"])
        self.assertEqual(audited.loc[1, "v004a_exclusion_reason"], "")
        prepared, _, _ = prepare_v004a_samples(frame)
        # 明性的 False 保留, 未知的 None 被排除 —— 未知标签不是负样本。
        self.assertEqual(prepared["code"].tolist(), ["600001"])


class GateAProvenanceTest(unittest.TestCase):
    """A9 —— provenance 写入 trading_sessions_v1 命名空间。"""

    def test_gate_a9_calendar_provenance_is_complete(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            calendar.is_trading_day("2026-09-28")
            provenance = calendar.provenance()
            self.assertEqual(
                provenance["calendar_semantics_version"], CALENDAR_SEMANTICS_VERSION
            )
            self.assertEqual(provenance["calendar_source"], "gate_source")
            self.assertEqual(provenance["calendar_coverage_start"], _GATE_COVERAGE[0])
            self.assertEqual(provenance["calendar_coverage_end"], _GATE_COVERAGE[1])
            self.assertEqual(provenance["calendar_origin"], "fetch")

    def test_gate_a9b_manifest_records_the_calendar_provenance(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calendar = _calendar(_config(root))
            calendar.is_trading_day("2026-09-28")
            output_dir = root / "reports" / "history_samples" / CALENDAR_SEMANTICS_VERSION
            output_dir.mkdir(parents=True, exist_ok=True)
            _, _, manifest_path = _write_history_universe_outputs(
                output_dir,
                start_date="2026-09-18",
                end_date="2026-09-30",
                lookback_days=5,
                signal_days=1,
                eval_days=1,
                hold_days=3,
                target_return_pct=7.0,
                universe_snapshot_mode="create-or-verify",
                candidates=pd.DataFrame(),
                universe_audit=pd.DataFrame(),
                universe_membership=pd.DataFrame(),
                calendar_provenance=calendar.provenance(),
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["calendar_semantics_version"], CALENDAR_SEMANTICS_VERSION
            )
            self.assertEqual(manifest["calendar_source"], "gate_source")
            self.assertEqual(manifest["calendar_coverage_start"], _GATE_COVERAGE[0])
            self.assertEqual(manifest["calendar_coverage_end"], _GATE_COVERAGE[1])
            self.assertIn("calendar_snapshot_sha256", manifest)

    def test_gate_a9c_corrected_namespace_requires_a_versioned_input_path(self) -> None:
        self.assertEqual(
            _input_calendar_semantics_namespace(
                f"reports/daily_signals/{CALENDAR_SEMANTICS_VERSION}/signals.csv"
            ),
            CALENDAR_SEMANTICS_VERSION,
        )
        self.assertEqual(
            _input_calendar_semantics_namespace("reports/daily_signals/signals.csv"),
            "unverified_calendar_semantics",
        )


class GateALegacyIsolationTest(unittest.TestCase):
    """Gate A 的负向边界: corrected 路径不得借用 legacy calendar semantics。"""

    def test_gate_a_legacy_adapter_is_opt_in_only(self) -> None:
        """默认日历不接受自然日距离; legacy 只能显式注入。"""
        with TemporaryDirectory() as temp:
            calendar = _calendar(_config(Path(temp)))
            # 自然日公式给出 8, session 语义给出 1 —— 两者必须不同。
            self.assertEqual(
                _LegacyCalendarDayDistance().session_distance("2026-09-30", "2026-10-08"), 8
            )
            self.assertEqual(calendar.session_distance("2026-09-30", "2026-10-08"), 1)
            self.assertEqual(
                calendar.session_distance("2026-09-30", "2026-10-09"), 2
            )
            legacy = _LegacyCalendarDayDistance().session_distance("2026-09-30", "2026-10-09")
            self.assertEqual(legacy, 9)

    def test_gate_a_calendar_failure_propagates_through_signal_builders(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = TradingCalendar(
                _config(Path(temp)),
                sources=("offline",),
                fetchers={"offline": lambda: (_ for _ in ()).throw(RuntimeError("down"))},
            )
            service = mock.Mock(trading_calendar=calendar)
            service.get_stock_bars.return_value = SimpleNamespace(
                quality={}, daily=pd.DataFrame(), minute_5m=pd.DataFrame()
            )
            pool = pd.DataFrame({
                "trade_date": ["2026-09-24"], "code": ["600000"], "name": ["Test"],
            })
            pool.attrs["trading_dates_covered"] = ("2026-09-24",)
            with self.assertRaises(TradingCalendarError):
                build_signals_for_pool(service, pool, "2026-09-24", days=1)


if __name__ == "__main__":
    unittest.main()

"""交易日历 (src/trading_calendar.py) 与 collect_limit_ups 节假日跳过契约。

全部离线: 日历取数通过 ``fetchers`` 注入, 不触碰网络。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock
import json

import pandas as pd

from src.config import DataConfig
from src.loaders import MarketDataService
from src.backtester import (
    _covered_limit_up_codes, _exact_consecutive_boards, build_signals_for_pool,
    evaluate_history_signal,
    _future_trade_dates, _input_calendar_semantics_namespace, _iter_trading_days,
    _next_trade_date, _path_metrics_by_horizon,
)
from src.cli import _build_signals
from src.history_samples import evaluate_history_candidate_only
from src.signal_engine import _count_consecutive_boards, generate_signal
from src.legacy_calendar_semantics import _LegacyCalendarDayDistance, _legacy_consecutive_boards
from src.v004a import annotate_v004a_input_eligibility, prepare_v004a_samples
from src.trading_calendar import (
    NonTradingDayError,
    TradingCalendar,
    TradingCalendarError,
)


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


def _fake_calendar(
    config: DataConfig,
    dates: list[str],
    *,
    source: str = "test_source",
    calls: list[str] | None = None,
) -> TradingCalendar:
    def fetcher() -> list[str]:
        if calls is not None:
            calls.append(source)
        if isinstance(dates, Exception):  # type: ignore[arg-type]
            raise dates  # type: ignore[misc]
        return list(dates)

    return TradingCalendar(config, sources=(source,), fetchers={source: fetcher})


class _FakeProvider:
    """记录被请求的日期, 返回一行可识别的涨停池。"""

    def __init__(self) -> None:
        self.requested: list[str] = []

    def fetch_limit_up_pool(self, trade_date: str) -> tuple[pd.DataFrame, str]:
        self.requested.append(trade_date)
        return (
            pd.DataFrame({"trade_date": [trade_date], "code": ["600000"], "source": ["fake"]}),
            "fake",
        )


# 2026-09-25 为中秋节假期, 09-26/09-27 为周末; 09-24 与 09-28 为交易日。
_Q3_DATES = [
    "2026-09-01",
    "2026-09-02",
    "2026-09-03",
    "2026-09-04",
    "2026-09-07",
    "2026-09-08",
    "2026-09-09",
    "2026-09-10",
    "2026-09-11",
    "2026-09-14",
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
    "2026-09-21",
    "2026-09-22",
    "2026-09-23",
    "2026-09-24",
    "2026-09-28",
    "2026-09-29",
    "2026-09-30",
    "2026-10-09",
]


class TradingCalendarTest(unittest.TestCase):
    def test_session_distance_weekend_holidays_and_reverse_contract(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(_config(Path(temp)), _Q3_DATES)
            for start, end, expected in (
                ("2026-09-18", "2026-09-21", 1),  # Friday -> Monday
                ("2026-09-17", "2026-09-21", 2),
                ("2026-09-16", "2026-09-21", 3),
                ("2026-09-18", "2026-09-22", 2),
                ("2026-09-24", "2026-09-24", 0),
                ("2026-09-24", "2026-09-28", 1),  # holiday + weekend
                ("2026-09-30", "2026-10-09", 1),  # long holiday
            ):
                with self.subTest(start=start, end=end):
                    self.assertEqual(calendar.session_distance(start, end), expected)
            self.assertEqual(calendar.previous_trading_day("2026-09-28"), "2026-09-24")
            self.assertEqual(calendar.next_trading_day("2026-09-24"), "2026-09-28")
            self.assertTrue(calendar.are_consecutive_sessions("2026-09-24", "2026-09-28"))
            self.assertFalse(calendar.are_consecutive_sessions("2026-09-23", "2026-09-28"))
            with self.assertRaises(ValueError):
                calendar.session_distance("2026-09-28", "2026-09-24")
            with self.assertRaises(TradingCalendarError):
                calendar.session_distance("2026-09-25", "2026-09-28")

    def test_single_day_holiday_and_incomplete_coverage_fail_closed(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(
                _config(Path(temp)),
                ["2026-09-22", "2026-09-24", "2026-09-28"],
            )
            self.assertEqual(calendar.session_distance("2026-09-22", "2026-09-24"), 1)
            self.assertEqual(_iter_trading_days("2026-09-22", "2026-09-28", calendar),
                             ["2026-09-22", "2026-09-24", "2026-09-28"])
            with self.assertRaises(TradingCalendarError):
                calendar.trading_days("2026-09-21", "2026-09-28")
            with self.assertRaises(TradingCalendarError):
                calendar.next_trading_day("2026-09-28")

    def test_signal_age_and_board_streak_use_market_sessions(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(_config(Path(temp)), _Q3_DATES)
            pool = pd.DataFrame({
                "code": ["600000"] * 3,
                "trade_date": ["2026-09-18", "2026-09-21", "2026-09-22"],
            })
            pool.attrs["trading_dates_covered"] = (
                "2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22"
            )
            self.assertEqual(_count_consecutive_boards(pool, "600000", "2026-09-21", calendar), 2)
            self.assertEqual(_count_consecutive_boards(pool, "600000", "2026-09-22", calendar), 3)
            uncovered = pool.copy()
            uncovered.attrs.clear()
            with self.assertRaises(TradingCalendarError):
                _count_consecutive_boards(uncovered, "600000", "2026-09-22", calendar)
            holiday_pool = pd.DataFrame({
                "code": ["600000", "600000"],
                "trade_date": ["2026-09-24", "2026-09-28"],
            })
            holiday_pool.attrs["trading_dates_covered"] = (
                "2026-09-23", "2026-09-24", "2026-09-28"
            )
            self.assertEqual(_count_consecutive_boards(holiday_pool, "600000", "2026-09-28", calendar), 2)
            long_holiday_pool = pd.DataFrame({
                "code": ["600000", "600000"],
                "trade_date": ["2026-09-30", "2026-10-09"],
            })
            long_holiday_pool.attrs["trading_dates_covered"] = (
                "2026-09-29", "2026-09-30", "2026-10-09"
            )
            self.assertEqual(
                _count_consecutive_boards(long_holiday_pool, "600000", "2026-10-09", calendar), 2
            )
            with (
                mock.patch("src.signal_engine.score_graph_quality", return_value=(50.0, [])),
                mock.patch("src.signal_engine.score_active_money", return_value=(50.0, [])),
                mock.patch("src.signal_engine.score_support_quality", return_value=(50.0, "", [])),
                mock.patch("src.signal_engine.score_theme", return_value=(50.0, [])),
                mock.patch("src.signal_engine.build_key_zones", return_value={}),
            ):
                signal = generate_signal(
                    "600000", "Test", pd.DataFrame({"date": ["2026-09-29"]}),
                    pd.DataFrame(), holiday_pool, d0_date="2026-09-28",
                    trading_calendar=calendar,
                )
            self.assertEqual(signal.days_since_d0, 1)
            self.assertEqual(signal.consecutive_boards, 2)

    def test_future_sessions_do_not_shift_when_stock_bars_are_missing(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(_config(Path(temp)), _Q3_DATES)
            minute = pd.DataFrame({
                "trade_date": ["2026-09-18", "2026-09-22"],
                "datetime": ["2026-09-18 15:00:00", "2026-09-22 09:35:00"],
            })
            self.assertEqual(_future_trade_dates(minute, "2026-09-18", calendar),
                             ["2026-09-21", "2026-09-22"])
            self.assertEqual(_next_trade_date(minute, "2026-09-18", calendar), "2026-09-21")

    def test_missing_market_d2_minute_bars_censor_labels_and_training(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(_config(Path(temp)), _Q3_DATES)
            service = mock.Mock(trading_calendar=calendar)
            signal = pd.Series({
                "trade_date": "2026-09-18", "requested_signal_date": "2026-09-18",
                "code": "600000", "allowed": True, "signal_type": "D2_LOW_ABSORB",
                "key_zones_json": json.dumps({"d1_close": 10.0}),
            })
            d3_only = pd.DataFrame({
                "trade_date": ["2026-09-18", "2026-09-22"],
                "datetime": ["2026-09-18 15:00:00", "2026-09-22 09:35:00"],
                "open": [10.0, 11.0], "high": [10.0, 12.0],
                "low": [10.0, 10.5], "close": [10.0, 11.5],
            })
            service.minute_cache.read.return_value = d3_only
            censored = evaluate_history_candidate_only(
                signal, service, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertEqual((censored["d2_trade_date"], censored["d3_trade_date"]),
                             ("2026-09-21", "2026-09-22"))
            self.assertEqual(censored["label_evaluation_reason"],
                             "missing_minute_session:2026-09-21")
            self.assertFalse(censored["candidate_evaluable"])
            self.assertEqual(censored["future_trade_days_available"], 1)
            self.assertEqual(
                _path_metrics_by_horizon(
                    d3_only, ["2026-09-21", "2026-09-22"], 10.0,
                    prefix="candidate", hold_days=3,
                ),
                {},
            )
            selected = signal.copy()
            selected["selected_for_execution"] = True
            backtest = evaluate_history_signal(
                selected, service, {}, hold_days=3, target_return_pct=7.0,
                stop_loss_pct=5.0, entry_price_mode="confirmation_close",
                top_n=3, include_all_allowed=False,
            )
            self.assertFalse(backtest["candidate_evaluable"])
            self.assertFalse(backtest["evaluable"])
            self.assertIsNone(backtest["candidate_d3_max_return_pct"])
            for field in ("candidate_d2_max_return_pct", "candidate_d3_max_return_pct",
                          "d2open_d3high_return_pct", "target7", "target10",
                          "target7_d2open_d3high", "target7_d2open_d3close"):
                self.assertIsNone(censored[field], field)

            complete = pd.concat([d3_only, pd.DataFrame({
                "trade_date": ["2026-09-21"], "datetime": ["2026-09-21 09:35:00"],
                "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2],
            })], ignore_index=True)
            service.minute_cache.read.return_value = complete
            evaluated = evaluate_history_candidate_only(
                signal, service, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertTrue(evaluated["candidate_evaluable"])
            self.assertEqual(evaluated["label_evaluation_reason"], "")
            self.assertTrue(evaluated["target7_d2open_d3high"])

            d3_missing = complete[complete["trade_date"] != "2026-09-22"].copy()
            d3_missing = pd.concat([d3_missing, pd.DataFrame({
                "trade_date": ["2026-09-23"], "datetime": ["2026-09-23 09:35:00"],
                "open": [11.0], "high": [12.0], "low": [10.5], "close": [11.5],
            })], ignore_index=True)
            service.minute_cache.read.return_value = d3_missing
            censored_d3 = evaluate_history_candidate_only(
                signal, service, {}, hold_days=3,
                target_return_pct=7.0, secondary_target_return_pct=10.0,
            )
            self.assertEqual(censored_d3["label_evaluation_reason"],
                             "missing_minute_session:2026-09-22")
            self.assertFalse(censored_d3["candidate_evaluable"])
            self.assertIsNone(censored_d3["target7_d2open_d3high"])

            training_input = pd.DataFrame([{
                "signal_date": "2026-09-18", "code": "600000",
                "eligible_for_trade": True, "target7_d2open_d3high": None,
                "d2open_d3high_return_pct": 20.0,
                "d2open_d3close_return_pct": 15.0,
                "candidate_base_price": 10.0, "d1_close_ma10_pct": 1.0,
                "d1_low_ma10_pct": 1.0, "trend_hold_score": 50.0,
                "total_score": 50.0, "theme_score": 50.0,
            }, {
                "signal_date": "2026-09-18", "code": "600001",
                "eligible_for_trade": True, "target7_d2open_d3high": True,
                "d2open_d3high_return_pct": 20.0,
                "d2open_d3close_return_pct": 15.0,
                "candidate_base_price": 10.0, "d1_close_ma10_pct": 1.0,
                "d1_low_ma10_pct": 1.0, "trend_hold_score": 50.0,
                "total_score": 50.0, "theme_score": 50.0,
            }])
            audited = annotate_v004a_input_eligibility(training_input)
            self.assertIn("missing_target_label", audited.loc[0, "v004a_exclusion_reason"])
            prepared, _, _ = prepare_v004a_samples(training_input)
            self.assertEqual(prepared["code"].tolist(), ["600001"])

    def test_signal_builders_propagate_calendar_failure(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(_config(Path(temp)), RuntimeError("calendar offline"))  # type: ignore[arg-type]
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
            with self.assertRaises(TradingCalendarError):
                _build_signals(service, pool, days=1, max_codes=None, force_refresh=False)

    def test_backtest_report_namespace_follows_signal_input_provenance(self) -> None:
        self.assertEqual(
            _input_calendar_semantics_namespace("reports/daily_signals/trading_sessions_v1/signals.csv"),
            "trading_sessions_v1",
        )
        self.assertEqual(
            _input_calendar_semantics_namespace("reports/daily_signals/signals.csv"),
            "unverified_calendar_semantics",
        )

    def test_board_history_fetches_prior_sessions_outside_natural_day_pool(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = _fake_calendar(_config(Path(temp)), _Q3_DATES)
            current = pd.DataFrame({"trade_date": ["2026-10-09"], "code": ["600000"]})
            current.attrs["trading_dates_covered"] = ("2026-10-09",)
            previous = pd.DataFrame({"trade_date": ["2026-09-30"], "code": ["600000"]})
            non_board = pd.DataFrame(columns=["trade_date", "code"])
            service = mock.Mock(trading_calendar=calendar)
            service.collect_limit_ups.side_effect = lambda trade_date, **kwargs: (
                previous if trade_date == "2026-09-30" else non_board
            )
            cache = _covered_limit_up_codes(current)
            self.assertEqual(_exact_consecutive_boards(service, "600000", "2026-10-09", cache), 2)
            self.assertEqual(_exact_consecutive_boards(service, "600000", "2026-10-09", cache), 2)
            self.assertEqual(service.collect_limit_ups.call_count, 2)

    def test_explicit_legacy_replay_remains_separate(self) -> None:
        pool = pd.DataFrame({
            "code": ["600000", "600000"],
            "trade_date": ["2026-09-18", "2026-09-21"],
        })
        self.assertEqual(_LegacyCalendarDayDistance().session_distance("2026-09-18", "2026-09-21"), 3)
        self.assertEqual(_legacy_consecutive_boards(pool, "600000", "2026-09-21"), 1)

    def test_identifies_mid_autumn_and_national_day_holidays(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calendar = _fake_calendar(config, _Q3_DATES)
            self.assertTrue(calendar.is_trading_day("2026-09-24"))
            self.assertTrue(calendar.is_trading_day("2026-09-28"))
            # 09-25 中秋节: weekday()==4, 旧逻辑会误判为交易日
            self.assertFalse(calendar.is_trading_day("2026-09-25"))
            self.assertFalse(calendar.is_trading_day("2026-09-26"))
            self.assertFalse(calendar.is_trading_day("2026-10-01"))

    def test_coverage_is_reported_and_cached(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calls: list[str] = []
            calendar = _fake_calendar(config, _Q3_DATES, calls=calls)
            calendar.is_trading_day("2026-09-28")
            self.assertEqual(calls, ["test_source"])
            self.assertEqual(calendar.coverage, ("2026-09-01", "2026-10-09"))
            self.assertEqual(calendar.last_origin, "fetch")

            # 新实例走缓存: 覆盖区间内零取数。
            second_calls: list[str] = []
            second = _fake_calendar(config, _Q3_DATES, calls=second_calls)
            self.assertTrue(second.is_trading_day("2026-09-24"))
            self.assertEqual(second_calls, [])
            self.assertEqual(second.last_origin, "cache")
            self.assertEqual(second.coverage, ("2026-09-01", "2026-10-09"))

    def test_refetches_when_cache_does_not_cover_request(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            _fake_calendar(config, _Q3_DATES).is_trading_day("2026-09-28")

            wider = _Q3_DATES + ["2026-12-31"]
            calls: list[str] = []
            calendar = _fake_calendar(config, wider, calls=calls)
            self.assertTrue(calendar.is_trading_day("2026-12-31"))
            self.assertEqual(calls, ["test_source"])

    def test_corrupt_cache_metadata_cannot_prove_coverage(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            initial = _fake_calendar(config, _Q3_DATES)
            initial.is_trading_day("2026-09-28")
            path = initial.meta_path("test_source")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["count"] = 1
            path.write_text(json.dumps(metadata), encoding="utf-8")
            unavailable = _fake_calendar(config, RuntimeError("source offline"))  # type: ignore[arg-type]
            with self.assertRaises(TradingCalendarError):
                unavailable.is_trading_day("2026-09-28")

    def test_corrupt_cache_payload_cannot_prove_coverage(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            initial = _fake_calendar(config, _Q3_DATES)
            initial.is_trading_day("2026-09-28")
            pd.to_pickle(["2026-09-28"], initial.data_path("test_source"))
            unavailable = _fake_calendar(config, RuntimeError("source offline"))  # type: ignore[arg-type]
            with self.assertRaises(TradingCalendarError):
                unavailable.is_trading_day("2026-09-28")

    def test_all_sources_failing_is_fail_closed(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calendar = TradingCalendar(
                config,
                sources=("broken_a", "broken_b"),
                fetchers={
                    "broken_a": lambda: (_ for _ in ()).throw(RuntimeError("source a down")),
                    "broken_b": lambda: [],
                },
            )
            with self.assertRaises(TradingCalendarError) as ctx:
                calendar.is_trading_day("2026-09-28")
            message = str(ctx.exception)
            self.assertIn("broken_a", message)
            self.assertIn("source a down", message)
            self.assertIn("broken_b", message)
            self.assertIn("返回空日历", message)

    def test_no_fetchers_and_no_cache_does_not_install_default_network_sources(self) -> None:
        with TemporaryDirectory() as temp:
            calendar = TradingCalendar(
                _config(Path(temp)), sources=("primary", "fallback"), fetchers={}
            )
            with self.assertRaises(TradingCalendarError) as ctx:
                calendar.session_distance("2026-09-24", "2026-09-28")
            self.assertIn("primary", str(ctx.exception))
            self.assertIn("fallback", str(ctx.exception))

    def test_derived_daily_cross_section_backs_off_when_sample_is_thin(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = _config(root)
            daily_dir = config.cache_dir / "daily_unadjusted"
            daily_dir.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                pd.DataFrame({"date": ["2026-09-24", "2026-09-28"]}).to_pickle(
                    daily_dir / f"60000{index}_daily.pkl"
                )
            calendar = TradingCalendar(config, sources=("cached_daily_cross_section",))
            with self.assertRaises(TradingCalendarError) as ctx:
                calendar.is_trading_day("2026-09-25")
            self.assertIn("样本不足", str(ctx.exception))

    def test_derived_daily_cross_section_proves_holiday(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            daily_dir = config.cache_dir / "daily_unadjusted"
            daily_dir.mkdir(parents=True, exist_ok=True)
            dates = [d for d in _Q3_DATES]
            for index in range(60):
                pd.DataFrame({"date": dates}).to_pickle(daily_dir / f"{600000 + index}_daily.pkl")
            calendar = TradingCalendar(config, sources=("cached_daily_cross_section",))
            self.assertTrue(calendar.is_trading_day("2026-09-28"))
            self.assertFalse(calendar.is_trading_day("2026-09-25"))


class CollectLimitUpsCalendarTest(unittest.TestCase):
    def _service(self, config: DataConfig, calendar: TradingCalendar) -> tuple[MarketDataService, _FakeProvider]:
        service = MarketDataService(config)
        provider = _FakeProvider()
        service.provider = provider  # type: ignore[assignment]
        service.trading_calendar = calendar
        return service, provider

    def test_holiday_is_skipped_without_any_fetch(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calendar = _fake_calendar(config, _Q3_DATES)
            service, provider = self._service(config, calendar)

            pool = service.collect_limit_ups(
                trade_date="2026-09-28", lookback_days=5, write_processed=False
            )

            # 修复前: 09-25 被当成交易日 -> 涨停池空 -> 全市场日线回扫 (数小时)。
            self.assertEqual(provider.requested, ["2026-09-24", "2026-09-28"])
            self.assertEqual(sorted(pool["trade_date"].unique().tolist()), ["2026-09-24", "2026-09-28"])

    def test_partial_trading_window_failure_does_not_publish_incomplete_pool(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            service, provider = self._service(config, _fake_calendar(config, _Q3_DATES))
            original = provider.fetch_limit_up_pool

            def fetch(date_text: str) -> tuple[pd.DataFrame, str]:
                if date_text == "2026-09-28":
                    raise RuntimeError("source unavailable")
                return original(date_text)

            provider.fetch_limit_up_pool = fetch  # type: ignore[method-assign]
            service._derive_limit_up_pool_from_daily = mock.Mock(side_effect=RuntimeError("fallback unavailable"))
            with self.assertRaisesRegex(RuntimeError, "incomplete limit-up pool"):
                service.collect_limit_ups(
                    trade_date="2026-09-28", lookback_days=5, write_processed=False
                )

    def test_window_of_only_non_trading_days_raises_non_trading_day_error(self) -> None:
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calendar = _fake_calendar(config, _Q3_DATES)
            service, provider = self._service(config, calendar)

            with self.assertRaises(NonTradingDayError):
                service.collect_limit_ups(
                    trade_date="2026-09-25", lookback_days=1, write_processed=False
                )
            self.assertEqual(provider.requested, [])

    def test_trading_day_with_failing_sources_still_raises_plain_error(self) -> None:
        """有交易日但取数全失败 = 真故障, 不能被误判为节假日。"""
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calendar = _fake_calendar(config, _Q3_DATES)
            service, _ = self._service(config, calendar)

            def boom(*_args: object, **_kwargs: object) -> pd.DataFrame:
                raise RuntimeError("source down")

            service.provider.fetch_limit_up_pool = boom  # type: ignore[method-assign]
            service._derive_limit_up_pool_from_daily = boom  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError) as ctx:
                service.collect_limit_ups(
                    trade_date="2026-09-28", lookback_days=1, write_processed=False
                )
            self.assertNotIsInstance(ctx.exception, NonTradingDayError)
            self.assertIn("source down", str(ctx.exception))
            self.assertIn("2026-09-28", str(ctx.exception))

    def test_result_is_unchanged_by_the_calendar(self) -> None:
        """日历只剔除节假日; 交易日产出在修复前后完全一致。"""
        with TemporaryDirectory() as temp:
            config = _config(Path(temp))
            calendar = _fake_calendar(config, _Q3_DATES)
            service, _ = self._service(config, calendar)
            cached_dir = config.cache_dir / "limit_ups"
            cached_dir.mkdir(parents=True, exist_ok=True)
            for date_text in _Q3_DATES:
                if date_text in ("2026-09-24", "2026-09-28"):
                    pd.DataFrame(
                        {"trade_date": [date_text], "code": ["600000"], "source": ["pre"]}
                    ).to_pickle(cached_dir / f"{date_text}_limitups.pkl")

            pool = service.collect_limit_ups(
                trade_date="2026-09-28", lookback_days=5, write_processed=False
            )
            self.assertEqual(sorted(pool["trade_date"].unique().tolist()), ["2026-09-24", "2026-09-28"])


if __name__ == "__main__":
    unittest.main()

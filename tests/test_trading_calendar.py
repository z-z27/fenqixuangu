"""交易日历 (src/trading_calendar.py) 与 collect_limit_ups 节假日跳过契约。

全部离线: 日历取数通过 ``fetchers`` 注入, 不触碰网络。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.config import DataConfig
from src.loaders import MarketDataService
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

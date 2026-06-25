from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cache import FrameCache, StockFrameCache
from .code_utils import normalize_stock_code
from .config import DataConfig, get_data_config
from .data_sources import MarketDataProvider
from .indicators import enrich_5min_indicators, enrich_daily_indicators


@dataclass
class StockBars:
    code: str
    daily: pd.DataFrame
    minute_5m: pd.DataFrame
    daily_source: str
    minute_source: str
    from_cache: bool


class MarketDataService:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or get_data_config()
        self.provider = MarketDataProvider(self.config)
        self.limit_up_cache = FrameCache(self.config.cache_dir / "limit_ups", "limitups")
        self.daily_cache = StockFrameCache(self.config.cache_dir / "daily", "daily")
        self.minute_cache = StockFrameCache(self.config.cache_dir / "minute_5m", "5min")

    def collect_limit_ups(
        self,
        trade_date: str | None = None,
        lookback_days: int = 1,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        anchor = pd.Timestamp(trade_date or datetime.now().strftime("%Y-%m-%d"))
        for offset in range(max(1, lookback_days)):
            current = anchor - pd.Timedelta(days=offset)
            if current.weekday() >= 5:
                continue
            date_text = current.strftime("%Y-%m-%d")
            cache_key = date_text
            cached = None if force_refresh else self.limit_up_cache.read(cache_key)
            if cached is not None and not cached.empty:
                frame = cached.copy()
            else:
                try:
                    frame, _ = self.provider.fetch_limit_up_pool(date_text)
                except Exception as exc:
                    errors.append(f"{date_text}: {exc}")
                    continue
                self.limit_up_cache.write(cache_key, frame)
                raw_path = self.config.raw_dir / "limit_ups" / f"limit_up_{date_text}.csv"
                frame.to_csv(raw_path, index=False, encoding="utf-8-sig")
            frames.append(frame)

        if not frames:
            raise RuntimeError("no limit-up data collected: " + " | ".join(errors))

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["trade_date", "code"]).drop_duplicates(["trade_date", "code"], keep="last")
        result = result.reset_index(drop=True)
        out_path = self.config.processed_dir / "recent_limitups.csv"
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        return result

    def get_stock_bars(
        self,
        code: str,
        days: int | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> StockBars:
        normalized = normalize_stock_code(code)
        day_count = int(days or self.config.default_5min_days)
        cached_daily = None if force_refresh else self.daily_cache.read(normalized)
        cached_minute = None if force_refresh else self.minute_cache.read(normalized)
        end_ts = pd.Timestamp(end_date or datetime.now().strftime("%Y-%m-%d"))
        start_ts = end_ts - pd.Timedelta(days=max(90, int(day_count * 2.8)))
        daily_source = "cache"
        minute_source = "cache"
        from_cache = True

        if cached_daily is None or cached_daily.empty or not _cache_covers(cached_daily, "date", day_count):
            daily, daily_source = self.provider.fetch_daily_history(
                normalized,
                start_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
                adjust=self.config.adjust,
            )
            cached_daily = daily
            self.daily_cache.write(normalized, cached_daily)
            from_cache = False

        if cached_minute is None or cached_minute.empty or not _cache_covers(cached_minute, "trade_date", day_count):
            minute_start = start_ts.strftime("%Y-%m-%d 09:30:00")
            minute_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d 15:00:00")
            minute, minute_source = self.provider.fetch_5min_history(
                normalized,
                minute_start,
                minute_end,
                adjust=self.config.adjust,
            )
            cached_minute = minute
            self.minute_cache.write(normalized, cached_minute)
            from_cache = False

        daily_recent = _keep_recent_trade_days(cached_daily, "date", day_count)
        minute_recent = _keep_recent_trade_days(cached_minute, "trade_date", day_count)
        daily_recent = enrich_daily_indicators(daily_recent)
        minute_recent = enrich_5min_indicators(minute_recent)

        daily_path = self.config.raw_dir / "daily" / f"{normalized}_daily.csv"
        minute_path = self.config.raw_dir / "minute_5m" / f"{normalized}_5min.csv"
        daily_recent.to_csv(daily_path, index=False, encoding="utf-8-sig")
        minute_recent.to_csv(minute_path, index=False, encoding="utf-8-sig")

        return StockBars(
            code=normalized,
            daily=daily_recent,
            minute_5m=minute_recent,
            daily_source=daily_source,
            minute_source=minute_source,
            from_cache=from_cache,
        )

    def collect_bars_for_limitups(
        self,
        limit_up_pool: pd.DataFrame,
        days: int | None = None,
        max_codes: int | None = None,
        force_refresh: bool = False,
    ) -> dict[str, StockBars]:
        result: dict[str, StockBars] = {}
        codes = limit_up_pool["code"].dropna().astype(str).drop_duplicates().tolist()
        if max_codes:
            codes = codes[: int(max_codes)]
        for code in codes:
            result[code] = self.get_stock_bars(code, days=days, force_refresh=force_refresh)
        return result


def load_limitup_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    frame["code"] = frame["code"].map(normalize_stock_code)
    return frame


def _keep_recent_trade_days(frame: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    trade_dates = sorted(result[date_col].dropna().unique().tolist())
    selected = set(trade_dates[-int(days) :])
    return result[result[date_col].isin(selected)].sort_values(date_col).reset_index(drop=True)


def _cache_covers(frame: pd.DataFrame, date_col: str, days: int) -> bool:
    if frame is None or frame.empty or date_col not in frame.columns:
        return False
    return int(frame[date_col].dropna().nunique()) >= int(days)

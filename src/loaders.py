from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import FrameCache, StockFrameCache
from .code_utils import is_main_board_code, normalize_stock_code
from .config import DataConfig, get_data_config
from .data_sources import MarketDataProvider
from .indicators import enrich_5min_indicators, enrich_daily_indicators


MAIN_BOARD_LIMIT_RATIO = Decimal("1.10")
LIMIT_UP_PRICE_TOLERANCE = 0.011
LIMIT_UP_MIN_PCT_CHG = 9.7
DERIVED_LIMITUP_PROGRESS_STEP = 100


@dataclass
class StockBars:
    code: str
    daily: pd.DataFrame
    minute_5m: pd.DataFrame
    daily_source: str
    minute_source: str
    from_cache: bool
    quality: dict[str, Any]


class DataQualityError(RuntimeError):
    def __init__(self, message: str, quality: dict[str, Any]):
        super().__init__(message)
        self.quality = quality


class MarketDataService:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or get_data_config()
        self.provider = MarketDataProvider(self.config)
        self.limit_up_cache = FrameCache(self.config.cache_dir / "limit_ups", "limitups")
        self.daily_cache = StockFrameCache(self.config.cache_dir / "daily", "daily")
        self.daily_unadjusted_cache = StockFrameCache(self.config.cache_dir / "daily_unadjusted", "daily")
        self.minute_cache = StockFrameCache(self.config.cache_dir / "minute_5m", "5min")
        self.universe_cache = FrameCache(self.config.cache_dir / "universe", "universe")

    def collect_limit_ups(
        self,
        trade_date: str | None = None,
        lookback_days: int = 1,
        force_refresh: bool = False,
        write_processed: bool = True,
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
                    try:
                        frame = self._derive_limit_up_pool_from_daily(
                            date_text,
                            force_refresh=force_refresh,
                        )
                    except Exception as fallback_exc:
                        errors.append(
                            f"{date_text}: source pool failed ({exc}); "
                            f"daily-derived fallback failed ({fallback_exc})"
                        )
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
        if write_processed:
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
        target_5min_days = int(days or self.config.default_5min_days)
        min_5min_days = min(target_5min_days, max(1, int(getattr(self.config, "min_5min_trade_days", 20))))
        warmup_days = max(int(self.config.indicator_warmup_trading_days), 120)
        daily_required_days = max(int(self.config.daily_history_days), target_5min_days + warmup_days)
        cached_daily = None if force_refresh else self.daily_cache.read(normalized)
        cached_minute = None if force_refresh else self.minute_cache.read(normalized)
        end_ts = pd.Timestamp(end_date or datetime.now().strftime("%Y-%m-%d"))
        end_date_text = end_ts.strftime("%Y-%m-%d")
        daily_start_ts = end_ts - pd.Timedelta(days=max(365, int(daily_required_days * 2.4)))
        minute_start_ts = end_ts - pd.Timedelta(days=max(90, int((target_5min_days + 5) * 2.8)))
        daily_source = "cache"
        minute_source = "cache"
        from_cache = True

        if cached_daily is None or cached_daily.empty or not _cache_covers(
            cached_daily,
            "date",
            daily_required_days,
            end_date=end_date_text,
        ):
            daily, daily_source = self.provider.fetch_daily_history(
                normalized,
                daily_start_ts.strftime("%Y-%m-%d"),
                end_date_text,
                adjust=self.config.adjust,
            )
            cached_daily = daily
            self.daily_cache.write(normalized, cached_daily)
            from_cache = False

        if cached_minute is None or cached_minute.empty or not _cache_covers(
            cached_minute,
            "trade_date",
            target_5min_days,
            end_date=end_date_text,
        ):
            minute_start = minute_start_ts.strftime("%Y-%m-%d 09:30:00")
            minute_end = end_ts.strftime("%Y-%m-%d 15:00:00")
            minute, minute_source = self.provider.fetch_5min_history(
                normalized,
                minute_start,
                minute_end,
                adjust=self.config.adjust,
            )
            cached_minute = minute
            self.minute_cache.write(normalized, cached_minute)
            from_cache = False

        daily_until_end = _filter_to_end_date(cached_daily, "date", end_date_text)
        minute_until_end = _filter_to_end_date(cached_minute, "trade_date", end_date_text)
        cached_daily = _merge_minute_amount(daily_until_end, minute_until_end)
        daily_history = _keep_recent_trade_days(cached_daily, "date", daily_required_days)
        daily_recent = _keep_recent_trade_days(cached_daily, "date", target_5min_days)
        minute_recent = _keep_recent_trade_days(minute_until_end, "trade_date", target_5min_days)
        daily_recent = enrich_daily_indicators(daily_recent, full_daily=daily_history)
        minute_recent = enrich_5min_indicators(minute_recent)
        _ensure_daily_ma_coverage(daily_recent, normalized)
        quality = _build_quality_report(
            code=normalized,
            daily=daily_recent,
            minute=minute_recent,
            daily_history=daily_history,
            daily_source=daily_source,
            minute_source=minute_source,
            from_cache=from_cache,
            daily_required_days=daily_required_days,
            minute_target_days=target_5min_days,
            minute_required_days=min_5min_days,
        )
        if quality["status"] != "ok":
            raise DataQualityError(f"{normalized} data quality check failed: {quality['warnings']}", quality)

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
            quality=quality,
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

    def _derive_limit_up_pool_from_daily(
        self,
        trade_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        date_text = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        universe = self._get_stock_universe(force_refresh=force_refresh)
        if universe.empty:
            raise RuntimeError("stock universe is empty")

        rows: list[dict[str, Any]] = []
        checked = 0
        date_seen = 0
        errors: list[str] = []
        total = int(len(universe))
        print(f"[derive-limitups] {date_text} start universe={total}", flush=True)
        for _, stock in universe.iterrows():
            code = normalize_stock_code(str(stock.get("code", "")))
            try:
                daily = self._get_daily_for_limitup_scan(
                    code,
                    end_date=date_text,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                if len(errors) < 20:
                    errors.append(f"{code}: {exc}")
                continue
            checked += 1
            if checked % DERIVED_LIMITUP_PROGRESS_STEP == 0:
                print(
                    f"[derive-limitups] {date_text} checked={checked}/{total} "
                    f"date_seen={date_seen} rows={len(rows)} errors={len(errors)}",
                    flush=True,
                )
            row = _derive_limit_up_row_from_daily(daily, stock, date_text)
            if row is None:
                if _daily_has_date(daily, date_text):
                    date_seen += 1
                continue
            date_seen += 1
            rows.append(row)

        if not rows:
            reason = f"no daily-derived limit-up rows for {date_text}; checked={checked}; date_seen={date_seen}"
            if errors:
                reason += "; sample_errors=" + " | ".join(errors[:5])
            raise RuntimeError(reason)

        result = pd.DataFrame(rows)
        result = result.sort_values(["trade_date", "code"]).drop_duplicates(["trade_date", "code"], keep="last")
        print(
            f"[derive-limitups] {date_text} done checked={checked}/{total} "
            f"date_seen={date_seen} rows={len(result)} errors={len(errors)}",
            flush=True,
        )
        return result.reset_index(drop=True)

    def _get_stock_universe(self, force_refresh: bool = False) -> pd.DataFrame:
        cached = None if force_refresh else self.universe_cache.read("eastmoney_main_board")
        if self._universe_has_min_size(cached):
            return cached.copy()

        fetch_error: Exception | None = None
        try:
            frame, _ = self.provider.fetch_stock_universe()
            if not self._universe_has_min_size(frame):
                raise RuntimeError(
                    f"stock universe too small: {len(frame)} < {self.config.min_limitup_universe_size}"
                )
            self.universe_cache.write("eastmoney_main_board", frame)
            return frame
        except Exception as exc:
            fetch_error = exc

        if self._universe_has_min_size(cached):
            return cached.copy()

        frame = self._stock_universe_from_daily_cache()
        if self._universe_has_min_size(frame):
            return frame
        raise RuntimeError(f"stock universe unavailable: {fetch_error}")

    def _universe_has_min_size(self, frame: pd.DataFrame | None) -> bool:
        if frame is None or frame.empty or "code" not in frame.columns:
            return False
        return int(frame["code"].dropna().astype(str).nunique()) >= int(self.config.min_limitup_universe_size)

    def _stock_universe_from_daily_cache(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.daily_cache.root.glob("*_daily.pkl")):
            suffix = "_daily.pkl"
            if not path.name.endswith(suffix):
                continue
            code = path.name[: -len(suffix)]
            try:
                normalized = normalize_stock_code(code)
            except ValueError:
                continue
            if not is_main_board_code(normalized):
                continue
            rows.append(
                {
                    "code": normalized,
                    "name": "",
                    "market": "SH" if normalized.startswith("6") else "SZ",
                    "industry": "",
                    "source": "daily_cache_universe",
                }
            )
        return pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True) if rows else pd.DataFrame()

    def _get_daily_for_limitup_scan(
        self,
        code: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        normalized = normalize_stock_code(code)
        cached = None if force_refresh else self.daily_unadjusted_cache.read(normalized)
        if _daily_has_date_and_previous(cached, end_date):
            return cached

        end_ts = pd.Timestamp(end_date)
        start_ts = end_ts - pd.Timedelta(days=120)
        fetched, _ = self.provider.fetch_daily_history(
            normalized,
            start_ts.strftime("%Y-%m-%d"),
            end_ts.strftime("%Y-%m-%d"),
            adjust="none",
        )
        merged = _merge_daily_frames(cached, fetched)
        self.daily_unadjusted_cache.write(normalized, merged)
        return merged


def load_limitup_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    frame["code"] = frame["code"].map(normalize_stock_code)
    return frame


def _derive_limit_up_row_from_daily(
    daily: pd.DataFrame,
    stock: pd.Series,
    trade_date: str,
) -> dict[str, Any] | None:
    data = _prepare_daily_limitup_scan_frame(daily)
    if data.empty:
        return None
    code = normalize_stock_code(str(stock.get("code", "")))
    if not is_main_board_code(code):
        return None

    matches = data.index[data["date"] == trade_date].tolist()
    if not matches:
        return None
    idx = matches[-1]
    if not _is_main_board_limit_up_day(data, idx):
        return None

    today = data.iloc[idx]
    prev = data.iloc[idx - 1]
    prev_close = _to_float(prev.get("close"))
    close = _to_float(today.get("close"))
    high = _to_float(today.get("high"))
    low = _to_float(today.get("low"))
    open_price = _to_float(today.get("open"))
    if prev_close is None or close is None or high is None:
        return None

    pct_chg = (close / prev_close - 1.0) * 100.0
    limit_price = _round_price_limit(prev_close, MAIN_BOARD_LIMIT_RATIO)
    consecutive = _count_consecutive_main_board_limit_ups(data, idx)
    amount = _to_float(today.get("amount"))
    total_market_cap = _to_float(stock.get("total_market_cap"))
    float_market_cap = _to_float(stock.get("float_market_cap"))
    name = "" if pd.isna(stock.get("name", "")) else str(stock.get("name", ""))
    industry = "" if pd.isna(stock.get("industry", "")) else str(stock.get("industry", ""))
    market = "SH" if code.startswith("6") else "SZ"
    return {
        "trade_date": trade_date,
        "code": code,
        "name": name,
        "market": market,
        "latest_price": close,
        "pct_chg": pct_chg,
        "amount": amount,
        "turnover_rate": _to_float(today.get("turnover_rate")),
        "float_market_cap": float_market_cap,
        "total_market_cap": total_market_cap,
        "industry": industry,
        "limit_up_time": "",
        "final_limit_up_time": "",
        "open_board_count": None,
        "seal_amount": None,
        "consecutive_limit_up_count": consecutive,
        "source": "daily_limitup_derived",
        "open": open_price,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "limit_price": limit_price,
        "limitup_basis": "daily_close_at_limit",
        "limitup_price_tolerance": LIMIT_UP_PRICE_TOLERANCE,
        "limitup_min_pct_chg": LIMIT_UP_MIN_PCT_CHG,
    }


def _prepare_daily_limitup_scan_frame(daily: pd.DataFrame | None) -> pd.DataFrame:
    if daily is None or daily.empty or "date" not in daily.columns:
        return pd.DataFrame()
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "amount", "turnover_rate"):
        if column not in data.columns:
            data[column] = None
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _is_main_board_limit_up_day(data: pd.DataFrame, idx: int) -> bool:
    if data is None or data.empty or idx <= 0 or idx >= len(data):
        return False
    today = data.iloc[idx]
    prev = data.iloc[idx - 1]
    prev_close = _to_float(prev.get("close"))
    close = _to_float(today.get("close"))
    high = _to_float(today.get("high"))
    if prev_close is None or prev_close <= 0 or close is None or high is None:
        return False

    limit_price = _round_price_limit(prev_close, MAIN_BOARD_LIMIT_RATIO)
    pct_chg = (close / prev_close - 1.0) * 100.0
    if pct_chg < LIMIT_UP_MIN_PCT_CHG:
        return False
    if close < limit_price - LIMIT_UP_PRICE_TOLERANCE:
        return False
    if high < limit_price - LIMIT_UP_PRICE_TOLERANCE:
        return False
    return True


def _count_consecutive_main_board_limit_ups(data: pd.DataFrame, idx: int) -> int:
    count = 0
    current = idx
    while current > 0 and _is_main_board_limit_up_day(data, current):
        count += 1
        current -= 1
    return count


def _daily_has_date(frame: pd.DataFrame | None, date_text: str) -> bool:
    if frame is None or frame.empty or "date" not in frame.columns:
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return bool((dates == date_text).any())


def _daily_has_date_and_previous(frame: pd.DataFrame | None, date_text: str) -> bool:
    if frame is None or frame.empty or "date" not in frame.columns:
        return False
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dates = sorted(data["date"].dropna().unique().tolist())
    if date_text not in dates:
        return False
    return dates.index(date_text) > 0


def _merge_daily_frames(cached: pd.DataFrame | None, fetched: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (cached, fetched) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "code" in result.columns:
        result["code"] = result["code"].map(normalize_stock_code)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return result.dropna(subset=["date"]).drop_duplicates(["code", "date"], keep="last").sort_values("date").reset_index(drop=True)


def _round_price_limit(prev_close: float, ratio: Decimal) -> float:
    value = Decimal(str(prev_close)) * ratio
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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


def _keep_recent_trade_days(frame: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    trade_dates = sorted(result[date_col].dropna().unique().tolist())
    selected = set(trade_dates[-int(days) :])
    return result[result[date_col].isin(selected)].sort_values(date_col).reset_index(drop=True)


def _filter_to_end_date(frame: pd.DataFrame, date_col: str, end_date: str) -> pd.DataFrame:
    if frame is None or frame.empty or date_col not in frame.columns:
        return frame
    result = frame.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    end_ts = pd.Timestamp(end_date)
    return result[dates <= end_ts].reset_index(drop=True)


def _merge_minute_amount(daily: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    if minute is None or minute.empty or "amount" not in minute.columns:
        return daily
    amount_col = pd.to_numeric(minute["amount"], errors="coerce")
    daily_amount = amount_col.groupby(minute["trade_date"]).sum().reset_index()
    daily_amount.columns = ["date", "amount_from_5m"]
    result = daily.merge(daily_amount, on="date", how="left")
    mask = result["amount_from_5m"].notna()
    result.loc[mask, "amount"] = result.loc[mask, "amount_from_5m"]
    return result.drop(columns=["amount_from_5m"])


def _cache_covers(frame: pd.DataFrame, date_col: str, days: int, end_date: str | None = None) -> bool:
    if frame is None or frame.empty or date_col not in frame.columns:
        return False
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        dates = dates[dates <= end_ts]
        if dates.empty or int(dates.dt.strftime("%Y-%m-%d").nunique()) < int(days):
            return False
        latest = dates.max().strftime("%Y-%m-%d")
        return latest == end_ts.strftime("%Y-%m-%d")
    if dates.empty or int(dates.dt.strftime("%Y-%m-%d").nunique()) < int(days):
        return False
    return True


def _ensure_daily_ma_coverage(daily: pd.DataFrame, code: str) -> None:
    required = ("ma5", "ma10", "ma20", "ma30")
    if daily is None or daily.empty:
        raise RuntimeError(f"{code} daily data is empty after indicator enrichment")
    latest = daily.iloc[-1]
    missing = [column for column in required if column not in daily.columns or pd.isna(latest.get(column))]
    if missing:
        raise RuntimeError(f"{code} missing required daily moving averages on latest row: {', '.join(missing)}")


def _build_quality_report(
    code: str,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    daily_history: pd.DataFrame,
    daily_source: str,
    minute_source: str,
    from_cache: bool,
    daily_required_days: int,
    minute_target_days: int,
    minute_required_days: int,
) -> dict[str, Any]:
    ma_columns = ("ma5", "ma10", "ma20", "ma30")
    latest = daily.iloc[-1] if daily is not None and not daily.empty else {}
    missing_latest_ma = [
        column
        for column in ma_columns
        if column not in daily.columns or pd.isna(latest.get(column))
    ]
    daily_history_rows = int(daily_history["date"].dropna().nunique()) if "date" in daily_history.columns else 0
    minute_trade_days = int(minute["trade_date"].dropna().nunique()) if "trade_date" in minute.columns else 0
    cross_check = _daily_minute_close_cross_check(daily, minute)
    missing_daily_amount = int(daily["amount"].isna().sum()) if "amount" in daily.columns else len(daily)
    missing_daily_volume = int(daily["volume"].isna().sum()) if "volume" in daily.columns else len(daily)
    missing_minute_amount = int(minute["amount"].isna().sum()) if "amount" in minute.columns else len(minute)
    missing_minute_volume = int(minute["volume"].isna().sum()) if "volume" in minute.columns else len(minute)
    zero_minute_volume = int((pd.to_numeric(minute.get("volume"), errors="coerce").fillna(0) == 0).sum()) if "volume" in minute.columns else 0
    warnings: list[str] = []
    hard_failures: list[str] = []
    if daily_history_rows < daily_required_days:
        hard_failures.append(f"daily history rows {daily_history_rows} < required {daily_required_days}")
    if minute_trade_days < minute_required_days:
        hard_failures.append(f"minute trade days {minute_trade_days} < required {minute_required_days}")
    if missing_latest_ma:
        hard_failures.append("missing latest daily MA: " + ",".join(missing_latest_ma))
    if not cross_check["passed"]:
        hard_failures.append(
            f"daily/minute close cross-check failed: matched={cross_check['matched_days']}, max_diff={cross_check['max_abs_difference']}"
        )
    warnings.extend(hard_failures)
    status = "failed" if hard_failures else "ok"
    return {
        "code": code,
        "status": status,
        "daily_source": daily_source,
        "minute_source": minute_source,
        "from_cache": bool(from_cache),
        "daily_rows": int(len(daily)),
        "daily_history_rows": daily_history_rows,
        "daily_required_days": int(daily_required_days),
        "minute_rows": int(len(minute)),
        "minute_trade_days": minute_trade_days,
        "minute_target_days": int(minute_target_days),
        "minute_required_days": int(minute_required_days),
        "daily_start": _date_min(daily, "date"),
        "daily_end": _date_max(daily, "date"),
        "minute_start": _date_min(minute, "datetime"),
        "minute_end": _date_max(minute, "datetime"),
        "daily_ma_coverage_ok": not missing_latest_ma,
        "missing_latest_daily_ma": ",".join(missing_latest_ma),
        "missing_daily_amount_count": missing_daily_amount,
        "missing_daily_volume_count": missing_daily_volume,
        "missing_minute_amount_count": missing_minute_amount,
        "missing_minute_volume_count": missing_minute_volume,
        "zero_minute_volume_count": zero_minute_volume,
        "daily_minute_close_matched_days": cross_check["matched_days"],
        "daily_minute_close_max_abs_diff": cross_check["max_abs_difference"],
        "daily_minute_close_check_ok": cross_check["passed"],
        "warnings": "; ".join(warnings),
        "error": "",
    }


def _daily_minute_close_cross_check(daily: pd.DataFrame, minute: pd.DataFrame) -> dict[str, Any]:
    if daily is None or daily.empty or minute is None or minute.empty:
        return {"matched_days": 0, "max_abs_difference": None, "passed": False}
    if "date" not in daily.columns or "trade_date" not in minute.columns:
        return {"matched_days": 0, "max_abs_difference": None, "passed": False}
    minute_close = (
        minute.sort_values("datetime")
        .groupby("trade_date", as_index=False)
        .agg(minute_close=("close", "last"))
    )
    comparison = daily[["date", "close"]].rename(columns={"date": "trade_date", "close": "daily_close"}).merge(
        minute_close,
        on="trade_date",
        how="inner",
    )
    if comparison.empty:
        return {"matched_days": 0, "max_abs_difference": None, "passed": False}
    diff = (
        pd.to_numeric(comparison["daily_close"], errors="coerce")
        - pd.to_numeric(comparison["minute_close"], errors="coerce")
    ).abs()
    max_diff = float(diff.max())
    return {
        "matched_days": int(len(comparison)),
        "max_abs_difference": max_diff,
        "passed": bool(max_diff <= 0.02),
    }


def _date_min(frame: pd.DataFrame, column: str) -> str:
    if frame is None or frame.empty or column not in frame.columns:
        return ""
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return "" if values.empty else values.min().strftime("%Y-%m-%d %H:%M:%S")


def _date_max(frame: pd.DataFrame, column: str) -> str:
    if frame is None or frame.empty or column not in frame.columns:
        return ""
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return "" if values.empty else values.max().strftime("%Y-%m-%d %H:%M:%S")

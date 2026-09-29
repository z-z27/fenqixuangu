"""交易日历: 多源级联 + 本地缓存 + fail-closed。

为什么需要它
------------
``MarketDataService.collect_limit_ups`` 原先只靠 ``current.weekday() >= 5``
排除周末, 节假日无法识别。以 2026-09-25 (中秋节) 为例: 该日被当成交易日,
涨停池接口返回空 -> 回退到全市场日线回扫 (3099 只, 实测每只 5~30s) ->
跑数小时, 而且结果必然是空的 (节假日没有日线 bar)。本模块提供权威交易日历,
让非交易日在发起任何网络请求之前就被剔除。

窗口语义 (刻意保持不变)
------------------------
``lookback_days`` 仍然是**自然日**, 与首版设计一致 (5 自然日用于覆盖周末与
短假期)。本模块**不改变窗口长度**, 只负责把窗口内的非交易日剔出去 ——
因此 ``collect_limit_ups`` 的返回值与修复前逐字节一致 (节假日本来就不产出
行, 只是过去要付出几小时网络代价才发现这件事)。

数据源级联 (按顺序, 首个「成功拉取且覆盖目标区间」者胜出)
--------------------------------------------------------
1. ``akshare_sina_trade_dates``   ak.tool_trade_date_hist_sina(), 覆盖到年末
2. ``baostock_query_trade_dates`` bs.query_trade_dates(), 覆盖到年末
3. ``cached_daily_cross_section`` 由本地日线缓存的横截面反推, 只覆盖历史

1/2 是权威日历源且含未来日期; 3 是离线兜底 (无网络时仍可工作), 其依据是
「在足够多的个股日线样本里, 某日期整体缺席 => 该日不是交易日」。全部失败时
抛 ``TradingCalendarError`` —— fail closed, 绝不猜一个日历继续跑。

缓存
----
``data/cache/trade_calendar/`` 下按源落盘 ``calendar_<source>.pkl`` 与
``calendar_<source>.meta.json`` (coverage / count / fetched_at)。请求区间被
缓存覆盖时不发起任何网络请求; 覆盖不到 (例如跨年) 才重新拉取。
"""

from __future__ import annotations

import json
import os
import threading
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .config import DataConfig


AKSHARE_SINA_SOURCE = "akshare_sina_trade_dates"
BAOSTOCK_SOURCE = "baostock_query_trade_dates"
CACHED_DAILY_SOURCE = "cached_daily_cross_section"

# 级联顺序即优先级; 见模块 docstring。
TRADING_CALENDAR_SOURCES: tuple[str, ...] = (
    AKSHARE_SINA_SOURCE,
    BAOSTOCK_SOURCE,
    CACHED_DAILY_SOURCE,
)

# ``cached_daily_cross_section`` 的置信门槛: 至少读到这么多个非空个股日线缓存
# 才承认其并集可以反推交易日。样本太小时「某日缺席」可能只是缓存没抓过,
# 不足以证明非交易日 -> 宁可让该源失败, 由级联抛错 (fail closed)。
CACHED_DAILY_MIN_SAMPLES = 50
CACHED_DAILY_MAX_SCAN = 400

# baostock 日历查询区间: 足够宽以覆盖任何合理的历史样本窗口。
BAOSTOCK_CALENDAR_START = "1990-01-01"
BAOSTOCK_CALENDAR_FUTURE_DAYS = 400


class TradingCalendarError(RuntimeError):
    """所有交易日历源均不可用 (fail closed)。"""


class NonTradingDayError(RuntimeError):
    """请求的窗口内没有任何交易日 (逐日被权威日历判定为非交易日)。

    由 ``MarketDataService.collect_limit_ups`` 抛出。调用方 (例如
    ``history_samples`` 的证明级联) 可据此把该日期归类为「已证明非交易日」,
    无需再做昂贵的日线扫描。
    """


def _normalize_date(value: object) -> str | None:
    """把源返回的任意日期表示统一成 ``YYYY-MM-DD``; 不可解析返回 None。"""
    if value is None:
        return None
    stamp = pd.to_datetime(value, errors="coerce")
    if stamp is None or pd.isna(stamp):
        return None
    return stamp.strftime("%Y-%m-%d")


def _normalize_dates(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    for value in values:
        text = _normalize_date(value)
        if text is not None:
            seen.add(text)
    return sorted(seen)


class TradingCalendar:
    """多源级联交易日历。

    ``fetchers`` 允许注入 (测试用); 缺省时按 ``sources`` 顺序使用真实数据源。
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        sources: tuple[str, ...] = TRADING_CALENDAR_SOURCES,
        fetchers: dict[str, Callable[[], list[str]]] | None = None,
    ):
        self.config = config
        self.sources = tuple(sources)
        self.root = Path(config.cache_dir) / "trade_calendar"
        self.fetchers = dict(fetchers) if fetchers else self._default_fetchers()
        self._lock = threading.Lock()
        self._dates: list[str] | None = None
        self._date_set: set[str] | None = None
        self._source: str | None = None
        self._start: str | None = None
        self._end: str | None = None
        self.last_origin: str | None = None

    # ---------------------------------------------------------------- 缓存 IO

    def data_path(self, source: str) -> Path:
        return self.root / f"calendar_{source}.pkl"

    def meta_path(self, source: str) -> Path:
        return self.root / f"calendar_{source}.meta.json"

    def _read_cache(self, source: str) -> tuple[list[str], dict] | None:
        data_path = self.data_path(source)
        meta_path = self.meta_path(source)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame = pd.read_pickle(data_path)
        except Exception:
            return None
        if frame is None or frame.empty or "trade_date" not in frame.columns:
            return None
        dates = _normalize_dates(frame["trade_date"].tolist())
        if not dates:
            return None
        return dates, meta

    def _write_cache(self, source: str, dates: list[str], meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data_path = self.data_path(source)
        meta_path = self.meta_path(source)
        # 原子写 (与 FrameCache / Baostock5mCache 同惯用法): tmp -> replace。
        tmp_data = data_path.with_name(
            f"{data_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp{data_path.suffix}")
        pd.DataFrame({"trade_date": dates}).to_pickle(tmp_data)
        tmp_data.replace(data_path)
        tmp_meta = meta_path.with_name(
            f"{meta_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp_meta.write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        tmp_meta.replace(meta_path)

    # ------------------------------------------------------------- 覆盖与加载

    def _covers(self, dates: list[str], start: str, end: str) -> bool:
        return bool(dates) and dates[0] <= start and dates[-1] >= end

    def ensure(self, start: str, end: str) -> None:
        """保证 ``[start, end]`` 已被某源覆盖; 必要时拉取。全部失败则抛错。"""
        if start > end:
            start, end = end, start
        with self._lock:
            if (
                self._date_set is not None
                and self._start is not None
                and self._end is not None
                and self._start <= start
                and self._end >= end
            ):
                return
            errors: list[str] = []
            # 1) 先看缓存: 命中覆盖区间则零网络开销。
            for source in self.sources:
                cached = self._read_cache(source)
                if cached is None:
                    continue
                dates, meta = cached
                if self._covers(dates, start, end):
                    self._adopt(source, dates, origin="cache")
                    return
                errors.append(
                    f"{source}: 缓存覆盖 [{dates[0]}, {dates[-1]}] 不含 [{start}, {end}]")
            # 2) 缓存不覆盖 -> 按顺序拉取。
            for source in self.sources:
                fetcher = self.fetchers.get(source)
                if fetcher is None:
                    errors.append(f"{source}: 无可用取数实现")
                    continue
                try:
                    dates = _normalize_dates(fetcher())
                except Exception as exc:  # noqa: BLE001 - 逐源降级, 汇总后抛错
                    errors.append(f"{source}: {type(exc).__name__}: {exc}")
                    continue
                if not dates:
                    errors.append(f"{source}: 返回空日历")
                    continue
                self._write_cache(
                    source,
                    dates,
                    {
                        "source": source,
                        "start": dates[0],
                        "end": dates[-1],
                        "count": len(dates),
                        "origin": "fetch",
                    },
                )
                if self._covers(dates, start, end):
                    self._adopt(source, dates, origin="fetch")
                    return
                errors.append(
                    f"{source}: 拉取成功但覆盖 [{dates[0]}, {dates[-1]}] 不含 [{start}, {end}]")
            raise TradingCalendarError(
                f"没有可用的交易日历源覆盖 [{start}, {end}]: " + " | ".join(errors))

    def _adopt(self, source: str, dates: list[str], *, origin: str) -> None:
        self._dates = dates
        self._date_set = set(dates)
        self._source = source
        self._start = dates[0]
        self._end = dates[-1]
        self.last_origin = origin

    # ------------------------------------------------------------------ 查询

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def coverage(self) -> tuple[str, str] | None:
        if self._start is None or self._end is None:
            return None
        return self._start, self._end

    def is_trading_day(self, date_text: str) -> bool:
        """``date_text`` 是否为交易日。区间未被覆盖时抛 ``TradingCalendarError``。"""
        target = _normalize_date(date_text)
        if target is None:
            raise TradingCalendarError(f"无法解析日期: {date_text!r}")
        self.ensure(target, target)
        assert self._date_set is not None  # ensure() 成功即已 adopt
        return target in self._date_set

    def is_trading_day_safe(self, date_text: str) -> bool | None:
        """同 ``is_trading_day``, 但日历不可用时返回 None (供可选优化路径使用)。"""
        try:
            return self.is_trading_day(date_text)
        except TradingCalendarError:
            return None

    def trading_days(self, start: str, end: str) -> list[str]:
        self.ensure(start, end)
        assert self._dates is not None
        return [date for date in self._dates if start <= date <= end]

    # --------------------------------------------------------------- 取数实现

    def _default_fetchers(self) -> dict[str, Callable[[], list[str]]]:
        return {
            AKSHARE_SINA_SOURCE: self._fetch_akshare_sina,
            BAOSTOCK_SOURCE: self._fetch_baostock,
            CACHED_DAILY_SOURCE: self._derive_from_cached_daily,
        }

    def _fetch_akshare_sina(self) -> list[str]:
        akshare = _load_module("akshare")
        raw = akshare.tool_trade_date_hist_sina()
        if raw is None or len(raw) == 0 or "trade_date" not in raw.columns:
            raise RuntimeError("akshare tool_trade_date_hist_sina 返回为空或缺少 trade_date 列")
        return _normalize_dates(raw["trade_date"].tolist())

    def _fetch_baostock(self) -> list[str]:
        baostock = _load_module("baostock")
        from .data_sources import login_baostock

        login_baostock()
        end_date = (
            pd.Timestamp.now() + pd.Timedelta(days=BAOSTOCK_CALENDAR_FUTURE_DAYS)
        ).strftime("%Y-%m-%d")
        result = baostock.query_trade_dates(
            start_date=BAOSTOCK_CALENDAR_START, end_date=end_date)
        if str(getattr(result, "error_code", "")) not in ("0", ""):
            raise RuntimeError(
                f"baostock query_trade_dates 失败: error_code={getattr(result, 'error_code', '')} "
                f"error_msg={getattr(result, 'error_msg', '')}")
        frame = result.get_data()
        if frame is None or frame.empty or "is_trading_day" not in frame.columns:
            raise RuntimeError("baostock query_trade_dates 返回为空或缺少 is_trading_day 列")
        trading = frame[frame["is_trading_day"].astype(str).str.strip() == "1"]
        return _normalize_dates(trading["calendar_date"].tolist())

    def _derive_from_cached_daily(self) -> list[str]:
        """由本地日线缓存横截面反推交易日 (离线兜底)。

        在足够多的个股样本里, 某日期整体缺席 => 不是交易日。样本不足时抛错
        (fail closed), 不拿稀疏缓存硬推。
        """
        candidates: list[Path] = []
        for name, suffix in (("daily_unadjusted", "daily"), ("daily", "daily")):
            root = Path(self.config.cache_dir) / name
            if not root.exists():
                continue
            candidates.extend(root.glob(f"*_{suffix}.pkl"))
        if not candidates:
            raise RuntimeError("本地日线缓存为空, 无法反推交易日")
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        dates: set[str] = set()
        sampled = 0
        for path in candidates[:CACHED_DAILY_MAX_SCAN]:
            try:
                frame = pd.read_pickle(path)
            except Exception:
                continue
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            try:
                values = pd.to_datetime(frame["date"], errors="coerce").dropna()
            except Exception:
                continue
            if values.empty:
                continue
            sampled += 1
            dates.update(values.dt.strftime("%Y-%m-%d").unique().tolist())
        if sampled < CACHED_DAILY_MIN_SAMPLES:
            raise RuntimeError(
                f"本地日线缓存样本不足 ({sampled} < {CACHED_DAILY_MIN_SAMPLES}), 无法反推交易日")
        return sorted(dates)


_MODULE_CACHE: dict[str, object] = {}


def _load_module(name: str):
    """惰性加载可选依赖 (与 data_sources.load_akshare 同惯用法)。"""
    if name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[name] = import_module(name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"missing dependency {name}; run python -m pip install -r requirements.txt"
            ) from exc
    return _MODULE_CACHE[name]


_CALENDARS: dict[str, TradingCalendar] = {}
_CALENDARS_LOCK = threading.Lock()


def get_trading_calendar(config: DataConfig | None = None) -> TradingCalendar:
    """进程内共享的交易日历 (按 cache_dir 去重; 构造本身无 IO)。"""
    if config is None:
        from .config import get_data_config

        config = get_data_config()
    key = str(Path(config.cache_dir).resolve())
    with _CALENDARS_LOCK:
        calendar = _CALENDARS.get(key)
        if calendar is None:
            calendar = TradingCalendar(config)
            _CALENDARS[key] = calendar
        return calendar

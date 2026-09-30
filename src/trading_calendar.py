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
因此 ``collect_limit_ups`` 的自然日窗口边界不变；窗口内的实际处理日期由
交易日历决定。来源失败时现在拒绝发布不完整的窗口结果。

证据方向: OPEN 与 CLOSED 是不对称的
-----------------------------------
日历会被问两类相反的问题:

* 「这一天是交易日吗?」-> 需要**正面证据**, 任何一个源列出这一天就够了。
* 「这一天不是交易日吗?」-> 需要**负面证据**, 只有当一个源**枚举**了整个自然
  日区间、且该日不在枚举结果里时才成立。

正面证据廉价, 负面证据昂贵。以本地日线缓存为例: 某日整体缺席, 既可能是休市,
也可能是缓存缺口 / 抓取失败 / 历史截断 / 供应商故障 —— 两者无法区分。因此:

* ``is_trading_day()`` 返回 ``False`` 之前, 必须先由**声明了覆盖区间**的源证明
  该日落在区间内; 区间之外抛 ``TradingCalendarError``, 而不是猜。
* ``cached_daily_cross_section`` **不再是日历源** (审计 C01)。它只能提供正面
  证据, 保留为 ``cached_daily_open_sessions()`` 供诊断 / 交叉核对。

数据源级联 (按顺序, 首个「成功拉取且覆盖目标区间」者胜出)
--------------------------------------------------------
1. ``akshare_sina_trade_dates``   ak.tool_trade_date_hist_sina()
2. ``baostock_query_trade_dates`` bs.query_trade_dates()

两者都枚举交易所交易日, 因此都能给出负面证据。全部失败时抛
``TradingCalendarError`` —— fail closed: 绝不猜一个日历继续跑, 也绝不退回
``weekday()`` 或本地缓存缺席。

覆盖区间 (coverage) != 返回的交易日列表
---------------------------------------
``coverage`` 定义为「在该**自然日**区间内, 缺席 == CLOSED」的区间, 由各源自己
**声明**并当场校验, 绝不从 ``dates[0]`` / ``dates[-1]`` 反推 —— 否则区间末端
一个已知的休市日 (例如年末最后一天) 会被误判成「源没覆盖」而白白失败 (审计
C03)。两个内置源的声明方式不同:

* baostock 对请求区间内的**每个自然日**返回一行 (``calendar_date`` /
  ``is_trading_day``), 因此请求区间本身就是覆盖区间; 取数时逐条核对行数与首尾,
  核对不过就退回「只覆盖它显式返回的部分」。
* akshare 的 sina 表是**按年预发布**的交易所日历 (含尚未发生的交易日), 因此
  覆盖到最后一个已发布年份的 12-31; 「含未来交易日」这一前提当场校验, 不成立
  (旧快照 / 被截断) 时覆盖退回最后一行, 之后的日子保持不可证明。

缓存
----
``data/cache/trade_calendar/`` 下按源落盘 ``calendar_<source>.pkl`` 与
``calendar_<source>.meta.json`` (schema / source / coverage / count)。请求区间被
缓存覆盖时不发起任何网络请求; 覆盖不到 (例如跨年) 才重新拉取。旧 schema 的缓存
没有覆盖声明, 无法追溯证明, 一律当作未命中重新拉取。

冻结快照
--------
``write_frozen_calendar_snapshot()`` 把当前已采纳的日历冻结成一个带 SHA-256 的
本地权威工件; ``TradingCalendar(..., frozen_snapshot=path)`` 让它成为级联里最高
优先级、只读的来源。这样 corrected rebuild 可以引用同一个日历工件, 而不是依赖
「今天 AkShare 恰好返回了什么」。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .config import DataConfig


AKSHARE_SINA_SOURCE = "akshare_sina_trade_dates"
BAOSTOCK_SOURCE = "baostock_query_trade_dates"
FROZEN_SNAPSHOT_SOURCE = "verified_calendar_snapshot"
# 不再是日历源 (审计 C01); 名称保留, 因为它仍出现在旧调用点与报告中,
# 显式引用它时会得到一条说明原因的报错。
CACHED_DAILY_SOURCE = "cached_daily_cross_section"
CALENDAR_SEMANTICS_VERSION = "trading_sessions_v1"

# 级联顺序即优先级; 见模块 docstring。这里只放**能给出 CLOSED 证据**的源。
TRADING_CALENDAR_SOURCES: tuple[str, ...] = (
    AKSHARE_SINA_SOURCE,
    BAOSTOCK_SOURCE,
)

# 缓存 schema: 2 起 meta 必须显式声明覆盖区间。旧缓存的覆盖无法追溯证明,
# 一律当作未命中重新拉取 (宁可 fail closed, 也不静默沿用无法验证的覆盖)。
CALENDAR_CACHE_SCHEMA = 2

# ``cached_daily_open_sessions`` 的置信门槛: 至少读到这么多个非空个股日线缓存
# 才承认其并集可以作为**正面**交易日证据。样本太小时连正面证据也不给。
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


def _coverage_covers(coverage: tuple[str, str] | None, start: str, end: str) -> bool:
    return coverage is not None and coverage[0] <= start and coverage[1] >= end


def _format_coverage(coverage: tuple[str, str] | None) -> str:
    return "无(仅正面证据)" if coverage is None else f"[{coverage[0]}, {coverage[1]}]"


@dataclass(frozen=True)
class CalendarFetch:
    """一个源的回答: OPEN 交易日列表 + 它能证明 CLOSED 的自然日区间。

    ``coverage`` 为 ``None`` 表示该结果**没有**负面证据能力: 它列出过的是交易日,
    但没列出的什么都不能说明。区间必须由源声明、或在取数时当场校验, 绝不从
    ``dates`` 的首尾反推。
    """

    dates: tuple[str, ...]
    coverage_start: str | None = None
    coverage_end: str | None = None

    @property
    def coverage(self) -> tuple[str, str] | None:
        if self.coverage_start is None or self.coverage_end is None:
            return None
        return self.coverage_start, self.coverage_end

    def covers(self, start: str, end: str) -> bool:
        return _coverage_covers(self.coverage, start, end)


def _as_fetch(raw: object) -> CalendarFetch:
    """归一化 fetcher 的返回值。

    ``CalendarFetch`` 原样使用。裸 ``list[str]`` 解释为**注入方声明**的枚举型
    权威源: 覆盖区间即其返回的首尾日期。内置源的取数实现一律返回显式
    ``CalendarFetch`` (见 ``_fetch_akshare_sina`` / ``_fetch_baostock``),
    绝不走这条捷径。
    """
    if isinstance(raw, CalendarFetch):
        return raw
    dates = tuple(_normalize_dates(raw))  # type: ignore[arg-type]
    if not dates:
        return CalendarFetch(dates=())
    return CalendarFetch(dates=dates, coverage_start=dates[0], coverage_end=dates[-1])


def _akshare_sina_coverage(dates: list[str], *, today: str | None = None) -> tuple[str, str] | None:
    """sina 表的覆盖区间 —— 从表本身当场判定, 不套用「覆盖到年末」的惯例。

    该表是**按年预发布**的交易所日历, 不是「已观测交易日」的日志: 它包含尚未发生
    的交易日。一个会列出未来交易日的表必然整年枚举, 所以
    ``[首行, 最后一个已发布年份的 12-31]`` 内每一天都被分类过, 缺席即休市。

    「含未来交易日」与「末行落在 12 月」两个前提当场校验。不成立时 (旧快照、被
    截断的表) 覆盖退回到最后一行, 其后的日子保持不可证明 —— 调用方 fail closed,
    而不是把「源没覆盖」误当休市。
    """
    if not dates:
        return None
    last = pd.Timestamp(dates[-1])
    reference = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now().normalize()
    if last <= reference or last.month != 12:
        return dates[0], dates[-1]
    return dates[0], f"{last.year:04d}-12-31"


def _baostock_coverage(frame: pd.DataFrame, end_date: str) -> tuple[str, str] | None:
    """baostock 的覆盖区间 —— 由响应当场校验, 不靠对 API 的假设。

    ``query_trade_dates`` 的约定是「对请求区间内的**每个自然日**返回一行, 用
    ``is_trading_day`` 标记是否开市」。约定成立时, 请求区间本身就是覆盖区间:
    区间内 ``is_trading_day == 0`` 的行直接证明了休市。这里逐条核对 (行数等于
    自然日数、首尾对齐); 核对不过就退回「只覆盖它显式返回的那段」, 区间之外
    保持不可证明 -> 调用方 fail closed。
    """
    if "calendar_date" not in frame.columns:
        return None
    calendar_dates = _normalize_dates(frame["calendar_date"].tolist())
    if not calendar_dates:
        return None
    expected_days = (pd.Timestamp(end_date) - pd.Timestamp(BAOSTOCK_CALENDAR_START)).days + 1
    if (
        len(calendar_dates) == expected_days
        and calendar_dates[0] == BAOSTOCK_CALENDAR_START
        and calendar_dates[-1] == end_date
    ):
        return BAOSTOCK_CALENDAR_START, end_date
    return calendar_dates[0], calendar_dates[-1]


class TradingCalendar:
    """多源级联交易日历。

    ``fetchers`` 允许注入 (测试用); 缺省时按 ``sources`` 顺序使用真实数据源。
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        sources: tuple[str, ...] = TRADING_CALENDAR_SOURCES,
        fetchers: dict[str, Callable[[], object]] | None = None,
        frozen_snapshot: str | Path | None = None,
    ):
        self.config = config
        self.sources = tuple(sources)
        self.root = Path(config.cache_dir) / "trade_calendar"
        self.fetchers = dict(fetchers) if fetchers is not None else self._default_fetchers()
        self._lock = threading.Lock()
        self._dates: list[str] | None = None
        self._date_set: set[str] | None = None
        self._source: str | None = None
        self._coverage: tuple[str, str] | None = None
        self.snapshot_hash: str | None = None
        self.last_origin: str | None = None
        if frozen_snapshot is not None:
            # 冻结快照优先于任何在线源: corrected rebuild 必须能引用同一个日历工件。
            snapshot = load_frozen_calendar_snapshot(Path(frozen_snapshot))
            self.sources = (FROZEN_SNAPSHOT_SOURCE, *self.sources)
            self.fetchers[FROZEN_SNAPSHOT_SOURCE] = lambda: snapshot.fetch
            self.snapshot_hash = snapshot.sha256

    # ---------------------------------------------------------------- 缓存 IO

    def data_path(self, source: str) -> Path:
        return self.root / f"calendar_{source}.pkl"

    def meta_path(self, source: str) -> Path:
        return self.root / f"calendar_{source}.meta.json"

    def _read_cache(self, source: str) -> tuple[list[str], tuple[str, str] | None, dict] | None:
        data_path = self.data_path(source)
        meta_path = self.meta_path(source)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame = pd.read_pickle(data_path)
        except Exception:
            return None
        if (
            not isinstance(meta, dict)
            or meta.get("schema") != CALENDAR_CACHE_SCHEMA
            or not isinstance(frame, pd.DataFrame)
            or frame.empty
            or "trade_date" not in frame.columns
        ):
            return None
        dates = _normalize_dates(frame["trade_date"].tolist())
        if not dates:
            return None
        if (
            meta.get("source") != source
            or meta.get("start") != dates[0]
            or meta.get("end") != dates[-1]
            or meta.get("count") != len(dates)
        ):
            return None
        coverage_start = _normalize_date(meta.get("coverage_start"))
        coverage_end = _normalize_date(meta.get("coverage_end"))
        if coverage_start is None or coverage_end is None or coverage_start > coverage_end:
            # 没有可验证的覆盖声明: 交易日列表照旧可用, 但负面证据不成立。
            return dates, None, meta
        return dates, (coverage_start, coverage_end), meta

    def _write_cache(self, source: str, fetch: CalendarFetch) -> None:
        dates = list(fetch.dates)
        coverage = fetch.coverage
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
            json.dumps(
                {
                    "schema": CALENDAR_CACHE_SCHEMA,
                    "source": source,
                    "start": dates[0],
                    "end": dates[-1],
                    "count": len(dates),
                    "coverage_start": coverage[0] if coverage else None,
                    "coverage_end": coverage[1] if coverage else None,
                    "origin": "fetch",
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=1,
            ),
            encoding="utf-8",
        )
        tmp_meta.replace(meta_path)

    # ------------------------------------------------------------- 覆盖与加载

    def _covers(self, start: str, end: str) -> bool:
        """当前已采纳的日历能否在 ``[start, end]`` 内给出 CLOSED 证据。"""
        return _coverage_covers(self._coverage, start, end)

    def ensure(self, start: str, end: str) -> None:
        """保证 ``[start, end]`` 被某源**证明**; 必要时拉取。全部失败则抛错。

        这里的「证明」是双向的: 区间内列出的日期是交易日, 区间内没列出的日期是
        非交易日。因此只接受声明了覆盖区间的源。
        """
        if start > end:
            start, end = end, start
        with self._lock:
            if self._date_set is not None and self._covers(start, end):
                return
            errors: list[str] = []
            # 1) 先看缓存: 命中覆盖区间则零网络开销。
            for source in self.sources:
                cached = self._read_cache(source)
                if cached is None:
                    continue
                dates, coverage, _meta = cached
                if _coverage_covers(coverage, start, end):
                    self._adopt(source, dates, coverage, origin="cache")
                    return
                errors.append(
                    f"{source}: 缓存覆盖 {_format_coverage(coverage)} 不含 [{start}, {end}]")
            # 2) 缓存不覆盖 -> 按顺序拉取。
            for source in self.sources:
                fetcher = self.fetchers.get(source)
                if fetcher is None:
                    errors.append(f"{source}: 无可用取数实现")
                    continue
                try:
                    fetch = _as_fetch(fetcher())
                except Exception as exc:  # noqa: BLE001 - 逐源降级, 汇总后抛错
                    errors.append(f"{source}: {type(exc).__name__}: {exc}")
                    continue
                if not fetch.dates:
                    errors.append(f"{source}: 返回空日历")
                    continue
                self._write_cache(source, fetch)
                if fetch.covers(start, end):
                    self._adopt(source, list(fetch.dates), fetch.coverage, origin="fetch")
                    return
                errors.append(
                    f"{source}: 拉取成功但覆盖 {_format_coverage(fetch.coverage)} "
                    f"不含 [{start}, {end}]")
            raise TradingCalendarError(
                f"没有可用的交易日历源覆盖 [{start}, {end}]: " + " | ".join(errors))

    def _adopt(
        self,
        source: str,
        dates: list[str],
        coverage: tuple[str, str] | None,
        *,
        origin: str,
    ) -> None:
        self._dates = dates
        self._date_set = set(dates)
        self._source = source
        self._coverage = coverage
        self.last_origin = origin

    # ------------------------------------------------------------------ 查询

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def coverage(self) -> tuple[str, str] | None:
        """当前源能对其中**缺席**日期下 CLOSED 结论的自然日区间。

        注意这是「源保证分类过」的区间, 不是「首尾交易日」—— 区间末端可以是
        休市日。``None`` 表示当前没有任何源, 或源只提供正面证据。
        """
        return self._coverage

    @property
    def closure_authority(self) -> bool:
        """当前日历能否把「不在列表里」解释成非交易日。"""
        return self._coverage is not None

    def provenance(self) -> dict[str, object]:
        """写入 corrected artifact manifest 的最小日历追溯信息 (审计 §17)。"""
        return {
            "calendar_semantics_version": CALENDAR_SEMANTICS_VERSION,
            "calendar_source": self._source,
            "calendar_coverage_start": self._coverage[0] if self._coverage else None,
            "calendar_coverage_end": self._coverage[1] if self._coverage else None,
            "calendar_origin": self.last_origin,
            "calendar_snapshot_sha256": self.snapshot_hash,
        }

    def is_trading_day(self, date_text: str) -> bool:
        """``date_text`` 是否为交易日。

        ``False`` 需要负面证据: ``ensure()`` 只有在某源声明覆盖了该日期所在区间
        时才会返回, 因此走到这里的 ``False`` 一定是「源枚举过、这一天不在里面」,
        而不是「源没看过这一天」。区间未被覆盖时抛 ``TradingCalendarError``。
        """
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
        """Return actual sessions in the inclusive interval; reversed ranges are invalid."""
        start = self._required_date(start)
        end = self._required_date(end)
        if start > end:
            raise ValueError("trading_days requires start <= end")
        self.ensure(start, end)
        assert self._dates is not None
        return self._dates[bisect_left(self._dates, start):bisect_right(self._dates, end)]

    @staticmethod
    def _required_date(value: object) -> str:
        normalized = _normalize_date(value)
        if normalized is None:
            raise TradingCalendarError(f"无法解析日期: {value!r}")
        return normalized

    def session_distance(self, start: str, end: str) -> int:
        """Number of actual sessions from start to end (0 on the same session).

        Both endpoints must be sessions. Reversed dates, missing coverage and
        non-session endpoints raise rather than approximating with weekdays.
        """
        start = self._required_date(start)
        end = self._required_date(end)
        if start > end:
            raise ValueError("session_distance requires start <= end")
        self.ensure(start, end)
        assert self._dates is not None and self._date_set is not None
        if start not in self._date_set or end not in self._date_set:
            raise TradingCalendarError(f"session_distance endpoints must be trading days: {start}, {end}")
        return bisect_left(self._dates, end) - bisect_left(self._dates, start)

    def are_consecutive_sessions(self, previous: str, next_date: str) -> bool:
        """True only when next_date immediately follows previous in market time."""
        return self.session_distance(previous, next_date) == 1

    def previous_trading_day(self, date_text: str) -> str:
        """Nearest session strictly before date_text; fail if coverage is insufficient."""
        target = self._required_date(date_text)
        self.ensure(target, target)
        assert self._dates is not None
        index = bisect_left(self._dates, target)
        if index:
            return self._dates[index - 1]
        start = (pd.Timestamp(target) - pd.Timedelta(days=32)).strftime("%Y-%m-%d")
        self.ensure(start, target)
        assert self._dates is not None
        index = bisect_left(self._dates, target)
        if not index:
            raise TradingCalendarError(f"no previous trading day covered before {target}")
        return self._dates[index - 1]

    def next_trading_day(self, date_text: str) -> str:
        """Nearest session strictly after date_text; fail if coverage is insufficient."""
        target = self._required_date(date_text)
        self.ensure(target, target)
        assert self._dates is not None
        index = bisect_right(self._dates, target)
        if index < len(self._dates):
            return self._dates[index]
        end = (pd.Timestamp(target) + pd.Timedelta(days=32)).strftime("%Y-%m-%d")
        self.ensure(target, end)
        assert self._dates is not None
        index = bisect_right(self._dates, target)
        if index == len(self._dates):
            raise TradingCalendarError(f"no next trading day covered after {target}")
        return self._dates[index]

    # --------------------------------------------------------------- 取数实现

    def _default_fetchers(self) -> dict[str, Callable[[], object]]:
        return {
            AKSHARE_SINA_SOURCE: self._fetch_akshare_sina,
            BAOSTOCK_SOURCE: self._fetch_baostock,
            # 保留注册只为让显式引用旧源名时得到一条说明原因的报错。
            CACHED_DAILY_SOURCE: self._refuse_cached_daily_as_calendar_source,
        }

    def _fetch_akshare_sina(self) -> CalendarFetch:
        akshare = _load_module("akshare")
        raw = akshare.tool_trade_date_hist_sina()
        if raw is None or len(raw) == 0 or "trade_date" not in raw.columns:
            raise RuntimeError("akshare tool_trade_date_hist_sina 返回为空或缺少 trade_date 列")
        dates = _normalize_dates(raw["trade_date"].tolist())
        coverage = _akshare_sina_coverage(dates)
        if coverage is None:
            return CalendarFetch(dates=tuple(dates))
        return CalendarFetch(dates=tuple(dates), coverage_start=coverage[0], coverage_end=coverage[1])

    def _fetch_baostock(self) -> CalendarFetch:
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
        dates = _normalize_dates(trading["calendar_date"].tolist())
        coverage = _baostock_coverage(frame, end_date)
        if coverage is None:
            return CalendarFetch(dates=tuple(dates))
        return CalendarFetch(dates=tuple(dates), coverage_start=coverage[0], coverage_end=coverage[1])

    def _refuse_cached_daily_as_calendar_source(self) -> CalendarFetch:
        raise TradingCalendarError(
            f"{CACHED_DAILY_SOURCE} 不再是交易日历源 (审计 C01): 本地日线缓存里某日"
            "缺席, 既可能是休市, 也可能是缓存缺口/抓取失败/历史截断, 无法证明 CLOSED。"
            f"请改用 {AKSHARE_SINA_SOURCE} / {BAOSTOCK_SOURCE}, 或改用 "
            "cached_daily_open_sessions() 取正面证据。")

    def cached_daily_open_sessions(self, start: str, end: str) -> list[str]:
        """本地日线缓存佐证过的交易日 —— **只有正面 OPEN 证据**, 供诊断/核对。

        绝不能用来证明某日 CLOSED, 也不能当作 D0/D1/D2 的会话枚举: 缓存缺口与
        休市在数据上不可区分 (审计 C01)。``start`` / ``end`` 只用于裁剪结果, 不
        意味着缓存覆盖了该区间。样本不足时抛 ``TradingCalendarError``。
        """
        start = self._required_date(start)
        end = self._required_date(end)
        if start > end:
            raise ValueError("cached_daily_open_sessions requires start <= end")
        return [date for date in self._derive_from_cached_daily() if start <= date <= end]

    def _derive_from_cached_daily(self) -> list[str]:
        """本地日线缓存横截面里出现过的日期。

        某日在足够多的个股日线样本里出现, 可以支持「该日是交易日」; 反过来,
        某日整体缺席**什么也不能证明** —— 这正是审计 C01 指出的证据方向问题。
        因此本方法只产出正面证据, 不作为日历源。
        """
        candidates: list[Path] = []
        for name, suffix in (("daily_unadjusted", "daily"), ("daily", "daily")):
            root = Path(self.config.cache_dir) / name
            if not root.exists():
                continue
            candidates.extend(root.glob(f"*_{suffix}.pkl"))
        if not candidates:
            raise TradingCalendarError("本地日线缓存为空, 给不出正面交易日证据")
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        dates: set[str] = set()
        sampled = 0
        for path in candidates[:CACHED_DAILY_MAX_SCAN]:
            try:
                frame = pd.read_pickle(path)
            except Exception:
                continue
            if not isinstance(frame, pd.DataFrame) or frame.empty or "date" not in frame.columns:
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
            raise TradingCalendarError(
                f"本地日线缓存样本不足 ({sampled} < {CACHED_DAILY_MIN_SAMPLES}), "
                "给不出正面交易日证据")
        return sorted(dates)


FROZEN_SNAPSHOT_SCHEMA = 1


def _snapshot_digest(source: str, coverage: tuple[str, str], dates: list[str]) -> str:
    """对「源 + 覆盖区间 + 有序交易日序列」求 SHA-256。

    不含时间戳等易变字段: 同一份日历重复冻结必须得到同一个 hash, 否则
    "artifact 没变" 这个判断本身就没有意义。
    """
    payload = json.dumps(
        {"source": source, "coverage": list(coverage), "dates": dates},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenCalendarSnapshot:
    """已校验的冻结日历工件。"""

    fetch: CalendarFetch
    source: str
    sha256: str


def write_frozen_calendar_snapshot(calendar: TradingCalendar, path: str | Path) -> FrozenCalendarSnapshot:
    """把当前已采纳的日历冻结成带 SHA-256 的权威工件 (审计 §18)。

    冻结前要求日历**必须有覆盖声明**: 只提供正面证据的日历没有可冻结的负面证据
    契约, 冻结它等于把「未知」写成「权威」。路径采用 tmp -> replace 原子写, 与
    其余缓存一致。
    """
    if calendar.source is None or calendar.coverage is None:
        raise TradingCalendarError(
            "只有在已采纳带覆盖声明的权威日历后才能冻结快照; 当前日历没有 CLOSED 证据能力")
    if calendar._dates is None:  # noqa: SLF001 - 同类内部状态, 同一模块内使用
        raise TradingCalendarError("日历尚未加载, 无法冻结快照")
    dates = list(calendar._dates)  # noqa: SLF001
    coverage = calendar.coverage
    digest = _snapshot_digest(calendar.source, coverage, dates)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp{target.suffix}")
    tmp.write_text(
        json.dumps(
            {
                "schema": FROZEN_SNAPSHOT_SCHEMA,
                "calendar_semantics_version": CALENDAR_SEMANTICS_VERSION,
                "source": calendar.source,
                "coverage_start": coverage[0],
                "coverage_end": coverage[1],
                "count": len(dates),
                "sha256": digest,
                "dates": dates,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=1,
        ),
        encoding="utf-8",
    )
    tmp.replace(target)
    return FrozenCalendarSnapshot(
        fetch=CalendarFetch(dates=tuple(dates), coverage_start=coverage[0], coverage_end=coverage[1]),
        source=calendar.source,
        sha256=digest,
    )


def load_frozen_calendar_snapshot(path: str | Path) -> FrozenCalendarSnapshot:
    """读取并校验冻结快照。文件缺失、结构不符或 hash 对不上一律抛错。

    不静默采纳: 一个被改动过的日历工件如果不报错, 就会以「权威」的名义把错误的
    会话边界带进 corrected rebuild。
    """
    target = Path(path)
    if not target.exists():
        raise TradingCalendarError(f"冻结日历快照不存在: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TradingCalendarError(f"冻结日历快照无法解析: {target}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != FROZEN_SNAPSHOT_SCHEMA:
        raise TradingCalendarError(f"冻结日历快照 schema 不受支持: {target}")
    if payload.get("calendar_semantics_version") != CALENDAR_SEMANTICS_VERSION:
        raise TradingCalendarError(
            f"冻结日历快照的日历语义版本与当前代码不一致: "
            f"{payload.get('calendar_semantics_version')!r} != {CALENDAR_SEMANTICS_VERSION!r}")
    source = payload.get("source")
    coverage = (_normalize_date(payload.get("coverage_start")), _normalize_date(payload.get("coverage_end")))
    dates = _normalize_dates(payload.get("dates") or [])
    if not isinstance(source, str) or not source or coverage[0] is None or coverage[1] is None or not dates:
        raise TradingCalendarError(f"冻结日历快照字段不完整: {target}")
    expected = _snapshot_digest(source, (coverage[0], coverage[1]), dates)
    if payload.get("sha256") != expected:
        raise TradingCalendarError(
            f"冻结日历快照 SHA-256 校验失败 (文件被改动过?): {target}")
    return FrozenCalendarSnapshot(
        fetch=CalendarFetch(dates=tuple(dates), coverage_start=coverage[0], coverage_end=coverage[1]),
        source=source,
        sha256=expected,
    )


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

from __future__ import annotations

from datetime import datetime
from importlib import import_module
import json
import socket
from typing import Any, Callable

import numpy as np
import pandas as pd

from .code_utils import (
    detect_market,
    is_excluded_name,
    is_main_board_code,
    normalize_stock_code,
    to_eastmoney_secid,
    to_market_symbol,
)
from .config import DataConfig
from .http_client import RequestClient


SINA_5M_SOURCE = "sina_5m"
BAOSTOCK_5M_SOURCE = "baostock_5m"
VALID_5M_SOURCES = (SINA_5M_SOURCE, BAOSTOCK_5M_SOURCE)
NORMALIZED_5M_INTERVAL = "5m"
BAOSTOCK_5M_FREQUENCY = "5"
BAOSTOCK_5M_ADJUSTFLAG_NO_ADJUST = "3"
# baostock 0.9.x 起数据服务迁移至 public-api.baostock.com (旧 www.baostock.com
# 的 10030 数据端口已退役, 0.8.x 客户端将无法登录)。
BAOSTOCK_SERVER_HOST = "public-api.baostock.com"
BAOSTOCK_SERVER_PORT = 10030

LISTING_METADATA_SOURCE = "eastmoney_stock_info_f189"
SUSPENSION_STATUS_SOURCE = "eastmoney_RPT_CUSTOM_SUSPEND_DATA_INTERFACE_via_akshare"
SUSPENSION_STATUS_COLUMNS = (
    "code",
    "name",
    "suspension_start_date",
    "suspension_end_date",
    "suspension_duration",
    "suspension_reason",
    "suspension_market",
    "expected_resume_date",
)


class MarketDataProvider:
    def __init__(self, config: DataConfig):
        self.config = config
        self.client = RequestClient(config)

    def fetch_limit_up_pool(self, trade_date: str | None = None) -> tuple[pd.DataFrame, str]:
        date_text = normalize_date_text(trade_date or datetime.now().strftime("%Y-%m-%d"))
        attempts: list[tuple[str, Callable[[], pd.DataFrame]]] = [
            ("akshare_zt_pool_em", lambda: self._fetch_limit_up_pool_akshare(date_text)),
        ]
        if date_text == datetime.now().strftime("%Y-%m-%d"):
            attempts.append(("eastmoney_spot_approx", self._fetch_limit_up_pool_from_spot))
        errors: list[str] = []
        for source, fetcher in attempts:
            try:
                frame = fetcher()
                if frame is None or frame.empty:
                    raise RuntimeError("empty limit-up pool")
                frame = normalize_limit_up_pool(frame, date_text, source)
                frame = filter_main_board(frame)
                if frame.empty:
                    raise RuntimeError("main-board limit-up pool is empty")
                return frame, source
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        raise RuntimeError("all limit-up sources failed: " + " | ".join(errors))

    def fetch_stock_universe(self) -> tuple[pd.DataFrame, str]:
        frame = self._fetch_spot_eastmoney()
        if frame is None or frame.empty:
            raise RuntimeError("empty stock universe")
        frame = normalize_stock_universe(frame, "eastmoney_spot_universe")
        frame = filter_main_board(frame)
        if frame.empty:
            raise RuntimeError("main-board stock universe is empty")
        return frame, "eastmoney_spot_universe"

    def fetch_listing_metadata(self, code: str) -> dict[str, str]:
        """Fetch a verified exchange listing date from Eastmoney stock metadata."""
        normalized = normalize_stock_code(code)
        payload = self.client.get_json(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "fltt": "2",
                "invt": "2",
                "fields": "f57,f58,f189",
                "secid": to_eastmoney_secid(normalized),
            },
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise RuntimeError(f"{normalized} Eastmoney listing metadata response missing data")
        returned_code = normalize_stock_code(data.get("f57", ""))
        if returned_code != normalized:
            raise RuntimeError(
                f"{normalized} Eastmoney listing metadata code mismatch: returned={returned_code}"
            )
        listing_date = normalize_listing_date(data.get("f189"))
        if not listing_date:
            raise RuntimeError(
                f"{normalized} Eastmoney listing metadata missing or invalid f189 listing date"
            )
        return {
            "code": normalized,
            "listing_date": listing_date,
            "metadata_source": LISTING_METADATA_SOURCE,
        }

    def fetch_suspension_status(self, query_date: str) -> tuple[pd.DataFrame, str]:
        """Fetch and normalize Eastmoney's suspension report for one query date.

        The upstream report may contain records whose suspension interval does not
        cover ``query_date``.  Callers must prove interval coverage rather than
        treating report membership as proof of an active suspension.
        """
        date_text = normalize_date_text(query_date)
        ak = load_akshare()
        raw = ak.stock_tfp_em(date=date_text.replace("-", ""))
        if raw is None:
            raise RuntimeError(f"Eastmoney suspension report returned no frame for {date_text}")
        return normalize_suspension_status_frame(raw), SUSPENSION_STATUS_SOURCE

    def probe_daily_sources_for_date(
        self,
        code: str,
        query_date: str,
        adjust: str = "none",
    ) -> dict[str, Any]:
        """Query Tencent and Sina independently for exact daily-date coverage."""
        normalized = normalize_stock_code(code)
        date_text = normalize_date_text(query_date)
        start_text = (pd.Timestamp(date_text) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        attempts = (
            (
                "tencent_daily",
                lambda: self._fetch_daily_tencent(normalized, start_text, date_text, adjust),
            ),
            (
                "sina_daily",
                lambda: self._fetch_daily_sina(normalized, start_text, date_text, adjust),
            ),
        )
        rows: list[dict[str, Any]] = []
        for source, fetcher in attempts:
            try:
                frame = normalize_daily_frame(fetcher(), normalized, source)
                dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
                if dates.empty:
                    raise RuntimeError("normalized daily frame has no valid dates")
                normalized_dates = dates.dt.strftime("%Y-%m-%d")
                rows.append(
                    {
                        "source": source,
                        "status": "ok",
                        "has_requested_date": bool(normalized_dates.eq(date_text).any()),
                        "latest_date": str(normalized_dates.max()),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "source": source,
                        "status": "error",
                        "has_requested_date": False,
                        "latest_date": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "code": normalized,
            "query_date": date_text,
            "sources": rows,
        }

    def fetch_daily_history(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str = "none",
    ) -> tuple[pd.DataFrame, str]:
        normalized = normalize_stock_code(code)
        source_attempts = [
            ("tencent_daily", lambda: self._fetch_daily_tencent(normalized, start_date, end_date, adjust)),
            ("sina_daily", lambda: self._fetch_daily_sina(normalized, start_date, end_date, adjust)),
        ]
        errors: list[str] = []
        for source, fetcher in source_attempts:
            try:
                frame = fetcher()
                if frame is None or frame.empty:
                    raise RuntimeError("empty daily frame")
                frame = normalize_daily_frame(frame, normalized, source)
                return frame, source
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        raise RuntimeError(f"{normalized} all daily sources failed: " + " | ".join(errors))

    def fetch_5min_history(
        self,
        code: str,
        start_datetime: str,
        end_datetime: str,
        adjust: str = "none",
        source: str = SINA_5M_SOURCE,
    ) -> tuple[pd.DataFrame, str]:
        """Fetch 5m bars from an explicitly selected source.

        source="sina_5m" (default, backward compatible) or "baostock_5m".
        There is NO silent fallback: the requested source alone is attempted;
        any failure raises RuntimeError. Unknown source raises ValueError.
        """
        if source not in VALID_5M_SOURCES:
            raise ValueError(
                f"unknown 5m source {source!r} (expected one of {sorted(VALID_5M_SOURCES)})")
        normalized = normalize_stock_code(code)
        if source == SINA_5M_SOURCE:
            try:
                frame = self._fetch_5min_sina(normalized, start_datetime, end_datetime, adjust)
                if frame is None or frame.empty:
                    raise RuntimeError("empty 5m frame")
                frame = normalize_5min_frame(frame, normalized, source, adjust)
                if frame.empty:
                    raise RuntimeError("normalized 5m frame is empty")
                return frame, source
            except Exception as exc:
                raise RuntimeError(f"{normalized} sina_5m failed: {exc}") from exc
        frame = self._fetch_5min_baostock(normalized, start_datetime, end_datetime, adjust)
        if frame is None or frame.empty:
            raise RuntimeError(f"{normalized} baostock_5m empty raw frame")
        frame = normalize_baostock_5m_frame(frame, normalized, source, adjust)
        if frame.empty:
            raise RuntimeError(f"{normalized} normalized baostock_5m frame is empty")
        validate_normalized_5m_frame(frame)
        return frame, source

    def _fetch_limit_up_pool_akshare(self, date_text: str) -> pd.DataFrame:
        ak = load_akshare()
        raw = ak.stock_zt_pool_em(date=date_text.replace("-", ""))
        if raw is None or raw.empty:
            raise RuntimeError("AkShare limit-up pool is empty")
        return raw

    def _fetch_limit_up_pool_from_spot(self) -> pd.DataFrame:
        spot = self._fetch_spot_eastmoney()
        if spot.empty:
            raise RuntimeError("spot data is empty")
        latest = pd.to_numeric(spot["latest_price"], errors="coerce")
        high = pd.to_numeric(spot["high"], errors="coerce")
        pct = pd.to_numeric(spot["pct_chg"], errors="coerce")
        approx = spot[(pct >= 9.8) & (latest > 0) & ((latest - high).abs() <= 0.001)]
        return approx.reset_index(drop=True)

    def _fetch_spot_eastmoney(self) -> pd.DataFrame:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        rows: list[dict] = []
        total = None
        page = 1
        while True:
            params = {
                "pn": str(page),
                "pz": "100",
                "po": "1",
                "np": "1",
                "fid": "f12",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
                "fields": "f2,f3,f4,f5,f6,f7,f8,f12,f14,f15,f16,f17,f18,f20,f21,f100",
            }
            payload = self.client.get_json(url, params=params)
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                raise RuntimeError("Eastmoney spot response missing data")
            page_rows = data.get("diff") or []
            if total is None:
                total = int(data.get("total") or 0)
            if not page_rows:
                break
            rows.extend(page_rows)
            if total and len(rows) >= total:
                break
            page += 1
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return pd.DataFrame(
            {
                "code": frame["f12"],
                "name": frame["f14"],
                "latest_price": frame["f2"],
                "pct_chg": frame["f3"],
                "change": frame["f4"],
                "volume": frame["f5"],
                "amount": frame["f6"],
                "amplitude": frame["f7"],
                "turnover_rate": frame["f8"],
                "high": frame["f15"],
                "low": frame["f16"],
                "open": frame["f17"],
                "prev_close": frame["f18"],
                "total_market_cap": frame["f20"],
                "float_market_cap": frame["f21"],
                "industry": frame["f100"],
            }
        )

    def _fetch_daily_tencent(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        market_symbol = to_market_symbol(code)
        adjust_map = {"none": "", "": "", "qfq": "qfq", "hfq": "hfq"}
        adjust_suffix = adjust_map.get(adjust, "")
        param = f"{market_symbol},day,,,500,{adjust_suffix}"
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        payload = self.client.get_json(url, params={"param": param})
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise RuntimeError("Tencent daily response missing data")
        stock = data.get(market_symbol)
        if not isinstance(stock, dict):
            raise RuntimeError("Tencent daily response missing stock data")
        key_map = {"qfq": "qfqday", "hfq": "hfqday"}
        key = key_map.get(adjust, "day")
        rows = stock.get(key) or stock.get("day") or []
        if not rows:
            raise RuntimeError("Tencent daily response contains no klines")
        rows = [row[:6] for row in rows]
        frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
        if frame.empty:
            raise RuntimeError("Tencent daily frame is empty after date filter")
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return _attach_derived_fields(frame)

    def _fetch_daily_sina(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        if adjust not in ("", "none"):
            raise RuntimeError("Sina daily endpoint does not provide adjusted data")
        market_symbol = to_market_symbol(code)
        url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/data/CN_MarketDataService.getKLineData"
        params = {"symbol": market_symbol, "scale": "240", "ma": "no", "datalen": "500"}
        headers = {
            "User-Agent": self.client.headers["User-Agent"],
            "Referer": "https://finance.sina.com.cn/",
        }
        text = self.client.get_text(url, params=params, headers=headers)
        payload = extract_sina_json_payload(text)
        raw = pd.DataFrame(json.loads(payload))
        if raw.empty:
            raise RuntimeError("Sina daily response is empty")
        frame = pd.DataFrame(
            {
                "date": raw["day"],
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
            }
        )
        frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
        if frame.empty:
            raise RuntimeError("Sina daily frame is empty after date filter")
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return _attach_derived_fields(frame)

    def _fetch_5min_sina(
        self,
        code: str,
        start_datetime: str,
        end_datetime: str,
        adjust: str,
    ) -> pd.DataFrame:
        if adjust not in ("", "none"):
            raise RuntimeError("Sina direct 5m endpoint does not provide adjusted data")
        url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
        params = {
            "symbol": to_market_symbol(code),
            "scale": "5",
            "ma": "no",
            "datalen": "1970",
        }
        headers = {
            "User-Agent": self.client.headers["User-Agent"],
            "Referer": "https://vip.stock.finance.sina.com.cn/mkt/",
        }
        text = self.client.get_text(url, params=params, headers=headers)
        payload = extract_sina_json_payload(text)
        rows = json.loads(payload)
        raw = pd.DataFrame(rows)
        if raw.empty:
            raise RuntimeError("Sina 5m response is empty")
        frame = pd.DataFrame(
            {
                "datetime": raw["day"],
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
                "amount": raw["amount"] if "amount" in raw.columns else None,
            }
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        return frame[(frame["datetime"] >= pd.Timestamp(start_datetime)) & (frame["datetime"] <= pd.Timestamp(end_datetime))]

    def _fetch_5min_baostock(
        self,
        code: str,
        start_datetime: str,
        end_datetime: str,
        adjust: str,
    ) -> pd.DataFrame:
        """Fetch raw 5m bars from BaoStock (frequency=5, adjustflag=3 -> none).

        Batch login lifecycle: login once per process (idempotent), queries
        reuse the session; explicit logout_baostock() ends it. NEVER login per
        bar / per event. Raises RuntimeError on any failure (no silent fallback).
        """
        if adjust not in ("", "none"):
            raise RuntimeError(
                f"BaoStock 5m only supports adjust='none' (adjustflag=3); got {adjust!r}")
        login_baostock()
        bs = load_baostock()
        bs_code = to_baostock_code(code)
        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields="date,time,code,open,high,low,close,volume,amount",
            start_date=str(start_datetime)[:10],
            end_date=str(end_datetime)[:10],
            frequency=BAOSTOCK_5M_FREQUENCY,
            adjustflag=BAOSTOCK_5M_ADJUSTFLAG_NO_ADJUST,
        )
        error_code = str(getattr(rs, "error_code", ""))
        if error_code not in ("0", ""):
            raise RuntimeError(
                f"baostock query failed for {bs_code}: "
                f"error_code={error_code} error_msg={getattr(rs, 'error_msg', '')}")
        rows: list[list[str]] = []
        while rs.next():
            rows.append(list(rs.get_row_data()))
        if not rows:
            raise RuntimeError(f"baostock 5m response is empty for {bs_code}")
        frame = pd.DataFrame(rows, columns=list(rs.fields))
        # baostock 对停牌日返回 OHLC 全为 0 的占位 bar (Sina 则省略该日)。
        # 这些占位行不是真实成交, 与 D1 的"缺日=停牌"语义冲突, 丢弃之。
        # 仅当 open/high/low/close 四项全为 0 时判定为占位; 不做任何价格调整。
        if {"open", "high", "low", "close"}.issubset(frame.columns):
            zero_ohlc = (
                pd.to_numeric(frame["open"], errors="coerce").fillna(-1).eq(0)
                & pd.to_numeric(frame["high"], errors="coerce").fillna(-1).eq(0)
                & pd.to_numeric(frame["low"], errors="coerce").fillna(-1).eq(0)
                & pd.to_numeric(frame["close"], errors="coerce").fillna(-1).eq(0)
            )
            frame = frame.loc[~zero_ohlc].reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"baostock 5m response is empty for {bs_code}")
        return frame

def normalize_limit_up_pool(frame: pd.DataFrame, trade_date: str, source: str) -> pd.DataFrame:
    result = frame.copy()
    rename = {
        "代码": "code",
        "名称": "name",
        "最新价": "latest_price",
        "涨跌幅": "pct_chg",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "流通市值": "float_market_cap",
        "总市值": "total_market_cap",
        "封板资金": "seal_amount",
        "首次封板时间": "limit_up_time",
        "最后封板时间": "final_limit_up_time",
        "炸板次数": "open_board_count",
        "连板数": "consecutive_limit_up_count",
        "所属行业": "industry",
    }
    result = result.rename(columns=rename)
    if "code" not in result.columns:
        raise RuntimeError("limit-up pool missing code")
    if "name" not in result.columns:
        result["name"] = ""
    result["code"] = result["code"].map(normalize_stock_code)
    result["trade_date"] = trade_date
    result["market"] = result["code"].map(detect_market)
    result["source"] = source
    for column in (
        "latest_price",
        "pct_chg",
        "amount",
        "turnover_rate",
        "float_market_cap",
        "total_market_cap",
        "seal_amount",
        "open_board_count",
        "consecutive_limit_up_count",
    ):
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("industry", "limit_up_time", "final_limit_up_time"):
        if column not in result.columns:
            result[column] = ""
    ordered = [
        "trade_date",
        "code",
        "name",
        "market",
        "latest_price",
        "pct_chg",
        "amount",
        "turnover_rate",
        "float_market_cap",
        "total_market_cap",
        "industry",
        "limit_up_time",
        "final_limit_up_time",
        "open_board_count",
        "seal_amount",
        "consecutive_limit_up_count",
        "source",
    ]
    for column in ordered:
        if column not in result.columns:
            result[column] = None
    return result[ordered].drop_duplicates(["trade_date", "code"]).reset_index(drop=True)


def normalize_stock_universe(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    if "code" not in result.columns:
        raise RuntimeError("stock universe missing code")
    if "name" not in result.columns:
        result["name"] = ""
    result["code"] = result["code"].map(normalize_stock_code)
    result["market"] = result["code"].map(detect_market)
    if "industry" not in result.columns:
        result["industry"] = ""
    for column in ("float_market_cap", "total_market_cap"):
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = source
    ordered = ["code", "name", "market", "industry", "float_market_cap", "total_market_cap", "source"]
    return result[ordered].drop_duplicates("code").reset_index(drop=True)


def filter_main_board(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = result[result["code"].map(is_main_board_code)]
    result = result[~result["name"].map(is_excluded_name)]
    return result.reset_index(drop=True)


def normalize_daily_frame(frame: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["code"] = normalize_stock_code(code)
    result["market"] = detect_market(code)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "change", "turnover_rate"):
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = source
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    result = result.drop_duplicates(["code", "date"]).sort_values("date").reset_index(drop=True)
    return result[
        [
            "date",
            "code",
            "market",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pct_chg",
            "change",
            "amplitude",
            "turnover_rate",
            "source",
        ]
    ]


def normalize_5min_frame(frame: pd.DataFrame, code: str, source: str, adjust: str) -> pd.DataFrame:
    result = frame.copy()
    result["code"] = normalize_stock_code(code)
    result["market"] = detect_market(code)
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result["trade_date"] = result["datetime"].dt.strftime("%Y-%m-%d")
    result["time"] = result["datetime"].dt.strftime("%H:%M:%S")
    for column in ("open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "change", "turnover_rate"):
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = source
    result["adjust"] = adjust or "none"
    result["interval"] = NORMALIZED_5M_INTERVAL
    result = result.dropna(subset=["datetime", "open", "high", "low", "close"])
    result = result.drop_duplicates(["code", "datetime"]).sort_values("datetime").reset_index(drop=True)
    return result[
        [
            "datetime",
            "trade_date",
            "time",
            "code",
            "market",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pct_chg",
            "change",
            "amplitude",
            "turnover_rate",
            "source",
            "adjust",
            "interval",
        ]
    ]


NORMALIZED_5M_COLUMNS = (
    "datetime",
    "trade_date",
    "time",
    "code",
    "market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "change",
    "amplitude",
    "turnover_rate",
    "source",
    "adjust",
    "interval",
)


def to_baostock_code(code: str) -> str:
    """6 位代码确定性转换到 baostock sh./sz. 前缀。

    规则 (v004c cross-source 标准): 6/9 开头 -> sh., 其余 -> sz.。
    4/8/9 开头的北交所代码按既有约定映射 sz. (仅历史研究窗口使用)。
    """
    normalized = normalize_stock_code(code)
    prefix = "sh." if normalized.startswith(("6", "9")) else "sz."
    return prefix + normalized


def parse_baostock_time17(value: object) -> str:
    """17 位 time 字符串 "YYYYMMDDHHMMSSmmm" -> "HH:MM:SS"。"""
    text = str(value).strip()
    if len(text) < 14:
        raise ValueError(f"invalid baostock time string {text!r} (expected >= 14 chars)")
    return f"{text[8:10]}:{text[10:12]}:{text[12:14]}"


def normalize_baostock_5m_frame(
    frame: pd.DataFrame,
    code: str,
    source: str,
    adjust: str,
) -> pd.DataFrame:
    """baostock 原始 5m 行 (date/time/code/open/high/low/close/volume/amount)
    -> 统一 normalized 5m contract (与 normalize_5min_frame 同 schema)。

    原始值均为字符串; volume=股, amount=元 (baostock 单位); 不做任何价格校准。
    """
    result = frame.copy()
    result["datetime"] = pd.to_datetime(
        result["date"].astype(str) + " " + result["time"].map(parse_baostock_time17),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    result["code"] = normalize_stock_code(code)
    result["market"] = detect_market(code)
    result["trade_date"] = result["datetime"].dt.strftime("%Y-%m-%d")
    result["time"] = result["datetime"].dt.strftime("%H:%M:%S")
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "amplitude",
        "turnover_rate",
    ):
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = source
    result["adjust"] = adjust or "none"
    result["interval"] = NORMALIZED_5M_INTERVAL
    result = result.dropna(subset=["datetime", "open", "high", "low", "close"])
    result = result.drop_duplicates(["code", "datetime"]).sort_values("datetime").reset_index(drop=True)
    return result[list(NORMALIZED_5M_COLUMNS)]


def validate_normalized_5m_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """§8 严格验证统一 normalized 5m contract (用于 baostock 路径)。

    datetime 可解析 / 严格升序 / 无重复 / OHLC finite>0 / high>=max(open,close)
    / low<=min(open,close) / volume>=0 / amount>=0 / source / adjust / interval。
    违反任一规则 -> RuntimeError (fail closed)。
    """
    if frame is None or frame.empty:
        raise RuntimeError("normalized 5m frame is empty")
    missing = [c for c in ("datetime", "open", "high", "low", "close", "volume", "amount",
                           "source", "adjust", "interval") if c not in frame.columns]
    if missing:
        raise RuntimeError(f"normalized 5m frame missing columns: {missing}")
    dt = pd.to_datetime(frame["datetime"], errors="coerce")
    if dt.isna().any():
        raise RuntimeError(f"normalized 5m frame has unparseable datetime ({int(dt.isna().sum())} rows)")
    if not dt.is_monotonic_increasing or dt.duplicated().any():
        raise RuntimeError("normalized 5m frame datetime must be strictly ascending and unique")
    numeric = frame[["open", "high", "low", "close", "volume", "amount"]].apply(
        pd.to_numeric, errors="coerce")
    for column in ("open", "high", "low", "close"):
        bad = numeric[column].isna() | (numeric[column] <= 0) | ~np.isfinite(numeric[column])
        if bad.any():
            raise RuntimeError(f"normalized 5m frame {column} must be finite and > 0 ({int(bad.sum())} rows)")
    high_bad = (numeric["high"] < numeric["open"]) | (numeric["high"] < numeric["close"])
    if high_bad.any():
        raise RuntimeError(f"normalized 5m frame high < open/close ({int(high_bad.sum())} rows)")
    low_bad = (numeric["low"] > numeric["open"]) | (numeric["low"] > numeric["close"])
    if low_bad.any():
        raise RuntimeError(f"normalized 5m frame low > open/close ({int(low_bad.sum())} rows)")
    for column in ("volume", "amount"):
        bad = numeric[column].isna() | (numeric[column] < 0) | ~np.isfinite(numeric[column])
        if bad.any():
            raise RuntimeError(f"normalized 5m frame {column} must be >= 0 ({int(bad.sum())} rows)")
    if not (frame["source"] == BAOSTOCK_5M_SOURCE).all():
        raise RuntimeError("normalized 5m frame source must be baostock_5m")
    if not (frame["adjust"] == "none").all():
        raise RuntimeError("normalized 5m frame adjustment must be none")
    if not (frame["interval"] == NORMALIZED_5M_INTERVAL).all():
        raise RuntimeError(f"normalized 5m frame interval must be {NORMALIZED_5M_INTERVAL}")
    return frame


_BAOSTOCK_MODULE = None
_BAOSTOCK_LOGGED_IN = False


def load_baostock():
    """惰性加载 baostock (与 load_akshare 同惯用法; 依赖声明于 pyproject/requirements)。"""
    global _BAOSTOCK_MODULE
    if _BAOSTOCK_MODULE is None:
        try:
            _BAOSTOCK_MODULE = import_module("baostock")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "missing dependency baostock; run python -m pip install -r requirements.txt") from exc
    return _BAOSTOCK_MODULE


def login_baostock() -> None:
    """批量生命周期: 进程内只 login 一次 (幂等), 由 logout_baostock() 显式结束。"""
    global _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK_LOGGED_IN:
        return
    bs = load_baostock()
    login_result = bs.login()
    error_code = str(getattr(login_result, "error_code", ""))
    if error_code not in ("0", ""):
        _BAOSTOCK_LOGGED_IN = False
        raise RuntimeError(
            f"baostock login failed: error_code={error_code} "
            f"error_msg={getattr(login_result, 'error_msg', '')}")
    _BAOSTOCK_LOGGED_IN = True


def logout_baostock() -> None:
    global _BAOSTOCK_LOGGED_IN
    if not _BAOSTOCK_LOGGED_IN:
        return
    try:
        load_baostock().logout()
    finally:
        _BAOSTOCK_LOGGED_IN = False


# ---------------------------------------------------------------------------
# 可选 SOCKS5 代理 (显式 opt-in; 仅拦截 baostock 服务器连接, 不影响其他流量)
# ---------------------------------------------------------------------------
_BAOSTOCK_SOCKS5_PROXY: tuple[str, int] | None = None
_SOCKET_CONNECT_PATCHED = False


def _socks5_connect(
    sock: socket.socket,
    host: str,
    port: int,
    proxy_host: str,
    proxy_port: int,
    timeout: float = 15.0,
) -> None:
    """SOCKS5 无认证 CONNECT 握手 (ATYP=3 域名); 失败 raise RuntimeError。"""
    orig_timeout = sock.gettimeout()
    sock.settimeout(timeout)
    try:
        sock.connect((proxy_host, proxy_port))
        sock.sendall(b"\x05\x01\x00")
        reply = sock.recv(2)
        if reply != b"\x05\x00":
            raise RuntimeError(f"socks5 auth failed: {reply!r}")
        host_bytes = host.encode("utf-8")
        if len(host_bytes) > 255:
            raise RuntimeError(f"socks5 target host too long: {host}")
        sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) +
                     host_bytes + port.to_bytes(2, "big"))
        head = sock.recv(4)
        if len(head) < 2 or head[0] != 0x05:
            raise RuntimeError(f"socks5 connect failed (protocol): {head!r}")
        if head[1] != 0x00:
            raise RuntimeError(f"socks5 connect rejected: code={head[1]}")
        # 消费剩余 bind 地址 (ATYP=1: +6, ATYP=3: +len+2, ATYP=4: +18)
        atyp = head[3]
        if atyp == 0x01:
            remain = 6
        elif atyp == 0x03:
            length = sock.recv(1)
            remain = 1 + int(length[0]) + 2
        elif atyp == 0x04:
            remain = 18
        else:
            remain = 0
        while remain > 0:
            chunk = sock.recv(remain)
            if not chunk:
                break
            remain -= len(chunk)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"socks5 connect to {host}:{port} via {proxy_host}:{proxy_port} failed: "
            f"{exc}") from exc
    finally:
        sock.settimeout(orig_timeout)


def set_baostock_socks5_proxy(host: str, port: int) -> None:
    """显式启用 SOCKS5 代理: 包装 socket.socket.connect, 仅拦截 baostock
    数据服务目标 (BAOSTOCK_SERVER_HOST:BAOSTOCK_SERVER_PORT), 其余连接保持原逻辑。"""
    global _BAOSTOCK_SOCKS5_PROXY, _SOCKET_CONNECT_PATCHED
    _BAOSTOCK_SOCKS5_PROXY = (host, int(port))
    if _SOCKET_CONNECT_PATCHED:
        return
    _orig_connect = socket.socket.connect

    def _patched_connect(self_sock: socket.socket, address: tuple) -> None:
        if _BAOSTOCK_SOCKS5_PROXY is not None and len(address) >= 2:
            host, port = str(address[0]), int(address[1])
            if host == BAOSTOCK_SERVER_HOST and port == BAOSTOCK_SERVER_PORT:
                _socks5_connect(self_sock, host, port, *_BAOSTOCK_SOCKS5_PROXY)
                return
        _orig_connect(self_sock, address)

    socket.socket.connect = _patched_connect  # type: ignore[method-assign]
    _SOCKET_CONNECT_PATCHED = True


def clear_baostock_socks5_proxy() -> None:
    """恢复原始 socket.socket.connect (幂等)。"""
    global _BAOSTOCK_SOCKS5_PROXY, _SOCKET_CONNECT_PATCHED
    _BAOSTOCK_SOCKS5_PROXY = None
    if _SOCKET_CONNECT_PATCHED:
        socket.socket.connect = _orig_socket_connect()  # type: ignore[method-assign]
        _SOCKET_CONNECT_PATCHED = False


def _orig_socket_connect():
    """恢复用的原始 connect (首次 patch 时保存, 存入本函数默认闭包)。"""
    return _PATCH_ORIG_CONNECT


_PATCH_ORIG_CONNECT = socket.socket.connect


def normalize_date_text(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def normalize_listing_date(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if len(text) != 8 or not text.isdigit():
        return ""
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def normalize_suspension_status_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the documented fields returned by ``stock_tfp_em``."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(SUSPENSION_STATUS_COLUMNS))
    aliases = {
        "code": ("代码", "SECURITY_CODE"),
        "name": ("名称", "SECURITY_NAME_ABBR"),
        "suspension_start_date": ("停牌时间", "SUSPEND_START_DATE"),
        "suspension_end_date": ("停牌截止时间", "SUSPEND_END_TIME"),
        "suspension_duration": ("停牌期限", "SUSPEND_EXPIRE"),
        "suspension_reason": ("停牌原因", "SUSPEND_REASON"),
        "suspension_market": ("所属市场", "TRADE_MARKET"),
        "expected_resume_date": ("预计复牌时间", "预计复牌日期", "PREDICT_RESUME_DATE"),
    }

    def source_column(names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)

    result = pd.DataFrame(index=frame.index)
    raw_codes = source_column(aliases["code"])
    normalized_codes: list[str] = []
    for value in raw_codes.tolist():
        try:
            normalized_codes.append(normalize_stock_code(value))
        except (TypeError, ValueError):
            normalized_codes.append("")
    result["code"] = normalized_codes
    result["name"] = source_column(aliases["name"]).map(_clean_text)
    result["suspension_start_date"] = source_column(
        aliases["suspension_start_date"]
    ).map(_normalize_iso_date_value)
    result["suspension_end_date"] = source_column(
        aliases["suspension_end_date"]
    ).map(_normalize_iso_date_value)
    result["suspension_duration"] = source_column(
        aliases["suspension_duration"]
    ).map(_clean_text)
    result["suspension_reason"] = source_column(aliases["suspension_reason"]).map(
        _clean_text
    )
    result["suspension_market"] = source_column(aliases["suspension_market"]).map(
        _clean_text
    )
    result["expected_resume_date"] = source_column(
        aliases["expected_resume_date"]
    ).map(_normalize_iso_date_value)
    result = result[result["code"].ne("")]
    return result[list(SUSPENSION_STATUS_COLUMNS)].sort_values(
        ["code", "suspension_start_date", "suspension_end_date"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_iso_date_value(value: object) -> str:
    if value is None or isinstance(value, bool) or pd.isna(value):
        return ""
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _clean_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, dict, tuple, set)) and pd.isna(value)):
        return ""
    return str(value).strip()


def extract_sina_json_payload(text: str) -> str:
    for marker in ("=(", "data("):
        start = text.find(marker)
        if start >= 0:
            payload = text[start + len(marker):].strip()
            if payload.endswith(");"):
                payload = payload[:-2]
            elif payload.endswith(")"):
                payload = payload[:-1]
            return payload.strip()
    raise RuntimeError("Sina response format is not JSONP")


def load_akshare():
    try:
        return import_module("akshare")
    except ModuleNotFoundError as exc:
        raise RuntimeError("missing dependency akshare; run python -m pip install -r requirements.txt") from exc


def _attach_derived_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("date").reset_index(drop=True)
    prev_close = result["close"].shift(1)
    result["change"] = result["close"] - prev_close
    result["pct_chg"] = (result["close"] - prev_close) / prev_close * 100
    result["amplitude"] = (result["high"] - result["low"]) / prev_close * 100
    result["amount"] = None
    result["turnover_rate"] = None
    return result

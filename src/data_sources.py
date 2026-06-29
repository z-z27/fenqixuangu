from __future__ import annotations

from datetime import datetime
from importlib import import_module
import json
from typing import Callable

import pandas as pd

from .code_utils import (
    detect_market,
    is_excluded_name,
    is_main_board_code,
    normalize_stock_code,
    to_market_symbol,
)
from .config import DataConfig
from .http_client import RequestClient


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
    ) -> tuple[pd.DataFrame, str]:
        normalized = normalize_stock_code(code)
        source_attempts = [
            ("sina_5m", lambda: self._fetch_5min_sina(normalized, start_datetime, end_datetime, adjust)),
        ]
        errors: list[str] = []
        for source, fetcher in source_attempts:
            try:
                frame = fetcher()
                if frame is None or frame.empty:
                    raise RuntimeError("empty 5m frame")
                frame = normalize_5min_frame(frame, normalized, source, adjust)
                if frame.empty:
                    raise RuntimeError("normalized 5m frame is empty")
                return frame, source
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        raise RuntimeError(f"{normalized} all 5m sources failed: " + " | ".join(errors))

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
        ]
    ]


def normalize_date_text(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


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

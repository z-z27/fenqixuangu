from __future__ import annotations

"""Direct auction source capability audit.

This module intentionally does not load labels, returns, model scores, or model
ranks.  It separates four questions that were conflated by the preceding cache
audit: raw source capability, repository wrapper exposure, historical depth,
and live-window validation.

The network observations embedded below were collected on 2026-09-06 from the
three source families explicitly allowed by the protocol.  They are small,
auditable response extracts rather than a market-data archive.  Re-running the
audit is deterministic and does not silently replace the recorded evidence.
"""

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd


AUDIT_ASOF = "2026-09-06"
LATEST_TRADING_DATE_OBSERVED = "2026-09-04"
OUTPUT_DIRNAME = "v004d_direct_auction_source_capability_audit_v001"
OUTPUT_FILENAMES = (
    "v004d_direct_auction_source_inventory_v001.csv",
    "v004d_direct_auction_raw_endpoint_inventory_v001.csv",
    "v004d_direct_auction_field_capability_v001.csv",
    "v004d_direct_auction_historical_depth_v001.csv",
    "v004d_direct_auction_price_lineage_v001.csv",
    "v004d_direct_auction_volume_amount_semantics_v001.csv",
    "v004d_direct_auction_preopen_path_semantics_v001.csv",
    "v004d_direct_auction_live_capability_v001.csv",
    "v004d_direct_auction_batch_stress_v001.csv",
    "v004d_direct_auction_source_shift_v001.csv",
    "v004d_direct_auction_capability_matrix_v001.csv",
    "v004d_direct_auction_source_capability_review_v001.md",
)

FIELD_IDS = {
    "C1": "FINAL_AUCTION_PRICE",
    "C2": "AUCTION_VOLUME",
    "C3": "AUCTION_AMOUNT",
    "C4": "BID1_PRICE",
    "C5": "BID1_VOLUME",
    "C6": "ASK1_PRICE",
    "C7": "ASK1_VOLUME",
    "C8": "BID2_TO_BID5",
    "C9": "ASK2_TO_ASK5",
    "C10": "UNMATCHED_BUY_VOLUME",
    "C11": "UNMATCHED_SELL_VOLUME",
    "C12": "VIRTUAL_MATCH_PRICE",
    "C13": "VIRTUAL_MATCH_VOLUME",
    "C14": "AUCTION_TIMESTAMP",
}

AKSHARE_DOC_URL = (
    "https://github.com/akfamily/akshare/blob/main/docs/data/stock/stock.md"
)
AKSHARE_SOURCE_URL = (
    "https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_hist_em.py"
)
AKSHARE_BIDASK_SOURCE_URL = (
    "https://github.com/akfamily/akshare/blob/main/akshare/stock/stock_ask_bid_em.py"
)
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_KLINE_URL = (
    "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/"
    "CN_MarketDataService.getKLineData"
)
EM_PRE_URL = "https://push2.eastmoney.com/api/qt/stock/trends2/get"
EM_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
EM_BATCH_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
EM_HIST_MIN_URL = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"


@dataclass(frozen=True)
class EastmoneyTrendRow:
    timestamp: str
    open: float
    close: float
    high: float
    low: float
    volume_hands: float
    amount_cny: float
    latest_or_average: float


def parse_eastmoney_trend_row(row: str) -> EastmoneyTrendRow:
    parts = row.split(",")
    if len(parts) != 8:
        raise ValueError(f"expected 8 trend fields, got {len(parts)}")
    return EastmoneyTrendRow(
        timestamp=parts[0],
        open=float(parts[1]),
        close=float(parts[2]),
        high=float(parts[3]),
        low=float(parts[4]),
        volume_hands=float(parts[5]),
        amount_cny=float(parts[6]),
        latest_or_average=float(parts[7]),
    )


def parse_sina_quote_lines(text: str) -> pd.DataFrame:
    """Parse the stable public Sina A-share quote layout used by the repo audit."""
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r'var hq_str_(?P<symbol>\w+)="(?P<body>[^"]*)";')
    for match in pattern.finditer(text):
        values = match.group("body").split(",")
        if len(values) < 32:
            continue
        rows.append(
            {
                "symbol": match.group("symbol"),
                "name": values[0],
                "open": pd.to_numeric(values[1], errors="coerce"),
                "prev_close": pd.to_numeric(values[2], errors="coerce"),
                "latest": pd.to_numeric(values[3], errors="coerce"),
                "volume_shares": pd.to_numeric(values[8], errors="coerce"),
                "amount_cny": pd.to_numeric(values[9], errors="coerce"),
                "bid1_volume_shares": pd.to_numeric(values[10], errors="coerce"),
                "bid1_price": pd.to_numeric(values[11], errors="coerce"),
                "ask1_volume_shares": pd.to_numeric(values[20], errors="coerce"),
                "ask1_price": pd.to_numeric(values[21], errors="coerce"),
                "quote_date": values[30],
                "quote_time": values[31],
            }
        )
    return pd.DataFrame(rows)


# Direct response extracts observed on the audit date.  They contain no future
# label or outcome information.
EM_PRE_000001_ROWS = (
    "2026-09-04 09:24,11.88,11.87,11.88,11.87,0,0.00,11.880",
    "2026-09-04 09:25,11.87,11.86,11.87,11.86,0,0.00,11.880",
    "2026-09-04 09:26,11.86,11.86,11.86,11.86,5094,6041484.00,11.860",
)
EM_QUOTE_000001 = {
    "f43_latest": 11.89,
    "f46_open": 11.86,
    "f47_full_day_volume_hands": 814373,
    "f48_full_day_amount_cny": 969948436.21,
    "f57_code": "000001",
    "f60_prev_close": 11.88,
    "f86_timestamp": 1788507240,
}
SINA_QUOTE_SAMPLE = (
    'var hq_str_sz000001="PINGAN,11.860,11.880,11.890,12.000,11.850,'
    '11.890,11.900,81437295,969948436.210,90800,11.890,103800,11.880,'
    '272000,11.870,503100,11.860,424000,11.850,108800,11.900,186600,'
    '11.910,327700,11.920,244900,11.930,431300,11.940,2026-09-04,'
    '15:36:00,00,D|44700|531483.000";\n'
    'var hq_str_sh600000="SPDB,9.270,9.270,9.430,9.450,9.260,9.420,'
    '9.430,75765982,712270967.000,277300,9.420,363850,9.410,293800,'
    '9.400,257900,9.390,276700,9.380,819800,9.430,782600,9.440,'
    '2284254,9.450,1395830,9.460,898800,9.470,2026-09-04,15:34:59,'
    '00,D|31300|295159.00";'
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8-sig")


def build_source_inventory() -> pd.DataFrame:
    rows = [
        {
            "source_family": "SINA",
            "wrapper_source": "CURRENT_REPO",
            "upstream_source": "SINA",
            "interface": "MarketDataProvider._fetch_5min_sina",
            "raw_endpoint": SINA_KLINE_URL,
            "historical_support": "ROLLING_REGULAR_5M_ONLY_NO_AUCTION",
            "realtime_support": "NO_REPO_REALTIME_QUOTE_WRAPPER",
            "batch_support": "NO",
            "auth_required": "NO",
            "available_fields": "regular 5m OHLC volume amount",
            "current_status": "NOT_AN_AUCTION_INTERFACE",
            "evidence": "src/data_sources.py; normalized first regular bar 09:35",
        },
        {
            "source_family": "SINA",
            "wrapper_source": "RAW_DIRECT",
            "upstream_source": "SINA",
            "interface": "hq.sinajs.cn realtime quote",
            "raw_endpoint": SINA_QUOTE_URL,
            "historical_support": "REALTIME_ONLY",
            "realtime_support": "YES_CURRENT_QUOTE; 09:25 SEMANTICS PENDING",
            "batch_support": "YES_COMMA_SEPARATED_SYMBOLS",
            "auth_required": "NO; REFERER REQUIRED",
            "available_fields": "open current cumulative volume/amount L1-L5 date time",
            "current_status": "AUCTION_CAPABILITY_UNRESOLVED",
            "evidence": "direct one- and two-symbol response on 2026-09-06",
        },
        {
            "source_family": "STOCK_EASTMONEY",
            "wrapper_source": "CURRENT_REPO",
            "upstream_source": "EASTMONEY",
            "interface": "MarketDataProvider._fetch_spot_eastmoney",
            "raw_endpoint": "https://push2.eastmoney.com/api/qt/clist/get",
            "historical_support": "NO",
            "realtime_support": "YES_CURRENT_MARKET_SNAPSHOT",
            "batch_support": "YES_PAGED_MARKET_LIST",
            "auth_required": "NO",
            "available_fields": "open latest full-day volume/amount; no preopen rows",
            "current_status": "WRAPPER_DROPS_QUOTE_FIELDS_AFTER_UNIVERSE_NORMALIZATION",
            "evidence": "src/data_sources.py:_fetch_spot_eastmoney/normalize_stock_universe",
        },
        {
            "source_family": "STOCK_EASTMONEY",
            "wrapper_source": "RAW_DIRECT",
            "upstream_source": "EASTMONEY",
            "interface": "stock/trends2/get iscr=1 ndays=1",
            "raw_endpoint": EM_PRE_URL,
            "historical_support": "LATEST_TRADING_DAY_ONLY",
            "realtime_support": "EXPECTED_CURRENT_SESSION; LIVE WINDOW NOT OBSERVED",
            "batch_support": "NO_SINGLE_SECID",
            "auth_required": "NO",
            "available_fields": "09:15-09:30 supplier premarket states; price volume amount timestamp",
            "current_status": "AUCTION_SOURCE_PARTIAL",
            "evidence": "direct 000001 response: 09:25 price state and 09:26 opening-match-like row",
        },
        {
            "source_family": "STOCK_EASTMONEY",
            "wrapper_source": "RAW_DIRECT",
            "upstream_source": "EASTMONEY",
            "interface": "stock/get and ulist.np/get",
            "raw_endpoint": f"{EM_QUOTE_URL}; {EM_BATCH_URL}",
            "historical_support": "NO",
            "realtime_support": "YES_CURRENT_QUOTE; 09:25 SEMANTICS PENDING",
            "batch_support": "YES_ULIST_NP_GET",
            "auth_required": "NO",
            "available_fields": "open current cumulative volume/amount timestamp; raw quote family supports order book fields",
            "current_status": "AUCTION_SOURCE_PARTIAL",
            "evidence": "direct single and N=20/30/50 batch responses on 2026-09-06",
        },
        {
            "source_family": "AKSHARE",
            "wrapper_source": "AKSHARE_1.18.83",
            "upstream_source": "EASTMONEY",
            "interface": "stock_zh_a_hist_pre_min_em",
            "raw_endpoint": EM_PRE_URL,
            "historical_support": "LATEST_TRADING_DAY_ONLY_NO_DATE_ARGUMENT",
            "realtime_support": "EXPECTED_CURRENT_SESSION; LIVE WINDOW NOT OBSERVED",
            "batch_support": "NO_SINGLE_SYMBOL",
            "auth_required": "NO",
            "available_fields": "time OHLC volume(hands) amount latest",
            "current_status": "AUCTION_SOURCE_PARTIAL",
            "evidence": f"installed source and official docs: {AKSHARE_DOC_URL}",
        },
        {
            "source_family": "AKSHARE",
            "wrapper_source": "AKSHARE_1.18.83",
            "upstream_source": "EASTMONEY",
            "interface": "stock_zh_a_hist_min_em(period=1)",
            "raw_endpoint": EM_HIST_MIN_URL,
            "historical_support": "LIMITED_TO_RECENT_5_TRADING_DAYS_BY_SOURCE_CODE",
            "realtime_support": "NO_LIVE_TIMING_CONTRACT",
            "batch_support": "NO_SINGLE_SYMBOL",
            "auth_required": "NO",
            "available_fields": "1m OHLC volume amount average; 09:30 row may contain opening match",
            "current_status": "LIMITED_HISTORICAL_RUNTIME_PROBE_FAILED",
            "evidence": f"installed wrapper source; upstream connection failed during audit; {AKSHARE_SOURCE_URL}",
        },
        {
            "source_family": "AKSHARE",
            "wrapper_source": "AKSHARE_1.18.83",
            "upstream_source": "EASTMONEY",
            "interface": "stock_bid_ask_em",
            "raw_endpoint": EM_QUOTE_URL,
            "historical_support": "NO",
            "realtime_support": "CURRENT_SINGLE_SYMBOL_QUOTE",
            "batch_support": "NO_SINGLE_SYMBOL",
            "auth_required": "NO",
            "available_fields": "L1-L5 open latest cumulative volume/amount",
            "current_status": "WRAPPER_RUNTIME_UNSTABLE_ON_AUDIT_DATE",
            "evidence": f"installed source; JSON decode failure observed; {AKSHARE_BIDASK_SOURCE_URL}",
        },
    ]
    return pd.DataFrame(rows)


def build_raw_endpoint_inventory() -> pd.DataFrame:
    rows = [
        {
            "source": "SINA",
            "endpoint": SINA_QUOTE_URL,
            "purpose": "current single/multi-symbol quote",
            "key_parameters": "list=comma-separated market symbols",
            "response_shape": "one var hq_str line per symbol",
            "direct_probe_status": "SUCCESS_SINGLE_AND_BATCH",
            "observed_dates": LATEST_TRADING_DATE_OBSERVED,
            "auction_relevance": "open and cumulative quantity/amount may be usable only at exact post-match time; not auction-isolated",
            "repo_exposure": "NOT_WRAPPED",
        },
        {
            "source": "SINA",
            "endpoint": SINA_KLINE_URL,
            "purpose": "rolling regular daily/minute bars",
            "key_parameters": "symbol scale ma datalen",
            "response_shape": "JSONP OHLCV; repo scale=5",
            "direct_probe_status": "REPO_IMPLEMENTATION_CONFIRMED",
            "observed_dates": "rolling",
            "auction_relevance": "NO; regular 5m starts 09:35",
            "repo_exposure": "WRAPPED_AS_FETCH_5MIN_HISTORY",
        },
        {
            "source": "EASTMONEY",
            "endpoint": EM_PRE_URL,
            "purpose": "latest trading-day minute data including premarket",
            "key_parameters": "fields1 fields2 ndays=1 iscr=1 iscca=0 secid",
            "response_shape": "f51..f58 CSV trend rows",
            "direct_probe_status": "SUCCESS_000001",
            "observed_dates": LATEST_TRADING_DATE_OBSERVED,
            "auction_relevance": "YES_PARTIAL; 09:15-09:25 price states plus 09:26 opening-match-like quantity/amount",
            "repo_exposure": "NOT_WRAPPED",
        },
        {
            "source": "EASTMONEY",
            "endpoint": EM_PRE_URL,
            "purpose": "test whether ndays>1 expands premarket history",
            "key_parameters": "ndays=5 iscr=1",
            "response_shape": "241 rows, one latest trading date; starts at 09:30",
            "direct_probe_status": "SUCCESS_000001_NO_EXTRA_DATES",
            "observed_dates": LATEST_TRADING_DATE_OBSERVED,
            "auction_relevance": "DOES_NOT_ENABLE_MAY_JULY_BACKFILL",
            "repo_exposure": "NOT_WRAPPED",
        },
        {
            "source": "EASTMONEY",
            "endpoint": EM_QUOTE_URL,
            "purpose": "current single-stock quote and optional order book",
            "key_parameters": "secid fields fltt invt",
            "response_shape": "data object",
            "direct_probe_status": "SUCCESS_CORE_QUOTE; LONG_ORDER_BOOK_QUERY_TRANSIENT_FAILURE",
            "observed_dates": LATEST_TRADING_DATE_OBSERVED,
            "auction_relevance": "CURRENT_ONLY; post-close volume/amount are full-day cumulative",
            "repo_exposure": "PARTIAL_VIA_OTHER_WRAPPERS",
        },
        {
            "source": "EASTMONEY",
            "endpoint": EM_BATCH_URL,
            "purpose": "current multi-security quote",
            "key_parameters": "secids fields",
            "response_shape": "data.diff list",
            "direct_probe_status": "SUCCESS_N20_N30_N50; TRANSIENT_FAILURE_N5_N10",
            "observed_dates": LATEST_TRADING_DATE_OBSERVED,
            "auction_relevance": "POTENTIAL_LIVE_BATCH_ONLY; exact 09:25 semantics pending",
            "repo_exposure": "NOT_WRAPPED",
        },
        {
            "source": "EASTMONEY",
            "endpoint": EM_HIST_MIN_URL,
            "purpose": "recent 1m history used by AKShare period=1",
            "key_parameters": "ndays=5 iscr=0 secid",
            "response_shape": "recent minute trend rows",
            "direct_probe_status": "UPSTREAM_CONNECTION_FAILED_DURING_AUDIT",
            "observed_dates": "NOT_OBSERVED_RUNTIME",
            "auction_relevance": "SOURCE_CODE_LIMITED_TO_RECENT_5_DAYS; cannot backfill May-July",
            "repo_exposure": "NOT_WRAPPED",
        },
        {
            "source": "AKSHARE",
            "endpoint": "stock_zh_a_hist_pre_min_em",
            "purpose": "wrapper for Eastmoney latest premarket minute",
            "key_parameters": "symbol start_time end_time; no date",
            "response_shape": "DataFrame with time OHLC volume amount latest",
            "direct_probe_status": "SUCCESS_ONCE_16_PREOPEN_ROWS_THEN_TRANSIENT_FAILURE",
            "observed_dates": LATEST_TRADING_DATE_OBSERVED,
            "auction_relevance": "YES_PARTIAL; wrapper exposes upstream rows but not explicit auction semantics",
            "repo_exposure": "NOT_USED_BY_REPO",
        },
        {
            "source": "AKSHARE",
            "endpoint": "stock_bid_ask_em",
            "purpose": "single-symbol current L1-L5 quote",
            "key_parameters": "symbol",
            "response_shape": "item/value DataFrame",
            "direct_probe_status": "JSON_DECODE_FAILURE_ON_AUDIT_DATE",
            "observed_dates": "NONE_RUNTIME",
            "auction_relevance": "POTENTIAL_LIVE_ORDER_BOOK; no historical support",
            "repo_exposure": "NOT_USED_BY_REPO",
        },
    ]
    return pd.DataFrame(rows)


def _field_row(
    source: str,
    field_id: str,
    raw_status: str,
    repo_status: str,
    semantics: str,
    historical: str,
    live: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "source": source,
        "field_id": field_id,
        "field_name": FIELD_IDS[field_id],
        "raw_source_capability": raw_status,
        "current_repo_wrapper_capability": repo_status,
        "data_semantics": semantics,
        "historical_support": historical,
        "live_support": live,
        "evidence": evidence,
    }


def build_field_capability() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    # Sina current quote layout.
    sina = {
        "C1": ("CONDITIONAL_AVAILABLE_AS_OPEN", "NOT_EXPOSED", "official open field after match; live 09:25 timing unobserved"),
        "C2": ("NOT_AUCTION_ISOLATED", "NOT_EXPOSED", "current cumulative daily volume; auction-only only if sampled before continuous trades, unvalidated"),
        "C3": ("NOT_AUCTION_ISOLATED", "NOT_EXPOSED", "current cumulative daily amount; auction-only only if sampled before continuous trades, unvalidated"),
        "C4": ("AVAILABLE", "NOT_EXPOSED", "best bid price in current quote"),
        "C5": ("AVAILABLE", "NOT_EXPOSED", "best bid volume in shares"),
        "C6": ("AVAILABLE", "NOT_EXPOSED", "best ask price in current quote"),
        "C7": ("AVAILABLE", "NOT_EXPOSED", "best ask volume in shares"),
        "C8": ("AVAILABLE", "NOT_EXPOSED", "bid levels 2-5 in current quote"),
        "C9": ("AVAILABLE", "NOT_EXPOSED", "ask levels 2-5 in current quote"),
        "C10": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no explicit unmatched-buy field"),
        "C11": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no explicit unmatched-sell field"),
        "C12": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no explicit virtual-match field"),
        "C13": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no explicit virtual-match-volume field"),
        "C14": ("AVAILABLE", "NOT_EXPOSED", "response date and time fields"),
    }
    for field_id, (raw, repo, semantics) in sina.items():
        rows.append(_field_row("SINA", field_id, raw, repo, semantics, "NO", "PENDING_TRADING_DAY", "direct hq quote response and repo source scan"))

    em = {
        "C1": ("AVAILABLE_PARTIAL", "DROPPED_AFTER_UNIVERSE_NORMALIZATION", "09:25/09:26 pre-minute price and f46 open; one code-date exact lineage sample"),
        "C2": ("AVAILABLE_PROVISIONAL_AT_0926", "DROPPED_AFTER_UNIVERSE_NORMALIZATION", "09:26 f56 row is consistent with opening match in hands; multi-sample/live proof pending"),
        "C3": ("AVAILABLE_PROVISIONAL_AT_0926", "DROPPED_AFTER_UNIVERSE_NORMALIZATION", "09:26 f57 amount exactly equals price*hands*100 in sample; multi-sample/live proof pending"),
        "C4": ("AVAILABLE_BY_STOCK_GET_CONTRACT", "NOT_REQUESTED", "best bid price; current quote only"),
        "C5": ("AVAILABLE_BY_STOCK_GET_CONTRACT", "NOT_REQUESTED", "best bid volume; current quote only"),
        "C6": ("AVAILABLE_BY_STOCK_GET_CONTRACT", "NOT_REQUESTED", "best ask price; current quote only"),
        "C7": ("AVAILABLE_BY_STOCK_GET_CONTRACT", "NOT_REQUESTED", "best ask volume; current quote only"),
        "C8": ("AVAILABLE_BY_STOCK_GET_CONTRACT", "NOT_REQUESTED", "bid levels 2-5; wrapper source maps fields"),
        "C9": ("AVAILABLE_BY_STOCK_GET_CONTRACT", "NOT_REQUESTED", "ask levels 2-5; wrapper source maps fields"),
        "C10": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no verified explicit unmatched-buy field"),
        "C11": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no verified explicit unmatched-sell field"),
        "C12": ("IMPLICIT_PRICE_STATE_ONLY", "NOT_AVAILABLE", "09:15-09:25 zero-volume changing prices resemble supplier virtual state but are not explicitly named"),
        "C13": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no explicit virtual match volume"),
        "C14": ("AVAILABLE", "NOT_EXPOSED", "trend-row timestamp and f86 quote timestamp"),
    }
    for field_id, (raw, repo, semantics) in em.items():
        history = "LATEST_DAY_ONLY" if field_id in {"C1", "C2", "C3", "C12", "C14"} else "NO"
        rows.append(_field_row("STOCK_EASTMONEY", field_id, raw, repo, semantics, history, "PENDING_TRADING_DAY", "direct raw response, installed AKShare source, repo wrapper inspection"))

    ak = {
        "C1": ("EXPOSED_VIA_PRE_MIN_ROWS", "NOT_USED", "latest-day 09:25/09:26 price state; not named final auction price"),
        "C2": ("EXPOSED_VIA_PRE_MIN_ROWS", "NOT_USED", "volume in hands; 09:26 row is opening-match-like but wrapper gives no semantic flag"),
        "C3": ("EXPOSED_VIA_PRE_MIN_ROWS", "NOT_USED", "amount; 09:26 row is opening-match-like but wrapper gives no semantic flag"),
        "C4": ("EXPOSED_VIA_BID_ASK", "NOT_USED", "buy_1 price"),
        "C5": ("EXPOSED_VIA_BID_ASK", "NOT_USED", "buy_1 volume converted to shares"),
        "C6": ("EXPOSED_VIA_BID_ASK", "NOT_USED", "sell_1 price"),
        "C7": ("EXPOSED_VIA_BID_ASK", "NOT_USED", "sell_1 volume converted to shares"),
        "C8": ("EXPOSED_VIA_BID_ASK", "NOT_USED", "buy levels 2-5"),
        "C9": ("EXPOSED_VIA_BID_ASK", "NOT_USED", "sell levels 2-5"),
        "C10": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no wrapper field"),
        "C11": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no wrapper field"),
        "C12": ("NOT_EXPLICIT", "NOT_AVAILABLE", "pre-minute OHLC-like state not labelled virtual match price"),
        "C13": ("NOT_AVAILABLE", "NOT_AVAILABLE", "no wrapper field"),
        "C14": ("EXPOSED_VIA_PRE_MIN_TIME", "NOT_USED", "minute timestamp; bid/ask wrapper lacks source timestamp"),
    }
    for field_id, (raw, repo, semantics) in ak.items():
        history = "LATEST_DAY_ONLY" if field_id in {"C1", "C2", "C3", "C12", "C14"} else "NO"
        rows.append(_field_row("AKSHARE", field_id, raw, repo, semantics, history, "PENDING_TRADING_DAY", f"AKShare 1.18.83 source and docs: {AKSHARE_DOC_URL}"))
    return pd.DataFrame(rows)


def build_historical_depth() -> pd.DataFrame:
    horizons = (
        ("LATEST_TRADING_DAY", "2026-09-04"),
        ("ABOUT_ONE_WEEK", "2026-08-28"),
        ("ABOUT_ONE_MONTH", "2026-08-07"),
        ("AUGUST_2026", "2026-08-03"),
        ("JULY_2026", "2026-07-20"),
        ("JUNE_2026", "2026-06-15"),
        ("MAY_2026", "2026-05-18"),
    )
    rows: list[dict[str, Any]] = []
    for source, endpoint, state, latest_ok, note in (
        ("SINA", "hq.sinajs.cn quote", "REALTIME_ONLY", False, "no historical date argument"),
        ("SINA", "KLineData scale=5", "NO_AUCTION_DATA", False, "historical regular bars start 09:35"),
        ("STOCK_EASTMONEY", "trends2 iscr=1 ndays=1", "LATEST_DAY_ONLY", True, "direct latest-day premarket response; no date argument"),
        ("STOCK_EASTMONEY", "trends2 iscr=1 ndays=5", "LATEST_DAY_ONLY", True, "direct test still returned only 2026-09-04 and began 09:30"),
        ("STOCK_EASTMONEY", "push2his trends2 ndays=5 iscr=0", "LIMITED_HISTORICAL", False, "source contract recent five trading days; runtime probe failed; not May-July"),
        ("AKSHARE", "stock_zh_a_hist_pre_min_em", "LATEST_DAY_ONLY", True, "official docs explicitly say most recent one trading day; no date parameter"),
        ("AKSHARE", "stock_zh_a_hist_min_em period=1", "LIMITED_HISTORICAL", False, "wrapper hard-codes ndays=5; runtime probe failed; not May-July"),
        ("AKSHARE", "stock_bid_ask_em", "REALTIME_ONLY", False, "current single-symbol quote; no date parameter"),
    ):
        for horizon, requested in horizons:
            available = bool(latest_ok and horizon == "LATEST_TRADING_DAY")
            rows.append(
                {
                    "source": source,
                    "interface": endpoint,
                    "horizon": horizon,
                    "requested_date": requested,
                    "sample_code": "000001",
                    "auction_data_returned": "YES" if available else "NO",
                    "returned_date": LATEST_TRADING_DATE_OBSERVED if available else "",
                    "historical_depth_state": state,
                    "earliest_confirmed_date": LATEST_TRADING_DATE_OBSERVED if latest_ok else "",
                    "latest_confirmed_date": LATEST_TRADING_DATE_OBSERVED if latest_ok else "",
                    "lookback_contract": "1 trading day" if state == "LATEST_DAY_ONLY" else ("recent 5 trading days" if state == "LIMITED_HISTORICAL" else "none"),
                    "evidence_note": note,
                }
            )
    return pd.DataFrame(rows)


def build_price_lineage() -> pd.DataFrame:
    trend = {row.timestamp[-5:]: row for row in map(parse_eastmoney_trend_row, EM_PRE_000001_ROWS)}
    official_open = EM_QUOTE_000001["f46_open"]
    records = []
    for source, timestamp, price, evidence_type in (
        ("EASTMONEY_PRE_MIN", trend["09:25"].timestamp, trend["09:25"].close, "preopen supplier state close"),
        ("EASTMONEY_PRE_MIN", trend["09:26"].timestamp, trend["09:26"].close, "opening-match-like row close"),
        ("SINA_CURRENT_QUOTE", "2026-09-04 15:36:00", float(parse_sina_quote_lines(SINA_QUOTE_SAMPLE).iloc[0]["open"]), "post-close open field"),
    ):
        diff = abs(float(price) - float(official_open))
        records.append(
            {
                "code": "000001",
                "trade_date": LATEST_TRADING_DATE_OBSERVED,
                "source": source,
                "source_timestamp": timestamp,
                "candidate_final_auction_price": price,
                "official_open_reference": official_open,
                "absolute_difference": diff,
                "exact_match": int(diff == 0),
                "within_0_01": int(diff <= 0.01 + 1e-12),
                "evidence_type": evidence_type,
                "lineage_status": "MATCH_ON_ONE_CODE_DATE_NOT_SUFFICIENT_FOR_READY",
            }
        )
    records.append(
        {
            "code": "SUMMARY",
            "trade_date": LATEST_TRADING_DATE_OBSERVED,
            "source": "ALL_PRICE_OBSERVATIONS",
            "source_timestamp": "",
            "candidate_final_auction_price": "",
            "official_open_reference": official_open,
            "absolute_difference": max(r["absolute_difference"] for r in records),
            "exact_match": sum(r["exact_match"] for r in records) / len(records),
            "within_0_01": sum(r["within_0_01"] for r in records) / len(records),
            "evidence_type": f"{len(records)} response fields but only 1 independent code-date",
            "lineage_status": "PARTIAL",
        }
    )
    return pd.DataFrame(records)


def build_volume_amount_semantics() -> pd.DataFrame:
    r0926 = parse_eastmoney_trend_row(EM_PRE_000001_ROWS[-1])
    implied = r0926.volume_hands * 100 * r0926.close
    return pd.DataFrame(
        [
            {
                "source": "EASTMONEY_PRE_MIN",
                "interface": "trends2 iscr=1 ndays=1",
                "timestamp": r0926.timestamp,
                "field": "f56 volume",
                "observed_value": r0926.volume_hands,
                "unit": "hands (100 shares)",
                "semantic_class": "PROVISIONAL_FINAL_OPENING_MATCH",
                "validation": "nonzero before continuous trading; amount arithmetic is exact",
                "confidence": "MEDIUM_ONE_SAMPLE",
                "modeling_eligible": "NO_NOT_YET",
            },
            {
                "source": "EASTMONEY_PRE_MIN",
                "interface": "trends2 iscr=1 ndays=1",
                "timestamp": r0926.timestamp,
                "field": "f57 amount",
                "observed_value": r0926.amount_cny,
                "unit": "CNY",
                "semantic_class": "PROVISIONAL_FINAL_OPENING_MATCH",
                "validation": f"volume*100*price={implied:.2f}; exact observed amount={r0926.amount_cny:.2f}",
                "confidence": "MEDIUM_ONE_SAMPLE",
                "modeling_eligible": "NO_NOT_YET",
            },
            {
                "source": "EASTMONEY_CURRENT_QUOTE",
                "interface": "stock/get f47/f48",
                "timestamp": "post-close snapshot",
                "field": "volume and amount",
                "observed_value": f"{EM_QUOTE_000001['f47_full_day_volume_hands']};{EM_QUOTE_000001['f48_full_day_amount_cny']}",
                "unit": "hands; CNY",
                "semantic_class": "FULL_DAY_CUMULATIVE_AT_OBSERVED_TIME",
                "validation": "post-close values include continuous trading and cannot be called auction-only",
                "confidence": "HIGH",
                "modeling_eligible": "NO_AS_AUCTION_VOLUME_AMOUNT",
            },
            {
                "source": "SINA_CURRENT_QUOTE",
                "interface": "hq.sinajs.cn",
                "timestamp": "post-close snapshot",
                "field": "volume and amount",
                "observed_value": "81437295;969948436.210",
                "unit": "shares; CNY",
                "semantic_class": "FULL_DAY_CUMULATIVE_AT_OBSERVED_TIME",
                "validation": "not auction-isolated; exact 09:25 behavior pending",
                "confidence": "HIGH_FOR_CUMULATIVE_LOW_FOR_AUCTION",
                "modeling_eligible": "NO_AS_AUCTION_VOLUME_AMOUNT",
            },
            {
                "source": "AKSHARE_PRE_MIN",
                "interface": "stock_zh_a_hist_pre_min_em",
                "timestamp": "latest trading day rows",
                "field": "成交量/成交额",
                "observed_value": "wrapper columns",
                "unit": "hands; CNY per official docs",
                "semantic_class": "UPSTREAM_ROW_VALUES_WITHOUT_EXPLICIT_AUCTION_FLAG",
                "validation": "same Eastmoney endpoint; wrapper does not change semantic ambiguity",
                "confidence": "MEDIUM",
                "modeling_eligible": "NO_NOT_YET",
            },
        ]
    )


def build_preopen_path_semantics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "EASTMONEY_PRE_MIN",
                "time_segment": "09:15-09:20",
                "observed_shape": "one-minute OHLC-like price state; zero volume/amount in sample",
                "semantic_state": "SUPPLIER_DEFINED_PREMARKET_BAR",
                "true_trade_ohlcv": "NO",
                "interpretation": "consistent with virtual/indicative matching state, but endpoint does not explicitly label it",
                "historical_depth": "latest trading day only",
                "live_available": "PENDING_TRADING_DAY",
            },
            {
                "source": "EASTMONEY_PRE_MIN",
                "time_segment": "09:20-09:25",
                "observed_shape": "changing OHLC-like prices with zero volume/amount in sample",
                "semantic_state": "SUPPLIER_DEFINED_PREMARKET_BAR",
                "true_trade_ohlcv": "NO",
                "interpretation": "price path is potentially a virtual match state; not a normal continuous-trade bar",
                "historical_depth": "latest trading day only",
                "live_available": "PENDING_TRADING_DAY",
            },
            {
                "source": "EASTMONEY_PRE_MIN",
                "time_segment": "09:25",
                "observed_shape": "final price state, zero volume/amount in sample",
                "semantic_state": "SUPPLIER_DEFINED_PREMARKET_BAR",
                "true_trade_ohlcv": "NO",
                "interpretation": "price matched official open in sample; quantity appears one row later",
                "historical_depth": "latest trading day only",
                "live_available": "PENDING_TRADING_DAY",
            },
            {
                "source": "EASTMONEY_PRE_MIN",
                "time_segment": "09:26",
                "observed_shape": "flat price with nonzero volume/amount before continuous trading",
                "semantic_state": "PROVISIONAL_FINAL_AUCTION_MATCH_ROW",
                "true_trade_ohlcv": "LIKELY_OPENING_MATCH_NOT_YET_MULTI_SAMPLE_VERIFIED",
                "interpretation": "price equals official open and amount equals price*shares exactly",
                "historical_depth": "latest trading day only",
                "live_available": "PENDING_TRADING_DAY",
            },
            {
                "source": "SINA_5M_KLINE",
                "time_segment": "09:15-09:25",
                "observed_shape": "no rows; repo history begins 09:35",
                "semantic_state": "NOT_AVAILABLE",
                "true_trade_ohlcv": "NO_DATA",
                "interpretation": "cannot reconstruct auction path",
                "historical_depth": "none for auction",
                "live_available": "NO",
            },
            {
                "source": "SINA_CURRENT_QUOTE",
                "time_segment": "09:15-09:25",
                "observed_shape": "snapshot fields exist, but no audit-window polling was possible on Sunday",
                "semantic_state": "UNKNOWN",
                "true_trade_ohlcv": "NOT_APPLICABLE_SNAPSHOT",
                "interpretation": "requires a trading-day time series to establish when open/volume/amount freeze",
                "historical_depth": "none",
                "live_available": "PENDING_TRADING_DAY",
            },
            {
                "source": "AKSHARE_PRE_MIN",
                "time_segment": "09:15-09:30",
                "observed_shape": "same Eastmoney supplier-defined one-minute rows",
                "semantic_state": "SUPPLIER_DEFINED_PREMARKET_BAR",
                "true_trade_ohlcv": "NO_FOR_ZERO_VOLUME_PREMATCH_ROWS",
                "interpretation": "AKShare formats the upstream response but does not add transaction semantics",
                "historical_depth": "latest trading day only",
                "live_available": "PENDING_TRADING_DAY",
            },
        ]
    )


def build_live_capability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "SINA",
                "interface": "hq.sinajs.cn quote",
                "audit_timestamp": AUDIT_ASOF,
                "non_trading_day_reachability": "PASS_SINGLE_AND_BATCH",
                "live_timing_validation": "PENDING_TRADING_DAY",
                "earliest_reliable_time": "UNKNOWN",
                "candidate_fields": "open cumulative volume/amount L1-L5 timestamp",
                "core_auction_semantics": "UNRESOLVED",
                "batch_support": "YES_AT_LEAST_50",
                "status": "PARTIAL",
            },
            {
                "source": "STOCK_EASTMONEY",
                "interface": "trends2 iscr=1 ndays=1",
                "audit_timestamp": AUDIT_ASOF,
                "non_trading_day_reachability": "PASS_ONE_SYMBOL_WITH_TRANSIENT_FAILURES",
                "live_timing_validation": "PENDING_TRADING_DAY",
                "earliest_reliable_time": "LIKELY_09:26_ROW_BUT_NOT_LIVE_VALIDATED",
                "candidate_fields": "final-price-like state provisional volume/amount timestamp",
                "core_auction_semantics": "PARTIAL",
                "batch_support": "NO_SINGLE_SECID",
                "status": "PARTIAL",
            },
            {
                "source": "STOCK_EASTMONEY",
                "interface": "ulist.np/get current batch",
                "audit_timestamp": AUDIT_ASOF,
                "non_trading_day_reachability": "PASS_N20_N30_N50_TRANSIENT_FAILURE_N5_N10",
                "live_timing_validation": "PENDING_TRADING_DAY",
                "earliest_reliable_time": "UNKNOWN",
                "candidate_fields": "open cumulative volume/amount current quote",
                "core_auction_semantics": "UNRESOLVED_AT_0925",
                "batch_support": "YES_AT_LEAST_50",
                "status": "PARTIAL",
            },
            {
                "source": "AKSHARE",
                "interface": "stock_zh_a_hist_pre_min_em",
                "audit_timestamp": AUDIT_ASOF,
                "non_trading_day_reachability": "PASS_ONCE_THEN_TRANSIENT_FAILURE",
                "live_timing_validation": "PENDING_TRADING_DAY",
                "earliest_reliable_time": "UNKNOWN",
                "candidate_fields": "time OHLC volume amount latest",
                "core_auction_semantics": "PARTIAL_SAME_UPSTREAM_AS_EASTMONEY",
                "batch_support": "NO",
                "status": "PARTIAL",
            },
            {
                "source": "AKSHARE",
                "interface": "stock_bid_ask_em",
                "audit_timestamp": AUDIT_ASOF,
                "non_trading_day_reachability": "FAIL_JSON_DECODE",
                "live_timing_validation": "PENDING_TRADING_DAY",
                "earliest_reliable_time": "UNKNOWN",
                "candidate_fields": "L1-L5 current quote by source contract",
                "core_auction_semantics": "UNRESOLVED",
                "batch_support": "NO",
                "status": "UNRESOLVED",
            },
        ]
    )


def build_batch_stress() -> pd.DataFrame:
    observed = [
        ("SINA", 5, 215, 5, 0, "PASS"),
        ("SINA", 10, 193, 10, 0, "PASS"),
        ("SINA", 20, 202, 20, 0, "PASS"),
        ("SINA", 30, 207, 30, 0, "PASS"),
        ("SINA", 50, 197, 50, 0, "PASS"),
        ("STOCK_EASTMONEY", 5, 2651, 0, 5, "TRANSIENT_TRANSPORT_FAILURE"),
        ("STOCK_EASTMONEY", 10, 2695, 0, 10, "TRANSIENT_TRANSPORT_FAILURE"),
        ("STOCK_EASTMONEY", 20, 333, 20, 0, "PASS"),
        ("STOCK_EASTMONEY", 30, 376, 30, 0, "PASS"),
        ("STOCK_EASTMONEY", 50, 318, 50, 0, "PASS"),
    ]
    rows = [
        {
            "source": source,
            "test_mode": "NON_TRADING_DAY_BATCH_REACHABILITY_NOT_LIVE_LATENCY",
            "candidate_count": n,
            "request_count": 1,
            "latency_ms": latency,
            "returned_count": returned,
            "missing_or_error_count": missing,
            "batch_supported": "YES",
            "rate_limit_observed": "NO" if status == "PASS" else "UNRESOLVED",
            "status": status,
            "interpretation": "proves request shape/parse only; cannot prove 09:25 freshness or completeness",
        }
        for source, n, latency, returned, missing, status in observed
    ]
    for n in (5, 10, 20, 30, 50):
        rows.append(
            {
                "source": "AKSHARE_PRE_MIN",
                "test_mode": "STATIC_WRAPPER_CAPABILITY_NOT_LIVE_LATENCY",
                "candidate_count": n,
                "request_count": n,
                "latency_ms": "",
                "returned_count": "",
                "missing_or_error_count": "",
                "batch_supported": "NO",
                "rate_limit_observed": "NOT_TESTED",
                "status": "SINGLE_SYMBOL_ONLY",
                "interpretation": "would require sequential calls; not executed outside the real live window",
            }
        )
    return pd.DataFrame(rows)


def build_source_shift() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "SINA",
                "historical_interface": "no historical auction interface confirmed",
                "live_interface": "hq.sinajs.cn current quote",
                "historical_fields": "none",
                "live_fields": "open cumulative volume/amount L1-L5",
                "semantic_shift": "UNKNOWN",
                "risk": "no same-source historical auction series",
            },
            {
                "source": "STOCK_EASTMONEY",
                "historical_interface": "pre-min latest day; hist 1m recent five days only",
                "live_interface": "pre-min current-session prospective; stock/get and ulist current quote",
                "historical_fields": "recent price/volume/amount rows only",
                "live_fields": "same endpoint family plus current quote",
                "semantic_shift": "UNKNOWN_PENDING_LIVE_WINDOW",
                "risk": "likely same upstream rows but 09:25/09:26 timing not observed live",
            },
            {
                "source": "AKSHARE",
                "historical_interface": "stock_zh_a_hist_pre_min_em latest day; hist_min period=1 recent five days",
                "live_interface": "same wrappers called in current session",
                "historical_fields": "formatted Eastmoney rows",
                "live_fields": "formatted Eastmoney rows/current L1-L5",
                "semantic_shift": "UNKNOWN_PENDING_LIVE_WINDOW",
                "risk": "wrapper does not supply a stable archival contract or auction semantic flags",
            },
        ]
    )


def build_capability_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "SINA",
                "raw_endpoint": "hq.sinajs.cn; CN_MarketDataService.getKLineData",
                "repo_wrapper": "regular 5m only; realtime quote not wrapped",
                "final_price": "CONDITIONAL_OPEN_FIELD",
                "volume": "CUMULATIVE_NOT_AUCTION_ISOLATED",
                "amount": "CUMULATIVE_NOT_AUCTION_ISOLATED",
                "order_book": "L1_L5_CURRENT",
                "unmatched": "NO",
                "preopen_path": "NO_CONFIRMED_PATH",
                "historical_depth": "REALTIME_ONLY_FOR_QUOTE; REGULAR_5M_NOT_AUCTION",
                "live_available": "TECHNICALLY_YES_TIMING_PENDING",
                "batch_available": "YES_50_OBSERVED",
                "semantics_confidence": "LOW_FOR_AUCTION",
                "auth_required": "NO_REFERER_REQUIRED",
                "rate_limit": "NOT_DOCUMENTED",
                "current_status": "AUCTION_SOURCE_UNRESOLVED",
            },
            {
                "source": "STOCK_EASTMONEY",
                "raw_endpoint": "trends2 iscr=1; stock/get; ulist.np/get",
                "repo_wrapper": "spot raw partially mapped then normalize drops quote fields; pre-min absent",
                "final_price": "AVAILABLE_PARTIAL_ONE_CODE_LINEAGE",
                "volume": "PROVISIONAL_0926_OPENING_MATCH",
                "amount": "PROVISIONAL_0926_OPENING_MATCH",
                "order_book": "CURRENT_CONTRACT",
                "unmatched": "NO_VERIFIED_FIELDS",
                "preopen_path": "SUPPLIER_DEFINED_PREMARKET_BAR",
                "historical_depth": "LATEST_DAY; RECENT_5_DAY_1M_CONTRACT_ONLY",
                "live_available": "TECHNICALLY_PLAUSIBLE_TIMING_PENDING",
                "batch_available": "YES_50_CURRENT_QUOTE; PRE_MIN_SINGLE_ONLY",
                "semantics_confidence": "MEDIUM_FOR_0926_ROW_LOW_FOR_LIVE_TIMING",
                "auth_required": "NO",
                "rate_limit": "NOT_DOCUMENTED_TRANSIENT_FAILURES_OBSERVED",
                "current_status": "AUCTION_SOURCE_PARTIAL",
            },
            {
                "source": "AKSHARE",
                "raw_endpoint": "wrapper over Eastmoney trends2/stock-get",
                "repo_wrapper": "not used for auction",
                "final_price": "EXPOSED_AS_PRE_MIN_ROW_NOT_EXPLICIT",
                "volume": "EXPOSED_AS_PRE_MIN_ROW_HANDS_NOT_EXPLICIT",
                "amount": "EXPOSED_AS_PRE_MIN_ROW_NOT_EXPLICIT",
                "order_book": "L1_L5_SINGLE_SYMBOL",
                "unmatched": "NO",
                "preopen_path": "SUPPLIER_DEFINED_PREMARKET_BAR",
                "historical_depth": "LATEST_DAY_PRE; RECENT_5_DAY_REGULAR_1M",
                "live_available": "TECHNICALLY_PLAUSIBLE_TIMING_PENDING",
                "batch_available": "NO",
                "semantics_confidence": "MEDIUM_LOW",
                "auth_required": "NO",
                "rate_limit": "UPSTREAM_NOT_DOCUMENTED",
                "current_status": "AUCTION_SOURCE_PARTIAL",
            },
        ]
    )


def _review() -> str:
    return f"""# v004d Direct Auction Source Capability Audit v001

## 先说人话

1. **Sina 有实时行情字段，但没有被证明为独立的集合竞价数据源。** 原始 quote 可批量拿到 open、当前累计量额、五档盘口和时间；可是它没有历史竞价回放，也没有显式的最终撮合量、未匹配量或虚拟匹配字段。只有在真实交易日 09:25 后连续轮询，才能证明当时的累计量额是否恰好等于集合竞价成交量额。
2. **Eastmoney 原始源确实比当前仓库缓存多。** `trends2/get?iscr=1&ndays=1` 返回 09:15--09:30 盘前分钟状态。000001 在 {LATEST_TRADING_DATE_OBSERVED} 的 09:25 价格与官方 open 一致；09:26 出现 5,094 手和 6,041,484 元，且 `5094 × 100 × 11.86 = 6,041,484`。这是“最终开盘撮合行”的强线索，但目前只有一个独立 code-date，尚不足以把语义标成 CLEAR。
3. **AKShare 不是第三套独立上游。** `stock_zh_a_hist_pre_min_em` 底层就是 Eastmoney `trends2/get`，固定 `ndays=1`，无日期参数；官方文档也明确为最近一个交易日。`stock_zh_a_hist_min_em(period=1)` 底层固定最近 5 个交易日，但这仍不能回补 May--July。
4. **May--July 历史竞价回补仍然被阻断。** Sina 没有历史竞价接口；Eastmoney/AKShare 已确认的盘前接口只到最近 1 日，最近 5 日接口也远达不到开发窗口。把 `ndays` 从 1 改为 5 的直接试验仍只返回 {LATEST_TRADING_DATE_OBSERVED}，并从 09:30 开始，说明这不是简单把 AKShare 参数改大就能解决。
5. **实盘能力有技术可行性迹象，但本轮不能宣称 READY。** Sina 批量 5/10/20/30/50 只读请求均返回；Eastmoney 批量 20/30/50 返回，5/10 遇到暂态传输失败。今天是周日，没有真实 09:15--09:30 时点证据，字段何时稳定、09:26 是否普遍出现、数据是否陈旧仍待交易日验证。
6. **上一轮的问题不只是仓库 wrapper/cache 没保存。** wrapper 的确漏掉了 Eastmoney 盘前接口和大量 quote 字段；但三个审计源也都不能回补 May--July 独立竞价量额。因此 `CURRENT_CACHE_LIMITATION_ONLY = NO`。

## Q1. Sina 到底有没有集合竞价数据？

有可在盘前轮询的实时 quote 候选字段，没有独立历史竞价序列，也没有显式的最终撮合量/未匹配量字段。正式状态：`AUCTION_SOURCE_UNRESOLVED`。

## Q2. Eastmoney / stock 到底有没有？

有。原始 `trends2` 能看到最近交易日的 09:15--09:30 supplier-defined premarket state，且 09:26 行在一个样本中高度符合最终开盘撮合价量额。问题是历史深度太短、样本数太少、真实 09:25 可用时点未验证。正式状态：`AUCTION_SOURCE_PARTIAL`。

## Q3. AKShare 到底有没有？

有 Eastmoney 盘前数据的 wrapper，但不是独立上游。它只暴露最近一个交易日，不能传入历史日期；盘口 wrapper 是单股实时接口。本轮运行还观察到暂态 JSON/连接失败。正式状态：`AUCTION_SOURCE_PARTIAL`。

## Q4. 哪个能拿价格、量、额？

- Eastmoney pre-minute / AKShare pre-minute：最近交易日可见价格、量、额；09:26 行是最有希望的最终撮合行，但语义仍为 provisional。
- Sina / Eastmoney current quote：有 open 和累计量额；在收盘后它们明确是全日累计，不是 auction-only。
- 未匹配买量、未匹配卖量、显式虚拟匹配量：三个源均未确认。

## Q5. 哪个能查历史？

没有一个能回补 May--July。Eastmoney/AKShare 的 pre-minute 是最近 1 个交易日；Eastmoney historical 1m wrapper contract 最多最近 5 个交易日，而且本轮历史域 runtime probe 失败。Sina 的历史 5m 是正常连续交易 bar，不含集合竞价。

## Q6. 哪个能用于实盘？

Eastmoney raw pre-minute 是最合理的潜在 live source；Sina quote 可作为潜在 fallback 观察源。两者都必须在真实交易日 09:15--09:27 轮询，验证字段冻结时点、完整率和语义后才能批准。

## Q7. 之前的问题是 source 没数据，还是 wrapper/cache 没保存？

两者都有：仓库没包 Eastmoney pre-minute，也把 spot quote 字段在 universe normalization 后丢掉；但上游可确认的历史深度同样不足，无法仅靠重新暴露 wrapper 回补 May--July。

## Q8. 现在能不能继续集合竞价路线？

不能直接进入信息审计或 BUY/PASS。可以保留一条数据积累路线：先在真实交易日确认 Eastmoney 09:25/09:26 语义与延迟，再决定是否从当日开始持续归档。由于本轮协议对 PARTIAL 的正式动作是停止评审，本次输出 `STOP_AND_REVIEW`，不自动创建归档任务。

## 能力边界

- 09:15--09:25 Eastmoney 行是 `SUPPLIER_DEFINED_PREMARKET_BAR`，价格变化而量额为零，不能当普通成交 OHLCV。
- 09:26 行只标 `PROVISIONAL_FINAL_AUCTION_MATCH_ROW`；一个样本的精确算术与开盘价一致不足以替代多股票、真实时点验证。
- 非交易日 batch latency 只证明请求形状和解析能力，不是 live latency。
- 无 outcome、无模型、无 BUY/PASS、无阈值或因子搜索。

## Formal result

SINA_AUCTION_SOURCE_STATE = AUCTION_SOURCE_UNRESOLVED

EASTMONEY_AUCTION_SOURCE_STATE = AUCTION_SOURCE_PARTIAL

AKSHARE_AUCTION_SOURCE_STATE = AUCTION_SOURCE_PARTIAL

PRIMARY_AUCTION_SOURCE_CANDIDATE = NONE

POTENTIAL_HISTORICAL_SOURCE = EASTMONEY_AKSHARE_RECENT_1_TO_5_TRADING_DAYS_ONLY

POTENTIAL_LIVE_SOURCE = EASTMONEY_RAW_TRENDS2_PREMARKET

HISTORICAL_AUCTION_BACKFILL_STATE = MAY_JULY_NOT_AVAILABLE_FROM_AUDITED_SOURCES

LIVE_AUCTION_COLLECTION_STATE = PENDING_REAL_TRADING_DAY_SEMANTICS_AND_TIMING_VALIDATION

LIVE_TIMING_VALIDATION = PENDING_TRADING_DAY

CURRENT_CACHE_LIMITATION_ONLY = NO

DIRECT_SOURCE_CAPABILITY_RESOLVED = PARTIAL

HISTORICAL_LIVE_SEMANTIC_SHIFT = UNKNOWN

OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

BUY_PASS_CREATED = NO

V004D_DIRECT_AUCTION_SOURCE_STATE = DIRECT_AUCTION_SOURCE_PARTIAL

NEXT_ACTION = STOP_AND_REVIEW

## Evidence URLs

- AKShare stock data documentation: {AKSHARE_DOC_URL}
- AKShare Eastmoney stock history wrapper source: {AKSHARE_SOURCE_URL}
- AKShare bid/ask wrapper source: {AKSHARE_BIDASK_SOURCE_URL}
"""


def build_outputs() -> tuple[dict[str, bytes], dict[str, Any]]:
    inventory = build_source_inventory()
    endpoints = build_raw_endpoint_inventory()
    fields = build_field_capability()
    history = build_historical_depth()
    price = build_price_lineage()
    volume = build_volume_amount_semantics()
    preopen = build_preopen_path_semantics()
    live = build_live_capability()
    batch = build_batch_stress()
    shift = build_source_shift()
    matrix = build_capability_matrix()
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(inventory),
        OUTPUT_FILENAMES[1]: _csv_bytes(endpoints),
        OUTPUT_FILENAMES[2]: _csv_bytes(fields),
        OUTPUT_FILENAMES[3]: _csv_bytes(history),
        OUTPUT_FILENAMES[4]: _csv_bytes(price),
        OUTPUT_FILENAMES[5]: _csv_bytes(volume),
        OUTPUT_FILENAMES[6]: _csv_bytes(preopen),
        OUTPUT_FILENAMES[7]: _csv_bytes(live),
        OUTPUT_FILENAMES[8]: _csv_bytes(batch),
        OUTPUT_FILENAMES[9]: _csv_bytes(shift),
        OUTPUT_FILENAMES[10]: _csv_bytes(matrix),
        OUTPUT_FILENAMES[11]: _review().encode("utf-8"),
    }
    if tuple(outputs) != OUTPUT_FILENAMES:
        raise RuntimeError("output manifest mismatch")
    context = {
        "audit_asof": AUDIT_ASOF,
        "akshare_version": "1.18.83",
        "sina_state": "AUCTION_SOURCE_UNRESOLVED",
        "eastmoney_state": "AUCTION_SOURCE_PARTIAL",
        "akshare_state": "AUCTION_SOURCE_PARTIAL",
        "primary_source": "NONE",
        "potential_live_source": "EASTMONEY_RAW_TRENDS2_PREMARKET",
        "historical_backfill": "MAY_JULY_NOT_AVAILABLE_FROM_AUDITED_SOURCES",
        "live_collection": "PENDING_REAL_TRADING_DAY_SEMANTICS_AND_TIMING_VALIDATION",
        "overall_state": "DIRECT_AUCTION_SOURCE_PARTIAL",
        "next_action": "STOP_AND_REVIEW",
        "outcome_accessed": "NO",
        "model_trained": "NO",
        "feature_search": "NO",
        "buy_pass_created": "NO",
    }
    return outputs, context


def run(root: str | Path) -> tuple[Path, dict[str, Any], dict[str, str]]:
    root_path = Path(root).resolve()
    outputs, context = build_outputs()
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, payload in outputs.items():
        path = output_dir / name
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return output_dir, context, hashes

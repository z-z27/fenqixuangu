# -*- coding: utf-8 -*-
"""v004c v002 — 涨停池完整性修复 (第二阶段)。

根因: derived 池扫描只读 daily_unadjusted 缓存 (3034 只), universe 快照
(3098 只) 中约 64 只缺缓存 -> 这些股票当日涨停也进不了池 (v001 的
"池源覆盖缺口" 实为日线缓存缺口, 如 002217@05-07 日线涨停但池缺失)。

本工具:
1. 找出 universe - daily_unadjusted 的缺失股票, 从 daily 缓存 (tencent,
   data/cache/daily) 或网络 (MarketDataProvider.fetch_daily_history) 补齐
   [CAL_START, CAL_END] 日线, append-only 合入 daily_unadjusted 缓存;
2. 对全部 required 交易日做完整性核对: universe 日线涨停集合 vs 池文件,
   输出缺失股票清单 (当日日线涨停但不在池文件);
3. 对有缺口的日期重新 derive (读补齐后的缓存) 并覆盖池文件 (canonical
   daily_limitup_derived 修正)。

用法:
    python tools/recover_v004c_pool_completeness_v002.py [--check-only] [--workers N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baostock as bs  # noqa: E402

from src.config import get_data_config  # noqa: E402
from src.loaders import MarketDataService  # noqa: E402
from src.v004c_pairwise_feature_contract import (  # noqa: E402
    CONTRACT_FEATURE_NAMES,
)

DAILY_UNADJUSTED_DIR = ROOT / "data" / "cache" / "daily_unadjusted"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
UNIVERSE_PKL = ROOT / "data" / "cache" / "universe" / "eastmoney_main_board_universe.pkl"

CAL_START = "2026-03-20"  # required 窗口最早 (04-03) 再留余量
CAL_END = "2026-07-01"

# 阶段 2 核对出的缺口股票 (2026-06 月, 25 只去重) — 硬编码自首次 check-only 输出
GAP_STOCKS = {
    "002404", "000068", "002987", "600586", "603065", "600522", "603773",
    "600162", "600500", "600711", "601996", "000887", "000608", "002051",
    "600703", "600707", "000955", "002484", "600322", "601872", "603380",
    "603757", "600909", "603713", "600367",
}


def load_trade_calendar() -> list[str]:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
    try:
        rs = bs.query_trade_dates(start_date="2026-02-01", end_date="2026-07-15")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()
    return [r[0] for r in rows if r[1] == "1"]


def _fetch_daily_baostock(code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """BaoStock 日线拉取 (实测稳定快速), 返回 date/open/high/low/close/volume 或 None。"""
    import baostock as bs
    sym = ("sh" if code.startswith("6") else "sz") + "." + code
    lg = bs.login()
    if lg.error_code != "0":
        return None
    try:
        rs = bs.query_history_k_data_plus(
            sym, "date,open,high,low,close,preclose,volume,amount,turn",
            start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        frame = pd.DataFrame(rows, columns=rs.fields)
        for col in ("open", "high", "low", "close", "volume", "amount", "turn"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["code"] = code
        return frame
    finally:
        bs.logout()


def is_limit_up(close, high, prev_close) -> bool:
    from decimal import Decimal, ROUND_HALF_UP
    if prev_close is None or prev_close <= 0 or close is None or high is None:
        return False
    limit_price = float((Decimal(str(prev_close)) * Decimal("1.10"))
                        .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    pct = (close / prev_close - 1.0) * 100.0
    return (pct >= 9.7 and close >= limit_price - 0.011 and high >= limit_price - 0.011)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    trade_dates = load_trade_calendar()

    universe = pd.read_pickle(UNIVERSE_PKL)
    universe["code"] = universe["code"].astype(str).str.zfill(6)
    universe_codes = set(universe["code"].tolist())
    cached_codes = set(p.name.replace("_daily.pkl", "") for p in DAILY_UNADJUSTED_DIR.glob("*_daily.pkl"))
    daily_cache_codes = set(p.name.replace("_daily.pkl", "") for p in DAILY_DIR.glob("*_daily.pkl"))
    missing = sorted(universe_codes - cached_codes)
    print(f"universe: {len(universe_codes)} cached_unadjusted: {len(cached_codes)} "
          f"missing: {len(missing)}")

    # ---- 阶段 1: 补齐缺失股票日线 (从 daily 缓存或网络, 并行) ----
    # 注意: 只补"当日可能涨停且需进池"的股票 — 通过阶段 2 核对得到的缺口清单驱动;
    #       universe - daily_unadjusted 的其余缺缓存股票多为退市/长期停牌 (无日线,
    #       不产生交易 -> 不在池), 无需补齐。
    if not args.check_only:
        svc = MarketDataService(get_data_config())
        copied = fetched = covered = failed = 0
        from concurrent.futures import ThreadPoolExecutor

        def fill_one(code: str) -> tuple[str, str]:
            """确保该股日线覆盖缺口日期 (缓存覆盖则跳过; 否则 daily 缓存/网络补齐)。"""
            target_dates = [d for d in required_window_check if d is not None]
            cached = svc.daily_unadjusted_cache.read(code)
            cached_dates: set[str] = set()
            if cached is not None and not cached.empty and "date" in cached.columns:
                cached_dates = set(pd.to_datetime(cached["date"], errors="coerce")
                                   .dt.strftime("%Y-%m-%d").dropna())
            if target_dates and all(d in cached_dates for d in target_dates):
                return code, "covered"
            src = DAILY_DIR / f"{code}_daily.pkl"
            src_frame = None
            if src.exists():
                try:
                    src_frame = pd.read_pickle(src)
                except Exception:
                    src_frame = None
            if src_frame is not None and not src_frame.empty and "date" in src_frame.columns:
                src_frame = src_frame.copy()
                src_frame["date"] = pd.to_datetime(src_frame["date"], errors="coerce") \
                    .dt.strftime("%Y-%m-%d")
                src_frame = src_frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
                merged = src_frame if cached is None else pd.concat(
                    [cached, src_frame], ignore_index=True).drop_duplicates("date", keep="last") \
                    .sort_values("date").reset_index(drop=True)
                svc.daily_unadjusted_cache.write(code, merged)
                return code, "copied"
            try:
                fetched_frame = _fetch_daily_baostock(code, CAL_START, CAL_END)
                if fetched_frame is not None and not fetched_frame.empty:
                    fetched_frame["date"] = pd.to_datetime(fetched_frame["date"], errors="coerce") \
                        .dt.strftime("%Y-%m-%d")
                    fetched_frame = fetched_frame.dropna(subset=["date"]) \
                        .drop_duplicates("date", keep="last")
                    merged = fetched_frame if cached is None else pd.concat(
                        [cached, fetched_frame], ignore_index=True) \
                        .drop_duplicates("date", keep="last") \
                        .sort_values("date").reset_index(drop=True)
                    svc.daily_unadjusted_cache.write(code, merged)
                    return code, "fetched"
                return code, "empty"
            except Exception:
                return code, "failed"

        # 补清单 = 6 月核对缺口股票 ∪ 全部 319 事件股 (事件股在 D0 必涨停,
        # 缺缓存则池必然漏它们, 如 002217@05-07 — check 因无日线看不见, 必须显式补)
        # 缺口相关股票 = 6 月核对缺口 (fill_one 内部按日期覆盖检查;
        # 002217 等事件股由 universe 修补 + 已复制的日线覆盖, 不在此拉网络)
        required_window_check = [d for d in trade_dates
                                 if "2026-04-03" <= d <= "2026-06-30"]
        fill_codes = sorted(GAP_STOCKS & universe_codes)
        # 串行 (baostock 全局连接不支持并发)
        for code in fill_codes:
            status = fill_one(code)
            if status == "copied":
                copied += 1
            elif status == "fetched":
                fetched += 1
            elif status == "covered":
                covered += 1
            else:
                failed += 1
        print(f"daily fill (gap stocks {len(fill_codes)}): "
              f"covered={covered} copied={copied} fetched={fetched} failed={failed}",
              flush=True)

    # ---- 阶段 2: 完整性核对 (universe 涨停集合 vs 池文件) ----
    # 核对范围 = required 窗口 (2026-04-03 .. 2026-06-30, 20 日 lookback 需要)
    trade_dates = load_trade_calendar()
    required = [d for d in trade_dates if "2026-04-03" <= d <= "2026-06-30"]

    def check_gaps() -> dict[str, list[str]]:
        # 一次性加载全部 daily_unadjusted 到内存
        daily_all: dict[str, pd.DataFrame] = {}
        for path in sorted(DAILY_UNADJUSTED_DIR.glob("*_daily.pkl")):
            code = path.name.replace("_daily.pkl", "")
            try:
                frame = pd.read_pickle(path)
            except Exception:
                continue
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
            frame = frame.sort_values("date").reset_index(drop=True)
            daily_all[code] = frame
        print(f"daily_all loaded: {len(daily_all)}", flush=True)

        zt_by_date: dict[str, set[str]] = {d: set() for d in required}
        for code, frame in daily_all.items():
            if code not in universe_codes:
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(float)
            highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
            dates = frame["date"].tolist()
            prev = None
            for i, d in enumerate(dates):
                if d in zt_by_date and is_limit_up(closes[i], highs[i], prev):
                    zt_by_date[d].add(code)
                prev = closes[i] if not np.isnan(closes[i]) else prev

        gaps: dict[str, list[str]] = {}
        for d in required:
            pool_path = POOL_DIR / f"{d}_limitups.pkl"
            if not pool_path.exists():
                gaps[d] = sorted(zt_by_date[d])
                continue
            pool = pd.read_pickle(pool_path)
            pool_codes = set(pool["code"].astype(str).str.zfill(6))
            missing_in_pool = sorted(zt_by_date[d] - pool_codes)
            if missing_in_pool:
                gaps[d] = missing_in_pool
        total_gap = sum(len(v) for v in gaps.values())
        print(f"completeness check over {len(required)} dates: "
              f"dates with gaps: {len(gaps)} total missing stocks: {total_gap}", flush=True)
        for d, codes in sorted(gaps.items()):
            print(f"  {d}: {len(codes)} missing -> {codes[:8]}{'...' if len(codes) > 8 else ''}",
                  flush=True)
        return gaps

    # ---- 阶段 2b: universe 快照修正 (事件股缺失补充, 如 002217) ----
    v002_for_universe = pd.read_csv(
        ROOT / "reports" / "research" / "v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv",
        encoding="utf-8-sig", dtype={"code": str})
    v002_for_universe["code"] = v002_for_universe["code"].astype(str).str.zfill(6)
    missing_from_universe = sorted(set(v002_for_universe["code"]) - universe_codes)
    if missing_from_universe and not args.check_only:
        from src.code_utils import is_main_board_code
        rows = []
        for code in missing_from_universe:
            if not is_main_board_code(code):
                continue
            sub = v002_for_universe[v002_for_universe["code"] == code]
            rows.append({
                "code": code,
                "name": str(sub.iloc[0].get("name", "")),
                "market": "SH" if code.startswith("6") else "SZ",
                "industry": "",
                "float_market_cap": None,
                "total_market_cap": None,
                "source": "v002_event_universe_repair",
            })
        if rows:
            repaired = pd.concat([universe, pd.DataFrame(rows)], ignore_index=True)
            repaired = repaired.drop_duplicates("code", keep="last")
            pd.to_pickle(repaired, UNIVERSE_PKL)
            universe_codes |= set(r["code"] for r in rows)
            print(f"universe repaired: added {len(rows)} -> {len(repaired)}", flush=True)

    # ---- 阶段 3: 循环 check -> derive 缺口日期 (覆盖池文件) ----
    if not args.check_only:
        svc = MarketDataService(get_data_config())
        for _round in range(2):
            gaps = check_gaps()
            if not gaps:
                break
            for d in sorted(gaps.keys()):
                print(f"[re-derive] {d} ...", flush=True)
                frame = svc._derive_limit_up_pool_from_daily(d, force_refresh=False,
                                                             workers=args.workers)
                svc.limit_up_cache.write(d, frame)
                print(f"[re-derive] {d}: rows={len(frame)}", flush=True)
        gaps = check_gaps()
        if gaps:
            print(f"REMAINING GAPS: {len(gaps)} dates, "
                  f"{sum(len(v) for v in gaps.values())} stocks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

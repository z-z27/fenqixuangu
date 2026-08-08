# -*- coding: utf-8 -*-
"""v004c Pool Data Correctness Fix v002 — 历史涨停池恢复工具。

对 319 事件 recent_pool_appearance_count_20d 所需窗口内缺失的涨停池日期,
复用 canonical daily_limitup_derived pipeline (MarketDataService.
_derive_limit_up_pool_from_daily) 逐日重建完整涨停池, 结果写入
data/cache/limit_ups/ (与现有 *_limitups.pkl 同格式, append-only, 不覆盖已有文件)。

恢复判定:
- derive 成功 (rows > 0)          -> recovered
- derive 抛错 / rows == 0         -> failed (记录原因)
- 该日期已存在缓存文件            -> skipped (already cached)

完整性: derive 为确定性算法 (日线涨停判定 + 连板计数), universe =
eastmoney_main_board 主板快照 (3098), 每个 required 交易日输出
date_seen/checked/rows/errors 供 v002 builder 写入 pool-date manifest。

用法:
    python tools/recover_v004c_pool_history_v002.py [--workers N] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baostock as bs  # noqa: E402

from src.config import get_data_config  # noqa: E402
from src.loaders import MarketDataService  # noqa: E402

V002_CSV = (ROOT / "reports" / "research"
            / "v004c_baostock_d1_dev_v002_20260506_20260630"
            / "v004c_baostock_d1_dev_v002.csv")
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
LOOKBACK_TRADE_DAYS = 20
CALENDAR_START = "2026-02-01"
CALENDAR_END = "2026-07-15"


def load_trade_calendar() -> list[str]:
    """正式交易日历 (baostock query_trade_dates)。"""
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
    try:
        rs = bs.query_trade_dates(start_date=CALENDAR_START, end_date=CALENDAR_END)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()
    return [r[0] for r in rows if r[1] == "1"]


def required_pool_dates(trade_dates: list[str], v002: pd.DataFrame) -> list[str]:
    """全开发集 required pool 交易日: 每个 D1 的 [D1-19, D1] 窗口并集。"""
    idx = {d: i for i, d in enumerate(trade_dates)}
    required: set[str] = set()
    for d1 in v002["signal_date"].astype(str).unique():
        i = idx[d1]
        required.update(trade_dates[max(0, i - (LOOKBACK_TRADE_DAYS - 1)): i + 1])
    return sorted(required)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trade_dates = load_trade_calendar()
    v002 = pd.read_csv(V002_CSV, encoding="utf-8-sig", dtype={"code": str})
    v002["signal_date"] = v002["signal_date"].astype(str)
    required = required_pool_dates(trade_dates, v002)

    cached = sorted(p.name.replace("_limitups.pkl", "") for p in POOL_DIR.glob("*_limitups.pkl"))
    cached_set = set(cached)
    missing = [d for d in required if d not in cached_set]

    print(f"required dates: {len(required)}  ({required[0]} .. {required[-1]})")
    print(f"cached in range: {len([d for d in required if d in cached_set])}")
    print(f"missing dates: {len(missing)}")
    for d in missing:
        print(f"  - {d}")

    if args.dry_run:
        return 0

    svc = MarketDataService(get_data_config())
    results: list[dict] = []
    for date_text in missing:
        print(f"\n[recover] {date_text} ...", flush=True)
        try:
            frame = svc._derive_limit_up_pool_from_daily(
                date_text, force_refresh=False, workers=args.workers)
            if frame is None or frame.empty:
                results.append({"trade_date": date_text, "status": "failed",
                                "reason": "derive returned no rows"})
                print(f"[recover] {date_text}: FAILED (no rows)")
                continue
            svc.limit_up_cache.write(date_text, frame)
            results.append({"trade_date": date_text, "status": "recovered",
                            "rows": len(frame), "reason": ""})
            print(f"[recover] {date_text}: recovered rows={len(frame)}", flush=True)
        except Exception as exc:
            results.append({"trade_date": date_text, "status": "failed",
                            "reason": str(exc)[:200]})
            print(f"[recover] {date_text}: FAILED ({type(exc).__name__}: {exc})", flush=True)

    df = pd.DataFrame(results)
    print("\n=== recovery summary ===")
    print(df.to_string(index=False))
    ok = int((df["status"] == "recovered").sum()) if len(df) else 0
    print(f"recovered: {ok}/{len(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

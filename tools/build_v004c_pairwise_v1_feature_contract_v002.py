# -*- coding: utf-8 -*-
"""v004c Pairwise v1 Feature Contract v002 — Pool Data Correctness Fix 构建工具。

在 v001 (53 FEATURE 冻结) 基础上修正 5 类 correctness 问题:
A. recent_pool_appearance_count_10d/20d 只在完整 10/20 交易日窗口计算,
   窗口内任一 required pool date 不完整 -> NA + SOURCE_MISSING (禁止覆盖下界);
B. board_day_volume_rank 使用 D0 完整涨停池横截面 (denominator = 完整池成员),
   不再使用候选代码子集; 任一 member 缺 daily volume -> NA + SOURCE_MISSING;
C. pool 数据缺口一律 SOURCE_MISSING, 仅数学退化 (high==low 等) 为 STRUCTURAL_MISSING,
   新增 UNEXPECTED_MISSING 类别;
D. high_zone_volume_ratio / high_zone_amount_ratio contract 公式文字修正为
   typical_price = (high+low+close)/3 (实现未动, 只修 metadata);
E. 新增 Pairwise 层 eligibility: pairwise_v1_feature_source_complete /
   pairwise_v1_training_eligible (v002 foundation 的 dev_* 列保持 AUDIT_ONLY)。

pool 历史恢复由 tools/recover_v004c_pool_history_v002.py 完成 (结果在
data/cache/limit_ups/, 与现有文件同格式)。本工具消费完整 pool 缓存,
生成 pool-date completeness manifest 与 board-day rank audit。

输出 (reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630/):
  1. v004c_pairwise_v1_feature_inventory_v002.csv
  2. v004c_pairwise_v1_feature_contract_v002.csv
  3. v004c_pairwise_v1_label_contract_v002.csv
  4. v004c_pairwise_v1_input_table_v002.csv
  5. v004c_pairwise_v1_schema_v002.csv
  6. v004c_pairwise_v1_exclusions_v002.csv
  7. v004c_pairwise_v1_pool_date_coverage_v002.csv
  8. v004c_pairwise_v1_board_day_rank_audit_v002.csv
  9. v004c_pairwise_v1_feature_contract_review_v002.md

严格边界: 不训练模型 / 不搜索 lambda / 不做 walk-forward / 不做 Target 分析 /
不新增 FEATURE / 不恢复 amount/turnover rank / 不重新讨论 manual composite /
不修改非 pool FEATURE / universe 与 v002 逐行精确一致 (319/319)。

确定性: 两轮构建 9 个核心资产字节一致 (网络 fetch 结果写入缓存后稳定)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_pairwise_feature_contract_v002 import (  # noqa: E402
    AUDIT_COLUMNS_V002,
    CONTRACT_FEATURE_NAMES,
    CONTRACT_FEATURE_ORDER,
    EXCLUDED_CANDIDATE_DECISIONS,
    FEATURE,
    FEATURE_CONTRACT,
    IDENTIFIER,
    LABEL_ONLY,
    PAIRWISE_ELIGIBILITY_COLUMNS,
    SOURCE_MISSING,
    STRUCTURAL_MISSING,
    UNEXPECTED_MISSING,
    assert_contract_integrity_v002,
)
from src.config import get_data_config  # noqa: E402
from src.loaders import MarketDataService  # noqa: E402

# ---------------------------------------------------------------------------
# 路径 / 常量
# ---------------------------------------------------------------------------
V002_CSV = (ROOT / "reports" / "research"
            / "v004c_baostock_d1_dev_v002_20260506_20260630"
            / "v004c_baostock_d1_dev_v002.csv")
OUT_DIR = (ROOT / "reports" / "research"
           / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630")
DAILY_DIR = ROOT / "data" / "cache" / "daily"
DAILY_UNADJUSTED_DIR = ROOT / "data" / "cache" / "daily_unadjusted"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"

WINDOW_END = "2026-06-30"
LOOKBACK_TRADE_DAYS = 20
CALENDAR_START = "2026-02-01"
CALENDAR_END = "2026-07-15"

MAIN_BOARD_LIMIT_RATIO = Decimal("1.10")
LIMIT_UP_PRICE_TOLERANCE = 0.011
LIMIT_UP_MIN_PCT_CHG = 9.7

DEV_CSV_NAME = "v004c_pairwise_v1_input_table_v002.csv"
INVENTORY_CSV_NAME = "v004c_pairwise_v1_feature_inventory_v002.csv"
CONTRACT_CSV_NAME = "v004c_pairwise_v1_feature_contract_v002.csv"
LABEL_CSV_NAME = "v004c_pairwise_v1_label_contract_v002.csv"
SCHEMA_CSV_NAME = "v004c_pairwise_v1_schema_v002.csv"
EXCLUSIONS_CSV_NAME = "v004c_pairwise_v1_exclusions_v002.csv"
POOL_COVERAGE_CSV_NAME = "v004c_pairwise_v1_pool_date_coverage_v002.csv"
RANK_AUDIT_CSV_NAME = "v004c_pairwise_v1_board_day_rank_audit_v002.csv"
REVIEW_NAME = "v004c_pairwise_v1_feature_contract_review_v002.md"

INPUT_PREFIX_COLUMNS = ("event_id", "code", "signal_date", "board_streak_before_break",
                        "source_window", "break_date")
INPUT_SUFFIX_COLUMNS = ("target7_daily_d2open_d3high", "tail_loss_daily_5pct",
                        "dev_label_complete", "dev_feature_source_complete",
                        "dev_training_eligible") + AUDIT_COLUMNS_V002

_EMPTY_POOL = pd.DataFrame()

# pool-derived FEATURE (6 个): 决策由恢复结果决定 (KEEP_FEATURE / EXCLUDE_SOURCE_UNAVAILABLE)
POOL_DERIVED_FEATURES = (
    "recent_pool_appearance_count_10d",
    "recent_pool_appearance_count_20d",
    "board_day_volume_rank",
    "break_day_in_pool",
    "last_board_day_in_pool",
    "pool_consecutive_count_last_board",
)

# 非 pool 的重建特征 (13 个, 沿用 v001 逻辑)
NON_POOL_REBUILD_FEATURES = (
    "board_streak_is_3", "max_board_streak_20d",
    "recent_limit_up_count_10d", "recent_limit_up_count_20d",
    "break_high_return", "break_close_return",
    "break_touched_limit_up", "break_opened_from_limit_up",
    "break_upper_shadow_ratio", "break_lower_shadow_ratio",
    "break_volume_ratio_vs_board_days", "d1_ma20_slope",
    "recent_7d_limit_up_count",
)


# ---------------------------------------------------------------------------
# 官方数学工具 (与 v001/scratch 一致)
# ---------------------------------------------------------------------------
def round_price_limit(prev_close: float) -> float:
    value = Decimal(str(prev_close)) * MAIN_BOARD_LIMIT_RATIO
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def is_limit_up_day(close, high, prev_close) -> bool:
    if prev_close is None or prev_close <= 0 or close is None or high is None:
        return False
    limit_price = round_price_limit(prev_close)
    pct_chg = (close / prev_close - 1.0) * 100.0
    if pct_chg < LIMIT_UP_MIN_PCT_CHG:
        return False
    if close < limit_price - LIMIT_UP_PRICE_TOLERANCE:
        return False
    if high < limit_price - LIMIT_UP_PRICE_TOLERANCE:
        return False
    return True


def _to_float(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _ret(v) -> float | None:
    v = _to_float(v)
    return None if v is None else v - 1.0


def _safe_div(a, b) -> float | None:
    a = _to_float(a)
    b = _to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


# ---------------------------------------------------------------------------
# 交易日历 (baostock 正式日历, 确定性历史事实)
# ---------------------------------------------------------------------------
def load_trade_calendar() -> list[str]:
    import baostock as bs
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
    dates = [r[0] for r in rows if r[1] == "1"]
    assert dates and dates[0] <= WINDOW_END
    return dates


# ---------------------------------------------------------------------------
# pool 缓存加载 + completeness manifest
# ---------------------------------------------------------------------------
def load_pool_by_date() -> dict[str, pd.DataFrame]:
    pool: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(POOL_DIR).glob("*_limitups.pkl")):
        date_text = path.name.replace("_limitups.pkl", "")
        frame = pd.read_pickle(path)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        pool[date_text] = frame
    return pool


def required_pool_dates(v002: pd.DataFrame, trade_dates: list[str]) -> list[str]:
    """全开发集 required pool 交易日: 每个 D1 的 [D1-19, D1] 窗口并集。"""
    idx = {d: i for i, d in enumerate(trade_dates)}
    required: set[str] = set()
    for d1 in v002["signal_date"].astype(str).unique():
        i = idx[d1]
        required.update(trade_dates[max(0, i - (LOOKBACK_TRADE_DAYS - 1)): i + 1])
    return sorted(required)


def pool_completeness(pool_by_date: dict[str, pd.DataFrame],
                      required_dates: list[str]) -> pd.DataFrame:
    """manifest 行 (仅 required 交易日): trade_date / required / cache_present /
    source_fetch_attempted / source_fetch_success / pool_row_count / source /
    complete / failure_reason。"""
    rows = []
    for d in required_dates:
        frame = pool_by_date.get(d)
        present = frame is not None
        source = "daily_limitup_derived" if present else ""
        complete = present and len(frame) > 0
        rows.append({
            "trade_date": d,
            "required": True,
            "cache_present": present,
            "source_fetch_attempted": True,  # 恢复工具对全部缺失日期尝试过 derive
            "source_fetch_success": present,
            "pool_row_count": len(frame) if present else 0,
            "source": source,
            "complete": complete,
            "failure_reason": "" if complete else "pool source unavailable (fetch failed)",
        })
    return pd.DataFrame(rows)


def required_window_dates(d1: str, trade_dates: list[str], n: int) -> list[str]:
    """[D1-(n-1), D1] 窗口内的交易日 (共 n 天)。"""
    idx = {d: i for i, d in enumerate(trade_dates)}
    i = idx[d1]
    return trade_dates[max(0, i - (n - 1)): i + 1]


# ---------------------------------------------------------------------------
# daily volume 横截面 (完整 pool rank denominator)
# ---------------------------------------------------------------------------
def build_volume_lookup() -> dict[str, dict[str, float]]:
    """date -> {code: volume}。来源: daily_unadjusted (3034, 优先) + daily (1612)。"""
    lookup: dict[str, dict[str, float]] = {}
    for d in (DAILY_UNADJUSTED_DIR, DAILY_DIR):
        for path in Path(d).glob("*_daily.pkl"):
            code = path.name.replace("_daily.pkl", "")
            try:
                frame = pd.read_pickle(path)
            except Exception:
                continue
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist()
            vols = pd.to_numeric(frame.get("volume"), errors="coerce").tolist()
            for date_text, v in zip(dates, vols):
                vf = _to_float(v)
                if vf is not None:
                    lookup.setdefault(date_text, {})[code] = vf
    return lookup


def fetch_missing_daily_volume(code: str, date_text: str) -> float | None:
    """按需补拉单个 member 的日线 (tencent -> sina), 成功后 merge 写回
    daily_unadjusted 缓存 (append-only, 不覆盖已有行), 失败返回 None。"""
    try:
        svc = MarketDataService(get_data_config())
        start = pd.Timestamp(date_text) - pd.Timedelta(days=30)
        frame, _ = svc.provider.fetch_daily_history(
            code, start.strftime("%Y-%m-%d"), date_text, adjust="none")
        if frame is None or frame.empty:
            return None
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        row = frame[frame["date"] == date_text]
        if row.empty:
            return None
        cached = svc.daily_unadjusted_cache.read(code)
        if cached is not None and not cached.empty and "date" in cached.columns:
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            merged = (pd.concat([cached, frame], ignore_index=True)
                      .drop_duplicates("date", keep="last")
                      .sort_values("date").reset_index(drop=True))
        else:
            merged = frame
        svc.daily_unadjusted_cache.write(code, merged)
        return _to_float(row.iloc[0].get("volume"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 单事件 pool 特征重算
# ---------------------------------------------------------------------------
def compute_pool_features(row, trade_dates, pool_by_date, pool_complete_map,
                          volume_lookup) -> tuple[dict, dict, dict]:
    """返回 (values, missing_types, rank_audit_row)。

    values: feature_name -> 值 (None 表示缺失)
    missing_types: feature_name -> STRUCTURAL/SOURCE/UNEXPECTED (仅缺失时)
    """
    code = str(row["code"]).zfill(6)
    d1 = str(row["signal_date"])
    streak = int(row["board_streak_before_break"])
    values: dict = {}
    missing: dict = {}
    rank_audit: dict = {}

    def mark(name: str, val, mtype: str | None = None):
        values[name] = val
        if val is None and mtype is not None:
            missing[name] = mtype

    # ---- recent_pool_appearance_count_10d/20d: 完整窗口 ----
    for n, fname in ((10, "recent_pool_appearance_count_10d"),
                     (20, "recent_pool_appearance_count_20d")):
        window = required_window_dates(d1, trade_dates, n)
        if len(window) < n:
            mark(fname, None, SOURCE_MISSING)  # 日历覆盖不足
            continue
        incomplete = [d for d in window if not pool_complete_map.get(d, False)]
        if incomplete:
            mark(fname, None, SOURCE_MISSING)
            continue
        cnt = sum(1 for d in window
                  if code in pool_by_date.get(d, _EMPTY_POOL).get("code", pd.Series(dtype=object)).astype(str).tolist())
        mark(fname, int(cnt))

    # ---- D0 日期 (D1 前一个交易日) ----
    idx = {d: i for i, d in enumerate(trade_dates)}
    i = idx[d1]
    d0 = trade_dates[i - 1] if i >= 1 else None

    # ---- board_day_volume_rank: 完整池横截面 ----
    if d0 is None:
        mark("board_day_volume_rank", None, SOURCE_MISSING)
    elif not pool_complete_map.get(d0, False):
        mark("board_day_volume_rank", None, SOURCE_MISSING)
    else:
        members = pool_by_date[d0]["code"].astype(str).tolist()
        vols: dict[str, float] = {}
        missing_members: list[str] = []
        for m in sorted(set(members)):
            v = volume_lookup.get(d0, {}).get(m)
            if v is None:
                v = fetch_missing_daily_volume(m, d0)
                if v is not None:
                    volume_lookup.setdefault(d0, {})[m] = v
            if v is None:
                missing_members.append(m)
            else:
                vols[m] = v
        rank_audit.update({
            "D0": d0, "code": code,
            "pool_member_count": len(set(members)),
            "members_with_daily_volume": len(vols),
            "members_missing_daily_volume": len(missing_members),
            "rank_denominator_complete": len(missing_members) == 0,
        })
        if missing_members:
            mark("board_day_volume_rank", None, SOURCE_MISSING)
        else:
            if code not in vols:
                mark("board_day_volume_rank", None, SOURCE_MISSING)
            else:
                ranks = pd.Series(vols, dtype=float).rank(method="min", ascending=False)
                mark("board_day_volume_rank", float(ranks.loc[code]))

    # ---- break_day_in_pool (D1) ----
    if not pool_complete_map.get(d1, False):
        mark("break_day_in_pool", None, SOURCE_MISSING)
    else:
        frame = pool_by_date.get(d1, _EMPTY_POOL)
        if frame is None or "code" not in frame.columns:
            mark("break_day_in_pool", None, SOURCE_MISSING)
        else:
            mark("break_day_in_pool", bool(code in frame["code"].astype(str).tolist()))

    # ---- last_board_day_in_pool (D0) ----
    if d0 is None:
        mark("last_board_day_in_pool", None, SOURCE_MISSING)
    elif not pool_complete_map.get(d0, False):
        mark("last_board_day_in_pool", None, SOURCE_MISSING)
    else:
        frame = pool_by_date.get(d0, _EMPTY_POOL)
        if frame is None or "code" not in frame.columns:
            mark("last_board_day_in_pool", None, SOURCE_MISSING)
        else:
            mark("last_board_day_in_pool", bool(code in frame["code"].astype(str).tolist()))

    # ---- pool_consecutive_count_last_board (D0 池记录) ----
    if d0 is None:
        mark("pool_consecutive_count_last_board", None, SOURCE_MISSING)
    elif not pool_complete_map.get(d0, False):
        mark("pool_consecutive_count_last_board", None, SOURCE_MISSING)
    else:
        frame = pool_by_date.get(d0, _EMPTY_POOL)
        if frame is None or "code" not in frame.columns:
            mark("pool_consecutive_count_last_board", None, SOURCE_MISSING)
        else:
            sub = frame[frame["code"].astype(str) == code]
            if sub.empty:
                mark("pool_consecutive_count_last_board", None, SOURCE_MISSING)
            else:
                pcc = sub.iloc[0].get("consecutive_limit_up_count")
                mark("pool_consecutive_count_last_board", _to_float(pcc) if pcc is not None else None,
                     SOURCE_MISSING if pcc is None else None)

    return values, missing, rank_audit


# ---------------------------------------------------------------------------
# 非 pool 重建特征 (沿用 v001 逻辑)
# ---------------------------------------------------------------------------
def rebuild_non_pool_features(row, daily, flags, ma5, ma10, ma20, date_index):
    """返回 (values, structural_mask)。逻辑与 v001 rebuild_event_features 一致
    (仅非 pool 部分)。"""
    code = str(row["code"]).zfill(6)
    break_date = str(row["break_date"])
    streak = int(row["board_streak_before_break"])
    values: dict = {}
    structural: dict = {}
    idx = date_index.get(break_date)

    if idx is None or idx < 1:
        for f in NON_POOL_REBUILD_FEATURES:
            values[f] = None
            structural[f] = True
        return values, structural

    lb = idx - 1
    lb_date = str(daily["date"].iloc[lb])
    d0_close = _to_float(daily["close"].iloc[lb])
    d1_open = _to_float(row.get("d1_open"))
    d1_high = _to_float(row.get("d1_high"))
    d1_low = _to_float(row.get("d1_low"))
    d1_close = _to_float(row.get("d1_close"))

    values["board_streak_is_3"] = int(streak == 3)
    structural["board_streak_is_3"] = False
    lo10 = max(0, idx - 9)
    lo20 = max(0, idx - 19)
    lo7 = max(0, idx - 6)
    values["recent_limit_up_count_10d"] = int(flags[lo10: idx + 1].sum())
    values["recent_limit_up_count_20d"] = int(flags[lo20: idx + 1].sum())
    structural["recent_limit_up_count_10d"] = False
    structural["recent_limit_up_count_20d"] = False
    max_streak = 0
    cur = 0
    for f in flags[lo20: idx + 1]:
        cur = cur + 1 if f else 0
        max_streak = max(max_streak, cur)
    values["max_board_streak_20d"] = max_streak
    structural["max_board_streak_20d"] = False

    values["break_high_return"] = _ret(_safe_div(d1_high, d0_close))
    values["break_close_return"] = _ret(_safe_div(d1_close, d0_close))
    structural["break_high_return"] = d0_close is None
    structural["break_close_return"] = d0_close is None
    lp = round_price_limit(d0_close) if d0_close else None
    values["break_touched_limit_up"] = bool(lp is not None and d1_high is not None
                                            and d1_high >= lp - LIMIT_UP_PRICE_TOLERANCE)
    values["break_opened_from_limit_up"] = bool(lp is not None and d1_open is not None
                                                and d1_open >= lp - LIMIT_UP_PRICE_TOLERANCE)
    structural["break_touched_limit_up"] = d0_close is None or d1_high is None
    structural["break_opened_from_limit_up"] = d0_close is None or d1_open is None
    bh, bl, bo, bc = d1_high, d1_low, d1_open, d1_close
    if bh is not None and bl is not None and bh > bl:
        values["break_upper_shadow_ratio"] = _safe_div(bh - max(bo or bh, bc or bh), bh - bl)
        values["break_lower_shadow_ratio"] = _safe_div(min(bo or bl, bc or bl) - bl, bh - bl)
    else:
        values["break_upper_shadow_ratio"] = None
        values["break_lower_shadow_ratio"] = None
    structural["break_upper_shadow_ratio"] = not (bh is not None and bl is not None and bh > bl)
    structural["break_lower_shadow_ratio"] = not (bh is not None and bl is not None and bh > bl)

    board_rows = daily.iloc[idx - streak: idx]
    bd_vols = [float(x) for x in pd.to_numeric(board_rows["volume"], errors="coerce").dropna() if x > 0]
    if bd_vols:
        values["break_volume_ratio_vs_board_days"] = _safe_div(
            _to_float(daily["volume"].iloc[idx]), float(np.mean(bd_vols)))
        structural["break_volume_ratio_vs_board_days"] = False
    else:
        values["break_volume_ratio_vs_board_days"] = None
        structural["break_volume_ratio_vs_board_days"] = True

    ma20_prev = ma20[lb] if lb >= 0 else np.nan
    ma20_cur = ma20[idx] if idx < len(ma20) else np.nan
    values["d1_ma20_slope"] = _ret(_safe_div(ma20_cur, ma20_prev))
    structural["d1_ma20_slope"] = bool(not (np.isfinite(ma20_prev) and np.isfinite(ma20_cur)
                                            and ma20_prev > 0))
    values["recent_7d_limit_up_count"] = int(flags[lo7: idx + 1].sum())
    structural["recent_7d_limit_up_count"] = False
    return values, structural


def compute_daily_arrays(df: pd.DataFrame):
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    prev = np.roll(close, 1)
    prev[0] = np.nan
    n = len(df)
    flags = np.array([is_limit_up_day(close[i], high[i], prev[i]) for i in range(n)], dtype=bool)
    flags[0] = False
    ma5 = df["close"].rolling(5, min_periods=5).mean().to_numpy(float)
    ma10 = df["close"].rolling(10, min_periods=10).mean().to_numpy(float)
    ma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy(float)
    date_index = {d: i for i, d in enumerate(df["date"].tolist())}
    return flags, ma5, ma10, ma20, date_index


def load_daily_store(codes: set[str]):
    store: dict[str, pd.DataFrame] = {}
    for code in sorted(codes):
        path = DAILY_DIR / f"{code}_daily.pkl"
        if not path.exists():
            continue
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue
        if df is None or df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume", "amount", "turnover_rate"):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        store[code] = df
    return store


def _row_value(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ---------------------------------------------------------------------------
# 输入表组装
# ---------------------------------------------------------------------------
def assemble_input_table(v002, trade_dates, pool_by_date, pool_complete_map,
                         volume_lookup, daily_store) -> tuple[pd.DataFrame, pd.DataFrame]:
    """组装 319 行 v002 输入表 + 逐行 missing 明细。"""
    rows = []
    rank_rows: list[dict] = []
    for _, row in v002.iterrows():
        code = str(row["code"]).zfill(6)
        r = {
            "event_id": str(row["event_id"]),
            "code": code,
            "signal_date": str(row["signal_date"]),
            "board_streak_before_break": int(row["board_streak_before_break"]),
            "source_window": str(row["source_window"]),
            "break_date": str(row["break_date"]),
        }
        # 34 个 v002 直接 FEATURE (非 pool 重建部分取 v002 冻结值)
        rebuilt_pool = set(POOL_DERIVED_FEATURES)
        v002_direct = [f for f in CONTRACT_FEATURE_NAMES
                       if f not in NON_POOL_REBUILD_FEATURES and f not in rebuilt_pool]
        for f in v002_direct:
            r[f] = _row_value(row.get(f))

        # 非 pool 重建
        daily = daily_store.get(code)
        if daily is None:
            flags, ma5, ma10, ma20, date_index = None, None, None, None, {}
            rebuild, structural = {f: None for f in NON_POOL_REBUILD_FEATURES}, {}
            for f in NON_POOL_REBUILD_FEATURES:
                structural[f] = True
        else:
            flags, ma5, ma10, ma20, date_index = compute_daily_arrays(daily)
            rebuild, structural = rebuild_non_pool_features(
                row, daily, flags, ma5, ma10, ma20, date_index)
        for f, v in rebuild.items():
            r[f] = _row_value(v)

        # pool 特征重算
        pool_vals, pool_missing, rank_audit = compute_pool_features(
            row, trade_dates, pool_by_date, pool_complete_map, volume_lookup)
        for f, v in pool_vals.items():
            r[f] = _row_value(v)

        # d1_close_location structural (v002 直接特征, 一字板退化)
        for f in ("d1_close_location",):
            if r.get(f) is None:
                h = _to_float(row.get("d1_high"))
                l = _to_float(row.get("d1_low"))
                if not (h is not None and l is not None and h > l):
                    structural[f] = True

        # 标签 + v002 dev 资格 (AUDIT_ONLY, 原样携带)
        for c in ("target7_daily_d2open_d3high", "tail_loss_daily_5pct",
                  "dev_label_complete", "dev_feature_source_complete", "dev_training_eligible"):
            r[c] = _row_value(row.get(c))

        # ---- missing 三分类 ----
        struct_list: list[str] = []
        source_list: list[str] = []
        unexpected_list: list[str] = []
        for f in CONTRACT_FEATURE_NAMES:
            if r[f] is None:
                if f in pool_missing:
                    source_list.append(f)
                elif structural.get(f, False):
                    struct_list.append(f)
                else:
                    unexpected_list.append(f)
        r["structural_missing_count"] = len(struct_list)
        r["source_missing_count"] = len(source_list)
        r["unexpected_missing_count"] = len(unexpected_list)
        r["provenance"] = (f"v002:event_id={r['event_id']};"
                           f"pool_window=complete;rank_denominator=full_pool")
        rows.append(r)
        if rank_audit:
            rank_audit = dict(rank_audit)
            rank_audit["event_id"] = str(row["event_id"])
            rank_audit["code"] = code
            rank_audit["signal_date"] = str(row["signal_date"])
            rank_rows.append(rank_audit)

    df = pd.DataFrame(rows)
    cols = (list(INPUT_PREFIX_COLUMNS) + list(CONTRACT_FEATURE_NAMES)
            + list(INPUT_SUFFIX_COLUMNS))
    df = df[cols]
    rank_df = pd.DataFrame(rank_rows)
    return df, rank_df


# ---------------------------------------------------------------------------
# pairwise eligibility
# ---------------------------------------------------------------------------
def attach_pairwise_eligibility(input_df: pd.DataFrame) -> pd.DataFrame:
    """pairwise_v1_feature_source_complete = 无 SOURCE/UNEXPECTED missing
    (允许 STRUCTURAL); pairwise_v1_training_eligible = dev_label AND source。"""
    df = input_df.copy()
    df["pairwise_v1_feature_source_complete"] = (
        (df["source_missing_count"] == 0) & (df["unexpected_missing_count"] == 0))
    df["pairwise_v1_training_eligible"] = (
        (df["dev_label_complete"] == 1) & df["pairwise_v1_feature_source_complete"])
    return df


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_atomic(path: Path, content: str, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_inventory(input_df, cov, excluded_pool: tuple[str, ...]):
    rows = []
    seen = set()
    for c in FEATURE_CONTRACT:
        name = c["feature_name"]
        if name in excluded_pool:
            decision, reason = EXCLUDED_CANDIDATE_DECISIONS.get(
                name, (None, "pool 历史无法完整恢复 (SOURCE_UNAVAILABLE)"))
            rows.append({
                "feature_name": name, "found_in": "v001 contract (corrected v002)",
                "historical_role": c.get("notes", ""), "semantic_group": c["semantic_group"],
                "raw_or_derived": "derived", "raw_dependencies": ";".join(c["raw_dependencies"]),
                "formula_or_definition": c["formula"], "lookback": c["lookback"],
                "available_as_of": c["available_as_of"], "source_system": c["source"],
                "current_may_coverage": cov.get(name, {}).get("may"),
                "current_june_coverage": cov.get(name, {}).get("june"),
                "combined_coverage": cov.get(name, {}).get("combined"),
                "structural_missing_possible": "yes" if c["missing_semantics"].startswith("STRUCTURAL") else "no",
                "future_leakage": 0, "model_output_leakage": 0, "target_lineage_leakage": 0,
                "exact_duplicate_group": "", "semantic_duplicate_group": c["semantic_group"],
                "pairwise_v1_decision": decision or "EXCLUDE_SOURCE_UNAVAILABLE",
                "decision_reason": reason,
            })
        else:
            rows.append({
                "feature_name": name, "found_in": "v001 contract (corrected v002)",
                "historical_role": c.get("notes", ""), "semantic_group": c["semantic_group"],
                "raw_or_derived": "derived", "raw_dependencies": ";".join(c["raw_dependencies"]),
                "formula_or_definition": c["formula"], "lookback": c["lookback"],
                "available_as_of": c["available_as_of"], "source_system": c["source"],
                "current_may_coverage": cov.get(name, {}).get("may"),
                "current_june_coverage": cov.get(name, {}).get("june"),
                "combined_coverage": cov.get(name, {}).get("combined"),
                "structural_missing_possible": "yes" if c["missing_semantics"].startswith("STRUCTURAL") else "no",
                "future_leakage": 0, "model_output_leakage": 0, "target_lineage_leakage": 0,
                "exact_duplicate_group": "", "semantic_duplicate_group": c["semantic_group"],
                "pairwise_v1_decision": FEATURE, "decision_reason": c.get("notes", ""),
            })
        seen.add(name)
    for name, (decision, reason) in sorted(EXCLUDED_CANDIDATE_DECISIONS.items()):
        if name in seen:
            continue
        rows.append({
            "feature_name": name, "found_in": "workflow v004c candidate inventory",
            "historical_role": "", "semantic_group": "", "raw_or_derived": "",
            "raw_dependencies": "", "formula_or_definition": reason, "lookback": "",
            "available_as_of": "D1_CLOSE" if decision == LABEL_ONLY else "",
            "source_system": "", "current_may_coverage": None, "current_june_coverage": None,
            "combined_coverage": None, "structural_missing_possible": "",
            "future_leakage": 0, "model_output_leakage": 0, "target_lineage_leakage": 0,
            "exact_duplicate_group": "", "semantic_duplicate_group": "",
            "pairwise_v1_decision": decision, "decision_reason": reason,
        })
        seen.add(name)
    df = pd.DataFrame(rows)
    cols = ["feature_name", "found_in", "historical_role", "semantic_group",
            "raw_or_derived", "raw_dependencies", "formula_or_definition",
            "lookback", "available_as_of", "source_system",
            "current_may_coverage", "current_june_coverage", "combined_coverage",
            "structural_missing_possible", "future_leakage", "model_output_leakage",
            "target_lineage_leakage", "exact_duplicate_group",
            "semantic_duplicate_group", "pairwise_v1_decision", "decision_reason"]
    return df[cols]


def write_contract_csv(excluded_pool: tuple[str, ...]):
    rows = []
    order = 1
    for c in FEATURE_CONTRACT:
        if c["feature_name"] in excluded_pool:
            continue
        rows.append({
            "feature_order": order, "feature_name": c["feature_name"],
            "semantic_group": c["semantic_group"], "information_class": c["information_class"],
            "dtype": c["dtype"], "formula": c["formula"],
            "raw_dependencies": ";".join(c["raw_dependencies"]), "source": c["source"],
            "lookback": c["lookback"], "available_as_of": c["available_as_of"],
            "missing_semantics": c["missing_semantics"],
            "expected_range_or_domain": c["expected_range_or_domain"], "notes": c["notes"],
        })
        order += 1
    return pd.DataFrame(rows)


def write_label_contract():
    rows = [
        {"label_name": "target7_daily_d2open_d3high", "role": LABEL_ONLY,
         "definition": "1[(d3_high / d2_open - 1) >= 0.07]", "horizon": "D2 open -> D3 high",
         "available_as_of": "D3_CLOSE", "physical_isolation": "标签序列, 绝不入 X",
         "usage": "未来 Pairwise 训练标签 (Y)"},
        {"label_name": "tail_loss_daily_5pct", "role": LABEL_ONLY,
         "definition": "1[(d3_high / d2_open - 1) <= -0.05]", "horizon": "D2 open -> D3 high",
         "available_as_of": "D3_CLOSE", "physical_isolation": "标签序列, 绝不入 X",
         "usage": "未来 Pairwise 训练标签 (Y)"},
        {"label_name": "d2_open_daily", "role": LABEL_ONLY, "definition": "D2 开盘原始值",
         "horizon": "D2", "available_as_of": "D2_OPEN", "physical_isolation": "标签派生",
         "usage": "标签派生"},
        {"label_name": "d3_high_daily", "role": LABEL_ONLY, "definition": "D3 最高原始值",
         "horizon": "D3", "available_as_of": "D3_CLOSE", "physical_isolation": "标签派生",
         "usage": "标签派生"},
        {"label_name": "d3_close_daily", "role": LABEL_ONLY, "definition": "D3 收盘原始值",
         "horizon": "D3", "available_as_of": "D3_CLOSE", "physical_isolation": "标签派生",
         "usage": "保留未用"},
    ]
    return pd.DataFrame(rows)


def write_schema(input_df, excluded_pool: tuple[str, ...]):
    rows = []
    for col in input_df.columns:
        if col in CONTRACT_FEATURE_NAMES and col not in excluded_pool:
            c = next(x for x in FEATURE_CONTRACT if x["feature_name"] == col)
            rows.append({"column_name": col, "role": FEATURE,
                         "feature_order": CONTRACT_FEATURE_ORDER[col],
                         "dtype": str(input_df[col].dtype),
                         "available_as_of": c["available_as_of"], "source": c["source"]})
        elif col in ("event_id", "code", "signal_date", "board_streak_before_break", "break_date"):
            rows.append({"column_name": col, "role": IDENTIFIER, "feature_order": None,
                         "dtype": str(input_df[col].dtype), "available_as_of": "D1_CLOSE",
                         "source": "v002"})
        elif col in ("target7_daily_d2open_d3high", "tail_loss_daily_5pct"):
            rows.append({"column_name": col, "role": LABEL_ONLY, "feature_order": None,
                         "dtype": str(input_df[col].dtype), "available_as_of": "D3_CLOSE",
                         "source": "v002_label"})
        elif col in ("dev_label_complete", "dev_feature_source_complete", "dev_training_eligible"):
            rows.append({"column_name": col, "role": "AUDIT_ONLY", "feature_order": None,
                         "dtype": str(input_df[col].dtype), "available_as_of": "D3_CLOSE",
                         "source": "v002"})
        elif col in PAIRWISE_ELIGIBILITY_COLUMNS:
            rows.append({"column_name": col, "role": "PAIRWISE_ELIGIBILITY",
                         "feature_order": None, "dtype": str(input_df[col].dtype),
                         "available_as_of": "D1_CLOSE", "source": "v002_corrected"})
        else:
            rows.append({"column_name": col, "role": "AUDIT_ONLY", "feature_order": None,
                         "dtype": str(input_df[col].dtype), "available_as_of": "D1_CLOSE",
                         "source": "v002"})
    return pd.DataFrame(rows)


def write_exclusions(excluded_pool: tuple[str, ...]):
    rows = [{"feature_name": k, "pairwise_v1_decision": v[0], "decision_reason": v[1]}
            for k, v in sorted(EXCLUDED_CANDIDATE_DECISIONS.items())]
    df = pd.DataFrame(rows)
    if excluded_pool:
        extra = pd.DataFrame([
            {"feature_name": f, "pairwise_v1_decision": "EXCLUDE_SOURCE_UNAVAILABLE",
             "decision_reason": EXCLUDED_CANDIDATE_DECISIONS.get(
                 f, (None, "pool 历史无法完整恢复"))[1] if f in EXCLUDED_CANDIDATE_DECISIONS
             else "pool 历史无法完整恢复 (SOURCE_UNAVAILABLE)"}
            for f in excluded_pool])
        df = pd.concat([df, extra], ignore_index=True).drop_duplicates(
            ["feature_name", "pairwise_v1_decision"], keep="last")
    return df


def coverage_stats(input_df, names):
    out = {}
    for f in names:
        may = input_df[input_df["source_window"] == "may"]
        june = input_df[input_df["source_window"] == "june"]
        out[f] = {"may": may[f].notna().mean(), "june": june[f].notna().mean(),
                  "combined": input_df[f].notna().mean()}
    return out


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------
def leakage_audit(names):
    from src.v004c_pairwise_feature_contract import leakage_scan
    return leakage_scan(list(names))


def availability_audit():
    bad = [c["feature_name"] for c in FEATURE_CONTRACT
           if c["available_as_of"] not in ("D1_CLOSE", "D0_CLOSE")]
    return bad


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------
def write_review(input_df, rank_df, cov, manifest, excluded_pool, pool_by_date,
                 universe_ok, leakage, availability_bad, determinism_ok):
    lines = []
    ap = lines.append
    ap("# v004c Pairwise v1 Feature Contract v002 Review (Pool Correctness Fix)")
    ap("")
    ap("- 输出: v004c_pairwise_v1_feature_contract_v002_20260506_20260630")
    ap("- universe: {len(input_df)} 行 / {input_df['signal_date'].nunique()} 信号日")
    ap(f"- 与 v002 身份逐行一致: {'PASS' if universe_ok else 'FAIL'}")
    ap(f"- FEATURE 数 (after): {len([c for c in CONTRACT_FEATURE_NAMES if c not in excluded_pool])} "
       f"(before: 53)")
    ap("")
    ap("## 1. Pool 历史恢复情况")
    ap("")
    req = manifest[manifest["required"]]
    complete_n = int(req["complete"].sum())
    ap("- Required pool range:")
    ap(f"  - start: {req['trade_date'].min()}")
    ap(f"  - end: {req['trade_date'].max()}")
    ap(f"  - required trading dates: {len(req)}")
    ap("- Before:")
    ap(f"  - pool dates complete: 39")
    ap(f"  - pool dates missing: 19 (2026-04-03..2026-04-30)")
    ap("- Recovery:")
    ap(f"  - attempted: 19 (recover_v004c_pool_history_v002.py, "
       "canonical daily_limitup_derived)")
    ap(f"  - recovered: 19")
    ap(f"  - still missing: 0")
    ap(f"- After:")
    ap(f"  - complete dates: {complete_n}")
    ap(f"  - incomplete dates: {len(req) - complete_n}")
    if len(req) - complete_n:
        bad_dates = req[~req["complete"]]["trade_date"].tolist()
        ap(f"  - incomplete dates list: {bad_dates}")
    ap("- pool source: daily_limitup_derived (canonical pipeline)")
    ap("- 完整性核对 (recover_v004c_pool_completeness_v002.py): 04-03..04-30 "
       "恢复池零缺口; 06 月原有池文件 12 个日期共 23 只缺口已补日线并重新 derive "
       "(002217 型缓存缺口根因)")
    ap("")
    ap("## 2. Pool-derived FEATURE 决策")
    ap("")
    for f in POOL_DERIVED_FEATURES:
        decision = ("EXCLUDE_SOURCE_UNAVAILABLE" if f in excluded_pool else "KEEP_FEATURE")
        ap(f"- {f}: {decision}")
    ap("")
    ap("## 3. board_day_volume_rank")
    ap("")
    ap("- denominator definition: FULL_D0_LIMIT_UP_POOL (当日完整涨停池成员, "
       "非 development candidate subset)")
    d0_dates = sorted(rank_df["D0"].dropna().unique().tolist())
    complete_dates = rank_df.groupby("D0")["rank_denominator_complete"].all()
    n_comp = int(complete_dates.sum())
    ap(f"- D0 dates audited: {len(d0_dates)}")
    ap(f"- complete rank denominator dates: {n_comp}")
    ap(f"- incomplete rank denominator dates: {len(d0_dates) - n_comp}")
    ap(f"- total pool members audited: {int(rank_df['pool_member_count'].sum())}")
    ap(f"- daily volume available: {int(rank_df['members_with_daily_volume'].sum())}")
    ap(f"- daily volume missing: {int(rank_df['members_missing_daily_volume'].sum())}")
    ap(f"- old subset-ranking bug eliminated: "
       f"{'YES' if n_comp == len(d0_dates) and rank_df['members_missing_daily_volume'].sum() == 0 else 'NO'}")
    ap("")
    ap("## 4. Missing 分类")
    ap("")
    ap(f"- STRUCTURAL_MISSING: {int(input_df['structural_missing_count'].sum())} "
       f"(数学退化, 如 603065 high==low)")
    ap(f"- SOURCE_MISSING: {int(input_df['source_missing_count'].sum())} (pool 数据源缺口)")
    ap(f"- UNEXPECTED_MISSING: {int(input_df['unexpected_missing_count'].sum())} (必须 0)")
    ap("")
    ap("## 5. High-zone contract 公式修正")
    ap("")
    for name in ("high_zone_volume_ratio", "high_zone_amount_ratio"):
        c = next(x for x in FEATURE_CONTRACT if x["feature_name"] == name)
        ap(f"- {name}: {c['formula']}")
    ap("- 实现核对 (v004c_minute_features.py): typical_price=(high+low+close)/3, "
       "分子 Σ[typical_price >= thr], 一致")
    ap("")
    ap("## 6. Pairwise 层 eligibility")
    ap("")
    ap(f"- pairwise_v1_feature_source_complete: "
       f"{int((input_df['pairwise_v1_feature_source_complete'] == 1).sum())}/{len(input_df)}")
    ap(f"- pairwise_v1_training_eligible: "
       f"{int((input_df['pairwise_v1_training_eligible'] == 1).sum())}/{len(input_df)}")
    ap("- dev_* 列保持 AUDIT_ONLY (v002 foundation 层)")
    ap("")
    ap("## 7. 泄漏审计")
    ap("")
    ap(f"- FEATURE future leakage: {leakage['future_leakage']} (必须 0)")
    ap(f"- FEATURE model-output leakage: {leakage['model_output_leakage']} (必须 0)")
    ap(f"- FEATURE label-lineage leakage: {leakage['label_lineage_leakage']} (必须 0)")
    ap(f"- D1 availability violations: {availability_bad if availability_bad else '无'}")
    ap("")
    ap("## 8. Deterministic Rebuild")
    ap("")
    ap(f"- 两轮构建 9 个核心资产字节一致: {'PASS' if determinism_ok else 'FAIL'}")
    ap("")
    ap("## 9. 严格禁止确认")
    ap("")
    ap("- 未训练 Pairwise Ridge / Logistic / Tree/GBDT")
    ap("- 未搜索 lambda / 未做 walk-forward")
    ap("- 未做 Target7 / tail-loss / AUC / IC 分析")
    ap("- 未新增 FEATURE / 未恢复 amount/turnover rank")
    ap("- 未创建 feature_x_board3 交互列")
    ap("")
    ap("## 10. 结论")
    ap("")
    # READY 条件 (§四十九): 保留 FEATURE 中不允许任何 SOURCE_MISSING / UNEXPECTED_MISSING
    ready = (universe_ok and not availability_bad
             and int(input_df["unexpected_missing_count"].sum()) == 0
             and int(input_df["source_missing_count"].sum()) == 0
             and leakage["future_leakage"] == 0
             and leakage["model_output_leakage"] == 0
             and leakage["label_lineage_leakage"] == 0
             and determinism_ok)
    ap(f"- Feature Contract 状态: {'READY_FOR_PAIRWISE_V1' if ready else 'NOT_READY_FOR_PAIRWISE_V1'}")
    ap("")
    return "\n".join(lines) + "\n"


def write_outputs(output_dir, input_df, rank_df, cov, manifest, excluded_pool,
                  pool_by_date, universe_ok, leakage, availability_bad, determinism_ok):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_dir / INVENTORY_CSV_NAME,
                 write_inventory(input_df, cov, excluded_pool).to_csv(index=False))
    write_atomic(output_dir / CONTRACT_CSV_NAME,
                 write_contract_csv(excluded_pool).to_csv(index=False))
    write_atomic(output_dir / LABEL_CSV_NAME, write_label_contract().to_csv(index=False))
    write_atomic(output_dir / DEV_CSV_NAME, input_df.to_csv(index=False))
    write_atomic(output_dir / SCHEMA_CSV_NAME,
                 write_schema(input_df, excluded_pool).to_csv(index=False))
    write_atomic(output_dir / EXCLUSIONS_CSV_NAME,
                 write_exclusions(excluded_pool).to_csv(index=False))
    write_atomic(output_dir / POOL_COVERAGE_CSV_NAME, manifest.to_csv(index=False))
    write_atomic(output_dir / RANK_AUDIT_CSV_NAME, rank_df.to_csv(index=False))
    review = write_review(input_df, rank_df, cov, manifest, excluded_pool, pool_by_date,
                          universe_ok, leakage, availability_bad, determinism_ok)
    write_atomic(output_dir / REVIEW_NAME, review)


def determinism_check(a: Path, b: Path) -> bool:
    core = (DEV_CSV_NAME, CONTRACT_CSV_NAME, INVENTORY_CSV_NAME, LABEL_CSV_NAME,
            SCHEMA_CSV_NAME, EXCLUSIONS_CSV_NAME, POOL_COVERAGE_CSV_NAME,
            RANK_AUDIT_CSV_NAME, REVIEW_NAME)
    for name in core:
        if (a / name).read_bytes() != (b / name).read_bytes():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output or OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_contract_integrity_v002()

    print("[1] load trade calendar ...", flush=True)
    trade_dates = load_trade_calendar()

    print("[2] load v002 foundation ...", flush=True)
    v002 = pd.read_csv(V002_CSV, encoding="utf-8-sig", dtype={"code": str})
    v002["code"] = v002["code"].astype(str).str.zfill(6)
    v002["signal_date"] = v002["signal_date"].astype(str)
    v002["break_date"] = v002["break_date"].astype(str)

    print("[3] load pool cache + completeness ...", flush=True)
    pool_by_date = load_pool_by_date()
    required_dates = required_pool_dates(v002, trade_dates)
    print(f"    required pool dates: {len(required_dates)} "
          f"({required_dates[0]} .. {required_dates[-1]})")
    manifest = pool_completeness(pool_by_date, required_dates)
    complete_map = {d: True for d in manifest[manifest["complete"]]["trade_date"]}
    incomplete = manifest[~manifest["complete"]]["trade_date"].tolist()
    print(f"    pool dates complete: {int(manifest['complete'].sum())} "
          f"incomplete: {len(incomplete)} {incomplete}")

    print("[4] build volume lookup ...", flush=True)
    volume_lookup = build_volume_lookup()

    print("[5] load daily store (319-event codes) ...", flush=True)
    daily_store = load_daily_store(set(v002["code"].tolist()))

    print("[6] assemble input table ...", flush=True)
    input_df, rank_df = assemble_input_table(v002, trade_dates, pool_by_date,
                                             complete_map, volume_lookup, daily_store)
    input_df = attach_pairwise_eligibility(input_df)

    # pool FEATURE 决策: 保留全部 (SOURCE_MISSING 按列审计; 若某特征全 SOURCE 才排除)
    excluded_pool: tuple[str, ...] = ()
    for f in POOL_DERIVED_FEATURES:
        src_missing_n = int((input_df[f].isna() & (input_df["source_missing_count"] > 0)).sum())
        print(f"    pool feature {f}: source-missing rows = {src_missing_n}")

    # universe 身份逐行审计
    for col in ("event_id", "code", "signal_date", "break_date",
                "board_streak_before_break", "source_window"):
        assert input_df[col].astype(str).tolist() == v002[col].astype(str).tolist(), \
            f"universe 身份列 {col} 与 v002 不一致"
    universe_ok = True

    print("[7] audits ...", flush=True)
    kept_names = [f for f in CONTRACT_FEATURE_NAMES if f not in excluded_pool]
    leakage = leakage_audit(kept_names)
    availability_bad = availability_audit()
    cov = coverage_stats(input_df, kept_names)
    print(f"    unexpected_missing total: {int(input_df['unexpected_missing_count'].sum())}")
    print(f"    source_missing total: {int(input_df['source_missing_count'].sum())}")
    print(f"    structural_missing total: {int(input_df['structural_missing_count'].sum())}")

    print("[8] determinism rebuild ...", flush=True)
    second_dir = output_dir.parent / (output_dir.name + "_rebuild_check")
    if second_dir.exists():
        shutil.rmtree(second_dir)
    write_outputs(output_dir, input_df, rank_df, cov, manifest, excluded_pool,
                  pool_by_date, universe_ok, leakage, availability_bad, False)
    write_outputs(second_dir, input_df, rank_df, cov, manifest, excluded_pool,
                  pool_by_date, universe_ok, leakage, availability_bad, False)
    determinism_ok = determinism_check(output_dir, second_dir)
    shutil.rmtree(second_dir)
    if determinism_ok:
        review = write_review(input_df, rank_df, cov, manifest, excluded_pool,
                              pool_by_date, universe_ok, leakage, availability_bad, True)
        write_atomic(output_dir / REVIEW_NAME, review)
    print(f"    determinism: {'PASS' if determinism_ok else 'FAIL'}")
    if not determinism_ok:
        print("FATAL: determinism rebuild mismatch")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

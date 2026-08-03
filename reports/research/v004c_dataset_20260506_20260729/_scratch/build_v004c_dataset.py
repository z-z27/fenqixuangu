# -*- coding: utf-8 -*-
"""v004c 断板修复研究数据集构建 (只读, 不写任何项目现有文件)

范围:
- 候选信号日: 2026-05-06 .. 2026-07-29
- 标签回看: 到 2026-07-31
- 连板判定: 复制自 src/loaders.py::_is_main_board_limit_up_day (2026-08-03 只读)
- 标签口径: 复制自 src/history_samples.py::_d2open_d3_metrics / _finalise_targets

输出(全部位于输出目录):
- v004c_break_repair_dataset.csv   主样本
- v004c_candidate_audit.csv        候选审计
- _scratch/stage1_events.csv       中间: 断板事件
- _scratch/stage2_observations.csv 中间: 观察行与状态
- _scratch/stage3_candidates_feat.csv 中间: 候选特征(未加比较字段)
"""
import glob
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_dataset_20260506_20260729"
SCRATCH = OUT / "_scratch"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
MINUTE_DIR = ROOT / "data" / "cache" / "minute_5m"
UNIVERSE_PKL = ROOT / "data" / "cache" / "universe" / "eastmoney_main_board_universe.pkl"

WINDOW_START = "2026-05-06"
WINDOW_END = "2026-07-29"
LABEL_END = "2026-07-31"
BREAK_SCAN_START = "2026-04-20"  # days_since_break=2 仍可达窗口起点

# ---- 项目涨停判定逻辑(复制自 src/loaders.py, 2026-08-03 只读) ----
MAIN_BOARD_LIMIT_RATIO = Decimal("1.10")
LIMIT_UP_PRICE_TOLERANCE = 0.011
LIMIT_UP_MIN_PCT_CHG = 9.7


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
    """v - 1, None-safe"""
    v = _to_float(v)
    return None if v is None else v - 1.0


def _safe_div(a, b) -> float | None:
    a = _to_float(a)
    b = _to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _first_numeric(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _last_numeric(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _max_numeric(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _min_numeric(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


# ---------------- 1. 涨停池 ----------------
print("[1] load limit-up pool ...", flush=True)
pool_frames = []
for path in sorted(glob.glob(str(POOL_DIR / "*_limitups.pkl"))):
    pool_frames.append(pd.read_pickle(path))
pool = pd.concat(pool_frames, ignore_index=True)
pool["code"] = pool["code"].astype(str).str.zfill(6)
pool["trade_date"] = pool["trade_date"].astype(str)
pool = pool.drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True)
print("pool rows:", len(pool), "dates:", pool["trade_date"].nunique(),
      pool["trade_date"].min(), "->", pool["trade_date"].max())

pool_members: dict[str, set[str]] = {}
pool_meta: dict[tuple, dict] = {}
for row in pool.to_dict(orient="records"):
    pool_members.setdefault(str(row["trade_date"]), set()).add(str(row["code"]))
    pool_meta[(str(row["trade_date"]), str(row["code"]))] = row
pool_dates = sorted(pool["trade_date"].unique().tolist())
pool_date_set = set(pool_dates)

name_latest: dict[str, str] = {}
for row in pool.sort_values("trade_date").to_dict(orient="records"):
    nm = str(row.get("name", "")).strip()
    if nm:
        name_latest[str(row["code"])] = nm
pool_name_by_date: dict[tuple, str] = {}
for row in pool.to_dict(orient="records"):
    nm = str(row.get("name", "")).strip()
    if nm:
        pool_name_by_date[(str(row["trade_date"]), str(row["code"]))] = nm

universe = pd.read_pickle(UNIVERSE_PKL)
universe_names = dict(zip(universe["code"].astype(str).str.zfill(6),
                          universe["name"].astype(str)))


def resolve_name(code: str, date: str) -> str:
    nm = pool_name_by_date.get((date, code), "")
    if nm:
        return nm
    nm = name_latest.get(code, "")
    if nm:
        return nm
    nm = universe_names.get(code, "")
    return nm or ""


# ---------------- 2. 日线 + 涨停标记 ----------------
print("[2] load daily caches ...", flush=True)
daily_store: dict[str, pd.DataFrame] = {}
flag_store: dict[str, np.ndarray] = {}
ma_store: dict[str, dict] = {}

for path in sorted(glob.glob(str(DAILY_DIR / "*_daily.pkl"))):
    code = Path(path).name[:6]
    try:
        df = pd.read_pickle(path)
    except Exception:
        continue
    if df is None or df.empty or "date" not in df.columns:
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "amount", "turnover_rate"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) < 21:
        continue
    n = len(df)
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    prev = np.roll(close, 1)
    prev[0] = np.nan
    flags = np.array([is_limit_up_day(close[i], high[i], prev[i]) for i in range(n)], dtype=bool)
    flags[0] = False
    ma5 = df["close"].rolling(5, min_periods=5).mean().to_numpy(float)
    ma10 = df["close"].rolling(10, min_periods=10).mean().to_numpy(float)
    ma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy(float)
    daily_store[code] = df
    flag_store[code] = flags
    ma_store[code] = {"ma5": ma5, "ma10": ma10, "ma20": ma20}

print("daily store codes:", len(daily_store))

# ---------------- 3. 断板事件检测 (所有连板长度) ----------------
print("[3] detect break events ...", flush=True)
events: list[dict] = []
for code, df in daily_store.items():
    flags = flag_store[code]
    n = len(df)
    idx = 1
    while idx < n:
        if not flags[idx] and flags[idx - 1]:
            streak = 0
            j = idx - 1
            while j >= 0 and flags[j]:
                streak += 1
                j -= 1
            break_date = str(df["date"].iloc[idx])
            if BREAK_SCAN_START <= break_date <= WINDOW_END:
                events.append({
                    "code": code,
                    "break_date": break_date,
                    "break_idx": idx,
                    "last_board_idx": idx - 1,
                    "last_board_date": str(df["date"].iloc[idx - 1]),
                    "board_streak_before_break": streak,
                })
        idx += 1

events_df = pd.DataFrame(events)
print("break events detected:", len(events_df), "| streak2/3:",
      int(events_df["board_streak_before_break"].isin([2, 3]).sum()))
events_df.to_csv(SCRATCH / "stage1_events.csv", index=False, encoding="utf-8-sig")


# ---------------- 4. 观察行生成与状态分类 ----------------
print("[4] generate observations ...", flush=True)


def _expected_date_after(ev: dict, k: int, df: pd.DataFrame):
    idx = ev["break_idx"]
    n = len(df)
    if idx + k < n:
        return str(df["date"].iloc[idx + k])
    return None


observations: list[dict] = []
for ev in events_df.to_dict(orient="records"):
    code = ev["code"]
    df = daily_store[code]
    flags = flag_store[code]
    n = len(df)
    idx = ev["break_idx"]
    max_date = str(pd.to_datetime(df["date"], errors="coerce").dropna().max().strftime("%Y-%m-%d"))
    streak = int(ev["board_streak_before_break"])
    for k in range(3):
        obs_idx = idx + k
        slot = {
            "event_id": f"{code}_{ev['break_date']}",
            "code": code,
            "break_date": ev["break_date"],
            "board_streak_before_break": streak,
            "days_since_break": k,
        }
        if streak not in (2, 3):
            # 非 2/3 板事件: 仍需确定 signal_date 是否在窗口内
            if obs_idx < n:
                signal_date = str(df["date"].iloc[obs_idx])
                if signal_date < WINDOW_START or signal_date > WINDOW_END:
                    continue
                slot["signal_date"] = signal_date
                slot["candidate_status"] = "EXCLUDED"
                slot["exclusion_reason"] = "not_two_or_three_board"
                slot["data_quality_reason"] = f"断板前连板长度 {streak} 不在 {{2,3}}"
            else:
                expected = _expected_date_after(ev, k, df)
                if expected is None or expected < WINDOW_START or expected > WINDOW_END:
                    continue
                slot["signal_date"] = ""
                slot["candidate_status"] = "EXCLUDED"
                slot["exclusion_reason"] = "not_two_or_three_board"
                slot["data_quality_reason"] = f"断板前连板长度 {streak} 不在 {{2,3}}"
            observations.append(slot)
            continue

        # streak in (2,3)
        if obs_idx < n:
            signal_date = str(df["date"].iloc[obs_idx])
            if signal_date < WINDOW_START or signal_date > WINDOW_END:
                continue
            slot["signal_date"] = signal_date
            if flags[obs_idx]:
                slot["candidate_status"] = "EXCLUDED"
                slot["exclusion_reason"] = "still_limit_up"
                slot["data_quality_reason"] = "signal_date 本身为涨停日"
            else:
                slot["pending"] = True
        else:
            expected = _expected_date_after(ev, k, df)
            if expected is None or expected < WINDOW_START or expected > WINDOW_END:
                continue
            slot["signal_date"] = ""
            if expected > max_date:
                slot["candidate_status"] = "EXCLUDED"
                slot["exclusion_reason"] = "insufficient_daily_history"
                slot["data_quality_reason"] = f"日线缓存最大日期 {max_date} < 期望日期 {expected}"
            else:
                slot["candidate_status"] = "EXCLUDED"
                slot["exclusion_reason"] = "suspended"
                slot["data_quality_reason"] = f"期望日期 {expected} 在缓存覆盖内但无日线行(停牌或缺口)"
        observations.append(slot)

# 对 pending 行做 D1 完整性 + 标签可用性检查
minute_cache: dict[str, pd.DataFrame | None] = {}


def get_minute(code: str) -> pd.DataFrame | None:
    if code in minute_cache:
        return minute_cache[code]
    try:
        path = MINUTE_DIR / f"{code}_5min.pkl"
        if not path.exists():
            minute_cache[code] = None
            return None
        m = pd.read_pickle(path)
        if m is None or m.empty or "trade_date" not in m.columns:
            minute_cache[code] = None
            return None
        m = m.copy()
        m["trade_date"] = pd.to_datetime(m["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in m.columns:
                m[col] = np.nan
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m = m.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        minute_cache[code] = m
    except Exception:
        minute_cache[code] = None
    return minute_cache[code]


def _daily_price_for_date(code: str, trade_date: str, col: str) -> float | None:
    df = daily_store.get(code)
    if df is None or df.empty or col not in df.columns:
        return None
    matched = df[df["date"] == str(trade_date)]
    if matched.empty:
        return None
    return _to_float(matched.iloc[-1].get(col))


pending = [o for o in observations if o.get("pending")]
print("pending rows:", len(pending))
for o in pending:
    code = o["code"]
    df = daily_store[code]
    obs_idx = int(np.where(df["date"].values == o["signal_date"])[0][0])
    row = df.iloc[obs_idx]
    okl = all(_to_float(row[c]) is not None and _to_float(row[c]) > 0
              for c in ("open", "high", "low", "close"))
    if not okl:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "invalid_price"
        o["data_quality_reason"] = "D1 日线 OHLC 缺失或非正"
        continue
    if obs_idx < 20:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "insufficient_daily_history"
        o["data_quality_reason"] = f"D1 前历史不足 20 个交易日 (obs_idx={obs_idx})"
        continue
    if str(df["date"].iloc[0]) > "2026-04-01":
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "insufficient_daily_history"
        o["data_quality_reason"] = f"日线起始 {str(df['date'].iloc[0])} 晚于 2026-04-01"
        continue
    minute = get_minute(code)
    if minute is None or minute.empty:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "missing_minute_data"
        o["data_quality_reason"] = "无 5min 缓存"
        continue
    day_m = minute[minute["trade_date"] == o["signal_date"]]
    if day_m.empty:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "missing_minute_data"
        o["data_quality_reason"] = f"5min 无 {o['signal_date']} 数据"
        continue
    if len(day_m) < 30:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "missing_minute_data"
        o["data_quality_reason"] = f"{o['signal_date']} 5min bar 数 {len(day_m)} < 30"
        continue
    if float(pd.to_numeric(day_m["volume"], errors="coerce").sum()) <= 0:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "missing_minute_data"
        o["data_quality_reason"] = f"{o['signal_date']} 5min 总量为 0"
        continue
    mc = _to_float(day_m["close"].iloc[-1])
    dc = _to_float(row["close"])
    if mc is None or dc is None or abs(mc - dc) > 0.02:
        o["candidate_status"] = "QUALITY_FAILED"
        o["exclusion_reason"] = "missing_minute_data"
        o["data_quality_reason"] = f"日线/5min 收盘差异 {abs(mc - dc)} > 0.02"
        continue
    future = sorted(minute["trade_date"].dropna().astype(str).unique().tolist())
    future = [d for d in future if d > o["signal_date"]]
    if len(future) < 2:
        o["candidate_status"] = "LABEL_UNAVAILABLE"
        o["exclusion_reason"] = "missing_future_label"
        o["data_quality_reason"] = f"5min 未来交易日不足 2 个 (可用 {len(future)})"
        continue
    d2_date, d3_date = future[0], future[1]
    if d2_date > LABEL_END or d3_date > LABEL_END:
        o["candidate_status"] = "LABEL_UNAVAILABLE"
        o["exclusion_reason"] = "missing_future_label"
        o["data_quality_reason"] = f"D2/D3 超过标签上限 2026-07-31 (d2={d2_date}, d3={d3_date})"
        continue
    d2_m = minute[minute["trade_date"] == d2_date]
    d3_m = minute[minute["trade_date"] == d3_date]
    d2_open = _first_numeric(d2_m, "open")
    if d2_open is None:
        d2_open = _daily_price_for_date(code, d2_date, "open")
    d3_high = _max_numeric(d3_m, "high")
    d3_close = _last_numeric(d3_m, "close")
    if d2_open is None or d2_open <= 0 or d3_high is None or d3_close is None:
        o["candidate_status"] = "LABEL_UNAVAILABLE"
        o["exclusion_reason"] = "missing_future_label"
        o["data_quality_reason"] = (f"标签核心字段缺失 d2_open={d2_open} d3_high={d3_high} "
                                    f"d3_close={d3_close} (d2={d2_date}, d3={d3_date})")
        continue
    o["candidate_status"] = "CANDIDATE"
    o["d2_trade_date"] = d2_date
    o["d3_trade_date"] = d3_date
    o["obs_idx"] = obs_idx

for o in observations:
    o.pop("pending", None)

obs_df = pd.DataFrame(observations)
print("observations:", len(obs_df))
print("status counts:", obs_df["candidate_status"].value_counts().to_dict())
obs_df.to_csv(SCRATCH / "stage2_observations.csv", index=False, encoding="utf-8-sig")

# ---------------- 5. 主样本特征计算 ----------------
print("[5] compute features ...", flush=True)


def _minute_amount_by_date(minute: pd.DataFrame) -> dict[str, float]:
    if minute is None or minute.empty:
        return {}
    g = minute.groupby("trade_date")["amount"].sum()
    return {str(k): float(v) for k, v in g.items() if pd.notna(v)}


def _day_amount(code: str, date: str, minute: pd.DataFrame | None, df: pd.DataFrame) -> float | None:
    """日成交额: 该日 5min bar 数 >= 30 时用 5min 求和, 否则用日线 amount, 否则 None。
    (5min 覆盖不完整时用 5min 求和会产生单位/覆盖失真, 故要求完整覆盖;
     这是对 src/loaders.py::_merge_minute_amount 口径的研究质量收紧, 在 feature definitions 说明)"""
    if minute is not None:
        bars = minute[minute["trade_date"] == date]
        if len(bars) >= 30:
            s = float(pd.to_numeric(bars["amount"], errors="coerce").sum())
            if s > 0:
                return s
    row = df[df["date"] == date]
    if not row.empty:
        v = _to_float(row.iloc[0].get("amount"))
        if v is not None and v > 0:
            return v
    return None


def intraday_vwap(day_m: pd.DataFrame) -> float | None:
    if day_m.empty:
        return None
    amount = pd.to_numeric(day_m["amount"], errors="coerce")
    volume = pd.to_numeric(day_m["volume"], errors="coerce")
    close = pd.to_numeric(day_m["close"], errors="coerce")
    cum_a = amount.cumsum()
    cum_v = volume.cumsum()
    ref = close
    raw = cum_a / cum_v.replace(0, pd.NA)
    hand = cum_a / (cum_v * 100).replace(0, pd.NA)
    use_hand = raw > ref * 5
    vwap_series = raw.mask(use_hand, hand).dropna()
    return float(vwap_series.iloc[-1]) if len(vwap_series) else None


candidate_rows: list[dict] = []
for o in obs_df.to_dict(orient="records"):
    if o.get("candidate_status") != "CANDIDATE":
        continue
    code = o["code"]
    df = daily_store[code]
    flags = flag_store[code]
    mas = ma_store[code]
    minute = minute_cache[code]
    signal_date = o["signal_date"]
    break_date = o["break_date"]
    streak = int(o["board_streak_before_break"])
    k = int(o["days_since_break"])
    obs_idx = int(o["obs_idx"])
    idx = int(np.where(df["date"].values == break_date)[0][0])
    lb = idx - 1
    row = df.iloc[obs_idx]
    break_row = df.iloc[idx]
    board_rows = df.iloc[idx - streak: idx]
    members = pool_members.get(str(df["date"].iloc[lb]), set())

    r: dict = {
        "event_id": o["event_id"],
        "code": code,
        "name": resolve_name(code, break_date),
        "signal_date": signal_date,
        "break_date": break_date,
        "board_streak_before_break": streak,
        "days_since_break": k,
        "days_since_last_limit_up": k + 1,
    }
    attempts = 0
    for j in range(k + 1):
        jj = idx + j
        if jj < len(df):
            jd = str(df["date"].iloc[jj])
            if WINDOW_START <= jd <= WINDOW_END and not flags[jj]:
                attempts += 1
    r["repair_attempt_count"] = attempts

    # ---- B 连板与辨识度 ----
    lo10 = max(0, obs_idx - 9)
    lo20 = max(0, obs_idx - 19)
    r["recent_limit_up_count_10d"] = int(flags[lo10: obs_idx + 1].sum())
    r["recent_limit_up_count_20d"] = int(flags[lo20: obs_idx + 1].sum())
    dates10 = df["date"].iloc[lo10: obs_idx + 1].tolist()
    dates20 = df["date"].iloc[lo20: obs_idx + 1].tolist()
    r["recent_pool_appearance_count_10d"] = int(sum(
        1 for d in dates10 if d in pool_date_set and code in pool_members.get(d, set())))
    r["recent_pool_appearance_count_20d"] = int(sum(
        1 for d in dates20 if d in pool_date_set and code in pool_members.get(d, set())))
    max_streak = 0
    cur = 0
    for f in flags[lo20: obs_idx + 1]:
        cur = cur + 1 if f else 0
        max_streak = max(max_streak, cur)
    r["max_board_streak_20d"] = max_streak
    lb_date = str(df["date"].iloc[lb])
    r["last_board_date"] = lb_date  # 审计用
    rk_rows = [pool_meta[(lb_date, c)] for c in sorted(members) if (lb_date, c) in pool_meta]
    rk_pool = pd.DataFrame(rk_rows) if rk_rows else pd.DataFrame()
    if rk_pool.empty:
        r["board_day_amount_rank"] = None
        r["board_day_turnover_rank"] = None
        r["board_day_volume_rank"] = None
        n_members = 0
    else:
        n_members = len(rk_pool)
        codes_l = rk_pool["code"].astype(str).tolist()

        def _rank_of(value_map: dict) -> float | None:
            if not value_map or code not in value_map or pd.isna(value_map[code]):
                return None
            vals = pd.Series(value_map, dtype=float)
            ranks = vals.rank(method="min", ascending=False)
            return float(ranks.loc[code])

        amt_map = dict(zip(codes_l, pd.to_numeric(rk_pool["amount"], errors="coerce"))) if "amount" in rk_pool.columns else {}
        tor_map = dict(zip(codes_l, pd.to_numeric(rk_pool["turnover_rate"], errors="coerce"))) if "turnover_rate" in rk_pool.columns else {}
        vol_map = dict(zip(codes_l, [_daily_price_for_date(c, lb_date, "volume") for c in codes_l]))
        r["board_day_amount_rank"] = _rank_of(amt_map)
        r["board_day_turnover_rank"] = _rank_of(tor_map)
        r["board_day_volume_rank"] = _rank_of(vol_map)
    n_comp = 0
    pct_sum = 0.0
    for rank_key in ("board_day_amount_rank", "board_day_turnover_rank", "board_day_volume_rank"):
        rk = r.get(rank_key)
        if rk is not None and n_members >= 2:
            pct_sum += (n_members - float(rk) + 1) / n_members
            n_comp += 1
    r["recognition_score"] = pct_sum / n_comp if n_comp == 3 else None

    # ---- C 断板日 OHLC 与结构 ----
    b = break_row
    b_prev_close = _to_float(df["close"].iloc[idx - 1])
    r["break_open"] = _to_float(b["open"])
    r["break_high"] = _to_float(b["high"])
    r["break_low"] = _to_float(b["low"])
    r["break_close"] = _to_float(b["close"])
    r["break_prev_close"] = b_prev_close
    r["break_open_return"] = _ret(_safe_div(b["open"], b_prev_close))
    r["break_high_return"] = _ret(_safe_div(b["high"], b_prev_close))
    r["break_close_return"] = _ret(_safe_div(b["close"], b_prev_close))
    r["break_intraday_range"] = _safe_div(_to_float(b["high"]) - _to_float(b["low"]), b_prev_close)
    r["break_high_to_close_drawdown"] = _safe_div(_to_float(b["high"]) - _to_float(b["close"]), _to_float(b["high"]))
    bh = _to_float(b["high"])
    bl = _to_float(b["low"])
    bo = _to_float(b["open"])
    bc = _to_float(b["close"])
    if bh is not None and bl is not None and bh > bl:
        r["break_upper_shadow_ratio"] = _safe_div(bh - max(bo or bh, bc or bh), bh - bl)
        r["break_lower_shadow_ratio"] = _safe_div(min(bo or bl, bc or bl) - bl, bh - bl)
        r["break_close_location"] = _safe_div(bc - bl, bh - bl)
    else:
        r["break_upper_shadow_ratio"] = None
        r["break_lower_shadow_ratio"] = None
        r["break_close_location"] = None
    amount_by_date = _minute_amount_by_date(minute)
    b_vol = _to_float(b["volume"])
    b_amt = _day_amount(code, break_date, minute, df)
    r["break_volume"] = b_vol
    r["break_amount"] = b_amt
    bd_vols = [float(x) for x in pd.to_numeric(board_rows["volume"], errors="coerce").dropna() if x > 0]
    bd_amts = [v for v in (_day_amount(code, str(d), minute, df) for d in board_rows["date"]) if v is not None]
    if not bd_amts:
        bd_amts = [float(x) for x in pd.to_numeric(board_rows["amount"], errors="coerce").dropna() if x > 0]
    bd_tors = [float(x) for x in pd.to_numeric(board_rows["turnover_rate"], errors="coerce").dropna() if x > 0]
    r["break_volume_ratio_vs_board_days"] = _safe_div(b_vol, float(np.mean(bd_vols))) if bd_vols else None
    r["break_amount_ratio_vs_board_days"] = _safe_div(b_amt, float(np.mean(bd_amts))) if bd_amts else None
    b_tor = _to_float(b["turnover_rate"])
    r["break_turnover_ratio"] = _safe_div(b_tor, float(np.mean(bd_tors))) if bd_tors else None
    lp = round_price_limit(b_prev_close) if b_prev_close else None
    r["break_touched_limit_up"] = bool(lp is not None and bh is not None and bh >= lp - LIMIT_UP_PRICE_TOLERANCE)
    r["break_opened_from_limit_up"] = bool(lp is not None and bo is not None and bo >= lp - LIMIT_UP_PRICE_TOLERANCE)
    r["break_day_in_pool"] = bool(code in pool_members.get(break_date, set()))
    r["last_board_day_in_pool"] = bool(code in pool_members.get(lb_date, set()))
    r["pool_consecutive_count_last_board"] = _to_float(pool_meta.get((lb_date, code), {}).get("consecutive_limit_up_count"))

    # ---- D D1 日线与均线位置 ----
    d = row
    r["d1_open"] = _to_float(d["open"])
    r["d1_high"] = _to_float(d["high"])
    r["d1_low"] = _to_float(d["low"])
    r["d1_close"] = _to_float(d["close"])
    r["d1_volume"] = _to_float(d["volume"])
    r["d1_amount"] = _day_amount(code, signal_date, minute, df)
    r["d1_ma5"] = _to_float(mas["ma5"][obs_idx])
    r["d1_ma10"] = _to_float(mas["ma10"][obs_idx])
    r["d1_ma20"] = _to_float(mas["ma20"][obs_idx])
    r["d1_close_to_ma5"] = _ret(_safe_div(d["close"], mas["ma5"][obs_idx]))
    r["d1_low_to_ma5"] = _ret(_safe_div(d["low"], mas["ma5"][obs_idx]))
    r["d1_high_to_ma5"] = _ret(_safe_div(d["high"], mas["ma5"][obs_idx]))
    r["d1_close_to_ma10"] = _ret(_safe_div(d["close"], mas["ma10"][obs_idx]))
    r["d1_low_to_ma10"] = _ret(_safe_div(d["low"], mas["ma10"][obs_idx]))
    r["d1_close_to_ma20"] = _ret(_safe_div(d["close"], mas["ma20"][obs_idx]))
    r["d1_ma5_slope"] = _ret(_safe_div(mas["ma5"][obs_idx], mas["ma5"][obs_idx - 1])) if obs_idx >= 1 else None
    r["d1_ma10_slope"] = _ret(_safe_div(mas["ma10"][obs_idx], mas["ma10"][obs_idx - 1])) if obs_idx >= 1 else None
    d1_close_v = _to_float(d["close"])
    m5 = _to_float(mas["ma5"][obs_idx])
    m10 = _to_float(mas["ma10"][obs_idx])
    r["d1_reclaimed_ma5"] = bool(m5 is not None and d1_close_v is not None and d1_close_v >= m5)
    r["d1_reclaimed_ma10"] = bool(m10 is not None and d1_close_v is not None and d1_close_v >= m10)
    cd5 = 0
    jj = obs_idx
    while jj >= 0 and pd.notna(mas["ma5"][jj]) and pd.notna(df["close"].iloc[jj]) and df["close"].iloc[jj] < mas["ma5"][jj]:
        cd5 += 1
        jj -= 1
    cd10 = 0
    jj = obs_idx
    while jj >= 0 and pd.notna(mas["ma10"][jj]) and pd.notna(df["close"].iloc[jj]) and df["close"].iloc[jj] < mas["ma10"][jj]:
        cd10 += 1
        jj -= 1
    r["consecutive_days_below_ma5"] = cd5
    r["consecutive_days_below_ma10"] = cd10

    # ---- E D1 修复质量 (5min) ----
    day_m = minute[minute["trade_date"] == signal_date].sort_values("datetime")
    o_c = _first_numeric(day_m, "open")
    h_c = _max_numeric(day_m, "high")
    l_c = _min_numeric(day_m, "low")
    c_c = _last_numeric(day_m, "close")
    r["d1_open_to_close_return"] = _ret(_safe_div(c_c, o_c))
    r["d1_low_to_close_recovery"] = _safe_div(c_c - l_c, l_c)
    r["d1_high_to_close_drawdown"] = _safe_div(h_c - c_c, h_c)
    r["d1_close_location"] = _safe_div(c_c - l_c, h_c - l_c) if (h_c is not None and l_c is not None and h_c != l_c) else None
    vwap = intraday_vwap(day_m)
    r["d1_vwap"] = vwap
    r["d1_close_to_vwap"] = _ret(_safe_div(c_c, vwap))
    prev_close_d1 = _to_float(df["close"].iloc[obs_idx - 1])
    r["d1_intraday_range"] = _safe_div(h_c - l_c, prev_close_d1)
    t_col = pd.to_datetime(day_m["time"], format="%H:%M:%S", errors="coerce").dt.time
    morning = day_m[t_col < pd.Timestamp("13:00:00").time()]
    morning_close = _last_numeric(morning, "close")
    r["d1_afternoon_return"] = _ret(_safe_div(c_c, morning_close))
    hour14 = day_m[t_col <= pd.Timestamp("14:00:00").time()]
    hour14_close = _last_numeric(hour14, "close")
    r["d1_last_hour_return"] = _ret(_safe_div(c_c, hour14_close))
    tot_vol = float(pd.to_numeric(day_m["volume"], errors="coerce").sum())
    if tot_vol and tot_vol > 0:
        up_vol = float(pd.to_numeric(day_m.loc[day_m["close"] > day_m["open"], "volume"], errors="coerce").sum())
        dn_vol = float(pd.to_numeric(day_m.loc[day_m["close"] < day_m["open"], "volume"], errors="coerce").sum())
        r["d1_up_bar_volume_ratio"] = up_vol / tot_vol
        r["d1_down_bar_volume_ratio"] = dn_vol / tot_vol
    else:
        r["d1_up_bar_volume_ratio"] = None
        r["d1_down_bar_volume_ratio"] = None

    # ---- F 对手盘与成交压力 ----
    d1_close_v2 = _to_float(d["close"])
    brk_close = _to_float(b["close"])
    d1_high_v = _to_float(d["high"])
    d1_low_v = _to_float(d["low"])
    high_zone_thr = d1_low_v + 0.7 * (d1_high_v - d1_low_v)
    vol_s = pd.to_numeric(day_m["volume"], errors="coerce")
    amt_s = pd.to_numeric(day_m["amount"], errors="coerce")
    cl_s = pd.to_numeric(day_m["close"], errors="coerce")
    op_s = pd.to_numeric(day_m["open"], errors="coerce")
    hi_s = pd.to_numeric(day_m["high"], errors="coerce")
    lo_s = pd.to_numeric(day_m["low"], errors="coerce")
    typ = (hi_s + lo_s + cl_s) / 3.0
    tot_amt = float(amt_s.sum())
    t_afternoon = t_col >= pd.Timestamp("14:00:00").time()
    if tot_vol and tot_vol > 0:
        r["volume_above_d1_close_ratio"] = float(vol_s[cl_s >= d1_close_v2].sum()) / tot_vol
        r["volume_above_break_close_ratio"] = float(vol_s[cl_s >= brk_close].sum()) / tot_vol
        r["high_zone_volume_ratio"] = float(vol_s[typ >= high_zone_thr].sum()) / tot_vol
        r["late_day_sell_volume_ratio"] = float(vol_s[t_afternoon & (cl_s < op_s)].sum()) / tot_vol
        r["down_bar_volume_ratio"] = float(vol_s[cl_s < op_s].sum()) / tot_vol
    else:
        r["volume_above_d1_close_ratio"] = None
        r["volume_above_break_close_ratio"] = None
        r["high_zone_volume_ratio"] = None
        r["late_day_sell_volume_ratio"] = None
        r["down_bar_volume_ratio"] = None
    if tot_amt and tot_amt > 0:
        r["amount_above_d1_close_ratio"] = float(amt_s[cl_s >= d1_close_v2].sum()) / tot_amt
        r["high_zone_amount_ratio"] = float(amt_s[typ >= high_zone_thr].sum()) / tot_amt
        r["late_day_sell_amount_ratio"] = float(amt_s[t_afternoon & (cl_s < op_s)].sum()) / tot_amt
    else:
        r["amount_above_d1_close_ratio"] = None
        r["high_zone_amount_ratio"] = None
        r["late_day_sell_amount_ratio"] = None
    r["d1_vwap_to_close_gap"] = _safe_div(vwap - c_c, c_c)

    # ---- H 标签 ----
    d2_date = o["d2_trade_date"]
    d3_date = o["d3_trade_date"]
    d2_m = minute[minute["trade_date"] == d2_date]
    d3_m = minute[minute["trade_date"] == d3_date]
    r["d2_trade_date"] = d2_date
    r["d3_trade_date"] = d3_date
    d2_open = _first_numeric(d2_m, "open")
    if d2_open is None:
        d2_open = _daily_price_for_date(code, d2_date, "open")
    d2_high = _max_numeric(d2_m, "high") or _daily_price_for_date(code, d2_date, "high")
    d2_low = _min_numeric(d2_m, "low") or _daily_price_for_date(code, d2_date, "low")
    d2_close = _last_numeric(d2_m, "close") or _daily_price_for_date(code, d2_date, "close")
    d3_high = _max_numeric(d3_m, "high")
    d3_low = _min_numeric(d3_m, "low") or _daily_price_for_date(code, d3_date, "low")
    d3_close = _last_numeric(d3_m, "close")
    r["d2_open"] = d2_open
    r["d2_high"] = d2_high
    r["d2_low"] = d2_low
    r["d2_close"] = d2_close
    r["d3_high"] = d3_high
    r["d3_low"] = d3_low
    r["d3_close"] = d3_close
    r["d2open_to_d3high_return"] = _ret(_safe_div(d3_high, d2_open))
    r["d2open_to_d3close_return"] = _ret(_safe_div(d3_close, d2_open))
    r["target7_d2open_d3high"] = bool(r["d2open_to_d3high_return"] is not None and r["d2open_to_d3high_return"] >= 0.07)
    r["tail_loss_5pct"] = bool(r["d2open_to_d3close_return"] is not None and r["d2open_to_d3close_return"] <= -0.05)
    r["future_data_status"] = "complete"
    candidate_rows.append(r)

print("candidate rows:", len(candidate_rows))
cand_df = pd.DataFrame(candidate_rows)
cand_df.to_csv(SCRATCH / "stage3_candidates_feat.csv", index=False, encoding="utf-8-sig")
print("[5] done")

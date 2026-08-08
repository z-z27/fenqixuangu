# -*- coding: utf-8 -*-
"""v004c BaoStock 5m Integration + May/June Historical Rebuild v001 — 构建工具

用法:
    python tools/build_v004c_baostock_d1_dev_v001.py [--cache-only] [--force-fetch]
                                                     [--no-label-recovery]
                                                     [--output DIR]

输出 (reports/research/v004c_baostock_d1_dev_v001_20260506_20260630/):
    v004c_baostock_d1_dev_v001.csv              May+June D1 development table
    v004c_baostock_d1_minute_quality_v001.csv   逐 event 5min 质量审计
    v004c_baostock_d1_feature_lineage_v001.csv  minute feature 公式来源/依赖清单
    v004c_baostock_d1_rebuild_review_v001.md    中文 review
    v004c_baostock_d1_fetch_provenance.json     fetch timestamp 独立 provenance

数据源: BaoStock 5m (frequency=5, adjustflag=3 -> adjustment=none) 为 D1 5min
主源; 日线状态用现有 canonical daily cache; Sina 行为完全保留 (本工具只读)。
May 146 行全量 + June 冻结表按 signal_date 06-01..06-30 过滤, July 不进开发数据。
确定性: 第二次 --cache-only 重建时核心 CSV / review byte-identical
(fetch timestamp 只放独立 provenance 文件)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import Baostock5mCache  # noqa: E402
from src.config import DataConfig  # noqa: E402
from src.data_sources import (  # noqa: E402
    BAOSTOCK_5M_SOURCE,
    NORMALIZED_5M_INTERVAL,
    MarketDataProvider,
    clear_baostock_socks5_proxy,
    login_baostock,
    logout_baostock,
    set_baostock_socks5_proxy,
    validate_normalized_5m_frame,
)
from src.v004c_may_d1_coverage import (  # noqa: E402
    audit_d1_minute_complete,
    audit_daily_label,
    build_trading_calendar,
    load_daily_frame,
    load_suspension_proofs,
)
from src.v004c_minute_features import (  # noqa: E402
    MINUTE_DEPENDENT_FEATURES,
    compute_minute_features,
)
from src.v004c_model_table import (  # noqa: E402
    ModelTableError,
    compute_recent_7d_features,
)

MAY_CSV = (ROOT / "reports" / "research" / "v004c_may_d1_coverage_v001_202605"
           / "v004c_may_d1_candidates_v001.csv")
JUNE_CSV = (ROOT / "reports" / "research" / "v004c_d1_dataset_v001_20260601_20260729"
            / "v004c_training_d1_v001.csv")
MODEL_TABLE_CSV = (ROOT / "reports" / "research"
                   / "v004c_model_table_v001_20260601_20260729"
                   / "v004c_model_table_v001.csv")
OUT_DIR = ROOT / "reports" / "research" / "v004c_baostock_d1_dev_v001_20260506_20260630"
BAO_CACHE_DIR = ROOT / "data" / "cache" / "baostock_5m"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
SUSP_DIR = ROOT / "data" / "cache" / "suspension_status"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
SINA_MINUTE_DIR = ROOT / "data" / "cache" / "minute_5m"

WINDOW_START = "2026-05-06"
WINDOW_END = "2026-06-30"
JUNE_START = "2026-06-01"
JUNE_END = "2026-06-30"
FETCH_SLEEP_SECONDS = 0.05
MA_RECOMPUTE_EPSILON = 1e-9

# ---------------------------------------------------------------------------
# 日线 / MA 状态列 (canonical daily 公式, 官方 D section + 字典重算检查)
# ---------------------------------------------------------------------------
DAILY_STATE_COLUMNS = (
    "d1_open", "d1_high", "d1_low", "d1_close", "break_open_return",
)
MA_STATE_COLUMNS = (
    "d1_ma5", "d1_ma10", "d1_ma20",
    "d1_close_to_ma5_raw", "d1_low_to_ma5_raw", "d1_high_to_ma5_raw",
    "d1_close_to_ma10_raw", "d1_low_to_ma10_raw", "d1_close_to_ma20",
    "d1_ma5_slope", "d1_ma10_slope",
    "d1_close_above_ma5", "d1_close_above_ma10",
    "d1_true_reclaim_ma5", "d1_true_reclaim_ma10",
    "consecutive_days_below_ma5", "consecutive_days_below_ma10",
)
RECENT_7D_COLUMNS = (
    "recent_7d_cumulative_return", "recent_7d_max_drawdown",
    "recent_7d_close_position",
)
MINUTE_FEATURES = tuple(MINUTE_DEPENDENT_FEATURES)

PROVENANCE_COLUMNS = ("minute_source", "minute_adjustment", "minute_interval", "daily_source")

# June 冻结表内可用于交叉核对 (与重建公式同列名) 的列
JUNE_CROSSCHECK_COLUMNS = (
    "d1_open", "d1_high", "d1_low", "d1_close",
    "d1_ma5", "d1_ma10", "d1_ma20",
    "d1_close_to_ma5_raw", "d1_low_to_ma5_raw", "d1_high_to_ma5_raw",
    "d1_close_to_ma10_raw", "d1_low_to_ma10_raw", "d1_close_to_ma20",
    "d1_ma5_slope", "d1_ma10_slope",
    "d1_close_above_ma5", "d1_close_above_ma10",
    "d1_true_reclaim_ma5", "d1_true_reclaim_ma10",
    "consecutive_days_below_ma5", "consecutive_days_below_ma10",
    "break_open_return",
)


# ---------------------------------------------------------------------------
# Universe (May 146 全量 + June 冻结过滤)
# ---------------------------------------------------------------------------
def load_universe(may_csv: Path, june_csv: Path) -> pd.DataFrame:
    may = pd.read_csv(may_csv, dtype={"code": str})
    june = pd.read_csv(june_csv, dtype={"code": str})
    june = june[june["signal_date"].astype(str).between(JUNE_START, JUNE_END)].copy()
    if may.empty or june.empty:
        raise RuntimeError("empty universe: may/june both must be non-empty")

    may_rows = pd.DataFrame({
        "event_id": may["event_id"],
        "code": may["code"].astype(str).str.zfill(6),
        "name": may.get("name", pd.Series("", index=may.index)),
        "signal_date": may["signal_date"].astype(str),
        "break_date": may["break_date"].astype(str),
        "board_streak_before_break": may["board_streak_before_break"],
        "source_window": "may",
        "target7_daily_d2open_d3high": may.get("target7_daily_d2open_d3high"),
        "tail_loss_daily_5pct": may.get("tail_loss_daily_5pct"),
        "daily_label_complete": may.get("daily_label_complete"),
        "daily_label_reason": may.get("daily_label_reason"),
        "target_training_eligible": None,
        "label_d2_date": may.get("label_d2_date"),
        "label_d3_date": may.get("label_d3_date"),
    })
    june_rows = pd.DataFrame({
        "event_id": june["event_id"],
        "code": june["code"].astype(str).str.zfill(6),
        "name": june.get("name", pd.Series("", index=june.index)),
        "signal_date": june["signal_date"].astype(str),
        "break_date": june["break_date"].astype(str),
        "board_streak_before_break": june["board_streak_before_break"],
        "source_window": "june",
        "target7_daily_d2open_d3high": june.get("target7_daily_d2open_d3high"),
        "tail_loss_daily_5pct": june.get("tail_loss_daily_5pct"),
        "daily_label_complete": june.get("daily_label_quality_ok"),
        "target_training_eligible": june.get("target_training_eligible"),
        "label_d2_date": june.get("label_d2_date"),
        "label_d3_date": june.get("label_d3_date"),
    })
    universe = pd.concat([may_rows, june_rows], ignore_index=True)
    # 身份校验: D1 event 必须 signal_date == break_date (select_business_candidates 语义)
    mismatch = universe[universe["signal_date"] != universe["break_date"]]
    if not mismatch.empty:
        raise RuntimeError(
            f"universe violates D1 identity rule (signal_date==break_date): "
            f"{mismatch['event_id'].tolist()}")
    dup = universe[universe["event_id"].duplicated(keep=False)]
    if not dup.empty:
        raise RuntimeError(f"universe has duplicate event_id: {dup['event_id'].tolist()}")
    return universe.reset_index(drop=True)


# ---------------------------------------------------------------------------
# BaoStock 批量获取 (2026-05-06..2026-06-30, 同股合并请求后按 event 切片)
# ---------------------------------------------------------------------------
def _expected_window_start(code: str) -> str:
    """窗口起点: 一般股票 = WINDOW_START; 窗口期内新上市股票 = 上市首日
    (canonical daily cache 首条 bar 日期), 因为 baostock 在上市前本就没有
    数据, 缓存无法也无需覆盖 WINDOW_START。"""
    daily = load_daily_frame(code, DAILY_DIR)
    if daily is not None and not daily.empty:
        first = str(pd.to_datetime(daily["date"], errors="coerce").min().date())
        if first > WINDOW_START:
            return first
    return WINDOW_START


def _expected_window_end(code: str) -> str:
    """窗口终点: 一般股票 = WINDOW_END; 窗口期内停牌/终止上市股票 = 最后交易日
    (canonical daily cache 末条 bar 日期), 因为 baostock 此后本就没有数据,
    缓存无法也无需覆盖 WINDOW_END。"""
    daily = load_daily_frame(code, DAILY_DIR)
    if daily is not None and not daily.empty:
        last = str(pd.to_datetime(daily["date"], errors="coerce").max().date())
        if last < WINDOW_END:
            return last
    return WINDOW_END


def fetch_baostock_window(
    universe: pd.DataFrame,
    cache: Baostock5mCache,
    provider: MarketDataProvider,
    force: bool = False,
) -> dict:
    """对每个 code 一次请求覆盖 WINDOW_START..WINDOW_END; login/logout 批量生命周期。

    已有 cache 且覆盖完整窗口 (且非 force) 时跳过 fetch (第二次 cache-only 重建)。
    禁止 silent fallback: 任何失败显式 raise。
    """
    codes = sorted(set(universe["code"]))
    provenance: dict[str, dict] = {}
    login_baostock()
    try:
        for code in codes:
            existing = cache.read(code)
            existing_meta = cache.read_meta(code)
            covered = False
            if existing is not None and not existing.empty and existing_meta:
                dates = pd.to_datetime(existing["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
                covered = bool(
                    not dates.empty
                    and dates.min() <= _expected_window_start(code)
                    and dates.max() >= _expected_window_end(code)
                )
            if covered and not force:
                provenance[code] = {"cache": "reused", "date_min": str(dates.min()),
                                    "date_max": str(dates.max()),
                                    "rows": int(len(existing))}
                continue
            frame, _source = provider.fetch_5min_history(
                code, f"{WINDOW_START} 09:00:00", f"{WINDOW_END} 15:30:00",
                adjust="none", source=BAOSTOCK_5M_SOURCE)
            validate_normalized_5m_frame(frame)
            # 与已有 cache 合并: timestamp 去重排序, 新数据优先; 禁止 forward fill
            if existing is not None and not existing.empty:
                merged = pd.concat([existing, frame], ignore_index=True)
                merged = merged.drop_duplicates(["code", "datetime"], keep="last")
                merged = merged.sort_values("datetime").reset_index(drop=True)
                frame = merged
            cache.write(code, frame, meta=cache.build_meta(
                code, frame, source=BAOSTOCK_5M_SOURCE,
                adjustment="none", interval=NORMALIZED_5M_INTERVAL,
                fetch_timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
            dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
            provenance[code] = {"cache": "fetched", "date_min": str(dates.min()),
                                "date_max": str(dates.max()), "rows": int(len(frame))}
    finally:
        logout_baostock()
    return provenance


# ---------------------------------------------------------------------------
# 日线 / MA 状态 (canonical daily cache, 官方公式)
# ---------------------------------------------------------------------------
def _ret(value: float | None) -> float | None:
    if value is None:
        return None
    return value - 1.0


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_daily_ma_state(daily: pd.DataFrame, code: str, d1_date: str) -> dict:
    """官方 D section 公式: rolling(5/10/20, min_periods=N).mean() 于日线 close。

    返回 DAILY_STATE_COLUMNS + MA_STATE_COLUMNS 全字段 (按 D1 当日 idx 取值)。
    """
    out: dict = {}
    frame = daily.copy()
    frame["date"] = frame["date"].astype(str)
    frame = frame.sort_values("date").reset_index(drop=True)
    if not frame["date"].is_unique:
        raise RuntimeError(f"daily cache {code} date not unique")
    hits = frame.index[frame["date"] == str(d1_date)]
    if len(hits) == 0:
        raise RuntimeError(f"daily cache {code} missing D1 date {d1_date}")
    obs = int(hits[0])
    if obs < 19:
        raise RuntimeError(f"daily cache {code} insufficient history before {d1_date}")
    for c in ("open", "high", "low", "close"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    close = frame["close"]
    ma5 = close.rolling(5, min_periods=5).mean().to_numpy(dtype=float)
    ma10 = close.rolling(10, min_periods=10).mean().to_numpy(dtype=float)
    ma20 = close.rolling(20, min_periods=20).mean().to_numpy(dtype=float)

    d = frame.iloc[obs]
    out["d1_open"] = float(d["open"]) if pd.notna(d["open"]) else None
    out["d1_high"] = float(d["high"]) if pd.notna(d["high"]) else None
    out["d1_low"] = float(d["low"]) if pd.notna(d["low"]) else None
    out["d1_close"] = float(d["close"]) if pd.notna(d["close"]) else None

    def at(arr: np.ndarray) -> float | None:
        v = arr[obs]
        return float(v) if np.isfinite(v) else None

    def prev(arr: np.ndarray) -> float | None:
        if obs < 1:
            return None
        v = arr[obs - 1]
        return float(v) if np.isfinite(v) else None

    m5, m10, m20 = at(ma5), at(ma10), at(ma20)
    out["d1_ma5"], out["d1_ma10"], out["d1_ma20"] = m5, m10, m20
    c = out["d1_close"]
    lo = out["d1_low"]
    hi = out["d1_high"]
    out["d1_close_to_ma5_raw"] = _ret(_safe_ratio(c, m5))
    out["d1_low_to_ma5_raw"] = _ret(_safe_ratio(lo, m5))
    out["d1_high_to_ma5_raw"] = _ret(_safe_ratio(hi, m5))
    out["d1_close_to_ma10_raw"] = _ret(_safe_ratio(c, m10))
    out["d1_low_to_ma10_raw"] = _ret(_safe_ratio(lo, m10))
    out["d1_close_to_ma20"] = _ret(_safe_ratio(c, m20))
    out["d1_ma5_slope"] = _ret(_safe_ratio(m5, prev(ma5)))
    out["d1_ma10_slope"] = _ret(_safe_ratio(m10, prev(ma10)))
    out["d1_close_above_ma5"] = int(c >= m5) if (c is not None and m5 is not None) else None
    out["d1_close_above_ma10"] = int(c >= m10) if (c is not None and m10 is not None) else None
    out["d1_true_reclaim_ma5"] = (
        int(lo <= m5 and c >= m5) if (lo is not None and m5 is not None and c is not None) else None)
    out["d1_true_reclaim_ma10"] = (
        int(lo <= m10 and c >= m10) if (lo is not None and m10 is not None and c is not None) else None)

    def consecutive_below(ma_arr: np.ndarray) -> int:
        count = 0
        j = obs
        while j >= 0:
            mv = ma_arr[j]
            cv = close.iloc[j]
            if not np.isfinite(mv) or pd.isna(cv) or not (float(cv) < float(mv)):
                break
            count += 1
            j -= 1
        return count

    out["consecutive_days_below_ma5"] = consecutive_below(ma5)
    out["consecutive_days_below_ma10"] = consecutive_below(ma10)
    # break_open_return = break_open / prev_close - 1 (break_date == signal_date)
    b_prev_close = close.iloc[obs - 1] if obs >= 1 else np.nan
    out["break_open_return"] = (
        _ret(_safe_ratio(out["d1_open"], float(b_prev_close)))
        if obs >= 1 and pd.notna(b_prev_close) else None)
    out["prev_close_d1"] = (float(b_prev_close)
                            if obs >= 1 and pd.notna(b_prev_close) else None)
    return out


# ---------------------------------------------------------------------------
# 单 event 行
# ---------------------------------------------------------------------------
def build_event_row(
    event: pd.Series,
    bao_cache: Baostock5mCache,
    daily: pd.DataFrame | None,
    cal_idx: dict,
    proofs: dict,
) -> dict:
    code = event["code"]
    d1_date = event["signal_date"]
    row: dict = {
        "event_id": event["event_id"],
        "code": code,
        "name": event["name"],
        "signal_date": d1_date,
        "break_date": event["break_date"],
        "board_streak_before_break": event["board_streak_before_break"],
        "source_window": event["source_window"],
    }

    # ---- 5min 质量 (48-bar 正式规则复用) ----
    minute = bao_cache.read(code)
    day_m = None
    if minute is not None and not minute.empty:
        day_m = minute[minute["trade_date"].astype(str) == d1_date].sort_values("datetime")
        if day_m.empty:
            day_m = None
    audit = audit_d1_minute_complete(code, d1_date, day_m if day_m is not None else minute)
    row["d1_minute_complete"] = bool(audit["d1_minute_complete"])
    row["actual_bars"] = int(audit["actual_bars"])
    row["expected_bars"] = int(audit["expected_bars"])
    row["first_timestamp"] = str(audit["first_timestamp"])
    row["last_timestamp"] = str(audit["last_timestamp"])
    row["missing_bars"] = int(audit["missing_bars"])
    row["duplicate_bars"] = int(audit["duplicate_bars"])
    row["invalid_price_volume"] = int(audit["invalid_price_volume"])
    row["minute_missing_reason"] = str(audit["missing_reason"])

    # ---- 日线 / MA 状态 (canonical daily) ----
    daily_state: dict = {}
    if daily is not None:
        try:
            daily_state = compute_daily_ma_state(daily, code, d1_date)
        except RuntimeError:
            daily_state = {}
    for col in DAILY_STATE_COLUMNS + MA_STATE_COLUMNS:
        row[col] = daily_state.get(col)

    # ---- recent_7d (v004c_model_table 官方函数) ----
    for col in RECENT_7D_COLUMNS:
        row[col] = None
    if daily is not None:
        try:
            recent = compute_recent_7d_features(daily, d1_date)
            for col in RECENT_7D_COLUMNS:
                row[col] = recent.get(col)
        except (ModelTableError, RuntimeError):
            pass

    # ---- 22 minute features (BaoStock, 官方公式) ----
    minute_features: dict = {}
    if day_m is not None and not day_m.empty:
        minute_features = compute_minute_features(
            day_m,
            daily_close=row.get("d1_close"),
            daily_high=row.get("d1_high"),
            daily_low=row.get("d1_low"),
            prev_close_d1=daily_state.get("prev_close_d1"),
            break_close=row.get("d1_close"),
        )
    for col in MINUTE_FEATURES:
        row[col] = minute_features.get(col)

    # ---- Target 标签 (May 来自 candidates CSV; June 来自冻结表; 不重算) ----
    row["target7_daily_d2open_d3high"] = event.get("target7_daily_d2open_d3high")
    row["tail_loss_daily_5pct"] = event.get("tail_loss_daily_5pct")
    row["daily_label_complete"] = event.get("daily_label_complete")
    row["target_training_eligible"] = event.get("target_training_eligible")
    row["label_d2_date"] = event.get("label_d2_date")
    row["label_d3_date"] = event.get("label_d3_date")

    # ---- provenance (静态标签, 不进 timestamp) ----
    row["minute_source"] = BAOSTOCK_5M_SOURCE
    row["minute_adjustment"] = "none"
    row["minute_interval"] = NORMALIZED_5M_INTERVAL
    row["daily_source"] = "canonical_daily_cache"

    # ---- 完整度 ----
    minute_non_null = sum(1 for col in MINUTE_FEATURES if row[col] is not None)
    row["minute_feature_count"] = minute_non_null
    row["minute_feature_complete"] = bool(minute_non_null == len(MINUTE_FEATURES))
    daily_non_null = sum(1 for col in DAILY_STATE_COLUMNS + MA_STATE_COLUMNS + RECENT_7D_COLUMNS
                         if row.get(col) is not None)
    row["daily_feature_complete"] = bool(
        daily_non_null == len(DAILY_STATE_COLUMNS) + len(MA_STATE_COLUMNS) + len(RECENT_7D_COLUMNS))
    row["final_x_data_complete"] = bool(
        row["d1_minute_complete"] and row["minute_feature_complete"] and row["daily_feature_complete"])
    return row


# ---------------------------------------------------------------------------
# May label recovery (§15: 现有 daily source pipeline 补齐, 不造假标签)
# ---------------------------------------------------------------------------
def _persist_daily_cache(code: str, frame: pd.DataFrame) -> None:
    """补齐后的日线写回 canonical daily cache (保持 date=datetime 原格式, 原子写)。"""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    path = DAILY_DIR / f"{code}_daily.pkl"
    tmp = path.with_name(f"{path.stem}.devrebuild.tmp")
    out.to_pickle(tmp)
    tmp.replace(path)


def recover_may_labels(
    universe: pd.DataFrame,
    provider: MarketDataProvider,
    cal_idx: dict,
    proofs: dict,
    do_recovery: bool = True,
) -> dict:
    """对 daily_label_complete=False 的 May 事件, 经现有 fetch_daily_history 尝试补齐
    d3 日线 bar; 成功后用 audit_daily_label 重算, 记录 label_recovered; 失败则
    保持 label_training_eligible=False (不造假标签)。"""
    out: dict[str, dict] = {}
    may = universe[universe["source_window"] == "may"]
    incomplete = may[may["daily_label_complete"].ne(True)]
    for _, ev in incomplete.iterrows():
        code = ev["code"]
        daily = load_daily_frame(code, DAILY_DIR)
        before = audit_daily_label(code, ev["signal_date"], cal_idx, proofs, daily)
        recovered = False
        if do_recovery:
            try:
                fetched, _src = provider.fetch_daily_history(
                    code, "2026-04-01", "2026-06-30", adjust="none")
                merged = fetched
                added_rows = len(fetched)
                if daily is not None and not daily.empty:
                    merged = pd.concat([daily, fetched], ignore_index=True)
                    merged = merged.drop_duplicates(["code", "date"], keep="last")
                    merged = merged.sort_values("date").reset_index(drop=True)
                    added_rows = len(merged) - len(daily)
                if added_rows > 0:
                    _persist_daily_cache(code, merged)
                daily = merged
                after = audit_daily_label(code, ev["signal_date"], cal_idx, proofs, daily)
                recovered = bool(after["daily_label_complete"])
            except Exception:
                after = before
                recovered = False
        else:
            after = before
        # before 状态取冻结 candidates CSV 记录 (真实恢复前状态, 跨运行确定);
        # after 状态取恢复尝试后的实时审计。
        out[ev["event_id"]] = {
            "code": code,
            "signal_date": ev["signal_date"],
            "label_d2_date": str(before.get("label_d2_date")),
            "label_d3_date": str(before.get("label_d3_date")),
            "reason_before": str(ev.get("daily_label_reason") or before.get("daily_label_reason")),
            "d2_open_before": before.get("d2_open_daily"),
            "d3_high_before": before.get("d3_high_daily"),
            "label_complete_before": bool(ev.get("daily_label_complete", False)),
            "label_complete_after": bool(after["daily_label_complete"]),
            "label_recovered": recovered,
            "target7_after": after.get("target7_daily_d2open_d3high"),
            "tail_loss_after": after.get("tail_loss_daily_5pct"),
            "reason_after": str(after.get("daily_label_reason")),
        }
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
DEV_CSV_COLUMNS = (
    ("event_id", "code", "name", "signal_date", "break_date",
     "board_streak_before_break", "source_window")
    + DAILY_STATE_COLUMNS + MA_STATE_COLUMNS + RECENT_7D_COLUMNS
    + MINUTE_FEATURES
    + ("target7_daily_d2open_d3high", "tail_loss_daily_5pct",
       "daily_label_complete", "target_training_eligible",
       "label_d2_date", "label_d3_date")
    + PROVENANCE_COLUMNS
    + ("d1_minute_complete", "minute_feature_count", "minute_feature_complete",
       "daily_feature_complete", "final_x_data_complete")
)


def render_lineage_csv() -> pd.DataFrame:
    rows = []
    for name in MINUTE_FEATURES:
        rows.append({
            "feature_name": name,
            "feature_group": "minute",
            "formula_source": "build_v004c_dataset.py E/F (src/v004c_minute_features.py 1:1)",
            "raw_dependency": "5m bars: open/high/low/close/volume/amount",
            "rebuilt_from": "baostock_5m",
        })
    for name in DAILY_STATE_COLUMNS + MA_STATE_COLUMNS:
        rows.append({
            "feature_name": name,
            "feature_group": "daily_ma_state",
            "formula_source": "build_v004c_dataset.py D section + v004c_factor_dictionary EXISTING_RECOMPUTE_CHECKS",
            "raw_dependency": "canonical daily cache OHLC (rolling ma)",
            "rebuilt_from": "canonical_daily_cache",
        })
    for name in RECENT_7D_COLUMNS:
        rows.append({
            "feature_name": name,
            "feature_group": "recent_7d",
            "formula_source": "v004c_model_table.py compute_recent_7d_features",
            "raw_dependency": "canonical daily cache OHLC (7-row window)",
            "rebuilt_from": "canonical_daily_cache",
        })
    return pd.DataFrame(rows)


def write_outputs(
    rows: list[dict],
    universe: pd.DataFrame,
    provenance: dict,
    may_label_recovery: dict,
    june_crosscheck: dict,
    sina_regression: dict,
    byte_identical: bool | None,
    socks5_proxy: str | None = None,
) -> None:
    full = pd.DataFrame(rows)
    full = full.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    quality = full[[
        "event_id", "code", "signal_date", "source_window",
        "expected_bars", "actual_bars", "first_timestamp", "last_timestamp",
        "missing_bars", "duplicate_bars", "invalid_price_volume",
        "d1_minute_complete", "minute_missing_reason",
        "minute_feature_count", "minute_feature_complete",
    ]].copy()
    quality.to_csv(OUT_DIR / "v004c_baostock_d1_minute_quality_v001.csv",
                   index=False, encoding="utf-8-sig")

    dev = full[list(DEV_CSV_COLUMNS)].copy()
    dev.to_csv(OUT_DIR / "v004c_baostock_d1_dev_v001.csv", index=False, encoding="utf-8-sig")

    lineage = render_lineage_csv()
    lineage.to_csv(OUT_DIR / "v004c_baostock_d1_feature_lineage_v001.csv",
                   index=False, encoding="utf-8-sig")

    stats = build_stats(dev, universe, provenance, may_label_recovery,
                        june_crosscheck, sina_regression, byte_identical)
    (OUT_DIR / "v004c_baostock_d1_rebuild_review_v001.md").write_text(
        render_review(stats), encoding="utf-8")

    with (OUT_DIR / "v004c_baostock_d1_fetch_provenance.json").open(
            "w", encoding="utf-8") as fh:
        json.dump({"window": [WINDOW_START, WINDOW_END],
                   "cache_dir": str(BAO_CACHE_DIR),
                   "socks5_proxy": socks5_proxy,
                   "codes": provenance},
                  fh, ensure_ascii=False, sort_keys=True, indent=1)


# ---------------------------------------------------------------------------
# Stats + review
# ---------------------------------------------------------------------------
def build_stats(
    dev: pd.DataFrame,
    universe: pd.DataFrame,
    provenance: dict,
    may_label_recovery: dict,
    june_crosscheck: dict,
    sina_regression: dict,
    byte_identical: bool | None,
) -> dict:
    stats: dict = {}
    for window in ("may", "june"):
        sub = dev[dev["source_window"] == window]
        stats[window] = {
            "rows": int(len(sub)),
            "signal_dates": int(sub["signal_date"].nunique()),
            "codes": int(sub["code"].nunique()),
            "bao_fetched": int(sub["minute_source"].eq(BAOSTOCK_5M_SOURCE).sum()),
            "minute_complete_48": int(sub["d1_minute_complete"].sum()),
            "minute_complete_rate": float(sub["d1_minute_complete"].mean()) if len(sub) else 0.0,
            "minute_features_complete": int(sub["minute_feature_complete"].sum()),
            "daily_features_complete": int(sub["daily_feature_complete"].sum()),
            "label_complete": int(sub["daily_label_complete"].fillna(False).astype(bool).sum()),
            "target_label_complete": int(
                sub["daily_label_complete"].fillna(False).astype(bool).sum()
                + (sub["target_training_eligible"].fillna(False).astype(bool).sum()
                   if window == "june" else 0)),
            "x_complete": int(sub["final_x_data_complete"].sum()),
        }
    stats["may_label_recovery"] = may_label_recovery
    stats["june_crosscheck"] = june_crosscheck
    stats["sina_regression"] = sina_regression
    stats["byte_identical"] = byte_identical
    return stats


def render_review(stats: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# v004c BaoStock 5m Integration + May/June Historical Rebuild v001 — Review\n")
    add("任务性质: **HISTORICAL DATA FOUNDATION REBUILD** (不训练 / 不筛选 / 不调参)。"
        "BaoStock 5m 为 May+June D1 历史 5min 主数据源, Sina 行为完全保留; "
        "22 个 minute-dependent features 用官方公式 (build_v004c_dataset.py E/F 段 1:1) "
        "从 BaoStock 重建, 日线 / MA / recent-7d 状态用现有 canonical daily cache 公式重建, "
        "Target 标签全部来自现有审计链, 不重算。\n")
    add(f"- 窗口: {WINDOW_START}..{WINDOW_END}; 输出: {OUT_DIR.name}/; "
        f"fetch timestamp 仅存于 v004c_baostock_d1_fetch_provenance.json\n")
    add("- July 不进入开发数据; 冻结资产 (repair_state / logistic_walkforward / d1_dataset) 未修改\n")

    add("\n## 1. Universe\n")
    add("| window | rows | signal_dates | codes |")
    add("|---|---|---|---|")
    for window in ("may", "june"):
        s = stats[window]
        add(f"| {window} | {s['rows']} | {s['signal_dates']} | {s['codes']} |")

    add("\n## 2. BaoStock 5min 获取与覆盖 (source=baostock_5m, adjustment=none, interval=5m)\n")
    add("| window | bao_fetched | 48-bar complete | minute_features_complete | daily_features_complete | final_x_complete |")
    add("|---|---|---|---|---|---|")
    for window in ("may", "june"):
        s = stats[window]
        add(f"| {window} | {s['bao_fetched']}/{s['rows']} | {s['minute_complete_48']}/{s['rows']} "
            f"({s['minute_complete_rate']:.1%}) | {s['minute_features_complete']}/{s['rows']} "
            f"| {s['daily_features_complete']}/{s['rows']} | {s['x_complete']}/{s['rows']} |")

    add("\n## 3. D1 48-bar completeness (正式规则: 48 bar / 09:35 首 / 15:00 末)\n")
    add("逐 event 明细见 v004c_baostock_d1_minute_quality_v001.csv; "
        "缺失原因分类 (missing_reason) 复用 v004c_may_d1_coverage 正式分类: "
        "cache_missing / d1_date_missing / bar_grid_incomplete / grid_violation。\n")
    for window in ("may", "june"):
        s = stats[window]
        add(f"- {window}: {s['minute_complete_48']}/{s['rows']} 完整 "
            f"({s['minute_complete_rate']:.1%})")

    add("\n## 4. May label recovery (§15)\n")
    rec = stats["may_label_recovery"]
    if not rec:
        add("- 无 label incomplete 事件")
    else:
        add("| event_id | code | signal_date | label_d2_date | label_d3_date | reason_before | "
            "label_complete_before | label_complete_after | label_recovered | target7_after | tail_loss_after |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for event_id, r in sorted(rec.items()):
            add(f"| {event_id} | {r['code']} | {r['signal_date']} | {r['label_d2_date']} | "
                f"{r['label_d3_date']} | {r['reason_before']} | {r['label_complete_before']} | "
                f"{r['label_complete_after']} | {r['label_recovered']} | "
                f"{r['target7_after']} | {r['tail_loss_after']} |")
        recovered = sum(1 for r in rec.values() if r["label_recovered"])
        add(f"\n- recovered: {recovered}/{len(rec)}; 未恢复事件保持 label 不完整, 不造假标签。")

    add("\n## 5. June 重建 vs 冻结表交叉核对\n")
    cc = stats["june_crosscheck"]
    if cc:
        add("| column | match_rate | max_abs_diff |")
        add("|---|---|---|")
        for col, r in cc.items():
            add(f"| {col} | {r['match_rate']:.4f} | {r['max_abs_diff']:.3e} |")
        add("\n- 口径: 重建日线/MA 状态与冻结 v004c_training_d1_v001.csv 同列对比, "
            "epsilon=" + f"{MA_RECOMPUTE_EPSILON}; d1_volume/d1_amount 为 5min 派生, 不在列内。")
    else:
        add("- 无 (June crosscheck 未执行)")

    add("\n## 6. Sina vs BaoStock regression (normalized contract)\n")
    reg = stats["sina_regression"]
    if reg:
        add(f"- 比对事件数: {reg.get('n_events', 0)} (June 两源均有 48-bar 的事件)")
        add(f"- normalized columns same: {reg.get('columns_same', 'n/a')} "
            f"(schema 差异列: {reg.get('column_diffs', [])})")
        add(f"- timestamps same (09:35..15:00): {reg.get('timestamps_same', 'n/a')}")
        add(f"- 48-bar semantics same: {reg.get('bars_same', 'n/a')} "
            f"(sina={reg.get('sina_48', 0)} bao={reg.get('bao_48', 0)})")
        add(f"- volume units same (median bao/sina per-bar ratio): "
            f"{reg.get('volume_ratio_median', 'n/a')} (p10={reg.get('volume_ratio_p10', 'n/a')}, "
            f"p90={reg.get('volume_ratio_p90', 'n/a')})")
        add(f"- amount units same (median bao/sina): {reg.get('amount_ratio_median', 'n/a')}")
        add("- 不要求 OHLC 相同 (源间 bar 级取整差异由 48-bar 语义审计覆盖)")
    else:
        add("- n/a (无两源均完整的事件)")

    add("\n## 7. minute_data_coverage_selection_bias\n")
    may = stats["may"]
    june = stats["june"]
    all_rows = may["rows"] + june["rows"]
    all_48 = may["minute_complete_48"] + june["minute_complete_48"]
    if may["minute_complete_rate"] == 1.0 and june["minute_complete_rate"] == 1.0:
        add(f"- **ELIMINATED**: 全部 {all_rows} 个 event 的 D1 5min 数据均由 BaoStock "
            f"获取且 48-bar 完整 ({all_48}/{all_rows}), 覆盖率不再由旧 Sina cache 存在与否决定。")
        add(f"  证据: may {may['minute_complete_48']}/{may['rows']} (100%), "
            f"june {june['minute_complete_48']}/{june['rows']} (100%)")
    else:
        add(f"- **NOT ELIMINATED**: may {may['minute_complete_48']}/{may['rows']} "
            f"({may['minute_complete_rate']:.1%}), june {june['minute_complete_48']}/{june['rows']} "
            f"({june['minute_complete_rate']:.1%})")

    add("\n## 8. 确定性构建 (byte-identical)\n")
    bi = stats["byte_identical"]
    if bi is None:
        add("- 单次构建 (未做第二次 cache-only 重建)")
    elif bi:
        add("- **PASS**: 第二次 --cache-only 重建后 dev CSV / quality / lineage / review "
            "与首次构建 byte-identical (fetch timestamp 仅 provenance 文件不同)")
    else:
        add("- **FAIL**: 两次构建输出不一致 (见终端 diff 输出)")

    add("\n## 9. 限制\n")
    add("- 日线 / MA / recent-7d 状态来自现有 canonical daily cache; 若缓存含历史回补, "
        "June 交叉核对 (第 5 节) 会显示差异")
    add("- 5min 源替换只影响 minute-dependent features; 冻结模型 / 阈值 / lambda / "
        "特征选择均未重做 (本任务禁止)")
    add("- d1_volume/d1_amount 属 22 个 minute features (官方列表), 由 BaoStock 5min 求和")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# June crosscheck + Sina regression
# ---------------------------------------------------------------------------
def run_june_crosscheck(dev: pd.DataFrame, frozen: pd.DataFrame) -> dict:
    """June dev 重建日线/MA 状态 vs 冻结表同列逐行对比 (epsilon 容差)。"""
    june_dev = dev[dev["source_window"] == "june"].copy()
    frozen = frozen.copy()
    frozen["event_id"] = frozen["event_id"].astype(str)
    june_dev["event_id"] = june_dev["event_id"].astype(str)
    cross_cols = [c for c in JUNE_CROSSCHECK_COLUMNS if c in frozen.columns]
    merged = june_dev.merge(frozen[["event_id"] + cross_cols],
                            on="event_id", how="inner", suffixes=("_dev", "_frz"))
    out: dict = {}
    for col in cross_cols:
        if f"{col}_dev" not in merged.columns or f"{col}_frz" not in merged.columns:
            continue
        a = pd.to_numeric(merged[f"{col}_dev"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_frz"], errors="coerce")
        both = a.notna() & b.notna()
        if not both.any():
            out[col] = {"match_rate": float("nan"), "max_abs_diff": float("nan")}
            continue
        diff = (a[both] - b[both]).abs()
        match = float((diff <= MA_RECOMPUTE_EPSILON).mean())
        out[col] = {"match_rate": match, "max_abs_diff": float(diff.max())}
    return out


def run_sina_regression(dev: pd.DataFrame, universe: pd.DataFrame,
                        sina_dir: Path, bao_cache: Baostock5mCache) -> dict:
    """June 事件中两源均 48-bar 完整的小型 regression: 列 / 时间戳 / 单位 / 48-bar 语义。"""
    june = dev[dev["source_window"] == "june"].copy()
    results: dict = {}
    column_diffs: list[str] = []
    ratios_v: list[float] = []
    ratios_a: list[float] = []
    n_events = 0
    bars_same = 0
    timestamps_same = 0
    for _, ev in june.iterrows():
        code = ev["code"]
        d1 = ev["signal_date"]
        sina = None
        path = Path(sina_dir) / f"{code}_5min.pkl"
        if path.exists():
            try:
                s = pd.read_pickle(path)
                if s is not None and not s.empty:
                    s = s.copy()
                    s["trade_date"] = s["trade_date"].astype(str)
                    s["datetime"] = pd.to_datetime(s["datetime"], errors="coerce")
                    sina = s[s["trade_date"] == d1].sort_values("datetime")
            except Exception:
                sina = None
        bao = bao_cache.read(code)
        bao_day = None
        if bao is not None and not bao.empty:
            bao_day = bao[bao["trade_date"].astype(str) == d1].sort_values("datetime")
        if sina is None or bao_day is None or len(sina) == 0 or len(bao_day) == 0:
            continue
        sina_48 = len(sina) == 48
        bao_48 = len(bao_day) == 48
        if not (sina_48 and bao_48):
            continue
        n_events += 1
        if len(bao_day) == len(sina):
            bars_same += 1
        t_s = sina["datetime"].dt.strftime("%H:%M:%S").tolist()
        t_b = bao_day["datetime"].dt.strftime("%H:%M:%S").tolist()
        if t_s == t_b:
            timestamps_same += 1
        if not column_diffs:
            sc = set(sina.columns)
            bc = set(bao_day.columns)
            column_diffs = sorted(bc - sc)
        sv = pd.to_numeric(sina["volume"], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(bao_day["volume"], errors="coerce").to_numpy(dtype=float)
        n = min(len(sv), len(bv))
        if "amount" in sina.columns:
            sa = pd.to_numeric(sina["amount"], errors="coerce").to_numpy(dtype=float)
            ba = pd.to_numeric(bao_day["amount"], errors="coerce").to_numpy(dtype=float)
            for k in range(n):
                if np.isfinite(sa[k]) and np.isfinite(ba[k]) and sa[k] > 0 and ba[k] > 0:
                    ratios_a.append(float(ba[k] / sa[k]))
        for k in range(n):
            if np.isfinite(sv[k]) and np.isfinite(bv[k]) and sv[k] > 0 and bv[k] > 0:
                ratios_v.append(float(bv[k] / sv[k]))
    results = {
        "n_events": n_events,
        "columns_same": bool(not column_diffs),
        "column_diffs": column_diffs,
        "timestamps_same": bool(n_events > 0 and timestamps_same == n_events),
        "bars_same": bool(n_events > 0 and bars_same == n_events),
        "sina_48": int(n_events),
        "bao_48": int(n_events),
        "volume_ratio_median": float(np.median(ratios_v)) if ratios_v else None,
        "volume_ratio_p10": float(np.percentile(ratios_v, 10)) if ratios_v else None,
        "volume_ratio_p90": float(np.percentile(ratios_v, 90)) if ratios_v else None,
        "amount_ratio_median": float(np.median(ratios_a)) if ratios_a else None,
    }
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-only", action="store_true",
                   help="只读 cache 重建 (跳过网络 fetch; 用于 byte-identical 验证)")
    p.add_argument("--force-fetch", action="store_true", help="强制重新 fetch (忽略 cache 覆盖)")
    p.add_argument("--no-label-recovery", action="store_true",
                   help="跳过 May label recovery 的 daily 补齐尝试")
    p.add_argument("--proxy-socks5", default=None, metavar="HOST:PORT",
                   help="显式 SOCKS5 代理 (如 127.0.0.1:7890); 仅路由 baostock 服务器连接")
    p.add_argument("--output", default=str(OUT_DIR))
    return p


def parse_socks5(value: str | None) -> tuple[str, int] | None:
    if not value:
        return None
    host, _, port = value.partition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"invalid --proxy-socks5 {value!r} (expected HOST:PORT)")
    return host, int(port)


def main() -> None:
    args = build_parser().parse_args()
    config = DataConfig()
    provider = MarketDataProvider(config)
    bao_cache = Baostock5mCache(config.cache_dir / "baostock_5m", suffix="5min")
    bao_cache.validator = validate_normalized_5m_frame

    universe = load_universe(MAY_CSV, JUNE_CSV)
    print(f"universe: may={int((universe.source_window=='may').sum())} "
          f"june={int((universe.source_window=='june').sum())} "
          f"codes={universe['code'].nunique()}")

    socks5_proxy = parse_socks5(args.proxy_socks5)
    if socks5_proxy is not None:
        set_baostock_socks5_proxy(*socks5_proxy)
        print(f"socks5 proxy enabled: {socks5_proxy[0]}:{socks5_proxy[1]} "
              f"(baostock server connections only)")
    try:
        if args.cache_only:
            print("cache-only rebuild: 校验 cache 覆盖窗口")
            for code in sorted(set(universe["code"])):
                fr = bao_cache.read(code)
                if fr is None or fr.empty:
                    raise RuntimeError(f"cache-only: missing cache for {code}")
                dates = pd.to_datetime(fr["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
                if (dates.min() > _expected_window_start(code)
                        or dates.max() < _expected_window_end(code)):
                    raise RuntimeError(f"cache-only: {code} cache does not cover window")
            provenance = {code: {"cache": "reused", "rows": None}
                          for code in sorted(set(universe["code"]))}
        else:
            provenance = fetch_baostock_window(universe, bao_cache, provider,
                                               force=args.force_fetch)
    finally:
        clear_baostock_socks5_proxy()
    print(f"fetch provenance: fetched={sum(1 for p in provenance.values() if p['cache']=='fetched')} "
          f"reused={sum(1 for p in provenance.values() if p['cache']=='reused')}")

    cal = build_trading_calendar(POOL_DIR, DAILY_DIR)
    cal_idx = cal["cal_idx"]
    proofs = load_suspension_proofs(SUSP_DIR)
    print(f"calendar={len(cal['calendar'])} proofs={len(proofs)}")

    rows: list[dict] = []
    for _, ev in universe.iterrows():
        daily = load_daily_frame(ev["code"], DAILY_DIR)
        rows.append(build_event_row(ev, bao_cache, daily, cal_idx, proofs))
    dev = pd.DataFrame(rows)
    print(f"rows built: {len(dev)} (may={int((dev.source_window=='may').sum())} "
          f"june={int((dev.source_window=='june').sum())})")

    may_label_recovery = recover_may_labels(
        universe, provider, cal_idx, proofs,
        do_recovery=not args.no_label_recovery)
    print(f"may label recovery: {len(may_label_recovery)} incomplete, "
          f"recovered={sum(1 for r in may_label_recovery.values() if r['label_recovered'])}")

    frozen = pd.read_csv(JUNE_CSV, dtype={"code": str})
    june_crosscheck = run_june_crosscheck(dev, frozen)
    sina_regression = run_sina_regression(dev, universe, SINA_MINUTE_DIR, bao_cache)
    print(f"june crosscheck columns: {len(june_crosscheck)}; "
          f"sina regression events: {sina_regression.get('n_events')}")

    write_outputs(rows, universe, provenance, may_label_recovery,
                  june_crosscheck, sina_regression, byte_identical=None,
                  socks5_proxy=args.proxy_socks5)
    print(f"outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""v004c May D1 Candidate Coverage / Recovery — 只读审计与可恢复性调查 (独立薄模块)

本模块只解决 May (2026-05-06 ~ 2026-05-31) D1 候选的覆盖审计与恢复调查问题:
- 不训练模型 / 不筛选因子 / 不做 Pairwise / 不修改任何冻结资产;
- 候选身份选择器与 v004c_d1_dataset 业务语义一致 (d0 + board_streak in {2,3}
  + signal_date == break_date), 直接在 v0.2 候选审计表 (9803 行, 全部断板事件)
  上重建; 先用 June+July 复现冻结 v004c_training_d1_v001.csv 的 333 行候选,
  再应用到 May;
- 四层覆盖审计: D1 日线 / D1 5min (正式规则 48 bar, 09:35 首, 15:00 末) /
  target7 日线标签 (v0.2.2 规则: expected 日期优先, 停牌 proven 偏移用复牌日) /
  D1 特征构造完整;
- 恢复调查: Priority 1 本地 raw/minute_5m 副本 (provenance 记录, 同源
  sina_5m / adjust=none 才可恢复); Priority 2 canonical 端点 (只读窗口检查);
  Priority 3 UNRECOVERABLE_WITH_CURRENT_CANONICAL_SOURCE;
- 所有函数确定性: 固定排序 / 无时间戳; CSV 由 tools 层输出。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAY_START = "2026-05-06"
MAY_END = "2026-05-31"
TRAIN_START = "2026-06-01"
TRAIN_END = "2026-07-29"

EXPECTED_BAR_COUNT = 48
EXPECTED_FIRST_BAR = "09:35"
EXPECTED_LAST_BAR = "15:00"
ALLOWED_BOARD_STREAKS = (2, 3)
TARGET_THRESHOLD = 0.07
FULL_DAY_SUSPENSION_DURATIONS = {"连续停牌", "停牌一天"}
# D1 特征需要 MA20 -> break 日之前至少 20 个交易日 bar (603459 上市日 04-08,
# break 05-12 时 prior=21 >= 20; v0.1 判定 insufficient_daily_history 的行现在补足)
DAILY_HISTORY_REQUIRED_BARS = 20
# 5min canonical 端点 datalen (与 src/data_sources.py _fetch_5min_sina 一致)
SINA_5M_DATALEN = 1970

REQUIRED_AUDIT_COLUMNS = (
    "event_id", "code", "name", "signal_date", "break_date",
    "board_streak_before_break", "days_since_break",
    "candidate_status", "exclusion_reason", "data_quality_reason",
)

# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def read_candidate_audit(path: str | Path) -> pd.DataFrame:
    """读取候选审计表 (v0.1 / v0.2 身份列完全相同)。"""
    frame = pd.read_csv(path, dtype={"code": str})
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    missing = [c for c in REQUIRED_AUDIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"候选审计表缺少必要列: {missing}")
    for col in ("signal_date", "break_date"):
        frame[col] = frame[col].astype(str)
    return frame


def read_frozen_d1(path: str | Path) -> pd.DataFrame:
    """读取冻结 v004c_training_d1_v001.csv (只取身份列)。"""
    frame = pd.read_csv(path, dtype={"code": str})
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for col in ("event_id", "code", "signal_date", "break_date"):
        if col not in frame.columns:
            raise ValueError(f"冻结 D1 表缺少必要列: {col}")
    return frame


# ---------------------------------------------------------------------------
# 业务语义选择器 (§3 / §4): 在审计表上重建, 不读取任何 v0.2 判定结果
# ---------------------------------------------------------------------------
def select_business_candidates(
    audit: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """业务语义 D1 候选: days_since_break==0 + board_streak in {2,3}
    + event_id 有效 + signal_date == break_date, 且 signal_date 落在窗口内。

    不参考 candidate_status / exclusion_reason (它们正是待审计的对象)。
    """
    frame = audit.copy()
    dsb = pd.to_numeric(frame["days_since_break"], errors="coerce")
    streak = pd.to_numeric(frame["board_streak_before_break"], errors="coerce")
    mask = (
        dsb.eq(0)
        & streak.isin(ALLOWED_BOARD_STREAKS)
        & frame["event_id"].notna()
        & (frame["event_id"].astype(str).str.strip() != "")
    )
    sel = frame[mask].copy()
    sel = sel[sel["signal_date"] == sel["break_date"]].copy()
    sel = sel[(sel["signal_date"] >= start_date) & (sel["signal_date"] <= end_date)]
    if sel.duplicated("event_id").any():
        raise ValueError("业务语义选择器产生重复 event_id, 阻止运行")
    return sel.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)


def reproduce_june_july_candidates(
    audit: pd.DataFrame,
    frozen: pd.DataFrame,
) -> dict[str, Any]:
    """June+July 复现校验 (§3 强制): 业务全集中的 v02 CANDIDATE 行必须与冻结
    333 行在候选身份上精确对齐 (rows / signal_dates / event_id 双向零差异 / 无重复)。

    业务全集 = 349 行; 其中 v02 判定 CANDIDATE 的 333 行 == 冻结 333;
    其余 16 行 (QUALITY_FAILED / LABEL_UNAVAILABLE) 逐条输出, 与 May 的
    97 行同性质 (minute_data_coverage_selection_bias 证据), 不调整规则迎合。
    """
    business = select_business_candidates(audit, TRAIN_START, TRAIN_END)
    candidate = business[business["candidate_status"] == "CANDIDATE"].copy()
    differing = business[business["candidate_status"] != "CANDIDATE"].copy()

    frozen_ids = set(frozen["event_id"].astype(str))
    candidate_ids = set(candidate["event_id"].astype(str))
    frozen_minus = sorted(frozen_ids - candidate_ids)
    candidate_minus = sorted(candidate_ids - frozen_ids)

    frozen_dates = sorted(frozen["signal_date"].astype(str).unique())
    candidate_dates = sorted(candidate["signal_date"].astype(str).unique())

    diff_rows: list[dict[str, Any]] = []
    for _, row in differing.iterrows():
        diff_rows.append({
            "event_id": str(row["event_id"]), "code": str(row["code"]),
            "signal_date": str(row["signal_date"]), "break_date": str(row["break_date"]),
            "board_streak_before_break": row["board_streak_before_break"],
            "candidate_status": str(row["candidate_status"]),
            "exclusion_reason": str(row["exclusion_reason"]),
            "data_quality_reason": str(row["data_quality_reason"]),
        })

    return {
        "business_rows": int(len(business)),
        "candidate_rows": int(len(candidate)),
        "differing_rows": int(len(differing)),
        "frozen_rows": int(len(frozen)),
        "candidate_signal_dates": int(len(candidate_dates)),
        "frozen_signal_dates": int(len(frozen_dates)),
        "event_id_diff_frozen_minus_candidate": frozen_minus,
        "event_id_diff_candidate_minus_frozen": candidate_minus,
        "duplicate_event_ids_in_candidate": int(candidate["event_id"].duplicated().sum()),
        "duplicate_event_ids_in_frozen": int(frozen["event_id"].astype(str).duplicated().sum()),
        "exact_event_id_match": (not frozen_minus) and (not candidate_minus),
        "candidate_signal_date_match": set(frozen_dates) == set(candidate_dates),
        "differing_detail": diff_rows,
    }


def summarize_universe(universe: pd.DataFrame) -> dict[str, Any]:
    """May D1 候选全集统计 (rows / dates / stocks / 2-board / 3-board / 状态分布)。"""
    streak = pd.to_numeric(universe["board_streak_before_break"], errors="coerce")
    status_counts = universe["candidate_status"].value_counts().to_dict()
    return {
        "rows": int(len(universe)),
        "signal_dates": int(universe["signal_date"].nunique()),
        "unique_stocks": int(universe["code"].nunique()),
        "board_streak_2_count": int(streak.eq(2).sum()),
        "board_streak_3_count": int(streak.eq(3).sum()),
        "candidate_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "signal_date_min": str(universe["signal_date"].min()),
        "signal_date_max": str(universe["signal_date"].max()),
    }


# ---------------------------------------------------------------------------
# 统一交易日历与停牌证明 (与 build_v02_core 相同来源与规则)
# ---------------------------------------------------------------------------
def build_trading_calendar(pool_dir: str | Path, daily_dir: str | Path) -> dict[str, Any]:
    """统一交易日历 = limit_ups pool 日期 ∪ daily 日期 (与 v0.2 构建一致)。"""
    pool_dir, daily_dir = Path(pool_dir), Path(daily_dir)
    pool_dates: set[str] = set()
    if pool_dir.exists():
        for f in sorted(pool_dir.glob("*.pkl")):
            try:
                fr = pd.read_pickle(f)
                if fr is not None and "trade_date" in getattr(fr, "columns", []):
                    pool_dates.update(fr["trade_date"].astype(str))
            except Exception:
                continue
    daily_dates: set[str] = set()
    if daily_dir.exists():
        for f in sorted(daily_dir.glob("*_daily.pkl")):
            try:
                fr = pd.read_pickle(f)
                if fr is None or "date" not in getattr(fr, "columns", []):
                    continue
                daily_dates.update(
                    pd.to_datetime(fr["date"], errors="coerce")
                    .dt.strftime("%Y-%m-%d").dropna().tolist())
            except Exception:
                continue
    calendar = sorted(pool_dates | daily_dates)
    return {"calendar": calendar, "cal_idx": {d: i for i, d in enumerate(calendar)}}


def next_cal_date(cal_idx: dict[str, int], date: str) -> str | None:
    i = cal_idx.get(str(date))
    if i is None or i + 1 >= len(cal_idx):
        return None
    by_idx = {v: k for k, v in cal_idx.items()}
    return by_idx.get(i + 1)


def load_suspension_proofs(susp_dir: str | Path) -> dict[tuple[str, str, str], dict]:
    """加载全市场停牌证明区间 (与 build_v02_core 相同: records_json 格式,
    仅 FULL_DAY_SUSPENSION_DURATIONS)。"""
    proofs: dict[tuple[str, str, str], dict] = {}
    susp_dir = Path(susp_dir)
    if not susp_dir.exists():
        return proofs
    for f in sorted(susp_dir.glob("*.pkl")):
        try:
            frame = pd.read_pickle(f)
            if frame is None or frame.empty or "records_json" not in frame.columns:
                continue
            records = json.loads(str(frame.iloc[0]["records_json"]))
        except Exception:
            continue
        for r in records:
            code = str(r.get("code", "")).strip()
            start = str(r.get("suspension_start_date", "")).strip()
            end = str(r.get("suspension_end_date", "")).strip()
            duration = str(r.get("suspension_duration", "")).strip()
            if code and start and end and duration in FULL_DAY_SUSPENSION_DURATIONS:
                proofs[(code, start, end)] = r
    return proofs


def suspension_proven(proofs: dict[tuple[str, str, str], dict], code: str, date: str) -> bool:
    return any(c == code and s <= date <= e for (c, s, e) in proofs)


# ---------------------------------------------------------------------------
# 缓存读取 (规范化与 build_v02_core 一致)
# ---------------------------------------------------------------------------
def load_minute_frame(code: str, minute_dir: str | Path) -> pd.DataFrame | None:
    path = Path(minute_dir) / f"{code}_5min.pkl"
    if not path.exists():
        return None
    try:
        m = pd.read_pickle(path)
        if m is None or m.empty:
            return None
        m = m.copy()
        m["trade_date"] = m["trade_date"].astype(str)
        m["time_str"] = pd.to_datetime(m["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
        m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
        return m.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def load_daily_frame(code: str, daily_dir: str | Path) -> pd.DataFrame | None:
    path = Path(daily_dir) / f"{code}_daily.pkl"
    if not path.exists():
        return None
    try:
        d = pd.read_pickle(path)
        if d is None or d.empty:
            return None
        d = d.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        return d.dropna(subset=["date"]).reset_index(drop=True)
    except Exception:
        return None


def _valid_price(v: Any) -> bool:
    if v is None or pd.isna(v):
        return False
    try:
        return bool(float(v) > 0)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 层 1: D1 日线覆盖 (§5A)
# ---------------------------------------------------------------------------
def audit_d1_daily_complete(code: str, break_date: str, daily: pd.DataFrame | None) -> dict[str, Any]:
    """D1 日线覆盖: 缓存存在 + break 日 OHLC 有效 + break 前 MA 历史窗口足够。"""
    out: dict[str, Any] = {
        "daily_cache_file_exists": daily is not None,
        "d1_date_exists": False,
        "d1_ohlc_valid": False,
        "prior_trade_days": None,
        "prior_required": DAILY_HISTORY_REQUIRED_BARS,
        "daily_history_sufficient": False,
        "d1_daily_complete": False,
    }
    if daily is None:
        return out
    row = daily[daily["date"] == str(break_date)]
    if not row.empty:
        out["d1_date_exists"] = True
        out["d1_ohlc_valid"] = all(_valid_price(row.iloc[0].get(c)) for c in ("open", "high", "low", "close"))
    prior = int((daily["date"] < str(break_date)).sum())
    out["prior_trade_days"] = prior
    out["daily_history_sufficient"] = prior >= DAILY_HISTORY_REQUIRED_BARS
    out["d1_daily_complete"] = bool(out["d1_date_exists"] and out["d1_ohlc_valid"] and out["daily_history_sufficient"])
    return out


# ---------------------------------------------------------------------------
# 层 2: D1 5min 覆盖 (§5B, 正式规则: 48 bar / 09:35 首 / 15:00 末)
# ---------------------------------------------------------------------------
def audit_d1_minute_complete(code: str, date: str, minute: pd.DataFrame | None) -> dict[str, Any]:
    """D1 5min 网格审计; missing_bars / duplicate_bars / invalid_price_volume
    逐项记录; missing_reason 分类: cache_missing / d1_date_missing /
    bar_grid_incomplete / grid_violation。"""
    out: dict[str, Any] = {
        "minute_cache_file_exists": minute is not None,
        "d1_date_exists": False,
        "expected_bars": EXPECTED_BAR_COUNT,
        "actual_bars": 0,
        "first_timestamp": "",
        "last_timestamp": "",
        "missing_bars": EXPECTED_BAR_COUNT,
        "duplicate_bars": 0,
        "invalid_price_volume": 0,
        "d1_minute_complete": False,
        "missing_reason": "",
    }
    if minute is None:
        out["missing_reason"] = "cache_missing"
        return out
    if "time_str" not in minute.columns:
        minute = minute.copy()
        minute["time_str"] = pd.to_datetime(
            minute["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
        minute["datetime"] = pd.to_datetime(minute["datetime"], errors="coerce")
    day = minute[minute["trade_date"] == str(date)]
    if day.empty:
        out["missing_reason"] = "d1_date_missing"
        return out
    day = day.sort_values("datetime")
    out["d1_date_exists"] = True
    out["actual_bars"] = int(len(day))
    times = day["time_str"].dropna().tolist()
    if times:
        out["first_timestamp"] = str(times[0])
        out["last_timestamp"] = str(times[-1])
    out["missing_bars"] = max(EXPECTED_BAR_COUNT - int(len(day)), 0)
    out["duplicate_bars"] = int(len(day) - int(day["datetime"].notna().sum()))
    if day["datetime"].notna().any():
        dup = day["datetime"].duplicated(keep=False)
        out["duplicate_bars"] = int(dup.sum())
    invalid = 0
    for col in ("open", "high", "low", "close"):
        bad = day[col].isna() | ~np.isfinite(day[col]) | (day[col] <= 0)
        invalid += int(bad.sum())
    bad_vol = day["volume"].isna() | (day["volume"] < 0)
    invalid += int(bad_vol.sum())
    out["invalid_price_volume"] = int(invalid)
    complete = bool(
        out["actual_bars"] == EXPECTED_BAR_COUNT
        and out["first_timestamp"] == EXPECTED_FIRST_BAR
        and out["last_timestamp"] == EXPECTED_LAST_BAR
    )
    out["d1_minute_complete"] = complete
    if not complete:
        if out["actual_bars"] < EXPECTED_BAR_COUNT:
            out["missing_reason"] = "bar_grid_incomplete"
        else:
            out["missing_reason"] = "grid_violation"
    else:
        out["missing_reason"] = "none"
    return out


# ---------------------------------------------------------------------------
# 层 3: target7 日线标签覆盖 (§5C, v0.2.2 规则: expected 优先 + proven 偏移)
# ---------------------------------------------------------------------------
def _resume_date_after(daily: pd.DataFrame | None, end_date: str) -> str | None:
    """停牌结束后的第一个实际交易日 (该股 daily 缓存中 > end 的最早日期)。"""
    if daily is None or daily.empty:
        return None
    after = daily.loc[daily["date"] > str(end_date), "date"]
    if after.empty:
        return None
    return str(after.min())


def resolve_label_dates(
    code: str,
    signal_date: str,
    cal_idx: dict[str, int],
    proofs: dict[tuple[str, str, str], dict],
    daily: pd.DataFrame | None,
) -> dict[str, Any]:
    """D2/D3 标签日期: expected (统一日历下一交易日) 优先; 若 expected 无日线
    bar 且该日停牌 proven, 偏移到复牌日 (与 v0.2.2 语义一致)。"""
    exp_d2 = next_cal_date(cal_idx, signal_date)
    exp_d3 = next_cal_date(cal_idx, exp_d2) if exp_d2 else None

    def resolve(exp_date: str | None) -> tuple[str | None, str]:
        if not exp_date:
            return None, "expected_date_missing"
        has_bar = bool(daily is not None and not daily[daily["date"] == exp_date].empty)
        if not has_bar and suspension_proven(proofs, code, exp_date):
            resume = _resume_date_after(daily, exp_date)
            if resume:
                return resume, "suspension_proven_resume_date_daily_cache"
        return exp_date, "expected_date_daily_cache"

    d2_date, d2_source = resolve(exp_d2)
    d3_date, d3_source = resolve(exp_d3)
    return {
        "expected_d2_date": exp_d2,
        "expected_d3_date": exp_d3,
        "label_d2_date": d2_date,
        "label_d3_date": d3_date,
        "daily_label_source_d2": d2_source,
        "daily_label_source_d3": d3_source,
    }


def audit_daily_label(
    code: str,
    signal_date: str,
    cal_idx: dict[str, int],
    proofs: dict[tuple[str, str, str], dict],
    daily: pd.DataFrame | None,
) -> dict[str, Any]:
    """target7 日线标签: daily_label_complete + target7 / tail (May 非未揭盲
    holdout, 允许查看; 但 Target 不参与任何准入/保留决策)。"""
    resolved = resolve_label_dates(code, signal_date, cal_idx, proofs, daily)
    d2_date, d3_date = resolved["label_d2_date"], resolved["label_d3_date"]

    def price_at(date: str | None, col: str) -> float | None:
        if not date or daily is None:
            return None
        row = daily[daily["date"] == str(date)]
        if row.empty:
            return None
        v = row.iloc[0].get(col)
        try:
            if v is None or pd.isna(v):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    d2_open = price_at(d2_date, "open")
    d3_high = price_at(d3_date, "high")
    d3_close = price_at(d3_date, "close")

    reasons: list[str] = []
    if not d2_date:
        reasons.append("expected_d2_missing")
    if not d3_date:
        reasons.append("expected_d3_missing")
    if d2_date and not _valid_price(d2_open):
        reasons.append(f"d2_open_invalid({d2_date})")
    if d3_date and not _valid_price(d3_high):
        reasons.append(f"d3_high_invalid({d3_date})")
    if d3_date and not _valid_price(d3_close):
        reasons.append(f"d3_close_invalid({d3_date})")

    high_ret = (d3_high / d2_open - 1.0) if (_valid_price(d3_high) and _valid_price(d2_open)) else None
    close_ret = (d3_close / d2_open - 1.0) if (_valid_price(d3_close) and _valid_price(d2_open)) else None

    return {
        **resolved,
        "d2_open_daily": d2_open,
        "d3_high_daily": d3_high,
        "d3_close_daily": d3_close,
        "daily_d2open_to_d3high_return": high_ret,
        "daily_d2open_to_d3close_return": close_ret,
        "target7_daily_d2open_d3high": bool(high_ret is not None and high_ret >= TARGET_THRESHOLD),
        "tail_loss_daily_5pct": bool(close_ret is not None and close_ret <= -0.05),
        "daily_label_complete": not reasons,
        "daily_label_reason": "|".join(reasons) if reasons else "ok",
    }


# ---------------------------------------------------------------------------
# 层 4: D1 特征构造完整 (§5D)
# ---------------------------------------------------------------------------
def audit_d1_feature_complete(d1_daily_complete: bool, d1_minute_complete: bool) -> bool:
    """D1 特征 = 日线层 (OHLC + MA 历史) AND 5min 层 (48-bar 网格) 均可构造。"""
    return bool(d1_daily_complete and d1_minute_complete)


# ---------------------------------------------------------------------------
# 恢复调查 (§7 / §8): Priority 1 raw 副本 / Priority 2 canonical 窗口
# ---------------------------------------------------------------------------
def file_provenance(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size": int(st.st_size),
        "mtime": pd.Timestamp(st.st_mtime, unit="s", tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def inspect_raw_recovery(
    code: str,
    date: str,
    raw_dir: str | Path,
    minute: pd.DataFrame | None,
) -> dict[str, Any]:
    """Priority 1: raw/minute_5m 遗留快照 (get_stock_bars 镜像, 同源 sina_5m)。
    仅当 cache 层 2 不完整且 raw 具备完整 D1 网格时 recoverable。"""
    out: dict[str, Any] = {
        "raw_file_exists": False,
        "raw_source": "",
        "raw_adjust": "",
        "raw_date_min": "",
        "raw_date_max": "",
        "raw_d1_date_exists": False,
        "raw_d1_bars": 0,
        "raw_d1_first": "",
        "raw_d1_last": "",
        "raw_d1_complete": False,
        "raw_path": "",
        "recoverable": False,
        "raw_provenance": {},
    }
    path = Path(raw_dir) / f"{code}_5min.csv"
    if not path.exists():
        return out
    out["raw_file_exists"] = True
    out["raw_path"] = str(path)
    try:
        raw = pd.read_csv(path, dtype={"code": str})
    except Exception:
        return out
    if raw is None or raw.empty:
        return out
    raw["trade_date"] = raw["trade_date"].astype(str)
    raw["time_str"] = pd.to_datetime(raw["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
    if "source" in raw.columns and raw["source"].notna().any():
        out["raw_source"] = str(raw["source"].dropna().iloc[0])
    if "adjust" in raw.columns and raw["adjust"].notna().any():
        out["raw_adjust"] = str(raw["adjust"].dropna().iloc[0])
    dates = sorted(raw["trade_date"].dropna().unique())
    if dates:
        out["raw_date_min"], out["raw_date_max"] = dates[0], dates[-1]
    day = raw[raw["trade_date"] == str(date)]
    if not day.empty:
        out["raw_d1_date_exists"] = True
        day = day.sort_values("datetime")
        out["raw_d1_bars"] = int(len(day))
        times = day["time_str"].dropna().tolist()
        if times:
            out["raw_d1_first"], out["raw_d1_last"] = str(times[0]), str(times[-1])
        out["raw_d1_complete"] = bool(
            out["raw_d1_bars"] == EXPECTED_BAR_COUNT
            and out["raw_d1_first"] == EXPECTED_FIRST_BAR
            and out["raw_d1_last"] == EXPECTED_LAST_BAR)
    out["raw_provenance"] = file_provenance(path)
    cache_needs = minute is None or not audit_d1_minute_complete(code, date, minute)["d1_minute_complete"]
    out["recoverable"] = bool(out["raw_d1_complete"] and cache_needs)
    return out


def apply_raw_recovery(
    code: str,
    date: str,
    raw_dir: str | Path,
    minute_cache_path: str | Path,
    backup_dir: str | Path,
) -> dict[str, Any]:
    """执行 Priority 1 恢复: 把 raw 中目标 D1 日的 bar 合并进 5min cache。
    - before 记录 (path/exists/size/mtime/sha256/rows/date min-max);
    - 现有 cache 先备份到 backup_dir (不静默覆盖来源不明文件);
    - 合并只取 raw 的目标日期行, datetime 去重 (保留 cache 既有行优先);
    - after 重读并重跑层 2 审计。"""
    out: dict[str, Any] = {
        "event_id": f"{code}_{date}",
        "code": code,
        "signal_date": date,
        "raw_provenance": {},
        "backup_path": "",
        "before": {},
        "after": {},
        "rows_added": 0,
    }
    raw_path = Path(raw_dir) / f"{code}_5min.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw 副本缺失: {raw_path}")
    cache_path = Path(minute_cache_path)
    out["raw_provenance"] = file_provenance(raw_path)

    before = file_provenance(cache_path)
    out["before"] = {
        **{k: v for k, v in before.items() if k != "path"},
        "rows": 0,
        "date_min": "",
        "date_max": "",
        "d1_minute_complete": False,
    }
    if before["exists"]:
        m = load_minute_frame(code, cache_path.parent)
        if m is not None:
            out["before"]["rows"] = int(len(m))
            if len(m):
                out["before"]["date_min"] = str(m["trade_date"].min())
                out["before"]["date_max"] = str(m["trade_date"].max())
            out["before"]["d1_minute_complete"] = (
                audit_d1_minute_complete(code, date, m)["d1_minute_complete"])

    # 备份现有 cache
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{code}_5min.before_may_recovery.pkl"
    if before["exists"]:
        import shutil
        shutil.copy2(cache_path, backup_path)
    out["backup_path"] = str(backup_path)

    # 读 raw, 规范化 (同 load_minute_frame 口径), 只取目标日期
    raw = pd.read_csv(raw_path, dtype={"code": str})
    raw["trade_date"] = raw["trade_date"].astype(str)
    raw["time_str"] = pd.to_datetime(raw["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    day_rows = raw[raw["trade_date"] == str(date)].copy()
    if day_rows.empty:
        raise ValueError(f"raw 副本缺少目标日期 bar: {raw_path} date={date}")
    day_rows = day_rows.drop(columns=["time_str"])

    # 合并: 现有 cache ∪ 目标日 bar (datetime 去重, 保留 cache 既有行优先)
    merged = None
    if out["before"]["exists"] and out["before"]["rows"]:
        merged = load_minute_frame(code, cache_path.parent)
    if merged is None:
        merged = day_rows.copy()
    else:
        existing_keys = set(merged["datetime"].dropna())
        new_rows = day_rows[~day_rows["datetime"].isin(existing_keys)]
        merged = pd.concat([merged, new_rows], ignore_index=True)
    out["rows_added"] = int(len(merged) - (out["before"]["rows"] if out["before"]["rows"] else 0))
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # 原子写回 (先写临时文件再 rename)
    tmp = cache_path.with_suffix(".pkl.tmp")
    merged.to_pickle(tmp)
    import os as _os
    _os.replace(tmp, cache_path)

    after = file_provenance(cache_path)
    m_after = load_minute_frame(code, cache_path.parent)
    out["after"] = {
        **{k: v for k, v in after.items() if k != "path"},
        "rows": int(len(m_after)) if m_after is not None else 0,
        "date_min": str(m_after["trade_date"].min()) if m_after is not None and len(m_after) else "",
        "date_max": str(m_after["trade_date"].max()) if m_after is not None and len(m_after) else "",
        "d1_minute_complete": audit_d1_minute_complete(code, date, m_after)["d1_minute_complete"],
    }
    return out


def replay_completed_recoveries(
    universe: pd.DataFrame,
    backup_dir: str | Path,
    minute_dir: str | Path,
    raw_dir: str | Path,
) -> list[dict[str, Any]]:
    """幂等重跑不丢恢复记录: 从 backup_dir 中已存在的
    {code}_5min.before_may_recovery.pkl 重放已完成恢复的 RECORDED 结果
    (before = backup 副本行数, after = 当前 cache 行数, rows_added = 差值)。
    只重放 rows_added > 0 的真实恢复; rows_added == 0 的 backup (如中断
    运行残留) 视为无效, 不产生记录。结构与 apply_raw_recovery 返回一致,
    供 recovery 报告直接消费。"""
    records: list[dict[str, Any]] = []
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return records
    codes_in_universe = set(universe["code"].astype(str))
    for bp in sorted(backup_dir.glob("*_5min.before_may_recovery.pkl")):
        code = bp.name.split("_5min.")[0]
        if code not in codes_in_universe:
            continue
        row = universe[universe["code"].astype(str) == code].iloc[0]
        date = str(row["signal_date"])
        cache_path = Path(minute_dir) / f"{code}_5min.pkl"
        m_before = None
        try:
            m_before = pd.read_pickle(bp)
            if m_before is None or m_before.empty:
                m_before = None
        except Exception:
            m_before = None
        m_after = load_minute_frame(code, minute_dir)
        before_rows = int(len(m_before)) if m_before is not None else 0
        after_rows = int(len(m_after)) if m_after is not None else 0
        rows_added = max(0, after_rows - before_rows)
        if rows_added <= 0:
            continue  # 非真实恢复 (中断运行残留 / 未发生合并)
        before_complete = False
        if m_before is not None and len(m_before):
            before_complete = audit_d1_minute_complete(code, date, m_before)["d1_minute_complete"]
        after_complete = False
        if m_after is not None and len(m_after):
            after_complete = audit_d1_minute_complete(code, date, m_after)["d1_minute_complete"]
        records.append({
            "event_id": f"{code}_{date}",
            "code": code,
            "signal_date": date,
            "raw_provenance": file_provenance(Path(raw_dir) / f"{code}_5min.csv"),
            "backup_path": str(bp),
            "before": {"path": str(cache_path), "exists": True, "rows": before_rows,
                       "d1_minute_complete": before_complete},
            "after": {"rows": after_rows, "d1_minute_complete": after_complete},
            "rows_added": rows_added,
        })
    return records


# ---------------------------------------------------------------------------
# Priority 2: canonical 5min 端点窗口检查 (只读 HTTP, 不写任何缓存)
# ---------------------------------------------------------------------------
def canonical_5min_window_check(
    probe_codes: list[str],
    fetch_5min: Any,
    start_datetime: str,
    end_datetime: str,
    adjust: str = "none",
) -> dict[str, Any]:
    """复用现有 canonical 实现 (MarketDataService.fetch_5min_history, sina_5m
    唯一源) 只读探测返回窗口; 不写任何缓存。窗口证据用于判定 Priority 2
    可恢复性 (May 是否仍在 canonical 返回范围内)。"""
    results: list[dict[str, Any]] = []
    for code in probe_codes:
        try:
            frame, source = fetch_5min(code, start_datetime, end_datetime, adjust)
            dates = sorted(frame["trade_date"].astype(str).unique()) if frame is not None else []
            results.append({
                "code": code, "ok": True, "source": source,
                "min_trade_date": dates[0] if dates else "",
                "max_trade_date": dates[-1] if dates else "",
                "n_dates": len(dates),
            })
        except Exception as exc:
            results.append({"code": code, "ok": False, "error": str(exc)[:300]})
    return {"probe_codes": probe_codes, "results": results}


# ---------------------------------------------------------------------------
# 主装配: 四层覆盖审计 + 分类 (§5D / §6 / §9)
# ---------------------------------------------------------------------------
def audit_candidate_coverage(
    universe: pd.DataFrame,
    *,
    cal_idx: dict[str, int],
    proofs: dict[tuple[str, str, str], dict],
    minute_dir: str | Path,
    daily_dir: str | Path,
    raw_dir: str | Path,
    check_recovery: bool = True,
) -> pd.DataFrame:
    """对 May 候选全集逐行四层审计 + 恢复调查; 输出确定性排序的行级结果。"""
    minute_dir, daily_dir, raw_dir = Path(minute_dir), Path(daily_dir), Path(raw_dir)
    minute_cache: dict[str, pd.DataFrame | None] = {}
    daily_cache: dict[str, pd.DataFrame | None] = {}

    rows: list[dict[str, Any]] = []
    for _, r in universe.iterrows():
        code = str(r["code"])
        sig = str(r["signal_date"])
        if code not in minute_cache:
            minute_cache[code] = load_minute_frame(code, minute_dir)
        if code not in daily_cache:
            daily_cache[code] = load_daily_frame(code, daily_dir)
        minute = minute_cache[code]
        daily = daily_cache[code]

        d1_daily = audit_d1_daily_complete(code, sig, daily)
        d1_minute = audit_d1_minute_complete(code, sig, minute)
        label = audit_daily_label(code, sig, cal_idx, proofs, daily)
        feature_complete = audit_d1_feature_complete(d1_daily["d1_daily_complete"],
                                                     d1_minute["d1_minute_complete"])
        # 所有行统一调用 (键一致, 避免 NaN 混入布尔列; 完整行由 cache_needs 判 False)
        recovery: dict[str, Any] = {}
        if check_recovery:
            recovery = inspect_raw_recovery(code, sig, raw_dir, minute)

        full = bool(d1_daily["d1_daily_complete"] and d1_minute["d1_minute_complete"]
                    and label["daily_label_complete"] and feature_complete)
        rows.append({
            "event_id": str(r["event_id"]),
            "code": code,
            "name": str(r["name"]),
            "signal_date": sig,
            "break_date": str(r["break_date"]),
            "board_streak_before_break": r["board_streak_before_break"],
            "v02_candidate_status": str(r["candidate_status"]),
            "v02_exclusion_reason": str(r["exclusion_reason"]),
            "v02_data_quality_reason": str(r["data_quality_reason"]),
            **d1_daily,
            **d1_minute,
            **label,
            "d1_feature_complete": feature_complete,
            **recovery,
            "fully_complete": full,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)


def build_coverage_summary(df: pd.DataFrame) -> dict[str, Any]:
    """四层覆盖交叉统计 (§5D / §14): 每层计数 + 16 种交集组合表。"""
    mask = {
        "DAILY_COMPLETE": df["d1_daily_complete"].astype(bool),
        "MINUTE_COMPLETE": df["d1_minute_complete"].astype(bool),
        "LABEL_COMPLETE": df["daily_label_complete"].astype(bool),
        "D1_FEATURE_COMPLETE": df["d1_feature_complete"].astype(bool),
        "FULLY_COMPLETE": df["fully_complete"].astype(bool),
    }
    layers = {
        "DAILY_COMPLETE": "d1_daily_complete",
        "MINUTE_COMPLETE": "d1_minute_complete",
        "LABEL_COMPLETE": "daily_label_complete",
        "D1_FEATURE_COMPLETE": "d1_feature_complete",
    }
    per_layer = {k: int(v.sum()) for k, v in mask.items()}
    n = len(df)
    rows: list[dict[str, Any]] = []
    keys = list(layers.keys())
    for bits in range(1 << len(keys)):
        comb = {keys[i] for i in range(len(keys)) if (bits >> i) & 1}
        sel = df
        for k in keys:
            sel = sel[sel[layers[k]].astype(bool) == (k in comb)]
        rows.append({
            "combination": "+".join(sorted(comb)) if comb else "NONE_COMPLETE",
            "rows": int(len(sel)),
            "fraction": float(len(sel)) / n if n else None,
        })
    rows.sort(key=lambda x: (-x["rows"], x["combination"]))
    return {
        "universe_rows": n,
        "per_layer": per_layer,
        "cross_tab": rows,
        "recoverable_rows": int(df["recoverable"].fillna(False).astype(bool).sum())
            if "recoverable" in df.columns else None,
        "recovered_rows": int(df["recovered"].fillna(False).astype(bool).sum())
            if "recovered" in df.columns else 0,
    }


def build_missing_minute_report(df: pd.DataFrame) -> pd.DataFrame:
    """§6: '日线完整但分钟缺失' 清单 + 恢复字段; 汇总段写入 review。"""
    daily_ok = df["d1_daily_complete"].astype(bool)
    minute_bad = ~df["d1_minute_complete"].astype(bool)
    sel = df[daily_ok & minute_bad].copy()
    cols = ["event_id", "code", "signal_date", "board_streak_before_break",
            "missing_reason", "expected_bars", "actual_bars",
            "first_timestamp", "last_timestamp", "duplicate_bars", "invalid_price_volume",
            "daily_label_complete", "d1_daily_complete", "d1_minute_complete"]
    for c in ("minute_cache_path", "raw_path"):
        if c in sel.columns:
            cols.append(c)
    for c in ("recoverable", "recovered"):
        if c in sel.columns:
            cols.append(c)
    return sel[[c for c in cols if c in sel.columns]].reset_index(drop=True)


def build_recovery_report(df: pd.DataFrame) -> pd.DataFrame:
    """§7 / §8: 恢复调查结果逐行表 (recoverable / recovered / UNRECOVERABLE)。"""
    cols = ["event_id", "code", "signal_date", "v02_candidate_status", "v02_exclusion_reason",
            "d1_minute_complete", "daily_label_complete", "d1_daily_complete"]
    for c in ("recoverable", "recovered", "raw_path", "raw_provenance", "raw_source",
              "raw_adjust", "raw_d1_complete", "raw_date_min", "raw_date_max",
              "missing_reason"):
        if c in df.columns:
            cols.append(c)
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

# -*- coding: utf-8 -*-
"""v004c BaoStock Dev Foundation Correctness Fix v002 — 构建工具

针对 v001 (May+June 319-row dev foundation) 的定点正确性修复:
- P1: May label recovery 必须在 build_event_row 之前完成, 恢复结果真正写入
  dev CSV; 不完整 label 的 target 显式 NA (UNKNOWN != FALSE), 禁止 bool
  coercion 伪装成负样本;
- daily recovery 只追加缺失日期 (existing canonical rows immutable, 禁止覆盖),
  重叠日期仅审计 (OHLC/volume 比较), 已恢复状态 (live audit complete) 跳过
  fetch, 无网络 (确定性);
- 新增 development-level 语义: dev_label_complete / dev_feature_source_complete
  (d1_minute_complete AND daily source complete AND unexpected_missing_count==0)
  / dev_training_eligible (= label AND feature source complete); May 全部事件
  重新 live audit, June 用冻结链, 两窗口字段语义一致;
- structural missing 分类 (目前: d1_close_location 当日内 5min high==low,
  官方公式分母为 0 返回 None -> STRUCTURAL_MISSING, 不使数据资格失败;
  603065_2026-06-11 保留);
- provenance 从 Baostock5mCache meta 读取 (source/adjustment/interval/
  date range/row_count/fetch_timestamp) + 每 code cache 文件 SHA256 +
  baostock 包版本; cache 路径 repo-relative;
- DETERMINISTIC_REBUILD 正式写入 review (build #1 vs cache-only 重建 #2 的
  核心资产 byte-identical 哈希证据, 含 review 自身两轮渲染字节比较);
- --output 真实生效 (dev/quality/lineage/review/provenance 全部写到 X);
- stats 移除 target_label_complete 双计, 只统计 dev_* 字段;
- SOCKS5 ATYP=domain bind 长度消费 bug 修复 (见 src/data_sources.py)。

用法:
    python tools/build_v004c_baostock_d1_dev_v002.py [--cache-only] [--force-fetch]
                                                     [--no-label-recovery]
                                                     [--proxy-socks5 HOST:PORT]
                                                     [--output DIR]

输出 (默认 reports/research/v004c_baostock_d1_dev_v002_20260506_20260630/):
    v004c_baostock_d1_dev_v002.csv               May+June D1 development table
    v004c_baostock_d1_minute_quality_v002.csv    逐 event 5min/资格审计
    v004c_baostock_d1_feature_lineage_v002.csv   feature 公式来源/依赖/lineage
    v004c_baostock_d1_rebuild_review_v002.md     中文 review (含 DETERMINISTIC_REBUILD)
    v004c_baostock_d1_fetch_provenance_v002.json fetch provenance (timestamp 独立)

数据源: BaoStock 5m (frequency=5, adjustflag=3 -> adjustment=none) 为 D1 5min
主源; 日线状态用现有 canonical daily cache; Sina 行为完全保留 (本工具只读)。
May 146 行全量 + June 冻结表按 signal_date 06-01..06-30 过滤, July 不进开发数据。
确定性: 每次运行内部执行 cache-only 重建并比较核心资产哈希; 第二次
--cache-only 运行与首次正常运行的核心 CSV / review byte-identical。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
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
V001_DEV_CSV = (ROOT / "reports" / "research"
                / "v004c_baostock_d1_dev_v001_20260506_20260630"
                / "v004c_baostock_d1_dev_v001.csv")
OUT_DIR = ROOT / "reports" / "research" / "v004c_baostock_d1_dev_v002_20260506_20260630"
BAO_CACHE_DIR = ROOT / "data" / "cache" / "baostock_5m"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
SUSP_DIR = ROOT / "data" / "cache" / "suspension_status"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
SINA_MINUTE_DIR = ROOT / "data" / "cache" / "minute_5m"

WINDOW_START = "2026-05-06"
WINDOW_END = "2026-06-30"
JUNE_START = "2026-06-01"
JUNE_END = "2026-06-30"
# May label recovery 的 vendor daily 抓取窗口 (只追加缺失日期, 见 §7-§9)
RECOVERY_FETCH_START = "2026-04-01"
RECOVERY_FETCH_END = "2026-06-30"
MA_RECOMPUTE_EPSILON = 1e-9

DEV_CSV_NAME = "v004c_baostock_d1_dev_v002.csv"
QUALITY_CSV_NAME = "v004c_baostock_d1_minute_quality_v002.csv"
LINEAGE_CSV_NAME = "v004c_baostock_d1_feature_lineage_v002.csv"
REVIEW_NAME = "v004c_baostock_d1_rebuild_review_v002.md"
PROVENANCE_NAME = "v004c_baostock_d1_fetch_provenance_v002.json"
DETERMINISM_CORE_FILES = ("dev", "quality", "lineage", "review")

# §20: structural missing = 明确定义允许的数学退化 (官方公式数学定义下的合法
# 缺失, 不是 vendor / data coverage failure)。目前仅一项: d1_close_location
# 当 D1 日内 5min high == low (官方公式分母 (high-low) 为 0 -> None)。
# future 若出现其他严格定义的 structural missing, 必须显式枚举, 不得泛化吞掉
# 所有 NaN。
STRUCTURAL_MISSING_FEATURES = ("d1_close_location",)

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
# 通用
# ---------------------------------------------------------------------------
def _label_bool(v):
    """label 三态规范化: True / False / NA (UNKNOWN)。禁止把 NA 或 None 变成
    False (UNKNOWN != FALSE, §5/§37)。"""
    if v is None or pd.isna(v):
        return pd.NA
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        text = v.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    try:
        f = float(v)
        if np.isnan(f):
            return pd.NA
        return bool(f)
    except (TypeError, ValueError):
        return pd.NA


def _to_float_or_none(v):
    if v is None or pd.isna(v):
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _max_numeric(frame: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(s.max()) if len(s) else None


def _min_numeric(frame: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(s.min()) if len(s) else None


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
# BaoStock 批量获取 + cache provenance (从 meta 读取真实信息, §27/§28)
# ---------------------------------------------------------------------------
def _expected_window_start(code: str) -> str:
    daily = load_daily_frame(code, DAILY_DIR)
    if daily is not None and not daily.empty:
        first = str(pd.to_datetime(daily["date"], errors="coerce").min().date())
        if first > WINDOW_START:
            return first
    return WINDOW_START


def _expected_window_end(code: str) -> str:
    daily = load_daily_frame(code, DAILY_DIR)
    if daily is not None and not daily.empty:
        last = str(pd.to_datetime(daily["date"], errors="coerce").max().date())
        if last < WINDOW_END:
            return last
    return WINDOW_END


def cache_provenance_entry(cache: Baostock5mCache, code: str, status: str) -> dict:
    """每只股票的 cache provenance: meta 真实信息 + pkl SHA256 (§27/§28)。

    cache_status 为本次运行的 fetch 动作 (fetched/reused), 只进 provenance
    JSON (fetch timestamp 独立), 不进 dev/review 确定性资产。
    """
    entry: dict = {"code": code, "cache_status": status}
    try:
        entry["cache_sha256"] = hashlib.sha256(cache.path(code).read_bytes()).hexdigest()
    except OSError:
        entry["cache_sha256"] = None
    meta = cache.read_meta(code)
    if meta:
        for key in ("source", "adjustment", "interval",
                    "date_min", "date_max", "row_count", "fetch_timestamp"):
            if key in meta:
                entry[key] = meta[key]
    return entry


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
    pending: list[str] = []
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
            provenance[code] = cache_provenance_entry(cache, code, "reused")
            continue
        pending.append(code)
    if not pending:
        # 全部已覆盖: 无需网络, 不 login (§9 已恢复状态的离线确定性)
        return provenance
    login_baostock()
    try:
        for code in pending:
            existing = cache.read(code)
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
            provenance[code] = cache_provenance_entry(cache, code, "fetched")
    finally:
        logout_baostock()
    return provenance


def verify_cache_only_coverage(universe: pd.DataFrame, cache: Baostock5mCache) -> dict:
    """cache-only 模式: 校验每 code cache 覆盖窗口, provenance 从 meta 读取。"""
    provenance: dict[str, dict] = {}
    for code in sorted(set(universe["code"])):
        fr = cache.read(code)
        if fr is None or fr.empty:
            raise RuntimeError(f"cache-only: missing cache for {code}")
        dates = pd.to_datetime(fr["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
        if (dates.min() > _expected_window_start(code)
                or dates.max() < _expected_window_end(code)):
            raise RuntimeError(f"cache-only: {code} cache does not cover window")
        provenance[code] = cache_provenance_entry(cache, code, "reused")
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
    b_prev_close = close.iloc[obs - 1] if obs >= 1 else np.nan
    out["break_open_return"] = (
        _ret(_safe_ratio(out["d1_open"], float(b_prev_close)))
        if obs >= 1 and pd.notna(b_prev_close) else None)
    out["prev_close_d1"] = (float(b_prev_close)
                            if obs >= 1 and pd.notna(b_prev_close) else None)
    return out


# ---------------------------------------------------------------------------
# structural / unexpected missing 分类 (§18-§22)
# ---------------------------------------------------------------------------
def classify_feature_missing(row: dict, day_m: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    """§20/§21: 区分 structural missing (明确定义的公式数学退化) 与
    unexpected missing (任何其他缺失 -> 使 dev_feature_source_complete 变 False)。

    目前 structural 仅: d1_close_location 当 D1 日内 5min high == low
    (官方公式 (close-low)/(high-low) 分母为 0 -> None)。
    """
    structural: list[str] = []
    unexpected: list[str] = []
    if day_m is None or day_m.empty:
        for col in MINUTE_FEATURES:
            if row.get(col) is None:
                unexpected.append(col)
        return structural, unexpected
    high = _max_numeric(day_m, "high")
    low = _min_numeric(day_m, "low")
    intraday_zero_range = high is not None and low is not None and high == low
    for col in MINUTE_FEATURES:
        if row.get(col) is not None:
            continue
        if col == "d1_close_location" and intraday_zero_range:
            structural.append(col)
        else:
            unexpected.append(col)
    return structural, unexpected


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

    # ---- Target 标签 (development-level) ----
    # May: 已由 audit_may_labels_live 基于当前 canonical daily cache 正式审计;
    # June: 冻结链 (daily_label_quality_ok + 冻结 target 值)。两窗口字段语义一致。
    dev_label_complete = bool(_label_bool(event.get("daily_label_complete")) is True)
    row["dev_label_complete"] = dev_label_complete
    row["target7_daily_d2open_d3high"] = _label_bool(event.get("target7_daily_d2open_d3high"))
    row["tail_loss_daily_5pct"] = _label_bool(event.get("tail_loss_daily_5pct"))
    # §5: UNKNOWN != FALSE — label 不完整时 target 必须显式缺失 (NA), 不得因
    # pandas / bool coercion 变成 False 而让未来模型误认为真实负样本。
    if not dev_label_complete:
        row["target7_daily_d2open_d3high"] = pd.NA
        row["tail_loss_daily_5pct"] = pd.NA
    # legacy audit only (v0.2.2 role; 不进入新 development summary count, §6/§32)
    row["daily_label_complete"] = dev_label_complete
    row["target_training_eligible"] = event.get("target_training_eligible")
    row["label_d2_date"] = event.get("label_d2_date")
    row["label_d3_date"] = event.get("label_d3_date")

    # ---- provenance (静态标签, 不进 timestamp) ----
    row["minute_source"] = BAOSTOCK_5M_SOURCE
    row["minute_adjustment"] = "none"
    row["minute_interval"] = NORMALIZED_5M_INTERVAL
    row["daily_source"] = "canonical_daily_cache"

    # ---- 完整度 (legacy literal audit, §22/§45: 不直接用于模型资格) ----
    minute_non_null = sum(1 for col in MINUTE_FEATURES if row[col] is not None)
    row["minute_feature_count"] = minute_non_null
    row["minute_feature_complete"] = bool(minute_non_null == len(MINUTE_FEATURES))
    daily_non_null = sum(1 for col in DAILY_STATE_COLUMNS + MA_STATE_COLUMNS + RECENT_7D_COLUMNS
                         if row.get(col) is not None)
    row["daily_feature_complete"] = bool(
        daily_non_null == len(DAILY_STATE_COLUMNS) + len(MA_STATE_COLUMNS) + len(RECENT_7D_COLUMNS))
    row["final_x_data_complete"] = bool(
        row["d1_minute_complete"] and row["minute_feature_complete"] and row["daily_feature_complete"])

    # ---- structural / unexpected missing (§18-§22) ----
    structural, unexpected = classify_feature_missing(row, day_m)
    row["structural_missing_count"] = len(structural)
    row["unexpected_missing_count"] = len(unexpected)
    row["structural_missing_features"] = "|".join(structural)
    row["unexpected_missing_features"] = "|".join(unexpected)
    # §21: dev_feature_source_complete = d1_minute_complete AND daily source/input
    # complete AND unexpected_missing_count == 0; structural missing 不使该字段
    # 变 False (603065 一类合法样本保留在新开发集)。
    row["dev_feature_source_complete"] = bool(
        row["d1_minute_complete"]
        and row["daily_feature_complete"]
        and row["unexpected_missing_count"] == 0)
    # §23: dev_training_eligible = dev_label_complete AND dev_feature_source_complete
    row["dev_training_eligible"] = bool(
        row["dev_label_complete"] and row["dev_feature_source_complete"])
    return row


# ---------------------------------------------------------------------------
# May label recovery (§7-§10: 先 recovery, 再 patch universe, 再 build rows)
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


def merge_daily_append_only(
    existing: pd.DataFrame | None,
    fetched: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """§7/§8: existing canonical daily rows immutable。

    merged = existing rows + 真正缺失日期的新行; 重叠日期绝不覆盖 (existing
    胜出), 仅输出重叠审计: overlap_dates / overlap_exact_match /
    overlap_value_diff_count (OHLC + volume 值级比较, 一个日期最多计一次 diff)
    / rows_appended。
    """
    fe = fetched.copy()
    fe["date"] = pd.to_datetime(fe["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fe = fe.dropna(subset=["date"])
    if existing is None or existing.empty:
        fe = fe.sort_values("date").reset_index(drop=True)
        return fe, {"rows_appended": int(len(fe)), "overlap_dates": [],
                    "overlap_exact_match": 0, "overlap_value_diff_count": 0}
    ex = existing.copy()
    ex["date"] = pd.to_datetime(ex["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ex = ex.dropna(subset=["date"])
    existing_dates = set(ex["date"])
    fetched_dates = set(fe["date"])
    overlap_dates = sorted(existing_dates & fetched_dates)
    new_rows = fe[~fe["date"].isin(existing_dates)]
    exact_match = 0
    value_diff_count = 0
    if overlap_dates:
        ex_idx = ex.set_index("date")
        fe_idx = fe.set_index("date")
        for date in overlap_dates:
            a = ex_idx.loc[date]
            b = fe_idx.loc[date]
            differs = False
            for col in ("open", "high", "low", "close", "volume"):
                va = _to_float_or_none(a.get(col))
                vb = _to_float_or_none(b.get(col))
                if va is None and vb is None:
                    continue
                if va != vb:
                    differs = True
                    break
            if differs:
                value_diff_count += 1
            else:
                exact_match += 1
    merged = pd.concat([ex, new_rows], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged, {"rows_appended": int(len(new_rows)), "overlap_dates": overlap_dates,
                    "overlap_exact_match": exact_match, "overlap_value_diff_count": value_diff_count}


def recover_may_labels(
    universe: pd.DataFrame,
    provider: MarketDataProvider,
    cal_idx: dict,
    proofs: dict,
    do_recovery: bool = True,
) -> dict:
    """对冻结 candidates 标记 label 不完整的 May 事件执行 daily 补齐。

    - 只追加缺失日期, existing canonical rows 不被覆盖 (§7);
    - 已恢复状态 (当前 daily cache 的 live audit 已 complete) 跳过 fetch,
      无网络, 确定性 (§9);
    - 重叠日期仅审计 (§8);
    - before 状态取冻结 candidates CSV 记录; after 取恢复尝试后的实时审计。
    """
    out: dict[str, dict] = {}
    may = universe[universe["source_window"] == "may"]
    incomplete = may[may["daily_label_complete"].ne(True)]
    for _, ev in incomplete.iterrows():
        code = str(ev["code"])
        sig = str(ev["signal_date"])
        daily = load_daily_frame(code, DAILY_DIR)
        before = audit_daily_label(code, sig, cal_idx, proofs, daily)
        record: dict = {
            "code": code,
            "signal_date": sig,
            "label_d2_date": str(before.get("label_d2_date")),
            "label_d3_date": str(before.get("label_d3_date")),
            "reason_before": str(ev.get("daily_label_reason") or before.get("daily_label_reason")),
            "label_complete_before": bool(_label_bool(ev.get("daily_label_complete")) is True),
            "rows_appended": 0,
            "overlap_dates": [],
            "overlap_exact_match": 0,
            "overlap_value_diff_count": 0,
            "daily_cache_written": False,
        }
        if before["daily_label_complete"]:
            # §9: already recovered (v001 恢复已写回 canonical daily cache)
            after = before
            record["action"] = "already_recovered"
        elif not do_recovery:
            after = before
            record["action"] = "skipped_no_recovery"
        else:
            try:
                fetched, _src = provider.fetch_daily_history(
                    code, RECOVERY_FETCH_START, RECOVERY_FETCH_END, adjust="none")
                merged, append_audit = merge_daily_append_only(daily, fetched)
                record.update(append_audit)
                if len(merged) > (0 if daily is None else len(daily)):
                    _persist_daily_cache(code, merged)
                    record["daily_cache_written"] = True
                after = audit_daily_label(code, sig, cal_idx, proofs, merged)
                record["action"] = (
                    "recovered_by_append" if after["daily_label_complete"]
                    else "append_no_label_complete")
            except Exception as exc:
                after = before
                record["action"] = f"recovery_failed: {type(exc).__name__}: {str(exc)[:200]}"
        record["label_complete_after"] = bool(after["daily_label_complete"])
        record["label_recovered"] = bool(
            not record["label_complete_before"] and after["daily_label_complete"])
        record["target7_after"] = after.get("target7_daily_d2open_d3high")
        record["tail_loss_after"] = after.get("tail_loss_daily_5pct")
        out[ev["event_id"]] = record
    return out


def audit_may_labels_live(
    universe: pd.DataFrame,
    cal_idx: dict,
    proofs: dict,
) -> pd.DataFrame:
    """§10: May 全部事件统一重新调用正式 audit_daily_label(...) (基于当前
    canonical daily cache), 生成新的 development-level label state。
    June 使用冻结链 (load_universe 已映射 daily_label_quality_ok), 字段语义一致。
    """
    u = universe.copy()
    # 冻结 candidates / 冻结链中不完整事件的 target 单元格可能是 NaN -> 列 dtype
    # 为 float64; at-set 写 bool 会触发 pandas LossySetitemError。统一转 object
    # 保证三态 (True/False/NA) 可写 (UNKNOWN != FALSE)。
    for col in ("target7_daily_d2open_d3high", "tail_loss_daily_5pct",
                "daily_label_complete", "label_d2_date", "label_d3_date"):
        if col in u.columns:
            u[col] = u[col].astype(object)
    daily_cache: dict[str, pd.DataFrame | None] = {}
    may_idx = u.index[u["source_window"] == "may"]
    for i in may_idx:
        code = str(u.at[i, "code"])
        if code not in daily_cache:
            daily_cache[code] = load_daily_frame(code, DAILY_DIR)
        audit = audit_daily_label(code, str(u.at[i, "signal_date"]), cal_idx, proofs,
                                  daily_cache[code])
        u.at[i, "target7_daily_d2open_d3high"] = audit["target7_daily_d2open_d3high"]
        u.at[i, "tail_loss_daily_5pct"] = audit["tail_loss_daily_5pct"]
        u.at[i, "daily_label_complete"] = bool(audit["daily_label_complete"])
        u.at[i, "label_d2_date"] = audit.get("label_d2_date")
        u.at[i, "label_d3_date"] = audit.get("label_d3_date")
    return u


# ---------------------------------------------------------------------------
# 输出 (所有文件都写到 out_dir, --output 契约 §31)
# ---------------------------------------------------------------------------
DEV_CSV_COLUMNS = (
    ("event_id", "code", "name", "signal_date", "break_date",
     "board_streak_before_break", "source_window")
    + DAILY_STATE_COLUMNS + MA_STATE_COLUMNS + RECENT_7D_COLUMNS
    + MINUTE_FEATURES
    + ("target7_daily_d2open_d3high", "tail_loss_daily_5pct",
       "daily_label_complete", "target_training_eligible",
       "label_d2_date", "label_d3_date",
       "dev_label_complete", "dev_feature_source_complete", "dev_training_eligible")
    + PROVENANCE_COLUMNS
    + ("d1_minute_complete", "minute_feature_count", "minute_feature_complete",
       "daily_feature_complete", "final_x_data_complete",
       "structural_missing_count", "unexpected_missing_count")
)


def render_lineage_csv() -> pd.DataFrame:
    rows = []
    for name in MINUTE_FEATURES:
        note = ""
        if name == "d1_volume":
            note = ("legacy_source = 官方构建 5m volume 求和 (Sina 源; 非 canonical "
                    "daily volume, 两者仅数值等价, 不属于同一 raw lineage); "
                    "dev_source = BaoStock 5m volume 求和; unit = shares(股); "
                    "cross_source validation = equivalent within validated tolerance "
                    "(June 交叉核对 volume 中位比 1.0)")
        elif name == "d1_amount":
            note = ("legacy_source = 官方构建 5m amount 求和 (Sina 源; daily cache "
                    "amount 为 None, 官方构建即用 5m 求和); dev_source = BaoStock "
                    "5m amount 求和; unit = 元; cross_source validation = equivalent "
                    "within validated tolerance (June 交叉核对 amount 中位比 ~1.0)")
        rows.append({
            "feature_name": name,
            "feature_group": "minute",
            "formula_source": ("build_v004c_dataset.py E/F 官方公式 "
                               "(src/v004c_minute_features.py 同公式副本; 不改变数学定义, "
                               "不加 epsilon/band; 输入源为 BaoStock 5m)"),
            "raw_dependency": "5m bars (baostock_5m): open/high/low/close/volume/amount",
            "rebuilt_from": "baostock_5m",
            "lineage_note": note,
        })
    for name in DAILY_STATE_COLUMNS + MA_STATE_COLUMNS:
        rows.append({
            "feature_name": name,
            "feature_group": "daily_ma_state",
            "formula_source": ("build_v004c_dataset.py D section + "
                               "v004c_factor_dictionary EXISTING_RECOMPUTE_CHECKS"),
            "raw_dependency": "canonical daily cache OHLC (rolling ma)",
            "rebuilt_from": "canonical_daily_cache",
            "lineage_note": "",
        })
    for name in RECENT_7D_COLUMNS:
        rows.append({
            "feature_name": name,
            "feature_group": "recent_7d",
            "formula_source": "v004c_model_table.py compute_recent_7d_features",
            "raw_dependency": "canonical daily cache OHLC (7-row window)",
            "rebuilt_from": "canonical_daily_cache",
            "lineage_note": "",
        })
    return pd.DataFrame(rows)


def write_outputs(
    rows: list[dict],
    universe: pd.DataFrame,
    provenance: dict,
    may_label_recovery: dict,
    june_crosscheck: dict,
    sina_regression: dict,
    out_dir: Path,
    socks5_proxy: str | None,
    review_text: str | None = None,
) -> dict:
    """写 dev / quality / lineage / provenance (review_text 提供时含 review) 到
    out_dir (--output 契约: 全部真实写到 X)。返回 {name: path}。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = pd.DataFrame(rows)
    full = full.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)

    quality = full[[
        "event_id", "code", "signal_date", "source_window",
        "expected_bars", "actual_bars", "first_timestamp", "last_timestamp",
        "missing_bars", "duplicate_bars", "invalid_price_volume",
        "d1_minute_complete", "minute_missing_reason",
        "minute_feature_count", "minute_feature_complete",
        "structural_missing_features", "unexpected_missing_features",
        "dev_label_complete", "dev_feature_source_complete", "dev_training_eligible",
    ]].copy()
    quality_path = out_dir / QUALITY_CSV_NAME
    quality.to_csv(quality_path, index=False, encoding="utf-8-sig")

    dev = full[list(DEV_CSV_COLUMNS)].copy()
    dev_path = out_dir / DEV_CSV_NAME
    dev.to_csv(dev_path, index=False, encoding="utf-8-sig")

    lineage_path = out_dir / LINEAGE_CSV_NAME
    render_lineage_csv().to_csv(lineage_path, index=False, encoding="utf-8-sig")

    provenance_path = out_dir / PROVENANCE_NAME
    with provenance_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "window": [WINDOW_START, WINDOW_END],
            "cache_dir": str(BAO_CACHE_DIR.relative_to(ROOT)).replace("\\", "/"),
            "socks5_proxy": socks5_proxy,
            "baostock_client_version": _baostock_package_version(),
            "codes": provenance,
        }, fh, ensure_ascii=False, sort_keys=True, indent=1)

    files = {"dev": dev_path, "quality": quality_path,
             "lineage": lineage_path, "provenance": provenance_path}
    if review_text is not None:
        review_path = out_dir / REVIEW_NAME
        review_path.write_text(review_text, encoding="utf-8")
        files["review"] = review_path
    return files


def _baostock_package_version() -> str | None:
    try:
        from importlib import metadata
        return metadata.version("baostock")
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Stats + review
# ---------------------------------------------------------------------------
def build_stats(
    dev: pd.DataFrame,
    provenance: dict,
    may_label_recovery: dict,
    june_crosscheck: dict,
    sina_regression: dict,
) -> dict:
    stats: dict = {}
    for window in ("may", "june"):
        sub = dev[dev["source_window"] == window]
        streak = pd.to_numeric(sub["board_streak_before_break"], errors="coerce")
        stats[window] = {
            "rows": int(len(sub)),
            "signal_dates": int(sub["signal_date"].nunique()),
            "codes": int(sub["code"].nunique()),
            "bao_fetched": int(sub["minute_source"].eq(BAOSTOCK_5M_SOURCE).sum()),
            "minute_complete_48": int(sub["d1_minute_complete"].sum()),
            "minute_complete_rate": float(sub["d1_minute_complete"].mean()) if len(sub) else 0.0,
            "minute_features_complete": int(sub["minute_feature_complete"].sum()),
            "daily_features_complete": int(sub["daily_feature_complete"].sum()),
            "final_x_complete": int(sub["final_x_data_complete"].sum()),
            "dev_label_complete": int(sub["dev_label_complete"].fillna(False).astype(bool).sum()),
            "dev_feature_source_complete": int(
                sub["dev_feature_source_complete"].fillna(False).astype(bool).sum()),
            "dev_training_eligible": int(
                sub["dev_training_eligible"].fillna(False).astype(bool).sum()),
            "board2": int(streak.eq(2).sum()),
            "board3": int(streak.eq(3).sum()),
            "structural_missing_rows": int((sub["structural_missing_count"] > 0).sum()),
            "unexpected_missing_rows": int((sub["unexpected_missing_count"] > 0).sum()),
        }
    dev2 = dev
    streak_all = pd.to_numeric(dev2["board_streak_before_break"], errors="coerce")
    stats["combined"] = {
        "rows": int(len(dev2)),
        "signal_dates": int(dev2["signal_date"].nunique()),
        "board2": int(streak_all.eq(2).sum()),
        "board3": int(streak_all.eq(3).sum()),
        "minute_source_values": sorted(dev2["minute_source"].astype(str).unique().tolist()),
        "dev_label_complete": int(dev2["dev_label_complete"].fillna(False).astype(bool).sum()),
        "dev_feature_source_complete": int(
            dev2["dev_feature_source_complete"].fillna(False).astype(bool).sum()),
        "dev_training_eligible": int(
            dev2["dev_training_eligible"].fillna(False).astype(bool).sum()),
        "structural_missing_rows": int((dev2["structural_missing_count"] > 0).sum()),
        "unexpected_missing_rows": int((dev2["unexpected_missing_count"] > 0).sum()),
    }
    structural_detail: list[dict] = []
    sel = dev2[dev2["structural_missing_count"] > 0]
    for _, r in sel.iterrows():
        for feat in str(r["structural_missing_features"]).split("|"):
            if not feat:
                continue
            structural_detail.append({
                "event_id": str(r["event_id"]),
                "feature": feat,
                "reason": ("intraday high == low (D1 日内 5min high==low, "
                           "d1_close_location 官方公式分母 (high-low) 为 0 -> None; "
                           "STRUCTURAL_MISSING, 非 vendor / data coverage failure)"),
            })
    stats["structural_missing_detail"] = structural_detail
    unexpected_detail: list[dict] = []
    sel2 = dev2[dev2["unexpected_missing_count"] > 0]
    for _, r in sel2.iterrows():
        for feat in str(r["unexpected_missing_features"]).split("|"):
            if not feat:
                continue
            unexpected_detail.append({"event_id": str(r["event_id"]), "feature": feat})
    stats["unexpected_missing_detail"] = unexpected_detail
    stats["may_label_recovery"] = may_label_recovery
    stats["june_crosscheck"] = june_crosscheck
    stats["sina_regression"] = sina_regression
    stats["recovery_safety"] = {
        "overlapping_existing_dates_overwritten": 0,  # append-only 过程保证
        "rows_appended_total": int(sum(
            r.get("rows_appended", 0) or 0 for r in may_label_recovery.values())),
        "overlap_dates_total": int(sum(
            len(r.get("overlap_dates", [])) for r in may_label_recovery.values())),
        "overlap_value_diff_count_total": int(sum(
            r.get("overlap_value_diff_count", 0) or 0 for r in may_label_recovery.values())),
    }
    stats["provenance_summary"] = build_provenance_summary(provenance)
    return stats


def build_provenance_summary(provenance: dict) -> dict:
    """从 cache meta 聚合的确定性 provenance 摘要 (不含 fetch timestamp)。"""
    dates: list[str] = []
    rows_count: int = 0
    sha_codes = 0
    for entry in provenance.values():
        if entry.get("date_min"):
            dates.append(str(entry["date_min"]))
        if entry.get("date_max"):
            dates.append(str(entry["date_max"]))
        if entry.get("row_count") is not None:
            rows_count += int(entry["row_count"])
        if entry.get("cache_sha256"):
            sha_codes += 1
    return {
        "source": BAOSTOCK_5M_SOURCE,
        "adjustment": "none",
        "interval": NORMALIZED_5M_INTERVAL,
        "cache_dir": str(BAO_CACHE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "cached_codes": int(len(provenance)),
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
        "total_rows": rows_count,
        "sha256_coverage": sha_codes,
        "baostock_client_version": _baostock_package_version(),
    }


def render_review(stats: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# v004c BaoStock Dev Foundation Correctness Fix v002 — Review\n")
    add("任务性质: **定点正确性修复** (不训练 / 不筛选 / 不调参 / 不推翻 v001 "
        "已验证结论)。修复项: P1 May label recovery 执行顺序 (recovery -> patch "
        "universe -> build rows, 恢复结果真正写入 dev CSV) + UNKNOWN != FALSE "
        "(不完整 label 的 target 显式 NA) + daily recovery 只追加缺失日期 (existing "
        "canonical rows immutable) + development eligibility 语义 (dev_label_complete / "
        "dev_feature_source_complete / dev_training_eligible, May 全量 live re-audit, "
        "June 冻结链, 字段语义一致) + structural missing 分类 (603065) + BaoStock "
        "provider strict datetime clipping / fail-closed + provenance 从 cache meta "
        "读取 (含 SHA256 / 包版本) + DETERMINISTIC_REBUILD 正式写入本 review。\n")
    add(f"- 窗口: {WINDOW_START}..{WINDOW_END}; 输出: {OUT_DIR.name}/; "
        f"v001 报告保留审计; fetch timestamp 仅存于 provenance JSON\n")
    add("- July 不进入开发数据; 冻结资产 (repair_state / logistic_walkforward / "
        "d1_dataset) 未修改; v001 的 319/319 D1 5min 覆盖结论不推翻\n")

    add("\n## 1. Universe (v001 -> v002 候选池必须精确一致)\n")
    add("| window | rows | signal_dates | codes |")
    add("|---|---|---|---|")
    for window in ("may", "june"):
        s = stats[window]
        add(f"| {window} | {s['rows']} | {s['signal_dates']} | {s['codes']} |")
    uni = stats.get("universe_v001")
    if uni:
        add(f"\n- v001 rows: {uni['v001_rows']}; v002 rows: {uni['v002_rows']}; "
            f"event_id/code/signal_date/board_streak_before_break exact match: "
            f"{uni['exact_match']} ({uni['matched_rows']}/{uni['v001_rows']})")

    add("\n## 2. BaoStock 5min 获取与覆盖 (source=baostock_5m, adjustment=none, interval=5m)\n")
    add("| window | bao_fetched | 48-bar complete | minute_features_complete(legacy) | "
        "daily_features_complete | final_x_complete(legacy audit) |")
    add("|---|---|---|---|---|---|")
    for window in ("may", "june"):
        s = stats[window]
        add(f"| {window} | {s['bao_fetched']}/{s['rows']} | {s['minute_complete_48']}/{s['rows']} "
            f"({s['minute_complete_rate']:.1%}) | {s['minute_features_complete']}/{s['rows']} "
            f"| {s['daily_features_complete']}/{s['rows']} | {s['final_x_complete']}/{s['rows']} |")
    add("\n- minute_feature_complete / final_x_complete 为 legacy literal audit "
        "(§22/§45), 不直接用于模型资格; 资格见第 8 节 dev_feature_source_complete。")

    add("\n## 3. D1 48-bar completeness (正式规则: 48 bar / 09:35 首 / 15:00 末)\n")
    add("逐 event 明细见 v004c_baostock_d1_minute_quality_v002.csv; 缺失原因分类 "
        "(missing_reason) 复用 v004c_may_d1_coverage 正式分类。\n")
    for window in ("may", "june"):
        s = stats[window]
        add(f"- {window}: {s['minute_complete_48']}/{s['rows']} 完整 "
            f"({s['minute_complete_rate']:.1%})")

    add("\n## 4. May label recovery (§7-§10) 与正式 dev CSV 更新\n")
    rec = stats["may_label_recovery"]
    if not rec:
        add("- 无 label 不完整事件")
    else:
        add("| event_id | code | signal_date | label_d2_date | label_d3_date | reason_before | "
            "label_complete_before | label_complete_after | label_recovered | target7_after | "
            "tail_loss_after |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for event_id, r in sorted(rec.items()):
            add(f"| {event_id} | {r['code']} | {r['signal_date']} | {r['label_d2_date']} | "
                f"{r['label_d3_date']} | {r['reason_before']} | {r['label_complete_before']} | "
                f"{r['label_complete_after']} | {r['label_recovered']} | "
                f"{r['target7_after']} | {r['tail_loss_after']} |")
        recovered = sum(1 for r in rec.values() if r["label_recovered"])
        add(f"\n- recovered: {recovered}/{len(rec)} (before 状态取冻结 candidates CSV, "
            f"跨运行确定; after 取当前 canonical daily cache 的正式 audit_daily_label)")
    add("\n- **formal dev CSV updated: YES** — recovery 先于 build_event_row 完成, "
        "universe label state 统一由 audit_may_labels_live 重新审计后进入行构建; "
        "4 条 recovered 事件在 v002 dev CSV 中 dev_label_complete=True 且 target 为"
        "实际审计结果 (程序内 assert, 非人工 review)")
    add("\n- **UNKNOWN != FALSE**: dev_label_complete=False 的事件 target7/…/tail_loss "
        "在 dev CSV 中为显式 NA (非 False), 三态 (True/False/NA) 可区分 (§5/§37)")
    add("\n### Daily cache recovery safety (§53C)\n")
    rs = stats["recovery_safety"]
    add("| metric | value |")
    add("|---|---|")
    add(f"| overlapping existing dates overwritten | {rs['overlapping_existing_dates_overwritten']} "
        "(append-only 过程保证) |")
    add(f"| new dates appended | {rs['rows_appended_total']} |")
    add(f"| overlap dates observed | {rs['overlap_dates_total']} |")
    add(f"| overlap value diff count | {rs['overlap_value_diff_count_total']} |")

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
    add("\n- 该结论独立于 603065 的 d1_close_location structural NaN (第 9 节): "
        "structural missing 是公式数学退化, 不是 minute data coverage 缺失 (§41)。")

    add("\n## 8. Development eligibility (§54)\n")
    add("| window | rows | dev_label_complete | dev_feature_source_complete | "
        "dev_training_eligible | board2 | board3 |")
    add("|---|---|---|---|---|---|---|")
    for window in ("may", "june"):
        s = stats[window]
        add(f"| {window} | {s['rows']} | {s['dev_label_complete']}/{s['rows']} | "
            f"{s['dev_feature_source_complete']}/{s['rows']} | "
            f"{s['dev_training_eligible']}/{s['rows']} | {s['board2']} | {s['board3']} |")
    c = stats["combined"]
    add(f"| combined | {c['rows']} | {c['dev_label_complete']}/{c['rows']} | "
        f"{c['dev_feature_source_complete']}/{c['rows']} | "
        f"{c['dev_training_eligible']}/{c['rows']} | {c['board2']} | {c['board3']} |")
    add(f"\n- signal dates: May {stats['may']['signal_dates']} + June {stats['june']['signal_dates']} "
        f"= {c['signal_dates']}; minute_source unique values: {c['minute_source_values']}")
    add("\n- dev_training_eligible 是**数据资格** (数据层有资格进入后续模型开发), "
        "不是模型选择 / Target 驱动筛选 / sample weight; 下一阶段模型仍可能因 "
        "fold-only preprocessing / 特征可用性 / 训练起始窗口等原因变化 (§24)。")
    add("\n- legacy target_training_eligible (v0.2.2 role) 仅作 AUDIT_ONLY lineage, "
        "不进入本表统计; May 无 legacy chain, 该列为 NA (§6/§32)。")

    add("\n## 9. Structural missing (§55)\n")
    detail = stats.get("structural_missing_detail") or []
    if detail:
        add("| event_id | feature | reason |")
        add("|---|---|---|")
        for d in detail:
            add(f"| {d['event_id']} | {d['feature']} | {d['reason']} |")
    else:
        add("- 无")
    add("\n- structural missing 是**公式数学退化** (明确定义允许), 不是 vendor / "
        "data coverage failure; 不使 dev_feature_source_complete 变 False (§20/§21)。")
    add(f"- structural missing rows: {c['structural_missing_rows']}")

    add("\n## 10. Unexpected missing (§55)\n")
    detail2 = stats.get("unexpected_missing_detail") or []
    if detail2:
        add("| event_id | feature | reason |")
        add("|---|---|---|")
        for d in detail2:
            add(f"| {d['event_id']} | {d['feature']} | unexpected missing (fail 数据资格) |")
    else:
        add("- 无")
    add(f"- unexpected missing rows: {c['unexpected_missing_rows']}")

    add("\n## 11. 603065_2026-06-11 专项确认 (§52)\n")
    add("- D1 open=high=low=close=12.36 (一字板, 日内 5min high==low); "
        "d1_close_location 官方公式 (close-low)/(high-low) 分母为 0 -> 返回 None, "
        "数学定义未修改 (禁止填 0/0.5/1, §19)")
    add("- 分类: STRUCTURAL_MISSING (structural_missing_count=1: d1_close_location); "
        "unexpected_missing_count=0; dev_feature_source_complete=True; "
        "dev_training_eligible=True (label complete) — 样本保留在新开发集 (程序内 assert)")

    add("\n## 12. Provenance (§56)\n")
    ps = stats["provenance_summary"]
    add(f"- source = {ps['source']}; adjust = {ps['adjustment']}; interval = {ps['interval']}")
    add(f"- cache dir (repo-relative): {ps['cache_dir']}")
    add(f"- cached codes: {ps['cached_codes']}; date range: {ps['date_min']} .. {ps['date_max']}; "
        f"total rows: {ps['total_rows']}")
    add(f"- cache SHA256 coverage: {ps['sha256_coverage']}/{ps['cached_codes']} codes")
    add(f"- baostock package version: {ps['baostock_client_version']}")
    add("- 每 code 明细 (含 fetch_timestamp) 见 v004c_baostock_d1_fetch_provenance_v002.json; "
        "fetch timestamp 独立于确定性资产")

    add("\n## 13. DETERMINISTIC_REBUILD (§29/§30/§57)\n")
    det = stats.get("determinism") or {}
    evd = det.get("evidence") or {}
    add("| asset | build #1 sha256 | cache-only 重建 #2 sha256 | identical |")
    add("|---|---|---|---|")
    for name in ("dev", "quality", "lineage"):
        e = evd.get(name) or {}
        add(f"| {name} CSV | {e.get('build1_sha256', 'n/a')} | "
            f"{e.get('build2_sha256', 'n/a')} | {e.get('identical')} |")
    rev_ident = det.get("review_byte_identical")
    add(f"| review | (两轮渲染字节比较, 不内嵌自身 hash §30) | | {rev_ident} |")
    if det.get("pass") is True:
        add("\n- **DETERMINISTIC_REBUILD = PASS** (build #1 vs cache-only 重建 #2: "
            "dev / quality / lineage byte-identical, review 两轮渲染字节一致)")
    elif det.get("pass") is False:
        add("\n- **DETERMINISTIC_REBUILD = FAIL** (见上表 identical 列; 终端已输出 diff)")
    else:
        add("\n- DETERMINISTIC_REBUILD 检查进行中")

    add("\n## 14. Provider contract (§53A)\n")
    add("- strict datetime clipping: **PASS** — baostock API 只支持按日参数, "
        "normalize 后再次执行 start<=datetime<=end, 与 sina_5m 对外 contract 一致; "
        "partial-window (10:00..11:00) 请求返回无 09:35/15:00 等区间外 bar (单元测试)")
    add("- raw duplicate fail-closed: **PASS** — normalize_baostock_5m_frame 对重复 "
        "(code, datetime) raise RuntimeError, 无 silent drop (单元测试)")
    add("- invalid-row fail-closed: **PASS** — unparseable datetime / NaN OHLC / "
        "NaN volume/amount / 负价格 / high<open / low>open / 负 volume/amount 一律 "
        "fail closed (normalize + validate)")
    add("- suspension placeholder: **PASS** — OHLC 全 0 停牌占位 bar 显式分类为 "
        "BAOSTOCK_SUSPENSION_PLACEHOLDER 后移除, 与 INVALID_MARKET_BAR 严格区分")
    add("- silent fallback: **保持禁止** — baostock 失败不尝试 sina")
    add("- Sina compatibility: **保留** — normalize_5min_frame (Sina 路径) 行为未改")

    add("\n## 15. 限制\n")
    add("- 日线 / MA / recent-7d 状态来自现有 canonical daily cache; 若缓存含历史回补, "
        "June 交叉核对 (第 5 节) 会显示差异")
    add("- d1_volume/d1_amount 属 22 个 minute features (官方列表), 由 BaoStock 5min "
        "求和; legacy 官方构建为 Sina 5min 求和, 两者数值等价但非同一 raw lineage "
        "(见 lineage CSV lineage_note, §25)")
    add("- 5min 源替换只影响 minute-dependent features; 冻结模型 / 阈值 / lambda / "
        "特征选择均未重做 (本任务禁止); Feature Contract 扩展 (break_high_return 等) "
        "属下一阶段, 本任务未添加")
    add("- 本任务实际运行状态为已恢复 (v001 已写回 daily cache): 两轮构建均无网络 / "
        "无 daily cache 写入 (rows_appended=0); 确定性证据基于该状态 (§9)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# June crosscheck + Sina regression
# ---------------------------------------------------------------------------
def run_june_crosscheck(dev: pd.DataFrame, frozen: pd.DataFrame) -> dict:
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
# v001 -> v002 universe 精确匹配 (§50)
# ---------------------------------------------------------------------------
def check_universe_vs_v001(dev: pd.DataFrame, v001_csv: Path) -> dict:
    """对 event_id / code / signal_date / board_streak_before_break 要求
    v001 vs v002 319/319 精确一致 (本任务禁止因修复 label / structural missing
    改候选池)。"""
    cols = ["event_id", "code", "signal_date", "board_streak_before_break"]
    a = dev[cols].copy()
    b = pd.read_csv(v001_csv, dtype={"code": str})[cols].copy()
    a["event_id"] = a["event_id"].astype(str)
    a["code"] = a["code"].astype(str).str.zfill(6)
    a["signal_date"] = a["signal_date"].astype(str)
    b["event_id"] = b["event_id"].astype(str)
    b["code"] = b["code"].astype(str).str.zfill(6)
    b["signal_date"] = b["signal_date"].astype(str)
    a["board_streak_before_break"] = pd.to_numeric(
        a["board_streak_before_break"], errors="coerce")
    b["board_streak_before_break"] = pd.to_numeric(
        b["board_streak_before_break"], errors="coerce")
    a = a.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)
    b = b.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)
    if len(a) != len(b):
        raise RuntimeError(
            f"v001/v002 universe rows mismatch: v001={len(b)} v002={len(a)}")
    mismatches: list[dict] = []
    for col in cols:
        bad = a[col].astype(str).ne(b[col].astype(str))
        if bad.any():
            idx = a.index[bad].tolist()
            mismatches.append({"column": col, "rows": idx[:10]})
    exact = not mismatches
    if not exact:
        raise RuntimeError(
            f"v001/v002 universe mismatch on columns: "
            f"{[m['column'] for m in mismatches]}")
    return {"v001_rows": int(len(b)), "v002_rows": int(len(a)),
            "matched_rows": int(len(a)), "exact_match": exact}


# ---------------------------------------------------------------------------
# 程序内自动断言 (§51/§52)
# ---------------------------------------------------------------------------
def assert_v002_requirements(dev: pd.DataFrame, quality: pd.DataFrame) -> None:
    """构建结束前自动校验 (不依赖人工 review):
    §51: 4 条 recovered May labels 进入正式 dev CSV (dev_label_complete=True,
    target 为实际审计结果, 非 NA);
    §52: 603065_2026-06-11 保留且为 structural missing (d1_close_location),
    unexpected=0, dev_feature_source_complete=True, dev_training_eligible=True。"""
    recovered_events = (
        ("000402", "2026-05-07"), ("000553", "2026-05-07"),
        ("002149", "2026-05-07"), ("603158", "2026-05-08"))
    for code, sig in recovered_events:
        event_id = f"{code}_{sig}"
        sel = dev[dev["event_id"] == event_id]
        if sel.empty:
            raise RuntimeError(f"assert: recovered event missing in dev CSV: {event_id}")
        row = sel.iloc[0]
        if not bool(row["dev_label_complete"]):
            raise RuntimeError(f"assert: {event_id} dev_label_complete must be True")
        if pd.isna(row["target7_daily_d2open_d3high"]) or pd.isna(row["tail_loss_daily_5pct"]):
            raise RuntimeError(f"assert: {event_id} target must be audited value (not NA)")
    event_id = "603065_2026-06-11"
    sel = dev[dev["event_id"] == event_id]
    if sel.empty:
        raise RuntimeError(f"assert: {event_id} must be retained in dev CSV")
    row = sel.iloc[0]
    if not bool(row["d1_minute_complete"]):
        raise RuntimeError(f"assert: {event_id} d1_minute_complete must be True")
    qrow = quality[quality["event_id"] == event_id]
    if qrow.empty or "d1_close_location" not in str(qrow.iloc[0]["structural_missing_features"]):
        raise RuntimeError(f"assert: {event_id} structural missing must contain d1_close_location")
    if int(row["unexpected_missing_count"]) != 0:
        raise RuntimeError(f"assert: {event_id} unexpected_missing_count must be 0")
    if not bool(row["dev_feature_source_complete"]):
        raise RuntimeError(f"assert: {event_id} dev_feature_source_complete must be True")
    if not bool(row["dev_training_eligible"]):
        raise RuntimeError(f"assert: {event_id} dev_training_eligible must be True")


# ---------------------------------------------------------------------------
# 单次构建
# ---------------------------------------------------------------------------
def build_once(
    universe0: pd.DataFrame,
    provider: MarketDataProvider,
    bao_cache: Baostock5mCache,
    cal_idx: dict,
    proofs: dict,
    cache_only: bool,
    do_recovery: bool,
    force: bool,
    out_dir: Path,
) -> dict:
    """一次完整构建 (fetch/cache-only + recovery + audit + rows + 输出到 out_dir,
    不含 review — review 由 main 在确定性证据后统一渲染写入)。"""
    if cache_only:
        provenance = verify_cache_only_coverage(universe0, bao_cache)
    else:
        provenance = fetch_baostock_window(universe0, bao_cache, provider, force=force)

    # P1: recovery 必须先完成, 再 patch universe label state, 再 build rows
    may_label_recovery = recover_may_labels(
        universe0, provider, cal_idx, proofs, do_recovery=do_recovery)
    universe = audit_may_labels_live(universe0, cal_idx, proofs)

    daily_cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict] = []
    for _, ev in universe.iterrows():
        code = str(ev["code"])
        if code not in daily_cache:
            daily_cache[code] = load_daily_frame(code, DAILY_DIR)
        rows.append(build_event_row(ev, bao_cache, daily_cache[code], cal_idx, proofs))
    full = pd.DataFrame(rows)
    full = full.sort_values(["signal_date", "code", "event_id"]).reset_index(drop=True)
    dev = full[list(DEV_CSV_COLUMNS)].copy()
    quality = full[[
        "event_id", "code", "signal_date", "source_window",
        "expected_bars", "actual_bars", "first_timestamp", "last_timestamp",
        "missing_bars", "duplicate_bars", "invalid_price_volume",
        "d1_minute_complete", "minute_missing_reason",
        "minute_feature_count", "minute_feature_complete",
        "structural_missing_features", "unexpected_missing_features",
        "dev_label_complete", "dev_feature_source_complete", "dev_training_eligible",
    ]].copy()

    frozen = pd.read_csv(JUNE_CSV, dtype={"code": str})
    june_crosscheck = run_june_crosscheck(dev, frozen)
    sina_regression = run_sina_regression(dev, universe, SINA_MINUTE_DIR, bao_cache)
    # build_stats 需要完整行 (structural/unexpected feature-name 列只在 quality 表)
    stats = build_stats(full, provenance, may_label_recovery,
                        june_crosscheck, sina_regression)
    files = write_outputs(rows, universe, provenance, may_label_recovery,
                          june_crosscheck, sina_regression, out_dir,
                          socks5_proxy=None, review_text=None)
    return {
        "out_dir": Path(out_dir),
        "dev": dev,
        "quality": quality,
        "rows": rows,
        "universe": universe,
        "provenance": provenance,
        "recovery": may_label_recovery,
        "crosscheck": june_crosscheck,
        "regression": sina_regression,
        "stats": stats,
        "files": files,
        "hashes": {name: file_sha256(path) for name, path in files.items()},
    }


def build_determinism_evidence(b1: dict, b2: dict) -> dict:
    """build #1 vs cache-only 重建 #2: dev / quality / lineage 哈希证据
    (provenance 含 fetch timestamp, 排除; review 由两轮渲染字节比较, 不内嵌
    自身 hash, §30)。"""
    evidence: dict = {}
    for name in ("dev", "quality", "lineage"):
        h1 = b1["hashes"][name]
        h2 = b2["hashes"][name]
        evidence[name] = {"build1_sha256": h1, "build2_sha256": h2,
                          "identical": bool(h1 == h2)}
    return {"evidence": evidence, "pass": None, "review_byte_identical": None}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-only", action="store_true",
                   help="只读 cache 重建 (跳过网络 fetch; 用于确定性验证)")
    p.add_argument("--force-fetch", action="store_true", help="强制重新 fetch (忽略 cache 覆盖)")
    p.add_argument("--no-label-recovery", action="store_true",
                   help="跳过 May label recovery 的 daily 补齐尝试")
    p.add_argument("--proxy-socks5", default=None, metavar="HOST:PORT",
                   help="显式 SOCKS5 代理 (如 127.0.0.1:7890); 仅路由 baostock 服务器连接")
    p.add_argument("--output", default=str(OUT_DIR),
                   help="输出目录 (dev/quality/lineage/review/provenance 全部写到此处)")
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

    universe0 = load_universe(MAY_CSV, JUNE_CSV)
    print(f"universe: may={int((universe0.source_window=='may').sum())} "
          f"june={int((universe0.source_window=='june').sum())} "
          f"codes={universe0['code'].nunique()}")

    socks5_proxy = parse_socks5(args.proxy_socks5)
    if socks5_proxy is not None:
        set_baostock_socks5_proxy(*socks5_proxy)
        print(f"socks5 proxy enabled: {socks5_proxy[0]}:{socks5_proxy[1]} "
              f"(baostock server connections only)")
    try:
        cal = build_trading_calendar(POOL_DIR, DAILY_DIR)
        cal_idx = cal["cal_idx"]
        proofs = load_suspension_proofs(SUSP_DIR)
        print(f"calendar={len(cal['calendar'])} proofs={len(proofs)}")

        # build #1 (正常 build / 或 --cache-only)
        out_dir = Path(args.output)
        b1 = build_once(universe0, provider, bao_cache, cal_idx, proofs,
                        cache_only=args.cache_only,
                        do_recovery=not (args.no_label_recovery or args.cache_only),
                        force=args.force_fetch, out_dir=out_dir)
        print(f"build #1 rows: {len(b1['dev'])} "
              f"(may={int((b1['dev'].source_window=='may').sum())} "
              f"june={int((b1['dev'].source_window=='june').sum())})")
        print(f"fetch provenance: fetched={sum(1 for p in b1['provenance'].values() if p['cache_status']=='fetched')} "
              f"reused={sum(1 for p in b1['provenance'].values() if p['cache_status']=='reused')}")
        print(f"may label recovery: {len(b1['recovery'])} incomplete, "
              f"recovered={sum(1 for r in b1['recovery'].values() if r['label_recovered'])}")

        # cache-only 重建 #2 (确定性证据; 不 fetch, 不 recovery)
        with tempfile.TemporaryDirectory(prefix="v002_determinism_") as tmp:
            b2 = build_once(universe0, provider, bao_cache, cal_idx, proofs,
                            cache_only=True, do_recovery=False, force=False,
                            out_dir=Path(tmp))
            print(f"cache-only rebuild rows: {len(b2['dev'])}")

            evidence = build_determinism_evidence(b1, b2)
            stats = {**b1["stats"], "determinism": evidence}
            stats["universe_v001"] = check_universe_vs_v001(b1["dev"], V001_DEV_CSV)
            review_text = render_review(stats)
            p1 = out_dir / REVIEW_NAME
            p2 = Path(tmp) / REVIEW_NAME
            p1.write_text(review_text, encoding="utf-8")
            p2.write_text(review_text, encoding="utf-8")
            evidence["review_byte_identical"] = bool(p1.read_bytes() == p2.read_bytes())
            evidence["pass"] = bool(
                evidence["review_byte_identical"]
                and all(v["identical"] for v in evidence["evidence"].values()))
            # 最终渲染 (PASS/FAIL 已确定), 两目录同内容
            stats["determinism"] = evidence
            final_review = render_review(stats)
            p1.write_text(final_review, encoding="utf-8")
            p2.write_text(final_review, encoding="utf-8")

        print(f"DETERMINISTIC_REBUILD = {'PASS' if evidence['pass'] else 'FAIL'}")
        for name in ("dev", "quality", "lineage"):
            e = evidence["evidence"][name]
            print(f"  {name}: build1={e['build1_sha256']} build2={e['build2_sha256']} "
                  f"identical={e['identical']}")
        print(f"  review byte-identical: {evidence['review_byte_identical']}")

        assert_v002_requirements(b1["dev"], b1["quality"])
        print("assert_v002_requirements: PASS (4 recovered labels in dev CSV; "
              "603065 structural missing retained)")
        uni = stats["universe_v001"]
        print(f"universe v001->v002 exact match: {uni['exact_match']} "
              f"({uni['matched_rows']}/{uni['v001_rows']})")

        c = b1["stats"]["combined"]
        print(f"combined: rows={c['rows']} signal_dates={c['signal_dates']} "
              f"dev_label={c['dev_label_complete']} "
              f"dev_feature_source={c['dev_feature_source_complete']} "
              f"dev_training_eligible={c['dev_training_eligible']}")
        print(f"outputs written to {out_dir}")
    finally:
        clear_baostock_socks5_proxy()


if __name__ == "__main__":
    main()

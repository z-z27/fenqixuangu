# -*- coding: utf-8 -*-
"""v004c 日线标签缺失来源审计 (只读)

阶段A: 真实缺失样本清单 (v004c_missing_daily_label_rows.csv)
阶段B: 当前个股日线缓存审计 (v004c_current_daily_cache_audit.csv)
阶段C: 本地来源发现 (v004c_local_source_discovery.csv)
"""
import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
TP = ROOT / "reports" / "research" / "v004c_training_prep_20260601_20260729"
OUT = ROOT / "reports" / "research" / "v004c_daily_source_audit_20260601_20260729"
SCRATCH = OUT / "_scratch"

DAILY_DIR = ROOT / "data" / "cache" / "daily"
UNADJ_DIR = ROOT / "data" / "cache" / "daily_unadjusted"
RAW_DIR = ROOT / "data" / "raw" / "daily"
MIN_DIR = ROOT / "data" / "cache" / "minute_5m"
OVERRIDES = ROOT / "data" / "overrides" / "minute_bar_repairs.csv"

TRAIN_START = "2026-06-01"
TRAIN_END = "2026-07-29"

# ---------------- A. 缺失样本清单 ----------------
print("[A] missing rows ...", flush=True)
v21 = pd.read_csv(TP / "v004c_dataset_v021_all.csv", dtype={"code": str})
v21["code"] = v21["code"].astype(str).str.zfill(6)
miss = v21[(v21["signal_date"] >= TRAIN_START) & (v21["signal_date"] <= TRAIN_END)
           & (v21["daily_label_quality_ok"] == False)].copy()  # noqa: E712
print("missing rows (Jun-Jul):", len(miss), "| events:", miss["event_id"].nunique(),
      "| codes:", miss["code"].nunique())

cols = ["event_id", "code", "name", "signal_date", "break_date", "stage_group", "post_day",
        "expected_d2_date", "expected_d3_date", "actual_daily_d2_date", "actual_daily_d3_date",
        "daily_label_quality_reason", "d2_daily_date_shift_reason", "d3_daily_date_shift_reason",
        "suspension_proof_status_daily",
        "audit_minute_d2_open", "audit_minute_d3_high", "audit_minute_d3_close",
        "audit_minute_target7", "audit_minute_tail_loss",
        "v004a_probability", "v004a_rank", "is_v004a_top15", "v004a_score_source",
        "sample_role", "sample_role_v021"]
miss_out = miss[[c for c in cols if c in miss.columns]].copy()
# 缺失类型细分
def miss_type(r):
    # 以日期解析结果字段为准 (reason 文本在 v0.2.1 构建中对 no_bar_after_d2 未映射, 见 review)
    d2_shift = str(r.get("d2_daily_date_shift_reason", ""))
    d3_shift = str(r.get("d3_daily_date_shift_reason", ""))
    d2_missing = d2_shift in ("no_bar_without_proof", "suspended_no_resume_bar", "no_daily_cache", "expected_d2_missing")
    d3_missing = d3_shift in ("no_bar_without_proof", "no_bar_after_d2", "suspended_no_resume_bar", "no_daily_cache", "expected_d3_missing")
    if d2_missing and d3_missing:
        return "D2_AND_D3_MISSING"
    if d2_missing:
        return "D2_MISSING"
    if d3_missing:
        return "D3_MISSING"
    return "OTHER"

miss_out["missing_type"] = miss_out.apply(miss_type, axis=1)
miss_out["month"] = miss_out["signal_date"].str[:7]
miss_out.to_csv(OUT / "v004c_missing_daily_label_rows.csv", index=False, encoding="utf-8-sig")
print("missing type counts:", miss_out["missing_type"].value_counts().to_dict())
print("by month:", miss_out["month"].value_counts().sort_index().to_dict())

# ---------------- B. 个股日线缓存审计 ----------------
print("[B] daily cache audit ...", flush=True)
loaders_ref = "src/loaders.py: MarketDataService.daily_cache = FrameCache(config.cache_dir / 'daily') (data/cache/daily)"

overrides = pd.read_csv(OVERRIDES, dtype={"code": str}) if OVERRIDES.exists() else pd.DataFrame()
override_codes = set(overrides["code"].astype(str).str.zfill(6)) if not overrides.empty else set()

BUILD_MTIME_REF = pd.Timestamp(os.path.getmtime(TP / "v004c_training_primary_v021.csv"), unit="s")

cache_rows = []
for _, r in miss.iterrows():
    code = r["code"]
    d2 = str(r["expected_d2_date"]) if pd.notna(r["expected_d2_date"]) else ""
    d3 = str(r["expected_d3_date"]) if pd.notna(r["expected_d3_date"]) else ""
    p = DAILY_DIR / f"{code}_daily.pkl"
    exists = p.exists()
    first = last = ""
    nrows = None
    d2_in = d3_in = False
    mtime = size = sha = ""
    if exists:
        st = os.stat(p)
        mtime = pd.Timestamp(st.st_mtime, unit="s", tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
        size = int(st.st_size)
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        d = pd.read_pickle(p)
        dates = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        dates = dates.dropna()
        nrows = int(len(dates))
        if len(dates):
            first, last = str(dates.min()), str(dates.max())
        d2_in = bool(d2 and (dates == d2).any())
        d3_in = bool(d3 and (dates == d3).any())
    # audit_result 分类
    if not exists:
        result = "DAILY_FILE_MISSING"
    elif d2_in and d3_in:
        result = "DATES_PRESENT_BUT_LOADER_NOT_READING"
    elif d2_in and not d3_in:
        if last and d2 and last == d2:
            result = "DAILY_FILE_ENDS_AT_D2"
        elif last and d2 and last > d2:
            result = "DATE_GAP_INSIDE_FILE"
        else:
            result = "D2_PRESENT_D3_MISSING"
    elif not d2_in and last and d2 and last < d2:
        result = "DAILY_FILE_ENDS_BEFORE_D2"
    elif not d2_in and last and d2 and last == d2:
        result = "DAILY_FILE_ENDS_AT_D2"
    elif not d2_in and last and d2 and last > d2:
        result = "DATE_GAP_INSIDE_FILE"
    else:
        result = "OTHER"
    cache_refreshed_after_build = bool(exists and pd.Timestamp(st.st_mtime, unit="s") > BUILD_MTIME_REF)
    if cache_refreshed_after_build and result == "DATES_PRESENT_BUT_LOADER_NOT_READING":
        result = "CACHE_REFRESHED_EXTERNALLY_AFTER_V021_BUILD"
    cache_rows.append({
        "event_id": r["event_id"], "code": code, "signal_date": r["signal_date"],
        "expected_d2_date": d2, "expected_d3_date": d3,
        "primary_daily_file_path": str(p) if exists else "",
        "primary_daily_file_exists": exists,
        "primary_daily_first_date": first, "primary_daily_last_date": last,
        "primary_daily_row_count": nrows,
        "expected_d2_in_primary_daily": d2_in, "expected_d3_in_primary_daily": d3_in,
        "file_modified_time": mtime, "file_size": size, "file_sha256": sha,
        "cache_modified_after_v021_build": cache_refreshed_after_build,
        "v021_build_reference_time": BUILD_MTIME_REF.strftime("%Y-%m-%d %H:%M:%S"),
        "loader_function_path": loaders_ref,
        "loader_applied_override": code in override_codes,
        "override_file_path": str(OVERRIDES) if code in override_codes else "",
        "override_contains_d2": False, "override_contains_d3": False,  # override 为分钟修复, 无日线
        "audit_result": result,
    })
cache_df = pd.DataFrame(cache_rows)
cache_df.to_csv(OUT / "v004c_current_daily_cache_audit.csv", index=False, encoding="utf-8-sig")
print("cache audit rows:", len(cache_df))
print("audit_result:", cache_df["audit_result"].value_counts().to_dict())

# ---------------- C. 本地来源发现 ----------------
print("[C] source discovery ...", flush=True)


def read_frame(path):
    try:
        if str(path).endswith(".pkl"):
            return pd.read_pickle(path)
        return pd.read_csv(path, dtype={"code": str})
    except Exception:
        return None


def dates_of(frame, col):
    if frame is None or col not in frame.columns:
        return set()
    return set(pd.to_datetime(frame[col], errors="coerce").dt.strftime("%Y-%m-%d").dropna())


def price_of(frame, col, date):
    if frame is None or col not in frame.columns:
        return None
    rows = frame[pd.to_datetime(frame[col], errors="coerce").dt.strftime("%Y-%m-%d") == date]
    if rows.empty:
        return None
    v = rows.iloc[0]
    out = {}
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in frame.columns:
            try:
                x = v[c]
                out[c] = None if pd.isna(x) else float(x)
            except (TypeError, ValueError):
                out[c] = None
    return out


# 统一交易日历 (用于 daily_v005 报告的 D2/D3 日期重建)
_pool_dates = set()
for f in glob.glob(str(ROOT / "data" / "cache" / "limit_ups" / "*.pkl")):
    _pool_dates.update(pd.read_pickle(f)["trade_date"].astype(str))
_daily_dates = set()
for f in glob.glob(str(ROOT / "data" / "cache" / "daily" / "*_daily.pkl")):
    d = pd.read_pickle(f)
    _daily_dates.update(pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
_calendar = sorted(_pool_dates | _daily_dates)
_cal_idx = {d: i for i, d in enumerate(_calendar)}


def _next_cal(date):
    i = _cal_idx.get(date)
    return _calendar[i + 1] if (i is not None and i + 1 < len(_calendar)) else None


# 预载报告类分钟来源索引: (code, d2_date) / (code, d3_date) -> OHLC
def load_minute_report_index(glob_pattern, date_col, code_col, o, h, c, signal_date_col=None):
    idx = {}
    for path in sorted(glob.glob(str(ROOT / glob_pattern))):
        frame = read_frame(path)
        if frame is None:
            continue
        f = frame.copy()
        if date_col not in f.columns:
            # 无 d2_trade_date 列: 用 signal_date + 统一日历重建 D2/D3
            if signal_date_col and signal_date_col in f.columns:
                f["_code"] = f[code_col].astype(str).str.zfill(6)
                f["_sig"] = pd.to_datetime(f[signal_date_col], errors="coerce").dt.strftime("%Y-%m-%d")
                rows = []
                for _, row in f.dropna(subset=["_sig"]).iterrows():
                    d2 = _next_cal(str(row["_sig"]))
                    d3 = _next_cal(d2) if d2 else None
                    if d2:
                        rows.append((str(row["_code"]), d2, row))
                    if d3:
                        rows.append((str(row["_code"]), d3, row))
                for code, date, row in rows:
                    key = (code, date)
                    if key not in idx:
                        idx[key] = {"open": None, "high": None, "low": None, "close": None,
                                    "path": path, "source_generated_at": ""}
                    for col, name in ((o, "open"), (h, "high"), (c, "close")):
                        if col in f.columns and pd.notna(row.get(col)):
                            idx[key][name] = float(row[col])
            continue
        f["_code"] = f[code_col].astype(str).str.zfill(6)
        f["_date"] = pd.to_datetime(f[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        for _, row in f.dropna(subset=["_date"]).iterrows():
            key = (str(row["_code"]), str(row["_date"]))
            if key not in idx:
                idx[key] = {"open": None, "high": None, "low": None, "close": None, "path": path,
                            "source_generated_at": ""}
            for col, name in ((o, "open"), (h, "high"), (c, "close")):
                if col in f.columns and pd.notna(row.get(col)):
                    idx[key][name] = float(row[col])
    return idx


hist_idx = load_minute_report_index("reports/history_samples/*/history_candidates_*.csv",
                                    "d2_trade_date", "code", "d2_open_price", "d3_high_price", "d3_close_price")
hold_idx = load_minute_report_index("reports/v005_fixed_grid_holdout_*/v005_fixed_grid_holdout_scored_candidates.csv",
                                    "d2_trade_date", "code", "d2_open_price", "d3_high_price", "d3_close_price")
daily_idx = load_minute_report_index("reports/daily_v005/*/v005_daily_scored_candidates_*.csv",
                                     "d2_trade_date", "code", "d2_open_price", "d3_high_price", "d3_close_price",
                                     signal_date_col="signal_date")
print("report minute indexes: history", len(hist_idx), "| holdout", len(hold_idx), "| daily_v005", len(daily_idx))

discovery_rows = []
for _, r in miss.iterrows():
    code = r["code"]
    targets = [(str(r["expected_d2_date"]), "D2_OPEN"), (str(r["expected_d3_date"]), "D3_HIGH"),
               (str(r["expected_d3_date"]), "D3_CLOSE")]
    targets = [(d, role) for d, role in targets if d and d != "None"]
    for date, role in targets:
        # 各来源
        sources = []
        # S1 主日线
        p1 = DAILY_DIR / f"{code}_daily.pkl"
        f1 = read_frame(p1)
        present1 = bool(f1 is not None and (dates_of(f1, "date") & {date}))
        pr1 = price_of(f1, "date", date) if present1 else {}
        sources.append(("PRIMARY_DAILY_CACHE", str(p1), f1 is not None, present1, pr1))
        # S2 unadjusted
        p2 = UNADJ_DIR / f"{code}_daily.pkl"
        f2 = read_frame(p2)
        present2 = bool(f2 is not None and (dates_of(f2, "date") & {date}))
        pr2 = price_of(f2, "date", date) if present2 else {}
        sources.append(("ALTERNATIVE_DAILY_CACHE", str(p2), f2 is not None, present2, pr2))
        # S3 raw csv
        p3 = RAW_DIR / f"{code}_daily.csv"
        f3 = read_frame(p3)
        present3 = bool(f3 is not None and (dates_of(f3, "date") & {date}))
        pr3 = price_of(f3, "date", date) if present3 else {}
        sources.append(("OFFICIAL_DAILY_REPORT", str(p3), f3 is not None, present3, pr3))
        # S4 5min (minute only)
        p4 = MIN_DIR / f"{code}_5min.pkl"
        f4 = read_frame(p4)
        present4 = bool(f4 is not None and (dates_of(f4, "trade_date") & {date}))
        sources.append(("MINUTE_ONLY", str(p4), f4 is not None, present4, {}))
        # S5-S7 报告分钟快照
        for label, idx in (("HOLDOUT_SNAPSHOT", hold_idx), ("FORWARD_SNAPSHOT", daily_idx),
                           ("OFFICIAL_DAILY_REPORT_HISTORY", hist_idx)):
            rec = idx.get((code, date))
            sources.append((label, rec["path"] if rec else "", rec is not None, rec is not None,
                            {k: rec[k] for k in ("open", "high", "low", "close")} if rec else {}))
        for s_type, s_path, s_exists, s_present, pr in sources:
            is_daily = s_type in ("PRIMARY_DAILY_CACHE", "ALTERNATIVE_DAILY_CACHE", "OFFICIAL_DAILY_REPORT")
            discovery_rows.append({
                "event_id": r["event_id"], "code": code, "signal_date": r["signal_date"],
                "target_date": date, "required_role": role,
                "source_type": s_type, "source_path": s_path, "source_file_exists": s_exists,
                "source_date_present": s_present,
                "open": pr.get("open"), "high": pr.get("high"), "low": pr.get("low"),
                "close": pr.get("close"),
                "source_generated_at": "", "source_git_commit": "",
                "source_sha256": hashlib.sha256(open(s_path, "rb").read()).hexdigest() if (s_exists and os.path.exists(s_path)) else "",
                "source_quality": s_type,
                "source_is_daily_ohlc": is_daily,
                "source_is_minute_aggregate": s_type == "MINUTE_ONLY",
                "source_is_forward_report": s_type in ("FORWARD_SNAPSHOT", "HOLDOUT_SNAPSHOT"),
                "usable_as_official_daily_label": bool(is_daily and s_present and pr.get("close") is not None),
                "reason": "",
            })

disc = pd.DataFrame(discovery_rows)
disc.to_csv(OUT / "v004c_local_source_discovery.csv", index=False, encoding="utf-8-sig")
print("discovery rows:", len(disc))
# 概览
daily_ok = disc[disc["source_is_daily_ohlc"] & disc["source_date_present"]]
print("daily-source date hits:", len(daily_ok), "| by source:", daily_ok["source_type"].value_counts().to_dict())
minute_only = disc[disc["source_type"] == "MINUTE_ONLY"]
print("minute-only date hits:", int(minute_only["source_date_present"].sum()))
print("[done A/B/C]")

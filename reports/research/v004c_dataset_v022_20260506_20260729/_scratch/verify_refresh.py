# -*- coding: utf-8 -*-
"""v004c dataset 0.2.2: 刷新日志核验 (只读)

对照刷新日志逐股验证:
- 日志行数/唯一code数/status/error/missing_required_dates/source/fetch_end_date
- 实际缓存文件存在且包含该股全部必要日期 (必要日期 = 该股缺失观察的 expected_d2/expected_d3)
"""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
REFRESH = ROOT / "reports" / "research" / "v004c_daily_refresh_20260803_224456"
AUDIT = ROOT / "reports" / "research" / "v004c_daily_source_audit_20260601_20260729"
OUT = ROOT / "reports" / "research" / "v004c_dataset_v022_20260506_20260729"
DAILY_DIR = ROOT / "data" / "cache" / "daily"

log = pd.read_csv(REFRESH / "v004c_daily_refresh_log.csv", dtype={"code": str})
log["code"] = log["code"].astype(str).str.zfill(6)
miss = pd.read_csv(AUDIT / "v004c_missing_daily_label_rows.csv", dtype={"code": str})
miss["code"] = miss["code"].astype(str).str.zfill(6)

# 日志层面检查
checks = {
    "log_row_count": len(log),
    "unique_code_count": int(log["code"].nunique()),
    "status_all_ok": bool((log["status"] == "ok").all()),
    "error_all_empty": bool(log["error"].isna().all()),
    "missing_required_dates_all_empty": bool(log["missing_required_dates"].isna().all()),
    "source_all_present": bool(log["source"].notna().all() & (log["source"].astype(str).str.len() > 0).all()),
    "fetch_end_date_all_20260803": bool((log["fetch_end_date"].astype(str) == "2026-08-03").all()),
}
print("log checks:", checks)

# 每股必要日期 = 该股缺失行的 expected_d2 ∪ expected_d3
required_by_code: dict[str, set] = {}
for _, r in miss.iterrows():
    required_by_code.setdefault(r["code"], set()).update(
        [str(r["expected_d2_date"]), str(r["expected_d3_date"])])
for code in log["code"]:
    required_by_code.setdefault(code, set())

rows = []
for _, r in log.iterrows():
    code = r["code"]
    req = sorted(required_by_code.get(code, set()))
    cache_path = DAILY_DIR / f"{code}_daily.pkl"
    cache_exists = cache_path.exists()
    first = last = ""
    nrows = None
    present_dates = set()
    if cache_exists:
        d = pd.read_pickle(cache_path)
        dates = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna()
        present_dates = set(dates)
        nrows = int(len(dates))
        if len(dates):
            first, last = str(dates.min()), str(dates.max())
    missing_in_cache = sorted(set(req) - present_dates)
    all_present = bool(cache_exists and not missing_in_cache)
    reason = []
    if not cache_exists:
        reason.append("cache_file_missing")
    if not all_present:
        reason.append("required_dates_missing:" + ",".join(missing_in_cache))
    if str(r["status"]) != "ok":
        reason.append("refresh_status_not_ok")
    if pd.notna(r["missing_required_dates"]):
        reason.append("log_missing_dates_nonempty")
    verification = "VERIFIED" if not reason else "FAILED:" + ";".join(reason)
    rows.append({
        "code": code,
        "refresh_status": str(r["status"]),
        "refresh_source": str(r["source"]),
        "fetched_first_date": str(r["fetched_first_date"]),
        "fetched_last_date": str(r["fetched_last_date"]),
        "required_date_count": int(r["required_date_count"]),
        "log_missing_required_dates": "" if pd.isna(r["missing_required_dates"]) else str(r["missing_required_dates"]),
        "current_cache_first_date": first,
        "current_cache_last_date": last,
        "current_cache_row_count": nrows,
        "all_required_dates_present": all_present,
        "verification_status": verification,
        "verification_reason": ";".join(reason) if reason else "ok",
        "required_dates": ",".join(req),
    })

ver = pd.DataFrame(rows)
ver.to_csv(OUT / "v004c_daily_refresh_verification.csv", index=False, encoding="utf-8-sig")
print("verification rows:", len(ver))
print("verification status:", ver["verification_status"].value_counts().to_dict())
print("all required dates present:", int(ver["all_required_dates_present"].sum()), "/", len(ver))

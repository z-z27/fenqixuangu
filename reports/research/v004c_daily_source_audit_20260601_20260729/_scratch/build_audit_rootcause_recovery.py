# -*- coding: utf-8 -*-
"""v004c 日线标签缺失: 根因 / 冲突 / 恢复映射 / 汇总 (只读, 不执行恢复)

阶段D: v004c_missing_daily_root_cause.csv
阶段E: v004c_local_daily_source_conflicts.csv
阶段F: v004c_daily_recovery_summary.csv
阶段G: v004c_proposed_daily_recovery_map.csv
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_daily_source_audit_20260601_20260729"

miss = pd.read_csv(OUT / "v004c_missing_daily_label_rows.csv", dtype={"code": str})
cache_audit = pd.read_csv(OUT / "v004c_current_daily_cache_audit.csv", dtype={"code": str})
disc = pd.read_csv(OUT / "v004c_local_source_discovery.csv", dtype={"code": str})
miss["code"] = miss["code"].astype(str).str.zfill(6)
disc["code"] = disc["code"].astype(str).str.zfill(6)

DAILY_TYPES = ("PRIMARY_DAILY_CACHE", "ALTERNATIVE_DAILY_CACHE", "OFFICIAL_DAILY_REPORT")
SNAP_TYPES = ("HOLDOUT_SNAPSHOT", "FORWARD_SNAPSHOT", "OFFICIAL_DAILY_REPORT_HISTORY")

# 每个 (code, date) 的日线来源命中
def daily_hits(code, date):
    sub = disc[(disc["code"] == code) & (disc["target_date"] == date) & disc["source_type"].isin(DAILY_TYPES)]
    hits = sub[sub["source_date_present"] == True]  # noqa: E712
    return hits


def snap_hits(code, date):
    sub = disc[(disc["code"] == code) & (disc["target_date"] == date) & disc["source_type"].isin(SNAP_TYPES)]
    hits = sub[sub["source_date_present"] == True]  # noqa: E712
    return hits


def minute_hit(code, date):
    sub = disc[(disc["code"] == code) & (disc["target_date"] == date) & (disc["source_type"] == "MINUTE_ONLY")]
    return bool(len(sub) and bool(sub["source_date_present"].iloc[0]))


# ---------------- D. 根因 ----------------
rows = []
for _, r in miss.iterrows():
    code = r["code"]
    d2, d3 = str(r["expected_d2_date"]), str(r["expected_d3_date"])
    ca = cache_audit[(cache_audit["event_id"] == r["event_id"])]
    aud = ca["audit_result"].iloc[0] if len(ca) else "OTHER"
    refreshed = bool(len(ca) and bool(ca["cache_modified_after_v021_build"].iloc[0]))
    d2_daily = daily_hits(code, d2)
    d3_daily = daily_hits(code, d3)
    d2_snap = snap_hits(code, d2)
    d3_snap = snap_hits(code, d3)
    d2_min = minute_hit(code, d2)
    d3_min = minute_hit(code, d3)
    evidence = []
    recoverable = False
    rec_sources = []
    confidence = "LOW"
    root = "UNKNOWN_REQUIRES_MANUAL_REVIEW"

    if refreshed:
        # 构建后被外部进程刷新, 主日线缓存现在已含日期
        root = "PRIMARY_CACHE_NOT_UPDATED_AFTER_FORWARD"
        evidence.append("primary daily cache refreshed externally after v0.2.1 build (mtime > build); dates now present")
        recoverable = True
        rec_sources.append("PRIMARY_DAILY_CACHE")
        confidence = "HIGH"
    elif not d2_daily.empty or not d3_daily.empty:
        root = "DATA_EXISTS_IN_OTHER_LOCAL_DAILY_SOURCE"
        for h in list(d2_daily.itertuples()) + list(d3_daily.itertuples()):
            evidence.append(f"{h.required_role}@{h.target_date} in {h.source_type}: {h.source_path}")
        recoverable = True
        rec_sources = sorted(set([h.source_type for h in list(d2_daily.itertuples()) + list(d3_daily.itertuples())]))
        confidence = "HIGH"
    elif not d2_snap.empty or not d3_snap.empty:
        root = "DATA_EXISTS_IN_FORWARD_OR_HOLDOUT_SNAPSHOT"
        for h in list(d2_snap.itertuples()) + list(d3_snap.itertuples()):
            evidence.append(f"{h.required_role}@{h.target_date} in {h.source_type} (minute 口径): {h.source_path}")
        confidence = "MEDIUM"
    elif d2_min or d3_min:
        root = "PRIMARY_CACHE_NOT_UPDATED_AFTER_FORWARD"
        evidence.append("5min 缓存证明期望日期有行情, 但无任何本地日线来源含该日期; 主日线缓存未随 forward 更新")
        confidence = "HIGH"
    else:
        root = "TRUE_LOCAL_DATA_ABSENCE"
        evidence.append("期望日期既无日线也无 5min 本地记录")
        confidence = "HIGH"

    manual = root in ("UNKNOWN_REQUIRES_MANUAL_REVIEW",)
    rows.append({
        "event_id": r["event_id"], "code": code, "signal_date": r["signal_date"],
        "expected_d2_date": d2, "expected_d3_date": d3,
        "root_cause": root,
        "evidence_paths": "; ".join(evidence),
        "evidence_summary": "; ".join(evidence[:4]),
        "recoverable_from_existing_local_data": recoverable,
        "recommended_recovery_source": "; ".join(rec_sources) if rec_sources else "",
        "confidence": confidence,
        "manual_review_required": manual,
        "audit_result": aud,
    })

rc = pd.DataFrame(rows)
rc.to_csv(OUT / "v004c_missing_daily_root_cause.csv", index=False, encoding="utf-8-sig")
print("[D] root cause counts:", rc["root_cause"].value_counts().to_dict())
print("    recoverable:", int(rc["recoverable_from_existing_local_data"].sum()), "/", len(rc))

# ---------------- E. 冲突 ----------------
print("[E] conflicts ...")
conflict_rows = []
seen = set()
for _, r in miss.iterrows():
    code = r["code"]
    for date in (str(r["expected_d2_date"]), str(r["expected_d3_date"])):
        if (code, date) in seen:
            continue
        seen.add((code, date))
        hits = daily_hits(code, date)
        if len(hits) < 2:
            continue
        recs = []
        for h in hits.itertuples():
            recs.append({"type": h.source_type, "path": h.source_path,
                         "open": h.open, "high": h.high, "low": h.low, "close": h.close})
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                price_equal = all(
                    (a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and abs(a[k] - b[k]) < 1e-9)
                    for k in ("open", "high", "low", "close"))
                # target7 是否可能翻转: 用本行另一日的 primary 价格估算
                other_date = str(r["expected_d3_date"]) if date == str(r["expected_d2_date"]) else str(r["expected_d2_date"])
                other_hits = daily_hits(code, other_date)
                t7_diff = tail_diff = None
                if not other_hits.empty:
                    o = other_hits.iloc[0]
                    if date == str(r["expected_d2_date"]):  # 冲突在 D2 open
                        base = o["high"]  # D3 high
                        ra = (base / a["open"] - 1) if (base and a["open"]) else None
                        rb = (base / b["open"] - 1) if (base and b["open"]) else None
                        if ra is not None and rb is not None:
                            t7_diff = bool((ra >= 0.07) != (rb >= 0.07))
                    else:  # 冲突在 D3 high
                        base = o["open"]  # D2 open
                        ra = (a["high"] / base - 1) if (a["high"] and base) else None
                        rb = (b["high"] / base - 1) if (b["high"] and base) else None
                        if ra is not None and rb is not None:
                            t7_diff = bool((ra >= 0.07) != (rb >= 0.07))
                conflict_rows.append({
                    "code": code, "trade_date": date,
                    "source_a": a["type"], "source_b": b["type"],
                    "open_a": a["open"], "open_b": b["open"],
                    "high_a": a["high"], "high_b": b["high"],
                    "low_a": a["low"], "low_b": b["low"],
                    "close_a": a["close"], "close_b": b["close"],
                    "volume_a": None, "volume_b": None, "amount_a": None, "amount_b": None,
                    "price_fields_equal": price_equal,
                    "target7_would_differ": t7_diff, "tail_loss_would_differ": tail_diff,
                    "recommended_authoritative_source": "PRIMARY_DAILY_CACHE",
                    "recommendation_reason": "主日线缓存为 loader 实际读取路径 (data/cache/daily); 冲突仅记录不合并",
                })
conf = pd.DataFrame(conflict_rows)
conf.to_csv(OUT / "v004c_local_daily_source_conflicts.csv", index=False, encoding="utf-8-sig")
print("[E] conflict rows:", len(conf), "| price-equal pairs:", int(conf["price_fields_equal"].sum()),
      "| target7 would differ:", int(conf["target7_would_differ"].fillna(False).sum()))

# ---------------- F. 恢复汇总 ----------------
print("[F] recovery summary ...")
sum_rows = []
rc2 = rc.copy()
rc2["month"] = rc2["signal_date"].str[:7]
for (root, month), g in rc2.groupby(["root_cause", "month"]):
    sum_rows.append({
        "root_cause": root, "month": month,
        "missing_row_count": int(len(g)),
        "independent_event_count": int(g["event_id"].nunique()),
        "independent_code_count": int(g["code"].nunique()),
        "recoverable_from_existing_local_daily": int(g["recoverable_from_existing_local_data"].sum()),
        "recoverable_from_snapshot": int(((g["root_cause"] == "DATA_EXISTS_IN_FORWARD_OR_HOLDOUT_SNAPSHOT")).sum()),
        "minute_only_available": int(((g["root_cause"] == "PRIMARY_CACHE_NOT_UPDATED_AFTER_FORWARD") & ~g["recoverable_from_existing_local_data"]).sum()),
        "true_absence_count": int((g["root_cause"] == "TRUE_LOCAL_DATA_ABSENCE").sum()),
        "manual_review_count": int(g["manual_review_required"].sum()),
    })
sum_df = pd.DataFrame(sum_rows)
# 总体六项
total = {
    "recoverable_from_existing_local_daily": int(rc["recoverable_from_existing_local_data"].sum()),
    "recoverable_only_from_snapshot_minute": int((rc["root_cause"] == "DATA_EXISTS_IN_FORWARD_OR_HOLDOUT_SNAPSHOT").sum()),
    "minute_only_available": int(((rc["root_cause"] == "PRIMARY_CACHE_NOT_UPDATED_AFTER_FORWARD") & ~rc["recoverable_from_existing_local_data"]).sum()),
    "truly_not_found": int((rc["root_cause"] == "TRUE_LOCAL_DATA_ABSENCE").sum()),
    "conflict_row_count": int(len(conf)),
    "manual_review_count": int(rc["manual_review_required"].sum()),
}
sum_df.to_csv(OUT / "v004c_daily_recovery_summary.csv", index=False, encoding="utf-8-sig")
print("[F] totals:", total)

# ---------------- G. 建议恢复映射 (不执行) ----------------
print("[G] recovery map ...")
map_rows = []
seen2 = set()
for _, r in rc.iterrows():
    if not r["recoverable_from_existing_local_data"]:
        continue
    code = r["code"]
    for date, role in ((r["expected_d2_date"], "D2_OPEN"), (r["expected_d3_date"], "D3_HIGH"),
                       (r["expected_d3_date"], "D3_CLOSE")):
        if (code, date) in seen2 and date == r["expected_d3_date"]:
            # D3_HIGH 与 D3_CLOSE 共享同一来源行, 但保留两行角色
            pass
        seen2.add((code, date))
        hits = daily_hits(code, date)
        if hits.empty:
            continue
        # 优先级: PRIMARY > ALTERNATIVE > OFFICIAL_DAILY_REPORT
        prio = {"PRIMARY_DAILY_CACHE": 0, "ALTERNATIVE_DAILY_CACHE": 1, "OFFICIAL_DAILY_REPORT": 2}
        h = hits.sort_values("source_type", key=lambda s: s.map(prio)).iloc[0]
        others = [x for x in hits.itertuples() if x.source_path != h["source_path"]]
        conflict_status = "AGREED" if not others else "MULTI_SOURCE"
        if others:
            price_equal = all(
                (getattr(o, "open") is None and h["open"] is None) or
                (getattr(o, "open") is not None and h["open"] is not None and abs(getattr(o, "open") - h["open"]) < 1e-9)
                for o in others)
            conflict_status = "CONFLICT" if not price_equal else "MULTI_SOURCE_AGREED"
        confidence = "HIGH" if h["source_type"] == "PRIMARY_DAILY_CACHE" else "MEDIUM"
        map_rows.append({
            "code": code, "trade_date": date, "required_role": role,
            "proposed_source_path": h["source_path"], "proposed_source_type": h["source_type"],
            "open": h["open"], "high": h["high"], "low": h["low"], "close": h["close"],
            "volume": None, "amount": None,
            "source_sha256": h["source_sha256"],
            "confidence": confidence, "conflict_status": conflict_status,
            "eligible_for_future_recovery": True,
            "reason": "只生成建议映射, 未写入任何缓存/数据集; 恢复须经人工审阅后执行",
        })
rec_map = pd.DataFrame(map_rows)
rec_map.to_csv(OUT / "v004c_proposed_daily_recovery_map.csv", index=False, encoding="utf-8-sig")
print("[G] recovery map rows:", len(rec_map))
print("[done D/E/F/G]")

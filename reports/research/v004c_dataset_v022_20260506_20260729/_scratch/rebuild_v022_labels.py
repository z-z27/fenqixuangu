# -*- coding: utf-8 -*-
"""v004c dataset 0.2.2: 日线标签重建 (只读)

- 读取更新后的 data/cache/daily (未复权, tencent_daily)
- 沿用 v0.2.1 的 expected_d2/expected_d3 日期与停牌证明规则 (不重新推导日期)
- 对 v0.2.1 有 proven 偏移的行 (actual != expected 且 suspension_proven) 使用其复牌日期
- 全量 1020 行重算; 输出全量表/训练表/五月表/标签质量
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
TP = ROOT / "reports" / "research" / "v004c_training_prep_20260601_20260729"
OUT = ROOT / "reports" / "research" / "v004c_dataset_v022_20260506_20260729"
DAILY_DIR = ROOT / "data" / "cache" / "daily"

TRAIN_START, TRAIN_END = "2026-06-01", "2026-07-29"
MAY_START, MAY_END = "2026-05-06", "2026-05-31"

v21 = pd.read_csv(TP / "v004c_dataset_v021_all.csv", dtype={"code": str})
v21["code"] = v21["code"].astype(str).str.zfill(6)
print("v0.2.1 rows:", len(v21))

daily_cache: dict[str, pd.DataFrame | None] = {}
daily_meta: dict[str, dict] = {}


def get_daily(code: str):
    if code in daily_cache:
        return daily_cache[code]
    try:
        p = DAILY_DIR / f"{code}_daily.pkl"
        if not p.exists():
            daily_cache[code] = None
            daily_meta[code] = {"exists": False}
            return None
        st = os.stat(p)
        daily_meta[code] = {
            "exists": True,
            "path": str(p),
            "mtime": pd.Timestamp(st.st_mtime, unit="s", tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"),
            "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
        }
        d = pd.read_pickle(p)
        d = d.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        daily_cache[code] = d.dropna(subset=["date"])
    except Exception:
        daily_cache[code] = None
        daily_meta[code] = {"exists": False}
    return daily_cache[code]


def bar_at(code: str, date: str):
    d = get_daily(code)
    if d is None or d.empty:
        return None
    rows = d[d["date"] == str(date)]
    if rows.empty:
        return None
    return rows.iloc[0]


def price(code: str, date: str, col: str):
    b = bar_at(code, date)
    if b is None:
        return None
    v = b.get(col)
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def valid_price(v):
    return v is not None and v > 0


rows = []
for _, r in v21.iterrows():
    code = r["code"]
    exp_d2 = str(r["expected_d2_date"]) if pd.notna(r["expected_d2_date"]) else ""
    exp_d3 = str(r["expected_d3_date"]) if pd.notna(r["expected_d3_date"]) else ""
    # 标签日期: expected 优先; v0.2.1 proven 偏移行用其复牌日期
    act_d2 = str(r["actual_daily_d2_date"]) if pd.notna(r["actual_daily_d2_date"]) else ""
    act_d3 = str(r["actual_daily_d3_date"]) if pd.notna(r["actual_daily_d3_date"]) else ""
    proven = str(r["suspension_proof_status_daily"]) == "proven"
    d2_date = exp_d2
    d3_date = exp_d3
    d2_source = "expected_date_daily_cache"
    d3_source = "expected_date_daily_cache"
    if exp_d2 and bar_at(code, exp_d2) is None and proven and act_d2 and act_d2 != exp_d2:
        d2_date = act_d2
        d2_source = "suspension_proven_resume_date_daily_cache"
    if exp_d3 and bar_at(code, exp_d3) is None and proven and act_d3 and act_d3 != exp_d3:
        d3_date = act_d3
        d3_source = "suspension_proven_resume_date_daily_cache"

    d2_open = price(code, d2_date, "open") if d2_date else None
    d3_high = price(code, d3_date, "high") if d3_date else None
    d3_close = price(code, d3_date, "close") if d3_date else None

    reasons = []
    if not d2_date:
        reasons.append("expected_d2_missing")
    if not d3_date:
        reasons.append("expected_d3_missing")
    if d2_date and not valid_price(d2_open):
        reasons.append(f"d2_open_invalid({d2_date})")
    if d3_date and not valid_price(d3_high):
        reasons.append(f"d3_high_invalid({d3_date})")
    if d3_date and not valid_price(d3_close):
        reasons.append(f"d3_close_invalid({d3_date})")
    daily_ok = not reasons

    high_ret = (d3_high / d2_open - 1.0) if (valid_price(d3_high) and valid_price(d2_open)) else None
    close_ret = (d3_close / d2_open - 1.0) if (valid_price(d3_close) and valid_price(d2_open)) else None
    target7 = bool(high_ret is not None and high_ret >= 0.07)
    tail = bool(close_ret is not None and close_ret <= -0.05)

    d1_ok = bool(r["d1_minute_complete"])
    meta = daily_meta.get(code, {"exists": False})
    rows.append({
        **r.to_dict(),
        "d2_open_daily": d2_open,
        "d3_high_daily": d3_high,
        "d3_close_daily": d3_close,
        "daily_d2open_to_d3high_return": high_ret,
        "daily_d2open_to_d3close_return": close_ret,
        "target7_daily_d2open_d3high": target7,
        "tail_loss_daily_5pct": tail,
        "label_d2_date": d2_date,
        "label_d3_date": d3_date,
        "daily_label_source_d2": d2_source,
        "daily_label_source_d3": d3_source,
        "daily_label_cache_path": meta.get("path", ""),
        "daily_label_cache_mtime": meta.get("mtime", ""),
        "daily_label_cache_sha256": meta.get("sha256", ""),
        "daily_label_quality_ok": daily_ok,
        "daily_label_quality_reason": "|".join(reasons) if reasons else "ok",
        "target_training_eligible": bool(daily_ok and d1_ok and valid_price(d2_open) and valid_price(d3_high)),
        "tail_training_eligible": bool(daily_ok and d1_ok and valid_price(d2_open) and valid_price(d3_close)),
    })

v22 = pd.DataFrame(rows)
print("v0.2.2 rows:", len(v22))
print("daily_label_quality_ok:", int(v22["daily_label_quality_ok"].sum()), "/", len(v22))
print("target7_daily:", int(v22["target7_daily_d2open_d3high"].sum()),
      "| tail_daily:", int(v22["tail_loss_daily_5pct"].sum()))

# sample_role_v022
def role022(r):
    sig = str(r["signal_date"])
    if MAY_START <= sig <= MAY_END:
        if r["daily_label_quality_ok"] and r["d1_minute_complete"]:
            return "SENSITIVITY_MAY_COMPLETE"
        return "AUDIT_ONLY_MAY_INCOMPLETE"
    if TRAIN_START <= sig <= TRAIN_END:
        if r["daily_label_quality_ok"] and r["d1_minute_complete"]:
            return "TRAIN_PRIMARY"
        return "EXCLUDED_QUALITY"
    return "EXCLUDED_QUALITY"

v22["sample_role_v022"] = v22.apply(role022, axis=1)
print("sample_role_v022:", v22["sample_role_v022"].value_counts().to_dict())

v22.to_csv(OUT / "v004c_dataset_v022_all.csv", index=False, encoding="utf-8-sig")

# 训练表 / 五月表
train = v22[v22["sample_role_v022"] == "TRAIN_PRIMARY"].reset_index(drop=True)
may = v22[v22["sample_role_v022"] == "SENSITIVITY_MAY_COMPLETE"].reset_index(drop=True)
train.to_csv(OUT / "v004c_training_primary_v022.csv", index=False, encoding="utf-8-sig")
may.to_csv(OUT / "v004c_may_sensitivity_v022.csv", index=False, encoding="utf-8-sig")
print("training v022:", len(train), "| may v022:", len(may))
print("training d0/post:", train["stage_group"].value_counts().to_dict())
print("training target7:", int(train["target7_daily_d2open_d3high"].sum()),
      "| tail:", int(train["tail_loss_daily_5pct"].sum()))

# 标签质量报告
lq_cols = ["event_id", "code", "signal_date", "stage_group", "sample_role_v022",
           "expected_d2_date", "expected_d3_date", "label_d2_date", "label_d3_date",
           "daily_label_source_d2", "daily_label_source_d3",
           "daily_label_cache_path", "daily_label_cache_mtime", "daily_label_cache_sha256",
           "d2_open_daily", "d3_high_daily", "d3_close_daily",
           "daily_label_quality_ok", "daily_label_quality_reason",
           "target_training_eligible", "tail_training_eligible"]
lq = v22[[c for c in lq_cols if c in v22.columns]].copy()
lq_rows = []
for reason, n in v22.loc[v22["daily_label_quality_ok"] == False, "daily_label_quality_reason"].value_counts().items():  # noqa: E712
    lq_rows.append({"section": "failure_summary", "daily_label_quality_reason": str(reason), "count": int(n)})
lq_sum = pd.DataFrame(lq_rows)
lq_out = pd.concat([lq.assign(section="per_row"), lq_sum], ignore_index=True)
lq_out.to_csv(OUT / "v004c_daily_label_quality_v022.csv", index=False, encoding="utf-8-sig")
print("label quality report:", len(lq_out), "rows; failures:", int((v22["daily_label_quality_ok"] == False).sum()))  # noqa: E712
print("[done]")

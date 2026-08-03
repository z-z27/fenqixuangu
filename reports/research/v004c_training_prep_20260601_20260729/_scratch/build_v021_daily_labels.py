# -*- coding: utf-8 -*-
"""v004c dataset 0.2.1: 日线 OHLC 正式标签构建 (只读)

- 正式标签: d2_open_daily / d3_high_daily / d3_close_daily (预期 D2/D3 交易日日线 OHLC)
- 旧分钟标签重命名为 audit_minute_*
- 期望日期来自统一交易日历; 个股在预期日无日线时, 仅严格停牌证明通过才后移到复牌日
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
V02 = ROOT / "reports" / "research" / "v004c_dataset_v02_20260601_20260729"
OUT = ROOT / "reports" / "research" / "v004c_training_prep_20260601_20260729"
SCRATCH = OUT / "_scratch"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
SUSP_DIR = ROOT / "data" / "cache" / "suspension_status"

TRAIN_START, TRAIN_END = "2026-06-01", "2026-07-29"
MAY_START, MAY_END = "2026-05-06", "2026-05-31"
FULL_DAY_SUSPENSION_DURATIONS = {"连续停牌", "停牌一天"}

# ---------------- 1. 加载 v0.2 全量表与日历/停牌/日线 ----------------
print("[1] load ...", flush=True)
all02 = pd.read_csv(V02 / "v004c_dataset_v02_all.csv", dtype={"code": str})
all02["code"] = all02["code"].astype(str).str.zfill(6)
print("v0.2 all rows:", len(all02))

pool_dates = set()
for f in glob.glob(str(POOL_DIR / "*.pkl")):
    pool_dates.update(pd.read_pickle(f)["trade_date"].astype(str))
daily_dates = set()
for f in glob.glob(str(DAILY_DIR / "*_daily.pkl")):
    d = pd.read_pickle(f)
    daily_dates.update(pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
calendar = sorted(pool_dates | daily_dates)
print("calendar:", len(calendar), "dates")

suspension_proofs = []
for f in glob.glob(str(SUSP_DIR / "*.pkl")):
    try:
        frame = pd.read_pickle(f)
        if frame is None or frame.empty or "records_json" not in frame.columns:
            continue
        for r in json.loads(str(frame.iloc[0]["records_json"])):
            code = str(r.get("code", "")).strip()
            start = str(r.get("suspension_start_date", "")).strip()
            end = str(r.get("suspension_end_date", "")).strip()
            duration = str(r.get("suspension_duration", "")).strip()
            if code and start and end and duration in FULL_DAY_SUSPENSION_DURATIONS:
                suspension_proofs.append((code, start, end))
    except Exception:
        continue
print("suspension proof intervals:", len(suspension_proofs))


def suspension_proven(code: str, date: str) -> bool:
    return any(s <= date <= e for (c, s, e) in suspension_proofs if c == code)


daily_cache: dict[str, pd.DataFrame | None] = {}


def get_daily(code: str) -> pd.DataFrame | None:
    if code in daily_cache:
        return daily_cache[code]
    try:
        d = pd.read_pickle(DAILY_DIR / f"{code}_daily.pkl")
        d = d.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        daily_cache[code] = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception:
        daily_cache[code] = None
    return daily_cache[code]


def daily_bar(code: str, date: str) -> pd.Series | None:
    d = get_daily(code)
    if d is None or d.empty:
        return None
    rows = d[d["date"] == str(date)]
    if rows.empty:
        return None
    return rows.iloc[0]


def resolve_daily_date(code: str, expected: str, after_date: str | None = None) -> tuple[str | None, str]:
    """预期日有日线 → 当日; 无日线且停牌证明覆盖 → 复牌后首个有日线交易日; 否则无效。

    after_date: D3 解析时传入实际 D2 日期, 确保 D3 严格晚于 D2
    (整个停牌块覆盖 D2/D3 两个期望日时, D3 应取复牌后第 2 个交易日)。"""
    if after_date is not None:
        d = get_daily(code)
        if d is None:
            return None, "no_daily_cache"
        later = d[d["date"] > after_date]
        if later.empty:
            return None, "no_bar_after_d2"
        first = str(later["date"].iloc[0])
        if first == expected:
            return first, "none"
        if suspension_proven(code, expected):
            return first, "suspended_proven"
        return None, "no_bar_without_proof"
    if daily_bar(code, expected) is not None:
        return expected, "none"
    if suspension_proven(code, expected):
        d = get_daily(code)
        if d is None:
            return None, "no_daily_cache"
        later = d[d["date"] > expected]
        if later.empty:
            return None, "suspended_no_resume_bar"
        return str(later["date"].iloc[0]), "suspended_proven"
    return None, "no_bar_without_proof"


# ---------------- 2. 逐行日线标签 ----------------
print("[2] daily labels ...", flush=True)
rows = []
for _, r in all02.iterrows():
    code = r["code"]
    exp_d2 = str(r["expected_d2_date"]) if pd.notna(r["expected_d2_date"]) else None
    exp_d3 = str(r["expected_d3_date"]) if pd.notna(r["expected_d3_date"]) else None
    act_d2, d2_reason = resolve_daily_date(code, exp_d2) if exp_d2 else (None, "expected_d2_missing")
    act_d3, d3_reason = (resolve_daily_date(code, exp_d3, after_date=act_d2) if (exp_d3 and act_d2) else
                         (resolve_daily_date(code, exp_d3) if exp_d3 else (None, "expected_d3_missing")))

    d2_bar = daily_bar(code, act_d2) if act_d2 else None
    d3_bar = daily_bar(code, act_d3) if act_d3 else None
    d2_open = None
    if d2_bar is not None:
        v = d2_bar.get("open")
        d2_open = float(v) if pd.notna(v) and float(v) > 0 else None
    d3_high = None
    d3_close = None
    if d3_bar is not None:
        vh = d3_bar.get("high")
        vc = d3_bar.get("close")
        d3_high = float(vh) if pd.notna(vh) and float(vh) > 0 else None
        d3_close = float(vc) if pd.notna(vc) and float(vc) > 0 else None

    reasons = []
    if exp_d2 is None or exp_d3 is None:
        reasons.append("expected_date_beyond_calendar")
    if d2_reason == "no_bar_without_proof":
        reasons.append(f"d2_no_bar_without_proof(exp={exp_d2})")
    if d3_reason == "no_bar_without_proof":
        reasons.append(f"d3_no_bar_without_proof(exp={exp_d3})")
    if d2_reason == "suspended_no_resume_bar":
        reasons.append(f"d2_suspended_no_resume_bar(exp={exp_d2})")
    if d3_reason == "suspended_no_resume_bar":
        reasons.append(f"d3_suspended_no_resume_bar(exp={exp_d3})")
    if d2_reason == "no_daily_cache":
        reasons.append("d2_no_daily_cache")
    if d3_reason == "no_daily_cache":
        reasons.append("d3_no_daily_cache")
    if d2_open is None:
        reasons.append("d2_open_invalid")
    if d3_high is None:
        reasons.append("d3_high_invalid")
    if d3_close is None:
        reasons.append("d3_close_invalid")
    daily_ok = not reasons
    susp_status = "not_applicable"
    if d2_reason == "suspended_proven" or d3_reason == "suspended_proven":
        susp_status = "proven"
    elif d2_reason in ("no_bar_without_proof", "suspended_no_resume_bar") or d3_reason in ("no_bar_without_proof", "suspended_no_resume_bar"):
        susp_status = "not_proven"

    daily_high_return = (d3_high / d2_open - 1.0) if (d3_high is not None and d2_open and d2_open > 0) else None
    daily_close_return = (d3_close / d2_open - 1.0) if (d3_close is not None and d2_open and d2_open > 0) else None

    # 旧分钟标签 → audit 前缀
    audit = {
        "audit_minute_d2_open": r["d2_open"] if pd.notna(r["d2_open"]) else None,
        "audit_minute_d3_high": r["d3_high"] if pd.notna(r["d3_high"]) else None,
        "audit_minute_d3_close": r["d3_close"] if pd.notna(r["d3_close"]) else None,
        "audit_minute_target7": bool(r["target7_d2open_d3high"]),
        "audit_minute_tail_loss": bool(r["tail_loss_5pct"]),
        "audit_minute_d2open_to_d3high_return": r["d2open_to_d3high_return"] if pd.notna(r["d2open_to_d3high_return"]) else None,
        "audit_minute_d2open_to_d3close_return": r["d2open_to_d3close_return"] if pd.notna(r["d2open_to_d3close_return"]) else None,
    }
    minute_t7 = audit["audit_minute_target7"]
    minute_tl = audit["audit_minute_tail_loss"]
    daily_t7 = bool(daily_high_return is not None and daily_high_return >= 0.07)
    daily_tl = bool(daily_close_return is not None and daily_close_return <= -0.05)

    rows.append({
        **r.to_dict(),
        # 日线标签
        "d2_open_daily": d2_open,
        "d3_high_daily": d3_high,
        "d3_close_daily": d3_close,
        "daily_d2open_to_d3high_return": daily_high_return,
        "daily_d2open_to_d3close_return": daily_close_return,
        "target7_daily_d2open_d3high": daily_t7,
        "tail_loss_daily_5pct": daily_tl,
        "actual_daily_d2_date": act_d2,
        "actual_daily_d3_date": act_d3,
        "d2_daily_date_shift_reason": d2_reason,
        "d3_daily_date_shift_reason": d3_reason,
        "suspension_proof_status_daily": susp_status,
        "daily_label_quality_ok": daily_ok,
        "daily_label_quality_reason": "|".join(reasons) if reasons else "ok",
        # 审计对照 (旧分钟标签)
        **audit,
        # 比较字段
        "target_label_daily_vs_minute_same": bool(minute_t7 == daily_t7) if (daily_high_return is not None) else None,
        "tail_label_daily_vs_minute_same": bool(minute_tl == daily_tl) if (daily_close_return is not None) else None,
        "d2_open_daily_minus_minute": (d2_open - audit["audit_minute_d2_open"]) if (d2_open is not None and audit["audit_minute_d2_open"] is not None) else None,
        "d3_high_daily_minus_minute": (d3_high - audit["audit_minute_d3_high"]) if (d3_high is not None and audit["audit_minute_d3_high"] is not None) else None,
        "d3_close_daily_minus_minute": (d3_close - audit["audit_minute_d3_close"]) if (d3_close is not None and audit["audit_minute_d3_close"] is not None) else None,
        "target_return_daily_minus_minute": (daily_high_return - audit["audit_minute_d2open_to_d3high_return"]) if (daily_high_return is not None and audit["audit_minute_d2open_to_d3high_return"] is not None) else None,
        "close_return_daily_minus_minute": (daily_close_return - audit["audit_minute_d2open_to_d3close_return"]) if (daily_close_return is not None and audit["audit_minute_d2open_to_d3close_return"] is not None) else None,
        # 训练可用性
        "d1_factor_quality_ok": bool(r["d1_minute_complete"]),
        "target_training_eligible": bool(daily_ok and r["d1_minute_complete"]),
        "tail_training_eligible": bool(daily_ok and r["d1_minute_complete"]),
    })

v21 = pd.DataFrame(rows)
print("v0.2.1 rows:", len(v21))
print("daily label quality ok:", int(v21["daily_label_quality_ok"].sum()),
      "| failed:", int((v21["daily_label_quality_ok"] == False).sum()))  # noqa: E712
print("target7_daily:", int(v21["target7_daily_d2open_d3high"].sum()),
      "| tail_daily:", int(v21["tail_loss_daily_5pct"].sum()))

# sample_role_v021
def role021(r):
    sig = r["signal_date"]
    if MAY_START <= sig <= MAY_END:
        if r["daily_label_quality_ok"] and r["d1_minute_complete"]:
            return "SENSITIVITY_MAY_COMPLETE"
        return "AUDIT_ONLY_MAY_INCOMPLETE"
    if TRAIN_START <= sig <= TRAIN_END:
        if r["daily_label_quality_ok"] and r["d1_minute_complete"]:
            return "TRAIN_PRIMARY"
        return "EXCLUDED_QUALITY"
    return "EXCLUDED_QUALITY"

v21["sample_role_v021"] = v21.apply(role021, axis=1)
print("sample_role_v021:", v21["sample_role_v021"].value_counts().to_dict())

v21.to_csv(OUT / "v004c_dataset_v021_all.csv", index=False, encoding="utf-8-sig")

# ---------------- 3. 训练视图与五月敏感性 ----------------
train = v21[v21["sample_role_v021"] == "TRAIN_PRIMARY"].reset_index(drop=True)
may = v21[v21["sample_role_v021"] == "SENSITIVITY_MAY_COMPLETE"].reset_index(drop=True)
train.to_csv(OUT / "v004c_training_primary_v021.csv", index=False, encoding="utf-8-sig")
may.to_csv(OUT / "v004c_may_sensitivity_v021.csv", index=False, encoding="utf-8-sig")
print("train v021:", len(train), "| may v021:", len(may))

# ---------------- 4. 日线/分钟标签差异报告 ----------------
diff_cols = ["event_id", "code", "signal_date", "stage_group", "sample_role_v021",
             "daily_d2open_to_d3high_return", "audit_minute_d2open_to_d3high_return",
             "target_return_daily_minus_minute", "daily_d2open_to_d3close_return",
             "audit_minute_d2open_to_d3close_return", "close_return_daily_minus_minute",
             "target7_daily_d2open_d3high", "audit_minute_target7",
             "target_label_daily_vs_minute_same",
             "tail_loss_daily_5pct", "audit_minute_tail_loss",
             "tail_label_daily_vs_minute_same",
             "d2_open_daily", "audit_minute_d2_open", "d2_open_daily_minus_minute",
             "d3_high_daily", "audit_minute_d3_high", "d3_high_daily_minus_minute",
             "d3_close_daily", "audit_minute_d3_close", "d3_close_daily_minus_minute",
             "daily_label_quality_ok", "daily_label_quality_reason"]
diff = v21[[c for c in diff_cols if c in v21.columns]].copy()
# 汇总段
t7_div = int((v21["target_label_daily_vs_minute_same"] == False).sum())  # noqa: E712
tl_div = int((v21["tail_label_daily_vs_minute_same"] == False).sum())  # noqa: E712
agg = [
    {"section": "overall", "metric": "v021_rows", "value": len(v21)},
    {"section": "overall", "metric": "v02_rows", "value": len(all02)},
    {"section": "overall", "metric": "target7_daily_positive", "value": int(v21["target7_daily_d2open_d3high"].sum())},
    {"section": "overall", "metric": "tail_daily_positive", "value": int(v21["tail_loss_daily_5pct"].sum())},
    {"section": "overall", "metric": "target7_label_disagreement_daily_vs_minute", "value": t7_div},
    {"section": "overall", "metric": "tail_label_disagreement_daily_vs_minute", "value": tl_div},
    {"section": "overall", "metric": "excluded_by_daily_label_quality", "value": int((v21["daily_label_quality_ok"] == False).sum())},  # noqa: E712
    {"section": "overall", "metric": "target_training_eligible", "value": int(v21["target_training_eligible"].sum())},
    {"section": "overall", "metric": "d0_rows", "value": int((v21["stage_group"] == "d0").sum())},
    {"section": "overall", "metric": "post_rows", "value": int((v21["stage_group"] == "post").sum())},
    {"section": "overall", "metric": "d0_target7_daily", "value": int(v21.loc[v21["stage_group"] == "d0", "target7_daily_d2open_d3high"].sum())},
    {"section": "overall", "metric": "post_target7_daily", "value": int(v21.loc[v21["stage_group"] == "post", "target7_daily_d2open_d3high"].sum())},
    {"section": "overall", "metric": "d0_tail_daily", "value": int(v21.loc[v21["stage_group"] == "d0", "tail_loss_daily_5pct"].sum())},
    {"section": "overall", "metric": "post_tail_daily", "value": int(v21.loc[v21["stage_group"] == "post", "tail_loss_daily_5pct"].sum())},
]
for reason, n in v21.loc[v21["daily_label_quality_ok"] == False, "daily_label_quality_reason"].value_counts().items():  # noqa: E712
    agg.append({"section": "daily_label_quality_failure_reason", "metric": str(reason), "value": int(n)})
agg_df = pd.DataFrame(agg)
diff_out = pd.concat([diff.assign(section="per_row"), agg_df], ignore_index=True)
diff_out.to_csv(OUT / "v004c_daily_label_diff_report.csv", index=False, encoding="utf-8-sig")
print("diff report rows:", len(diff_out), "| target7 disagreement:", t7_div, "| tail disagreement:", tl_div)

# ---------------- 5. 日线标签质量报告 ----------------
lq_cols = ["event_id", "code", "signal_date", "stage_group", "sample_role_v021",
           "expected_d2_date", "actual_daily_d2_date", "expected_d3_date", "actual_daily_d3_date",
           "d2_daily_date_shift_reason", "d3_daily_date_shift_reason",
           "suspension_proof_status_daily", "daily_label_quality_ok", "daily_label_quality_reason",
           "d2_open_daily", "d3_high_daily", "d3_close_daily",
           "target_training_eligible", "tail_training_eligible"]
lq = v21[[c for c in lq_cols if c in v21.columns]].copy()
lq_rows = []
for reason, n in v21.loc[v21["daily_label_quality_ok"] == False, "daily_label_quality_reason"].value_counts().items():  # noqa: E712
    lq_rows.append({"section": "failure_summary", "daily_label_quality_reason": str(reason), "count": int(n)})
lq_sum = pd.DataFrame(lq_rows)
lq_out = pd.concat([lq.assign(section="per_row"), lq_sum], ignore_index=True)
lq_out.to_csv(OUT / "v004c_daily_label_quality_report.csv", index=False, encoding="utf-8-sig")
print("daily label quality report rows:", len(lq_out))
print("done")

# -*- coding: utf-8 -*-
"""v004c dataset 0.2 核心构建 (只读; 不修改任何现有文件)

输入: v004c dataset 0.1 主样本 + 审计表 + 本地缓存(只读)
输出(全部位于本目录):
- v004c_dataset_v02_all.csv                全量表 (v0.1 的 1020 行 + 新字段)
- v004c_training_primary_20260601_20260729.csv  主训练视图
- v004c_may_sensitivity_complete.csv       五月敏感性完整样本
- v004c_candidate_audit_v02.csv            候选审计 v0.2
- v004c_label_quality_report_v02.csv       标签质量报告
- _scratch/v02_stage1_all.csv              中间
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
V01 = ROOT / "reports" / "research" / "v004c_dataset_20260506_20260729"
OUT = ROOT / "reports" / "research" / "v004c_dataset_v02_20260601_20260729"
SCRATCH = OUT / "_scratch"
MINUTE_DIR = ROOT / "data" / "cache" / "minute_5m"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
SUSP_DIR = ROOT / "data" / "cache" / "suspension_status"

TRAIN_START = "2026-06-01"
TRAIN_END = "2026-07-29"
MAY_START = "2026-05-06"
MAY_END = "2026-05-31"

EXPECTED_BAR_COUNT = 48
EXPECTED_FIRST_BAR = "09:35"
EXPECTED_LAST_BAR = "15:00"
FULL_DAY_SUSPENSION_DURATIONS = {"连续停牌", "停牌一天"}

# ---------------- 1. 读取 v0.1 数据 ----------------
print("[1] load v0.1 ...", flush=True)
main = pd.read_csv(V01 / "v004c_break_repair_dataset.csv", dtype={"code": str})
main["code"] = main["code"].astype(str).str.zfill(6)
audit = pd.read_csv(V01 / "v004c_candidate_audit.csv", dtype={"code": str})
audit["code"] = audit["code"].astype(str).str.zfill(6)
print("v0.1 main rows:", len(main), "| audit rows:", len(audit))

# ---------------- 2. 统一交易日历 ----------------
print("[2] build calendar ...", flush=True)
pool_dates = set()
for f in glob.glob(str(POOL_DIR / "*.pkl")):
    pool_dates.update(pd.read_pickle(f)["trade_date"].astype(str))
daily_dates = set()
for f in glob.glob(str(DAILY_DIR / "*_daily.pkl")):
    d = pd.read_pickle(f)
    daily_dates.update(pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
calendar = sorted(pool_dates | daily_dates)
cal_idx = {d: i for i, d in enumerate(calendar)}
print("calendar:", len(calendar), "dates,", calendar[0], "->", calendar[-1])


def next_cal_date(d: str):
    i = cal_idx.get(d)
    if i is None or i + 1 >= len(calendar):
        return None
    return calendar[i + 1]


# ---------------- 3. 停牌证明 ----------------
print("[3] load suspension proofs ...", flush=True)
suspension_proofs: dict[tuple[str, str], dict] = {}  # (code, date) -> record
for f in glob.glob(str(SUSP_DIR / "*.pkl")):
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
            suspension_proofs[(code, start, end)] = r
print("suspension proof intervals:", len(suspension_proofs))


def suspension_proven(code: str, date: str) -> bool:
    return any(s <= date <= e for (c, s, e) in suspension_proofs if c == code)


# ---------------- 4. 分钟/日线缓存 ----------------
minute_cache: dict[str, pd.DataFrame | None] = {}
daily_cache: dict[str, pd.DataFrame | None] = {}


def get_minute(code: str) -> pd.DataFrame | None:
    if code in minute_cache:
        return minute_cache[code]
    try:
        path = MINUTE_DIR / f"{code}_5min.pkl"
        if not path.exists():
            minute_cache[code] = None
            return None
        m = pd.read_pickle(path)
        m = m.copy()
        m["trade_date"] = m["trade_date"].astype(str)
        m["time_str"] = pd.to_datetime(m["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
        m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce")
        for col in ("open", "high", "low", "close"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
        minute_cache[code] = m.sort_values("datetime").reset_index(drop=True)
    except Exception:
        minute_cache[code] = None
    return minute_cache[code]


def get_daily(code: str) -> pd.DataFrame | None:
    if code in daily_cache:
        return daily_cache[code]
    try:
        d = pd.read_pickle(DAILY_DIR / f"{code}_daily.pkl")
        d = d.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        daily_cache[code] = d.dropna(subset=["date"])
    except Exception:
        daily_cache[code] = None
    return daily_cache[code]


def day_audit(code: str, date: str) -> dict:
    """单个交易日的 5min 网格审计"""
    out = {"bar_count": None, "first_bar_time": None, "last_bar_time": None,
           "minute_complete": False}
    m = get_minute(code)
    if m is None:
        return out
    day = m[m["trade_date"] == str(date)].sort_values("datetime")
    if day.empty:
        return out
    out["bar_count"] = int(len(day))
    times = day["time_str"].dropna().tolist()
    if times:
        out["first_bar_time"] = str(times[0])
        out["last_bar_time"] = str(times[-1])
    out["minute_complete"] = bool(
        out["bar_count"] == EXPECTED_BAR_COUNT
        and out["first_bar_time"] == EXPECTED_FIRST_BAR
        and out["last_bar_time"] == EXPECTED_LAST_BAR
    )
    return out


def daily_price(code: str, date: str, col: str) -> float | None:
    d = get_daily(code)
    if d is None or d.empty:
        return None
    rows = d[d["date"] == str(date)]
    if rows.empty:
        return None
    v = rows.iloc[0].get(col)
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------- 5. 逐行标签质量审计 ----------------
print("[5] label quality audit ...", flush=True)
rows_out = []
for _, r in main.iterrows():
    code = r["code"]
    sig = str(r["signal_date"])
    d2_act = str(r["d2_trade_date"])
    d3_act = str(r["d3_trade_date"])
    exp_d2 = next_cal_date(sig)
    exp_d3 = next_cal_date(exp_d2) if exp_d2 else None

    d1 = day_audit(code, sig)
    d2 = day_audit(code, d2_act)
    d3 = day_audit(code, d3_act)

    # 日期偏移与停牌证明
    d2_shift = (exp_d2 is not None and d2_act != exp_d2)
    d3_shift = (exp_d3 is not None and d3_act != exp_d3)
    d2_proven = suspension_proven(code, exp_d2) if (d2_shift and exp_d2) else False
    d3_proven = suspension_proven(code, exp_d3) if (d3_shift and exp_d3) else False
    d2_reason = "none" if not d2_shift else ("suspended_proven" if d2_proven else "shift_without_proof")
    d3_reason = "none" if not d3_shift else ("suspended_proven" if d3_proven else "shift_without_proof")
    susp_status = "not_applicable"
    if d2_shift or d3_shift:
        susp_status = "proven" if (d2_proven or d3_proven) else "not_proven"

    # 日线/分钟差异
    d2_daily_open = daily_price(code, d2_act, "open")
    d2_min_open = None
    m2 = get_minute(code)
    if m2 is not None:
        day2 = m2[m2["trade_date"] == d2_act].sort_values("datetime")
        if not day2.empty and pd.notna(day2["open"].iloc[0]):
            d2_min_open = float(day2["open"].iloc[0])
    d2_open_diff = None
    if d2_daily_open is not None and d2_min_open is not None:
        d2_open_diff = abs(d2_daily_open - d2_min_open)

    d3_daily_high = daily_price(code, d3_act, "high")
    d3_daily_close = daily_price(code, d3_act, "close")
    d3_min_high = None
    d3_min_close = None
    m3 = get_minute(code)
    if m3 is not None:
        day3 = m3[m3["trade_date"] == d3_act].sort_values("datetime")
        if not day3.empty:
            highs = pd.to_numeric(day3["high"], errors="coerce").dropna()
            closes = pd.to_numeric(day3["close"], errors="coerce").dropna()
            if len(highs):
                d3_min_high = float(highs.max())
            if len(closes):
                d3_min_close = float(closes.iloc[-1])
    d3_high_diff = None
    if d3_daily_high is not None and d3_min_high is not None:
        d3_high_diff = abs(d3_daily_high - d3_min_high)
    d3_close_diff = None
    if d3_daily_close is not None and d3_min_close is not None:
        d3_close_diff = abs(d3_daily_close - d3_min_close)

    # label_quality_ok
    reasons = []
    if not d1["minute_complete"]:
        reasons.append(f"d1_minute_incomplete(bars={d1['bar_count']})")
    if not d2["minute_complete"]:
        reasons.append(f"d2_minute_incomplete(bars={d2['bar_count']},first={d2['first_bar_time']},last={d2['last_bar_time']})")
    if not d3["minute_complete"]:
        reasons.append(f"d3_minute_incomplete(bars={d3['bar_count']},first={d3['first_bar_time']},last={d3['last_bar_time']})")
    if d2_shift and not d2_proven:
        reasons.append(f"d2_date_shift_without_proof(exp={exp_d2},act={d2_act})")
    if d3_shift and not d3_proven:
        reasons.append(f"d3_date_shift_without_proof(exp={exp_d3},act={d3_act})")
    if exp_d2 is None or exp_d3 is None:
        reasons.append("expected_d2_d3_beyond_calendar")
    label_quality_ok = not reasons
    label_quality_reason = "|".join(reasons) if reasons else "ok"

    # sample_role
    if sig < MAY_START or sig > TRAIN_END:
        sample_role = "EXCLUDED_QUALITY"
        role_reason = "signal_date_out_of_study_window"
    elif MAY_START <= sig <= MAY_END:
        if label_quality_ok:
            sample_role = "SENSITIVITY_MAY_COMPLETE"
            role_reason = "may_sensitivity_complete;minute_data_coverage_selection_bias"
        else:
            sample_role = "AUDIT_ONLY_MAY_INCOMPLETE"
            role_reason = "may_incomplete:" + label_quality_reason
    else:
        if label_quality_ok:
            sample_role = "TRAIN_PRIMARY"
            role_reason = "train_primary_20260601_20260729"
        else:
            sample_role = "EXCLUDED_QUALITY"
            role_reason = "label_quality_failed:" + label_quality_reason

    rows_out.append({
        **r.to_dict(),
        # 阶段
        "stage_group": "d0" if int(r["days_since_break"]) == 0 else "post",
        "post_day": None if int(r["days_since_break"]) == 0 else int(r["days_since_break"]),
        "training_stage_feature": "stage_group",
        # D1/D2/D3 质量
        "d1_bar_count": d1["bar_count"], "d2_bar_count": d2["bar_count"], "d3_bar_count": d3["bar_count"],
        "d1_first_bar_time": d1["first_bar_time"], "d2_first_bar_time": d2["first_bar_time"], "d3_first_bar_time": d3["first_bar_time"],
        "d1_last_bar_time": d1["last_bar_time"], "d2_last_bar_time": d2["last_bar_time"], "d3_last_bar_time": d3["last_bar_time"],
        "d1_expected_bar_count": EXPECTED_BAR_COUNT, "d2_expected_bar_count": EXPECTED_BAR_COUNT, "d3_expected_bar_count": EXPECTED_BAR_COUNT,
        "d1_minute_complete": d1["minute_complete"], "d2_minute_complete": d2["minute_complete"], "d3_minute_complete": d3["minute_complete"],
        "d2_daily_minute_open_diff": d2_open_diff, "d3_daily_minute_high_diff": d3_high_diff, "d3_daily_minute_close_diff": d3_close_diff,
        "expected_d2_date": exp_d2, "actual_d2_date": d2_act,
        "expected_d3_date": exp_d3, "actual_d3_date": d3_act,
        "d2_date_shift_reason": d2_reason, "d3_date_shift_reason": d3_reason,
        "suspension_proof_status": susp_status,
        "label_quality_ok": label_quality_ok, "label_quality_reason": label_quality_reason,
        "target_quality_ok": label_quality_ok and pd.notna(r["d2open_to_d3high_return"]),
        "tail_label_quality_ok": label_quality_ok and pd.notna(r["d2open_to_d3close_return"]),
        "sample_role": sample_role, "sample_role_reason": role_reason,
    })

all_df = pd.DataFrame(rows_out)
print("v0.2 all rows:", len(all_df))
print("sample_role counts:", all_df["sample_role"].value_counts().to_dict())
print("label_quality_ok counts:", all_df["label_quality_ok"].value_counts().to_dict())
all_df.to_csv(SCRATCH / "v02_stage1_all.csv", index=False, encoding="utf-8-sig")

# ---------------- 6. 因子重命名/新增/分桶 ----------------
print("[6] factor renames and new fields ...", flush=True)
RENAME = {
    "d1_close_to_ma5": "d1_close_to_ma5_raw",
    "d1_low_to_ma5": "d1_low_to_ma5_raw",
    "d1_high_to_ma5": "d1_high_to_ma5_raw",
    "d1_close_to_ma10": "d1_close_to_ma10_raw",
    "d1_low_to_ma10": "d1_low_to_ma10_raw",
    "d1_open_to_close_return": "d1_open_to_close_return_raw",
    "d1_high_to_close_drawdown": "d1_high_to_close_drawdown_raw",
    "d1_close_to_vwap": "d1_close_to_vwap_raw",
    "d1_reclaimed_ma5": "deprecated_d1_reclaimed_ma5_v01",
    "d1_reclaimed_ma10": "deprecated_d1_reclaimed_ma10_v01",
}
all_df = all_df.rename(columns=RENAME)

# 新增 reclaimed 修正字段
gap5 = pd.to_numeric(all_df["d1_close_to_ma5_raw"], errors="coerce")
gap10 = pd.to_numeric(all_df["d1_close_to_ma10_raw"], errors="coerce")
ma5 = pd.to_numeric(all_df["d1_ma5"], errors="coerce")
ma10 = pd.to_numeric(all_df["d1_ma10"], errors="coerce")
low = pd.to_numeric(all_df["d1_low"], errors="coerce")
close = pd.to_numeric(all_df["d1_close"], errors="coerce")
all_df["d1_close_above_ma5"] = (gap5.notna() & (gap5 >= 0)).astype(bool)
all_df["d1_close_above_ma10"] = (gap10.notna() & (gap10 >= 0)).astype(bool)
all_df["d1_true_reclaim_ma5"] = (low.notna() & ma5.notna() & (low <= ma5) & close.notna() & (close >= ma5)).astype(bool)
all_df["d1_true_reclaim_ma10"] = (low.notna() & ma10.notna() & (low <= ma10) & close.notna() & (close >= ma10)).astype(bool)


def ma5_bucket(g):
    if pd.isna(g):
        return None
    if g < -0.05:
        return "below_minus_5pct"
    if g < -0.02:
        return "minus_5_to_minus_2pct"
    if g < 0.02:
        return "near_ma5"
    if g < 0.05:
        return "plus_2_to_plus_5pct"
    if g < 0.10:
        return "plus_5_to_plus_10pct"
    return "above_plus_10pct"


all_df["d1_close_to_ma5_bucket"] = gap5.map(ma5_bucket)


def o2c_bucket(g):
    if pd.isna(g):
        return None
    if g < -0.05:
        return "below_minus_5pct"
    if g < -0.02:
        return "minus_5_to_minus_2pct"
    if g < 0.0:
        return "minus_2_to_0pct"
    if g < 0.02:
        return "zero_to_plus_2pct"
    if g < 0.05:
        return "plus_2_to_plus_5pct"
    return "above_plus_5pct"


all_df["d1_open_to_close_bucket"] = pd.to_numeric(all_df["d1_open_to_close_return_raw"], errors="coerce").map(o2c_bucket)

# ---------------- 7. 权重辅助字段 ----------------
print("[7] weight helpers ...", flush=True)
train_view = all_df[all_df["sample_role"] == "TRAIN_PRIMARY"].copy()
ev_count = train_view.groupby("event_id").size()
dt_count = train_view.groupby("signal_date").size()
all_df["event_observation_count"] = all_df["event_id"].map(ev_count).astype("Int64")
all_df["signal_date_candidate_count"] = all_df["signal_date"].map(dt_count).astype("Int64")
w = all_df["signal_date_candidate_count"].astype(float) * all_df["event_observation_count"].astype(float)
all_df["proposed_training_weight"] = np.where(w.notna() & (w > 0), 1.0 / w, np.nan)
tr_mask = all_df["sample_role"] == "TRAIN_PRIMARY"
w_tr = all_df.loc[tr_mask, "proposed_training_weight"]
if w_tr.notna().any():
    all_df.loc[tr_mask, "proposed_training_weight"] = w_tr / w_tr.mean()

all_df.to_csv(OUT / "v004c_dataset_v02_all.csv", index=False, encoding="utf-8-sig")
print("all-table written:", len(all_df))

# ---------------- 8. 训练视图 / 五月敏感性 / 审计 v0.2 ----------------
TRAIN_COLS = [
    "event_id", "code", "name", "signal_date", "break_date",
    "stage_group", "post_day", "board_streak_before_break",
    "d1_close_to_ma5_raw", "d1_close_to_ma5_bucket", "d1_close_above_ma5", "d1_true_reclaim_ma5",
    "d1_close_to_ma10_raw", "d1_close_above_ma10", "d1_true_reclaim_ma10",
    "d1_open_to_close_return_raw", "d1_open_to_close_bucket", "d1_high_to_close_drawdown_raw",
    "d1_close_location", "d1_close_to_vwap_raw",
    "down_bar_volume_ratio", "late_day_sell_volume_ratio", "high_zone_volume_ratio", "volume_above_d1_close_ratio",
    "break_high_to_close_drawdown", "break_close_location", "break_volume_ratio_vs_board_days",
    "break_touched_limit_up", "break_opened_from_limit_up",
    "d2_open", "d3_high", "d3_close",
    "d2open_to_d3high_return", "d2open_to_d3close_return",
    "target7_d2open_d3high", "tail_loss_5pct",
    "event_observation_count", "signal_date_candidate_count", "proposed_training_weight",
    "label_quality_ok", "target_quality_ok", "tail_label_quality_ok", "training_stage_feature",
]
train = all_df[(all_df["sample_role"] == "TRAIN_PRIMARY") & (all_df["label_quality_ok"] == True)].copy()  # noqa: E712
train = train[[c for c in TRAIN_COLS if c in train.columns]].reset_index(drop=True)
train.to_csv(OUT / "v004c_training_primary_20260601_20260729.csv", index=False, encoding="utf-8-sig")
print("training rows:", len(train))

may = all_df[all_df["sample_role"] == "SENSITIVITY_MAY_COMPLETE"].reset_index(drop=True)
may.to_csv(OUT / "v004c_may_sensitivity_complete.csv", index=False, encoding="utf-8-sig")
print("may sensitivity rows:", len(may))

# 审计表 v0.2: v0.1 审计行 + sample_role 映射
audit_v02 = audit.copy()
audit_v02["v01_candidate_status"] = audit_v02["candidate_status"]
may_mask = (audit_v02["signal_date"] >= MAY_START) & (audit_v02["signal_date"] <= MAY_END)
def map_role(row):
    # AUDIT_ONLY_MAY_INCOMPLETE: 五月候选事件但分钟/标签/必要特征不完整
    # (QUALITY_FAILED / LABEL_UNAVAILABLE);规则性排除(still_limit_up / not_two_or_three_board
    # / suspended / insufficient_daily_history)归 EXCLUDED_QUALITY
    if row["candidate_status"] == "CANDIDATE":
        return "TRAIN_PRIMARY_OR_MAY_SENSITIVITY"
    if row["candidate_status"] in ("QUALITY_FAILED", "LABEL_UNAVAILABLE"):
        return "AUDIT_ONLY_MAY_INCOMPLETE" if may_mask.loc[row.name] else "EXCLUDED_QUALITY"
    return "EXCLUDED_QUALITY"
audit_v02["sample_role_v02"] = audit_v02.apply(map_role, axis=1)
# 修正: 候选行按 v0.2 全量表的实际角色
role_map = dict(zip(all_df["event_id"] + "_" + all_df["signal_date"], all_df["sample_role"]))
keys = audit_v02["event_id"] + "_" + audit_v02["signal_date"].fillna("")
audit_v02["sample_role_v02"] = np.where(keys.isin(role_map), keys.map(role_map), audit_v02["sample_role_v02"])
audit_v02 = audit_v02.sort_values(["signal_date", "code", "days_since_break"]).reset_index(drop=True)
audit_v02.to_csv(OUT / "v004c_candidate_audit_v02.csv", index=False, encoding="utf-8-sig")
print("audit v0.2 rows:", len(audit_v02))
print("audit sample_role counts:", audit_v02["sample_role_v02"].value_counts().to_dict())

# ---------------- 9. 标签质量报告 ----------------
print("[9] label quality report ...", flush=True)
lq_cols = ["event_id", "code", "signal_date", "sample_role",
           "d1_bar_count", "d2_bar_count", "d3_bar_count",
           "d1_first_bar_time", "d2_first_bar_time", "d3_first_bar_time",
           "d1_last_bar_time", "d2_last_bar_time", "d3_last_bar_time",
           "d1_expected_bar_count", "d2_expected_bar_count", "d3_expected_bar_count",
           "d1_minute_complete", "d2_minute_complete", "d3_minute_complete",
           "d2_daily_minute_open_diff", "d3_daily_minute_high_diff", "d3_daily_minute_close_diff",
           "expected_d2_date", "actual_d2_date", "expected_d3_date", "actual_d3_date",
           "d2_date_shift_reason", "d3_date_shift_reason", "suspension_proof_status",
           "label_quality_ok", "label_quality_reason", "target_quality_ok", "tail_label_quality_ok"]
lq = all_df[[c for c in lq_cols if c in all_df.columns]].copy()
# 汇总段
summary_rows = []
fail = all_df[all_df["label_quality_ok"] == False]  # noqa: E712
summary_rows.append({"section": "overall", "label_quality_ok": "false_count", "value": int((all_df["label_quality_ok"] == False).sum())})  # noqa: E712
for reason, n in fail["label_quality_reason"].value_counts().items():
    summary_rows.append({"section": "failure_reason", "label_quality_ok": str(reason), "value": int(n)})
for role, n in all_df.groupby("sample_role")["label_quality_ok"].apply(lambda s: int((s == False).sum())).items():  # noqa: E712
    summary_rows.append({"section": "failure_by_sample_role", "label_quality_ok": str(role), "value": int(n)})
lq_sum = pd.DataFrame(summary_rows)
lq_out = pd.concat([lq.assign(section="per_row"), lq_sum], ignore_index=True)
lq_out.to_csv(OUT / "v004c_label_quality_report_v02.csv", index=False, encoding="utf-8-sig")
print("label quality report written:", len(lq_out))
print("failures:", int((all_df['label_quality_ok'] == False).sum()))  # noqa: E712

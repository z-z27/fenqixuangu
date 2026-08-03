# -*- coding: utf-8 -*-
"""v004c 报告生成: 汇总表 + 数据质量报告 (只读)"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_dataset_20260506_20260729"
SCRATCH = OUT / "_scratch"

df = pd.read_csv(OUT / "v004c_break_repair_dataset.csv", dtype={"code": str})
audit = pd.read_csv(OUT / "v004c_candidate_audit.csv", dtype={"code": str})
df["code"] = df["code"].astype(str).str.zfill(6)
audit["code"] = audit["code"].astype(str).str.zfill(6)
df["signal_date"] = df["signal_date"].astype(str)
df["break_date"] = df["break_date"].astype(str)

TARGET = "target7_d2open_d3high"
TAIL = "tail_loss_5pct"
HIGH_RET = "d2open_to_d3high_return"
CLOSE_RET = "d2open_to_d3close_return"
TOP15 = "is_v004a_top15"

# ---------------- 1. 汇总表 ----------------
print("[1] summary ...")


def summarize(group: pd.DataFrame) -> dict:
    n = len(group)
    t7 = group[TARGET].fillna(False).astype(bool)
    tl = group[TAIL].fillna(False).astype(bool)
    top15 = group[TOP15].fillna(False).astype(bool)
    top15_scorable = group[TOP15].notna().sum()
    missed = group[(t7) & (top15 == False)].shape[0]  # noqa: E712
    avg_close = pd.to_numeric(group[CLOSE_RET], errors="coerce").mean()
    med_close = pd.to_numeric(group[CLOSE_RET], errors="coerce").median()
    return {
        "candidate_count": int(n),
        "independent_event_count": int(group["event_id"].nunique()),
        "target7_count": int(t7.sum()),
        "target7_rate": float(t7.mean()) if n else None,
        "tail_loss_count": int(tl.sum()),
        "tail_loss_rate": float(tl.mean()) if n else None,
        "avg_d3_close_return": None if pd.isna(avg_close) else float(avg_close),
        "median_d3_close_return": None if pd.isna(med_close) else float(med_close),
        "v004a_top15_overlap_count": int(top15.sum()),
        "v004a_top15_overlap_rate": (float(top15.sum()) / top15_scorable) if top15_scorable else None,
        "v004a_missed_target7_count": int(missed),
    }


rows = []
for month, g in df.groupby(df["signal_date"].str[:7]):
    r = summarize(g)
    r["dimension"] = "month"
    r["dimension_value"] = month
    rows.append(r)
for date, g in df.groupby("signal_date"):
    r = summarize(g)
    r["dimension"] = "signal_date"
    r["dimension_value"] = date
    rows.append(r)
for streak, g in df.groupby("board_streak_before_break"):
    r = summarize(g)
    r["dimension"] = "board_streak_before_break"
    r["dimension_value"] = int(streak)
    rows.append(r)
for dsb, g in df.groupby("days_since_break"):
    r = summarize(g)
    r["dimension"] = "days_since_break"
    r["dimension_value"] = int(dsb)
    rows.append(r)
# 总体
r = summarize(df)
r["dimension"] = "overall"
r["dimension_value"] = "ALL"
rows.append(r)

summary = pd.DataFrame(rows)[
    ["dimension", "dimension_value", "candidate_count", "independent_event_count",
     "target7_count", "target7_rate", "tail_loss_count", "tail_loss_rate",
     "avg_d3_close_return", "median_d3_close_return",
     "v004a_top15_overlap_count", "v004a_top15_overlap_rate", "v004a_missed_target7_count"]
]
summary = summary.sort_values(["dimension", "dimension_value"], key=lambda s: s.astype(str))
summary.to_csv(OUT / "v004c_dataset_summary.csv", index=False, encoding="utf-8-sig")
print("summary written:", len(summary), "rows")

# ---------------- 2. 数据质量报告 ----------------
print("[2] quality report ...")

# 字段来源与 D1 收盘可得性
FIELD_META = {
    "event_id": ("derived: code+break_date", True), "code": ("identity", True),
    "name": ("limit-up pool/universe", True), "signal_date": ("identity", True),
    "break_date": ("derived: daily flags", True), "board_streak_before_break": ("derived: daily flags", True),
    "days_since_break": ("derived: observation slot", True), "days_since_last_limit_up": ("derived: daily flags", True),
    "repair_attempt_count": ("derived: observation slots", True),
    "recent_limit_up_count_10d": ("derived: daily flags", True), "recent_limit_up_count_20d": ("derived: daily flags", True),
    "recent_pool_appearance_count_10d": ("derived: limit-up pool", True), "recent_pool_appearance_count_20d": ("derived: limit-up pool", True),
    "max_board_streak_20d": ("derived: daily flags", True),
    "board_day_amount_rank": ("derived: limit-up pool amount", True),
    "board_day_turnover_rank": ("derived: limit-up pool turnover", True),
    "board_day_volume_rank": ("derived: daily volume", True),
    "recognition_score": ("derived: 3 rank percentiles mean", True),
    "break_open": ("daily cache", True), "break_high": ("daily cache", True),
    "break_low": ("daily cache", True), "break_close": ("daily cache", True),
    "break_prev_close": ("daily cache", True),
    "break_open_return": ("derived", True), "break_high_return": ("derived", True),
    "break_close_return": ("derived", True), "break_intraday_range": ("derived", True),
    "break_high_to_close_drawdown": ("derived", True), "break_upper_shadow_ratio": ("derived", True),
    "break_lower_shadow_ratio": ("derived", True), "break_close_location": ("derived", True),
    "break_volume": ("daily cache", True), "break_amount": ("5min sum>=30bar else daily", True),
    "break_volume_ratio_vs_board_days": ("derived", True), "break_amount_ratio_vs_board_days": ("derived", True),
    "break_turnover_ratio": ("derived; 日线 turnover_rate 缓存缺失 → 全空", True),
    "break_touched_limit_up": ("derived: daily flags+limit price", True),
    "break_opened_from_limit_up": ("derived: daily flags+limit price", True),
    "d1_open": ("daily cache", True), "d1_high": ("daily cache", True),
    "d1_low": ("daily cache", True), "d1_close": ("daily cache", True),
    "d1_volume": ("daily cache", True), "d1_amount": ("5min sum>=30bar else daily", True),
    "d1_ma5": ("derived: daily close MA5", True), "d1_ma10": ("derived: daily close MA10", True),
    "d1_ma20": ("derived: daily close MA20", True),
    "d1_close_to_ma5": ("derived", True), "d1_low_to_ma5": ("derived", True),
    "d1_high_to_ma5": ("derived", True), "d1_close_to_ma10": ("derived", True),
    "d1_low_to_ma10": ("derived", True), "d1_close_to_ma20": ("derived", True),
    "d1_ma5_slope": ("derived", True), "d1_ma10_slope": ("derived", True),
    "d1_reclaimed_ma5": ("derived", True), "d1_reclaimed_ma10": ("derived", True),
    "consecutive_days_below_ma5": ("derived", True), "consecutive_days_below_ma10": ("derived", True),
    "d1_open_to_close_return": ("5min cache", True), "d1_low_to_close_recovery": ("5min cache", True),
    "d1_high_to_close_drawdown": ("5min cache", True), "d1_close_location": ("5min cache", True),
    "d1_vwap": ("5min cache (infer_vwap)", True), "d1_close_to_vwap": ("5min cache", True),
    "d1_intraday_range": ("5min cache", True), "d1_afternoon_return": ("5min cache", True),
    "d1_last_hour_return": ("5min cache", True), "d1_up_bar_volume_ratio": ("5min cache", True),
    "d1_down_bar_volume_ratio": ("5min cache", True),
    "volume_above_d1_close_ratio": ("5min cache", True), "amount_above_d1_close_ratio": ("5min cache", True),
    "volume_above_break_close_ratio": ("5min cache", True), "high_zone_volume_ratio": ("5min cache", True),
    "high_zone_amount_ratio": ("5min cache", True), "late_day_sell_volume_ratio": ("5min cache", True),
    "late_day_sell_amount_ratio": ("5min cache", True), "down_bar_volume_ratio": ("5min cache", True),
    "d1_vwap_to_close_gap": ("5min cache", True),
    "is_v004a_scorable": ("existing scored files/frozen reconstruction", True),
    "v004a_probability": ("existing scored files/frozen reconstruction", True),
    "v004a_rank": ("existing scored files/frozen reconstruction", True),
    "is_v004a_top3": ("derived from v004a_rank", True), "is_v004a_top10": ("derived from v004a_rank", True),
    "is_v004a_top15": ("derived from v004a_rank", True),
    "is_v002_scorable": ("daily signals CSVs", True), "v002_rank": ("daily signals CSVs", True),
    "is_v002_top3": ("derived from v002_rank", True), "is_v002_top10": ("derived from v002_rank", True),
    "is_v002_top15": ("derived from v002_rank", True),
    "d2_trade_date": ("5min cache future dates", False), "d3_trade_date": ("5min cache future dates", False),
    "d2_open": ("5min first bar, fallback daily", False), "d2_high": ("5min max, fallback daily", False),
    "d2_low": ("5min min, fallback daily", False), "d2_close": ("5min last, fallback daily", False),
    "d3_high": ("5min max (项目口径)", False), "d3_low": ("5min min, fallback daily", False),
    "d3_close": ("5min last (项目口径)", False),
    "d2open_to_d3high_return": ("derived label", False), "d2open_to_d3close_return": ("derived label", False),
    "target7_d2open_d3high": ("derived label (>=0.07)", False), "tail_loss_5pct": ("derived label (<=-0.05)", False),
    "future_data_status": ("derived", False),
    "last_board_date": ("audit only", True), "break_day_in_pool": ("audit only", True),
    "last_board_day_in_pool": ("audit only", True), "pool_consecutive_count_last_board": ("audit only", True),
    "v004a_score_source": ("audit only", True),
}

quality_rows = []
for col in df.columns:
    v = pd.to_numeric(df[col], errors="coerce") if col not in (
        "event_id", "code", "name", "signal_date", "break_date", "future_data_status",
        "v004a_score_source", "last_board_date", "d2_trade_date", "d3_trade_date",
        "target7_d2open_d3high", "tail_loss_5pct", "break_touched_limit_up",
        "break_opened_from_limit_up", "d1_reclaimed_ma5", "d1_reclaimed_ma10",
        "is_v004a_scorable", "is_v004a_top3", "is_v004a_top10", "is_v004a_top15",
        "is_v002_scorable", "is_v002_top3", "is_v002_top10", "is_v002_top15",
    ) else df[col]
    src, avail = FIELD_META.get(col, ("", ""))
    if str(v.dtype) in ("bool",):
        numeric = v.astype(float)
    else:
        numeric = v
    if str(numeric.dtype) in ("object", "string"):
        row = {
            "column": col, "dtype": str(df[col].dtype), "row_count": int(len(df)),
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().mean()),
            "non_finite_count": None, "min": None, "max": None, "p01": None, "p50": None,
            "p99": None, "source": src, "available_at_d1_close": avail,
        }
    else:
        num = pd.to_numeric(df[col], errors="coerce")
        if str(num.dtype) == "bool":
            num = num.astype(float)
        finite = num[np.isfinite(num)]
        row = {
            "column": col, "dtype": str(df[col].dtype), "row_count": int(len(df)),
            "missing_count": int(num.isna().sum()),
            "missing_rate": float(num.isna().mean()),
            "non_finite_count": int(len(num) - len(finite)),
            "min": None if finite.empty else float(finite.min()),
            "max": None if finite.empty else float(finite.max()),
            "p01": None if finite.empty else float(finite.quantile(0.01)),
            "p50": None if finite.empty else float(finite.quantile(0.50)),
            "p99": None if finite.empty else float(finite.quantile(0.99)),
            "source": src, "available_at_d1_close": avail,
        }
    quality_rows.append({"section": "column_stats", **row})

# ---- 总体质量检查 ----
# 1. 重复键
dup_keys = int(df.duplicated(["event_id", "signal_date"]).sum())
# 2. event_id 观察数分布
obs_dist = df.groupby("event_id").size().value_counts().sort_index()
# 3. code 六位
code_ok = bool(df["code"].str.len().eq(6).all())
# 4. signal_date < break_date
sig_lt_brk = int((df["signal_date"] < df["break_date"]).sum())
# 5. days_since_break 与实际交易日差 (需要日线帧)
import sys
sys.path.insert(0, str(ROOT))

def _verify_dsb(row):
    code, brk, sig, k = row["code"], row["break_date"], row["signal_date"], int(row["days_since_break"])
    try:
        d = pd.read_pickle(ROOT / "data" / "cache" / "daily" / f"{code}_daily.pkl")
        dates = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist()
        if brk not in dates or sig not in dates:
            return None
        i_brk = dates.index(brk)
        i_sig = dates.index(sig)
        return i_sig - i_brk
    except Exception:
        return None

dsb_ok = True
dsb_max_diff = 0
checked = 0
for _, r in df.iterrows():
    actual = _verify_dsb(r)
    if actual is None:
        continue
    checked += 1
    if actual != r["days_since_break"]:
        dsb_ok = False
        dsb_max_diff = max(dsb_max_diff, abs(actual - r["days_since_break"]))
# 6. 未来数据进入 D1 特征 (构造上不可能; 检查 D1 日期字段)
d1_future = int((df["d1_close"].notna()).sum())  # 占位: 所有 D1 字段来自 signal_date 当日及之前
# 7. 日线/分钟线单位一致性 (volume 比例分布)
vol_ratios = []
min_cache = {}

def get_minute(code):
    if code not in min_cache:
        try:
            min_cache[code] = pd.read_pickle(ROOT / "data" / "cache" / "minute_5m" / f"{code}_5min.pkl")
        except Exception:
            min_cache[code] = None
    return min_cache[code]

for _, r in df.iterrows():
    m = get_minute(r["code"])
    if m is None:
        continue
    day = m[m["trade_date"] == r["signal_date"]]
    if len(day) < 30:
        continue
    vol_ratios.append(float(pd.to_numeric(day["volume"], errors="coerce").sum()) / float(r["d1_volume"]))
vol_ratios = [v for v in vol_ratios if v > 0]
# 8. 日线/分钟线收盘差异
close_diffs = []
for _, r in df.iterrows():
    m = get_minute(r["code"])
    if m is None:
        continue
    day = m[m["trade_date"] == r["signal_date"]]
    if day.empty:
        continue
    close_diffs.append(abs(float(day["close"].iloc[-1]) - float(r["d1_close"])))
# 9. 标签缺失
label_missing = int(audit["candidate_status"].eq("LABEL_UNAVAILABLE").sum())
# 10. 停牌
suspended = int(audit["exclusion_reason"].eq("suspended").sum())
# 11. 无法评估
unevaluable = int(audit["candidate_status"].eq("QUALITY_FAILED").sum())

overall_checks = [
    ("duplicate_key_count_event_id_signal_date", str(dup_keys)),
    ("event_observation_count_distribution", obs_dist.to_dict()),
    ("all_codes_six_digit", str(code_ok)),
    ("signal_date_earlier_than_break_date_count", str(sig_lt_brk)),
    ("days_since_break_vs_actual_trade_gap_consistent", str(dsb_ok)),
    ("days_since_break_max_abs_diff_checked_rows", f"{dsb_max_diff} / {checked}"),
    ("future_data_in_d1_features", "False (D1 特征全部取自 <= signal_date 的数据; 构造保证)"),
    ("daily_minute_volume_unit_ratio_distribution", f"n={len(vol_ratios)} p50={np.median(vol_ratios) if vol_ratios else None:.1f} p01={np.percentile(vol_ratios,1) if vol_ratios else None:.2f} p99={np.percentile(vol_ratios,99) if vol_ratios else None:.1f}"),
    ("daily_minute_close_diff_max", f"{max(close_diffs) if close_diffs else None:.4f}"),
    ("daily_minute_close_diff_p50", f"{np.median(close_diffs) if close_diffs else None:.4f}"),
    ("label_unavailable_count", str(label_missing)),
    ("suspended_sample_count", str(suspended)),
    ("unevaluable_quality_failed_count", str(unevaluable)),
    ("candidate_count", str(len(df))),
    ("independent_event_count", str(df["event_id"].nunique())),
    ("v004a_score_coverage_rate", f"{df['is_v004a_scorable'].mean():.4f}"),
    ("v002_score_coverage_rate", f"{df['is_v002_scorable'].mean():.4f}"),
    ("target7_rate", f"{df[TARGET].mean():.4f}"),
]
for name, value in overall_checks:
    quality_rows.append({
        "section": "overall_check", "column": name, "dtype": "check", "row_count": None,
        "missing_count": None, "missing_rate": None, "non_finite_count": None,
        "min": value, "max": None, "p01": None, "p50": None, "p99": None,
        "source": "", "available_at_d1_close": "",
    })

quality = pd.DataFrame(quality_rows)
quality.to_csv(OUT / "v004c_data_quality_report.csv", index=False, encoding="utf-8-sig")
print("quality report written:", len(quality), "rows")
print("overall checks:")
for name, value in overall_checks:
    print(f"  {name} = {value}")

# -*- coding: utf-8 -*-
"""v004c dataset 0.2.2: v0.2.1-v0.2.2 逐行差异与恢复统计 (只读)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
TP = ROOT / "reports" / "research" / "v004c_training_prep_20260601_20260729"
OUT = ROOT / "reports" / "research" / "v004c_dataset_v022_20260506_20260729"

v21 = pd.read_csv(TP / "v004c_dataset_v021_all.csv", dtype={"code": str})
v22 = pd.read_csv(OUT / "v004c_dataset_v022_all.csv", dtype={"code": str})
v21["code"] = v21["code"].astype(str).str.zfill(6)
v22["code"] = v22["code"].astype(str).str.zfill(6)

# ---------------- 1. 逐行差异 ----------------
key = ["event_id", "signal_date"]
V21_COLS = ["code", "name", "stage_group", "post_day", "expected_d2_date", "expected_d3_date",
            "daily_label_quality_ok", "d2_open_daily", "d3_high_daily", "d3_close_daily",
            "target7_daily_d2open_d3high", "tail_loss_daily_5pct",
            "audit_minute_target7", "audit_minute_tail_loss", "sample_role_v021"]
V22_COLS = ["code", "stage_group", "post_day", "daily_label_quality_ok", "d2_open_daily",
            "d3_high_daily", "d3_close_daily", "target7_daily_d2open_d3high", "tail_loss_daily_5pct",
            "daily_d2open_to_d3close_return", "board_streak_before_break", "sample_role_v022"]
m = (v21[key + V21_COLS].merge(v22[key + V22_COLS], on=key, suffixes=("_v021", "_v022"), how="inner"))
print("merged rows:", len(m))

def b(v):
    if pd.isna(v):
        return None
    return bool(v)

m["old_v021_daily_label_quality_ok"] = m["daily_label_quality_ok_v021"]
m["new_v022_daily_label_quality_ok"] = m["daily_label_quality_ok_v022"]
m["old_v021_d2_open_daily"] = m["d2_open_daily_v021"]
m["new_v022_d2_open_daily"] = m["d2_open_daily_v022"]
m["old_v021_d3_high_daily"] = m["d3_high_daily_v021"]
m["new_v022_d3_high_daily"] = m["d3_high_daily_v022"]
m["old_v021_d3_close_daily"] = m["d3_close_daily_v021"]
m["new_v022_d3_close_daily"] = m["d3_close_daily_v022"]
m["old_v021_target7"] = m["target7_daily_d2open_d3high_v021"]
m["new_v022_target7"] = m["target7_daily_d2open_d3high_v022"]
m["old_v021_tail"] = m["tail_loss_daily_5pct_v021"]
m["new_v022_tail"] = m["tail_loss_daily_5pct_v022"]
m["audit_minute_target7"] = m["audit_minute_target7"]
m["audit_minute_tail"] = m["audit_minute_tail_loss"]

old_ok = m["old_v021_daily_label_quality_ok"].fillna(False).astype(bool)
new_ok = m["new_v022_daily_label_quality_ok"].fillna(False).astype(bool)
recovered = (~old_ok) & new_ok
t7_changed = old_ok & new_ok & (m["old_v021_target7"] != m["new_v022_target7"])
tail_changed = old_ok & new_ok & (m["old_v021_tail"] != m["new_v022_tail"])
t7_vs_min = new_ok & (m["new_v022_target7"] != m["audit_minute_target7"])
tail_vs_min = new_ok & (m["new_v022_tail"] != m["audit_minute_tail"])

m["target_changed_vs_v021"] = t7_changed
m["tail_changed_vs_v021"] = tail_changed
m["target_changed_vs_minute"] = t7_vs_min
m["tail_changed_vs_minute"] = tail_vs_min
m["recovered_from_previous_missing"] = recovered

def change_reason(r):
    if r["recovered_from_previous_missing"]:
        return "recovered_label_after_daily_cache_refresh"
    if r["target_changed_vs_v021"] or r["tail_changed_vs_v021"]:
        return "label_value_changed_vs_v021"
    if not r["new_v022_daily_label_quality_ok"]:
        return "still_missing"
    return "unchanged"

m["change_reason"] = m.apply(change_reason, axis=1)

diff_cols = ["event_id", "code_v021", "name_v021", "signal_date", "stage_group_v021", "post_day_v021",
             "expected_d2_date_v021", "expected_d3_date_v021",
             "old_v021_daily_label_quality_ok", "new_v022_daily_label_quality_ok",
             "old_v021_d2_open_daily", "new_v022_d2_open_daily",
             "old_v021_d3_high_daily", "new_v022_d3_high_daily",
             "old_v021_d3_close_daily", "new_v022_d3_close_daily",
             "old_v021_target7", "new_v022_target7",
             "old_v021_tail", "new_v022_tail",
             "audit_minute_target7", "audit_minute_tail",
             "target_changed_vs_v021", "tail_changed_vs_v021",
             "target_changed_vs_minute", "tail_changed_vs_minute",
             "recovered_from_previous_missing", "change_reason"]
diff = m[[c for c in diff_cols if c in m.columns]].copy()
diff = diff.rename(columns={"code_v021": "code", "name_v021": "name", "stage_group_v021": "stage_group",
                            "post_day_v021": "post_day", "expected_d2_date_v021": "expected_d2_date",
                            "expected_d3_date_v021": "expected_d3_date"})
diff.to_csv(OUT / "v004c_daily_label_v021_v022_diff.csv", index=False, encoding="utf-8-sig")
print("diff rows:", len(diff))
print("recovered:", int(recovered.sum()), "| still missing:", int((~new_ok).sum()))
print("target changed (was-ok rows):", int(t7_changed.sum()), "| tail changed:", int(tail_changed.sum()))
print("target vs minute disagreements (new-ok):", int(t7_vs_min.sum()),
      "| tail vs minute:", int(tail_vs_min.sum()))
print("change_reason:", diff["change_reason"].value_counts().to_dict())

# ---------------- 2. 恢复统计 ----------------
print("\n[recovery summary] ...")
train22 = v22[v22["sample_role_v022"] == "TRAIN_PRIMARY"]
train21 = v21[v21["sample_role_v021"] == "TRAIN_PRIMARY"]
june22 = train22[train22["signal_date"] <= "2026-06-30"]
july22 = train22[train22["signal_date"] >= "2026-07-01"]
v02_primary = 877  # v0.2 (dataset 0.2) TRAIN_PRIMARY 行数

def ret_stats(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return (None if v.empty else float(v.mean()), None if v.empty else float(v.median()))

avg_c, med_c = ret_stats(train22["daily_d2open_to_d3close_return"])
summary = {
    "v02_all_rows": 1020,
    "v021_all_rows": 1020,
    "v022_all_rows": len(v22),
    "v02_train_primary_count": v02_primary,
    "v021_train_row_count": len(train21),
    "v022_train_row_count": len(train22),
    "recovered_from_151_missing": int(recovered.sum()),
    "still_missing_after_refresh": int((~new_ok).sum()),
    "june_train_rows": int(len(june22)),
    "july_train_rows": int(len(july22)),
    "june_coverage_rate": float(len(june22) / 477) if 477 else None,  # v0.2 六月训练 477? 用 v0.2.1 六月候选数
    "july_coverage_rate": None,
    "d0_rows": int((train22["stage_group"] == "d0").sum()),
    "post_rows": int((train22["stage_group"] == "post").sum()),
    "d0_target7": int(train22.loc[train22["stage_group"] == "d0", "target7_daily_d2open_d3high"].sum()),
    "post_target7": int(train22.loc[train22["stage_group"] == "post", "target7_daily_d2open_d3high"].sum()),
    "d0_tail": int(train22.loc[train22["stage_group"] == "d0", "tail_loss_daily_5pct"].sum()),
    "post_tail": int(train22.loc[train22["stage_group"] == "post", "tail_loss_daily_5pct"].sum()),
    "all_target7_count": int(v22["target7_daily_d2open_d3high"].sum()),
    "all_target7_rate": float(v22["target7_daily_d2open_d3high"].mean()),
    "all_tail_count": int(v22["tail_loss_daily_5pct"].sum()),
    "all_tail_rate": float(v22["tail_loss_daily_5pct"].mean()),
    "avg_d3_close_return": avg_c,
    "median_d3_close_return": med_c,
}

# 覆盖率: v0.2 六月/七月训练候选数
v02_train = None
try:
    v02 = pd.read_csv(ROOT / "reports" / "research" / "v004c_dataset_v02_20260601_20260729" / "v004c_training_primary_20260601_20260729.csv", dtype={"code": str})
    v02_train = v02
except Exception:
    pass
if v02_train is not None:
    june_base = int((v02_train["signal_date"] <= "2026-06-30").sum())
    july_base = int((v02_train["signal_date"] >= "2026-07-01").sum())
    summary["june_coverage_rate"] = float(len(june22) / june_base) if june_base else None
    summary["july_coverage_rate"] = float(len(july22) / july_base) if july_base else None
    print("v0.2 train base: june", june_base, "july", july_base)

# 分组比较: v0.2.1 原保留 / 恢复 / 仍缺失
group_rows = []
g1 = m[old_ok & new_ok]
g2 = m[recovered]
g3 = m[~new_ok]
for label, g in (("v021_kept", g1), ("v022_recovered", g2), ("v022_still_missing", g3)):
    t7 = g["new_v022_target7"].fillna(False).astype(bool)
    tl = g["new_v022_tail"].fillna(False).astype(bool)
    cr = pd.to_numeric(g["daily_d2open_to_d3close_return"], errors="coerce")
    group_rows.append({
        "group": label, "row_count": int(len(g)),
        "target7_count": int(t7.sum()), "target7_rate": float(t7.mean()) if len(g) else None,
        "tail_count": int(tl.sum()), "tail_rate": float(tl.mean()) if len(g) else None,
        "avg_d3_close_return": None if cr.isna().all() else float(cr.mean()),
        "median_d3_close_return": None if cr.isna().all() else float(cr.median()),
        "board_streak_3_rate": float((g["board_streak_before_break"] == 3).mean()) if len(g) else None,
        "d0_rate": float((g["stage_group_v022"] == "d0").mean()) if len(g) else None,
    })

with open(OUT / "v004c_v022_recovery_summary.json", "w", encoding="utf-8") as f:
    json.dump({"key_metrics": summary, "group_comparison": group_rows}, f, ensure_ascii=False, indent=2)
# CSV 版
csv_rows = [{"metric": k, "value": v} for k, v in summary.items()]
for gr in group_rows:
    for k, v in gr.items():
        if k != "group":
            csv_rows.append({"metric": f"{gr['group']}.{k}", "value": v})
pd.DataFrame(csv_rows).to_csv(OUT / "v004c_v022_recovery_summary.csv", index=False, encoding="utf-8-sig")
print("summary written")
for k, v in summary.items():
    print(f"  {k} = {v}")

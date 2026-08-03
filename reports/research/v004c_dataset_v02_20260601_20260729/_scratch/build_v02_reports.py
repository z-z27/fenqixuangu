# -*- coding: utf-8 -*-
"""v004c dataset 0.2 报告: 因子质量 / 冗余 / v01-v02差异 / 汇总 (只读)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_dataset_v02_20260601_20260729"
SCRATCH = OUT / "_scratch"

all_df = pd.read_csv(OUT / "v004c_dataset_v02_all.csv", dtype={"code": str})
all_df["code"] = all_df["code"].astype(str).str.zfill(6)
train_df = pd.read_csv(OUT / "v004c_training_primary_20260601_20260729.csv", dtype={"code": str})
may_df = pd.read_csv(OUT / "v004c_may_sensitivity_complete.csv", dtype={"code": str})

TRAIN_START, TRAIN_END = "2026-06-01", "2026-07-29"

# ---------------- 字段角色与公式元数据 ----------------
IDENTITY = ["event_id", "code", "name", "signal_date", "break_date", "board_streak_before_break",
            "stage_group", "post_day", "training_stage_feature", "sample_role", "sample_role_reason",
            "future_data_status"]
TRAINING = ["d1_close_to_ma5_raw", "d1_close_to_ma5_bucket", "d1_close_above_ma5", "d1_true_reclaim_ma5",
            "d1_close_to_ma10_raw", "d1_close_above_ma10", "d1_true_reclaim_ma10",
            "d1_open_to_close_return_raw", "d1_open_to_close_bucket", "d1_high_to_close_drawdown_raw",
            "d1_close_location", "d1_close_to_vwap_raw",
            "down_bar_volume_ratio", "late_day_sell_volume_ratio", "high_zone_volume_ratio",
            "volume_above_d1_close_ratio",
            "break_high_to_close_drawdown", "break_close_location", "break_volume_ratio_vs_board_days",
            "break_touched_limit_up", "break_opened_from_limit_up"]
AUDIT_ONLY = ["recognition_score", "board_day_amount_rank", "board_day_turnover_rank", "board_day_volume_rank",
              "v004a_probability", "v004a_rank", "v002_rank",
              "is_v004a_scorable", "is_v004a_top3", "is_v004a_top10", "is_v004a_top15",
              "is_v002_scorable", "is_v002_top3", "is_v002_top10", "is_v002_top15", "v004a_score_source",
              "recent_limit_up_count_10d", "recent_limit_up_count_20d",
              "recent_pool_appearance_count_10d", "recent_pool_appearance_count_20d", "max_board_streak_20d",
              "last_board_date", "break_day_in_pool", "last_board_day_in_pool", "pool_consecutive_count_last_board"]
LABEL = ["d2_trade_date", "d3_trade_date", "d2_open", "d2_high", "d2_low", "d2_close",
         "d3_high", "d3_low", "d3_close", "d2open_to_d3high_return", "d2open_to_d3close_return",
         "target7_d2open_d3high", "tail_loss_5pct"]
LABEL_QUALITY = ["d1_bar_count", "d2_bar_count", "d3_bar_count",
                 "d1_first_bar_time", "d2_first_bar_time", "d3_first_bar_time",
                 "d1_last_bar_time", "d2_last_bar_time", "d3_last_bar_time",
                 "d1_expected_bar_count", "d2_expected_bar_count", "d3_expected_bar_count",
                 "d1_minute_complete", "d2_minute_complete", "d3_minute_complete",
                 "d2_daily_minute_open_diff", "d3_daily_minute_high_diff", "d3_daily_minute_close_diff",
                 "expected_d2_date", "actual_d2_date", "expected_d3_date", "actual_d3_date",
                 "d2_date_shift_reason", "d3_date_shift_reason", "suspension_proof_status",
                 "label_quality_ok", "label_quality_reason", "target_quality_ok", "tail_label_quality_ok"]
WEIGHT_HELPER = ["event_observation_count", "signal_date_candidate_count", "proposed_training_weight"]
DEPRECATED_REDUNDANT = ["days_since_break", "days_since_last_limit_up", "repair_attempt_count",
                        "amount_above_d1_close_ratio", "high_zone_amount_ratio",
                        "late_day_sell_amount_ratio", "d1_down_bar_volume_ratio",
                        "break_amount_ratio_vs_board_days", "d1_vwap_to_close_gap",
                        "deprecated_d1_reclaimed_ma5_v01", "deprecated_d1_reclaimed_ma10_v01"]
# 其余 = research_hold (保留在研究全量表, 未进入 v0.2 训练视图)
RESEARCH_HOLD = [c for c in all_df.columns if c not in (
    IDENTITY + TRAINING + AUDIT_ONLY + LABEL + LABEL_QUALITY + WEIGHT_HELPER + DEPRECATED_REDUNDANT)]

ROLE = {}
for c in IDENTITY:
    ROLE[c] = "identity"
for c in TRAINING:
    ROLE[c] = "training"
for c in AUDIT_ONLY:
    ROLE[c] = "audit_only"
for c in LABEL:
    ROLE[c] = "label"
for c in LABEL_QUALITY:
    ROLE[c] = "label_quality"
for c in WEIGHT_HELPER:
    ROLE[c] = "weight_helper"
for c in DEPRECATED_REDUNDANT:
    ROLE[c] = "deprecated_redundant"
for c in RESEARCH_HOLD:
    ROLE[c] = "research_hold"

FORMULAS = {
    "d1_close_to_ma5_raw": "d1_close / d1_ma5 - 1", "d1_low_to_ma5_raw": "d1_low / d1_ma5 - 1",
    "d1_high_to_ma5_raw": "d1_high / d1_ma5 - 1", "d1_close_to_ma10_raw": "d1_close / d1_ma10 - 1",
    "d1_low_to_ma10_raw": "d1_low / d1_ma10 - 1", "d1_close_to_ma20": "d1_close / d1_ma20 - 1",
    "d1_close_above_ma5": "d1_close >= d1_ma5", "d1_close_above_ma10": "d1_close >= d1_ma10",
    "d1_true_reclaim_ma5": "d1_low <= d1_ma5 AND d1_close >= d1_ma5",
    "d1_true_reclaim_ma10": "d1_low <= d1_ma10 AND d1_close >= d1_ma10",
    "deprecated_d1_reclaimed_ma5_v01": "v0.1 的 d1_reclaimed_ma5 (d1_close >= d1_ma5), 已废弃",
    "deprecated_d1_reclaimed_ma10_v01": "v0.1 的 d1_reclaimed_ma10 (d1_close >= d1_ma10), 已废弃",
    "d1_close_to_ma5_bucket": "分桶(d1_close_to_ma5_raw): <-5%/-5~-2%/-2~+2%/+2~+5%/+5~+10%/>=+10%",
    "d1_open_to_close_return_raw": "d1_close / d1_open - 1 (5min 首/末bar)",
    "d1_open_to_close_bucket": "分桶(d1_open_to_close_return_raw): <-5%/-5~-2%/-2~0%/0~+2%/+2~+5%/>=+5%",
    "d1_low_to_close_recovery": "(d1_close - d1_low) / d1_low",
    "d1_high_to_close_drawdown_raw": "(d1_high - d1_close) / d1_high",
    "d1_close_location": "(d1_close - d1_low) / (d1_high - d1_low)",
    "d1_close_to_vwap_raw": "d1_close / d1_vwap - 1",
    "d1_vwap_to_close_gap": "(d1_vwap - d1_close) / d1_close (冗余反向, 已从训练视图移除)",
    "d1_intraday_range": "(d1_high - d1_low) / 前收",
    "d1_afternoon_return": "d1_close / 13:00 前末根bar close - 1",
    "d1_last_hour_return": "d1_close / 14:00 bar close - 1",
    "d1_up_bar_volume_ratio": "Σvol(close>open) / Σvol",
    "d1_down_bar_volume_ratio": "Σvol(close<open) / Σvol (与 down_bar_volume_ratio 相同, 冗余)",
    "volume_above_d1_close_ratio": "Σvol(bar close>=d1_close) / Σvol",
    "amount_above_d1_close_ratio": "Σamt(bar close>=d1_close) / Σamt (体积额双版本, 冗余)",
    "volume_above_break_close_ratio": "Σvol(bar close>=break_close) / Σvol",
    "high_zone_volume_ratio": "Σvol(typical>=d1_low+0.7*(d1_high-d1_low)) / Σvol",
    "high_zone_amount_ratio": "Σamt(typical>=高位阈值) / Σamt (冗余)",
    "late_day_sell_volume_ratio": "Σvol(time>=14:00 & close<open) / Σvol",
    "late_day_sell_amount_ratio": "Σamt(time>=14:00 & close<open) / Σamt (冗余)",
    "down_bar_volume_ratio": "Σvol(close<open) / Σvol",
    "break_open_return": "break_open / break_prev_close - 1",
    "break_high_return": "break_high / break_prev_close - 1",
    "break_close_return": "break_close / break_prev_close - 1",
    "break_intraday_range": "(break_high - break_low) / break_prev_close",
    "break_high_to_close_drawdown": "(break_high - break_close) / break_high",
    "break_upper_shadow_ratio": "(break_high - max(open,close)) / (break_high - break_low)",
    "break_lower_shadow_ratio": "(min(open,close) - break_low) / (break_high - break_low)",
    "break_close_location": "(break_close - break_low) / (break_high - break_low)",
    "break_volume_ratio_vs_board_days": "break_volume / mean(板日volume)",
    "break_amount_ratio_vs_board_days": "break_amount / mean(板日amount) (体积额双版本, 冗余)",
    "break_turnover_ratio": "break_turnover_rate / mean(板日turnover_rate) (日线无数据, 全空)",
    "break_touched_limit_up": "break_high >= 涨停价 - 0.011",
    "break_opened_from_limit_up": "break_open >= 涨停价 - 0.011",
    "recognition_score": "mean(3个板日排名百分位), 初步代理, audit_only",
    "board_day_amount_rank": "末板日 amount 在当日涨停池降序名次, audit_only",
    "board_day_turnover_rank": "末板日 turnover 在当日涨停池降序名次, audit_only",
    "board_day_volume_rank": "末板日 volume 在当日涨停池降序名次, audit_only",
    "recent_limit_up_count_10d": "近10个交易日涨停天数(含信号日)",
    "recent_limit_up_count_20d": "近20个交易日涨停天数(含信号日)",
    "recent_pool_appearance_count_10d": "近10日涨停池出现天数",
    "recent_pool_appearance_count_20d": "近20日涨停池出现天数",
    "max_board_streak_20d": "近20日最大连续涨停天数",
    "d1_ma5_slope": "ma5_today / ma5_prev - 1", "d1_ma10_slope": "ma10_today / ma10_prev - 1",
    "consecutive_days_below_ma5": "自信号日(含)向前连续 close<ma5 天数",
    "consecutive_days_below_ma10": "自信号日(含)向前连续 close<ma10 天数",
    "stage_group": "days_since_break==0 → d0; ∈{1,2} → post",
    "post_day": "d0 为空; days_since_break==1 → 1; ==2 → 2",
    "days_since_break": "冗余阶段表达(训练视图已移除), 审计保留",
    "days_since_last_limit_up": "冗余阶段表达(训练视图已移除), 审计保留",
    "repair_attempt_count": "冗余阶段表达(训练视图已移除), 审计保留",
    "training_stage_feature": "未来模型应使用的阶段字段: stage_group",
    "d2open_to_d3high_return": "d3_high / d2_open - 1",
    "d2open_to_d3close_return": "d3_close / d2_open - 1",
    "target7_d2open_d3high": "d2open_to_d3high_return >= 0.07",
    "tail_loss_5pct": "d2open_to_d3close_return <= -0.05",
    "proposed_training_weight": "1/(signal_date_candidate_count × event_observation_count), TRAIN_PRIMARY 内均值归一化为1",
    "event_observation_count": "该 event_id 在 TRAIN_PRIMARY 中的行数",
    "signal_date_candidate_count": "该 signal_date 在 TRAIN_PRIMARY 中的行数",
    "sample_role": "TRAIN_PRIMARY / SENSITIVITY_MAY_COMPLETE / AUDIT_ONLY_MAY_INCOMPLETE / EXCLUDED_QUALITY",
    "label_quality_ok": "D1/D2/D3 分钟网格完整 + 日期偏移有严格停牌证明",
    "d2_daily_minute_open_diff": "|日线D2开盘 - D2首根5min bar开盘|",
    "d3_daily_minute_high_diff": "|日线D3最高 - D3 5min最高|",
    "d3_daily_minute_close_diff": "|日线D3收盘 - D3末根5min收盘|",
    "d1_bar_count": "信号日 5min bar 数", "d2_bar_count": "D2 5min bar 数", "d3_bar_count": "D3 5min bar 数",
    "d1_first_bar_time": "信号日首根 bar 时间", "d2_first_bar_time": "D2 首根 bar 时间", "d3_first_bar_time": "D3 首根 bar 时间",
    "d1_last_bar_time": "信号日末根 bar 时间", "d2_last_bar_time": "D2 末根 bar 时间", "d3_last_bar_time": "D3 末根 bar 时间",
    "d1_expected_bar_count": "48 (完整交易日网格)", "d2_expected_bar_count": "48", "d3_expected_bar_count": "48",
    "d1_minute_complete": "bar数=48 且 首09:35 且 末15:00", "d2_minute_complete": "同 D1", "d3_minute_complete": "同 D1",
    "expected_d2_date": "统一交易日历中 signal_date 的下一交易日",
    "actual_d2_date": "实际用于标签的 D2 日期(个股 5min 序列)",
    "expected_d3_date": "统一交易日历中 expected_d2 的下一交易日",
    "actual_d3_date": "实际用于标签的 D3 日期",
    "d2_date_shift_reason": "none / suspended_proven / shift_without_proof",
    "d3_date_shift_reason": "none / suspended_proven / shift_without_proof",
    "suspension_proof_status": "not_applicable / proven / not_proven",
    "target_quality_ok": "label_quality_ok 且 d2open_to_d3high_return 可计算",
    "tail_label_quality_ok": "label_quality_ok 且 d2open_to_d3close_return 可计算",
}

# ---------------- 1. 因子质量报告 ----------------
print("[1] factor quality report ...")
rows = []
for col in all_df.columns:
    v = all_df[col]
    non_null = v.notna()
    numeric = pd.to_numeric(v, errors="coerce") if str(v.dtype) not in ("object", "string") else None
    if numeric is not None:
        num = numeric
        if str(num.dtype) == "bool":
            num = num.astype(float)
        finite = num[np.isfinite(num)]
        q = finite.quantile([0.01, 0.10, 0.50, 0.90, 0.99]) if len(finite) else pd.Series([np.nan] * 5, index=[0.01, 0.10, 0.50, 0.90, 0.99])
        row = {
            "column": col, "feature_role": ROLE.get(col, "research_hold"),
            "dtype": str(v.dtype), "available_at_d1_close": "yes",
            "missing_count": int(v.isna().sum()), "missing_rate": float(v.isna().mean()),
            "non_finite_count": int(len(num) - len(finite)) if len(num) else None,
            "min": None if finite.empty else float(finite.min()),
            "p01": None if finite.empty else float(q[0.01]), "p10": None if finite.empty else float(q[0.10]),
            "p50": None if finite.empty else float(q[0.50]), "p90": None if finite.empty else float(q[0.90]),
            "p99": None if finite.empty else float(q[0.99]), "max": None if finite.empty else float(finite.max()),
            "unique_count": int(v.nunique(dropna=True)),
            "train_primary_available_count": int(train_df[col].notna().sum()) if col in train_df.columns else 0,
            "may_sensitivity_available_count": int(may_df[col].notna().sum()) if col in may_df.columns else 0,
            "formula": FORMULAS.get(col, ""), "source": "", "notes": "",
        }
    else:
        row = {
            "column": col, "feature_role": ROLE.get(col, "research_hold"),
            "dtype": str(v.dtype), "available_at_d1_close": "yes",
            "missing_count": int(v.isna().sum()), "missing_rate": float(v.isna().mean()),
            "non_finite_count": None, "min": None, "p01": None, "p10": None, "p50": None,
            "p90": None, "p99": None, "max": None,
            "unique_count": int(v.nunique(dropna=True)),
            "train_primary_available_count": int(train_df[col].notna().sum()) if col in train_df.columns else 0,
            "may_sensitivity_available_count": int(may_df[col].notna().sum()) if col in may_df.columns else 0,
            "formula": FORMULAS.get(col, ""), "source": "", "notes": "",
        }
    rows.append(row)

fq = pd.DataFrame(rows)
fq.to_csv(OUT / "v004c_factor_quality_report_v02.csv", index=False, encoding="utf-8-sig")
print("factor quality rows:", len(fq))

# ---------------- 2. 冗余报告 ----------------
print("[2] redundancy report ...")
PAIRS = [
    ("volume_above_d1_close_ratio", "amount_above_d1_close_ratio", "volume_amount_dual", "volume_above_d1_close_ratio", "amount_above_d1_close_ratio", "同一信息(收盘价上方成交)的成交量/成交额双版本, 保留成交量"),
    ("high_zone_volume_ratio", "high_zone_amount_ratio", "volume_amount_dual", "high_zone_volume_ratio", "high_zone_amount_ratio", "高位区成交的 volume/amount 双版本, 保留 volume"),
    ("late_day_sell_volume_ratio", "late_day_sell_amount_ratio", "volume_amount_dual", "late_day_sell_volume_ratio", "late_day_sell_amount_ratio", "尾盘抛压的 volume/amount 双版本, 保留 volume"),
    ("down_bar_volume_ratio", "d1_down_bar_volume_ratio", "identical_formula", "down_bar_volume_ratio", "d1_down_bar_volume_ratio", "完全相同的公式(Σvol close<open / Σvol), 保留一个"),
    ("break_volume_ratio_vs_board_days", "break_amount_ratio_vs_board_days", "volume_amount_dual", "break_volume_ratio_vs_board_days", "break_amount_ratio_vs_board_days", "断板量比的 volume/amount 双版本, 保留 volume"),
    ("d1_close_to_vwap_raw", "d1_vwap_to_close_gap", "mirror_gap", "d1_close_to_vwap_raw", "d1_vwap_to_close_gap", "同一 VWAP 偏离的反向表达, 只保留一种方向"),
    ("deprecated_d1_reclaimed_ma5_v01", "d1_close_above_ma5", "identical_formula", "d1_close_above_ma5", "deprecated_d1_reclaimed_ma5_v01", "v0.1 reclaimed 字段与 d1_close_above_ma5 公式完全相同, 旧字段废弃"),
    ("deprecated_d1_reclaimed_ma10_v01", "d1_close_above_ma10", "identical_formula", "d1_close_above_ma10", "deprecated_d1_reclaimed_ma10_v01", "同上(ma10)"),
    ("days_since_break", "post_day", "stage_duplicate", "post_day", "days_since_break", "阶段重复表达: post_day 与 days_since_break 在 post 段完全一致"),
    ("days_since_last_limit_up", "post_day", "stage_duplicate", "post_day", "days_since_last_limit_up", "days_since_last_limit_up = post_day + 1 (常数平移)"),
    ("repair_attempt_count", "post_day", "stage_duplicate", "post_day", "repair_attempt_count", "完整事件下 repair_attempt_count = post_day + 1 (常数平移)"),
    ("d1_open_to_close_return_raw", "d1_low_to_close_recovery", "high_correlation_observed", "", "", "观测相关(非硬删除, 仅记录)"),
    ("break_high_to_close_drawdown", "d1_high_to_close_drawdown_raw", "same_day_for_d0", "", "", "days_since_break=0 时断板日即 D1, 两字段数值相同; 其余阶段不同(仅记录)"),
    ("break_close_location", "d1_close_location", "same_day_for_d0", "", "", "同上"),
]
red_rows = []
for a, b, rel, retained, excluded, reason in PAIRS:
    va = pd.to_numeric(all_df[a], errors="coerce") if a in all_df.columns else None
    vb = pd.to_numeric(all_df[b], errors="coerce") if b in all_df.columns else None
    if va is None or vb is None:
        continue
    mask = va.notna() & vb.notna()
    pear = spearman = None
    if mask.sum() > 2:
        pear = float(np.corrcoef(va[mask], vb[mask])[0, 1])
        spearman = float(pd.Series(va[mask]).corr(pd.Series(vb[mask]), method="spearman"))
    red_rows.append({
        "feature_a": a, "feature_b": b, "pearson": pear, "spearman": spearman,
        "semantic_relation": rel, "retained_feature": retained,
        "excluded_from_training_feature": excluded, "reason": reason,
    })

# 训练视图数值因子两两扫描, 找出未列入但 |spearman|>=0.85 的高相关对
train_numeric = [c for c in TRAINING if c in all_df.columns and str(all_df[c].dtype) not in ("object", "string") and c not in ("break_touched_limit_up", "break_opened_from_limit_up")]
tnum = all_df[train_numeric].apply(pd.to_numeric, errors="coerce") if train_numeric else pd.DataFrame()
listed = set((a, b) for a, b, *_ in PAIRS)
for i in range(len(tnum.columns)):
    for j in range(i + 1, len(tnum.columns)):
        a, b = tnum.columns[i], tnum.columns[j]
        if (a, b) in listed or (b, a) in listed:
            continue
        mask = tnum[a].notna() & tnum[b].notna()
        if mask.sum() < 10:
            continue
        sp = float(tnum[a][mask].corr(tnum[b][mask], method="spearman"))
        if abs(sp) >= 0.85:
            red_rows.append({
                "feature_a": a, "feature_b": b, "pearson": float(np.corrcoef(tnum[a][mask], tnum[b][mask])[0, 1]),
                "spearman": sp, "semantic_relation": "high_correlation_sweep",
                "retained_feature": "both", "excluded_from_training_feature": "",
                "reason": "训练视图扫描 |spearman|>=0.85; 仅记录, 不自动删除",
            })
red = pd.DataFrame(red_rows)
red.to_csv(OUT / "v004c_factor_redundancy_report_v02.csv", index=False, encoding="utf-8-sig")
print("redundancy rows:", len(red))

# ---------------- 3. v01-v02 差异 ----------------
print("[3] v01-v02 diff ...")
v01 = pd.read_csv(ROOT / "reports" / "research" / "v004c_dataset_20260506_20260729" / "v004c_break_repair_dataset.csv", dtype={"code": str})
v01["code"] = v01["code"].astype(str).str.zfill(6)
diff_rows = []
# 标签值核对 (v0.1 标签 vs v0.2 全量表标签)
t7_changes = 0
tl_changes = 0
for _, r in all_df.iterrows():
    old = v01[(v01["event_id"] == r["event_id"]) & (v01["signal_date"] == r["signal_date"])]
    if old.empty:
        continue
    o = old.iloc[0]
    # sample_role
    diff_rows.append({"event_id": r["event_id"], "signal_date": r["signal_date"], "code": r["code"],
                      "change_type": "sample_role_changed", "field": "sample_role",
                      "old_value": "v01_candidate", "new_value": r["sample_role"],
                      "reason": r.get("sample_role_reason", "")})
    # 标签值变化
    o_t7 = bool(o["target7_d2open_d3high"])
    n_t7 = bool(r["target7_d2open_d3high"])
    o_tl = bool(o["tail_loss_5pct"])
    n_tl = bool(r["tail_loss_5pct"])
    if o_t7 != n_t7:
        t7_changes += 1
        diff_rows.append({"event_id": r["event_id"], "signal_date": r["signal_date"], "code": r["code"],
                          "change_type": "label_quality_changed", "field": "target7_d2open_d3high",
                          "old_value": str(o_t7), "new_value": str(n_t7), "reason": "标签重算变化"})
    if o_tl != n_tl:
        tl_changes += 1
        diff_rows.append({"event_id": r["event_id"], "signal_date": r["signal_date"], "code": r["code"],
                          "change_type": "label_quality_changed", "field": "tail_loss_5pct",
                          "old_value": str(o_tl), "new_value": str(n_tl), "reason": "标签重算变化"})
    # 标签质量排除
    if not bool(r["label_quality_ok"]):
        diff_rows.append({"event_id": r["event_id"], "signal_date": r["signal_date"], "code": r["code"],
                          "change_type": "candidate_excluded_by_label_quality", "field": "label_quality_ok",
                          "old_value": "true", "new_value": "false", "reason": r["label_quality_reason"]})
    # 无变化行
    if o_t7 == n_t7 and o_tl == n_tl and bool(r["label_quality_ok"]):
        diff_rows.append({"event_id": r["event_id"], "signal_date": r["signal_date"], "code": r["code"],
                          "change_type": "unchanged", "field": "", "old_value": "", "new_value": "", "reason": ""})

# 字段级变化
FIELD_CHANGES = []
for old, new in [("d1_close_to_ma5", "d1_close_to_ma5_raw"), ("d1_low_to_ma5", "d1_low_to_ma5_raw"),
                 ("d1_high_to_ma5", "d1_high_to_ma5_raw"), ("d1_close_to_ma10", "d1_close_to_ma10_raw"),
                 ("d1_low_to_ma10", "d1_low_to_ma10_raw"),
                 ("d1_open_to_close_return", "d1_open_to_close_return_raw"),
                 ("d1_high_to_close_drawdown", "d1_high_to_close_drawdown_raw"),
                 ("d1_close_to_vwap", "d1_close_to_vwap_raw"),
                 ("d1_reclaimed_ma5", "deprecated_d1_reclaimed_ma5_v01"),
                 ("d1_reclaimed_ma10", "deprecated_d1_reclaimed_ma10_v01")]:
    FIELD_CHANGES.append({"event_id": "", "signal_date": "", "code": "", "change_type": "feature_renamed",
                          "field": old, "old_value": old, "new_value": new, "reason": "v0.2 命名修正"})
for f in ["d1_close_above_ma5", "d1_close_above_ma10", "d1_true_reclaim_ma5", "d1_true_reclaim_ma10",
          "d1_close_to_ma5_bucket", "d1_open_to_close_bucket", "stage_group", "post_day",
          "sample_role", "training_stage_feature"]:
    FIELD_CHANGES.append({"event_id": "", "signal_date": "", "code": "", "change_type": "feature_added",
                          "field": f, "old_value": "", "new_value": "added", "reason": "v0.2 新增字段"})
for f in ["amount_above_d1_close_ratio", "high_zone_amount_ratio", "late_day_sell_amount_ratio",
          "d1_down_bar_volume_ratio", "break_amount_ratio_vs_board_days", "d1_vwap_to_close_gap"]:
    FIELD_CHANGES.append({"event_id": "", "signal_date": "", "code": "", "change_type": "redundant_feature_removed_from_training",
                          "field": f, "old_value": "in_training_view", "new_value": "all_table_only",
                          "reason": "冗余字段, 从训练视图移除, 全量表保留"})
for f in ["days_since_break", "days_since_last_limit_up", "repair_attempt_count"]:
    FIELD_CHANGES.append({"event_id": "", "signal_date": "", "code": "", "change_type": "redundant_feature_removed_from_training",
                          "field": f, "old_value": "in_training_view", "new_value": "all_table_only",
                          "reason": "重复阶段表达, 训练视图只保留 stage_group/post_day"})
for f in ["recognition_score", "board_day_amount_rank", "board_day_turnover_rank", "board_day_volume_rank",
          "v004a_probability", "v004a_rank", "v002_rank"]:
    FIELD_CHANGES.append({"event_id": "", "signal_date": "", "code": "", "change_type": "feature_role_marked_audit_only",
                          "field": f, "old_value": "research", "new_value": "audit_only",
                          "reason": "缺失率高/单因子排序能力不足/现有模型覆盖不完整/避免变成 v004a 下游重排"})

diff_df = pd.DataFrame(diff_rows + FIELD_CHANGES)
diff_df.to_csv(OUT / "v004c_dataset_v01_v02_diff.csv", index=False, encoding="utf-8-sig")
print("diff rows:", len(diff_df), "| target7 changes:", t7_changes, "| tail changes:", tl_changes)

# ---------------- 4. 汇总 ----------------
print("[4] summary ...")
def summarize(g: pd.DataFrame) -> dict:
    t7 = g["target7_d2open_d3high"].fillna(False).astype(bool)
    tl = g["tail_loss_5pct"].fillna(False).astype(bool)
    close_ret = pd.to_numeric(g["d2open_to_d3close_return"], errors="coerce")
    return {
        "row_count": int(len(g)),
        "independent_event_count": int(g["event_id"].nunique()),
        "target7_count": int(t7.sum()),
        "target7_rate": float(t7.mean()) if len(g) else None,
        "tail_loss_count": int(tl.sum()),
        "tail_loss_rate": float(tl.mean()) if len(g) else None,
        "avg_d3_close_return": None if close_ret.isna().all() else float(close_ret.mean()),
        "median_d3_close_return": None if close_ret.isna().all() else float(close_ret.median()),
        "label_quality_failure_count": int((g["label_quality_ok"] == False).sum()),  # noqa: E712
    }

srows = []
all_df["month"] = all_df["signal_date"].str[:7]
for dim, col in [("sample_role", "sample_role"), ("month", "month"), ("signal_date", "signal_date"),
                 ("stage_group", "stage_group"), ("post_day", "post_day"),
                 ("board_streak_before_break", "board_streak_before_break"),
                 ("d1_close_to_ma5_bucket", "d1_close_to_ma5_bucket")]:
    for val, g in all_df.groupby(col, dropna=False):
        r = summarize(g)
        r["dimension"] = dim
        r["dimension_value"] = "NA" if pd.isna(val) else str(val)
        srows.append(r)
r = summarize(all_df)
r["dimension"] = "overall"
r["dimension_value"] = "ALL"
srows.append(r)
summary = pd.DataFrame(srows)
summary = summary.sort_values(["dimension", "dimension_value"])
summary.to_csv(OUT / "v004c_dataset_summary_v02.csv", index=False, encoding="utf-8-sig")
print("summary rows:", len(summary))
print("done")

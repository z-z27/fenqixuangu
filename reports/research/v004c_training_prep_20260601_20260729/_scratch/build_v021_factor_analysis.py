# -*- coding: utf-8 -*-
"""v004c 正式因子分析 (只读, 不拟合任何模型)

对象: v0.2.1 TRAIN_PRIMARY (728 行) 与日线标签
四模型头: d0_target / d0_tail / post_target / post_tail
单因子分析: 覆盖率 / AUC(秩法) / 分桶命中率 / 月稳定性 / 时间子区间方向一致性 / 极端贡献 / 相关性
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_training_prep_20260601_20260729"

EPS = 1e-9
T7 = "target7_daily_d2open_d3high"
TL = "tail_loss_daily_5pct"
CLOSE_RET = "daily_d2open_to_d3close_return"

train = pd.read_csv(OUT / "v004c_training_primary_v021.csv", dtype={"code": str})
train["code"] = train["code"].astype(str).str.zfill(6)
print("train rows:", len(train))

# ---------------- 因子定义 ----------------
def hinge(s, k):
    return (s - k).clip(lower=0.0)


FACTORS = [
    # A. MA5 位置
    ("d1_close_to_ma5_raw", "d1_close / d1_ma5 - 1", "continuous"),
    ("ma5_gap_clipped", "clip(d1_close_to_ma5_raw, -0.15, 0.20)", "continuous"),
    ("ma5_below_minus_2_hinge", "max(-0.02 - d1_close_to_ma5_raw, 0)", "hinge"),
    ("ma5_above_5_hinge", "max(d1_close_to_ma5_raw - 0.05, 0)", "hinge"),
    ("ma5_above_10_hinge", "max(d1_close_to_ma5_raw - 0.10, 0)", "hinge"),
    ("d1_true_reclaim_ma5", "d1_low <= d1_ma5 AND d1_close >= d1_ma5", "flag"),
    ("d1_close_to_ma5_bucket", "分桶(-5/-2/+2/+5/+10%)", "bucket"),
    # B. D1 修复幅度
    ("d1_open_to_close_return_raw", "d1_close / d1_open - 1", "continuous"),
    ("d1_oc_return_clipped", "clip(d1_open_to_close_return_raw, -0.10, 0.10)", "continuous"),
    ("d1_repair_above_2_hinge", "max(d1_open_to_close_return_raw - 0.02, 0)", "hinge"),
    ("d1_repair_above_5_flag", "1 if d1_open_to_close_return_raw >= 0.05 else 0", "flag"),
    ("d1_open_to_close_bucket", "分桶(-5/-2/0/+2/+5%)", "bucket"),
    ("d1_high_to_close_drawdown_raw", "(d1_high - d1_close) / d1_high", "continuous"),
    # C. 收盘承接
    ("d1_close_location", "(d1_close - d1_low) / (d1_high - d1_low)", "continuous"),
    ("d1_close_to_vwap_raw", "d1_close / d1_vwap - 1", "continuous"),
    ("volume_above_d1_close_ratio", "Σvol(bar close >= d1_close) / Σvol", "continuous"),
    # D. 尾盘与全天抛压
    ("late_day_sell_volume_ratio", "Σvol(time>=14:00 & close<open) / Σvol", "continuous"),
    ("down_bar_volume_ratio", "Σvol(close<open) / Σvol", "continuous"),
    # E. 断板结构
    ("break_volume_ratio_vs_board_days", "break_volume / mean(板日volume)", "continuous"),
    ("break_volume_log_ratio", "log(max(break_volume_ratio_vs_board_days, 1e-9))", "continuous"),
    ("break_volume_distance_from_one", "abs(log(break_volume_ratio_vs_board_days))", "continuous"),
    ("break_high_to_close_drawdown", "(break_high - break_close) / break_high", "continuous"),
    ("break_close_location", "(break_close - break_low) / (break_high - break_low)", "continuous"),
    ("break_touched_limit_up", "break_high >= 涨停价 - 0.011", "flag"),
    ("break_opened_from_limit_up", "break_open >= 涨停价 - 0.011", "flag"),
    # F. 身份与老化
    ("board_streak_is_3", "board_streak_before_break == 3", "flag"),
    ("post_day", "post_day (仅 post 头)", "ordinal"),
]

# 计算因子列
train["ma5_gap_clipped"] = train["d1_close_to_ma5_raw"].clip(-0.15, 0.20)
train["ma5_below_minus_2_hinge"] = (-0.02 - train["d1_close_to_ma5_raw"]).clip(lower=0.0).fillna(0.0)
train["ma5_above_5_hinge"] = hinge(train["d1_close_to_ma5_raw"], 0.05).clip(lower=0.0).fillna(0.0)
train["ma5_above_10_hinge"] = hinge(train["d1_close_to_ma5_raw"], 0.10).clip(lower=0.0).fillna(0.0)
train["d1_oc_return_clipped"] = train["d1_open_to_close_return_raw"].clip(-0.10, 0.10)
train["d1_repair_above_2_hinge"] = hinge(train["d1_open_to_close_return_raw"], 0.02).clip(lower=0.0).fillna(0.0)
train["d1_repair_above_5_flag"] = (train["d1_open_to_close_return_raw"] >= 0.05).astype(float)
train["break_volume_log_ratio"] = np.log(train["break_volume_ratio_vs_board_days"].clip(lower=EPS))
train["break_volume_distance_from_one"] = train["break_volume_log_ratio"].abs()
train["board_streak_is_3"] = (train["board_streak_before_break"] == 3).astype(float)

# ---------------- 工具函数 ----------------
def rank_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    """秩法 AUC (Mann-Whitney U); 更高 score → 更可能为正类"""
    m = y_true.notna() & y_score.notna()
    y = y_true[m].astype(float)
    s = y_score[m].astype(float)
    pos = y == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = s.rank(method="average")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def make_buckets(s: pd.Series) -> pd.Series:
    """连续/偏斜 → 秩五分位; 二值/少值 → 自然桶"""
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().nunique() <= 2:
        return v.map(lambda x: 1 if x > 0 else 0)
    r = v.rank(method="first")
    return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")


def bucket_stats(frame: pd.DataFrame, factor: str, target: str) -> list[dict]:
    out = []
    if factor.endswith("_bucket"):
        b = frame[factor]
    else:
        b = make_buckets(factor_series(frame, factor))
    tmp = frame.assign(_b=b, _t7=frame[T7].fillna(False).astype(bool),
                       _tl=frame[TL].fillna(False).astype(bool),
                       _cr=pd.to_numeric(frame[CLOSE_RET], errors="coerce"))
    for bi, g in tmp.dropna(subset=["_b"]).groupby("_b"):
        out.append({
            "bucket": bi, "bucket_n": int(len(g)),
            "bucket_target7_rate": float(g["_t7"].mean()),
            "bucket_tail_rate": float(g["_tl"].mean()),
            "bucket_avg_close_return": None if g["_cr"].isna().all() else float(g["_cr"].mean()),
        })
    return out


def extreme_contribution(frame: pd.DataFrame, factor: str) -> dict:
    b = make_buckets(factor_series(frame, factor))
    tmp = frame.assign(_b=b, _t7=frame[T7].fillna(False).astype(bool),
                       _tl=frame[TL].fillna(False).astype(bool))
    t7_total = int(tmp["_t7"].sum())
    tl_total = int(tmp["_tl"].sum())
    if t7_total == 0 or tl_total == 0:
        return {"extreme_bucket_t7_share": None, "extreme_bucket_tail_share": None,
                "expected_equal_share": None}
    # 极端 = 桶内 t7 率最高/最低的桶
    rates = tmp.groupby("_b")["_t7"].mean()
    hi = rates.idxmax()
    lo = rates.idxmin()
    return {
        "extreme_bucket_t7_share": float(tmp.loc[tmp["_b"] == hi, "_t7"].sum()) / t7_total,
        "extreme_bucket_tail_share": float(tmp.loc[tmp["_b"] == lo, "_tl"].sum()) / tl_total,
        "expected_equal_share": 1.0 / max(1, int(rates.nunique())),
    }


def direction(auc_val: float | None) -> str:
    if auc_val is None:
        return "NA"
    if abs(auc_val - 0.5) < 0.005:
        return "flat"
    return "up" if auc_val > 0.5 else "down"


BUCKET_ORDINAL = {
    "below_minus_5pct": 1, "minus_5_to_minus_2pct": 2, "near_ma5": 3,
    "plus_2_to_plus_5pct": 4, "plus_5_to_plus_10pct": 5, "above_plus_10pct": 6,
    "minus_2_to_0pct": 2.5, "zero_to_plus_2pct": 3.5, "above_plus_5pct": 6,
}


def factor_series(frame: pd.DataFrame, factor: str) -> pd.Series:
    """因子数值序列: bucket 因子转有序编码, 其余 to_numeric"""
    if factor.endswith("_bucket"):
        return frame[factor].map(BUCKET_ORDINAL).astype(float)
    return pd.to_numeric(frame[factor], errors="coerce")


def interpret_buckets(bkt_df: pd.DataFrame, head: str, factor: str) -> dict:
    rows = bkt_df[(bkt_df["model_head"] == head) & (bkt_df["factor"] == factor)].sort_values("bucket")
    if rows.empty:
        return {"target7_interpretation": "", "tail_interpretation": ""}
    t7 = rows["bucket_target7_rate"].tolist()
    tl = rows["bucket_tail_rate"].tolist()
    return {
        "target7_interpretation": "桶率 " + " ".join(f"{v:.2f}" for v in t7),
        "tail_interpretation": "桶尾 " + " ".join(f"{v:.2f}" for v in tl),
    }


# ---------------- 分析循环 ----------------
HEADS = [
    ("d0_target", "d0", T7),
    ("d0_tail", "d0", TL),
    ("post_target", "post", T7),
    ("post_tail", "post", TL),
]

univariate_rows = []
bucket_rows = []
stability_rows = []
redundancy_rows = []
max_corr = {}

for head_name, stage, target in HEADS:
    frame = train[train["stage_group"] == stage].copy()
    if stage == "post":
        frame = frame[frame["post_day"].notna()]
    print(f"\n== {head_name}: n={len(frame)} ==", flush=True)
    print(f"   target rate: {frame[target].fillna(False).mean():.3f}")
    dates = sorted(frame["signal_date"].unique())
    june = frame[frame["signal_date"] <= "2026-06-30"]
    july = frame[frame["signal_date"] >= "2026-07-01"]
    thirds = np.array_split(dates, 3)
    sub_frames = [(f"s{i+1}", frame[frame["signal_date"].isin(list(ds))]) for i, ds in enumerate(thirds)]

    # 相关性矩阵 (该头内候选因子)
    factor_cols = [f for f, _, _ in FACTORS if f in frame.columns]
    corr_frame = frame[factor_cols].apply(pd.to_numeric, errors="coerce")
    pear = corr_frame.corr(method="pearson")
    spear = corr_frame.corr(method="spearman")
    pair_rows = []
    seen = set()
    for i, a in enumerate(factor_cols):
        for b in factor_cols[i + 1:]:
            seen.add((a, b))
            sp = spear.loc[a, b]
            if pd.notna(sp) and abs(sp) >= 0.70:
                pair_rows.append({
                    "model_head": head_name, "feature_a": a, "feature_b": b,
                    "pearson": float(pear.loc[a, b]) if pd.notna(pear.loc[a, b]) else None,
                    "spearman": float(sp),
                })
    redundancy_rows.extend(pair_rows)
    for a in factor_cols:
        others = [sp for (x, y), sp in spear.stack().items() if x == a and y != a and pd.notna(sp)]
        max_corr[(head_name, a)] = max((abs(v) for v in others), default=None)

    for factor, formula, ftype in FACTORS:
        if factor not in frame.columns:
            continue
        if factor == "post_day" and stage != "post":
            continue
        s = factor_series(frame, factor)
        n_avail = int(s.notna().sum())
        miss_rate = float(s.isna().mean())
        auc_val = rank_auc(frame[target], s)
        auc_june = rank_auc(june[target], factor_series(june, factor)) if len(june) else None
        auc_july = rank_auc(july[target], factor_series(july, factor)) if len(july) else None
        sub_dirs = []
        for _, sf in sub_frames:
            if sf.empty:
                continue
            sub_dirs.append(direction(rank_auc(sf[target], factor_series(sf, factor))))
        overall_dir = direction(auc_val)
        n_agree = sum(1 for d in sub_dirs if d == overall_dir)
        consistency = (n_agree / len(sub_dirs)) if sub_dirs else None
        ext = extreme_contribution(frame, factor) if n_avail else {}
        buckets = bucket_stats(frame, factor, target)
        for b in buckets:
            bucket_rows.append({"model_head": head_name, "factor": factor, **b})
        univariate_rows.append({
            "model_head": head_name, "factor": factor, "formula": formula, "factor_type": ftype,
            "n_available": n_avail, "missing_rate": round(miss_rate, 4),
            "univariate_auc": auc_val, "june_auc": auc_june, "july_auc": auc_july,
            "overall_direction": overall_dir, "subinterval_directions": "|".join(sub_dirs),
            "temporal_direction_consistency": consistency,
            "extreme_bucket_t7_share": ext.get("extreme_bucket_t7_share"),
            "extreme_bucket_tail_share": ext.get("extreme_bucket_tail_share"),
            "max_abs_correlation": max_corr.get((head_name, factor)),
        })
        _fmt = lambda v: "NA" if v is None else f"{v:.3f}"
        print(f"  {factor}: auc={_fmt(auc_val)} june={_fmt(auc_june)} july={_fmt(auc_july)} "
              f"sub={sub_dirs} miss={miss_rate:.2f}")

uni = pd.DataFrame(univariate_rows)
bkt = pd.DataFrame(bucket_rows)
stab = uni[["model_head", "factor", "n_available", "missing_rate", "univariate_auc",
            "june_auc", "july_auc", "overall_direction", "subinterval_directions",
            "temporal_direction_consistency", "extreme_bucket_t7_share", "extreme_bucket_tail_share"]].copy()
stab["june_direction"] = stab["june_auc"].map(direction)
stab["july_direction"] = stab["july_auc"].map(direction)
uni.to_csv(OUT / "v004c_factor_univariate_formal.csv", index=False, encoding="utf-8-sig")
bkt.to_csv(OUT / "v004c_factor_bucket_formal.csv", index=False, encoding="utf-8-sig")
stab.to_csv(OUT / "v004c_factor_stability_formal.csv", index=False, encoding="utf-8-sig")
red = pd.DataFrame(redundancy_rows)
red.to_csv(OUT / "v004c_factor_redundancy_formal.csv", index=False, encoding="utf-8-sig")
print("\nunivariate rows:", len(uni), "| bucket rows:", len(bkt), "| stability:", len(stab),
      "| redundancy pairs:", len(red))

# ---------------- 准入决策 ----------------
print("\n[admission decisions] ...", flush=True)

# 组定义 (用于组内表达多样性限制)
GROUPS = {
    "A": ["d1_close_to_ma5_raw", "ma5_gap_clipped", "ma5_below_minus_2_hinge", "ma5_above_5_hinge",
          "ma5_above_10_hinge", "d1_true_reclaim_ma5", "d1_close_to_ma5_bucket"],
    "B": ["d1_open_to_close_return_raw", "d1_oc_return_clipped", "d1_repair_above_2_hinge",
          "d1_repair_above_5_flag", "d1_open_to_close_bucket", "d1_high_to_close_drawdown_raw"],
    "C": ["d1_close_location", "d1_close_to_vwap_raw", "volume_above_d1_close_ratio"],
    "D": ["late_day_sell_volume_ratio", "down_bar_volume_ratio"],
    "E": ["break_volume_ratio_vs_board_days", "break_volume_log_ratio", "break_volume_distance_from_one",
          "break_high_to_close_drawdown", "break_close_location", "break_touched_limit_up", "break_opened_from_limit_up"],
    "F": ["board_streak_is_3", "post_day"],
}
# 组内名额 (D 组: 两因子相关 < 0.85 时允许 2, 否则 1)
_dd = train[["late_day_sell_volume_ratio", "down_bar_volume_ratio"]].apply(pd.to_numeric, errors="coerce")
D_PAIR_CORR = float(_dd.corr(method="spearman").iloc[0, 1]) if len(_dd) > 2 else None
GROUP_CAPS = {"A": 2, "B": 2, "C": 1, "D": 2 if (D_PAIR_CORR is not None and abs(D_PAIR_CORR) < 0.85) else 1,
              "E": 3, "F": 1}
print("D 组 spearman:", D_PAIR_CORR, "| caps:", GROUP_CAPS)


def pair_corr(head_name: str, a: str, b: str) -> float | None:
    pr = red[(red["model_head"] == head_name) &
             (((red["feature_a"] == a) & (red["feature_b"] == b)) |
              ((red["feature_a"] == b) & (red["feature_b"] == a)))]
    if pr.empty:
        return None
    return float(pr["spearman"].iloc[0])


def group_of(f: str) -> str:
    for g, fs in GROUPS.items():
        if f in fs:
            return g
    return "?"


def factor_df(f: str) -> int:
    return 5 if f.endswith("_bucket") else 1


decision_rows = []
for head_name, stage, target in HEADS:
    h = uni[uni["model_head"] == head_name].set_index("factor")
    ranked = h.sort_values(["univariate_auc", "temporal_direction_consistency"],
                           ascending=[False, False])
    admitted: list[str] = []
    for factor in ranked.index:
        r = ranked.loc[factor]
        group = group_of(factor)
        auc = r["univariate_auc"]
        cons = r["temporal_direction_consistency"]
        miss = r["missing_rate"]
        maxc = r["max_abs_correlation"]
        df = factor_df(factor)
        reasons = []
        decision = "AUDIT_ONLY"
        if miss > 0.15:
            decision = "REJECT_LOW_COVERAGE"
            reasons.append(f"coverage={1-miss:.2f} < 0.85")
        elif auc is None or auc < 0.51:
            decision = "AUDIT_ONLY"
            reasons.append(f"auc={auc:.3f} 低于 0.51, 证据弱")
        elif cons is not None and cons < 2/3:
            decision = "REJECT_UNSTABLE"
            reasons.append(f"方向一致率 {cons:.2f} < 0.67")
        else:
            # 与已准入因子的相关性
            corr_with_admitted = [abs(c) for c in (pair_corr(head_name, factor, a) for a in admitted) if c is not None]
            max_adm = max(corr_with_admitted) if corr_with_admitted else 0.0
            if max_adm >= 0.98:
                decision = "REJECT_REDUNDANT"
                reasons.append(f"与已准入因子 |spearman|={max_adm:.2f} >= 0.98, 信息等价")
            elif max_adm > 0.85:
                decision = "ADMIT_ALTERNATIVE"
                reasons.append(f"与已准入因子 |spearman|={max_adm:.2f} > 0.85, 高度相关作备用")
            elif sum(1 for a in admitted if group_of(a) == group) >= GROUP_CAPS[group]:
                decision = "ADMIT_ALTERNATIVE"
                reasons.append(f"组{group}名额已满 (cap={GROUP_CAPS[group]}), 降级备用")
            elif len(admitted) >= 6:
                decision = "ADMIT_ALTERNATIVE"
                reasons.append("主要名额已满 (6), 降级备用")
            elif auc >= 0.53 and cons >= 2/3:
                decision = "ADMIT_PRIMARY"
                admitted.append(factor)
                reasons.append(f"auc={auc:.3f} 方向一致率={cons:.2f}")
            elif auc >= 0.51 and cons >= 2/3 and len(admitted) < 4:
                decision = "ADMIT_PRIMARY"
                admitted.append(factor)
                reasons.append(f"补充: auc={auc:.3f} 方向一致率={cons:.2f} (为达到最低 4 因子表达; 证据弱, 需 walk-forward 验证)")
            else:
                decision = "ADMIT_ALTERNATIVE"
                reasons.append(f"auc={auc:.3f} 中等, 作备用")
        interpretation = interpret_buckets(bkt, head_name, factor)
        decision_rows.append({
            "model_head": head_name, "factor_name": factor,
            "mathematical_expression": FACTORS[[f for f, _, _ in FACTORS].index(factor)][1],
            "effective_degrees_of_freedom": df,
            "coverage": round(1 - miss, 4), "univariate_auc": auc,
            "june_direction": direction(r["june_auc"]), "july_direction": direction(r["july_auc"]),
            "temporal_direction_consistency": cons, "max_abs_correlation": maxc,
            "target7_interpretation": interpretation["target7_interpretation"],
            "tail_interpretation": interpretation["tail_interpretation"],
            "decision": decision, "reason": "; ".join(reasons) if reasons else "",
        })
    prim = [d["factor_name"] for d in decision_rows if d["model_head"] == head_name and d["decision"] == "ADMIT_PRIMARY"]
    alt = [d["factor_name"] for d in decision_rows if d["model_head"] == head_name and d["decision"] == "ADMIT_ALTERNATIVE"]
    df_total = sum(factor_df(f) for f in prim)
    print(f"{head_name}: PRIMARY={prim} (df={df_total}) | ALT={alt}")


dec = pd.DataFrame(decision_rows)
dec.to_csv(OUT / "v004c_factor_admission_decision.csv", index=False, encoding="utf-8-sig")
print("\nadmission rows:", len(dec))
print("decisions:", dec["decision"].value_counts().to_dict())
print("done")

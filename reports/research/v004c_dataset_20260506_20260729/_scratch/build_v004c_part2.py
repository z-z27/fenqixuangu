# -*- coding: utf-8 -*-
"""v004c Part 2: 比较字段 (v004a/v002) + 冻结 v004a 只读重建验证 + 最终主样本/审计输出

- v004a 现有评分文件: grid_v2 (walk-forward), holdout 0626/0701 (冻结), daily_v005 (冻结)
- v004a 缺口 (2026-05-06..06-01): 用冻结系数 + history candidates scorable 池只读重建,
  先与 holdout 冻结评分核对通过才采用
- v002: 现成 daily signals daily_rank (06-25, 07-01..07-31), 其余留空
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\fenqixuangu")
OUT = ROOT / "reports" / "research" / "v004c_dataset_20260506_20260729"
SCRATCH = OUT / "_scratch"

COEFF_FILE = ROOT / "configs" / "models" / "v004a_coefficients_2026-06-26.csv"
GRID_V2 = ROOT / "reports" / "v004a" / "grid_v2_scored" / "v004a_scored_candidates.csv"
HOLDOUT_0626 = ROOT / "reports" / "v005_fixed_grid_holdout_2026-06-26_2026-06-30" / "v005_fixed_grid_holdout_scored_candidates.csv"
HOLDOUT_0701 = ROOT / "reports" / "v005_fixed_grid_holdout_2026-07-01_2026-07-03" / "v005_fixed_grid_holdout_scored_candidates.csv"
DAILY_V005 = ROOT / "reports" / "daily_v005"
SIGNALS_DIR = ROOT / "reports" / "daily_signals"
HC_0529 = ROOT / "reports" / "history_samples" / "2026-05-06_2026-06-29" / "history_candidates_2026-05-06_2026-06-29.csv"
HC_0630 = ROOT / "reports" / "history_samples" / "2026-06-26_2026-06-30" / "history_candidates_2026-06-26_2026-06-30.csv"
HC_0703 = ROOT / "reports" / "history_samples" / "2026-07-01_2026-07-03" / "history_candidates_2026-07-01_2026-07-03.csv"

FROZEN_L2 = 0.3
FROZEN_PW = 1.5


def norm_code(series):
    return series.astype(str).str.zfill(6)


# 名称解析(与 part1 相同来源: 涨停池精确日期 → 最新 → universe)
_pool_all: pd.DataFrame | None = None
_universe_names: dict | None = None


def _resolve_name(code: str, date: str) -> str:
    global _pool_all, _universe_names
    if _pool_all is None:
        _pool_all = pd.concat(
            [pd.read_pickle(p) for p in sorted(glob.glob(str(ROOT / "data" / "cache" / "limit_ups" / "*_limitups.pkl")))],
            ignore_index=True)
        _pool_all["code"] = norm_code(_pool_all["code"])
        _pool_all["trade_date"] = _pool_all["trade_date"].astype(str)
    if _universe_names is None:
        uni = pd.read_pickle(ROOT / "data" / "cache" / "universe" / "eastmoney_main_board_universe.pkl")
        _universe_names = dict(zip(uni["code"].astype(str).str.zfill(6), uni["name"].astype(str)))
    row = _pool_all[(_pool_all["trade_date"] == date) & (_pool_all["code"] == code)]
    if not row.empty and str(row.iloc[0]["name"]).strip():
        return str(row.iloc[0]["name"]).strip()
    latest = _pool_all[_pool_all["code"] == code].sort_values("trade_date")
    if not latest.empty and str(latest.iloc[-1]["name"]).strip():
        return str(latest.iloc[-1]["name"]).strip()
    return str(_universe_names.get(code, ""))


# ---------------- 1. 加载候选与现有评分 ----------------
cand = pd.read_csv(SCRATCH / "stage3_candidates_feat.csv", dtype={"code": str})
cand["code"] = norm_code(cand["code"])
cand["signal_date"] = cand["signal_date"].astype(str)
cand["break_date"] = cand["break_date"].astype(str)
print("candidates:", len(cand))

# v004a 现有评分表 (只保留 logistic_v004a_weighted, 优先级 daily_v005 > holdout_0701 > holdout_0626 > grid_v2)
frames = []
grid = pd.read_csv(GRID_V2, dtype={"code": str})
grid["code"] = norm_code(grid["code"])
grid = grid[(grid["evaluation_scope"] == "walk_forward")
            & (grid["model_id"] == "logistic_v004a_weighted")
            & (pd.to_numeric(grid["l2"], errors="coerce") == FROZEN_L2)
            & (pd.to_numeric(grid["positive_weight"], errors="coerce") == FROZEN_PW)]
grid = grid.drop_duplicates(["signal_date", "code"]).reset_index(drop=True)
grid["source"] = "grid_v2_walk_forward"
grid["priority"] = 0
frames.append(grid[["signal_date", "code", "model_score", "model_probability", "model_rank", "source", "priority"]])

for path, src, prio in ((HOLDOUT_0626, "holdout_0626_frozen", 1), (HOLDOUT_0701, "holdout_0701_frozen", 2)):
    h = pd.read_csv(path, dtype={"code": str})
    h["code"] = norm_code(h["code"])
    h = h[h["model_id"] == "logistic_v004a_weighted"]
    h = h.drop_duplicates(["signal_date", "code"]).reset_index(drop=True)
    h["source"] = src
    h["priority"] = prio
    frames.append(h[["signal_date", "code", "model_score", "model_probability", "model_rank", "source", "priority"]])

for path in sorted(glob.glob(str(DAILY_V005 / "*" / "v005_daily_scored_candidates_*.csv"))):
    d = pd.read_csv(path, dtype={"code": str})
    d["code"] = norm_code(d["code"])
    d = d[d["model_id"] == "logistic_v004a_weighted"]
    d["source"] = "daily_v005_frozen"
    d["priority"] = 3
    frames.append(d[["signal_date", "code", "model_score", "model_probability", "model_rank", "source", "priority"]])

v004a_existing = pd.concat(frames, ignore_index=True)
v004a_existing["signal_date"] = v004a_existing["signal_date"].astype(str)
v004a_existing = (v004a_existing.sort_values(["signal_date", "code", "priority"], ascending=[True, True, False])
                  .drop_duplicates(["signal_date", "code"], keep="first")
                  .drop(columns=["priority"])
                  .reset_index(drop=True))
print("v004a existing rows:", len(v004a_existing), "dates:",
      v004a_existing["signal_date"].min(), "->", v004a_existing["signal_date"].max())
print("v004a existing source counts:", v004a_existing["source"].value_counts().to_dict())

# ---------------- 2. 冻结 v004a 只读重建 ----------------
print("\n[2] frozen v004a reconstruction ...")
coeff = pd.read_csv(COEFF_FILE)
coeff = coeff[coeff["selection_policy"] == "training"].copy()
beta_terms = coeff["term"].tolist()
beta = coeff["coefficient"].to_numpy(float)
print("coefficient terms:", len(beta_terms))
assert beta_terms[0] == "intercept"

BASE_FEATS = ["rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct", "rank_trend_hold_score",
              "rank_total_score", "rank_theme_score", "rank_days_since_d0",
              "rank_log_candidate_base_price", "rank_active_money_score", "rank_d1_close_vwap_pct"]
INTER_FEATS = ["inter_close_low", "inter_close_trend", "inter_total_trend",
               "inter_total_active", "inter_low_active", "spread_close_low"]
BUCKET_FEATS = ["days_since_d0_le1", "days_since_d0_eq2", "days_since_d0_ge3"]


def build_scored_from_samples(hc: pd.DataFrame, coeff_df: pd.DataFrame):
    """从 history candidates 重建 scorable 池并应用冻结系数 (只读, 复制自 v004a.prepare_v004a_samples)"""
    hc = hc.copy()
    hc["code"] = norm_code(hc["code"])
    hc["signal_date"] = hc["signal_date"].astype(str)
    hc["eligible_for_trade"] = hc["eligible_for_trade"].fillna(False).astype(bool)
    for col in ("d2open_d3high_return_pct", "d2open_d3close_return_pct", "candidate_base_price"):
        hc[col] = pd.to_numeric(hc[col], errors="coerce")
    scorable = hc[
        hc["eligible_for_trade"]
        & hc["d2open_d3high_return_pct"].notna()
        & hc["d2open_d3close_return_pct"].notna()
        & hc["candidate_base_price"].notna()
        & (hc["candidate_base_price"] > 0)
    ].copy()
    if scorable.empty:
        return pd.DataFrame()
    scorable["log_candidate_base_price"] = np.log(pd.to_numeric(scorable["candidate_base_price"], errors="coerce").astype(float))
    rank_specs = [("d1_close_ma10_pct", "rank_d1_close_ma10_pct"),
                  ("d1_low_ma10_pct", "rank_d1_low_ma10_pct"),
                  ("trend_hold_score", "rank_trend_hold_score"),
                  ("total_score", "rank_total_score"),
                  ("theme_score", "rank_theme_score"),
                  ("days_since_d0", "rank_days_since_d0"),
                  ("log_candidate_base_price", "rank_log_candidate_base_price"),
                  ("active_money_score", "rank_active_money_score"),
                  ("d1_close_vwap_pct", "rank_d1_close_vwap_pct")]
    for raw_col, rank_col in rank_specs:
        if raw_col not in scorable.columns:
            scorable[raw_col] = np.nan
        scorable[rank_col] = (
            scorable.groupby("signal_date", dropna=False)[raw_col]
            .transform(lambda v: pd.to_numeric(v, errors="coerce").rank(pct=True, method="average"))
            .astype(float)
            .fillna(0.5)
        )
    days_values = pd.to_numeric(scorable["days_since_d0"], errors="coerce")
    scorable["days_since_d0_le1"] = (days_values <= 1).fillna(False).astype(int)
    scorable["days_since_d0_eq2"] = (days_values == 2).fillna(False).astype(int)
    scorable["days_since_d0_ge3"] = (days_values >= 3).fillna(False).astype(int)
    scorable["inter_close_low"] = scorable["rank_d1_close_ma10_pct"].astype(float) * scorable["rank_d1_low_ma10_pct"].astype(float)
    scorable["inter_close_trend"] = scorable["rank_d1_close_ma10_pct"].astype(float) * scorable["rank_trend_hold_score"].astype(float)
    scorable["inter_total_trend"] = scorable["rank_total_score"].astype(float) * scorable["rank_trend_hold_score"].astype(float)
    scorable["inter_total_active"] = scorable["rank_total_score"].astype(float) * scorable["rank_active_money_score"].astype(float)
    scorable["inter_low_active"] = scorable["rank_d1_low_ma10_pct"].astype(float) * scorable["rank_active_money_score"].astype(float)
    scorable["spread_close_low"] = scorable["rank_d1_close_ma10_pct"].astype(float) - scorable["rank_d1_low_ma10_pct"].astype(float)
    feature_cols = [*BASE_FEATS, *INTER_FEATS, *BUCKET_FEATS]
    for col in feature_cols:
        scorable[col] = pd.to_numeric(scorable[col], errors="coerce").fillna(0.0).astype(float)
    # 按 beta 顺序组装 (intercept 之后 18 项)
    x_cols = [col for col in beta_terms[1:] if col in feature_cols]
    assert len(x_cols) == len(beta_terms) - 1, f"feature/coefficient mismatch: {len(x_cols)} vs {len(beta_terms) - 1}"
    x = np.column_stack([scorable[col].to_numpy(float) for col in x_cols])
    prob = 1.0 / (1.0 + np.exp(-np.clip(x @ beta[1:] + beta[0], -40, 40)))
    scorable["model_probability"] = prob
    scorable["graph_quality_score"] = pd.to_numeric(scorable["graph_quality_score"], errors="coerce").fillna(0.0)
    result = scorable.sort_values(["signal_date", "model_probability", "graph_quality_score", "code"],
                                  ascending=[True, False, False, True])
    result["model_rank"] = result.groupby("signal_date").cumcount() + 1
    return result[["signal_date", "code", "model_probability", "model_rank"]]


# 验证 1: 06-26..06-30 vs holdout_0626 (冻结)
val_report = []
hc_0630 = pd.read_csv(HC_0630)
recon_0630 = build_scored_from_samples(hc_0630, coeff)
h0626 = pd.read_csv(HOLDOUT_0626, dtype={"code": str})
h0626["code"] = norm_code(h0626["code"])
h0626["signal_date"] = h0626["signal_date"].astype(str)
h0626 = h0626[h0626["model_id"] == "logistic_v004a_weighted"].drop_duplicates(["signal_date", "code"])
m = recon_0630.merge(h0626[["signal_date", "code", "model_probability", "model_rank"]],
                     on=["signal_date", "code"], suffixes=("_recon", "_file"))
if m.empty:
    print("WARN: recon 0626 has no overlap with holdout file")
else:
    prob_diff = float(np.abs(m["model_probability_recon"] - m["model_probability_file"]).max())
    rank_match = float((m["model_rank_recon"] == m["model_rank_file"]).mean())
    print(f"recon vs holdout_0626: overlap={len(m)} prob_max_diff={prob_diff:.2e} rank_match_rate={rank_match:.4f}")
    val_report.append({"validation": "recon_vs_holdout_0626", "overlap_rows": len(m),
                       "prob_max_abs_diff": prob_diff, "rank_match_rate": rank_match})

# 验证 2: 07-01..07-03 vs holdout_0701 (冻结)
hc_0703 = pd.read_csv(HC_0703)
recon_0703 = build_scored_from_samples(hc_0703, coeff)
h0701 = pd.read_csv(HOLDOUT_0701, dtype={"code": str})
h0701["code"] = norm_code(h0701["code"])
h0701["signal_date"] = h0701["signal_date"].astype(str)
h0701 = h0701[h0701["model_id"] == "logistic_v004a_weighted"].drop_duplicates(["signal_date", "code"])
m2 = recon_0703.merge(h0701[["signal_date", "code", "model_probability", "model_rank"]],
                      on=["signal_date", "code"], suffixes=("_recon", "_file"))
if not m2.empty:
    prob_diff2 = float(np.abs(m2["model_probability_recon"] - m2["model_probability_file"]).max())
    rank_match2 = float((m2["model_rank_recon"] == m2["model_rank_file"]).mean())
    print(f"recon vs holdout_0701: overlap={len(m2)} prob_max_diff={prob_diff2:.2e} rank_match_rate={rank_match2:.4f}")
    val_report.append({"validation": "recon_vs_holdout_0701", "overlap_rows": len(m2),
                       "prob_max_abs_diff": prob_diff2, "rank_match_rate": rank_match2})

pd.DataFrame(val_report).to_csv(SCRATCH / "v004a_recon_validation.csv", index=False, encoding="utf-8-sig")

# 重建 05-06..06-01
hc_0529 = pd.read_csv(HC_0529)
recon_0529 = build_scored_from_samples(hc_0529, coeff)
recon_may = recon_0529[recon_0529["signal_date"] <= "2026-06-01"].copy()
recon_may["source"] = "frozen_reconstruction"
print("recon May rows:", len(recon_may), "dates:", sorted(recon_may["signal_date"].unique()))

# ---------------- 3. v002 现成评分 ----------------
v002_frames = []
for path in sorted(glob.glob(str(SIGNALS_DIR / "signals_*.csv"))):
    date = Path(path).stem.replace("signals_", "")
    if date < "2026-06-25":
        continue
    s = pd.read_csv(path, dtype={"code": str})
    s["code"] = norm_code(s["code"])
    if "daily_rank" not in s.columns or "research_score" not in s.columns:
        continue
    s = s[["trade_date", "code", "research_score", "daily_rank"]].copy()
    s = s.rename(columns={"trade_date": "signal_date"})
    v002_frames.append(s)
v002_existing = pd.concat(v002_frames, ignore_index=True) if v002_frames else pd.DataFrame()
if not v002_existing.empty:
    v002_existing["signal_date"] = v002_existing["signal_date"].astype(str)
    v002_existing = v002_existing.drop_duplicates(["signal_date", "code"], keep="first").reset_index(drop=True)
print("v002 existing rows:", len(v002_existing),
      "dates:", sorted(v002_existing["signal_date"].unique()))

# ---------------- 4. 合并比较字段 ----------------
cand = cand.merge(v004a_existing[["signal_date", "code", "model_probability", "model_rank", "source"]],
                  on=["signal_date", "code"], how="left", suffixes=("", "_x"))
cand = cand.merge(recon_may[["signal_date", "code", "model_probability", "model_rank"]],
                  on=["signal_date", "code"], how="left", suffixes=("", "_recon"))
# 组装 v004a 字段: 现有评分优先, 缺口用冻结重建
has_existing = cand["model_probability"].notna()
has_recon = cand["model_probability_recon"].notna()
cand["v004a_probability"] = cand["model_probability"].fillna(cand["model_probability_recon"])
cand["v004a_rank"] = cand["model_rank"].fillna(cand["model_rank_recon"])
cand["v004a_score_source"] = np.where(has_existing, cand["source"],
                              np.where(has_recon, "frozen_reconstruction", ""))
cand["is_v004a_scorable"] = cand["v004a_probability"].notna()
cand["is_v004a_top3"] = cand["is_v004a_scorable"] & (cand["v004a_rank"] <= 3)
cand["is_v004a_top10"] = cand["is_v004a_scorable"] & (cand["v004a_rank"] <= 10)
cand["is_v004a_top15"] = cand["is_v004a_scorable"] & (cand["v004a_rank"] <= 15)
cand = cand.merge(v002_existing[["signal_date", "code", "daily_rank"]],
                  on=["signal_date", "code"], how="left", suffixes=("", "_v2"))
cand["v002_rank"] = cand["daily_rank"]
cand["is_v002_scorable"] = cand["v002_rank"].notna()
cand["is_v002_top3"] = cand["is_v002_scorable"] & (cand["v002_rank"] <= 3)
cand["is_v002_top10"] = cand["is_v002_scorable"] & (cand["v002_rank"] <= 10)
cand["is_v002_top15"] = cand["is_v002_scorable"] & (cand["v002_rank"] <= 15)

print("\nv004a scorable in candidates:", int(cand["is_v004a_scorable"].sum()),
      "| v002 scorable:", int(cand["is_v002_scorable"].sum()))
print("v004a_score_source:", cand["v004a_score_source"].value_counts(dropna=False).to_dict())

# 覆盖缺口记录
missing_v004a = cand[~cand["is_v004a_scorable"]]
if not missing_v004a.empty:
    print("WARN v004a missing dates:", sorted(missing_v004a["signal_date"].unique()))
missing_v002 = cand[~cand["is_v002_scorable"]]
print("v002 missing dates:", sorted(missing_v002["signal_date"].unique()))

# ---------------- 5. 标签交叉核对 (与项目 history candidates) ----------------
hc_labels = pd.read_csv(HC_0529, dtype={"code": str})
hc_labels["code"] = norm_code(hc_labels["code"])
hc_labels["signal_date"] = hc_labels["signal_date"].astype(str)
cmp_lab = cand.merge(
    hc_labels[["signal_date", "code", "d2_open_price", "d3_high_price", "d3_close_price",
               "d2open_d3high_return_pct", "target7_d2open_d3high"]],
    on=["signal_date", "code"], how="inner", suffixes=("_mine", "_proj"))
if not cmp_lab.empty:
    diff_high = (pd.to_numeric(cmp_lab["d2open_to_d3high_return"], errors="coerce")
                 - pd.to_numeric(cmp_lab["d2open_d3high_return_pct"], errors="coerce") / 100.0).abs()
    diff_high = diff_high[diff_high.notna()]
    print(f"\nlabel cross-check vs project history candidates: overlap={len(cmp_lab)} "
          f"high_return max_abs_diff={diff_high.max() if len(diff_high) else None:.6f}")
    t7_mine = cmp_lab["target7_d2open_d3high_mine"].fillna(False).astype(bool)
    t7_proj = cmp_lab["target7_d2open_d3high_proj"].fillna(False).astype(bool)
    print(f"target7 disagreement count: {(t7_mine != t7_proj).sum()} / {len(cmp_lab)}")
    cmp_lab.to_csv(SCRATCH / "label_crosscheck.csv", index=False, encoding="utf-8-sig")

# ---------------- 6. 最终主样本输出 ----------------
main_cols = [
    # A 身份和日期
    "event_id", "code", "name", "signal_date", "break_date",
    "board_streak_before_break", "days_since_break", "days_since_last_limit_up", "repair_attempt_count",
    # B 连板和辨识度代理
    "recent_limit_up_count_10d", "recent_limit_up_count_20d",
    "recent_pool_appearance_count_10d", "recent_pool_appearance_count_20d",
    "max_board_streak_20d", "board_day_amount_rank", "board_day_turnover_rank",
    "board_day_volume_rank", "recognition_score",
    # C 断板日 OHLC 和结构
    "break_open", "break_high", "break_low", "break_close", "break_prev_close",
    "break_open_return", "break_high_return", "break_close_return", "break_intraday_range",
    "break_high_to_close_drawdown", "break_upper_shadow_ratio", "break_lower_shadow_ratio",
    "break_close_location", "break_volume", "break_amount",
    "break_volume_ratio_vs_board_days", "break_amount_ratio_vs_board_days", "break_turnover_ratio",
    "break_touched_limit_up", "break_opened_from_limit_up",
    # D D1 日线与均线位置
    "d1_open", "d1_high", "d1_low", "d1_close", "d1_volume", "d1_amount",
    "d1_ma5", "d1_ma10", "d1_ma20",
    "d1_close_to_ma5", "d1_low_to_ma5", "d1_high_to_ma5",
    "d1_close_to_ma10", "d1_low_to_ma10", "d1_close_to_ma20",
    "d1_ma5_slope", "d1_ma10_slope", "d1_reclaimed_ma5", "d1_reclaimed_ma10",
    "consecutive_days_below_ma5", "consecutive_days_below_ma10",
    # E D1 修复质量
    "d1_open_to_close_return", "d1_low_to_close_recovery", "d1_high_to_close_drawdown",
    "d1_close_location", "d1_vwap", "d1_close_to_vwap", "d1_intraday_range",
    "d1_afternoon_return", "d1_last_hour_return", "d1_up_bar_volume_ratio", "d1_down_bar_volume_ratio",
    # F 对手盘和成交压力代理
    "volume_above_d1_close_ratio", "amount_above_d1_close_ratio",
    "volume_above_break_close_ratio",
    "high_zone_volume_ratio", "high_zone_amount_ratio",
    "late_day_sell_volume_ratio", "late_day_sell_amount_ratio",
    "down_bar_volume_ratio", "d1_vwap_to_close_gap",
    # G 与现有模型的比较字段
    "is_v004a_scorable", "v004a_probability", "v004a_rank",
    "is_v004a_top3", "is_v004a_top10", "is_v004a_top15",
    "is_v002_scorable", "v002_rank",
    "is_v002_top3", "is_v002_top10", "is_v002_top15",
    # H 标签字段
    "d2_trade_date", "d3_trade_date",
    "d2_open", "d2_high", "d2_low", "d2_close",
    "d3_high", "d3_low", "d3_close",
    "d2open_to_d3high_return", "d2open_to_d3close_return",
    "target7_d2open_d3high", "tail_loss_5pct", "future_data_status",
]
extra_audit_cols = ["last_board_date", "break_day_in_pool", "last_board_day_in_pool",
                    "pool_consecutive_count_last_board", "v004a_score_source"]
missing_cols = [c for c in main_cols + extra_audit_cols if c not in cand.columns]
if missing_cols:
    print("WARN missing columns:", missing_cols)
main = cand[[c for c in main_cols + extra_audit_cols if c in cand.columns]].copy()
main = main.sort_values(["signal_date", "code", "days_since_break"]).reset_index(drop=True)
# code 保字符串前导零
main["code"] = main["code"].astype(str).str.zfill(6)
main["event_id"] = main["event_id"].astype(str)
out_path = OUT / "v004c_break_repair_dataset.csv"
main.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\nmain dataset written:", out_path, "rows:", len(main))

# ---------------- 7. 审计表 ----------------
obs = pd.read_csv(SCRATCH / "stage2_observations.csv", dtype={"code": str})
obs["code"] = norm_code(obs["code"])
obs["signal_date"] = obs["signal_date"].astype(str)
obs = obs.rename(columns={"exclusion_reason": "exclusion_reason_raw",
                          "data_quality_reason": "data_quality_reason_raw"})
obs["name"] = obs.apply(lambda r: _resolve_name(r["code"], r["break_date"]), axis=1)
audit = obs[["event_id", "code", "name", "signal_date", "break_date",
             "board_streak_before_break", "days_since_break",
             "candidate_status", "exclusion_reason_raw", "data_quality_reason_raw"]].copy()
audit = audit.rename(columns={"exclusion_reason_raw": "exclusion_reason",
                              "data_quality_reason_raw": "data_quality_reason"})
# QUALITY_FAILED 行的 exclusion_reason 保留明细; CANDIDATE/LABEL_UNAVAILABLE 补充原因
audit.loc[audit["candidate_status"] == "CANDIDATE", "exclusion_reason"] = ""
audit.loc[audit["candidate_status"] == "CANDIDATE", "data_quality_reason"] = ""
audit.loc[audit["candidate_status"] == "LABEL_UNAVAILABLE", "exclusion_reason"] = "missing_future_label"
audit["signal_date"] = audit["signal_date"].replace("", pd.NA)
audit = audit.sort_values(["signal_date", "code", "days_since_break"]).reset_index(drop=True)
audit_path = OUT / "v004c_candidate_audit.csv"
audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
print("audit written:", audit_path, "rows:", len(audit))
print("status counts:", audit["candidate_status"].value_counts().to_dict())

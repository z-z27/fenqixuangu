# -*- coding: utf-8 -*-
"""Probe: can we rebuild v004a rank features from history candidates to match grid_v2?
Read-only. Writes nothing outside the v004c output dir."""
import sys
sys.path.insert(0, r"F:\fenqixuangu")
import pandas as pd
import numpy as np

BASE = r"F:\fenqixuangu\reports\history_samples"
GRID = r"F:\fenqixuangu\reports\v004a\grid_v2_scored\v004a_scored_candidates.csv"

hc = pd.read_csv(f"{BASE}\\2026-05-06_2026-06-29\\history_candidates_2026-05-06_2026-06-29.csv")
grid = pd.read_csv(GRID)
grid["code"] = grid["code"].astype(str).str.zfill(6)
# Frozen policy config: l2=0.30, positive_weight=1.5
grid = grid[(grid["l2"].astype(float) == 0.3) & (grid["positive_weight"].astype(float) == 1.5)].copy()
grid = grid.drop_duplicates(["signal_date", "code"]).reset_index(drop=True)
print("grid after frozen filter:", len(grid), "dates:", grid["signal_date"].nunique())
print("hc cols:", [c for c in ["signal_date","code","eligible_for_trade","d2open_d3high_return_pct","d2open_d3close_return_pct","candidate_base_price","d1_close_ma10_pct","d1_low_ma10_pct","trend_hold_score","total_score","theme_score","days_since_d0","active_money_score","d1_close_vwap_pct","graph_quality_score"] if c in hc.columns])
print("hc dates:", hc["signal_date"].nunique(), hc["signal_date"].min(), "->", hc["signal_date"].max())
print("grid dates:", grid["signal_date"].nunique(), grid["signal_date"].min(), "->", grid["signal_date"].max())

# Rebuild scorable pool from hc using the project's eligibility logic
hc["code"] = hc["code"].astype(str).str.zfill(6)
hc["eligible_for_trade"] = hc["eligible_for_trade"].fillna(False).astype(bool)
hc["d2open_d3high_return_pct"] = pd.to_numeric(hc["d2open_d3high_return_pct"], errors="coerce")
hc["d2open_d3close_return_pct"] = pd.to_numeric(hc["d2open_d3close_return_pct"], errors="coerce")
hc["candidate_base_price"] = pd.to_numeric(hc["candidate_base_price"], errors="coerce")
scorable = hc[
    hc["eligible_for_trade"]
    & hc["d2open_d3high_return_pct"].notna()
    & hc["d2open_d3close_return_pct"].notna()
    & hc["candidate_base_price"].notna()
    & (hc["candidate_base_price"] > 0)
].copy()
print("hc scorable rows:", len(scorable), "grid rows:", len(grid))

# Match counts per date
g = grid.groupby("signal_date")["code"].nunique()
s = scorable.groupby("signal_date")["code"].nunique()
cmp = pd.DataFrame({"grid": g, "hc_scorable": s}).fillna(0).astype(int)
print(cmp.to_string())
print("dates where counts differ:", (cmp["grid"] != cmp["hc_scorable"]).sum())

# Compare the 5 base rank feature columns on an overlap date
base_specs = ["d1_close_ma10_pct","d1_low_ma10_pct","trend_hold_score","total_score","theme_score"]
for date in sorted(g.index.intersection(s.index))[:3]:
    sub = scorable[scorable["signal_date"] == date].copy()
    gr = grid[grid["signal_date"] == date].copy()
    for col in base_specs:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    m = sub.merge(gr[["code"] + [f"rank_{c}" for c in base_specs] + ["model_rank"]], on="code", how="inner")
    print(f"--- {date} matched codes: {len(m)} (grid {len(gr)}, hc {len(sub)})")
    if m.empty:
        continue
    for c in base_specs:
        rk = m["rank_" + c].fillna(0.5)
        recomputed = sub.groupby("signal_date")[c].transform(lambda v: pd.to_numeric(v, errors="coerce").rank(pct=True, method="average")).astype(float).fillna(0.5)
        # recomputed is aligned on sub; map to m via code
        rr = sub.assign(_rk=recomputed)[["code","_rk"]].merge(m[["code"]], on="code")
        diff = float(np.abs(rr["_rk"].values - rk.values).max())
        print(f"  {c}: max rank diff vs grid = {diff:.6f}")

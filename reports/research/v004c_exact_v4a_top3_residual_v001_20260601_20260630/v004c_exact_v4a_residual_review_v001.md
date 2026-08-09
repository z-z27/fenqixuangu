# Did Exact v4a Top3 False Positives Differ From Rank4–10 Missed Winners?

| Feature | FP Median | MW Median | MW-FP | Cliff Delta | Date Direction % | Bootstrap 95% |
| --- | --- | --- | --- | --- | --- | --- |
| d1_high_to_close_drawdown_raw | 0.044880 | 0.063030 | 0.018150 | 0.3196 | 73.3333% | [0.001121, 0.042351] |
| d1_close_location | 0.505952 | 0.293233 | -0.212719 | -0.3182 | 73.3333% | [-0.304403, -0.031484] |
| d1_intraday_range | 0.096352 | 0.099398 | 0.003046 | 0.1276 | 53.3333% | [-0.009887, 0.031095] |
| d1_low_to_close_recovery | 0.046240 | 0.036429 | -0.009812 | -0.2258 | 66.6667% | [-0.023610, 0.000982] |
| d1_open_to_close_return_raw | 0.014905 | -0.027397 | -0.042302 | -0.3284 | 73.3333% | [-0.059192, 0.004030] |
| d1_last_hour_return | 0.003681 | -0.001112 | -0.004792 | -0.1224 | 66.6667% | [-0.015842, 0.010606] |
| late_day_sell_volume_ratio | 0.059264 | 0.068676 | 0.009412 | 0.0953 | 80.0000% | [-0.013343, 0.022356] |
| late_day_sell_amount_ratio | 0.059745 | 0.068034 | 0.008289 | 0.0953 | 73.3333% | [-0.013110, 0.022241] |

Cliff's delta is MW relative to FP. It is an effect size, not prediction proof or a causality test.

# Is There Evidence of More Unfinished Repair Among Missed Winners?

## Damage / Disagreement

- Directional conclusion: **MIXED**
- Gate-consistent fields: **d1_high_to_close_drawdown_raw**
- Weak expected-direction fields: **d1_high_to_close_drawdown_raw, d1_intraday_range, late_day_sell_volume_ratio, late_day_sell_amount_ratio**

## Repair Completion

- Directional conclusion: **YES**
- Gate-consistent fields: **d1_close_location, d1_low_to_close_recovery, d1_open_to_close_return_raw**
- Weak expected-direction fields: **d1_close_location, d1_low_to_close_recovery, d1_open_to_close_return_raw, d1_last_hour_return**

REPAIR_ROOM_RESIDUAL_EVIDENCE: **MIXED**

# Are the Missed Winners Already Near the Top3 Boundary?

- MW rank distribution: Rank4=10, Rank5=4, Rank6=4, Rank7=6, Rank8=2, Rank9=2, Rank10=3
- Score gap to Rank3 mean: **0.018404**
- Score gap median: **0.015804**
- Score gap p25/p75: **0.006852 / 0.024274**
- Score gap min/max: **0.000103 / 0.060620**

# How Much Could a Perfect Top10 Reranker Recover?

- Actual V4A Top3: **1.8539%**
- Top10 hindsight oracle Rank1: **6.9721%**
- Top10 hindsight oracle Top3: **6.0627%**
- Top10 oracle Top1 Target7 / Top3 precision: **90.4762% / 63.3333%**
- Full-universe oracle Top3: **6.1177%**
- Top10 improvement: **4.2088%**
- Recovery ratio: **98.7104%**
- Oracle calculations are **HINDSIGHT_DIAGNOSTIC_ONLY**.

# Is Top10 Recall Sufficient for a Second-Stage Reranker?

- Target7 recall@3/@5/@7/@10: **30.5085% / 54.2373% / 71.1864% / 83.0508%**
- Date-equal winner capture Top3/Top5/Top10: **47.3684% / 69.4737% / 90.7769%**
- Dates with replacement opportunity: **15 / 21**
- Total possible replacements: **21**
- TOP10_RERANK_FEASIBILITY: **YES**

## Group and Source Gates

- OOF rows / dates: **173 / 21**
- Join matched / unmatched / duplicate / mismatch: **173 / 0 / 0 / 0**
- FP: **44 rows / 21 dates**
- MW: **31 rows / 15 dates**
- TP: **18 rows / 15 dates**
- RN: **57 rows / 17 dates**
- Dates containing both FP and MW: **15**
- Exact frozen parity: **PASS**

## Outcome Severity and Context Audits

- FP near miss / weak / loss / severe counts: **8 / 15 / 21 / 5**
- FP raw return mean / median / p25 / p75 / worst: **-0.1173% / 0.4929% / -2.3474% / 4.5238% / -24.4558%**
- FP capped return mean / median / p25 / p75 / worst: **-0.1173% / 0.4929% / -2.3474% / 4.5238% / -24.4558%**
- MW 7–10% / 10–12% / >=12% counts: **14 / 7 / 10**
- FP board2 / board3: **32 (72.7273%) / 12 (27.2727%)**
- MW board2 / board3: **26 (83.8710%) / 5 (16.1290%)**
- FP group-date candidate count mean / median / p25 / p75 / min / max: **8.24 / 7.00 / 6.00 / 10.00 / 2 / 17**
- MW group-date candidate count mean / median / p25 / p75 / min / max: **9.73 / 8.00 / 7.00 / 13.00 / 5 / 17**

## Formal Decision

- Q1 exact frozen OOF used: **YES**
- Q2 Top3 false positives: **44 rows / 21 dates**
- Q3 Rank4–10 missed winners: **31 rows / 15 dates**
- Q4 more D1 damage/disagreement: **MIXED**
- Q5 FP more fully repaired by D1 close: **YES**
- Q6 pattern stable across dates: **MIXED**
- Q9 future Top10→Top3 experiment justified: **YES**
- Q10 unfinished-repair hypothesis: **MIXED**

REPAIR_ROOM_RESIDUAL_EVIDENCE: **MIXED**

TOP10_RERANK_FEASIBILITY: **YES**

- New model trained: **NO**
- Feature selection: **NO**
- New features: **NO**
- Score modification: **NO**
- July accessed: **NO**

Top10 contains factual reranking room, but the current repair-room explanation is not supported strongly enough to use these eight fields automatically.

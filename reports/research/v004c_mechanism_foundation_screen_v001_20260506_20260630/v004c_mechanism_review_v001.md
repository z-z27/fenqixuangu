# v004c Mechanism Information Foundation Screen v001

Status: `CAPABILITY_SCREEN_ONLY` — PRE-MODEL ONLY; no production artifact was trained.

## 1. Did We Find Any Basic Ranking Ability?

| Representation | Stage | Rank1 | Baseline | Excess | Top3 | Baseline | Excess | Winner Capture | Repair Concordance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CS_RANK | M0 | 2.811% | 2.999% | -0.187% | 2.341% | 2.784% | -0.442% | 56.000% | 0.5086 |
| CS_RANK | M1 | 2.811% | 2.999% | -0.187% | 2.308% | 2.784% | -0.476% | 53.333% | 0.5083 |
| RAW | M0 | 2.321% | 2.999% | -0.678% | 2.961% | 2.784% | 0.178% | 59.333% | 0.5052 |
| RAW | M1 | 2.321% | 2.999% | -0.678% | 2.961% | 2.784% | 0.178% | 59.333% | 0.5032 |

M2 was not run because a historical D1-safe theme/group membership source was unavailable.

## 2. Did Same-Tier Competition Add Information?

- RAW: NO — ΔRank1 0.000%, ΔTop3 0.000%, Δwinner capture 0.000%, Δconcordance -0.0020.
- CS_RANK: NO — ΔRank1 0.000%, ΔTop3 -0.033%, Δwinner capture -2.667%, Δconcordance -0.0003.

Answer: **NO**.

## 3. Did Theme / Group Position Add Information?

NOT TESTED — HISTORICAL D1-SAFE SOURCE UNAVAILABLE

## 4. Raw vs Cross-Sectional Rank Coordinates

Better supported: **MIXED**. This is based only on the fixed RAW/CS_RANK A/B.

## 5. May vs June

| Representation | Stage | May Rank1 Excess | May Top3 Excess | June Rank1 Excess | June Top3 Excess | Concordance | Temporal |
|---|---:|---:|---:|---:|---:|---:|---|
| CS_RANK | M0 | -0.704% | -0.692% | 0.009% | -0.355% | 0.5086 | WEAK |
| CS_RANK | M1 | -0.704% | -1.013% | 0.009% | -0.288% | 0.5083 | WEAK |
| RAW | M0 | -1.649% | 0.753% | -0.308% | -0.024% | 0.5052 | TEMPORALLY_MIXED |
| RAW | M1 | -1.649% | 0.753% | -0.308% | -0.024% | 0.5032 | TEMPORALLY_MIXED |

## 6. Permutation Sanity

Strongest fixed candidate: `RAW M1`. Real Rank1 excess -0.678%, Top3 excess 0.178%, concordance 0.5032. Permuted-target pipeline: Rank1 excess 0.283%, Top3 excess 0.360%, concordance 0.5024.
Clearly better than permutation: **NO**.

## 7. Data Sources and Leakage Boundary

Same-tier cohorts are reconstructed from complete frozen formal D0 pools at the exact streak, with D1 advancement from the complete D1 pool and D1 close returns from canonical daily caches. The group block is unavailable because no point-in-time membership snapshot exists. The two new path descriptors use only each event's frozen 48 BaoStock bars through D1 close. No July signal event was accessed; outcomes are isolated until after the data-collection gate.

## Final Questions

- Q1 Existing low-DF D1 blocks: **NO**
- Q2 Same-tier incremental value: **NO**
- Q3 Theme/group incremental value: **NOT_TESTED**
- Q4 Best coordinate: **MIXED**
- Q5 Stable across May and June: **NO**
- Q6 Clearly beats permutation: **NO**
- Q7 FOUNDATIONAL_SIGNAL: **NOT_ESTABLISHED**

The newly collected D1 mechanism information has not yet demonstrated sufficient stable OOF ranking ability to justify formal model development.

No feature was selected from outcome performance; no block, sign, interaction, lambda grid, or model family was changed after the screen began. July OOT and Tail were not run.

DATA_COLLECTION_GATE: **PASS**

Deterministic rebuild: **PASS** (the CLI builds all ten outputs twice and writes only if every byte matches).

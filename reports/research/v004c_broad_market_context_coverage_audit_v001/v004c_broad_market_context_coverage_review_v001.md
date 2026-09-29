# v004c Broad Market Context Coverage Audit v001

This is a blind data-coverage audit. No security outcome, model score, or model ranking was opened.

## Q1. 当前 stock source 原始层到底有哪些字段？

Eastmoney spot currently requests: latest_price, pct_chg, change, volume, amount, amplitude, turnover_rate, code, name, high, low, open, prev_close, total_market_cap, float_market_cap, industry.

## Q2. 哪些字段现在被 normalize 丢掉了？

Preserved raw fields: code, name, total_market_cap, float_market_cap, industry.
Dropped raw fields: latest_price, pct_chg, change, volume, amount, amplitude, turnover_rate, high, low, open, prev_close.

## Q3. May-August 能不能可靠重建全主板每日横截面？

M1-M3 can be reconstructed on dates meeting the 95% return-coverage gate, but the result uses an August current-universe snapshot as a historical approximation. Date-level membership, historical ST/name state, delistings, and complete listing-date history are unavailable. This is a lineage risk, not proof of exact historical membership.

- 2026-05: 18/18 M1-M3 full-quality dates; median coverage 97.88%.
- 2026-06: 21/21 M1-M3 full-quality dates; median coverage 97.80%.
- 2026-07: 23/23 M1-M3 full-quality dates; median coverage 97.90%.
- 2026-08: 13/21 M1-M3 full-quality dates; median coverage 97.84%.

## Q4. 4个固定 context 指标各自有多少有效日期？

M1=75, M2=75, M3=75, M4=0. M4 is not reconstructed because historical float market cap is absent.

## Q5. 历史 universe 是否存在明显偏差？

YES. Only one current universe snapshot exists; listing metadata and suspension snapshots are sparse, and historical ST/name membership cannot be recovered exactly.

## Q6. 历史 liquidity 能不能重建？

HISTORICAL_LIQUIDITY_AVAILABLE = NO.

## Q7. index source 当前有没有？

INDEX_SOURCE_AVAILABLE = NO.

## Q8. 正式 historical limit-down pool 有没有？

HISTORICAL_LIMIT_DOWN_POOL = NOT_AVAILABLE.

## Q9. 下一步是否授权做4变量 information audit？

NO under the exact full-main-board lineage contract. Resolve historical universe membership first; do not interpret the reconstructed values against outcomes yet.

BROAD_MARKET_CONTEXT_DATA_STATE = BROAD_CONTEXT_DATA_LINEAGE_RISK

LABEL_OR_OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

NEW_FEATURE_SEARCH = NO

NEXT_ACTION = FIX_DATA_LINEAGE

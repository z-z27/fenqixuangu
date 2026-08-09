# v004c Mechanism Source Audit v001

## Same-tier

- source: frozen formal full-main-board pools (`daily_limitup_derived`; five cached `akshare_zt_pool_em` dates) plus canonical daily cache
- historical reconstructable: YES
- D1 safe: YES; latest input is peer D1 close / D1 limit-up state
- coverage: 319/319 events
- status: AVAILABLE
- recovery note: a BaoStock 000567 fallback was cached with timestamp/SHA but not used because a higher-priority repository daily_unadjusted source covered D0/D1

## Theme/group

- source: none accepted; current universe `industry` metadata was rejected
- historical membership: NO point-in-time D1-effective snapshot found
- D1 safe: NOT ESTABLISHED
- coverage: 0/319 events (explicitly unavailable)
- status: UNAVAILABLE

## 5m path

- source: frozen BaoStock 5m cache (`adjustment=none`, `interval=5m`)
- coverage: 319/319 events with exactly 48 D1 bars per event
- D1 safe: YES; only bars through D1 15:00 are sliced

## Leakage boundary

- No signal event after 2026-06-30 was read.
- D2/D3 prices, raw repair, capped repair, Target7, and repair-rank target are OUTCOME_ONLY.
- Mechanism X is completed and the data-collection gate passes before outcomes are loaded.
- Theme/group is omitted, not imputed or guessed.

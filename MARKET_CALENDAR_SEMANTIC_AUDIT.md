# MARKET CALENDAR SEMANTIC AUDIT

Phase 0 code audit, 2026-09-30. No historical dataset, coefficient, OOF, backtest, or Board diagnostic was regenerated or overwritten.

## Scope and count

The unit counted below is a **logical date decision site** in production and frozen research code: date arithmetic, market-date enumeration, trading horizon selection, or a consumer whose D-number changes when the selected dates change. Pure date parsing, formatting, equality, as-of ordering, and column-name references are excluded. Repository search covered `src/`, `tools/`, `reports/` Python files, and `tests/` for `.days`, `weekday`, `bdate_range`, `date_range`, `Timedelta(days=`, date subtraction, D0/D1/D2/D3, `lookback`, `horizon`, `holding`, listing and suspension paths, and their call sites. There are **42 decision sites** in this scope: **20 NATURAL_DAY_CORRECT**, **19 TRADING_SESSION_REQUIRED**, **3 NEEDS_VERIFICATION**. This is not a count of every textual date reference in the repository.

## Confirmed bugs and fixes

| ID | File / function | Old behavior | Intended semantic | Action |
|---|---|---|---|---|
| T01 | `src/signal_engine.py` / `generate_signal` | Timestamp subtraction `.days` for `days_since_d0` | Sessions from D0, same day = 0 | Uses `TradingCalendar.session_distance`; threshold `> 3` unchanged. |
| T02 | `src/signal_engine.py` / `_count_consecutive_boards` | Calendar gap `<= 2` split Friday and Monday | Adjacent exchange sessions | Uses `are_consecutive_sessions`. |
| T03 | `src/backtester.py` / `run_full_history_backtest`, `_iter_weekdays` | Enumerated weekdays as signal dates | Actual exchange sessions | Uses `_iter_trading_days` backed by `trading_days`. |
| T04 | `src/history_samples.py` / `run_history_sample_generation` | Reused weekday signal-date enumeration | Actual exchange sessions | Uses the service's calendar; an all-closed range raises. |
| T05 | `src/cli.py` / `warmup_limitups` | Iterated `date_range` and excluded only weekends | Actual sessions to warm | Uses the service's calendar. |
| T06 | `src/backtester.py` / `_latest_possible_market_date` | Rolled weekends backward; holidays survived | Last exchange session on or before today | Uses `is_trading_day` / `previous_trading_day`. |
| T07 | `src/backtester.py` / `_listing_history_exclusion_decision`, `maximum_possible_weekday_trade_days` | Weekday upper bound labelled as possible trade-day count | Exact exchange session count in listing interval | Uses calendar `trading_days`; proof method version changed. |
| T08 | `src/backtester.py` / `_future_trade_dates` | Took the next dates present in one stock's minute bars; a missing D2 advanced D3 into D2 | Market sessions even if stock bars are absent | Returns calendar sessions through the last observed date; rejects off-calendar minute dates. |
| T09 | `src/backtester.py` / `_next_trade_date` | Took next observed stock minute date | Next exchange session | Uses `next_trading_day`; an absent D2 bar stays absent. |
| T10 | `src/history_samples.py` / `_collect_limitups_for_history_sample` | Weekend shortcut and daily-scan text could classify a trading date as closed | Calendar alone decides whether a date is a session | Queries calendar for each natural-window date; conflicting source errors remain unresolved. |
| T11 | `src/history_samples.py` / `_cached_daily_proves_non_trading` | Independently inferred closure from a stock-cache absence | Calendar-owned decision | Compatibility helper delegates to the calendar; production lookback no longer calls it. |
| T12 | `src/v004d_auction_data_feasibility_audit.py` / `_validate_live_window` | Weekday test admitted exchange holidays | Real trading session and clock window | Uses `is_trading_day`. |
| T13 | `src/v005_fixed_grid_holdout.py` / `_load_history_universe_context` | Validated every manifest against weekday enumeration | Corrected manifests must contain actual sessions | Uses calendar for `trading_sessions_v1` manifests; explicit legacy branch retains old manifest validation. |
| T14 | `src/backtester.py` / `_path_metrics_by_horizon` | Sliced T08's observed-stock dates | D2/D3/D5/D10 market-session horizon | Consumes the corrected calendar-session sequence; formulas unchanged. |
| T15 | `src/history_samples.py` / `_d2open_d3_metrics` | T08 could shift the D2 open/D3 high dates after a bar gap | Fixed market D2 and D3 | Consumes the corrected sequence; missing prices remain missing. |
| T16 | `src/backtester.py` / `_after_buy_rows` | Holding window used T08's stock-observed dates | Market-session holding horizon | Consumes the corrected sequence; price/execution rules unchanged. |
| T17 | `src/backtester.py` / `evaluate_top_signal` | Could execute on a later stock-observed date as D2 | Execute only on the next market session | Uses T09; absent D2 bars prevent execution. |
| T18 | `src/backtester.py` / `_exact_consecutive_boards`; `src/signal_engine.py` / `_count_consecutive_boards` | A five-natural-day pool could omit the pre-holiday board or the first non-board session, yielding an unproven count | Complete streak through the first prior non-board market session | Core signal builders fetch missing prior sessions once per date; direct counting requires explicit covered-session provenance and fails closed if the prior session is uncovered. |
| T19 | `src/backtester.py` / `_path_metrics_by_horizon`, `evaluate_history_signal`; `src/history_samples.py` / `evaluate_history_candidate_only`, `_finalise_targets`; `src/v004a.py` / `annotate_v004a_input_eligibility`, `prepare_v004a_samples` | Missing D2 minute bars could leave later cumulative returns, turn unknown targets into `False`, and admit them into training | A D2/D3 label needs the actual market sessions' minute coverage; an unknown label is not a negative | Require every session in a horizon window to have bars; missing D2/D3 is recorded as `label_evaluation_reason`, target values remain null, `candidate_evaluable=False`, execution evaluation requires the relevant early sessions, and null targets are excluded before v004a preparation. |

`TradingCalendar` now provides `is_trading_day`, inclusive `trading_days`, strict `session_distance`, `are_consecutive_sessions`, `previous_trading_day`, and `next_trading_day`. `session_distance` requires both endpoints to be sessions and `start <= end`; a reversed interval raises `ValueError`, while off-session endpoints or unavailable coverage raise `TradingCalendarError`. It never estimates by weekday. The process-shared calendar is injected from `MarketDataService` into normal signal generation, so a stock does not trigger its own calendar download.

The corrected historical sample path is `reports/history_samples/trading_sessions_v1/…`; its canonical snapshot root is `data/snapshots/trading_sessions_v1/…` (under the configured snapshot directory), and its manifest records `calendar_semantics_version`. Corrected CLI signal reports and full-history backtests use versioned subdirectories. Backtests that read an explicit signal file or directory use the corrected namespace only when the input path contains `trading_sessions_v1`; otherwise they write to `unverified_calendar_semantics`, avoiding a false corrected label and any overwrite of legacy output. This path check is a provenance guard, not proof of file contents; a later rebuild should add input manifest verification. Legacy artifacts remain in their existing locations. `collect_limit_ups` now rejects a partially failed trading-date window instead of publishing an incomplete pool.

## Natural-day uses intentionally preserved

| ID | File / function | Existing behavior | Intended semantic | Action |
|---|---|---|---|---|
| A01 | `src/data_sources.py` / `probe_daily_sources_for_date` | Query starts 14 calendar days earlier | API query buffer | Preserved. |
| A02 | `src/loaders.py` / `collect_limit_ups` | `lookback_days=N` spans N inclusive natural days | Natural-day candidate-pool window | Preserved; only dates processed within it are sessions. |
| A03 | `src/loaders.py` / `get_stock_bars` | Expands daily API query with a natural-day margin | Fetch buffer for a requested number of bars | Preserved. |
| A04 | `src/loaders.py` / `get_stock_bars` | Expands minute API query with a natural-day margin | Fetch buffer | Preserved. |
| A05 | `src/loaders.py` / minute-cache coverage fetch | Expands minute query with a natural-day margin | Fetch buffer | Preserved. |
| A06 | `src/loaders.py` / `_get_daily_for_limitup_scan` | Queries 120 prior calendar days | Fetch buffer | Preserved. |
| A07 | `src/backtester.py` / `_future_end_date` | Adds calendar days to request future bars | Overfetch buffer; actual horizon is T08/T14 | Preserved. |
| A08 | `src/trading_calendar.py` / `_fetch_baostock` | Queries 400 future calendar days | Calendar-source coverage buffer | Preserved. |
| A09 | `src/trading_calendar.py` / `previous_trading_day` | Expands 32 calendar days if the loaded calendar has no predecessor | Bounded source-coverage request, not session counting | Preserved as a fetch bound; fails closed if insufficient. |
| A10 | `src/trading_calendar.py` / `next_trading_day` | Same 32-day bound forward | Source-coverage request | Preserved; fails closed if insufficient. |
| A11 | `src/history_samples.py` / `_collect_limitups_for_history_sample` | Enumerates every date in `lookback_days` | Natural-day lookback and audit window | Preserved; T10 classifies each date. |
| A12 | `src/v004c_pair_capped7_july_forward.py` / `_load_five_day_pool` | Loads five calendar-day cache paths | Frozen experiment's source-pool window | Preserved as legacy study input. |
| A13 | `src/v004c_regime_aware_stage1_august_oot.py` / `_load_august_five_day_pool` | Same five-calendar-day source window | Frozen experiment input | Preserved. |
| A14 | `src/v004c_v4a_architecture_transfer.py` / `_load_five_calendar_day_pool` | Same five-calendar-day source window | Frozen experiment input | Preserved. |
| A15 | `src/v004c_frozen_stage1_contribution_ablation_audit.py` / `build_days_since_d0_parity` | Computes `.dt.days` to compare with frozen raw values | Explicit **legacy calendar-semantics parity audit** | Preserved; this value must never be treated as a corrected feature. |
| A16 | `src/v005_fixed_grid_holdout.py` / legacy manifest branch | Validates old manifests against their weekday enumeration contract | Read-only validation of legacy provenance | Preserved only when `calendar_semantics_version` is absent. |
| A17 | `src/legacy_calendar_semantics.py` / `_LegacyCalendarDayDistance` | Replays old natural-day D0 distance | Explicit legacy baseline only | Isolated from normal signal generation and called only by frozen V4C reconstruction. |
| A18 | `src/legacy_calendar_semantics.py` / `_legacy_consecutive_boards` | Replays old calendar-gap board count | Explicit legacy baseline only | Isolated from normal signal generation; no corrected artifact uses it. |
| A19 | `tools/build_v004c_pairwise_v1_feature_contract_v002.py` / `fetch_missing_daily_volume` | Fetches 30 natural days before a named session | Provider query buffer for one daily volume | Preserved. |
| A20 | `tools/recover_v004c_board3_historical_inputs_v001.py` / `_minute_window_by_code` | Fetches 90 natural days before the first signal | Provider query buffer for frozen minute context | Preserved. |

## [待核验] sites and remaining risks

| ID | File / function | Old/current behavior | Intended semantic | Phase 0.5 status |
|---|---|---|---|---|
| C01 | `src/trading_calendar.py` / `_derive_from_cached_daily` | Fifty local stock caches can infer a missing market date as closed; a correlated cache outage could imitate a holiday. | Reliable calendar fallback | **RESOLVED.** The cached-daily union is no longer a calendar source; it survives as positive OPEN evidence only. See Phase 0.5 §C01. |
| C02 | Frozen V4C reconstruction modules (`v004c_v4a_architecture_transfer.py`, `v004c_pair_capped7_july_forward.py`, `v004c_regime_aware_stage1_august_oot.py`) | Stored metrics were produced under old calendar semantics; an initial broad run observed frozen parity/metric failures. | Preserve exact legacy replay separately from corrected experiments | **PARTIALLY RESOLVED — gate FAILED.** The adapter is verified for 2 of 3 frozen modules; the third fails on a missing local 5m cache file, not on calendar semantics. See Phase 0.5 §C02/§Gate B. |
| C03 | `src/trading_calendar.py` / `_covers` | Coverage is inferred from first and last returned trading dates, not explicit exchange-calendar coverage metadata. | Fail closed at calendar boundaries | **RESOLVED.** Coverage is now declared per source and validated at fetch time. See Phase 0.5 §C03. |

## Tests and interpretation

New offline regression cases cover Friday→Monday/Tuesdays, same-session zero, single-day and long holidays, board streaks over weekends/holidays, a long-holiday board-history fetch, an absent D2 stock bar, missing-D2 label censoring and training exclusion with a complete-D2/D3 control, reversed/off-session and out-of-coverage errors, unavailable sources and corrupt calendar-cache metadata/payloads, propagation of calendar failures through both signal builders, backtest output provenance, natural-day `collect_limit_ups` windows, incomplete pool failure, and explicit legacy replay isolation. The focused calendar/listing/history/signal-date suite passed (**68 tests, 23 subtests**). The non-V4C formal suite passed (**157 tests, 39 subtests**), and V4D tests passed (**12 tests** across the two V4D test modules). The full `tests/` suite was attempted before the isolated legacy adapter was added and did **not** pass: frozen V4C parity and metric assertions failed or errored, and the broad run was stopped after frozen-research failures. The frozen suite has not been rerun after the adapter because it executes historical model reconstruction outside this phase's validation scope. Root-level unrestricted pytest also collects an executable research scratch file under `reports/`; formal validation uses `tests/`. *Superseded by Phase 0.5: the frozen suite has now been re-run as Gate B and the full suite passes (see Phase 0.5 §Test results).*

The corrected code changes candidate eligibility and historical signal-date enumeration. Consequently the future corrected cross-sectional universe, percentile ranks, Board2/Board3 membership, and training matrix can differ. No historical impact percentage has been extrapolated and no corrected historical dataset has been generated. Future labels based on complete, contiguous real minute sessions are not directly changed in date identity. Under the confirmed label policy, missing D2/D3 stock minute sessions make those labels unevaluable, leave target values null rather than `False`, and exclude the row from v004a training preparation; later observed bars cannot fill the gap.

**Rebuild gate: not yet open.** Verify the cached-daily calendar fallback in C01 and establish a passing corrected-semantics gate; C03 describes boundary-coverage limitations that must be understood for the requested rebuild range. Verify C02 before a strict legacy-versus-corrected comparison. Then create a corrected historical dataset in the versioned namespace; only after that should same-architecture V4C retraining begin. *Phase 0.5 (below) closed C01 and C03, established the corrected-dataset gate, and left legacy parity failing on a non-calendar cause.*

---

# PHASE 0.5 — MARKET CALENDAR SEMANTIC INTEGRITY PRE-REBUILD GATE

Phase 0.5, 2026-09-30. Closes the three `[待核验]` sites above and establishes **two strictly independent validation gates**. No model was trained, no corrected historical dataset was generated, and no legacy artifact — snapshot, coefficient, OOF, backtest, or report — was overwritten. Every code change is confined to calendar semantics and validation gates: no feature contract, target definition, threshold, ranking rule, Board policy, or Stage 1 / Stage 2 architecture was touched. The working-tree diff covers 3 source files and 2 test files, all of them calendar or offline-determinism related, and no hunk in it matches `threshold`, `target`, `feature`, `stage1`, `stage2`, `ranking`, `board_policy`, or `strategy`.

## C01 — cached-daily closure inference

**Old behavior.** `TradingCalendar._derive_from_cached_daily()` inferred market sessions from the union of dates present in up to 400 local per-stock daily caches, and that union was registered as a first-class calendar source (`cached_daily_cross_section`) in the production cascade. A date absent from every sampled cache was therefore answered **CLOSED**.

**Risk.** Absence of evidence was being read as evidence of absence. A cache gap, a failed fetch, a truncated history, or a provider outage is indistinguishable from a holiday at the data level. Because the daily caches are refreshed together, a *correlated* outage across ~50 files could fabricate a holiday — silently renumbering D0/D1/D2 session identities, shifting every rolling window, and changing `session_distance`, board streaks, and every downstream target, with no error surfaced anywhere.

**New contract — evidence direction.** The module docstring now states the asymmetry explicitly: `is_trading_day()` may return `False` **only** when a source that enumerates a whole natural-day interval has classified that day and the day is missing from its enumeration. Presence may prove OPEN; absence must not prove CLOSED.

- `TRADING_CALENDAR_SOURCES` is now exactly `(AKSHARE_SINA_SOURCE, BAOSTOCK_SOURCE)` — only sources capable of negative evidence.
- `CalendarFetch.coverage` must be source-declared or fetch-time-validated; `None` means "this result carries no closure authority at all".
- `TradingCalendar.closure_authority` exposes whether the adopted calendar can currently interpret non-membership as CLOSED.
- All sources unavailable ⇒ `TradingCalendarError`. **There is no `weekday()` fallback, no local-cache-absence fallback, and no guessing.**

**Code changes.**

- `src/trading_calendar.py` — `CACHED_DAILY_SOURCE` removed from `TRADING_CALENDAR_SOURCES`. The fallback was **not mechanically deleted**: `_derive_from_cached_daily()` is retained verbatim and re-exposed as `cached_daily_open_sessions(start, end)`, which yields positive OPEN evidence for diagnostics and cross-checking, gated by `CACHED_DAILY_MIN_SAMPLES = 50` so a thin sample cannot even assert OPEN. Referring to the old source name by name now raises `_refuse_cached_daily_as_calendar_source()`, an error that explains C01 rather than a bare "unknown source".

**Tests.** `tests/test_trading_calendar.py` covers the refusal path, the positive-evidence ceiling, and the minimum-sample gate. `tests/test_calendar_semantic_gate.py::GateACalendarAuthorityTest` (A1a–A1d) asserts that a cached-daily-only calendar cannot answer `False`, that an uncovered date raises instead of returning `False`, and that a fully unavailable cascade fails closed.

**Remaining limitations.** `cached_daily_open_sessions()` is a heuristic over a rolling cache and is **not** used by any production decision path; it is diagnostic only. It cannot detect a date that is a session but was never cached for any sampled stock.

## C03 — coverage versus returned sessions

**Coverage semantics.** `coverage` is redefined as *the natural-day interval within which absence means CLOSED*. It is **never** reconstructed from `dates[0]` / `dates[-1]`. `_covers()` now delegates to `_coverage_covers(self._coverage, start, end)` over that declared interval.

**Source boundary semantics.** Each source declares its own coverage, checked at fetch time rather than assumed from convention (the previous behavior extrapolated coverage from the returned list, so a known-CLOSED boundary — e.g. the last day of a year — was misread as a coverage failure and raised spuriously).

- `_akshare_sina_coverage()` — the Sina table is an exchange calendar **published by year**, i.e. it contains not-yet-occurred sessions. A table that lists future sessions necessarily enumerates whole years, so coverage runs to `12-31` of the last published year. Both premises (last row is in the future; last row's month is December) are validated on the spot; if either fails (a stale or truncated table) coverage falls back to the last row and later dates stay unprovable.
- `_baostock_coverage()` — `query_trade_dates` returns one row per **natural day** in the requested range, flagging `is_trading_day`. When that contract holds, the requested range *is* the coverage range. It is verified per response (row count equals the calendar-day count, first and last rows aligned); if verification fails, coverage falls back to the explicitly returned span.

**Failure behavior.** An out-of-coverage date raises `TradingCalendarError` — it is never judged CLOSED, and never backfilled by `weekday()`. Cache schema was bumped to `CALENDAR_CACHE_SCHEMA = 2`: a cache written before coverage was declared carries no provable coverage and is treated as a miss and re-fetched, rather than silently trusted.

**Remaining limitations.** Coverage is only as good as each source's own contract. The akshare coverage value is *inferred* from two validated structural premises rather than read from an explicit vendor field; a future change to the Sina table's shape would downgrade coverage to "last row only" (fail-closed, but an availability regression). The frozen-snapshot path (§ below) exists to remove this live dependency for the rebuild.

## C02 — legacy parity

`src/legacy_calendar_semantics.py` replays the **old** natural-day formulas and is imported by exactly the three frozen V4C reconstruction modules plus parity tests. Its job is to reproduce old behavior exactly — not to make old tests accept new behavior. No old metric was adjusted, no expected value updated, no golden file overwritten, no test deleted or weakened, no threshold relaxed.

Static isolation holds: `src/` importers are exactly `v004c_v4a_architecture_transfer.py`, `v004c_pair_capped7_july_forward.py`, `v004c_regime_aware_stage1_august_oot.py`. No corrected production module imports it. This invariant is asserted by the gate tool.

| Frozen module | Parity test | Result |
|---|---|---|
| `src/v004c_v4a_architecture_transfer.py` | `tests/test_v004c_original_v4a_direct_transfer.py` | **8 passed** |
| `src/v004c_pair_capped7_july_forward.py` | `tests/test_v004c_pair_capped7_july_forward.py` | **5 passed, 5 errors** |
| `src/v004c_regime_aware_stage1_august_oot.py` | `tests/test_v004c_regime_aware_stage1_august_oot.py` | **5 passed** |

**Difference and located cause.** All five errors are the same failure:

```
RuntimeError: FATAL: expected 48 canonical D1 bars for 000017@2026-07-01, got 0
  at src/v004c_pair_capped7_july_forward.py:268  (_read_minute_through)
```

`_read_minute_through` is a pure local-file read of `data/cache/minute_5m/<code>_5min.pkl`; it issues no calendar call, and `reconstruct_forward_raw_row` calls it **before** `generate_legacy_calendar_signal` (line 325). The legacy adapter is therefore not implicated in the failure.

The cause is a **cache-window rollover, and it is now pinned exactly.** `config.default_5min_days = 40`, so refreshing a stock's 5m cache keeps only the most recent ~40 sessions. `data/cache/minute_5m/000017_5min.pkl` spans exactly 2026-07-06 → 2026-09-01 — 40 sessions — and carries mtime 2026-09-01 16:48, i.e. it was last refreshed on 2026-09-01, at which point 2026-07-01 aged out of the window. Meanwhile `data/cache/daily/000017_daily.pkl` still spans 2025-06-26 → 2026-09-01 and *does* contain 2026-07-01, because the daily cache uses `daily_history_days = 180`. The two caches have different retention, so the frozen experiment's D1 survived in one and was silently deleted from the other.

This is not a market-wide gap: **858 stocks do have the canonical 48 bars on 2026-07-01.** It is one stock's file, truncated by its own refresh schedule — the 5m cache is refreshed per stock on demand, so a stock refreshed later loses older sessions that a stock refreshed earlier retains. `git status --porcelain` on the three frozen modules and on `src/legacy_calendar_semantics.py` is empty — none of them changed. **This is a data-retention defect, not a calendar-semantics defect.** It is reported, not patched: restoring the input would require a live re-fetch that could itself alter frozen content, which is out of this phase's scope.

**Interpretation.** The legacy conclusions remain intact and are labelled here as `legacy_calendar_semantics evidence` — Stage 1 Top3 below the universe baseline, Board3 overpromotion, PAIR_CAPPED7 failure, and the Board3-only model result. They are neither deleted nor promoted, and Phase 0.5 did not re-run any of them. The one module whose input survived replays exactly.

## Gate A — CORRECTED_DATASET_GENERATION_GATE

Requirement coverage: calendar authority (A1), session semantics — `session_distance` / `previous_trading_day` / `next_trading_day` / `are_consecutive_sessions` / `trading_days` (A2), D0 age by market session (A3), board continuity across weekend / single holiday / long holiday (A4), historical signal-date enumeration that is not weekday≈trading-day (A5), future horizons D2/D3/D5/D10 by market session identity (A6), missing session ⇒ `target=null` and `candidate_evaluable=False` (A7), null targets excluded before v004a training (A8), and provenance carrying `calendar_semantics_version=trading_sessions_v1` into the new namespace (A9).

Live coverage verification for the intended rebuild range `2026-05-06 … 2026-07-31`: **62 sessions**, first `2026-05-06`, last `2026-07-31`, adopted from `akshare_sina_trade_dates` with declared coverage `[1990-12-19, 2026-12-31]`. October 2026 sessions resolve to `2026-10-08, 10-09, 10-12 … 10-16`, and `2026-09-25` resolves CLOSED — confirming the long-holiday fixture.

```
CORRECTED_DATASET_GENERATION_GATE = PASS
```

## Gate B — LEGACY_PARITY_GATE

```text
LEGACY_PARITY_GATE = FAIL
  isolation: OK — no unexpected src/ importer
  test_v004c_original_v4a_direct_transfer.py       8 passed          PASS
  test_v004c_pair_capped7_july_forward.py          5 passed, 5 errors FAIL
  test_v004c_regime_aware_stage1_august_oot.py     5 passed          PASS
```

Blocker: the frozen 5m input `000017@2026-07-01` is no longer on disk — it aged out of the 40-session 5m cache window (§C02). The gate is **not** reopened by relaxing the frozen module: the missing input is an external-state problem, and the correct remedy is to restore the frozen input and re-run the gate **unchanged**. Gate B cannot be made to pass by weakening an assertion.

## Test results

All core semantic tests are offline and deterministic (injected calendars, mocked providers, frozen local fixtures). Live coverage validation is a separate opt-in integration check (`--check-coverage`).

| Command | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|
| `python tools/run_calendar_semantic_gate_v001.py --check-coverage` (Gate A offline core, 6 modules) | 128 | 0 | 0 | 0 |
| `python -m pytest tests/test_v004c_original_v4a_direct_transfer.py -q` | 8 | 0 | 0 | 0 |
| `python -m pytest tests/test_v004c_pair_capped7_july_forward.py -q` | 5 | 0 | 5 | 0 |
| `python -m pytest tests/test_v004c_regime_aware_stage1_august_oot.py -q` | 5 | 0 | 0 | 0 |
| `python -m unittest discover -s tests -t .` (full suite, 264.855s) | 866 | 0 | 0 | 0 |

The full suite was run **with `data/cache/trade_calendar/` deleted**. The directory is recreated empty by `DataConfig.ensure_directories()`, but **no calendar file is written into it** — proving no unit test reaches the live calendar source. This is the offline-determinism invariant for the whole suite.

The gate tool was re-run from a clean state and reproduced its recorded result exactly (`CORRECTED_DATASET_GENERATION_GATE = PASS`, `LEGACY_PARITY_GATE = FAIL`), with the same per-module counts — the verdicts are stable, not a one-off run.

New regression cases required by this phase: Friday→Monday = 1; Thursday→Monday = 2; Wednesday→Monday = 3; Friday→Tuesday = 2; same session = 0; last session before a holiday → first after = 1; long holiday likewise = 1; Friday + Monday limit-up ⇒ board 2; Friday + Monday + Tuesday ⇒ board 3; long-holiday board continuity; and the future-label case — signal Friday, real D2 Monday, real D3 Tuesday, minute bars present Friday and Tuesday but missing Monday ⇒ D2 = Monday, D3 = Tuesday, Monday recorded missing, `candidate_evaluable=False`, `target=null`, and **Tuesday never becomes D2**.

## Frozen calendar snapshot

`data/calendars/trading_sessions_v1/verified_calendar_snapshot_v001.json` (SHA-256 `46725561116adac799f8d976d6b88c3912ae4f296a25260d88a3f9727473e4ae`, 8797 sessions, coverage `[1990-12-19, 2026-12-31]`, source `akshare_sina_trade_dates`) was written via `write_frozen_calendar_snapshot()`. Loaded offline with an empty fetcher map and a deliberately unreachable source list, it reproduces the same 62 rebuild-range sessions — so the corrected rebuild can bind to one immutable calendar artifact instead of "whatever AkShare happened to return today". **Recommended before the full rebuild.**

## Defects discovered during Phase 0.5

1. **A prior fixture encoded a wrong long-holiday calendar.** `_Q3_DATES` in `tests/test_trading_calendar.py` omitted `2026-10-08`, so a long-holiday regression was silently asserting the wrong successors. Corrected against the live authoritative calendar; this is exactly the class of error the fixture hardcoding was meant to prevent.
2. **Offline determinism was being violated.** `_latest_possible_market_date()` fell back to the process-global `get_trading_calendar()`, so two tests could trigger a real network fetch. `MarketDataService.trading_calendar` is now threaded through `_future_end_date()` at both call sites, and `tests/test_v005_fixture_regression.py` injects a fixture calendar. Unit tests must never depend on the network.
3. **`baostock` is unusable on the installed pandas.** `baostock`'s `resultset.py:146` calls `DataFrame.append`, removed in pandas ≥ 2.0 (installed: **pandas 3.0.2**). This triggers only on multi-page results, so the baostock calendar fallback fails for the full 1990→future range. `git show HEAD:src/trading_calendar.py` confirms the call site is unchanged, i.e. this predates Phase 0.5. **Consequence: effective authoritative-source redundancy is 1 (akshare only).** Not patched — third-party, out of scope.
4. **5m cache retention (40 sessions) is far shorter than daily cache retention (180 days), and the 5m cache is refreshed per stock.** All 62 authoritative sessions in the rebuild range do have 5m data for some stocks, but availability is uneven and *asymmetric across stocks*: a session carries ~272–284 cached stocks in mid-May versus ~1315 at end-July, and within a single session one stock may have the full 48 bars while another has none, depending on when each was last refreshed. The corrected pipeline censors these correctly (target `null`, `candidate_evaluable=False`, reason `missing_minute_session`), and censoring a *missing* session is correct behavior — but because the truncation is per-stock and time-dependent, it is **not reproducible**: re-running a rebuild at a different time can censor different rows. Any frozen experiment whose D1 is older than 40 sessions (in practice, anything before roughly `today − 8 weeks`) is **not replayable from cache** — this is what broke Gate B and it will break any attempt to re-derive a past corrected sample. It must be treated as a **precondition of the corrected rebuild**: the rebuild's minute inputs need to be frozen as artifacts, not read from a self-refreshing cache. This is tracked separately from calendar semantics.

`missing_minute_session` reasons now distinguish `provider_gap` (the session's daily bar exists ⇒ the stock was trading) from `suspended` (the daily cache spans the date on both sides but has no bar for it). Where neither can be shown the reason is left bare — byte-identical to its pre-Phase-0.5 form, so no downstream parser is forced to upgrade. **The target definition is unchanged.**

## Phase 0.5 verdict

```text
CORRECTED_DATASET_GENERATION_GATE = PASS
LEGACY_PARITY_GATE               = FAIL
```

Two independent gates, deliberately never mixed: Gate A validates that corrected calendar semantics are trustworthy and fully covered over the rebuild range; Gate B validates that frozen legacy results still replay. Gate B's failure is caused by a missing local cache file, so it does **not** invalidate Gate A, and Gate A's pass does **not** excuse Gate B.

**Legacy-versus-corrected comparison gate: NOT OPEN** (it is downstream of Gate B and cannot be evaluated while frozen replay is incomplete).

**Next phase: B — Corrected Historical Dataset Rebuild**, conditional on first restoring the frozen 5m input so Gate B can be re-run. Phase A is closed (C01 and C03 resolved; no remaining calendar-semantic blocker). Phase C (V4C retrain) is not yet permissible: it requires a corrected dataset that does not exist. Note that Gate B failing for a *data-retention* reason also signals an open risk for phase B itself — because the 5m cache retains only 40 sessions and refreshes per stock, minute inputs for `2026-05-06 … 2026-07-31` are already older than the window and are **not reproducible**. Phase B must therefore freeze its minute inputs as artifacts before generating the corrected dataset, or its censoring pattern will differ on every run.

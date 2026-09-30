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

| ID | File / function | Old/current behavior | Intended semantic | Action needed before rebuild |
|---|---|---|---|---|
| C01 | `src/trading_calendar.py` / `_derive_from_cached_daily` | Fifty local stock caches can infer a missing market date as closed; a correlated cache outage could imitate a holiday. | Reliable calendar fallback | Verify provenance/coverage of cached daily samples or require an authoritative source for negative closure claims. |
| C02 | Frozen V4C reconstruction modules (`v004c_v4a_architecture_transfer.py`, `v004c_pair_capped7_july_forward.py`, `v004c_regime_aware_stage1_august_oot.py`) | Stored metrics were produced under old calendar semantics; an initial broad run observed frozen parity/metric failures. | Preserve exact legacy replay separately from corrected experiments | Calls now use the explicit legacy adapter A17/A18. Full frozen parity remains unverified; do not reinterpret old conclusions. |
| C03 | `src/trading_calendar.py` / `_covers` | Coverage is inferred from first and last returned trading dates, not explicit exchange-calendar coverage metadata. | Fail closed at calendar boundaries | Confirm source coverage for boundary holidays; current code raises when coverage cannot be shown, which can affect availability. |

## Tests and interpretation

New offline regression cases cover Friday→Monday/Tuesdays, same-session zero, single-day and long holidays, board streaks over weekends/holidays, a long-holiday board-history fetch, an absent D2 stock bar, missing-D2 label censoring and training exclusion with a complete-D2/D3 control, reversed/off-session and out-of-coverage errors, unavailable sources and corrupt calendar-cache metadata/payloads, propagation of calendar failures through both signal builders, backtest output provenance, natural-day `collect_limit_ups` windows, incomplete pool failure, and explicit legacy replay isolation. The focused calendar/listing/history/signal-date suite passed (**68 tests, 23 subtests**). The non-V4C formal suite passed (**157 tests, 39 subtests**), and V4D tests passed (**12 tests** across the two V4D test modules). The full `tests/` suite was attempted before the isolated legacy adapter was added and did **not** pass: frozen V4C parity and metric assertions failed or errored, and the broad run was stopped after frozen-research failures. The frozen suite has not been rerun after the adapter because it executes historical model reconstruction outside this phase's validation scope. Root-level unrestricted pytest also collects an executable research scratch file under `reports/`; formal validation uses `tests/`.

The corrected code changes candidate eligibility and historical signal-date enumeration. Consequently the future corrected cross-sectional universe, percentile ranks, Board2/Board3 membership, and training matrix can differ. No historical impact percentage has been extrapolated and no corrected historical dataset has been generated. Future labels based on complete, contiguous real minute sessions are not directly changed in date identity. Under the confirmed label policy, missing D2/D3 stock minute sessions make those labels unevaluable, leave target values null rather than `False`, and exclude the row from v004a training preparation; later observed bars cannot fill the gap.

**Rebuild gate: not yet open.** Verify the cached-daily calendar fallback in C01 and establish a passing corrected-semantics gate; C03 describes boundary-coverage limitations that must be understood for the requested rebuild range. Verify C02 before a strict legacy-versus-corrected comparison. Then create a corrected historical dataset in the versioned namespace; only after that should same-architecture V4C retraining begin.

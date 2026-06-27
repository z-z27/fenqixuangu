# Research Sample Analysis Workflow

This workflow turns historical backtest rows into a factor research dataset.

## Goal

The goal is not to keep only good trades. The goal is to keep every useful case:

- `success`: D2 executed and the simple D3 sell model realized the target-min return.
- `failed`: D2 executed but D3 realized return did not reach the target-min return.
- `missed_selected`: selected for execution, D2 did not execute, but D3 had target-min opportunity.
- `missed_unselected`: not selected for execution, but D3 had target-min opportunity.
- `ordinary`: no D2 execution and no D3 target-min opportunity.
- `data_issue`: insufficient or inconsistent data.

The next strategy changes should come from comparing success samples against failed, missed, and ordinary samples.

## Usage

First run an existing historical backtest:

```powershell
python -m src.cli backtest-run --start-date 2026-01-01 --end-date 2026-06-30 --top-n 3 --hold-days 10
```

Then build research samples from the generated `history_trades` CSV:

```powershell
python -m src.research_analysis `
  --history-trades-file reports/backtest_runs/2026-01-01_2026-06-30/backtest_results/history_trades_2026-01-01_2026-06-30.csv `
  --target-min-return-pct 7 `
  --target-max-return-pct 10
```

The command writes:

```text
research_samples_<start>_<end>.csv
factor_compare_<start>_<end>.csv
research_review_<start>_<end>.md
```

## First D3 sell model

The first research sell model is intentionally simple:

```text
If D3 max return reaches target-min, record realized return as target-min.
Otherwise, use D3 close return.
```

This keeps the research layer stable and conservative enough for factor comparison. More complex D3 sell rules can be added later after the success / failed / missed comparison is working.

## How to use the output

Use `research_samples` to inspect individual stocks and sample groups.

Use `factor_compare` to ask questions such as:

- Which `support_type` bucket has the highest success rate?
- Does `days_since_d0 = 1` outperform later windows?
- Are high `support_score` samples actually better than high `active_money_score` samples?
- Are missed samples concentrated in a certain low-absorb-zone width?
- Are failed samples concentrated in a certain invalid-distance bucket?

Only promote a factor into strategy rules when it separates success samples from failed / missed / ordinary samples across enough historical rows.

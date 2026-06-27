# Return Distribution Research Workflow

This workflow turns historical backtest rows into an all-candidate factor research dataset.

The goal is not to optimize TopN, not to tune execution rules, and not to compare only `success` vs `failed` labels. The goal is to keep every useful candidate row and analyze profit/loss and opportunity distributions across the full sample universe.

## Research Goal

The current research phase should answer questions like:

- Which candidates later produced large D3 close gains?
- Which candidates produced large intraday D3 opportunities but weak closes?
- Which candidates had large losses or drawdowns?
- Which factors separate high-return rows from low-return rows?
- Do factor quantiles show monotonic return differences?
- Are TopN, selected, and executed fields explaining anything, or are they only metadata?

Old sample-group labels such as `success`, `failed`, `missed_selected`, `missed_unselected`, and `ordinary` are still preserved, but they are not the primary research target.

## Primary Research Targets

The return-distribution layer focuses on these columns:

```text
primary_d3_return_pct        = candidate_d3_close_return_pct
opportunity_d3_return_pct    = candidate_d3_max_return_pct
risk_d3_drawdown_pct         = candidate_d3_max_drawdown_pct
realized_d3_return_pct       = d3_realized_return_pct, only meaningful for executed rows
```

Use `primary_d3_return_pct` to study actual D3 close profit/loss across all candidates.

Use `opportunity_d3_return_pct` to study whether a candidate had a tradable-looking D3 upside path.

Use `risk_d3_drawdown_pct` to study downside path risk.

Use `realized_d3_return_pct` only as an execution-state diagnostic, not as the main all-sample label.

## Default Usage

Run full-candidate research without `--max-codes` for formal data collection:

```powershell
python -m src.watch_research `
  --start-date 2026-06-01 `
  --end-date 2026-06-25
```

This command now builds the default all-sample return-distribution outputs.

Do not add `--max-codes` for formal factor research. That parameter truncates daily candidates and biases the good/bad sample universe.

## Default Output Files

The default workflow writes these files under `reports/research_runs/<run_name>/research_results/`:

```text
research_samples_<start>_<end>.csv
return_samples_<start>_<end>.csv
return_bucket_compare_<start>_<end>.csv
factor_quantile_report_<start>_<end>.csv
profit_loss_compare_<start>_<end>.csv
daily_return_summary_<start>_<end>.csv
return_distribution_report_<start>_<end>.md
```

The most important outputs are:

- `return_samples`: enriched all-candidate sample table with return buckets.
- `return_bucket_compare`: compares D3 close-return, D3 opportunity, drawdown, and realized-return buckets.
- `factor_quantile_report`: checks whether factor quantiles produce monotonic return differences.
- `profit_loss_compare`: compares high-return and low-return groups directly.
- `daily_return_summary`: shows whether the dataset is dominated by one or two trading days.

## Optional Legacy Diagnostics

The old sample-group factor discovery report is now optional:

```powershell
python -m src.watch_research `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --with-factor-discovery
```

This generates:

```text
group_compare_<start>_<end>.csv
factor_discovery_<start>_<end>.csv
factor_discovery_<start>_<end>.md
```

Use these only as legacy diagnostics for old sample labels. Do not use them as the primary factor-research report.

## How to Read the Research Outputs

First inspect `return_distribution_report` and `profit_loss_compare`.

Look for:

- large positive D3 close-return groups;
- large negative D3 close-return groups;
- high D3 opportunity but weak D3 close groups;
- high drawdown groups;
- factor averages that separate profit and loss groups.

Then inspect `factor_quantile_report`.

For each factor, ask:

```text
Does higher factor value produce higher average return?
Does higher factor value produce higher D3 target rate?
Does higher factor value also increase loss or drawdown?
Is the relationship monotonic across quantiles?
```

A factor is not useful just because the top bucket looks good once. It becomes useful only if it separates profit/loss or opportunity/no-opportunity groups consistently across enough rows and enough trading days.

## Research Discipline

At this stage, do not directly modify strategy rules from one report.

The correct sequence is:

```text
collect full samples
→ analyze return distribution
→ analyze factor quantiles
→ check daily stability
→ identify candidate factors
→ only then modify factor weights or execution logic
```

TopN, selected, eligible, and executed columns are metadata. They can explain whether the current system captured an opportunity, but they should not define the primary good/bad labels during factor discovery.

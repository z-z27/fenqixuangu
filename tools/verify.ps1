$ErrorActionPreference = "Stop"

python -m compileall -q src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.run_daily_v005 --help | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.v005_fixed_grid_holdout --help | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.cli backtest-history --help | Out-Null
exit $LASTEXITCODE

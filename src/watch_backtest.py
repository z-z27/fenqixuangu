from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "backtest-run",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--top-n",
        str(args.top_n),
        "--hold-days",
        str(args.hold_days),
        "--target-return-pct",
        str(args.target_return_pct),
        "--stop-loss-pct",
        str(args.stop_loss_pct),
        "--lookback-days",
        str(args.lookback_days),
    ]
    if args.signal_days is not None:
        command.extend(["--signal-days", str(args.signal_days)])
    if args.eval_days is not None:
        command.extend(["--eval-days", str(args.eval_days)])
    if args.max_codes is not None:
        command.extend(["--max-codes", str(args.max_codes)])
    if args.force_refresh:
        command.append("--force-refresh")
    if args.include_all_allowed:
        command.append("--include-all-allowed")
    if args.include_small:
        command.append("--include-small")
    command.extend(["--entry-price-mode", args.entry_price_mode])

    run_root = Path("reports") / "backtest_runs" / f"{args.start_date}_{args.end_date}"
    print("[watch] starting:", " ".join(command), flush=True)
    print(f"[watch] run root: {run_root}", flush=True)
    process = subprocess.Popen(command)
    last_snapshot = ""
    started_at = time.time()
    try:
        while process.poll() is None:
            snapshot = build_snapshot(run_root, started_at)
            if snapshot != last_snapshot:
                print(snapshot, flush=True)
                last_snapshot = snapshot
            time.sleep(max(1, args.interval_seconds))
    except KeyboardInterrupt:
        print("[watch] interrupted; terminating child process...", flush=True)
        process.terminate()
        return process.wait()

    snapshot = build_snapshot(run_root, started_at)
    if snapshot != last_snapshot:
        print(snapshot, flush=True)
    print(f"[watch] child process exited with code {process.returncode}", flush=True)
    return int(process.returncode or 0)


def build_snapshot(run_root: Path, started_at: float) -> str:
    elapsed = int(time.time() - started_at)
    signals_dir = run_root / "daily_signals"
    quality_dir = run_root / "data_quality"
    result_dir = run_root / "backtest_results"
    signal_files = sorted(signals_dir.glob("signals_*.csv")) if signals_dir.exists() else []
    quality_files = sorted(quality_dir.glob("data_quality_*.csv")) if quality_dir.exists() else []
    result_files = sorted(result_dir.glob("*.csv")) if result_dir.exists() else []
    latest = latest_files(run_root, limit=5)
    lines = [
        f"[watch] elapsed={elapsed}s signal_files={len(signal_files)} quality_files={len(quality_files)} result_csv={len(result_files)}",
    ]
    if latest:
        lines.append("[watch] latest files:")
        lines.extend(f"  - {path}" for path in latest)
    return "\n".join(lines)


def latest_files(root: Path, limit: int = 5) -> list[str]:
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in files[:limit]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run backtest-run while printing file-level progress")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--target-return-pct", type=float, default=7.0)
    parser.add_argument("--stop-loss-pct", type=float, default=3.0)
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--signal-days", type=int, default=None)
    parser.add_argument("--eval-days", type=int, default=None)
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--include-all-allowed", action="store_true")
    parser.add_argument("--include-small", action="store_true")
    parser.add_argument("--entry-price-mode", choices=["zone_max", "confirmation_close"], default="zone_max")
    parser.add_argument("--interval-seconds", type=int, default=5)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

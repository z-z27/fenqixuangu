from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .research_analysis import (
    DEFAULT_TARGET_MAX_RETURN_PCT,
    DEFAULT_TARGET_MIN_RETURN_PCT,
    run_research_analysis,
)
from .factor_discovery import run_factor_discovery


DEFAULT_RESEARCH_TOP_N = 999


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    date_suffix = f"{args.start_date}_{args.end_date}"
    source_run_root = Path("reports") / "backtest_runs" / date_suffix
    output_root = Path(args.output_root)
    run_name = args.run_name or build_default_run_name(args)
    research_root = unique_run_dir(output_root / run_name)

    command = build_backtest_command(args)
    print("[research] mode: full candidate research", flush=True)
    print("[research] starting backtest:", " ".join(command), flush=True)
    print(f"[research] temporary backtest root: {source_run_root}", flush=True)
    print(f"[research] final research root: {research_root}", flush=True)

    completed = subprocess.run(command)
    if completed.returncode != 0:
        print(f"[research] backtest failed with code {completed.returncode}", flush=True)
        return int(completed.returncode)

    if not source_run_root.exists():
        print(f"[research] missing backtest output root: {source_run_root}", flush=True)
        return 1

    print("[research] copying backtest outputs into research run folder...", flush=True)
    shutil.copytree(source_run_root, research_root, dirs_exist_ok=True)

    history_trades_file = research_root / "backtest_results" / f"history_trades_{date_suffix}.csv"
    if not history_trades_file.exists():
        print(f"[research] missing history trades file: {history_trades_file}", flush=True)
        return 1

    research_results_dir = research_root / "research_results"

    print("[research] building success/failed/missed samples...", flush=True)
    samples, factor_compare, samples_csv, factor_csv, markdown_path = run_research_analysis(
        history_trades_file=history_trades_file,
        output_dir=research_results_dir,
        target_min_return_pct=args.target_min_return_pct,
        target_max_return_pct=args.target_max_return_pct,
    )

    print("[research] running factor discovery...", flush=True)
    fd_group, fd_discovery, fd_group_csv, fd_discovery_csv, fd_md = run_factor_discovery(
        samples_file=samples_csv,
        output_dir=research_results_dir,
    )

    write_manifest(
        path=research_root / "research_manifest.md",
        args=args,
        command=command,
        source_run_root=source_run_root,
        research_root=research_root,
        history_trades_file=history_trades_file,
        samples_csv=samples_csv,
        factor_csv=factor_csv,
        markdown_path=markdown_path,
        sample_count=len(samples),
        factor_count=len(factor_compare),
    )

    print("[research] done", flush=True)
    print(f"research root: {research_root}", flush=True)
    print(f"history trades: {history_trades_file}", flush=True)
    print(f"samples csv: {samples_csv}", flush=True)
    print(f"factor csv: {factor_csv}", flush=True)
    print(f"factor discovery csv: {fd_discovery_csv}", flush=True)
    print(f"factor discovery md: {fd_md}", flush=True)
    print(f"manifest: {research_root / 'research_manifest.md'}", flush=True)
    print(f"samples: {len(samples)}", flush=True)
    print(f"factor rows: {len(factor_compare)}", flush=True)
    print(f"discovery rows: {len(fd_discovery)}", flush=True)
    return 0


def build_backtest_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.watch_backtest",
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
        "--interval-seconds",
        str(args.interval_seconds),
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
    return command


def build_default_run_name(args: argparse.Namespace) -> str:
    candidate_part = "all" if args.max_codes is None else f"max{args.max_codes}"
    return f"{args.start_date}_{args.end_date}_research_top{args.top_n}_{candidate_part}"


def unique_run_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.name}_v{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate unique research run directory under {path.parent}")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    command: list[str],
    source_run_root: Path,
    research_root: Path,
    history_trades_file: Path,
    samples_csv: Path,
    factor_csv: Path,
    markdown_path: Path,
    sample_count: int,
    factor_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Research Run Manifest",
        "",
        "## Purpose",
        "",
        "This run is for factor discovery, not final TopN live-trading simulation.",
        "It keeps success, failed, missed, ordinary, and data-issue samples for later comparison.",
        "",
        "## Parameters",
        "",
        f"- start date: `{args.start_date}`",
        f"- end date: `{args.end_date}`",
        f"- top_n: `{args.top_n}`",
        f"- max_codes: `{args.max_codes if args.max_codes is not None else 'all'}`",
        f"- hold_days: `{args.hold_days}`",
        f"- target_return_pct: `{args.target_return_pct}`",
        f"- stop_loss_pct: `{args.stop_loss_pct}`",
        f"- target_min_return_pct: `{args.target_min_return_pct}`",
        f"- target_max_return_pct: `{args.target_max_return_pct}`",
        f"- entry_price_mode: `{args.entry_price_mode}`",
        "",
        "## Outputs",
        "",
        f"- samples: `{samples_csv}`",
        f"- factor compare: `{factor_csv}`",
        f"- factor discovery csv: `{fd_discovery_csv}`",
        f"- factor discovery md: `{fd_md}`",
        "",
        "## Counts",
        "",
        f"- samples: **{sample_count}**",
        f"- factor rows: **{factor_count}**",
        f"- discovery rows: **{len(fd_discovery)}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

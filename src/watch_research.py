from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .factor_discovery import run_factor_discovery
from .research_analysis import (
    DEFAULT_TARGET_MAX_RETURN_PCT,
    DEFAULT_TARGET_MIN_RETURN_PCT,
    run_research_analysis,
)


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

    print("[research] building factor discovery reports...", flush=True)
    group_compare, discovery, group_csv, discovery_csv, discovery_markdown = run_factor_discovery(
        samples_file=samples_csv,
        output_dir=research_results_dir,
        min_group_size=args.discovery_min_group_size,
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
    print(f"markdown: {markdown_path}", flush=True)
    print(f"group compare csv: {group_csv}", flush=True)
    print(f"factor discovery csv: {discovery_csv}", flush=True)
    print(f"factor discovery markdown: {discovery_markdown}", flush=True)
    print(f"manifest: {research_root / 'research_manifest.md'}", flush=True)
    print(f"samples: {len(samples)}", flush=True)
    print(f"factor rows: {len(factor_compare)}", flush=True)
    print(f"group rows: {len(group_compare)}", flush=True)
    print(f"discovery rows: {len(discovery)}", flush=True)
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
        "## Command",
        "",
        "```text",
        " ".join(command),
        "```",
        "",
        "## Output Files",
        "",
        f"- source backtest root: `{source_run_root}`",
        f"- research root: `{research_root}`",
        f"- history trades: `{history_trades_file}`",
        f"- research samples: `{samples_csv}`",
        f"- factor compare: `{factor_csv}`",
        f"- research review: `{markdown_path}`",
        "",
        "## Counts",
        "",
        f"- samples: **{sample_count}**",
        f"- factor rows: **{factor_count}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-candidate historical research and store outputs in reports/research_runs")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_RESEARCH_TOP_N, help="Research default selects nearly all ranked candidates")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--target-return-pct", type=float, default=7.0)
    parser.add_argument("--stop-loss-pct", type=float, default=3.0)
    parser.add_argument("--target-min-return-pct", type=float, default=DEFAULT_TARGET_MIN_RETURN_PCT)
    parser.add_argument("--target-max-return-pct", type=float, default=DEFAULT_TARGET_MAX_RETURN_PCT)
    parser.add_argument("--discovery-min-group-size", type=int, default=3)
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--signal-days", type=int, default=None)
    parser.add_argument("--eval-days", type=int, default=None)
    parser.add_argument("--max-codes", type=int, default=None, help="Optional speed cap; omit for full candidate research")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--include-all-allowed", action="store_true")
    parser.add_argument("--include-small", action="store_true")
    parser.add_argument("--entry-price-mode", choices=["zone_max", "confirmation_close"], default="zone_max")
    parser.add_argument("--interval-seconds", type=int, default=5)
    parser.add_argument("--output-root", default="reports/research_runs")
    parser.add_argument("--run-name", default=None, help="Optional folder name under output-root; auto-suffixed if it already exists")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

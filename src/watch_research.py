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
from .return_distribution import run_return_distribution_analysis


DEFAULT_RESEARCH_TOP_N = 999


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    date_suffix = f"{args.start_date}_{args.end_date}"
    source_run_root = Path("reports") / "backtest_runs" / date_suffix
    output_root = Path(args.output_root)
    run_name = args.run_name or build_default_run_name(args)
    research_root = unique_run_dir(output_root / run_name)

    command = build_backtest_command(args)
    print("[research] mode: full candidate return-distribution research", flush=True)
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
    print("[research] building full research samples...", flush=True)
    samples, factor_compare, samples_csv, factor_csv, markdown_path = run_research_analysis(
        history_trades_file=history_trades_file,
        output_dir=research_results_dir,
        target_min_return_pct=args.target_min_return_pct,
        target_max_return_pct=args.target_max_return_pct,
    )

    print("[research] building all-sample return distribution reports...", flush=True)
    (
        return_samples,
        bucket_compare,
        factor_quantiles,
        profit_loss_compare,
        return_samples_csv,
        bucket_csv,
        quantile_csv,
        profit_loss_csv,
        return_markdown_path,
    ) = run_return_distribution_analysis(
        samples_file=samples_csv,
        output_dir=research_results_dir,
        target_return_pct=args.target_return_pct,
        loss_cutoff_pct=args.return_loss_cutoff_pct,
        quantiles=args.return_quantiles,
    )
    daily_return_csv = research_results_dir / f"daily_return_summary_{date_suffix}.csv"

    discovery_outputs: tuple[object, ...] | None = None
    if args.with_factor_discovery:
        print("[research] building legacy sample-group factor discovery reports...", flush=True)
        discovery_outputs = run_factor_discovery(
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
        return_samples_csv=return_samples_csv,
        bucket_csv=bucket_csv,
        quantile_csv=quantile_csv,
        profit_loss_csv=profit_loss_csv,
        daily_return_csv=daily_return_csv,
        return_markdown_path=return_markdown_path,
        discovery_outputs=discovery_outputs,
        sample_count=len(samples),
        factor_count=len(factor_compare),
        return_bucket_count=len(bucket_compare),
        factor_quantile_count=len(factor_quantiles),
        profit_loss_group_count=len(profit_loss_compare),
    )

    print("[research] done", flush=True)
    print(f"research root: {research_root}", flush=True)
    print(f"history trades: {history_trades_file}", flush=True)
    print(f"research samples csv: {samples_csv}", flush=True)
    print(f"research review markdown: {markdown_path}", flush=True)
    print(f"return samples csv: {return_samples_csv}", flush=True)
    print(f"return bucket csv: {bucket_csv}", flush=True)
    print(f"factor quantile csv: {quantile_csv}", flush=True)
    print(f"profit/loss csv: {profit_loss_csv}", flush=True)
    print(f"daily return csv: {daily_return_csv}", flush=True)
    print(f"return distribution markdown: {return_markdown_path}", flush=True)
    if discovery_outputs is not None:
        _, discovery, group_csv, discovery_csv, discovery_markdown = discovery_outputs
        print(f"legacy group compare csv: {group_csv}", flush=True)
        print(f"legacy factor discovery csv: {discovery_csv}", flush=True)
        print(f"legacy factor discovery markdown: {discovery_markdown}", flush=True)
        print(f"legacy discovery rows: {len(discovery)}", flush=True)
    print(f"manifest: {research_root / 'research_manifest.md'}", flush=True)
    print(f"samples: {len(samples)}", flush=True)
    print(f"return samples: {len(return_samples)}", flush=True)
    print(f"return bucket rows: {len(bucket_compare)}", flush=True)
    print(f"factor quantile rows: {len(factor_quantiles)}", flush=True)
    print(f"profit/loss groups: {len(profit_loss_compare)}", flush=True)
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
    return_samples_csv: Path,
    bucket_csv: Path,
    quantile_csv: Path,
    profit_loss_csv: Path,
    daily_return_csv: Path,
    return_markdown_path: Path,
    discovery_outputs: tuple[object, ...] | None,
    sample_count: int,
    factor_count: int,
    return_bucket_count: int,
    factor_quantile_count: int,
    profit_loss_group_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Research Run Manifest",
        "",
        "## Purpose",
        "",
        "This run is for all-candidate return-distribution research, not final TopN live-trading simulation.",
        "It keeps all useful rows and treats realized trade state, TopN, and execution fields as metadata rather than primary labels.",
        "The main research target is profit/loss and opportunity distribution across the full sample universe.",
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
        f"- return_loss_cutoff_pct: `{args.return_loss_cutoff_pct}`",
        f"- return_quantiles: `{args.return_quantiles}`",
        f"- entry_price_mode: `{args.entry_price_mode}`",
        f"- with_factor_discovery: `{args.with_factor_discovery}`",
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
        f"- legacy factor compare: `{factor_csv}`",
        f"- legacy research review: `{markdown_path}`",
        f"- return samples: `{return_samples_csv}`",
        f"- return bucket compare: `{bucket_csv}`",
        f"- factor quantile report: `{quantile_csv}`",
        f"- profit/loss compare: `{profit_loss_csv}`",
        f"- daily return summary: `{daily_return_csv}`",
        f"- return distribution report: `{return_markdown_path}`",
        "",
    ]
    if discovery_outputs is not None:
        _, _, group_csv, discovery_csv, discovery_markdown = discovery_outputs
        lines.extend(
            [
                "## Optional Legacy Factor Discovery Outputs",
                "",
                f"- group compare: `{group_csv}`",
                f"- factor discovery csv: `{discovery_csv}`",
                f"- factor discovery markdown: `{discovery_markdown}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Counts",
            "",
            f"- samples: **{sample_count}**",
            f"- legacy factor rows: **{factor_count}**",
            f"- return bucket rows: **{return_bucket_count}**",
            f"- factor quantile rows: **{factor_quantile_count}**",
            f"- profit/loss groups: **{profit_loss_group_count}**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-candidate historical research and store return-distribution outputs")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_RESEARCH_TOP_N, help="Research default selects nearly all ranked candidates")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--target-return-pct", type=float, default=7.0)
    parser.add_argument("--stop-loss-pct", type=float, default=3.0)
    parser.add_argument("--target-min-return-pct", type=float, default=DEFAULT_TARGET_MIN_RETURN_PCT)
    parser.add_argument("--target-max-return-pct", type=float, default=DEFAULT_TARGET_MAX_RETURN_PCT)
    parser.add_argument("--return-loss-cutoff-pct", type=float, default=-3.0)
    parser.add_argument("--return-quantiles", type=int, default=5)
    parser.add_argument("--with-factor-discovery", action="store_true", help="Also build legacy sample-group discovery reports")
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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

NUMERIC_COLUMNS = [
    "daily_rank",
    "days_since_d0",
    "consecutive_boards",
    "total_score",
    "graph_quality_score",
    "support_score",
    "active_money_score",
    "theme_score",
    "low_absorb_width_pct",
    "invalid_distance_pct",
    "candidate_d3_max_return_pct",
    "candidate_d3_close_return_pct",
    "d3_realized_return_pct",
]

BOOL_COLUMNS = [
    "allowed_bool",
    "eligible_for_trade",
    "selected_by_topn",
    "selected_for_execution",
    "executed",
]


def run_strategy_experiments(
    samples_file: str | Path,
    output_dir: str | Path | None = None,
    target_return_pct: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path]:
    """Build strategy experiment proposals from research samples.

    This module does not change the live strategy. It converts research findings
    into explicit hypotheses that can be tested in the next backtest iteration.
    """
    source_path = Path(samples_file)
    samples = pd.read_csv(source_path, dtype={"code": str})
    if samples.empty:
        raise RuntimeError(f"research samples file is empty: {source_path}")

    frame = prepare_samples(samples)
    experiments = build_experiment_table(frame, target_return_pct=target_return_pct)
    watch_candidates = build_watch_upgrade_candidates(frame, target_return_pct=target_return_pct)
    entry_candidates = build_relaxed_entry_candidates(frame, target_return_pct=target_return_pct)

    out_dir = Path(output_dir) if output_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_samples_path(source_path)

    experiments_csv = out_dir / f"strategy_experiments_{suffix}.csv"
    watch_csv = out_dir / f"watch_upgrade_candidates_{suffix}.csv"
    entry_csv = out_dir / f"relaxed_entry_candidates_{suffix}.csv"
    markdown_path = out_dir / f"strategy_experiments_{suffix}.md"

    experiments.to_csv(experiments_csv, index=False, encoding="utf-8-sig")
    watch_candidates.to_csv(watch_csv, index=False, encoding="utf-8-sig")
    entry_candidates.to_csv(entry_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        build_markdown(
            frame,
            experiments,
            watch_candidates,
            entry_candidates,
            samples_file=source_path,
            target_return_pct=target_return_pct,
        ),
        encoding="utf-8",
    )
    return experiments, watch_candidates, entry_candidates, experiments_csv, watch_csv, entry_csv, markdown_path


def prepare_samples(samples: pd.DataFrame) -> pd.DataFrame:
    frame = samples.copy()
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOL_COLUMNS:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].astype(str).str.lower().isin({"true", "1", "yes"})
    for column in ["sample_group", "signal_type", "position_level", "support_type", "failure_reason", "name", "trade_date"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def build_experiment_table(samples: pd.DataFrame, target_return_pct: float) -> pd.DataFrame:
    rows = [
        watch_upgrade_experiment(samples, target_return_pct),
        relaxed_entry_experiment(samples, target_return_pct),
        active_money_penalty_experiment(samples),
        d0_watch_tracking_experiment(samples, target_return_pct),
    ]
    return pd.DataFrame(rows)


def watch_upgrade_experiment(samples: pd.DataFrame, target_return_pct: float) -> dict[str, Any]:
    watch = samples[samples["signal_type"] == "WATCH_ONLY"]
    watch_hit = watch[watch["sample_group"] == "missed_unselected"]
    watch_ordinary = watch[watch["sample_group"] == "ordinary"]
    candidate_rule = (
        "signal_type == WATCH_ONLY and graph_quality_score >= 80 "
        "and days_since_d0 <= 1 and total_score >= 60"
    )
    rule_hits = watch[
        (watch["graph_quality_score"] >= 80)
        & (watch["days_since_d0"] <= 1)
        & (watch["total_score"] >= 60)
    ]
    target_hits = int((rule_hits["candidate_d3_max_return_pct"] >= target_return_pct).sum())
    return {
        "experiment_id": "watch_upgrade_high_quality",
        "priority": 1,
        "target_problem": "WATCH_ONLY contains many D3 target opportunities but is not trade-eligible.",
        "hypothesis": "A subset of WATCH_ONLY with stronger graph quality and near-D0 timing should be promoted to small watch / candidate status.",
        "candidate_rule": candidate_rule,
        "supporting_rows": int(len(watch)),
        "positive_rows": int(len(watch_hit)),
        "baseline_rows": int(len(watch_ordinary)),
        "rule_rows": int(len(rule_hits)),
        "rule_target_hits": target_hits,
        "rule_target_hit_rate": _safe_rate(target_hits, len(rule_hits)),
        "expected_effect": "Reduce missed_unselected opportunities by promoting selected WATCH_ONLY candidates for D2 entry evaluation.",
        "main_risk": "May import noisy zero-position candidates; validate on more dates before changing live TopN.",
        "next_action": "Use watch_upgrade_candidates CSV to inspect names, then test a code-level WATCH_ONLY upgrade rule.",
    }


def relaxed_entry_experiment(samples: pd.DataFrame, target_return_pct: float) -> dict[str, Any]:
    selected_missed = samples[samples["sample_group"] == "missed_selected"]
    zone_low = selected_missed[selected_missed["failure_reason"] == "zone_too_low"]
    target_hits = int((zone_low["candidate_d3_max_return_pct"] >= target_return_pct).sum())
    return {
        "experiment_id": "relaxed_d2_entry_zone",
        "priority": 2,
        "target_problem": "Selected candidates are missed because the low-absorb entry zone is too conservative.",
        "hypothesis": "For selected zone_too_low cases, confirmation-close or a wider entry zone may convert missed_selected into executed samples.",
        "candidate_rule": "For D2_LOW_ABSORB selected candidates, test confirmation_close or a wider zone before invalidation.",
        "supporting_rows": int(len(selected_missed)),
        "positive_rows": int(len(zone_low)),
        "baseline_rows": int(len(samples[samples["sample_group"] == "success"])),
        "rule_rows": int(len(zone_low)),
        "rule_target_hits": target_hits,
        "rule_target_hit_rate": _safe_rate(target_hits, len(zone_low)),
        "expected_effect": "Increase D2 execution rate for already-selected candidates without changing candidate selection.",
        "main_risk": "Earlier entry may also increase failed executions; compare success vs failed after the experiment.",
        "next_action": "Run backtest with --entry-price-mode confirmation_close and compare factor_discovery outputs.",
    }


def active_money_penalty_experiment(samples: pd.DataFrame) -> dict[str, Any]:
    success = samples[samples["sample_group"] == "success"]
    failed = samples[samples["sample_group"] == "failed"]
    success_mean = _mean(success, "active_money_score")
    failed_mean = _mean(failed, "active_money_score")
    return {
        "experiment_id": "active_money_cap_or_penalty",
        "priority": 3,
        "target_problem": "High active_money_score may select volatile failed trades rather than stable winners.",
        "hypothesis": "active_money_score should be capped or combined with support/graph quality instead of blindly rewarded.",
        "candidate_rule": "Do not add standalone weight for active_money_score; require support_score >= 80 or graph_quality_score >= 80 when active_money_score is high.",
        "supporting_rows": int(len(success) + len(failed)),
        "positive_rows": int(len(success)),
        "baseline_rows": int(len(failed)),
        "rule_rows": 0,
        "rule_target_hits": 0,
        "rule_target_hit_rate": None,
        "expected_effect": "Reduce high-activity false positives in executed trades.",
        "main_risk": "Could filter real momentum names; validate with success_vs_failed on longer windows.",
        "next_action": f"Compare active_money_score means: success={_fmt(success_mean)}, failed={_fmt(failed_mean)}.",
    }


def d0_watch_tracking_experiment(samples: pd.DataFrame, target_return_pct: float) -> dict[str, Any]:
    d0 = samples[samples["days_since_d0"] == 0]
    d0_hits = int((d0["candidate_d3_max_return_pct"] >= target_return_pct).sum())
    return {
        "experiment_id": "d0_watch_tracking",
        "priority": 4,
        "target_problem": "D0 candidates often have later D3 opportunity but may not fit current D2 low-absorb timing.",
        "hypothesis": "D0 should be tracked separately instead of being judged by the same D2 entry mechanism immediately.",
        "candidate_rule": "Create a D0 tracking bucket; evaluate D1/D2 transition before promoting to D2_LOW_ABSORB.",
        "supporting_rows": int(len(d0)),
        "positive_rows": d0_hits,
        "baseline_rows": int(len(d0) - d0_hits),
        "rule_rows": int(len(d0)),
        "rule_target_hits": d0_hits,
        "rule_target_hit_rate": _safe_rate(d0_hits, len(d0)),
        "expected_effect": "Avoid discarding early strong candidates before they enter a low-absorb structure.",
        "main_risk": "Tracking bucket is research-only first; do not treat it as immediate buy signal.",
        "next_action": "Keep D0 samples separated in factor reports and compare D0 hit vs miss factors.",
    }


def build_watch_upgrade_candidates(samples: pd.DataFrame, target_return_pct: float) -> pd.DataFrame:
    watch = samples[samples["signal_type"] == "WATCH_ONLY"].copy()
    if watch.empty:
        return pd.DataFrame()
    watch["experiment_score"] = (
        watch["graph_quality_score"].fillna(0) * 0.35
        + watch["support_score"].fillna(0) * 0.25
        + watch["total_score"].fillna(0) * 0.25
        + watch["consecutive_boards"].fillna(0) * 5
        - watch["days_since_d0"].fillna(9) * 4
    )
    watch["target_hit_label"] = watch["candidate_d3_max_return_pct"] >= target_return_pct
    columns = existing_columns(
        watch,
        [
            "code",
            "name",
            "trade_date",
            "sample_group",
            "target_hit_label",
            "experiment_score",
            "candidate_d3_max_return_pct",
            "candidate_d3_close_return_pct",
            "days_since_d0",
            "total_score",
            "graph_quality_score",
            "support_score",
            "active_money_score",
            "theme_score",
            "consecutive_boards",
            "support_type",
            "position_level",
        ],
    )
    return watch.sort_values(["target_hit_label", "experiment_score"], ascending=[False, False])[columns].reset_index(drop=True)


def build_relaxed_entry_candidates(samples: pd.DataFrame, target_return_pct: float) -> pd.DataFrame:
    frame = samples[(samples["sample_group"] == "missed_selected") | (samples["failure_reason"] == "zone_too_low")].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["target_hit_label"] = frame["candidate_d3_max_return_pct"] >= target_return_pct
    frame["entry_relax_priority"] = (
        frame["candidate_d3_max_return_pct"].fillna(0)
        + frame["support_score"].fillna(0) * 0.05
        + frame["graph_quality_score"].fillna(0) * 0.03
        - frame["invalid_distance_pct"].fillna(0) * 0.5
    )
    columns = existing_columns(
        frame,
        [
            "code",
            "name",
            "trade_date",
            "sample_group",
            "failure_reason",
            "target_hit_label",
            "entry_relax_priority",
            "candidate_d3_max_return_pct",
            "candidate_d3_close_return_pct",
            "low_absorb_width_pct",
            "invalid_distance_pct",
            "days_since_d0",
            "total_score",
            "graph_quality_score",
            "support_score",
            "active_money_score",
            "support_type",
            "position_level",
        ],
    )
    return frame.sort_values(["target_hit_label", "entry_relax_priority"], ascending=[False, False])[columns].reset_index(drop=True)


def build_markdown(
    samples: pd.DataFrame,
    experiments: pd.DataFrame,
    watch_candidates: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    samples_file: Path,
    target_return_pct: float,
) -> str:
    lines = [
        "# Strategy Experiment Proposals",
        "",
        f"- source: `{samples_file}`",
        f"- records: **{len(samples)}**",
        f"- target return: **{target_return_pct:.2f}%**",
        "",
        "## Purpose",
        "",
        "These are research hypotheses, not live trading rules. Use them to decide what to test in the next code/backtest iteration.",
        "",
        "## Experiments",
        "",
        "| priority | experiment | problem | candidate rule | evidence | next action |",
        "|---:|---|---|---|---|---|",
    ]
    for _, row in experiments.sort_values("priority").iterrows():
        evidence = (
            f"positive={int(row.get('positive_rows') or 0)}, "
            f"rule_hits={int(row.get('rule_target_hits') or 0)}/{int(row.get('rule_rows') or 0)}"
        )
        lines.append(
            "| {priority} | {experiment} | {problem} | {rule} | {evidence} | {next_action} |".format(
                priority=int(row.get("priority") or 0),
                experiment=str(row.get("experiment_id") or ""),
                problem=str(row.get("target_problem") or "").replace("|", "/"),
                rule=str(row.get("candidate_rule") or "").replace("|", "/"),
                evidence=evidence,
                next_action=str(row.get("next_action") or "").replace("|", "/"),
            )
        )
    lines.append("")

    lines.extend(["## WATCH_ONLY Upgrade Candidates", ""])
    append_candidate_table(lines, watch_candidates.head(20), score_column="experiment_score")
    lines.append("")

    lines.extend(["## Relaxed Entry Candidates", ""])
    append_candidate_table(lines, entry_candidates.head(20), score_column="entry_relax_priority")
    lines.append("")

    lines.extend(
        [
            "## Recommended Test Order",
            "",
            "1. Test WATCH_ONLY upgrade on a wider date range, still in research mode.",
            "2. Test `--entry-price-mode confirmation_close` separately; do not combine with WATCH upgrade at first.",
            "3. Only after either experiment improves success/failed structure, run Top3/Top5 execution simulation.",
            "4. Do not promote any rule based only on this short date window.",
        ]
    )
    return "\n".join(lines)


def append_candidate_table(lines: list[str], frame: pd.DataFrame, score_column: str) -> None:
    if frame.empty:
        lines.append("No candidates.")
        return
    columns = ["code", "name", "trade_date", "sample_group", score_column, "candidate_d3_max_return_pct", "total_score", "graph_quality_score", "support_score", "active_money_score"]
    columns = existing_columns(frame, columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for _, row in frame.iterrows():
        values = [_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")


def existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _cell(value: Any) -> str:
    text = _fmt(value) if isinstance(value, float) else str(value)
    return text.replace("|", "/")


def _suffix_from_samples_path(path: Path) -> str:
    stem = path.stem
    prefix = "research_samples_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build strategy experiment proposals from research samples")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-return-pct", type=float, default=7.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiments, watch_candidates, entry_candidates, experiments_csv, watch_csv, entry_csv, markdown_path = run_strategy_experiments(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        target_return_pct=args.target_return_pct,
    )
    print(f"experiments: {len(experiments)}")
    print(f"watch candidates: {len(watch_candidates)}")
    print(f"entry candidates: {len(entry_candidates)}")
    print(f"experiments csv: {experiments_csv}")
    print(f"watch candidates csv: {watch_csv}")
    print(f"entry candidates csv: {entry_csv}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

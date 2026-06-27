from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


NUMERIC_DISCOVERY_FACTORS = [
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
]

SEGMENT_COLUMNS = [
    "signal_type",
    "position_level",
    "support_type",
    "allowed_bool",
    "eligible_for_trade",
    "selected_by_topn",
    "selected_for_execution",
    "executed",
    "failure_reason",
]

COMPARISONS = [
    ("success_vs_failed", "success", "failed", "Legacy executed-trade label comparison."),
    ("missed_selected_vs_success", "missed_selected", "success", "Legacy selected-but-not-executed comparison."),
    ("missed_unselected_vs_ordinary", "missed_unselected", "ordinary", "Legacy filtered-opportunity comparison."),
]


def run_factor_discovery(
    samples_file: str | Path,
    output_dir: str | Path | None = None,
    min_group_size: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    """Build optional legacy sample-group diagnostics.

    This module is intentionally secondary to return_distribution.py. It compares
    historical sample-group labels, but it should not be used as the primary
    factor-research layer and it does not emit strategy-change recommendations.
    """
    source_path = Path(samples_file)
    samples = pd.read_csv(source_path, dtype={"code": str})
    if samples.empty:
        raise RuntimeError(f"research samples file is empty: {source_path}")

    prepared = prepare_samples(samples)
    group_compare = build_group_compare(prepared)
    discovery = build_factor_discovery(prepared, min_group_size=min_group_size)

    out_dir = Path(output_dir) if output_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_samples_path(source_path)
    group_csv = out_dir / f"group_compare_{suffix}.csv"
    discovery_csv = out_dir / f"factor_discovery_{suffix}.csv"
    markdown_path = out_dir / f"factor_discovery_{suffix}.md"

    group_compare.to_csv(group_csv, index=False, encoding="utf-8-sig")
    discovery.to_csv(discovery_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        build_factor_discovery_markdown(
            prepared,
            group_compare,
            discovery,
            samples_file=source_path,
            min_group_size=min_group_size,
        ),
        encoding="utf-8",
    )
    return group_compare, discovery, group_csv, discovery_csv, markdown_path


def prepare_samples(samples: pd.DataFrame) -> pd.DataFrame:
    frame = samples.copy()
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in NUMERIC_DISCOVERY_FACTORS + [
        "candidate_d3_max_return_pct",
        "candidate_d3_close_return_pct",
        "d3_realized_return_pct",
    ]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in SEGMENT_COLUMNS + ["sample_group"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def build_group_compare(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in samples.groupby("sample_group", dropna=False):
        rows.append(group_summary_row(str(group_name), group))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def group_summary_row(group_name: str, group: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": group_name,
        "count": int(len(group)),
        "avg_candidate_d3_max_return_pct": _mean(group, "candidate_d3_max_return_pct"),
        "median_candidate_d3_max_return_pct": _median(group, "candidate_d3_max_return_pct"),
        "avg_candidate_d3_close_return_pct": _mean(group, "candidate_d3_close_return_pct"),
        "median_candidate_d3_close_return_pct": _median(group, "candidate_d3_close_return_pct"),
        "avg_d3_realized_return_pct": _mean(group, "d3_realized_return_pct"),
        "median_d3_realized_return_pct": _median(group, "d3_realized_return_pct"),
    }
    for factor in NUMERIC_DISCOVERY_FACTORS:
        row[f"avg_{factor}"] = _mean(group, factor)
        row[f"median_{factor}"] = _median(group, factor)
    return row


def build_factor_discovery(samples: pd.DataFrame, min_group_size: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison, positive_group, negative_group, description in COMPARISONS:
        positive = samples[samples["sample_group"] == positive_group]
        negative = samples[samples["sample_group"] == negative_group]
        rows.extend(
            numeric_comparison_rows(
                comparison,
                description,
                positive_group,
                negative_group,
                positive,
                negative,
                min_group_size=min_group_size,
            )
        )
        rows.extend(
            segment_comparison_rows(
                comparison,
                description,
                positive_group,
                negative_group,
                positive,
                negative,
                min_group_size=min_group_size,
            )
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["priority", "comparison", "evidence_score"], ascending=[True, True, False]).reset_index(drop=True)


def numeric_comparison_rows(
    comparison: str,
    description: str,
    positive_group_name: str,
    negative_group_name: str,
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    min_group_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in NUMERIC_DISCOVERY_FACTORS:
        pos_values = pd.to_numeric(positive[factor], errors="coerce").dropna() if factor in positive else pd.Series(dtype=float)
        neg_values = pd.to_numeric(negative[factor], errors="coerce").dropna() if factor in negative else pd.Series(dtype=float)
        pos_count = int(len(pos_values))
        neg_count = int(len(neg_values))
        pos_mean = _series_mean(pos_values)
        neg_mean = _series_mean(neg_values)
        diff = None if pos_mean is None or neg_mean is None else pos_mean - neg_mean
        pooled_std = _pooled_std(pos_values, neg_values)
        effect = None if diff is None or pooled_std in (None, 0) else diff / pooled_std
        enough = pos_count >= min_group_size and neg_count >= min_group_size
        direction = numeric_direction(diff, effect, enough)
        rows.append(
            {
                "comparison": comparison,
                "comparison_description": description,
                "factor_type": "numeric",
                "factor": factor,
                "bucket_or_value": "",
                "positive_group": positive_group_name,
                "negative_group": negative_group_name,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "difference": diff,
                "effect_size": effect,
                "positive_rate": None,
                "negative_rate": None,
                "lift": None,
                "evidence_score": abs(effect) if effect is not None else 0.0,
                "priority": priority_from_direction(direction, enough),
                "direction": direction,
                "interpretation": numeric_interpretation(factor, diff, effect, enough),
            }
        )
    return rows


def segment_comparison_rows(
    comparison: str,
    description: str,
    positive_group_name: str,
    negative_group_name: str,
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    min_group_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combined = pd.concat([positive, negative], ignore_index=True)
    for factor in SEGMENT_COLUMNS:
        if factor not in combined.columns:
            continue
        values = sorted(str(value) for value in combined[factor].fillna("").unique())
        for value in values:
            pos_total = int(len(positive))
            neg_total = int(len(negative))
            pos_count = int((positive[factor].fillna("").astype(str) == value).sum()) if pos_total else 0
            neg_count = int((negative[factor].fillna("").astype(str) == value).sum()) if neg_total else 0
            if pos_count == 0 and neg_count == 0:
                continue
            pos_rate = _safe_rate(pos_count, pos_total)
            neg_rate = _safe_rate(neg_count, neg_total)
            lift = None if pos_rate is None or neg_rate in (None, 0) else pos_rate / neg_rate
            diff = None if pos_rate is None or neg_rate is None else pos_rate - neg_rate
            enough = pos_total >= min_group_size and neg_total >= min_group_size and (pos_count + neg_count) >= min_group_size
            direction = segment_direction(diff, enough)
            rows.append(
                {
                    "comparison": comparison,
                    "comparison_description": description,
                    "factor_type": "segment",
                    "factor": factor,
                    "bucket_or_value": value,
                    "positive_group": positive_group_name,
                    "negative_group": negative_group_name,
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "positive_mean": None,
                    "negative_mean": None,
                    "difference": diff,
                    "effect_size": None,
                    "positive_rate": pos_rate,
                    "negative_rate": neg_rate,
                    "lift": lift,
                    "evidence_score": abs(diff or 0.0),
                    "priority": priority_from_direction(direction, enough),
                    "direction": direction,
                    "interpretation": segment_interpretation(factor, value, diff, lift, enough),
                }
            )
    return rows


def build_factor_discovery_markdown(
    samples: pd.DataFrame,
    group_compare: pd.DataFrame,
    discovery: pd.DataFrame,
    samples_file: Path,
    min_group_size: int,
) -> str:
    lines: list[str] = [
        "# Legacy Sample-Group Factor Discovery Report",
        "",
        f"- source: `{samples_file}`",
        f"- records: **{len(samples)}**",
        f"- min comparison group size: **{min_group_size}**",
        "- status: **optional legacy diagnostic, not the primary factor research report**",
        "",
        "## Scope",
        "",
        "This report compares old sample-group labels such as success, failed, missed, and ordinary.",
        "It is useful for execution-state diagnostics, but the default research workflow now uses return-distribution analysis across all samples.",
        "Do not read this report as a strategy recommendation layer.",
        "",
        "## Sample Group Counts",
        "",
    ]
    counts = samples["sample_group"].fillna("unknown").value_counts()
    for group, count in counts.items():
        lines.append(f"- {group}: **{int(count)}**")
    lines.append("")

    if not group_compare.empty:
        lines.extend(["## Group Compare Snapshot", ""])
        view = group_compare.head(12)
        lines.extend(
            [
                "| group | count | avg total | avg support | avg active money | avg D3 max | avg D3 close | avg realized D3 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in view.iterrows():
            lines.append(
                "| {group} | {count} | {total} | {support} | {active} | {d3max} | {d3close} | {realized} |".format(
                    group=row.get("group", ""),
                    count=int(row.get("count") or 0),
                    total=_fmt(row.get("avg_total_score")),
                    support=_fmt(row.get("avg_support_score")),
                    active=_fmt(row.get("avg_active_money_score")),
                    d3max=_fmt(row.get("avg_candidate_d3_max_return_pct")),
                    d3close=_fmt(row.get("avg_candidate_d3_close_return_pct")),
                    realized=_fmt(row.get("avg_d3_realized_return_pct")),
                )
            )
        lines.append("")

    if not discovery.empty:
        lines.extend(["## Label-Based Findings", ""])
        top = discovery[discovery["direction"] != "insufficient_sample"].head(30)
        if top.empty:
            lines.append("No findings have enough sample size yet.")
        else:
            lines.extend(
                [
                    "| comparison | factor | value | direction | evidence | interpretation |",
                    "|---|---|---|---|---:|---|",
                ]
            )
            for _, row in top.iterrows():
                lines.append(
                    "| {comparison} | {factor} | {value} | {direction} | {evidence} | {interpretation} |".format(
                        comparison=row.get("comparison", ""),
                        factor=row.get("factor", ""),
                        value=row.get("bucket_or_value", ""),
                        direction=row.get("direction", ""),
                        evidence=_fmt(row.get("evidence_score")),
                        interpretation=str(row.get("interpretation", "")).replace("|", "/"),
                    )
                )
        lines.append("")

    lines.extend(
        [
            "## Research Notes",
            "",
            "- Use this only as an auxiliary diagnostic for old sample-group labels.",
            "- The primary research question should be profit/loss and opportunity distribution across all candidates.",
            "- TopN, execution, and selection fields are metadata here, not the main target variable.",
            "- Do not change factor weights from this report alone; validate with return-distribution and cross-date stability analysis.",
        ]
    )
    return "\n".join(lines)


def numeric_direction(diff: float | None, effect: float | None, enough: bool) -> str:
    if not enough:
        return "insufficient_sample"
    if diff is None or effect is None:
        return "no_signal"
    if abs(effect) < 0.2:
        return "weak_or_no_separation"
    if effect > 0:
        return "positive_group_higher"
    return "negative_group_higher"


def segment_direction(diff: float | None, enough: bool) -> str:
    if not enough:
        return "insufficient_sample"
    if diff is None:
        return "no_signal"
    if abs(diff) < 0.10:
        return "weak_or_no_separation"
    if diff > 0:
        return "positive_group_overrepresented"
    return "negative_group_overrepresented"


def numeric_interpretation(factor: str, diff: float | None, effect: float | None, enough: bool) -> str:
    if not enough:
        return f"Not enough samples to judge `{factor}` yet."
    if diff is None or effect is None:
        return f"No usable numeric comparison for `{factor}`."
    if abs(effect) < 0.2:
        return f"`{factor}` has weak separation in this label comparison."
    side = "positive label group" if diff > 0 else "negative label group"
    return f"`{factor}` is higher in the {side}; treat as an auxiliary hypothesis only."


def segment_interpretation(factor: str, value: str, diff: float | None, lift: float | None, enough: bool) -> str:
    label = f"`{factor}={value}`"
    if not enough:
        return f"Not enough samples to judge {label}."
    if diff is None:
        return f"No usable segment comparison for {label}."
    if abs(diff) < 0.10:
        return f"{label} has weak segment separation."
    side = "positive label group" if diff > 0 else "negative label group"
    lift_text = "" if lift is None else f" lift={lift:.2f}."
    return f"{label} is overrepresented in the {side}.{lift_text}"


def priority_from_direction(direction: str, enough: bool) -> int:
    if not enough or direction == "insufficient_sample":
        return 9
    if direction in {"positive_group_higher", "positive_group_overrepresented", "negative_group_higher", "negative_group_overrepresented"}:
        return 1
    if direction == "weak_or_no_separation":
        return 5
    return 7


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _series_mean(values: pd.Series) -> float | None:
    return None if values.empty else float(values.mean())


def _pooled_std(left: pd.Series, right: pd.Series) -> float | None:
    if left.empty or right.empty:
        return None
    return float(pd.concat([left, right], ignore_index=True).std(ddof=0))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _to_float(value)
    return "" if number is None else f"{number:.2f}"


def _suffix_from_samples_path(path: Path) -> str:
    stem = path.stem
    prefix = "research_samples_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build optional legacy sample-group factor diagnostics")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-group-size", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    group_compare, discovery, group_csv, discovery_csv, markdown_path = run_factor_discovery(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        min_group_size=args.min_group_size,
    )
    print(f"group rows: {len(group_compare)}")
    print(f"discovery rows: {len(discovery)}")
    print(f"group csv: {group_csv}")
    print(f"discovery csv: {discovery_csv}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


FACTOR_COLUMNS = [
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

RETURN_COLUMNS = [
    "candidate_d2_max_return_pct",
    "candidate_d2_close_return_pct",
    "candidate_d2_max_drawdown_pct",
    "candidate_d3_max_return_pct",
    "candidate_d3_close_return_pct",
    "candidate_d3_max_drawdown_pct",
    "candidate_d5_max_return_pct",
    "candidate_d5_close_return_pct",
    "candidate_d5_max_drawdown_pct",
    "candidate_d10_max_return_pct",
    "candidate_d10_close_return_pct",
    "candidate_d10_max_drawdown_pct",
    "d3_realized_return_pct",
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
    "sample_group",
]


def run_return_distribution_analysis(
    samples_file: str | Path,
    output_dir: str | Path | None = None,
    target_return_pct: float = 7.0,
    loss_cutoff_pct: float = -3.0,
    quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    """Analyze all research samples by return distribution.

    This is a research-only layer. It does not optimize strategy parameters and
    does not use TopN/execution status as the primary label. The primary labels
    are future return buckets and profit/loss quantiles.
    """
    source_path = Path(samples_file)
    samples = pd.read_csv(source_path, dtype={"code": str})
    if samples.empty:
        raise RuntimeError(f"research samples file is empty: {source_path}")

    enriched = prepare_return_samples(samples)
    bucket_compare = build_return_bucket_compare(
        enriched,
        target_return_pct=target_return_pct,
        loss_cutoff_pct=loss_cutoff_pct,
    )
    factor_quantiles = build_factor_quantile_report(
        enriched,
        target_return_pct=target_return_pct,
        loss_cutoff_pct=loss_cutoff_pct,
        quantiles=quantiles,
    )
    profit_loss_compare = build_profit_loss_compare(
        enriched,
        target_return_pct=target_return_pct,
        loss_cutoff_pct=loss_cutoff_pct,
    )
    daily_summary = build_daily_return_summary(
        enriched,
        target_return_pct=target_return_pct,
        loss_cutoff_pct=loss_cutoff_pct,
    )

    out_dir = Path(output_dir) if output_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_samples_path(source_path)

    enriched_csv = out_dir / f"return_samples_{suffix}.csv"
    bucket_csv = out_dir / f"return_bucket_compare_{suffix}.csv"
    quantile_csv = out_dir / f"factor_quantile_report_{suffix}.csv"
    profit_loss_csv = out_dir / f"profit_loss_compare_{suffix}.csv"
    daily_csv = out_dir / f"daily_return_summary_{suffix}.csv"
    markdown_path = out_dir / f"return_distribution_report_{suffix}.md"

    enriched.to_csv(enriched_csv, index=False, encoding="utf-8-sig")
    bucket_compare.to_csv(bucket_csv, index=False, encoding="utf-8-sig")
    factor_quantiles.to_csv(quantile_csv, index=False, encoding="utf-8-sig")
    profit_loss_compare.to_csv(profit_loss_csv, index=False, encoding="utf-8-sig")
    daily_summary.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        build_markdown_report(
            enriched,
            bucket_compare,
            factor_quantiles,
            profit_loss_compare,
            daily_summary,
            samples_file=source_path,
            target_return_pct=target_return_pct,
            loss_cutoff_pct=loss_cutoff_pct,
            quantiles=quantiles,
        ),
        encoding="utf-8",
    )

    return (
        enriched,
        bucket_compare,
        factor_quantiles,
        profit_loss_compare,
        enriched_csv,
        bucket_csv,
        quantile_csv,
        profit_loss_csv,
        markdown_path,
    )


def prepare_return_samples(samples: pd.DataFrame) -> pd.DataFrame:
    frame = samples.copy()
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)

    for column in FACTOR_COLUMNS + RETURN_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in SEGMENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    # Primary research target for all candidates: what happened by D3 close.
    # Opportunity target: whether the path ever offered a D3 target opportunity.
    frame["primary_d3_return_pct"] = frame["candidate_d3_close_return_pct"]
    frame["opportunity_d3_return_pct"] = frame["candidate_d3_max_return_pct"]
    frame["risk_d3_drawdown_pct"] = frame["candidate_d3_max_drawdown_pct"]
    frame["realized_d3_return_pct"] = frame["d3_realized_return_pct"]

    frame["primary_return_bucket"] = frame["primary_d3_return_pct"].apply(bucket_close_return)
    frame["opportunity_return_bucket"] = frame["opportunity_d3_return_pct"].apply(bucket_opportunity_return)
    frame["drawdown_bucket"] = frame["risk_d3_drawdown_pct"].apply(bucket_drawdown)
    frame["realized_return_bucket"] = frame.apply(realized_bucket, axis=1)

    frame["all_sample_target7"] = frame["opportunity_d3_return_pct"] >= 7.0
    frame["all_sample_target10"] = frame["opportunity_d3_return_pct"] >= 10.0
    frame["all_sample_close_profit"] = frame["primary_d3_return_pct"] > 0.0
    frame["all_sample_close_loss"] = frame["primary_d3_return_pct"] < 0.0
    frame["all_sample_big_loss"] = frame["primary_d3_return_pct"] <= -3.0

    return frame


def bucket_close_return(value: Any) -> str:
    value = _to_float(value)
    if value is None:
        return "data_issue"
    if value >= 10:
        return "close_>=10"
    if value >= 7:
        return "close_7_to_10"
    if value >= 3:
        return "close_3_to_7"
    if value >= 0:
        return "close_0_to_3"
    if value >= -3:
        return "close_-3_to_0"
    if value >= -7:
        return "close_-7_to_-3"
    return "close_<-7"


def bucket_opportunity_return(value: Any) -> str:
    value = _to_float(value)
    if value is None:
        return "data_issue"
    if value >= 15:
        return "opportunity_>=15"
    if value >= 10:
        return "opportunity_10_to_15"
    if value >= 7:
        return "opportunity_7_to_10"
    if value >= 3:
        return "opportunity_3_to_7"
    if value >= 0:
        return "opportunity_0_to_3"
    return "opportunity_<0"


def bucket_drawdown(value: Any) -> str:
    value = _to_float(value)
    if value is None:
        return "data_issue"
    if value >= 0:
        return "drawdown_none_or_positive"
    if value >= -3:
        return "drawdown_0_to_-3"
    if value >= -7:
        return "drawdown_-3_to_-7"
    if value >= -10:
        return "drawdown_-7_to_-10"
    return "drawdown_<-10"


def realized_bucket(row: pd.Series) -> str:
    executed = str(row.get("executed", "")).lower() in {"true", "1", "yes"}
    if not executed:
        return "not_executed"
    value = _to_float(row.get("realized_d3_return_pct"))
    if value is None:
        return "executed_data_issue"
    if value >= 10:
        return "realized_>=10"
    if value >= 7:
        return "realized_7_to_10"
    if value >= 3:
        return "realized_3_to_7"
    if value >= 0:
        return "realized_0_to_3"
    if value >= -3:
        return "realized_-3_to_0"
    if value >= -7:
        return "realized_-7_to_-3"
    return "realized_<-7"


def build_return_bucket_compare(
    samples: pd.DataFrame,
    target_return_pct: float,
    loss_cutoff_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket_type, column in [
        ("primary_d3_close_return", "primary_return_bucket"),
        ("opportunity_d3_max_return", "opportunity_return_bucket"),
        ("candidate_d3_drawdown", "drawdown_bucket"),
        ("executed_realized_d3_return", "realized_return_bucket"),
    ]:
        for bucket, group in samples.groupby(column, dropna=False):
            row = summarize_frame(
                group,
                target_return_pct=target_return_pct,
                loss_cutoff_pct=loss_cutoff_pct,
            )
            row.update({"bucket_type": bucket_type, "bucket": bucket})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["bucket_type", "bucket"]).reset_index(drop=True)


def build_factor_quantile_report(
    samples: pd.DataFrame,
    target_return_pct: float,
    loss_cutoff_pct: float,
    quantiles: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = [
        ("all_primary_d3_close", "primary_d3_return_pct", "all_candidates"),
        ("all_opportunity_d3_max", "opportunity_d3_return_pct", "all_candidates"),
        ("executed_realized_d3", "realized_d3_return_pct", "executed_only"),
    ]
    for factor in FACTOR_COLUMNS:
        if factor not in samples.columns:
            continue
        for target_name, target_column, universe_name in targets:
            frame = samples.copy()
            if universe_name == "executed_only":
                frame = frame[frame["realized_return_bucket"] != "not_executed"]
            columns = list(dict.fromkeys([factor, target_column, "opportunity_d3_return_pct", "primary_d3_return_pct"]))
            frame = frame[columns].dropna()
            if len(frame) < max(quantiles * 2, 10):
                continue
            bins = _quantile_bins(frame[factor], quantiles)
            if bins is None:
                continue
            frame = frame.assign(factor_quantile=bins)
            for quantile, group in frame.groupby("factor_quantile"):
                row = {
                    "factor": factor,
                    "target": target_name,
                    "universe": universe_name,
                    "quantile": int(quantile) + 1,
                    "count": int(len(group)),
                    "factor_min": _round(group[factor].min()),
                    "factor_max": _round(group[factor].max()),
                    "factor_mean": _round(group[factor].mean()),
                    "target_mean": _round(group[target_column].mean()),
                    "target_median": _round(group[target_column].median()),
                    "target_p25": _round(group[target_column].quantile(0.25)),
                    "target_p75": _round(group[target_column].quantile(0.75)),
                    "target7_rate": _rate(group[target_column] >= target_return_pct),
                    "target10_rate": _rate(group[target_column] >= 10.0),
                    "loss_rate": _rate(group[target_column] <= loss_cutoff_pct),
                    "opportunity7_rate": _rate(group["opportunity_d3_return_pct"] >= target_return_pct),
                    "primary_close_mean": _round(group["primary_d3_return_pct"].mean()),
                    "opportunity_max_mean": _round(group["opportunity_d3_return_pct"].mean()),
                }
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["factor", "target", "universe", "quantile"]).reset_index(drop=True)


def build_profit_loss_compare(
    samples: pd.DataFrame,
    target_return_pct: float,
    loss_cutoff_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = samples["primary_d3_return_pct"].dropna()
    opportunity = samples["opportunity_d3_return_pct"].dropna()

    top20_primary = primary.quantile(0.80) if not primary.empty else None
    bottom20_primary = primary.quantile(0.20) if not primary.empty else None
    top20_opportunity = opportunity.quantile(0.80) if not opportunity.empty else None

    masks: list[tuple[str, str, pd.Series]] = [
        ("all_candidates", "all_d3_close_profit_ge_target", samples["primary_d3_return_pct"] >= target_return_pct),
        ("all_candidates", "all_d3_close_loss_le_cutoff", samples["primary_d3_return_pct"] <= loss_cutoff_pct),
        ("all_candidates", "all_d3_opportunity_ge_target", samples["opportunity_d3_return_pct"] >= target_return_pct),
        ("all_candidates", "all_d3_opportunity_lt_3", samples["opportunity_d3_return_pct"] < 3.0),
    ]
    if top20_primary is not None:
        masks.append(("all_candidates", "all_d3_close_top20pct", samples["primary_d3_return_pct"] >= top20_primary))
    if bottom20_primary is not None:
        masks.append(("all_candidates", "all_d3_close_bottom20pct", samples["primary_d3_return_pct"] <= bottom20_primary))
    if top20_opportunity is not None:
        masks.append(("all_candidates", "all_d3_opportunity_top20pct", samples["opportunity_d3_return_pct"] >= top20_opportunity))

    executed_mask = samples["realized_return_bucket"] != "not_executed"
    masks.extend(
        [
            ("executed_only", "executed_realized_profit_ge_target", executed_mask & (samples["realized_d3_return_pct"] >= target_return_pct)),
            ("executed_only", "executed_realized_loss_le_cutoff", executed_mask & (samples["realized_d3_return_pct"] <= loss_cutoff_pct)),
        ]
    )

    for universe, group_name, mask in masks:
        group = samples[mask.fillna(False)]
        row = summarize_frame(
            group,
            target_return_pct=target_return_pct,
            loss_cutoff_pct=loss_cutoff_pct,
        )
        row.update({"universe": universe, "group": group_name})
        rows.append(row)

    return pd.DataFrame(rows)


def build_daily_return_summary(
    samples: pd.DataFrame,
    target_return_pct: float,
    loss_cutoff_pct: float,
) -> pd.DataFrame:
    date_column = "signal_date" if "signal_date" in samples.columns else "trade_date"
    if date_column not in samples.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for signal_date, group in samples.groupby(date_column, dropna=False):
        row = summarize_frame(
            group,
            target_return_pct=target_return_pct,
            loss_cutoff_pct=loss_cutoff_pct,
        )
        row.update({"signal_date": signal_date})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def summarize_frame(
    frame: pd.DataFrame,
    target_return_pct: float,
    loss_cutoff_pct: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "count": int(len(frame)),
        "avg_primary_d3_close_return": _mean(frame, "primary_d3_return_pct"),
        "median_primary_d3_close_return": _median(frame, "primary_d3_return_pct"),
        "avg_opportunity_d3_max_return": _mean(frame, "opportunity_d3_return_pct"),
        "median_opportunity_d3_max_return": _median(frame, "opportunity_d3_return_pct"),
        "avg_candidate_d3_drawdown": _mean(frame, "risk_d3_drawdown_pct"),
        "primary_target_rate": _rate(frame["primary_d3_return_pct"] >= target_return_pct),
        "primary_loss_rate": _rate(frame["primary_d3_return_pct"] <= loss_cutoff_pct),
        "opportunity_target_rate": _rate(frame["opportunity_d3_return_pct"] >= target_return_pct),
        "opportunity_10_rate": _rate(frame["opportunity_d3_return_pct"] >= 10.0),
        "executed_rate": _segment_rate(frame, "executed", "True"),
    }
    for factor in FACTOR_COLUMNS:
        if factor in frame.columns:
            row[f"avg_{factor}"] = _mean(frame, factor)
            row[f"median_{factor}"] = _median(frame, factor)
    for segment in ["signal_type", "position_level", "support_type", "sample_group"]:
        if segment in frame.columns and not frame.empty:
            mode = frame[segment].mode(dropna=True)
            row[f"top_{segment}"] = "" if mode.empty else str(mode.iloc[0])
    return row


def build_markdown_report(
    samples: pd.DataFrame,
    bucket_compare: pd.DataFrame,
    factor_quantiles: pd.DataFrame,
    profit_loss_compare: pd.DataFrame,
    daily_summary: pd.DataFrame,
    samples_file: Path,
    target_return_pct: float,
    loss_cutoff_pct: float,
    quantiles: int,
) -> str:
    lines = [
        "# Return Distribution Report",
        "",
        f"- source: `{samples_file}`",
        f"- records: **{len(samples)}**",
        f"- primary all-sample return: `candidate_d3_close_return_pct`",
        f"- opportunity all-sample return: `candidate_d3_max_return_pct`",
        f"- target return threshold: **{target_return_pct:.2f}%**",
        f"- loss cutoff threshold: **{loss_cutoff_pct:.2f}%**",
        "",
        "## Research Principle",
        "",
        "This report treats every candidate as research data. It does not use TopN selection or execution status as the primary label.",
        "The goal is to compare profit/loss and opportunity distributions so later factor changes can be based on data rather than trade-state labels.",
        "",
        "## All-Sample Return Buckets",
        "",
    ]

    primary_buckets = bucket_compare[bucket_compare["bucket_type"] == "primary_d3_close_return"].copy()
    append_table(
        lines,
        primary_buckets,
        [
            "bucket",
            "count",
            "avg_primary_d3_close_return",
            "primary_target_rate",
            "primary_loss_rate",
            "avg_opportunity_d3_max_return",
            "opportunity_target_rate",
            "avg_total_score",
            "avg_graph_quality_score",
            "avg_support_score",
            "avg_active_money_score",
        ],
        max_rows=20,
    )

    lines.extend(["", "## Profit/Loss Comparison Groups", ""])
    append_table(
        lines,
        profit_loss_compare,
        [
            "universe",
            "group",
            "count",
            "avg_primary_d3_close_return",
            "avg_opportunity_d3_max_return",
            "primary_target_rate",
            "primary_loss_rate",
            "opportunity_target_rate",
            "avg_total_score",
            "avg_graph_quality_score",
            "avg_support_score",
            "avg_active_money_score",
        ],
        max_rows=20,
    )

    lines.extend(["", f"## Factor Quantile Report ({quantiles} buckets)", ""])
    lines.append("Use the CSV for full detail. The preview below shows all-sample D3 close-return quantiles.")
    lines.append("")
    preview = factor_quantiles[
        (factor_quantiles.get("target", "") == "all_primary_d3_close")
        & (factor_quantiles.get("universe", "") == "all_candidates")
    ].copy()
    append_table(
        lines,
        preview,
        [
            "factor",
            "quantile",
            "count",
            "factor_mean",
            "target_mean",
            "target_median",
            "target7_rate",
            "loss_rate",
            "opportunity7_rate",
        ],
        max_rows=40,
    )

    lines.extend(["", "## Daily Summary", ""])
    append_table(
        lines,
        daily_summary,
        [
            "signal_date",
            "count",
            "avg_primary_d3_close_return",
            "avg_opportunity_d3_max_return",
            "primary_target_rate",
            "primary_loss_rate",
            "opportunity_target_rate",
            "executed_rate",
        ],
        max_rows=40,
    )

    lines.extend(
        [
            "",
            "## How to Use This Report",
            "",
            "- First compare high-return buckets against low-return buckets.",
            "- Then check whether factor quantiles show monotonic return differences.",
            "- Treat execution and TopN fields as metadata, not primary labels.",
            "- Do not modify strategy from a single run; use this output to decide which factors deserve deeper validation.",
        ]
    )
    return "\n".join(lines)


def append_table(lines: list[str], frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> None:
    if frame is None or frame.empty:
        lines.append("No rows.")
        return
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        lines.append("No matching columns.")
        return
    lines.append("| " + " | ".join(existing) + " |")
    lines.append("|" + "---|" * len(existing))
    for _, row in frame.head(max_rows).iterrows():
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in existing) + " |")


def _quantile_bins(series: pd.Series, quantiles: int) -> pd.Series | None:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.nunique(dropna=True) < 2:
        return None
    try:
        bins = pd.qcut(clean.rank(method="first"), q=quantiles, labels=False, duplicates="drop")
    except ValueError:
        return None
    return bins


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return _round(series.mean())


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return _round(series.median())


def _segment_rate(frame: pd.DataFrame, column: str, value: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = frame[column].astype(str)
    return _round(float((series == value).mean()), 4)


def _rate(mask: pd.Series) -> float | None:
    if mask is None or len(mask) == 0:
        return None
    return _round(float(mask.fillna(False).mean()), 4)


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 4) -> float | None:
    value = _to_float(value)
    if value is None:
        return None
    return round(value, digits)


def _cell(value: Any) -> str:
    value = "" if value is None else value
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/")


def _suffix_from_samples_path(path: Path) -> str:
    stem = path.stem
    prefix = "research_samples_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze all research samples by profit/loss and return distribution")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-return-pct", type=float, default=7.0)
    parser.add_argument("--loss-cutoff-pct", type=float, default=-3.0)
    parser.add_argument("--quantiles", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    (
        enriched,
        bucket_compare,
        factor_quantiles,
        profit_loss_compare,
        enriched_csv,
        bucket_csv,
        quantile_csv,
        profit_loss_csv,
        markdown_path,
    ) = run_return_distribution_analysis(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        target_return_pct=args.target_return_pct,
        loss_cutoff_pct=args.loss_cutoff_pct,
        quantiles=args.quantiles,
    )
    print(f"return samples: {len(enriched)}")
    print(f"return buckets: {len(bucket_compare)}")
    print(f"factor quantile rows: {len(factor_quantiles)}")
    print(f"profit/loss groups: {len(profit_loss_compare)}")
    print(f"return samples csv: {enriched_csv}")
    print(f"return bucket csv: {bucket_csv}")
    print(f"factor quantile csv: {quantile_csv}")
    print(f"profit/loss csv: {profit_loss_csv}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

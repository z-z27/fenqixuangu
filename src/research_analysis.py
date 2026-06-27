from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_data_config

DEFAULT_TARGET_MIN_RETURN_PCT = 7.0
DEFAULT_TARGET_MAX_RETURN_PCT = 10.0

NUMERIC_FACTORS = [
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
]

CATEGORICAL_FACTORS = [
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


def run_research_analysis(
    history_trades_file: str | Path,
    output_dir: str | Path | None = None,
    target_min_return_pct: float = DEFAULT_TARGET_MIN_RETURN_PCT,
    target_max_return_pct: float = DEFAULT_TARGET_MAX_RETURN_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    """Build a research sample set from historical backtest rows.

    This is intentionally a post-processing layer: it does not change signal
    generation or execution logic. It keeps both good and bad cases so the
    strategy can compare success / failed / missed / ordinary samples later.
    """
    source_path = Path(history_trades_file)
    trades = pd.read_csv(source_path, dtype={"code": str})
    if trades.empty:
        raise RuntimeError(f"history trades file is empty: {source_path}")

    samples = build_research_samples(
        trades,
        target_min_return_pct=target_min_return_pct,
        target_max_return_pct=target_max_return_pct,
    )
    factor_compare = build_factor_compare(samples)

    out_dir = Path(output_dir) if output_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_history_path(source_path)
    samples_csv = out_dir / f"research_samples_{suffix}.csv"
    factor_csv = out_dir / f"factor_compare_{suffix}.csv"
    markdown_path = out_dir / f"research_review_{suffix}.md"

    samples.to_csv(samples_csv, index=False, encoding="utf-8-sig")
    factor_compare.to_csv(factor_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        build_research_markdown(
            samples,
            factor_compare,
            history_trades_file=source_path,
            target_min_return_pct=target_min_return_pct,
            target_max_return_pct=target_max_return_pct,
        ),
        encoding="utf-8",
    )
    return samples, factor_compare, samples_csv, factor_csv, markdown_path


def build_research_samples(
    trades: pd.DataFrame,
    target_min_return_pct: float = DEFAULT_TARGET_MIN_RETURN_PCT,
    target_max_return_pct: float = DEFAULT_TARGET_MAX_RETURN_PCT,
) -> pd.DataFrame:
    frame = trades.copy()
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)

    _ensure_bool_columns(
        frame,
        [
            "allowed_bool",
            "eligible_for_trade",
            "selected_by_topn",
            "selected_for_execution",
            "candidate_evaluable",
            "evaluable",
            "executed",
            "target_hit",
            "stop_hit",
        ],
    )
    _ensure_numeric_columns(
        frame,
        set(NUMERIC_FACTORS)
        | {
            "buy_price",
            "d3_max_return_pct",
            "d3_close_return_pct",
            "d3_max_drawdown_pct",
            "d5_max_return_pct",
            "d10_max_return_pct",
            "candidate_d3_max_return_pct",
            "candidate_d3_close_return_pct",
            "candidate_d5_max_return_pct",
            "candidate_d10_max_return_pct",
        },
    )

    frame["d3_target_min_hit"] = frame["d3_max_return_pct"] >= float(target_min_return_pct)
    frame["d3_target_max_hit"] = frame["d3_max_return_pct"] >= float(target_max_return_pct)
    frame["candidate_d3_target_min_hit"] = frame["candidate_d3_max_return_pct"] >= float(target_min_return_pct)
    frame["candidate_d3_target_max_hit"] = frame["candidate_d3_max_return_pct"] >= float(target_max_return_pct)

    realized = []
    sell_reason = []
    for _, row in frame.iterrows():
        if not bool(row.get("executed", False)):
            realized.append(pd.NA)
            sell_reason.append("")
            continue
        d3_max = _to_float(row.get("d3_max_return_pct"))
        d3_close = _to_float(row.get("d3_close_return_pct"))
        if d3_max is not None and d3_max >= float(target_min_return_pct):
            realized.append(float(target_min_return_pct))
            sell_reason.append("d3_target_min")
        elif d3_close is not None:
            realized.append(d3_close)
            sell_reason.append("d3_close")
        else:
            realized.append(pd.NA)
            sell_reason.append("missing_d3")
    frame["d3_realized_return_pct"] = realized
    frame["d3_sell_reason"] = sell_reason
    frame["d3_realized_success_min"] = pd.to_numeric(frame["d3_realized_return_pct"], errors="coerce") >= float(
        target_min_return_pct
    )
    frame["d3_realized_success_max"] = pd.to_numeric(frame["d3_realized_return_pct"], errors="coerce") >= float(
        target_max_return_pct
    )

    frame["sample_group"] = frame.apply(
        lambda row: classify_sample_group(row, target_min_return_pct=target_min_return_pct),
        axis=1,
    )
    frame["research_note"] = frame.apply(_research_note, axis=1)
    return frame


def classify_sample_group(row: pd.Series, target_min_return_pct: float = DEFAULT_TARGET_MIN_RETURN_PCT) -> str:
    failure_reason = str(row.get("failure_reason") or "")
    if failure_reason == "data_issue" or str(row.get("data_reason") or "").startswith("missing"):
        return "data_issue"

    executed = bool(row.get("executed", False))
    selected = bool(row.get("selected_for_execution", False))
    d3_realized = _to_float(row.get("d3_realized_return_pct"))
    candidate_d3 = _to_float(row.get("candidate_d3_max_return_pct"))

    if executed:
        if d3_realized is not None and d3_realized >= float(target_min_return_pct):
            return "success"
        return "failed"

    if candidate_d3 is not None and candidate_d3 >= float(target_min_return_pct):
        return "missed_selected" if selected else "missed_unselected"
    return "ordinary"


def build_factor_compare(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    factors = [factor for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS if factor in samples.columns]
    for factor in factors:
        bucketed = samples.copy()
        bucketed["factor_bucket"] = _bucket_factor(bucketed[factor], factor)
        for bucket, group in bucketed.groupby("factor_bucket", dropna=False):
            rows.append(_factor_bucket_row(factor, bucket, group))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["factor", "bucket"]).reset_index(drop=True)


def build_research_markdown(
    samples: pd.DataFrame,
    factor_compare: pd.DataFrame,
    history_trades_file: Path,
    target_min_return_pct: float,
    target_max_return_pct: float,
) -> str:
    lines = ["# Research Sample Analysis", ""]
    lines.extend(
        [
            f"- source: `{history_trades_file}`",
            f"- target min return: **{target_min_return_pct:.2f}%**",
            f"- target max return: **{target_max_return_pct:.2f}%**",
            f"- records: **{len(samples)}**",
            "",
        ]
    )

    lines.extend(["## Sample Groups", ""])
    if samples.empty or "sample_group" not in samples.columns:
        lines.append("No sample groups.")
    else:
        counts = samples["sample_group"].fillna("unknown").value_counts()
        for group, count in counts.items():
            lines.append(f"- {group}: **{int(count)}**")
    lines.append("")

    executed = samples[samples.get("executed", False).fillna(False).astype(bool)] if "executed" in samples else pd.DataFrame()
    lines.extend(
        [
            "## D3 Realized Performance",
            "",
            f"- executed rows: **{len(executed)}**",
            f"- realized success min rate: **{_format_pct(_mean_bool(executed, 'd3_realized_success_min'))}**",
            f"- realized success max rate: **{_format_pct(_mean_bool(executed, 'd3_realized_success_max'))}**",
            f"- avg D3 realized return: **{_format_number(_mean_numeric(executed, 'd3_realized_return_pct'))}%**",
            f"- median D3 realized return: **{_format_number(_median_numeric(executed, 'd3_realized_return_pct'))}%**",
            "",
        ]
    )

    if not factor_compare.empty:
        lines.extend(
            [
                "## Factor Snapshot",
                "",
                "| factor | bucket | records | success rate | failed rate | missed rate | avg realized D3% | avg candidate D3 max% |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        preferred = {"daily_rank", "days_since_d0", "support_type", "total_score", "support_score", "graph_quality_score"}
        view = factor_compare[factor_compare["factor"].isin(preferred)].head(80)
        for _, row in view.iterrows():
            lines.append(
                "| {factor} | {bucket} | {records} | {success} | {failed} | {missed} | {realized} | {candidate} |".format(
                    factor=row.get("factor", ""),
                    bucket=row.get("bucket", ""),
                    records=int(row.get("record_count") or 0),
                    success=_format_pct(row.get("success_rate")),
                    failed=_format_pct(row.get("failed_rate")),
                    missed=_format_pct(row.get("missed_rate")),
                    realized=_format_number(row.get("avg_d3_realized_return_pct")),
                    candidate=_format_number(row.get("avg_candidate_d3_max_return_pct")),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `success`: D2 triggered, and the simple D3 sell model realized at least the target min return.",
            "- `failed`: D2 triggered, but D3 realized return did not reach the target min return.",
            "- `missed_selected`: selected for execution, D2 did not execute, but candidate D3 max return reached the target min return.",
            "- `missed_unselected`: not selected for execution, but candidate D3 max return reached the target min return.",
            "- `ordinary`: no D2 execution and no D3 target-min opportunity.",
            "- The first D3 sell model is intentionally simple: if D3 max return reaches target min, sell at target min; otherwise sell at D3 close.",
        ]
    )
    return "\n".join(lines)


def _factor_bucket_row(factor: str, bucket: Any, group: pd.DataFrame) -> dict[str, Any]:
    record_count = len(group)
    success_count = int((group["sample_group"] == "success").sum())
    failed_count = int((group["sample_group"] == "failed").sum())
    missed_count = int(group["sample_group"].isin({"missed_selected", "missed_unselected"}).sum())
    ordinary_count = int((group["sample_group"] == "ordinary").sum())
    data_issue_count = int((group["sample_group"] == "data_issue").sum())
    return {
        "factor": factor,
        "bucket": "" if pd.isna(bucket) else str(bucket),
        "record_count": int(record_count),
        "success_count": success_count,
        "failed_count": failed_count,
        "missed_count": missed_count,
        "ordinary_count": ordinary_count,
        "data_issue_count": data_issue_count,
        "success_rate": _safe_rate(success_count, record_count),
        "failed_rate": _safe_rate(failed_count, record_count),
        "missed_rate": _safe_rate(missed_count, record_count),
        "avg_d3_realized_return_pct": _mean_numeric(group, "d3_realized_return_pct"),
        "median_d3_realized_return_pct": _median_numeric(group, "d3_realized_return_pct"),
        "avg_candidate_d3_max_return_pct": _mean_numeric(group, "candidate_d3_max_return_pct"),
        "avg_d3_max_return_pct": _mean_numeric(group, "d3_max_return_pct"),
        "avg_d3_close_return_pct": _mean_numeric(group, "d3_close_return_pct"),
    }


def _bucket_factor(values: pd.Series, factor: str) -> pd.Series:
    if factor in CATEGORICAL_FACTORS:
        return values.fillna("").astype(str)
    numeric = pd.to_numeric(values, errors="coerce")
    if factor == "daily_rank":
        return numeric.map(lambda value: "" if pd.isna(value) else str(int(value)) if value <= 3 else ">3")
    if factor == "days_since_d0":
        return numeric.map(lambda value: "" if pd.isna(value) else str(int(value)) if value <= 3 else ">3")
    if factor in {"low_absorb_width_pct", "invalid_distance_pct"}:
        return pd.cut(numeric, bins=[-1000, 1, 2, 3, 5, 1000], labels=["<=1", "1-2", "2-3", "3-5", ">5"])
    if factor.startswith("candidate_d3"):
        return pd.cut(numeric, bins=[-1000, 0, 3, 7, 10, 1000], labels=["<=0", "0-3", "3-7", "7-10", ">10"])
    return pd.cut(numeric, bins=[-1000, 50, 60, 70, 80, 1000], labels=["<=50", "50-60", "60-70", "70-80", ">80"])


def _research_note(row: pd.Series) -> str:
    group = row.get("sample_group", "")
    if group == "success":
        return "D2 executed and D3 realized target-min return. Study its D1 factors as positive samples."
    if group == "failed":
        return "D2 executed but D3 did not realize target-min return. Compare against success samples."
    if group == "missed_selected":
        return "Selected but not executed; D3 had target-min opportunity. Check whether buy zone was too conservative."
    if group == "missed_unselected":
        return "Not selected but D3 had target-min opportunity. Check ranking/filter factors."
    if group == "ordinary":
        return "No execution and no D3 target-min opportunity. Useful as negative/ordinary baseline."
    if group == "data_issue":
        return "Insufficient or inconsistent data; exclude from factor conclusions."
    return ""


def _ensure_bool_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].astype(str).str.lower().isin({"true", "1", "yes"})


def _ensure_numeric_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _suffix_from_history_path(path: Path) -> str:
    stem = path.stem
    prefix = "history_trades_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


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


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _mean_bool(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    return float(frame[column].fillna(False).astype(bool).mean())


def _mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _median_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _format_pct(value: Any) -> str:
    number = _to_float(value)
    return "" if number is None else f"{number * 100:.2f}%"


def _format_number(value: Any) -> str:
    number = _to_float(value)
    return "" if number is None else f"{number:.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build success/failed/missed research samples from history_trades CSV")
    parser.add_argument("--history-trades-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-min-return-pct", type=float, default=DEFAULT_TARGET_MIN_RETURN_PCT)
    parser.add_argument("--target-max-return-pct", type=float, default=DEFAULT_TARGET_MAX_RETURN_PCT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples, factor_compare, samples_csv, factor_csv, markdown_path = run_research_analysis(
        history_trades_file=args.history_trades_file,
        output_dir=args.output_dir,
        target_min_return_pct=args.target_min_return_pct,
        target_max_return_pct=args.target_max_return_pct,
    )
    print(f"samples: {len(samples)}")
    print(f"factor rows: {len(factor_compare)}")
    print(f"samples csv: {samples_csv}")
    print(f"factor csv: {factor_csv}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

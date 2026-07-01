from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_data_config
from .ranking_backtest import DEFAULT_TARGET_COLUMN, TARGET_DESCRIPTIONS, TARGET_RETURN_COLUMNS


DEFAULT_TARGET_RETURN_PCT = 7.0

NUMERIC_FEATURE_COLUMNS = [
    "total_score",
    "graph_quality_score",
    "active_money_score",
    "active_cooling_score",
    "support_score",
    "theme_score",
    "trend_hold_score",
    "entry_width_score",
    "d1_low_ma10_pct",
    "d1_close_ma10_pct",
    "d1_close_vwap_pct",
    "low_absorb_width_pct",
    "invalid_distance_pct",
    "days_since_d0",
    "consecutive_boards",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "signal_type",
    "support_type",
]

LABEL_COLUMNS = [
    "target7",
    "target7_d2open_d3high",
    "target7_d2open_d3close",
    "target10",
    "candidate_evaluable",
    "d2_open_price",
    "d3_high_price",
    "d3_close_price",
    "d2open_d3high_return_pct",
    "d2open_d3close_return_pct",
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
]

BOOLEAN_LABEL_COLUMNS = {
    "target7",
    "target7_d2open_d3high",
    "target7_d2open_d3close",
    "target10",
    "candidate_evaluable",
}

FORBIDDEN_ANALYSIS_INPUT_PATTERNS = [
    "candidate_d2_*",
    "candidate_d3_*",
    "candidate_d5_*",
    "candidate_d10_*",
    "d2_trade_date",
    "d3_trade_date",
    "d2_open_price",
    "d3_high_price",
    "d3_close_price",
    "d2open_d3*",
    "analysis_target",
    "analysis_return_pct",
    "target7",
    "target7_*",
    "target10",
    "executed",
    "selected_for_execution",
    "selected_by_topn",
    "buy_price",
    "zone_buy_price",
    "confirmation_price",
    "execution_date",
    "buy_time",
    "entry_price_mode",
    "execution_reason",
    "target_hit",
    "stop_hit",
    "first_outcome",
    "first_outcome_time",
    "failure_reason",
    "d3_realized_return_pct",
    "d3_sell_reason",
]

EXECUTION_ONLY_COLUMNS = {
    "executed",
    "selected_for_execution",
    "selected_by_topn",
    "buy_price",
    "zone_buy_price",
    "confirmation_price",
    "execution_date",
    "buy_time",
    "entry_price_mode",
    "execution_reason",
    "target_hit",
    "stop_hit",
    "first_outcome",
    "first_outcome_time",
    "failure_reason",
    "d3_realized_return_pct",
    "d3_sell_reason",
}


def run_factor_analysis(
    samples_file: str | Path,
    output_dir: str | Path | None = None,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    min_bucket_size: int = 10,
    eligible_only: bool = True,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    """Run basic factor analysis on clean history candidate samples.

    This is a research-support layer only. It produces descriptive factor reports
    for manual review. It deliberately does not generate ranking_model_*.json.
    A ranking model must be written later from human analysis of these outputs.
    """
    source_path = Path(samples_file)
    samples = pd.read_csv(source_path, dtype={"code": str})
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)
    return_column = _return_column_for_target(target_column)
    prepared = prepare_history_candidates(
        samples,
        target_return_pct=target_return_pct,
        eligible_only=eligible_only,
        target_column=target_column,
    )
    feature_columns = available_feature_columns(prepared)
    validate_analysis_features(feature_columns, prepared.columns)
    evaluable_count = int(prepared.attrs.get("evaluable_count", len(prepared)))

    factor_summary = build_factor_summary(
        prepared,
        feature_columns,
        target_return_pct=target_return_pct,
        target_column=target_column,
        return_column=return_column,
        evaluable_count=evaluable_count,
    )
    factor_buckets = build_factor_bucket_report(
        prepared,
        feature_columns,
        min_bucket_size=min_bucket_size,
        target_column=target_column,
        return_column=return_column,
    )
    daily_stability = build_factor_daily_stability(
        prepared,
        feature_columns,
        min_bucket_size=min_bucket_size,
        target_column=target_column,
        return_column=return_column,
    )
    pair_review = build_factor_pair_review(
        prepared,
        feature_columns,
        min_bucket_size=min_bucket_size,
        target_column=target_column,
        return_column=return_column,
    )

    target_suffix = _target_file_suffix(target_column)
    out_dir = Path(output_dir) if output_dir else get_data_config().reports_dir / "factor_analysis" / f"{_suffix_from_samples_path(source_path)}{target_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_samples_path(source_path)
    factor_summary_csv = out_dir / f"factor_summary_{suffix}{target_suffix}.csv"
    factor_buckets_csv = out_dir / f"factor_buckets_{suffix}{target_suffix}.csv"
    daily_stability_csv = out_dir / f"factor_daily_stability_{suffix}{target_suffix}.csv"
    pair_review_csv = out_dir / f"factor_pair_review_{suffix}{target_suffix}.csv"
    markdown_path = out_dir / f"factor_analysis_report_{suffix}{target_suffix}.md"

    factor_summary.to_csv(factor_summary_csv, index=False, encoding="utf-8-sig")
    factor_buckets.to_csv(factor_buckets_csv, index=False, encoding="utf-8-sig")
    daily_stability.to_csv(daily_stability_csv, index=False, encoding="utf-8-sig")
    pair_review.to_csv(pair_review_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        build_factor_analysis_markdown(
            samples=prepared,
            factor_summary=factor_summary,
            factor_buckets=factor_buckets,
            daily_stability=daily_stability,
            pair_review=pair_review,
            samples_path=source_path,
            eligible_only=eligible_only,
            target_column=target_column,
            return_column=return_column,
            evaluable_count=evaluable_count,
        ),
        encoding="utf-8",
    )
    return (
        factor_summary,
        factor_buckets,
        daily_stability,
        pair_review,
        factor_summary_csv,
        factor_buckets_csv,
        daily_stability_csv,
        pair_review_csv,
        markdown_path,
    )


def prepare_history_candidates(
    samples: pd.DataFrame,
    target_return_pct: float,
    eligible_only: bool = True,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    frame = samples.copy()
    if frame.empty:
        raise RuntimeError("history candidate sample file is empty")
    leaked = sorted(EXECUTION_ONLY_COLUMNS.intersection(frame.columns))
    if leaked:
        raise RuntimeError(f"input appears to be execution/backtest data, not clean history candidates: {leaked}")
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)
    return_column = _return_column_for_target(target_column)
    if return_column not in frame.columns:
        raise RuntimeError(f"factor analysis target {target_column} requires missing return column {return_column}")
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in NUMERIC_FEATURE_COLUMNS + LABEL_COLUMNS:
        if column in frame.columns:
            if column in BOOLEAN_LABEL_COLUMNS:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)

    frame["analysis_return_pct"] = pd.to_numeric(frame[return_column], errors="coerce")
    frame = frame[frame["analysis_return_pct"].notna()].copy()
    if frame.empty:
        raise RuntimeError(f"no evaluable rows remain after filtering non-null {return_column}")

    if target_column in frame.columns:
        analysis_target = _bool_series(frame[target_column])
    else:
        analysis_target = frame["analysis_return_pct"] >= float(target_return_pct)
    frame["analysis_target"] = analysis_target.fillna(False).astype(bool)
    frame["target7"] = frame["analysis_target"]
    frame["candidate_d3_max_return_pct"] = frame["analysis_return_pct"]
    evaluable_count = int(len(frame))

    if eligible_only and "eligible_for_trade" in frame.columns:
        frame["eligible_for_trade"] = _bool_series(frame["eligible_for_trade"])
        frame = frame[frame["eligible_for_trade"]].copy()
    if "signal_date" not in frame.columns:
        raise RuntimeError("missing signal_date in history candidates")
    if frame.empty:
        raise RuntimeError("no evaluable rows remain after factor-analysis filters")
    result = frame.reset_index(drop=True)
    result.attrs["target_column"] = target_column
    result.attrs["return_column"] = return_column
    result.attrs["target_description"] = _target_description(target_column)
    result.attrs["evaluable_count"] = evaluable_count
    return result


def available_feature_columns(samples: pd.DataFrame) -> list[str]:
    return [column for column in NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS if column in samples.columns]


def validate_analysis_features(feature_columns: list[str], sample_columns: pd.Index) -> None:
    forbidden = [column for column in feature_columns if _matches_any_pattern(column, FORBIDDEN_ANALYSIS_INPUT_PATTERNS)]
    if forbidden:
        raise RuntimeError(f"forbidden future/label/execution columns used as factor inputs: {forbidden}")
    missing = [column for column in feature_columns if column not in sample_columns]
    if missing:
        raise RuntimeError(f"factor feature columns missing in samples: {missing}")


def build_factor_summary(
    samples: pd.DataFrame,
    feature_columns: list[str],
    target_return_pct: float,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
    evaluable_count: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = samples["target7"].fillna(False).astype(bool)
    d3 = pd.to_numeric(samples.get("candidate_d3_max_return_pct"), errors="coerce")
    evaluable_count = int(evaluable_count if evaluable_count is not None else len(samples))
    for feature in feature_columns:
        if feature in NUMERIC_FEATURE_COLUMNS:
            numeric = pd.to_numeric(samples[feature], errors="coerce")
            top_rate = _top_quantile_target_rate(numeric, target)
            bottom_rate = _bottom_quantile_target_rate(numeric, target)
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "count": int(len(samples)),
                    "evaluable_count": evaluable_count,
                    "target_column": target_column,
                    "return_column": return_column,
                    "target_description": _target_description(target_column),
                    "non_null_count": int(numeric.notna().sum()),
                    "target7_count": int(target.sum()),
                    "target7_rate": _safe_rate(int(target.sum()), int(len(target))),
                    "mean": _mean(numeric),
                    "median": _median(numeric),
                    "target7_mean": _mean(numeric[target]),
                    "non_target7_mean": _mean(numeric[~target]),
                    "mean_diff": _diff(_mean(numeric[target]), _mean(numeric[~target])),
                    "corr_target7": _corr(numeric, target.astype(int)),
                    "corr_candidate_d3_max": _corr(numeric, d3),
                    "top_quantile_target7_rate": top_rate,
                    "bottom_quantile_target7_rate": bottom_rate,
                    "quantile_rate_spread": _diff(top_rate, bottom_rate),
                    "direction_hint": _direction_hint(numeric, target),
                    "target_return_pct": float(target_return_pct),
                }
            )
        else:
            rows.append(
                _categorical_summary_row(
                    samples,
                    feature,
                    target,
                    target_return_pct=target_return_pct,
                    target_column=target_column,
                    return_column=return_column,
                    evaluable_count=evaluable_count,
                )
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["analysis_signal_strength"] = result.apply(_analysis_signal_strength, axis=1)
    return result.sort_values(["analysis_signal_strength", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_factor_bucket_report(
    samples: pd.DataFrame,
    feature_columns: list[str],
    min_bucket_size: int = 10,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = samples["target7"].fillna(False).astype(bool)
    d3 = pd.to_numeric(samples.get("candidate_d3_max_return_pct"), errors="coerce")
    base_rate = _safe_rate(int(target.sum()), int(len(target))) or 0.0
    for feature in feature_columns:
        if feature in NUMERIC_FEATURE_COLUMNS:
            bucket = _numeric_bucket(pd.to_numeric(samples[feature], errors="coerce"))
        else:
            bucket = samples[feature].fillna("").astype(str)
        bucket_frame = pd.DataFrame({"bucket": bucket, "target7": target, "candidate_d3_max_return_pct": d3})
        for bucket_value, group in bucket_frame.groupby("bucket", dropna=False):
            count = int(len(group))
            if count < int(min_bucket_size):
                continue
            target_count = int(group["target7"].fillna(False).astype(bool).sum())
            rate = _safe_rate(target_count, count)
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric" if feature in NUMERIC_FEATURE_COLUMNS else "categorical",
                    "target_column": target_column,
                    "return_column": return_column,
                    "bucket": str(bucket_value),
                    "count": count,
                    "target7_count": target_count,
                    "target7_rate": rate,
                    "target7_rate_lift_vs_base": _diff(rate, base_rate),
                    "avg_candidate_d3_max_return_pct": _mean(pd.to_numeric(group["candidate_d3_max_return_pct"], errors="coerce")),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["feature", "target7_rate", "count"], ascending=[True, False, False]).reset_index(drop=True)


def build_factor_daily_stability(
    samples: pd.DataFrame,
    feature_columns: list[str],
    min_bucket_size: int = 10,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature not in NUMERIC_FEATURE_COLUMNS:
            continue
        for signal_date, group in samples.groupby("signal_date", dropna=False):
            if len(group) < int(min_bucket_size):
                continue
            target = group["target7"].fillna(False).astype(bool)
            values = pd.to_numeric(group[feature], errors="coerce")
            top_rate = _top_quantile_target_rate(values, target)
            bottom_rate = _bottom_quantile_target_rate(values, target)
            rows.append(
                {
                    "signal_date": str(signal_date),
                    "feature": feature,
                    "target_column": target_column,
                    "return_column": return_column,
                    "count": int(len(group)),
                    "target7_rate": _safe_rate(int(target.sum()), int(len(target))),
                    "top_quantile_target7_rate": top_rate,
                    "bottom_quantile_target7_rate": bottom_rate,
                    "quantile_rate_spread": _diff(top_rate, bottom_rate),
                    "corr_target7": _corr(values, target.astype(int)),
                    "direction_hint": _direction_hint(values, target),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["feature", "signal_date"]).reset_index(drop=True)


def build_factor_pair_review(
    samples: pd.DataFrame,
    feature_columns: list[str],
    min_bucket_size: int = 10,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
) -> pd.DataFrame:
    numeric_features = [feature for feature in feature_columns if feature in NUMERIC_FEATURE_COLUMNS]
    rows: list[dict[str, Any]] = []
    target = samples["target7"].fillna(False).astype(bool)
    base_rate = _safe_rate(int(target.sum()), int(len(target))) or 0.0
    for idx, left in enumerate(numeric_features):
        left_values = pd.to_numeric(samples[left], errors="coerce")
        left_top = left_values >= left_values.quantile(0.7)
        for right in numeric_features[idx + 1 :]:
            right_values = pd.to_numeric(samples[right], errors="coerce")
            right_top = right_values >= right_values.quantile(0.7)
            mask = left_top & right_top & left_values.notna() & right_values.notna()
            count = int(mask.sum())
            if count < int(min_bucket_size):
                continue
            pair_target_count = int(target[mask].sum())
            pair_rate = _safe_rate(pair_target_count, count)
            rows.append(
                {
                    "left_feature": left,
                    "right_feature": right,
                    "target_column": target_column,
                    "return_column": return_column,
                    "pair_rule": "both_top_30pct",
                    "count": count,
                    "target7_count": pair_target_count,
                    "target7_rate": pair_rate,
                    "target7_rate_lift_vs_base": _diff(pair_rate, base_rate),
                    "feature_corr": _corr(left_values, right_values),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["target7_rate_lift_vs_base", "target7_rate", "count"], ascending=[False, False, False]).reset_index(drop=True)


def build_factor_analysis_markdown(
    samples: pd.DataFrame,
    factor_summary: pd.DataFrame,
    factor_buckets: pd.DataFrame,
    daily_stability: pd.DataFrame,
    pair_review: pd.DataFrame,
    samples_path: Path,
    eligible_only: bool,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
    evaluable_count: int | None = None,
) -> str:
    target = samples["target7"].fillna(False).astype(bool)
    target_description = _target_description(target_column)
    evaluable_count = int(evaluable_count if evaluable_count is not None else len(samples))
    lines = ["# Factor Analysis Report", ""]
    lines.extend(
        [
            "## Scope",
            "",
            "This report is a descriptive factor-analysis layer. It does not generate ranking_model_*.json.",
            "A ranking model must be manually constructed after reviewing these outputs.",
            "",
            "## Sample",
            "",
            f"- samples file: `{samples_path}`",
            f"- target column: **{target_column}**",
            f"- return column: **{return_column}**",
            f"- target definition: {target_description}",
            f"- evaluable rows: **{evaluable_count}**",
            f"- rows after filters: **{len(samples)}**",
            f"- dates: **{samples['signal_date'].dropna().nunique()}**",
            f"- eligible only: **{eligible_only}**",
            f"- target rows: **{int(target.sum())}**",
            f"- target rate: **{_format_pct(_safe_rate(int(target.sum()), len(samples)))}**",
            "",
            "## Top Factor Signals",
            "",
            "| feature | type | strength | corr target | corr return | top q rate | bottom q rate | direction |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not factor_summary.empty:
        for _, row in factor_summary.head(30).iterrows():
            lines.append(
                f"| {row.get('feature', '')} | {row.get('feature_type', '')} | {_format_number(row.get('analysis_signal_strength'))} | {_format_number(row.get('corr_target7'))} | {_format_number(row.get('corr_candidate_d3_max'))} | {_format_pct(row.get('top_quantile_target7_rate'))} | {_format_pct(row.get('bottom_quantile_target7_rate'))} | {row.get('direction_hint', '')} |"
            )
    if not factor_buckets.empty:
        lines.extend(["", "## Strong Buckets", "", "| feature | bucket | count | target rate | lift | avg return% |", "|---|---|---:|---:|---:|---:|"])
        strong = factor_buckets.sort_values(["target7_rate_lift_vs_base", "target7_rate", "count"], ascending=[False, False, False]).head(40)
        for _, row in strong.iterrows():
            lines.append(
                f"| {row.get('feature', '')} | {row.get('bucket', '')} | {int(row.get('count', 0))} | {_format_pct(row.get('target7_rate'))} | {_format_pct(row.get('target7_rate_lift_vs_base'))} | {_format_number(row.get('avg_candidate_d3_max_return_pct'))} |"
            )
    if not pair_review.empty:
        lines.extend(["", "## Pair Review", "", "| left | right | rule | count | target rate | lift | corr |", "|---|---|---|---:|---:|---:|---:|"])
        for _, row in pair_review.head(30).iterrows():
            lines.append(
                f"| {row.get('left_feature', '')} | {row.get('right_feature', '')} | {row.get('pair_rule', '')} | {int(row.get('count', 0))} | {_format_pct(row.get('target7_rate'))} | {_format_pct(row.get('target7_rate_lift_vs_base'))} | {_format_number(row.get('feature_corr'))} |"
            )
    if not daily_stability.empty:
        stable = daily_stability.groupby("feature", dropna=False).agg(
            days=("signal_date", "count"),
            avg_spread=("quantile_rate_spread", "mean"),
            avg_corr=("corr_target7", "mean"),
        ).reset_index().sort_values("avg_spread", ascending=False)
        lines.extend(["", "## Daily Stability Snapshot", "", "| feature | days | avg spread | avg corr |", "|---|---:|---:|---:|"])
        for _, row in stable.head(30).iterrows():
            lines.append(f"| {row.get('feature', '')} | {int(row.get('days', 0))} | {_format_number(row.get('avg_spread'))} | {_format_number(row.get('avg_corr'))} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These outputs are evidence for manual research. They are not a trading model and are not a validated ranking model.",
            "Legacy target7 is a D1-close proxy target based on the D2+D3 high window.",
        ]
    )
    return "\n".join(lines)


def _categorical_summary_row(
    samples: pd.DataFrame,
    feature: str,
    target: pd.Series,
    target_return_pct: float,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
    evaluable_count: int | None = None,
) -> dict[str, Any]:
    values = samples[feature].fillna("").astype(str)
    grouped = pd.DataFrame({"value": values, "target7": target}).groupby("value", dropna=False)
    bucket_rates = []
    for _, group in grouped:
        if len(group) == 0:
            continue
        bucket_rates.append(_safe_rate(int(group["target7"].sum()), int(len(group))) or 0.0)
    spread = max(bucket_rates) - min(bucket_rates) if bucket_rates else 0.0
    return {
        "feature": feature,
        "feature_type": "categorical",
        "count": int(len(samples)),
        "evaluable_count": int(evaluable_count if evaluable_count is not None else len(samples)),
        "target_column": target_column,
        "return_column": return_column,
        "target_description": _target_description(target_column),
        "non_null_count": int(values.astype(bool).sum()),
        "target7_count": int(target.sum()),
        "target7_rate": _safe_rate(int(target.sum()), int(len(target))),
        "mean": None,
        "median": None,
        "target7_mean": None,
        "non_target7_mean": None,
        "mean_diff": None,
        "corr_target7": None,
        "corr_candidate_d3_max": None,
        "top_quantile_target7_rate": None,
        "bottom_quantile_target7_rate": None,
        "quantile_rate_spread": spread,
        "direction_hint": f"bucket_spread={spread:.4f}",
        "target_return_pct": float(target_return_pct),
    }


def _analysis_signal_strength(row: pd.Series) -> float:
    if row.get("feature_type") == "categorical":
        return abs(float(row.get("quantile_rate_spread") or 0.0))
    components = [
        abs(float(row.get("corr_target7") or 0.0)),
        abs(float(row.get("corr_candidate_d3_max") or 0.0)),
        abs(float(row.get("quantile_rate_spread") or 0.0)),
    ]
    return float(sum(components))


def _numeric_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    non_null = numeric.dropna()
    if non_null.nunique() <= 1:
        return numeric.map(lambda value: "missing" if pd.isna(value) else str(value))
    try:
        return pd.qcut(numeric, q=min(5, int(non_null.nunique())), duplicates="drop")
    except Exception:
        return pd.cut(numeric, bins=5, duplicates="drop")


def _top_quantile_target_rate(values: pd.Series, target: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = numeric.quantile(0.8)
    if pd.isna(threshold):
        return None
    mask = numeric >= threshold
    if not bool(mask.any()):
        return None
    return _safe_rate(int(target[mask].sum()), int(mask.sum()))


def _bottom_quantile_target_rate(values: pd.Series, target: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = numeric.quantile(0.2)
    if pd.isna(threshold):
        return None
    mask = numeric <= threshold
    if not bool(mask.any()):
        return None
    return _safe_rate(int(target[mask].sum()), int(mask.sum()))


def _direction_hint(values: pd.Series, target: pd.Series) -> str:
    top = _top_quantile_target_rate(values, target)
    bottom = _bottom_quantile_target_rate(values, target)
    if top is None or bottom is None:
        return "unknown"
    if top > bottom:
        return "higher_better"
    if top < bottom:
        return "lower_better"
    return "flat"


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def _matches_any_pattern(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(column, pattern) for pattern in patterns)


def _return_column_for_target(target_column: str) -> str:
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)
    return TARGET_RETURN_COLUMNS.get(target_column, target_column)


def _target_description(target_column: str) -> str:
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)
    return TARGET_DESCRIPTIONS.get(target_column, f"custom boolean target column: {target_column}")


def _target_file_suffix(target_column: str) -> str:
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)
    if target_column == DEFAULT_TARGET_COLUMN:
        return ""
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in target_column)
    return f"_{safe}"


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["left"].nunique() <= 1 or frame["right"].nunique() <= 1:
        return None
    return float(frame["left"].corr(frame["right"]))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _suffix_from_samples_path(path: Path) -> str:
    stem = path.stem
    for prefix in ("history_candidates_", "research_samples_", "return_samples_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run descriptive factor analysis from clean history candidates.")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-return-pct", type=float, default=DEFAULT_TARGET_RETURN_PCT)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--min-bucket-size", type=int, default=10)
    parser.add_argument("--all-candidates", action="store_true", help="do not filter to eligible_for_trade rows; target return data is still required")
    args = parser.parse_args(argv)
    factor_summary, factor_buckets, daily_stability, pair_review, summary_csv, buckets_csv, stability_csv, pair_csv, markdown_path = run_factor_analysis(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        target_return_pct=args.target_return_pct,
        min_bucket_size=args.min_bucket_size,
        eligible_only=not args.all_candidates,
        target_column=args.target_column,
    )
    print(f"factor summary rows: {len(factor_summary)}")
    print(f"factor bucket rows: {len(factor_buckets)}")
    print(f"daily stability rows: {len(daily_stability)}")
    print(f"pair review rows: {len(pair_review)}")
    print(f"target column: {args.target_column}")
    print(f"factor summary csv: {summary_csv}")
    print(f"factor buckets csv: {buckets_csv}")
    print(f"daily stability csv: {stability_csv}")
    print(f"pair review csv: {pair_csv}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

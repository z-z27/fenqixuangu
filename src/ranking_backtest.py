from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_data_config


DEFAULT_TARGET_RETURN_PCT = 7.0
DEFAULT_TARGET_COLUMN = "target7"
SUPPORTED_MODEL_TYPES = {"linear_score", "bucket_score", "interaction_rules"}

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

DEFAULT_FORBIDDEN_INPUT_PATTERNS = [
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
    "ranking_target",
    "ranking_return_pct",
    "target7",
    "target7_*",
    "target10",
    *sorted(EXECUTION_ONLY_COLUMNS),
]

TARGET_RETURN_COLUMNS = {
    "target7": "candidate_d3_max_return_pct",
    "target7_d2open_d3high": "d2open_d3high_return_pct",
    "target7_d2open_d3close": "d2open_d3close_return_pct",
}

TARGET_DESCRIPTIONS = {
    "target7": "legacy D1 close proxy target: D2+D3 high window >= target return",
    "target7_d2open_d3high": "D2 open entry reference, D3 intraday high >= target return",
    "target7_d2open_d3close": "D2 open entry reference, D3 close >= target return",
}


def run_ranking_backtest(
    samples_file: str | Path,
    model_file: str | Path,
    output_dir: str | Path | None = None,
    top_n: int = 3,
    target_return_pct: float | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    """Validate a research ranking model against clean history candidate samples.

    This layer scores D1-night candidates, ranks them by signal_date, selects
    TopN, and evaluates the requested target column.
    """
    samples_path = Path(samples_file)
    model_path = Path(model_file)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    target_pct = float(target_return_pct if target_return_pct is not None else model.get("target_return_pct", DEFAULT_TARGET_RETURN_PCT))
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)

    samples = pd.read_csv(samples_path, dtype={"code": str})
    candidates = prepare_ranking_samples(samples, target_return_pct=target_pct, target_column=target_column)
    validate_ranking_model(model, candidates.columns)
    scored = score_candidates(candidates, model)
    ranked = rank_candidates(scored)
    topn = select_daily_topn(ranked, top_n=top_n)
    return_column = _return_column_for_target(target_column)
    daily = build_daily_ranking_report(ranked, topn, top_n=top_n, target_column=target_column, return_column=return_column)
    failures = build_failure_report(daily)
    summary = build_ranking_summary(
        ranked,
        topn,
        daily,
        model,
        samples_path,
        model_path,
        top_n=top_n,
        target_return_pct=target_pct,
        target_column=target_column,
        return_column=return_column,
    )

    out_dir = Path(output_dir) if output_dir else _default_output_dir(model, samples_path, top_n, target_column)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_samples_path(samples_path)
    model_id = str(model.get("model_id", model_path.stem))
    target_suffix = _target_file_suffix(target_column)
    summary_csv = out_dir / f"ranking_backtest_summary_{model_id}_top{top_n}_{suffix}{target_suffix}.csv"
    daily_csv = out_dir / f"ranking_backtest_daily_{model_id}_top{top_n}_{suffix}{target_suffix}.csv"
    topn_csv = out_dir / f"ranking_backtest_topn_{model_id}_top{top_n}_{suffix}{target_suffix}.csv"
    failures_csv = out_dir / f"ranking_backtest_failures_{model_id}_top{top_n}_{suffix}{target_suffix}.csv"
    markdown_path = out_dir / f"ranking_backtest_{model_id}_top{top_n}_{suffix}{target_suffix}.md"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    topn.to_csv(topn_csv, index=False, encoding="utf-8-sig")
    failures.to_csv(failures_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(build_ranking_markdown(summary, daily, failures, model, samples_path, model_path), encoding="utf-8")
    return ranked, topn, daily, failures, summary, summary_csv, daily_csv, topn_csv, failures_csv, markdown_path


def prepare_ranking_samples(samples: pd.DataFrame, target_return_pct: float, target_column: str = DEFAULT_TARGET_COLUMN) -> pd.DataFrame:
    frame = samples.copy()
    if frame.empty:
        raise RuntimeError("history candidate sample file is empty")
    leaked = sorted(EXECUTION_ONLY_COLUMNS.intersection(frame.columns))
    if leaked:
        raise RuntimeError(f"ranking_backtest input contains execution-only columns: {leaked}")
    target_column = str(target_column or DEFAULT_TARGET_COLUMN)
    return_column = _return_column_for_target(target_column)
    required = ["signal_date", "code"]
    if target_column not in frame.columns and return_column not in frame.columns:
        required.append(target_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"ranking_backtest input missing required columns: {missing}")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["signal_date"] = frame["signal_date"].astype(str)
    if "candidate_d3_max_return_pct" in frame.columns:
        frame["candidate_d3_max_return_pct"] = pd.to_numeric(frame["candidate_d3_max_return_pct"], errors="coerce")
    if return_column in frame.columns and return_column != target_column:
        frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    if target_column in frame.columns:
        frame[target_column] = _bool_series(frame[target_column])
    else:
        if return_column not in frame.columns:
            raise RuntimeError(f"ranking_backtest target {target_column} requires missing return column {return_column}")
        frame[target_column] = frame[return_column] >= float(target_return_pct)
    frame["ranking_target"] = frame[target_column].fillna(False).astype(bool)
    if return_column in frame.columns and return_column != target_column:
        frame["ranking_return_pct"] = pd.to_numeric(frame[return_column], errors="coerce")
    else:
        frame["ranking_return_pct"] = pd.NA
    if "eligible_for_trade" in frame.columns:
        frame["eligible_for_trade"] = _bool_series(frame["eligible_for_trade"])
    else:
        frame["eligible_for_trade"] = True
    if "graph_quality_score" not in frame.columns:
        frame["graph_quality_score"] = 0.0
    frame["graph_quality_score"] = pd.to_numeric(frame["graph_quality_score"], errors="coerce").fillna(0.0)
    return frame


def validate_ranking_model(model: dict[str, Any], sample_columns: pd.Index) -> None:
    model_type = str(model.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise RuntimeError(f"unsupported model_type={model_type}; supported={sorted(SUPPORTED_MODEL_TYPES)}")
    features = list(model.get("feature_columns") or [])
    if not features:
        raise RuntimeError("ranking model has no feature_columns")
    missing = [feature for feature in features if feature not in sample_columns]
    if missing:
        raise RuntimeError(f"ranking model feature columns missing in samples: {missing}")
    forbidden = sorted(set(DEFAULT_FORBIDDEN_INPUT_PATTERNS).union(model.get("forbidden_input_patterns") or []))
    forbidden_features = [feature for feature in features if _matches_any_pattern(feature, forbidden)]
    if forbidden_features:
        raise RuntimeError(f"ranking model uses forbidden future/label/execution inputs: {forbidden_features}")


def score_candidates(candidates: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    model_type = str(model.get("model_type"))
    if model_type == "linear_score":
        return score_linear(candidates, model)
    if model_type == "bucket_score":
        return score_bucket(candidates, model)
    if model_type == "interaction_rules":
        return score_interaction_rules(candidates, model)
    raise RuntimeError(f"unsupported model_type={model_type}")


def score_linear(candidates: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    result = candidates.copy()
    body = model.get("model", {})
    weights = body.get("weights", {}) or {}
    score = pd.Series(float(body.get("intercept", 0.0)), index=result.index, dtype="float64")
    for feature, weight in weights.items():
        if feature not in result.columns:
            raise RuntimeError(f"linear_score feature missing: {feature}")
        values = pd.to_numeric(result[feature], errors="coerce")
        if body.get("normalization") == "rank_percentile_per_sample_file":
            values = values.rank(pct=True).fillna(0.0)
        else:
            values = values.fillna(0.0)
        score = score + float(weight) * values
    result[str(model.get("score_column", "research_score"))] = score.astype(float)
    return result


def score_bucket(candidates: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    result = candidates.copy()
    body = model.get("model", {})
    score = pd.Series(float(body.get("base_score", 0.0)), index=result.index, dtype="float64")
    for rule in body.get("rules", []) or []:
        feature = rule.get("feature")
        if feature not in result.columns:
            raise RuntimeError(f"bucket_score feature missing: {feature}")
        values = pd.to_numeric(result[feature], errors="coerce")
        feature_score = pd.Series(float(rule.get("missing_score", 0.0)), index=result.index, dtype="float64")
        for bucket in rule.get("buckets", []) or []:
            mask = values.notna()
            min_value = bucket.get("min")
            max_value = bucket.get("max")
            if min_value is not None:
                if bool(bucket.get("include_min", True)):
                    mask = mask & (values >= float(min_value))
                else:
                    mask = mask & (values > float(min_value))
            if max_value is not None:
                if bool(bucket.get("include_max", True)):
                    mask = mask & (values <= float(max_value))
                else:
                    mask = mask & (values < float(max_value))
            feature_score.loc[mask] = float(bucket.get("score", 0.0))
        score = score + feature_score
    result[str(model.get("score_column", "research_score"))] = score.astype(float)
    return result


def score_interaction_rules(candidates: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    result = candidates.copy()
    body = model.get("model", {})
    score = pd.Series(float(body.get("base_score", 0.0)), index=result.index, dtype="float64")
    for rule in body.get("rules", []) or []:
        mask = pd.Series(True, index=result.index)
        for condition in rule.get("when", []) or []:
            mask = mask & _condition_mask(result, condition)
        score.loc[mask] = score.loc[mask] + float(rule.get("score", 0.0))
    result[str(model.get("score_column", "research_score"))] = score.astype(float)
    return result


def _condition_mask(frame: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    feature = condition.get("feature")
    if feature not in frame.columns:
        raise RuntimeError(f"interaction_rules feature missing: {feature}")
    values = pd.to_numeric(frame[feature], errors="coerce")
    op = condition.get("op")
    value = float(condition.get("value"))
    if op == ">=":
        return values >= value
    if op == ">":
        return values > value
    if op == "<=":
        return values <= value
    if op == "<":
        return values < value
    if op == "==":
        return values == value
    raise RuntimeError(f"unsupported interaction op={op}")


def rank_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    score_column = "research_score"
    if score_column not in scored.columns:
        raise RuntimeError("scored candidates missing research_score")
    eligible = scored[scored["eligible_for_trade"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        return eligible
    eligible["research_score"] = pd.to_numeric(eligible["research_score"], errors="coerce").fillna(float("-inf"))
    if "candidate_d3_max_return_pct" in eligible.columns:
        eligible["candidate_d3_max_return_pct"] = pd.to_numeric(eligible["candidate_d3_max_return_pct"], errors="coerce")
    if "ranking_return_pct" in eligible.columns:
        eligible["ranking_return_pct"] = pd.to_numeric(eligible["ranking_return_pct"], errors="coerce")
    eligible = eligible.sort_values(
        ["signal_date", "research_score", "graph_quality_score", "code"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    eligible["daily_rank"] = eligible.groupby("signal_date").cumcount() + 1
    return eligible


def select_daily_topn(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()
    return ranked[ranked["daily_rank"] <= int(top_n)].copy().reset_index(drop=True)


def build_daily_ranking_report(
    ranked: pd.DataFrame,
    topn: pd.DataFrame,
    top_n: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = sorted(ranked["signal_date"].dropna().astype(str).unique().tolist()) if not ranked.empty else []
    for signal_date in dates:
        day_all = ranked[ranked["signal_date"].astype(str) == signal_date]
        day_top = topn[topn["signal_date"].astype(str) == signal_date]
        top_target = day_top["ranking_target"].fillna(False).astype(bool) if not day_top.empty else pd.Series(dtype=bool)
        returns = pd.to_numeric(day_top.get("ranking_return_pct"), errors="coerce") if not day_top.empty else pd.Series(dtype=float)
        rows.append(
            {
                "signal_date": signal_date,
                "top_n": int(top_n),
                "target_column": target_column,
                "return_column": return_column,
                "eligible_count": int(len(day_all)),
                "topn_count": int(len(day_top)),
                "topn_target_count": int(top_target.sum()) if len(day_top) else 0,
                "topn_target_rate": _safe_rate(int(top_target.sum()), int(len(day_top))) if len(day_top) else None,
                "daily_hit": bool(top_target.any()) if len(day_top) else False,
                "top1_code": _joined(day_top.head(1), "code"),
                "top1_score": _first_number(day_top, "research_score"),
                "top1_return_pct": _first_number(day_top, "ranking_return_pct"),
                "best_topn_return_pct": _max_number(returns),
                "avg_topn_return_pct": _mean(returns),
                "topn_codes": _joined(day_top, "code"),
                "topn_scores": _joined_numbers(day_top, "research_score"),
                "topn_returns": _joined_numbers(day_top, "ranking_return_pct"),
            }
        )
    return pd.DataFrame(rows)


def build_failure_report(daily: pd.DataFrame) -> pd.DataFrame:
    columns = ["signal_date", "top_n", "target_column", "topn_codes", "topn_scores", "topn_returns", "reason"]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    failures = daily[~daily["daily_hit"].fillna(False).astype(bool)].copy()
    if failures.empty:
        return pd.DataFrame(columns=columns)
    failures["reason"] = "no_topn_target_hit"
    return failures[columns].reset_index(drop=True)


def build_ranking_summary(
    ranked: pd.DataFrame,
    topn: pd.DataFrame,
    daily: pd.DataFrame,
    model: dict[str, Any],
    samples_path: Path,
    model_path: Path,
    top_n: int,
    target_return_pct: float,
    target_column: str = DEFAULT_TARGET_COLUMN,
    return_column: str = "candidate_d3_max_return_pct",
) -> pd.DataFrame:
    top_targets = topn["ranking_target"].fillna(False).astype(bool) if not topn.empty else pd.Series(dtype=bool)
    daily_hits = daily["daily_hit"].fillna(False).astype(bool) if not daily.empty else pd.Series(dtype=bool)
    top1_returns = pd.to_numeric(daily.get("top1_return_pct"), errors="coerce") if not daily.empty else pd.Series(dtype=float)
    topn_returns = pd.to_numeric(topn.get("ranking_return_pct"), errors="coerce") if not topn.empty else pd.Series(dtype=float)
    topn_target_rate = _safe_rate(int(top_targets.sum()), int(len(topn)))
    avg_top1_return = _mean(top1_returns)
    avg_topn_return = _mean(topn_returns)
    legacy_fields = {}
    if target_column == DEFAULT_TARGET_COLUMN:
        legacy_fields = {
            "topn_target7_count": int(top_targets.sum()),
            "topn_target7_rate": topn_target_rate,
            "avg_top1_candidate_d3_max_return_pct": avg_top1_return,
            "avg_topn_candidate_d3_max_return_pct": avg_topn_return,
        }
    return pd.DataFrame(
        [
            {
                "model_id": model.get("model_id", model_path.stem),
                "model_type": model.get("model_type"),
                "samples_file": str(samples_path),
                "model_file": str(model_path),
                "date_count": int(daily["signal_date"].nunique()) if not daily.empty else 0,
                "candidate_count": int(len(ranked)),
                "eligible_count": int(len(ranked)),
                "top_n": int(top_n),
                "target_return_pct": float(target_return_pct),
                "target_column": target_column,
                "target_description": _target_description(target_column),
                "return_column": return_column,
                "topn_row_count": int(len(topn)),
                "topn_target_count": int(top_targets.sum()),
                "topn_target_rate": topn_target_rate,
                "daily_hit_count": int(daily_hits.sum()),
                "daily_fail_count": int((~daily_hits).sum()) if len(daily_hits) else 0,
                "daily_hit_rate": _safe_rate(int(daily_hits.sum()), int(len(daily_hits))),
                "avg_top1_return_pct": avg_top1_return,
                "avg_topn_return_pct": avg_topn_return,
                **legacy_fields,
            }
        ]
    )


def build_ranking_markdown(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    failures: pd.DataFrame,
    model: dict[str, Any],
    samples_path: Path,
    model_path: Path,
) -> str:
    item = summary.iloc[0] if not summary.empty else {}
    lines = [f"# Ranking Backtest: {model.get('model_id', model_path.stem)}", ""]
    lines.extend(
        [
            "## Scope",
            "",
            "This report validates a research ranking model against clean history candidate samples. It does not simulate D2 execution and does not modify the formal daily strategy.",
            "",
            "## Inputs",
            "",
            f"- samples file: `{samples_path}`",
            f"- model file: `{model_path}`",
            f"- model type: **{model.get('model_type')}**",
            f"- target column: **{item.get('target_column', DEFAULT_TARGET_COLUMN) if len(summary) else DEFAULT_TARGET_COLUMN}**",
            f"- return column: **{item.get('return_column', '') if len(summary) else ''}**",
            f"- target definition: {item.get('target_description', '') if len(summary) else ''}",
            "",
            "## Summary",
            "",
            f"- dates: **{int(item.get('date_count', 0)) if len(summary) else 0}**",
            f"- eligible candidates: **{int(item.get('eligible_count', 0)) if len(summary) else 0}**",
            f"- top N: **{int(item.get('top_n', 0)) if len(summary) else 0}**",
            f"- topN row target rate: **{_format_pct(item.get('topn_target_rate')) if len(summary) else ''}**",
            f"- daily hit rate: **{_format_pct(item.get('daily_hit_rate')) if len(summary) else ''}**",
            f"- avg top1 selected return: **{_format_number(item.get('avg_top1_return_pct')) if len(summary) else ''}%**",
            f"- avg topN selected return: **{_format_number(item.get('avg_topn_return_pct')) if len(summary) else ''}%**",
            "",
        ]
    )
    if not failures.empty:
        lines.extend(["## Failure Dates", "", "| signal_date | top_n | codes | returns |", "|---|---:|---|---|"])
        for _, row in failures.head(80).iterrows():
            lines.append(f"| {row.get('signal_date', '')} | {row.get('top_n', '')} | {row.get('topn_codes', '')} | {row.get('topn_returns', '')} |")
        lines.append("")
    if not daily.empty:
        lines.extend(["## Daily Snapshot", "", "| date | hit | top codes | best return% | avg return% |", "|---|---|---|---:|---:|"])
        for _, row in daily.head(120).iterrows():
            lines.append(
                f"| {row.get('signal_date', '')} | {bool(row.get('daily_hit'))} | {row.get('topn_codes', '')} | {_format_number(row.get('best_topn_return_pct'))} | {_format_number(row.get('avg_topn_return_pct'))} |"
            )
    return "\n".join(lines)


def _default_output_dir(model: dict[str, Any], samples_path: Path, top_n: int, target_column: str = DEFAULT_TARGET_COLUMN) -> Path:
    model_id = str(model.get("model_id", "ranking_model"))
    target_suffix = _target_file_suffix(target_column)
    return get_data_config().reports_dir / "ranking_backtests" / f"{model_id}_top{int(top_n)}_{_suffix_from_samples_path(samples_path)}{target_suffix}"


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


def _matches_any_pattern(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(column, pattern) for pattern in patterns)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _max_number(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _first_number(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.iloc[0])


def _joined(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    return ",".join(frame[column].astype(str).tolist())


def _joined_numbers(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce")
    return ",".join("" if pd.isna(value) else f"{float(value):.4f}" for value in values.tolist())


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
    parser = argparse.ArgumentParser(description="Validate a research ranking model against clean history candidates.")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--target-return-pct", type=float, default=None)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    args = parser.parse_args(argv)
    ranked, topn, daily, failures, summary, summary_csv, daily_csv, topn_csv, failures_csv, markdown_path = run_ranking_backtest(
        samples_file=args.samples_file,
        model_file=args.model_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        target_return_pct=args.target_return_pct,
        target_column=args.target_column,
    )
    item = summary.iloc[0] if not summary.empty else {}
    print(f"eligible candidates: {len(ranked)}")
    print(f"topn rows: {len(topn)}")
    print(f"target column: {item.get('target_column', '') if len(summary) else ''}")
    print(f"daily hit rate: {item.get('daily_hit_rate', '') if len(summary) else ''}")
    print(f"topn target rate: {item.get('topn_target_rate', '') if len(summary) else ''}")
    print(f"summary csv: {summary_csv}")
    print(f"daily csv: {daily_csv}")
    print(f"topn csv: {topn_csv}")
    print(f"failures csv: {failures_csv}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

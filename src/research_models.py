from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_data_config


DEFAULT_TARGET_RETURN_PCT = 7.0
DEFAULT_MODEL_ID = "ranking_model_v001"
DEFAULT_MODEL_TYPE = "bucket_score"
SUPPORTED_MODEL_TYPES = {"linear_score", "bucket_score", "interaction_rules"}

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
    "target10",
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

FORBIDDEN_INPUT_PATTERNS = [
    "candidate_d2_*",
    "candidate_d3_*",
    "candidate_d5_*",
    "candidate_d10_*",
    "target7",
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


def run_research_model_generation(
    samples_file: str | Path,
    output_dir: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_type: str = DEFAULT_MODEL_TYPE,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    min_bucket_size: int = 10,
    max_features: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path, Path, Path]:
    """Analyze clean history candidate samples and emit a research ranking model.

    This layer reads history_candidates_*.csv only. It treats target7 and all
    candidate future-return columns as labels/evaluation fields, never as model
    input features. It does not modify signal_engine.py or config.py.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise RuntimeError(f"unsupported model_type={model_type}; supported={sorted(SUPPORTED_MODEL_TYPES)}")
    source_path = Path(samples_file)
    samples = pd.read_csv(source_path, dtype={"code": str})
    prepared = prepare_history_candidates(samples, target_return_pct=target_return_pct)
    feature_columns = available_feature_columns(prepared)
    validate_model_inputs(feature_columns, prepared.columns)

    factor_summary = build_factor_summary(prepared, feature_columns, target_return_pct=target_return_pct)
    factor_buckets = build_factor_bucket_report(prepared, feature_columns, min_bucket_size=min_bucket_size)
    model = build_research_ranking_model(
        factor_summary=factor_summary,
        factor_buckets=factor_buckets,
        samples=prepared,
        feature_columns=feature_columns,
        model_id=model_id,
        model_type=model_type,
        target_return_pct=target_return_pct,
        max_features=max_features,
        min_bucket_size=min_bucket_size,
    )

    out_dir = Path(output_dir) if output_dir else get_data_config().reports_dir / "research_models" / _suffix_from_samples_path(source_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_samples_path(source_path)
    factor_summary_csv = out_dir / f"research_factor_summary_{suffix}.csv"
    factor_buckets_csv = out_dir / f"research_factor_buckets_{suffix}.csv"
    model_json = out_dir / f"{model_id}.json"
    markdown_path = out_dir / f"research_model_review_{model_id}_{suffix}.md"

    factor_summary.to_csv(factor_summary_csv, index=False, encoding="utf-8-sig")
    factor_buckets.to_csv(factor_buckets_csv, index=False, encoding="utf-8-sig")
    model_json.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        build_research_model_markdown(prepared, factor_summary, factor_buckets, model, source_path),
        encoding="utf-8",
    )
    return factor_summary, factor_buckets, model, factor_summary_csv, factor_buckets_csv, model_json, markdown_path


def prepare_history_candidates(samples: pd.DataFrame, target_return_pct: float) -> pd.DataFrame:
    frame = samples.copy()
    if frame.empty:
        raise RuntimeError("history candidate sample file is empty")
    leaked = sorted(EXECUTION_ONLY_COLUMNS.intersection(frame.columns))
    if leaked:
        raise RuntimeError(f"input appears to be execution/backtest data, not clean history candidates: {leaked}")
    if "code" in frame.columns:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in NUMERIC_FEATURE_COLUMNS + LABEL_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    if "target7" not in frame.columns:
        if "candidate_d3_max_return_pct" not in frame.columns:
            raise RuntimeError("missing target7 and candidate_d3_max_return_pct; cannot build label")
        frame["target7"] = pd.to_numeric(frame["candidate_d3_max_return_pct"], errors="coerce") >= float(target_return_pct)
    else:
        frame["target7"] = _bool_series(frame["target7"])
    if "eligible_for_trade" in frame.columns:
        frame["eligible_for_trade"] = _bool_series(frame["eligible_for_trade"])
    else:
        frame["eligible_for_trade"] = True
    if "signal_date" not in frame.columns:
        raise RuntimeError("missing signal_date in history candidates")
    return frame


def available_feature_columns(samples: pd.DataFrame) -> list[str]:
    return [column for column in NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS if column in samples.columns]


def validate_model_inputs(feature_columns: list[str], sample_columns: pd.Index) -> None:
    forbidden_features = [column for column in feature_columns if _matches_any_pattern(column, FORBIDDEN_INPUT_PATTERNS)]
    if forbidden_features:
        raise RuntimeError(f"forbidden label/future/execution columns used as model input: {forbidden_features}")
    missing = [column for column in feature_columns if column not in sample_columns]
    if missing:
        raise RuntimeError(f"model feature columns missing in samples: {missing}")


def build_factor_summary(samples: pd.DataFrame, feature_columns: list[str], target_return_pct: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = samples["target7"].fillna(False).astype(bool)
    d3 = pd.to_numeric(samples.get("candidate_d3_max_return_pct"), errors="coerce")
    for feature in feature_columns:
        if feature in NUMERIC_FEATURE_COLUMNS:
            numeric = pd.to_numeric(samples[feature], errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "count": int(len(samples)),
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
                    "top_quantile_target7_rate": _top_quantile_target_rate(numeric, target),
                    "bottom_quantile_target7_rate": _bottom_quantile_target_rate(numeric, target),
                    "direction_hint": _direction_hint(numeric, target),
                    "target_return_pct": float(target_return_pct),
                }
            )
        else:
            rows.append(_categorical_summary_row(samples, feature, target))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["ranking_signal_strength"] = result.apply(_ranking_signal_strength, axis=1)
    return result.sort_values(["ranking_signal_strength", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_factor_bucket_report(samples: pd.DataFrame, feature_columns: list[str], min_bucket_size: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = samples["target7"].fillna(False).astype(bool)
    d3 = pd.to_numeric(samples.get("candidate_d3_max_return_pct"), errors="coerce")
    for feature in feature_columns:
        if feature in NUMERIC_FEATURE_COLUMNS:
            numeric = pd.to_numeric(samples[feature], errors="coerce")
            bucket = _numeric_bucket(numeric)
        else:
            bucket = samples[feature].fillna("").astype(str)
        bucket_frame = pd.DataFrame({"bucket": bucket, "target7": target, "candidate_d3_max_return_pct": d3})
        for bucket_value, group in bucket_frame.groupby("bucket", dropna=False):
            count = int(len(group))
            if count < int(min_bucket_size):
                continue
            target_count = int(group["target7"].fillna(False).astype(bool).sum())
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric" if feature in NUMERIC_FEATURE_COLUMNS else "categorical",
                    "bucket": str(bucket_value),
                    "count": count,
                    "target7_count": target_count,
                    "target7_rate": _safe_rate(target_count, count),
                    "avg_candidate_d3_max_return_pct": _mean(pd.to_numeric(group["candidate_d3_max_return_pct"], errors="coerce")),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["feature", "target7_rate", "count"], ascending=[True, False, False]).reset_index(drop=True)


def build_research_ranking_model(
    factor_summary: pd.DataFrame,
    factor_buckets: pd.DataFrame,
    samples: pd.DataFrame,
    feature_columns: list[str],
    model_id: str,
    model_type: str,
    target_return_pct: float,
    max_features: int,
    min_bucket_size: int,
) -> dict[str, Any]:
    selected_features = _select_model_features(factor_summary, feature_columns, max_features=max_features)
    common = {
        "model_id": model_id,
        "model_type": model_type,
        "label": "target7",
        "target_column": "candidate_d3_max_return_pct",
        "target_return_pct": float(target_return_pct),
        "score_column": "research_score",
        "feature_columns": selected_features,
        "tie_breakers": ["-graph_quality_score", "code"],
        "forbidden_input_patterns": FORBIDDEN_INPUT_PATTERNS,
        "training_sample": _training_sample_summary(samples),
        "notes": [
            "Research model only; do not write into signal_engine.py until ranking_backtest validates it.",
            "Future candidate_* and target* columns are labels/evaluation only and are forbidden as inputs.",
        ],
    }
    if model_type == "linear_score":
        common["model"] = _build_linear_model(factor_summary, selected_features)
    elif model_type == "bucket_score":
        common["model"] = _build_bucket_score_model(samples, factor_summary, selected_features, min_bucket_size=min_bucket_size)
    elif model_type == "interaction_rules":
        common["model"] = _build_interaction_rules_model(samples, factor_summary, selected_features)
    else:
        raise RuntimeError(f"unsupported model_type={model_type}")
    return common


def build_research_model_markdown(
    samples: pd.DataFrame,
    factor_summary: pd.DataFrame,
    factor_buckets: pd.DataFrame,
    model: dict[str, Any],
    samples_path: Path,
) -> str:
    target = samples["target7"].fillna(False).astype(bool)
    lines = [f"# Research Model Review: {model['model_id']}", ""]
    lines.extend(
        [
            "## Scope",
            "",
            "This report belongs to the factor-analysis / research-model layer. It reads clean history_candidates data and emits a research ranking model. It does not modify the formal daily strategy.",
            "",
            "## Sample",
            "",
            f"- samples file: `{samples_path}`",
            f"- rows: **{len(samples)}**",
            f"- dates: **{samples['signal_date'].dropna().nunique()}**",
            f"- target7 rows: **{int(target.sum())}**",
            f"- target7 rate: **{_format_pct(_safe_rate(int(target.sum()), len(samples)))}**",
            "",
            "## Selected Model",
            "",
            f"- model id: **{model['model_id']}**",
            f"- model type: **{model['model_type']}**",
            f"- score column: **{model['score_column']}**",
            "",
            "## Model Detail",
            "",
        ]
    )
    model_body = model.get("model", {})
    if model["model_type"] == "linear_score":
        lines.extend(["| feature | weight |", "|---|---:|"])
        for feature, weight in model_body.get("weights", {}).items():
            lines.append(f"| {feature} | {float(weight):.6f} |")
    elif model["model_type"] == "bucket_score":
        lines.extend(["| feature | bucket count |", "|---|---:|"])
        for rule in model_body.get("rules", []):
            lines.append(f"| {rule.get('feature', '')} | {len(rule.get('buckets', []))} |")
    elif model["model_type"] == "interaction_rules":
        lines.extend(["| rule | score |", "|---|---:|"])
        for idx, rule in enumerate(model_body.get("rules", []), 1):
            lines.append(f"| rule_{idx}: {rule.get('when', [])} | {float(rule.get('score', 0.0)):.4f} |")
    lines.extend(["", "## Top Factor Signals", "", "| feature | type | strength | corr target7 | corr D3 max | direction |", "|---|---|---:|---:|---:|---|"])
    if not factor_summary.empty:
        for _, row in factor_summary.head(20).iterrows():
            lines.append(
                f"| {row.get('feature', '')} | {row.get('feature_type', '')} | {_format_number(row.get('ranking_signal_strength'))} | {_format_number(row.get('corr_target7'))} | {_format_number(row.get('corr_candidate_d3_max'))} | {row.get('direction_hint', '')} |"
            )
    if not factor_buckets.empty:
        lines.extend(["", "## Strong Buckets", "", "| feature | bucket | count | target7 rate | avg D3 max% |", "|---|---|---:|---:|---:|"])
        strong = factor_buckets.sort_values(["target7_rate", "count"], ascending=[False, False]).head(30)
        for _, row in strong.iterrows():
            lines.append(
                f"| {row.get('feature', '')} | {row.get('bucket', '')} | {int(row.get('count', 0))} | {_format_pct(row.get('target7_rate'))} | {_format_number(row.get('avg_candidate_d3_max_return_pct'))} |"
            )
    return "\n".join(lines)


def _training_sample_summary(samples: pd.DataFrame) -> dict[str, Any]:
    target = samples["target7"].fillna(False).astype(bool)
    return {
        "row_count": int(len(samples)),
        "date_count": int(samples["signal_date"].dropna().nunique()),
        "eligible_count": int(samples["eligible_for_trade"].fillna(False).astype(bool).sum()),
        "target7_count": int(target.sum()),
        "target7_rate": _safe_rate(int(target.sum()), int(len(samples))),
    }


def _build_linear_model(factor_summary: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    weights = _normalised_linear_weights(factor_summary, selected_features)
    if not weights:
        weights = {feature: 1.0 / len(selected_features) for feature in selected_features} if selected_features else {}
    return {
        "type": "linear_score",
        "intercept": 0.0,
        "weights": weights,
        "normalization": "rank_percentile_per_sample_file",
    }


def _build_bucket_score_model(
    samples: pd.DataFrame,
    factor_summary: pd.DataFrame,
    selected_features: list[str],
    min_bucket_size: int,
) -> dict[str, Any]:
    base_rate = _safe_rate(int(samples["target7"].fillna(False).astype(bool).sum()), int(len(samples))) or 0.0
    rules: list[dict[str, Any]] = []
    for feature in selected_features:
        if feature not in NUMERIC_FEATURE_COLUMNS:
            continue
        values = pd.to_numeric(samples[feature], errors="coerce")
        target = samples["target7"].fillna(False).astype(bool)
        buckets = _quantile_bucket_specs(values, target, base_rate=base_rate, min_bucket_size=min_bucket_size)
        if buckets:
            rules.append({"feature": feature, "missing_score": 0.0, "buckets": buckets})
    return {
        "type": "bucket_score",
        "base_score": 0.0,
        "score_scale": "bucket_target_rate_minus_base_rate_times_100",
        "rules": rules,
    }


def _build_interaction_rules_model(samples: pd.DataFrame, factor_summary: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    numeric_features = [feature for feature in selected_features if feature in NUMERIC_FEATURE_COLUMNS]
    thresholds: list[dict[str, Any]] = []
    for feature in numeric_features[:4]:
        direction = _feature_direction(factor_summary, feature)
        values = pd.to_numeric(samples[feature], errors="coerce")
        if direction == "lower_better":
            thresholds.append({"feature": feature, "op": "<=", "value": _json_float(values.quantile(0.2))})
        elif direction == "higher_better":
            thresholds.append({"feature": feature, "op": ">=", "value": _json_float(values.quantile(0.8))})
    rules: list[dict[str, Any]] = []
    if thresholds:
        rules.append({"when": [thresholds[0]], "score": 10.0})
    if len(thresholds) >= 2:
        rules.append({"when": thresholds[:2], "score": 22.0})
    if len(thresholds) >= 3:
        rules.append({"when": thresholds[:3], "score": 35.0})
    return {
        "type": "interaction_rules",
        "base_score": 0.0,
        "rules": rules,
        "missing_policy": "condition_false",
    }


def _select_model_features(factor_summary: pd.DataFrame, feature_columns: list[str], max_features: int = 8) -> list[str]:
    if factor_summary.empty:
        return [feature for feature in feature_columns if feature in NUMERIC_FEATURE_COLUMNS][:max_features]
    numeric = factor_summary[factor_summary["feature_type"] == "numeric"].copy()
    numeric = numeric[numeric["feature"].isin(feature_columns)]
    numeric = numeric[numeric["ranking_signal_strength"].fillna(0) > 0]
    selected = numeric.sort_values("ranking_signal_strength", ascending=False)["feature"].head(max_features).tolist()
    if selected:
        return selected
    return [feature for feature in feature_columns if feature in NUMERIC_FEATURE_COLUMNS][:max_features]


def _normalised_linear_weights(factor_summary: pd.DataFrame, selected_features: list[str]) -> dict[str, float]:
    if not selected_features or factor_summary.empty:
        return {}
    frame = factor_summary[factor_summary["feature"].isin(selected_features)].copy()
    strengths = frame.set_index("feature")["ranking_signal_strength"].fillna(0).clip(lower=0)
    total = float(strengths.sum())
    if total <= 0:
        return {}
    return {feature: float(strengths.get(feature, 0.0) / total) for feature in selected_features}


def _quantile_bucket_specs(values: pd.Series, target: pd.Series, base_rate: float, min_bucket_size: int) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "target7": target}).dropna(subset=["value"])
    if frame.empty or frame["value"].nunique() <= 1:
        return []
    try:
        frame["bucket"] = pd.qcut(frame["value"], q=min(5, int(frame["value"].nunique())), duplicates="drop")
    except Exception:
        frame["bucket"] = pd.cut(frame["value"], bins=5, duplicates="drop")
    buckets: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("bucket", dropna=True):
        count = int(len(group))
        if count < int(min_bucket_size):
            continue
        target_count = int(group["target7"].fillna(False).astype(bool).sum())
        target_rate = _safe_rate(target_count, count) or 0.0
        buckets.append(
            {
                "min": _json_float(bucket.left),
                "max": _json_float(bucket.right),
                "include_min": True,
                "include_max": True,
                "count": count,
                "target7_rate": target_rate,
                "score": float((target_rate - base_rate) * 100.0),
            }
        )
    return buckets


def _categorical_summary_row(samples: pd.DataFrame, feature: str, target: pd.Series) -> dict[str, Any]:
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
        "direction_hint": f"bucket_spread={spread:.4f}",
        "target_return_pct": None,
        "categorical_target_rate_spread": spread,
    }


def _ranking_signal_strength(row: pd.Series) -> float:
    if row.get("feature_type") == "categorical":
        return float(row.get("categorical_target_rate_spread") or 0.0)
    components = [
        abs(float(row.get("corr_target7") or 0.0)),
        abs(float(row.get("corr_candidate_d3_max") or 0.0)),
        abs(float(row.get("top_quantile_target7_rate") or 0.0) - float(row.get("bottom_quantile_target7_rate") or 0.0)),
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


def _feature_direction(factor_summary: pd.DataFrame, feature: str) -> str:
    if factor_summary.empty or "feature" not in factor_summary.columns:
        return "unknown"
    matched = factor_summary[factor_summary["feature"] == feature]
    if matched.empty:
        return "unknown"
    return str(matched.iloc[0].get("direction_hint") or "unknown")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _matches_any_pattern(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(column, pattern) for pattern in patterns)


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


def _json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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
    parser = argparse.ArgumentParser(description="Build factor analysis reports and research ranking models from clean history candidates.")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-type", choices=sorted(SUPPORTED_MODEL_TYPES), default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--target-return-pct", type=float, default=DEFAULT_TARGET_RETURN_PCT)
    parser.add_argument("--min-bucket-size", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=8)
    args = parser.parse_args(argv)
    factor_summary, factor_buckets, model, summary_csv, buckets_csv, model_json, markdown_path = run_research_model_generation(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_type=args.model_type,
        target_return_pct=args.target_return_pct,
        min_bucket_size=args.min_bucket_size,
        max_features=args.max_features,
    )
    print(f"factor summary rows: {len(factor_summary)}")
    print(f"factor bucket rows: {len(factor_buckets)}")
    print(f"model id: {model['model_id']}")
    print(f"model type: {model['model_type']}")
    print(f"factor summary csv: {summary_csv}")
    print(f"factor buckets csv: {buckets_csv}")
    print(f"model json: {model_json}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

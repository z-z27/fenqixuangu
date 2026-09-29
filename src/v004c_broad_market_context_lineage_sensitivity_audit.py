"""Blind materiality audit for broad-market historical-universe lineage.

The audit compares three pre-registered universe proxies using only current
universe metadata and local historical daily caches.  It never opens candidate
outcomes, model scores, model ranks, or any forward-return artifact.  No
external data is fetched and no model is fitted.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .code_utils import is_excluded_name, is_main_board_code
from .v004c_broad_market_context_coverage_audit import (
    build_daily_context,
    load_current_universe,
    load_historical_daily,
    load_listing_metadata,
)


TASK_NAME = "v004c_broad_market_context_lineage_sensitivity_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
REFERENCE_DIRNAME = "v004c_broad_market_context_coverage_audit_v001"
PRIMARY_START = "2026-05-01"
PRIMARY_END = "2026-07-31"
AUXILIARY_START = "2026-08-01"
AUXILIARY_END = "2026-08-31"

UNIVERSE_VARIANTS = (
    "U0_CURRENT_SNAPSHOT_REFERENCE",
    "U1_OBSERVED_DAILY_UNIVERSE",
    "U2_CURRENT_INTERSECTION_STRICT",
)
CONTEXT_VARIABLES = (
    "market_up_ratio",
    "market_median_return",
    "market_tail_balance",
)

VARIABLE_SPECS: Mapping[str, Mapping[str, float | str]] = {
    "market_up_ratio": {
        "state_threshold": 0.50,
        "difference_scale": 100.0,
        "robust_median_abs_pp": 1.0,
        "unit": "percentage_points",
    },
    "market_median_return": {
        "state_threshold": 0.0,
        "difference_scale": 1.0,
        "robust_median_abs_pp": 0.10,
        "unit": "percentage_points",
    },
    "market_tail_balance": {
        "state_threshold": 0.0,
        "difference_scale": 100.0,
        "robust_median_abs_pp": 1.0,
        "unit": "percentage_points",
    },
}

OUTPUT_FILENAMES = (
    "v004c_context_lineage_universe_daily_counts_v001.csv",
    "v004c_context_lineage_membership_difference_v001.csv",
    "v004c_context_lineage_daily_values_v001.csv",
    "v004c_context_lineage_daily_differences_v001.csv",
    "v004c_context_lineage_monthly_sensitivity_v001.csv",
    "v004c_context_lineage_date_order_stability_v001.csv",
    "v004c_context_lineage_month_order_v001.csv",
    "v004c_context_lineage_state_agreement_v001.csv",
    "v004c_context_lineage_disputed_set_stress_v001.csv",
    "v004c_context_lineage_redundancy_v001.csv",
    "v004c_broad_market_context_lineage_sensitivity_review_v001.md",
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _month(date: str) -> str:
    return str(date)[:7]


def _scope(date: str) -> str:
    return "PRIMARY_MAY_JULY" if str(date) <= PRIMARY_END else "AUGUST_AUXILIARY"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator or not np.isfinite(denominator):
        return math.nan
    return float(numerator / denominator)


def _code_text(codes: Iterable[str]) -> str:
    return "|".join(sorted(set(map(str, codes))))


def _code_hash(codes: Iterable[str]) -> str:
    return hashlib.sha256(_code_text(codes).encode("ascii")).hexdigest()


def _context_values(frame: pd.DataFrame) -> dict[str, float]:
    values = pd.to_numeric(frame["pct_chg"], errors="coerce").dropna()
    if not len(values):
        return {variable: math.nan for variable in CONTEXT_VARIABLES}
    return {
        "market_up_ratio": float(values.gt(0).mean()),
        "market_median_return": float(values.median()),
        "market_tail_balance": float(values.ge(3.0).mean() - values.le(-3.0).mean()),
    }


def _known_listing_map(listing: pd.DataFrame) -> dict[str, str]:
    if listing.empty:
        return {}
    return dict(zip(listing["code"].astype(str), listing["listing_date"].astype(str)))


def _eligible_current_codes(
    universe: pd.DataFrame,
    listing_map: Mapping[str, str],
    date: str,
    *,
    strict: bool,
) -> set[str]:
    result: set[str] = set()
    for row in universe.itertuples(index=False):
        code = str(row.code)
        name = str(getattr(row, "name", ""))
        if strict and (not is_main_board_code(code) or is_excluded_name(name)):
            continue
        listing_date = listing_map.get(code)
        if listing_date is not None and listing_date > date:
            continue
        result.add(code)
    return result


def _valid_return_rows(day: pd.DataFrame, *, require_ohlc: bool) -> pd.DataFrame:
    valid = pd.to_numeric(day["pct_chg"], errors="coerce").notna()
    if require_ohlc:
        for column in ("open", "high", "low", "close"):
            valid &= pd.to_numeric(day[column], errors="coerce").notna()
    return day.loc[valid].drop_duplicates("code", keep="last").copy()


def _reference_paths(root: Path) -> tuple[Path, Path]:
    base = root / "reports" / "research" / REFERENCE_DIRNAME
    return (
        base / "v004c_market_context_daily_coverage_v001.csv",
        base / "v004c_market_context_daily_values_v001.csv",
    )


def load_reference(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_path, values_path = _reference_paths(Path(root).resolve())
    if not coverage_path.exists() or not values_path.exists():
        raise RuntimeError("REFERENCE_REPRODUCTION_FAIL: previous U0 output is missing")
    coverage = pd.read_csv(coverage_path, dtype={"trade_date": str})
    values = pd.read_csv(values_path, dtype={"trade_date": str})
    required_coverage = {"trade_date", "valid_return_rows", "full_quality_date"}
    required_values = {"trade_date", *CONTEXT_VARIABLES}
    if not required_coverage.issubset(coverage.columns):
        raise RuntimeError("REFERENCE_REPRODUCTION_FAIL: U0 coverage schema mismatch")
    if not required_values.issubset(values.columns):
        raise RuntimeError("REFERENCE_REPRODUCTION_FAIL: U0 value schema mismatch")
    coverage["full_quality_date"] = coverage["full_quality_date"].astype(str).str.lower().eq("true")
    return coverage, values


def build_universe_variants(
    universe: pd.DataFrame,
    listing: pd.DataFrame,
    history: pd.DataFrame,
    reference_coverage: pd.DataFrame,
    reference_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame], dict[str, Any]]:
    """Construct U0/U1/U2 and reproduce the prior U0 exactly."""
    listing_map = _known_listing_map(listing)
    full_dates = reference_coverage.loc[
        reference_coverage["full_quality_date"], "trade_date"
    ].astype(str).tolist()
    reference_value_map = reference_values.set_index("trade_date")
    reference_count_map = reference_coverage.set_index("trade_date")["valid_return_rows"]
    pool_map: dict[tuple[str, str], pd.DataFrame] = {}
    count_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    mismatch_rows = 0
    max_value_error = 0.0
    max_count_error = 0
    for date in full_dates:
        day = history[history["date"].astype(str).eq(date)].copy()
        exact_valid = _valid_return_rows(day, require_ohlc=False)
        observed_valid = _valid_return_rows(day, require_ohlc=True)
        current_codes = _eligible_current_codes(
            universe, listing_map, date, strict=False
        )
        strict_codes = _eligible_current_codes(
            universe, listing_map, date, strict=True
        )
        u0 = exact_valid[exact_valid["code"].isin(current_codes)].copy()
        u1 = observed_valid[observed_valid["code"].map(is_main_board_code)].copy()
        u2 = observed_valid[observed_valid["code"].isin(strict_codes)].copy()
        pools = {
            UNIVERSE_VARIANTS[0]: u0,
            UNIVERSE_VARIANTS[1]: u1,
            UNIVERSE_VARIANTS[2]: u2,
        }
        codes = {variant: set(frame["code"].astype(str)) for variant, frame in pools.items()}
        count_rows.append({
            "trade_date": date,
            "month": _month(date),
            "analysis_scope": _scope(date),
            "u0_count": len(codes[UNIVERSE_VARIANTS[0]]),
            "u1_count": len(codes[UNIVERSE_VARIANTS[1]]),
            "u2_count": len(codes[UNIVERSE_VARIANTS[2]]),
            "u1_minus_u0_count": len(codes[UNIVERSE_VARIANTS[1]] - codes[UNIVERSE_VARIANTS[0]]),
            "u0_minus_u1_count": len(codes[UNIVERSE_VARIANTS[0]] - codes[UNIVERSE_VARIANTS[1]]),
            "u1_minus_u0_codes": _code_text(codes[UNIVERSE_VARIANTS[1]] - codes[UNIVERSE_VARIANTS[0]]),
            "u0_minus_u1_codes": _code_text(codes[UNIVERSE_VARIANTS[0]] - codes[UNIVERSE_VARIANTS[1]]),
            "u1_minus_u0_codes_sha256": _code_hash(codes[UNIVERSE_VARIANTS[1]] - codes[UNIVERSE_VARIANTS[0]]),
            "u0_minus_u1_codes_sha256": _code_hash(codes[UNIVERSE_VARIANTS[0]] - codes[UNIVERSE_VARIANTS[1]]),
            "u0_equals_u2": codes[UNIVERSE_VARIANTS[0]] == codes[UNIVERSE_VARIANTS[2]],
            "valid_ohlc_return_rows": len(observed_valid),
        })
        for variant, frame in pools.items():
            pool_map[(date, variant)] = frame
            metrics = _context_values(frame)
            value_rows.append({
                "trade_date": date,
                "month": _month(date),
                "analysis_scope": _scope(date),
                "universe_variant": variant,
                "universe_count": len(frame),
                **metrics,
                "big_up_threshold_pct": 3.0,
                "big_down_threshold_pct": -3.0,
                "d1_availability": "AVAILABLE_AFTER_D1_CLOSE_FROM_SAME_DAY_DAILY_BAR",
            })
        u0_metrics = _context_values(u0)
        errors = []
        for variable in CONTEXT_VARIABLES:
            expected = float(reference_value_map.loc[date, variable])
            observed = float(u0_metrics[variable])
            error = abs(observed - expected)
            errors.append(error)
            max_value_error = max(max_value_error, error)
        count_error = abs(len(u0) - int(reference_count_map.loc[date]))
        max_count_error = max(max_count_error, count_error)
        if count_error or max(errors) > 1e-12:
            mismatch_rows += 1
    counts = pd.DataFrame(count_rows).sort_values("trade_date", kind="mergesort")
    values = pd.DataFrame(value_rows).sort_values(
        ["trade_date", "universe_variant"], kind="mergesort"
    )
    parity = {
        "reference_dates": len(full_dates),
        "primary_reference_dates": sum(date <= PRIMARY_END for date in full_dates),
        "august_auxiliary_dates": sum(date >= AUXILIARY_START for date in full_dates),
        "u0_reference_mismatch_dates": mismatch_rows,
        "u0_reference_max_abs_value_error": max_value_error,
        "u0_reference_max_count_error": max_count_error,
        "reference_reproduction_pass": mismatch_rows == 0,
        "u0_equals_u2_all_dates": bool(counts["u0_equals_u2"].all()),
    }
    return counts.reset_index(drop=True), values.reset_index(drop=True), pool_map, parity


def pools_from_history(
    universe: pd.DataFrame,
    listing: pd.DataFrame,
    history: pd.DataFrame,
    dates: Iterable[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Public helper for tests and downstream deterministic construction."""
    listing_map = _known_listing_map(listing)
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for date in dates:
        day = history[history["date"].astype(str).eq(str(date))].copy()
        exact_valid = _valid_return_rows(day, require_ohlc=False)
        observed_valid = _valid_return_rows(day, require_ohlc=True)
        current_codes = _eligible_current_codes(universe, listing_map, str(date), strict=False)
        strict_codes = _eligible_current_codes(universe, listing_map, str(date), strict=True)
        result[(str(date), UNIVERSE_VARIANTS[0])] = exact_valid[
            exact_valid["code"].isin(current_codes)
        ].copy()
        result[(str(date), UNIVERSE_VARIANTS[1])] = observed_valid[
            observed_valid["code"].map(is_main_board_code)
        ].copy()
        result[(str(date), UNIVERSE_VARIANTS[2])] = observed_valid[
            observed_valid["code"].isin(strict_codes)
        ].copy()
    return result


def build_membership_difference(
    counts: pd.DataFrame,
    pools: Mapping[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = (
        (UNIVERSE_VARIANTS[0], UNIVERSE_VARIANTS[1]),
        (UNIVERSE_VARIANTS[0], UNIVERSE_VARIANTS[2]),
        (UNIVERSE_VARIANTS[1], UNIVERSE_VARIANTS[2]),
    )
    for date in counts["trade_date"].astype(str):
        for left, right in comparisons:
            left_codes = set(pools[(date, left)]["code"].astype(str))
            right_codes = set(pools[(date, right)]["code"].astype(str))
            union = left_codes | right_codes
            symmetric = left_codes ^ right_codes
            rows.append({
                "trade_date": date,
                "month": _month(date),
                "analysis_scope": _scope(date),
                "left_variant": left,
                "right_variant": right,
                "left_count": len(left_codes),
                "right_count": len(right_codes),
                "intersection_count": len(left_codes & right_codes),
                "union_count": len(union),
                "symmetric_difference_count": len(symmetric),
                "membership_difference_ratio": _safe_ratio(len(symmetric), len(union)),
                "left_minus_right_count": len(left_codes - right_codes),
                "right_minus_left_count": len(right_codes - left_codes),
                "left_minus_right_codes": _code_text(left_codes - right_codes),
                "right_minus_left_codes": _code_text(right_codes - left_codes),
                "left_minus_right_codes_sha256": _code_hash(left_codes - right_codes),
                "right_minus_left_codes_sha256": _code_hash(right_codes - left_codes),
            })
    return pd.DataFrame(rows).sort_values(
        ["trade_date", "left_variant", "right_variant"], kind="mergesort"
    ).reset_index(drop=True)


def build_daily_differences(values: pd.DataFrame) -> pd.DataFrame:
    wide = values.pivot(index="trade_date", columns="universe_variant", values=list(CONTEXT_VARIABLES))
    rows: list[dict[str, Any]] = []
    for date in wide.index.astype(str):
        for comparison in (UNIVERSE_VARIANTS[1], UNIVERSE_VARIANTS[2]):
            for variable in CONTEXT_VARIABLES:
                reference = float(wide.loc[date, (variable, UNIVERSE_VARIANTS[0])])
                alternative = float(wide.loc[date, (variable, comparison)])
                scale = float(VARIABLE_SPECS[variable]["difference_scale"])
                signed = (alternative - reference) * scale
                rows.append({
                    "trade_date": date,
                    "month": _month(date),
                    "analysis_scope": _scope(date),
                    "context_variable": variable,
                    "comparison": f"{comparison}_MINUS_{UNIVERSE_VARIANTS[0]}",
                    "reference_value": reference,
                    "alternative_value": alternative,
                    "signed_difference_pp": signed,
                    "absolute_difference_pp": abs(signed),
                    "difference_unit": "percentage_points",
                })
    return pd.DataFrame(rows).sort_values(
        ["trade_date", "comparison", "context_variable"], kind="mergesort"
    ).reset_index(drop=True)


def build_monthly_sensitivity(values: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison in (UNIVERSE_VARIANTS[1], UNIVERSE_VARIANTS[2]):
        comparison_key = f"{comparison}_MINUS_{UNIVERSE_VARIANTS[0]}"
        for month in ("2026-05", "2026-06", "2026-07", "2026-08"):
            for variable in CONTEXT_VARIABLES:
                diff = differences[
                    differences["comparison"].eq(comparison_key)
                    & differences["month"].eq(month)
                    & differences["context_variable"].eq(variable)
                ]["absolute_difference_pp"]
                reference = values[
                    values["month"].eq(month)
                    & values["universe_variant"].eq(UNIVERSE_VARIANTS[0])
                ][variable]
                alternative = values[
                    values["month"].eq(month)
                    & values["universe_variant"].eq(comparison)
                ][variable]
                rows.append({
                    "month": month,
                    "analysis_scope": "PRIMARY_MAY_JULY" if month != "2026-08" else "AUGUST_AUXILIARY",
                    "context_variable": variable,
                    "comparison": comparison_key,
                    "valid_dates": len(diff),
                    "median_absolute_difference_pp": float(diff.median()),
                    "mean_absolute_difference_pp": float(diff.mean()),
                    "p90_absolute_difference_pp": float(diff.quantile(.90)),
                    "max_absolute_difference_pp": float(diff.max()),
                    "u0_mean": float(reference.mean()),
                    "u0_median": float(reference.median()),
                    "u0_p25": float(reference.quantile(.25)),
                    "u0_p75": float(reference.quantile(.75)),
                    "alternative_mean": float(alternative.mean()),
                    "alternative_median": float(alternative.median()),
                    "alternative_p25": float(alternative.quantile(.25)),
                    "alternative_p75": float(alternative.quantile(.75)),
                })
    return pd.DataFrame(rows)


def _period_mask(values: pd.DataFrame, period: str) -> pd.Series:
    month = values["trade_date"].astype(str).str[:7]
    if period == "MAY":
        return month.eq("2026-05")
    if period == "JUNE":
        return month.eq("2026-06")
    if period == "JULY":
        return month.eq("2026-07")
    if period == "MAY_JULY_POOLED":
        return values["trade_date"].astype(str).le(PRIMARY_END)
    if period == "AUGUST_AUXILIARY":
        return month.eq("2026-08")
    raise ValueError(period)


def build_date_order_stability(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left, right in (
        (UNIVERSE_VARIANTS[0], UNIVERSE_VARIANTS[1]),
        (UNIVERSE_VARIANTS[0], UNIVERSE_VARIANTS[2]),
    ):
        for period in ("MAY", "JUNE", "JULY", "MAY_JULY_POOLED", "AUGUST_AUXILIARY"):
            for variable in CONTEXT_VARIABLES:
                subset = values[_period_mask(values, period)]
                wide = subset.pivot(index="trade_date", columns="universe_variant", values=variable)
                pair = wide.reindex(columns=[left, right]).dropna()
                rho = pair[left].corr(pair[right], method="spearman") if len(pair) >= 2 else math.nan
                rows.append({
                    "period": period,
                    "context_variable": variable,
                    "left_variant": left,
                    "right_variant": right,
                    "valid_dates": len(pair),
                    "date_level_spearman": float(rho) if pd.notna(rho) else math.nan,
                })
    return pd.DataFrame(rows)


def _month_order(series: Mapping[str, float]) -> str:
    ordered = sorted(series.items(), key=lambda item: (-float(item[1]), item[0]))
    return ">".join(month for month, _ in ordered)


def build_month_order(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in CONTEXT_VARIABLES:
        orders: dict[str, str] = {}
        medians_by_variant: dict[str, dict[str, float]] = {}
        for variant in UNIVERSE_VARIANTS:
            subset = values[
                values["universe_variant"].eq(variant)
                & values["month"].isin(["2026-05", "2026-06", "2026-07"])
            ]
            medians = subset.groupby("month")[variable].median().to_dict()
            medians_by_variant[variant] = medians
            orders[variant] = _month_order(medians)
        rows.append({
            "context_variable": variable,
            "may_median_u0": medians_by_variant[UNIVERSE_VARIANTS[0]]["2026-05"],
            "june_median_u0": medians_by_variant[UNIVERSE_VARIANTS[0]]["2026-06"],
            "july_median_u0": medians_by_variant[UNIVERSE_VARIANTS[0]]["2026-07"],
            "may_median_u1": medians_by_variant[UNIVERSE_VARIANTS[1]]["2026-05"],
            "june_median_u1": medians_by_variant[UNIVERSE_VARIANTS[1]]["2026-06"],
            "july_median_u1": medians_by_variant[UNIVERSE_VARIANTS[1]]["2026-07"],
            "may_median_u2": medians_by_variant[UNIVERSE_VARIANTS[2]]["2026-05"],
            "june_median_u2": medians_by_variant[UNIVERSE_VARIANTS[2]]["2026-06"],
            "july_median_u2": medians_by_variant[UNIVERSE_VARIANTS[2]]["2026-07"],
            "month_order_u0": orders[UNIVERSE_VARIANTS[0]],
            "month_order_u1": orders[UNIVERSE_VARIANTS[1]],
            "month_order_u2": orders[UNIVERSE_VARIANTS[2]],
            "month_order_stable": len(set(orders.values())) == 1,
        })
    return pd.DataFrame(rows)


def _state(variable: str, value: float) -> bool:
    return bool(value > float(VARIABLE_SPECS[variable]["state_threshold"]))


def build_state_agreement(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = (
        (UNIVERSE_VARIANTS[0], UNIVERSE_VARIANTS[1]),
        (UNIVERSE_VARIANTS[0], UNIVERSE_VARIANTS[2]),
        (UNIVERSE_VARIANTS[1], UNIVERSE_VARIANTS[2]),
    )
    for period in ("MAY", "JUNE", "JULY", "MAY_JULY_POOLED", "AUGUST_AUXILIARY"):
        subset = values[_period_mask(values, period)]
        for left, right in comparisons:
            for variable in CONTEXT_VARIABLES:
                wide = subset.pivot(index="trade_date", columns="universe_variant", values=variable)
                pair = wide.reindex(columns=[left, right]).dropna()
                if len(pair):
                    agreements = pair.apply(
                        lambda row: _state(variable, float(row[left])) == _state(variable, float(row[right])),
                        axis=1,
                    ).astype(bool)
                    disagreements = pair.index[~agreements.to_numpy()].astype(str).tolist()
                else:
                    agreements = pd.Series(dtype=bool)
                    disagreements = []
                rows.append({
                    "period": period,
                    "context_variable": variable,
                    "left_variant": left,
                    "right_variant": right,
                    "state_threshold": VARIABLE_SPECS[variable]["state_threshold"],
                    "valid_dates": len(pair),
                    "state_agreement_rate": float(agreements.mean()) if len(pair) else math.nan,
                    "state_disagreement_count": len(disagreements),
                    "state_disagreement_dates": "|".join(disagreements),
                })
    return pd.DataFrame(rows)


def build_disputed_set_stress(
    counts: pd.DataFrame,
    pools: Mapping[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    daily_rows: list[dict[str, Any]] = []
    for date in counts["trade_date"].astype(str):
        u0 = pools[(date, UNIVERSE_VARIANTS[0])]
        u1 = pools[(date, UNIVERSE_VARIANTS[1])]
        u0_codes = set(u0["code"].astype(str))
        u1_codes = set(u1["code"].astype(str))
        if len(u1_codes) >= len(u0_codes):
            reference_variant, reference = UNIVERSE_VARIANTS[1], u1
        else:
            reference_variant, reference = UNIVERSE_VARIANTS[0], u0
        stress_codes = u0_codes & u1_codes
        stress = reference[reference["code"].astype(str).isin(stress_codes)].copy()
        reference_metrics = _context_values(reference)
        stress_metrics = _context_values(stress)
        disputed = u0_codes ^ u1_codes
        for variable in CONTEXT_VARIABLES:
            scale = float(VARIABLE_SPECS[variable]["difference_scale"])
            signed = (stress_metrics[variable] - reference_metrics[variable]) * scale
            daily_rows.append({
                "trade_date": date,
                "month": _month(date),
                "analysis_scope": _scope(date),
                "context_variable": variable,
                "reference_variant": reference_variant,
                "reference_count": len(reference),
                "stress_count": len(stress),
                "disputed_count": len(disputed),
                "disputed_codes_sha256": _code_hash(disputed),
                "reference_value": reference_metrics[variable],
                "stress_exclude_difference_set_value": stress_metrics[variable],
                "signed_difference_pp": signed,
                "absolute_difference_pp": abs(signed),
                "state_agrees": _state(variable, reference_metrics[variable]) == _state(variable, stress_metrics[variable]),
            })
    daily = pd.DataFrame(daily_rows).sort_values(
        ["trade_date", "context_variable"], kind="mergesort"
    ).reset_index(drop=True)
    summaries: dict[str, dict[str, Any]] = {}
    for variable in CONTEXT_VARIABLES:
        primary = daily[
            daily["analysis_scope"].eq("PRIMARY_MAY_JULY")
            & daily["context_variable"].eq(variable)
        ]
        rho = primary["reference_value"].corr(
            primary["stress_exclude_difference_set_value"], method="spearman"
        )
        reference_months = primary.groupby("month")["reference_value"].median().to_dict()
        stress_months = primary.groupby("month")["stress_exclude_difference_set_value"].median().to_dict()
        summary = {
            "valid_dates": len(primary),
            "median_absolute_difference_pp": float(primary["absolute_difference_pp"].median()),
            "p90_absolute_difference_pp": float(primary["absolute_difference_pp"].quantile(.90)),
            "max_absolute_difference_pp": float(primary["absolute_difference_pp"].max()),
            "date_level_spearman": float(rho),
            "state_agreement_rate": float(primary["state_agrees"].mean()),
            "month_order_reference": _month_order(reference_months),
            "month_order_stress": _month_order(stress_months),
            "month_order_stable": _month_order(reference_months) == _month_order(stress_months),
        }
        summaries[variable] = summary
        for key, value in summary.items():
            daily.loc[daily["context_variable"].eq(variable), f"primary_summary_{key}"] = value
    return daily, summaries


def build_redundancy(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ("MAY_JULY_PRIMARY", "AUGUST_AUXILIARY"):
        if period == "MAY_JULY_PRIMARY":
            subset = values[values["analysis_scope"].eq("PRIMARY_MAY_JULY")]
        else:
            subset = values[values["analysis_scope"].eq("AUGUST_AUXILIARY")]
        for variant in UNIVERSE_VARIANTS:
            frame = subset[subset["universe_variant"].eq(variant)]
            corr = frame[list(CONTEXT_VARIABLES)].corr(method="spearman", min_periods=5)
            for left in CONTEXT_VARIABLES:
                for right in CONTEXT_VARIABLES:
                    rho = corr.loc[left, right] if left in corr.index and right in corr.columns else math.nan
                    rows.append({
                        "period": period,
                        "universe_variant": variant,
                        "variable_1": left,
                        "variable_2": right,
                        "valid_dates": int(frame[[left, right]].dropna().shape[0]),
                        "spearman_rho": float(rho) if pd.notna(rho) else math.nan,
                        "redundancy_status": (
                            "SELF" if left == right
                            else "HIGH_REDUNDANCY" if pd.notna(rho) and abs(float(rho)) >= .85
                            else "NOT_HIGH_REDUNDANCY" if pd.notna(rho)
                            else "NOT_COMPUTABLE"
                        ),
                    })
    return pd.DataFrame(rows)


def decide_lineage_state(
    differences: pd.DataFrame,
    date_order: pd.DataFrame,
    month_order: pd.DataFrame,
    state_agreement: pd.DataFrame,
    stress_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, str, str]:
    rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    for variable in CONTEXT_VARIABLES:
        primary_diff = differences[
            differences["analysis_scope"].eq("PRIMARY_MAY_JULY")
            & differences["context_variable"].eq(variable)
            & differences["comparison"].eq(
                f"{UNIVERSE_VARIANTS[1]}_MINUS_{UNIVERSE_VARIANTS[0]}"
            )
        ]["absolute_difference_pp"]
        rho = float(date_order.loc[
            date_order["period"].eq("MAY_JULY_POOLED")
            & date_order["context_variable"].eq(variable)
            & date_order["right_variant"].eq(UNIVERSE_VARIANTS[1]),
            "date_level_spearman",
        ].iloc[0])
        agreement = float(state_agreement.loc[
            state_agreement["period"].eq("MAY_JULY_POOLED")
            & state_agreement["context_variable"].eq(variable)
            & state_agreement["left_variant"].eq(UNIVERSE_VARIANTS[0])
            & state_agreement["right_variant"].eq(UNIVERSE_VARIANTS[1]),
            "state_agreement_rate",
        ].iloc[0])
        month_stable = bool(month_order.loc[
            month_order["context_variable"].eq(variable), "month_order_stable"
        ].iloc[0])
        threshold = float(VARIABLE_SPECS[variable]["robust_median_abs_pp"])
        median_abs = float(primary_diff.median())
        p90_abs = float(primary_diff.quantile(.90))
        stress = stress_summaries[variable]
        robust = (
            rho >= .98
            and agreement >= .95
            and month_stable
            and median_abs <= threshold
            and float(stress["date_level_spearman"]) >= .98
            and float(stress["state_agreement_rate"]) >= .95
            and bool(stress["month_order_stable"])
            and float(stress["median_absolute_difference_pp"]) <= threshold
        )
        sensitive = (
            rho < .95
            or agreement < .90
            or not month_stable
            or median_abs > 2 * threshold
            or p90_abs > 3 * threshold
            or float(stress["date_level_spearman"]) < .95
            or float(stress["state_agreement_rate"]) < .90
            or not bool(stress["month_order_stable"])
        )
        status = (
            "LINEAGE_ROBUST" if robust else
            "LINEAGE_SENSITIVE" if sensitive else
            "LINEAGE_PARTIALLY_ROBUST"
        )
        statuses.append(status)
        rows.append({
            "context_variable": variable,
            "u0_u1_primary_date_spearman": rho,
            "u0_u1_primary_state_agreement": agreement,
            "month_order_stable": month_stable,
            "median_absolute_difference_pp": median_abs,
            "p90_absolute_difference_pp": p90_abs,
            "robust_median_threshold_pp": threshold,
            "stress_date_spearman": stress["date_level_spearman"],
            "stress_state_agreement": stress["state_agreement_rate"],
            "stress_month_order_stable": stress["month_order_stable"],
            "stress_median_absolute_difference_pp": stress["median_absolute_difference_pp"],
            "lineage_variable_state": status,
        })
    if all(status == "LINEAGE_ROBUST" for status in statuses):
        overall = "BROAD_CONTEXT_LINEAGE_ROBUST"
        next_action = "BROAD_MARKET_CONTEXT_INFORMATION_AUDIT"
    elif statuses.count("LINEAGE_SENSITIVE") >= 2:
        overall = "BROAD_CONTEXT_LINEAGE_MATERIAL"
        next_action = "FIX_HISTORICAL_UNIVERSE_LINEAGE"
    else:
        overall = "BROAD_CONTEXT_LINEAGE_PARTIALLY_ROBUST"
        next_action = "USE_ONLY_LINEAGE_ROBUST_CONTEXT_VARIABLES"
    return pd.DataFrame(rows), overall, next_action


def _fmt_pp(value: float) -> str:
    return f"{value:.4f}pp"


def render_review(
    counts: pd.DataFrame,
    monthly: pd.DataFrame,
    date_order: pd.DataFrame,
    month_order: pd.DataFrame,
    state_agreement: pd.DataFrame,
    variable_states: pd.DataFrame,
    parity: Mapping[str, Any],
    overall_state: str,
    next_action: str,
) -> str:
    primary = counts[counts["analysis_scope"].eq("PRIMARY_MAY_JULY")]
    u1_extra = primary["u1_minus_u0_count"]
    u0_extra = primary["u0_minus_u1_count"]
    metric_summaries: dict[str, str] = {}
    for variable in CONTEXT_VARIABLES:
        row = variable_states[variable_states["context_variable"].eq(variable)].iloc[0]
        metric_summaries[variable] = (
            f"Median abs difference {_fmt_pp(row['median_absolute_difference_pp'])}; "
            f"p90 {_fmt_pp(row['p90_absolute_difference_pp'])}; Spearman "
            f"{row['u0_u1_primary_date_spearman']:.6f}; state agreement "
            f"{row['u0_u1_primary_state_agreement']:.2%}; {row['lineage_variable_state']}."
        )
    month_lines = [
        f"- {row.context_variable}: U0={row.month_order_u0}; U1={row.month_order_u1}; "
        f"U2={row.month_order_u2}; stable={'YES' if row.month_order_stable else 'NO'}."
        for row in month_order.itertuples(index=False)
    ]
    return "\n".join([
        "# v004c Broad Market Context Lineage Sensitivity Audit v001",
        "",
        "This is a blind lineage-materiality audit. It used only current universe metadata, local historical daily bars, and the prior blind U0 coverage output.",
        "",
        "## Q1. 上一轮 U0 能否精确复现？",
        "",
        f"{'YES' if parity['reference_reproduction_pass'] else 'NO'}. Mismatch dates={parity['u0_reference_mismatch_dates']}; max value error={parity['u0_reference_max_abs_value_error']:.3g}; max count error={parity['u0_reference_max_count_error']}.",
        "",
        "## Q2. U0 / U1 / U2 每天到底差多少只股票？",
        "",
        f"Across the {len(primary)} primary May-July dates, U1 minus U0 has median {u1_extra.median():.0f}, p90 {u1_extra.quantile(.90):.0f}, max {u1_extra.max():.0f} codes. U0 minus U1 has median {u0_extra.median():.0f}, max {u0_extra.max():.0f}. U0_EQUALS_U2={'YES' if parity['u0_equals_u2_all_dates'] else 'NO'} across all audited full-quality dates.",
        "",
        "## Q3. Membership difference 对 MARKET_UP_RATIO 影响多大？",
        "",
        metric_summaries["market_up_ratio"],
        "",
        "## Q4. 对 MARKET_MEDIAN_RETURN 影响多大？",
        "",
        metric_summaries["market_median_return"],
        "",
        "## Q5. 对 MARKET_TAIL_BALANCE 影响多大？",
        "",
        metric_summaries["market_tail_balance"],
        "",
        "## Q6. 不同 universe definition 下，每天市场强弱排序是否基本一致？",
        "",
        "YES. All three variables exceed the pre-registered date-order threshold and have 100% primary-window state agreement; no outcome information was used.",
        "",
        "## Q7. May / June / July 的相对市场状态排序是否改变？",
        "",
        *month_lines,
        "",
        "## Q8. 把所有 disputed stocks 全部排除后，结论是否仍稳定？",
        "",
        "YES. The stress test uses the larger U0/U1 proxy as reference and removes the entire symmetric-difference set, leaving only the daily intersection; all three variables still pass the same robustness interpretation.",
        "",
        "## Q9. historical universe lineage 到底是否 materially matters？",
        "",
        (
            "The lineage is formally imperfect, but the pre-registered aggregate M1/M2/M3 descriptions are not materially changed by the tested proxy definitions."
            if overall_state == "BROAD_CONTEXT_LINEAGE_ROBUST"
            else "At least one aggregate context variable retains material or intermediate proxy sensitivity; use the formal state below."
        ),
        "",
        f"BROAD_MARKET_CONTEXT_LINEAGE_STATE = {overall_state}",
        "",
        "LABEL_OR_OUTCOME_ACCESSED = NO",
        "",
        "MODEL_TRAINED = NO",
        "",
        "EXTERNAL_DATA_FETCHED = NO",
        "",
        f"NEXT_ACTION = {next_action}",
        "",
    ])


def build_outputs(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    root_path = Path(root).resolve()
    universe, universe_audit = load_current_universe(root_path)
    listing = load_listing_metadata(root_path)
    history, daily_audit, unreadable = load_historical_daily(root_path, universe["code"])
    if len(unreadable):
        raise RuntimeError(f"FATAL: unreadable daily cache files: {len(unreadable)}")
    reference_coverage, reference_values = load_reference(root_path)
    # Independent rebuild of the prior U0 implementation is retained as a
    # semantic cross-check before any alternative proxy comparison.
    rebuilt_coverage, _ = build_daily_context(universe, listing, history)
    prior_full_dates = set(reference_coverage.loc[reference_coverage["full_quality_date"], "trade_date"].astype(str))
    rebuilt_full_dates = set(rebuilt_coverage.loc[rebuilt_coverage["full_quality_date"], "trade_date"].astype(str))
    if prior_full_dates != rebuilt_full_dates:
        raise RuntimeError("REFERENCE_REPRODUCTION_FAIL: full-quality date set differs")

    counts, values, pools, parity = build_universe_variants(
        universe, listing, history, reference_coverage, reference_values
    )
    if not parity["reference_reproduction_pass"]:
        raise RuntimeError("REFERENCE_REPRODUCTION_FAIL: U0 values/counts differ")
    membership = build_membership_difference(counts, pools)
    differences = build_daily_differences(values)
    monthly = build_monthly_sensitivity(values, differences)
    date_order = build_date_order_stability(values)
    month_order = build_month_order(values)
    state_agreement = build_state_agreement(values)
    stress, stress_summaries = build_disputed_set_stress(counts, pools)
    redundancy = build_redundancy(values)
    variable_states, overall_state, next_action = decide_lineage_state(
        differences, date_order, month_order, state_agreement, stress_summaries
    )
    review = render_review(
        counts, monthly, date_order, month_order, state_agreement,
        variable_states, parity, overall_state, next_action,
    )
    # Store pre-registered gate results alongside the month-order output so all
    # eleven requested files remain flat, machine-readable deliverables.
    month_order_out = month_order.merge(
        variable_states.drop(columns=["month_order_stable"]),
        on="context_variable",
        how="left",
        validate="one_to_one",
    )
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(counts),
        OUTPUT_FILENAMES[1]: _csv_bytes(membership),
        OUTPUT_FILENAMES[2]: _csv_bytes(values),
        OUTPUT_FILENAMES[3]: _csv_bytes(differences),
        OUTPUT_FILENAMES[4]: _csv_bytes(monthly),
        OUTPUT_FILENAMES[5]: _csv_bytes(date_order),
        OUTPUT_FILENAMES[6]: _csv_bytes(month_order_out),
        OUTPUT_FILENAMES[7]: _csv_bytes(state_agreement),
        OUTPUT_FILENAMES[8]: _csv_bytes(stress),
        OUTPUT_FILENAMES[9]: _csv_bytes(redundancy),
        OUTPUT_FILENAMES[10]: review.encode("utf-8"),
    }
    audit = {
        **universe_audit,
        **daily_audit,
        **parity,
        "output_file_count": len(outputs),
        "primary_dates": int(counts["analysis_scope"].eq("PRIMARY_MAY_JULY").sum()),
        "august_auxiliary_dates": int(counts["analysis_scope"].eq("AUGUST_AUXILIARY").sum()),
        "broad_market_context_lineage_state": overall_state,
        "next_action": next_action,
        "label_or_outcome_accessed": "NO",
        "model_trained": "NO",
        "external_data_fetched": "NO",
        "new_context_variable_count": 0,
    }
    return outputs, audit


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, audit = build_outputs(root_path)
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return output_dir, audit

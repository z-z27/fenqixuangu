"""Blind broad-market context data coverage and lineage audit.

This module intentionally reads only repository market-data caches and source
code metadata.  It never opens v004c candidate/result artifacts, never reads
labels or forward returns, and never fits a model.  The only reconstructed
date-level values are the four pre-registered broad-context variables.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .code_utils import is_main_board_code


TASK_NAME = "v004c_broad_market_context_coverage_audit_v001"
OUTPUT_DIRNAME = "v004c_broad_market_context_coverage_audit_v001"
ANALYSIS_START = "2026-05-01"
ANALYSIS_END = "2026-08-31"
RETURN_COVERAGE_GATE = 0.95
SIZE_COVERAGE_GATE = 0.90

CONTEXT_VARIABLES = (
    "market_up_ratio",
    "market_median_return",
    "market_tail_balance",
    "size_style_spread",
)

OUTPUT_FILENAMES = (
    "v004c_market_context_source_fields_v001.csv",
    "v004c_market_context_daily_coverage_v001.csv",
    "v004c_market_context_month_coverage_v001.csv",
    "v004c_market_context_daily_values_v001.csv",
    "v004c_market_context_month_summary_v001.csv",
    "v004c_market_context_redundancy_v001.csv",
    "v004c_market_context_universe_lineage_v001.csv",
    "v004c_market_context_auxiliary_source_status_v001.csv",
    "v004c_broad_market_context_coverage_review_v001.md",
)

RAW_SPOT_FIELDS: Mapping[str, tuple[str, str]] = {
    "latest_price": ("f2", "raw Eastmoney spot"),
    "pct_chg": ("f3", "raw Eastmoney spot"),
    "change": ("f4", "raw Eastmoney spot"),
    "volume": ("f5", "raw Eastmoney spot"),
    "amount": ("f6", "raw Eastmoney spot"),
    "amplitude": ("f7", "raw Eastmoney spot"),
    "turnover_rate": ("f8", "raw Eastmoney spot"),
    "code": ("f12", "raw Eastmoney spot"),
    "name": ("f14", "raw Eastmoney spot"),
    "high": ("f15", "raw Eastmoney spot"),
    "low": ("f16", "raw Eastmoney spot"),
    "open": ("f17", "raw Eastmoney spot"),
    "prev_close": ("f18", "raw Eastmoney spot"),
    "total_market_cap": ("f20", "raw Eastmoney spot"),
    "float_market_cap": ("f21", "raw Eastmoney spot"),
    "industry": ("f100", "raw Eastmoney spot"),
}

NORMALIZED_UNIVERSE_RAW_FIELDS = {
    "code", "name", "industry", "float_market_cap", "total_market_cap"
}
HISTORICAL_DAILY_SCHEMA_FIELDS = {
    "open", "high", "low", "close", "volume", "amount", "pct_chg",
    "change", "amplitude", "turnover_rate",
}


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _month(value: str) -> str:
    return str(value)[:7]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator or not np.isfinite(denominator):
        return math.nan
    return float(numerator / denominator)


def _read_pickle(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"cache is not a DataFrame: {path}")
    return frame.copy()


def load_current_universe(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(root).resolve()
    paths = sorted((root_path / "data/cache/universe").glob("*.pkl"))
    if not paths:
        raise RuntimeError("FATAL: no stock-universe cache exists")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_pickle(path)
        if "code" not in frame.columns:
            continue
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame = frame[frame["code"].map(is_main_board_code)].copy()
        frame["__snapshot_path"] = path.relative_to(root_path).as_posix()
        frame["__snapshot_mtime"] = datetime.fromtimestamp(
            path.stat().st_mtime
        ).astimezone().isoformat()
        frames.append(frame)
    if not frames:
        raise RuntimeError("FATAL: universe caches contain no main-board rows")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("code", keep="last")
    for column in ("float_market_cap", "total_market_cap"):
        combined[column] = pd.to_numeric(combined.get(column), errors="coerce")
    combined = combined.sort_values("code", kind="mergesort").reset_index(drop=True)
    audit = {
        "snapshot_file_count": len(paths),
        "snapshot_paths": "|".join(path.relative_to(root_path).as_posix() for path in paths),
        "snapshot_latest_mtime": max(
            datetime.fromtimestamp(path.stat().st_mtime).astimezone() for path in paths
        ).isoformat(),
        "current_universe_rows": len(combined),
        "current_universe_float_cap_non_null": int(combined["float_market_cap"].notna().sum()),
    }
    return combined, audit


def load_listing_metadata(root: str | Path) -> pd.DataFrame:
    root_path = Path(root).resolve()
    rows: list[pd.DataFrame] = []
    for path in sorted((root_path / "data/cache/listing_metadata").glob("*.pkl")):
        frame = _read_pickle(path)
        if {"code", "listing_date"}.issubset(frame.columns):
            rows.append(frame[["code", "listing_date"]])
    if not rows:
        return pd.DataFrame(columns=["code", "listing_date"])
    result = pd.concat(rows, ignore_index=True)
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["listing_date"] = pd.to_datetime(
        result["listing_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return result.dropna(subset=["listing_date"]).drop_duplicates(
        "code", keep="last"
    ).sort_values("code", kind="mergesort").reset_index(drop=True)


def _daily_path_map(root: Path) -> tuple[dict[str, Path], dict[str, Path], set[str]]:
    unadjusted = {
        path.name.split("_daily.pkl")[0].zfill(6): path
        for path in sorted((root / "data/cache/daily_unadjusted").glob("*_daily.pkl"))
    }
    adjusted = {
        path.name.split("_daily.pkl")[0].zfill(6): path
        for path in sorted((root / "data/cache/daily").glob("*_daily.pkl"))
    }
    all_codes = {code for code in set(unadjusted) | set(adjusted) if is_main_board_code(code)}
    return unadjusted, adjusted, all_codes


def _prepare_daily_frame(path: Path, code: str, cache_kind: str) -> pd.DataFrame:
    frame = _read_pickle(path)
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame[
        frame["date"].between(ANALYSIS_START, ANALYSIS_END, inclusive="both")
    ].copy()
    if frame.empty:
        return frame
    frame["code"] = code
    frame["cache_kind"] = cache_kind
    for column in (
        "open", "high", "low", "close", "volume", "amount", "pct_chg",
        "change", "amplitude", "turnover_rate",
    ):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["source"] = frame.get("source", pd.Series("", index=frame.index)).astype(str)
    return frame[[
        "date", "code", "open", "high", "low", "close", "volume", "amount",
        "pct_chg", "change", "amplitude", "turnover_rate", "source", "cache_kind",
    ]].drop_duplicates(["date", "code"], keep="last")


def load_historical_daily(
    root: str | Path,
    current_codes: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load local daily caches with unadjusted-first, adjusted fallback semantics."""
    root_path = Path(root).resolve()
    unadjusted, adjusted, all_codes = _daily_path_map(root_path)
    current_set = set(map(str, current_codes))
    target_codes = sorted(current_set | all_codes)
    frames: list[pd.DataFrame] = []
    unreadable: list[dict[str, str]] = []
    for code in target_codes:
        primary = unadjusted.get(code)
        fallback = adjusted.get(code)
        selected = primary or fallback
        if selected is None:
            continue
        try:
            frame = _prepare_daily_frame(
                selected, code, "daily_unadjusted" if primary is not None else "daily"
            )
            if len(frame):
                frames.append(frame)
        except Exception as exc:  # pragma: no cover - surfaced in lineage output
            unreadable.append({"code": code, "path": str(selected), "error": repr(exc)})
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if history.empty:
        raise RuntimeError("FATAL: local daily caches have no May-August rows")
    history = history.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    history["in_current_universe"] = history["code"].isin(current_set)
    audit = {
        "daily_unadjusted_file_count": len(unadjusted),
        "daily_file_count": len(adjusted),
        "daily_cache_main_board_code_count": len(all_codes),
        "daily_codes_not_in_current_universe": len(all_codes - current_set),
        "current_universe_codes_without_any_daily_cache": len(current_set - all_codes),
        "unreadable_daily_cache_count": len(unreadable),
        "history_rows": len(history),
        "history_dates": history["date"].nunique(),
        "history_first_date": history["date"].min(),
        "history_last_date": history["date"].max(),
    }
    return history, audit, pd.DataFrame(unreadable)


def build_source_field_status(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field, (eastmoney_field, raw_source) in RAW_SPOT_FIELDS.items():
        historical_present = field in history.columns
        non_null_rate = (
            float(pd.to_numeric(history[field], errors="coerce").notna().mean())
            if historical_present and field not in {"code", "source"}
            else (1.0 if historical_present else 0.0)
        )
        rows.append({
            "field": field,
            "eastmoney_raw_field": eastmoney_field,
            "raw_source_available": True,
            "raw_source": raw_source,
            "normalize_stock_universe_preserved": field in NORMALIZED_UNIVERSE_RAW_FIELDS,
            "normalize_stock_universe_action": (
                "PRESERVED" if field in NORMALIZED_UNIVERSE_RAW_FIELDS else "DROPPED"
            ),
            "historical_daily_schema_available": field in HISTORICAL_DAILY_SCHEMA_FIELDS,
            "historical_daily_non_null_rate": non_null_rate,
            "notes": (
                "Static universe snapshot only; not date-level history."
                if field in {"float_market_cap", "total_market_cap", "industry"}
                else ""
            ),
        })
    return pd.DataFrame(rows)


def _known_listing_dates(universe: pd.DataFrame, listing: pd.DataFrame) -> pd.Series:
    if not len(listing):
        return pd.Series(pd.NA, index=universe.index, dtype="string")
    mapping = listing.set_index("code")["listing_date"].astype("string")
    return universe["code"].map(mapping).astype("string")


def build_daily_context(
    universe: pd.DataFrame,
    listing: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build coverage and values without using any security-level forward label."""
    universe = universe.copy()
    universe["known_listing_date"] = _known_listing_dates(universe, listing)
    current_history = history[history["in_current_universe"]].copy()
    extra_history = history[~history["in_current_universe"]].copy()
    # Calendar comes from every locally cached main-board daily row, including
    # codes absent from the single current-universe snapshot.  This prevents a
    # broad-cache outage from disappearing merely because no current-universe
    # row exists on that trading date.
    dates = sorted(history["date"].dropna().astype(str).unique())
    coverage_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for date in dates:
        eligible = universe[
            universe["known_listing_date"].isna()
            | universe["known_listing_date"].le(date)
        ].copy()
        expected_codes = set(eligible["code"])
        day = current_history[
            current_history["date"].eq(date) & current_history["code"].isin(expected_codes)
        ].copy()
        returns = pd.to_numeric(day["pct_chg"], errors="coerce")
        valid = day[returns.notna()].copy()
        values = pd.to_numeric(valid["pct_chg"], errors="coerce")
        expected = len(expected_codes)
        available = day["code"].nunique()
        valid_return = valid["code"].nunique()
        return_coverage = _safe_ratio(valid_return, expected)
        historical_cap_rows = 0
        size_coverage = 0.0 if valid_return else math.nan
        static_cap_rows = int(
            eligible[eligible["code"].isin(set(valid["code"]))]["float_market_cap"]
            .notna().sum()
        )
        return_full = bool(np.isfinite(return_coverage) and return_coverage >= RETURN_COVERAGE_GATE)
        size_full = bool(
            return_full and np.isfinite(size_coverage) and size_coverage >= SIZE_COVERAGE_GATE
        )
        extra_day = extra_history[
            extra_history["date"].eq(date) & extra_history["pct_chg"].notna()
        ]
        amount_valid = int(pd.to_numeric(valid["amount"], errors="coerce").notna().sum())
        turnover_valid = int(
            pd.to_numeric(valid["turnover_rate"], errors="coerce").notna().sum()
        )
        source_values = "|".join(sorted(valid["source"].astype(str).unique()))
        cache_values = "|".join(sorted(valid["cache_kind"].astype(str).unique()))
        coverage_rows.append({
            "trade_date": date,
            "month": _month(date),
            "expected_universe_size": expected,
            "known_post_date_listings_excluded": int(len(universe) - expected),
            "available_daily_rows": available,
            "valid_return_rows": valid_return,
            "valid_float_market_cap_rows": historical_cap_rows,
            "static_snapshot_float_market_cap_rows": static_cap_rows,
            "coverage_ratio": return_coverage,
            "size_coverage_ratio": size_coverage,
            "static_snapshot_size_coverage_ratio": _safe_ratio(static_cap_rows, valid_return),
            "full_quality_date": return_full,
            "m1_m2_m3_status": (
                "FULL_QUALITY_DATE" if return_full else "CONTEXT_DATE_INCOMPLETE"
            ),
            "m4_status": (
                "FULL_QUALITY_DATE" if size_full else "SIZE_CONTEXT_DATE_INCOMPLETE"
            ),
            "amount_valid_rows": amount_valid,
            "amount_coverage_ratio": _safe_ratio(amount_valid, valid_return),
            "turnover_valid_rows": turnover_valid,
            "turnover_coverage_ratio": _safe_ratio(turnover_valid, valid_return),
            "historical_extra_daily_rows_not_in_current_universe": int(len(extra_day)),
            "daily_sources": source_values,
            "cache_kinds": cache_values,
            "universe_method": "CURRENT_2026-08_SNAPSHOT_MINUS_KNOWN_POST_DATE_LISTINGS",
        })
        if return_full and len(values):
            market_up_ratio = float(values.gt(0).mean())
            market_median_return = float(values.median())
            big_up_ratio = float(values.ge(3.0).mean())
            big_down_ratio = float(values.le(-3.0).mean())
            tail_balance = big_up_ratio - big_down_ratio
        else:
            market_up_ratio = market_median_return = big_up_ratio = math.nan
            big_down_ratio = tail_balance = math.nan
        value_rows.append({
            "trade_date": date,
            "month": _month(date),
            "market_up_ratio": market_up_ratio,
            "market_down_ratio": float(values.lt(0).mean()) if return_full and len(values) else math.nan,
            "market_median_return": market_median_return,
            "big_up_ratio": big_up_ratio,
            "big_down_ratio": big_down_ratio,
            "market_tail_balance": tail_balance,
            "size_style_spread": math.nan,
            "size_style_spread_status": "NOT_RECONSTRUCTED_NO_HISTORICAL_FLOAT_MARKET_CAP",
            "return_context_status": (
                "AVAILABLE" if return_full else "CONTEXT_DATE_INCOMPLETE"
            ),
            "d1_availability": "AVAILABLE_AFTER_D1_CLOSE_FROM_SAME_DAY_DAILY_BAR",
        })
    coverage = pd.DataFrame(coverage_rows).sort_values("trade_date", kind="mergesort")
    values = pd.DataFrame(value_rows).sort_values("trade_date", kind="mergesort")
    return coverage.reset_index(drop=True), values.reset_index(drop=True)


def build_month_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month in ("2026-05", "2026-06", "2026-07", "2026-08"):
        frame = coverage[coverage["month"].eq(month)]
        rows.append({
            "month": month,
            "date_count": len(frame),
            "median_coverage": float(frame["coverage_ratio"].median()) if len(frame) else math.nan,
            "min_coverage": float(frame["coverage_ratio"].min()) if len(frame) else math.nan,
            "p10_coverage": float(frame["coverage_ratio"].quantile(.10)) if len(frame) else math.nan,
            "full_quality_date_count": int(frame["full_quality_date"].sum()),
            "full_quality_date_rate": float(frame["full_quality_date"].mean()) if len(frame) else math.nan,
            "median_size_coverage": float(frame["size_coverage_ratio"].median()) if len(frame) else math.nan,
            "min_size_coverage": float(frame["size_coverage_ratio"].min()) if len(frame) else math.nan,
            "size_full_quality_date_count": int(frame["m4_status"].eq("FULL_QUALITY_DATE").sum()),
            "median_amount_coverage": float(frame["amount_coverage_ratio"].median()) if len(frame) else math.nan,
            "median_turnover_coverage": float(frame["turnover_coverage_ratio"].median()) if len(frame) else math.nan,
        })
    return pd.DataFrame(rows)


def build_month_summary(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    months = ("2026-05", "2026-06", "2026-07", "2026-08")
    for variable in CONTEXT_VARIABLES:
        month_medians: dict[str, float] = {}
        for month in months:
            series = pd.to_numeric(
                values.loc[values["month"].eq(month), variable], errors="coerce"
            ).dropna()
            month_medians[month] = float(series.median()) if len(series) else math.nan
            rows.append({
                "variable": variable,
                "month": month,
                "valid_dates": len(series),
                "mean": float(series.mean()) if len(series) else math.nan,
                "median": float(series.median()) if len(series) else math.nan,
                "std": float(series.std(ddof=1)) if len(series) > 1 else math.nan,
                "p25": float(series.quantile(.25)) if len(series) else math.nan,
                "p75": float(series.quantile(.75)) if len(series) else math.nan,
                "min": float(series.min()) if len(series) else math.nan,
                "max": float(series.max()) if len(series) else math.nan,
                "may_vs_june_median_difference": math.nan,
                "june_vs_july_median_difference": math.nan,
                "may_vs_july_median_difference": math.nan,
                "july_vs_august_median_difference": math.nan,
            })
        differences = {
            "may_vs_june_median_difference": month_medians["2026-05"] - month_medians["2026-06"],
            "june_vs_july_median_difference": month_medians["2026-06"] - month_medians["2026-07"],
            "may_vs_july_median_difference": month_medians["2026-05"] - month_medians["2026-07"],
            "july_vs_august_median_difference": month_medians["2026-07"] - month_medians["2026-08"],
        }
        for row in rows[-4:]:
            row.update(differences)
    return pd.DataFrame(rows)


def build_redundancy(values: pd.DataFrame) -> pd.DataFrame:
    corr = values[list(CONTEXT_VARIABLES)].corr(method="spearman", min_periods=5)
    rows: list[dict[str, Any]] = []
    for left in CONTEXT_VARIABLES:
        for right in CONTEXT_VARIABLES:
            rho = corr.loc[left, right] if left in corr.index and right in corr.columns else math.nan
            rows.append({
                "variable_1": left,
                "variable_2": right,
                "spearman_rho": float(rho) if pd.notna(rho) else math.nan,
                "overlap_dates": int(values[[left, right]].dropna().shape[0]),
                "redundancy_status": (
                    "SELF" if left == right
                    else "HIGH_REDUNDANCY" if pd.notna(rho) and abs(float(rho)) >= .85
                    else "NOT_HIGH_REDUNDANCY" if pd.notna(rho)
                    else "NOT_COMPUTABLE"
                ),
            })
    return pd.DataFrame(rows)


def build_universe_lineage(
    root: str | Path,
    universe: pd.DataFrame,
    listing: pd.DataFrame,
    history: pd.DataFrame,
    universe_audit: Mapping[str, Any],
    daily_audit: Mapping[str, Any],
) -> pd.DataFrame:
    root_path = Path(root).resolve()
    suspension_files = sorted((root_path / "data/cache/suspension_status").glob("*.pkl"))
    snapshot_rows = [
        {
            "component": "CURRENT_UNIVERSE_SNAPSHOT",
            "status": "AVAILABLE_SINGLE_SNAPSHOT",
            "count": universe_audit["snapshot_file_count"],
            "coverage_or_value": universe_audit["current_universe_rows"],
            "source": universe_audit["snapshot_paths"],
            "notes": "Captured in August 2026; not a date-level May-August membership history.",
        },
        {
            "component": "DATE_LEVEL_HISTORICAL_UNIVERSE_SNAPSHOTS",
            "status": "NOT_AVAILABLE",
            "count": 0,
            "coverage_or_value": 0,
            "source": "data/cache/universe",
            "notes": "No one-snapshot-per-trade-date archive exists.",
        },
        {
            "component": "LISTING_DATE_METADATA",
            "status": "PARTIAL",
            "count": len(listing),
            "coverage_or_value": _safe_ratio(len(set(listing["code"]) & set(universe["code"])), len(universe)),
            "source": "data/cache/listing_metadata",
            "notes": "Only cached codes can be excluded before known listing date.",
        },
        {
            "component": "HISTORICAL_ST_NAME_STATUS",
            "status": "NOT_AVAILABLE",
            "count": 0,
            "coverage_or_value": 0,
            "source": "none",
            "notes": "Current name filter cannot recover historical ST/name changes.",
        },
        {
            "component": "HISTORICAL_SUSPENSION_STATUS",
            "status": "PARTIAL",
            "count": len(suspension_files),
            "coverage_or_value": len(suspension_files),
            "source": "data/cache/suspension_status",
            "notes": "Sparse date snapshots; daily-bar absence remains observable but membership is not exact.",
        },
        {
            "component": "DAILY_CACHE_CODE_COVERAGE",
            "status": "AVAILABLE_WITH_SCOPE_GAP",
            "count": daily_audit["daily_cache_main_board_code_count"],
            "coverage_or_value": _safe_ratio(
                daily_audit["daily_cache_main_board_code_count"], len(universe)
            ),
            "source": "data/cache/daily_unadjusted + daily fallback",
            "notes": (
                f"{daily_audit['daily_codes_not_in_current_universe']} cached main-board codes are "
                f"outside current universe; {daily_audit['current_universe_codes_without_any_daily_cache']} "
                "current-universe codes have no daily cache."
            ),
        },
        {
            "component": "HISTORICAL_FLOAT_MARKET_CAP",
            "status": "NOT_AVAILABLE",
            "count": 0,
            "coverage_or_value": 0,
            "source": "daily cache schema",
            "notes": "Only a static August snapshot has float market cap; M4 is not reconstructed.",
        },
        {
            "component": "D1_AVAILABILITY_PROOF_M1_M2_M3",
            "status": "PASS",
            "count": history["date"].nunique(),
            "coverage_or_value": 1,
            "source": "same-day unadjusted daily close/pct_chg cache",
            "notes": "Uses only D1 date cross-section after D1 close; no later rows are used.",
        },
        {
            "component": "D1_AVAILABILITY_PROOF_M4",
            "status": "FAIL_NO_HISTORICAL_CAP",
            "count": 0,
            "coverage_or_value": 0,
            "source": "none",
            "notes": "Static current cap cannot prove point-in-time size membership.",
        },
        {
            "component": "PCT_CHG_UNIT",
            "status": "PERCENTAGE_POINTS",
            "count": int(history["pct_chg"].notna().sum()),
            "coverage_or_value": float(history["pct_chg"].abs().quantile(.99)),
            "source": "normalized daily cache",
            "notes": "M3 uses fixed +3/-3 percentage-point cutoffs.",
        },
    ]
    return pd.DataFrame(snapshot_rows)


def _availability_status(ratios: pd.Series) -> str:
    ratios = pd.to_numeric(ratios, errors="coerce").dropna()
    if not len(ratios) or float(ratios.max()) == 0:
        return "NO"
    if float((ratios >= .95).mean()) >= .80:
        return "YES"
    return "PARTIAL"


def build_auxiliary_status(root: str | Path, coverage: pd.DataFrame) -> pd.DataFrame:
    root_path = Path(root).resolve()
    index_candidates = [
        path for path in (root_path / "data/cache").rglob("*")
        if path.is_file() and "index" in path.name.lower()
    ]
    limit_down_candidates = [
        path for path in (root_path / "data/cache").rglob("*")
        if path.is_file() and ("limit_down" in path.name.lower() or "limitdown" in path.name.lower())
    ]
    amount_status = _availability_status(coverage["amount_coverage_ratio"])
    turnover_status = _availability_status(coverage["turnover_coverage_ratio"])
    liquidity_status = (
        "YES" if amount_status == "YES" and turnover_status == "YES"
        else "NO" if amount_status == "NO" and turnover_status == "NO"
        else "PARTIAL"
    )
    rows = [
        {
            "item": "FULL_MARKET_TOTAL_AMOUNT",
            "status": amount_status,
            "source": "historical daily cache amount",
            "coverage_note": f"median row coverage={coverage['amount_coverage_ratio'].median():.6f}",
        },
        {
            "item": "MEDIAN_TURNOVER_RATE",
            "status": turnover_status,
            "source": "historical daily cache turnover_rate",
            "coverage_note": f"median row coverage={coverage['turnover_coverage_ratio'].median():.6f}",
        },
        {
            "item": "HISTORICAL_LIQUIDITY_AVAILABLE",
            "status": liquidity_status,
            "source": "coverage-only note; not a context variable",
            "coverage_note": "No liquidity factor constructed.",
        },
        {
            "item": "INDEX_SOURCE_AVAILABLE",
            "status": "YES" if index_candidates else "NO",
            "source": "|".join(path.relative_to(root_path).as_posix() for path in index_candidates),
            "coverage_note": "No index series opened or used.",
        },
        {
            "item": "HISTORICAL_LIMIT_DOWN_POOL",
            "status": "AVAILABLE" if limit_down_candidates else "NOT_AVAILABLE",
            "source": "|".join(path.relative_to(root_path).as_posix() for path in limit_down_candidates),
            "coverage_note": "No pct_chg approximation was used.",
        },
    ]
    return pd.DataFrame(rows)


def decide_state(
    coverage: pd.DataFrame,
    lineage: pd.DataFrame,
) -> tuple[str, str]:
    full_rate = float(coverage["full_quality_date"].mean()) if len(coverage) else 0.0
    if full_rate < .80:
        return "BROAD_CONTEXT_DATA_NOT_READY", "FIX_DATA_COVERAGE"
    historical_universe = lineage.loc[
        lineage["component"].eq("DATE_LEVEL_HISTORICAL_UNIVERSE_SNAPSHOTS"), "status"
    ].iloc[0]
    listing_status = lineage.loc[
        lineage["component"].eq("LISTING_DATE_METADATA"), "status"
    ].iloc[0]
    if historical_universe != "AVAILABLE" or listing_status != "COMPLETE":
        return "BROAD_CONTEXT_DATA_LINEAGE_RISK", "FIX_DATA_LINEAGE"
    if coverage["m4_status"].ne("FULL_QUALITY_DATE").any():
        return "BROAD_CONTEXT_DATA_PARTIAL", "BROAD_MARKET_CONTEXT_INFORMATION_AUDIT"
    return "BROAD_CONTEXT_DATA_READY", "BROAD_MARKET_CONTEXT_INFORMATION_AUDIT"


def render_review(
    source_fields: pd.DataFrame,
    coverage: pd.DataFrame,
    month_coverage: pd.DataFrame,
    lineage: pd.DataFrame,
    auxiliary: pd.DataFrame,
    state: str,
    next_action: str,
) -> str:
    preserved = source_fields.loc[
        source_fields["normalize_stock_universe_preserved"], "field"
    ].tolist()
    dropped = source_fields.loc[
        ~source_fields["normalize_stock_universe_preserved"], "field"
    ].tolist()
    valid_counts = {
        "M1": int(coverage["market_context_m1_valid"].sum()),
        "M2": int(coverage["market_context_m2_valid"].sum()),
        "M3": int(coverage["market_context_m3_valid"].sum()),
        "M4": int(coverage["market_context_m4_valid"].sum()),
    }
    lookup = auxiliary.set_index("item")["status"].to_dict()
    month_lines = []
    for row in month_coverage.itertuples(index=False):
        month_lines.append(
            f"- {row.month}: {row.full_quality_date_count}/{row.date_count} M1-M3 full-quality dates; "
            f"median coverage {row.median_coverage:.2%}."
        )
    return "\n".join([
        "# v004c Broad Market Context Coverage Audit v001",
        "",
        "This is a blind data-coverage audit. No security outcome, model score, or model ranking was opened.",
        "",
        "## Q1. 当前 stock source 原始层到底有哪些字段？",
        "",
        "Eastmoney spot currently requests: " + ", ".join(source_fields["field"].tolist()) + ".",
        "",
        "## Q2. 哪些字段现在被 normalize 丢掉了？",
        "",
        "Preserved raw fields: " + ", ".join(preserved) + ".",
        "Dropped raw fields: " + ", ".join(dropped) + ".",
        "",
        "## Q3. May-August 能不能可靠重建全主板每日横截面？",
        "",
        "M1-M3 can be reconstructed on dates meeting the 95% return-coverage gate, but the result uses an August current-universe snapshot as a historical approximation. Date-level membership, historical ST/name state, delistings, and complete listing-date history are unavailable. This is a lineage risk, not proof of exact historical membership.",
        "",
        *month_lines,
        "",
        "## Q4. 4个固定 context 指标各自有多少有效日期？",
        "",
        f"M1={valid_counts['M1']}, M2={valid_counts['M2']}, M3={valid_counts['M3']}, M4={valid_counts['M4']}. M4 is not reconstructed because historical float market cap is absent.",
        "",
        "## Q5. 历史 universe 是否存在明显偏差？",
        "",
        "YES. Only one current universe snapshot exists; listing metadata and suspension snapshots are sparse, and historical ST/name membership cannot be recovered exactly.",
        "",
        "## Q6. 历史 liquidity 能不能重建？",
        "",
        f"HISTORICAL_LIQUIDITY_AVAILABLE = {lookup.get('HISTORICAL_LIQUIDITY_AVAILABLE', 'UNKNOWN')}.",
        "",
        "## Q7. index source 当前有没有？",
        "",
        f"INDEX_SOURCE_AVAILABLE = {lookup.get('INDEX_SOURCE_AVAILABLE', 'UNKNOWN')}.",
        "",
        "## Q8. 正式 historical limit-down pool 有没有？",
        "",
        f"HISTORICAL_LIMIT_DOWN_POOL = {lookup.get('HISTORICAL_LIMIT_DOWN_POOL', 'NOT_AVAILABLE')}.",
        "",
        "## Q9. 下一步是否授权做4变量 information audit？",
        "",
        "NO under the exact full-main-board lineage contract. Resolve historical universe membership first; do not interpret the reconstructed values against outcomes yet.",
        "",
        f"BROAD_MARKET_CONTEXT_DATA_STATE = {state}",
        "",
        "LABEL_OR_OUTCOME_ACCESSED = NO",
        "",
        "MODEL_TRAINED = NO",
        "",
        "NEW_FEATURE_SEARCH = NO",
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
    source_fields = build_source_field_status(history)
    coverage, values = build_daily_context(universe, listing, history)
    coverage["market_context_m1_valid"] = coverage["full_quality_date"]
    coverage["market_context_m2_valid"] = coverage["full_quality_date"]
    coverage["market_context_m3_valid"] = coverage["full_quality_date"]
    coverage["market_context_m4_valid"] = coverage["m4_status"].eq("FULL_QUALITY_DATE")
    month_coverage = build_month_coverage(coverage)
    month_summary = build_month_summary(values)
    redundancy = build_redundancy(values)
    lineage = build_universe_lineage(
        root_path, universe, listing, history, universe_audit, daily_audit
    )
    auxiliary = build_auxiliary_status(root_path, coverage)
    state, next_action = decide_state(coverage, lineage)
    review = render_review(
        source_fields, coverage, month_coverage, lineage, auxiliary, state, next_action
    )
    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(source_fields),
        OUTPUT_FILENAMES[1]: _csv_bytes(coverage),
        OUTPUT_FILENAMES[2]: _csv_bytes(month_coverage),
        OUTPUT_FILENAMES[3]: _csv_bytes(values),
        OUTPUT_FILENAMES[4]: _csv_bytes(month_summary),
        OUTPUT_FILENAMES[5]: _csv_bytes(redundancy),
        OUTPUT_FILENAMES[6]: _csv_bytes(lineage),
        OUTPUT_FILENAMES[7]: _csv_bytes(auxiliary),
        OUTPUT_FILENAMES[8]: review.encode("utf-8"),
    }
    audit = {
        **universe_audit,
        **daily_audit,
        "output_file_count": len(outputs),
        "trade_dates": len(coverage),
        "first_trade_date": coverage["trade_date"].min(),
        "last_trade_date": coverage["trade_date"].max(),
        "m1_m2_m3_full_quality_dates": int(coverage["full_quality_date"].sum()),
        "m4_full_quality_dates": int(coverage["market_context_m4_valid"].sum()),
        "broad_market_context_data_state": state,
        "next_action": next_action,
        "label_or_outcome_accessed": "NO",
        "model_trained": "NO",
        "new_feature_search": "NO",
        "external_data_fetches": 0,
    }
    return outputs, audit


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, audit = build_outputs(root_path)
    output_dir = root_path / "reports/research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return output_dir, audit

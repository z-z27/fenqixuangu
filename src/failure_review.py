from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .code_utils import normalize_stock_code
from .config import get_data_config
from .loaders import DataQualityError, MarketDataService


def review_failed_data(
    quality_file: str | Path | None = None,
    days: int | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, Path, Path, Path]:
    config = get_data_config()
    quality_path = Path(quality_file) if quality_file else _latest_quality_file(config.reports_dir / "data_quality")
    quality = pd.read_csv(quality_path, dtype={"code": str})
    failed = quality[quality["status"].astype(str) == "failed"].copy()
    trade_date = _infer_trade_date(quality, quality_path)
    rows: list[dict[str, Any]] = []
    service = MarketDataService(config)

    for _, item in failed.iterrows():
        code = normalize_stock_code(str(item["code"]))
        original_reason = str(item.get("warnings") or item.get("error") or "")
        review = _inspect_failed_code(
            service,
            code,
            original_reason,
            int(days or config.default_5min_days),
            trade_date,
        )
        review.update(
            {
                "code": code,
                "name": item.get("name", ""),
                "trade_date": trade_date,
                "d0_date": item.get("d0_date", ""),
                "original_reason": original_reason,
                "repair_attempted": bool(force_refresh),
                "repair_status": "not_attempted",
                "repair_error": "",
            }
        )
        if force_refresh:
            try:
                bars = service.get_stock_bars(code, days=days, end_date=trade_date, force_refresh=True)
                review["repair_status"] = "repaired"
                review["post_repair_daily_rows"] = bars.quality.get("daily_history_rows")
                review["post_repair_minute_trade_days"] = bars.quality.get("minute_trade_days")
                review["post_repair_warning"] = bars.quality.get("warnings", "")
            except DataQualityError as exc:
                review["repair_status"] = "still_failed"
                review["repair_error"] = str(exc)
                review["post_repair_daily_rows"] = exc.quality.get("daily_history_rows")
                review["post_repair_minute_trade_days"] = exc.quality.get("minute_trade_days")
                review["post_repair_warning"] = exc.quality.get("warnings", "")
            except Exception as exc:
                review["repair_status"] = "fetch_failed"
                review["repair_error"] = str(exc)
        rows.append(review)

    frame = pd.DataFrame(rows)
    csv_path, md_path = write_failed_data_review_reports(
        frame,
        config.reports_dir / "data_quality",
        trade_date,
    )
    exclusion_path = write_exclusion_list(frame, config.processed_dir, trade_date)
    return frame, csv_path, md_path, exclusion_path


def write_failed_data_review_reports(
    frame: pd.DataFrame,
    output_dir: Path,
    trade_date: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"failed_data_review_{trade_date}.csv"
    md_path = output_dir / f"failed_data_review_{trade_date}.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_failed_data_review_markdown(frame, trade_date), encoding="utf-8")
    return csv_path, md_path


def write_exclusion_list(frame: pd.DataFrame, output_dir: Path, trade_date: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"data_quality_exclusions_{trade_date}.csv"
    if frame.empty:
        pd.DataFrame(columns=["code", "name", "trade_date", "reason", "action"]).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        return path
    exclusions = frame.copy()
    if "repair_status" in exclusions.columns:
        exclusions = exclusions[exclusions["repair_status"].astype(str) != "repaired"].copy()
    output = pd.DataFrame(
        {
            "code": exclusions.get("code", pd.Series(dtype=object)),
            "name": exclusions.get("name", pd.Series(dtype=object)),
            "trade_date": exclusions.get("trade_date", pd.Series(dtype=object)),
            "reason": exclusions.get("original_reason", pd.Series(dtype=object)),
            "action": "exclude_until_data_quality_passes",
        }
    )
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_failed_data_review_markdown(frame: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} Failed Data Review", ""]
    if frame.empty:
        lines.append("No failed data rows.")
        return "\n".join(lines)
    categories = frame["failure_category"].value_counts(dropna=False).to_dict()
    repair = frame["repair_status"].value_counts(dropna=False).to_dict() if "repair_status" in frame.columns else {}
    lines.extend(["## Summary", ""])
    lines.append(f"- failed codes reviewed: **{len(frame)}**")
    for key, value in categories.items():
        lines.append(f"- category `{key}`: **{int(value)}**")
    for key, value in repair.items():
        lines.append(f"- repair `{key}`: **{int(value)}**")
    lines.append("")

    lines.extend(
        [
            "## Details",
            "",
            "| code | name | category | action | max close diff | mismatch date | repair |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for _, item in frame.sort_values("code").iterrows():
        lines.append(
            "| {code} | {name} | {category} | {action} | {diff} | {date} | {repair} |".format(
                code=item.get("code", ""),
                name=item.get("name", ""),
                category=item.get("failure_category", ""),
                action=item.get("recommended_action", ""),
                diff=_fmt(item.get("max_close_diff")),
                date=item.get("max_close_diff_date", ""),
                repair=item.get("repair_status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Failed rows remain excluded from signal generation unless a forced refetch returns a full `ok` quality result.",
            "- Insufficient daily history is not manually padded; wait until enough listed trading days exist or use a validated alternate source.",
            "- Daily/5m close mismatches are not loosened here; the source pair must pass the same threshold before the code is tradable.",
        ]
    )
    return "\n".join(lines)


def _inspect_failed_code(
    service: MarketDataService,
    code: str,
    original_reason: str,
    days: int,
    trade_date: str,
) -> dict[str, Any]:
    daily = service.daily_cache.read(code)
    minute = service.minute_cache.read(code)
    daily_for_review = _filter_to_end_date(daily, "date", trade_date)
    minute_for_review = _filter_to_end_date(minute, "trade_date", trade_date)
    daily_rows = int(daily["date"].dropna().nunique()) if daily is not None and not daily.empty and "date" in daily.columns else 0
    minute_trade_days = (
        int(minute["trade_date"].dropna().nunique())
        if minute is not None and not minute.empty and "trade_date" in minute.columns
        else 0
    )
    mismatch = _close_mismatch_details(daily_for_review, minute_for_review, days)
    category = _classify_failure(original_reason, daily_rows, mismatch)
    action = _recommended_action(category)
    return {
        "failure_category": category,
        "recommended_action": action,
        "cached_daily_rows": daily_rows,
        "cached_minute_trade_days": minute_trade_days,
        "required_minute_trade_days": days,
        "daily_start": _date_min(daily, "date"),
        "daily_end": _date_max(daily, "date"),
        "minute_start": _date_min(minute, "datetime"),
        "minute_end": _date_max(minute, "datetime"),
        "max_close_diff": mismatch.get("max_close_diff"),
        "max_close_diff_date": mismatch.get("max_close_diff_date", ""),
        "daily_close_at_max_diff": mismatch.get("daily_close"),
        "minute_close_at_max_diff": mismatch.get("minute_close"),
    }


def _classify_failure(reason: str, daily_rows: int, mismatch: dict[str, Any]) -> str:
    text = reason.lower()
    if "daily history rows" in text or daily_rows < 120:
        return "insufficient_daily_history"
    if "close cross-check" in text or "close mismatch" in text or mismatch.get("max_close_diff"):
        return "daily_5m_close_mismatch"
    if "minute trade days" in text:
        return "insufficient_5m_history"
    if "network" in text or "connection" in text or "timed out" in text:
        return "fetch_error"
    return "unknown"


def _recommended_action(category: str) -> str:
    if category == "insufficient_daily_history":
        return "exclude_until_180_daily_rows_or_validated_alternate_source"
    if category == "insufficient_5m_history":
        return "force_refetch_5m_then_exclude_if_still_short"
    if category == "daily_5m_close_mismatch":
        return "force_refetch_and_compare_sources_before_release"
    if category == "fetch_error":
        return "retry_with_network_then_keep_excluded_on_failure"
    return "manual_source_audit_required"


def _close_mismatch_details(daily: pd.DataFrame | None, minute: pd.DataFrame | None, days: int) -> dict[str, Any]:
    if daily is None or daily.empty or minute is None or minute.empty:
        return {}
    if "date" not in daily.columns or "trade_date" not in minute.columns:
        return {}
    daily_recent = _keep_recent_trade_days(daily, "date", days)
    minute_recent = _keep_recent_trade_days(minute, "trade_date", days)
    minute_close = (
        minute_recent.sort_values("datetime")
        .groupby("trade_date", as_index=False)
        .agg(minute_close=("close", "last"))
    )
    comparison = daily_recent[["date", "close"]].rename(columns={"date": "trade_date", "close": "daily_close"}).merge(
        minute_close,
        on="trade_date",
        how="inner",
    )
    if comparison.empty:
        return {}
    comparison["diff"] = (
        pd.to_numeric(comparison["daily_close"], errors="coerce")
        - pd.to_numeric(comparison["minute_close"], errors="coerce")
    ).abs()
    if comparison["diff"].dropna().empty:
        return {}
    row = comparison.loc[comparison["diff"].idxmax()]
    return {
        "max_close_diff": float(row["diff"]),
        "max_close_diff_date": row["trade_date"],
        "daily_close": float(row["daily_close"]),
        "minute_close": float(row["minute_close"]),
    }


def _latest_quality_file(root: Path) -> Path:
    files = sorted(root.glob("data_quality_*.csv"))
    if not files:
        raise FileNotFoundError(f"no data quality CSV found under {root}")
    return files[-1]


def _infer_trade_date(frame: pd.DataFrame, path: Path) -> str:
    if "trade_date" in frame.columns and frame["trade_date"].notna().any():
        return str(frame["trade_date"].dropna().max())
    stem = path.stem.replace("data_quality_", "")
    return stem or pd.Timestamp.now().strftime("%Y-%m-%d")


def _keep_recent_trade_days(frame: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    trade_dates = sorted(result[date_col].dropna().unique().tolist())
    selected = set(trade_dates[-int(days) :])
    return result[result[date_col].isin(selected)].sort_values(date_col).reset_index(drop=True)


def _filter_to_end_date(frame: pd.DataFrame | None, date_col: str, end_date: str) -> pd.DataFrame | None:
    if frame is None or frame.empty or date_col not in frame.columns:
        return frame
    result = frame.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    return result[dates <= pd.Timestamp(end_date)].reset_index(drop=True)


def _date_min(frame: pd.DataFrame | None, column: str) -> str:
    if frame is None or frame.empty or column not in frame.columns:
        return ""
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return "" if values.empty else values.min().strftime("%Y-%m-%d %H:%M:%S")


def _date_max(frame: pd.DataFrame | None, column: str) -> str:
    if frame is None or frame.empty or column not in frame.columns:
        return ""
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return "" if values.empty else values.max().strftime("%Y-%m-%d %H:%M:%S")


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .code_utils import normalize_stock_code
from .config import DataConfig, get_data_config
from .indicators import enrich_daily_indicators
from .loaders import MarketDataService, load_limitup_file


MA_PERIODS = (5, 10, 20, 30)
PRICE_COLUMNS = ("open", "high", "low", "close")
MINUTE_REQUIRED_COLUMNS = ("datetime", "open", "high", "low", "close", "volume", "amount")
DAILY_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def run_data_acceptance(
    limitup_file: str | Path = "data/processed/recent_limitups.csv",
    days: int | None = None,
    max_codes: int | None = None,
    reference_root: str | Path = r"F:\dataaccept",
) -> tuple[pd.DataFrame, Path, Path]:
    config = get_data_config()
    pool = load_limitup_file(limitup_file)
    codes = pool["code"].dropna().astype(str).drop_duplicates().tolist()
    if max_codes:
        codes = codes[: int(max_codes)]
    trade_date = _latest_trade_date(pool)
    rows = [
        validate_code_data(
            code=code,
            days=days,
            trade_date=trade_date,
            reference_root=reference_root,
            config=config,
        )
        for code in codes
    ]
    frame = pd.DataFrame(rows)
    csv_path, md_path = write_data_acceptance_reports(
        frame,
        config.reports_dir / "data_acceptance",
        trade_date,
    )
    return frame, csv_path, md_path


def validate_code_data(
    code: str,
    days: int | None = None,
    trade_date: str | None = None,
    reference_root: str | Path = r"F:\dataaccept",
    config: DataConfig | None = None,
) -> dict[str, Any]:
    cfg = config or get_data_config()
    service = MarketDataService(cfg)
    normalized = normalize_stock_code(code)
    day_count = int(days or cfg.default_5min_days)
    warmup_days = max(int(cfg.indicator_warmup_trading_days), 120)
    daily_required_days = max(int(cfg.daily_history_days), day_count + warmup_days)
    end_date_text = pd.Timestamp(trade_date or pd.Timestamp.now().strftime("%Y-%m-%d")).strftime("%Y-%m-%d")

    daily = service.daily_cache.read(normalized)
    minute = service.minute_cache.read(normalized)
    warnings: list[str] = []
    failures: list[str] = []

    row: dict[str, Any] = {
        "code": normalized,
        "status": "accepted",
        "daily_cache_exists": daily is not None and not daily.empty,
        "minute_cache_exists": minute is not None and not minute.empty,
        "validation_end_date": end_date_text,
        "daily_rows": int(len(daily)) if daily is not None else 0,
        "daily_required_days": daily_required_days,
        "minute_rows": int(len(minute)) if minute is not None else 0,
        "minute_required_days": day_count,
        "daily_start": _date_min(daily, "date"),
        "daily_end": _date_max(daily, "date"),
        "minute_start": _date_min(minute, "datetime"),
        "minute_end": _date_max(minute, "datetime"),
    }

    if daily is None or daily.empty:
        failures.append("daily cache missing")
        daily = pd.DataFrame()
    if minute is None or minute.empty:
        failures.append("5m cache missing")
        minute = pd.DataFrame()

    row["daily_future_rows_ignored"] = _future_row_count(daily, "date", end_date_text)
    row["minute_future_rows_ignored"] = _future_row_count(minute, "trade_date", end_date_text)
    daily_until_end = _filter_to_end_date(daily, "date", end_date_text)
    minute_until_end = _filter_to_end_date(minute, "trade_date", end_date_text)
    if row["daily_future_rows_ignored"]:
        warnings.append(f"ignored {row['daily_future_rows_ignored']} daily rows after {end_date_text}")
    if row["minute_future_rows_ignored"]:
        warnings.append(f"ignored {row['minute_future_rows_ignored']} 5m rows after {end_date_text}")

    daily_report = _validate_daily_frame(daily_until_end, daily_required_days)
    failures.extend(daily_report.pop("_failures", []))
    warnings.extend(daily_report.pop("_warnings", []))
    row.update(daily_report)

    minute_report = _validate_minute_frame(minute_until_end, day_count)
    failures.extend(minute_report.pop("_failures", []))
    warnings.extend(minute_report.pop("_warnings", []))
    row.update(minute_report)

    ma_report = _validate_ma_calculation(daily_until_end, day_count)
    failures.extend(ma_report.pop("_failures", []))
    warnings.extend(ma_report.pop("_warnings", []))
    row.update(ma_report)

    close_report = _validate_daily_minute_close(daily_until_end, minute_until_end, day_count)
    failures.extend(close_report.pop("_failures", []))
    warnings.extend(close_report.pop("_warnings", []))
    row.update(close_report)

    reference_report = _compare_with_dataaccept_reference(
        normalized,
        daily_until_end,
        minute_until_end,
        Path(reference_root),
    )
    failures.extend(reference_report.pop("_failures", []))
    warnings.extend(reference_report.pop("_warnings", []))
    row.update(reference_report)

    row["status"] = "failed" if failures else "accepted"
    row["warnings"] = "; ".join(warnings)
    row["failures"] = "; ".join(failures)
    return row


def write_data_acceptance_reports(
    frame: pd.DataFrame,
    output_dir: Path,
    trade_date: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"data_acceptance_{trade_date}.csv"
    md_path = output_dir / f"data_acceptance_{trade_date}.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_data_acceptance_markdown(frame, trade_date), encoding="utf-8")
    return csv_path, md_path


def build_data_acceptance_markdown(frame: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} Data Acceptance Report", ""]
    if frame.empty:
        lines.append("No data acceptance rows.")
        return "\n".join(lines)

    accepted = int((frame["status"] == "accepted").sum()) if "status" in frame.columns else 0
    failed = int((frame["status"] == "failed").sum()) if "status" in frame.columns else 0
    ma_ok = _bool_count(frame, "ma_recalc_check_ok")
    close_ok = _bool_count(frame, "daily_minute_close_check_ok")
    ref_daily = _bool_count(frame, "reference_daily_compared")
    ref_minute = _bool_count(frame, "reference_5m_compared")

    lines.extend(
        [
            "## Summary",
            "",
            f"- total codes: **{len(frame)}**",
            f"- accepted: **{accepted}**",
            f"- failed: **{failed}**",
            f"- MA recalc check passed: **{ma_ok}**",
            f"- daily/5m close check passed: **{close_ok}**",
            f"- dataaccept daily comparisons: **{ref_daily}**",
            f"- dataaccept 5m comparisons: **{ref_minute}**",
            "",
        ]
    )

    failed_rows = frame[frame["status"] == "failed"] if "status" in frame.columns else pd.DataFrame()
    if not failed_rows.empty:
        lines.extend(["## Failed Codes", "", "| code | failures | warnings |", "|---|---|---|"])
        for _, item in failed_rows.sort_values("code").head(100).iterrows():
            lines.append(
                f"| {item.get('code', '')} | {_clean_cell(item.get('failures', ''))} | {_clean_cell(item.get('warnings', ''))} |"
            )
        if len(failed_rows) > 100:
            lines.append(f"| ... | another {len(failed_rows) - 100} rows in CSV | |")
        lines.append("")

    compared = frame[
        frame.get("reference_daily_compared", False).fillna(False).astype(bool)
        | frame.get("reference_5m_compared", False).fillna(False).astype(bool)
    ]
    if not compared.empty:
        lines.extend(
            [
                "## Dataaccept Cross Checks",
                "",
                "| code | daily dates | daily max close diff | 5m bars | 5m max close diff | volume ratio |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, item in compared.sort_values("code").iterrows():
            lines.append(
                "| {code} | {daily_dates} | {daily_diff} | {minute_bars} | {minute_diff} | {volume_ratio} |".format(
                    code=item.get("code", ""),
                    daily_dates=_fmt_number(item.get("reference_daily_matched_dates")),
                    daily_diff=_fmt_number(item.get("reference_daily_max_close_diff")),
                    minute_bars=_fmt_number(item.get("reference_5m_matched_bars")),
                    minute_diff=_fmt_number(item.get("reference_5m_max_close_diff")),
                    volume_ratio=_fmt_number(item.get("reference_daily_volume_ratio_median")),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Rules",
            "",
            "- daily MA5/MA10/MA20/MA30 are recalculated from full daily history and compared with project indicators.",
            "- at least 120 trading days of warmup are required; current default requires 180 daily rows.",
            "- 5m data must cover the requested trading-day window.",
            "- daily close must match the last 5m close for overlapping dates within 0.02.",
            r"- overlapping `F:\dataaccept` caches are compared when readable; missing references are warnings, not failures.",
        ]
    )
    return "\n".join(lines)


def _validate_daily_frame(daily: pd.DataFrame, required_days: int) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"_failures": failures, "_warnings": warnings}
    missing = [column for column in DAILY_REQUIRED_COLUMNS if column not in daily.columns]
    report["daily_missing_required_columns"] = ",".join(missing)
    if missing:
        failures.append("daily missing required columns: " + ",".join(missing))
        return report

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    date_count = int(data["date"].dropna().dt.strftime("%Y-%m-%d").nunique())
    report["daily_trade_days"] = date_count
    report["daily_duplicate_date_count"] = int(data["date"].duplicated().sum())
    report["daily_missing_ohlc_count"] = int(data[list(PRICE_COLUMNS)].isna().any(axis=1).sum())
    report["daily_invalid_high_low_count"] = _invalid_high_low_count(data)
    report["daily_non_positive_price_count"] = _non_positive_price_count(data)
    report["daily_sorted"] = bool(data["date"].dropna().is_monotonic_increasing)

    if date_count < required_days:
        failures.append(f"daily trade days {date_count} < required {required_days}")
    if report["daily_duplicate_date_count"] > 0:
        failures.append("daily duplicate date")
    if report["daily_missing_ohlc_count"] > 0:
        failures.append("daily missing OHLC")
    if report["daily_invalid_high_low_count"] > 0:
        failures.append("daily invalid OHLC high/low")
    if report["daily_non_positive_price_count"] > 0:
        failures.append("daily non-positive price")
    if not report["daily_sorted"]:
        failures.append("daily date not sorted")
    return report


def _validate_minute_frame(minute: pd.DataFrame, required_days: int) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"_failures": failures, "_warnings": warnings}
    missing = [column for column in MINUTE_REQUIRED_COLUMNS if column not in minute.columns]
    report["minute_missing_required_columns"] = ",".join(missing)
    if missing:
        failures.append("5m missing required columns: " + ",".join(missing))
        return report

    data = minute.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    if "trade_date" not in data.columns:
        data["trade_date"] = data["datetime"].dt.strftime("%Y-%m-%d")
    trade_days = int(data["trade_date"].dropna().nunique())
    bars_per_day = data.groupby("trade_date")["datetime"].count()
    report["minute_trade_days"] = trade_days
    report["minute_duplicate_datetime_count"] = int(data["datetime"].duplicated().sum())
    report["minute_missing_ohlc_count"] = int(data[list(PRICE_COLUMNS)].isna().any(axis=1).sum())
    report["minute_missing_volume_count"] = int(data["volume"].isna().sum())
    report["minute_missing_amount_count"] = int(data["amount"].isna().sum())
    report["minute_zero_volume_count"] = int((pd.to_numeric(data["volume"], errors="coerce").fillna(0) == 0).sum())
    report["minute_invalid_high_low_count"] = _invalid_high_low_count(data)
    report["minute_non_positive_price_count"] = _non_positive_price_count(data)
    report["minute_sorted"] = bool(data["datetime"].dropna().is_monotonic_increasing)
    report["minute_min_bars_per_day"] = int(bars_per_day.min()) if not bars_per_day.empty else 0
    report["minute_max_bars_per_day"] = int(bars_per_day.max()) if not bars_per_day.empty else 0

    if trade_days < required_days:
        failures.append(f"5m trade days {trade_days} < required {required_days}")
    if report["minute_duplicate_datetime_count"] > 0:
        failures.append("5m duplicate datetime")
    if report["minute_missing_ohlc_count"] > 0:
        failures.append("5m missing OHLC")
    if report["minute_invalid_high_low_count"] > 0:
        failures.append("5m invalid OHLC high/low")
    if report["minute_non_positive_price_count"] > 0:
        failures.append("5m non-positive price")
    if not report["minute_sorted"]:
        failures.append("5m datetime not sorted")
    if report["minute_missing_volume_count"] > 0:
        warnings.append("5m missing volume")
    if report["minute_missing_amount_count"] > 0:
        warnings.append("5m missing amount")
    return report


def _validate_ma_calculation(daily: pd.DataFrame, days: int) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"_failures": failures, "_warnings": warnings}
    if daily is None or daily.empty or "date" not in daily.columns or "close" not in daily.columns:
        failures.append("MA cannot be checked without daily date/close")
        report["ma_recalc_check_ok"] = False
        return report

    base = daily.copy().sort_values("date").reset_index(drop=True)
    base["close"] = pd.to_numeric(base["close"], errors="coerce")
    independent = base[["date", "close"]].copy()
    for period in MA_PERIODS:
        independent[f"ma{period}_expected"] = independent["close"].rolling(
            period,
            min_periods=period,
        ).mean()

    recent = _keep_recent_trade_days(base, "date", days)
    enriched = enrich_daily_indicators(recent, full_daily=base)
    comparison = enriched[["date", *[f"ma{period}" for period in MA_PERIODS]]].merge(
        independent[["date", *[f"ma{period}_expected" for period in MA_PERIODS]]],
        on="date",
        how="left",
    )
    max_diff = 0.0
    missing_latest: list[str] = []
    if comparison.empty:
        failures.append("MA comparison has no rows")
    else:
        latest = comparison.iloc[-1]
        for period in MA_PERIODS:
            actual = latest.get(f"ma{period}")
            expected = latest.get(f"ma{period}_expected")
            if pd.isna(actual) or pd.isna(expected):
                missing_latest.append(f"ma{period}")
                continue
            max_diff = max(max_diff, abs(float(actual) - float(expected)))
    report["ma_recalc_max_abs_diff"] = max_diff
    report["ma_recalc_check_ok"] = bool(not missing_latest and max_diff <= 1e-9)
    report["missing_latest_ma"] = ",".join(missing_latest)
    if missing_latest:
        failures.append("missing latest MA: " + ",".join(missing_latest))
    if max_diff > 1e-9:
        failures.append(f"MA recalc mismatch max_diff={max_diff}")
    return report


def _validate_daily_minute_close(daily: pd.DataFrame, minute: pd.DataFrame, days: int) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"_failures": failures, "_warnings": warnings}
    if daily is None or daily.empty or minute is None or minute.empty:
        failures.append("daily/5m close check missing data")
        report["daily_minute_close_check_ok"] = False
        return report
    if "date" not in daily.columns or "trade_date" not in minute.columns:
        failures.append("daily/5m close check missing date columns")
        report["daily_minute_close_check_ok"] = False
        return report
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
        failures.append("daily/5m close check has no matched dates")
        report["daily_minute_close_check_ok"] = False
        report["daily_minute_close_matched_days"] = 0
        return report
    diff = (
        pd.to_numeric(comparison["daily_close"], errors="coerce")
        - pd.to_numeric(comparison["minute_close"], errors="coerce")
    ).abs()
    max_diff = float(diff.max())
    report["daily_minute_close_matched_days"] = int(len(comparison))
    report["daily_minute_close_max_abs_diff"] = max_diff
    report["daily_minute_close_check_ok"] = bool(max_diff <= 0.02)
    if max_diff > 0.02:
        failures.append(f"daily/5m close mismatch max_diff={max_diff}")
    return report


def _compare_with_dataaccept_reference(
    code: str,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    reference_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "_failures": failures,
        "_warnings": warnings,
        "reference_daily_compared": False,
        "reference_5m_compared": False,
    }
    ref_daily, daily_warning = _read_dataaccept_daily(code, reference_root)
    if daily_warning:
        warnings.append(daily_warning)
    if ref_daily is not None and not ref_daily.empty:
        daily_compare = _compare_daily_reference(daily, ref_daily)
        failures.extend(daily_compare.pop("_failures", []))
        warnings.extend(daily_compare.pop("_warnings", []))
        report.update(daily_compare)

    ref_minute, minute_warning = _read_dataaccept_5m(code, reference_root)
    if minute_warning:
        warnings.append(minute_warning)
    if ref_minute is not None and not ref_minute.empty:
        minute_compare = _compare_minute_reference(minute, ref_minute)
        failures.extend(minute_compare.pop("_failures", []))
        warnings.extend(minute_compare.pop("_warnings", []))
        report.update(minute_compare)
    return report


def _compare_daily_reference(daily: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"_failures": failures, "_warnings": warnings}
    if daily is None or daily.empty:
        warnings.append("dataaccept daily reference exists but project daily is missing")
        return report
    ref = _normalize_reference_daily(reference)
    project = daily.copy()
    project["date"] = pd.to_datetime(project["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    comparison = project.merge(ref, on="date", how="inner", suffixes=("_project", "_ref"))
    report["reference_daily_compared"] = True
    report["reference_daily_matched_dates"] = int(len(comparison))
    if comparison.empty:
        warnings.append("dataaccept daily reference has no overlapping dates")
        return report
    max_close_diff = _max_abs_diff(comparison["close_project"], comparison["close_ref"])
    report["reference_daily_max_close_diff"] = max_close_diff
    for column in ("open", "high", "low", "close"):
        report[f"reference_daily_max_{column}_diff"] = _max_abs_diff(
            comparison[f"{column}_project"],
            comparison[f"{column}_ref"],
        )
    if "volume_project" in comparison.columns and "volume_ref" in comparison.columns:
        ratio = _median_ratio(comparison["volume_ref"], comparison["volume_project"])
        report["reference_daily_volume_ratio_median"] = ratio
        if ratio is not None and not (0.9 <= ratio <= 1.1):
            warnings.append(f"dataaccept daily volume unit ratio median={ratio}")
    if max_close_diff is not None and max_close_diff > 0.02:
        failures.append(f"dataaccept daily close mismatch max_diff={max_close_diff}")
    return report


def _compare_minute_reference(minute: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"_failures": failures, "_warnings": warnings}
    if minute is None or minute.empty:
        warnings.append("dataaccept 5m reference exists but project 5m is missing")
        return report
    ref = _normalize_reference_minute(reference)
    project = minute.copy()
    project["datetime"] = pd.to_datetime(project["datetime"], errors="coerce")
    comparison = project.merge(ref, on="datetime", how="inner", suffixes=("_project", "_ref"))
    report["reference_5m_compared"] = True
    report["reference_5m_matched_bars"] = int(len(comparison))
    if comparison.empty:
        warnings.append("dataaccept 5m reference has no overlapping bars")
        return report
    max_close_diff = _max_abs_diff(comparison["close_project"], comparison["close_ref"])
    report["reference_5m_max_close_diff"] = max_close_diff
    for column in ("open", "high", "low", "close"):
        report[f"reference_5m_max_{column}_diff"] = _max_abs_diff(
            comparison[f"{column}_project"],
            comparison[f"{column}_ref"],
        )
    if max_close_diff is not None and max_close_diff > 0.02:
        failures.append(f"dataaccept 5m close mismatch max_diff={max_close_diff}")
    return report


def _read_dataaccept_daily(code: str, root: Path) -> tuple[pd.DataFrame | None, str]:
    path = root / "data" / "cache" / "stock_daily" / f"{code}_daily_none.pkl"
    if not path.exists():
        return None, f"dataaccept daily reference missing for {code}"
    try:
        return pd.read_pickle(path), ""
    except Exception as exc:
        return None, f"dataaccept daily reference unreadable for {code}: {exc}"


def _read_dataaccept_5m(code: str, root: Path) -> tuple[pd.DataFrame | None, str]:
    base = root / "data" / "cache" / "stock_5min"
    pkl_path = base / f"{code}_5min_none.pkl"
    parquet_path = base / f"{code}_5min_none.parquet"
    warning = ""
    if pkl_path.exists():
        try:
            return pd.read_pickle(pkl_path), ""
        except Exception as exc:
            warning = f"dataaccept 5m pkl reference unreadable for {code}: {exc}"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), ""
        except Exception as exc:
            warning = f"dataaccept 5m parquet reference unreadable for {code}: {exc}"
    exported = _read_dataaccept_5m_export(code, root)
    if exported is not None and not exported.empty:
        return exported, warning
    if warning:
        return None, warning
    return None, f"dataaccept 5m reference missing for {code}"


def _read_dataaccept_5m_export(code: str, root: Path) -> pd.DataFrame | None:
    export_root = root / "data" / "exports"
    paths = sorted(export_root.glob(f"{code}_5min_*_analysis_package/data.csv"))
    if not paths:
        return None
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return None
    return max(frames, key=len)


def _normalize_reference_daily(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename = {
        "trade_date": "date",
        "daily_open": "open",
        "daily_high": "high",
        "daily_low": "low",
        "daily_close": "close",
        "daily_volume": "volume",
    }
    result = result.rename(columns=rename)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")


def _normalize_reference_minute(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["datetime", "close"]).drop_duplicates("datetime", keep="last")


def _keep_recent_trade_days(frame: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    trade_dates = sorted(result[date_col].dropna().unique().tolist())
    selected = set(trade_dates[-int(days) :])
    return result[result[date_col].isin(selected)].sort_values(date_col).reset_index(drop=True)


def _filter_to_end_date(frame: pd.DataFrame, date_col: str, end_date: str) -> pd.DataFrame:
    if frame is None or frame.empty or date_col not in frame.columns:
        return frame
    result = frame.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    return result[dates <= pd.Timestamp(end_date)].reset_index(drop=True)


def _future_row_count(frame: pd.DataFrame, date_col: str, end_date: str) -> int:
    if frame is None or frame.empty or date_col not in frame.columns:
        return 0
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    return int((dates > pd.Timestamp(end_date)).sum())


def _invalid_high_low_count(frame: pd.DataFrame) -> int:
    if any(column not in frame.columns for column in PRICE_COLUMNS):
        return 0
    data = frame.copy()
    for column in PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid_high = data["high"] < data[["open", "close", "low"]].max(axis=1)
    invalid_low = data["low"] > data[["open", "close", "high"]].min(axis=1)
    return int((invalid_high | invalid_low).sum())


def _non_positive_price_count(frame: pd.DataFrame) -> int:
    if any(column not in frame.columns for column in PRICE_COLUMNS):
        return 0
    data = frame[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return int((data <= 0).any(axis=1).sum())


def _max_abs_diff(left, right) -> float | None:
    diff = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs().dropna()
    return None if diff.empty else float(diff.max())


def _median_ratio(numerator, denominator) -> float | None:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)
    ratio = (pd.to_numeric(numerator, errors="coerce") / denom).dropna()
    return None if ratio.empty else float(ratio.median())


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


def _latest_trade_date(pool: pd.DataFrame) -> str:
    if pool is None or pool.empty or "trade_date" not in pool.columns:
        return pd.Timestamp.now().strftime("%Y-%m-%d")
    return str(pool["trade_date"].dropna().max())


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _clean_cell(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return text.replace("|", "/")[:240]


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

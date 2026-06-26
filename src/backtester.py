from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import get_data_config
from .indicators import enrich_5min_indicators
from .loaders import MarketDataService


DEFAULT_TOP_N = 3
DEFAULT_TARGET_RETURN_PCT = 7.0


def simulate_d2_execution(
    signal_row: pd.Series,
    minute_d2: pd.DataFrame,
    reclaim_minutes: int = 30,
    price_mode: str = "confirmation_close",
) -> dict:
    """Minimal D2 execution simulator using only intraday rows up to trigger time."""
    if minute_d2.empty:
        return {"executed": False, "reason": "D2 minute data is empty"}

    invalid_price = signal_row.get("invalid_price")
    low_absorb_min = signal_row.get("low_absorb_min")
    low_absorb_max = signal_row.get("low_absorb_max")
    rows = minute_d2.sort_values("datetime").reset_index(drop=True)
    break_started_at = None
    for _, row in rows.iterrows():
        row_time = pd.Timestamp(row["datetime"])
        low = float(row["low"])
        close = float(row["close"])
        vwap = row.get("intraday_vwap")
        if pd.notna(invalid_price):
            invalid = float(invalid_price)
            if low < invalid and close < invalid and break_started_at is None:
                break_started_at = row_time
            elif close >= invalid:
                break_started_at = None

            if break_started_at is not None and row_time - break_started_at > pd.Timedelta(minutes=reclaim_minutes):
                return {
                    "executed": False,
                    "reason": f"跌破失效位且 {reclaim_minutes} 分钟内未收回",
                    "time": str(row["datetime"]),
                }
        in_zone = (
            pd.notna(low_absorb_min)
            and pd.notna(low_absorb_max)
            and low <= float(low_absorb_max)
            and close >= float(low_absorb_min)
        )
        if in_zone and pd.notna(vwap) and close >= float(vwap):
            zone_buy_price = float(low_absorb_max)
            return {
                "executed": True,
                "reason": "回踩低吸区并站回 VWAP",
                "time": str(row["datetime"]),
                "price": zone_buy_price if price_mode == "zone_max" else close,
                "zone_buy_price": zone_buy_price,
                "confirmation_price": close,
            }
    return {"executed": False, "reason": "D2 未触发低吸验证"}


def run_top3_signal_backtest(
    signals_file: str | Path,
    top_n: int = DEFAULT_TOP_N,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    include_small: bool = False,
    fetch_through_date: str | None = None,
    days: int | None = None,
    force_refresh: bool = False,
    entry_price_mode: str = "zone_max",
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    signals = pd.read_csv(signals_file, dtype={"code": str})
    if signals.empty:
        raise RuntimeError("signals file is empty")

    selected = select_top_signals(signals, top_n=top_n, include_small=include_small)
    service = MarketDataService()
    if fetch_through_date and not selected.empty:
        _prefetch_selected_bars(
            selected,
            service=service,
            end_date=fetch_through_date,
            days=days,
            force_refresh=force_refresh,
        )
    rows = [
        evaluate_top_signal(
            row,
            service=service,
            target_return_pct=target_return_pct,
            entry_price_mode=entry_price_mode,
        )
        for _, row in selected.iterrows()
    ]
    trades = pd.DataFrame(rows)
    summary = build_top3_summary(trades, top_n=top_n, target_return_pct=target_return_pct)
    trade_date = _signals_trade_date(signals)
    output_dir = get_data_config().reports_dir / "backtest_results"
    csv_path, md_path = write_top3_backtest_reports(trades, summary, output_dir, trade_date)
    return trades, summary, csv_path, md_path


def _prefetch_selected_bars(
    selected: pd.DataFrame,
    service: MarketDataService,
    end_date: str,
    days: int | None,
    force_refresh: bool,
) -> None:
    for code in selected["code"].astype(str).str.zfill(6).drop_duplicates():
        try:
            service.get_stock_bars(code, days=days, end_date=end_date, force_refresh=force_refresh)
        except Exception:
            # The backtest row will still record missing or unusable D2 data.
            continue


def select_top_signals(signals: pd.DataFrame, top_n: int = DEFAULT_TOP_N, include_small: bool = False) -> pd.DataFrame:
    frame = signals.copy()
    frame["allowed"] = frame["allowed"].astype(str).str.lower().isin({"true", "1", "yes"})
    eligible = frame[frame["allowed"]].copy()
    if not include_small:
        eligible = eligible[eligible["signal_type"].astype(str) == "D2_LOW_ABSORB"].copy()
    if eligible.empty:
        return eligible

    eligible["position_priority"] = eligible["position_level"].map({"normal": 0, "small": 1}).fillna(9)
    for column in ("total_score", "graph_quality_score", "support_score", "active_money_score"):
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    eligible = eligible.sort_values(
        [
            "trade_date",
            "position_priority",
            "total_score",
            "graph_quality_score",
            "support_score",
            "active_money_score",
        ],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)
    eligible["daily_rank"] = eligible.groupby("trade_date").cumcount() + 1
    return eligible[eligible["daily_rank"] <= int(top_n)].reset_index(drop=True)


def evaluate_top_signal(
    signal_row: pd.Series,
    service: MarketDataService,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    entry_price_mode: str = "zone_max",
) -> dict[str, Any]:
    code = str(signal_row["code"]).zfill(6)
    signal_date = str(signal_row["trade_date"])
    result = {
        "signal_date": signal_date,
        "code": code,
        "name": signal_row.get("name", ""),
        "daily_rank": int(signal_row.get("daily_rank", 0)),
        "signal_type": signal_row.get("signal_type", ""),
        "position_level": signal_row.get("position_level", ""),
        "total_score": signal_row.get("total_score"),
        "graph_quality_score": signal_row.get("graph_quality_score"),
        "support_score": signal_row.get("support_score"),
        "active_money_score": signal_row.get("active_money_score"),
        "low_absorb_min": signal_row.get("low_absorb_min"),
        "low_absorb_max": signal_row.get("low_absorb_max"),
        "invalid_price": signal_row.get("invalid_price"),
        "d2_date": "",
        "executed": False,
        "buy_time": "",
        "buy_price": None,
        "confirmation_price": None,
        "entry_price_mode": entry_price_mode,
        "execution_reason": "",
        "d2_max_return_pct": None,
        "d2_close_return_pct": None,
        "target_return_pct": float(target_return_pct),
        "target_hit": False,
        "evaluable": False,
        "data_reason": "",
    }
    minute = service.minute_cache.read(code)
    if minute is None or minute.empty:
        result["data_reason"] = "missing 5m cache"
        return result
    if "trade_date" not in minute.columns or "datetime" not in minute.columns:
        result["data_reason"] = "5m cache missing date columns"
        return result

    d2_date = _next_trade_date(minute, signal_date)
    if not d2_date:
        result["data_reason"] = "missing D2 5m data"
        return result
    minute_d2 = minute[minute["trade_date"].astype(str) == d2_date].copy()
    if minute_d2.empty:
        result["data_reason"] = "D2 5m rows empty"
        return result

    minute_d2 = enrich_5min_indicators(minute_d2)
    execution = simulate_d2_execution(signal_row, minute_d2, price_mode=entry_price_mode)
    result["d2_date"] = d2_date
    result["executed"] = bool(execution.get("executed"))
    result["execution_reason"] = execution.get("reason", "")
    if not execution.get("executed"):
        result["data_reason"] = "not triggered"
        result["evaluable"] = True
        return result

    buy_time = str(execution.get("time", ""))
    buy_price = float(execution.get("price"))
    after_buy = minute_d2[pd.to_datetime(minute_d2["datetime"], errors="coerce") >= pd.Timestamp(buy_time)].copy()
    if after_buy.empty:
        result["data_reason"] = "no bars after buy"
        result["evaluable"] = True
        return result

    max_high = float(pd.to_numeric(after_buy["high"], errors="coerce").max())
    close_price = float(pd.to_numeric(after_buy["close"], errors="coerce").iloc[-1])
    max_return_pct = (max_high / buy_price - 1.0) * 100.0
    close_return_pct = (close_price / buy_price - 1.0) * 100.0
    result.update(
        {
            "buy_time": buy_time,
            "buy_price": buy_price,
            "confirmation_price": execution.get("confirmation_price"),
            "d2_max_return_pct": max_return_pct,
            "d2_close_return_pct": close_return_pct,
            "target_hit": bool(max_return_pct >= float(target_return_pct)),
            "evaluable": True,
            "data_reason": "",
        }
    )
    return result


def build_top3_summary(
    trades: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "top_n": int(top_n),
                    "target_return_pct": float(target_return_pct),
                    "selected_count": 0,
                }
            ]
        )
    evaluable = trades[trades["evaluable"].fillna(False).astype(bool)].copy()
    executed = evaluable[evaluable["executed"].fillna(False).astype(bool)].copy()
    target_hit_count = int(executed["target_hit"].fillna(False).astype(bool).sum()) if not executed.empty else 0
    return pd.DataFrame(
        [
            {
                "top_n": int(top_n),
                "target_return_pct": float(target_return_pct),
                "selected_count": int(len(trades)),
                "evaluable_count": int(len(evaluable)),
                "executed_count": int(len(executed)),
                "execution_rate": _safe_rate(len(executed), len(evaluable)),
                "target_hit_count": target_hit_count,
                "target_hit_rate": _safe_rate(target_hit_count, len(executed)),
                "avg_d2_max_return_pct": _mean_or_none(executed, "d2_max_return_pct"),
                "avg_d2_close_return_pct": _mean_or_none(executed, "d2_close_return_pct"),
                "median_d2_max_return_pct": _median_or_none(executed, "d2_max_return_pct"),
                "median_d2_close_return_pct": _median_or_none(executed, "d2_close_return_pct"),
            }
        ]
    )


def write_top3_backtest_reports(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    trade_date: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"top3_backtest_{trade_date}.csv"
    md_path = output_dir / f"top3_backtest_{trade_date}.md"
    trades.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_top3_backtest_markdown(trades, summary, trade_date), encoding="utf-8")
    return csv_path, md_path


def build_top3_backtest_markdown(trades: pd.DataFrame, summary: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} Top3 Backtest", ""]
    if summary.empty:
        lines.append("No summary.")
        return "\n".join(lines)
    item = summary.iloc[0]
    lines.extend(
        [
            "## Summary",
            "",
            f"- top_n: **{int(item.get('top_n', 0))}**",
            f"- target return: **{float(item.get('target_return_pct', 0)):.2f}%**",
            f"- selected: **{int(item.get('selected_count', 0))}**",
            f"- evaluable: **{int(item.get('evaluable_count', 0))}**",
            f"- executed: **{int(item.get('executed_count', 0))}**",
            f"- execution rate: **{_format_pct(item.get('execution_rate'))}**",
            f"- target hit rate: **{_format_pct(item.get('target_hit_rate'))}**",
            f"- avg D2 max return: **{_format_number(item.get('avg_d2_max_return_pct'))}%**",
            f"- avg D2 close return: **{_format_number(item.get('avg_d2_close_return_pct'))}%**",
            "",
        ]
    )
    if trades.empty:
        lines.append("No selected trades.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Selected",
            "",
            "| date | rank | code | name | score | executed | buy | confirm | max% | close% | hit | reason |",
            "|---|---:|---|---|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in trades.iterrows():
        lines.append(
            "| {date} | {rank} | {code} | {name} | {score} | {executed} | {buy} | {confirm} | {max_ret} | {close_ret} | {hit} | {reason} |".format(
                date=row.get("signal_date", ""),
                rank=row.get("daily_rank", ""),
                code=row.get("code", ""),
                name=row.get("name", ""),
                score=_format_number(row.get("total_score")),
                executed=str(bool(row.get("executed"))),
                buy=_format_number(row.get("buy_price")),
                confirm=_format_number(row.get("confirmation_price")),
                max_ret=_format_number(row.get("d2_max_return_pct")),
                close_ret=_format_number(row.get("d2_close_return_pct")),
                hit=str(bool(row.get("target_hit"))),
                reason=str(row.get("execution_reason") or row.get("data_reason") or "")[:120],
            )
        )
    return "\n".join(lines)


def _next_trade_date(minute: pd.DataFrame, signal_date: str) -> str:
    dates = sorted(pd.to_datetime(minute["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    for date in dates:
        if date > signal_date:
            return date
    return ""


def _signals_trade_date(signals: pd.DataFrame) -> str:
    if "trade_date" in signals.columns and signals["trade_date"].notna().any():
        return str(signals["trade_date"].dropna().max())
    return pd.Timestamp.now().strftime("%Y-%m-%d")


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _median_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"

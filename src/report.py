from __future__ import annotations

from pathlib import Path

import pandas as pd

from .signal_engine import Signal


def write_signal_reports(signals: list[Signal], output_dir: Path, trade_date: str | None = None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if trade_date is None:
        trade_date = signals[0].trade_date if signals else pd.Timestamp.now().strftime("%Y-%m-%d")
    frame = pd.DataFrame([signal.to_dict() for signal in signals])
    csv_path = output_dir / f"signals_{trade_date}.csv"
    md_path = output_dir / f"signals_{trade_date}.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_markdown_report(frame, trade_date), encoding="utf-8")
    return csv_path, md_path


def write_data_quality_reports(quality_rows: list[dict], output_dir: Path, trade_date: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(quality_rows)
    csv_path = output_dir / f"data_quality_{trade_date}.csv"
    md_path = output_dir / f"data_quality_{trade_date}.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_data_quality_markdown(frame, trade_date), encoding="utf-8")
    return csv_path, md_path


def build_markdown_report(frame: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} D2 交易预案", ""]

    if frame.empty:
        lines.append("无信号。")
        return "\n".join(lines)

    allowed = frame[frame["allowed"] == True]
    normal = allowed[allowed["signal_type"] == "D2_LOW_ABSORB"]
    small = allowed[allowed["signal_type"] == "D2_WATCH_OR_SMALL"]
    watch = frame[frame["allowed"] != True]

    def _days_count(df, d):
        return len(df[df["days_since_d0"] == d])

    lines.append("## 概览")
    lines.append("")
    lines.append(f"- 总信号: **{len(frame)}**")
    lines.append(f"- 推荐交易 (D2_LOW_ABSORB): **{len(normal)}** 只")
    lines.append(f"  - 天数=1 (D2最佳窗口): **{_days_count(normal, 1)}** 只")
    lines.append(f"  - 天数=2 (D2次优窗口): **{_days_count(normal, 2)}** 只")
    lines.append(f"  - 天数=3 (D2末端窗口): **{_days_count(normal, 3)}** 只")
    lines.append(f"- 小仓试探 (D2_WATCH_OR_SMALL): **{len(small)}** 只")
    lines.append(f"- 仅观察 (WATCH_ONLY): **{len(watch)}** 只")
    if len(allowed) > 0:
        lines.append(f"- 平均总分 (可交易): **{allowed['total_score'].mean():.1f}**")
        lines.append(f"- 总分区间 (可交易): **{allowed['total_score'].min():.1f} ~ {allowed['total_score'].max():.1f}**")
    lines.append("")

    header = "| # | 代码 | 名称 | D0日期 | 连板 | 天数 | 总分 | 图形 | 活跃 | 承接 | 题材 | 低吸区间 | 失效位 |"
    sep    = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    # Small position section
    lines.append("## 小仓试探 — D2 观察或小仓 (small 仓位)")
    lines.append("")
    if small.empty:
        lines.append("无。")
        lines.append("")
    else:
        sorted_small = small.sort_values("total_score", ascending=False).reset_index(drop=True)
        lines.append(header)
        lines.append(sep)
        for rank, (_, row) in enumerate(sorted_small.iterrows(), 1):
            lines.append(_render_row(rank, row))
        lines.append("")

    # Recommended trades: split by timing tier
    lines.append("## 推荐交易 — D2 低吸 (normal 仓位)")
    lines.append("")
    if normal.empty:
        lines.append("无。")
        lines.append("")
    else:
        tier_config = [
            (1, "D2 最佳窗口 — 昨日涨停，今日分歧，明日执行"),
            (2, "D2 次优窗口 — 前日涨停，分歧第二日"),
            (3, "D2 末端窗口 — 分歧第三日，注意时效衰减"),
        ]
        for days_val, tier_label in tier_config:
            tier_df = normal[normal["days_since_d0"] == days_val]
            if tier_df.empty:
                continue
            sorted_tier = tier_df.sort_values("total_score", ascending=False).reset_index(drop=True)
            lines.append(f"### {tier_label}（共 {len(sorted_tier)} 只）")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for rank, (_, row) in enumerate(sorted_tier.iterrows(), 1):
                lines.append(_render_row(rank, row))
            lines.append("")

    # Watch list
    lines.append("## 观察列表 — 不交易，仅跟踪")
    lines.append("")
    if watch.empty:
        lines.append("无。")
        lines.append("")
    else:
        sorted_watch = watch.sort_values(["days_since_d0", "total_score"], ascending=[True, False]).reset_index(drop=True)
        lines.append(header)
        lines.append(sep)
        for rank, (_, row) in enumerate(sorted_watch.iterrows(), 1):
            lines.append(_render_row(rank, row))
        lines.append("")

    lines.append("## 字段说明")
    lines.append("")
    lines.append("| 字段 | 含义 |")
    lines.append("|---|---|")
    lines.append("| D0日期 | 最近一次涨停日期（涨停起点） |")
    lines.append("| 连板 | D0 前的连续涨停板数 |")
    lines.append("| 天数 | 距 D0 的天数（1=D0次日/D1分歧日，值越小越优先） |")
    lines.append("| 总分 | 图形×0.35 + 活跃×0.25 + 承接×0.25 + 题材×0.15 |")
    lines.append("| 图形 | 趋势结构、均线排列、K线位置 (≥60 可交易) |")
    lines.append("| 活跃 | 量能放大、振幅、分时VWAP争夺 (≥55) |")
    lines.append("| 承接 | 低点支撑、VWAP站位、关键位收回 (≥60 可交易，<45 无效) |")
    lines.append("| 题材 | 同行业涨停联动强度 |")
    lines.append("| 低吸区间 | D2 低吸价格范围 |")
    lines.append("| 失效位 | 跌破此价格放弃预案 |")
    lines.append("")
    lines.append("> 优先关注「天数=1」的最佳窗口标的；天数越大，分歧时效越弱。")
    lines.append("> D2 只验证预案中的关键位，不临时追高。")
    return "\n".join(lines)


def build_data_quality_markdown(frame: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} 数据质量报告", ""]
    if frame.empty:
        lines.append("无数据质量记录。")
        return "\n".join(lines)

    status_counts = frame["status"].value_counts(dropna=False) if "status" in frame.columns else pd.Series(dtype=int)
    ok_count = int(status_counts.get("ok", 0))
    failed_count = int(status_counts.get("failed", 0))
    ma_ok = _bool_count(frame, "daily_ma_coverage_ok")
    close_ok = _bool_count(frame, "daily_minute_close_check_ok")
    lines.append("## 概览")
    lines.append("")
    lines.append(f"- 总标的: **{len(frame)}**")
    lines.append(f"- 数据成功: **{ok_count}**")
    lines.append(f"- 数据失败: **{failed_count}**")
    lines.append(f"- 最新日线 MA 覆盖正常: **{ma_ok}**")
    lines.append(f"- 日线/分钟收盘价校验通过: **{close_ok}**")
    lines.append("")

    lines.append("## 数据源")
    lines.append("")
    lines.append("| 类型 | 来源 | 数量 |")
    lines.append("|---|---|---:|")
    for source, count in _source_counts(frame, "daily_source").items():
        lines.append(f"| 日线 | {source} | {count} |")
    for source, count in _source_counts(frame, "minute_source").items():
        lines.append(f"| 5分钟 | {source} | {count} |")
    lines.append("")

    failed = frame[frame.get("status", "") == "failed"] if "status" in frame.columns else pd.DataFrame()
    if not failed.empty:
        lines.append("## 失败标的")
        lines.append("")
        lines.append("| 代码 | 名称 | D0日期 | 错误 |")
        lines.append("|---|---|---|---|")
        for _, row in failed.sort_values("code").iterrows():
            lines.append(
                f"| {row.get('code', '')} | {row.get('name', '')} | {row.get('d0_date', '')} | {str(row.get('error', ''))[:180]} |"
            )
        lines.append("")

    warnings = frame[(frame.get("warnings", "").fillna("").astype(str) != "")] if "warnings" in frame.columns else pd.DataFrame()
    if not warnings.empty:
        lines.append("## 质量提示")
        lines.append("")
        lines.append("| 代码 | 名称 | 日线源 | 5分钟源 | 提示 |")
        lines.append("|---|---|---|---|---|")
        for _, row in warnings.sort_values("code").head(80).iterrows():
            lines.append(
                f"| {row.get('code', '')} | {row.get('name', '')} | {row.get('daily_source', '')} | {row.get('minute_source', '')} | {row.get('warnings', '')} |"
            )
        if len(warnings) > 80:
            lines.append(f"| ... | ... | ... | ... | 另有 {len(warnings) - 80} 条，详见 CSV |")
        lines.append("")

    lines.append("## 字段说明")
    lines.append("")
    lines.append("| 字段 | 含义 |")
    lines.append("|---|---|")
    lines.append("| daily_history_rows | 参与日线 MA 计算的日线历史交易日数量 |")
    lines.append("| daily_required_days | 当前策略要求的最少日线历史交易日数量 |")
    lines.append("| daily_ma_coverage_ok | 最新行 MA5/MA10/MA20/MA30 是否都存在 |")
    lines.append("| daily_minute_close_check_ok | 日线收盘与当日最后一根 5 分钟收盘是否基本一致 |")
    lines.append("| missing_*_count | 对应字段缺失数量 |")
    return "\n".join(lines)


def _render_row(rank: int, row) -> str:
    zone = format_zone(row.get("low_absorb_min"), row.get("low_absorb_max"))
    invalid = row.get("invalid_price")
    invalid_text = "" if pd.isna(invalid) or invalid is None else f"{float(invalid):.2f}"
    return (
        f"| {rank} | {row.get('code', '')} | {row.get('name', '')} | "
        f"{row.get('d0_date', '')} | "
        f"{int(row.get('consecutive_boards', 0))} | "
        f"{int(row.get('days_since_d0', 0))} | "
        f"{float(row.get('total_score', 0)):.1f} | "
        f"{float(row.get('graph_quality_score', 0)):.0f} | "
        f"{float(row.get('active_money_score', 0)):.0f} | "
        f"{float(row.get('support_score', 0)):.0f} | "
        f"{float(row.get('theme_score', 0)):.0f} | "
        f"{zone} | {invalid_text} |"
    )


def format_zone(low, high) -> str:
    if pd.isna(low) or pd.isna(high) or low is None or high is None:
        return ""
    lo = float(low)
    hi = float(high)
    if abs(lo - hi) < 0.01:
        return f"{lo:.2f}"
    return f"{lo:.2f}~{hi:.2f}"


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _source_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].fillna("").astype(str).replace("", "unknown").value_counts()
    return {str(index): int(value) for index, value in counts.items()}

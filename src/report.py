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


def build_markdown_report(frame: pd.DataFrame, trade_date: str) -> str:
    lines = [f"# {trade_date} D2 交易预案", ""]
    if frame.empty:
        lines.append("无信号。")
        return "\n".join(lines)

    show = frame.sort_values(["allowed", "total_score"], ascending=[False, False])
    lines.append("| 代码 | 名称 | 类型 | 允许 | 仓位 | 总分 | 图形 | 活跃 | 承接 | 题材 | 低吸区 | 失效位 |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|")
    for _, row in show.iterrows():
        low_zone = format_zone(row.get("low_absorb_min"), row.get("low_absorb_max"))
        invalid = row.get("invalid_price")
        invalid_text = "" if pd.isna(invalid) else f"{float(invalid):.2f}"
        lines.append(
            "| {code} | {name} | {signal_type} | {allowed} | {position_level} | "
            "{total_score:.2f} | {graph_quality_score:.2f} | {active_money_score:.2f} | "
            "{support_score:.2f} | {theme_score:.2f} | {low_zone} | {invalid} |".format(
                code=row.get("code", ""),
                name=row.get("name", ""),
                signal_type=row.get("signal_type", ""),
                allowed="是" if bool(row.get("allowed")) else "否",
                position_level=row.get("position_level", ""),
                total_score=float(row.get("total_score", 0)),
                graph_quality_score=float(row.get("graph_quality_score", 0)),
                active_money_score=float(row.get("active_money_score", 0)),
                support_score=float(row.get("support_score", 0)),
                theme_score=float(row.get("theme_score", 0)),
                low_zone=low_zone,
                invalid=invalid_text,
            )
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("本报告由 D1 收盘后数据生成。D2 只验证预案中的关键位，不临时追高。")
    return "\n".join(lines)


def format_zone(low, high) -> str:
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(low):.2f}-{float(high):.2f}"

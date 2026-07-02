from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("reports/v005_set_selector")
DEFAULT_INITIAL_TRAIN_DAYS = 8
DEFAULT_OBJECTIVES = "all"
TOP_N = 3

SELECTION_OBJECTIVES = (
    "all_hit_then_zero",
    "target_then_all_hit",
    "realized_then_all_hit",
    "all_hit_zero_realized",
    "target_all_hit_realized",
)

REQUIRED_SELECTED_COLUMNS = [
    "grid_id",
    "signal_date",
    "codes",
    "hit_count",
    "all_hit",
    "avg_high_return",
    "avg_realized_return",
]

SUMMARY_COLUMNS = [
    "selection_objective",
    "initial_train_days",
    "validation_date_count",
    "selected_ticket_count",
    "top3_target_rate",
    "top3_all_hit_rate",
    "hit_count_0_days",
    "hit_count_1_days",
    "hit_count_2_days",
    "hit_count_3_days",
    "avg_top3_high_return",
    "avg_top3_realized_return",
    "selected_grid_count",
    "grid_switch_count",
    "selected_grid_ids",
]

HISTORY_COLUMNS = [
    "selection_objective",
    "validation_date",
    "selected_grid_id",
    "train_date_count",
    "train_top3_target_rate",
    "train_top3_all_hit_rate",
    "train_hit_count_0_days",
    "train_hit_count_3_days",
    "train_avg_top3_realized_return",
    "validation_codes",
    "validation_hit_count",
    "validation_all_hit",
    "validation_avg_high_return",
    "validation_avg_realized_return",
]


def run_objective_sweep(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    objectives: str = DEFAULT_OBJECTIVES,
    top_n: int = TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    root = Path(input_dir)
    selected_path = root / "v005_selected_combos.csv"
    if not selected_path.exists():
        raise RuntimeError(f"missing {selected_path}; run `python -m src.v005_set_selector` first")

    selected = pd.read_csv(selected_path, dtype={"codes": str})
    missing = [column for column in REQUIRED_SELECTED_COLUMNS if column not in selected.columns]
    if missing:
        raise RuntimeError(f"{selected_path} missing required columns: {missing}")
    selected = prepare_selected_combos(selected)

    objective_list = parse_objectives(objectives)
    history_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for objective in objective_list:
        history, summary = walk_forward_for_objective(
            selected_combos=selected,
            objective=objective,
            initial_train_days=int(initial_train_days),
            top_n=int(top_n),
        )
        history_rows.extend(history.to_dict("records"))
        summary_rows.append(summary)

    history_frame = pd.DataFrame(history_rows)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = summary_frame.sort_values(
        ["top3_all_hit_rate", "hit_count_0_days", "avg_top3_realized_return", "top3_target_rate"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)

    summary_csv = root / "v005_wf_objective_summary.csv"
    history_csv = root / "v005_wf_objective_history.csv"
    report_path = root / "v005_wf_objective_report.md"
    summary_frame[SUMMARY_COLUMNS].to_csv(summary_csv, index=False, encoding="utf-8-sig")
    history_frame[HISTORY_COLUMNS].to_csv(history_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(summary_frame, history_frame, selected_path, initial_train_days, objective_list, top_n), encoding="utf-8")
    return summary_frame[SUMMARY_COLUMNS], history_frame[HISTORY_COLUMNS], report_path


def prepare_selected_combos(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["grid_id"] = pd.to_numeric(result["grid_id"], errors="coerce").astype("Int64")
    result["signal_date"] = result["signal_date"].astype(str)
    result["codes"] = result["codes"].astype(str)
    result["hit_count"] = pd.to_numeric(result["hit_count"], errors="coerce").fillna(0).astype(int)
    result["all_hit"] = _bool_series(result["all_hit"])
    result["avg_high_return"] = pd.to_numeric(result["avg_high_return"], errors="coerce")
    result["avg_realized_return"] = pd.to_numeric(result["avg_realized_return"], errors="coerce")
    result = result.dropna(subset=["grid_id"]).copy()
    result["grid_id"] = result["grid_id"].astype(int)
    return result


def parse_objectives(text: str) -> list[str]:
    if str(text).strip().lower() == "all":
        objectives = list(SELECTION_OBJECTIVES)
    else:
        objectives = [part.strip() for part in str(text).split(",") if part.strip()]
    if not objectives:
        raise RuntimeError("objective list is empty")
    bad = [objective for objective in objectives if objective not in SELECTION_OBJECTIVES]
    if bad:
        raise RuntimeError(f"unsupported objective(s): {bad}; expected one of {SELECTION_OBJECTIVES}")
    deduped: list[str] = []
    for objective in objectives:
        if objective not in deduped:
            deduped.append(objective)
    return deduped


def walk_forward_for_objective(
    selected_combos: pd.DataFrame,
    objective: str,
    initial_train_days: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = sorted(selected_combos["signal_date"].dropna().astype(str).unique().tolist())
    if int(initial_train_days) <= 0:
        raise RuntimeError("initial_train_days must be positive")
    if int(initial_train_days) >= len(dates):
        raise RuntimeError(f"initial_train_days={initial_train_days} must be smaller than date_count={len(dates)}")

    history_rows: list[dict[str, Any]] = []
    validation_rows: list[pd.Series] = []
    for valid_index in range(int(initial_train_days), len(dates)):
        validation_date = dates[valid_index]
        train_dates = dates[:valid_index]
        train_panel = selected_combos[selected_combos["signal_date"].isin(train_dates)].copy()
        train_summary = summarize_grid_panel(train_panel, top_n=top_n)
        if train_summary.empty:
            raise RuntimeError(f"empty train summary before validation date {validation_date}")
        train_summary = sort_summary_by_objective(train_summary, objective).reset_index(drop=True)
        chosen = train_summary.iloc[0]
        grid_id = int(chosen["grid_id"])
        validation = selected_combos[(selected_combos["signal_date"] == validation_date) & (selected_combos["grid_id"] == grid_id)].copy()
        if validation.empty:
            raise RuntimeError(f"missing validation row for grid_id={grid_id}, validation_date={validation_date}")
        validation_row = validation.iloc[0]
        validation_rows.append(validation_row)
        history_rows.append(
            {
                "selection_objective": objective,
                "validation_date": validation_date,
                "selected_grid_id": grid_id,
                "train_date_count": int(chosen["date_count"]),
                "train_top3_target_rate": float(chosen["top3_target_rate"]),
                "train_top3_all_hit_rate": float(chosen["top3_all_hit_rate"]),
                "train_hit_count_0_days": int(chosen["hit_count_0_days"]),
                "train_hit_count_3_days": int(chosen["hit_count_3_days"]),
                "train_avg_top3_realized_return": float(chosen["avg_top3_realized_return"]),
                "validation_codes": str(validation_row["codes"]),
                "validation_hit_count": int(validation_row["hit_count"]),
                "validation_all_hit": bool(validation_row["all_hit"]),
                "validation_avg_high_return": float(validation_row["avg_high_return"]),
                "validation_avg_realized_return": float(validation_row["avg_realized_return"]),
            }
        )

    validation_frame = pd.DataFrame(validation_rows)
    history_frame = pd.DataFrame(history_rows)
    summary = summarize_validation(validation_frame, objective, initial_train_days, top_n, history_frame)
    return history_frame[HISTORY_COLUMNS], summary


def summarize_grid_panel(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grid_id, group in frame.groupby("grid_id", dropna=False):
        rows.append(summarize_combo_rows(group, grid_id=int(grid_id), top_n=top_n))
    return pd.DataFrame(rows)


def summarize_combo_rows(frame: pd.DataFrame, grid_id: int, top_n: int) -> dict[str, Any]:
    hit_count = pd.to_numeric(frame["hit_count"], errors="coerce").fillna(0).astype(int)
    return {
        "grid_id": int(grid_id),
        "date_count": int(frame["signal_date"].nunique()),
        "selected_ticket_count": int(len(frame) * int(top_n)),
        "top3_target_rate": _safe_rate(int(hit_count.sum()), int(len(frame) * int(top_n))),
        "top3_all_hit_rate": _safe_rate(int(frame["all_hit"].astype(bool).sum()), int(len(frame))),
        "hit_count_0_days": int((hit_count == 0).sum()),
        "hit_count_1_days": int((hit_count == 1).sum()),
        "hit_count_2_days": int((hit_count == 2).sum()),
        "hit_count_3_days": int((hit_count == 3).sum()),
        "avg_top3_high_return": _mean(frame["avg_high_return"]),
        "avg_top3_realized_return": _mean(frame["avg_realized_return"]),
    }


def summarize_validation(
    validation_frame: pd.DataFrame,
    objective: str,
    initial_train_days: int,
    top_n: int,
    history_frame: pd.DataFrame,
) -> dict[str, Any]:
    base = summarize_combo_rows(validation_frame, grid_id=-1, top_n=top_n)
    grid_ids = history_frame["selected_grid_id"].astype(int).tolist() if not history_frame.empty else []
    return {
        "selection_objective": objective,
        "initial_train_days": int(initial_train_days),
        "validation_date_count": int(base["date_count"]),
        "selected_ticket_count": int(base["selected_ticket_count"]),
        "top3_target_rate": float(base["top3_target_rate"]),
        "top3_all_hit_rate": float(base["top3_all_hit_rate"]),
        "hit_count_0_days": int(base["hit_count_0_days"]),
        "hit_count_1_days": int(base["hit_count_1_days"]),
        "hit_count_2_days": int(base["hit_count_2_days"]),
        "hit_count_3_days": int(base["hit_count_3_days"]),
        "avg_top3_high_return": float(base["avg_top3_high_return"]),
        "avg_top3_realized_return": float(base["avg_top3_realized_return"]),
        "selected_grid_count": int(len(set(grid_ids))),
        "grid_switch_count": int(sum(1 for previous, current in zip(grid_ids, grid_ids[1:]) if previous != current)),
        "selected_grid_ids": ",".join(str(grid_id) for grid_id in sorted(set(grid_ids))),
    }


def sort_summary_by_objective(summary: pd.DataFrame, objective: str) -> pd.DataFrame:
    if objective == "all_hit_then_zero":
        return summary.sort_values(
            ["top3_all_hit_rate", "hit_count_0_days", "top3_target_rate", "avg_top3_realized_return"],
            ascending=[False, True, False, False],
        )
    if objective == "target_then_all_hit":
        return summary.sort_values(
            ["top3_target_rate", "top3_all_hit_rate", "hit_count_0_days", "avg_top3_realized_return"],
            ascending=[False, False, True, False],
        )
    if objective == "realized_then_all_hit":
        return summary.sort_values(
            ["avg_top3_realized_return", "top3_all_hit_rate", "hit_count_0_days", "top3_target_rate"],
            ascending=[False, False, True, False],
        )
    if objective == "all_hit_zero_realized":
        return summary.sort_values(
            ["top3_all_hit_rate", "hit_count_0_days", "avg_top3_realized_return", "top3_target_rate"],
            ascending=[False, True, False, False],
        )
    if objective == "target_all_hit_realized":
        return summary.sort_values(
            ["top3_target_rate", "top3_all_hit_rate", "avg_top3_realized_return", "hit_count_0_days"],
            ascending=[False, False, False, True],
        )
    raise RuntimeError(f"unsupported objective: {objective}")


def build_report(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    selected_path: Path,
    initial_train_days: int,
    objectives: list[str],
    top_n: int,
) -> str:
    lines = [
        "# v005 walk-forward objective sweep",
        "",
        "## Scope",
        "",
        "This is a research-only post-processing sweep over `v005_selected_combos.csv`.",
        "It does not generate new combinations and does not touch daily production logic.",
        "Each validation date selects its grid using only earlier selected-combo outcomes.",
        "",
        "## Configuration",
        "",
        f"- selected combos: `{selected_path}`",
        f"- initial_train_days: `{initial_train_days}`",
        f"- top_n: `{top_n}`",
        f"- objectives: `{','.join(objectives)}`",
        "",
        "## Objective summary",
        "",
    ]
    lines.extend(_markdown_table(summary, SUMMARY_COLUMNS))
    lines.extend(["", "## Objective history", ""])
    lines.extend(_markdown_table(history, HISTORY_COLUMNS))
    lines.extend(
        [
            "",
            "## Research interpretation checklist",
            "",
            "- Prefer objectives that preserve top3_all_hit_rate while improving avg_top3_realized_return.",
            "- Reject objectives that improve realized return only by collapsing all-hit rate or increasing 0-hit days.",
            "- Treat this as an objective-selection diagnostic, not a production model.",
        ]
    )
    return "\n".join(lines)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def _safe_rate(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def _mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return np.nan
    return float(series.mean())


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame is None or frame.empty:
        return ["_No rows._"]
    usable = [column for column in columns if column in frame.columns]
    table = frame[usable].copy()
    for column in table.columns:
        table[column] = table[column].map(_format_markdown_value)
    lines = [
        "| " + " | ".join(usable) + " |",
        "| " + " | ".join(["---"] * len(usable)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in usable) + " |")
    return lines


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    return text.replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare v005 walk-forward grid-selection objectives.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    parser.add_argument("--objectives", default=DEFAULT_OBJECTIVES, help="Comma-separated objective list, or 'all'.")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args(argv)

    summary, history, report_path = run_objective_sweep(
        input_dir=args.input_dir,
        initial_train_days=args.initial_train_days,
        objectives=args.objectives,
        top_n=args.top_n,
    )
    print(f"objective summary rows: {len(summary)}")
    print(f"objective history rows: {len(history)}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best objective: {best['selection_objective']}")
        print(f"best top3_all_hit_rate: {float(best['top3_all_hit_rate']):.4f}")
        print(f"best avg_realized_return: {float(best['avg_top3_realized_return']):.4f}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

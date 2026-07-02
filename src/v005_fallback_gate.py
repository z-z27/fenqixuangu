from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_V005_DIR = Path("reports/v005_set_selector")
DEFAULT_TOP_N = 3

TARGET = "target7_d2open_d3high"
HIGH = "d2open_d3high_return_pct"
REAL = "realized_return_pct"


def load(path):
    df = pd.read_csv(path)
    return df


def topn(df, col, n=3):
    return (df.sort_values(["signal_date", col, "code"]) 
              .groupby("signal_date")
              .head(n))


def summarize(group):
    hit = int(group[TARGET].astype(bool).sum())
    return hit, float(group[REAL].mean()) if len(group) else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v005-dir", default=str(DEFAULT_V005_DIR))
    args = p.parse_args()

    vdir = Path(args.v005_dir)

    combos = pd.read_csv(vdir / "v005_selected_combos.csv")
    hist = pd.read_csv(vdir / "v005_wf_objective_history.csv")

    results = []

    for _, row in hist.iterrows():
        date = str(row["validation_date"])
        grid = int(row["selected_grid_id"])

        day = combos[(combos["signal_date"] == date) & (combos["grid_id"] == grid)]
        if day.empty:
            continue
        codes = day.iloc[0]["codes"].split(",")

        results.append({
            "date": date,
            "hit": int(day.iloc[0]["hit_count"]),
            "real": float(day.iloc[0]["avg_realized_return"]),
        })

    df = pd.DataFrame(results)

    print("days:", len(df))
    print("avg hit:", df["hit"].mean() if len(df) else 0)
    print("avg real:", df["real"].mean() if len(df) else 0)


if __name__ == "__main__":
    main()

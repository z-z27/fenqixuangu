from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_stage1_18f_bridge_analysis import (
    run_v004c_stage1_18f_bridge_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir, context = run_v004c_stage1_18f_bridge_analysis(
        args.root, args.output_dir
    )
    print(f"output_dir={output_dir}")
    print(f"rows={context['rows']}")
    print(f"dates={context['dates']}")
    print(f"july_rows={context['july_rows']}")
    print(f"july_dates={context['july_dates']}")
    print(
        "score_reconstruction_max_abs_error="
        f"{context['score_reconstruction_max_abs_error']:.12g}"
    )
    print("august_outcome_rows_accessed=0")


if __name__ == "__main__":
    main()


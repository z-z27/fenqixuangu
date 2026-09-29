from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_s2_winner_vs_loss_failure_attribution import (
    analyze,
    output_sha256s,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen-S2 winner-vs-loss failure attribution (no fitting)."
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    context = analyze(args.root)
    output_dir = write_outputs(context, args.output_dir)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"ROWS={context['audit']['rows']}")
    print(f"DATES={context['audit']['dates']}")
    print(f"MODEL_REFITS={context['audit']['model_refits']}")
    print(f"AUGUST_SIGNAL_DATE_OUTCOME_ROWS_ACCESSED={context['audit']['august_signal_date_outcome_rows_accessed']}")
    print(f"WINNER_LOSS_FAILURE_ATTRIBUTION={context['state']}")
    print("NEXT_ACTION=STOP_AND_REVIEW")
    for name, digest in output_sha256s(output_dir).items():
        print(f"SHA256 {digest} {name}")


if __name__ == "__main__":
    main()

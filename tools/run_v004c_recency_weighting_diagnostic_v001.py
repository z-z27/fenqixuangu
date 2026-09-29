from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_recency_weighting_diagnostic import (
    analyze,
    output_sha256s,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    context = analyze(args.root)
    output_dir = write_outputs(context, args.output_dir)
    print(f"output_dir={output_dir}")
    print(f"train_rows={len(context['train'])}")
    print(f"train_dates={context['train']['signal_date'].nunique()}")
    print(f"july_rows={len(context['july'])}")
    print(f"july_dates={context['july']['signal_date'].nunique()}")
    print(f"equal_parity={context['parity']}")
    print(f"recency_weighting_state={context['state']}")
    print("next_action=STOP_AND_REVIEW")
    print("august_signal_outcome_accessed=NO")
    for name, digest in output_sha256s(output_dir).items():
        print(f"sha256 {name} {digest}")


if __name__ == "__main__":
    main()


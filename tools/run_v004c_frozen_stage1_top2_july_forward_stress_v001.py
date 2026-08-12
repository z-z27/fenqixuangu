from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_frozen_stage1_top2_july_forward_stress import (
    run_v004c_frozen_stage1_top2_july_forward_stress,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir, context = run_v004c_frozen_stage1_top2_july_forward_stress(
        args.root, args.output_dir
    )
    print(f"output_dir={output_dir}")
    print(f"prediction_lock_sha256={context['provenance']['prediction_lock_sha256']}")
    print(f"signal={context['decision']['signal']}")
    print(f"failure={context['decision']['primary_failure']}")
    print(f"architecture_state={context['decision']['architecture_state']}")


if __name__ == "__main__":
    main()

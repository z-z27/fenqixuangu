"""Build the frozen-S2 replaceable-loss manual-review data pack."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_s2_replaceable_loss_case_pack import write_outputs


def main() -> int:
    output_dir, metrics = write_outputs(ROOT)
    print(output_dir)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


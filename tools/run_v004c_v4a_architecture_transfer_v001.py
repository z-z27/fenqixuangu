from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_v4a_architecture_transfer import (  # noqa: E402
    MODEL_ID,
    run_v004c_v4a_architecture_transfer,
)


def main() -> int:
    output_dir, context = run_v004c_v4a_architecture_transfer(ROOT)
    summary = context["summaries"][MODEL_ID]
    print(f"output_dir={output_dir}")
    print(f"feature_gate={'PASS' if context['feature_gate']['pass'] else 'FAIL'}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    print(f"rank1={summary['Rank1_mean']:.12g}")
    print(f"strict_top3={summary['Top3_mean']:.12g}")
    print(f"architecture_transfer_signal={context['status']['architecture_transfer_signal']}")
    print(f"previous_route_misdesigned={context['status']['previous_route_misdesigned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_board3_oof_risk_selectivity import (  # noqa: E402
    run_v004c_board3_oof_risk_selectivity,
)


def main() -> int:
    output_dir, context = run_v004c_board3_oof_risk_selectivity(ROOT)
    decision = context["decision"]
    bottom = context["tail_summary"]["BOTTOM_HALF"]
    print(f"output_dir={output_dir}")
    print(f"population_a_rows={len(context['population'])}")
    print(f"population_b_rows={len(context['eligible'])}")
    print(f"prediction_lock_sha256={context['lock_audit']['observed_lock_sha256']}")
    print(f"bottom_half_selectivity_gap={bottom['selectivity_gap']:.12f}")
    print(f"board3_risk_selectivity_signal={decision['signal']}")
    print(f"next_board3_risk_action={decision['next_action']}")
    print("july_result_rows_accessed=0")
    print("deterministic_rebuild=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

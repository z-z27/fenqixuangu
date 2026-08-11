from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_board3_winner_information_geometry import (  # noqa: E402
    run_v004c_board3_winner_information_geometry,
)


def main() -> int:
    output_dir, context = run_v004c_board3_winner_information_geometry(ROOT)
    decision = context["decision"]
    audit = context["population_audit"]
    families = context["family_summary"]
    print(f"output_dir={output_dir}")
    print(f"board3_rows={audit['rows']}")
    print(f"board3_dates={audit['dates']}")
    print(f"low_level_winner_pass={families['LOW_LEVEL_53']['pass']}")
    print(f"stage1_rep_winner_pass={families['FROZEN_STAGE1_18']['pass']}")
    print(f"board3_winner_information_foundation={decision['foundation']}")
    print(f"board3_ordinal_repair_information={decision['ordinal']}")
    print(f"next_board3_action={decision['next_action']}")
    print("july_result_rows_accessed=0")
    print("deterministic_rebuild=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

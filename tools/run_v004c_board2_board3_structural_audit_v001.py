from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_board2_board3_structural_audit import (  # noqa: E402
    run_v004c_board2_board3_structural_audit,
)


def main() -> None:
    output_dir, context = run_v004c_board2_board3_structural_audit(ROOT)
    decision = context["decision"]
    print(f"output_dir={output_dir}")
    print(f"matured_rows={context['matured_rows']}")
    print(f"strict_dates={len(context['stage1_parity']) and 17}")
    print(f"BOARD_STRUCTURE_MODE={decision['mode']}")
    print(
        "BOARD2_BOARD3_SEPARATION_EXPERIMENT_WARRANTED="
        f"{'YES' if decision['separation_warranted'] else 'NO'}"
    )
    print(f"JULY_RESULT_ROWS_ACCESSED={context['july_result_rows_accessed']}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_board2_board3_stage1_feature_response import (  # noqa: E402
    run_v004c_board2_board3_stage1_feature_response,
)


def main() -> None:
    output_dir, context = run_v004c_board2_board3_stage1_feature_response(ROOT)
    decision = context["decision"]
    population = context["population"]
    print(f"output_dir={output_dir}")
    print(f"rows={len(population)}")
    print(f"dates={population['signal_date'].nunique()}")
    print(f"formal_passes={decision['formal_pass_count']}")
    print(f"mechanism={decision['mechanism']}")
    print(f"direction={decision['direction']}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")


if __name__ == "__main__":
    main()

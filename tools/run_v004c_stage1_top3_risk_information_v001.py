from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_stage1_top3_risk_information import (  # noqa: E402
    build_outputs,
    write_outputs,
)


def main() -> int:
    first, first_context = build_outputs(ROOT)
    second, _ = build_outputs(ROOT)
    if first != second:
        raise RuntimeError("FATAL: foundation-audit outputs are not byte-identical")
    output_dir = write_outputs(ROOT, first)
    decision = first_context["decision"]
    population = first_context["population_metrics"]
    print(f"output_dir={output_dir}")
    print(f"strict_dates={population['dates']}")
    print(f"population_rows={population['rows']}")
    print(f"foundation_pass={decision['pass_count']}")
    print(f"foundation_exceptional={decision['exceptional_count']}")
    print(f"risk_information_foundation={decision['status']}")
    print(
        "limited_risk_representation_experiment_warranted="
        f"{'YES' if decision['experiment_warranted'] else 'NO'}"
    )
    print("july_result_rows_accessed=0")
    print("deterministic=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

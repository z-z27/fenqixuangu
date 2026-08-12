from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_original_v4a_direct_transfer import (  # noqa: E402
    run_v004c_original_v4a_direct_transfer,
)


def main() -> None:
    output_dir, context = run_v004c_original_v4a_direct_transfer(ROOT)
    print(f"output_dir={output_dir}")
    print(f"ORIGINAL_V4A_PROVENANCE={context['provenance']}")
    print(f"leaking_folds={context['leaking_folds']}/{context['folds']}")
    print(f"strict_v4c_coverage={context['coverage']:.12f}")
    print(f"JULY_RESULT_ROWS_ACCESSED={context['july_result_rows_accessed']}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")


if __name__ == "__main__":
    main()


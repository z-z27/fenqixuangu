from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_board3_confirmed_harmful_ablation import (  # noqa: E402
    run_v004c_board3_confirmed_harmful_ablation,
)


def main() -> None:
    output_dir, context = run_v004c_board3_confirmed_harmful_ablation(ROOT)
    print(f"output_dir={output_dir}")
    print(f"prediction_lock_sha256={context['prediction_lock_sha256']}")
    print(f"signal={context['decision']['signal']}")
    print(f"direction={context['decision']['direction']}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")


if __name__ == "__main__":
    main()

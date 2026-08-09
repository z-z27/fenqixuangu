from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_pair_capped7_july_forward import (  # noqa: E402
    run_v004c_pair_capped7_july_forward,
)


def main() -> int:
    output_dir, context = run_v004c_pair_capped7_july_forward(ROOT)
    practical = context["practical"]
    combined = practical[practical["scope"].eq("COMBINED")].set_index("model")
    print(f"output_dir={output_dir}")
    print(f"prediction_rows={context['july_rows']}")
    print(f"prediction_dates={context['july_dates']}")
    print(f"prediction_lock_sha256={context['prediction_lock_sha256']}")
    print(f"control_top3={combined.loc['CONTROL_V4A', 'Top3_mean']:.12g}")
    print(f"pair_capped7_top3={combined.loc['PAIR_CAPPED7', 'Top3_mean']:.12g}")
    print(f"forward_stress_signal={context['decision']['signal']}")
    print(
        "primary_failure_attribution="
        f"{context['decision']['primary_failure_attribution']}"
    )
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_overextension_curve import (  # noqa: E402
    CONTROL_MODEL_ID,
    CURVE_MODEL_ID,
    run_v004c_overextension_curve,
)


def main() -> int:
    output_dir, context = run_v004c_overextension_curve(ROOT)
    control = context["summaries"][CONTROL_MODEL_ID]
    curve = context["summaries"][CURVE_MODEL_ID]
    print(f"output_dir={output_dir}")
    print(f"control_parity={'PASS' if context['control_parity']['pass'] else 'FAIL'}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    print(f"control_rank3={control['Rank3_mean']:.12g}")
    print(f"curve_rank3={curve['Rank3_mean']:.12g}")
    print(f"control_top3={control['Top3_mean']:.12g}")
    print(f"curve_top3={curve['Top3_mean']:.12g}")
    print(
        "overextension_curvature_signal="
        f"{context['status']['overextension_curvature_signal']}"
    )
    print(
        "non_monotonic_model_hypothesis="
        f"{context['status']['non_monotonic_model_hypothesis']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

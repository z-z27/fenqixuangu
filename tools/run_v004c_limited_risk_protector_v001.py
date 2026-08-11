from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_limited_risk_protector import (  # noqa: E402
    MODEL_ID,
    POLICY_STAGE1,
    build_outputs,
    run_v004c_limited_risk_protector,
)


def main() -> int:
    first, _ = build_outputs(ROOT)
    second, _ = build_outputs(ROOT)
    if first != second:
        raise RuntimeError("FATAL: limited risk protector outputs are not byte-identical")
    output_dir, context = run_v004c_limited_risk_protector(ROOT)
    practical = context["practical"]

    def top3(policy: str) -> float:
        row = practical[
            practical["scope"].eq("COMBINED") & practical["policy"].eq(policy)
        ].iloc[0]
        return float(row["Top3_mean"])

    print(f"output_dir={output_dir}")
    print(f"strict_dates={len(context['strict_dates'])}")
    print(f"strict_stage1_top3={top3(POLICY_STAGE1):.12f}")
    print(f"limited_risk_top3={top3(MODEL_ID):.12f}")
    print(f"risk_identification_signal={context['decision']['risk_identification_signal']}")
    print(f"risk_protector_signal={context['decision']['risk_protector_signal']}")
    print(
        "july_confirmation_candidate="
        f"{'YES' if context['decision']['july_confirmation_candidate'] else 'NO'}"
    )
    print("deterministic=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

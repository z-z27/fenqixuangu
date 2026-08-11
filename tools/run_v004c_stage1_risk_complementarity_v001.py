from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_stage1_risk_complementarity import (  # noqa: E402
    build_outputs,
    run_v004c_stage1_risk_complementarity,
)


def main() -> int:
    first, context = build_outputs(ROOT)
    second, _ = build_outputs(ROOT)
    deterministic = first == second
    if not deterministic:
        raise RuntimeError("FATAL: strict June architecture outputs are not byte-identical")
    output_dir, context = run_v004c_stage1_risk_complementarity(ROOT)
    practical = context["practical"]

    def top3(policy: str) -> float:
        row = practical[
            practical["scope"].eq("COMBINED") & practical["policy"].eq(policy)
        ].iloc[0]
        return float(row["Top3_mean"])

    print(f"output_dir={output_dir}")
    print(f"strict_dates={context['strict_dates']}")
    print(f"unavailable_dates={'|'.join(context['unavailable_dates'])}")
    print(f"strict_stage1_top3={top3('STRICT_STAGE1'):.12f}")
    print(f"strict_capped_top3={top3('STRICT_CAPPED'):.12f}")
    print(
        "oracle_loss_stage1_backfill_top3="
        f"{top3('STAGE1_WITH_ORACLE_LOSS_VETO_STAGE1_BACKFILL'):.12f}"
    )
    print(
        "oracle_risk_oracle_backfill_top3="
        f"{top3('STAGE1_WITH_ORACLE_RISK_AND_ORACLE_BACKFILL'):.12f}"
    )
    print(f"signal={context['decision']['signal']}")
    print(f"deterministic={'PASS' if deterministic else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

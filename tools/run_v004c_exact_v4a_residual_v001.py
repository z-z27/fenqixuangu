from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_exact_v4a_residual import (  # noqa: E402
    GROUP_FP,
    GROUP_MW,
    run_v004c_exact_v4a_residual,
)


def main() -> int:
    output_dir, context = run_v004c_exact_v4a_residual(ROOT)
    print(f"output_dir={output_dir}")
    print(f"exact_oof_parity={'PASS' if context['parity']['pass'] else 'FAIL'}")
    print(f"join_matched={context['join_audit']['matched_rows']}")
    print(f"fp_rows={context['groups'][GROUP_FP]['rows']}")
    print(f"mw_rows={context['groups'][GROUP_MW]['rows']}")
    print(
        "repair_room_residual_evidence="
        f"{context['evidence']['repair_room_residual_evidence']}"
    )
    print(
        "top10_rerank_feasibility="
        f"{context['top10']['top10_rerank_feasibility']}"
    )
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

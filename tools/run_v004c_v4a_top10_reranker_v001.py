from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_v4a_top10_reranker import (  # noqa: E402
    run_v004c_v4a_top10_reranker,
)


def main() -> int:
    output_dir, context = run_v004c_v4a_top10_reranker(ROOT)
    control = context["summaries"]["CONTROL"]
    reranker = context["summaries"]["RERANKER"]
    print(f"output_dir={output_dir}")
    print(f"stage1_control_parity={'PASS' if context['parity']['pass'] else 'FAIL'}")
    print(f"self_label_leakage_rows={context['self_label_leakage_rows']}")
    print(f"current_test_date_leakage_rows={context['current_test_date_leakage_rows']}")
    print(f"control_top3={control['Top3_mean']:.12g}")
    print(f"reranker_top3={reranker['Top3_mean']:.12g}")
    print(f"reranker_signal={context['status']['reranker_signal']}")
    print(f"forward_stress_candidate={context['status']['forward_stress_candidate']}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

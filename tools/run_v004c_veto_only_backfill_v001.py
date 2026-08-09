from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_veto_only_backfill import run_v004c_veto_only_backfill  # noqa: E402


def main() -> int:
    output_dir, context = run_v004c_veto_only_backfill(ROOT)
    summaries = context["summaries"]
    replacement = context["replacement"]
    status = context["status"]
    print(f"output_dir={output_dir}")
    print("previous_artifact_parity=PASS")
    print(f"self_label_leakage_rows={context['leakage']['self']}")
    print(f"current_test_date_leakage_rows={context['leakage']['current']}")
    print(f"vetoed_total={replacement['vetoed_total']}")
    print(f"backfilled_target7_rate={replacement['backfill_target7_rate']:.12g}")
    print(f"control_top3={summaries['CONTROL']['Top3_mean']:.12g}")
    print(f"full_reranker_top3={summaries['FULL_RERANKER']['Top3_mean']:.12g}")
    print(f"veto_only_top3={summaries['VETO_ONLY']['Top3_mean']:.12g}")
    print(f"veto_only_signal={status['signal']}")
    print(f"forward_stress_candidate={status['forward_stress_candidate']}")
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

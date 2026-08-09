from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_top10_target_information import (  # noqa: E402
    run_v004c_top10_target_information,
)


def main() -> int:
    output_dir, context = run_v004c_top10_target_information(ROOT)
    practical = context["practical"]
    combined = practical[practical["scope"].eq("COMBINED")].set_index("model")
    print(f"output_dir={output_dir}")
    print(f"stage1_parity={'PASS' if context['parity']['pass'] else 'FAIL'}")
    print(f"self_label_leakage_rows={context['self_label_leakage_rows']}")
    print(f"current_test_date_leakage_rows={context['current_test_date_leakage_rows']}")
    for model in ("PAIR_BINARY7", "PAIR_CAPPED7", "PAIR_RAW"):
        print(f"{model.lower()}_top3={combined.loc[model, 'Top3_mean']:.12g}")
    print(f"target_information_signal={context['decision']['target_information_signal']}")
    print(f"above7_information_signal={context['decision']['above7_information_signal']}")
    print(
        "target_objective_trading_candidate="
        f"{context['decision']['target_objective_trading_candidate']}"
    )
    print(f"deterministic_rebuild={context['deterministic_rebuild']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

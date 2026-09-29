"""Run the fixed v004c broad-market context information audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_broad_market_context_information_audit import analyze, output_sha256s, write_outputs


def main() -> None:
    root = ROOT
    context = analyze(root)
    output_dir = write_outputs(context)
    payload = {
        **context["audit"],
        **context["august_audit"],
        **context["august_summary"],
        "broad_market_context_information_state": context["state"],
        "primary_context_variable": "MARKET_UP_RATIO",
        "august_role": "POST_HOC_AUXILIARY_ONLY",
        "model_trained": "NO",
        "new_context_variable_added": "NO",
        "next_action": context["next_action"],
        "output_dir": str(output_dir),
        "sha256": output_sha256s(output_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

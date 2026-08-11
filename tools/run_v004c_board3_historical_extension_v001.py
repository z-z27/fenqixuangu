"""Run the frozen v004c Board3 Jan-Jun historical-extension audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_board3_historical_extension import (  # noqa: E402
    run_v004c_board3_historical_extension,
)


DEFAULT_OUTPUT = (
    ROOT / "reports/research"
    / "v004c_board3_historical_extension_temporal_compatibility_v001_20260101_20260630"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    context = run_v004c_board3_historical_extension(ROOT, output)
    headline = {
        "data_status": context["data_status"],
        "historical_board3_rows": context["historical_board3_rows"],
        "historical_board3_dates": context["historical_board3_dates"],
        "extended_oof_rows": context["expanded_risk"]["rows"],
        "extended_oof_dates": context["expanded_risk"]["dates"],
        "expanded_signal": context["expanded_signal"],
        "history_effect": context["history_effect"],
        "temporal_compatibility": context["temporal_compatibility"],
        "conclusion": context["conclusion"],
        "next_action": context["next_action"],
        "extended_prediction_lock_sha256": context["extended_lock_sha256"],
        "july_result_rows_accessed": context["july_result_rows_accessed"],
    }
    print(json.dumps(headline, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

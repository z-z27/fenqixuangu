from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_blind_5m_seal_path_information_audit import (  # noqa: E402
    analyze,
    output_sha256s,
    write_outputs,
)


def main() -> int:
    context = analyze(ROOT)
    output_dir = write_outputs(context)
    hashes = output_sha256s(output_dir)
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "output_dir": str(output_dir),
        "rows": context["audit"]["rows"],
        "dates": context["audit"]["dates"],
        "model_trained": context["audit"]["model_trained"],
        "family_state": context["family_state"],
        "source_shift_impact": context["source_shift_impact"],
        "rescue_found": context["rescue_found"],
        "rank2_6_incremental_information": context["head_found"],
        "stage1_research_state": context["state_parts"][1],
        "next_action": context["state_parts"][2],
        "aggregate_sha256": aggregate,
        "file_sha256": hashes,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

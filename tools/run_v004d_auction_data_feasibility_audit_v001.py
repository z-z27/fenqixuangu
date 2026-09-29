from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004d_auction_data_feasibility_audit import run


def main() -> None:
    output_dir, context, hashes = run(ROOT)
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(hashes[name].encode("ascii"))
    formal = context["formal"]
    print(f"output_dir={output_dir}")
    print(f"candidate_rows={context['candidate']['all_candidate_rows']}")
    print(f"primary_candidate_d2_rows={context['history']['primary_rows']}")
    print(f"historical_state={formal['historical_state']}")
    print(f"live_state={formal['live_state']}")
    print(f"semantics_state={formal['semantics_state']}")
    print(f"feasibility_state={formal['feasibility_state']}")
    print(f"next_action={formal['next_action']}")
    print(f"aggregate_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()


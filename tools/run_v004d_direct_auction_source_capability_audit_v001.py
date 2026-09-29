from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004d_direct_auction_source_capability_audit import run


def main() -> None:
    output_dir, context, hashes = run(ROOT)
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(hashes[name].encode("ascii"))
    print(f"output_dir={output_dir}")
    print(f"sina_state={context['sina_state']}")
    print(f"eastmoney_state={context['eastmoney_state']}")
    print(f"akshare_state={context['akshare_state']}")
    print(f"historical_backfill={context['historical_backfill']}")
    print(f"live_collection={context['live_collection']}")
    print(f"overall_state={context['overall_state']}")
    print(f"next_action={context['next_action']}")
    print(f"aggregate_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()


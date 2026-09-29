from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_d1_coarse_retrieval_capacity_audit import run


def main() -> None:
    output_dir, context, hashes = run(ROOT)
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(hashes[name].encode("ascii"))
    decision = context["decision"]
    print(f"output_dir={output_dir}")
    print(f"rows={context['audit']['rows']}")
    print(f"dates={context['audit']['dates']}")
    print(f"state={decision['state']}")
    print(f"recommended_k={decision['recommended_k'] or 'NONE'}")
    print(f"temporal_state={decision['temporal_state']}")
    print(f"v004c_new_role={decision['v004c_new_role']}")
    print(f"next_action={decision['next_action']}")
    print(f"aggregate_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()


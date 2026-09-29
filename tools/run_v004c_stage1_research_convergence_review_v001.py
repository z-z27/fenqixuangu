from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_stage1_research_convergence_review import build_outputs


def main() -> int:
    paths = build_outputs(ROOT)
    for name, path in paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

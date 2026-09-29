"""Run the blind v004c conservative 5-minute seal-path coverage audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_5m_conservative_seal_path_coverage_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_audit(args.root, args.output_dir)
    print(f"output_dir={result['output_dir']}")
    print(f"5M_SEAL_PATH_DATA_STATE={result['data_state']}")
    print(f"F1_READY={result['ready']['F1']}")
    print(f"F2_READY={result['ready']['F2']}")
    print(f"F3_READY={result['ready']['F3']}")
    print(f"OUTCOME_ACCESSED=NO")


if __name__ == "__main__":
    main()


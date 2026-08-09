# -*- coding: utf-8 -*-
"""Run the v004c mechanism-information foundation capability screen.

The run is cache-only, PRE-MODEL, and deterministic.  It executes the complete
pipeline twice in memory and writes reports only when all ten outputs are
byte-identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_mechanism_foundation import build_pipeline_outputs  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "reports/research/v004c_mechanism_foundation_screen_v001_20260506_20260630"
)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    first, first_summary = build_pipeline_outputs(ROOT)
    second, second_summary = build_pipeline_outputs(ROOT)
    if set(first) != set(second):
        raise RuntimeError("FATAL: deterministic rebuild output set differs")
    mismatches = [name for name in sorted(first) if first[name] != second[name]]
    if mismatches:
        raise RuntimeError(
            "FATAL: deterministic rebuild byte mismatch: " + ",".join(mismatches)
        )
    if first_summary["questions"] != second_summary["questions"]:
        raise RuntimeError("FATAL: deterministic summary mismatch")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in sorted(second):
        _atomic_write(output / name, second[name])

    questions = second_summary["questions"]
    print(f"DATA_COLLECTION_GATE={'PASS' if second_summary['gate']['pass'] else 'FAIL'}")
    print(f"STRONGEST={questions['strongest_key']}")
    print(f"FOUNDATIONAL_SIGNAL={questions['foundational_signal']}")
    print("DETERMINISTIC_REBUILD=PASS")
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

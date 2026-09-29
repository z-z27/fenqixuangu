from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_frozen_stage1_contribution_ablation_audit import (
    analyze,
    write_outputs,
)


# These are researcher interpretations of the four fixed outputs.  The audit
# module deliberately contains no automatic best-model selector.
RESEARCH_CONCLUSION = "MIXED"
NEXT_ACTION = "INSUFFICIENT_EVIDENCE"
ANSWER_NOTES = {
    "Q1": (
        "YES. A0 reconstructs the accepted frozen snapshot within the fixed "
        "numeric tolerance and reproduces every deterministic daily rank."
    ),
    "Q2": (
        "NO STABLE VALUE. A1 changed no Top3 membership on any of the 60 "
        "mature dates and left Top3 economic/risk results exactly unchanged; "
        "its date-equal Target7 AUC was also effectively unchanged."
    ),
    "Q3": (
        "NO STABLE VALUE. A2 repaired July ranking and Top3 loss contamination, "
        "but reduced May/June Top3 excess and worsened June loss contamination. "
        "The benefit is therefore July-concentrated rather than cross-month."
    ),
    "Q4": (
        "MIXED POSITIVE SIGNAL. A3 improved date-equal Target7 AUC and Top3 "
        "capped excess in May, June, and mature July, but it added one net LOSS "
        "slot overall and worsened June Top3 LOSS excess/negative-date rate."
    ),
    "Q5": (
        "PARTIAL. A3 is the only challenger with same-direction AUC and return "
        "improvement in all three months, but the simplification sequence is "
        "not monotone (A1 is inert and A2 harms May/June) and risk quality does "
        "not improve consistently."
    ),
    "Q6": (
        "NO. The A3 return/ranking pattern is post-hoc on consumed May--July "
        "data, economically modest, and conflicts with LOSS contamination. "
        "This audit does not provide sufficient authorization to fit a reduced "
        "Stage1 Logistic."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    context = analyze(args.root)
    output_dir = write_outputs(
        context,
        RESEARCH_CONCLUSION,
        NEXT_ACTION,
        ANSWER_NOTES,
        args.output_dir,
    )
    print(f"output_dir={output_dir}")
    print(f"a0_score_max_abs_error={context['parity']['reconstructed_score_max_abs_error']:.12g}")
    print(f"a0_rank_mismatch={context['parity']['reconstructed_rank_mismatch_count']}")
    print(f"july_identity_mismatch={context['parity']['july_locked_population_identity_mismatch_count']}")
    print(f"days_since_d0_mismatch_rows={context['days_audit']['mismatch_rows']}")
    print(f"research_conclusion={RESEARCH_CONCLUSION}")
    print(f"next_action={NEXT_ACTION}")


if __name__ == "__main__":
    main()

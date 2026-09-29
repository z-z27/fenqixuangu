from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_reduced7f_stage1_temporal_validation import (
    BASELINE_MODEL,
    CHALLENGER_MODEL,
    analyze,
    output_sha256s,
    summary_row,
    write_outputs,
)


# Fixed researcher interpretation after the one preregistered challenger.
# There is deliberately no automatic best-model selector or follow-on variant.
MODEL_STATE = "REDUCED7F_TEMPORAL_SIGNAL_SUPPORTED"
NEXT_ACTION = "FREEZE_REDUCED7F_FOR_AUGUST_HOLDOUT"
ANSWERS = {
    "Q1": (
        "YES. The learner receives exactly the ordered preregistered seven rank "
        "features; no total_score, D0 timing, interaction, raw absolute, board, "
        "or newly engineered input is present."
    ),
    "Q2": (
        "YES. Every Reduced7F fixed/expanding fit satisfies "
        "label_available_date < test signal_date and historical signal_date < "
        "test signal_date. Leakage rows are zero. The Current18F comparator is "
        "read from archived strict June OOF and the July frozen lock, with zero "
        "Current18F fits."
    ),
    "Q3": (
        "YES, modestly. On the 17 strict June dates, Reduced7F improves date-equal "
        "Target7 AUC by 1.73pp and Top3 capped excess by 0.25pp; Top3 LOSS excess "
        "improves by 1.96pp. The paired Top3 delta is +0.25pp with P(delta>0) "
        "77.84%. Negative-date rate is unchanged, while the worst day is 0.80pp worse."
    ),
    "Q4": (
        "YES, modestly. On 21 mature July dates, the once-frozen May+June "
        "Reduced7F fit improves date-equal Target7 AUC by 5.24pp and Top3 capped "
        "excess by 0.17pp. LOSS excess, negative-date rate, and worst day are "
        "unchanged; paired P(delta>0) is 72.45%."
    ),
    "Q5": (
        "YES. The 38-date expanding OOF improves date-equal Target7 AUC by "
        "5.71pp and Top3 capped excess by 0.56pp, with 11 positive, 5 negative, "
        "and 22 unchanged dates; the date-bootstrap P(delta>0) is 98.54%. "
        "Reduced7F adjacent coefficient cosine has median about 0.997 and six "
        "of seven features never flip sign; rank_trend_hold_score flips five times."
    ),
    "Q6": (
        "YES, with a tail caveat. Upside ranking and Top3 excess improve in both "
        "fixed chronological folds and expanding OOF. LOSS excess is better or "
        "unchanged in the fixed folds and better overall; negative-date rate is "
        "not worse. The June/overall worst Top3 day is 0.80pp worse, so this does "
        "not establish uniform tail improvement."
    ),
    "Q7": (
        "YES. The preregistered Reduced7F challenger has consistent chronological "
        "direction across June and mature July without systematic LOSS/negative-"
        "date deterioration. Freeze the final 485-row/60-date coefficients for "
        "one later August final holdout; do not inspect August in this task."
    ),
}


def _head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    context = analyze(args.root, _head(args.root))
    output_dir = write_outputs(
        context, MODEL_STATE, NEXT_ACTION, ANSWERS, args.output_dir
    )
    fold_a_18 = summary_row(context, "FOLD_A_MAY_TO_JUNE", BASELINE_MODEL)
    fold_a_7 = summary_row(context, "FOLD_A_MAY_TO_JUNE", CHALLENGER_MODEL)
    fold_b_18 = summary_row(context, "FOLD_B_MAY_JUNE_TO_JULY", BASELINE_MODEL)
    fold_b_7 = summary_row(context, "FOLD_B_MAY_JUNE_TO_JULY", CHALLENGER_MODEL)
    print(f"output_dir={output_dir}")
    print(f"mature_rows={context['final_spec']['training_rows']}")
    print(f"mature_dates={context['final_spec']['training_dates']}")
    print(f"current18f_fit_count={context['leakage']['current18f_fit_count']}")
    print(f"fold_a_auc_delta={fold_a_7.target7_within_date_auc-fold_a_18.target7_within_date_auc:.12g}")
    print(f"fold_b_auc_delta={fold_b_7.target7_within_date_auc-fold_b_18.target7_within_date_auc:.12g}")
    print(f"model_state={MODEL_STATE}")
    print(f"next_action={NEXT_ACTION}")
    for name, digest in output_sha256s(output_dir).items():
        print(f"sha256 {name} {digest}")


if __name__ == "__main__":
    main()

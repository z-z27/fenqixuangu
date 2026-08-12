"""Original-v4a direct-transfer provenance and strict temporal-contract audit.

The archived original-v4a score artifact is authoritative when it is usable.
This module therefore audits it *before* any V4C outcome is attached.  The
canonical archived walk-forward split is signal-date chronological, but its
training rows are not label-maturity chronological.  A strict point-in-time
benchmark cannot silently replace that artifact with a different refit and
still call the result original v4a.  Consequently the production path in this
module fails closed with ``ORIGINAL_V4A_PROVENANCE = BLOCKED`` and writes only
the score/provenance/coverage locks needed to review that conclusion.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .v004a import (
    DEFAULT_INITIAL_TRAIN_DAYS,
    MODEL_ID_V004A,
    SCOPE_WALK_FORWARD,
)
from .v004c_board2_board3_structural_audit import load_matured_population
from .v004c_v4a_architecture_transfer import load_authoritative_input


TASK_NAME = "ORIGINAL V4A DIRECT TRANSFER + TOP2 VS TOP3 BENCHMARK"
EXPECTED_STARTING_HEAD = "9fc73b5e2fbbc2ad8cd773f7ffec8b40c380a165"
OUTPUT_RELATIVE_DIR = Path(
    "reports/research/"
    "v004c_original_v4a_direct_transfer_top2_top3_v001_20260506_20260626"
)

MODEL_ID = "logistic_v004a_weighted"
L2 = 0.30
POSITIVE_WEIGHT = 1.50
INITIAL_TRAIN_DAYS = 18
FEATURE_COUNT = 18
# Operational audit note: an initial provenance probe loaded the canonical
# sample table's return columns before a D3-date sentinel discovered that the
# tail contains July-matured rows.  The values were never used or reported,
# but the task contract defines loading as access.  The exact count was not
# subsequently queried; this lower bound makes the experiment fail closed.
JULY_RESULT_ROWS_ACCESSED_MINIMUM = 1

NEW_MODEL = False
NEW_FEATURE = False
HYPERPARAMETER_SEARCH = False
STAGE2 = False
BOARD3_REPAIR = False

ORIGINAL_SCORE_PATH = Path(
    "reports/v004a/grid_v2_scored/v004a_scored_candidates.csv"
)
ORIGINAL_COEFFICIENT_PATH = Path(
    "reports/v004a/grid_v2_scored/v004a_coefficients.csv"
)
ORIGINAL_REPORT_PATH = Path("reports/v004a/grid_v2_scored/v004a_report.md")
ORIGINAL_INPUT_PATH = Path(
    "reports/history_samples/2026-05-06_2026-06-29/"
    "history_candidates_2026-05-06_2026-06-29_dedup.csv"
)
CURRENT_STRICT_PATH = Path(
    "reports/research/v004c_stage1_risk_complementarity_v001_20260601_20260630/"
    "v004c_stage1_risk_strict_oof_v001.csv"
)

EXPECTED_STRICT_DATES = (
    "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08",
    "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
    "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18",
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
    "2026-06-26",
)

FEATURE_COLUMNS = (
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_total_score",
    "rank_theme_score",
    "rank_days_since_d0",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
    "inter_close_low",
    "inter_close_trend",
    "inter_total_trend",
    "inter_total_active",
    "inter_low_active",
    "spread_close_low",
    "days_since_d0_le1",
    "days_since_d0_eq2",
    "days_since_d0_ge3",
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def rank_scores_after_scoring(
    frame: pd.DataFrame,
    score_column: str,
    event_column: str = "event_id",
) -> pd.Series:
    """Rank already-scored rows without recomputing any input feature."""
    result = pd.Series(index=frame.index, dtype="int64")
    for _, day in frame.groupby("signal_date", sort=True):
        ordered = day.sort_values(
            [score_column, event_column],
            ascending=[False, True],
            kind="mergesort",
        )
        result.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return result.astype(int)


def top2_top3_values(rank_returns: Iterable[float]) -> tuple[float, float, float]:
    """Pure helper for the frozen Top2/Top3 marginal-return definition."""
    values = np.asarray(list(rank_returns), dtype=float)
    if len(values) < 3:
        raise ValueError("Top2/Top3 marginal value requires three ranks")
    top2 = float(values[:2].mean())
    top3 = float(values[:3].mean())
    return top2, top3, float(top3 - top2)


def _read_selected_scores(root: Path) -> pd.DataFrame:
    path = root / ORIGINAL_SCORE_PATH
    required = [
        "model_id", "evaluation_scope", "l2", "positive_weight",
        "feature_set", "interaction_set", "signal_date", "code",
        "model_score", "model_probability", "model_rank",
    ]
    frame = pd.read_csv(path, usecols=required, dtype={"code": str})
    frame = frame[
        frame["model_id"].eq(MODEL_ID)
        & frame["evaluation_scope"].eq(SCOPE_WALK_FORWARD)
        & pd.to_numeric(frame["l2"]).eq(L2)
        & pd.to_numeric(frame["positive_weight"]).eq(POSITIVE_WEIGHT)
    ].copy()
    frame["signal_date"] = frame["signal_date"].astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["original_v4a_score"] = pd.to_numeric(frame["model_score"])
    frame["original_v4a_rank_full_universe"] = pd.to_numeric(
        frame["model_rank"]
    ).astype(int)
    if frame.duplicated(["signal_date", "code"]).any():
        raise RuntimeError("FATAL: duplicate archived original-v4a score key")
    feature_sets = frame["feature_set"].dropna().astype(str).unique().tolist()
    if feature_sets != [",".join(FEATURE_COLUMNS)]:
        raise RuntimeError("FATAL: archived original-v4a feature contract changed")
    return frame.sort_values(
        ["signal_date", "original_v4a_rank_full_universe", "code"],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_fold_contract(root: Path) -> pd.DataFrame:
    path = root / ORIGINAL_COEFFICIENT_PATH
    frame = pd.read_csv(
        path,
        usecols=[
            "model_id", "evaluation_scope", "l2", "positive_weight",
            "feature_set", "fold_index", "train_start", "train_end",
            "predict_date", "term", "coefficient",
        ],
    )
    frame = frame[
        frame["model_id"].eq(MODEL_ID)
        & frame["evaluation_scope"].eq(SCOPE_WALK_FORWARD)
        & pd.to_numeric(frame["l2"]).eq(L2)
        & pd.to_numeric(frame["positive_weight"]).eq(POSITIVE_WEIGHT)
    ].copy()
    if frame.empty:
        raise RuntimeError("FATAL: frozen original-v4a coefficient rows unavailable")
    if frame["feature_set"].dropna().astype(str).unique().tolist() != [
        ",".join(FEATURE_COLUMNS)
    ]:
        raise RuntimeError("FATAL: coefficient feature contract mismatch")
    folds = frame.groupby("predict_date", sort=True).agg(
        fold_index=("fold_index", "first"),
        train_start=("train_start", "first"),
        train_end=("train_end", "first"),
        coefficient_terms=("term", "nunique"),
    ).reset_index()
    folds["predict_date"] = folds["predict_date"].astype(str)
    # The direct-transfer comparison ends on June 26.  The archived June 29
    # fold would require July D3 label metadata and is outside this task.
    folds = folds[folds["predict_date"].le("2026-06-26")].copy()
    if not folds["coefficient_terms"].eq(FEATURE_COUNT + 1).all():
        raise RuntimeError("FATAL: original-v4a coefficient term count mismatch")
    return folds


def _read_canonical_samples(root: Path, max_signal_date: str) -> pd.DataFrame:
    """Read the exact input fields needed by canonical eligibility + maturity audit."""
    path = root / ORIGINAL_INPUT_PATH
    usecols = [
        "signal_date", "code", "eligible_for_trade", "candidate_base_price",
        "d2_trade_date", "d3_trade_date",
    ]
    raw = pd.read_csv(path, usecols=usecols, dtype={"code": str})
    raw["signal_date"] = raw["signal_date"].astype(str)
    raw = raw[raw["signal_date"].le(str(max_signal_date))].copy()
    eligible = raw["eligible_for_trade"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    base_positive = pd.to_numeric(
        raw["candidate_base_price"], errors="coerce"
    ).gt(0)
    codes = raw["code"].astype(str).str.zfill(6)
    duplicates = pd.DataFrame({
        "signal_date": raw["signal_date"], "code": codes,
    }).duplicated(["signal_date", "code"], keep=False)
    result = raw[eligible & base_positive & ~duplicates].copy()
    result["signal_date"] = result["signal_date"].astype(str)
    result["d3_trade_date"] = result["d3_trade_date"].astype(str)
    result["code"] = result["code"].astype(str).str.zfill(6)
    # Historical report data quality establishes that all 4,124 canonical
    # eligible rows have both return columns.  For the pre-June-26 partition,
    # eligibility/base/key checks are therefore sufficient and deliberately
    # avoid loading any outcome value.
    if result["d3_trade_date"].ge("2026-07-01").any():
        raise RuntimeError("FATAL: July outcome metadata entered provenance audit")
    return result


def audit_temporal_contract(samples: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds.itertuples(index=False):
        train = samples[
            samples["signal_date"].ge(str(fold.train_start))
            & samples["signal_date"].le(str(fold.train_end))
        ].copy()
        immature = train[train["d3_trade_date"].ge(str(fold.predict_date))]
        rows.append({
            "predict_date": str(fold.predict_date),
            "fold_index": int(fold.fold_index),
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "train_rows": int(len(train)),
            "train_dates": int(train["signal_date"].nunique()),
            "latest_training_signal_date": str(train["signal_date"].max()),
            "latest_training_label_available_date": str(train["d3_trade_date"].max()),
            "current_or_future_label_rows": int(len(immature)),
            "current_or_future_label_signal_dates": int(
                immature["signal_date"].nunique()
            ),
            "strict_point_in_time_pass": bool(immature.empty),
        })
    return pd.DataFrame(rows)


def _full_score_lock(scores: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    counts = scores.groupby("signal_date")["code"].transform("size").astype(int)
    fold_meta = folds[[
        "predict_date", "train_start", "train_end", "fold_index",
    ]].rename(columns={"predict_date": "signal_date"})
    result = scores[[
        "signal_date", "code", "original_v4a_score",
        "original_v4a_rank_full_universe",
    ]].copy()
    result["original_universe_candidate_count"] = counts
    result = result.merge(fold_meta, on="signal_date", validate="many_to_one")
    return result.rename(columns={
        "train_start": "earliest_train_date",
        "train_end": "latest_train_date",
        "fold_index": "train_dates",
    }).sort_values(
        ["signal_date", "original_v4a_rank_full_universe", "code"],
        kind="mergesort",
    ).reset_index(drop=True)


def _coverage_lock(root: Path, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    strict = pd.read_csv(
        root / CURRENT_STRICT_PATH,
        usecols=[
            "event_id", "signal_date", "code", "candidate_count",
            "stage1_score", "stage1_rank",
        ],
        dtype={"event_id": str, "code": str},
    )
    strict["signal_date"] = strict["signal_date"].astype(str)
    strict["code"] = strict["code"].astype(str).str.zfill(6)
    if tuple(sorted(strict["signal_date"].unique())) != EXPECTED_STRICT_DATES:
        raise RuntimeError("FATAL: current strict V4C date contract changed")
    if len(strict) != 155:
        raise RuntimeError("FATAL: current strict V4C row contract changed")
    selected = scores[[
        "signal_date", "code", "original_v4a_score",
        "original_v4a_rank_full_universe",
    ]]
    joined = strict.merge(
        selected, on=["signal_date", "code"], how="left", validate="one_to_one"
    )
    joined["original_v4a_scorable"] = joined["original_v4a_score"].notna()
    joined["missing_reason"] = np.where(
        joined["original_v4a_scorable"], "", "NOT_IN_ARCHIVED_ORIGINAL_V4A_SCORE"
    )
    covered = joined[joined["original_v4a_scorable"]].copy()
    if not covered.empty:
        covered["original_v4a_rank_within_v4c_intersection"] = rank_scores_after_scoring(
            covered, "original_v4a_score"
        )
        joined = joined.merge(
            covered[["event_id", "original_v4a_rank_within_v4c_intersection"]],
            on="event_id", how="left", validate="one_to_one",
        )
    else:
        joined["original_v4a_rank_within_v4c_intersection"] = pd.NA
    audit = {
        "strict_dates": int(strict["signal_date"].nunique()),
        "strict_rows": int(len(strict)),
        "covered_dates": int(covered["signal_date"].nunique()),
        "covered_rows": int(len(covered)),
        "coverage": float(len(covered) / len(strict)),
        "missing_rows": int(len(strict) - len(covered)),
        "zero_coverage_dates": sorted(
            set(strict["signal_date"]) - set(covered["signal_date"])
        ),
    }
    return joined.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True), audit


def _provenance_rows(
    root: Path,
    scores: pd.DataFrame,
    temporal: pd.DataFrame,
    coverage: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for artifact_type, relative in (
        ("ORIGINAL_INPUT", ORIGINAL_INPUT_PATH),
        ("ARCHIVED_SCORE", ORIGINAL_SCORE_PATH),
        ("ARCHIVED_COEFFICIENTS", ORIGINAL_COEFFICIENT_PATH),
        ("HISTORICAL_REPORT", ORIGINAL_REPORT_PATH),
        ("CURRENT_STRICT_COMPARATOR", CURRENT_STRICT_PATH),
    ):
        path = root / relative
        rows.append({
            "artifact_type": artifact_type,
            "path": relative.as_posix(),
            "exists": path.is_file(),
            "model_id": MODEL_ID,
            "historical_config": "ARCHIVED_FROZEN_GRID_SELECTION",
            "feature_count": FEATURE_COUNT,
            "initial_train_days": INITIAL_TRAIN_DAYS,
            "l2": L2,
            "positive_weight": POSITIVE_WEIGHT,
            "input_sample_source": ORIGINAL_INPUT_PATH.as_posix(),
            "score_source": "ARCHIVED",
            "parity_status": "FAIL_TEMPORAL_CONTRACT",
            "notes": (
                f"score_rows={len(scores)}; score_dates={scores.signal_date.nunique()}; "
                f"folds_with_immature_labels={int((~temporal.strict_point_in_time_pass).sum())}; "
                f"strict_v4c_coverage={coverage['coverage']:.12f}"
            ),
        })
    return pd.DataFrame(rows)


def _review(context: dict[str, Any]) -> str:
    c = context
    lines = [
        "# v004c Original-v4a Direct-Transfer & Top2-vs-Top3 Benchmark v001",
        "",
        "## 1. Experimental Contract",
        "",
        "- Original-v4a must score its complete original universe before V4C intersection.",
        "- Archived scores have priority; no replacement refit is allowed without archived parity.",
        "- Strict point-in-time chronology requires every training label to be available before the test date.",
        "- No model, feature, hyperparameter, Stage2, or Board3 repair was used.",
        "",
        "## 2. Provenance Result",
        "",
        f"- ORIGINAL_V4A_PROVENANCE: **{c['provenance']}**",
        f"- Archived score rows/dates: **{c['score_rows']} / {c['score_dates']}**",
        f"- Frozen configuration: **{MODEL_ID}; L2={L2}; positive_weight={POSITIVE_WEIGHT}; features={FEATURE_COUNT}; initial_train_days={INITIAL_TRAIN_DAYS}**",
        f"- Full-universe score-lock SHA256: **{c['score_lock_sha256']}**",
        "",
        "## 3. Strict Temporal Audit",
        "",
        f"- Audited folds: **{c['folds']}**",
        f"- Folds with current/future D3 labels in training: **{c['leaking_folds']}**",
        f"- Current/future-label rows per fold: **{c['min_leak_rows']}..{c['max_leak_rows']}**",
        f"- Maximum training label-available date: **{c['latest_training_label_available_date']}**",
        "- The archived walk-forward split is chronological by signal date, not by label availability.",
        "- A strict reconstruction that removes those rows would not be score/rank-parity with archived original v4a and would therefore be a different model instance.",
        "",
        "## 4. V4C Coverage (QA Only; No Outcomes Attached)",
        "",
        f"- Strict V4C rows/dates: **{c['strict_rows']} / {c['strict_dates']}**",
        f"- Archived original-v4a covered rows/dates: **{c['covered_rows']} / {c['covered_dates']}**",
        f"- Coverage: **{c['coverage']:.4%}**",
        f"- Missing rows: **{c['missing_rows']}**",
        f"- Zero-coverage dates: **{', '.join(c['zero_coverage_dates']) or 'none'}**",
        "",
        "## 5. Stop Decision",
        "",
        "- Direct-transfer practical benchmark: **NOT RUN**",
        "- Top2-vs-Top3 outcome benchmark: **NOT RUN**",
        "- Bootstrap / LODO / membership attribution: **NOT RUN**",
        "- ORIGINAL_V4A_DIRECT_TRANSFER_SIGNAL: **INVALID**",
        "- FINAL_TRADING_CAPACITY_SIGNAL: **INVALID**",
        "- NEXT_V4C_ACTION: **INVALID**",
        "- JULY_RESULT_ROWS_ACCESSED: **>=1 (exact count not queried; provenance probe breach)**",
        "",
        "The benchmark is blocked because no archived original-v4a score artifact satisfies the required strict point-in-time label-maturity contract. Re-fitting canonical features with a new maturity filter would be a new reconstructed instance and cannot be presented as archived original-v4a direct transfer without a separate, explicitly authorized experiment.",
        "",
        "JULY_RESULT_ROWS_ACCESSED >= 1; JULY_USED_FOR_MODEL_OR_OUTCOME_CONCLUSION = NO",
    ]
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if any((NEW_MODEL, NEW_FEATURE, HYPERPARAMETER_SEARCH, STAGE2, BOARD3_REPAIR)):
        raise RuntimeError("FATAL: experimental scope changed")
    if MODEL_ID != MODEL_ID_V004A or INITIAL_TRAIN_DAYS != DEFAULT_INITIAL_TRAIN_DAYS:
        raise RuntimeError("FATAL: original-v4a frozen constants changed")
    if len(FEATURE_COLUMNS) != FEATURE_COUNT:
        raise RuntimeError("FATAL: original-v4a feature count changed")

    base = load_authoritative_input(root)
    matured, raw_audit = load_matured_population(root)
    if raw_audit != {"rows": 319, "dates": 39, "board2": 261, "board3": 58}:
        raise RuntimeError("FATAL: V4C authoritative parity failed")
    if len(matured) != 307 or matured["signal_date"].nunique() != 37:
        raise RuntimeError("FATAL: V4C matured parity failed")

    scores = _read_selected_scores(root)
    folds = _read_fold_contract(root)
    samples = _read_canonical_samples(root, str(folds["train_end"].max()))
    temporal = audit_temporal_contract(samples, folds)
    full_lock = _full_score_lock(scores, folds)
    coverage_lock, coverage = _coverage_lock(root, scores)

    leaking_folds = int((~temporal["strict_point_in_time_pass"]).sum())
    provenance = "BLOCKED" if leaking_folds else "ARCHIVED_SCORE_ARTIFACT"
    # This module is intentionally fail-closed.  If a future clean archived
    # artifact is supplied, the practical benchmark must be implemented and
    # reviewed explicitly rather than falling through an unaudited path here.
    if provenance != "BLOCKED":
        raise RuntimeError("FATAL: clean original-v4a artifact needs reviewed benchmark path")

    full_lock_bytes = _csv_bytes(full_lock)
    context = {
        "provenance": provenance,
        "score_rows": int(len(scores)),
        "score_dates": int(scores["signal_date"].nunique()),
        "folds": int(len(temporal)),
        "leaking_folds": leaking_folds,
        "min_leak_rows": int(temporal["current_or_future_label_rows"].min()),
        "max_leak_rows": int(temporal["current_or_future_label_rows"].max()),
        "latest_training_label_available_date": str(
            temporal["latest_training_label_available_date"].max()
        ),
        "score_lock_sha256": _sha(full_lock_bytes),
        "v4c_rows": int(len(base)),
        "v4c_dates": int(base["signal_date"].nunique()),
        "matured_rows": int(len(matured)),
        "matured_dates": int(matured["signal_date"].nunique()),
        **coverage,
        "july_result_rows_accessed": ">=1 (exact count not queried)",
        "july_used_for_model_or_outcome_conclusion": False,
        "direct_transfer_signal": "INVALID",
        "current_top2_vs_top3": "INVALID",
        "original_top2_vs_top3": "INVALID",
        "final_trading_capacity_signal": "INVALID",
        "next_v4c_action": "INVALID",
        "blocker": (
            "Archived original-v4a folds contain training labels whose D3 "
            "availability date is not earlier than the fold test date."
        ),
    }
    provenance_frame = _provenance_rows(root, scores, temporal, coverage)
    outputs = {
        "v004c_original_v4a_provenance_v001.csv": _csv_bytes(provenance_frame),
        "v004c_original_v4a_temporal_audit_v001.csv": _csv_bytes(temporal),
        "v004c_original_v4a_full_score_lock_v001.csv": full_lock_bytes,
        "v004c_original_v4a_v4c_intersection_v001.csv": _csv_bytes(coverage_lock),
        "v004c_original_v4a_direct_transfer_review_v001.md": _review(context).encode("utf-8"),
    }
    return outputs, context


def run_v004c_original_v4a_direct_transfer(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs_a, context_a = build_outputs(root_path)
    outputs_b, context_b = build_outputs(root_path)
    if outputs_a != outputs_b or context_a != context_b:
        raise RuntimeError("FATAL: deterministic rebuild failed")
    output_dir = root_path / OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs_a.items():
        (output_dir / name).write_bytes(data)
    context_a["deterministic_rebuild"] = "PASS"
    return output_dir, context_a

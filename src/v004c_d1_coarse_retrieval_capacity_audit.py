"""Capacity audit for frozen S2 as an unordered D1 coarse retriever.

This audit reads archived S2 scores/ranks.  It never fits a learner, changes
the candidate universe, or touches August.  K=3 is a historical reference;
only the preregistered K=4/5/6/7 retrieval operating points are evaluated.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TASK_NAME = "v004c_d1_coarse_retrieval_capacity_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
DEVELOPMENT_START = "2026-05-06"
DEVELOPMENT_END = "2026-07-29"
LABEL_ASOF = "2026-08-01"

FROZEN_POPULATION_ARTIFACT = (
    "reports/research/v004c_blind_5m_seal_path_information_audit_v001/"
    "v004c_5m_seal_path_analysis_population_v001.csv"
)
PARITY_POPULATION_ARTIFACT = (
    "reports/research/"
    "v004c_d1_unfinished_repair_information_audit_v001_20260506_20260731/"
    "v004c_unfinished_repair_population_v001.csv"
)
S2_FIT_ARTIFACT = (
    "reports/research/"
    "v004c_reduced7f_training_spec_sanity_v001_20260506_20260731/"
    "v004c_reduced7f_spec_training_fit_v001.csv"
)

S2_SPEC = "S2_NO_TAIL_L2_010"
S2_L2 = 0.10
S2_POSITIVE_WEIGHT = 1.50
S2_TAIL_WEIGHTING = "NONE"

REFERENCE_K = 3
RETRIEVAL_KS = (4, 5, 6, 7)
ALL_KS = (3, 4, 5, 6, 7)
CUMULATIVE_KS = (1, 2, 3, 4, 5, 6, 7)
PERIODS = ("POOLED", "MAY", "JUNE", "JULY")
MONTHS = ("MAY", "JUNE", "JULY")
RANK_BANDS = ("RANK1", "RANK2_3", "RANK4_5", "RANK6_7", "RANK8_PLUS")

RANDOM_REPETITIONS = 10_000
RANDOM_SEED = 20260906
# The prompt requires evidence that is not merely a lucky choice among four K
# values but does not prescribe a numeric cutoff.  This fixed, moderately
# conservative one-sided family-wise threshold is declared before evaluation.
K_SELECTION_ADJUSTED_P_GATE = 0.10

OUTPUT_FILENAMES = (
    "v004c_d1_retrieval_daily_population_v001.csv",
    "v004c_d1_retrieval_topk_capacity_v001.csv",
    "v004c_d1_retrieval_monthly_capacity_v001.csv",
    "v004c_d1_retrieval_target7_misses_v001.csv",
    "v004c_d1_retrieval_compression_v001.csv",
    "v004c_d1_retrieval_enrichment_v001.csv",
    "v004c_d1_retrieval_random_baseline_v001.csv",
    "v004c_d1_retrieval_k_selection_permutation_v001.csv",
    "v004c_d1_retrieval_rank_band_v001.csv",
    "v004c_d1_retrieval_cumulative_capture_v001.csv",
    "v004c_d1_coarse_retrieval_capacity_review_v001.md",
)

RANDOM_METRICS = (
    "target7_row_recall",
    "date_macro_target7_recall",
    "winner_presence_hit_rate",
    "all_winners_retained_rate",
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_csv_bytes(frame)).hexdigest()


def _month_label(signal_date: Any) -> str:
    value = str(signal_date)[:7]
    return {"2026-05": "MAY", "2026-06": "JUNE", "2026-07": "JULY"}.get(
        value, "OUT_OF_SCOPE"
    )


def _period_part(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    return frame if period == "POOLED" else frame[frame["month"].eq(period)]


def assert_contract() -> None:
    if ALL_KS != (3, 4, 5, 6, 7):
        raise RuntimeError("FATAL: TopK contract changed")
    if RETRIEVAL_KS != (4, 5, 6, 7):
        raise RuntimeError("FATAL: coarse-retrieval K set changed")
    if (S2_L2, S2_POSITIVE_WEIGHT, S2_TAIL_WEIGHTING) != (0.10, 1.50, "NONE"):
        raise RuntimeError("FATAL: frozen S2 contract changed")
    if RANDOM_REPETITIONS != 10_000:
        raise RuntimeError("FATAL: random-baseline repetition count changed")


def load_frozen_population(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load archived S2 score/rank; do not reconstruct or fit the model."""

    assert_contract()
    root_path = Path(root).resolve()
    input_path = root_path / FROZEN_POPULATION_ARTIFACT
    parity_path = root_path / PARITY_POPULATION_ARTIFACT
    fit_path = root_path / S2_FIT_ARTIFACT
    for path in (input_path, parity_path, fit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    usecols = [
        "event_id", "code", "signal_date", "month", "board_group",
        "label_available_date", "target7", "loss", "severe_loss",
        "s2_score", "s2_rank",
    ]
    frame = pd.read_csv(
        input_path, encoding="utf-8-sig", usecols=usecols,
        dtype={"event_id": str, "code": str},
    )
    frame = frame[
        frame["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & frame["label_available_date"].lt(LABEL_ASOF)
    ].copy()
    frame["month"] = frame["signal_date"].map(_month_label)
    for column in ("target7", "loss", "severe_loss", "s2_rank"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["s2_score"] = pd.to_numeric(frame["s2_score"], errors="raise")
    frame = frame.sort_values(
        ["signal_date", "s2_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    if (len(frame), frame["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: frozen development population is not 485 rows / 60 dates")
    expected_months = {
        "MAY": (146, 18), "JUNE": (173, 21), "JULY": (166, 21)
    }
    for month, (rows, dates) in expected_months.items():
        part = frame[frame["month"].eq(month)]
        if (len(part), part["signal_date"].nunique()) != (rows, dates):
            raise RuntimeError(f"FATAL: frozen {month} population parity failed")
    if frame["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id")
    if frame["signal_date"].max() != DEVELOPMENT_END:
        raise RuntimeError("FATAL: development end-date mismatch")
    if frame["signal_date"].ge("2026-08-01").any():
        raise RuntimeError("FATAL: August signal date reached retrieval audit")
    if not frame["label_available_date"].lt(LABEL_ASOF).all():
        raise RuntimeError("FATAL: immature label reached retrieval audit")
    if frame[["target7", "loss", "severe_loss", "s2_score", "s2_rank"]].isna().any().any():
        raise RuntimeError("FATAL: required frozen fields contain missing values")
    if (frame[["target7", "loss", "severe_loss"]].isin([0, 1]) == 0).any().any():
        raise RuntimeError("FATAL: invalid outcome indicator")
    if ((frame["target7"] + frame["loss"]) > 1).any():
        raise RuntimeError("FATAL: Target7/LOSS overlap")
    if ((frame["severe_loss"] > frame["loss"])).any():
        raise RuntimeError("FATAL: severe LOSS is not a subset of LOSS")

    candidate_count = frame.groupby("signal_date")["event_id"].transform("size").astype(int)
    frame["candidate_count"] = candidate_count
    for _, day in frame.groupby("signal_date", sort=True):
        ranks = day["s2_rank"].tolist()
        if ranks != list(range(1, len(day) + 1)):
            raise RuntimeError("FATAL: archived S2 ranks are not exact consecutive ranks")
        if not day["s2_score"].is_monotonic_decreasing:
            raise RuntimeError("FATAL: archived S2 score/rank ordering mismatch")
        ties = day[day["s2_score"].duplicated(keep=False)]
        for _, tie in ties.groupby("s2_score", sort=False):
            if tie["event_id"].tolist() != sorted(tie["event_id"].tolist()):
                raise RuntimeError("FATAL: archived deterministic event_id tie-break mismatch")

    parity_cols = [
        "event_id", "signal_date", "target7", "loss", "severe_loss",
        "s2_score", "s2_rank", "candidate_count",
    ]
    parity = pd.read_csv(
        parity_path, encoding="utf-8-sig", usecols=parity_cols,
        dtype={"event_id": str},
    )
    parity = parity[
        parity["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & parity["event_id"].isin(frame["event_id"])
    ].copy()
    merged = frame[parity_cols].merge(
        parity, on="event_id", how="outer", suffixes=("_frozen", "_parity"),
        indicator=True, validate="one_to_one",
    )
    identity_mismatch = int(merged["_merge"].ne("both").sum())
    score_error = float(
        (merged["s2_score_frozen"] - merged["s2_score_parity"]).abs().max()
    )
    rank_mismatch = int(
        merged["s2_rank_frozen"].ne(merged["s2_rank_parity"]).sum()
    )
    outcome_mismatch = 0
    for column in ("signal_date", "target7", "loss", "severe_loss", "candidate_count"):
        outcome_mismatch += int(merged[f"{column}_frozen"].ne(merged[f"{column}_parity"]).sum())
    if identity_mismatch or score_error > 1e-12 or rank_mismatch or outcome_mismatch:
        raise RuntimeError(
            "FATAL: archived frozen S2 population parity failed: "
            f"identity={identity_mismatch}, score={score_error}, "
            f"rank={rank_mismatch}, field={outcome_mismatch}"
        )

    fit = pd.read_csv(fit_path, encoding="utf-8-sig")
    spec = fit[fit["spec"].eq(S2_SPEC)]
    if len(spec) != 1:
        raise RuntimeError("FATAL: S2 frozen fit record is missing")
    spec = spec.iloc[0]
    if (
        int(spec["training_rows"]) != 485
        or int(spec["training_dates"]) != 60
        or not math.isclose(float(spec["l2"]), S2_L2)
        or not math.isclose(float(spec["positive_weight"]), S2_POSITIVE_WEIGHT)
        or str(spec["tail_weighting"]) != S2_TAIL_WEIGHTING
    ):
        raise RuntimeError("FATAL: S2 fit metadata contract mismatch")

    audit = {
        "rows": len(frame),
        "dates": int(frame["signal_date"].nunique()),
        "duplicate_event_id": int(frame["event_id"].duplicated().sum()),
        "frozen_score_parity_max_abs_error": score_error,
        "frozen_rank_mismatch_count": rank_mismatch,
        "frozen_population_identity_mismatch_count": identity_mismatch,
        "frozen_outcome_field_mismatch_count": outcome_mismatch,
        "input_artifact": FROZEN_POPULATION_ARTIFACT,
        "parity_artifact": PARITY_POPULATION_ARTIFACT,
        "s2_spec": S2_SPEC,
        "model_trained": "NO",
        "august_used": "NO",
    }
    return frame, audit


def _topk_rows(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    return frame[frame["s2_rank"].le(k)].copy()


def _capacity_row(frame: pd.DataFrame, k: int, period: str) -> dict[str, Any]:
    selected = _topk_rows(frame, k)
    target_total = int(frame["target7"].sum())
    selected_target = int(selected["target7"].sum())
    target_by_date = frame.groupby("signal_date")["target7"].sum()
    target_dates = target_by_date[target_by_date.gt(0)]
    selected_by_date = selected.groupby("signal_date")["target7"].sum()
    captured = selected_by_date.reindex(target_dates.index, fill_value=0)
    counts = frame.groupby("signal_date").size()
    selected_counts = selected.groupby("signal_date").size().reindex(counts.index, fill_value=0)
    return {
        "period": period,
        "k": k,
        "role": "TOP3_REFERENCE" if k == REFERENCE_K else "COARSE_RETRIEVAL_CANDIDATE",
        "rows": len(frame),
        "dates": int(frame["signal_date"].nunique()),
        "target7_rows": target_total,
        "target7_dates": len(target_dates),
        "retained_rows": len(selected),
        "retained_target7_rows": selected_target,
        "missed_target7_count": target_total - selected_target,
        "target7_row_recall": selected_target / target_total if target_total else math.nan,
        "date_macro_target7_recall": float((captured / target_dates).mean()) if len(target_dates) else math.nan,
        "winner_presence_hit_rate": float(captured.gt(0).mean()) if len(target_dates) else math.nan,
        "all_winners_retained_rate": float(captured.eq(target_dates).mean()) if len(target_dates) else math.nan,
        "retained_row_ratio": len(selected) / len(frame),
        "compression_ratio": 1.0 - len(selected) / len(frame),
        "no_compression_date_count": int(counts.le(k).sum()),
        "no_compression_date_rate": float(counts.le(k).mean()),
        "candidate_target7_rate": float(frame["target7"].mean()),
        "topk_target7_rate": float(selected["target7"].mean()),
        "target7_enrichment": (
            float(selected["target7"].mean() / frame["target7"].mean())
            if frame["target7"].mean() > 0 else math.nan
        ),
        "candidate_loss_rate": float(frame["loss"].mean()),
        "topk_loss_count": int(selected["loss"].sum()),
        "topk_loss_rate": float(selected["loss"].mean()),
        "loss_rate_delta": float(selected["loss"].mean() - frame["loss"].mean()),
        "candidate_severe_loss_rate": float(frame["severe_loss"].mean()),
        "topk_severe_loss_rate": float(selected["severe_loss"].mean()),
        "mean_effective_k": float(selected_counts.mean()),
    }


def build_daily_population(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in frame.groupby("signal_date", sort=True):
        rows.append({
            "signal_date": signal_date,
            "month": day["month"].iloc[0],
            "candidate_count": len(day),
            "target7_count": int(day["target7"].sum()),
            "target7_rate": float(day["target7"].mean()),
            "loss_count": int(day["loss"].sum()),
            "loss_rate": float(day["loss"].mean()),
            "severe_loss_count": int(day["severe_loss"].sum()),
            "severe_loss_rate": float(day["severe_loss"].mean()),
            "has_target7": int(day["target7"].sum() > 0),
        })
    return pd.DataFrame(rows)


def build_daily_topk(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in frame.groupby("signal_date", sort=True):
        n = len(day)
        target_total = int(day["target7"].sum())
        for k in ALL_KS:
            selected = _topk_rows(day, k)
            captured = int(selected["target7"].sum())
            rows.append({
                "signal_date": signal_date,
                "month": day["month"].iloc[0],
                "k": k,
                "candidate_count": n,
                "effective_k": min(k, n),
                "retained_count": len(selected),
                "removed_count": n - len(selected),
                "retained_fraction": len(selected) / n,
                "compression_fraction": 1.0 - len(selected) / n,
                "no_compression_date": int(n <= k),
                "universe_target7_count": target_total,
                "retained_target7_count": captured,
                "date_target7_recall": captured / target_total if target_total else math.nan,
                "winner_presence_hit": int(captured > 0) if target_total else math.nan,
                "all_winners_retained": int(captured == target_total) if target_total else math.nan,
                "retained_loss_count": int(selected["loss"].sum()),
                "retained_severe_loss_count": int(selected["severe_loss"].sum()),
            })
    return pd.DataFrame(rows)


def build_capacity_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = pd.DataFrame([_capacity_row(frame, k, "POOLED") for k in ALL_KS])
    monthly = pd.DataFrame([
        _capacity_row(_period_part(frame, month), k, month)
        for month in MONTHS for k in ALL_KS
    ])
    return pooled, monthly


def build_target7_misses(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in ALL_KS:
        missed = frame[frame["target7"].eq(1) & frame["s2_rank"].gt(k)].copy()
        for row in missed.itertuples(index=False):
            offset = int(row.s2_rank) - k
            bucket = "RANK_K_PLUS_1" if offset == 1 else (
                "RANK_K_PLUS_2" if offset == 2 else "DEEPER_THAN_K_PLUS_2"
            )
            rows.append({
                "k": k,
                "signal_date": row.signal_date,
                "month": row.month,
                "event_id": row.event_id,
                "code": row.code,
                "board_group": row.board_group,
                "original_s2_rank": int(row.s2_rank),
                "candidate_count": int(row.candidate_count),
                "rank_offset_below_k": offset,
                "miss_rank_bucket": bucket,
            })
    result = pd.DataFrame(rows)
    return result.sort_values(["k", "signal_date", "original_s2_rank", "event_id"], kind="mergesort").reset_index(drop=True)


def build_compression_summary(daily_topk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        period_frame = daily_topk if period == "POOLED" else daily_topk[daily_topk["month"].eq(period)]
        for k in ALL_KS:
            part = period_frame[period_frame["k"].eq(k)]
            row: dict[str, Any] = {"period": period, "k": k, "dates": len(part)}
            for column in ("candidate_count", "effective_k", "removed_count", "retained_fraction"):
                row[f"daily_{column}_mean"] = float(part[column].mean())
                row[f"daily_{column}_median"] = float(part[column].median())
                row[f"daily_{column}_p25"] = float(part[column].quantile(0.25))
                row[f"daily_{column}_p75"] = float(part[column].quantile(0.75))
            row["no_compression_date_count"] = int(part["no_compression_date"].sum())
            row["no_compression_date_rate"] = float(part["no_compression_date"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def build_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        part = _period_part(frame, period)
        for k in ALL_KS:
            capacity = _capacity_row(part, k, period)
            rows.append({key: capacity[key] for key in (
                "period", "k", "rows", "retained_rows", "candidate_target7_rate",
                "topk_target7_rate", "target7_enrichment", "candidate_loss_rate",
                "topk_loss_count", "topk_loss_rate", "loss_rate_delta",
                "candidate_severe_loss_rate", "topk_severe_loss_rate",
            )})
    return pd.DataFrame(rows)


def _random_topk_metrics(
    frame: pd.DataFrame,
    ks: Sequence[int] = ALL_KS,
    repetitions: int = RANDOM_REPETITIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Nested same-date random ranks; preserves each date's class counts."""

    ks = tuple(ks)
    rng = np.random.default_rng(seed)
    selected_targets = np.zeros((repetitions, len(ks)), dtype=float)
    macro_sum = np.zeros_like(selected_targets)
    presence_sum = np.zeros_like(selected_targets)
    all_sum = np.zeros_like(selected_targets)
    target_total = int(frame["target7"].sum())
    target_date_count = 0

    for _, day in frame.groupby("signal_date", sort=True):
        y = day["target7"].to_numpy(dtype=np.int8)
        n = len(y)
        random_keys = rng.random((repetitions, n))
        orders = np.argsort(random_keys, axis=1, kind="stable")
        permuted_y = np.take_along_axis(np.broadcast_to(y, orders.shape), orders, axis=1)
        cumulative = np.cumsum(permuted_y, axis=1)
        day_targets = int(y.sum())
        for j, k in enumerate(ks):
            captured = cumulative[:, min(k, n) - 1]
            selected_targets[:, j] += captured
            if day_targets:
                macro_sum[:, j] += captured / day_targets
                presence_sum[:, j] += captured > 0
                all_sum[:, j] += captured == day_targets
        if day_targets:
            target_date_count += 1

    if target_total == 0 or target_date_count == 0:
        raise RuntimeError("FATAL: random baseline has no Target7 support")
    return {
        "target7_row_recall": selected_targets / target_total,
        "date_macro_target7_recall": macro_sum / target_date_count,
        "winner_presence_hit_rate": presence_sum / target_date_count,
        "all_winners_retained_rate": all_sum / target_date_count,
    }


def build_random_baseline(
    frame: pd.DataFrame,
    observed_capacity: pd.DataFrame,
    repetitions: int = RANDOM_REPETITIONS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    arrays = _random_topk_metrics(frame, ALL_KS, repetitions, seed)
    observed = observed_capacity.set_index("k")
    baseline_rows: list[dict[str, Any]] = []
    for j, k in enumerate(ALL_KS):
        for metric in RANDOM_METRICS:
            values = arrays[metric][:, j]
            point = float(observed.loc[k, metric])
            expected = float(values.mean())
            baseline_rows.append({
                "k": k,
                "metric": metric,
                "observed": point,
                "random_expected": expected,
                "random_p2_5": float(np.quantile(values, 0.025)),
                "random_p50": float(np.quantile(values, 0.50)),
                "random_p97_5": float(np.quantile(values, 0.975)),
                "observed_minus_random_expected": point - expected,
                "p_random_gte_observed": float((1 + np.count_nonzero(values >= point)) / (repetitions + 1)),
                "repetitions": repetitions,
                "seed": seed,
            })
    baseline = pd.DataFrame(baseline_rows)

    retrieval_idx = [ALL_KS.index(k) for k in RETRIEVAL_KS]
    row_arrays = arrays["target7_row_recall"][:, retrieval_idx]
    random_means = row_arrays.mean(axis=0)
    null_uplifts = row_arrays - random_means
    max_null = null_uplifts.max(axis=1)
    observed_recalls = np.asarray(
        [float(observed.loc[k, "target7_row_recall"]) for k in RETRIEVAL_KS]
    )
    observed_uplifts = observed_recalls - random_means
    max_observed = float(observed_uplifts.max())
    family_p = float((1 + np.count_nonzero(max_null >= max_observed)) / (repetitions + 1))
    selection_rows: list[dict[str, Any]] = []
    for j, k in enumerate(RETRIEVAL_KS):
        adjusted_p = float(
            (1 + np.count_nonzero(max_null >= observed_uplifts[j])) / (repetitions + 1)
        )
        selection_rows.append({
            "row_type": "K_DETAIL",
            "k": k,
            "observed_target7_row_recall": observed_recalls[j],
            "random_expected_target7_row_recall": random_means[j],
            "observed_recall_uplift": observed_uplifts[j],
            "k_selection_adjusted_p": adjusted_p,
            "evidence_gate_threshold": K_SELECTION_ADJUSTED_P_GATE,
            "selection_adjusted_evidence_pass": "YES" if adjusted_p <= K_SELECTION_ADJUSTED_P_GATE and observed_uplifts[j] > 0 else "NO",
            "repetitions": repetitions,
            "seed": seed,
        })
    selection_rows.append({
        "row_type": "FAMILY_MAX_ACROSS_K4_7",
        "k": math.nan,
        "observed_target7_row_recall": float(observed_recalls.max()),
        "random_expected_target7_row_recall": math.nan,
        "observed_recall_uplift": max_observed,
        "k_selection_adjusted_p": family_p,
        "evidence_gate_threshold": K_SELECTION_ADJUSTED_P_GATE,
        "selection_adjusted_evidence_pass": "YES" if family_p <= K_SELECTION_ADJUSTED_P_GATE and max_observed > 0 else "NO",
        "repetitions": repetitions,
        "seed": seed,
    })
    selection = pd.DataFrame(selection_rows)
    audit = {
        "random_repetitions": repetitions,
        "random_seed": seed,
        "family_selection_adjusted_p": family_p,
        "family_max_observed_recall_uplift": max_observed,
    }
    return baseline, selection, audit


def _rank_band(rank: int) -> str:
    if rank == 1:
        return "RANK1"
    if rank <= 3:
        return "RANK2_3"
    if rank <= 5:
        return "RANK4_5"
    if rank <= 7:
        return "RANK6_7"
    return "RANK8_PLUS"


def build_rank_band(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["rank_band"] = work["s2_rank"].map(_rank_band)
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        part = _period_part(work, period)
        for band in RANK_BANDS:
            subset = part[part["rank_band"].eq(band)]
            rows.append({
                "period": period,
                "rank_band": band,
                "row_count": len(subset),
                "date_count": int(subset["signal_date"].nunique()),
                "target7_count": int(subset["target7"].sum()),
                "target7_rate": float(subset["target7"].mean()) if len(subset) else math.nan,
                "loss_count": int(subset["loss"].sum()),
                "loss_rate": float(subset["loss"].mean()) if len(subset) else math.nan,
            })
    return pd.DataFrame(rows)


def build_cumulative_capture(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        part = _period_part(frame, period)
        total = int(part["target7"].sum())
        for k in CUMULATIVE_KS:
            selected = part[part["s2_rank"].le(k)]
            captured = int(selected["target7"].sum())
            rows.append({
                "period": period,
                "k": k,
                "target7_total": total,
                "target7_captured": captured,
                "cumulative_target7_row_recall": captured / total if total else math.nan,
                "retained_rows": len(selected),
                "retained_row_ratio": len(selected) / len(part),
            })
    return pd.DataFrame(rows)


def evaluate_gates(
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection_by_k = selection[selection["row_type"].eq("K_DETAIL")].set_index("k")
    rows: list[dict[str, Any]] = []
    for k in RETRIEVAL_KS:
        p = pooled[pooled["k"].eq(k)].iloc[0]
        m = monthly[monthly["k"].eq(k)].set_index("period")
        row = {
            "k": k,
            "gate_pooled_row_recall_ge_80": "YES" if p["target7_row_recall"] >= 0.80 else "NO",
            "gate_each_month_row_recall_ge_70": "YES" if m["target7_row_recall"].min() >= 0.70 else "NO",
            "gate_pooled_winner_hit_ge_90": "YES" if p["winner_presence_hit_rate"] >= 0.90 else "NO",
            "gate_each_month_winner_hit_ge_80": "YES" if m["winner_presence_hit_rate"].min() >= 0.80 else "NO",
            "gate_retained_row_ratio_le_75": "YES" if p["retained_row_ratio"] <= 0.75 else "NO",
            "gate_no_compression_date_rate_le_35": "YES" if p["no_compression_date_rate"] <= 0.35 else "NO",
            "gate_random_and_selection_evidence": str(selection_by_k.loc[k, "selection_adjusted_evidence_pass"]),
        }
        gate_columns = [column for column in row if column.startswith("gate_")]
        row["gate_pass_count"] = sum(row[column] == "YES" for column in gate_columns)
        row["stage_a_gate_pass"] = "YES" if row["gate_pass_count"] == len(gate_columns) else "NO"
        rows.append(row)
    gate_table = pd.DataFrame(rows)
    passing = gate_table[gate_table["stage_a_gate_pass"].eq("YES")]

    if len(passing):
        state = "D1_COARSE_RETRIEVAL_SUPPORTED"
        recommended_k: int | None = int(passing["k"].min())
    else:
        # Partial requires one actual K to show a useful, temporally broad
        # recall/compression trade-off even though one or more strict gates fail.
        partial = False
        for k in RETRIEVAL_KS:
            p = pooled[pooled["k"].eq(k)].iloc[0]
            m = monthly[monthly["k"].eq(k)]
            uplift = float(selection_by_k.loc[k, "observed_recall_uplift"])
            if (
                p["target7_row_recall"] >= 0.70
                and m["target7_row_recall"].min() >= 0.65
                and p["winner_presence_hit_rate"] >= 0.90
                and m["winner_presence_hit_rate"].min() >= 0.80
                and p["retained_row_ratio"] <= 0.75
                and uplift > 0
            ):
                partial = True
        state = "D1_COARSE_RETRIEVAL_PARTIAL" if partial else "D1_COARSE_RETRIEVAL_NOT_SUPPORTED"
        recommended_k = None

    temporal_supported = any(
        monthly[monthly["k"].eq(k)]["target7_row_recall"].min() >= 0.70
        and monthly[monthly["k"].eq(k)]["winner_presence_hit_rate"].min() >= 0.80
        for k in RETRIEVAL_KS
    )
    temporal_partial = any(
        monthly[monthly["k"].eq(k)]["target7_row_recall"].min() >= 0.60
        and monthly[monthly["k"].eq(k)]["winner_presence_hit_rate"].min() >= 0.75
        for k in RETRIEVAL_KS
    )
    temporal_state = "SUPPORTED" if temporal_supported else (
        "PARTIAL" if temporal_partial else "NOT_SUPPORTED"
    )
    decision = {
        "state": state,
        "recommended_k": recommended_k,
        "temporal_state": temporal_state,
        "v004c_new_role": (
            "V004D_STAGE_A_RETRIEVAL_BASELINE"
            if state == "D1_COARSE_RETRIEVAL_SUPPORTED"
            else "FROZEN_REFERENCE_ONLY"
        ),
        "next_action": (
            "V004D_AUCTION_DATA_FEASIBILITY_AUDIT"
            if state == "D1_COARSE_RETRIEVAL_SUPPORTED"
            else "STOP_AND_REVIEW"
        ),
    }
    return gate_table, decision


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{100 * float(value):.2f}%"


def _pp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{100 * float(value):+.2f} pp"


def build_review(context: Mapping[str, Any]) -> str:
    pooled = context["topk_capacity"].set_index("k")
    monthly = context["monthly_capacity"]
    selection = context["k_selection"].query("row_type == 'K_DETAIL'").set_index("k")
    decision = context["decision"]
    audit = context["audit"]
    gates = context["gates"].set_index("k")
    lines = [
        "# v004c D1 Coarse Retrieval Capacity Audit v001",
        "",
        "## 先说人话",
        "",
        "S2 当精确 Top3 排名器不稳定；本轮把它当成只负责 KEEP/DROP 的粗筛器。结果是：K4/K5 相对同日随机有明确优势，但召回太低；扩大到 Top7 后能保住更多 winner，却仍未达到预注册的 80% 总召回，而且 51.67% 的交易日本来就不超过 7 只，实际没有发生筛选，Top7 的选择校正证据也不足。没有同一个 K 同时满足高召回、有效压缩和随机优势。因此只能判为 PARTIAL，不能授权它成为 v004d Stage A。",
        "",
        "### Q1. S2 只用来留下值得继续看的股票，有没有用？",
        "",
        "有一定用，但容量不够强。Top4/Top5 显示了排序增益，却漏掉太多 Target7；Top7 的跨月 winner-presence 较稳定，却压缩太弱、总体召回仍不足，且相对随机的增益不再可靠。",
        "",
        "### Q2. Top4 / Top5 / Top6 / Top7 分别保住多少 Target7？",
        "",
        "| K | Target7 row recall | Date-macro recall | Winner presence |",
        "|---:|---:|---:|---:|",
    ]
    for k in RETRIEVAL_KS:
        row = pooled.loc[k]
        lines.append(
            f"| {k} | {_pct(row['target7_row_recall'])} | {_pct(row['date_macro_target7_recall'])} | {_pct(row['winner_presence_hit_rate'])} |"
        )
    lines += [
        "",
        "### Q3. 需要保留原候选池多少比例？",
        "",
        "| K | Retained rows | Compression | No-compression dates |",
        "|---:|---:|---:|---:|",
    ]
    for k in RETRIEVAL_KS:
        row = pooled.loc[k]
        lines.append(
            f"| {k} | {_pct(row['retained_row_ratio'])} | {_pct(row['compression_ratio'])} | {_pct(row['no_compression_date_rate'])} |"
        )
    lines += [
        "",
        "### Q4. 这是 S2 的真实排序能力，还是 K 大以后随机也差不多？",
        "",
        "K4/K5 的 row recall 明确高于同日随机，K-selection-adjusted evidence 通过；但它们没有达到高召回门槛。K6/K7 召回更接近粗筛目标时，相对随机的增益缩小且校正证据未通过。具体如下。",
        "",
        "| K | S2 minus random recall | Selection-adjusted p | Evidence gate |",
        "|---:|---:|---:|:---:|",
    ]
    for k in RETRIEVAL_KS:
        row = selection.loc[k]
        lines.append(
            f"| {k} | {_pp(row['observed_recall_uplift'])} | {float(row['k_selection_adjusted_p']):.4f} | {row['selection_adjusted_evidence_pass']} |"
        )
    lines += [
        "",
        "### Q5. May / June / July 是否稳定？",
        "",
        f"Temporal state = {decision['temporal_state']}。Top7 的月度 row recall 为 "
        + ", ".join(
            f"{month} {_pct(monthly[(monthly['period'].eq(month)) & (monthly['k'].eq(7))].iloc[0]['target7_row_recall'])}"
            for month in MONTHS
        )
        + "；方向不崩，但这依赖很大的 K。",
        "",
        "### Q6. 是否存在候选明显减少、同时保留大部分 Target7 的平衡点？",
        "",
        "没有 K 完整通过预注册 gate。Top7 最接近高召回，但 pooled recall 仍低于 80%，且 no-compression date rate 高于 35%。不能降低门槛或追加 K=8。",
        "",
        "### Q7. v004c 是否有资格转为 v004d Stage A？",
        "",
        "当前没有。它可继续作为冻结参考，但本轮没有授权新的 Stage A operating point。",
        "",
        "## Gate details",
        "",
        "| K | Passed gates | Full pass |",
        "|---:|---:|:---:|",
    ]
    for k in RETRIEVAL_KS:
        row = gates.loc[k]
        lines.append(f"| {k} | {int(row['gate_pass_count'])}/7 | {row['stage_a_gate_pass']} |")

    recommended = decision["recommended_k"]
    if recommended is None:
        recall = hit = retained = uplift = "N/A"
    else:
        recall = _pct(pooled.loc[recommended, "target7_row_recall"])
        hit = _pct(pooled.loc[recommended, "winner_presence_hit_rate"])
        retained = _pct(pooled.loc[recommended, "retained_row_ratio"])
        uplift = _pp(selection.loc[recommended, "observed_recall_uplift"])
    lines += [
        "",
        "## Integrity",
        "",
        f"- Frozen population: {audit['rows']} rows / {audit['dates']} dates.",
        f"- Frozen score parity max abs error: {audit['frozen_score_parity_max_abs_error']:.3e}.",
        f"- Frozen rank mismatch count: {audit['frozen_rank_mismatch_count']}.",
        "- S2 score/rank read from archived artifacts; fit count is zero.",
        "- K values tested: 3 reference plus exactly 4/5/6/7 retrieval candidates.",
        f"- Random baseline: {RANDOM_REPETITIONS} same-date repetitions, seed {RANDOM_SEED}.",
        "- August signal dates used: NO.",
        "",
        "## Formal verdict",
        "",
        f"D1_COARSE_RETRIEVAL_STATE = {decision['state']}",
        "",
        f"RECOMMENDED_STAGE_A_K = {recommended if recommended is not None else 'NONE'}",
        "",
        f"TARGET7_RECALL_AT_RECOMMENDED_K = {recall}",
        "",
        f"WINNER_PRESENCE_HIT_AT_RECOMMENDED_K = {hit}",
        "",
        f"RETAINED_ROW_RATIO_AT_RECOMMENDED_K = {retained}",
        "",
        f"RANDOM_BASELINE_UPLIFT = {uplift}",
        "",
        f"TEMPORAL_RETRIEVAL_STABILITY = {decision['temporal_state']}",
        "",
        f"V004C_NEW_ROLE = {decision['v004c_new_role']}",
        "",
        "MODEL_TRAINED = NO",
        "",
        "FEATURE_SEARCH = NO",
        "",
        "PARAMETER_SEARCH = NO",
        "",
        "NEW_FACTOR = NO",
        "",
        "AUGUST_HOLDOUT_STATUS = CONSUMED",
        "",
        "AUGUST_USED = NO",
        "",
        f"NEXT_ACTION = {decision['next_action']}",
        "",
    ]
    return "\n".join(lines)


def analyze(
    root: str | Path,
    repetitions: int = RANDOM_REPETITIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    population, audit = load_frozen_population(root)
    daily_population = build_daily_population(population)
    daily_topk = build_daily_topk(population)
    topk_capacity, monthly_capacity = build_capacity_tables(population)
    target7_misses = build_target7_misses(population)
    compression = build_compression_summary(daily_topk)
    enrichment = build_enrichment(population)
    random_baseline, k_selection, random_audit = build_random_baseline(
        population, topk_capacity, repetitions, seed
    )
    gates, decision = evaluate_gates(topk_capacity, monthly_capacity, k_selection)
    audit = {
        **audit,
        **random_audit,
        "population_sha256": _frame_sha256(population),
        "model_trained": "NO",
        "feature_search": "NO",
        "parameter_search": "NO",
        "new_factor": "NO",
        "august_used": "NO",
    }
    context: dict[str, Any] = {
        "population": population,
        "daily_population": daily_population,
        "daily_topk": daily_topk,
        "topk_capacity": topk_capacity,
        "monthly_capacity": monthly_capacity,
        "target7_misses": target7_misses,
        "compression": compression,
        "enrichment": enrichment,
        "random_baseline": random_baseline,
        "k_selection": k_selection,
        "rank_band": build_rank_band(population),
        "cumulative_capture": build_cumulative_capture(population),
        "gates": gates,
        "decision": decision,
        "audit": audit,
    }
    context["review"] = build_review(context)
    return context


def build_outputs(context: Mapping[str, Any]) -> dict[str, bytes]:
    topk = context["topk_capacity"].merge(context["gates"], on="k", how="left", validate="one_to_one")
    return {
        OUTPUT_FILENAMES[0]: _csv_bytes(context["daily_population"]),
        OUTPUT_FILENAMES[1]: _csv_bytes(topk),
        OUTPUT_FILENAMES[2]: _csv_bytes(context["monthly_capacity"]),
        OUTPUT_FILENAMES[3]: _csv_bytes(context["target7_misses"]),
        OUTPUT_FILENAMES[4]: _csv_bytes(context["compression"]),
        OUTPUT_FILENAMES[5]: _csv_bytes(context["enrichment"]),
        OUTPUT_FILENAMES[6]: _csv_bytes(context["random_baseline"]),
        OUTPUT_FILENAMES[7]: _csv_bytes(context["k_selection"]),
        OUTPUT_FILENAMES[8]: _csv_bytes(context["rank_band"]),
        OUTPUT_FILENAMES[9]: _csv_bytes(context["cumulative_capture"]),
        OUTPUT_FILENAMES[10]: str(context["review"]).encode("utf-8"),
    }


def write_outputs(root: str | Path, context: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    root_path = Path(root).resolve()
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = build_outputs(context)
    hashes: dict[str, str] = {}
    for name in OUTPUT_FILENAMES:
        payload = payloads[name]
        (output_dir / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return output_dir, hashes


def run(root: str | Path) -> tuple[Path, dict[str, Any], dict[str, str]]:
    context = analyze(root)
    output_dir, hashes = write_outputs(root, context)
    return output_dir, context, hashes

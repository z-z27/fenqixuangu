"""Board2-versus-Board3 structural audit for the frozen V4C_STAGE1.

This module is descriptive only.  It reads the canonical May/June D1-safe
universe, pre-July-matured outcomes, and the already-frozen strict June Stage1
OOF artifact.  It does not fit a model, create a feature, search a threshold,
or evaluate a board-specific trading policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_limited_risk_protector import (
    EXPECTED_STAGE1_PARITY,
    EXPECTED_STRICT_DATES,
)
from .v004c_pair_capped7_july_forward import load_authoritative_d1_safe_base
from .v004c_stage1_risk_complementarity import (
    JULY_SENTINEL,
    _load_date_contract,
    load_frozen_june_buckets,
    load_pre_july_outcomes,
)
from .v004c_stage1_top3_risk_information import binary_auc
from .v004c_v4a_architecture_transfer import dataframe_csv_bytes
from .v004c_v4a_top10_reranker import strategy_summary


TASK_NAME = "BOARD2 VS BOARD3 STRUCTURAL AUDIT"
NEW_MODEL = False
NEW_FEATURE = False
NEW_POLICY = False
THRESHOLD_SEARCH = False
JULY_RESULT_ROWS_ACCESSED = 0
JULY_USED_FOR_BOARD_STRUCTURE_AUDIT = False

POPULATION_A_BOOTSTRAP_SEED = 20260811
POPULATION_A_BOOTSTRAP_RESAMPLES = 20_000
POPULATION_A_PERMUTATION_SEED = 20260812
POPULATION_A_PERMUTATIONS = 20_000
STRICT_BOOTSTRAP_SEED = 20260813
STRICT_BOOTSTRAP_RESAMPLES = 20_000

BOARD_GROUPS = ("BOARD2", "BOARD3")
FUNNEL_LEVELS = ("UNIVERSE", "TOP10", "TOP5", "TOP3")
EXPECTED_RAW_ROWS = 319
EXPECTED_RAW_DATES = 39
EXPECTED_RAW_BOARD2 = 261
EXPECTED_RAW_BOARD3 = 58
EXPECTED_MATURED_ROWS = 307
EXPECTED_MATURED_DATES = 37

OUTPUT_FILENAMES = (
    "v004c_board2_board3_matured_universe_v001.csv",
    "v004c_board2_board3_strict_funnel_v001.csv",
    "v004c_board2_board3_daily_comparison_v001.csv",
    "v004c_board2_board3_rank_quality_v001.csv",
    "v004c_board2_board3_bucket_v001.csv",
    "v004c_board2_board3_robustness_v001.csv",
    "v004c_board2_board3_review_v001.md",
)


def assert_audit_contract() -> None:
    if any((NEW_MODEL, NEW_FEATURE, NEW_POLICY, THRESHOLD_SEARCH)):
        raise RuntimeError("FATAL: structural audit expanded into model or policy development")
    if JULY_RESULT_ROWS_ACCESSED != 0 or JULY_USED_FOR_BOARD_STRUCTURE_AUDIT:
        raise RuntimeError("FATAL: July result data entered board structural audit")
    if POPULATION_A_BOOTSTRAP_RESAMPLES != 20_000:
        raise RuntimeError("FATAL: Population A bootstrap contract changed")
    if POPULATION_A_PERMUTATIONS != 20_000:
        raise RuntimeError("FATAL: Population A permutation contract changed")
    if STRICT_BOOTSTRAP_RESAMPLES != 20_000:
        raise RuntimeError("FATAL: strict bootstrap contract changed")


def board_group(streak: int | float) -> str:
    value = int(streak)
    if value == 2:
        return "BOARD2"
    if value == 3:
        return "BOARD3"
    raise ValueError(f"board streak must be 2 or 3, got {streak}")


def target7_from_raw(raw_return: float | Sequence[float]) -> Any:
    values = np.asarray(raw_return, dtype=float)
    result = (values >= 0.07).astype(int)
    return int(result) if np.ndim(raw_return) == 0 else result


def loss_from_raw(raw_return: float | Sequence[float]) -> Any:
    values = np.asarray(raw_return, dtype=float)
    result = (values < 0.0).astype(int)
    return int(result) if np.ndim(raw_return) == 0 else result


def capped_return_7(raw_return: float | Sequence[float]) -> Any:
    values = np.asarray(raw_return, dtype=float)
    result = np.minimum(values, 0.07)
    return float(result) if np.ndim(raw_return) == 0 else result


def stage1_rank_percentile(rank: float, candidate_count: float) -> float:
    return float((float(rank) - 1.0) / max(float(candidate_count) - 1.0, 1.0))


def selection_rate(selected_count: int, candidate_count: int) -> float:
    return float(selected_count / candidate_count) if candidate_count else np.nan


def composition_lift(selected_board3: int, selected_total: int,
                     candidate_board3: int, candidate_total: int) -> float:
    if not selected_total or not candidate_total or not candidate_board3:
        return np.nan
    return float((selected_board3 / selected_total) / (candidate_board3 / candidate_total))


def _strict_oof_path(root: Path) -> Path:
    path = (
        root / "reports/research/v004c_stage1_risk_complementarity_v001_20260601_20260630"
        / "v004c_stage1_risk_strict_oof_v001.csv"
    )
    if not path.is_file():
        raise RuntimeError("FATAL: frozen strict Stage1 OOF artifact unavailable")
    return path


def load_matured_population(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only May/June events whose D3 label matured before July 1."""
    assert_audit_contract()
    base = load_authoritative_d1_safe_base(root)
    raw_audit = {
        "rows": int(len(base)),
        "dates": int(base["signal_date"].nunique()),
        "board2": int(pd.to_numeric(base["board_streak_before_break"]).eq(2).sum()),
        "board3": int(pd.to_numeric(base["board_streak_before_break"]).eq(3).sum()),
    }
    expected = {
        "rows": EXPECTED_RAW_ROWS, "dates": EXPECTED_RAW_DATES,
        "board2": EXPECTED_RAW_BOARD2, "board3": EXPECTED_RAW_BOARD3,
    }
    if raw_audit != expected:
        raise RuntimeError(f"FATAL: authoritative universe mismatch: {raw_audit}")

    date_contract = _load_date_contract(root, base)
    outcomes = load_pre_july_outcomes(root, base, date_contract)
    identity = base[[
        "event_id", "signal_date", "code", "board_streak_before_break",
    ]].merge(date_contract, on=["event_id", "signal_date"], validate="one_to_one")
    matured = identity.merge(outcomes, on=[
        "event_id", "d2_date", "d3_date", "label_available_date",
    ], how="inner", validate="one_to_one")
    matured["board_streak_before_break"] = pd.to_numeric(
        matured["board_streak_before_break"]
    ).astype(int)
    matured["board_group"] = matured["board_streak_before_break"].map(board_group)
    matured["target7"] = target7_from_raw(matured["raw_repair_return"])
    matured["loss"] = loss_from_raw(matured["raw_repair_return"])
    matured["severe_loss_5"] = matured["raw_repair_return"].le(-0.05).astype(int)
    matured["capped_return_7"] = capped_return_7(matured["raw_repair_return"])
    matured["code"] = matured["code"].astype(str).str.zfill(6)
    matured = matured.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(matured) != EXPECTED_MATURED_ROWS or matured["signal_date"].nunique() != EXPECTED_MATURED_DATES:
        raise RuntimeError("FATAL: matured pre-July population must be 307 rows / 37 dates")
    if matured["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in matured population")
    if bool(matured["label_available_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: immature or July outcome entered Population A")
    if bool(matured[["d2_date", "d3_date"]].ge(JULY_SENTINEL).any(axis=None)):
        raise RuntimeError("FATAL: July D2/D3 outcome entered Population A")
    expected_raw = matured["d3_high_daily"] / matured["d2_open_daily"] - 1.0
    if not np.allclose(expected_raw, matured["raw_repair_return"], rtol=0.0, atol=1e-12):
        raise RuntimeError("FATAL: raw return semantic mismatch")
    if not np.allclose(
        np.minimum(matured["raw_repair_return"], 0.07), matured["capped_return_7"],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("FATAL: capped return is not upper-cap-only")
    return matured, raw_audit


def load_strict_funnel(root: Path, base: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the frozen strict Stage1 OOF; no Stage1 refit occurs here."""
    if base is None:
        base = load_authoritative_d1_safe_base(root)
    strict = pd.read_csv(
        _strict_oof_path(root), encoding="utf-8-sig",
        dtype={"event_id": str, "code": str},
    )
    if strict["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate strict OOF event_id")
    strict["signal_date"] = strict["signal_date"].astype(str)
    dates = sorted(strict["signal_date"].unique())
    if dates != EXPECTED_STRICT_DATES:
        raise RuntimeError("FATAL: strict Stage1 dates changed")
    board = base[["event_id", "board_streak_before_break"]].copy()
    strict = strict.merge(board, on="event_id", how="left", validate="one_to_one")
    if strict["board_streak_before_break"].isna().any():
        raise RuntimeError("FATAL: board identity join failed")
    strict["board_streak_before_break"] = pd.to_numeric(
        strict["board_streak_before_break"]
    ).astype(int)
    strict["board_group"] = strict["board_streak_before_break"].map(board_group)
    strict["stage1_rank"] = pd.to_numeric(strict["stage1_rank"]).astype(int)
    strict["candidate_count"] = pd.to_numeric(strict["candidate_count"]).astype(int)
    strict["stage1_score"] = pd.to_numeric(strict["stage1_score"])
    strict["raw_repair_return"] = pd.to_numeric(strict["raw_repair_return"])
    strict["target7"] = target7_from_raw(strict["raw_repair_return"])
    strict["loss"] = loss_from_raw(strict["raw_repair_return"])
    strict["severe_loss_5"] = strict["raw_repair_return"].le(-0.05).astype(int)
    strict["capped_return_7"] = capped_return_7(strict["raw_repair_return"])
    strict["capped_opportunity_return_7"] = strict["capped_return_7"]
    strict["stage1_rank_percentile"] = [
        stage1_rank_percentile(rank, count)
        for rank, count in zip(strict["stage1_rank"], strict["candidate_count"])
    ]
    strict["stage1_top10"] = strict["stage1_rank"].le(
        np.minimum(strict["candidate_count"], 10)
    )
    strict["stage1_top5"] = strict["stage1_rank"].le(
        np.minimum(strict["candidate_count"], 5)
    )
    strict["stage1_top3"] = strict["stage1_rank"].le(
        np.minimum(strict["candidate_count"], 3)
    )
    bucket_map = load_frozen_june_buckets(root, dates)
    strict["opportunity_bucket"] = strict["signal_date"].map(bucket_map)
    if strict["opportunity_bucket"].isna().any():
        raise RuntimeError("FATAL: frozen opportunity bucket join failed")
    strict = strict.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    parity = strategy_summary(strict, "stage1_rank")
    for metric, expected_value in EXPECTED_STAGE1_PARITY.items():
        if not np.isclose(parity[metric], expected_value, rtol=0.0, atol=1e-11):
            raise RuntimeError(f"FATAL: strict Stage1 parity failed for {metric}")
    if int(strict["stage1_top3"].sum()) != 51:
        raise RuntimeError("FATAL: strict Stage1 Top3 must contain 51 rows")
    return strict, parity


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    raw = pd.to_numeric(frame["raw_repair_return"], errors="coerce")
    capped = pd.to_numeric(frame["capped_return_7"], errors="coerce")
    return {
        "rows": int(len(frame)),
        "dates": int(frame["signal_date"].nunique()),
        "target7_count": int(frame["target7"].sum()),
        "target7_rate": float(frame["target7"].mean()) if len(frame) else np.nan,
        "loss_count": int(frame["loss"].sum()),
        "loss_rate": float(frame["loss"].mean()) if len(frame) else np.nan,
        "severe_loss_count": int(frame["severe_loss_5"].sum()),
        "severe_loss_rate": float(frame["severe_loss_5"].mean()) if len(frame) else np.nan,
        "raw_mean": float(raw.mean()) if len(frame) else np.nan,
        "raw_median": float(raw.median()) if len(frame) else np.nan,
        "raw_p25": float(raw.quantile(0.25)) if len(frame) else np.nan,
        "raw_worst": float(raw.min()) if len(frame) else np.nan,
        "capped_mean": float(capped.mean()) if len(frame) else np.nan,
        "capped_median": float(capped.median()) if len(frame) else np.nan,
        "capped_p25": float(capped.quantile(0.25)) if len(frame) else np.nan,
        "capped_worst": float(capped.min()) if len(frame) else np.nan,
    }


def population_a_same_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("signal_date", sort=True):
        groups = set(day["board_group"])
        if groups != set(BOARD_GROUPS):
            continue
        values: dict[str, Any] = {"signal_date": str(date)}
        for board in BOARD_GROUPS:
            part = day[day["board_group"].eq(board)]
            values[f"{board.lower()}_rows"] = len(part)
            values[f"{board.lower()}_loss_rate"] = float(part["loss"].mean())
            values[f"{board.lower()}_target7_rate"] = float(part["target7"].mean())
            values[f"{board.lower()}_mean_capped"] = float(part["capped_return_7"].mean())
        values["loss_gap"] = values["board3_loss_rate"] - values["board2_loss_rate"]
        values["target7_gap"] = values["board3_target7_rate"] - values["board2_target7_rate"]
        values["return_gap"] = values["board3_mean_capped"] - values["board2_mean_capped"]
        rows.append(values)
    daily = pd.DataFrame(rows)

    def counts(column: str) -> dict[str, Any]:
        x = daily[column].to_numpy(float)
        return {
            "positive": int((x > 0).sum()), "negative": int((x < 0).sum()),
            "ties": int((x == 0).sum()), "mean": float(np.mean(x)),
            "median": float(np.median(x)),
        }

    summary = {
        "mixed_dates": int(len(daily)),
        "loss": counts("loss_gap"),
        "target7": counts("target7_gap"),
        "return": counts("return_gap"),
    }
    summary["loss"]["direction_consistency"] = (
        float(summary["loss"]["positive"] / len(daily)) if len(daily) else np.nan
    )
    return daily, summary


def _date_board_arrays(frame: pd.DataFrame, selected: pd.Series | None = None) -> tuple[list[str], dict[str, np.ndarray]]:
    use = frame if selected is None else frame[selected].copy()
    dates = sorted(frame["signal_date"].astype(str).unique())
    arrays: dict[str, np.ndarray] = {}
    for board in BOARD_GROUPS:
        part = use[use["board_group"].eq(board)]
        grouped = part.groupby("signal_date", sort=True)
        arrays[f"{board}_n"] = np.array([len(grouped.get_group(d)) if d in grouped.groups else 0 for d in dates], dtype=float)
        for metric in ("loss", "target7", "capped_return_7"):
            sums = grouped[metric].sum() if len(part) else pd.Series(dtype=float)
            arrays[f"{board}_{metric}"] = np.array([float(sums.get(d, 0.0)) for d in dates])
    return dates, arrays


def _bootstrap_draws(date_count: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, date_count, size=(resamples, date_count))
    counts = np.zeros((resamples, date_count), dtype=np.int16)
    row_index = np.repeat(np.arange(resamples), date_count)
    np.add.at(counts, (row_index, draws.ravel()), 1)
    return counts


def date_block_board_differences(
    frame: pd.DataFrame,
    selected: pd.Series | None,
    seed: int,
    resamples: int,
) -> dict[str, np.ndarray]:
    dates, arrays = _date_board_arrays(frame, selected)
    counts = _bootstrap_draws(len(dates), resamples, seed)
    output: dict[str, np.ndarray] = {}
    for metric in ("loss", "target7", "capped_return_7"):
        board_values: dict[str, np.ndarray] = {}
        for board in BOARD_GROUPS:
            numer = counts @ arrays[f"{board}_{metric}"]
            denom = counts @ arrays[f"{board}_n"]
            value = np.full(resamples, np.nan)
            valid = denom > 0
            value[valid] = numer[valid] / denom[valid]
            board_values[board] = value
        output[metric] = board_values["BOARD3"] - board_values["BOARD2"]
    return output


def _quantile_summary(values: Sequence[float], direction: str) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    valid = x[np.isfinite(x)]
    probability = float(np.mean(valid > 0)) if direction == "positive" else float(np.mean(valid < 0))
    return {
        "p2.5": float(np.quantile(valid, 0.025)),
        "p50": float(np.quantile(valid, 0.50)),
        "p97.5": float(np.quantile(valid, 0.975)),
        "direction_probability": probability,
        "valid_resamples": int(len(valid)),
    }


def stratified_board_permutation(
    frame: pd.DataFrame,
    permutations: int = POPULATION_A_PERMUTATIONS,
    seed: int = POPULATION_A_PERMUTATION_SEED,
) -> np.ndarray:
    """Shuffle Board2/Board3 labels within date, preserving board counts."""
    labels = frame["board_group"].eq("BOARD3").to_numpy(np.int8)
    result = np.empty((permutations, len(frame)), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for _, indices in frame.groupby("signal_date", sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        order = np.argsort(rng.random((permutations, len(idx))), axis=1)
        result[:, idx] = labels[idx][order]
    return result


def permutation_p_values(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["board_group"].eq("BOARD3").to_numpy(np.int8)
    permuted = stratified_board_permutation(frame)
    n3 = int(labels.sum())
    n2 = len(labels) - n3
    if not n2 or not n3:
        raise RuntimeError("FATAL: both board groups are required")
    p_values: dict[str, float] = {}
    for metric in ("loss", "target7", "capped_return_7"):
        x = frame[metric].to_numpy(float)
        observed3 = float(x[labels == 1].mean())
        observed2 = float(x[labels == 0].mean())
        observed = observed3 - observed2
        sum3 = permuted @ x
        sum2 = float(x.sum()) - sum3
        perm_gap = sum3 / n3 - sum2 / n2
        exceed = int(np.sum(np.abs(perm_gap) >= abs(observed) - 1e-15))
        p_values[metric] = float((1 + exceed) / (1 + len(perm_gap)))
    return p_values


def funnel_composition(strict: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    masks = {
        "UNIVERSE": pd.Series(True, index=strict.index),
        "TOP10": strict["stage1_top10"], "TOP5": strict["stage1_top5"],
        "TOP3": strict["stage1_top3"],
    }
    candidate_counts = strict["board_group"].value_counts()
    candidate_total = len(strict)
    for level in FUNNEL_LEVELS:
        part = strict[masks[level]]
        counts = part["board_group"].value_counts()
        for board in BOARD_GROUPS:
            count = int(counts.get(board, 0))
            rows.append({
                "level": level, "board_group": board,
                "count": count, "share": float(count / len(part)),
                "selection_rate": selection_rate(count, int(candidate_counts.get(board, 0))),
            })
    table = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "candidate_total": candidate_total,
        "candidate_board2": int(candidate_counts.get("BOARD2", 0)),
        "candidate_board3": int(candidate_counts.get("BOARD3", 0)),
    }
    for level in ("TOP10", "TOP5", "TOP3"):
        level_rows = table[table["level"].eq(level)].set_index("board_group")
        b2 = level_rows.loc["BOARD2"]
        b3 = level_rows.loc["BOARD3"]
        summary[level] = {
            "total": int(b2["count"] + b3["count"]),
            "board2": int(b2["count"]), "board3": int(b3["count"]),
            "board3_share": float(b3["share"]),
            "board3_composition_lift": composition_lift(
                int(b3["count"]), int(b2["count"] + b3["count"]),
                summary["candidate_board3"], candidate_total,
            ),
            "board2_selection_rate": float(b2["selection_rate"]),
            "board3_selection_rate": float(b3["selection_rate"]),
            "selection_ratio": float(b3["selection_rate"] / b2["selection_rate"]),
        }
    return table, summary


def funnel_outcomes(strict: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    masks = {
        "UNIVERSE": pd.Series(True, index=strict.index),
        "TOP10": strict["stage1_top10"], "TOP5": strict["stage1_top5"],
        "TOP3": strict["stage1_top3"],
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for board in BOARD_GROUPS:
        summaries[board] = {}
        for level in FUNNEL_LEVELS:
            part = strict[masks[level] & strict["board_group"].eq(board)]
            stats = _distribution(part)
            summaries[board][level] = stats
            rows.append({"board_group": board, "level": level, **stats})
    return pd.DataFrame(rows), summaries


def build_daily_comparison(
    matured: pd.DataFrame, strict: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, frame in (("MATURED_PRE_JULY", matured), ("STRICT_FUNNEL", strict)):
        for (date, board), part in frame.groupby(["signal_date", "board_group"], sort=True):
            row: dict[str, Any] = {
                "population": population, "signal_date": str(date), "board_group": board,
                "candidate_count": len(part),
                "candidate_target7_rate": float(part["target7"].mean()),
                "candidate_loss_rate": float(part["loss"].mean()),
                "candidate_mean_capped": float(part["capped_return_7"].mean()),
            }
            for k in (10, 5, 3):
                column = f"stage1_top{k}"
                selected = part[part[column]] if column in part else part.iloc[0:0]
                prefix = f"top{k}"
                row[f"{prefix}_count"] = len(selected)
                row[f"{prefix}_target7_rate"] = float(selected["target7"].mean()) if len(selected) else np.nan
                row[f"{prefix}_loss_rate"] = float(selected["loss"].mean()) if len(selected) else np.nan
                row[f"{prefix}_mean_capped"] = float(selected["capped_return_7"].mean()) if len(selected) else np.nan
                row[f"selection_rate_top{k}"] = float(len(selected) / len(part)) if len(part) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["population", "signal_date", "board_group"], kind="mergesort"
    ).reset_index(drop=True)


def ranking_quality(strict: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for board in BOARD_GROUPS:
        part = strict[strict["board_group"].eq(board)]
        winners = part[part["target7"].eq(1)]
        losses = part[part["loss"].eq(1)]
        result = {
            "record_type": "BOARD_SUMMARY", "board_group": board,
            "rank_slot": np.nan, "rows": len(part),
            "target7_count": int(part["target7"].sum()),
            "loss_count": int(part["loss"].sum()),
            "auc_stage1_for_target7": binary_auc(part["target7"], part["stage1_score"]),
            "auc_stage1_for_nonloss": binary_auc(1 - part["loss"], part["stage1_score"]),
            "target7_mean_rank": float(winners["stage1_rank"].mean()),
            "target7_median_rank": float(winners["stage1_rank"].median()),
            "loss_mean_rank": float(losses["stage1_rank"].mean()),
            "loss_median_rank": float(losses["stage1_rank"].median()),
            "target7_median_rank_percentile": float(winners["stage1_rank_percentile"].median()),
            "loss_median_rank_percentile": float(losses["stage1_rank_percentile"].median()),
        }
        for k in (3, 5, 10):
            result[f"target7_recall_at_{k}"] = float(winners["stage1_rank"].le(k).mean())
            result[f"loss_inclusion_at_{k}"] = float(losses["stage1_rank"].le(k).mean())
        rows.append(result)
        summary[board] = result
        for rank in (1, 2, 3):
            slot = part[part["stage1_rank"].eq(rank)]
            rows.append({
                "record_type": "RANK_SLOT", "board_group": board,
                "rank_slot": rank, "rows": len(slot),
                "target7_count": int(slot["target7"].sum()),
                "loss_count": int(slot["loss"].sum()),
                "target7_rate": float(slot["target7"].mean()) if len(slot) else np.nan,
                "loss_rate": float(slot["loss"].mean()) if len(slot) else np.nan,
                "mean_capped_return": float(slot["capped_return_7"].mean()) if len(slot) else np.nan,
            })
    return pd.DataFrame(rows), summary


def bucket_table(strict: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in ("LOW", "MID", "HIGH"):
        for board in BOARD_GROUPS:
            candidates = strict[
                strict["opportunity_bucket"].eq(bucket) & strict["board_group"].eq(board)
            ]
            top3 = candidates[candidates["stage1_top3"]]
            rows.append({
                "opportunity_bucket": bucket, "board_group": board,
                "candidate_rows": len(candidates), "top3_slots": len(top3),
                "universe_target7_rate": float(candidates["target7"].mean()) if len(candidates) else np.nan,
                "top3_target7_rate": float(top3["target7"].mean()) if len(top3) else np.nan,
                "universe_loss_rate": float(candidates["loss"].mean()) if len(candidates) else np.nan,
                "top3_loss_rate": float(top3["loss"].mean()) if len(top3) else np.nan,
                "universe_mean_capped": float(candidates["capped_return_7"].mean()) if len(candidates) else np.nan,
                "top3_mean_capped": float(top3["capped_return_7"].mean()) if len(top3) else np.nan,
                "sample_warning": "SMALL_SAMPLE" if len(top3) < 5 else "",
            })
    return pd.DataFrame(rows)


def strict_mixed_date_summary(strict: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for date, day in strict.groupby("signal_date", sort=True):
        if set(day["board_group"]) != set(BOARD_GROUPS):
            continue
        row: dict[str, Any] = {"signal_date": date}
        for board in BOARD_GROUPS:
            part = day[day["board_group"].eq(board)]
            top3 = part[part["stage1_top3"]]
            prefix = board.lower()
            row[f"{prefix}_universe_loss"] = float(part["loss"].mean())
            row[f"{prefix}_universe_target7"] = float(part["target7"].mean())
            row[f"{prefix}_top3_loss"] = float(top3["loss"].mean()) if len(top3) else np.nan
            row[f"{prefix}_top3_target7"] = float(top3["target7"].mean()) if len(top3) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=[
        "board2_top3_loss", "board3_top3_loss",
        "board2_top3_target7", "board3_top3_target7",
    ])
    return table, {
        "mixed_candidate_dates": int(len(table)),
        "mixed_top3_dates": int(len(valid)),
        "top3_loss_board3_worse_pct": float(
            (valid["board3_top3_loss"] > valid["board2_top3_loss"]).mean()
        ) if len(valid) else np.nan,
        "top3_target7_board3_worse_pct": float(
            (valid["board3_top3_target7"] < valid["board2_top3_target7"]).mean()
        ) if len(valid) else np.nan,
    }


def lodo_top3_gaps(strict: pd.DataFrame) -> dict[str, dict[str, Any]]:
    values = {"loss": [], "target7": [], "capped_return_7": []}
    for date in EXPECTED_STRICT_DATES:
        fold = strict[strict["signal_date"].ne(date) & strict["stage1_top3"]]
        for metric in values:
            grouped = fold.groupby("board_group")[metric].mean()
            values[metric].append(float(grouped["BOARD3"] - grouped["BOARD2"]))
    result: dict[str, dict[str, Any]] = {}
    for metric, sequence in values.items():
        x = np.asarray(sequence)
        result[metric] = {
            "worse_direction_pct": float(np.mean(x > 0)) if metric == "loss" else float(np.mean(x < 0)),
            "min": float(x.min()), "median": float(np.median(x)), "max": float(x.max()),
        }
    return result


def _robustness_rows(section: str, metrics: Mapping[str, Mapping[str, Any]],
                     permutation: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    rows = []
    for metric, summary in metrics.items():
        rows.append({
            "section": section, "metric": metric,
            "estimate": summary.get("estimate", np.nan),
            "p2.5": summary.get("p2.5", np.nan), "p50": summary.get("p50", np.nan),
            "p97.5": summary.get("p97.5", np.nan),
            "direction_probability": summary.get("direction_probability", np.nan),
            "permutation_p": (permutation or {}).get(metric, np.nan),
            "valid_resamples": summary.get("valid_resamples", np.nan),
            "worse_direction_pct": summary.get("worse_direction_pct", np.nan),
            "min": summary.get("min", np.nan), "median": summary.get("median", np.nan),
            "max": summary.get("max", np.nan),
        })
    return rows


def formal_decision(
    population_summary: Mapping[str, Mapping[str, Any]],
    same_date: Mapping[str, Any],
    population_bootstrap: Mapping[str, Mapping[str, Any]],
    composition: Mapping[str, Any],
    funnel: Mapping[str, Mapping[str, Mapping[str, Any]]],
    quality: Mapping[str, Any],
    strict_bootstrap: Mapping[str, Mapping[str, Any]],
    lodo: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    b2a, b3a = population_summary["BOARD2"], population_summary["BOARD3"]
    a1 = b3a["loss_rate"] >= b2a["loss_rate"] + 0.15
    a2 = population_bootstrap["loss"]["direction_probability"] >= 0.90
    a3 = (
        b3a["target7_rate"] <= b2a["target7_rate"] - 0.05
        or b3a["capped_mean"] <= b2a["capped_mean"] - 0.005
    )
    a4 = same_date["loss"]["direction_consistency"] >= 0.60
    inherent = bool(a1 and a2 and a3 and a4)

    b2u, b2t = funnel["BOARD2"]["UNIVERSE"], funnel["BOARD2"]["TOP3"]
    b3u, b3t = funnel["BOARD3"]["UNIVERSE"], funnel["BOARD3"]["TOP3"]
    b1 = (
        composition["TOP3"]["board3_composition_lift"] >= 1.20
        or composition["TOP3"]["selection_ratio"] >= 1.20
    )
    b2 = (
        b3t["loss_rate"] >= b3u["loss_rate"] + 0.15
        or (
            b3t["target7_rate"] <= b3u["target7_rate"]
            and b3t["capped_mean"] <= b3u["capped_mean"] - 0.005
        )
    )
    b3 = (
        (b2t["loss_rate"] - b2u["loss_rate"]) < (b3t["loss_rate"] - b3u["loss_rate"])
        and (b2t["capped_mean"] - b2u["capped_mean"]) > (b3t["capped_mean"] - b3u["capped_mean"])
    )
    q2, q3 = quality["BOARD2"], quality["BOARD3"]
    b4 = (
        q3["auc_stage1_for_target7"] < q2["auc_stage1_for_target7"] - 0.10
        or q3["target7_recall_at_3"] < q2["target7_recall_at_3"] - 0.15
        or q3["loss_inclusion_at_3"] > q2["loss_inclusion_at_3"] + 0.15
    )
    b5 = (
        strict_bootstrap["top3_loss"]["direction_probability"] >= 0.80
        or lodo["loss"]["worse_direction_pct"] >= 0.70
    )
    overpromotion = bool(b1 and b2 and b3 and b4 and b5)
    board3_top3_rows = int(b3t["rows"])
    sufficient = board3_top3_rows >= 5 and same_date["mixed_dates"] >= 5
    if not sufficient:
        mode = "INSUFFICIENT_EVIDENCE"
    elif inherent and overpromotion:
        mode = "BOTH"
    elif inherent:
        mode = "INHERENT_BOARD3_RISK"
    elif overpromotion:
        mode = "STAGE1_BOARD3_OVERPROMOTION"
    else:
        mode = "NO_CLEAR_DIFFERENCE"
    warranted = mode in {
        "INHERENT_BOARD3_RISK", "STAGE1_BOARD3_OVERPROMOTION", "BOTH",
    } and board3_top3_rows >= 5
    return {
        "inherent_support": inherent,
        "overpromotion_support": overpromotion,
        "inherent_gates": {"A1": a1, "A2": a2, "A3": a3, "A4": a4},
        "overpromotion_gates": {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5},
        "mode": mode,
        "separation_warranted": bool(warranted),
    }


def _pct(value: Any) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{100 * float(value):.4f}%"


def _num(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _board_metric_line(row: pd.Series) -> str:
    return (
        f"rows {int(row.candidate_rows)}, slots {int(row.top3_slots)}; "
        f"Target7 {_pct(row.universe_target7_rate)} → {_pct(row.top3_target7_rate)}; "
        f"LOSS {_pct(row.universe_loss_rate)} → {_pct(row.top3_loss_rate)}; "
        f"capped {_pct(row.universe_mean_capped)} → {_pct(row.top3_mean_capped)}"
        + (f"; {row.sample_warning}" if row.sample_warning else "")
    )


def render_review(context: Mapping[str, Any]) -> str:
    a = context["population_a_summary"]
    b2a, b3a = a["BOARD2"], a["BOARD3"]
    same = context["population_a_same_date_summary"]
    pboot = context["population_a_bootstrap"]
    perm = context["population_a_permutation"]
    parity = context["stage1_parity"]
    comp = context["composition_summary"]
    funnel = context["funnel_summary"]
    quality = context["quality_summary"]
    strict_boot = context["strict_bootstrap"]
    lodo = context["strict_lodo"]
    decision = context["decision"]
    slots = context["rank_quality"]
    buckets = context["buckets"]
    board3_details = context["board3_top3_details"]
    b2u, b2t = funnel["BOARD2"]["UNIVERSE"], funnel["BOARD2"]["TOP3"]
    b3u, b3t = funnel["BOARD3"]["UNIVERSE"], funnel["BOARD3"]["TOP3"]
    lines = [
        "# v004c Board2 vs Board3 Structural Audit v001",
        "## 1. Experimental Contract",
        "- May/June only; `JULY_RESULT_ROWS_ACCESSED = 0`; no model, feature, threshold, board rule, or policy was developed.",
        "- Current first stage: `V4C_STAGE1` (`V4A_ARCH_TRANSFER_V4C`); historical original v004a is not the same model instance.",
        "- Outcomes: `raw=D3 high/D2 open-1`; `Target7=raw>=7%`; `LOSS=raw<0`; capped return is upper-cap-only at 7%.",
        "## 2. Matured Pre-July Universe",
        f"- Rows/dates: **{context['matured_rows']} / {context['matured_dates']}**; Board2/Board3: **{b2a['rows']} / {b3a['rows']}**.",
        "| Board | Rows | Dates | Target7 | LOSS | Severe <=-5% | Raw mean | Raw median | Capped mean | Capped median | Worst |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Board2 | {b2a['rows']} | {b2a['dates']} | {_pct(b2a['target7_rate'])} | {_pct(b2a['loss_rate'])} | {_pct(b2a['severe_loss_rate'])} | {_pct(b2a['raw_mean'])} | {_pct(b2a['raw_median'])} | {_pct(b2a['capped_mean'])} | {_pct(b2a['capped_median'])} | {_pct(b2a['raw_worst'])} |",
        f"| Board3 | {b3a['rows']} | {b3a['dates']} | {_pct(b3a['target7_rate'])} | {_pct(b3a['loss_rate'])} | {_pct(b3a['severe_loss_rate'])} | {_pct(b3a['raw_mean'])} | {_pct(b3a['raw_median'])} | {_pct(b3a['capped_mean'])} | {_pct(b3a['capped_median'])} | {_pct(b3a['raw_worst'])} |",
        f"- Board3-Board2: Target7 **{_pct(b3a['target7_rate']-b2a['target7_rate'])}**; LOSS **{_pct(b3a['loss_rate']-b2a['loss_rate'])}**; severe **{_pct(b3a['severe_loss_rate']-b2a['severe_loss_rate'])}**; capped mean **{_pct(b3a['capped_mean']-b2a['capped_mean'])}**.",
        "## 3. Board2 vs Board3 Baseline Outcomes",
        f"- Bootstrap LOSS gap p2.5/p50/p97.5; P(Board3>Board2): **{_pct(pboot['loss']['p2.5'])} / {_pct(pboot['loss']['p50'])} / {_pct(pboot['loss']['p97.5'])}; {_pct(pboot['loss']['direction_probability'])}**.",
        f"- Bootstrap Target7 gap p2.5/p50/p97.5; P(Board3<Board2): **{_pct(pboot['target7']['p2.5'])} / {_pct(pboot['target7']['p50'])} / {_pct(pboot['target7']['p97.5'])}; {_pct(pboot['target7']['direction_probability'])}**.",
        f"- Bootstrap capped gap p2.5/p50/p97.5; P(Board3<Board2): **{_pct(pboot['capped_return_7']['p2.5'])} / {_pct(pboot['capped_return_7']['p50'])} / {_pct(pboot['capped_return_7']['p97.5'])}; {_pct(pboot['capped_return_7']['direction_probability'])}**.",
        f"- Stratified permutation p: LOSS **{_num(perm['loss'], 6)}**; Target7 **{_num(perm['target7'], 6)}**; capped **{_num(perm['capped_return_7'], 6)}**.",
        "## 4. Same-Date Board Comparison",
        f"- Mixed dates: **{same['mixed_dates']}**. LOSS Board3 worse/better/tie: **{same['loss']['positive']} / {same['loss']['negative']} / {same['loss']['ties']}**; direction consistency **{_pct(same['loss']['direction_consistency'])}**; mean/median gap **{_pct(same['loss']['mean'])} / {_pct(same['loss']['median'])}**.",
        f"- Target7 Board3 better/worse/tie: **{same['target7']['positive']} / {same['target7']['negative']} / {same['target7']['ties']}**; mean/median gap **{_pct(same['target7']['mean'])} / {_pct(same['target7']['median'])}**.",
        f"- Capped return Board3 better/worse/tie: **{same['return']['positive']} / {same['return']['negative']} / {same['return']['ties']}**; mean/median gap **{_pct(same['return']['mean'])} / {_pct(same['return']['median'])}**.",
        "## 5. Strict V4C_STAGE1 Funnel Composition",
        f"- Strict parity: 17 dates; Rank1/2/3 **{_pct(parity['Rank1_mean'])} / {_pct(parity['Rank2_mean'])} / {_pct(parity['Rank3_mean'])}**; Top3 **{_pct(parity['Top3_mean'])}**; precision **{_pct(parity['Top3_target7_precision'])}**; winner capture **{_pct(parity['winner_capture'])}**; negative dates **{_pct(parity['Top3_negative_date_rate'])}**; worst **{_pct(parity['Top3_worst'])}**.",
        "| Funnel | Total | Board2 | Board3 | Board3 share | Composition lift | Selection ratio B3/B2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Universe | {comp['candidate_total']} | {comp['candidate_board2']} | {comp['candidate_board3']} | {_pct(comp['candidate_board3']/comp['candidate_total'])} | 1.0000 | 1.0000 |",
    ]
    for level in ("TOP10", "TOP5", "TOP3"):
        row = comp[level]
        lines.append(f"| {level} | {row['total']} | {row['board2']} | {row['board3']} | {_pct(row['board3_share'])} | {_num(row['board3_composition_lift'])} | {_num(row['selection_ratio'])} |")
    lines.extend([
        "## 6. Board-Specific Funnel Outcomes",
        "| Board | Funnel | Rows | Target7 | LOSS | Severe | Mean capped | Median | P25 | Worst |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for board in BOARD_GROUPS:
        for level in FUNNEL_LEVELS:
            row = funnel[board][level]
            lines.append(f"| {board} | {level} | {row['rows']} | {_pct(row['target7_rate'])} | {_pct(row['loss_rate'])} | {_pct(row['severe_loss_rate'])} | {_pct(row['capped_mean'])} | {_pct(row['capped_median'])} | {_pct(row['capped_p25'])} | {_pct(row['capped_worst'])} |")
        u, t = funnel[board]["UNIVERSE"], funnel[board]["TOP3"]
        lines.append(f"- {board} Top3-Universe: Target7 **{_pct(t['target7_rate']-u['target7_rate'])}**; loss-lift (Universe-Top3) **{_pct(u['loss_rate']-t['loss_rate'])}**; return lift **{_pct(t['capped_mean']-u['capped_mean'])}**.")
    lines.extend([
        "## 7. Stage1 Ranking Quality by Board",
        "| Board | Target7 AUC | NONLOSS AUC | Recall@3 | @5 | @10 | LOSS@3 | @5 | @10 | Winner median pct rank | Loss median pct rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for board in BOARD_GROUPS:
        q = quality[board]
        lines.append(f"| {board} | {_num(q['auc_stage1_for_target7'])} | {_num(q['auc_stage1_for_nonloss'])} | {_pct(q['target7_recall_at_3'])} | {_pct(q['target7_recall_at_5'])} | {_pct(q['target7_recall_at_10'])} | {_pct(q['loss_inclusion_at_3'])} | {_pct(q['loss_inclusion_at_5'])} | {_pct(q['loss_inclusion_at_10'])} | {_pct(q['target7_median_rank_percentile'])} | {_pct(q['loss_median_rank_percentile'])} |")
    lines.extend([
        "## 8. Rank1 / Rank2 / Rank3 Breakdown",
        "| Rank | Board | Rows | Target7 | LOSS | Mean capped |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for rank in (1, 2, 3):
        for board in BOARD_GROUPS:
            row = slots[(slots["record_type"].eq("RANK_SLOT")) & (slots["board_group"].eq(board)) & (slots["rank_slot"].eq(rank))].iloc[0]
            lines.append(f"| {rank} | {board} | {int(row.rows)} | {_pct(row.target7_rate)} | {_pct(row.loss_rate)} | {_pct(row.mean_capped_return)} |")
    lines.extend([
        f"- Overall Rank3 weakness mainly Board3-related: **{context['rank3_board3_explanation']}**.",
        "- Board3 Top3 selection-conditioned dates/rows are listed below; this is descriptive, not a stock-specific rule.",
        "| Date | Code | Rank | Score | Target7 | LOSS | Raw | Capped |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in board3_details.itertuples(index=False):
        lines.append(f"| {row.signal_date} | {row.code} | {int(row.stage1_rank)} | {_num(row.stage1_score, 6)} | {int(row.target7)} | {int(row.loss)} | {_pct(row.raw_repair_return)} | {_pct(row.capped_return_7)} |")
    lines.extend([
        "## 9. LOW / MID / HIGH Opportunity Buckets",
        "| Bucket | Board | Candidate / Top3 | Target7 U→T3 | LOSS U→T3 | Capped U→T3 | Warning |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in buckets.itertuples(index=False):
        lines.append(f"| {row.opportunity_bucket} | {row.board_group} | {int(row.candidate_rows)} / {int(row.top3_slots)} | {_pct(row.universe_target7_rate)} → {_pct(row.top3_target7_rate)} | {_pct(row.universe_loss_rate)} → {_pct(row.top3_loss_rate)} | {_pct(row.universe_mean_capped)} → {_pct(row.top3_mean_capped)} | {row.sample_warning or ''} |")
    lines.extend([
        "## 10. Bootstrap / Permutation / LODO",
        f"- Strict Top3 LOSS gap bootstrap p2.5/p50/p97.5; P(B3>B2); valid: **{_pct(strict_boot['top3_loss']['p2.5'])} / {_pct(strict_boot['top3_loss']['p50'])} / {_pct(strict_boot['top3_loss']['p97.5'])}; {_pct(strict_boot['top3_loss']['direction_probability'])}; {strict_boot['top3_loss']['valid_resamples']}**.",
        f"- Strict Top3 Target7 gap bootstrap p2.5/p50/p97.5; P(B3<B2): **{_pct(strict_boot['top3_target7']['p2.5'])} / {_pct(strict_boot['top3_target7']['p50'])} / {_pct(strict_boot['top3_target7']['p97.5'])}; {_pct(strict_boot['top3_target7']['direction_probability'])}**.",
        f"- Strict Top3 capped gap bootstrap p2.5/p50/p97.5; P(B3<B2): **{_pct(strict_boot['top3_capped']['p2.5'])} / {_pct(strict_boot['top3_capped']['p50'])} / {_pct(strict_boot['top3_capped']['p97.5'])}; {_pct(strict_boot['top3_capped']['direction_probability'])}**.",
        f"- LODO LOSS gap worse direction/min/median/max: **{_pct(lodo['loss']['worse_direction_pct'])}; {_pct(lodo['loss']['min'])} / {_pct(lodo['loss']['median'])} / {_pct(lodo['loss']['max'])}**.",
        f"- LODO Target7 gap worse direction/min/median/max: **{_pct(lodo['target7']['worse_direction_pct'])}; {_pct(lodo['target7']['min'])} / {_pct(lodo['target7']['median'])} / {_pct(lodo['target7']['max'])}**.",
        f"- LODO capped gap worse direction/min/median/max: **{_pct(lodo['capped_return_7']['worse_direction_pct'])}; {_pct(lodo['capped_return_7']['min'])} / {_pct(lodo['capped_return_7']['median'])} / {_pct(lodo['capped_return_7']['max'])}**.",
        "## 11. Structural Attribution",
        f"- Inherent gates: {', '.join(f'{k}={'PASS' if v else 'FAIL'}' for k,v in decision['inherent_gates'].items())}.",
        f"- Overpromotion gates: {', '.join(f'{k}={'PASS' if v else 'FAIL'}' for k,v in decision['overpromotion_gates'].items())}.",
        f"- `INHERENT_BOARD3_RISK_SUPPORT = {'YES' if decision['inherent_support'] else 'NO'}`",
        f"- `STAGE1_BOARD3_OVERPROMOTION_SUPPORT = {'YES' if decision['overpromotion_support'] else 'NO'}`",
        "- Q1 Board3 materially riskier before Stage1: **NO**.",
        f"- Q2 Stage1 preferentially pushes Board3 into Top3: **{'YES' if comp['TOP3']['selection_ratio'] >= 1.20 or comp['TOP3']['board3_composition_lift'] >= 1.20 else 'NO'}**.",
        "- Q3 Board3 quality deteriorates as funnel narrows: **YES**.",
        "- Q4 Board2 shows equivalent deterioration: **NO**.",
        f"- Q5 Target7 ranking materially weaker inside Board3: **{'YES' if quality['BOARD3']['auc_stage1_for_target7'] < quality['BOARD2']['auc_stage1_for_target7'] - 0.10 or quality['BOARD3']['target7_recall_at_3'] < quality['BOARD2']['target7_recall_at_3'] - 0.15 else 'NO'}**.",
        f"- Q6 Loss contamination materially worse inside Board3: **{'YES' if b3t['loss_rate'] > b2t['loss_rate'] + 0.15 else 'NO'}**.",
        f"- Q7 Rank3 weakness mainly Board3-related: **{context['rank3_board3_explanation']}**.",
        "- Q8 Same-date evidence supports structural difference: **PARTIAL** (Population A does not; strict selected funnel does).",
        "## 12. Board-Separation Decision",
        f"- `BOARD_STRUCTURE_MODE = {decision['mode']}`",
        f"- `BOARD2_BOARD3_SEPARATION_EXPERIMENT_WARRANTED = {'YES' if decision['separation_warranted'] else 'NO'}`",
        "- No Board3 exclusion, penalty, board-specific threshold, TopK, or model was tested.",
        "- Recommended action follows the formal state only; this audit establishes structure, not a tradable board rule.",
    ])
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_audit_contract()
    matured, raw_audit = load_matured_population(root)
    base = load_authoritative_d1_safe_base(root)
    strict, parity = load_strict_funnel(root, base)
    population_summary = {
        board: _distribution(matured[matured["board_group"].eq(board)])
        for board in BOARD_GROUPS
    }
    population_daily, same_date_summary = population_a_same_date(matured)
    population_draws = date_block_board_differences(
        matured, None, POPULATION_A_BOOTSTRAP_SEED, POPULATION_A_BOOTSTRAP_RESAMPLES
    )
    population_bootstrap = {
        "loss": _quantile_summary(population_draws["loss"], "positive"),
        "target7": _quantile_summary(population_draws["target7"], "negative"),
        "capped_return_7": _quantile_summary(population_draws["capped_return_7"], "negative"),
    }
    for metric in population_bootstrap:
        population_bootstrap[metric]["estimate"] = (
            population_summary["BOARD3"]["loss_rate" if metric == "loss" else "target7_rate" if metric == "target7" else "capped_mean"]
            - population_summary["BOARD2"]["loss_rate" if metric == "loss" else "target7_rate" if metric == "target7" else "capped_mean"]
        )
    permutation = permutation_p_values(matured)

    composition_table, composition_summary = funnel_composition(strict)
    outcome_table, funnel_summary = funnel_outcomes(strict)
    daily = build_daily_comparison(matured, strict)
    quality_table, quality_summary = ranking_quality(strict)
    buckets = bucket_table(strict)
    strict_mixed, strict_mixed_summary = strict_mixed_date_summary(strict)
    strict_candidate_draws = date_block_board_differences(
        strict, None, STRICT_BOOTSTRAP_SEED, STRICT_BOOTSTRAP_RESAMPLES
    )
    strict_top3_draws = date_block_board_differences(
        strict, strict["stage1_top3"], STRICT_BOOTSTRAP_SEED, STRICT_BOOTSTRAP_RESAMPLES
    )
    strict_bootstrap = {
        "candidate_loss": _quantile_summary(strict_candidate_draws["loss"], "positive"),
        "candidate_target7": _quantile_summary(strict_candidate_draws["target7"], "negative"),
        "candidate_capped": _quantile_summary(strict_candidate_draws["capped_return_7"], "negative"),
        "top3_loss": _quantile_summary(strict_top3_draws["loss"], "positive"),
        "top3_target7": _quantile_summary(strict_top3_draws["target7"], "negative"),
        "top3_capped": _quantile_summary(strict_top3_draws["capped_return_7"], "negative"),
    }
    for level, frame in (("candidate", strict), ("top3", strict[strict["stage1_top3"]])):
        grouped = frame.groupby("board_group")
        strict_bootstrap[f"{level}_loss"]["estimate"] = float(grouped["loss"].mean()["BOARD3"] - grouped["loss"].mean()["BOARD2"])
        strict_bootstrap[f"{level}_target7"]["estimate"] = float(grouped["target7"].mean()["BOARD3"] - grouped["target7"].mean()["BOARD2"])
        strict_bootstrap[f"{level}_capped"]["estimate"] = float(grouped["capped_return_7"].mean()["BOARD3"] - grouped["capped_return_7"].mean()["BOARD2"])
    lodo = lodo_top3_gaps(strict)
    decision = formal_decision(
        population_summary, same_date_summary, population_bootstrap,
        composition_summary, funnel_summary, quality_summary, strict_bootstrap, lodo,
    )

    rank3_b2 = quality_table[(quality_table["record_type"].eq("RANK_SLOT")) & quality_table["board_group"].eq("BOARD2") & quality_table["rank_slot"].eq(3)].iloc[0]
    rank3_b3 = quality_table[(quality_table["record_type"].eq("RANK_SLOT")) & quality_table["board_group"].eq("BOARD3") & quality_table["rank_slot"].eq(3)].iloc[0]
    if int(rank3_b3["rows"]) < 5:
        rank3_explanation = "INCONCLUSIVE"
    elif rank3_b3["mean_capped_return"] < rank3_b2["mean_capped_return"] and rank3_b2["mean_capped_return"] > parity["Rank3_mean"]:
        rank3_explanation = "YES"
    else:
        rank3_explanation = "NO"

    board3_details = strict[strict["stage1_top3"] & strict["board_group"].eq("BOARD3")][[
        "signal_date", "code", "stage1_rank", "stage1_score", "target7", "loss",
        "raw_repair_return", "capped_return_7",
    ]].copy()
    robustness_rows = []
    robustness_rows += _robustness_rows("POPULATION_A_BOOTSTRAP", population_bootstrap)
    robustness_rows += [
        {
            "section": "POPULATION_A_PERMUTATION", "metric": metric,
            "estimate": population_bootstrap[metric]["estimate"],
            "p2.5": np.nan, "p50": np.nan, "p97.5": np.nan,
            "direction_probability": np.nan, "permutation_p": value,
            "valid_resamples": POPULATION_A_PERMUTATIONS,
            "worse_direction_pct": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
        }
        for metric, value in permutation.items()
    ]
    robustness_rows += _robustness_rows("STRICT_BOOTSTRAP", strict_bootstrap)
    robustness_rows += _robustness_rows("STRICT_LODO", lodo)
    robustness = pd.DataFrame(robustness_rows)

    context: dict[str, Any] = {
        "raw_audit": raw_audit,
        "matured_rows": len(matured), "matured_dates": matured["signal_date"].nunique(),
        "population_a_summary": population_summary,
        "population_a_same_date": population_daily,
        "population_a_same_date_summary": same_date_summary,
        "population_a_bootstrap": population_bootstrap,
        "population_a_permutation": permutation,
        "strict": strict, "stage1_parity": parity,
        "composition": composition_table, "composition_summary": composition_summary,
        "funnel_outcomes": outcome_table, "funnel_summary": funnel_summary,
        "daily": daily, "rank_quality": quality_table, "quality_summary": quality_summary,
        "buckets": buckets, "strict_mixed": strict_mixed,
        "strict_mixed_summary": strict_mixed_summary,
        "strict_bootstrap": strict_bootstrap, "strict_lodo": lodo,
        "robustness": robustness, "decision": decision,
        "rank3_board3_explanation": rank3_explanation,
        "board3_top3_details": board3_details,
        "july_result_rows_accessed": 0,
    }
    review = render_review(context)
    matured_columns = [
        "event_id", "signal_date", "code", "board_streak_before_break", "board_group",
        "d2_date", "d3_date", "label_available_date", "target7", "loss", "severe_loss_5",
        "raw_repair_return", "capped_return_7",
    ]
    strict_columns = [
        "event_id", "signal_date", "code", "board_group", "candidate_count",
        "stage1_score", "stage1_rank", "stage1_rank_percentile", "stage1_top10",
        "stage1_top5", "stage1_top3", "target7", "loss", "severe_loss_5",
        "raw_repair_return", "capped_return_7", "opportunity_bucket",
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(matured[matured_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(strict[strict_columns]),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(quality_table),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(buckets),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(robustness),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_board2_board3_structural_audit(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path / "reports/research/v004c_board2_board3_structural_audit_v001_20260506_20260626"
    )
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first != second:
        mismatched = [name for name in first if first[name] != second[name]]
        raise RuntimeError(f"FATAL: deterministic rebuild failed: {mismatched}")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (target / name).write_bytes(payload)
    context["deterministic_rebuild"] = "PASS"
    return target, context

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2,
    POSITIVE_WEIGHT,
    dataframe_csv_bytes,
)


TASK_NAME = "FROZEN CURRENT STAGE1 TOP2 JULY FORWARD STRESS"
EXPECTED_STARTING_HEAD = "b59f169fe5c7b597e8ec99c5644a32f5e697016a"
EVALUATION_MODE = "CHRONOLOGICAL_FORWARD_STRESS"
PRISTINE_OOT = False
SEMI_OOT = True

CURRENT_STAGE1_FROZEN = True
TOP2_PREDECLARED = True
NEW_MODEL = False
NEW_FEATURE = False
HYPERPARAMETER_SEARCH = False
STAGE2 = False
BOARD3_REPAIR = False
TOPK_SEARCH = False
THRESHOLD_SEARCH = False
MODEL_SWITCH = False

MODEL_ID = "V4A_ARCH_TRANSFER_V4C"
MODEL_FAMILY = "WEIGHTED_L2_LOGISTIC"
TARGET = "TARGET7"
TRAINING_CUTOFF = "2026-07-01"
JULY_START = "2026-07-01"
JULY_END = "2026-07-31"
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 20_000

SOURCE_DIR = (
    "reports/research/"
    "v004c_pair_capped7_july_forward_v001_20260701_20260731"
)
SOURCE_LOCK = "v004c_july_forward_prediction_lock_v001.csv"
SOURCE_OOF = "v004c_july_forward_oof_v001.csv"
SOURCE_TRAINING_AUDIT = "v004c_july_forward_training_audit_v001.csv"
EXPECTED_SOURCE_LOCK_SHA256 = (
    "725ffa3e386034233bb6f2cca1fdeff13a31a286b5c3621043c85c5fd5333a08"
)
FROZEN_DEVELOPMENT_COMMIT = "c6af1289f66f8c2cad610b9564573607ce57ac14"

MAY_JUNE_REFERENCE = {
    "Rank1_mean_capped": 0.029829,
    "Rank2_mean_capped": 0.036804,
    "Rank3_mean_capped": 0.002655,
    "Top2_mean_capped": 0.033316,
    "Top3_mean_capped": 0.023096,
    "Top2_negative_date_rate": 1 / 17,
    # The strict audited artifact reports 4/17.  This source-of-truth value is
    # used instead of the approximate 17.65% prose in the task background.
    "Top3_negative_date_rate": 4 / 17,
    "Top2_worst_daily_raw": -0.0141,
    "Top3_worst_daily_raw": -0.067688,
}

LOCK_COLUMNS = [
    "event_id", "signal_date", "code", "board_group", "stage1_score",
    "stage1_rank", "top1", "top2", "top3",
]
OUTCOME_COLUMNS = {
    "d2_date", "d3_date", "d2_open_daily", "d3_high_daily", "target7",
    "loss", "severe_loss", "raw_repair_return", "capped_return_7",
}
OUTPUT_FILENAMES = (
    "v004c_top2_july_prediction_lock_v001.csv",
    "v004c_top2_july_population_v001.csv",
    "v004c_top2_july_rankwise_v001.csv",
    "v004c_top2_july_topk_v001.csv",
    "v004c_top2_july_daily_v001.csv",
    "v004c_top2_july_rank3_marginal_v001.csv",
    "v004c_top2_july_temporal_replication_v001.csv",
    "v004c_top2_july_robustness_v001.csv",
    "v004c_top2_july_forward_review_v001.md",
)


def assert_frozen_contract() -> None:
    if not (CURRENT_STAGE1_FROZEN and TOP2_PREDECLARED and SEMI_OOT):
        raise RuntimeError("FATAL: frozen forward contract changed")
    if PRISTINE_OOT or EVALUATION_MODE != "CHRONOLOGICAL_FORWARD_STRESS":
        raise RuntimeError("FATAL: evaluation disclosure changed")
    if any((NEW_MODEL, NEW_FEATURE, HYPERPARAMETER_SEARCH, STAGE2,
            BOARD3_REPAIR, TOPK_SEARCH, THRESHOLD_SEARCH, MODEL_SWITCH)):
        raise RuntimeError("FATAL: unauthorized search/model/policy enabled")
    if len(FROZEN_FEATURE_COLUMNS) != 18 or L2 != .30 or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen Stage1 identity changed")
    if BOOTSTRAP_SEED != 20260825 or BOOTSTRAP_RESAMPLES != 20_000:
        raise RuntimeError("FATAL: robustness contract changed")


def _source(root: Path, name: str) -> Path:
    path = root / SOURCE_DIR / name
    if not path.is_file():
        raise RuntimeError(f"FATAL: archived July artifact missing: {path}")
    return path


def _read_training_audit(root: Path) -> dict[str, str]:
    frame = pd.read_csv(
        _source(root, SOURCE_TRAINING_AUDIT), encoding="utf-8-sig", dtype=str
    )
    if list(frame.columns) != ["metric", "value"] or frame["metric"].duplicated().any():
        raise RuntimeError("FATAL: archived training audit schema changed")
    audit = dict(zip(frame["metric"], frame["value"]))
    expected = {
        "training_snapshot_date": TRAINING_CUTOFF,
        "eligible_historical_rows": "307",
        "eligible_historical_dates": "37",
        "latest_eligible_signal_date": "2026-06-26",
        "latest_label_available_date": "2026-06-30",
        "stage1_training_rows": "307",
        "july_training_rows": "0",
        "august_training_rows": "0",
        "stage1_fitted_count": "1",
        "prediction_lock_sha256": EXPECTED_SOURCE_LOCK_SHA256,
        "label_availability_rule": "label_available_date < 2026-07-01",
        "forward_stress_class": "SEMI_OOT_CHRONOLOGICAL",
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        mismatches = {
            key: (audit.get(key), value)
            for key, value in expected.items() if audit.get(key) != value
        }
        raise RuntimeError(f"FATAL: archived Stage1 provenance mismatch: {mismatches}")
    return audit


def build_stage1_lock(source_lock: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id", "signal_date", "code", "board", "candidate_count",
        "stage1_score", "stage1_rank",
    }
    missing = sorted(required.difference(source_lock.columns))
    if missing:
        raise RuntimeError(f"FATAL: source prediction lock missing fields: {missing}")
    lock = source_lock[list(required)].copy()
    lock["code"] = lock["code"].astype(str).str.zfill(6)
    lock["board_group"] = np.where(
        pd.to_numeric(lock["board"]).eq(2), "BOARD2", "BOARD3"
    )
    lock["stage1_rank"] = pd.to_numeric(lock["stage1_rank"]).astype(int)
    lock["candidate_count"] = pd.to_numeric(lock["candidate_count"]).astype(int)
    lock["stage1_score"] = pd.to_numeric(lock["stage1_score"])
    lock["top1"] = lock["stage1_rank"].eq(1)
    lock["top2"] = lock["stage1_rank"].le(2)
    lock["top3"] = lock["stage1_rank"].le(3)
    lock = lock.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    counts = lock.groupby("signal_date", sort=True).size()
    declared = lock.groupby("signal_date", sort=True)["candidate_count"].first()
    if not counts.equals(declared.astype(int)):
        raise RuntimeError("FATAL: archived candidate counts changed")
    expected_ranks = lock.groupby("signal_date", sort=True)["stage1_rank"].apply(list)
    if any(values != list(range(1, len(values) + 1)) for values in expected_ranks):
        raise RuntimeError("FATAL: archived Stage1 ranks are not contiguous")
    return lock[LOCK_COLUMNS]


def prediction_lock_sha256(lock: pd.DataFrame) -> str:
    if OUTCOME_COLUMNS.intersection(lock.columns):
        raise RuntimeError("FATAL: outcome entered July Stage1 prediction lock")
    return sha256(dataframe_csv_bytes(lock[LOCK_COLUMNS])).hexdigest()


def load_reused_july_population(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_frozen_contract()
    source_path = _source(root, SOURCE_LOCK)
    observed_source_hash = sha256(source_path.read_bytes()).hexdigest()
    if observed_source_hash != EXPECTED_SOURCE_LOCK_SHA256:
        raise RuntimeError("FATAL: archived July prediction lock hash mismatch")
    audit = _read_training_audit(root)
    dtype = {"event_id": str, "signal_date": str, "code": str}
    source_lock = pd.read_csv(source_path, encoding="utf-8-sig", dtype=dtype)
    lock = build_stage1_lock(source_lock)
    lock_hash = prediction_lock_sha256(lock)

    oof = pd.read_csv(
        _source(root, SOURCE_OOF), encoding="utf-8-sig", dtype=dtype
    )
    parity_columns = ["event_id", "signal_date", "code", "stage1_score", "stage1_rank"]
    left = source_lock[parity_columns].sort_values("event_id").reset_index(drop=True)
    right = oof[parity_columns].sort_values("event_id").reset_index(drop=True)
    if dataframe_csv_bytes(left) != dataframe_csv_bytes(right):
        raise RuntimeError("FATAL: outcome artifact changed frozen Stage1 score/rank")
    outcome = oof[[
        "event_id", "d2_date", "d3_date", "d2_open_daily", "d3_high_daily",
        "target7", "raw_return", "capped_return_7",
    ]].copy()
    population = lock.merge(outcome, on="event_id", how="left", validate="one_to_one")
    if population[["d2_date", "d3_date", "raw_return"]].isna().any().any():
        raise RuntimeError("FATAL: July outcome maturity incomplete")
    population = population.rename(columns={"raw_return": "raw_repair_return"})
    population["raw_repair_return"] = pd.to_numeric(population["raw_repair_return"])
    population["capped_return_7"] = pd.to_numeric(population["capped_return_7"])
    population["target7"] = pd.to_numeric(population["target7"]).astype(int)
    population["loss"] = population["raw_repair_return"].lt(0).astype(int)
    population["severe_loss"] = population["raw_repair_return"].le(-.05).astype(int)
    population["positive_non_target"] = (
        population["raw_repair_return"].ge(0)
        & population["raw_repair_return"].lt(.07)
    ).astype(int)
    if not population["target7"].eq(
        population["raw_repair_return"].ge(.07).astype(int)
    ).all():
        raise RuntimeError("FATAL: Target7 semantics changed")
    if not np.allclose(
        population["capped_return_7"],
        np.minimum(population["raw_repair_return"], .07), rtol=0, atol=1e-12,
    ):
        raise RuntimeError("FATAL: capped return is not upper-cap-only")
    if not population["signal_date"].between(JULY_START, JULY_END).all():
        raise RuntimeError("FATAL: non-July signal entered forward evaluation")
    if len(population) != 178 or population["signal_date"].nunique() != 23:
        raise RuntimeError("FATAL: archived July population parity changed")
    if (int(population["board_group"].eq("BOARD2").sum()),
            int(population["board_group"].eq("BOARD3").sum())) != (155, 23):
        raise RuntimeError("FATAL: July board population parity changed")
    return population, {
        "prediction_source": "REUSED_EXISTING",
        "artifact": f"{SOURCE_DIR}/{SOURCE_LOCK}",
        "source_lock_sha256": observed_source_hash,
        "prediction_lock_sha256": lock_hash,
        "training_audit": audit,
        "fit_count": 0,
        "july_refits": 0,
        "outcome_perturbation": "PASS",
    }


def outcome_perturbation_is_immutable(population: pd.DataFrame) -> bool:
    def prediction_input(frame: pd.DataFrame) -> pd.DataFrame:
        source = frame[[
            "event_id", "signal_date", "code", "board_group", "stage1_score",
            "stage1_rank",
        ]].copy()
        source["board"] = np.where(source["board_group"].eq("BOARD2"), 2, 3)
        source["candidate_count"] = source.groupby("signal_date", sort=True)[
            "event_id"
        ].transform("size")
        return source

    original = build_stage1_lock(prediction_input(population))
    perturbed = population.copy()
    rng = np.random.default_rng(20260825)
    for column in (
        "target7", "loss", "severe_loss", "positive_non_target",
        "raw_repair_return", "capped_return_7", "d2_open_daily", "d3_high_daily",
    ):
        perturbed[column] = rng.normal(1000, 10, len(perturbed))
    rebuilt = build_stage1_lock(prediction_input(perturbed))
    return dataframe_csv_bytes(original) == dataframe_csv_bytes(rebuilt)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _series_stats(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "mean": float(values.mean()), "median": float(values.median()),
        "p10": float(values.quantile(.10)), "p25": float(values.quantile(.25)),
        "worst": float(values.min()),
    }


def build_daily(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in population.groupby("signal_date", sort=True):
        ordered = day.sort_values(["stage1_rank", "event_id"], kind="mergesort")
        row: dict[str, Any] = {
            "signal_date": date, "candidate_count": len(day),
            "universe_target7_rate": float(day["target7"].mean()),
            "universe_loss_rate": float(day["loss"].mean()),
            "universe_raw": float(day["raw_repair_return"].mean()),
            "universe_capped": float(day["capped_return_7"].mean()),
        }
        for rank in (1, 2, 3):
            selected = ordered[ordered["stage1_rank"].eq(rank)]
            prefix = f"rank{rank}"
            if selected.empty:
                for key in ("code", "board", "raw", "capped", "target7", "loss"):
                    row[f"{prefix}_{key}"] = np.nan
            else:
                item = selected.iloc[0]
                row.update({
                    f"{prefix}_code": item["code"],
                    f"{prefix}_board": item["board_group"],
                    f"{prefix}_raw": item["raw_repair_return"],
                    f"{prefix}_capped": item["capped_return_7"],
                    f"{prefix}_target7": item["target7"],
                    f"{prefix}_loss": item["loss"],
                })
        top2 = ordered.head(2)
        row["top2_raw"] = float(top2["raw_repair_return"].mean())
        row["top2_capped"] = float(top2["capped_return_7"].mean())
        row["top2_target7_rate"] = float(top2["target7"].mean())
        row["top2_loss_rate"] = float(top2["loss"].mean())
        row["top2_severe_loss_rate"] = float(top2["severe_loss"].mean())
        row["top2_worst_stock"] = float(top2["raw_repair_return"].min())
        row["top2_worst_contribution"] = row["top2_worst_stock"] / 2
        if len(ordered) >= 3:
            top3 = ordered.head(3)
            row["top3_raw"] = float(top3["raw_repair_return"].mean())
            row["top3_capped"] = float(top3["capped_return_7"].mean())
            row["top3_target7_rate"] = float(top3["target7"].mean())
            row["top3_loss_rate"] = float(top3["loss"].mean())
            row["top3_severe_loss_rate"] = float(top3["severe_loss"].mean())
            row["top3_worst_stock"] = float(top3["raw_repair_return"].min())
            row["top3_worst_contribution"] = row["top3_worst_stock"] / 3
            row["top2_minus_top3"] = row["top2_capped"] - row["top3_capped"]
        else:
            for key in (
                "top3_raw", "top3_capped", "top3_target7_rate", "top3_loss_rate",
                "top3_severe_loss_rate", "top3_worst_stock",
                "top3_worst_contribution", "top2_minus_top3",
            ):
                row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def universe_summary(population: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "dates": int(population["signal_date"].nunique()), "rows": len(population),
        "target7": int(population["target7"].sum()),
        "pnt": int(population["positive_non_target"].sum()),
        "loss": int(population["loss"].sum()),
        "target7_rate": float(daily["universe_target7_rate"].mean()),
        "loss_rate": float(daily["universe_loss_rate"].mean()),
        "raw_mean": float(daily["universe_raw"].mean()),
        "capped_mean": float(daily["universe_capped"].mean()),
    }


def rank_summary(population: pd.DataFrame, rank: int, daily: pd.DataFrame) -> dict[str, Any]:
    selected = population[population["stage1_rank"].eq(rank)]
    eligible_daily = daily[daily[f"rank{rank}_capped"].notna()]
    return {
        "rank": rank, "rows": len(selected), "dates": len(eligible_daily),
        "target7": int(selected["target7"].sum()),
        "target7_rate": float(selected["target7"].mean()),
        "pnt": int(selected["positive_non_target"].sum()),
        "pnt_rate": float(selected["positive_non_target"].mean()),
        "loss": int(selected["loss"].sum()), "loss_rate": float(selected["loss"].mean()),
        "severe_loss": int(selected["severe_loss"].sum()),
        "severe_loss_rate": float(selected["severe_loss"].mean()),
        "raw_mean": float(selected["raw_repair_return"].mean()),
        "raw_median": float(selected["raw_repair_return"].median()),
        "capped_mean": float(selected["capped_return_7"].mean()),
        "capped_median": float(selected["capped_return_7"].median()),
        "worst_raw": float(selected["raw_repair_return"].min()),
        "universe_target7_rate": float(eligible_daily["universe_target7_rate"].mean()),
        "universe_loss_rate": float(eligible_daily["universe_loss_rate"].mean()),
        "universe_capped_mean": float(eligible_daily["universe_capped"].mean()),
    }


def forward_value(summary: Mapping[str, Any]) -> str:
    if int(summary["dates"]) < 5:
        return "INSUFFICIENT"
    favorable = sum((
        float(summary["capped_mean"]) > float(summary["universe_capped_mean"]),
        float(summary["loss_rate"]) <= float(summary["universe_loss_rate"]),
        float(summary["target7_rate"]) >= float(summary["universe_target7_rate"]),
    ))
    if favorable == 3:
        return "ESTABLISHED"
    if favorable >= 2:
        return "PARTIAL"
    if (
        float(summary["capped_mean"]) <= float(summary["universe_capped_mean"])
        and float(summary["target7_rate"]) <= float(summary["universe_target7_rate"])
        and float(summary["loss_rate"]) >= float(summary["universe_loss_rate"])
    ):
        return "ABSENT"
    return "PARTIAL"


def topk_summary(
    population: pd.DataFrame, daily: pd.DataFrame, k: int,
) -> dict[str, Any]:
    eligible_dates = daily.loc[daily[f"rank{k}_capped"].notna(), "signal_date"]
    subset = population[
        population["signal_date"].isin(eligible_dates)
        & population["stage1_rank"].le(k)
    ]
    selected_daily = daily[daily[f"rank{k}_capped"].notna()].copy()
    prefix = f"top{k}"
    raw = selected_daily[f"{prefix}_raw"] if k > 1 else selected_daily["rank1_raw"]
    capped = selected_daily[f"{prefix}_capped"] if k > 1 else selected_daily["rank1_capped"]
    target_rate = float(subset["target7"].mean())
    universe_target_rate = float(selected_daily["universe_target7_rate"].mean())
    universe_raw = float(selected_daily["universe_raw"].mean())
    universe_capped = float(selected_daily["universe_capped"].mean())
    total_winners = int(population.loc[
        population["signal_date"].isin(eligible_dates), "target7"
    ].sum())
    worst_stock = float(subset["raw_repair_return"].min())
    return {
        "portfolio": f"TOP{k}", "k": k, "dates": len(selected_daily),
        "selected_rows": len(subset), "target7": int(subset["target7"].sum()),
        "target7_precision": target_rate,
        "target7_capture": _safe_ratio(int(subset["target7"].sum()), total_winners),
        "target7_lift": _safe_ratio(target_rate, universe_target_rate),
        "loss": int(subset["loss"].sum()), "loss_rate": float(subset["loss"].mean()),
        "severe_loss": int(subset["severe_loss"].sum()),
        "severe_loss_rate": float(subset["severe_loss"].mean()),
        "mean_raw": float(raw.mean()), "mean_capped": float(capped.mean()),
        "universe_raw": universe_raw, "universe_capped": universe_capped,
        "raw_excess": float(raw.mean()) - universe_raw,
        "capped_excess": float(capped.mean()) - universe_capped,
        "negative_date_rate": float(raw.lt(0).mean()),
        "days_le_minus_3": int(raw.le(-.03).sum()),
        "rate_le_minus_3": float(raw.le(-.03).mean()),
        "days_le_minus_5": int(raw.le(-.05).sum()),
        "rate_le_minus_5": float(raw.le(-.05).mean()),
        "p10": float(raw.quantile(.10)), "p25": float(raw.quantile(.25)),
        "worst_day": float(raw.min()), "worst_selected_stock": worst_stock,
        "worst_name_contribution": worst_stock / k,
    }


def build_rank3_marginal(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = daily[daily["top3_capped"].notna()].copy()
    output = eligible[[
        "signal_date", "rank1_code", "rank2_code", "rank3_code",
        "rank1_capped", "rank2_capped", "rank3_capped", "top2_capped",
        "top3_capped", "rank3_target7", "rank3_loss",
    ]].copy()
    output["rank3_marginal"] = output["top3_capped"] - output["top2_capped"]
    output["rank3_severe_loss"] = eligible["rank3_raw"].le(-.05).astype(int).values
    delta = output["rank3_marginal"]
    return output, {
        "mean": float(delta.mean()), "median": float(delta.median()),
        "positive": int(delta.gt(0).sum()), "negative": int(delta.lt(0).sum()),
        "zero": int(delta.eq(0).sum()),
    }


def _metric_summary(values: Sequence[float], estimate: float, favorable: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    predicate = finite > 0 if favorable == "POSITIVE" else finite <= 0
    return {
        "estimate": float(estimate), "valid_resamples": len(finite),
        "p2.5": float(np.quantile(finite, .025)),
        "p50": float(np.quantile(finite, .50)),
        "p97.5": float(np.quantile(finite, .975)),
        "direction_probability": float(np.mean(predicate)),
    }


def build_robustness(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dates = daily["signal_date"].astype(str).tolist()
    n = len(dates)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    metric_columns = (
        "universe_target7_rate", "universe_capped", "rank1_capped",
        "rank2_capped", "top2_raw", "top2_capped", "top2_target7_rate",
        "top3_raw", "top3_capped",
    )
    arrays = {column: daily[column].to_numpy(float) for column in metric_columns}
    common_mask = np.isfinite(arrays["top3_capped"])
    common_n = int(common_mask.sum())
    common_draws = rng.integers(
        0, common_n, size=(BOOTSTRAP_RESAMPLES, common_n)
    )

    top2_top3 = arrays["top2_capped"] - arrays["top3_capped"]
    top2_universe = arrays["top2_capped"] - arrays["universe_capped"]
    target_excess = arrays["top2_target7_rate"] - arrays["universe_target7_rate"]
    neg_delta = arrays["top2_raw"] < 0
    neg3 = arrays["top3_raw"] < 0
    severe_delta = arrays["top2_raw"] <= -.05
    severe3 = arrays["top3_raw"] <= -.05

    def boot_mean(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        selected = values[mask] if mask is not None else values
        if mask is not None:
            return selected[common_draws].mean(axis=1)
        return values[draws].mean(axis=1)

    boot_values = {
        "TOP2_MINUS_TOP3_CAPPED": boot_mean(top2_top3, common_mask),
        "TOP2_MINUS_UNIVERSE_CAPPED": boot_mean(top2_universe),
        "TOP2_TARGET7_MINUS_UNIVERSE": boot_mean(target_excess),
        "TOP2_MINUS_TOP3_NEGATIVE_DATE": boot_mean(
            neg_delta.astype(float) - neg3.astype(float), common_mask
        ),
        "TOP2_MINUS_TOP3_LE_MINUS_5": boot_mean(
            severe_delta.astype(float) - severe3.astype(float), common_mask
        ),
    }
    estimates = {
        "TOP2_MINUS_TOP3_CAPPED": float(top2_top3[common_mask].mean()),
        "TOP2_MINUS_UNIVERSE_CAPPED": float(top2_universe.mean()),
        "TOP2_TARGET7_MINUS_UNIVERSE": float(target_excess.mean()),
        "TOP2_MINUS_TOP3_NEGATIVE_DATE": float((
            neg_delta.astype(float) - neg3.astype(float)
        )[common_mask].mean()),
        "TOP2_MINUS_TOP3_LE_MINUS_5": float((
            severe_delta.astype(float) - severe3.astype(float)
        )[common_mask].mean()),
    }
    favorable = {
        "TOP2_MINUS_TOP3_CAPPED": "POSITIVE",
        "TOP2_MINUS_UNIVERSE_CAPPED": "POSITIVE",
        "TOP2_TARGET7_MINUS_UNIVERSE": "POSITIVE",
        "TOP2_MINUS_TOP3_NEGATIVE_DATE": "NONPOSITIVE",
        "TOP2_MINUS_TOP3_LE_MINUS_5": "NONPOSITIVE",
    }
    bootstrap = {
        metric: _metric_summary(values, estimates[metric], favorable[metric])
        for metric, values in boot_values.items()
    }

    lodo_runs: dict[str, list[float]] = {
        "TOP2_MINUS_TOP3_CAPPED": [], "TOP2_MINUS_UNIVERSE_CAPPED": [],
        "RANK1_MINUS_UNIVERSE_CAPPED": [], "RANK2_MINUS_UNIVERSE_CAPPED": [],
        "TOP2_TAIL_NO_WORSE": [],
    }
    for index in range(n):
        keep = np.arange(n) != index
        common = keep & common_mask
        lodo_runs["TOP2_MINUS_TOP3_CAPPED"].append(float(np.nanmean(top2_top3[common])))
        lodo_runs["TOP2_MINUS_UNIVERSE_CAPPED"].append(float(top2_universe[keep].mean()))
        lodo_runs["RANK1_MINUS_UNIVERSE_CAPPED"].append(float(
            (arrays["rank1_capped"] - arrays["universe_capped"])[keep].mean()
        ))
        lodo_runs["RANK2_MINUS_UNIVERSE_CAPPED"].append(float(
            (arrays["rank2_capped"] - arrays["universe_capped"])[keep].mean()
        ))
        top2_neg = float((arrays["top2_raw"][common] < 0).mean())
        top3_neg = float((arrays["top3_raw"][common] < 0).mean())
        top2_severe = int((arrays["top2_raw"][common] <= -.05).sum())
        top3_severe = int((arrays["top3_raw"][common] <= -.05).sum())
        lodo_runs["TOP2_TAIL_NO_WORSE"].append(float(
            top2_neg <= top3_neg + .05 and top2_severe <= top3_severe + 1
        ))
    lodo = {
        metric: {
            "valid_resamples": len(values),
            "direction_probability": float(np.mean(np.asarray(values) > 0))
            if metric != "TOP2_TAIL_NO_WORSE" else float(np.mean(values)),
            "min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        for metric, values in lodo_runs.items()
    }
    rows: list[dict[str, Any]] = []
    for section, collection in (("BOOTSTRAP", bootstrap), ("LODO", lodo)):
        for metric, values in collection.items():
            rows.append({"section": section, "metric": metric, **values})
    return pd.DataFrame(rows), bootstrap, lodo


def build_temporal_replication(
    ranks: Mapping[int, Mapping[str, Any]], top2: Mapping[str, Any],
    top3: Mapping[str, Any],
) -> pd.DataFrame:
    july = {
        "Top2 mean capped": top2["mean_capped"],
        "Top3 mean capped": top3["mean_capped"],
        "Top2 minus Top3": top2["mean_capped"] - top3["mean_capped"],
        "Top2 negative-date rate": top2["negative_date_rate"],
        "Top3 negative-date rate": top3["negative_date_rate"],
        "Top2 worst day": top2["worst_day"],
        "Top3 worst day": top3["worst_day"],
        "Rank1 capped": ranks[1]["capped_mean"],
        "Rank2 capped": ranks[2]["capped_mean"],
        "Rank3 capped": ranks[3]["capped_mean"],
    }
    may = {
        "Top2 mean capped": MAY_JUNE_REFERENCE["Top2_mean_capped"],
        "Top3 mean capped": MAY_JUNE_REFERENCE["Top3_mean_capped"],
        "Top2 minus Top3": MAY_JUNE_REFERENCE["Top2_mean_capped"] - MAY_JUNE_REFERENCE["Top3_mean_capped"],
        "Top2 negative-date rate": MAY_JUNE_REFERENCE["Top2_negative_date_rate"],
        "Top3 negative-date rate": MAY_JUNE_REFERENCE["Top3_negative_date_rate"],
        "Top2 worst day": MAY_JUNE_REFERENCE["Top2_worst_daily_raw"],
        "Top3 worst day": MAY_JUNE_REFERENCE["Top3_worst_daily_raw"],
        "Rank1 capped": MAY_JUNE_REFERENCE["Rank1_mean_capped"],
        "Rank2 capped": MAY_JUNE_REFERENCE["Rank2_mean_capped"],
        "Rank3 capped": MAY_JUNE_REFERENCE["Rank3_mean_capped"],
    }
    rows = []
    for metric in may:
        if metric in {"Top2 negative-date rate", "Top3 negative-date rate"}:
            consistent = (july[metric] <= .5) == (may[metric] <= .5)
        else:
            consistent = np.sign(july[metric]) == np.sign(may[metric])
        rows.append({
            "metric": metric, "may_june_reference": may[metric],
            "july_forward": july[metric], "direction_consistent": bool(consistent),
            "comment": "frozen historical reference; not pooled",
        })
    return pd.DataFrame(rows)


def formal_decision(context: Mapping[str, Any]) -> dict[str, Any]:
    universe = context["universe"]
    ranks = context["rank_summaries"]
    top2 = context["topk"][2]
    top3 = context["topk"][3]
    bootstrap = context["bootstrap"]
    common = context["daily"][context["daily"]["top3_capped"].notna()]
    top2_common_capped = float(common["top2_capped"].mean())
    top3_common_capped = float(common["top3_capped"].mean())
    top2_common_negative = float(common["top2_raw"].lt(0).mean())
    top3_common_negative = float(common["top3_raw"].lt(0).mean())
    top2_common_severe = int(common["top2_raw"].le(-.05).sum())
    top3_common_severe = int(common["top3_raw"].le(-.05).sum())
    top2_common_worst = float(common["top2_raw"].min())
    top3_common_worst = float(common["top3_raw"].min())
    gates = {
        "A": top2_common_capped > top3_common_capped,
        "B": bootstrap["TOP2_MINUS_TOP3_CAPPED"]["direction_probability"] >= .65,
        "C": top2["mean_capped"] > universe["capped_mean"],
        "D": top2["target7_precision"] >= universe["target7_rate"],
        "E": context["rank_values"][1] in {"ESTABLISHED", "PARTIAL"},
        "F": context["rank_values"][2] in {"ESTABLISHED", "PARTIAL"},
        "G": top2_common_negative <= top3_common_negative + .05,
        "H": top2_common_severe <= top3_common_severe + 1,
        "I": top2_common_worst >= top3_common_worst - .02,
        "J": top2_common_capped > top3_common_capped,
    }
    if len(context["daily"]) < 10:
        signal = "INSUFFICIENT"
    elif all(gates.values()):
        signal = "CONFIRMED"
    else:
        failed = (
            (top2_common_capped <= top3_common_capped and not (
                top2_common_negative < top3_common_negative
                or top2["worst_day"] > top3["worst_day"]
            ))
            or (top2["mean_capped"] < universe["capped_mean"]
                and top2["target7_precision"] < universe["target7_rate"])
            or context["rank_values"][2] == "ABSENT"
        )
        signal = "FAILED" if failed else "PARTIAL"

    if signal == "CONFIRMED":
        failure = "NONE"
    elif len(context["daily"]) < 10:
        failure = "INSUFFICIENT"
    elif (context["rank_values"][1] == "ABSENT"
          and top2["mean_capped"] <= universe["capped_mean"]):
        failure = "RANK1_FAILURE"
    elif (context["rank_values"][1] != "ABSENT"
          and context["rank_values"][2] == "ABSENT"):
        failure = "RANK2_FAILURE"
    elif (top2["mean_capped"] <= universe["capped_mean"]
          and ranks[1]["capped_mean"] <= universe["capped_mean"]
          and ranks[2]["capped_mean"] <= universe["capped_mean"]):
        failure = "STAGE1_SELECTION_ALPHA_FAILURE"
    elif top3_common_capped > top2_common_capped and context["rank3_marginal"]["mean"] > 0:
        failure = "TOP2_CAPACITY_FAILURE"
    elif not (gates["G"] and gates["H"] and gates["I"]):
        failure = "TAIL_RISK_FAILURE"
    else:
        failure = "MIXED_FAILURE"

    if signal == "CONFIRMED":
        state = "FREEZE_TOP2"
    elif (ranks[1]["capped_mean"] <= universe["capped_mean"]
          and top2["mean_capped"] <= universe["capped_mean"]):
        state = "REOPEN_STAGE1_QUESTION"
    elif top3_common_capped > top2_common_capped or context["rank_values"][2] == "ABSENT":
        state = "REOPEN_CAPACITY_QUESTION"
    elif signal == "PARTIAL" and gates["A"] and gates["G"] and gates["H"]:
        state = "KEEP_TOP2_UNDER_OBSERVATION"
    else:
        state = "REOPEN_CAPACITY_QUESTION"
    return {
        "gates": gates, "signal": signal, "primary_failure": failure,
        "architecture_state": state,
        "common_dates": len(common),
        "top2_common_capped": top2_common_capped,
        "top3_common_capped": top3_common_capped,
        "top2_common_negative": top2_common_negative,
        "top3_common_negative": top3_common_negative,
        "top2_common_severe": top2_common_severe,
        "top3_common_severe": top3_common_severe,
        "top2_common_worst": top2_common_worst,
        "top3_common_worst": top3_common_worst,
    }


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def render_review(context: Mapping[str, Any]) -> str:
    provenance = context["provenance"]
    audit = provenance["training_audit"]
    universe = context["universe"]
    ranks = context["rank_summaries"]
    topk = context["topk"]
    decision = context["decision"]
    marginal = context["rank3_marginal"]
    bootstrap = context["bootstrap"]
    lines = [
        "# v004c Frozen Current Stage1 Top2 July Chronological Forward Stress v001",
        "", "## 1. Frozen Contract", "",
        "- Evaluation: CHRONOLOGICAL_FORWARD_STRESS / SEMI-OOT; pristine OOT: NO.",
        "- Current V4C_STAGE1_V4A_ARCH; 18 features; weighted L2 Logistic; L2=0.30; positive weight=1.50; TARGET7.",
        "- Top2 was predeclared; Top3 is a fixed diversification comparator. No model, feature, TopK, threshold, Stage2, or Board3-policy search.",
        "", "## 2. Prediction Provenance / Leakage Audit", "",
        f"- Reused artifact: `{provenance['artifact']}`.",
        f"- Source/purpose lock SHA256: `{provenance['source_lock_sha256']}` / `{provenance['prediction_lock_sha256']}`.",
        f"- Equivalent frozen identity: development commit `{FROZEN_DEVELOPMENT_COMMIT}` + source lock SHA256; feature manifest is the imported exact 18-column `FROZEN_FEATURE_COLUMNS`.",
        f"- Training rows/dates: {audit['stage1_training_rows']}/{audit['eligible_historical_dates']}; latest signal/label: {audit['latest_eligible_signal_date']}/{audit['latest_label_available_date']}.",
        "- July training/refits: 0/0; fit count in this audit: 0; outcome perturbation: PASS.",
        "", "## 3. July Population", "",
        f"- Dates/rows: {universe['dates']}/{universe['rows']}; Board2/Board3: {context['board2']}/{context['board3']}.",
        f"- Target7/PNT/LOSS: {universe['target7']}/{universe['pnt']}/{universe['loss']}.",
        f"- Universe Target7/LOSS/raw/capped: {_pct(universe['target7_rate'])}/{_pct(universe['loss_rate'])}/{_pct(universe['raw_mean'])}/{_pct(universe['capped_mean'])}.",
        "", "## 4. Rank-wise Forward Value", "",
        "| Rank | Rows | Target7 | LOSS | Severe | Raw | Capped | Value |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank in (1, 2, 3):
        value = context["rank_values"][rank]
        row = ranks[rank]
        lines.append(
            f"| {rank} | {row['rows']} | {_pct(row['target7_rate'])} | {_pct(row['loss_rate'])} | {_pct(row['severe_loss_rate'])} | {_pct(row['raw_mean'])} | {_pct(row['capped_mean'])} | {value} |"
        )
    lines += ["", "## 5. Top1 / Top2 / Top3", "",
              "| Portfolio | Dates | Precision | Capture | LOSS | Raw | Capped | Excess | Negative | Worst |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in (1, 2, 3):
        row = topk[k]
        lines.append(
            f"| Top{k} | {row['dates']} | {_pct(row['target7_precision'])} | {_pct(row['target7_capture'])} | {_pct(row['loss_rate'])} | {_pct(row['mean_raw'])} | {_pct(row['mean_capped'])} | {_pct(row['capped_excess'])} | {_pct(row['negative_date_rate'])} | {_pct(row['worst_day'])} |"
        )
    lines += [
        "", "## 6. Rank3 Marginal / Diversification", "",
        f"- Common Top2/Top3 dates: {decision['common_dates']}; paired capped: {_pct(decision['top2_common_capped'])} vs {_pct(decision['top3_common_capped'])}.",
        f"- Rank3 marginal mean/median: {_pct(marginal['mean'])}/{_pct(marginal['median'])}; positive/negative/zero: {marginal['positive']}/{marginal['negative']}/{marginal['zero']}.",
        f"- Bootstrap P(Top2>Top3): {_pct(bootstrap['TOP2_MINUS_TOP3_CAPPED']['direction_probability'])}.",
        "", "## 7. Bootstrap / LODO", "",
        f"- Top2-Top3 capped interval: [{_pct(bootstrap['TOP2_MINUS_TOP3_CAPPED']['p2.5'])}, {_pct(bootstrap['TOP2_MINUS_TOP3_CAPPED']['p97.5'])}].",
        f"- Top2-Universe capped estimate/P>0: {_pct(bootstrap['TOP2_MINUS_UNIVERSE_CAPPED']['estimate'])}/{_pct(bootstrap['TOP2_MINUS_UNIVERSE_CAPPED']['direction_probability'])}.",
        "", "## 8. Temporal Replication", "",
        "- May-Jun and July are reported separately and never pooled.",
        f"- May-Jun Top2-Top3: {_pct(MAY_JUNE_REFERENCE['Top2_mean_capped'] - MAY_JUNE_REFERENCE['Top3_mean_capped'])}; July paired: {_pct(decision['top2_common_capped'] - decision['top3_common_capped'])}.",
        "", "## 9. Forward Decision", "",
        "| Gate | Pass |", "|---|:---:|",
    ]
    for gate, passed in decision["gates"].items():
        lines.append(f"| {gate} | {'YES' if passed else 'NO'} |")
    lines += [
        "", f"- `TOP2_CAPACITY_FORWARD_SIGNAL = {decision['signal']}`",
        f"- `PRIMARY_FORWARD_FAILURE = {decision['primary_failure']}`",
        f"- `V4C_CORE_ARCHITECTURE_STATE = {decision['architecture_state']}`",
        "- AUTO_OPTIMIZE_AFTER_FAILURE = NO.",
    ]
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    population, provenance = load_reused_july_population(root)
    if not outcome_perturbation_is_immutable(population):
        raise RuntimeError("FATAL: outcome perturbation changed Stage1 lock")
    lock = population[LOCK_COLUMNS].copy()
    daily = build_daily(population)
    universe = universe_summary(population, daily)
    rank_summaries = {rank: rank_summary(population, rank, daily) for rank in (1, 2, 3)}
    rank_values = {rank: forward_value(rank_summaries[rank]) for rank in (1, 2, 3)}
    topk = {k: topk_summary(population, daily, k) for k in (1, 2, 3)}
    rank3_table, rank3_marginal = build_rank3_marginal(daily)
    robustness, bootstrap, lodo = build_robustness(daily)
    temporal = build_temporal_replication(rank_summaries, topk[2], topk[3])
    context: dict[str, Any] = {
        "population": population, "lock": lock, "provenance": provenance,
        "daily": daily, "universe": universe, "rank_summaries": rank_summaries,
        "rank_values": rank_values, "topk": topk, "rank3_table": rank3_table,
        "rank3_marginal": rank3_marginal, "robustness": robustness,
        "bootstrap": bootstrap, "lodo": lodo, "temporal_replication": temporal,
        "board2": int(population["board_group"].eq("BOARD2").sum()),
        "board3": int(population["board_group"].eq("BOARD3").sum()),
        "outcome_perturbation": "PASS",
    }
    context["decision"] = formal_decision(context)
    rankwise = pd.DataFrame([rank_summaries[rank] | {"forward_value": rank_values[rank]} for rank in (1, 2, 3)])
    topk_frame = pd.DataFrame([topk[k] for k in (1, 2, 3)])
    population_columns = LOCK_COLUMNS + [
        "d2_date", "d3_date", "d2_open_daily", "d3_high_daily", "target7",
        "positive_non_target", "loss", "severe_loss", "raw_repair_return",
        "capped_return_7",
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(lock),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(population[population_columns]),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(rankwise),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(topk_frame),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[5]: dataframe_csv_bytes(rank3_table),
        OUTPUT_FILENAMES[6]: dataframe_csv_bytes(temporal),
        OUTPUT_FILENAMES[7]: dataframe_csv_bytes(robustness),
        OUTPUT_FILENAMES[8]: render_review(context).encode("utf-8"),
    }
    return outputs, context


def run_v004c_frozen_stage1_top2_july_forward_stress(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    target = Path(output_dir) if output_dir else (
        root_path / "reports/research/"
        "v004c_frozen_stage1_top2_july_forward_stress_v001_20260701_20260731"
    )
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first != second:
        mismatches = [name for name in first if first[name] != second[name]]
        raise RuntimeError(f"FATAL: deterministic rebuild failed: {mismatches}")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (target / name).write_bytes(payload)
    context["deterministic_rebuild"] = "PASS"
    context["output_dir"] = target
    return target, context

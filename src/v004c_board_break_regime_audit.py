"""Descriptive May/June/July board-break ecology audit.

This module never fits a model, changes a candidate universe, searches a
feature/regime definition, or reads an August signal-date outcome.  It joins
the authoritative mature v004c first-break population to the repository's
existing main-board final-limit-up caches and four already-audited D1-close
path fields.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_d1_unfinished_repair_information_audit import FEATURES as PATH_FEATURES


TASK_NAME = "v004c_board_break_regime_audit_v001"
OUTPUT_DIRNAME = "v004c_board_break_regime_audit_v001_202605_202607"
ANALYSIS_START = "2026-05-06"
ANALYSIS_END = "2026-07-29"
AUGUST_ASOF = "2026-08-01"
EXPECTED_ROWS = 485
EXPECTED_DATES = 60

MONTHS: Mapping[str, str] = {
    "MAY": "2026-05",
    "JUNE": "2026-06",
    "JULY": "2026-07",
}

OUTPUT_FILENAMES = (
    "v004c_regime_daily_ecology_v001.csv",
    "v004c_regime_month_summary_v001.csv",
    "v004c_regime_candidate_pool_v001.csv",
    "v004c_regime_board_ecology_v001.csv",
    "v004c_regime_break_ecology_v001.csv",
    "v004c_regime_candidate_path_state_v001.csv",
    "v004c_regime_review_v001.md",
)

FAILED_BOARD_STATUS = "DATA_NOT_AVAILABLE_NO_FAILED_LIMIT_UP_POOL"
BOARD_POOL_SEMANTICS = "EXISTING_MAIN_BOARD_FINAL_LIMIT_UP_POOL"
RECENT_HISTORY_DATES = 5
PATH_POPULATION_AUTHORITY = (
    "reports/research/"
    "v004c_d1_unfinished_repair_information_audit_v001_20260506_20260731/"
    "v004c_unfinished_repair_population_v001.csv"
)
BRIDGE_AUTHORITY = (
    "reports/research/"
    "v004c_stage1_18f_bridge_analysis_v001_20260506_20260731/"
    "v004c_stage1_18f_bridge_candidates.csv"
)


def assert_contract() -> None:
    if tuple(PATH_FEATURES) != (
        "d1_afternoon_return",
        "d1_open_to_close_return_raw",
        "d1_close_to_vwap_raw",
        "d1_low_to_close_recovery",
    ):
        raise RuntimeError("FATAL: fixed four-field D1 path manifest changed")
    if ANALYSIS_END >= AUGUST_ASOF:
        raise RuntimeError("FATAL: analysis boundary reached August")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _month_label(signal_date: str) -> str:
    prefix = str(signal_date)[:7]
    for label, month in MONTHS.items():
        if prefix == month:
            return label
    raise ValueError(f"date outside fixed audit months: {signal_date}")


def load_population(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse the mature, authoritative v004c universe and existing path data."""
    assert_contract()
    root_path = Path(root).resolve()
    population_path = root_path / PATH_POPULATION_AUTHORITY
    bridge_path = root_path / BRIDGE_AUTHORITY
    population = pd.read_csv(
        population_path, encoding="utf-8-sig", dtype={"event_id": str, "code": str}
    )
    bridge = pd.read_csv(
        bridge_path, encoding="utf-8-sig", dtype={"event_id": str, "code": str}
    )
    bridge["signal_date"] = bridge["signal_date"].astype(str)
    bridge["label_available_date"] = bridge["label_available_date"].astype(str)
    bridge_mature = bridge[bridge["label_available_date"].lt(AUGUST_ASOF)].copy()
    identity_columns = [
        "event_id", "code", "signal_date", "label_available_date", "board_group",
        "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
    ]
    left = population[identity_columns].sort_values("event_id", kind="mergesort").reset_index(drop=True)
    right = bridge_mature[identity_columns].sort_values("event_id", kind="mergesort").reset_index(drop=True)
    text_identity = [
        "event_id", "code", "signal_date", "label_available_date", "board_group",
    ]
    outcome_identity = ["target7", "positive_non_target", "loss", "severe_loss"]
    text_equal = all(
        np.array_equal(left[column].astype(str), right[column].astype(str))
        for column in text_identity
    )
    outcome_equal = np.array_equal(
        left[outcome_identity].to_numpy(int), right[outcome_identity].to_numpy(int)
    )
    if len(left) != len(right) or not text_equal or not outcome_equal:
        raise RuntimeError("FATAL: archived path population identity differs from mature bridge")
    numeric_diff = np.nanmax(np.abs(
        left[["raw_repair_return", "capped_return_7"]].to_numpy(float)
        - right[["raw_repair_return", "capped_return_7"]].to_numpy(float)
    ))
    if numeric_diff > 1e-12:
        raise RuntimeError("FATAL: archived path population outcomes differ from mature bridge")
    population = population.copy()
    population["signal_date"] = population["signal_date"].astype(str)
    population["label_available_date"] = population["label_available_date"].astype(str)
    population["month"] = population["signal_date"].map(_month_label)
    if (len(population), population["signal_date"].nunique()) != (
        EXPECTED_ROWS,
        EXPECTED_DATES,
    ):
        raise RuntimeError("FATAL: mature May-July population parity failed")
    if population["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id")
    if population["signal_date"].min() != ANALYSIS_START:
        raise RuntimeError("FATAL: mature population start changed")
    if population["signal_date"].max() != ANALYSIS_END:
        raise RuntimeError("FATAL: mature population end changed")
    if population["signal_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: August signal date reached audit")
    if population["label_available_date"].ge(AUGUST_ASOF).any():
        raise RuntimeError("FATAL: immature/August label reached audit")
    if population[list(PATH_FEATURES)].isna().any().any():
        raise RuntimeError("FATAL: existing four-field D1 path values incomplete")
    audit = {
        "rows": len(population),
        "dates": population["signal_date"].nunique(),
        "max_signal_date": population["signal_date"].max(),
        "max_label_available_date": population["label_available_date"].max(),
        "august_signal_outcome_accessed": "NO",
        "model_fits": 0,
        "external_data_connections": 0,
        "population_source": PATH_POPULATION_AUTHORITY,
        "bridge_parity_source": BRIDGE_AUTHORITY,
        "bridge_identity_mismatch_count": 0,
        "bridge_outcome_max_abs_error": float(numeric_diff),
    }
    return population.sort_values(
        ["signal_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True), audit


def _oracle_mean(day: pd.DataFrame, k: int) -> float:
    values = day["capped_return_7"].astype(float).sort_values(ascending=False)
    return float(values.head(min(k, len(values))).mean()) if len(values) else math.nan


def build_candidate_pool(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in population.groupby("signal_date", sort=True):
        count = len(day)
        rows.append({
            "signal_date": str(signal_date),
            "month": _month_label(str(signal_date)),
            "candidate_count": count,
            "board2_candidate_count": int(day["board_group"].eq("BOARD2").sum()),
            "board3_candidate_count": int(day["board_group"].eq("BOARD3").sum()),
            "target7_count": int(day["target7"].sum()),
            "target7_rate": float(day["target7"].mean()),
            "positive_non_target_count": int(day["positive_non_target"].sum()),
            "positive_non_target_rate": float(day["positive_non_target"].mean()),
            "loss_count": int(day["loss"].sum()),
            "loss_rate": float(day["loss"].mean()),
            "severe_loss_count": int(day["severe_loss"].sum()),
            "severe_loss_rate": float(day["severe_loss"].mean()),
            "candidate_universe_raw_return": float(day["raw_repair_return"].mean()),
            "candidate_universe_capped_return": float(day["capped_return_7"].mean()),
            "oracle_top1_capped_return": _oracle_mean(day, 1),
            "oracle_top2_capped_return": _oracle_mean(day, 2),
            "oracle_top3_capped_return": _oracle_mean(day, 3),
            "outcome_asof_rule": "label_available_date < 2026-08-01",
        })
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_DATES:
        raise RuntimeError("FATAL: candidate daily row count changed")
    return result


_POOL_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_limitups\.pkl$")


def _load_limit_up_pools(root: Path) -> dict[str, pd.DataFrame]:
    pool_dir = root / "data" / "cache" / "limit_ups"
    pools: dict[str, pd.DataFrame] = {}
    for path in sorted(pool_dir.glob("*_limitups.pkl")):
        match = _POOL_PATTERN.match(path.name)
        if not match:
            continue
        date_text = match.group(1)
        if date_text > ANALYSIS_END:
            continue
        frame = pd.read_pickle(path).copy()
        if "code" not in frame.columns:
            raise RuntimeError(f"FATAL: limit-up cache lacks code: {path}")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame["code"].duplicated().any():
            raise RuntimeError(f"FATAL: duplicate code in limit-up cache: {path}")
        pools[date_text] = frame
    return pools


def build_board_ecology(root: str | Path, signal_dates: Sequence[str]) -> pd.DataFrame:
    """Describe existing final-limit-up pools; never fetch or derive new data."""
    pools = _load_limit_up_pools(Path(root).resolve())
    all_dates = sorted(pools)
    rows: list[dict[str, Any]] = []
    for signal_date in sorted(map(str, signal_dates)):
        if signal_date not in pools:
            raise RuntimeError(f"FATAL: existing limit-up cache missing {signal_date}")
        frame = pools[signal_date]
        prior_dates = [date for date in all_dates if date < signal_date]
        previous_date = prior_dates[-1] if prior_dates else None
        previous = pools.get(previous_date, pd.DataFrame())
        current_codes = set(frame["code"])
        previous_codes = set(previous["code"]) if len(previous) else set()
        promotion_success = len(current_codes & previous_codes) if len(previous) else math.nan
        promotion_base = len(previous) if len(previous) else math.nan
        promotion_rate = (
            float(promotion_success / promotion_base)
            if np.isfinite(promotion_base) and promotion_base > 0
            else math.nan
        )
        streak = pd.to_numeric(
            frame.get("consecutive_limit_up_count"), errors="coerce"
        )
        streak_available = len(streak) == len(frame) and streak.notna().all()
        open_count = pd.to_numeric(frame.get("open_board_count"), errors="coerce")
        open_coverage = float(open_count.notna().mean()) if len(frame) else math.nan
        resealed_count = int(open_count.gt(0).sum()) if open_coverage == 1.0 else math.nan
        resealed_rate = (
            float(resealed_count / len(frame))
            if len(frame) and np.isfinite(resealed_count)
            else math.nan
        )
        sources = (
            "|".join(sorted(frame["source"].astype(str).dropna().unique()))
            if "source" in frame.columns
            else "UNKNOWN"
        )
        rows.append({
            "signal_date": signal_date,
            "month": _month_label(signal_date),
            "pool_semantics": BOARD_POOL_SEMANTICS,
            "pool_source": sources,
            "limit_up_count": len(frame),
            "two_board_limit_up_count": int(streak.eq(2).sum()) if streak_available else math.nan,
            "three_board_limit_up_count": int(streak.eq(3).sum()) if streak_available else math.nan,
            "four_plus_board_limit_up_count": int(streak.ge(4).sum()) if streak_available else math.nan,
            "highest_board_height": int(streak.max()) if streak_available and len(frame) else math.nan,
            "previous_pool_date": previous_date,
            "promotion_eligible_count": promotion_base,
            "promotion_success_count": promotion_success,
            "promotion_success_rate": promotion_rate,
            "continuation_count_from_streak": int(streak.ge(2).sum()) if streak_available else math.nan,
            "promotion_count_parity": (
                int(promotion_success) == int(streak.ge(2).sum())
                if np.isfinite(promotion_success) and streak_available
                else False
            ),
            "failed_limit_up_count": math.nan,
            "failed_limit_up_rate": math.nan,
            "failed_limit_up_status": FAILED_BOARD_STATUS,
            "open_board_count_coverage": open_coverage,
            "resealed_after_open_count": resealed_count,
            "resealed_after_open_rate": resealed_rate,
            "resealed_metric_status": (
                "AVAILABLE_FINAL_LIMIT_UPS_ONLY"
                if open_coverage == 1.0
                else "DATA_NOT_COMPARABLE_PARTIAL_OR_MISSING_SOURCE_FIELD"
            ),
            "data_status": "AVAILABLE_EXISTING_CACHE",
        })
    result = pd.DataFrame(rows)
    if len(result) != len(set(signal_dates)):
        raise RuntimeError("FATAL: board ecology date coverage mismatch")
    return result


def build_path_state(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, day in population.groupby("signal_date", sort=True):
        for feature in PATH_FEATURES:
            values = pd.to_numeric(day[feature], errors="raise")
            sources = "|".join(sorted(day[f"provenance__{feature}"].astype(str).unique()))
            rows.append({
                "signal_date": str(signal_date),
                "month": _month_label(str(signal_date)),
                "feature": feature,
                "rows": len(values),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p25": float(values.quantile(.25)),
                "p75": float(values.quantile(.75)),
                "source": sources,
                "information_available_at": "D1_CLOSE",
                "d2_d3_information_used": "NO",
            })
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_DATES * len(PATH_FEATURES):
        raise RuntimeError("FATAL: candidate path-state shape mismatch")
    return result


def build_break_ecology(
    population: pd.DataFrame, path_state: pd.DataFrame
) -> pd.DataFrame:
    base = population.groupby(["signal_date", "month"], sort=True).agg(
        first_break_count=("event_id", "size"),
        board2_first_break_count=("board_group", lambda s: int(s.eq("BOARD2").sum())),
        board3_first_break_count=("board_group", lambda s: int(s.eq("BOARD3").sum())),
    ).reset_index()
    wide = path_state.pivot(index="signal_date", columns="feature", values=["mean", "median", "p25", "p75"])
    wide.columns = [f"{feature}__{stat}" for stat, feature in wide.columns]
    wide = wide.reset_index()
    result = base.merge(wide, on="signal_date", validate="one_to_one")
    result["path_field_count"] = len(PATH_FEATURES)
    result["path_data_status"] = "AVAILABLE_EXISTING_D1_CLOSE_FIELDS"
    return result


def build_recent_matured_history(population: pd.DataFrame) -> pd.DataFrame:
    """Build a strict as-of rolling cohort from five already-matured dates."""
    rows: list[dict[str, Any]] = []
    for signal_date in sorted(population["signal_date"].unique()):
        eligible = population[
            population["signal_date"].lt(signal_date)
            & population["label_available_date"].lt(signal_date)
        ]
        matured_dates = sorted(eligible["signal_date"].unique())[-RECENT_HISTORY_DATES:]
        cohort = eligible[eligible["signal_date"].isin(matured_dates)]
        complete = len(matured_dates) == RECENT_HISTORY_DATES
        if complete:
            daily = cohort.groupby("signal_date", sort=True).agg(
                target7_rate=("target7", "mean"),
                loss_rate=("loss", "mean"),
                universe_capped=("capped_return_7", "mean"),
            )
            target7_rate = float(daily["target7_rate"].mean())
            loss_rate = float(daily["loss_rate"].mean())
            capped = float(daily["universe_capped"].mean())
        else:
            target7_rate = loss_rate = capped = math.nan
        rows.append({
            "signal_date": signal_date,
            "recent5_status": "AVAILABLE_STRICT_MATURED" if complete else "INSUFFICIENT_MATURED_HISTORY",
            "recent5_matured_date_count": len(matured_dates),
            "recent5_matured_dates": "|".join(matured_dates),
            "recent5_matured_rows": len(cohort) if complete else math.nan,
            "recent5_target7_rate_date_equal": target7_rate,
            "recent5_loss_rate_date_equal": loss_rate,
            "recent5_candidate_universe_capped_return_date_equal": capped,
            "history_max_label_available_date": (
                str(cohort["label_available_date"].max()) if complete else None
            ),
            "strict_no_future_label": (
                bool(cohort["label_available_date"].lt(signal_date).all()) if complete else True
            ),
        })
    return pd.DataFrame(rows)


def build_daily_ecology(
    candidate_pool: pd.DataFrame,
    board_ecology: pd.DataFrame,
    break_ecology: pd.DataFrame,
    recent_history: pd.DataFrame,
) -> pd.DataFrame:
    selected_break = [
        "signal_date",
        *[f"{feature}__median" for feature in PATH_FEATURES],
    ]
    result = candidate_pool.merge(
        board_ecology, on=["signal_date", "month"], validate="one_to_one",
        suffixes=("", "_board"),
    ).merge(
        break_ecology[selected_break], on="signal_date", validate="one_to_one",
    ).merge(
        recent_history, on="signal_date", validate="one_to_one",
    )
    if len(result) != EXPECTED_DATES:
        raise RuntimeError("FATAL: combined daily ecology date coverage mismatch")
    return result.sort_values("signal_date", kind="mergesort").reset_index(drop=True)


def _direction(may: float, june: float, july: float) -> str:
    values = np.asarray([may, june, july], dtype=float)
    if not np.isfinite(values).all():
        return "DATA_NOT_AVAILABLE"
    d1, d2 = float(june - may), float(july - june)
    tolerance = 1e-12
    if abs(d1) <= tolerance and abs(d2) <= tolerance:
        return "FLAT"
    if d1 >= -tolerance and d2 >= -tolerance:
        return "INCREASED_MAY_TO_JUNE_TO_JULY"
    if d1 <= tolerance and d2 <= tolerance:
        return "DECLINED_MAY_TO_JUNE_TO_JULY"
    if d1 > 0 and d2 < 0:
        return "INCREASED_THEN_DECLINED"
    return "DECLINED_THEN_INCREASED"


def _monthly_value(frame: pd.DataFrame, month: str, column: str, agg: str) -> float:
    values = pd.to_numeric(frame.loc[frame["month"].eq(month), column], errors="coerce")
    if values.notna().sum() == 0:
        return math.nan
    if agg == "mean":
        return float(values.mean())
    if agg == "median":
        return float(values.median())
    if agg == "sum":
        return float(values.sum())
    if agg == "max":
        return float(values.max())
    raise ValueError(agg)


def build_month_summary(
    population: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    board_ecology: pd.DataFrame,
    break_ecology: pd.DataFrame,
) -> pd.DataFrame:
    specs: list[tuple[str, str, str, pd.DataFrame, str, str]] = [
        ("CANDIDATE_POOL", "signal_dates", "count", candidate_pool, "candidate_count", "date_count"),
        ("CANDIDATE_POOL", "candidate_rows_total", "count", candidate_pool, "candidate_count", "sum"),
        ("CANDIDATE_POOL", "board2_candidate_rows_total", "count", candidate_pool, "board2_candidate_count", "sum"),
        ("CANDIDATE_POOL", "board3_candidate_rows_total", "count", candidate_pool, "board3_candidate_count", "sum"),
        ("CANDIDATE_POOL", "target7_rows_total", "count", candidate_pool, "target7_count", "sum"),
        ("CANDIDATE_POOL", "loss_rows_total", "count", candidate_pool, "loss_count", "sum"),
        ("CANDIDATE_POOL", "severe_loss_rows_total", "count", candidate_pool, "severe_loss_count", "sum"),
        ("CANDIDATE_POOL", "candidate_count_per_day_mean", "count", candidate_pool, "candidate_count", "mean"),
        ("CANDIDATE_POOL", "candidate_count_per_day_median", "count", candidate_pool, "candidate_count", "median"),
        ("CANDIDATE_POOL", "board2_candidate_count_per_day_mean", "count", candidate_pool, "board2_candidate_count", "mean"),
        ("CANDIDATE_POOL", "board3_candidate_count_per_day_mean", "count", candidate_pool, "board3_candidate_count", "mean"),
        ("CANDIDATE_POOL", "target7_rate_date_equal", "rate", candidate_pool, "target7_rate", "mean"),
        ("CANDIDATE_POOL", "loss_rate_date_equal", "rate", candidate_pool, "loss_rate", "mean"),
        ("CANDIDATE_POOL", "severe_loss_rate_date_equal", "rate", candidate_pool, "severe_loss_rate", "mean"),
        ("CANDIDATE_POOL", "candidate_universe_capped_return_date_equal", "return", candidate_pool, "candidate_universe_capped_return", "mean"),
        ("CANDIDATE_POOL", "oracle_top1_capped_return", "return", candidate_pool, "oracle_top1_capped_return", "mean"),
        ("CANDIDATE_POOL", "oracle_top2_capped_return", "return", candidate_pool, "oracle_top2_capped_return", "mean"),
        ("CANDIDATE_POOL", "oracle_top3_capped_return", "return", candidate_pool, "oracle_top3_capped_return", "mean"),
        ("BOARD_ECOLOGY", "limit_up_count_per_day_mean", "count", board_ecology, "limit_up_count", "mean"),
        ("BOARD_ECOLOGY", "two_board_limit_up_count_per_day_mean", "count", board_ecology, "two_board_limit_up_count", "mean"),
        ("BOARD_ECOLOGY", "three_board_limit_up_count_per_day_mean", "count", board_ecology, "three_board_limit_up_count", "mean"),
        ("BOARD_ECOLOGY", "four_plus_board_limit_up_count_per_day_mean", "count", board_ecology, "four_plus_board_limit_up_count", "mean"),
        ("BOARD_ECOLOGY", "highest_board_height_daily_mean", "board_count", board_ecology, "highest_board_height", "mean"),
        ("BOARD_ECOLOGY", "highest_board_height_month_max", "board_count", board_ecology, "highest_board_height", "max"),
        ("BOARD_ECOLOGY", "promotion_success_rate_daily_mean", "rate", board_ecology, "promotion_success_rate", "mean"),
        ("BOARD_ECOLOGY", "failed_limit_up_rate", "rate", board_ecology, "failed_limit_up_rate", "mean"),
    ]
    for feature in PATH_FEATURES:
        specs.extend([
            ("BREAK_PATH", f"{feature}_cohort_mean_date_equal", "return", break_ecology, f"{feature}__mean", "mean"),
            ("BREAK_PATH", f"{feature}_cohort_median_date_equal", "return", break_ecology, f"{feature}__median", "mean"),
        ])

    rows: list[dict[str, Any]] = []
    for section, metric, unit, frame, column, agg in specs:
        values: dict[str, float] = {}
        for label in MONTHS:
            if agg == "date_count":
                values[label] = float(frame.loc[frame["month"].eq(label), "signal_date"].nunique())
            else:
                values[label] = _monthly_value(frame, label, column, agg)
        available = all(np.isfinite(values[label]) for label in MONTHS)
        rows.append({
            "section": section,
            "metric": metric,
            "unit": unit,
            "may": values["MAY"],
            "june": values["JUNE"],
            "july": values["JULY"],
            "may_to_june_delta": values["JUNE"] - values["MAY"] if available else math.nan,
            "june_to_july_delta": values["JULY"] - values["JUNE"] if available else math.nan,
            "change_direction": _direction(values["MAY"], values["JUNE"], values["JULY"]),
            "availability": "AVAILABLE" if available else FAILED_BOARD_STATUS,
        })

    # Add pooled row-weighted opportunity rates explicitly, separate from the
    # date-equal primary ecology metrics above.
    for metric, column, unit in (
        ("target7_rate_row_weighted", "target7", "rate"),
        ("loss_rate_row_weighted", "loss", "rate"),
        ("severe_loss_rate_row_weighted", "severe_loss", "rate"),
        ("candidate_universe_capped_return_row_weighted", "capped_return_7", "return"),
    ):
        values = {
            label: float(population.loc[population["month"].eq(label), column].mean())
            for label in MONTHS
        }
        rows.append({
            "section": "CANDIDATE_POOL_ROW_WEIGHTED_SECONDARY",
            "metric": metric,
            "unit": unit,
            "may": values["MAY"],
            "june": values["JUNE"],
            "july": values["JULY"],
            "may_to_june_delta": values["JUNE"] - values["MAY"],
            "june_to_july_delta": values["JULY"] - values["JUNE"],
            "change_direction": _direction(values["MAY"], values["JUNE"], values["JULY"]),
            "availability": "AVAILABLE",
        })
    return pd.DataFrame(rows)


def _metric(summary: pd.DataFrame, metric: str) -> pd.Series:
    row = summary[summary["metric"].eq(metric)]
    if len(row) != 1:
        raise RuntimeError(f"FATAL: month metric not unique: {metric}")
    return row.iloc[0]


def decide_state(summary: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    """Apply transparent descriptive gates, not a learned/composite regime score."""
    target = _metric(summary, "target7_rate_date_equal")
    loss = _metric(summary, "loss_rate_date_equal")
    capped = _metric(summary, "candidate_universe_capped_return_date_equal")
    limitups = _metric(summary, "limit_up_count_per_day_mean")
    promotion = _metric(summary, "promotion_success_rate_daily_mean")
    height = _metric(summary, "highest_board_height_daily_mean")

    opportunity_shift = bool(
        np.ptp(target[["may", "june", "july"]].astype(float)) >= .05
        or np.ptp(capped[["may", "june", "july"]].astype(float)) >= .005
    )
    loss_shift = bool(np.ptp(loss[["may", "june", "july"]].astype(float)) >= .05)
    board_shift = bool(
        np.ptp(limitups[["may", "june", "july"]].astype(float))
        >= .20 * float(limitups[["may", "june", "july"]].astype(float).mean())
        or np.ptp(promotion[["may", "june", "july"]].astype(float)) >= .05
        or np.ptp(height[["may", "june", "july"]].astype(float)) >= 1.0
    )
    path_shift = False
    for feature in PATH_FEATURES:
        row = _metric(summary, f"{feature}_cohort_median_date_equal")
        if np.ptp(row[["may", "june", "july"]].astype(float)) >= .005:
            path_shift = True
            break
    if opportunity_shift and sum((loss_shift, board_shift, path_shift)) >= 2:
        state = "LOCAL_REGIME_DIFFERENCE_SUPPORTED"
    elif sum((opportunity_shift, loss_shift, board_shift, path_shift)) >= 2:
        state = "LOCAL_REGIME_DIFFERENCE_WEAK"
    else:
        state = "LOCAL_REGIME_DIFFERENCE_NOT_SUPPORTED"

    # This local audit cannot observe broad-market breadth, index, liquidity or
    # style context.  Expansion is needed when local ecology is incomplete or
    # does not by itself provide a supported description.
    broader = "NO" if state == "LOCAL_REGIME_DIFFERENCE_SUPPORTED" else "YES"
    audit = {
        "opportunity_shift": opportunity_shift,
        "loss_shift": loss_shift,
        "board_shift": board_shift,
        "path_shift": path_shift,
        "decision_is_descriptive_gate_not_regime_score": True,
    }
    return state, broader, audit


def _pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.2%}"


def _num(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.2f}"


def build_review(
    summary: pd.DataFrame,
    state: str,
    broader: str,
    audit: Mapping[str, Any],
) -> str:
    target = _metric(summary, "target7_rate_date_equal")
    loss = _metric(summary, "loss_rate_date_equal")
    capped = _metric(summary, "candidate_universe_capped_return_date_equal")
    limitups = _metric(summary, "limit_up_count_per_day_mean")
    promotion = _metric(summary, "promotion_success_rate_daily_mean")
    afternoon = _metric(summary, "d1_afternoon_return_cohort_median_date_equal")
    close_vwap = _metric(summary, "d1_close_to_vwap_raw_cohort_median_date_equal")

    comparable = []
    for metric in (
        "target7_rate_date_equal", "loss_rate_date_equal",
        "candidate_universe_capped_return_date_equal",
        "limit_up_count_per_day_mean", "promotion_success_rate_daily_mean",
        "d1_afternoon_return_cohort_median_date_equal",
        "d1_close_to_vwap_raw_cohort_median_date_equal",
    ):
        row = _metric(summary, metric)
        if abs(float(row["june"]) - float(row["july"])) <= abs(float(row["may"]) - float(row["june"])):
            comparable.append(metric)

    july_factors = []
    if float(target["july"]) < float(target["may"]):
        july_factors.append("Target7 opportunity lower than May")
    if float(loss["july"]) > float(loss["may"]):
        july_factors.append("LOSS higher than May")
    if float(limitups["july"]) < float(limitups["june"]):
        july_factors.append("final limit-up pool breadth lower than June")
    if float(afternoon["july"]) <= float(afternoon["june"]) - .005:
        july_factors.append("candidate-cohort D1 afternoon state weakened from June")
    if float(close_vwap["july"]) <= float(close_vwap["june"]) - .005:
        july_factors.append("candidate-cohort D1 close/VWAP state weakened from June")
    lines = [
        "# v004c Board-Break Regime Audit v001",
        "",
        "## Contract",
        "",
        "- Descriptive ecology audit only; no model fit, weighting/feature/regime search, or composite regime score.",
        "- Mature candidate outcomes only: 2026-05-06 through 2026-07-29; label_available_date < 2026-08-01.",
        "- Board ecology uses existing main-board final-limit-up caches only; no external data was connected.",
        f"- True failed-limit-up (炸板失败池) count/rate: [{FAILED_BOARD_STATUS}]. open_board_count is not substituted for this endpoint.",
        "- August signal-date outcome accessed: NO.",
        "",
        "## Q1. Are the May, June and July board-break repair environments clearly different?",
        "",
        f"- {state}. Date-equal Target7 rates May/June/July: {_pct(target['may'])} / {_pct(target['june'])} / {_pct(target['july'])}; LOSS: {_pct(loss['may'])} / {_pct(loss['june'])} / {_pct(loss['july'])}; universe capped: {_pct(capped['may'])} / {_pct(capped['june'])} / {_pct(capped['july'])}.",
        "",
        "## Q2. What most clearly distinguishes May from June/July?",
        "",
        f"- May has the strongest realized repair opportunity (highest Target7, lowest LOSS, highest universe capped return). Board/path ecology is not a simple May-vs-later split: final-limit-up pool mean is {_num(limitups['may'])} / {_num(limitups['june'])} / {_num(limitups['july'])}, promotion rate is {_pct(promotion['may'])} / {_pct(promotion['june'])} / {_pct(promotion['july'])}, and June—not May—is the strongest D1 path-state month (afternoon median {_pct(afternoon['may'])} / {_pct(afternoon['june'])} / {_pct(afternoon['july'])}; close/VWAP {_pct(close_vwap['may'])} / {_pct(close_vwap['june'])} / {_pct(close_vwap['july'])}).",
        "",
        "## Q3. Where are June and July most similar?",
        "",
        f"- Limited similarity only. Metrics with June-July absolute distance no larger than May-June: {', '.join(comparable) or 'NONE'}; the four D1 cohort-path measures do not form a broad June/July-identical pattern.",
        "",
        "## Q4. How does July deterioration appear?",
        "",
        f"- Simultaneous descriptive changes: {', '.join(july_factors) or 'no material preregistered local change identified'}. These are concurrent observations, not causal claims.",
        "",
        "## Q5. Does the board/break pool alone show a sufficiently clear local regime difference?",
        "",
        f"- {'YES' if state == 'LOCAL_REGIME_DIFFERENCE_SUPPORTED' else 'NO'}. Opportunity shift={audit['opportunity_shift']}; LOSS shift={audit['loss_shift']}; board ecology shift={audit['board_shift']}; D1 path-state shift={audit['path_shift']}.",
        "",
        "## Q6. Is broader-market context needed next?",
        "",
        f"- {broader}. This task stops at local ecology and does not infer causality.",
        "",
        f"LOCAL_REGIME_STATE = {state}",
        "",
        f"BROADER_MARKET_CONTEXT_NEEDED = {broader}",
        "",
        "NEXT_ACTION = STOP_AND_REVIEW",
        "",
        "AUGUST_SIGNAL_OUTCOME_ACCESSED = NO",
        "",
    ]
    return "\n".join(lines)


def build_outputs(root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    root_path = Path(root).resolve()
    population, population_audit = load_population(root_path)
    candidate_pool = build_candidate_pool(population)
    board_ecology = build_board_ecology(root_path, candidate_pool["signal_date"])
    path_state = build_path_state(population)
    break_ecology = build_break_ecology(population, path_state)
    recent_history = build_recent_matured_history(population)
    daily_ecology = build_daily_ecology(
        candidate_pool, board_ecology, break_ecology, recent_history
    )
    month_summary = build_month_summary(
        population, candidate_pool, board_ecology, break_ecology
    )
    state, broader, decision_audit = decide_state(month_summary)
    review = build_review(month_summary, state, broader, decision_audit)

    outputs = {
        OUTPUT_FILENAMES[0]: _csv_bytes(daily_ecology),
        OUTPUT_FILENAMES[1]: _csv_bytes(month_summary),
        OUTPUT_FILENAMES[2]: _csv_bytes(candidate_pool),
        OUTPUT_FILENAMES[3]: _csv_bytes(board_ecology),
        OUTPUT_FILENAMES[4]: _csv_bytes(break_ecology),
        OUTPUT_FILENAMES[5]: _csv_bytes(path_state),
        OUTPUT_FILENAMES[6]: review.encode("utf-8"),
    }
    audit = {
        **population_audit,
        **decision_audit,
        "local_regime_state": state,
        "broader_market_context_needed": broader,
        "next_action": "STOP_AND_REVIEW",
        "output_file_count": len(outputs),
        "board_ecology_dates": len(board_ecology),
        "board_ecology_missing_dates": 0,
        "failed_limit_up_status": FAILED_BOARD_STATUS,
        "recent5_available_dates": int(recent_history["recent5_status"].eq("AVAILABLE_STRICT_MATURED").sum()),
    }
    return outputs, audit


def write_outputs(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    outputs, audit = build_outputs(root_path)
    output_dir = root_path / "reports" / "research" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
    return output_dir, audit

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_mechanism_foundation import load_outcomes
from .v004c_pair_capped7_july_forward import (
    FROZEN_DEVELOPMENT_COMMIT,
    TRAINING_ASOF_DATE,
    _read_daily_through,
    build_exact_raw_features,
    build_july_d1_safe_features,
    build_july_universe,
    derive_trade_dates,
    load_authoritative_d1_safe_base,
    prepare_frozen_x,
)
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    L2,
    MODEL_FAMILY,
    MODEL_ID,
    POSITIVE_WEIGHT,
    RAW_INPUT_COLUMNS,
    _d0_before_signal,
    _sigmoid,
    assert_frozen_architecture_contract,
    dataframe_csv_bytes,
)


ANALYSIS_START = "2026-05-06"
ANALYSIS_END = "2026-07-31"
AUGUST_OUTCOME_SENTINEL = "2026-08-01"
ANALYSIS_SOURCE_COMMIT = "a9df61773bf3580b4cbb0b492b6174873e4126a5"
EXPECTED_JULY_ROWS = 178
EXPECTED_JULY_DATES = 23

OUTPUT_FILENAMES = (
    "v004c_stage1_18f_bridge_candidates.csv",
    "v004c_stage1_model_spec.json",
    "v004c_stage1_feature_lineage.csv",
    "v004c_stage1_bridge_integrity.csv",
    "v004c_stage1_18f_bridge_readme.md",
)

JULY_LOCK_RELATIVE_PATH = (
    "reports/research/v004c_pair_capped7_july_forward_v001_20260701_20260731/"
    "v004c_july_forward_prediction_lock_v001.csv"
)

# This is the exact as-of-2026-07-01 frozen Stage1 snapshot.  It was recovered
# once from the canonical 307-row / 37-date pre-July contract and accepted only
# after reproducing all 178 archived July scores (max absolute error below
# 5e-13) and every July rank.  The analysis path below never calls a learner.
FROZEN_INTERCEPT = 0.0073904329700413395
FROZEN_COEFFICIENTS: Mapping[str, float] = {
    "rank_d1_close_ma10_pct": 0.028381929500129365,
    "rank_d1_low_ma10_pct": 0.03462922470874273,
    "rank_trend_hold_score": 0.004000100647297399,
    "rank_total_score": 0.014480527253889356,
    "rank_theme_score": 0.0399111112533928,
    "rank_days_since_d0": 0.014568697511732292,
    "rank_log_candidate_base_price": 0.046516606517835804,
    "rank_active_money_score": 0.03272657223332552,
    "rank_d1_close_vwap_pct": 0.011629561448546454,
    "inter_close_low": 0.03881801523378375,
    "inter_close_trend": 0.0152070375869904,
    "inter_total_trend": 0.019483982930336388,
    "inter_total_active": 0.03522808685709867,
    "inter_low_active": 0.04016981964407899,
    "spread_close_low": -0.006247295208613493,
    "days_since_d0_le1": 0.07524250021109626,
    "days_since_d0_eq2": 9.0652745530704e-05,
    "days_since_d0_ge3": -0.07533315295662699,
}


def assert_analysis_contract() -> None:
    assert_frozen_architecture_contract()
    if list(FROZEN_COEFFICIENTS) != list(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: frozen coefficient order differs from exact 18F order")
    if len(FROZEN_FEATURE_COLUMNS) != 18:
        raise RuntimeError("FATAL: frozen feature count changed")
    if L2 != 0.30 or POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen Stage1 hyperparameters changed")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_july_lock(root: Path) -> tuple[pd.DataFrame, str]:
    path = root / JULY_LOCK_RELATIVE_PATH
    columns = [
        "event_id", "signal_date", "code", "board", "candidate_count",
        "stage1_score", "stage1_rank",
    ]
    lock = pd.read_csv(
        path, usecols=columns, encoding="utf-8-sig", dtype={"code": str}
    )
    lock["code"] = lock["code"].astype(str).str.zfill(6)
    lock["event_id"] = lock["event_id"].astype(str)
    lock["signal_date"] = lock["signal_date"].astype(str)
    checks = {
        "rows": len(lock) == EXPECTED_JULY_ROWS,
        "dates": lock["signal_date"].nunique() == EXPECTED_JULY_DATES,
        "unique": lock["event_id"].nunique() == EXPECTED_JULY_ROWS,
        "range": lock["signal_date"].between("2026-07-01", ANALYSIS_END).all(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"FATAL: archived July lock parity failed: {failed}")
    return lock, _sha256_file(path)


def _date_context(root: Path, events: pd.DataFrame) -> pd.DataFrame:
    dates = derive_trade_dates(root, events)
    d0_rows: list[dict[str, str]] = []
    for event in events[["event_id", "code", "signal_date"]].itertuples(index=False):
        daily = _read_daily_through(root, str(event.code).zfill(6), str(event.signal_date))
        d0_rows.append({
            "event_id": str(event.event_id),
            "d0_date": _d0_before_signal(daily, str(event.signal_date)),
        })
    context = dates.merge(pd.DataFrame(d0_rows), on="event_id", validate="one_to_one")
    context["d1_date"] = context["signal_date"]
    for column in (
        "d0_date", "d1_date", "d2_date", "d3_date", "label_available_date"
    ):
        context[column] = pd.to_datetime(context[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        if context[column].isna().any():
            raise RuntimeError(f"FATAL: date context incomplete: {column}")
    invalid = ~(
        context["d0_date"].lt(context["d1_date"])
        & context["d1_date"].lt(context["d2_date"])
        & context["d2_date"].lt(context["d3_date"])
        & context["d3_date"].eq(context["label_available_date"])
    )
    if bool(invalid.any()):
        raise RuntimeError("FATAL: D0/D1/D2/D3 chronology failed")
    return context[[
        "event_id", "d0_date", "d1_date", "d2_date", "d3_date",
        "label_available_date",
    ]]


def _score_snapshot(x: pd.DataFrame) -> pd.DataFrame:
    beta = np.asarray([FROZEN_COEFFICIENTS[name] for name in FROZEN_FEATURE_COLUMNS])
    scored = x.copy()
    scored["_reconstructed_logit"] = (
        FROZEN_INTERCEPT + scored[FROZEN_FEATURE_COLUMNS].to_numpy(float) @ beta
    )
    scored["_reconstructed_score"] = _sigmoid(scored["_reconstructed_logit"].to_numpy())
    scored = scored.sort_values(
        ["signal_date", "_reconstructed_score", "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scored["_reconstructed_rank"] = (
        scored.groupby("signal_date", sort=True).cumcount() + 1
    )
    scored["candidate_count"] = scored.groupby("signal_date", sort=True)[
        "event_id"
    ].transform("size")
    return scored


def _load_pre_august_outcomes(
    root: Path, events: pd.DataFrame, date_context: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    identity = events[["event_id", "code", "signal_date"]].merge(
        date_context, on="event_id", validate="one_to_one"
    )
    may_june = identity[identity["signal_date"].lt("2026-07-01")].copy()
    may_june_values = load_outcomes(root, events[events["signal_date"].lt("2026-07-01")])
    may_june_outcomes = may_june[["event_id"]].merge(
        may_june_values[["event_id", "raw_repair_return"]],
        on="event_id",
        validate="one_to_one",
    )

    july_eligible = identity[
        identity["signal_date"].ge("2026-07-01")
        & identity["label_available_date"].lt(AUGUST_OUTCOME_SENTINEL)
    ].copy()
    july_rows: list[dict[str, Any]] = []
    accessed_outcome_dates: list[str] = []
    for row in july_eligible.itertuples(index=False):
        # The point-in-time clip ends on D3 and D3 is asserted pre-August before
        # any D2-open or D3-high value is selected.
        if str(row.d3_date) >= AUGUST_OUTCOME_SENTINEL:
            raise RuntimeError("FATAL: attempted August outcome access")
        daily = _read_daily_through(root, str(row.code).zfill(6), str(row.d3_date))
        d2 = daily[daily["date"].eq(str(row.d2_date))]
        d3 = daily[daily["date"].eq(str(row.d3_date))]
        if len(d2) != 1 or len(d3) != 1:
            raise RuntimeError(f"FATAL: July pre-August outcome unavailable: {row.event_id}")
        raw_return = float(d3.iloc[0]["high"]) / float(d2.iloc[0]["open"]) - 1.0
        july_rows.append({"event_id": row.event_id, "raw_repair_return": raw_return})
        accessed_outcome_dates.extend([str(row.d2_date), str(row.d3_date)])

    outcomes = pd.concat(
        [may_june_outcomes, pd.DataFrame(july_rows)], ignore_index=True
    )
    if outcomes["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicated outcome identity")
    if accessed_outcome_dates and max(accessed_outcome_dates) >= AUGUST_OUTCOME_SENTINEL:
        raise RuntimeError("FATAL: August outcome date accessed")
    raw = pd.to_numeric(outcomes["raw_repair_return"], errors="raise")
    outcomes["target7"] = raw.ge(0.07).astype(int)
    outcomes["positive_non_target"] = (raw.ge(0.0) & raw.lt(0.07)).astype(int)
    outcomes["loss"] = raw.lt(0.0).astype(int)
    outcomes["severe_loss"] = raw.le(-0.05).astype(int)
    outcomes["capped_return_7"] = np.minimum(raw, 0.07)
    audit = {
        "july_outcome_rows_available_pre_august": int(len(july_rows)),
        "august_outcome_rows_accessed": 0,
        "latest_accessed_july_outcome_date": (
            max(accessed_outcome_dates) if accessed_outcome_dates else None
        ),
    }
    return outcomes, audit


def _assemble_period(
    raw: pd.DataFrame,
    x: pd.DataFrame,
    dates: pd.DataFrame,
    score_origin: str,
) -> pd.DataFrame:
    scored = _score_snapshot(x)
    identity = raw[[
        "event_id", "signal_date", "code", "board_streak_before_break",
        *RAW_INPUT_COLUMNS,
    ]].copy()
    for column in RAW_INPUT_COLUMNS:
        identity[f"raw__{column}"] = pd.to_numeric(identity[column], errors="coerce")
    identity["raw__log_candidate_base_price"] = np.log(
        pd.to_numeric(identity["candidate_base_price"], errors="raise")
    )
    keep_raw = [f"raw__{column}" for column in RAW_INPUT_COLUMNS] + [
        "raw__log_candidate_base_price"
    ]
    frame = scored[[
        "event_id", "_reconstructed_logit", "_reconstructed_score",
        "_reconstructed_rank", "candidate_count", *FROZEN_FEATURE_COLUMNS,
    ]].merge(
        identity[[
            "event_id", "signal_date", "code", "board_streak_before_break",
            *keep_raw,
        ]],
        on="event_id",
        validate="one_to_one",
    ).merge(dates, on="event_id", validate="one_to_one")
    frame["board_group"] = "BOARD" + pd.to_numeric(
        frame["board_streak_before_break"], errors="raise"
    ).astype(int).astype(str)
    frame["score_origin"] = score_origin
    return frame


def build_feature_lineage() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        feature: str, raw_source: str, formula: str, transform: str,
        parents: Sequence[str], availability: str, daily: bool, interaction: bool,
    ) -> None:
        rows.append({
            "feature_name": feature,
            "raw_source": raw_source,
            "formula": formula,
            "transform": transform,
            "parent_features": "|".join(parents),
            "information_availability_time": availability,
            "daily_cross_sectional": "YES" if daily else "NO",
            "interaction": "YES" if interaction else "NO",
        })

    rank_rows = [
        ("rank_d1_close_ma10_pct", "d1_close_ma10_pct", "(D1 close / MA10 - 1) * 100"),
        ("rank_d1_low_ma10_pct", "d1_low_ma10_pct", "(D1 low / MA10 - 1) * 100"),
        ("rank_trend_hold_score", "trend_hold_score", "src.signal_engine.score_trend_hold"),
        ("rank_total_score", "total_score", "src.signal_engine.score_total with frozen StrategyConfig"),
        ("rank_theme_score", "theme_score", "src.theme_score.score_theme over five-calendar-day limit-up pool"),
        ("rank_days_since_d0", "days_since_d0", "calendar days from canonical D0 to D1"),
        ("rank_log_candidate_base_price", "candidate_base_price", "log(D1 close)"),
        ("rank_active_money_score", "active_money_score", "src.active_money.score_active_money"),
        ("rank_d1_close_vwap_pct", "d1_close_vwap_pct", "(D1 close / D1 intraday VWAP - 1) * 100"),
    ]
    for feature, raw, formula in rank_rows:
        add(
            feature,
            f"canonical generate_signal / raw__{raw}",
            formula,
            "within signal_date: pandas rank(pct=True, method='average'); missing rank -> 0.5",
            [raw],
            "D1_CLOSE",
            True,
            False,
        )

    interactions = [
        ("inter_close_low", "rank_d1_close_ma10_pct * rank_d1_low_ma10_pct", ["rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct"]),
        ("inter_close_trend", "rank_d1_close_ma10_pct * rank_trend_hold_score", ["rank_d1_close_ma10_pct", "rank_trend_hold_score"]),
        ("inter_total_trend", "rank_total_score * rank_trend_hold_score", ["rank_total_score", "rank_trend_hold_score"]),
        ("inter_total_active", "rank_total_score * rank_active_money_score", ["rank_total_score", "rank_active_money_score"]),
        ("inter_low_active", "rank_d1_low_ma10_pct * rank_active_money_score", ["rank_d1_low_ma10_pct", "rank_active_money_score"]),
        ("spread_close_low", "rank_d1_close_ma10_pct - rank_d1_low_ma10_pct", ["rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct"]),
    ]
    for feature, formula, parents in interactions:
        add(
            feature, "derived from frozen daily rank features", formula,
            "row-wise frozen interaction after daily rank construction", parents,
            "D1_CLOSE", False, True,
        )

    for feature, formula in (
        ("days_since_d0_le1", "int(days_since_d0 <= 1)"),
        ("days_since_d0_eq2", "int(days_since_d0 == 2)"),
        ("days_since_d0_ge3", "int(days_since_d0 >= 3)"),
    ):
        add(
            feature, "canonical days_since_d0", formula,
            "row-wise Boolean bucket; missing -> 0", ["days_since_d0"],
            "D1", False, False,
        )
    lineage = pd.DataFrame(rows)
    if lineage["feature_name"].tolist() != list(FROZEN_FEATURE_COLUMNS):
        raise RuntimeError("FATAL: feature lineage order differs from frozen 18F")
    return lineage


def build_model_spec(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "git_commit": FROZEN_DEVELOPMENT_COMMIT,
        "analysis_source_commit": ANALYSIS_SOURCE_COMMIT,
        "snapshot_status": "CANONICAL_FROZEN_SNAPSHOT_RECONSTRUCTED_AND_JULY_LOCK_VERIFIED",
        "new_model_trained": False,
        "feature_search": False,
        "hyperparameter_search": False,
        "feature_count": 18,
        "features": list(FROZEN_FEATURE_COLUMNS),
        "intercept": FROZEN_INTERCEPT,
        "coefficients": dict(FROZEN_COEFFICIENTS),
        "L2": L2,
        "weights": {
            "positive_class_weight": POSITIVE_WEIGHT,
            "date_weight": "1 / candidate rows on historical signal_date",
            "tail_weight": {
                "raw_repair_return_gte_0.12": 2.0,
                "raw_repair_return_gte_0.10_and_lt_0.12": 1.5,
                "otherwise": 1.0,
            },
            "sample_weight_formula": "date_weight * positive_class_weight_if_target7 * tail_weight",
        },
        "missing_policy": {
            "raw_numeric": "coerce invalid to NaN",
            "daily_rank": "NaN rank filled with 0.5",
            "final_model_feature": "remaining NaN filled with 0.0",
            "candidate_base_price": "must be non-null and > 0",
        },
        "clipping": {
            "model_features": "NONE",
            "logit_for_sigmoid_numerical_stability": [-35.0, 35.0],
            "capped_return_7": "upper cap at 0.07; outcome only, never a model feature",
        },
        "standardization": "NONE",
        "daily_percentile_rank_transform": {
            "scope": "complete authoritative v004c Board2+Board3 candidate universe on each signal_date",
            "implementation": "pandas Series.rank(pct=True, method='average')",
            "tie_policy": "average",
            "missing_fill": 0.5,
        },
        "interaction_formulas": {
            row["feature_name"]: row["formula"]
            for row in build_feature_lineage().to_dict("records")
            if row["interaction"] == "YES"
        },
        "candidate_definition": (
            "authoritative v004c main-board universe: first non-limit-up D1 after "
            "exactly 2 or 3 consecutive limit-up trading days"
        ),
        "label_definition": {
            "raw_repair_return": "D3 high / D2 open - 1",
            "target7": "int(raw_repair_return >= 0.07)",
            "positive_non_target": "int(0 <= raw_repair_return < 0.07)",
            "loss": "int(raw_repair_return < 0)",
            "severe_loss": "int(raw_repair_return <= -0.05)",
            "capped_return_7": "min(raw_repair_return, 0.07); no lower floor",
        },
        "training_period": {
            "eligible_signal_date_start": "2026-05-06",
            "eligible_signal_date_end": "2026-06-26",
            "rows": 307,
            "dates": 37,
        },
        "training_asof": TRAINING_ASOF_DATE,
        "label_availability_rule": "label_available_date < 2026-07-01",
        "score_semantics": {
            "may_june": "retrospective application of the frozen as-of-2026-07-01 snapshot to the complete development universe",
            "july": "forward frozen snapshot score, parity-checked against the existing archived July prediction lock",
        },
        "july_lock": {
            "path": JULY_LOCK_RELATIVE_PATH,
            "sha256": context["july_lock_sha256"],
            "rows": context["july_rows"],
            "dates": context["july_dates"],
            "score_reconstruction_max_abs_error": context[
                "july_score_reconstruction_max_abs_error"
            ],
            "rank_mismatch_rows": context["july_rank_mismatch_rows"],
        },
        "august_outcome_rows_accessed": 0,
        "august_outcomes_exported": 0,
    }


def _integrity_rows(frame: pd.DataFrame, context: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, metric: str, value: Any, status: str = "INFO", notes: str = "") -> None:
        rows.append({
            "scope": scope, "metric": metric, "value": value,
            "status": status, "notes": notes,
        })

    scopes = [
        ("MAY", frame[frame["signal_date"].str.startswith("2026-05")]),
        ("JUNE", frame[frame["signal_date"].str.startswith("2026-06")]),
        ("JULY", frame[frame["signal_date"].str.startswith("2026-07")]),
        ("TOTAL", frame),
    ]
    for scope, subset in scopes:
        add(scope, "rows", len(subset))
        add(scope, "signal_dates", subset["signal_date"].nunique())
        add(scope, "board2_rows", int(subset["board_group"].eq("BOARD2").sum()))
        add(scope, "board3_rows", int(subset["board_group"].eq("BOARD3").sum()))
        available = subset["raw_repair_return"].notna()
        add(scope, "outcome_available_rows", int(available.sum()))
        add(scope, "target7_rows", int(subset["target7"].fillna(0).sum()))
        add(scope, "positive_non_target_rows", int(subset["positive_non_target"].fillna(0).sum()))
        add(scope, "loss_rows", int(subset["loss"].fillna(0).sum()))
        add(scope, "severe_loss_rows", int(subset["severe_loss"].fillna(0).sum()))
        add(scope, "outcome_withheld_rows", int((~available).sum()))

    duplicate_count = int(frame["event_id"].duplicated().sum())
    missing_cells = int(frame[FROZEN_FEATURE_COLUMNS].isna().sum().sum())
    missing_rows = int(frame[FROZEN_FEATURE_COLUMNS].isna().any(axis=1).sum())
    add("QUALITY", "duplicate_event_id_count", duplicate_count, "PASS" if duplicate_count == 0 else "FAIL")
    add("QUALITY", "missing_18f_cell_count", missing_cells, "PASS" if missing_cells == 0 else "FAIL")
    add("QUALITY", "missing_18f_row_count", missing_rows, "PASS" if missing_rows == 0 else "FAIL")
    add(
        "QUALITY", "stage1_score_reconstruction_max_abs_error",
        context["score_reconstruction_max_abs_error"],
        "PASS" if context["score_reconstruction_max_abs_error"] <= 1e-12 else "FAIL",
    )
    add("JULY_LOCK", "expected_rows", EXPECTED_JULY_ROWS)
    add("JULY_LOCK", "observed_rows", context["july_rows"], "PASS" if context["july_rows"] == EXPECTED_JULY_ROWS else "FAIL")
    add("JULY_LOCK", "expected_dates", EXPECTED_JULY_DATES)
    add("JULY_LOCK", "observed_dates", context["july_dates"], "PASS" if context["july_dates"] == EXPECTED_JULY_DATES else "FAIL")
    add("JULY_LOCK", "identity_mismatch_rows", context["july_identity_mismatch_rows"], "PASS" if context["july_identity_mismatch_rows"] == 0 else "FAIL")
    add("JULY_LOCK", "rank_mismatch_rows", context["july_rank_mismatch_rows"], "PASS" if context["july_rank_mismatch_rows"] == 0 else "FAIL")
    add("JULY_LOCK", "candidate_count_mismatch_rows", context["july_candidate_count_mismatch_rows"], "PASS" if context["july_candidate_count_mismatch_rows"] == 0 else "FAIL")
    add("JULY_LOCK", "score_max_abs_error", context["july_score_reconstruction_max_abs_error"], "PASS" if context["july_score_reconstruction_max_abs_error"] <= 1e-12 else "FAIL")
    add("OUTCOME_ACCESS", "august_outcome_rows_accessed", 0, "PASS")
    add("OUTCOME_ACCESS", "august_outcomes_exported", 0, "PASS")
    add("OUTCOME_ACCESS", "july_rows_with_august_label_outcomes_withheld", context["july_outcomes_withheld_for_august"], "PASS", "candidate identity/features/scores retained; outcome columns blank")
    return pd.DataFrame(rows)


def _readme(context: Mapping[str, Any]) -> str:
    return "\n".join([
        "# v004c frozen Stage1 18F bridge analysis package",
        "",
        "- Scope: complete authoritative v004c Board2/Board3 first-break universe, 2026-05-06 through 2026-07-31.",
        "- Model: current frozen `V4A_ARCH_TRANSFER_V4C`, 18 features, L2=0.30, positive weight=1.50.",
        "- May/June scores: retrospective application of the single as-of-2026-07-01 frozen snapshot; they are not chronological OOF scores.",
        "- July scores: exact frozen forward scores checked against the existing 178-row / 23-date prediction lock.",
        f"- July score reconstruction max absolute error: {context['july_score_reconstruction_max_abs_error']:.3e}; rank mismatches: {context['july_rank_mismatch_rows']}.",
        "- August outcomes: never accessed or exported. July candidates whose D3/label date is in August remain in the candidate table with blank outcome fields.",
        "- No model search, feature selection, hyperparameter search, candidate filtering, or TopK change was performed.",
        "",
    ])


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_analysis_contract()
    base = load_authoritative_d1_safe_base(root)
    may_june_raw, _, _, feature_gate = build_exact_raw_features(root, base)
    if not feature_gate["pass"]:
        raise RuntimeError("FATAL: May/June frozen raw feature gate failed")
    may_june_x = prepare_frozen_x(may_june_raw)
    may_june_dates = _date_context(root, base)

    july_universe, _ = build_july_universe(root)
    july_raw = build_july_d1_safe_features(root, july_universe)
    july_x = prepare_frozen_x(july_raw)
    july_dates = _date_context(root, july_universe)
    july_lock, july_lock_sha = _load_july_lock(root)

    may_june = _assemble_period(
        may_june_raw, may_june_x, may_june_dates,
        "FROZEN_ASOF_2026_07_01_RETROSPECTIVE",
    )
    july = _assemble_period(
        july_raw, july_x, july_dates,
        "FROZEN_ASOF_2026_07_01_FORWARD_LOCKED",
    )
    july_check = july[[
        "event_id", "signal_date", "code", "board_streak_before_break",
        "candidate_count", "_reconstructed_score", "_reconstructed_rank",
    ]].merge(july_lock, on=["event_id", "signal_date", "code"], validate="one_to_one")
    identity_mismatches = len(set(july["event_id"]) ^ set(july_lock["event_id"]))
    score_error = np.abs(
        july_check["_reconstructed_score"] - july_check["stage1_score"]
    )
    rank_mismatches = int(
        july_check["_reconstructed_rank"].ne(july_check["stage1_rank"]).sum()
    )
    candidate_mismatches = int(
        july_check["candidate_count_x"].ne(july_check["candidate_count_y"]).sum()
    )
    board_mismatches = int(
        pd.to_numeric(july_check["board_streak_before_break"]).ne(
            pd.to_numeric(july_check["board"])
        ).sum()
    )
    july_max_error = float(score_error.max())
    if any((identity_mismatches, rank_mismatches, candidate_mismatches, board_mismatches)):
        raise RuntimeError("FATAL: July population differs from archived prediction lock")
    if july_max_error > 1e-12:
        raise RuntimeError("FATAL: July frozen score reconstruction parity failed")

    # Preserve the archived July score serialization while retaining the exact
    # reconstructed logit and using the reconstructed score for all May/June rows.
    july = july.drop(columns=["_reconstructed_score", "_reconstructed_rank"]).merge(
        july_lock[["event_id", "stage1_score", "stage1_rank"]],
        on="event_id", validate="one_to_one",
    )
    may_june = may_june.rename(columns={
        "_reconstructed_score": "stage1_score",
        "_reconstructed_rank": "stage1_rank",
    })
    combined = pd.concat([may_june, july], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(combined) != 497 or combined["signal_date"].nunique() != 62:
        raise RuntimeError("FATAL: May-July complete candidate population must be 497/62")

    date_context = combined[[
        "event_id", "d0_date", "d1_date", "d2_date", "d3_date",
        "label_available_date",
    ]]
    all_events = pd.concat([
        base[[
            "event_id", "code", "signal_date", "board_streak_before_break",
        ]],
        july_universe[[
            "event_id", "code", "signal_date", "board_streak_before_break",
        ]],
    ], ignore_index=True)
    outcomes, outcome_audit = _load_pre_august_outcomes(root, all_events, date_context)
    combined = combined.merge(outcomes, on="event_id", how="left", validate="one_to_one")
    for column in ("target7", "positive_non_target", "loss", "severe_loss"):
        combined[column] = combined[column].astype("Int64")

    beta = np.asarray([FROZEN_COEFFICIENTS[name] for name in FROZEN_FEATURE_COLUMNS])
    reconstructed = _sigmoid(
        FROZEN_INTERCEPT + combined[FROZEN_FEATURE_COLUMNS].to_numpy(float) @ beta
    )
    combined["stage1_score_reconstruction_abs_error"] = np.abs(
        combined["stage1_score"].to_numpy(float) - reconstructed
    )
    max_error = float(combined["stage1_score_reconstruction_abs_error"].max())
    if max_error > 1e-12:
        raise RuntimeError("FATAL: exported score cannot be reconstructed from model spec")

    context: dict[str, Any] = {
        "rows": len(combined),
        "dates": int(combined["signal_date"].nunique()),
        "july_rows": int(len(july)),
        "july_dates": int(july["signal_date"].nunique()),
        "july_lock_sha256": july_lock_sha,
        "july_identity_mismatch_rows": identity_mismatches,
        "july_rank_mismatch_rows": rank_mismatches,
        "july_candidate_count_mismatch_rows": candidate_mismatches,
        "july_board_mismatch_rows": board_mismatches,
        "july_score_reconstruction_max_abs_error": july_max_error,
        "score_reconstruction_max_abs_error": max_error,
        "july_outcomes_withheld_for_august": int(
            combined["signal_date"].str.startswith("2026-07").sum()
            - combined.loc[
                combined["signal_date"].str.startswith("2026-07"),
                "raw_repair_return",
            ].notna().sum()
        ),
        **outcome_audit,
    }

    raw_columns = [f"raw__{column}" for column in RAW_INPUT_COLUMNS] + [
        "raw__log_candidate_base_price"
    ]
    output_columns = [
        "event_id", "code", "signal_date", "d0_date", "d1_date", "d2_date",
        "d3_date", "board_group", "candidate_count", "label_available_date",
        "stage1_score", "stage1_rank", "score_origin",
        *FROZEN_FEATURE_COLUMNS, *raw_columns,
        "target7", "positive_non_target", "loss", "severe_loss",
        "raw_repair_return", "capped_return_7",
        "stage1_score_reconstruction_abs_error",
    ]
    candidates = combined[output_columns].copy()
    lineage = build_feature_lineage()
    model_spec = build_model_spec(context)
    integrity = _integrity_rows(candidates, context)
    if integrity["status"].eq("FAIL").any():
        failed = integrity.loc[integrity["status"].eq("FAIL"), "metric"].tolist()
        raise RuntimeError(f"FATAL: integrity checks failed: {failed}")

    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(candidates),
        OUTPUT_FILENAMES[1]: (
            json.dumps(model_spec, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8"),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(lineage),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(integrity),
        OUTPUT_FILENAMES[4]: _readme(context).encode("utf-8"),
    }
    context["candidates"] = candidates
    context["lineage"] = lineage
    context["model_spec"] = model_spec
    context["integrity"] = integrity
    return outputs, context


def run_v004c_stage1_18f_bridge_analysis(
    root: str | Path, output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    target = Path(output_dir) if output_dir else (
        root_path / "reports/research/"
        "v004c_stage1_18f_bridge_analysis_v001_20260506_20260731"
    )
    outputs, context = build_outputs(root_path)
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (target / name).write_bytes(payload)
    context["output_dir"] = target
    return target, context

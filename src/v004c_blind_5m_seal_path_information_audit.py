"""Blind information audit for frozen conservative 5-minute seal-path proxies.

The three proxies are loaded verbatim from the preceding coverage/lineage
lock.  This module does not reconstruct minute bars, change proxy semantics,
fit a model, or inspect August signal-date outcomes.  The archived S2
coefficients are used only to reproduce the frozen reference score/rank and
identify its already-defined wrong Target7-vs-LOSS pairs.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


TASK_NAME = "v004c_blind_5m_seal_path_information_audit_v001"
OUTPUT_DIRNAME = TASK_NAME
DEVELOPMENT_START = "2026-05-06"
DEVELOPMENT_END = "2026-07-29"
TRAINING_ASOF = "2026-08-01"

COVERAGE_DIR = (
    "reports/research/v004c_5m_conservative_seal_path_coverage_audit_v001"
)
RECONSTRUCTION_ARTIFACT = (
    f"{COVERAGE_DIR}/v004c_5m_seal_path_reconstruction_values_v001.csv"
)
PHASE_A_LOCK_ARTIFACT = f"{COVERAGE_DIR}/v004c_5m_seal_path_phase_a_lock_v001.csv"
COVERAGE_ARTIFACT = f"{COVERAGE_DIR}/v004c_5m_seal_path_coverage_v001.csv"

BRIDGE_ARTIFACT = (
    "reports/research/v004c_stage1_18f_bridge_analysis_v001_20260506_20260731/"
    "v004c_stage1_18f_bridge_candidates.csv"
)
S2_ARTIFACT_DIR = (
    "reports/research/"
    "v004c_reduced7f_training_spec_sanity_v001_20260506_20260731"
)
S2_COEFFICIENT_ARTIFACT = (
    f"{S2_ARTIFACT_DIR}/v004c_reduced7f_spec_coefficients_v001.csv"
)
S2_FIT_ARTIFACT = f"{S2_ARTIFACT_DIR}/v004c_reduced7f_spec_training_fit_v001.csv"

S2_SPEC = "S2_NO_TAIL_L2_010"
S2_L2 = 0.10
S2_POSITIVE_WEIGHT = 1.50
S2_TAIL_WEIGHTING = "NONE"

S2_FEATURES = (
    "rank_d1_close_ma10_pct",
    "rank_d1_low_ma10_pct",
    "rank_trend_hold_score",
    "rank_theme_score",
    "rank_log_candidate_base_price",
    "rank_active_money_score",
    "rank_d1_close_vwap_pct",
)

F1 = "F1_FINAL_CONFIRMED_STABLE_LOCK_TIME"
F2 = "F2_CONFIRMED_REOPEN"
F3 = "F3_D0_VS_PREV_LOCK_DETERIORATION"
FIXED_PROXIES = (F1, F2, F3)
PROXY_SHORT = {F1: "F1", F2: "F2", F3: "F3"}
VALID_COLUMNS = {F1: "F1_valid", F2: "F2_valid", F3: "F3_valid"}

PERIODS = ("POOLED", "MAY", "JUNE", "JULY")
MONTHS = ("MAY", "JUNE", "JULY")
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260908
PERMUTATION_REPETITIONS = 10_000
PERMUTATION_SEED = 20260909

# Conservative, fixed interpretation thresholds.  They do not pick a best
# proxy; they encode the user's hierarchy: wrong-pair rescue, Rank2-6,
# temporal/source stability, then pooled strength.
MATERIAL_PRIMARY = 0.55
MATERIAL_HEAD = 0.53
MATERIAL_RESCUE = 0.55
PARTIAL_SIGNAL = 0.54
MIN_PAIR_DATES = 10
MIN_F2_INFORMATIVE_PAIRS = 20
SELECTION_ADJUSTED_P_GATE = 0.10

OUTPUT_FILENAMES = (
    "v004c_5m_seal_path_analysis_population_v001.csv",
    "v004c_5m_seal_path_feature_values_v001.csv",
    "v004c_5m_seal_path_pair_concordance_v001.csv",
    "v004c_5m_seal_path_monthly_stability_v001.csv",
    "v004c_5m_seal_path_rankwise_v001.csv",
    "v004c_5m_seal_path_rank2_6_v001.csv",
    "v004c_5m_seal_path_s2_wrong_pair_rescue_v001.csv",
    "v004c_5m_seal_path_source_robustness_v001.csv",
    "v004c_5m_seal_path_board_robustness_v001.csv",
    "v004c_5m_seal_path_missingness_v001.csv",
    "v004c_5m_seal_path_date_bootstrap_v001.csv",
    "v004c_5m_seal_path_family_permutation_v001.csv",
    "v004c_blind_5m_seal_path_information_review_v001.md",
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_csv_bytes(frame)).hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _month_label(signal_date: Any) -> str:
    month = str(signal_date)[:7]
    return {"2026-05": "MAY", "2026-06": "JUNE", "2026-07": "JULY"}.get(
        month, "OUT_OF_SCOPE"
    )


def _period_part(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    return frame if period == "POOLED" else frame[frame["month"].eq(period)]


def _source_name(cache_kind: Any) -> str:
    value = str(cache_kind).strip()
    if value == "baostock_5m":
        return "BAOSTOCK"
    if value == "minute_5m":
        return "SINA"
    return "MISSING"


def assert_contract() -> None:
    if FIXED_PROXIES != (F1, F2, F3) or len(FIXED_PROXIES) != 3:
        raise RuntimeError("FATAL: proxy family is not exact frozen F1/F2/F3")
    if (S2_L2, S2_POSITIVE_WEIGHT, S2_TAIL_WEIGHTING) != (0.10, 1.50, "NONE"):
        raise RuntimeError("FATAL: frozen S2 specification changed")
    if len(S2_FEATURES) != 7:
        raise RuntimeError("FATAL: frozen S2 feature manifest changed")


def _load_phase_a(root: Path) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    lock_path = root / PHASE_A_LOCK_ARTIFACT
    values_path = root / RECONSTRUCTION_ARTIFACT
    coverage_path = root / COVERAGE_ARTIFACT
    if not (lock_path.is_file() and values_path.is_file() and coverage_path.is_file()):
        raise FileNotFoundError("preceding 5m coverage lock/artifacts are missing")
    lock_frame = pd.read_csv(lock_path, encoding="utf-8-sig", dtype=str).fillna("")
    if not {"key", "value"}.issubset(lock_frame.columns):
        raise RuntimeError("FATAL: invalid preceding Phase-A lock schema")
    lock = dict(zip(lock_frame["key"], lock_frame["value"]))
    expected = {
        "candidate_rows": "485",
        "signal_dates": "60",
        "F1_FINAL_STABLE_LOCK_DATA": "READY",
        "F2_CONFIRMED_REOPEN_DATA": "READY",
        "F3_LOCK_DETERIORATION_DATA": "READY",
        "TEMPORAL_SOURCE_SHIFT": "YES",
        "LIMIT_PRICE_LINEAGE": "PASS",
        "5M_SEAL_PATH_DATA_STATE": "5M_SEAL_PATH_DATA_READY",
        "OUTCOME_ACCESSED": "NO",
        "MODEL_TRAINED": "NO",
        "FEATURE_SEARCH": "NO",
        "WINDOW_SEARCH": "NO",
        "EXTERNAL_DATA_FETCHED": "NO",
    }
    mismatches = {key: (lock.get(key), value) for key, value in expected.items() if lock.get(key) != value}
    if mismatches:
        raise RuntimeError(f"FATAL: preceding coverage lock failed: {mismatches}")

    values = pd.read_csv(values_path, encoding="utf-8-sig", dtype={"event_id": str, "code": str})
    required = {
        "event_id", "code", "signal_date", "month", "board_group", "d1_date", "d0_date",
        "prev_board_day", "d0_cache_kind", "prev_cache_kind", *FIXED_PROXIES,
        *VALID_COLUMNS.values(),
    }
    missing = sorted(required - set(values.columns))
    if missing:
        raise RuntimeError(f"FATAL: reconstruction artifact missing columns: {missing}")
    values = values[values["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)].copy()
    if (len(values), values["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: Phase-A reconstruction population parity failed")
    if values["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in reconstruction lock")
    if values["signal_date"].ge(TRAINING_ASOF).any():
        raise RuntimeError("FATAL: August signal date reached primary analysis")
    for feature in FIXED_PROXIES:
        values[feature] = pd.to_numeric(values[feature], errors="coerce")
        valid_col = VALID_COLUMNS[feature]
        values[valid_col] = pd.to_numeric(values[valid_col], errors="raise").astype(int)
        invalid_nonmissing = values[valid_col].eq(0) & values[feature].notna()
        valid_missing = values[valid_col].eq(1) & values[feature].isna()
        if invalid_nonmissing.any() or valid_missing.any():
            raise RuntimeError(f"FATAL: {feature} validity/value mismatch")
    if not values.loc[values[VALID_COLUMNS[F2]].eq(1), F2].isin([0.0, 1.0]).all():
        raise RuntimeError("FATAL: F2 is not the frozen binary conservative proxy")
    values["d0_source"] = values["d0_cache_kind"].map(_source_name)
    values["prev_source"] = values["prev_cache_kind"].map(_source_name)
    values["f3_source_group"] = np.select(
        [
            values["d0_source"].eq("BAOSTOCK") & values["prev_source"].eq("BAOSTOCK"),
            values["d0_source"].eq("SINA") & values["prev_source"].eq("SINA"),
            values["d0_source"].ne("MISSING") & values["prev_source"].ne("MISSING"),
        ],
        ["BAOSTOCK_BOTH", "SINA_BOTH", "MIXED_SOURCE"],
        default="MISSING",
    )
    values = values.sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)

    coverage = pd.read_csv(coverage_path, encoding="utf-8-sig")
    if "variable" not in coverage.columns:
        raise RuntimeError("FATAL: preceding coverage table schema changed")
    for feature in FIXED_PROXIES:
        short = PROXY_SHORT[feature]
        match = coverage[coverage["variable"].astype(str).eq(short)]
        if match.empty or not match["field_data_ready"].astype(str).eq("YES").any():
            raise RuntimeError(f"FATAL: {short} did not pass preceding coverage gate")
    return values, lock, coverage


def _load_s2_and_outcomes(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    bridge_path = root / BRIDGE_ARTIFACT
    coefficient_path = root / S2_COEFFICIENT_ARTIFACT
    fit_path = root / S2_FIT_ARTIFACT
    usecols = [
        "event_id", "code", "signal_date", "d0_date", "d1_date", "board_group",
        "label_available_date", "target7", "loss", "severe_loss", *S2_FEATURES,
    ]
    bridge = pd.read_csv(
        bridge_path, encoding="utf-8-sig", usecols=usecols,
        dtype={"event_id": str, "code": str},
    )
    bridge = bridge[
        bridge["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & bridge["label_available_date"].lt(TRAINING_ASOF)
    ].copy()
    if (len(bridge), bridge["signal_date"].nunique()) != (485, 60):
        raise RuntimeError("FATAL: mature May-July S2 population parity failed")
    if bridge["event_id"].duplicated().any():
        raise RuntimeError("FATAL: duplicate event_id in S2 source")
    if bridge[["target7", "loss", "severe_loss", *S2_FEATURES]].isna().any().any():
        raise RuntimeError("FATAL: required outcome/S2 feature missing")
    if bridge["signal_date"].ge(TRAINING_ASOF).any():
        raise RuntimeError("FATAL: August signal date reached primary analysis")

    coefficients = pd.read_csv(coefficient_path, encoding="utf-8-sig")
    frozen = coefficients[
        coefficients["stage"].eq("FULL_DEVELOPMENT_IN_SAMPLE")
        & coefficients["prediction_set"].eq("FULL_DEVELOPMENT_IN_SAMPLE")
        & coefficients["spec"].eq(S2_SPEC)
    ].copy()
    expected_features = {"__INTERCEPT__", *S2_FEATURES}
    if len(frozen) != 8 or set(frozen["feature"]) != expected_features:
        raise RuntimeError("FATAL: archived S2 coefficient identity mismatch")
    if not frozen["l2"].eq(S2_L2).all() or not frozen["positive_weight"].eq(S2_POSITIVE_WEIGHT).all():
        raise RuntimeError("FATAL: archived S2 hyperparameter mismatch")
    if not frozen["tail_weighting"].astype(str).eq(S2_TAIL_WEIGHTING).all():
        raise RuntimeError("FATAL: archived S2 tail setting mismatch")
    by_feature = frozen.set_index("feature")["coefficient"]
    beta = np.asarray(
        [float(by_feature.loc["__INTERCEPT__"])]
        + [float(by_feature.loc[name]) for name in S2_FEATURES], dtype=float,
    )
    x = bridge[list(S2_FEATURES)].to_numpy(float)
    bridge["s2_logit"] = beta[0] + x @ beta[1:]
    bridge["s2_score"] = _sigmoid(bridge["s2_logit"].to_numpy(float))
    pieces: list[pd.DataFrame] = []
    for _, day in bridge.groupby("signal_date", sort=True):
        ordered = day.sort_values(
            ["s2_score", "event_id"], ascending=[False, True], kind="mergesort"
        ).copy()
        ordered["s2_rank"] = np.arange(1, len(ordered) + 1)
        pieces.append(ordered)
    bridge = pd.concat(pieces, ignore_index=True).sort_values(
        ["signal_date", "s2_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    fit = pd.read_csv(fit_path, encoding="utf-8-sig")
    fit_row = fit[fit["spec"].eq(S2_SPEC)]
    if len(fit_row) != 1:
        raise RuntimeError("FATAL: archived S2 fit identity missing")
    fit_row = fit_row.iloc[0]
    parity = {
        "min": abs(float(bridge["s2_score"].min()) - float(fit_row["score_min"])),
        "max": abs(float(bridge["s2_score"].max()) - float(fit_row["score_max"])),
        "std": abs(float(bridge["s2_score"].std(ddof=0)) - float(fit_row["score_std"])),
    }
    if max(parity.values()) > 1e-12:
        raise RuntimeError(f"FATAL: frozen S2 score parity failed: {parity}")
    bridge["target7"] = bridge["target7"].astype(int)
    bridge["loss"] = bridge["loss"].astype(int)
    bridge["severe_loss"] = bridge["severe_loss"].astype(int)
    bridge["month"] = bridge["signal_date"].map(_month_label)
    audit = {
        "model_trained": "NO",
        "model_refits": 0,
        "score_reconstruction_max_summary_error": max(parity.values()),
        "coefficient_artifact": S2_COEFFICIENT_ARTIFACT,
        "s2_spec": S2_SPEC,
    }
    return bridge, audit


def load_analysis_population(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_contract()
    root_path = Path(root).resolve()
    features, lock, coverage = _load_phase_a(root_path)
    s2, s2_audit = _load_s2_and_outcomes(root_path)
    feature_cols = [
        "event_id", "code", "signal_date", "month", "board_group", "d1_date", "d0_date",
        "prev_board_day", "d0_cache_kind", "prev_cache_kind", "d0_source", "prev_source",
        "f3_source_group", *FIXED_PROXIES, *VALID_COLUMNS.values(),
        "F1_missing_reason", "F2_missing_reason", "F3_missing_reason",
    ]
    merged = s2.merge(
        features[feature_cols],
        on=["event_id", "code", "signal_date", "month", "board_group", "d1_date", "d0_date"],
        how="inner", validate="one_to_one",
    )
    if len(merged) != 485 or set(merged["event_id"]) != set(features["event_id"]):
        raise RuntimeError("FATAL: Phase-A and outcome/S2 event identity mismatch")
    if not merged["label_available_date"].lt(TRAINING_ASOF).all():
        raise RuntimeError("FATAL: immature label reached audit")
    if merged["signal_date"].ge(TRAINING_ASOF).any():
        raise RuntimeError("FATAL: August signal-date outcome reached audit")
    audit = {
        **s2_audit,
        "rows": len(merged),
        "dates": int(merged["signal_date"].nunique()),
        "phase_a_lock": lock,
        "phase_a_values_sha256": _frame_sha256(features),
        "coverage_rows": len(coverage),
        "august_used_in_primary_analysis": "NO",
        "feature_search": "NO",
        "window_search": "NO",
    }
    return merged.sort_values(["signal_date", "s2_rank", "event_id"], kind="mergesort").reset_index(drop=True), audit


def _make_pairs(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = frame[frame[VALID_COLUMNS[feature]].eq(1)].copy()
    for signal_date, day in valid.groupby("signal_date", sort=True):
        winners = day[day["target7"].eq(1)].sort_values("event_id", kind="mergesort")
        losses = day[day["loss"].eq(1)].sort_values("event_id", kind="mergesort")
        for winner in winners.itertuples(index=False):
            for loser in losses.itertuples(index=False):
                t7_value = float(getattr(winner, feature))
                loss_value = float(getattr(loser, feature))
                if feature == F2:
                    better = int(t7_value == 0.0 and loss_value == 1.0)
                    worse = int(t7_value == 1.0 and loss_value == 0.0)
                else:
                    better = int(t7_value < loss_value)
                    worse = int(t7_value > loss_value)
                tie = int(not better and not worse)
                rows.append(
                    {
                        "feature": feature,
                        "proxy": PROXY_SHORT[feature],
                        "signal_date": str(signal_date),
                        "month": str(winner.month),
                        "target7_event_id": str(winner.event_id),
                        "loss_event_id": str(loser.event_id),
                        "target7_board_group": str(winner.board_group),
                        "loss_board_group": str(loser.board_group),
                        "target7_source": str(winner.d0_source),
                        "loss_source": str(loser.d0_source),
                        "target7_value": t7_value,
                        "loss_value": loss_value,
                        "target7_s2_score": float(winner.s2_score),
                        "loss_s2_score": float(loser.s2_score),
                        "target7_s2_rank": int(winner.s2_rank),
                        "loss_s2_rank": int(loser.s2_rank),
                        "quality_better": better,
                        "quality_worse": worse,
                        "quality_tie": tie,
                        "quality_concordance_value": 1.0 if better else 0.0 if worse else 0.5,
                        "s2_wrong_pair": int(float(winner.s2_score) <= float(loser.s2_score)),
                    }
                )
    return pd.DataFrame(rows)


def _pair_summary(pairs: pd.DataFrame, feature: str, period: str, scope: str) -> dict[str, Any]:
    part = _period_part(pairs, period)
    informative = int((part["quality_better"] + part["quality_worse"]).sum()) if len(part) else 0
    better = int(part["quality_better"].sum()) if len(part) else 0
    if feature == F2:
        quality = better / informative if informative else math.nan
    else:
        quality = float(part["quality_concordance_value"].mean()) if len(part) else math.nan
    return {
        "feature": feature,
        "proxy": PROXY_SHORT[feature],
        "period": period,
        "scope": scope,
        "pair_count": len(part),
        "date_count": int(part["signal_date"].nunique()) if len(part) else 0,
        "strict_quality_better_rate": float(part["quality_better"].mean()) if len(part) else math.nan,
        "quality_oriented_concordance": quality,
        "tie_adjusted_concordance": float(part["quality_concordance_value"].mean()) if len(part) else math.nan,
        "informative_pair_count": informative,
        "tie_count": int(part["quality_tie"].sum()) if len(part) else 0,
        "tie_rate": float(part["quality_tie"].mean()) if len(part) else math.nan,
        "target7_confirmed_reopen_rate": float(part["target7_value"].mean()) if feature == F2 and len(part) else math.nan,
        "loss_confirmed_reopen_rate": float(part["loss_value"].mean()) if feature == F2 and len(part) else math.nan,
        "t7_minus_loss_reopen_rate": float(part["target7_value"].mean() - part["loss_value"].mean()) if feature == F2 and len(part) else math.nan,
        "quality_direction": "LOWER_IS_STRONGER; F2=0 means NOT_CONFIRMED, not proven no reopen",
    }


def build_pair_outputs(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_tables = {feature: _make_pairs(frame, feature) for feature in FIXED_PROXIES}
    rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    for feature, pairs in pair_tables.items():
        for period in PERIODS:
            rows.append(_pair_summary(pairs, feature, period, "ALL_CANDIDATES"))
        head = pairs[
            pairs["target7_s2_rank"].between(2, 6)
            & pairs["loss_s2_rank"].between(2, 6)
        ].copy()
        wrong = pairs[pairs["s2_wrong_pair"].eq(1)].copy()
        for period in PERIODS:
            head_rows.append(_pair_summary(head, feature, period, "S2_RANK2_6"))
            summary = _pair_summary(wrong, feature, period, "S2_WRONG_PAIR")
            summary["wrong_pair_count"] = summary.pop("pair_count")
            summary["error_rescue_rate"] = summary["quality_oriented_concordance"]
            rescue_rows.append(summary)
    return pair_tables, pd.DataFrame(rows), pd.DataFrame(head_rows), pd.DataFrame(rescue_rows)


def _monthly_stability(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FIXED_PROXIES:
        values = {
            month: float(pair_summary.loc[
                pair_summary["feature"].eq(feature) & pair_summary["period"].eq(month),
                "quality_oriented_concordance",
            ].iloc[0])
            for month in MONTHS
        }
        finite = [value for value in values.values() if np.isfinite(value)]
        flip = bool(finite and max(finite) > 0.55 and min(finite) < 0.45)
        may_june_same = (
            (values["MAY"] > 0.55 and values["JUNE"] > 0.55)
            or (values["MAY"] < 0.45 and values["JUNE"] < 0.45)
        )
        july_opposite = (
            values["JULY"] < 0.45 if values["MAY"] > 0.55 and values["JUNE"] > 0.55
            else values["JULY"] > 0.55 if values["MAY"] < 0.45 and values["JUNE"] < 0.45
            else False
        )
        rows.append(
            {
                "feature": feature,
                "proxy": PROXY_SHORT[feature],
                "may_quality_concordance": values["MAY"],
                "june_quality_concordance": values["JUNE"],
                "july_quality_concordance": values["JULY"],
                "minimum": min(finite) if finite else math.nan,
                "maximum": max(finite) if finite else math.nan,
                "range": max(finite) - min(finite) if finite else math.nan,
                "temporal_direction_flip": "YES" if flip else "NO",
                "may_june_same_strong_direction": "YES" if may_june_same else "NO",
                "july_opposite_to_may_june": "YES" if july_opposite else "NO",
                "source_effect_risk": "TEMPORAL_OR_SOURCE_CONFOUNDING" if may_june_same and july_opposite else "PRESENT_BUT_NOT_DETERMINATIVE",
            }
        )
    return pd.DataFrame(rows)


def build_rankwise(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        period_frame = _period_part(frame, period)
        for rank in (1, 2, 3):
            selected = period_frame[period_frame["s2_rank"].eq(rank)]
            for feature in FIXED_PROXIES:
                valid = selected[selected[VALID_COLUMNS[feature]].eq(1)]
                rows.append(
                    {
                        "period": period,
                        "selection": f"RANK{rank}",
                        "feature": feature,
                        "proxy": PROXY_SHORT[feature],
                        "rows": len(selected),
                        "dates": int(selected["signal_date"].nunique()),
                        "target7_count": int(selected["target7"].sum()),
                        "target7_rate": float(selected["target7"].mean()) if len(selected) else math.nan,
                        "loss_count": int(selected["loss"].sum()),
                        "loss_rate": float(selected["loss"].mean()) if len(selected) else math.nan,
                        "severe_loss_count": int(selected["severe_loss"].sum()),
                        "severe_loss_rate": float(selected["severe_loss"].mean()) if len(selected) else math.nan,
                        "proxy_valid_rows": len(valid),
                        "proxy_missing_rate": 1.0 - len(valid) / len(selected) if len(selected) else math.nan,
                        "target7_proxy_mean": float(valid.loc[valid["target7"].eq(1), feature].mean()),
                        "loss_proxy_mean": float(valid.loc[valid["loss"].eq(1), feature].mean()),
                        "quality_direction": "LOWER_IS_STRONGER",
                    }
                )
    return pd.DataFrame(rows)


def build_board_robustness(pair_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, pairs in pair_tables.items():
        for board in ("BOARD2", "BOARD3"):
            scoped = pairs[
                pairs["target7_board_group"].eq(board)
                & pairs["loss_board_group"].eq(board)
            ].copy()
            for period in PERIODS:
                row = _pair_summary(scoped, feature, period, f"{board}_ONLY")
                row["support_warning"] = "LOW_N" if board == "BOARD3" else ""
                rows.append(row)
    return pd.DataFrame(rows)


def build_source_robustness(
    frame: pd.DataFrame, pair_tables: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    material_flags: dict[str, bool] = {}
    for feature in FIXED_PROXIES:
        source_col = "f3_source_group" if feature == F3 else "d0_source"
        groups = ("BAOSTOCK_BOTH", "SINA_BOTH") if feature == F3 else ("BAOSTOCK", "SINA")
        summaries: dict[str, dict[str, float]] = {}
        for source in groups:
            part = frame[frame[source_col].eq(source)]
            values = part.loc[part[VALID_COLUMNS[feature]].eq(1), feature].astype(float)
            summaries[source] = {
                "median": float(values.median()) if len(values) else math.nan,
                "p25": float(values.quantile(.25)) if len(values) else math.nan,
                "p75": float(values.quantile(.75)) if len(values) else math.nan,
                "rate": float(values.mean()) if feature == F2 and len(values) else math.nan,
            }
            rows.append(
                {
                    "section": "OUTCOME_BLIND_VALUE_DISTRIBUTION",
                    "feature": feature,
                    "proxy": PROXY_SHORT[feature],
                    "period": "POOLED",
                    "source_group": source,
                    "rows": len(part),
                    "valid_rows": len(values),
                    "missing_rate": 1.0 - len(values) / len(part) if len(part) else math.nan,
                    "median": summaries[source]["median"],
                    "p25": summaries[source]["p25"],
                    "p75": summaries[source]["p75"],
                    "confirmed_reopen_rate": summaries[source]["rate"],
                    "pair_count": math.nan,
                    "date_count": math.nan,
                    "quality_oriented_concordance": math.nan,
                }
            )
        first, second = groups
        pooled_values = frame.loc[frame[VALID_COLUMNS[feature]].eq(1), feature].astype(float)
        if feature == F2:
            magnitude = abs(summaries[first]["rate"] - summaries[second]["rate"])
            material = bool(np.isfinite(magnitude) and magnitude >= .10)
        else:
            iqr = float(pooled_values.quantile(.75) - pooled_values.quantile(.25)) if len(pooled_values) else math.nan
            magnitude = abs(summaries[first]["median"] - summaries[second]["median"])
            material = bool(np.isfinite(iqr) and iqr > 0 and magnitude / iqr >= .50)
        material_flags[feature] = material
        rows.append(
            {
                "section": "OUTCOME_BLIND_SOURCE_SHIFT_SUMMARY",
                "feature": feature,
                "proxy": PROXY_SHORT[feature],
                "period": "POOLED",
                "source_group": f"{second}_MINUS_{first}",
                "rows": len(frame),
                "valid_rows": int(frame[VALID_COLUMNS[feature]].sum()),
                "missing_rate": float(1.0 - frame[VALID_COLUMNS[feature]].mean()),
                "median": magnitude,
                "p25": math.nan,
                "p75": math.nan,
                "confirmed_reopen_rate": magnitude if feature == F2 else math.nan,
                "pair_count": math.nan,
                "date_count": math.nan,
                "quality_oriented_concordance": math.nan,
                "measurement_shift_material": "YES" if material else "NO",
            }
        )

        pairs = pair_tables[feature]
        for period in ("MAY_JUNE", "MAY", "JUNE", "JULY"):
            part = pairs[pairs["month"].isin(["MAY", "JUNE"])] if period == "MAY_JUNE" else pairs[pairs["month"].eq(period)]
            summary = _pair_summary(part, feature, "POOLED", "TIME_BLOCK")
            rows.append(
                {
                    "section": "OUTCOME_PAIR_TIME_BLOCK",
                    "feature": feature,
                    "proxy": PROXY_SHORT[feature],
                    "period": period,
                    "source_group": "MONTH_SOURCE_CONFOUNDED",
                    "rows": math.nan,
                    "valid_rows": math.nan,
                    "missing_rate": math.nan,
                    "median": math.nan,
                    "p25": math.nan,
                    "p75": math.nan,
                    "confirmed_reopen_rate": math.nan,
                    "pair_count": summary["pair_count"],
                    "date_count": summary["date_count"],
                    "quality_oriented_concordance": summary["quality_oriented_concordance"],
                    "measurement_shift_material": "",
                }
            )
    return pd.DataFrame(rows), {"material_flags": material_flags}


def build_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FIXED_PROXIES:
        for availability, mask in (
            ("AVAILABLE", frame[VALID_COLUMNS[feature]].eq(1)),
            ("UNAVAILABLE", frame[VALID_COLUMNS[feature]].eq(0)),
        ):
            part = frame[mask]
            rows.append(
                {
                    "feature": feature,
                    "proxy": PROXY_SHORT[feature],
                    "availability": availability,
                    "rows": len(part),
                    "dates": int(part["signal_date"].nunique()),
                    "target7_rate": float(part["target7"].mean()) if len(part) else math.nan,
                    "loss_rate": float(part["loss"].mean()) if len(part) else math.nan,
                    "severe_loss_rate": float(part["severe_loss"].mean()) if len(part) else math.nan,
                    "may_share": float(part["month"].eq("MAY").mean()) if len(part) else math.nan,
                    "june_share": float(part["month"].eq("JUNE").mean()) if len(part) else math.nan,
                    "july_share": float(part["month"].eq("JULY").mean()) if len(part) else math.nan,
                    "board2_share": float(part["board_group"].eq("BOARD2").mean()) if len(part) else math.nan,
                    "board3_share": float(part["board_group"].eq("BOARD3").mean()) if len(part) else math.nan,
                    "s2_rank_median": float(part["s2_rank"].median()) if len(part) else math.nan,
                    "rank1_share": float(part["s2_rank"].eq(1).mean()) if len(part) else math.nan,
                    "rank2_share": float(part["s2_rank"].eq(2).mean()) if len(part) else math.nan,
                    "rank3_share": float(part["s2_rank"].eq(3).mean()) if len(part) else math.nan,
                    "rank2_6_share": float(part["s2_rank"].between(2, 6).mean()) if len(part) else math.nan,
                    "audit_role": "MISSINGNESS_BIAS_DIAGNOSTIC_ONLY",
                }
            )
    return pd.DataFrame(rows)


def _metric_from_pairs(pairs: pd.DataFrame, feature: str, metric: str) -> float:
    if pairs.empty:
        return math.nan
    if metric == "QUALITY_CONCORDANCE":
        if feature == F2:
            informative = pairs["quality_better"].sum() + pairs["quality_worse"].sum()
            return float(pairs["quality_better"].sum() / informative) if informative else math.nan
        return float(pairs["quality_concordance_value"].mean())
    if metric == "LOSS_MINUS_T7_REOPEN_RATE":
        return float(pairs["loss_value"].mean() - pairs["target7_value"].mean())
    raise ValueError(metric)


def _bootstrap_pairs(
    pairs: pd.DataFrame, feature: str, metric: str, seed: int
) -> tuple[float, float, float, float, float]:
    if pairs.empty:
        return (math.nan,) * 5
    dates = np.asarray(sorted(pairs["signal_date"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    by_date = {date: pairs[pairs["signal_date"].eq(date)] for date in dates}
    draws = rng.integers(0, len(dates), size=(BOOTSTRAP_REPETITIONS, len(dates)))
    # Aggregate sufficient statistics so repeated-date sampling remains a true
    # signal-date cluster bootstrap without materializing pair rows repeatedly.
    if metric == "QUALITY_CONCORDANCE" and feature != F2:
        sums = np.asarray([by_date[d]["quality_concordance_value"].sum() for d in dates], dtype=float)
        counts = np.asarray([len(by_date[d]) for d in dates], dtype=float)
        estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    elif metric == "QUALITY_CONCORDANCE":
        correct = np.asarray([by_date[d]["quality_better"].sum() for d in dates], dtype=float)
        informative = np.asarray([
            by_date[d]["quality_better"].sum() + by_date[d]["quality_worse"].sum()
            for d in dates
        ], dtype=float)
        numerator = correct[draws].sum(axis=1)
        denominator = informative[draws].sum(axis=1)
        estimates = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    else:
        loss_sum = np.asarray([by_date[d]["loss_value"].sum() for d in dates], dtype=float)
        loss_count = np.asarray([len(by_date[d]) for d in dates], dtype=float)
        t7_sum = np.asarray([by_date[d]["target7_value"].sum() for d in dates], dtype=float)
        t7_count = loss_count.copy()
        estimates = (
            loss_sum[draws].sum(axis=1) / loss_count[draws].sum(axis=1)
            - t7_sum[draws].sum(axis=1) / t7_count[draws].sum(axis=1)
        )
    observed = _metric_from_pairs(pairs, feature, metric)
    null = 0.5 if metric == "QUALITY_CONCORDANCE" else 0.0
    return (
        observed,
        float(np.nanmedian(estimates)),
        float(np.nanpercentile(estimates, 2.5)),
        float(np.nanpercentile(estimates, 97.5)),
        float(np.nanmean(estimates > null)),
    )


def build_bootstrap(pair_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counter = 0
    for feature, pairs in pair_tables.items():
        endpoint_frames = {
            "PRIMARY": pairs,
            "S2_WRONG_PAIR_RESCUE": pairs[pairs["s2_wrong_pair"].eq(1)],
            "S2_RANK2_6": pairs[
                pairs["target7_s2_rank"].between(2, 6)
                & pairs["loss_s2_rank"].between(2, 6)
            ],
        }
        for endpoint, endpoint_pairs in endpoint_frames.items():
            endpoint_periods = PERIODS if endpoint != "S2_RANK2_6" else ("POOLED",)
            metrics = ("QUALITY_CONCORDANCE", "LOSS_MINUS_T7_REOPEN_RATE") if feature == F2 else ("QUALITY_CONCORDANCE",)
            for period in endpoint_periods:
                part = _period_part(endpoint_pairs, period)
                for metric in metrics:
                    seed = BOOTSTRAP_SEED + counter
                    counter += 1
                    observed, median, p025, p975, probability = _bootstrap_pairs(part, feature, metric, seed)
                    rows.append(
                        {
                            "feature": feature,
                            "proxy": PROXY_SHORT[feature],
                            "endpoint": endpoint,
                            "period": period,
                            "metric": metric,
                            "pair_count": len(part),
                            "date_count": int(part["signal_date"].nunique()) if len(part) else 0,
                            "observed": observed,
                            "bootstrap_median": median,
                            "p2_5": p025,
                            "p97_5": p975,
                            "p_above_null": probability,
                            "null_value": 0.5 if metric == "QUALITY_CONCORDANCE" else 0.0,
                            "sampling_unit": "SIGNAL_DATE",
                            "repetitions": BOOTSTRAP_REPETITIONS,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def _permuted_effect(day_values: np.ndarray, labels: np.ndarray, feature_index: int) -> tuple[float, float]:
    winners = day_values[labels == 1, feature_index]
    losses = day_values[labels == 0, feature_index]
    winners = winners[np.isfinite(winners)]
    losses = losses[np.isfinite(losses)]
    if not len(winners) or not len(losses):
        return 0.0, 0.0
    if feature_index == 1:  # F2
        better = float(((winners[:, None] == 0.0) & (losses[None, :] == 1.0)).sum())
        worse = float(((winners[:, None] == 1.0) & (losses[None, :] == 0.0)).sum())
        return better, better + worse
    differences = losses[None, :] - winners[:, None]
    return float((differences > 0).sum() + .5 * (differences == 0).sum()), float(differences.size)


def build_family_permutation(
    frame: pd.DataFrame, pair_tables: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    binary = frame[frame["target7"].eq(1) | frame["loss"].eq(1)].copy()
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, day in binary.groupby("signal_date", sort=True):
        labels = day["target7"].to_numpy(int)
        values = day[list(FIXED_PROXIES)].to_numpy(float)
        groups.append((labels, values))
    observed = {
        feature: _metric_from_pairs(pair_tables[feature], feature, "QUALITY_CONCORDANCE")
        for feature in FIXED_PROXIES
    }
    rng = np.random.default_rng(PERMUTATION_SEED)
    max_abs = np.empty(PERMUTATION_REPETITIONS, dtype=float)
    for repetition in range(PERMUTATION_REPETITIONS):
        numerators = np.zeros(3, dtype=float)
        denominators = np.zeros(3, dtype=float)
        for labels, values in groups:
            permuted = rng.permutation(labels)
            for feature_index in range(3):
                numerator, denominator = _permuted_effect(values, permuted, feature_index)
                numerators[feature_index] += numerator
                denominators[feature_index] += denominator
        concordance = np.divide(
            numerators, denominators,
            out=np.full(3, .5, dtype=float), where=denominators > 0,
        )
        max_abs[repetition] = float(np.max(np.abs(concordance - .5)))
    rows: list[dict[str, Any]] = []
    for feature in FIXED_PROXIES:
        deviation = abs(observed[feature] - .5) if np.isfinite(observed[feature]) else math.nan
        rows.append(
            {
                "row_type": "SUMMARY",
                "permutation_index": math.nan,
                "feature": feature,
                "proxy": PROXY_SHORT[feature],
                "observed_quality_concordance": observed[feature],
                "observed_abs_effect": deviation,
                "family_selection_adjusted_p": float(np.mean(max_abs >= deviation)) if np.isfinite(deviation) else math.nan,
                "max_abs_effect": math.nan,
                "null_p50": float(np.percentile(max_abs, 50)),
                "null_p95": float(np.percentile(max_abs, 95)),
                "eligible_proxy_count": 3,
                "permutations": PERMUTATION_REPETITIONS,
                "seed": PERMUTATION_SEED,
                "permutation_rule": "WITHIN_SIGNAL_DATE_LABEL_IDENTITY; PRESERVE_T7_LOSS_COUNTS",
            }
        )
    rows.extend(
        {
            "row_type": "PERMUTATION",
            "permutation_index": index + 1,
            "feature": "",
            "proxy": "",
            "observed_quality_concordance": math.nan,
            "observed_abs_effect": math.nan,
            "family_selection_adjusted_p": math.nan,
            "max_abs_effect": value,
            "null_p50": math.nan,
            "null_p95": math.nan,
            "eligible_proxy_count": 3,
            "permutations": PERMUTATION_REPETITIONS,
            "seed": PERMUTATION_SEED,
            "permutation_rule": "WITHIN_SIGNAL_DATE_LABEL_IDENTITY; PRESERVE_T7_LOSS_COUNTS",
        }
        for index, value in enumerate(max_abs)
    )
    return pd.DataFrame(rows)


def _lookup(table: pd.DataFrame, feature: str, period: str, value: str) -> float:
    row = table[table["feature"].eq(feature) & table["period"].eq(period)]
    return float(row.iloc[0][value]) if len(row) == 1 else math.nan


def classify_results(
    pair_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    head: pd.DataFrame,
    rescue: pd.DataFrame,
    bootstrap: pd.DataFrame,
    permutation: pd.DataFrame,
    source_context: Mapping[str, Any],
) -> tuple[dict[str, str], str, str, str, str, str]:
    proxy_states: dict[str, str] = {}
    for feature in FIXED_PROXIES:
        primary = _lookup(pair_summary, feature, "POOLED", "quality_oriented_concordance")
        primary_pairs = int(_lookup(pair_summary, feature, "POOLED", "pair_count"))
        primary_dates = int(_lookup(pair_summary, feature, "POOLED", "date_count"))
        informative = int(_lookup(pair_summary, feature, "POOLED", "informative_pair_count"))
        head_value = _lookup(head, feature, "POOLED", "quality_oriented_concordance")
        rescue_value = _lookup(rescue, feature, "POOLED", "error_rescue_rate")
        stability = monthly[monthly["feature"].eq(feature)].iloc[0]
        perm = permutation[
            permutation["row_type"].eq("SUMMARY") & permutation["feature"].eq(feature)
        ].iloc[0]
        primary_boot = bootstrap[
            bootstrap["feature"].eq(feature)
            & bootstrap["endpoint"].eq("PRIMARY")
            & bootstrap["period"].eq("POOLED")
            & bootstrap["metric"].eq("QUALITY_CONCORDANCE")
        ].iloc[0]
        rescue_boot = bootstrap[
            bootstrap["feature"].eq(feature)
            & bootstrap["endpoint"].eq("S2_WRONG_PAIR_RESCUE")
            & bootstrap["period"].eq("POOLED")
            & bootstrap["metric"].eq("QUALITY_CONCORDANCE")
        ].iloc[0]
        insufficient = (
            primary_pairs < 30 or primary_dates < MIN_PAIR_DATES
            or (feature == F2 and informative < MIN_F2_INFORMATIVE_PAIRS)
        )
        if insufficient:
            state = "INSUFFICIENT_SAMPLE"
        elif stability["may_june_same_strong_direction"] == "YES" and stability["july_opposite_to_may_june"] == "YES":
            state = "SOURCE_CONFOUNDED"
        elif stability["temporal_direction_flip"] == "YES":
            state = "TEMPORALLY_UNSTABLE"
        else:
            monthly_values = [
                float(stability["may_quality_concordance"]),
                float(stability["june_quality_concordance"]),
                float(stability["july_quality_concordance"]),
            ]
            temporal_support = min(monthly_values) >= .48 and sum(value >= .52 for value in monthly_values) >= 2
            boot_support = float(primary_boot["p2_5"]) > .48 and float(rescue_boot["p2_5"]) > .48
            selection_support = float(perm["family_selection_adjusted_p"]) <= SELECTION_ADJUSTED_P_GATE
            if (
                primary >= MATERIAL_PRIMARY and head_value >= MATERIAL_HEAD
                and rescue_value >= MATERIAL_RESCUE and temporal_support
                and boot_support and selection_support
            ):
                state = "STABLE_INCREMENTAL_INFORMATION"
            elif primary >= MATERIAL_PRIMARY and .475 <= rescue_value <= .525:
                state = "REDUNDANT_WITH_S2"
            elif max(primary, head_value, rescue_value) >= PARTIAL_SIGNAL:
                state = "PARTIAL_INFORMATION"
            else:
                state = "NO_INFORMATION"
        proxy_states[feature] = state

    temporal_source_confound = any(value == "SOURCE_CONFOUNDED" for value in proxy_states.values())
    measurement_material = any(source_context["material_flags"].values())
    source_shift_impact = (
        "UNRESOLVED" if temporal_source_confound
        else "MATERIAL" if measurement_material
        else "LOW"
    )
    if "STABLE_INCREMENTAL_INFORMATION" in proxy_states.values():
        family = "PREBREAK_5M_SEAL_PATH_INFORMATION_SUPPORTED"
        broader = "NOT_YET_ESTABLISHED"
        research_state = "ONE_CONFIRMATION_EXPERIMENT_JUSTIFIED"
        next_action = "FREEZE_SINGLE_SEAL_PATH_RELATION_FOR_CONFIRMATION"
    elif temporal_source_confound:
        family = "PREBREAK_5M_SEAL_PATH_INFORMATION_INCONCLUSIVE"
        broader = "[待核验]"
        research_state = "STOP_AND_REVIEW"
        next_action = "STOP_AND_REVIEW"
    elif any(value in {"PARTIAL_INFORMATION", "REDUNDANT_WITH_S2"} for value in proxy_states.values()):
        family = "PREBREAK_5M_SEAL_PATH_INFORMATION_PARTIAL"
        broader = "[待核验]"
        research_state = "STOP_AND_REVIEW"
        next_action = "STOP_AND_REVIEW"
    else:
        family = "PREBREAK_5M_SEAL_PATH_INFORMATION_NOT_SUPPORTED"
        broader = "SUPPORTED"
        research_state = "CURRENT_D1_INFORMATION_RESEARCH_EXHAUSTED"
        next_action = "STOP_STAGE1_FACTOR_MINING_AND_REVIEW_PIPELINE"
    rescue_found = (
        "YES" if "STABLE_INCREMENTAL_INFORMATION" in proxy_states.values()
        else "PARTIAL" if any(_lookup(rescue, f, "POOLED", "error_rescue_rate") >= PARTIAL_SIGNAL for f in FIXED_PROXIES)
        else "NO"
    )
    head_found = (
        "YES" if any(_lookup(head, f, "POOLED", "quality_oriented_concordance") >= MATERIAL_PRIMARY for f in FIXED_PROXIES)
        else "PARTIAL" if any(_lookup(head, f, "POOLED", "quality_oriented_concordance") >= MATERIAL_HEAD for f in FIXED_PROXIES)
        else "NO"
    )
    return proxy_states, family, source_shift_impact, rescue_found, head_found, broader + "|" + research_state + "|" + next_action


def render_review(context: Mapping[str, Any]) -> str:
    pair_summary = context["pair_concordance"]
    head = context["rank2_6"]
    rescue = context["wrong_pair_rescue"]
    monthly = context["monthly_stability"]
    proxy_states = context["proxy_states"]
    family = context["family_state"]
    source_impact = context["source_shift_impact"]
    rescue_found = context["rescue_found"]
    head_found = context["head_found"]
    state_parts = context["state_parts"]
    lines = [
        "# v004c Blind 5m Seal-path Information Audit v001",
        "",
        "## 简单结论",
        "",
    ]
    pooled_bits = []
    rescue_bits = []
    head_bits = []
    month_bits = []
    for feature in FIXED_PROXIES:
        short = PROXY_SHORT[feature]
        pooled_bits.append(f"{short}={_lookup(pair_summary, feature, 'POOLED', 'quality_oriented_concordance'):.3f}")
        rescue_bits.append(f"{short}={_lookup(rescue, feature, 'POOLED', 'error_rescue_rate'):.3f}")
        head_bits.append(f"{short}={_lookup(head, feature, 'POOLED', 'quality_oriented_concordance'):.3f}")
        row = monthly[monthly["feature"].eq(feature)].iloc[0]
        month_bits.append(
            f"{short}: {row['may_quality_concordance']:.3f}/{row['june_quality_concordance']:.3f}/{row['july_quality_concordance']:.3f}"
        )
    answers = (
        ("Q1. Target7 和 LOSS 在断板前封板路径上到底有没有区别？",
         "同日质量方向 concordance 为 " + ", ".join(pooled_bits) + "。正式判断必须结合头部、错配救援和时间稳定性，不能只看 pooled。"),
        ("Q2. 这个区别是 May/June/July 都存在，还是只在某个月？",
         "逐月 May/June/July 分别为 " + "; ".join(month_bits) + "。任何 >0.55 与 <0.45 的跨月组合均已标记方向翻转。"),
        ("Q3. 当 S2 自己排错时，F1/F2/F3 谁还能把 winner 和 LOSS 分开？",
         "S2 wrong-pair rescue 为 " + ", ".join(rescue_bits) + f"；正式 rescue 状态={rescue_found}。"),
        ("Q4. Rank2/Rank3 是否真正得到额外信息？",
         "固定 Rank2-6 concordance 为 " + ", ".join(head_bits) + f"；正式增量状态={head_found}。Rank1 仅作描述，不主导结论。"),
        ("Q5. July 与 May/June 差异有多少可能来自 Sina vs BaoStock？",
         f"上一轮已预注册 source shift；本轮盲分布与月份方向联合判断为 SOURCE_SHIFT_IMPACT={source_impact}。没有按 source 重标化或改 proxy。"),
        ("Q6. 这是真正新增信息，还是又一个 development 局部关系？",
         f"Family 正式状态={family}。三个 proxy 逐项状态：" + ", ".join(f"{PROXY_SHORT[k]}={v}" for k, v in proxy_states.items()) + "。"),
        ("Q7. 当前 Stage1 D1 因子研究：继续一次 confirmation 还是正式停止？",
         state_parts[2]),
    )
    for question, answer in answers:
        lines.extend([f"### {question}", "", answer, ""])
    lines.extend(
        [
            "## 审计边界",
            "",
            "- Population: 485 rows / 60 dates, 2026-05-06 through 2026-07-29.",
            "- Labels: only rows with label_available_date < 2026-08-01.",
            "- Proxies: exact locked F1/F2/F3; no reconstruction change and no F4.",
            "- Continuous concordance uses 0.5 for exact ties and also exports the strict-better rate; F2 concordance excludes ties.",
            "- S2 score comes from archived coefficients; learner fit count is zero.",
            "- No raw/capped return is loaded into the output contract or tested.",
            "- Bootstrap and permutation both use signal_date as the cluster/permutation boundary.",
            "",
            "## Final State",
            "",
            f"F1_INFORMATION_STATE = {proxy_states[F1]}",
            "",
            f"F2_INFORMATION_STATE = {proxy_states[F2]}",
            "",
            f"F3_INFORMATION_STATE = {proxy_states[F3]}",
            "",
            f"PREBREAK_5M_SEAL_PATH_INFORMATION_STATE = {family}",
            "",
            f"SOURCE_SHIFT_IMPACT = {source_impact}",
            "",
            f"S2_WRONG_PAIR_RESCUE_FOUND = {rescue_found}",
            "",
            f"RANK2_6_INCREMENTAL_INFORMATION = {head_found}",
            "",
            f"BROADER_D1_INFORMATION_LIMITATION = {state_parts[0]}",
            "",
            f"STAGE1_D1_INFORMATION_RESEARCH_STATE = {state_parts[1]}",
            "",
            "MODEL_TRAINED = NO",
            "",
            "FEATURE_SEARCH = NO",
            "",
            "WINDOW_SEARCH = NO",
            "",
            "AUGUST_HOLDOUT_STATUS = CONSUMED",
            "",
            "AUGUST_USED_IN_PRIMARY_ANALYSIS = NO",
            "",
            f"NEXT_ACTION = {state_parts[2]}",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(root: str | Path) -> dict[str, Any]:
    frame, audit = load_analysis_population(root)
    pair_tables, pair_summary, rank2_6, wrong_rescue = build_pair_outputs(frame)
    monthly = _monthly_stability(pair_summary)
    rankwise = build_rankwise(frame)
    board = build_board_robustness(pair_tables)
    source, source_context = build_source_robustness(frame, pair_tables)
    missingness = build_missingness(frame)
    bootstrap = build_bootstrap(pair_tables)
    permutation = build_family_permutation(frame, pair_tables)
    proxy_states, family, source_impact, rescue_found, head_found, packed = classify_results(
        pair_summary, monthly, rank2_6, wrong_rescue, bootstrap, permutation, source_context
    )
    state_parts = packed.split("|", 2)
    population_cols = [
        "event_id", "code", "signal_date", "month", "d0_date", "d1_date", "prev_board_day",
        "board_group", "label_available_date", "target7", "loss", "severe_loss",
        "s2_score", "s2_rank", "d0_source", "prev_source", "f3_source_group",
        *FIXED_PROXIES, *VALID_COLUMNS.values(),
    ]
    feature_cols = [
        "event_id", "code", "signal_date", "month", "d0_date", "prev_board_day", "d1_date",
        "board_group", "d0_cache_kind", "prev_cache_kind", "d0_source", "prev_source",
        "f3_source_group", *FIXED_PROXIES, *VALID_COLUMNS.values(),
        "F1_missing_reason", "F2_missing_reason", "F3_missing_reason",
    ]
    context: dict[str, Any] = {
        "audit": audit,
        "analysis_population": frame[population_cols].copy(),
        "feature_values": frame[feature_cols].copy(),
        "pair_concordance": pair_summary,
        "monthly_stability": monthly,
        "rankwise": rankwise,
        "rank2_6": rank2_6,
        "wrong_pair_rescue": wrong_rescue,
        "source_robustness": source,
        "board_robustness": board,
        "missingness": missingness,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "proxy_states": proxy_states,
        "family_state": family,
        "source_shift_impact": source_impact,
        "rescue_found": rescue_found,
        "head_found": head_found,
        "state_parts": state_parts,
    }
    context["review"] = render_review(context)
    return context


def build_outputs(context: Mapping[str, Any]) -> dict[str, bytes]:
    tables = (
        "analysis_population", "feature_values", "pair_concordance", "monthly_stability",
        "rankwise", "rank2_6", "wrong_pair_rescue", "source_robustness",
        "board_robustness", "missingness", "bootstrap", "permutation",
    )
    outputs = {
        OUTPUT_FILENAMES[index]: _csv_bytes(context[name])
        for index, name in enumerate(tables)
    }
    outputs[OUTPUT_FILENAMES[-1]] = str(context["review"]).encode("utf-8")
    return outputs


def write_outputs(
    context: Mapping[str, Any], output_dir: str | Path | None = None
) -> Path:
    root = Path(__file__).resolve().parents[1]
    target = Path(output_dir).resolve() if output_dir is not None else (
        root / "reports" / "research" / OUTPUT_DIRNAME
    )
    first = build_outputs(context)
    second = build_outputs(context)
    if first != second:
        raise RuntimeError("FATAL: deterministic output serialization failed")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (target / name).write_bytes(payload)
    return target


def output_sha256s(output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }


__all__ = [
    "BOOTSTRAP_REPETITIONS", "BOOTSTRAP_SEED", "F1", "F2", "F3",
    "FIXED_PROXIES", "OUTPUT_FILENAMES", "PERMUTATION_REPETITIONS",
    "PERMUTATION_SEED", "TASK_NAME", "_make_pairs", "_monthly_stability",
    "analyze", "assert_contract", "build_family_permutation", "load_analysis_population",
    "output_sha256s", "write_outputs",
]

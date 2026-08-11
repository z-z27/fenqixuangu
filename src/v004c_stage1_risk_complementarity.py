from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v004c_pair_capped7_july_forward import (
    _fit_stage1_snapshot,
    _read_daily,
    _score_stage1,
    attach_training_outcomes,
    capped7_utility,
    derive_trade_dates,
    label_is_available,
    load_authoritative_d1_safe_base,
    prepare_frozen_x,
)
from .v004c_top10_target_information import (
    PAIR_FEATURE_COLUMNS,
    PAIR_L2,
    build_same_date_pairs,
    fit_pairwise_weighted_ridge,
)
from .v004c_v4a_architecture_transfer import (
    FROZEN_FEATURE_COLUMNS,
    INITIAL_TRAIN_DATES,
    L2 as STAGE1_L2,
    POSITIVE_WEIGHT as STAGE1_POSITIVE_WEIGHT,
    build_exact_raw_features,
    dataframe_csv_bytes,
)
from .v004c_v4a_top10_reranker import (
    RESIDUAL_RAW_FIELDS,
    TOP_K,
    build_stage2_features,
    strategy_summary,
)


TASK_NAME = "JUNE-ONLY ARCHITECTURE DIAGNOSTIC"
JUNE_START = "2026-06-01"
JUNE_END = "2026-06-30"
JULY_SENTINEL = "2026-07-01"
MIN_STAGE1_TRAIN_DATES = INITIAL_TRAIN_DATES
MIN_CROSSFIT_BASE_DATES = INITIAL_TRAIN_DATES - 1
JULY_RESULT_ROWS_ACCESSED = 0
NEW_MODEL_DEVELOPED = False
NEW_FEATURES = False
THRESHOLD_SEARCH = False
HYPERPARAMETER_SEARCH = False

POLICY_STAGE1 = "STRICT_STAGE1"
POLICY_CAPPED = "STRICT_CAPPED"
POLICY_WINNER = "CAPPED_WITH_ORACLE_WINNER_PROTECTION"
POLICY_LOSS_PERMISSION = "CAPPED_WITH_ORACLE_LOSS_PERMISSION"
POLICY_STAGE1_BACKFILL = "STAGE1_WITH_ORACLE_LOSS_VETO_STAGE1_BACKFILL"
POLICY_ORACLE_BACKFILL = "STAGE1_WITH_ORACLE_RISK_AND_ORACLE_BACKFILL"
POLICY_TOP10_ORACLE = "TOP10_ORACLE"

POLICIES = (
    POLICY_STAGE1,
    POLICY_CAPPED,
    POLICY_WINNER,
    POLICY_LOSS_PERMISSION,
    POLICY_STAGE1_BACKFILL,
    POLICY_ORACLE_BACKFILL,
    POLICY_TOP10_ORACLE,
)

OUTPUT_FILENAMES = (
    "v004c_stage1_risk_strict_oof_v001.csv",
    "v004c_stage1_risk_temporal_audit_v001.csv",
    "v004c_stage1_risk_swap_attribution_v001.csv",
    "v004c_stage1_risk_oracle_daily_v001.csv",
    "v004c_stage1_risk_practical_v001.csv",
    "v004c_stage1_risk_review_v001.md",
)


def assert_diagnostic_contract() -> None:
    if len(FROZEN_FEATURE_COLUMNS) != 18:
        raise RuntimeError("FATAL: frozen V4C_STAGE1 feature count changed")
    if STAGE1_L2 != 0.30 or STAGE1_POSITIVE_WEIGHT != 1.50:
        raise RuntimeError("FATAL: frozen V4C_STAGE1 hyperparameters changed")
    if PAIR_FEATURE_COLUMNS != [
        "stage1_strength", "closing_completion_gap", "strength_x_gap"
    ]:
        raise RuntimeError("FATAL: frozen PAIR_CAPPED7 predictors changed")
    if PAIR_L2 != 0.30 or TOP_K != 10:
        raise RuntimeError("FATAL: frozen PAIR_CAPPED7 contract changed")
    if any((NEW_MODEL_DEVELOPED, NEW_FEATURES, THRESHOLD_SEARCH, HYPERPARAMETER_SEARCH)):
        raise RuntimeError("FATAL: diagnostic scope expanded into model development")
    if JULY_RESULT_ROWS_ACCESSED != 0:
        raise RuntimeError("FATAL: July result data entered June architecture diagnostic")


def outcome_state(raw_return: float) -> str:
    value = float(raw_return)
    if value >= 0.07:
        return "TARGET7"
    if value >= 0.0:
        return "POSITIVE_NON_TARGET"
    return "LOSS"


def _load_date_contract(root: Path, base: pd.DataFrame) -> pd.DataFrame:
    date_only = derive_trade_dates(root, base)
    authority_path = (
        root / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv"
    )
    authority = pd.read_csv(
        authority_path,
        usecols=["event_id", "label_d2_date", "label_d3_date"],
        encoding="utf-8-sig",
    ).rename(columns={"label_d2_date": "authority_d2", "label_d3_date": "authority_d3"})
    checked = date_only.merge(authority, on="event_id", validate="one_to_one")
    for column in ("d2_date", "d3_date", "label_available_date"):
        checked[column] = pd.to_datetime(
            checked[column].astype(str), errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        if checked[column].isna().any():
            raise RuntimeError(f"FATAL: invalid canonical date value in {column}")
    for column in ("authority_d2", "authority_d3"):
        checked[column] = pd.to_datetime(
            checked[column], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    # The authority may leave label dates blank when its own development-label
    # coverage gate was unavailable.  The stock-specific daily trading rows are
    # the canonical source of D2/D3; populated authority dates are parity checks.
    authority_complete = checked[["authority_d2", "authority_d3"]].notna().all(axis=1)
    if bool(checked.loc[authority_complete, "d2_date"].ne(
        checked.loc[authority_complete, "authority_d2"]
    ).any()):
        raise RuntimeError("FATAL: canonical D2 date mismatch")
    if bool(checked.loc[authority_complete, "d3_date"].ne(
        checked.loc[authority_complete, "authority_d3"]
    ).any()):
        raise RuntimeError("FATAL: canonical D3 date mismatch")
    if bool(checked["label_available_date"].ne(checked["d3_date"]).any()):
        raise RuntimeError("FATAL: label_available_date must equal D3 date")
    return checked[[
        "event_id", "signal_date", "d2_date", "d3_date", "label_available_date"
    ]].sort_values(["signal_date", "event_id"], kind="mergesort").reset_index(drop=True)


def load_pre_july_outcomes(
    root: Path, base: pd.DataFrame, date_contract: pd.DataFrame
) -> pd.DataFrame:
    identity = base[["event_id", "code", "signal_date"]].merge(
        date_contract, on=["event_id", "signal_date"], validate="one_to_one"
    )
    eligible = identity[identity["label_available_date"].lt(JULY_SENTINEL)].copy()
    rows: list[dict[str, Any]] = []
    accessed_dates: list[str] = []
    for row in eligible.itertuples(index=False):
        daily = _read_daily(root, str(row.code).zfill(6))
        d2 = daily[daily["date"].eq(row.d2_date)]
        d3 = daily[daily["date"].eq(row.d3_date)]
        if len(d2) != 1 or len(d3) != 1:
            raise RuntimeError(f"FATAL: canonical outcome row unavailable {row.event_id}")
        if row.d2_date >= JULY_SENTINEL or row.d3_date >= JULY_SENTINEL:
            raise RuntimeError("FATAL: July D2/D3 outcome selected")
        d2_open = float(d2.iloc[0]["open"])
        d3_high = float(d3.iloc[0]["high"])
        raw_return = d3_high / d2_open - 1.0
        rows.append({
            "event_id": row.event_id,
            "d2_date": row.d2_date,
            "d3_date": row.d3_date,
            "label_available_date": row.label_available_date,
            "d2_open_daily": d2_open,
            "d3_high_daily": d3_high,
            "raw_repair_return": raw_return,
            "capped_opportunity_return_7": float(capped7_utility(raw_return)),
            "target7": int(raw_return >= 0.07),
        })
        accessed_dates.extend([str(row.d2_date), str(row.d3_date)])
    outcomes = pd.DataFrame(rows).sort_values("event_id", kind="mergesort").reset_index(drop=True)
    if outcomes.empty or outcomes["event_id"].duplicated().any():
        raise RuntimeError("FATAL: pre-July outcome identity invalid")
    if accessed_dates and max(accessed_dates) >= JULY_SENTINEL:
        raise RuntimeError("FATAL: July outcome row accessed")
    expected = outcomes["d3_high_daily"] / outcomes["d2_open_daily"] - 1.0
    if not np.allclose(expected, outcomes["raw_repair_return"], rtol=0.0, atol=1e-12):
        raise RuntimeError("FATAL: raw outcome semantic mismatch")
    expected_cap = np.minimum(outcomes["raw_repair_return"].to_numpy(float), 0.07)
    if not np.allclose(
        expected_cap,
        outcomes["capped_opportunity_return_7"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("FATAL: capped return must be upper-cap only")
    return outcomes


def prepare_diagnostic_samples(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assert_diagnostic_contract()
    base = load_authoritative_d1_safe_base(root)
    raw, _, _, feature_gate = build_exact_raw_features(root, base)
    if not feature_gate["pass"]:
        raise RuntimeError("FATAL: frozen raw feature gate failed")
    base_x = prepare_frozen_x(raw)
    residual = base[["event_id", *RESIDUAL_RAW_FIELDS]].copy()
    for field in RESIDUAL_RAW_FIELDS:
        residual[field] = pd.to_numeric(residual[field], errors="coerce")
    base_x = base_x.merge(residual, on="event_id", validate="one_to_one")
    dates = _load_date_contract(root, base)
    outcomes = load_pre_july_outcomes(root, base, dates)
    samples = attach_training_outcomes(base_x, outcomes)
    audit = {
        "rows": int(len(base)),
        "dates": int(base["signal_date"].nunique()),
        "may_rows": int(base["signal_date"].str.startswith("2026-05").sum()),
        "may_dates": int(base.loc[base["signal_date"].str.startswith("2026-05"), "signal_date"].nunique()),
        "june_rows": int(base["signal_date"].str.startswith("2026-06").sum()),
        "june_dates": int(base.loc[base["signal_date"].str.startswith("2026-06"), "signal_date"].nunique()),
        "board2": int(base["board_streak_before_break"].eq(2).sum()),
        "board3": int(base["board_streak_before_break"].eq(3).sum()),
        "july_result_rows_accessed": JULY_RESULT_ROWS_ACCESSED,
    }
    if audit != {
        "rows": 319, "dates": 39, "may_rows": 146, "may_dates": 18,
        "june_rows": 173, "june_dates": 21, "board2": 261, "board3": 58,
        "july_result_rows_accessed": 0,
    }:
        raise RuntimeError(f"FATAL: authoritative universe mismatch {audit}")
    return base_x, samples, dates, audit


def _fit_strict_stage2(
    base_x: pd.DataFrame,
    matured: pd.DataFrame,
    test_date: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    matured_dates = sorted(matured["signal_date"].astype(str).unique())
    if len(matured_dates) < MIN_STAGE1_TRAIN_DATES:
        raise RuntimeError("FATAL: strict Stage2 history below frozen minimum")
    meta_parts: list[pd.DataFrame] = []
    self_label_rows = 0
    current_test_rows = 0
    for meta_date in matured_dates:
        base_train = matured[~matured["signal_date"].eq(meta_date)].copy()
        if base_train["signal_date"].nunique() < MIN_CROSSFIT_BASE_DATES:
            raise RuntimeError("FATAL: crossfit Stage1 history below frozen minimum")
        if bool(base_train["signal_date"].eq(meta_date).any()):
            self_label_rows += int(base_train["signal_date"].eq(meta_date).sum())
        if bool(base_train["label_available_date"].ge(test_date).any()):
            current_test_rows += int(base_train["label_available_date"].ge(test_date).sum())
        beta = _fit_stage1_snapshot(base_train)
        score_rows = base_x[base_x["signal_date"].eq(meta_date)].copy()
        scored = _score_stage1(score_rows, beta)
        top10 = build_stage2_features(scored[scored["stage1_top10"]].copy())
        labels = matured[[
            "event_id", "target7", "raw_repair_return",
            "capped_opportunity_return_7", "label_available_date",
        ]]
        top10 = top10.merge(labels, on="event_id", validate="one_to_one")
        if len(top10) != min(TOP_K, len(score_rows)):
            raise RuntimeError("FATAL: strict historical Top10 label coverage incomplete")
        if bool(top10["label_available_date"].ge(test_date).any()):
            current_test_rows += int(top10["label_available_date"].ge(test_date).sum())
        meta_parts.append(top10)
    meta = pd.concat(meta_parts, ignore_index=True)
    pairs = build_same_date_pairs(meta)
    x_columns = [f"x_{feature}" for feature in PAIR_FEATURE_COLUMNS]
    beta = fit_pairwise_weighted_ridge(
        pairs[x_columns].to_numpy(float),
        pairs["pair_y_capped7"].to_numpy(float),
        pairs["pair_weight"].to_numpy(float),
        l2=PAIR_L2,
    )
    return beta, meta, pairs, {
        "self_label_leakage_rows": self_label_rows,
        "current_test_leakage_rows": current_test_rows,
    }


def _score_strict_test(
    base_x: pd.DataFrame,
    outcomes: pd.DataFrame,
    matured: pd.DataFrame,
    test_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage1_beta = _fit_stage1_snapshot(matured)
    stage2_beta, meta, pairs, leakage = _fit_strict_stage2(base_x, matured, test_date)
    test_x = base_x[base_x["signal_date"].eq(test_date)].copy()
    scored = _score_stage1(test_x, stage1_beta)
    scored["capped_pair_score"] = np.nan
    scored["capped_pair_rank"] = np.nan
    top10 = build_stage2_features(scored[scored["stage1_top10"]].copy())
    top10["capped_pair_score"] = top10[PAIR_FEATURE_COLUMNS].to_numpy(float) @ stage2_beta
    top10 = top10.sort_values(
        ["capped_pair_score", "stage1_rank", "event_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top10["capped_pair_rank"] = np.arange(1, len(top10) + 1)
    values = top10[["event_id", "closing_completion_gap", "strength_x_gap", "capped_pair_score", "capped_pair_rank"]]
    scored = scored.drop(columns=[
        "closing_completion_gap", "strength_x_gap",
        "capped_pair_score", "capped_pair_rank",
    ], errors="ignore").merge(
        values, on="event_id", how="left", validate="one_to_one"
    )
    inside = scored[scored["stage1_top10"]].sort_values(
        ["capped_pair_rank", "stage1_rank", "event_id"], kind="mergesort"
    )
    outside = scored[~scored["stage1_top10"]].sort_values(
        ["stage1_rank", "event_id"], kind="mergesort"
    )
    ordered = pd.concat([inside, outside])
    final_rank = pd.Series(index=scored.index, dtype=int)
    final_rank.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    scored["final_capped_rank"] = final_rank.astype(int)
    outcome_columns = [
        "event_id", "d2_date", "d3_date", "label_available_date",
        "d2_open_daily", "d3_high_daily", "raw_repair_return",
        "capped_opportunity_return_7", "target7",
    ]
    test_outcomes = outcomes.loc[
        outcomes["event_id"].isin(scored["event_id"]), outcome_columns
    ].copy()
    if len(test_outcomes) != len(scored):
        raise RuntimeError("FATAL: strict June test outcome unavailable without July")
    scored = scored.merge(test_outcomes, on="event_id", validate="one_to_one")
    scored["outcome_state"] = scored["raw_repair_return"].map(outcome_state)
    if bool(scored["signal_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: non-June signal entered strict OOF")
    return scored, {
        "stage2_meta_rows": int(len(meta)),
        "stage2_meta_dates": int(meta["signal_date"].nunique()),
        "stage2_pair_count": int(len(pairs)),
        **leakage,
    }


def reconstruct_strict_june(
    base_x: pd.DataFrame,
    samples: pd.DataFrame,
    date_contract: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    june_dates = sorted(
        date for date in base_x["signal_date"].astype(str).unique()
        if JUNE_START <= date <= JUNE_END
    )
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    unavailable: list[str] = []
    counts = base_x.groupby("signal_date", sort=True)["event_id"].size().to_dict()
    for test_date in june_dates:
        matured = samples[samples["label_available_date"].lt(test_date)].copy()
        matured_dates = sorted(matured["signal_date"].astype(str).unique())
        test_contract = date_contract[date_contract["signal_date"].eq(test_date)]
        immature_rows = date_contract[
            date_contract["signal_date"].lt(test_date)
            & date_contract["label_available_date"].ge(test_date)
        ]
        reasons: list[str] = []
        if len(matured_dates) < MIN_STAGE1_TRAIN_DATES:
            reasons.append(f"MATURED_DATES_{len(matured_dates)}_LT_{MIN_STAGE1_TRAIN_DATES}")
        if test_contract.empty or bool(test_contract["label_available_date"].ge(JULY_SENTINEL).any()):
            reasons.append("JULY_OUTCOME_FORBIDDEN")
        available = not reasons
        row: dict[str, Any] = {
            "test_date": test_date,
            "strict_date_available": available,
            "unavailable_reason": "|".join(reasons),
            "matured_training_rows": int(len(matured)),
            "matured_training_dates": int(len(matured_dates)),
            "immature_rows_excluded": int(len(immature_rows)),
            "latest_training_signal_date": str(matured["signal_date"].max()) if len(matured) else "",
            "latest_training_label_available_date": str(matured["label_available_date"].max()) if len(matured) else "",
            "stage1_training_rows": int(len(matured)),
            "stage2_meta_rows": 0,
            "stage2_meta_dates": 0,
            "stage2_pair_count": 0,
            "self_label_leakage_rows": 0,
            "current_test_leakage_rows": 0,
            "july_rows_accessed": 0,
        }
        if not available:
            unavailable.append(test_date)
            audits.append(row)
            continue
        scored, fold = _score_strict_test(base_x, samples, matured, test_date)
        if len(scored) != counts[test_date]:
            raise RuntimeError("FATAL: strict prediction changed date universe")
        row.update(fold)
        predictions.append(scored)
        audits.append(row)
    if not predictions:
        raise RuntimeError("FATAL: no strict June date available")
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audits)
    if int(audit["self_label_leakage_rows"].sum()) != 0:
        raise RuntimeError("FATAL: self-label leakage")
    if int(audit["current_test_leakage_rows"].sum()) != 0:
        raise RuntimeError("FATAL: current-test leakage")
    if int(audit["july_rows_accessed"].sum()) != 0:
        raise RuntimeError("FATAL: July result rows accessed")
    if bool(oof["signal_date"].ge(JULY_SENTINEL).any()):
        raise RuntimeError("FATAL: July signal entered strict OOF")
    return oof, audit, unavailable


def _ordered_ids(day: pd.DataFrame, rank_column: str) -> list[str]:
    return day.sort_values([rank_column, "event_id"], kind="mergesort")["event_id"].astype(str).tolist()


def winner_protection_ids(day: pd.DataFrame, k: int | None = None) -> list[str]:
    size = min(3, len(day)) if k is None else int(k)
    control = day[day["stage1_rank"].le(size)].sort_values(["stage1_rank", "event_id"], kind="mergesort")
    protected = control[control["target7"].eq(1)]["event_id"].astype(str).tolist()
    final = list(protected)
    capped = day[day["stage1_top10"]].sort_values(["capped_pair_rank", "stage1_rank", "event_id"], kind="mergesort")
    for event_id in capped["event_id"].astype(str):
        if len(final) >= size:
            break
        if event_id not in final:
            final.append(event_id)
    return final


def loss_permission_ids(day: pd.DataFrame, k: int | None = None) -> list[str]:
    size = min(3, len(day)) if k is None else int(k)
    control = day[day["stage1_rank"].le(size)].sort_values(["stage1_rank", "event_id"], kind="mergesort")
    capped_ids = set(day[day["final_capped_rank"].le(size)]["event_id"].astype(str))
    approved = set(control[
        ~control["event_id"].astype(str).isin(capped_ids)
        & control["raw_repair_return"].lt(0.0)
    ]["event_id"].astype(str))
    final = [event_id for event_id in control["event_id"].astype(str) if event_id not in approved]
    capped = day[day["stage1_top10"]].sort_values(["capped_pair_rank", "stage1_rank", "event_id"], kind="mergesort")
    for event_id in capped["event_id"].astype(str):
        if len(final) >= size:
            break
        if event_id not in final:
            final.append(event_id)
    return final


def stage1_loss_veto_ids(
    day: pd.DataFrame, oracle_backfill: bool, k: int | None = None
) -> list[str]:
    size = min(3, len(day)) if k is None else int(k)
    control = day[day["stage1_rank"].le(size)].sort_values(["stage1_rank", "event_id"], kind="mergesort")
    final = control[control["raw_repair_return"].ge(0.0)]["event_id"].astype(str).tolist()
    original_control_ids = set(control["event_id"].astype(str))
    pool = day[
        day["stage1_rank"].le(min(TOP_K, len(day)))
        & ~day["event_id"].astype(str).isin(original_control_ids)
    ].copy()
    if oracle_backfill:
        pool = pool.sort_values(
            ["capped_opportunity_return_7", "event_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    else:
        pool = pool.sort_values(["stage1_rank", "event_id"], kind="mergesort")
    for event_id in pool["event_id"].astype(str):
        if len(final) >= size:
            break
        if event_id not in final:
            final.append(event_id)
    # A three-candidate date has no external replacement.  A veto cannot
    # reduce the required k=min(3,n) portfolio size, so unreplaced original
    # members remain in frozen Stage1 order.
    if len(final) < size:
        for event_id in control["event_id"].astype(str):
            if len(final) >= size:
                break
            if event_id not in final:
                final.append(event_id)
    return final


def top10_oracle_ids(day: pd.DataFrame, k: int | None = None) -> list[str]:
    size = min(3, len(day)) if k is None else int(k)
    return day[day["stage1_rank"].le(min(TOP_K, len(day)))].sort_values(
        ["capped_opportunity_return_7", "event_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(size)["event_id"].astype(str).tolist()


def _rank_from_selected(day: pd.DataFrame, selected: Sequence[str]) -> pd.Series:
    selected_ids = [str(value) for value in selected]
    if len(selected_ids) != len(set(selected_ids)) or len(selected_ids) != min(3, len(day)):
        raise RuntimeError("FATAL: oracle policy membership invalid")
    remainder = [event_id for event_id in _ordered_ids(day, "stage1_rank") if event_id not in selected_ids]
    order = selected_ids + remainder
    rank_map = {event_id: rank for rank, event_id in enumerate(order, 1)}
    return day["event_id"].astype(str).map(rank_map).astype(int)


def add_policy_ranks(oof: pd.DataFrame) -> pd.DataFrame:
    result = oof.copy()
    columns = {
        POLICY_STAGE1: "rank_strict_stage1",
        POLICY_CAPPED: "rank_strict_capped",
        POLICY_WINNER: "rank_winner_protect",
        POLICY_LOSS_PERMISSION: "rank_loss_permission",
        POLICY_STAGE1_BACKFILL: "rank_oracle_loss_stage1_backfill",
        POLICY_ORACLE_BACKFILL: "rank_oracle_risk_oracle_backfill",
        POLICY_TOP10_ORACLE: "rank_top10_oracle",
    }
    for column in columns.values():
        result[column] = 0
    for _, day in result.groupby("signal_date", sort=True):
        index = day.index
        stage1_ids = _ordered_ids(day, "stage1_rank")[: min(3, len(day))]
        capped_ids = _ordered_ids(day, "final_capped_rank")[: min(3, len(day))]
        selections = {
            POLICY_STAGE1: stage1_ids,
            POLICY_CAPPED: capped_ids,
            POLICY_WINNER: winner_protection_ids(day),
            POLICY_LOSS_PERMISSION: loss_permission_ids(day),
            POLICY_STAGE1_BACKFILL: stage1_loss_veto_ids(day, False),
            POLICY_ORACLE_BACKFILL: stage1_loss_veto_ids(day, True),
            POLICY_TOP10_ORACLE: top10_oracle_ids(day),
        }
        for policy, selected in selections.items():
            result.loc[index, columns[policy]] = _rank_from_selected(day, selected).to_numpy()
    for column in columns.values():
        result[column] = result[column].astype(int)
    return result


def load_frozen_june_buckets(root: Path, strict_dates: Iterable[str]) -> dict[str, str]:
    path = (
        root / "reports/research/v004c_v4a_architecture_transfer_v001_20260506_20260630"
        / "v004c_v4a_transfer_opportunity_terciles_v001.csv"
    )
    source = pd.read_csv(path, encoding="utf-8-sig", usecols=["opportunity_bucket", "dates"])
    full: dict[str, str] = {}
    for row in source.itertuples(index=False):
        for date in str(row.dates).split("|"):
            if date >= JULY_SENTINEL:
                raise RuntimeError("FATAL: July date in frozen June bucket file")
            full[date] = str(row.opportunity_bucket)
    dates = set(str(value) for value in strict_dates)
    if not dates.issubset(full):
        raise RuntimeError("FATAL: strict June date missing frozen opportunity bucket")
    return {date: full[date] for date in sorted(dates)}


def _policy_rank_columns() -> dict[str, str]:
    return {
        POLICY_STAGE1: "rank_strict_stage1",
        POLICY_CAPPED: "rank_strict_capped",
        POLICY_WINNER: "rank_winner_protect",
        POLICY_LOSS_PERMISSION: "rank_loss_permission",
        POLICY_STAGE1_BACKFILL: "rank_oracle_loss_stage1_backfill",
        POLICY_ORACLE_BACKFILL: "rank_oracle_risk_oracle_backfill",
        POLICY_TOP10_ORACLE: "rank_top10_oracle",
    }


def _raw_downside(frame: pd.DataFrame, rank_column: str) -> dict[str, float]:
    selected = frame[frame[rank_column].le(np.minimum(3, frame["candidate_count"]))]
    daily = selected.groupby("signal_date", sort=True)["raw_repair_return"].mean()
    return {
        "selected_stock_raw_mean": float(selected["raw_repair_return"].mean()),
        "selected_stock_raw_median": float(selected["raw_repair_return"].median()),
        "worst_selected_stock": float(selected["raw_repair_return"].min()),
        "worst_top3_date_raw": float(daily.min()),
    }


def build_practical(
    oof: pd.DataFrame, bucket_map: Mapping[str, str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    rank_columns = _policy_rank_columns()
    for scope in ("COMBINED", "LOW", "MID", "HIGH"):
        dates = sorted(oof["signal_date"].unique()) if scope == "COMBINED" else [
            date for date, bucket in bucket_map.items() if bucket == scope
        ]
        subset = oof[oof["signal_date"].isin(dates)].copy()
        for policy, rank_column in rank_columns.items():
            summary = strategy_summary(subset, rank_column)
            if scope == "COMBINED":
                summaries[policy] = summary
            row: dict[str, Any] = {"scope": scope, "policy": policy}
            row.update(summary)
            if scope == "COMBINED":
                row.update(_raw_downside(subset, rank_column))
            rows.append(row)
        stage1 = strategy_summary(subset, rank_columns[POLICY_STAGE1])
        universe: dict[str, Any] = {
            "scope": scope,
            "policy": "UNIVERSE",
            "date_count": int(subset["signal_date"].nunique()),
            "universe_target7_rate": stage1["universe_target7_rate"],
            "winner_capture": np.nan,
            "Top3_days_le_minus_3": np.nan,
            "Top3_days_le_minus_5": np.nan,
        }
        for label in ("Rank1", "Rank2", "Rank3", "Top1", "Top2", "Top3"):
            baseline = stage1[f"{label}_baseline"]
            for metric in (
                "mean", "median", "p25", "worst", "baseline", "excess",
                "positive_date_rate", "negative_date_rate", "beat_universe",
                "target7_precision", "dates",
            ):
                universe[f"{label}_{metric}"] = np.nan
            universe[f"{label}_mean"] = baseline
            universe[f"{label}_baseline"] = baseline
            universe[f"{label}_excess"] = 0.0
            universe[f"{label}_target7_precision"] = stage1["universe_target7_rate"]
            universe[f"{label}_dates"] = stage1[f"{label}_dates"]
        rows.append(universe)
    return pd.DataFrame(rows), summaries


def _selected_ids(day: pd.DataFrame, rank_column: str) -> set[str]:
    return set(day[day[rank_column].le(min(3, len(day)))]["event_id"].astype(str))


def build_swap_attribution(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_count = int(oof["signal_date"].nunique())
    for date, day in oof.groupby("signal_date", sort=True):
        k = min(3, len(day))
        stage1_ids = _selected_ids(day, "rank_strict_stage1")
        capped_ids = _selected_ids(day, "rank_strict_capped")
        demoted = day[day["event_id"].astype(str).isin(stage1_ids - capped_ids)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        promoted = day[day["event_id"].astype(str).isin(capped_ids - stage1_ids)].sort_values(
            ["final_capped_rank", "event_id"], kind="mergesort"
        )
        if len(demoted) != len(promoted):
            raise RuntimeError("FATAL: swap membership cardinality mismatch")
        for slot, (left, right) in enumerate(zip(demoted.itertuples(), promoted.itertuples()), 1):
            pair_delta = float(right.capped_opportunity_return_7 - left.capped_opportunity_return_7)
            daily_contribution = pair_delta / k
            rows.append({
                "signal_date": date,
                "slot_pair": slot,
                "demoted_code": str(left.code).zfill(6),
                "demoted_event_id": left.event_id,
                "demoted_stage1_rank": int(left.stage1_rank),
                "demoted_target7": int(left.target7),
                "demoted_raw_return": float(left.raw_repair_return),
                "demoted_state": outcome_state(left.raw_repair_return),
                "promoted_code": str(right.code).zfill(6),
                "promoted_event_id": right.event_id,
                "promoted_capped_rank": int(right.final_capped_rank),
                "promoted_target7": int(right.target7),
                "promoted_raw_return": float(right.raw_repair_return),
                "promoted_state": outcome_state(right.raw_repair_return),
                "pair_delta_capped_return": pair_delta,
                "daily_slot_contribution": daily_contribution,
                "overall_mean_contribution": daily_contribution / date_count,
            })
    columns = [
        "signal_date", "slot_pair", "demoted_code", "demoted_event_id",
        "demoted_stage1_rank", "demoted_target7", "demoted_raw_return", "demoted_state",
        "promoted_code", "promoted_event_id", "promoted_capped_rank", "promoted_target7",
        "promoted_raw_return", "promoted_state", "pair_delta_capped_return",
        "daily_slot_contribution", "overall_mean_contribution",
    ]
    swaps = pd.DataFrame(rows, columns=columns)
    stage1_daily = oof[oof["rank_strict_stage1"].le(3)].groupby("signal_date")["capped_opportunity_return_7"].mean()
    capped_daily = oof[oof["rank_strict_capped"].le(3)].groupby("signal_date")["capped_opportunity_return_7"].mean()
    observed = float((capped_daily - stage1_daily).mean())
    attributed = float(swaps["overall_mean_contribution"].sum()) if len(swaps) else 0.0
    if not np.isclose(attributed, observed, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"FATAL: swap attribution does not close {attributed} != {observed}")
    categories = [
        f"{left} -> {right}"
        for left in ("TARGET7", "POSITIVE_NON_TARGET", "LOSS")
        for right in ("TARGET7", "POSITIVE_NON_TARGET", "LOSS")
    ]
    category_stats: dict[str, dict[str, float | int]] = {}
    for category in categories:
        left, right = category.split(" -> ")
        subset = swaps[(swaps["demoted_state"].eq(left)) & (swaps["promoted_state"].eq(right))]
        category_stats[category] = {
            "count": int(len(subset)),
            "contribution": float(subset["overall_mean_contribution"].sum()) if len(subset) else 0.0,
        }
    by_demoted = {
        state: float(swaps.loc[swaps["demoted_state"].eq(state), "overall_mean_contribution"].sum())
        if len(swaps) else 0.0
        for state in ("TARGET7", "POSITIVE_NON_TARGET", "LOSS")
    }
    nonloss = by_demoted["TARGET7"] + by_demoted["POSITIVE_NON_TARGET"]
    loss = by_demoted["LOSS"]
    if loss > 0.0 and loss > max(nonloss, 0.0):
        primary = "DEMOTING_LOSSES"
    elif nonloss > 0.0 and nonloss > max(loss, 0.0):
        primary = "DEMOTING_NONLOSS"
    else:
        primary = "MIXED"
    return swaps, {
        "observed_delta": observed,
        "attributed_total": attributed,
        "closure": "PASS",
        "categories": category_stats,
        "by_demoted": by_demoted,
        "primary_source": primary,
    }


def membership_accounting(oof: pd.DataFrame, swaps: pd.DataFrame) -> dict[str, Any]:
    stage1 = oof[oof["rank_strict_stage1"].le(3)]
    demoted_ids = set(swaps["demoted_event_id"].astype(str)) if len(swaps) else set()
    promoted_ids = set(swaps["promoted_event_id"].astype(str)) if len(swaps) else set()
    demoted = oof[oof["event_id"].astype(str).isin(demoted_ids)]
    promoted = oof[oof["event_id"].astype(str).isin(promoted_ids)]
    winner_ids = set(stage1[stage1["target7"].eq(1)]["event_id"].astype(str))
    loss_ids = set(stage1[stage1["raw_repair_return"].lt(0.0)]["event_id"].astype(str))
    stage1_backfill_ids: list[str] = []
    oracle_backfill_ids: list[str] = []
    for _, day in oof.groupby("signal_date", sort=True):
        original = _selected_ids(day, "rank_strict_stage1")
        stage1_backfill_ids.extend(
            event_id for event_id in _selected_ids(day, "rank_oracle_loss_stage1_backfill")
            if event_id not in original
        )
        oracle_backfill_ids.extend(
            event_id for event_id in _selected_ids(day, "rank_oracle_risk_oracle_backfill")
            if event_id not in original
        )
    stage1_backfill = oof[oof["event_id"].astype(str).isin(stage1_backfill_ids)]
    oracle_backfill = oof[oof["event_id"].astype(str).isin(oracle_backfill_ids)]
    demoted_total = len(demoted)
    result = {
        "changed_dates": int(swaps["signal_date"].nunique()) if len(swaps) else 0,
        "changed_slots": int(len(swaps)),
        "stage1_winners_total": len(winner_ids),
        "stage1_winners_demoted": len(winner_ids & demoted_ids),
        "stage1_winners_capped_preserved": len(winner_ids - demoted_ids),
        "stage1_losses_total": len(loss_ids),
        "stage1_losses_demoted": len(loss_ids & demoted_ids),
        "stage1_losses_capped_preserved": len(loss_ids - demoted_ids),
        "demotion_loss_precision": float(len(loss_ids & demoted_ids) / demoted_total) if demoted_total else np.nan,
        "demotion_target7_error_rate": float(len(winner_ids & demoted_ids) / demoted_total) if demoted_total else np.nan,
        "promoted_total": int(len(promoted)),
        "promoted_target7": int(promoted["target7"].sum()),
        "promoted_positive_non_target": int(promoted["outcome_state"].eq("POSITIVE_NON_TARGET").sum()),
        "promoted_losses": int(promoted["outcome_state"].eq("LOSS").sum()),
        "promoted_severe_losses": int(promoted["raw_repair_return"].le(-0.05).sum()),
        "net_target7_slots": int(promoted["target7"].sum() - demoted["target7"].sum()),
        "stage1_backfilled_total": int(len(stage1_backfill)),
        "stage1_backfilled_target7": int(stage1_backfill["target7"].sum()),
        "stage1_backfilled_positive_non_target": int(stage1_backfill["outcome_state"].eq("POSITIVE_NON_TARGET").sum()),
        "stage1_backfilled_losses": int(stage1_backfill["outcome_state"].eq("LOSS").sum()),
        "stage1_backfilled_severe_losses": int(stage1_backfill["raw_repair_return"].le(-0.05).sum()),
        "oracle_replacements": int(len(oracle_backfill)),
        "oracle_replacement_target7_rate": float(oracle_backfill["target7"].mean()) if len(oracle_backfill) else np.nan,
        "oracle_replacement_positive_rate": float(oracle_backfill["raw_repair_return"].ge(0.0).mean()) if len(oracle_backfill) else np.nan,
    }
    return result


def build_oracle_daily(oof: pd.DataFrame, bucket_map: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = _policy_rank_columns()
    for date, day in oof.groupby("signal_date", sort=True):
        row: dict[str, Any] = {
            "signal_date": date,
            "candidate_count": int(len(day)),
            "k": min(3, len(day)),
            "universe": float(day["capped_opportunity_return_7"].mean()),
            "opportunity_bucket": bucket_map[date],
        }
        for policy, rank_column in columns.items():
            selected = day[day[rank_column].le(min(3, len(day)))]
            row[policy.lower()] = float(selected["capped_opportunity_return_7"].mean())
        row["delta_capped_vs_stage1"] = row[POLICY_CAPPED.lower()] - row[POLICY_STAGE1.lower()]
        row["delta_winnerprotect_vs_stage1"] = row[POLICY_WINNER.lower()] - row[POLICY_STAGE1.lower()]
        row["delta_losspermission_vs_stage1"] = row[POLICY_LOSS_PERMISSION.lower()] - row[POLICY_STAGE1.lower()]
        row["delta_stage1backfill_vs_stage1"] = row[POLICY_STAGE1_BACKFILL.lower()] - row[POLICY_STAGE1.lower()]
        row["delta_oracleupper_vs_stage1"] = row[POLICY_ORACLE_BACKFILL.lower()] - row[POLICY_STAGE1.lower()]
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_accounting(oof: pd.DataFrame, swaps: pd.DataFrame) -> dict[str, Any]:
    stage1 = oof[oof["rank_strict_stage1"].le(3)]
    capped = oof[oof["rank_strict_capped"].le(3)]
    stage1_ids = set(stage1["event_id"].astype(str))
    capped_ids = set(capped["event_id"].astype(str))
    demoted = stage1[~stage1["event_id"].astype(str).isin(capped_ids)]
    protected = demoted[demoted["target7"].eq(1)]
    approved = demoted[demoted["raw_repair_return"].lt(0.0)]
    blocked = demoted[demoted["raw_repair_return"].ge(0.0)]
    simple_selected = oof[oof["rank_oracle_loss_stage1_backfill"].le(3)]
    simple_ids = set(simple_selected["event_id"].astype(str))
    stage1_losses = stage1[stage1["raw_repair_return"].lt(0.0)]
    actually_removed = stage1_losses[
        ~stage1_losses["event_id"].astype(str).isin(simple_ids)
    ]
    return {
        "capped_original_demotions": int(len(demoted)),
        "protected_stage1_winners": int(len(protected)),
        "winner_protect_preserved_winners": int(len(protected)),
        "oracle_approved_loss_demotions": int(len(approved)),
        "blocked_nonloss_demotions": int(len(blocked)),
        "blocked_target7_demotions": int(blocked["target7"].sum()),
        "stage1_top3_actual_losses": int(len(stage1_losses)),
        "stage1_actual_losses_removed": int(len(actually_removed)),
        "stage1_members": len(stage1_ids),
        "swap_rows": int(len(swaps)),
    }


def _scope(practical: pd.DataFrame, scope: str, policy: str) -> pd.Series:
    rows = practical[(practical["scope"].eq(scope)) & (practical["policy"].eq(policy))]
    if len(rows) != 1:
        raise RuntimeError(f"FATAL: practical row unavailable {scope}/{policy}")
    return rows.iloc[0]


def formal_decision(practical: pd.DataFrame, membership: Mapping[str, Any]) -> dict[str, Any]:
    stage1 = _scope(practical, "COMBINED", POLICY_STAGE1)
    capped = _scope(practical, "COMBINED", POLICY_CAPPED)
    winner = _scope(practical, "COMBINED", POLICY_WINNER)
    simple = _scope(practical, "COMBINED", POLICY_STAGE1_BACKFILL)
    oracle = _scope(practical, "COMBINED", POLICY_ORACLE_BACKFILL)
    gates = {
        "A_simple_top3_plus_50bp": float(simple.Top3_mean) >= float(stage1.Top3_mean) + 0.005,
        "B_simple_precision_preserved": float(simple.Top3_target7_precision) >= float(stage1.Top3_target7_precision),
        "C_simple_risk_improved": (
            float(simple.Top3_negative_date_rate) <= float(stage1.Top3_negative_date_rate)
            and float(simple.Top3_worst) >= float(stage1.Top3_worst)
        ),
        "D_oracle_headroom_100bp": float(oracle.Top3_mean) >= float(stage1.Top3_mean) + 0.01,
        "E_winner_protection_repairs_capped": (
            float(winner.Top3_mean) > float(capped.Top3_mean)
            and float(winner.Top3_target7_precision) >= float(capped.Top3_target7_precision)
        ),
    }
    if all(gates.values()):
        signal = "STRONG"
    elif (
        gates["D_oracle_headroom_100bp"]
        or gates["E_winner_protection_repairs_capped"]
        or float(simple.Top3_mean) > float(stage1.Top3_mean)
        or float(simple.Top3_negative_date_rate) < float(stage1.Top3_negative_date_rate)
        or float(simple.Top3_worst) > float(stage1.Top3_worst)
    ):
        signal = "PARTIAL"
    else:
        signal = "ABSENT"
    simple_delta = float(simple.Top3_mean - stage1.Top3_mean)
    oracle_delta = float(oracle.Top3_mean - stage1.Top3_mean)
    replacement_bottleneck = bool(oracle_delta >= 0.01 and simple_delta < 0.005)
    if simple_delta >= 0.005:
        main_bottleneck = "RISK_IDENTIFICATION"
    elif replacement_bottleneck:
        main_bottleneck = "REPLACEMENT_SELECTION"
    elif oracle_delta > 0.0:
        main_bottleneck = "BOTH"
    else:
        main_bottleneck = "NEITHER"
    if float(oracle.Top3_mean) > float(winner.Top3_mean):
        preferred = "STAGE1_OWNS_SELECTION_PLUS_RISK_PROTECTION"
    elif float(winner.Top3_mean) > float(oracle.Top3_mean):
        preferred = "CAPPED_OWNS_RERANKING_PLUS_WINNER_PROTECTION"
    else:
        preferred = "NEITHER"
    if membership["stage1_winners_demoted"] > 0 and preferred.startswith("STAGE1_"):
        v5_risk = "HIGH"
    elif membership["stage1_winners_demoted"] > 0:
        v5_risk = "MEDIUM"
    else:
        v5_risk = "LOW"
    return {
        "gates": gates,
        "signal": signal,
        "risk_protector_training_warranted": signal == "STRONG",
        "simple_delta": simple_delta,
        "oracle_delta": oracle_delta,
        "replacement_gap": oracle_delta - simple_delta,
        "replacement_bottleneck": replacement_bottleneck,
        "main_bottleneck": main_bottleneck,
        "preferred_direction": preferred,
        "v5_repetition_risk": v5_risk,
    }


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def build_review(context: Mapping[str, Any]) -> str:
    practical = context["practical"]
    membership = context["membership"]
    experiments = context["experiments"]
    attribution = context["attribution"]
    decision = context["decision"]
    stage1 = _scope(practical, "COMBINED", POLICY_STAGE1)
    capped = _scope(practical, "COMBINED", POLICY_CAPPED)
    lines = [
        "# v004c Stage1 × Risk-Protection Architecture Diagnostic v001",
        "",
        "## 1. Experimental Contract",
        "",
        "- Scope: June-only architecture diagnostic; no new model, feature, target, threshold, or tuning.",
        "- July result rows accessed: 0. JULY_USED_FOR_ARCHITECTURE_DESIGN: NO.",
        "- Oracle policies are non-tradable counterfactual upper bounds.",
        "",
        "## 2. Naming and Frozen Specifications",
        "",
        "- Current Stage1: V4C_STAGE1 (architecture source V4A_ARCH_TRANSFER_V4C).",
        "- Historical original v004a is not the same fitted model instance.",
        "- Stage2: PAIR_CAPPED7; same-date weighted Ridge, L2 0.30, zero intercept, three frozen predictors.",
        "",
        "## 3. Strict June Temporal Reconstruction",
        "",
        f"- Raw June dates: 21; common strict dates: {context['strict_dates']}.",
        f"- Unavailable dates: {'|'.join(context['unavailable_dates'])}.",
        "- Training rule: label_available_date < test_date; minimum 18 matured dates.",
        f"- SELF_LABEL_LEAKAGE_ROWS: {context['self_label_leakage_rows']}; CURRENT_TEST_LEAKAGE_ROWS: {context['current_test_leakage_rows']}.",
        "- Legacy June values retain temporal caveat: 337 immature-label rows across 21 folds.",
        "",
        "## 4. STRICT_STAGE1 vs STRICT_CAPPED",
        "",
    ]
    rows = []
    for policy in ("UNIVERSE", POLICY_STAGE1, POLICY_CAPPED):
        row = _scope(practical, "COMBINED", policy)
        rows.append([
            policy, _pct(row.Rank1_mean), _pct(row.Rank2_mean), _pct(row.Rank3_mean),
            _pct(row.Top2_mean), _pct(row.Top3_mean), _pct(row.Top3_target7_precision),
            _pct(row.Top3_negative_date_rate), _pct(row.Top3_worst),
        ])
    lines += _markdown_table(
        ["Policy", "Rank1", "Rank2", "Rank3", "Top2", "Top3", "Top3 precision", "Negative dates", "Worst"],
        rows,
    )
    lines += [
        "",
        "## 5. Swap Attribution",
        "",
        f"- Changed dates / slots: {membership['changed_dates']} / {membership['changed_slots']}.",
        f"- Demotion loss precision: {_pct(membership['demotion_loss_precision'])}; Target7 error rate: {_pct(membership['demotion_target7_error_rate'])}.",
        f"- Contribution from demoting Target7: {_pct(attribution['by_demoted']['TARGET7'])}.",
        f"- Contribution from demoting positive non-target: {_pct(attribution['by_demoted']['POSITIVE_NON_TARGET'])}.",
        f"- Contribution from demoting loss: {_pct(attribution['by_demoted']['LOSS'])}.",
        f"- Attributed total / observed delta: {_pct(attribution['attributed_total'])} / {_pct(attribution['observed_delta'])}; closure PASS.",
        "",
        "## 6. Winner-Protection Oracle",
        "",
        f"- Stage1 winners protected from CAPPED demotion: {experiments['protected_stage1_winners']}.",
        (
            f"- Winner-protected Top3: {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_mean)}; "
            f"delta vs Stage1 {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_mean-stage1.Top3_mean)}; "
            f"delta vs CAPPED {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_mean-capped.Top3_mean)}."
        ),
        (
            f"- Precision {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_target7_precision)}; "
            f"winner capture {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).winner_capture)}; "
            f"negative dates {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_negative_date_rate)}; "
            f"worst {_pct(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_worst)}; "
            f"days <= -5% {int(_scope(practical, 'COMBINED', POLICY_WINNER).Top3_days_le_minus_5)}."
        ),
        "",
        "## 7. Loss-Only Intervention Oracle",
        "",
        f"- Original demotions: {experiments['capped_original_demotions']}; approved actual-loss demotions: {experiments['oracle_approved_loss_demotions']}; blocked non-loss: {experiments['blocked_nonloss_demotions']}.",
        (
            f"- Loss-permission Top3: {_pct(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).Top3_mean)}; "
            f"delta vs Stage1 {_pct(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).Top3_mean-stage1.Top3_mean)}; "
            f"precision {_pct(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).Top3_target7_precision)}; "
            f"winner capture {_pct(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).winner_capture)}."
        ),
        (
            f"- Negative dates {_pct(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).Top3_negative_date_rate)}; "
            f"worst {_pct(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).Top3_worst)}; "
            f"days <= -5% {int(_scope(practical, 'COMBINED', POLICY_LOSS_PERMISSION).Top3_days_le_minus_5)}."
        ),
        "",
        "## 8. Perfect Loss Detection + Stage1 Backfill",
        "",
        f"- Stage1 Top3 actual losses: {experiments['stage1_top3_actual_losses']}; actually removed with available backfill: {experiments['stage1_actual_losses_removed']}.",
        f"- Simple backfill Top3: {_pct(_scope(practical, 'COMBINED', POLICY_STAGE1_BACKFILL).Top3_mean)}; delta {_pct(decision['simple_delta'])}; precision {_pct(_scope(practical, 'COMBINED', POLICY_STAGE1_BACKFILL).Top3_target7_precision)}; winner capture {_pct(_scope(practical, 'COMBINED', POLICY_STAGE1_BACKFILL).winner_capture)}.",
        f"- Negative dates {_pct(_scope(practical, 'COMBINED', POLICY_STAGE1_BACKFILL).Top3_negative_date_rate)}; worst {_pct(_scope(practical, 'COMBINED', POLICY_STAGE1_BACKFILL).Top3_worst)}.",
        f"- Simple backfills: {membership['stage1_backfilled_total']}; Target7 {membership['stage1_backfilled_target7']}; positive non-target {membership['stage1_backfilled_positive_non_target']}; loss {membership['stage1_backfilled_losses']}.",
        "",
        "## 9. Architecture Upper Bounds",
        "",
    ]
    upper_rows = []
    for policy in (POLICY_STAGE1_BACKFILL, POLICY_ORACLE_BACKFILL, POLICY_WINNER, POLICY_TOP10_ORACLE):
        row = _scope(practical, "COMBINED", policy)
        upper_rows.append([policy, _pct(row.Top3_mean), _pct(row.Top3_target7_precision), _pct(row.winner_capture), _pct(row.Top3_negative_date_rate), _pct(row.Top3_worst)])
    lines += _markdown_table(
        ["Policy", "Top3", "Precision", "Winner capture", "Negative dates", "Worst"],
        upper_rows,
    )
    lines += [
        "",
        "Raw downside audit:",
        "",
    ]
    raw_rows = []
    for policy in (POLICY_STAGE1, POLICY_CAPPED, POLICY_WINNER, POLICY_LOSS_PERMISSION, POLICY_STAGE1_BACKFILL):
        row = _scope(practical, "COMBINED", policy)
        raw_rows.append([
            policy,
            _pct(row.selected_stock_raw_mean),
            _pct(row.selected_stock_raw_median),
            _pct(row.worst_selected_stock),
            _pct(row.worst_top3_date_raw),
        ])
    lines += _markdown_table(
        ["Policy", "Raw mean", "Raw median", "Worst stock", "Worst Top3 date"],
        raw_rows,
    )
    lines += [
        "",
        "## 10. LOW / MID / HIGH",
        "",
    ]
    bucket_rows = []
    for policy in ("UNIVERSE", POLICY_STAGE1, POLICY_CAPPED, POLICY_WINNER, POLICY_LOSS_PERMISSION, POLICY_STAGE1_BACKFILL, POLICY_ORACLE_BACKFILL):
        bucket_rows.append([
            policy,
            _pct(_scope(practical, "LOW", policy).Top3_mean),
            _pct(_scope(practical, "MID", policy).Top3_mean),
            _pct(_scope(practical, "HIGH", policy).Top3_mean),
            _pct(_scope(practical, "HIGH", policy).Top3_target7_precision),
        ])
    lines += _markdown_table(["Policy", "LOW Top3", "MID Top3", "HIGH Top3", "HIGH precision"], bucket_rows)
    lines += [
        "",
        "## 11. Architecture Decision",
        "",
        f"- ARCHITECTURE_COMPLEMENTARITY_SIGNAL: **{decision['signal']}**.",
        f"- RISK_PROTECTOR_TRAINING_WARRANTED: **{'YES' if decision['risk_protector_training_warranted'] else 'NO'}**.",
        f"- Preferred theoretical direction: **{decision['preferred_direction']}**.",
        f"- Replacement selection major bottleneck: **{'YES' if decision['replacement_bottleneck'] else 'NO'}**.",
        "- Formal gates: " + "; ".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, passed in decision["gates"].items()
        ) + ".",
        "- This diagnostic establishes architecture headroom only; it does not prove a learnable or production-ready risk model.",
        "",
        "## 12. v5 Structural-Repetition Risk",
        "",
        f"- V5_REPETITION_RISK_FOR_FULL_RERANKER: **{decision['v5_repetition_risk']}**.",
        "- Another full Top10 reranker is not justified by this task.",
    ]
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    base_x, samples, date_contract, input_audit = prepare_diagnostic_samples(root)
    strict, temporal, unavailable = reconstruct_strict_june(base_x, samples, date_contract)
    strict = add_policy_ranks(strict)
    strict_dates = sorted(strict["signal_date"].unique())
    bucket_map = load_frozen_june_buckets(root, strict_dates)
    practical, summaries = build_practical(strict, bucket_map)
    swaps, attribution = build_swap_attribution(strict)
    membership = membership_accounting(strict, swaps)
    experiments = experiment_accounting(strict, swaps)
    oracle_daily = build_oracle_daily(strict, bucket_map)
    decision = formal_decision(practical, membership)
    self_leak = int(temporal["self_label_leakage_rows"].sum())
    test_leak = int(temporal["current_test_leakage_rows"].sum())
    context: dict[str, Any] = {
        "input": input_audit,
        "strict_dates": len(strict_dates),
        "strict_date_list": strict_dates,
        "unavailable_dates": unavailable,
        "self_label_leakage_rows": self_leak,
        "current_test_leakage_rows": test_leak,
        "july_result_rows_accessed": 0,
        "practical": practical,
        "summaries": summaries,
        "membership": membership,
        "experiments": experiments,
        "attribution": attribution,
        "decision": decision,
    }
    review = build_review(context)
    oof_columns = [
        "event_id", "signal_date", "code", "candidate_count", "label_available_date",
        "stage1_score", "stage1_rank", "stage1_top10", "closing_completion_gap",
        "strength_x_gap", "capped_pair_score", "capped_pair_rank", "final_capped_rank",
        "target7", "raw_repair_return", "capped_opportunity_return_7", "outcome_state",
        *_policy_rank_columns().values(),
    ]
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(strict[oof_columns]),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(temporal),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(swaps),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(oracle_daily),
        OUTPUT_FILENAMES[4]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[5]: review.encode("utf-8"),
    }
    return outputs, context


def run_v004c_stage1_risk_complementarity(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root)
    target = Path(output_dir) if output_dir is not None else (
        root_path / "reports/research/v004c_stage1_risk_complementarity_v001_20260601_20260630"
    )
    outputs, context = build_outputs(root_path)
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        (target / filename).write_bytes(payload)
    return target, context

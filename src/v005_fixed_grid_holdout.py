from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .daily_ranking import load_ranking_model
from .history_samples import HISTORY_CANDIDATE_COLUMNS, HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS
from .policy_config import frozen_policy_input_checks, get_default_policy, normalized_sha256, validate_model_dates_for_signal_date
from .provenance import file_sha256, runtime_provenance
from .ranking_backtest import validate_ranking_model
from .v004a import (
    DEFAULT_TARGET_RETURN_PCT,
    MODEL_ID_V004A,
    SCOPE_WALK_FORWARD,
    _load_manual_models,
    _score_logistic_frame,
    _score_manual_and_hand_models,
    add_scored_model_rank,
    build_scored_candidates_output,
    annotate_v004a_input_eligibility,
    prepare_v004a_samples,
)
from .universe_audit import (
    UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
    UniverseAuditError,
    atomic_write_csv,
    atomic_write_text,
    canonical_row_lines,
    canonical_rows_sha256,
    code_set_sha256,
    count_duplicate_keys,
    normalize_code_series,
    require_nonempty_codes,
    require_requested_signal_date_match,
    require_unique_keys,
    write_json,
)
from .v005_set_selector import (
    DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
    DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
    DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
    DEFAULT_EXTREME_PRICE_PENALTY_GRID,
    DEFAULT_EXTREME_VWAP_PENALTY_GRID,
    DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    DEFAULT_TOP_N,
    DEFAULT_V004A_L2,
    DEFAULT_V004A_POSITIVE_WEIGHT,
    GRID_PARAM_COLUMNS,
    TARGET_COLUMN,
    V002_MODEL_ID,
    build_candidate_pool,
    build_combo_candidates,
    build_rule_grid,
    explode_selected_combos,
    prepare_scored_candidates,
    score_combos,
    select_best_combo_by_date,
)
from .v005_fallback_gate import (
    DAILY_COLUMNS as FALLBACK_DAILY_COLUMNS,
    PRIMARY_POLICY,
    REPLACEMENT_COLUMNS,
    SUMMARY_COLUMNS as FALLBACK_SUMMARY_COLUMNS,
    build_baseline,
    build_context,
    build_topn,
    ctx_for_codes,
    is_policy_fallback,
    metrics,
    norm,
    parse_codes,
    replacement_detail,
    summarize,
)

DEFAULT_POLICY = get_default_policy()
DEFAULT_SAMPLES_FILE = None
DEFAULT_COEFFICIENTS_FILE = DEFAULT_POLICY.coefficients_path
DEFAULT_RANKING_MODEL_FILE = DEFAULT_POLICY.ranking_model_path
DEFAULT_OUTPUT_DIR = Path("reports/v005_fixed_grid_holdout")
DEFAULT_GRID_ID = DEFAULT_POLICY.grid_id
DEFAULT_V002_MODEL_LABEL = "v002_top3_control"
DEFAULT_V004A_MODEL_LABEL = "v004a_top3_control"

HOLDOUT_UNIVERSE_MEMBERSHIP_COLUMNS = [
    "signal_date",
    "code",
    "name",
    "eligible_for_trade",
    "v004a_scorable_bool",
    "v004a_model_score",
    "v004a_model_rank",
    "v002_model_score",
    "v002_model_rank",
    "in_v004a_topk",
    "in_v002_topk",
    "in_v005_candidate_pool",
    "in_final_top3",
    "final_top3_rank",
    "target7_d2open_d3high",
    "d2open_d3high_return_pct",
    "d2open_d3close_return_pct",
    "realized_return_pct",
]

HOLDOUT_DAILY_COLUMNS = [
    "strategy",
    "signal_date",
    "action",
    "selected_codes",
    "source_codes",
    "selected_grid_id",
    "hit_count",
    "all_hit",
    "avg_high_return",
    "avg_realized_return",
    "rank1_hit",
    "rank2_hit",
    "rank3_hit",
    "v002_codes",
    "v002_hit_count",
    "v002_all_hit",
    "v002_avg_realized_return",
    "v004a_codes",
    "v004a_hit_count",
    "v004a_all_hit",
    "v004a_avg_realized_return",
    "baseline_v005_codes",
    "baseline_v005_hit_count",
    "baseline_v005_all_hit",
    "baseline_v005_avg_realized_return",
    "gate_v002_extreme_vwap_count",
    "gate_v002_extreme_close_low_count",
    "gate_v005_avg_v002_rank",
    "gate_v005_has_risk_ticket",
    "gate_triggered",
]


def run_fixed_grid_holdout(
    samples_file: str | Path | None = DEFAULT_SAMPLES_FILE,
    scored_file: str | Path | None = None,
    coefficients_file: str | Path = DEFAULT_COEFFICIENTS_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = DEFAULT_POLICY.top_n,
    candidate_top_k: int = DEFAULT_POLICY.candidate_top_k,
    grid_id: int = DEFAULT_GRID_ID,
    coefficient_predict_date: str = DEFAULT_POLICY.coefficient_predict_date,
    v004a_l2: float = DEFAULT_POLICY.v004a_l2,
    v004a_positive_weight: float = DEFAULT_POLICY.v004a_positive_weight,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    min_forward_dates: int = DEFAULT_POLICY.min_forward_dates,
    ranking_model_file: str | Path = DEFAULT_RANKING_MODEL_FILE,
    history_universe_manifest_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    if abs(float(target_return_pct) - DEFAULT_POLICY.target_return_pct) > 1e-12:
        raise RuntimeError(
            "Frozen v005 policy requires target_return_pct=7.0 because the locked target is "
            "target7_d2open_d3high."
        )
    _, ranking_meta = load_ranking_model(ranking_model_file)
    ranking_model_id = str(ranking_meta["model_id"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = Path(samples_file) if samples_file else None
    raw_samples = (
        pd.read_csv(samples_path, dtype={"code": str})
        if samples_path is not None and samples_path.is_file()
        else pd.DataFrame()
    )
    explicit_history_manifest_path = (
        Path(history_universe_manifest_file) if history_universe_manifest_file else None
    )
    inferred_history_manifest_path = _infer_history_universe_manifest(samples_path)
    manifest_backed = bool(
        explicit_history_manifest_path is not None
        or (
            inferred_history_manifest_path is not None
            and inferred_history_manifest_path.is_file()
        )
    )
    legacy_duplicate_meta = _empty_legacy_duplicate_meta()
    if samples_path is not None and samples_path.is_file():
        require_nonempty_codes(raw_samples, "code", "holdout samples")
        raw_samples["code"] = normalize_code_series(raw_samples["code"])
        if "signal_date" not in raw_samples.columns:
            raise UniverseAuditError("holdout samples are missing signal_date")
        raw_samples["signal_date"] = raw_samples["signal_date"].fillna("").astype(str)
        if raw_samples["signal_date"].isin({"", "nan", "NaT", "None"}).any():
            raise UniverseAuditError("holdout samples contain empty or invalid signal_date values")
        if "requested_signal_date" in raw_samples.columns:
            require_requested_signal_date_match(raw_samples)
        if manifest_backed:
            require_unique_keys(
                raw_samples,
                ("signal_date", "code"),
                "manifest-backed holdout samples",
            )
        else:
            raw_samples, legacy_duplicate_meta = _validate_and_exclude_legacy_duplicates(
                raw_samples
            )
    history_universe = _load_history_universe_context(
        samples_path=samples_path,
        raw_samples=raw_samples,
        explicit_manifest_path=explicit_history_manifest_path,
        legacy_duplicate_meta=legacy_duplicate_meta,
    )

    if scored_file:
        scored_path = Path(scored_file)
        if not scored_path.exists():
            raise RuntimeError(f"missing scored_file: {scored_path}")
        scored = prepare_scored_candidates(scored_path)
        holdout_scored_path = scored_path
        coefficient_meta: dict[str, Any] = {
            "source": "pre_scored_file",
            "ranking_model_path": str(ranking_meta["model_path"]),
            "ranking_model_normalized_sha256": str(ranking_meta["model_normalized_sha256"]),
            "ranking_model_id": ranking_model_id,
        }
        data_quality = pd.DataFrame()
    else:
        if not samples_file:
            raise RuntimeError("either --samples-file or --scored-file is required")
        holdout_scored_path = out_dir / "v005_fixed_grid_holdout_scored_candidates.csv"
        scored, data_quality, coefficient_meta = build_holdout_scored_candidates(
            samples_file=Path(samples_file),
            coefficients_file=Path(coefficients_file),
            output_path=holdout_scored_path,
            coefficient_predict_date=coefficient_predict_date,
            v004a_l2=float(v004a_l2),
            v004a_positive_weight=float(v004a_positive_weight),
            target_return_pct=float(target_return_pct),
            ranking_model_file=Path(ranking_model_file),
        )

    _validate_configured_scored_coverage(
        scored=scored,
        raw_samples=raw_samples,
        samples_available=samples_path is not None and samples_path.is_file(),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        ranking_model_id=ranking_model_id,
    )
    candidate_pool = build_candidate_pool(
        scored,
        candidate_top_k=int(candidate_top_k),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        v002_model_id=ranking_model_id,
    )
    if candidate_pool.empty:
        raise RuntimeError("holdout produced no eligible candidate-pool rows")
    if coefficient_meta.get("coefficient_predict_date") and coefficient_meta.get("coefficient_train_end"):
        first_signal_date = str(candidate_pool["signal_date"].dropna().astype(str).min())
        validate_model_dates_for_signal_date(
            predict_date=str(coefficient_meta["coefficient_predict_date"]),
            train_end=str(coefficient_meta["coefficient_train_end"]),
            signal_date=first_signal_date,
        )
    combo_candidates = build_combo_candidates(candidate_pool, top_n=int(top_n))
    grid = build_default_grid()
    fixed_params = select_grid_params(grid, grid_id=int(grid_id))
    selected_combos = select_best_combo_by_date(score_combos(combo_candidates, fixed_params))
    daily_top3 = explode_selected_combos(selected_combos, candidate_pool, top_n=int(top_n))

    summary, policy_daily, replacement = apply_fixed_policy(
        scored_path=holdout_scored_path,
        selected_combos=selected_combos,
        top_n=int(top_n),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        v002_model_id=ranking_model_id,
    )
    universe_membership, universe_audit, universe_manifest_path = _write_holdout_universe_outputs(
        output_dir=out_dir,
        scored=scored,
        scored_path=holdout_scored_path,
        samples_path=samples_path,
        raw_samples=raw_samples,
        candidate_pool=candidate_pool,
        selected_combos=selected_combos,
        policy_daily=policy_daily,
        candidate_top_k=int(candidate_top_k),
        top_n=int(top_n),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        ranking_model_id=ranking_model_id,
        coefficient_meta=coefficient_meta,
        history_universe=history_universe,
    )
    universe_manifest_sha256 = file_sha256(universe_manifest_path)
    frozen_checks = frozen_policy_input_checks(
        DEFAULT_POLICY,
        coefficients_file=coefficients_file,
        coefficient_predict_date=str(coefficient_meta.get("coefficient_predict_date", "")),
        coefficient_train_end=str(coefficient_meta.get("coefficient_train_end", "")),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
        ranking_model_file=ranking_model_file,
        ranking_model_id=ranking_model_id,
        target_column=TARGET_COLUMN,
        target_return_pct=float(target_return_pct),
        grid_id=int(grid_id),
        candidate_top_k=int(candidate_top_k),
        top_n=int(top_n),
        min_forward_dates=int(min_forward_dates),
    )
    frozen_checks["samples_input_scored_in_this_run"] = not bool(scored_file)
    frozen_policy_inputs_verified = all(frozen_checks.values())
    deployment_status = DEFAULT_POLICY.deployment_status if frozen_policy_inputs_verified else "custom_research_only"
    readiness = assess_holdout_readiness(
        policy_daily,
        min_forward_dates=int(min_forward_dates),
        frozen_policy_inputs_verified=frozen_policy_inputs_verified,
        deployment_status=deployment_status,
    )
    run_meta = pd.DataFrame(
        [
            {
                **DEFAULT_POLICY.provenance(),
                **runtime_provenance(),
                "policy_version": (
                    DEFAULT_POLICY.policy_version
                    if frozen_policy_inputs_verified
                    else f"{DEFAULT_POLICY.policy_version}+custom_override"
                ),
                "deployment_status": deployment_status,
                "input_mode": "pre_scored" if scored_file else "samples",
                "samples_file": str(samples_path) if samples_path else "",
                "samples_file_sha256": file_sha256(samples_path) if samples_path and samples_path.is_file() else "",
                "scored_file": str(holdout_scored_path),
                "scored_file_sha256": file_sha256(holdout_scored_path),
                "frozen_policy_inputs_verified": frozen_policy_inputs_verified,
                "matches_frozen_manifest": frozen_policy_inputs_verified,
                "failed_frozen_policy_checks": ",".join(
                    key for key, passed in frozen_checks.items() if not passed
                ),
                "coefficients_file": str(coefficients_file),
                "coefficients_normalized_sha256": (
                    normalized_sha256(coefficients_file) if Path(coefficients_file).is_file() else ""
                ),
                "coefficient_predict_date": str(coefficient_meta.get("coefficient_predict_date", "")),
                "coefficient_train_end": str(coefficient_meta.get("coefficient_train_end", "")),
                "v004a_l2": float(v004a_l2),
                "v004a_positive_weight": float(v004a_positive_weight),
                "ranking_model_path": str(ranking_meta["model_path"]),
                "ranking_model_normalized_sha256": str(ranking_meta["model_normalized_sha256"]),
                "ranking_model_id": ranking_model_id,
                "v002_source_model_id": ranking_model_id,
                "target_return_pct": float(target_return_pct),
                "grid_id": int(grid_id),
                "candidate_top_k": int(candidate_top_k),
                "top_n": int(top_n),
                "min_forward_dates": int(min_forward_dates),
                "target_metric_kind": "d2open_to_d3_intraday_high_opportunity_proxy",
                "transaction_costs_included": False,
                "slippage_included": False,
                "cache_snapshot_complete": False,
                "candidate_universe_snapshot_complete": bool(history_universe["snapshot_complete"]),
                "candidate_universe_snapshot_verified": bool(history_universe["snapshot_verified"]),
                "candidate_universe_manifest_path": str(universe_manifest_path),
                "candidate_universe_manifest_sha256": universe_manifest_sha256,
                "universe_snapshot_schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
                "universe_audit_status": str(history_universe["audit_status"]),
                "legacy_duplicate_key_count": int(history_universe["legacy_duplicate_key_count"]),
                "legacy_duplicate_row_count": int(history_universe["legacy_duplicate_row_count"]),
                "legacy_duplicate_codes": str(history_universe["legacy_duplicate_codes"]),
            }
        ]
    )

    atomic_write_csv(out_dir / "v005_fixed_grid_combo_candidates.csv", combo_candidates)
    atomic_write_csv(out_dir / "v005_fixed_grid_selected_combos.csv", selected_combos)
    atomic_write_csv(out_dir / "v005_fixed_grid_daily_top3.csv", daily_top3)
    atomic_write_csv(out_dir / "v005_fixed_grid_holdout_summary.csv", summary)
    atomic_write_csv(out_dir / "v005_fixed_grid_holdout_daily.csv", policy_daily)
    atomic_write_csv(out_dir / "v005_fixed_grid_holdout_replacement.csv", replacement)
    atomic_write_csv(out_dir / "v005_fixed_grid_holdout_readiness.csv", readiness)
    atomic_write_csv(out_dir / "v005_fixed_grid_holdout_run_meta.csv", run_meta)
    if not data_quality.empty:
        atomic_write_csv(out_dir / "v005_fixed_grid_holdout_data_quality.csv", data_quality)

    report_path = out_dir / "v005_fixed_grid_holdout_report.md"
    atomic_write_text(
        report_path,
        make_report(
            samples_file=samples_file,
            scored_file=holdout_scored_path,
            coefficients_file=coefficients_file,
            output_dir=out_dir,
            grid_id=int(grid_id),
            grid_params=fixed_params,
            coefficient_meta=coefficient_meta,
            candidate_pool=candidate_pool,
            combo_candidates=combo_candidates,
            selected_combos=selected_combos,
            daily_top3=daily_top3,
            summary=summary,
            policy_daily=policy_daily,
            replacement=replacement,
            readiness=readiness,
            run_meta=run_meta,
            universe_audit=universe_audit,
        ),
        encoding="utf-8",
    )
    return summary, policy_daily, replacement, daily_top3, report_path


def _empty_legacy_duplicate_meta() -> dict[str, Any]:
    return {
        "legacy_duplicate_key_count": 0,
        "legacy_duplicate_row_count": 0,
        "legacy_duplicate_codes": "",
    }


def _validate_and_exclude_legacy_duplicates(
    raw_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = raw_samples.copy().reset_index(drop=True)
    duplicate_mask = frame.duplicated(["signal_date", "code"], keep=False)
    if not duplicate_mask.any():
        return frame, _empty_legacy_duplicate_meta()

    annotated = annotate_v004a_input_eligibility(frame)
    duplicate_rows = frame.loc[duplicate_mask].copy()
    keys = duplicate_rows[["signal_date", "code"]].drop_duplicates().sort_values(
        ["signal_date", "code"], kind="mergesort"
    )
    invalid_content: list[str] = []
    scorable_keys: list[str] = []
    for signal_date, code in keys.itertuples(index=False, name=None):
        key_mask = frame["signal_date"].eq(signal_date) & frame["code"].eq(code)
        group = frame.loc[key_mask]
        canonical_lines = canonical_row_lines(
            group,
            HISTORY_CANDIDATE_COLUMNS,
            ("signal_date", "code"),
        )
        key_text = f"{signal_date}|{code}"
        if len(set(canonical_lines)) != 1:
            invalid_content.append(key_text)

        annotated_group = annotated.loc[key_mask]
        intrinsic_reasons = annotated_group["v004a_exclusion_reason"].fillna("").astype(str).map(
            lambda value: "|".join(
                reason
                for reason in value.split("|")
                if reason and reason != "duplicate_signal_code"
            )
        )
        explicitly_scorable = (
            frame.loc[key_mask, "v004a_scorable_bool"].map(_bool_value).any()
            if "v004a_scorable_bool" in frame.columns
            else False
        )
        if explicitly_scorable or intrinsic_reasons.eq("").any():
            scorable_keys.append(key_text)

    if invalid_content:
        raise UniverseAuditError(
            "legacy duplicate groups contain non-identical HISTORY_CANDIDATE_COLUMNS rows: "
            f"{invalid_content[:10]}"
        )
    if scorable_keys:
        raise UniverseAuditError(
            "legacy duplicate groups contain valid scorable rows and cannot be excluded: "
            f"{scorable_keys[:10]}"
        )

    metadata = {
        "legacy_duplicate_key_count": int(len(keys)),
        "legacy_duplicate_row_count": int(duplicate_mask.sum()),
        "legacy_duplicate_codes": ",".join(sorted(keys["code"].astype(str).unique().tolist())),
    }
    return frame.loc[~duplicate_mask].reset_index(drop=True), metadata


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_history_universe_context(
    samples_path: Path | None,
    raw_samples: pd.DataFrame,
    explicit_manifest_path: Path | None,
    legacy_duplicate_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duplicate_meta = legacy_duplicate_meta or _empty_legacy_duplicate_meta()
    manifest_path = explicit_manifest_path or _infer_history_universe_manifest(samples_path)
    empty = {
        "manifest_path": "",
        "manifest_sha256": "",
        "snapshot_complete": False,
        "snapshot_verified": False,
        "audit_status": (
            "LEGACY_UNVERIFIED_DUPLICATES_EXCLUDED"
            if int(duplicate_meta["legacy_duplicate_key_count"]) > 0
            else "LEGACY_UNVERIFIED"
        ),
        "date_statuses": {},
        **duplicate_meta,
    }
    if manifest_path is None:
        return empty
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        if explicit_manifest_path is not None:
            raise UniverseAuditError(f"history universe manifest is missing: {manifest_path}")
        return empty
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    schema = int(manifest.get("universe_snapshot_schema_version", -1))
    if schema != UNIVERSE_SNAPSHOT_SCHEMA_VERSION:
        raise UniverseAuditError(
            f"unsupported history universe manifest schema_version={schema}: {manifest_path}"
        )
    expected_candidates_hash = str(manifest.get("history_candidates_canonical_rows_sha256", ""))
    actual_candidates_hash = canonical_rows_sha256(
        raw_samples,
        HISTORY_CANDIDATE_COLUMNS,
        ("requested_signal_date", "code"),
    )
    if not expected_candidates_hash or actual_candidates_hash != expected_candidates_hash:
        raise UniverseAuditError(
            "history universe manifest does not match samples input: "
            f"expected_candidates_hash={expected_candidates_hash}, actual_candidates_hash={actual_candidates_hash}"
        )
    suffix = manifest_path.stem.removeprefix("history_universe_manifest_")
    membership_path = manifest_path.parent / f"history_universe_membership_{suffix}.csv"
    if not membership_path.is_file():
        raise UniverseAuditError(f"history universe membership file is missing: {membership_path}")
    membership = pd.read_csv(membership_path, dtype={"code": str})
    actual_membership_hash = canonical_rows_sha256(
        membership,
        HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS,
        ("requested_signal_date", "stage", "member_key"),
    )
    expected_membership_hash = str(manifest.get("membership_canonical_rows_sha256", ""))
    if actual_membership_hash != expected_membership_hash:
        raise UniverseAuditError(
            "history universe membership hash mismatch: "
            f"expected={expected_membership_hash}, actual={actual_membership_hash}"
        )
    date_records = list(manifest.get("dates", []))
    requested_date_values = [str(row.get("requested_signal_date", "")) for row in date_records]
    if len([value for value in requested_date_values if value]) != len(
        set(value for value in requested_date_values if value)
    ):
        raise UniverseAuditError("history universe manifest contains duplicate requested dates")
    date_statuses = {
        str(row.get("requested_signal_date", "")): str(row.get("snapshot_status", ""))
        for row in date_records
    }
    verified_statuses = {"CREATED_CANONICAL", "VERIFIED_MATCH"}
    successful_snapshot_dates = sorted(
        date for date, status in date_statuses.items() if date and status in verified_statuses
    )
    manifest_requested_dates = [str(value) for value in manifest.get("requested_dates", [])]
    manifest_generated_dates = [str(value) for value in manifest.get("generated_dates", [])]
    if not manifest_requested_dates:
        raise UniverseAuditError("history universe manifest is missing requested_dates")
    if len(manifest_requested_dates) != len(set(manifest_requested_dates)):
        raise UniverseAuditError("history universe manifest requested_dates contains duplicates")
    if len(manifest_generated_dates) != len(set(manifest_generated_dates)):
        raise UniverseAuditError("history universe manifest generated_dates contains duplicates")
    expected_requested_dates = [
        timestamp.strftime("%Y-%m-%d")
        for timestamp in pd.date_range(
            pd.Timestamp(str(manifest.get("start_date", ""))),
            pd.Timestamp(str(manifest.get("end_date", ""))),
            freq="D",
        )
        if timestamp.weekday() < 5
    ]
    if set(manifest_requested_dates) != set(expected_requested_dates):
        raise UniverseAuditError(
            "history universe manifest requested_dates is incomplete for start/end range: "
            f"expected={expected_requested_dates}, manifest={sorted(manifest_requested_dates)}"
        )
    if set(manifest_requested_dates) != set(date_statuses):
        raise UniverseAuditError(
            "history universe manifest requested_dates does not match date audit records: "
            f"requested={sorted(manifest_requested_dates)}, audited={sorted(date_statuses)}"
        )
    if set(manifest_generated_dates) != set(successful_snapshot_dates):
        raise UniverseAuditError(
            "history universe manifest generated_dates does not match successful snapshots: "
            f"manifest={sorted(manifest_generated_dates)}, snapshots={successful_snapshot_dates}"
        )
    sample_signal_dates = sorted(
        raw_samples["signal_date"].dropna().astype(str).unique().tolist()
        if "signal_date" in raw_samples.columns
        else []
    )
    manifest_sample_signal_dates = sorted(
        str(value) for value in manifest.get("sample_signal_dates", [])
    )
    if len(manifest_sample_signal_dates) != len(set(manifest_sample_signal_dates)):
        raise UniverseAuditError("history universe manifest sample_signal_dates contains duplicates")
    if manifest_sample_signal_dates != sample_signal_dates:
        raise UniverseAuditError(
            "history universe manifest sample_signal_dates does not match samples: "
            f"manifest={manifest_sample_signal_dates}, samples={sample_signal_dates}"
        )
    sample_counts = (
        raw_samples.assign(signal_date=raw_samples["signal_date"].astype(str))
        .groupby("signal_date", dropna=False)
        .size()
        .to_dict()
        if not raw_samples.empty and "signal_date" in raw_samples.columns
        else {}
    )
    unexpected_sample_dates = sorted(set(sample_counts).difference(successful_snapshot_dates))
    if unexpected_sample_dates:
        raise UniverseAuditError(
            "history universe samples contain dates without successful snapshots: "
            f"{unexpected_sample_dates}"
        )
    for row in date_records:
        requested_date = str(row.get("requested_signal_date", ""))
        snapshot_status = str(row.get("snapshot_status", ""))
        if snapshot_status not in verified_statuses:
            continue
        candidate_count = pd.to_numeric(
            pd.Series([row.get("candidate_row_count")]), errors="coerce"
        ).iloc[0]
        if pd.isna(candidate_count) or float(candidate_count) < 0 or not float(candidate_count).is_integer():
            raise UniverseAuditError(
                f"history universe manifest has invalid candidate_row_count for {requested_date}: "
                f"{row.get('candidate_row_count')}"
            )
        actual_count = int(sample_counts.get(requested_date, 0))
        if actual_count != int(candidate_count):
            raise UniverseAuditError(
                "history universe samples row count does not match candidate_row_count: "
                f"date={requested_date}, samples={actual_count}, audit={int(candidate_count)}"
            )
    allowed_complete_statuses = {*verified_statuses, "PROVEN_NON_TRADING_DATE"}
    all_dates_accounted_for = bool(date_statuses) and all(
        status in allowed_complete_statuses for status in date_statuses.values()
    )
    lookback_complete = all(
        not str(row.get("lookback_unresolved_dates", "")).strip()
        for row in date_records
    )
    generation_status_consistent = all(
        (
            str(row.get("generation_status", "")) == "generated"
            if str(row.get("snapshot_status", "")) in verified_statuses
            else str(row.get("generation_status", "")) == "non_trading"
            if str(row.get("snapshot_status", "")) == "PROVEN_NON_TRADING_DATE"
            else False
        )
        for row in date_records
    )
    all_dates_accounted_for = (
        all_dates_accounted_for and generation_status_consistent and lookback_complete
    )
    computed_complete = all_dates_accounted_for and not unexpected_sample_dates
    manifest_complete = bool(manifest.get("candidate_universe_snapshot_complete", computed_complete))
    manifest_verified = bool(manifest.get("candidate_universe_snapshot_verified", manifest_complete))
    if manifest_complete and not computed_complete:
        raise UniverseAuditError("history universe manifest incorrectly claims a complete candidate snapshot")
    snapshot_complete = computed_complete and manifest_complete
    snapshot_verified = snapshot_complete and manifest_verified
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "snapshot_complete": snapshot_complete,
        "snapshot_verified": snapshot_verified,
        "audit_status": "VERIFIED" if snapshot_verified else str(
            manifest.get("universe_audit_status", "MANIFEST_UNVERIFIED")
        ),
        "date_statuses": date_statuses,
        **duplicate_meta,
    }


def _infer_history_universe_manifest(samples_path: Path | None) -> Path | None:
    if samples_path is None:
        return None
    prefix = "history_candidates_"
    if not samples_path.stem.startswith(prefix):
        return None
    suffix = samples_path.stem[len(prefix):]
    return samples_path.parent / f"history_universe_manifest_{suffix}.json"


def _write_holdout_universe_outputs(
    output_dir: Path,
    scored: pd.DataFrame,
    scored_path: Path,
    samples_path: Path | None,
    raw_samples: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    selected_combos: pd.DataFrame,
    policy_daily: pd.DataFrame,
    candidate_top_k: int,
    top_n: int,
    v004a_l2: float,
    v004a_positive_weight: float,
    ranking_model_id: str,
    coefficient_meta: dict[str, Any],
    history_universe: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    v004a = scored[
        scored["model_id"].astype(str).eq(MODEL_ID_V004A)
        & scored["evaluation_scope"].astype(str).eq(SCOPE_WALK_FORWARD)
        & pd.to_numeric(scored["l2"], errors="coerce").sub(float(v004a_l2)).abs().le(1e-9)
        & pd.to_numeric(scored["positive_weight"], errors="coerce").sub(float(v004a_positive_weight)).abs().le(1e-9)
    ].copy()
    v002 = scored[
        scored["model_id"].astype(str).eq(str(ranking_model_id))
        & scored["evaluation_scope"].astype(str).eq(SCOPE_WALK_FORWARD)
    ].copy()
    require_nonempty_codes(v004a, "code", "configured v004a scored rows")
    require_nonempty_codes(v002, "code", "configured v002 scored rows")
    v004a["code"] = normalize_code_series(v004a["code"])
    v002["code"] = normalize_code_series(v002["code"])
    v004a["signal_date"] = v004a["signal_date"].astype(str)
    v002["signal_date"] = v002["signal_date"].astype(str)
    duplicate_scored_key_count = count_duplicate_keys(v004a, ("signal_date", "code")) + count_duplicate_keys(
        v002, ("signal_date", "code")
    )
    require_unique_keys(v004a, ("signal_date", "code"), "holdout v004a scored rows")
    require_unique_keys(v002, ("signal_date", "code"), "holdout v002 scored rows")

    samples_available = samples_path is not None and samples_path.is_file()
    annotated_raw = annotate_v004a_input_eligibility(raw_samples) if samples_available else pd.DataFrame()
    if samples_available:
        require_nonempty_codes(annotated_raw, "code", "holdout annotated samples")
        annotated_raw["signal_date"] = annotated_raw["signal_date"].astype(str)
        annotated_raw["code"] = normalize_code_series(annotated_raw["code"])
        expected_scorable = annotated_raw[
            annotated_raw["v004a_scorable_bool"].fillna(False).astype(bool)
        ].copy()
        require_nonempty_codes(expected_scorable, "code", "expected scorable samples")
        require_unique_keys(
            expected_scorable,
            ("signal_date", "code"),
            "expected scorable samples",
        )
        _require_exact_scored_keys(expected_scorable, v004a, "configured v004a scored rows")
        _require_exact_scored_keys(expected_scorable, v002, "configured v002 scored rows")
        membership_base = expected_scorable.copy()
    else:
        expected_scorable = pd.DataFrame()
        membership_base = v004a.copy()

    v004a = v004a.rename(
        columns={"model_score": "v004a_model_score", "model_rank": "v004a_model_rank"}
    )
    v002 = v002[["signal_date", "code", "model_score", "model_rank"]].rename(
        columns={"model_score": "v002_model_score", "model_rank": "v002_model_rank"}
    )
    v004a_payload_columns = ["signal_date", "code", "v004a_model_score", "v004a_model_rank"]
    for column in HOLDOUT_UNIVERSE_MEMBERSHIP_COLUMNS:
        if (
            column in v004a.columns
            and column not in membership_base.columns
            and column not in v004a_payload_columns
            and not column.startswith("v002_")
            and not column.startswith("in_")
            and column != "final_top3_rank"
        ):
            v004a_payload_columns.append(column)
    membership = membership_base.merge(
        v004a[v004a_payload_columns],
        on=["signal_date", "code"],
        how="left",
        validate="one_to_one",
    ).merge(
        v002,
        on=["signal_date", "code"],
        how="left",
        validate="one_to_one",
    )
    if "name" not in membership.columns:
        membership["name"] = ""
    if "v004a_scorable_bool" not in membership.columns:
        membership["v004a_scorable_bool"] = True
    if "eligible_for_trade" not in membership.columns:
        membership["eligible_for_trade"] = True

    v004a_top_keys = _key_set_from_rank(v004a, "v004a_model_rank", candidate_top_k)
    v002_top_keys = _key_set_from_rank(v002, "v002_model_rank", candidate_top_k)
    candidate_keys = set(
        zip(candidate_pool["signal_date"].astype(str), normalize_code_series(candidate_pool["code"]))
    )
    final_rank = _final_top3_rank_map(policy_daily, selected_combos, top_n)
    membership_keys = list(zip(membership["signal_date"].astype(str), membership["code"].astype(str)))
    membership["in_v004a_topk"] = [key in v004a_top_keys for key in membership_keys]
    membership["in_v002_topk"] = [key in v002_top_keys for key in membership_keys]
    membership["in_v005_candidate_pool"] = [key in candidate_keys for key in membership_keys]
    membership["in_final_top3"] = [key in final_rank for key in membership_keys]
    membership["final_top3_rank"] = [final_rank.get(key, pd.NA) for key in membership_keys]
    for column in HOLDOUT_UNIVERSE_MEMBERSHIP_COLUMNS:
        if column not in membership.columns:
            membership[column] = pd.NA
    membership = membership[HOLDOUT_UNIVERSE_MEMBERSHIP_COLUMNS].sort_values(
        ["signal_date", "code"], kind="mergesort"
    ).reset_index(drop=True)
    require_nonempty_codes(membership, "code", "holdout membership")

    if samples_available:
        annotated_raw["__audit_date"] = (
            annotated_raw["requested_signal_date"].astype(str)
            if "requested_signal_date" in annotated_raw.columns
            else annotated_raw["signal_date"].astype(str)
        )
        annotated_raw["code"] = normalize_code_series(annotated_raw["code"])
    audit_rows: list[dict[str, Any]] = []
    dates = sorted(membership["signal_date"].dropna().astype(str).unique().tolist())
    for signal_date in dates:
        day = membership[membership["signal_date"].astype(str).eq(signal_date)].copy()
        day_v4 = v004a[v004a["signal_date"].astype(str).eq(signal_date)].sort_values(
            ["v004a_model_rank", "code"], kind="mergesort"
        )
        day_v2 = v002[v002["signal_date"].astype(str).eq(signal_date)].sort_values(
            ["v002_model_rank", "code"], kind="mergesort"
        )
        day_candidate = candidate_pool[candidate_pool["signal_date"].astype(str).eq(signal_date)].sort_values(
            ["v004a_model_rank", "code"], kind="mergesort"
        )
        day_final = day[day["in_final_top3"].fillna(False).astype(bool)].sort_values(
            ["final_top3_rank", "code"], kind="mergesort"
        )
        if samples_available:
            day_eligible = annotated_raw[
                annotated_raw["__audit_date"].eq(signal_date)
                & annotated_raw["eligible_for_trade"].fillna(False).astype(bool)
            ]
            eligible_count: Any = int(len(day_eligible))
            eligible_hash = code_set_sha256(day_eligible)
            eligible_count_available = True
            eligible_source = "samples_annotated_v004a_input"
        else:
            day_eligible = pd.DataFrame()
            eligible_count = pd.NA
            eligible_hash = ""
            eligible_count_available = False
            eligible_source = "unavailable_pre_scored_only"
        day_v4_top = day_v4[pd.to_numeric(day_v4["v004a_model_rank"], errors="coerce").le(candidate_top_k)]
        day_v2_top = day_v2[pd.to_numeric(day_v2["v002_model_rank"], errors="coerce").le(candidate_top_k)]
        audit_rows.append(
            {
                "signal_date": signal_date,
                "eligible_count": eligible_count,
                "eligible_count_available": eligible_count_available,
                "eligible_source": eligible_source,
                "eligible_code_set_sha256": eligible_hash,
                "scorable_count": int(len(day)),
                "scorable_code_set_sha256": code_set_sha256(day),
                "v004a_topk_count": int(len(day_v4_top)),
                "v004a_topk_code_set_sha256": code_set_sha256(day_v4_top),
                "v002_topk_count": int(len(day_v2_top)),
                "v002_topk_code_set_sha256": code_set_sha256(day_v2_top),
                "v005_candidate_pool_count": int(len(day_candidate)),
                "v005_candidate_pool_code_set_sha256": code_set_sha256(day_candidate),
                "final_top3_count": int(len(day_final)),
                "final_top3_code_set_sha256": code_set_sha256(day_final),
                "missing_v002_score_count": int(pd.to_numeric(day["v002_model_score"], errors="coerce").isna().sum()),
                "duplicate_scored_key_count": int(duplicate_scored_key_count),
                "snapshot_status": history_universe.get("date_statuses", {}).get(
                    signal_date, str(history_universe["audit_status"])
                ),
                "candidate_universe_snapshot_verified": bool(history_universe["snapshot_verified"]),
                "legacy_duplicate_key_count": int(history_universe["legacy_duplicate_key_count"]),
                "legacy_duplicate_row_count": int(history_universe["legacy_duplicate_row_count"]),
                "legacy_duplicate_codes": str(history_universe["legacy_duplicate_codes"]),
                "v004a_topk_codes": ",".join(day_v4_top["code"].astype(str).tolist()),
                "v002_topk_codes": ",".join(day_v2_top["code"].astype(str).tolist()),
                "v005_candidate_pool_codes": ",".join(day_candidate["code"].astype(str).tolist()),
                "final_top3_codes": ",".join(day_final["code"].astype(str).tolist()),
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    membership_path = output_dir / "v005_fixed_grid_holdout_universe_membership.csv"
    audit_path = output_dir / "v005_fixed_grid_holdout_universe_audit.csv"
    manifest_path = output_dir / "v005_fixed_grid_holdout_universe_manifest.json"
    atomic_write_csv(
        membership_path,
        membership,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    atomic_write_csv(
        audit_path,
        audit,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    final_members = membership[membership["in_final_top3"].fillna(False).astype(bool)]
    manifest = {
        "universe_snapshot_schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "samples_file": str(samples_path) if samples_path else "",
        "samples_file_sha256": file_sha256(samples_path) if samples_path and samples_path.is_file() else "",
        "scored_file": str(scored_path),
        "scored_file_sha256": file_sha256(scored_path),
        "history_universe_manifest_path": str(history_universe["manifest_path"]),
        "history_universe_manifest_sha256": str(history_universe["manifest_sha256"]),
        "candidate_universe_snapshot_complete": bool(history_universe["snapshot_complete"]),
        "candidate_universe_snapshot_verified": bool(history_universe["snapshot_verified"]),
        "universe_audit_status": str(history_universe["audit_status"]),
        "legacy_duplicate_key_count": int(history_universe["legacy_duplicate_key_count"]),
        "legacy_duplicate_row_count": int(history_universe["legacy_duplicate_row_count"]),
        "legacy_duplicate_codes": str(history_universe["legacy_duplicate_codes"]),
        "eligible_count_available": bool(samples_available),
        "eligible_source": (
            "samples_annotated_v004a_input"
            if samples_available
            else "unavailable_pre_scored_only"
        ),
        "candidate_top_k": int(candidate_top_k),
        "top_n": int(top_n),
        "ranking_model_id": str(ranking_model_id),
        "coefficient_metadata": coefficient_meta,
        "dates": json.loads(audit.to_json(orient="records")) if not audit.empty else [],
        "membership_canonical_rows_sha256": canonical_rows_sha256(
            membership,
            HOLDOUT_UNIVERSE_MEMBERSHIP_COLUMNS,
            ("signal_date", "code"),
        ),
        "final_top3_code_set_sha256": code_set_sha256(final_members),
        "final_top3_canonical_rows_sha256": canonical_rows_sha256(
            final_members,
            HOLDOUT_UNIVERSE_MEMBERSHIP_COLUMNS,
            ("signal_date", "final_top3_rank", "code"),
        ),
    }
    write_json(manifest_path, manifest)
    return membership, audit, manifest_path


def _require_exact_scored_keys(
    expected_scorable: pd.DataFrame,
    scored_rows: pd.DataFrame,
    label: str,
    preview_limit: int = 10,
) -> None:
    expected_keys = set(
        zip(
            expected_scorable["signal_date"].astype(str),
            normalize_code_series(expected_scorable["code"]),
        )
    )
    actual_keys = set(
        zip(
            scored_rows["signal_date"].astype(str),
            normalize_code_series(scored_rows["code"]),
        )
    )
    if expected_keys == actual_keys:
        return
    missing = sorted(expected_keys.difference(actual_keys))
    extra = sorted(actual_keys.difference(expected_keys))
    missing_preview = [f"{signal_date}|{code}" for signal_date, code in missing[: int(preview_limit)]]
    extra_preview = [f"{signal_date}|{code}" for signal_date, code in extra[: int(preview_limit)]]
    raise UniverseAuditError(
        f"{label} keys do not match expected scorable samples: "
        f"expected_count={len(expected_keys)}, actual_count={len(actual_keys)}, "
        f"missing_count={len(missing)}, extra_count={len(extra)}, "
        f"missing_preview={missing_preview}, extra_preview={extra_preview}"
    )


def _validate_configured_scored_coverage(
    scored: pd.DataFrame,
    raw_samples: pd.DataFrame,
    samples_available: bool,
    v004a_l2: float,
    v004a_positive_weight: float,
    ranking_model_id: str,
) -> None:
    v004a = scored[
        scored["model_id"].astype(str).eq(MODEL_ID_V004A)
        & scored["evaluation_scope"].astype(str).eq(SCOPE_WALK_FORWARD)
        & pd.to_numeric(scored["l2"], errors="coerce").sub(float(v004a_l2)).abs().le(1e-9)
        & pd.to_numeric(scored["positive_weight"], errors="coerce")
        .sub(float(v004a_positive_weight))
        .abs()
        .le(1e-9)
    ].copy()
    v002 = scored[
        scored["model_id"].astype(str).eq(str(ranking_model_id))
        & scored["evaluation_scope"].astype(str).eq(SCOPE_WALK_FORWARD)
    ].copy()
    require_nonempty_codes(v004a, "code", "configured v004a scored rows")
    require_nonempty_codes(v002, "code", "configured v002 scored rows")
    v004a["code"] = normalize_code_series(v004a["code"])
    v002["code"] = normalize_code_series(v002["code"])
    v004a["signal_date"] = v004a["signal_date"].astype(str)
    v002["signal_date"] = v002["signal_date"].astype(str)
    require_unique_keys(v004a, ("signal_date", "code"), "configured v004a scored rows")
    require_unique_keys(v002, ("signal_date", "code"), "configured v002 scored rows")
    if samples_available:
        annotated = annotate_v004a_input_eligibility(raw_samples)
        expected = annotated[
            annotated["v004a_scorable_bool"].fillna(False).astype(bool)
        ].copy()
        require_nonempty_codes(expected, "code", "expected scorable samples")
        require_unique_keys(expected, ("signal_date", "code"), "expected scorable samples")
        _require_exact_scored_keys(expected, v004a, "configured v004a scored rows")
        _require_exact_scored_keys(expected, v002, "configured v002 scored rows")
        return
    _require_exact_scored_keys(v004a, v002, "configured v002 scored rows")


def _key_set_from_rank(frame: pd.DataFrame, rank_column: str, top_k: int) -> set[tuple[str, str]]:
    selected = frame[pd.to_numeric(frame[rank_column], errors="coerce").le(int(top_k))]
    return set(zip(selected["signal_date"].astype(str), selected["code"].astype(str)))


def _final_top3_rank_map(
    policy_daily: pd.DataFrame,
    selected_combos: pd.DataFrame,
    top_n: int,
) -> dict[tuple[str, str], int]:
    mapping: dict[tuple[str, str], int] = {}
    policy = policy_daily[
        policy_daily.get("strategy", pd.Series(dtype=str)).astype(str).eq(PRIMARY_POLICY)
    ] if not policy_daily.empty else pd.DataFrame()
    if not policy.empty:
        for _, row in policy.sort_values("signal_date", kind="mergesort").iterrows():
            for rank, code in enumerate(parse_codes(row.get("selected_codes", ""))[: int(top_n)], start=1):
                mapping[(str(row["signal_date"]), str(code).zfill(6))] = rank
        return mapping
    for _, row in selected_combos.sort_values("signal_date", kind="mergesort").iterrows():
        for rank, code in enumerate(parse_codes(row.get("codes", ""))[: int(top_n)], start=1):
            mapping[(str(row["signal_date"]), str(code).zfill(6))] = rank
    return mapping


def build_holdout_scored_candidates(
    samples_file: Path,
    coefficients_file: Path,
    output_path: Path,
    coefficient_predict_date: str,
    v004a_l2: float,
    v004a_positive_weight: float,
    target_return_pct: float,
    ranking_model_file: Path = DEFAULT_RANKING_MODEL_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not samples_file.exists():
        raise RuntimeError(f"missing samples file: {samples_file}")
    raw = pd.read_csv(samples_file, dtype={"code": str})
    samples, feature_info, data_quality = prepare_v004a_samples(raw, target_return_pct=float(target_return_pct))
    beta, feature_columns, coefficient_meta = load_fixed_v004a_beta(
        coefficients_file=coefficients_file,
        coefficient_predict_date=coefficient_predict_date,
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    missing_features = [column for column in feature_columns if column not in samples.columns]
    if missing_features:
        raise RuntimeError(
            "holdout samples are missing v004a coefficient features: "
            + ", ".join(missing_features)
            + "; rerun history sample generation with the same factor columns used by v004a."
        )

    manual_models = _load_manual_models()
    ranking_model, ranking_meta = load_ranking_model(ranking_model_file)
    validate_ranking_model(ranking_model, samples.columns)
    manual_models.pop(V002_MODEL_ID, None)
    manual_models[str(ranking_meta["model_id"])] = ranking_model
    scored_frames = _score_manual_and_hand_models(samples, manual_models)
    scored_frames.append(
        _score_logistic_frame(
            samples,
            beta=beta,
            feature_columns=feature_columns,
            model_id=MODEL_ID_V004A,
            l2=float(v004a_l2),
            positive_weight=float(v004a_positive_weight),
            feature_set=",".join(feature_columns),
            interaction_set=feature_info.get("interaction_set", "none"),
        )
    )
    scored = add_scored_model_rank(pd.concat(scored_frames, ignore_index=True))
    output = build_scored_candidates_output(scored, feature_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_path, output)
    prepared = prepare_scored_candidates(output_path)
    coefficient_meta.update(
        {
            "ranking_model_path": str(ranking_meta["model_path"]),
            "ranking_model_normalized_sha256": str(ranking_meta["model_normalized_sha256"]),
            "ranking_model_id": str(ranking_meta["model_id"]),
        }
    )
    return prepared, data_quality, coefficient_meta


def load_fixed_v004a_beta(
    coefficients_file: Path,
    coefficient_predict_date: str,
    v004a_l2: float,
    v004a_positive_weight: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not coefficients_file.exists():
        raise RuntimeError(
            f"missing coefficients file: {coefficients_file}; "
            "run the prior v004a research once, or pass --scored-file with already scored holdout rows."
        )
    coef = pd.read_csv(coefficients_file)
    required = ["model_id", "evaluation_scope", "l2", "positive_weight", "predict_date", "term", "coefficient"]
    missing = [column for column in required if column not in coef.columns]
    if missing:
        raise RuntimeError(f"coefficients file missing columns: {missing}")
    frame = coef[
        (coef["model_id"].astype(str) == MODEL_ID_V004A)
        & (coef["evaluation_scope"].astype(str) == SCOPE_WALK_FORWARD)
        & (pd.to_numeric(coef["l2"], errors="coerce").sub(float(v004a_l2)).abs() <= 1e-9)
        & (pd.to_numeric(coef["positive_weight"], errors="coerce").sub(float(v004a_positive_weight)).abs() <= 1e-9)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"no v004a coefficients for l2={v004a_l2:g}, positive_weight={v004a_positive_weight:g}")
    frame["predict_date"] = frame["predict_date"].astype(str)
    if coefficient_predict_date and coefficient_predict_date != "latest":
        chosen_date = str(coefficient_predict_date)
    else:
        chosen_date = max(frame["predict_date"].dropna().astype(str).tolist())
    chosen = frame[frame["predict_date"] == chosen_date].copy()
    if chosen.empty:
        available = sorted(frame["predict_date"].dropna().astype(str).unique().tolist())
        raise RuntimeError(f"predict_date={chosen_date!r} not found in coefficients. Available tail: {available[-10:]}")
    if "fold_index" in chosen.columns:
        max_fold = pd.to_numeric(chosen["fold_index"], errors="coerce").max()
        chosen = chosen[pd.to_numeric(chosen["fold_index"], errors="coerce") == max_fold].copy()
    feature_set = str(chosen["feature_set"].dropna().astype(str).iloc[0]) if "feature_set" in chosen.columns else ""
    feature_columns = [part.strip() for part in feature_set.split(",") if part.strip()]
    if not feature_columns:
        feature_columns = [str(term) for term in chosen["term"].astype(str).tolist() if str(term) != "intercept"]
    coef_by_term = dict(zip(chosen["term"].astype(str), pd.to_numeric(chosen["coefficient"], errors="coerce")))
    missing_terms = [term for term in ["intercept", *feature_columns] if term not in coef_by_term]
    if missing_terms:
        raise RuntimeError(f"chosen coefficient fold is missing terms: {missing_terms}")
    beta = np.array([float(coef_by_term["intercept"]), *[float(coef_by_term[column]) for column in feature_columns]], dtype=float)
    meta = {
        "coefficients_file": str(coefficients_file),
        "coefficients_normalized_sha256": normalized_sha256(coefficients_file),
        "coefficient_predict_date": chosen_date,
        "coefficient_terms": len(feature_columns),
        "v004a_l2": float(v004a_l2),
        "v004a_positive_weight": float(v004a_positive_weight),
    }
    if "train_start" in chosen.columns:
        meta["coefficient_train_start"] = str(chosen["train_start"].dropna().astype(str).iloc[0])
    if "train_end" in chosen.columns:
        meta["coefficient_train_end"] = str(chosen["train_end"].dropna().astype(str).iloc[0])
    return beta, feature_columns, meta


def build_default_grid() -> pd.DataFrame:
    return build_rule_grid(
        min_total_rank_weight_grid=DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
        avg_total_rank_weight_grid=DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
        contains_v004a_top3_bonus_grid=DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
        contains_v002_top3_bonus_grid=DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
        extreme_vwap_penalty_grid=DEFAULT_EXTREME_VWAP_PENALTY_GRID,
        extreme_close_low_penalty_grid=DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
        extreme_price_penalty_grid=DEFAULT_EXTREME_PRICE_PENALTY_GRID,
        rank_dispersion_weight_grid=DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    )


def select_grid_params(grid: pd.DataFrame, grid_id: int) -> pd.Series:
    row = grid[pd.to_numeric(grid["grid_id"], errors="coerce") == int(grid_id)].copy()
    if row.empty:
        raise RuntimeError(f"grid_id={grid_id} not found in default v005 grid")
    return row.iloc[0]


def apply_fixed_policy(
    scored_path: Path,
    selected_combos: pd.DataFrame,
    top_n: int,
    v004a_l2: float,
    v004a_positive_weight: float,
    v002_model_id: str = V002_MODEL_ID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ctx = build_context(scored_path, v004a_l2, v004a_positive_weight, v002_model_id=v002_model_id)
    history = selected_combos[["signal_date", "grid_id"]].copy().rename(columns={"grid_id": "selected_grid_id"})
    history["grid_id"] = history["selected_grid_id"]
    baseline = build_baseline(history[["signal_date", "grid_id"]], selected_combos, ctx, top_n)
    v002 = build_topn(ctx, "v002_model_rank", top_n)
    v004a = build_topn(ctx, "v004a_model_rank", top_n)

    daily_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    for _, base in baseline.iterrows():
        date = str(base["signal_date"])
        day_ctx = ctx[ctx["signal_date"] == date].copy()
        v002_day = _one_date_row(v002, date, "v002")
        v004a_day = _one_date_row(v004a, date, "v004a")
        baseline_codes = parse_codes(base["codes"])
        v002_codes = parse_codes(v002_day["codes"])
        v004a_codes = parse_codes(v004a_day["codes"])
        choices = [
            ("baseline_v005_fixed_grid", "keep_v005_fixed_grid", baseline_codes, baseline_codes, False),
            policy_choice(base, baseline_codes, v002_codes),
            (DEFAULT_V002_MODEL_LABEL, "control_v002_top3", v002_codes, v002_codes, False),
            (DEFAULT_V004A_MODEL_LABEL, "control_v004a_top3", v004a_codes, v004a_codes, False),
        ]
        for strategy, action, selected_codes, source_codes, triggered in choices:
            selected = ctx_for_codes(day_ctx, selected_codes)
            met = metrics(selected, selected_codes, top_n)
            daily_rows.append(
                {
                    "strategy": strategy,
                    "signal_date": date,
                    "action": action,
                    "selected_codes": ",".join(norm(selected_codes)),
                    "source_codes": ",".join(norm(source_codes)),
                    "selected_grid_id": int(base["selected_grid_id"]),
                    **met,
                    "v002_codes": v002_day["codes"],
                    "v002_hit_count": int(v002_day["hit_count"]),
                    "v002_all_hit": bool(v002_day["all_hit"]),
                    "v002_avg_realized_return": float(v002_day["avg_realized_return"]),
                    "v004a_codes": v004a_day["codes"],
                    "v004a_hit_count": int(v004a_day["hit_count"]),
                    "v004a_all_hit": bool(v004a_day["all_hit"]),
                    "v004a_avg_realized_return": float(v004a_day["avg_realized_return"]),
                    "baseline_v005_codes": base["codes"],
                    "baseline_v005_hit_count": int(base["hit_count"]),
                    "baseline_v005_all_hit": bool(base["all_hit"]),
                    "baseline_v005_avg_realized_return": float(base["avg_realized_return"]),
                    "gate_v002_extreme_vwap_count": int(base["v002_extreme_vwap_count"]),
                    "gate_v002_extreme_close_low_count": int(base["v002_extreme_close_low_count"]),
                    "gate_v005_avg_v002_rank": float(base["v005_avg_v002_rank"]),
                    "gate_v005_has_risk_ticket": bool(base["v005_has_risk_ticket"]),
                    "gate_triggered": bool(triggered),
                }
            )
            replacement_rows.extend(
                replacement_detail(strategy, date, action, norm(selected_codes), baseline_codes, v002_codes, v004a_codes, day_ctx)
            )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        daily = pd.DataFrame(columns=HOLDOUT_DAILY_COLUMNS)
    else:
        daily = daily[HOLDOUT_DAILY_COLUMNS].sort_values(["strategy", "signal_date"])
    summary = summarize(daily, top_n)[FALLBACK_SUMMARY_COLUMNS] if not daily.empty else pd.DataFrame(columns=FALLBACK_SUMMARY_COLUMNS)
    replacement = pd.DataFrame(replacement_rows)
    if replacement.empty:
        replacement = pd.DataFrame(columns=REPLACEMENT_COLUMNS)
    else:
        replacement = replacement[REPLACEMENT_COLUMNS].sort_values(["strategy", "signal_date", "selection_bucket", "code"])
    return summary, daily, replacement


def policy_choice(base: pd.Series, baseline_codes: list[str], v002_codes: list[str]) -> tuple[str, str, list[str], list[str], bool]:
    if is_policy_fallback(base):
        return PRIMARY_POLICY, "fallback_to_v002_regime_policy", v002_codes, v002_codes, True
    return PRIMARY_POLICY, "keep_v005_fixed_grid", baseline_codes, baseline_codes, False


def _one_date_row(frame: pd.DataFrame, date: str, label: str) -> pd.Series:
    row = frame[frame["signal_date"].astype(str) == str(date)].copy()
    if row.empty:
        raise RuntimeError(f"missing {label} TopN row for signal_date={date}")
    return row.iloc[0]


def assess_holdout_readiness(
    policy_daily: pd.DataFrame,
    min_forward_dates: int,
    frozen_policy_inputs_verified: bool = True,
    deployment_status: str | None = None,
) -> pd.DataFrame:
    threshold = int(min_forward_dates)
    if threshold <= 0:
        raise ValueError("min_forward_dates must be positive")
    frozen_policy_inputs_verified = bool(frozen_policy_inputs_verified) and threshold == DEFAULT_POLICY.min_forward_dates
    rows = policy_daily.copy()
    if not rows.empty and "strategy" in rows.columns:
        rows = rows[rows["strategy"].astype(str) == PRIMARY_POLICY].copy()
    dates = sorted(rows.get("signal_date", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    date_count = len(dates)
    threshold_met = date_count >= threshold
    if not frozen_policy_inputs_verified:
        readiness_status = "UNVERIFIED_POLICY_INPUTS"
        reason = "Frozen policy inputs are not fully verified; this result is custom research only."
    elif threshold_met:
        readiness_status = "FORWARD_SAMPLE_THRESHOLD_MET"
        reason = "Unique signal dates in this holdout input meet the review threshold; deployment still requires execution and risk review."
    else:
        readiness_status = "INSUFFICIENT_FORWARD_SAMPLE"
        reason = f"Only {date_count} unique signal dates are present in this holdout input; at least {threshold} are required before deployment review."
    effective_deployment_status = deployment_status or (
        DEFAULT_POLICY.deployment_status if frozen_policy_inputs_verified else "custom_research_only"
    )
    return pd.DataFrame(
        [
            {
                "policy_version": (
                    DEFAULT_POLICY.policy_version
                    if frozen_policy_inputs_verified
                    else f"{DEFAULT_POLICY.policy_version}+custom_override"
                ),
                "deployment_status": effective_deployment_status,
                "research_only": True,
                "forward_date_count": date_count,
                "forward_signal_date_count": date_count,
                "min_forward_dates": threshold,
                "frozen_policy_inputs_verified": bool(frozen_policy_inputs_verified),
                "sample_threshold_met": threshold_met,
                "readiness_status": readiness_status,
                "deployable": False,
                "first_forward_date": dates[0] if dates else "",
                "last_forward_date": dates[-1] if dates else "",
                "reason": reason,
            }
        ]
    )


def make_report(
    samples_file: str | Path | None,
    scored_file: Path,
    coefficients_file: str | Path,
    output_dir: Path,
    grid_id: int,
    grid_params: pd.Series,
    coefficient_meta: dict[str, Any],
    candidate_pool: pd.DataFrame,
    combo_candidates: pd.DataFrame,
    selected_combos: pd.DataFrame,
    daily_top3: pd.DataFrame,
    summary: pd.DataFrame,
    policy_daily: pd.DataFrame,
    replacement: pd.DataFrame,
    readiness: pd.DataFrame,
    run_meta: pd.DataFrame,
    universe_audit: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# v005 fixed-grid holdout test",
        "",
        "## Scope",
        "",
        "This report evaluates the locked v005 set selector on a holdout samples file.",
        "It does not run factor discovery, does not select a new grid, and does not tune the fallback policy.",
        "The D2-open to D3-high target is an opportunity proxy, not executable net PnL.",
        "Readiness counts unique signal dates in this holdout input; it does not claim stock, D0, or sample independence.",
        "",
        "## Readiness gate",
        "",
    ]
    lines.extend(md_table(readiness, list(readiness.columns)))
    lines.extend(["", "## Universe Reproducibility", ""])
    audit = universe_audit if universe_audit is not None else pd.DataFrame()
    lines.extend(md_table(audit, list(audit.columns)))
    lines.extend(
        [
            "",
            "A matching candidate-universe hash proves only that the audited member sets are reproducible.",
            "`cache_snapshot_complete=False` still means the complete daily and 5-minute market-data cache is not frozen.",
            "Candidate-universe changes alter cross-sectional percentile features; runs with different universe hashes must not be compared directly.",
            "The v005 candidate pool remains the configured v004a TopK merged with v002 audit information.",
        ]
    )
    lines.extend(["", "## Run provenance", ""])
    if run_meta.empty:
        lines.append("_No provenance metadata._")
    else:
        for key, value in run_meta.iloc[0].items():
            lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Configuration",
        "",
        f"- samples file: `{samples_file or ''}`",
        f"- scored file: `{scored_file}`",
        f"- coefficients file: `{coefficients_file}`",
        f"- output dir: `{output_dir}`",
        f"- fixed v005 grid_id: `{grid_id}`",
        f"- candidate_pool rows: `{len(candidate_pool)}`",
        f"- combo candidate rows: `{len(combo_candidates)}`",
        f"- selected combo rows: `{len(selected_combos)}`",
        f"- daily top3 rows: `{len(daily_top3)}`",
        "",
        "## Fixed coefficient metadata",
        "",
    ])
    for key, value in coefficient_meta.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Fixed grid params", ""])
    grid_df = pd.DataFrame([{column: grid_params[column] for column in GRID_PARAM_COLUMNS if column in grid_params.index}])
    lines.extend(md_table(grid_df, GRID_PARAM_COLUMNS))
    lines.extend(["", "## Strategy summary", ""])
    lines.extend(md_table(summary, FALLBACK_SUMMARY_COLUMNS))
    lines.extend(["", "## Daily policy rows", ""])
    lines.extend(md_table(policy_daily, HOLDOUT_DAILY_COLUMNS))
    lines.extend(["", "## Fixed-grid selected combos", ""])
    lines.extend(
        md_table(
            selected_combos,
            [
                "grid_id",
                "signal_date",
                "combo_index",
                "codes",
                "combo_score",
                "hit_count",
                "all_hit",
                "avg_high_return",
                "avg_realized_return",
                "avg_v004a_rank",
                "avg_v002_rank",
                "extreme_price_count",
                "extreme_vwap_count",
                "extreme_close_low_count",
                "min_total_rank",
                "avg_total_rank",
            ],
        )
    )
    lines.extend(["", "## Replacement detail", ""])
    lines.extend(md_table(replacement, REPLACEMENT_COLUMNS))
    return "\n".join(lines)


def md_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df is None or df.empty:
        return ["_No rows._"]
    use = [column for column in columns if column in df.columns]
    rows = ["| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for _, row in df[use].iterrows():
        rows.append("| " + " | ".join(fmt(row[column]) for column in use) + " |")
    return rows


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed-grid v005 holdout evaluation without retuning.")
    parser.add_argument("--samples-file", default=DEFAULT_SAMPLES_FILE, help="Holdout history_candidates CSV. Required unless --scored-file is provided.")
    parser.add_argument("--scored-file", default=None, help="Optional pre-scored holdout candidates CSV; skips v004a/v002 holdout scoring.")
    parser.add_argument("--coefficients-file", default=str(DEFAULT_COEFFICIENTS_FILE), help="Existing v004a_coefficients.csv used to score holdout samples.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=DEFAULT_POLICY.top_n)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_POLICY.candidate_top_k)
    parser.add_argument("--grid-id", type=int, default=DEFAULT_GRID_ID)
    parser.add_argument("--coefficient-predict-date", default=DEFAULT_POLICY.coefficient_predict_date)
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_POLICY.v004a_l2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_POLICY.v004a_positive_weight)
    parser.add_argument("--target-return-pct", type=float, default=DEFAULT_TARGET_RETURN_PCT)
    parser.add_argument("--min-forward-dates", type=int, default=DEFAULT_POLICY.min_forward_dates)
    parser.add_argument("--ranking-model", default=str(DEFAULT_RANKING_MODEL_FILE))
    parser.add_argument("--history-universe-manifest", default=None)
    args = parser.parse_args(argv)

    summary, daily, replacement, daily_top3, report_path = run_fixed_grid_holdout(
        samples_file=args.samples_file,
        scored_file=args.scored_file,
        coefficients_file=args.coefficients_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        candidate_top_k=args.candidate_top_k,
        grid_id=args.grid_id,
        coefficient_predict_date=args.coefficient_predict_date,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
        target_return_pct=args.target_return_pct,
        min_forward_dates=args.min_forward_dates,
        ranking_model_file=args.ranking_model,
        history_universe_manifest_file=args.history_universe_manifest,
    )
    print(f"summary rows: {len(summary)}")
    print(f"daily rows: {len(daily)}")
    print(f"replacement rows: {len(replacement)}")
    print(f"daily top3 rows: {len(daily_top3)}")
    readiness_path = Path(args.output_dir) / "v005_fixed_grid_holdout_readiness.csv"
    readiness = pd.read_csv(readiness_path).iloc[0]
    print(f"readiness: {readiness['readiness_status']} ({int(readiness['forward_signal_date_count'])}/{int(readiness['min_forward_dates'])} unique signal dates)")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best strategy: {best['strategy']}")
        print(f"best top3_all_hit_rate: {float(best['top3_all_hit_rate']):.4f}")
        print(f"best avg_realized_return: {float(best['avg_top3_realized_return']):.4f}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

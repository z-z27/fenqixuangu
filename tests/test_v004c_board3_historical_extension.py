from __future__ import annotations

import inspect
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.v004c_board3_historical_extension as audit
import tools.recover_v004c_board3_historical_inputs_v001 as recovery
from src.v004c_board3_oof_risk_selectivity import (
    EXPECTED_COMMON_DATES,
    EXPECTED_LOCK_SHA256,
    date_bootstrap_counts,
    derive_relative_risk_surface,
    load_locked_population,
    tail_selectivity,
)
from src.v004c_pair_capped7_july_forward import prepare_frozen_x
from src.v004c_v4a_architecture_transfer import RAW_INPUT_COLUMNS, dataframe_csv_bytes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    first = audit.build_outputs(ROOT)
    second = audit.build_outputs(ROOT)
    assert first[0] == second[0]
    return first


def _csv(outputs: dict[str, bytes], filename: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(outputs[filename]), encoding="utf-8-sig")


def test_starting_and_frozen_contract():
    assert audit.EXPECTED_STARTING_HEAD == "2445eea52e8e4999bb6f122a1cc34b0cb0c9ea3a"
    audit.assert_contract()
    assert audit.HISTORICAL_BACKCAST_REPLICATION is True
    assert audit.PRISTINE_OOT is False
    assert audit.RECENT_START == "2026-05-01"
    assert audit.EXTENDED_START == "2026-01-01"
    assert audit.MIN_TRAIN_BOARD3_SIGNAL_DATES == 18
    assert len(audit.FROZEN_FEATURE_COLUMNS) == 18
    assert audit.L2 == .30
    assert audit.POSITIVE_WEIGHT == 1.50
    assert not any((
        audit.NEW_FEATURE, audit.NEW_LEARNER, audit.HYPERPARAMETER_SEARCH,
        audit.WINDOW_SEARCH, audit.MONTH_EXCLUSION, audit.RISK_THRESHOLD_SEARCH,
        audit.VETO_BACKFILL,
    ))
    assert audit.JULY_RESULT_ROWS_ACCESSED == 0


def test_baostock_recovery_uses_existing_authoritative_helpers():
    source = inspect.getsource(recovery)
    for helper in (
        "_prepare_daily_limitup_scan_frame",
        "_is_main_board_limit_up_day",
        "_count_consecutive_main_board_limit_ups",
        "_derive_limit_up_row_from_daily",
    ):
        assert helper in source
    assert recovery.DAILY_END == "2026-06-30"
    assert recovery.CANDIDATE_START == "2026-01-01"
    assert recovery.CANDIDATE_END == "2026-04-30"
    assert recovery.JULY_SENTINEL == "2026-07-01"


def test_recovered_jan_apr_candidate_and_source_parity():
    candidates = audit.load_historical_candidates(ROOT)
    daily, pools, minutes = audit.load_recovery_audits(ROOT)
    assert len(candidates) == 482
    assert candidates.signal_date.nunique() == 76
    assert int(candidates.board_streak_before_break.eq(2).sum()) == 391
    assert int(candidates.board_streak_before_break.eq(3).sum()) == 91
    assert candidates.signal_date.between("2026-01-01", "2026-04-30").all()
    assert candidates.event_id.nunique() == len(candidates)
    assert int(daily.status.eq("failed").sum()) == 0
    assert len(pools) == 85
    assert int(pools.status.eq("empty").sum()) == 0
    assert int(pools.rows.sum()) == 4919
    assert len(minutes) == 390
    assert int(minutes.status.eq("failed").sum()) == 0


def test_existing_may_june_and_matured_parity(built):
    _, context = built
    assert context["may_june_parity"] == {
        "rows": 319, "dates": 39, "may_rows": 146, "may_dates": 18,
        "june_rows": 173, "june_dates": 21, "board2": 261, "board3": 58,
        "july_result_rows_accessed": 0,
    }
    assert context["total_candidates"] == 801
    assert context["candidate_dates"] == 115
    assert context["board2_candidates"] == 652
    assert context["board3_candidates"] == 149
    assert context["board3_rows"] == 146
    assert context["board3_dates"] == 80
    assert context["historical_board3_rows"] == 91
    assert context["historical_board3_dates"] == 52
    assert context["july_result_rows_accessed"] == 0


def test_historical_feature_coverage_is_complete_after_suspension_rejection(built):
    outputs, context = built
    coverage = _csv(outputs, audit.OUTPUT_FILENAMES[0])
    failed = coverage[
        coverage.section.eq("FEATURE_RECONSTRUCTION") & coverage.status.eq("FAIL")
    ]
    assert context["data_status"] == "COMPLETE"
    assert failed.empty
    candidates = audit.load_historical_candidates(ROOT)
    assert "603778_2026-01-07" not in set(candidates.event_id)
    assert context["historical_board3_rows"] == 91


def test_full_date_ranks_are_computed_before_board3_filter():
    rows = []
    for index in range(10):
        row = {
            "event_id": f"e{index}", "signal_date": "2026-01-05",
            "code": f"{index + 1:06d}",
            "board_streak_before_break": 3 if index >= 8 else 2,
        }
        row.update({column: float(index + 1) for column in RAW_INPUT_COLUMNS})
        row["candidate_base_price"] = float(index + 1)
        rows.append(row)
    raw = pd.DataFrame(rows)
    full = prepare_frozen_x(raw)
    board3_after = full[full.board_streak_before_break.eq(3)]
    board3_recomputed = prepare_frozen_x(raw.iloc[8:].copy())
    assert board3_after.rank_d1_close_ma10_pct.tolist() == pytest.approx([.9, 1.0])
    assert board3_recomputed.rank_d1_close_ma10_pct.tolist() == pytest.approx([.5, 1.0])


def test_maturity_and_minimum_date_gates_are_exact():
    assert audit.extended_fold_is_eligible(17, 2) == (
        False, "MATURED_BOARD3_DATES_17_LT_18"
    )
    assert audit.extended_fold_is_eligible(18, 2) == (True, "")
    assert audit.extended_fold_is_eligible(18, 1) == (False, "UNFITTABLE_SINGLE_CLASS")


def test_recent_lock_and_shared_rows_are_exact(built):
    outputs, context = built
    recent, lock_audit = load_locked_population(ROOT)
    assert lock_audit["observed_lock_sha256"] == EXPECTED_LOCK_SHA256
    assert EXPECTED_LOCK_SHA256 == (
        "2f85e79738dbbedf19382db0da5f1136de02e84c69557ea5c85341bcf62fbd6b"
    )
    assert sorted(recent.signal_date.unique()) == EXPECTED_COMMON_DATES
    assert len(recent) == 19
    shared = _csv(outputs, audit.OUTPUT_FILENAMES[5])
    assert sorted(shared.signal_date.unique()) == EXPECTED_COMMON_DATES
    assert len(shared) == 19
    assert sorted(shared.event_id) == sorted(recent.event_id)
    assert context["compatibility"]["recent"]["rows"] == 19


def test_extended_oof_prediction_lock_and_support(built):
    outputs, context = built
    lock = _csv(outputs, "v004c_board3_historical_oof_prediction_lock_v001.csv")
    oof = _csv(outputs, audit.OUTPUT_FILENAMES[2])
    assert context["first_eligible_date"] == "2026-02-12"
    assert oof.signal_date.nunique() == 60
    assert len(oof) == 105
    assert int(oof.target7.sum()) == 44
    assert int(oof.positive_non_target.sum()) == 31
    assert int(oof.loss.sum()) == 30
    assert context["extended_lock_sha256"] == (
        "5612478c534992755faee9782c308c184f891751703a3f646e5d6706c2b1866f"
    )
    lock_bytes = outputs["v004c_board3_historical_oof_prediction_lock_v001.csv"]
    assert hashlib.sha256(lock_bytes).hexdigest() == context["extended_lock_sha256"]
    forbidden = {"target7", "loss", "raw_repair_return", "capped_return_7"}
    assert not forbidden.intersection(lock.columns)
    assert (oof.label_available_date < "2026-07-01").all()


def test_outcome_perturbation_cannot_change_prediction_lock(built):
    outputs, _ = built
    lock = _csv(outputs, "v004c_board3_historical_oof_prediction_lock_v001.csv")
    predictions = lock.copy()
    for column in ("target7", "loss", "raw_repair_return", "capped_return_7"):
        predictions[column] = np.arange(len(predictions), dtype=float) * 997.0
    rebuilt = audit.build_extended_prediction_lock(predictions)
    assert dataframe_csv_bytes(rebuilt) == dataframe_csv_bytes(lock)


def test_bottom_half_and_worst_one_rules_are_unchanged():
    frame = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(1, 5)],
        "signal_date": ["d"] * 4, "code": [f"{i:06d}" for i in range(1, 5)],
        "board3_candidate_count": [4] * 4,
        "board3_only_score": [.9, .8, .7, .6],
        "board3_only_internal_rank": [1, 2, 3, 4],
    })
    surface = derive_relative_risk_surface(frame)
    assert surface.risk_rank_percentile.tolist() == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert surface.loc[surface.bottom_half_flag, "board3_only_internal_rank"].tolist() == [3, 4]
    assert surface.loc[surface.worst_one_flag, "board3_only_internal_rank"].tolist() == [4]


def test_selectivity_arithmetic_is_exact():
    frame = pd.DataFrame({
        "signal_date": ["d"] * 10,
        "loss": [1] * 4 + [0] * 6,
        "target7": [0] * 4 + [1] * 4 + [0] * 2,
        "positive_non_target": [0] * 8 + [1] * 2,
        "flag": [True, True, True, False, True, False, False, False, True, False],
    })
    result = tail_selectivity(frame, "flag", "SYNTHETIC")
    assert result["loss_capture_rate"] == .75
    assert result["winner_removal_rate"] == .25
    assert result["pnt_removal_rate"] == .50
    assert result["selectivity_gap"] == .50


def test_expanded_risk_and_shared_compatibility_decisions(built):
    _, context = built
    expanded = context["expanded_risk"]
    assert expanded["dates"] == 60
    assert expanded["rows"] == 105
    assert expanded["nonloss_auc"] == pytest.approx(.6137777777777778)
    assert expanded["pair"]["concordance"] == pytest.approx(.7666666666666667)
    assert expanded["bottom"]["loss_capture_rate"] == pytest.approx(2 / 3)
    assert expanded["bottom"]["winner_removal_rate"] == pytest.approx(.5)
    assert expanded["bottom"]["selectivity_gap"] == pytest.approx(1 / 6)
    assert context["expanded_signal"] == "PARTIAL"

    recent, extended = context["compatibility"]["recent"], context["compatibility"]["extended"]
    assert recent["nonloss_auc"] == pytest.approx(9 / 13)
    assert extended["nonloss_auc"] == pytest.approx(34 / 39)
    assert recent["bottom"]["selectivity_gap"] == pytest.approx(.25)
    assert extended["bottom"]["selectivity_gap"] == pytest.approx(0)
    assert recent["bottom"]["winner_removal_rate"] == pytest.approx(.50)
    assert extended["bottom"]["winner_removal_rate"] == pytest.approx(.75)
    assert context["history_effect"] == "HARMFUL"
    assert context["temporal_compatibility"] == "DRIFTED"
    assert context["conclusion"] == "REJECT_OLD_HISTORY_EXTENSION"
    assert context["next_action"] == "REJECT_EXTENDED_HISTORY"


def test_harmful_gate_cannot_be_overridden_by_pooled_results():
    compatibility = {
        "recent": {
            "nonloss_auc": .70, "pair": {"concordance": .80},
            "bottom": {"selectivity_gap": .30, "winner_removal_rate": .30,
                       "loss_capture_rate": .70},
        },
        "extended": {
            "nonloss_auc": .58, "pair": {"concordance": .80},
            "bottom": {"selectivity_gap": .30, "winner_removal_rate": .30,
                       "loss_capture_rate": .70},
        },
    }
    bootstrap = {
        "SELECTIVITY_GAP_DELTA": {"direction_probability": 1.0},
        "NONLOSS_AUC_DELTA": {"direction_probability": 1.0},
    }
    assert audit.history_extension_effect(compatibility, bootstrap) == "HARMFUL"


def test_bootstrap_is_date_block_and_lodo_has_no_selection(built):
    _, context = built
    dates, counts = date_bootstrap_counts(
        context["expanded_risk"]["eligible"].signal_date,
        resamples=audit.BOOTSTRAP_RESAMPLES,
        seed=audit.EXPANDED_BOOTSTRAP_SEED,
    )
    assert counts.shape == (20_000, len(dates))
    assert (counts.sum(axis=1) == len(dates)).all()
    assert context["expanded_bootstrap"]["BOTTOM_HALF_SELECTIVITY_GAP"][
        "valid_resamples"
    ] <= 20_000
    assert context["expanded_lodo"]["BOTTOM_HALF_SELECTIVITY_GAP"][
        "valid_resamples"
    ] == 60
    assert context["compat_lodo"]["NONLOSS_AUC_DELTA"]["valid_resamples"] == 8


def test_no_month_cherry_pick_window_search_policy_or_july_path(built):
    _, context = built
    source = inspect.getsource(audit)
    assert "GOOD_MONTHS_ONLY" not in source
    assert "EXCLUDE_MONTH" not in source
    assert "GridSearch" not in source
    assert "RandomForest" not in source
    assert "XGBoost" not in source
    assert context["july_result_rows_accessed"] == 0
    assert context["history_effect"] == "HARMFUL"
    assert context["next_action"] == "REJECT_EXTENDED_HISTORY"


def test_output_contract_and_byte_identical_double_build(built):
    outputs, _ = built
    assert set(outputs) == {
        *audit.OUTPUT_FILENAMES,
        "v004c_board3_historical_oof_prediction_lock_v001.csv",
        "v004c_board3_historical_fold_audit_v001.csv",
    }
    monthly = _csv(outputs, audit.OUTPUT_FILENAMES[1])
    assert monthly.all_candidates.sum() == 801
    assert monthly.board2_candidates.sum() == 652
    assert monthly.board3_candidates.sum() == 149
    assert monthly.matured_board3_rows.sum() == 146

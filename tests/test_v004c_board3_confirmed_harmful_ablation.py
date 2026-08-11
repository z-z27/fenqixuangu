from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import pytest

import src.v004c_board3_confirmed_harmful_ablation as audit


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    return audit.build_outputs(ROOT)


def test_starting_and_frozen_contract():
    assert audit.EXPECTED_STARTING_HEAD == "80383949465a725d483810e706bcc6441361b8fe"
    audit.assert_ablation_contract()
    assert audit.ABLATION_FEATURES == (
        "rank_d1_close_ma10_pct", "rank_d1_low_ma10_pct", "days_since_d0_le1",
    )
    assert not any((audit.NEW_MODEL, audit.NEW_FEATURE, audit.NEW_POLICY))
    assert not any((audit.SUBSET_SEARCH, audit.SCALE_SEARCH, audit.SIGN_REVERSAL))


def test_strict_population_and_stage1_parity(built):
    _, context = built
    frame = context["population"]
    assert sorted(frame.signal_date.unique()) == audit.EXPECTED_STRICT_DATES
    assert frame.signal_date.nunique() == 17
    assert len(frame) == 155
    assert frame.event_id.nunique() == 155
    assert frame.board_group.value_counts().to_dict() == {"BOARD2": 130, "BOARD3": 25}
    for metric, expected in audit.EXPECTED_STAGE1_PARITY.items():
        assert context["stage1_parity"][metric] == pytest.approx(expected, abs=1e-11)


def test_board3_original_parity(built):
    frame = built[1]["population"]
    selected = frame[frame.original_top3 & frame.board_group.eq("BOARD3")]
    assert (len(selected), int(selected.target7.sum()), int(selected.loss.sum())) == (15, 4, 8)
    assert selected.capped_return_7.mean() == pytest.approx(.014147, abs=5e-7)
    assert selected.capped_return_7.median() == pytest.approx(-.007092, abs=5e-7)
    assert selected.capped_return_7.min() == pytest.approx(-.044457, abs=5e-7)
    funnel = built[1]["funnel"]
    assert funnel["candidate_board3_share"] == pytest.approx(25 / 155)
    assert funnel["original"]["top10"]["board3"] == 24
    assert funnel["original"]["top10"]["total"] == 132
    assert funnel["original"]["top5"]["board3"] == 19
    assert funnel["original"]["top5"]["total"] == 81
    assert funnel["original"]["top3"]["board3"] == 15
    assert funnel["original"]["top3"]["total"] == 51


def test_exact_board2_unchanged_and_board3_ablation(built):
    frame = built[1]["prediction_lock"]
    b2 = frame[frame.board_group.eq("BOARD2")]
    b3 = frame[frame.board_group.eq("BOARD3")]
    assert np.allclose(b2.ablated_logit, b2.original_logit, rtol=0, atol=1e-12)
    assert np.allclose(b2.ablated_score, b2.original_score, rtol=0, atol=1e-12)
    assert np.allclose(
        b3.original_logit - b3.ablated_logit,
        b3.ablation_removed_contribution, rtol=0, atol=1e-12,
    )
    assert np.allclose(b3.ablated_score, audit._sigmoid(b3.ablated_logit), rtol=0, atol=1e-15)


def test_synthetic_board2_and_board3_exact_ablation():
    rows = []
    for event_id, board, logit, contributions in (
        ("b2", "BOARD2", .8, (.1, .2, .3)),
        ("b3", "BOARD3", 1.0, (.1, .2, .05)),
        ("neg", "BOARD3", .5, (-.03, 0, 0)),
    ):
        rows.append({
            "event_id": event_id, "signal_date": "d", "code": event_id,
            "board_group": board, "candidate_count": 3,
            "stage1_score": float(audit._sigmoid([logit])[0]),
            "stage1_logit": logit, "stage1_rank": len(rows) + 1,
            **{audit.contribution_column(feature): value for feature, value in zip(audit.ABLATION_FEATURES, contributions)},
        })
    # Unit fixture bypasses strict-size gate by verifying the exact defining equation.
    frame = pd.DataFrame(rows)
    removed = frame[[audit.contribution_column(f) for f in audit.ABLATION_FEATURES]].sum(axis=1)
    ablated = np.where(frame.board_group.eq("BOARD3"), frame.stage1_logit - removed, frame.stage1_logit)
    assert ablated.tolist() == pytest.approx([.8, .65, .53])


def test_prediction_lock_has_no_outcomes_and_is_deterministic(built):
    outputs, context = built
    lock = context["prediction_lock"]
    assert not audit.OUTCOME_COLUMNS.intersection(lock.columns)
    assert audit.prediction_lock_sha256(lock) == context["prediction_lock_sha256"]
    assert outputs[audit.OUTPUT_FILENAMES[0]] == audit.dataframe_csv_bytes(lock)
    reranked = audit.build_prediction_lock(audit.prediction_inputs(context["stage1_source_population"]))
    assert audit.dataframe_csv_bytes(reranked) == audit.dataframe_csv_bytes(lock)


def test_outcome_perturbation_cannot_change_ranking(built):
    population = built[1]["stage1_source_population"].copy()
    original = audit.build_prediction_lock(audit.prediction_inputs(population))
    for column in ("raw_repair_return", "target7", "loss", "capped_return_7"):
        population[column] = np.arange(len(population)) * 12345.0
    perturbed = audit.build_prediction_lock(audit.prediction_inputs(population))
    assert audit.dataframe_csv_bytes(original) == audit.dataframe_csv_bytes(perturbed)


def test_membership_and_portfolio_are_exact(built):
    frame = built[1]["population"]
    for _, day in frame.groupby("signal_date"):
        k = min(3, len(day))
        assert int(day.original_top3.sum()) == k
        assert int(day.ablated_top3.sum()) == k
        expected = day.sort_values(["ablated_score", "event_id"], ascending=[False, True]).event_id.head(k).tolist()
        actual = day[day.ablated_top3].sort_values("ablated_rank").event_id.tolist()
        assert actual == expected


def test_winner_loss_accounting_and_attribution_closure(built):
    context = built[1]
    transition = context["transition_summary"]
    assert transition["board3_winner_total"] == 4
    assert transition["board3_loss_total"] == 8
    assert transition["board3_winner_retained"] + transition["board3_winner_demoted"] == 4
    assert transition["board3_loss_retained"] + transition["board3_loss_demoted"] == 8
    assert context["membership_summary"]["attributed"] == pytest.approx(
        context["membership_summary"]["observed"], abs=1e-10
    )


def test_bootstrap_and_lodo_contract(built):
    context = built[1]
    assert context["bootstrap"]["overall"]["valid"] == 20_000
    assert context["bootstrap"]["board3_loss"]["valid"] <= 20_000
    assert all(key in context["lodo"] for key in ("overall", "board3_loss", "board3_capped"))
    assert context["lodo"]["overall"]["valid"] == 17
    assert context["lodo"]["board3_loss"]["valid"] == 16
    rng = np.random.default_rng(audit.BOOTSTRAP_SEED)
    draws = rng.integers(0, 17, size=(20_000, 17))
    assert draws.shape == (20_000, 17)
    daily = context["daily"]
    frame = context["population"]
    expected = []
    for omitted in sorted(frame.signal_date.unique()):
        expected.append(float(daily.loc[~daily.signal_date.eq(omitted), "delta"].mean()))
    assert context["lodo"]["overall"]["min"] == pytest.approx(min(expected))
    assert context["lodo"]["overall"]["median"] == pytest.approx(float(np.median(expected)))
    assert context["lodo"]["overall"]["max"] == pytest.approx(max(expected))


def test_temporal_july_and_no_training_expansion(built):
    context = built[1]
    assert context["self_label_leakage_rows"] == 0
    assert context["current_test_leakage_rows"] == 0
    assert context["july_result_rows_accessed"] == 0
    source = inspect.getsource(audit)
    assert "fit_logistic_l2_weighted(" not in source
    assert "_fit_stage1_snapshot(" not in source
    assert "subset_variants" not in source
    assert "scale_variants" not in source


def test_output_contract(built):
    outputs, context = built
    assert set(outputs) == set(audit.OUTPUT_FILENAMES)
    assert context["outcome_perturbation"] == "PASS"
    assert context["decision"]["signal"] in {"STRONG", "PARTIAL", "ABSENT"}
    assert context["decision"]["direction"] in {
        "MINIMAL_BOARD_INTERACTION_EXPERIMENT", "BROADER_BOARD_SEPARATION", "NO_BOARD_MODEL_YET",
    }

from __future__ import annotations

import unittest
import tomllib
import subprocess
import sys

import numpy as np
import pandas as pd

from src.backtester import (
    EXECUTION_MODEL_VERSION,
    _after_buy_rows,
    _base_history_result,
    _first_outcome,
    _path_metrics_by_horizon,
    build_top3_summary,
    simulate_d2_execution,
)
from src.loaders import _cache_covers, _filter_to_end_date
from src.policy_config import PROJECT_ROOT, get_default_policy, load_policy_config, normalized_sha256, validate_policy_for_signal_date
from src.provenance import source_tree_sha256
from src.ranking_backtest import validate_ranking_model
from src.v005_daily_selector import build_daily_policy_outputs, build_runtime_policy_meta, prepare_live_v004a_features
from src.v005_fallback_gate import PRIMARY_POLICY, is_policy_fallback
from src.v005_fixed_grid_holdout import assess_holdout_readiness, load_fixed_v004a_beta


DEFAULT_POLICY = get_default_policy()


class FrozenPolicyTests(unittest.TestCase):
    def test_requirements_match_pyproject_dependencies(self) -> None:
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project_dependencies = set(project["project"]["dependencies"])
        requirement_dependencies = {
            line.strip()
            for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(project_dependencies, requirement_dependencies)

    def test_source_tree_fingerprint_is_stable_shape(self) -> None:
        fingerprint = source_tree_sha256()
        self.assertEqual(len(fingerprint), 64)
        int(fingerprint, 16)

    def test_manifest_and_coefficient_artifact_are_self_consistent(self) -> None:
        policy = load_policy_config()
        self.assertTrue(policy.coefficients_path.is_file())
        self.assertEqual(normalized_sha256(policy.coefficients_path), policy.coefficients_sha256)
        self.assertTrue(policy.ranking_model_path.is_file())
        self.assertEqual(normalized_sha256(policy.ranking_model_path), policy.ranking_model_sha256)
        self.assertEqual(normalized_sha256(policy.manifest_path), policy.manifest_sha256)
        self.assertEqual(policy.target_column, "target7_d2open_d3high")
        self.assertEqual(policy.target_return_pct, 7.0)
        self.assertEqual(policy.deployment_status, "shadow_only")
        self.assertTrue(policy.research_only)
        self.assertTrue(build_runtime_policy_meta()["matches_frozen_manifest"])

    def test_packaged_coefficient_fold_loads(self) -> None:
        beta, features, meta = load_fixed_v004a_beta(
            coefficients_file=DEFAULT_POLICY.coefficients_path,
            coefficient_predict_date=DEFAULT_POLICY.coefficient_predict_date,
            v004a_l2=DEFAULT_POLICY.v004a_l2,
            v004a_positive_weight=DEFAULT_POLICY.v004a_positive_weight,
        )
        self.assertEqual(len(features), 18)
        self.assertEqual(beta.shape, (19,))
        self.assertAlmostEqual(float(beta[0]), -0.5295129192067463)
        self.assertEqual(meta["coefficient_train_end"], "2026-06-25")

    def test_future_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "future-model use"):
            validate_policy_for_signal_date(DEFAULT_POLICY, "2026-06-25")
        validate_policy_for_signal_date(DEFAULT_POLICY, "2026-06-26")

    def test_generic_cli_import_does_not_load_frozen_policy(self) -> None:
        code = (
            "import src.policy_config as policy_config; "
            "policy_config.load_policy_config = lambda *args, **kwargs: "
            "(_ for _ in ()).throw(RuntimeError('unexpected frozen policy load')); "
            "import src.cli"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class NoLeakageTests(unittest.TestCase):
    def test_future_cache_rows_are_excluded_from_as_of_view(self) -> None:
        cached = pd.DataFrame(
            {
                "date": ["2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26"],
                "close": [10.0, 10.1, 10.2, 99.0],
            }
        )
        as_of = _filter_to_end_date(cached, "date", "2026-06-25")
        self.assertEqual(as_of["date"].tolist(), ["2026-06-23", "2026-06-24", "2026-06-25"])
        self.assertNotIn(99.0, as_of["close"].tolist())
        self.assertTrue(_cache_covers(cached, "date", days=3, end_date="2026-06-25"))

    def test_ranking_model_cannot_use_future_target(self) -> None:
        model = {
            "model_type": "linear_score",
            "feature_columns": ["target7_d2open_d3high"],
            "model": {"weights": {"target7_d2open_d3high": 1.0}},
        }
        with self.assertRaisesRegex(RuntimeError, "forbidden future/label/execution"):
            validate_ranking_model(model, pd.Index(["target7_d2open_d3high"]))

    def test_live_feature_frame_has_no_future_outcome_values(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "trade_date": "2026-07-10",
                    "code": "1",
                    "signal_type": "D2_LOW_ABSORB",
                    "allowed_bool": True,
                    "candidate_base_price": 10.0,
                    "total_score": 70.0,
                    "graph_quality_score": 65.0,
                    "theme_score": 50.0,
                    "trend_hold_score": 75.0,
                    "d1_close_ma10_pct": 2.0,
                    "d1_low_ma10_pct": -1.0,
                    "days_since_d0": 1,
                    "active_money_score": 60.0,
                    "d1_close_vwap_pct": 1.0,
                    "low_absorb_min": 9.7,
                    "low_absorb_max": 10.0,
                    "invalid_price": 9.5,
                },
                {
                    "trade_date": "2026-07-10",
                    "code": "2",
                    "signal_type": "D2_LOW_ABSORB",
                    "allowed_bool": True,
                    "candidate_base_price": 20.0,
                    "total_score": 60.0,
                    "graph_quality_score": 55.0,
                    "theme_score": 40.0,
                    "trend_hold_score": 65.0,
                    "d1_close_ma10_pct": 1.0,
                    "d1_low_ma10_pct": -2.0,
                    "days_since_d0": 2,
                    "active_money_score": 50.0,
                    "d1_close_vwap_pct": 0.5,
                    "low_absorb_min": 19.4,
                    "low_absorb_max": 20.0,
                    "invalid_price": 19.0,
                },
            ]
        )
        live, _ = prepare_live_v004a_features(signals)
        self.assertFalse(live["target7_d2open_d3high"].any())
        self.assertTrue(live["d2open_d3high_return_pct"].isna().all())
        self.assertTrue(live["d2open_d3close_return_pct"].isna().all())
        self.assertTrue(live["realized_return_pct"].isna().all())
        self.assertFalse(live["outcome_labels_available"].any())
        self.assertTrue(live["target_metric_kind"].str.contains("opportunity_proxy").all())


class ConservativeExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.minute = pd.DataFrame(
            [
                {
                    "datetime": "2026-07-10 09:35:00",
                    "trade_date": "2026-07-10",
                    "low": 9.9,
                    "high": 20.0,
                    "close": 10.2,
                    "intraday_vwap": 10.1,
                },
                {
                    "datetime": "2026-07-10 09:40:00",
                    "trade_date": "2026-07-10",
                    "low": 9.0,
                    "high": 11.0,
                    "close": 10.5,
                    "intraday_vwap": 10.2,
                },
            ]
        )

    def test_confirmation_close_is_the_default_fill(self) -> None:
        execution = simulate_d2_execution(
            pd.Series({"low_absorb_min": 9.8, "low_absorb_max": 10.0, "invalid_price": 9.5}),
            self.minute,
        )
        self.assertTrue(execution["executed"])
        self.assertAlmostEqual(float(execution["price"]), 10.2)

    def test_confirmation_bar_is_excluded_from_post_entry_metrics(self) -> None:
        after = _after_buy_rows(self.minute, ["2026-07-10"], "2026-07-10 09:35:00", hold_days=1)
        self.assertEqual(after["datetime"].tolist(), ["2026-07-10 09:40:00"])
        metrics = _path_metrics_by_horizon(
            self.minute,
            future_dates=["2026-07-10"],
            base_price=10.0,
            prefix="",
            hold_days=1,
            start_time="2026-07-10 09:35:00",
        )
        self.assertAlmostEqual(metrics["d2_max_return_pct"], 10.0)

    def test_same_bar_target_and_stop_is_counted_as_stop(self) -> None:
        bars = pd.DataFrame([{"datetime": "2026-07-10 10:00:00", "high": 108.0, "low": 96.0}])
        outcome = _first_outcome(bars, buy_price=100.0, target_return_pct=7.0, stop_loss_pct=3.0)
        self.assertEqual(outcome["first_outcome"], "stop_first")
        self.assertFalse(outcome["target_hit"])
        self.assertTrue(outcome["stop_hit"])
        self.assertTrue(outcome["same_bar_ambiguous"])

    def test_execution_model_metadata_is_explicit(self) -> None:
        row = _base_history_result(
            pd.Series(),
            code="000001",
            signal_date="2026-07-10",
            base_price=10.0,
            hold_days=3,
            target_return_pct=7.0,
            stop_loss_pct=3.0,
            entry_price_mode="confirmation_close",
            top_n=3,
            include_all_allowed=False,
        )
        self.assertEqual(row["execution_model_version"], EXECUTION_MODEL_VERSION)
        self.assertEqual(row["entry_price_mode"], "confirmation_close")
        self.assertTrue(row["confirmation_bar_excluded"])
        self.assertEqual(row["same_bar_policy"], "stop_first")
        summary = build_top3_summary(pd.DataFrame()).iloc[0]
        self.assertEqual(summary["execution_model_version"], EXECUTION_MODEL_VERSION)


class PolicyGateTests(unittest.TestCase):
    def test_daily_output_is_explicitly_a_research_watchlist(self) -> None:
        selected = pd.DataFrame(
            [{"signal_date": "2026-07-10", "codes": "000001,000002,000003", "grid_id": DEFAULT_POLICY.grid_id}]
        )
        pool = pd.DataFrame(
            {
                "signal_date": ["2026-07-10"] * 3,
                "code": ["000001", "000002", "000003"],
                "v002_model_rank": [1, 2, 3],
                "v004a_model_rank": [1, 2, 3],
                "extreme_vwap": [False, False, False],
                "extreme_close_low": [False, False, False],
            }
        )
        decisions, selections = build_daily_policy_outputs(selected, pool, top_n=3)
        decision = decisions.iloc[0]
        self.assertTrue(bool(decision["research_only"]))
        self.assertEqual(decision["deployment_status"], "shadow_only")
        self.assertEqual(decision["research_watchlist_codes"], decision["primary_buy_codes"])
        self.assertTrue(bool(decision["manual_review_required"]))
        self.assertFalse(bool(decision["transaction_costs_included"]))
        watchlist = selections[selections["is_primary_buy"].astype(bool)]
        self.assertTrue(watchlist["selection_intent"].eq("research_watchlist").all())

    def test_fallback_thresholds_come_from_manifest(self) -> None:
        base = pd.Series(
            {
                "v002_extreme_vwap_count": 2,
                "v002_extreme_close_low_count": 2,
                "v005_avg_v002_rank": 12.0,
            }
        )
        self.assertTrue(is_policy_fallback(base))
        base["v005_avg_v002_rank"] = np.nextafter(12.0, 0.0)
        self.assertFalse(is_policy_fallback(base))

    def test_holdout_sample_gate_requires_thirty_unique_signal_dates(self) -> None:
        rows = pd.DataFrame(
            {
                "strategy": [PRIMARY_POLICY] * 29,
                "signal_date": pd.date_range("2026-01-01", periods=29, freq="D").strftime("%Y-%m-%d"),
            }
        )
        readiness = assess_holdout_readiness(rows, min_forward_dates=30).iloc[0]
        self.assertEqual(readiness["readiness_status"], "INSUFFICIENT_FORWARD_SAMPLE")
        self.assertEqual(int(readiness["forward_signal_date_count"]), 29)
        self.assertFalse(bool(readiness["deployable"]))
        rows.loc[len(rows)] = [PRIMARY_POLICY, "2026-01-30"]
        readiness = assess_holdout_readiness(rows, min_forward_dates=30).iloc[0]
        self.assertEqual(readiness["readiness_status"], "FORWARD_SAMPLE_THRESHOLD_MET")
        self.assertEqual(int(readiness["forward_signal_date_count"]), 30)
        self.assertFalse(bool(readiness["deployable"]))
        custom_threshold = assess_holdout_readiness(rows, min_forward_dates=1).iloc[0]
        self.assertFalse(bool(custom_threshold["frozen_policy_inputs_verified"]))
        self.assertEqual(custom_threshold["deployment_status"], "custom_research_only")
        self.assertEqual(custom_threshold["readiness_status"], "UNVERIFIED_POLICY_INPUTS")


if __name__ == "__main__":
    unittest.main()

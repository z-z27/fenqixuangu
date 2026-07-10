from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.daily_ranking import apply_daily_research_ranking
from src.policy_config import get_default_policy, normalized_sha256
from src.v005_daily_selector import (
    build_runtime_policy_meta,
    require_single_signal_date,
    run_v005_daily_selector,
    signals_to_frame,
)
from src.v005_fixed_grid_holdout import run_fixed_grid_holdout


FIXTURES = Path(__file__).parent / "fixtures"
SIGNALS_FIXTURE = FIXTURES / "v005_daily_signals_single_date.csv"
CUSTOM_RANKING_MODEL = FIXTURES / "ranking_model_custom_reverse_v002.json"
FROZEN_RANKING_MODEL_COPY = FIXTURES / "ranking_model_v002_core_momentum_support.json"
FROZEN_COEFFICIENT_COPY = FIXTURES / "v004a_coefficients_2026-06-26.csv"
EXPECTED_FINAL_CODES = "000002,000004,000003"
EXPECTED_V002_CODES = "000001,000002,000003"
EXPECTED_CUSTOM_V002_CODES = "000008,000007,000006"


class V005FixtureRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = get_default_policy()
        cls.signals = pd.read_csv(SIGNALS_FIXTURE, dtype={"code": str})
        cls.temp_dir = TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.default_result = run_v005_daily_selector(cls.signals, output_dir=root / "frozen")
        cls.custom_result = run_v005_daily_selector(
            cls.signals,
            output_dir=root / "custom",
            ranking_model_file=CUSTOM_RANKING_MODEL,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_frozen_artifact_fixture_copies_match_canonical_files(self) -> None:
        self.assertEqual(normalized_sha256(FROZEN_RANKING_MODEL_COPY), self.policy.ranking_model_sha256)
        self.assertEqual(normalized_sha256(FROZEN_COEFFICIENT_COPY), self.policy.coefficients_sha256)

    def test_frozen_daily_pipeline_regression(self) -> None:
        decisions, selections, combos, scored, _ = self.default_result
        decision = decisions.iloc[0]

        self.assertEqual(require_single_signal_date(signals_to_frame(self.signals)), "2026-07-10")
        self.assertEqual(decision["primary_buy_codes"], EXPECTED_FINAL_CODES)
        self.assertEqual(decision["research_watchlist_codes"], EXPECTED_FINAL_CODES)
        self.assertEqual(decision["v005_baseline_codes"], EXPECTED_FINAL_CODES)
        self.assertEqual(decision["v002_codes"], EXPECTED_V002_CODES)
        self.assertEqual(decision["v004a_codes"], "000001,000002,000004")
        self.assertFalse(bool(decision["fallback_triggered"]))
        self.assertEqual(int(decision["selected_grid_id"]), 4)
        self.assertEqual(int(decision["candidate_pool_count"]), 8)
        self.assertTrue(bool(decision["research_only"]))
        self.assertTrue(bool(decision["matches_frozen_manifest"]))
        self.assertTrue(bool(decision["frozen_policy_inputs_verified"]))
        self.assertEqual(decision["deployment_status"], "shadow_only")
        self.assertEqual(combos.iloc[0]["codes"], EXPECTED_FINAL_CODES)
        self.assertAlmostEqual(float(combos.iloc[0]["combo_score"]), 0.2758333333333334)
        self.assertTrue(combos["avg_high_return"].isna().all())
        self.assertTrue(combos["avg_realized_return"].isna().all())

        primary = selections[selections["is_primary_buy"].astype(bool)].sort_values("buy_priority")
        self.assertEqual(primary["code"].tolist(), EXPECTED_FINAL_CODES.split(","))
        self.assertTrue(primary["selection_intent"].eq("research_watchlist").all())
        self.assertFalse(scored["target7_d2open_d3high"].astype(bool).any())
        for column in ("d2open_d3high_return_pct", "d2open_d3close_return_pct", "realized_return_pct"):
            self.assertTrue(scored[column].isna().all(), column)

    def test_custom_ranking_model_changes_internal_v002_rows(self) -> None:
        default_decisions, _, _, default_scored, _ = self.default_result
        custom_decisions, _, _, custom_scored, _ = self.custom_result
        custom = custom_decisions.iloc[0]

        self.assertEqual(default_decisions.iloc[0]["v002_codes"], EXPECTED_V002_CODES)
        self.assertEqual(custom["v002_codes"], EXPECTED_CUSTOM_V002_CODES)
        self.assertFalse(bool(custom["matches_frozen_manifest"]))
        self.assertFalse(bool(custom["frozen_policy_inputs_verified"]))
        self.assertEqual(custom["deployment_status"], "custom_research_only")
        self.assertEqual(custom["v002_source_model_id"], "fixture_custom_reverse_v002")

        self.assertTrue(default_scored["model_id"].eq(self.policy.ranking_model_id).any())
        self.assertFalse(custom_scored["model_id"].eq(self.policy.ranking_model_id).any())
        custom_v002 = custom_scored[custom_scored["model_id"].eq("fixture_custom_reverse_v002")]
        self.assertEqual(custom_v002.sort_values("model_rank").head(3)["code"].tolist(), EXPECTED_CUSTOM_V002_CODES.split(","))
        ranked_signals, ranking_meta = apply_daily_research_ranking(self.signals, CUSTOM_RANKING_MODEL)
        self.assertEqual(ranking_meta["model_id"], "fixture_custom_reverse_v002")
        self.assertEqual(
            ranked_signals.sort_values("daily_rank").head(3)["code"].tolist(),
            EXPECTED_CUSTOM_V002_CODES.split(","),
        )

    def test_non_frozen_min_forward_dates_is_custom_research(self) -> None:
        meta = build_runtime_policy_meta(min_forward_dates=1)
        self.assertFalse(meta["frozen_policy_inputs_verified"])
        self.assertFalse(meta["matches_frozen_manifest"])
        self.assertEqual(meta["deployment_status"], "custom_research_only")
        self.assertIn("min_forward_dates", meta["failed_frozen_policy_checks"])

    def test_non_seven_target_is_rejected_before_input_access(self) -> None:
        with TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(
                RuntimeError,
                r"^Frozen v005 policy requires target_return_pct=7\.0 because the locked target is target7_d2open_d3high\.$",
            ):
                run_fixed_grid_holdout(
                    samples_file=Path(output_dir) / "does-not-exist.csv",
                    output_dir=Path(output_dir) / "out",
                    target_return_pct=5.0,
                )

    def test_single_signal_date_is_accepted(self) -> None:
        self.assertEqual(require_single_signal_date(signals_to_frame(self.signals)), "2026-07-10")

    def test_multiple_signal_dates_are_rejected(self) -> None:
        frame = self.signals.copy()
        second_date = frame.iloc[[0]].copy()
        second_date["trade_date"] = "2026-07-11"
        frame = pd.concat([frame, second_date], ignore_index=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"v005 daily selector requires exactly one signal_date; found \['2026-07-10', '2026-07-11'\]",
        ):
            require_single_signal_date(signals_to_frame(frame))


if __name__ == "__main__":
    unittest.main()

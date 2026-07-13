from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.daily_ranking import apply_daily_research_ranking
from src.history_samples import (
    HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS,
    _write_history_universe_outputs,
)
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
HOLDOUT_FIXTURE = FIXTURES / "v005_holdout_samples_single_date.csv"
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
        cls.holdout_dir = root / "holdout"
        cls.holdout_result = run_fixed_grid_holdout(
            samples_file=HOLDOUT_FIXTURE,
            output_dir=cls.holdout_dir,
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

    def test_holdout_writes_complete_universe_funnel(self) -> None:
        _, daily, _, _, _ = self.holdout_result
        membership_path = self.holdout_dir / "v005_fixed_grid_holdout_universe_membership.csv"
        audit_path = self.holdout_dir / "v005_fixed_grid_holdout_universe_audit.csv"
        manifest_path = self.holdout_dir / "v005_fixed_grid_holdout_universe_manifest.json"
        self.assertTrue(membership_path.is_file())
        self.assertTrue(audit_path.is_file())
        self.assertTrue(manifest_path.is_file())
        audit = pd.read_csv(audit_path).iloc[0]
        self.assertEqual(int(audit["scorable_count"]), 8)
        self.assertEqual(int(audit["v004a_topk_count"]), 8)
        self.assertEqual(int(audit["final_top3_count"]), 3)
        self.assertEqual(audit["final_top3_codes"], EXPECTED_FINAL_CODES)
        self.assertEqual(audit["snapshot_status"], "LEGACY_UNVERIFIED")
        run_meta = pd.read_csv(self.holdout_dir / "v005_fixed_grid_holdout_run_meta.csv").iloc[0]
        self.assertFalse(bool(run_meta["cache_snapshot_complete"]))
        self.assertFalse(bool(run_meta["candidate_universe_snapshot_verified"]))
        self.assertEqual(run_meta["universe_audit_status"], "LEGACY_UNVERIFIED")
        policy = daily[daily["strategy"].eq("policy_v005_v002_regime_fallback")].iloc[0]
        self.assertEqual(policy["selected_codes"], EXPECTED_FINAL_CODES)

    def test_shuffled_holdout_input_preserves_ranks_pool_selection_and_hashes(self) -> None:
        raw = pd.read_csv(HOLDOUT_FIXTURE, dtype={"code": str})
        shuffled = raw.sample(frac=1.0, random_state=27).reset_index(drop=True)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            shuffled_path = root / "shuffled_samples.csv"
            shuffled.to_csv(shuffled_path, index=False, encoding="utf-8-sig")
            shuffled_dir = root / "holdout"
            run_fixed_grid_holdout(samples_file=shuffled_path, output_dir=shuffled_dir)

            baseline_scored = pd.read_csv(
                self.holdout_dir / "v005_fixed_grid_holdout_scored_candidates.csv",
                dtype={"code": str},
            )
            shuffled_scored = pd.read_csv(
                shuffled_dir / "v005_fixed_grid_holdout_scored_candidates.csv",
                dtype={"code": str},
            )
            keys = ["model_id", "evaluation_scope", "l2", "positive_weight", "signal_date", "code"]
            columns = [*keys, "model_score", "model_rank"]
            left = baseline_scored[columns].sort_values(keys, na_position="last").reset_index(drop=True)
            right = shuffled_scored[columns].sort_values(keys, na_position="last").reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right)

            baseline_combo = pd.read_csv(self.holdout_dir / "v005_fixed_grid_selected_combos.csv")
            shuffled_combo = pd.read_csv(shuffled_dir / "v005_fixed_grid_selected_combos.csv")
            pd.testing.assert_frame_equal(baseline_combo, shuffled_combo)

            baseline_membership = pd.read_csv(
                self.holdout_dir / "v005_fixed_grid_holdout_universe_membership.csv",
                dtype={"code": str},
            )
            shuffled_membership = pd.read_csv(
                shuffled_dir / "v005_fixed_grid_holdout_universe_membership.csv",
                dtype={"code": str},
            )
            pd.testing.assert_frame_equal(baseline_membership, shuffled_membership)

            hash_columns = [column for column in pd.read_csv(
                self.holdout_dir / "v005_fixed_grid_holdout_universe_audit.csv"
            ).columns if column.endswith("sha256")]
            baseline_audit = pd.read_csv(self.holdout_dir / "v005_fixed_grid_holdout_universe_audit.csv")
            shuffled_audit = pd.read_csv(shuffled_dir / "v005_fixed_grid_holdout_universe_audit.csv")
            pd.testing.assert_frame_equal(baseline_audit[hash_columns], shuffled_audit[hash_columns])

            shuffled_scored_path = root / "shuffled_scored.csv"
            baseline_scored.sample(frac=1.0, random_state=72).to_csv(
                shuffled_scored_path,
                index=False,
                encoding="utf-8-sig",
            )
            pre_scored_dir = root / "pre_scored_holdout"
            run_fixed_grid_holdout(scored_file=shuffled_scored_path, output_dir=pre_scored_dir)
            pre_scored_combo = pd.read_csv(pre_scored_dir / "v005_fixed_grid_selected_combos.csv")
            pd.testing.assert_frame_equal(baseline_combo, pre_scored_combo)
            pre_scored_membership = pd.read_csv(
                pre_scored_dir / "v005_fixed_grid_holdout_universe_membership.csv",
                dtype={"code": str},
            )
            pd.testing.assert_frame_equal(baseline_membership, pre_scored_membership)
            pre_scored_audit = pd.read_csv(pre_scored_dir / "v005_fixed_grid_holdout_universe_audit.csv")
            comparable_hash_columns = [
                column for column in hash_columns if column != "eligible_code_set_sha256"
            ]
            pd.testing.assert_frame_equal(
                baseline_audit[comparable_hash_columns],
                pre_scored_audit[comparable_hash_columns],
            )
            self.assertTrue(pd.isna(pre_scored_audit.iloc[0]["eligible_count"]))
            self.assertFalse(bool(pre_scored_audit.iloc[0]["eligible_count_available"]))
            self.assertEqual(
                pre_scored_audit.iloc[0]["eligible_source"],
                "unavailable_pre_scored_only",
            )

    def test_holdout_rejects_missing_expected_v004a_or_v002_scored_rows(self) -> None:
        baseline_scored = pd.read_csv(
            self.holdout_dir / "v005_fixed_grid_holdout_scored_candidates.csv",
            dtype={"code": str},
        )
        cases = (
            ("logistic_v004a_weighted", "configured v004a scored rows"),
            (self.policy.ranking_model_id, "configured v002 scored rows"),
        )
        for model_id, expected_label in cases:
            with self.subTest(model_id=model_id), TemporaryDirectory() as temp:
                root = Path(temp)
                remove_mask = baseline_scored["model_id"].astype(str).eq(model_id) & baseline_scored[
                    "code"
                ].astype(str).eq("000008")
                self.assertEqual(int(remove_mask.sum()), 1)
                changed = baseline_scored.loc[~remove_mask].copy()
                scored_path = root / "missing_scored_row.csv"
                changed.to_csv(scored_path, index=False, encoding="utf-8-sig")
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"{expected_label} keys do not match expected scorable samples.*missing_count=1",
                ):
                    run_fixed_grid_holdout(
                        samples_file=HOLDOUT_FIXTURE,
                        scored_file=scored_path,
                        output_dir=root / "holdout",
                    )

    def test_scored_only_holdout_does_not_claim_eligible_count(self) -> None:
        scored_path = self.holdout_dir / "v005_fixed_grid_holdout_scored_candidates.csv"
        with TemporaryDirectory() as temp:
            output_dir = Path(temp) / "holdout"
            _, daily, _, _, _ = run_fixed_grid_holdout(
                scored_file=scored_path,
                output_dir=output_dir,
            )
            audit = pd.read_csv(output_dir / "v005_fixed_grid_holdout_universe_audit.csv").iloc[0]
            self.assertTrue(pd.isna(audit["eligible_count"]))
            self.assertFalse(bool(audit["eligible_count_available"]))
            self.assertEqual(audit["eligible_source"], "unavailable_pre_scored_only")
            policy = daily[daily["strategy"].eq("policy_v005_v002_regime_fallback")].iloc[0]
            self.assertEqual(policy["selected_codes"], EXPECTED_FINAL_CODES)

    def test_legacy_nonidentical_non_scorable_duplicates_are_deterministic(self) -> None:
        raw = pd.read_csv(HOLDOUT_FIXTURE, dtype={"code": str})
        noneligible_rows = []
        for reason in ("Reason C", "Reason A", "Reason B"):
            row = raw.iloc[0].copy()
            row["code"] = "000009"
            row["name"] = "Legacy Noneligible"
            row["eligible_for_trade"] = False
            row["allowed_bool"] = False
            row["reasons"] = reason
            noneligible_rows.append(row)
        missing_return_rows = []
        for score, reason in ((51.0, "Missing B"), (49.0, "Missing A")):
            row = raw.iloc[0].copy()
            row["code"] = "000010"
            row["name"] = "Legacy Missing Return"
            row["eligible_for_trade"] = True
            row["allowed_bool"] = True
            row["d2open_d3high_return_pct"] = None
            row["d2open_d3close_return_pct"] = None
            row["target7_d2open_d3high"] = False
            row["total_score"] = score
            row["reasons"] = reason
            missing_return_rows.append(row)
        legacy = pd.concat(
            [raw, pd.DataFrame([*noneligible_rows, *missing_return_rows])],
            ignore_index=True,
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = []
            for label, samples in (
                ("original", legacy),
                ("shuffled", legacy.sample(frac=1.0, random_state=73)),
            ):
                samples_path = root / f"legacy_samples_{label}.csv"
                samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
                output_dir = root / f"holdout_{label}"
                _, daily, _, _, _ = run_fixed_grid_holdout(
                    samples_file=samples_path,
                    output_dir=output_dir,
                )
                outputs.append((output_dir, daily))

            baseline_dir, baseline_daily = outputs[0]
            shuffled_dir, shuffled_daily = outputs[1]
            run_meta = pd.read_csv(
                baseline_dir / "v005_fixed_grid_holdout_run_meta.csv"
            ).iloc[0]
            self.assertEqual(
                run_meta["universe_audit_status"],
                "LEGACY_UNVERIFIED_DUPLICATES_EXCLUDED",
            )
            self.assertEqual(int(run_meta["legacy_duplicate_key_count"]), 2)
            self.assertEqual(int(run_meta["legacy_duplicate_row_count"]), 5)
            self.assertEqual(run_meta["legacy_duplicate_codes"], "000009,000010")
            self.assertEqual(int(run_meta["legacy_nonidentical_duplicate_key_count"]), 2)
            self.assertEqual(
                run_meta["legacy_duplicate_representative_policy"],
                "min_canonical_history_candidate_row_json_v1",
            )
            details = json.loads(run_meta["legacy_duplicate_details"])
            self.assertEqual([item["row_count"] for item in details], [3, 2])
            self.assertEqual(
                [item["unique_canonical_row_count"] for item in details], [3, 2]
            )
            self.assertEqual(
                [item["intrinsic_exclusion_reasons"] for item in details],
                [
                    ["not_eligible"],
                    ["missing_high_return|missing_close_return"],
                ],
            )
            self.assertTrue(
                all(item["representative_rows_sha256"] for item in details)
            )
            membership = pd.read_csv(
                baseline_dir / "v005_fixed_grid_holdout_universe_membership.csv",
                dtype={"code": str},
            )
            self.assertEqual(len(membership), 8)
            self.assertFalse(membership["code"].isin(["000009", "000010"]).any())
            audit = pd.read_csv(
                baseline_dir / "v005_fixed_grid_holdout_universe_audit.csv"
            ).iloc[0]
            self.assertEqual(int(audit["eligible_count"]), 9)
            scored = pd.read_csv(
                baseline_dir / "v005_fixed_grid_holdout_scored_candidates.csv",
                dtype={"code": str},
            )
            self.assertFalse(scored["code"].isin(["000009", "000010"]).any())

            shuffled_meta = pd.read_csv(
                shuffled_dir / "v005_fixed_grid_holdout_run_meta.csv"
            ).iloc[0]
            for column in (
                "legacy_duplicate_key_count",
                "legacy_duplicate_row_count",
                "legacy_duplicate_codes",
                "legacy_nonidentical_duplicate_key_count",
                "legacy_duplicate_representative_policy",
                "legacy_duplicate_details",
            ):
                self.assertEqual(run_meta[column], shuffled_meta[column], column)
            shuffled_audit = pd.read_csv(
                shuffled_dir / "v005_fixed_grid_holdout_universe_audit.csv"
            ).iloc[0]
            for column in (
                "eligible_count",
                "legacy_duplicate_key_count",
                "legacy_duplicate_row_count",
                "legacy_duplicate_codes",
                "legacy_nonidentical_duplicate_key_count",
                "legacy_duplicate_representative_policy",
                "legacy_duplicate_details",
            ):
                self.assertEqual(audit[column], shuffled_audit[column], column)
            baseline_manifest = json.loads(
                (baseline_dir / "v005_fixed_grid_holdout_universe_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            shuffled_manifest = json.loads(
                (shuffled_dir / "v005_fixed_grid_holdout_universe_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                baseline_manifest["membership_canonical_rows_sha256"],
                shuffled_manifest["membership_canonical_rows_sha256"],
            )
            for daily in (baseline_daily, shuffled_daily):
                policy = daily[
                    daily["strategy"].eq("policy_v005_v002_regime_fallback")
                ].iloc[0]
                self.assertEqual(policy["selected_codes"], EXPECTED_FINAL_CODES)
                self.assertFalse(bool(policy["gate_triggered"]))

    def test_legacy_intrinsically_scorable_duplicates_are_rejected(self) -> None:
        raw = pd.read_csv(HOLDOUT_FIXTURE, dtype={"code": str})
        valid = raw.iloc[0].copy()
        valid["code"] = "000009"
        different = valid.copy()
        different["name"] = "Different but still scorable"
        with TemporaryDirectory() as temp:
            root = Path(temp)
            samples = pd.concat(
                [raw, pd.DataFrame([valid, different])], ignore_index=True
            )
            path = root / "legacy_invalid.csv"
            samples.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(RuntimeError, "valid scorable rows"):
                run_fixed_grid_holdout(samples_file=path, output_dir=root / "out")

    def test_manifest_backed_samples_reject_even_non_scorable_duplicates(self) -> None:
        raw = pd.read_csv(HOLDOUT_FIXTURE, dtype={"code": str})
        duplicate = raw.iloc[0].copy()
        duplicate["code"] = "000009"
        duplicate["eligible_for_trade"] = False
        duplicate["allowed_bool"] = False
        with TemporaryDirectory() as temp:
            root = Path(temp)
            suffix = "2026-07-10_2026-07-10"
            samples_path = root / f"history_candidates_{suffix}.csv"
            pd.concat(
                [raw, pd.DataFrame([duplicate, duplicate])], ignore_index=True
            ).to_csv(samples_path, index=False, encoding="utf-8-sig")
            (root / f"history_universe_manifest_{suffix}.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "manifest-backed holdout samples contains duplicate keys",
            ):
                run_fixed_grid_holdout(
                    samples_file=samples_path,
                    output_dir=root / "holdout",
                )

    def test_scored_only_requires_exact_v002_key_coverage(self) -> None:
        baseline = pd.read_csv(
            self.holdout_dir / "v005_fixed_grid_holdout_scored_candidates.csv",
            dtype={"code": str},
        )
        v002_mask = baseline["model_id"].astype(str).eq(self.policy.ranking_model_id)
        missing = baseline.loc[
            ~(v002_mask & baseline["code"].astype(str).eq("000008"))
        ].copy()
        extra_row = baseline.loc[v002_mask].iloc[0].copy()
        extra_row["code"] = "000009"
        extra = pd.concat([baseline, pd.DataFrame([extra_row])], ignore_index=True)
        for label, changed, expected in (
            ("missing", missing, "missing_count=1"),
            ("extra", extra, "extra_count=1"),
        ):
            with self.subTest(label=label), TemporaryDirectory() as temp:
                root = Path(temp)
                scored_path = root / f"{label}.csv"
                changed.to_csv(scored_path, index=False, encoding="utf-8-sig")
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"configured v002 scored rows keys do not match expected scorable samples.*{expected}",
                ):
                    run_fixed_grid_holdout(
                        scored_file=scored_path,
                        output_dir=root / "holdout",
                    )

    def test_holdout_verifies_history_universe_manifest_and_rejects_changed_samples(self) -> None:
        raw = pd.read_csv(HOLDOUT_FIXTURE, dtype={"code": str})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            history_dir = root / "history"
            history_dir.mkdir()
            suffix = "2026-07-10_2026-07-10"
            samples_path = history_dir / f"history_candidates_{suffix}.csv"
            raw.to_csv(samples_path, index=False, encoding="utf-8-sig")
            membership = pd.DataFrame(
                [
                    {
                        "requested_signal_date": "2026-07-10",
                        "actual_signal_date": "2026-07-10",
                        "stage": "scorable_pool",
                        "member_key": f"2026-07-10|{code}",
                        "source_trade_date": "",
                        "code": code,
                        "name": name,
                        "d0_date": "",
                        "included_bool": True,
                        "exclusion_reason": "",
                    }
                    for code, name in raw[["code", "name"]].itertuples(index=False, name=None)
                ],
                columns=HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS,
            )
            audit = pd.DataFrame(
                [
                    {
                        "requested_signal_date": "2026-07-10",
                        "actual_signal_date": "2026-07-10",
                        "generation_status": "generated",
                        "snapshot_status": "CREATED_CANONICAL",
                        "candidate_row_count": 8,
                        "lookback_unresolved_dates": "",
                    }
                ]
            )
            _write_history_universe_outputs(
                output_dir=history_dir,
                start_date="2026-07-10",
                end_date="2026-07-10",
                lookback_days=5,
                signal_days=10,
                eval_days=10,
                hold_days=10,
                target_return_pct=7.0,
                universe_snapshot_mode="create-or-verify",
                candidates=raw,
                universe_audit=audit,
                universe_membership=membership,
            )
            verified_dir = root / "verified"
            run_fixed_grid_holdout(samples_file=samples_path, output_dir=verified_dir)
            run_meta = pd.read_csv(verified_dir / "v005_fixed_grid_holdout_run_meta.csv").iloc[0]
            self.assertTrue(bool(run_meta["candidate_universe_snapshot_verified"]))
            self.assertTrue(bool(run_meta["candidate_universe_snapshot_complete"]))
            self.assertFalse(bool(run_meta["cache_snapshot_complete"]))
            self.assertEqual(run_meta["universe_audit_status"], "VERIFIED")

            manifest_path = history_dir / f"history_universe_manifest_{suffix}.json"
            original_manifest = manifest_path.read_bytes()
            manifest = json.loads(original_manifest.decode("utf-8"))
            manifest["dates"][0]["snapshot_status"] = "MISSING_EXACT_SIGNAL_DATE"
            manifest["candidate_universe_snapshot_complete"] = False
            manifest["candidate_universe_snapshot_verified"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "generated_dates does not match successful snapshots",
            ):
                run_fixed_grid_holdout(samples_file=samples_path, output_dir=root / "missing-date")
            manifest_path.write_bytes(original_manifest)

            changed = raw.copy()
            changed.loc[0, "total_score"] = float(changed.loc[0, "total_score"]) + 1.0
            changed.to_csv(samples_path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(RuntimeError, "does not match samples input"):
                run_fixed_grid_holdout(samples_file=samples_path, output_dir=root / "changed")


if __name__ == "__main__":
    unittest.main()

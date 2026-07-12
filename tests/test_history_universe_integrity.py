from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

from src import cli
from src.history_samples import (
    HISTORY_CANDIDATE_COLUMNS,
    LookbackResolution,
    _assess_history_snapshot_completeness,
    _collect_limitups_for_history_sample,
    _publish_history_attempt,
    _raw_stage_snapshot,
    _standardize_raw_source_pool,
    run_history_sample_generation,
)
from src.universe_audit import stage_identity
from src.v005_fixed_grid_holdout import _load_history_universe_context


class _LimitUpCache:
    def __init__(self, values: dict[str, pd.DataFrame] | None = None) -> None:
        self.values = values or {}

    def read(self, trade_date: str) -> pd.DataFrame | None:
        value = self.values.get(trade_date)
        return None if value is None else value.copy()


class _HistoryService:
    def __init__(
        self,
        collected: dict[str, pd.DataFrame | Exception],
        cached: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.collected = collected
        self.limit_up_cache = _LimitUpCache(cached)
        self.collect_calls: list[str] = []

    def collect_limit_ups(self, trade_date: str, **_: object) -> pd.DataFrame:
        self.collect_calls.append(trade_date)
        value = self.collected.get(trade_date)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError(f"unresolved {trade_date}")
        return value.copy()


def _limitup_frame(trade_date: str, code: str = "000001") -> pd.DataFrame:
    return pd.DataFrame(
        [{"trade_date": trade_date, "code": code, "name": f"Name {code}"}]
    )


class HistorySnapshotCompletenessTests(unittest.TestCase):
    def test_missing_exact_signal_date_is_incomplete(self) -> None:
        audit = pd.DataFrame(
            [
                {
                    "requested_signal_date": "2026-07-10",
                    "generation_status": "failed",
                    "snapshot_status": "MISSING_EXACT_SIGNAL_DATE",
                }
            ]
        )
        result = _assess_history_snapshot_completeness(
            candidates=pd.DataFrame(columns=["signal_date"]),
            universe_audit=audit,
            universe_snapshot_mode="create-or-verify",
            requested_dates=["2026-07-10"],
        )
        self.assertFalse(result["snapshot_complete"])
        self.assertEqual(result["audit_status"], "INCOMPLETE")

    def test_proven_non_trading_date_can_preserve_completeness(self) -> None:
        audit = pd.DataFrame(
            [
                {
                    "requested_signal_date": "2026-07-10",
                    "generation_status": "non_trading",
                    "snapshot_status": "PROVEN_NON_TRADING_DATE",
                }
            ]
        )
        result = _assess_history_snapshot_completeness(
            candidates=pd.DataFrame(columns=["signal_date"]),
            universe_audit=audit,
            universe_snapshot_mode="create-or-verify",
            requested_dates=["2026-07-10"],
        )
        self.assertTrue(result["snapshot_complete"])
        self.assertEqual(result["proven_non_trading_dates"], ["2026-07-10"])

    def test_off_mode_failure_is_explicitly_partial_unverified(self) -> None:
        audit = pd.DataFrame(
            [
                {
                    "requested_signal_date": "2026-07-10",
                    "generation_status": "failed",
                    "snapshot_status": "GENERATION_FAILED",
                }
            ]
        )
        result = _assess_history_snapshot_completeness(
            candidates=pd.DataFrame(columns=["signal_date"]),
            universe_audit=audit,
            universe_snapshot_mode="off",
            requested_dates=["2026-07-10"],
        )
        self.assertFalse(result["snapshot_complete"])
        self.assertEqual(result["audit_status"], "PARTIAL_UNVERIFIED")


class HistoryGenerationBoundaryIntegrationTests(unittest.TestCase):
    def test_proven_non_trading_date_completes_without_canonical(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            service = _HistoryService(
                {"2026-07-10": RuntimeError("date_seen=0")}
            )
            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=service),
                mock.patch(
                    "src.history_samples._cached_daily_proves_non_trading",
                    return_value=False,
                ),
            ):
                candidates, _, run_log, _, *_ = run_history_sample_generation(
                    start_date="2026-07-10",
                    end_date="2026-07-10",
                    lookback_days=1,
                    universe_snapshot_mode="create-or-verify",
                )

            self.assertTrue(candidates.empty)
            self.assertEqual(run_log.iloc[0]["status"], "skipped_non_trading")
            stable = config.reports_dir / "history_samples" / "2026-07-10_2026-07-10"
            audit = pd.read_csv(
                stable / "history_universe_audit_2026-07-10_2026-07-10.csv"
            ).iloc[0]
            self.assertEqual(audit["generation_status"], "non_trading")
            self.assertEqual(audit["snapshot_status"], "PROVEN_NON_TRADING_DATE")
            manifest = json.loads(
                (stable / "history_universe_manifest_2026-07-10_2026-07-10.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["candidate_universe_snapshot_complete"])
            self.assertEqual(manifest["requested_dates"], ["2026-07-10"])
            self.assertEqual(manifest["generated_dates"], [])
            canonical = config.snapshot_dir / "history_universe" / "2026-07-10" / "canonical"
            self.assertFalse(canonical.exists())

    def test_existing_canonical_cannot_be_reclassified_as_non_trading(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            canonical = config.snapshot_dir / "history_universe" / "2026-07-10" / "canonical"
            canonical.mkdir(parents=True)
            (canonical / "manifest.json").write_text("{}", encoding="utf-8")
            service = _HistoryService(
                {"2026-07-10": RuntimeError("date_seen=0")}
            )
            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=service),
                mock.patch(
                    "src.history_samples._cached_daily_proves_non_trading",
                    return_value=False,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "SNAPSHOT_STATUS_CONFLICT"):
                    run_history_sample_generation(
                        start_date="2026-07-10",
                        end_date="2026-07-10",
                        lookback_days=1,
                        universe_snapshot_mode="create-or-verify",
                    )
            stable = config.reports_dir / "history_samples" / "2026-07-10_2026-07-10"
            attempt = next((stable / "attempts").iterdir())
            audit = pd.read_csv(
                attempt / "history_universe_audit_2026-07-10_2026-07-10.csv"
            ).iloc[0]
            self.assertEqual(audit["snapshot_status"], "SNAPSHOT_STATUS_CONFLICT")

    def test_zero_candidate_trading_date_is_complete_and_manifest_verifies(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            service = _HistoryService({"2026-07-10": _limitup_frame("2026-07-10")})
            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=service),
                mock.patch(
                    "src.history_samples._cached_daily_proves_non_trading",
                    return_value=False,
                ),
                mock.patch("src.history_samples.build_signals_for_pool", return_value=([], [])),
            ):
                candidates, _, _, _, candidates_path, *_ = run_history_sample_generation(
                    start_date="2026-07-10",
                    end_date="2026-07-10",
                    lookback_days=1,
                    universe_snapshot_mode="create-or-verify",
                )

            self.assertTrue(candidates.empty)
            persisted = pd.read_csv(candidates_path, dtype={"code": str})
            self.assertTrue(persisted.empty)
            self.assertEqual(list(persisted.columns), HISTORY_CANDIDATE_COLUMNS)
            stable = candidates_path.parent
            audit = pd.read_csv(
                stable / "history_universe_audit_2026-07-10_2026-07-10.csv"
            ).iloc[0]
            self.assertEqual(int(audit["candidate_row_count"]), 0)
            self.assertEqual(audit["generation_status"], "generated")
            manifest_path = stable / "history_universe_manifest_2026-07-10_2026-07-10.json"
            context = _load_history_universe_context(
                samples_path=candidates_path,
                raw_samples=persisted,
                explicit_manifest_path=manifest_path,
            )
            self.assertTrue(context["snapshot_complete"])
            self.assertTrue(context["snapshot_verified"])

    def test_quality_failure_keeps_actionable_attempt_report(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            service = _HistoryService({"2026-07-10": _limitup_frame("2026-07-10")})
            quality_failure = {
                "code": "000001",
                "name": "Failure",
                "trade_date": "2026-07-10",
                "d0_date": "2026-07-10",
                "status": "failed",
                "error_class": "DataQualityError",
                "error": "minute coverage failed",
                "data_quality_reason": "minute_trade_days below minimum",
                "is_data_quality_error": True,
                "warnings": "minute_trade_days below minimum",
            }
            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=service),
                mock.patch(
                    "src.history_samples._cached_daily_proves_non_trading",
                    return_value=False,
                ),
                mock.patch(
                    "src.history_samples.build_signals_for_pool",
                    return_value=([], [quality_failure]),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"failed_count=1.*failed_codes=000001.*DataQualityError",
                ):
                    run_history_sample_generation(
                        start_date="2026-07-10",
                        end_date="2026-07-10",
                        lookback_days=1,
                        universe_snapshot_mode="create-or-verify",
                    )
            stable = config.reports_dir / "history_samples" / "2026-07-10_2026-07-10"
            attempt = next((stable / "attempts").iterdir())
            quality = pd.read_csv(attempt / "data_quality" / "data_quality_2026-07-10.csv")
            self.assertEqual(quality.iloc[0]["code"], 1)
            self.assertEqual(quality.iloc[0]["error_class"], "DataQualityError")
            self.assertEqual(quality.iloc[0]["error"], "minute coverage failed")
            self.assertEqual(
                quality.iloc[0]["data_quality_reason"],
                "minute_trade_days below minimum",
            )
            self.assertTrue(bool(quality.iloc[0]["is_data_quality_error"]))
            run_log = pd.read_csv(
                attempt / "history_generation_log_2026-07-10_2026-07-10.csv",
                dtype={"quality_failed_codes": str},
            ).iloc[0]
            self.assertEqual(int(run_log["quality_failed"]), 1)
            self.assertEqual(run_log["quality_failed_codes"], "000001")


class StrictLookbackResolutionTests(unittest.TestCase):
    def test_strict_cold_cache_resolves_pre_start_window_and_matches_warm_hash(self) -> None:
        collected = {
            "2026-07-02": _limitup_frame("2026-07-02", "000002"),
            "2026-07-03": _limitup_frame("2026-07-03", "000003"),
            "2026-07-06": _limitup_frame("2026-07-06", "000006"),
        }
        cold = _HistoryService(collected)
        with mock.patch(
            "src.history_samples._cached_daily_proves_non_trading", return_value=False
        ):
            cold_result = _collect_limitups_for_history_sample(
                service=cold,
                requested_date="2026-07-06",
                start_date="2026-07-06",
                lookback_days=5,
                force_refresh=False,
                workers=1,
                strict=True,
            )
        self.assertEqual(cold.collect_calls, ["2026-07-06", "2026-07-03", "2026-07-02"])
        self.assertEqual(cold_result.unresolved_dates, ())
        self.assertEqual(
            cold_result.expected_dates,
            (
                "2026-07-02",
                "2026-07-03",
                "2026-07-04",
                "2026-07-05",
                "2026-07-06",
            ),
        )
        self.assertEqual(
            cold_result.data_dates,
            ("2026-07-02", "2026-07-03", "2026-07-06"),
        )
        self.assertEqual(
            cold_result.proven_non_trading_dates,
            ("2026-07-04", "2026-07-05"),
        )

        warm = _HistoryService({}, cached=collected)
        with mock.patch(
            "src.history_samples._cached_daily_proves_non_trading", return_value=False
        ):
            warm_result = _collect_limitups_for_history_sample(
                service=warm,
                requested_date="2026-07-06",
                start_date="2026-07-06",
                lookback_days=5,
                force_refresh=False,
                workers=1,
                strict=True,
            )
        self.assertEqual(warm.collect_calls, [])
        cold_hash = stage_identity(
            _raw_stage_snapshot(_standardize_raw_source_pool(cold_result.frame, "2026-07-06"))
        )["rows_sha256"]
        warm_hash = stage_identity(
            _raw_stage_snapshot(_standardize_raw_source_pool(warm_result.frame, "2026-07-06"))
        )["rows_sha256"]
        self.assertEqual(cold_hash, warm_hash)

    def test_unresolved_pre_start_lookback_hard_fails_without_canonical(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            service = _HistoryService(
                {
                    "2026-07-02": RuntimeError("network unavailable"),
                    "2026-07-03": _limitup_frame("2026-07-03", "000003"),
                    "2026-07-06": _limitup_frame("2026-07-06", "000006"),
                }
            )
            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=service),
                mock.patch(
                    "src.history_samples._cached_daily_proves_non_trading",
                    return_value=False,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "LOOKBACK_UNRESOLVED"):
                    run_history_sample_generation(
                        start_date="2026-07-06",
                        end_date="2026-07-06",
                        lookback_days=5,
                        universe_snapshot_mode="create-or-verify",
                    )
            self.assertIn("2026-07-02", service.collect_calls)
            canonical = config.snapshot_dir / "history_universe" / "2026-07-06" / "canonical"
            self.assertFalse(canonical.exists())
            stable = config.reports_dir / "history_samples" / "2026-07-06_2026-07-06"
            attempt = next((stable / "attempts").iterdir())
            audit = pd.read_csv(
                attempt / "history_universe_audit_2026-07-06_2026-07-06.csv"
            ).iloc[0]
            self.assertEqual(audit["snapshot_status"], "LOOKBACK_UNRESOLVED")
            self.assertIn("2026-07-02", audit["lookback_unresolved_dates"])


class HistoryAttemptIsolationTests(unittest.TestCase):
    def test_publication_error_rolls_back_all_stable_files(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            stable = root / "stable"
            attempt = stable / "attempts" / "attempt-1"
            attempt.mkdir(parents=True)
            suffix = "2026-07-10_2026-07-10"
            names = (
                f"history_candidates_{suffix}.csv",
                f"history_candidates_summary_{suffix}.csv",
                f"history_generation_log_{suffix}.csv",
                f"history_future_fetch_{suffix}.csv",
                f"history_universe_audit_{suffix}.csv",
                f"history_universe_membership_{suffix}.csv",
                f"history_candidates_review_{suffix}.md",
                f"history_universe_manifest_{suffix}.json",
            )
            for name in names:
                (stable / name).write_bytes(f"old-{name}".encode("utf-8"))
                (attempt / name).write_bytes(f"new-{name}".encode("utf-8"))
            before = {name: (stable / name).read_bytes() for name in names}
            original_replace = Path.replace
            injected = False

            def fail_second_commit(path: Path, target: Path) -> Path:
                nonlocal injected
                if (
                    not injected
                    and ".publish-" in path.name
                    and Path(target).name == f"history_candidates_summary_{suffix}.csv"
                ):
                    injected = True
                    raise OSError("injected publication failure")
                return original_replace(path, target)

            with mock.patch("pathlib.Path.replace", autospec=True, side_effect=fail_second_commit):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    _publish_history_attempt(
                        attempt_root=attempt,
                        stable_root=stable,
                        start_date="2026-07-10",
                        end_date="2026-07-10",
                        manifest_source=attempt / f"history_universe_manifest_{suffix}.json",
                    )
            for name, payload in before.items():
                self.assertEqual((stable / name).read_bytes(), payload)
            self.assertEqual(list(stable.glob(".*.publish-*")), [])
            self.assertEqual(list(stable.glob(".*.backup-*")), [])

    def test_failed_rerun_preserves_successful_stable_outputs_byte_for_byte(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            stable = config.reports_dir / "history_samples" / "2026-07-10_2026-07-10"
            stable.mkdir(parents=True)
            names = (
                "history_candidates_2026-07-10_2026-07-10.csv",
                "history_universe_membership_2026-07-10_2026-07-10.csv",
                "history_universe_manifest_2026-07-10_2026-07-10.json",
            )
            expected: dict[str, bytes] = {}
            for index, name in enumerate(names, start=1):
                payload = f"previous-success-{index}\n".encode("utf-8")
                (stable / name).write_bytes(payload)
                expected[name] = payload

            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=object()),
                mock.patch(
                    "src.history_samples._collect_limitups_for_history_sample",
                    side_effect=RuntimeError("injected generation failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected generation failure"):
                    run_history_sample_generation(
                        start_date="2026-07-10",
                        end_date="2026-07-10",
                        universe_snapshot_mode="create-or-verify",
                    )

            for name, payload in expected.items():
                self.assertEqual((stable / name).read_bytes(), payload)
            attempts = sorted((stable / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            attempt = attempts[0]
            self.assertTrue(
                (attempt / "history_generation_log_2026-07-10_2026-07-10.csv").is_file()
            )
            self.assertTrue(
                (attempt / "history_universe_audit_2026-07-10_2026-07-10.csv").is_file()
            )
            self.assertTrue(
                (attempt / "history_universe_membership_2026-07-10_2026-07-10.csv").is_file()
            )

    def test_cli_returns_nonzero_for_generation_failure_in_strict_snapshot_modes(self) -> None:
        for mode in ("create-or-verify", "verify-only"):
            with self.subTest(mode=mode), TemporaryDirectory() as temp:
                root = Path(temp)
                config = SimpleNamespace(
                    reports_dir=root / "reports",
                    snapshot_dir=root / "snapshots",
                )
                with (
                    mock.patch("src.history_samples.get_data_config", return_value=config),
                    mock.patch("src.history_samples.MarketDataService", return_value=object()),
                    mock.patch(
                        "src.history_samples._collect_limitups_for_history_sample",
                        side_effect=RuntimeError("injected generation failure"),
                    ),
                    redirect_stderr(StringIO()),
                ):
                    exit_code = cli.main(
                        [
                            "generate-history-samples",
                            "--start-date",
                            "2026-07-10",
                            "--end-date",
                            "2026-07-10",
                            "--universe-snapshot-mode",
                            mode,
                        ]
                    )
                self.assertEqual(exit_code, 1)
                stable = config.reports_dir / "history_samples" / "2026-07-10_2026-07-10"
                self.assertFalse(
                    (stable / "history_universe_manifest_2026-07-10_2026-07-10.json").exists()
                )


if __name__ == "__main__":
    unittest.main()

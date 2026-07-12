from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

from src import cli
from src.history_samples import (
    _assess_history_snapshot_completeness,
    _publish_history_attempt,
    run_history_sample_generation,
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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

import src.universe_audit as universe_audit
from src.history_samples import _normalise_history_candidate_columns
from src.universe_audit import (
    StageSnapshot,
    UniverseAuditError,
    UniverseSnapshotMismatch,
    canonical_rows_sha256,
    code_set_sha256,
    compare_stage_frames,
    create_or_verify_snapshot,
    key_set_sha256,
    require_nonempty_codes,
    UniverseSnapshotError,
)
from src.v004a import annotate_v004a_input_eligibility, prepare_v004a_samples


ROW_COLUMNS = ("requested_signal_date", "code", "name", "score", "flag_bool")
KEY_COLUMNS = ("requested_signal_date", "code")


def _snapshot(frame: pd.DataFrame) -> StageSnapshot:
    return StageSnapshot(
        stage="signal_pool",
        frame=frame,
        key_columns=KEY_COLUMNS,
        row_columns=ROW_COLUMNS,
    )


def _stage_snapshot(stage: str, frame: pd.DataFrame) -> StageSnapshot:
    return StageSnapshot(
        stage=stage,
        frame=frame,
        key_columns=KEY_COLUMNS,
        row_columns=ROW_COLUMNS,
    )


class StableUniverseHashTests(unittest.TestCase):
    def test_hash_is_independent_of_order_code_format_newlines_and_missing_representation(self) -> None:
        csv_lf = (
            "requested_signal_date,code,name,score,flag_bool\n"
            "2026-07-10,1,A,1.25,true\n"
            "2026-07-10,2,,2.5,false\n"
        )
        csv_crlf = csv_lf.replace("\n", "\r\n")
        left = pd.read_csv(StringIO(csv_lf), dtype={"code": str})
        right = pd.read_csv(StringIO(csv_crlf), dtype={"code": str}).iloc[::-1].reset_index(drop=True)
        right.loc[right["code"].eq("1"), "code"] = "000001"
        right.loc[right["code"].eq("2"), "name"] = None
        right = pd.concat(
            [
                right,
                pd.DataFrame(
                    [{"requested_signal_date": "2026-07-10", "code": None, "name": "ignored", "score": 99, "flag_bool": True}]
                ),
            ],
            ignore_index=True,
        )
        self.assertEqual(code_set_sha256(left), code_set_sha256(right))
        self.assertEqual(key_set_sha256(left, KEY_COLUMNS), key_set_sha256(right, KEY_COLUMNS))
        self.assertEqual(
            canonical_rows_sha256(left, ROW_COLUMNS, KEY_COLUMNS),
            canonical_rows_sha256(right, ROW_COLUMNS, KEY_COLUMNS),
        )

    def test_added_and_removed_members_change_hash_and_diff(self) -> None:
        base = pd.DataFrame(
            [
                {"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True},
                {"requested_signal_date": "2026-07-10", "code": "2", "name": "B", "score": 2.0, "flag_bool": False},
            ]
        )
        changed = pd.DataFrame(
            [
                {"requested_signal_date": "2026-07-10", "code": "2", "name": "B", "score": 2.0, "flag_bool": False},
                {"requested_signal_date": "2026-07-10", "code": "3", "name": "C", "score": 3.0, "flag_bool": True},
            ]
        )
        self.assertNotEqual(code_set_sha256(base), code_set_sha256(changed))
        diff = compare_stage_frames(_snapshot(base), _snapshot(changed))
        self.assertEqual(set(diff["diff_type"]), {"added", "removed"})
        self.assertEqual(set(diff["code"]), {"000001", "000003"})

    def test_row_change_preserves_code_hash_but_changes_rows_hash(self) -> None:
        base = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True}]
        )
        changed = base.copy()
        changed["score"] = 1.5
        self.assertEqual(code_set_sha256(base), code_set_sha256(changed))
        self.assertNotEqual(
            canonical_rows_sha256(base, ROW_COLUMNS, KEY_COLUMNS),
            canonical_rows_sha256(changed, ROW_COLUMNS, KEY_COLUMNS),
        )
        diff = compare_stage_frames(_snapshot(base), _snapshot(changed))
        self.assertEqual(diff["diff_type"].tolist(), ["changed"])


class CanonicalTextEncodingTests(unittest.TestCase):
    def test_real_float_round_trip_uses_identical_text_hash_and_self_verifies(self) -> None:
        value = 9.1357
        legacy_text = format(value, ".17g")
        reparsed = float(pd.read_csv(StringIO(f"score\n{legacy_text}\n")).iloc[0, 0])
        self.assertNotEqual(legacy_text, format(reparsed, ".17g"))
        self.assertEqual(
            universe_audit.canonical_scalar_text(value, "score"),
            universe_audit.canonical_scalar_text(reparsed, "score"),
        )
        base = pd.DataFrame(
            [
                {
                    "requested_signal_date": "2026-07-10",
                    "code": "000001",
                    "name": "A",
                    "score": value,
                    "flag_bool": True,
                }
            ]
        )
        round_tripped = base.copy()
        round_tripped["score"] = reparsed
        self.assertEqual(
            canonical_rows_sha256(base, ROW_COLUMNS, KEY_COLUMNS),
            canonical_rows_sha256(round_tripped, ROW_COLUMNS, KEY_COLUMNS),
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            status, _, manifest = create_or_verify_snapshot(
                "2026-07-10",
                {"signal_pool": _snapshot(base)},
                root / "snapshots",
                root / "run",
            )
            self.assertEqual(status, "CREATED_CANONICAL")
            self.assertEqual(
                manifest["stages"]["signal_pool"]["rows_sha256"],
                canonical_rows_sha256(base, ROW_COLUMNS, KEY_COLUMNS),
            )
            status, _, _ = create_or_verify_snapshot(
                "2026-07-10",
                {"signal_pool": _snapshot(round_tripped)},
                root / "snapshots",
                root / "run",
                mode="verify-only",
            )
            self.assertEqual(status, "VERIFIED_MATCH")

    def test_difference_after_twelfth_significant_digit_is_semantically_equal(self) -> None:
        left = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.2345678901234, "flag_bool": True}]
        )
        right = left.copy()
        right["score"] = 1.23456789012349
        self.assertEqual(
            universe_audit.canonical_scalar_text(left.iloc[0]["score"], "score"),
            universe_audit.canonical_scalar_text(right.iloc[0]["score"], "score"),
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            create_or_verify_snapshot(
                "2026-07-10", {"signal_pool": _snapshot(left)}, root / "snapshots", root / "run"
            )
            status, _, _ = create_or_verify_snapshot(
                "2026-07-10",
                {"signal_pool": _snapshot(right)},
                root / "snapshots",
                root / "run",
                mode="verify-only",
            )
            self.assertEqual(status, "VERIFIED_MATCH")

    def test_difference_within_twelfth_significant_digit_is_a_mismatch(self) -> None:
        left = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.23456789012, "flag_bool": True}]
        )
        right = left.copy()
        right["score"] = 1.23456789013
        self.assertNotEqual(
            universe_audit.canonical_scalar_text(left.iloc[0]["score"], "score"),
            universe_audit.canonical_scalar_text(right.iloc[0]["score"], "score"),
        )
        self.assertNotEqual(
            canonical_rows_sha256(left, ROW_COLUMNS, KEY_COLUMNS),
            canonical_rows_sha256(right, ROW_COLUMNS, KEY_COLUMNS),
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            create_or_verify_snapshot(
                "2026-07-10", {"signal_pool": _snapshot(left)}, root / "snapshots", root / "run"
            )
            with self.assertRaises(UniverseSnapshotMismatch):
                create_or_verify_snapshot(
                    "2026-07-10",
                    {"signal_pool": _snapshot(right)},
                    root / "snapshots",
                    root / "run",
                    mode="verify-only",
                )

    def test_non_float_business_fields_remain_strict(self) -> None:
        columns = (
            "requested_signal_date",
            "code",
            "included_bool",
            "exclusion_reason",
            "signal_type",
            "score",
        )

        def snapshot(frame: pd.DataFrame) -> StageSnapshot:
            return StageSnapshot("signal_pool", frame, KEY_COLUMNS, columns)

        base_row = {
            "requested_signal_date": "2026-07-10",
            "code": "000001",
            "included_bool": True,
            "exclusion_reason": "",
            "signal_type": "D2_LOW_ABSORB",
            "score": 1.0,
        }
        changes = {
            "code": "000002",
            "requested_signal_date": "2026-07-11",
            "included_bool": False,
            "exclusion_reason": "suspended_on_signal_date",
            "signal_type": "WATCH_ONLY",
        }
        for field, value in changes.items():
            with self.subTest(field=field), TemporaryDirectory() as temp:
                root = Path(temp)
                base = pd.DataFrame([base_row])
                changed = pd.DataFrame([{**base_row, field: value}])
                create_or_verify_snapshot(
                    "2026-07-10",
                    {"signal_pool": snapshot(base)},
                    root / "snapshots",
                    root / "run",
                )
                with self.assertRaises(UniverseSnapshotMismatch):
                    create_or_verify_snapshot(
                        "2026-07-10",
                        {"signal_pool": snapshot(changed)},
                        root / "snapshots",
                        root / "run",
                        mode="verify-only",
                    )

    def test_missing_values_share_one_empty_text_representation(self) -> None:
        values = (None, float("nan"), pd.NA, pd.NaT)
        self.assertEqual(
            {universe_audit.canonical_scalar_text(value, "score") for value in values},
            {""},
        )

    def test_negative_and_positive_zero_share_one_representation(self) -> None:
        self.assertEqual(
            universe_audit.canonical_scalar_text(-0.0, "score"),
            "0",
        )
        self.assertEqual(
            universe_audit.canonical_scalar_text(0.0, "score"),
            "0",
        )

    def test_non_finite_numeric_values_are_rejected_with_context(self) -> None:
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                frame = pd.DataFrame(
                    [{"requested_signal_date": "2026-07-10", "code": "000001", "name": "A", "score": value, "flag_bool": True}]
                )
                with self.assertRaisesRegex(
                    UniverseAuditError,
                    r"stage=signal_pool, code=000001, field=score",
                ):
                    universe_audit.canonical_stage_frame(_snapshot(frame))
        with self.assertRaisesRegex(
            UniverseAuditError,
            r"stage=signal_pool, code=000001, field=score",
        ):
            universe_audit.canonical_scalar_text(
                float("nan"),
                "score",
                stage="signal_pool",
                code="000001",
                nan_is_missing=False,
            )

    def test_snapshot_reader_disables_type_inference_and_preserves_codes(self) -> None:
        frame = pd.DataFrame(
            [
                {"requested_signal_date": "2026-07-10", "code": "000001", "name": None, "score": 1.25, "flag_bool": True},
                {"requested_signal_date": "2026-07-10", "code": "000002", "name": "B", "score": 2.5, "flag_bool": False},
            ]
        )
        original_read_csv = pd.read_csv
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(
                universe_audit.pd,
                "read_csv",
                wraps=original_read_csv,
            ) as read_csv:
                status, manifest_path, _ = create_or_verify_snapshot(
                    "2026-07-10",
                    {"signal_pool": _snapshot(frame)},
                    root / "snapshots",
                    root / "run",
                )
            self.assertEqual(status, "CREATED_CANONICAL")
            self.assertEqual(read_csv.call_args.kwargs["dtype"], str)
            self.assertFalse(read_csv.call_args.kwargs["keep_default_na"])
            self.assertFalse(read_csv.call_args.kwargs["na_filter"])
            loaded = original_read_csv(
                manifest_path.parent / "signal_pool.csv",
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
            self.assertEqual(loaded["code"].tolist(), ["000001", "000002"])
            self.assertEqual(loaded["name"].tolist(), ["", "B"])
            self.assertEqual(loaded["flag_bool"].tolist(), ["true", "false"])
            self.assertTrue(all(isinstance(value, str) for value in loaded.iloc[0].tolist()))

    def test_float_hashes_are_input_order_independent(self) -> None:
        frame = pd.DataFrame(
            [
                {"requested_signal_date": "2026-07-10", "code": "000001", "name": "A", "score": 9.1357, "flag_bool": True},
                {"requested_signal_date": "2026-07-10", "code": "000002", "name": "B", "score": 21.755001535818117, "flag_bool": False},
            ]
        )
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        self.assertEqual(code_set_sha256(frame), code_set_sha256(shuffled))
        self.assertEqual(key_set_sha256(frame, KEY_COLUMNS), key_set_sha256(shuffled, KEY_COLUMNS))
        self.assertEqual(
            canonical_rows_sha256(frame, ROW_COLUMNS, KEY_COLUMNS),
            canonical_rows_sha256(shuffled, ROW_COLUMNS, KEY_COLUMNS),
        )


class HistoryUniverseIntegrityTests(unittest.TestCase):
    def test_empty_or_unnormalizable_codes_are_a_hard_failure(self) -> None:
        frame = pd.DataFrame({"code": ["000001", " ", None, "not-a-code"]})
        with self.assertRaisesRegex(
            UniverseAuditError,
            r"test layer contains empty or unnormalizable codes.*invalid_count=3",
        ):
            require_nonempty_codes(frame, "code", "test layer")

    def test_duplicate_requested_date_code_is_a_hard_failure(self) -> None:
        rows = pd.DataFrame(
            [
                _candidate_row("2026-07-10", "2026-07-10", "1"),
                _candidate_row("2026-07-10", "2026-07-10", "000001"),
            ]
        )
        with self.assertRaisesRegex(UniverseAuditError, "duplicate keys"):
            _normalise_history_candidate_columns(rows)

    def test_requested_and_actual_signal_date_mismatch_is_a_hard_failure(self) -> None:
        rows = pd.DataFrame([_candidate_row("2026-07-10", "2026-07-11", "1")])
        with self.assertRaisesRegex(
            UniverseAuditError,
            r"requested_signal_date=2026-07-10, signal_date=2026-07-11, code=000001",
        ):
            _normalise_history_candidate_columns(rows)

    def test_v004a_filter_uses_the_shared_annotation(self) -> None:
        raw = pd.DataFrame(
            [
                _candidate_row("2026-07-10", "2026-07-10", "1"),
                {**_candidate_row("2026-07-10", "2026-07-10", "2"), "d2open_d3high_return_pct": None},
                {**_candidate_row("2026-07-10", "2026-07-10", "3"), "candidate_base_price": 0.0},
            ]
        )
        annotated = annotate_v004a_input_eligibility(raw)
        expected = annotated[annotated["v004a_scorable_bool"]][["signal_date", "code"]].reset_index(drop=True)
        prepared, _, quality = prepare_v004a_samples(raw)
        actual = prepared[["signal_date", "code"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(int(quality.iloc[0]["final_rows"]), len(expected))

    def test_v004a_exclusion_reasons_have_stable_order(self) -> None:
        row = _candidate_row("2026-07-10", "2026-07-11", "1")
        row.update(
            {
                "eligible_for_trade": False,
                "d2open_d3high_return_pct": None,
                "d2open_d3close_return_pct": None,
                "candidate_base_price": 0.0,
            }
        )
        annotated = annotate_v004a_input_eligibility(pd.DataFrame([row]))
        self.assertEqual(
            annotated.iloc[0]["v004a_exclusion_reason"],
            "not_eligible|missing_high_return|missing_close_return|invalid_base_price|signal_date_mismatch",
        )


class CanonicalSnapshotTests(unittest.TestCase):
    def test_failed_second_stage_write_leaves_no_canonical_or_temp_directory(self) -> None:
        frame = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True}]
        )
        stages = {
            "raw_source_pool": _stage_snapshot("raw_source_pool", frame),
            "signal_pool": _stage_snapshot("signal_pool", frame),
        }
        original = universe_audit.write_canonical_csv
        calls = 0

        def fail_second(path: Path, snapshot: StageSnapshot) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-stage failure")
            original(path, snapshot)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch("src.universe_audit.write_canonical_csv", side_effect=fail_second):
                with self.assertRaises(UniverseSnapshotError) as caught:
                    create_or_verify_snapshot(
                        "2026-07-10", stages, root / "snapshots", root / "run"
                    )
            self.assertEqual(caught.exception.status, "SNAPSHOT_WRITE_FAILED")
            date_root = root / "snapshots" / "history_universe" / "2026-07-10"
            self.assertFalse((date_root / "canonical").exists())
            self.assertEqual(list(date_root.glob(".canonical.tmp-*")), [])

    def test_malformed_manifest_is_reported_as_corrupt_canonical(self) -> None:
        frame = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True}]
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, manifest_path, _ = create_or_verify_snapshot(
                "2026-07-10", {"signal_pool": _snapshot(frame)}, root / "snapshots", root / "run"
            )
            manifest_path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(UniverseSnapshotError) as caught:
                create_or_verify_snapshot(
                    "2026-07-10", {"signal_pool": _snapshot(frame)}, root / "snapshots", root / "run"
                )
            self.assertEqual(caught.exception.status, "CORRUPT_CANONICAL")

    def test_missing_or_corrupt_stage_csv_is_a_hard_failure(self) -> None:
        frame = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True}]
        )
        for corruption, expected_status in (("missing", "INCOMPLETE_CANONICAL"), ("corrupt", "CORRUPT_CANONICAL")):
            with self.subTest(corruption=corruption), TemporaryDirectory() as temp:
                root = Path(temp)
                _, manifest_path, _ = create_or_verify_snapshot(
                    "2026-07-10", {"signal_pool": _snapshot(frame)}, root / "snapshots", root / "run"
                )
                stage_path = manifest_path.parent / "signal_pool.csv"
                if corruption == "missing":
                    stage_path.unlink()
                else:
                    stage_path.write_bytes(b"\xff\xfe\x00broken")
                with self.assertRaises(UniverseSnapshotError) as caught:
                    create_or_verify_snapshot(
                        "2026-07-10", {"signal_pool": _snapshot(frame)}, root / "snapshots", root / "run"
                    )
                self.assertEqual(caught.exception.status, expected_status)

    def test_empty_code_never_creates_canonical(self) -> None:
        frame = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": None, "name": "A", "score": 1.0, "flag_bool": True}]
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(UniverseAuditError, "empty or unnormalizable codes"):
                create_or_verify_snapshot(
                    "2026-07-10", {"signal_pool": _snapshot(frame)}, root / "snapshots", root / "run"
                )
            canonical = root / "snapshots" / "history_universe" / "2026-07-10" / "canonical"
            self.assertFalse(canonical.exists())

    def test_concurrent_create_publishes_one_complete_canonical(self) -> None:
        frame = pd.DataFrame(
            [
                {"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True},
                {"requested_signal_date": "2026-07-10", "code": "2", "name": "B", "score": 2.0, "flag_bool": False},
            ]
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)

            def create(index: int) -> str:
                status, _, _ = create_or_verify_snapshot(
                    "2026-07-10",
                    {"signal_pool": _snapshot(frame)},
                    root / "snapshots",
                    root / f"run-{index}",
                )
                return status

            with ThreadPoolExecutor(max_workers=2) as executor:
                statuses = sorted(executor.map(create, (1, 2)))
            self.assertEqual(statuses, ["CREATED_CANONICAL", "VERIFIED_MATCH"])
            canonical = root / "snapshots" / "history_universe" / "2026-07-10" / "canonical"
            self.assertTrue((canonical / "manifest.json").is_file())
            self.assertTrue((canonical / "signal_pool.csv").is_file())
            self.assertEqual(list(canonical.parent.glob(".canonical.tmp-*")), [])

    def test_verify_only_requires_canonical_and_off_never_creates_it(self) -> None:
        frame = pd.DataFrame(
            [{"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True}]
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(UniverseSnapshotError, "required but missing"):
                create_or_verify_snapshot(
                    "2026-07-10",
                    {"signal_pool": _snapshot(frame)},
                    root / "snapshots",
                    root / "run",
                    mode="verify-only",
                )
            status, manifest_path, _ = create_or_verify_snapshot(
                "2026-07-10",
                {"signal_pool": _snapshot(frame)},
                root / "snapshots",
                root / "run",
                mode="off",
            )
            self.assertEqual(status, "UNVERIFIED_OFF")
            self.assertFalse(manifest_path.exists())

    def test_create_verify_and_mismatch_without_overwrite(self) -> None:
        base = pd.DataFrame(
            [
                {"requested_signal_date": "2026-07-10", "code": "1", "name": "A", "score": 1.0, "flag_bool": True},
                {"requested_signal_date": "2026-07-10", "code": "2", "name": "B", "score": 2.0, "flag_bool": False},
            ]
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            status, manifest_path, _ = create_or_verify_snapshot(
                "2026-07-10", {"signal_pool": _snapshot(base)}, root / "snapshots", root / "run"
            )
            self.assertEqual(status, "CREATED_CANONICAL")
            original_manifest = manifest_path.read_bytes()

            status, _, _ = create_or_verify_snapshot(
                "2026-07-10",
                {"signal_pool": _snapshot(base.iloc[::-1].reset_index(drop=True))},
                root / "snapshots",
                root / "run",
            )
            self.assertEqual(status, "VERIFIED_MATCH")

            added = pd.concat(
                [
                    base,
                    pd.DataFrame(
                        [{"requested_signal_date": "2026-07-10", "code": "3", "name": "C", "score": 3.0, "flag_bool": True}]
                    ),
                ],
                ignore_index=True,
            )
            with self.assertRaises(UniverseSnapshotMismatch):
                create_or_verify_snapshot(
                    "2026-07-10", {"signal_pool": _snapshot(added)}, root / "snapshots", root / "run"
                )
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            diff = pd.read_csv(root / "run" / "universe_snapshot_diff_2026-07-10.csv", dtype={"code": str})
            self.assertEqual(diff["diff_type"].tolist(), ["added"])
            self.assertEqual(diff["code"].tolist(), ["000003"])


def _candidate_row(requested: str, actual: str, code: str) -> dict[str, object]:
    return {
        "requested_signal_date": requested,
        "signal_date": actual,
        "code": code,
        "eligible_for_trade": True,
        "target7_d2open_d3high": True,
        "d2open_d3high_return_pct": 8.0,
        "d2open_d3close_return_pct": 4.0,
        "candidate_base_price": 10.0,
        "d1_close_ma10_pct": 2.0,
        "d1_low_ma10_pct": -1.0,
        "trend_hold_score": 80.0,
        "total_score": 75.0,
        "theme_score": 60.0,
        "graph_quality_score": 70.0,
        "days_since_d0": 1,
    }


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

from src.backtester import build_signals_for_pool
from src.config import DataConfig
from src.history_samples import (
    HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS,
    _annotate_raw_source_exclusions,
    _build_history_stage_snapshots,
    _build_history_universe_membership,
    _quality_exclusion_metadata,
    _standardize_raw_source_pool,
    _standardize_signal_pool,
    run_history_sample_generation,
)
from src.loaders import (
    DataQualityError,
    MarketDataService,
    _build_quality_report,
    required_daily_trade_days,
)
from src.universe_audit import canonical_rows_sha256, stage_identity


SIGNAL_DATE = "2026-06-26"


def _quality_failure(*failure_codes: str, code: str = "603407") -> dict[str, object]:
    messages = {
        "daily_history_shortfall": "daily history rows 34 < required 180",
        "missing_latest_daily_ma": "missing latest daily MA: ma30",
        "minute_history_shortfall": "minute trade days 3 < required 4",
        "daily_minute_close_mismatch": "daily/minute close cross-check failed",
    }
    failures = [
        {"code": failure_code, "message": messages[failure_code]}
        for failure_code in failure_codes
    ]
    return {
        "code": code,
        "status": "failed",
        "daily_history_rows": 34,
        "daily_required_days": 180,
        "hard_failure_codes": tuple(failure_codes),
        "hard_failure_codes_json": json.dumps(list(failure_codes), separators=(",", ":")),
        "hard_failures": failures,
        "hard_failures_json": json.dumps(failures, sort_keys=True, separators=(",", ":")),
        "warnings": "; ".join(messages[item] for item in failure_codes),
    }


def _excluded_quality(code: str, listing_date: str) -> dict[str, object]:
    maximum = len(pd.bdate_range(listing_date, SIGNAL_DATE, inclusive="both"))
    evidence = {
        "listing_date": listing_date,
        "listing_date_source": "eastmoney_stock_info_f189",
        "maximum_possible_trade_days": maximum,
        "proof_method": "weekday_upper_bound_v1",
        "required_trade_days": 180,
        "signal_date": SIGNAL_DATE,
    }
    return {
        **_quality_failure("daily_history_shortfall", code=code),
        "name": f"Name {code}",
        "trade_date": SIGNAL_DATE,
        "d0_date": SIGNAL_DATE,
        "status": "excluded",
        "quality_excluded": True,
        "exclusion_reason": "insufficient_listing_history",
        "listing_date": listing_date,
        "listing_date_source": "eastmoney_stock_info_f189",
        "signal_date": SIGNAL_DATE,
        "required_trade_days": 180,
        "maximum_possible_trade_days": maximum,
        "proof_method": "weekday_upper_bound_v1",
        "exclusion_evidence_json": json.dumps(
            evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "is_data_quality_error": False,
        "error_class": "",
        "error": "",
        "daily_source": "tencent_daily",
        "minute_source": "sina_5m",
        "daily_ma_coverage_ok": True,
        "daily_minute_close_check_ok": True,
    }


class _FailureService:
    def __init__(self, quality: dict[str, object], metadata: dict[str, object] | Exception) -> None:
        self.quality = quality
        self.metadata = metadata
        self.metadata_calls = 0

    def get_stock_bars(self, *_: object, **__: object) -> object:
        raise DataQualityError("short daily history", dict(self.quality))

    def resolve_listing_metadata(self, *_: object, **__: object) -> dict[str, object]:
        self.metadata_calls += 1
        if isinstance(self.metadata, Exception):
            raise self.metadata
        return dict(self.metadata)


class _FrameCache:
    def read(self, _: str) -> None:
        return None


class _OneDateHistoryService:
    def __init__(self, pool: pd.DataFrame) -> None:
        self.pool = pool
        self.limit_up_cache = _FrameCache()

    def collect_limit_ups(self, *_: object, **__: object) -> pd.DataFrame:
        return self.pool.copy()


class ListingHistoryDecisionTests(unittest.TestCase):
    def _run(self, service: _FailureService, code: str = "603407") -> dict[str, object]:
        pool = pd.DataFrame(
            [{"trade_date": SIGNAL_DATE, "code": code, "name": f"Name {code}"}]
        )
        signals, quality_rows = build_signals_for_pool(service, pool, SIGNAL_DATE, days=10)
        self.assertEqual(signals, [])
        self.assertEqual(len(quality_rows), 1)
        return quality_rows[0]

    def test_new_listing_is_structurally_excluded(self) -> None:
        service = _FailureService(
            _quality_failure("daily_history_shortfall"),
            {
                "code": "603407",
                "listing_date": "2026-05-11",
                "metadata_source": "eastmoney_stock_info_f189",
                "schema_version": 1,
                "from_cache": False,
            },
        )
        row = self._run(service)
        self.assertEqual(row["status"], "excluded")
        self.assertEqual(row["exclusion_reason"], "insufficient_listing_history")
        self.assertFalse(row["is_data_quality_error"])
        self.assertLess(int(row["maximum_possible_trade_days"]), 180)
        self.assertEqual(row["proof_method"], "weekday_upper_bound_v1")
        self.assertEqual(service.metadata_calls, 1)

    def test_old_listing_with_short_data_remains_a_quality_failure(self) -> None:
        service = _FailureService(
            _quality_failure("daily_history_shortfall", code="000001"),
            {
                "code": "000001",
                "listing_date": "1991-04-03",
                "metadata_source": "eastmoney_stock_info_f189",
                "schema_version": 1,
                "from_cache": True,
            },
        )
        row = self._run(service, code="000001")
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["is_data_quality_error"])
        self.assertEqual(row["listing_history_check_status"], "sufficient_history_possible")

    def test_unknown_listing_date_remains_a_quality_failure(self) -> None:
        service = _FailureService(
            _quality_failure("daily_history_shortfall"),
            RuntimeError("listing metadata unavailable"),
        )
        row = self._run(service)
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["is_data_quality_error"])
        self.assertEqual(row["listing_history_check_status"], "listing_metadata_unavailable")

    def test_other_quality_failure_prevents_structural_exclusion(self) -> None:
        service = _FailureService(
            _quality_failure("daily_history_shortfall", "daily_minute_close_mismatch"),
            {
                "code": "603407",
                "listing_date": "2026-05-11",
                "metadata_source": "eastmoney_stock_info_f189",
            },
        )
        row = self._run(service)
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["is_data_quality_error"])
        self.assertEqual(row["listing_history_check_status"], "not_applicable")
        self.assertEqual(service.metadata_calls, 0)

    def test_weekday_upper_bound_179_excludes_and_180_does_not(self) -> None:
        for periods, expected_status in ((179, "excluded"), (180, "failed")):
            with self.subTest(periods=periods):
                listing_date = pd.bdate_range(end=SIGNAL_DATE, periods=periods)[0].strftime(
                    "%Y-%m-%d"
                )
                service = _FailureService(
                    _quality_failure("daily_history_shortfall"),
                    {
                        "code": "603407",
                        "listing_date": listing_date,
                        "metadata_source": "eastmoney_stock_info_f189",
                    },
                )
                row = self._run(service)
                self.assertEqual(int(row["maximum_possible_trade_days"]), periods)
                self.assertEqual(row["status"], expected_status)


class ListingMetadataCacheTests(unittest.TestCase):
    def test_listing_metadata_is_fetched_once_then_reused_from_stable_cache(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = DataConfig(
                raw_dir=root / "raw",
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            first_service = MarketDataService(config)
            first_fetch = mock.Mock(
                return_value={
                    "code": "603407",
                    "listing_date": "2026-05-11",
                    "metadata_source": "eastmoney_stock_info_f189",
                }
            )
            first_service.provider.fetch_listing_metadata = first_fetch
            first = first_service.resolve_listing_metadata("603407")
            self.assertFalse(first["from_cache"])
            first_fetch.assert_called_once_with("603407")

            second_service = MarketDataService(config)
            second_fetch = mock.Mock(side_effect=AssertionError("provider should not be called"))
            second_service.provider.fetch_listing_metadata = second_fetch
            second = second_service.resolve_listing_metadata("603407")
            self.assertTrue(second["from_cache"])
            self.assertEqual(second["listing_date"], first["listing_date"])
            self.assertEqual(second["metadata_source"], first["metadata_source"])
            second_fetch.assert_not_called()

            cache_path = root / "cache" / "listing_metadata" / "603407_listing_metadata.pkl"
            cached = pd.read_pickle(cache_path)
            self.assertEqual(
                set(cached.columns),
                {"code", "listing_date", "metadata_source", "schema_version", "fetched_at"},
            )


class ListingQualityStructureTests(unittest.TestCase):
    def test_quality_report_exposes_structured_failure_codes(self) -> None:
        dates = pd.bdate_range(end=SIGNAL_DATE, periods=34).strftime("%Y-%m-%d")
        daily_history = pd.DataFrame({"date": dates})
        daily = pd.DataFrame(
            {
                "date": dates[-10:],
                "close": [10.0] * 10,
                "amount": [1.0] * 10,
                "volume": [1.0] * 10,
                "ma5": [10.0] * 10,
                "ma10": [10.0] * 10,
                "ma20": [10.0] * 10,
                "ma30": [10.0] * 10,
            }
        )
        minute = pd.DataFrame(
            {
                "trade_date": [dates[-1]],
                "datetime": [f"{dates[-1]} 15:00:00"],
                "close": [10.0],
                "amount": [1.0],
                "volume": [1.0],
            }
        )
        report = _build_quality_report(
            code="603407",
            daily=daily,
            minute=minute,
            daily_history=daily_history,
            daily_source="daily",
            minute_source="minute",
            from_cache=False,
            daily_required_days=180,
            minute_target_days=1,
            minute_required_days=1,
        )
        self.assertEqual(report["hard_failure_codes"], ("daily_history_shortfall",))
        self.assertEqual(
            json.loads(str(report["hard_failure_codes_json"])),
            ["daily_history_shortfall"],
        )

    def test_required_daily_days_uses_one_shared_formula(self) -> None:
        config = DataConfig(
            daily_history_days=180,
            default_5min_days=40,
            indicator_warmup_trading_days=120,
        )
        self.assertEqual(required_daily_trade_days(config, 10), 180)
        self.assertEqual(required_daily_trade_days(config, 100), 220)


class ListingMembershipAndAuditTests(unittest.TestCase):
    def test_strict_history_run_records_exclusion_and_verify_only_matches(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            pool = pd.DataFrame(
                [{"trade_date": SIGNAL_DATE, "code": "603407", "name": "长裕集团"}]
            )
            service = _OneDateHistoryService(pool)
            exclusion = _excluded_quality("603407", "2026-05-11")
            for mode, expected_snapshot_status in (
                ("create-or-verify", "CREATED_CANONICAL"),
                ("verify-only", "VERIFIED_MATCH"),
            ):
                with (
                    mock.patch("src.history_samples.get_data_config", return_value=config),
                    mock.patch("src.history_samples.MarketDataService", return_value=service),
                    mock.patch(
                        "src.history_samples._cached_daily_proves_non_trading",
                        return_value=False,
                    ),
                    mock.patch(
                        "src.history_samples.build_signals_for_pool",
                        return_value=([], [dict(exclusion)]),
                    ),
                ):
                    candidates, _, run_log, _, *_ = run_history_sample_generation(
                        start_date=SIGNAL_DATE,
                        end_date=SIGNAL_DATE,
                        lookback_days=1,
                        universe_snapshot_mode=mode,
                    )
                self.assertTrue(candidates.empty)
                self.assertEqual(int(run_log.iloc[0]["quality_excluded"]), 1)
                self.assertEqual(int(run_log.iloc[0]["quality_failed"]), 0)
                self.assertEqual(run_log.iloc[0]["snapshot_status"], expected_snapshot_status)

            stable = config.reports_dir / "history_samples" / f"{SIGNAL_DATE}_{SIGNAL_DATE}"
            membership = pd.read_csv(
                stable / f"history_universe_membership_{SIGNAL_DATE}_{SIGNAL_DATE}.csv",
                dtype={"code": str},
            )
            raw = membership[membership["stage"].eq("raw_source_pool")].iloc[0]
            signal = membership[membership["stage"].eq("signal_pool")].iloc[0]
            self.assertTrue(bool(raw["included_bool"]))
            self.assertFalse(bool(signal["included_bool"]))
            self.assertEqual(signal["exclusion_reason"], "insufficient_listing_history")
            self.assertEqual(signal["listing_date"], "2026-05-11")
            self.assertEqual(signal["listing_date_source"], "eastmoney_stock_info_f189")
            self.assertEqual(signal["signal_date"], SIGNAL_DATE)
            self.assertEqual(int(signal["required_trade_days"]), 180)
            self.assertLess(int(signal["maximum_possible_trade_days"]), 180)
            self.assertEqual(signal["proof_method"], "weekday_upper_bound_v1")
            self.assertEqual(
                json.loads(signal["exclusion_evidence_json"])["listing_date"],
                "2026-05-11",
            )

    def test_downstream_integrity_failure_still_persists_exclusion_membership(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(
                reports_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            pool = pd.DataFrame(
                [
                    {"trade_date": SIGNAL_DATE, "code": "603407", "name": "长裕集团"},
                    {"trade_date": SIGNAL_DATE, "code": "603001", "name": "Mismatch"},
                ]
            )
            service = _OneDateHistoryService(pool)
            mismatched_signal = pd.DataFrame(
                [
                    {
                        "trade_date": "2026-06-24",
                        "code": "603001",
                        "name": "Mismatch",
                        "allowed": True,
                        "signal_type": "D2_LOW_ABSORB",
                    }
                ]
            )
            with (
                mock.patch("src.history_samples.get_data_config", return_value=config),
                mock.patch("src.history_samples.MarketDataService", return_value=service),
                mock.patch(
                    "src.history_samples._cached_daily_proves_non_trading",
                    return_value=False,
                ),
                mock.patch(
                    "src.history_samples.build_signals_for_pool",
                    return_value=([], [_excluded_quality("603407", "2026-05-11")]),
                ),
                mock.patch(
                    "src.history_samples._signals_to_frame",
                    return_value=mismatched_signal,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "signal_date_mismatch"):
                    run_history_sample_generation(
                        start_date=SIGNAL_DATE,
                        end_date=SIGNAL_DATE,
                        lookback_days=1,
                        universe_snapshot_mode="create-or-verify",
                    )

            attempts = (
                config.reports_dir
                / "history_samples"
                / f"{SIGNAL_DATE}_{SIGNAL_DATE}"
                / "attempts"
            )
            attempt = next(attempts.iterdir())
            membership = pd.read_csv(
                attempt / f"history_universe_membership_{SIGNAL_DATE}_{SIGNAL_DATE}.csv",
                dtype={"code": str},
            )
            excluded = membership[
                membership["stage"].eq("signal_pool")
                & membership["code"].eq("603407")
            ].iloc[0]
            self.assertFalse(bool(excluded["included_bool"]))
            self.assertEqual(excluded["exclusion_reason"], "insufficient_listing_history")

    def test_shuffled_inputs_preserve_four_stage_and_exclusion_hashes(self) -> None:
        raw_input = pd.DataFrame(
            [
                {"trade_date": SIGNAL_DATE, "code": code, "name": f"Name {code}"}
                for code in ("000001", "000002", "000003", "603407", "603459")
            ]
        )
        quality = [
            _excluded_quality("603407", "2026-05-11"),
            _excluded_quality("603459", "2026-04-08"),
        ]
        signal_input = pd.DataFrame(
            [
                {
                    "trade_date": SIGNAL_DATE,
                    "code": code,
                    "name": f"Name {code}",
                    "allowed": True,
                    "signal_type": "D2_LOW_ABSORB",
                }
                for code in ("000001", "000002", "000003")
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "requested_signal_date": SIGNAL_DATE,
                    "signal_date": SIGNAL_DATE,
                    "code": code,
                    "v004a_scorable_bool": True,
                }
                for code in ("000001", "000002", "000003")
            ]
        )

        def build(
            raw_rows: pd.DataFrame,
            signal_rows: pd.DataFrame,
            candidate_rows: pd.DataFrame,
            quality_rows: list[dict[str, object]],
        ) -> tuple[dict[str, str], str, str]:
            raw = _annotate_raw_source_exclusions(
                _standardize_raw_source_pool(raw_rows, SIGNAL_DATE), quality_rows
            )
            signals = _standardize_signal_pool(signal_rows, SIGNAL_DATE, SIGNAL_DATE)
            stages = _build_history_stage_snapshots(
                SIGNAL_DATE, SIGNAL_DATE, raw, signals, candidate_rows
            )
            stage_hashes = {
                name: str(stage_identity(snapshot)["rows_sha256"])
                for name, snapshot in stages.items()
            }
            membership = _build_history_universe_membership(
                SIGNAL_DATE,
                SIGNAL_DATE,
                raw,
                signals,
                candidate_rows,
                quality_rows=quality_rows,
            )
            membership_hash = canonical_rows_sha256(
                membership,
                HISTORY_UNIVERSE_MEMBERSHIP_COLUMNS,
                ("requested_signal_date", "stage", "member_key"),
            )
            exclusion_hash = _quality_exclusion_metadata(quality_rows)[
                "quality_exclusion_evidence_sha256"
            ]
            return stage_hashes, membership_hash, exclusion_hash

        baseline = build(raw_input, signal_input, candidates, quality)
        shuffled = build(
            raw_input.sample(frac=1.0, random_state=27).reset_index(drop=True),
            signal_input.sample(frac=1.0, random_state=72).reset_index(drop=True),
            candidates.sample(frac=1.0, random_state=7).reset_index(drop=True),
            list(reversed(quality)),
        )
        self.assertEqual(baseline, shuffled)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

from src.backtester import build_signals_for_pool
from src.config import DataConfig
from src.data_sources import (
    SUSPENSION_STATUS_SOURCE,
    normalize_suspension_status_frame,
)
from src.history_samples import (
    HISTORY_CANDIDATE_COLUMNS,
    _annotate_raw_source_exclusions,
    _build_history_universe_membership,
    _quality_exclusion_metadata,
    _standardize_raw_source_pool,
    _standardize_signal_pool,
)
from src.loaders import DataQualityError, MarketDataService, _build_quality_report
from src.universe_audit import UniverseAuditError, require_requested_signal_date_match


SIGNAL_DATE = "2026-06-26"
STALE_DATE = "2026-06-24"


def _coverage_quality(
    *,
    code: str = "603001",
    daily_has_requested_date: bool = False,
    minute_has_requested_date: bool = False,
) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    latest_daily = SIGNAL_DATE if daily_has_requested_date else STALE_DATE
    latest_minute = SIGNAL_DATE if minute_has_requested_date else STALE_DATE
    if not daily_has_requested_date:
        failures.extend(
            [
                {
                    "code": "missing_signal_date_daily_bar",
                    "message": f"daily data does not contain requested signal date {SIGNAL_DATE}",
                },
                {
                    "code": "stale_daily_end_date",
                    "message": f"latest daily date {STALE_DATE} < requested {SIGNAL_DATE}",
                },
            ]
        )
    if not minute_has_requested_date:
        failures.extend(
            [
                {
                    "code": "missing_signal_date_minute_bar",
                    "message": f"minute data does not contain requested signal date {SIGNAL_DATE}",
                },
                {
                    "code": "stale_minute_end_date",
                    "message": f"latest minute trade date {STALE_DATE} < requested {SIGNAL_DATE}",
                },
            ]
        )
    codes = [item["code"] for item in failures]
    return {
        "code": code,
        "status": "failed",
        "daily_source": "tencent_daily",
        "minute_source": "sina_5m",
        "from_cache": False,
        "daily_history_rows": 180,
        "daily_required_days": 180,
        "requested_signal_date": SIGNAL_DATE,
        "latest_daily_date": latest_daily,
        "daily_has_requested_date": daily_has_requested_date,
        "latest_minute_trade_date": latest_minute,
        "minute_has_requested_date": minute_has_requested_date,
        "hard_failure_codes": tuple(codes),
        "hard_failure_codes_json": json.dumps(codes, separators=(",", ":")),
        "hard_failures": failures,
        "hard_failures_json": json.dumps(failures, sort_keys=True, separators=(",", ":")),
        "warnings": "; ".join(item["message"] for item in failures),
    }


def _provider_probe(*, tencent_has_date: bool = False, sina_has_date: bool = False):
    return {
        "code": "603001",
        "query_date": SIGNAL_DATE,
        "sources": [
            {
                "source": "tencent_daily",
                "status": "ok",
                "has_requested_date": tencent_has_date,
                "latest_date": SIGNAL_DATE if tencent_has_date else STALE_DATE,
                "error": "",
            },
            {
                "source": "sina_daily",
                "status": "ok",
                "has_requested_date": sina_has_date,
                "latest_date": SIGNAL_DATE if sina_has_date else STALE_DATE,
                "error": "",
            },
        ],
    }


def _suspension_proof(*, from_cache: bool = False) -> dict[str, object]:
    return {
        "code": "603001",
        "signal_date": SIGNAL_DATE,
        "suspension_start_date": "2026-06-25",
        "suspension_end_date": "2026-07-01",
        "suspension_duration": "连续停牌",
        "suspension_reason": "刊登重要公告",
        "suspension_market": "上交所主板",
        "expected_resume_date": "2026-07-02",
        "suspension_source": SUSPENSION_STATUS_SOURCE,
        "proof_method": "eastmoney_suspension_status_v1",
        "normalized_rows_sha256": "a" * 64,
        "schema_version": 1,
        "from_cache": from_cache,
    }


class _CoverageFailureService:
    def __init__(
        self,
        quality: dict[str, object],
        *,
        probe: dict[str, object] | Exception | None = None,
        proof: dict[str, object] | Exception | None = None,
    ) -> None:
        self.quality = quality
        self.probe = _provider_probe() if probe is None else probe
        self.proof = proof
        self.probe_calls = 0
        self.resolver_calls = 0

    def get_stock_bars(self, *_: object, **__: object) -> object:
        raise DataQualityError("exact signal-date coverage failed", dict(self.quality))

    def probe_daily_signal_date_sources(self, *_: object, **__: object) -> dict[str, object]:
        self.probe_calls += 1
        if isinstance(self.probe, Exception):
            raise self.probe
        return dict(self.probe)

    def resolve_suspension_on_date(self, *_: object, **__: object):
        self.resolver_calls += 1
        if isinstance(self.proof, Exception):
            raise self.proof
        return None if self.proof is None else dict(self.proof)


def _pool(code: str = "603001") -> pd.DataFrame:
    return pd.DataFrame([{"trade_date": SIGNAL_DATE, "code": code, "name": "奥康国际"}])


class SignalDateCoverageDecisionTests(unittest.TestCase):
    def test_exact_daily_and_minute_date_coverage_generates_signal(self) -> None:
        history_dates = pd.bdate_range(end=SIGNAL_DATE, periods=180).strftime("%Y-%m-%d")
        recent_dates = history_dates[-10:]
        daily = pd.DataFrame(
            {
                "date": recent_dates,
                "close": [10.0] * len(recent_dates),
                "amount": [1.0] * len(recent_dates),
                "volume": [1.0] * len(recent_dates),
                "ma5": [10.0] * len(recent_dates),
                "ma10": [10.0] * len(recent_dates),
                "ma20": [10.0] * len(recent_dates),
                "ma30": [10.0] * len(recent_dates),
            }
        )
        minute = pd.DataFrame(
            {
                "trade_date": [SIGNAL_DATE],
                "datetime": [f"{SIGNAL_DATE} 15:00:00"],
                "close": [10.0],
                "amount": [1.0],
                "volume": [1.0],
            }
        )
        quality = _build_quality_report(
            code="603001",
            daily=daily,
            minute=minute,
            daily_history=pd.DataFrame({"date": history_dates}),
            daily_source="tencent_daily",
            minute_source="sina_5m",
            from_cache=False,
            daily_required_days=180,
            minute_target_days=1,
            minute_required_days=1,
            requested_signal_date=SIGNAL_DATE,
        )
        self.assertEqual(quality["status"], "ok")
        self.assertTrue(quality["daily_has_requested_date"])
        self.assertTrue(quality["minute_has_requested_date"])
        service = SimpleNamespace(
            get_stock_bars=mock.Mock(
                return_value=SimpleNamespace(
                    quality=quality,
                    daily=daily,
                    minute_5m=minute,
                )
            )
        )
        expected_signal = object()
        with mock.patch("src.backtester.generate_signal", return_value=expected_signal) as generate:
            signals, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        self.assertEqual(signals, [expected_signal])
        self.assertEqual(rows[0]["status"], "ok")
        generate.assert_called_once()

    def test_stale_bars_with_strict_suspension_proof_are_excluded_before_signal(self) -> None:
        service = _CoverageFailureService(
            _coverage_quality(), proof=_suspension_proof()
        )
        with mock.patch("src.backtester.generate_signal") as generate:
            signals, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        self.assertEqual(signals, [])
        generate.assert_not_called()
        row = rows[0]
        self.assertEqual(row["status"], "excluded")
        self.assertEqual(row["exclusion_reason"], "suspended_on_signal_date")
        self.assertFalse(row["is_data_quality_error"])
        self.assertEqual(row["suspension_start_date"], "2026-06-25")
        self.assertEqual(row["suspension_end_date"], "2026-07-01")
        self.assertEqual(service.probe_calls, 1)
        self.assertEqual(service.resolver_calls, 1)

    def test_stale_bars_without_suspension_proof_remain_quality_failure(self) -> None:
        service = _CoverageFailureService(_coverage_quality(), proof=None)
        _, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        row = rows[0]
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["is_data_quality_error"])
        self.assertIn("suspension_status_not_proven", row["hard_failure_codes"])

    def test_daily_current_but_minute_stale_is_not_a_suspension_exclusion(self) -> None:
        service = _CoverageFailureService(
            _coverage_quality(daily_has_requested_date=True),
            proof=_suspension_proof(),
        )
        _, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("signal_date_cross_frequency_conflict", rows[0]["hard_failure_codes"])
        self.assertEqual(service.probe_calls, 0)
        self.assertEqual(service.resolver_calls, 0)

    def test_tencent_stale_sina_current_is_provider_conflict_not_suspension(self) -> None:
        service = _CoverageFailureService(
            _coverage_quality(),
            probe=_provider_probe(tencent_has_date=False, sina_has_date=True),
            proof=_suspension_proof(),
        )
        _, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("daily_provider_coverage_conflict", rows[0]["hard_failure_codes"])
        self.assertEqual(service.resolver_calls, 0)

    def test_provider_or_suspension_resolver_failure_remains_quality_failure(self) -> None:
        cases = (
            (
                _CoverageFailureService(
                    _coverage_quality(), probe=RuntimeError("Tencent unavailable")
                ),
                "daily_provider_coverage_unconfirmed",
            ),
            (
                _CoverageFailureService(
                    _coverage_quality(), proof=RuntimeError("suspension API unavailable")
                ),
                "suspension_status_unavailable",
            ),
        )
        for service, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                _, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
                self.assertEqual(rows[0]["status"], "failed")
                self.assertTrue(rows[0]["is_data_quality_error"])
                self.assertIn(expected_code, rows[0]["hard_failure_codes"])

    def test_suspension_proof_with_unknown_end_date_remains_quality_failure(self) -> None:
        invalid_proof = _suspension_proof()
        invalid_proof["suspension_end_date"] = ""
        service = _CoverageFailureService(_coverage_quality(), proof=invalid_proof)
        _, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertTrue(rows[0]["is_data_quality_error"])
        self.assertIn("suspension_status_invalid", rows[0]["hard_failure_codes"])


class SuspensionResolverCacheTests(unittest.TestCase):
    @staticmethod
    def _normalized_cross_section() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "code": "603001",
                    "name": "奥康国际",
                    "suspension_start_date": "2026-06-25",
                    "suspension_end_date": "2026-07-01",
                    "suspension_duration": "连续停牌",
                    "suspension_reason": "刊登重要公告",
                    "suspension_market": "上交所主板",
                    "expected_resume_date": "2026-07-02",
                },
                {
                    "code": "600000",
                    "name": "Future",
                    "suspension_start_date": "2026-07-13",
                    "suspension_end_date": "2026-07-13",
                    "suspension_duration": "停牌一天",
                    "suspension_reason": "未来事件",
                    "suspension_market": "上交所主板",
                    "expected_resume_date": "2026-07-14",
                },
            ]
        )

    def test_date_cross_section_is_cached_and_reused_with_same_hash(self) -> None:
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
                return_value=(self._normalized_cross_section(), SUSPENSION_STATUS_SOURCE)
            )
            first_service.provider.fetch_suspension_status = first_fetch
            first = first_service.resolve_suspension_on_date("603001", SIGNAL_DATE)
            self.assertIsNotNone(first)
            self.assertFalse(first["from_cache"])
            first_fetch.assert_called_once_with(SIGNAL_DATE)

            second_service = MarketDataService(config)
            second_fetch = mock.Mock(side_effect=AssertionError("provider must not be called"))
            second_service.provider.fetch_suspension_status = second_fetch
            second = second_service.resolve_suspension_on_date("603001", SIGNAL_DATE)
            self.assertIsNotNone(second)
            self.assertTrue(second["from_cache"])
            second_fetch.assert_not_called()
            stable_fields = {
                "code",
                "signal_date",
                "suspension_start_date",
                "suspension_end_date",
                "suspension_reason",
                "suspension_source",
                "proof_method",
                "normalized_rows_sha256",
            }
            self.assertEqual(
                {key: first[key] for key in stable_fields},
                {key: second[key] for key in stable_fields},
            )
            cache_path = (
                root
                / "cache"
                / "suspension_status"
                / f"{SIGNAL_DATE}_suspension_status.pkl"
            )
            cached = pd.read_pickle(cache_path)
            self.assertTrue(
                {
                    "query_date",
                    "source",
                    "schema_version",
                    "fetched_at",
                    "normalized_rows_sha256",
                    "records_json",
                }.issubset(cached.columns)
            )

    def test_report_membership_alone_does_not_prove_active_or_full_day_suspension(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "代码": "603001",
                    "名称": "奥康国际",
                    "停牌时间": "2026-07-13",
                    "停牌截止时间": "2026-07-13",
                    "停牌期限": "停牌一天",
                    "停牌原因": "未来事件",
                    "所属市场": "上交所主板",
                    "预计复牌时间": "2026-07-14",
                }
            ]
        )
        normalized = normalize_suspension_status_frame(raw)
        self.assertEqual(normalized.iloc[0]["suspension_start_date"], "2026-07-13")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = DataConfig(cache_dir=root / "cache")
            service = MarketDataService(config)
            service.provider.fetch_suspension_status = mock.Mock(
                return_value=(normalized, SUSPENSION_STATUS_SOURCE)
            )
            self.assertIsNone(service.resolve_suspension_on_date("603001", SIGNAL_DATE))


class SuspensionAuditTests(unittest.TestCase):
    def _excluded_row(self, *, from_cache: bool = False) -> dict[str, object]:
        service = _CoverageFailureService(
            _coverage_quality(), proof=_suspension_proof(from_cache=from_cache)
        )
        _, rows = build_signals_for_pool(service, _pool(), SIGNAL_DATE, days=10)
        return rows[0]

    def test_membership_records_raw_inclusion_and_signal_exclusion_with_evidence(self) -> None:
        excluded = self._excluded_row()
        raw = _annotate_raw_source_exclusions(
            _standardize_raw_source_pool(_pool(), SIGNAL_DATE), [excluded]
        )
        signals = _standardize_signal_pool(pd.DataFrame(), SIGNAL_DATE, SIGNAL_DATE)
        candidates = pd.DataFrame(columns=HISTORY_CANDIDATE_COLUMNS)
        membership = _build_history_universe_membership(
            SIGNAL_DATE,
            SIGNAL_DATE,
            raw,
            signals,
            candidates,
            quality_rows=[excluded],
        )
        raw_row = membership[membership["stage"].eq("raw_source_pool")].iloc[0]
        signal_row = membership[membership["stage"].eq("signal_pool")].iloc[0]
        self.assertTrue(bool(raw_row["included_bool"]))
        self.assertFalse(bool(signal_row["included_bool"]))
        self.assertEqual(signal_row["exclusion_reason"], "suspended_on_signal_date")
        self.assertEqual(signal_row["latest_daily_date"], STALE_DATE)
        self.assertEqual(signal_row["latest_minute_trade_date"], STALE_DATE)
        self.assertEqual(signal_row["proof_method"], "eastmoney_suspension_status_v1")
        self.assertEqual(
            json.loads(signal_row["exclusion_evidence_json"])["suspension_start_date"],
            "2026-06-25",
        )

    def test_exclusion_evidence_hash_is_order_and_cache_hit_independent(self) -> None:
        first = self._excluded_row(from_cache=False)
        second = self._excluded_row(from_cache=True)
        second = {**second, "code": "603002"}
        evidence = json.loads(str(second["exclusion_evidence_json"]))
        evidence["code"] = "603002"
        second["exclusion_evidence_json"] = json.dumps(
            evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        forward = _quality_exclusion_metadata([first, second])[
            "quality_exclusion_evidence_sha256"
        ]
        reverse = _quality_exclusion_metadata([second, first])[
            "quality_exclusion_evidence_sha256"
        ]
        self.assertEqual(forward, reverse)

    def test_final_requested_signal_date_invariant_remains_hard_failure(self) -> None:
        mismatched = pd.DataFrame(
            [
                {
                    "requested_signal_date": SIGNAL_DATE,
                    "signal_date": STALE_DATE,
                    "code": "603001",
                }
            ]
        )
        with self.assertRaisesRegex(UniverseAuditError, "signal_date_mismatch"):
            require_requested_signal_date_match(mismatched)


if __name__ == "__main__":
    unittest.main()

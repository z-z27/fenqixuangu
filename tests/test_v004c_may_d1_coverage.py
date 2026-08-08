# -*- coding: utf-8 -*-
"""v004c May D1 Candidate Coverage / Recovery — 测试

覆盖 (§15):
- June+July selector regression: 业务语义选择器在审计表上的 CANDIDATE 子集与
  冻结表 event_id 精确对齐 (rows / dates / 零差异 / 无重复); 差异被正确检出
- May selector 确定性 (两次运行同 rows / 同排序 / 同分类)
- Completeness 分类 fixtures: complete / missing cache / missing D1 date /
  missing bars / duplicate timestamp / bad first-last bar / invalid price /
  daily history 不足 / 停牌 proven 标签偏移
- Target 独立性: 翻转 May Target 不影响 universe / minute / daily / feature /
  recovery eligibility / fully 分类
- Recovery: raw 副本合并 (backup + before/after provenance + 幂等)
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_may_d1_coverage import (
    EXPECTED_BAR_COUNT,
    apply_raw_recovery,
    audit_candidate_coverage,
    audit_d1_daily_complete,
    audit_d1_minute_complete,
    audit_daily_label,
    build_coverage_summary,
    build_trading_calendar,
    canonical_5min_window_check,
    load_suspension_proofs,
    next_cal_date,
    read_candidate_audit,
    replay_completed_recoveries,
    reproduce_june_july_candidates,
    select_business_candidates,
    summarize_universe,
)

BASE = Path(__file__).resolve().parent.parent


def bar_times():
    """48-bar 5min 网格: 09:35..11:30 + 13:05..15:00。"""
    morning = [f"09:{m:02d}" for m in range(35, 60, 5)] + [f"10:{m:02d}" for m in range(0, 60, 5)] + \
              [f"11:{m:02d}" for m in range(0, 31, 5)]
    afternoon = [f"13:{m:02d}" for m in range(5, 60, 5)] + [f"14:{m:02d}" for m in range(0, 60, 5)] + ["15:00"]
    return morning + afternoon


def make_minute_frame(code: str, date: str, n_bars: int | None = None,
                      first_idx: int = 0, last_idx: int | None = None,
                      duplicate_timestamp: bool = False, bad_price_idx: int | None = None) -> pd.DataFrame:
    times = bar_times()
    last_idx = len(times) - 1 if last_idx is None else last_idx
    n_bars = (last_idx - first_idx + 1) if n_bars is None else n_bars
    idxs = list(range(first_idx, last_idx + 1))
    if n_bars != len(idxs):
        idxs = idxs[:n_bars]
    rows = []
    for i, ti in enumerate(idxs):
        rows.append({
            "datetime": f"{date} {times[ti]}:00",
            "trade_date": date,
            "time": f"{times[ti]}:00",
            "code": code, "market": "1",
            "open": 10.0 + i * 0.01, "high": 10.05 + i * 0.01,
            "low": 9.95 + i * 0.01, "close": 10.01 + i * 0.01,
            "volume": 1000 + i, "amount": 10000.0 + i,
            "pct_chg": 0.0, "change": 0.0, "amplitude": 0.0,
            "turnover_rate": 0.0, "source": "sina_5m", "adjust": "none",
        })
    frame = pd.DataFrame(rows)
    if duplicate_timestamp and len(frame) >= 2:
        frame.loc[1, "datetime"] = frame.loc[0, "datetime"]
        frame.loc[1, "time"] = frame.loc[0, "time"]
    if bad_price_idx is not None and bad_price_idx < len(frame):
        frame.loc[bad_price_idx, "close"] = np.nan
    return frame


def make_daily_frame(code: str, dates: list[str], prices_start: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "code": code, "market": "1",
        "open": prices_start, "high": prices_start * 1.02, "low": prices_start * 0.98,
        "close": prices_start * 1.01, "volume": 1_000_000, "amount": 1e7,
        "pct_chg": 0.0, "change": 0.0, "amplitude": 0.0,
        "turnover_rate": 0.0, "source": "tencent_daily",
    })


def make_audit_rows(rows: list[dict]) -> pd.DataFrame:
    cols = ["event_id", "code", "name", "signal_date", "break_date",
            "board_streak_before_break", "days_since_break",
            "candidate_status", "exclusion_reason", "data_quality_reason",
            "v01_candidate_status", "sample_role_v02"]
    for r in rows:
        for c in cols:
            r.setdefault(c, "")
    return pd.DataFrame(rows)[cols]


# 日历日期 (覆盖 5 月与 6/7 月样本的 d0..d3)
CAL_DATES = [f"2026-05-{d:02d}" for d in range(1, 31)] + \
            [f"2026-06-{d:02d}" for d in range(1, 8)] + \
            [f"2026-07-{d:02d}" for d in range(1, 8)]


class BaseFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="maycov_"))
        self.pool_dir = self.tmp / "limit_ups"
        self.daily_dir = self.tmp / "daily"
        self.minute_dir = self.tmp / "minute_5m"
        self.raw_dir = self.tmp / "raw_minute"
        self.susp_dir = self.tmp / "suspension_status"
        self.backup_dir = self.tmp / "backups"
        for d in (self.pool_dir, self.daily_dir, self.minute_dir, self.raw_dir,
                  self.susp_dir, self.backup_dir):
            d.mkdir(parents=True, exist_ok=True)
        # 统一日历: pool 覆盖全部日期
        pool = pd.DataFrame({"trade_date": CAL_DATES, "code": "000001",
                             "name": "x", "date": CAL_DATES, "close": 10.0})
        pool.to_pickle(self.pool_dir / "limit_ups.pkl")
        self.cal = build_trading_calendar(self.pool_dir, self.daily_dir)
        self.cal_idx = self.cal["cal_idx"]

    def write_daily(self, code: str, dates: list[str]) -> None:
        make_daily_frame(code, dates).to_pickle(self.daily_dir / f"{code}_daily.pkl")

    def write_minute(self, code: str, frame: pd.DataFrame) -> None:
        frame.to_pickle(self.minute_dir / f"{code}_5min.pkl")

    def write_raw(self, code: str, frames: list[pd.DataFrame]) -> None:
        pd.concat(frames, ignore_index=True).to_csv(self.raw_dir / f"{code}_5min.csv",
                                                    index=False, encoding="utf-8-sig")

    def write_suspension(self, code: str, start: str, end: str) -> None:
        records = [{"code": code, "suspension_start_date": start,
                    "suspension_end_date": end, "suspension_duration": "停牌一天"}]
        frame = pd.DataFrame({"records_json": [json.dumps(records)]})
        frame.to_pickle(self.susp_dir / f"{code}.pkl")


# ---------------------------------------------------------------------------
# 1. selector regression
# ---------------------------------------------------------------------------
class TestSelector(BaseFixture):
    def _full_audit(self) -> pd.DataFrame:
        rows = [
            # June+July 候选 (与 frozen 一致)
            {"event_id": "000001_2026-06-01", "code": "000001", "name": "A",
             "signal_date": "2026-06-01", "break_date": "2026-06-01",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "CANDIDATE"},
            {"event_id": "000002_2026-07-01", "code": "000002", "name": "B",
             "signal_date": "2026-07-01", "break_date": "2026-07-01",
             "board_streak_before_break": 3, "days_since_break": 0,
             "candidate_status": "CANDIDATE"},
            # May 候选
            {"event_id": "000003_2026-05-06", "code": "000003", "name": "M1",
             "signal_date": "2026-05-06", "break_date": "2026-05-06",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "CANDIDATE"},
            {"event_id": "000004_2026-05-07", "code": "000004", "name": "M2",
             "signal_date": "2026-05-07", "break_date": "2026-05-07",
             "board_streak_before_break": 3, "days_since_break": 0,
             "candidate_status": "QUALITY_FAILED", "exclusion_reason": "missing_minute_data"},
            {"event_id": "000005_2026-05-08", "code": "000005", "name": "M3",
             "signal_date": "2026-05-08", "break_date": "2026-05-08",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "LABEL_UNAVAILABLE", "exclusion_reason": "missing_future_label"},
            # 非候选: 1 板 / post / signal!=break
            {"event_id": "000006_2026-05-09", "code": "000006", "name": "S1",
             "signal_date": "2026-05-09", "break_date": "2026-05-09",
             "board_streak_before_break": 1, "days_since_break": 0,
             "candidate_status": "EXCLUDED", "exclusion_reason": "not_two_or_three_board"},
            {"event_id": "000007_2026-05-06", "code": "000007", "name": "S2",
             "signal_date": "2026-05-06", "break_date": "2026-05-06",
             "board_streak_before_break": 2, "days_since_break": 1,
             "candidate_status": "EXCLUDED", "exclusion_reason": "not_d1"},
            {"event_id": "000008_2026-05-10", "code": "000008", "name": "S3",
             "signal_date": "2026-05-10", "break_date": "2026-05-06",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "EXCLUDED", "exclusion_reason": "signal_not_break"},
        ]
        return make_audit_rows(rows)

    def test_june_july_reproduction_exact(self):
        audit = self._full_audit()
        frozen = make_audit_rows([
            {"event_id": "000001_2026-06-01", "code": "000001", "signal_date": "2026-06-01",
             "break_date": "2026-06-01", "board_streak_before_break": 2, "days_since_break": 0},
            {"event_id": "000002_2026-07-01", "code": "000002", "signal_date": "2026-07-01",
             "break_date": "2026-07-01", "board_streak_before_break": 3, "days_since_break": 0},
        ])
        r = reproduce_june_july_candidates(audit, frozen)
        self.assertEqual(r["business_rows"], 2)
        self.assertEqual(r["candidate_rows"], 2)
        self.assertEqual(r["frozen_rows"], 2)
        self.assertTrue(r["exact_event_id_match"])
        self.assertTrue(r["candidate_signal_date_match"])
        self.assertEqual(r["candidate_signal_dates"], 2)
        self.assertEqual(r["duplicate_event_ids_in_candidate"], 0)
        self.assertEqual(r["event_id_diff_frozen_minus_candidate"], [])
        self.assertEqual(r["event_id_diff_candidate_minus_frozen"], [])
        self.assertEqual(r["differing_rows"], 0)

    def test_mismatch_detected(self):
        audit = self._full_audit()
        frozen = make_audit_rows([
            {"event_id": "000001_2026-06-01", "code": "000001", "signal_date": "2026-06-01",
             "break_date": "2026-06-01", "board_streak_before_break": 2, "days_since_break": 0},
            {"event_id": "000099_2026-07-02", "code": "000099", "signal_date": "2026-07-02",
             "break_date": "2026-07-02", "board_streak_before_break": 3, "days_since_break": 0},
        ])
        r = reproduce_june_july_candidates(audit, frozen)
        self.assertFalse(r["exact_event_id_match"])
        self.assertEqual(r["event_id_diff_frozen_minus_candidate"], ["000099_2026-07-02"])
        self.assertEqual(r["event_id_diff_candidate_minus_frozen"], ["000002_2026-07-01"])
        self.assertEqual(r["frozen_rows"], 2)
        self.assertEqual(r["candidate_rows"], 2)

    def test_may_universe_and_stats(self):
        audit = self._full_audit()
        may = select_business_candidates(audit, "2026-05-06", "2026-05-31")
        self.assertEqual(len(may), 3)
        u = summarize_universe(may)
        self.assertEqual(u["rows"], 3)
        self.assertEqual(u["signal_dates"], 3)
        self.assertEqual(u["unique_stocks"], 3)
        self.assertEqual(u["board_streak_2_count"], 2)
        self.assertEqual(u["board_streak_3_count"], 1)
        self.assertEqual(u["candidate_status_counts"]["CANDIDATE"], 1)

    def test_may_selector_deterministic(self):
        audit = self._full_audit()
        a = select_business_candidates(audit, "2026-05-06", "2026-05-31")
        b = select_business_candidates(audit, "2026-05-06", "2026-05-31")
        self.assertTrue(a.equals(b))

    def test_signal_not_break_excluded(self):
        audit = self._full_audit()
        may = select_business_candidates(audit, "2026-05-06", "2026-05-31")
        self.assertNotIn("000008_2026-05-10", may["event_id"].tolist())

    def test_read_audit_zfill(self):
        path = self.tmp / "audit.csv"
        audit = self._full_audit()
        audit["code"] = audit["code"].astype(int).astype(str)
        audit.to_csv(path, index=False, encoding="utf-8-sig")
        loaded = read_candidate_audit(path)
        self.assertEqual(loaded["code"].iloc[0], "000001")


# ---------------------------------------------------------------------------
# 2. completeness fixtures
# ---------------------------------------------------------------------------
class TestCompleteness(BaseFixture):
    def _may_rows(self):
        return make_audit_rows([
            {"event_id": "000010_2026-05-06", "code": "000010", "name": "T1",
             "signal_date": "2026-05-06", "break_date": "2026-05-06",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "CANDIDATE"},
        ])

    def _daily_dates_full(self):
        return ["2026-04-01"] * 0 + [f"2026-04-{d:02d}" for d in range(1, 29)] + \
               [f"2026-05-{d:02d}" for d in range(1, 11)]

    def test_minute_complete(self):
        m = make_minute_frame("000010", "2026-05-06")
        out = audit_d1_minute_complete("000010", "2026-05-06", m)
        self.assertTrue(out["d1_minute_complete"])
        self.assertEqual(out["actual_bars"], EXPECTED_BAR_COUNT)
        self.assertEqual(out["first_timestamp"], "09:35")
        self.assertEqual(out["last_timestamp"], "15:00")
        self.assertEqual(out["missing_bars"], 0)
        self.assertEqual(out["duplicate_bars"], 0)
        self.assertEqual(out["invalid_price_volume"], 0)
        self.assertEqual(out["missing_reason"], "none")

    def test_missing_cache(self):
        out = audit_d1_minute_complete("000010", "2026-05-06", None)
        self.assertFalse(out["d1_minute_complete"])
        self.assertFalse(out["minute_cache_file_exists"])
        self.assertEqual(out["missing_reason"], "cache_missing")
        self.assertEqual(out["missing_bars"], EXPECTED_BAR_COUNT)

    def test_missing_d1_date(self):
        m = make_minute_frame("000010", "2026-05-07")
        out = audit_d1_minute_complete("000010", "2026-05-06", m)
        self.assertFalse(out["d1_minute_complete"])
        self.assertTrue(out["minute_cache_file_exists"])
        self.assertFalse(out["d1_date_exists"])
        self.assertEqual(out["missing_reason"], "d1_date_missing")

    def test_missing_bars(self):
        m = make_minute_frame("000010", "2026-05-06", n_bars=36)
        out = audit_d1_minute_complete("000010", "2026-05-06", m)
        self.assertFalse(out["d1_minute_complete"])
        self.assertEqual(out["actual_bars"], 36)
        self.assertEqual(out["missing_bars"], 12)
        self.assertEqual(out["missing_reason"], "bar_grid_incomplete")

    def test_duplicate_timestamp(self):
        m = make_minute_frame("000010", "2026-05-06", duplicate_timestamp=True)
        out = audit_d1_minute_complete("000010", "2026-05-06", m)
        # 正式规则 (v0.2 day_audit) 只查 48 bar / 首 09:35 / 末 15:00;
        # duplicate 作为独立质量列记录, 不改变完整判定
        self.assertGreaterEqual(out["duplicate_bars"], 1)
        self.assertTrue(out["d1_minute_complete"])

    def test_bad_first_last_bar(self):
        m = make_minute_frame("000010", "2026-05-06")
        m.loc[0, "time"] = "09:40:00"
        m.loc[0, "datetime"] = "2026-05-06 09:40:00"
        m.loc[len(m) - 1, "time"] = "14:55:00"
        m.loc[len(m) - 1, "datetime"] = "2026-05-06 14:55:00"
        out = audit_d1_minute_complete("000010", "2026-05-06", m)
        self.assertFalse(out["d1_minute_complete"])
        self.assertEqual(out["actual_bars"], EXPECTED_BAR_COUNT)
        self.assertEqual(out["first_timestamp"], "09:40")
        self.assertEqual(out["last_timestamp"], "14:55")
        self.assertEqual(out["missing_reason"], "grid_violation")

    def test_invalid_price(self):
        m = make_minute_frame("000010", "2026-05-06", bad_price_idx=10)
        out = audit_d1_minute_complete("000010", "2026-05-06", m)
        # 与 v0.2 day_audit 一致: invalid 价格只记录, 不改变完整判定
        self.assertGreater(out["invalid_price_volume"], 0)
        self.assertTrue(out["d1_minute_complete"])

    def test_daily_complete(self):
        d = make_daily_frame("000010", self._daily_dates_full())
        out = audit_d1_daily_complete("000010", "2026-05-06", d)
        self.assertTrue(out["d1_daily_complete"])
        self.assertTrue(out["d1_date_exists"])
        self.assertTrue(out["d1_ohlc_valid"])
        self.assertGreaterEqual(out["prior_trade_days"], 20)

    def test_daily_insufficient_history(self):
        d = make_daily_frame("000010", ["2026-05-01", "2026-05-05", "2026-05-06"])
        out = audit_d1_daily_complete("000010", "2026-05-06", d)
        self.assertFalse(out["d1_daily_complete"])
        self.assertFalse(out["daily_history_sufficient"])
        self.assertTrue(out["d1_date_exists"])

    def test_daily_missing_date(self):
        d = make_daily_frame("000010", ["2026-05-01", "2026-05-07"])
        out = audit_d1_daily_complete("000010", "2026-05-06", d)
        self.assertFalse(out["d1_daily_complete"])
        self.assertFalse(out["d1_date_exists"])

    def test_label_expected_dates(self):
        d = make_daily_frame("000010", self._daily_dates_full())
        out = audit_daily_label("000010", "2026-05-06", self.cal_idx, {}, d)
        self.assertTrue(out["daily_label_complete"])
        self.assertEqual(out["label_d2_date"], "2026-05-07")
        self.assertEqual(out["label_d3_date"], "2026-05-08")
        self.assertEqual(out["daily_label_source_d2"], "expected_date_daily_cache")
        self.assertIsInstance(out["target7_daily_d2open_d3high"], bool)
        self.assertFalse(out["target7_daily_d2open_d3high"])  # 2% < 7%
        # 抬高 D3 高价 -> target 翻转为 True
        d2 = d.copy()
        d2.loc[d2["date"] == "2026-05-08", "high"] = d2.loc[d2["date"] == "2026-05-08", "high"] * 1.2
        out2 = audit_daily_label("000010", "2026-05-06", self.cal_idx, {}, d2)
        self.assertTrue(out2["target7_daily_d2open_d3high"])

    def test_label_suspension_resume(self):
        # 该股 daily 缺 05-07 (停牌) -> D2 偏移到复牌日 05-08 (与 v0.2.2 一致:
        # expected_d3=05-08 有 bar, 故 D3 保持 expected 不偏移)
        d = make_daily_frame("000010", ["2026-04-01", "2026-05-06", "2026-05-08", "2026-05-11"])
        self.write_suspension("000010", "2026-05-07", "2026-05-07")
        proofs = load_suspension_proofs(self.susp_dir)
        out = audit_daily_label("000010", "2026-05-06", self.cal_idx, proofs, d)
        self.assertEqual(out["label_d2_date"], "2026-05-08")
        self.assertEqual(out["daily_label_source_d2"], "suspension_proven_resume_date_daily_cache")
        self.assertEqual(out["label_d3_date"], "2026-05-08")
        self.assertEqual(out["daily_label_source_d3"], "expected_date_daily_cache")
        self.assertTrue(out["daily_label_complete"])

    def test_label_missing_d2_without_proof(self):
        d = make_daily_frame("000010", ["2026-04-01", "2026-05-06", "2026-05-08", "2026-05-11"])
        out = audit_daily_label("000010", "2026-05-06", self.cal_idx, {}, d)
        self.assertFalse(out["daily_label_complete"])
        self.assertIn("d2_open_invalid", out["daily_label_reason"])

    def test_next_cal_date_bounds(self):
        self.assertEqual(next_cal_date(self.cal_idx, "2026-05-06"), "2026-05-07")
        self.assertIsNone(next_cal_date(self.cal_idx, "2026-07-07"))
        self.assertIsNone(next_cal_date(self.cal_idx, "2030-01-01"))


# ---------------------------------------------------------------------------
# 3. 端到端 coverage 审计 + Target 独立性
# ---------------------------------------------------------------------------
class TestCoverageEndToEnd(BaseFixture):
    def setUp(self):
        super().setUp()
        rows = [
            {"event_id": "000010_2026-05-06", "code": "000010", "name": "F1",
             "signal_date": "2026-05-06", "break_date": "2026-05-06",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "CANDIDATE"},
            {"event_id": "000011_2026-05-07", "code": "000011", "name": "F2",
             "signal_date": "2026-05-07", "break_date": "2026-05-07",
             "board_streak_before_break": 3, "days_since_break": 0,
             "candidate_status": "QUALITY_FAILED", "exclusion_reason": "missing_minute_data"},
            {"event_id": "000012_2026-05-08", "code": "000012", "name": "F3",
             "signal_date": "2026-05-08", "break_date": "2026-05-08",
             "board_streak_before_break": 2, "days_since_break": 0,
             "candidate_status": "CANDIDATE"},
        ]
        self.audit = make_audit_rows(rows)
        self.may = select_business_candidates(self.audit, "2026-05-06", "2026-05-31")
        # F1: 完全完整; F2: 分钟缺 (cache 无文件); F3: 分钟缺 bars (36)
        apr = [f"2026-04-{d:02d}" for d in range(1, 26)]
        self.f1_dates = apr + ["2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08"]
        self.write_daily("000010", self.f1_dates)
        self.write_daily("000011", apr + ["2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-11"])
        self.write_daily("000012", apr + ["2026-05-05", "2026-05-06", "2026-05-07",
                                          "2026-05-08", "2026-05-11", "2026-05-12"])
        self.write_minute("000010", make_minute_frame("000010", "2026-05-06"))
        # F3 部分 bar 且首 bar 异常
        self.write_minute("000012", make_minute_frame("000012", "2026-05-08", n_bars=36))

    def test_audit_classification(self):
        df = audit_candidate_coverage(self.may, cal_idx=self.cal_idx, proofs={},
                                      minute_dir=self.minute_dir, daily_dir=self.daily_dir,
                                      raw_dir=self.raw_dir)
        by_id = df.set_index("event_id")
        self.assertTrue(by_id.at["000010_2026-05-06", "d1_daily_complete"])
        self.assertTrue(by_id.at["000010_2026-05-06", "d1_minute_complete"])
        self.assertTrue(by_id.at["000010_2026-05-06", "daily_label_complete"])
        self.assertTrue(by_id.at["000010_2026-05-06", "d1_feature_complete"])
        self.assertTrue(by_id.at["000010_2026-05-06", "fully_complete"])
        self.assertEqual(by_id.at["000011_2026-05-07", "missing_reason"], "cache_missing")
        self.assertFalse(by_id.at["000011_2026-05-07", "d1_minute_complete"])
        self.assertEqual(by_id.at["000012_2026-05-08", "missing_reason"], "bar_grid_incomplete")
        self.assertEqual(by_id.at["000012_2026-05-08", "missing_bars"], 12)
        summary = build_coverage_summary(df)
        self.assertEqual(summary["per_layer"]["MINUTE_COMPLETE"], 1)
        self.assertEqual(summary["per_layer"]["FULLY_COMPLETE"], 1)
        self.assertEqual(summary["universe_rows"], 3)

    def test_daily_complete_but_minute_missing(self):
        df = audit_candidate_coverage(self.may, cal_idx=self.cal_idx, proofs={},
                                      minute_dir=self.minute_dir, daily_dir=self.daily_dir,
                                      raw_dir=self.raw_dir)
        sel = df[df["d1_daily_complete"].astype(bool) & ~df["d1_minute_complete"].astype(bool)]
        self.assertEqual(len(sel), 2)  # F2, F3 日线完整但分钟缺

    def test_target_independence(self):
        """翻转 May Target (改 D3 高价) 不影响 universe / minute / daily /
        feature / recovery eligibility / fully 分类。"""
        d = make_daily_frame("000010", self.f1_dates)
        base = audit_candidate_coverage(self.may, cal_idx=self.cal_idx, proofs={},
                                        minute_dir=self.minute_dir, daily_dir=self.daily_dir,
                                        raw_dir=self.raw_dir)
        # 翻转: 把 d3_high 抬高 50% (F1 的 target 从 ~0.02 翻到 >=0.07)
        d2 = d.copy()
        d2.loc[d2["date"] == "2026-05-08", "high"] = d2.loc[d2["date"] == "2026-05-08", "high"] * 1.5
        d2.to_pickle(self.daily_dir / "000010_daily.pkl")
        flipped = audit_candidate_coverage(self.may, cal_idx=self.cal_idx, proofs={},
                                           minute_dir=self.minute_dir, daily_dir=self.daily_dir,
                                           raw_dir=self.raw_dir)
        identity_cols = [c for c in base.columns
                         if c not in ("target7_daily_d2open_d3high", "d2_open_daily", "d3_high_daily",
                                      "d3_close_daily", "daily_d2open_to_d3high_return",
                                      "daily_d2open_to_d3close_return", "daily_label_reason")]
        for col in identity_cols:
            self.assertEqual(base[col].astype(str).tolist(), flipped[col].astype(str).tolist(),
                             f"Target 独立性破坏: 列 {col}")
        self.assertNotEqual(
            base["target7_daily_d2open_d3high"].astype(bool).tolist(),
            flipped["target7_daily_d2open_d3high"].astype(bool).tolist(),
            "测试自身必须翻转 target")


# ---------------------------------------------------------------------------
# 4. recovery
# ---------------------------------------------------------------------------
class TestRecovery(BaseFixture):
    def setUp(self):
        super().setUp()
        self.code = "000020"
        self.date = "2026-05-06"
        # cache 只有 6 月数据 (无 05-06)
        self.write_minute(self.code, make_minute_frame(self.code, "2026-06-01"))
        # raw 含 05-06 完整 + 06-01
        self.write_raw(self.code, [
            make_minute_frame(self.code, self.date),
            make_minute_frame(self.code, "2026-06-01"),
        ])
        self.cache_path = self.minute_dir / f"{self.code}_5min.pkl"

    def test_apply_recovery(self):
        rec = apply_raw_recovery(self.code, self.date, self.raw_dir,
                                 self.cache_path, self.backup_dir)
        self.assertEqual(rec["rows_added"], EXPECTED_BAR_COUNT)
        self.assertEqual(rec["before"]["rows"], EXPECTED_BAR_COUNT)
        self.assertFalse(rec["before"]["d1_minute_complete"])
        self.assertTrue(rec["after"]["d1_minute_complete"])
        self.assertEqual(rec["after"]["rows"], EXPECTED_BAR_COUNT * 2)
        self.assertTrue(rec["raw_provenance"]["exists"])
        self.assertIn("sha256", rec["raw_provenance"])
        self.assertTrue(rec["before"]["exists"])
        self.assertTrue(rec["after"]["exists"])
        # 备份存在
        self.assertTrue(Path(rec["backup_path"]).exists())
        # cache 现在包含目标日
        reloaded = pd.read_pickle(self.cache_path)
        self.assertIn(self.date, reloaded["trade_date"].astype(str).tolist())
        # 幂等: 再执行 rows_added=0
        rec2 = apply_raw_recovery(self.code, self.date, self.raw_dir,
                                  self.cache_path, self.backup_dir)
        self.assertEqual(rec2["rows_added"], 0)

    def test_raw_missing_target_date_raises(self):
        self.write_raw(self.code, [make_minute_frame(self.code, "2026-06-01")])
        with self.assertRaises(ValueError):
            apply_raw_recovery(self.code, self.date, self.raw_dir,
                               self.cache_path, self.backup_dir)

    def test_replay_completed_recovery(self):
        rec = apply_raw_recovery(self.code, self.date, self.raw_dir,
                                 self.cache_path, self.backup_dir)
        self.assertEqual(rec["rows_added"], EXPECTED_BAR_COUNT)
        universe = pd.DataFrame([{
            "event_id": f"{self.code}_{self.date}", "code": self.code,
            "signal_date": self.date,
        }])
        replayed = replay_completed_recoveries(
            universe, backup_dir=self.backup_dir, minute_dir=self.minute_dir,
            raw_dir=self.raw_dir)
        self.assertEqual(len(replayed), 1)
        r = replayed[0]
        self.assertEqual(r["event_id"], f"{self.code}_{self.date}")
        self.assertEqual(r["rows_added"], EXPECTED_BAR_COUNT)
        self.assertEqual(r["before"]["rows"], EXPECTED_BAR_COUNT)
        self.assertEqual(r["after"]["rows"], EXPECTED_BAR_COUNT * 2)
        self.assertFalse(r["before"]["d1_minute_complete"])
        self.assertTrue(r["after"]["d1_minute_complete"])
        self.assertTrue(Path(r["backup_path"]).exists())
        self.assertIn("sha256", r["raw_provenance"])
        # 重放必须幂等
        self.assertEqual(len(replay_completed_recoveries(
            universe, backup_dir=self.backup_dir, minute_dir=self.minute_dir,
            raw_dir=self.raw_dir)), 1)

    def test_replay_skips_stale_backup(self):
        # 无合并发生的残留 backup (rows 与 cache 相同) -> 跳过
        self.write_minute(self.code, make_minute_frame(self.code, "2026-06-01"))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.cache_path,
                     self.backup_dir / f"{self.code}_5min.before_may_recovery.pkl")
        universe = pd.DataFrame([{
            "event_id": f"{self.code}_{self.date}", "code": self.code,
            "signal_date": self.date,
        }])
        replayed = replay_completed_recoveries(
            universe, backup_dir=self.backup_dir, minute_dir=self.minute_dir,
            raw_dir=self.raw_dir)
        self.assertEqual(replayed, [])


# ---------------------------------------------------------------------------
# 5. canonical 窗口检查 (mock, 只读)
# ---------------------------------------------------------------------------
class TestCanonicalWindow(unittest.TestCase):
    def test_window_reported(self):
        frame = make_minute_frame("000980", "2026-06-10", n_bars=10)
        frame = pd.concat([frame, make_minute_frame("000980", "2026-08-07", n_bars=10)],
                          ignore_index=True)

        def fake_fetch(code, start, end, adjust="none"):
            return frame.copy(), "sina_5m"

        out = canonical_5min_window_check(["000980"], fake_fetch,
                                          start_datetime="2026-05-06 09:30:00",
                                          end_datetime="2026-05-31 15:00:00")
        self.assertEqual(out["results"][0]["min_trade_date"], "2026-06-10")
        self.assertEqual(out["results"][0]["max_trade_date"], "2026-08-07")
        self.assertTrue(out["results"][0]["ok"])

    def test_fetch_failure_reported(self):
        def bad_fetch(code, start, end, adjust="none"):
            raise RuntimeError("boom")

        out = canonical_5min_window_check(["000980"], bad_fetch,
                                          start_datetime="2026-05-06 09:30:00",
                                          end_datetime="2026-05-31 15:00:00")
        self.assertFalse(out["results"][0]["ok"])
        self.assertIn("boom", out["results"][0]["error"])


if __name__ == "__main__":
    unittest.main()

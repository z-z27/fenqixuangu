# -*- coding: utf-8 -*-
"""v004c Pairwise v1 Feature Contract v002 — Pool Correctness Fix 测试。

覆盖任务要求:
- §三十七 rank 完整池测试: 池 A/B/C/D, 候选只有 A, volume B>C>A>D -> A rank=3;
- §三十八 pool lookback 缺 1 天 -> recent_pool_appearance_count_20d = NA + SOURCE_MISSING;
- §三十九 完整窗口计数 = 3;
- §四十 last_board_day_in_pool: 完整池不在 -> False; 池不完整 -> NA/SOURCE_MISSING;
- §四十一 603065 一字板 -> d1_close_location / break_upper/lower_shadow_ratio
  仍为 STRUCTURAL_MISSING (不受本任务影响);
- §四十二 high_zone formula metadata 必须含 typical_price=(high+low+close)/3,
  不能再写 bar close >= threshold;
- 契约完整性 / universe 身份 / pairwise eligibility / 三类 missing 审计。

约定: importlib 加载 v002 builder (与 v001 测试一致)。
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "v004c_pairwise_v1_feature_contract_builder_v002",
    BASE / "tools" / "build_v004c_pairwise_v1_feature_contract_v002.py")
b = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(b)

from src.v004c_pairwise_feature_contract_v002 import (  # noqa: E402
    CONTRACT_FEATURE_NAMES,
    FEATURE_CONTRACT,
    PAIRWISE_ELIGIBILITY_COLUMNS,
    SOURCE_MISSING,
    assert_contract_integrity_v002,
)

OUT_DIR = b.OUT_DIR
DEV_CSV = OUT_DIR / b.DEV_CSV_NAME

# ---------------------------------------------------------------------------
# 假数据工具
# ---------------------------------------------------------------------------
def make_row(code="000001", d1="2026-05-06", streak=2):
    return {"code": code, "signal_date": d1, "board_streak_before_break": streak}


def make_pool(codes, consecutive=1):
    return pd.DataFrame({"code": [c for c in codes],
                         "consecutive_limit_up_count": [consecutive] * len(codes)})


def make_trade_dates(n=20, start="2026-04-03"):
    """连续假交易日 (不含周末), 共 n 天。"""
    from datetime import date, timedelta
    out = []
    d = date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class PoolRankFullDenominatorTestCase(unittest.TestCase):
    """§三十七: rank 必须在完整涨停池上计算, 而非候选子集。"""

    def test_a_rank_3_in_full_pool(self):
        dates = make_trade_dates(20)
        d0 = dates[-2]
        d1 = dates[-1]
        # 完整池: A/B/C/D; volume: B > C > A > D -> rank: B=1 C=2 A=3 D=4
        pool = make_pool(["000001", "000002", "000003", "000004"])
        # 候选事件只有 A=000001 (旧实现会只在 {A} 上排名 -> A rank=1)
        row = make_row(code="000001", d1=d1)
        values, missing, audit = b.compute_pool_features(
            row, dates,
            {d0: pool, d1: make_pool([])},
            {d: True for d in dates},
            {d0: {"000001": 50.0, "000002": 100.0, "000003": 90.0, "000004": 10.0}},
        )
        self.assertEqual(values["board_day_volume_rank"], 3.0)
        self.assertIsNone(missing.get("board_day_volume_rank"))
        self.assertTrue(audit["rank_denominator_complete"])
        self.assertEqual(audit["pool_member_count"], 4)

    def test_missing_member_volume_is_source_missing(self):
        dates = make_trade_dates(20)
        d0, d1 = dates[-2], dates[-1]
        pool = make_pool(["000001", "000002", "000003"])
        # C=000003 无 daily volume -> rank denominator 不完整 -> NA + SOURCE_MISSING
        values, missing, audit = b.compute_pool_features(
            make_row("000001", d1), dates,
            {d0: pool, d1: make_pool([])},
            {d: True for d in dates},
            {d0: {"000001": 50.0, "000002": 100.0}},
        )
        self.assertIsNone(values["board_day_volume_rank"])
        self.assertEqual(missing.get("board_day_volume_rank"), SOURCE_MISSING)
        self.assertFalse(audit["rank_denominator_complete"])
        self.assertEqual(audit["members_missing_daily_volume"], 1)


class PoolLookbackIncompleteWindowTestCase(unittest.TestCase):
    """§三十八: 20 交易日窗口缺 1 天 -> 20d 计数 = NA + SOURCE_MISSING (禁止下界)。"""

    def _setup(self):
        dates = make_trade_dates(20)
        d1 = dates[-1]
        pool_by_date = {d: make_pool(["000001"]) for d in dates}
        complete = {d: True for d in dates}
        missing_date = dates[10]
        del pool_by_date[missing_date]
        complete[missing_date] = False
        return dates, d1, pool_by_date, complete

    def test_missing_one_day_na_source(self):
        dates, d1, pool_by_date, complete = self._setup()
        values, missing, _ = b.compute_pool_features(
            make_row("000001", d1), dates, pool_by_date, complete, {})
        self.assertIsNone(values["recent_pool_appearance_count_20d"])
        self.assertIsNone(values["recent_pool_appearance_count_10d"])
        self.assertEqual(missing.get("recent_pool_appearance_count_20d"), SOURCE_MISSING)
        self.assertEqual(missing.get("recent_pool_appearance_count_10d"), SOURCE_MISSING)


class PoolCompleteWindowCountTestCase(unittest.TestCase):
    """§三十九: 完整 20 日窗口, 股票出现 3 天 -> 计数 = 3。"""

    def test_count_3_in_complete_window(self):
        dates = make_trade_dates(20)
        d1 = dates[-1]
        appear = {dates[0], dates[5], dates[12]}
        pool_by_date = {d: (make_pool(["000001"]) if d in appear else make_pool([]))
                        for d in dates}
        complete = {d: True for d in dates}
        values, missing, _ = b.compute_pool_features(
            make_row("000001", d1), dates, pool_by_date, complete, {})
        self.assertEqual(values["recent_pool_appearance_count_20d"], 3)
        self.assertEqual(values["recent_pool_appearance_count_10d"],
                         sum(1 for d in dates[-10:] if d in appear))
        # 完整窗口 -> count 特征不缺失 (rank/pcc 因测试未提供 d0 volume 可允许 source)
        self.assertNotIn("recent_pool_appearance_count_20d", missing)
        self.assertNotIn("recent_pool_appearance_count_10d", missing)


class LastBoardDayInPoolSemanticsTestCase(unittest.TestCase):
    """§四十: 完整池不在 -> False; 池不完整 -> NA/SOURCE_MISSING。"""

    def test_complete_pool_absent_is_false(self):
        dates = make_trade_dates(20)
        d0, d1 = dates[-2], dates[-1]
        values, missing, _ = b.compute_pool_features(
            make_row("000001", d1), dates,
            {d0: make_pool(["000002", "000003"]), d1: make_pool([])},
            {d: True for d in dates}, {})
        self.assertFalse(values["last_board_day_in_pool"])
        self.assertIsNone(missing.get("last_board_day_in_pool"))

    def test_incomplete_pool_na_source(self):
        dates = make_trade_dates(20)
        d0, d1 = dates[-2], dates[-1]
        complete = {d: True for d in dates}
        complete[d0] = False
        values, missing, _ = b.compute_pool_features(
            make_row("000001", d1), dates, {}, complete, {})
        self.assertIsNone(values["last_board_day_in_pool"])
        self.assertIsNone(values["pool_consecutive_count_last_board"])
        self.assertEqual(missing.get("last_board_day_in_pool"), SOURCE_MISSING)
        self.assertEqual(missing.get("pool_consecutive_count_last_board"), SOURCE_MISSING)


class PoolConsecutiveSemanticsTestCase(unittest.TestCase):
    """pool_consecutive_count_last_board: 池完整且有记录 -> 实际值。"""

    def test_value_when_pool_record_exists(self):
        dates = make_trade_dates(20)
        d0, d1 = dates[-2], dates[-1]
        pool = pd.DataFrame({"code": ["000001"], "consecutive_limit_up_count": [3]})
        values, missing, _ = b.compute_pool_features(
            make_row("000001", d1), dates,
            {d0: pool, d1: make_pool([])},
            {d: True for d in dates}, {})
        self.assertEqual(values["pool_consecutive_count_last_board"], 3.0)
        self.assertEqual(missing, {})


class ContractFormulaMetadataTestCase(unittest.TestCase):
    """§四十二: high_zone 公式必须含 typical_price=(high+low+close)/3。"""

    def test_high_zone_formulas_corrected(self):
        assert_contract_integrity_v002()
        for name in ("high_zone_volume_ratio", "high_zone_amount_ratio"):
            c = next(x for x in FEATURE_CONTRACT if x["feature_name"] == name)
            self.assertIn("typical_price", c["formula"])
            self.assertIn("(high + low + close) / 3", c["formula"])
            self.assertNotIn("bar close >=", c["formula"])

    def test_contract_feature_set_unchanged(self):
        self.assertEqual(len(CONTRACT_FEATURE_NAMES), 53)
        self.assertEqual(len(set(CONTRACT_FEATURE_NAMES)), 53)

    def test_pool_missing_semantics_source(self):
        for name in ("recent_pool_appearance_count_10d",
                     "recent_pool_appearance_count_20d",
                     "board_day_volume_rank",
                     "break_day_in_pool",
                     "last_board_day_in_pool",
                     "pool_consecutive_count_last_board"):
            c = next(x for x in FEATURE_CONTRACT if x["feature_name"] == name)
            self.assertIn(SOURCE_MISSING, c["missing_semantics"], name)


class RealOutputRegressionTestCase(unittest.TestCase):
    """对 v002 builder 真实输出做回归 (builder 已运行后)。"""

    @classmethod
    def setUpClass(cls):
        if not DEV_CSV.exists():
            raise unittest.SkipTest("v002 output not built yet")
        cls.dev = pd.read_csv(DEV_CSV, encoding="utf-8-sig", dtype={"code": str})

    def test_universe_identity(self):
        v002 = pd.read_csv(b.V002_CSV, encoding="utf-8-sig", dtype={"code": str})
        v002["code"] = v002["code"].astype(str).str.zfill(6)
        for col in ("event_id", "code", "signal_date", "break_date",
                    "board_streak_before_break", "source_window"):
            self.assertEqual(self.dev[col].astype(str).tolist(),
                             v002[col].astype(str).tolist(), col)

    def test_universe_counts(self):
        self.assertEqual(len(self.dev), 319)
        self.assertEqual(self.dev["signal_date"].nunique(), 39)

    def test_pairwise_eligibility_columns(self):
        for col in PAIRWISE_ELIGIBILITY_COLUMNS:
            self.assertIn(col, self.dev.columns)
        # eligible 是 feature_source 的子集
        self.assertTrue(
            (self.dev["pairwise_v1_training_eligible"] <=
             self.dev["pairwise_v1_feature_source_complete"]).all())

    def test_unexpected_missing_zero(self):
        self.assertEqual(int(self.dev["unexpected_missing_count"].sum()), 0)

    def test_603065_structural_unchanged(self):
        """§四十一: 603065 一字板数学退化仍 STRUCTURAL。"""
        row = self.dev[self.dev["code"] == "603065"]
        self.assertEqual(len(row), 1)
        r = row.iloc[0]
        for f in ("d1_close_location", "break_upper_shadow_ratio",
                  "break_lower_shadow_ratio"):
            self.assertTrue(pd.isna(r[f]), f)
        self.assertGreaterEqual(r["structural_missing_count"], 3)
        self.assertEqual(r["source_missing_count"], 0)
        self.assertEqual(r["unexpected_missing_count"], 0)

    def test_high_zone_contract_csv(self):
        ctr = pd.read_csv(OUT_DIR / b.CONTRACT_CSV_NAME, encoding="utf-8-sig")
        hz = ctr[ctr["feature_name"].isin(
            ("high_zone_volume_ratio", "high_zone_amount_ratio"))]
        self.assertEqual(len(hz), 2)
        for _, r in hz.iterrows():
            self.assertIn("typical_price", r["formula"])
            self.assertNotIn("bar close >=", r["formula"])

    def test_rank_audit_exists(self):
        aud = pd.read_csv(OUT_DIR / b.RANK_AUDIT_CSV_NAME, encoding="utf-8-sig")
        self.assertIn("rank_denominator_complete", aud.columns)
        self.assertIn("pool_member_count", aud.columns)


if __name__ == "__main__":
    unittest.main()

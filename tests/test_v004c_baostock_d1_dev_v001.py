# -*- coding: utf-8 -*-
"""v004c BaoStock D1 dev rebuild (May/June) — 构建工具测试

覆盖 (§15/§26/§35/§41 构建契约):
- load_universe: May 146 全量 / June 冻结按 06-01..06-30 过滤 / identity 规则
  (signal_date==break_date) / 无重复 event_id
- compute_daily_ma_state: 官方 rolling MA / close_to_ma 系列 / slope /
  consecutive days below / true reclaim / break_open_return (prev_close 语义)
- build_event_row: 48-bar 完整 -> 22 minute features 全非空; provenance 静态标签;
  final_x_data_complete; 确定性 (两次运行逐字段一致)
- write_outputs: 两次输出 byte-identical (确定性构建契约)
- lineage CSV: 22 minute + 20 daily/MA + 3 recent_7d = 45 行
- run_june_crosscheck: 同值 1.0 匹配 / 扰动被检出
- _persist_daily_cache 往返 (load_daily_frame 可读回)
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.cache import Baostock5mCache
from src.data_sources import (
    BAOSTOCK_5M_SOURCE,
    NORMALIZED_5M_INTERVAL,
    normalize_baostock_5m_frame,
    validate_normalized_5m_frame,
)
from src.v004c_may_d1_coverage import EXPECTED_BAR_COUNT, load_daily_frame
from src.v004c_minute_features import MINUTE_DEPENDENT_FEATURES
from tests.test_v004c_may_d1_coverage import bar_times

BASE = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "v004c_baostock_d1_dev_builder",
    BASE / "tools" / "build_v004c_baostock_d1_dev_v001.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

FIELDS = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]


def make_bao_day_frame(code: str = "600000", date: str = "2026-05-06") -> pd.DataFrame:
    """48-bar 标准日 (09:35..15:00) 的 baostock 原始行 -> normalized frame。"""
    rows = []
    no_dash = date.replace("-", "")
    for i, t in enumerate(bar_times()):
        hhmm = t.replace(":", "")
        price = 10.0 + i * 0.01
        rows.append([
            date, f"{no_dash}{hhmm}00{i:03d}", f"sh.{code}",
            f"{price:.2f}", f"{price + 0.05:.2f}", f"{price - 0.05:.2f}",
            f"{price + 0.01:.2f}", str(1000 + i), str(10000.0 + i),
        ])
    raw = pd.DataFrame(rows, columns=FIELDS)
    return normalize_baostock_5m_frame(raw, code, BAOSTOCK_5M_SOURCE, "none")


def make_daily_frame(code: str = "600000", n_prior: int = 29,
                     d1_date: str = "2026-05-06") -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(d1_date), periods=n_prior + 1)
    closes = 10.0 + np.arange(n_prior + 1) * 0.1
    rows = []
    for i, d in enumerate(dates):
        c = closes[i]
        rows.append({
            "code": code,
            "date": d.strftime("%Y-%m-%d"),
            "open": round(c * 0.99, 4),
            "high": round(c * 1.02, 4),
            "low": round(c * 0.98, 4),
            "close": round(c, 4),
        })
    return pd.DataFrame(rows)


def make_event_series(code: str = "600000", signal_date: str = "2026-05-06",
                      window: str = "may") -> pd.Series:
    return pd.Series({
        "event_id": f"{code}_{signal_date}",
        "code": code,
        "name": "测试",
        "signal_date": signal_date,
        "break_date": signal_date,
        "board_streak_before_break": 3,
        "source_window": window,
        "target7_daily_d2open_d3high": True,
        "tail_loss_daily_5pct": False,
        "daily_label_complete": True,
        "target_training_eligible": True,
        "label_d2_date": "2026-05-07",
        "label_d3_date": "2026-05-08",
    })


class BuilderUnitTestCase(unittest.TestCase):
    def test_load_universe_real_files(self):
        universe = builder.load_universe(builder.MAY_CSV, builder.JUNE_CSV)
        may = universe[universe["source_window"] == "may"]
        june = universe[universe["source_window"] == "june"]
        self.assertEqual(len(may), 146)
        self.assertGreaterEqual(len(june), 170)
        self.assertEqual(len(universe), len(may) + len(june))
        self.assertTrue((universe["signal_date"] == universe["break_date"]).all())
        self.assertEqual(universe["event_id"].nunique(), len(universe))
        # June 仅含 06-01..06-30
        self.assertTrue(june["signal_date"].between("2026-06-01", "2026-06-30").all())
        # May 无训练资格列 (frozen chain 无 May rows) -> 由 candidates 提供
        self.assertFalse(june["target_training_eligible"].isna().all())

    def test_compute_daily_ma_state_official_formulas(self):
        daily = make_daily_frame(n_prior=29, d1_date="2026-05-06")
        state = builder.compute_daily_ma_state(daily, "600000", "2026-05-06")
        closes = np.array(daily["close"], dtype=float)
        i = 29
        self.assertAlmostEqual(state["d1_close"], closes[i])
        ma5 = closes[i - 4:i + 1].mean()
        ma10 = closes[i - 9:i + 1].mean()
        ma20 = closes[i - 19:i + 1].mean()
        self.assertAlmostEqual(state["d1_ma5"], ma5)
        self.assertAlmostEqual(state["d1_ma10"], ma10)
        self.assertAlmostEqual(state["d1_ma20"], ma20)
        self.assertAlmostEqual(state["d1_close_to_ma5_raw"], closes[i] / ma5 - 1.0)
        self.assertAlmostEqual(state["d1_ma5_slope"], ma5 / closes[i - 5:i].mean() - 1.0)
        # break_open_return = d1_open / prev_close - 1
        self.assertAlmostEqual(state["break_open_return"],
                               state["d1_open"] / closes[i - 1] - 1.0)
        self.assertAlmostEqual(state["prev_close_d1"], closes[i - 1])
        # 连续低于 MA5: closes 单调上升 -> 0 天
        self.assertEqual(state["consecutive_days_below_ma5"], 0)
        # low = close*0.98 < ma5 且 close > ma5 -> true reclaim
        self.assertEqual(state["d1_true_reclaim_ma5"], 1)
        self.assertEqual(state["d1_close_above_ma5"], 1)

    def test_build_event_row_full_48bar(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            frame = make_bao_day_frame()
            cache.write("600000", frame, meta=None)
            daily = make_daily_frame()
            event = make_event_series()
            row = builder.build_event_row(event, cache, daily, {}, {})
        self.assertTrue(row["d1_minute_complete"])
        self.assertEqual(row["actual_bars"], EXPECTED_BAR_COUNT)
        self.assertEqual(row["first_timestamp"], "09:35")
        self.assertEqual(row["last_timestamp"], "15:00")
        self.assertEqual(row["missing_bars"], 0)
        for name in MINUTE_DEPENDENT_FEATURES:
            self.assertIsNotNone(row[name], f"{name} should be computed")
        self.assertEqual(row["minute_feature_count"], len(MINUTE_DEPENDENT_FEATURES))
        self.assertTrue(row["minute_feature_complete"])
        self.assertTrue(row["daily_feature_complete"])
        self.assertTrue(row["final_x_data_complete"])
        # provenance 静态标签
        self.assertEqual(row["minute_source"], BAOSTOCK_5M_SOURCE)
        self.assertEqual(row["minute_adjustment"], "none")
        self.assertEqual(row["minute_interval"], NORMALIZED_5M_INTERVAL)
        self.assertEqual(row["daily_source"], "canonical_daily_cache")
        # recent_7d
        closes = np.array(make_daily_frame()["close"], dtype=float)
        self.assertAlmostEqual(row["recent_7d_cumulative_return"],
                               closes[-1] / closes[-7] - 1.0)

    def test_build_event_row_missing_day_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            frame = make_bao_day_frame()
            cache.write("600000", frame, meta=None)
            daily = make_daily_frame()
            event = make_event_series(signal_date="2026-05-07")  # cache 无此日
            row = builder.build_event_row(event, cache, daily, {}, {})
        self.assertFalse(row["d1_minute_complete"])
        self.assertEqual(row["minute_missing_reason"], "d1_date_missing")

    def test_build_event_row_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600000", make_bao_day_frame(), meta=None)
            daily = make_daily_frame()
            event = make_event_series()
            row1 = builder.build_event_row(event, cache, daily, {}, {})
            row2 = builder.build_event_row(event, cache, daily, {}, {})
        self.assertEqual(row1, row2)

    def test_lineage_rows_47(self):
        lineage = builder.render_lineage_csv()
        self.assertEqual(len(lineage), 22 + 22 + 3)
        self.assertEqual(len(lineage[lineage["feature_group"] == "minute"]), 22)
        self.assertEqual(len(lineage[lineage["feature_group"] == "daily_ma_state"]), 22)
        self.assertEqual(len(lineage[lineage["feature_group"] == "recent_7d"]), 3)

    def test_write_outputs_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600000", make_bao_day_frame(), meta=None)
            daily = make_daily_frame()
            rows = [builder.build_event_row(make_event_series(), cache, daily, {}, {})]
            universe = pd.DataFrame([{
                "event_id": "600000_2026-05-06", "code": "600000", "name": "测试",
                "signal_date": "2026-05-06", "break_date": "2026-05-06",
                "board_streak_before_break": 3, "source_window": "may",
            }])
            provenance = {"600000": {"cache": "reused", "rows": 48}}
            builder.OUT_DIR = Path(tmp) / "out"
            builder.write_outputs(rows, universe, provenance, {}, {}, {}, None)
            names = ("v004c_baostock_d1_dev_v001.csv",
                     "v004c_baostock_d1_minute_quality_v001.csv",
                     "v004c_baostock_d1_feature_lineage_v001.csv",
                     "v004c_baostock_d1_rebuild_review_v001.md",
                     "v004c_baostock_d1_fetch_provenance.json")
            first = {n: (Path(tmp) / "out" / n).read_bytes() for n in names}
            # 同目录第二次写入 -> 覆盖后仍 byte-identical (确定性重建契约)
            builder.write_outputs(rows, universe, provenance, {}, {}, {}, None)
            for n in names:
                self.assertEqual(
                    first[n], (Path(tmp) / "out" / n).read_bytes(),
                    f"{n} must be byte-identical on rerun")

    def test_june_crosscheck_matches_and_detects(self):
        dev = pd.DataFrame({
            "event_id": ["A", "B"], "source_window": ["june", "june"],
            "d1_close": [10.0, 11.0], "d1_ma5": [9.9, 10.9],
        })
        frozen = pd.DataFrame({
            "event_id": ["A", "B"], "d1_close": [10.0, 11.5], "d1_ma5": [9.9, 10.9],
        })
        cc = builder.run_june_crosscheck(dev, frozen)
        self.assertEqual(cc["d1_close"]["match_rate"], 0.5)
        self.assertEqual(cc["d1_ma5"]["match_rate"], 1.0)
        self.assertAlmostEqual(cc["d1_close"]["max_abs_diff"], 0.5)

    def test_persist_daily_roundtrip(self):
        daily = make_daily_frame(n_prior=4, d1_date="2026-05-08")
        with tempfile.TemporaryDirectory() as tmp:
            builder.DAILY_DIR = Path(tmp)
            builder._persist_daily_cache("600000", daily)
            back = load_daily_frame("600000", Path(tmp))
        self.assertIsNotNone(back)
        self.assertEqual(len(back), 5)
        self.assertEqual(back["date"].iloc[-1], "2026-05-08")
        self.assertAlmostEqual(float(back["close"].iloc[-1]),
                               float(daily["close"].iloc[-1]))

    def test_baostock_frame_48bar_audit_complete(self):
        frame = make_bao_day_frame()
        from src.v004c_may_d1_coverage import audit_d1_minute_complete
        audit = audit_d1_minute_complete("600000", "2026-05-06", frame)
        self.assertTrue(audit["d1_minute_complete"])
        self.assertEqual(audit["actual_bars"], 48)
        self.assertEqual(audit["missing_bars"], 0)


if __name__ == "__main__":
    unittest.main()

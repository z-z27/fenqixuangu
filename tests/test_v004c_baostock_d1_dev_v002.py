# -*- coding: utf-8 -*-
"""v004c BaoStock D1 dev foundation v002 — 构建工具测试 (正确性修复)

覆盖 (§3/§5/§7/§8/§9/§10/§18-§23/§27/§28/§31/§32/§37/§43/§44/§50):
- Test A (§7/§10): May label 不完整事件 -> recovery 先于 build_event_row 完成,
  daily 只追加缺失日期, dev row dev_label_complete=True, target = 恢复后审计值
- Test B (§7/§8): 重叠 fetch 只审计不覆盖 (existing wins), 仅追加缺失日期
- Test C (§5/§37): recovery 失败 -> dev_label_complete=False, target 显式 NA
  (UNKNOWN != FALSE, 不得变 False)
- §44: D1 日内 5min high==low (一字板, 603065 型) -> d1_close_location None 为
  structural missing (structural=1, unexpected=0), dev_feature_source_complete
  仍 True; label complete -> dev_training_eligible True
- §37: True/False/NA 三态 CSV round-trip 可区分
- §50: v001 -> v002 universe 精确匹配函数
- §31: write_outputs --output 契约 (5 文件全部真实写到 out_dir)
- §32: stats 无 target_label_complete 双计, 只报告 dev_* 资格字段
- §27/§28: provenance 从 cache meta 真实读取 + pkl SHA256
"""
import hashlib
import importlib.util
import json
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
from src.v004c_may_d1_coverage import load_daily_frame
from src.v004c_minute_features import MINUTE_DEPENDENT_FEATURES
from tests.test_v004c_may_d1_coverage import bar_times

BASE = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "v004c_baostock_d1_dev_builder_v002",
    BASE / "tools" / "build_v004c_baostock_d1_dev_v002.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

FIELDS = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]

CAL_IDX = {"2026-05-07": 0, "2026-05-08": 1, "2026-05-11": 2, "2026-05-12": 3}


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


def make_flat_day_frame(code: str = "603065", date: str = "2026-06-11",
                        price: float = 12.36) -> pd.DataFrame:
    """48-bar 全 high==low==open==close 的一字板日 (603065 型, §44)。"""
    rows = []
    no_dash = date.replace("-", "")
    for i, t in enumerate(bar_times()):
        hhmm = t.replace(":", "")
        rows.append([
            date, f"{no_dash}{hhmm}00{i:03d}", f"sh.{code}",
            f"{price:.2f}", f"{price:.2f}", f"{price:.2f}", f"{price:.2f}",
            str(1000 + i), str(10000.0 + i),
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


def make_daily_rows(code: str, dates: list[str], base_close: float = 12.0) -> pd.DataFrame:
    """vendor 日线返回 (normalize_daily_frame 风格列)。"""
    rows = []
    for i, d in enumerate(dates):
        c = base_close + i * 0.1
        rows.append({
            "date": d,
            "code": code,
            "market": "sz",
            "open": round(c * 0.99, 4),
            "high": round(c * 1.02, 4),
            "low": round(c * 0.98, 4),
            "close": round(c, 4),
            "volume": 1_000_000 + i * 100,
            "amount": round(c * 1_000_000.0, 2),
            "pct_chg": 1.0,
            "change": round(c * 0.01, 4),
            "amplitude": 2.0,
            "turnover_rate": 1.0,
            "source": "baostock_daily",
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


def make_incomplete_event(code: str = "600001", signal_date: str = "2026-05-07") -> pd.DataFrame:
    """May label 不完整事件 (frozen candidates 风格)。"""
    return pd.DataFrame([{
        "event_id": f"{code}_{signal_date}",
        "code": code,
        "name": "测试",
        "signal_date": signal_date,
        "break_date": signal_date,
        "board_streak_before_break": 3,
        "source_window": "may",
        "target7_daily_d2open_d3high": np.nan,
        "tail_loss_daily_5pct": np.nan,
        "daily_label_complete": False,
        "daily_label_reason": "d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11)",
        "target_training_eligible": None,
        "label_d2_date": "2026-05-08",
        "label_d3_date": "2026-05-11",
    }])


class FakeDailyProvider:
    def __init__(self, daily_rows=None, fail=False):
        self.daily_rows = daily_rows
        self.fail = fail
        self.calls = 0

    def fetch_daily_history(self, code, start_date, end_date, adjust="none"):
        self.calls += 1
        if self.fail:
            raise RuntimeError("vendor daily fetch failed (fake)")
        return self.daily_rows.copy(), "baostock_daily"


class V002RecoveryTestCase(unittest.TestCase):
    """Test A (§3/§7/§10): recovery 先于 build_event_row, 结果进入 dev CSV。"""

    def test_recovery_appends_missing_dates_and_updates_dev_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder.DAILY_DIR = Path(tmp) / "daily"
            builder.DAILY_DIR.mkdir(parents=True, exist_ok=True)
            # 已有 canonical daily 05-05..05-08 (immutable); d3=05-11 缺失 -> 不完整
            existing = make_daily_frame(n_prior=29, d1_date="2026-05-08")
            builder._persist_daily_cache("600001", existing)
            # vendor 返回 05-09..05-12 -> 只应追加这 4 个缺失日期
            fetched = make_daily_rows("600001", ["2026-05-09", "2026-05-10",
                                                 "2026-05-11", "2026-05-12"])
            provider = FakeDailyProvider(fetched)
            universe = make_incomplete_event()
            record = builder.recover_may_labels(universe, provider, CAL_IDX, {},
                                                do_recovery=True)
            r = record["600001_2026-05-07"]
            self.assertEqual(r["reason_before"],
                             "d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11)")
            self.assertFalse(r["label_complete_before"])
            self.assertTrue(r["label_complete_after"])
            self.assertTrue(r["label_recovered"])
            self.assertEqual(r["rows_appended"], 4)
            self.assertEqual(r["overlap_dates"], [])
            self.assertTrue(r["daily_cache_written"])
            # canonical daily cache 现在含 d3 日期 (追加, 未覆盖旧行)
            daily_now = load_daily_frame("600001", builder.DAILY_DIR)
            self.assertIn("2026-05-11", set(daily_now["date"].astype(str)))
            self.assertEqual(len(daily_now), 34)
            # §10: May 全量 live re-audit
            patched = builder.audit_may_labels_live(universe, CAL_IDX, {})
            ev = patched.iloc[0]
            self.assertTrue(bool(ev["daily_label_complete"]))
            # dev row: recovery 结果真正进入行构建
            cache = Baostock5mCache(Path(tmp) / "bao", suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600001", make_bao_day_frame(code="600001", date="2026-05-07"),
                        meta=None)
            row = builder.build_event_row(ev, cache, daily_now, CAL_IDX, {})
            self.assertTrue(row["dev_label_complete"])
            self.assertIsNotNone(row["target7_daily_d2open_d3high"])
            self.assertEqual(row["target7_daily_d2open_d3high"], r["target7_after"])
            self.assertEqual(row["tail_loss_daily_5pct"], r["tail_loss_after"])
            self.assertFalse(pd.isna(row["target7_daily_d2open_d3high"]))

    def test_recovery_already_complete_skips_fetch(self):
        # §9: 已恢复状态 (live audit 已 complete) -> 无网络, 无写入
        with tempfile.TemporaryDirectory() as tmp:
            builder.DAILY_DIR = Path(tmp) / "daily"
            builder.DAILY_DIR.mkdir(parents=True, exist_ok=True)
            existing = make_daily_frame(n_prior=29, d1_date="2026-05-08")
            extra = make_daily_rows("600001", ["2026-05-09", "2026-05-10",
                                               "2026-05-11", "2026-05-12"])
            merged = pd.concat([existing, extra[["date", "code", "open", "high",
                                                 "low", "close"]]], ignore_index=True)
            builder._persist_daily_cache("600001", merged)
            provider = FakeDailyProvider(fail=True)  # 若尝试 fetch 则失败 -> 测试抓 bug
            universe = make_incomplete_event()
            record = builder.recover_may_labels(universe, provider, CAL_IDX, {},
                                                do_recovery=True)
            r = record["600001_2026-05-07"]
            self.assertEqual(provider.calls, 0)
            self.assertTrue(r["label_complete_after"])
            self.assertEqual(r["action"], "already_recovered")
            self.assertEqual(r["rows_appended"], 0)


class V002AppendOnlyMergeTestCase(unittest.TestCase):
    """Test B (§7/§8): existing canonical rows immutable, 仅追加缺失日期。"""

    def test_overlapping_fetch_existing_wins_only_missing_appended(self):
        existing = make_daily_frame(n_prior=2, d1_date="2026-05-08")  # 05-06..05-08
        fetched = make_daily_rows("600000", ["2026-05-06", "2026-05-07",
                                             "2026-05-08", "2026-05-11"])
        # vendor 的 3 个重叠日给出不同 close (模拟上游差异)
        fetched.loc[fetched["date"] == "2026-05-06", "close"] = 99.0
        fetched.loc[fetched["date"] == "2026-05-07", "close"] = 99.5
        fetched.loc[fetched["date"] == "2026-05-08", "close"] = 100.0
        merged, audit = builder.merge_daily_append_only(existing, fetched)
        self.assertEqual(audit["rows_appended"], 1)
        self.assertEqual(audit["overlap_dates"],
                         ["2026-05-06", "2026-05-07", "2026-05-08"])
        self.assertEqual(audit["overlap_exact_match"], 0)
        self.assertEqual(audit["overlap_value_diff_count"], 3)
        self.assertEqual(len(merged), 4)
        # existing 胜出: 重叠日 close 保持原值
        merged_map = dict(zip(merged["date"], merged["close"]))
        ex_map = dict(zip(existing["date"], existing["close"]))
        for d in ("2026-05-06", "2026-05-07", "2026-05-08"):
            self.assertEqual(float(merged_map[d]), float(ex_map[d]), f"{d} must be immutable")
            self.assertNotEqual(float(merged_map[d]), 99.0)
        self.assertIn("2026-05-11", merged_map)

    def test_overlap_exact_match_counts(self):
        existing = make_daily_frame(n_prior=1, d1_date="2026-05-07")
        # existing 补上与 vendor 一致的 volume, 值级比较才真正"精确匹配"
        existing["volume"] = [1_000_000, 1_000_100]
        fetched = make_daily_rows("600000", ["2026-05-06", "2026-05-07"],
                                  base_close=10.0)
        merged, audit = builder.merge_daily_append_only(existing, fetched)
        self.assertEqual(audit["overlap_dates"], ["2026-05-06", "2026-05-07"])
        self.assertEqual(audit["overlap_exact_match"], 2)
        self.assertEqual(audit["overlap_value_diff_count"], 0)
        self.assertEqual(audit["rows_appended"], 0)
        self.assertEqual(len(merged), 2)


class V002RecoveryFailureTestCase(unittest.TestCase):
    """Test C (§5/§37): recovery 失败 -> label 保持不完整, target 显式 NA。"""

    def test_recovery_failure_keeps_label_incomplete_na_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder.DAILY_DIR = Path(tmp) / "daily"
            builder.DAILY_DIR.mkdir(parents=True, exist_ok=True)
            builder._persist_daily_cache("600001",
                                         make_daily_frame(n_prior=29, d1_date="2026-05-08"))
            provider = FakeDailyProvider(fail=True)
            universe = make_incomplete_event()
            record = builder.recover_may_labels(universe, provider, CAL_IDX, {},
                                                do_recovery=True)
            r = record["600001_2026-05-07"]
            self.assertIn("recovery_failed", r["action"])
            self.assertFalse(r["label_complete_after"])
            self.assertFalse(r["label_recovered"])
            patched = builder.audit_may_labels_live(universe, CAL_IDX, {})
            ev = patched.iloc[0]
            self.assertFalse(bool(ev["daily_label_complete"]))
            cache = Baostock5mCache(Path(tmp) / "bao", suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600001", make_bao_day_frame(code="600001", date="2026-05-07"),
                        meta=None)
            daily_now = load_daily_frame("600001", builder.DAILY_DIR)
            row = builder.build_event_row(ev, cache, daily_now, CAL_IDX, {})
            self.assertFalse(row["dev_label_complete"])
            # §37: UNKNOWN != FALSE — NA 是真实缺失 (NAType), 绝不是 False
            self.assertTrue(pd.isna(row["target7_daily_d2open_d3high"]))
            self.assertTrue(pd.isna(row["tail_loss_daily_5pct"]))
            self.assertIsNot(row["target7_daily_d2open_d3high"], False)
            self.assertIsNot(row["tail_loss_daily_5pct"], False)

    def test_no_incomplete_events_returns_empty(self):
        # complete 事件不进 recovery 记录
        universe = pd.DataFrame([{
            "event_id": "600000_2026-05-06", "code": "600000", "name": "",
            "signal_date": "2026-05-06", "break_date": "2026-05-06",
            "board_streak_before_break": 3, "source_window": "may",
            "target7_daily_d2open_d3high": True, "tail_loss_daily_5pct": False,
            "daily_label_complete": True, "daily_label_reason": "ok",
            "target_training_eligible": None, "label_d2_date": None,
            "label_d3_date": None,
        }])
        record = builder.recover_may_labels(universe, FakeDailyProvider(),
                                            {"2026-05-06": 0}, {}, do_recovery=True)
        self.assertEqual(record, {})


class V002StructuralMissingTestCase(unittest.TestCase):
    """§44: 一字板日 (high==low) 的 d1_close_location 是 structural missing。"""

    def test_flat_bar_day_structural_not_eligibility_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("603065", make_flat_day_frame(), meta=None)
            daily = make_daily_frame(code="603065", n_prior=29, d1_date="2026-06-11")
            event = make_event_series(code="603065", signal_date="2026-06-11",
                                      window="june")
            event["daily_label_complete"] = True
            row = builder.build_event_row(event, cache, daily, {}, {})
        self.assertTrue(row["d1_minute_complete"])
        self.assertEqual(row["actual_bars"], 48)
        self.assertIsNone(row["d1_close_location"])  # 官方公式分母 0 -> None, 数学定义未改
        self.assertEqual(row["structural_missing_count"], 1)
        self.assertEqual(row["unexpected_missing_count"], 0)
        self.assertEqual(row["structural_missing_features"], "d1_close_location")
        self.assertEqual(row["unexpected_missing_features"], "")
        # §21/§23: structural missing 不使资格失败
        self.assertTrue(row["dev_feature_source_complete"])
        self.assertTrue(row["dev_training_eligible"])
        # 其余 21 个 minute features 不受一字板影响
        for name in MINUTE_DEPENDENT_FEATURES:
            if name != "d1_close_location":
                self.assertIsNotNone(row[name], f"{name} must stay computed")

    def test_classify_feature_missing_distinguishes_structural(self):
        flat = make_flat_day_frame()
        row = {"d1_close_location": None,
               **{f: 1.0 for f in MINUTE_DEPENDENT_FEATURES if f != "d1_close_location"}}
        structural, unexpected = builder.classify_feature_missing(row, flat)
        self.assertEqual(structural, ["d1_close_location"])
        self.assertEqual(unexpected, [])
        # 同样缺失, 但日内非一字板 (high != low) -> unexpected
        normal = make_bao_day_frame()
        structural2, unexpected2 = builder.classify_feature_missing(row, normal)
        self.assertEqual(structural2, [])
        self.assertEqual(unexpected2, ["d1_close_location"])


class V002LabelStateTestCase(unittest.TestCase):
    """§37: True/False/NA 三态 CSV round-trip。"""

    def test_label_three_state_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600000", make_bao_day_frame(), meta=None)
            daily = make_daily_frame()
            ev_true = make_event_series()  # label complete, target True
            ev_false = make_event_series(code="600001", signal_date="2026-05-07")
            ev_false["target7_daily_d2open_d3high"] = False  # complete label, 真实负样本
            ev_na = make_event_series(code="600002", signal_date="2026-05-07")
            ev_na["daily_label_complete"] = False  # UNKNOWN -> NA
            rows = [
                builder.build_event_row(ev_true, cache, daily, {}, {}),
                builder.build_event_row(ev_false, cache, daily, {}, {}),
                builder.build_event_row(ev_na, cache, daily, {}, {}),
            ]
            universe = pd.DataFrame([{
                "event_id": "600000_2026-05-06", "code": "600000", "name": "测试",
                "signal_date": "2026-05-06", "break_date": "2026-05-06",
                "board_streak_before_break": 3, "source_window": "may",
            }, {
                "event_id": "600001_2026-05-07", "code": "600001", "name": "测试",
                "signal_date": "2026-05-07", "break_date": "2026-05-07",
                "board_streak_before_break": 3, "source_window": "may",
            }, {
                "event_id": "600002_2026-05-07", "code": "600002", "name": "测试",
                "signal_date": "2026-05-07", "break_date": "2026-05-07",
                "board_streak_before_break": 3, "source_window": "may",
            }])
            out = Path(tmp) / "out"
            builder.write_outputs(rows, universe, {}, {}, {}, {}, out, None)
            dev = pd.read_csv(out / builder.DEV_CSV_NAME)
            t0 = dev.loc[0, "target7_daily_d2open_d3high"]
            t1 = dev.loc[1, "target7_daily_d2open_d3high"]
            self.assertIs(t0, True)
            self.assertIs(t1, False)  # 显式 False 与 NA 区分
            self.assertTrue(pd.isna(dev.loc[2, "target7_daily_d2open_d3high"]))
            self.assertIsNot(dev.loc[2, "target7_daily_d2open_d3high"], False)
            self.assertTrue(pd.isna(dev.loc[2, "tail_loss_daily_5pct"]))
            self.assertFalse(bool(dev.loc[2, "dev_label_complete"]))


class V002UniverseAndOutputsTestCase(unittest.TestCase):
    """§50 universe 精确匹配 + §31 --output 契约 + §27/§28 provenance。"""

    def test_universe_v001_v002_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            v001 = Path(tmp) / "v001.csv"
            pd.DataFrame([
                {"event_id": "600000_2026-05-06", "code": "600000",
                 "signal_date": "2026-05-06", "board_streak_before_break": 3},
                {"event_id": "600001_2026-06-01", "code": "600001",
                 "signal_date": "2026-06-01", "board_streak_before_break": 2},
            ]).to_csv(v001, index=False)
            dev = pd.DataFrame([
                {"event_id": "600000_2026-05-06", "code": "600000",
                 "signal_date": "2026-05-06", "board_streak_before_break": 3},
                {"event_id": "600001_2026-06-01", "code": "600001",
                 "signal_date": "2026-06-01", "board_streak_before_break": 2},
            ])
            result = builder.check_universe_vs_v001(dev, v001)
            self.assertTrue(result["exact_match"])
            self.assertEqual(result["matched_rows"], 2)
            self.assertEqual(result["v001_rows"], 2)
            # 扰动 board_streak -> 必须检出
            dev_bad = dev.copy()
            dev_bad.loc[0, "board_streak_before_break"] = 2
            with self.assertRaises(RuntimeError):
                builder.check_universe_vs_v001(dev_bad, v001)

    def test_write_outputs_all_five_files_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600000", make_bao_day_frame(), meta=None)
            rows = [builder.build_event_row(make_event_series(), cache,
                                            make_daily_frame(), {}, {})]
            universe = pd.DataFrame([{
                "event_id": "600000_2026-05-06", "code": "600000", "name": "测试",
                "signal_date": "2026-05-06", "break_date": "2026-05-06",
                "board_streak_before_break": 3, "source_window": "may",
            }])
            out = Path(tmp) / "custom_out"
            files = builder.write_outputs(rows, universe,
                                          {"600000": {"cache_status": "reused"}},
                                          {}, {}, {}, out, "127.0.0.1:7890",
                                          review_text="# v002 test review\n")
            for name in (builder.DEV_CSV_NAME, builder.QUALITY_CSV_NAME,
                         builder.LINEAGE_CSV_NAME, builder.PROVENANCE_NAME,
                         builder.REVIEW_NAME):
                self.assertTrue((out / name).exists(), f"{name} missing in {out}")
            self.assertEqual(files["dev"].parent, out)
            self.assertEqual(files["review"].parent, out)
            prov = json.loads((out / builder.PROVENANCE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(prov["socks5_proxy"], "127.0.0.1:7890")
            self.assertEqual(
                prov["cache_dir"],
                str(builder.BAO_CACHE_DIR.relative_to(builder.ROOT)).replace("\\", "/"))
            self.assertIn("600000", prov["codes"])
            self.assertEqual(prov["codes"]["600000"]["cache_status"], "reused")

    def test_stats_dev_fields_no_legacy_double_count(self):
        # §32: target_label_complete 双计移除; 只报告 dev_* 资格字段
        dev = pd.DataFrame({
            "event_id": ["A", "B", "C"],
            "code": ["600000", "600001", "600002"],
            "source_window": ["may", "may", "june"],
            "signal_date": ["2026-05-06", "2026-05-07", "2026-06-01"],
            "board_streak_before_break": [2, 3, 2],
            "minute_source": ["baostock_5m"] * 3,
            "d1_minute_complete": [True, True, True],
            "minute_feature_complete": [True, True, True],
            "daily_feature_complete": [True, True, True],
            "final_x_data_complete": [True, True, True],
            "dev_label_complete": [True, False, True],
            "dev_feature_source_complete": [True, True, True],
            "dev_training_eligible": [True, False, True],
            "structural_missing_count": [0, 1, 0],
            "unexpected_missing_count": [0, 0, 0],
            "structural_missing_features": ["", "d1_close_location", ""],
            "unexpected_missing_features": ["", "", ""],
        })
        stats = builder.build_stats(dev, {"600000": {}}, {}, {}, {})
        self.assertNotIn("target_label_complete", stats["combined"])
        for window in ("may", "june", "combined"):
            self.assertIn("dev_label_complete", stats[window])
            self.assertIn("dev_feature_source_complete", stats[window])
            self.assertIn("dev_training_eligible", stats[window])
        self.assertEqual(stats["may"]["dev_label_complete"], 1)
        self.assertEqual(stats["may"]["dev_training_eligible"], 1)
        self.assertEqual(stats["combined"]["dev_training_eligible"], 2)
        self.assertEqual(stats["combined"]["structural_missing_rows"], 1)
        self.assertEqual(stats["combined"]["unexpected_missing_rows"], 0)
        self.assertEqual(stats["structural_missing_detail"][0]["feature"],
                         "d1_close_location")
        self.assertEqual(stats["recovery_safety"]["overlapping_existing_dates_overwritten"], 0)

    def test_cache_provenance_entry_from_meta(self):
        # §27/§28: provenance 从 cache meta 真实读取 + pkl SHA256
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            frame = make_bao_day_frame()
            cache.write("600000", frame, meta=cache.build_meta(
                "600000", frame, source=BAOSTOCK_5M_SOURCE, adjustment="none",
                interval=NORMALIZED_5M_INTERVAL,
                fetch_timestamp="2026-08-08 00:00:00"))
            entry = builder.cache_provenance_entry(cache, "600000", "reused")
            self.assertEqual(entry["cache_status"], "reused")
            self.assertEqual(entry["source"], BAOSTOCK_5M_SOURCE)
            self.assertEqual(entry["adjustment"], "none")
            self.assertEqual(entry["interval"], NORMALIZED_5M_INTERVAL)
            self.assertEqual(entry["row_count"], 48)
            self.assertEqual(entry["date_min"], "2026-05-06")
            self.assertEqual(entry["date_max"], "2026-05-06")
            self.assertEqual(entry["fetch_timestamp"], "2026-08-08 00:00:00")
            self.assertEqual(
                entry["cache_sha256"],
                hashlib.sha256(cache.path("600000").read_bytes()).hexdigest())

    def test_provenance_summary_excludes_fetch_timestamp(self):
        # 确定性: 摘要不含 fetch timestamp / 不含 cache_status 计数
        summary = builder.build_provenance_summary({
            "600000": {"cache_status": "fetched",
                       "date_min": "2026-05-06", "date_max": "2026-06-30",
                       "row_count": 48, "cache_sha256": "abc",
                       "fetch_timestamp": "2026-08-08 00:00:00"},
        })
        self.assertEqual(summary["cached_codes"], 1)
        self.assertEqual(summary["total_rows"], 48)
        self.assertEqual(summary["date_min"], "2026-05-06")
        self.assertNotIn("fetch_timestamp", summary)
        self.assertNotIn("cache_status", summary)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""v004c 阶段1: D1 首次断板训练数据冻结与审计 — 测试 (阶段1.1收尾修复)

覆盖: 原 35 项 + 阶段1.1 新增:
- 严格二元字段 (Target7=2/-1/0.5/"yes"/""/NaN/inf 失败; 合法值通过)
- Git provenance fail closed (非 Git 目录 / 空 HEAD / 39位 / 非hex / 有效40位)
- 股票代码严格六位 (12345/1234567/ABC/NaN 失败)
- 缓存非法日期 (not-a-date / 2026-13-01 / 空白 / NaT / 重复)
- 输出目录门禁 (不存在/空 通过; stale.txt/未知子目录 失败)
- 邻接证据表 (333 行语义, 唯一 event_id, 全 verified, D3 前序列 SHA)
- git_status_after 语义 (写入内容与采集值一致)
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_d1_dataset import (
    DatasetValidationError,
    GitProvenance,
    build_canonical_time_fields,
    classify_column_lineage,
    collect_git_provenance,
    discover_v004c_schema,
    normalize_binary_series,
    read_v004c_source,
    run_v004c_d1_dataset_build,
    select_d1_first_break_rows,
    validate_d1_dataset,
    validate_git_head,
    validate_source_manifest,
)

BASE = Path(__file__).resolve().parent.parent

WEEKDAY_CAL = [
    "2026-05-29", "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
    "2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
]


def make_git_prov(head: str = "testhead", status: str = "dirty") -> GitProvenance:
    return GitProvenance(git_root=Path.cwd(), git_head=head, git_status=status,
                         git_dirty=bool(status.strip()))


def make_input_df(**overrides) -> pd.DataFrame:
    rows = [
        ("E001", "000001", "2026-06-02", "2026-06-02", 2, 0, "d0", True, True, 10.0, 11.0, 10.5, 0.10, "2026-06-01", "2026-06-03", "2026-06-04"),
        ("E002", "000002", "2026-06-03", "2026-06-03", 3, 0, "d0", True, False, 20.0, 21.0, 19.8, 0.05, "2026-06-02", "2026-06-04", "2026-06-05"),
        ("E003", "000003", "2026-06-04", "2026-06-04", 2, 0, "d0", True, True, 5.0, 5.4, 4.9, 0.08, "2026-06-03", "2026-06-05", "2026-06-08"),
        ("E004", "000004", "2026-06-03", "2026-06-02", 2, 1, "post", True, True, 8.0, 9.0, 8.5, 0.125, "2026-06-01", "2026-06-03", "2026-06-04"),
        ("E005", "000005", "2026-06-04", "2026-06-02", 2, 2, "post", True, False, 3.0, 3.1, 2.8, 0.033, "2026-06-01", "2026-06-03", "2026-06-04"),
    ]
    frame = pd.DataFrame(rows, columns=[
        "event_id", "code", "signal_date", "break_date", "board_streak_before_break",
        "days_since_break", "stage_group", "daily_label_quality_ok",
        "target7_daily_d2open_d3high", "d2_open_daily", "d3_high_daily", "d3_close_daily",
        "daily_d2open_to_d3high_return", "last_board_date", "label_d2_date", "label_d3_date"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["name"] = "测试"
    frame["d1_close"] = frame["d3_close_daily"] * 0.9
    frame["d1_open"] = frame["d2_open_daily"] * 0.95
    frame["d1_high"] = frame["d3_high_daily"] * 0.98
    frame["d1_low"] = frame["d1_close"] * 0.97
    frame["d1_ma5"] = frame["d1_close"] * 0.99
    frame["break_close"] = frame["d1_close"] * 1.0
    frame["break_volume"] = 1_000_000.0
    frame["v004a_probability"] = 0.5
    frame["v004a_rank"] = 3
    frame["is_v004a_top15"] = True
    frame["v002_rank"] = 5
    frame["recognition_score"] = 0.4
    frame["d2_trade_date"] = frame["label_d2_date"]
    frame["d3_trade_date"] = frame["label_d3_date"]
    frame["d2open_to_d3high_return"] = frame["daily_d2open_to_d3high_return"]
    frame["tail_loss_5pct"] = False
    frame["tail_loss_daily_5pct"] = False
    frame["d1_bar_count"] = 48
    frame["d2_bar_count"] = 48
    frame["d3_bar_count"] = 48
    frame["d1_minute_complete"] = True
    frame["d2_minute_complete"] = True
    frame["d3_minute_complete"] = True
    frame["future_data_status"] = "complete"
    frame["target_training_eligible"] = True
    frame["tail_training_eligible"] = True
    frame["daily_label_source_d2"] = "expected_date_daily_cache"
    frame["daily_label_source_d3"] = "expected_date_daily_cache"
    frame["daily_label_cache_path"] = "data/cache/daily/000001_daily.pkl"
    frame["daily_label_cache_mtime"] = "2026-08-03 22:00:00"
    frame["daily_label_cache_sha256"] = "abc"
    for k, v in overrides.items():
        frame[k] = v
    for c in ("daily_label_quality_ok", "target7_daily_d2open_d3high",
              "tail_loss_5pct", "tail_loss_daily_5pct"):
        if c in frame.columns:
            frame[c] = frame[c].astype(object)
    return frame


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(tmp: Path, csv_path: Path, **overrides) -> Path:
    base = {
        "dataset_version": "v004c-dataset-0.2.2",
        "candidate_definition_version": "break-repair-0.1",
        "label_definition_version": "v004c-daily-label-0.2",
        "adjustment": "none",
        "output_files": {
            "v004c_training_primary_v022.csv": {
                "rows": 5, "bytes": csv_path.stat().st_size, "sha256": _sha(csv_path)},
        },
    }
    base.update(overrides)
    p = tmp / "source_manifest.json"
    p.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    return p


def make_cache_dir(tmp: Path, code_dates: dict[str, list[str]] | None = None,
                   bad_dates: dict[str, list[str]] | None = None) -> Path:
    cache = tmp / "daily_cache"
    cache.mkdir(parents=True, exist_ok=True)
    if code_dates is None:
        code_dates = {"000001": WEEKDAY_CAL, "000002": WEEKDAY_CAL, "000003": WEEKDAY_CAL}
    for code, dates in code_dates.items():
        df = pd.DataFrame({
            "date": dates, "open": 10.0, "high": 11.0, "low": 9.5,
            "close": 10.5, "volume": 1000, "amount": 10500.0})
        df.to_pickle(cache / f"{code}_daily.pkl")
    for code, dates in (bad_dates or {}).items():
        df = pd.DataFrame({
            "date": dates, "open": 10.0, "high": 11.0, "low": 9.5,
            "close": 10.5, "volume": 1000, "amount": 10500.0})
        df.to_pickle(cache / f"{code}_daily.pkl")
    return cache


def _prepare(tmp: Path, df: pd.DataFrame | None = None, **manifest_overrides) -> tuple[Path, Path, Path]:
    df = make_input_df() if df is None else df
    csv_path = tmp / "input.csv"
    df.to_csv(csv_path, index=False)
    manifest_path = make_manifest(tmp, csv_path, **manifest_overrides)
    cache_dir = make_cache_dir(tmp)
    return csv_path, manifest_path, cache_dir


class V004cD1DatasetTest(unittest.TestCase):
    maxDiff = None

    # ---------- 阶段1.1: 严格二元字段 ----------
    def test_binary_invalid_values_fail(self):
        bad_cases = [2, -1, 0.5, "yes", "no", "Y", "N", "", "  ", float("inf"), float("-inf")]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                series = pd.Series([True, bad, False])
                with self.assertRaises(DatasetValidationError) as ctx:
                    normalize_binary_series(series, column_name="target7_daily_d2open_d3high")
                self.assertIn("target7_daily_d2open_d3high", str(ctx.exception))

    def test_binary_nan_fails(self):
        series = pd.Series([True, np.nan, False])
        with self.assertRaises(DatasetValidationError):
            normalize_binary_series(series, column_name="target7_daily_d2open_d3high")

    def test_binary_valid_values_pass(self):
        series = pd.Series([True, False, 1, 0, 1.0, 0.0, "true", "false", "1", "0",
                            " True ", "FALSE"])
        out = normalize_binary_series(series, column_name="target7_daily_d2open_d3high")
        self.assertEqual(list(out), [True, False, True, False, True, False,
                                     True, False, True, False, True, False])

    def test_binary_invalid_blocks_freeze(self):
        for field in ("target7_daily_d2open_d3high", "daily_label_quality_ok",
                      "tail_loss_daily_5pct"):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    df = make_input_df()
                    df.loc[df["event_id"] == "E002", field] = 2
                    csv_path, manifest_path, cache_dir = _prepare(Path(tmp), df)
                    with self.assertRaises(DatasetValidationError) as ctx:
                        run_v004c_d1_dataset_build(
                            input_file=csv_path, input_manifest=manifest_path,
                            output_dir=Path(tmp) / "out", daily_cache_dir=cache_dir,
                            git_provenance_before=make_git_prov(),
                            git_provenance_after=make_git_prov())
                    self.assertIn("binary_field_invalid", str(ctx.exception))
                    self.assertIn(field, str(ctx.exception))

    def test_binary_target_stats_real_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            out = root / "out"
            result = run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            v = result["manifest"]["validation"]
            self.assertEqual(v["target7_positive_count"], 2)
            self.assertAlmostEqual(v["target7_rate"], 2 / 3, places=10)

    # ---------- 阶段1.1: Git provenance ----------
    def test_git_head_validation(self):
        for bad in ("", "abc", "f94fe4b3226e60d033b0a99e8ceb54cfba8aff8",  # 39 位
                    "zzzfe4b3226e60d033b0a99e8ceb54cfba8aff8d"):  # 非 hex
            with self.subTest(bad=bad):
                with self.assertRaises(DatasetValidationError):
                    validate_git_head(bad)
        validate_git_head("f94fe4b3226e60d033b0a99e8ceb54cfba8aff8d")

    def test_git_provenance_non_git_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path.cwd()
            os.chdir(tmp)
            try:
                with self.assertRaises(DatasetValidationError):
                    collect_git_provenance(Path(tmp))
            finally:
                os.chdir(old)

    def test_git_provenance_valid_repo_passes(self):
        prov = collect_git_provenance(Path.cwd())
        self.assertRegex(prov.git_head, r"^[0-9a-f]{40}$")
        self.assertTrue(prov.git_root.is_absolute())

    # ---------- 阶段1.1: 股票代码严格六位 ----------
    def test_code_format_strict_six_digits(self):
        for good in ("000001", "600000"):
            with self.subTest(good=good):
                df = make_input_df()
                df.loc[0, "code"] = good
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "in.csv"
                    df.to_csv(path, index=False)
                    read_df, _ = read_v004c_source(path)
                    self.assertTrue(read_df["code"].str.fullmatch(r"\d{6}").all())

    def test_code_format_invalid_fails(self):
        for bad in (None, float("nan"), "", "12345", "1234567", "ABC", "12345A", " 000001"):
            with self.subTest(bad=bad):
                df = make_input_df()
                df.loc[0, "code"] = bad
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "in.csv"
                    df.to_csv(path, index=False)
                    with self.assertRaises(DatasetValidationError):
                        read_v004c_source(path)

    # ---------- 阶段1.1: 缓存非法日期 ----------
    def test_cache_invalid_dates_fail(self):
        for bad_date in ("not-a-date", "2026-13-01", "", None):
            with self.subTest(bad_date=bad_date):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    csv_path, manifest_path, _ = _prepare(root)
                    dates = list(WEEKDAY_CAL)
                    dates[3] = bad_date
                    cache_dir = make_cache_dir(root / "cache2",
                                               code_dates={"000002": WEEKDAY_CAL, "000003": WEEKDAY_CAL},
                                               bad_dates={"000001": dates})
                    with self.assertRaises(DatasetValidationError) as ctx:
                        run_v004c_d1_dataset_build(
                            input_file=csv_path, input_manifest=manifest_path,
                            output_dir=root / "out", daily_cache_dir=cache_dir,
                            git_provenance_before=make_git_prov(),
                            git_provenance_after=make_git_prov())
                    # 冻结被阻止; 内部"非法日期"细节在失败审计中
                    self.assertIn("trade_date_adjacency", str(ctx.exception))
                    failed_dirs = [p for p in root.iterdir() if p.name.startswith("out_failed_")]
                    self.assertEqual(len(failed_dirs), 1)
                    audit = json.loads((failed_dirs[0] / "failure_audit.json").read_text(encoding="utf-8"))
                    self.assertIn("非法日期", " ".join(audit["hard_failures"]) + audit["message"])
                    self.assertFalse((root / "out").exists())

    # ---------- 阶段1.1: 输出目录门禁 ----------
    def test_output_dir_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            # 不存在 → 允许
            out1 = root / "out_missing"
            run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out1, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            # 空目录 → 允许
            out2 = root / "out_empty"
            out2.mkdir()
            run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out2, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            # 含 stale.txt → 失败
            out3 = root / "out_stale"
            out3.mkdir()
            (out3 / "stale.txt").write_text("old", encoding="utf-8")
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=out3, daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("输出目录非空", str(ctx.exception))
            self.assertTrue((out3 / "stale.txt").exists())  # 不得自动删除未知文件
            # 含未知子目录 → 失败
            out4 = root / "out_subdir"
            out4.mkdir()
            (out4 / "unknown_sub").mkdir()
            with self.assertRaises(DatasetValidationError):
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=out4, daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())

    # ---------- 阶段1.1: 邻接证据表 ----------
    def test_adjacency_evidence_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            out = root / "out"
            result = run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            ev = pd.read_csv(out / "v004c_d1_trade_date_adjacency_evidence.csv", dtype={"code": str})
            self.assertEqual(len(ev), 3)
            self.assertEqual(ev["event_id"].nunique(), 3)
            self.assertTrue((ev["adjacency_verified"] == True).all())  # noqa: E712
            self.assertTrue(ev["daily_cache_sha256"].notna().all())
            self.assertTrue(ev["date_sequence_through_d3_sha256"].notna().all())
            self.assertIn("v004c_d1_trade_date_adjacency_evidence.csv", result["manifest"]["output_files"])
            self.assertEqual(result["manifest"]["output_files"]["v004c_d1_trade_date_adjacency_evidence.csv"]["role"],
                             "provenance")

    # ---------- 阶段1.1: git_status_after 语义 ----------
    def test_git_status_after_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            out = root / "out"
            result = run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(head="h1", status="before-status"),
                git_provenance_after=make_git_prov(head="h1", status="after-status"))
            self.assertEqual((out / "git_status_after.txt").read_text(encoding="utf-8").strip(),
                             "after-status")
            self.assertEqual(result["manifest"]["git_status_after"].strip(), "after-status")
            self.assertEqual(result["manifest"]["git_dirty_before"], True)

    # ---------- 阶段1.1: 失败审计目录 ----------
    def test_failure_audit_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = make_input_df()
            df.loc[df["event_id"] == "E002", "target7_daily_d2open_d3high"] = 2
            csv_path, manifest_path, cache_dir = _prepare(root, df)
            out = root / "out"
            with self.assertRaises(DatasetValidationError):
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=out, daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            failed_dirs = [p for p in root.iterdir() if p.name.startswith("out_failed_")]
            self.assertEqual(len(failed_dirs), 1)
            audit = json.loads((failed_dirs[0] / "failure_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["failure_stage"], "validation")
            self.assertIn("binary_field_invalid", " ".join(audit["hard_failures"]))
            self.assertFalse(out.exists())  # 正式输出目录未被污染

    # ---------- 原有: 保留与筛选 ----------
    def test_keeps_stage_d0_rows(self):
        df = make_input_df()
        d1, _ = select_d1_first_break_rows(df, discover_v004c_schema(df))
        self.assertEqual(len(d1), 3)
        self.assertTrue((d1["stage_group"] == "d0").all())

    def test_excludes_post_rows(self):
        df = make_input_df()
        _, excluded = select_d1_first_break_rows(df, discover_v004c_schema(df))
        self.assertEqual(len(excluded), 2)
        self.assertTrue((excluded["exclusion_reason"] == "not_d1_first_break_observation").all())

    def test_stage_inconsistency_fails(self):
        df = make_input_df()
        df.loc[df["event_id"] == "E001", "days_since_break"] = 1
        with self.assertRaises(DatasetValidationError):
            select_d1_first_break_rows(df, discover_v004c_schema(df))

    def test_duplicate_event_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = make_input_df()
            dup = df[df["event_id"] == "E001"].copy()
            dup["event_id"] = "E002"
            df2 = pd.concat([df, dup], ignore_index=True)
            csv_path, manifest_path, cache_dir = _prepare(Path(tmp), df2)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=Path(tmp) / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("duplicate_event_id", str(ctx.exception))

    def test_duplicate_code_break_date_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = make_input_df()
            dup = df[df["event_id"] == "E001"].copy()
            dup["event_id"] = "E099"
            df2 = pd.concat([df, dup], ignore_index=True)
            csv_path, manifest_path, cache_dir = _prepare(Path(tmp), df2)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=Path(tmp) / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("duplicate_code_break_date", str(ctx.exception))

    def test_signal_not_equal_break_fails(self):
        df = make_input_df()
        df.loc[df["event_id"] == "E002", "signal_date"] = "2026-06-05"
        with self.assertRaises(DatasetValidationError):
            select_d1_first_break_rows(df, discover_v004c_schema(df))

    def test_board_streak_not_two_or_three_excluded(self):
        df = make_input_df()
        df.loc[df["event_id"] == "E003", "board_streak_before_break"] = 1
        d1, excluded = select_d1_first_break_rows(df, discover_v004c_schema(df))
        self.assertEqual(len(d1), 2)
        self.assertIn("board_streak_not_two_or_three", set(excluded["exclusion_reason"]))

    def test_target7_missing_excluded(self):
        df = make_input_df()
        df.loc[df["event_id"] == "E002", "target7_daily_d2open_d3high"] = pd.NA
        d1, excluded = select_d1_first_break_rows(df, discover_v004c_schema(df))
        self.assertEqual(len(d1), 2)
        self.assertIn("target7_missing", set(excluded["exclusion_reason"]))

    def test_label_quality_false_excluded(self):
        df = make_input_df()
        df.loc[df["event_id"] == "E003", "daily_label_quality_ok"] = False
        d1, excluded = select_d1_first_break_rows(df, discover_v004c_schema(df))
        self.assertEqual(len(d1), 2)
        self.assertIn("daily_label_quality_failed", set(excluded["exclusion_reason"]))

    def test_round_trip_float_stable(self):
        tricky = pd.DataFrame({
            "event_id": ["E001"], "code": ["000001"], "signal_date": ["2026-06-02"],
            "break_date": ["2026-06-02"], "days_since_break": [0], "stage_group": ["d0"],
            "board_streak_before_break": [2], "daily_label_quality_ok": [True],
            "target7_daily_d2open_d3high": [True],
            "d2_open_daily": [7.000000000000001],
            "d3_high_daily": [7.49], "d3_close_daily": [7.1],
            "daily_d2open_to_d3high_return": [0.07000000000000001],
            "last_board_date": ["2026-06-01"], "label_d2_date": ["2026-06-03"],
            "label_d3_date": ["2026-06-04"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.csv"
            tricky.to_csv(path, index=False)
            df1, _ = read_v004c_source(path)
            df2, _ = read_v004c_source(path)
            self.assertEqual(df1["daily_d2open_to_d3high_return"].iloc[0],
                             df2["daily_d2open_to_d3high_return"].iloc[0])
            self.assertEqual(df1["d2_open_daily"].iloc[0], 7.000000000000001)

    def test_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            hashes = []
            for i in range(2):
                out = root / f"out{i}"
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=out, daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
                hashes.append(_sha(out / "v004c_training_d1_v001.csv"))
            self.assertEqual(hashes[0], hashes[1])

    def test_future_fields_not_trainable(self):
        df = make_input_df()
        d1, _ = select_d1_first_break_rows(df, discover_v004c_schema(df))
        d1 = build_canonical_time_fields(d1, discover_v004c_schema(d1))
        lineage = classify_column_lineage(d1)
        for col in ("d2_open_daily", "d3_high_daily", "target7_daily_d2open_d3high",
                    "d2open_to_d3high_return", "label_d2_date", "d2_entry_date", "d3_outcome_date"):
            row = lineage[lineage["column_name"] == col]
            self.assertFalse(bool(row["allowed_for_future_feature_analysis"].iloc[0]), col)
            self.assertIn(row["column_role"].iloc[0], ("label", "outcome_audit"))

    def test_model_fields_not_trainable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            out = root / "out"
            run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            train = pd.read_csv(out / "v004c_training_d1_v001.csv")
            for col in ("v004a_probability", "v004a_rank", "is_v004a_top15", "v002_rank", "recognition_score"):
                self.assertNotIn(col, train.columns)
            lineage = pd.read_csv(out / "v004c_d1_column_lineage.csv")
            for col in ("v004a_probability", "v002_rank", "recognition_score"):
                row = lineage[lineage["column_name"] == col]
                self.assertEqual(row["column_role"].iloc[0], "existing_model_audit")
                self.assertFalse(bool(row["allowed_for_future_feature_analysis"].iloc[0]))

    def test_target_recompute_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = make_input_df()
            df.loc[df["event_id"] == "E002", "daily_d2open_to_d3high_return"] = 0.10
            df.loc[df["event_id"] == "E002", "target7_daily_d2open_d3high"] = True
            csv_path, manifest_path, cache_dir = _prepare(Path(tmp), df)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=Path(tmp) / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("target7_recompute_mismatch", str(ctx.exception))

    def test_date_relation_error_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = make_input_df()
            df.loc[df["event_id"] == "E002", "label_d2_date"] = "2026-06-03"
            csv_path, manifest_path, cache_dir = _prepare(Path(tmp), df)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=Path(tmp) / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("time_relation", str(ctx.exception))

    def test_no_network_dependency(self):
        source = (BASE / "src" / "v004c_d1_dataset.py").read_text(encoding="utf-8")
        for token in ("requests", "urllib", "socket", "akshare", "http.client", "import aiohttp"):
            self.assertNotIn(token, source)

    def test_manifest_version_fields_wrong_fails(self):
        cases = [
            ("dataset_version", "v004c-dataset-0.2.1"),
            ("candidate_definition_version", "break-repair-0.9"),
            ("label_definition_version", "v004c-daily-label-0.1"),
            ("adjustment", "qfq"),
        ]
        for field, bad_value in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    csv_path, manifest_path, _ = _prepare(Path(tmp), **{field: bad_value})
                    frame, manifest = read_v004c_source(csv_path, manifest_path)
                    with self.assertRaises(DatasetValidationError) as ctx:
                        validate_source_manifest(manifest, csv_path)
                    self.assertIn(field, str(ctx.exception))

    def test_manifest_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "in.csv"
            make_input_df().to_csv(csv_path, index=False)
            with self.assertRaises(DatasetValidationError):
                validate_source_manifest(None, csv_path)

    def test_manifest_json_broken_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "in.csv"
            make_input_df().to_csv(csv_path, index=False)
            broken = Path(tmp) / "broken.json"
            broken.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                read_v004c_source(csv_path, broken)

    def test_manifest_csv_sha_wrong_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, manifest_path, _ = _prepare(
                Path(tmp), output_files={"v004c_training_primary_v022.csv": {
                    "rows": 5, "bytes": 1, "sha256": "0" * 64}})
            frame, manifest = read_v004c_source(csv_path, manifest_path)
            with self.assertRaises(DatasetValidationError) as ctx:
                validate_source_manifest(manifest, csv_path)
            self.assertIn("SHA256", str(ctx.exception))

    def test_manifest_correct_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, manifest_path, cache_dir = _prepare(Path(tmp))
            frame, manifest = read_v004c_source(csv_path, manifest_path)
            self.assertEqual(validate_source_manifest(manifest, csv_path)["dataset_version"],
                             "v004c-dataset-0.2.2")

    def test_cache_dir_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, _ = _prepare(root)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=root / "out", daily_cache_dir=root / "missing_cache",
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("缓存目录缺失", str(ctx.exception))

    def test_stock_cache_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, _ = _prepare(root)
            cache_dir = make_cache_dir(root / "cache2",
                                       code_dates={"000002": WEEKDAY_CAL, "000003": WEEKDAY_CAL})
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=root / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("trade_date_adjacency", str(ctx.exception))

    def test_cache_duplicate_dates_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, _ = _prepare(root)
            dup_dates = WEEKDAY_CAL + ["2026-06-03"]
            cache_dir = make_cache_dir(root / "cache2",
                                       code_dates={"000002": WEEKDAY_CAL, "000003": WEEKDAY_CAL},
                                       bad_dates={"000001": dup_dates})
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=root / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("trade_date_adjacency", str(ctx.exception))

    def test_d2_not_next_trade_day_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = make_input_df()
            df.loc[df["event_id"] == "E001", "label_d2_date"] = "2026-06-04"
            csv_path, manifest_path, cache_dir = _prepare(root, df)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=root / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("trade_date_adjacency", str(ctx.exception))

    def test_d3_not_next_trade_day_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = make_input_df()
            df.loc[df["event_id"] == "E001", "label_d3_date"] = "2026-06-05"
            csv_path, manifest_path, cache_dir = _prepare(root, df)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=root / "out", daily_cache_dir=cache_dir,
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("trade_date_adjacency", str(ctx.exception))

    def test_adjacency_success_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            out = root / "out"
            result = run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            adj = result["manifest"]["validation"]["d2_d3_adjacency"]
            self.assertEqual(adj["verified_rows"], 3)
            self.assertEqual(adj["mismatch_rows"], 0)
            self.assertEqual(adj["not_verified_rows"], 0)
            cache_meta = result["manifest"]["daily_cache"]
            self.assertEqual(cache_meta["file_count"], 3)
            files = cache_meta["files"]
            self.assertIn("000001", files)
            for key in ("raw_row_count", "valid_date_count", "invalid_date_count",
                        "duplicate_date_count", "min_date", "max_date", "sha256"):
                self.assertIn(key, files["000001"], key)

    def test_absolute_date_fields_not_trainable(self):
        df = make_input_df()
        d1, _ = select_d1_first_break_rows(df, discover_v004c_schema(df))
        d1 = build_canonical_time_fields(d1, discover_v004c_schema(d1))
        lineage = classify_column_lineage(d1)
        for col in ("signal_date", "break_date", "last_board_date", "d0_last_board_date",
                    "d1_break_date", "d1_signal_date", "d2_entry_date", "d3_outcome_date"):
            row = lineage[lineage["column_name"] == col]
            self.assertEqual(len(row), 1, col)
            self.assertFalse(bool(row["allowed_for_future_feature_analysis"].iloc[0]), col)
            self.assertNotIn(row["column_role"].iloc[0], ("d0_feature", "d1_feature"), col)

    def test_data_quality_available_as_of_split(self):
        df = make_input_df()
        d1, _ = select_d1_first_break_rows(df, discover_v004c_schema(df))
        d1 = build_canonical_time_fields(d1, discover_v004c_schema(d1))
        lineage = classify_column_lineage(d1)
        expectations = {
            "d1_bar_count": "D1_CLOSE", "d1_minute_complete": "D1_CLOSE",
            "d2_bar_count": "POST_D3", "d3_bar_count": "POST_D3",
            "future_data_status": "POST_D3", "target_training_eligible": "POST_D3",
            "daily_label_source_d2": "POST_D3",
            "daily_label_cache_mtime": "BUILD_TIME", "daily_label_cache_sha256": "BUILD_TIME",
        }
        for col, expected in expectations.items():
            row = lineage[lineage["column_name"] == col]
            self.assertEqual(len(row), 1, col)
            self.assertEqual(row["available_as_of"].iloc[0], expected, col)
            self.assertFalse(bool(row["allowed_for_future_feature_analysis"].iloc[0]), col)

    def test_expected_signal_dates_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            with self.assertRaises(DatasetValidationError) as ctx:
                run_v004c_d1_dataset_build(
                    input_file=csv_path, input_manifest=manifest_path,
                    output_dir=root / "out", daily_cache_dir=cache_dir,
                    expected_stats={"signal_dates": 999},
                    git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
            self.assertIn("signal_dates", str(ctx.exception))

    def test_invalid_label_price_fails(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    df = make_input_df()
                    df.loc[df["event_id"] == "E002", "d2_open_daily"] = bad
                    csv_path, manifest_path, cache_dir = _prepare(root, df)
                    with self.assertRaises(DatasetValidationError) as ctx:
                        run_v004c_d1_dataset_build(
                            input_file=csv_path, input_manifest=manifest_path,
                            output_dir=root / "out", daily_cache_dir=cache_dir,
                            git_provenance_before=make_git_prov(), git_provenance_after=make_git_prov())
                    self.assertIn("invalid_label_price", str(ctx.exception))

    def test_repo_relative_posix(self):
        from src.v004c_d1_dataset import _repo_relative_posix

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = Path.cwd()
            os.chdir(root)
            try:
                rel = _repo_relative_posix(root / "out" / "x.csv", git_root=root)
                self.assertNotIn("\\", rel)
                self.assertFalse(Path(rel).is_absolute())
                self.assertEqual(rel, "out/x.csv")
            finally:
                os.chdir(old)

    def test_full_package_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, manifest_path, cache_dir = _prepare(root)
            out = root / "out"
            result = run_v004c_d1_dataset_build(
                input_file=csv_path, input_manifest=manifest_path,
                output_dir=out, daily_cache_dir=cache_dir,
                git_provenance_before=make_git_prov(head="testhead", status="dirty-before"),
                git_provenance_after=make_git_prov(head="testhead", status="dirty-after"))
            expected = [
                "v004c_d1_snapshot_v001.csv", "v004c_training_d1_v001.csv",
                "v004c_d1_existing_model_audit_v001.csv", "v004c_d1_excluded_rows_v001.csv",
                "v004c_d1_data_quality.csv", "v004c_d1_candidate_audit.csv",
                "v004c_d1_column_lineage.csv", "v004c_d1_trade_date_adjacency_evidence.csv",
                "v004c_d1_data_manifest.json", "v004c_d1_dataset_review.md",
                "git_head.txt", "git_status_before.txt", "git_status_after.txt",
            ]
            for name in expected:
                self.assertTrue((out / name).exists(), name)
            non_manifest = [n for n in expected if n != "v004c_d1_data_manifest.json"]
            self.assertEqual(set(result["manifest"]["output_files"].keys()), set(non_manifest))
            for meta in result["manifest"]["output_files"].values():
                self.assertNotIn("\\", meta["path"])
            self.assertEqual((out / "git_head.txt").read_text(encoding="utf-8").strip(), "testhead")
            self.assertEqual((out / "git_status_before.txt").read_text(encoding="utf-8").strip(), "dirty-before")
            review = (out / "v004c_d1_dataset_review.md").read_text(encoding="utf-8")
            self.assertIn("| D1 输出行数 | 3 |", review)
            self.assertIn("| Target7 正样本 | 2 (0.666667) |", review)
            self.assertIn("| 冻结状态 | FROZEN |", review)
            self.assertIn("git_status_after = 本次构建所有非 manifest 输出文件写完后的 Git 状态", review)
            quality = pd.read_csv(out / "v004c_d1_data_quality.csv")
            q = dict(zip(quality["metric"], quality["value"]))
            self.assertEqual(q["selected_d1_rows"], "3")
            self.assertTrue(result["manifest"]["frozen"])
            self.assertEqual(result["manifest"]["output_dir_gate"]["unmanaged_output_files"], [])


if __name__ == "__main__":
    unittest.main()

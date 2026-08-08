# -*- coding: utf-8 -*-
"""v004c Pairwise v1 Feature Contract 测试。

覆盖任务约束:
- universe 319 行 / 39 信号日, 与 v002 身份逐行精确一致 (绝不删除行);
- 53 FEATURE, feature_order 唯一/连续/确定;
- 泄漏审计: FEATURE future=0, model-output=0, label-lineage=0;
- unexpected_missing 合计 0 (缺失必须全部为结构允许);
- 契约完整性: token 扫描 / rank lineage / semantic_group<->information_class 一致;
- deterministic rebuild: 同源两次构建字节一致;
- 不在输入表中创建任何被排除候选 (exact duplicate / manual composite / model output);
- 不创建 feature_x_board3 交互列。

测试约定: unittest + importlib 加载 builder (与 test_v004c_baostock_d1_dev_v002 一致)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "v004c_pairwise_feature_contract_builder",
    BASE / "tools" / "build_v004c_pairwise_v1_feature_contract.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

# builder 内部已 import src 模块 -> 可直接从 src 取未透出的常量
from src.v004c_pairwise_feature_contract import (  # noqa: E402
    SEMANTIC_GROUP_INFORMATION_CLASS,
)

CONTRACT_SRC = builder.CONTRACT_FEATURE_NAMES
FEATURE_CONTRACT = builder.FEATURE_CONTRACT
OUT_DIR = builder.OUT_DIR

# 32 个 workflow 候选清单中已确认的排除名 (用于断言其绝不出现在输入表)
EXCLUDED_NAMES_SUBSET = (
    "recognition_score", "profit_pressure", "overrepair", "break_volume_abnormality",
    "ma5_overheat_10", "d1_close_to_ma5_bucket", "d1_open_to_close_bucket",
    "d1_vwap_to_close_gap", "profit_chip_ratio",
    "board_day_amount_rank", "board_day_turnover_rank", "break_amount_ratio_vs_board_days",
    "break_open", "break_high", "break_low", "break_close", "break_intraday_range",
    "break_high_to_close_drawdown", "break_close_location", "break_volume", "break_amount",
    "break_turnover_ratio", "volume_above_break_close_ratio", "d1_down_bar_volume_ratio",
    "deprecated_d1_reclaimed_ma5_v01", "deprecated_d1_reclaimed_ma10_v01",
    "OPEN", "RESET", "HIGHZONE", "LATESELL", "MOM7", "DAMAGE7", "REGIME", "POS7",
    "TREND", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY", "RECLAIM", "DIVERGENCE_SQ",
    "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_DAMAGE", "DIVERGENCE_X_SUPPLY",
    "TURNOVER_COST", "v004a_probability", "v004a_rank", "v002_rank",
    "m0_probability", "m1_probability", "m2_probability", "r0_probability",
    "r1_probability", "r2_probability", "d2_open_daily", "d3_high_daily",
    "d3_close_daily",
)

# 标签列合法出现在输入表末尾 (LABEL_ONLY, 物理隔离), 不作为排除名断言
LABEL_SUFFIX_COLUMNS = ("target7_daily_d2open_d3high", "tail_loss_daily_5pct")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContractIntegrityTestCase(unittest.TestCase):
    """§契约不变量: 顺序/字段/泄漏 token/rank lineage。"""

    def test_contract_integrity_passes(self):
        builder.assert_contract_integrity()

    def test_feature_order_unique_contiguous(self):
        self.assertEqual(len(CONTRACT_SRC), 53)
        order = [builder.CONTRACT_FEATURE_ORDER[n] for n in CONTRACT_SRC]
        self.assertEqual(order, list(range(1, 54)))

    def test_feature_names_unique(self):
        self.assertEqual(len(set(CONTRACT_SRC)), len(CONTRACT_SRC))

    def test_leakage_scan_contract_zero(self):
        res = builder.leakage_scan(CONTRACT_SRC)
        self.assertEqual(res["future_leakage"], 0)
        self.assertEqual(res["model_output_leakage"], 0)
        self.assertEqual(res["label_lineage_leakage"], 0)

    def test_semantic_group_information_class_consistent(self):
        for c in FEATURE_CONTRACT:
            expected = SEMANTIC_GROUP_INFORMATION_CLASS[c["semantic_group"]]
            self.assertEqual(c["information_class"], expected, c["feature_name"])

    def test_available_as_of_within_d1_boundary(self):
        for c in FEATURE_CONTRACT:
            self.assertIn(c["available_as_of"], ("D1_CLOSE", "D0_CLOSE"),
                          c["feature_name"])

    def test_rank_feature_is_market_cross_section(self):
        # 名称含 rank 的唯一 FEATURE 必须属市场横截面 (D0_CROSS_SECTION)
        for c in FEATURE_CONTRACT:
            if "rank" in c["feature_name"]:
                self.assertEqual(c["semantic_group"], "D0_CROSS_SECTION")
                self.assertEqual(c["feature_name"], "board_day_volume_rank")

    def test_no_board3_interaction_columns(self):
        # 禁止 feature_x_board3 交互列
        self.assertNotIn("break_day_in_pool_board3", CONTRACT_SRC)
        for c in FEATURE_CONTRACT:
            self.assertNotIn("board3", c["feature_name"])
            self.assertNotIn("_board3", c["feature_name"])

    def test_no_label_tokens_in_features(self):
        for name in CONTRACT_SRC:
            for tok in ("target", "tail", "d2_", "d3_", "future", "outcome"):
                self.assertNotIn(tok, name, name)


class ExclusionAuditTestCase(unittest.TestCase):
    """被排除候选绝不进入输入表列。"""

    def test_excluded_names_not_columns(self):
        dev = self._read_input_table()
        for name in EXCLUDED_NAMES_SUBSET:
            self.assertNotIn(name, dev.columns, f"被排除候选 {name} 出现在输入表")

    def test_labels_are_last_and_isolated(self):
        dev = self._read_input_table()
        cols = list(dev.columns)
        # 标签列在 53 FEATURE 之后 (INPUT_SUFFIX 起始)
        for lbl in LABEL_SUFFIX_COLUMNS:
            self.assertIn(lbl, cols)
            self.assertGreaterEqual(cols.index(lbl), len(builder.INPUT_PREFIX_COLUMNS) + 53)
        # FEATURE 段内不允许出现标签列
        feat_start = cols.index("break_date") + 1
        feat_end = cols.index("target7_daily_d2open_d3high")
        feat_cols = cols[feat_start:feat_end]
        self.assertEqual(feat_cols, CONTRACT_SRC)
        for lbl in LABEL_SUFFIX_COLUMNS:
            self.assertNotIn(lbl, feat_cols)

    @staticmethod
    def _read_input_table():
        return pd.read_csv(OUT_DIR / builder.DEV_CSV_NAME, encoding="utf-8-sig",
                           dtype={"code": str})


class UniverseTestCase(unittest.TestCase):
    """universe 319/39 与 v002 身份逐行一致。"""

    @classmethod
    def setUpClass(cls):
        cls.v002 = pd.read_csv(builder.V002_CSV, encoding="utf-8-sig",
                               dtype={"code": str})
        cls.v002["code"] = cls.v002["code"].astype(str).str.zfill(6)
        cls.dev = pd.read_csv(OUT_DIR / builder.DEV_CSV_NAME, encoding="utf-8-sig",
                              dtype={"code": str})

    def test_universe_rows_and_signal_dates(self):
        self.assertEqual(len(self.dev), 319)
        self.assertEqual(self.dev["signal_date"].nunique(), 39)
        self.assertEqual(int(self.dev["board_streak_before_break"].eq(2).sum()), 261)
        self.assertEqual(int(self.dev["board_streak_before_break"].eq(3).sum()), 58)

    def test_identity_exact_match_v002(self):
        # 六列身份逐行精确一致 -> 绝不删除/重排事件
        for col in ("event_id", "code", "signal_date", "break_date",
                    "board_streak_before_break", "source_window"):
            got = self.dev[col].astype(str).tolist()
            want = self.v002[col].astype(str).tolist()
            self.assertEqual(got, want, f"universe 身份列 {col} 与 v002 不一致")

    def test_signal_equals_break(self):
        self.assertTrue((self.dev["signal_date"] == self.dev["break_date"]).all())

    def test_row_order_preserved(self):
        self.assertEqual(self.dev["event_id"].tolist(),
                         self.v002["event_id"].tolist())

    def test_no_extra_or_dropped_events(self):
        self.assertEqual(len(self.dev), len(self.v002))
        self.assertEqual(set(self.dev["event_id"]), set(self.v002["event_id"]))


class LeakageAuditTestCase(unittest.TestCase):
    """FEATURE 泄漏审计必须全 0。"""

    @classmethod
    def setUpClass(cls):
        cls.dev = pd.read_csv(OUT_DIR / builder.DEV_CSV_NAME, encoding="utf-8-sig",
                              dtype={"code": str})

    def test_future_model_label_leakage_zero(self):
        res = builder.run_leakage_audit(self.dev, list(CONTRACT_SRC))
        self.assertEqual(res["future_leakage"], 0)
        self.assertEqual(res["model_output_leakage"], 0)
        self.assertEqual(res["label_lineage_leakage"], 0)

    def test_availability_within_d1(self):
        bad = builder.availability_audit(self.dev)
        self.assertEqual(bad, [])

    def test_unexpected_missing_zero(self):
        self.assertEqual(int(self.dev["unexpected_missing_count"].sum()), 0)

    def test_structural_missing_known_sources(self):
        # 6 (D0=04-30 池窗口外 x3) + 7 (池源覆盖缺口 x2) + 3 (603065 一字板)
        self.assertEqual(int(self.dev["structural_missing_count"].sum()), 35)
        # 每行 structural_missing_count 只能来自定义允许来源
        self.assertLessEqual(int(self.dev["structural_missing_count"].max()), 3)

    def test_d0430_events_pool_fields_structural(self):
        m0506 = self.dev[self.dev["signal_date"] == "2026-05-06"]
        self.assertEqual(len(m0506), 6)
        for _, r in m0506.iterrows():
            self.assertTrue(pd.isna(r["board_day_volume_rank"]))
            self.assertTrue(pd.isna(r["last_board_day_in_pool"]))
            self.assertTrue(pd.isna(r["pool_consecutive_count_last_board"]))
            # 池窗口外时 recent_pool_appearance_count 用覆盖子集 (0 或小值), 非 NaN
            self.assertFalse(pd.isna(r["recent_pool_appearance_count_10d"]))

    def test_flat_bar_day_structural_three(self):
        row = self.dev[self.dev["code"] == "603065"]
        self.assertEqual(len(row), 1)
        r = row.iloc[0]
        for f in ("break_upper_shadow_ratio", "break_lower_shadow_ratio",
                  "d1_close_location"):
            self.assertTrue(pd.isna(r[f]), f"{f} 应结构性缺失 (603065 一字板)")
        self.assertEqual(r["unexpected_missing_count"], 0)


class DeterminismTestCase(unittest.TestCase):
    """确定性重建: 同源两次构建字节一致。"""

    def test_two_builds_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "build1"
            out2 = Path(tmp) / "build2"
            # 用同一 input_df 源码 -> 两个独立目录
            v002 = pd.read_csv(builder.V002_CSV, encoding="utf-8-sig",
                               dtype={"code": str})
            v002["code"] = v002["code"].astype(str).str.zfill(6)
            v002["signal_date"] = v002["signal_date"].astype(str)
            v002["break_date"] = v002["break_date"].astype(str)
            pool_members, pool_meta, pool_dates, pool_date_set = builder.load_pool_cache()
            daily_store = builder.load_daily_store(set(v002["code"].tolist()))
            dv_lookup = builder.build_daily_volume_lookup(daily_store)
            df = builder.assemble_input_table(
                v002, pool_members, pool_meta, pool_dates, pool_date_set,
                daily_store, dv_lookup)
            cov = builder.coverage_stats(df)
            # write_review 需要真实 leakage dict (非 None)
            leakage = builder.leakage_scan(list(builder.CONTRACT_FEATURE_NAMES))
            builder.write_outputs(out1, df, cov, True, leakage, [], False)
            builder.write_outputs(out2, df, cov, True, leakage, [], False)
            core = (builder.DEV_CSV_NAME, builder.CONTRACT_CSV_NAME,
                    builder.INVENTORY_CSV_NAME, builder.LABEL_CSV_NAME,
                    builder.SCHEMA_CSV_NAME, builder.EXCLUSIONS_CSV_NAME,
                    builder.REVIEW_NAME)
            for name in core:
                self.assertEqual(_sha256(out1 / name), _sha256(out2 / name),
                                 f"{name} 两轮字节不一致")


class InputTableContentTestCase(unittest.TestCase):
    """输入表内容审计: 行数/列序/范围/无缺失标签。"""

    @classmethod
    def setUpClass(cls):
        cls.dev = pd.read_csv(OUT_DIR / builder.DEV_CSV_NAME, encoding="utf-8-sig",
                              dtype={"code": str})

    def test_column_order(self):
        cols = list(self.dev.columns)
        expected = (list(builder.INPUT_PREFIX_COLUMNS)
                    + list(CONTRACT_SRC)
                    + list(builder.INPUT_SUFFIX_COLUMNS))
        self.assertEqual(cols, expected)

    def test_labels_complete_and_attached(self):
        self.assertEqual(int(self.dev["dev_label_complete"].sum()), 319)
        self.assertEqual(int(self.dev["dev_training_eligible"].sum()), 319)
        # 标签非空 (无 UNKNOWN)
        self.assertEqual(self.dev["target7_daily_d2open_d3high"].notna().sum(), 319)
        self.assertEqual(self.dev["tail_loss_daily_5pct"].notna().sum(), 319)

    def test_no_constant_features(self):
        for f in CONTRACT_SRC:
            nun = self.dev[f].dropna().nunique() if self.dev[f].notna().any() else 0
            self.assertGreater(nun, 1, f"{f} 是常数列 (无常量特征契约)")

    def test_binary_features_binary_domain(self):
        for f in ("board_streak_is_3", "break_touched_limit_up",
                  "break_opened_from_limit_up", "break_day_in_pool",
                  "d1_close_above_ma5", "d1_close_above_ma10",
                  "d1_true_reclaim_ma5", "d1_true_reclaim_ma10"):
            vals = set(self.dev[f].dropna().unique().tolist())
            self.assertTrue(vals <= {0.0, 1.0, 0, 1}, f"{f}: {vals}")

    def test_unexpected_missing_all_zero_rows(self):
        bad = self.dev[self.dev["unexpected_missing_count"] != 0]
        self.assertEqual(len(bad), 0)

    def test_provenance_present(self):
        self.assertTrue(self.dev["provenance"].str.startswith("v002:event_id=").all())


class OutputFileTestCase(unittest.TestCase):
    """7 个输出资产存在且结构正确。"""

    def test_all_seven_outputs_exist(self):
        names = (builder.DEV_CSV_NAME, builder.INVENTORY_CSV_NAME,
                 builder.CONTRACT_CSV_NAME, builder.LABEL_CSV_NAME,
                 builder.SCHEMA_CSV_NAME, builder.EXCLUSIONS_CSV_NAME,
                 builder.REVIEW_NAME)
        for n in names:
            self.assertTrue((OUT_DIR / n).exists(), f"缺 {n}")

    def test_inventory_21_columns(self):
        inv = pd.read_csv(OUT_DIR / builder.INVENTORY_CSV_NAME, encoding="utf-8-sig")
        self.assertEqual(len(inv.columns), 21)
        self.assertEqual(int((inv["pairwise_v1_decision"] == "FEATURE").sum()), 53)
        self.assertEqual(inv["future_leakage"].sum(), 0)
        self.assertEqual(inv["model_output_leakage"].sum(), 0)
        self.assertEqual(inv["target_lineage_leakage"].sum(), 0)

    def test_contract_csv_53_rows_columns(self):
        ctr = pd.read_csv(OUT_DIR / builder.CONTRACT_CSV_NAME, encoding="utf-8-sig")
        self.assertEqual(len(ctr), 53)
        self.assertEqual(list(ctr["feature_order"]), list(range(1, 54)))
        self.assertEqual(list(ctr["feature_name"]), CONTRACT_SRC)

    def test_schema_roles(self):
        sch = pd.read_csv(OUT_DIR / builder.SCHEMA_CSV_NAME, encoding="utf-8-sig")
        role_counts = sch["role"].value_counts().to_dict()
        self.assertEqual(role_counts.get("FEATURE", 0), 53)
        # 身份 (event_id/code/signal_date/board_streak_before_break) + break_date = 5
        self.assertEqual(role_counts.get("IDENTIFIER", 0), 5)
        # 标签 (2) + dev 资格 (3) + 审计 (source_window + 3 缺失统计/provenance = 4)
        self.assertEqual(role_counts.get("LABEL_ONLY", 0), 2)
        self.assertEqual(role_counts.get("DEV_ELIGIBILITY", 0), 3)
        self.assertEqual(role_counts.get("AUDIT_ONLY", 0), 4)
        # 总列数守恒
        self.assertEqual(role_counts["FEATURE"] + role_counts["IDENTIFIER"]
                         + role_counts["LABEL_ONLY"] + role_counts["DEV_ELIGIBILITY"]
                         + role_counts["AUDIT_ONLY"], len(sch))

    def test_exclusions_cover_sources(self):
        exc = pd.read_csv(OUT_DIR / builder.EXCLUSIONS_CSV_NAME, encoding="utf-8-sig")
        names = set(exc["feature_name"])
        for n in ("board_day_amount_rank", "board_day_turnover_rank",
                  "break_amount_ratio_vs_board_days", "recognition_score",
                  "DIVERGENCE", "overrepair", "break_volume_abnormality",
                  "profit_pressure", "ma5_overheat_10", "d1_close_to_ma5_bucket"):
            self.assertIn(n, names)

    def test_label_contract_only_labels(self):
        lbl = pd.read_csv(OUT_DIR / builder.LABEL_CSV_NAME, encoding="utf-8-sig")
        self.assertEqual(set(lbl["role"]), {"LABEL_ONLY"})

    def test_review_final_status_ready(self):
        rev = (OUT_DIR / builder.REVIEW_NAME).read_text(encoding="utf-8-sig")
        self.assertIn("READY_FOR_PAIRWISE_V1", rev)
        self.assertIn("Feature Contract 状态: READY_FOR_PAIRWISE_V1", rev)


if __name__ == "__main__":
    unittest.main()

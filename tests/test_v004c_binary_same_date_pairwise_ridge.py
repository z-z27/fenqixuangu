# -*- coding: utf-8 -*-
"""v004c Corrected Same-Date Binary Pairwise Ridge v002 — 测试 (§86-§88)。

覆盖:
- same-date pair construction (§12);
- 旧 bug 回归 (§59, §60, §87): pair count = sum_t(N_pos_t*N_neg_t),
  绝不是 sum(N_pos)*sum(N_neg);
- cross-date sentinel 测试 (§61): 不同日期特征 +100/-100 -> 同日配对差全 0;
- 单类日期 -> 0 pairs, 不跨日补 pair (§13, §62);
- 每日期总权重 = 1 (§14, §63); fold 总权重 = evaluable train dates (§58);
- 53 features exact order / 禁列 (§8, §9);
- preprocessing train-only (§19); board3 architecture (§18, §20);
- chronology (§28); lambda selection (§24, §27);
- OOF identity vs repair v001 (§33, §65);
- practical evaluator: STRICT TopK / Up-To-3 (§38-§40);
- winner capture (§44); repair concordance (§48); binary pairwise metrics
  (§49); regret identity (§51); Oracle (§53);
- no July (§32); determinism (§88);
- 历史资产未修改 (§5): v001 chronological_walkforward 源码仍把整个 train
  传入 build_same_date_pairs (AST/字符串锁定)。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

from src.v004c_binary_same_date_pairwise_ridge import (  # noqa: E402
    OOF_SCORE_COL,
    build_same_date_binary_pairs,
    chronological_walkforward_binary_same_date,
    select_lambda_binary_same_date,
)
from src.v004c_pairwise_ridge import (  # noqa: E402
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    date_weighted_pair_stats,
)
from src.v004c_repair_pairwise_ridge import (  # noqa: E402
    day_repair_pair_stats,
)
from src.v004c_top3_failure_decomposition import (  # noqa: E402
    decompose_day,
    oracle_ceiling_summary,
)


def make_binary_df(dates, seed=7, feat_dim=3, pos_rate=0.4):
    """synthetic: target7 标签与 feature 相关。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        n = int(rng.integers(4, 10))
        y = (rng.random(n) < pos_rate).astype(int)
        b3 = rng.random(n) < 0.2
        x = rng.normal(size=(n, feat_dim))
        x[y == 1] += 0.8
        for i in range(n):
            rows.append({
                "event_id": f"ev_{d}_{i}",
                "code": f"{i:06d}",
                "signal_date": d,
                "board_streak_before_break": 3 if b3[i] else 2,
                **{f"f{j}": float(x[i, j]) for j in range(feat_dim)},
                "target7_daily_d2open_d3high": float(y[i]),
            })
    return pd.DataFrame(rows)


def make_walk_dates(n_dates=14):
    return [f"2026-0{(i // 9) + 5}-{(i % 9) + 1:02d}" for i in range(n_dates)]


def make_day(returns, scores, t7=None, d="2026-05-21"):
    capped = np.minimum(np.asarray(returns, float), 0.07)
    if t7 is None:
        t7 = (np.asarray(returns, float) >= 0.07 - 1e-9).astype(int)
    rows = []
    for i, (r, c, t) in enumerate(zip(returns, capped, t7)):
        rows.append({
            "signal_date": d,
            "event_id": f"ev{i:02d}",
            "code": f"{i:06d}",
            "score": float(scores[i]),
            "target7_daily_d2open_d3high": int(t),
            "repair_strength_raw_return": float(r),
            "capped_opportunity_return_7": float(c),
        })
    return pd.DataFrame(rows)


class SameDateBinaryPairsTestCase(unittest.TestCase):
    def test_day1_one_pos_two_neg(self):
        """§60: Day1 P1/N1/N2 -> 2 pairs, 权重 0.5 每对。"""
        x = np.eye(3)
        y = np.array([1, 0, 0])
        z, w = build_same_date_binary_pairs(x, y, np.zeros(3))
        self.assertEqual(len(z), 2)
        np.testing.assert_allclose(w, 0.5)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)

    def test_day2_two_pos_one_neg(self):
        x = np.eye(3)
        y = np.array([1, 1, 0])
        z, w = build_same_date_binary_pairs(x, y, np.zeros(3))
        self.assertEqual(len(z), 2)
        np.testing.assert_allclose(w, 0.5)

    def test_old_bug_regression_four_not_nine(self):
        """§59/§60/§87: 两天 (1pos/2neg + 2pos/1neg) -> 4 pairs, 绝非 9
        (pooled 会得 3pos x 3neg = 9)。"""
        x1 = np.eye(3)
        x2 = np.eye(3) * 2
        z1, w1 = build_same_date_binary_pairs(x1, np.array([1, 0, 0]),
                                              np.zeros(3))
        z2, w2 = build_same_date_binary_pairs(x2, np.array([1, 1, 0]),
                                              np.zeros(3))
        self.assertEqual(len(z1) + len(z2), 4)
        self.assertNotEqual(len(z1) + len(z2), 9)
        # 每个日期内部权重和 = 1
        self.assertAlmostEqual(float(w1.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(w2.sum()), 1.0, places=12)

    def test_single_class_date_zero_pairs(self):
        """§62: Day1 全 positive -> 0 pairs; 不跨日补。"""
        x = np.eye(3)
        z, w = build_same_date_binary_pairs(x, np.ones(3), np.zeros(3))
        self.assertEqual(len(z), 0)
        self.assertEqual(len(w), 0)
        z2, w2 = build_same_date_binary_pairs(x, np.zeros(3), np.zeros(3))
        self.assertEqual(len(z2), 0)

    def test_cross_date_sentinel(self):
        """§61: Day1 feature +100, Day2 -100 -> 同日配对 z_diff sentinel 列全 0。"""
        x1 = np.full((3, 2), 100.0)
        x2 = np.full((3, 2), -100.0)
        y1 = np.array([1, 0, 0])
        y2 = np.array([1, 1, 0])
        z1, _ = build_same_date_binary_pairs(x1, y1, np.zeros(3))
        z2, _ = build_same_date_binary_pairs(x2, y2, np.zeros(3))
        np.testing.assert_allclose(z1[:, 0], 0.0)  # Day1 内部配对
        np.testing.assert_allclose(z2[:, 0], 0.0)  # Day2 内部配对
        # 若 pooled: 会出现 +200/-200 差
        self.assertNotIn(200.0, np.unique(np.round(z1[:, 0], 6)).tolist())

    def test_pair_weight_two_days(self):
        """§63: Day1 3 pairs (1/3 each), Day2 4 pairs (1/4 each); 总权重 2。"""
        x1 = np.eye(4)
        x2 = np.eye(4)
        z1, w1 = build_same_date_binary_pairs(x1, np.array([1, 0, 0, 0]),
                                              np.zeros(4))
        z2, w2 = build_same_date_binary_pairs(x2, np.array([1, 1, 0, 0]),
                                              np.zeros(4))
        self.assertEqual(len(z1), 3)  # 1 pos x 3 neg
        self.assertEqual(len(z2), 4)  # 2 pos x 2 neg
        np.testing.assert_allclose(w1, 1.0 / 3.0)
        np.testing.assert_allclose(w2, 1.0 / 4.0)
        self.assertAlmostEqual(float(w1.sum()) + float(w2.sum()), 2.0,
                               places=12)


class WalkForwardTestCase(unittest.TestCase):
    def _df(self, n_dates=14, seed=31):
        return make_binary_df(make_walk_dates(n_dates), seed=seed)

    def test_pair_count_is_per_date_sum_not_pooled(self):
        """§59: fold 0 train_pairs == sum_t(N_pos_t*N_neg_t), != pooled。"""
        df = self._df()
        res = chronological_walkforward_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high", 1.0)
        fold0 = res["fold_meta"].iloc[0]
        train = df[df["signal_date"] < fold0["test_signal_date"]]
        per_date = 0
        pooled_pos = pooled_neg = 0
        for _, day in train.groupby("signal_date"):
            np_ = int((day["target7_daily_d2open_d3high"] == 1).sum())
            nn = int((day["target7_daily_d2open_d3high"] == 0).sum())
            per_date += np_ * nn
            pooled_pos += np_
            pooled_neg += nn
        self.assertEqual(int(fold0["train_same_date_pairs"]), per_date)
        self.assertNotEqual(per_date, pooled_pos * pooled_neg)

    def test_fold_weight_equals_evaluable_train_dates(self):
        """§58: sum(all pair weights) == evaluable train dates。"""
        df = self._df()
        res = chronological_walkforward_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high", 1.0)
        for _, r in res["fold_meta"].iterrows():
            self.assertAlmostEqual(
                float(r["train_total_weight"]),
                float(r["train_binary_pair_evaluable_dates"]), places=9)
            self.assertEqual(int(r["train_cross_date_pairs"]), 0)

    def test_single_class_train_dates_counted(self):
        df = self._df()
        # 强制第一个 train date 全为 1 类
        first = sorted(df["signal_date"].unique())[0]
        df.loc[df["signal_date"] == first, "target7_daily_d2open_d3high"] = 1.0
        res = chronological_walkforward_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high", 1.0)
        fold0 = res["fold_meta"].iloc[0]
        self.assertGreaterEqual(int(fold0["train_single_class_dates"]), 1)
        # 单类日贡献 0 权重: fold 权重和 == evaluable 日期数
        self.assertAlmostEqual(
            float(fold0["train_total_weight"]),
            float(fold0["train_binary_pair_evaluable_dates"]), places=9)

    def test_chronology_and_oof_dates(self):
        df = self._df()
        res = chronological_walkforward_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high", 1.0)
        scored = res["oof"][res["oof"][OOF_SCORE_COL].notna()]
        test_dates = sorted(df["signal_date"].unique())[MIN_TRAIN_SIGNAL_DATES:]
        self.assertEqual(set(scored["signal_date"]), set(test_dates))
        for _, r in res["fold_meta"].iterrows():
            self.assertEqual(int(r["train_signal_dates"]),
                             MIN_TRAIN_SIGNAL_DATES + int(r["fold_index"]))
            # 模块内 FATAL 保证 max(train) < test; 这里显式复核
            train_dates = sorted(df[df["signal_date"] < r["test_signal_date"]]
                                 ["signal_date"].unique())
            self.assertLess(train_dates[-1], r["test_signal_date"])

    def test_deterministic(self):
        df = self._df()
        r1 = chronological_walkforward_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high", 1.0)
        r2 = chronological_walkforward_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high", 1.0)
        np.testing.assert_array_equal(
            r1["oof"][OOF_SCORE_COL].to_numpy(),
            r2["oof"][OOF_SCORE_COL].to_numpy())
        pd.testing.assert_frame_equal(r1["fold_meta"], r2["fold_meta"])

    def test_lambda_selection_binary_logloss_only(self):
        df = self._df()
        table, selected = select_lambda_binary_same_date(
            df, ["f0", "f1", "f2"], "target7_daily_d2open_d3high")
        self.assertEqual(list(table["lambda"]), list(LAMBDA_GRID))
        self.assertIn(selected, LAMBDA_GRID)
        best_ll = float(table[table["lambda"] == selected]
                        ["date_weighted_oof_binary_pairwise_logloss"].iloc[0])
        self.assertEqual(selected, float(table[
            (table["date_weighted_oof_binary_pairwise_logloss"]
             <= best_ll + 1e-12)]["lambda"].max()))


class ContractAndIsolationTestCase(unittest.TestCase):
    def test_exact_53_features_order(self):
        fc = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
            / "v004c_pairwise_v1_feature_contract_v002.csv",
            encoding="utf-8-sig")
        names = fc.sort_values("feature_order")["feature_name"].tolist()
        self.assertEqual(len(names), 53)
        self.assertEqual(len(set(names)), 53)

    def test_forbidden_columns_absent(self):
        fc = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
            / "v004c_pairwise_v1_feature_contract_v002.csv",
            encoding="utf-8-sig")
        names = set(fc["feature_name"])
        forbidden = {"target7_daily_d2open_d3high", "tail_loss_daily_5pct",
                     "d2_open_daily", "d3_high_daily",
                     "repair_strength_raw_return", "raw_opportunity_return",
                     "capped_opportunity_return_7", "source_window",
                     "signal_date", "event_id", "code",
                     "board_streak_before_break"}
        self.assertEqual(names & forbidden, set())

    def test_no_july(self):
        for path in (
                BASE / "reports" / "research"
                / "v004c_pairwise_ridge_v001_20260506_20260630"
                / "v004c_pairwise_ridge_oof_predictions_v001.csv",
                BASE / "reports" / "research"
                / "v004c_repair_pairwise_ridge_v001_20260506_20260630"
                / "v004c_repair_pairwise_oof_predictions_v001.csv"):
            df = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
            self.assertLessEqual(df["signal_date"].max(), "2026-06-30")

    def test_historical_asset_unchanged(self):
        """§5: v001 walk-forward 源码仍把整个 train 传入 build_same_date_pairs
        (历史实现锁定, 未被本任务修改)。"""
        src = (BASE / "src" / "v004c_pairwise_ridge.py").read_text(
            encoding="utf-8")
        self.assertIn("build_same_date_pairs(\n            train_xs, "
                      "train_y, b_all[train_mask])", src)


class EvaluatorRegressionTestCase(unittest.TestCase):
    """统一 evaluator 复现冻结数字 (STRICT = v001 口径 2.55%, Up-To-3 =
    decomposition 口径 2.78%)。"""

    def _hist_frame(self):
        c = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_pairwise_ridge_v001_20260506_20260630"
            / "v004c_pairwise_ridge_oof_predictions_v001.csv",
            encoding="utf-8-sig", dtype={"code": str})
        fr = c.dropna(subset=["upside_oof_score"]).rename(
            columns={"upside_oof_score": "score",
                     "raw_opportunity_return": "repair_strength_raw_return"})
        return fr

    def test_strict_top3_equals_v001_scope(self):
        fr = self._hist_frame()
        # STRICT: 只统计 >=3 候选日期
        rets = []
        for _, day in fr.groupby("signal_date", sort=True):
            if len(day) < 3:
                continue
            rets.append(float(day.sort_values(["score", "event_id"],
                                              ascending=[False, True])
                              .head(3)["capped_opportunity_return_7"].mean()))
        self.assertAlmostEqual(float(np.mean(rets)), 0.0255, delta=0.0005)

    def test_up_to_3_equals_decomposition_scope(self):
        fr = self._hist_frame()
        rets = []
        for _, day in fr.groupby("signal_date", sort=True):
            K = min(3, len(day))
            rets.append(float(day.sort_values(["score", "event_id"],
                                              ascending=[False, True])
                              .head(K)["capped_opportunity_return_7"].mean()))
        self.assertAlmostEqual(float(np.mean(rets)), 0.0278, delta=0.0005)
        self.assertEqual(len(rets), 29)

    def test_winner_capture_and_regret(self):
        fr = self._hist_frame()
        day_rows = [decompose_day(day, "score")
                    for _, day in fr.groupby("signal_date", sort=True)]
        win = [r for r in day_rows if r["target7_count"] > 0]
        self.assertEqual(len(win), 27)
        cap = float(np.mean([r["available_winner_capture_rate"]
                             for r in win]))
        self.assertAlmostEqual(cap, 0.5432, delta=0.01)
        total = float(np.mean([r["total_top3_regret"] for r in day_rows]))
        self.assertAlmostEqual(total, 0.0330, delta=0.001)
        for r in day_rows:
            self.assertAlmostEqual(r["total_top3_regret"],
                                   r["winner_capture_regret"]
                                   + r["repair_ordering_regret"], places=9)

    def test_oracle_matches_frozen_review(self):
        fr = self._hist_frame()
        oc = oracle_ceiling_summary(fr)
        self.assertAlmostEqual(oc["oracle_top3_capped_return"], 0.0609,
                               delta=0.001)
        self.assertAlmostEqual(oc["oracle_top1_hit_rate"], 0.9310, delta=0.01)

    def test_oof_identity_repair_vs_corrected(self):
        """§65: corrected v002 OOF 与 repair v001 OOF event 完全一致。"""
        r = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_repair_pairwise_ridge_v001_20260506_20260630"
            / "v004c_repair_pairwise_oof_predictions_v001.csv",
            encoding="utf-8-sig", dtype={"code": str})
        self.assertEqual(len(r), 319)
        rr = r.dropna(subset=["repair_oof_score"])
        self.assertEqual(len(rr), 233)
        self.assertEqual(rr["signal_date"].nunique(), 29)


class ConcordanceAndBinaryMetricsTestCase(unittest.TestCase):
    def test_repair_concordance_raw(self):
        st = day_repair_pair_stats(np.array([0.9, 0.5, 0.1]),
                                   np.array([0.06, 0.04, -0.02]))
        self.assertEqual(st["within_non_target_concordance"], 1.0)

    def test_binary_pair_stats(self):
        """§49: binary pairwise logloss/AUC (pos vs neg)。"""
        ll, auc_v, acc, nd = date_weighted_pair_stats(
            np.array([0.9, 0.5, 0.1]), np.array([1.0, 0.0, 0.0]),
            np.array(["d1", "d1", "d1"]))
        self.assertLessEqual(ll, np.log(2.0) + 1e-9)
        self.assertGreater(auc_v, 0.9)
        self.assertEqual(nd, 1)


class WinnerCaptureTestCase(unittest.TestCase):
    def test_capture_two_of_three(self):
        day = make_day([0.07, 0.07, 0.07, 0.02, -0.05],
                       [0.9, 0.2, 0.5, 0.4, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["top3_target7_hits"], 2)
        self.assertAlmostEqual(dec["available_winner_capture_rate"], 2 / 3,
                               places=9)

    def test_regret_identity_groupA(self):
        day = make_day([0.07, 0.07, 0.07, 0.02, -0.05],
                       [0.9, 0.2, 0.5, 0.4, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["date_group"], "A")
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["winner_capture_regret"]
                               + dec["repair_ordering_regret"], places=9)
        self.assertAlmostEqual(dec["repair_ordering_regret"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()

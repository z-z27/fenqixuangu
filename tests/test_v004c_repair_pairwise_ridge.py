# -*- coding: utf-8 -*-
"""v004c Repair-Strength Pairwise Ridge v001 — 测试 (§99-§112)。

覆盖:
- pair 构造 (§100): A=+10% B=+8% C=+6% D=-2% -> 6 个 oriented pairs;
- >7% 内部排序保留 (§101): 10%>8%>6% 不被 cap 成 tie;
- <7% 排序保留 (§102): 6%>4%>-3%;
- raw tie 排除 (§18);
- pair 权重 (§25, §103): 6 pairs -> 1/6 sum=1; 45 pairs -> 1/45 sum=1;
- pair type 无类别权重 (§22, §104): 同日跨阈值/非目标/目标内同权;
- gap 诊断 (§20);
- 4-type repair concordance + score tie 0.5 + raw tie exclude (§59-§61, §112);
- date-weighted 聚合 (§37, §61);
- walk-forward chronology (warmup 10 -> 4 OOF test dates) + determinism;
- lambda tie -> 更大 lambda (§39);
- exact 53 features / 禁列 (§105, §107);
- winner capture (§110) / regret identity + Group A/C 性质 (§111);
- control A/B evaluator 回归 (§109): 冻结 Top3 Failure Decomposition 数字;
- Oracle 口径 (§53); no July (§108); determinism (§113)。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

from src.v004c_pairwise_ridge import (  # noqa: E402
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    compute_practical_rank_metrics,
)
from src.v004c_repair_pairwise_ridge import (  # noqa: E402
    OOF_SCORE_COL,
    build_same_date_repair_pairs,
    chronological_walkforward_repair,
    date_weighted_repair_pair_stats,
    day_repair_pair_gaps,
    day_repair_pair_stats,
    repair_gap_stats,
    select_lambda_repair,
)
from src.v004c_top3_failure_decomposition import (  # noqa: E402
    decompose_day,
    oracle_ceiling_summary,
)

TOL = 1e-9


def make_repair_df(dates, seed=7, feat_dim=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        n = int(rng.integers(4, 10))
        raw = rng.normal(0.01, 0.04, n)
        b3 = rng.random(n) < 0.2
        for i in range(n):
            rows.append({
                "event_id": f"ev_{d}_{i}",
                "code": f"{i:06d}",
                "signal_date": d,
                "board_streak_before_break": 3 if b3[i] else 2,
                **{f"f{j}": float(rng.normal() + 2.0 * raw[i])
                   for j in range(feat_dim)},
                "repair_strength_raw_return": float(raw[i]),
            })
    return pd.DataFrame(rows)


def make_walk_dates(n_dates=14):
    """14 个递增日期 (warmup 10 -> 4 个 OOF test dates)。"""
    return [f"2026-0{(i // 9) + 5}-{(i % 9) + 1:02d}" for i in range(n_dates)]


def make_day(returns, scores, t7=None, d="2026-05-21"):
    """单日评价 frame (score + outcome)。"""
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


def pair_set(z_diff, p):
    """从 z_diff (one-hot x, board3=0) 还原 (high, low) 索引对集合。"""
    out = set()
    for row in z_diff:
        hi = int(np.argmax(row[:p]))
        lo = int(np.argmin(row[:p]))
        out.add((hi, lo))
    return out


class PairConstructionTestCase(unittest.TestCase):
    def test_four_stock_six_pairs(self):
        """§100: A=+10% B=+8% C=+6% D=-2% -> 6 oriented pairs。"""
        x = np.eye(4)
        raw = np.array([0.10, 0.08, 0.06, -0.02])
        b3 = np.zeros(4)
        z_diff, w, meta = build_same_date_repair_pairs(x, raw, b3)
        self.assertEqual(len(z_diff), 6)
        self.assertEqual(meta["n_pairs"], 6)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        np.testing.assert_allclose(w, 1.0 / 6.0)
        self.assertEqual(pair_set(z_diff, 4),
                         {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)})

    def test_above7_internal_ordering_retained(self):
        """§101: 10%>9%>8% 全部 >7%, 不被 cap 成 tie; A>B pair 存在。"""
        x = np.eye(3)
        raw = np.array([0.10, 0.09, 0.08])
        b3 = np.zeros(3)
        z_diff, w, meta = build_same_date_repair_pairs(x, raw, b3)
        self.assertEqual(len(z_diff), 3)
        self.assertEqual(pair_set(z_diff, 3), {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(meta["n_within_target7"], 3)

    def test_threshold_classification_below7(self):
        """10%>8%>6%: 6% 低于 7% 线 -> (10,8) within_t7, (10,6)/(8,6) cross。"""
        x = np.eye(3)
        raw = np.array([0.10, 0.08, 0.06])
        z_diff, _, meta = build_same_date_repair_pairs(x, raw, np.zeros(3))
        self.assertEqual(len(z_diff), 3)  # 排序保留, 无 cap 成 tie
        self.assertEqual(pair_set(z_diff, 3), {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(meta["n_within_target7"], 1)
        self.assertEqual(meta["n_cross_threshold"], 2)

    def test_below7_ordering_retained(self):
        """§102: 6%>4%>-3% -> 3 pairs。"""
        x = np.eye(3)
        raw = np.array([0.06, 0.04, -0.03])
        z_diff, _, meta = build_same_date_repair_pairs(x, raw, np.zeros(3))
        self.assertEqual(len(z_diff), 3)
        self.assertEqual(pair_set(z_diff, 3), {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(meta["n_within_non_target"], 3)

    def test_raw_tie_excluded(self):
        """§18: |raw_i - raw_j| <= 1e-12 -> no pair。"""
        x = np.eye(3)
        raw = np.array([0.10, 0.10, 0.05])
        z_diff, w, meta = build_same_date_repair_pairs(x, raw, np.zeros(3))
        self.assertEqual(len(z_diff), 2)  # 两个 10% 之间不 pair
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertEqual(pair_set(z_diff, 3), {(0, 2), (1, 2)})
        self.assertEqual(meta["n_cross_threshold"], 2)

    def test_pair_weight_six_and_45(self):
        """§103: 6 pairs -> 1/6 sum=1; 45 pairs -> 1/45 sum=1。"""
        x6 = np.eye(4)
        raw6 = np.array([0.10, 0.08, 0.06, -0.02])
        z6, w6, _ = build_same_date_repair_pairs(x6, raw6, np.zeros(4))
        self.assertEqual(len(z6), 6)
        np.testing.assert_allclose(w6, 1.0 / 6.0)
        self.assertAlmostEqual(float(w6.sum()), 1.0, places=12)
        x45 = np.eye(10)
        raw45 = np.arange(10, dtype=float) * 0.01 + 0.001
        z45, w45, _ = build_same_date_repair_pairs(x45, raw45, np.zeros(10))
        self.assertEqual(len(z45), 45)
        np.testing.assert_allclose(w45, 1.0 / 45.0)
        self.assertAlmostEqual(float(w45.sum()), 1.0, places=12)

    def test_no_pair_type_multipliers(self):
        """§22/§104: 同日 cross/within-non-target/within-target7 同权 1/N。"""
        x = np.eye(3)
        raw = np.array([0.10, 0.06, -0.03])  # (10,6) cross (10,-3) cross (6,-3) nt
        _, w, meta = build_same_date_repair_pairs(x, raw, np.zeros(3))
        self.assertEqual(meta["n_cross_threshold"], 2)
        self.assertEqual(meta["n_within_non_target"], 1)
        self.assertEqual(meta["n_within_target7"], 0)
        np.testing.assert_allclose(w, 1.0 / 3.0)  # 所有 type 同权
        x2 = np.eye(3)
        raw2 = np.array([0.10, 0.08, 0.06])
        _, w2, meta2 = build_same_date_repair_pairs(x2, raw2, np.zeros(3))
        self.assertEqual(meta2["n_within_target7"], 1)
        np.testing.assert_allclose(w2, 1.0 / 3.0)

    def test_gap_stats(self):
        """§20: gap 分布 + 小 gap 比例。"""
        gaps = day_repair_pair_gaps(np.array([0.10, 0.08, 0.06, -0.02]))
        self.assertEqual(len(gaps), 6)
        stats = repair_gap_stats(gaps)
        # gaps: [0.02, 0.04, 0.12, 0.02, 0.10, 0.08]
        self.assertAlmostEqual(stats["gap_median"], 0.06, places=9)
        self.assertAlmostEqual(stats["gap_max"], 0.12, places=9)
        self.assertAlmostEqual(stats["pct_gap_lt_0_5pp"], 0.0, places=9)
        self.assertAlmostEqual(stats["pct_gap_lt_1pp"], 0.0, places=9)
        # §20 只要求 <0.1pp / <0.5pp / <1pp 三档
        stats2 = repair_gap_stats(np.zeros(0))
        self.assertTrue(np.isnan(stats2["gap_median"]))


class RepairPairStatsTestCase(unittest.TestCase):
    def test_within_non_target_concordance(self):
        st = day_repair_pair_stats(np.array([0.9, 0.5, 0.1]),
                                   np.array([0.06, 0.04, -0.02]))
        self.assertEqual(st["within_non_target_concordance"], 1.0)
        self.assertEqual(st["n_within_non_target"], 3)
        self.assertIsNone(st["cross_concordance"])

    def test_reversed_zero(self):
        st = day_repair_pair_stats(np.array([0.1, 0.5, 0.9]),
                                   np.array([0.06, 0.04, -0.02]))
        self.assertEqual(st["within_non_target_concordance"], 0.0)

    def test_score_tie_half_credit(self):
        st = day_repair_pair_stats(np.array([0.5, 0.5]),
                                   np.array([0.06, 0.04]))
        self.assertEqual(st["within_non_target_concordance"], 0.5)

    def test_raw_tie_excluded_from_stats(self):
        st = day_repair_pair_stats(np.array([0.9, 0.8, 0.1]),
                                   np.array([0.10, 0.10, 0.05]))
        self.assertEqual(st["n_pairs"], 2)
        self.assertEqual(st["n_cross"], 2)

    def test_within_target7_pairs(self):
        st = day_repair_pair_stats(np.array([0.9, 0.8, 0.1]),
                                   np.array([0.10, 0.08, 0.06]))
        self.assertEqual(st["n_within_target7"], 1)  # (10,8)
        self.assertEqual(st["n_cross"], 2)  # (10,6), (8,6)
        self.assertIsNone(st["within_non_target_concordance"])

    def test_non_finite_scores_skip(self):
        st = day_repair_pair_stats(np.array([np.nan, 0.5, 0.1]),
                                   np.array([0.06, 0.04, -0.02]))
        self.assertIsNone(st)

    def test_date_weighted(self):
        d1 = day_repair_pair_stats(np.array([0.9, 0.5, 0.1]),
                                   np.array([0.06, 0.04, -0.02]))
        d2 = day_repair_pair_stats(np.array([0.1, 0.5, 0.9]),
                                   np.array([0.06, 0.04, -0.02]))
        agg = date_weighted_repair_pair_stats(
            np.array([0.9, 0.5, 0.1, 0.1, 0.5, 0.9]),
            np.array([0.06, 0.04, -0.02, 0.06, 0.04, -0.02]),
            np.array(["d1", "d1", "d1", "d2", "d2", "d2"]))
        self.assertEqual(agg["evaluable_dates"], 2)
        self.assertAlmostEqual(agg["within_non_target_concordance"], 0.5,
                               places=9)
        self.assertAlmostEqual(agg["all_concordance"], 0.5, places=9)
        self.assertTrue(np.isfinite(agg["pairwise_logloss"]))


class WalkForwardRepairTestCase(unittest.TestCase):
    def _df(self, n_dates=14, seed=31):
        return make_repair_df(make_walk_dates(n_dates), seed=seed)

    def test_chronology_and_oof_dates(self):
        df = self._df()
        res = chronological_walkforward_repair(
            df, ["f0", "f1", "f2"], "repair_strength_raw_return", 1.0)
        oof = res["oof"]
        scored = oof[oof[OOF_SCORE_COL].notna()]
        test_dates = sorted(df["signal_date"].unique())[MIN_TRAIN_SIGNAL_DATES:]
        self.assertEqual(set(scored["signal_date"]), set(test_dates))
        self.assertEqual(len(test_dates), 4)
        for _, r in res["fold_meta"].iterrows():
            fold_i = int(r["fold_index"])
            self.assertEqual(int(r["train_signal_dates"]),
                             MIN_TRAIN_SIGNAL_DATES + fold_i)
        # 最后 fold 有有限 repair logloss (同日 distinct raw -> 有 pairs)
        last = res["fold_meta"].iloc[-1]
        self.assertTrue(np.isfinite(last["test_repair_pairwise_logloss"]))
        self.assertGreater(int(res["fold_meta"]["train_pairs"].sum()), 0)

    def test_walkforward_deterministic(self):
        df = self._df()
        r1 = chronological_walkforward_repair(
            df, ["f0", "f1", "f2"], "repair_strength_raw_return", 1.0)
        r2 = chronological_walkforward_repair(
            df, ["f0", "f1", "f2"], "repair_strength_raw_return", 1.0)
        np.testing.assert_array_equal(
            r1["oof"][OOF_SCORE_COL].to_numpy(),
            r2["oof"][OOF_SCORE_COL].to_numpy())
        pd.testing.assert_frame_equal(r1["fold_meta"], r2["fold_meta"])

    def test_lambda_grid_and_selection(self):
        df = self._df()
        table, selected = select_lambda_repair(
            df, ["f0", "f1", "f2"], "repair_strength_raw_return")
        self.assertEqual(list(table["lambda"]), list(LAMBDA_GRID))
        self.assertIn(selected, LAMBDA_GRID)
        best_ll = float(table[table["lambda"] == selected]
                        ["date_weighted_oof_repair_pairwise_logloss"].iloc[0])
        # tie -> 更大 lambda (§39)
        self.assertEqual(selected, float(table[
            (table["date_weighted_oof_repair_pairwise_logloss"]
             <= best_ll + 1e-12)]["lambda"].max()))

    def test_builder_pairs_are_same_date_only(self):
        """§26: walk-forward train pairs 每日期独立构造 (跨日禁止)。"""
        df = make_repair_df(make_walk_dates(12), seed=5)
        res = chronological_walkforward_repair(
            df, ["f0", "f1", "f2"], "repair_strength_raw_return", 1.0)
        fold0 = res["fold_meta"].iloc[0]
        train = df[df["signal_date"] < fold0["test_signal_date"]]
        per_date = 0
        for _, day in train.groupby("signal_date"):
            raw = day["repair_strength_raw_return"].to_numpy()
            n = len(raw)
            per_date += sum(1 for i in range(n) for j in range(i + 1, n)
                            if abs(raw[i] - raw[j]) > 1e-12)
        self.assertEqual(int(fold0["train_pairs"]), per_date)


class WinnerCaptureAndRegretTestCase(unittest.TestCase):
    def test_winner_capture_two_of_three(self):
        """§110: 5 candidates / 3 Target7, Top3 抓到 2 -> capture 2/3。"""
        day = make_day([0.07, 0.07, 0.07, 0.02, -0.05],
                       [0.9, 0.2, 0.5, 0.4, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["top3_target7_hits"], 2)
        self.assertAlmostEqual(dec["available_winner_capture_rate"], 2 / 3,
                               places=9)
        self.assertEqual(dec["missed_target7_winners"], 1)

    def test_regret_identity_groupA_repair_zero(self):
        """§111: total = winner + repair; Group A winner-fixed 后 repair = 0。"""
        day = make_day([0.07, 0.07, 0.07, 0.02, -0.05],
                       [0.9, 0.2, 0.5, 0.4, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["date_group"], "A")
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["winner_capture_regret"]
                               + dec["repair_ordering_regret"], places=9)
        self.assertAlmostEqual(dec["winner_fixed_top3_return"], 0.07,
                               places=9)
        self.assertAlmostEqual(dec["repair_ordering_regret"], 0.0, places=9)

    def test_groupC_winner_regret_zero(self):
        day = make_day([0.06, 0.05, 0.02, -0.01, -0.07],
                       [0.9, 0.5, 0.3, 0.2, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["date_group"], "C")
        self.assertEqual(dec["winner_capture_regret"], 0.0)
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["repair_ordering_regret"], places=9)


class ContractAndIsolationTestCase(unittest.TestCase):
    def test_exact_53_features(self):
        fc = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
            / "v004c_pairwise_v1_feature_contract_v002.csv",
            encoding="utf-8-sig")
        names = fc.sort_values("feature_order")["feature_name"].tolist()
        self.assertEqual(len(names), 53)
        self.assertEqual(len(set(names)), 53)

    def test_forbidden_columns_absent_from_x(self):
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
                / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
                / "v004c_pairwise_v1_input_table_v002.csv"):
            df = pd.read_csv(path, encoding="utf-8-sig",
                             dtype={"code": str})
            self.assertLessEqual(df["signal_date"].max(), "2026-06-30")


class ControlEvaluatorRegressionTestCase(unittest.TestCase):
    """§109: 统一 evaluator 重算 Binary Control 必须复现冻结分解数字。"""

    def _control_frame(self):
        c = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_pairwise_ridge_v001_20260506_20260630"
            / "v004c_pairwise_ridge_oof_predictions_v001.csv",
            encoding="utf-8-sig", dtype={"code": str})
        fr = c.dropna(subset=["upside_oof_score"]).rename(
            columns={"upside_oof_score": "score"})
        self.assertEqual(len(fr), 233)
        self.assertEqual(fr["signal_date"].nunique(), 29)
        return fr

    def test_control_top3_matches_frozen_decomposition(self):
        """§109: 两种口径都复现冻结 review: practical Top3 排除 <3 候选日期
        (v001 口径 2.55%), decomposition Top3 用 K=min(3,n) 含全部 29 天
        (Top3 Failure Decomposition 口径 2.78%)。"""
        fr = self._control_frame()
        m = compute_practical_rank_metrics(fr)
        t3 = float(m[m["rank_position"] == "Top3"]
                   ["mean_capped_return"].iloc[0])
        self.assertAlmostEqual(t3, 0.0255, delta=0.0005)  # v001 口径 2.55%
        r1 = float(m[m["rank_position"] == 1]["mean_capped_return"].iloc[0])
        self.assertAlmostEqual(r1, 0.0282, delta=0.0005)  # 2.82%
        day_rows = [decompose_day(day, "score")
                    for _, day in fr.groupby("signal_date", sort=True)]
        self.assertEqual(len(day_rows), 29)
        dec_t3 = float(np.mean([r["model_top3_capped_return"]
                                for r in day_rows]))
        self.assertAlmostEqual(dec_t3, 0.0278, delta=0.0005)  # 分解口径 2.78%

    def test_control_capture_matches_frozen_decomposition(self):
        fr = self._control_frame()
        day_rows = [decompose_day(day, "score")
                    for _, day in fr.groupby("signal_date", sort=True)]
        win = [r for r in day_rows if r["target7_count"] > 0]
        self.assertEqual(len(win), 27)
        cap = float(np.mean([r["available_winner_capture_rate"]
                             for r in win]))
        self.assertAlmostEqual(cap, 0.5432, delta=0.01)  # 冻结 review 54.32%
        total = float(np.mean([r["total_top3_regret"] for r in day_rows]))
        self.assertAlmostEqual(total, 0.0330, delta=0.001)  # 3.30pp

    def test_control_oracle_matches_frozen_review(self):
        fr = self._control_frame()
        oc = oracle_ceiling_summary(fr)
        self.assertAlmostEqual(oc["oracle_top3_capped_return"], 0.0609,
                               delta=0.001)  # 冻结 review 6.09%
        self.assertAlmostEqual(oc["oracle_top1_hit_rate"], 0.9310,
                               delta=0.01)  # 93.10%


class DeterminismTestCase(unittest.TestCase):
    def test_build_pairs_deterministic(self):
        x = np.random.default_rng(3).normal(size=(8, 3))
        raw = np.array([0.10, 0.08, 0.06, 0.04, 0.02, 0.00, -0.02, -0.04])
        b3 = np.random.default_rng(4).random(8) < 0.3
        z1, w1, _ = build_same_date_repair_pairs(x, raw, b3)
        z2, w2, _ = build_same_date_repair_pairs(x, raw, b3)
        np.testing.assert_array_equal(z1, z2)
        np.testing.assert_array_equal(w1, w2)


if __name__ == "__main__":
    unittest.main()

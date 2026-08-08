# -*- coding: utf-8 -*-
"""v004c Date-Conditional Pairwise Ridge v001 — 数值正确性测试。

覆盖任务要求 (§76):
- same-date pairs only / no cross-date pairs;
- each pairable date total weight = 1;
- single-class date generates no pairs;
- train-only clipping / median imputation / standardization;
- board3 interaction 构造 (preprocessing 后, 不单独标准化);
- lambda penalty 行为 (更大 lambda -> 系数范数通常更小);
- gradient correctness (finite difference vs analytic);
- deterministic fitting;
- walk-forward train < test;
- exact 53 feature contract; labels / source_window / signal_date /
  outcome return columns absent from X;
- capped return 上限 +7%, 负收益不截断; target7-return consistency;
- rank1/top2/top3 return 计算; daily universe baseline; beat-universe rate。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

from src.v004c_pairwise_ridge import (  # noqa: E402
    BOARD3_STREAK,
    CAP_RETURN_7,
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    NOT_EVALUABLE_SINGLE_CLASS,
    TARGET7_THRESHOLD,
    _same_date_pair_stats,
    build_board3_interaction,
    build_same_date_pairs,
    chronological_walkforward,
    compute_capped_opportunity_return,
    compute_practical_rank_metrics,
    date_weighted_pair_stats,
    fit_pairwise_ridge,
    fit_preprocessor,
    pairwise_ridge_objective,
    score_from_z,
    select_lambda,
    transform_preprocessor,
)

TOL = 1e-6


def make_day(n, rng, feat_dim=3, pos_rate=0.4):
    """synthetic 单日数据: positive 的 feature 平均更高。"""
    x = rng.normal(size=(n, feat_dim))
    y = (rng.random(n) < pos_rate).astype(int)
    x[y == 1] += 0.6
    board3 = rng.random(n) < 0.2
    return x, y, board3


def make_df(dates, rng, feat_dim=3, pos_rate=0.4, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        n = int(rng.integers(4, 10))
        x, y, b3 = make_day(n, rng, feat_dim, pos_rate)
        for i in range(n):
            rows.append({
                "event_id": f"ev_{d}_{i}",
                "code": f"{i:06d}",
                "signal_date": d,
                "board_streak_before_break": 3 if b3[i] else 2,
                **{f"f{j}": x[i, j] for j in range(feat_dim)},
                "target": y[i],
                "capped_opportunity_return_7": 0.0,
            })
    df = pd.DataFrame(rows)
    df["target"] = df["target"].astype(float)
    return df


def make_walk_dates(n_dates=14):
    """14 个递增日期 (warmup 10 -> 4 个 OOF test dates)。"""
    return [f"2026-0{(i // 9) + 5}-{(i % 9) + 1:02d}" for i in range(n_dates)]


class SameDatePairsTestCase(unittest.TestCase):
    def test_same_date_only_positive_vs_negative(self):
        rng = np.random.default_rng(3)
        x, y, b3 = make_day(8, rng)
        z_diff, w = build_same_date_pairs(x, y, b3)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        self.assertEqual(len(z_diff), n_pos * n_neg)
        # 每对权重 = 1/(N_pos*N_neg) -> 该日总权重 = 1
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertAlmostEqual(float(w[0]), 1.0 / (n_pos * n_neg), places=9)
        # z_diff 结构: [scaled_x, b, b*scaled_x]
        self.assertEqual(z_diff.shape[1], 2 * x.shape[1] + 1)

    def test_single_class_date_no_pairs(self):
        rng = np.random.default_rng(4)
        x, _, b3 = make_day(6, rng)
        y_all_pos = np.ones(6, dtype=int)
        z_diff, w = build_same_date_pairs(x, y_all_pos, b3)
        self.assertEqual(len(z_diff), 0)
        y_all_neg = np.zeros(6, dtype=int)
        z_diff2, w2 = build_same_date_pairs(x, y_all_neg, b3)
        self.assertEqual(len(z_diff2), 0)

    def test_board3_interaction_shape_and_value(self):
        rng = np.random.default_rng(5)
        x, y, b3 = make_day(5, rng)
        z = build_board3_interaction(x, b3)
        p = x.shape[1]
        self.assertEqual(z.shape, (5, 2 * p + 1))
        np.testing.assert_allclose(z[:, :p], x)
        np.testing.assert_allclose(z[:, p], b3.astype(float))
        np.testing.assert_allclose(z[:, p + 1:], x * b3[:, None])
        # interaction 不重新标准化: z 列就是 scaled_feature * b
        scaled = transform_preprocessor(x, fit_preprocessor(x))
        z2 = build_board3_interaction(scaled, b3)
        np.testing.assert_allclose(z2[:, p + 1:], scaled * b3[:, None])


class PreprocessingTestCase(unittest.TestCase):
    def _synthetic(self):
        rng = np.random.default_rng(11)
        train = np.vstack([rng.normal(0, 1, (40, 3)),
                           np.array([[np.nan, np.nan, np.nan]])])
        test = np.array([[100.0, -100.0, np.nan]])
        return train, test

    def test_train_only_params(self):
        train, test = self._synthetic()
        params = fit_preprocessor(train)
        scaled = transform_preprocessor(test, params)
        # test 100 -> clip 到 train q99; -100 -> clip 到 train q01
        q99 = params.q99
        q01 = params.q01
        self.assertLessEqual(float(scaled[0, 0]), float(transform_preprocessor(
            np.array([[q99[0]]]), fit_preprocessor(train[:, :1]))[0, 0]) + TOL)
        # 均值中心化: train scaled 均值 ≈ 0
        train_scaled = transform_preprocessor(train, params)
        self.assertAlmostEqual(float(train_scaled[np.isfinite(train_scaled)].mean()),
                               0.0, places=6)
        # 恒等: 用同一 params 处理 train 应复现
        again = transform_preprocessor(train, params)
        np.testing.assert_allclose(train_scaled, again)

    def test_train_median_impute_nan(self):
        rng = np.random.default_rng(12)
        train = rng.normal(size=(30, 2))
        train[5, 0] = np.nan
        params = fit_preprocessor(train)
        med = float(np.nanmedian(train[:, 0]))
        scaled = transform_preprocessor(np.array([[np.nan, 5.0]]), params)
        expected0 = (med - params.mean[0]) / params.std[0]
        self.assertAlmostEqual(float(scaled[0, 0]), expected0, places=9)

    def test_constant_feature_scaled_zero(self):
        train = np.ones((10, 2))
        train[:, 1] = 7.0
        params = fit_preprocessor(train)
        self.assertTrue(params.constant_in_fold.all())
        scaled = transform_preprocessor(np.array([[9.0, 9.0]]), params)
        np.testing.assert_allclose(scaled, 0.0)
        self.assertAlmostEqual(float(params.mean[0]), 1.0, places=9)
        # std 0 -> constant 标记 (占位 std=1, transform 直接置 0)
        self.assertEqual(float(params.std[1]), 1.0)


class FitCorrectnessTestCase(unittest.TestCase):
    def test_fit_recovers_signal(self):
        """positive feature > negative feature -> 拟合后 positive score 更高。"""
        rng = np.random.default_rng(21)
        x = rng.normal(size=(60, 3))
        y = np.concatenate([np.ones(30), np.zeros(30)]).astype(int)
        x[:30] += 1.5
        b3 = np.zeros(60)
        z_diff, w = build_same_date_pairs(x, y, b3)
        theta = fit_pairwise_ridge(z_diff, w, 0.1)
        z = build_board3_interaction(x, b3)
        scores = score_from_z(z, theta)
        self.assertGreater(float(scores[:30].mean()),
                           float(scores[30:].mean()))

    def test_lambda_penalty_monotone(self):
        """更大 lambda -> 系数范数通常不更大 (§66, tolerance 内)。"""
        rng = np.random.default_rng(22)
        x = rng.normal(size=(50, 4))
        y = (rng.random(50) < 0.5).astype(int)
        b3 = rng.random(50) < 0.3
        z_diff, w = build_same_date_pairs(x, y, b3)
        norms = []
        for lam in (0.01, 1.0, 100.0):
            theta = fit_pairwise_ridge(z_diff, w, lam)
            norms.append(float(np.linalg.norm(theta)))
        self.assertLessEqual(norms[1], norms[0] + 0.2)
        self.assertLessEqual(norms[2], norms[1] + 0.2)

    def test_gradient_finite_difference(self):
        """analytic gradient vs finite difference (§67)。"""
        rng = np.random.default_rng(23)
        x = rng.normal(size=(40, 3))
        y = (rng.random(40) < 0.5).astype(int)
        b3 = rng.random(40) < 0.4
        z_diff, w = build_same_date_pairs(x, y, b3)
        lam = 0.7
        theta0 = rng.normal(size=z_diff.shape[1])
        _, grad = pairwise_ridge_objective(theta0, z_diff, w, lam)
        eps = 1e-6
        fd = np.zeros_like(theta0)
        for j in range(len(theta0)):
            plus = theta0.copy()
            minus = theta0.copy()
            plus[j] += eps
            minus[j] -= eps
            l_plus, _ = pairwise_ridge_objective(plus, z_diff, w, lam)
            l_minus, _ = pairwise_ridge_objective(minus, z_diff, w, lam)
            fd[j] = (l_plus - l_minus) / (2 * eps)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-5)

    def test_deterministic_fit(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=(40, 3))
        y = (rng.random(40) < 0.5).astype(int)
        b3 = rng.random(40) < 0.3
        z_diff, w = build_same_date_pairs(x, y, b3)
        t1 = fit_pairwise_ridge(z_diff, w, 1.0)
        t2 = fit_pairwise_ridge(z_diff, w, 1.0)
        np.testing.assert_array_equal(t1, t2)


class WalkForwardTestCase(unittest.TestCase):
    def _df(self, n_dates=14, seed=31):
        return make_df(make_walk_dates(n_dates), np.random.default_rng(seed),
                       feat_dim=3)

    def test_train_before_test_chronology(self):
        df = self._df()
        res = chronological_walkforward(df, ["f0", "f1", "f2"], "target", 1.0)
        oof = res["oof"]
        scored = oof[oof["target_oof_score"].notna()]
        # 只有 11th.. 日期有 OOF score
        test_dates = sorted(df["signal_date"].unique())[MIN_TRAIN_SIGNAL_DATES:]
        self.assertEqual(set(scored["signal_date"]), set(test_dates))
        # fold_meta: 每 fold train 日期数 = warmup + fold index, 且
        # train 日期严格早于 test 日期
        for _, r in res["fold_meta"].iterrows():
            fold_i = int(r["fold_index"])
            self.assertEqual(int(r["train_signal_dates"]),
                             MIN_TRAIN_SIGNAL_DATES + fold_i)
            self.assertGreaterEqual(int(r["train_rows"]), 0)

    def test_fold_meta_has_pairs(self):
        df = self._df()
        res = chronological_walkforward(df, ["f0", "f1", "f2"], "target", 1.0)
        self.assertGreater(int(res["fold_meta"]["train_pairs"].sum()), 0)

    def test_single_class_fold_marked(self):
        df = self._df()
        # 强制最后一个 test date 全为 0 类
        last = sorted(df["signal_date"].unique())[-1]
        df.loc[df["signal_date"] == last, "target"] = 0.0
        res = chronological_walkforward(df, ["f0", "f1", "f2"], "target", 1.0)
        last_fold = res["fold_meta"].iloc[-1]
        self.assertTrue(np.isnan(last_fold["test_pairwise_logloss"]))

    def test_walkforward_deterministic(self):
        df = self._df()
        r1 = chronological_walkforward(df, ["f0", "f1", "f2"], "target", 1.0)
        r2 = chronological_walkforward(df, ["f0", "f1", "f2"], "target", 1.0)
        np.testing.assert_array_equal(
            r1["oof"]["target_oof_score"].to_numpy(),
            r2["oof"]["target_oof_score"].to_numpy())

    def test_single_class_date_stats(self):
        # 单日单类: 单日统计返回 NOT_EVALUABLE_SINGLE_CLASS
        ll, auc_v, acc_v, _ = _same_date_pair_stats(
            np.array([0.1, 0.2]), np.array([1, 1]))
        self.assertEqual(ll, NOT_EVALUABLE_SINGLE_CLASS)
        self.assertIsNone(auc_v)
        # 全部日期单类 -> 聚合 = NaN, evaluable dates = 0
        ll2, auc2, acc2, nd2 = date_weighted_pair_stats(
            np.array([0.1, 0.2]), np.array([1, 1]), np.array(["d1", "d1"]))
        self.assertTrue(np.isnan(ll2))
        self.assertEqual(nd2, 0)

    def test_lambda_tie_prefers_larger(self):
        """tie (相同 logloss) -> 更大 lambda (§25)。"""
        df = self._df()
        table, selected = select_lambda(df, ["f0", "f1", "f2"], "target")
        self.assertEqual(list(table["lambda"]), list(LAMBDA_GRID))
        self.assertIn(selected, LAMBDA_GRID)
        # selected = 最小 logloss 中更大的 lambda
        best_ll = float(table[table["lambda"] == selected]
                        ["date_weighted_oof_pairwise_logloss"].iloc[0])
        self.assertEqual(selected, float(table[
            (table["date_weighted_oof_pairwise_logloss"] <= best_ll + 1e-12)]
            ["lambda"].max()))


class OutcomeTestCase(unittest.TestCase):
    def test_cap_upper_and_no_lower_clip(self):
        out = compute_capped_opportunity_return(
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([11.2, 10.8, 10.6, 9.2]),  # +12%, +8%, +6%, -8%
            np.array([1.0, 1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(out["capped_opportunity_return_7"].iloc[0]),
                               CAP_RETURN_7, places=9)
        self.assertAlmostEqual(float(out["capped_opportunity_return_7"].iloc[1]),
                               CAP_RETURN_7, places=9)
        self.assertAlmostEqual(float(out["capped_opportunity_return_7"].iloc[2]),
                               0.06, places=9)
        self.assertAlmostEqual(float(out["capped_opportunity_return_7"].iloc[3]),
                               -0.08, places=9)

    def test_target7_return_consistency_fatal(self):
        with self.assertRaises(RuntimeError):
            compute_capped_opportunity_return(
                np.array([10.0]), np.array([11.0]),
                np.array([0.0]))  # raw=+10% 但标签 0 -> FATAL

    def test_consistency_ok(self):
        compute_capped_opportunity_return(
            np.array([10.0, 10.0]), np.array([10.6, 10.7]),
            np.array([0.0, 1.0]))  # +6% -> 0, +7% -> 1


class PracticalMetricsTestCase(unittest.TestCase):
    def _oof(self):
        """构造 2 个日期的 OOF 表 (含 warmup-free, 全部有 score)。"""
        rows = [
            # day1: score 高者 capped return 高
            {"signal_date": "2026-05-21", "event_id": "a",
             "score": 0.9, "capped_opportunity_return_7": 0.06},
            {"signal_date": "2026-05-21", "event_id": "b",
             "score": 0.2, "capped_opportunity_return_7": 0.02},
            {"signal_date": "2026-05-21", "event_id": "c",
             "score": -0.1, "capped_opportunity_return_7": -0.05},
            # day2: 3 只
            {"signal_date": "2026-05-22", "event_id": "d",
             "score": 0.5, "capped_opportunity_return_7": 0.07},
            {"signal_date": "2026-05-22", "event_id": "e",
             "score": 0.0, "capped_opportunity_return_7": 0.01},
            {"signal_date": "2026-05-22", "event_id": "f",
             "score": -0.4, "capped_opportunity_return_7": -0.02},
        ]
        return pd.DataFrame(rows)

    def test_rank_and_top_returns(self):
        oof = self._oof()
        m = compute_practical_rank_metrics(oof)
        r1 = m[m["rank_position"] == 1].iloc[0]
        # Rank1: day1=0.06, day2=0.07 -> mean 0.065
        self.assertAlmostEqual(r1["mean_capped_return"], 0.065, places=9)
        t3 = m[m["rank_position"] == "Top3"].iloc[0]
        # Top3: day1 需要 3 只 (0.06+0.02-0.05)/3=0.01; day2 (0.07+0.01-0.02)/3=0.02
        self.assertAlmostEqual(t3["mean_capped_return"], 0.015, places=9)

    def test_baseline_and_beat_rate(self):
        oof = self._oof()
        m = compute_practical_rank_metrics(oof)
        r1 = m[m["rank_position"] == 1].iloc[0]
        # day1 base=(0.06+0.02-0.05)/3=0.01, day2 base=(0.07+0.01-0.02)/3=0.02
        # Rank1 excess = ((0.06-0.01)+(0.07-0.02))/2 = 0.05
        self.assertAlmostEqual(r1["mean_excess_vs_daily_universe"], 0.05, places=9)
        # beat rate: 0.06>0.01 ✓, 0.07>0.02 ✓ -> 1.0
        self.assertAlmostEqual(r1["beat_daily_universe_rate"], 1.0, places=9)
        # Rank5 无可用日期 (每天最多 3 只)
        r5 = m[m["rank_position"] == 5].iloc[0]
        self.assertEqual(int(r5["number_of_dates_available"]), 0)
        self.assertTrue(np.isnan(r5["mean_capped_return"]))

    def test_daily_universe_baseline(self):
        oof = self._oof()
        base = float(oof.groupby("signal_date")["capped_opportunity_return_7"]
                     .mean().mean())
        self.assertAlmostEqual(base, 0.015, places=9)  # (0.01+0.02)/2

    def test_builder_day_rank_is_per_date(self):
        """OOF 排名必须按 signal_date 分组 (跨日期不混合), score 降序 + id 升序。"""
        import importlib.util
        import sys as _sys
        spec = importlib.util.spec_from_file_location(
            "ridge_builder", BASE / "tools" / "build_v004c_pairwise_ridge_v001.py")
        b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b)
        scores = np.array([0.5, 0.9, 0.1, 0.8, 0.3, 0.7])
        ids = np.array(["a", "b", "c", "d", "e", "f"])
        dates = np.array(["2026-05-21", "2026-05-21", "2026-05-21",
                          "2026-05-22", "2026-05-22", "2026-05-22"])
        rank = b._day_rank(scores, ids, dates)
        # day1: 0.9>0.5>0.1 -> [2,1,3]; day2: 0.8>0.7>0.3 -> [1,3,2]
        np.testing.assert_array_equal(rank, [2, 1, 3, 1, 3, 2])


class ContractExposureTestCase(unittest.TestCase):
    def test_53_features(self):
        """读取真实 feature contract (v002), 断言 53 且无重复。"""
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
                     "raw_opportunity_return", "capped_opportunity_return_7",
                     "source_window", "signal_date", "event_id", "code",
                     "board_streak_before_break"}
        self.assertEqual(names & forbidden, set())


if __name__ == "__main__":
    unittest.main()

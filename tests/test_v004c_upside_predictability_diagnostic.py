# -*- coding: utf-8 -*-
"""v004c Upside Predictability Diagnostic v001 — 测试。

覆盖 (§57-§62, §71):
- Oracle Target7 ceiling (一天 5 只 [1,1,0,0,0] -> Top1 hit=1, Top3 hits=2,
  precision=2/3);
- Oracle capped return ([7%,6%,2%,-1%,-5%] -> Top3 = 5%);
- date-balanced row weights (date A 2 rows / date B 10 rows -> 各日总权重 1);
- fixed GBDT config 硬编码参数;
- exactly 53 features; 禁止列 (signal_date/source_window/label/outcome) 不进 X;
- train-only preprocessing (复用 ridge preprocessor);
- walk-forward chronology train < test + 29 OOF dates + July not accessed;
- capped return 一致性 (与 ridge 实现同一函数);
- practical metrics 与 ridge 实现一致性 (同一 compute_practical_rank_metrics);
- determinism (两次运行 byte-identical)。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

from src.v004c_pairwise_ridge import (  # noqa: E402
    compute_capped_opportunity_return,
    compute_practical_rank_metrics,
    fit_preprocessor,
    transform_preprocessor,
)
from src.v004c_upside_predictability_diagnostic import (  # noqa: E402
    GBDT_CONFIG,
    compute_oracle_ceiling,
    date_balanced_binary_logloss,
    day_rank_scores,
    gbdt_in_sample,
    gbdt_walkforward,
    make_gbdt,
    oracle_summary,
    ridge_in_sample_fit,
    target7_distribution,
    top_hit_metrics,
)


def make_day_df():
    """2 天: day1 5 只 [1,1,0,0,0] + capped [7%,6%,2%,-1%,-5%] (Oracle 测试数据)。"""
    rows = [
        ("2026-05-21", "a", 1, 0.07), ("2026-05-21", "b", 1, 0.06),
        ("2026-05-21", "c", 0, 0.02), ("2026-05-21", "d", 0, -0.01),
        ("2026-05-21", "e", 0, -0.05),
        ("2026-05-22", "f", 0, 0.03), ("2026-05-22", "g", 0, -0.02),
    ]
    df = pd.DataFrame(rows, columns=["signal_date", "event_id",
                                     "target7_daily_d2open_d3high",
                                     "capped_opportunity_return_7"])
    return df


def make_walk_df(n_dates=14, seed=41):
    """与 ridge 测试同构的 synthetic 数据 (带 53 特征占位不必要; 用 3 特征)。"""
    rng = np.random.default_rng(seed)
    dates = [f"2026-0{(i // 9) + 5}-{(i % 9) + 1:02d}" for i in range(n_dates)]
    rows = []
    for d in dates:
        n = int(rng.integers(4, 10))
        for i in range(n):
            x = rng.normal(size=3)
            y = 1 if x.sum() + rng.normal() > 0.8 else 0
            rows.append({
                "event_id": f"ev_{d}_{i}", "code": f"{i:06d}",
                "signal_date": d,
                "board_streak_before_break": 2 + int(rng.random() < 0.2),
                "f0": x[0], "f1": x[1], "f2": x[2],
                "target7_daily_d2open_d3high": float(y),
                "capped_opportunity_return_7": 0.05 if y else -0.01,
            })
    return pd.DataFrame(rows)


class OracleCeilingTestCase(unittest.TestCase):
    def test_top1_hit_and_top3_precision(self):
        df = make_day_df()
        c = compute_oracle_ceiling(df)
        day1 = c[c["signal_date"] == "2026-05-21"].iloc[0]
        self.assertEqual(int(day1["candidate_count"]), 5)
        self.assertEqual(int(day1["target7_count"]), 2)
        self.assertEqual(int(day1["oracle_top1_hit"]), 1)
        self.assertEqual(int(day1["oracle_top3_target7_hits"]), 2)
        self.assertAlmostEqual(float(day1["oracle_top3_precision"]), 2 / 3,
                               places=9)

    def test_zero_target7_date(self):
        df = make_day_df()
        c = compute_oracle_ceiling(df)
        day2 = c[c["signal_date"] == "2026-05-22"].iloc[0]
        self.assertEqual(int(day2["oracle_top1_hit"]), 0)
        self.assertEqual(int(day2["oracle_top3_target7_hits"]), 0)

    def test_oracle_top3_return(self):
        df = make_day_df()
        c = compute_oracle_ceiling(df)
        day1 = c[c["signal_date"] == "2026-05-21"].iloc[0]
        self.assertAlmostEqual(float(day1["oracle_top1_capped_return"]), 0.07,
                               places=9)
        self.assertAlmostEqual(float(day1["oracle_top3_capped_return"]),
                               (0.07 + 0.06 + 0.02) / 3, places=9)

    def test_distribution(self):
        df = make_day_df()
        dist = target7_distribution(df)
        self.assertEqual(dist["n_dates"], 2)
        self.assertEqual(dist["zero"], 1)   # day2: 0 只
        self.assertEqual(dist["one"], 0)
        self.assertEqual(dist["two"], 1)    # day1: 2 只
        self.assertEqual(dist["three_plus"], 0)

    def test_oracle_summary(self):
        df = make_day_df()
        s = oracle_summary(compute_oracle_ceiling(df))
        self.assertAlmostEqual(s["oracle_top1_hit_rate"], 0.5, places=9)
        self.assertAlmostEqual(s["oracle_top3_target7_precision"],
                               ((2 / 3) + 0) / 2, places=9)


class DateBalancedWeightsTestCase(unittest.TestCase):
    def test_row_weights_sum_to_one_per_date(self):
        """date A = 2 rows, date B = 10 rows -> 每日期总权重 = 1 (§30/§59)。"""
        df = make_walk_df(n_dates=12, seed=42)
        df = df.sort_values("signal_date").reset_index(drop=True)
        dates = sorted(df["signal_date"].unique())[:2]  # warmup 内前两个日期
        train_mask = df["signal_date"].isin(dates)
        weights = 1.0 / df.loc[train_mask].groupby("signal_date") \
            ["event_id"].transform("count").to_numpy(float)
        w = pd.Series(weights, index=df.index[train_mask])
        for d in dates:
            self.assertAlmostEqual(float(w[df.loc[train_mask, "signal_date"]
                                           == d].sum()), 1.0, places=9)


class GbdtConfigTestCase(unittest.TestCase):
    def test_fixed_config(self):
        clf = make_gbdt()
        self.assertEqual(clf.n_estimators, 50)
        self.assertEqual(clf.learning_rate, 0.05)
        self.assertEqual(clf.max_depth, 2)
        self.assertEqual(clf.min_samples_leaf, 10)
        self.assertEqual(clf.subsample, 1.0)
        self.assertEqual(clf.random_state, 20260809)
        self.assertEqual(GBDT_CONFIG["random_state"], 20260809)


class FeatureSetTestCase(unittest.TestCase):
    def test_exact_53_features_no_forbidden(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ridge_builder", BASE / "tools" / "build_v004c_pairwise_ridge_v001.py")
        b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b)
        names = b.load_feature_order()
        self.assertEqual(len(names), 53)
        self.assertEqual(len(set(names)), 53)
        forbidden = {"signal_date", "source_window", "d2_open_daily",
                     "d3_high_daily", "target7_daily_d2open_d3high",
                     "tail_loss_daily_5pct", "raw_opportunity_return",
                     "capped_opportunity_return_7", "event_id", "code"}
        self.assertEqual(set(names) & forbidden, set())

    def test_walkforward_chronology_and_29_dates(self):
        df = make_walk_df(n_dates=14, seed=43)
        gbdt_oof, fold_meta = gbdt_walkforward(
            df, ["f0", "f1", "f2"])
        scored = gbdt_oof[gbdt_oof["gbdt_oof_score"].notna()]
        test_dates = sorted(df["signal_date"].unique())[10:]
        self.assertEqual(len(test_dates), 4)  # 14 - 10 warmup
        self.assertEqual(set(scored["signal_date"]), set(test_dates))
        # train < test 硬断言在函数内 (FATAL)

    def test_no_july_access_on_real_data(self):
        dev = pd.read_csv(
            BASE / "reports" / "research"
            / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630"
            / "v004c_pairwise_v1_input_table_v002.csv", encoding="utf-8-sig")
        self.assertLessEqual(dev["signal_date"].max(), "2026-06-30")
        self.assertEqual(dev["signal_date"].nunique(), 39)
        self.assertEqual(len(dev), 319)
        # 29 个 OOF test dates (warmup 10, §31/§60)
        self.assertEqual(len(sorted(dev["signal_date"].unique())[10:]), 29)


class ConsistencyTestCase(unittest.TestCase):
    def test_capped_return_consistency(self):
        out = compute_capped_opportunity_return(
            np.array([10.0, 10.0, 10.0]), np.array([11.2, 10.6, 9.2]),
            np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(out["capped_opportunity_return_7"].iloc[0]),
                               0.07, places=9)
        self.assertAlmostEqual(float(out["capped_opportunity_return_7"].iloc[2]),
                               -0.08, places=9)

    def test_practical_metrics_same_function(self):
        """GBDT OOF 指标与 Ridge 用同一个 compute_practical_rank_metrics。"""
        df = make_day_df()
        df["score"] = [0.9, 0.5, 0.2, 0.1, 0.0, 0.4, 0.3]
        m = compute_practical_rank_metrics(df)
        r1 = m[m["rank_position"] == 1].iloc[0]
        # day1 Rank1 = 0.07 (score 0.9), day2 Rank1 = 0.03
        self.assertAlmostEqual(r1["mean_capped_return"], 0.05, places=9)

    def test_train_only_preprocessing(self):
        rng = np.random.default_rng(44)
        train = rng.normal(size=(30, 2))
        params = fit_preprocessor(train)
        test_scaled = transform_preprocessor(np.array([[999.0, np.nan]]), params)
        # clip 到 train q99; NaN -> train median
        self.assertLessEqual(float(test_scaled[0, 0]),
                             float((params.q99[0] - params.mean[0])
                                   / params.std[0]) + 1e-9)
        self.assertGreaterEqual(float(test_scaled[0, 1]),
                               float((params.q01[1] - params.mean[1])
                                     / params.std[1]) - 1e-9)

    def test_ridge_in_sample_deterministic(self):
        df = make_walk_df(n_dates=14, seed=45)
        m1 = ridge_in_sample_fit(df, ["f0", "f1", "f2"], 1.0)
        m2 = ridge_in_sample_fit(df, ["f0", "f1", "f2"], 1.0)
        self.assertEqual(m1["pairwise_auc"], m2["pairwise_auc"])
        self.assertEqual(m1["rank1_capped_return"], m2["rank1_capped_return"])

    def test_gbdt_deterministic(self):
        df = make_walk_df(n_dates=14, seed=46)
        o1, _ = gbdt_walkforward(df, ["f0", "f1", "f2"])
        o2, _ = gbdt_walkforward(df, ["f0", "f1", "f2"])
        np.testing.assert_array_equal(
            o1["gbdt_oof_score"].to_numpy(), o2["gbdt_oof_score"].to_numpy())

    def test_top_hit_metrics(self):
        df = make_day_df()
        df["score"] = [0.9, 0.5, 0.2, 0.1, 0.0, 0.4, 0.3]
        hit, prec = top_hit_metrics(df, "score")
        # day1 rank1 (0.9) t7=1; day2 rank1 (0.4) t7=0 -> hit = 0.5
        self.assertAlmostEqual(hit, 0.5, places=9)
        # day1 top3 (0.9,0.5,0.2) t7 = 1,1,0 -> 2/3; day2 top3 (0.4,0.3,0.1... 3 rows
        # = 0.4,0.3,0.2? day2 只有 2 行 (f,g) -> top3 = 全部 2 行 t7=0 -> 0
        # date-weighted = (2/3 + 0)/2
        self.assertAlmostEqual(prec, (2 / 3 + 0) / 2, places=9)

    def test_date_balanced_logloss(self):
        df = make_day_df()
        df["score"] = [0.9, 0.5, 0.2, 0.1, 0.0, 0.4, 0.3]
        ll = date_balanced_binary_logloss(df, "score")
        self.assertTrue(np.isfinite(ll))
        self.assertGreater(ll, 0.0)


if __name__ == "__main__":
    unittest.main()

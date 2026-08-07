"""v004c M0/M1/M2 Logistic expanding-date walk-forward 测试。

覆盖任务要求 (§50~§59):
- date leakage: test date 严格不在 train (max(train.signal_date) < test)
- same-date atomic: 同一 signal_date 全部候选要么全 train 要么全 test
- future-X invariance: 修改未来日期 X 后更早 fold 系数/概率不变
- future-target invariance: 修改未来日期 Target 后更早 fold fit/预测不变
- fold transform 顺序: q01/q99 -> clip -> clipped mean/std (人工极端值区分新旧实现)
- M0: [1,0,1,0] -> p=0.5, 所有 test candidate m0_probability = 0.5
- 模型成员: M1 = 4 factors, M2 = 7 factors; 无 POS7/TREND/SUPPLY
- REGIME: factor 输入保持 0/1, 不被 z-score
- deterministic ranking: 平局用 event_id 升序, 重复运行一致
- deterministic rebuild: 全 June walk-forward 两次运行 6 资产字节级一致

June 开发窗口 (只读 June): July rows 不进入任何计算。
"""

import filecmp
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_factor_spec import (
    CONTINUOUS_PRIMITIVES,
    CORE_FACTORS,
    M1_FACTORS,
    M2_FACTORS,
    SENSITIVITY_FACTORS,
    construct_factors,
)
from src.v004c_logistic_walkforward import (
    LOGISTIC_KWARGS,
    OUTPUT_FILES,
    TARGET_COLUMN,
    WalkForwardError,
    brier_score,
    daily_rank,
    fit_fold_transform,
    fit_m0,
    first_oof_index,
    load_june_data,
    logloss,
    run_walkforward,
    signal_dates,
    split_by_date,
    write_reports,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE_CSV = (REPO_ROOT
                   / "reports/research/v004c_model_table_v001_20260601_20260729"
                   / "v004c_model_table_v001.csv")


def _df_with(**cols) -> pd.DataFrame:
    """构造包含全部 11 个 primitive 列的 DataFrame (其余列填 linspace)."""
    n = max(len(v) if isinstance(v, (list, np.ndarray, pd.Series)) else 1
            for v in cols.values())
    data = {p: np.linspace(1.0, 2.0, n) for p in CONTINUOUS_PRIMITIVES}
    data["board_streak_is_3"] = np.zeros(n)
    for k, v in cols.items():
        data[k] = v if isinstance(v, (list, np.ndarray, pd.Series)) else [v] * n
    return pd.DataFrame(data)


def synthetic_june_df(n_dates: int = 15, rows_per_date: int = 8,
                      seed: int = 42) -> pd.DataFrame:
    """合成 June 数据: 15 日期 x 8 候选, 全部 primitive 随机 N(0,1),
    REGIME 0/1, Target 0/1 (seed 固定 => 可复现)."""
    rng = np.random.default_rng(seed)
    dates = [f"2026-06-{d:02d}" for d in range(1, n_dates + 1)]
    codes = [f"{100000 + i:06d}" for i in range(rows_per_date)]
    rows = [{"event_id": f"{code}_{d}", "code": code, "signal_date": d}
            for d in dates for code in codes]
    df = pd.DataFrame(rows)
    n = len(df)
    rng2 = np.random.default_rng(seed)
    for p in CONTINUOUS_PRIMITIVES:
        df[p] = rng2.normal(0.0, 1.0, n)
    df["board_streak_is_3"] = rng2.integers(0, 2, n)
    df[TARGET_COLUMN] = rng2.integers(0, 2, n)
    return df


class JuneLoaderTests(unittest.TestCase):
    """June 唯一开发窗口: 只读 June, July 不进入任何计算."""

    def test_loads_only_june_rows(self):
        df = load_june_data(MODEL_TABLE_CSV)
        self.assertEqual(len(df), 173)
        self.assertEqual(df["signal_date"].nunique(), 21)
        self.assertEqual(df["signal_date"].min(), "2026-06-01")
        self.assertEqual(df["signal_date"].max(), "2026-06-30")
        # July rows 不得出现
        self.assertTrue((df["signal_date"] >= "2026-06-01").all())
        self.assertTrue((df["signal_date"] <= "2026-06-30").all())

    def test_target_parsed_to_int_and_ids_unique(self):
        df = load_june_data(MODEL_TABLE_CSV)
        self.assertEqual(sorted(df[TARGET_COLUMN].unique()), [0, 1])
        self.assertEqual(int(df[TARGET_COLUMN].sum()), 59)
        self.assertEqual(df["event_id"].nunique(), len(df))
        self.assertFalse(pd.api.types.is_numeric_dtype(df["code"]))  # 不转数值
        # 只含标识 + primitive + target 列
        expected = ["event_id", "code", "signal_date"] \
            + list(CONTINUOUS_PRIMITIVES) + ["board_streak_is_3", TARGET_COLUMN]
        self.assertEqual(df.columns.tolist(), expected)

    def test_window_outside_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            df = load_june_data(MODEL_TABLE_CSV)
            bad = df.iloc[[0]].copy()
            bad["signal_date"] = "2026-08-01"
            pd.concat([bad, df]).to_csv(p, index=False)
            with self.assertRaises(WalkForwardError):
                load_june_data(p)

    def test_sorted_deterministically(self):
        df = load_june_data(MODEL_TABLE_CSV)
        self.assertTrue((df["signal_date"].values ==
                         np.sort(df["signal_date"].values)).all())
        g = df.groupby("signal_date", sort=True)["event_id"]
        for _, ids in g:
            self.assertEqual(list(ids), sorted(ids))


class DateSplitTests(unittest.TestCase):
    """§50 date leakage + §51 same-date atomic split."""

    def setUp(self):
        self.df = synthetic_june_df()

    def test_split_train_strictly_before_test(self):
        train, test = split_by_date(self.df, "2026-06-12")
        self.assertLess(str(train["signal_date"].max()), "2026-06-12")
        self.assertTrue((test["signal_date"] == "2026-06-12").all())
        # 同一天所有候选要么全 train 要么全 test (atomic)
        self.assertEqual(set(test["signal_date"]), {"2026-06-12"})
        self.assertNotIn("2026-06-12", set(train["signal_date"]))

    def test_first_oof_index_meets_start_conditions(self):
        idx = first_oof_index(self.df)
        dates = signal_dates(self.df)
        self.assertGreaterEqual(idx, 10)
        d = dates[idx]
        prior = self.df[self.df["signal_date"] < d]
        self.assertGreaterEqual(prior["signal_date"].nunique(), 10)
        self.assertGreaterEqual(len(prior), 70)
        self.assertGreaterEqual(int((prior[TARGET_COLUMN] == 1).sum()), 15)
        self.assertGreaterEqual(int((prior[TARGET_COLUMN] == 0).sum()), 30)
        # 更早日期必须不满足 (首个满足点)
        for earlier in dates[:idx]:
            p = self.df[self.df["signal_date"] < earlier]
            if p["signal_date"].nunique() < 10 or len(p) < 70:
                continue
            self.assertTrue(int((p[TARGET_COLUMN] == 1).sum()) < 15
                            or int((p[TARGET_COLUMN] == 0).sum()) < 30)

    def test_real_june_first_oof_date(self):
        df = load_june_data(MODEL_TABLE_CSV)
        r = run_walkforward(df)
        self.assertEqual(r["first_test_date"], "2026-06-17")
        it = r["initial_train"]
        self.assertEqual((it["dates"], it["rows"], it["positive"],
                          it["negative"]), (12, 72, 20, 52))
        self.assertEqual(len(r["oof_dates"]), 9)
        self.assertEqual(r["oof_rows"], 101)

    def test_predictions_contain_only_oof_rows(self):
        df = synthetic_june_df()
        r = run_walkforward(df)
        pred_dates = sorted(r["predictions"]["signal_date"].unique())
        self.assertEqual(pred_dates, r["oof_dates"])
        # 每个 OOF 日期覆盖当天全部候选 (不塞入初始 training 行)
        for d in r["oof_dates"]:
            n_data = int((df["signal_date"] == d).sum())
            n_pred = int((r["predictions"]["signal_date"] == d).sum())
            self.assertEqual(n_pred, n_data)


class FoldTransformTests(unittest.TestCase):
    """§54 fold transform 顺序: q01/q99 -> clip -> clipped mean/std."""

    def test_clipped_statistics_not_raw(self):
        # 人工极端值区分顺序 A (clip 后拟合 mu/sigma) 与顺序 B (raw 拟合)
        raw = np.array([0.0, 1.0, 2.0, 3.0, 1000.0])
        train = _df_with(break_open_return=raw)
        params = fit_fold_transform(train)["break_open_return"]
        q01 = np.quantile(raw, 0.01)
        q99 = np.quantile(raw, 0.99)
        clipped = np.clip(raw, q01, q99)
        self.assertAlmostEqual(params["q01"], float(q01), places=12)
        self.assertAlmostEqual(params["q99"], float(q99), places=12)
        self.assertAlmostEqual(params["mu"], float(clipped.mean()), places=12)
        self.assertAlmostEqual(params["sigma"], float(clipped.std(ddof=0)),
                               places=12)
        self.assertNotAlmostEqual(params["mu"], float(raw.mean()), places=6)
        self.assertNotAlmostEqual(params["sigma"], float(raw.std(ddof=0)),
                                  places=6)

    def test_fold_params_only_from_train(self):
        # test 行的极端值不得改变 train 拟合的参数
        train = _df_with(d1_ma10_slope=np.linspace(-3.0, 3.0, 120))
        params = fit_fold_transform(train)
        test = _df_with(d1_ma10_slope=np.linspace(300.0, 400.0, 8))
        _ = construct_factors(test, params)  # 只用 train 参数 apply
        self.assertEqual(fit_fold_transform(train), params)

    def test_regime_stays_raw_01(self):
        # §57: REGIME (board_streak_is_3) 保持 0/1, 不被 z-score
        rng = np.random.default_rng(3)
        n = 60
        df = _df_with(board_streak_is_3=rng.integers(0, 2, n),
                      d1_ma10_slope=np.linspace(-5.0, 5.0, n))
        params = fit_fold_transform(df)
        f = construct_factors(df, params)
        self.assertTrue(set(np.unique(f["REGIME"])).issubset({0.0, 1.0}))
        np.testing.assert_array_equal(
            f["REGIME"], df["board_streak_is_3"].to_numpy(dtype=float))
        # 连续 factor 已标准化 (mean~0), REGIME 不应是标准化结果
        self.assertNotAlmostEqual(f["REGIME"].mean(), 0.0, places=4)
        self.assertTrue(-0.5 < f["REGIME"].mean() < 0.5)


class M0Tests(unittest.TestCase):
    """§55 M0: training-fold prevalence."""

    def test_m0_half(self):
        p, alpha = fit_m0(np.array([1, 0, 1, 0]))
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(alpha, 0.0)

    def test_m0_single_class_fails(self):
        with self.assertRaises(WalkForwardError):
            fit_m0(np.array([1, 1, 1]))
        with self.assertRaises(WalkForwardError):
            fit_m0(np.array([0, 0, 0]))

    def test_run_level_m0_equals_train_prevalence(self):
        df = synthetic_june_df()
        r = run_walkforward(df)
        for row in r["folds"].itertuples(index=False):
            m = r["predictions"]["signal_date"] == row.test_signal_date
            p = r["predictions"].loc[m, "m0_probability"].to_numpy()
            self.assertTrue(np.allclose(p, row.m0_probability, rtol=1e-12))
            self.assertAlmostEqual(row.m0_probability, row.train_base_rate,
                                   places=12)


class ModelMembersTests(unittest.TestCase):
    """§56 模型成员: M1/M2 严格等于规格成员, 无 SENSITIVITY/SUPPLY."""

    def test_m1_members(self):
        self.assertEqual(M1_FACTORS, ("OPEN", "RESET", "HIGHZONE", "LATESELL"))
        for f in SENSITIVITY_FACTORS:
            self.assertNotIn(f, M1_FACTORS)
        self.assertNotIn("SUPPLY", M1_FACTORS)

    def test_m2_members(self):
        self.assertEqual(M2_FACTORS, CORE_FACTORS)
        self.assertEqual(len(M2_FACTORS), 7)
        for f in SENSITIVITY_FACTORS:
            self.assertNotIn(f, M2_FACTORS)
        self.assertNotIn("SUPPLY", M2_FACTORS)
        self.assertNotIn("POS7", M2_FACTORS)
        self.assertNotIn("TREND", M2_FACTORS)

    def test_logistic_kwargs_fixed(self):
        self.assertEqual(LOGISTIC_KWARGS["penalty"], "l2")
        self.assertEqual(LOGISTIC_KWARGS["C"], 1.0)
        self.assertEqual(LOGISTIC_KWARGS["solver"], "lbfgs")
        self.assertEqual(LOGISTIC_KWARGS["fit_intercept"], True)
        self.assertIsNone(LOGISTIC_KWARGS["class_weight"])
        self.assertGreaterEqual(LOGISTIC_KWARGS["max_iter"], 1000)


class InvarianceTests(unittest.TestCase):
    """§52 future-X invariance + §53 future-target invariance."""

    def _compare_earlier(self, r1, r2, before_date):
        c1 = r1["coefficients"]
        c2 = r2["coefficients"]
        m1 = c1[c1["test_signal_date"] < before_date].reset_index(drop=True)
        m2 = c2[c2["test_signal_date"] < before_date].reset_index(drop=True)
        pd.testing.assert_frame_equal(m1, m2, check_exact=True)
        p1 = r1["predictions"]
        p2 = r2["predictions"]
        q1 = p1[p1["signal_date"] < before_date].reset_index(drop=True)
        q2 = p2[p2["signal_date"] < before_date].reset_index(drop=True)
        pd.testing.assert_frame_equal(q1, q2, check_exact=True)
        f1 = r1["folds"]
        f2 = r2["folds"]
        h1 = f1[f1["test_signal_date"] < before_date].reset_index(drop=True)
        h2 = f2[f2["test_signal_date"] < before_date].reset_index(drop=True)
        pd.testing.assert_frame_equal(h1, h2, check_exact=True)

    def test_future_x_invariance(self):
        df = synthetic_june_df()
        r1 = run_walkforward(df)
        df2 = df.copy()
        dates = signal_dates(df)
        # 修改一个中间 OOF 日期 (非首个) 的 X 为极端值
        future = dates[12]
        mask = df2["signal_date"] == future
        df2.loc[mask, "break_open_return"] = 1e6
        r2 = run_walkforward(df2)
        self._compare_earlier(r1, r2, future)
        # 修改生效证明: 被改日期自身是 test 行 (不进自身 fold 的 training),
        # 但它是后续 fold 的 training 行 -> 最后一个 fold 的系数必然变化
        last_date = dates[-1]
        c1 = r1["coefficients"][r1["coefficients"]["test_signal_date"] == last_date]
        c2 = r2["coefficients"][r2["coefficients"]["test_signal_date"] == last_date]
        self.assertFalse(c1["coefficient"].equals(c2["coefficient"]))

    def test_future_target_invariance(self):
        df = synthetic_june_df()
        r1 = run_walkforward(df)
        df2 = df.copy()
        dates = signal_dates(df)
        future = dates[12]
        mask = df2["signal_date"] == future
        idx = np.flatnonzero(mask.to_numpy())[0]
        df2.loc[df2.index[idx], TARGET_COLUMN] = \
            1 - int(df2.loc[df2.index[idx], TARGET_COLUMN])
        r2 = run_walkforward(df2)
        self._compare_earlier(r1, r2, future)
        # 更早 fold 的预测不变, 被修改行本身属于 future 不参与更早 fold
        self.assertEqual(len(r1["predictions"]), len(r2["predictions"]))


class RankingTests(unittest.TestCase):
    """§58 deterministic ranking (tie-break event_id 升序)."""

    def test_daily_rank_ties_broken_by_event_id(self):
        df = pd.DataFrame({
            "event_id": ["c_01", "a_01", "b_01", "e_01", "d_01"],
            "p": [0.5, 0.9, 0.5, 0.5, 0.9],
        })
        r1 = daily_rank(df, "p")
        r2 = daily_rank(df, "p")
        pd.testing.assert_series_equal(r1, r2)
        expected = {"a_01": 1, "d_01": 2, "b_01": 3, "c_01": 4, "e_01": 5}
        # 按排序后的 index 对齐 event_id 与 rank (r1.index = 排序后原行位置)
        ordered_events = df.loc[r1.index, "event_id"].tolist()
        self.assertEqual(dict(zip(ordered_events, r1)), expected)
        # 打乱输入顺序结果不变 (确定性不依赖行序)
        df_shuffled = df.sample(frac=1.0, random_state=7)
        r3 = daily_rank(df_shuffled, "p")
        ordered_events3 = df_shuffled.loc[r3.index, "event_id"].tolist()
        self.assertEqual(dict(zip(ordered_events3, r3)), expected)

    def test_ranks_are_permutation_per_date(self):
        df = synthetic_june_df()
        r = run_walkforward(df)
        for d in r["oof_dates"]:
            sub = r["predictions"][r["predictions"]["signal_date"] == d]
            for col in ("m1_rank_daily", "m2_rank_daily"):
                self.assertEqual(sorted(sub[col]), list(range(1, len(sub) + 1)))


class MetricsTests(unittest.TestCase):
    """LogLoss/Brier 基本性质 + M0 结构化无判别."""

    def test_logloss_perfect_and_eps_clip(self):
        self.assertAlmostEqual(logloss(np.array([1.0, 0.0]),
                                       np.array([1.0, 0.0])), 0.0, places=12)
        # p=0 对 y=1: eps clip 到 1e-15 => -log(1e-15)
        self.assertAlmostEqual(logloss(np.array([1.0]), np.array([0.0])),
                               -np.log(1e-15), places=6)

    def test_brier(self):
        self.assertAlmostEqual(brier_score(np.array([1.0, 0.0]),
                                           np.array([0.75, 0.25])),
                               0.0625, places=12)

    def test_m0_metrics_structurally_none(self):
        df = synthetic_june_df()
        r = run_walkforward(df)
        m = r["metrics"].set_index("model")
        self.assertAlmostEqual(float(m.loc["M0", "auc"]), 0.5)
        self.assertAlmostEqual(float(m.loc["M0", "average_precision"]),
                               float(m.loc["M0", "observed_rate"]), places=12)
        self.assertEqual(m.loc["M0", "top1_target_rate"], "NA")
        self.assertEqual(m.loc["M0", "top3_target_rate"], "NA")
        self.assertEqual(m.loc["M0", "dates_with_top3_hit"], "NA")
        self.assertEqual(m.loc["M0", "zero_hit_dates"], "NA")


class DeterministicRebuildTests(unittest.TestCase):
    """§59 deterministic rebuild: 全 June 两次运行 6 资产字节级一致."""

    def test_real_june_byte_identical_rebuild(self):
        df = load_june_data(MODEL_TABLE_CSV)
        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "run1"
            d2 = Path(tmp) / "run2"
            r1 = run_walkforward(df, input_csv=MODEL_TABLE_CSV)
            write_reports(r1, d1)
            r2 = run_walkforward(df, input_csv=MODEL_TABLE_CSV)
            write_reports(r2, d2)
            for name in OUTPUT_FILES:
                self.assertTrue(filecmp.cmp(d1 / name, d2 / name,
                                            shallow=False), name)
        self.assertEqual(len(r1["predictions"]), 101)
        self.assertEqual(len(r1["folds"]), 9)
        self.assertEqual(len(r1["coefficients"]), 117)

    def test_real_june_status_is_stable(self):
        # 状态判定确定性 (不因两次运行改变); 结果本身由数据决定, 不做预设
        df = load_june_data(MODEL_TABLE_CSV)
        r1 = run_walkforward(df)
        r2 = run_walkforward(df)
        self.assertEqual(r1["status"], r2["status"])
        self.assertTrue(r1["status"] in
                        {"M0_ONLY", "M1_CANDIDATE", "M2_CANDIDATE",
                         "REVIEW_REQUIRED", "REJECT_NO_STABLE_MODEL"})
        self.assertFalse(r1["m1_failed_folds"])
        self.assertFalse(r1["m2_failed_folds"])
        self.assertEqual(r1["anomalies"], [])


class MetricsRangeTests(unittest.TestCase):
    """汇总表完整性: 3 个模型行, M0/M1/M2 概率均值落在合法区间."""

    def test_metrics_table_shape_and_ranges(self):
        df = synthetic_june_df()
        r = run_walkforward(df)
        m = r["metrics"]
        self.assertEqual(list(m["model"]), ["M0", "M1", "M2"])
        for col in ("logloss", "brier", "auc", "average_precision"):
            for _, row in m.iterrows():
                self.assertTrue(np.isfinite(row[col]))
        for pcol in ("m0_probability", "m1_probability", "m2_probability"):
            p = r["predictions"][pcol].to_numpy(dtype=float)
            self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
            self.assertTrue(np.all(np.isfinite(p)))


if __name__ == "__main__":
    unittest.main()

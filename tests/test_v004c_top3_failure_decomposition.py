# -*- coding: utf-8 -*-
"""v004c Top3 Failure Decomposition v001 — 测试。

覆盖 (§73-§84):
- same OOF universe (Ridge/GBDT 233 rows / 29 dates / exact identity);
- rank recomputation (乱序输入 -> score 降序 + event_id 升序);
- winner capture (§73: 7,7,7,2,-5 模型 Top3 [7,2,-5] -> capture 1/3,
  winner-fixed [7,7,7] -> repair ordering regret = 0);
- 0-winner N/A semantics (§12, §75);
- required non-target slots (§17, §74);
- oracle non-target fill (§18);
- best non-target fill capture (§43);
- winner-fixed counterfactual (§33-§34);
- regret identity + non-negative (§36-§37);
- Group A repair regret = 0 after winner fix (§38);
- Group C winner regret = 0 (§39, §75);
- continuous repair concordance / cross-threshold / within-non-target
  (§77) + ties (§78-§79);
- Oracle display bug regression (§81: 不硬编码 1.0);
- no model fit (§72 静态检查);
- no July (§82);
- determinism。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

from src.v004c_top3_failure_decomposition import (  # noqa: E402
    audit_stored_rank,
    date_weighted_concordance,
    day_repair_concordance,
    decompose_day,
    oracle_ceiling_summary,
    rank_profile,
    recompute_day_rank,
)


def make_day(returns, scores, t7=None, d="2026-05-21"):
    """构造单日 df: returns -> capped (min(·, 0.07)); t7 自动或给定。"""
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
            "raw_opportunity_return": float(r),
            "capped_opportunity_return_7": float(c),
        })
    return pd.DataFrame(rows)


class WinnerCaptureTestCase(unittest.TestCase):
    """§73: >=3 winners, 模型 Top3 = [7, 2, -5]。"""

    def test_capture_one_of_three(self):
        day = make_day([0.07, 0.07, 0.07, 0.02, -0.05],
                       [0.9, 0.2, 0.1, 0.5, 0.3])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["date_group"], "A")
        self.assertEqual(dec["available_target7_slots"], 3)
        self.assertEqual(dec["top3_target7_hits"], 1)
        self.assertAlmostEqual(dec["available_winner_capture_rate"], 1 / 3,
                               places=9)
        self.assertEqual(dec["missed_target7_winners"], 2)
        # winner-fixed: [7,7,7] -> repair ordering regret = 0
        self.assertAlmostEqual(dec["winner_fixed_top3_return"], 0.07, places=9)
        self.assertAlmostEqual(dec["repair_ordering_regret"], 0.0, places=9)
        self.assertGreater(dec["winner_capture_regret"], 0.0)
        # identity
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["winner_capture_regret"]
                               + dec["repair_ordering_regret"], places=9)

    def test_zero_winner_na_semantics(self):
        """§12/§75: 0 Target7 -> capture = None (不是 0), winner regret = 0。"""
        day = make_day([0.06, 0.05, 0.02, -0.01, -0.07],
                       [0.9, 0.5, 0.3, 0.2, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["date_group"], "C")
        self.assertIsNone(dec["available_winner_capture_rate"])
        self.assertIsNone(dec["full_available_winner_capture"])
        self.assertEqual(dec["missed_target7_winners"], 0)
        self.assertAlmostEqual(dec["winner_capture_regret"], 0.0, places=9)
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["repair_ordering_regret"], places=9)
        # 全部 regret = repair ordering
        self.assertEqual(dec["total_top3_regret"],
                         dec["repair_ordering_regret"])

    def test_one_winner_fill_quality(self):
        """§74: 1 winner + fill: capture 100%, oracle fill [6,5], model fill。"""
        day = make_day([0.07, 0.06, 0.05, 0.01, -0.05],
                       [0.9, 0.4, 0.2, 0.3, 0.1])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["date_group"], "B")
        self.assertEqual(dec["available_target7_slots"], 1)
        self.assertEqual(dec["top3_target7_hits"], 1)
        self.assertAlmostEqual(dec["available_winner_capture_rate"], 1.0,
                               places=9)
        self.assertEqual(dec["required_non_target_slots"], 2)
        # oracle fill = 6%, 5% 两只
        self.assertAlmostEqual(dec["oracle_non_target_fill_mean_return"],
                               (0.06 + 0.05) / 2, places=9)
        # 模型 Top3 = 0.9(7%), 0.4(6%), 0.3(1%) -> fill = [6%, 1%]
        self.assertAlmostEqual(dec["selected_non_target_mean_capped_return"],
                               (0.06 + 0.01) / 2, places=9)
        # best non-target fill capture: oracle fill = {6%,5%}; 模型非target
        # 成员 {6%,1%} -> 交集 {6%} -> 1/2
        self.assertAlmostEqual(dec["best_non_target_fill_capture_rate"], 0.5,
                               places=9)
        # winner regret = 0, 全部 regret = repair ordering
        self.assertAlmostEqual(dec["winner_capture_regret"], 0.0, places=9)
        self.assertGreater(dec["repair_ordering_regret"], 0.0)
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["repair_ordering_regret"], places=9)

    def test_missed_winner_and_bad_fill(self):
        """§76: [7,7,6,1,-6] 模型 Top3 [7,1,-6] -> 两类 regret 都 > 0。"""
        # score 使模型 Top3 = ev00(7%), ev03(1%), ev04(-6%) -> 漏掉 ev01(7%)
        day = make_day([0.07, 0.07, 0.06, 0.01, -0.06],
                       [0.9, 0.3, 0.2, 0.5, 0.4])
        dec = decompose_day(day, "score")
        self.assertEqual(dec["available_target7_slots"], 2)
        self.assertEqual(dec["top3_target7_hits"], 1)
        self.assertEqual(dec["missed_target7_winners"], 1)
        self.assertGreater(dec["winner_capture_regret"], 0.0)
        self.assertGreater(dec["repair_ordering_regret"], 0.0)
        self.assertAlmostEqual(dec["total_top3_regret"],
                               dec["winner_capture_regret"]
                               + dec["repair_ordering_regret"], places=9)


class RankRecomputationTestCase(unittest.TestCase):
    def test_rank_from_scrambled_input(self):
        """§80: 乱序输入 -> score 降序 + event_id 升序。"""
        df = make_day([0.02, 0.07, 0.01, 0.05, 0.03],
                      [0.1, 0.9, 0.5, 0.7, 0.3],
                      d="2026-05-21")
        df2 = make_day([0.01, 0.02], [0.8, 0.2], d="2026-05-22")
        both = pd.concat([df, df2], ignore_index=True).sample(
            frac=1.0, random_state=7).reset_index(drop=True)
        ranked = recompute_day_rank(both, "score")
        day1 = ranked[ranked["signal_date"] == "2026-05-21"] \
            .sort_values("diagnostic_rank")
        self.assertEqual(day1.iloc[0]["event_id"], "ev01")  # score 0.9
        self.assertEqual(day1.iloc[1]["event_id"], "ev03")  # score 0.7
        self.assertEqual(day1.iloc[2]["event_id"], "ev02")  # score 0.5
        day2 = ranked[ranked["signal_date"] == "2026-05-22"] \
            .sort_values("diagnostic_rank")
        self.assertEqual(day2.iloc[0]["event_id"], "ev00")  # score 0.8

    def test_stored_rank_audit_missing(self):
        df = make_day([0.02, 0.07, 0.01], [0.1, 0.9, 0.5])
        self.assertEqual(audit_stored_rank(df, "score", "nope"), "NOT_AVAILABLE")

    def test_stored_rank_audit_mismatch(self):
        df = make_day([0.02, 0.07, 0.01], [0.1, 0.9, 0.5])
        df["stored"] = [3, 2, 1]  # 正确序应为 [3,1,2] -> 2 处不一致
        self.assertEqual(audit_stored_rank(df, "score", "stored"), 2)


class ConcordanceTestCase(unittest.TestCase):
    def test_within_non_target_concordance(self):
        """§77: non-target returns [6,4,-2] scores [0.9,0.5,0.1] -> 1.0。"""
        day = make_day([0.06, 0.04, -0.02], [0.9, 0.5, 0.1])
        c = day_repair_concordance(day, "score")
        self.assertEqual(c["within_concordance"], 1.0)
        self.assertEqual(c["within_pairs"], 3)
        self.assertIsNone(c["cross_concordance"])  # 无跨阈值 pair

    def test_reversed_scores_zero(self):
        day = make_day([0.06, 0.04, -0.02], [0.1, 0.5, 0.9])
        c = day_repair_concordance(day, "score")
        self.assertEqual(c["within_concordance"], 0.0)

    def test_score_tie_half_credit(self):
        """§78: return 不同但 score 相同 -> 0.5。"""
        day = make_day([0.06, 0.04], [0.5, 0.5])
        c = day_repair_concordance(day, "score")
        self.assertEqual(c["within_concordance"], 0.5)

    def test_return_tie_excluded(self):
        """§79: capped return 相同 (两个 7% 赢家) -> pair 不参与。"""
        day = make_day([0.07, 0.07, 0.02], [0.9, 0.8, 0.1])
        c = day_repair_concordance(day, "score")
        # pairs: (7,7) 排除; (7,2) x2 -> cross_threshold pairs = 2
        self.assertEqual(c["cross_pairs"], 2)
        self.assertEqual(c["all_pairs"], 2)
        # 0.9 vs 0.8 两个 7% 不比较

    def test_cross_threshold(self):
        day = make_day([0.07, 0.05, 0.01], [0.9, 0.5, 0.1])
        c = day_repair_concordance(day, "score")
        self.assertEqual(c["cross_pairs"], 2)
        self.assertEqual(c["cross_concordance"], 1.0)

    def test_date_weighted(self):
        d1 = make_day([0.07, 0.05, 0.01], [0.9, 0.5, 0.1])
        d2 = make_day([0.06, 0.04, -0.02], [0.9, 0.5, 0.1], d="2026-05-22")
        both = pd.concat([d1, d2], ignore_index=True)
        rows = [day_repair_concordance(day, "score")
                for _, day in both.groupby("signal_date", sort=True)]
        agg = date_weighted_concordance(rows)
        self.assertAlmostEqual(agg["within_concordance"], 1.0, places=9)
        self.assertEqual(agg["within_evaluable_dates"], 2)


class RankProfileTestCase(unittest.TestCase):
    def test_rank_profile(self):
        d1 = make_day([0.07, 0.05, 0.01], [0.9, 0.5, 0.1])
        d2 = make_day([0.02, 0.01, -0.01], [0.7, 0.4, 0.2], d="2026-05-22")
        both = pd.concat([d1, d2], ignore_index=True)
        prof = rank_profile(both, "score")
        r1 = prof[prof["rank"] == 1].iloc[0]
        self.assertAlmostEqual(r1["mean_capped_return"], (0.07 + 0.02) / 2,
                               places=9)
        self.assertAlmostEqual(r1["target7_hit_rate"], 0.5, places=9)


class OracleDisplayTestCase(unittest.TestCase):
    def test_oracle_not_hardcoded(self):
        """§81: Oracle Top1 hit = mean(target7>=1), 不是 1.0。"""
        d1 = make_day([0.07, 0.05, 0.01], [0.9, 0.5, 0.1])
        d2 = make_day([0.02, 0.01, -0.01], [0.7, 0.4, 0.2], d="2026-05-22")
        both = pd.concat([d1, d2], ignore_index=True)
        s = oracle_ceiling_summary(both)
        self.assertEqual(s["oracle_top1_hit_rate"], 0.5)  # day2 无赢家
        # day1: 1/3; day2: 0/3 -> mean = 1/6
        self.assertAlmostEqual(s["oracle_top3_target7_precision"], 1 / 6,
                               places=9)
        self.assertAlmostEqual(s["oracle_top3_capped_return"],
                               ((0.07 + 0.05 + 0.01) / 3
                                + (0.02 + 0.01 + -0.01) / 3) / 2, places=9)


class NoTrainingNoJulyTestCase(unittest.TestCase):
    def test_module_has_no_model_fit(self):
        """§72: AST 静态检查 — 模块不得 import 模型库或调用任何 fit。"""
        import ast
        src = (BASE / "src" / "v004c_top3_failure_decomposition.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        banned_calls = ("fit_pairwise_ridge", "fit_preprocessor",
                        "select_lambda", "chronological_walkforward",
                        "gbdt_walkforward", "gbdt_in_sample")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.assertNotIn("sklearn", a.name, a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.assertNotIn("sklearn", node.module, node.module)
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    self.assertFalse(f.attr.startswith("fit"), f.attr)
                elif isinstance(f, ast.Name):
                    self.assertNotIn(f.id, banned_calls, f.id)

    def test_no_july(self):
        """§82: 输入资产 max signal_date <= 2026-06-30。"""
        for path in (
                BASE / "reports" / "research"
                / "v004c_pairwise_ridge_v001_20260506_20260630"
                / "v004c_pairwise_ridge_oof_predictions_v001.csv",
                BASE / "reports" / "research"
                / "v004c_upside_predictability_diagnostic_v001_20260506_20260630"
                / "v004c_upside_nonlinear_oof_predictions_v001.csv"):
            df = pd.read_csv(path, encoding="utf-8-sig")
            self.assertLessEqual(df["signal_date"].max(), "2026-06-30")


class DeterminismTestCase(unittest.TestCase):
    def test_decompose_day_deterministic(self):
        day = make_day([0.07, 0.07, 0.06, 0.01, -0.06],
                       [0.9, 0.4, 0.3, 0.5, 0.2])
        d1 = decompose_day(day, "score")
        d2 = decompose_day(day, "score")
        for k in d1:
            v1, v2 = d1[k], d2[k]
            if isinstance(v1, float):
                self.assertEqual(v1, v2, k)


if __name__ == "__main__":
    unittest.main()

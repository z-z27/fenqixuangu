"""v004c Repair-State Model v002 — Frozen July Retrospective OOT 测试。

覆盖任务 §41 的 50 项 (完整列表见下方 docstring 注释)。全部测试确定性
(固定 seed / 显式构造); 真实数据测试只读正式 model table / frozen JSON /
June development metrics。

§41 检查清单:
 1  frozen JSON 存在
 2  model_version 匹配
 3  factor_spec_version 匹配
 4  R1 factor order 严格 5
 5  R2 factor order 严格 8
 6  coefficients 全精度完全匹配 frozen JSON
 7  July rows = 160
 8  July dates = 21
 9  Phase A 完全不加载 Target (usecols 扫描)
10  Phase A rank 不依赖 Target
11  Phase A hash 在 Phase B 前后不变
12  src 无 fit/搜索实现 token
13  tools 无 fit/搜索实现 token
14  spec transform refit 不被调用
15  model fit 不被调用
16  construct 只调用一次
17  R0 用冻结 June prevalence 59/173
18  R0 不用 July prevalence
19  R1/R2 从 frozen 参数手工 sigmoid 重建
20  frozen transform 全 finite
21  概率 finite 且 0<p<1
23  ranking probability 降序
24  tie -> event_id 升序
25  Target 不能打破 tie
26  pooled metrics 手工对照
27  daily AUC 只算正负并存日期
28  pair-weighted within-date AUC 参考实现对照
29  Top1 matched baseline 正确
30  Top3 matched baseline 正确
31  Top3 lift 正确
32  hit-date 字段不是 pass gate 的输入
33  §26 probability gates
34  §27 ranking gates
35  §28 incremental gate
36  §29 Case A-E 最终选择
37  Top1 不参与 hard gate
38  bootstrap 首个 replicate 对照独立参考实现
39  replicates = 2000
40  seed = 20260808 (确定性)
41  bootstrap 不改变 gates
42  prediction dataframe 在 Target 加入前后完全一致
43  改 July Target 改 metrics 不改 predictions
44  改 July Target 不改 rank
45  改 July Target 不改 transform
46  改 July Target 不改 contribution
47  历史冻结资产 (June freeze 9 文件) 不修改
48  src / model table 不修改
49  deterministic rebuild (9 资产 byte-identical)
50  full unittest discover (由任务执行命令验证)
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.v004c_logistic_walkforward import (
    JULY_END,
    JULY_START,
    TARGET_COLUMN,
    brier_score,
    logloss,
    roc_auc,
)
from src.v004c_repair_state_spec_v002 import (
    R1_FACTORS,
    R2_FACTORS,
    REPAIR_PRIMITIVES,
)
import src.v004c_repair_state_model_v002 as model_mod
from src.v004c_repair_state_model_v002 import (
    FACTOR_SPEC_VERSION,
    MODEL_VERSION,
)
import src.v004c_repair_state_spec_v002 as spec_mod
import src.v004c_repair_state_july_oot as july_mod
from src.v004c_repair_state_july_oot import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_MIXED,
    CONFIDENCE_STRONG,
    FROZEN_JUNE_POSITIVE,
    FROZEN_JUNE_TOTAL,
    FROZEN_MODEL_CONTRACT_MISMATCH,
    PREDICTION_HASH_COLUMNS,
    PROMOTE_R1_FORWARD_SHADOW,
    PROMOTE_R2_FORWARD_SHADOW,
    REJECT_REPAIR_STATE_V002_OOT,
    JulyOOTError,
    build_target_blind_predictions,
    calibration_terciles,
    cluster_bootstrap,
    confidence_label,
    daily_ranking_table,
    disagreement_audit,
    evaluate_gates,
    evaluate_july,
    interaction_audit_july,
    june_prevalence,
    load_frozen_model,
    matched_baselines,
    open_target,
    pair_weighted_within_date_auc,
    prediction_hash,
    read_july_target,
    ranking_concentration,
    select_july_frame,
    topk_stats,
    transform_audit,
    validate_frozen_contract,
    within_date_auc_stats,
)
from tools.v004c_repair_state_july_oot import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    KNOWN_JULY_DATES,
    KNOWN_JULY_ROWS,
    OUTPUT_FILES,
    check_git_preconditions,
    run_oot,
)
from tests.test_v004c_repair_state_model_v002 import (
    _FakeResult,
    _count_rows,
    _fake_git,
    _rewrite_july_target,
    _rmtree,
    _sha256_file,
    rewrite_target_column,
    synth_model_csv,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_INPUT = (REPO_ROOT / "reports/research"
              / "v004c_model_table_v001_20260601_20260729"
              / "v004c_model_table_v001.csv")
REAL_FROZEN = (REPO_ROOT
               / "reports/research/v004c_repair_state_model_freeze_v001_202606"
               / "v004c_repair_state_frozen_model_v001.json")
REAL_JUNE_METRICS = (REPO_ROOT
                     / "reports/research/v004c_repair_state_model_freeze_v001_202606"
                     / "v004c_repair_state_development_metrics_v001.csv")
JUNE_FREEZE_DIR = REAL_FROZEN.parent

# 冻结系数全精度 (frozen JSON 全精度, 只读引用)
R1_INTERCEPT = -0.6730455750012894
R1_COEFS = {
    "OPEN": 0.07483231325397154,
    "DIVERGENCE": -0.08794881141190218,
    "CLOSE_DAMAGE": 0.4371984738633886,
    "SUPPLY": -0.1458378254358144,
    "RECLAIM": 0.3440150544488799,
}
R2_INTERCEPT = -0.6765726244947639
R2_COEFS = {
    "OPEN": 0.11990402218639971,
    "DIVERGENCE": -0.1566019473151536,
    "CLOSE_DAMAGE": 0.5397001296627395,
    "SUPPLY": -0.1629768806766291,
    "RECLAIM": 0.414053328248222,
    "DIVERGENCE_SQ": 0.2787008248926637,
    "DIVERGENCE_X_RECLAIM": -0.28630680595148444,
    "DIVERGENCE_X_DAMAGE": -0.3644586048782202,
}

_REAL_CACHE: dict | None = None


def _real_pipeline() -> dict:
    """真实 July 全流程 (Phase A + Phase B + 评价), module 级缓存.

    复现 run_oot 的核心路径但不写报告 (快速 / 无副作用)。
    """
    global _REAL_CACHE
    if _REAL_CACHE is None:
        frozen = load_frozen_model(REAL_FROZEN)
        july_df, stats = select_july_frame(REAL_INPUT)
        blind = build_target_blind_predictions(frozen, july_df)
        rows = set(int(v) for v in blind["predictions"]["_data_row"].tolist())
        targets = read_july_target(REAL_INPUT, rows)
        preds = open_target(blind["predictions"], targets)
        if prediction_hash(preds) != blind["pre_target_prediction_sha256"]:
            raise AssertionError("真实数据两阶段 hash 不一致")
        _REAL_CACHE = {
            "frozen": frozen, "july_df": july_df, "blind": blind,
            "preds": preds, "stats": stats,
        }
    return _REAL_CACHE


def _synth_exact(tmp) -> pathlib.Path:
    """精确形状合成 model table (173/21/59 + 160/21), 与正式输入同构."""
    return synth_model_csv(tmp, seed=7,
                           june_days=21, june_total=173,
                           june_positive=59,
                           july_days=21, july_total=160)


# ---------------------------------------------------------------------------
# §41 1-6: 冻结 JSON 身份与系数
# ---------------------------------------------------------------------------

class TestFrozenJsonContract(unittest.TestCase):

    def test_frozen_json_exists(self):
        self.assertTrue(REAL_FROZEN.exists(), "frozen JSON 不存在")

    def test_model_version_matches(self):
        frozen = load_frozen_model(REAL_FROZEN)
        self.assertEqual(frozen["model_version"], MODEL_VERSION)
        self.assertEqual(frozen["model_version"], "v004c_repair_state_model_v001")

    def test_factor_spec_version_matches(self):
        frozen = load_frozen_model(REAL_FROZEN)
        self.assertEqual(frozen["factor_spec_version"], FACTOR_SPEC_VERSION)
        self.assertEqual(frozen["factor_spec_version"],
                         "v004c_repair_state_spec_v002")

    def test_r1_factor_order_exact(self):
        frozen = load_frozen_model(REAL_FROZEN)
        self.assertEqual(frozen["R1"]["factor_order"], list(R1_FACTORS))
        self.assertEqual(len(frozen["R1"]["factor_order"]), 5)

    def test_r2_factor_order_exact(self):
        frozen = load_frozen_model(REAL_FROZEN)
        self.assertEqual(frozen["R2"]["factor_order"], list(R2_FACTORS))
        self.assertEqual(len(frozen["R2"]["factor_order"]), 8)
        self.assertEqual(frozen["R2"]["factor_order"][:5], list(R1_FACTORS))

    def test_coefficients_full_precision(self):
        frozen = load_frozen_model(REAL_FROZEN)
        r1 = frozen["R1"]
        self.assertEqual(r1["intercept"], R1_INTERCEPT)
        for f in R1_FACTORS:
            self.assertEqual(r1["coefficients"][f], R1_COEFS[f], f)
        r2 = frozen["R2"]
        self.assertEqual(r2["intercept"], R2_INTERCEPT)
        for f in R2_FACTORS:
            self.assertEqual(r2["coefficients"][f], R2_COEFS[f], f)

    def test_contract_mismatch_raises(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_contract_"))
        try:
            frozen = json.loads(REAL_FROZEN.read_text(encoding="utf-8"))
            frozen["training_rows"] = 172
            bad = tmp / "bad_frozen.json"
            bad.write_text(json.dumps(frozen), encoding="utf-8")
            with self.assertRaises(JulyOOTError) as ctx:
                load_frozen_model(bad)
            self.assertIn(FROZEN_MODEL_CONTRACT_MISMATCH, str(ctx.exception))
            frozen2 = json.loads(REAL_FROZEN.read_text(encoding="utf-8"))
            frozen2["research_status"]["july_target_seen"] = True
            (tmp / "bad2.json").write_text(json.dumps(frozen2),
                                           encoding="utf-8")
            problems = validate_frozen_contract(frozen2)
            self.assertTrue(any("july_target_seen" in p for p in problems))
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §41 7-8: July 切片
# ---------------------------------------------------------------------------

class TestJulySlice(unittest.TestCase):

    def test_real_july_rows_160(self):
        july_df, stats = select_july_frame(REAL_INPUT)
        self.assertEqual(len(july_df), KNOWN_JULY_ROWS)
        self.assertEqual(len(july_df), 160)
        self.assertEqual(stats["july_x_rows"], 160)

    def test_real_july_dates_21(self):
        july_df, _ = select_july_frame(REAL_INPUT)
        dates = sorted(pd.unique(july_df["signal_date"]))
        self.assertEqual(len(dates), KNOWN_JULY_DATES)
        self.assertEqual(len(dates), 21)
        self.assertEqual(dates[0], JULY_START)
        self.assertEqual(dates[-1], JULY_END)

    def test_tool_rejects_known_shape_mismatch(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_shape_"))
        try:
            csv = synth_model_csv(tmp, seed=11,
                                  june_days=21, june_total=173,
                                  june_positive=59,
                                  july_days=21, july_total=100)
            with self.assertRaises(JulyOOTError):
                run_oot(csv, output_dir=tmp / "out", git_check=False)
        finally:
            _rmtree(tmp)

    def test_out_of_window_rows_rejected(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_window_"))
        try:
            csv = synth_model_csv(tmp, seed=11, window_outside=True)
            with self.assertRaises(JulyOOTError):
                select_july_frame(csv)
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §41 9-11/42: 两阶段打开契约
# ---------------------------------------------------------------------------

class TestTwoPhase(unittest.TestCase):

    def test_phase_a_reads_no_target_column(self):
        """Phase A 的数据读取不得触碰 Target 列 (usecols 扫描 + 全表禁止)."""
        calls: list[dict] = []
        real_read_csv = pd.read_csv

        def spy(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return real_read_csv(*args, **kwargs)

        with mock.patch("src.v004c_repair_state_model_v002.pd.read_csv",
                        side_effect=spy):
            select_july_frame(REAL_INPUT)
        self.assertTrue(calls, "Phase A 没有发生任何 csv 读取")
        for c in calls:
            usecols = c["kwargs"].get("usecols")
            nrows = c["kwargs"].get("nrows")
            if usecols is None and nrows == 0:
                continue  # 仅表头嗅探 (nrows=0), 不读数据
            self.assertIsNotNone(usecols,
                                 "存在无 usecols 的全表数据读取调用")
            for col in usecols:
                lowered = str(col).lower()
                for token in ("target", "d2_", "d3_", "future"):
                    self.assertNotIn(token, lowered,
                                     f"Phase A 请求列 {col} 含 {token}")

    def test_phase_a_ranks_independent_of_target(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_blind_"))
        try:
            src = _synth_exact(tmp)
            dst = tmp / "flipped.csv"
            _rewrite_july_target(src, dst,
                                 ["True" if i % 3 else "False"
                                  for i in range(160)])
            frozen = load_frozen_model(REAL_FROZEN)
            a = build_target_blind_predictions(frozen,
                                               select_july_frame(src)[0])
            b = build_target_blind_predictions(frozen,
                                               select_july_frame(dst)[0])
            for col in ("r1_probability", "r2_probability",
                        "r1_rank", "r2_rank"):
                self.assertTrue(a["predictions"][col].equals(
                    b["predictions"][col]), col)
            self.assertEqual(a["pre_target_prediction_sha256"],
                             b["pre_target_prediction_sha256"])
        finally:
            _rmtree(tmp)

    def test_pre_post_hash_identical(self):
        c = _real_pipeline()
        self.assertEqual(prediction_hash(c["preds"]),
                         c["blind"]["pre_target_prediction_sha256"])

    def test_prediction_columns_unchanged_after_open(self):
        c = _real_pipeline()
        pre = c["blind"]["predictions"].copy()
        post = c["preds"]
        for col in PREDICTION_HASH_COLUMNS:
            self.assertTrue(pre[col].equals(post[col]), col)

    def test_open_target_single_open_contract(self):
        c = _real_pipeline()
        with self.assertRaises(JulyOOTError):
            open_target(c["preds"], {})


# ---------------------------------------------------------------------------
# §41 12-16: 无任何重新拟合 / 只复用 spec apply
# ---------------------------------------------------------------------------

class TestNoRefit(unittest.TestCase):

    FORBIDDEN_TOKENS = (
        "LogisticRegression", ".fit(", "fit_logistic(",
        "fit_repair_state_transform_v2", "quantile(",
        "GridSearchCV(", "optuna.", "cross_val_score(", "polyfit",
    )

    def test_src_no_forbidden_impl_tokens(self):
        text = (REPO_ROOT / "src/v004c_repair_state_july_oot.py").read_text(
            encoding="utf-8")
        for token in self.FORBIDDEN_TOKENS:
            self.assertNotIn(token, text, f"src 出现 {token}")

    def test_tools_no_forbidden_impl_tokens(self):
        text = (REPO_ROOT / "tools/v004c_repair_state_july_oot.py").read_text(
            encoding="utf-8")
        for token in self.FORBIDDEN_TOKENS:
            self.assertNotIn(token, text, f"tools 出现 {token}")

    def test_no_transform_refit_during_blind(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_norefit_"))
        try:
            csv = _synth_exact(tmp)
            july_df, _ = select_july_frame(csv)
            frozen = load_frozen_model(REAL_FROZEN)
            with mock.patch.object(
                    spec_mod, "fit_repair_state_transform_v2",
                    side_effect=AssertionError("refit 被调用")) as spy, \
                    mock.patch.object(
                        model_mod, "fit_model",
                        side_effect=AssertionError("fit_model 被调用")) as spy2:
                build_target_blind_predictions(frozen, july_df)
            spy.assert_not_called()
            spy2.assert_not_called()
        finally:
            _rmtree(tmp)

    def test_no_model_fit_during_full_run(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_nofit_"))
        try:
            csv = _synth_exact(tmp)
            with mock.patch.object(
                    spec_mod, "fit_repair_state_transform_v2",
                    side_effect=AssertionError("refit 被调用")), \
                    mock.patch.object(
                        model_mod, "fit_model",
                        side_effect=AssertionError("fit_model 被调用")):
                run_oot(csv, output_dir=tmp / "out", git_check=False)
        finally:
            _rmtree(tmp)

    def test_construct_called_once_per_blind_build(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_cons_"))
        try:
            csv = _synth_exact(tmp)
            july_df, _ = select_july_frame(csv)
            frozen = load_frozen_model(REAL_FROZEN)
            with mock.patch.object(
                    july_mod, "construct_repair_factors_v2",
                    wraps=july_mod.construct_repair_factors_v2) as spy:
                build_target_blind_predictions(frozen, july_df)
            self.assertEqual(spy.call_count, 1)
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §41 17-18: R0 = 冻结 June prevalence
# ---------------------------------------------------------------------------

class TestR0(unittest.TestCase):

    def test_r0_is_frozen_june_prevalence(self):
        frozen = load_frozen_model(REAL_FROZEN)
        self.assertEqual(june_prevalence(frozen), FROZEN_JUNE_POSITIVE
                         / FROZEN_JUNE_TOTAL)
        c = _real_pipeline()
        r0 = c["preds"]["r0_probability"]
        self.assertEqual(len(set(r0)), 1)
        self.assertEqual(float(r0.iloc[0]), 59 / 173)

    def test_r0_never_uses_july_prevalence(self):
        c = _real_pipeline()
        r0 = float(c["preds"]["r0_probability"].iloc[0])
        observed = float(c["preds"][TARGET_COLUMN].mean())
        self.assertEqual(r0, 59 / 173)
        self.assertNotEqual(r0, observed)  # July 正例率 0.2562 != 0.3410


# ---------------------------------------------------------------------------
# §41 19-21: 从冻结参数手工重建预测
# ---------------------------------------------------------------------------

class TestPredictionRebuild(unittest.TestCase):

    def test_manual_sigmoid_rebuild_real(self):
        c = _real_pipeline()
        factors = c["blind"]["factors"]
        for tag, block in (("r1", c["frozen"]["R1"]),
                           ("r2", c["frozen"]["R2"])):
            order = list(block["factor_order"])
            coef = np.asarray([block["coefficients"][f] for f in order])
            logit = (factors[order].to_numpy(dtype=float) @ coef
                     + block["intercept"])
            manual = 1.0 / (1.0 + np.exp(-logit))
            np.testing.assert_allclose(
                c["preds"]["{}_probability".format(tag)].to_numpy(dtype=float),
                manual, rtol=0, atol=1e-12, err_msg=tag)

    def test_frozen_transform_factors_finite(self):
        c = _real_pipeline()
        self.assertTrue(
            np.all(np.isfinite(c["blind"]["factors"].to_numpy(dtype=float))))

    def test_probabilities_finite_in_open_unit(self):
        c = _real_pipeline()
        for col in ("r0_probability", "r1_probability", "r2_probability"):
            p = c["preds"][col].to_numpy(dtype=float)
            self.assertTrue(np.all(np.isfinite(p)), col)
            self.assertTrue(np.all((p > 0.0) & (p < 1.0)), col)


# ---------------------------------------------------------------------------
# §41 23-25: ranking 规则
# ---------------------------------------------------------------------------

class TestRankingRules(unittest.TestCase):

    def test_ranks_probability_desc(self):
        c = _real_pipeline()
        preds = c["preds"]
        for d in sorted(pd.unique(preds["signal_date"])):
            sub = preds[preds["signal_date"] == d].copy()
            for tag in ("r1", "r2"):
                pcol = "{}_probability".format(tag)
                rcol = "{}_rank".format(tag)
                expected = sub.sort_values([pcol, "event_id"],
                                           ascending=[False, True])
                expected = pd.Series(np.arange(1, len(expected) + 1),
                                     index=expected.index)
                np.testing.assert_array_equal(
                    sub[rcol].to_numpy(dtype=int),
                    expected.loc[sub.index].to_numpy(dtype=int),
                    "date {} {}".format(d, tag))

    def test_tie_broken_by_event_id_asc(self):
        """全 tie 场景: 所有 factor 恒同 -> 概率恒同 -> rank 按 event_id
        升序 (禁止用 Target / code 打破 tie)。"""
        rows = 60
        dates = [f"2026-07-{d:02d}" for d in range(1, 4)]
        july_df = pd.DataFrame({
            "event_id": [f"E{i:05d}" for i in range(rows)],
            "code": ["600000"] * rows,
            "signal_date": [dates[i % 3] for i in range(rows)],
        })
        for col in REPAIR_PRIMITIVES:
            july_df[col] = 0.0
        frozen = load_frozen_model(REAL_FROZEN)
        blind = build_target_blind_predictions(frozen, july_df)
        preds = blind["predictions"]
        for tag in ("r1", "r2"):
            pcol = "{}_probability".format(tag)
            rcol = "{}_rank".format(tag)
            self.assertEqual(len(set(preds[pcol])), 1,
                             f"{tag} 概率应全部相等 (恒同 factor)")
            for d in sorted(pd.unique(preds["signal_date"])):
                sub = preds[preds["signal_date"] == d].sort_values("event_id")
                self.assertEqual(list(sub[rcol]),
                                 list(range(1, len(sub) + 1)),
                                 f"{tag} {d}")

    def test_target_never_breaks_tie(self):
        rows = 60
        dates = [f"2026-07-{d:02d}" for d in range(1, 4)]
        base = pd.DataFrame({
            "event_id": [f"E{i:05d}" for i in range(rows)],
            "code": ["600000"] * rows,
            "signal_date": [dates[i % 3] for i in range(rows)],
        })
        for col in REPAIR_PRIMITIVES:
            base[col] = 0.0
        frozen = load_frozen_model(REAL_FROZEN)
        a = base.copy()
        a[TARGET_COLUMN] = [1, 0] * (rows // 2)
        b = base.copy()
        b[TARGET_COLUMN] = [0, 1] * (rows // 2)
        blind_a = build_target_blind_predictions(frozen, a)
        blind_b = build_target_blind_predictions(frozen, b)
        for col in ("r1_rank", "r2_rank", "r1_probability", "r2_probability"):
            self.assertTrue(blind_a["predictions"][col].equals(
                blind_b["predictions"][col]), col)
        self.assertEqual(blind_a["pre_target_prediction_sha256"],
                         blind_b["pre_target_prediction_sha256"])

    def test_daily_ranking_tie_top1_smallest_event_id(self):
        preds = pd.DataFrame({
            "event_id": ["E200", "E100", "E300", "E400"],
            "code": ["600000"] * 4,
            "signal_date": ["2026-07-01"] * 4,
            TARGET_COLUMN: [1, 0, 1, 0],
            "r1_probability": [0.5, 0.5, 0.8, 0.2],
            "r2_probability": [0.5, 0.5, 0.8, 0.2],
        })
        daily = daily_ranking_table(preds)
        row = daily.iloc[0]
        self.assertEqual(row["r1_top1_event_id"], "E300")  # 概率最高
        tied = preds.copy()
        tied["r1_probability"] = [0.5, 0.5, 0.5, 0.5]
        tied["r2_probability"] = [0.5, 0.5, 0.5, 0.5]
        row2 = daily_ranking_table(tied).iloc[0]
        self.assertEqual(row2["r1_top1_event_id"], "E100")  # tie -> event_id
        self.assertEqual(row2["r2_top1_event_id"], "E100")


# ---------------------------------------------------------------------------
# §41 26-31: 指标 / baselines / lift
# ---------------------------------------------------------------------------

class TestMetricsCorrect(unittest.TestCase):

    def test_pooled_metrics_manual(self):
        c = _real_pipeline()
        preds = c["preds"]
        y = preds[TARGET_COLUMN].to_numpy(dtype=int)
        m = evaluate_july(preds)["metrics_df"].set_index("model")
        for tag in ("r1", "r2"):
            p = preds["{}_probability".format(tag)].to_numpy(dtype=float)
            self.assertEqual(m.loc[tag.upper(), "logloss"], logloss(y, p))
            self.assertEqual(m.loc[tag.upper(), "brier"], brier_score(y, p))
            self.assertEqual(m.loc[tag.upper(), "auc"], roc_auc(y, p))
        p0 = preds["r0_probability"].to_numpy(dtype=float)
        self.assertEqual(m.loc["R0", "logloss"], logloss(y, p0))
        self.assertEqual(m.loc["R0", "brier"], brier_score(y, p0))
        self.assertEqual(m.loc["R0", "auc"], 0.5)

    def test_daily_auc_only_mixed_dates(self):
        preds = pd.DataFrame({
            "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "code": ["600000"] * 6,
            "signal_date": (["2026-07-01"] * 2 + ["2026-07-02"] * 2
                            + ["2026-07-03"] * 2),
            TARGET_COLUMN: [1, 0, 1, 1, 0, 1],
            "r1_probability": [0.9, 0.1, 0.5, 0.5, 0.8, 0.9],
            "r2_probability": [0.9, 0.1, 0.5, 0.5, 0.8, 0.9],
        })
        daily = daily_ranking_table(preds)
        row2 = daily[daily["signal_date"] == "2026-07-02"].iloc[0]
        self.assertFalse(row2["r1_auc_valid"])
        self.assertFalse(row2["r2_auc_valid"])
        pw = pair_weighted_within_date_auc(preds, "r1_probability")
        # 07-01: (pos E1, neg E2) -> 0.9>0.1 正确; 07-03: (pos E6, neg E5)
        # -> 0.9>0.8 正确; 07-02 单类别不产生 pair
        self.assertEqual(pw, 1.0)
        stats = within_date_auc_stats(daily, "r1")
        self.assertEqual(stats["valid_auc_dates"], 2)

    def test_pair_weighted_auc_reference(self):
        """独立参考实现 (显式双层循环, 只计同日 pos-neg 对) 对照."""
        c = _real_pipeline()
        preds = c["preds"]

        def ref(pcol):
            total = 0
            correct = 0.0
            for d in sorted(pd.unique(preds["signal_date"])):
                sub = preds[preds["signal_date"] == d]
                y = sub[TARGET_COLUMN].to_numpy(dtype=int)
                p = sub[pcol].to_numpy(dtype=float)
                for i in range(len(sub)):
                    for j in range(len(sub)):
                        if y[i] == 1 and y[j] == 0:
                            total += 1
                            correct += (p[i] > p[j]) + 0.5 * (p[i] == p[j])
            return correct / total

        for tag in ("r1", "r2"):
            got = pair_weighted_within_date_auc(
                preds, "{}_probability".format(tag))
            self.assertAlmostEqual(got, ref(
                "{}_probability".format(tag)), places=12, msg=tag)

    def test_top1_baseline_matched(self):
        c = _real_pipeline()
        preds = c["preds"]
        daily = daily_ranking_table(preds)
        base = matched_baselines(preds, daily)
        manual = float(daily["candidate_rate"].mean())
        self.assertEqual(base["top1_baseline"], manual)
        self.assertEqual(base["mean_daily_candidate_rate"], manual)
        self.assertEqual(base["pooled_candidate_rate"],
                         float(preds[TARGET_COLUMN].mean()))

    def test_top3_baseline_matched(self):
        c = _real_pipeline()
        preds = c["preds"]
        daily = daily_ranking_table(preds)
        base = matched_baselines(preds, daily)
        k = daily["k"].to_numpy(dtype=int)
        rates = daily["candidate_rate"].to_numpy(dtype=float)
        manual = float(np.sum(k * rates) / np.sum(k))
        self.assertEqual(base["top3_baseline"], manual)

    def test_top3_lift(self):
        c = _real_pipeline()
        preds = c["preds"]
        daily = daily_ranking_table(preds)
        base = matched_baselines(preds, daily)
        for tag in ("r1", "r2"):
            stats = topk_stats(daily, base, tag)
            self.assertEqual(stats["top3_lift"],
                             stats["top3_target_rate"] - base["top3_baseline"])
            self.assertEqual(stats["top1_lift"],
                             stats["top1_target_rate"] - base["top1_baseline"])
            self.assertEqual(stats["top3_picks"], int(daily["k"].sum()))


# ---------------------------------------------------------------------------
# §41 32-37: 预声明判定门
# ---------------------------------------------------------------------------

def _mk_metrics(r0ll, r0br, r1ll, r1br, r2ll, r2br) -> pd.DataFrame:
    return pd.DataFrame({"model": ["R0", "R1", "R2"],
                         "logloss": [r0ll, r1ll, r2ll],
                         "brier": [r0br, r1br, r2br]})


def _mk_ctx(pw1=0.60, pw2=0.60, lift1=0.05, lift2=0.05,
            t3r1=0.40, t3r2=0.45):
    within = {
        "R1": {"pair_weighted_within_date_auc": pw1},
        "R2": {"pair_weighted_within_date_auc": pw2},
    }
    topk = {
        "R1": {"top3_lift": lift1, "top3_target_rate": t3r1},
        "R2": {"top3_lift": lift2, "top3_target_rate": t3r2},
    }
    return within, topk


class TestGates(unittest.TestCase):

    def _gate_source(self) -> str:
        """evaluate_gates 的实现源码 (判定逻辑本身, 不含数据层)."""
        src = (REPO_ROOT / "src/v004c_repair_state_july_oot.py").read_text(
            encoding="utf-8")
        start = src.index("def evaluate_gates")
        end = src.index("def cluster_bootstrap")
        return src[start:end]

    def test_gates_do_not_require_hit_date_fields(self):
        """hit-date / Top1 字段不是 gate 的输入 (缺键不报错)."""
        metrics = _mk_metrics(0.60, 0.22, 0.58, 0.21, 0.57, 0.20)
        within, topk = _mk_ctx()
        gates = evaluate_gates(metrics, within, topk)
        self.assertTrue(gates["R1_PROBABILITY_PASS"])
        self.assertIsInstance(gates["FINAL_DECISION"], str)

    def test_probability_gate_rules(self):
        """§26: R1_PROBABILITY_PASS = LL<R0 且 Brier<R0."""
        within, topk = _mk_ctx()
        g = evaluate_gates(_mk_metrics(0.60, 0.22, 0.58, 0.21,
                                       0.62, 0.23), within, topk)
        self.assertTrue(g["R1_PROBABILITY_PASS"])
        self.assertFalse(g["R2_PROBABILITY_PASS"])
        g2 = evaluate_gates(_mk_metrics(0.60, 0.22, 0.61, 0.21,
                                        0.58, 0.21), within, topk)
        self.assertFalse(g2["R1_PROBABILITY_PASS"])  # LL 未低于 R0
        g3 = evaluate_gates(_mk_metrics(0.60, 0.22, 0.58, 0.23,
                                        0.58, 0.21), within, topk)
        self.assertFalse(g3["R1_PROBABILITY_PASS"])  # Brier 未低于 R0

    def test_ranking_gate_rules(self):
        """§27: R1_RANKING_PASS = pwAUC>0.50 且 Top3 lift>0."""
        metrics = _mk_metrics(0.60, 0.22, 0.58, 0.21, 0.58, 0.21)
        g = evaluate_gates(metrics, *_mk_ctx(pw1=0.60, lift1=0.05))
        self.assertTrue(g["R1_RANKING_PASS"])
        g2 = evaluate_gates(metrics, *_mk_ctx(pw1=0.50, lift1=0.05))
        self.assertFalse(g2["R1_RANKING_PASS"])  # 0.50 不是 > 0.50
        g3 = evaluate_gates(metrics, *_mk_ctx(pw1=0.60, lift1=0.0))
        self.assertFalse(g3["R1_RANKING_PASS"])  # lift 未 > 0

    def test_incremental_gate_rules(self):
        """§28: R2_INCREMENTAL_PASS 四条件全部成立."""
        metrics = _mk_metrics(0.60, 0.22, 0.58, 0.21, 0.57, 0.20)
        g = evaluate_gates(metrics, *_mk_ctx(pw1=0.60, pw2=0.62,
                                             t3r1=0.40, t3r2=0.45))
        self.assertTrue(g["R2_INCREMENTAL_PASS"])
        g2 = evaluate_gates(_mk_metrics(0.60, 0.22, 0.58, 0.21,
                                        0.59, 0.20),
                            *_mk_ctx(pw1=0.60, pw2=0.62,
                                     t3r1=0.40, t3r2=0.45))
        self.assertFalse(g2["R2_INCREMENTAL_PASS"])  # LL 未低于 R1
        g3 = evaluate_gates(metrics, *_mk_ctx(pw1=0.60, pw2=0.59,
                                              t3r1=0.40, t3r2=0.45))
        self.assertFalse(g3["R2_INCREMENTAL_PASS"])  # pwAUC 低于 R1
        g4 = evaluate_gates(metrics, *_mk_ctx(pw1=0.60, pw2=0.62,
                                              t3r1=0.40, t3r2=0.38))
        self.assertFalse(g4["R2_INCREMENTAL_PASS"])  # Top3 rate 低于 R1

    def test_final_selection_cases(self):
        """§29: Case A-E 全覆盖."""
        within, topk = _mk_ctx()
        # Case A: R1 NO + R2 NO -> REJECT
        g = evaluate_gates(_mk_metrics(0.60, 0.22, 0.62, 0.23, 0.63, 0.24),
                           *_mk_ctx(pw1=0.40, pw2=0.40, lift1=-0.1, lift2=-0.1))
        self.assertEqual(g["FINAL_DECISION"], REJECT_REPAIR_STATE_V002_OOT)
        self.assertFalse(g["R2_ONLY_OOT_PASS"])
        # Case B: R1 YES + R2 NO -> PROMOTE_R1
        g = evaluate_gates(_mk_metrics(0.60, 0.22, 0.58, 0.21, 0.63, 0.24),
                           *_mk_ctx(pw1=0.60, pw2=0.40, lift1=0.05, lift2=-0.1))
        self.assertEqual(g["FINAL_DECISION"], PROMOTE_R1_FORWARD_SHADOW)
        # Case C: 双 YES + incremental NO -> PROMOTE_R1 (简约原则)
        # (R2 概率 gate 通过, 但 R2 LogLoss 0.59 未低于 R1 的 0.58)
        g = evaluate_gates(_mk_metrics(0.60, 0.22, 0.58, 0.21, 0.59, 0.21),
                           *_mk_ctx(pw1=0.60, pw2=0.62,
                                    t3r1=0.40, t3r2=0.38))
        self.assertEqual(g["FINAL_DECISION"], PROMOTE_R1_FORWARD_SHADOW)
        # Case D: 双 YES + incremental YES -> PROMOTE_R2
        g = evaluate_gates(_mk_metrics(0.60, 0.22, 0.58, 0.21, 0.57, 0.20),
                           *_mk_ctx(pw1=0.60, pw2=0.62,
                                    t3r1=0.40, t3r2=0.45))
        self.assertEqual(g["FINAL_DECISION"], PROMOTE_R2_FORWARD_SHADOW)
        # Case E: R1 NO + R2 YES -> PROMOTE_R2, R2_ONLY = YES
        g = evaluate_gates(_mk_metrics(0.60, 0.22, 0.62, 0.23, 0.58, 0.21),
                           *_mk_ctx(pw1=0.40, pw2=0.60, lift1=-0.1, lift2=0.05))
        self.assertEqual(g["FINAL_DECISION"], PROMOTE_R2_FORWARD_SHADOW)
        self.assertTrue(g["R2_ONLY_OOT_PASS"])

    def test_top1_not_a_hard_gate(self):
        """Top1 字段不进 evaluate_gates 读取路径 (签名只含 metrics/within
        /topk 的 §26-28 字段)。"""
        metrics = _mk_metrics(0.60, 0.22, 0.58, 0.21, 0.57, 0.20)
        within, topk = _mk_ctx()
        g = evaluate_gates(metrics, within, topk)
        self.assertEqual(g["FINAL_DECISION"], PROMOTE_R2_FORWARD_SHADOW)
        self.assertNotIn("top1_target_rate", self._gate_source())

    def test_confidence_label_rules(self):
        """§32: 晋级模型 LL/Brier/Top3 lift CI 下界全部 > 0 => STRONG."""
        cols = ("r1_ll_improvement_vs_r0", "r1_brier_improvement_vs_r0",
                "r1_top3_lift", "r2_pair_weighted_within_date_auc")
        ci = pd.DataFrame({
            "statistic": cols,
            "lower": [0.01, 0.01, 0.01, -0.1],
            "upper": [0.2, 0.2, 0.2, 0.3],
            "n_finite": [2000] * 4,
        })
        self.assertEqual(
            confidence_label(PROMOTE_R1_FORWARD_SHADOW, ci),
            CONFIDENCE_STRONG)
        ci2 = ci.copy()
        ci2.loc[ci2["statistic"] == "r1_top3_lift", "lower"] = -0.01
        self.assertEqual(
            confidence_label(PROMOTE_R1_FORWARD_SHADOW, ci2),
            CONFIDENCE_MIXED)
        self.assertEqual(
            confidence_label(REJECT_REPAIR_STATE_V002_OOT, ci), "N/A")


# ---------------------------------------------------------------------------
# §41 38-41: bootstrap
# ---------------------------------------------------------------------------

def _ref_bootstrap_first(preds, seed=BOOTSTRAP_SEED) -> dict:
    """独立参考实现: 同 RNG 序列 + 同公式, 但按规范重写 (对照首个 replicate)."""
    rng = np.random.default_rng(seed)
    dates = sorted(pd.unique(preds["signal_date"]))
    n_dates = len(dates)
    y = preds[TARGET_COLUMN].to_numpy(dtype=int)
    date_col = preds["signal_date"].to_numpy()
    p1 = preds["r1_probability"].to_numpy(dtype=float)
    p2 = preds["r2_probability"].to_numpy(dtype=float)
    eid = preds["event_id"].astype(str).to_numpy()
    p0 = float(preds["r0_probability"].iloc[0])
    per_date = [np.flatnonzero(date_col == d) for d in dates]
    sampled = rng.choice(n_dates, size=n_dates, replace=True)
    blocks = [per_date[i] for i in sampled]
    idx = np.concatenate(blocks)
    y_r, p1_r, p2_r = y[idx], p1[idx], p2[idx]
    ll0 = logloss(y_r, np.full(len(y_r), p0))
    br0 = brier_score(y_r, np.full(len(y_r), p0))
    ll1 = logloss(y_r, p1_r)
    br1 = brier_score(y_r, p1_r)
    ll2 = logloss(y_r, p2_r)
    br2 = brier_score(y_r, p2_r)
    total1 = correct1 = total2 = correct2 = 0
    hits1 = hits2 = 0
    k_sum = krate_sum = 0
    for blk in blocks:
        yb, eid_b = y[blk], eid[blk]
        for tag, pb in (("1", p1[blk]), ("2", p2[blk])):
            pos = np.flatnonzero(yb == 1)
            neg = np.flatnonzero(yb == 0)
            if len(pos) and len(neg):
                n = len(pos) * len(neg)
                pp = pb[pos][:, None]
                pn = pb[neg][None, :]
                if tag == "1":
                    total1 += n
                    correct1 += float(np.sum(pp > pn) + 0.5 * np.sum(pp == pn))
                else:
                    total2 += n
                    correct2 += float(np.sum(pp > pn) + 0.5 * np.sum(pp == pn))
        k = min(3, len(blk))
        k_sum += k
        krate_sum += k * float(yb.mean())
        o1 = np.lexsort((eid_b, -p1[blk]))[:k]
        o2 = np.lexsort((eid_b, -p2[blk]))[:k]
        hits1 += int(np.sum(yb[o1]))
        hits2 += int(np.sum(yb[o2]))
    return {
        "r1_ll_improvement_vs_r0": ll0 - ll1,
        "r1_brier_improvement_vs_r0": br0 - br1,
        "r2_ll_improvement_vs_r0": ll0 - ll2,
        "r2_brier_improvement_vs_r0": br0 - br2,
        "r2_ll_improvement_vs_r1": ll1 - ll2,
        "r2_brier_improvement_vs_r1": br1 - br2,
        "r1_pair_weighted_within_date_auc": correct1 / total1,
        "r2_pair_weighted_within_date_auc": correct2 / total2,
        "r1_top3_lift": hits1 / k_sum - krate_sum / k_sum,
        "r2_top3_lift": hits2 / k_sum - krate_sum / k_sum,
    }


class TestBootstrap(unittest.TestCase):

    def test_first_replicate_matches_reference(self):
        c = _real_pipeline()
        rep, _ = cluster_bootstrap(c["preds"])
        ref = _ref_bootstrap_first(c["preds"])
        for col in ("r1_ll_improvement_vs_r0", "r1_brier_improvement_vs_r0",
                    "r2_ll_improvement_vs_r0", "r2_brier_improvement_vs_r0",
                    "r2_ll_improvement_vs_r1", "r2_brier_improvement_vs_r1",
                    "r1_pair_weighted_within_date_auc",
                    "r2_pair_weighted_within_date_auc",
                    "r1_top3_lift", "r2_top3_lift"):
            self.assertEqual(float(rep.iloc[0][col]), ref[col], col)

    def test_replicates_2000(self):
        c = _real_pipeline()
        rep, ci = cluster_bootstrap(c["preds"])
        self.assertEqual(len(rep), BOOTSTRAP_REPLICATES)
        self.assertEqual(len(rep), 2000)
        self.assertEqual(len(ci), 10)
        self.assertTrue((ci["n_finite"] == 2000).all())

    def test_seed_deterministic(self):
        c = _real_pipeline()
        rep1, _ = cluster_bootstrap(c["preds"])
        rep2, _ = cluster_bootstrap(c["preds"])
        self.assertTrue(rep1.equals(rep2))
        rep3, _ = cluster_bootstrap(c["preds"], seed=BOOTSTRAP_SEED + 1)
        self.assertFalse(rep1.iloc[0].equals(rep3.iloc[0]))
        # 默认 seed 常量
        self.assertEqual(BOOTSTRAP_SEED, 20260808)

    def test_bootstrap_does_not_change_gates(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_bs_"))
        try:
            csv = _synth_exact(tmp)
            r = run_oot(csv, output_dir=tmp / "out", git_check=False)
            eval2 = evaluate_july(r["predictions"])
            self.assertEqual(r["decision"],
                             eval2["gates"]["FINAL_DECISION"])
            rep, ci = cluster_bootstrap(r["predictions"])
            self.assertEqual(len(rep), 2000)
            self.assertEqual(r["decision"],
                             eval2["gates"]["FINAL_DECISION"])
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §41 43-46: July Target 不变性
# ---------------------------------------------------------------------------

class TestTargetInvariance(unittest.TestCase):

    def test_metrics_change_predictions_do_not(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_invar_"))
        try:
            src = _synth_exact(tmp)
            dst = tmp / "flipped.csv"
            _rewrite_july_target(src, dst,
                                 ["True" if i % 3 else "False"
                                  for i in range(160)])
            ra = run_oot(src, output_dir=tmp / "a", git_check=False)
            rb = run_oot(dst, output_dir=tmp / "b", git_check=False)
            pred_a = pd.read_csv(tmp / "a" / OUTPUT_FILES[0],
                                 encoding="utf-8-sig")
            pred_b = pd.read_csv(tmp / "b" / OUTPUT_FILES[0],
                                 encoding="utf-8-sig")
            drop = pred_a.drop(columns=[TARGET_COLUMN]).to_csv(index=False)
            drop2 = pred_b.drop(columns=[TARGET_COLUMN]).to_csv(index=False)
            self.assertEqual(drop, drop2, "predictions 受 Target 影响")
            # 两阶段 hash 也不受 Target 影响 (预测部分一致)
            self.assertEqual(ra["two_phase"]["PRE_TARGET_PREDICTION_SHA256"],
                             rb["two_phase"]["PRE_TARGET_PREDICTION_SHA256"])
            self.assertEqual(ra["two_phase"]["POST_TARGET_PREDICTION_SHA256"],
                             rb["two_phase"]["POST_TARGET_PREDICTION_SHA256"])
            # metrics 必须变化
            self.assertNotEqual(
                float(ra["eval"]["metrics_df"].set_index("model")
                      .loc["R1", "logloss"]),
                float(rb["eval"]["metrics_df"].set_index("model")
                      .loc["R1", "logloss"]))
        finally:
            _rmtree(tmp)

    def test_ranks_unchanged(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_invar2_"))
        try:
            src = _synth_exact(tmp)
            dst = tmp / "flipped.csv"
            _rewrite_july_target(src, dst,
                                 ["True" if i % 3 else "False"
                                  for i in range(160)])
            run_oot(src, output_dir=tmp / "a", git_check=False)
            run_oot(dst, output_dir=tmp / "b", git_check=False)
            pa = pd.read_csv(tmp / "a" / OUTPUT_FILES[0],
                             encoding="utf-8-sig")
            pb = pd.read_csv(tmp / "b" / OUTPUT_FILES[0],
                             encoding="utf-8-sig")
            for col in ("r1_rank", "r2_rank"):
                self.assertTrue(
                    (pa[col].to_numpy(dtype=int)
                     == pb[col].to_numpy(dtype=int)).all(), col)
        finally:
            _rmtree(tmp)

    def test_transform_unchanged(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_invar3_"))
        try:
            src = _synth_exact(tmp)
            dst = tmp / "flipped.csv"
            _rewrite_july_target(src, dst,
                                 ["True" if i % 3 else "False"
                                  for i in range(160)])
            run_oot(src, output_dir=tmp / "a", git_check=False)
            run_oot(dst, output_dir=tmp / "b", git_check=False)
            ta = pd.read_csv(tmp / "a" / OUTPUT_FILES[6],
                             encoding="utf-8-sig")
            tb = pd.read_csv(tmp / "b" / OUTPUT_FILES[6],
                             encoding="utf-8-sig")
            self.assertEqual(ta.to_csv(index=False), tb.to_csv(index=False))
        finally:
            _rmtree(tmp)

    def test_contributions_unchanged(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_invar4_"))
        try:
            src = _synth_exact(tmp)
            dst = tmp / "flipped.csv"
            _rewrite_july_target(src, dst,
                                 ["True" if i % 3 else "False"
                                  for i in range(160)])
            run_oot(src, output_dir=tmp / "a", git_check=False)
            run_oot(dst, output_dir=tmp / "b", git_check=False)
            ia = pd.read_csv(tmp / "a" / OUTPUT_FILES[7],
                             encoding="utf-8-sig")
            ib = pd.read_csv(tmp / "b" / OUTPUT_FILES[7],
                             encoding="utf-8-sig")
            self.assertEqual(ia.to_csv(index=False), ib.to_csv(index=False))
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §41 47-48: 历史资产不可变
# ---------------------------------------------------------------------------

class TestHistoricalAssets(unittest.TestCase):

    def _snapshot(self) -> dict:
        paths = [REAL_INPUT]
        paths += sorted(JUNE_FREEZE_DIR.glob("*"))
        for p in sorted((REPO_ROOT / "src").glob("v004c_repair_state_*.py")):
            paths.append(p)
        for name in ("v004c_factor_spec.py", "v004c_logistic_walkforward.py",
                     "v004c_repair_state_spec.py"):
            paths.append(REPO_ROOT / "src" / name)
        return {str(p): _sha256_file(p) for p in paths}

    def test_june_freeze_assets_untouched(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_hist_"))
        try:
            csv = _synth_exact(tmp)
            before = self._snapshot()
            run_oot(csv, output_dir=tmp / "out", git_check=False)
            after = self._snapshot()
            self.assertEqual(before, after, "历史资产被修改")
        finally:
            _rmtree(tmp)

    def test_src_and_input_untouched(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_hist2_"))
        try:
            csv = _synth_exact(tmp)
            before = self._snapshot()
            run_oot(csv, output_dir=tmp / "out", git_check=False)
            after = self._snapshot()
            self.assertEqual(before, after)
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §41 49: deterministic rebuild
# ---------------------------------------------------------------------------

class TestDeterministicRebuild(unittest.TestCase):

    def test_two_runs_byte_identical(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="july_det_"))
        try:
            csv = _synth_exact(tmp)
            run_oot(csv, output_dir=tmp / "run1", git_check=False)
            run_oot(csv, output_dir=tmp / "run2", git_check=False)
            for name in OUTPUT_FILES:
                self.assertEqual(
                    _sha256_file(tmp / "run1" / name),
                    _sha256_file(tmp / "run2" / name),
                    f"rebuild 资产变化: {name}")
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# git 前置 (§一; 与 June 同模式)
# ---------------------------------------------------------------------------

class TestGitPreconditions(unittest.TestCase):

    def test_ok_when_branch_and_head_match(self):
        check_git_preconditions(run_fn=_fake_git(EXPECTED_BRANCH, EXPECTED_HEAD))

    def test_rejects_wrong_branch(self):
        with self.assertRaises(JulyOOTError):
            check_git_preconditions(run_fn=_fake_git("main", EXPECTED_HEAD))

    def test_rejects_wrong_head(self):
        with self.assertRaises(JulyOOTError):
            check_git_preconditions(
                run_fn=_fake_git(EXPECTED_BRANCH, "0" * 40))


if __name__ == "__main__":
    unittest.main()

"""v004c Repair-State Model v002 — June full-development fit + 冻结模型测试。

覆盖任务 §28 的 35 项 (36-39: 旧套件 45/40/51 + full discover, 由任务执行
命令验证):

1  branch/head 前置由运行脚本检查
2  June rows = 173
3  June dates = 21
4  positive = 59
5  negative = 114
6  July Target 从未加载 (解析/保留/使用)
7  R0 = exact June prevalence
8  R1 严格 5 factors
9  R2 严格 8 factors
10 R1 factor order 固定
11 R2 factor order 固定
12 Logistic 参数完全固定
13 class_weight=None
14 no hyperparameter search
15 transform 只 fit June
16 primitive clip-first
17 CLOSE_DAMAGE 等于 v002 定义
18 RECLAIM 等于 v002 定义
19 derived no second clip
20 model 不重新 fit derived transform
21 coefficients 全部 finite
22 probability finite 且 0<p<1
23 daily ranking tie event_id ascending
24 R0 ranking N/A
25 Target 不能用于 ranking
26 within-date AUC 只对同时存在正负的日期计算
27 pair-weighted AUC 实现正确
28 interaction contribution 计算正确
29 frozen JSON 可以完整重建 R1 预测
30 frozen JSON 可以完整重建 R2 预测
31 从 JSON 重建预测与原预测误差 < 1e-12
32 修改 July X 不能改变 June coefficients
33 修改/随机化 July Target 不能改变任何正式输出
34 删除 July Target 列后正式 fit 仍完全成功
35 deterministic rebuild

全部测试确定性 (固定 seed / 显式构造); 真实数据测试只读正式 model table。
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.v004c_logistic_walkforward import (
    LOGISTIC_KWARGS,
    TARGET_COLUMN,
)
from src.v004c_repair_state_spec_v002 import (
    CLOSE_DAMAGE_COMPOSITE_KEY,
    DERIVED_FACTORS,
    R1_FACTORS,
    R2_FACTORS,
    RECLAIM_COMPOSITE_KEY,
    REPAIR_PRIMITIVES,
    RepairStateError,
)
from src.v004c_repair_state_spec import (  # v001 composite (公式零复制对照)
    composite_reclaim,
)
from src.v004c_factor_spec import composite_reset
import src.v004c_repair_state_model_v002 as model_mod
from src.v004c_repair_state_model_v002 import (
    FROZEN_REPAIR_STATE_MODEL_V001,
    MODEL_FREEZE_REVIEW_REQUIRED,
    RepairStateModelError,
    fit_model,
    load_june_frame,
    load_x_window,
    pair_weighted_within_date_auc,
    read_june_target,
    within_date_auc_stats,
    x_only_sha256,
)
from tools.v004c_repair_state_june_fit import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    OUTPUT_FILES,
    check_git_preconditions,
    run_fit,
)

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_INPUT = (REPO_ROOT / "reports/research"
              / "v004c_model_table_v001_20260601_20260729"
              / "v004c_model_table_v001.csv")


# ---------------------------------------------------------------------------
# 合成数据生成器 (确定性)
# ---------------------------------------------------------------------------

def _synth_x(n: int, seed: int = 7) -> pd.DataFrame:
    """8 个 core primitive (低相关 base; d1_intraday_range 与 RECLAIM 解耦)."""
    rng = np.random.default_rng(seed)
    z_div = rng.normal(size=n)
    z_rec = rng.normal(size=n)
    z_sup = rng.normal(size=n)
    return pd.DataFrame({
        "break_open_return": z_div + rng.normal(scale=0.5, size=n),
        "d1_intraday_range": z_div + rng.normal(scale=0.3, size=n),
        "d1_open_to_close_return_raw": -z_div * 0.8 + rng.normal(scale=0.5, size=n),
        "d1_high_to_close_drawdown_raw": z_div * 0.9 + rng.normal(scale=0.4, size=n),
        "d1_close_to_vwap_raw": -z_div * 0.7 + rng.normal(scale=0.5, size=n),
        "late_day_sell_volume_ratio": z_sup + rng.normal(scale=0.5, size=n),
        "d1_low_to_close_recovery": z_rec + rng.normal(scale=0.6, size=n),
        "d1_afternoon_return": z_rec + rng.normal(scale=0.6, size=n),
    })


def synth_model_csv(directory, june_days: int = 3, july_days: int = 2,
                    rows_per_day: int = 6, seed: int = 7,
                    july_target_kind: str = "binary",
                    june_target_kind: str = "binary",
                    force_june_single_class: str = "no",
                    window_outside: bool = False,
                    june_total: int | None = None,
                    july_total: int | None = None,
                    june_positive: int | None = None) -> pathlib.Path:
    """合成 model table CSV (ID + 8 core primitive + target), 返回 csv 路径.

    july_target_kind: "binary" | "broken" (非法字符串) | "empty" ("")
    june_target_kind: 同; force_june_single_class: "positive"/"negative"
    强制 June 只有单一类别; window_outside: True 时追加窗口外行 (2026-08-01);
    june_total/july_total: 覆盖行数 (日期循环分配, 支持 173/21 精确形状);
    june_positive: 指定 June positive 精确个数 (用于 run_fit 防御检查)。
    """
    directory = pathlib.Path(directory)
    n_june = june_total if june_total is not None else june_days * rows_per_day
    n_july = july_total if july_total is not None else july_days * rows_per_day
    n = n_june + n_july
    x = _synth_x(n, seed)
    june_dates = [f"2026-06-{d:02d}" for d in range(1, june_days + 1)]
    july_dates = [f"2026-07-{d:02d}" for d in range(1, july_days + 1)]
    date_pool = ([june_dates[i % len(june_dates)] for i in range(n_june)]
                 + [july_dates[i % len(july_dates)] for i in range(n_july)])
    if window_outside:
        n = n + 2
        date_pool = date_pool + ["2026-08-01", "2026-08-01"]
        x = pd.concat([x, _synth_x(2, seed + 999)], axis=0, ignore_index=True)
    ids = pd.DataFrame({
        "event_id": [f"E{i:05d}" for i in range(n)],
        "code": [f"{600000 + i % 3000}" for i in range(n)],
        "signal_date": date_pool,
    })
    rng = np.random.default_rng(seed + 100)
    t = rng.integers(0, 2, size=n).astype(int)
    if june_positive is not None:
        t[:n_june] = 0
        t[rng.choice(n_june, size=june_positive, replace=False)] = 1
    elif force_june_single_class == "positive":
        t[:n_june] = 1
    elif force_june_single_class == "negative":
        t[:n_june] = 0
    june_vals = (["True" if v else "False" for v in t[:n_june]]
                 if june_target_kind == "binary"
                 else [june_target_kind] * n_june)
    july_vals = (["True" if v else "False" for v in t[n_june:n]]
                 if july_target_kind == "binary"
                 else [july_target_kind] * (n - n_june))
    table = pd.concat([ids, x], axis=1)
    table[TARGET_COLUMN] = june_vals + july_vals
    csv = directory / "synth_model_table.csv"
    table.to_csv(csv, index=False)
    return csv


def rewrite_target_column(src, dst, date_predicate, new_value) -> None:
    """逐行复制 CSV, 对满足 date_predicate 的行替换 target 单元格.

    X 列字节原样保留 (不改精度), 保证 X-only hash 与模型资产不受重写影响。
    """
    with open(src, newline="", encoding="utf-8-sig") as fin, \
            open(dst, "w", newline="", encoding="utf-8-sig") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        header = next(reader)
        tgt_idx = header.index(TARGET_COLUMN)
        date_idx = header.index("signal_date")
        writer.writerow(header)
        for row in reader:
            if date_predicate(row[date_idx]):
                row[tgt_idx] = new_value
            writer.writerow(row)


def _manual_primitive_params(df: pd.DataFrame) -> dict:
    """手工 FOLD_CLIP_Z (与 spec fit 相同顺序; 只用于对照)."""
    params = {}
    for col in REPAIR_PRIMITIVES:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = x[np.isfinite(x)]
        q01, q99 = np.quantile(finite, [0.01, 0.99])
        clipped = np.clip(finite, q01, q99)
        params[col] = {"q01": float(q01), "q99": float(q99),
                       "mu": float(clipped.mean()),
                       "sigma": float(clipped.std(ddof=0))}
    return params


def _manual_z(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = {}
    for col in REPAIR_PRIMITIVES:
        p = params[col]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        out[col] = (np.clip(x, p["q01"], p["q99"]) - p["mu"]) / p["sigma"]
    return pd.DataFrame(out, index=df.index)


def _manual_full_params(df: pd.DataFrame) -> dict:
    """手工完整 params (primitive + composite + derived)."""
    params = _manual_primitive_params(df)
    z = _manual_z(df, params)
    g_c = composite_reset(z)
    g_q = composite_reclaim(z)
    params[CLOSE_DAMAGE_COMPOSITE_KEY] = {
        "mu": float(g_c.mean()), "sigma": float(g_c.std(ddof=0))}
    params[RECLAIM_COMPOSITE_KEY] = {
        "mu": float(g_q.mean()), "sigma": float(g_q.std(ddof=0))}
    c_c = params[CLOSE_DAMAGE_COMPOSITE_KEY]
    c_q = params[RECLAIM_COMPOSITE_KEY]
    base = pd.DataFrame({
        "OPEN": z["break_open_return"],
        "DIVERGENCE": z["d1_intraday_range"],
        "CLOSE_DAMAGE": (g_c - c_c["mu"]) / c_c["sigma"],
        "SUPPLY": z["late_day_sell_volume_ratio"],
        "RECLAIM": (g_q - c_q["mu"]) / c_q["sigma"],
    }, index=df.index)
    for name, raw in (("DIVERGENCE_SQ", base["DIVERGENCE"] ** 2),
                      ("DIVERGENCE_X_RECLAIM",
                       base["DIVERGENCE"] * base["RECLAIM"]),
                      ("DIVERGENCE_X_DAMAGE",
                       base["DIVERGENCE"] * base["CLOSE_DAMAGE"])):
        params[name] = {"mu": float(raw.mean()), "sigma": float(raw.std(ddof=0))}
    return params, base


def _sha256_file(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rmtree(path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _real_results() -> dict:
    """真实 model table 的 fit_model 结果 (module 级缓存, 只跑一次)."""
    if not hasattr(_real_results, "_cache"):
        june, stats = load_june_frame(REAL_INPUT)
        meta = {"input": str(REAL_INPUT), "x_only_sha256": x_only_sha256(REAL_INPUT),
                "july_x_rows": stats["july_x_rows"],
                "july_x_date_count": stats["july_x_date_count"]}
        _real_results._cache = fit_model(june, meta=meta)
    return _real_results._cache


def _synth_fit(csv_path=None, **kwargs) -> dict:
    """合成 CSV 的 fit_model 结果 (每调用新建 tempdir)."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_model_"))
    csv_path = csv_path if csv_path is not None else synth_model_csv(tmp)
    june, stats = load_june_frame(csv_path)
    meta = {"input": str(csv_path), "x_only_sha256": x_only_sha256(csv_path),
            "july_x_rows": stats["july_x_rows"],
            "july_x_date_count": stats["july_x_date_count"]}
    r = fit_model(june, meta=meta)
    return r, tmp


# ---------------------------------------------------------------------------
# §28 1: branch/head 前置由运行脚本检查
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _fake_git(branch: str, head: str, fail=False):
    def run(cmd):
        if fail:
            return _FakeResult("", 1, "boom")
        if cmd == ["git", "branch", "--show-current"]:
            return _FakeResult(branch)
        if cmd == ["git", "rev-parse", "HEAD"]:
            return _FakeResult(head)
        return _FakeResult("", 1, "unknown command")
    return run


class TestGitPreconditions(unittest.TestCase):
    """§28 1: 运行脚本必须检查 branch/HEAD, 不匹配则拒绝."""

    def test_ok_when_branch_and_head_match(self):
        check_git_preconditions(run_fn=_fake_git(EXPECTED_BRANCH, EXPECTED_HEAD))

    def test_rejects_wrong_branch(self):
        with self.assertRaises(RepairStateModelError):
            check_git_preconditions(run_fn=_fake_git("main", EXPECTED_HEAD))

    def test_rejects_wrong_head(self):
        with self.assertRaises(RepairStateModelError):
            check_git_preconditions(
                run_fn=_fake_git(EXPECTED_BRANCH, "0" * 40))

    def test_rejects_git_failure(self):
        with self.assertRaises(RepairStateModelError):
            check_git_preconditions(run_fn=_fake_git("", "", fail=True))


# ---------------------------------------------------------------------------
# §28 2-5 (+7): June 切片真实数据
# ---------------------------------------------------------------------------

class TestJuneSlice(unittest.TestCase):
    """§28 2-5: June rows=173 / dates=21 / positive=59 / negative=114."""

    @classmethod
    def setUpClass(cls):
        cls.june, cls.stats = load_june_frame(REAL_INPUT)

    def test_june_rows_173(self):
        self.assertEqual(len(self.june), 173)

    def test_june_dates_21(self):
        self.assertEqual(self.june["signal_date"].nunique(), 21)

    def test_june_positive_59(self):
        self.assertEqual(int((self.june["target"] == 1).sum()), 59)

    def test_june_negative_114(self):
        self.assertEqual(int((self.june["target"] == 0).sum()), 114)

    def test_window_stats(self):
        self.assertEqual(self.stats["july_x_rows"], 160)
        self.assertEqual(self.stats["out_of_window_rows"], 0)

    def test_r0_equals_june_prevalence(self):
        """§28 7: R0 = exact June prevalence (59/173)."""
        r = _real_results()
        self.assertAlmostEqual(r["R0"]["probability"], 59.0 / 173.0,
                               places=15)
        self.assertEqual(r["june_positive"], 59)
        self.assertEqual(r["june_negative"], 114)


# ---------------------------------------------------------------------------
# §28 6: July Target 从未加载
# ---------------------------------------------------------------------------

class TestJulyBoundary(unittest.TestCase):
    """§28 6: July Target 从未解析 / 保留 / 使用."""

    def test_july_target_broken_value_ignored(self):
        """July 行 Target 为非法字符串 => 不影响 (July 值从未被解析)."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_july_"))
        try:
            csv = synth_model_csv(tmp, july_target_kind="broken")
            june, stats = load_june_frame(csv)
            self.assertEqual(len(june), 3 * 6)
            self.assertTrue(june["target"].isin([0, 1]).all())
            self.assertEqual(stats["july_x_rows"], 2 * 6)
        finally:
            _rmtree(tmp)

    def test_june_target_broken_value_fails(self):
        """June 行 Target 非法 => 显式失败 (June 值必须被解析)."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_july_"))
        try:
            src = synth_model_csv(tmp)
            broken = tmp / "broken_june.csv"
            rewrite_target_column(
                src, broken,
                lambda d: d.startswith("2026-06"), "BROKEN")
            with self.assertRaises(RepairStateModelError):
                load_june_frame(broken)
        finally:
            _rmtree(tmp)

    def test_x_loading_never_requests_target_column(self):
        """Step A 的 pandas 读取只发生 2 次 (nrows=0 + usecols), 且 usecols
        从不包含 target; Step B 用逐行 csv 流式, 不走 pandas."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_july_"))
        try:
            csv = synth_model_csv(tmp)
            real_read = pd.read_csv
            calls = []

            def spy(*args, **kwargs):
                calls.append(kwargs.get("usecols"))
                return real_read(*args, **kwargs)

            with mock.patch("pandas.read_csv", side_effect=spy):
                june, _ = load_june_frame(csv)
            self.assertEqual(len(june), 18)
            for usecols in calls:
                if usecols is not None:
                    for col in usecols:
                        self.assertNotIn("target", col.lower())
            self.assertEqual(len(calls), 2)
        finally:
            _rmtree(tmp)

    def test_read_june_target_returns_only_june_rows(self):
        """Step B 返回值只覆盖 June 行号, 无任何 July 行."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_july_"))
        try:
            csv = synth_model_csv(tmp, june_days=3, july_days=2,
                                  rows_per_day=6)
            window = load_x_window(csv)
            june_mask = (window["signal_date"] >= JUNE_START) \
                & (window["signal_date"] <= JUNE_END)
            june_rows = set(int(i) + 1 for i in np.flatnonzero(june_mask.to_numpy()))
            values = read_june_target(csv, june_rows)
            self.assertEqual(sorted(values.keys()), sorted(june_rows))
            self.assertTrue(all(v in (0, 1) for v in values.values()))
        finally:
            _rmtree(tmp)

    def test_out_of_window_rows_rejected(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_july_"))
        try:
            csv = synth_model_csv(tmp, window_outside=True)
            with self.assertRaises(RepairStateModelError):
                load_june_frame(csv)
        finally:
            _rmtree(tmp)

    def test_no_june_rows_rejected(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_july_"))
        try:
            csv = synth_model_csv(tmp, june_days=0, july_days=2)
            with self.assertRaises(RepairStateModelError):
                load_june_frame(csv)
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §28 8-14: 冻结结构 / Logistic 参数 / 无超参搜索
# ---------------------------------------------------------------------------

class TestFrozenStructure(unittest.TestCase):
    """§28 8-14: R1/R2 成员与顺序 / Logistic 固定 / 无搜索."""

    def test_r1_strict_5_factors(self):
        self.assertEqual(len(R1_FACTORS), 5)
        self.assertEqual(R1_FACTORS,
                         ("OPEN", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY",
                          "RECLAIM"))

    def test_r2_strict_8_factors(self):
        self.assertEqual(len(R2_FACTORS), 8)
        self.assertEqual(R2_FACTORS,
                         ("OPEN", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY",
                          "RECLAIM", "DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM",
                          "DIVERGENCE_X_DAMAGE"))

    def test_r2_contains_r1_first(self):
        self.assertEqual(R2_FACTORS[:5], R1_FACTORS)

    def test_forbidden_factors_not_in_core(self):
        for forbidden in ("DIVERGENCE_X_SUPPLY", "TURNOVER_COST", "HIGHZONE",
                          "MOM7", "DAMAGE7", "REGIME", "POS7", "TREND"):
            self.assertNotIn(forbidden, R1_FACTORS)
            self.assertNotIn(forbidden, R2_FACTORS)

    def test_logistic_params_frozen(self):
        """§28 12/13: Logistic 参数完全固定, class_weight=None."""
        self.assertEqual(LOGISTIC_KWARGS, {
            "penalty": "l2", "C": 1.0, "solver": "lbfgs",
            "fit_intercept": True, "class_weight": None, "max_iter": 1000})
        self.assertIsNone(LOGISTIC_KWARGS["class_weight"])
        frozen = _real_results()["frozen"]
        alg = frozen["algorithm"]
        self.assertEqual(alg["penalty"], "l2")
        self.assertEqual(alg["C"], 1.0)
        self.assertEqual(alg["solver"], "lbfgs")
        self.assertTrue(alg["fit_intercept"])
        self.assertIsNone(alg["class_weight"])
        self.assertEqual(alg["max_iter"], 1000)
        self.assertIsNone(alg["random_state"])

    def test_no_hyperparameter_search_in_source(self):
        """§28 14: 源码无超参搜索实现 (搜索构造器 / CV / 调参循环).

        注意: review 文本和 docstring 会合法地提到 "GridSearch" 等禁止词,
        因此扫描的是实际实现构造 (构造器调用 / import / 调参循环), 不是单词。
        """
        for rel in ("src/v004c_repair_state_model_v002.py",
                    "tools/v004c_repair_state_june_fit.py"):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            for token in ("GridSearchCV(", "RandomizedSearchCV(",
                          "HalvingGridSearchCV", "BayesSearchCV",
                          "optuna.", "cross_val_score(", "ParameterGrid(",
                          "import optuna"):
                self.assertNotIn(token, text, f"{rel} 出现 {token}")

    def test_frozen_json_structure(self):
        """§25: frozen JSON 的必需键全部存在."""
        frozen = _real_results()["frozen"]
        for key in ("model_version", "factor_spec_version", "training_period",
                    "training_rows", "training_dates", "target_column",
                    "algorithm", "R1", "R2", "primitive_transforms",
                    "composite_transforms", "derived_transforms", "ranking",
                    "research_status"):
            self.assertIn(key, frozen)
        self.assertEqual(frozen["model_version"], "v004c_repair_state_model_v001")
        self.assertEqual(frozen["factor_spec_version"],
                         "v004c_repair_state_spec_v002")
        self.assertEqual(frozen["training_rows"], 173)
        self.assertEqual(frozen["training_dates"], 21)
        self.assertEqual(frozen["target_column"], TARGET_COLUMN)
        self.assertEqual(frozen["R1"]["factor_order"], list(R1_FACTORS))
        self.assertEqual(frozen["R2"]["factor_order"], list(R2_FACTORS))
        self.assertEqual(frozen["ranking"],
                         {"probability": "descending",
                          "tie_break": "event_id ascending"})
        rs = frozen["research_status"]
        self.assertTrue(rs["post_june_hypothesis"])
        self.assertFalse(rs["june_is_validation"])
        self.assertFalse(rs["july_target_seen"])
        self.assertEqual(
            set(frozen["primitive_transforms"].keys()), set(REPAIR_PRIMITIVES))
        self.assertEqual(set(frozen["composite_transforms"].keys()),
                         {"CLOSE_DAMAGE", "RECLAIM"})
        self.assertEqual(set(frozen["derived_transforms"].keys()),
                         set(DERIVED_FACTORS))


class TestFitUsesSpecOnly(unittest.TestCase):
    """§28 15/20: transform 只 fit June; spec 只调用一次 (model 不重算)."""

    def test_spec_fit_called_once_with_june_only(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_spy_"))
        try:
            csv = synth_model_csv(tmp)
            june, _ = load_june_frame(csv)
            with mock.patch.object(
                    model_mod, "fit_repair_state_transform_v2",
                    wraps=model_mod.fit_repair_state_transform_v2) as fit_spy, \
                    mock.patch.object(
                        model_mod, "construct_repair_factors_v2",
                        wraps=model_mod.construct_repair_factors_v2) as cons_spy:
                r = fit_model(june, meta={"input": str(csv)})
            self.assertEqual(fit_spy.call_count, 1)
            # construct 被调用 2 次: 1) factor 构造 2) frozen JSON 重建验证;
            # 两者都不是"重新 fit" (fit_repair_state_transform_v2 只调用一次)
            self.assertEqual(cons_spy.call_count, 2)
            fit_df = fit_spy.call_args[0][0]
            dates = pd.unique(fit_df["signal_date"])
            self.assertTrue((dates >= JUNE_START).all())
            self.assertTrue((dates <= JUNE_END).all())
            self.assertEqual(r["status"], FROZEN_REPAIR_STATE_MODEL_V001)
        finally:
            _rmtree(tmp)

    def test_params_match_manual_clip_first(self):
        """§28 16-19: params 与手工 FOLD_CLIP_Z (clip-first) 完全一致;
        derived 无 q01/q99 (no second clip)."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_clip_"))
        try:
            csv = synth_model_csv(tmp)
            june, _ = load_june_frame(csv)
            r = fit_model(june)
            manual, _ = _manual_full_params(june)
            for col in REPAIR_PRIMITIVES:
                p, m = r["params"][col], manual[col]
                for k in ("q01", "q99", "mu", "sigma"):
                    self.assertEqual(p[k], m[k], f"primitive {col}.{k}")
            for key in (CLOSE_DAMAGE_COMPOSITE_KEY, RECLAIM_COMPOSITE_KEY):
                p, m = r["params"][key], manual[key]
                for k in ("mu", "sigma"):
                    self.assertEqual(p[k], m[k], f"composite {key}.{k}")
            for name in DERIVED_FACTORS:
                p, m = r["params"][name], manual[name]
                for k in ("mu", "sigma"):
                    self.assertEqual(p[k], m[k], f"derived {name}.{k}")
                self.assertNotIn("q01", r["params"][name])
                self.assertNotIn("q99", r["params"][name])
        finally:
            _rmtree(tmp)

    def test_factors_match_manual_definitions(self):
        """§28 17/18: CLOSE_DAMAGE / RECLAIM 等 factor 与 v002 定义逐行相等."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_clip_"))
        try:
            csv = synth_model_csv(tmp)
            june, _ = load_june_frame(csv)
            r = fit_model(june)
            manual, manual_base = _manual_full_params(june)
            for col in ("OPEN", "DIVERGENCE", "CLOSE_DAMAGE", "SUPPLY",
                        "RECLAIM"):
                np.testing.assert_array_equal(
                    r["factors"][col].to_numpy(dtype=float),
                    manual_base[col].to_numpy(dtype=float))
            for name in DERIVED_FACTORS:
                m = manual[name]
                self.assertNotIn("q01", m)
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §28 21-25: 系数 / 概率 / ranking
# ---------------------------------------------------------------------------

class TestFitOutput(unittest.TestCase):
    """§28 21-25: coefficients finite / 0<p<1 / ranking 规则."""

    def test_coefficients_finite(self):
        """§28 21: 全部系数 finite."""
        r = _real_results()
        for model in ("R1", "R2"):
            coefs = np.asarray([r[model]["coefs"]["INTERCEPT"]]
                               + [r[model]["coefs"][f] for f in
                                  (R1_FACTORS if model == "R1" else R2_FACTORS)],
                               dtype=float)
            self.assertTrue(np.all(np.isfinite(coefs)))
            self.assertTrue(r["health"][model]["coefficients_finite"])

    def test_probabilities_finite_in_open_interval(self):
        """§28 22: probability finite 且 0 < p < 1."""
        r = _real_results()
        for model in ("R1", "R2"):
            p = r["predictions"][f"{model.lower()}_probability"].to_numpy(
                dtype=float)
            self.assertTrue(np.all(np.isfinite(p)))
            self.assertTrue(np.all(p > 0.0))
            self.assertTrue(np.all(p < 1.0))
            self.assertLess(r["health"][f"{model}_probability"]["min"], 1.0)

    def test_ranking_tie_breaks_by_event_id(self):
        """§28 23: 同日同概率 -> event_id 升序 (tie-break 确定性)."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_rank_"))
        try:
            # 6 行同日期; 第 0/1 行 X 完全相同 (事件 E100/E010 乱序),
            # 其余 4 行与它们以及彼此不同 (保证 composite 不退化)
            base = _synth_x(6, seed=5)
            df = pd.concat([base.iloc[[0]], base.iloc[[0]], base.iloc[[2]],
                            base.iloc[[3]], base.iloc[[4]], base.iloc[[5]]],
                           ignore_index=True)
            df["event_id"] = ["E100", "E010", "E001", "E002", "E003", "E004"]
            df["code"] = ["600000"] * 6
            df["signal_date"] = ["2026-06-01"] * 6
            df["target"] = [1, 0, 1, 0, 1, 0]
            r = fit_model(df)
            preds = r["predictions"].set_index("event_id")
            for model in ("R1", "R2"):
                rk = preds[f"{model.lower()}_rank_daily"]
                p = preds[f"{model.lower()}_probability"]
                # E010 与 E100 X 相同 => 概率相同 => rank 按 event_id 升序
                self.assertAlmostEqual(p["E010"], p["E100"], places=12)
                self.assertLess(int(rk["E010"]), int(rk["E100"]))
                # rank 覆盖 1..6, 每行一个确定 rank
                self.assertEqual(sorted(int(v) for v in rk.values),
                                 list(range(1, 7)))
        finally:
            _rmtree(tmp)

    def test_r0_ranking_na(self):
        """§28 24: R0 的 ranking / within-date 字段 = NA."""
        r = _real_results()
        row = r["metrics"].set_index("model").loc["R0"]
        self.assertEqual(row["top1_target_rate"], "NA")
        self.assertEqual(row["top3_target_rate"], "NA")
        self.assertEqual(row["valid_auc_dates"], "NA")
        self.assertEqual(row["pair_weighted_within_date_auc"], "NA")

    def test_target_not_used_for_ranking(self):
        """§28 25: ranking 只按 probability + event_id; 调换 Target 排名不变."""
        r = _real_results()
        preds = r["predictions"]
        d = preds["signal_date"].iloc[0]
        sub = preds[preds["signal_date"] == d].copy()
        swapped = sub.copy()
        swapped[TARGET_COLUMN] = 1 - swapped[TARGET_COLUMN].astype(int)
        rank_a = model_mod.daily_rank(sub, "r1_probability")
        rank_b = model_mod.daily_rank(swapped, "r1_probability")
        pd.testing.assert_series_equal(rank_a, rank_b)

    def test_r0_probability_constant_across_rows(self):
        preds = _real_results()["predictions"]
        self.assertTrue(
            (preds["r0_probability"] == preds["r0_probability"].iloc[0]).all())


# ---------------------------------------------------------------------------
# §28 26-27: within-date AUC
# ---------------------------------------------------------------------------

class TestWithinDate(unittest.TestCase):
    """§28 26/27: daily AUC 只对同日正负并存计算; pair-weighted 正确."""

    def test_single_class_date_excluded_from_summary(self):
        """§28 26: 同日只有单一类别 => 不计入 daily AUC 汇总."""
        daily = pd.DataFrame({
            "signal_date": ["2026-06-01", "2026-06-02", "2026-06-03"],
            "r1_auc_valid": [True, False, True],
            "r1_daily_auc": [0.6, np.nan, 0.4],
        })
        s = within_date_auc_stats(daily, "r1")
        self.assertEqual(s["valid_auc_dates"], 2)
        self.assertAlmostEqual(s["mean_daily_auc"], 0.5)
        self.assertAlmostEqual(s["median_daily_auc"], 0.5)
        self.assertEqual(s["dates_auc_gt_0_5"], 1)
        self.assertEqual(s["dates_auc_eq_0_5"], 0)
        self.assertEqual(s["dates_auc_lt_0_5"], 1)

    def test_pair_weighted_manual(self):
        """§28 27: pair-weighted within-date AUC 手工对照 (跨日对不计)."""
        preds = pd.DataFrame({
            "signal_date": ["2026-06-01"] * 3 + ["2026-06-02"] * 3,
            TARGET_COLUMN: [1, 1, 0, 1, 0, 0],
            "prob": [0.9, 0.8, 0.2, 0.5, 0.5, 0.4],
        })
        # d1: pos={0.9,0.8} vs neg={0.2} -> 2/2
        # d2: pos={0.5} vs neg={0.5,0.4} -> (0.5 tie)*0.5 + 1.0 = 1.5/2
        # 总计 (2 + 1.5) / (2 + 2) = 0.875
        self.assertAlmostEqual(
            pair_weighted_within_date_auc(preds, "prob"), 0.875)

    def test_pair_weighted_no_cross_date_pairs(self):
        """d1 全正 + d2 全负 => 无同日对 => nan."""
        preds = pd.DataFrame({
            "signal_date": ["2026-06-01"] * 2 + ["2026-06-02"] * 2,
            TARGET_COLUMN: [1, 1, 0, 0],
            "prob": [0.9, 0.8, 0.2, 0.1],
        })
        self.assertTrue(np.isnan(pair_weighted_within_date_auc(preds, "prob")))

    def test_daily_auc_in_daily_ranking_matches_sklearn(self):
        """fit 后的 daily AUC 与 sklearn roc_auc_score 手工对照."""
        r, tmp = _synth_fit()
        try:
            preds = r["predictions"]
            daily = r["daily_ranking"]
            for d in daily["signal_date"]:
                sub = preds[preds["signal_date"] == d]
                y = sub[TARGET_COLUMN].to_numpy(dtype=int)
                valid = (y == 1).any() and (y == 0).any()
                row = daily[daily["signal_date"] == d].iloc[0]
                self.assertEqual(bool(row["r1_auc_valid"]), valid)
                if valid:
                    p = sub["r1_probability"].to_numpy(dtype=float)
                    from sklearn.metrics import roc_auc_score
                    self.assertAlmostEqual(row["r1_daily_auc"],
                                           roc_auc_score(y, p))
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §28 28: interaction contribution
# ---------------------------------------------------------------------------

class TestInteractionAudit(unittest.TestCase):
    """§28 28: contribution_j = coefficient_j * factor_j 计算正确."""

    def test_contribution_equals_coefficient_times_factor(self):
        r, tmp = _synth_fit()
        try:
            audit = r["interaction"]["audit_df"]
            contrib = audit["contribution"].to_numpy(dtype=float)
            manual = (audit["coefficient"].to_numpy(dtype=float)
                      * audit["factor_value"].to_numpy(dtype=float))
            np.testing.assert_allclose(contrib, manual, rtol=1e-12)
            self.assertTrue(
                (audit["abs_contribution"].to_numpy(dtype=float)
                 == np.abs(contrib)).all())
        finally:
            _rmtree(tmp)

    def test_max_abs_contribution_is_row_max(self):
        r, tmp = _synth_fit()
        try:
            factors = r["factors"]
            r2 = r["R2"]
            for f in R2_FACTORS:
                manual_max = float(
                    np.max(np.abs(r2["coefs"][f] * factors[f].to_numpy(dtype=float))))
                self.assertAlmostEqual(
                    r["interaction"]["max_per_factor"][f]["max_abs_contribution"],
                    manual_max, places=12)
        finally:
            _rmtree(tmp)

    def test_top10_sections_have_10_rows(self):
        r, tmp = _synth_fit()
        try:
            audit = r["interaction"]["audit_df"]
            for name in ("DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_DAMAGE"):
                sub = audit[audit["section"] == f"top10_{name}"]
                self.assertEqual(len(sub), 10)
                self.assertEqual(len(sub), sub["rank"].nunique())
        finally:
            _rmtree(tmp)

    def test_no_interaction_anomalies_on_synth(self):
        r, tmp = _synth_fit()
        try:
            self.assertEqual(r["interaction"]["anomalies"], [])
        finally:
            _rmtree(tmp)


# ---------------------------------------------------------------------------
# §28 29-31: frozen JSON 重建预测
# ---------------------------------------------------------------------------

class TestFrozenReproduction(unittest.TestCase):
    """§28 29/30/31: 从 JSON 重建 R1/R2 预测, 误差 < 1e-12."""

    def test_rebuild_from_json_round_trip(self):
        r = _real_results()
        frozen = json.loads(json.dumps(r["frozen"]))  # 模拟磁盘 round-trip
        from src.v004c_repair_state_model_v002 import (
            manual_logistic_predict,
            spec_params_from_frozen,
        )
        june, _ = load_june_frame(REAL_INPUT)
        from src.v004c_repair_state_spec_v002 import construct_repair_factors_v2
        params = spec_params_from_frozen(frozen)
        factors = construct_repair_factors_v2(june, params)
        p1r = manual_logistic_predict(frozen["R1"], factors)
        p2r = manual_logistic_predict(frozen["R2"], factors)
        p1 = r["predictions"]["r1_probability"].to_numpy(dtype=float)
        p2 = r["predictions"]["r2_probability"].to_numpy(dtype=float)
        self.assertLess(float(np.max(np.abs(p1r - p1))), 1e-12)
        self.assertLess(float(np.max(np.abs(p2r - p2))), 1e-12)
        self.assertLess(r["repro_errors"]["R1"], 1e-12)
        self.assertLess(r["repro_errors"]["R2"], 1e-12)

    def test_frozen_status(self):
        r = _real_results()
        self.assertEqual(r["status"], FROZEN_REPAIR_STATE_MODEL_V001)
        self.assertEqual(r["status_reasons"], [])


# ---------------------------------------------------------------------------
# §28 32-35: leakage 不变性 / deterministic rebuild
# ---------------------------------------------------------------------------

class TestLeakageInvariance(unittest.TestCase):
    """§28 32/33/34: July Target 随机化 / 删除 / July X 修改的不变性."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_leak_"))
        # run_fit 有硬编码防御检查 (173/21/59/114) => 用精确 June 形状;
        # 四个版本必须使用同一个输入文件名 (审计资产记录 input 路径)
        staging = synth_model_csv(cls.tmp, seed=42,
                                  june_days=21, june_total=173,
                                  june_positive=59,
                                  july_days=21, july_total=160)
        cls.table = cls.tmp / "model_table.csv"
        shutil.copyfile(staging, cls.table)
        cls.dir_a = cls.tmp / "A"
        cls.dir_b = cls.tmp / "B"
        cls.dir_c = cls.tmp / "C"
        cls.dir_d = cls.tmp / "D"
        # A: 原 July Target
        cls.hash_a = x_only_sha256(cls.table)
        cls.r_a = run_fit(cls.table, cls.dir_a, git_check=False)
        # B: July Target 随机化 (固定 seed, 确定性; 原地改写再 fit)
        rng = np.random.default_rng(2026)
        n_july = _count_rows(cls.table, lambda d: d.startswith("2026-07"))
        july_vals = ["True" if v else "False"
                     for v in rng.integers(0, 2, size=n_july)]
        tmp_b = cls.tmp / "_b.csv"
        _rewrite_july_target(cls.table, tmp_b, july_vals)
        shutil.move(str(tmp_b), cls.table)
        cls.hash_b = x_only_sha256(cls.table)
        cls.r_b = run_fit(cls.table, cls.dir_b, git_check=False)
        # C: July 行 Target 置空 (删除语义)
        tmp_c = cls.tmp / "_c.csv"
        rewrite_target_column(cls.table, tmp_c,
                              lambda d: d.startswith("2026-07"), "")
        shutil.move(str(tmp_c), cls.table)
        cls.hash_c = x_only_sha256(cls.table)
        cls.r_c = run_fit(cls.table, cls.dir_c, git_check=False)
        # D: July X 修改 (d1_intraday_range 全变)
        tmp_d = cls.tmp / "_d.csv"
        _rewrite_july_x(cls.table, tmp_d, "d1_intraday_range")
        shutil.move(str(tmp_d), cls.table)
        cls.hash_d = x_only_sha256(cls.table)
        cls.r_d = run_fit(cls.table, cls.dir_d, git_check=False)

    @classmethod
    def tearDownClass(cls):
        _rmtree(cls.tmp)

    def test_july_target_randomization_invariance(self):
        """§28 33: July Target 随机化 => 9 个资产 byte-identical."""
        for name in OUTPUT_FILES:
            self.assertEqual(
                _sha256_file(self.dir_a / name),
                _sha256_file(self.dir_b / name),
                f"Version B 资产变化: {name}")

    def test_july_target_removal_invariance(self):
        """§28 34: July 行 Target 删除/置空 => fit 成功且 byte-identical."""
        for name in OUTPUT_FILES:
            self.assertEqual(
                _sha256_file(self.dir_a / name),
                _sha256_file(self.dir_c / name),
                f"Version C 资产变化: {name}")

    def test_july_x_mutation_june_coefficients_unchanged(self):
        """§28 32: 修改 July X 不能改变 June coefficients."""
        for model in ("R1", "R2"):
            a = json.loads((self.dir_a / OUTPUT_FILES[7]).read_text(
                encoding="utf-8"))[model]["coefficients"]
            d = json.loads((self.dir_d / OUTPUT_FILES[7]).read_text(
                encoding="utf-8"))[model]["coefficients"]
            for f in a:
                self.assertEqual(a[f], d[f], f"{model}.{f}")

    def test_june_predictions_identical_across_versions(self):
        """§28 33 补充: June predictions 在 A/B/C 中逐位一致."""
        for name in ("A", "B", "C"):
            pa = pd.read_csv(self.dir_a / OUTPUT_FILES[3])
            pb = pd.read_csv(getattr(self, f"dir_{name.lower()}") / OUTPUT_FILES[3])
            np.testing.assert_array_equal(
                pa["r1_probability"].to_numpy(dtype=float),
                pb["r1_probability"].to_numpy(dtype=float))
            np.testing.assert_array_equal(
                pa["r2_probability"].to_numpy(dtype=float),
                pb["r2_probability"].to_numpy(dtype=float))

    def test_x_only_hash_ignores_july_target(self):
        self.assertEqual(self.hash_a, self.hash_b)
        self.assertEqual(self.hash_a, self.hash_c)
        self.assertNotEqual(self.hash_a, self.hash_d)


def _count_rows(src, date_predicate) -> int:
    with open(src, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        date_idx = header.index("signal_date")
        return sum(1 for row in reader if date_predicate(row[date_idx]))


def _rewrite_july_target(src, dst, july_values):
    """把 July 行 target 替换为给定值列表 (逐行复制, X 字节不变)."""
    it = iter(july_values)
    with open(src, newline="", encoding="utf-8-sig") as fin, \
            open(dst, "w", newline="", encoding="utf-8-sig") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        header = next(reader)
        tgt_idx = header.index(TARGET_COLUMN)
        date_idx = header.index("signal_date")
        writer.writerow(header)
        for row in reader:
            if row[date_idx].startswith("2026-07"):
                row[tgt_idx] = next(it)
            writer.writerow(row)


def _rewrite_july_x(src, dst, x_col):
    """把 July 行的 x_col 单元格改写 (+5.0), 其余单元格原样复制."""
    with open(src, newline="", encoding="utf-8-sig") as fin, \
            open(dst, "w", newline="", encoding="utf-8-sig") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        header = next(reader)
        tgt_idx = header.index(x_col)
        date_idx = header.index("signal_date")
        writer.writerow(header)
        for row in reader:
            if row[date_idx].startswith("2026-07"):
                row[tgt_idx] = repr(float(row[tgt_idx]) + 5.0)
            writer.writerow(row)


class TestDeterministicRebuild(unittest.TestCase):
    """§28 35: 两次运行 9 个资产字节级一致."""

    def test_two_runs_byte_identical(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="repair_det_"))
        try:
            src = synth_model_csv(tmp, seed=99,
                                  june_days=21, june_total=173,
                                  june_positive=59,
                                  july_days=21, july_total=160)
            d1 = tmp / "run1"
            d2 = tmp / "run2"
            run_fit(src, d1, git_check=False)
            run_fit(src, d2, git_check=False)
            for name in OUTPUT_FILES:
                self.assertEqual(
                    _sha256_file(d1 / name),
                    _sha256_file(d2 / name),
                    f"rebuild 资产变化: {name}")
        finally:
            _rmtree(tmp)


if __name__ == "__main__":
    unittest.main()

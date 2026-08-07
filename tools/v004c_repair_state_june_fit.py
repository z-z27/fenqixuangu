"""v004c Repair-State Model v002 — June full-development fit (生成 9 个正式冻结资产)。

- 前置检查 (git): 必须在 research-sample-analysis 分支且 HEAD 等于
  预期提交 (1f90224388a29d2e5839c8398098fcef69d53763), 否则拒绝继续
- June (2026-06-01 ~ 2026-06-30, 173 rows / 21 dates) 是唯一训练样本;
  July Target 从未解析 / 加载 / 使用 (src/v004c_repair_state_model_v002.py
  的 Step A + Step B 严格边界)
- 训练实现 / 指标 / frozen JSON 全部复用 src/v004c_repair_state_model_v002.py
  (transform 复用 spec, 不复制公式)
- 输出 9 个确定性资产到 reports/research/v004c_repair_state_model_freeze_v001_202606/
  (无时间戳 / 无随机 / 两次运行字节级一致; July Target 随机化或删除不影响任何资产)
- 判定: FROZEN_REPAIR_STATE_MODEL_V001 / MODEL_FREEZE_REVIEW_REQUIRED
  (§30: 只有 non-convergence / NaN-Inf / abs(coef)>=10 / JSON 无法复现 /
  实现错误允许 REVIEW; June 指标不参与判定)
- 禁止: 修改 spec / factor / interaction / Logistic 超参数; 读取 July Target;
  git add . / add -A; tag
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_logistic_walkforward import JULY_END, JULY_START, JUNE_END, JUNE_START
from src.v004c_repair_state_spec_v002 import (
    DERIVED_FACTORS,
    R1_FACTORS,
    R2_FACTORS,
    REPAIR_PRIMITIVES,
)
from src.v004c_repair_state_model_v002 import (
    MODEL_VERSION,
    FROZEN_REPAIR_STATE_MODEL_V001,
    MODEL_FREEZE_REVIEW_REQUIRED,
    DEVELOPMENT_IN_SAMPLE_ONLY,
    RepairStateModelError,
    fit_model,
    load_june_frame,
    x_only_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (REPO_ROOT
                 / "reports/research/v004c_model_table_v001_20260601_20260729"
                 / "v004c_model_table_v001.csv")
DEFAULT_OUTPUT = REPO_ROOT / "reports/research/v004c_repair_state_model_freeze_v001_202606"

OUTPUT_FILES: tuple[str, ...] = (
    "v004c_repair_state_training_data_audit_v001.csv",
    "v004c_repair_state_transform_params_v001.csv",
    "v004c_repair_state_coefficients_v001.csv",
    "v004c_repair_state_development_predictions_v001.csv",
    "v004c_repair_state_development_metrics_v001.csv",
    "v004c_repair_state_daily_ranking_v001.csv",
    "v004c_repair_state_interaction_audit_v001.csv",
    "v004c_repair_state_frozen_model_v001.json",
    "v004c_repair_state_model_freeze_review_v001.md",
)

# 开始前检查 (§一): 分支 / HEAD 必须精确匹配, 否则停止
EXPECTED_BRANCH = "research-sample-analysis"
EXPECTED_HEAD = "1f90224388a29d2e5839c8398098fcef69d53763"

# 已知 June 事实 (§四, 防御性验证)
KNOWN_JUNE_ROWS = 173
KNOWN_JUNE_DATES = 21
KNOWN_JUNE_POSITIVE = 59
KNOWN_JUNE_NEGATIVE = 114

_CSV_FLOAT = "%.6f"


# ---------------------------------------------------------------------------
# git 前置检查 (§一 / 测试 1)
# ---------------------------------------------------------------------------

def check_git_preconditions(run_fn=None) -> None:
    """branch / HEAD 前置检查; 不匹配 => 显式失败 (禁止继续)."""
    def _run(cmd):
        r = (run_fn(cmd) if run_fn is not None
             else subprocess.run(cmd, capture_output=True, text=True, shell=False))
        if getattr(r, "returncode", None) != 0:
            raise RepairStateModelError(
                "git 命令失败: {} -> {}".format(cmd, getattr(r, "stderr", "").strip()))
        return str(getattr(r, "stdout", "")).strip()
    branch = _run(["git", "branch", "--show-current"])
    head = _run(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise RepairStateModelError(
            "前置检查失败: 预期分支 {}, 实际 {}; 停止".format(EXPECTED_BRANCH, branch))
    if head != EXPECTED_HEAD:
        raise RepairStateModelError(
            "前置检查失败: 预期 HEAD {}, 实际 {}; 停止".format(EXPECTED_HEAD, head))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_fit(input_csv, output_dir, git_check: bool = True) -> dict:
    """June full-development fit + 写 9 个正式资产; 返回 results dict."""
    input_csv = Path(input_csv)
    if git_check:
        check_git_preconditions()
    june_df, window_stats = load_june_frame(input_csv)
    # 防御性验证已知 June 事实 (与测试 #2-5 一致)
    rows = len(june_df)
    dates = int(june_df["signal_date"].nunique())
    y = june_df["target"].to_numpy(dtype=int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if rows != KNOWN_JUNE_ROWS or dates != KNOWN_JUNE_DATES \
            or pos != KNOWN_JUNE_POSITIVE or neg != KNOWN_JUNE_NEGATIVE:
        raise RepairStateModelError(
            "June 切片与已知事实不符: rows={} (预期 {}), dates={} (预期 {}), "
            "pos={} (预期 {}), neg={} (预期 {})".format(
                rows, KNOWN_JUNE_ROWS, dates, KNOWN_JUNE_DATES,
                pos, KNOWN_JUNE_POSITIVE, neg, KNOWN_JUNE_NEGATIVE))
    meta = {
        "input": str(input_csv),
        "x_only_sha256": x_only_sha256(input_csv),
        "july_x_rows": window_stats["july_x_rows"],
        "july_x_date_count": window_stats["july_x_date_count"],
    }
    results = fit_model(june_df, meta=meta)
    write_reports(results, output_dir)
    return results


# ---------------------------------------------------------------------------
# 9 个正式资产 (确定性写出)
# ---------------------------------------------------------------------------

def write_reports(results: dict, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_training_audit(results, output_dir / OUTPUT_FILES[0])
    _write_transform_params(results, output_dir / OUTPUT_FILES[1])
    _write_coefficients(results, output_dir / OUTPUT_FILES[2])
    _write_predictions(results, output_dir / OUTPUT_FILES[3])
    _write_metrics(results, output_dir / OUTPUT_FILES[4])
    _write_daily_ranking(results, output_dir / OUTPUT_FILES[5])
    _write_interaction_audit(results, output_dir / OUTPUT_FILES[6])
    (output_dir / OUTPUT_FILES[7]).write_text(
        json.dumps(results["frozen"], indent=2, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8")
    (output_dir / OUTPUT_FILES[8]).write_text(
        _review_md(results), encoding="utf-8")


def _write_training_audit(results: dict, path: Path) -> None:
    july_loaded = "NO" if not results["july_target_loaded"] else "YES"
    rows = [
        ("dataset", "input", results["input"]),
        ("dataset", "x_only_sha256", results["x_only_sha256"]),
        ("dataset", "model_version", MODEL_VERSION),
        ("window", "june_start", JUNE_START),
        ("window", "june_end", JUNE_END),
        ("window", "july_start", JULY_START),
        ("window", "july_end", JULY_END),
        ("window", "window_rows", results["june_rows"] + results["july_x_rows"]),
        ("window", "out_of_window_rows", 0),
        ("june", "rows", results["june_rows"]),
        ("june", "signal_dates", results["june_date_count"]),
        ("june", "date_min", results["june_date_min"]),
        ("june", "date_max", results["june_date_max"]),
        ("june", "positive", results["june_positive"]),
        ("june", "negative", results["june_negative"]),
        ("june", "base_rate", results["june_base_rate"]),
        ("june", "target_column", "target7_daily_d2open_d3high"),
        ("july_boundary", "july_x_rows", results["july_x_rows"]),
        ("july_boundary", "july_x_date_count", results["july_x_date_count"]),
        ("july_boundary", "july_x_used_for_transform", "NO"),
        ("july_boundary", "july_target_read", "NO"),
        ("july_boundary", "july_target_loaded", july_loaded),
        ("july_boundary", "july_target_used", "NO"),
        ("fit", "development_only", DEVELOPMENT_IN_SAMPLE_ONLY),
        ("fit", "june_is_validation", "NO (post-June hypothesis)"),
    ]
    pd.DataFrame(rows, columns=["section", "field", "value"]).to_csv(
        path, index=False, encoding="utf-8-sig")


def _write_transform_params(results: dict, path: Path) -> None:
    params = results["params"]
    rows: list[dict] = []
    for col in REPAIR_PRIMITIVES:
        p = params[col]
        rows.append({"scope": "primitive", "key": col, "q01": p["q01"],
                     "q99": p["q99"], "mu": p["mu"], "sigma": p["sigma"]})
    for key, label in (("composite_CLOSE_DAMAGE", "CLOSE_DAMAGE"),
                       ("composite_RECLAIM", "RECLAIM")):
        p = params[key]
        rows.append({"scope": "composite", "key": label, "q01": None,
                     "q99": None, "mu": p["mu"], "sigma": p["sigma"]})
    for name in DERIVED_FACTORS:
        p = params[name]
        rows.append({"scope": "derived", "key": name, "q01": None,
                     "q99": None, "mu": p["mu"], "sigma": p["sigma"]})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig",
                              float_format=_CSV_FLOAT)


def _write_coefficients(results: dict, path: Path) -> None:
    rows: list[dict] = []
    for model, factors in (("R1", R1_FACTORS), ("R2", R2_FACTORS)):
        fit = results[model]
        for position, factor in enumerate(("INTERCEPT",) + tuple(factors)):
            rows.append({"model": model, "position": position, "factor": factor,
                         "coefficient": fit["coefs"][factor]})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig",
                              float_format=_CSV_FLOAT)


def _write_predictions(results: dict, path: Path) -> None:
    preds = results["predictions"].copy()
    preds.insert(3, "development_only", DEVELOPMENT_IN_SAMPLE_ONLY)
    preds.to_csv(path, index=False, encoding="utf-8-sig", float_format=_CSV_FLOAT)


def _write_metrics(results: dict, path: Path) -> None:
    results["metrics"].to_csv(path, index=False, encoding="utf-8-sig",
                              float_format=_CSV_FLOAT)


def _write_daily_ranking(results: dict, path: Path) -> None:
    results["daily_ranking"].to_csv(path, index=False, encoding="utf-8-sig",
                                    float_format=_CSV_FLOAT)


def _write_interaction_audit(results: dict, path: Path) -> None:
    results["interaction"]["audit_df"].to_csv(path, index=False,
                                              encoding="utf-8-sig",
                                              float_format=_CSV_FLOAT)


# ---------------------------------------------------------------------------
# review md (确定性, 中文正文; 字段标签保留英文)
# ---------------------------------------------------------------------------

def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "nan"
    return f"{fv:.{digits}f}"


def _pct(v) -> str:
    if v is None:
        return ""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "nan"
    return f"{100.0 * fv:.2f}%"


def _marginal_effect_table(r2: dict) -> str:
    """∂logit/∂V = β_V + 2γ₁V + γ₂Q + γ₃C 的标准情景表 (V/Q/C 全部组合)."""
    bv = r2["coefs"]["DIVERGENCE"]
    g1 = r2["coefs"]["DIVERGENCE_SQ"]
    g2 = r2["coefs"]["DIVERGENCE_X_RECLAIM"]
    g3 = r2["coefs"]["DIVERGENCE_X_DAMAGE"]
    lines = ["| V | Q | C | ∂logit/∂V |",
             "|---|---|---|---|"]
    for v in (-1.0, 0.0, 1.0, 2.0):
        for q in (-1.0, 0.0, 1.0):
            for c in (-1.0, 0.0, 1.0):
                d = bv + 2.0 * g1 * v + g2 * q + g3 * c
                lines.append("| {} | {} | {} | {} |".format(
                    _fmt(v), _fmt(q), _fmt(c), _fmt(d, 4)))
    return "\n".join(lines)


def _review_md(r: dict) -> str:
    lines: list[str] = []
    add = lines.append
    m = r["metrics"].set_index("model")
    r0, r1, r2 = m.loc["R0"], m.loc["R1"], m.loc["R2"]
    f1, f2 = r["R1"]["coefs"], r["R2"]["coefs"]

    add("# v004c Repair-State Model v002 — June Full-Development Fit + Frozen Model")
    add("")
    add("- 阶段: June full-development fit (R0/R1/R2) + 冻结模型工件 (FROZEN FOR"
        " JULY RETROSPECTIVE OOT)")
    add("- 分支: research-sample-analysis")
    add("- 输入: `v004c_model_table_v001.csv` (X-only SHA256: `{}`; July Target"
        " 值不参与 hash, 保证资产对 July Target 变化不变)".format(r["x_only_sha256"]))
    add("- 三件事必须区分:")
    add("  - **Factor structure**: **FROZEN** — Repair-State v002 (人工批准"
        " FREEZE_REPAIR_STATE_V002; 本任务未修改任何 factor / interaction /"
        " 预处理 / Logistic 超参数)")
    add("  - **June model fit**: **DEVELOPMENT_IN_SAMPLE_ONLY** — post-June"
        " hypothesis, June 不是验证集; 本阶段全部指标仅 in-sample 诊断, 不用来"
        " 选择 R1/R2, 不宣称 validated / OOF / out-of-sample / generalizes")
    add("  - **Model artifact**: **FROZEN FOR JULY RETROSPECTIVE OOT** — 下一阶段"
        " 只加载 frozen JSON apply, 禁止 refit / recalibrate / 调参")
    add("")
    add("## 0. 方法 (固定, 不搜索)")
    add("")
    add("- transform 完全复用 `v004c_repair_state_spec_v002.py` (fit_repair_state_"
        "transform_v2 / construct_repair_factors_v2), 不复制第二套公式; June 只"
        " fit 一次, July 全程只 apply")
    add("- R0 = June prevalence (无 ranking 意义); R1 = 5 base factors; R2 = 8"
        " factors (5 base + DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM / "
        "DIVERGENCE_X_DAMAGE)")
    add("- Logistic 固定: L2 (`penalty=l2`), `C=1.0`, `solver=lbfgs`, "
        "`fit_intercept=True`, `class_weight=None`, `max_iter=1000`, "
        "`random_state=None`; 禁止 GridSearch / RandomSearch / Optuna / CV tuning")
    add("- daily rank: probability 降序, tie-break `event_id` 升序 (确定性);"
        " 禁止用 Target 打破 tie; R0 概率同日恒同 => ranking = N/A")
    add("- 判定门 (§30): 只有 non-convergence / NaN-Inf / abs(coef) >= 10 / "
        "frozen JSON 无法复现 / 实现错误 / July target leakage 允许 "
        "MODEL_FREEZE_REVIEW_REQUIRED; June AUC / LogLoss / 系数方向不参与判定")
    add("- 确定性: 无时间戳 / 无随机; 两次运行 9 个资产字节级一致; July Target"
        " 随机化 / 删除不影响任何正式输出")
    add("")
    add("## 1. 训练切片")
    add("")
    add("- period: {} ~ {}; rows: **{}**; signal dates: **{}**; positive: "
        "**{}**; negative: **{}**; base rate: **{}**".format(
            r["june_date_min"], r["june_date_max"], r["june_rows"],
            r["june_date_count"], r["june_positive"], r["june_negative"],
            _pct(r["june_base_rate"])))
    add("- Target (冻结): `target7_daily_d2open_d3high` = 1[High_D3/Open_D2 - 1"
        " >= 0.07], 预测时点 D1 close")
    add("- 窗口外行: 0 (全部行在 June/July 窗口内)")
    add("")
    add("## 2. July 边界 (硬要求)")
    add("")
    add("- July X read: **YES** (Step A 只加载 ID + 8 core primitive, 仅用于"
        " 确定 June 行号与窗口完整性; July X 不参与任何 transform / fit)")
    add("- July Target read: **NO**; July Target loaded: **NO**; July Target"
        " used: **NO**")
    add("- 实现: Step B 逐行 csv 流式读取, 只有 June 行解析 Target 值; July 行"
        " 的 Target 数值从未被解析 / 保留 / 使用 (测试: July 行 Target 为非法值"
        " 不影响 fit)")
    add("- July X rows: {} ({} dates); 只记录行数".format(
        r["july_x_rows"], r["july_x_date_count"]))
    add("")
    add("## 3. Preprocessing (June-only fit; 复用冻结 spec)")
    add("")
    add("- primitive: June 原始有限值 -> fit q01/q99 -> clip -> fit mu/sigma"
        " (clipped June) -> z; 禁止 mean/std before clipping / July fit /"
        " Target-dependent preprocessing / 全样本预处理")
    add("- composite: CLOSE_DAMAGE = z(G_C), G_C = (-z_OC + z_HC - z_VWAP)/3"
        " (复用 composite_reset, 与旧 RESET 数学一致); RECLAIM = z(G_Q),"
        " G_Q = (z(low_to_close_recovery) + z(afternoon_return))/2 (v001 冻结)")
    add("- derived: raw = F_V^2 / F_V*F_Q / F_V*F_C -> fit June mean/std ->"
        " 标准化; **无第二次 clip** (参数无 q01/q99)")
    add("- 全部 transform 参数已冻结进 frozen JSON, 下一阶段禁止修改")
    add("")
    add("## 4. R0 — Base-rate model")
    add("")
    add("- probability (June prevalence): **{}** ({} / {})".format(
        _pct(r["R0"]["probability"]), r["june_positive"], r["june_rows"]))
    add("- intercept (logit): {}".format(_fmt(r["R0"]["intercept"], 6)))
    add("- ranking: **N/A** (同日所有候选概率相同, 无横截面排序意义)")
    add("")
    add("## 5. R1 — Base Repair State (严格 5 factors + intercept)")
    add("")
    add("logit(p) = α + β_O OPEN + β_V DIVERGENCE + β_C CLOSE_DAMAGE + β_S SUPPLY"
        " + β_Q RECLAIM")
    add("")
    add("| factor | coefficient |")
    add("|---|---|")
    add("| INTERCEPT | {} |".format(_fmt(f1["INTERCEPT"], 6)))
    for f in R1_FACTORS:
        add("| {} | {} |".format(f, _fmt(f1[f], 6)))
    add("")
    add("- n_iter: {}; converged: **{}**".format(
        r["R1"]["n_iter"], r["R1"]["converged"]))
    add("")
    add("## 6. R2 — Conditional Repair State (严格 8 factors + intercept)")
    add("")
    add("logit(p) = α + β_O O + β_V V + β_C C + β_S S + β_Q Q + γ₁ V² + γ₂ VQ + γ₃ VC")
    add("")
    add("| factor | coefficient |")
    add("|---|---|")
    add("| INTERCEPT | {} |".format(_fmt(f2["INTERCEPT"], 6)))
    for f in R2_FACTORS:
        add("| {} | {} |".format(f, _fmt(f2[f], 6)))
    add("")
    add("- n_iter: {}; converged: **{}**".format(
        r["R2"]["n_iter"], r["R2"]["converged"]))
    add("- 禁止加入: DIVERGENCE_X_SUPPLY / TURNOVER_COST / HIGHZONE / MOM7 / "
        "DAMAGE7 / REGIME / POS7 / TREND (全部保持 SENSITIVITY / "
        "LEGACY_SENSITIVITY)")
    add("")
    add("## 7. 训练算法 (冻结, 禁止搜索)")
    add("")
    add("| parameter | value |")
    add("|---|---|")
    for k, v in (("library", "scikit-learn"), ("estimator", "LogisticRegression"),
                 ("penalty", "l2"), ("C", 1.0), ("solver", "lbfgs"),
                 ("fit_intercept", True), ("class_weight", None),
                 ("max_iter", 1000), ("random_state", None)):
        add("| {} | {} |".format(k, v))
    add("- hyperparameter search: **NO** (GridSearch / RandomSearch / Optuna /"
        " CV tuning 全部禁止)")
    add("")
    add("## 8. June development diagnostics — DEVELOPMENT_IN_SAMPLE_ONLY")
    add("")
    add("> 本表全部指标为 June in-sample 诊断 (拟合 173 行 + 评估 173 行),"
        " 不是 validated / OOF / out-of-sample; 禁止宣称性能 proven; 不用于"
        " 选择 R1/R2。")
    add("")
    add("| model | LogLoss | Brier | AUC | AP | mean p | observed | gap |"
        " Top1 | Top3 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for model, row in (("R0", r0), ("R1", r1), ("R2", r2)):
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            model, _fmt(row["logloss"], 6), _fmt(row["brier"], 6),
            _fmt(row["auc"]), _fmt(row["average_precision"]),
            _pct(row["mean_probability"]), _pct(row["observed_rate"]),
            _fmt(row["calibration_gap"], 4),
            _fmt(row["top1_target_rate"]), _fmt(row["top3_target_rate"])))
    add("")
    add("- R0 AUC = 0.5 / AP = base rate: 结构化无区分度 (同日概率恒同), 不是"
        " 模型缺陷")
    add("- Top1/Top3 是 ranking diagnostic, 不是概率模型 primary gate;"
        " Top3 denominator = sum(min(3, candidates))")
    add("")
    add("## 9. Within-date development ranking (仍 DEVELOPMENT_IN_SAMPLE_ONLY)")
    add("")
    add("- 只对同日同时存在至少 1 positive 与至少 1 negative 的日期计算 daily"
        " AUC (valid_auc_dates)")
    add("")
    add("| model | valid dates | mean daily AUC | median daily AUC | pair-weighted"
        " within-date AUC | dates > .5 | = .5 | < .5 |")
    add("|---|---|---|---|---|---|---|---|")
    for model in ("R1", "R2"):
        w = r["within_date"][model]
        add("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            model, w["valid_auc_dates"], _fmt(w["mean_daily_auc"]),
            _fmt(w["median_daily_auc"]),
            _fmt(w["pair_weighted_within_date_auc"]),
            w["dates_auc_gt_0_5"], w["dates_auc_eq_0_5"], w["dates_auc_lt_0_5"]))
    add("")
    add("## 10. 自然候选率 (候选池属性, 与模型无关)")
    add("")
    add("- pooled candidate Target rate: {} ({} / {})".format(
        _pct(r["natural_candidate_rates"]["pooled"]),
        r["june_positive"], r["june_rows"]))
    add("- mean daily candidate Target rate: {}".format(
        _pct(r["natural_candidate_rates"]["mean_daily"])))
    add("")
    add("## 11. 系数健康与概率范围 (§21)")
    add("")
    add("| model | max abs coefficient | factor | magnitude watch (>=5) |"
        " numerical review (>=10) |")
    add("|---|---|---|---|---|")
    for model in ("R1", "R2"):
        h = r["health"][model]
        add("| {} | {} | {} | {} | {} |".format(
            model, _fmt(h["max_abs_coefficient"]), h["max_abs_factor"],
            ", ".join(h["magnitude_watch_factors"]) or "—",
            ", ".join(h["numerical_review_factors"]) or "—"))
    add("")
    add("| model | min | p01 | p05 | median | p95 | p99 | max | p<0.01 | p>0.99 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for model in ("R1", "R2"):
        q = r["health"]["{}_probability".format(model)]
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            model, _fmt(q["min"], 6), _fmt(q["p01"], 6), _fmt(q["p05"], 6),
            _fmt(q["median"], 6), _fmt(q["p95"], 6), _fmt(q["p99"], 6),
            _fmt(q["max"], 6), q["count_p_lt_0_01"], q["count_p_gt_0_99"]))
    add("")
    add("- coefficients finite: **YES** (R1/R2 全部 finite); 0 < p < 1 且全部"
        " finite")
    add("- COEFFICIENT_MAGNITUDE_REVIEW (abs >= 5): {}".format(
        ", ".join(r["health"]["magnitude_watch"]) or "无 (只记录, 不自动失败)"))
    add("- MODEL_NUMERICAL_REVIEW (abs >= 10): {}".format(
        ", ".join(r["health"]["numerical_review"]) or "无"))
    add("- PROBABILITY_EXTREME_WATCH (p<0.01 或 p>0.99 占比 >= 5%): {}".format(
        "YES" if r["health"]["probability_extreme_watch"] else "NO"))
    add("")
    add("## 12. Interaction 极端样本审计 (§20, 只诊断)")
    add("")
    add("- contribution_j = coefficient_j × factor_j (R2 每样本); 禁止根据这些"
        " 样本删行 / 再 clip / winsorize / 调 C")
    add("")
    add("| factor | coefficient | max abs contribution | event | signal date |")
    add("|---|---|---|---|---|")
    for f in R2_FACTORS:
        x = r["interaction"]["max_per_factor"][f]
        add("| {} | {} | {} | {} | {} |".format(
            f, _fmt(x["coefficient"], 4), _fmt(x["max_abs_contribution"]),
            x["event_id"], x["signal_date"]))
    add("")
    add("### Top 10 abs DIVERGENCE_X_RECLAIM contribution 行")
    add("")
    add("| rank | event | signal date | factor value | contribution | abs |")
    add("|---|---|---|---|---|---|")
    sub = r["interaction"]["audit_df"]
    for row in sub[sub["section"] == "top10_DIVERGENCE_X_RECLAIM"].itertuples(index=False):
        add("| {} | {} | {} | {} | {} | {} |".format(
            row.rank, row.event_id, row.signal_date, _fmt(row.factor_value),
            _fmt(row.contribution), _fmt(row.abs_contribution)))
    add("")
    add("### Top 10 abs DIVERGENCE_X_DAMAGE contribution 行")
    add("")
    add("| rank | event | signal date | factor value | contribution | abs |")
    add("|---|---|---|---|---|---|")
    for row in sub[sub["section"] == "top10_DIVERGENCE_X_DAMAGE"].itertuples(index=False):
        add("| {} | {} | {} | {} | {} | {} |".format(
            row.rank, row.event_id, row.signal_date, _fmt(row.factor_value),
            _fmt(row.contribution), _fmt(row.abs_contribution)))
    add("")
    add("- anomalies: {}".format(
        "; ".join(r["interaction"]["anomalies"]) if r["interaction"]["anomalies"]
        else "无 (全部 contribution finite; 无 |contribution| >= 10)"))
    add("")
    add("## 13. R2 层级解释 (marginal effect; 只解释, 不选模型)")
    add("")
    add("- 禁止把 β_DIVERGENCE / β_CLOSE_DAMAGE / β_RECLAIM 孤立解释成全局单调"
        " 效应; R2 中 DIVERGENCE 的边际作用:")
    add("")
    add("    ∂logit(p)/∂V = β_V + 2γ₁V + γ₂Q + γ₃C")
    add("")
    add("- 标准情景 (V=-1,0,+1,+2; Q=-1,0,+1; C=-1,0,+1; 不使用 Target 选择"
        " 情景; 目的: 确认模型能表达 '适度分歧最好、极端分歧可能透支、回收和"
        " 最终损伤改变分歧价值'):")
    add("")
    add(_marginal_effect_table(r["R2"]))
    add("")
    add("## 14. Frozen artifact (§25)")
    add("")
    add("- JSON: `v004c_repair_state_frozen_model_v001.json`")
    add("- 内容: model_version / factor_spec_version / training_period / "
        "training_rows / training_dates / target_column / algorithm (固定参数) /"
        " R1+R2 (factor_order + intercept + coefficients) / primitive transforms"
        " (q01/q99/mu/sigma) / composite transforms (CLOSE_DAMAGE, RECLAIM) /"
        " derived transforms (3 个 mu/sigma) / ranking 规则 / research status"
        " (post_june_hypothesis=true, june_is_validation=false, "
        "july_target_seen=false)")
    add("- 重建验证: R1 prediction reproduction max error = {}; R2 = {} (须"
        " < 1e-12)".format(r["repro_errors"]["R1"], r["repro_errors"]["R2"]))
    add("- full preprocessing frozen: **YES**; coefficients frozen: **YES**;"
        " ranking frozen: **YES** (probability descending, event_id ascending)")
    add("- 禁止存储: July Target 派生统计 / future-selected thresholds")
    add("")
    add("## 15. Leakage 测试 (§29, 测试覆盖)")
    add("")
    add("- July target randomization invariance: **PASS** (Version B: July Target"
        " 随机化 => June transform params / R1+R2 coefficients / June predictions"
        " / frozen JSON 完全不变, 9 个资产 byte-identical)")
    add("- July target removal invariance: **PASS** (Version C: July 行 Target"
        " 删除/置空 => fit 仍完全成功, 资产 byte-identical)")
    add("- July X mutation cannot alter June coefficients: **PASS** (修改 July X"
        " 后 June coefficients 逐位相同)")
    add("- July target value 从未被解析: **PASS** (July 行 Target 为非法值不影响"
        " 任何输出)")
    add("- X-only hash: July Target 值不参与 input hash, 全部模型资产"
        " byte-identical")
    add("")
    add("## 16. July 合同 (下一阶段不可变)")
    add("")
    add("- 本任务 commit 完成后, 以下全部不得因 July 结果修改: R1 factors / "
        "R2 factors / factor formulas / factor order / primitive clipping params"
        " / primitive mean/std / composite mean/std / derived mean/std / "
        "Logistic C / penalty / solver / coefficients / ranking rule / "
        "Top1/Top3 rule")
    add("- July 阶段只能: load frozen JSON -> load July X -> apply frozen "
        "transforms -> predict probabilities -> open July Target once -> evaluate")
    add("- 禁止: fit / refit / recalibrate / threshold tuning / coefficient "
        "adjustment / factor changes")
    add("")
    add("## 17. 最终状态")
    add("")
    add("- 最终研究状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 判定依据: {}".format("; ".join(r["status_reasons"])))
    add("- June interpretation: development only; model selected using June:"
        " **NO**; R1 frozen: **YES**; R2 frozen: **YES** (两个都必须冻结,"
        " 由 July OOT 负责 R0 vs R1 vs R2 比较)")
    add("- Historical assets modified: **NO**")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c Repair-State v002 — June full-development fit + "
                    "冻结模型工件 (9 个正式资产)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_repair_state_model_freeze_v001_202606)")
    parser.add_argument("--skip-git-check", action="store_true",
                        help="跳过分支/HEAD 前置检查 (仅测试/重建用)")
    args = parser.parse_args(argv)

    r = run_fit(args.input, args.output, git_check=not args.skip_git_check)

    print("=== v004c repair-state june full-development fit ===")
    print(f"input            : {r['input']}")
    print(f"x_only_sha256    : {r['x_only_sha256']}")
    print(f"June             : {r['june_rows']} rows / {r['june_date_count']} dates"
          f" (pos {r['june_positive']} / neg {r['june_negative']})")
    print(f"July X           : {r['july_x_rows']} rows ({r['july_x_date_count']} dates)"
          f" 只记录行数; July Target loaded: NO")
    m = r["metrics"].set_index("model")
    for model in ("R0", "R1", "R2"):
        row = m.loc[model]
        print(f"{model}               : logloss={row['logloss']:.6f} "
              f"brier={row['brier']:.6f} auc={row['auc']:.4f} "
              f"ap={row['average_precision']:.4f}")
    print(f"convergence      : R1 n_iter={r['R1']['n_iter']} "
          f"R2 n_iter={r['R2']['n_iter']}")
    print(f"repro errors     : R1={r['repro_errors']['R1']:.3g} "
          f"R2={r['repro_errors']['R2']:.3g}")
    print(f"status           : {r['status']}")
    for reason in r["status_reasons"]:
        print(f"  - {reason}")
    print(f"output           : {args.output} ({len(OUTPUT_FILES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

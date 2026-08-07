"""v004c Repair-State Model v002 — Frozen July Retrospective OOT (July 考试)。

- 前置检查 (git): 必须在 research-sample-analysis 分支且 HEAD 等于
  预期提交 (99daa9fa4be12af91a7271a5b0f863d7e0978724), 否则拒绝继续
- 冻结模型唯一来源: v004c_repair_state_frozen_model_v001.json (身份校验
  失败 => FROZEN_MODEL_CONTRACT_MISMATCH); 禁止从源码重新拟合 / 禁止
  硬编码新系数副本
- July (2026-07-01 ~ 2026-07-29, 160 rows / 21 dates) 是正式 OOT 考试:
  两阶段打开 (Phase A target-blind 预测 + PRE_TARGET_PREDICTION_SHA256;
  Phase B 一次性打开 July Target, POST hash 必须与 PRE 一致), 打开后
  只允许 evaluation / reporting
- 本任务是考试, 不是训练: 禁止 refit / recalibrate / threshold tuning /
  feature selection / 根据 July 结果重新设计模型
- 判定门 (§26-29) 预声明, 只依赖评价指标, 不因 July 结果修改; bootstrap
  (2000 reps, seed=20260808, unit=signal_date) 只做不确定性描述
- 输出 9 个确定性资产到 reports/research/v004c_repair_state_july_oot_v001_202607/
  (无时间戳 / 无随机 / 两次运行字节级一致); 不输出任何新 model JSON
- 禁止: 修改冻结 JSON / 历史资产; git add . / add -A; tag
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_logistic_walkforward import (
    JULY_END,
    JULY_START,
    JUNE_END,
    JUNE_START,
    TARGET_COLUMN,
)
from src.v004c_repair_state_spec_v002 import R2_FACTORS
from src.v004c_repair_state_july_oot import (
    BOOTSTRAP_REPLICATES,
    CONFIDENCE_MIXED,
    CONFIDENCE_STRONG,
    FROZEN_JUNE_POSITIVE,
    FROZEN_JUNE_TOTAL,
    PROMOTE_R1_FORWARD_SHADOW,
    PROMOTE_R2_FORWARD_SHADOW,
    R0_PREVALENCE,
    REJECT_REPAIR_STATE_V002_OOT,
    JulyOOTError,
    build_target_blind_predictions,
    calibration_terciles,
    cluster_bootstrap,
    confidence_label,
    disagreement_audit,
    evaluate_july,
    interaction_audit_july,
    load_frozen_model,
    open_target,
    prediction_hash,
    ranking_concentration,
    read_july_target,
    select_july_frame,
    transform_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (REPO_ROOT
                 / "reports/research/v004c_model_table_v001_20260601_20260729"
                 / "v004c_model_table_v001.csv")
DEFAULT_FROZEN = (REPO_ROOT
                  / "reports/research/v004c_repair_state_model_freeze_v001_202606"
                  / "v004c_repair_state_frozen_model_v001.json")
DEFAULT_JUNE_METRICS = (REPO_ROOT
                        / "reports/research/v004c_repair_state_model_freeze_v001_202606"
                        / "v004c_repair_state_development_metrics_v001.csv")
DEFAULT_OUTPUT = REPO_ROOT / "reports/research/v004c_repair_state_july_oot_v001_202607"

OUTPUT_FILES: tuple[str, ...] = (
    "v004c_repair_state_july_predictions_v001.csv",
    "v004c_repair_state_july_metrics_v001.csv",
    "v004c_repair_state_july_daily_ranking_v001.csv",
    "v004c_repair_state_july_within_date_auc_v001.csv",
    "v004c_repair_state_july_bootstrap_v001.csv",
    "v004c_repair_state_july_calibration_v001.csv",
    "v004c_repair_state_july_transform_audit_v001.csv",
    "v004c_repair_state_july_interaction_audit_v001.csv",
    "v004c_repair_state_july_oot_review_v001.md",
)

# 开始前检查 (§一): 分支 / HEAD 必须精确匹配, 否则停止
EXPECTED_BRANCH = "research-sample-analysis"
EXPECTED_HEAD = "99daa9fa4be12af91a7271a5b0f863d7e0978724"

# 已知 July 事实 (§六, 防御性验证)
KNOWN_JULY_ROWS = 160
KNOWN_JULY_DATES = 21

_CSV_FLOAT = "%.6f"

COMPARISON_STATS: tuple[str, ...] = (
    "r1_ll_improvement_vs_r0",
    "r1_brier_improvement_vs_r0",
    "r2_ll_improvement_vs_r0",
    "r2_brier_improvement_vs_r0",
    "r2_ll_improvement_vs_r1",
    "r2_brier_improvement_vs_r1",
)

GATE_FIELDS: tuple[str, ...] = (
    "R1_PROBABILITY_PASS", "R1_RANKING_PASS", "R1_OOT_PASS",
    "R2_PROBABILITY_PASS", "R2_RANKING_PASS", "R2_OOT_PASS",
    "R2_INCREMENTAL_PASS", "R2_ONLY_OOT_PASS",
)


# ---------------------------------------------------------------------------
# git 前置检查 (§一)
# ---------------------------------------------------------------------------

def check_git_preconditions(run_fn=None) -> None:
    """branch / HEAD 前置检查; 不匹配 => 显式失败 (禁止继续)."""
    def _run(cmd):
        r = (run_fn(cmd) if run_fn is not None
             else subprocess.run(cmd, capture_output=True, text=True, shell=False))
        if getattr(r, "returncode", None) != 0:
            raise JulyOOTError(
                "git 命令失败: {} -> {}".format(cmd, getattr(r, "stderr", "").strip()))
        return str(getattr(r, "stdout", "")).strip()
    branch = _run(["git", "branch", "--show-current"])
    head = _run(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise JulyOOTError(
            "前置检查失败: 预期分支 {}, 实际 {}; 停止".format(EXPECTED_BRANCH, branch))
    if head != EXPECTED_HEAD:
        raise JulyOOTError(
            "前置检查失败: 预期 HEAD {}, 实际 {}; 停止".format(EXPECTED_HEAD, head))


# ---------------------------------------------------------------------------
# R0 交叉验证 (§九: R0 = 冻结 June prevalence, 用 June 开发 metrics 交叉验证)
# ---------------------------------------------------------------------------

def _verify_june_prevalence(june_metrics_csv) -> float:
    """June 开发 metrics 的 R1 pooled_candidate_rate 必须等于 59/173 (全精度).

    R0 是冻结常量, 绝不使用 July prevalence; 该交叉验证只证明常量来源。
    """
    june_metrics_csv = Path(june_metrics_csv)
    if not june_metrics_csv.exists():
        raise JulyOOTError("June 开发 metrics CSV 不存在: {}".format(june_metrics_csv))
    df = pd.read_csv(june_metrics_csv, encoding="utf-8-sig")
    row = df.loc[df["model"] == "R1", "pooled_candidate_rate"]
    if len(row) != 1:
        raise JulyOOTError("June metrics 缺少唯一 R1 行")
    val = float(row.iloc[0])
    if abs(val - R0_PREVALENCE) > 1e-12:
        raise JulyOOTError(
            "June metrics pooled_candidate_rate = {!r} 与冻结 June prevalence "
            "{}/{} 不一致".format(val, FROZEN_JUNE_POSITIVE, FROZEN_JUNE_TOTAL))
    return val


# ---------------------------------------------------------------------------
# 主流程: run_oot (纯逻辑 + 确定性写出; 无随机 / 无时间戳)
# ---------------------------------------------------------------------------

def run_oot(input_csv, frozen_json=None, june_metrics_csv=None,
            output_dir=None, git_check: bool = True) -> dict:
    """July OOT 全流程: 冻结加载 -> 身份校验 -> July 切片 -> Phase A 预测
    -> Phase B 一次性打开 Target -> 评价 -> 9 个资产。

    - Phase A 完成时计算 PRE_TARGET_PREDICTION_SHA256 (target-blind)
    - Phase B 打开 Target 后 POST hash 必须与 PRE 一致, 否则显式失败
    - July 160 rows / 21 dates 防御性验证 (与测试 #7-8 一致)
    """
    input_csv = Path(input_csv)
    frozen_json = Path(frozen_json) if frozen_json else Path(DEFAULT_FROZEN)
    june_metrics_csv = (Path(june_metrics_csv) if june_metrics_csv
                        else Path(DEFAULT_JUNE_METRICS))
    output_dir = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT)
    if git_check:
        check_git_preconditions()

    frozen = load_frozen_model(frozen_json)
    _verify_june_prevalence(june_metrics_csv)

    july_df, stats = select_july_frame(input_csv)
    if len(july_df) != KNOWN_JULY_ROWS \
            or int(july_df["signal_date"].nunique()) != KNOWN_JULY_DATES:
        raise JulyOOTError(
            "July 切片与已知事实不符: rows={} (预期 {}), dates={} (预期 {})".format(
                len(july_df), KNOWN_JULY_ROWS,
                int(july_df["signal_date"].nunique()), KNOWN_JULY_DATES))

    # ---- Phase A (TARGET BLIND): 冻结 transform -> 手工概率 -> rank ----
    blind = build_target_blind_predictions(frozen, july_df)
    pre_hash = blind["pre_target_prediction_sha256"]

    # ---- Phase B (OPEN JULY TARGET ONCE): 逐行流式只解析 July Target ----
    july_row_numbers = set(
        int(v) for v in blind["predictions"]["_data_row"].tolist())
    target_by_row = read_july_target(input_csv, july_row_numbers)
    preds = open_target(blind["predictions"], target_by_row)
    post_hash = prediction_hash(preds)
    if post_hash != pre_hash:
        raise JulyOOTError(
            "两阶段契约被破坏: 打开 Target 后预测 hash 改变 "
            "(PRE={} POST={})".format(pre_hash, post_hash))
    observed = float(preds[TARGET_COLUMN].mean())

    # ---- 评价 (打开 Target 后只读) ----
    eval_res = evaluate_july(preds)
    rep_df, ci_df = cluster_bootstrap(preds)
    decision = eval_res["gates"]["FINAL_DECISION"]
    confidence = confidence_label(decision, ci_df)
    calibration = {
        "R1": calibration_terciles(preds, "r1"),
        "R2": calibration_terciles(preds, "r2"),
    }
    concentration = {
        "R1": ranking_concentration(preds, "r1"),
        "R2": ranking_concentration(preds, "r2"),
    }
    disagreement = disagreement_audit(eval_res["daily_df"])
    transform_df = transform_audit(blind["july_raw"], frozen)
    audit_df, max_per_factor, anomalies = interaction_audit_july(
        blind["factors"], frozen, blind["july_raw"])

    m = eval_res["metrics_df"].set_index("model")
    comparisons = {
        "r1_ll_improvement_vs_r0": float(m.loc["R0", "logloss"]
                                         - m.loc["R1", "logloss"]),
        "r1_brier_improvement_vs_r0": float(m.loc["R0", "brier"]
                                            - m.loc["R1", "brier"]),
        "r2_ll_improvement_vs_r0": float(m.loc["R0", "logloss"]
                                         - m.loc["R2", "logloss"]),
        "r2_brier_improvement_vs_r0": float(m.loc["R0", "brier"]
                                            - m.loc["R2", "brier"]),
        "r2_ll_improvement_vs_r1": float(m.loc["R1", "logloss"]
                                         - m.loc["R2", "logloss"]),
        "r2_brier_improvement_vs_r1": float(m.loc["R1", "brier"]
                                            - m.loc["R2", "brier"]),
    }

    results = {
        "input": str(input_csv),
        "frozen_json": str(frozen_json),
        "june_metrics_csv": str(june_metrics_csv),
        "output_dir": str(output_dir),
        "stats": stats,
        "july_rows": int(len(preds)),
        "july_dates": int(preds["signal_date"].nunique()),
        "observed_rate": observed,
        "two_phase": {
            "PHASE_A_COMPLETE": "YES",
            "PRE_TARGET_PREDICTION_SHA256": pre_hash,
            "TARGET_OPENED": "YES",
            "POST_TARGET_PREDICTION_SHA256": post_hash,
            "IDENTICAL": "YES",
        },
        "predictions": preds,
        "factors": blind["factors"],
        "eval": eval_res,
        "comparisons": comparisons,
        "bootstrap": {"replicates_df": rep_df, "ci_df": ci_df},
        "confidence": confidence,
        "calibration": calibration,
        "concentration": concentration,
        "disagreement": disagreement,
        "transform_audit_df": transform_df,
        "interaction": {
            "audit_df": audit_df,
            "max_per_factor": max_per_factor,
            "anomalies": anomalies,
        },
        "decision": decision,
        "forward_shadow_candidate": (
            "R1" if decision == PROMOTE_R1_FORWARD_SHADOW
            else ("R2" if decision == PROMOTE_R2_FORWARD_SHADOW else "NONE")),
    }
    write_reports(results, output_dir)
    return results


# ---------------------------------------------------------------------------
# 9 个正式资产 (确定性写出; §38)
# ---------------------------------------------------------------------------

def write_reports(results: dict, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(results, output_dir / OUTPUT_FILES[0])
    _write_metrics(results, output_dir / OUTPUT_FILES[1])
    _write_daily_ranking(results, output_dir / OUTPUT_FILES[2])
    _write_within_date_auc(results, output_dir / OUTPUT_FILES[3])
    _write_bootstrap(results, output_dir / OUTPUT_FILES[4])
    _write_calibration(results, output_dir / OUTPUT_FILES[5])
    results["transform_audit_df"].to_csv(output_dir / OUTPUT_FILES[6],
                                         index=False, encoding="utf-8-sig",
                                         float_format=_CSV_FLOAT)
    results["interaction"]["audit_df"].to_csv(output_dir / OUTPUT_FILES[7],
                                              index=False, encoding="utf-8-sig",
                                              float_format=_CSV_FLOAT)
    (output_dir / OUTPUT_FILES[8]).write_text(_review_md(results),
                                              encoding="utf-8")


def _write_predictions(results: dict, path: Path) -> None:
    preds = results["predictions"].copy()
    factors = results["factors"]
    for f in R2_FACTORS:
        preds[f] = factors[f].to_numpy(dtype=float)
    order = (["event_id", "code", "signal_date", TARGET_COLUMN,
              "r0_probability", "r1_probability", "r1_rank",
              "r2_probability", "r2_rank"] + list(R2_FACTORS))
    preds[order].to_csv(path, index=False, encoding="utf-8-sig",
                        float_format=_CSV_FLOAT)


def _metrics_csv(results: dict) -> pd.DataFrame:
    """metrics CSV: section 长表 (model / gate / two_phase / decision /
    comparison 行)。"""
    m = results["eval"]["metrics_df"]
    rows: list[dict] = []
    for _, r in m.iterrows():
        for field in m.columns:
            rows.append({"section": "model", "model": r["model"],
                         "field": field, "value": r[field]})
    gates = results["eval"]["gates"]
    for field in GATE_FIELDS:
        rows.append({"section": "gate", "model": "",
                     "field": field, "value": "YES" if gates[field] else "NO"})
    for field, value in results["two_phase"].items():
        rows.append({"section": "two_phase", "model": "",
                     "field": field, "value": value})
    rows.append({"section": "decision", "model": "",
                 "field": "FINAL_DECISION", "value": results["decision"]})
    rows.append({"section": "decision", "model": "",
                 "field": "CONFIDENCE", "value": results["confidence"]})
    rows.append({"section": "decision", "model": "",
                 "field": "forward_shadow_candidate",
                 "value": results["forward_shadow_candidate"]})
    rows.append({"section": "decision", "model": "",
                 "field": "observed_rate", "value": results["observed_rate"]})
    for field, value in results["comparisons"].items():
        rows.append({"section": "comparison", "model": "",
                     "field": field, "value": value})
    return pd.DataFrame(rows)


def _write_metrics(results: dict, path: Path) -> None:
    _metrics_csv(results).to_csv(path, index=False, encoding="utf-8-sig",
                                 float_format=_CSV_FLOAT)


def _write_daily_ranking(results: dict, path: Path) -> None:
    results["eval"]["daily_df"].to_csv(path, index=False, encoding="utf-8-sig",
                                       float_format=_CSV_FLOAT)


def _write_within_date_auc(results: dict, path: Path) -> None:
    daily = results["eval"]["daily_df"]
    wd = results["eval"]["within_date"]
    rows: list[dict] = []
    for _, r in daily.iterrows():
        for tag in ("r1", "r2"):
            rows.append({
                "section": "date", "signal_date": r["signal_date"],
                "model": tag, "daily_auc": r["{}_daily_auc".format(tag)],
                "auc_valid": r["{}_auc_valid".format(tag)],
                "field": "", "value": ""})
    for tag in ("r1", "r2"):
        for field in ("valid_auc_dates", "mean_daily_auc", "median_daily_auc",
                      "pair_weighted_within_date_auc", "dates_auc_gt_0_5",
                      "dates_auc_eq_0_5", "dates_auc_lt_0_5"):
            rows.append({
                "section": "summary", "signal_date": "",
                "model": "R1" if tag == "r1" else "R2",
                "daily_auc": "", "auc_valid": "",
                "field": field, "value": wd[tag.upper()][field]})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig",
                              float_format=_CSV_FLOAT)


def _write_bootstrap(results: dict, path: Path) -> None:
    """bootstrap CSV: 2000 replicate 行 (10 个统计) + 10 个 CI 行 (2.5/97.5
    percentile, n_finite, replicates)。"""
    rep_df = results["bootstrap"]["replicates_df"]
    ci_df = results["bootstrap"]["ci_df"]
    rows: list[dict] = []
    for i, r in rep_df.iterrows():
        row = {"section": "replicate", "replicate": int(i) + 1,
               "statistic": "", "lower": "", "upper": "",
               "n_finite": "", "replicates": ""}
        for col in COMPARISON_STATS:
            row[col] = r[col]
        for col in ("r1_pair_weighted_within_date_auc",
                    "r2_pair_weighted_within_date_auc",
                    "r1_top3_lift", "r2_top3_lift"):
            row[col] = r[col]
        rows.append(row)
    for _, r in ci_df.iterrows():
        rows.append({
            "section": "ci", "replicate": "",
            "statistic": r["statistic"], "lower": r["lower"],
            "upper": r["upper"], "n_finite": r["n_finite"],
            "replicates": r["replicates"],
            "r1_ll_improvement_vs_r0": "", "r1_brier_improvement_vs_r0": "",
            "r2_ll_improvement_vs_r0": "", "r2_brier_improvement_vs_r0": "",
            "r2_ll_improvement_vs_r1": "", "r2_brier_improvement_vs_r1": "",
            "r1_pair_weighted_within_date_auc": "",
            "r2_pair_weighted_within_date_auc": "",
            "r1_top3_lift": "", "r2_top3_lift": ""})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig",
                              float_format=_CSV_FLOAT)


def _write_calibration(results: dict, path: Path) -> None:
    rows: list[dict] = []
    for tag, label in (("r1", "R1"), ("r2", "R2")):
        for _, r in results["calibration"][label].iterrows():
            rows.append({"section": "tercile", "model": label,
                         "tercile": r["tercile"], "count": r["count"],
                         "mean_predicted": r["mean_predicted"],
                         "observed_rate": r["observed_rate"], "spread": ""})
        p = results["predictions"]["{}_probability".format(tag)]
        rows.append({"section": "all", "model": label, "tercile": "ALL",
                     "count": int(len(p)),
                     "mean_predicted": float(p.mean()),
                     "observed_rate": results["observed_rate"], "spread": ""})
        c = results["concentration"][label]
        rows.append({"section": "concentration", "model": label,
                     "tercile": "TOP20", "count": c["bucket_size"],
                     "mean_predicted": "",
                     "observed_rate": c["top20_target_rate"], "spread": ""})
        rows.append({"section": "concentration", "model": label,
                     "tercile": "BOTTOM20", "count": c["bucket_size"],
                     "mean_predicted": "",
                     "observed_rate": c["bottom20_target_rate"], "spread": ""})
        rows.append({"section": "concentration", "model": label,
                     "tercile": "SPREAD", "count": "",
                     "mean_predicted": "",
                     "observed_rate": "", "spread": c["spread"]})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig",
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
    return f"{fv * 100:.2f}%"


def _yes_no(v: bool) -> str:
    return "YES" if v else "NO"


def _conc_rows(model: str, conc: dict) -> str:
    """§34 concentration markdown 行 (TOP20 / BOTTOM20 / SPREAD)."""
    return "| {} | TOP20 | {} | {} | |\n| {} | BOTTOM20 | {} | {} | |\n" \
           "| {} | SPREAD | | | {} |".format(
               model, conc["bucket_size"], _fmt(conc["top20_target_rate"], 4),
               model, conc["bucket_size"], _fmt(conc["bottom20_target_rate"], 4),
               model, _fmt(conc["spread"], 4))


def _review_md(r: dict) -> str:
    g = r["eval"]["gates"]
    m = r["eval"]["metrics_df"].set_index("model")
    wd = r["eval"]["within_date"]
    topk = r["eval"]["topk"]
    b = r["eval"]["baselines"]
    ci = r["bootstrap"]["ci_df"].set_index("statistic")
    daily = r["eval"]["daily_df"]
    two = r["two_phase"]
    s = r["stats"]
    d = r["disagreement"]

    def model_row(model):
        row = m.loc[model]
        return "| {} | {} | {} | {} | {} | {} | {} | {}".format(
            model, _fmt(row["logloss"], 6), _fmt(row["brier"], 6),
            _fmt(row["auc"], 4), _fmt(row["average_precision"], 4),
            _fmt(row["mean_probability"], 6), _fmt(row["observed_rate"], 6),
            _fmt(row["calibration_gap"], 6))

    def wd_row(tag):
        label = "R1" if tag == "r1" else "R2"
        return "| {} | {} | {} | {} | {} |".format(
            label, wd[label]["valid_auc_dates"],
            _fmt(wd[label]["mean_daily_auc"], 4),
            _fmt(wd[label]["median_daily_auc"], 4),
            _fmt(wd[label]["pair_weighted_within_date_auc"], 4))

    def topk_row(tag):
        label = "R1" if tag == "r1" else "R2"
        k = topk[label]
        return "| {} | {} | {} | {} | {} | {} | {}".format(
            label, k["top1_target_count"], _fmt(k["top1_target_rate"], 4),
            _fmt(k["top1_baseline"], 4), _fmt(k["top1_lift"], 4),
            _fmt(k["top3_target_rate"], 4), _fmt(k["top3_lift"], 4))

    def ci_row(field):
        row = ci.loc[field]
        return "| {} | {} | {} | {} |".format(
            field, _fmt(row["lower"], 6), _fmt(row["upper"], 6),
            int(row["n_finite"]))

    return """# v004c Repair-State Model v002 — Frozen July Retrospective OOT Review

OOT 考试: 冻结 June 模型 (Repair-State v002) 在 July (2026-07-01 ~ 2026-07-29)
上的回顾式正式考试。本报告只评价冻结模型, 不训练 / 不重拟合 / 不调参 /
不根据 July 结果修改模型。

## 1. Method (方法)

- 冻结模型唯一来源: `{frozen_json}` (frozen model JSON, 全精度系数)
- transform / factor / 概率全部复用冻结 JSON 与 spec v002 的 apply 逻辑
  (src/v004c_repair_state_model_v002.py + src/v004c_repair_state_spec_v002.py);
  不复制第三套公式; 不调用任何模型拟合
- 概率重建: logit = X @ beta + intercept; p = 1 / (1 + exp(-logit))
  (numpy/scipy sigmoid; 无 refit / recalibrate)
- R0 = 冻结 June prevalence = {pos}/{total} = {r0}
- 判定门 (§26-29) 在打开 July Target 前预声明; bootstrap (replicates=2000,
  seed=20260808, unit=signal_date) 只做不确定性描述, 不改变 pass/fail
- 所有输出确定性: 无时间戳 / 固定 seed / CSV float_format %.6f

## 2. Frozen model contract (冻结模型身份)

| field | value |
|---|---|
| model_version | {mv} |
| factor_spec_version | {fsv} |
| training_period | {june_start} ~ {june_end} |
| training_rows | {june_total} |
| training_dates | 21 |
| post_june_hypothesis | true |
| june_is_validation | false |
| july_target_seen | false |
| R1 factor_order | OPEN, DIVERGENCE, CLOSE_DAMAGE, SUPPLY, RECLAIM (5) |
| R2 factor_order | R1 + DIVERGENCE_SQ, DIVERGENCE_X_RECLAIM, DIVERGENCE_X_DAMAGE (8) |
| ranking | probability descending, tie -> event_id ascending |

身份校验通过 (无 FROZEN_MODEL_CONTRACT_MISMATCH)。

## 3. July slice (July 切片)

| field | value |
|---|---|
| window_rows | {win_rows} |
| out_of_window_rows | {out_rows} |
| july rows | {july_rows} |
| july dates | {july_dates} |
| july date range | {july_start} ~ {july_end} |

防御性验证通过: rows = {july_rows} (预期 160), dates = {july_dates} (预期 21)。

## 4. Two-phase audit (两阶段打开审计)

| field | value |
|---|---|
| PHASE_A_COMPLETE | {a_phase} |
| PRE_TARGET_PREDICTION_SHA256 | {pre} |
| TARGET_OPENED | {opened} |
| POST_TARGET_PREDICTION_SHA256 | {post} |
| IDENTICAL | {same} |

Phase A (target-blind) 完成后才允许打开 July Target; 打开后 POST hash 与
PRE hash 完全一致, 证明打开 Target 没有改变任何模型预测 / 排名。July Target
一旦打开即 permanently consumed, 不得再称 holdout / unseen / blind。

## 5. R0 frozen probability (R0 冻结概率)

R0 = 冻结 June prevalence = {pos}/{total} = {r0} (全行常数)。来源为冻结
metadata (FROZEN_JUNE_POSITIVE=59 / FROZEN_JUNE_TOTAL=173), 并由 June
development metrics CSV 的 R1 pooled_candidate_rate 交叉验证
(_verify_june_prevalence, 1e-12 内一致)。R0 不使用 July prevalence;
R0 ranking = N/A (无 ranking 规则)。

## 6. Probability metrics (LogLoss / Brier / AUC / AP)

| model | logloss | brier | auc | average_precision | mean_probability | observed_rate | calibration_gap |
|---|---|---|---|---|---|---|---|
{r0_row}
{r1_row}
{r2_row}

- 自然候选池 pooled Target7 rate = {pooled} (mean daily = {mean_daily})
- improvement = baseline - candidate (positive = better):
  R1 vs R0: LogLoss {r1_ll0}, Brier {r1_br0}; R2 vs R0: LogLoss {r2_ll0},
  Brier {r2_br0}; R2 vs R1: LogLoss {r2_ll1}, Brier {r2_br1}

## 7. Within-date ranking (同日 ranking 主指标)

pair-weighted within-date AUC (Mann-Whitney U 的 within-date 版本; 只计同日
(positive, negative) 对, 跨日对不计):

| model | valid_auc_dates | mean_daily_auc | median_daily_auc | pair_weighted_auc |
|---|---|---|---|---|
{r1_wd_row}
{r2_wd_row}

## 8. Natural candidate baselines (自然候选基线)

- Top1Baseline = mean daily candidate rate = {top1_base}
- Top3Baseline = sum(k_t * CandidateRate_t) / sum(k_t) = {top3_base}
  (k_t = min(3, candidates))
- Top1 / Top3 不是硬 gate (21 个 picks 方差大, 只作 secondary)

## 9. R1-R2 ranking (Top1 / Top3 vs matched baseline)

| model | top1 hits | top1 rate | top1 baseline | top1 lift | top3 rate | top3 lift |
|---|---|---|---|---|---|---|
{r1_topk_row}
{r2_topk_row}

Top3 picks: R1 = {r1_picks}, R2 = {r2_picks}; Top3 命中日期: R1 = {r1_dates},
R2 = {r2_dates}; zero-hit 日期: R1 = {r1_zero}, R2 = {r2_zero}。

## 10. OOT gates (预声明判定门, §26-29)

| gate | value |
|---|---|
| R1_PROBABILITY_PASS | {r1_prob} |
| R1_RANKING_PASS | {r1_rank} |
| R1_OOT_PASS | {r1_oot} |
| R2_PROBABILITY_PASS | {r2_prob} |
| R2_RANKING_PASS | {r2_rank} |
| R2_OOT_PASS | {r2_oot} |
| R2_INCREMENTAL_PASS | {r2_inc} |
| R2_ONLY_OOT_PASS | {r2_only} |
| FINAL_DECISION | {decision} |

- R1_PROBABILITY_PASS: R1 LogLoss < R0 LogLoss AND R1 Brier < R0 Brier
- R1_RANKING_PASS: R1 pair-weighted within-date AUC > 0.50 AND R1 Top3 lift > 0
- R2_INCREMENTAL_PASS (§28): R2 LogLoss < R1 LogLoss AND R2 Brier < R1 Brier
  AND R2 pair-weighted AUC >= R1 AND R2 Top3 rate >= R1
- 最终晋级 (§29): Case A 双 NO -> REJECT; Case B R1 YES R2 NO -> PROMOTE R1;
  Case C 双 YES + incremental NO -> PROMOTE R1 (简约); Case D 双 YES +
  incremental YES -> PROMOTE R2; Case E R1 NO R2 YES -> PROMOTE R2 (R2_ONLY=YES)
- Top1 不参与 hard gate (只作 secondary)

## 11. Bootstrap uncertainty (信号日 cluster bootstrap, 95% percentile CI)

unit = signal_date (有放回抽 21 日期, 重复日期允许重复), replicates=2000,
seed=20260808。CI 只描述证据精度, 不改变 §26-29 的 pass/fail。

| statistic | lower | upper | n_finite |
|---|---|---|---|
{ci_rows}

## 12. Confidence (§32)

| field | value |
|---|---|
| Confidence | {confidence} |

CONFIDENCE_STRONG 要求被晋级模型的 LogLoss improvement CI 下界 > 0 AND
Brier improvement CI 下界 > 0 AND Top3 lift CI 下界 > 0; 否则
CONFIDENCE_MIXED (MIXED 不改变晋级结果, 只提醒 July 只有 21 个日期,
后续 August+ forward shadow 更重要)。

## 13. Calibration (校准描述, 禁止 recalibration)

mean p / observed / gap 见 §6; 概率 tercile (按概率三等分):

| model | tercile | count | mean_predicted | observed_rate |
|---|---|---|---|---|
{r1_cal_rows}
{r2_cal_rows}

只描述, 不做 Platt / isotonic / 任何校准。

## 14. Concentration (ranking 集中度, §34)

| model | bucket | count | target7_rate | spread |
|---|---|---|---|---|
{r1_conc_rows}
{r2_conc_rows}

Top20% / Bottom20% 按 probability 降序 (tie -> event_id 升序) 分桶;
只描述, 不是硬 gate。

## 15. R1-R2 disagreement (分歧审计, §35)

| field | value |
|---|---|
| same_top1_dates | {same_dates} |
| different_top1_dates | {diff} |
| mean_top3_overlap | {overlap} |

逐日详情见 daily ranking CSV (same_top1 / top3_overlap 列)。

## 16. Transform audit (冻结 transform 审计, §37)

| primitive | below_q01 | above_q99 | total_clipped | clip_rate |
|---|---|---|---|---|
{transform_rows}

frozen q01/q99 未被修改; 只描述越界, 禁止删除越界行 / 重新 clip。

## 17. Interaction audit (R2 contribution 审计, §36)

- max |contribution| = coefficient * factor (July 全样本):
{interaction_rows}
- 非有限 contribution / |contribution| >= 10 记录: {anomaly_count} 条
  ({anomalies})
- 只记录; 禁止 clip / drop row / 调整 coefficient

## 18. Q1 — R1 是否真正超过冻结 June base-rate R0?

根据 §6 (LogLoss / Brier) 与 §7/§9 (within-date ranking / Top3 lift) 的
预声明 gate 判定: R1_OOT_PASS = {r1_oot}。
结论必须依赖 pre-registered gate, 而非任何 post-hoc 解释。

## 19. Q2 — R2 的三个条件项是否在 July 提供 R1 之外的真实增量价值?

R2_INCREMENTAL_PASS = {r2_inc} (§28 四条件全部成立才为 YES)。逐项:
- R2 LogLoss < R1 LogLoss: {q2_ll}
- R2 Brier < R1 Brier: {q2_br}
- R2 pair-weighted AUC >= R1: {q2_auc}
- R2 Top3 rate >= R1: {q2_top3}
任何一项失败都意味着条件项 (DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM /
DIVERGENCE_X_DAMAGE) 在 July 上的增量价值未获得证据支持 (简约原则)。

## 20. Q3 — 模型是否把 July 候选池天然 Target7 成功率浓缩成更高 Top1/Top3 强修复成功率?

- 自然候选池 pooled rate = {pooled}; Top3 rate: R1 = {r1_t3rate},
  R2 = {r2_t3rate}; Top3 lift: R1 = {r1_t3lift}, R2 = {r2_t3lift}
- pair-weighted within-date AUC: R1 = {r1_pwauc}, R2 = {r2_pwauc}
- Top1 (secondary): R1 = {r1_t1rate} vs baseline {top1_base};
  R2 = {r2_t1rate}
浓缩效果以 Top3 lift > 0 与 within-date AUC > 0.50 为证据; 21 个日期下
Top1 的方差过大, 不作判定依据。

## 21. Important interpretation (重要解释)

- July 是 OOT 考试: 冻结模型未被 refit / recalibrate / 调参; 冻结 JSON
  未被修改; July 结果未回写训练阶段
- July Target 已打开, permanently consumed: 不得再称 holdout / unseen /
  blind; 下一阶段只能 August+ forward shadow (或 REJECT 后先做失败归因:
  calibration failure / within-date ranking failure / temporal instability /
  factor-state failure / nonlinearity failure)
- 本次判定: {decision}; forward_shadow_candidate = {fsc};
  Confidence = {confidence}
- 历史冻结资产 (v001/v002 freeze, walkforward, model table, src spec/model)
  全部未修改 (July 只能新增, 不回写)

## 22. Next step (下一阶段)

无论 PASS / FAIL / MIXED, 本任务到此停止。若晋级, 下一阶段只允许
August+ forward shadow (冻结模型只读, 输出预测, 不修改模型); 若 REJECT,
先做失败归因再决定是否修订模型假设 (修订需要新的 pre-registered 流程)。
""".format(
        frozen_json=r["frozen_json"],
        mv="v004c_repair_state_model_v001", fsv="v004c_repair_state_spec_v002",
        june_start=JUNE_START, june_end=JUNE_END, june_total=FROZEN_JUNE_TOTAL,
        pos=FROZEN_JUNE_POSITIVE, total=FROZEN_JUNE_TOTAL,
        r0=_fmt(R0_PREVALENCE, 6),
        win_rows=s["window_rows"], out_rows=s["out_of_window_rows"],
        july_rows=r["july_rows"], july_dates=r["july_dates"],
        july_start=JULY_START, july_end=JULY_END,
        a_phase=two["PHASE_A_COMPLETE"], pre=two["PRE_TARGET_PREDICTION_SHA256"],
        opened=two["TARGET_OPENED"], post=two["POST_TARGET_PREDICTION_SHA256"],
        same=two["IDENTICAL"],
        r0_row=model_row("R0"), r1_row=model_row("R1"), r2_row=model_row("R2"),
        pooled=_fmt(b["pooled_candidate_rate"], 4),
        mean_daily=_fmt(b["mean_daily_candidate_rate"], 4),
        r1_ll0=_fmt(r["comparisons"]["r1_ll_improvement_vs_r0"], 6),
        r1_br0=_fmt(r["comparisons"]["r1_brier_improvement_vs_r0"], 6),
        r2_ll0=_fmt(r["comparisons"]["r2_ll_improvement_vs_r0"], 6),
        r2_br0=_fmt(r["comparisons"]["r2_brier_improvement_vs_r0"], 6),
        r2_ll1=_fmt(r["comparisons"]["r2_ll_improvement_vs_r1"], 6),
        r2_br1=_fmt(r["comparisons"]["r2_brier_improvement_vs_r1"], 6),
        r1_wd_row=wd_row("r1"), r2_wd_row=wd_row("r2"),
        top1_base=_fmt(b["top1_baseline"], 4),
        top3_base=_fmt(b["top3_baseline"], 4),
        r1_topk_row=topk_row("r1"), r2_topk_row=topk_row("r2"),
        r1_picks=topk["R1"]["top3_picks"], r2_picks=topk["R2"]["top3_picks"],
        r1_dates=topk["R1"]["dates_with_top3_hit"],
        r2_dates=topk["R2"]["dates_with_top3_hit"],
        r1_zero=topk["R1"]["zero_hit_dates"],
        r2_zero=topk["R2"]["zero_hit_dates"],
        r1_prob=_yes_no(g["R1_PROBABILITY_PASS"]),
        r1_rank=_yes_no(g["R1_RANKING_PASS"]),
        r1_oot=_yes_no(g["R1_OOT_PASS"]),
        r2_prob=_yes_no(g["R2_PROBABILITY_PASS"]),
        r2_rank=_yes_no(g["R2_RANKING_PASS"]),
        r2_oot=_yes_no(g["R2_OOT_PASS"]),
        r2_inc=_yes_no(g["R2_INCREMENTAL_PASS"]),
        r2_only=_yes_no(g["R2_ONLY_OOT_PASS"]),
        decision=r["decision"],
        ci_rows="\n".join(ci_row(f) for f in (
            "r1_ll_improvement_vs_r0", "r1_brier_improvement_vs_r0",
            "r2_ll_improvement_vs_r0", "r2_brier_improvement_vs_r0",
            "r2_ll_improvement_vs_r1", "r2_brier_improvement_vs_r1",
            "r1_pair_weighted_within_date_auc",
            "r2_pair_weighted_within_date_auc",
            "r1_top3_lift", "r2_top3_lift")),
        confidence=r["confidence"],
        r1_cal_rows="\n".join(
            "| R1 | {} | {} | {} | {} |".format(
                row["tercile"], row["count"], _fmt(row["mean_predicted"], 4),
                _fmt(row["observed_rate"], 4))
            for _, row in r["calibration"]["R1"].iterrows()),
        r2_cal_rows="\n".join(
            "| R2 | {} | {} | {} | {} |".format(
                row["tercile"], row["count"], _fmt(row["mean_predicted"], 4),
                _fmt(row["observed_rate"], 4))
            for _, row in r["calibration"]["R2"].iterrows()),
        r1_conc_rows=_conc_rows("R1", r["concentration"]["R1"]),
        r2_conc_rows=_conc_rows("R2", r["concentration"]["R2"]),
        same_dates=d["same_top1_dates"], diff=d["different_top1_dates"],
        overlap=_fmt(d["mean_top3_overlap"], 4),
        transform_rows="\n".join(
            "| {} | {} | {} | {} | {} |".format(
                row["primitive"], row["below_count"], row["above_count"],
                row["total_clipped"], _fmt(row["clip_rate"], 4))
            for _, row in r["transform_audit_df"].iterrows()),
        interaction_rows="\n".join(
            "| {} | coef={} | max_abs_contrib={} | event_id={} | date={} |".format(
                f, _fmt(info["coefficient"], 6),
                _fmt(info["max_abs_contribution"], 4),
                info["event_id"], info["signal_date"])
            for f, info in r["interaction"]["max_per_factor"].items()),
        anomaly_count=len(r["interaction"]["anomalies"]),
        anomalies=("; ".join(r["interaction"]["anomalies"])
                   if r["interaction"]["anomalies"] else "无"),
        q2_ll=_yes_no(bool(m.loc["R2", "logloss"] < m.loc["R1", "logloss"])),
        q2_br=_yes_no(bool(m.loc["R2", "brier"] < m.loc["R1", "brier"])),
        q2_auc=_yes_no(bool(wd["R2"]["pair_weighted_within_date_auc"]
                            >= wd["R1"]["pair_weighted_within_date_auc"])),
        q2_top3=_yes_no(bool(topk["R2"]["top3_target_rate"]
                             >= topk["R1"]["top3_target_rate"])),
        r1_t3rate=_fmt(topk["R1"]["top3_target_rate"], 4),
        r2_t3rate=_fmt(topk["R2"]["top3_target_rate"], 4),
        r1_t3lift=_fmt(topk["R1"]["top3_lift"], 4),
        r2_t3lift=_fmt(topk["R2"]["top3_lift"], 4),
        r1_pwauc=_fmt(wd["R1"]["pair_weighted_within_date_auc"], 4),
        r2_pwauc=_fmt(wd["R2"]["pair_weighted_within_date_auc"], 4),
        r1_t1rate=_fmt(topk["R1"]["top1_target_rate"], 4),
        r2_t1rate=_fmt(topk["R2"]["top1_target_rate"], 4),
        fsc=r["forward_shadow_candidate"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c Repair-State v002 — Frozen July Retrospective OOT "
                    "(9 个正式资产)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--frozen", default=str(DEFAULT_FROZEN),
                        help="冻结模型 JSON 路径 (默认正式 frozen JSON)")
    parser.add_argument("--june-metrics", default=str(DEFAULT_JUNE_METRICS),
                        help="June development metrics CSV (R0 交叉验证)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_repair_state_july_oot_v001_202607)")
    parser.add_argument("--skip-git-check", action="store_true",
                        help="跳过分支/HEAD 前置检查 (仅测试/重建用)")
    args = parser.parse_args(argv)

    r = run_oot(args.input, frozen_json=args.frozen,
                june_metrics_csv=args.june_metrics, output_dir=args.output,
                git_check=not args.skip_git_check)

    print("=== v004c repair-state july retrospective OOT ===")
    print(f"input            : {r['input']}")
    print(f"frozen           : {r['frozen_json']}")
    print(f"July             : {r['july_rows']} rows / {r['july_dates']} dates")
    two = r["two_phase"]
    print(f"two-phase        : PRE={two['PRE_TARGET_PREDICTION_SHA256'][:12]}… "
          f"POST={two['POST_TARGET_PREDICTION_SHA256'][:12]}… "
          f"IDENTICAL={two['IDENTICAL']}")
    m = r["eval"]["metrics_df"].set_index("model")
    for model in ("R0", "R1", "R2"):
        row = m.loc[model]
        print(f"{model}               : logloss={row['logloss']:.6f} "
              f"brier={row['brier']:.6f} auc={row['auc']:.4f} "
              f"ap={row['average_precision']:.4f}")
    g = r["eval"]["gates"]
    print(f"gates            : R1_OOT={g['R1_OOT_PASS']} "
          f"R2_OOT={g['R2_OOT_PASS']} R2_INC={g['R2_INCREMENTAL_PASS']}")
    print(f"decision         : {r['decision']}")
    print(f"confidence       : {r['confidence']}")
    print(f"forward shadow   : {r['forward_shadow_candidate']}")
    print(f"output           : {args.output} ({len(OUTPUT_FILES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""v004c Repair-State Model v002 — June full-development fit + 冻结模型工件。

本模块把已冻结的 Repair-State v002 因子规格 (src/v004c_repair_state_spec_v002.py,
人工批准 FREEZE_REPAIR_STATE_V002) 在完整 June 开发集上执行正式 fit:

- factor structure 已冻结 (R1 严格 5 / R2 严格 8); 本模块禁止修改 factor /
  interaction / 预处理 / Logistic 超参数
- June (2026-06-01 ~ 2026-06-30, 173 rows / 21 dates) 是唯一 training sample
  (post-June hypothesis: June 不是验证集, 本阶段只生成
  DEVELOPMENT_IN_SAMPLE_ONLY 诊断, 不做模型选择)
- July 严格边界 (硬要求): Step A 只加载 X (ID + 8 core primitive) 确定 June
  行; Step B 用逐行 csv 流式只解析 June 行的 Target 值 —— July 行的 Target
  值从未被解析 / 保留 / 使用 (July target read = NO / loaded = NO / used = NO)
- 全部 transform 参数 (primitive q01/q99/mu/sigma, composite mu/sigma,
  derived mu/sigma) 只来自 June, 一次拟合后冻结为 frozen JSON
  (v004c_repair_state_frozen_model_v001.json), 供下一阶段 July retrospective
  OOT 只 apply 不 refit
- R0 = June prevalence (无 ranking 意义); R1 = 5 base factors + intercept;
  R2 = 8 factors + intercept; Logistic 固定 L2 C=1.0 lbfgs
  (penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True, class_weight=None,
  max_iter=1000, random_state=None; 禁止搜索)
- transform 完全复用 spec: fit_repair_state_transform_v2 (FOLD_CLIP_Z) +
  construct_repair_factors_v2, 不复制第二套公式

明确禁止 (本模块不实现): 超参搜索 (GridSearch / RandomSearch / Optuna / CV
tuning), feature selection, interaction 搜索, 根据 June 指标修改模型,
读取 July Target, 修改 spec 模块。
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_factor_spec import FORBIDDEN_TOKENS, X_ID_COLUMNS
from src.v004c_logistic_walkforward import (
    D2OPEN_D3HIGH_THRESHOLD,
    JULY_END,
    JULY_START,
    JUNE_END,
    JUNE_START,
    LOGISTIC_KWARGS,
    PROB_EPS,
    TARGET_COLUMN,
    average_precision,
    brier_score,
    calibration_gap,
    daily_rank,
    fit_logistic,
    logloss,
    predict_logistic,
    roc_auc,
)
from src.v004c_repair_state_spec_v002 import (
    CLOSE_DAMAGE_COMPOSITE_KEY,
    DERIVED_FACTORS,
    R1_FACTORS,
    R2_FACTORS,
    RECLAIM_COMPOSITE_KEY,
    REPAIR_PRIMITIVES,
    construct_repair_factors_v2,
    fit_repair_state_transform_v2,
)

# ---------------------------------------------------------------------------
# 冻结身份 / 状态常量
# ---------------------------------------------------------------------------

MODEL_VERSION = "v004c_repair_state_model_v001"
FACTOR_SPEC_VERSION = "v004c_repair_state_spec_v002"

# 模型只使用 8 个 core primitive (sensitivity primitive 不进模型, 不加载)
X_PRIMITIVE_COLUMNS = REPAIR_PRIMITIVES

FROZEN_REPAIR_STATE_MODEL_V001 = "FROZEN_REPAIR_STATE_MODEL_V001"
MODEL_FREEZE_REVIEW_REQUIRED = "MODEL_FREEZE_REVIEW_REQUIRED"
DEVELOPMENT_IN_SAMPLE_ONLY = "DEVELOPMENT_IN_SAMPLE_ONLY"

# 数值健康阈值 (§21, 只记录 / 判定, 禁止自动调 C)
COEFFICIENT_MAGNITUDE_WATCH = 5.0    # abs(coef) >= 5 => COEFFICIENT_MAGNITUDE_REVIEW
COEFFICIENT_NUMERICAL_REVIEW = 10.0  # abs(coef) >= 10 => MODEL_NUMERICAL_REVIEW (冻结阻塞)
PROB_EXTREME_SHARE = 0.05            # p<0.01 或 p>0.99 占比 >= 5% => PROBABILITY_EXTREME_WATCH
REPRO_MAX_ERROR = 1e-12              # frozen JSON 重建预测允许的最大误差

# 判定门 (§30): 只有实现错误 / non-convergence / NaN-Inf / abs(coef) >= 10 /
# frozen JSON 无法复现 / July target leakage 允许 MODEL_FREEZE_REVIEW_REQUIRED;
# June 指标 (AUC / LogLoss / 系数方向) 一律不得修改模型。


class RepairStateModelError(Exception):
    """v004c Repair-State 模型冻结失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# July 严格边界: Step A (X 全窗口, 无 Target) + Step B (逐行流式, 只解析 June Target)
# ---------------------------------------------------------------------------

def _parse_target_value(raw) -> int:
    """单个 Target 值解析: True/False/1/0 -> int; 其他值显式失败."""
    lowered = str(raw).strip().lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    if lowered not in mapping:
        raise RepairStateModelError(
            "June 行 Target {} 出现无法解析的值: {!r}".format(TARGET_COLUMN, lowered))
    return mapping[lowered]


def load_x_window(input_csv) -> pd.DataFrame:
    """Step A: 显式 usecols 只加载 ID + 8 core primitive (全窗口 June+July X).

    - 请求列含禁止 token (target/d2_/d3_/future/...) => 直接失败
    - 请求列缺失 => 显式失败 (不得静默缺列)
    - Target 列绝不在请求列中
    """
    cols = list(X_ID_COLUMNS) + list(X_PRIMITIVE_COLUMNS)
    for c in cols:
        lowered = c.lower()
        for token in FORBIDDEN_TOKENS:
            if token in lowered:
                raise RepairStateModelError(
                    "请求列 {} 包含禁止 token '{}'".format(c, token))
    header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = [c for c in cols if c not in header]
    if missing:
        raise RepairStateModelError("model table 缺少请求列: {}".format(missing))
    df = pd.read_csv(input_csv, usecols=cols,
                     dtype={"code": str, "signal_date": str})
    df = df[cols]  # 统一为请求列序 (usecols 返回 CSV 原始列序)
    return df


def read_june_target(input_csv, june_row_numbers) -> dict:
    """Step B: 逐行 csv 流式读取, 只有 June 行解析 Target 值.

    July 行的 Target 值从未被解析 / 保留 / 使用 (直接跳过, 不进入任何
    数据结构); July 行 Target 为空 / 非法字符串均不影响读取结果。

    行号对齐: 正式 model table 的原始行结束符为 \\r\\r\\n, 用 csv.reader
    读取时每行数据后会产生一个空行。为与 Step A (pandas read_csv, 跳过
    空行) 的行号对齐, 这里同样跳过空行, 用"非空数据行计数"作为行号
    (数据行号 = pandas 行号 + 1)。解析行数必须与 Step A 的 June 行数一致,
    否则显式失败 (行号对齐错误不得静默)。
    """
    values: dict[int, int] = {}
    with open(input_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        tgt_idx = header.index(TARGET_COLUMN)
        data_row = 0
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue  # 空行 (\\r\\r\\n 残留): 跳过, 与 pandas 行号对齐
            data_row += 1
            if data_row not in june_row_numbers:
                continue  # July / 非 June 行: Target 值不解析, 直接跳过
            values[data_row] = _parse_target_value(row[tgt_idx])
    if len(values) != len(june_row_numbers):
        raise RepairStateModelError(
            "Step B 解析 June Target 行数 {} 与 Step A June 行数 {} 不一致 "
            "(行号对齐错误)".format(len(values), len(june_row_numbers)))
    return values


def load_june_frame(input_csv) -> tuple[pd.DataFrame, dict]:
    """Step A + Step B: June X + June Target (July Target 从未解析).

    返回 (june_df, window_stats); window_stats 只含 X 侧统计
    (july_x_rows / july_x_date_count / window_rows / out_of_window_rows),
    July X 只用于行数统计, 不参与任何 transform / fit。
    """
    input_csv = Path(input_csv)
    window = load_x_window(input_csv)
    dates = window["signal_date"]
    june_mask = (dates >= JUNE_START) & (dates <= JUNE_END)
    july_mask = (dates >= JULY_START) & (dates <= JULY_END)
    if not bool((june_mask | july_mask).all()):
        raise RepairStateModelError(
            "model table 存在 June/July 窗口外行 (signal_date 窗口 "
            "{}~{}); 拒绝继续".format(JUNE_START, JULY_END))
    if int(june_mask.sum()) == 0:
        raise RepairStateModelError("model table 中没有 June 行")
    # 数据行号 = pandas 行号 + 1 (第 0 行是 header; read_june_target 与
    # pandas 一致地跳过空行, 见该函数 docstring)
    june_row_numbers = set(int(i) + 1 for i in np.flatnonzero(june_mask.to_numpy()))
    target_by_row = read_june_target(input_csv, june_row_numbers)
    june = window.loc[june_mask].copy()
    june["target"] = [int(target_by_row[int(i) + 1]) for i in june.index]
    june = june.sort_values(["signal_date", "event_id"]).reset_index(drop=True)
    window_stats = {
        "window_rows": int(len(window)),
        "june_rows": int(june_mask.sum()),
        "july_x_rows": int(july_mask.sum()),
        "july_x_date_count": int(july_mask.sum()
                                 and window.loc[july_mask, "signal_date"].nunique()),
        "out_of_window_rows": int((~(june_mask | july_mask)).sum()),
    }
    return june, window_stats


def x_only_sha256(input_csv) -> str:
    """只对 X 列 (ID + 8 core primitive) 内容计算 SHA256.

    July Target 值不参与 hash => 修改 / 随机化 / 删除 July Target 后
    全部正式资产 byte-identical (§29)。
    """
    df = load_x_window(input_csv)
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# metrics / ranking (复用 walkforward 同一数学; 不复制第二套公式)
# ---------------------------------------------------------------------------

def mean_daily_candidate_rate(preds: pd.DataFrame) -> float:
    """每日 candidate Target rate 的均值 (全部日期, 与 daily AUC 无关)."""
    rates = preds.groupby("signal_date")[TARGET_COLUMN].mean()
    return float(rates.mean()) if len(rates) else float("nan")


def topk_aggregate(preds: pd.DataFrame, prob_col: str) -> dict:
    """每日 Top1/Top3 (probability 降序, tie -> event_id 升序; 禁止用 Target).

    - top1_target_rate: Top1 命中日期的比例 (denominator = 日期数)
    - top3_target_rate: Top3 命中数 / sum(min(3, candidates)) (denominator
      = 总 picks)
    """
    daily: list[dict] = []
    for d in sorted(pd.unique(preds["signal_date"])):
        sub = preds[preds["signal_date"] == d]
        ordered = sub.sort_values([prob_col, "event_id"],
                                  ascending=[False, True]).reset_index(drop=True)
        k = min(3, len(ordered))
        top3 = ordered.head(k)
        daily.append({
            "top1": int(ordered.iloc[0][TARGET_COLUMN]),
            "hits": int(top3[TARGET_COLUMN].sum()),
            "k": k,
        })
    if not daily:
        raise RepairStateModelError("daily ranking: 无任何日期")
    d = pd.DataFrame(daily)
    return {
        "top1_target_rate": float((d["top1"] == 1).sum() / len(d)),
        "top3_target_rate": float(d["hits"].sum() / d["k"].sum()),
        "top3_picks": int(d["k"].sum()),
        "dates_with_top3_hit": int((d["hits"] > 0).sum()),
        "zero_hit_dates": int((d["hits"] == 0).sum()),
    }


def pair_weighted_within_date_auc(preds: pd.DataFrame, prob_col: str) -> float:
    """同日 (positive, negative) 全部对中, 概率排序正确的比例。

    Mann-Whitney U 的 within-date 版本: 跨日对不计入。正确对 =
    (p_pos > p_neg) + 0.5 * (p_pos == p_neg)。
    """
    total = 0
    correct = 0.0
    for d in sorted(pd.unique(preds["signal_date"])):
        sub = preds[preds["signal_date"] == d]
        y = sub[TARGET_COLUMN].to_numpy(dtype=int)
        p = sub[prob_col].to_numpy(dtype=float)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue  # 同日只有单一类别: 该日不产生 pair
        pp = p[pos_idx][:, None]
        pn = p[neg_idx][None, :]
        n = int(pp.shape[0] * pn.shape[1])
        total += n
        correct += float(np.sum(pp > pn) + 0.5 * np.sum(pp == pn))
    if total == 0:
        return float("nan")
    return float(correct / total)


def within_date_auc_stats(daily: pd.DataFrame, prob_tag: str) -> dict:
    """§19: 只对同日同时存在 positive 与 negative 的日期计算 daily AUC.

    prob_tag = "r1"/"r2" (对应 daily ranking 的 *_daily_auc / *_auc_valid 列).
    注意: 这些只是 DEVELOPMENT_IN_SAMPLE_ONLY 诊断, 不作为冻结判据。
    """
    auc_col = f"{prob_tag}_daily_auc"
    valid_col = f"{prob_tag}_auc_valid"
    valid = daily.loc[daily[valid_col], auc_col].to_numpy(dtype=float)
    n = len(valid)
    return {
        "valid_auc_dates": n,
        "mean_daily_auc": float(valid.mean()) if n else float("nan"),
        "median_daily_auc": float(np.median(valid)) if n else float("nan"),
        "dates_auc_gt_0_5": int(np.sum(valid > 0.5)),
        "dates_auc_eq_0_5": int(np.sum(valid == 0.5)),
        "dates_auc_lt_0_5": int(np.sum(valid < 0.5)),
    }


# ---------------------------------------------------------------------------
# interaction 极端样本审计 (§20; 只诊断, 禁止删行 / 再 clip / 调 C)
# ---------------------------------------------------------------------------

def interaction_audit(factors: pd.DataFrame, r2: dict,
                      df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """R2 每样本 logit contribution_j = coefficient_j * factor_j.

    返回 (audit_df, max_per_factor, anomalies):
    - audit_df 长表: 8 个 factor 的 max abs contribution 行 + top 10 abs
      DIVERGENCE_X_RECLAIM 行 + top 10 abs DIVERGENCE_X_DAMAGE 行
    - anomalies: 非有限 contribution / |contribution| 极端 (>= 10) 记录
      (只记录, 禁止根据这些样本删行 / 再 clip / winsorize / 调 C)
    """
    coefs = r2["coefs"]
    contrib = pd.DataFrame(
        {f: coefs[f] * factors[f].to_numpy(dtype=float) for f in R2_FACTORS},
        index=df.index)
    if not np.all(np.isfinite(contrib.to_numpy(dtype=float))):
        raise RepairStateModelError("interaction contribution 包含非有限值")

    max_per_factor: dict[str, dict] = {}
    rows: list[dict] = []
    anomalies: list[str] = []
    for f in R2_FACTORS:
        abs_c = contrib[f].abs()
        i = int(abs_c.idxmax())
        max_abs = float(abs_c.iloc[i])
        if max_abs >= COEFFICIENT_NUMERICAL_REVIEW:
            anomalies.append("{} max |contribution| = {:.4f} >= 10 (只记录)".format(
                f, max_abs))
        max_per_factor[f] = {
            "coefficient": float(coefs[f]),
            "max_abs_contribution": max_abs,
            "event_id": str(df.iloc[i]["event_id"]),
            "code": str(df.iloc[i]["code"]),
            "signal_date": str(df.iloc[i]["signal_date"]),
            "factor_value": float(factors.iloc[i][f]),
        }
        rows.append({
            "section": "max_abs_contribution", "factor": f, "rank": 0,
            "coefficient": float(coefs[f]),
            "max_abs_contribution": max_abs,
            "event_id": str(df.iloc[i]["event_id"]),
            "code": str(df.iloc[i]["code"]),
            "signal_date": str(df.iloc[i]["signal_date"]),
            "factor_value": float(factors.iloc[i][f]),
            "contribution": float(contrib.iloc[i][f]),
            "abs_contribution": float(abs_c.iloc[i]),
        })
    for name in ("DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_DAMAGE"):
        order = contrib[name].abs().sort_values(ascending=False).index
        for rank, i in enumerate(order[:10], start=1):
            rows.append({
                "section": "top10_{}".format(name), "factor": name, "rank": rank,
                "coefficient": float(coefs[name]),
                "max_abs_contribution": max_per_factor[name]["max_abs_contribution"],
                "event_id": str(df.loc[i, "event_id"]),
                "code": str(df.loc[i, "code"]),
                "signal_date": str(df.loc[i, "signal_date"]),
                "factor_value": float(factors.loc[i, name]),
                "contribution": float(contrib.loc[i, name]),
                "abs_contribution": float(contrib.loc[i, name] ** 2) ** 0.5,
            })
    audit_df = pd.DataFrame(rows)
    return audit_df, max_per_factor, anomalies


# ---------------------------------------------------------------------------
# frozen JSON (§25) 与重建 (§28 29/30/31)
# ---------------------------------------------------------------------------

def build_frozen_json(params: dict, r1: dict, r2: dict,
                      june_stats: dict) -> dict:
    """§25 frozen model JSON (不可再修改; 下一阶段只 apply 不 refit)."""
    return {
        "model_version": MODEL_VERSION,
        "factor_spec_version": FACTOR_SPEC_VERSION,
        "training_period": {"start": JUNE_START, "end": JUNE_END},
        "training_rows": int(june_stats["rows"]),
        "training_dates": int(june_stats["dates"]),
        "target_column": TARGET_COLUMN,
        "algorithm": {
            "library": "scikit-learn",
            "estimator": "LogisticRegression",
            **LOGISTIC_KWARGS,
            "random_state": None,
        },
        "R1": {
            "factor_order": list(R1_FACTORS),
            "intercept": float(r1["coefs"]["INTERCEPT"]),
            "coefficients": {f: float(r1["coefs"][f]) for f in R1_FACTORS},
        },
        "R2": {
            "factor_order": list(R2_FACTORS),
            "intercept": float(r2["coefs"]["INTERCEPT"]),
            "coefficients": {f: float(r2["coefs"][f]) for f in R2_FACTORS},
        },
        "primitive_transforms": {
            col: {k: float(params[col][k])
                  for k in ("q01", "q99", "mu", "sigma")}
            for col in REPAIR_PRIMITIVES},
        "composite_transforms": {
            "CLOSE_DAMAGE": {k: float(params[CLOSE_DAMAGE_COMPOSITE_KEY][k])
                             for k in ("mu", "sigma")},
            "RECLAIM": {k: float(params[RECLAIM_COMPOSITE_KEY][k])
                        for k in ("mu", "sigma")},
        },
        "derived_transforms": {
            name: {k: float(params[name][k]) for k in ("mu", "sigma")}
            for name in DERIVED_FACTORS},
        "ranking": {
            "probability": "descending",
            "tie_break": "event_id ascending",
        },
        "research_status": {
            "post_june_hypothesis": True,
            "june_is_validation": False,
            "july_target_seen": False,
        },
    }


def spec_params_from_frozen(frozen: dict) -> dict:
    """把 frozen JSON 的 transform 段合并回 spec 的 params 结构 (零重算)."""
    params: dict = dict(frozen["primitive_transforms"])
    params[CLOSE_DAMAGE_COMPOSITE_KEY] = frozen["composite_transforms"]["CLOSE_DAMAGE"]
    params[RECLAIM_COMPOSITE_KEY] = frozen["composite_transforms"]["RECLAIM"]
    params.update(frozen["derived_transforms"])
    return params


def manual_logistic_predict(model_block: dict, factors: pd.DataFrame) -> np.ndarray:
    """用 frozen 系数手工重建概率 (logit = X @ coef + intercept; sigmoid)."""
    order = list(model_block["factor_order"])
    coef = np.asarray([model_block["coefficients"][f] for f in order], dtype=float)
    X = factors[order].to_numpy(dtype=float)
    logit = X @ coef + float(model_block["intercept"])
    return 1.0 / (1.0 + np.exp(-logit))


def reconstruction_errors(frozen: dict, df: pd.DataFrame,
                          p1: np.ndarray, p2: np.ndarray) -> dict:
    """frozen JSON 重建 R1/R2 预测 vs 原训练预测的 max abs 误差 (须 < 1e-12)."""
    params = spec_params_from_frozen(frozen)
    factors = construct_repair_factors_v2(df, params)
    r1r = manual_logistic_predict(frozen["R1"], factors)
    r2r = manual_logistic_predict(frozen["R2"], factors)
    return {
        "R1": float(np.max(np.abs(r1r - p1))) if len(p1) else float("nan"),
        "R2": float(np.max(np.abs(r2r - p2))) if len(p2) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 主流程: fit_model (纯逻辑, 不写文件; 确定性, 无随机)
# ---------------------------------------------------------------------------

def fit_model(june_df: pd.DataFrame, meta: dict | None = None) -> dict:
    """June full-development fit 主流程 (R0 / R1 / R2 + 全部诊断 + frozen JSON).

    - transform 只 fit June: 复用 spec fit_repair_state_transform_v2 (一次)
    - 无超参搜索: 固定 L2 Logistic (LOGISTIC_KWARGS)
    - June 指标全部显式标记 DEVELOPMENT_IN_SAMPLE_ONLY, 不用于模型选择
    """
    df = june_df.sort_values(["signal_date", "event_id"]).reset_index(drop=True)
    y = df["target"].to_numpy(dtype=int)
    dates = sorted(pd.unique(df["signal_date"]))
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if len(np.unique(y)) < 2:
        raise RepairStateModelError("June target 只有单一类别, 禁止继续")
    if not np.all(np.isfinite(df[list(X_PRIMITIVE_COLUMNS)].to_numpy(dtype=float))):
        raise RepairStateModelError("June X primitive 包含非有限值")

    # 1) June-only transform + factor 构造 (复用冻结 spec, 不复制公式)
    params = fit_repair_state_transform_v2(df)
    factors = construct_repair_factors_v2(df, params)

    # 2) R0 = June prevalence (无 ranking 意义)
    p0 = float(y.mean())
    alpha0 = float(np.log(p0 / (1.0 - p0)))

    # 3) R1 / R2 (固定 Logistic; ConvergenceWarning 只记录 converged 标志)
    r1 = fit_logistic(factors, y, R1_FACTORS)
    r2 = fit_logistic(factors, y, R2_FACTORS)
    p1 = predict_logistic(r1, factors, R1_FACTORS)
    p2 = predict_logistic(r2, factors, R2_FACTORS)

    # 4) predictions + daily rank (probability 降序, tie -> event_id 升序)
    preds = pd.DataFrame({
        "event_id": df["event_id"].values,
        "code": df["code"].values,
        "signal_date": df["signal_date"].values,
        TARGET_COLUMN: y,
        "r0_probability": np.full(len(df), p0),
        "r1_probability": p1,
        "r2_probability": p2,
    })
    for model, pcol in (("R1", "r1_probability"), ("R2", "r2_probability")):
        ranks = pd.Series(np.zeros(len(preds), dtype=int), index=preds.index)
        for d in dates:
            mask = preds["signal_date"] == d
            ranks.loc[mask] = daily_rank(preds.loc[mask], pcol)
        preds["{}_rank_daily".format(model.lower())] = ranks

    # 5) daily ranking 表 (含 per-date daily AUC 有效性)
    daily_rows: list[dict] = []
    for d in dates:
        sub = preds[preds["signal_date"] == d]
        y_d = sub[TARGET_COLUMN].to_numpy(dtype=int)
        valid = bool((y_d == 1).any() and (y_d == 0).any())
        row = {
            "signal_date": d,
            "candidate_count": int(len(sub)),
            "candidate_target_rate": float(y_d.mean()),
        }
        for model, pcol in (("R1", "r1_probability"), ("R2", "r2_probability")):
            ordered = sub.sort_values([pcol, "event_id"],
                                      ascending=[False, True]).reset_index(drop=True)
            k = min(3, len(ordered))
            top3 = ordered.head(k)
            row["{}_top1_target".format(model.lower())] = int(ordered.iloc[0][TARGET_COLUMN])
            row["{}_top3_hits".format(model.lower())] = int(top3[TARGET_COLUMN].sum())
            row["{}_top3_target_rate".format(model.lower())] = (
                float(top3[TARGET_COLUMN].sum() / k) if k else float("nan"))
            row["{}_daily_auc".format(model.lower())] = (
                float(roc_auc(y_d, sub[pcol].to_numpy(dtype=float))) if valid
                else float("nan"))
            row["{}_auc_valid".format(model.lower())] = valid
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    # 6) metrics (DEVELOPMENT_IN_SAMPLE_ONLY; R0 的 ranking / within-date = NA)
    observed_rate = float(y.mean())
    metrics_rows: list[dict] = []
    for model, pcol in (("R0", "r0_probability"),
                        ("R1", "r1_probability"),
                        ("R2", "r2_probability")):
        p = preds[pcol].to_numpy(dtype=float)
        row = {
            "model": model,
            "development_only": DEVELOPMENT_IN_SAMPLE_ONLY,
            "logloss": logloss(y, p),
            "brier": brier_score(y, p),
            "auc": 0.5 if model == "R0" else roc_auc(y, p),
            "average_precision": (observed_rate if model == "R0"
                                  else average_precision(y, p)),
            "mean_probability": float(p.mean()),
            "observed_rate": observed_rate,
            "calibration_gap": calibration_gap(float(p.mean()), observed_rate),
        }
        if model == "R0":
            for key in ("top1_target_rate", "top3_target_rate", "top3_picks",
                        "dates_with_top3_hit", "zero_hit_dates",
                        "pooled_candidate_rate", "mean_daily_candidate_rate",
                        "valid_auc_dates", "mean_daily_auc", "median_daily_auc",
                        "pair_weighted_within_date_auc", "dates_auc_gt_0_5",
                        "dates_auc_eq_0_5", "dates_auc_lt_0_5"):
                row[key] = "NA"
        else:
            tag = model.lower()
            row.update(topk_aggregate(preds, pcol))
            row.update(within_date_auc_stats(daily, tag))
            row["pair_weighted_within_date_auc"] = pair_weighted_within_date_auc(
                preds, pcol)
            row["pooled_candidate_rate"] = observed_rate
            row["mean_daily_candidate_rate"] = mean_daily_candidate_rate(preds)
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    within_date = {
        "R1": {**within_date_auc_stats(daily, "r1"),
               "pair_weighted_within_date_auc": pair_weighted_within_date_auc(
                   preds, "r1_probability")},
        "R2": {**within_date_auc_stats(daily, "r2"),
               "pair_weighted_within_date_auc": pair_weighted_within_date_auc(
                   preds, "r2_probability")},
    }

    # 7) interaction 极端样本审计 (§20, 只诊断)
    audit_df, max_per_factor, interaction_anomalies = interaction_audit(
        factors, r2, df)

    # 8) 数值健康 (§21)
    health = coefficient_health(r1, r2, preds)

    # 9) frozen JSON + 重建验证 (§28 29/30/31)
    june_stats = {"rows": len(df), "dates": len(dates)}
    frozen = build_frozen_json(params, r1, r2, june_stats)
    repro = reconstruction_errors(frozen, df, p1, p2)

    # 10) 判定 (§30; June 指标不参与)
    status, reasons = final_freeze_status(r1, r2, health, repro)

    meta = meta or {}
    return {
        "input": meta.get("input", ""),
        "x_only_sha256": meta.get("x_only_sha256", ""),
        "june_rows": len(df),
        "june_date_count": len(dates),
        "june_date_min": dates[0],
        "june_date_max": dates[-1],
        "june_positive": pos,
        "june_negative": neg,
        "june_base_rate": observed_rate,
        "july_x_rows": meta.get("july_x_rows", 0),
        "july_x_date_count": meta.get("july_x_date_count", 0),
        "july_target_loaded": False,
        "params": params,
        "factors": factors,
        "R0": {"probability": p0, "intercept": alpha0},
        "R1": r1,
        "R2": r2,
        "predictions": preds,
        "metrics": metrics,
        "daily_ranking": daily,
        "within_date": within_date,
        "natural_candidate_rates": {
            "pooled": observed_rate,
            "mean_daily": mean_daily_candidate_rate(preds),
        },
        "interaction": {
            "audit_df": audit_df,
            "max_per_factor": max_per_factor,
            "anomalies": interaction_anomalies,
        },
        "health": health,
        "frozen": frozen,
        "repro_errors": repro,
        "status": status,
        "status_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 数值健康与判定 (§21 / §30)
# ---------------------------------------------------------------------------

def _prob_quantiles(p: np.ndarray) -> dict:
    q = np.quantile(p, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "min": float(p.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "max": float(p.max()),
    }


def coefficient_health(r1: dict, r2: dict, preds: pd.DataFrame) -> dict:
    """§21 系数与概率健康; 只记录, 禁止自动调 C."""
    out: dict = {"magnitude_watch": [], "numerical_review": [],
                 "probability_extreme_watch": False}
    for model, fit in (("R1", r1), ("R2", r2)):
        coefs = np.asarray([fit["coefs"]["INTERCEPT"]]
                           + [fit["coefs"][f] for f in
                              (R1_FACTORS if model == "R1" else R2_FACTORS)],
                           dtype=float)
        finite = bool(np.all(np.isfinite(coefs)))
        max_abs = float(np.max(np.abs(coefs))) if finite else float("nan")
        max_factor = "INTERCEPT"
        if finite:
            names = ["INTERCEPT"] + list(R1_FACTORS if model == "R1" else R2_FACTORS)
            max_factor = names[int(np.argmax(np.abs(coefs)))]
        watch = [names[i] for i in range(len(coefs)) if abs(coefs[i]) >=
                 COEFFICIENT_MAGNITUDE_WATCH] if finite else []
        severe = [names[i] for i in range(len(coefs)) if abs(coefs[i]) >=
                  COEFFICIENT_NUMERICAL_REVIEW] if finite else []
        out[model] = {
            "converged": fit["converged"],
            "n_iter": fit["n_iter"],
            "coefficients_finite": finite,
            "max_abs_coefficient": max_abs,
            "max_abs_factor": max_factor,
            "magnitude_watch_factors": watch,
            "numerical_review_factors": severe,
        }
        out["magnitude_watch"].extend(
            "{}/{}".format(model, f) for f in watch)
        out["numerical_review"].extend(
            "{}/{}".format(model, f) for f in severe)
        p = preds["{}_probability".format(model.lower())].to_numpy(dtype=float)
        lt = int(np.sum(p < 0.01))
        gt = int(np.sum(p > 0.99))
        extreme_share = (lt + gt) / len(p) if len(p) else 0.0
        out["{}_probability".format(model)] = {
            **_prob_quantiles(p),
            "count_p_lt_0_01": lt,
            "count_p_gt_0_99": gt,
        }
        if extreme_share >= PROB_EXTREME_SHARE:
            out["probability_extreme_watch"] = True
    return out


def final_freeze_status(r1: dict, r2: dict, health: dict,
                        repro: dict) -> tuple[str, list[str]]:
    """§30 判定门: 只有实现错误 / non-convergence / NaN-Inf / abs(coef) >= 10
    / frozen JSON 无法复现 才允许 MODEL_FREEZE_REVIEW_REQUIRED.

    June 指标 (AUC / LogLoss / 系数方向) 一律不参与。
    """
    reasons: list[str] = []
    if not r1["converged"]:
        reasons.append("R1 未收敛 (n_iter={} >= 1000)".format(r1["n_iter"]))
    if not r2["converged"]:
        reasons.append("R2 未收敛 (n_iter={} >= 1000)".format(r2["n_iter"]))
    if not health["R1"]["coefficients_finite"] or \
            not health["R2"]["coefficients_finite"]:
        reasons.append("coefficients 包含 NaN / inf")
    if health["numerical_review"]:
        reasons.append("abs(coefficient) >= 10: {} (MODEL_NUMERICAL_REVIEW, "
                       "禁止自动调 C)".format(", ".join(health["numerical_review"])))
    if repro["R1"] > REPRO_MAX_ERROR or repro["R2"] > REPRO_MAX_ERROR:
        reasons.append("frozen JSON 重建预测误差 > 1e-12 "
                       "(R1={:.3g}, R2={:.3g})".format(repro["R1"], repro["R2"]))
    if reasons:
        return MODEL_FREEZE_REVIEW_REQUIRED, reasons
    return FROZEN_REPAIR_STATE_MODEL_V001, []

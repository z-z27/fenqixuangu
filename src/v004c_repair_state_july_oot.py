"""v004c Repair-State Model v002 — Frozen July Retrospective OOT (July 考试)。

本模块执行正式 July OOT 评价: 只加载冻结模型 JSON -> apply July X -> 生成
target-blind 预测 -> 一次性打开 July Target -> 评价。禁止任何形式的重新
训练 / recalibration / threshold tuning / 模型修改 (July 是考试, 不是开发集)。

两阶段打开 (硬契约, §7):
- Phase A (TARGET BLIND): 加载冻结 JSON + 身份校验 (§4, 不匹配 =>
  FROZEN_MODEL_CONTRACT_MISMATCH) -> 选择 July 行 -> 只 apply 冻结 transform
  (spec_params_from_frozen + construct_repair_factors_v2, 复用 spec v002 的
  apply 逻辑, 不复制第三套公式) -> 手工重建 R0/R1/R2 概率 (logit = X@beta +
  intercept; sigmoid; 不调用任何 sklearn 拟合) -> daily rank (probability 降序,
  tie -> event_id 升序) -> 冻结 prediction dataframe + 记录
  PRE_TARGET_PREDICTION_SHA256
- Phase B (OPEN JULY TARGET ONCE): 逐行 csv 流式只解析 July 行的 Target 值
  (June / 其他行跳过, 数值从未被解析), 合并到已冻结 predictions; 之后只允许
  evaluation / reporting; 验证 POST hash == PRE hash (打开 Target 没有改变
  任何模型预测)

R0 = 冻结 June prevalence 59/173 (来自 frozen metadata, 禁止用 July
prevalence 当 R0)。判定门 (§26-29) 在本模块中预声明, 全部只依赖评价指标
(LogLoss / Brier / pair-weighted within-date AUC / Top3 lift), 不依赖
Top1 (方差太大, 只作 secondary), 结果不因 July 结果修改。

bootstrap 不确定性 (unit = signal_date, replicates = 2000, seed = 20260808)
只用于描述 21 个 signal date 下的证据精度, 不改变预声明的 pass/fail 规则。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v004c_logistic_walkforward import (
    JULY_END,
    JULY_START,
    JUNE_END,
    JUNE_START,
    TARGET_COLUMN,
    average_precision,
    brier_score,
    calibration_gap,
    daily_rank,
    logloss,
    roc_auc,
)
from src.v004c_repair_state_model_v002 import (
    FACTOR_SPEC_VERSION,
    MODEL_VERSION,
    RepairStateModelError,
    _parse_target_value,
    interaction_audit,
    load_x_window,
    manual_logistic_predict,
    mean_daily_candidate_rate,
    pair_weighted_within_date_auc,
    spec_params_from_frozen,
    within_date_auc_stats,
)
from src.v004c_repair_state_spec_v002 import (
    R1_FACTORS,
    R2_FACTORS,
    REPAIR_PRIMITIVES,
    construct_repair_factors_v2,
)

# ---------------------------------------------------------------------------
# 冻结身份 / OOT 状态常量
# ---------------------------------------------------------------------------

# 冻结元数据: June prevalence (来自冻结阶段记录; 禁止 July 统计替代)
FROZEN_JUNE_POSITIVE = 59
FROZEN_JUNE_TOTAL = 173
R0_PREVALENCE = FROZEN_JUNE_POSITIVE / FROZEN_JUNE_TOTAL  # = 59/173

FROZEN_MODEL_CONTRACT_MISMATCH = "FROZEN_MODEL_CONTRACT_MISMATCH"
REJECT_REPAIR_STATE_V002_OOT = "REJECT_REPAIR_STATE_V002_OOT"
PROMOTE_R1_FORWARD_SHADOW = "PROMOTE_R1_FORWARD_SHADOW"
PROMOTE_R2_FORWARD_SHADOW = "PROMOTE_R2_FORWARD_SHADOW"
CONFIDENCE_STRONG = "CONFIDENCE_STRONG"
CONFIDENCE_MIXED = "CONFIDENCE_MIXED"

# 正式 ranking gate: pair-weighted within-date AUC 阈值 (§26/27)
PAIR_WEIGHTED_AUC_THRESHOLD = 0.50

# bootstrap (§31): 固定参数, 只做不确定性报告
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_CI_LEVELS = (2.5, 97.5)

# §16: Phase A hash 覆盖的预测列 (prediction rows + R0 prob + R1/R2 prob/rank)
PREDICTION_HASH_COLUMNS: tuple[str, ...] = (
    "event_id", "code", "signal_date",
    "r0_probability", "r1_probability", "r1_rank",
    "r2_probability", "r2_rank",
)

# R2 系数 × factor 审计重点 (§36)
DIVERGENCE_AUDIT_FACTORS: tuple[str, ...] = (
    "DIVERGENCE", "DIVERGENCE_SQ", "DIVERGENCE_X_RECLAIM", "DIVERGENCE_X_DAMAGE",
)


class JulyOOTError(Exception):
    """v004c July OOT 失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# §4/§5: 冻结模型身份校验 (任何不匹配 => FROZEN_MODEL_CONTRACT_MISMATCH)
# ---------------------------------------------------------------------------

def validate_frozen_contract(frozen: dict) -> list[str]:
    """校验 frozen JSON 身份字段; 返回问题列表 (空 = 通过)."""
    problems: list[str] = []
    if frozen.get("model_version") != MODEL_VERSION:
        problems.append("model_version = {!r} != {}".format(
            frozen.get("model_version"), MODEL_VERSION))
    if frozen.get("factor_spec_version") != FACTOR_SPEC_VERSION:
        problems.append("factor_spec_version = {!r} != {}".format(
            frozen.get("factor_spec_version"), FACTOR_SPEC_VERSION))
    period = frozen.get("training_period", {})
    if period.get("start") != JUNE_START or period.get("end") != JUNE_END:
        problems.append("training_period = {!r} != {}/{}".format(
            period, JUNE_START, JUNE_END))
    if frozen.get("training_rows") != FROZEN_JUNE_TOTAL:
        problems.append("training_rows = {!r} != {}".format(
            frozen.get("training_rows"), FROZEN_JUNE_TOTAL))
    if frozen.get("training_dates") != 21:
        problems.append("training_dates = {!r} != 21".format(
            frozen.get("training_dates")))
    status = frozen.get("research_status", {})
    if status.get("post_june_hypothesis") is not True:
        problems.append("post_june_hypothesis != true")
    if status.get("june_is_validation") is not False:
        problems.append("june_is_validation != false")
    if status.get("july_target_seen") is not False:
        problems.append("july_target_seen != false (冻结模型不能见过 July Target)")
    r1 = frozen.get("R1", {})
    r2 = frozen.get("R2", {})
    if list(r1.get("factor_order")) != list(R1_FACTORS):
        problems.append("R1 factor_order != {}".format(list(R1_FACTORS)))
    if list(r2.get("factor_order")) != list(R2_FACTORS):
        problems.append("R2 factor_order != {}".format(list(R2_FACTORS)))
    for model_block, names in ((r1, R1_FACTORS), (r2, R2_FACTORS)):
        coefs = model_block.get("coefficients", {})
        for f in names:
            if f not in coefs:
                problems.append("{}.coefficients 缺少 {}".format(
                    "R1" if names is R1_FACTORS else "R2", f))
        if "intercept" not in model_block:
            problems.append("{} 缺少 intercept".format(
                "R1" if names is R1_FACTORS else "R2"))
    for col in REPAIR_PRIMITIVES:
        p = frozen.get("primitive_transforms", {}).get(col)
        if not p or not all(k in p for k in ("q01", "q99", "mu", "sigma")):
            problems.append("primitive_transforms[{}] 不完整".format(col))
    for key in ("CLOSE_DAMAGE", "RECLAIM"):
        p = frozen.get("composite_transforms", {}).get(key)
        if not p or not all(k in p for k in ("mu", "sigma")):
            problems.append("composite_transforms[{}] 不完整".format(key))
    return problems


def load_frozen_model(json_path) -> dict:
    """加载冻结模型 JSON + 身份校验; 不匹配 => 显式失败."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise JulyOOTError("冻结模型 JSON 不存在: {}".format(json_path))
    frozen = json.loads(json_path.read_text(encoding="utf-8"))
    problems = validate_frozen_contract(frozen)
    if problems:
        raise JulyOOTError(
            "{}: {}".format(FROZEN_MODEL_CONTRACT_MISMATCH,
                            "; ".join(problems)))
    return frozen


def june_prevalence(frozen: dict) -> float:
    """R0 July p = 冻结 June prevalence (禁止 July prevalence)."""
    if int(frozen["training_rows"]) != FROZEN_JUNE_TOTAL:
        raise JulyOOTError("frozen training_rows 与冻结元数据不一致")
    return R0_PREVALENCE


# ---------------------------------------------------------------------------
# §6: July 切片 (只读 ID + 8 core primitive; Target read = NO)
# ---------------------------------------------------------------------------

def select_july_frame(input_csv) -> tuple[pd.DataFrame, dict]:
    """Step A: 加载全窗口 X, 选择 July 行 (原始 index 保留, 供行号映射).

    窗口完整性: 全部行必须落在 June/July 窗口内 (与冻结阶段相同约定)。
    Target 列绝不在加载列中 (load_x_window 显式 usecols + 禁止 token)。
    """
    input_csv = Path(input_csv)
    window = load_x_window(input_csv)
    dates = window["signal_date"]
    june_mask = (dates >= JUNE_START) & (dates <= JUNE_END)
    july_mask = (dates >= JULY_START) & (dates <= JULY_END)
    if not bool((june_mask | july_mask).all()):
        raise JulyOOTError("model table 存在 June/July 窗口外行; 拒绝继续")
    if int(july_mask.sum()) == 0:
        raise JulyOOTError("model table 中没有 July 行")
    july = window.loc[july_mask].copy()
    stats = {
        "window_rows": int(len(window)),
        "june_x_rows": int(june_mask.sum()),
        "july_x_rows": int(july_mask.sum()),
        "july_x_date_count": int(july["signal_date"].nunique()),
        "out_of_window_rows": int((~(june_mask | july_mask)).sum()),
    }
    return july, stats


# ---------------------------------------------------------------------------
# Phase B: 逐行流式读取 July Target (一次性打开; June 值从不解析)
# ---------------------------------------------------------------------------

def read_july_target(input_csv, july_row_numbers) -> dict:
    """Phase B: 逐行 csv 流式读取, 只有 July 行解析 Target 值.

    与冻结阶段 read_june_target 相同的行号对齐约定 (跳过 \\r\\r\\n 空行,
    数据行号 = pandas 行号 + 1)。June / 其他行的 Target 值从未被解析 /
    保留 / 使用。解析行数必须与 Step A 的 July 行数一致, 否则显式失败。
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
            if data_row not in july_row_numbers:
                continue  # June / 非 July 行: Target 值不解析, 直接跳过
            values[data_row] = _parse_target_value(row[tgt_idx])
    if len(values) != len(july_row_numbers):
        raise JulyOOTError(
            "Phase B 解析 July Target 行数 {} 与 Step A July 行数 {} 不一致 "
            "(行号对齐错误)".format(len(values), len(july_row_numbers)))
    return values


# ---------------------------------------------------------------------------
# Phase A: target-blind 预测 + 冻结 hash (§15/§16)
# ---------------------------------------------------------------------------

def prediction_hash(preds: pd.DataFrame) -> str:
    """§16: 只对预测列 (不含 Target / factor) 计算 SHA256.

    打开 Target 前后该 hash 必须完全一致 (证明打开 Target 没有改变任何
    模型预测 / 排名)。
    """
    payload = preds[list(PREDICTION_HASH_COLUMNS)].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_target_blind_predictions(frozen: dict,
                                   july_df: pd.DataFrame) -> dict:
    """Phase A 全部计算 (TARGET BLIND): 冻结 transform -> 手工概率 -> rank.

    - params 只来自 frozen JSON (spec_params_from_frozen, 零重算)
    - factors 复用 spec apply (construct_repair_factors_v2, 不复制公式)
    - R1/R2 概率 = sigmoid(X@beta + intercept) (manual_logistic_predict,
      机制上无 refit; 无 sklearn 拟合)
    - R0 = 冻结 June prevalence (全行常数)
    - daily rank: probability 降序, tie -> event_id 升序; Target 不参与
    - 返回 preds (含内部 _data_row 供 Phase B 行号映射) + pre-target hash
    """
    ordered = july_df.sort_values(["signal_date", "event_id"])
    df = ordered.reset_index(drop=True)
    data_rows = (ordered.index.to_numpy(dtype=int) + 1).astype(np.int64)
    params = spec_params_from_frozen(frozen)
    factors = construct_repair_factors_v2(df, params)
    p0 = june_prevalence(frozen)
    p1 = manual_logistic_predict(frozen["R1"], factors)
    p2 = manual_logistic_predict(frozen["R2"], factors)
    if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
        raise JulyOOTError("July 概率包含非有限值")
    if not np.all((p1 > 0.0) & (p1 < 1.0)) or not np.all((p2 > 0.0) & (p2 < 1.0)):
        raise JulyOOTError("July 概率超出 (0, 1) 开区间")
    preds = pd.DataFrame({
        "event_id": df["event_id"].values,
        "code": df["code"].values,
        "signal_date": df["signal_date"].values,
        "_data_row": data_rows,
        "r0_probability": np.full(len(df), p0),
        "r1_probability": p1,
        "r2_probability": p2,
    })
    for tag, pcol in (("r1", "r1_probability"), ("r2", "r2_probability")):
        ranks = pd.Series(np.zeros(len(preds), dtype=int), index=preds.index)
        for d in sorted(pd.unique(df["signal_date"])):
            mask = preds["signal_date"] == d
            ranks.loc[mask] = daily_rank(preds.loc[mask], pcol)
        preds["{}_rank".format(tag)] = ranks
    return {
        "predictions": preds,
        "factors": factors,
        "july_raw": df,
        "pre_target_prediction_sha256": prediction_hash(preds),
    }


def open_target(preds: pd.DataFrame, target_by_row: dict) -> pd.DataFrame:
    """Phase B: 把 July Target 一次性合并进已冻结 predictions.

    - 只允许一次 (已含 Target => 显式失败)
    - 不重算 transform / 概率 / rank (只加一列)
    - 行号映射来自 Phase A 的 _data_row
    """
    preds = preds.copy()
    if TARGET_COLUMN in preds.columns:
        raise JulyOOTError("Target 已打开 (一次性契约, 禁止第二次打开)")
    values = preds["_data_row"].map(target_by_row)
    if values.isna().any():
        raise JulyOOTError("July Target 行号映射缺失 (行对齐错误)")
    preds[TARGET_COLUMN] = values.astype(int)
    return preds


# ---------------------------------------------------------------------------
# §17-24: 指标 / daily ranking / within-date / matched baselines
# ---------------------------------------------------------------------------

def daily_ranking_table(preds: pd.DataFrame) -> pd.DataFrame:
    """§24 逐日 ranking 表: candidates / Target7 count / rate / Top1/Top3 /
    daily AUC (只对正负并存的日期) / Top1 事件与 R1-R2 分歧。

    k_t = min(3, candidate_count) (§22)。
    """
    rows: list[dict] = []
    for d in sorted(pd.unique(preds["signal_date"])):
        sub = preds[preds["signal_date"] == d]
        y_d = sub[TARGET_COLUMN].to_numpy(dtype=int)
        row = {
            "signal_date": d,
            "candidate_count": int(len(sub)),
            "target7_count": int((y_d == 1).sum()),
            "candidate_rate": float(y_d.mean()),
            "k": min(3, len(sub)),
        }
        top3_ids: dict[str, set] = {}
        for tag, pcol in (("r1", "r1_probability"), ("r2", "r2_probability")):
            ordered = sub.sort_values([pcol, "event_id"],
                                      ascending=[False, True]).reset_index(drop=True)
            k = min(3, len(ordered))
            top3 = ordered.head(k)
            top3_ids[tag] = set(top3["event_id"].astype(str))
            valid = bool((y_d == 1).any() and (y_d == 0).any())
            row["{}_top1_event_id".format(tag)] = str(ordered.iloc[0]["event_id"])
            row["{}_top1_target".format(tag)] = int(ordered.iloc[0][TARGET_COLUMN])
            row["{}_top3_hits".format(tag)] = int(top3[TARGET_COLUMN].sum())
            row["{}_top3_target_rate".format(tag)] = (
                float(top3[TARGET_COLUMN].sum() / k) if k else float("nan"))
            row["{}_daily_auc".format(tag)] = (
                float(roc_auc(y_d, sub[pcol].to_numpy(dtype=float))) if valid
                else float("nan"))
            row["{}_auc_valid".format(tag)] = valid
        row["same_top1"] = ("YES" if row["r1_top1_event_id"]
                            == row["r2_top1_event_id"] else "NO")
        row["top3_overlap"] = len(top3_ids["r1"] & top3_ids["r2"])
        rows.append(row)
    return pd.DataFrame(rows)


def matched_baselines(preds: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """§21/§22 matched baselines + 自然候选率.

    Top1Baseline = (1/T) * sum_t CandidateRate_t (mean daily candidate rate)
    Top3Baseline = sum_t (k_t * CandidateRate_t) / sum_t k_t
    """
    rates = daily["candidate_rate"].to_numpy(dtype=float)
    k = daily["k"].to_numpy(dtype=int)
    return {
        "top1_baseline": float(rates.mean()),
        "top3_baseline": float(np.sum(k * rates) / np.sum(k)),
        "pooled_candidate_rate": float(preds[TARGET_COLUMN].mean()),
        "mean_daily_candidate_rate": float(rates.mean()),
    }


def topk_stats(daily: pd.DataFrame, baselines: dict, tag: str) -> dict:
    """每日 Top1/Top3 聚合 (Top1 rate 分母 = 日期数; Top3 rate 分母 =
    sum(min(3, candidates))) + matched lift (§21/§22)。"""
    top1 = daily["{}_top1_target".format(tag)].to_numpy(dtype=int)
    hits = daily["{}_top3_hits".format(tag)].to_numpy(dtype=int)
    k = daily["k"].to_numpy(dtype=int)
    n_dates = len(daily)
    top1_rate = float((top1 == 1).sum() / n_dates)
    top3_rate = float(hits.sum() / k.sum()) if k.sum() else float("nan")
    return {
        "top1_picks": n_dates,
        "top1_target_count": int((top1 == 1).sum()),
        "top1_target_rate": top1_rate,
        "top1_baseline": baselines["top1_baseline"],
        "top1_lift": top1_rate - baselines["top1_baseline"],
        "top3_picks": int(k.sum()),
        "top3_target_count": int(hits.sum()),
        "top3_target_rate": top3_rate,
        "top3_baseline": baselines["top3_baseline"],
        "top3_lift": top3_rate - baselines["top3_baseline"],
        "dates_with_top3_hit": int((hits > 0).sum()),
        "zero_hit_dates": int((hits == 0).sum()),
    }


def _metrics_row(model: str, ll: float, br: float, auc: float, ap: float,
                 mean_p: float, observed: float) -> dict:
    """单行 metrics: R0 的 ranking / within-date 字段全部为 "NA" (无 ranking
    规则, §九), R1/R2 由调用方填充。"""
    row: dict = {
        "model": model,
        "development_only": "OOT_JULY_202607",
        "logloss": ll, "brier": br, "auc": auc, "average_precision": ap,
        "mean_probability": mean_p, "observed_rate": observed,
        "calibration_gap": calibration_gap(mean_p, observed),
        "top1_target_rate": "NA", "top3_target_rate": "NA",
        "top3_picks": "NA", "dates_with_top3_hit": "NA",
        "zero_hit_dates": "NA", "pooled_candidate_rate": "NA",
        "mean_daily_candidate_rate": "NA", "valid_auc_dates": "NA",
        "mean_daily_auc": "NA", "median_daily_auc": "NA",
        "pair_weighted_within_date_auc": "NA",
        "dates_auc_gt_0_5": "NA", "dates_auc_eq_0_5": "NA",
        "dates_auc_lt_0_5": "NA",
    }
    return row


def evaluate_july(preds: pd.DataFrame) -> dict:
    """§17-24 全部评价指标装配 (Target 已打开后调用; 只读预测).

    返回 {metrics_df, daily_df, baselines, topk, within_date, gates,
    observed_rate}: metrics df 的行 = R0/R1/R2; R0 用冻结 June prevalence
    常数, auc = 0.5, average_precision = July observed (常数 p 的 PR 曲线
    单步), ranking 字段 NA; R1/R2 全字段。
    """
    daily = daily_ranking_table(preds)
    baselines = matched_baselines(preds, daily)
    topk = {
        "R1": topk_stats(daily, baselines, "r1"),
        "R2": topk_stats(daily, baselines, "r2"),
    }
    within_date = {
        "R1": {**within_date_auc_stats(daily, "r1"),
               "pair_weighted_within_date_auc":
                   pair_weighted_within_date_auc(preds, "r1_probability")},
        "R2": {**within_date_auc_stats(daily, "r2"),
               "pair_weighted_within_date_auc":
                   pair_weighted_within_date_auc(preds, "r2_probability")},
    }
    y = preds[TARGET_COLUMN].to_numpy(dtype=int)
    p0 = preds["r0_probability"].to_numpy(dtype=float)
    p1 = preds["r1_probability"].to_numpy(dtype=float)
    p2 = preds["r2_probability"].to_numpy(dtype=float)
    observed = float(y.mean())

    def full_row(model, p):
        return _metrics_row(model, logloss(y, p), brier_score(y, p),
                            roc_auc(y, p), average_precision(y, p),
                            float(p.mean()), observed)

    rows = [_metrics_row("R0", logloss(y, p0), brier_score(y, p0),
                         0.5, observed, float(p0.mean()), observed),
            full_row("R1", p1), full_row("R2", p2)]
    for r, k in zip(rows[1:], (topk["R1"], topk["R2"])):
        r.update({
            "top1_target_rate": k["top1_target_rate"],
            "top3_target_rate": k["top3_target_rate"],
            "top3_picks": k["top3_picks"],
            "dates_with_top3_hit": k["dates_with_top3_hit"],
            "zero_hit_dates": k["zero_hit_dates"],
            "pooled_candidate_rate": baselines["pooled_candidate_rate"],
            "mean_daily_candidate_rate": baselines["mean_daily_candidate_rate"],
        })
    for r, wd in zip(rows[1:], (within_date["R1"], within_date["R2"])):
        r.update({
            "valid_auc_dates": wd["valid_auc_dates"],
            "mean_daily_auc": wd["mean_daily_auc"],
            "median_daily_auc": wd["median_daily_auc"],
            "pair_weighted_within_date_auc": wd["pair_weighted_within_date_auc"],
            "dates_auc_gt_0_5": wd["dates_auc_gt_0_5"],
            "dates_auc_eq_0_5": wd["dates_auc_eq_0_5"],
            "dates_auc_lt_0_5": wd["dates_auc_lt_0_5"],
        })
    metrics = pd.DataFrame(rows)
    return {
        "metrics_df": metrics,
        "daily_df": daily,
        "baselines": baselines,
        "topk": topk,
        "within_date": within_date,
        "gates": evaluate_gates(metrics, within_date, topk),
        "observed_rate": observed,
    }


def _within_date_block_stats(blocks, y, p, eid) -> tuple[float, float]:
    """bootstrap 单次重采样: pair-weighted within-date AUC + Top3 lift.

    blocks = 每个日期出现 (occurrence) 的行下标列表; 只计 occurrence 内部
    的 (positive, negative) 对 (跨日对不计, 与 §19 定义一致); Top3 lift =
    Top3 命中率 - sum(k_t * CandidateRate_t) / sum(k_t) (§22)。
    """
    total = 0
    correct = 0.0
    hits_sum = 0
    k_sum = 0
    krate_sum = 0.0
    for idx in blocks:
        yb = y[idx]
        pb = p[idx]
        eb = eid[idx]
        pos_i = np.flatnonzero(yb == 1)
        neg_i = np.flatnonzero(yb == 0)
        if len(pos_i) and len(neg_i):
            pp = pb[pos_i][:, None]
            pn = pb[neg_i][None, :]
            n = int(pp.shape[0] * pn.shape[1])
            total += n
            correct += float(np.sum(pp > pn) + 0.5 * np.sum(pp == pn))
        k = min(3, len(idx))
        k_sum += k
        krate_sum += k * float(yb.mean())
        order = np.lexsort((eb, -pb))[:k]  # probability 降序, event_id 升序
        hits_sum += int(np.sum(yb[order]))
    pw = float(correct / total) if total else float("nan")
    if k_sum:
        top3_rate = hits_sum / k_sum
        baseline = krate_sum / k_sum
        lift = top3_rate - baseline
    else:
        lift = float("nan")
    return pw, lift


# ---------------------------------------------------------------------------
# §25-29: 预声明判定门 (读取 Target 前已定义, 不因 July 结果修改)
# ---------------------------------------------------------------------------

def evaluate_gates(metrics: pd.DataFrame, within_date: dict,
                   topk: dict) -> dict:
    """§26-29 判定门 (纯函数, 只依赖评价指标; Top1 不参与 hard gate)。

    - R1_PROBABILITY_PASS: R1 LogLoss < R0 LogLoss AND R1 Brier < R0 Brier
    - R1_RANKING_PASS: R1 pair-weighted within-date AUC > 0.50 AND
      R1 Top3 lift > 0
    - R1_OOT_PASS = 两者都成立; R2 同构
    - R2_INCREMENTAL_PASS (§28): R2 LL < R1 LL AND R2 Brier < R1 Brier AND
      R2 pair-weighted AUC >= R1 AND R2 Top3 rate >= R1
    - 最终晋级 (§29 Case A-E): REJECT / PROMOTE_R1 / PROMOTE_R2
      (Case E: R1 失败 + R2 通过 => PROMOTE_R2, R2_ONLY_OOT_PASS = YES)
    """
    m = metrics.set_index("model")
    r0, r1, r2 = m.loc["R0"], m.loc["R1"], m.loc["R2"]
    pw1 = within_date["R1"]["pair_weighted_within_date_auc"]
    pw2 = within_date["R2"]["pair_weighted_within_date_auc"]

    r1_prob_pass = bool(r1["logloss"] < r0["logloss"]
                        and r1["brier"] < r0["brier"])
    r1_rank_pass = bool(pw1 > PAIR_WEIGHTED_AUC_THRESHOLD
                        and topk["R1"]["top3_lift"] > 0.0)
    r1_pass = r1_prob_pass and r1_rank_pass

    r2_prob_pass = bool(r2["logloss"] < r0["logloss"]
                        and r2["brier"] < r0["brier"])
    r2_rank_pass = bool(pw2 > PAIR_WEIGHTED_AUC_THRESHOLD
                        and topk["R2"]["top3_lift"] > 0.0)
    r2_pass = r2_prob_pass and r2_rank_pass

    r2_incremental = bool(r2["logloss"] < r1["logloss"]
                          and r2["brier"] < r1["brier"]
                          and pw2 >= pw1
                          and topk["R2"]["top3_target_rate"]
                          >= topk["R1"]["top3_target_rate"])

    r2_only = False
    if not r1_pass and not r2_pass:
        decision = REJECT_REPAIR_STATE_V002_OOT            # Case A
    elif r1_pass and not r2_pass:
        decision = PROMOTE_R1_FORWARD_SHADOW                # Case B
    elif r1_pass and r2_pass:
        decision = (PROMOTE_R2_FORWARD_SHADOW if r2_incremental
                    else PROMOTE_R1_FORWARD_SHADOW)         # Case C / D
    else:                                                   # not r1 and r2
        decision = PROMOTE_R2_FORWARD_SHADOW                # Case E
        r2_only = True
    return {
        "R1_PROBABILITY_PASS": r1_prob_pass,
        "R1_RANKING_PASS": r1_rank_pass,
        "R1_OOT_PASS": r1_pass,
        "R2_PROBABILITY_PASS": r2_prob_pass,
        "R2_RANKING_PASS": r2_rank_pass,
        "R2_OOT_PASS": r2_pass,
        "R2_INCREMENTAL_PASS": r2_incremental,
        "FINAL_DECISION": decision,
        "R2_ONLY_OOT_PASS": r2_only,
    }


# ---------------------------------------------------------------------------
# §31/§32: signal-date cluster bootstrap (只做不确定性报告)
# ---------------------------------------------------------------------------

def cluster_bootstrap(preds: pd.DataFrame,
                      replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§31: 以 signal_date 为单位的 cluster bootstrap (有放回抽 21 个日期,
    被抽中日期带入该日全部候选; 重复日期允许重复出现)。

    返回 (replicates_df, ci_df): 每次重采样计算
    - R1/R2 LogLoss / Brier improvement vs R0 (baseline - candidate,
      positive = better)
    - R2 LogLoss / Brier improvement vs R1
    - R1/R2 pair-weighted within-date AUC
    - R1/R2 Top3 lift
    95% percentile CI (2.5 / 97.5) 只用于描述证据强弱, 不改变 §26-29 的
    pass/fail 规则 (测试 41)。
    """
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

    rows: list[dict] = []
    for _rep in range(replicates):
        sampled = rng.choice(n_dates, size=n_dates, replace=True)
        blocks = [per_date[i] for i in sampled]
        idx = (np.concatenate(blocks) if blocks else
               np.asarray([], dtype=np.int64))
        y_r, p1_r, p2_r = y[idx], p1[idx], p2[idx]
        ll0 = logloss(y_r, np.full(len(y_r), p0))
        br0 = brier_score(y_r, np.full(len(y_r), p0))
        ll1 = logloss(y_r, p1_r)
        br1 = brier_score(y_r, p1_r)
        ll2 = logloss(y_r, p2_r)
        br2 = brier_score(y_r, p2_r)
        pw1, lift1 = _within_date_block_stats(blocks, y, p1, eid)
        pw2, lift2 = _within_date_block_stats(blocks, y, p2, eid)
        rows.append({
            "r1_ll_improvement_vs_r0": ll0 - ll1,
            "r1_brier_improvement_vs_r0": br0 - br1,
            "r2_ll_improvement_vs_r0": ll0 - ll2,
            "r2_brier_improvement_vs_r0": br0 - br2,
            "r2_ll_improvement_vs_r1": ll1 - ll2,
            "r2_brier_improvement_vs_r1": br1 - br2,
            "r1_pair_weighted_within_date_auc": pw1,
            "r2_pair_weighted_within_date_auc": pw2,
            "r1_top3_lift": lift1,
            "r2_top3_lift": lift2,
        })
    rep_df = pd.DataFrame(rows)
    ci_rows: list[dict] = []
    for col in rep_df.columns:
        vals = rep_df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            lower, upper = np.percentile(finite, BOOTSTRAP_CI_LEVELS)
        else:
            lower, upper = float("nan"), float("nan")
        ci_rows.append({
            "statistic": col, "lower": float(lower), "upper": float(upper),
            "n_finite": int(len(finite)), "replicates": replicates})
    return rep_df, pd.DataFrame(ci_rows)


def confidence_label(decision: str, ci_df: pd.DataFrame) -> str:
    """§32: 被晋级模型 LogLoss / Brier / Top3 lift 的 CI 下界全部 > 0 =>
    CONFIDENCE_STRONG, 否则 CONFIDENCE_MIXED (REJECT 无晋级 => N/A).

    CONFIDENCE_MIXED 不改变预声明晋级结果, 只提醒 July 只有 21 个日期,
    后续 August+ forward shadow 更重要。
    """
    if decision == REJECT_REPAIR_STATE_V002_OOT:
        return "N/A"
    tag = "r1" if decision == PROMOTE_R1_FORWARD_SHADOW else "r2"
    c = ci_df.set_index("statistic")
    strong = bool(
        float(c.loc["{}_ll_improvement_vs_r0".format(tag), "lower"]) > 0.0
        and float(c.loc["{}_brier_improvement_vs_r0".format(tag), "lower"]) > 0.0
        and float(c.loc["{}_top3_lift".format(tag), "lower"]) > 0.0)
    return CONFIDENCE_STRONG if strong else CONFIDENCE_MIXED


# ---------------------------------------------------------------------------
# §33/§34/§35: calibration / concentration / disagreement (只描述)
# ---------------------------------------------------------------------------

def calibration_terciles(preds: pd.DataFrame, tag: str) -> pd.DataFrame:
    """§33: 按预测概率 rank 三等分 (LOW/MID/HIGH), 输出 count / mean p /
    observed Target7 rate。只描述, 禁止 recalibration / Platt / isotonic。"""
    p = preds["{}_probability".format(tag)].to_numpy(dtype=float)
    y = preds[TARGET_COLUMN].to_numpy(dtype=int)
    order = np.argsort(p, kind="stable")  # 升序 -> LOW 是底部 1/3
    groups = np.array_split(order, 3)
    rows: list[dict] = []
    for label, idx in zip(("LOW", "MID", "HIGH"), groups):
        rows.append({
            "tercile": label,
            "count": int(len(idx)),
            "mean_predicted": float(p[idx].mean()),
            "observed_rate": float(y[idx].mean()),
        })
    return pd.DataFrame(rows)


def ranking_concentration(preds: pd.DataFrame, tag: str) -> dict:
    """§34: Top20% / Bottom20% probability bucket 的 Target7 rate + spread
    (probability 降序, tie -> event_id 升序)。只描述, 不是硬 gate。"""
    p = preds["{}_probability".format(tag)].to_numpy(dtype=float)
    y = preds[TARGET_COLUMN].to_numpy(dtype=int)
    eid = preds["event_id"].astype(str).to_numpy()
    order = np.lexsort((eid, -p))  # probability 降序, event_id 升序
    n = len(p)
    k20 = max(1, int(np.ceil(n * 0.2)))
    top_rate = float(y[order[:k20]].mean())
    bottom_rate = float(y[order[-k20:]].mean())
    return {
        "bucket_size": k20,
        "top20_target_rate": top_rate,
        "bottom20_target_rate": bottom_rate,
        "spread": top_rate - bottom_rate,
    }


def disagreement_audit(daily: pd.DataFrame) -> dict:
    """§35: R1/R2 Top1 一致日期数 / 不一致日期数 / Top3 平均重叠. 只描述,
    禁止根据结果修改模型。"""
    same = int((daily["same_top1"] == "YES").sum())
    n = len(daily)
    return {
        "same_top1_dates": same,
        "different_top1_dates": n - same,
        "mean_top3_overlap": float(daily["top3_overlap"].mean()),
    }


# ---------------------------------------------------------------------------
# §36/§37: 审计 (只诊断)
# ---------------------------------------------------------------------------

def transform_audit(july_raw: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """§37: 每个 primitive 的 July 值相对冻结 q01/q99 的越界统计.

    只描述; 禁止修改 frozen q01/q99 / 删除越界行。
    """
    rows: list[dict] = []
    n = len(july_raw)
    for col in REPAIR_PRIMITIVES:
        p = frozen["primitive_transforms"][col]
        x = pd.to_numeric(july_raw[col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            raise JulyOOTError("July primitive {} 包含非有限值".format(col))
        below = int(np.sum(x < p["q01"]))
        above = int(np.sum(x > p["q99"]))
        rows.append({
            "primitive": col,
            "below_count": below,
            "above_count": above,
            "total_clipped": below + above,
            "clip_rate": float((below + above) / n),
        })
    return pd.DataFrame(rows)


def interaction_audit_july(factors: pd.DataFrame, frozen: dict,
                           july_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """§36: July 上 R2 每样本 logit contribution_j = coefficient_j * factor_j.

    复用冻结阶段 interaction_audit 实现 (同一长表结构: max abs contribution
    + top 10 abs DIVERGENCE_X_RECLAIM / DIVERGENCE_X_DAMAGE)。只记录,
    禁止 clip / drop row / 调整 coefficient。
    """
    r2_coefs = dict(frozen["R2"]["coefficients"])
    # factors 由 reset 后的 df 构建 (index = 0..n-1), 审计需与 factors 同轴;
    # july_df 的原始 index 只用于 Phase B 行号映射, 这里用 reset 副本
    return interaction_audit(factors, {"coefs": r2_coefs},
                             july_df.reset_index(drop=True))

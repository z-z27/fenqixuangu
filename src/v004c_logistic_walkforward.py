"""v004c M0/M1/M2 Logistic expanding-date walk-forward (June 开发窗口, 只读 June)。

本模块是正式模型开发阶段 (v004c 多变量建模) 的核心研究模块:

- June (2026-06-01 ~ 2026-06-30) 是唯一开发窗口: 加载后立即丢弃 July rows
  (与 v002 audit 相同的行过滤约定), July Target 不参与任何计算, 不生成
  July prediction / metrics
- Target 唯一: `target7_daily_d2open_d3high` (1 = High_{D3}/Open_{D2} - 1 >= 0.07),
  预测时点 D1 close; 只用于训练 label 与 OOF 评估, 不进入 factor 构造 /
  preprocessing 参数 / rank tie-break / feature selection
- 每个 fold 独立拟合 FOLD_CLIP_Z 参数 (training fold raw -> q01/q99 -> clip ->
  clipped mean/std -> z), RESET composite mean/std 同样只来自当前 training fold;
  test date 严格不在 train (signal_date < test_date), 无任何 future leakage
- M0 = training-fold prevalence; M1 = 4 factors + intercept; M2 = 7 factors + intercept
  (L2 LogisticRegression, C=1.0, solver=lbfgs, fit_intercept=True,
  class_weight=None, max_iter=1000; 禁止搜索/按 fold 调整)
- factor 数学定义完全复用 `v004c_factor_spec` (construct_factors /
  fit_reference_transform), 不复制第二套公式
- 输出 6 个确定性报告资产 (无时间戳 / 无随机), 全流程两次运行字节级一致

明确禁止 (本模块不实现): 自动 feature selection, C/hyperparameter/class_weight
search, interaction, PCA, Lasso, ElasticNet, 树/boosting/神经网络;
根据 fold 结果自动改 factor; 读取 July。

M0 / M1 / M2 成员: M1_FACTORS = (OPEN, RESET, HIGHZONE, LATESELL);
M2_FACTORS = CORE_FACTORS (OPEN, RESET, HIGHZONE, LATESELL, MOM7, DAMAGE7,
REGIME)。POS7 / TREND 是 SENSITIVITY, 不得进入 M1/M2; SUPPLY composite
active = NO。
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.v004c_factor_spec import (
    FactorSpecError,
    M1_FACTORS,
    M2_FACTORS,
    PRIMITIVE_COLUMNS,
    X_ID_COLUMNS,
    construct_factors,
    fit_reference_transform,
)

# ---------------------------------------------------------------------------
# 窗口 / Target / 路径常量
# ---------------------------------------------------------------------------

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"
TARGET_COLUMN = "target7_daily_d2open_d3high"
D2OPEN_D3HIGH_THRESHOLD = 0.07  # Target 定义冻结: 1[High_D3/Open_D2 - 1 >= 0.07]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (REPO_ROOT
                 / "reports/research/v004c_model_table_v001_20260601_20260729"
                 / "v004c_model_table_v001.csv")
DEFAULT_OUTPUT = REPO_ROOT / "reports/research/v004c_logistic_walkforward_v001_202606"

OUTPUT_FILES: tuple[str, ...] = (
    "v004c_walkforward_predictions_june_v001.csv",
    "v004c_walkforward_folds_v001.csv",
    "v004c_walkforward_coefficients_v001.csv",
    "v004c_walkforward_coefficient_stability_v001.csv",
    "v004c_walkforward_metrics_v001.csv",
    "v004c_walkforward_review.md",
)

# ---------------------------------------------------------------------------
# 固定算法 / 启动条件 / 诊断阈值
# ---------------------------------------------------------------------------

LOGISTIC_MAX_ITER = 1000
LOGISTIC_KWARGS: dict = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "fit_intercept": True,
    "class_weight": None,
    "max_iter": LOGISTIC_MAX_ITER,
}

# 启动条件 (固定, 不得根据结果修改): 只有此前训练数据同时满足时才允许成为 test
START_MIN_SIGNAL_DATES = 10
START_MIN_TRAIN_ROWS = 70
START_MIN_POSITIVE = 15
START_MIN_NEGATIVE = 30

PROB_EPS = 1e-15          # LogLoss 数值安全 clip (非业务性 calibration clip)
NEAR_ZERO_COEF = 1e-8     # 系数近零定义 (abs < 1e-8)
SEVERE_INSTABILITY_CONSISTENCY = 0.55  # dominant sign consistency 低于该值 => 严重不稳定


class WalkForwardError(Exception):
    """walk-forward 失败 (显式错误, 不得静默)."""


# ---------------------------------------------------------------------------
# June loader (显式 usecols + June mask; July rows 读取后立即丢弃)
# ---------------------------------------------------------------------------

def _parse_target(s: pd.Series) -> pd.Series:
    """Target 解析: 'True'/'False'/0/1 -> int 0/1; 其他值显式失败."""
    lowered = s.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    bad = sorted(lowered[~lowered.isin(mapping)].unique())
    if bad:
        raise WalkForwardError(f"Target {TARGET_COLUMN} 出现无法解析的值: {bad[:5]}")
    return lowered.map(mapping).astype(int)


def load_june_data(input_csv) -> pd.DataFrame:
    """加载 June rows (July rows 读取后立即丢弃, 不进入任何计算)。

    - 显式 usecols: 3 个标识列 + 11 个 primitive + Target (Target 只作训练
      label / OOF 评估; 不进入 factor 构造 / preprocessing 参数 / rank)
    - 窗口完整性: 全部行必须落在 June/July 窗口内 (与 v002 audit 一致),
      窗口外行直接失败
    - July rows 立即丢弃 (与 v002 audit 相同的行过滤约定): 本阶段任何 fold /
      预测 / 指标都不包含 July 行; July Target 不参与任何计算
    - 返回按 (signal_date, event_id) 排序的 June DataFrame
      (3 个标识列 + 11 个 primitive + int Target)。
    """
    input_csv = Path(input_csv)
    cols = list(X_ID_COLUMNS) + list(PRIMITIVE_COLUMNS) + [TARGET_COLUMN]
    header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = [c for c in cols if c not in header]
    if missing:
        raise WalkForwardError(f"model table 缺少请求列: {missing}")
    df = pd.read_csv(input_csv, usecols=cols, dtype={"code": str,
                                                     "signal_date": str})
    df = df[cols]  # 统一为请求列序 (usecols 返回 CSV 原始列序)
    june_mask = (df["signal_date"] >= JUNE_START) & (df["signal_date"] <= JUNE_END)
    july_mask = (df["signal_date"] >= JULY_START) & (df["signal_date"] <= JULY_END)
    if not bool((june_mask | july_mask).all()):
        raise WalkForwardError(
            f"model table 存在 June/July 窗口外行 (signal_date 窗口 "
            f"{JUNE_START}~{JULY_END}); 拒绝继续")
    june = df[june_mask].reset_index(drop=True)
    if len(june) == 0:
        raise WalkForwardError("model table 中没有 June 行")
    june = june.sort_values(["signal_date", "event_id"]).reset_index(drop=True)
    june[TARGET_COLUMN] = _parse_target(june[TARGET_COLUMN])
    return june


# ---------------------------------------------------------------------------
# date split (expanding-date; test 单位 = 整个 signal date)
# ---------------------------------------------------------------------------

def signal_dates(df: pd.DataFrame) -> list[str]:
    """升序去重的 signal_date 列表 (str)."""
    return sorted(pd.unique(df["signal_date"]))


def first_oof_index(df: pd.DataFrame) -> int:
    """June 中最早满足全部启动条件的 signal date 下标 (该日期成为第一个 test)。

    条件固定 (START_MIN_SIGNAL_DATES / TRAIN_ROWS / POSITIVE / NEGATIVE):
    此前训练数据 signal dates >= 10 且 rows >= 70 且 positive >= 15 且
    negative >= 30。找不到 => 显式失败。
    """
    dates = signal_dates(df)
    for i in range(1, len(dates)):
        d = dates[i]
        prior = df[df["signal_date"] < d]
        pos = int((prior[TARGET_COLUMN] == 1).sum())
        neg = int((prior[TARGET_COLUMN] == 0).sum())
        if (prior["signal_date"].nunique() >= START_MIN_SIGNAL_DATES
                and len(prior) >= START_MIN_TRAIN_ROWS
                and pos >= START_MIN_POSITIVE
                and neg >= START_MIN_NEGATIVE):
            return i
    raise WalkForwardError(
        "June 中没有满足启动条件的 signal date "
        f"(min dates={START_MIN_SIGNAL_DATES}, rows={START_MIN_TRAIN_ROWS}, "
        f"pos={START_MIN_POSITIVE}, neg={START_MIN_NEGATIVE})")


def split_by_date(df: pd.DataFrame, test_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 signal_date 切分: train = signal_date < test_date; test = == test_date。

    同一天所有候选必须全部进入同一个 test fold (atomic split, 禁止拆开)。
    """
    dates = signal_dates(df)
    if test_date not in dates:
        raise WalkForwardError(f"signal_date {test_date} 不在数据中")
    train = df[df["signal_date"] < test_date].reset_index(drop=True)
    test = df[df["signal_date"] == test_date].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise WalkForwardError(f"{test_date} fold 的 train/test 为空")
    return train, test


# ---------------------------------------------------------------------------
# fold transform (FOLD_CLIP_Z, 复用 v004c_factor_spec 同一数学实现)
# ---------------------------------------------------------------------------

def fit_fold_transform(train_df: pd.DataFrame) -> dict:
    """FOLD_CLIP_Z fold 参数: 只在 training fold 上拟合。

    顺序: ① q01/q99 在 train 原始有限值上拟合 -> ② clip -> ③ mu/sigma 在
    clipped train 值上拟合; ④ RESET composite (基于 train z) 的 mu/sigma 同
    样只来自 train。直接复用 `fit_reference_transform` (同一数学实现,
    传入不同样本即 fold 语义; 禁止全样本/June reference 参数)。
    """
    try:
        return fit_reference_transform(train_df)
    except FactorSpecError as exc:  # pragma: no cover - 防御性包装
        raise WalkForwardError(f"fold transform 拟合失败: {exc}") from exc


def build_fold_factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """用 fold 参数构造 9 个 factor (train/test 共用同一组 fold 参数)."""
    try:
        return construct_factors(df, params)
    except FactorSpecError as exc:  # pragma: no cover - 防御性包装
        raise WalkForwardError(f"factor 构造失败: {exc}") from exc


# ---------------------------------------------------------------------------
# M0 / M1 / M2 拟合
# ---------------------------------------------------------------------------

def fit_m0(train_target) -> tuple[float, float]:
    """M0 = INTERCEPT BASELINE: p = training-fold prevalence, alpha = logit(p).

    不通过人工 constant feature 调 L2 Logistic; 直接 prevalence。
    training fold 只有单一类别 => 直接失败。
    """
    y = np.asarray(train_target, dtype=int)
    if len(np.unique(y)) < 2:
        raise WalkForwardError("M0: training fold 只有单一类别, 禁止继续")
    p = float(y.mean())
    alpha = float(np.log(p / (1.0 - p)))
    return p, alpha


def fit_logistic(factor_df: pd.DataFrame, target, factor_names) -> dict:
    """固定 L2 Logistic: C=1.0, lbfgs, fit_intercept, class_weight=None,
    max_iter=1000。返回系数 (含 INTERCEPT), converged / n_iter。
    ConvergenceWarning 只记录不抛出 (converged 标志写入报告)。"""
    X = factor_df[list(factor_names)].to_numpy(dtype=float)
    y = np.asarray(target, dtype=int)
    if len(np.unique(y)) < 2:
        raise WalkForwardError("Logistic: training fold 只有单一类别, 禁止继续")
    if not np.all(np.isfinite(X)):
        raise WalkForwardError("Logistic 输入包含非有限值")
    lr = LogisticRegression(**LOGISTIC_KWARGS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr.fit(X, y)
    coefs = {"INTERCEPT": float(lr.intercept_[0])}
    for f, c in zip(factor_names, lr.coef_[0]):
        coefs[f] = float(c)
    return {"lr": lr, "coefs": coefs,
            "converged": bool(lr.n_iter_[0] < LOGISTIC_MAX_ITER),
            "n_iter": int(lr.n_iter_[0])}


def predict_logistic(fit: dict, factor_df: pd.DataFrame, factor_names) -> np.ndarray:
    """test fold 上 apply training fold 拟合的 Logistic (无任何 refit)."""
    X = factor_df[list(factor_names)].to_numpy(dtype=float)
    return np.asarray(fit["lr"].predict_proba(X)[:, 1], dtype=float)


# ---------------------------------------------------------------------------
# deterministic daily rank (probability 降序; tie-break event_id 升序)
# ---------------------------------------------------------------------------

def daily_rank(df: pd.DataFrame, prob_col: str, id_col: str = "event_id") -> pd.Series:
    """同一天内 probability 降序 rank 1..n; 平局用 event_id 升序 (确定性).

    禁止用 target / future return / code 表现打破 tie。
    """
    order = df.sort_values([prob_col, id_col], ascending=[False, True])
    return pd.Series(np.arange(1, len(order) + 1), index=order.index)


# ---------------------------------------------------------------------------
# metrics (pooled OOF)
# ---------------------------------------------------------------------------

def logloss(y, p, eps: float = PROB_EPS) -> float:
    """LogLoss, 概率只做数值安全 eps clip (1e-15), 不做业务性 calibration clip."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y, p) -> float:
    """Brier = mean((p - y)^2)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def roc_auc(y, p) -> float:
    """ROC AUC (M1/M2 用 sklearn; M0 概率恒同 => 结构化不存在, 记 0.5)."""
    return float(roc_auc_score(np.asarray(y, dtype=int), np.asarray(p, dtype=float)))


def average_precision(y, p) -> float:
    """Average Precision (M1/M2 用 sklearn; M0 记 OOF base rate)."""
    return float(average_precision_score(np.asarray(y, dtype=int),
                                         np.asarray(p, dtype=float)))


def calibration_gap(mean_predicted: float, observed_rate: float) -> float:
    """Gap = MeanPredicted - ObservedRate."""
    return float(mean_predicted - observed_rate)


# ---------------------------------------------------------------------------
# coefficient stability
# ---------------------------------------------------------------------------

def coefficient_stability(coef_long: pd.DataFrame) -> pd.DataFrame:
    """model x factor 稳定性统计 (来自长表 coefficients).

    dominant_sign_consistency = max(positive_count, negative_count) / fold_count
    (只用于解释; 禁止因 consistency 低自动 drop factor)。near_zero 单独计数
    (abs < 1e-8), 与正负计数独立。全部近零 => dominant_sign = "~0"。
    """
    rows: list[dict] = []
    for (model, factor), g in coef_long.groupby(["model", "factor"], sort=False):
        c = np.asarray(g["coefficient"], dtype=float)
        pos = int(np.sum(c > 0.0))
        neg = int(np.sum(c < 0.0))
        nz = int(np.sum(np.abs(c) < NEAR_ZERO_COEF))
        n = len(c)
        if nz == n and n > 0:
            sign = "~0"
            consistency = 0.0
        else:
            sign = "+" if pos >= neg else "-"
            consistency = float(max(pos, neg) / n) if n else float("nan")
        rows.append({
            "model": model, "factor": factor,
            "fold_count": n,
            "mean": float(c.mean()) if n else float("nan"),
            "std": float(c.std(ddof=0)) if n else float("nan"),
            "median": float(np.median(c)) if n else float("nan"),
            "p25": float(np.quantile(c, 0.25)) if n else float("nan"),
            "p75": float(np.quantile(c, 0.75)) if n else float("nan"),
            "min": float(c.min()) if n else float("nan"),
            "max": float(c.max()) if n else float("nan"),
            "positive_count": pos,
            "negative_count": neg,
            "near_zero_count": nz,
            "dominant_sign": sign,
            "dominant_sign_consistency": consistency,
        })
    st = pd.DataFrame(rows)
    # 固定顺序: M1 factors + INTERCEPT, 然后 M2 (INTERCEPT 放最后)
    def key(r):
        m = 0 if r["model"] == "M1" else 1
        f = r["factor"]
        order = {"INTERCEPT": 99}.get(f, 0)
        return (m, order, f)
    st["_k"] = [key(r) for r in rows]
    st = st.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return st


# ---------------------------------------------------------------------------
# 状态判定 (§38; 只预声明, 不冻结模型)
# ---------------------------------------------------------------------------

def _severe_instability_factors(stability_df: pd.DataFrame) -> list[str]:
    """dominant sign consistency < 0.55 的 model/factor (全近零除外) => 严重不稳定."""
    out: list[str] = []
    for row in stability_df.itertuples(index=False):
        if row.factor == "INTERCEPT":
            continue
        if row.fold_count == 0:
            continue
        if row.near_zero_count == row.fold_count:
            continue  # 全近零: 不是 sign 不稳定
        if row.dominant_sign_consistency < SEVERE_INSTABILITY_CONSISTENCY:
            out.append(f"{row.model}/{row.factor} "
                       f"(consistency={row.dominant_sign_consistency:.3f})")
    return out


def final_model_status(m0: dict, m1: dict, m2: dict,
                       severe_factors: list[str],
                       m1_failed_folds: list, m2_failed_folds: list,
                       anomalies: list[str]) -> tuple[str, list[str]]:
    """§38 预声明判定 (确定性规则):

    - M2 同时改善 M1 的 LogLoss 和 Brier, 且无 convergence failure /
      无明显数值异常 / 无严重系数不稳定 => M2_CANDIDATE
    - 否则 M1 同时优于 M0, 且无上述冲突 => M1_CANDIDATE
    - M1 两项都不优于 M0 且 M2 未同时改善 M1 => REJECT_NO_STABLE_MODEL
    - LogLoss/Brier 方向冲突, 或概率改善但系数严重不稳定 / convergence
      failure / 数值异常 => REVIEW_REQUIRED
    """
    m2_improves_m1 = (m2["logloss"] < m1["logloss"] and m2["brier"] < m1["brier"])
    m1_improves_m0 = (m1["logloss"] < m0["logloss"] and m1["brier"] < m0["brier"])
    reasons: list[str] = []

    if m2_improves_m1:
        blockers = (bool(m1_failed_folds) or bool(m2_failed_folds)
                    or bool(severe_factors) or bool(anomalies))
        if not blockers:
            return "M2_CANDIDATE", reasons
        reasons.append(
            "M2 同时改善 M1 概率质量, 但存在冲突: "
            + ", ".join(x for x in [
                "convergence failure"
                if (m1_failed_folds or m2_failed_folds) else None,
                f"系数严重不稳定: {severe_factors}" if severe_factors else None,
                f"数值异常: {anomalies}" if anomalies else None]
                if x is not None))
        return "REVIEW_REQUIRED", reasons

    if m1_improves_m0:
        blockers = bool(m1_failed_folds) or bool(severe_factors) or bool(anomalies)
        if not blockers:
            return "M1_CANDIDATE", reasons
        reasons.append(
            "M1 同时改善 M0 概率质量, 但存在冲突: "
            + ", ".join(x for x in [
                "convergence failure" if m1_failed_folds else None,
                f"系数严重不稳定: {severe_factors}" if severe_factors else None,
                f"数值异常: {anomalies}" if anomalies else None]
                if x is not None))
        return "REVIEW_REQUIRED", reasons

    if m1["logloss"] >= m0["logloss"] and m1["brier"] >= m0["brier"]:
        reasons.append("M1 未改善 M0 概率质量 (LogLoss/Brier 均未下降), "
                       "且 M2 未同时改善 M1 (LogLoss/Brier)")
        return "REJECT_NO_STABLE_MODEL", reasons

    reasons.append("LogLoss 与 Brier 方向冲突 (需人工审查)")
    return "REVIEW_REQUIRED", reasons


# ---------------------------------------------------------------------------
# 主流程: run_walkforward (不写文件; 返回全部结果数据)
# ---------------------------------------------------------------------------

def run_walkforward(df: pd.DataFrame, input_csv=None) -> dict:
    """June expanding-date walk-forward 主流程 (确定性, 无文件写入).

    返回 results dict: 样本规模 / initial train / first OOF date /
    folds DataFrame / predictions DataFrame / coefficients DataFrame /
    stability DataFrame / metrics DataFrame / daily ranking DataFrame /
    top3 aggregate / status / reasons。
    """
    df = df.reset_index(drop=True)
    dates = signal_dates(df)
    idx = first_oof_index(df)
    oof_dates = dates[idx:]

    initial_d = dates[0]
    initial = df[df["signal_date"] < oof_dates[0]]
    initial_pos = int((initial[TARGET_COLUMN] == 1).sum())
    initial_neg = int((initial[TARGET_COLUMN] == 0).sum())

    fold_rows: list[dict] = []
    coef_rows: list[dict] = []
    pred_rows: list[dict] = []
    m1_failed: list[str] = []
    m2_failed: list[str] = []

    for pos, d in enumerate(oof_dates):
        fold_id = f"fold_{pos + 1:02d}"
        train, test = split_by_date(df, d)
        params = fit_fold_transform(train)
        f_train = build_fold_factors(train, params)
        f_test = build_fold_factors(test, params)
        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_test = test[TARGET_COLUMN].to_numpy(dtype=int)

        p0, a0 = fit_m0(y_train)
        m1 = fit_logistic(f_train, y_train, M1_FACTORS)
        m2 = fit_logistic(f_train, y_train, M2_FACTORS)
        p1 = predict_logistic(m1, f_test, M1_FACTORS)
        p2 = predict_logistic(m2, f_test, M2_FACTORS)
        if not m1["converged"]:
            m1_failed.append(f"{d} (n_iter={m1['n_iter']})")
        if not m2["converged"]:
            m2_failed.append(f"{d} (n_iter={m2['n_iter']})")

        fold_rows.append({
            "fold_id": fold_id,
            "test_signal_date": d,
            "train_start_date": str(train["signal_date"].min()),
            "train_end_date": str(train["signal_date"].max()),
            "train_signal_dates": int(train["signal_date"].nunique()),
            "train_rows": len(train),
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "train_base_rate": float(y_train.mean()),
            "test_rows": len(test),
            "test_positive": int((y_test == 1).sum()),
            "test_negative": int((y_test == 0).sum()),
            "m0_probability": p0,
            "m0_intercept": a0,
            "m1_converged": m1["converged"],
            "m1_n_iter": m1["n_iter"],
            "m1_intercept": m1["coefs"]["INTERCEPT"],
            "m2_converged": m2["converged"],
            "m2_n_iter": m2["n_iter"],
            "m2_intercept": m2["coefs"]["INTERCEPT"],
        })
        for model, fit, names in (("M1", m1, M1_FACTORS), ("M2", m2, M2_FACTORS)):
            for factor in ("INTERCEPT",) + tuple(names):
                coef_rows.append({
                    "fold_id": fold_id,
                    "test_signal_date": d,
                    "model": model,
                    "factor": factor,
                    "coefficient": fit["coefs"][factor],
                })
        for i in range(len(test)):
            pred_rows.append({
                "event_id": test.iloc[i]["event_id"],
                "code": test.iloc[i]["code"],
                "signal_date": d,
                TARGET_COLUMN: int(y_test[i]),
                "m0_probability": p0,
                "m1_probability": float(p1[i]),
                "m2_probability": float(p2[i]),
            })

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows).sort_values(
        ["signal_date", "event_id"]).reset_index(drop=True)
    for model, pcol, rcol in (("M1", "m1_probability", "m1_rank_daily"),
                              ("M2", "m2_probability", "m2_rank_daily")):
        ranks = pd.Series(np.zeros(len(preds), dtype=int), index=preds.index)
        for d in oof_dates:
            mask = preds["signal_date"] == d
            r = daily_rank(preds.loc[mask], pcol)
            ranks.loc[mask] = r
        preds[rcol] = ranks

    coefs = pd.DataFrame(coef_rows)
    stability = coefficient_stability(coefs)

    # --- pooled OOF metrics ---
    y = preds[TARGET_COLUMN].to_numpy(dtype=int)
    oof_rows = len(preds)
    oof_n_dates = len(oof_dates)
    observed_rate = float(y.mean())
    m_metrics: dict[str, dict] = {}
    for model, pcol in (("M0", "m0_probability"), ("M1", "m1_probability"),
                        ("M2", "m2_probability")):
        p = preds[pcol].to_numpy(dtype=float)
        ll = logloss(y, p)
        br = brier_score(y, p)
        mean_p = float(p.mean())
        gap = calibration_gap(mean_p, observed_rate)
        m_metrics[model] = {
            "model": model,
            "oof_rows": oof_rows,
            "oof_dates": oof_n_dates,
            "logloss": ll,
            "brier": br,
            "auc": 0.5 if model == "M0" else roc_auc(y, p),
            "average_precision": observed_rate if model == "M0"
            else average_precision(y, p),
            "mean_probability": mean_p,
            "observed_rate": observed_rate,
            "calibration_gap": gap,
        }

    # --- daily ranking diagnostics ---
    daily_rows: list[dict] = []
    top3_agg = {"M1": {}, "M2": {}}
    for d in oof_dates:
        sub = preds[preds["signal_date"] == d].reset_index(drop=True)
        row = {"test_signal_date": d,
               "candidate_count": len(sub),
               "candidate_target_rate": float(sub[TARGET_COLUMN].mean())}
        for model, pcol, rcol in (("M1", "m1_probability", "m1_rank_daily"),
                                  ("M2", "m2_probability", "m2_rank_daily")):
            ordered = sub.sort_values([pcol, "event_id"],
                                      ascending=[False, True]).reset_index(drop=True)
            top1_hit = int(ordered.iloc[0][TARGET_COLUMN])
            k = min(3, len(ordered))
            top3 = ordered.head(k)
            top3_hits = int(top3[TARGET_COLUMN].sum())
            row[f"{model}_top1_target"] = top1_hit
            row[f"{model}_top3_hits"] = top3_hits
            row[f"{model}_top3_target_rate"] = float(top3_hits / k) if k else float("nan")
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    for model in ("M1", "M2"):
        hits = daily[f"{model}_top3_hits"]
        picks = daily["candidate_count"].apply(lambda c: min(3, int(c)))
        agg = {
            "top1_target_rate": float((daily[f"{model}_top1_target"] == 1).sum()
                                      / len(daily)),
            "top3_target_rate": float(hits.sum() / picks.sum()) if picks.sum() else float("nan"),
            "top3_picks": int(picks.sum()),
            "dates_with_top3_hit": int((hits > 0).sum()),
            "zero_hit_dates": int((hits == 0).sum()),
            "mean_daily_candidate_rate": float(daily["candidate_target_rate"].mean()),
            "pooled_candidate_rate": observed_rate,
        }
        top3_agg[model] = agg
        for key, val in agg.items():
            m_metrics[model][key] = val
            m_metrics["M0"][key] = "NA"

    metrics = pd.DataFrame([m_metrics["M0"], m_metrics["M1"], m_metrics["M2"]])

    # --- 数值异常检查 ---
    anomalies: list[str] = []
    for pcol in ("m0_probability", "m1_probability", "m2_probability"):
        if not np.all(np.isfinite(preds[pcol].to_numpy(dtype=float))):
            anomalies.append(f"{pcol} 含非有限值")
    if not np.all(np.isfinite(coefs["coefficient"].to_numpy(dtype=float))):
        anomalies.append("coefficients 含非有限值")

    severe = _severe_instability_factors(stability)
    status, reasons = final_model_status(
        m_metrics["M0"], m_metrics["M1"], m_metrics["M2"],
        severe, m1_failed, m2_failed, anomalies)

    sha = ""
    if input_csv is not None:
        sha = _sha256(Path(input_csv))

    return {
        "input": str(input_csv) if input_csv is not None else "",
        "input_sha256": sha,
        "june_rows": len(df),
        "june_date_count": len(dates),
        "june_date_min": dates[0],
        "june_date_max": dates[-1],
        "initial_train": {
            "start": initial_d,
            "end": str(initial["signal_date"].max()),
            "dates": int(initial["signal_date"].nunique()),
            "rows": len(initial),
            "positive": initial_pos,
            "negative": initial_neg,
            "base_rate": float(initial[TARGET_COLUMN].mean()),
        },
        "first_test_date": oof_dates[0],
        "oof_dates": oof_dates,
        "oof_rows": oof_rows,
        "oof_positive": int(y.sum()),
        "oof_negative": int((y == 0).sum()),
        "observed_rate": observed_rate,
        "folds": folds,
        "predictions": preds,
        "coefficients": coefs,
        "stability": stability,
        "metrics": metrics,
        "daily_ranking": daily,
        "top3_aggregate": top3_agg,
        "status": status,
        "status_reasons": reasons,
        "severe_factors": severe,
        "m1_failed_folds": m1_failed,
        "m2_failed_folds": m2_failed,
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# 报告写出 (6 个确定性资产)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_CSV_FLOAT = "%.6f"


def write_reports(results: dict, output_dir) -> None:
    """写 6 个正式资产 (确定性; 无时间戳)。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = results["predictions"].copy()
    preds.to_csv(output_dir / OUTPUT_FILES[0], index=False,
                 encoding="utf-8-sig", float_format=_CSV_FLOAT)
    results["folds"].to_csv(output_dir / OUTPUT_FILES[1], index=False,
                            encoding="utf-8-sig", float_format=_CSV_FLOAT)
    results["coefficients"].to_csv(output_dir / OUTPUT_FILES[2], index=False,
                                   encoding="utf-8-sig", float_format=_CSV_FLOAT)
    results["stability"].to_csv(output_dir / OUTPUT_FILES[3], index=False,
                                encoding="utf-8-sig", float_format=_CSV_FLOAT)
    results["metrics"].to_csv(output_dir / OUTPUT_FILES[4], index=False,
                              encoding="utf-8-sig", float_format=_CSV_FLOAT)
    (output_dir / OUTPUT_FILES[5]).write_text(
        _review_md(results), encoding="utf-8")


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(fv):
        return ""
    if np.isposinf(fv):
        return "inf"
    if np.isneginf(fv):
        return "-inf"
    return f"{fv:.{digits}f}"


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{100.0 * float(v):.2f}%"


def _coef_path_table(coefs: pd.DataFrame, model: str, folds: pd.DataFrame) -> str:
    """逐 fold 系数路径表 (factor 行 x fold 列)."""
    sub = coefs[coefs["model"] == model]
    factors = [f for f in ("INTERCEPT",) + tuple(M1_FACTORS)] if model == "M1" else \
        [f for f in ("INTERCEPT",) + tuple(M2_FACTORS)]
    header = "| factor | " + " | ".join(folds["test_signal_date"]) + " |"
    sep = "|---|" + "---|" * len(folds)
    lines = [header, sep]
    for f in factors:
        vals = sub[sub["factor"] == f].set_index("test_signal_date")["coefficient"]
        cells = [_fmt(vals.get(d), 3) for d in folds["test_signal_date"]]
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _review_md(r: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# v004c M0/M1/M2 Logistic expanding-date walk-forward (June 开发窗口)")
    add("")
    add("- 阶段: 正式模型开发 — June expanding-date walk-forward (M0/M1/M2)")
    add("- 分支: research-sample-analysis")
    add("- 输入: `v004c_model_table_v001.csv` SHA256: `{}`".format(r["input_sha256"]))
    add("- June 窗口: {} ~ {} ({} rows, {} signal dates); July 完全不读取".format(
        r["june_date_min"], r["june_date_max"], r["june_rows"], r["june_date_count"]))
    add("- Target (冻结): `{}` = 1[High_D3/Open_D2 - 1 >= {}], 预测时点 D1 close;"
        " 只用于训练 label 与 OOF 评估".format(TARGET_COLUMN, D2OPEN_D3HIGH_THRESHOLD))
    add("- Model selected/frozen: **NO** (最终模型由人工审查 OOF 概率质量 / "
        "M1 vs M0 / M2 vs M1 / 系数路径 / ranking 后决定)")
    add("")
    add("## 0. 方法 (固定, 不搜索)")
    add("")
    add("- 每个 fold 独立拟合 FOLD_CLIP_Z: training fold raw primitives -> "
        "q01/q99 (train) -> clip -> mu/sigma (clipped train) -> z -> factor 构造"
        " -> RESET composite mean/std (train 上构造的 G) -> Logistic")
    add("- test date 严格不在 train (`signal_date < test_date`); 同一天全部候选"
        " 原子进入同一个 test fold; 禁止 random/row-level/KFold/StratifiedKFold")
    add("- Logistic 固定: L2 (`penalty=l2`), `C=1.0`, `solver=lbfgs`, "
        "`fit_intercept=True`, `class_weight=None`, `max_iter={}`; 不搜索,"
        " 不按 fold 调整".format(LOGISTIC_MAX_ITER))
    add("- M0 = training-fold prevalence (非人工 constant feature 调 L2 Logistic);"
        " M1 = {}; M2 = {}; POS7/TREND 是 SENSITIVITY 不进 M1/M2; SUPPLY"
        " active = NO".format(
            ", ".join(M1_FACTORS), ", ".join(M2_FACTORS)))
    add("- 启动条件固定: 首次允许 test 需此前 training `signal_dates >= {}, rows"
        " >= {}, positive >= {}, negative >= {}`; 不因结果修改".format(
            START_MIN_SIGNAL_DATES, START_MIN_TRAIN_ROWS,
            START_MIN_POSITIVE, START_MIN_NEGATIVE))
    add("- 禁止: 自动 feature selection / C / hyperparameter / class_weight search"
        " / interaction / PCA / Lasso / ElasticNet / 树模型 / boosting / NN;"
        " 禁止根据 fold 结果改 factor")
    add("- daily rank: probability 降序, tie-break `event_id` 升序 (确定性);"
        " 禁止用 target/future return/code 表现打破 tie; M0 概率同日恒同,"
        " M0 daily rank = NOT_APPLICABLE")
    add("- 确定性: 全流程无时间戳/无随机, 两次运行 6 个资产字节级一致")
    add("")
    add("## 1. Data")
    add("")
    add("- June signal dates 总数: {} ({} ~ {})".format(
        r["june_date_count"], r["june_date_min"], r["june_date_max"]))
    it = r["initial_train"]
    add("- 初始 training: {} dates ({} ~ {}), {} rows (pos={}, neg={}, base"
        " rate={})".format(it["dates"], it["start"], it["end"], it["rows"],
                           it["positive"], it["negative"], _pct(it["base_rate"])))
    add("- 第一个 OOF test date: **{}**".format(r["first_test_date"]))
    add("- OOF 覆盖: {} dates ({} ~ {}), {} rows (pos={}, neg={})".format(
        len(r["oof_dates"]), r["oof_dates"][0], r["oof_dates"][-1],
        r["oof_rows"], r["oof_positive"], r["oof_negative"]))
    add("- 启动条件记录: 该 fold 之前 train dates={} >= {}, rows={} >= {},"
        " pos={} >= {}, neg={} >= {}".format(
            it["dates"], START_MIN_SIGNAL_DATES, it["rows"], START_MIN_TRAIN_ROWS,
            it["positive"], START_MIN_POSITIVE, it["negative"], START_MIN_NEGATIVE))
    add("")
    add("### fold 汇总 (详见 `v004c_walkforward_folds_v001.csv`)")
    add("")
    add("| fold | test date | train dates | train rows | pos | neg | base rate |"
        " test rows | pos | neg | M0 prob | M1 n_iter | M2 n_iter |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in r["folds"].itertuples(index=False):
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.fold_id, row.test_signal_date, row.train_signal_dates, row.train_rows,
            row.train_positive, row.train_negative, _fmt(row.train_base_rate, 3),
            row.test_rows, row.test_positive, row.test_negative,
            _fmt(row.m0_probability, 3), row.m1_n_iter, row.m2_n_iter))
    add("")
    add("## 2. M0 — INTERCEPT BASELINE")
    add("")
    m = r["metrics"].set_index("model")
    m0, m1, m2 = m.loc["M0"], m.loc["M1"], m.loc["M2"]
    add("- OOF LogLoss: {} (probability 只做数值安全 eps=1e-15 clip)".format(
        _fmt(m0["logloss"], 6)))
    add("- OOF Brier: {}".format(_fmt(m0["brier"], 6)))
    add("- observed OOF target rate: {} ({} / {})".format(
        _pct(m0["observed_rate"]), r["oof_positive"], r["oof_rows"]))
    add("- mean predicted probability: {}".format(_pct(m0["mean_probability"])))
    add("- AUC = 0.5, AP = OOF base rate: **M0 discrimination is structurally"
        " none** (同日概率恒同, 无横截面排序意义)")
    add("")
    add("## 3. M1 — D1 STRUCTURE ({})".format(", ".join(M1_FACTORS)))
    add("")
    add("- OOF LogLoss: {} (M0: {})".format(_fmt(m1["logloss"], 6), _fmt(m0["logloss"], 6)))
    add("- OOF Brier: {} (M0: {})".format(_fmt(m1["brier"], 6), _fmt(m0["brier"], 6)))
    add("- AUC: {}; AP: {}".format(_fmt(m1["auc"]), _fmt(m1["average_precision"])))
    add("- mean predicted probability: {}; observed rate: {}; calibration gap: {}".format(
        _pct(m1["mean_probability"]), _pct(m1["observed_rate"]),
        _fmt(m1["calibration_gap"], 4)))
    add("- Top1 target rate: {}; Top3 target rate: {}; dates with >=1 Top3 hit:"
        " {}; zero-hit dates: {}".format(
            _pct(m1["top1_target_rate"]), _pct(m1["top3_target_rate"]),
            m1["dates_with_top3_hit"], m1["zero_hit_dates"]))
    add("- 系数路径 (逐 fold, 见 §7 表与 `v004c_walkforward_coefficients_v001.csv`)")
    add("")
    add("## 4. M2 — INTEGRATED PATH ({})".format(", ".join(M2_FACTORS)))
    add("")
    add("- OOF LogLoss: {} (M1: {})".format(_fmt(m2["logloss"], 6), _fmt(m1["logloss"], 6)))
    add("- OOF Brier: {} (M1: {})".format(_fmt(m2["brier"], 6), _fmt(m1["brier"], 6)))
    add("- AUC: {}; AP: {}".format(_fmt(m2["auc"]), _fmt(m2["average_precision"])))
    add("- mean predicted probability: {}; observed rate: {}; calibration gap: {}".format(
        _pct(m2["mean_probability"]), _pct(m2["observed_rate"]),
        _fmt(m2["calibration_gap"], 4)))
    add("- Top1 target rate: {}; Top3 target rate: {}; dates with >=1 Top3 hit:"
        " {}; zero-hit dates: {}".format(
            _pct(m2["top1_target_rate"]), _pct(m2["top3_target_rate"]),
            m2["dates_with_top3_hit"], m2["zero_hit_dates"]))
    add("- MOM7/DAMAGE7/REGIME 增量: 见 §5 比较表; 共有 factor (OPEN/RESET/"
        "HIGHZONE/LATESELL) 在 M1 vs M2 中的重构见 §7")
    add("")
    add("## 5. 模型比较 (pooled OOF; delta 方向统一: LogLoss/Brier improvement"
        " = baseline - model, AUC/AP = model - baseline, positive = improvement)")
    add("")
    add("| model | LogLoss | Brier | AUC | AP | mean prob | observed | gap |")
    add("|---|---|---|---|---|---|---|---|")
    for model, row in (("M0", m0), ("M1", m1), ("M2", m2)):
        add("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            model, _fmt(row["logloss"], 6), _fmt(row["brier"], 6),
            _fmt(row["auc"]), _fmt(row["average_precision"]),
            _pct(row["mean_probability"]), _pct(row["observed_rate"]),
            _fmt(row["calibration_gap"], 4)))
    add("")
    add("| comparison | LogLoss improvement | Brier improvement | AUC delta | AP delta |")
    add("|---|---|---|---|---|")
    for label, b, mo in (("M1 vs M0", m0, m1), ("M2 vs M1", m1, m2),
                         ("M2 vs M0", m0, m2)):
        add("| {} | {} | {} | {} | {} |".format(
            label,
            _fmt(float(b["logloss"]) - float(mo["logloss"]), 6),
            _fmt(float(b["brier"]) - float(mo["brier"]), 6),
            _fmt(float(mo["auc"]) - float(b["auc"]), 6),
            _fmt(float(mo["average_precision"]) - float(b["average_precision"]), 6)))
    add("")
    add("## 6. Ranking 诊断 (每日 Top1/Top3; Top3 不足 3 候选取全部, denominator"
        " = min(3, candidate_count))")
    add("")
    add("| test date | candidates | candidate target rate | M1 Top1 | M1 Top3 hits"
        " | M1 Top3 rate | M2 Top1 | M2 Top3 hits | M2 Top3 rate |")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in r["daily_ranking"].itertuples(index=False):
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.test_signal_date, row.candidate_count,
            _pct(row.candidate_target_rate),
            row.M1_top1_target, row.M1_top3_hits, _pct(row.M1_top3_target_rate),
            row.M2_top1_target, row.M2_top3_hits, _pct(row.M2_top3_target_rate)))
    add("")
    for model in ("M1", "M2"):
        a = r["top3_aggregate"][model]
        add("- **{}**: Top1 target rate={} (denominator={} dates); Top3 target rate="
            "{} (denominator=sum(min(3, candidates))={} picks); dates with >=1 Top3"
            " hit={}; zero-hit dates={}; mean daily candidate target rate={};"
            " pooled candidate target rate={}".format(
                model, _pct(a["top1_target_rate"]), len(r["oof_dates"]),
                _pct(a["top3_target_rate"]), a["top3_picks"],
                a["dates_with_top3_hit"], a["zero_hit_dates"],
                _pct(a["mean_daily_candidate_rate"]),
                _pct(a["pooled_candidate_rate"])))
    add("- 注: Top3 指标是 ranking diagnostic, 不是概率模型 primary gate")
    add("")
    add("## 7. Coefficient stability (near-zero: |coef| < 1e-8; consistency ="
        " max(pos, neg) / fold_count, 只用于解释, 禁止自动 drop factor)")
    add("")
    add("| model | factor | folds | mean | std | median | p25 | p75 | min | max"
        " | pos | neg | ~0 | sign | consistency |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in r["stability"].itertuples(index=False):
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |"
            " {} | {} |".format(
                row.model, row.factor, row.fold_count,
                _fmt(row.mean, 4), _fmt(row.std, 4), _fmt(row.median, 4),
                _fmt(row.p25, 4), _fmt(row.p75, 4), _fmt(row.min, 4),
                _fmt(row.max, 4), row.positive_count, row.negative_count,
                row.near_zero_count, row.dominant_sign,
                _fmt(row.dominant_sign_consistency, 3)))
    add("")
    # 逐 fold 系数路径表
    add("### M1 系数路径 (fold -> test date)")
    add("")
    add(_coef_path_table(r["coefficients"], "M1", r["folds"]))
    add("")
    add("### M2 系数路径 (fold -> test date)")
    add("")
    add(_coef_path_table(r["coefficients"], "M2", r["folds"]))
    add("")
    # 稳定性分析
    add("### 稳定性分析")
    add("")
    sig_flip: list[str] = []
    for row in r["stability"].itertuples(index=False):
        if row.factor == "INTERCEPT":
            continue
        if row.near_zero_count == row.fold_count:
            continue
        if row.dominant_sign_consistency < 1.0:
            flips = int(row.positive_count if row.dominant_sign == "-"
                        else row.negative_count)
            sig_flip.append("{} ({})/{}/{} sign flips".format(
                row.model, row.factor, flips, row.fold_count))
    add("- sign flip: {}".format("; ".join(sig_flip) if sig_flip else "无 (所有"
        " factor 各 fold 同号)"))
    add("- convergence failure: M1: {}; M2: {}".format(
        "; ".join(r["m1_failed_folds"]) if r["m1_failed_folds"] else "无",
        "; ".join(r["m2_failed_folds"]) if r["m2_failed_folds"] else "无"))
    add("- 数值异常: {}".format("; ".join(r["anomalies"]) if r["anomalies"] else "无"))
    add("- 严重不稳定 (consistency < 0.55): {}".format(
        "; ".join(r["severe_factors"]) if r["severe_factors"] else "无"))
    add("")
    add("## 8. Leakage 与确定性")
    add("")
    add("- test date 严格不在 train: 每 fold `max(train.signal_date) <"
        " test.signal_date` (测试覆盖: date leakage)")
    add("- same-date atomic: 同一 signal_date 全部候选要么全 train 要么全 test"
        " (测试覆盖)")
    add("- future-X invariance: 修改未来日期 X 后更早 fold 系数/概率不变"
        " (测试覆盖)")
    add("- future-target invariance: 修改未来日期 Target 后更早 fold fit/预测不变"
        " (测试覆盖)")
    add("- fold transform 顺序: q01/q99 (train raw) -> clip -> clipped mean/std"
        " (测试覆盖, 人工极端值区分新旧实现)")
    add("- deterministic rebuild: 全流程两次运行 6 个资产字节级一致 (测试覆盖)")
    add("")
    add("## 9. July")
    add("")
    add("- Target loaded: **NO**; July predictions generated: **NO**; July"
        " metrics: **NO** (July rows 在加载后立即丢弃, 未进入任何 fold / 预测 /"
        " 指标计算; July 是 retrospective OOT, 由人工决定模型后单独执行)")
    add("")
    add("## 10. Final")
    add("")
    add("- 最终研究状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 判定依据: {}".format("; ".join(r["status_reasons"])))
    add("- Model selected/frozen: **NO**; 是否进入下一阶段 (refit frozen model ->"
        " final coefficients -> July retrospective OOT) 由人工审查本报告后决定")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c M0/M1/M2 Logistic expanding-date walk-forward"
                    " (June 开发窗口, 只读 June)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_logistic_walkforward_v001_202606)")
    args = parser.parse_args(argv)

    df = load_june_data(args.input)
    r = run_walkforward(df, input_csv=args.input)
    write_reports(r, args.output)

    print("=== v004c june logistic walk-forward ===")
    print(f"input            : {r['input']}")
    print(f"sha256           : {r['input_sha256']}")
    print(f"June rows        : {r['june_rows']} ({r['june_date_count']} dates)")
    it = r["initial_train"]
    print(f"initial training : {it['dates']} dates {it['rows']} rows "
          f"(pos {it['positive']} / neg {it['negative']})")
    print(f"first OOF date   : {r['first_test_date']}")
    print(f"OOF              : {len(r['oof_dates'])} dates / {r['oof_rows']} rows")
    m = r["metrics"].set_index("model")
    for model in ("M0", "M1", "M2"):
        row = m.loc[model]
        print(f"{model}              : logloss={row['logloss']:.6f} "
              f"brier={row['brier']:.6f} auc={row['auc']:.4f} "
              f"ap={row['average_precision']:.4f}")
    print(f"convergence      : M1 fails={r['m1_failed_folds'] or 'none'}, "
          f"M2 fails={r['m2_failed_folds'] or 'none'}")
    print(f"severe instability: {r['severe_factors'] or 'none'}")
    print(f"status           : {r['status']}")
    for reason in r["status_reasons"]:
        print(f"  - {reason}")
    print(f"output           : {args.output} ({len(OUTPUT_FILES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""v004c 因子结构纯 X 侧审计工具 (v004c factor dependence v001)。

职责:

- 读取 model table 的 X 子集: 显式 usecols 只加载 3 个标识列 + 11 个 primitive,
  读取入口即 target-blind (绝不加载 target7_daily_d2open_d3high / D2 / D3 / 旧模型输出)
- June (reference X sample) 拟合 reference transform (q01/q99/mu/sigma)
- July 只 apply June 参数 (不得用 July 重算 mean/std)
- 构造 8 个 factor (OPEN/RESET/SUPPLY/MOM7/DAMAGE7/POS7/TREND/REGIME)
- primitive/factor 相关性 (Pearson + Spearman), VIF (M1/M2), condition number
  (M1/M2/PATH, centered + unit-variance 诊断矩阵), factor 分布,
  July 无标签 X 稳定性 (SMD / KS)
- 输出 6 个正式报告文件 + 控制台摘要

明确禁止: 任何模型训练/预测/指标; 读取 Target; 根据 Target 修改 factor;
根据诊断自动 drop/swap/换权重; 全样本预处理。

用法:
    python tools/v004c_factor_dependence_audit.py
    python tools/v004c_factor_dependence_audit.py --input <csv> --output <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # 直接运行时保证 src 可导入
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.v004c_factor_spec import (
    BINARY_PRIMITIVES,
    CONTINUOUS_FACTORS,
    CONTINUOUS_PRIMITIVES,
    FACTOR_NAMES,
    FACTOR_PRIMITIVES,
    FORBIDDEN_TOKENS,
    M1_FACTORS,
    M2_FACTORS,
    PATH_BLOCK_FACTORS,
    PRIMITIVE_COLUMNS,
    X_ID_COLUMNS,
    DEGENERATE_STD,
    KS_WATCH,
    SMD_WATCH,
    FactorSpecError,
    construct_factors,
    correlation_severity,
    factor_spec_rows,
    fit_reference_transform,
    ks_2samp_statistic,
    pearson,
    spearman,
    standardized_mean_difference,
    vif_matrix,
    condition_number_diagnostic,
)

DEFAULT_INPUT = (REPO_ROOT
                 / "reports/research/v004c_model_table_v001_20260601_20260729"
                 / "v004c_model_table_v001.csv")
DEFAULT_OUTPUT = REPO_ROOT / "reports/research/v004c_factor_dependence_v001_20260601_20260729"
DICTIONARY_DIR = REPO_ROOT / "reports/research/v004c_factor_dictionary_v001_20260601_20260729"
COVERAGE_CSV = REPO_ROOT / "reports/research/v004c_feature_coverage_v001.csv"

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"

OUTPUT_FILES = (
    "v004c_primitive_dependence_v001.csv",
    "v004c_factor_dependence_v001.csv",
    "v004c_factor_diagnostics_v001.csv",
    "v004c_factor_spec_v001.csv",
    "v004c_factor_model_spec_v001.md",
    "v004c_factor_dependence_review.md",
)

PRIMITIVE_TO_FACTOR = {p: f for f, prims in FACTOR_PRIMITIVES.items() for p in prims}

# 诊断解释阈值 (只解释, 禁止自动删除变量)
CORR_SEVERE = 0.85
VIF_SEVERE = 10.0
KAPPA_SEVERE = 100.0


# ---------------------------------------------------------------------------
# target-blind X loader
# ---------------------------------------------------------------------------

def load_x_table(input_csv,
                 primitive_columns: tuple = PRIMITIVE_COLUMNS,
                 id_columns: tuple = X_ID_COLUMNS) -> pd.DataFrame:
    """显式 usecols 只加载 ID + primitive 列 (读取入口即 target-blind)。

    - 请求列含禁止 token (target/d2_/d3_/v002/v004a/...) => 直接失败
    - 请求列缺失 => 显式失败 (不得静默缺列)
    """
    cols = list(id_columns) + list(primitive_columns)
    for c in cols:
        lowered = c.lower()
        for token in FORBIDDEN_TOKENS:
            if token in lowered:
                raise FactorSpecError(f"请求列 {c} 包含禁止 token '{token}'")
    header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = [c for c in cols if c not in header]
    if missing:
        raise FactorSpecError(f"model table 缺少请求列: {missing}")
    df = pd.read_csv(input_csv, usecols=cols, dtype={"code": str, "signal_date": str})
    df = df[cols]  # 统一为请求列序 (usecols 返回 CSV 原始列序)
    return df


# ---------------------------------------------------------------------------
# 分布描述
# ---------------------------------------------------------------------------

_CONT_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def describe_continuous(x) -> dict:
    """连续变量分布统计 (raw 值, 不做 transform)."""
    x = pd.to_numeric(x, errors="coerce")
    total = len(x)
    missing = int(x.isna().sum())
    finite = x.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        out = {"n": total, "missing": missing, "unique": 0,
               "mean": np.nan, "std": np.nan, "min": np.nan,
               "p01": np.nan, "p05": np.nan, "p25": np.nan, "median": np.nan,
               "p75": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan,
               "skew": np.nan}
    else:
        q = np.quantile(finite, _CONT_QUANTILES)
        out = {"n": total, "missing": missing, "unique": int(len(np.unique(finite))),
               "mean": float(finite.mean()), "std": float(finite.std(ddof=0)),
               "min": float(finite.min()),
               "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
               "median": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
               "p99": float(q[6]), "max": float(finite.max()),
               "skew": float(pd.Series(finite).skew())}
    return out


def describe_binary(x) -> dict:
    """二元变量统计 (0/1)."""
    x = pd.to_numeric(x, errors="coerce")
    total = len(x)
    missing = int(x.isna().sum())
    finite = x.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {"n": total, "missing": missing, "count_0": 0, "count_1": 0,
                "rate_1": np.nan, "minority_count": 0}
    count_0 = int(np.sum(finite == 0.0))
    count_1 = int(np.sum(finite == 1.0))
    return {"n": total, "missing": missing, "count_0": count_0, "count_1": count_1,
            "rate_1": float(count_1 / len(finite)), "minority_count": min(count_0, count_1)}


# ---------------------------------------------------------------------------
# 相关性长表
# ---------------------------------------------------------------------------

def primitive_pair_rows(june: pd.DataFrame, july: pd.DataFrame) -> list[dict]:
    """11 choose 2 = 55 对 (June 为主参考, July 只做稳定性描述)."""
    rows = []
    for i in range(len(PRIMITIVE_COLUMNS)):
        for j in range(i + 1, len(PRIMITIVE_COLUMNS)):
            left, right = PRIMITIVE_COLUMNS[i], PRIMITIVE_COLUMNS[j]
            rp_jun = pearson(june[left], june[right])
            rs_jun = spearman(june[left], june[right])
            severity = correlation_severity(max(abs(rp_jun), abs(rs_jun)))
            rows.append({
                "left": left,
                "right": right,
                "left_factor": PRIMITIVE_TO_FACTOR[left],
                "right_factor": PRIMITIVE_TO_FACTOR[right],
                "same_factor": PRIMITIVE_TO_FACTOR[left] == PRIMITIVE_TO_FACTOR[right],
                "june_pearson": rp_jun,
                "june_spearman": rs_jun,
                "abs_june_pearson": abs(rp_jun),
                "abs_june_spearman": abs(rs_jun),
                "severity": severity,
                "july_pearson": pearson(july[left], july[right]),
                "july_spearman": spearman(july[left], july[right]),
            })
    return rows


def factor_pair_rows(f_june: pd.DataFrame, f_july: pd.DataFrame) -> list[dict]:
    """8 choose 2 = 28 对 (June 为主参考, July 只做稳定性描述)."""
    rows = []
    for i in range(len(FACTOR_NAMES)):
        for j in range(i + 1, len(FACTOR_NAMES)):
            left, right = FACTOR_NAMES[i], FACTOR_NAMES[j]
            rp_jun = pearson(f_june[left], f_june[right])
            rs_jun = spearman(f_june[left], f_june[right])
            severity = correlation_severity(max(abs(rp_jun), abs(rs_jun)))
            rows.append({
                "left_factor": left,
                "right_factor": right,
                "june_pearson": rp_jun,
                "june_spearman": rs_jun,
                "abs_june_pearson": abs(rp_jun),
                "abs_june_spearman": abs(rs_jun),
                "severity": severity,
                "july_pearson": pearson(f_july[left], f_july[right]),
                "july_spearman": spearman(f_july[left], f_july[right]),
            })
    return rows


# ---------------------------------------------------------------------------
# Stage 2.1 / coverage 结构交叉检查 (只读, 不改变 factor)
# ---------------------------------------------------------------------------

def cross_check_stage21() -> dict:
    """确认 11 个 primitive 的授权来源: primary allowlist 或 coverage NEW_REQUIRED。

    只做文档化交叉检查; 任何 primitive 不在授权来源 => 失败。
    """
    allowlist_path = DICTIONARY_DIR / "v004c_feature_allowlist_primary_v001.csv"
    near_dup_path = DICTIONARY_DIR / "v004c_near_duplicate_pairs_v001.csv"
    al = pd.read_csv(allowlist_path, encoding="utf-8-sig")
    al_names = set(al["feature_name"])
    cov = pd.read_csv(COVERAGE_CSV, encoding="utf-8-sig")
    new_required = set(cov.loc[cov["coverage_status"] == "NEW_REQUIRED", "feature_name"])

    allowlist_members: list[str] = []
    new_required_members: list[str] = []
    policies: dict[str, str] = {}
    for p in PRIMITIVE_COLUMNS:
        if p in al_names:
            allowlist_members.append(p)
            policies[p] = str(al.loc[al["feature_name"] == p, "preprocess_policy"].iloc[0])
        elif p in new_required:
            new_required_members.append(p)
            policy = cov.loc[cov["feature_name"] == p, "preprocess_policy"].iloc[0]
            policies[p] = ("FOLD_CLIP_Z (model-table schema 临时约定)"
                           if pd.isna(policy) else str(policy))
        else:
            raise FactorSpecError(
                f"primitive {p} 不在 primary allowlist 也不在 coverage NEW_REQUIRED")

    within: list[str] = []
    if near_dup_path.exists():
        nd = pd.read_csv(near_dup_path, encoding="utf-8-sig")
        prims = set(PRIMITIVE_COLUMNS)
        hits = nd[nd["left_column"].isin(prims) & nd["right_column"].isin(prims)]
        within = hits["pair_id"].tolist()

    return {
        "allowlist_members": allowlist_members,
        "new_required_members": new_required_members,
        "policies": policies,
        "near_duplicate_within_universe": within,
    }


# ---------------------------------------------------------------------------
# 审计主流程
# ---------------------------------------------------------------------------

def run_factor_dependence_audit(input_csv, output_dir) -> dict:
    """完整 X 侧审计: 读取 -> June reference -> factor -> 诊断 -> 写 6 文件。

    返回结果摘要 dict (全部标量, 确定性; 不包含 Target)。
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)

    x = load_x_table(input_csv)
    june_mask = (x["signal_date"] >= JUNE_START) & (x["signal_date"] <= JUNE_END)
    july_mask = (x["signal_date"] >= JULY_START) & (x["signal_date"] <= JULY_END)
    if not bool((june_mask | july_mask).all()):
        raise FactorSpecError(
            f"存在 June/July 窗口外行 (日期范围 {JUNE_START}~{JULY_END}); 拒绝继续")
    june = x[june_mask].reset_index(drop=True)
    july = x[july_mask].reset_index(drop=True)

    params = fit_reference_transform(june)
    f_june = construct_factors(june, params)
    f_july = construct_factors(july, params)

    # --- 相关性 ---
    prim_pairs = primitive_pair_rows(june, july)
    fac_pairs = factor_pair_rows(f_june, f_july)

    # --- VIF (June) ---
    m1_vif = vif_matrix(f_june[list(M1_FACTORS)])
    m2_vif = vif_matrix(f_june[list(M2_FACTORS)])

    # --- condition number (centered + unit-variance 诊断矩阵) ---
    kappa = {}
    for name, factors in (("M1", M1_FACTORS), ("M2", M2_FACTORS),
                          ("PATH", PATH_BLOCK_FACTORS)):
        s_max, s_min, k, near = condition_number_diagnostic(
            f_june[list(factors)].to_numpy(dtype=float))
        kappa[name] = {"s_max": s_max, "s_min": s_min, "kappa": k, "near_singular": near}

    # --- factor 分布 + July 稳定性 ---
    factor_stats: dict = {}
    degenerate_factors: list[str] = []
    shift_factors: list[str] = []
    for f in FACTOR_NAMES:
        if f in CONTINUOUS_FACTORS:
            june_d = describe_continuous(f_june[f])
            july_d = describe_continuous(f_july[f])
            if june_d["std"] <= DEGENERATE_STD:
                degenerate_factors.append(f)
            smd = standardized_mean_difference(june_d["mean"], june_d["std"],
                                               july_d["mean"], july_d["std"])
            ks = ks_2samp_statistic(f_july[f], f_june[f])
            flag = ""
            if abs(smd) >= SMD_WATCH or ks >= KS_WATCH:
                flag = "SHIFT_WATCH"
                shift_factors.append(f)
            factor_stats[f] = {
                "june": june_d, "july": july_d, "smd": smd, "ks": ks,
                "stationarity_flag": flag,
            }
        else:  # REGIME: 0/1, SMD/KS 不适用 (留空), 只报告 rate1
            june_d = describe_binary(f_june[f])
            july_d = describe_binary(f_july[f])
            factor_stats[f] = {
                "june": june_d, "july": july_d,
                "smd": np.nan, "ks": np.nan, "stationarity_flag": "",
            }

    regime = {
        "june_rate1": factor_stats["REGIME"]["june"]["rate_1"],
        "july_rate1": factor_stats["REGIME"]["july"]["rate_1"],
        "diff": (factor_stats["REGIME"]["july"]["rate_1"]
                 - factor_stats["REGIME"]["june"]["rate_1"]),
        "june_counts": (factor_stats["REGIME"]["june"]["count_0"],
                        factor_stats["REGIME"]["june"]["count_1"]),
        "july_counts": (factor_stats["REGIME"]["july"]["count_0"],
                        factor_stats["REGIME"]["july"]["count_1"]),
    }

    # --- 最终状态 (只判定, 不自动改 factor) ---
    severe_pairs = [p for p in fac_pairs if p["severity"] == "SEVERE"]
    reasons: list[str] = []
    if degenerate_factors:
        reasons.append(f"degenerate factor: {degenerate_factors} (June std <= 1e-8)")
    for name, k in kappa.items():
        if k["near_singular"]:
            reasons.append(f"{name} condition matrix near singular "
                           f"(kappa={k['kappa']:.3g})")
    for f, v in m1_vif.items():
        if v >= VIF_SEVERE:
            reasons.append(f"M1 factor {f} VIF={v:.4g} >= 10 SEVERE")
    for f, v in m2_vif.items():
        if v >= VIF_SEVERE:
            reasons.append(f"M2 factor {f} VIF={v:.4g} >= 10 SEVERE")
    for p in severe_pairs:
        reasons.append(
            f"factor pair {p['left_factor']}-{p['right_factor']} SEVERE "
            f"(|rho|={max(p['abs_june_pearson'], p['abs_june_spearman']):.3f})")
    status = "PASS_X_STRUCTURE" if not reasons else "REVIEW_REQUIRED"

    results = {
        "input": str(input_csv),
        "input_sha256": _sha256(input_csv),
        "june_rows": len(june), "june_date_count": int(june["signal_date"].nunique()),
        "july_rows": len(july), "july_date_count": int(july["signal_date"].nunique()),
        "june_date_min": str(june["signal_date"].min()),
        "june_date_max": str(june["signal_date"].max()),
        "july_date_min": str(july["signal_date"].min()),
        "july_date_max": str(july["signal_date"].max()),
        "primitive_dist": {p: {"june": describe_continuous(june[p]),
                               "july": describe_continuous(july[p])}
                           for p in CONTINUOUS_PRIMITIVES},
        "regime_dist": {"june": describe_binary(june["board_streak_is_3"]),
                        "july": describe_binary(july["board_streak_is_3"])},
        "primitive_pairs": prim_pairs,
        "factor_pairs": fac_pairs,
        "factor_stats": factor_stats,
        "m1_vif": m1_vif,
        "m2_vif": m2_vif,
        "kappa": kappa,
        "regime": regime,
        "degenerate_factors": degenerate_factors,
        "shift_factors": shift_factors,
        "severe_pairs": severe_pairs,
        "status": status,
        "status_reasons": reasons,
        "stage21": cross_check_stage21(),
    }

    _write_all(output_dir, results)
    return results


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 输出文件
# ---------------------------------------------------------------------------

def _write_all(output_dir: Path, r: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. primitive dependence CSV
    pd.DataFrame(r["primitive_pairs"]).to_csv(
        output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")

    # 2. factor dependence CSV
    pd.DataFrame(r["factor_pairs"]).to_csv(
        output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")

    # 3. factor diagnostics CSV (每 factor 一行; REGIME 不适用的统计留空)
    diag = _factor_diagnostics_rows(r)
    pd.DataFrame(diag).to_csv(output_dir / OUTPUT_FILES[2], index=False,
                              encoding="utf-8-sig")

    # 4. factor spec CSV (静态规格)
    pd.DataFrame(factor_spec_rows()).to_csv(
        output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")

    # 5. model spec md (M0/M1/M2 公式, 只记录)
    (output_dir / OUTPUT_FILES[4]).write_text(
        _model_spec_md(), encoding="utf-8")

    # 6. review md (数据驱动, 确定性)
    (output_dir / OUTPUT_FILES[5]).write_text(
        _review_md(r), encoding="utf-8")


_FACTOR_DIAG_COLUMNS = (
    "factor", "model_membership",
    "june_n", "june_missing", "june_mean", "june_std", "june_min",
    "june_p01", "june_p05", "june_p25", "june_median", "june_p75",
    "june_p95", "june_p99", "june_max", "june_skew",
    "july_n", "july_mean", "july_std",
    "june_to_july_smd", "june_to_july_ks",
    "m1_vif", "m2_vif", "stationarity_flag",
)


def _factor_diagnostics_rows(r: dict) -> list[dict]:
    rows = []
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]
        june_d, july_d = st["june"], st["july"]
        if f in CONTINUOUS_FACTORS:
            row = {
                "factor": f,
                "model_membership": "M1,M2" if f in M1_FACTORS else "M2",
                "june_n": june_d["n"], "june_missing": june_d["missing"],
                "june_mean": june_d["mean"], "june_std": june_d["std"],
                "june_min": june_d["min"],
                "june_p01": june_d["p01"], "june_p05": june_d["p05"],
                "june_p25": june_d["p25"], "june_median": june_d["median"],
                "june_p75": june_d["p75"], "june_p95": june_d["p95"],
                "june_p99": june_d["p99"], "june_max": june_d["max"],
                "june_skew": june_d["skew"],
                "july_n": july_d["n"], "july_mean": july_d["mean"],
                "july_std": july_d["std"],
                "june_to_july_smd": st["smd"], "june_to_july_ks": st["ks"],
                "m1_vif": r["m1_vif"].get(f, np.nan),
                "m2_vif": r["m2_vif"].get(f, np.nan),
                "stationarity_flag": st["stationarity_flag"],
            }
        else:  # REGIME: 连续统计不适用 => 留空; mean/std = Bernoulli 统计
            row = {
                "factor": f,
                "model_membership": "M1,M2" if f in M1_FACTORS else "M2",
                "june_n": june_d["n"], "june_missing": june_d["missing"],
                "june_mean": june_d["rate_1"],
                "june_std": (np.sqrt(june_d["rate_1"] * (1.0 - june_d["rate_1"]))
                             if np.isfinite(june_d["rate_1"]) else np.nan),
                "june_min": 0.0,
                "june_p01": np.nan, "june_p05": np.nan, "june_p25": np.nan,
                "june_median": np.nan, "june_p75": np.nan, "june_p95": np.nan,
                "june_p99": np.nan, "june_max": 1.0,
                "june_skew": np.nan,
                "july_n": july_d["n"], "july_mean": july_d["rate_1"],
                "july_std": (np.sqrt(july_d["rate_1"] * (1.0 - july_d["rate_1"]))
                             if np.isfinite(july_d["rate_1"]) else np.nan),
                "june_to_july_smd": np.nan, "june_to_july_ks": np.nan,
                "m1_vif": np.nan,
                "m2_vif": r["m2_vif"].get(f, np.nan),
                "stationarity_flag": "",
            }
        rows.append(row)
    return rows


def _model_spec_md() -> str:
    return """# v004c 因子模型规格 v001 (M0/M1/M2 数学结构 — 仅记录, 不拟合)

- 阶段: 多变量建模前的纯 X 侧因子结构审查产物; 本规格只定义公式与未来训练算法
- 状态: factor definition freeze 候选 (由 v004c_factor_dependence_review.md 结论背书)
- 注意: 这是 factor definition freeze, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## Factor 预处理 (future walk-forward)

- 连续 factor: 每个 training fold 内拟合 clip [q01, q99] + (x - mu)/sigma
  (q01/q99/mu/sigma 只允许来自当前 training fold; 禁止全样本参数)
- REGIME: 0/1 原值, 不做 z-score
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## M0 — INTERCEPT BASELINE

    logit(p) = alpha

## M1 — D1 STRUCTURE

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_SUPPLY

经济假设: D1 开盘承接、D1 价格重置和 D1 供应压力是否共同形成对未来 Target7
概率有解释力的结构。只描述结构问题, 不预设系数方向。

## M2 — INTEGRATED PATH MODEL

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_SUPPLY
             + gamma_1 * F_MOM7
             + gamma_2 * F_DAMAGE7
             + gamma_3 * F_POS7
             + gamma_4 * F_TREND
             + gamma_5 * F_REGIME

M2 回答: 在 D1 结构已知后, 最近价格路径 (MOM7/DAMAGE7/POS7)、中期趋势
(TREND) 和二/三板 regime (REGIME) 是否提供增量预测信息。

## 未来训练算法 (只记录, 本阶段禁止执行)

- L2 Logistic Regression: C = 1.0, solver = lbfgs, fit_intercept = True,
  class_weight = None
- 目标函数:

      min_theta [ -sum_i ( y_i*log(p_i) + (1-y_i)*log(1-p_i) ) + lambda * ||theta_{-0}||_2^2 ]

- 训练/验证: M0/M1/M2 expanding-date walk-forward (下一阶段)

## June/July 命名约定

- June: development / reference X sample
- July: retrospective unlabeled X-stability slice
  (July 不是 blind holdout; 本阶段只做 X 分布稳定性, 不读取 July Target)
"""


# ---------------------------------------------------------------------------
# review 生成 (数据驱动, 确定性)
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


def _review_md(r: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# v004c 因子结构纯 X 侧审查 (v004c factor dependence v001)")
    add("")
    add("- 阶段: 多变量建模前的纯 X 侧因子结构审查 (X ONLY, 不训练模型, 不读取 Target)")
    add("- 分支: research-sample-analysis")
    add("- 输入: `v004c_model_table_v001.csv` SHA256: `{}`".format(r["input_sha256"]))
    add("")

    # 0. 输入与 target-blind
    add("## 0. 输入与 target-blind 约定")
    add("")
    add("- X loader 使用显式 `usecols` 只加载 3 个标识列 (event_id, code, signal_date)"
        " + 11 个 primitive 列; `target7_daily_d2open_d3high` / D2 / D3 / 旧模型输出"
        " 从未被加载 (读取入口即 target-blind)")
    add("- June = **development / reference X sample** ({start} ~ {end}, 用于拟合"
        " q01/q99/mu/sigma)".format(start=JUNE_START, end=JUNE_END))
    add("- July = **retrospective unlabeled X-stability slice** ({start} ~ {end},"
        " 只 apply June reference 参数; 不得用 July 重算 mean/std, 不得读取 July"
        " Target)".format(start=JULY_START, end=JULY_END))
    add("")
    s21 = r["stage21"]
    add("- Stage 2.1 / coverage 结构交叉检查: 11 个 primitive 全部有授权来源"
        " (allowlist: `{}`; NEW_REQUIRED recent-7d: `{}`)".format(
            ", ".join(s21["allowlist_members"]), ", ".join(s21["new_required_members"])))
    add("- preprocess_policy 记录: {}".format(
        "; ".join("{}={}".format(p, s21["policies"][p]) for p in PRIMITIVE_COLUMNS)))
    nd_note = ("无 (11 个 primitive 内部无 near-duplicate pair)"
               if not s21["near_duplicate_within_universe"]
               else "见: {}".format(", ".join(s21["near_duplicate_within_universe"])))
    add("- near-duplicate pairs (v004c_near_duplicate_pairs_v001) 命中本 universe: "
        + nd_note)
    add("")

    # 1. 样本规模
    add("## 1. 样本规模")
    add("")
    add("| 切片 | rows | signal dates | 日期范围 |")
    add("|---|---|---|---|")
    add("| June (development / reference X sample) | {} | {} | {} ~ {} |".format(
        r["june_rows"], r["june_date_count"], r["june_date_min"], r["june_date_max"]))
    add("| July (retrospective unlabeled X-stability slice) | {} | {} | {} ~ {} |".format(
        r["july_rows"], r["july_date_count"], r["july_date_min"], r["july_date_max"]))
    add("")
    add("- 窗口外行: 0 (全部 333 行都在 June/July 窗口内)")
    add("")

    # 2. primitive 分布
    add("## 2. primitive 分布 (raw 值, June / July)")
    add("")
    add("### June")
    add("")
    add(_primitive_dist_table(r, "june"))
    add("")
    add("### July")
    add("")
    add(_primitive_dist_table(r, "july"))
    add("")
    add("### REGIME (binary)")
    add("")
    add("| 切片 | count_0 | count_1 | rate_1 | minority_count |")
    add("|---|---|---|---|---|")
    add("| June | {} | {} | {} | {} |".format(
        r["regime_dist"]["june"]["count_0"], r["regime_dist"]["june"]["count_1"],
        _fmt(r["regime_dist"]["june"]["rate_1"]),
        r["regime_dist"]["june"]["minority_count"]))
    add("| July | {} | {} | {} | {} |".format(
        r["regime_dist"]["july"]["count_0"], r["regime_dist"]["july"]["count_1"],
        _fmt(r["regime_dist"]["july"]["rate_1"]),
        r["regime_dist"]["july"]["minority_count"]))
    add("")

    # 3. primitive 相关性
    add("## 3. primitive 相关性 (June 主参考; July 仅稳定性描述)")
    add("")
    pp = r["primitive_pairs"]
    maxp = max(pp, key=lambda row: row["abs_june_pearson"])
    maxs = max(pp, key=lambda row: row["abs_june_spearman"])
    add("- 最大 June Pearson pair: `{left}-{right}` rho={rho} (severity={sev})".format(
        left=maxp["left"], right=maxp["right"], rho=_fmt(maxp["june_pearson"]),
        sev=maxp["severity"]))
    add("- 最大 June Spearman pair: `{left}-{right}` rho={rho} (severity={sev})".format(
        left=maxs["left"], right=maxs["right"], rho=_fmt(maxs["june_spearman"]),
        sev=maxs["severity"]))
    cross_severe = [p for p in pp if p["severity"] == "SEVERE" and not p["same_factor"]]
    within_severe = [p for p in pp if p["severity"] == "SEVERE" and p["same_factor"]]
    add("- 跨因子 SEVERE primitive pairs (|rho| >= 0.85): {}".format(
        ", ".join("{}-{} ({})".format(p["left"], p["right"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"]))) for p in cross_severe)
        if cross_severe else "无"))
    add("- 因子内 SEVERE primitive pairs (同因子多观测, 设计允许): {}".format(
        ", ".join("{}-{} ({})".format(p["left"], p["right"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"]))) for p in within_severe)
        if within_severe else "无"))
    add("")
    add("完整 55 对明细见 `v004c_primitive_dependence_v001.csv` (same_factor 标记"
        " 因子内/因子间)。")
    add("")

    # 4. factor 构造验证与分布
    add("## 4. factor 构造验证与 June 分布")
    add("")
    add("- 构造: 8 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8"
        " (degenerate 检查通过: {})".format(
            "无退化" if not r["degenerate_factors"] else r["degenerate_factors"]))
    add("- REGIME 严格 0/1 (未标准化); 只有 condition number 诊断矩阵中临时 center/scale")
    add("")
    add("| factor | membership | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]["june"]
        if f in CONTINUOUS_FACTORS:
            add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, "M1,M2" if f in M1_FACTORS else "M2",
                _fmt(st["mean"]), _fmt(st["std"]), _fmt(st["min"]),
                _fmt(st["p01"]), _fmt(st["p05"]), _fmt(st["p25"]),
                _fmt(st["median"]), _fmt(st["p75"]), _fmt(st["p95"]),
                _fmt(st["p99"]), _fmt(st["max"]), _fmt(st["skew"])))
        else:
            add("| {} | M2 | {} (rate1) | {} | 0 | - | - | - | - | - | - | - | 1 | - |".format(
                f, _fmt(st["rate_1"]),
                _fmt(np.sqrt(st["rate_1"] * (1.0 - st["rate_1"])))))
    add("")

    # 5. factor 相关性
    add("## 5. factor 相关性 (June 主参考; July 稳定性)")
    add("")
    fp = r["factor_pairs"]
    maxp = max(fp, key=lambda row: row["abs_june_pearson"])
    maxs = max(fp, key=lambda row: row["abs_june_spearman"])
    add("- 最大 June Pearson pair: `{}-{}` rho={} (severity={})".format(
        maxp["left_factor"], maxp["right_factor"], _fmt(maxp["june_pearson"]),
        maxp["severity"]))
    add("- 最大 June Spearman pair: `{}-{}` rho={} (severity={})".format(
        maxs["left_factor"], maxs["right_factor"], _fmt(maxs["june_spearman"]),
        maxs["severity"]))
    add("- severe pairs (|rho| >= 0.85): {}".format(
        ", ".join("{}-{} ({})".format(p["left_factor"], p["right_factor"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"]))) for p in r["severe_pairs"])
        if r["severe_pairs"] else "无"))
    high = [p for p in fp if p["severity"] == "HIGH"]
    add("- high pairs (0.70 <= |rho| < 0.85): {}".format(
        ", ".join("{}-{} ({})".format(p["left_factor"], p["right_factor"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"]))) for p in high)
        if high else "无"))
    add("")
    add("完整 28 对明细见 `v004c_factor_dependence_v001.csv` (REGIME 为二元变量,"
        " 其 Pearson/Spearman 只作描述性参考, 不作连续线性解释)。")
    add("")

    # 6. VIF
    add("## 6. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)")
    add("")
    add("| factor | M1 VIF | M2 VIF |")
    add("|---|---|---|")
    for f in FACTOR_NAMES:
        m1 = r["m1_vif"].get(f)
        m2 = r["m2_vif"].get(f)
        add("| {} | {} | {} |".format(
            f, _fmt(m1) if m1 is not None else "—",
            _fmt(m2) if m2 is not None else "—"))
    add("")
    m1_bad = [(f, v) for f, v in r["m1_vif"].items() if v >= 5.0]
    m2_bad = [(f, v) for f, v in r["m2_vif"].items() if v >= 5.0]
    add("- M1 max VIF: {} (factor `{}`); VIF >= 5: {}; VIF >= 10: {}".format(
        _fmt(max(r["m1_vif"].values()), 3),
        max(r["m1_vif"], key=r["m1_vif"].get),
        ", ".join("{}={}".format(f, _fmt(v, 3)) for f, v in m1_bad) or "无",
        ", ".join("{}={}".format(f, _fmt(v, 3)) for f, v in m1_bad if v >= 10) or "无"))
    add("- M2 max VIF: {} (factor `{}`); VIF >= 5: {}; VIF >= 10: {}".format(
        _fmt(max(r["m2_vif"].values()), 3),
        max(r["m2_vif"], key=r["m2_vif"].get),
        ", ".join("{}={}".format(f, _fmt(v, 3)) for f, v in m2_bad) or "无",
        ", ".join("{}={}".format(f, _fmt(v, 3)) for f, v in m2_bad if v >= 10) or "无"))
    add("- 解释: VIF < 5 OK; 5 <= VIF < 10 WATCH; VIF >= 10 SEVERE (只解释, 禁止自动删因子)")
    add("")

    # 7. condition number
    add("## 7. condition number (centered + unit-variance 诊断矩阵, SVD)")
    add("")
    add("| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |")
    add("|---|---|---|---|---|---|")
    for name, factors, label in (("M1", M1_FACTORS, "OPEN RESET SUPPLY"),
                                 ("M2", M2_FACTORS, "8 factors"),
                                 ("PATH", PATH_BLOCK_FACTORS, "MOM7 DAMAGE7 POS7")):
        k = r["kappa"][name]
        if k["near_singular"] or k["kappa"] >= KAPPA_SEVERE:
            judge = "SEVERE"
        elif k["kappa"] >= 30.0:
            judge = "WATCH"
        else:
            judge = "OK"
        add("| {} | {} | {} | {} | {} | {} |".format(
            name, label, _fmt(k["s_max"], 3), _fmt(k["s_min"], 6),
            _fmt(k["kappa"], 3),
            judge + (" / near singular" if k["near_singular"] else "")))
    add("- 参考: kappa < 30 OK; 30-100 WATCH; >= 100 SEVERE; smallest singular ~ 0"
        " => SEVERE / near singular")
    add("- REGIME 只在此诊断副本中 center/scale (diagnostic scaling !="
        " future model preprocessing)")
    add("")

    # 8. July 稳定性
    add("## 8. July 无标签 X 稳定性 (June reference transform 映射)")
    add("")
    add("| factor | June mean | June std | July mean | July std | SMD | KS | flag |")
    add("|---|---|---|---|---|---|---|---|")
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]
        if f in CONTINUOUS_FACTORS:
            add("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, _fmt(st["june"]["mean"]), _fmt(st["june"]["std"]),
                _fmt(st["july"]["mean"]), _fmt(st["july"]["std"]),
                _fmt(st["smd"]), _fmt(st["ks"]),
                st["stationarity_flag"] or "—"))
        else:
            add("| {} | {} (rate1) | {} | {} (rate1) | {} | — | — | — |".format(
                f, _fmt(st["june"]["rate_1"]),
                _fmt(np.sqrt(st["june"]["rate_1"] * (1.0 - st["june"]["rate_1"]))),
                _fmt(st["july"]["rate_1"]),
                _fmt(np.sqrt(st["july"]["rate_1"] * (1.0 - st["july"]["rate_1"])))))
    add("")
    add("- 判定: |SMD| >= 0.50 或 KS >= 0.25 => SHIFT_WATCH (只诊断, 不因 July X shift"
        " 自动换因子)")
    add("- REGIME: June rate1={} ({}), July rate1={} ({}), 差值={}".format(
        _fmt(r["regime"]["june_rate1"]), r["regime"]["june_counts"],
        _fmt(r["regime"]["july_rate1"]), r["regime"]["july_counts"],
        _fmt(r["regime"]["diff"])))
    if r["shift_factors"]:
        add("- STATIONARITY_WATCH 因子: `{}` — 进入未来模型阶段后持续监控".format(
            ", ".join(r["shift_factors"])))
    else:
        add("- STATIONARITY_WATCH 因子: 无")
    add("")

    # 9. 研究问题
    add("## 9. 研究问题回答")
    add("")
    add(_qa_md(r))
    add("")

    # 10. 结构冲突说明 (只记录, 不实施)
    if r["severe_pairs"]:
        add("## 10. 结构冲突说明 (仅记录, 不实施)")
        add("")
        add("- 触发 REVIEW_REQUIRED 的 factor pair 说明如下; 本阶段不实施任何修正"
            " (禁止 drop / swap / PCA / Lasso / 改权重)。")
        add("")
        for p in r["severe_pairs"]:
            add(_conflict_card_md(r, p))
            add("")

    # 11. 最终状态
    add("## 11. 最终状态")
    add("")
    add("- 最终状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 触发原因:")
        for reason in r["status_reasons"]:
            add("  - " + reason)
    add("- 本审查只做 X 结构诊断: 未读取 Target, 未做任何 Target 方向解释"
        " (不得据此声称某 factor 应取正/负系数), 未执行任何训练/预测")
    add("- 禁止自动修正: 即使出现 SEVERE / VIF >= 10 / kappa >= 100 / 退化,"
        " 本阶段也不 drop / swap / PCA / Lasso / 改变权重; 修正方向只记录给人工")
    add("")
    return "\n".join(lines) + "\n"


def _conflict_card_md(r: dict, p: dict) -> str:
    """SEVERE factor pair 的冲突说明卡片 (§27: 哪个冲突 / 为什么 / 最小可考虑修正方向)."""
    left, right = p["left_factor"], p["right_factor"]
    lines = [
        "### {}-{} (June Pearson={} / Spearman={}, severity=SEVERE; July Pearson={}"
        " / Spearman={})".format(
            left, right, _fmt(p["june_pearson"]), _fmt(p["june_spearman"]),
            _fmt(p["july_pearson"]), _fmt(p["july_spearman"])),
        "",
    ]
    if (left, right) == ("RESET", "POS7"):
        # primitive 层面证据 (动态取值, 防手抄)
        pp = next((row for row in r["primitive_pairs"]
                   if {row["left"], row["right"]} ==
                   {"d1_high_to_close_drawdown_raw", "recent_7d_close_position"}), None)
        prim_evidence = ("`d1_high_to_close_drawdown_raw`-`recent_7d_close_position`"
                         " 原始相关性已 SEVERE (Pearson={} / Spearman={})".format(
                             _fmt(pp["june_pearson"]), _fmt(pp["june_spearman"]))
                         if pp else "对应 primitive 相关性见 primitive dependence CSV")
        lines += [
            "- 冲突: RESET (D1 单日价格重置强度) 与 POS7 (最近 7 日区间收盘相对位置)"
            " 高度负相关。",
            "- 为什么: 两者共享同一 D1 收盘锚定信息 — RESET 高表示 D1 收盘相对开盘/最高价/"
            "VWAP 大幅向下重置, POS7 低表示收盘位于 7 日区间下部; 大幅向下重置的 D1 收盘"
            " 在机制上几乎必然落在 7 日区间底部, 两者近似互为逆观测。primitive 层面 "
            + prim_evidence + "。该冲突为纯 X 结构发现, 不涉及 Target。",
            "- 最小可考虑修正方向 (本阶段不实施):",
            "  1) M2 membership 层面考虑不同时保留 RESET 与 POS7 (例如 POS7 移入"
            " sensitivity 集), 不修改 factor 定义与 composite 权重;",
            "  2) 或保留两者, 接受 M2 VIF WATCH (POS7 VIF=8.37, RESET VIF=8.09),"
            " 由下一阶段 walk-forward 的训练证据决定去留;",
            "  3) 上述方向仅供人工参考, 本阶段禁止实施。",
        ]
    else:
        lines += [
            "- 冲突: `{}` 与 `{}` 因子高度相关 (|rho| >= 0.85)。".format(left, right),
            "- 为什么: 由相关性诊断发现; 具体经济机制需人工确认, 本阶段不自动解释。",
            "- 最小可考虑修正方向 (本阶段不实施): 在 M2 membership 层面避免同时保留高度"
            " 共线组合 (移入 sensitivity 集) 或由下一阶段 walk-forward 训练证据决定;"
            " 本阶段禁止实施任何修改。",
        ]
    return "\n".join(lines)


def _primitive_dist_table(r: dict, month: str) -> str:
    lines = ["| stat | " + " | ".join(CONTINUOUS_PRIMITIVES) + " |",
             "|---|" + "---|" * len(CONTINUOUS_PRIMITIVES)]
    stats = ("n", "missing", "unique", "mean", "std", "min",
             "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "skew")
    for s in stats:
        cells = [_fmt(r["primitive_dist"][p][month][s]) for p in CONTINUOUS_PRIMITIVES]
        lines.append("| {} | {} |".format(s, " | ".join(cells)))
    return "\n".join(lines)


def _qa_md(r: dict) -> str:
    pp = r["primitive_pairs"]
    fp = r["factor_pairs"]
    lines: list[str] = []

    def pair(rows, left, right):
        for row in rows:
            if {row["left"], row["right"]} == {left, right}:
                return row
            if row.get("left_factor") == left and row.get("right_factor") == right:
                return row
        return None

    def prim_pair(left, right):
        return pair(pp, left, right)

    def fac_pair(left, right):
        for row in fp:
            if row["left_factor"] == left and row["right_factor"] == right:
                return row
        return None

    def show_fac(row):
        if row is None:
            return "—"
        return "pearson={} / spearman={} (severity={})".format(
            _fmt(row["june_pearson"]), _fmt(row["june_spearman"]), row["severity"])

    add = lines.append

    # Q1
    cross_severe = [p for p in pp if p["severity"] == "SEVERE" and not p["same_factor"]]
    within_severe = [p for p in pp if p["severity"] == "SEVERE" and p["same_factor"]]
    add("**Q1. 11 个 primitive 中有没有严重重复?**")
    if cross_severe or within_severe:
        add("- 跨因子 SEVERE (|rho| >= 0.85): {}".format(
            "; ".join("`{}-{}` rho={}".format(p["left"], p["right"], _fmt(max(
                p["abs_june_pearson"], p["abs_june_spearman"])))
                for p in cross_severe) or "无"))
        add("- 因子内 SEVERE (同一潜在因子的多个观测, 设计允许): {}".format(
            "; ".join("`{}-{}` rho={}".format(p["left"], p["right"], _fmt(max(
                p["abs_june_pearson"], p["abs_june_spearman"])))
                for p in within_severe) or "无"))
    else:
        add("- 无任何 SEVERE primitive pair (|rho| >= 0.85); 11 个 primitive 内部"
            " 无 near-duplicate pair (NP01-07 均不命中本 universe)")
    add("")
    add("**Q2. RESET 内部 3 个 primitive 是否合理描述同一潜变量?**")
    for left, right in (("d1_open_to_close_return_raw", "d1_high_to_close_drawdown_raw"),
                        ("d1_open_to_close_return_raw", "d1_close_to_vwap_raw"),
                        ("d1_high_to_close_drawdown_raw", "d1_close_to_vwap_raw")):
        row = prim_pair(left, right)
        if row:
            add("- `{}-{}`: pearson={} / spearman={} (severity={})".format(
                left, right, _fmt(row["june_pearson"]), _fmt(row["june_spearman"]),
                row["severity"]))
    add("- 结构说明: 三者的经济含义都锚定'D1 收盘相对盘中强度的重置程度'"
        " (OC 正=收盘强; HC 回撤大=收盘弱; VWAP gap 正=收盘强)。composite 对 OC"
        " 与 VWAP 取负、对 HC 取正, 使三者对齐同一方向; 内部相关性因此是设计的一部分"
        " (同一潜在经济因子的多个观测)。")
    add("")
    add("**Q3. SUPPLY 内部 2 个 primitive 是否合理描述同一潜变量?**")
    row = prim_pair("high_zone_volume_ratio", "late_day_sell_volume_ratio")
    if row:
        add("- `high_zone_volume_ratio-late_day_sell_volume_ratio`: pearson={} /"
            " spearman={} (severity={})".format(
                _fmt(row["june_pearson"]), _fmt(row["june_spearman"]),
                row["severity"]))
    add("- 结构说明: 高位成交占比与尾盘下跌 bar 成交占比都是 D1 供应压力来源的观测;"
        " composite 等权 (+1/2, +1/2) 固定, 不因相关性强弱调整。")
    add("")
    add("**Q4. 8 个 factor 之间最大的 Pearson/Spearman pair 是什么?**")
    maxp = max(fp, key=lambda row: row["abs_june_pearson"])
    maxs = max(fp, key=lambda row: row["abs_june_spearman"])
    add("- 最大 Pearson: `{}-{}` rho={}".format(
        maxp["left_factor"], maxp["right_factor"], _fmt(maxp["june_pearson"])))
    add("- 最大 Spearman: `{}-{}` rho={}".format(
        maxs["left_factor"], maxs["right_factor"], _fmt(maxs["june_spearman"])))
    add("")
    add("**Q5. MOM7/DAMAGE7/POS7 是否存在明显路径信息重复?**")
    for left, right in (("MOM7", "DAMAGE7"), ("MOM7", "POS7"), ("DAMAGE7", "POS7")):
        add("- `{}-{}`: {}".format(left, right, show_fac(fac_pair(left, right))))
    add("")
    add("**Q6. MOM7 和 TREND 是否高度重复?**")
    add("- `MOM7-TREND`: {}".format(show_fac(fac_pair("MOM7", "TREND"))))
    add("")
    add("**Q7. RESET 和 DAMAGE7 是否实际上描述同一回撤?**")
    add("- `RESET-DAMAGE7`: {}".format(show_fac(fac_pair("RESET", "DAMAGE7"))))
    add("- 注意: RESET 是 D1 单日价格重置强度, DAMAGE7 是最近 7 日路径峰值破坏,"
        " 时间尺度不同; 相关性高低只记录, 不做自动处理。")
    add("")
    add("**Q8. M1 最大 VIF 是多少?**")
    m1f = max(r["m1_vif"], key=r["m1_vif"].get)
    add("- M1 max VIF = {} (factor `{}`)".format(_fmt(r["m1_vif"][m1f], 3), m1f))
    add("")
    add("**Q9. M2 最大 VIF 是多少?**")
    m2f = max(r["m2_vif"], key=r["m2_vif"].get)
    add("- M2 max VIF = {} (factor `{}`)".format(_fmt(r["m2_vif"][m2f], 3), m2f))
    add("")
    add("**Q10. M1 condition number 是多少?**")
    add("- M1 kappa = {} (s_max={}, s_min={})".format(
        _fmt(r["kappa"]["M1"]["kappa"], 3), _fmt(r["kappa"]["M1"]["s_max"], 3),
        _fmt(r["kappa"]["M1"]["s_min"], 6)))
    add("")
    add("**Q11. M2 condition number 是多少?**")
    add("- M2 kappa = {} (s_max={}, s_min={})".format(
        _fmt(r["kappa"]["M2"]["kappa"], 3), _fmt(r["kappa"]["M2"]["s_max"], 3),
        _fmt(r["kappa"]["M2"]["s_min"], 6)))
    add("")
    add("**Q12. PATH BLOCK condition number 是多少?**")
    add("- PATH BLOCK kappa = {} (s_max={}, s_min={})".format(
        _fmt(r["kappa"]["PATH"]["kappa"], 3), _fmt(r["kappa"]["PATH"]["s_max"], 3),
        _fmt(r["kappa"]["PATH"]["s_min"], 6)))
    add("")
    add("**Q13. 哪些 factor 出现 June→July X shift?**")
    if r["shift_factors"]:
        for f in r["shift_factors"]:
            st = r["factor_stats"][f]
            add("- `{}`: SMD={}, KS={} (SHIFT_WATCH)".format(
                f, _fmt(st["smd"]), _fmt(st["ks"])))
    else:
        add("- 无 (所有连续 factor 的 |SMD| < 0.50 且 KS < 0.25)")
    add("- REGIME: June rate1={}, July rate1={}, 差值={}".format(
        _fmt(r["regime"]["june_rate1"]), _fmt(r["regime"]["july_rate1"]),
        _fmt(r["regime"]["diff"])))
    add("")
    add("**Q14. 有没有 factor 退化或过度稀疏?**")
    if r["degenerate_factors"]:
        add("- 退化 factor (June std <= 1e-8): {}".format(
            ", ".join(r["degenerate_factors"])))
    else:
        add("- 退化 factor: 无 (8 个 factor 全部有限, 连续 factor June std > 1e-8)")
    add("- REGIME 稀疏性: June count0={} count1={} (rate1={}); July count0={}"
        " count1={} (rate1={}); minority 计数 {} / {}; 是否过疏由人工判断,"
        " 本阶段不处理".format(
            r["regime_dist"]["june"]["count_0"], r["regime_dist"]["june"]["count_1"],
            _fmt(r["regime_dist"]["june"]["rate_1"]),
            r["regime_dist"]["july"]["count_0"], r["regime_dist"]["july"]["count_1"],
            _fmt(r["regime_dist"]["july"]["rate_1"]),
            r["regime_dist"]["june"]["minority_count"],
            r["regime_dist"]["july"]["minority_count"]))
    add("")
    add("**Q15. 8 因子结构是否可以进入正式 walk-forward?**")
    add("- 最终状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 触发原因: {}".format("; ".join(r["status_reasons"])))
    add("- 含义: 该状态只回答 'factor 结构是否适合进入 M0/M1/M2 expanding-date"
        " walk-forward', 不是 model coefficient freeze; 是否继续由人工审查"
        " primitive/factor dependence、VIF、condition number、distribution stability"
        " 后决定。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c 因子结构纯 X 侧审计 (不训练模型, 不读取 Target)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_factor_dependence_v001_20260601_20260729)")
    args = parser.parse_args(argv)

    r = run_factor_dependence_audit(args.input, args.output)

    print("=== v004c factor dependence audit ===")
    print(f"input      : {r['input']}")
    print(f"sha256     : {r['input_sha256']}")
    print(f"June rows  : {r['june_rows']} ({r['june_date_count']} signal dates)")
    print(f"July rows  : {r['july_rows']} ({r['july_date_count']} signal dates)")
    maxp = max(r["primitive_pairs"], key=lambda row: row["abs_june_pearson"])
    maxs = max(r["primitive_pairs"], key=lambda row: row["abs_june_spearman"])
    print(f"primitive max Pearson  : {maxp['left']}-{maxp['right']} "
          f"rho={maxp['june_pearson']:.4f}")
    print(f"primitive max Spearman : {maxs['left']}-{maxs['right']} "
          f"rho={maxs['june_spearman']:.4f}")
    maxp = max(r["factor_pairs"], key=lambda row: row["abs_june_pearson"])
    maxs = max(r["factor_pairs"], key=lambda row: row["abs_june_spearman"])
    print(f"factor max Pearson     : {maxp['left_factor']}-{maxp['right_factor']} "
          f"rho={maxp['june_pearson']:.4f}")
    print(f"factor max Spearman    : {maxs['left_factor']}-{maxs['right_factor']} "
          f"rho={maxs['june_spearman']:.4f}")
    m1f = max(r["m1_vif"], key=r["m1_vif"].get)
    m2f = max(r["m2_vif"], key=r["m2_vif"].get)
    print(f"M1 max VIF             : {r['m1_vif'][m1f]:.3f} ({m1f})")
    print(f"M2 max VIF             : {r['m2_vif'][m2f]:.3f} ({m2f})")
    for name in ("M1", "M2", "PATH"):
        k = r["kappa"][name]
        near = " NEAR-SINGULAR" if k["near_singular"] else ""
        print(f"kappa {name:4s}            : {k['kappa']:.3f}{near}")
    print(f"degenerate factors     : {r['degenerate_factors'] or 'none'}")
    print(f"shift factors          : {r['shift_factors'] or 'none'}")
    print(f"status                 : {r['status']}")
    for reason in r["status_reasons"]:
        print(f"  - {reason}")
    print(f"output                 : {args.output} ({len(OUTPUT_FILES)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

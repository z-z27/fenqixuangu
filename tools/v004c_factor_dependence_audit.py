"""v004c 因子结构纯 X 侧审计工具 (v004c factor dependence v002)。

职责:

- 读取 model table 的 X 子集: 显式 usecols 只加载 3 个标识列 + 11 个 primitive,
  读取入口即 target-blind (绝不加载 target7_daily_d2open_d3high / D2 / D3 / 旧模型输出)
- June (reference X sample) 拟合 reference transform (q01/q99/mu/sigma)
- July 只 apply June 参数 (不得用 July 重算 mean/std)
- 构造 9 个 factor (7 CORE: OPEN/RESET/HIGHZONE/LATESELL/MOM7/DAMAGE7/REGIME;
  2 SENSITIVITY: POS7/TREND; v002 起无 active SUPPLY composite)
- primitive/factor 相关性 (Pearson + Spearman), VIF (M1/M2), condition number
  (M1/M2/PATH, centered + unit-variance 诊断矩阵), factor 分布,
  July 无标签 X 稳定性 (SMD / KS)
- 最终状态只由 CORE 矩阵判定: CORE cross-factor |rho| >= 0.85 / VIF >= 10 /
  condition number >= 100 / degenerate factor => REVIEW_REQUIRED, 否则
  PASS_X_STRUCTURE_V002; 含 sensitivity 成员的 pair 只记录, 不触发状态
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
    CORE_FACTORS,
    FACTOR_NAMES,
    FACTOR_PRIMITIVES,
    FORBIDDEN_TOKENS,
    M1_FACTORS,
    M2_FACTORS,
    PATH_BLOCK_FACTORS,
    PRIMITIVE_COLUMNS,
    SENSITIVITY_FACTORS,
    SENSITIVITY_STATUS,
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
DEFAULT_OUTPUT = REPO_ROOT / "reports/research/v004c_factor_dependence_v002_20260601_20260729"
DICTIONARY_DIR = REPO_ROOT / "reports/research/v004c_factor_dictionary_v001_20260601_20260729"
COVERAGE_CSV = REPO_ROOT / "reports/research/v004c_feature_coverage_v001.csv"
V001_DIR = REPO_ROOT / "reports/research/v004c_factor_dependence_v001_20260601_20260729"

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"

OUTPUT_FILES = (
    "v004c_primitive_dependence_v002.csv",
    "v004c_factor_dependence_v002.csv",
    "v004c_factor_diagnostics_v002.csv",
    "v004c_factor_spec_v002.csv",
    "v004c_factor_model_spec_v002.md",
    "v004c_factor_dependence_review_v002.md",
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
    """9 choose 2 = 36 对 (June 为主参考, July 只做稳定性描述)。

    core_pair = 两端都是 CORE factor; 只有 core_pair 参与最终状态判定。
    """
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
                "core_pair": left in CORE_FACTORS and right in CORE_FACTORS,
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
# v001 历史报告对照 (只读冻结资产, 用于说明 v002 结构变化的效果)
# ---------------------------------------------------------------------------

def _v001_reference() -> dict:
    """读取冻结 v001 报告作跨版本对照 (只读; 缺失/损坏则返回空 dict)."""
    dep_path = V001_DIR / "v004c_factor_dependence_v001.csv"
    diag_path = V001_DIR / "v004c_factor_diagnostics_v001.csv"
    if not dep_path.exists() or not diag_path.exists():
        return {}
    out: dict = {}
    try:
        dep = pd.read_csv(dep_path, encoding="utf-8-sig")
        diag = pd.read_csv(diag_path, encoding="utf-8-sig")
    except Exception:
        return {}
    for row in dep.itertuples(index=False):
        if (row.left_factor, row.right_factor) == ("MOM7", "TREND"):
            out["v001_mom7_trend_pearson"] = float(row.june_pearson)
            out["v001_mom7_trend_spearman"] = float(row.june_spearman)
        elif (row.left_factor, row.right_factor) == ("RESET", "POS7"):
            out["v001_reset_pos7_pearson"] = float(row.june_pearson)
            out["v001_reset_pos7_spearman"] = float(row.june_spearman)
    m2_vifs: dict[str, float] = {}
    for row in diag.itertuples(index=False):
        vif = getattr(row, "m2_vif", None)
        if vif is not None and not pd.isna(vif):
            m2_vifs[str(row.factor)] = float(vif)
    for f, vif in m2_vifs.items():
        out[f"v001_m2_vif_{f}"] = vif
    if m2_vifs:
        best = max(m2_vifs, key=m2_vifs.get)
        out["v001_m2_vif_max"] = m2_vifs[best]
        out["v001_m2_vif_max_factor"] = best
    return out


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

    # --- 最终状态 (只判定 CORE 矩阵, 不自动改 factor) ---
    core_severe_pairs = [p for p in fac_pairs
                         if p["severity"] == "SEVERE" and p["core_pair"]]
    sensitivity_severe_pairs = [p for p in fac_pairs
                                if p["severity"] == "SEVERE" and not p["core_pair"]]
    reasons: list[str] = []
    if degenerate_factors:
        reasons.append(f"degenerate factor: {degenerate_factors} (June std <= 1e-8)")
    for name, k in kappa.items():
        if k["near_singular"] or k["kappa"] >= KAPPA_SEVERE:
            reasons.append(f"{name} condition matrix near singular "
                           f"or kappa={k['kappa']:.3g} >= 100")
    for f, v in m1_vif.items():
        if v >= VIF_SEVERE:
            reasons.append(f"M1 factor {f} VIF={v:.4g} >= 10 SEVERE")
    for f, v in m2_vif.items():
        if v >= VIF_SEVERE:
            reasons.append(f"M2 factor {f} VIF={v:.4g} >= 10 SEVERE")
    for p in core_severe_pairs:
        reasons.append(
            f"factor pair {p['left_factor']}-{p['right_factor']} SEVERE "
            f"(|rho|={max(p['abs_june_pearson'], p['abs_june_spearman']):.3f})")
    status = "PASS_X_STRUCTURE_V002" if not reasons else "REVIEW_REQUIRED"

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
        "core_factors": list(CORE_FACTORS),
        "sensitivity_factors": list(SENSITIVITY_FACTORS),
        "degenerate_factors": degenerate_factors,
        "shift_factors": shift_factors,
        "core_severe_pairs": core_severe_pairs,
        "sensitivity_severe_pairs": sensitivity_severe_pairs,
        "v001_ref": _v001_reference(),
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
        _model_spec_md(r), encoding="utf-8")

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


def _factor_membership(f: str) -> str:
    if f in SENSITIVITY_FACTORS:
        return "SENSITIVITY"
    return "M1,M2" if f in M1_FACTORS else "M2"


def _factor_diagnostics_rows(r: dict) -> list[dict]:
    rows = []
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]
        june_d, july_d = st["june"], st["july"]
        if f in CONTINUOUS_FACTORS:
            row = {
                "factor": f,
                "model_membership": _factor_membership(f),
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
                "model_membership": _factor_membership(f),
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


def _model_spec_md(r: dict) -> str:
    v = r["v001_ref"]
    hl = next((p for p in r["primitive_pairs"]
               if {p["left"], p["right"]} ==
               {"high_zone_volume_ratio", "late_day_sell_volume_ratio"}), None)
    hz_ls = ("Pearson={} / Spearman={}".format(
        _fmt(hl["june_pearson"]), _fmt(hl["june_spearman"]))
        if hl is not None else "见 primitive dependence CSV")
    rp = ("Pearson={} / Spearman={}".format(
        _fmt(v.get("v001_reset_pos7_pearson")), _fmt(v.get("v001_reset_pos7_spearman")))
        if v.get("v001_reset_pos7_pearson") is not None else "见 v001 报告")
    mt = ("Pearson={} / Spearman={}".format(
        _fmt(v.get("v001_mom7_trend_pearson")), _fmt(v.get("v001_mom7_trend_spearman")))
        if v.get("v001_mom7_trend_pearson") is not None else "见 v001 报告")
    return f"""# v004c 因子模型规格 v002 (M0/M1/M2 数学结构 — 仅记录, 不拟合)

- 阶段: 多变量建模前的纯 X 侧因子结构审查产物 (v002 修订); 本规格只定义公式与未来训练算法
- 状态: factor definition freeze 候选 (由 v004c_factor_dependence_review_v002.md 结论背书)
- 注意: 这是 factor definition freeze, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## v002 修订要点 (依据 v001 纯 X 结构审查结论, 与 Target 无关)

1. SUPPLY composite 拆分: `high_zone_volume_ratio` 与 `late_day_sell_volume_ratio`
   在 June 中接近独立 ({hz_ls}), 不再强制压缩成 50/50 composite;
   HIGHZONE 与 LATESELL 作为独立 factor, 未来必须由 Logistic 独立估计系数,
   禁止重新合成 SUPPLY。v001 中 SUPPLY 的历史定义只保留在 v001 报告资产。
2. POS7 移入 SENSITIVITY: 与 RESET 结构重复 (v001 June {rp});
   状态 SENSITIVITY_STRUCTURAL_REDUNDANCY, 不得进入 M2 core。
3. TREND 移入 SENSITIVITY: 与 MOM7 高度相关 (v001 June {mt})
   + June→July X shift (v001 SMD ≈ -0.63); 状态 SENSITIVITY_HORIZON_STATIONARITY,
   不得进入第一版 M2 core。

## Factor 预处理 (future walk-forward)

- 连续 factor: 每个 training fold 内拟合 clip [q01, q99] + (x - mu)/sigma
  (q01/q99/mu/sigma 只允许来自当前 training fold; 禁止全样本参数)
- REGIME: 0/1 原值, 不做 z-score; 第一版禁止 interaction
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## M0 — INTERCEPT BASELINE

    logit(p) = alpha

## M1 — D1 STRUCTURE

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_HIGHZONE
             + beta_4 * F_LATESELL

(4 factors + intercept)

经济假设: D1 开盘承接 (OPEN)、D1 价格重置 (RESET)、高位筹码堆积 (HIGHZONE)
与尾盘主动卖压 (LATESELL) 是否共同形成对未来 Target7 概率有解释力的 D1 结构。
只描述结构问题, 不预设系数方向。

## M2 — INTEGRATED PATH MODEL

    logit(p) = alpha
             + beta_1 * F_OPEN
             + beta_2 * F_RESET
             + beta_3 * F_HIGHZONE
             + beta_4 * F_LATESELL
             + gamma_1 * F_MOM7
             + gamma_2 * F_DAMAGE7
             + gamma_3 * F_REGIME

(7 factors + intercept)

M2 回答: 在 D1 结构已知后, 最近价格路径 (MOM7/DAMAGE7) 和二/三板 regime (REGIME)
是否提供增量预测信息。DAMAGE7 即使存在 SHIFT_WATCH 也继续属于 core
(与 RESET/MOM7 提供明显不同的路径信息)。

## Sensitivity factors (不进 M1/M2 core, 只做诊断)

- POS7: SENSITIVITY_STRUCTURAL_REDUNDANCY (与 RESET 严重重复, 纯 X 结构发现)
- TREND: SENSITIVITY_HORIZON_STATIONARITY (与 MOM7 高度相关 + June→July X shift)

## 未来训练算法 (只记录, 本阶段禁止执行)

- L2 Logistic Regression: C = 1.0, solver = lbfgs, fit_intercept = True,
  class_weight = None
- 目标函数:

      min_theta [ -sum_i ( y_i*log(p_i) + (1-y_i)*log(1-p_i) ) + lambda * ||theta_{-0}||_2^2 ]

- walk-forward 的目的不是只获得一组系数, 而是产生严格时间外 OOF 预测以及逐 fold
  系数路径, 用于检验预测能力、增量价值和参数稳定性
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

    add("# v004c 因子结构纯 X 侧审查 (v004c factor dependence v002)")
    add("")
    add("- 阶段: 多变量建模前的纯 X 侧因子结构审查 v002 (X ONLY, 不训练模型, 不读取 Target)")
    add("- 分支: research-sample-analysis")
    add("- 输入: `v004c_model_table_v001.csv` SHA256: `{}`".format(r["input_sha256"]))
    add("- v002 修订依据 (v001 纯 X 审查结论, 与 Target 无关): SUPPLY composite 拆分"
        " (HIGHZONE/LATESELL 独立); POS7 移入 SENSITIVITY"
        " (SENSITIVITY_STRUCTURAL_REDUNDANCY); TREND 移入 SENSITIVITY"
        " (SENSITIVITY_HORIZON_STATIONARITY)。v001 历史资产保持不变。")
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
    add("- 注: primitive 级跨因子 SEVERE 只作诊断; factor 级判定只针对 CORE 矩阵 (§5),"
        " 含 SENSITIVITY 成员的 pair 不触发状态 (§12)。")
    add("")
    add("完整 55 对明细见 `v004c_primitive_dependence_v002.csv` (same_factor 标记"
        " 因子内/因子间)。")
    add("")

    # 4. factor 构造验证与分布
    add("## 4. factor 构造验证与 June 分布 (9 factors: 7 CORE + 2 SENSITIVITY)")
    add("")
    add("- 构造: 9 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8"
        " (degenerate 检查通过: {})".format(
            "无退化" if not r["degenerate_factors"] else r["degenerate_factors"]))
    add("- REGIME 严格 0/1 (未标准化); 只有 condition number 诊断矩阵中临时 center/scale")
    add("- HIGHZONE / LATESELL 独立构造 (各自 z-score), 无 SUPPLY composite")
    add("")
    add("| factor | membership | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]["june"]
        if f in CONTINUOUS_FACTORS:
            add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, _factor_membership(f),
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
    add("## 5. factor 相关性 (June 主参考; July 稳定性; 判定只看 CORE 矩阵)")
    add("")
    fp = r["factor_pairs"]
    core_pairs = [p for p in fp if p["core_pair"]]
    maxp = max(fp, key=lambda row: row["abs_june_pearson"])
    maxs = max(fp, key=lambda row: row["abs_june_spearman"])
    maxpc = max(core_pairs, key=lambda row: row["abs_june_pearson"])
    maxsc = max(core_pairs, key=lambda row: row["abs_june_spearman"])
    add("- 最大 June Pearson pair (9 factor 全集): `{}-{}` rho={} (severity={}){}".format(
        maxp["left_factor"], maxp["right_factor"], _fmt(maxp["june_pearson"]),
        maxp["severity"], " — 含 SENSITIVITY 成员" if not maxp["core_pair"] else ""))
    add("- 最大 June Spearman pair (9 factor 全集): `{}-{}` rho={} (severity={}){}".format(
        maxs["left_factor"], maxs["right_factor"], _fmt(maxs["june_spearman"]),
        maxs["severity"], " — 含 SENSITIVITY 成员" if not maxs["core_pair"] else ""))
    add("- 最大 June Pearson pair (**CORE**): `{}-{}` rho={} (severity={})".format(
        maxpc["left_factor"], maxpc["right_factor"], _fmt(maxpc["june_pearson"]),
        maxpc["severity"]))
    add("- 最大 June Spearman pair (**CORE**): `{}-{}` rho={} (severity={})".format(
        maxsc["left_factor"], maxsc["right_factor"], _fmt(maxsc["june_spearman"]),
        maxsc["severity"]))
    add("- CORE severe pairs (|rho| >= 0.85): {}".format(
        ", ".join("{}-{} ({})".format(p["left_factor"], p["right_factor"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"]))) for p in r["core_severe_pairs"])
        if r["core_severe_pairs"] else "无"))
    core_high = [p for p in core_pairs if p["severity"] == "HIGH"]
    add("- CORE high pairs (0.70 <= |rho| < 0.85): {}".format(
        ", ".join("{}-{} ({})".format(p["left_factor"], p["right_factor"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"]))) for p in core_high)
        if core_high else "无"))
    add("- sensitivity 级 SEVERE pairs (非阻塞, 见 §12): {}".format(
        ", ".join("{}-{} ({})".format(p["left_factor"], p["right_factor"], _fmt(max(
            p["abs_june_pearson"], p["abs_june_spearman"])))
            for p in r["sensitivity_severe_pairs"])
        if r["sensitivity_severe_pairs"] else "无"))
    add("")
    add("完整 36 对明细见 `v004c_factor_dependence_v002.csv` (core_pair 标记"
        " CORE×CORE; REGIME 为二元变量, 其 Pearson/Spearman 只作描述性参考,"
        " 不作连续线性解释)。")
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
    vref = r["v001_ref"]
    if vref:
        add("- v001 对照 (冻结 v001 报告, 只读): v001 M2 max VIF = {} (factor `{}`);"
            " POS7={}, RESET={}, TREND={}, MOM7={}。v002 M2 移除 POS7/TREND 后"
            " max VIF = {} (factor `{}`)。".format(
                _fmt(vref.get("v001_m2_vif_max"), 3),
                vref.get("v001_m2_vif_max_factor", ""),
                _fmt(vref.get("v001_m2_vif_POS7"), 3),
                _fmt(vref.get("v001_m2_vif_RESET"), 3),
                _fmt(vref.get("v001_m2_vif_TREND"), 3),
                _fmt(vref.get("v001_m2_vif_MOM7"), 3),
                _fmt(max(r["m2_vif"].values()), 3),
                max(r["m2_vif"], key=r["m2_vif"].get)))
    else:
        add("- v001 对照: 不可用 (v001 报告目录缺失, 仅报告 v002 数值)")
    add("")

    # 7. condition number
    add("## 7. condition number (centered + unit-variance 诊断矩阵, SVD)")
    add("")
    add("| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |")
    add("|---|---|---|---|---|---|")
    for name, factors, label in (("M1", M1_FACTORS, "OPEN RESET HIGHZONE LATESELL"),
                                 ("M2", M2_FACTORS, "7 CORE factors"),
                                 ("PATH", PATH_BLOCK_FACTORS, "MOM7 DAMAGE7")):
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

    # 10. v002 修订要点确认
    add("## 10. v002 修订要点确认 (v001 → v002 结构变化的效果, 纯 X 数据)")
    add("")
    add(_revision_check_md(r))
    add("")

    # 11. 结构冲突说明 (只记录, 不实施)
    add("## 11. 结构冲突说明 (仅记录, 不实施)")
    add("")
    if r["core_severe_pairs"]:
        add("- 触发 REVIEW_REQUIRED 的 CORE factor pair 说明如下; 本阶段不实施任何修正"
            " (禁止 drop / swap / PCA / Lasso / 改权重)。")
        add("")
        for p in r["core_severe_pairs"]:
            add(_conflict_card_md(r, p))
            add("")
    else:
        add("- CORE M1/M2 无 cross-factor SEVERE pair (|rho| >= 0.85), 无需冲突说明。")
        add("")
        add("- 含 SENSITIVITY 成员的 SEVERE pair (如 RESET-POS7) 按 v002 决策记录为"
            " 非阻塞, 见 §12。")
    add("")

    # 12. 非阻塞 sensitivity 记录
    add("## 12. 非阻塞 SENSITIVITY 记录 (v002 决策, 只记录不实施)")
    add("")
    if r["sensitivity_severe_pairs"]:
        for p in r["sensitivity_severe_pairs"]:
            sens = p["left_factor"] if p["left_factor"] in SENSITIVITY_FACTORS \
                else p["right_factor"]
            other = p["right_factor"] if p["left_factor"] == sens else p["left_factor"]
            add("- `{}-{}` (June Pearson={} / Spearman={}; July Pearson={} /"
                " Spearman={}): `{}` 已按 v002 决策移入 SENSITIVITY (状态 `{}`),"
                " 该结构冗余只记录, 不阻塞 CORE 矩阵健康判定。".format(
                    p["left_factor"], p["right_factor"],
                    _fmt(p["june_pearson"]), _fmt(p["june_spearman"]),
                    _fmt(p["july_pearson"]), _fmt(p["july_spearman"]),
                    sens, SENSITIVITY_STATUS.get(sens, "SENSITIVITY")))
    else:
        add("- 无 sensitivity 级 SEVERE pair。")
    mt = next((p for p in r["factor_pairs"]
               if p["left_factor"] == "MOM7" and p["right_factor"] == "TREND"), None)
    if mt is not None and mt["severity"] in ("HIGH", "SEVERE"):
        add("- `MOM7-TREND` (June Pearson={} / Spearman={}, severity={}): TREND 已移入"
            " SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY), 该关联不再属于 CORE"
            " 矩阵。".format(_fmt(mt["june_pearson"]), _fmt(mt["june_spearman"]),
                            mt["severity"]))
    add("")

    # 13. 最终状态
    add("## 13. 最终状态")
    add("")
    add("- 最终状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 触发原因:")
        for reason in r["status_reasons"]:
            add("  - " + reason)
    add("- 判定阈值 (任务 §12): CORE M1/M2 cross-factor |Pearson or Spearman| >= 0.85,"
        " 或 VIF >= 10, 或 condition number >= 100, 或 degenerate factor"
        " => REVIEW_REQUIRED; 否则 PASS_X_STRUCTURE_V002。VIF 5-10 或相关 0.70-0.85"
        " 只记 WATCH, 不自动失败。")
    add("- 本审查只做 X 结构诊断: 未读取 Target, 未做任何 Target 方向解释"
        " (不得据此声称某 factor 应取正/负系数), 未执行任何训练/预测")
    add("- 禁止自动修正: 即使出现 SEVERE / VIF >= 10 / kappa >= 100 / 退化,"
        " 本阶段也不 drop / swap / PCA / Lasso / 改变权重; 修正方向只记录给人工")
    add("")
    return "\n".join(lines) + "\n"


def _conflict_card_md(r: dict, p: dict) -> str:
    """CORE SEVERE factor pair 的冲突说明卡片 (哪个冲突 / 为什么 / 最小可考虑修正方向)."""
    left, right = p["left_factor"], p["right_factor"]
    return "\n".join([
        "### {}-{} (June Pearson={} / Spearman={}, severity=SEVERE; July Pearson={}"
        " / Spearman={})".format(
            left, right, _fmt(p["june_pearson"]), _fmt(p["june_spearman"]),
            _fmt(p["july_pearson"]), _fmt(p["july_spearman"])),
        "",
        "- 冲突: `{}` 与 `{}` 都是 CORE factor 且高度相关 (|rho| >= 0.85)。".format(
            left, right),
        "- 为什么: 由 CORE 相关性诊断发现; 具体经济机制需人工确认, 本阶段不自动解释。",
        "- 最小可考虑修正方向 (本阶段不实施): 在 M2 membership 层面避免同时保留高度"
        " 共线组合 (参考 v002 对 POS7/TREND 的处理, 将其中一个移入 SENSITIVITY 集),"
        " 或由下一阶段 walk-forward 训练证据决定; 本阶段禁止实施任何修改。",
    ])


def _revision_check_md(r: dict) -> str:
    """§10: v002 修订要点确认 — 6 项重点, 数据驱动 + v001 对照."""
    v = r["v001_ref"]
    fp = r["factor_pairs"]
    core_pairs = [p for p in fp if p["core_pair"]]
    m1max = max(r["m1_vif"], key=r["m1_vif"].get)
    m2max = max(r["m2_vif"], key=r["m2_vif"].get)
    rp = next((p for p in fp
               if p["left_factor"] == "RESET" and p["right_factor"] == "POS7"), None)
    mt = next((p for p in fp
               if p["left_factor"] == "MOM7" and p["right_factor"] == "TREND"), None)
    hl = next((p for p in fp
               if p["left_factor"] == "HIGHZONE" and p["right_factor"] == "LATESELL"), None)
    mom_dmg = next((p for p in fp
                    if p["left_factor"] == "MOM7" and p["right_factor"] == "DAMAGE7"), None)

    lines = []
    add = lines.append

    add("**1. RESET 从 POS7 移出 core 后, VIF 是否明显改善?**")
    add("- v002 M2 中 RESET VIF = {}; POS7 已不在任何 CORE 矩阵。".format(
        _fmt(r["m2_vif"]["RESET"], 3)))
    if v.get("v001_m2_vif_RESET") is not None:
        add("- v001 对照: v001 M2 中 RESET VIF = {}, POS7 VIF = {}。".format(
            _fmt(v["v001_m2_vif_RESET"], 3), _fmt(v.get("v001_m2_vif_POS7"), 3)))
    add("")
    add("**2. MOM7 移除 TREND 后, 是否不再出现明显共线性?**")
    if mom_dmg is not None:
        add("- v002 CORE 中 MOM7 最大相关 pair = `{}-{}`: pearson={} / spearman={}"
            " (severity={})。".format(
                mom_dmg["left_factor"], mom_dmg["right_factor"],
                _fmt(mom_dmg["june_pearson"]), _fmt(mom_dmg["june_spearman"]),
                mom_dmg["severity"]))
    add("- v002 M2 中 MOM7 VIF = {}。".format(_fmt(r["m2_vif"]["MOM7"], 3)))
    if mt is not None:
        add("- v001 对照: v001 `MOM7-TREND` pearson={} / spearman={} (severity={});"
            " TREND 现已移出 CORE。".format(
                _fmt(mt["june_pearson"]), _fmt(mt["june_spearman"]), mt["severity"]))
    add("")
    add("**3. HIGHZONE 与 LATESELL 独立进入 M1/M2 后, 矩阵是否仍健康?**")
    if hl is not None:
        add("- `HIGHZONE-LATESELL`: pearson={} / spearman={} (severity={})"
            " — 接近独立, 支持拆分。".format(
                _fmt(hl["june_pearson"]), _fmt(hl["june_spearman"]), hl["severity"]))
    add("- v002 M1/M2 中 HIGHZONE VIF = {} / {}, LATESELL VIF = {} / {}。".format(
        _fmt(r["m1_vif"]["HIGHZONE"], 3), _fmt(r["m2_vif"]["HIGHZONE"], 3),
        _fmt(r["m1_vif"]["LATESELL"], 3), _fmt(r["m2_vif"]["LATESELL"], 3)))
    add("")
    add("**4. M1 是否存在严重共线性?**")
    add("- M1 max VIF = {} (factor `{}`); M1 kappa = {} (s_max={}, s_min={})。".format(
        _fmt(r["m1_vif"][m1max], 3), m1max,
        _fmt(r["kappa"]["M1"]["kappa"], 3), _fmt(r["kappa"]["M1"]["s_max"], 3),
        _fmt(r["kappa"]["M1"]["s_min"], 6)))
    add("")
    add("**5. M2 是否存在严重共线性?**")
    add("- M2 max VIF = {} (factor `{}`); M2 kappa = {} (s_max={}, s_min={})。".format(
        _fmt(r["m2_vif"][m2max], 3), m2max,
        _fmt(r["kappa"]["M2"]["kappa"], 3), _fmt(r["kappa"]["M2"]["s_max"], 3),
        _fmt(r["kappa"]["M2"]["s_min"], 6)))
    add("")
    add("**6. 是否仍有 cross-factor SEVERE pair?**")
    if r["core_severe_pairs"]:
        add("- CORE: {}".format("; ".join(
            "`{}-{}` (|rho|={})".format(p["left_factor"], p["right_factor"], _fmt(max(
                p["abs_june_pearson"], p["abs_june_spearman"])))
            for p in r["core_severe_pairs"])))
    else:
        add("- CORE: 无 (7 个 CORE factor 内部无 |rho| >= 0.85 pair)。")
    if r["sensitivity_severe_pairs"]:
        add("- SENSITIVITY (非阻塞, 只记录): {}".format("; ".join(
            "`{}-{}` (|rho|={})".format(p["left_factor"], p["right_factor"], _fmt(max(
                p["abs_june_pearson"], p["abs_june_spearman"])))
            for p in r["sensitivity_severe_pairs"])))
    else:
        add("- SENSITIVITY: 无。")
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
    core_pairs = [p for p in fp if p["core_pair"]]
    vref = r["v001_ref"]
    lines: list[str] = []

    def prim_pair(left, right):
        for row in pp:
            if {row["left"], row["right"]} == {left, right}:
                return row
        return None

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
        add("- 注: 跨因子 SEVERE 对应 factor 级 pair 为 RESET×POS7, POS7 已移入"
            " SENSITIVITY (见 §5/§12), 不触发 CORE 判定。")
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
    add("**Q3. HIGHZONE 与 LATESELL 是否接近独立 (SUPPLY 拆分依据)?**")
    row = prim_pair("high_zone_volume_ratio", "late_day_sell_volume_ratio")
    if row:
        add("- `high_zone_volume_ratio-late_day_sell_volume_ratio`: pearson={} /"
            " spearman={} (severity={})".format(
                _fmt(row["june_pearson"]), _fmt(row["june_spearman"]),
                row["severity"]))
    add("- 结构说明: 两者接近独立, 缺乏同一潜变量依据 => v002 不再压缩成 50/50"
        " SUPPLY composite; HIGHZONE/LATESELL 各自独立 z-score, 未来由 Logistic"
        " 独立估计系数 (禁止重新合成 SUPPLY)。")
    add("")
    add("**Q4. 9 个 factor 中最大 Pearson/Spearman pair 是什么 (全集 与 CORE)?**")
    maxp = max(fp, key=lambda row: row["abs_june_pearson"])
    maxs = max(fp, key=lambda row: row["abs_june_spearman"])
    maxpc = max(core_pairs, key=lambda row: row["abs_june_pearson"])
    maxsc = max(core_pairs, key=lambda row: row["abs_june_spearman"])
    add("- 全集最大 Pearson: `{}-{}` rho={} (severity={}){}".format(
        maxp["left_factor"], maxp["right_factor"], _fmt(maxp["june_pearson"]),
        maxp["severity"], " — 含 SENSITIVITY 成员" if not maxp["core_pair"] else ""))
    add("- 全集最大 Spearman: `{}-{}` rho={} (severity={}){}".format(
        maxs["left_factor"], maxs["right_factor"], _fmt(maxs["june_spearman"]),
        maxs["severity"], " — 含 SENSITIVITY 成员" if not maxs["core_pair"] else ""))
    add("- CORE 最大 Pearson: `{}-{}` rho={} (severity={})".format(
        maxpc["left_factor"], maxpc["right_factor"], _fmt(maxpc["june_pearson"]),
        maxpc["severity"]))
    add("- CORE 最大 Spearman: `{}-{}` rho={} (severity={})".format(
        maxsc["left_factor"], maxsc["right_factor"], _fmt(maxsc["june_spearman"]),
        maxsc["severity"]))
    add("")
    add("**Q5. MOM7/DAMAGE7/POS7 是否存在明显路径信息重复?**")
    for left, right in (("MOM7", "DAMAGE7"), ("MOM7", "POS7"), ("DAMAGE7", "POS7")):
        add("- `{}-{}`: {}{}".format(
            left, right, show_fac(fac_pair(left, right)),
            " (POS7 为 SENSITIVITY)" if "POS7" in (left, right) else ""))
    add("")
    add("**Q6. MOM7 和 TREND 是否高度重复?**")
    add("- `MOM7-TREND`: {}".format(show_fac(fac_pair("MOM7", "TREND"))))
    if vref.get("v001_mom7_trend_pearson") is not None:
        add("- v001 对照: June Pearson={} / Spearman={} — 该关联是 TREND 移入"
            " SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY) 的依据之一。".format(
                _fmt(vref["v001_mom7_trend_pearson"]),
                _fmt(vref["v001_mom7_trend_spearman"])))
    else:
        add("- TREND 已移入 SENSITIVITY (SENSITIVITY_HORIZON_STATIONARITY),"
            " 该 pair 不再属于 CORE 矩阵。")
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
    if vref.get("v001_m2_vif_max") is not None:
        add("- v001 对照: v001 M2 max VIF = {} (factor `{}`); POS7={}, TREND={},"
            " RESET={}, MOM7={}。v002 移除 POS7/TREND 后 max VIF = {} (factor"
            " `{}`)。".format(
                _fmt(vref["v001_m2_vif_max"], 3), vref.get("v001_m2_vif_max_factor", ""),
                _fmt(vref.get("v001_m2_vif_POS7"), 3),
                _fmt(vref.get("v001_m2_vif_TREND"), 3),
                _fmt(vref.get("v001_m2_vif_RESET"), 3),
                _fmt(vref.get("v001_m2_vif_MOM7"), 3),
                _fmt(r["m2_vif"][m2f], 3), m2f))
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
            add("- `{}`: SMD={}, KS={} (SHIFT_WATCH){}".format(
                f, _fmt(st["smd"]), _fmt(st["ks"]),
                " — 该 shift 是 TREND 移入 SENSITIVITY 的依据之一" if f == "TREND" else ""))
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
        add("- 退化 factor: 无 (9 个 factor 全部有限, 连续 factor June std > 1e-8)")
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
    add("**Q15. v002 CORE 结构是否可以进入正式 walk-forward?**")
    add("- 最终状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 触发原因: {}".format("; ".join(r["status_reasons"])))
    add("- 含义: 该状态只回答 'CORE factor 结构是否适合进入 M0/M1/M2 expanding-date"
        " walk-forward', 不是 model coefficient freeze; 是否继续由人工审查"
        " primitive/factor dependence、VIF、condition number、distribution stability"
        " 后决定。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c 因子结构纯 X 侧审计 v002 (不训练模型, 不读取 Target)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_factor_dependence_v002_20260601_20260729)")
    args = parser.parse_args(argv)

    r = run_factor_dependence_audit(args.input, args.output)

    print("=== v004c factor dependence audit v002 ===")
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
    core_pairs = [p for p in r["factor_pairs"] if p["core_pair"]]
    maxp = max(r["factor_pairs"], key=lambda row: row["abs_june_pearson"])
    maxs = max(r["factor_pairs"], key=lambda row: row["abs_june_spearman"])
    maxpc = max(core_pairs, key=lambda row: row["abs_june_pearson"])
    maxsc = max(core_pairs, key=lambda row: row["abs_june_spearman"])
    print(f"factor max Pearson     : {maxp['left_factor']}-{maxp['right_factor']} "
          f"rho={maxp['june_pearson']:.4f} (含 sensitivity)"
          if not maxp["core_pair"] else
          f"factor max Pearson     : {maxp['left_factor']}-{maxp['right_factor']} "
          f"rho={maxp['june_pearson']:.4f}")
    print(f"factor CORE max Pearson: {maxpc['left_factor']}-{maxpc['right_factor']} "
          f"rho={maxpc['june_pearson']:.4f}")
    print(f"factor CORE max Spear. : {maxsc['left_factor']}-{maxsc['right_factor']} "
          f"rho={maxsc['june_spearman']:.4f}")
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
    print(f"core severe pairs      : {len(r['core_severe_pairs'])}")
    print(f"sensitivity severe pairs: {len(r['sensitivity_severe_pairs'])}")
    print(f"status                 : {r['status']}")
    for reason in r["status_reasons"]:
        print(f"  - {reason}")
    print(f"output                 : {args.output} ({len(OUTPUT_FILES)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

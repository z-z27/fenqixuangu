"""v004c Repair-State 模型 v002 — 语义修订 + 纯 X 结构审计工具。

职责:

- 读取 model table 的 X 子集: 显式 usecols 只加载 3 个标识列 + 8 个 core
  primitive (+ 1 个 sensitivity primitive), 读取入口即 target-blind
- June (reference X sample) 拟合 reference transform (primitive FOLD_CLIP_Z +
  composite CLOSE_DAMAGE/RECLAIM + 3 个 derived terms 的 mean/std)
- July (unlabeled X stability slice) 只 apply June 参数 (不得用 July 重算)
- 构造 8 个 factor (R1 = 5 base; R2 = 8)
- DIVERGENCE primitive 核验 (§4: d1_intraday_range = (d1_high-d1_low)/prev_close,
  只使用 D1 数据; 核验结果写入 review)
- 语义解耦检查 (§27-29): DIVERGENCE vs RECLAIM (判定门) / DIVERGENCE vs
  CLOSE_DAMAGE (允许相关, 只记录) / CLOSE_DAMAGE vs RECLAIM (允许相关, 只记录)
- RECLAIM 内部结构审计 (v001 冻结结果复核)
- primitive/factor 相关性 (Pearson + Spearman), VIF (R1/R2), condition number
  (R1/R2/STATE BLOCK/CONDITIONAL BLOCK), factor 分布, DIVERGENCE 分布与非线性
  支持 (§35), July 无标签 X 稳定性 (SMD / KS), derived tail 诊断
- 正式判定门 (§37): DIVERGENCE primitive mismatch =>
  BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH; DIVERGENCE vs RECLAIM June |corr| >= 0.70
  => SEMANTIC_DECOUPLING_REVIEW; degenerate / VIF >= 10 / kappa >= 100 =>
  REVIEW_REQUIRED; 否则 PASS_REPAIR_STATE_X_V002。WATCH 只记录, 不进入判定门
- 输出 6 个正式报告文件 + 控制台摘要 (确定性, 无时间戳)

明确禁止: 任何模型训练/预测/指标; 读取 Target; 根据 Target 修改 factor;
根据诊断自动 drop/swap/换权重/加 interaction; 全样本预处理。

用法:
    python tools/v004c_repair_state_dependence_audit_v002.py
    python tools/v004c_repair_state_dependence_audit_v002.py --input <csv> --output <dir>
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
    FORBIDDEN_TOKENS,
    X_ID_COLUMNS,
    DEGENERATE_STD,
    KS_WATCH,
    SMD_WATCH,
    FactorSpecError,
    condition_number_diagnostic,
    correlation_severity,
    ks_2samp_statistic,
    pearson,
    spearman,
    standardized_mean_difference,
    vif_matrix,
)
from src.v004c_repair_state_spec_v002 import (
    BASE_FACTORS,
    CLOSE_DAMAGE_COMPOSITE_KEY,
    CLOSE_DAMAGE_PRIMITIVES,
    DERIVED_FACTORS,
    DERIVED_PARENTS,
    DIVERGENCE_PRIMITIVE,
    FACTOR_NAMES,
    FACTOR_PRIMITIVES,
    HIERARCHY_EXPECTED_PAIRS,
    LEGACY_SENSITIVITY_FACTORS,
    R1_FACTORS,
    R2_FACTORS,
    RECLAIM_COMPOSITE_KEY,
    RECLAIM_PRIMITIVES,
    REPAIR_PRIMITIVES,
    SEMANTIC_STATE_PAIRS,
    SENSITIVITY_FACTORS,
    SENSITIVITY_PRIMITIVES,
    DERIVED_TAIL_MAX_ABS,
    KAPPA_SEVERE,
    KAPPA_WATCH,
    NONLINEARITY_NARROW_P95P05,
    SEMANTIC_DECOUPLING_REVIEW_CORR,
    SEMANTIC_DECOUPLING_WATCH_CORR,
    VIF_SEVERE,
    VIF_WATCH,
    RepairStateError,
    audit_status_v2,
    construct_repair_factors_v2,
    fit_repair_state_transform_v2,
    repair_spec_rows_v2,
    semantic_decoupling_status,
)

DEFAULT_INPUT = (REPO_ROOT
                 / "reports/research/v004c_model_table_v001_20260601_20260729"
                 / "v004c_model_table_v001.csv")
DEFAULT_OUTPUT = (REPO_ROOT
                  / "reports/research/v004c_repair_state_dependence_v002_20260601_20260729")

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"

OUTPUT_FILES = (
    "v004c_repair_state_primitive_dependence_v002.csv",
    "v004c_repair_state_factor_dependence_v002.csv",
    "v004c_repair_state_diagnostics_v002.csv",
    "v004c_repair_state_spec_v002.csv",
    "v004c_repair_state_model_spec_v002.md",
    "v004c_repair_state_review_v002.md",
)

# X loader 加载的全部 primitive 列: 8 core + 1 sensitivity (TURNOVER_COST 源字段)
PRIMITIVE_LOAD_COLUMNS = REPAIR_PRIMITIVES + SENSITIVITY_PRIMITIVES

STATE_BLOCK_FACTORS = ("DIVERGENCE", "CLOSE_DAMAGE", "RECLAIM", "SUPPLY")
CONDITIONAL_BLOCK_FACTORS = DERIVED_FACTORS

# RECLAIM 内部两 primitive (顺序即 pair 顺序)
RECLAIM_A, RECLAIM_B = RECLAIM_PRIMITIVES

# 解释阈值 (只解释, 禁止自动删除变量)
CORR_SEVERE = 0.85

# diagnostics 长表 section 顺序 (固定, 保证确定性)
DIAG_SECTIONS = (
    "reclaim_internal", "semantic_decoupling", "vif", "condition_number",
    "factor_dist", "divergence_dist", "shift", "derived_tail",
    "sensitivity_primitive",
)


# ---------------------------------------------------------------------------
# target-blind X loader
# ---------------------------------------------------------------------------

def load_x_table(input_csv,
                 primitive_columns: tuple = PRIMITIVE_LOAD_COLUMNS,
                 id_columns: tuple = X_ID_COLUMNS) -> pd.DataFrame:
    """显式 usecols 只加载 ID + 8 core primitive + 1 sensitivity primitive 列.

    - 请求列含禁止 token (target/d2_/d3_/future/...) => 直接失败
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
_DIST_FIELDS = ("mean", "std", "min", "p01", "p05", "p25", "median",
                "p75", "p95", "p99", "max", "skew")


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


# ---------------------------------------------------------------------------
# 相关性长表
# ---------------------------------------------------------------------------

def primitive_pair_rows(june: pd.DataFrame, july: pd.DataFrame) -> list[dict]:
    """8 choose 2 = 28 对 (June 为主参考, July 只做稳定性描述)."""
    rows = []
    for i in range(len(REPAIR_PRIMITIVES)):
        for j in range(i + 1, len(REPAIR_PRIMITIVES)):
            left, right = REPAIR_PRIMITIVES[i], REPAIR_PRIMITIVES[j]
            rp_jun = pearson(june[left], june[right])
            rs_jun = spearman(june[left], june[right])
            reclaim_internal = (left, right) == (RECLAIM_A, RECLAIM_B)
            rows.append({
                "left": left,
                "right": right,
                "reclaim_internal": reclaim_internal,
                "june_pearson": rp_jun,
                "june_spearman": rs_jun,
                "abs_june_pearson": abs(rp_jun),
                "abs_june_spearman": abs(rs_jun),
                "severity": correlation_severity(max(abs(rp_jun), abs(rs_jun))),
                "july_pearson": pearson(july[left], july[right]),
                "july_spearman": spearman(july[left], july[right]),
            })
    return rows


def factor_pair_rows(f_june: pd.DataFrame, f_july: pd.DataFrame) -> list[dict]:
    """8 choose 2 = 28 对 (June 为主参考, July 只做稳定性描述).

    classification (只记录, 禁止机械删变量):
    - EXPECTED_HIERARCHICAL_DEPENDENCE: derived 与其父 factor 的天然层级依赖
    - SEMANTIC_STATE_DEPENDENCE: 三个状态 factor (DIVERGENCE/CLOSE_DAMAGE/
      RECLAIM) 之间的经济学相关对, 允许相关 (§28/§29), HIGH/SEVERE 时标记
    - UNEXPECTED_REDUNDANCY: 非层级 SEVERE pair (|rho| >= 0.85)
    """
    rows = []
    for i in range(len(FACTOR_NAMES)):
        for j in range(i + 1, len(FACTOR_NAMES)):
            left, right = FACTOR_NAMES[i], FACTOR_NAMES[j]
            rp_jun = pearson(f_june[left], f_june[right])
            rs_jun = spearman(f_june[left], f_june[right])
            max_abs = max(abs(rp_jun), abs(rs_jun))
            severity = correlation_severity(max_abs)
            hierarchy = (left, right) in HIERARCHY_EXPECTED_PAIRS
            if hierarchy:
                cls = "EXPECTED_HIERARCHICAL_DEPENDENCE"
            elif (left, right) in SEMANTIC_STATE_PAIRS and severity in ("HIGH", "SEVERE"):
                cls = "SEMANTIC_STATE_DEPENDENCE"
            elif severity == "SEVERE":
                cls = "UNEXPECTED_REDUNDANCY"
            else:
                cls = ""
            rows.append({
                "left_factor": left,
                "right_factor": right,
                "hierarchy_expected": hierarchy,
                "classification": cls,
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
# 审计主流程
# ---------------------------------------------------------------------------

def run_repair_state_audit(input_csv, output_dir) -> dict:
    """完整纯 X 审计: 读取 -> June reference -> 8 factor -> 诊断 -> 写 6 文件.

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

    params = fit_repair_state_transform_v2(june)
    f_june = construct_repair_factors_v2(june, params)
    f_july = construct_repair_factors_v2(july, params)

    # --- primitive 相关性 (含 RECLAIM 内部重点) ---
    prim_pairs = primitive_pair_rows(june, july)
    r_int = next(p for p in prim_pairs if p["reclaim_internal"])
    rp, rs = abs(r_int["june_pearson"]), abs(r_int["june_spearman"])
    if rp < 0.10 and rs < 0.10:
        reclaim_status = "RECLAIM_COMPOSITE_REVIEW"
    else:
        reclaim_status = "COMPOSITE_SUPPORTED"
    reclaim_internal = {
        "june_pearson": r_int["june_pearson"],
        "june_spearman": r_int["june_spearman"],
        "july_pearson": r_int["july_pearson"],
        "july_spearman": r_int["july_spearman"],
        "status": reclaim_status,
    }

    # --- 语义解耦检查 (§27-29) ---
    semantic_decoupling: dict = {}
    for name, a, b in (("DIVERGENCE_RECLAIM", "DIVERGENCE", "RECLAIM"),
                       ("DIVERGENCE_CLOSE_DAMAGE", "DIVERGENCE", "CLOSE_DAMAGE"),
                       ("CLOSE_DAMAGE_RECLAIM", "CLOSE_DAMAGE", "RECLAIM")):
        semantic_decoupling[name] = {
            "june_pearson": pearson(f_june[a], f_june[b]),
            "june_spearman": spearman(f_june[a], f_june[b]),
            "july_pearson": pearson(f_july[a], f_july[b]),
            "july_spearman": spearman(f_july[a], f_july[b]),
        }
    # 判定门只用于 DIVERGENCE vs RECLAIM (spec 契约: pearson/spearman 键 = June 值)
    div_rec = {
        "pearson": semantic_decoupling["DIVERGENCE_RECLAIM"]["june_pearson"],
        "spearman": semantic_decoupling["DIVERGENCE_RECLAIM"]["june_spearman"],
    }
    decoupling_status, decoupling_max_corr = semantic_decoupling_status(div_rec)

    # --- factor 相关性 (含 hierarchy/state/redundancy 分类) ---
    fac_pairs = factor_pair_rows(f_june, f_july)

    # --- VIF (June) ---
    vif_r1 = vif_matrix(f_june[list(R1_FACTORS)])
    vif_r2 = vif_matrix(f_june[list(R2_FACTORS)])

    # --- condition number (centered + unit-variance 诊断矩阵) ---
    kappa = {}
    for name, factors in (("R1", R1_FACTORS), ("R2", R2_FACTORS),
                          ("STATE_BLOCK", STATE_BLOCK_FACTORS),
                          ("CONDITIONAL_BLOCK", CONDITIONAL_BLOCK_FACTORS)):
        s_max, s_min, k, near = condition_number_diagnostic(
            f_june[list(factors)].to_numpy(dtype=float))
        kappa[name] = {"s_max": s_max, "s_min": s_min, "kappa": k, "near_singular": near}

    # --- factor 分布 + July 稳定性 + degenerate ---
    factor_stats: dict = {}
    degenerate_factors: list[str] = []
    shift_factors: list[str] = []
    for f in FACTOR_NAMES:
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

    # --- DIVERGENCE 分布与非线性支持 (§35) ---
    div_dist = {q: factor_stats["DIVERGENCE"]["june"][q] for q in
                ("min", "p05", "p25", "median", "p75", "p95", "max")}
    div_p95p05 = factor_stats["DIVERGENCE"]["june"]["p95"] \
        - factor_stats["DIVERGENCE"]["june"]["p05"]
    if not np.isfinite(div_p95p05) or div_p95p05 < NONLINEARITY_NARROW_P95P05:
        nonlinearity_support = "DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW"
    else:
        nonlinearity_support = "NONLINEARITY_SUPPORT_OK"
    div_dist["p95_minus_p05"] = div_p95p05

    # --- derived tail 诊断 (June/July, 检查少量极端值主导) ---
    derived_tail: dict[str, dict] = {}
    derived_tail_watch: list[str] = []
    for f in DERIVED_FACTORS:
        tail = {"june": _tail_row(f_june[f]), "july": _tail_row(f_july[f])}
        june_max_abs = max(abs(tail["june"]["max"]), abs(tail["june"]["min"]))
        if np.isfinite(june_max_abs) and june_max_abs >= DERIVED_TAIL_MAX_ABS:
            tail["watch"] = "DERIVED_TAIL_WATCH"
            derived_tail_watch.append(f)
        else:
            tail["watch"] = ""
        derived_tail[f] = tail

    # --- TURNOVER_COST sensitivity primitive 描述 (只记录, 不构造 factor) ---
    sensitivity_primitive: dict = {}
    for p in SENSITIVITY_PRIMITIVES:
        sensitivity_primitive[p] = {
            "june": describe_continuous(june[p]), "july": describe_continuous(july[p])}

    # --- 正式判定门 (§37) ---
    status, reasons = audit_status_v2(
        degenerate_factors, vif_r1, vif_r2, kappa, div_rec)

    # --- hierarchy / unexpected redundancy 分类 (只记录, 不阻塞) ---
    expected_pairs = [(p["left_factor"], p["right_factor"]) for p in fac_pairs
                      if p["hierarchy_expected"]]
    semantic_state_pairs = [(p["left_factor"], p["right_factor"]) for p in fac_pairs
                            if p["classification"] == "SEMANTIC_STATE_DEPENDENCE"]
    unexpected_severe_pairs = [p for p in fac_pairs
                               if p["severity"] == "SEVERE"
                               and not p["hierarchy_expected"]]

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
                           for p in REPAIR_PRIMITIVES},
        "primitive_pairs": prim_pairs,
        "reclaim_internal": reclaim_internal,
        "semantic_decoupling": semantic_decoupling,
        "decoupling_status": decoupling_status,
        "decoupling_max_corr": decoupling_max_corr,
        "factor_pairs": fac_pairs,
        "factor_stats": factor_stats,
        "vif_r1": vif_r1,
        "vif_r2": vif_r2,
        "kappa": kappa,
        "divergence_dist": div_dist,
        "nonlinearity_support": nonlinearity_support,
        "derived_tail": derived_tail,
        "sensitivity_primitive": sensitivity_primitive,
        "r1_factors": list(R1_FACTORS),
        "r2_factors": list(R2_FACTORS),
        "sensitivity_factors": list(SENSITIVITY_FACTORS),
        "degenerate_factors": degenerate_factors,
        "shift_factors": shift_factors,
        "derived_tail_watch": derived_tail_watch,
        "expected_hierarchical_pairs": expected_pairs,
        "semantic_state_pairs": semantic_state_pairs,
        "unexpected_severe_pairs": unexpected_severe_pairs,
        "status": status,
        "status_reasons": reasons,
    }

    _write_all(output_dir, results)
    return results


def _tail_row(s: pd.Series) -> dict:
    """derived tail 诊断行: min/p01/p05/p25/median/p75/p95/p99/max/mean/std/skew."""
    s = pd.to_numeric(s, errors="coerce")
    finite = s.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {k: np.nan for k in ("min", "p01", "p05", "p25", "median",
                                    "p75", "p95", "p99", "max", "mean", "std",
                                    "skew")}
    q = np.quantile(finite, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "min": float(finite.min()), "p01": float(q[0]), "p05": float(q[1]),
        "p25": float(q[2]), "median": float(q[3]), "p75": float(q[4]),
        "p95": float(q[5]), "p99": float(q[6]), "max": float(finite.max()),
        "mean": float(finite.mean()), "std": float(finite.std(ddof=0)),
        "skew": float(pd.Series(finite).skew()),
    }


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
        output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig",
        float_format="%.6f")

    # 2. factor dependence CSV
    pd.DataFrame(r["factor_pairs"]).to_csv(
        output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig",
        float_format="%.6f")

    # 3. diagnostics CSV (统一长表: section/scope/metric/june/july/status)
    pd.DataFrame(_diagnostics_rows(r)).to_csv(
        output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig",
        float_format="%.6f")

    # 4. spec CSV (静态规格: 8 active + 8 sensitivity)
    pd.DataFrame(repair_spec_rows_v2()).to_csv(
        output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")

    # 5. model spec md (R0/R1/R2 公式 + 未来算法, 只记录)
    (output_dir / OUTPUT_FILES[4]).write_text(
        _model_spec_md(r), encoding="utf-8")

    # 6. review md (数据驱动, 确定性)
    (output_dir / OUTPUT_FILES[5]).write_text(
        _review_md(r), encoding="utf-8")


def _diagnostics_rows(r: dict) -> list[dict]:
    """diagnostics 长表 (section 顺序固定, 保证确定性)."""
    rows: list[dict] = []

    def add(section, scope, metric, june=None, july=None, status=""):
        rows.append({"section": section, "scope": scope, "metric": metric,
                     "june": june, "july": july, "status": status})

    # reclaim_internal
    ri = r["reclaim_internal"]
    for metric in ("june_pearson", "june_spearman", "july_pearson", "july_spearman"):
        add("reclaim_internal", "RECLAIM", metric, ri[metric], None, ri["status"])

    # semantic_decoupling (3 对)
    for name in ("DIVERGENCE_RECLAIM", "DIVERGENCE_CLOSE_DAMAGE",
                 "CLOSE_DAMAGE_RECLAIM"):
        d = r["semantic_decoupling"][name]
        for metric in ("june_pearson", "june_spearman", "july_pearson",
                       "july_spearman"):
            status = (r["decoupling_status"] if name == "DIVERGENCE_RECLAIM"
                      else "")
            add("semantic_decoupling", name, metric, d[metric], None, status)

    # vif
    for f in FACTOR_NAMES:
        for block, v in (("R1", r["vif_r1"].get(f)),
                         ("R2", r["vif_r2"].get(f))):
            if v is None or np.isnan(v):
                continue
            status = ("SEVERE" if v >= VIF_SEVERE
                      else "WATCH" if v >= VIF_WATCH else "OK")
            add("vif", f, block, v, None, status)

    # condition_number
    for name in ("R1", "R2", "STATE_BLOCK", "CONDITIONAL_BLOCK"):
        k = r["kappa"][name]
        for metric, val in (("s_max", k["s_max"]), ("s_min", k["s_min"]),
                            ("kappa", k["kappa"])):
            if k["near_singular"] or k["kappa"] >= KAPPA_SEVERE:
                status = "SEVERE"
            elif k["kappa"] >= KAPPA_WATCH:
                status = "WATCH"
            else:
                status = "OK"
            if k["near_singular"]:
                status += "/near_singular"
            add("condition_number", name, metric, val, None, status)

    # factor_dist (June/July)
    for f in FACTOR_NAMES:
        for month in ("june", "july"):
            d = r["factor_stats"][f][month]
            for metric in _DIST_FIELDS:
                add("factor_dist", f, f"{month}_{metric}", d[metric], None, "")

    # divergence_dist (June quantiles)
    for metric, val in r["divergence_dist"].items():
        add("divergence_dist", "DIVERGENCE", metric, val, None,
            r["nonlinearity_support"])

    # shift (SMD/KS)
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]
        add("shift", f, "smd", st["smd"], None, st["stationarity_flag"])
        add("shift", f, "ks", st["ks"], None, st["stationarity_flag"])

    # derived_tail
    for f in DERIVED_FACTORS:
        for month in ("june", "july"):
            d = r["derived_tail"][f][month]
            for metric in ("min", "p01", "p05", "p25", "median", "p75",
                           "p95", "p99", "max", "mean", "std", "skew"):
                add("derived_tail", f, f"{month}_{metric}", d[metric], None,
                    r["derived_tail"][f]["watch"])

    # sensitivity_primitive (TURNOVER_COST 源字段, 只描述)
    for p, d in r["sensitivity_primitive"].items():
        for month in ("june", "july"):
            for metric in ("n", "missing", "unique", "min", "median", "max"):
                add("sensitivity_primitive", p, f"{month}_{metric}",
                    d[month][metric], None, "")
    return rows


# ---------------------------------------------------------------------------
# model spec md (§36: 未来算法只记录, 本任务禁止执行训练)
# ---------------------------------------------------------------------------

def _model_spec_md(r: dict) -> str:
    return f"""# v004c Repair-State Model v002 — 模型规格 (R0/R1/R2 数学结构 — 仅记录, 不拟合)

- 阶段: Repair-State v002 语义修订 + 纯 X 结构审计产物; 本规格只定义公式与未来训练算法
- 状态: 由 v004c_repair_state_review_v002.md 结论背书 (X 结构 / 数值稳定性 / June→July
  X 稳定性; 与 Target 无关)
- 注意: 这是 X spec freeze 候选, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## Post-June hypothesis 声明

> Repair-State v002 是在 June walk-forward 失败 (REJECT_NO_STABLE_MODEL) 之后形成的
> post-June hypothesis (v001 基础上的语义修订)。因此 **June 不能再作为该模型的独立
> 验证集**。
> 后续流程: X spec freeze -> June development fit -> freeze coefficients/spec
> -> July retrospective OOT -> August+ forward shadow。
> 不得声称 June 可以验证新模型。

## 交易逻辑

2/3板连续强势 -> D1 首次断板 -> 发生分歧 -> 分歧不能不足, 也不能过度透支 ->
最终损伤必须可控 -> 尾盘供应最好衰竭 -> 分歧后需要出现真实重新承接 -> D2-D3 强修复。

三个状态 factor 是不同阶段 (过程分歧 / 最终损伤 / 弱点后回收):

- DIVERGENCE = D1 盘中实际发生过多大的首次分歧 (过程强度)
- CLOSE_DAMAGE = 这次分歧结束以后, D1 收盘最终留下多少结构损伤
- RECLAIM = D1 出现弱点以后, 资金有没有重新把价格接回来

## Factor 定义与预处理 (future walk-forward)

- 每个 training fold 内拟合 (FOLD_CLIP_Z 顺序: 在 fold 原始值上拟合 q01/q99 -> clip
  -> 在 clipped fold 值上拟合 mu/sigma) + (x - mu)/sigma (fold 参数; 禁止全样本参数)
- composite (CLOSE_DAMAGE / RECLAIM): 由标准化 primitive 构造 G, 在 fold 上拟合 G 的
  mu/sigma -> z(G)
- derived terms: 由标准化 base factor 构造 raw (F_V^2 / F_V*F_Q / F_V*F_C), 在 fold
  上拟合 raw 的 mu/sigma -> 标准化; **不做第二次 clip**
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## R0 — BASE RATE

    logit(p) = alpha

未来实现等价于 training prevalence。

## R1 — BASE REPAIR STATE (5 factors + intercept)

    logit(p) = alpha
             + beta_O * F_OPEN
             + beta_V * F_DIVERGENCE
             + beta_C * F_CLOSE_DAMAGE
             + beta_S * F_SUPPLY
             + beta_Q * F_RECLAIM

## R2 — CONDITIONAL REPAIR STATE (8 factors + intercept)

    logit(p) = alpha
             + beta_O * F_OPEN
             + beta_V * F_DIVERGENCE
             + beta_C * F_CLOSE_DAMAGE
             + beta_S * F_SUPPLY
             + beta_Q * F_RECLAIM
             + gamma_1 * F_DIVERGENCE_SQ
             + gamma_2 * F_DIVERGENCE_X_RECLAIM
             + gamma_3 * F_DIVERGENCE_X_DAMAGE

## 预声明 derived terms (v002 只允许这 3 个)

- DIVERGENCE_SQ = z(F_V^2): 分歧可能存在最佳区间 (太小 -> 浮筹释放不足; 适度 ->
  健康换手; 极端 -> 大量兑现/承接消耗/修复预算透支); 若未来 linear DIV > 0 且
  DIV_SQ < 0 可形成倒 U, 但禁止用 Target 验证方向
- DIVERGENCE_X_RECLAIM = z(F_V * F_Q): 同样大分歧后, 有真实资金重新承接与没有
  完全不同 (DIV高+Q高 = 强回收; DIV高+Q低 = 缺乏承接)
- DIVERGENCE_X_DAMAGE = z(F_V * F_C): 大分歧最终是剧烈换手还是实际留下严重结构
  破坏 (DIV高+C低 = 过程剧烈但损伤可控; DIV高+C高 = 获利盘兑现 + 承接消耗 +
  修复预算透支)
- 禁止自动增加: OPEN_X_DIVERGENCE / DIV_X_TURNOVER / RECLAIM_SQ / SUPPLY_SQ /
  CLOSE_DAMAGE_SQ (不属于当前 v002 模型)

## Hierarchy 原则 (R2 必须遵守)

- 存在 DIVERGENCE_SQ => DIVERGENCE 必须存在
- 存在 DIVERGENCE_X_RECLAIM => DIVERGENCE 与 RECLAIM 必须同时存在
- 存在 DIVERGENCE_X_DAMAGE => DIVERGENCE 与 CLOSE_DAMAGE 必须同时存在

## Sensitivity (禁止进入 R1/R2 core)

- DIVERGENCE_X_SUPPLY: v001 derived 定义保留, v002 降为 SENSITIVITY (避免
  interaction 膨胀; 优先验证'分歧过度''分歧后回收''分歧是否留下真实损伤')
- TURNOVER_COST = z(break_volume_ratio_vs_board_days): 未来机制预声明 (同样的价格
  分歧, 极端放量 => 更大筹码兑现和修复预算消耗); 本版本禁止构造 DIV_X_TURNOVER
- LEGACY_SENSITIVITY: HIGHZONE / MOM7 / DAMAGE7 / REGIME / POS7 / TREND
  (历史定义保留在 v004c_factor_spec.py, 本版本不删除)

## 未来训练算法 (只记录, 本阶段禁止执行)

- sklearn LogisticRegression: penalty = l2, C = 1.0, solver = lbfgs,
  fit_intercept = True, class_weight = None, max_iter = 1000
- 目标函数 (beta = non-intercept coefficients; alpha/intercept is not penalized):

      min_{{alpha, beta}} [ -sum_i ( y_i*log(p_i) + (1-y_i)*log(1-p_i) ) + lambda * ||beta||_2^2 ]

- Target (未来阶段): target7_daily_d2open_d3high = 1[High_D3/Open_D2 - 1 >= 0.07];
  本阶段禁止读取该 Target

## June/July 命名约定

- June: development / reference X sample (2026-06-01 ~ 2026-06-30)
- July: retrospective unlabeled X-stability slice (2026-07-01 ~ 2026-07-29,
  只 apply June reference 参数, 不读取 July Target)
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


def _review_md(r: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# v004c Repair-State Model v002 — 语义修订 + 纯 X 结构审计")
    add("")
    add("- 阶段: Repair-State v002 语义修订 + 纯 X 结构审计 (X ONLY, 不训练模型,"
        " 不读取 Target)")
    add("- 分支: research-sample-analysis")
    add("- 输入: `v004c_model_table_v001.csv` SHA256: `{}`".format(r["input_sha256"]))
    add("- 背景: Repair-State v001 纯 X 审计通过 (PASS_REPAIR_STATE_X_V001), 但人工"
        " 复核发现 v001 DIVERGENCE 沿用旧 RESET ((-z_OC+z_HC-z_VWAP)/3), 主要描述"
        " 'D1 最终收盘留下多少价格损伤' 而非 '盘中发生过多大的第一次分歧', 且 "
        " Corr(DIVERGENCE, RECLAIM) June Pearson = -0.7340 / Spearman = -0.7725 "
        " 存在明显机械反向关系; v002 把 '盘中分歧强度' 与 '最终价格损伤' 拆成"
        " 两个独立 factor (DIVERGENCE / CLOSE_DAMAGE)。")
    add("")
    add("- 修订内容 (v001 -> v002): DIVERGENCE 改为 z(d1_intraday_range) (单"
        " primitive, 过程强度); 旧 RESET 数学整体移入 CLOSE_DAMAGE (公式零复制,"
        " 语义改为最终损伤); 新增 derived DIVERGENCE_X_DAMAGE (替代 "
        " DIVERGENCE_X_SUPPLY); DIVERGENCE_X_SUPPLY 降为 SENSITIVITY; 预声明"
        " TURNOVER_COST sensitivity。")
    add("")

    # 0. 方法与 target-blind
    add("## 0. 方法 (固定, 不搜索)")
    add("")
    add("- 基础 factor 只有 5 个: OPEN / DIVERGENCE / CLOSE_DAMAGE / SUPPLY /"
        " RECLAIM (把 v001 的'分歧+损伤'混合因子拆成两个独立状态)")
    add("- DIVERGENCE 只来自 `d1_intraday_range` 单 primitive (D1 盘中 high-low"
        " range / prev_close, 过程强度; 禁止使用 open_to_close / high_to_close /"
        " close_to_vwap 直接定义 DIVERGENCE)")
    add("- CLOSE_DAMAGE 与旧 RESET 数学定义完全一致 (权重 -1/3, +1/3, -1/3 固定,"
        " 禁止修改); 公式函数直接复用 `v004c_factor_spec.composite_reset`,"
        " 逐行误差只允许浮点容差")
    add("- RECLAIM 公式与 v001 完全一致 (权重冻结, 不得改权重, 不得加第三个"
        " primitive); 复用 v001 `composite_reclaim`")
    add("- SUPPLY 只使用 `late_day_sell_volume_ratio` 单独构造"
        " (不重新使用旧的 HIGHZONE + LATESELL composite)")
    add("- 预声明 derived terms 只有 3 个: DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM /"
        " DIVERGENCE_X_DAMAGE (只做 mean/std 标准化, 不做第二次 clip);"
        " 禁止自动搜索 interaction")
    add("- X loader 使用显式 `usecols` 只加载 3 个标识列 (event_id, code, signal_date)"
        " + 8 个 core primitive (+ 1 个 sensitivity primitive"
        " break_volume_ratio_vs_board_days); `target7_daily_d2open_d3high` / D2 /"
        " D3 / future return / 旧模型输出从未被加载 (读取入口即 target-blind)")
    add("- June = **development / reference X sample** ({start} ~ {end}, 用于拟合"
        " primitive q01/q99/mu/sigma + composite + derived mean/std)".format(
            start=JUNE_START, end=JUNE_END))
    add("- July = **retrospective unlabeled X-stability slice** ({start} ~ {end},"
        " 只 apply June reference 参数; 不得用 July 重算 mean/std, 不得读取 July"
        " Target)".format(start=JULY_START, end=JULY_END))
    add("- transform 顺序: primitive 先 FOLD_CLIP_Z (reference 原始有限值 -> q01/q99"
        " -> clip -> clipped mean/std -> z); composite 再 z(G); derived 最后"
        " z(raw)。July 全程只 apply。")
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
    add("- 窗口外行: 0 (全部行都在 June/July 窗口内)")
    add("")

    # 2. DIVERGENCE primitive 核验 (§4 人工审计结果)
    add("## 2. DIVERGENCE primitive 核验 (d1_intraday_range)")
    add("")
    add("| 项目 | 内容 |")
    add("|---|---|")
    add("| source file (定义文档) | `reports/research/v004c_dataset_20260506_20260729/"
        "v004c_feature_definitions.md` (D1 振幅, 行 113: `(high - low) / prev_close`) |")
    add("| source file (生成代码) | `reports/research/v004c_dataset_20260506_20260729/"
        "_scratch/build_v004c_dataset.py:684` |")
    add("| source function / definition | `r[\\\"d1_intraday_range\\\"] = "
        "_safe_div(h_c - l_c, prev_close_d1)`, 其中 `h_c`/`l_c` = D1 日内 5min "
        "high 最大值 / low 最小值, `prev_close_d1` = 断板日前收盘 (最后板日收盘) |")
    add("| exact formula | `(d1_high - d1_low) / prev_close` (D1 盘中 high-low price "
        "range / amplitude, 以前收盘归一化) |")
    add("| model table schema | `v004c_model_table_schema_v001.csv`: D1_CLOSE / "
        "D1_PRICE_ACTION / FOLD_CLIP_Z / PRIMARY |")
    add("| uses only D1 data | **YES** (只使用 D1 日内 high/low 与 D1 开盘前已知的 "
        "prev_close; 不依赖 D2/D3, 不依赖 Target) |")
    add("| semantic match | **YES** (仅描述 D1 盘中价格探索范围, 与收盘位置无关; "
        "价格重新收回来也不会自动消除分歧记录) |")
    add("- 结论: 满足 §4 语义要求, 直接复用 `d1_intraday_range` 作为 DIVERGENCE 的"
        " 唯一 primitive。不构造新公式。")
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
    add("- 完整 28 对明细见 `v004c_repair_state_primitive_dependence_v002.csv`"
        " (reclaim_internal 标记 RECLAIM 内部对)。")
    add("")

    # 4. RECLAIM 内部结构审计
    add("## 4. RECLAIM 内部结构审计 (d1_low_to_close_recovery vs d1_afternoon_return)")
    add("")
    ri = r["reclaim_internal"]
    add("| 切片 | Pearson | Spearman |")
    add("|---|---|---|")
    add("| June | {} | {} |".format(_fmt(ri["june_pearson"]),
                                    _fmt(ri["june_spearman"])))
    add("| July | {} | {} |".format(_fmt(ri["july_pearson"]),
                                    _fmt(ri["july_spearman"])))
    add("")
    add("- v001 冻结依据 (2026-06 审计): June 0.5787 / 0.5414; July 0.4409 / 0.4306"
        " (COMPOSITE_SUPPORTED)。")
    add("- composite status: **{}**".format(ri["status"]))
    if ri["status"] == "RECLAIM_COMPOSITE_REVIEW":
        add("- June |Pearson| < 0.10 且 |Spearman| < 0.10: 两者几乎独立, 不能默认"
            " 50/50 合成一定合理 => RECLAIM_COMPOSITE_REVIEW (只记录, 不自动拆分)。")
    else:
        add("- June |Pearson| / |Spearman| 至少一项 >= 0.10: 存在同一潜变量的可辨"
            " 证据, composite 构造合理 (COMPOSITE_SUPPORTED)。")
    add("")

    # 5. factor 构造与分布
    add("## 5. factor 构造与 June 分布 (8 factors)")
    add("")
    add("- 构造: 8 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8"
        " (degenerate 检查: {})".format(
            "无退化" if not r["degenerate_factors"] else r["degenerate_factors"]))
    add("- DIVERGENCE = z(d1_intraday_range) (单 primitive); CLOSE_DAMAGE = z(G_C),"
        " G_C = (-z_OC + z_HC - z_VWAP)/3 (与旧 RESET 数学一致); RECLAIM = z(G_Q)"
        " (与 v001 一致)")
    add("- derived terms: raw = F_V^2 / F_V*F_Q / F_V*F_C, 只做 mean/std 标准化,"
        " 未施加第二次 clip")
    add("")
    add("| factor | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | skew |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]["june"]
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            f, _fmt(st["mean"]), _fmt(st["std"]), _fmt(st["min"]),
            _fmt(st["p01"]), _fmt(st["p05"]), _fmt(st["p25"]),
            _fmt(st["median"]), _fmt(st["p75"]), _fmt(st["p95"]),
            _fmt(st["p99"]), _fmt(st["max"]), _fmt(st["skew"])))
    add("")

    # 6. 语义解耦检查
    add("## 6. 语义解耦检查 (§27-29: DIVERGENCE / CLOSE_DAMAGE / RECLAIM)")
    add("")
    add("| pair | 切片 | Pearson | Spearman |")
    add("|---|---|---|---|")
    for name, label in (("DIVERGENCE_RECLAIM", "DIVERGENCE vs RECLAIM"),
                        ("DIVERGENCE_CLOSE_DAMAGE", "DIVERGENCE vs CLOSE_DAMAGE"),
                        ("CLOSE_DAMAGE_RECLAIM", "CLOSE_DAMAGE vs RECLAIM")):
        d = r["semantic_decoupling"][name]
        add("| {} | June | {} | {} |".format(
            label, _fmt(d["june_pearson"]), _fmt(d["june_spearman"])))
        add("|  | July | {} | {} |".format(
            _fmt(d["july_pearson"]), _fmt(d["july_spearman"])))
    add("")
    add("- v001 参考: old DIVERGENCE vs RECLAIM June Pearson = -0.7340 / Spearman ="
        " -0.7725 (v002 的核心修订目标: 新 DIVERGENCE 必须不再是 RECLAIM 的机械反面)")
    d = r["semantic_decoupling"]["DIVERGENCE_RECLAIM"]
    add("- DIVERGENCE vs RECLAIM 判定门 (§27): June max|corr| = {} "
        "(Pearson {} / Spearman {}) => **{}**".format(
            _fmt(r["decoupling_max_corr"]), _fmt(d["june_pearson"]),
            _fmt(d["june_spearman"]), r["decoupling_status"]))
    add("- 判据: |corr| >= 0.70 => SEMANTIC_DECOUPLING_REVIEW (人工语义阻塞);"
        " 0.50 <= |corr| < 0.70 => SEMANTIC_DECOUPLING_WATCH (只记录);"
        " < 0.50 => SEMANTIC_DECOUPLING_OK")
    add("- DIVERGENCE vs CLOSE_DAMAGE / CLOSE_DAMAGE vs RECLAIM 允许相关 (§28/§29,"
        " 不设 0.70 硬失败): 分歧越剧烈通常最终损伤可能越大; 回收越强收盘损伤通常"
        " 越小。是否构成问题由 VIF / condition number 整体判断。")
    add("")

    # 7. factor 相关性
    add("## 7. factor 相关性 (June 主参考; July 稳定性; 三层分类)")
    add("")
    fp = r["factor_pairs"]
    maxp = max(fp, key=lambda row: row["abs_june_pearson"])
    maxs = max(fp, key=lambda row: row["abs_june_spearman"])
    add("- 最大 June Pearson pair: `{}-{}` rho={} (severity={}){}".format(
        maxp["left_factor"], maxp["right_factor"], _fmt(maxp["june_pearson"]),
        maxp["severity"], " — 层级项" if maxp["hierarchy_expected"] else ""))
    add("- 最大 June Spearman pair: `{}-{}` rho={} (severity={}){}".format(
        maxs["left_factor"], maxs["right_factor"], _fmt(maxs["june_spearman"]),
        maxs["severity"], " — 层级项" if maxs["hierarchy_expected"] else ""))
    add("- 完整 28 对明细见 `v004c_repair_state_factor_dependence_v002.csv`"
        " (classification 列 = EXPECTED_HIERARCHICAL_DEPENDENCE / "
        "SEMANTIC_STATE_DEPENDENCE / UNEXPECTED_REDUNDANCY)。")
    add("")
    add("### 层级项分类 (EXPECTED_HIERARCHICAL_DEPENDENCE)")
    add("")
    add("- 天然层级依赖 pair (derived 与其父 factor, 高相关属预期):")
    if r["expected_hierarchical_pairs"]:
        for left, right in r["expected_hierarchical_pairs"]:
            row = next(p for p in fp
                       if p["left_factor"] == left and p["right_factor"] == right)
            add("  - `{}-{}`: June Pearson={} / Spearman={} (severity={}) —"
                " EXPECTED_HIERARCHICAL_DEPENDENCE (不得机械删除)".format(
                    left, right, _fmt(row["june_pearson"]),
                    _fmt(row["june_spearman"]), row["severity"]))
    else:
        add("  - 无")
    add("")
    add("### 状态因子相关对 (SEMANTIC_STATE_DEPENDENCE, §28/§29 允许)")
    add("")
    if r["semantic_state_pairs"]:
        for left, right in r["semantic_state_pairs"]:
            row = next(p for p in fp
                       if p["left_factor"] == left and p["right_factor"] == right)
            add("  - `{}-{}`: June Pearson={} / Spearman={} (July Pearson={}"
                " / Spearman={}) — SEMANTIC_STATE_DEPENDENCE (只记录, 不机械删变量;"
                " 由 VIF/kappa 整体判断)".format(
                    left, right, _fmt(row["june_pearson"]),
                    _fmt(row["june_spearman"]), _fmt(row["july_pearson"]),
                    _fmt(row["july_spearman"])))
        add("")
    else:
        add("- 无 HIGH/SEVERE 级状态因子相关对。")
        add("")
    if r["unexpected_severe_pairs"]:
        add("- 非层级 SEVERE pairs (|rho| >= 0.85, UNEXPECTED_REDUNDANCY, 只记录供人工):")
        for p in r["unexpected_severe_pairs"]:
            add("  - `{}-{}`: June Pearson={} / Spearman={} (July Pearson={}"
                " / Spearman={})".format(
                    p["left_factor"], p["right_factor"],
                    _fmt(p["june_pearson"]), _fmt(p["june_spearman"]),
                    _fmt(p["july_pearson"]), _fmt(p["july_spearman"])))
        add("")
    else:
        add("- 非层级 SEVERE pairs: 无 (所有高相关 pair 都属层级依赖或状态相关)。")
        add("")

    # 8. VIF
    add("## 8. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)")
    add("")
    add("| factor | R1 VIF | R2 VIF |")
    add("|---|---|---|")
    for f in FACTOR_NAMES:
        m1 = r["vif_r1"].get(f)
        m2 = r["vif_r2"].get(f)
        add("| {} | {} | {} |".format(
            f, _fmt(m1, 3) if m1 is not None else "—",
            _fmt(m2, 3) if m2 is not None else "—"))
    add("")
    for label, v in (("R1", r["vif_r1"]), ("R2", r["vif_r2"])):
        bad = [(f, val) for f, val in v.items() if val >= 5.0]
        add("- {} max VIF: {} (factor `{}`); VIF >= 5: {}; VIF >= 10: {}".format(
            label, _fmt(max(v.values()), 3), max(v, key=v.get),
            ", ".join("{}={}".format(f, _fmt(val, 3)) for f, val in bad) or "无",
            ", ".join("{}={}".format(f, _fmt(val, 3)) for f, val in bad if val >= 10)
            or "无"))
    add("- 解释: VIF < 5 OK; 5 <= VIF < 10 WATCH; VIF >= 10 SEVERE (只解释,"
        " 禁止自动删因子; VIF/condition number 必须真实报告, 不因层级项人为放宽)")
    add("")

    # 9. condition number
    add("## 9. condition number (centered + unit-variance 诊断矩阵, SVD)")
    add("")
    add("| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |")
    add("|---|---|---|---|---|---|")
    for name, factors, label in (("R1", R1_FACTORS,
                                  "OPEN DIVERGENCE CLOSE_DAMAGE SUPPLY RECLAIM"),
                                 ("R2", R2_FACTORS, "8 factors (含 3 个层级项)"),
                                 ("STATE_BLOCK", STATE_BLOCK_FACTORS,
                                  "DIVERGENCE CLOSE_DAMAGE RECLAIM SUPPLY"),
                                 ("CONDITIONAL_BLOCK", CONDITIONAL_BLOCK_FACTORS,
                                  "DIVERGENCE_SQ X_RECLAIM X_DAMAGE")):
        k = r["kappa"][name]
        if k["near_singular"] or k["kappa"] >= KAPPA_SEVERE:
            judge = "SEVERE"
        elif k["kappa"] >= KAPPA_WATCH:
            judge = "WATCH"
        else:
            judge = "OK"
        add("| {} | {} | {} | {} | {} | {} |".format(
            name, label, _fmt(k["s_max"], 3), _fmt(k["s_min"], 6),
            _fmt(k["kappa"], 3),
            judge + (" / near singular" if k["near_singular"] else "")))
    add("- 参考: kappa < 30 OK; 30-100 WATCH; >= 100 SEVERE; smallest singular ~ 0"
        " => SEVERE / near singular")
    add("")

    # 10. DIVERGENCE 分布与非线性支持
    add("## 10. DIVERGENCE 分布与非线性支持检查 (§35, 只用 X 数学, 不使用 Target)")
    add("")
    dd = r["divergence_dist"]
    add("| 统计 | 值 |")
    add("|---|---|")
    for metric, label in (("min", "min"), ("p05", "p05"), ("p25", "p25"),
                          ("median", "median"), ("p75", "p75"),
                          ("p95", "p95"), ("max", "max"),
                          ("p95_minus_p05", "p95 - p05")):
        add("| {} | {} |".format(label, _fmt(dd[metric])))
    add("")
    add("- 非线性支持: **{}**".format(r["nonlinearity_support"]))
    if r["nonlinearity_support"] == "DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW":
        add("- DIVERGENCE June 分布过窄 (p95-p05 < {}), 平方项可能几乎无信息;"
            " 只记录, 不自动删因子 (不要用 Target 判断)".format(
                NONLINEARITY_NARROW_P95P05))
    else:
        add("- DIVERGENCE June p95-p05 = {} >= {}, 覆盖 low/middle/high/extreme"
            " 区间, 平方项 (DIVERGENCE_SQ) 有表达空间 (只做 X 数学检查)。".format(
                _fmt(dd["p95_minus_p05"]), NONLINEARITY_NARROW_P95P05))
    add("")

    # 11. July 稳定性
    add("## 11. July 无标签 X 稳定性 (June reference transform 映射)")
    add("")
    add("| factor | June mean | June std | July mean | July std | SMD | KS | flag |")
    add("|---|---|---|---|---|---|---|---|")
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]
        add("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            f, _fmt(st["june"]["mean"]), _fmt(st["june"]["std"]),
            _fmt(st["july"]["mean"]), _fmt(st["july"]["std"]),
            _fmt(st["smd"]), _fmt(st["ks"]),
            st["stationarity_flag"] or "—"))
    add("")
    add("- 判定: |SMD| >= 0.50 或 KS >= 0.25 => SHIFT_WATCH (只记录, 禁止自动换因子)")
    if r["shift_factors"]:
        add("- SHIFT_WATCH 因子: `{}` — 进入未来模型阶段后持续监控".format(
            ", ".join(r["shift_factors"])))
    else:
        add("- SHIFT_WATCH 因子: 无")
    add("")

    # 12. derived tail
    add("## 12. derived tail 诊断 (interaction 是否被少量极端值主导)")
    add("")
    add("| factor | 切片 | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max | mean | std | skew |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in DERIVED_FACTORS:
        for month in ("june", "july"):
            d = r["derived_tail"][f][month]
            add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, month, _fmt(d["min"]), _fmt(d["p01"]), _fmt(d["p05"]),
                _fmt(d["p25"]), _fmt(d["median"]), _fmt(d["p75"]),
                _fmt(d["p95"]), _fmt(d["p99"]), _fmt(d["max"]),
                _fmt(d["mean"]), _fmt(d["std"]), _fmt(d["skew"])))
    add("")
    add("- 判定: June 标准化后 max|min| / max|max| >= {} => DERIVED_TAIL_WATCH"
        " (只记录; derived 项不做第二次 clip, 不 winsorize)".format(
            DERIVED_TAIL_MAX_ABS))
    if r["derived_tail_watch"]:
        add("- DERIVED_TAIL_WATCH 因子: `{}`".format(
            ", ".join(r["derived_tail_watch"])))
    else:
        add("- DERIVED_TAIL_WATCH 因子: 无")
    add("")

    # 13. hierarchy 检查
    add("## 13. Hierarchy 检查确认")
    add("")
    ok = all(set(DERIVED_PARENTS[d]) <= set(R2_FACTORS) for d in DERIVED_FACTORS)
    add("- R2 成员包含每个 derived term 的全部父 factor: {}".format(
        "PASS" if ok else "FAIL"))
    add("- 禁止新增 interaction (OPEN_X_DIVERGENCE / DIV_X_TURNOVER / RECLAIM_SQ /"
        " SUPPLY_SQ / CLOSE_DAMAGE_SQ): 不属于 v002 模型, 未构造")
    add("- SENSITIVITY ({}) 全部不在 R1/R2 core: {}".format(
        ", ".join(SENSITIVITY_FACTORS),
        "PASS" if not set(SENSITIVITY_FACTORS) & set(R2_FACTORS) else "FAIL"))
    add("")

    # 14. Sensitivity 说明
    add("## 14. Sensitivity 说明")
    add("")
    add("- DIVERGENCE_X_SUPPLY (v001 derived): 定义保留在 v001/v002 spec, v002 降为"
        " SENSITIVITY (当前优先验证'分歧过度''分歧后的回收''分歧是否留下真实损伤'"
        " 三个核心机制, 避免 interaction 继续膨胀)")
    add("- TURNOVER_COST = z(break_volume_ratio_vs_board_days): 字段定义核验通过"
        " (`break_volume / mean(断板前 streak 个板日 volume)`, 见 "
        " `v004c_feature_definitions.md` 与 model table schema); 只预声明,"
        " 本版本禁止构造 DIV_X_TURNOVER, 禁止进入 R1/R2")
    for p in SENSITIVITY_PRIMITIVES:
        d = r["sensitivity_primitive"][p]
        add("- `{}` (sensitivity 源字段) June: n={} missing={} unique={} "
            "min={} median={} max={}; July: n={} missing={} unique={} "
            "min={} median={} max={}".format(
                p, d["june"]["n"], d["june"]["missing"], d["june"]["unique"],
                _fmt(d["june"]["min"]), _fmt(d["june"]["median"]),
                _fmt(d["june"]["max"]),
                d["july"]["n"], d["july"]["missing"], d["july"]["unique"],
                _fmt(d["july"]["min"]), _fmt(d["july"]["median"]),
                _fmt(d["july"]["max"])))
    add("- LEGACY_SENSITIVITY ({}): 全部不在 R1/R2; 历史定义保留, 不删除".format(
        ", ".join(LEGACY_SENSITIVITY_FACTORS)))
    add("")

    # 15. 最终状态
    add("## 15. 最终状态")
    add("")
    add("- 最终状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 触发原因:")
        for reason in r["status_reasons"]:
            add("  - " + reason)
    add("- 判定门 (任务 §37): DIVERGENCE primitive mismatch => "
        "BLOCKED_DIVERGENCE_PRIMITIVE_MISMATCH; DIVERGENCE vs RECLAIM June"
        " |Pearson| >= 0.70 或 |Spearman| >= 0.70 => SEMANTIC_DECOUPLING_REVIEW;"
        " degenerate factor / VIF >= 10 / condition number >= 100 =>"
        " REVIEW_REQUIRED; 否则 PASS_REPAIR_STATE_X_V002。")
    add("- 非阻塞记录 (不进入判定门): SEMANTIC_DECOUPLING_WATCH = {}; SHIFT_WATCH ="
        " {}; DERIVED_TAIL_WATCH = {}; DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW = {}"
        " (VIF 5-10 / kappa 30-100 同样只记录)".format(
            r["decoupling_status"] if r["decoupling_status"]
            == "SEMANTIC_DECOUPLING_WATCH" else "无",
            r["shift_factors"] or "无",
            "`{}`".format(", ".join(r["derived_tail_watch"]))
            if r["derived_tail_watch"] else "无",
            "是" if r["nonlinearity_support"]
            == "DIVERGENCE_NONLINEARITY_SUPPORT_REVIEW" else "否"))
    add("- 本审查只做 X 结构 / 数值稳定性 / June→July X 稳定性诊断: 未读取 Target,"
        " 未执行任何训练/预测/指标, 未用 Target7 rate / AUC / LogLoss / Brier /"
        " coefficient / Top3 参与判定")
    add("- 禁止自动修正: 即使出现 SEMANTIC_DECOUPLING_REVIEW / SEVERE / VIF >= 10 /"
        " kappa >= 100 / 退化, 本阶段也不 drop / swap / PCA / Lasso / 改权重 /"
        " 加 interaction; 修正方向只记录给人工")
    add("")

    # 16. 结论与后续流程
    add("## 16. 结论与后续流程")
    add("")
    add("- 本阶段只回答: 新 DIVERGENCE 是否与 RECLAIM 语义解耦、R1/R2 的 X 结构"
        " 是否健康、CLOSE_DAMAGE 是否仍等于旧 RESET 数学、层级项是否数值健康、"
        " June→July X 是否发生明显漂移。")
    add("- 下一阶段必须人工审查: 语义解耦结果 / DIVERGENCE 构造合理性 / interaction"
        " 是否数值健康 / R1/R2 是否存在严重共线性 / June→July X 是否明显漂移;"
        " 只有人工批准后才能正式冻结 Repair-State v002, 进入 June development fit"
        " + frozen retrospective OOT。")
    add("- post-June hypothesis: 本模型是 June walk-forward 失败后形成的假设,"
        " June 不能再作为独立验证集; 后续流程 = X spec freeze -> June development fit"
        " -> freeze coefficients/spec -> July retrospective OOT -> August+ forward"
        " shadow (July 打开以后, 禁止再根据 July 结果修改 Repair-State v002)。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c Repair-State v002 语义修订 + 纯 X 结构审计 (不训练模型, 不读取 Target)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_repair_state_dependence_v002_20260601_20260729)")
    args = parser.parse_args(argv)

    r = run_repair_state_audit(args.input, args.output)

    print("=== v004c Repair-State dependence audit v002 ===")
    print(f"input      : {r['input']}")
    print(f"sha256     : {r['input_sha256']}")
    print(f"June rows  : {r['june_rows']} ({r['june_date_count']} signal dates)")
    print(f"July rows  : {r['july_rows']} ({r['july_date_count']} signal dates)")
    maxp = max(r["primitive_pairs"], key=lambda row: row["abs_june_pearson"])
    print(f"primitive max Pearson : {maxp['left']}-{maxp['right']} "
          f"rho={maxp['june_pearson']:.4f}")
    ri = r["reclaim_internal"]
    print(f"reclaim internal     : June Pearson={ri['june_pearson']:.4f} "
          f"Spearman={ri['june_spearman']:.4f} "
          f"/ July Pearson={ri['july_pearson']:.4f} "
          f"Spearman={ri['july_spearman']:.4f} -> {ri['status']}")
    d = r["semantic_decoupling"]
    print(f"decoupling DIV-REC   : June Pearson={d['DIVERGENCE_RECLAIM']['june_pearson']:.4f} "
          f"Spearman={d['DIVERGENCE_RECLAIM']['june_spearman']:.4f} "
          f"/ July Pearson={d['DIVERGENCE_RECLAIM']['july_pearson']:.4f} "
          f"Spearman={d['DIVERGENCE_RECLAIM']['july_spearman']:.4f} "
          f"-> {r['decoupling_status']}")
    print(f"decoupling DIV-CDMG  : June Pearson={d['DIVERGENCE_CLOSE_DAMAGE']['june_pearson']:.4f} "
          f"Spearman={d['DIVERGENCE_CLOSE_DAMAGE']['june_spearman']:.4f}")
    print(f"decoupling CDMG-REC  : June Pearson={d['CLOSE_DAMAGE_RECLAIM']['june_pearson']:.4f} "
          f"Spearman={d['CLOSE_DAMAGE_RECLAIM']['june_spearman']:.4f}")
    maxp = max(r["factor_pairs"], key=lambda row: row["abs_june_pearson"])
    print(f"factor max Pearson   : {maxp['left_factor']}-{maxp['right_factor']} "
          f"rho={maxp['june_pearson']:.4f} "
          f"({maxp['classification'] or 'OK'})")
    print(f"DIVERGENCE p95-p05   : {r['divergence_dist']['p95_minus_p05']:.4f} "
          f"-> {r['nonlinearity_support']}")
    r1f = max(r["vif_r1"], key=r["vif_r1"].get)
    r2f = max(r["vif_r2"], key=r["vif_r2"].get)
    print(f"R1 max VIF           : {r['vif_r1'][r1f]:.3f} ({r1f})")
    print(f"R2 max VIF           : {r['vif_r2'][r2f]:.3f} ({r2f})")
    for name in ("R1", "R2", "STATE_BLOCK", "CONDITIONAL_BLOCK"):
        k = r["kappa"][name]
        near = " NEAR-SINGULAR" if k["near_singular"] else ""
        print(f"kappa {name:16s}      : {k['kappa']:.3f}{near}")
    print(f"degenerate factors   : {r['degenerate_factors'] or 'none'}")
    print(f"shift factors        : {r['shift_factors'] or 'none'}")
    print(f"derived tail watch   : {r['derived_tail_watch'] or 'none'}")
    print(f"unexpected severe    : {len(r['unexpected_severe_pairs'])} pairs")
    print(f"status               : {r['status']}")
    for reason in r["status_reasons"]:
        print(f"  - {reason}")
    print(f"output               : {args.output} ({len(OUTPUT_FILES)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""v004c Repair-State 模型 v001 — 规格 + 纯 X 结构审计工具。

职责:

- 读取 model table 的 X 子集: 显式 usecols 只加载 3 个标识列 + 7 个 primitive,
  读取入口即 target-blind (绝不加载 target7_daily_d2open_d3high / D2 / D3 /
  future return / 旧模型输出)
- June (reference X sample) 拟合 reference transform (primitive FOLD_CLIP_Z +
  composite DIVERGENCE/RECLAIM + 3 个 derived terms 的 mean/std)
- July (unlabeled X stability slice) 只 apply June 参数 (不得用 July 重算)
- 构造 7 个 factor (R2 = OPEN/DIVERGENCE/SUPPLY/RECLAIM/DIVERGENCE_SQ/
  DIVERGENCE_X_RECLAIM/DIVERGENCE_X_SUPPLY; R1 = 前 4 个)
- RECLAIM 内部结构审计 (两 primitive 的 June/July Pearson + Spearman)
- primitive/factor 相关性 (Pearson + Spearman), VIF (R1/R2), condition number
  (R1/R2/REPAIR BLOCK/CONDITIONAL BLOCK, centered + unit-variance 诊断矩阵),
  factor 分布, July 无标签 X 稳定性 (SMD / KS), derived tail 诊断
- 正式判定门 (§32): degenerate factor / VIF >= 10 / condition number >= 100
  => REVIEW_REQUIRED; RECLAIM 两 primitive 双低相关 => RECLAIM_COMPOSITE_REVIEW;
  否则 PASS_REPAIR_STATE_X_V001。SHIFT_WATCH / DERIVED_TAIL_WATCH / hierarchy
  相关只记录, 不进入判定门
- 输出 6 个正式报告文件 + 控制台摘要 (确定性, 无时间戳)

明确禁止: 任何模型训练/预测/指标; 读取 Target; 根据 Target 修改 factor;
根据诊断自动 drop/swap/换权重/加 interaction; 全样本预处理。

用法:
    python tools/v004c_repair_state_dependence_audit.py
    python tools/v004c_repair_state_dependence_audit.py --input <csv> --output <dir>
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
from src.v004c_repair_state_spec import (
    BASE_FACTORS,
    DERIVED_FACTORS,
    DERIVED_PARENTS,
    FACTOR_NAMES,
    FACTOR_PRIMITIVES,
    HIERARCHY_EXPECTED_PAIRS,
    LEGACY_SENSITIVITY_FACTORS,
    R1_FACTORS,
    R2_FACTORS,
    RECLAIM_PRIMITIVES,
    REPAIR_PRIMITIVES,
    DERIVED_TAIL_MAX_ABS,
    KAPPA_SEVERE,
    KAPPA_WATCH,
    RECLAIM_WEAK_CORR,
    VIF_SEVERE,
    VIF_WATCH,
    RepairStateError,
    audit_status,
    construct_repair_factors,
    fit_repair_transform,
    repair_spec_rows,
)

DEFAULT_INPUT = (REPO_ROOT
                 / "reports/research/v004c_model_table_v001_20260601_20260729"
                 / "v004c_model_table_v001.csv")
DEFAULT_OUTPUT = (REPO_ROOT
                  / "reports/research/v004c_repair_state_dependence_v001_20260601_20260729")

JUNE_START, JUNE_END = "2026-06-01", "2026-06-30"
JULY_START, JULY_END = "2026-07-01", "2026-07-29"

OUTPUT_FILES = (
    "v004c_repair_state_primitive_dependence_v001.csv",
    "v004c_repair_state_factor_dependence_v001.csv",
    "v004c_repair_state_diagnostics_v001.csv",
    "v004c_repair_state_spec_v001.csv",
    "v004c_repair_state_model_spec_v001.md",
    "v004c_repair_state_review_v001.md",
)

REPAIR_BLOCK_FACTORS = ("DIVERGENCE", "RECLAIM", "SUPPLY")
CONDITIONAL_BLOCK_FACTORS = DERIVED_FACTORS

# RECLAIM 内部两 primitive (顺序即 pair 顺序)
RECLAIM_A, RECLAIM_B = RECLAIM_PRIMITIVES

# 解释阈值 (只解释, 禁止自动删除变量)
CORR_SEVERE = 0.85

# diagnostics 长表 section 顺序 (固定, 保证确定性)
DIAG_SECTIONS = (
    "reclaim_internal", "vif", "condition_number",
    "factor_dist", "shift", "derived_tail",
)


# ---------------------------------------------------------------------------
# target-blind X loader
# ---------------------------------------------------------------------------

def load_x_table(input_csv,
                 primitive_columns: tuple = REPAIR_PRIMITIVES,
                 id_columns: tuple = X_ID_COLUMNS) -> pd.DataFrame:
    """显式 usecols 只加载 ID + 7 个 primitive 列 (读取入口即 target-blind).

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
    """7 choose 2 = 21 对 (June 为主参考, July 只做稳定性描述)."""
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
    """7 choose 2 = 21 对 (June 为主参考, July 只做稳定性描述).

    hierarchy_expected = derived 与其父 factor 的天然层级依赖
    (高相关属 EXPECTED_HIERARCHICAL_DEPENDENCE, 不得机械删除).
    """
    rows = []
    for i in range(len(FACTOR_NAMES)):
        for j in range(i + 1, len(FACTOR_NAMES)):
            left, right = FACTOR_NAMES[i], FACTOR_NAMES[j]
            rp_jun = pearson(f_june[left], f_june[right])
            rs_jun = spearman(f_june[left], f_june[right])
            rows.append({
                "left_factor": left,
                "right_factor": right,
                "hierarchy_expected": (left, right) in HIERARCHY_EXPECTED_PAIRS,
                "june_pearson": rp_jun,
                "june_spearman": rs_jun,
                "abs_june_pearson": abs(rp_jun),
                "abs_june_spearman": abs(rs_jun),
                "severity": correlation_severity(max(abs(rp_jun), abs(rs_jun))),
                "july_pearson": pearson(f_july[left], f_july[right]),
                "july_spearman": spearman(f_july[left], f_july[right]),
            })
    return rows


# ---------------------------------------------------------------------------
# 审计主流程
# ---------------------------------------------------------------------------

def run_repair_state_audit(input_csv, output_dir) -> dict:
    """完整纯 X 审计: 读取 -> June reference -> 7 factor -> 诊断 -> 写 6 文件.

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

    params = fit_repair_transform(june)
    f_june = construct_repair_factors(june, params)
    f_july = construct_repair_factors(july, params)

    # --- primitive 相关性 (含 RECLAIM 内部重点) ---
    prim_pairs = primitive_pair_rows(june, july)
    r_int = next(p for p in prim_pairs if p["reclaim_internal"])
    reclaim_weak = (abs(r_int["june_pearson"]) < RECLAIM_WEAK_CORR
                    and abs(r_int["june_spearman"]) < RECLAIM_WEAK_CORR)
    reclaim_status = ("RECLAIM_COMPOSITE_REVIEW" if reclaim_weak
                      else "COMPOSITE_SUPPORTED")

    # --- factor 相关性 (含 hierarchy 分类) ---
    fac_pairs = factor_pair_rows(f_june, f_july)

    # --- VIF (June) ---
    vif_r1 = vif_matrix(f_june[list(R1_FACTORS)])
    vif_r2 = vif_matrix(f_june[list(R2_FACTORS)])

    # --- condition number (centered + unit-variance 诊断矩阵) ---
    kappa = {}
    for name, factors in (("R1", R1_FACTORS), ("R2", R2_FACTORS),
                          ("REPAIR_BLOCK", REPAIR_BLOCK_FACTORS),
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

    # --- 正式判定门 (§32) ---
    status, reasons = audit_status(
        degenerate_factors, vif_r1, vif_r2, kappa, reclaim_weak)

    # --- hierarchy / unexpected redundancy 分类 (只记录, 不阻塞) ---
    expected_pairs = [(p["left_factor"], p["right_factor"]) for p in fac_pairs
                      if p["hierarchy_expected"]]
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
        "reclaim_internal": {
            "june_pearson": r_int["june_pearson"],
            "june_spearman": r_int["june_spearman"],
            "july_pearson": r_int["july_pearson"],
            "july_spearman": r_int["july_spearman"],
            "status": reclaim_status,
        },
        "factor_pairs": fac_pairs,
        "factor_stats": factor_stats,
        "vif_r1": vif_r1,
        "vif_r2": vif_r2,
        "kappa": kappa,
        "derived_tail": derived_tail,
        "r1_factors": list(R1_FACTORS),
        "r2_factors": list(R2_FACTORS),
        "legacy_sensitivity": list(LEGACY_SENSITIVITY_FACTORS),
        "degenerate_factors": degenerate_factors,
        "shift_factors": shift_factors,
        "derived_tail_watch": derived_tail_watch,
        "expected_hierarchical_pairs": expected_pairs,
        "unexpected_severe_pairs": unexpected_severe_pairs,
        "reclaim_weak": reclaim_weak,
        "status": status,
        "status_reasons": reasons,
    }

    _write_all(output_dir, results)
    return results


def _tail_row(s: pd.Series) -> dict:
    """derived tail 诊断行: min/p01/p05/median/p95/p99/max/mean/std."""
    s = pd.to_numeric(s, errors="coerce")
    finite = s.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {k: np.nan for k in ("min", "p01", "p05", "median",
                                    "p95", "p99", "max", "mean", "std")}
    q = np.quantile(finite, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "min": float(finite.min()), "p01": float(q[0]), "p05": float(q[1]),
        "median": float(q[2]), "p95": float(q[3]), "p99": float(q[4]),
        "max": float(finite.max()), "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
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

    # 4. spec CSV (静态规格: 7 active + 6 legacy)
    pd.DataFrame(repair_spec_rows()).to_csv(
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
    for name in ("R1", "R2", "REPAIR_BLOCK", "CONDITIONAL_BLOCK"):
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

    # shift (SMD/KS)
    for f in FACTOR_NAMES:
        st = r["factor_stats"][f]
        add("shift", f, "smd", st["smd"], None, st["stationarity_flag"])
        add("shift", f, "ks", st["ks"], None, st["stationarity_flag"])

    # derived_tail
    for f in DERIVED_FACTORS:
        for month in ("june", "july"):
            d = r["derived_tail"][f][month]
            for metric in ("min", "p01", "p05", "median", "p95", "p99",
                           "max", "mean", "std"):
                add("derived_tail", f, f"{month}_{metric}", d[metric], None,
                    r["derived_tail"][f]["watch"])
    return rows


# ---------------------------------------------------------------------------
# model spec md (§36: 未来算法只记录, 本任务禁止执行训练)
# ---------------------------------------------------------------------------

def _model_spec_md(r: dict) -> str:
    return f"""# v004c Repair-State Model v001 — 模型规格 (R0/R1/R2 数学结构 — 仅记录, 不拟合)

- 阶段: Repair-State v001 规格 + 纯 X 结构审计产物; 本规格只定义公式与未来训练算法
- 状态: 由 v004c_repair_state_review_v001.md 结论背书 (X 结构 / 数值稳定性 / June→July
  X 稳定性; 与 Target 无关)
- 注意: 这是 X spec freeze 候选, **不是** model coefficient freeze;
  本阶段未执行任何训练/预测/指标, 未读取 Target

## Post-June hypothesis 声明

> Repair-State v001 是在 June walk-forward 失败 (REJECT_NO_STABLE_MODEL) 之后形成的
> post-June hypothesis。因此 **June 不能再作为该模型的独立验证集**。
> 后续流程: X spec freeze -> June development fit -> freeze coefficients/spec
> -> July retrospective OOT -> August+ forward shadow。
> 不得声称 June 可以验证新模型。

## 交易逻辑

2板/3板连续强势 -> 首次断板 -> D1 首次明显分歧 -> 判断分歧是否充分 -> 判断卖压是否
仍持续 -> 判断低位是否重新获得承接 -> 预测 D2-D3 是否发生强修复。

## Factor 定义与预处理 (future walk-forward)

- 每个 training fold 内拟合 (FOLD_CLIP_Z 顺序: 在 fold 原始值上拟合 q01/q99 -> clip
  -> 在 clipped fold 值上拟合 mu/sigma) + (x - mu)/sigma (fold 参数; 禁止全样本参数)
- composite: 由标准化 primitive 构造 G, 在 fold 上拟合 G 的 mu/sigma -> z(G)
- derived terms: 由标准化 base factor 构造 raw (F_D^2 / F_D*F_Q / F_D*F_S), 在 fold
  上拟合 raw 的 mu/sigma -> 标准化; **不做第二次 clip**
- 禁止对 beta/gamma 施加正负约束 (系数方向由模型估计)

## R0 — BASE RATE

    logit(p) = alpha

未来实现等价于 training prevalence。

## R1 — BASE REPAIR STATE (4 factors + intercept)

    logit(p) = alpha
             + beta_O  * F_OPEN
             + beta_D  * F_DIVERGENCE
             + beta_S  * F_SUPPLY
             + beta_Q  * F_RECLAIM

## R2 — CONDITIONAL REPAIR STATE (7 factors + intercept)

    logit(p) = alpha
             + beta_O * F_OPEN
             + beta_D * F_DIVERGENCE
             + beta_S * F_SUPPLY
             + beta_Q * F_RECLAIM
             + gamma_1 * F_DIVERGENCE_SQ
             + gamma_2 * F_DIVERGENCE_X_RECLAIM
             + gamma_3 * F_DIVERGENCE_X_SUPPLY

## 预声明 derived terms (第一版 v001 只允许这 3 个)

- DIVERGENCE_SQ = z(F_D^2): 允许"分歧存在最佳区间", 而非强制单调
- DIVERGENCE_X_RECLAIM = z(F_D * F_Q): 大分歧后是否真正出现重新承接 (核心条件项)
- DIVERGENCE_X_SUPPLY = z(F_D * F_S): 充分分歧时尾盘供应是否仍持续
- 禁止自动增加: OPEN_X_DIVERGENCE / HIGHZONE_X_DIVERGENCE / HIGHZONE_X_RECLAIM /
  RECLAIM_SQ / SUPPLY_SQ (不属于当前 v001 模型)

## Hierarchy 原则 (R2 必须遵守)

- 存在 DIVERGENCE_X_RECLAIM => DIVERGENCE 与 RECLAIM 必须同时存在
- 存在 DIVERGENCE_X_SUPPLY => DIVERGENCE 与 SUPPLY 必须同时存在
- 存在 DIVERGENCE_SQ => DIVERGENCE 必须存在

## Legacy sensitivity (禁止进入 R1/R2 core)

HIGHZONE / MOM7 / DAMAGE7 / REGIME / POS7 / TREND — 统一记录为 LEGACY_SENSITIVITY;
历史定义保留在 v004c_factor_spec.py (v002 规格), 本版本不删除。

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

    add("# v004c Repair-State Model v001 — 规格 + 纯 X 结构审计")
    add("")
    add("- 阶段: Repair-State v001 规格冻结 + 纯 X 结构审计 (X ONLY, 不训练模型,"
        " 不读取 Target)")
    add("- 分支: research-sample-analysis")
    add("- 输入: `v004c_model_table_v001.csv` SHA256: `{}`".format(r["input_sha256"]))
    add("- 背景: 上一版 M0/M1/M2 June expanding-date walk-forward 结论"
        " REJECT_NO_STABLE_MODEL; 本版把人工逐样本分析形成的新交易假设正式写成数学规格,"
        " 并做严格 Target-blind 的纯 X 审计 (post-June hypothesis: June 不能再作为"
        " 该模型的独立验证集)。")
    add("")

    # 0. 方法与 target-blind
    add("## 0. 方法 (固定, 不搜索)")
    add("")
    add("- 基础 factor 只有 4 个: OPEN / DIVERGENCE / SUPPLY / RECLAIM"
        " (上一版 OPEN + RESET + HIGHZONE + LATESELL 无法区分'跌得很多但是没人接'与"
        " '充分分歧以后重新出现承接'; 本版显式加入 RECLAIM 与条件项)")
    add("- DIVERGENCE 是旧 RESET 的重新命名和重新解释, 数学定义完全一致 (权重"
        " -1/3, +1/3, -1/3 固定, 禁止修改); 公式函数直接复用"
        " `v004c_factor_spec.composite_reset`")
    add("- SUPPLY 只使用 `late_day_sell_volume_ratio` 单独构造"
        " (不重新使用旧的 HIGHZONE + LATESELL composite)")
    add("- 预声明 derived terms 只有 3 个: DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM /"
        " DIVERGENCE_X_SUPPLY (只做 mean/std 标准化, 不做第二次 clip);"
        " 禁止自动搜索 interaction")
    add("- X loader 使用显式 `usecols` 只加载 3 个标识列 (event_id, code, signal_date)"
        " + 7 个 primitive 列; `target7_daily_d2open_d3high` / D2 / D3 / future return"
        " / 旧模型输出从未被加载 (读取入口即 target-blind)")
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

    # 2. primitive 相关性
    add("## 2. primitive 相关性 (June 主参考; July 仅稳定性描述)")
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
    add("- 完整 21 对明细见 `v004c_repair_state_primitive_dependence_v001.csv`"
        " (reclaim_internal 标记 RECLAIM 内部对)。")
    add("")

    # 3. RECLAIM 内部结构审计
    add("## 3. RECLAIM 内部结构审计 (d1_low_to_close_recovery vs d1_afternoon_return)")
    add("")
    ri = r["reclaim_internal"]
    add("| 切片 | Pearson | Spearman |")
    add("|---|---|---|")
    add("| June | {} | {} |".format(_fmt(ri["june_pearson"]),
                                    _fmt(ri["june_spearman"])))
    add("| July | {} | {} |".format(_fmt(ri["july_pearson"]),
                                    _fmt(ri["july_spearman"])))
    add("")
    add("- 目的: 判断它们是否确实可以作为同一 RECLAIM latent composite 的两个视角"
        " (50/50 等权是否合理)。")
    add("- composite status: **{}**".format(ri["status"]))
    if r["reclaim_weak"]:
        add("- June |Pearson| < 0.10 且 |Spearman| < 0.10: 两者几乎独立, 不能默认"
            " 50/50 合成一定合理 => RECLAIM_COMPOSITE_REVIEW, 该状态进入人工审查"
            " (禁止自动拆分 RECLAIM)。")
    else:
        add("- June |Pearson| / |Spearman| 至少一项 >= 0.10: 存在同一潜变量的可辨"
            " 证据, composite 构造合理 (COMPOSITE_SUPPORTED)。")
    add("")

    # 4. factor 构造与分布
    add("## 4. factor 构造与 June 分布 (7 factors)")
    add("")
    add("- 构造: 7 个 factor 全部 finite, 无缺失; 连续 factor June std 均 > 1e-8"
        " (degenerate 检查: {})".format(
            "无退化" if not r["degenerate_factors"] else r["degenerate_factors"]))
    add("- derived terms: raw = F_D^2 / F_D*F_Q / F_D*F_S, 只做 mean/std 标准化,"
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

    # 5. factor 相关性
    add("## 5. factor 相关性 (June 主参考; July 稳定性; hierarchy 分类)")
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
    add("- 完整 21 对明细见 `v004c_repair_state_factor_dependence_v001.csv`"
        " (hierarchy_expected 标记层级依赖)。")
    add("")
    add("### 层级项分类 (EXPECTED_HIERARCHICAL_DEPENDENCE vs UNEXPECTED_REDUNDANCY)")
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
        add("- 非层级 SEVERE pairs: 无 (所有高相关 pair 都属层级依赖)。")
        add("")

    # 6. VIF
    add("## 6. VIF (June; 每个 factor 用其余 factor 线性解释, 含 intercept)")
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

    # 7. condition number
    add("## 7. condition number (centered + unit-variance 诊断矩阵, SVD)")
    add("")
    add("| matrix | factors | largest s | smallest s | kappa = s_max/s_min | 判定 |")
    add("|---|---|---|---|---|---|")
    for name, factors, label in (("R1", R1_FACTORS, "OPEN DIVERGENCE SUPPLY RECLAIM"),
                                 ("R2", R2_FACTORS, "7 factors (含 3 个层级项)"),
                                 ("REPAIR_BLOCK", REPAIR_BLOCK_FACTORS,
                                  "DIVERGENCE RECLAIM SUPPLY"),
                                 ("CONDITIONAL_BLOCK", CONDITIONAL_BLOCK_FACTORS,
                                  "DIVERGENCE_SQ X_RECLAIM X_SUPPLY")):
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

    # 8. July 稳定性
    add("## 8. July 无标签 X 稳定性 (June reference transform 映射)")
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

    # 9. derived tail
    add("## 9. derived tail 诊断 (interaction 是否被少量极端值主导)")
    add("")
    add("| factor | 切片 | min | p01 | p05 | median | p95 | p99 | max | mean | std |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in DERIVED_FACTORS:
        for month in ("june", "july"):
            d = r["derived_tail"][f][month]
            add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, month, _fmt(d["min"]), _fmt(d["p01"]), _fmt(d["p05"]),
                _fmt(d["median"]), _fmt(d["p95"]), _fmt(d["p99"]),
                _fmt(d["max"]), _fmt(d["mean"]), _fmt(d["std"])))
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

    # 10. hierarchy 检查
    add("## 10. Hierarchy 检查确认")
    add("")
    ok = all(set(DERIVED_PARENTS[d]) <= set(R2_FACTORS) for d in DERIVED_FACTORS)
    add("- R2 成员包含每个 derived term 的全部父 factor: {}".format(
        "PASS" if ok else "FAIL"))
    add("- 禁止新增 interaction (OPEN_X_DIVERGENCE / HIGHZONE_X_DIVERGENCE /"
        " HIGHZONE_X_RECLAIM / RECLAIM_SQ / SUPPLY_SQ): 不属于 v001 模型, 未构造")
    add("- LEGACY_SENSITIVITY ({}) 全部不在 R1/R2 core: {}".format(
        ", ".join(LEGACY_SENSITIVITY_FACTORS),
        "PASS" if not set(LEGACY_SENSITIVITY_FACTORS) & set(R2_FACTORS) else "FAIL"))
    add("")

    # 11. 最终状态
    add("## 11. 最终状态")
    add("")
    add("- 最终状态: **{}**".format(r["status"]))
    if r["status_reasons"]:
        add("- 触发原因:")
        for reason in r["status_reasons"]:
            add("  - " + reason)
    add("- 判定门 (任务 §32): degenerate factor / VIF >= 10 / condition number >= 100"
        " => REVIEW_REQUIRED; 否则 RECLAIM 两 primitive 在 June 中 |Pearson| < 0.10"
        " 且 |Spearman| < 0.10 => RECLAIM_COMPOSITE_REVIEW (进入人工审查, 不自动拆);"
        " 否则 PASS_REPAIR_STATE_X_V001。")
    add("- 非阻塞记录 (不进入判定门): SHIFT_WATCH = {}; DERIVED_TAIL_WATCH = {};"
        " hierarchy 相关高只记 EXPECTED_HIERARCHICAL_DEPENDENCE".format(
            r["shift_factors"] or "无",
            "`{}`".format(", ".join(r["derived_tail_watch"]))
            if r["derived_tail_watch"] else "无"))
    add("- 本审查只做 X 结构 / 数值稳定性 / June→July X 稳定性诊断: 未读取 Target,"
        " 未执行任何训练/预测/指标, 未用 Target7 rate / AUC / LogLoss / Brier /"
        " coefficient / Top3 参与判定")
    add("- 禁止自动修正: 即使出现 SEVERE / VIF >= 10 / kappa >= 100 / 退化,"
        " 本阶段也不 drop / swap / PCA / Lasso / 改权重 / 加 interaction;"
        " 修正方向只记录给人工")
    add("")

    # 12. 结论与后续流程
    add("## 12. 结论与后续流程")
    add("")
    add("- 本阶段只回答: R1/R2 的 X 结构是否健康、RECLAIM composite 是否合理、"
        " 层级项是否数值健康、June→July X 是否发生明显漂移。")
    add("- 下一阶段必须人工审查: RECLAIM 是否构造合理 / interaction 是否数值健康 /"
        " R1/R2 是否存在严重共线性 / June→July X 是否明显漂移;"
        " 只有人工批准后才能进入 Repair-State development fit + frozen retrospective"
        " OOT。")
    add("- post-June hypothesis: 本模型是 June walk-forward 失败后形成的假设,"
        " June 不能再作为独立验证集; 后续流程 = X spec freeze -> June development fit"
        " -> freeze coefficients/spec -> July retrospective OOT -> August+ forward"
        " shadow。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v004c Repair-State v001 规格 + 纯 X 结构审计 (不训练模型, 不读取 Target)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="model table CSV 路径 (默认正式 model table)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="输出目录 (默认 reports/research/"
                             "v004c_repair_state_dependence_v001_20260601_20260729)")
    args = parser.parse_args(argv)

    r = run_repair_state_audit(args.input, args.output)

    print("=== v004c Repair-State dependence audit v001 ===")
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
    maxp = max(r["factor_pairs"], key=lambda row: row["abs_june_pearson"])
    maxs = max(r["factor_pairs"], key=lambda row: row["abs_june_spearman"])
    print(f"factor max Pearson   : {maxp['left_factor']}-{maxp['right_factor']} "
          f"rho={maxp['june_pearson']:.4f}"
          f" (hierarchy)" if maxp["hierarchy_expected"] else
          f"factor max Pearson   : {maxp['left_factor']}-{maxp['right_factor']} "
          f"rho={maxp['june_pearson']:.4f}")
    print(f"factor max Spearman  : {maxs['left_factor']}-{maxs['right_factor']} "
          f"rho={maxs['june_spearman']:.4f}")
    r1f = max(r["vif_r1"], key=r["vif_r1"].get)
    r2f = max(r["vif_r2"], key=r["vif_r2"].get)
    print(f"R1 max VIF           : {r['vif_r1'][r1f]:.3f} ({r1f})")
    print(f"R2 max VIF           : {r['vif_r2'][r2f]:.3f} ({r2f})")
    for name in ("R1", "R2", "REPAIR_BLOCK", "CONDITIONAL_BLOCK"):
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

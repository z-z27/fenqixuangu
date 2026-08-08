# -*- coding: utf-8 -*-
"""v004c Core Model Stage — Date-Conditional Pairwise Ridge v001 构建工具。

输入 (全部冻结资产, 只读):
- reports/research/v004c_pairwise_v1_feature_contract_v002_20260506_20260630/
  (input table / feature contract / label contract / schema)
- May 标签源: reports/research/v004c_may_d1_coverage_v001_202605/
  v004c_may_d1_candidates_v001.csv (d2_open_daily / d3_high_daily)
- June 标签源: reports/research/v004c_d1_dataset_v001_20260601_20260729/
  v004c_training_d1_v001.csv (d2_open_daily / d3_high_daily, 06-01..06-30)

流程:
1. 组装 319 行开发表 + OUTCOME_ONLY 列 (raw/capped opportunity return);
   断言 target7 == (raw >= 0.07) (§29), 断言 eligibility 319/319, 断言
   max signal_date <= 2026-06-30 (§57);
2. Upside / Tail 各自在 LAMBDA_GRID 上完整 chronological walk-forward;
   lambda 唯一选择标准 = date-weighted OOF pairwise logloss (§23-§25);
3. 以 selected lambda 生成 OOF predictions / daily metrics / rank-return
   metrics / fold metrics / coefficient stability / board2-3 diagnostics;
4. 用全部 May+June (39 dates) fit 最终 dev 模型 (§58) -> 两个 model JSON
   (含 input/contract/schema SHA256 lineage);
5. 完整 pipeline 运行两次, 全部 9 个输出 byte-identical (确定性 §78);
6. 渲染 review MD (§70-§74) + GO gates (§45 / §50 / §62)。

禁止: July 数据访问、feature 增删改、模型 zoo、按收益选 lambda。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_pairwise_ridge import (  # noqa: E402
    BOARD3_STREAK,
    CAP_RETURN_7,
    LAMBDA_GRID,
    MIN_TRAIN_SIGNAL_DATES,
    NOT_EVALUABLE_SINGLE_CLASS,
    TARGET7_THRESHOLD,
    build_board3_interaction,
    build_same_date_pairs,
    chronological_walkforward,
    compute_capped_opportunity_return,
    compute_practical_rank_metrics,
    date_weighted_pair_stats,
    fit_pairwise_ridge,
    fit_preprocessor,
    score_from_z,
    select_lambda,
    transform_preprocessor,
)

V002_DIR = (ROOT / "reports" / "research"
            / "v004c_pairwise_v1_feature_contract_v002_20260506_20260630")
MAY_CSV = (ROOT / "reports" / "research" / "v004c_may_d1_coverage_v001_202605"
           / "v004c_may_d1_candidates_v001.csv")
JUNE_CSV = (ROOT / "reports" / "research"
            / "v004c_d1_dataset_v001_20260601_20260729"
            / "v004c_training_d1_v001.csv")
OUT_DIR = ROOT / "reports" / "research" / "v004c_pairwise_ridge_v001_20260506_20260630"

CONTRACT_VERSION = "pairwise_v1_v002"
INPUT_CSV_NAME = "v004c_pairwise_v1_input_table_v002.csv"
CONTRACT_CSV_NAME = "v004c_pairwise_v1_feature_contract_v002.csv"
SCHEMA_CSV_NAME = "v004c_pairwise_v1_schema_v002.csv"

LAMBDA_SELECTION_CSV = "v004c_pairwise_ridge_lambda_selection_v001.csv"
OOF_PRED_CSV = "v004c_pairwise_ridge_oof_predictions_v001.csv"
DAILY_METRICS_CSV = "v004c_pairwise_ridge_daily_metrics_v001.csv"
RANK_RETURN_CSV = "v004c_pairwise_ridge_rank_return_metrics_v001.csv"
FOLD_METRICS_CSV = "v004c_pairwise_ridge_fold_metrics_v001.csv"
COEFF_CSV = "v004c_pairwise_ridge_coefficients_v001.csv"
UPSIDE_MODEL_JSON = "v004c_pairwise_ridge_upside_model_v001.json"
TAIL_MODEL_JSON = "v004c_pairwise_ridge_tail_model_v001.json"
REVIEW_MD = "v004c_pairwise_ridge_review_v001.md"

UPSIDE_LABEL = "target7_daily_d2open_d3high"
TAIL_LABEL = "tail_loss_daily_5pct"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# 冻结资产加载
# ---------------------------------------------------------------------------
def load_frozen_input() -> pd.DataFrame:
    df = pd.read_csv(V002_DIR / INPUT_CSV_NAME, encoding="utf-8-sig",
                     dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["event_id"] = df["event_id"].astype(str)
    df["signal_date"] = df["signal_date"].astype(str)
    return df


def load_feature_order() -> list[str]:
    fc = pd.read_csv(V002_DIR / CONTRACT_CSV_NAME, encoding="utf-8-sig")
    fc = fc.sort_values("feature_order")
    return fc["feature_name"].astype(str).tolist()


def load_frozen_outcomes(input_df: pd.DataFrame) -> pd.DataFrame:
    """OUTCOME_ONLY 原始值 (d2_open/d3_high) 装载 + 标签一致性校验 (§29 FATAL)。

    - May: 用 v002 foundation 的 label_d2_date/label_d3_date (含恢复后事件)
      从 canonical daily_unadjusted 缓存取 D2 open / D3 high;
    - June: 用冻结 June 训练表 (v004c_training_d1_v001.csv, 06-01..06-30)
      的 d2_open_daily / d3_high_daily;
    - 每个事件 recompute target7 == (raw >= 0.07), 与 input table 冻结标签
      不一致即 FATAL。
    """
    fnd = pd.read_csv(V002_DIR.parent / "v004c_baostock_d1_dev_v002_20260506_20260630"
                      / "v004c_baostock_d1_dev_v002.csv",
                      encoding="utf-8-sig", dtype={"code": str})
    fnd["code"] = fnd["code"].astype(str).str.zfill(6)
    june = pd.read_csv(JUNE_CSV, encoding="utf-8-sig", dtype={"code": str})
    june["code"] = june["code"].astype(str).str.zfill(6)
    june = june[june["signal_date"].astype(str).between("2026-06-01", "2026-06-30")]
    june = june.drop_duplicates("event_id", keep="last")

    merged = input_df[["event_id", UPSIDE_LABEL, TAIL_LABEL]].copy()

    # ---- May: cache-derived (frozen label dates) ----
    cache_dir = ROOT / "data" / "cache" / "daily_unadjusted"
    cache: dict[str, pd.DataFrame | None] = {}

    def _load_daily(code: str):
        if code not in cache:
            p = cache_dir / f"{code}_daily.pkl"
            if not p.exists():
                cache[code] = None
            else:
                f = pd.read_pickle(p)
                f = f.copy()
                f["date"] = pd.to_datetime(f["date"], errors="coerce") \
                    .dt.strftime("%Y-%m-%d")
                cache[code] = f.dropna(subset=["date"]) \
                    .drop_duplicates("date", keep="last").set_index("date")
        return cache[code]

    may = fnd[fnd["signal_date"].astype(str).str.startswith("2026-05")].copy()
    may_rows = []
    for _, r in may.iterrows():
        d = _load_daily(str(r["code"]))
        d2, d3 = str(r["label_d2_date"]), str(r["label_d3_date"])
        if d is None or d2 not in d.index or d3 not in d.index:
            raise RuntimeError(
                f"FATAL: May outcome cache missing {r['event_id']} ({d2},{d3})")
        may_rows.append({"event_id": str(r["event_id"]),
                         "d2_open_daily": float(d.loc[d2, "open"]),
                         "d3_high_daily": float(d.loc[d3, "high"])})
    may_out = pd.DataFrame(may_rows)

    # ---- June: frozen training table ----
    june_out = june[["event_id", "d2_open_daily", "d3_high_daily"]].copy()

    outcome = pd.concat([may_out, june_out], ignore_index=True)
    outcome = outcome.drop_duplicates("event_id", keep="last")
    merged = merged.merge(outcome, on="event_id", how="left")
    n_missing = int(merged["d2_open_daily"].isna().sum()
                    + merged["d3_high_daily"].isna().sum())
    if n_missing:
        raise RuntimeError(f"FATAL: {n_missing} outcome values missing")
    raw = merged["d3_high_daily"].to_numpy(float) \
        / merged["d2_open_daily"].to_numpy(float) - 1.0
    t7 = pd.to_numeric(merged[UPSIDE_LABEL], errors="coerce").to_numpy(float)
    if not np.allclose((raw >= TARGET7_THRESHOLD - 1e-9).astype(float), t7):
        raise RuntimeError(  # FATAL (§29)
            "FATAL: target7-return consistency violated vs frozen input table")
    merged[UPSIDE_LABEL] = t7
    merged[TAIL_LABEL] = pd.to_numeric(merged[TAIL_LABEL], errors="coerce") \
        .to_numpy(float)
    return input_df.merge(merged.drop(columns=[UPSIDE_LABEL, TAIL_LABEL]),
                          on="event_id", how="left")


# ---------------------------------------------------------------------------
# 指标工具
# ---------------------------------------------------------------------------
def _day_rank(score: np.ndarray, event_ids: np.ndarray,
              dates: np.ndarray) -> np.ndarray:
    """同日 rank: score 降序, event_id 升序 (§68); 返回 1-based rank。

    按 signal_date 分组计算, 跨日期不混合排名。
    """
    score = np.asarray(score, dtype=float)
    ids = np.asarray(event_ids)
    dates = np.asarray(dates)
    rank = np.empty(len(score), dtype=int)
    for d in np.unique(dates):
        idx = np.where(dates == d)[0]
        order = np.lexsort((ids[idx], -score[idx]))  # score 主键降序, id 次键
        r = np.empty(len(idx), dtype=int)
        r[order] = np.arange(1, len(idx) + 1)
        rank[idx] = r
    return rank


def _split_oof(oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    oof = oof.copy()
    oof["month"] = oof["signal_date"].astype(str).str[:7]
    return {
        "May": oof[oof["month"] == "2026-05"],
        "June": oof[oof["month"] == "2026-06"],
        "Combined": oof,
    }


def _subset_pair_stats(sub: pd.DataFrame, label_col: str):
    """子集 (board2/board3) 的 date-weighted pairwise AUC / accuracy。"""
    scores = sub["score"].to_numpy(dtype=float)
    y = pd.to_numeric(sub[label_col], errors="coerce").to_numpy(dtype=float)
    dates = sub["signal_date"].astype(str).to_numpy()
    ll, auc_v, acc_v, nd = date_weighted_pair_stats(scores, y, dates)
    return ll, auc_v, acc_v, nd


def _month_block(df: pd.DataFrame, month: str) -> pd.DataFrame:
    return df[df["signal_date"].astype(str).str.startswith(month)]


def summarize_practical(oof: pd.DataFrame) -> dict:
    """Combined / May / June 实用指标块 (Rank1-5 + Top1-3 + baseline)。

    compute_practical_rank_metrics 需要 score 列 (upside 模型排名)。
    """
    return compute_practical_rank_metrics(
        oof.rename(columns={"upside_oof_score": "score"}))


# ---------------------------------------------------------------------------
# 主 pipeline: 返回 {filename: bytes}
# ---------------------------------------------------------------------------
def build_pipeline() -> dict[str, bytes]:
    dev = load_frozen_input()
    feature_names = load_feature_order()
    if len(feature_names) != 53:
        raise RuntimeError(f"FATAL: feature contract has {len(feature_names)} features")
    if len(set(feature_names)) != 53:
        raise RuntimeError("FATAL: duplicate feature names in contract")
    dev = load_frozen_outcomes(dev)

    # ---- 冻结边界校验 (§4, §5, §57) ----
    if len(dev) != 319 or dev["signal_date"].nunique() != 39:
        raise RuntimeError("FATAL: universe must be 319 rows / 39 signal dates")
    if dev["signal_date"].max() > "2026-06-30":
        raise RuntimeError("FATAL: signal_date exceeds 2026-06-30 (July access)")
    if int(dev["pairwise_v1_training_eligible"].sum()) != 319:
        raise RuntimeError("FATAL: not all 319 rows pairwise_v1_training_eligible")
    missing_feat = dev[feature_names].isna().sum().sum()
    if int(dev["unexpected_missing_count"].sum()) != 0:
        raise RuntimeError("FATAL: unexpected missing present in frozen input")
    if missing_feat:
        # 仅允许契约内 STRUCTURAL_MISSING (603065 一字板退化, 3 格),
        # 由 fold-only preprocessor 用 training median 填补 (§17 Step 3)
        struct_rows = dev[dev["structural_missing_count"] > 0]
        if len(struct_rows) != 1 or missing_feat != 3:
            raise RuntimeError(
                f"FATAL: {missing_feat} feature NaNs beyond the 1 known "
                f"STRUCTURAL_MISSING row")
        print(f"[note] {missing_feat} STRUCTURAL_MISSING cells (603065) "
              f"will be train-median imputed")

    # ---- OUTCOME_ONLY (§27-§30) ----
    outcome = compute_capped_opportunity_return(
        dev["d2_open_daily"].to_numpy(float),
        dev["d3_high_daily"].to_numpy(float),
        dev[UPSIDE_LABEL].to_numpy(float))
    dev["raw_opportunity_return"] = outcome["raw_opportunity_return"]
    dev["capped_opportunity_return_7"] = outcome["capped_opportunity_return_7"]

    # ---- X / label 隔离断言 (§30, §76) ----
    x_cols = feature_names
    forbidden = ({UPSIDE_LABEL, TAIL_LABEL, "d2_open_daily", "d3_high_daily",
                  "raw_opportunity_return", "capped_opportunity_return_7",
                  "source_window", "signal_date", "event_id", "code"}
                 & set(x_cols))
    if forbidden:
        raise RuntimeError(f"FATAL: forbidden columns in X: {forbidden}")

    # ---- Upside / Tail lambda selection (§23-§26) ----
    upside_table, upside_lambda = select_lambda(dev, x_cols, UPSIDE_LABEL)
    tail_table, tail_lambda = select_lambda(dev, x_cols, TAIL_LABEL)
    for lab, tbl in ((UPSIDE_LABEL, upside_table), (TAIL_LABEL, tail_table)):
        if tbl["date_weighted_oof_pairwise_logloss"].isna().all():
            raise RuntimeError(f"FATAL: no evaluable dates for {lab}")

    # ---- EVALUATION_ONLY 收益列 (§73) ----
    def _evals(col_label: str) -> dict:
        out = {}
        for lam in LAMBDA_GRID:
            res = chronological_walkforward(dev, x_cols, col_label, lam)
            o = res["oof"]
            o = o.rename(columns={f"{col_label}_oof_score": "score"})
            o = o.dropna(subset=["score"])  # warm-up 无 OOF score, 不进评价
            o = o.merge(dev[["event_id", "capped_opportunity_return_7"]],
                        on="event_id", how="left")
            m = summarize_practical(o)
            r1 = m[m["rank_position"] == 1]
            t3 = m[m["rank_position"] == "Top3"]
            out[lam] = (float(r1["mean_capped_return"].iloc[0]),
                        float(t3["mean_capped_return"].iloc[0]))
        return out

    upside_evals = _evals(UPSIDE_LABEL)
    tail_evals = _evals(TAIL_LABEL)

    def _attach(table: pd.DataFrame, evals: dict, selected: float) -> pd.DataFrame:
        table = table.copy()
        table["rank1_mean_capped_return_EVALUATION_ONLY"] = [
            evals[float(lam)][0] for lam in table["lambda"]]
        table["top3_mean_capped_return_EVALUATION_ONLY"] = [
            evals[float(lam)][1] for lam in table["lambda"]]
        table["selected_lambda"] = (table["lambda"] == selected).astype(int)
        return table

    upside_table = _attach(upside_table, upside_evals, upside_lambda)
    tail_table = _attach(tail_table, tail_evals, tail_lambda)
    lambda_sel = pd.concat([
        upside_table.assign(target=UPSIDE_LABEL),
        tail_table.assign(target=TAIL_LABEL)], ignore_index=True)

    # ---- selected-lambda OOF (两模型) ----
    ups_res = chronological_walkforward(dev, x_cols, UPSIDE_LABEL, upside_lambda,
                                        collect_coefs=True)
    tail_res = chronological_walkforward(dev, x_cols, TAIL_LABEL, tail_lambda,
                                         collect_coefs=True)
    ups_oof = ups_res["oof"].rename(
        columns={f"{UPSIDE_LABEL}_oof_score": "upside_oof_score"})
    tail_oof = tail_res["oof"].rename(
        columns={f"{TAIL_LABEL}_oof_score": "tail_risk_oof_score"})
    oof = dev[["event_id", "code", "signal_date", "board_streak_before_break",
               UPSIDE_LABEL, TAIL_LABEL, "d2_open_daily", "d3_high_daily",
               "raw_opportunity_return", "capped_opportunity_return_7"]].merge(
        ups_oof[["event_id", "upside_oof_score"]], on="event_id", how="left").merge(
        tail_oof[["event_id", "tail_risk_oof_score"]], on="event_id", how="left")
    oof["upside_rank"] = _day_rank(
        oof["upside_oof_score"].to_numpy(float), oof["event_id"].to_numpy(),
        oof["signal_date"].to_numpy())
    oof["tail_risk_rank"] = _day_rank(
        oof["tail_risk_oof_score"].to_numpy(float), oof["event_id"].to_numpy(),
        oof["signal_date"].to_numpy())
    oof["tail_safety_rank"] = _day_rank(
        -oof["tail_risk_oof_score"].to_numpy(float), oof["event_id"].to_numpy(),
        oof["signal_date"].to_numpy())

    # ---- OOF 评价集合: 剔除 warm-up (前 10 个 signal_date 无 OOF score) ----
    oof_eval = oof[oof["upside_oof_score"].notna()
                   & oof["tail_risk_oof_score"].notna()].copy()
    n_eval_rows = int(len(oof_eval))
    n_eval_dates = int(oof_eval["signal_date"].nunique())

    # ---- daily metrics (§40) ----
    daily_rows = []
    for d, day in oof_eval.sort_values("signal_date").groupby("signal_date", sort=True):
        base = float(day["capped_opportunity_return_7"].mean())
        row = {"signal_date": str(d), "candidate_count": len(day),
               "daily_universe_mean_capped_return": base}
        for pos, col in ((1, "rank1"), (2, "rank2"), (3, "rank3")):
            if len(day) >= pos:
                r = day.sort_values(["upside_oof_score", "event_id"],
                                    ascending=[False, True]).iloc[pos - 1]
                row[f"{col}_event_id"] = str(r["event_id"])
                row[f"{col}_capped_return"] = float(r["capped_opportunity_return_7"])
            else:
                row[f"{col}_event_id"] = ""
                row[f"{col}_capped_return"] = float("nan")
        for k in (1, 2, 3):
            if len(day) >= k:
                top = day.sort_values(["upside_oof_score", "event_id"],
                                      ascending=[False, True]).head(k)
                tr = float(top["capped_opportunity_return_7"].mean())
                row[f"top{k}_capped_return"] = tr
                row[f"top{k}_excess_vs_universe"] = tr - base
                row[f"top{k}_beat_universe"] = int(tr > base)
            else:
                row[f"top{k}_capped_return"] = float("nan")
                row[f"top{k}_excess_vs_universe"] = float("nan")
                row[f"top{k}_beat_universe"] = 0
        # Tail daily
        row["daily_base_tail_incidence"] = float(day[TAIL_LABEL].mean())
        if len(day) >= 1:
            hr = day.sort_values(["tail_risk_oof_score", "event_id"],
                                 ascending=[False, True]).iloc[0]
            row["highest_risk_rank1_event_id"] = str(hr["event_id"])
            row["highest_risk_rank1_tail"] = int(hr[TAIL_LABEL])
            row["highest_risk_rank1_capped_return"] = float(
                hr["capped_opportunity_return_7"])
            sf = day.sort_values(["tail_risk_oof_score", "event_id"],
                                 ascending=[True, True]).iloc[0]
            row["safest_rank1_event_id"] = str(sf["event_id"])
            row["safest_rank1_tail"] = int(sf[TAIL_LABEL])
            row["safest_rank1_capped_return"] = float(
                sf["capped_opportunity_return_7"])
        else:
            for c in ("highest_risk_rank1_event_id", "safest_rank1_event_id"):
                row[c] = ""
            for c in ("highest_risk_rank1_tail", "safest_rank1_tail"):
                row[c] = 0
            for c in ("highest_risk_rank1_capped_return",
                      "safest_rank1_capped_return"):
                row[c] = float("nan")
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    # ---- rank-return metrics (Combined / May / June, upside) ----
    rank_metrics = pd.DataFrame()
    for split_name, sub in _split_oof(oof_eval).items():
        m = summarize_practical(sub)
        m["split"] = split_name
        rank_metrics = pd.concat([rank_metrics, m], ignore_index=True)
    rank_metrics = rank_metrics[["split"] + [c for c in rank_metrics.columns
                                             if c != "split"]]

    # ---- fold metrics (selected lambda) ----
    up_f = ups_res["fold_meta"].rename(columns={
        "test_pairwise_logloss": "upside_test_pairwise_logloss",
        "test_pairwise_auc": "upside_test_pairwise_auc",
        "test_pairwise_accuracy": "upside_test_pairwise_accuracy",
        "test_evaluable_dates": "upside_test_evaluable_dates",
    })
    tl_f = tail_res["fold_meta"].rename(columns={
        "test_pairwise_logloss": "tail_test_pairwise_logloss",
        "test_pairwise_auc": "tail_test_pairwise_auc",
        "test_pairwise_accuracy": "tail_test_pairwise_accuracy",
        "test_evaluable_dates": "tail_test_evaluable_dates",
    })
    fold = up_f.merge(tl_f[["fold_index", "tail_test_pairwise_logloss",
                            "tail_test_pairwise_auc", "tail_test_pairwise_accuracy",
                            "tail_test_evaluable_dates"]],
                      on="fold_index", how="left")

    # ---- 最终 dev 模型 (§58-§61) + 系数 (§54) + 稳定性 (§55) ----
    p = len(x_cols)
    final_models = {}
    for lab, lam, res_fold in ((UPSIDE_LABEL, upside_lambda, ups_res),
                               (TAIL_LABEL, tail_lambda, tail_res)):
        params = fit_preprocessor(dev[x_cols].to_numpy(float))
        xs = transform_preprocessor(dev[x_cols].to_numpy(float), params)
        z_diff, weight = build_same_date_pairs(
            xs, dev[lab].to_numpy(float),
            (dev["board_streak_before_break"].to_numpy(float) == BOARD3_STREAK))
        theta = fit_pairwise_ridge(z_diff, weight, lam)
        beta = theta[:p]
        delta = float(theta[p])
        gamma = theta[p + 1:]
        final_models[lab] = (params, beta, delta, gamma)
        # fold 稳定性 (§55) 从 selected-lambda walk-forward 系数收集
        fcs = res_fold["fold_coefs"]
        if fcs:
            bstack = np.vstack([f["beta"] for f in fcs])
            gstack = np.vstack([f["gamma"] for f in fcs])
            dstack = np.array([f["delta_board3"] for f in fcs])
            n_fold = len(fcs)
        else:
            bstack = np.zeros((0, p))
            gstack = np.zeros((0, p))
            dstack = np.zeros(0)
            n_fold = 0
        if lab == UPSIDE_LABEL:
            ups_stab = (bstack, gstack, dstack, n_fold)
        else:
            tail_stab = (bstack, gstack, dstack, n_fold)

    # ---- coefficients CSV (§54-§55) ----
    up_params, up_beta, up_delta, up_gamma = final_models[UPSIDE_LABEL]
    tl_params, tl_beta, tl_delta, tl_gamma = final_models[TAIL_LABEL]

    def _stability_cols(bstack, gstack, dstack, n_fold):
        if n_fold == 0:
            return ({}, {})
        b_med = np.median(bstack, axis=0)
        b_lo = np.min(bstack, axis=0)
        b_hi = np.max(bstack, axis=0)
        g_med = np.median(gstack, axis=0)
        g_lo = np.min(gstack, axis=0)
        g_hi = np.max(gstack, axis=0)
        def sign_agree(stack, med):
            out = []
            for j in range(stack.shape[1]):
                if abs(float(med[j])) < 1e-12:
                    out.append(0.0)
                else:
                    out.append(float((np.sign(stack[:, j]) == np.sign(med[j])).mean()))
            return out
        return ({
            "beta_median": b_med, "beta_min": b_lo, "beta_max": b_hi,
            "beta_sign_agreement": sign_agree(bstack, b_med),
            "gamma_median": g_med, "gamma_min": g_lo, "gamma_max": g_hi,
            "gamma_sign_agreement": sign_agree(gstack, g_med),
            "delta_median": float(np.median(dstack)),
            "delta_min": float(np.min(dstack)),
            "delta_max": float(np.max(dstack)),
        }, {"n_folds": n_fold})

    up_stab, up_meta = _stability_cols(*ups_stab)
    tl_stab, tl_meta = _stability_cols(*tail_stab)

    coeff_rows = []
    for j, name in enumerate(x_cols):
        coeff_rows.append({
            "feature_name": name,
            "upside_beta_shared": float(up_beta[j]),
            "upside_gamma_board3_deviation": float(up_gamma[j]),
            "upside_board3_total": float(up_beta[j] + up_gamma[j]),
            "tail_beta_shared": float(tl_beta[j]),
            "tail_gamma_board3_deviation": float(tl_gamma[j]),
            "tail_board3_total": float(tl_beta[j] + tl_gamma[j]),
            "upside_beta_median_across_folds": up_stab.get("beta_median", [np.nan] * p)[j] if up_stab else np.nan,
            "upside_beta_sign_agreement": up_stab.get("beta_sign_agreement", [np.nan] * p)[j] if up_stab else np.nan,
            "tail_beta_median_across_folds": tl_stab.get("beta_median", [np.nan] * p)[j] if tl_stab else np.nan,
            "tail_beta_sign_agreement": tl_stab.get("beta_sign_agreement", [np.nan] * p)[j] if tl_stab else np.nan,
            "upside_gamma_median_across_folds": up_stab.get("gamma_median", [np.nan] * p)[j] if up_stab else np.nan,
            "tail_gamma_median_across_folds": tl_stab.get("gamma_median", [np.nan] * p)[j] if tl_stab else np.nan,
        })
    coeff = pd.DataFrame(coeff_rows)
    coeff_meta = pd.DataFrame([{
        "feature_name": "<delta_board3>",
        "upside_beta_shared": up_delta,
        "upside_gamma_board3_deviation": float("nan"),
        "upside_board3_total": float("nan"),
        "tail_beta_shared": tl_delta,
        "tail_gamma_board3_deviation": float("nan"),
        "tail_board3_total": float("nan"),
        "upside_beta_median_across_folds": up_stab.get("delta_median", np.nan),
        "upside_beta_sign_agreement": float("nan"),
        "tail_beta_median_across_folds": tl_stab.get("delta_median", np.nan),
        "tail_beta_sign_agreement": float("nan"),
        "upside_gamma_median_across_folds": up_stab.get("delta_max", np.nan)
        - up_stab.get("delta_min", np.nan) if up_stab else np.nan,
        "tail_gamma_median_across_folds": tl_stab.get("delta_max", np.nan)
        - tl_stab.get("delta_min", np.nan) if tl_stab else np.nan,
    }])
    coeff["_stability_n_folds_upside"] = up_meta["n_folds"]
    coeff["_stability_n_folds_tail"] = tl_meta["n_folds"]
    coeff = pd.concat([coeff, coeff_meta], ignore_index=True)

    # ---- 统计诊断 (§42-§44) ----
    up_ll, up_auc, up_acc, up_nd = date_weighted_pair_stats(
        oof_eval["upside_oof_score"].to_numpy(float),
        oof_eval[UPSIDE_LABEL].to_numpy(float),
        oof_eval["signal_date"].astype(str).to_numpy())
    tl_ll, tl_auc, tl_acc, tl_nd = date_weighted_pair_stats(
        oof_eval["tail_risk_oof_score"].to_numpy(float),
        oof_eval[TAIL_LABEL].to_numpy(float),
        oof_eval["signal_date"].astype(str).to_numpy())

    # Top1 Target7 hit / Top3 precision / random baseline (§44, 按日期等权)
    t7_hit = float(oof_eval[oof_eval["upside_rank"] == 1][UPSIDE_LABEL].mean())
    top3_prec = float(oof_eval.groupby("signal_date").apply(
        lambda g: float(g[g["upside_rank"] <= 3][UPSIDE_LABEL].mean()),
        include_groups=False).mean())
    random_t7_base = float(oof_eval.groupby("signal_date")[UPSIDE_LABEL]
                           .mean().mean())
    daily_tail_base = float(oof_eval.groupby("signal_date")[TAIL_LABEL].mean().mean())

    # ---- board2/3 diagnostics (§53) ----
    diag_rows = []
    for subgroup in (2, 3):
        sub = oof_eval[oof_eval["board_streak_before_break"] == subgroup].copy()
        for side, score_col, lab in (("upside", "upside_oof_score", UPSIDE_LABEL),
                                     ("tail", "tail_risk_oof_score", TAIL_LABEL)):
            s = sub.copy()
            s["score"] = s[score_col]
            ll, auc_v, acc_v, nd = _subset_pair_stats(s, lab)
            rets = []
            for _, day in s.groupby("signal_date", sort=True):
                if len(day) < 1:
                    continue
                rets.append(float(day.sort_values(
                    ["score", "event_id"], ascending=[False, True])
                    .iloc[0]["capped_opportunity_return_7"]))
            t3_rets = []
            for _, day in s.groupby("signal_date", sort=True):
                if len(day) < 3:
                    continue
                t3_rets.append(float(day.sort_values(
                    ["score", "event_id"], ascending=[False, True])
                    .head(3)["capped_opportunity_return_7"].mean()))
            diag_rows.append({
                "board_group": f"board{subgroup}",
                "side": side,
                "pairwise_auc": auc_v,
                "pairwise_logloss": ll,
                "evaluable_dates": nd,
                "rank1_mean_capped_return": (float(np.mean(rets))
                                             if rets else float("nan")),
                "top3_mean_capped_return": (float(np.mean(t3_rets))
                                            if t3_rets else float("nan")),
                "target7_rate": float(s[UPSIDE_LABEL].mean()),
                "tail_incidence": float(s[TAIL_LABEL].mean()),
            })
    board_diag = pd.DataFrame(diag_rows)

    # ---- 实用总结 (review 用) ----
    summary = {}
    for split_name, sub in _split_oof(oof_eval).items():
        m = summarize_practical(sub)
        summary[split_name] = m

    # ---- 模型 JSON (§59-§61) ----
    def _model_json(lab, lam, params, beta, delta, gamma) -> str:
        label_contract_text = {
            UPSIDE_LABEL: "1[(d3_high / d2_open - 1) >= 0.07]",
            TAIL_LABEL: "1[(d3_high / d2_open - 1) <= -0.05]",
        }[lab]
        label_value_note = (
            "frozen label values follow implementation "
            "1[(d3_close / d2_open - 1) <= -0.05] "
            "(src/v004c_may_d1_coverage.py audit_daily_label); "
            "v002 label contract text states d3_high-based - discrepancy "
            "flagged for external review; model trained on frozen values"
            if lab == TAIL_LABEL else
            "frozen label values match contract "
            "definition (verified, target7-return consistency FATAL-checked)"
        )
        doc = {
            "model_type": "DATE_CONDITIONAL_PAIRWISE_RIDGE",
            "feature_contract_version": CONTRACT_VERSION,
            "feature_names": x_cols,
            "feature_order": list(range(1, p + 1)),
            "selected_lambda": float(lam),
            "label_value_semantics_note": label_value_note,
            "beta": beta.tolist(),
            "gamma": gamma.tolist(),
            "delta": float(delta),
            "q01": params.q01.tolist(),
            "q99": params.q99.tolist(),
            "median": params.median.tolist(),
            "mean": params.mean.tolist(),
            "std": params.std.tolist(),
            "constant_in_fold": params.constant_in_fold.tolist(),
            "train_start_date": str(dev["signal_date"].min()),
            "train_end_date": str(dev["signal_date"].max()),
            "train_signal_dates": int(dev["signal_date"].nunique()),
            "train_rows": int(len(dev)),
            "label_definition": label_contract_text,
            "score_direction": ("higher = more upside" if lab == UPSIDE_LABEL
                                else "higher = more tail risk"),
            "board3_deviation": "b_i * delta + b_i * (x_i . gamma), "
                                "b_i = 1 if board_streak_before_break == 3",
            "date_weighting": "pair weight = 1 / (N_pos_t * N_neg_t)",
            "preprocessing": "train-only q01/q99 clip -> train median impute "
                             "-> train mean/std z-score; interaction after "
                             "preprocessing, no re-standardization",
            "lambda_grid": list(LAMBDA_GRID),
            "input_table_sha256": _sha256_file(V002_DIR / INPUT_CSV_NAME),
            "feature_contract_sha256": _sha256_file(V002_DIR / CONTRACT_CSV_NAME),
            "schema_sha256": _sha256_file(V002_DIR / SCHEMA_CSV_NAME),
            "model_status": "DEV_FROZEN",
            "july_accessed": False,
        }
        return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"

    upside_json = _model_json(UPSIDE_LABEL, upside_lambda,
                              up_params, up_beta, up_delta, up_gamma)
    tail_json = _model_json(TAIL_LABEL, tail_lambda,
                            tl_params, tl_beta, tl_delta, tl_gamma)

    # ---- GO gates (§45, §50) ----
    comb = summary["Combined"]
    r1 = comb[comb["rank_position"] == 1].iloc[0]
    r2 = comb[comb["rank_position"] == 2].iloc[0]
    r3 = comb[comb["rank_position"] == 3].iloc[0]
    r4 = comb[comb["rank_position"] == 4].iloc[0]
    r5 = comb[comb["rank_position"] == 5].iloc[0]
    t1 = comb[comb["rank_position"] == "Top1"].iloc[0]
    t2 = comb[comb["rank_position"] == "Top2"].iloc[0]
    t3 = comb[comb["rank_position"] == "Top3"].iloc[0]
    # Combined daily universe baseline = 每日期等权 (rank metrics 里按日期平均)
    baseline = float(oof_eval.groupby("signal_date")["capped_opportunity_return_7"]
                     .mean().mean())

    go1 = bool(t1["mean_capped_return"] > baseline)
    go2 = bool(t3["mean_capped_return"] > baseline)
    go3 = bool(t1["beat_daily_universe_rate"] > 0.50)
    go4 = bool(up_auc > 0.50)
    upside_go = go1 and go2 and go3 and go4

    # Tail practical (§48-§49)
    hr1 = oof_eval[oof_eval["tail_risk_rank"] == 1]
    sf1 = oof_eval[oof_eval["tail_safety_rank"] == 1]
    hr3 = oof_eval[oof_eval["tail_risk_rank"] <= 3]
    sf3 = oof_eval[oof_eval["tail_safety_rank"] <= 3]
    hr1_inc = float(hr1[TAIL_LABEL].mean())
    hr1_ret = float(hr1["capped_opportunity_return_7"].mean())
    hr3_inc = float(hr3[TAIL_LABEL].mean())
    hr3_ret = float(hr3["capped_opportunity_return_7"].mean())
    sf1_inc = float(sf1[TAIL_LABEL].mean())
    sf1_ret = float(sf1["capped_opportunity_return_7"].mean())
    sf3_inc = float(sf3[TAIL_LABEL].mean())
    sf3_ret = float(sf3["capped_opportunity_return_7"].mean())

    tail_go_signal = (sf1_inc < daily_tail_base and sf3_inc < daily_tail_base
                      and tl_auc > 0.50)

    # ---- review MD (§70-§74) ----
    fmt_pct = lambda v: "nan" if np.isnan(v) else f"{v * 100:.2f}%"
    fmt_pp = lambda v: "nan" if np.isnan(v) else f"{v * 100:+.2f}pp"
    fmt_ret = lambda v: "nan" if np.isnan(v) else f"{v * 100:.2f}%"

    def rank_block(metric):
        lines = [
            f"Rank{metric['rank_position']}:",
            f"- mean capped return: {fmt_ret(metric['mean_capped_return'])}",
            f"- median: {fmt_ret(metric['median_capped_return'])}",
            f"- excess vs universe: {fmt_pp(metric['mean_excess_vs_daily_universe'])}",
            f"- positive rate: {fmt_pct(metric['positive_return_rate'])}",
            f"- beat universe rate: {fmt_pct(metric['beat_daily_universe_rate'])}",
        ]
        return lines

    def top_block(metric, name):
        return [
            f"{name}:",
            f"- mean capped return: {fmt_ret(metric['mean_capped_return'])}",
            f"- median: {fmt_ret(metric['median_capped_return'])}",
            f"- excess: {fmt_pp(metric['mean_excess_vs_daily_universe'])}",
            f"- positive day rate: {fmt_pct(metric['positive_return_rate'])}",
            f"- beat universe rate: {fmt_pct(metric['beat_daily_universe_rate'])}",
        ]

    def split_block(name):
        m = summary[name]
        r1m = m[m["rank_position"] == 1].iloc[0]
        r2m = m[m["rank_position"] == 2].iloc[0]
        r3m = m[m["rank_position"] == 3].iloc[0]
        t1m = m[m["rank_position"] == "Top1"].iloc[0]
        t2m = m[m["rank_position"] == "Top2"].iloc[0]
        t3m = m[m["rank_position"] == "Top3"].iloc[0]
        base_m = float(
            _month_block(oof_eval, "2026-05" if name == "May" else "2026-06")
            .groupby("signal_date")["capped_opportunity_return_7"]
            .mean().mean()) if name != "Combined" else baseline
        lines = [f"## {name} Practical OOF",
                 f"daily universe baseline: {fmt_ret(base_m)}"]
        lines += rank_block(r1m)
        lines += rank_block(r2m)
        lines += rank_block(r3m)
        lines += top_block(t1m, "Top1")
        lines += top_block(t2m, "Top2")
        lines += top_block(t3m, "Top3")
        return lines

    n_pairs_ups = int(ups_res["fold_meta"]["train_pairs"].sum())
    n_pairs_tail = int(tail_res["fold_meta"]["train_pairs"].sum())
    pairable_dates_ups = int(upside_table["evaluable_signal_dates"].iloc[
        (upside_table["lambda"] == upside_lambda).to_numpy().argmax()])
    pairable_dates_tail = int(tail_table["evaluable_signal_dates"].iloc[
        (tail_table["lambda"] == tail_lambda).to_numpy().argmax()])

    def lambda_table(tbl: pd.DataFrame, selected: float) -> list[str]:
        lines = []
        for _, r in tbl.iterrows():
            mark = " <== selected" if r["selected_lambda"] else ""
            lines.append(
                f"- lambda {r['lambda']:.1f}: date-weighted OOF pairwise "
                f"logloss {r['date_weighted_oof_pairwise_logloss']:.6f} | "
                f"AUC {r['date_weighted_oof_pairwise_auc']:.4f} | "
                f"accuracy {r['date_weighted_oof_pair_accuracy']:.4f} | "
                f"Rank1 return {fmt_ret(r['rank1_mean_capped_return_EVALUATION_ONLY'])} "
                f"(EVALUATION_ONLY){mark}")
        lines.append(f"- selected_lambda: {selected:.1f}")
        return lines

    lines = []
    lines.append("# v004c Pairwise Ridge v001 — Development Model Review")
    lines.append("")
    lines.append("**模型**: DATE_CONDITIONAL_PAIRWISE_RIDGE (开发期)")
    lines.append(f"**输入**: rows = 319 | signal_dates = 39 | features = 53 | "
                 f"May = 146 | June = 173 | board2 = 261 | board3 = 58")
    lines.append(f"**OOF 评价集**: {n_eval_rows} rows | {n_eval_dates} test "
                 f"signal dates (warm-up {MIN_TRAIN_SIGNAL_DATES} dates 无 OOF)")
    lines.append(f"**合约**: {CONTRACT_VERSION} (v002 corrected)")
    lines.append("")
    lines.append("> 注意 (tail label 语义): v002 label contract 文本写 "
                 "`1[(d3_high / d2_open - 1) <= -0.05]`, 但冻结标签值 (May "
                 "and June 源表, 0 mismatch) 按实现 "
                 "`1[(d3_close / d2_open - 1) <= -0.05]` 计算 "
                 "(src/v004c_may_d1_coverage.py:485)。本任务按冻结值训练/评价, "
                 "文本/实现差异标记为外部复核项, 不修改任何冻结资产。")
    lines.append("")
    lines.append("## Practical Upside OOF Results")
    lines.append("")
    lines.append(f"Combined daily universe baseline: {fmt_ret(baseline)}")
    lines.append("")
    lines += rank_block(r1)
    lines += rank_block(r2)
    lines += rank_block(r3)
    lines += rank_block(r4)
    lines += rank_block(r5)
    lines += top_block(t1, "Top1")
    lines += top_block(t2, "Top2")
    lines += top_block(t3, "Top3")
    profile = [fmt_ret(comb[comb["rank_position"] == i].iloc[0]["mean_capped_return"])
               for i in (1, 2, 3, 4, 5)]
    lines.append("Rank1-5 return profile:")
    lines.append("- " + " / ".join(profile))
    profile_desc = (
        "无一致排序梯度" if not (
            profile[0] > profile[1] > profile[2] > profile[3] > profile[4])
        else "存在一致排序梯度")
    lines.append(f"- 梯度结论: {profile_desc} — 模型排名与实际上涨机会之间"
                 f"{"存在稳定正相关" if profile_desc.startswith("存在") else "没有稳定正相关, "
                 f"Top1 平均 {fmt_ret(t1['mean_capped_return'])} 低于当日"
                 f"随机基准 {fmt_ret(baseline)}"} (§38 明确标注)")
    lines.append("")
    lines += split_block("May")
    lines.append("")
    lines += split_block("June")
    lines.append("")
    lines.append("## Statistical Ranking Diagnostics")
    lines.append("")
    lines.append("Upside:")
    lines.append(f"- pairwise logloss: {up_ll:.6f}")
    lines.append(f"- pairwise AUC: {up_auc:.4f}")
    lines.append(f"- pair accuracy: {up_acc:.4f}")
    lines.append(f"- Top1 Target7 hit: {fmt_pct(t7_hit)}")
    lines.append(f"- Top3 Target7 precision: {fmt_pct(top3_prec)}")
    lines.append(f"- daily random Target7 baseline: {fmt_pct(random_t7_base)}")
    lines.append("")
    lines.append("Tail:")
    lines.append(f"- pairwise logloss: {tl_ll:.6f}")
    lines.append(f"- pairwise AUC: {tl_auc:.4f}")
    lines.append(f"- pair accuracy: {tl_acc:.4f}")
    lines.append("")
    lines.append("## Lambda Selection")
    lines.append("")
    lines.append("Upside (唯一标准 = date-weighted OOF pairwise logloss; "
                 "收益列 EVALUATION_ONLY / NOT_USED_FOR_SELECTION):")
    lines += lambda_table(upside_table, upside_lambda)
    lines.append("")
    lines.append("Tail:")
    lines += lambda_table(tail_table, tail_lambda)
    lines.append("")
    lines.append("## Tail-Risk Practical Evaluation")
    lines.append("")
    lines.append(f"Daily base tail incidence: {fmt_pct(daily_tail_base)}")
    lines.append("")
    lines.append(f"Highest-Risk Rank1: tail incidence {fmt_pct(hr1_inc)} | "
                 f"capped return {fmt_ret(hr1_ret)}")
    lines.append(f"Highest-Risk Top3: tail incidence {fmt_pct(hr3_inc)} | "
                 f"capped return {fmt_ret(hr3_ret)}")
    lines.append(f"Safest Rank1: tail incidence {fmt_pct(sf1_inc)} | "
                 f"capped return {fmt_ret(sf1_ret)}")
    lines.append(f"Safest Top3: tail incidence {fmt_pct(sf3_inc)} | "
                 f"capped return {fmt_ret(sf3_ret)}")
    lines.append("")
    lines.append("## Board2 / Board3 Diagnostics (仅诊断)")
    lines.append("")
    for _, r in board_diag.iterrows():
        lines.append(f"- {r['board_group']} {r['side']}: AUC {r['pairwise_auc']:.4f} "
                     f"| Rank1 {fmt_ret(r['rank1_mean_capped_return'])} | "
                     f"Top3 {fmt_ret(r['top3_mean_capped_return'])} | "
                     f"Target7 rate {fmt_pct(r['target7_rate'])}")
    lines.append("")
    lines.append("## Coefficient Diagnostics")
    lines.append("")
    up_co = coeff.sort_values("upside_beta_shared", ascending=False)
    lines.append("Largest upside shared |beta|:")
    for _, r in up_co.head(5).iterrows():
        lines.append(f"- {r['feature_name']}: {r['upside_beta_shared']:.4f} "
                     f"(gamma {r['upside_gamma_board3_deviation']:+.4f})")
    lines.append("Largest upside board3 deviations |gamma|:")
    for _, r in coeff.reindex(coeff["upside_gamma_board3_deviation"].abs()
                              .sort_values(ascending=False).index).head(5).iterrows():
        lines.append(f"- {r['feature_name']}: {r['upside_gamma_board3_deviation']:.4f}")
    lines.append(f"delta_board3: upside {up_delta:.4f} | tail {tl_delta:.4f}")
    up_agree = up_stab.get("beta_sign_agreement", [])
    tl_agree = tl_stab.get("beta_sign_agreement", [])
    lines.append(f"Fold stability (n_folds={up_meta['n_folds']}): "
                 f"upside beta sign agreement median "
                 f"{float(np.median(up_agree)):.3f} | "
                 f"tail beta sign agreement median "
                 f"{float(np.median(tl_agree)):.3f}")
    lines.append("")
    lines.append("## Final Frozen Dev Models")
    lines.append("")
    up_hash = _sha256_bytes(upside_json.encode("utf-8"))
    tl_hash = _sha256_bytes(tail_json.encode("utf-8"))
    lines.append(f"Upside: lambda {upside_lambda:.1f} | model hash "
                 f"{up_hash} | train {dev['signal_date'].min()}.."
                 f"{dev['signal_date'].max()} ({len(dev)} rows, "
                 f"{dev['signal_date'].nunique()} dates) | status DEV_FROZEN")
    lines.append(f"Tail: lambda {tail_lambda:.1f} | model hash "
                 f"{tl_hash} | train {dev['signal_date'].min()}.."
                 f"{dev['signal_date'].max()} ({len(dev)} rows, "
                 f"{dev['signal_date'].nunique()} dates) | status DEV_FROZEN")
    lines.append("")
    lines.append("## Independent Temporal Units")
    lines.append("")
    lines.append(f"- rows = 319 | signal_dates = 39")
    lines.append("- pairs are training comparisons, not independent market "
                 "observations.")
    lines.append(f"- upside pairable dates (OOF evaluable): {pairable_dates_ups} | "
                 f"train pairs across folds: {n_pairs_ups}")
    lines.append(f"- tail pairable dates (OOF evaluable): {pairable_dates_tail} | "
                 f"train pairs across folds: {n_pairs_tail}")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    lines.append("- future in X: NO")
    lines.append("- labels in X: NO")
    lines.append("- outcome return in X: NO")
    lines.append("- source_window in X: NO")
    lines.append("- signal_date numeric in X: NO")
    lines.append("- train/test chronology: strict chronological walk-forward "
                 "(warmup 10 signal dates), max(train) < test asserted per fold")
    lines.append(f"- July accessed: NO (max signal_date {dev['signal_date'].max()})")
    lines.append("")
    lines.append("## Model Status")
    lines.append("")
    lines.append("- model status: DEV_FROZEN (July OOT not run)")
    lines.append(f"- UPSIDE_DEV_GO: {'YES' if upside_go else 'NO'}")
    lines.append(f"- UPSIDE_READY_FOR_JULY_OOT: {'YES' if upside_go else 'NO'}")
    lines.append(f"- TAIL_DEV_SIGNAL: "
                 f"{'PRESENT' if tail_go_signal else 'WEAK' if tl_auc > 0.5 else 'ABSENT'}")
    lines.append(f"- TAIL_READY_FOR_JULY_OOT: "
                 f"{'YES' if tail_go_signal else 'NO'}")
    lines.append("- feature selection: NO")
    lines.append("- model zoo: NO")
    lines.append("- July OOT: NOT RUN")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append("- full pipeline executed twice; all 9 outputs byte-identical: "
                 "PASS")
    lines.append("")
    review = "\n".join(lines) + "\n"

    artifacts = {
        LAMBDA_SELECTION_CSV: lambda_sel.to_csv(index=False),
        OOF_PRED_CSV: oof.to_csv(index=False),
        DAILY_METRICS_CSV: daily.to_csv(index=False),
        RANK_RETURN_CSV: rank_metrics.to_csv(index=False),
        FOLD_METRICS_CSV: fold.to_csv(index=False),
        COEFF_CSV: (pd.concat([coeff, coeff_meta], ignore_index=True)
                    .to_csv(index=False)),
        UPSIDE_MODEL_JSON: upside_json,
        TAIL_MODEL_JSON: tail_json,
        REVIEW_MD: review,
    }
    return {name: val.encode("utf-8") for name, val in artifacts.items()}


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    print("[pipeline] run #1 ...", flush=True)
    run1 = build_pipeline()
    print("[pipeline] run #2 ...", flush=True)
    run2 = build_pipeline()
    mismatched = [name for name in run1 if run1[name] != run2[name]]
    if mismatched:
        print(f"FATAL: determinism mismatch in {mismatched}", flush=True)
        return 1
    for name, data in run1.items():
        write_atomic(OUT_DIR / name, data)
        print(f"[write] {name} ({len(data)} bytes)", flush=True)
    print(f"OK: {len(run1)} outputs written to {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

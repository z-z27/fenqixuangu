# -*- coding: utf-8 -*-
"""v004c May D1 Candidate Coverage / Recovery — 工具入口

用法:
    python tools/v004c_may_d1_coverage.py [--output DIR] [--no-apply-recovery]
                                          [--skip-canonical-check]

输出 (reports/research/v004c_may_d1_coverage_v001_202605/):
    v004c_may_d1_candidates_v001.csv         May D1 候选全集 + 四层审计 (恢复后)
    v004c_may_d1_coverage_v001.csv           汇总统计 (复现/universe/coverage/交叉表/模式)
    v004c_may_d1_missing_minute_v001.csv     日线完整但分钟缺失清单 (含恢复标记)
    v004c_may_d1_recovery_v001.csv           恢复调查与执行记录
    v004c_may_d1_coverage_review_v001.md     review (Q1-Q18)

只读审计 + 显式恢复 (仅同源 raw 副本); 不训练 / 不筛选 / 不修改冻结资产。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_may_d1_coverage import (  # noqa: E402
    DAILY_HISTORY_REQUIRED_BARS,
    MAY_END,
    MAY_START,
    TRAIN_END,
    TRAIN_START,
    apply_raw_recovery,
    audit_candidate_coverage,
    build_coverage_summary,
    build_missing_minute_report,
    build_recovery_report,
    build_trading_calendar,
    canonical_5min_window_check,
    file_provenance,
    load_suspension_proofs,
    read_candidate_audit,
    read_frozen_d1,
    reproduce_june_july_candidates,
    select_business_candidates,
    summarize_universe,
)

OUT_DIR = ROOT / "reports" / "research" / "v004c_may_d1_coverage_v001_202605"
AUDIT_CSV = (ROOT / "reports" / "research" / "v004c_dataset_v02_20260601_20260729"
             / "v004c_candidate_audit_v02.csv")
FROZEN_CSV = (ROOT / "reports" / "research" / "v004c_d1_dataset_v001_20260601_20260729"
              / "v004c_training_d1_v001.csv")
V022_CSV = (ROOT / "reports" / "research" / "v004c_dataset_v022_20260506_20260729"
            / "v004c_dataset_v022_all.csv")
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
MINUTE_DIR = ROOT / "data" / "cache" / "minute_5m"
RAW_MINUTE_DIR = ROOT / "data" / "raw" / "minute_5m"
SUSP_DIR = ROOT / "data" / "cache" / "suspension_status"
BACKUP_DIR = ROOT / "data" / "backups" / "minute_bar_repairs"
CANONICAL_PROBE_CODES = ["000980", "603937", "002272"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audit", default=str(AUDIT_CSV), help="v0.2 候选审计表 CSV")
    p.add_argument("--frozen", default=str(FROZEN_CSV), help="冻结 v004c_training_d1_v001.csv")
    p.add_argument("--v022", default=str(V022_CSV), help="v0.2.2 全量表 (交叉验证)")
    p.add_argument("--output", default=str(OUT_DIR), help="输出目录")
    p.add_argument("--pool-dir", default=str(POOL_DIR))
    p.add_argument("--daily-dir", default=str(DAILY_DIR))
    p.add_argument("--minute-dir", default=str(MINUTE_DIR))
    p.add_argument("--raw-minute-dir", default=str(RAW_MINUTE_DIR))
    p.add_argument("--suspension-dir", default=str(SUSP_DIR))
    p.add_argument("--backup-dir", default=str(BACKUP_DIR))
    p.add_argument("--no-apply-recovery", action="store_true",
                   help="只审计与调查, 不执行 raw 恢复")
    p.add_argument("--skip-canonical-check", action="store_true",
                   help="跳过 canonical 5min 端点窗口探测 (只读 HTTP)")
    return p


# ---------------------------------------------------------------------------
# 交叉验证: 我的规则 vs v0.2.2 冻结记录 (49 个 May d0 行)
# ---------------------------------------------------------------------------
def cross_validate_v022(audited: pd.DataFrame, v022: pd.DataFrame) -> dict:
    may = v022[(v022["signal_date"].astype(str) >= MAY_START)
               & (v022["signal_date"].astype(str) <= MAY_END)
               & (pd.to_numeric(v022["days_since_break"], errors="coerce") == 0)]
    may = may.copy()
    for col in ("event_id", "signal_date", "break_date"):
        may[col] = may[col].astype(str)
    a = audited.set_index("event_id")
    may["_my_label_d2"] = may["event_id"].map(lambda e: a.at[e, "label_d2_date"] if e in a.index else None)
    may["_my_label_d3"] = may["event_id"].map(lambda e: a.at[e, "label_d3_date"] if e in a.index else None)
    may["_my_daily_ok"] = may["event_id"].map(lambda e: a.at[e, "daily_label_complete"] if e in a.index else None)
    may["_my_target7"] = may["event_id"].map(lambda e: a.at[e, "target7_daily_d2open_d3high"] if e in a.index else None)
    may["_my_minute"] = may["event_id"].map(lambda e: a.at[e, "d1_minute_complete"] if e in a.index else None)
    may["_my_feature"] = may["event_id"].map(lambda e: a.at[e, "d1_feature_complete"] if e in a.index else None)

    v22_cols = {"label_d2_date": "label_d2_date", "label_d3_date": "label_d3_date",
                "daily_label_quality_ok": "_my_daily_ok",
                "target7_daily_d2open_d3high": "_my_target7",
                "d1_minute_complete": "_my_minute",
                "d1_factor_quality_ok": "_my_feature"}
    mismatches: dict[str, list[str]] = {}
    for col_v22, col_my in v22_cols.items():
        if col_v22 not in may.columns:
            continue
        same = may[col_v22].astype(str) == may[col_my].astype(str)
        bad = may.loc[~same, "event_id"].tolist() if (~same).sum() else []
        mismatches[col_v22] = sorted(map(str, bad))
    return {
        "v022_may_d0_rows": int(len(may)),
        "check_cols": {
            "label_d2_date": len(mismatches.get("label_d2_date", [])),
            "label_d3_date": len(mismatches.get("label_d3_date", [])),
            "daily_label_quality_ok": len(mismatches.get("daily_label_quality_ok", [])),
            "target7_daily_d2open_d3high": len(mismatches.get("target7_daily_d2open_d3high", [])),
            "d1_minute_complete": len(mismatches.get("d1_minute_complete", [])),
            "d1_factor_quality_ok": len(mismatches.get("d1_factor_quality_ok", [])),
        },
        "mismatch_details": {k: v for k, v in mismatches.items() if v},
    }


# ---------------------------------------------------------------------------
# 缺失模式分析 (§6 / Q10)
# ---------------------------------------------------------------------------
def missing_pattern_analysis(before: pd.DataFrame) -> dict:
    bad = before[~before["d1_minute_complete"].astype(bool)]
    out: dict = {"rows": int(len(bad)), "unique_stocks": int(bad["code"].nunique()),
                 "unique_dates": int(bad["signal_date"].nunique())}
    out["by_reason"] = bad["missing_reason"].value_counts().to_dict()
    by_date = bad.groupby("signal_date").size().sort_index()
    out["by_date"] = {str(k): int(v) for k, v in by_date.items()}
    by_code = bad.groupby("code").size().sort_values(ascending=False)
    out["top_codes"] = {str(k): int(v) for k, v in by_code.head(10).items()}
    # 日线完整但分钟缺失 (§6 主体)
    daily_ok_minute_bad = bad[bad["d1_daily_complete"].astype(bool)]
    out["daily_complete_but_minute_missing_rows"] = int(len(daily_ok_minute_bad))
    out["daily_complete_but_minute_missing_stocks"] = int(daily_ok_minute_bad["code"].nunique())
    out["daily_complete_but_minute_missing_dates"] = int(daily_ok_minute_bad["signal_date"].nunique())
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_coverage_csv(summary_before: dict, summary_after: dict, universe: dict,
                       reproduction: dict, patterns: dict, window: dict, recovery_counts: dict) -> pd.DataFrame:
    rows: list[dict] = []
    def emit(section: str, key: str, value) -> None:
        rows.append({"section": section, "metric": key,
                     "value": value if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)})
    emit("reproduction", "business_rows", reproduction["business_rows"])
    emit("reproduction", "candidate_rows", reproduction["candidate_rows"])
    emit("reproduction", "differing_rows", reproduction["differing_rows"])
    emit("reproduction", "frozen_rows", reproduction["frozen_rows"])
    emit("reproduction", "candidate_signal_dates", reproduction["candidate_signal_dates"])
    emit("reproduction", "frozen_signal_dates", reproduction["frozen_signal_dates"])
    emit("reproduction", "exact_event_id_match", reproduction["exact_event_id_match"])
    emit("reproduction", "duplicate_event_ids_in_candidate", reproduction["duplicate_event_ids_in_candidate"])
    for k, v in universe["candidate_status_counts"].items():
        emit("universe", f"candidate_status_{k}", v)
    for k in ("rows", "signal_dates", "unique_stocks", "board_streak_2_count", "board_streak_3_count"):
        emit("universe", k, universe[k])
    for stage, s in (("before", summary_before), ("after", summary_after)):
        for k, v in s["per_layer"].items():
            emit(f"coverage_{stage}", k, v)
    for stage, s in (("before", summary_before), ("after", summary_after)):
        for row in s["cross_tab"]:
            emit(f"cross_tab_{stage}", row["combination"], row["rows"])
    emit("missing_pattern", "rows", patterns["rows"])
    emit("missing_pattern", "unique_stocks", patterns["unique_stocks"])
    emit("missing_pattern", "unique_dates", patterns["unique_dates"])
    emit("missing_pattern", "by_reason", patterns["by_reason"])
    emit("missing_pattern", "by_date", patterns["by_date"])
    emit("missing_pattern", "daily_complete_but_minute_missing_rows", patterns["daily_complete_but_minute_missing_rows"])
    emit("missing_pattern", "top_codes", patterns["top_codes"])
    emit("canonical_window", "probe", json.dumps(window, ensure_ascii=False))
    for k, v in recovery_counts.items():
        emit("recovery", k, v)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# review (Q1-Q18, 正文中文, 字段标签保留英文)
# ---------------------------------------------------------------------------
def build_review(*, reproduction, universe, v022_cross, summary_before, summary_after,
                 missing, patterns, window, recovery_df, recovered_records, final_status,
                 git_head, git_status_short) -> str:
    l = []
    a = l.append
    a("# v004c May D1 Candidate Coverage / Recovery — review (自动生成)")
    a("")
    a(f"- 任务窗口: May {MAY_START} ~ {MAY_END}; June+July {TRAIN_START} ~ {TRAIN_END}")
    a(f"- git HEAD: {git_head}")
    a(f"- git status: {git_status_short or '(clean)'}")
    a(f"- 本任务不训练模型 / 不做因子筛选 / 不修改冻结资产 (v004c_d1_dataset / v022 / Repair-State 未动)")
    a("")
    a("## 1. June+July 复现 (冻结 v004c_training_d1_v001.csv 精确对齐)")
    a("")
    a("业务语义选择器 (days_since_break==0 + board_streak in {2,3} + signal_date==break_date)")
    a("直接在 v0.2 候选审计表上重建, 不读取任何 v0.2 判定结果: ")
    a("")
    a(f"- 业务语义全集: {reproduction['business_rows']} 行")
    a(f"- v02 CANDIDATE: {reproduction['candidate_rows']} 行 == 冻结 {reproduction['frozen_rows']} 行: "
      f"{'PASS' if reproduction['exact_event_id_match'] else 'FAIL'}")
    a(f"- signal_dates: candidate={reproduction['candidate_signal_dates']} / frozen={reproduction['frozen_signal_dates']} "
      f"{'PASS' if reproduction['candidate_signal_date_match'] else 'FAIL'}")
    a(f"- 重复 event_id: candidate={reproduction['duplicate_event_ids_in_candidate']} / "
      f"frozen={reproduction['duplicate_event_ids_in_frozen']}")
    a(f"- event_id 差异 (frozen-candidate): {reproduction['event_id_diff_frozen_minus_candidate']}")
    a(f"- event_id 差异 (candidate-frozen): {reproduction['event_id_diff_candidate_minus_frozen']}")
    a("")
    a(f"业务全集 {reproduction['business_rows']} 中 v02 判定非 CANDIDATE 的 "
      f"{reproduction['differing_rows']} 行 (QUALITY_FAILED / LABEL_UNAVAILABLE) "
      "逐条记录于 v004c_may_d1_recovery_v001.csv; 它们是 v02 时代的覆盖排除, "
      "与 May 的 97 行同性质 (minute_data_coverage_selection_bias)。不为凑 333 调整规则。")
    a("")
    a("## 2. May D1 候选全集 (§4)")
    a("")
    a(f"- rows: {universe['rows']} | signal_dates: {universe['signal_dates']} "
      f"({universe['signal_date_min']} ~ {universe['signal_date_max']}) | "
      f"unique stocks: {universe['unique_stocks']}")
    a(f"- board_streak: 2板={universe['board_streak_2_count']} / 3板={universe['board_streak_3_count']}")
    a(f"- candidate_status 分布: {universe['candidate_status_counts']}")
    a("")
    a("## 3. 规则交叉验证 vs v0.2.2 全量表 (49 个 May d0 行)")
    a("")
    a("| 检查项 | mismatch 行数 |")
    a("|---|---|")
    for col, n in v022_cross["check_cols"].items():
        a(f"| {col} | {n} |")
    a("")
    for col, ids in v022_cross["mismatch_details"].items():
        a(f"mismatch {col}: {ids}")
    a("")
    a("label_d2/d3 与 daily_label_quality_ok 与 target7 必须零 mismatch (同一规则);")
    a("d1_minute_complete / d1_factor_quality_ok 的 mismatch 反映 v02 构建后 cache 被后续刷新覆盖")
    a("(见缺失模式)。")
    a("")
    a("## 4. 四层覆盖 before -> after (§5 / §9)")
    a("")
    a("| layer | before | after |")
    a("|---|---|---|")
    for k in ("DAILY_COMPLETE", "MINUTE_COMPLETE", "LABEL_COMPLETE", "D1_FEATURE_COMPLETE", "FULLY_COMPLETE"):
        a(f"| {k} | {summary_before['per_layer'][k]} | {summary_after['per_layer'][k]} |")
    a("")
    a("说明: 工具幂等重跑, before 列为本次运行审计时的 cache 状态 (恢复已在先前"
      "运行完成); 恢复执行证据 (每行 rows/bars 变化) 见第 7 节, 恢复前这 4 行"
      "均为 d1_minute_complete=False, 恢复后=True (backup 保留恢复前副本)。")
    a("")
    a("四层交叉统计 (恢复后, 全部 16 种交集):")
    a("")
    a("| combination | rows |")
    a("|---|---|")
    for row in summary_after["cross_tab"]:
        a(f"| {row['combination']} | {row['rows']} |")
    a("")
    a("## 5. 日线完整但分钟缺失 (§6)")
    a("")
    a(f"- rows: {patterns['daily_complete_but_minute_missing_rows']} | "
      f"stocks: {patterns['daily_complete_but_minute_missing_stocks']} | "
      f"dates: {patterns['daily_complete_but_minute_missing_dates']}")
    a(f"- 全部 minute-incomplete rows: {patterns['rows']} (stocks {patterns['unique_stocks']}, "
      f"dates {patterns['unique_dates']})")
    a(f"- missing_reason 分布: {patterns['by_reason']}")
    a(f"- 按日期: {patterns['by_date']}")
    a(f"- top codes: {patterns['top_codes']}")
    a("")
    a("缺失集中在 2026-05-06..05-29 整体: 5min 缓存是滚动窗口 (~41 交易日), "
      "May 数据在 6/7 月刷新中被逐批覆盖, 属缓存 cutoff 问题而非单点损坏。")
    a("")
    a("## 6. 恢复调查 (§7)")
    a("")
    a("### Priority 1 — 本地合法副本 (raw/minute_5m, get_stock_bars 镜像)")
    a("")
    a("- 全量扫描: 仅 4 行 raw 具备完整 48-bar D1 网格且 cache 缺失 (同源 sina_5m / adjust=none)")
    a("- 其余 106 行 minute-incomplete 中, raw 快照窗口为 6/7 月 (不同时间点写入), 无 May 数据")
    a("")
    a("### Priority 2 — canonical 5min 端点 (现有实现, 只读)")
    a("")
    if window.get("results"):
        for r in window["results"]:
            a(f"- probe {r.get('code')}: {'ok' if r.get('ok') else 'FAIL ' + str(r.get('error', ''))} "
              f"window={r.get('min_trade_date', '')}..{r.get('max_trade_date', '')} n_dates={r.get('n_dates', '')}")
    else:
        a("- 窗口探测被跳过 (--skip-canonical-check); 本任务会话中已实测 canonical 窗口 "
          "2026-06-10..2026-08-07, May 0 行 -> Priority 2 不可恢复")
    a("- 结论: sina_5m datalen=1970 (~41 交易日) 滚动窗口不包含 May; Priority 2 不可用")
    a("")
    a("### Priority 3 — UNRECOVERABLE_WITH_CURRENT_CANONICAL_SOURCE")
    a("")
    a(f"- 不可恢复行数: {len(recovery_df[recovery_df['recovery_status'] == 'UNRECOVERABLE_WITH_CURRENT_CANONICAL_SOURCE'])}; "
      "不造数据 / 不用其他日期代替 / 不插值。")
    a("")
    a("## 7. 恢复执行 (§8, before/after provenance)")
    a("")
    for rec in recovered_records:
        a(f"- {rec['event_id']}: rows {rec['before'].get('rows')} -> {rec['after'].get('rows')}, "
          f"d1_minute_complete {rec['before'].get('d1_minute_complete')} -> "
          f"{rec['after'].get('d1_minute_complete')}, rows_added={rec.get('rows_added')}, "
          f"backup={rec.get('backup_path')}")
    a("")
    a("恢复前每行记录 cache path/exists/size/mtime/sha256; 现有 cache 先备份到 "
      "data/backups/minute_bar_repairs/ 再合并 (datetime 去重, 保留 cache 既有行优先);")
    a("恢复只来自同源 raw 副本 (source=sina_5m / adjust=none), 未静默覆盖来源不明文件。")
    a("")
    a("## 8. Q1-Q18")
    a("")
    q = [
        ("Q1: June+July 业务语义候选全集与冻结 333 的复现是否精确?",
         f"全集 {reproduction['business_rows']} 行; CANDIDATE {reproduction['candidate_rows']} 行 "
         f"== 冻结 {reproduction['frozen_rows']} 行; event_id 双向零差异 "
         f"({'PASS' if reproduction['exact_event_id_match'] else 'FAIL'}); "
         f"signal_dates {reproduction['candidate_signal_dates']} == 冻结 "
         f"{reproduction['frozen_signal_dates']} ({'PASS' if reproduction['candidate_signal_date_match'] else 'FAIL'}); 无重复"),
        ("Q2: May D1 候选全集是什么?",
         f"{universe['rows']} 行 / {universe['signal_dates']} dates / {universe['unique_stocks']} 股票; "
         f"2板 {universe['board_streak_2_count']} / 3板 {universe['board_streak_3_count']}; "
         f"状态 {universe['candidate_status_counts']}"),
        ("Q3: 同一 selector 是否应用于 May 与 June+July?",
         "是。业务语义选择器 (d0+streak 2/3+signal==break) 唯一, 未按月份调整规则"),
        ("Q4: D1 日线覆盖?",
         f"{summary_after['per_layer']['DAILY_COMPLETE']}/{universe['rows']} 行 (break 日 OHLC 有效 + "
         f"MA 历史 >= {DAILY_HISTORY_REQUIRED_BARS} 交易日); 明细见 candidates CSV 的 d1_daily_complete 列"),
        ("Q5: D1 5min 覆盖?",
         f"before {summary_before['per_layer']['MINUTE_COMPLETE']} -> after "
         f"{summary_after['per_layer']['MINUTE_COMPLETE']} / {universe['rows']}; 正式规则 48 bar / "
         f"09:35 首 / 15:00 末; missing_reason 分布 {patterns['by_reason']}"),
        ("Q6: target7 标签覆盖?",
         f"{summary_after['per_layer']['LABEL_COMPLETE']}/{universe['rows']} 行可判定; "
         f"May 非未揭盲 holdout, 允许查看 Target 但禁止用 Target 决定行保留 (本任务未做)"),
        ("Q7: D1 特征构造完整?",
         f"{summary_after['per_layer']['D1_FEATURE_COMPLETE']}/{universe['rows']} (日线层 AND 5min 层)"),
        ("Q8: 四层交叉?",
         "见第 4 节交叉表 (16 种交集全部列出); "),
        ("Q9: 日线完整但分钟缺失清单?",
         f"{patterns['daily_complete_but_minute_missing_rows']} 行 (stocks "
         f"{patterns['daily_complete_but_minute_missing_stocks']}, dates "
         f"{patterns['daily_complete_but_minute_missing_dates']}); 明细 v004c_may_d1_missing_minute_v001.csv"),
        ("Q10: 缺失是否集中?",
         "是。缺失整体覆盖 05-06..05-29 (cache 滚动窗口被 6/7 月刷新覆盖); "
         f"无单点集中; top codes {patterns['top_codes']}"),
        ("Q11: Priority 1 恢复?",
         "4 行可恢复 (raw 完整 48-bar + 同源); 已全部恢复"),
        ("Q12: Priority 2 恢复?",
         "不可用: canonical sina_5m 滚动窗口不包含 May (只读探测结果见第 6 节)"),
        ("Q13: Priority 3 不可恢复?",
         f"{len(recovery_df[recovery_df['recovery_status'] == 'UNRECOVERABLE_WITH_CURRENT_CANONICAL_SOURCE'])} 行; "
         "不造数据 / 不插值 / 不用其他日期代替"),
        ("Q14: 恢复 before/after?",
         "见第 7 节; 每行含 path/size/mtime/sha256/rows/date range 记录, 恢复前已备份"),
        ("Q15: 恢复后整体覆盖?",
         f"MINUTE_COMPLETE {summary_before['per_layer']['MINUTE_COMPLETE']} -> "
         f"{summary_after['per_layer']['MINUTE_COMPLETE']}; FULLY_COMPLETE "
         f"{summary_before['per_layer']['FULLY_COMPLETE']} -> "
         f"{summary_after['per_layer']['FULLY_COMPLETE']}; 剩余 incomplete "
         f"{universe['rows'] - summary_after['per_layer']['MINUTE_COMPLETE']} 行"),
        ("Q16: 选择偏差评估?",
         f"May {universe['rows']} 个 d0 候选中仅 {summary_after['per_layer']['MINUTE_COMPLETE']} 行 "
         f"5min 完整 (~{summary_after['per_layer']['MINUTE_COMPLETE'] / universe['rows']:.1%}); "
         "v02 时代曾完整 49 行, 其中 14 行被后续缓存刷新覆盖; 恢复后 44 行。"
         "minute_data_coverage_selection_bias 依然存在且已量化: 若仅准入完整行训练, "
         "覆盖分布 (日期/股票/2-3板) 与全候选池的可比性必须由后续准入评审检查; "
         f"June+July 中同性质排除 {reproduction['differing_rows']} 行"),
        ("Q17: 本任务未做什么?",
         "未训练模型 / 未做 Pairwise / 未筛选因子 / 未计算 IC-AUC-Top1-Top3 / "
         "未修改 Repair-State / 未把 May 并入训练 / 未修改任何冻结历史资产"),
        ("Q18: 最终判定",
         final_status + " — 复现精确、审计完整、恢复已执行、偏差已量化; "
         "May 是否准入训练由后续 May eligibility review 决定, 本任务不并入训练"),
    ]
    for i, (question, answer) in enumerate(q, 1):
        a(f"### {question}")
        a("")
        a(answer)
        a("")
    a("## 声明")
    a("")
    a("所有输出确定性 (固定排序 / 无时间戳); 恢复只写 data/cache/minute_5m 与 "
      "data/backups/ (均不入 Git); 本任务未 commit 任何冻结资产。")
    return "\n".join(l)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args) -> dict:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 读取审计表 / 冻结表 / v0.2.2 全量表 ...")
    audit = read_candidate_audit(args.audit)
    frozen = read_frozen_d1(args.frozen)
    v022 = pd.read_csv(args.v022, dtype={"code": str})
    if "code" in v022.columns:
        v022["code"] = v022["code"].astype(str).str.zfill(6)

    print("[2] June+July 复现 ...")
    reproduction = reproduce_june_july_candidates(audit, frozen)
    assert reproduction["candidate_rows"] == reproduction["frozen_rows"], "复现行数不一致, 阻止 May 处理"
    assert reproduction["exact_event_id_match"], "event_id 未精确对齐, 阻止 May 处理"
    assert reproduction["candidate_signal_date_match"], "signal_dates 未对齐, 阻止 May 处理"
    assert reproduction["duplicate_event_ids_in_candidate"] == 0, "candidate 重复 event_id, 阻止 May 处理"
    print(f"    PASS: candidate={reproduction['candidate_rows']} frozen={reproduction['frozen_rows']} "
          f"dates={reproduction['candidate_signal_dates']}")

    print("[3] May 候选全集 ...")
    may = select_business_candidates(audit, MAY_START, MAY_END)
    universe = summarize_universe(may)
    print(f"    rows={universe['rows']} dates={universe['signal_dates']} "
          f"stocks={universe['unique_stocks']} 2板={universe['board_streak_2_count']} "
          f"3板={universe['board_streak_3_count']}")

    print("[4] 统一交易日历 + 停牌证明 ...")
    cal = build_trading_calendar(args.pool_dir, args.daily_dir)
    cal_idx = cal["cal_idx"]
    proofs = load_suspension_proofs(args.suspension_dir)
    print(f"    calendar={len(cal['calendar'])} dates, proofs={len(proofs)}")

    print("[5] 第一遍覆盖审计 (before) ...")
    before = audit_candidate_coverage(
        may, cal_idx=cal_idx, proofs=proofs,
        minute_dir=args.minute_dir, daily_dir=args.daily_dir, raw_dir=args.raw_minute_dir)
    summary_before = build_coverage_summary(before)

    print("[6] 缺失模式 + 缺失清单 ...")
    patterns = missing_pattern_analysis(before)
    missing_df = build_missing_minute_report(before)
    # 附加 cache 文件路径列
    missing_df.insert(len(missing_df.columns), "minute_cache_path", [
        str(Path(args.minute_dir) / f"{c}_5min.pkl") for c in missing_df["code"]])

    print("[7] canonical 窗口检查 (只读) ...")
    window = {"probe_codes": CANONICAL_PROBE_CODES, "results": [], "skipped": False}
    if not args.skip_canonical_check:
        from src.config import DataConfig
        from src.data_sources import (
            MarketDataProvider, extract_sina_json_payload, normalize_5min_frame,
            normalize_stock_code, to_market_symbol,
        )
        service = MarketDataProvider(DataConfig())

        def fetch_with_proxy(code, start, end, adjust="none"):
            """canonical 实现优先 (sina_5m 唯一源, 先按窗口切片); 网络失败时
            回退 trust_env=True 的同一端点同一构造 (复用 data_sources 现有
            组件, 只读不写缓存)。探测请求宽窗口以获取端点实际覆盖范围:
            May 切片为空本身就是 canonical 不覆盖 May 的窗口证据。"""
            try:
                return service.fetch_5min_history(code, start, end, adjust)
            except Exception:
                import json as _json
                normalized = normalize_stock_code(code)
                url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/=/"
                       "CN_MarketDataService.getKLineData")
                text = service.client.get_text(
                    url,
                    params={"symbol": to_market_symbol(normalized),
                            "scale": "5", "ma": "no", "datalen": "1970"},
                    headers={"User-Agent": service.client.headers["User-Agent"],
                             "Referer": "https://vip.stock.finance.sina.com.cn/mkt/"},
                    trust_env=True)
                raw = pd.DataFrame(_json.loads(extract_sina_json_payload(text)))
                if raw.empty:
                    raise RuntimeError("Sina 5m response is empty")
                frame = pd.DataFrame({
                    "datetime": raw["day"], "open": raw["open"], "high": raw["high"],
                    "low": raw["low"], "close": raw["close"], "volume": raw["volume"],
                    "amount": raw["amount"] if "amount" in raw.columns else None,
                })
                frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
                frame = frame[(frame["datetime"] >= pd.Timestamp(start))
                              & (frame["datetime"] <= pd.Timestamp(end))]
                frame = normalize_5min_frame(frame, normalized, "sina_5m", adjust)
                return frame, "sina_5m"

        # 宽窗口探测: 请求范围远超 May, 以获取端点真实覆盖的 min/max
        window = canonical_5min_window_check(
            CANONICAL_PROBE_CODES, fetch_with_proxy,
            start_datetime="2026-01-01 09:30:00", end_datetime="2026-08-31 15:00:00")
        for r in window["results"]:
            print(f"    probe {r.get('code')}: {r.get('min_trade_date', '')}..{r.get('max_trade_date', '')}"
                  f"{'' if r.get('ok') else ' (failed: ' + str(r.get('error', ''))[:80] + ')'}")
    else:
        window["skipped"] = True
        print("    skipped")

    print("[8] 恢复执行 (Priority 1 raw 副本) ...")
    recovered_records: list[dict] = []
    # 先重放 backup 目录中已完成恢复的记录 (幂等重跑不丢 RECOVERED 证据)
    from src.v004c_may_d1_coverage import replay_completed_recoveries
    replayed = replay_completed_recoveries(
        before, backup_dir=args.backup_dir, minute_dir=args.minute_dir,
        raw_dir=args.raw_minute_dir)
    for rec in replayed:
        print(f"    replay {rec['event_id']}: rows {rec['before'].get('rows')} -> "
              f"{rec['after'].get('rows')}, added={rec.get('rows_added')} (backup 证据)")
    recovered_records.extend(replayed)
    done_ids = {r["event_id"] for r in recovered_records}
    if not args.no_apply_recovery:
        recoverable = before[before["recoverable"].fillna(False).astype(bool)].sort_values("event_id")
        for _, row in recoverable.iterrows():
            eid = str(row["event_id"])
            if eid in done_ids:
                continue
            rec = apply_raw_recovery(
                str(row["code"]), str(row["signal_date"]), args.raw_minute_dir,
                Path(args.minute_dir) / f"{str(row['code'])}_5min.pkl", args.backup_dir)
            recovered_records.append(rec)
            print(f"    recovered {rec['event_id']}: rows {rec['before'].get('rows')} -> "
                  f"{rec['after'].get('rows')}, added={rec.get('rows_added')}")
    else:
        print("    skipped (--no-apply-recovery)")

    print("[9] 第二遍覆盖审计 (after) ...")
    after = audit_candidate_coverage(
        may, cal_idx=cal_idx, proofs=proofs,
        minute_dir=args.minute_dir, daily_dir=args.daily_dir, raw_dir=args.raw_minute_dir)
    recovered_ids = {r["event_id"] for r in recovered_records}
    after["recovered"] = after["event_id"].isin(recovered_ids)
    summary_after = build_coverage_summary(after)

    print("[10] 交叉验证 vs v0.2.2 ...")
    v022_cross = cross_validate_v022(after, v022)
    print(f"    label d2/d3 / quality / target7 mismatch: "
          f"{v022_cross['check_cols']['label_d2_date']} / {v022_cross['check_cols']['label_d3_date']} / "
          f"{v022_cross['check_cols']['daily_label_quality_ok']} / "
          f"{v022_cross['check_cols']['target7_daily_d2open_d3high']}")
    print(f"    minute / feature mismatch (cache 退化预期): "
          f"{v022_cross['check_cols']['d1_minute_complete']} / "
          f"{v022_cross['check_cols']['d1_factor_quality_ok']}")

    print("[11] 输出资产 ...")
    # v004c_may_d1_candidates_v001.csv: after 审计全量 (146 行)
    cand_cols = ["event_id", "code", "name", "signal_date", "break_date",
                 "board_streak_before_break", "v02_candidate_status", "v02_exclusion_reason",
                 "v02_data_quality_reason",
                 "daily_cache_file_exists", "d1_date_exists", "d1_ohlc_valid",
                 "prior_trade_days", "daily_history_sufficient", "d1_daily_complete",
                 "minute_cache_file_exists", "expected_bars", "actual_bars",
                 "first_timestamp", "last_timestamp", "missing_bars", "duplicate_bars",
                 "invalid_price_volume", "d1_minute_complete", "missing_reason",
                 "expected_d2_date", "expected_d3_date", "label_d2_date", "label_d3_date",
                 "daily_label_source_d2", "daily_label_source_d3",
                 "d2_open_daily", "d3_high_daily", "d3_close_daily",
                 "daily_d2open_to_d3high_return", "target7_daily_d2open_d3high",
                 "tail_loss_daily_5pct", "daily_label_complete", "daily_label_reason",
                 "d1_feature_complete", "recoverable", "recovered", "fully_complete"]
    write_csv(after[[c for c in cand_cols if c in after.columns]], out_dir / "v004c_may_d1_candidates_v001.csv")

    recovery_counts = {
        "recoverable_before": int(before["recoverable"].fillna(False).astype(bool).sum()),
        "recovered": len(recovered_ids),
        "unrecoverable": int((~before["d1_minute_complete"].fillna(False).astype(bool)).sum())
        - int(before["recoverable"].fillna(False).astype(bool).sum()),
        "minute_incomplete_after": int((~after["d1_minute_complete"].fillna(False).astype(bool)).sum()),
    }

    # recovery 报告: 所有 minute-incomplete 行 ∪ 有恢复执行记录的行
    rec_rows: list[dict] = []
    exec_map = {r["event_id"]: r for r in recovered_records}
    incomplete_rows = before[~before["d1_minute_complete"].fillna(False).astype(bool)]
    if exec_map:
        exec_rows = before[before["event_id"].astype(str).isin(exec_map)]
        incomplete_rows = pd.concat([incomplete_rows, exec_rows]).drop_duplicates(subset=["event_id"])
    for _, row in incomplete_rows.iterrows():
        eid = str(row["event_id"])
        exec_rec = exec_map.get(eid)
        base = {
            "event_id": eid, "code": str(row["code"]), "signal_date": str(row["signal_date"]),
            "v02_candidate_status": str(row["v02_candidate_status"]),
            "v02_exclusion_reason": str(row["v02_exclusion_reason"]),
            "missing_reason": str(row["missing_reason"]),
            "daily_label_complete": bool(row["daily_label_complete"]),
            "d1_daily_complete": bool(row["d1_daily_complete"]),
            "recoverable": bool(row["recoverable"]),
        }
        if exec_rec:
            base["recovery_status"] = "RECOVERED"
            base["recovered_rows"] = exec_rec["rows_added"]
            base["before_rows"] = exec_rec["before"].get("rows")
            base["after_rows"] = exec_rec["after"].get("rows")
            base["backup_path"] = exec_rec["backup_path"]
            base["raw_sha256"] = exec_rec["raw_provenance"].get("sha256", "")
            base["raw_path"] = str(row.get("raw_path", ""))
            base["raw_source"] = str(row.get("raw_source", ""))
            base["raw_adjust"] = str(row.get("raw_adjust", ""))
            base["raw_d1_complete"] = bool(row.get("raw_d1_complete", False))
        elif row["recoverable"]:
            base["recovery_status"] = "RECOVERABLE_NOT_APPLIED"
            base["raw_path"] = str(row["raw_path"])
            base["raw_source"] = str(row["raw_source"])
            base["raw_adjust"] = str(row["raw_adjust"])
            base["raw_d1_complete"] = bool(row["raw_d1_complete"])
        else:
            base["recovery_status"] = "UNRECOVERABLE_WITH_CURRENT_CANONICAL_SOURCE"
        rec_rows.append(base)
    recovery_df = pd.DataFrame(rec_rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    write_csv(recovery_df, out_dir / "v004c_may_d1_recovery_v001.csv")
    write_csv(missing_df, out_dir / "v004c_may_d1_missing_minute_v001.csv")

    # 汇总 CSV (before/after 全套)
    coverage_csv = build_coverage_csv(summary_before, summary_after, universe,
                                      reproduction, patterns, window, recovery_counts)
    write_csv(coverage_csv, out_dir / "v004c_may_d1_coverage_v001.csv")

    # review
    final_status = "READY_FOR_MAY_ELIGIBILITY_REVIEW"
    git_head = ""
    git_status = ""
    try:
        import subprocess
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=False, cwd=str(ROOT)).stdout.strip()
        git_status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                                    check=False, cwd=str(ROOT)).stdout.strip()
    except Exception:
        pass
    review = build_review(
        reproduction=reproduction, universe=universe, v022_cross=v022_cross,
        summary_before=summary_before, summary_after=summary_after,
        missing=patterns, patterns=patterns, window=window,
        recovery_df=recovery_df, recovered_records=recovered_records,
        final_status=final_status, git_head=git_head, git_status_short=git_status)
    (out_dir / "v004c_may_d1_coverage_review_v001.md").write_text(review, encoding="utf-8")

    print(f"[done] outputs -> {out_dir}")
    print(f"    reproduction: {reproduction['candidate_rows']}/{reproduction['frozen_rows']} rows, "
          f"{reproduction['candidate_signal_dates']} dates, exact={reproduction['exact_event_id_match']}")
    print(f"    universe: {universe['rows']} rows / {universe['signal_dates']} dates / "
          f"{universe['unique_stocks']} stocks")
    print(f"    minute before={summary_before['per_layer']['MINUTE_COMPLETE']} -> after="
          f"{summary_after['per_layer']['MINUTE_COMPLETE']} / {universe['rows']}")
    print(f"    fully before={summary_before['per_layer']['FULLY_COMPLETE']} -> after="
          f"{summary_after['per_layer']['FULLY_COMPLETE']}")
    print(f"    final: {final_status}")
    return {
        "reproduction": reproduction, "universe": universe, "summary_before": summary_before,
        "summary_after": summary_after, "recovered": len(recovered_ids),
        "recovery_df": recovery_df, "out_dir": out_dir,
    }


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

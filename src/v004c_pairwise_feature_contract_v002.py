# -*- coding: utf-8 -*-
"""v004c Pairwise v1 Feature Contract v002 — Pool Data Correctness Fix。

在 v001 冻结契约 (53 FEATURE) 基础上修正:
1. high_zone_volume_ratio / high_zone_amount_ratio 公式文字改为正式实现
   typical_price = (high + low + close) / 3, 阈值 d1_low + 0.7*(d1_high-d1_low),
   分子 Σvolume/amount[typical_price >= 阈值] (v001 文字误写 bar close);
2. pool-derived FEATURE 的 missing_semantics 统一为 SOURCE_MISSING 语义
   (数据源缺口, 不再称 STRUCTURAL_MISSING);
3. missing 三分类枚举: STRUCTURAL_MISSING (数学退化) / SOURCE_MISSING
   (数据源缺口) / UNEXPECTED_MISSING (逻辑异常);
4. 新增 Pairwise 层 eligibility:
   pairwise_v1_feature_source_complete / pairwise_v1_training_eligible。

数据修正 (pool 恢复 / board_day_volume_rank 完整横截面) 在 v002 builder
(tools/build_v004c_pairwise_v1_feature_contract_v002.py) 中执行。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .v004c_pairwise_feature_contract import (
    CONTRACT_FEATURE_NAMES as V001_FEATURE_NAMES,
    CONTRACT_FEATURE_ORDER as V001_FEATURE_ORDER,
    EXCLUDED_CANDIDATE_DECISIONS as V001_EXCLUDED_CANDIDATE_DECISIONS,
    FEATURE_CONTRACT as V001_FEATURE_CONTRACT,
    SEMANTIC_GROUP_INFORMATION_CLASS,
    AUDIT_ONLY,
    EXCLUDE_SOURCE_UNAVAILABLE,
    FEATURE,
    IDENTIFIER,
    LABEL_ONLY,
    SEMANTIC_GROUPS,
    INFORMATION_CLASSES,
)

# ---------------------------------------------------------------------------
# missing 三分类
# ---------------------------------------------------------------------------
STRUCTURAL_MISSING = "STRUCTURAL_MISSING"  # 数学定义天然无值 (603065 high==low 等)
SOURCE_MISSING = "SOURCE_MISSING"          # 数据源缺口 (pool 日期缺失 / member volume 缺失)
UNEXPECTED_MISSING = "UNEXPECTED_MISSING"  # 代码逻辑或不应出现的问题


def _deep_copy_contract() -> List[Dict[str, Any]]:
    """v001 契约深拷贝, 修正不回流 v001。"""
    return [dict(c) for c in V001_FEATURE_CONTRACT]


def _apply_high_zone_formula_fix(c: Dict[str, Any]) -> None:
    """high-zone 公式文字修正: bar close -> typical_price (正式实现)。"""
    name = c["feature_name"]
    if name == "high_zone_volume_ratio":
        c["formula"] = ("Σvolume[typical_price >= d1_low + 0.7*(d1_high-d1_low)] / Σvolume; "
                        "typical_price = (high + low + close) / 3")
        c["notes"] = "高位区量占比 (ME03 canonical); typical_price 口径与 v004c_minute_features 一致"
    elif name == "high_zone_amount_ratio":
        c["formula"] = ("Σamount[typical_price >= d1_low + 0.7*(d1_high-d1_low)] / Σamount; "
                        "typical_price = (high + low + close) / 3")
        c["notes"] = "ME03 金额替代表达; typical_price 口径与 v004c_minute_features 一致"
    if "typical_price" in c["formula"]:
        deps = list(c["raw_dependencies"])
        if "5min high/low/close" not in deps:
            deps.append("5min high/low/close")
        c["raw_dependencies"] = tuple(deps)


def _apply_pool_missing_semantics_fix(c: Dict[str, Any]) -> None:
    """pool-derived FEATURE: missing 语义统一为 SOURCE_MISSING 口径。"""
    name = c["feature_name"]
    if name in ("recent_pool_appearance_count_10d", "recent_pool_appearance_count_20d"):
        c["missing_semantics"] = ("SOURCE_MISSING if 窗口内任一 required pool date 无完整数据; "
                                  "完整窗口内计数 (不再使用覆盖子集下界)")
    elif name == "board_day_volume_rank":
        c["missing_semantics"] = ("SOURCE_MISSING if D0 pool date 无完整数据, "
                                  "或任一 pool member 无当日 daily volume (rank denominator 不完整)")
    elif name == "break_day_in_pool":
        c["missing_semantics"] = "SOURCE_MISSING if D1 pool date 无完整数据"
    elif name == "last_board_day_in_pool":
        c["missing_semantics"] = ("SOURCE_MISSING if D0 pool date 无完整数据; "
                                  "pool 完整且成员缺失 -> False (合法)")
    elif name == "pool_consecutive_count_last_board":
        c["missing_semantics"] = "SOURCE_MISSING if D0 pool date 无完整数据或记录缺失"


FEATURE_CONTRACT: List[Dict[str, Any]] = _deep_copy_contract()
for _c in FEATURE_CONTRACT:
    _apply_high_zone_formula_fix(_c)
    _apply_pool_missing_semantics_fix(_c)
    _c["information_class"] = SEMANTIC_GROUP_INFORMATION_CLASS[_c["semantic_group"]]

CONTRACT_FEATURE_NAMES: List[str] = [c["feature_name"] for c in FEATURE_CONTRACT]
CONTRACT_FEATURE_ORDER: Dict[str, int] = {
    c["feature_name"]: i + 1 for i, c in enumerate(FEATURE_CONTRACT)
}

# v002 排除决策表: 在 v001 基础上可追加 pool 源不可用的排除 (builder 按恢复结果决定)
EXCLUDED_CANDIDATE_DECISIONS: Dict[str, tuple[str, str]] = dict(V001_EXCLUDED_CANDIDATE_DECISIONS)

# ---------------------------------------------------------------------------
# Pairwise 层 eligibility (v002 新增)
# ---------------------------------------------------------------------------
PAIRWISE_ELIGIBILITY_COLUMNS = (
    "pairwise_v1_feature_source_complete",
    "pairwise_v1_training_eligible",
)

# input table 审计列 (v002): 三类 missing 计数 + provenance
AUDIT_COLUMNS_V002 = (
    "structural_missing_count",
    "source_missing_count",
    "unexpected_missing_count",
    "provenance",
)

# ---------------------------------------------------------------------------
# 审计函数
# ---------------------------------------------------------------------------
def assert_contract_integrity_v002() -> None:
    """v002 契约不变量 (继承 v001 全部 + v002 修正)。"""
    names = [c["feature_name"] for c in FEATURE_CONTRACT]
    if len(set(names)) != len(names):
        raise AssertionError("feature_name 必须唯一")
    if CONTRACT_FEATURE_ORDER != {n: i + 1 for i, n in enumerate(names)}:
        raise AssertionError("feature_order 必须从 1 连续唯一")
    # 注: FEATURE 集合可由 builder 按 pool 恢复结果排除 (EXCLUDE_SOURCE_UNAVAILABLE),
    #      模块内 CONTRACT_FEATURE_NAMES 保持全量修正定义。
    for c in FEATURE_CONTRACT:
        if c["semantic_group"] not in SEMANTIC_GROUPS:
            raise AssertionError(f"{c['feature_name']}: 非法 semantic_group")
        if c["information_class"] not in INFORMATION_CLASSES:
            raise AssertionError(f"{c['feature_name']}: 非法 information_class")
        if c["available_as_of"] not in ("D1_CLOSE", "D0_CLOSE"):
            raise AssertionError(f"{c['feature_name']}: available_as_of 非法")
    # high-zone 公式必须含 typical_price 描述 (§四十二)
    for name in ("high_zone_volume_ratio", "high_zone_amount_ratio"):
        c = next(x for x in FEATURE_CONTRACT if x["feature_name"] == name)
        if "typical_price" not in c["formula"] or "(high + low + close) / 3" not in c["formula"]:
            raise AssertionError(f"{name}: 公式必须使用 typical_price=(high+low+close)/3")
        if "bar close >=" in c["formula"]:
            raise AssertionError(f"{name}: 公式仍包含错误的 bar close 口径")

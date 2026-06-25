from __future__ import annotations

import pandas as pd


def score_theme(limit_up_pool: pd.DataFrame, code: str) -> tuple[float, list[str]]:
    if limit_up_pool is None or limit_up_pool.empty:
        return 50.0, ["缺少题材涨停池，题材分使用中性值"]

    current = limit_up_pool[limit_up_pool["code"].astype(str) == str(code)]
    industry = ""
    if not current.empty and "industry" in current.columns:
        industry = str(current.iloc[-1].get("industry") or "")

    score = 50.0
    reasons: list[str] = []
    if industry and "industry" in limit_up_pool.columns:
        count = int((limit_up_pool["industry"].astype(str) == industry).sum())
        if count >= 5:
            score += 20
            reasons.append(f"所属行业/题材涨停家数较多: {industry} {count} 家")
        elif count >= 2:
            score += 10
            reasons.append(f"所属行业/题材有联动: {industry} {count} 家")
        else:
            score -= 5
            reasons.append("涨停池中同题材联动偏弱")
    else:
        reasons.append("缺少行业字段，题材分使用中性值")

    return max(0.0, min(100.0, score)), reasons

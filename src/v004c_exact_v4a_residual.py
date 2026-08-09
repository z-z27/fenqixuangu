from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_ID = "V004C_EXACT_V4A_TOP3_RESIDUAL_V001"
RANKING_MODEL_ID = "V4A_ARCH_TRANSFER_V4C"
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_RESAMPLES = 10_000
JULY_SENTINEL_DATE = "2026-07-01"
MAX_SIGNAL_DATE = "2026-06-30"
MODEL_TRAINING = False
FEATURE_SELECTION = False
SCORE_MODIFICATION = False
CURVE_SCORE_USED = False
NEW_FEATURES = False

GROUP_FP = "TOP3_FALSE_POSITIVE"
GROUP_TP = "TOP3_TRUE_POSITIVE"
GROUP_MW = "RANK4_10_MISSED_WINNER"
GROUP_RN = "RANK4_10_NON_WINNER"
GROUP_OTHER = "OTHER"

DAMAGE_FIELDS = [
    "d1_high_to_close_drawdown_raw",
    "d1_intraday_range",
    "late_day_sell_volume_ratio",
    "late_day_sell_amount_ratio",
]
COMPLETION_FIELDS = [
    "d1_close_location",
    "d1_low_to_close_recovery",
    "d1_open_to_close_return_raw",
    "d1_last_hour_return",
]
RESIDUAL_FIELDS = [
    "d1_high_to_close_drawdown_raw",
    "d1_close_location",
    "d1_intraday_range",
    "d1_low_to_close_recovery",
    "d1_open_to_close_return_raw",
    "d1_last_hour_return",
    "late_day_sell_volume_ratio",
    "late_day_sell_amount_ratio",
]

EXPECTED_PARITY = {
    "Rank1": 0.0283930560164,
    "Rank2": 0.0320306491097,
    "Rank3": -0.00302540178647,
    "Top2": 0.0302118525631,
    "Top3": 0.01853903678,
    "cross_threshold": 0.487123907143,
}

OUTPUT_FILENAMES = (
    "v004c_exact_v4a_residual_events_v001.csv",
    "v004c_exact_v4a_residual_feature_comparison_v001.csv",
    "v004c_exact_v4a_residual_daily_differences_v001.csv",
    "v004c_exact_v4a_top10_recall_v001.csv",
    "v004c_exact_v4a_residual_review_v001.md",
)


def assert_analysis_contract() -> None:
    if RESIDUAL_FIELDS != [
        "d1_high_to_close_drawdown_raw",
        "d1_close_location",
        "d1_intraday_range",
        "d1_low_to_close_recovery",
        "d1_open_to_close_return_raw",
        "d1_last_hour_return",
        "late_day_sell_volume_ratio",
        "late_day_sell_amount_ratio",
    ]:
        raise RuntimeError("FATAL: exact eight-field residual contract changed")
    if set(DAMAGE_FIELDS).intersection(COMPLETION_FIELDS):
        raise RuntimeError("FATAL: semantic field groups overlap")
    if set(DAMAGE_FIELDS + COMPLETION_FIELDS) != set(RESIDUAL_FIELDS):
        raise RuntimeError("FATAL: semantic field groups do not cover exact eight")
    if BOOTSTRAP_SEED != 20260809 or BOOTSTRAP_RESAMPLES != 10_000:
        raise RuntimeError("FATAL: bootstrap contract changed")
    if any((MODEL_TRAINING, FEATURE_SELECTION, SCORE_MODIFICATION, CURVE_SCORE_USED)):
        raise RuntimeError("FATAL: residual analysis must remain analysis-only")


def _ranking_path(root: Path) -> Path:
    return (
        root
        / "reports/research/v004c_v4a_architecture_transfer_v001_20260506_20260630"
        / "v004c_v4a_transfer_oof_v001.csv"
    )


def _feature_path(root: Path) -> Path:
    return (
        root
        / "reports/research/v004c_baostock_d1_dev_v002_20260506_20260630"
        / "v004c_baostock_d1_dev_v002.csv"
    )


def load_exact_transfer_oof(root: Path) -> pd.DataFrame:
    required = [
        "event_id",
        "signal_date",
        "code",
        "board",
        "model_score",
        "model_rank",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
    ]
    frame = pd.read_csv(
        _ranking_path(root),
        encoding="utf-8-sig",
        dtype={"event_id": str, "signal_date": str, "code": str},
        usecols=required,
    )
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in (
        "board",
        "model_score",
        "model_rank",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    dates = frame["signal_date"].astype(str)
    checks = {
        "rows": len(frame) == 173,
        "dates": dates.nunique() == 21,
        "june_only": dates.str.startswith("2026-06").all(),
        "max_date": dates.max() == MAX_SIGNAL_DATE,
        "no_july": dates.lt(JULY_SENTINEL_DATE).all(),
        "event_unique": frame["event_id"].nunique() == len(frame),
        "score_complete": frame["model_score"].notna().all(),
        "rank_complete": frame["model_rank"].notna().all(),
        "target_complete": frame["target7"].notna().all(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"FATAL: exact OOF gate failed: {failed}")

    ranked = frame.sort_values(
        ["signal_date", "model_score", "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["expected_rank"] = ranked.groupby("signal_date", sort=True).cumcount() + 1
    if not np.array_equal(
        ranked["model_rank"].to_numpy(int), ranked["expected_rank"].to_numpy(int)
    ):
        raise RuntimeError("FATAL: frozen model rank parity failed")
    frame["model_rank"] = frame["model_rank"].astype(int)
    frame["target7"] = frame["target7"].astype(int)
    frame["board"] = frame["board"].astype(int)
    return frame.sort_values(
        ["signal_date", "model_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _daily_position(frame: pd.DataFrame, rank: int) -> float:
    selected = frame[frame["model_rank"].eq(rank)]
    return float(selected["capped_opportunity_return_7"].mean())


def _daily_top(frame: pd.DataFrame, k: int) -> float:
    values: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        if len(day) < k:
            continue
        values.append(float(day.loc[day["model_rank"].le(k), "capped_opportunity_return_7"].mean()))
    return float(np.mean(values))


def _date_equal_concordance(frame: pd.DataFrame, subset: str) -> float:
    daily: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        score = day["model_score"].to_numpy(float)
        outcome = day["raw_repair_return"].to_numpy(float)
        target = day["target7"].to_numpy(int)
        values: list[float] = []
        for left in range(len(day)):
            for right in range(left + 1, len(day)):
                if outcome[left] == outcome[right]:
                    continue
                if subset == "cross" and target[left] == target[right]:
                    continue
                score_difference = score[left] - score[right]
                outcome_difference = outcome[left] - outcome[right]
                if score_difference == 0.0:
                    values.append(0.5)
                else:
                    values.append(float(score_difference * outcome_difference > 0.0))
        if values:
            daily.append(float(np.mean(values)))
    return float(np.mean(daily)) if daily else np.nan


def control_parity(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = {
        "Rank1": _daily_position(frame, 1),
        "Rank2": _daily_position(frame, 2),
        "Rank3": _daily_position(frame, 3),
        "Top2": _daily_top(frame, 2),
        "Top3": _daily_top(frame, 3),
        "cross_threshold": _date_equal_concordance(frame, "cross"),
    }
    parity = all(
        np.isclose(metrics[name], expected, rtol=0.0, atol=5e-12)
        for name, expected in EXPECTED_PARITY.items()
    )
    if not parity:
        raise RuntimeError(
            f"FATAL: exact V4A_ARCH_TRANSFER_V4C parity failed: {metrics}"
        )
    return {**metrics, "pass": True}


def load_and_join_authoritative_features(
    root: Path, oof: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
        "event_id",
        "signal_date",
        "code",
        "board_streak_before_break",
        *RESIDUAL_FIELDS,
    ]
    authority = pd.read_csv(
        _feature_path(root),
        encoding="utf-8-sig",
        dtype={"event_id": str, "signal_date": str, "code": str},
        usecols=columns,
    )
    authority["code"] = authority["code"].astype(str).str.zfill(6)
    if bool(authority["signal_date"].astype(str).ge(JULY_SENTINEL_DATE).any()):
        raise RuntimeError("FATAL: July row encountered in authoritative feature source")
    duplicate = int(authority["event_id"].duplicated(keep=False).sum())
    joined = oof.merge(
        authority,
        on="event_id",
        how="left",
        suffixes=("_oof", "_bao"),
        indicator=True,
        validate="one_to_one" if duplicate == 0 else "many_to_many",
    )
    matched = int(joined["_merge"].eq("both").sum())
    unmatched = int(joined["_merge"].ne("both").sum())
    code_mismatch = joined["code_oof"].astype(str).ne(joined["code_bao"].astype(str))
    date_mismatch = joined["signal_date_oof"].astype(str).ne(
        joined["signal_date_bao"].astype(str)
    )
    board_mismatch = pd.to_numeric(joined["board"], errors="coerce").ne(
        pd.to_numeric(joined["board_streak_before_break"], errors="coerce")
    )
    mismatch = int((code_mismatch | date_mismatch | board_mismatch).sum())
    audit = {
        "oof_rows": int(len(oof)),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "duplicate_joins": duplicate,
        "code_date_board_mismatches": mismatch,
    }
    if unmatched or duplicate or mismatch or len(joined) != len(oof):
        raise RuntimeError(f"FATAL: authoritative feature join gate failed: {audit}")
    result = joined.drop(columns=["_merge", "code_bao", "signal_date_bao"]).rename(
        columns={"code_oof": "code", "signal_date_oof": "signal_date"}
    )
    for feature in RESIDUAL_FIELDS:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result.sort_values(
        ["signal_date", "model_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True), audit


def assign_residual_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rank = result["model_rank"].astype(int)
    target = result["target7"].astype(int)
    conditions = [
        rank.le(3) & target.eq(0),
        rank.le(3) & target.eq(1),
        rank.between(4, 10, inclusive="both") & target.eq(1),
        rank.between(4, 10, inclusive="both") & target.eq(0),
    ]
    result["residual_group"] = np.select(
        conditions, [GROUP_FP, GROUP_TP, GROUP_MW, GROUP_RN], default=GROUP_OTHER
    )
    result["candidate_count"] = result.groupby("signal_date", sort=True)[
        "event_id"
    ].transform("size").astype(int)
    rank3_score = result.loc[result["model_rank"].eq(3)].set_index("signal_date")[
        "model_score"
    ]
    result["score_gap_to_rank3"] = np.nan
    missed = result["residual_group"].eq(GROUP_MW)
    result.loc[missed, "score_gap_to_rank3"] = (
        result.loc[missed, "signal_date"].map(rank3_score)
        - result.loc[missed, "model_score"]
    )
    if bool(result.loc[missed, "score_gap_to_rank3"].lt(-1e-15).any()):
        raise RuntimeError("FATAL: missed-winner score gap to Rank3 is negative")
    return result


def group_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for group in (GROUP_FP, GROUP_TP, GROUP_MW, GROUP_RN):
        subset = frame[frame["residual_group"].eq(group)]
        result[group] = {
            "rows": int(len(subset)),
            "dates": int(subset["signal_date"].nunique()),
        }
    fp_dates = set(frame.loc[frame["residual_group"].eq(GROUP_FP), "signal_date"])
    mw_dates = set(frame.loc[frame["residual_group"].eq(GROUP_MW), "signal_date"])
    result["BOTH"] = {"dates": len(fp_dates.intersection(mw_dates)), "rows": 0}
    return result


def cliffs_delta(mw_values: Sequence[float], fp_values: Sequence[float]) -> float:
    mw = np.asarray(mw_values, dtype=float)
    fp = np.asarray(fp_values, dtype=float)
    mw = mw[np.isfinite(mw)]
    fp = fp[np.isfinite(fp)]
    if len(mw) == 0 or len(fp) == 0:
        return np.nan
    differences = mw[:, None] - fp[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def deterministic_date_bootstrap(daily_differences: Sequence[float]) -> tuple[float, float, float]:
    values = np.asarray(daily_differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw = generator.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    bootstrap_means = values[draw].mean(axis=1)
    quantiles = np.quantile(bootstrap_means, [0.025, 0.50, 0.975])
    return tuple(float(value) for value in quantiles)


def _describe(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    return {
        "rows": int(len(numeric)),
        "n": int(len(valid)),
        "missing": int(numeric.isna().sum()),
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "median": float(valid.median()) if len(valid) else np.nan,
        "p25": float(valid.quantile(0.25)) if len(valid) else np.nan,
        "p75": float(valid.quantile(0.75)) if len(valid) else np.nan,
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
    }


def build_feature_comparison(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp = frame[frame["residual_group"].eq(GROUP_FP)]
    mw = frame[frame["residual_group"].eq(GROUP_MW)]
    comparison_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for feature in RESIDUAL_FIELDS:
        semantic_group = (
            "DAMAGE_DISAGREEMENT" if feature in DAMAGE_FIELDS else "REPAIR_COMPLETION"
        )
        fp_description = _describe(fp[feature])
        mw_description = _describe(mw[feature])
        comparable_dates = sorted(
            set(fp.loc[fp[feature].notna(), "signal_date"]).intersection(
                set(mw.loc[mw[feature].notna(), "signal_date"])
            )
        )
        for date in comparable_dates:
            fp_values = fp.loc[fp["signal_date"].eq(date), feature].dropna()
            mw_values = mw.loc[mw["signal_date"].eq(date), feature].dropna()
            if fp_values.empty or mw_values.empty:
                continue
            fp_mean = float(fp_values.mean())
            mw_mean = float(mw_values.mean())
            daily_rows.append({
                "signal_date": date,
                "feature": feature,
                "semantic_group": semantic_group,
                "FP_count": int(len(fp_values)),
                "MW_count": int(len(mw_values)),
                "FP_mean": fp_mean,
                "MW_mean": mw_mean,
                "MW_minus_FP": mw_mean - fp_mean,
            })
        feature_daily = [
            row for row in daily_rows if row["feature"] == feature
        ]
        differences = np.asarray(
            [row["MW_minus_FP"] for row in feature_daily], dtype=float
        )
        positive = float(np.mean(differences > 0)) if len(differences) else np.nan
        negative = float(np.mean(differences < 0)) if len(differences) else np.nan
        zero = float(np.mean(differences == 0)) if len(differences) else np.nan
        bootstrap = deterministic_date_bootstrap(differences)
        direction_consistency = positive if feature in DAMAGE_FIELDS else negative
        cliff = cliffs_delta(mw[feature].to_numpy(float), fp[feature].to_numpy(float))
        comparison_rows.append({
            "feature": feature,
            "semantic_group": semantic_group,
            **{f"FP_{key}": value for key, value in fp_description.items()},
            **{f"MW_{key}": value for key, value in mw_description.items()},
            "MW_minus_FP_mean": mw_description["mean"] - fp_description["mean"],
            "MW_minus_FP_median": mw_description["median"] - fp_description["median"],
            "cliffs_delta_MW_vs_FP": cliff,
            "comparable_dates": int(len(differences)),
            "daily_diff_mean": float(differences.mean()) if len(differences) else np.nan,
            "daily_diff_median": float(np.median(differences)) if len(differences) else np.nan,
            "positive_difference_date_pct": positive,
            "negative_difference_date_pct": negative,
            "zero_difference_date_pct": zero,
            "daily_diff_p25": float(np.quantile(differences, 0.25)) if len(differences) else np.nan,
            "daily_diff_p75": float(np.quantile(differences, 0.75)) if len(differences) else np.nan,
            "direction_consistency_pct": direction_consistency,
            "bootstrap_p025": bootstrap[0],
            "bootstrap_p50": bootstrap[1],
            "bootstrap_p975": bootstrap[2],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        })
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(daily_rows).sort_values(
            ["feature", "signal_date"], kind="mergesort"
        ).reset_index(drop=True),
    )


def residual_evidence(feature_comparison: pd.DataFrame) -> dict[str, Any]:
    damage = feature_comparison[
        feature_comparison["semantic_group"].eq("DAMAGE_DISAGREEMENT")
    ]
    completion = feature_comparison[
        feature_comparison["semantic_group"].eq("REPAIR_COMPLETION")
    ]
    coherent_damage = damage[
        damage["direction_consistency_pct"].ge(0.65)
        & damage["cliffs_delta_MW_vs_FP"].ge(0.20)
        & damage["comparable_dates"].ge(6)
    ]
    coherent_completion = completion[
        completion["direction_consistency_pct"].ge(0.65)
        & completion["cliffs_delta_MW_vs_FP"].le(-0.20)
        & completion["comparable_dates"].ge(6)
    ]
    weak_damage = damage[
        damage["positive_difference_date_pct"].gt(0.50)
        & damage["cliffs_delta_MW_vs_FP"].gt(0.0)
    ]
    weak_completion = completion[
        completion["negative_difference_date_pct"].gt(0.50)
        & completion["cliffs_delta_MW_vs_FP"].lt(0.0)
    ]
    comparable_gate = bool(feature_comparison["comparable_dates"].max() >= 6)
    supported = bool(
        len(coherent_damage) >= 2
        and len(coherent_completion) >= 1
        and comparable_gate
    )
    if supported:
        evidence = "SUPPORTED"
    elif len(weak_damage) + len(weak_completion) > 0:
        evidence = "MIXED"
    else:
        evidence = "NOT_SUPPORTED"

    def block_answer(coherent: pd.DataFrame, weak: pd.DataFrame, minimum: int) -> str:
        if len(coherent) >= minimum:
            return "YES"
        if len(weak) > 0:
            return "MIXED"
        return "NO"

    return {
        "repair_room_residual_evidence": evidence,
        "damage_answer": block_answer(coherent_damage, weak_damage, 2),
        "completion_answer": block_answer(coherent_completion, weak_completion, 1),
        "date_stability_answer": (
            "YES" if evidence == "SUPPORTED" else "MIXED" if evidence == "MIXED" else "NO"
        ),
        "coherent_damage_fields": coherent_damage["feature"].tolist(),
        "coherent_completion_fields": coherent_completion["feature"].tolist(),
        "weak_damage_fields": weak_damage["feature"].tolist(),
        "weak_completion_fields": weak_completion["feature"].tolist(),
        "comparable_date_gate": comparable_gate,
    }


def _strict_top3_return(day: pd.DataFrame, rank_column: str = "model_rank") -> float:
    if len(day) < 3:
        return np.nan
    return float(day.loc[day[rank_column].le(3), "capped_opportunity_return_7"].mean())


def _hindsight_top(day: pd.DataFrame, pool_rank_limit: int | None, k: int) -> pd.DataFrame:
    pool = day if pool_rank_limit is None else day[day["model_rank"].le(pool_rank_limit)]
    return pool.sort_values(
        ["capped_opportunity_return_7", "event_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(k)


def build_top10_recall(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("signal_date", sort=True):
        ordered = day.sort_values(["model_rank", "event_id"], kind="mergesort")
        target_count = int(day["target7"].sum())
        top10_oracle = _hindsight_top(day, 10, 3)
        full_oracle = _hindsight_top(day, None, 3)
        eligible_top3 = len(day) >= 3
        fp_count = int(
            (day["model_rank"].le(3) & day["target7"].eq(0)).sum()
        )
        mw_count = int(
            (day["model_rank"].between(4, 10) & day["target7"].eq(1)).sum()
        )
        row: dict[str, Any] = {
            "signal_date": date,
            "candidate_count": int(len(day)),
            "target7_count": target_count,
            "fp_top3_count": fp_count,
            "mw_rank4_10_count": mw_count,
            "possible_replacements": min(fp_count, mw_count),
            "actual_top3_capped": _strict_top3_return(day),
            "top10_oracle_top3_capped": (
                float(top10_oracle["capped_opportunity_return_7"].mean())
                if eligible_top3
                else np.nan
            ),
            "full_oracle_top3_capped": (
                float(full_oracle["capped_opportunity_return_7"].mean())
                if eligible_top3
                else np.nan
            ),
            "top10_oracle_rank1_capped": float(
                _hindsight_top(day, 10, 1)["capped_opportunity_return_7"].iloc[0]
            ),
            "top10_oracle_top1_target7": float(
                _hindsight_top(day, 10, 1)["target7"].iloc[0]
            ),
            "top10_oracle_top3_target7_precision": (
                float(top10_oracle["target7"].mean()) if eligible_top3 else np.nan
            ),
            "purpose": "HINDSIGHT_DIAGNOSTIC_ONLY",
        }
        for k in (3, 5, 7, 10):
            row[f"top{k}_target7"] = int(
                ordered.loc[ordered["model_rank"].le(k), "target7"].sum()
            )
        rows.append(row)
    daily = pd.DataFrame(rows)
    total_winners = int(frame["target7"].sum())

    def recall(k: int) -> float:
        return float(
            frame.loc[frame["model_rank"].le(k), "target7"].sum() / total_winners
        )

    def date_equal_capture(k: int) -> float:
        values: list[float] = []
        for _, day in frame.groupby("signal_date", sort=True):
            target_count = int(day["target7"].sum())
            if target_count <= 0:
                continue
            selected = int(day.loc[day["model_rank"].le(k), "target7"].sum())
            values.append(float(selected / min(k, target_count)))
        return float(np.mean(values)) if values else np.nan

    recalls = {k: recall(k) for k in (3, 5, 7, 10)}
    captures = {k: date_equal_capture(k) for k in (3, 5, 10)}
    actual_top3 = float(daily["actual_top3_capped"].mean())
    top10_top3 = float(daily["top10_oracle_top3_capped"].mean())
    full_top3 = float(daily["full_oracle_top3_capped"].mean())
    improvement = top10_top3 - actual_top3
    denominator = full_top3 - actual_top3
    recovery_ratio = improvement / denominator if denominator > 0 else np.nan
    replacement_dates = int(daily["possible_replacements"].gt(0).sum())
    capture_gain = captures[10] - captures[3]
    yes = bool(
        capture_gain >= 0.10
        and improvement >= 0.01
        and replacement_dates >= 11
    )
    if yes:
        feasibility = "YES"
    elif recalls[10] < 0.50 or captures[10] <= captures[3] or improvement < 0.005 or replacement_dates == 0:
        feasibility = "NO"
    else:
        feasibility = "INCONCLUSIVE"
    summary = {
        "target7_total": total_winners,
        "recall_at_3": recalls[3],
        "recall_at_5": recalls[5],
        "recall_at_7": recalls[7],
        "recall_at_10": recalls[10],
        "winner_capture_top3": captures[3],
        "winner_capture_top5": captures[5],
        "winner_capture_top10": captures[10],
        "actual_top3": actual_top3,
        "top10_oracle_rank1": float(daily["top10_oracle_rank1_capped"].mean()),
        "top10_oracle_top3": top10_top3,
        "full_oracle_top3": full_top3,
        "top10_oracle_top1_target7": float(daily["top10_oracle_top1_target7"].mean()),
        "top10_oracle_top3_target7_precision": float(
            daily["top10_oracle_top3_target7_precision"].mean()
        ),
        "top10_oracle_improvement": improvement,
        "top10_oracle_recovery_ratio": float(recovery_ratio),
        "dates_with_replacements": replacement_dates,
        "total_possible_replacements": int(daily["possible_replacements"].sum()),
        "winner_capture_gain_top10_vs_top3": capture_gain,
        "top10_rerank_feasibility": feasibility,
    }
    return daily, summary


def missed_winner_audit(frame: pd.DataFrame) -> dict[str, Any]:
    mw = frame[frame["residual_group"].eq(GROUP_MW)]
    gap = mw["score_gap_to_rank3"].dropna()
    return {
        "rank_counts": {
            rank: int(mw["model_rank"].eq(rank).sum()) for rank in range(4, 11)
        },
        "score_gap_mean": float(gap.mean()),
        "score_gap_median": float(gap.median()),
        "score_gap_p25": float(gap.quantile(0.25)),
        "score_gap_p75": float(gap.quantile(0.75)),
        "score_gap_min": float(gap.min()),
        "score_gap_max": float(gap.max()),
    }


def outcome_severity(frame: pd.DataFrame) -> dict[str, Any]:
    fp = frame[frame["residual_group"].eq(GROUP_FP)]
    mw = frame[frame["residual_group"].eq(GROUP_MW)]
    raw_fp = fp["raw_repair_return"].astype(float)
    capped_fp = fp["capped_opportunity_return_7"].astype(float)

    def count_rate(mask: pd.Series, total: int) -> dict[str, float | int]:
        count = int(mask.sum())
        return {"count": count, "rate": float(count / total) if total else np.nan}

    return {
        "fp_raw": {
            **_describe(raw_fp),
            "worst": float(raw_fp.min()),
        },
        "fp_capped": {
            **_describe(capped_fp),
            "worst": float(capped_fp.min()),
        },
        "fp_buckets": {
            "near_miss_5_7": count_rate(raw_fp.ge(0.05) & raw_fp.lt(0.07), len(fp)),
            "weak_0_5": count_rate(raw_fp.ge(0.0) & raw_fp.lt(0.05), len(fp)),
            "loss_below_0": count_rate(raw_fp.lt(0.0), len(fp)),
            "severe_le_minus_5": count_rate(raw_fp.le(-0.05), len(fp)),
        },
        "mw_buckets": {
            "7_10": count_rate(mw["raw_repair_return"].ge(0.07) & mw["raw_repair_return"].lt(0.10), len(mw)),
            "10_12": count_rate(mw["raw_repair_return"].ge(0.10) & mw["raw_repair_return"].lt(0.12), len(mw)),
            "ge_12": count_rate(mw["raw_repair_return"].ge(0.12), len(mw)),
        },
    }


def board_and_candidate_audit(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, group in (("FP", GROUP_FP), ("MW", GROUP_MW)):
        subset = frame[frame["residual_group"].eq(group)]
        date_candidate_counts = subset[["signal_date", "candidate_count"]].drop_duplicates(
            "signal_date", keep="first"
        )["candidate_count"]
        board2 = int(subset["board"].eq(2).sum())
        board3 = int(subset["board"].eq(3).sum())
        result[label] = {
            "rows": int(len(subset)),
            "board2_count": board2,
            "board2_rate": float(board2 / len(subset)) if len(subset) else np.nan,
            "board3_count": board3,
            "board3_rate": float(board3 / len(subset)) if len(subset) else np.nan,
            "candidate_count_dates": int(len(date_candidate_counts)),
            "candidate_count_mean": float(date_candidate_counts.mean()),
            "candidate_count_median": float(date_candidate_counts.median()),
            "candidate_count_p25": float(date_candidate_counts.quantile(0.25)),
            "candidate_count_p75": float(date_candidate_counts.quantile(0.75)),
            "candidate_count_min": int(date_candidate_counts.min()),
            "candidate_count_max": int(date_candidate_counts.max()),
        }
    return result


def residual_events_output(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id",
        "signal_date",
        "code",
        "board",
        "model_score",
        "model_rank",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
        "residual_group",
        *RESIDUAL_FIELDS,
        "score_gap_to_rank3",
        "candidate_count",
    ]
    return frame[columns].sort_values(
        ["signal_date", "model_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _number(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return result


def build_review(
    parity: Mapping[str, Any],
    join_audit: Mapping[str, int],
    groups: Mapping[str, Mapping[str, int]],
    comparison: pd.DataFrame,
    evidence: Mapping[str, Any],
    missed: Mapping[str, Any],
    top10: Mapping[str, Any],
    severity: Mapping[str, Any],
    board: Mapping[str, Any],
) -> str:
    lines = [
        "# Did Exact v4a Top3 False Positives Differ From Rank4–10 Missed Winners?",
        "",
    ]
    lines.extend(_markdown_table(
        ["Feature", "FP Median", "MW Median", "MW-FP", "Cliff Delta", "Date Direction %", "Bootstrap 95%"],
        [[
            row.feature,
            _number(row.FP_median),
            _number(row.MW_median),
            _number(row.MW_minus_FP_median),
            _number(row.cliffs_delta_MW_vs_FP, 4),
            _pct(row.direction_consistency_pct),
            f"[{_number(row.bootstrap_p025)}, {_number(row.bootstrap_p975)}]",
        ] for row in comparison.itertuples()],
    ))
    lines.extend([
        "",
        "Cliff's delta is MW relative to FP. It is an effect size, not prediction proof or a causality test.",
        "",
        "# Is There Evidence of More Unfinished Repair Among Missed Winners?",
        "",
        "## Damage / Disagreement",
        "",
        f"- Directional conclusion: **{evidence['damage_answer']}**",
        f"- Gate-consistent fields: **{', '.join(evidence['coherent_damage_fields']) or 'NONE'}**",
        f"- Weak expected-direction fields: **{', '.join(evidence['weak_damage_fields']) or 'NONE'}**",
        "",
        "## Repair Completion",
        "",
        f"- Directional conclusion: **{evidence['completion_answer']}**",
        f"- Gate-consistent fields: **{', '.join(evidence['coherent_completion_fields']) or 'NONE'}**",
        f"- Weak expected-direction fields: **{', '.join(evidence['weak_completion_fields']) or 'NONE'}**",
        "",
        f"REPAIR_ROOM_RESIDUAL_EVIDENCE: **{evidence['repair_room_residual_evidence']}**",
        "",
        "# Are the Missed Winners Already Near the Top3 Boundary?",
        "",
        "- MW rank distribution: " + ", ".join(
            f"Rank{rank}={count}" for rank, count in missed["rank_counts"].items()
        ),
        f"- Score gap to Rank3 mean: **{_number(missed['score_gap_mean'])}**",
        f"- Score gap median: **{_number(missed['score_gap_median'])}**",
        f"- Score gap p25/p75: **{_number(missed['score_gap_p25'])} / {_number(missed['score_gap_p75'])}**",
        f"- Score gap min/max: **{_number(missed['score_gap_min'])} / {_number(missed['score_gap_max'])}**",
        "",
        "# How Much Could a Perfect Top10 Reranker Recover?",
        "",
        f"- Actual V4A Top3: **{_pct(top10['actual_top3'])}**",
        f"- Top10 hindsight oracle Rank1: **{_pct(top10['top10_oracle_rank1'])}**",
        f"- Top10 hindsight oracle Top3: **{_pct(top10['top10_oracle_top3'])}**",
        f"- Top10 oracle Top1 Target7 / Top3 precision: **{_pct(top10['top10_oracle_top1_target7'])} / {_pct(top10['top10_oracle_top3_target7_precision'])}**",
        f"- Full-universe oracle Top3: **{_pct(top10['full_oracle_top3'])}**",
        f"- Top10 improvement: **{_pct(top10['top10_oracle_improvement'])}**",
        f"- Recovery ratio: **{_pct(top10['top10_oracle_recovery_ratio'])}**",
        "- Oracle calculations are **HINDSIGHT_DIAGNOSTIC_ONLY**.",
        "",
        "# Is Top10 Recall Sufficient for a Second-Stage Reranker?",
        "",
        f"- Target7 recall@3/@5/@7/@10: **{_pct(top10['recall_at_3'])} / {_pct(top10['recall_at_5'])} / {_pct(top10['recall_at_7'])} / {_pct(top10['recall_at_10'])}**",
        f"- Date-equal winner capture Top3/Top5/Top10: **{_pct(top10['winner_capture_top3'])} / {_pct(top10['winner_capture_top5'])} / {_pct(top10['winner_capture_top10'])}**",
        f"- Dates with replacement opportunity: **{top10['dates_with_replacements']} / 21**",
        f"- Total possible replacements: **{top10['total_possible_replacements']}**",
        f"- TOP10_RERANK_FEASIBILITY: **{top10['top10_rerank_feasibility']}**",
        "",
        "## Group and Source Gates",
        "",
        f"- OOF rows / dates: **{join_audit['oof_rows']} / 21**",
        f"- Join matched / unmatched / duplicate / mismatch: **{join_audit['matched_rows']} / {join_audit['unmatched_rows']} / {join_audit['duplicate_joins']} / {join_audit['code_date_board_mismatches']}**",
        f"- FP: **{groups[GROUP_FP]['rows']} rows / {groups[GROUP_FP]['dates']} dates**",
        f"- MW: **{groups[GROUP_MW]['rows']} rows / {groups[GROUP_MW]['dates']} dates**",
        f"- TP: **{groups[GROUP_TP]['rows']} rows / {groups[GROUP_TP]['dates']} dates**",
        f"- RN: **{groups[GROUP_RN]['rows']} rows / {groups[GROUP_RN]['dates']} dates**",
        f"- Dates containing both FP and MW: **{groups['BOTH']['dates']}**",
        f"- Exact frozen parity: **{'PASS' if parity['pass'] else 'FAIL'}**",
        "",
        "## Outcome Severity and Context Audits",
        "",
        f"- FP near miss / weak / loss / severe counts: **{severity['fp_buckets']['near_miss_5_7']['count']} / {severity['fp_buckets']['weak_0_5']['count']} / {severity['fp_buckets']['loss_below_0']['count']} / {severity['fp_buckets']['severe_le_minus_5']['count']}**",
        f"- FP raw return mean / median / p25 / p75 / worst: **{_pct(severity['fp_raw']['mean'])} / {_pct(severity['fp_raw']['median'])} / {_pct(severity['fp_raw']['p25'])} / {_pct(severity['fp_raw']['p75'])} / {_pct(severity['fp_raw']['worst'])}**",
        f"- FP capped return mean / median / p25 / p75 / worst: **{_pct(severity['fp_capped']['mean'])} / {_pct(severity['fp_capped']['median'])} / {_pct(severity['fp_capped']['p25'])} / {_pct(severity['fp_capped']['p75'])} / {_pct(severity['fp_capped']['worst'])}**",
        f"- MW 7–10% / 10–12% / >=12% counts: **{severity['mw_buckets']['7_10']['count']} / {severity['mw_buckets']['10_12']['count']} / {severity['mw_buckets']['ge_12']['count']}**",
        f"- FP board2 / board3: **{board['FP']['board2_count']} ({_pct(board['FP']['board2_rate'])}) / {board['FP']['board3_count']} ({_pct(board['FP']['board3_rate'])})**",
        f"- MW board2 / board3: **{board['MW']['board2_count']} ({_pct(board['MW']['board2_rate'])}) / {board['MW']['board3_count']} ({_pct(board['MW']['board3_rate'])})**",
        f"- FP group-date candidate count mean / median / p25 / p75 / min / max: **{_number(board['FP']['candidate_count_mean'], 2)} / {_number(board['FP']['candidate_count_median'], 2)} / {_number(board['FP']['candidate_count_p25'], 2)} / {_number(board['FP']['candidate_count_p75'], 2)} / {board['FP']['candidate_count_min']} / {board['FP']['candidate_count_max']}**",
        f"- MW group-date candidate count mean / median / p25 / p75 / min / max: **{_number(board['MW']['candidate_count_mean'], 2)} / {_number(board['MW']['candidate_count_median'], 2)} / {_number(board['MW']['candidate_count_p25'], 2)} / {_number(board['MW']['candidate_count_p75'], 2)} / {board['MW']['candidate_count_min']} / {board['MW']['candidate_count_max']}**",
        "",
        "## Formal Decision",
        "",
        f"- Q1 exact frozen OOF used: **YES**",
        f"- Q2 Top3 false positives: **{groups[GROUP_FP]['rows']} rows / {groups[GROUP_FP]['dates']} dates**",
        f"- Q3 Rank4–10 missed winners: **{groups[GROUP_MW]['rows']} rows / {groups[GROUP_MW]['dates']} dates**",
        f"- Q4 more D1 damage/disagreement: **{evidence['damage_answer']}**",
        f"- Q5 FP more fully repaired by D1 close: **{evidence['completion_answer']}**",
        f"- Q6 pattern stable across dates: **{evidence['date_stability_answer']}**",
        f"- Q9 future Top10→Top3 experiment justified: **{top10['top10_rerank_feasibility']}**",
        f"- Q10 unfinished-repair hypothesis: **{evidence['repair_room_residual_evidence']}**",
        "",
        f"REPAIR_ROOM_RESIDUAL_EVIDENCE: **{evidence['repair_room_residual_evidence']}**",
        "",
        f"TOP10_RERANK_FEASIBILITY: **{top10['top10_rerank_feasibility']}**",
        "",
        "- New model trained: **NO**",
        "- Feature selection: **NO**",
        "- New features: **NO**",
        "- Score modification: **NO**",
        "- July accessed: **NO**",
    ])
    if (
        evidence["repair_room_residual_evidence"] == "SUPPORTED"
        and top10["top10_rerank_feasibility"] == "YES"
    ):
        lines.extend([
            "",
            "A future task may design one very small Top10→Top3 residual reranker / false-positive veto capability test. No reranker is implemented here.",
        ])
    elif top10["top10_rerank_feasibility"] == "YES":
        lines.extend([
            "",
            "Top10 contains factual reranking room, but the current repair-room explanation is not supported strongly enough to use these eight fields automatically.",
        ])
    elif evidence["repair_room_residual_evidence"] == "SUPPORTED":
        lines.extend([
            "",
            "Residual states differ, but v4a Top10 recall/replacement coverage is insufficient to justify a Top10 reranker.",
        ])
    else:
        lines.extend([
            "",
            "Stop this residual line. Do not create another explanatory feature set automatically.",
        ])
    return "\n".join(lines) + "\n"


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=False, lineterminator="\n", float_format="%.12g", na_rep=""
    )
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    assert_analysis_contract()
    oof = load_exact_transfer_oof(root)
    parity = control_parity(oof)
    joined, join_audit = load_and_join_authoritative_features(root, oof)
    grouped = assign_residual_groups(joined)
    groups = group_counts(grouped)
    comparison, daily = build_feature_comparison(grouped)
    evidence = residual_evidence(comparison)
    top10_daily, top10 = build_top10_recall(grouped)
    missed = missed_winner_audit(grouped)
    severity = outcome_severity(grouped)
    board = board_and_candidate_audit(grouped)
    events = residual_events_output(grouped)
    review = build_review(
        parity,
        join_audit,
        groups,
        comparison,
        evidence,
        missed,
        top10,
        severity,
        board,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(events),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(comparison),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[3]: dataframe_csv_bytes(top10_daily),
        OUTPUT_FILENAMES[4]: review.encode("utf-8"),
    }
    context = {
        "oof": oof,
        "parity": parity,
        "join_audit": join_audit,
        "grouped": grouped,
        "groups": groups,
        "events": events,
        "feature_comparison": comparison,
        "daily_differences": daily,
        "evidence": evidence,
        "top10_daily": top10_daily,
        "top10": top10,
        "missed_winner": missed,
        "severity": severity,
        "board_audit": board,
        "output_sha256": {
            name: sha256_bytes(value) for name, value in outputs.items()
        },
    }
    return outputs, context


def run_v004c_exact_v4a_residual(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first.keys() != second.keys() or any(
        first[name] != second[name] for name in first
    ):
        raise RuntimeError("FATAL: deterministic residual rebuild failed")
    output_dir = (
        root_path
        / "reports/research/v004c_exact_v4a_top3_residual_v001_20260601_20260630"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (output_dir / name).write_bytes(first[name])
    context["deterministic_rebuild"] = "PASS"
    context["output_dir"] = output_dir
    return output_dir, context

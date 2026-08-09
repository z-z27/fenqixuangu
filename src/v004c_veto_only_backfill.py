from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


JULY_SENTINEL_DATE = "2026-07-01"
TOP_K = 10
STRICT_K = 3
MODEL_COLUMNS = {
    "CONTROL": "stage1_rank",
    "FULL_RERANKER": "final_rerank_rank",
    "VETO_ONLY": "veto_only_rank",
    "TOP10_ORACLE": "top10_oracle_rank",
    "FULL_ORACLE": "full_oracle_rank",
}

SOURCE_DIR = Path(
    "reports/research/"
    "v004c_v4a_top10_residual_reranker_v001_20260506_20260630"
)
SOURCE_FILES = {
    "oof": "v004c_v4a_top10_reranker_oof_v001.csv",
    "membership": "v004c_v4a_top10_reranker_membership_v001.csv",
    "practical": "v004c_v4a_top10_reranker_practical_v001.csv",
    "fold_audit": "v004c_v4a_top10_reranker_fold_audit_v001.csv",
    "review": "v004c_v4a_top10_reranker_review_v001.md",
}
BUCKET_PATH = Path(
    "reports/research/"
    "v004c_v4a_architecture_transfer_v001_20260506_20260630/"
    "v004c_v4a_transfer_opportunity_terciles_v001.csv"
)
OUTPUT_DIR = Path(
    "reports/research/v004c_veto_only_backfill_v001_20260601_20260630"
)
OUTPUT_FILENAMES = (
    "v004c_veto_only_daily_v001.csv",
    "v004c_veto_only_membership_v001.csv",
    "v004c_veto_only_practical_v001.csv",
    "v004c_veto_only_review_v001.md",
)

REQUIRED_OOF_COLUMNS = {
    "event_id",
    "signal_date",
    "code",
    "candidate_count",
    "stage1_score",
    "stage1_rank",
    "stage1_top10",
    "stage2_score",
    "stage2_rank_within_top10",
    "final_rerank_rank",
    "control_top3",
    "reranker_top3",
    "target7",
    "raw_repair_return",
    "capped_opportunity_return_7",
}


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")


def _as_bool(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes"})


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.4f}%"


def _pp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:+.4f}pp"


def _num(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _codes(frame: pd.DataFrame, rank_column: str = "stage1_rank") -> str:
    if frame.empty:
        return ""
    return "|".join(
        frame.sort_values([rank_column, "event_id"], kind="mergesort")["code"]
        .astype(str)
        .str.zfill(6)
    )


def _values(frame: pd.DataFrame, column: str, rank_column: str = "stage1_rank") -> str:
    if frame.empty:
        return ""
    ordered = frame.sort_values([rank_column, "event_id"], kind="mergesort")
    return "|".join(str(value) for value in ordered[column].tolist())


def _returns(frame: pd.DataFrame, rank_column: str = "stage1_rank") -> str:
    if frame.empty:
        return ""
    ordered = frame.sort_values([rank_column, "event_id"], kind="mergesort")
    return "|".join(format(float(value), ".12g") for value in ordered["raw_repair_return"])


def load_frozen_artifacts(root: Path) -> dict[str, Any]:
    source_dir = root / SOURCE_DIR
    missing = [name for name in SOURCE_FILES.values() if not (source_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"FATAL: previous frozen reranker artifact unavailable: {missing}")
    oof = pd.read_csv(source_dir / SOURCE_FILES["oof"], encoding="utf-8-sig")
    absent = sorted(REQUIRED_OOF_COLUMNS - set(oof.columns))
    if absent:
        raise RuntimeError(f"FATAL: frozen OOF schema mismatch: {absent}")
    if len(oof) != 173 or oof["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: frozen June OOF identity mismatch")
    if oof["event_id"].duplicated().any():
        raise RuntimeError("FATAL: frozen June OOF event identity duplicated")
    if oof["signal_date"].astype(str).ge(JULY_SENTINEL_DATE).any():
        raise RuntimeError("FATAL: July sentinel triggered")
    if not oof["signal_date"].astype(str).str.startswith("2026-06").all():
        raise RuntimeError("FATAL: non-June row in frozen OOF")

    numeric = [
        "candidate_count",
        "stage1_score",
        "stage1_rank",
        "stage2_score",
        "stage2_rank_within_top10",
        "final_rerank_rank",
        "target7",
        "raw_repair_return",
        "capped_opportunity_return_7",
    ]
    for column in numeric:
        oof[column] = pd.to_numeric(oof[column], errors="coerce")
    if oof[numeric].isna().any().any():
        # Stage2 columns are expected to be missing outside frozen Stage1 Top10.
        allowed = {"stage2_score", "stage2_rank_within_top10"}
        unexpected = [
            column
            for column in numeric
            if column not in allowed and oof[column].isna().any()
        ]
        if unexpected:
            raise RuntimeError(f"FATAL: unexpected frozen OOF missing values: {unexpected}")
    oof["stage1_rank"] = oof["stage1_rank"].astype(int)
    oof["final_rerank_rank"] = oof["final_rerank_rank"].astype(int)
    oof["target7"] = oof["target7"].astype(int)
    oof["control_top3"] = _as_bool(oof["control_top3"])
    oof["reranker_top3"] = _as_bool(oof["reranker_top3"])
    oof["stage1_top10"] = _as_bool(oof["stage1_top10"])

    cap_expected = np.minimum(oof["raw_repair_return"].to_numpy(float), 0.07)
    if not np.allclose(
        cap_expected,
        oof["capped_opportunity_return_7"].to_numpy(float),
        rtol=0.0,
        atol=5e-12,
    ):
        raise RuntimeError("FATAL: capped opportunity return semantic mismatch")
    if not np.array_equal(
        oof["target7"].to_numpy(int),
        oof["raw_repair_return"].ge(0.07).astype(int).to_numpy(),
    ):
        raise RuntimeError("FATAL: Target7 semantic mismatch")
    if not np.array_equal(oof["control_top3"], oof["stage1_rank"].le(3)):
        raise RuntimeError("FATAL: CONTROL membership parity mismatch")
    if not np.array_equal(oof["reranker_top3"], oof["final_rerank_rank"].le(3)):
        raise RuntimeError("FATAL: FULL_RERANKER membership parity mismatch")

    fold = pd.read_csv(source_dir / SOURCE_FILES["fold_audit"], encoding="utf-8-sig")
    for column in ("self_label_leakage_rows", "current_test_date_leakage_rows"):
        if column not in fold or pd.to_numeric(fold[column], errors="coerce").sum() != 0:
            raise RuntimeError(f"FATAL: previous leakage audit failed: {column}")
    if len(fold) != 21:
        raise RuntimeError("FATAL: previous fold audit date coverage mismatch")
    review = (source_dir / SOURCE_FILES["review"]).read_text(encoding="utf-8")
    if "July accessed = **NO**" not in review or "Stage1 control parity = **PASS**" not in review:
        raise RuntimeError("FATAL: previous review leakage/parity audit failed")

    membership = pd.read_csv(
        source_dir / SOURCE_FILES["membership"], encoding="utf-8-sig"
    )
    practical = pd.read_csv(
        source_dir / SOURCE_FILES["practical"], encoding="utf-8-sig"
    )
    return {
        "oof": oof.sort_values(
            ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
        ).reset_index(drop=True),
        "fold_audit": fold,
        "membership": membership,
        "practical": practical,
        "review": review,
        "source_path": source_dir / SOURCE_FILES["oof"],
    }


def apply_veto_only_day(day: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = day.sort_values(["stage1_rank", "event_id"], kind="mergesort").copy()
    if ordered["signal_date"].nunique() != 1:
        raise RuntimeError("FATAL: veto policy input must contain exactly one date")
    expected_ranks = list(range(1, len(ordered) + 1))
    if ordered["stage1_rank"].astype(int).tolist() != expected_ranks:
        raise RuntimeError("FATAL: frozen Stage1 ranks are not contiguous")
    target_size = min(STRICT_K, len(ordered))
    control = ordered[ordered["stage1_rank"].le(target_size)].copy()
    if control["stage2_rank_within_top10"].isna().any():
        raise RuntimeError("FATAL: CONTROL member lacks frozen Stage2 rank")
    vetoed = control[control["stage2_rank_within_top10"].gt(3)].copy()
    vetoed_ids = set(vetoed["event_id"])
    final_ids = [
        row.event_id
        for row in control.itertuples()
        if row.event_id not in vetoed_ids
    ]
    for row in ordered.itertuples():
        if row.stage1_rank < 4:
            continue
        if row.stage1_rank > TOP_K or len(final_ids) >= target_size:
            break
        if row.event_id not in final_ids:
            final_ids.append(row.event_id)
    if len(final_ids) != target_size:
        raise RuntimeError("FATAL: Stage1 Rank4-10 backfill could not fill final Top3")
    final = ordered[ordered["event_id"].isin(final_ids)].sort_values(
        ["stage1_rank", "event_id"], kind="mergesort"
    )
    backfilled = final[~final["event_id"].isin(control["event_id"])].copy()
    if len(backfilled) != len(vetoed):
        raise RuntimeError("FATAL: veto/backfill slot count mismatch")
    expected_backfill = (
        ordered[
            ordered["stage1_rank"].between(4, TOP_K)
            & ~ordered["event_id"].isin(control["event_id"])
        ]
        .head(len(vetoed))["event_id"]
        .tolist()
    )
    if backfilled["event_id"].tolist() != expected_backfill:
        raise RuntimeError("FATAL: replacement did not strictly follow Stage1 order")

    result = ordered.copy()
    result["veto_only_top3"] = result["event_id"].isin(final_ids)
    result["veto_only_rank"] = np.nan
    result.loc[final.index, "veto_only_rank"] = np.arange(1, len(final) + 1)
    outside = result[~result["veto_only_top3"]].sort_values(
        ["stage1_rank", "event_id"], kind="mergesort"
    )
    result.loc[outside.index, "veto_only_rank"] = np.arange(
        len(final) + 1, len(result) + 1
    )
    result["veto_only_rank"] = result["veto_only_rank"].astype(int)
    return result, {
        "vetoed_ids": vetoed["event_id"].tolist(),
        "backfilled_ids": backfilled["event_id"].tolist(),
        "control_ids": control["event_id"].tolist(),
        "final_ids": final["event_id"].tolist(),
    }


def build_veto_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    policies: dict[str, dict[str, Any]] = {}
    for date, day in oof.groupby("signal_date", sort=True):
        result, audit = apply_veto_only_day(day)
        parts.append(result)
        policies[str(date)] = audit
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != 173 or result["signal_date"].nunique() != 21:
        raise RuntimeError("FATAL: veto-only OOF identity changed")
    return result, policies


def load_opportunity_buckets(root: Path, dates: Iterable[str]) -> dict[str, str]:
    path = root / BUCKET_PATH
    source = pd.read_csv(path, encoding="utf-8-sig", usecols=["opportunity_bucket", "dates"])
    mapping: dict[str, str] = {}
    for row in source.itertuples(index=False):
        for date in str(row.dates).split("|"):
            if date >= JULY_SENTINEL_DATE or date in mapping:
                raise RuntimeError("FATAL: invalid frozen opportunity bucket date")
            mapping[date] = str(row.opportunity_bucket)
    if set(mapping) != set(str(date) for date in dates):
        raise RuntimeError("FATAL: frozen opportunity bucket coverage mismatch")
    if any(list(mapping.values()).count(bucket) != 7 for bucket in ("LOW", "MID", "HIGH")):
        raise RuntimeError("FATAL: frozen opportunity bucket allocation mismatch")
    return mapping


def _selected_daily(
    frame: pd.DataFrame, rank_column: str, kind: str, k: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("signal_date", sort=True):
        if len(day) < k:
            continue
        if kind == "rank":
            selected = day[day[rank_column].eq(k)]
        else:
            selected = day[day[rank_column].le(k)]
        selected_return = float(selected["capped_opportunity_return_7"].mean())
        baseline = float(day["capped_opportunity_return_7"].mean())
        rows.append({
            "signal_date": date,
            "selected_return": selected_return,
            "baseline": baseline,
            "target7_precision": float(selected["target7"].mean()),
            "positive": float(selected_return > 0.0),
            "negative": float(selected_return < 0.0),
            "beat": float(selected_return > baseline),
        })
    return pd.DataFrame(rows)


def _winner_capture(frame: pd.DataFrame, rank_column: str) -> float:
    values: list[float] = []
    for _, day in frame.groupby("signal_date", sort=True):
        if len(day) < STRICT_K:
            continue
        target_count = int(day["target7"].sum())
        if target_count <= 0:
            continue
        selected = day[day[rank_column].le(STRICT_K)]
        values.append(float(selected["target7"].sum() / min(STRICT_K, target_count)))
    return float(np.mean(values)) if values else np.nan


def strategy_summary(frame: pd.DataFrame, rank_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date_count": int(frame["signal_date"].nunique()),
        "universe_target7_rate": float(
            frame.groupby("signal_date", sort=True)["target7"].mean().mean()
        ),
        "winner_capture": _winner_capture(frame, rank_column),
    }
    for label, kind, k in (
        ("Rank1", "rank", 1),
        ("Rank2", "rank", 2),
        ("Rank3", "rank", 3),
        ("Top2", "top", 2),
        ("Top3", "top", 3),
    ):
        daily = _selected_daily(frame, rank_column, kind, k)
        result[f"{label}_mean"] = float(daily["selected_return"].mean())
        result[f"{label}_median"] = float(daily["selected_return"].median())
        result[f"{label}_p25"] = float(daily["selected_return"].quantile(0.25))
        result[f"{label}_worst"] = float(daily["selected_return"].min())
        result[f"{label}_baseline"] = float(daily["baseline"].mean())
        result[f"{label}_excess"] = result[f"{label}_mean"] - result[f"{label}_baseline"]
        result[f"{label}_target7_precision"] = float(daily["target7_precision"].mean())
        result[f"{label}_positive_date_rate"] = float(daily["positive"].mean())
        result[f"{label}_negative_date_rate"] = float(daily["negative"].mean())
        result[f"{label}_beat_universe"] = float(daily["beat"].mean())
        result[f"{label}_dates"] = int(len(daily))
    top3 = _selected_daily(frame, rank_column, "top", 3)
    result["days_le_minus3"] = int(top3["selected_return"].le(-0.03).sum())
    result["days_le_minus5"] = int(top3["selected_return"].le(-0.05).sum())
    return result


def _oracle_rank(frame: pd.DataFrame, top10_only: bool) -> pd.Series:
    ranks = pd.Series(index=frame.index, dtype=int)
    for _, day in frame.groupby("signal_date", sort=True):
        pool = day[day["stage1_rank"].le(TOP_K)] if top10_only else day
        ordered_pool = pool.sort_values(
            ["capped_opportunity_return_7", "event_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        outside = day[~day.index.isin(ordered_pool.index)].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        ordered = pd.concat([ordered_pool, outside])
        ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return ranks.astype(int)


def practical_table(
    oof: pd.DataFrame, bucket_map: Mapping[str, str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    evaluated = oof.copy()
    evaluated["top10_oracle_rank"] = _oracle_rank(evaluated, True)
    evaluated["full_oracle_rank"] = _oracle_rank(evaluated, False)
    rows: list[dict[str, Any]] = []
    combined: dict[str, dict[str, Any]] = {}
    for scope in ("COMBINED", "LOW", "MID", "HIGH"):
        dates = (
            sorted(evaluated["signal_date"].unique())
            if scope == "COMBINED"
            else sorted(date for date, bucket in bucket_map.items() if bucket == scope)
        )
        subset = evaluated[evaluated["signal_date"].isin(dates)]
        summaries = {
            model: strategy_summary(subset, rank_column)
            for model, rank_column in MODEL_COLUMNS.items()
        }
        if scope == "COMBINED":
            combined.update(summaries)
        for model, summary in summaries.items():
            row: dict[str, Any] = {"scope": scope, "model": model}
            for label in ("Rank1", "Rank2", "Rank3", "Top2", "Top3"):
                for metric in (
                    "mean",
                    "median",
                    "p25",
                    "worst",
                    "baseline",
                    "excess",
                    "target7_precision",
                    "positive_date_rate",
                    "negative_date_rate",
                    "beat_universe",
                    "dates",
                ):
                    row[f"{label}_{metric}"] = summary[f"{label}_{metric}"]
            row["Top1_target7_hit"] = summary["Rank1_target7_precision"]
            row["winner_capture"] = summary["winner_capture"]
            row["universe_target7_rate"] = summary["universe_target7_rate"]
            row["days_le_minus3"] = summary["days_le_minus3"]
            row["days_le_minus5"] = summary["days_le_minus5"]
            rows.append(row)
        control = summaries["CONTROL"]
        universe: dict[str, Any] = {
            "scope": scope,
            "model": "UNIVERSE",
            "Top1_target7_hit": control["universe_target7_rate"],
            "winner_capture": np.nan,
            "universe_target7_rate": control["universe_target7_rate"],
            "days_le_minus3": np.nan,
            "days_le_minus5": np.nan,
        }
        for label in ("Rank1", "Rank2", "Rank3", "Top2", "Top3"):
            baseline = control[f"{label}_baseline"]
            universe[f"{label}_mean"] = baseline
            universe[f"{label}_median"] = np.nan
            universe[f"{label}_p25"] = np.nan
            universe[f"{label}_worst"] = np.nan
            universe[f"{label}_baseline"] = baseline
            universe[f"{label}_excess"] = 0.0
            universe[f"{label}_target7_precision"] = control["universe_target7_rate"]
            universe[f"{label}_positive_date_rate"] = np.nan
            universe[f"{label}_negative_date_rate"] = np.nan
            universe[f"{label}_beat_universe"] = np.nan
            universe[f"{label}_dates"] = control[f"{label}_dates"]
        rows.append(universe)
    return pd.DataFrame(rows), combined


def validate_previous_parity(
    summaries: Mapping[str, Mapping[str, Any]],
    previous_practical: pd.DataFrame,
    previous_membership: pd.DataFrame,
) -> dict[str, Any]:
    previous = previous_practical[previous_practical["scope"].eq("COMBINED")]
    previous = previous.set_index("model")
    comparisons = {
        "CONTROL": ("CONTROL",),
        "FULL_RERANKER": ("RERANKER",),
        "TOP10_ORACLE": ("TOP10_ORACLE",),
        "FULL_ORACLE": ("FULL_ORACLE",),
    }
    for current, (prior,) in comparisons.items():
        if prior not in previous.index:
            raise RuntimeError(f"FATAL: previous practical row unavailable: {prior}")
        for metric in ("Rank1_mean", "Rank2_mean", "Rank3_mean", "Top2_mean", "Top3_mean"):
            if metric not in previous.columns:
                raise RuntimeError(f"FATAL: previous practical metric unavailable: {metric}")
            if not np.isclose(
                float(summaries[current][metric]),
                float(previous.loc[prior, metric]),
                rtol=0.0,
                atol=5e-12,
            ):
                raise RuntimeError(f"FATAL: previous practical parity failed: {current}/{metric}")
    required_membership = {"PRESERVED", "DEMOTED", "PROMOTED"}
    if set(previous_membership["membership_status"].unique()) != required_membership:
        raise RuntimeError("FATAL: previous membership artifact parity failed")
    return {
        "pass": True,
        "control_top3": summaries["CONTROL"]["Top3_mean"],
        "full_reranker_top3": summaries["FULL_RERANKER"]["Top3_mean"],
    }


def membership_and_daily(
    oof: pd.DataFrame,
    policies: Mapping[str, Mapping[str, Any]],
    bucket_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    membership_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    pair_counts = {
        "bad_to_winner": 0,
        "bad_to_bad": 0,
        "winner_to_winner": 0,
        "winner_to_bad": 0,
    }
    possible = 0
    full_promoted_total = 0
    full_promoted_target7 = 0
    for date, day in oof.groupby("signal_date", sort=True):
        day = day.sort_values(["stage1_rank", "event_id"], kind="mergesort")
        policy = policies[str(date)]
        control = day[day["control_top3"]]
        full = day[day["reranker_top3"]]
        veto = day[day["veto_only_top3"]]
        vetoed = day[day["event_id"].isin(policy["vetoed_ids"])].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        backfilled = day[day["event_id"].isin(policy["backfilled_ids"])].sort_values(
            ["stage1_rank", "event_id"], kind="mergesort"
        )
        full_promoted = day[~day["control_top3"] & day["reranker_top3"]]
        full_promoted_total += len(full_promoted)
        full_promoted_target7 += int(full_promoted["target7"].sum())
        for demoted, promoted in zip(vetoed.itertuples(), backfilled.itertuples()):
            left = "winner" if int(demoted.target7) == 1 else "bad"
            right = "winner" if int(promoted.target7) == 1 else "bad"
            pair_counts[f"{left}_to_{right}"] += 1
        fp_count = int((day["stage1_rank"].le(3) & day["target7"].eq(0)).sum())
        mw_count = int((day["stage1_rank"].between(4, 10) & day["target7"].eq(1)).sum())
        possible += min(fp_count, mw_count)

        changed = bool(len(vetoed))
        if changed:
            union_ids = set(control["event_id"]) | set(full["event_id"]) | set(veto["event_id"])
            for row in day[day["event_id"].isin(union_ids)].itertuples():
                if row.event_id in policy["vetoed_ids"]:
                    action = "VETOED"
                elif row.event_id in policy["backfilled_ids"]:
                    action = "BACKFILLED"
                elif bool(row.reranker_top3) and not bool(row.control_top3):
                    action = "FULL_RERANKER_ONLY_PROMOTED"
                else:
                    action = "PRESERVED"
                membership_rows.append({
                    "signal_date": row.signal_date,
                    "event_id": row.event_id,
                    "code": str(row.code).zfill(6),
                    "stage1_rank": int(row.stage1_rank),
                    "stage2_rank_within_top10": int(row.stage2_rank_within_top10),
                    "control_top3": bool(row.control_top3),
                    "full_reranker_top3": bool(row.reranker_top3),
                    "veto_only_top3": bool(row.veto_only_top3),
                    "membership_action": action,
                    "target7": int(row.target7),
                    "raw_repair_return": float(row.raw_repair_return),
                    "capped_opportunity_return_7": float(row.capped_opportunity_return_7),
                })

        universe_return = float(day["capped_opportunity_return_7"].mean())
        control_top3_return = float(control["capped_opportunity_return_7"].mean())
        full_top3_return = float(full["capped_opportunity_return_7"].mean())
        veto_top3_return = float(veto["capped_opportunity_return_7"].mean())
        row = {
            "signal_date": date,
            "candidate_count": len(day),
            "candidate_count_lt3": bool(len(day) < 3),
            "universe_top3_return": universe_return,
            "control_rank1_return": float(control.iloc[0]["capped_opportunity_return_7"]),
            "control_rank2_return": (
                float(control.iloc[1]["capped_opportunity_return_7"]) if len(control) >= 2 else np.nan
            ),
            "control_rank3_return": (
                float(control.iloc[2]["capped_opportunity_return_7"]) if len(control) >= 3 else np.nan
            ),
            "control_top2_return": float(control.head(2)["capped_opportunity_return_7"].mean()),
            "control_top3_return": control_top3_return,
            "full_reranker_top3_return": full_top3_return,
            "veto_only_rank1_return": float(veto.iloc[0]["capped_opportunity_return_7"]),
            "veto_only_rank2_return": (
                float(veto.iloc[1]["capped_opportunity_return_7"]) if len(veto) >= 2 else np.nan
            ),
            "veto_only_rank3_return": (
                float(veto.iloc[2]["capped_opportunity_return_7"]) if len(veto) >= 3 else np.nan
            ),
            "veto_only_top2_return": float(veto.head(2)["capped_opportunity_return_7"].mean()),
            "veto_only_top3_return": veto_top3_return,
            "veto_only_minus_control": veto_top3_return - control_top3_return,
            "control_target7_count": int(control["target7"].sum()),
            "full_reranker_target7_count": int(full["target7"].sum()),
            "veto_only_target7_count": int(veto["target7"].sum()),
            "veto_count": len(vetoed),
            "backfill_count": len(backfilled),
            "opportunity_bucket": bucket_map[str(date)],
        }
        daily_rows.append(row)
        if changed:
            changed_rows.append({
                "signal_date": str(date),
                "control_top3_codes": _codes(control),
                "vetoed_codes": _codes(vetoed),
                "vetoed_stage1_ranks": _values(vetoed, "stage1_rank"),
                "vetoed_target7": _values(vetoed, "target7"),
                "vetoed_raw_returns": _returns(vetoed),
                "backfilled_codes": _codes(backfilled),
                "backfilled_stage1_ranks": _values(backfilled, "stage1_rank"),
                "backfilled_target7": _values(backfilled, "target7"),
                "backfilled_raw_returns": _returns(backfilled),
                "full_reranker_promoted_codes": _codes(full_promoted),
                "control_top3_return": control_top3_return,
                "full_reranker_top3_return": full_top3_return,
                "veto_only_top3_return": veto_top3_return,
                "veto_only_minus_control": veto_top3_return - control_top3_return,
            })

    membership = pd.DataFrame(membership_rows).sort_values(
        ["signal_date", "stage1_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    daily = pd.DataFrame(daily_rows).sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    vetoed = membership[membership["membership_action"].eq("VETOED")]
    backfilled = membership[membership["membership_action"].eq("BACKFILLED")]
    successful = pair_counts["bad_to_winner"]
    failed = sum(pair_counts.values()) - successful
    stats = {
        "veto_dates": int((daily["veto_count"] > 0).sum()),
        "vetoed_total": len(vetoed),
        "vetoed_target7": int(vetoed["target7"].sum()),
        "vetoed_non_target": int(vetoed["target7"].eq(0).sum()),
        "vetoed_loss": int(vetoed["raw_repair_return"].lt(0).sum()),
        "vetoed_severe_loss": int(vetoed["raw_repair_return"].le(-0.05).sum()),
        "vetoed_raw_mean": float(vetoed["raw_repair_return"].mean()),
        "vetoed_raw_median": float(vetoed["raw_repair_return"].median()),
        "veto_non_target_rate": float(vetoed["target7"].eq(0).mean()),
        "backfilled_total": len(backfilled),
        "backfilled_target7": int(backfilled["target7"].sum()),
        "backfilled_non_target": int(backfilled["target7"].eq(0).sum()),
        "backfilled_loss": int(backfilled["raw_repair_return"].lt(0).sum()),
        "backfilled_severe_loss": int(backfilled["raw_repair_return"].le(-0.05).sum()),
        "backfill_target7_rate": float(backfilled["target7"].mean()),
        "full_promoted_total": full_promoted_total,
        "full_promoted_target7": full_promoted_target7,
        "full_promoted_target7_rate": (
            float(full_promoted_target7 / full_promoted_total)
            if full_promoted_total
            else np.nan
        ),
        **pair_counts,
        "successful_replacements": successful,
        "failed_replacements": failed,
        "net_target7_slots_gained": int(backfilled["target7"].sum() - vetoed["target7"].sum()),
        "possible_replacement_slots": possible,
        "replacement_capture": float(successful / possible) if possible else np.nan,
    }
    return daily, membership, stats, changed_rows


def changed_date_summary(daily: pd.DataFrame) -> dict[str, Any]:
    changed = daily[daily["veto_count"].gt(0)].copy()
    delta = changed["veto_only_minus_control"]
    all_delta = daily["veto_only_minus_control"]
    return {
        "changed_dates": len(changed),
        "better": int(delta.gt(0).sum()),
        "worse": int(delta.lt(0).sum()),
        "equal": int(delta.eq(0).sum()),
        "changed_mean": float(delta.mean()),
        "changed_median": float(delta.median()),
        "changed_best": float(delta.max()),
        "changed_worst": float(delta.min()),
        "all_mean": float(all_delta.mean()),
        "all_median": float(all_delta.median()),
        "all_positive": int(all_delta.gt(0).sum()),
        "all_negative": int(all_delta.lt(0).sum()),
        "all_zero": int(all_delta.eq(0).sum()),
    }


def _scope_row(practical: pd.DataFrame, scope: str, model: str) -> pd.Series:
    rows = practical[practical["scope"].eq(scope) & practical["model"].eq(model)]
    if len(rows) != 1:
        raise RuntimeError(f"FATAL: practical row unavailable: {scope}/{model}")
    return rows.iloc[0]


def formal_status(
    practical: pd.DataFrame,
    replacement: Mapping[str, Any],
    previous_parity: bool,
    leakage_pass: bool,
) -> dict[str, Any]:
    control = _scope_row(practical, "COMBINED", "CONTROL")
    full = _scope_row(practical, "COMBINED", "FULL_RERANKER")
    veto = _scope_row(practical, "COMBINED", "VETO_ONLY")
    low_c = _scope_row(practical, "LOW", "CONTROL")
    low_v = _scope_row(practical, "LOW", "VETO_ONLY")
    high_c = _scope_row(practical, "HIGH", "CONTROL")
    high_v = _scope_row(practical, "HIGH", "VETO_ONLY")
    q = {
        "Q1": bool(previous_parity and leakage_pass),
        "Q2": True,
        "Q3": True,
        "Q4": bool(veto.Top3_mean > control.Top3_mean),
        "Q5": bool(veto.Top3_mean > veto.Top3_baseline),
        "Q6": bool(veto.Top3_mean - control.Top3_mean >= 0.005),
        "Q7": bool(veto.Top3_target7_precision > control.Top3_target7_precision),
        "Q8": bool(veto.Top3_target7_precision > veto.universe_target7_rate),
        "Q9": bool(replacement["backfill_target7_rate"] > replacement["full_promoted_target7_rate"]),
        "Q10": bool(replacement["backfill_target7_rate"] >= 0.50),
        "Q11": bool(replacement["successful_replacements"] > replacement["failed_replacements"]),
        "Q12": bool(replacement["net_target7_slots_gained"] > 0),
        "Q13": bool(low_v.Top3_mean - low_c.Top3_mean >= 0.005),
        "Q14": bool(low_v.Top3_negative_date_rate <= low_c.Top3_negative_date_rate),
        "Q15": bool(high_v.Top3_mean >= high_c.Top3_mean - 0.005),
        "Q16": bool(
            veto.Top3_negative_date_rate <= control.Top3_negative_date_rate
            and veto.Top3_worst >= control.Top3_worst - 0.01
        ),
    }
    strong = all(q[f"Q{i}"] for i in range(4, 17))
    invalid = not q["Q1"] or not q["Q2"] or not q["Q3"]
    if invalid:
        signal = "INVALID"
    elif strong:
        signal = "STRONG"
    elif (
        veto.Top3_mean <= control.Top3_mean
        or replacement["backfill_target7_rate"] < 0.50
        or veto.Top3_target7_precision <= control.Top3_target7_precision
    ):
        signal = "ABSENT"
    else:
        signal = "PARTIAL"
    q.update({
        "signal": signal,
        "forward_stress_candidate": "YES" if signal == "STRONG" else "NO",
        "full_reranker_top3": float(full.Top3_mean),
    })
    return q


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "<br>")


def build_review(
    source_path: Path,
    practical: pd.DataFrame,
    replacement: Mapping[str, Any],
    changed: Mapping[str, Any],
    changed_rows: Sequence[Mapping[str, Any]],
    leakage: Mapping[str, int],
    status: Mapping[str, Any],
    candidate_lt3_dates: Sequence[str],
) -> str:
    control = _scope_row(practical, "COMBINED", "CONTROL")
    full = _scope_row(practical, "COMBINED", "FULL_RERANKER")
    veto = _scope_row(practical, "COMBINED", "VETO_ONLY")
    universe = _scope_row(practical, "COMBINED", "UNIVERSE")
    top10 = _scope_row(practical, "COMBINED", "TOP10_ORACLE")
    lines = [
        "# v004c Veto-Only Backfill Audit v001",
        "",
        "## 1. Input and Frozen Artifact Parity",
        "",
        f"- Source: `{source_path.as_posix()}`",
        "- June rows / dates: **173 / 21**",
        "- Previous artifact parity: **PASS**",
        f"- SELF_LABEL_LEAKAGE_ROWS / current-test leakage: **{leakage['self']} / {leakage['current']}**",
        "- July accessed: **NO**",
        "- Candidate-count < 3 dates: " + ("|".join(candidate_lt3_dates) if candidate_lt3_dates else "NONE"),
        "",
        "## 2. Veto Policy Definition",
        "",
        "- Veto iff frozen CONTROL member has `stage2_rank_within_top10 > 3`.",
        "- Backfill strictly follows frozen `stage1_rank` from Rank4 through Rank10.",
        "- Stage2 score never controls replacement order.",
        "- No training, tuning, new feature, threshold search, or July access.",
        "",
        "## 3. Vetoed Member Quality",
        "",
        f"- Veto dates / members: **{replacement['veto_dates']} / {replacement['vetoed_total']}**",
        f"- Target7 / non-target: **{replacement['vetoed_target7']} / {replacement['vetoed_non_target']}**",
        f"- Loss / severe loss: **{replacement['vetoed_loss']} / {replacement['vetoed_severe_loss']}**",
        f"- Non-target rate: **{_pct(replacement['veto_non_target_rate'])}**",
        f"- Raw-return mean / median: **{_pct(replacement['vetoed_raw_mean'])} / {_pct(replacement['vetoed_raw_median'])}**",
        "",
        "## 4. Stage1-Order Backfill Quality",
        "",
        f"- Backfilled total / Target7 / non-target: **{replacement['backfilled_total']} / {replacement['backfilled_target7']} / {replacement['backfilled_non_target']}**",
        f"- Backfill Target7 rate: **{_pct(replacement['backfill_target7_rate'])}**",
        f"- FULL_RERANKER promoted Target7 rate: **{_pct(replacement['full_promoted_target7_rate'])}**",
        f"- Backfill minus FULL promotion: **{_pp(replacement['backfill_target7_rate'] - replacement['full_promoted_target7_rate'])}**",
        f"- Bad→winner / bad→bad / winner→winner / winner→bad: **{replacement['bad_to_winner']} / {replacement['bad_to_bad']} / {replacement['winner_to_winner']} / {replacement['winner_to_bad']}**",
        f"- Successful / failed / possible / capture: **{replacement['successful_replacements']} / {replacement['failed_replacements']} / {replacement['possible_replacement_slots']} / {_pct(replacement['replacement_capture'])}**",
        "",
        "## 5. Practical June Performance",
        "",
    ]
    rows = []
    for model in ("UNIVERSE", "CONTROL", "FULL_RERANKER", "VETO_ONLY", "TOP10_ORACLE", "FULL_ORACLE"):
        row = _scope_row(practical, "COMBINED", model)
        rows.append([
            model,
            _pct(row.Rank1_mean),
            _pct(row.Rank2_mean),
            _pct(row.Rank3_mean),
            _pct(row.Top2_mean),
            _pct(row.Top3_mean),
            _pct(row.Top3_target7_precision),
            _pct(row.Top3_negative_date_rate),
            _pct(row.Top3_worst),
        ])
    lines.extend(_markdown_table(
        ["Model", "Rank1", "Rank2", "Rank3", "Top2", "Top3", "Top3 Precision", "Negative Dates", "Worst"],
        rows,
    ))
    lines.extend([
        "",
        f"- VETO_ONLY − CONTROL: **{_pp(veto.Top3_mean - control.Top3_mean)}**",
        f"- VETO_ONLY − FULL_RERANKER: **{_pp(veto.Top3_mean - full.Top3_mean)}**",
        f"- VETO_ONLY − UNIVERSE: **{_pp(veto.Top3_mean - universe.Top3_mean)}**",
        f"- TOP10 Oracle Top3: **{_pct(top10.Top3_mean)}**",
        "",
        "## 6. LOW / MID / HIGH Opportunity",
        "",
    ])
    opportunity_rows = []
    for scope in ("LOW", "MID", "HIGH"):
        for model in ("UNIVERSE", "CONTROL", "FULL_RERANKER", "VETO_ONLY"):
            row = _scope_row(practical, scope, model)
            opportunity_rows.append([
                scope,
                model,
                _pct(row.Rank1_mean),
                _pct(row.Top2_mean),
                _pct(row.Top3_mean),
                _pct(row.Top3_excess),
                _pct(row.Top3_target7_precision),
                _pct(row.Top3_negative_date_rate),
                _pct(row.Top3_worst),
            ])
    lines.extend(_markdown_table(
        ["Bucket", "Model", "Rank1", "Top2", "Top3", "Top3 Excess", "Top3 Precision", "Negative Dates", "Worst"],
        opportunity_rows,
    ))
    lines.extend([
        "",
        "## 7. Changed-Date Attribution",
        "",
        f"- Changed / better / worse / equal dates: **{changed['changed_dates']} / {changed['better']} / {changed['worse']} / {changed['equal']}**",
        f"- Changed-date delta mean / median / best / worst: **{_pp(changed['changed_mean'])} / {_pp(changed['changed_median'])} / {_pp(changed['changed_best'])} / {_pp(changed['changed_worst'])}**",
        "",
    ])
    lines.extend(_markdown_table(
        ["Date", "CONTROL Top3", "Vetoed", "Veto Target7", "Backfilled", "Backfill Target7", "FULL promoted", "CONTROL", "FULL", "VETO_ONLY", "Delta"],
        [[
            row["signal_date"],
            _markdown_cell(row["control_top3_codes"]),
            _markdown_cell(row["vetoed_codes"]),
            _markdown_cell(row["vetoed_target7"]),
            _markdown_cell(row["backfilled_codes"]),
            _markdown_cell(row["backfilled_target7"]),
            _markdown_cell(row["full_reranker_promoted_codes"]),
            _pct(row["control_top3_return"]),
            _pct(row["full_reranker_top3_return"]),
            _pct(row["veto_only_top3_return"]),
            _pp(row["veto_only_minus_control"]),
        ] for row in changed_rows],
    ))
    lines.extend([
        "",
        "## 8. Downside Risk Comparison",
        "",
        f"- All-date mean / median delta: **{_pp(changed['all_mean'])} / {_pp(changed['all_median'])}**",
        f"- Positive / negative / zero dates: **{changed['all_positive']} / {changed['all_negative']} / {changed['all_zero']}**",
        f"- CONTROL negative rate / p25 / median / worst: **{_pct(control.Top3_negative_date_rate)} / {_pct(control.Top3_p25)} / {_pct(control.Top3_median)} / {_pct(control.Top3_worst)}**",
        f"- VETO_ONLY negative rate / p25 / median / worst: **{_pct(veto.Top3_negative_date_rate)} / {_pct(veto.Top3_p25)} / {_pct(veto.Top3_median)} / {_pct(veto.Top3_worst)}**",
        "",
        "## 9. Formal Decision",
        "",
    ])
    for index in range(1, 17):
        lines.append(f"- Q{index}: **{'YES' if status[f'Q{index}'] else 'NO'}**")
    lines.extend([
        "",
        f"VETO_ONLY_SIGNAL: **{status['signal']}**",
        "",
        f"FORWARD_STRESS_CANDIDATE: **{status['forward_stress_candidate']}**",
        "",
        f"- VETO_ONLY Top3 > CONTROL: **{'YES' if status['Q4'] else 'NO'}**",
        f"- VETO_ONLY Top3 > Universe: **{'YES' if status['Q5'] else 'NO'}**",
        f"- Top3 precision improved: **{'YES' if status['Q7'] else 'NO'}**",
        f"- LOW downside materially improved: **{'YES' if status['Q13'] and status['Q14'] else 'NO'}**",
        f"- HIGH upside preserved: **{'YES' if status['Q15'] else 'NO'}**",
        f"- Closing-completion / veto route: **{'CONTINUE' if status['signal'] == 'STRONG' else 'STOP'}**",
    ])
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    frozen = load_frozen_artifacts(root)
    oof, policies = build_veto_oof(frozen["oof"])
    bucket_map = load_opportunity_buckets(root, oof["signal_date"].unique())
    practical, summaries = practical_table(oof, bucket_map)
    parity = validate_previous_parity(
        summaries, frozen["practical"], frozen["membership"]
    )
    daily, membership, replacement, changed_rows = membership_and_daily(
        oof, policies, bucket_map
    )
    changed = changed_date_summary(daily)
    leakage = {
        "self": int(pd.to_numeric(frozen["fold_audit"]["self_label_leakage_rows"]).sum()),
        "current": int(pd.to_numeric(frozen["fold_audit"]["current_test_date_leakage_rows"]).sum()),
        "july": 0,
    }
    status = formal_status(
        practical,
        replacement,
        parity["pass"],
        leakage["self"] == leakage["current"] == leakage["july"] == 0,
    )
    candidate_lt3_dates = daily.loc[daily["candidate_count_lt3"], "signal_date"].tolist()
    review = build_review(
        frozen["source_path"].relative_to(root),
        practical,
        replacement,
        changed,
        changed_rows,
        leakage,
        status,
        candidate_lt3_dates,
    )
    outputs = {
        OUTPUT_FILENAMES[0]: dataframe_csv_bytes(daily),
        OUTPUT_FILENAMES[1]: dataframe_csv_bytes(membership),
        OUTPUT_FILENAMES[2]: dataframe_csv_bytes(practical),
        OUTPUT_FILENAMES[3]: review.encode("utf-8"),
    }
    context = {
        "oof": oof,
        "policies": policies,
        "bucket_map": bucket_map,
        "practical": practical,
        "summaries": summaries,
        "parity": parity,
        "leakage": leakage,
        "daily": daily,
        "membership": membership,
        "replacement": replacement,
        "changed": changed,
        "changed_rows": changed_rows,
        "status": status,
        "candidate_lt3_dates": candidate_lt3_dates,
        "source_path": frozen["source_path"],
    }
    return outputs, context


def run_v004c_veto_only_backfill(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve()
    first, context = build_outputs(root_path)
    second, _ = build_outputs(root_path)
    if first.keys() != second.keys() or any(first[name] != second[name] for name in first):
        raise RuntimeError("FATAL: veto-only deterministic rebuild failed")
    output_dir = root_path / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (output_dir / name).write_bytes(first[name])
    context["deterministic_rebuild"] = "PASS"
    context["output_dir"] = output_dir
    return output_dir, context

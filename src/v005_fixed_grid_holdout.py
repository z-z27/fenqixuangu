from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v004a import (
    DEFAULT_TARGET_RETURN_PCT,
    MODEL_ID_V004A,
    SCOPE_WALK_FORWARD,
    _load_manual_models,
    _score_logistic_frame,
    _score_manual_and_hand_models,
    add_scored_model_rank,
    build_scored_candidates_output,
    prepare_v004a_samples,
)
from .v005_set_selector import (
    DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
    DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
    DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
    DEFAULT_EXTREME_PRICE_PENALTY_GRID,
    DEFAULT_EXTREME_VWAP_PENALTY_GRID,
    DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
    DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    DEFAULT_TOP_N,
    DEFAULT_V004A_L2,
    DEFAULT_V004A_POSITIVE_WEIGHT,
    GRID_PARAM_COLUMNS,
    TARGET_COLUMN,
    build_candidate_pool,
    build_combo_candidates,
    build_rule_grid,
    explode_selected_combos,
    prepare_scored_candidates,
    score_combos,
    select_best_combo_by_date,
)
from .v005_fallback_gate import (
    DAILY_COLUMNS as FALLBACK_DAILY_COLUMNS,
    PRIMARY_POLICY,
    REPLACEMENT_COLUMNS,
    SUMMARY_COLUMNS as FALLBACK_SUMMARY_COLUMNS,
    build_baseline,
    build_context,
    build_topn,
    ctx_for_codes,
    is_policy_fallback,
    metrics,
    norm,
    parse_codes,
    replacement_detail,
    summarize,
)

DEFAULT_SAMPLES_FILE = Path("reports/history_samples/2026-06-26_2026-06-30/history_candidates_2026-06-26_2026-06-30.csv")
DEFAULT_COEFFICIENTS_FILE = Path("reports/v004a/grid_v2_scored/v004a_coefficients.csv")
DEFAULT_OUTPUT_DIR = Path("reports/v005_fixed_grid_holdout")
DEFAULT_GRID_ID = 4
DEFAULT_V002_MODEL_LABEL = "v002_top3_control"
DEFAULT_V004A_MODEL_LABEL = "v004a_top3_control"

HOLDOUT_DAILY_COLUMNS = [
    "strategy",
    "signal_date",
    "action",
    "selected_codes",
    "source_codes",
    "selected_grid_id",
    "hit_count",
    "all_hit",
    "avg_high_return",
    "avg_realized_return",
    "rank1_hit",
    "rank2_hit",
    "rank3_hit",
    "v002_codes",
    "v002_hit_count",
    "v002_all_hit",
    "v002_avg_realized_return",
    "v004a_codes",
    "v004a_hit_count",
    "v004a_all_hit",
    "v004a_avg_realized_return",
    "baseline_v005_codes",
    "baseline_v005_hit_count",
    "baseline_v005_all_hit",
    "baseline_v005_avg_realized_return",
    "gate_v002_extreme_vwap_count",
    "gate_v002_extreme_close_low_count",
    "gate_v005_avg_v002_rank",
    "gate_v005_has_risk_ticket",
    "gate_triggered",
]


def run_fixed_grid_holdout(
    samples_file: str | Path | None = DEFAULT_SAMPLES_FILE,
    scored_file: str | Path | None = None,
    coefficients_file: str | Path = DEFAULT_COEFFICIENTS_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = DEFAULT_TOP_N,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    grid_id: int = DEFAULT_GRID_ID,
    coefficient_predict_date: str = "latest",
    v004a_l2: float = DEFAULT_V004A_L2,
    v004a_positive_weight: float = DEFAULT_V004A_POSITIVE_WEIGHT,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scored_file:
        scored_path = Path(scored_file)
        if not scored_path.exists():
            raise RuntimeError(f"missing scored_file: {scored_path}")
        scored = prepare_scored_candidates(scored_path)
        holdout_scored_path = scored_path
        coefficient_meta: dict[str, Any] = {"source": "pre_scored_file"}
        data_quality = pd.DataFrame()
    else:
        if not samples_file:
            raise RuntimeError("either --samples-file or --scored-file is required")
        holdout_scored_path = out_dir / "v005_fixed_grid_holdout_scored_candidates.csv"
        scored, data_quality, coefficient_meta = build_holdout_scored_candidates(
            samples_file=Path(samples_file),
            coefficients_file=Path(coefficients_file),
            output_path=holdout_scored_path,
            coefficient_predict_date=coefficient_predict_date,
            v004a_l2=float(v004a_l2),
            v004a_positive_weight=float(v004a_positive_weight),
            target_return_pct=float(target_return_pct),
        )

    candidate_pool = build_candidate_pool(
        scored,
        candidate_top_k=int(candidate_top_k),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    combo_candidates = build_combo_candidates(candidate_pool, top_n=int(top_n))
    grid = build_default_grid()
    fixed_params = select_grid_params(grid, grid_id=int(grid_id))
    selected_combos = select_best_combo_by_date(score_combos(combo_candidates, fixed_params))
    daily_top3 = explode_selected_combos(selected_combos, candidate_pool, top_n=int(top_n))

    summary, policy_daily, replacement = apply_fixed_policy(
        scored_path=holdout_scored_path,
        selected_combos=selected_combos,
        top_n=int(top_n),
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )

    combo_candidates.to_csv(out_dir / "v005_fixed_grid_combo_candidates.csv", index=False, encoding="utf-8-sig")
    selected_combos.to_csv(out_dir / "v005_fixed_grid_selected_combos.csv", index=False, encoding="utf-8-sig")
    daily_top3.to_csv(out_dir / "v005_fixed_grid_daily_top3.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "v005_fixed_grid_holdout_summary.csv", index=False, encoding="utf-8-sig")
    policy_daily.to_csv(out_dir / "v005_fixed_grid_holdout_daily.csv", index=False, encoding="utf-8-sig")
    replacement.to_csv(out_dir / "v005_fixed_grid_holdout_replacement.csv", index=False, encoding="utf-8-sig")
    if not data_quality.empty:
        data_quality.to_csv(out_dir / "v005_fixed_grid_holdout_data_quality.csv", index=False, encoding="utf-8-sig")

    report_path = out_dir / "v005_fixed_grid_holdout_report.md"
    report_path.write_text(
        make_report(
            samples_file=samples_file,
            scored_file=holdout_scored_path,
            coefficients_file=coefficients_file,
            output_dir=out_dir,
            grid_id=int(grid_id),
            grid_params=fixed_params,
            coefficient_meta=coefficient_meta,
            candidate_pool=candidate_pool,
            combo_candidates=combo_candidates,
            selected_combos=selected_combos,
            daily_top3=daily_top3,
            summary=summary,
            policy_daily=policy_daily,
            replacement=replacement,
        ),
        encoding="utf-8",
    )
    return summary, policy_daily, replacement, daily_top3, report_path


def build_holdout_scored_candidates(
    samples_file: Path,
    coefficients_file: Path,
    output_path: Path,
    coefficient_predict_date: str,
    v004a_l2: float,
    v004a_positive_weight: float,
    target_return_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not samples_file.exists():
        raise RuntimeError(f"missing samples file: {samples_file}")
    raw = pd.read_csv(samples_file, dtype={"code": str})
    samples, feature_info, data_quality = prepare_v004a_samples(raw, target_return_pct=float(target_return_pct))
    beta, feature_columns, coefficient_meta = load_fixed_v004a_beta(
        coefficients_file=coefficients_file,
        coefficient_predict_date=coefficient_predict_date,
        v004a_l2=float(v004a_l2),
        v004a_positive_weight=float(v004a_positive_weight),
    )
    missing_features = [column for column in feature_columns if column not in samples.columns]
    if missing_features:
        raise RuntimeError(
            "holdout samples are missing v004a coefficient features: "
            + ", ".join(missing_features)
            + "; rerun history sample generation with the same factor columns used by v004a."
        )

    manual_models = _load_manual_models()
    scored_frames = _score_manual_and_hand_models(samples, manual_models)
    scored_frames.append(
        _score_logistic_frame(
            samples,
            beta=beta,
            feature_columns=feature_columns,
            model_id=MODEL_ID_V004A,
            l2=float(v004a_l2),
            positive_weight=float(v004a_positive_weight),
            feature_set=",".join(feature_columns),
            interaction_set=feature_info.get("interaction_set", "none"),
        )
    )
    scored = add_scored_model_rank(pd.concat(scored_frames, ignore_index=True))
    output = build_scored_candidates_output(scored, feature_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    prepared = prepare_scored_candidates(output_path)
    return prepared, data_quality, coefficient_meta


def load_fixed_v004a_beta(
    coefficients_file: Path,
    coefficient_predict_date: str,
    v004a_l2: float,
    v004a_positive_weight: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not coefficients_file.exists():
        raise RuntimeError(
            f"missing coefficients file: {coefficients_file}; "
            "run the prior v004a research once, or pass --scored-file with already scored holdout rows."
        )
    coef = pd.read_csv(coefficients_file)
    required = ["model_id", "evaluation_scope", "l2", "positive_weight", "predict_date", "term", "coefficient"]
    missing = [column for column in required if column not in coef.columns]
    if missing:
        raise RuntimeError(f"coefficients file missing columns: {missing}")
    frame = coef[
        (coef["model_id"].astype(str) == MODEL_ID_V004A)
        & (coef["evaluation_scope"].astype(str) == SCOPE_WALK_FORWARD)
        & (pd.to_numeric(coef["l2"], errors="coerce").sub(float(v004a_l2)).abs() <= 1e-9)
        & (pd.to_numeric(coef["positive_weight"], errors="coerce").sub(float(v004a_positive_weight)).abs() <= 1e-9)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"no v004a coefficients for l2={v004a_l2:g}, positive_weight={v004a_positive_weight:g}")
    frame["predict_date"] = frame["predict_date"].astype(str)
    if coefficient_predict_date and coefficient_predict_date != "latest":
        chosen_date = str(coefficient_predict_date)
    else:
        chosen_date = max(frame["predict_date"].dropna().astype(str).tolist())
    chosen = frame[frame["predict_date"] == chosen_date].copy()
    if chosen.empty:
        available = sorted(frame["predict_date"].dropna().astype(str).unique().tolist())
        raise RuntimeError(f"predict_date={chosen_date!r} not found in coefficients. Available tail: {available[-10:]}")
    if "fold_index" in chosen.columns:
        max_fold = pd.to_numeric(chosen["fold_index"], errors="coerce").max()
        chosen = chosen[pd.to_numeric(chosen["fold_index"], errors="coerce") == max_fold].copy()
    feature_set = str(chosen["feature_set"].dropna().astype(str).iloc[0]) if "feature_set" in chosen.columns else ""
    feature_columns = [part.strip() for part in feature_set.split(",") if part.strip()]
    if not feature_columns:
        feature_columns = [str(term) for term in chosen["term"].astype(str).tolist() if str(term) != "intercept"]
    coef_by_term = dict(zip(chosen["term"].astype(str), pd.to_numeric(chosen["coefficient"], errors="coerce")))
    missing_terms = [term for term in ["intercept", *feature_columns] if term not in coef_by_term]
    if missing_terms:
        raise RuntimeError(f"chosen coefficient fold is missing terms: {missing_terms}")
    beta = np.array([float(coef_by_term["intercept"]), *[float(coef_by_term[column]) for column in feature_columns]], dtype=float)
    meta = {
        "coefficients_file": str(coefficients_file),
        "coefficient_predict_date": chosen_date,
        "coefficient_terms": len(feature_columns),
        "v004a_l2": float(v004a_l2),
        "v004a_positive_weight": float(v004a_positive_weight),
    }
    if "train_start" in chosen.columns:
        meta["coefficient_train_start"] = str(chosen["train_start"].dropna().astype(str).iloc[0])
    if "train_end" in chosen.columns:
        meta["coefficient_train_end"] = str(chosen["train_end"].dropna().astype(str).iloc[0])
    return beta, feature_columns, meta


def build_default_grid() -> pd.DataFrame:
    return build_rule_grid(
        min_total_rank_weight_grid=DEFAULT_MIN_TOTAL_RANK_WEIGHT_GRID,
        avg_total_rank_weight_grid=DEFAULT_AVG_TOTAL_RANK_WEIGHT_GRID,
        contains_v004a_top3_bonus_grid=DEFAULT_CONTAINS_V004A_TOP3_BONUS_GRID,
        contains_v002_top3_bonus_grid=DEFAULT_CONTAINS_V002_TOP3_BONUS_GRID,
        extreme_vwap_penalty_grid=DEFAULT_EXTREME_VWAP_PENALTY_GRID,
        extreme_close_low_penalty_grid=DEFAULT_EXTREME_CLOSE_LOW_PENALTY_GRID,
        extreme_price_penalty_grid=DEFAULT_EXTREME_PRICE_PENALTY_GRID,
        rank_dispersion_weight_grid=DEFAULT_RANK_DISPERSION_WEIGHT_GRID,
    )


def select_grid_params(grid: pd.DataFrame, grid_id: int) -> pd.Series:
    row = grid[pd.to_numeric(grid["grid_id"], errors="coerce") == int(grid_id)].copy()
    if row.empty:
        raise RuntimeError(f"grid_id={grid_id} not found in default v005 grid")
    return row.iloc[0]


def apply_fixed_policy(
    scored_path: Path,
    selected_combos: pd.DataFrame,
    top_n: int,
    v004a_l2: float,
    v004a_positive_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ctx = build_context(scored_path, v004a_l2, v004a_positive_weight)
    history = selected_combos[["signal_date", "grid_id"]].copy().rename(columns={"grid_id": "selected_grid_id"})
    history["grid_id"] = history["selected_grid_id"]
    baseline = build_baseline(history[["signal_date", "grid_id"]], selected_combos, ctx, top_n)
    v002 = build_topn(ctx, "v002_model_rank", top_n)
    v004a = build_topn(ctx, "v004a_model_rank", top_n)

    daily_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    for _, base in baseline.iterrows():
        date = str(base["signal_date"])
        day_ctx = ctx[ctx["signal_date"] == date].copy()
        v002_day = _one_date_row(v002, date, "v002")
        v004a_day = _one_date_row(v004a, date, "v004a")
        baseline_codes = parse_codes(base["codes"])
        v002_codes = parse_codes(v002_day["codes"])
        v004a_codes = parse_codes(v004a_day["codes"])
        choices = [
            ("baseline_v005_fixed_grid", "keep_v005_fixed_grid", baseline_codes, baseline_codes, False),
            policy_choice(base, baseline_codes, v002_codes),
            (DEFAULT_V002_MODEL_LABEL, "control_v002_top3", v002_codes, v002_codes, False),
            (DEFAULT_V004A_MODEL_LABEL, "control_v004a_top3", v004a_codes, v004a_codes, False),
        ]
        for strategy, action, selected_codes, source_codes, triggered in choices:
            selected = ctx_for_codes(day_ctx, selected_codes)
            met = metrics(selected, selected_codes, top_n)
            daily_rows.append(
                {
                    "strategy": strategy,
                    "signal_date": date,
                    "action": action,
                    "selected_codes": ",".join(norm(selected_codes)),
                    "source_codes": ",".join(norm(source_codes)),
                    "selected_grid_id": int(base["selected_grid_id"]),
                    **met,
                    "v002_codes": v002_day["codes"],
                    "v002_hit_count": int(v002_day["hit_count"]),
                    "v002_all_hit": bool(v002_day["all_hit"]),
                    "v002_avg_realized_return": float(v002_day["avg_realized_return"]),
                    "v004a_codes": v004a_day["codes"],
                    "v004a_hit_count": int(v004a_day["hit_count"]),
                    "v004a_all_hit": bool(v004a_day["all_hit"]),
                    "v004a_avg_realized_return": float(v004a_day["avg_realized_return"]),
                    "baseline_v005_codes": base["codes"],
                    "baseline_v005_hit_count": int(base["hit_count"]),
                    "baseline_v005_all_hit": bool(base["all_hit"]),
                    "baseline_v005_avg_realized_return": float(base["avg_realized_return"]),
                    "gate_v002_extreme_vwap_count": int(base["v002_extreme_vwap_count"]),
                    "gate_v002_extreme_close_low_count": int(base["v002_extreme_close_low_count"]),
                    "gate_v005_avg_v002_rank": float(base["v005_avg_v002_rank"]),
                    "gate_v005_has_risk_ticket": bool(base["v005_has_risk_ticket"]),
                    "gate_triggered": bool(triggered),
                }
            )
            replacement_rows.extend(
                replacement_detail(strategy, date, action, norm(selected_codes), baseline_codes, v002_codes, v004a_codes, day_ctx)
            )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        daily = pd.DataFrame(columns=HOLDOUT_DAILY_COLUMNS)
    else:
        daily = daily[HOLDOUT_DAILY_COLUMNS].sort_values(["strategy", "signal_date"])
    summary = summarize(daily, top_n)[FALLBACK_SUMMARY_COLUMNS] if not daily.empty else pd.DataFrame(columns=FALLBACK_SUMMARY_COLUMNS)
    replacement = pd.DataFrame(replacement_rows)
    if replacement.empty:
        replacement = pd.DataFrame(columns=REPLACEMENT_COLUMNS)
    else:
        replacement = replacement[REPLACEMENT_COLUMNS].sort_values(["strategy", "signal_date", "selection_bucket", "code"])
    return summary, daily, replacement


def policy_choice(base: pd.Series, baseline_codes: list[str], v002_codes: list[str]) -> tuple[str, str, list[str], list[str], bool]:
    if is_policy_fallback(base):
        return PRIMARY_POLICY, "fallback_to_v002_regime_policy", v002_codes, v002_codes, True
    return PRIMARY_POLICY, "keep_v005_fixed_grid", baseline_codes, baseline_codes, False


def _one_date_row(frame: pd.DataFrame, date: str, label: str) -> pd.Series:
    row = frame[frame["signal_date"].astype(str) == str(date)].copy()
    if row.empty:
        raise RuntimeError(f"missing {label} TopN row for signal_date={date}")
    return row.iloc[0]


def make_report(
    samples_file: str | Path | None,
    scored_file: Path,
    coefficients_file: str | Path,
    output_dir: Path,
    grid_id: int,
    grid_params: pd.Series,
    coefficient_meta: dict[str, Any],
    candidate_pool: pd.DataFrame,
    combo_candidates: pd.DataFrame,
    selected_combos: pd.DataFrame,
    daily_top3: pd.DataFrame,
    summary: pd.DataFrame,
    policy_daily: pd.DataFrame,
    replacement: pd.DataFrame,
) -> str:
    lines = [
        "# v005 fixed-grid holdout test",
        "",
        "## Scope",
        "",
        "This report evaluates the locked v005 set selector on a holdout samples file.",
        "It does not run factor discovery, does not select a new grid, and does not tune the fallback policy.",
        "",
        "## Configuration",
        "",
        f"- samples file: `{samples_file or ''}`",
        f"- scored file: `{scored_file}`",
        f"- coefficients file: `{coefficients_file}`",
        f"- output dir: `{output_dir}`",
        f"- fixed v005 grid_id: `{grid_id}`",
        f"- candidate_pool rows: `{len(candidate_pool)}`",
        f"- combo candidate rows: `{len(combo_candidates)}`",
        f"- selected combo rows: `{len(selected_combos)}`",
        f"- daily top3 rows: `{len(daily_top3)}`",
        "",
        "## Fixed coefficient metadata",
        "",
    ]
    for key, value in coefficient_meta.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Fixed grid params", ""])
    grid_df = pd.DataFrame([{column: grid_params[column] for column in GRID_PARAM_COLUMNS if column in grid_params.index}])
    lines.extend(md_table(grid_df, GRID_PARAM_COLUMNS))
    lines.extend(["", "## Strategy summary", ""])
    lines.extend(md_table(summary, FALLBACK_SUMMARY_COLUMNS))
    lines.extend(["", "## Daily policy rows", ""])
    lines.extend(md_table(policy_daily, HOLDOUT_DAILY_COLUMNS))
    lines.extend(["", "## Fixed-grid selected combos", ""])
    lines.extend(
        md_table(
            selected_combos,
            [
                "grid_id",
                "signal_date",
                "combo_index",
                "codes",
                "combo_score",
                "hit_count",
                "all_hit",
                "avg_high_return",
                "avg_realized_return",
                "avg_v004a_rank",
                "avg_v002_rank",
                "extreme_price_count",
                "extreme_vwap_count",
                "extreme_close_low_count",
                "min_total_rank",
                "avg_total_rank",
            ],
        )
    )
    lines.extend(["", "## Replacement detail", ""])
    lines.extend(md_table(replacement, REPLACEMENT_COLUMNS))
    return "\n".join(lines)


def md_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df is None or df.empty:
        return ["_No rows._"]
    use = [column for column in columns if column in df.columns]
    rows = ["| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for _, row in df[use].iterrows():
        rows.append("| " + " | ".join(fmt(row[column]) for column in use) + " |")
    return rows


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed-grid v005 holdout evaluation without retuning.")
    parser.add_argument("--samples-file", default=str(DEFAULT_SAMPLES_FILE), help="Holdout history_candidates CSV. Ignored when --scored-file is provided.")
    parser.add_argument("--scored-file", default=None, help="Optional pre-scored holdout candidates CSV; skips v004a/v002 holdout scoring.")
    parser.add_argument("--coefficients-file", default=str(DEFAULT_COEFFICIENTS_FILE), help="Existing v004a_coefficients.csv used to score holdout samples.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--grid-id", type=int, default=DEFAULT_GRID_ID)
    parser.add_argument("--coefficient-predict-date", default="latest", help="Use coefficients from this predict_date, or 'latest'.")
    parser.add_argument("--v004a-l2", type=float, default=DEFAULT_V004A_L2)
    parser.add_argument("--v004a-positive-weight", type=float, default=DEFAULT_V004A_POSITIVE_WEIGHT)
    parser.add_argument("--target-return-pct", type=float, default=DEFAULT_TARGET_RETURN_PCT)
    args = parser.parse_args(argv)

    summary, daily, replacement, daily_top3, report_path = run_fixed_grid_holdout(
        samples_file=args.samples_file,
        scored_file=args.scored_file,
        coefficients_file=args.coefficients_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        candidate_top_k=args.candidate_top_k,
        grid_id=args.grid_id,
        coefficient_predict_date=args.coefficient_predict_date,
        v004a_l2=args.v004a_l2,
        v004a_positive_weight=args.v004a_positive_weight,
        target_return_pct=args.target_return_pct,
    )
    print(f"summary rows: {len(summary)}")
    print(f"daily rows: {len(daily)}")
    print(f"replacement rows: {len(replacement)}")
    print(f"daily top3 rows: {len(daily_top3)}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best strategy: {best['strategy']}")
        print(f"best top3_all_hit_rate: {float(best['top3_all_hit_rate']):.4f}")
        print(f"best avg_realized_return: {float(best['avg_top3_realized_return']):.4f}")
    print(f"markdown: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_MANIFEST = PROJECT_ROOT / "configs" / "policy_v005_v1.json"


@dataclass(frozen=True)
class FallbackGateConfig:
    extreme_vwap_count_min: int
    extreme_close_low_confirm_count_min: int
    extreme_close_low_dominant_count_min: int
    v005_avg_v002_rank_min: float


@dataclass(frozen=True)
class PolicyConfig:
    manifest_path: Path
    schema_version: int
    policy_version: str
    policy_id: str
    deployment_status: str
    research_only: bool
    metric_scope: str
    ranking_model_path: Path
    ranking_model_sha256: str
    ranking_model_id: str
    signal_score_weights: tuple[tuple[str, float], ...]
    coefficients_path: Path
    coefficients_sha256: str
    coefficient_model_id: str
    coefficient_predict_date: str
    coefficient_train_start: str
    coefficient_train_end: str
    v004a_l2: float
    v004a_positive_weight: float
    grid_id: int
    candidate_top_k: int
    top_n: int
    fallback_gate: FallbackGateConfig
    min_forward_dates: int

    def provenance(self) -> dict[str, Any]:
        return {
            "policy_manifest": str(self.manifest_path),
            "policy_version": self.policy_version,
            "policy_id": self.policy_id,
            "deployment_status": self.deployment_status,
            "research_only": self.research_only,
            "metric_scope": self.metric_scope,
            "ranking_model_path": str(self.ranking_model_path),
            "ranking_model_normalized_sha256": self.ranking_model_sha256,
            "ranking_model_id": self.ranking_model_id,
            "signal_score_weights": json.dumps(dict(self.signal_score_weights), sort_keys=True),
            "coefficients_file": str(self.coefficients_path),
            "coefficients_normalized_sha256": self.coefficients_sha256,
            "coefficient_predict_date": self.coefficient_predict_date,
            "coefficient_train_start": self.coefficient_train_start,
            "coefficient_train_end": self.coefficient_train_end,
            "v004a_l2": self.v004a_l2,
            "v004a_positive_weight": self.v004a_positive_weight,
            "grid_id": self.grid_id,
            "candidate_top_k": self.candidate_top_k,
            "top_n": self.top_n,
            "min_forward_dates": self.min_forward_dates,
        }


def normalized_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_policy_config(path: str | Path = DEFAULT_POLICY_MANIFEST) -> PolicyConfig:
    manifest_path = _resolve_project_path(path)
    if not manifest_path.is_file():
        raise RuntimeError(f"missing policy manifest: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("schema_version", "policy_version", "policy_id", "deployment_status", "ranking_model", "signal_score", "coefficients", "selector", "fallback_gate", "readiness"):
        if key not in raw:
            raise RuntimeError(f"policy manifest missing key: {key}")

    ranking_model = raw["ranking_model"]
    coefficients = raw["coefficients"]
    selector = raw["selector"]
    gate = raw["fallback_gate"]
    readiness = raw["readiness"]
    ranking_model_path = _resolve_project_path(ranking_model["path"])
    if not ranking_model_path.is_file():
        raise RuntimeError(f"missing frozen ranking model: {ranking_model_path}")
    expected_ranking_sha256 = str(ranking_model["normalized_sha256"]).lower()
    actual_ranking_sha256 = normalized_sha256(ranking_model_path)
    if actual_ranking_sha256 != expected_ranking_sha256:
        raise RuntimeError(
            f"frozen ranking model checksum mismatch: expected={expected_ranking_sha256}, actual={actual_ranking_sha256}, file={ranking_model_path}"
        )

    coefficients_path = _resolve_project_path(coefficients["path"])
    if not coefficients_path.is_file():
        raise RuntimeError(f"missing frozen coefficient artifact: {coefficients_path}")
    expected_sha256 = str(coefficients["normalized_sha256"]).lower()
    actual_sha256 = normalized_sha256(coefficients_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"frozen coefficient checksum mismatch: expected={expected_sha256}, actual={actual_sha256}, file={coefficients_path}"
        )

    train_start = _parse_date(coefficients["train_start"], "coefficients.train_start")
    train_end = _parse_date(coefficients["train_end"], "coefficients.train_end")
    predict_date = _parse_date(coefficients["predict_date"], "coefficients.predict_date")
    if train_start > train_end or train_end >= predict_date:
        raise RuntimeError("policy coefficient dates must satisfy train_start <= train_end < predict_date")

    config = PolicyConfig(
        manifest_path=manifest_path,
        schema_version=int(raw["schema_version"]),
        policy_version=str(raw["policy_version"]),
        policy_id=str(raw["policy_id"]),
        deployment_status=str(raw["deployment_status"]),
        research_only=bool(raw.get("research_only", True)),
        metric_scope=str(raw.get("metric_scope", "opportunity_proxy")),
        ranking_model_path=ranking_model_path,
        ranking_model_sha256=expected_ranking_sha256,
        ranking_model_id=str(ranking_model["model_id"]),
        signal_score_weights=tuple((str(key), float(value)) for key, value in raw["signal_score"].items()),
        coefficients_path=coefficients_path,
        coefficients_sha256=expected_sha256,
        coefficient_model_id=str(coefficients["model_id"]),
        coefficient_predict_date=predict_date.isoformat(),
        coefficient_train_start=train_start.isoformat(),
        coefficient_train_end=train_end.isoformat(),
        v004a_l2=float(coefficients["l2"]),
        v004a_positive_weight=float(coefficients["positive_weight"]),
        grid_id=int(selector["grid_id"]),
        candidate_top_k=int(selector["candidate_top_k"]),
        top_n=int(selector["top_n"]),
        fallback_gate=FallbackGateConfig(
            extreme_vwap_count_min=int(gate["extreme_vwap_count_min"]),
            extreme_close_low_confirm_count_min=int(gate["extreme_close_low_confirm_count_min"]),
            extreme_close_low_dominant_count_min=int(gate["extreme_close_low_dominant_count_min"]),
            v005_avg_v002_rank_min=float(gate["v005_avg_v002_rank_min"]),
        ),
        min_forward_dates=int(readiness["min_forward_dates"]),
    )
    _validate_policy_values(config)
    return config


def validate_policy_for_signal_date(config: PolicyConfig, signal_date: str) -> None:
    validate_model_dates_for_signal_date(
        predict_date=config.coefficient_predict_date,
        train_end=config.coefficient_train_end,
        signal_date=signal_date,
    )


def validate_model_dates_for_signal_date(predict_date: str, train_end: str, signal_date: str) -> None:
    signal = _parse_date(signal_date, "signal_date")
    predict = _parse_date(predict_date, "coefficient_predict_date")
    train_end_date = _parse_date(train_end, "coefficient_train_end")
    if predict > signal:
        raise RuntimeError(
            f"coefficient predict_date={predict.isoformat()} is after signal_date={signal.isoformat()}; refusing future-model use"
        )
    if train_end_date >= signal:
        raise RuntimeError(
            f"coefficient train_end={train_end_date.isoformat()} is not before signal_date={signal.isoformat()}; refusing lookahead"
        )


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise RuntimeError(f"invalid ISO date for {field}: {value!r}") from exc


def _validate_policy_values(config: PolicyConfig) -> None:
    from .config import StrategyConfig

    if config.schema_version != 1:
        raise RuntimeError(f"unsupported policy schema_version={config.schema_version}")
    if config.deployment_status != "shadow_only" or not config.research_only:
        raise RuntimeError("v005 policy must remain research_only with deployment_status=shadow_only")
    if config.top_n <= 0 or config.candidate_top_k < config.top_n:
        raise RuntimeError("policy selector must satisfy 0 < top_n <= candidate_top_k")
    if config.grid_id <= 0 or config.min_forward_dates <= 0:
        raise RuntimeError("policy grid_id and min_forward_dates must be positive")
    if config.fallback_gate.extreme_vwap_count_min > config.top_n:
        raise RuntimeError("fallback extreme_vwap_count_min cannot exceed top_n")
    if config.fallback_gate.extreme_close_low_confirm_count_min > config.top_n:
        raise RuntimeError("fallback extreme_close_low_confirm_count_min cannot exceed top_n")
    if config.fallback_gate.extreme_close_low_dominant_count_min > config.top_n:
        raise RuntimeError("fallback extreme_close_low_dominant_count_min cannot exceed top_n")
    strategy = StrategyConfig()
    expected_weights = {
        "trend_hold_weight": strategy.trend_hold_weight,
        "graph_quality_weight": strategy.graph_quality_weight,
        "active_cooling_weight": strategy.active_cooling_weight,
        "entry_width_weight": strategy.entry_width_weight,
        "theme_weight": strategy.theme_weight,
        "support_weight": strategy.support_weight,
    }
    if dict(config.signal_score_weights) != expected_weights:
        raise RuntimeError("policy manifest signal_score weights disagree with StrategyConfig")


DEFAULT_POLICY = load_policy_config()

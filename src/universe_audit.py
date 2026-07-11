from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


UNIVERSE_SNAPSHOT_SCHEMA_VERSION = 1
UNIVERSE_STAGES = (
    "raw_source_pool",
    "signal_pool",
    "eligible_pool",
    "scorable_pool",
    "v004a_topk",
    "v002_topk",
    "v005_candidate_pool",
    "final_top3",
)
UNIVERSE_SNAPSHOT_MODES = ("create-or-verify", "verify-only", "off")


class UniverseAuditError(RuntimeError):
    pass


class UniverseSnapshotError(UniverseAuditError):
    def __init__(self, message: str, status: str, canonical_manifest_path: Path) -> None:
        super().__init__(message)
        self.status = status
        self.canonical_manifest_path = canonical_manifest_path


class UniverseSnapshotMismatch(UniverseSnapshotError):
    pass


@dataclass(frozen=True)
class StageSnapshot:
    stage: str
    frame: pd.DataFrame
    key_columns: tuple[str, ...]
    row_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage not in UNIVERSE_STAGES:
            raise ValueError(f"unknown universe stage: {self.stage}")
        missing_keys = [column for column in self.key_columns if column not in self.row_columns]
        if missing_keys:
            raise ValueError(f"stage {self.stage} key columns must be present in row_columns: {missing_keys}")


def normalize_code(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            text = str(int(numeric))
        else:
            text = str(value).strip()
    else:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
    if not text:
        return ""
    return text.zfill(6) if text.isdigit() else text


def normalize_code_series(values: pd.Series) -> pd.Series:
    return values.map(normalize_code).astype(str)


def code_set_sha256(values: Iterable[Any] | pd.DataFrame, code_column: str = "code") -> str:
    if isinstance(values, pd.DataFrame):
        source = values.get(code_column, pd.Series(dtype=object)).tolist()
    else:
        source = list(values)
    codes = sorted({code for code in (normalize_code(value) for value in source) if code})
    return _sha256_text("\n".join(codes))


def key_set_sha256(frame: pd.DataFrame, key_columns: Iterable[str]) -> str:
    keys = sorted(set(_canonical_key_values(frame, tuple(key_columns))))
    return _sha256_text("\n".join(keys))


def canonical_rows_sha256(
    frame: pd.DataFrame,
    columns: Iterable[str],
    sort_columns: Iterable[str] | None = None,
) -> str:
    lines = canonical_row_lines(frame, columns=columns, sort_columns=sort_columns)
    return _sha256_text("\n".join(lines))


def canonical_row_lines(
    frame: pd.DataFrame,
    columns: Iterable[str],
    sort_columns: Iterable[str] | None = None,
) -> list[str]:
    selected_columns = tuple(str(column) for column in columns)
    if not selected_columns:
        return []
    normalized = canonicalize_frame(frame, selected_columns)
    if "code" in selected_columns and not normalized.empty:
        normalized = normalized[
            normalized["code"].notna() & normalized["code"].astype(str).ne("")
        ].reset_index(drop=True)
    lines = [
        json.dumps(
            {column: row[column] for column in selected_columns},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in normalized.to_dict(orient="records")
    ]
    if sort_columns:
        sort_names = tuple(str(column) for column in sort_columns)
        unknown = [column for column in sort_names if column not in selected_columns]
        if unknown:
            raise ValueError(f"sort columns are not present in canonical columns: {unknown}")
        records = list(zip(normalized.to_dict(orient="records"), lines))
        records.sort(
            key=lambda item: tuple(_sort_token(item[0].get(column)) for column in sort_names) + (item[1],)
        )
        return [line for _, line in records]
    return sorted(lines)


def canonicalize_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected_columns = tuple(str(column) for column in columns)
    source = frame.copy()
    for column in selected_columns:
        if column not in source.columns:
            source[column] = None
    rows: list[dict[str, Any]] = []
    for raw_row in source[list(selected_columns)].to_dict(orient="records"):
        rows.append({column: _canonical_value(raw_row.get(column), column) for column in selected_columns})
    return pd.DataFrame(rows, columns=list(selected_columns))


def canonical_stage_frame(snapshot: StageSnapshot) -> pd.DataFrame:
    frame = canonicalize_frame(snapshot.frame, snapshot.row_columns)
    if frame.empty:
        return frame
    lines = canonical_row_lines(frame, snapshot.row_columns, snapshot.key_columns)
    decoded = [json.loads(line) for line in lines]
    return pd.DataFrame(decoded, columns=list(snapshot.row_columns))


def stage_identity(snapshot: StageSnapshot) -> dict[str, Any]:
    canonical = canonical_stage_frame(snapshot)
    return {
        "stage": snapshot.stage,
        "row_count": int(len(canonical)),
        "code_count": int(canonical.get("code", pd.Series(dtype=str)).replace("", pd.NA).dropna().nunique()),
        "code_set_sha256": code_set_sha256(canonical),
        "key_set_sha256": key_set_sha256(canonical, snapshot.key_columns),
        "rows_sha256": canonical_rows_sha256(canonical, snapshot.row_columns, snapshot.key_columns),
        "key_columns": list(snapshot.key_columns),
        "row_columns": list(snapshot.row_columns),
    }


def count_duplicate_keys(frame: pd.DataFrame, key_columns: Iterable[str]) -> int:
    keys = tuple(str(column) for column in key_columns)
    canonical = canonicalize_frame(frame, keys)
    if canonical.empty:
        return 0
    return int(canonical.duplicated(list(keys), keep=False).sum())


def require_unique_keys(
    frame: pd.DataFrame,
    key_columns: Iterable[str],
    label: str,
    preview_limit: int = 10,
) -> None:
    keys = tuple(str(column) for column in key_columns)
    canonical = canonicalize_frame(frame, keys)
    if canonical.empty:
        return
    duplicate_mask = canonical.duplicated(list(keys), keep=False)
    if not duplicate_mask.any():
        return
    preview = canonical.loc[duplicate_mask, list(keys)].drop_duplicates().head(int(preview_limit))
    duplicate_keys = ["|".join(str(row[column]) for column in keys) for _, row in preview.iterrows()]
    raise UniverseAuditError(
        f"{label} contains duplicate keys for {list(keys)}: {duplicate_keys}; "
        f"duplicate_count={int(duplicate_mask.sum())}"
    )


def require_requested_signal_date_match(
    frame: pd.DataFrame,
    requested_column: str = "requested_signal_date",
    actual_column: str = "signal_date",
    code_column: str = "code",
) -> None:
    if frame.empty:
        return
    required = [requested_column, actual_column, code_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise UniverseAuditError(f"signal date alignment check missing columns: {missing}")
    requested = frame[requested_column].map(lambda value: _canonical_value(value, requested_column))
    actual = frame[actual_column].map(lambda value: _canonical_value(value, actual_column))
    mismatch = requested.ne(actual)
    if not mismatch.any():
        return
    row = frame.loc[mismatch].iloc[0]
    code = normalize_code(row.get(code_column))
    raise UniverseAuditError(
        "signal_date_mismatch: "
        f"requested_signal_date={row.get(requested_column)!s}, "
        f"signal_date={row.get(actual_column)!s}, code={code}"
    )


def compare_stage_frames(canonical: StageSnapshot, current: StageSnapshot) -> pd.DataFrame:
    if canonical.stage != current.stage:
        raise ValueError(f"stage mismatch: {canonical.stage} != {current.stage}")
    if canonical.key_columns != current.key_columns or canonical.row_columns != current.row_columns:
        raise ValueError(f"stage schema mismatch for {canonical.stage}")
    require_unique_keys(canonical.frame, canonical.key_columns, f"canonical {canonical.stage}")
    require_unique_keys(current.frame, current.key_columns, f"current {current.stage}")
    canonical_map = _rows_by_member_key(canonical)
    current_map = _rows_by_member_key(current)
    rows: list[dict[str, Any]] = []
    all_keys = sorted(set(canonical_map).union(current_map))
    for member_key in all_keys:
        old = canonical_map.get(member_key)
        new = current_map.get(member_key)
        if old is None:
            diff_type = "added"
        elif new is None:
            diff_type = "removed"
        elif old["row_json"] != new["row_json"]:
            diff_type = "changed"
        else:
            continue
        old_row = old["row"] if old else {}
        new_row = new["row"] if new else {}
        item: dict[str, Any] = {
            "stage": canonical.stage,
            "code": new_row.get("code", old_row.get("code", "")),
            "source_trade_date": new_row.get("source_trade_date", old_row.get("source_trade_date")),
            "diff_type": diff_type,
            "canonical_member_key": member_key if old else "",
            "current_member_key": member_key if new else "",
            "canonical_row_json": old["row_json"] if old else "",
            "current_row_json": new["row_json"] if new else "",
        }
        for column in canonical.row_columns:
            item[f"canonical_{column}"] = old_row.get(column)
            item[f"current_{column}"] = new_row.get(column)
        rows.append(item)
    return pd.DataFrame(rows)


def create_or_verify_snapshot(
    requested_signal_date: str,
    stages: Mapping[str, StageSnapshot],
    snapshot_root: Path,
    run_output_dir: Path,
    mode: str = "create-or-verify",
) -> tuple[str, Path, dict[str, Any]]:
    if mode not in UNIVERSE_SNAPSHOT_MODES:
        raise ValueError(f"unsupported universe snapshot mode={mode!r}; expected one of {UNIVERSE_SNAPSHOT_MODES}")
    ordered = _ordered_stages(stages)
    current_manifest = build_snapshot_manifest(requested_signal_date, ordered)
    canonical_dir = Path(snapshot_root) / "history_universe" / str(requested_signal_date) / "canonical"
    canonical_manifest_path = canonical_dir / "manifest.json"
    if mode == "off":
        return "UNVERIFIED_OFF", canonical_manifest_path, current_manifest

    if not canonical_manifest_path.is_file():
        if mode == "verify-only":
            raise UniverseSnapshotError(
                f"canonical universe snapshot is required but missing for {requested_signal_date}: {canonical_manifest_path}",
                status="MISSING_CANONICAL",
                canonical_manifest_path=canonical_manifest_path,
            )
        if canonical_dir.exists() and any(canonical_dir.iterdir()):
            raise UniverseSnapshotError(
                f"canonical snapshot directory is non-empty but manifest is missing; refusing overwrite: {canonical_dir}",
                status="INCOMPLETE_CANONICAL",
                canonical_manifest_path=canonical_manifest_path,
            )
        _write_snapshot(canonical_dir, ordered, current_manifest)
        return "CREATED_CANONICAL", canonical_manifest_path, current_manifest

    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8-sig"))
    canonical_stages = _load_canonical_stages(canonical_dir, canonical_manifest)
    mismatch_rows: list[pd.DataFrame] = []
    mismatch_summaries: list[str] = []
    for stage_name, current in ordered.items():
        if stage_name not in canonical_stages:
            mismatch_summaries.append(f"{stage_name}: missing from canonical manifest")
            continue
        old = canonical_stages[stage_name]
        old_identity = stage_identity(old)
        new_identity = stage_identity(current)
        diff = compare_stage_frames(old, current)
        if not diff.empty:
            mismatch_rows.append(diff)
        if _identity_tuple(old_identity) != _identity_tuple(new_identity):
            added = int((diff.get("diff_type", pd.Series(dtype=str)) == "added").sum()) if not diff.empty else 0
            removed = int((diff.get("diff_type", pd.Series(dtype=str)) == "removed").sum()) if not diff.empty else 0
            changed = int((diff.get("diff_type", pd.Series(dtype=str)) == "changed").sum()) if not diff.empty else 0
            mismatch_summaries.append(
                f"{stage_name}: old_count={old_identity['row_count']} new_count={new_identity['row_count']} "
                f"added={added} removed={removed} changed={changed} "
                f"old_hash={old_identity['rows_sha256']} new_hash={new_identity['rows_sha256']}"
            )
    extra_stages = sorted(set(canonical_stages).difference(ordered))
    mismatch_summaries.extend(f"{stage}: missing from current run" for stage in extra_stages)
    if mismatch_summaries:
        run_output_dir = Path(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        current_dir = run_output_dir / f"universe_snapshot_current_{requested_signal_date}"
        _write_snapshot(current_dir, ordered, current_manifest, allow_existing=True)
        diff_frame = pd.concat(mismatch_rows, ignore_index=True) if mismatch_rows else pd.DataFrame()
        diff_csv = run_output_dir / f"universe_snapshot_diff_{requested_signal_date}.csv"
        diff_md = run_output_dir / f"universe_snapshot_diff_{requested_signal_date}.md"
        diff_frame.to_csv(diff_csv, index=False, encoding="utf-8-sig", lineterminator="\n")
        diff_md.write_text(_snapshot_diff_markdown(requested_signal_date, mismatch_summaries, diff_frame), encoding="utf-8")
        raise UniverseSnapshotMismatch(
            f"universe snapshot mismatch for {requested_signal_date}; " + " | ".join(mismatch_summaries),
            status="SNAPSHOT_MISMATCH",
            canonical_manifest_path=canonical_manifest_path,
        )
    return "VERIFIED_MATCH", canonical_manifest_path, current_manifest


def build_snapshot_manifest(
    requested_signal_date: str,
    stages: Mapping[str, StageSnapshot],
) -> dict[str, Any]:
    return {
        "universe_snapshot_schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "requested_signal_date": str(requested_signal_date),
        "stages": {stage: stage_identity(snapshot) for stage, snapshot in _ordered_stages(stages).items()},
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_canonical_csv(path: Path, snapshot: StageSnapshot) -> None:
    frame = canonical_stage_frame(snapshot)
    csv_frame = frame.copy()
    for column in csv_frame.columns:
        csv_frame[column] = csv_frame[column].map(_csv_value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", na_rep="null")


def _write_snapshot(
    directory: Path,
    stages: Mapping[str, StageSnapshot],
    manifest: Mapping[str, Any],
    allow_existing: bool = False,
) -> None:
    directory = Path(directory)
    if directory.exists() and not allow_existing and any(directory.iterdir()):
        raise UniverseAuditError(f"refusing to overwrite non-empty snapshot directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for stage, snapshot in _ordered_stages(stages).items():
        write_canonical_csv(directory / f"{stage}.csv", snapshot)
    write_json(directory / "manifest.json", manifest)


def _load_canonical_stages(directory: Path, manifest: Mapping[str, Any]) -> dict[str, StageSnapshot]:
    schema_version = int(manifest.get("universe_snapshot_schema_version", -1))
    if schema_version != UNIVERSE_SNAPSHOT_SCHEMA_VERSION:
        raise UniverseAuditError(f"unsupported universe snapshot schema_version={schema_version}")
    stages: dict[str, StageSnapshot] = {}
    for stage, meta in (manifest.get("stages") or {}).items():
        path = Path(directory) / f"{stage}.csv"
        if not path.is_file():
            raise UniverseAuditError(f"canonical snapshot stage file is missing: {path}")
        frame = pd.read_csv(path, dtype={"code": str}, keep_default_na=True)
        snapshot = StageSnapshot(
            stage=str(stage),
            frame=frame,
            key_columns=tuple(meta.get("key_columns") or []),
            row_columns=tuple(meta.get("row_columns") or []),
        )
        actual = stage_identity(snapshot)
        if _identity_tuple(actual) != _identity_tuple(meta):
            raise UniverseAuditError(f"canonical snapshot content does not match manifest for stage={stage}")
        stages[str(stage)] = snapshot
    return stages


def _ordered_stages(stages: Mapping[str, StageSnapshot]) -> dict[str, StageSnapshot]:
    unknown = sorted(set(stages).difference(UNIVERSE_STAGES))
    if unknown:
        raise ValueError(f"unknown universe stages: {unknown}")
    return {stage: stages[stage] for stage in UNIVERSE_STAGES if stage in stages}


def _identity_tuple(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(identity.get("row_count", -1)),
        int(identity.get("code_count", -1)),
        str(identity.get("code_set_sha256", "")),
        str(identity.get("key_set_sha256", "")),
        str(identity.get("rows_sha256", "")),
    )


def _rows_by_member_key(snapshot: StageSnapshot) -> dict[str, dict[str, Any]]:
    canonical = canonical_stage_frame(snapshot)
    output: dict[str, dict[str, Any]] = {}
    for row in canonical.to_dict(orient="records"):
        key = _member_key(row, snapshot.key_columns)
        row_json = json.dumps(
            {column: row.get(column) for column in snapshot.row_columns},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output[key] = {"row": row, "row_json": row_json}
    return output


def _canonical_key_values(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> list[str]:
    canonical = canonicalize_frame(frame, key_columns)
    values: list[str] = []
    for row in canonical.to_dict(orient="records"):
        if "code" in key_columns and not normalize_code(row.get("code")):
            continue
        values.append(_member_key(row, key_columns))
    return values


def _member_key(row: Mapping[str, Any], key_columns: Iterable[str]) -> str:
    return json.dumps([row.get(column) for column in key_columns], ensure_ascii=False, separators=(",", ":"))


def _canonical_value(value: Any, column: str) -> Any:
    if _is_missing(value):
        return None
    if column == "code":
        return normalize_code(value)
    if column.endswith("_date") or column in {"date", "signal_date", "trade_date"}:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    if isinstance(value, (bool,)) or value.__class__.__name__ == "bool_":
        return bool(value)
    if column.endswith("_bool"):
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "no", "n", "f"}:
            return False
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number == 0:
            number = 0.0
        return format(number, ".17g")
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text if text else None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool,)) or missing.__class__.__name__ == "bool_" else False


def _sort_token(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot_diff_markdown(
    requested_signal_date: str,
    summaries: list[str],
    diff: pd.DataFrame,
) -> str:
    lines = [f"# Universe Snapshot Diff {requested_signal_date}", "", "## Summary", ""]
    lines.extend(f"- {summary}" for summary in summaries)
    lines.extend(["", "## Members", ""])
    if diff.empty:
        lines.append("_No member-level rows were available; inspect stage schema differences._")
    else:
        columns = [
            column
            for column in ("stage", "diff_type", "code", "source_trade_date", "canonical_member_key", "current_member_key")
            if column in diff.columns
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in diff[columns].iterrows():
            lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)

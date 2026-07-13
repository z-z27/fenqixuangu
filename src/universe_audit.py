from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping
import uuid

import pandas as pd
from pandas.errors import ParserError


UNIVERSE_SNAPSHOT_SCHEMA_VERSION = 1
CANONICAL_FLOAT_SIGNIFICANT_DIGITS = 12
CANONICAL_MISSING_TEXT = ""
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


def require_nonempty_codes(
    frame: pd.DataFrame,
    code_column: str,
    label: str,
    preview_limit: int = 10,
) -> None:
    if code_column not in frame.columns:
        raise UniverseAuditError(f"{label} is missing required code column {code_column!r}")
    if frame.empty:
        return
    normalized = frame[code_column].map(normalize_code)
    invalid = normalized.eq("") | ~normalized.astype(str).str.fullmatch(r"\d{6}")
    if not invalid.any():
        return
    preview_rows = frame.loc[invalid].head(int(preview_limit))
    preview = [
        {"index": str(index), "value": repr(row.get(code_column))}
        for index, row in preview_rows.iterrows()
    ]
    raise UniverseAuditError(
        f"{label} contains empty or unnormalizable codes in column {code_column!r}: "
        f"invalid_count={int(invalid.sum())}, preview={preview}"
    )


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


def canonicalize_frame(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    stage: str = "",
) -> pd.DataFrame:
    selected_columns = tuple(str(column) for column in columns)
    source = frame.copy()
    for column in selected_columns:
        if column not in source.columns:
            source[column] = None
    rows: list[dict[str, Any]] = []
    for raw_row in source[list(selected_columns)].to_dict(orient="records"):
        code = normalize_code(raw_row.get("code")) if "code" in raw_row else ""
        rows.append(
            {
                column: canonical_scalar_text(
                    raw_row.get(column),
                    column,
                    stage=stage,
                    code=code,
                )
                for column in selected_columns
            }
        )
    return pd.DataFrame(rows, columns=list(selected_columns))


def canonical_stage_frame(snapshot: StageSnapshot) -> pd.DataFrame:
    frame = canonicalize_frame(
        snapshot.frame,
        snapshot.row_columns,
        stage=snapshot.stage,
    )
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
    requested = frame[requested_column].map(
        lambda value: canonical_scalar_text(value, requested_column)
    )
    actual = frame[actual_column].map(
        lambda value: canonical_scalar_text(value, actual_column)
    )
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
    for stage, snapshot in ordered.items():
        require_nonempty_codes(snapshot.frame, "code", stage)
        require_unique_keys(snapshot.frame, snapshot.key_columns, stage)
    current_manifest = build_snapshot_manifest(requested_signal_date, ordered)
    canonical_dir = Path(snapshot_root) / "history_universe" / str(requested_signal_date) / "canonical"
    canonical_manifest_path = canonical_dir / "manifest.json"
    if mode == "off":
        return "UNVERIFIED_OFF", canonical_manifest_path, current_manifest

    if canonical_dir.exists():
        if not canonical_manifest_path.is_file():
            raise UniverseSnapshotError(
                f"canonical snapshot directory exists but manifest is missing: {canonical_dir}",
                status="INCOMPLETE_CANONICAL",
                canonical_manifest_path=canonical_manifest_path,
            )
        _verify_existing_snapshot(
            requested_signal_date=requested_signal_date,
            canonical_dir=canonical_dir,
            current_stages=ordered,
            current_manifest=current_manifest,
            run_output_dir=run_output_dir,
        )
        return "VERIFIED_MATCH", canonical_manifest_path, current_manifest

    if mode == "verify-only":
        raise UniverseSnapshotError(
            f"canonical universe snapshot is required but missing for {requested_signal_date}: {canonical_manifest_path}",
            status="MISSING_CANONICAL",
            canonical_manifest_path=canonical_manifest_path,
        )

    return _create_snapshot_atomically(
        requested_signal_date=requested_signal_date,
        canonical_dir=canonical_dir,
        stages=ordered,
        manifest=current_manifest,
        run_output_dir=run_output_dir,
    )


def _verify_existing_snapshot(
    requested_signal_date: str,
    canonical_dir: Path,
    current_stages: Mapping[str, StageSnapshot],
    current_manifest: Mapping[str, Any],
    run_output_dir: Path,
) -> None:
    canonical_manifest_path = canonical_dir / "manifest.json"
    _, canonical_stages = _load_snapshot(
        canonical_dir,
        expected_requested_signal_date=requested_signal_date,
    )
    mismatch_rows: list[pd.DataFrame] = []
    mismatch_summaries: list[str] = []
    for stage_name, current in current_stages.items():
        if stage_name not in canonical_stages:
            mismatch_summaries.append(f"{stage_name}: missing from canonical manifest")
            continue
        old = canonical_stages[stage_name]
        old_identity = stage_identity(old)
        new_identity = stage_identity(current)
        if old.key_columns != current.key_columns or old.row_columns != current.row_columns:
            mismatch_summaries.append(
                f"{stage_name}: stage schema changed; "
                f"old_keys={list(old.key_columns)} new_keys={list(current.key_columns)} "
                f"old_rows={list(old.row_columns)} new_rows={list(current.row_columns)}"
            )
            continue
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
    extra_stages = sorted(set(canonical_stages).difference(current_stages))
    mismatch_summaries.extend(f"{stage}: missing from current run" for stage in extra_stages)
    if mismatch_summaries:
        run_output_dir = Path(run_output_dir)
        try:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            current_dir = run_output_dir / f"universe_snapshot_current_{requested_signal_date}"
            _write_snapshot(current_dir, current_stages, current_manifest, allow_existing=True)
            diff_frame = pd.concat(mismatch_rows, ignore_index=True) if mismatch_rows else pd.DataFrame()
            diff_csv = run_output_dir / f"universe_snapshot_diff_{requested_signal_date}.csv"
            diff_md = run_output_dir / f"universe_snapshot_diff_{requested_signal_date}.md"
            atomic_write_csv(diff_csv, diff_frame, index=False, encoding="utf-8-sig", lineterminator="\n")
            atomic_write_text(
                diff_md,
                _snapshot_diff_markdown(requested_signal_date, mismatch_summaries, diff_frame),
                encoding="utf-8",
            )
        except Exception as exc:
            raise UniverseSnapshotError(
                f"failed to write universe snapshot mismatch diagnostics for {requested_signal_date}: {exc}",
                status="SNAPSHOT_WRITE_FAILED",
                canonical_manifest_path=canonical_manifest_path,
            ) from exc
        raise UniverseSnapshotMismatch(
            f"universe snapshot mismatch for {requested_signal_date}; " + " | ".join(mismatch_summaries),
            status="SNAPSHOT_MISMATCH",
            canonical_manifest_path=canonical_manifest_path,
        )


def _create_snapshot_atomically(
    requested_signal_date: str,
    canonical_dir: Path,
    stages: Mapping[str, StageSnapshot],
    manifest: Mapping[str, Any],
    run_output_dir: Path,
) -> tuple[str, Path, dict[str, Any]]:
    canonical_dir = Path(canonical_dir)
    canonical_manifest_path = canonical_dir / "manifest.json"
    parent = canonical_dir.parent
    temp_dir = parent / f".canonical.tmp-{uuid.uuid4().hex}"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=False, exist_ok=False)
        _write_snapshot(temp_dir, stages, manifest, allow_existing=True)
        try:
            _, reloaded = _load_snapshot(
                temp_dir,
                expected_requested_signal_date=requested_signal_date,
            )
        except UniverseSnapshotError as exc:
            raise UniverseSnapshotError(
                f"temporary canonical snapshot validation failed for {requested_signal_date}: {exc}",
                status="SNAPSHOT_WRITE_FAILED",
                canonical_manifest_path=canonical_manifest_path,
            ) from exc
        _require_exact_stage_identities(
            expected=stages,
            actual=reloaded,
            manifest_path=canonical_manifest_path,
            status="SNAPSHOT_WRITE_FAILED",
        )

        if canonical_dir.exists():
            _remove_tree(temp_dir)
            _verify_existing_snapshot(
                requested_signal_date=requested_signal_date,
                canonical_dir=canonical_dir,
                current_stages=stages,
                current_manifest=manifest,
                run_output_dir=run_output_dir,
            )
            return "VERIFIED_MATCH", canonical_manifest_path, dict(manifest)

        try:
            os.rename(temp_dir, canonical_dir)
        except OSError as exc:
            if canonical_dir.exists():
                _remove_tree(temp_dir)
                _verify_existing_snapshot(
                    requested_signal_date=requested_signal_date,
                    canonical_dir=canonical_dir,
                    current_stages=stages,
                    current_manifest=manifest,
                    run_output_dir=run_output_dir,
                )
                return "VERIFIED_MATCH", canonical_manifest_path, dict(manifest)
            raise UniverseSnapshotError(
                f"failed to atomically publish canonical snapshot {canonical_dir}: {exc}",
                status="SNAPSHOT_WRITE_FAILED",
                canonical_manifest_path=canonical_manifest_path,
            ) from exc

        try:
            _, published = _load_snapshot(
                canonical_dir,
                expected_requested_signal_date=requested_signal_date,
            )
            _require_exact_stage_identities(
                expected=stages,
                actual=published,
                manifest_path=canonical_manifest_path,
                status="CORRUPT_CANONICAL",
            )
        except UniverseSnapshotError:
            raise
        return "CREATED_CANONICAL", canonical_manifest_path, dict(manifest)
    except UniverseSnapshotError:
        raise
    except Exception as exc:
        raise UniverseSnapshotError(
            f"failed to write canonical snapshot for {requested_signal_date}: {exc}",
            status="SNAPSHOT_WRITE_FAILED",
            canonical_manifest_path=canonical_manifest_path,
        ) from exc
    finally:
        _remove_tree(temp_dir)


def build_snapshot_manifest(
    requested_signal_date: str,
    stages: Mapping[str, StageSnapshot],
) -> dict[str, Any]:
    return {
        "universe_snapshot_schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "requested_signal_date": str(requested_signal_date),
        "stages": {stage: stage_identity(snapshot) for stage, snapshot in _ordered_stages(stages).items()},
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_file_path(path)
    try:
        with temp_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path.parent)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(Path(path), text.encode(encoding))


def atomic_write_csv(
    path: Path,
    frame: pd.DataFrame,
    *,
    index: bool = False,
    encoding: str = "utf-8-sig",
    lineterminator: str = "\n",
    na_rep: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_file_path(path)
    try:
        with temp_path.open("x", encoding=encoding, newline="") as handle:
            frame.to_csv(
                handle,
                index=index,
                lineterminator=lineterminator,
                na_rep=na_rep,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path.parent)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_copy_file(source: Path, destination: Path) -> None:
    source = Path(source)
    if not source.is_file():
        raise OSError(f"publish source file is missing: {source}")
    atomic_write_bytes(Path(destination), source.read_bytes())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_canonical_csv(path: Path, snapshot: StageSnapshot) -> None:
    frame = canonical_stage_frame(snapshot)
    atomic_write_csv(
        Path(path),
        frame,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        na_rep=CANONICAL_MISSING_TEXT,
    )


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
    manifest_path = Path(directory) / "manifest.json"
    try:
        schema_version = int(manifest.get("universe_snapshot_schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise UniverseSnapshotError(
            f"invalid canonical snapshot schema version in {manifest_path}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        ) from exc
    if schema_version != UNIVERSE_SNAPSHOT_SCHEMA_VERSION:
        raise UniverseSnapshotError(
            f"unsupported universe snapshot schema_version={schema_version}: {manifest_path}",
            status="UNSUPPORTED_SCHEMA",
            canonical_manifest_path=manifest_path,
        )
    stage_metadata = manifest.get("stages")
    if not isinstance(stage_metadata, Mapping) or not stage_metadata:
        raise UniverseSnapshotError(
            f"canonical snapshot manifest has no valid stages mapping: {manifest_path}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        )
    stages: dict[str, StageSnapshot] = {}
    for stage, meta in stage_metadata.items():
        if stage not in UNIVERSE_STAGES or not isinstance(meta, Mapping):
            raise UniverseSnapshotError(
                f"canonical snapshot manifest has invalid stage metadata for {stage!r}: {manifest_path}",
                status="CORRUPT_CANONICAL",
                canonical_manifest_path=manifest_path,
            )
        path = Path(directory) / f"{stage}.csv"
        if not path.is_file():
            raise UniverseSnapshotError(
                f"canonical snapshot stage file is missing: {path}",
                status="INCOMPLETE_CANONICAL",
                canonical_manifest_path=manifest_path,
            )
        try:
            frame = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
            snapshot = StageSnapshot(
                stage=str(stage),
                frame=frame,
                key_columns=tuple(meta.get("key_columns") or []),
                row_columns=tuple(meta.get("row_columns") or []),
            )
            require_nonempty_codes(snapshot.frame, "code", f"canonical {stage}")
            require_unique_keys(snapshot.frame, snapshot.key_columns, f"canonical {stage}")
            actual = stage_identity(snapshot)
            if _full_identity_tuple(actual) != _full_identity_tuple(meta):
                raise UniverseSnapshotError(
                    f"canonical snapshot content does not match manifest for stage={stage}",
                    status="CORRUPT_CANONICAL",
                    canonical_manifest_path=manifest_path,
                )
        except UniverseSnapshotError:
            raise
        except (UnicodeDecodeError, OSError, ParserError, ValueError, TypeError, UniverseAuditError) as exc:
            raise UniverseSnapshotError(
                f"failed to read canonical snapshot stage {stage}: {path}: {exc}",
                status="CORRUPT_CANONICAL",
                canonical_manifest_path=manifest_path,
            ) from exc
        stages[str(stage)] = snapshot
    extra_csv = sorted(
        path.stem
        for path in Path(directory).glob("*.csv")
        if path.stem not in stage_metadata
    )
    if extra_csv:
        raise UniverseSnapshotError(
            f"canonical snapshot contains stage files absent from manifest: {extra_csv}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        )
    return stages


def _load_snapshot(
    directory: Path,
    expected_requested_signal_date: str,
) -> tuple[dict[str, Any], dict[str, StageSnapshot]]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise UniverseSnapshotError(
            f"canonical snapshot manifest is missing: {manifest_path}",
            status="INCOMPLETE_CANONICAL",
            canonical_manifest_path=manifest_path,
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UniverseSnapshotError(
            f"canonical snapshot manifest is malformed: {manifest_path}: {exc}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        ) from exc
    except OSError as exc:
        raise UniverseSnapshotError(
            f"canonical snapshot manifest cannot be read: {manifest_path}: {exc}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        ) from exc
    if not isinstance(manifest, dict):
        raise UniverseSnapshotError(
            f"canonical snapshot manifest must be a JSON object: {manifest_path}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        )
    manifest_date = str(manifest.get("requested_signal_date", ""))
    if manifest_date != str(expected_requested_signal_date):
        raise UniverseSnapshotError(
            "canonical snapshot requested date does not match its location: "
            f"expected={expected_requested_signal_date}, manifest={manifest_date}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        )
    try:
        stages = _load_canonical_stages(directory, manifest)
    except UniverseSnapshotError:
        raise
    except (OSError, UnicodeDecodeError, ParserError, ValueError, TypeError) as exc:
        raise UniverseSnapshotError(
            f"canonical snapshot cannot be loaded: {directory}: {exc}",
            status="CORRUPT_CANONICAL",
            canonical_manifest_path=manifest_path,
        ) from exc
    return manifest, stages


def _require_exact_stage_identities(
    expected: Mapping[str, StageSnapshot],
    actual: Mapping[str, StageSnapshot],
    manifest_path: Path,
    status: str,
) -> None:
    if set(expected) != set(actual):
        raise UniverseSnapshotError(
            f"snapshot stage set changed while publishing: expected={sorted(expected)}, actual={sorted(actual)}",
            status=status,
            canonical_manifest_path=manifest_path,
        )
    for stage in expected:
        expected_identity = stage_identity(expected[stage])
        actual_identity = stage_identity(actual[stage])
        if _full_identity_tuple(expected_identity) != _full_identity_tuple(actual_identity):
            raise UniverseSnapshotError(
                f"snapshot stage identity changed while publishing: stage={stage}",
                status=status,
                canonical_manifest_path=manifest_path,
            )


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


def _full_identity_tuple(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(identity.get("stage", "")),
        tuple(identity.get("key_columns") or ()),
        tuple(identity.get("row_columns") or ()),
        *_identity_tuple(identity),
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


def canonical_scalar_text(
    value: Any,
    column: str,
    *,
    stage: str = "",
    code: str = "",
    nan_is_missing: bool = True,
) -> str:
    """Encode one audit scalar into the only canonical snapshot representation."""
    if column == "code":
        return normalize_code(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number):
            if nan_is_missing:
                return CANONICAL_MISSING_TEXT
            _raise_non_finite_canonical_value(value, column, stage, code)
    if _is_missing(value):
        return CANONICAL_MISSING_TEXT
    if column.endswith("_date") or column in {"date", "signal_date", "trade_date"}:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    if isinstance(value, (bool,)) or value.__class__.__name__ == "bool_":
        return "true" if bool(value) else "false"
    if column.endswith("_bool"):
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0", "yes", "y", "t"}:
            return "true"
        if text in {"false", "0", "0.0", "no", "n", "f"}:
            return "false"
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            _raise_non_finite_canonical_value(value, column, stage, code)
        if number == 0.0:
            return "0"
        return format(number, f".{CANONICAL_FLOAT_SIGNIFICANT_DIGITS}g")
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text if text else CANONICAL_MISSING_TEXT


def _raise_non_finite_canonical_value(
    value: Any,
    column: str,
    stage: str,
    code: str,
) -> None:
    raise UniverseAuditError(
        "non-finite canonical numeric value: "
        f"stage={stage or '<unspecified>'}, code={code or '<unknown>'}, "
        f"field={column}, value={value!r}"
    )


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _temporary_file_path(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"


def _fsync_parent(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _remove_tree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        pass


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

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


TRACKED_PACKAGES = ("pandas", "numpy", "requests", "curl_cffi", "akshare", "openpyxl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def runtime_provenance() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "source_tree_sha256": source_tree_sha256(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": json.dumps(versions, sort_keys=True),
    }


def source_tree_sha256() -> str:
    paths = [
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "configs").rglob("*.json")),
        *sorted((PROJECT_ROOT / "configs").rglob("*.yaml")),
        *sorted((PROJECT_ROOT / "configs" / "models").rglob("*.csv")),
        *sorted((PROJECT_ROOT / "reports" / "manual_models").rglob("*.json")),
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pyproject.toml",
    ]
    hasher = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool | str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return bool(result.stdout.strip())

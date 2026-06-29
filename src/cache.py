from __future__ import annotations

import os
import threading
from pathlib import Path

import pandas as pd

from .code_utils import normalize_stock_code


class FrameCache:
    def __init__(self, root: Path, suffix: str):
        self.root = Path(root)
        self.suffix = suffix
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        safe_key = str(key).replace("/", "_").replace("\\", "_")
        return self.root / f"{safe_key}_{self.suffix}.pkl"

    def read(self, key: str) -> pd.DataFrame | None:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None

    def write(self, key: str, frame: pd.DataFrame) -> None:
        path = self.path(key)
        tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp{path.suffix}")
        _pickle_safe_frame(frame).to_pickle(tmp_path)
        tmp_path.replace(path)


class StockFrameCache(FrameCache):
    def path(self, key: str) -> Path:
        code = normalize_stock_code(key)
        return self.root / f"{code}_{self.suffix}.pkl"


def _pickle_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        array_type = type(result[column].array).__name__.lower()
        if pd.api.types.is_string_dtype(result[column].dtype) or "arrow" in array_type:
            result[column] = pd.Series(result[column].tolist(), index=result.index, dtype=object)
    result.columns = pd.Index(result.columns.tolist(), dtype=object)
    return result

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

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


class Baostock5mCache(StockFrameCache):
    """BaoStock 5m 历史缓存 (data/cache/baostock_5m/, 不覆盖 minute_5m/)。

    每个缓存文件可确定: code / source / date range / adjustment /
    fetch timestamp (独立 meta json, 不写入 pkl frame 本体)。
    原子写: tmp write -> 读回验证 -> atomic replace。
    """

    def __init__(self, root: Path, suffix: str = "5min"):
        super().__init__(root, suffix)
        self.validator: Any = None  # set externally: validate_normalized_5m_frame

    def meta_path(self, key: str) -> Path:
        code = normalize_stock_code(key)
        return self.root / f"{code}_{self.suffix}.meta.json"

    def read_meta(self, key: str) -> dict | None:
        path = self.meta_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def write(
        self,
        key: str,
        frame: pd.DataFrame,
        meta: dict | None = None,
    ) -> None:
        path = self.path(key)
        tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp{path.suffix}")
        _pickle_safe_frame(frame).to_pickle(tmp_path)
        if self.validator is not None:
            read_back = pd.read_pickle(tmp_path)
            self.validator(read_back)  # fail closed on contract violation
        tmp_path.replace(path)
        if meta is not None:
            meta_path = self.meta_path(key)
            tmp_meta = meta_path.with_name(
                f"{meta_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp_meta.write_text(
                json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1),
                encoding="utf-8",
            )
            tmp_meta.replace(meta_path)

    def build_meta(
        self,
        key: str,
        frame: pd.DataFrame,
        source: str,
        adjustment: str,
        interval: str,
        fetch_timestamp: str,
    ) -> dict:
        """确定性 provenance meta (fetch timestamp 除外)。"""
        frame = frame.copy()
        dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
        return {
            "code": normalize_stock_code(key),
            "source": source,
            "adjustment": adjustment,
            "interval": interval,
            "date_min": str(dates.min()) if not dates.empty else "",
            "date_max": str(dates.max()) if not dates.empty else "",
            "row_count": int(len(frame)),
            "fetch_timestamp": fetch_timestamp,
        }


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pickle_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        array_type = type(result[column].array).__name__.lower()
        if pd.api.types.is_string_dtype(result[column].dtype) or "arrow" in array_type:
            result[column] = pd.Series(result[column].tolist(), index=result.index, dtype=object)
    result.columns = pd.Index(result.columns.tolist(), dtype=object)
    return result

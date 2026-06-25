from __future__ import annotations

import re


_CODE_PATTERN = re.compile(r"^\d{1,6}$")

SH_MAIN_BOARD_PREFIXES = ("600", "601", "603", "605")
SZ_MAIN_BOARD_PREFIXES = ("000", "001", "002", "003")


def normalize_stock_code(code: str) -> str:
    text = str(code).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if not _CODE_PATTERN.fullmatch(text):
        raise ValueError(f"invalid stock code: {code}")
    return text.zfill(6)


def detect_market(code: str) -> str:
    normalized = normalize_stock_code(code)
    if normalized.startswith("6"):
        return "SH"
    if normalized.startswith(("0", "2", "3")):
        return "SZ"
    if normalized.startswith(("4", "8", "9")):
        return "BJ"
    return "UNKNOWN"


def to_market_symbol(code: str) -> str:
    normalized = normalize_stock_code(code)
    market = detect_market(normalized)
    if market == "SH":
        return f"sh{normalized}"
    if market == "SZ":
        return f"sz{normalized}"
    if market == "BJ":
        return f"bj{normalized}"
    raise ValueError(f"unknown market for code: {code}")


def to_eastmoney_secid(code: str) -> str:
    normalized = normalize_stock_code(code)
    market = detect_market(normalized)
    market_id = "1" if market == "SH" else "0"
    return f"{market_id}.{normalized}"


def is_main_board_code(code: str) -> bool:
    normalized = normalize_stock_code(code)
    return normalized.startswith(SH_MAIN_BOARD_PREFIXES + SZ_MAIN_BOARD_PREFIXES)


def is_excluded_name(name: str) -> bool:
    text = str(name or "").upper()
    return "ST" in text or "退" in text

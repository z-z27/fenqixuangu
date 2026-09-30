"""Phase 0.5 的两道验证闸门 (Gate A / Gate B)。

    CORRECTED_DATASET_GENERATION_GATE  —— 能否安全地生成 corrected historical dataset
    LEGACY_PARITY_GATE                 —— 冻结的 legacy 结果是否仍能被精确重放

两道闸门**严格独立**, 不得混用; ``Gate A PASS / Gate B FAIL`` 是允许的合法状态。

用法::

    python tools/run_calendar_semantic_gate_v001.py                    # 离线部分
    python tools/run_calendar_semantic_gate_v001.py --check-coverage   # 追加权威日历覆盖核验 (触网)

不训练模型、不生成 corrected 数据集、不覆盖任何 legacy artifact。
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: V4C 研究的样本区间 (reports/research/v4c_*_20260506_20260731)。
DEFAULT_REBUILD_START = "2026-05-06"
DEFAULT_REBUILD_END = "2026-07-31"

#: Gate A 的离线确定性核心。任何一项失败 = 闸门关闭。
GATE_A_OFFLINE_MODULES = (
    "tests.test_calendar_semantic_gate",
    "tests.test_trading_calendar",
    "tests.test_history_universe_integrity",
    "tests.test_listing_history_exclusion",
    "tests.test_signal_date_coverage",
    "tests.test_baostock_5m_source",
)

#: 冻结 V4C 重放模块 -> 对应的 parity 测试 (pytest)。
FROZEN_V4C_PARITY = {
    "src/v004c_v4a_architecture_transfer.py": "tests/test_v004c_original_v4a_direct_transfer.py",
    "src/v004c_pair_capped7_july_forward.py": "tests/test_v004c_pair_capped7_july_forward.py",
    "src/v004c_regime_aware_stage1_august_oot.py": "tests/test_v004c_regime_aware_stage1_august_oot.py",
}

#: 允许调用 legacy calendar semantics 的生产模块 —— 只有这三个冻结重放模块。
#: ``tests/`` 与 ``tools/`` 下引用它属于**对照/钉差异**用途, 不是生产路径, 因此
#: 隔离不变量只对 ``src/`` 生效: **corrected 生产代码永远不得触达 legacy**。
LEGACY_CALLERS_ALLOWED = frozenset(FROZEN_V4C_PARITY)

LEGACY_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+[\w.]*legacy_calendar_semantics\s+import|import\s+[\w.]*legacy_calendar_semantics)",
    re.MULTILINE,
)
PY_SCAN_ROOTS = ("src", "tools", "tests")


def _run_unittest_modules(modules: tuple[str, ...]) -> dict[str, object]:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(list(modules))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return {
        "modules": list(modules),
        "ran": int(result.testsRun),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "ok": bool(result.wasSuccessful()),
        "detail": [
            {"test": str(test), "traceback": text}
            for test, text in [*result.failures, *result.errors]
        ],
    }


def _run_pytest_module(path: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    summary = ""
    for line in reversed(output.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    return {
        "module": path,
        "returncode": int(completed.returncode),
        "summary": summary,
        "ok": completed.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-25:]),
    }


def _legacy_importers() -> list[str]:
    """静态扫描: 谁 import 了 legacy calendar semantics。"""
    importers = []
    for scan_root in PY_SCAN_ROOTS:
        for path in sorted((ROOT / scan_root).rglob("*.py")):
            if path.name == "legacy_calendar_semantics.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if LEGACY_IMPORT_PATTERN.search(text):
                importers.append(path.relative_to(ROOT).as_posix())
    return importers


def _check_legacy_isolation() -> dict[str, object]:
    importers = _legacy_importers()
    unexpected = [
        name for name in importers
        if name.startswith("src/") and name not in LEGACY_CALLERS_ALLOWED
    ]
    return {
        "importers": importers,
        "production_importers": [n for n in importers if n.startswith("src/")],
        "unexpected_importers": unexpected,
        "invariant": "corrected 生产代码 (src/) 永远不得 import legacy_calendar_semantics",
        "ok": not unexpected,
    }


def _check_coverage(start: str, end: str) -> dict[str, object]:
    """集成核验 (触网): 权威日历是否覆盖重建区间 + 未来 horizon。"""
    from src.trading_calendar import TradingCalendar, TradingCalendarError
    from src.config import get_data_config

    config = get_data_config()
    try:
        calendar = TradingCalendar(config)
        sessions = calendar.trading_days(start, end)
        provenance = calendar.provenance()
        coverage = calendar.coverage
        return {
            "ok": bool(sessions) and coverage is not None,
            "start": start,
            "end": end,
            "session_count": len(sessions),
            "first_session": sessions[0] if sessions else None,
            "last_session": sessions[-1] if sessions else None,
            "coverage": list(coverage) if coverage else None,
            "provenance": provenance,
            "error": None,
        }
    except TradingCalendarError as exc:
        return {
            "ok": False, "start": start, "end": end,
            "session_count": 0, "coverage": None, "provenance": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-coverage", action="store_true",
                        help="追加权威日历覆盖核验 (会触发真实取数)")
    parser.add_argument("--start", default=DEFAULT_REBUILD_START)
    parser.add_argument("--end", default=DEFAULT_REBUILD_END)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    report: dict[str, object] = {"gate_a": {}, "gate_b": {}}

    print("=" * 72)
    print("GATE A —— CORRECTED_DATASET_GENERATION_GATE")
    print("=" * 72)
    offline = _run_unittest_modules(GATE_A_OFFLINE_MODULES)
    report["gate_a"]["offline"] = offline
    print(f"  离线确定性核心: ran={offline['ran']} failures={offline['failures']} "
          f"errors={offline['errors']} skipped={offline['skipped']} -> "
          f"{'OK' if offline['ok'] else 'FAILED'}")
    for item in offline["detail"]:
        print(f"    - {item['test']}")

    coverage_result = None
    if args.check_coverage:
        coverage_result = _check_coverage(args.start, args.end)
        report["gate_a"]["coverage"] = coverage_result
        print(f"  权威日历覆盖 {args.start}..{args.end}: "
              f"sessions={coverage_result['session_count']} "
              f"coverage={coverage_result['coverage']} -> "
              f"{'OK' if coverage_result['ok'] else 'FAILED'}")
        if coverage_result.get("error"):
            print(f"    error: {coverage_result['error']}")
    else:
        print("  (跳过权威日历覆盖核验; 加 --check-coverage 触发, 会触网)")

    gate_a_ok = bool(offline["ok"]) and (
        coverage_result is None or bool(coverage_result["ok"])
    )
    report["gate_a"]["pass"] = gate_a_ok

    print()
    print("=" * 72)
    print("GATE B —— LEGACY_PARITY_GATE")
    print("=" * 72)
    isolation = _check_legacy_isolation()
    report["gate_b"]["isolation"] = isolation
    print(f"  legacy 适配器隔离: importers={isolation['importers']} -> "
          f"{'OK' if isolation['ok'] else 'FAILED'}")
    for name in isolation["unexpected_importers"]:
        print(f"    unexpected importer: {name}")

    parity = []
    for module, test_path in FROZEN_V4C_PARITY.items():
        result = _run_pytest_module(test_path)
        result["frozen_module"] = module
        parity.append(result)
        print(f"  {test_path}: {result['summary']} -> "
              f"{'OK' if result['ok'] else 'FAILED'}")
    report["gate_b"]["parity"] = parity
    gate_b_ok = bool(isolation["ok"]) and all(item["ok"] for item in parity)
    report["gate_b"]["pass"] = gate_b_ok

    print()
    print("=" * 72)
    print(f"CORRECTED_DATASET_GENERATION_GATE = {'PASS' if gate_a_ok else 'FAIL'}")
    print(f"LEGACY_PARITY_GATE = {'PASS' if gate_b_ok else 'FAIL'}")
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"report -> {args.json_out}")

    return 0 if gate_a_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

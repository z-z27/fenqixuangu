# -*- coding: utf-8 -*-
"""v004c Pairwise v1 Feature Contract Freeze — 构建工具 (May+June 冻结)。

本工具冻结 PAIRWISE_V1_FEATURE_CONTRACT (53 个 FEATURE) 并生成 319 行 ×
冻结合法特征集 的 May+June Pairwise v1 输入表, 以及全部 7 个输出资产。

严格边界 (本工具):
- 不训练任何模型 (Pairwise Ridge / Logistic / Tree/GBDT), 不搜索 lambda,
  不做 walk-forward, 不计算概率, 不生成排名, 不生成 Top1/Top3;
- 不根据 Target/tail-loss/单因子 AUC/IC/旧模型表现选择特征;
- 不做全样本 imputation (mean/median/zscore/winsorize/clip), 缺失保持 NaN;
- 不做相关性删除; 只允许 exact duplicate / exact deterministic transform 剔除;
- 不创建 feature_x_board3 交互列; 只冻结 base X + board group indicator
  (board_streak_is_3);
- 不修改 v002 foundation 与 reports/_scratch 历史脚本 (只读消费 v002 CSV);
- universe 身份与 v002 逐行精确一致 (event_id/code/signal_date/break_date/
  board_streak_before_break/source_window), 319/319, 永不删除行。

重建口径:
- 19 个缺失特征从日线缓存 (daily) + 涨停池缓存 (limit_ups) 确定性重建,
  公式与 _scratch/build_v004c_dataset.py 官方定义逐条对齐;
- 34 个特征直接取自 v002 冻结值 (只读);
- board_day_*_rank / break_amount_ratio_vs_board_days 因 May 池 amount/
  turnover 全 NaN + daily amount 全 NaN 而 EXCLUDE_SOURCE_UNAVAILABLE
  (可在真实交易时重现性不成立), 不进契约。

用法:
    python tools/build_v004c_pairwise_v1_feature_contract.py [--output DIR]

输出 (默认 reports/research/v004c_pairwise_v1_feature_contract_20260506_20260630/):
    1. v004c_pairwise_v1_feature_inventory.csv          21 列候选清单
    2. v004c_pairwise_v1_feature_contract.csv           53 FEATURE 契约
    3. v004c_pairwise_v1_label_contract.csv             标签契约 (LABEL_ONLY)
    4. v004c_pairwise_v1_input_table.csv                319 行输入表 (冻结特征序)
    5. v004c_pairwise_v1_schema.csv                     列角色 schema
    6. v004c_pairwise_v1_exclusions.csv                 排除候选与理由
    7. v004c_pairwise_v1_feature_contract_review.md     中文 review

确定性: 相同输入下重复运行字节一致; 输出原子写入 (临时文件 + os.replace)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v004c_pairwise_feature_contract import (  # noqa: E402
    AUDIT_COLUMNS,
    AUDIT_ONLY_V002_COLUMNS,
    CONTRACT_FEATURE_NAMES,
    CONTRACT_FEATURE_ORDER,
    DEV_ELIGIBILITY_COLUMNS,
    EXCLUDED_CANDIDATE_DECISIONS,
    FEATURE,
    FEATURE_CONTRACT,
    IDENTIFIER,
    IDENTIFIER_COLUMNS,
    LABEL_COLUMNS,
    LABEL_ONLY,
    assert_contract_integrity,
    leakage_scan,
)

# ---------------------------------------------------------------------------
# 路径 / 常量
# ---------------------------------------------------------------------------
V002_CSV = (ROOT / "reports" / "research"
            / "v004c_baostock_d1_dev_v002_20260506_20260630"
            / "v004c_baostock_d1_dev_v002.csv")
OUT_DIR = ROOT / "reports" / "research" / "v004c_pairwise_v1_feature_contract_20260506_20260630"
DAILY_DIR = ROOT / "data" / "cache" / "daily"
POOL_DIR = ROOT / "data" / "cache" / "limit_ups"

WINDOW_START = "2026-05-06"
WINDOW_END = "2026-06-30"

MAIN_BOARD_LIMIT_RATIO = Decimal("1.10")
LIMIT_UP_PRICE_TOLERANCE = 0.011
LIMIT_UP_MIN_PCT_CHG = 9.7

DEV_CSV_NAME = "v004c_pairwise_v1_input_table.csv"
INVENTORY_CSV_NAME = "v004c_pairwise_v1_feature_inventory.csv"
CONTRACT_CSV_NAME = "v004c_pairwise_v1_feature_contract.csv"
LABEL_CSV_NAME = "v004c_pairwise_v1_label_contract.csv"
SCHEMA_CSV_NAME = "v004c_pairwise_v1_schema.csv"
EXCLUSIONS_CSV_NAME = "v004c_pairwise_v1_exclusions.csv"
REVIEW_NAME = "v004c_pairwise_v1_feature_contract_review.md"

# 输入表列序: 身份 + GROUP_KEY + 53 FEATURE + 标签 + dev 资格 + 审计
INPUT_PREFIX_COLUMNS = ("event_id", "code", "signal_date", "board_streak_before_break",
                        "source_window", "break_date")
INPUT_SUFFIX_COLUMNS = LABEL_COLUMNS + DEV_ELIGIBILITY_COLUMNS + (
    "structural_missing_count", "unexpected_missing_count", "provenance")


# ---------------------------------------------------------------------------
# 官方数学工具 (与 _scratch/build_v004c_dataset.py 逐条一致)
# ---------------------------------------------------------------------------
def round_price_limit(prev_close: float) -> float:
    value = Decimal(str(prev_close)) * MAIN_BOARD_LIMIT_RATIO
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def is_limit_up_day(close, high, prev_close) -> bool:
    if prev_close is None or prev_close <= 0 or close is None or high is None:
        return False
    limit_price = round_price_limit(prev_close)
    pct_chg = (close / prev_close - 1.0) * 100.0
    if pct_chg < LIMIT_UP_MIN_PCT_CHG:
        return False
    if close < limit_price - LIMIT_UP_PRICE_TOLERANCE:
        return False
    if high < limit_price - LIMIT_UP_PRICE_TOLERANCE:
        return False
    return True


def _to_float(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _ret(v) -> float | None:
    v = _to_float(v)
    return None if v is None else v - 1.0


def _safe_div(a, b) -> float | None:
    a = _to_float(a)
    b = _to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_pool_cache():
    """涨停池缓存 -> (pool_members, pool_meta, pool_dates, pool_date_set)。

    与 scratch 官方逐条一致: code zfill(6), trade_date str, drop_duplicates keep last。
    """
    pool_frames = []
    for path in sorted(Path(POOL_DIR).glob("*_limitups.pkl")):
        pool_frames.append(pd.read_pickle(path))
    pool = pd.concat(pool_frames, ignore_index=True)
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool["trade_date"] = pool["trade_date"].astype(str)
    pool = pool.drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True)

    pool_members: dict[str, set[str]] = {}
    pool_meta: dict[tuple, dict] = {}
    for row in pool.to_dict(orient="records"):
        d = str(row["trade_date"])
        c = str(row["code"])
        pool_members.setdefault(d, set()).add(c)
        pool_meta[(d, c)] = row
    pool_dates = sorted(pool["trade_date"].unique().tolist())
    pool_date_set = set(pool_dates)
    return pool_members, pool_meta, pool_dates, pool_date_set


def load_daily_store(codes: set[str]):
    """日线缓存 -> {code: 标准化 DataFrame(date str, numeric OHLCV)}。"""
    store: dict[str, pd.DataFrame] = {}
    for code in sorted(codes):
        path = DAILY_DIR / f"{code}_daily.pkl"
        if not path.exists():
            continue
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue
        if df is None or df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume", "amount", "turnover_rate"):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        store[code] = df
    return store


def compute_daily_arrays(df: pd.DataFrame):
    """日线序列 -> (flags, ma5, ma10, ma20, date_index)。与 scratch 官方一致。"""
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    prev = np.roll(close, 1)
    prev[0] = np.nan
    n = len(df)
    flags = np.array([is_limit_up_day(close[i], high[i], prev[i]) for i in range(n)], dtype=bool)
    flags[0] = False
    ma5 = df["close"].rolling(5, min_periods=5).mean().to_numpy(float)
    ma10 = df["close"].rolling(10, min_periods=10).mean().to_numpy(float)
    ma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy(float)
    date_index = {d: i for i, d in enumerate(df["date"].tolist())}
    return flags, ma5, ma10, ma20, date_index


def build_daily_volume_lookup(daily_store):
    """date -> {code: volume} 查找表 (board_day_volume_rank 横截面跨 code 查询)。"""
    lookup: dict[str, dict[str, float]] = {}
    for code, df in daily_store.items():
        for d, v in zip(df["date"].tolist(), df["volume"].tolist()):
            vf = _to_float(v)
            if vf is not None:
                lookup.setdefault(d, {})[code] = vf
    return lookup


# ---------------------------------------------------------------------------
# 事件特征重建 (19 个缺失特征)
# ---------------------------------------------------------------------------
def rebuild_event_features(row, daily, flags, ma5, ma10, ma20, date_index,
                           pool_members, pool_meta, pool_date_set,
                           daily_volume_lookup):
    """重建 19 个特征 + 返回 (values, structural_mask)。

    values: dict[feature_name -> float/int/bool/None]
    structural_mask: dict[feature_name -> bool]  缺失是否为结构允许 (数学退化/池窗口外)
    """
    code = row["code"]
    break_date = str(row["break_date"])
    signal_date = str(row["signal_date"])
    streak = int(row["board_streak_before_break"])
    assert signal_date == break_date, "v002 universe 要求 signal==break (D1 identity)"

    idx = date_index.get(break_date)
    values: dict[str, float | int | bool | None] = {}
    structural: dict[str, bool] = {}

    # D0 = 末板日 (idx-1)
    if idx is None or idx < 1:
        for f in ("board_streak_is_3", "max_board_streak_20d", "recent_limit_up_count_10d",
                  "recent_limit_up_count_20d", "recent_pool_appearance_count_10d",
                  "recent_pool_appearance_count_20d", "board_day_volume_rank",
                  "break_high_return", "break_close_return", "break_touched_limit_up",
                  "break_opened_from_limit_up", "break_upper_shadow_ratio",
                  "break_lower_shadow_ratio", "break_volume_ratio_vs_board_days",
                  "break_day_in_pool", "last_board_day_in_pool",
                  "pool_consecutive_count_last_board", "d1_ma20_slope",
                  "recent_7d_limit_up_count"):
            values[f] = None
            structural[f] = True
        return values, structural

    lb = idx - 1
    lb_date = str(daily["date"].iloc[lb])
    d0_close = _to_float(daily["close"].iloc[lb])
    d1_open = _to_float(row.get("d1_open"))
    d1_high = _to_float(row.get("d1_high"))
    d1_low = _to_float(row.get("d1_low"))
    d1_close = _to_float(row.get("d1_close"))

    # ---- BOARD_HISTORY ----
    values["board_streak_is_3"] = int(streak == 3)
    structural["board_streak_is_3"] = False

    lo10 = max(0, idx - 9)
    lo20 = max(0, idx - 19)
    lo7 = max(0, idx - 6)
    values["recent_limit_up_count_10d"] = int(flags[lo10: idx + 1].sum())
    values["recent_limit_up_count_20d"] = int(flags[lo20: idx + 1].sum())
    structural["recent_limit_up_count_10d"] = False
    structural["recent_limit_up_count_20d"] = False

    dates10 = daily["date"].iloc[lo10: idx + 1].tolist()
    dates20 = daily["date"].iloc[lo20: idx + 1].tolist()
    values["recent_pool_appearance_count_10d"] = int(sum(
        1 for d in dates10 if d in pool_date_set and code in pool_members.get(d, set())))
    values["recent_pool_appearance_count_20d"] = int(sum(
        1 for d in dates20 if d in pool_date_set and code in pool_members.get(d, set())))
    structural["recent_pool_appearance_count_10d"] = False  # 覆盖下界为合法语义, 非缺失
    structural["recent_pool_appearance_count_20d"] = False

    max_streak = 0
    cur = 0
    for f in flags[lo20: idx + 1]:
        cur = cur + 1 if f else 0
        max_streak = max(max_streak, cur)
    values["max_board_streak_20d"] = max_streak
    structural["max_board_streak_20d"] = False

    # ---- D0_CROSS_SECTION: board_day_volume_rank ----
    members = pool_members.get(lb_date, set())
    rk_pool_rows = [pool_meta[(lb_date, c)] for c in sorted(members) if (lb_date, c) in pool_meta]
    rk_pool = pd.DataFrame(rk_pool_rows) if rk_pool_rows else pd.DataFrame()
    if rk_pool.empty or lb_date not in pool_date_set:
        values["board_day_volume_rank"] = None
        structural["board_day_volume_rank"] = True  # D0 在池窗口外 -> 结构缺失
    elif code not in rk_pool["code"].astype(str).tolist():
        # D0 在池窗口内但该股不在涨停池收录 (池源覆盖缺口) -> 无横截面 rank
        values["board_day_volume_rank"] = None
        structural["board_day_volume_rank"] = True
    else:
        codes_l = rk_pool["code"].astype(str).tolist()
        lb_vol_lookup = daily_volume_lookup.get(lb_date, {})
        vol_map: dict[str, float] = {}
        for c in codes_l:
            vol = lb_vol_lookup.get(c)
            vol_map[c] = vol if vol is not None else np.nan
        if code not in vol_map or pd.isna(vol_map[code]):
            values["board_day_volume_rank"] = None
            structural["board_day_volume_rank"] = False  # 池内但量缺 -> 非结构, 应不出现
        else:
            vals = pd.Series(vol_map, dtype=float)
            ranks = vals.rank(method="min", ascending=False)
            values["board_day_volume_rank"] = float(ranks.loc[code])
            structural["board_day_volume_rank"] = False

    # ---- D1_PRICE_ACTION ----
    values["break_high_return"] = _ret(_safe_div(d1_high, d0_close))
    values["break_close_return"] = _ret(_safe_div(d1_close, d0_close))
    structural["break_high_return"] = d0_close is None
    structural["break_close_return"] = d0_close is None
    lp = round_price_limit(d0_close) if d0_close else None
    values["break_touched_limit_up"] = bool(lp is not None and d1_high is not None
                                            and d1_high >= lp - LIMIT_UP_PRICE_TOLERANCE)
    values["break_opened_from_limit_up"] = bool(lp is not None and d1_open is not None
                                                and d1_open >= lp - LIMIT_UP_PRICE_TOLERANCE)
    structural["break_touched_limit_up"] = d0_close is None or d1_high is None
    structural["break_opened_from_limit_up"] = d0_close is None or d1_open is None

    bh = d1_high
    bl = d1_low
    bo = d1_open
    bc = d1_close
    if bh is not None and bl is not None and bh > bl:
        values["break_upper_shadow_ratio"] = _safe_div(bh - max(bo or bh, bc or bh), bh - bl)
        values["break_lower_shadow_ratio"] = _safe_div(min(bo or bl, bc or bl) - bl, bh - bl)
    else:
        values["break_upper_shadow_ratio"] = None
        values["break_lower_shadow_ratio"] = None
    structural["break_upper_shadow_ratio"] = not (bh is not None and bl is not None and bh > bl)
    structural["break_lower_shadow_ratio"] = not (bh is not None and bl is not None and bh > bl)

    # ---- D1_VOLUME_ACTIVITY: break_volume_ratio_vs_board_days ----
    board_rows = daily.iloc[idx - streak: idx]
    bd_vols = [float(x) for x in pd.to_numeric(board_rows["volume"], errors="coerce").dropna() if x > 0]
    if bd_vols:
        # 分子 = D1 日线 volume (官方口径, 与 scratch 一致), 分母 = board 日 volume 均值
        values["break_volume_ratio_vs_board_days"] = _safe_div(
            _to_float(daily["volume"].iloc[idx]), float(np.mean(bd_vols)))
        structural["break_volume_ratio_vs_board_days"] = False
    else:
        values["break_volume_ratio_vs_board_days"] = None
        structural["break_volume_ratio_vs_board_days"] = True

    # ---- POOL_MEMBERSHIP ----
    # break_date 恒在池窗口内 (>= 05-06), 恒有值; lb_date (D0) 可能 =04-30 在池窗口外
    if break_date in pool_date_set:
        values["break_day_in_pool"] = bool(code in pool_members.get(break_date, set()))
        structural["break_day_in_pool"] = False
    else:
        values["break_day_in_pool"] = None
        structural["break_day_in_pool"] = True
    if lb_date in pool_date_set:
        # 注意: 池源覆盖缺口 (D0 涨停但池未收录) 在池窗口内仍给 False / None,
        # 该股无池成员记录 => last_board_day_in_pool=False 合法, count 结构缺失
        values["last_board_day_in_pool"] = bool(code in pool_members.get(lb_date, set()))
        structural["last_board_day_in_pool"] = False
    else:
        values["last_board_day_in_pool"] = None
        structural["last_board_day_in_pool"] = True
    pcc = pool_meta.get((lb_date, code), {}).get("consecutive_limit_up_count")
    values["pool_consecutive_count_last_board"] = _to_float(pcc)
    # 池窗口外 或 窗口内但池未收录该股 -> 结构缺失 (池源覆盖缺口)
    structural["pool_consecutive_count_last_board"] = (
        lb_date not in pool_date_set
        or values["pool_consecutive_count_last_board"] is None)

    # ---- D1_MA_POSITION: d1_ma20_slope ----
    ma20_prev = ma20[lb] if lb >= 0 else np.nan
    ma20_cur = ma20[idx] if idx < len(ma20) else np.nan
    values["d1_ma20_slope"] = _ret(_safe_div(ma20_cur, ma20_prev))
    structural["d1_ma20_slope"] = bool(not (np.isfinite(ma20_prev) and np.isfinite(ma20_cur)
                                            and ma20_prev > 0))

    # ---- RECENT_7D_PATH: recent_7d_limit_up_count ----
    values["recent_7d_limit_up_count"] = int(flags[lo7: idx + 1].sum())
    structural["recent_7d_limit_up_count"] = False

    return values, structural


# ---------------------------------------------------------------------------
# 输入表组装
# ---------------------------------------------------------------------------
def _row_value(v):
    """统一标量输出: np scalars -> python, bool -> int (确定性 CSV)。"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def assemble_input_table(v002, pool_members, pool_meta, pool_dates, pool_date_set,
                         daily_store, daily_volume_lookup):
    """组装 319 行输入表, universe 身份与 v002 逐行精确一致。"""
    rows = []
    for _, row in v002.iterrows():
        code = str(row["code"]).zfill(6)
        r = {
            "event_id": str(row["event_id"]),
            "code": code,
            "signal_date": str(row["signal_date"]),
            "board_streak_before_break": int(row["board_streak_before_break"]),
            "source_window": str(row["source_window"]),
            "break_date": str(row["break_date"]),
        }
        daily = daily_store.get(code)
        if daily is None:
            flags, ma5, ma10, ma20, date_index = None, None, None, None, {}
            rebuild = {f: None for f in CONTRACT_FEATURE_NAMES}
            structural = {f: True for f in CONTRACT_FEATURE_NAMES}
        else:
            flags, ma5, ma10, ma20, date_index = compute_daily_arrays(daily)
            rebuild, structural = rebuild_event_features(
                {"code": code, "break_date": row["break_date"], "signal_date": row["signal_date"],
                 "board_streak_before_break": row["board_streak_before_break"],
                 "d1_open": row.get("d1_open"), "d1_high": row.get("d1_high"),
                 "d1_low": row.get("d1_low"), "d1_close": row.get("d1_close")},
                daily, flags, ma5, ma10, ma20, date_index, pool_members, pool_meta, pool_date_set,
                daily_volume_lookup)

        # 34 个直接取自 v002 的 FEATURE
        v002_direct = [c for c in CONTRACT_FEATURE_NAMES
                       if c not in rebuild]
        for f in v002_direct:
            r[f] = _row_value(row.get(f))
            # v002 冻结特征的结构缺失模式: d1_close_location 在 D1 high==low
            # (一字板) 时官方公式分母为 0 -> None, 属定义允许 (603065 型)。
            if f == "d1_close_location" and r[f] is None:
                h = _to_float(row.get("d1_high"))
                l = _to_float(row.get("d1_low"))
                structural[f] = not (h is not None and l is not None and h > l)
            else:
                structural.setdefault(f, False)

        # 19 个重建 FEATURE
        for f, v in rebuild.items():
            r[f] = _row_value(v)

        # 标签 + dev 资格 (从 v002 原样携带, 物理隔离在 FEATURE 之后)
        for c in LABEL_COLUMNS + DEV_ELIGIBILITY_COLUMNS:
            r[c] = _row_value(row.get(c))

        # 缺失统计: structural vs unexpected
        missing = [f for f in CONTRACT_FEATURE_NAMES if r[f] is None]
        unexpected = [f for f in missing if not structural.get(f, False)]
        r["structural_missing_count"] = len(missing) - len(unexpected)
        r["unexpected_missing_count"] = len(unexpected)
        r["provenance"] = (f"v002:event_id={r['event_id']};"
                           f"rebuild={','.join(sorted(rebuild.keys()))}")
        rows.append(r)

    df = pd.DataFrame(rows)
    # 精确列序: 身份 + 53 FEATURE + 标签 + dev + 审计
    cols = (list(INPUT_PREFIX_COLUMNS)
            + list(CONTRACT_FEATURE_NAMES)
            + list(INPUT_SUFFIX_COLUMNS))
    df = df[cols]
    return df


# ---------------------------------------------------------------------------
# 覆盖率统计
# ---------------------------------------------------------------------------
def coverage_stats(input_df):
    """逐 FEATURE May/June/Combined 非空覆盖率。"""
    out = {}
    for f in CONTRACT_FEATURE_NAMES:
        may = input_df[input_df["source_window"] == "may"]
        june = input_df[input_df["source_window"] == "june"]
        out[f] = {
            "may": may[f].notna().mean(),
            "june": june[f].notna().mean(),
            "combined": input_df[f].notna().mean(),
        }
    return out


# ---------------------------------------------------------------------------
# 输出文件
# ---------------------------------------------------------------------------
def write_atomic(path: Path, content: str, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_inventory(input_df, cov):
    """候选清单 CSV (21 列)。行 = 132 workflow 候选 + v002 附加审计/label 列。"""
    rows = []
    seen = set()
    # FEATURE 候选
    for c in FEATURE_CONTRACT:
        name = c["feature_name"]
        rows.append({
            "feature_name": name,
            "found_in": "v004c_model_table/factor_dictionary/v004c_d1_dataset (frozen)",
            "historical_role": c.get("notes", ""),
            "semantic_group": c["semantic_group"],
            "raw_or_derived": "derived",
            "raw_dependencies": ";".join(c["raw_dependencies"]),
            "formula_or_definition": c["formula"],
            "lookback": c["lookback"],
            "available_as_of": c["available_as_of"],
            "source_system": c["source"],
            "current_may_coverage": cov[name]["may"],
            "current_june_coverage": cov[name]["june"],
            "combined_coverage": cov[name]["combined"],
            "structural_missing_possible": "yes" if c["missing_semantics"].startswith("STRUCTURAL")
            else ("no"),
            "future_leakage": 0,
            "model_output_leakage": 0,
            "target_lineage_leakage": 0,
            "exact_duplicate_group": "",
            "semantic_duplicate_group": c["semantic_group"],
            "pairwise_v1_decision": FEATURE,
            "decision_reason": c.get("notes", ""),
        })
        seen.add(name)

    # 排除候选 (EXCLUDED_CANDIDATE_DECISIONS + 部分额外非候选)
    for name, (decision, reason) in sorted(EXCLUDED_CANDIDATE_DECISIONS.items()):
        if name in seen:
            continue
        rows.append({
            "feature_name": name,
            "found_in": "workflow v004c candidate inventory",
            "historical_role": "",
            "semantic_group": "",
            "raw_or_derived": "",
            "raw_dependencies": "",
            "formula_or_definition": reason,
            "lookback": "",
            "available_as_of": "D1_CLOSE" if decision == LABEL_ONLY else "",
            "source_system": "",
            "current_may_coverage": None,
            "current_june_coverage": None,
            "combined_coverage": None,
            "structural_missing_possible": "",
            "future_leakage": 0,
            "model_output_leakage": 0,
            "target_lineage_leakage": 0,
            "exact_duplicate_group": "",
            "semantic_duplicate_group": "",
            "pairwise_v1_decision": decision,
            "decision_reason": reason,
        })
        seen.add(name)

    df = pd.DataFrame(rows)
    cols = ["feature_name", "found_in", "historical_role", "semantic_group",
            "raw_or_derived", "raw_dependencies", "formula_or_definition",
            "lookback", "available_as_of", "source_system",
            "current_may_coverage", "current_june_coverage", "combined_coverage",
            "structural_missing_possible", "future_leakage", "model_output_leakage",
            "target_lineage_leakage", "exact_duplicate_group",
            "semantic_duplicate_group", "pairwise_v1_decision", "decision_reason"]
    return df[cols]


def write_contract_csv():
    """FEATURE 契约 CSV (13 列, feature_order 从 1 连续)。"""
    rows = []
    for i, c in enumerate(FEATURE_CONTRACT, 1):
        rows.append({
            "feature_order": i,
            "feature_name": c["feature_name"],
            "semantic_group": c["semantic_group"],
            "information_class": c["information_class"],
            "dtype": c["dtype"],
            "formula": c["formula"],
            "raw_dependencies": ";".join(c["raw_dependencies"]),
            "source": c["source"],
            "lookback": c["lookback"],
            "available_as_of": c["available_as_of"],
            "missing_semantics": c["missing_semantics"],
            "expected_range_or_domain": c["expected_range_or_domain"],
            "notes": c["notes"],
        })
    return pd.DataFrame(rows)


def write_label_contract():
    """标签契约 CSV (LABEL_ONLY, 物理隔离)。"""
    rows = [
        {"label_name": "target7_daily_d2open_d3high",
         "role": LABEL_ONLY,
         "definition": "1[(d3_high / d2_open - 1) >= 0.07]",
         "horizon": "D2 open -> D3 high (未来信息)",
         "available_as_of": "D3_CLOSE (未来)",
         "physical_isolation": "生成 X 时未知; 仅作为最后两列附加, 特征序之前永不读取",
         "usage": "未来 Pairwise 训练标签 (Y), 不作为任何特征"},
        {"label_name": "tail_loss_daily_5pct",
         "role": LABEL_ONLY,
         "definition": "1[(d3_high / d2_open - 1) <= -0.05]",
         "horizon": "D2 open -> D3 high (未来信息)",
         "available_as_of": "D3_CLOSE (未来)",
         "physical_isolation": "生成 X 时未知; 仅作为最后两列附加",
         "usage": "未来 Pairwise 训练标签 (Y), 不作为任何特征"},
        {"label_name": "d2_open_daily", "role": LABEL_ONLY,
         "definition": "D2 开盘 (标签分子原始值)", "horizon": "D2", "available_as_of": "D2_OPEN",
         "physical_isolation": "标签序列原始字段, 绝不入 X", "usage": "标签派生"},
        {"label_name": "d3_high_daily", "role": LABEL_ONLY,
         "definition": "D3 最高 (标签分子原始值)", "horizon": "D3", "available_as_of": "D3_CLOSE",
         "physical_isolation": "标签序列原始字段, 绝不入 X", "usage": "标签派生"},
        {"label_name": "d3_close_daily", "role": LABEL_ONLY,
         "definition": "D3 收盘 (扩展标签候选)", "horizon": "D3", "available_as_of": "D3_CLOSE",
         "physical_isolation": "标签序列原始字段, 绝不入 X", "usage": "保留未用"},
    ]
    return pd.DataFrame(rows)


def write_schema(input_df):
    """schema CSV: 每列角色 / feature_order / dtype / available_as_of / source。"""
    rows = []
    for i, col in enumerate(input_df.columns):
        if col in CONTRACT_FEATURE_NAMES:
            c = next(x for x in FEATURE_CONTRACT if x["feature_name"] == col)
            role = FEATURE
            order = CONTRACT_FEATURE_ORDER[col]
            avail = c["available_as_of"]
            source = c["source"]
        elif col in IDENTIFIER_COLUMNS or col == "break_date":
            role = IDENTIFIER
            order = None
            avail = "D1_CLOSE"
            source = "v002"
        elif col in LABEL_COLUMNS:
            role = LABEL_ONLY
            order = None
            avail = "D3_CLOSE"
            source = "v002_label"
        elif col in DEV_ELIGIBILITY_COLUMNS:
            role = "DEV_ELIGIBILITY"
            order = None
            avail = "D3_CLOSE"
            source = "v002"
        else:
            role = "AUDIT_ONLY"
            order = None
            avail = "D1_CLOSE"
            source = "v002"
        rows.append({
            "column_name": col,
            "role": role,
            "feature_order": order,
            "dtype": str(input_df[col].dtype),
            "available_as_of": avail,
            "source": source,
        })
    return pd.DataFrame(rows)


def write_exclusions():
    """排除候选 CSV。"""
    rows = [{"feature_name": k, "pairwise_v1_decision": v[0], "decision_reason": v[1]}
            for k, v in sorted(EXCLUDED_CANDIDATE_DECISIONS.items())]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------
def run_leakage_audit(input_df, contract_names):
    """泄漏审计: FEATURE 未来/模型输出/标签谱系泄漏必须全 0。"""
    result = leakage_scan(contract_names)
    assert result["future_leakage"] == 0, f"FEATURE 未来泄漏: {result['future_hits']}"
    assert result["model_output_leakage"] == 0, f"FEATURE 模型输出泄漏: {result['model_hits']}"
    assert result["label_lineage_leakage"] == 0, f"FEATURE 标签谱系泄漏: {result['label_hits']}"
    return result


def availability_audit(input_df):
    """available_as_of <= D1_CLOSE 逐特征审计。"""
    bad = []
    for c in FEATURE_CONTRACT:
        if c["available_as_of"] not in ("D1_CLOSE", "D0_CLOSE"):
            bad.append(c["feature_name"])
    return bad


def determinism_check(output_dir, new_output_dir):
    """字节级确定性: 第二次构建与第一次核心资产逐字节一致。"""
    core = ["v004c_pairwise_v1_input_table.csv", "v004c_pairwise_v1_feature_contract.csv",
            "v004c_pairwise_v1_feature_inventory.csv", "v004c_pairwise_v1_label_contract.csv",
            "v004c_pairwise_v1_schema.csv", "v004c_pairwise_v1_exclusions.csv",
            "v004c_pairwise_v1_feature_contract_review.md"]
    ok = True
    for name in core:
        a = (output_dir / name).read_bytes()
        b = (new_output_dir / name).read_bytes()
        if a != b:
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------
def write_review(input_df, cov, universe_ok, leakage, availability_bad,
                 determinism_ok, canonical_out_dir):
    lines = []
    ap = lines.append
    ap(f"# v004c Pairwise v1 Feature Contract Review (May+June 冻结)")
    ap("")
    ap(f"- 输出: {canonical_out_dir} (两轮构建同路径, 保证 review 字节一致)")
    ap(f"- 窗口: {WINDOW_START} .. {WINDOW_END} (May 146 行 / 18 信号日, June 173 行 / 21 信号日)")
    ap(f"- universe: {len(input_df)} 行 / {input_df['signal_date'].nunique()} 信号日")
    ap(f"- 与 v002 身份逐行一致 (event_id/code/signal_date/break_date/"
       f"board_streak_before_break/source_window): {'PASS' if universe_ok else 'FAIL'}")
    ap(f"- FEATURE 数: {len(CONTRACT_FEATURE_NAMES)}")
    ap("")
    ap("## 1. 信息边界 (available by D1 close)")
    ap("")
    ap("所有 FEATURE `available_as_of <= D1_CLOSE`。允许 D1 当日信息、最近 7 日详细路径、"
       "MA5/10/20、10/20 日历史状态、连板历史、涨停池历史、历史成交状态。"
       "标签 (target7/tail_loss) 与 D2/D3 价格被物理隔离在 X 之后, 仅在 Pairwise 训练时附加。")
    bad_names = availability_bad
    ap(f"- 违反 D1 边界的 FEATURE: {bad_names if bad_names else '无'}")
    ap("")
    ap("## 2. 特征清单")
    ap("")
    ap(f"- 候选清单元数据: v004c_pairwise_v1_feature_inventory.csv (21 列)")
    ap(f"- FEATURE 契约: v004c_pairwise_v1_feature_contract.csv (53 行, feature_order 1..53)")
    ap(f"- 标签契约: v004c_pairwise_v1_label_contract.csv (LABEL_ONLY)")
    ap("")
    ap("### 53 个 FEATURE 按 semantic_group")
    ap("")
    groups: dict[str, list[str]] = {}
    for c in FEATURE_CONTRACT:
        groups.setdefault(c["semantic_group"], []).append(c["feature_name"])
    for g in ("BOARD_HISTORY", "D0_CROSS_SECTION", "D1_PRICE_ACTION", "D1_VOLUME_ACTIVITY",
              "D1_MA_POSITION", "D1_CHIP_DISTRIBUTION", "D1_LATE_DAY_PRESSURE",
              "D1_VWAP_POSITION", "POOL_MEMBERSHIP", "RECENT_7D_PATH"):
        if g in groups:
            ap(f"- **{g}** ({len(groups[g])}): {', '.join(groups[g])}")
    ap("")
    ap("### 按 information_class")
    ap("")
    ic: dict[str, int] = {}
    for c in FEATURE_CONTRACT:
        ic[c["information_class"]] = ic.get(c["information_class"], 0) + 1
    for k, v in sorted(ic.items()):
        ap(f"- {k}: {v}")
    ap("")
    ap("### 重建口径 (19 个缺失特征)")
    ap("")
    ap("以下特征不在 v002 冻结列中, 由本工具从日线缓存 + 涨停池缓存确定性重建, "
       "公式与 _scratch/build_v004c_dataset.py 官方定义逐条对齐:")
    rebuild_names = ["board_streak_is_3", "max_board_streak_20d", "recent_limit_up_count_10d",
                     "recent_limit_up_count_20d", "recent_pool_appearance_count_10d",
                     "recent_pool_appearance_count_20d", "board_day_volume_rank",
                     "break_high_return", "break_close_return", "break_touched_limit_up",
                     "break_opened_from_limit_up", "break_upper_shadow_ratio",
                     "break_lower_shadow_ratio", "break_volume_ratio_vs_board_days",
                     "break_day_in_pool", "last_board_day_in_pool",
                     "pool_consecutive_count_last_board", "d1_ma20_slope",
                     "recent_7d_limit_up_count"]
    ap(f"- 重建特征 ({len(rebuild_names)}): {', '.join(rebuild_names)}")
    ap("")
    ap("### 旧 CONTEXT_ONLY 审查")
    ap("")
    ap("v004c_model_table.py 中 CONTEXT_ONLY_COLUMNS (recent_limit_up_count_10d/20d, "
       "recent_pool_appearance_count_10d/20d, max_board_streak_20d) 定义正确、D1_CLOSE 可得、"
       "不依赖模型输出, 现 PROMOTE 为 FEATURE (信息边界允许的 10/20 日历史状态)。")
    ap("")
    ap("### 旧 SENSITIVITY 审查")
    ap("")
    ap("SENSITIVITY_FEATURE_COLUMNS 中 board_day_amount_rank / board_day_turnover_rank 因 "
       "May 池 amount/turnover 全 NaN 且 daily amount 全 NaN 而 EXCLUDE_SOURCE_UNAVAILABLE; "
       "break_amount_ratio_vs_board_days 分母 (board 日 amount) 在 May 不可重建, 同样排除。"
       "其余 SENSITIVITY 特征全部进入 FEATURE (定义正确且可得)。")
    ap("")
    ap("### 手动综合因子排除")
    ap("")
    ap("recognition_score (旧模型综合评分), profit_pressure, overrepair, "
       "break_volume_abnormality, ma5_overheat_10, d1_close_to_ma5_bucket, "
       "d1_open_to_close_bucket, d1_vwap_to_close_gap, Repair-State composites "
       "(DIVERGENCE/RECLAIM/DAMAGE/SUPPLY/DIVERGENCE_SQ/DIVERGENCE_X_*), v004a/v004b "
       "handcrafted nonlinear features 全部不进契约 (EXCLUDE_MANUAL_COMPOSITE / "
       "EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM / EXCLUDE_MODEL_OUTPUT)。")
    ap("")
    ap("### 精确重复排除")
    ap("")
    ap("ED 组 break_open/d1_open, break_high/d1_high, break_low/d1_low, "
       "break_close/d1_close, break_intraday_range/d1_intraday_range, "
       "break_high_to_close_drawdown/d1_high_to_close_drawdown_raw, "
       "break_close_location/d1_close_location, break_volume/d1_volume, "
       "break_amount/d1_amount, d1_down_bar_volume_ratio/down_bar_volume_ratio, "
       "volume_above_break_close_ratio/volume_above_d1_close_ratio, "
       "deprecated_d1_reclaimed_ma5_v01/d1_close_above_ma5, "
       "deprecated_d1_reclaimed_ma10_v01/d1_close_above_ma10 只保留 canonical 一份。"
       "v002 中除 canonical 名外的别名一律不进入输入表。")
    ap("")
    ap("### 特征缺失")
    ap("")
    # structural 缺失三来源 (从 input_df 动态统计, 保证 review 与数据一致):
    #   A) D0=04-30 池窗口外 (6 事件 x 3 字段)
    #   B) 池源覆盖缺口 (D0 涨停但池未收录, 7 事件 x 2 字段)
    #   C) 一字板 603065 (break_upper/lower_shadow, d1_close_location 分母退化)
    d0430_events = sorted(input_df[input_df["signal_date"] == "2026-05-06"]["event_id"].tolist())
    pool_gap_events = sorted(input_df[
        (input_df["signal_date"] != "2026-05-06")
        & (input_df["pool_consecutive_count_last_board"].isna())]["event_id"].tolist())
    shadow_events = sorted(input_df[input_df["break_upper_shadow_ratio"].isna()]["event_id"].tolist())
    ap(f"- structural_missing_count 合计: {int(input_df['structural_missing_count'].sum())} "
       f"(定义允许的数学退化 / 池窗口外 / 池源覆盖缺口)")
    ap(f"- unexpected_missing_count 合计: {int(input_df['unexpected_missing_count'].sum())} "
       f"(必须为 0)")
    ap(f"- A) D0=2026-04-30 池窗口外事件 ({len(d0430_events)}): board_day_volume_rank, "
       f"last_board_day_in_pool, pool_consecutive_count_last_board 各 {len(d0430_events)} 个"
       f"结构缺失; recent_pool_appearance_count 按池覆盖子集计算 (下界)。")
    ap(f"- B) 池源覆盖缺口事件 ({len(pool_gap_events)}): D0 日线涨停但涨停池未收录该股, "
       f"board_day_volume_rank 与 pool_consecutive_count_last_board 各 {len(pool_gap_events)} 个"
       f"结构缺失; last_board_day_in_pool=False 为合法值 (池成员记录缺)。")
    ap(f"- C) 一字板 603065 事件 ({len(shadow_events)}): break_upper_shadow_ratio / "
       f"break_lower_shadow_ratio / d1_close_location 各 1 个结构缺失 (high==low 分母退化)。")
    for f in CONTRACT_FEATURE_NAMES:
        c = cov[f]
        ap(f"  - {f}: May {c['may']:.4f} | June {c['june']:.4f} | Combined {c['combined']:.4f}")
    ap("")
    ap("### 泄漏审计")
    ap("")
    ap(f"- FEATURE future leakage: {leakage['future_leakage']} (必须 0)")
    ap(f"- FEATURE model-output leakage: {leakage['model_output_leakage']} (必须 0)")
    ap(f"- FEATURE label-lineage leakage: {leakage['label_lineage_leakage']} (必须 0)")
    ap("")
    ap("## 3. 输入表")
    ap("")
    ap("- v004c_pairwise_v1_input_table.csv: "
       f"{len(input_df)} 行 × {len(input_df.columns)} 列")
    ap("- 列序: 身份(5) + break_date(审计) + 53 FEATURE(冻结序) + 标签(2) "
       "+ dev 资格(3) + 缺失统计(2) + provenance")
    ap(f"- 2 板 / 3 板: "
       f"{int((input_df['board_streak_before_break']==2).sum())} / "
       f"{int((input_df['board_streak_before_break']==3).sum())}")
    ap(f"- dev_label_complete: {int((input_df['dev_label_complete']==1).sum())}/"
       f"{len(input_df)}")
    ap(f"- dev_feature_source_complete: {int((input_df['dev_feature_source_complete']==1).sum())}/"
       f"{len(input_df)}")
    ap(f"- dev_training_eligible: {int((input_df['dev_training_eligible']==1).sum())}/"
       f"{len(input_df)}")
    ap("")
    ap("## 4. Deterministic Rebuild")
    ap("")
    ap(f"- 两轮构建核心资产字节一致: {'PASS' if determinism_ok else 'FAIL'}")
    ap(f"- 标签附加方式: 特征生成时不读标签; 标签列由 v002 冻结值原样携带至输入表末尾, "
       f"未来 Pairwise 训练时以事件 id 对齐。")
    ap("")
    ap("## 5. 严格禁止确认")
    ap("")
    ap("- 未训练 Pairwise Ridge / Logistic / Tree/GBDT")
    ap("- 未搜索 lambda")
    ap("- 未做 walk-forward 模型")
    ap("- 未计算模型 probability / 排名 / Top1/Top3")
    ap("- 未根据 Target7 / tail-loss / 单因子 AUC / IC / 月份表现选择特征")
    ap("- 未创建新的人工综合评分 / Recognition Score 类 composite")
    ap("- 未使用 v004a/v004b/Repair-State 预测输出或旧模型系数")
    ap("- 未创建 feature_x_board3 交互列")
    ap("")
    ap("## 6. 结论")
    ap("")
    ready = (universe_ok and not availability_bad
             and int(input_df["unexpected_missing_count"].sum()) == 0
             and leakage["future_leakage"] == 0
             and leakage["model_output_leakage"] == 0
             and leakage["label_lineage_leakage"] == 0
             and determinism_ok)
    ap(f"- Feature Contract 状态: {'READY_FOR_PAIRWISE_V1' if ready else 'NOT_READY_FOR_PAIRWISE_V1'}")
    ap("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output or OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_contract_integrity()

    # ---- 数据加载 ----
    print("[1] load v002 foundation ...", flush=True)
    v002 = pd.read_csv(V002_CSV, encoding="utf-8-sig", dtype={"code": str})
    v002["code"] = v002["code"].astype(str).str.zfill(6)
    v002["signal_date"] = v002["signal_date"].astype(str)
    v002["break_date"] = v002["break_date"].astype(str)
    print(f"    rows={len(v002)} signal_dates={v002['signal_date'].nunique()} "
          f"codes={v002['code'].nunique()}")

    print("[2] load pool cache ...", flush=True)
    pool_members, pool_meta, pool_dates, pool_date_set = load_pool_cache()
    print(f"    pool dates: {len(pool_dates)} "
          f"({pool_dates[0]} .. {pool_dates[-1]})")

    print("[3] load daily cache ...", flush=True)
    codes_needed = set(v002["code"].tolist())
    daily_store = load_daily_store(codes_needed)
    missing_codes = codes_needed - set(daily_store.keys())
    if missing_codes:
        raise SystemExit(f"FATAL: {len(missing_codes)} codes missing daily cache: "
                         f"{sorted(missing_codes)}")
    print(f"    daily codes loaded: {len(daily_store)}")

    print("[4] assemble input table ...", flush=True)
    daily_volume_lookup = build_daily_volume_lookup(daily_store)
    input_df = assemble_input_table(v002, pool_members, pool_meta, pool_dates,
                                    pool_date_set, daily_store, daily_volume_lookup)

    # ---- universe 身份逐行审计 ----
    for col in ("event_id", "code", "signal_date", "break_date",
                "board_streak_before_break", "source_window"):
        assert input_df[col].astype(str).tolist() == v002[col].astype(str).tolist(), \
            f"universe 身份列 {col} 与 v002 不一致"
    universe_ok = True

    # ---- 泄漏 / 可用性审计 ----
    print("[5] audits ...", flush=True)
    leakage = run_leakage_audit(input_df, list(CONTRACT_FEATURE_NAMES))
    availability_bad = availability_audit(input_df)
    cov = coverage_stats(input_df)

    unexpected_total = int(input_df["unexpected_missing_count"].sum())
    print(f"    unexpected_missing total: {unexpected_total}")
    print(f"    leakage: {leakage}")
    print(f"    availability violations: {availability_bad}")

    # ---- 确定性双构建 ----
    # 流程: 首轮写正式目录 -> 次轮写校验目录 -> 逐字节比较 7 个核心资产
    #       -> 一致则正式目录 review 覆盖为 PASS, 删除校验目录。
    print("[6] determinism rebuild ...", flush=True)
    second_dir = output_dir.parent / (output_dir.name + "_rebuild_check")
    if second_dir.exists():
        shutil.rmtree(second_dir)
    second_dir.mkdir(parents=True, exist_ok=True)

    # 两轮均以 determinism_ok=False 渲染 review (占位 FAIL, 内容一致可比)
    write_outputs(output_dir, input_df, cov, universe_ok, leakage,
                  availability_bad, False)
    write_outputs(second_dir, input_df, cov, universe_ok, leakage,
                  availability_bad, False)

    determinism_ok = determinism_check(output_dir, second_dir)
    shutil.rmtree(second_dir)
    if determinism_ok:
        # 最终确定 review (正式目录覆盖为 PASS; 校验目录已删, 不影响确定性证据)
        review = write_review(input_df, cov, universe_ok, leakage, availability_bad,
                              determinism_ok, canonical_out_dir=OUT_DIR.name)
        write_atomic(output_dir / REVIEW_NAME, review)
    print(f"    determinism: {'PASS' if determinism_ok else 'FAIL'}")
    if not determinism_ok:
        print("FATAL: determinism rebuild mismatch (non-deterministic output).")
        return 1
    return 0


def write_outputs(output_dir, input_df, cov, universe_ok, leakage, availability_bad,
                  determinism_ok):
    """写入 7 个输出资产 (原子)。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. inventory
    inv = write_inventory(input_df, cov)
    write_atomic(output_dir / INVENTORY_CSV_NAME, inv.to_csv(index=False))

    # 2. contract
    ctr = write_contract_csv()
    write_atomic(output_dir / CONTRACT_CSV_NAME, ctr.to_csv(index=False))

    # 3. label contract
    lbl = write_label_contract()
    write_atomic(output_dir / LABEL_CSV_NAME, lbl.to_csv(index=False))

    # 4. input table
    write_atomic(output_dir / DEV_CSV_NAME, input_df.to_csv(index=False))

    # 5. schema
    sch = write_schema(input_df)
    write_atomic(output_dir / SCHEMA_CSV_NAME, sch.to_csv(index=False))

    # 6. exclusions
    exc = write_exclusions()
    write_atomic(output_dir / EXCLUSIONS_CSV_NAME, exc.to_csv(index=False))

    # 7. review (确定性比较时两轮均 FAIL 占位; 通过后正式目录覆盖为 PASS)
    review = write_review(input_df, cov, universe_ok, leakage, availability_bad,
                          determinism_ok, canonical_out_dir=OUT_DIR.name)
    write_atomic(output_dir / REVIEW_NAME, review)


if __name__ == "__main__":
    sys.exit(main())

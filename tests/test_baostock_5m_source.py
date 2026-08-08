# -*- coding: utf-8 -*-
"""BaoStock 5m source adapter — 测试

覆盖 (§8 / §34 适配器契约):
- to_baostock_code 确定性转换 (6/9 -> sh., 其余 -> sz.)
- parse_baostock_time17 17 位 time 解析 / 非法拒绝
- _fetch_5min_baostock: adjustflag=3 (adjust='none') 强制 / 请求参数 / error_code
  失败 raise / 空结果 raise / 挂单零行(等同空结果) raise
- fetch_5min_history 单源 dispatch: source 校验 (ValueError) / baostock 路径
  18 列 schema / numeric coercion / volume=股 / amount=元 (不校准) /
  NO silent fallback (baostock 失败不得尝试 sina)
- normalize_baostock_5m_frame: **fail closed** (v002 修复) — unparseable
  datetime / NaN 核心数值 / 重复 (code, datetime) 一律 RuntimeError, 无 silent drop
- fetch_5min_history (baostock 路径): normalize 后**严格 datetime 裁剪**
  start<=dt<=end (partial-window 无区间外 bar); invalid OHLC / 负 volume 经
  全链路 fail closed; 停牌占位 (OHLC 全 0) 显式分类移除
- validate_normalized_5m_frame: 严格升序 / 重复拒绝 / OHLC finite>0 /
  high>=max(o,c) / low<=min(o,c) / volume/amount>=0 / source/adjust/interval 标记
- login/logout 批量生命周期: 幂等 login (进程内一次)
- Baostock5mCache: 原子写 / 读回一致 / meta 确定性 (timestamp 独立)
- 可选 SOCKS5 代理 (显式 opt-in): 仅拦截 BAOSTOCK_SERVER_HOST:10030 连接;
  其他地址不受影响; clear 恢复原始 connect
"""
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import src.data_sources as ds
from src.cache import Baostock5mCache
from src.config import DataConfig
from src.data_sources import (
    BAOSTOCK_5M_SOURCE,
    NORMALIZED_5M_INTERVAL,
    MarketDataProvider,
    normalize_baostock_5m_frame,
    parse_baostock_time17,
    to_baostock_code,
    validate_normalized_5m_frame,
)

BASE = Path(__file__).resolve().parent.parent

FIELDS = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]


class FakeResultSet:
    def __init__(self, rows, error_code="0", error_msg=""):
        self.fields = list(FIELDS)
        self.error_code = error_code
        self.error_msg = error_msg
        self._rows = [list(r) for r in rows]
        self._i = -1

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


class FakeSocks5Server:
    """本地 SOCKS5 服务端: 记录 CONNECT 目标, 回复成功, 之后 echo 回显。"""

    def __init__(self, bind_atyp=0x01):
        """bind_atyp=0x03 时, bind 回复用 ATYP=3 (domain) 编码 — 用于复现
        §33 domain-bind 长度消费 bug (旧实现多算一个 length byte)。"""
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.bind_atyp = bind_atyp
        self.targets: list[tuple[str, int]] = []
        self.payloads: list[bytes] = []
        self.running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while self.running:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        conn.settimeout(5)
        try:
            greeting = conn.recv(3)
            if greeting[:1] != b"\x05":
                # 非 SOCKS 客户端: 原样回显 (用于验证非 baostock 目标未被拦截)
                self.payloads.append(greeting)
                conn.sendall(greeting)
                return
            conn.sendall(b"\x05\x00")
            req = conn.recv(4)
            if len(req) < 4 or req[3] == 0x03:
                ln = conn.recv(1)[0]
                host = conn.recv(ln).decode()
                port = struct.unpack(">H", conn.recv(2))[0]
            else:
                host, port = "", 0
            self.targets.append((host, port))
            if self.bind_atyp == 0x03:
                domain = b"fake-bind.example.com"
                conn.sendall(b"\x05\x00\x00\x03" + bytes([len(domain)])
                             + domain + b"\x00\x50")
            else:
                conn.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                self.payloads.append(data)
                conn.sendall(data)
        except Exception:
            pass
        finally:
            conn.close()

    def stop(self):
        self.running = False
        self.srv.close()


class PlainEchoServer:
    """普通 TCP echo (无 SOCKS 协议)。"""

    def __init__(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.first_bytes: list[bytes] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            conn, _ = self.srv.accept()
        except OSError:
            return
        conn.settimeout(5)
        try:
            data = conn.recv(4096)
            self.first_bytes.append(data)
            conn.sendall(data)
        except Exception:
            pass
        finally:
            conn.close()

    def stop(self):
        self.srv.close()


class FakeBaostock:
    def __init__(self, result_sets):
        self.result_sets = list(result_sets)
        self.login_calls = 0
        self.logout_calls = 0
        self.queries = []

    def login(self):
        self.login_calls += 1
        return SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        self.logout_calls += 1

    def query_history_k_data_plus(self, **kwargs):
        self.queries.append(kwargs)
        if not self.result_sets:
            raise RuntimeError("no fake result set configured")
        return self.result_sets[min(len(self.queries) - 1, len(self.result_sets) - 1)]


def raw_bao_rows(date="2026-05-06", n=4, code="sh.600000"):
    rows = []
    for i in range(n):
        rows.append([
            date, f"{date.replace('-', '')}0935{i:02d}000", code,
            "10.00", "10.10", "9.90", f"{10.01 + i * 0.01:.2f}",
            f"{1000 + i}", f"{10000.0 + i}",
        ])
    return rows


def make_bao_query(rows=None, n=4, error_code="0", error_msg=""):
    rows = raw_bao_rows(n=n) if rows is None else rows
    return FakeResultSet(rows, error_code=error_code, error_msg=error_msg)


class BaostockSourceTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_module = ds._BAOSTOCK_MODULE
        self._orig_logged = ds._BAOSTOCK_LOGGED_IN
        self.provider = MarketDataProvider(DataConfig())
        self.addCleanup(self._restore)
        self.fake = None

    def _restore(self):
        ds.clear_baostock_socks5_proxy()  # 还原 socket.socket.connect (幂等)
        ds._BAOSTOCK_MODULE = self._orig_module
        ds._BAOSTOCK_LOGGED_IN = self._orig_logged

    def _install_fake(self, fake):
        ds._BAOSTOCK_MODULE = fake
        ds._BAOSTOCK_LOGGED_IN = False
        self.fake = fake

    # ------------------------------------------------------------ code/time
    def test_to_baostock_code_prefix_rules(self):
        self.assertEqual(to_baostock_code("600000"), "sh.600000")
        self.assertEqual(to_baostock_code("900901"), "sh.900901")
        self.assertEqual(to_baostock_code("000001"), "sz.000001")
        self.assertEqual(to_baostock_code("300750"), "sz.300750")
        self.assertEqual(to_baostock_code("830799"), "sz.830799")
        # 6 位输入契约 (provider 先 normalize, 不直接喂带前缀代码)

    def test_parse_baostock_time17(self):
        self.assertEqual(parse_baostock_time17("20260506093500001"), "09:35:00")
        self.assertEqual(parse_baostock_time17("20260506150000000"), "15:00:00")
        with self.assertRaises(ValueError):
            parse_baostock_time17("20260506")

    # ------------------------------------------------------------- fetch/adjust
    def test_baostock_fetch_rejects_adjust(self):
        fake = FakeBaostock([make_bao_query()])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider._fetch_5min_baostock("600000", "2026-05-06 09:00:00",
                                               "2026-05-06 15:30:00", adjust="hfq")

    def test_baostock_fetch_query_parameters(self):
        fake = FakeBaostock([make_bao_query()])
        self._install_fake(fake)
        frame = self.provider._fetch_5min_baostock(
            "600000", "2026-05-06 09:00:00", "2026-06-30 15:30:00", adjust="none")
        self.assertEqual(len(fake.queries), 1)
        q = fake.queries[0]
        self.assertEqual(q["code"], "sh.600000")
        self.assertEqual(q["frequency"], "5")
        self.assertEqual(q["adjustflag"], "3")
        self.assertEqual(q["fields"], "date,time,code,open,high,low,close,volume,amount")
        self.assertEqual(q["start_date"], "2026-05-06")
        self.assertEqual(q["end_date"], "2026-06-30")
        self.assertEqual(len(frame), 4)

    def test_baostock_fetch_query_error_raises(self):
        fake = FakeBaostock([make_bao_query(error_code="10001", error_msg="fail")])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider._fetch_5min_baostock(
                "600000", "2026-05-06 09:00:00", "2026-06-30 15:30:00", adjust="none")

    def test_baostock_fetch_empty_result_raises(self):
        fake = FakeBaostock([make_bao_query(rows=[], n=0)])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider._fetch_5min_baostock(
                "600000", "2026-05-06 09:00:00", "2026-05-06 15:30:00", adjust="none")

    def test_baostock_fetch_suspension_zero_bars_dropped(self):
        # 停牌日 OHLC 全 0 占位 bar 被丢弃, 保留真实成交 bar (Sina 同语义)
        valid = raw_bao_rows(date="2026-05-06", n=2)
        zero = [
            ["2026-05-07", "2026050709350000", "sh.600000",
             "0", "0", "0", "0", "0", "0"],
            ["2026-05-07", "2026050709400000", "sh.600000",
             "0", "0", "0", "0", "0", "0"],
        ]
        fake = FakeBaostock([make_bao_query(rows=valid + zero, n=4)])
        self._install_fake(fake)
        frame = self.provider._fetch_5min_baostock(
            "600000", "2026-05-06 09:00:00", "2026-05-07 15:30:00", adjust="none")
        self.assertEqual(len(frame), 2)
        self.assertNotIn("2026-05-07", set(frame["date"]))

    def test_baostock_fetch_all_zero_bars_raises(self):
        # 窗口内全部为停牌占位 bar -> 丢弃后为空 -> 显式失败, 不做 silent fallback
        zero = [
            ["2026-05-07", "2026050709350000", "sh.600000",
             "0", "0", "0", "0", "0", "0"],
        ]
        fake = FakeBaostock([make_bao_query(rows=zero, n=1)])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider._fetch_5min_baostock(
                "600000", "2026-05-07 09:00:00", "2026-05-07 15:30:00", adjust="none")

    # ---------------------------------------------------------------- dispatch
    def test_fetch_5min_unknown_source_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.provider.fetch_5min_history("600000", "2026-05-06 09:00:00",
                                             "2026-05-06 15:30:00", source="magic")

    def test_fetch_5min_baostock_dispatch_no_silent_fallback(self):
        fake = FakeBaostock([make_bao_query(error_code="1", error_msg="boom")])
        self._install_fake(fake)

        def _sina(*a, **k):
            raise AssertionError("sina_5m must NOT be attempted on baostock failure")

        self.provider._fetch_5min_sina = _sina
        with self.assertRaises(RuntimeError) as ctx:
            self.provider.fetch_5min_history("600000", "2026-05-06 09:00:00",
                                             "2026-05-06 15:30:00",
                                             adjust="none", source=BAOSTOCK_5M_SOURCE)
        self.assertIn("baostock", str(ctx.exception).lower())

    def test_fetch_5min_baostock_dispatch_normalized_schema(self):
        fake = FakeBaostock([make_bao_query(n=4)])
        self._install_fake(fake)
        frame, source = self.provider.fetch_5min_history(
            "600000", "2026-05-06 09:00:00", "2026-05-06 15:30:00",
            adjust="none", source=BAOSTOCK_5M_SOURCE)
        self.assertEqual(source, BAOSTOCK_5M_SOURCE)
        self.assertEqual(list(frame.columns), list(ds.NORMALIZED_5M_COLUMNS))
        self.assertEqual(len(frame), 4)
        self.assertTrue((frame["source"] == BAOSTOCK_5M_SOURCE).all())
        self.assertTrue((frame["adjust"] == "none").all())
        self.assertTrue((frame["interval"] == NORMALIZED_5M_INTERVAL).all())

    # --------------------------------------------------------- normalization
    def test_normalize_numeric_coercion_and_units(self):
        rows = raw_bao_rows(n=2)
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        # volume=股, amount=元, 原样保留 (禁止价格/单位校准)
        self.assertEqual(float(norm["volume"].iloc[0]), 1000.0)
        self.assertEqual(float(norm["amount"].iloc[0]), 10000.0)
        self.assertEqual(float(norm["close"].iloc[0]), 10.01)
        self.assertEqual(norm["trade_date"].iloc[0], "2026-05-06")
        self.assertEqual(norm["time"].iloc[0], "09:35:00")
        self.assertEqual(norm["code"].iloc[0], "600000")

    def test_normalize_unparseable_datetime_fails_closed(self):
        # §5/§14/§15: 原始坏行 (unparseable datetime) 必须 fail closed, 禁止
        # silent drop (v001 用 dropna 吞掉 -> 校验器看不见, 已修复)
        rows = raw_bao_rows(n=2)
        rows[0][0] = "not-a-date"
        raw = pd.DataFrame(rows, columns=FIELDS)
        with self.assertRaises(RuntimeError):
            normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")

    def test_normalize_duplicate_rows_fail_closed(self):
        # §5/§15: 重复 (code, datetime) 必须 fail closed, 禁止 silent dedup
        rows = raw_bao_rows(n=2)
        raw = pd.DataFrame(rows + [rows[1]], columns=FIELDS)
        with self.assertRaises(RuntimeError):
            normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")

    def test_normalize_nan_core_numeric_fails_closed(self):
        # §15: NaN volume 等核心数值列 fail closed (不再 fillna 后放行)
        rows = raw_bao_rows(n=2)
        rows[0][7] = ""  # volume 空串 -> coerce NaN
        raw = pd.DataFrame(rows, columns=FIELDS)
        with self.assertRaises(RuntimeError):
            normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")

    # -------------------------------------------------------------- validator
    def test_fetch_5min_baostock_partial_window_strict_clip(self):
        # §4/§43: baostock API 只支持按日参数; normalize 后必须再次严格裁剪
        # start<=datetime<=end — 10:00..11:00 的请求不得交付 09:35 / 15:00 等
        # 区间外 bar (partial-window future-bar 泄漏防护)
        rows = [
            ["2026-05-06", "20260506093500001", "sh.600000",
             "10.00", "10.10", "9.90", "10.01", "1000", "10000"],
            ["2026-05-06", "20260506100000000", "sh.600000",
             "10.00", "10.10", "9.90", "10.02", "1000", "10000"],
            ["2026-05-06", "20260506105500000", "sh.600000",
             "10.00", "10.10", "9.90", "10.03", "1000", "10000"],
            ["2026-05-06", "20260506110000000", "sh.600000",
             "10.00", "10.10", "9.90", "10.04", "1000", "10000"],
            ["2026-05-06", "20260506110500000", "sh.600000",
             "10.00", "10.10", "9.90", "10.05", "1000", "10000"],
            ["2026-05-06", "20260506150000000", "sh.600000",
             "10.00", "10.10", "9.90", "10.06", "1000", "10000"],
        ]
        fake = FakeBaostock([make_bao_query(rows=rows, n=6)])
        self._install_fake(fake)
        frame, _ = self.provider.fetch_5min_history(
            "600000", "2026-05-06 10:00:00", "2026-05-06 11:00:00",
            adjust="none", source=BAOSTOCK_5M_SOURCE)
        times = frame["datetime"].dt.strftime("%H:%M:%S").tolist()
        self.assertEqual(times, ["10:00:00", "10:55:00", "11:00:00"])
        for outside in ("09:35:00", "11:05:00", "15:00:00"):
            self.assertNotIn(outside, times)

    def test_fetch_5min_baostock_invalid_ohlc_fails_closed(self):
        # §43: invalid OHLC (high < open) 经 fetch 全链路 fail closed, 不做
        # silent drop / 静默修正
        rows = raw_bao_rows(n=2)
        rows[0][3] = "12.0"  # open > high
        fake = FakeBaostock([make_bao_query(rows=rows, n=2)])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider.fetch_5min_history(
                "600000", "2026-05-06 09:00:00", "2026-05-06 15:30:00",
                adjust="none", source=BAOSTOCK_5M_SOURCE)

    def test_fetch_5min_baostock_negative_volume_fails_closed(self):
        rows = raw_bao_rows(n=2)
        rows[0][7] = "-5"
        fake = FakeBaostock([make_bao_query(rows=rows, n=2)])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider.fetch_5min_history(
                "600000", "2026-05-06 09:00:00", "2026-05-06 15:30:00",
                adjust="none", source=BAOSTOCK_5M_SOURCE)

    def test_suspension_placeholder_explicit_classification(self):
        # §43: 停牌占位 (OHLC 全 0) 与一般坏行严格分类, 常量可审计
        self.assertEqual(ds.BAOSTOCK_SUSPENSION_PLACEHOLDER,
                         "baostock_suspension_placeholder")
        self.assertEqual(ds.INVALID_MARKET_BAR, "invalid_market_bar")
        # 全 0 占位被显式移除 (行为已由 test_baostock_fetch_suspension_zero_bars_dropped
        # 覆盖); 下面验证"部分 0"行不是占位, 会被 normalize fail closed:
        rows = raw_bao_rows(n=1)
        rows[0][5] = "0.0"  # low=0 但并非 OHLC 全 0 -> 不是停牌占位
        fake = FakeBaostock([make_bao_query(rows=rows, n=1)])
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            self.provider.fetch_5min_history(
                "600000", "2026-05-06 09:00:00", "2026-05-06 15:30:00",
                adjust="none", source=BAOSTOCK_5M_SOURCE)

    def test_validate_ok_full_48bar_day(self):
        # 48 bar 覆盖 09:35..15:00 每 5 分钟一格 (正式 D1 规则), datetime 唯一
        rows = []
        for i in range(48):
            minute_of_day = 9 * 60 + 35 + i * 5
            hh = minute_of_day // 60
            mm = minute_of_day % 60
            t17 = f"20260506{hh:02d}{mm:02d}00{i:03d}"
            rows.append([
                "2026-05-06", t17, "sh.600000",
                "10.00", "10.10", "9.90", "10.02", "1000", "10000",
            ])
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        self.assertEqual(len(norm), 48)
        validate_normalized_5m_frame(norm)  # 不 raise

    def test_validate_rejects_duplicate_datetime(self):
        # normalize 之后 (datetime 唯一) 手工制造重复 -> validator 拒绝
        rows = raw_bao_rows(n=3)
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        norm.loc[norm.index[-1], "datetime"] = norm.loc[norm.index[0], "datetime"]
        with self.assertRaises(RuntimeError):
            validate_normalized_5m_frame(norm)

    def test_validate_rejects_invalid_ohlc(self):
        rows = raw_bao_rows(n=2)
        rows[0][5] = "-1.0"  # low < 0
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        with self.assertRaises(RuntimeError):
            validate_normalized_5m_frame(norm)

    def test_validate_rejects_high_below_open(self):
        rows = raw_bao_rows(n=2)
        rows[0][3] = "12.0"  # open above high
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        with self.assertRaises(RuntimeError):
            validate_normalized_5m_frame(norm)

    def test_validate_rejects_wrong_source_marker(self):
        rows = raw_bao_rows(n=2)
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        norm.loc[0, "source"] = "sina_5m"
        with self.assertRaises(RuntimeError):
            validate_normalized_5m_frame(norm)

    # ----------------------------------------------------------- login lifecycle
    def test_login_idempotent_and_logout(self):
        fake = FakeBaostock([make_bao_query()])
        self._install_fake(fake)
        ds.login_baostock()
        ds.login_baostock()
        self.assertEqual(fake.login_calls, 1)
        ds.logout_baostock()
        self.assertEqual(fake.logout_calls, 1)
        self.assertFalse(ds._BAOSTOCK_LOGGED_IN)

    def test_login_failure_raises(self):
        fake = FakeBaostock([make_bao_query()])

        def bad_login():
            return SimpleNamespace(error_code="403", error_msg="denied")

        fake.login = bad_login
        self._install_fake(fake)
        with self.assertRaises(RuntimeError):
            ds.login_baostock()

    # ---------------------------------------------------------------- cache
    def test_baostock_cache_deterministic_write_read(self):
        fake = FakeBaostock([make_bao_query(n=4)])
        self._install_fake(fake)
        frame, _ = self.provider.fetch_5min_history(
            "600000", "2026-05-06 09:00:00", "2026-05-06 15:30:00",
            adjust="none", source=BAOSTOCK_5M_SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600000", frame, meta=cache.build_meta(
                "600000", frame, source=BAOSTOCK_5M_SOURCE, adjustment="none",
                interval=NORMALIZED_5M_INTERVAL,
                fetch_timestamp="2026-08-08 00:00:00"))
            back = cache.read("600000")
            self.assertIsNotNone(back)
            self.assertEqual(list(back.columns), list(frame.columns))
            self.assertEqual(len(back), len(frame))
            # 值级比较 (cache 有意将 str 列转 object, 不要求 dtype 相同)
            for col in frame.columns:
                self.assertEqual(
                    back[col].astype(object).fillna("<<NA>>").tolist(),
                    frame[col].astype(object).fillna("<<NA>>").tolist(),
                    f"column {col} differs after cache round-trip")
            meta = cache.read_meta("600000")
            self.assertEqual(meta["code"], "600000")
            self.assertEqual(meta["source"], BAOSTOCK_5M_SOURCE)
            self.assertEqual(meta["adjustment"], "none")
            self.assertEqual(meta["interval"], NORMALIZED_5M_INTERVAL)
            self.assertEqual(meta["row_count"], 4)
            # meta 除 timestamp 外确定性
            meta2 = cache.build_meta(
                "600000", frame, source=BAOSTOCK_5M_SOURCE, adjustment="none",
                interval=NORMALIZED_5M_INTERVAL, fetch_timestamp="2026-08-09 00:00:00")
            meta.pop("fetch_timestamp")
            meta2.pop("fetch_timestamp")
            self.assertEqual(meta, meta2)

    # ------------------------------------------------------------ socks5 proxy
    def test_socks5_proxy_routes_baostock_connection(self):
        proxy = FakeSocks5Server()
        self.addCleanup(proxy.stop)
        ds.set_baostock_socks5_proxy("127.0.0.1", proxy.port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        try:
            s.connect((ds.BAOSTOCK_SERVER_HOST, ds.BAOSTOCK_SERVER_PORT))  # 应被拦截走 SOCKS5
            s.sendall(b"ping")
            self.assertEqual(s.recv(16), b"ping")  # proxy echo 回显
        finally:
            s.close()
        self.assertEqual(proxy.targets, [(ds.BAOSTOCK_SERVER_HOST, ds.BAOSTOCK_SERVER_PORT)])
        self.assertIn(b"ping", proxy.payloads)

    def test_socks5_proxy_does_not_intercept_other_hosts(self):
        echo = PlainEchoServer()
        self.addCleanup(echo.stop)
        proxy = FakeSocks5Server()
        self.addCleanup(proxy.stop)
        ds.set_baostock_socks5_proxy("127.0.0.1", proxy.port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        try:
            s.connect(("127.0.0.1", echo.port))  # 非 baostock 目标 -> 直连
            s.sendall(b"ping")
            self.assertEqual(s.recv(16), b"ping")
        finally:
            s.close()
        # 直连 echo 收到原始载荷, 未经过 SOCKS 握手
        self.assertEqual(echo.first_bytes, [b"ping"])
        self.assertEqual(proxy.targets, [])

    def test_socks5_proxy_domain_bind_reply_length(self):
        # §33 bug 回归: ATYP=0x03 (domain) bind 回复的 length 字节已在 recv(1)
        # 消费, 剩余待消费 = domain bytes + 2; 旧实现多算一个 length byte ->
        # recv 多等一个不存在的字节直至超时 (挂起)。修复后握手完成, echo 可达。
        proxy = FakeSocks5Server(bind_atyp=0x03)
        self.addCleanup(proxy.stop)
        ds.set_baostock_socks5_proxy("127.0.0.1", proxy.port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(8)
        try:
            s.connect((ds.BAOSTOCK_SERVER_HOST, ds.BAOSTOCK_SERVER_PORT))
            s.sendall(b"ping")
            self.assertEqual(s.recv(16), b"ping")  # 证明 bind 回复已完整消费
        finally:
            s.close()
        self.assertEqual(proxy.targets,
                         [(ds.BAOSTOCK_SERVER_HOST, ds.BAOSTOCK_SERVER_PORT)])
        self.assertIn(b"ping", proxy.payloads)

    def test_socks5_proxy_clear_restores_original_connect(self):
        echo = PlainEchoServer()
        self.addCleanup(echo.stop)
        proxy = FakeSocks5Server()
        self.addCleanup(proxy.stop)
        ds.set_baostock_socks5_proxy("127.0.0.1", proxy.port)
        ds.clear_baostock_socks5_proxy()
        self.assertIsNone(ds._BAOSTOCK_SOCKS5_PROXY)
        self.assertFalse(ds._SOCKET_CONNECT_PATCHED)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        try:
            s.connect(("127.0.0.1", echo.port))
            s.sendall(b"ping")
            self.assertEqual(s.recv(16), b"ping")
        finally:
            s.close()
        self.assertEqual(echo.first_bytes, [b"ping"])

    def test_baostock_cache_merge_dedup_latest(self):
        rows = raw_bao_rows(n=2)
        raw = pd.DataFrame(rows, columns=FIELDS)
        norm = normalize_baostock_5m_frame(raw, "600000", BAOSTOCK_5M_SOURCE, "none")
        with tempfile.TemporaryDirectory() as tmp:
            cache = Baostock5mCache(Path(tmp), suffix="5min")
            cache.validator = validate_normalized_5m_frame
            cache.write("600000", norm, meta=None)
            # 相同 datetime 的新行 (时间 15:00) 追加
            extra = pd.DataFrame([[
                "2026-05-06", "20260506150000000", "sh.600000",
                "11.0", "11.1", "10.9", "11.05", "2000", "22000",
            ]], columns=FIELDS)
            extra_norm = normalize_baostock_5m_frame(
                extra, "600000", BAOSTOCK_5M_SOURCE, "none")
            merged = pd.concat([norm, extra_norm], ignore_index=True)
            merged = merged.drop_duplicates(["code", "datetime"], keep="last")
            merged = merged.sort_values("datetime").reset_index(drop=True)
            validate_normalized_5m_frame(merged)
            self.assertEqual(len(merged), 3)
            self.assertEqual(merged["time"].iloc[-1], "15:00:00")


if __name__ == "__main__":
    unittest.main()

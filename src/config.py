from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


@dataclass(frozen=True)
class DataConfig:
    timeout: float = float(os.getenv("FQ_TIMEOUT", "15.0"))
    retries: int = int(os.getenv("FQ_RETRIES", "3"))
    min_interval: float = float(os.getenv("FQ_MIN_INTERVAL", "0.5"))
    timezone: str = os.getenv("FQ_TIMEZONE", "Asia/Shanghai")
    adjust: str = os.getenv("FQ_ADJUST", "none")
    daily_history_days: int = int(os.getenv("FQ_DAILY_HISTORY_DAYS", "180"))
    default_5min_days: int = int(os.getenv("FQ_5MIN_DAYS", "40"))
    min_5min_trade_days: int = int(os.getenv("FQ_MIN_5MIN_DAYS", "20"))
    indicator_warmup_trading_days: int = int(os.getenv("FQ_INDICATOR_WARMUP_TRADING_DAYS", "120"))
    disable_proxy: bool = os.getenv("FQ_DISABLE_PROXY", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    raw_dir: Path = Path(os.getenv("FQ_RAW_DIR", "data/raw"))
    cache_dir: Path = Path(os.getenv("FQ_CACHE_DIR", "data/cache"))
    processed_dir: Path = Path(os.getenv("FQ_PROCESSED_DIR", "data/processed"))
    reports_dir: Path = Path(os.getenv("FQ_REPORTS_DIR", "reports"))

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir / "limit_ups",
            self.raw_dir / "daily",
            self.raw_dir / "minute_5m",
            self.cache_dir / "daily",
            self.cache_dir / "minute_5m",
            self.cache_dir / "limit_ups",
            self.processed_dir,
            self.reports_dir / "daily_signals",
            self.reports_dir / "backtest_results",
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StrategyConfig:
    # Top3 precision-oriented D1 score weights. The score uses only information
    # available after D1 close, before the planned D2 buy.
    graph_quality_weight: float = 0.40
    trend_hold_weight: float = 0.20
    active_cooling_weight: float = 0.15
    entry_width_weight: float = 0.10
    theme_weight: float = 0.10
    support_weight: float = 0.05

    # Candidate grading thresholds. Hard blocks are intentionally loose; Top3
    # accuracy is controlled mainly by the probability-oriented total_score.
    normal_signal_min_score: float = 72
    small_signal_min_score: float = 60
    min_graph_quality_trade: float = 70
    min_graph_quality_watch: float = 50
    min_active_money: float = 35
    max_active_money_trade: float = 90
    min_support_trade: float = 40
    weak_support_min: float = 20
    max_low_absorb_width_trade: float = 5.0

    close_position_strong: float = 0.70
    close_position_weak: float = 0.30
    high_volume_fail_close_pos: float = 0.20
    high_volume_fail_vwap_dev: float = -0.03
    key_zone_reclaim_minutes: int = 30


@lru_cache(maxsize=1)
def get_data_config() -> DataConfig:
    config = DataConfig()
    config.ensure_directories()
    return config


@lru_cache(maxsize=1)
def get_strategy_config() -> StrategyConfig:
    return StrategyConfig()

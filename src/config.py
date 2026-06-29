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
    min_limitup_universe_size: int = int(os.getenv("FQ_MIN_LIMITUP_UNIVERSE_SIZE", "2000"))
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
    # Clean D1 ranking weights. One final score is used by daily usage,
    # historical samples, and ranking validation: total_score.
    trend_hold_weight: float = 0.35
    graph_quality_weight: float = 0.25
    active_cooling_weight: float = 0.20
    entry_width_weight: float = 0.10
    theme_weight: float = 0.10
    support_weight: float = 0.00

    # Loose display thresholds only. They must not create a second hidden
    # ranking chain. Non-stale candidates remain comparable by total_score.
    normal_signal_min_score: float = 60
    small_signal_min_score: float = 50

    # Risk thresholds kept for explicit hard-block checks and later research.
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

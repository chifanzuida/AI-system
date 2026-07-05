"""Configuration for stock forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ModelType = Literal["lightgbm", "xgboost", "linear", "arima"]
MarketType = Literal["us", "cn", "csv"]
TargetType = Literal["log_return", "close"]


@dataclass
class StockConfig:
    """Runtime settings for a single symbol workflow."""

    symbol: str = "AAPL"
    market: MarketType = "us"
    csv_path: str | None = None

    start_date: str = "2020-01-01"
    end_date: str | None = None

    target: TargetType = "log_return"
    train_ratio: float = 0.8
    forecast_horizon: int = 5

    model_type: ModelType = "lightgbm"
    lags: int = 30
    lags_past_covariates: int = 10
    random_state: int = 42

    backtest_start: float = 0.6
    backtest_retrain: bool = True

    output_dir: Path = field(default_factory=lambda: Path("stock_analysis/output"))

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "forecasts").mkdir(exist_ok=True)
        (self.output_dir / "backtests").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

    @property
    def model_path(self) -> Path:
        safe = self.symbol.replace(".", "_").replace("/", "_")
        return self.output_dir / "models" / f"{safe}_{self.model_type}.pkl"

    @property
    def pipeline_path(self) -> Path:
        safe = self.symbol.replace(".", "_").replace("/", "_")
        return self.output_dir / "models" / f"{safe}_pipeline.pkl"

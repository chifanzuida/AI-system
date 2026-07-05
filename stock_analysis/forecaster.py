"""End-to-end stock forecasting pipeline powered by Darts."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.metrics import mae, mape, rmse
from darts.models import ARIMA, LightGBMModel, LinearRegressionModel
from darts.models.forecasting.forecasting_model import ForecastingModel

from stock_analysis.config import StockConfig
from stock_analysis.data.loader import load_ohlcv, save_ohlcv_cache
from stock_analysis.features import build_feature_frame


class StockForecaster:
    """Train, backtest, and predict stock series from real market data."""

    def __init__(self, config: StockConfig | None = None) -> None:
        self.config = config or StockConfig()
        self.raw_df: pd.DataFrame | None = None
        self.feature_df: pd.DataFrame | None = None
        self.target: TimeSeries | None = None
        self.past_cov: TimeSeries | None = None
        self.target_pipeline: Pipeline | None = None
        self.cov_pipeline: Pipeline | None = None
        self.model: ForecastingModel | None = None
        self._target_scaled: TimeSeries | None = None
        self._past_cov_scaled: TimeSeries | None = None

    def fetch_data(self) -> pd.DataFrame:
        """Download or load OHLCV and cache locally."""
        cfg = self.config
        self.raw_df = load_ohlcv(
            symbol=cfg.symbol,
            market=cfg.market,
            start=cfg.start_date,
            end=cfg.end_date,
            csv_path=cfg.csv_path,
        )
        cache_path = cfg.output_dir / "data" / f"{cfg.symbol.replace('.', '_')}.csv"
        save_ohlcv_cache(self.raw_df, cache_path)
        return self.raw_df

    def prepare(self, df: pd.DataFrame | None = None) -> tuple[TimeSeries, TimeSeries]:
        """Build Darts TimeSeries with scaled target and past covariates."""
        if df is None:
            if self.raw_df is None:
                self.fetch_data()
            df = self.raw_df

        self.feature_df = build_feature_frame(df, target=self.config.target)
        target_col = self.feature_df.attrs["target_col"]
        cov_cols = self.feature_df.attrs["past_cov_cols"]

        self.target = TimeSeries.from_dataframe(
            self.feature_df,
            time_col="date",
            value_cols=target_col,
            fill_missing_dates=False,
            freq="B",
        )
        self.past_cov = TimeSeries.from_dataframe(
            self.feature_df,
            time_col="date",
            value_cols=cov_cols,
            fill_missing_dates=False,
            freq="B",
        )

        self.target_pipeline = Pipeline([MissingValuesFiller(), Scaler()])
        self.cov_pipeline = Pipeline([MissingValuesFiller(), Scaler()])

        self._target_scaled = self.target_pipeline.fit_transform(self.target)
        self._past_cov_scaled = self.cov_pipeline.fit_transform(self.past_cov)
        return self._target_scaled, self._past_cov_scaled

    def _build_model(self) -> ForecastingModel:
        cfg = self.config
        encoders = {
            "cyclic": {"future": ["dayofweek", "month"]},
            "datetime_attribute": {"future": ["dayofweek"]},
        }

        if cfg.model_type == "lightgbm":
            return LightGBMModel(
                lags=cfg.lags,
                lags_past_covariates=cfg.lags_past_covariates,
                output_chunk_length=cfg.forecast_horizon,
                add_encoders=encoders,
                random_state=cfg.random_state,
                verbose=-1,
            )
        if cfg.model_type == "xgboost":
            from darts.models import XGBModel

            return XGBModel(
                lags=cfg.lags,
                lags_past_covariates=cfg.lags_past_covariates,
                output_chunk_length=cfg.forecast_horizon,
                add_encoders=encoders,
                random_state=cfg.random_state,
            )
        if cfg.model_type == "linear":
            return LinearRegressionModel(
                lags=cfg.lags,
                lags_past_covariates=cfg.lags_past_covariates,
                output_chunk_length=cfg.forecast_horizon,
                add_encoders=encoders,
                random_state=cfg.random_state,
            )
        if cfg.model_type == "arima":
            return ARIMA(p=2, d=0, q=1)

        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    def train(self, df: pd.DataFrame | None = None, use_full_data: bool = False) -> ForecastingModel:
        """Fit model. Set ``use_full_data=True`` for production forecasting."""
        target_scaled, past_cov_scaled = self.prepare(df)
        if use_full_data:
            train = target_scaled
            cov_train = past_cov_scaled
        else:
            split = int(len(target_scaled) * self.config.train_ratio)
            train = target_scaled[:split]
            cov_train = past_cov_scaled[:split]

        self.model = self._build_model()
        if self.config.model_type == "arima":
            self.model.fit(train)
        else:
            self.model.fit(train, past_covariates=cov_train)

        self._save_artifacts()
        return self.model

    def backtest(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Run rolling historical forecasts and compute metrics."""
        target_scaled, past_cov_scaled = self.prepare(df)
        if self.model is None:
            self.model = self._build_model()

        forecasts = self.model.historical_forecasts(
            series=target_scaled,
            past_covariates=past_cov_scaled,
            start=self.config.backtest_start,
            forecast_horizon=self.config.forecast_horizon,
            stride=1,
            retrain=self.config.backtest_retrain,
            verbose=True,
        )

        actual = target_scaled.slice(forecasts.start_time(), forecasts.end_time())
        metrics = {
            "mae": float(mae(actual, forecasts)),
            "rmse": float(rmse(actual, forecasts)),
            "mape": float(mape(actual, forecasts)),
            "n_points": len(forecasts),
        }

        result_df = self._forecast_to_frame(forecasts, label="backtest")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = self.config.symbol.replace(".", "_")
        out_path = self.config.output_dir / "backtests" / f"{safe}_{ts}.csv"
        result_df.to_csv(out_path, index=False)

        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics["backtest_path"] = str(out_path)
        return metrics

    def predict(self, df: pd.DataFrame | None = None, n: int | None = None) -> pd.DataFrame:
        """
        Produce a formal forward forecast.

        Trains on all available history unless a model is already loaded.
        """
        target_scaled, past_cov_scaled = self.prepare(df)
        horizon = n or self.config.forecast_horizon

        if self.model is None:
            self.train(df, use_full_data=True)

        assert self.model is not None

        if self.config.model_type == "arima":
            forecast_scaled = self.model.predict(n=horizon, series=target_scaled)
        else:
            forecast_scaled = self.model.predict(
                n=horizon,
                series=target_scaled,
                past_covariates=past_cov_scaled,
            )

        assert self.target_pipeline is not None
        forecast = self.target_pipeline.inverse_transform(forecast_scaled)
        result = self._forecast_to_frame(forecast, label="forecast")

        last_close = float(self.feature_df["close"].iloc[-1])
        if self.config.target == "log_return":
            prices = [last_close]
            for lr in forecast.values().flatten():
                prices.append(prices[-1] * float(np.exp(lr)))
            result["predicted_close"] = prices[1:]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = self.config.symbol.replace(".", "_")
        out_path = self.config.output_dir / "forecasts" / f"{safe}_{ts}.csv"
        result.to_csv(out_path, index=False)
        result.attrs["output_path"] = str(out_path)
        return result

    def load(self) -> None:
        """Load a saved model and preprocessing pipelines."""
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        self.model = ForecastingModel.load(str(self.config.model_path))
        with open(self.config.pipeline_path, "rb") as f:
            pipelines = pickle.load(f)
        self.target_pipeline = pipelines["target"]
        self.cov_pipeline = pipelines["cov"]

    def _save_artifacts(self) -> None:
        assert self.model is not None
        assert self.target_pipeline is not None
        assert self.cov_pipeline is not None

        self.model.save(str(self.config.model_path))
        with open(self.config.pipeline_path, "wb") as f:
            pickle.dump(
                {"target": self.target_pipeline, "cov": self.cov_pipeline},
                f,
            )

        meta = {
            "symbol": self.config.symbol,
            "market": self.config.market,
            "model_type": self.config.model_type,
            "target": self.config.target,
            "forecast_horizon": self.config.forecast_horizon,
            "trained_at": datetime.now().isoformat(),
            "rows": len(self.feature_df) if self.feature_df is not None else 0,
        }
        meta_path = self.config.model_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def _forecast_to_frame(series: TimeSeries, label: str) -> pd.DataFrame:
        values = series.values().flatten()
        times = series.time_index
        target_name = series.components[0] if series.width == 1 else "value"
        return pd.DataFrame(
            {
                "date": times,
                "type": label,
                target_name: values,
            }
        )

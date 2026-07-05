"""Command-line interface for stock forecasting."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock forecasting with Darts and real market data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", default="AAPL", help="Ticker symbol")
    common.add_argument(
        "--market",
        choices=["us", "cn", "csv"],
        default="us",
        help="Data source: yfinance (us), akshare (cn), or local csv",
    )
    common.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    common.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    common.add_argument("--csv-path", default=None, help="Local OHLCV CSV path")
    common.add_argument(
        "--model",
        choices=["lightgbm", "xgboost", "linear", "arima"],
        default="lightgbm",
    )
    common.add_argument("--horizon", type=int, default=5, help="Forecast horizon (days)")
    common.add_argument(
        "--target",
        choices=["log_return", "close"],
        default="log_return",
    )

    sub.add_parser("fetch", parents=[common], help="Download and cache market data")
    sub.add_parser("train", parents=[common], help="Train on train/val split")
    predict_p = sub.add_parser(
        "predict",
        parents=[common],
        help="Train on full history and produce forward forecast",
    )
    predict_p.add_argument("--load", action="store_true", help="Use saved model if present")
    sub.add_parser("backtest", parents=[common], help="Rolling backtest with metrics")

    return parser


def main(argv: list[str] | None = None) -> int:
    from stock_analysis.config import StockConfig
    from stock_analysis.forecaster import StockForecaster

    args = _build_parser().parse_args(argv)
    config = StockConfig(
        symbol=args.symbol,
        market=args.market,
        csv_path=args.csv_path,
        start_date=args.start,
        end_date=args.end,
        model_type=args.model,
        forecast_horizon=args.horizon,
        target=args.target,
    )
    forecaster = StockForecaster(config)

    if args.command == "fetch":
        df = forecaster.fetch_data()
        print(f"Fetched {len(df)} rows for {config.symbol}")
        print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"Cached: {config.output_dir / 'data'}")
        return 0

    if args.command == "train":
        forecaster.train()
        print(f"Model saved: {config.model_path}")
        return 0

    if args.command == "predict":
        if getattr(args, "load", False) and config.model_path.exists():
            forecaster.load()
            result = forecaster.predict()
        else:
            result = forecaster.predict()
        print(result.to_string(index=False))
        print(f"\nSaved: {result.attrs.get('output_path')}")
        return 0

    if args.command == "backtest":
        metrics = forecaster.backtest()
        print("Backtest metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

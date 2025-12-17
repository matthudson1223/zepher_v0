#!/usr/bin/env python3
"""
Bitcoin Volatility Trading System - Main Entry Point

A system for collecting Bitcoin price data, calculating volatility metrics,
and generating trading signals based on volatility patterns.

Usage:
    python main.py fetch --days 365
    python main.py analyze
    python main.py plot --output charts/btc_analysis.html
    python main.py demo
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import get_config
from src.data.collector import ExchangeCollector
from src.data.database import Database
from src.indicators.volatility import VolatilityCalculator
from src.utils.logger import setup_logger
from src.visualization.charts import ChartBuilder


def setup_logging():
    """Initialize logging based on configuration."""
    config = get_config()
    return setup_logger(
        name="btc_volatility",
        level=config.get("logging.level", "INFO"),
        log_file=config.get_path("logging.file"),
    )


def cmd_fetch(args):
    """Fetch OHLCV data from exchange and store in database."""
    logger = setup_logging()
    config = get_config()

    logger.info(f"Starting data fetch: {args.symbol} from {args.exchange}")

    # Initialize collector
    collector = ExchangeCollector(args.exchange)

    # Initialize database
    db_path = config.get_path("database.path")
    db = Database(db_path)

    # Calculate date range
    end_date = datetime.now(timezone.utc)

    # Check for existing data to avoid refetching
    latest = db.get_latest_timestamp(args.exchange, args.symbol, args.timeframe)
    if latest and not args.force:
        # Resume from last timestamp
        start_date = datetime.fromtimestamp(latest / 1000, tz=timezone.utc)
        logger.info(f"Resuming from {start_date.date()}")
    else:
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"Fetching {args.days} days of data")

    # Fetch data with progress callback
    def progress(current_date, total):
        logger.info(f"Progress: {current_date.date()} - {total} candles fetched")

    df = collector.fetch_ohlcv_range(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress if args.verbose else None,
    )

    if df.empty:
        logger.warning("No data fetched")
        return 1

    # Store in database
    inserted = db.insert_ohlcv(
        df,
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    logger.info(f"Fetch complete: {inserted} new records stored")

    # Show summary
    summary = db.get_data_summary()
    print("\nData Summary:")
    print(summary.to_string(index=False))

    return 0


def cmd_analyze(args):
    """Analyze stored data and display volatility metrics."""
    logger = setup_logging()
    config = get_config()

    logger.info("Starting volatility analysis")

    # Load data from database
    db_path = config.get_path("database.path")
    db = Database(db_path)

    df = db.get_ohlcv(args.exchange, args.symbol, args.timeframe)

    if df.empty:
        logger.error("No data found. Run 'fetch' command first.")
        return 1

    # Calculate indicators
    calc = VolatilityCalculator(
        annualization_factor=config.get("volatility.annualization_factor", 365),
        bb_period=config.get("bollinger.period", 20),
        bb_std=config.get("bollinger.std_dev", 2.0),
        atr_period=config.get("atr.period", 14),
    )

    df = calc.add_indicators_to_df(df)

    # Display latest metrics
    latest = df.iloc[-1]
    print("\n" + "=" * 60)
    print(f"VOLATILITY ANALYSIS: {args.symbol}")
    print(f"Data Range: {df.index[0].date()} to {df.index[-1].date()}")
    print("=" * 60)

    print(f"\nLatest Price: ${latest['close']:,.2f}")
    print(f"\nHistorical Volatility (Annualized):")
    print(f"  7-Day:   {latest['vol_7d']:.2%}")
    print(f"  30-Day:  {latest['vol_30d']:.2%}")
    print(f"  90-Day:  {latest['vol_90d']:.2%}")

    print(f"\nATR (14-period): ${latest['atr']:,.2f}")

    print(f"\nBollinger Bands (20, 2):")
    print(f"  Upper:  ${latest['bb_upper']:,.2f}")
    print(f"  Middle: ${latest['bb_middle']:,.2f}")
    print(f"  Lower:  ${latest['bb_lower']:,.2f}")
    print(f"  %B:     {latest['bb_percent_b']:.2%}")

    print(f"\nVolatility Percentile: {latest['vol_percentile']:.1f}%")

    # Interpretation
    print("\nInterpretation:")
    if latest["vol_percentile"] >= 80:
        print("  ⚠️  HIGH volatility regime - consider mean reversion strategies")
    elif latest["vol_percentile"] <= 20:
        print("  📊 LOW volatility regime - watch for breakout opportunities")
    else:
        print("  📈 NORMAL volatility regime")

    if latest["bb_percent_b"] > 1:
        print("  🔴 Price ABOVE upper Bollinger Band - overbought signal")
    elif latest["bb_percent_b"] < 0:
        print("  🟢 Price BELOW lower Bollinger Band - oversold signal")

    print("=" * 60)

    return 0


def cmd_plot(args):
    """Generate interactive chart with price and indicators."""
    logger = setup_logging()
    config = get_config()

    logger.info("Generating chart")

    # Load data
    db_path = config.get_path("database.path")
    db = Database(db_path)

    df = db.get_ohlcv(args.exchange, args.symbol, args.timeframe)

    if df.empty:
        logger.error("No data found. Run 'fetch' command first.")
        return 1

    # Limit to last N days if specified
    if args.last_days:
        cutoff = df.index[-1] - timedelta(days=args.last_days)
        df = df[df.index >= cutoff]

    # Calculate indicators
    calc = VolatilityCalculator()
    df = calc.add_indicators_to_df(df)

    # Build chart
    builder = ChartBuilder(df, title=f"{args.symbol} Volatility Analysis")
    builder.add_candlestick()
    builder.add_bollinger_bands()
    builder.add_volume()

    if args.show_percentile:
        builder.add_volatility_percentile()
    else:
        builder.add_volatility()

    # Save or show
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save_html(output_path)

    print(f"Chart saved to: {output_path.absolute()}")

    if args.show:
        builder.show()

    return 0


def cmd_demo(args):
    """Run a complete demo: fetch data, analyze, and create chart."""
    logger = setup_logging()

    print("\n" + "=" * 60)
    print("BITCOIN VOLATILITY TRADING SYSTEM - DEMO")
    print("=" * 60)

    # Step 1: Fetch data - try multiple exchanges
    exchanges_to_try = ["kraken", "coinbase", "binance"]
    df = pd.DataFrame()

    for exchange_id in exchanges_to_try:
        print(f"\n[1/3] Fetching BTC/USDT data from {exchange_id.capitalize()}...")
        try:
            collector = ExchangeCollector(exchange_id)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=90)  # 90 days for demo

            df = collector.fetch_ohlcv_range(
                symbol="BTC/USDT",
                timeframe="1d",
                start_date=start_date,
                end_date=end_date,
            )

            if not df.empty:
                print(f"Successfully fetched data from {exchange_id.capitalize()}")
                break
        except Exception as e:
            print(f"  {exchange_id} unavailable: {e}")
            continue

    if df.empty:
        print("Failed to fetch data from any exchange. Check your internet connection.")
        return 1

    print(f"✓ Fetched {len(df)} daily candles")

    # Convert timestamp to datetime index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Step 2: Calculate volatility
    print("\n[2/3] Calculating volatility metrics...")

    calc = VolatilityCalculator()
    df = calc.add_indicators_to_df(df)

    latest = df.iloc[-1]
    print(f"✓ 30-Day Volatility: {latest['vol_30d']:.2%} (annualized)")
    print(f"✓ Volatility Percentile: {latest['vol_percentile']:.1f}%")

    # Step 3: Create chart
    print("\n[3/3] Generating interactive chart...")

    output_path = Path("output/btc_demo_chart.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    builder = ChartBuilder(df, title="BTC/USDT - 90 Day Volatility Analysis")
    builder.add_candlestick()
    builder.add_bollinger_bands()
    builder.add_volume()
    builder.add_volatility()
    builder.save_html(output_path)

    print(f"✓ Chart saved to: {output_path.absolute()}")

    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nLatest BTC Price: ${latest['close']:,.2f}")
    print(f"Bollinger Band Position: {latest['bb_percent_b']:.1%}")

    if latest['vol_percentile'] >= 80:
        print("\n⚠️  Volatility is HIGH - potential mean reversion opportunity")
    elif latest['vol_percentile'] <= 20:
        print("\n📊 Volatility is LOW - watch for breakout")
    else:
        print("\n📈 Volatility is in NORMAL range")

    print(f"\nOpen the chart: {output_path.absolute()}")

    return 0


# Need pandas for demo
import pandas as pd


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Volatility Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                     # Run complete demo
  python main.py fetch --days 365         # Fetch 1 year of data
  python main.py analyze                  # Show volatility metrics
  python main.py plot --output chart.html # Generate chart
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch OHLCV data")
    fetch_parser.add_argument(
        "--exchange",
        default="kraken",
        choices=["binance", "coinbase", "kraken"],
        help="Exchange to fetch from (default: kraken)",
    )
    fetch_parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)",
    )
    fetch_parser.add_argument(
        "--timeframe",
        default="1d",
        choices=["1h", "4h", "1d"],
        help="Candle timeframe (default: 1d)",
    )
    fetch_parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history to fetch (default: 365)",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Force refetch all data (ignore existing)",
    )
    fetch_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress during fetch",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze volatility")
    analyze_parser.add_argument(
        "--exchange",
        default="kraken",
        help="Exchange (default: kraken)",
    )
    analyze_parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)",
    )
    analyze_parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe (default: 1d)",
    )

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate chart")
    plot_parser.add_argument(
        "--exchange",
        default="kraken",
        help="Exchange (default: kraken)",
    )
    plot_parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)",
    )
    plot_parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe (default: 1d)",
    )
    plot_parser.add_argument(
        "--output",
        default="output/btc_chart.html",
        help="Output file path (default: output/btc_chart.html)",
    )
    plot_parser.add_argument(
        "--last-days",
        type=int,
        default=90,
        help="Only plot last N days (default: 90)",
    )
    plot_parser.add_argument(
        "--show-percentile",
        action="store_true",
        help="Show volatility percentile instead of raw volatility",
    )
    plot_parser.add_argument(
        "--show",
        action="store_true",
        help="Open chart in browser after saving",
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run complete demo")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "fetch": cmd_fetch,
        "analyze": cmd_analyze,
        "plot": cmd_plot,
        "demo": cmd_demo,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

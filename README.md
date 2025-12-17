# Bitcoin Volatility Trading System

A Python-based system for collecting Bitcoin price data, calculating volatility metrics, and generating trading signals based on volatility patterns.

## Features

- **Data Collection**: Fetch OHLCV data from multiple exchanges (Binance, Coinbase, Kraken) via ccxt
- **Volatility Metrics**:
  - Historical volatility (7d, 30d, 90d rolling windows)
  - Bollinger Bands (configurable period and standard deviation)
  - ATR (Average True Range)
  - Volatility percentile rankings
- **Trading Strategies** (Phase 2 - Complete):
  - Mean Reversion: Trade price extremes in high volatility regimes
  - Breakout: Capture directional moves after volatility compression
  - Volatility Compression: Anticipate expansions after squeeze periods
- **Signal Generation**:
  - Automated signal generation based on market conditions
  - Signal strength scoring (0-100)
  - Confluence detection when strategies agree
  - Signal filtering and validation
- **Visualization**: Interactive Plotly charts with price, Bollinger Bands, volatility overlays, and trading signals
- **Data Storage**: SQLite database for persistent historical data and trading signals
- **CLI Interface**: Easy-to-use command-line interface with strategy commands

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd zepher_v0

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the demo to see the system in action:

```bash
python main.py demo
```

This will:
1. Fetch 90 days of BTC/USDT data from Kraken (default)
2. Calculate volatility metrics (Historical Volatility, Bollinger Bands, ATR)
3. Generate an interactive HTML chart

## Usage

### Fetch Data

```bash
# Fetch 1 year of daily BTC/USDT data from Kraken (default)
python main.py fetch --days 365

# Fetch from a different exchange
python main.py fetch --exchange coinbase --days 180

# Fetch hourly data
python main.py fetch --timeframe 1h --days 30
```

### Analyze Volatility

```bash
# Display current volatility metrics
python main.py analyze
```

Output includes:
- Current price
- Historical volatility (7d, 30d, 90d annualized)
- ATR value
- Bollinger Band levels
- Volatility percentile ranking

### Generate Charts

```bash
# Create interactive chart
python main.py plot --output charts/btc_analysis.html

# Plot last 30 days only
python main.py plot --last-days 30

# Show volatility percentile instead of raw volatility
python main.py plot --show-percentile

# Open in browser after saving
python main.py plot --show
```

### Generate Trading Signals

```bash
# Generate trading signals based on current market conditions
python main.py signal

# Analyze last 50 days for signals
python main.py signal --lookback-days 50

# Show more signals
python main.py signal --limit 20

# Show historical signal summary
python main.py signal --show-history

# Reset strategy positions before generating
python main.py signal --reset-positions

# Verbose output with detailed indicators
python main.py signal -v
```

Signal output includes:
- Active trading signals from all enabled strategies
- Signal strength and reasoning
- Current market regime (HIGH/LOW/NORMAL volatility)
- Active positions tracking

## Project Structure

```
zepher_v0/
├── src/
│   ├── config.py              # Configuration management
│   ├── data/
│   │   ├── collector.py       # Exchange data fetching
│   │   └── database.py        # SQLite storage with signals
│   ├── indicators/
│   │   └── volatility.py      # Volatility calculations
│   ├── strategies/            # Trading strategies (Phase 2)
│   │   ├── base.py            # Base strategy class
│   │   ├── mean_reversion.py  # Mean reversion strategy
│   │   ├── breakout.py        # Breakout strategy
│   │   └── compression.py     # Volatility compression strategy
│   ├── signals/               # Signal generation (Phase 2)
│   │   ├── generator.py       # Signal generation engine
│   │   ├── filters.py         # Signal filtering
│   │   └── validator.py       # Signal validation
│   ├── visualization/
│   │   └── charts.py          # Plotly charts with signals
│   └── utils/
│       └── logger.py          # Logging setup
├── config/
│   └── settings.yaml          # Strategy parameters
├── tests/
│   └── test_strategies.py    # Strategy unit tests
├── data/                      # SQLite database
├── logs/                      # Log files
├── output/                    # Generated charts
├── main.py                    # CLI entry point
└── requirements.txt
```

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
# Volatility parameters
volatility:
  windows:
    short: 7
    medium: 30
    long: 90
  annualization_factor: 365

# Bollinger Bands
bollinger:
  period: 20
  std_dev: 2.0

# ATR
atr:
  period: 14

# Trading Strategies (Phase 2)
strategies:
  mean_reversion:
    enabled: true
    vol_percentile_min: 80    # Only trade in high volatility
    bb_upper_threshold: 1.0   # Short when %B > 1
    bb_lower_threshold: 0.0   # Long when %B < 0

  breakout:
    enabled: true
    vol_percentile_max: 20    # Only trade in low volatility
    bb_squeeze_threshold: 0.03 # Bollinger squeeze detection
    volume_multiplier: 1.5    # Volume confirmation

  compression:
    enabled: true
    bb_width_percentile: 10   # Bottom 10% of BB width
    expansion_threshold: 1.5  # Exit when BB expands 50%
```

## API Keys (Optional)

For higher rate limits, add API keys to a `.env` file:

```
KRAKEN_API_KEY=your_api_key
KRAKEN_SECRET=your_secret
# Optional:
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret
```

## Programmatic Usage

```python
from src.data.collector import ExchangeCollector
from src.data.database import Database
from src.indicators.volatility import VolatilityCalculator
from src.signals.generator import SignalGenerator
from src.visualization.charts import ChartBuilder
from src.config import get_config

# Fetch data
collector = ExchangeCollector("binance")
df = collector.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=90)

# Calculate indicators
calc = VolatilityCalculator()
df = calc.add_indicators_to_df(df)

# Generate signals (Phase 2)
config = get_config()
db = Database(config.get_path("database.path"))
signal_gen = SignalGenerator(config.data, database=db)
signals = signal_gen.generate_signals(df)

# Create chart with signals
builder = ChartBuilder(df, title="BTC/USDT Analysis")
builder.add_candlestick()
builder.add_bollinger_bands()
builder.add_signals(signals)  # Add trading signals to chart
builder.add_volume()
builder.add_volatility()
builder.save_html("chart.html")
```

## Roadmap

- [x] **Phase 2**: Trading strategies (mean reversion, breakout, volatility compression) ✅ Complete
- [ ] **Phase 3**: Backtesting framework with performance metrics
- [ ] **Phase 4**: Risk management and paper trading

## License

MIT

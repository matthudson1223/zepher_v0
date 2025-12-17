# Bitcoin Volatility Trading System

A Python-based system for collecting Bitcoin price data, calculating volatility metrics, and generating trading signals based on volatility patterns.

## Features

- **Data Collection**: Fetch OHLCV data from multiple exchanges (Binance, Coinbase, Kraken) via ccxt
- **Volatility Metrics**:
  - Historical volatility (7d, 30d, 90d rolling windows)
  - Bollinger Bands (configurable period and standard deviation)
  - ATR (Average True Range)
  - Volatility percentile rankings
- **Visualization**: Interactive Plotly charts with price, Bollinger Bands, and volatility overlays
- **Data Storage**: SQLite database for persistent historical data
- **CLI Interface**: Easy-to-use command-line interface

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

## Project Structure

```
zepher_v0/
├── src/
│   ├── config.py              # Configuration management
│   ├── data/
│   │   ├── collector.py       # Exchange data fetching
│   │   └── database.py        # SQLite storage
│   ├── indicators/
│   │   └── volatility.py      # Volatility calculations
│   ├── visualization/
│   │   └── charts.py          # Plotly charts
│   └── utils/
│       └── logger.py          # Logging setup
├── config/
│   └── settings.yaml          # Strategy parameters
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
from src.indicators.volatility import VolatilityCalculator
from src.visualization.charts import ChartBuilder

# Fetch data
collector = ExchangeCollector("binance")
df = collector.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=90)

# Calculate indicators
calc = VolatilityCalculator()
df = calc.add_indicators_to_df(df)

# Create chart
builder = ChartBuilder(df, title="BTC/USDT Analysis")
builder.add_candlestick()
builder.add_bollinger_bands()
builder.add_volume()
builder.add_volatility()
builder.save_html("chart.html")
```

## Roadmap

- [ ] **Phase 2**: Trading strategies (mean reversion, breakout, volatility compression)
- [ ] **Phase 3**: Backtesting framework with performance metrics
- [ ] **Phase 4**: Risk management and paper trading

## License

MIT

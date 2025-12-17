"""
Exchange data collector module for fetching OHLCV data.

Uses ccxt library to interface with cryptocurrency exchanges.
Supports multiple exchanges with rate limiting and error handling.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("btc_volatility.collector")


class ExchangeCollector:
    """
    Collector for fetching OHLCV data from cryptocurrency exchanges.

    Handles rate limiting, pagination, and error recovery when fetching
    historical price data from supported exchanges.

    Attributes:
        exchange_id: Name of the exchange (e.g., "kraken").
        exchange: ccxt exchange instance.

    Example:
        >>> collector = ExchangeCollector("kraken")
        >>> df = collector.fetch_ohlcv("BTC/USDT", "1d", limit=30)
        >>> print(df.tail())
    """

    # Supported exchanges and their configurations
    EXCHANGE_CONFIGS = {
        "binance": {
            "class": ccxt.binance,
            "rate_limit_ms": 50,  # 1200 requests/minute = 50ms between requests
            "max_candles": 1000,
        },
        "binanceus": {
            "class": ccxt.binanceus,
            "rate_limit_ms": 50,
            "max_candles": 1000,
        },
        "coinbase": {
            "class": ccxt.coinbase,
            "rate_limit_ms": 200,  # 300 requests/minute
            "max_candles": 300,
        },
        "kraken": {
            "class": ccxt.kraken,
            "rate_limit_ms": 200,
            "max_candles": 720,
        },
        "kucoin": {
            "class": ccxt.kucoin,
            "rate_limit_ms": 100,
            "max_candles": 1500,
        },
    }

    # Timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
    }

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        sandbox: bool = False,
    ):
        """
        Initialize the exchange collector.

        Args:
            exchange_id: Exchange identifier (binance, coinbase, kraken).
            api_key: Optional API key for authenticated requests.
            secret: Optional API secret.
            sandbox: Whether to use sandbox/testnet mode.

        Raises:
            ValueError: If exchange_id is not supported.
        """
        exchange_id = exchange_id.lower()

        if exchange_id not in self.EXCHANGE_CONFIGS:
            supported = ", ".join(self.EXCHANGE_CONFIGS.keys())
            raise ValueError(
                f"Unsupported exchange: {exchange_id}. Supported: {supported}"
            )

        self.exchange_id = exchange_id
        self._config = self.EXCHANGE_CONFIGS[exchange_id]

        # Initialize exchange
        exchange_class = self._config["class"]
        exchange_options = {
            "enableRateLimit": True,
            "rateLimit": self._config["rate_limit_ms"],
        }

        if api_key and secret:
            exchange_options["apiKey"] = api_key
            exchange_options["secret"] = secret

        self.exchange = exchange_class(exchange_options)

        if sandbox:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"Initialized {exchange_id} collector")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime | int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data for a trading pair.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w).
            since: Start time as datetime or Unix timestamp (ms).
                   If None, fetches the most recent candles.
            limit: Number of candles to fetch. If None, uses exchange max.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.

        Raises:
            ccxt.NetworkError: On network/connectivity issues.
            ccxt.ExchangeError: On exchange-side errors.

        Example:
            >>> df = collector.fetch_ohlcv(
            ...     "BTC/USDT",
            ...     timeframe="1d",
            ...     since=datetime(2024, 1, 1),
            ...     limit=100
            ... )
        """
        # Convert datetime to Unix timestamp (ms)
        if isinstance(since, datetime):
            since = int(since.timestamp() * 1000)

        # Use exchange max if limit not specified
        if limit is None:
            limit = self._config["max_candles"]

        # Validate timeframe
        if timeframe not in self.TIMEFRAME_MS:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        logger.info(
            f"Fetching {symbol} {timeframe} data from {self.exchange_id} "
            f"(limit={limit}, since={since})"
        )

        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            logger.info(f"Fetched {len(df)} candles for {symbol}")

            return df

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            raise

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a date range, handling pagination automatically.

        Makes multiple API calls to fetch all data within the specified range,
        respecting rate limits and handling pagination.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive). If None, uses current time.
            progress_callback: Optional callback(current_date, total_candles)
                              called after each batch.

        Returns:
            DataFrame with all OHLCV data in the range.

        Example:
            >>> df = collector.fetch_ohlcv_range(
            ...     "BTC/USDT",
            ...     "1d",
            ...     start_date=datetime(2023, 1, 1),
            ...     end_date=datetime(2024, 1, 1)
            ... )
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Ensure timezone-aware datetimes
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        timeframe_ms = self.TIMEFRAME_MS[timeframe]
        max_candles = self._config["max_candles"]

        all_data = []
        current_since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        logger.info(
            f"Fetching {symbol} {timeframe} data from "
            f"{start_date.date()} to {end_date.date()}"
        )

        request_count = 0
        max_requests = 1000  # Safety limit

        while current_since < end_timestamp and request_count < max_requests:
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=max_candles,
                )

                if df.empty:
                    break

                # Filter to only include data within range
                df = df[df["timestamp"] <= end_timestamp]

                if df.empty:
                    break

                all_data.append(df)

                # Update progress
                if progress_callback:
                    current_dt = datetime.fromtimestamp(
                        df["timestamp"].iloc[-1] / 1000, tz=timezone.utc
                    )
                    total_candles = sum(len(d) for d in all_data)
                    progress_callback(current_dt, total_candles)

                # Move to next batch
                last_timestamp = int(df["timestamp"].iloc[-1])
                current_since = last_timestamp + timeframe_ms

                # If we got fewer than expected, we've reached the end
                if len(df) < max_candles:
                    break

                # Rate limiting pause
                time.sleep(self._config["rate_limit_ms"] / 1000)

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                continue
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.error(f"Error fetching data: {e}")
                # Retry after a short pause
                time.sleep(5)
                request_count += 1
                continue

            request_count += 1

        if not all_data:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Combine all data
        result = pd.concat(all_data, ignore_index=True)

        # Remove any duplicates (can happen at batch boundaries)
        result = result.drop_duplicates(subset=["timestamp"], keep="first")
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Total fetched: {len(result)} candles for {symbol}")

        return result

    def get_available_symbols(self) -> list[str]:
        """
        Get list of available trading pairs on the exchange.

        Returns:
            List of symbol strings (e.g., ["BTC/USDT", "ETH/USDT", ...]).
        """
        self.exchange.load_markets()
        return list(self.exchange.symbols)

    def get_exchange_info(self) -> dict:
        """
        Get exchange information including rate limits and capabilities.

        Returns:
            Dictionary with exchange metadata.
        """
        return {
            "id": self.exchange_id,
            "name": self.exchange.name,
            "countries": getattr(self.exchange, "countries", []),
            "rate_limit_ms": self._config["rate_limit_ms"],
            "max_candles_per_request": self._config["max_candles"],
            "timeframes": list(self.exchange.timeframes.keys())
            if hasattr(self.exchange, "timeframes")
            else list(self.TIMEFRAME_MS.keys()),
        }


def fetch_btc_data(
    exchange_id: str = "kraken",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    days: int = 365,
) -> pd.DataFrame:
    """
    Convenience function to fetch Bitcoin historical data.

    Args:
        exchange_id: Exchange to fetch from.
        symbol: Trading pair.
        timeframe: Candle timeframe.
        days: Number of days of history to fetch.

    Returns:
        DataFrame with OHLCV data.

    Example:
        >>> df = fetch_btc_data(days=30)
        >>> print(df.tail())
    """
    collector = ExchangeCollector(exchange_id)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    return collector.fetch_ohlcv_range(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

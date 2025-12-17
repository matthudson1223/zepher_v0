"""
Volatility indicators and technical analysis calculations.

Provides various volatility metrics including historical volatility,
Bollinger Bands, ATR, and percentile rankings.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("btc_volatility.indicators")


@dataclass
class BollingerBands:
    """Container for Bollinger Bands values."""

    upper: pd.Series
    middle: pd.Series  # SMA
    lower: pd.Series
    bandwidth: pd.Series  # (upper - lower) / middle
    percent_b: pd.Series  # (price - lower) / (upper - lower)


@dataclass
class VolatilityMetrics:
    """Container for all volatility metrics."""

    historical_vol_7d: pd.Series
    historical_vol_30d: pd.Series
    historical_vol_90d: pd.Series
    atr: pd.Series
    bollinger: BollingerBands
    vol_percentile: pd.Series


class VolatilityCalculator:
    """
    Calculator for volatility metrics and technical indicators.

    Computes historical volatility, Bollinger Bands, ATR, and volatility
    percentile rankings for price data.

    Attributes:
        annualization_factor: Factor for annualizing volatility (365 for crypto).
        bb_period: Bollinger Bands period.
        bb_std: Bollinger Bands standard deviation multiplier.
        atr_period: ATR calculation period.

    Example:
        >>> calc = VolatilityCalculator()
        >>> metrics = calc.calculate_all(df)
        >>> print(metrics.historical_vol_30d.tail())
    """

    def __init__(
        self,
        annualization_factor: int = 365,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        percentile_lookback: int = 252,
    ):
        """
        Initialize the volatility calculator.

        Args:
            annualization_factor: Days per year for annualization (365 for crypto).
            bb_period: Period for Bollinger Bands SMA.
            bb_std: Standard deviation multiplier for Bollinger Bands.
            atr_period: Period for ATR calculation.
            percentile_lookback: Lookback period for percentile ranking.
        """
        self.annualization_factor = annualization_factor
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.percentile_lookback = percentile_lookback

        logger.debug(
            f"VolatilityCalculator initialized: "
            f"ann_factor={annualization_factor}, bb_period={bb_period}, "
            f"bb_std={bb_std}, atr_period={atr_period}"
        )

    def historical_volatility(
        self,
        prices: pd.Series,
        window: int,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Calculate historical (realized) volatility using log returns.

        Historical volatility is the standard deviation of log returns,
        optionally annualized.

        Args:
            prices: Series of closing prices.
            window: Rolling window size in periods.
            annualize: Whether to annualize the volatility.

        Returns:
            Series of rolling volatility values.

        Formula:
            vol = std(ln(P_t / P_{t-1})) * sqrt(annualization_factor)

        Example:
            >>> vol_30d = calc.historical_volatility(df["close"], window=30)
        """
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        # Calculate rolling standard deviation
        rolling_vol = log_returns.rolling(window=window).std()

        # Annualize if requested
        if annualize:
            rolling_vol = rolling_vol * np.sqrt(self.annualization_factor)

        return rolling_vol

    def bollinger_bands(
        self,
        prices: pd.Series,
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> BollingerBands:
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) and upper/lower bands
        at a specified number of standard deviations.

        Args:
            prices: Series of closing prices.
            period: SMA period. If None, uses instance default.
            std_dev: Standard deviation multiplier. If None, uses instance default.

        Returns:
            BollingerBands dataclass with upper, middle, lower, bandwidth, percent_b.

        Example:
            >>> bb = calc.bollinger_bands(df["close"])
            >>> print(bb.upper.tail())
        """
        period = period or self.bb_period
        std_dev = std_dev or self.bb_std

        # Middle band (SMA)
        middle = prices.rolling(window=period).mean()

        # Standard deviation
        rolling_std = prices.rolling(window=period).std()

        # Upper and lower bands
        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)

        # Bandwidth: (upper - lower) / middle
        # Measures the width of the bands relative to the middle band
        bandwidth = (upper - lower) / middle

        # %B: (price - lower) / (upper - lower)
        # Shows where price is relative to the bands (0 = at lower, 1 = at upper)
        percent_b = (prices - lower) / (upper - lower)

        return BollingerBands(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
        )

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range
        of an asset price for the period.

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of closing prices.
            period: ATR period. If None, uses instance default.

        Returns:
            Series of ATR values.

        Formula:
            TR = max(high - low, |high - prev_close|, |low - prev_close|)
            ATR = EMA(TR, period)

        Example:
            >>> atr_values = calc.atr(df["high"], df["low"], df["close"])
        """
        period = period or self.atr_period

        # Previous close
        prev_close = close.shift(1)

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the exponential moving average of True Range
        atr_values = true_range.ewm(span=period, adjust=False).mean()

        return atr_values

    def volatility_percentile(
        self,
        volatility: pd.Series,
        lookback: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate the percentile ranking of current volatility.

        Shows where current volatility stands relative to historical values.
        Useful for identifying volatility extremes.

        Args:
            volatility: Series of volatility values.
            lookback: Lookback period for percentile calculation.

        Returns:
            Series of percentile values (0-100).

        Example:
            >>> vol_pct = calc.volatility_percentile(vol_30d, lookback=252)
            >>> # 95 means current vol is higher than 95% of past year
        """
        lookback = lookback or self.percentile_lookback

        def rolling_percentile(x):
            """Calculate percentile of last value within the window."""
            if len(x) < 2:
                return np.nan
            return (x[:-1] < x.iloc[-1]).sum() / (len(x) - 1) * 100

        percentile = volatility.rolling(window=lookback).apply(
            rolling_percentile, raw=False
        )

        return percentile

    def calculate_all(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        windows: Optional[dict[str, int]] = None,
    ) -> VolatilityMetrics:
        """
        Calculate all volatility metrics at once.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume).
            price_col: Column name for closing prices.
            windows: Dictionary of window sizes {"short": 7, "medium": 30, "long": 90}.

        Returns:
            VolatilityMetrics dataclass with all calculated metrics.

        Example:
            >>> metrics = calc.calculate_all(df)
            >>> df["vol_30d"] = metrics.historical_vol_30d
            >>> df["bb_upper"] = metrics.bollinger.upper
        """
        if windows is None:
            windows = {"short": 7, "medium": 30, "long": 90}

        prices = df[price_col]

        logger.info(f"Calculating volatility metrics for {len(df)} data points")

        # Historical volatility for different windows
        vol_7d = self.historical_volatility(prices, windows["short"])
        vol_30d = self.historical_volatility(prices, windows["medium"])
        vol_90d = self.historical_volatility(prices, windows["long"])

        # ATR
        atr_values = self.atr(df["high"], df["low"], df["close"])

        # Bollinger Bands
        bb = self.bollinger_bands(prices)

        # Volatility percentile (based on 30-day vol)
        vol_pct = self.volatility_percentile(vol_30d)

        logger.info("Volatility metrics calculation complete")

        return VolatilityMetrics(
            historical_vol_7d=vol_7d,
            historical_vol_30d=vol_30d,
            historical_vol_90d=vol_90d,
            atr=atr_values,
            bollinger=bb,
            vol_percentile=vol_pct,
        )

    def add_indicators_to_df(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Add all volatility indicators as columns to a DataFrame.

        Args:
            df: DataFrame with OHLCV data.
            inplace: If True, modify df in place. If False, return a copy.

        Returns:
            DataFrame with added indicator columns.

        Example:
            >>> df_with_indicators = calc.add_indicators_to_df(df)
            >>> print(df_with_indicators.columns)
        """
        if not inplace:
            df = df.copy()

        metrics = self.calculate_all(df)

        # Historical volatility
        df["vol_7d"] = metrics.historical_vol_7d
        df["vol_30d"] = metrics.historical_vol_30d
        df["vol_90d"] = metrics.historical_vol_90d

        # ATR
        df["atr"] = metrics.atr

        # Bollinger Bands
        df["bb_upper"] = metrics.bollinger.upper
        df["bb_middle"] = metrics.bollinger.middle
        df["bb_lower"] = metrics.bollinger.lower
        df["bb_bandwidth"] = metrics.bollinger.bandwidth
        df["bb_percent_b"] = metrics.bollinger.percent_b

        # Volatility percentile
        df["vol_percentile"] = metrics.vol_percentile

        return df


def calculate_30d_volatility(prices: pd.Series) -> pd.Series:
    """
    Convenience function to calculate 30-day historical volatility.

    Args:
        prices: Series of closing prices.

    Returns:
        Series of annualized 30-day volatility values.

    Example:
        >>> vol = calculate_30d_volatility(df["close"])
        >>> print(f"Current volatility: {vol.iloc[-1]:.2%}")
    """
    calc = VolatilityCalculator()
    return calc.historical_volatility(prices, window=30, annualize=True)

"""
Volatility Compression Trading Strategy

This strategy identifies periods of volatility compression (squeeze) and trades
the anticipated expansion that typically follows such periods.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from .base import Strategy, Signal, SignalType


class VolatilityCompressionStrategy(Strategy):
    """
    Volatility Compression Strategy

    This strategy detects when volatility is compressing (Bollinger Bands narrowing)
    and positions for the expansion that typically follows.

    Entry Conditions:
    - Bollinger Band width at multi-period lows
    - Volatility trending down for multiple periods
    - Price consolidating within narrow range

    Exit Conditions:
    - Volatility expands beyond threshold
    - Profit target reached
    - Maximum holding period exceeded
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Volatility Compression Strategy"""
        super().__init__("VolatilityCompression", config)

        # Strategy-specific parameters
        self.compression_period = config.get('compression_period', 14)  # Look for 14-period squeeze
        self.bb_width_percentile = config.get('bb_width_percentile', 10)  # BB width in bottom 10%
        self.expansion_threshold = config.get('expansion_threshold', 1.5)  # Exit when BB expands 50%
        self.max_holding_period = config.get('max_holding_period', 20)  # Max periods to hold
        self.vol_decline_periods = config.get('vol_decline_periods', 5)  # Periods of declining vol
        self.price_range_threshold = config.get('price_range_threshold', 0.02)  # 2% price range

        # Track compression state
        self.compression_start = None
        self.entry_bb_width = None
        self.holding_periods = 0

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate signals for entire DataFrame"""
        signals = []

        # Validate required columns
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent_b', 'bb_bandwidth',
            'vol_percentile', 'vol_7d', 'vol_30d', 'atr'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing indicators for compression: {missing_columns}")
            return signals

        # Calculate additional metrics for compression detection
        df['bb_width_sma'] = df['bb_bandwidth'].rolling(window=self.compression_period).mean()
        df['bb_width_min'] = df['bb_bandwidth'].rolling(window=self.compression_period).min()
        df['vol_7d_change'] = df['vol_7d'].pct_change(periods=self.vol_decline_periods)
        df['price_range'] = (df['high'].rolling(window=self.compression_period).max() -
                            df['low'].rolling(window=self.compression_period).min()) / df['close']

        # Calculate BB width percentile
        lookback = min(100, len(df))
        df['bb_width_percentile'] = df['bb_bandwidth'].rolling(window=lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
        )

        # Process each row
        for idx, row in df.iterrows():
            # Skip if not enough data
            if pd.isna(row.get('bb_width_sma')) or pd.isna(row.get('bb_width_percentile')):
                continue

            # Check for signals based on current position
            if self.position is None:
                # Look for entry signals
                entry_signal = self.check_entry(row)
                if entry_signal:
                    signals.append(entry_signal)
                    # Determine position direction based on price position
                    if row['bb_percent_b'] >= 0.5:
                        self.position = 'long'  # Price in upper half, expect upward expansion
                    else:
                        self.position = 'short'  # Price in lower half, expect downward expansion
                    self.compression_start = pd.Timestamp(row.name)
                    self.entry_bb_width = row['bb_bandwidth']
                    self.holding_periods = 0
            else:
                # Increment holding period
                self.holding_periods += 1

                # Look for exit signals
                exit_signal = self.check_exit(row)
                if exit_signal:
                    signals.append(exit_signal)
                    self.position = None
                    self.compression_start = None
                    self.entry_bb_width = None
                    self.holding_periods = 0

        return self.filter_conflicting_signals(signals)

    def check_entry(self, row: pd.Series) -> Optional[Signal]:
        """Check for compression entry conditions"""
        timestamp = pd.Timestamp(row.name) if isinstance(row.name, (int, str, pd.Timestamp)) else datetime.now()

        # Check if BB width is at multi-period lows
        if row['bb_width_percentile'] > self.bb_width_percentile:
            return None

        # Check if volatility is declining
        vol_declining = False
        if not pd.isna(row.get('vol_7d_change')):
            if row['vol_7d_change'] < -0.1:  # 10% decline in 7-day vol
                vol_declining = True

        # Check if price is consolidating
        price_consolidating = False
        if not pd.isna(row.get('price_range')):
            if row['price_range'] < self.price_range_threshold:
                price_consolidating = True

        # Need at least 2 of 3 conditions beyond the BB width condition
        conditions_met = sum([vol_declining, price_consolidating, row['bb_bandwidth'] < row['bb_width_sma'] * 0.8])

        if conditions_met >= 2:
            # Determine signal direction based on price position and momentum
            signal_type = self._determine_compression_direction(row)
            strength = self._calculate_compression_strength(row, vol_declining, price_consolidating)

            if strength >= self.min_strength and signal_type:
                indicators = {
                    'bb_bandwidth': row['bb_bandwidth'],
                    'bb_width_percentile': row['bb_width_percentile'],
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_7d': row['vol_7d'],
                    'vol_percentile': row['vol_percentile'],
                    'price_range': row.get('price_range', 0)
                }

                conditions = [
                    f"BB squeeze detected (width={row['bb_bandwidth']:.4f})",
                    f"Width at {row['bb_width_percentile']:.0f} percentile",
                ]
                if vol_declining:
                    conditions.append(f"Volatility declining ({row['vol_7d_change']:.1%})")
                if price_consolidating:
                    conditions.append(f"Price consolidating (range={row['price_range']:.1%})")

                metadata = {
                    'compression_type': 'squeeze',
                    'expected_direction': 'long' if signal_type == SignalType.LONG else 'short'
                }

                return Signal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strategy_name=self.name,
                    strength=strength,
                    price=row['close'],
                    reason=self.format_reason(conditions),
                    indicators=indicators,
                    metadata=metadata
                )

        return None

    def check_exit(self, row: pd.Series) -> Optional[Signal]:
        """Check for exit conditions"""
        timestamp = pd.Timestamp(row.name) if isinstance(row.name, (int, str, pd.Timestamp)) else datetime.now()

        exit_conditions = []

        # Primary exit: Volatility expanded
        if self.entry_bb_width and row['bb_bandwidth'] > self.entry_bb_width * self.expansion_threshold:
            exit_conditions.append(
                f"Volatility expanded ({row['bb_bandwidth']/self.entry_bb_width:.1f}x entry)"
            )

        # Time-based exit: Maximum holding period exceeded
        if self.holding_periods >= self.max_holding_period:
            exit_conditions.append(f"Max holding period reached ({self.holding_periods} periods)")

        # Profit exits based on position
        if self.position == 'long':
            # Exit if price breaks strongly above upper band
            if row['bb_percent_b'] > 1.2:
                exit_conditions.append(f"Strong breakout above bands (%B={row['bb_percent_b']:.2f})")

            # Exit if volatility regime becomes high (good for mean reversion, not expansion)
            if row['vol_percentile'] > 80:
                exit_conditions.append(f"High volatility regime ({row['vol_percentile']:.0f}%ile)")

            if exit_conditions:
                indicators = {
                    'bb_bandwidth': row['bb_bandwidth'],
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'expansion_ratio': row['bb_bandwidth'] / self.entry_bb_width if self.entry_bb_width else 1
                }

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT_LONG,
                    strategy_name=self.name,
                    strength=75.0,
                    price=row['close'],
                    reason=self.format_reason(exit_conditions),
                    indicators=indicators
                )

        elif self.position == 'short':
            # Exit if price breaks strongly below lower band
            if row['bb_percent_b'] < -0.2:
                exit_conditions.append(f"Strong breakout below bands (%B={row['bb_percent_b']:.2f})")

            # Exit if volatility regime becomes high
            if row['vol_percentile'] > 80:
                exit_conditions.append(f"High volatility regime ({row['vol_percentile']:.0f}%ile)")

            if exit_conditions:
                indicators = {
                    'bb_bandwidth': row['bb_bandwidth'],
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'expansion_ratio': row['bb_bandwidth'] / self.entry_bb_width if self.entry_bb_width else 1
                }

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT_SHORT,
                    strategy_name=self.name,
                    strength=75.0,
                    price=row['close'],
                    reason=self.format_reason(exit_conditions),
                    indicators=indicators
                )

        return None

    def _determine_compression_direction(self, row: pd.Series) -> Optional[SignalType]:
        """
        Determine the likely direction of expansion based on price position
        and other factors during compression
        """
        # Simple heuristic: Trade in direction of price position within bands
        # More sophisticated versions could use:
        # - Trend before compression
        # - Volume patterns
        # - Market structure

        if row['bb_percent_b'] >= 0.6:
            # Price in upper portion of bands, expect upward expansion
            return SignalType.LONG
        elif row['bb_percent_b'] <= 0.4:
            # Price in lower portion of bands, expect downward expansion
            return SignalType.SHORT
        else:
            # Price near middle, check other factors
            # For now, slight bias towards long
            if row['close'] > row['bb_middle']:
                return SignalType.LONG
            else:
                return SignalType.SHORT

    def _calculate_compression_strength(self, row: pd.Series, vol_declining: bool, price_consolidating: bool) -> float:
        """Calculate signal strength for compression setup"""
        strength = 0.0

        # Base strength from BB width percentile
        if row['bb_width_percentile'] <= 5:
            strength += 35  # Extreme compression
        elif row['bb_width_percentile'] <= 10:
            strength += 25
        else:
            strength += 15

        # Add strength from BB width relative to average
        if row['bb_bandwidth'] < row['bb_width_sma'] * 0.5:
            strength += 20  # Width less than half of average
        elif row['bb_bandwidth'] < row['bb_width_sma'] * 0.7:
            strength += 10

        # Add strength from volatility decline
        if vol_declining:
            strength += 15

        # Add strength from price consolidation
        if price_consolidating:
            strength += 15

        # Add strength from low overall volatility
        if row['vol_percentile'] < 30:
            strength += 15

        return min(100, strength)
"""
Mean Reversion Trading Strategy

This strategy trades based on the assumption that prices will revert to the mean
after extreme moves, especially during high volatility regimes.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from .base import Strategy, Signal, SignalType


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy

    Entry Conditions:
    - Price touches or exceeds Bollinger Bands (bb_percent_b > 1 or < 0)
    - High volatility regime (vol_percentile >= 80)

    Exit Conditions:
    - Price returns to middle band (bb_percent_b near 0.5)
    - Or opposite band is touched
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Mean Reversion Strategy"""
        super().__init__("MeanReversion", config)

        # Strategy-specific parameters
        self.bb_upper_threshold = config.get('bb_upper_threshold', 1.0)  # %B for short entry
        self.bb_lower_threshold = config.get('bb_lower_threshold', 0.0)  # %B for long entry
        self.exit_target = config.get('exit_target', 0.5)  # %B target for exit
        self.exit_tolerance = config.get('exit_tolerance', 0.1)  # Tolerance around exit target
        self.vol_percentile_min = config.get('vol_percentile_min', 80)  # Min volatility for trades
        self.min_bb_width = config.get('min_bb_width', 0.02)  # Minimum BB width to avoid false signals

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate signals for entire DataFrame"""
        signals = []

        # Validate required columns
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent_b', 'bb_bandwidth',
            'vol_percentile', 'atr'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing indicators for mean reversion: {missing_columns}")
            return signals

        # Process each row
        for idx, row in df.iterrows():
            # Skip if not enough data
            if pd.isna(row['bb_percent_b']) or pd.isna(row['vol_percentile']):
                continue

            # Check for signals based on current position
            if self.position is None:
                # Look for entry signals
                entry_signal = self.check_entry(row)
                if entry_signal:
                    signals.append(entry_signal)
                    # Update position
                    if entry_signal.signal_type == SignalType.LONG:
                        self.position = 'long'
                    elif entry_signal.signal_type == SignalType.SHORT:
                        self.position = 'short'
            else:
                # Look for exit signals
                exit_signal = self.check_exit(row)
                if exit_signal:
                    signals.append(exit_signal)
                    self.position = None

        return self.filter_conflicting_signals(signals)

    def check_entry(self, row: pd.Series) -> Optional[Signal]:
        """Check for entry conditions"""
        timestamp = pd.Timestamp(row.name) if isinstance(row.name, (int, str, pd.Timestamp)) else datetime.now()
        vol_regime = self.get_volatility_regime(row['vol_percentile'])

        # Only trade in high volatility regimes
        if row['vol_percentile'] < self.vol_percentile_min:
            return None

        # Check for minimum Bollinger Band width to avoid false signals
        if row['bb_bandwidth'] < self.min_bb_width:
            return None

        # Long entry: Price at or below lower band
        if row['bb_percent_b'] <= self.bb_lower_threshold:
            # Calculate signal strength
            strength = self._calculate_long_strength(row)

            if strength >= self.min_strength:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'bb_bandwidth': row['bb_bandwidth'],
                    'atr': row['atr']
                }

                conditions = [
                    f"Price at lower band (%B={row['bb_percent_b']:.2f})",
                    f"{vol_regime} volatility regime ({row['vol_percentile']:.0f}%ile)",
                    f"BB width sufficient ({row['bb_bandwidth']:.3f})"
                ]

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.LONG,
                    strategy_name=self.name,
                    strength=strength,
                    price=row['close'],
                    reason=self.format_reason(conditions),
                    indicators=indicators,
                    metadata={'vol_regime': vol_regime}
                )

        # Short entry: Price at or above upper band
        elif row['bb_percent_b'] >= self.bb_upper_threshold:
            # Calculate signal strength
            strength = self._calculate_short_strength(row)

            if strength >= self.min_strength:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'bb_bandwidth': row['bb_bandwidth'],
                    'atr': row['atr']
                }

                conditions = [
                    f"Price at upper band (%B={row['bb_percent_b']:.2f})",
                    f"{vol_regime} volatility regime ({row['vol_percentile']:.0f}%ile)",
                    f"BB width sufficient ({row['bb_bandwidth']:.3f})"
                ]

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SHORT,
                    strategy_name=self.name,
                    strength=strength,
                    price=row['close'],
                    reason=self.format_reason(conditions),
                    indicators=indicators,
                    metadata={'vol_regime': vol_regime}
                )

        return None

    def check_exit(self, row: pd.Series) -> Optional[Signal]:
        """Check for exit conditions"""
        timestamp = pd.Timestamp(row.name) if isinstance(row.name, (int, str, pd.Timestamp)) else datetime.now()

        if self.position == 'long':
            # Exit long position
            exit_conditions = []

            # Primary exit: Price returned to middle band
            if abs(row['bb_percent_b'] - self.exit_target) <= self.exit_tolerance:
                exit_conditions.append(f"Price at middle band (%B={row['bb_percent_b']:.2f})")

            # Secondary exit: Price hit upper band (take profit)
            elif row['bb_percent_b'] >= self.bb_upper_threshold:
                exit_conditions.append(f"Price at upper band (%B={row['bb_percent_b']:.2f})")

            # Risk exit: Volatility regime changed to low
            elif row['vol_percentile'] < 50:
                exit_conditions.append(f"Volatility regime changed ({row['vol_percentile']:.0f}%ile)")

            if exit_conditions:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile']
                }

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT_LONG,
                    strategy_name=self.name,
                    strength=75.0,  # Exit signals typically have high confidence
                    price=row['close'],
                    reason=self.format_reason(exit_conditions),
                    indicators=indicators
                )

        elif self.position == 'short':
            # Exit short position
            exit_conditions = []

            # Primary exit: Price returned to middle band
            if abs(row['bb_percent_b'] - self.exit_target) <= self.exit_tolerance:
                exit_conditions.append(f"Price at middle band (%B={row['bb_percent_b']:.2f})")

            # Secondary exit: Price hit lower band (take profit)
            elif row['bb_percent_b'] <= self.bb_lower_threshold:
                exit_conditions.append(f"Price at lower band (%B={row['bb_percent_b']:.2f})")

            # Risk exit: Volatility regime changed to low
            elif row['vol_percentile'] < 50:
                exit_conditions.append(f"Volatility regime changed ({row['vol_percentile']:.0f}%ile)")

            if exit_conditions:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile']
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

    def _calculate_long_strength(self, row: pd.Series) -> float:
        """Calculate signal strength for long entry"""
        strength = 0.0

        # Base strength from how far below lower band
        if row['bb_percent_b'] < 0:
            # Stronger signal the further below 0
            strength += min(30, abs(row['bb_percent_b']) * 30)
        else:
            strength += (1 - row['bb_percent_b']) * 20

        # Add strength from high volatility
        if row['vol_percentile'] >= 90:
            strength += 30
        elif row['vol_percentile'] >= 80:
            strength += 20
        else:
            strength += 10

        # Add strength from wide Bollinger Bands
        if row['bb_bandwidth'] > 0.05:
            strength += 20
        elif row['bb_bandwidth'] > 0.03:
            strength += 10

        # Add strength from ATR (higher ATR = more volatile = better for mean reversion)
        if 'atr' in row and not pd.isna(row['atr']):
            if row['atr'] > row['close'] * 0.03:  # ATR > 3% of price
                strength += 20

        return min(100, strength)

    def _calculate_short_strength(self, row: pd.Series) -> float:
        """Calculate signal strength for short entry"""
        strength = 0.0

        # Base strength from how far above upper band
        if row['bb_percent_b'] > 1:
            # Stronger signal the further above 1
            strength += min(30, (row['bb_percent_b'] - 1) * 30)
        else:
            strength += row['bb_percent_b'] * 20

        # Add strength from high volatility
        if row['vol_percentile'] >= 90:
            strength += 30
        elif row['vol_percentile'] >= 80:
            strength += 20
        else:
            strength += 10

        # Add strength from wide Bollinger Bands
        if row['bb_bandwidth'] > 0.05:
            strength += 20
        elif row['bb_bandwidth'] > 0.03:
            strength += 10

        # Add strength from ATR
        if 'atr' in row and not pd.isna(row['atr']):
            if row['atr'] > row['close'] * 0.03:  # ATR > 3% of price
                strength += 20

        return min(100, strength)
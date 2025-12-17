"""
Breakout Trading Strategy

This strategy trades breakouts from periods of low volatility/consolidation,
anticipating strong directional moves after volatility compression.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from .base import Strategy, Signal, SignalType


class BreakoutStrategy(Strategy):
    """
    Breakout Strategy

    Entry Conditions:
    - Low volatility regime (vol_percentile <= 20)
    - Bollinger Band squeeze detected (narrow bandwidth)
    - Price breaks above upper band or below lower band
    - Volume confirmation (spike above average)
    - ATR expansion confirmation

    Exit Conditions:
    - Fixed profit target based on ATR
    - Volatility returns to normal levels
    - Opposite signal generated
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Breakout Strategy"""
        super().__init__("Breakout", config)

        # Strategy-specific parameters
        self.vol_percentile_max = config.get('vol_percentile_max', 20)  # Max volatility for setup
        self.bb_squeeze_threshold = config.get('bb_squeeze_threshold', 0.03)  # BB width for squeeze
        self.volume_multiplier = config.get('volume_multiplier', 1.5)  # Volume spike threshold
        self.atr_expansion = config.get('atr_expansion', 1.2)  # ATR must expand by 20%
        self.profit_target_atr = config.get('profit_target_atr', 2.0)  # Take profit at 2x ATR
        self.stop_loss_atr = config.get('stop_loss_atr', 1.0)  # Stop loss at 1x ATR
        self.lookback_period = config.get('lookback_period', 20)  # Periods to look back for squeeze

        # Track entry price and ATR for exits
        self.entry_price = None
        self.entry_atr = None

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
            print(f"Warning: Missing indicators for breakout: {missing_columns}")
            return signals

        # Calculate rolling volume average for comparison
        df['volume_sma'] = df['volume'].rolling(window=self.lookback_period).mean()
        df['atr_sma'] = df['atr'].rolling(window=self.lookback_period).mean()

        # Process each row
        for idx, row in df.iterrows():
            # Skip if not enough data
            if pd.isna(row.get('volume_sma')) or pd.isna(row['bb_percent_b']):
                continue

            # Check for signals based on current position
            if self.position is None:
                # Look for entry signals
                entry_signal = self.check_entry(row)
                if entry_signal:
                    signals.append(entry_signal)
                    # Update position and entry info
                    if entry_signal.signal_type == SignalType.LONG:
                        self.position = 'long'
                    elif entry_signal.signal_type == SignalType.SHORT:
                        self.position = 'short'
                    self.entry_price = row['close']
                    self.entry_atr = row['atr']
            else:
                # Look for exit signals
                exit_signal = self.check_exit(row)
                if exit_signal:
                    signals.append(exit_signal)
                    self.position = None
                    self.entry_price = None
                    self.entry_atr = None

        return self.filter_conflicting_signals(signals)

    def check_entry(self, row: pd.Series) -> Optional[Signal]:
        """Check for entry conditions"""
        timestamp = pd.Timestamp(row.name) if isinstance(row.name, (int, str, pd.Timestamp)) else datetime.now()
        vol_regime = self.get_volatility_regime(row['vol_percentile'])

        # Only trade in low volatility regimes (looking for breakout)
        if row['vol_percentile'] > self.vol_percentile_max:
            return None

        # Check for Bollinger Band squeeze
        if row['bb_bandwidth'] > self.bb_squeeze_threshold:
            return None  # No squeeze detected

        # Check volume spike
        volume_spike = False
        if 'volume_sma' in row and not pd.isna(row['volume_sma']):
            if row['volume'] > row['volume_sma'] * self.volume_multiplier:
                volume_spike = True

        # Check ATR expansion
        atr_expanding = False
        if 'atr_sma' in row and not pd.isna(row['atr_sma']):
            if row['atr'] > row['atr_sma'] * self.atr_expansion:
                atr_expanding = True

        # Long breakout: Price breaks above upper band with confirmation
        if row['bb_percent_b'] > 1.0 and (volume_spike or atr_expanding):
            strength = self._calculate_long_breakout_strength(row, volume_spike, atr_expanding)

            if strength >= self.min_strength:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'bb_bandwidth': row['bb_bandwidth'],
                    'atr': row['atr'],
                    'volume_ratio': row['volume'] / row.get('volume_sma', row['volume'])
                }

                conditions = [
                    f"Breakout above upper band (%B={row['bb_percent_b']:.2f})",
                    f"LOW volatility squeeze ({row['bb_bandwidth']:.3f})",
                ]
                if volume_spike:
                    conditions.append(f"Volume spike ({row['volume']/row['volume_sma']:.1f}x avg)")
                if atr_expanding:
                    conditions.append(f"ATR expansion ({row['atr']/row['atr_sma']:.1f}x avg)")

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.LONG,
                    strategy_name=self.name,
                    strength=strength,
                    price=row['close'],
                    reason=self.format_reason(conditions),
                    indicators=indicators,
                    metadata={'vol_regime': vol_regime, 'breakout_type': 'bullish'}
                )

        # Short breakout: Price breaks below lower band with confirmation
        elif row['bb_percent_b'] < 0.0 and (volume_spike or atr_expanding):
            strength = self._calculate_short_breakout_strength(row, volume_spike, atr_expanding)

            if strength >= self.min_strength:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'bb_bandwidth': row['bb_bandwidth'],
                    'atr': row['atr'],
                    'volume_ratio': row['volume'] / row.get('volume_sma', row['volume'])
                }

                conditions = [
                    f"Breakout below lower band (%B={row['bb_percent_b']:.2f})",
                    f"LOW volatility squeeze ({row['bb_bandwidth']:.3f})",
                ]
                if volume_spike:
                    conditions.append(f"Volume spike ({row['volume']/row['volume_sma']:.1f}x avg)")
                if atr_expanding:
                    conditions.append(f"ATR expansion ({row['atr']/row['atr_sma']:.1f}x avg)")

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SHORT,
                    strategy_name=self.name,
                    strength=strength,
                    price=row['close'],
                    reason=self.format_reason(conditions),
                    indicators=indicators,
                    metadata={'vol_regime': vol_regime, 'breakout_type': 'bearish'}
                )

        return None

    def check_exit(self, row: pd.Series) -> Optional[Signal]:
        """Check for exit conditions"""
        timestamp = pd.Timestamp(row.name) if isinstance(row.name, (int, str, pd.Timestamp)) else datetime.now()

        if self.position == 'long' and self.entry_price and self.entry_atr:
            exit_conditions = []

            # Take profit: Price moved up by target ATR multiple
            profit_target = self.entry_price + (self.entry_atr * self.profit_target_atr)
            if row['close'] >= profit_target:
                exit_conditions.append(f"Profit target reached (${row['close']:.2f} >= ${profit_target:.2f})")

            # Stop loss: Price moved down by stop ATR multiple
            stop_loss = self.entry_price - (self.entry_atr * self.stop_loss_atr)
            if row['close'] <= stop_loss:
                exit_conditions.append(f"Stop loss triggered (${row['close']:.2f} <= ${stop_loss:.2f})")

            # Volatility exit: Volatility returned to normal/high levels
            if row['vol_percentile'] > 50:
                exit_conditions.append(f"Volatility normalized ({row['vol_percentile']:.0f}%ile)")

            # Reversal exit: Price returned inside bands
            if 0.2 < row['bb_percent_b'] < 0.8:
                exit_conditions.append(f"Price returned to bands (%B={row['bb_percent_b']:.2f})")

            if exit_conditions:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'profit_pct': ((row['close'] - self.entry_price) / self.entry_price) * 100
                }

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT_LONG,
                    strategy_name=self.name,
                    strength=80.0,
                    price=row['close'],
                    reason=self.format_reason(exit_conditions),
                    indicators=indicators
                )

        elif self.position == 'short' and self.entry_price and self.entry_atr:
            exit_conditions = []

            # Take profit: Price moved down by target ATR multiple
            profit_target = self.entry_price - (self.entry_atr * self.profit_target_atr)
            if row['close'] <= profit_target:
                exit_conditions.append(f"Profit target reached (${row['close']:.2f} <= ${profit_target:.2f})")

            # Stop loss: Price moved up by stop ATR multiple
            stop_loss = self.entry_price + (self.entry_atr * self.stop_loss_atr)
            if row['close'] >= stop_loss:
                exit_conditions.append(f"Stop loss triggered (${row['close']:.2f} >= ${stop_loss:.2f})")

            # Volatility exit: Volatility returned to normal/high levels
            if row['vol_percentile'] > 50:
                exit_conditions.append(f"Volatility normalized ({row['vol_percentile']:.0f}%ile)")

            # Reversal exit: Price returned inside bands
            if 0.2 < row['bb_percent_b'] < 0.8:
                exit_conditions.append(f"Price returned to bands (%B={row['bb_percent_b']:.2f})")

            if exit_conditions:
                indicators = {
                    'bb_percent_b': row['bb_percent_b'],
                    'vol_percentile': row['vol_percentile'],
                    'profit_pct': ((self.entry_price - row['close']) / self.entry_price) * 100
                }

                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT_SHORT,
                    strategy_name=self.name,
                    strength=80.0,
                    price=row['close'],
                    reason=self.format_reason(exit_conditions),
                    indicators=indicators
                )

        return None

    def _calculate_long_breakout_strength(self, row: pd.Series, volume_spike: bool, atr_expanding: bool) -> float:
        """Calculate signal strength for long breakout"""
        strength = 0.0

        # Base strength from breakout magnitude
        if row['bb_percent_b'] > 1.2:
            strength += 30
        elif row['bb_percent_b'] > 1.1:
            strength += 20
        else:
            strength += 10

        # Add strength from squeeze tightness
        if row['bb_bandwidth'] < 0.01:
            strength += 25  # Very tight squeeze
        elif row['bb_bandwidth'] < 0.02:
            strength += 15
        else:
            strength += 5

        # Add strength from low volatility percentile
        if row['vol_percentile'] <= 10:
            strength += 25
        elif row['vol_percentile'] <= 20:
            strength += 15

        # Add strength from confirmations
        if volume_spike:
            strength += 20
        if atr_expanding:
            strength += 20

        return min(100, strength)

    def _calculate_short_breakout_strength(self, row: pd.Series, volume_spike: bool, atr_expanding: bool) -> float:
        """Calculate signal strength for short breakout"""
        strength = 0.0

        # Base strength from breakout magnitude
        if row['bb_percent_b'] < -0.2:
            strength += 30
        elif row['bb_percent_b'] < -0.1:
            strength += 20
        else:
            strength += 10

        # Add strength from squeeze tightness
        if row['bb_bandwidth'] < 0.01:
            strength += 25  # Very tight squeeze
        elif row['bb_bandwidth'] < 0.02:
            strength += 15
        else:
            strength += 5

        # Add strength from low volatility percentile
        if row['vol_percentile'] <= 10:
            strength += 25
        elif row['vol_percentile'] <= 20:
            strength += 15

        # Add strength from confirmations
        if volume_spike:
            strength += 20
        if atr_expanding:
            strength += 20

        return min(100, strength)
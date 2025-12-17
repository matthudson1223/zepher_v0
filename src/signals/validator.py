"""
Signal Validation Module

Validates trading signals against market conditions and risk parameters.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
from ..strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


class SignalValidator:
    """
    Validate signals against market conditions and risk criteria
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Signal Validator

        Args:
            config: Validation configuration
        """
        self.config = config
        self.check_liquidity = config.get('check_liquidity', True)
        self.min_volume = config.get('min_volume', 100000)  # Minimum USD volume
        self.check_spread = config.get('check_spread', True)
        self.max_spread_percent = config.get('max_spread_percent', 0.5)  # 0.5% max spread
        self.check_market_hours = config.get('check_market_hours', False)
        self.validate_indicators = config.get('validate_indicators', True)
        self.max_position_size = config.get('max_position_size', 1)  # Max concurrent positions

    def validate_signals(self, signals: List[Signal], df: pd.DataFrame) -> List[Signal]:
        """
        Validate all signals against criteria

        Args:
            signals: List of signals to validate
            df: DataFrame with market data

        Returns:
            List of validated signals
        """
        if not signals:
            return signals

        validated = []
        for signal in signals:
            if self._validate_signal(signal, df):
                validated.append(signal)
            else:
                logger.debug(f"Signal validation failed: {signal}")

        logger.info(f"Validated {len(validated)} of {len(signals)} signals")
        return validated

    def _validate_signal(self, signal: Signal, df: pd.DataFrame) -> bool:
        """
        Validate individual signal

        Args:
            signal: Signal to validate
            df: DataFrame with market data

        Returns:
            True if signal is valid
        """
        # Get market data at signal timestamp
        market_data = self._get_market_data_at_time(signal.timestamp, df)
        if market_data is None:
            logger.warning(f"No market data for signal at {signal.timestamp}")
            return False

        # Check liquidity
        if self.check_liquidity and not self._validate_liquidity(signal, market_data):
            return False

        # Check spread
        if self.check_spread and not self._validate_spread(signal, market_data):
            return False

        # Check market hours
        if self.check_market_hours and not self._validate_market_hours(signal):
            return False

        # Validate indicators
        if self.validate_indicators and not self._validate_indicators(signal):
            return False

        # Check position limits
        if not self._validate_position_limits(signal, signals):
            return False

        return True

    def _get_market_data_at_time(self, timestamp: datetime, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Get market data row closest to signal timestamp

        Args:
            timestamp: Signal timestamp
            df: DataFrame with market data

        Returns:
            Market data row or None
        """
        # Convert timestamp to match dataframe index
        if isinstance(df.index, pd.DatetimeIndex):
            # Find closest timestamp
            closest_idx = df.index.get_indexer([pd.Timestamp(timestamp)], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(df):
                return df.iloc[closest_idx]
        else:
            # Try to match by position
            for idx, row in df.iterrows():
                if hasattr(row, 'timestamp') and row['timestamp'] == timestamp:
                    return row

        return None

    def _validate_liquidity(self, signal: Signal, market_data: pd.Series) -> bool:
        """
        Validate signal has sufficient liquidity

        Args:
            signal: Signal to validate
            market_data: Market data at signal time

        Returns:
            True if liquidity is sufficient
        """
        if 'volume' not in market_data or pd.isna(market_data['volume']):
            logger.debug(f"No volume data for signal at {signal.timestamp}")
            return True  # Pass if no data

        # Calculate USD volume (volume * price)
        usd_volume = market_data['volume'] * signal.price

        if usd_volume < self.min_volume:
            logger.debug(f"Insufficient liquidity: ${usd_volume:.0f} < ${self.min_volume}")
            return False

        return True

    def _validate_spread(self, signal: Signal, market_data: pd.Series) -> bool:
        """
        Validate bid-ask spread is acceptable

        Args:
            signal: Signal to validate
            market_data: Market data at signal time

        Returns:
            True if spread is acceptable
        """
        # Calculate approximate spread using high-low as proxy
        if 'high' in market_data and 'low' in market_data:
            high = market_data['high']
            low = market_data['low']

            if high > 0 and low > 0:
                spread_percent = ((high - low) / low) * 100

                if spread_percent > self.max_spread_percent:
                    logger.debug(f"Spread too wide: {spread_percent:.2f}% > {self.max_spread_percent}%")
                    return False

        return True

    def _validate_market_hours(self, signal: Signal) -> bool:
        """
        Validate signal occurs during market hours

        Args:
            signal: Signal to validate

        Returns:
            True if during market hours
        """
        # Crypto markets are 24/7, but could implement restrictions
        # For example, avoid low liquidity hours
        hour = signal.timestamp.hour

        # Example: Avoid 2-6 AM UTC (typically lowest liquidity)
        if 2 <= hour <= 6:
            logger.debug(f"Signal outside preferred hours: {hour}:00 UTC")
            # Could return False to filter, but for crypto we'll allow
            pass

        return True

    def _validate_indicators(self, signal: Signal) -> bool:
        """
        Validate signal indicators are within reasonable ranges

        Args:
            signal: Signal to validate

        Returns:
            True if indicators are valid
        """
        indicators = signal.indicators

        # Check for NaN values
        for key, value in indicators.items():
            if pd.isna(value):
                logger.debug(f"Invalid indicator {key}: NaN")
                return False

        # Validate specific indicator ranges
        if 'bb_percent_b' in indicators:
            bb_percent = indicators['bb_percent_b']
            # %B can be outside 0-1, but extreme values might be errors
            if bb_percent < -2 or bb_percent > 3:
                logger.debug(f"Suspicious bb_percent_b: {bb_percent}")
                return False

        if 'vol_percentile' in indicators:
            vol_pct = indicators['vol_percentile']
            if vol_pct < 0 or vol_pct > 100:
                logger.debug(f"Invalid vol_percentile: {vol_pct}")
                return False

        if 'bb_bandwidth' in indicators:
            bandwidth = indicators['bb_bandwidth']
            if bandwidth < 0 or bandwidth > 1:
                logger.debug(f"Invalid bb_bandwidth: {bandwidth}")
                return False

        return True

    def _validate_position_limits(self, signal: Signal, all_signals: List[Signal]) -> bool:
        """
        Validate position limits aren't exceeded

        Args:
            signal: Signal to validate
            all_signals: All signals for context

        Returns:
            True if position limits are respected
        """
        # This is a simplified check
        # In production, would track actual positions
        if signal.signal_type.is_entry():
            # Count current open positions
            # For now, we'll allow all signals
            pass

        return True

    def validate_exit_signal(self, signal: Signal, entry_signal: Optional[Signal]) -> bool:
        """
        Validate exit signal against entry

        Args:
            signal: Exit signal to validate
            entry_signal: Original entry signal

        Returns:
            True if exit is valid
        """
        if not signal.signal_type.is_exit():
            return True

        if entry_signal is None:
            logger.warning(f"Exit signal without entry: {signal}")
            return False

        # Validate exit matches entry type
        if signal.signal_type == SignalType.EXIT_LONG and entry_signal.signal_type != SignalType.LONG:
            logger.warning(f"EXIT_LONG without LONG entry")
            return False

        if signal.signal_type == SignalType.EXIT_SHORT and entry_signal.signal_type != SignalType.SHORT:
            logger.warning(f"EXIT_SHORT without SHORT entry")
            return False

        # Validate timing
        if signal.timestamp <= entry_signal.timestamp:
            logger.warning(f"Exit signal before entry")
            return False

        return True
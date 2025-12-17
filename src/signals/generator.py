"""
Signal Generation Engine

Coordinates multiple trading strategies to generate comprehensive trading signals.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging

from ..strategies import (
    MeanReversionStrategy,
    BreakoutStrategy,
    VolatilityCompressionStrategy,
    Strategy,
    SignalType
)
from ..strategies.base import Signal
from .filters import SignalFilter
from .validator import SignalValidator
from ..data.database import Database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Main signal generation engine that coordinates multiple strategies
    """

    def __init__(self, config: Dict[str, Any], database: Optional[Database] = None):
        """
        Initialize Signal Generator

        Args:
            config: Configuration dictionary with strategy settings
            database: Optional database instance for storing signals
        """
        self.config = config
        self.database = database
        self.strategies = self._initialize_strategies(config.get('strategies', {}))
        self.filter = SignalFilter(config.get('filters', {}))
        self.validator = SignalValidator(config.get('validation', {}))

    def _initialize_strategies(self, strategies_config: Dict[str, Any]) -> List[Strategy]:
        """Initialize enabled strategies"""
        strategies = []

        # Mean Reversion Strategy
        if strategies_config.get('mean_reversion', {}).get('enabled', True):
            strategies.append(MeanReversionStrategy(
                strategies_config.get('mean_reversion', {})
            ))
            logger.info("Mean Reversion strategy initialized")

        # Breakout Strategy
        if strategies_config.get('breakout', {}).get('enabled', True):
            strategies.append(BreakoutStrategy(
                strategies_config.get('breakout', {})
            ))
            logger.info("Breakout strategy initialized")

        # Volatility Compression Strategy
        if strategies_config.get('compression', {}).get('enabled', True):
            strategies.append(VolatilityCompressionStrategy(
                strategies_config.get('compression', {})
            ))
            logger.info("Volatility Compression strategy initialized")

        if not strategies:
            logger.warning("No strategies enabled!")

        return strategies

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Generate signals from all enabled strategies

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            List of validated and filtered signals
        """
        all_signals = []

        # Generate signals from each strategy
        for strategy in self.strategies:
            try:
                strategy_signals = strategy.generate_signals(df.copy())
                logger.info(f"{strategy.name}: Generated {len(strategy_signals)} signals")
                all_signals.extend(strategy_signals)
            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")

        # Apply filtering
        filtered_signals = self.filter.filter_signals(all_signals)
        logger.info(f"Filtered to {len(filtered_signals)} signals")

        # Validate signals
        validated_signals = self.validator.validate_signals(filtered_signals, df)
        logger.info(f"Validated {len(validated_signals)} signals")

        # Check for confluence (multiple strategies agreeing)
        final_signals = self._enhance_with_confluence(validated_signals)

        # Store signals in database if available
        if self.database:
            self._store_signals(final_signals)

        return final_signals

    def _enhance_with_confluence(self, signals: List[Signal]) -> List[Signal]:
        """
        Enhance signals with confluence information when multiple strategies agree

        Args:
            signals: List of signals to analyze

        Returns:
            Enhanced signals with confluence data
        """
        if len(signals) < 2:
            return signals

        # Group signals by timestamp and type
        from collections import defaultdict
        signal_groups = defaultdict(list)

        for signal in signals:
            key = (signal.timestamp, signal.signal_type)
            signal_groups[key].append(signal)

        enhanced_signals = []

        for (timestamp, signal_type), group_signals in signal_groups.items():
            if len(group_signals) > 1:
                # Multiple strategies agree - create enhanced signal
                # Use the strongest signal as base
                base_signal = max(group_signals, key=lambda x: x.strength)

                # Enhance strength based on confluence
                confluence_boost = min(20, len(group_signals) * 10)
                enhanced_strength = min(100, base_signal.strength + confluence_boost)

                # Combine reasons
                strategies_involved = [s.strategy_name for s in group_signals]
                combined_reason = f"CONFLUENCE: {', '.join(strategies_involved)} agree. " + base_signal.reason

                # Create enhanced signal
                enhanced_signal = Signal(
                    timestamp=base_signal.timestamp,
                    signal_type=base_signal.signal_type,
                    strategy_name=f"Confluence-{'-'.join(strategies_involved)}",
                    strength=enhanced_strength,
                    price=base_signal.price,
                    reason=combined_reason,
                    indicators=base_signal.indicators,
                    metadata={
                        **base_signal.metadata,
                        'confluence_count': len(group_signals),
                        'strategies': strategies_involved
                    }
                )
                enhanced_signals.append(enhanced_signal)
            else:
                # Single strategy signal
                enhanced_signals.extend(group_signals)

        return enhanced_signals

    def _store_signals(self, signals: List[Signal]):
        """Store signals in database"""
        try:
            for signal in signals:
                self.database.store_signal(signal.to_dict())
            logger.info(f"Stored {len(signals)} signals in database")
        except Exception as e:
            logger.error(f"Error storing signals: {e}")

    def get_latest_signals(self, df: pd.DataFrame, limit: int = 10) -> List[Signal]:
        """
        Get only the most recent signals

        Args:
            df: DataFrame with OHLCV data and indicators
            limit: Maximum number of signals to return

        Returns:
            List of most recent signals
        """
        all_signals = self.generate_signals(df)

        # Sort by timestamp descending and return limit
        sorted_signals = sorted(all_signals, key=lambda x: x.timestamp, reverse=True)
        return sorted_signals[:limit]

    def get_active_positions(self) -> Dict[str, Any]:
        """
        Get currently active positions based on signals

        Returns:
            Dictionary of active positions by strategy
        """
        positions = {}
        for strategy in self.strategies:
            if strategy.position:
                positions[strategy.name] = {
                    'position': strategy.position,
                    'entry_price': getattr(strategy, 'entry_price', None),
                    'entry_time': getattr(strategy, 'entry_time', None)
                }
        return positions

    def reset_positions(self):
        """Reset all strategy positions"""
        for strategy in self.strategies:
            strategy.position = None
            if hasattr(strategy, 'entry_price'):
                strategy.entry_price = None
            if hasattr(strategy, 'entry_atr'):
                strategy.entry_atr = None
        logger.info("All positions reset")

    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for each strategy

        Returns:
            Dictionary of performance metrics
        """
        if not self.database:
            return {}

        performance = {}
        for strategy in self.strategies:
            # This would query the database for historical performance
            # Implementation depends on database schema
            performance[strategy.name] = {
                'enabled': strategy.enabled,
                'current_position': strategy.position,
                # Add more metrics as needed
            }

        return performance
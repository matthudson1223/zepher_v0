"""
Base Strategy Class for Trading Strategies

This module defines the abstract base class and common types for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class SignalType(Enum):
    """Types of trading signals"""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"

    def is_entry(self) -> bool:
        """Check if this is an entry signal"""
        return self in [SignalType.LONG, SignalType.SHORT]

    def is_exit(self) -> bool:
        """Check if this is an exit signal"""
        return self in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]


@dataclass
class Signal:
    """Represents a trading signal"""
    timestamp: datetime
    signal_type: SignalType
    strategy_name: str
    strength: float  # 0-100 confidence score
    price: float
    reason: str
    indicators: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return (f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"{self.strategy_name} | {self.signal_type.value} | "
                f"Strength: {self.strength:.1f}% | Price: ${self.price:,.2f}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for database storage"""
        return {
            'timestamp': int(self.timestamp.timestamp()),
            'signal_type': self.signal_type.value,
            'strategy_name': self.strategy_name,
            'strength': self.strength,
            'price': self.price,
            'reason': self.reason,
            'indicators': self.indicators,
            'metadata': self.metadata
        }


class Strategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy

        Args:
            name: Strategy name
            config: Strategy configuration dictionary
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.min_strength = config.get('min_strength', 50.0)
        self.position = None  # Track current position

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on price data and indicators

        Args:
            df: DataFrame with OHLCV data and calculated indicators
                Required columns: open, high, low, close, volume
                Additional indicator columns as needed by strategy

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def check_entry(self, row: pd.Series) -> Optional[Signal]:
        """
        Check if entry conditions are met

        Args:
            row: Current row of data with indicators

        Returns:
            Signal object if entry conditions met, None otherwise
        """
        pass

    @abstractmethod
    def check_exit(self, row: pd.Series) -> Optional[Signal]:
        """
        Check if exit conditions are met

        Args:
            row: Current row of data with indicators

        Returns:
            Signal object if exit conditions met, None otherwise
        """
        pass

    def calculate_signal_strength(self, indicators: Dict[str, float]) -> float:
        """
        Calculate signal strength based on indicators
        Override in subclasses for custom logic

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Signal strength from 0-100
        """
        return 50.0  # Default neutral strength

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate if signal meets minimum criteria

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        return signal.strength >= self.min_strength

    def filter_conflicting_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter out conflicting signals (e.g., both LONG and SHORT at same time)

        Args:
            signals: List of signals to filter

        Returns:
            Filtered list of non-conflicting signals
        """
        if not signals:
            return signals

        # Group signals by timestamp
        from collections import defaultdict
        by_time = defaultdict(list)
        for signal in signals:
            by_time[signal.timestamp].append(signal)

        filtered = []
        for timestamp, time_signals in by_time.items():
            # If multiple signals at same time, take the strongest
            if len(time_signals) > 1:
                # Separate entry and exit signals
                entries = [s for s in time_signals if s.signal_type.is_entry()]
                exits = [s for s in time_signals if s.signal_type.is_exit()]
                holds = [s for s in time_signals if s.signal_type == SignalType.HOLD]

                # Process exits first (close positions before opening new ones)
                if exits:
                    filtered.extend(sorted(exits, key=lambda x: x.strength, reverse=True)[:1])

                # Then process entries
                if entries:
                    filtered.extend(sorted(entries, key=lambda x: x.strength, reverse=True)[:1])

                # Include holds only if no other signals
                if not entries and not exits and holds:
                    filtered.extend(holds[:1])
            else:
                filtered.extend(time_signals)

        return sorted(filtered, key=lambda x: x.timestamp)

    def get_volatility_regime(self, vol_percentile: float) -> str:
        """
        Determine volatility regime based on percentile

        Args:
            vol_percentile: Volatility percentile (0-100)

        Returns:
            Regime string: 'HIGH', 'LOW', or 'NORMAL'
        """
        if vol_percentile >= 80:
            return 'HIGH'
        elif vol_percentile <= 20:
            return 'LOW'
        else:
            return 'NORMAL'

    def format_reason(self, conditions: List[str]) -> str:
        """
        Format reason string from list of conditions

        Args:
            conditions: List of condition strings

        Returns:
            Formatted reason string
        """
        if not conditions:
            return "Signal generated"
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return " AND ".join(conditions)
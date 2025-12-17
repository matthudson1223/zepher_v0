"""
Trading Strategies Module for Zepher v0

This module provides various trading strategies based on volatility indicators.
"""

from .base import Strategy, SignalType
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .compression import VolatilityCompressionStrategy

__all__ = [
    'Strategy',
    'SignalType',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'VolatilityCompressionStrategy'
]
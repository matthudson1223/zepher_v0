"""
Unit tests for trading strategies

Tests the mean reversion, breakout, and volatility compression strategies.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies import (
    MeanReversionStrategy,
    BreakoutStrategy,
    VolatilityCompressionStrategy,
    SignalType
)


class TestStrategyBase(unittest.TestCase):
    """Base class for strategy tests with common setup"""

    def setUp(self):
        """Create sample data for testing"""
        # Create 100 days of sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

        # Generate synthetic price data
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(100) * 500)

        self.df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 100,
            'high': prices + abs(np.random.randn(100) * 200),
            'low': prices - abs(np.random.randn(100) * 200),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100),
        }, index=dates)

        # Add indicators
        self._add_indicators()

    def _add_indicators(self):
        """Add volatility indicators to the dataframe"""
        # Calculate simple moving average (middle band)
        self.df['bb_middle'] = self.df['close'].rolling(window=20).mean()

        # Calculate standard deviation
        std = self.df['close'].rolling(window=20).std()

        # Bollinger Bands
        self.df['bb_upper'] = self.df['bb_middle'] + (std * 2)
        self.df['bb_lower'] = self.df['bb_middle'] - (std * 2)

        # Bollinger Band %B
        self.df['bb_percent_b'] = (
            (self.df['close'] - self.df['bb_lower']) /
            (self.df['bb_upper'] - self.df['bb_lower'])
        )

        # Bollinger Band Width
        self.df['bb_bandwidth'] = (
            (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        )

        # ATR (simplified)
        high_low = self.df['high'] - self.df['low']
        self.df['atr'] = high_low.rolling(window=14).mean()

        # Volatility (simplified - using rolling std)
        returns = self.df['close'].pct_change()
        self.df['vol_7d'] = returns.rolling(window=7).std() * np.sqrt(365)
        self.df['vol_30d'] = returns.rolling(window=30).std() * np.sqrt(365)
        self.df['vol_90d'] = returns.rolling(window=90).std() * np.sqrt(365)

        # Volatility percentile (simplified)
        self.df['vol_percentile'] = self.df['vol_30d'].rolling(window=30).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
        )


class TestMeanReversionStrategy(TestStrategyBase):
    """Test Mean Reversion Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        config = {
            'enabled': True,
            'min_strength': 50.0,
            'bb_upper_threshold': 1.0,
            'bb_lower_threshold': 0.0,
            'vol_percentile_min': 80
        }

        strategy = MeanReversionStrategy(config)

        self.assertEqual(strategy.name, "MeanReversion")
        self.assertTrue(strategy.enabled)
        self.assertEqual(strategy.min_strength, 50.0)
        self.assertEqual(strategy.bb_upper_threshold, 1.0)
        self.assertEqual(strategy.vol_percentile_min, 80)

    def test_long_signal_generation(self):
        """Test generation of long signals in high volatility"""
        config = {
            'enabled': True,
            'min_strength': 30.0,
            'vol_percentile_min': 70
        }

        strategy = MeanReversionStrategy(config)

        # Modify data to create long signal condition
        self.df.loc[self.df.index[-5:], 'bb_percent_b'] = -0.1  # Below lower band
        self.df.loc[self.df.index[-5:], 'vol_percentile'] = 85  # High volatility

        signals = strategy.generate_signals(self.df)

        # Should have at least one long signal
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        self.assertGreater(len(long_signals), 0)

        # Check signal properties
        signal = long_signals[0]
        self.assertGreaterEqual(signal.strength, 30.0)
        self.assertIn("lower band", signal.reason.lower())

    def test_short_signal_generation(self):
        """Test generation of short signals in high volatility"""
        config = {
            'enabled': True,
            'min_strength': 30.0,
            'vol_percentile_min': 70
        }

        strategy = MeanReversionStrategy(config)

        # Modify data to create short signal condition
        self.df.loc[self.df.index[-5:], 'bb_percent_b'] = 1.1  # Above upper band
        self.df.loc[self.df.index[-5:], 'vol_percentile'] = 85  # High volatility

        signals = strategy.generate_signals(self.df)

        # Should have at least one short signal
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        self.assertGreater(len(short_signals), 0)

        # Check signal properties
        signal = short_signals[0]
        self.assertGreaterEqual(signal.strength, 30.0)
        self.assertIn("upper band", signal.reason.lower())

    def test_no_signal_in_low_volatility(self):
        """Test that no signals are generated in low volatility"""
        config = {
            'enabled': True,
            'vol_percentile_min': 80
        }

        strategy = MeanReversionStrategy(config)

        # Set low volatility
        self.df['vol_percentile'] = 30  # Low volatility everywhere

        signals = strategy.generate_signals(self.df)

        # Should have no entry signals
        entry_signals = [s for s in signals if s.signal_type.is_entry()]
        self.assertEqual(len(entry_signals), 0)


class TestBreakoutStrategy(TestStrategyBase):
    """Test Breakout Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        config = {
            'enabled': True,
            'vol_percentile_max': 20,
            'bb_squeeze_threshold': 0.03,
            'volume_multiplier': 1.5
        }

        strategy = BreakoutStrategy(config)

        self.assertEqual(strategy.name, "Breakout")
        self.assertTrue(strategy.enabled)
        self.assertEqual(strategy.vol_percentile_max, 20)
        self.assertEqual(strategy.bb_squeeze_threshold, 0.03)

    def test_breakout_signal_with_volume_spike(self):
        """Test breakout signal generation with volume confirmation"""
        config = {
            'enabled': True,
            'min_strength': 30.0,
            'vol_percentile_max': 30,
            'bb_squeeze_threshold': 0.05,
            'volume_multiplier': 1.2
        }

        strategy = BreakoutStrategy(config)

        # Create breakout conditions
        self.df.loc[self.df.index[-10:], 'vol_percentile'] = 15  # Low volatility
        self.df.loc[self.df.index[-10:], 'bb_bandwidth'] = 0.02  # Squeeze
        self.df.loc[self.df.index[-3:], 'bb_percent_b'] = 1.1  # Breakout above

        # Add volume spike
        avg_volume = self.df['volume'].mean()
        self.df.loc[self.df.index[-3:], 'volume'] = avg_volume * 2

        signals = strategy.generate_signals(self.df)

        # Should have breakout signals
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        self.assertGreater(len(long_signals), 0)

    def test_no_signal_in_high_volatility(self):
        """Test that no signals are generated in high volatility"""
        config = {
            'enabled': True,
            'vol_percentile_max': 20
        }

        strategy = BreakoutStrategy(config)

        # Set high volatility
        self.df['vol_percentile'] = 80  # High volatility everywhere

        signals = strategy.generate_signals(self.df)

        # Should have no entry signals
        entry_signals = [s for s in signals if s.signal_type.is_entry()]
        self.assertEqual(len(entry_signals), 0)


class TestVolatilityCompressionStrategy(TestStrategyBase):
    """Test Volatility Compression Strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        config = {
            'enabled': True,
            'compression_period': 14,
            'bb_width_percentile': 10,
            'expansion_threshold': 1.5
        }

        strategy = VolatilityCompressionStrategy(config)

        self.assertEqual(strategy.name, "VolatilityCompression")
        self.assertTrue(strategy.enabled)
        self.assertEqual(strategy.compression_period, 14)
        self.assertEqual(strategy.expansion_threshold, 1.5)

    def test_compression_signal_generation(self):
        """Test generation of signals during volatility compression"""
        config = {
            'enabled': True,
            'min_strength': 30.0,
            'bb_width_percentile': 20,
            'compression_period': 10
        }

        strategy = VolatilityCompressionStrategy(config)

        # Create compression conditions
        # Gradually decrease BB width to simulate compression
        for i in range(20):
            self.df.loc[self.df.index[-(20-i):], 'bb_bandwidth'] = 0.05 - (i * 0.002)

        # Set other required conditions
        self.df['bb_width_percentile'] = 5  # Very low percentile
        self.df['vol_7d_change'] = -0.15  # Declining volatility
        self.df['price_range'] = 0.015  # Tight price range

        signals = strategy.generate_signals(self.df)

        # Should have at least one signal
        self.assertGreater(len(signals), 0)

    def test_exit_on_expansion(self):
        """Test exit signal when volatility expands"""
        config = {
            'enabled': True,
            'expansion_threshold': 1.5
        }

        strategy = VolatilityCompressionStrategy(config)

        # Simulate entry
        strategy.position = 'long'
        strategy.entry_bb_width = 0.02
        strategy.holding_periods = 5

        # Create expansion condition
        self.df.loc[self.df.index[-1], 'bb_bandwidth'] = 0.035  # 75% expansion

        # Check for exit signal
        exit_signal = strategy.check_exit(self.df.iloc[-1])

        self.assertIsNotNone(exit_signal)
        self.assertEqual(exit_signal.signal_type, SignalType.EXIT_LONG)


class TestSignalFiltering(unittest.TestCase):
    """Test signal filtering and validation"""

    def test_conflicting_signals(self):
        """Test that conflicting signals are properly filtered"""
        from src.strategies.base import Strategy, Signal

        # Create a mock strategy
        class MockStrategy(Strategy):
            def generate_signals(self, df):
                return []
            def check_entry(self, row):
                return None
            def check_exit(self, row):
                return None

        strategy = MockStrategy("Test", {})

        # Create conflicting signals at same timestamp
        timestamp = datetime.now()
        long_signal = Signal(
            timestamp=timestamp,
            signal_type=SignalType.LONG,
            strategy_name="Test",
            strength=60.0,
            price=50000,
            reason="Test long",
            indicators={}
        )

        short_signal = Signal(
            timestamp=timestamp,
            signal_type=SignalType.SHORT,
            strategy_name="Test",
            strength=70.0,
            price=50000,
            reason="Test short",
            indicators={}
        )

        signals = [long_signal, short_signal]
        filtered = strategy.filter_conflicting_signals(signals)

        # Should only have one signal (the stronger one)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].signal_type, SignalType.SHORT)


if __name__ == '__main__':
    unittest.main()
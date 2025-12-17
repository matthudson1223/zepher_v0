"""
Signal Filtering Module

Filters trading signals based on various criteria to reduce false signals.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from ..strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


class SignalFilter:
    """
    Filter signals to reduce noise and false positives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Signal Filter

        Args:
            config: Filter configuration
        """
        self.config = config
        self.min_strength = config.get('min_strength', 50.0)
        self.min_time_between_signals = config.get('min_time_between_signals', 60)  # minutes
        self.max_signals_per_day = config.get('max_signals_per_day', 10)
        self.filter_conflicting = config.get('filter_conflicting', True)
        self.filter_whipsaws = config.get('filter_whipsaws', True)
        self.whipsaw_window = config.get('whipsaw_window', 30)  # minutes

    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply all filters to signals

        Args:
            signals: List of raw signals

        Returns:
            Filtered list of signals
        """
        if not signals:
            return signals

        filtered = signals.copy()

        # Filter by minimum strength
        filtered = self._filter_by_strength(filtered)

        # Filter conflicting signals
        if self.filter_conflicting:
            filtered = self._filter_conflicting(filtered)

        # Filter whipsaws (rapid reversals)
        if self.filter_whipsaws:
            filtered = self._filter_whipsaws(filtered)

        # Filter by time between signals
        filtered = self._filter_by_time_spacing(filtered)

        # Limit signals per day
        filtered = self._limit_daily_signals(filtered)

        logger.info(f"Filtered {len(signals)} signals to {len(filtered)}")
        return filtered

    def _filter_by_strength(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals below minimum strength threshold"""
        filtered = [s for s in signals if s.strength >= self.min_strength]
        removed = len(signals) - len(filtered)
        if removed > 0:
            logger.debug(f"Removed {removed} weak signals (< {self.min_strength}% strength)")
        return filtered

    def _filter_conflicting(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter conflicting signals at the same timestamp

        Rules:
        - Can't have both LONG and SHORT at same time
        - Exit signals take priority over entry signals
        - Stronger signals take priority within same type
        """
        if len(signals) < 2:
            return signals

        # Group by timestamp
        from collections import defaultdict
        by_time = defaultdict(list)
        for signal in signals:
            by_time[signal.timestamp].append(signal)

        filtered = []
        for timestamp, time_signals in by_time.items():
            if len(time_signals) == 1:
                filtered.extend(time_signals)
            else:
                # Handle conflicts
                exits = [s for s in time_signals if s.signal_type.is_exit()]
                entries = [s for s in time_signals if s.signal_type.is_entry()]
                holds = [s for s in time_signals if s.signal_type == SignalType.HOLD]

                # Exits first (close before open)
                if exits:
                    # Take strongest exit
                    filtered.append(max(exits, key=lambda x: x.strength))

                # Then entries (but not conflicting ones)
                if entries:
                    longs = [s for s in entries if s.signal_type == SignalType.LONG]
                    shorts = [s for s in entries if s.signal_type == SignalType.SHORT]

                    if longs and shorts:
                        # Conflict - take stronger signal
                        strongest_long = max(longs, key=lambda x: x.strength)
                        strongest_short = max(shorts, key=lambda x: x.strength)

                        if strongest_long.strength > strongest_short.strength:
                            filtered.append(strongest_long)
                            logger.debug(f"Resolved LONG/SHORT conflict at {timestamp} - chose LONG")
                        else:
                            filtered.append(strongest_short)
                            logger.debug(f"Resolved LONG/SHORT conflict at {timestamp} - chose SHORT")
                    else:
                        # No conflict
                        if longs:
                            filtered.append(max(longs, key=lambda x: x.strength))
                        if shorts:
                            filtered.append(max(shorts, key=lambda x: x.strength))

                # Holds only if no other signals
                if not exits and not entries and holds:
                    filtered.append(holds[0])

        return sorted(filtered, key=lambda x: x.timestamp)

    def _filter_whipsaws(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter whipsaw signals (rapid reversals)

        A whipsaw is when opposite signals occur within a short time window
        """
        if len(signals) < 2:
            return signals

        filtered = []
        signals_sorted = sorted(signals, key=lambda x: x.timestamp)

        for i, signal in enumerate(signals_sorted):
            is_whipsaw = False

            # Look ahead for opposite signals within window
            for j in range(i + 1, len(signals_sorted)):
                future_signal = signals_sorted[j]
                time_diff = (future_signal.timestamp - signal.timestamp).total_seconds() / 60

                if time_diff > self.whipsaw_window:
                    break  # Outside window

                # Check for whipsaw patterns
                if signal.signal_type == SignalType.LONG and future_signal.signal_type == SignalType.SHORT:
                    is_whipsaw = True
                    logger.debug(f"Whipsaw detected: LONG at {signal.timestamp} followed by SHORT at {future_signal.timestamp}")
                    break
                elif signal.signal_type == SignalType.SHORT and future_signal.signal_type == SignalType.LONG:
                    is_whipsaw = True
                    logger.debug(f"Whipsaw detected: SHORT at {signal.timestamp} followed by LONG at {future_signal.timestamp}")
                    break

            if not is_whipsaw:
                filtered.append(signal)

        return filtered

    def _filter_by_time_spacing(self, signals: List[Signal]) -> List[Signal]:
        """
        Ensure minimum time between signals of the same type
        """
        if len(signals) < 2:
            return signals

        filtered = []
        last_signal_time = {}  # Track last signal time by type

        for signal in sorted(signals, key=lambda x: x.timestamp):
            signal_type = signal.signal_type

            # Check if enough time has passed since last signal of this type
            if signal_type in last_signal_time:
                time_diff = (signal.timestamp - last_signal_time[signal_type]).total_seconds() / 60
                if time_diff < self.min_time_between_signals:
                    logger.debug(f"Filtered {signal_type} signal at {signal.timestamp} - too close to previous")
                    continue

            filtered.append(signal)
            last_signal_time[signal_type] = signal.timestamp

        return filtered

    def _limit_daily_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Limit the number of signals per day
        """
        if len(signals) <= self.max_signals_per_day:
            return signals

        # Group by date
        from collections import defaultdict
        by_date = defaultdict(list)
        for signal in signals:
            date = signal.timestamp.date()
            by_date[date].append(signal)

        filtered = []
        for date, date_signals in by_date.items():
            if len(date_signals) <= self.max_signals_per_day:
                filtered.extend(date_signals)
            else:
                # Take the strongest signals for the day
                strongest = sorted(date_signals, key=lambda x: x.strength, reverse=True)
                filtered.extend(strongest[:self.max_signals_per_day])
                logger.debug(f"Limited signals on {date} from {len(date_signals)} to {self.max_signals_per_day}")

        return sorted(filtered, key=lambda x: x.timestamp)
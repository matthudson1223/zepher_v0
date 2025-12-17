"""
Interactive charting module using Plotly.

Creates professional trading charts with Bollinger Bands, volatility indicators,
and other technical analysis overlays.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

logger = get_logger("btc_volatility.charts")


class ChartBuilder:
    """
    Builder for creating interactive trading charts with Plotly.

    Creates multi-panel charts with price, Bollinger Bands, volume,
    and volatility indicators.

    Example:
        >>> builder = ChartBuilder(df)
        >>> fig = builder.add_candlestick().add_bollinger_bands().add_volume().build()
        >>> fig.show()
    """

    # Color scheme
    COLORS = {
        "up": "#26a69a",  # Green for bullish
        "down": "#ef5350",  # Red for bearish
        "bb_upper": "#2196F3",  # Blue
        "bb_lower": "#2196F3",
        "bb_middle": "#FFA726",  # Orange
        "bb_fill": "rgba(33, 150, 243, 0.1)",  # Light blue fill
        "volume": "#7E57C2",  # Purple
        "volatility": "#FF7043",  # Deep orange
        "grid": "#2a2a2a",
        "background": "#1a1a1a",
        "text": "#e0e0e0",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        title: str = "BTC/USDT",
        height: int = 800,
    ):
        """
        Initialize the chart builder.

        Args:
            df: DataFrame with OHLCV data and optional indicator columns.
                Expected columns: open, high, low, close, volume.
                Optional: bb_upper, bb_middle, bb_lower, vol_30d, etc.
            title: Chart title.
            height: Chart height in pixels.
        """
        self.df = df.copy()
        self.title = title
        self.height = height

        # Ensure we have a datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "timestamp" in self.df.columns:
                self.df.set_index("timestamp", inplace=True)

        # Track which subplots we need
        self._has_volume = False
        self._has_volatility = False
        self._traces: list[tuple[go.Trace, int]] = []  # (trace, row)

        logger.debug(f"ChartBuilder initialized with {len(df)} data points")

    def _create_figure(self) -> go.Figure:
        """Create the subplot figure based on what will be displayed."""
        rows = 1
        row_heights = [0.6]
        subplot_titles = [self.title]

        if self._has_volume:
            rows += 1
            row_heights.append(0.2)
            subplot_titles.append("Volume")

        if self._has_volatility:
            rows += 1
            row_heights.append(0.2)
            subplot_titles.append("Volatility")

        # Normalize row heights
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        return fig

    def add_candlestick(self) -> "ChartBuilder":
        """
        Add candlestick chart for price data.

        Returns:
            Self for method chaining.
        """
        trace = go.Candlestick(
            x=self.df.index,
            open=self.df["open"],
            high=self.df["high"],
            low=self.df["low"],
            close=self.df["close"],
            name="Price",
            increasing=dict(line=dict(color=self.COLORS["up"])),
            decreasing=dict(line=dict(color=self.COLORS["down"])),
        )
        self._traces.append((trace, 1))

        return self

    def add_line_price(self) -> "ChartBuilder":
        """
        Add line chart for closing price (alternative to candlestick).

        Returns:
            Self for method chaining.
        """
        trace = go.Scatter(
            x=self.df.index,
            y=self.df["close"],
            mode="lines",
            name="Close",
            line=dict(color=self.COLORS["bb_middle"], width=1),
        )
        self._traces.append((trace, 1))

        return self

    def add_bollinger_bands(
        self,
        upper_col: str = "bb_upper",
        middle_col: str = "bb_middle",
        lower_col: str = "bb_lower",
        show_fill: bool = True,
    ) -> "ChartBuilder":
        """
        Add Bollinger Bands overlay to the price chart.

        Args:
            upper_col: Column name for upper band.
            middle_col: Column name for middle band (SMA).
            lower_col: Column name for lower band.
            show_fill: Whether to fill the area between bands.

        Returns:
            Self for method chaining.
        """
        # Check if columns exist
        required_cols = [upper_col, middle_col, lower_col]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            logger.warning(f"Missing Bollinger Band columns: {missing}")
            return self

        # Upper band
        upper_trace = go.Scatter(
            x=self.df.index,
            y=self.df[upper_col],
            mode="lines",
            name="BB Upper",
            line=dict(color=self.COLORS["bb_upper"], width=1, dash="dash"),
        )
        self._traces.append((upper_trace, 1))

        # Middle band (SMA)
        middle_trace = go.Scatter(
            x=self.df.index,
            y=self.df[middle_col],
            mode="lines",
            name="BB Middle",
            line=dict(color=self.COLORS["bb_middle"], width=1),
        )
        self._traces.append((middle_trace, 1))

        # Lower band
        lower_trace = go.Scatter(
            x=self.df.index,
            y=self.df[lower_col],
            mode="lines",
            name="BB Lower",
            line=dict(color=self.COLORS["bb_lower"], width=1, dash="dash"),
            fill="tonexty" if show_fill else None,
            fillcolor=self.COLORS["bb_fill"] if show_fill else None,
        )
        self._traces.append((lower_trace, 1))

        return self

    def add_volume(self, col: str = "volume") -> "ChartBuilder":
        """
        Add volume bars as a subplot.

        Args:
            col: Column name for volume data.

        Returns:
            Self for method chaining.
        """
        if col not in self.df.columns:
            logger.warning(f"Volume column '{col}' not found")
            return self

        self._has_volume = True

        # Color bars based on price direction
        colors = [
            self.COLORS["up"] if close >= open_
            else self.COLORS["down"]
            for close, open_ in zip(self.df["close"], self.df["open"])
        ]

        trace = go.Bar(
            x=self.df.index,
            y=self.df[col],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        )
        # Row will be assigned dynamically
        self._traces.append((trace, "volume"))

        return self

    def add_volatility(
        self,
        col: str = "vol_30d",
        name: str = "30D Volatility",
    ) -> "ChartBuilder":
        """
        Add volatility indicator as a subplot.

        Args:
            col: Column name for volatility data.
            name: Display name for the indicator.

        Returns:
            Self for method chaining.
        """
        if col not in self.df.columns:
            logger.warning(f"Volatility column '{col}' not found")
            return self

        self._has_volatility = True

        trace = go.Scatter(
            x=self.df.index,
            y=self.df[col],
            mode="lines",
            name=name,
            line=dict(color=self.COLORS["volatility"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 112, 67, 0.2)",
        )
        self._traces.append((trace, "volatility"))

        return self

    def add_volatility_percentile(
        self,
        col: str = "vol_percentile",
        threshold_high: float = 80,
        threshold_low: float = 20,
    ) -> "ChartBuilder":
        """
        Add volatility percentile with threshold lines.

        Args:
            col: Column name for percentile data.
            threshold_high: High volatility threshold (e.g., 80).
            threshold_low: Low volatility threshold (e.g., 20).

        Returns:
            Self for method chaining.
        """
        if col not in self.df.columns:
            logger.warning(f"Percentile column '{col}' not found")
            return self

        self._has_volatility = True

        # Main percentile line
        trace = go.Scatter(
            x=self.df.index,
            y=self.df[col],
            mode="lines",
            name="Vol Percentile",
            line=dict(color=self.COLORS["volatility"], width=1.5),
        )
        self._traces.append((trace, "volatility"))

        # High threshold line
        high_line = go.Scatter(
            x=[self.df.index[0], self.df.index[-1]],
            y=[threshold_high, threshold_high],
            mode="lines",
            name=f"High ({threshold_high}%)",
            line=dict(color=self.COLORS["down"], width=1, dash="dot"),
        )
        self._traces.append((high_line, "volatility"))

        # Low threshold line
        low_line = go.Scatter(
            x=[self.df.index[0], self.df.index[-1]],
            y=[threshold_low, threshold_low],
            mode="lines",
            name=f"Low ({threshold_low}%)",
            line=dict(color=self.COLORS["up"], width=1, dash="dot"),
        )
        self._traces.append((low_line, "volatility"))

        return self

    def build(self) -> go.Figure:
        """
        Build and return the final figure.

        Returns:
            Plotly Figure object ready for display or export.
        """
        fig = self._create_figure()

        # Determine row mapping
        row_map = {"volume": 2 if self._has_volume else None}
        if self._has_volatility:
            row_map["volatility"] = (
                3 if self._has_volume else 2
            )

        # Add all traces
        for trace, row in self._traces:
            if isinstance(row, str):
                row = row_map.get(row)
            if row:
                fig.add_trace(trace, row=row, col=1)

        # Apply dark theme styling
        fig.update_layout(
            height=self.height,
            template="plotly_dark",
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"]),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
            ),
            margin=dict(l=60, r=30, t=50, b=30),
            xaxis_rangeslider_visible=False,
        )

        # Style axes
        fig.update_xaxes(
            gridcolor=self.COLORS["grid"],
            showgrid=True,
            zeroline=False,
        )
        fig.update_yaxes(
            gridcolor=self.COLORS["grid"],
            showgrid=True,
            zeroline=False,
        )

        logger.info("Chart built successfully")

        return fig

    def save_html(
        self,
        filepath: str | Path,
        include_plotlyjs: bool = True,
    ) -> None:
        """
        Save the chart as an interactive HTML file.

        Args:
            filepath: Output file path.
            include_plotlyjs: Whether to include Plotly.js in the file.
        """
        fig = self.build()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(
            filepath,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
        )

        logger.info(f"Chart saved to {filepath}")

    def show(self) -> None:
        """Display the chart in a browser or notebook."""
        fig = self.build()
        fig.show()


def create_bollinger_chart(
    df: pd.DataFrame,
    title: str = "BTC/USDT with Bollinger Bands",
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    """
    Convenience function to create a price chart with Bollinger Bands.

    Args:
        df: DataFrame with OHLCV and Bollinger Band columns.
        title: Chart title.
        save_path: Optional path to save HTML file.

    Returns:
        Plotly Figure object.

    Example:
        >>> from src.indicators import VolatilityCalculator
        >>> calc = VolatilityCalculator()
        >>> df = calc.add_indicators_to_df(ohlcv_data)
        >>> fig = create_bollinger_chart(df)
        >>> fig.show()
    """
    builder = ChartBuilder(df, title=title)
    builder.add_candlestick()
    builder.add_bollinger_bands()
    builder.add_volume()
    builder.add_volatility()

    if save_path:
        builder.save_html(save_path)

    return builder.build()

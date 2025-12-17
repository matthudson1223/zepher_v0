"""
SQLite database module for storing OHLCV data.

Provides persistent storage for historical price data with support for
multiple exchanges, symbols, and timeframes.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("btc_volatility.database")


class Database:
    """
    SQLite database manager for OHLCV data storage.

    Handles storage and retrieval of historical price data with support for
    multiple exchanges, trading pairs, and timeframes.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> db = Database("data.db")
        >>> db.insert_ohlcv(df, exchange="kraken", symbol="BTC/USDT", timeframe="1d")
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Will be created if it doesn't exist.
        """
        self.db_path = Path(db_path)

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Yields:
            SQLite connection with row factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Enable foreign keys and WAL mode for better performance
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            # OHLCV data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, symbol, timeframe, timestamp)
                )
            """)

            # Create indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
                ON ohlcv(exchange, symbol, timeframe, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp
                ON ohlcv(timestamp)
            """)

            # Metadata table for tracking data freshness
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    first_timestamp INTEGER,
                    last_timestamp INTEGER,
                    record_count INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, symbol, timeframe)
                )
            """)

        logger.debug("Database schema initialized")

    def insert_ohlcv(
        self,
        df: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> int:
        """
        Insert OHLCV data into the database.

        Uses INSERT OR IGNORE to skip duplicates based on the unique constraint.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume.
                Timestamp can be datetime or Unix milliseconds.
            exchange: Exchange name (e.g., "binance").
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1d").

        Returns:
            Number of rows inserted.

        Raises:
            ValueError: If required columns are missing from DataFrame.
        """
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to insert")
            return 0

        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            # Convert timestamp to Unix milliseconds if needed
            ts = row["timestamp"]
            if isinstance(ts, (datetime, pd.Timestamp)):
                ts = int(ts.timestamp() * 1000)
            else:
                ts = int(ts)

            records.append((
                exchange,
                symbol,
                timeframe,
                ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ))

        # Insert data
        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO ohlcv
                (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            inserted = cursor.rowcount

            # Update metadata
            self._update_metadata(conn, exchange, symbol, timeframe)

        logger.info(
            f"Inserted {inserted}/{len(records)} records for "
            f"{exchange}:{symbol}:{timeframe}"
        )

        return inserted

    def _update_metadata(
        self,
        conn: sqlite3.Connection,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Update metadata table with current data statistics."""
        conn.execute("""
            INSERT INTO data_metadata (exchange, symbol, timeframe, first_timestamp, last_timestamp, record_count)
            SELECT ?, ?, ?,
                   MIN(timestamp),
                   MAX(timestamp),
                   COUNT(*)
            FROM ohlcv
            WHERE exchange = ? AND symbol = ? AND timeframe = ?
            ON CONFLICT(exchange, symbol, timeframe) DO UPDATE SET
                first_timestamp = excluded.first_timestamp,
                last_timestamp = excluded.last_timestamp,
                record_count = excluded.record_count,
                updated_at = CURRENT_TIMESTAMP
        """, (exchange, symbol, timeframe, exchange, symbol, timeframe))

    def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime | int] = None,
        end_time: Optional[datetime | int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from the database.

        Args:
            exchange: Exchange name.
            symbol: Trading pair.
            timeframe: Candle timeframe.
            start_time: Start timestamp (datetime or Unix ms). Inclusive.
            end_time: End timestamp (datetime or Unix ms). Inclusive.
            limit: Maximum number of records to return.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            Index is set to datetime timestamp.

    Example:
        >>> data = db.get_ohlcv("kraken", "BTC/USDT", "1d")
        >>> print(data.head())
    """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE exchange = ? AND symbol = ? AND timeframe = ?
        """
        params: list = [exchange, symbol, timeframe]

        # Add time filters
        if start_time:
            if isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            if isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)
            query += " AND timestamp <= ?"
            params.append(end_time)

        # Order by timestamp
        query += " ORDER BY timestamp ASC"

        # Add limit
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        # Execute query
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            logger.warning(
                f"No data found for {exchange}:{symbol}:{timeframe}"
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Convert timestamp to datetime index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        logger.debug(f"Retrieved {len(df)} records for {exchange}:{symbol}:{timeframe}")

        return df

    def get_latest_timestamp(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> Optional[int]:
        """
        Get the most recent timestamp for a given exchange/symbol/timeframe.

        Useful for determining where to resume data fetching.

        Args:
            exchange: Exchange name.
            symbol: Trading pair.
            timeframe: Candle timeframe.

        Returns:
            Unix timestamp in milliseconds, or None if no data exists.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT last_timestamp FROM data_metadata
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
                """,
                (exchange, symbol, timeframe),
            )
            row = cursor.fetchone()

        return row["last_timestamp"] if row else None

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get a summary of all data stored in the database.

        Returns:
            DataFrame with columns: exchange, symbol, timeframe, record_count,
            first_date, last_date.
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    exchange,
                    symbol,
                    timeframe,
                    record_count,
                    datetime(first_timestamp/1000, 'unixepoch') as first_date,
                    datetime(last_timestamp/1000, 'unixepoch') as last_date,
                    updated_at
                FROM data_metadata
                ORDER BY exchange, symbol, timeframe
                """,
                conn,
            )

        return df

    def delete_data(
        self,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """
        Delete OHLCV data matching the specified criteria.

        Args:
            exchange: Exchange name. If None, matches all exchanges.
            symbol: Trading pair. If None, matches all symbols.
            timeframe: Candle timeframe. If None, matches all timeframes.

        Returns:
            Number of rows deleted.

        Warning:
            If all parameters are None, ALL data will be deleted.
        """
        conditions = []
        params = []

        if exchange:
            conditions.append("exchange = ?")
            params.append(exchange)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM ohlcv WHERE {where_clause}",
                params,
            )
            deleted = cursor.rowcount

            # Also clean up metadata
            conn.execute(
                f"DELETE FROM data_metadata WHERE {where_clause}",
                params,
            )

        logger.info(f"Deleted {deleted} records")

        return deleted

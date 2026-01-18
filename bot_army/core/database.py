"""Database management for TimescaleDB."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json

try:
    import asyncpg
except ImportError:
    asyncpg = None

logger = logging.getLogger("database")


class DatabaseManager:
    """Manages PostgreSQL/TimescaleDB connections and operations."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "polymania",
        user: str = "polymania",
        password: str = "polymania"
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create connection pool."""
        if asyncpg is None:
            logger.warning("asyncpg not installed - using mock database")
            return
        
        self._pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=2,
            max_size=10
        )
        logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}")
    
    async def disconnect(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Disconnected from TimescaleDB")
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        if not self._pool:
            return "OK"
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict]:
        """Fetch rows from query."""
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def fetchone(self, query: str, *args) -> Optional[Dict]:
        """Fetch single row."""
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def store_batch(self, data: List[Any], table: str = None):
        """Store batch of data."""
        if not self._pool or not data:
            return
        # Implemented in subclass
        pass


class TimeseriesDB(DatabaseManager):
    
    async def store_batch(self, data: List[Any], table: str = None):
        """Store batch of data based on type."""
        if not self._pool or not data:
            return
        
        # Determine type from first item
        first = data[0]
        class_name = first.__class__.__name__
        
        try:
            if class_name == "MarketData":
                await self._store_market_data_batch(data)
            elif class_name == "Event":
                await self._store_events_batch(data)
            elif class_name == "Trade":
                for item in data:
                    await self.store_trade(item)
            elif class_name == "Signal":
                for item in data:
                    await self.store_signal(item)
            else:
                logger.debug(f"Unknown data type for batch storage: {class_name}")
        except Exception as e:
            logger.error(f"Error storing batch: {e}")
    
    async def _store_market_data_batch(self, data: List):
        """Efficiently store market data batch."""
        values = []
        for d in data:
            values.append((
                d.timestamp, d.market_id, d.event_id, d.outcome,
                d.price, d.bid, d.ask, d.spread,
                d.volume_24h, d.volume_1h, d.open_interest
            ))
        
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO market_data 
                (time, market_id, event_id, outcome, price, bid, ask, spread, 
                 volume_24h, volume_1h, open_interest)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                values
            )
        logger.info(f"Stored {len(data)} market data records")
    
    async def _store_events_batch(self, data: List):
        """Efficiently store events batch."""
        for event in data:
            try:
                await self.execute(
                    """
                    INSERT INTO events 
                    (event_id, title, slug, category, start_date, end_date,
                     volume, liquidity, outcomes, metadata, collected_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (event_id) DO UPDATE SET
                        volume = EXCLUDED.volume,
                        liquidity = EXCLUDED.liquidity,
                        collected_at = EXCLUDED.collected_at
                    """,
                    event.event_id, event.title, event.slug, event.category,
                    event.start_date, event.end_date, event.volume, event.liquidity,
                    json.dumps(event.outcomes), json.dumps(event.metadata),
                    event.collected_at
                )
            except Exception as e:
                logger.debug(f"Error storing event {event.event_id}: {e}")
        logger.info(f"Stored {len(data)} events")
    """TimescaleDB-specific operations for time-series data."""
    
    async def init_schema(self):
        """Initialize database schema with hypertables."""
        schema = """
        -- Market data hypertable
        CREATE TABLE IF NOT EXISTS market_data (
            time TIMESTAMPTZ NOT NULL,
            market_id TEXT NOT NULL,
            event_id TEXT,
            outcome TEXT,
            price DOUBLE PRECISION,
            bid DOUBLE PRECISION,
            ask DOUBLE PRECISION,
            spread DOUBLE PRECISION,
            volume_24h DOUBLE PRECISION,
            volume_1h DOUBLE PRECISION,
            open_interest DOUBLE PRECISION
        );
        
        -- Signals table
        CREATE TABLE IF NOT EXISTS signals (
            time TIMESTAMPTZ NOT NULL,
            signal_id TEXT PRIMARY KEY,
            market_id TEXT,
            event_id TEXT,
            signal_type TEXT,
            confidence DOUBLE PRECISION,
            price_target DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            take_profit DOUBLE PRECISION,
            strategy_name TEXT,
            features JSONB,
            ml_scores JSONB,
            metadata JSONB
        );
        
        -- Trades table
        CREATE TABLE IF NOT EXISTS trades (
            time TIMESTAMPTZ NOT NULL,
            trade_id TEXT PRIMARY KEY,
            signal_id TEXT,
            market_id TEXT,
            event_id TEXT,
            side TEXT,
            quantity DOUBLE PRECISION,
            price DOUBLE PRECISION,
            value DOUBLE PRECISION,
            fees DOUBLE PRECISION,
            pnl DOUBLE PRECISION,
            status TEXT,
            strategy_name TEXT,
            metadata JSONB
        );
        
        -- Positions table
        CREATE TABLE IF NOT EXISTS positions (
            position_id TEXT PRIMARY KEY,
            market_id TEXT,
            event_id TEXT,
            outcome TEXT,
            side TEXT,
            quantity DOUBLE PRECISION,
            entry_price DOUBLE PRECISION,
            current_price DOUBLE PRECISION,
            unrealized_pnl DOUBLE PRECISION,
            realized_pnl DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            take_profit DOUBLE PRECISION,
            strategy_name TEXT,
            opened_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ
        );
        
        -- Portfolio snapshots hypertable
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            time TIMESTAMPTZ NOT NULL,
            snapshot_id TEXT,
            total_value DOUBLE PRECISION,
            cash DOUBLE PRECISION,
            positions_value DOUBLE PRECISION,
            unrealized_pnl DOUBLE PRECISION,
            realized_pnl DOUBLE PRECISION,
            total_pnl DOUBLE PRECISION,
            position_count INTEGER,
            win_rate DOUBLE PRECISION,
            sharpe_ratio DOUBLE PRECISION,
            max_drawdown DOUBLE PRECISION
        );

        -- Strategy allocation snapshots
        CREATE TABLE IF NOT EXISTS strategy_allocations (
            time TIMESTAMPTZ NOT NULL,
            strategy_name TEXT NOT NULL,
            weight DOUBLE PRECISION,
            trade_count INTEGER,
            win_rate DOUBLE PRECISION,
            total_pnl DOUBLE PRECISION
        );

        -- Strategy correlation snapshots
        CREATE TABLE IF NOT EXISTS strategy_correlations (
            time TIMESTAMPTZ NOT NULL,
            strategy_a TEXT NOT NULL,
            strategy_b TEXT NOT NULL,
            correlation DOUBLE PRECISION
        );
        
        -- ML features table
        CREATE TABLE IF NOT EXISTS ml_features (
            time TIMESTAMPTZ NOT NULL,
            market_id TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value DOUBLE PRECISION,
            feature_metadata JSONB
        );
        
        -- Model predictions table
        CREATE TABLE IF NOT EXISTS ml_predictions (
            time TIMESTAMPTZ NOT NULL,
            prediction_id TEXT,
            market_id TEXT,
            model_name TEXT,
            prediction DOUBLE PRECISION,
            confidence DOUBLE PRECISION,
            features_used JSONB,
            actual_outcome DOUBLE PRECISION,
            error DOUBLE PRECISION
        );
        
        -- Events table
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            title TEXT,
            slug TEXT,
            category TEXT,
            start_date TIMESTAMPTZ,
            end_date TIMESTAMPTZ,
            volume DOUBLE PRECISION,
            liquidity DOUBLE PRECISION,
            outcomes JSONB,
            metadata JSONB,
            collected_at TIMESTAMPTZ
        );

        -- External signals (news/telegram/etc)
        CREATE TABLE IF NOT EXISTS external_signals (
            time TIMESTAMPTZ NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            channel TEXT,
            market_id TEXT,
            event_id TEXT,
            sentiment_score DOUBLE PRECISION,
            magnitude DOUBLE PRECISION,
            label TEXT,
            text TEXT,
            url TEXT,
            metadata JSONB,
            UNIQUE (source, source_id)
        );

        -- External correlation snapshots
        CREATE TABLE IF NOT EXISTS external_correlations (
            time TIMESTAMPTZ NOT NULL,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            correlation DOUBLE PRECISION,
            window_hours INTEGER,
            sample_count INTEGER
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_market_data_market ON market_data (market_id, time DESC);
        CREATE INDEX IF NOT EXISTS idx_signals_market ON signals (market_id, time DESC);
        CREATE INDEX IF NOT EXISTS idx_trades_market ON trades (market_id, time DESC);
        CREATE INDEX IF NOT EXISTS idx_ml_features_market ON ml_features (market_id, time DESC);
        CREATE INDEX IF NOT EXISTS idx_strategy_allocations_time ON strategy_allocations (time DESC);
        CREATE INDEX IF NOT EXISTS idx_strategy_correlations_time ON strategy_correlations (time DESC);
        CREATE INDEX IF NOT EXISTS idx_external_signals_time ON external_signals (time DESC);
        CREATE INDEX IF NOT EXISTS idx_external_signals_source ON external_signals (source, time DESC);
        CREATE INDEX IF NOT EXISTS idx_external_correlations_time ON external_correlations (time DESC);
        """
        
        try:
            # Execute schema
            for statement in schema.split(';'):
                statement = statement.strip()
                if statement:
                    await self.execute(statement)
            
            # Create hypertables (TimescaleDB specific)
            hypertables = [
                "market_data", "portfolio_snapshots", 
                "ml_features", "ml_predictions"
            ]
            for table in hypertables:
                try:
                    await self.execute(
                        f"SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE)"
                    )
                except Exception:
                    pass  # Already a hypertable

            # Schema upgrades for existing tables
            try:
                await self.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS pnl DOUBLE PRECISION")
            except Exception:
                pass

            # Strategy performance view for dashboards
            await self.execute("DROP VIEW IF EXISTS strategy_performance")
            await self.execute(
                """
                CREATE VIEW strategy_performance AS
                SELECT
                    strategy_name,
                    COUNT(pnl) AS trade_count,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
                    COALESCE(SUM(pnl), 0) AS total_pnl,
                    CASE
                        WHEN COUNT(pnl) > 0 THEN
                            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::DOUBLE PRECISION / COUNT(pnl)
                        ELSE 0
                    END AS win_rate
                FROM trades
                WHERE status = 'FILLED' AND pnl IS NOT NULL
                GROUP BY strategy_name
                """
            )

            await self.execute("DROP VIEW IF EXISTS go_live_readiness")
            await self.execute(
                """
                CREATE VIEW go_live_readiness AS
                SELECT
                    time,
                    sharpe_ratio,
                    max_drawdown,
                    total_pnl,
                    CASE
                        WHEN sharpe_ratio >= 1.75
                             AND max_drawdown <= 0.10
                             AND total_pnl > 0
                        THEN 'READY'
                        ELSE 'NOT_READY'
                    END AS status
                FROM portfolio_snapshots
                ORDER BY time DESC
                LIMIT 1
                """
            )

            await self.execute("DROP VIEW IF EXISTS go_live_checklist")
            await self.execute(
                """
                CREATE VIEW go_live_checklist AS
                WITH latest AS (
                    SELECT
                        time,
                        sharpe_ratio,
                        max_drawdown,
                        total_pnl
                    FROM portfolio_snapshots
                    ORDER BY time DESC
                    LIMIT 1
                )
                SELECT
                    'Sharpe Ratio' AS criteria,
                    '>= 1.75' AS threshold,
                    sharpe_ratio AS value,
                    CASE WHEN sharpe_ratio >= 1.75 THEN 'PASS' ELSE 'FAIL' END AS status
                FROM latest
                UNION ALL
                SELECT
                    'Max Drawdown' AS criteria,
                    '<= 10%' AS threshold,
                    max_drawdown AS value,
                    CASE WHEN max_drawdown <= 0.10 THEN 'PASS' ELSE 'FAIL' END AS status
                FROM latest
                UNION ALL
                SELECT
                    'Total PnL' AS criteria,
                    '> 0' AS threshold,
                    total_pnl AS value,
                    CASE WHEN total_pnl > 0 THEN 'PASS' ELSE 'FAIL' END AS status
                FROM latest
                """
            )
            
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
    
    async def store_market_data(self, data: 'MarketData'):
        """Store market tick data."""
        await self.execute(
            """
            INSERT INTO market_data 
            (time, market_id, event_id, outcome, price, bid, ask, spread, 
             volume_24h, volume_1h, open_interest)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            data.timestamp, data.market_id, data.event_id, data.outcome,
            data.price, data.bid, data.ask, data.spread,
            data.volume_24h, data.volume_1h, data.open_interest
        )
    
    async def store_signal(self, signal: 'Signal'):
        """Store trading signal."""
        await self.execute(
            """
            INSERT INTO signals 
            (time, signal_id, market_id, event_id, signal_type, confidence,
             price_target, stop_loss, take_profit, strategy_name, features, 
             ml_scores, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (signal_id) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                ml_scores = EXCLUDED.ml_scores
            """,
            signal.timestamp, signal.signal_id, signal.market_id, signal.event_id,
            signal.signal_type.value, signal.confidence, signal.price_target,
            signal.stop_loss, signal.take_profit, signal.strategy_name,
            json.dumps(signal.features), json.dumps(signal.ml_scores),
            json.dumps(signal.metadata)
        )
    
    async def store_trade(self, trade: 'Trade'):
        """Store executed trade."""
        await self.execute(
            """
            INSERT INTO trades 
            (time, trade_id, signal_id, market_id, event_id, side, quantity,
             price, value, fees, pnl, status, strategy_name, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (trade_id) DO UPDATE SET
                status = EXCLUDED.status,
                value = EXCLUDED.value
            """,
            trade.created_at, trade.trade_id, trade.signal_id, trade.market_id,
            trade.event_id, trade.side, trade.quantity, trade.price, trade.value,
            trade.fees, trade.pnl, trade.status.value, trade.strategy_name,
            json.dumps(trade.metadata)
        )
    
    async def store_ml_features(
        self,
        market_id: str,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store ML features for a market."""
        if not features:
            return
        
        ts = timestamp or datetime.utcnow()
        meta_json = json.dumps(metadata or {})
        values = []
        for name, value in features.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            values.append((ts, market_id, name, numeric_value, meta_json))
        
        if not values:
            return
        
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ml_features
                (time, market_id, feature_name, feature_value, feature_metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                values
            )

    async def store_ml_prediction(self, prediction: 'Prediction'):
        """Store ML model prediction."""
        prediction_id = f"{prediction.market_id}:{prediction.model_name}:{prediction.timestamp.isoformat()}"
        await self.execute(
            """
            INSERT INTO ml_predictions
            (time, prediction_id, market_id, model_name, prediction, confidence,
             features_used, actual_outcome, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            prediction.timestamp,
            prediction_id,
            prediction.market_id,
            prediction.model_name,
            prediction.prediction,
            prediction.confidence,
            json.dumps(prediction.features_used),
            None,
            None
        )

    async def store_portfolio_snapshot(self, snapshot: 'PortfolioSnapshot'):
        """Store portfolio snapshot."""
        await self.execute(
            """
            INSERT INTO portfolio_snapshots
            (time, snapshot_id, total_value, cash, positions_value, unrealized_pnl,
             realized_pnl, total_pnl, position_count, win_rate, sharpe_ratio, max_drawdown)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            snapshot.timestamp, snapshot.snapshot_id, snapshot.total_value,
            snapshot.cash, snapshot.positions_value, snapshot.unrealized_pnl,
            snapshot.realized_pnl, snapshot.total_pnl, snapshot.position_count,
            snapshot.win_rate, snapshot.sharpe_ratio, snapshot.max_drawdown
        )
    
    async def replace_positions(self, positions: List['Position']):
        """Replace positions table with current open positions."""
        await self.execute("DELETE FROM positions")
        if not positions:
            return
        
        values = [
            (
                p.position_id, p.market_id, p.event_id, p.outcome, p.side,
                p.quantity, p.entry_price, p.current_price, p.unrealized_pnl,
                p.realized_pnl, p.stop_loss, p.take_profit, p.strategy_name,
                p.opened_at, p.updated_at
            )
            for p in positions
        ]
        
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO positions
                (position_id, market_id, event_id, outcome, side, quantity,
                 entry_price, current_price, unrealized_pnl, realized_pnl,
                 stop_loss, take_profit, strategy_name, opened_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                values
            )
    
    async def get_market_history(
        self, 
        market_id: str, 
        hours: int = 24
    ) -> List[Dict]:
        """Get market price history."""
        return await self.fetch(
            """
            SELECT * FROM market_data
            WHERE market_id = $1 AND time > NOW() - ($2 * INTERVAL '1 hour')
            ORDER BY time DESC
            """,
            market_id, hours
        )
    
    async def get_ohlcv(
        self,
        market_id: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict]:
        """Get OHLCV candles using TimescaleDB time_bucket."""
        return await self.fetch(
            f"""
            SELECT 
                time_bucket('{interval}', time) AS bucket,
                first(price, time) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price, time) AS close,
                sum(volume_1h) AS volume
            FROM market_data
            WHERE market_id = $1
            GROUP BY bucket
            ORDER BY bucket DESC
            LIMIT $2
            """,
            market_id, limit
        )
    
    async def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio value history."""
        return await self.fetch(
            """
            SELECT * FROM portfolio_snapshots
            WHERE time > NOW() - ($1 * INTERVAL '1 day')
            ORDER BY time DESC
            """,
            days
        )
    
    async def get_strategy_performance(self) -> List[Dict]:
        """Get performance by strategy."""
        return await self.fetch(
            """
            SELECT
                strategy_name,
                trade_count,
                wins,
                losses,
                total_pnl,
                win_rate
            FROM strategy_performance
            ORDER BY total_pnl DESC
            """
        )

    async def store_strategy_allocations(
        self,
        timestamp: datetime,
        allocations: Dict[str, Dict[str, float]]
    ):
        """Store strategy allocation snapshot."""
        for name, metrics in allocations.items():
            await self.execute(
                """
                INSERT INTO strategy_allocations
                (time, strategy_name, weight, trade_count, win_rate, total_pnl)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                timestamp,
                name,
                metrics.get("weight"),
                metrics.get("trade_count"),
                metrics.get("win_rate"),
                metrics.get("total_pnl")
            )

    async def store_strategy_correlations(
        self,
        timestamp: datetime,
        correlations: List[Dict[str, Any]]
    ):
        """Store strategy correlation snapshot."""
        for row in correlations:
            await self.execute(
                """
                INSERT INTO strategy_correlations
                (time, strategy_a, strategy_b, correlation)
                VALUES ($1, $2, $3, $4)
                """,
                timestamp,
                row.get("strategy_a"),
                row.get("strategy_b"),
                row.get("correlation")
            )

    async def store_external_signals(self, signals: List[Dict[str, Any]]):
        """Store external signals (news/telegram/etc)."""
        if not signals:
            return
        
        for signal in signals:
            await self.execute(
                """
                INSERT INTO external_signals
                (time, source, source_id, channel, market_id, event_id,
                 sentiment_score, magnitude, label, text, url, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (source, source_id) DO NOTHING
                """,
                signal.get("time") or datetime.utcnow(),
                signal.get("source"),
                signal.get("source_id"),
                signal.get("channel"),
                signal.get("market_id"),
                signal.get("event_id"),
                signal.get("sentiment_score"),
                signal.get("magnitude"),
                signal.get("label"),
                signal.get("text"),
                signal.get("url"),
                json.dumps(signal.get("metadata") or {})
            )

    async def refresh_external_correlations(self, window_hours: int = 168):
        """Compute and store external sentiment correlations."""
        await self.execute(
            """
            INSERT INTO external_correlations
            (time, source, target, correlation, window_hours, sample_count)
            WITH sentiment AS (
                SELECT
                    time_bucket('1 hour', time) AS bucket,
                    source,
                    AVG(sentiment_score) AS avg_score
                FROM external_signals
                WHERE time > NOW() - ($1 * INTERVAL '1 hour')
                GROUP BY bucket, source
            ),
            portfolio AS (
                SELECT
                    time_bucket('1 hour', time) AS bucket,
                    last(total_value, time) AS total_value
                FROM portfolio_snapshots
                WHERE time > NOW() - ($1 * INTERVAL '1 hour')
                GROUP BY bucket
            ),
            returns AS (
                SELECT
                    bucket,
                    (total_value / LAG(total_value) OVER (ORDER BY bucket) - 1) AS ret
                FROM portfolio
            )
            SELECT
                NOW() AS time,
                sentiment.source,
                'portfolio_return' AS target,
                corr(sentiment.avg_score, returns.ret) AS correlation,
                $1 AS window_hours,
                COUNT(*) AS sample_count
            FROM sentiment
            JOIN returns USING (bucket)
            WHERE returns.ret IS NOT NULL
            GROUP BY sentiment.source
            """,
            window_hours
        )
        await self.execute(
            """
            INSERT INTO external_correlations
            (time, source, target, correlation, window_hours, sample_count)
            WITH sentiment AS (
                SELECT
                    time_bucket('1 hour', time) AS bucket,
                    source,
                    market_id,
                    AVG(sentiment_score) AS avg_score
                FROM external_signals
                WHERE time > NOW() - ($1 * INTERVAL '1 hour')
                  AND market_id IS NOT NULL
                GROUP BY bucket, source, market_id
            ),
            market_prices AS (
                SELECT
                    time_bucket('1 hour', time) AS bucket,
                    market_id,
                    last(price, time) AS price
                FROM market_data
                WHERE time > NOW() - ($1 * INTERVAL '1 hour')
                GROUP BY bucket, market_id
            ),
            returns AS (
                SELECT
                    bucket,
                    market_id,
                    (price / LAG(price) OVER (PARTITION BY market_id ORDER BY bucket) - 1) AS ret
                FROM market_prices
            )
            SELECT
                NOW() AS time,
                sentiment.source,
                ('market_return:' || sentiment.market_id) AS target,
                corr(sentiment.avg_score, returns.ret) AS correlation,
                $1 AS window_hours,
                COUNT(*) AS sample_count
            FROM sentiment
            JOIN returns USING (bucket, market_id)
            WHERE returns.ret IS NOT NULL
            GROUP BY sentiment.source, sentiment.market_id
            """,
            window_hours
        )

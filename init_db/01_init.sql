-- PolyMania Bot Army - Database Initialization
-- =============================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

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

SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_market_data_market ON market_data (market_id, time DESC);

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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

CREATE INDEX IF NOT EXISTS idx_signals_market ON signals (market_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy_name, time DESC);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trade_id TEXT PRIMARY KEY,
    signal_id TEXT REFERENCES signals(signal_id),
    market_id TEXT,
    event_id TEXT,
    side TEXT,
    quantity DOUBLE PRECISION,
    price DOUBLE PRECISION,
    value DOUBLE PRECISION,
    fees DOUBLE PRECISION,
    status TEXT,
    strategy_name TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_trades_market ON trades (market_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_signal ON trades (signal_id);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
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
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_market ON positions (market_id);

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

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- ML features table
CREATE TABLE IF NOT EXISTS ml_features (
    time TIMESTAMPTZ NOT NULL,
    market_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value DOUBLE PRECISION,
    feature_metadata JSONB
);

SELECT create_hypertable('ml_features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ml_features_market ON ml_features (market_id, time DESC);

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

SELECT create_hypertable('ml_predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON ml_predictions (model_name, time DESC);

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
    collected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_category ON events (category);
CREATE INDEX IF NOT EXISTS idx_events_end ON events (end_date);

-- Strategy performance view
CREATE OR REPLACE VIEW strategy_performance AS
SELECT 
    strategy_name,
    COUNT(*) as trade_count,
    SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN value < 0 THEN 1 ELSE 0 END) as losses,
    SUM(value) as total_pnl,
    AVG(value) as avg_pnl,
    STDDEV(value) as pnl_stddev,
    CASE WHEN COUNT(*) > 0 
         THEN SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
         ELSE 0 END as win_rate
FROM trades
WHERE status = 'FILLED'
GROUP BY strategy_name;

-- Daily PnL view
CREATE OR REPLACE VIEW daily_pnl AS
SELECT 
    time_bucket('1 day', time) as day,
    SUM(value) as pnl,
    COUNT(*) as trades
FROM trades
WHERE status = 'FILLED'
GROUP BY day
ORDER BY day DESC;

-- Continuous aggregate for market OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS market_ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    market_id,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume_1h) AS volume
FROM market_data
GROUP BY bucket, market_id;

-- Refresh policy
SELECT add_continuous_aggregate_policy('market_ohlcv_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Retention policy (keep 90 days of detailed data)
SELECT add_retention_policy('market_data', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('ml_features', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('ml_predictions', INTERVAL '30 days', if_not_exists => TRUE);

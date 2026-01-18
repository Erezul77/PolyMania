# PolyMania Trading System - Complete Documentation

## 🎯 System Overview

PolyMania is a **world-class algorithmic trading system** designed for prediction markets (Polymarket). It combines multiple independent trading strategies with advanced machine learning to make profitable trading decisions.

### What It Does
1. **Collects market data** from Polymarket (prices, orderbooks, trades)
2. **Analyzes markets** using 12 different trading strategies
3. **Selects best signals** using ML-based tournament system
4. **Manages risk** with Kelly criterion and portfolio optimization
5. **Executes trades** when confidence is high enough
6. **Learns continuously** from trade outcomes

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     PolyMania Bot Army                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Collectors │  │  Strategies │  │  ML Engine  │             │
│  │  - Polymarket│  │  - 12 Types │  │  - Meta     │             │
│  │  - External │  │  - Tournament│  │  - Regime   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Signal Processing & Selection               │   │
│  │  - Thompson Sampling | Correlation Filter | Meta-Learner │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Risk Management                        │   │
│  │  - Kelly Sizing | Risk Parity | Drawdown Control | VaR   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Trade Execution                         │   │
│  │  - Order Placement | Position Tracking | PnL Calculation  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
     ┌──────────┐         ┌──────────┐         ┌──────────┐
     │TimescaleDB│         │  Redis   │         │ Grafana  │
     │(Persistence)│       │ (Cache)  │         │(Dashboard)│
     └──────────┘         └──────────┘         └──────────┘
```

---

## 📊 The 12 Trading Strategies

Each strategy operates independently and competes in the tournament:

### 1. Momentum Strategy
- **Logic**: Buy when price is rising with increasing momentum
- **Indicators**: RSI, MACD, Rate of Change
- **Best in**: Trending markets
- **Timeframe**: Short-term (minutes to hours)

### 2. Mean Reversion Strategy
- **Logic**: Buy oversold, sell overbought - prices return to mean
- **Indicators**: Bollinger Bands, RSI extremes, Z-score
- **Best in**: Range-bound/choppy markets
- **Timeframe**: Short-term

### 3. Statistical Arbitrage (Stat Arb)
- **Logic**: Exploit price deviations from statistical fair value
- **Indicators**: Cointegration, spread analysis, Hurst exponent
- **Best in**: Markets with related assets
- **Timeframe**: Medium-term

### 4. Breakout Strategy
- **Logic**: Enter when price breaks through support/resistance
- **Indicators**: Price channels, volume confirmation, ATR
- **Best in**: After consolidation periods
- **Timeframe**: Any

### 5. Trend Following Strategy
- **Logic**: Follow established trends, ride momentum
- **Indicators**: Moving average crossovers, ADX, Supertrend
- **Best in**: Strong trending markets
- **Timeframe**: Medium to long-term

### 6. Volatility Strategy
- **Logic**: Trade based on volatility expansion/contraction
- **Indicators**: ATR, Bollinger Band width, VIX-like measures
- **Best in**: Volatility regime changes
- **Timeframe**: Any

### 7. Counter-Trend Strategy
- **Logic**: Fade extreme moves, anticipate reversals
- **Indicators**: Divergences, exhaustion patterns, overbought/oversold
- **Best in**: After extended moves
- **Timeframe**: Short-term

### 8. Volume Profile Strategy
- **Logic**: Trade based on volume patterns and VWAP
- **Indicators**: Volume profile, VWAP, OBV
- **Best in**: Liquid markets
- **Timeframe**: Intraday

### 9. Sentiment Strategy
- **Logic**: Trade based on market sentiment indicators
- **Indicators**: Fear/Greed, social sentiment, news analysis
- **Best in**: Sentiment extremes (contrarian)
- **Timeframe**: Medium-term

### 10. Event-Driven Strategy
- **Logic**: Trade around known events (elections, announcements)
- **Indicators**: Event calendar, time decay, IV
- **Best in**: Prediction markets with clear resolution dates
- **Timeframe**: Event-specific

### 11. ML Classifier Strategy
- **Logic**: Use machine learning to predict direction
- **Indicators**: Gradient boosting on engineered features
- **Best in**: Markets with stable patterns
- **Timeframe**: Any

### 12. Market Regime Strategy
- **Logic**: Adapt to detected market regime
- **Indicators**: HMM states, volatility clustering
- **Best in**: All markets (adaptive)
- **Timeframe**: Any

---

## 🧠 Machine Learning Components

### 1. Tournament System (Thompson Sampling)
**Purpose**: Select which strategy signals to follow

**How it works**:
- Each strategy has a "score" based on its historical performance
- Uses Thompson Sampling (Beta distribution) for exploration/exploitation
- Good strategies get higher weights, bad ones get lower
- Correlation penalty prevents following similar strategies

**Key metrics**:
- Win rate per strategy
- Sharpe ratio per strategy
- Correlation between strategies

### 2. Meta-Learner
**Purpose**: Learn which strategy to trust in different conditions

**How it works**:
- Trains on historical outcomes
- Input features: market regime, recent strategy performance, time features
- Output: probability each strategy will be profitable
- Uses Gradient Boosting Classifier

**Training**:
- Minimum 30 samples per strategy
- Retrains every 50 new outcomes
- Cross-validated for robustness

### 3. Walk-Forward Optimizer
**Purpose**: Ensure ML models don't overfit

**How it works**:
- Rolling training window (30 days)
- Out-of-sample validation (7 days)
- Auto-retrains when performance degrades
- Detects and responds to regime changes

**Triggers for retrain**:
- Scheduled (every 24 hours)
- Performance degradation > 30%
- Regime change detected

### 4. Regime Detector
**Purpose**: Identify current market state

**Detected regimes**:
| Regime | Description |
|--------|-------------|
| `trending_up` | Strong upward price movement |
| `trending_down` | Strong downward price movement |
| `mean_reverting` | Prices oscillating around mean |
| `high_volatility` | Above-normal price swings |
| `low_volatility` | Calm, compressed ranges |
| `breakout` | Breaking out of consolidation |
| `consolidation` | Tight range, low activity |
| `crisis` | High vol + strong downtrend |
| `recovery` | Coming out of crisis |

**Methods used**:
- Hidden Markov Model (HMM)
- Volatility clustering (GARCH-like)
- Trend strength analysis (ADX-based)

### 5. Multi-Timeframe Analysis
**Purpose**: Combine signals across timeframes for confluence

**Timeframes analyzed**:
- 1 minute (scalping)
- 5 minutes (short-term)
- 15 minutes (intraday)
- 1 hour (swing)
- 4 hours (position)
- 1 day (macro)

**Output**:
- Direction (BUY/SELL/NEUTRAL)
- Confluence score (how many timeframes agree)
- Confidence (signal strength)

### 6. Sentiment Engine
**Purpose**: Incorporate external sentiment data

**Sources**:
- Social media mentions (simulated)
- News headlines (simulated)
- Fear/Greed index

**Contrarian signals**:
- Extreme fear (index < 15) → BUY signal
- Extreme greed (index > 85) → SELL signal

### 7. Portfolio Optimizer
**Purpose**: Optimal capital allocation across strategies

**Methods available**:
- Max Sharpe Ratio
- Minimum Variance
- Risk Parity
- Maximum Diversification
- Equal Weight

**Constraints**:
- Min weight: 2% per strategy
- Max weight: 40% per strategy
- Rebalances daily

---

## 💰 Risk Management

### Kelly Criterion
**Formula**: `f* = (bp - q) / b`
- b = reward/risk ratio
- p = probability of winning
- q = probability of losing

**Implementation**:
- Uses fractional Kelly (50% of full Kelly)
- Requires minimum 40% win rate
- Requires minimum 1:1 reward/risk
- Capped at 25% per position

### Drawdown Protection
| Drawdown | Action |
|----------|--------|
| 0-10% | Full capacity |
| 10-20% | Warning zone (80-100%) |
| 20-30% | Linear reduction |
| >30% | EMERGENCY STOP |

### Value at Risk (VaR)
- VaR 95%: Expected max loss 95% of the time
- VaR 99%: Expected max loss 99% of the time
- CVaR (Expected Shortfall): Average loss when VaR is breached

### Position Sizing
Final position size = MIN(Kelly, Risk Parity, Heat Limit, Max Position)

---

## 🗄️ Database Schema (TimescaleDB)

### Core Tables

#### `market_data` (Hypertable)
Stores raw market data from Polymarket.
```sql
CREATE TABLE market_data (
    time           TIMESTAMPTZ NOT NULL,
    market_id      TEXT NOT NULL,
    question       TEXT,
    outcome        TEXT,
    price          DOUBLE PRECISION,
    volume         DOUBLE PRECISION,
    liquidity      DOUBLE PRECISION,
    bid_price      DOUBLE PRECISION,
    ask_price      DOUBLE PRECISION,
    spread         DOUBLE PRECISION,
    PRIMARY KEY (time, market_id)
);
```

#### `signals` (Hypertable)
Stores all generated trading signals.
```sql
CREATE TABLE signals (
    time           TIMESTAMPTZ NOT NULL,
    market_id      TEXT NOT NULL,
    strategy_name  TEXT NOT NULL,
    signal_type    TEXT,           -- BUY, SELL, HOLD
    confidence     DOUBLE PRECISION,
    features       JSONB,
    metadata       JSONB,
    PRIMARY KEY (time, market_id, strategy_name)
);
```

#### `trades` (Hypertable)
Stores executed trades.
```sql
CREATE TABLE trades (
    time           TIMESTAMPTZ NOT NULL,
    trade_id       TEXT NOT NULL,
    market_id      TEXT NOT NULL,
    strategy_name  TEXT,
    side           TEXT,           -- BUY, SELL
    price          DOUBLE PRECISION,
    size           DOUBLE PRECISION,
    value          DOUBLE PRECISION,
    status         TEXT,           -- PENDING, FILLED, CANCELLED
    pnl            DOUBLE PRECISION,
    PRIMARY KEY (time, trade_id)
);
```

#### `positions` (Hypertable)
Tracks open positions.
```sql
CREATE TABLE positions (
    time           TIMESTAMPTZ NOT NULL,
    market_id      TEXT NOT NULL,
    strategy_name  TEXT,
    side           TEXT,
    entry_price    DOUBLE PRECISION,
    current_price  DOUBLE PRECISION,
    size           DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    realized_pnl   DOUBLE PRECISION,
    PRIMARY KEY (time, market_id)
);
```

#### `portfolio_snapshots` (Hypertable)
Periodic portfolio snapshots for performance tracking.
```sql
CREATE TABLE portfolio_snapshots (
    time           TIMESTAMPTZ NOT NULL,
    total_value    DOUBLE PRECISION,
    cash           DOUBLE PRECISION,
    positions_value DOUBLE PRECISION,
    total_pnl      DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    realized_pnl   DOUBLE PRECISION,
    sharpe_ratio   DOUBLE PRECISION,
    max_drawdown   DOUBLE PRECISION,
    win_rate       DOUBLE PRECISION,
    PRIMARY KEY (time)
);
```

### ML Tables

#### `ml_features` (Hypertable)
Stores computed features for ML training.
```sql
CREATE TABLE ml_features (
    time           TIMESTAMPTZ NOT NULL,
    market_id      TEXT NOT NULL,
    features       JSONB,          -- All computed features
    regime         TEXT,           -- Detected market regime
    PRIMARY KEY (time, market_id)
);
```

#### `ml_predictions` (Hypertable)
Stores ML model predictions.
```sql
CREATE TABLE ml_predictions (
    time           TIMESTAMPTZ NOT NULL,
    market_id      TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    prediction     DOUBLE PRECISION,
    confidence     DOUBLE PRECISION,
    features_hash  TEXT,
    PRIMARY KEY (time, market_id, model_name)
);
```

### Strategy Tables

#### `strategy_allocations` (Hypertable)
Tracks strategy weight allocations over time.
```sql
CREATE TABLE strategy_allocations (
    time           TIMESTAMPTZ NOT NULL,
    strategy_name  TEXT NOT NULL,
    weight         DOUBLE PRECISION,
    score          DOUBLE PRECISION,
    win_rate       DOUBLE PRECISION,
    sharpe         DOUBLE PRECISION,
    PRIMARY KEY (time, strategy_name)
);
```

#### `strategy_correlations` (Hypertable)
Tracks correlations between strategies.
```sql
CREATE TABLE strategy_correlations (
    time           TIMESTAMPTZ NOT NULL,
    strategy_1     TEXT NOT NULL,
    strategy_2     TEXT NOT NULL,
    correlation    DOUBLE PRECISION,
    PRIMARY KEY (time, strategy_1, strategy_2)
);
```

---

## 📈 What to Expect

### Startup Phase (First 1-7 Days)
- System collects data and learns market patterns
- All strategies start with equal weights
- Few trades as confidence builds
- Meta-learner and Walk-Forward need minimum samples

**Expected metrics**:
- Signals: 100-500/day across all strategies
- Trades: 5-20/day (conservative)
- Win rate: Uncertain (learning)
- Sharpe: Not meaningful yet

### Learning Phase (7-30 Days)
- Thompson Sampling starts differentiating strategies
- Meta-learner begins predicting which strategy to trust
- Regime detector calibrates to market patterns
- Portfolio optimizer starts rebalancing

**Expected metrics**:
- Strategies diverge in weights (10%-40% range)
- Win rate stabilizes (target: 50-60%)
- Sharpe starts becoming meaningful
- Drawdowns should be limited by risk management

### Mature Phase (30+ Days)
- Full ML pipeline operational
- Walk-forward retraining active
- Strong strategies get more capital
- System adapts to regime changes

**Expected metrics**:
- Top 3-4 strategies dominate
- Win rate: 52-58%
- Sharpe ratio: 1.0-2.5 (annualized)
- Max drawdown: <20%
- Calmar ratio: >1.0

### Key Performance Indicators (KPIs)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Win Rate | >50% | 52-55% | >58% |
| Sharpe Ratio | >1.0 | 1.5-2.0 | >2.5 |
| Max Drawdown | <20% | <15% | <10% |
| Profit Factor | >1.2 | 1.5-2.0 | >2.5 |
| Calmar Ratio | >1.0 | 1.5-2.0 | >3.0 |

### GO-LIVE Criteria
The system signals "READY FOR LIVE" when:
- Sharpe Ratio ≥ 1.75
- Max Drawdown ≤ 10%
- Total PnL > 0
- Win Rate > 52%
- At least 100 trades completed

---

## 🖥️ Monitoring & Dashboards

### Grafana Dashboards

**1. Strategy Tournament** (`/d/polymania-leaderboard`)
- Strategy leaderboard with rankings
- Signals by strategy (bar chart)
- Weight evolution over time
- Signal share pie chart
- Recent signals table
- Recent trades table

**2. Main Dashboard** (if available)
- Portfolio value over time
- Current positions
- PnL tracking
- Risk metrics (VaR, Sharpe)

### Key Things to Monitor

**Daily**:
- Strategy weights (are they changing appropriately?)
- Win rate per strategy
- Portfolio drawdown
- Number of trades

**Weekly**:
- Sharpe ratio trend
- Strategy correlations
- Meta-learner accuracy
- Walk-forward retrain events

**Monthly**:
- Overall PnL
- Strategy performance comparison
- Regime detection accuracy
- System health

---

## 🔧 Configuration

### Main Configuration Options
Located in `/app/config/`:

```yaml
# Trading settings
trading:
  min_confidence: 0.6       # Minimum signal confidence to trade
  max_positions: 10         # Maximum concurrent positions
  position_size_pct: 0.10   # Default position size
  
# Risk settings
risk:
  max_drawdown: 0.20        # Stop trading at 20% drawdown
  max_position_pct: 0.20    # Max 20% in one position
  target_volatility: 0.15   # 15% annualized target vol

# ML settings
ml:
  retrain_interval: 24      # Hours between retraining
  min_train_samples: 100    # Minimum samples to train
  lookback_days: 30         # Training data lookback
```

---

## 🚨 Alerts & Warnings

### Critical Alerts
- **EMERGENCY STOP**: Drawdown > 30%
- **Connection Lost**: Database or API disconnection
- **Model Failure**: ML prediction errors

### Warnings
- **High Correlation**: Strategies becoming too similar
- **Low Activity**: Few signals generated
- **Degraded Performance**: Win rate dropping significantly

---

## 📝 Summary

PolyMania is a comprehensive algorithmic trading system that:

1. **Collects** real-time market data from Polymarket
2. **Analyzes** markets with 12 independent strategies
3. **Selects** best signals using ML tournament system
4. **Sizes** positions using Kelly criterion and risk parity
5. **Executes** trades with strict risk controls
6. **Learns** continuously from outcomes

The system is designed to be:
- **Robust**: Multiple strategies provide diversification
- **Adaptive**: ML learns from changing markets
- **Safe**: Multiple layers of risk management
- **Transparent**: Full logging and monitoring

**Access Points**:
- Grafana: `http://158.178.130.222:3000`
- Dashboard: `polymania-leaderboard`
- Default credentials: Check `.env` file on VM

---

*Documentation generated: January 18, 2026*
*System Version: 2.0 (Tournament + Advanced ML)*

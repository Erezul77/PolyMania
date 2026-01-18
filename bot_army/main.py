"""
PolyMania Bot Army - Main Entry Point
=====================================

Orchestrates all components:
- Data collectors
- ML analyzers
- Trading strategies
- Execution engine
- Risk management
"""

import asyncio
import logging
import signal
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any
import uuid
import numpy as np

from .core.base import BotConfig
from .core.database import TimeseriesDB
from .core.cache import CacheManager
from .core.logger import setup_logger
from .core.events import Trade, OrderStatus, Signal, SignalType
from .collectors.polymarket import PolymarketCollector
from .collectors.external_signals import ExternalSignalsCollector
from .collectors.external_sources import ExternalSourcesCollector
from .collectors.trades import TradesCollector
from .collectors.events import EventsCollector
from .analyzers.feature_engine import FeatureEngine
from .analyzers.ml_models import MLEngine, SKLEARN_AVAILABLE
from .analyzers.pattern_detector import PatternDetector
from .analyzers.regime_detector import RegimeDetector
from .analyzers.sentiment import SentimentAnalyzer
from .strategies.ensemble import EnsembleStrategy
from .strategies.tournament import StrategyTournament
from .strategies import create_strategy_zoo
from .execution.risk_manager import RiskManager, RiskConfig
from .execution.order_engine import OrderEngine, ExecutionMode
from .execution.portfolio import PortfolioManager, PortfolioConfig

# Setup logging
logger = setup_logger("bot_army", log_file="logs/bot_army.log")


class BotArmy:
    """
    Main orchestrator for the Bot Army trading system.
    """
    
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "polymania",
        db_user: str = "polymania",
        db_password: str = "polymania",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        execution_mode: str = "paper"
    ):
        logger.info("Initializing Bot Army...")
        
        # Infrastructure
        self.db = TimeseriesDB(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )
        self.cache = CacheManager(host=redis_host, port=redis_port)
        
        # Collectors
        self.market_collector = PolymarketCollector(
            BotConfig(name="market_collector", interval_seconds=10),
            db_manager=self.db,
            cache_manager=self.cache
        )
        self.trades_collector = TradesCollector(
            BotConfig(name="trades_collector", interval_seconds=5),
            db_manager=self.db,
            cache_manager=self.cache
        )
        self.events_collector = EventsCollector(
            BotConfig(name="events_collector", interval_seconds=60),
            db_manager=self.db,
            cache_manager=self.cache
        )
        
        # Analyzers
        self.feature_engine = FeatureEngine()
        self.ml_engine = MLEngine()
        self.pattern_detector = PatternDetector()
        self.regime_detector = RegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        self.external_signals = ExternalSignalsCollector(
            paths=[],
            sentiment_analyzer=self.sentiment_analyzer,
            db_manager=self.db
        )
        self.external_sources = ExternalSourcesCollector(
            db_manager=self.db,
            sentiment_analyzer=self.sentiment_analyzer,
            newsapi_key="",
            news_keywords="",
            fred_api_key="",
            fred_series="",
            weather_locations="",
            min_interval_seconds=600
        )
        
        # Strategy Tournament (12 independent strategies competing)
        self.strategy_zoo = create_strategy_zoo()
        self.tournament = StrategyTournament(
            strategies=self.strategy_zoo,
            min_weight=0.02,
            max_weight=0.35,
            decay_factor=0.95,
            correlation_threshold=0.7
        )
        # Legacy ensemble for backwards compatibility
        self.strategy = EnsembleStrategy()
        
        # Execution
        mode = ExecutionMode.PAPER if execution_mode == "paper" else ExecutionMode.LIVE
        self.order_engine = OrderEngine(mode=mode)
        self.risk_manager = RiskManager(RiskConfig())
        self.portfolio = PortfolioManager(PortfolioConfig())
        
        # State
        self._running = False
        self._cycle_count = 0
        self._signals_generated = 0
        self._trades_executed = 0
        self._strategy_pnl_history = defaultdict(list)
        self._ml_ready = False
        self._ml_last_train = {}
    
    async def start(self):
        """Start the Bot Army."""
        logger.info("Starting Bot Army...")
        
        try:
            # Connect to infrastructure
            logger.info("Connecting to database...")
            await self.db.connect()
            logger.info("Database connected!")
            
            logger.info("Connecting to cache...")
            await self.cache.connect()
            logger.info("Cache connected!")
            
            # Initialize database schema
            logger.info("Initializing database schema...")
            await self.db.init_schema()
            logger.info("Schema initialized!")

            # Load ML models if available
            self._load_ml_models()
            
            # Start collectors
            logger.info("Starting collectors...")
            await self.market_collector.start()
            await self.trades_collector.start()
            await self.events_collector.start()
            logger.info("Collectors started!")
            
            self._running = True
            
            # Main trading loop
            logger.info("Starting trading loop...")
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting Bot Army: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the Bot Army."""
        logger.info("Stopping Bot Army...")
        self._running = False
        
        # Stop collectors
        await self.market_collector.stop()
        await self.trades_collector.stop()
        await self.events_collector.stop()
        
        # Disconnect
        await self.db.disconnect()
        await self.cache.disconnect()
        
        logger.info("Bot Army stopped")
    
    async def _trading_loop(self):
        """Main trading loop."""
        while self._running:
            try:
                self._cycle_count += 1
                cycle_start = datetime.utcnow()
                
                # Get market data
                markets = self.market_collector.markets_cache
                
                for market_id, market_data in markets.items():
                    await self._process_market(market_id, market_data)
                
                # Update portfolio
                await self._update_portfolio()
                
                # Check stop losses
                await self._check_exits()
                
                # Publish status
                await self._publish_status()

                # External signals + correlations
                if self._cycle_count % 6 == 0:
                    await self._collect_external_signals()
                if self._cycle_count % 30 == 0:
                    await self.db.refresh_external_correlations(window_hours=168)
                await self._collect_external_sources()
                
                # Wait for next cycle
                elapsed = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(1, 10 - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)

    async def _collect_external_signals(self):
        """Ingest external signals from CSV sources."""
        import os
        paths = os.getenv(
            "EXTERNAL_SIGNALS_PATHS",
            "/app/data/signals.csv,/app/data/telegram_hits.csv"
        )
        self.external_signals.paths = [p.strip() for p in paths.split(",") if p.strip()]
        ingested = await self.external_signals.poll()
        if ingested:
            logger.info(f"Ingested {ingested} external signals")

    async def _collect_external_sources(self):
        """Ingest news/macro/weather sources."""
        import os
        self.external_sources.newsapi_key = os.getenv("NEWS_API_KEY", "")
        self.external_sources.news_keywords = os.getenv("NEWS_KEYWORDS", "")
        self.external_sources.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.external_sources.fred_series = os.getenv("FRED_SERIES", "")
        self.external_sources.weather_locations = os.getenv("WEATHER_LOCATIONS", "")
        interval = int(os.getenv("EXTERNAL_SOURCES_INTERVAL_SECONDS", "600"))
        self.external_sources.min_interval_seconds = interval
        ingested = await self.external_sources.poll()
        if ingested:
            logger.info(f"Ingested {ingested} external source signals")

    def _load_ml_models(self):
        """Load saved ML models if available."""
        try:
            self.ml_engine.load_model("direction_rf")
            self._ml_ready = self._is_ml_ready()
        except Exception as e:
            logger.debug(f"ML model load failed: {e}")

    def _is_ml_ready(self) -> bool:
        """Check if ML model is trained and ready."""
        model = self.ml_engine.models.get("direction_rf")
        if model is None:
            return False
        return bool(getattr(model, "classes_", None)) or bool(getattr(model, "n_estimators_", None))

    def _build_training_samples(
        self,
        market_id: str,
        price_history: list,
        window_size: int,
        horizon: int,
        max_samples: int
    ):
        """Build ML training samples from price history."""
        if len(price_history) < window_size + horizon + 5:
            return [], [], []
        
        start_idx = max(window_size, len(price_history) - max_samples - horizon)
        end_idx = len(price_history) - horizon
        X = []
        y = []
        feature_names = None
        
        for i in range(start_idx, end_idx):
            window_history = price_history[:i]
            feature_set = self.feature_engine.compute_all_features(
                market_id=market_id,
                price_history=window_history,
                orderbook={},
                trades=[]
            )
            if not feature_set.features:
                continue
            
            if feature_names is None:
                feature_names = sorted(feature_set.features.keys())
            
            price_now = price_history[i - 1].get("price")
            price_future = price_history[i - 1 + horizon].get("price")
            if not price_now or not price_future:
                continue
            
            vector = [feature_set.features.get(name, 0.0) for name in feature_names]
            X.append(vector)
            y.append(1 if price_future > price_now else 0)
        
        return X, y, feature_names or []

    async def _maybe_train_ml(self, market_id: str, price_history: list):
        """Train ML model periodically using recent price history."""
        if not SKLEARN_AVAILABLE:
            return
        
        import os
        min_samples = int(os.getenv("ML_TRAIN_MIN_SAMPLES", "80"))
        max_samples = int(os.getenv("ML_TRAIN_MAX_SAMPLES", "200"))
        horizon = int(os.getenv("ML_TRAIN_HORIZON", "3"))
        window = int(os.getenv("ML_TRAIN_WINDOW", "20"))
        interval_cycles = int(os.getenv("ML_TRAIN_INTERVAL_CYCLES", "60"))
        
        last_cycle = self._ml_last_train.get(market_id, 0)
        if self._ml_ready and (self._cycle_count - last_cycle) < interval_cycles:
            return
        
        X, y, feature_names = self._build_training_samples(
            market_id=market_id,
            price_history=price_history,
            window_size=window,
            horizon=horizon,
            max_samples=max_samples
        )
        
        if len(y) < min_samples:
            return
        
        metrics = self.ml_engine.train(
            np.array(X),
            np.array(y),
            model_name="direction_rf",
            feature_names=feature_names
        )
        self.ml_engine.save_model("direction_rf")
        self._ml_ready = True
        self._ml_last_train[market_id] = self._cycle_count
        logger.info(
            f"Trained ML model for {market_id} with {len(y)} samples; "
            f"cv_mean={metrics.get('cv_mean', 0):.3f}"
        )
    
    async def _process_market(self, market_id: str, market_data):
        """Process a single market."""
        try:
            # Get price history
            price_history = await self.db.get_market_history(market_id, hours=24)
            price_history = list(reversed(price_history))
            prices = [p.get("price", 0) for p in price_history if p.get("price")]
            
            if len(prices) < 20:
                return
            
            # Compute features
            features = self.feature_engine.compute_all_features(
                market_id=market_id,
                price_history=price_history,
                orderbook=market_data.to_dict() if hasattr(market_data, 'to_dict') else {},
                trades=self.trades_collector.get_recent_trades(market_id)
            )

            # Detect patterns
            patterns = self.pattern_detector.detect_all(market_id, prices)
            
            # Detect regime
            regime = self.regime_detector.detect(market_id, prices)
            self.strategy.set_regime(regime.primary.value)
            
            # Set regime for regime-aware strategies
            for strat in self.strategy_zoo.values():
                if hasattr(strat, "set_regime"):
                    strat.set_regime(regime.primary.value, 0.5)

            # Train ML periodically
            await self._maybe_train_ml(market_id, price_history)
            
            # ML prediction
            if features.features and self._ml_ready:
                prediction = self.ml_engine.predict(
                    features.to_vector(),
                    market_id
                )
                features.features["ml_prediction"] = prediction.prediction
                features.features["ml_confidence"] = prediction.confidence
                await self.db.store_ml_prediction(prediction)
            elif features.features:
                features.features["ml_prediction"] = 0.5
                features.features["ml_confidence"] = 0.0

            # Persist ML features (including ML prediction fields if present)
            await self.db.store_ml_features(market_id, features.features, features.timestamp)
            
            # === TOURNAMENT: Run all 12 strategies independently ===
            orderbook_dict = market_data.to_dict() if hasattr(market_data, 'to_dict') else {}
            trades_list = [t.to_dict() for t in self.trades_collector.get_recent_trades(market_id)]
            
            # Get signals from all strategies
            all_results = self.tournament.analyze_all(
                market_id=market_id,
                features=features.features,
                price_history=prices,
                orderbook=orderbook_dict,
                trades=trades_list
            )
            
            # Store ALL signals (for tracking/analysis)
            for result in all_results:
                if hasattr(market_data, "event_id"):
                    result.signal.event_id = market_data.event_id
                await self.db.store_signal(result.signal)
                self._signals_generated += 1
            
            # Select best signals (Thompson Sampling + correlation filter)
            best_results = self.tournament.select_best_signals(all_results, max_signals=2)
            
            # Execute the best signals
            for result in best_results:
                await self._execute_signal(result.signal)
            
        except Exception as e:
            logger.debug(f"Error processing {market_id}: {e}")
    
    async def _execute_signal(self, signal):
        """Execute a trading signal."""
        # Risk check
        approved, reason, size = self.risk_manager.check_signal(
            signal,
            self.portfolio.total_value,
            self.order_engine.positions
        )
        
        if not approved:
            logger.info(f"Signal rejected: {reason}")
            return
        
        # Execute
        result = await self.order_engine.execute_signal(signal, size)
        
        if result.status.value == "FILLED":
            self._trades_executed += 1
            realized_pnl = self.portfolio.update_position(
                signal.market_id,
                result.filled_quantity,
                result.filled_price,
                "BUY" if signal.signal_type.value == "BUY" else "SELL",
                strategy_name=getattr(signal, "strategy_name", "")
            )
            await self._record_trade(signal, result, realized_pnl)
            logger.info(
                f"Trade executed: {signal.signal_type.value} "
                f"{result.filled_quantity:.2f} @ {result.filled_price:.4f}"
            )

            if realized_pnl is not None:
                strategy_name = getattr(signal, "strategy_name", "") or "unknown"
                history = self._strategy_pnl_history[strategy_name]
                history.append(realized_pnl)
                self._strategy_pnl_history[strategy_name] = history[-200:]
                
                # Update tournament (Thompson Sampling scores)
                self.tournament.record_outcome(strategy_name, realized_pnl)
                
                # Legacy ensemble update
                self.strategy.update_strategy_performance(
                    strategy_name, realized_pnl, realized_pnl > 0
                )
    
    async def _update_portfolio(self):
        """Update portfolio with current prices."""
        prices = {
            market_id: data.price
            for market_id, data in self.market_collector.markets_cache.items()
        }
        
        self.portfolio.update_prices(prices)
        self.order_engine.update_prices(prices)
        self.risk_manager.update_metrics(
            self.portfolio.total_value,
            self.order_engine.positions
        )
        
        snapshot = self.portfolio.take_snapshot()
        await self.db.store_portfolio_snapshot(snapshot)
        positions = list({**self.order_engine.positions, **self.order_engine._paper_positions}.values())
        await self.db.replace_positions(positions)
        
        if self._cycle_count % 6 == 0:
            await self._store_strategy_allocations()
        if self._cycle_count % 30 == 0:
            await self._store_strategy_correlations()
    
    async def _check_exits(self):
        """Check and execute stop losses / take profits."""
        prices = {
            market_id: data.price
            for market_id, data in self.market_collector.markets_cache.items()
        }
        
        positions_to_close = self.order_engine.check_stops(prices)
        
        for market_id in positions_to_close:
            if market_id in prices:
                result = await self.order_engine.close_position(market_id, prices[market_id])
                if result and result.status == OrderStatus.FILLED:
                    signal = Signal(
                        market_id=market_id,
                        signal_type=SignalType.SELL,
                        confidence=1.0,
                        price_target=prices[market_id],
                        strategy_name="risk_exit",
                        metadata={"reason": "stop_or_take_profit"}
                    )
                    await self._record_trade(signal, result)
    
    async def _publish_status(self):
        """Publish system status to cache."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle_count": self._cycle_count,
            "signals_generated": self._signals_generated,
            "trades_executed": self._trades_executed,
            "portfolio_value": self.portfolio.total_value,
            "position_count": len(self.order_engine.positions),
            "risk_level": self.risk_manager.risk_level.value,
            "collectors": {
                "markets": self.market_collector.stats,
                "trades": self.trades_collector.stats,
                "events": self.events_collector.stats
            },
            "tournament": self.tournament.get_stats(),
            "strategy": self.strategy.stats,
            "execution": self.order_engine.stats
        }
        
        await self.cache.set("bot_army:status", status, ttl=60)

    async def _record_trade(self, signal, result, realized_pnl: float = None):
        """Persist executed trade to the database."""
        if result.status != OrderStatus.FILLED:
            return
        
        trade = Trade(
            trade_id=result.trade_id or str(uuid.uuid4()),
            signal_id=signal.signal_id,
            market_id=signal.market_id,
            event_id=getattr(signal, "event_id", ""),
            side="BUY" if signal.signal_type.value == "BUY" else "SELL",
            quantity=result.filled_quantity,
            price=result.filled_price,
            value=result.filled_quantity * result.filled_price,
            fees=result.fees,
            pnl=realized_pnl,
            status=result.status,
            strategy_name=getattr(signal, "strategy_name", ""),
            executed_at=result.executed_at,
            metadata=result.to_dict() if hasattr(result, "to_dict") else {}
        )
        await self.db.store_trade(trade)

    async def _store_strategy_allocations(self):
        """Persist strategy weights alongside performance metrics (from tournament)."""
        # Get weights from tournament (12 strategies)
        weights = self.tournament.get_weights()
        if not weights:
            return
        
        performance = await self.db.get_strategy_performance()
        perf_map = {row.get("strategy_name"): row for row in performance}
        
        allocations = {}
        for name, weight in weights.items():
            perf = perf_map.get(name, {})
            tournament_stats = self.tournament.stats.get(name)
            allocations[name] = {
                "weight": weight,
                "trade_count": perf.get("trade_count") or (tournament_stats.trades if tournament_stats else 0),
                "win_rate": perf.get("win_rate") or (tournament_stats.win_rate if tournament_stats else 0),
                "total_pnl": perf.get("total_pnl") or (tournament_stats.total_pnl if tournament_stats else 0)
            }
        
        await self.db.store_strategy_allocations(datetime.utcnow(), allocations)

    async def _store_strategy_correlations(self):
        """Persist strategy return correlations for analysis."""
        names = list(self._strategy_pnl_history.keys())
        if len(names) < 2:
            return
        
        correlations = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                series_a = self._strategy_pnl_history.get(a, [])
                series_b = self._strategy_pnl_history.get(b, [])
                n = min(len(series_a), len(series_b))
                if n < 5:
                    continue
                window_a = np.array(series_a[-n:], dtype=float)
                window_b = np.array(series_b[-n:], dtype=float)
                if np.std(window_a) == 0 or np.std(window_b) == 0:
                    continue
                corr = float(np.corrcoef(window_a, window_b)[0, 1])
                correlations.append({
                    "strategy_a": a,
                    "strategy_b": b,
                    "correlation": corr
                })
        
        if correlations:
            await self.db.store_strategy_correlations(datetime.utcnow(), correlations)
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "signals_generated": self._signals_generated,
            "trades_executed": self._trades_executed,
            "portfolio": self.portfolio.get_performance_summary(),
            "risk": self.risk_manager.metrics.to_dict(),
            "strategy": self.strategy.stats
        }


async def main():
    """Main entry point."""
    import os
    
    # Get configuration from environment
    db_host = os.getenv("DB_HOST", "timescaledb")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "polymania")
    db_user = os.getenv("DB_USER", "polymania")
    db_password = os.getenv("DB_PASSWORD", "polymania")
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    execution_mode = os.getenv("EXECUTION_MODE", "paper")
    
    # Create bot army
    army = BotArmy(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        redis_host=redis_host,
        redis_port=redis_port,
        execution_mode=execution_mode
    )
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def shutdown_handler():
        asyncio.create_task(army.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)
    
    # Start
    try:
        await army.start()
    except KeyboardInterrupt:
        await army.stop()


if __name__ == "__main__":
    asyncio.run(main())

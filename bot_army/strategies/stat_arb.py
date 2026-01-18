"""Statistical Arbitrage Strategy."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import Signal, SignalType

logger = logging.getLogger("strategy.stat_arb")


class StatArbStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy.
    
    Identifies mispricings and arbitrage opportunities between
    related markets or within single markets using statistical analysis.
    
    Strategies:
    1. Mean reversion within single market
    2. Pair trading between correlated markets
    3. Cross-sectional arbitrage
    4. Probability mispricing detection
    """
    
    DEFAULT_PARAMS = {
        "zscore_entry": 1.0,
        "zscore_exit": 0.5,
        "lookback_window": 50,
        "min_correlation": 0.7,
        "half_life_max": 30,
        "probability_edge_threshold": 0.02
    }
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="stat_arb",
                min_confidence=0.40,
                position_size_pct=0.1,
                stop_loss_pct=0.15,
                take_profit_pct=0.1,
                params={}
            )
        
        config.params = {**self.DEFAULT_PARAMS, **config.params}
        super().__init__(config)
        
        # Pairs tracking
        self._pairs: Dict[str, Dict] = {}
        self._correlations: Dict[Tuple[str, str], float] = {}
        self._spreads: Dict[str, List[float]] = defaultdict(list)
    
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        """Analyze market for stat arb opportunities."""
        
        if not self.should_generate_signal(market_id):
            return None
        
        if len(price_history) < self.config.params["lookback_window"]:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        reasoning = []
        signals = []
        
        # 1. Mean reversion analysis
        mean_rev_signal = self._analyze_mean_reversion(market_id, prices, features)
        if mean_rev_signal:
            signals.append(mean_rev_signal)
            reasoning.append(f"Mean reversion: z-score {mean_rev_signal['zscore']:.2f}")
        
        # 2. Probability mispricing
        prob_signal = self._analyze_probability_mispricing(current_price, features)
        if prob_signal:
            signals.append(prob_signal)
            reasoning.append(f"Probability edge: {prob_signal['edge']:.1%}")
        
        # 3. Technical dislocation
        tech_signal = self._analyze_technical_dislocation(prices, features)
        if tech_signal:
            signals.append(tech_signal)
            reasoning.append(f"Technical dislocation: {tech_signal['type']}")
        
        if not signals:
            return None
        
        # Combine signals
        combined = self._combine_signals(signals)
        
        signal_type = combined["signal_type"]
        confidence = combined["confidence"]
        
        if signal_type == SignalType.HOLD or confidence < self.config.min_confidence:
            return None
        
        # Create signal
        signal = self.create_signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            features={
                "zscore": combined.get("zscore", 0),
                "edge": combined.get("edge", 0),
                "half_life": combined.get("half_life", 0)
            }
        )
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=confidence,
            reasoning=reasoning,
            metadata=combined
        )
    
    def _analyze_mean_reversion(
        self,
        market_id: str,
        prices: np.ndarray,
        features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Analyze mean reversion opportunity."""
        window = self.config.params["lookback_window"]
        
        # Calculate statistics
        mean = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        
        if std == 0:
            return None
        
        zscore = (prices[-1] - mean) / std
        
        # Check half-life (mean reversion speed)
        half_life = self._calculate_half_life(prices[-window:])
        
        if half_life > self.config.params["half_life_max"]:
            return None  # Too slow to mean revert
        
        # Store spread for tracking
        self._spreads[market_id].append(zscore)
        self._spreads[market_id] = self._spreads[market_id][-100:]
        
        entry_threshold = self.config.params["zscore_entry"]
        
        if zscore > entry_threshold:
            return {
                "type": "mean_reversion",
                "signal_type": SignalType.SELL,
                "confidence": min(0.9, abs(zscore) / 3),
                "zscore": zscore,
                "half_life": half_life,
                "target": mean
            }
        elif zscore < -entry_threshold:
            return {
                "type": "mean_reversion",
                "signal_type": SignalType.BUY,
                "confidence": min(0.9, abs(zscore) / 3),
                "zscore": zscore,
                "half_life": half_life,
                "target": mean
            }
        
        return None
    
    def _analyze_probability_mispricing(
        self,
        current_price: float,
        features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect probability mispricing.
        
        In prediction markets, prices should reflect true probabilities.
        Look for deviations from fair value.
        """
        edge_threshold = self.config.params["probability_edge_threshold"]
        
        # Get sentiment and other signals
        sentiment = features.get("sentiment_score", 0)
        momentum = features.get("momentum", 0)
        flow = features.get("trade_net_flow", 0)
        
        # Estimate fair probability
        # Using multiple signals to estimate true probability
        signal_estimate = 0.5 + sentiment * 0.1 + momentum * 0.05
        signal_estimate = max(0.1, min(0.9, signal_estimate))
        
        # Calculate edge
        if current_price < signal_estimate - edge_threshold:
            # Underpriced - BUY
            edge = signal_estimate - current_price
            return {
                "type": "probability_edge",
                "signal_type": SignalType.BUY,
                "confidence": min(0.85, edge * 5),
                "edge": edge,
                "fair_value": signal_estimate
            }
        elif current_price > signal_estimate + edge_threshold:
            # Overpriced - SELL
            edge = current_price - signal_estimate
            return {
                "type": "probability_edge",
                "signal_type": SignalType.SELL,
                "confidence": min(0.85, edge * 5),
                "edge": edge,
                "fair_value": signal_estimate
            }
        
        return None
    
    def _analyze_technical_dislocation(
        self,
        prices: np.ndarray,
        features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect technical price dislocations."""
        if len(prices) < 30:
            return None
        
        current = prices[-1]
        
        # Bollinger Band position
        bb_position = features.get("bb_position", 0.5)
        
        # RSI extremes
        rsi = features.get("rsi", 50)
        
        # Price vs SMA deviation
        sma_20 = np.mean(prices[-20:])
        deviation = (current - sma_20) / sma_20
        
        # Detect dislocation
        if bb_position > 0.95 and rsi > 75:
            return {
                "type": "technical_overbought",
                "signal_type": SignalType.SELL,
                "confidence": 0.65,
                "bb_position": bb_position,
                "rsi": rsi,
                "deviation": deviation
            }
        elif bb_position < 0.05 and rsi < 25:
            return {
                "type": "technical_oversold",
                "signal_type": SignalType.BUY,
                "confidence": 0.65,
                "bb_position": bb_position,
                "rsi": rsi,
                "deviation": deviation
            }
        
        return None
    
    def _calculate_half_life(self, prices: np.ndarray) -> float:
        """Calculate half-life of mean reversion."""
        if len(prices) < 10:
            return float('inf')
        
        # Ornstein-Uhlenbeck process estimation
        spread = prices - np.mean(prices)
        lag_spread = spread[:-1]
        diff_spread = np.diff(spread)
        
        if np.std(lag_spread) == 0:
            return float('inf')
        
        # Linear regression to estimate theta
        beta = np.sum(lag_spread * diff_spread) / np.sum(lag_spread ** 2)
        
        if beta >= 0:
            return float('inf')  # No mean reversion
        
        half_life = -np.log(2) / beta
        return half_life
    
    def _combine_signals(self, signals: List[Dict]) -> Dict[str, Any]:
        """Combine multiple arb signals."""
        if not signals:
            return {"signal_type": SignalType.HOLD, "confidence": 0}
        
        # Weighted voting
        buy_score = 0
        sell_score = 0
        
        weights = {
            "mean_reversion": 0.4,
            "probability_edge": 0.35,
            "technical_overbought": 0.25,
            "technical_oversold": 0.25
        }
        
        for sig in signals:
            weight = weights.get(sig["type"], 0.2)
            if sig["signal_type"] == SignalType.BUY:
                buy_score += sig["confidence"] * weight
            elif sig["signal_type"] == SignalType.SELL:
                sell_score += sig["confidence"] * weight
        
        # Determine final signal
        if buy_score > sell_score and buy_score > 0.3:
            return {
                "signal_type": SignalType.BUY,
                "confidence": min(0.9, buy_score),
                "buy_score": buy_score,
                "sell_score": sell_score,
                **signals[0]  # Include details from primary signal
            }
        elif sell_score > buy_score and sell_score > 0.3:
            return {
                "signal_type": SignalType.SELL,
                "confidence": min(0.9, sell_score),
                "buy_score": buy_score,
                "sell_score": sell_score,
                **signals[0]
            }
        
        return {"signal_type": SignalType.HOLD, "confidence": 0}
    
    def add_pair(
        self,
        market_a: str,
        market_b: str,
        correlation: float
    ):
        """Add a tradeable pair."""
        if correlation >= self.config.params["min_correlation"]:
            pair_id = f"{market_a}_{market_b}"
            self._pairs[pair_id] = {
                "market_a": market_a,
                "market_b": market_b,
                "correlation": correlation
            }
            self._correlations[(market_a, market_b)] = correlation
    
    def analyze_pairs(
        self,
        prices_a: List[float],
        prices_b: List[float],
        pair_id: str
    ) -> Optional[StrategyResult]:
        """Analyze pair for arbitrage opportunity."""
        if len(prices_a) != len(prices_b) or len(prices_a) < 30:
            return None
        
        # Calculate spread
        prices_a = np.array(prices_a)
        prices_b = np.array(prices_b)
        
        # Hedge ratio (beta)
        beta = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)
        
        # Spread = A - beta * B
        spread = prices_a - beta * prices_b
        
        # Z-score
        zscore = (spread[-1] - np.mean(spread)) / np.std(spread)
        
        entry = self.config.params["zscore_entry"]
        
        if zscore > entry:
            # Spread too high - sell A, buy B
            return StrategyResult(
                signal=Signal(
                    market_id=pair_id,
                    signal_type=SignalType.SELL,
                    confidence=min(0.85, abs(zscore) / 3)
                ),
                strategy_name=self.config.name,
                score=abs(zscore),
                reasoning=[f"Pair zscore: {zscore:.2f} - Sell {pair_id.split('_')[0]}, Buy {pair_id.split('_')[1]}"],
                metadata={"zscore": zscore, "beta": beta, "action": "sell_a_buy_b"}
            )
        elif zscore < -entry:
            # Spread too low - buy A, sell B
            return StrategyResult(
                signal=Signal(
                    market_id=pair_id,
                    signal_type=SignalType.BUY,
                    confidence=min(0.85, abs(zscore) / 3)
                ),
                strategy_name=self.config.name,
                score=abs(zscore),
                reasoning=[f"Pair zscore: {zscore:.2f} - Buy {pair_id.split('_')[0]}, Sell {pair_id.split('_')[1]}"],
                metadata={"zscore": zscore, "beta": beta, "action": "buy_a_sell_b"}
            )
        
        return None

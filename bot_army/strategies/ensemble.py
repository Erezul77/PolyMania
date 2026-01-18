"""Ensemble strategy combining multiple strategies."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from .momentum import MomentumStrategy
from .stat_arb import StatArbStrategy
from .mean_reversion import MeanReversionStrategy
from ..core.events import Signal, SignalType

logger = logging.getLogger("strategy.ensemble")


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that combines multiple sub-strategies.
    
    Features:
    - Weighted voting from multiple strategies
    - Adaptive weight adjustment based on performance
    - Regime-aware strategy selection
    - Confidence aggregation
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="ensemble",
                min_confidence=0.3,
                position_size_pct=0.1,
                stop_loss_pct=0.1,
                take_profit_pct=0.15
            )
        
        super().__init__(config)
        
        # Initialize sub-strategies
        self.strategies: Dict[str, BaseStrategy] = {
            "momentum": MomentumStrategy(),
            "stat_arb": StatArbStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
        
        # Strategy weights (adaptive)
        self.weights: Dict[str, float] = {
            "momentum": 0.35,
            "stat_arb": 0.35,
            "mean_reversion": 0.30
        }
        
        # Performance tracking for weight adaptation
        self._strategy_performance: Dict[str, Dict] = {
            name: {"wins": 0, "losses": 0, "total_pnl": 0}
            for name in self.strategies
        }
        
        # Regime-specific weights
        self.regime_weights = {
            "TRENDING_UP": {"momentum": 0.5, "stat_arb": 0.3, "mean_reversion": 0.2},
            "TRENDING_DOWN": {"momentum": 0.5, "stat_arb": 0.3, "mean_reversion": 0.2},
            "MEAN_REVERTING": {"momentum": 0.2, "stat_arb": 0.3, "mean_reversion": 0.5},
            "HIGH_VOLATILITY": {"momentum": 0.4, "stat_arb": 0.4, "mean_reversion": 0.2},
            "LOW_VOLATILITY": {"momentum": 0.3, "stat_arb": 0.3, "mean_reversion": 0.4},
            "RANGING": {"momentum": 0.25, "stat_arb": 0.35, "mean_reversion": 0.4}
        }
        
        self._current_regime = "RANGING"
    
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        """Analyze using all strategies and combine results."""
        
        if not self.should_generate_signal(market_id):
            return None
        
        # Collect signals from all strategies
        strategy_results: List[Tuple[str, StrategyResult]] = []
        
        for name, strategy in self.strategies.items():
            try:
                result = strategy.analyze(
                    market_id=market_id,
                    features=features,
                    price_history=price_history,
                    orderbook=orderbook,
                    trades=trades
                )
                if result:
                    strategy_results.append((name, result))
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
        
        if not strategy_results:
            return None
        
        # Combine signals using weighted voting
        combined = self._combine_signals(strategy_results)
        
        if combined["signal_type"] == SignalType.HOLD:
            return None
        
        if combined["confidence"] < self.config.min_confidence:
            return None
        
        # Create ensemble signal
        signal = self.create_signal(
            market_id=market_id,
            signal_type=combined["signal_type"],
            confidence=combined["confidence"],
            price=price_history[-1],
            features=features
        )
        
        # Use best strategy's stops if available
        best_strategy = combined.get("best_strategy")
        if best_strategy:
            for name, result in strategy_results:
                if name == best_strategy:
                    signal.stop_loss = result.signal.stop_loss
                    signal.take_profit = result.signal.take_profit
                    break
            signal.strategy_name = best_strategy
            signal.metadata["best_strategy"] = best_strategy
        else:
            signal.metadata["best_strategy"] = None
        signal.metadata["ensemble"] = True
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=combined["score"],
            reasoning=combined["reasoning"],
            metadata={
                "strategy_votes": combined["votes"],
                "strategy_weights": self.weights.copy(),
                "regime": self._current_regime
            }
        )
    
    def _combine_signals(
        self,
        results: List[Tuple[str, StrategyResult]]
    ) -> Dict[str, Any]:
        """Combine signals from multiple strategies."""
        
        # Get current weights (regime-adjusted)
        weights = self.regime_weights.get(self._current_regime, self.weights)
        
        # Aggregate votes
        buy_score = 0.0
        sell_score = 0.0
        votes = {}
        reasoning = []
        
        for name, result in results:
            weight = weights.get(name, 0.25)
            signal = result.signal
            
            votes[name] = {
                "signal": signal.signal_type.value,
                "confidence": signal.confidence,
                "weight": weight
            }
            
            if signal.signal_type == SignalType.BUY:
                buy_score += signal.confidence * weight
                reasoning.append(f"{name}: BUY ({signal.confidence:.0%})")
            elif signal.signal_type == SignalType.SELL:
                sell_score += signal.confidence * weight
                reasoning.append(f"{name}: SELL ({signal.confidence:.0%})")
        
        # Normalize scores
        total_weight = sum(weights.get(name, 0.25) for name, _ in results)
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        agreement_bonus = self._calculate_agreement_bonus(results)
        
        if buy_score > sell_score:
            final_confidence = buy_score * (1 + agreement_bonus)
            signal_type = SignalType.BUY
            score = buy_score
        elif sell_score > buy_score:
            final_confidence = sell_score * (1 + agreement_bonus)
            signal_type = SignalType.SELL
            score = sell_score
        else:
            return {"signal_type": SignalType.HOLD, "confidence": 0, "score": 0, "votes": votes, "reasoning": []}
        
        # Find best performing strategy for this signal type
        best_strategy = None
        best_confidence = 0
        for name, result in results:
            if result.signal.signal_type == signal_type:
                if result.signal.confidence > best_confidence:
                    best_confidence = result.signal.confidence
                    best_strategy = name
        
        return {
            "signal_type": signal_type,
            "confidence": min(0.95, final_confidence),
            "score": score,
            "votes": votes,
            "reasoning": reasoning,
            "best_strategy": best_strategy,
            "agreement_bonus": agreement_bonus
        }
    
    def _calculate_agreement_bonus(
        self,
        results: List[Tuple[str, StrategyResult]]
    ) -> float:
        """Calculate bonus for strategy agreement."""
        if len(results) < 2:
            return 0
        
        signals = [r.signal.signal_type for _, r in results]
        buy_count = signals.count(SignalType.BUY)
        sell_count = signals.count(SignalType.SELL)
        
        # All agree
        if buy_count == len(results) or sell_count == len(results):
            return 0.2
        
        # Majority agrees (at least 2/3)
        majority = len(results) * 2 // 3
        if buy_count >= majority or sell_count >= majority:
            return 0.1
        
        return 0
    
    def set_regime(self, regime: str):
        """Set current market regime."""
        self._current_regime = regime
        
        # Also update sub-strategies if they support it
        for strategy in self.strategies.values():
            if hasattr(strategy, 'set_regime'):
                strategy.set_regime(regime, 0.5)
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        pnl: float,
        profitable: bool
    ):
        """Update performance metrics for a strategy."""
        if strategy_name not in self._strategy_performance:
            return
        
        perf = self._strategy_performance[strategy_name]
        perf["total_pnl"] += pnl
        
        if profitable:
            perf["wins"] += 1
        else:
            perf["losses"] += 1
        
        # Adapt weights based on performance
        self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt strategy weights based on recent performance."""
        total_trades = sum(
            p["wins"] + p["losses"]
            for p in self._strategy_performance.values()
        )
        
        if total_trades < 20:
            return  # Not enough data
        
        # Calculate win rates
        win_rates = {}
        for name, perf in self._strategy_performance.items():
            trades = perf["wins"] + perf["losses"]
            if trades > 0:
                win_rates[name] = perf["wins"] / trades
            else:
                win_rates[name] = 0.5
        
        # Adjust weights based on win rates
        total_wr = sum(win_rates.values())
        if total_wr > 0:
            for name in self.weights:
                if name in win_rates:
                    # Blend current weight with performance-based weight
                    perf_weight = win_rates[name] / total_wr
                    self.weights[name] = 0.7 * self.weights[name] + 0.3 * perf_weight
        
        # Normalize weights
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total

    def get_current_weights(self) -> Dict[str, float]:
        """Get normalized weights for the current regime."""
        weights = self.regime_weights.get(self._current_regime, self.weights)
        total = sum(weights.values()) or 1.0
        return {name: weight / total for name, weight in weights.items()}
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        base_stats = super().stats
        
        base_stats["sub_strategies"] = {
            name: {
                "weight": self.weights.get(name, 0),
                **self._strategy_performance.get(name, {}),
                **strategy.stats
            }
            for name, strategy in self.strategies.items()
        }
        
        base_stats["regime"] = self._current_regime
        
        return base_stats

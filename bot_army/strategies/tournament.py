"""
Strategy Tournament Manager
===========================
World-class strategy competition system inspired by Renaissance/Two Sigma.

Features:
- Runs all strategies independently
- Tracks per-strategy performance with decay
- Thompson Sampling for exploration/exploitation
- Correlation penalty to avoid similar strategies
- Dynamic capital allocation
- Meta-learner for strategy selection
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import Signal, SignalType

logger = logging.getLogger("strategy.tournament")

# Import meta-learner (optional - graceful fallback if not available)
try:
    from .meta_learner import MetaLearner, StrategyContext
    META_LEARNER_AVAILABLE = True
except ImportError:
    META_LEARNER_AVAILABLE = False
    logger.warning("MetaLearner not available")


@dataclass
class StrategyStats:
    """Performance statistics for a single strategy."""
    name: str
    signals: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    recent_pnl: List[float] = field(default_factory=list)
    recent_signals: List[Dict] = field(default_factory=list)
    sharpe: float = 0.0
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    score: float = 50.0  # Thompson Sampling score
    weight: float = 0.1  # Current allocation weight
    last_signal_time: datetime = None
    correlation_penalty: float = 0.0
    
    def update_sharpe(self):
        """Calculate Sharpe ratio from recent PnL."""
        if len(self.recent_pnl) < 5:
            self.sharpe = 0.0
            return
        
        returns = np.array(self.recent_pnl[-50:])
        if np.std(returns) > 0:
            self.sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            self.sharpe = 0.0
    
    def update_score(self):
        """Update Thompson Sampling score."""
        # Beta distribution parameters (wins+1, losses+1)
        alpha = self.wins + 1
        beta = self.losses + 1
        
        # Sample from beta distribution (exploration)
        base_score = np.random.beta(alpha, beta) * 100
        
        # Add Sharpe bonus
        sharpe_bonus = max(0, self.sharpe * 10)
        
        # Apply correlation penalty
        penalty = self.correlation_penalty * 20
        
        self.score = max(1, base_score + sharpe_bonus - penalty)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "signals": self.signals,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "score": self.score,
            "weight": self.weight
        }


class StrategyTournament:
    """
    Strategy Tournament Manager.
    
    Runs all strategies in competition, tracks performance,
    and dynamically allocates based on Thompson Sampling.
    """
    
    def __init__(
        self,
        strategies: Dict[str, BaseStrategy],
        initial_weights: Dict[str, float] = None,
        min_weight: float = 0.02,
        max_weight: float = 0.4,
        decay_factor: float = 0.95,
        correlation_threshold: float = 0.7,
        use_meta_learner: bool = True
    ):
        self.strategies = strategies
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor
        self.correlation_threshold = correlation_threshold
        
        # Initialize stats for each strategy
        self.stats: Dict[str, StrategyStats] = {}
        for name in strategies:
            self.stats[name] = StrategyStats(name=name)
        
        # Set initial weights
        if initial_weights:
            for name, weight in initial_weights.items():
                if name in self.stats:
                    self.stats[name].weight = weight
        else:
            # Equal weights initially
            equal_weight = 1.0 / len(strategies)
            for stats in self.stats.values():
                stats.weight = equal_weight
        
        # Signal history for correlation
        self._signal_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        # Outcome buffer for labeling
        self._pending_outcomes: Dict[str, Dict] = {}
        
        # Initialize meta-learner
        self.meta_learner = None
        if use_meta_learner and META_LEARNER_AVAILABLE:
            try:
                self.meta_learner = MetaLearner(
                    strategy_names=list(strategies.keys()),
                    min_samples=30,
                    retrain_interval=50
                )
                logger.info("Meta-learner initialized")
            except Exception as e:
                logger.warning(f"Could not initialize meta-learner: {e}")
        
        # Feature cache for meta-learner training
        self._last_features: Dict[str, np.ndarray] = {}
        
        logger.info(f"Tournament initialized with {len(strategies)} strategies")
    
    def analyze_all(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> List[StrategyResult]:
        """
        Run all strategies and return their signals.
        Each signal is independent and tracked separately.
        """
        results = []
        
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
                    # Ensure strategy name is set correctly
                    result.signal.strategy_name = name
                    result.strategy_name = name
                    
                    # Track signal
                    self.stats[name].signals += 1
                    self.stats[name].last_signal_time = datetime.utcnow()
                    self.stats[name].recent_signals.append({
                        "market_id": market_id,
                        "type": result.signal.signal_type.value,
                        "confidence": result.signal.confidence,
                        "time": datetime.utcnow().isoformat()
                    })
                    
                    # Keep only recent signals
                    self.stats[name].recent_signals = self.stats[name].recent_signals[-100:]
                    
                    # Record for correlation
                    signal_value = 1 if result.signal.signal_type == SignalType.BUY else -1
                    self._signal_history[market_id].append((name, signal_value))
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
        
        return results
    
    def select_best_signals(
        self,
        results: List[StrategyResult],
        features: Dict[str, float] = None,
        regime: str = "unknown",
        max_signals: int = 3
    ) -> List[StrategyResult]:
        """
        Select the best signals based on strategy scores and correlation.
        Uses Thompson Sampling + Meta-Learner for exploration/exploitation.
        """
        if not results:
            return []
        
        # Update scores
        for stats in self.stats.values():
            stats.update_score()
        
        # Use meta-learner if available
        meta_scores = {}
        if self.meta_learner and features and META_LEARNER_AVAILABLE:
            try:
                # Build context for meta-learner
                contexts = []
                for result in results:
                    ctx = StrategyContext(
                        strategy_name=result.strategy_name,
                        signal_type=result.signal.signal_type.value,
                        confidence=result.signal.confidence,
                        market_id=result.signal.market_id,
                        features=features,
                        regime=regime
                    )
                    contexts.append(ctx)
                
                # Get meta-learner predictions
                best_strategy, meta_scores = self.meta_learner.predict_best_strategy(
                    contexts, features, regime
                )
                
                # Store features for learning
                if features:
                    meta_features = self.meta_learner._build_meta_features(contexts, features, regime)
                    for result in results:
                        self._last_features[result.strategy_name] = meta_features.to_vector(
                            self.meta_learner.strategy_names
                        )
                
            except Exception as e:
                logger.debug(f"Meta-learner error: {e}")
        
        # Score each result (combine Thompson + Meta-learner)
        scored_results = []
        for result in results:
            name = result.strategy_name
            stats = self.stats.get(name)
            if stats:
                # Base score: Thompson sampling * confidence * weight
                base_score = stats.score * result.signal.confidence * (1 + stats.weight)
                
                # Meta-learner boost (if available)
                meta_boost = meta_scores.get(name, 0.5) * 50  # 0-50 bonus
                
                combined_score = base_score + meta_boost
                scored_results.append((combined_score, result))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Select top signals, avoiding correlated strategies
        selected = []
        selected_names = set()
        
        for score, result in scored_results:
            if len(selected) >= max_signals:
                break
            
            name = result.strategy_name
            
            # Check correlation with already selected
            is_correlated = False
            for selected_name in selected_names:
                corr = self._get_strategy_correlation(name, selected_name)
                if corr > self.correlation_threshold:
                    is_correlated = True
                    break
            
            if not is_correlated:
                selected.append(result)
                selected_names.add(name)
        
        return selected
    
    def record_outcome(
        self,
        strategy_name: str,
        pnl: float,
        trade_id: str = None
    ):
        """Record trade outcome for a strategy."""
        if strategy_name not in self.stats:
            return
        
        stats = self.stats[strategy_name]
        stats.trades += 1
        stats.total_pnl += pnl
        stats.recent_pnl.append(pnl)
        stats.recent_pnl = stats.recent_pnl[-100:]  # Keep last 100
        
        profitable = pnl > 0
        if profitable:
            stats.wins += 1
        else:
            stats.losses += 1
        
        # Update metrics
        if stats.trades > 0:
            stats.win_rate = stats.wins / stats.trades
            stats.avg_pnl = stats.total_pnl / stats.trades
        
        stats.update_sharpe()
        stats.update_score()
        
        # Feed to meta-learner for learning
        if self.meta_learner and strategy_name in self._last_features:
            try:
                features = self._last_features[strategy_name]
                self.meta_learner.record_outcome(
                    strategy_name=strategy_name,
                    features=features,
                    pnl=pnl,
                    profitable=profitable
                )
            except Exception as e:
                logger.debug(f"Meta-learner record error: {e}")
        
        # Adapt weights
        self._adapt_weights()
        
        logger.info(f"Strategy {strategy_name}: PnL={pnl:.4f}, Total={stats.total_pnl:.4f}, "
                   f"WinRate={stats.win_rate:.1%}, Score={stats.score:.1f}")
    
    def _adapt_weights(self):
        """Adapt strategy weights based on performance."""
        # Calculate raw weights from scores
        total_score = sum(s.score for s in self.stats.values())
        if total_score == 0:
            return
        
        for name, stats in self.stats.items():
            # Base weight from score
            base_weight = stats.score / total_score
            
            # Apply Sharpe bonus
            sharpe_mult = 1 + max(0, stats.sharpe) * 0.2
            
            # Apply win rate adjustment
            wr_mult = 0.5 + stats.win_rate
            
            # New weight
            new_weight = base_weight * sharpe_mult * wr_mult
            
            # Smooth transition (decay)
            stats.weight = stats.weight * self.decay_factor + new_weight * (1 - self.decay_factor)
            
            # Clamp
            stats.weight = max(self.min_weight, min(self.max_weight, stats.weight))
        
        # Normalize
        total_weight = sum(s.weight for s in self.stats.values())
        for stats in self.stats.values():
            stats.weight /= total_weight
    
    def _get_strategy_correlation(self, name1: str, name2: str) -> float:
        """Calculate signal correlation between two strategies."""
        # Collect recent signals for both strategies
        signals1 = []
        signals2 = []
        
        for market_id, history in self._signal_history.items():
            s1 = [v for n, v in history if n == name1]
            s2 = [v for n, v in history if n == name2]
            
            # Match by time (simplified - use last signals)
            if s1 and s2:
                signals1.append(s1[-1])
                signals2.append(s2[-1])
        
        if len(signals1) < 5:
            return 0.0
        
        # Calculate correlation
        try:
            corr = np.corrcoef(signals1, signals2)[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def update_correlations(self):
        """Update correlation penalties for all strategies."""
        names = list(self.stats.keys())
        correlations = {}
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                corr = self._get_strategy_correlation(name1, name2)
                correlations[(name1, name2)] = corr
                correlations[(name2, name1)] = corr
        
        # Calculate penalty for each strategy (avg correlation with others)
        for name, stats in self.stats.items():
            corrs = [correlations.get((name, other), 0) for other in names if other != name]
            stats.correlation_penalty = np.mean(corrs) if corrs else 0.0
    
    def get_leaderboard(self) -> List[Dict]:
        """Get strategy leaderboard sorted by score."""
        self.update_correlations()
        
        leaderboard = []
        for name, stats in self.stats.items():
            stats.update_score()
            leaderboard.append(stats.to_dict())
        
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        return leaderboard
    
    def get_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return {name: stats.weight for name, stats in self.stats.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tournament statistics."""
        return {
            "total_strategies": len(self.strategies),
            "total_signals": sum(s.signals for s in self.stats.values()),
            "total_trades": sum(s.trades for s in self.stats.values()),
            "total_pnl": sum(s.total_pnl for s in self.stats.values()),
            "leaderboard": self.get_leaderboard()[:5],  # Top 5
            "weights": self.get_weights()
        }

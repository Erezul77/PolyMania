"""
PolyMania Strategy Zoo
======================
12+ diverse trading strategies for the tournament system.
"""

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult

# Core strategies
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .stat_arb import StatArbStrategy

# Advanced strategies
from .breakout import BreakoutStrategy
from .trend_following import TrendFollowingStrategy
from .volatility import VolatilityStrategy
from .counter_trend import CounterTrendStrategy
from .volume_profile import VolumeProfileStrategy
from .sentiment_strategy import SentimentStrategy
from .event_driven import EventDrivenStrategy
from .ml_classifier import MLClassifierStrategy
from .market_regime import MarketRegimeStrategy

# Ensemble (legacy)
from .ensemble import EnsembleStrategy

# Tournament system
from .tournament import StrategyTournament, StrategyStats

# Meta-learning & Walk-forward
from .meta_learner import MetaLearner
from .walk_forward import WalkForwardOptimizer, WalkForwardConfig


def create_strategy_zoo() -> dict:
    """
    Create all strategies for the tournament.
    Returns dict of strategy_name -> strategy_instance.
    """
    return {
        # Core strategies
        "momentum": MomentumStrategy(),
        "mean_reversion": MeanReversionStrategy(),
        "stat_arb": StatArbStrategy(),
        
        # Advanced strategies
        "breakout": BreakoutStrategy(),
        "trend_following": TrendFollowingStrategy(),
        "volatility": VolatilityStrategy(),
        "counter_trend": CounterTrendStrategy(),
        "volume_profile": VolumeProfileStrategy(),
        "sentiment": SentimentStrategy(),
        "event_driven": EventDrivenStrategy(),
        "ml_classifier": MLClassifierStrategy(),
        "market_regime": MarketRegimeStrategy(),
    }


__all__ = [
    # Base
    "BaseStrategy",
    "StrategyConfig", 
    "StrategyResult",
    
    # Strategies
    "MomentumStrategy",
    "MeanReversionStrategy",
    "StatArbStrategy",
    "BreakoutStrategy",
    "TrendFollowingStrategy",
    "VolatilityStrategy",
    "CounterTrendStrategy",
    "VolumeProfileStrategy",
    "SentimentStrategy",
    "EventDrivenStrategy",
    "MLClassifierStrategy",
    "MarketRegimeStrategy",
    "EnsembleStrategy",
    
    # Tournament & ML
    "StrategyTournament",
    "StrategyStats",
    "MetaLearner",
    "WalkForwardOptimizer",
    "WalkForwardConfig",
    "create_strategy_zoo",
]
